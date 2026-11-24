#!/usr/bin/env python3
"""Run MESA 26.4.1's GYRE on captured models: verify, coverage, oracles.

Inputs are the `pulse_tap` captures in the `mesa_histories` data bundle
(`runs/<suite>/<case>/pulse_tap/`): for every GYRE call MESA made in the
reference runs, the exact `set_model` input (`models/gyre_eval_*.gyre`) and
every mode GYRE returned (`modes.tsv`, IEEE-754 bits).

Subcommands:
  verify    run a driver on captured models and compare every mode with
            `modes.tsv` bit for bit (l, m, n_pg, n_p, n_g, frequency in uHz
            and Hz, inertia, growth rate in rad/s and Hz).
  coverage  run the coverage driver, then summarise gcov JSON output for
            GYRE (`.fypp` lines and procedures) and reference LAPACK/BLAS.
  oracles   run the tapped driver on the representative calls in
            REPRESENTATIVE and write header-stamped hex oracle files.

See README.md for the build steps that produce the drivers.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

HOME = pathlib.Path.home()

# GYRE-calling cases of MESA 26.4.1. For the star cases the degree is fixed
# by x_integer_ctrl(3) in the case inlist; for the astero cases it comes from
# the calls recorded in modes.tsv (astero calls l = 0 first, then 1..3).
CASES: dict[str, list[int] | None] = {
    "star/gyre_in_mesa_bcep": [0],
    "star/gyre_in_mesa_envelope": [0],
    "star/gyre_in_mesa_ms": [0],
    "star/gyre_in_mesa_rsg": [0],
    "star/gyre_in_mesa_spb": [1],
    "star/gyre_in_mesa_wd": [1],
    "star/rsp_gyre": [0],
    "astero/astero_gyre": [0],
    "astero/fast_from_file": None,
}

# One call per equation set / difference scheme / boundary combination.
# (case, eval_id, eigenfunction stride)
REPRESENTATIVE: list[tuple[str, int, int]] = [
    ("astero/astero_gyre", 1, 64),  # rad, COLLOC_GL4, JCD vars + outer bound
    ("astero/fast_from_file", 216, 64),  # rad + ad (l=0,1,2), COLLOC_GL4, JCD, no refinement
    ("star/gyre_in_mesa_ms", 1, 64),  # rad + nad, COLLOC_GL2, GYRE vars, VACUUM
    ("star/gyre_in_mesa_bcep", 1, 64),  # rad + nad, MAGNUS_GL2 (2x2 eigen, ZGEEV)
    ("star/gyre_in_mesa_rsg", 1, 64),  # rad + nad, MAGNUS_GL2, red supergiant
    ("star/gyre_in_mesa_spb", 1, 64),  # ad + nad l=1, MAGNUS_GL2 (DGEEV, ZGEEV), INVERSE scan
    ("star/gyre_in_mesa_wd", 1, 64),  # ad l=1 adiabatic only, COLLOC_GL2, INVERSE scan
    ("star/gyre_in_mesa_envelope", 1, 64),  # nad, ZERO_R inner bound, add_center = F
    ("star/rsp_gyre", 1, 1),  # nad, ZERO_R; full eigenfunctions (RSP velocities)
]

EVAL_RE = re.compile(r"gyre_eval_(\d+)_step_(\d+)\.gyre$")

PROVENANCE = {
    "mesa_version": "r26.4.1",
    "mesa_git_commit": "c8c202662922a22b3efefa824473ad615c623afd",
    "mesa_zenodo_record": "19722306",
    "gyre_version": "8.1 (bundled with MESA 26.4.1, gyre/gyre-8.1.tar.gz)",
    "mesasdk_version": "x86_64-linux-26.6.1",
    "compiler": "GCC version 15.2.0 (gfortran)",
    "math_backend": "CRMATH (USE_CRMATH = YES); LAPACK/BLAS netlib 3.12.0, SDK crmath math slot",
    "gyre_fflags": (
        "-std=f2018 -finit-real=snan -fmax-errors=25 -fPIC "
        "-ffpe-trap=invalid,overflow,zero -O2 -fno-backtrace -fopenmp -ffp-contract=off"
    ),
    "gyre_config": "FRONTENDS=no TOOLS=no IFACES=yes FORUM=no CRMATH=yes SHARED=no",
}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Case:
    """Captured GYRE calls of one MESA test case."""

    def __init__(self, key: str, histories: pathlib.Path, mesa_dir: pathlib.Path):
        self.key = key
        suite, name = key.split("/")
        self.run_dir = histories / "runs" / suite / name
        self.tap_dir = self.run_dir / "pulse_tap"
        self.gyre_in = mesa_dir / suite / "test_suite" / name / "gyre.in"
        self.fixed_l = CASES[key]
        self.models: dict[int, pathlib.Path] = {}
        for p in sorted((self.tap_dir / "models").glob("gyre_eval_*.gyre")):
            m = EVAL_RE.search(p.name)
            if m:
                self.models[int(m.group(1))] = p
        self.rows: dict[int, list[dict[str, str]]] = collections.defaultdict(list)
        with (self.tap_dir / "modes.tsv").open() as fh:
            header = fh.readline().rstrip("\n").split("\t")
            for line in fh:
                row = dict(zip(header, line.rstrip("\n").split("\t"), strict=True))
                if row["code"] == "gyre":
                    self.rows[int(row["eval_id"])].append(row)

    def degrees(self, eval_id: int) -> list[int]:
        if self.fixed_l is not None:
            return self.fixed_l
        ls = sorted({int(r["l"]) for r in self.rows.get(eval_id, [])})
        return ls or [0]

    def evals(self, max_evals: int | None) -> list[int]:
        ids = sorted(self.models)
        if max_evals is None or len(ids) <= max_evals:
            return ids
        # Evenly spaced sample including the first and last call
        n = len(ids)
        picks = {ids[(i * (n - 1)) // (max_evals - 1)] for i in range(max_evals)}
        return sorted(picks)


def parse_driver_modes(path: pathlib.Path) -> list[dict[str, object]]:
    modes: dict[int, dict[str, object]] = {}
    order: list[int] = []
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            tag = f[0]
            if tag == "MODE":
                idx = int(f[1])
                modes[idx] = {
                    "l": int(f[2]),
                    "m": int(f[3]),
                    "n_p": int(f[4]),
                    "n_g": int(f[5]),
                    "n_pg": int(f[6]),
                    "n": int(f[7]),
                }
                order.append(idx)
            elif tag in ("OMEGA", "FREQ_UHZ", "FREQ_HZ", "FREQ_RAD_PER_SEC", "E_NORM"):
                modes[int(f[1])][tag] = f[2].lower().split()
    return [modes[i] for i in order]


def compare(case: Case, eval_id: int, got: list[dict[str, object]]) -> list[str]:
    """Bitwise comparison against modes.tsv; returns mismatch descriptions."""
    ref = case.rows.get(eval_id, [])
    errs: list[str] = []
    if len(ref) != len(got):
        errs.append(f"mode count {len(got)} != reference {len(ref)}")
    for k, (r, g) in enumerate(zip(ref, got, strict=False)):  # count checked above
        checks = [
            ("l", int(r["l"]), g["l"]),
            ("m", int(r["m"]), g["m"]),
            ("n_pg", int(r["n_pg"]), g["n_pg"]),
            ("n_p", int(r["n_p"]), g["n_p"]),
            ("n_g", int(r["n_g"]), g["n_g"]),
            ("freq_uHz", r["freq_uHz_bits"][2:], g["FREQ_UHZ"][0]),
            ("freq_Hz", r["freq_Hz_bits"][2:], g["FREQ_HZ"][0]),
            ("inertia", r["inertia_bits"][2:], g["E_NORM"][0]),
            ("growth_rad_s", r["growth_rate_rad_s_bits"][2:], g["FREQ_RAD_PER_SEC"][1]),
            ("growth_Hz", r["growth_rate_Hz_bits"][2:], g["FREQ_HZ"][1]),
        ]
        for name, want, have in checks:
            if want != have:
                errs.append(f"mode {k + 1} {name}: {have} != reference {want}")
    return errs


def run_driver(
    driver: pathlib.Path,
    case: Case,
    eval_id: int,
    out: pathlib.Path,
    eig_stride: int,
    env_extra: dict[str, str] | None = None,
) -> None:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = env.get("GYRE_ORACLE_THREADS", "1")
    if env_extra:
        env.update(env_extra)
    ls = ",".join(str(x) for x in case.degrees(eval_id))
    subprocess.run(
        [str(driver), str(case.gyre_in), str(case.models[eval_id]), ls, str(out), str(eig_stride)],
        check=True,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def cmd_verify(args: argparse.Namespace) -> int:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    cases = [Case(k, args.histories, args.mesa_dir) for k in args.cases]
    for case in cases:
        for e in case.evals(args.max_evals):
            jobs.append((case, e))
    results: dict[str, list[tuple[int, int, list[str]]]] = collections.defaultdict(list)

    def work(job: tuple[Case, int]) -> tuple[Case, int, list[str], int]:
        case, e = job
        f = out / f"{case.key.replace('/', '__')}__e{e:06d}.tsv"
        run_driver(args.driver, case, e, f, 0)
        got = parse_driver_modes(f)
        errs = compare(case, e, got)
        if not args.keep:
            f.unlink()
        return case, e, errs, len(got)

    with concurrent.futures.ThreadPoolExecutor(args.jobs) as ex:
        for case, e, errs, n in ex.map(work, jobs):
            results[case.key].append((e, n, errs))
    total_modes = total_bad = 0
    lines = ["case\tcalls\tmodes\tcalls_with_mismatch\tfirst_mismatch"]
    for case in cases:
        rs = sorted(results[case.key])
        n_modes = sum(n for _, n, _ in rs)
        bad = [(e, errs) for e, _, errs in rs if errs]
        total_modes += n_modes
        total_bad += len(bad)
        first = f"eval {bad[0][0]}: {bad[0][1][0]}" if bad else "-"
        lines.append(f"{case.key}\t{len(rs)}\t{n_modes}\t{len(bad)}\t{first}")
    report = "\n".join(lines) + "\n"
    head = [f"# FlexMESA GYRE oracle check: {args.label} (#170, package G0)"]
    head += [f"# {k}: {v}" for k, v in PROVENANCE.items()]
    head += [
        f"# driver_build: {args.label}",
        "# reference: pulse_tap/modes.tsv of the mesa_histories bundle (MESA's own run)",
        "# compared bitwise: l, m, n_pg, n_p, n_g, freq_uHz, freq_Hz, inertia, "
        "growth_rate_rad_s, growth_rate_Hz",
        f"# omp_num_threads: {os.environ.get('GYRE_ORACLE_THREADS', '1')}",
    ]
    (out / "verify_summary.tsv").write_text("\n".join(head) + "\n" + report)
    sys.stdout.write(report)
    print(f"total modes compared: {total_modes}; calls with mismatch: {total_bad}")
    return 1 if total_bad else 0


def _gcov_json(gcda_dir: pathlib.Path, objdir: pathlib.Path) -> list[dict]:
    docs = []
    gcdas = sorted(gcda_dir.glob("*.gcda"))
    for chunk in range(0, len(gcdas), 50):
        part = gcdas[chunk : chunk + 50]
        res = subprocess.run(
            [
                "gcov",
                "--json-format",
                "--stdout",
                "--branch-probabilities",
                "-o",
                str(objdir),
                *[str(p) for p in part],
            ],
            check=True,
            capture_output=True,
            cwd=objdir,
        )
        for line in res.stdout.decode().splitlines():
            line = line.strip()
            if line.startswith("{"):
                docs.append(json.loads(line))
    return docs


def summarise_coverage(docs: list[dict], key_re: re.Pattern[str]) -> dict:
    files: dict[str, dict] = {}
    for doc in docs:
        for f in doc.get("files", []):
            m = key_re.search(f["file"])
            if not m:
                continue
            key = m.group(1)
            ent = files.setdefault(key, {"lines": {}, "functions": {}})
            for ln in f.get("lines", []):
                n = ln["line_number"]
                ent["lines"][n] = ent["lines"].get(n, 0) + ln["count"]
            for fn in f.get("functions", []):
                name = fn.get("demangled_name") or fn["name"]
                prev = ent["functions"].get(name)
                cnt = fn["execution_count"]
                if prev is None:
                    ent["functions"][name] = {
                        "count": cnt,
                        "start": fn["start_line"],
                        "end": fn["end_line"],
                        "blocks": fn.get("blocks", 0),
                        "blocks_executed": fn.get("blocks_executed", 0),
                    }
                else:
                    prev["count"] += cnt
                    prev["blocks_executed"] = max(
                        prev["blocks_executed"], fn.get("blocks_executed", 0)
                    )
    return files


def _coverage_pass(
    args: argparse.Namespace, cases: list[Case], out: pathlib.Path, reset: bool
) -> dict[str, list[int]]:
    """Run the coverage driver on the selected calls and summarise gcov."""
    out.mkdir(parents=True, exist_ok=True)
    runs = out / "runs"
    runs.mkdir(exist_ok=True)
    if reset:
        for d in (args.gyre_build, args.lapack_obj):
            for p in d.glob("*.gcda"):
                p.unlink()
    per_case: dict[str, list[int]] = {}
    jobs = []
    for case in cases:
        sel = case.evals(args.max_evals)
        per_case[case.key] = sel
        for e in sel:
            jobs.append((case, e))

    def work(job: tuple[Case, int]) -> None:
        case, e = job
        f = runs / f"{case.key.replace('/', '__')}__e{e:06d}.tsv"
        run_driver(args.driver, case, e, f, 16)  # exercise eigenfunction outputs too
        f.unlink()

    if not args.summarise_only:
        with concurrent.futures.ThreadPoolExecutor(args.jobs) as ex:
            list(ex.map(work, jobs))

    gyre_docs = _gcov_json(args.gyre_build, args.gyre_build)
    gyre = summarise_coverage(gyre_docs, re.compile(r"/gyre/src/(.+\.(?:fypp|inc))$"))
    lap_docs = _gcov_json(args.lapack_obj, args.lapack_obj)
    lap = summarise_coverage(lap_docs, re.compile(r"/lapack-3\.12\.0/(.+\.[fF](?:90)?)$"))
    (out / "coverage_selection.json").write_text(json.dumps(per_case, indent=1))
    (out / "gyre_coverage.json").write_text(json.dumps(gyre, indent=0, sort_keys=True))
    (out / "lapack_coverage.json").write_text(json.dumps(lap, indent=0, sort_keys=True))
    print(
        f"wrote {out}/gyre_coverage.json ({len(gyre)} files) and "
        f"lapack_coverage.json ({len(lap)} files)"
    )
    return per_case


def cmd_coverage(args: argparse.Namespace) -> int:
    cases = [Case(k, args.histories, args.mesa_dir) for k in args.cases]
    if args.per_case:
        # One pass per case with counters reset, for per-case reach
        for case in cases:
            _coverage_pass(args, [case], args.out / case.key.replace("/", "__"), True)
    else:
        _coverage_pass(args, cases, args.out, False)
    return 0


def header_lines(
    kind: str, case: Case, eval_id: int, variant: str, eig_stride: int, extra: dict[str, str]
) -> list[str]:
    model = case.models[eval_id]
    lines = [f"# FlexMESA GYRE oracle: {kind} (#170, package G0)"]
    for k, v in PROVENANCE.items():
        lines.append(f"# {k}: {v}")
    lines += [
        f"# gyre_build: {variant}",
        f"# case: {case.key}",
        f"# gyre_call_eval_id: {eval_id}",
        f"# model_file: pulse_tap/models/{model.name} (mesa_histories bundle, runs/{case.key}/)",
        f"# model_sha256: {sha256(model)}",
        f"# gyre_in: {case.key.split('/')[0]}/test_suite/{case.key.split('/')[1]}/gyre.in",
        f"# gyre_in_sha256: {sha256(case.gyre_in)}",
        f"# degrees_called: {','.join(str(x) for x in case.degrees(eval_id))}",
        f"# eigenfunction_stride: {eig_stride}",
        "# omp_num_threads: 1",
    ]
    for k, v in extra.items():
        lines.append(f"# {k}: {v}")
    lines.append("# values: IEEE-754 binary64 bit patterns in hex (complex as re im)")
    return lines


def _stamp(path: pathlib.Path, head: list[str]) -> None:
    body = path.read_text()
    path.write_text("\n".join(head) + "\n" + body)


def cmd_oracles(args: argparse.Namespace) -> int:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    bad = 0
    summary = ["case\teval\tmodes\tmatch_reference\tmodes_bytes\ttaps_bytes"]
    for key, e, stride in REPRESENTATIVE:
        case = Case(key, args.histories, args.mesa_dir)
        stem = f"{key.replace('/', '__')}__e{e:06d}"
        modes = out / f"{stem}.modes.tsv"
        taps = out / f"{stem}.taps.tsv"
        tap_env = {
            "GYRE_TAP_FILE": str(taps),
            "GYRE_TAP_MATRIX_BUILDS": str(args.matrix_builds),
            "GYRE_TAP_MATRIX_BLOCKS": str(args.matrix_blocks),
            "GYRE_TAP_LAPACK_MAX": str(args.lapack_max),
        }
        run_driver(args.driver, case, e, modes, stride, tap_env)
        errs = compare(case, e, parse_driver_modes(modes))
        bad += bool(errs)
        tap_desc = {
            "tap_controls": " ".join(
                f"{k}={v}" for k, v in tap_env.items() if k != "GYRE_TAP_FILE"
            ),
        }
        _stamp(
            modes,
            header_lines(
                "modes, eigenfunction samples, model coefficients",
                case,
                e,
                "taps overlay (MESA flags)",
                stride,
                {},
            ),
        )
        _stamp(
            taps,
            header_lines(
                "intermediate taps (gyre_tap_m)",
                case,
                e,
                "taps overlay (MESA flags)",
                stride,
                tap_desc,
            ),
        )
        summary.append(
            f"{key}\t{e}\t{len(case.rows.get(e, []))}\t"
            f"{'yes' if not errs else 'NO: ' + errs[0]}\t"
            f"{modes.stat().st_size}\t{taps.stat().st_size}"
        )
    text = "\n".join(summary) + "\n"
    (out / "oracle_summary.tsv").write_text(text)
    sys.stdout.write(text)
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--histories", type=pathlib.Path, default=HOME / ".cache/flexmesa/mesa_histories"
    )
    ap.add_argument(
        "--mesa-dir",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get("MESA_DIR", HOME / "mesa-26.04.1")),
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--driver", type=pathlib.Path, required=True)
        p.add_argument("--out", type=pathlib.Path, required=True)
        p.add_argument("--cases", nargs="*", default=list(CASES))
        p.add_argument("--jobs", type=int, default=4)
        p.add_argument("--max-evals", type=int, default=None)

    p = sub.add_parser("verify")
    common(p)
    p.add_argument("--keep", action="store_true")
    p.add_argument(
        "--label",
        default="pristine MESA GYRE build",
        help="description of the driver build, written to the summary header",
    )
    p = sub.add_parser("coverage")
    common(p)
    p.add_argument("--gyre-build", type=pathlib.Path, required=True)
    p.add_argument("--lapack-obj", type=pathlib.Path, required=True)
    p.add_argument("--summarise-only", action="store_true")
    p.add_argument(
        "--per-case", action="store_true", help="reset counters and summarise each case separately"
    )
    p = sub.add_parser("oracles")
    common(p)
    p.add_argument("--matrix-builds", type=int, default=1)
    p.add_argument("--matrix-blocks", type=int, default=4)
    p.add_argument("--lapack-max", type=int, default=4)
    args = ap.parse_args()
    return {"verify": cmd_verify, "coverage": cmd_coverage, "oracles": cmd_oracles}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
