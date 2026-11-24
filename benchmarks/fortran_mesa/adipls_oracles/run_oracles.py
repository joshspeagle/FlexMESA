#!/usr/bin/env python3
"""Replay MESA 26.4.1's ADIPLS calls from captured inputs (#171, package A0).

Inputs come from two places:

* the glue-tap capture of a MESA run against the overlay of
  build_mesa_overlay.sh (`<captures>/<case>/a0tap/`: calls.tsv with the
  controls MESA set for every ADIPLS call, the model arrays passed to
  adipls, the FGONG and AMDL arrays of every stored model, and modes.tsv
  with obs_st and csummm of every returned mode);
* the `pulse_tap/modes.tsv` of the same case in the `mesa_histories` data
  bundle (MESA's own reference run), which holds every mode's degree, order,
  cyclic frequency and inertia as IEEE-754 bits.

Subcommands:
  verify    run a driver on every captured call of each case and compare
            every mode bit for bit with the bundle (el, order, frequency,
            inertia) and with the glue tap (obs_st(1:10), csummm(1:50)).
            --replay chain    MESA's sequence in one process, including the
                              srdist mesh redistribution of astero_adipls;
            --replay direct   the same sequence, reading the arrays MESA
                              passed to adipls (no srdist);
            --replay isolated every call in a fresh process (direct arrays).
  coverage  run the coverage driver on every call (or a per-case sample) and
            summarise gcov output for the ADIPLS sources.
  oracles   run the tapped driver on the calls in REPRESENTATIVE and write
            header-stamped hex oracle files.
  libm      run the libm-wrapped driver with glibc and with crlibm results
            and tabulate the mismatches and their effect on the modes.

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
import struct
import subprocess
import sys
import tempfile

HOME = pathlib.Path.home()

# ADIPLS-calling cases of MESA 26.4.1 and whether the case inlist sets
# do_redistribute_mesh, which makes MESA read redistrb.c.pruned.in at start-up
# (run_adipls, setup_redist). Whether a stored model is actually redistributed
# is decided per call; astero_adipls's run_star_extras passes .false., so no
# capture contains a redistributed model (no amdl_redist file).
CASES: dict[str, bool] = {
    "astero/astero_adipls": True,
    "astero/surface_effects": False,
    "astero/example_astero": False,
    "astero/fast_scan_grid": False,
    "astero/fast_simplex": False,
}

# Representative calls for the intermediate oracles: (case, call_id, eig stride).
REPRESENTATIVE: list[tuple[str, int, int]] = [
    ("astero/astero_adipls", 1, 64),  # l = 0, model with added atmosphere, 1738 points
    ("astero/surface_effects", 2, 64),  # l = 1, 1638 points
    ("astero/surface_effects", 4, 64),  # l = 3, same model
    ("astero/example_astero", 29, 64),  # l = 2, 1805 points
    ("astero/fast_scan_grid", 211, 16),  # l = 1, coarse model (498 points)
]

PROVENANCE = {
    "mesa_version": "r26.4.1",
    "mesa_git_commit": "c8c202662922a22b3efefa824473ad615c623afd",
    "mesa_zenodo_record": "19722306",
    "adipls_version": "ADIPLS 2010 (adipack.c) as bundled with MESA 26.4.1",
    "mesasdk_version": "x86_64-linux-26.6.1",
    "compiler": "GCC version 15.2.0 (gfortran)",
    "adipls_fflags": (
        "gfortran -std=legacy -w -c (no -O flag, i.e. -O0; adipls/make/makefile_adipls:78-86)"
    ),
    "math_backend": (
        "system libm (glibc) for pow/exp/log; libgcc __powidf2 for integer powers; no CRMATH"
    ),
}

HEX_RE = re.compile(r"^[0-9A-Fa-f]{16}$")


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hex_to_float(h: str) -> float:
    return float(struct.unpack(">d", bytes.fromhex(h))[0])


def read_tsv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        return [
            dict(zip(header, ln.rstrip("\n").split("\t"), strict=True)) for ln in fh if ln.strip()
        ]


class Case:
    """Captured ADIPLS calls of one MESA test case."""

    def __init__(self, key: str, args: argparse.Namespace):
        self.key = key
        suite, name = key.split("/")
        self.redist = CASES[key]
        self.tap = args.captures / name / "a0tap"
        self.test_dir = args.mesa_dir / suite / "test_suite" / name
        self.adipls_in = self.test_dir / "adipls.c.pruned.in"
        self.redist_in = self.test_dir / "redistrb.c.pruned.in"
        self.calls = read_tsv(self.tap / "calls.tsv")
        self.tap_modes: dict[int, list[dict[str, str]]] = collections.defaultdict(list)
        for r in read_tsv(self.tap / "modes.tsv"):
            self.tap_modes[int(r["call_id"])].append(r)
        self.bundle_modes = [
            r
            for r in read_tsv(args.histories / "runs" / suite / name / "pulse_tap" / "modes.tsv")
            if r["code"] == "adipls"
        ]

    def call_line(self, row: dict[str, str], replay: str) -> str:
        amdl = str(self.tap / row["amdl_file"])
        stored = row["stored_before_call"]
        redist_in = "-"
        redist_out = self.tap / f"store_{int(row['store_id']):06d}.amdl_redist.bin"
        if replay == "chain" and redist_out.exists():
            # MESA redistributed this stored model (run_adipls do_redistribute_mesh)
            redist_in = str(self.tap / f"store_{int(row['store_id']):06d}.amdl_store.bin")
        names = [
            "para_el", "para_els1", "para_dels", "para_dfsig1", "para_dfsig2", "para_sig1",
            "para_sig2", "para_dfsig", "para_eltrw1", "para_eltrw2", "para_sgtrw1", "para_sgtrw2",
            "ipara_nsel", "ipara_nsig1", "ipara_nsig2", "ipara_itrsig", "ipara_nsig",
            "ipara_istsig", "ipara_inomd1", "ipara_iscan", "irotkr", "nprtkr", "igm1kr", "npgmkr",
        ]  # fmt: skip
        return "\t".join([row["call_id"], stored, amdl, redist_in] + [row[n] for n in names])

    def redist_arg(self, replay: str) -> str:
        return str(self.redist_in) if (replay == "chain" and self.redist) else "-"


def parse_driver(path: pathlib.Path) -> tuple[dict[int, list[dict]], dict[int, list[str]]]:
    """Modes per call and CALL records from a driver output file."""
    modes: dict[int, list[dict]] = collections.defaultdict(list)
    calls: dict[int, list[str]] = {}
    cur: dict | None = None
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[0] == "MODE":
                cur = {
                    "el": int(f[3]), "order": int(f[4]), "em": int(f[5]),
                    "freq": f[6].lower(), "inertia": f[7].lower(),
                }  # fmt: skip
                modes[int(f[1])].append(cur)
            elif f[0] == "OBS" and cur is not None:
                cur["obs"] = [v.lower() for v in f[3:]]
            elif f[0] == "CSUM" and cur is not None:
                cur["csum"] = [v.lower() for v in f[3:]]
            elif f[0] == "CALL":
                calls[int(f[1])] = f[2:]
    return modes, calls


def compare_case(case: Case, modes: dict[int, list[dict]], calls: dict[int, list[str]]) -> dict:
    """Bitwise comparison against the bundle and the glue tap."""
    errs: list[str] = []
    n_modes = 0
    seq: list[dict] = []
    for row in case.calls:
        cid = int(row["call_id"])
        got = modes.get(cid, [])
        if cid not in calls:
            errs.append(f"call {cid}: no CALL record (driver stopped)")
            continue
        ierr, nmis = int(calls[cid][0]), int(calls[cid][1])
        if ierr < 0:
            errs.append(f"call {cid}: ierr_param {ierr}")
        if nmis:
            errs.append(f"call {cid}: {nmis} control values differ from MESA's commons")
        ref = case.tap_modes.get(cid, [])
        if len(ref) != len(got):
            errs.append(f"call {cid}: {len(got)} modes != tap {len(ref)}")
        for k, (r, g) in enumerate(zip(ref, got, strict=False)):
            obs = r["obs_st(1:10)"].lower().split()
            csum = r["csummm(1:50)"].lower().split()
            for name, want, have in (
                ("el", int(r["el"]), g["el"]),
                ("order", int(r["order"]), g["order"]),
                ("em", int(r["em"]), g["em"]),
                ("freq", r["cyclic_freq_uHz"].lower(), g["freq"]),
                ("inertia", r["inertia"].lower(), g["inertia"]),
            ):
                if want != have:
                    errs.append(f"call {cid} mode {k + 1} {name}: {have} != tap {want}")
            bad_obs = [i + 1 for i, (a, b) in enumerate(zip(obs, g["obs"], strict=True)) if a != b]
            bad_cs = [i + 1 for i, (a, b) in enumerate(zip(csum, g["csum"], strict=True)) if a != b]
            if bad_obs:
                errs.append(f"call {cid} mode {k + 1} obs_st{bad_obs} differ")
            if bad_cs:
                errs.append(f"call {cid} mode {k + 1} csummm{bad_cs} differ")
        n_modes += len(got)
        seq.extend(got)
    # Ordered comparison with MESA's reference run (bundle)
    b = case.bundle_modes
    if len(b) != len(seq):
        errs.append(f"{len(seq)} modes in total != bundle {len(b)}")
    for k, (r, g) in enumerate(zip(b, seq, strict=False)):
        for name, want, have in (
            ("el", int(r["l"]), g["el"]),
            ("order", int(r["n_pg"]), g["order"]),
            ("freq", r["freq_uHz_bits"][2:], g["freq"]),
            ("inertia", r["inertia_bits"][2:], g["inertia"]),
        ):
            if want != have:
                errs.append(f"bundle mode {k + 1} {name}: {have} != {want}")
    return {"calls": len(case.calls), "modes": n_modes, "errors": errs}


def run_driver(
    driver: pathlib.Path,
    case: Case,
    rows: list[dict[str, str]],
    replay: str,
    out: pathlib.Path,
    eig_stride: int = 0,
    env_extra: dict[str, str] | None = None,
) -> None:
    with tempfile.TemporaryDirectory(
        prefix="adipls_a0_", dir=os.environ.get("ADIPLS_ORACLE_TMP")
    ) as td:
        tdp = pathlib.Path(td)
        calls = tdp / "calls.list"
        calls.write_text("".join(case.call_line(r, replay) + "\n" for r in rows))
        env = dict(os.environ)
        env.update(env_extra or {})
        res = subprocess.run(
            [
                str(driver),
                str(case.adipls_in),
                case.redist_arg(replay),
                str(calls),
                str(out),
                str(eig_stride),
            ],
            cwd=td,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if res.returncode != 0:
            raise RuntimeError(f"driver failed for {case.key}: {res.stderr.decode()[-2000:]}")


def check_redist(case: Case, out: pathlib.Path) -> list[str]:
    """srdist output of the driver against the arrays MESA passed to adipls."""
    errs = []
    for row in case.calls:
        want = case.tap / f"store_{int(row['store_id']):06d}.amdl_redist.bin"
        if row["stored_before_call"] != "T" or not want.exists():
            continue
        cid = int(row["call_id"])
        got = pathlib.Path(f"{out}.redist_{cid:06d}.bin")
        if got.read_bytes() != want.read_bytes():
            errs.append(f"call {cid}: srdist output differs from MESA's redistributed model")
        got.unlink()
    return errs


def env_opts(args: argparse.Namespace) -> dict[str, str]:
    return dict(e.split("=", 1) for e in getattr(args, "env", []) or [])


def cmd_verify(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    lines = ["case\treplay\tcalls\tmodes\tcalls_with_mismatch\tfirst_mismatch"]
    total_bad = 0
    total_modes = 0
    for key in args.cases:
        case = Case(key, args)
        stem = args.out / key.replace("/", "__")
        if args.replay == "isolated":
            outs = {}

            def work(
                row: dict[str, str], case: Case = case, stem: pathlib.Path = stem
            ) -> pathlib.Path:
                o = pathlib.Path(f"{stem}.call{int(row['call_id']):06d}.tsv")
                run_driver(args.driver, case, [row], "direct", o, env_extra=env_opts(args))
                return o

            with concurrent.futures.ThreadPoolExecutor(args.jobs) as ex:
                for row, o in zip(case.calls, ex.map(work, case.calls), strict=True):
                    outs[int(row["call_id"])] = o
            modes: dict[int, list[dict]] = {}
            calls: dict[int, list[str]] = {}
            for o in outs.values():
                m, c = parse_driver(o)
                modes.update(m)
                calls.update(c)
                o.unlink()
            res = compare_case(case, modes, calls)
        else:
            o = pathlib.Path(f"{stem}.{args.replay}.tsv")
            run_driver(args.driver, case, case.calls, args.replay, o, env_extra=env_opts(args))
            modes, calls = parse_driver(o)
            res = compare_case(case, modes, calls)
            if args.replay == "chain" and case.redist:
                res["errors"] += check_redist(case, o)
            if not args.keep:
                o.unlink()
        bad_calls = {e.split(":")[0] for e in res["errors"]}
        total_bad += len(res["errors"])
        total_modes += res["modes"]
        first = res["errors"][0] if res["errors"] else "-"
        lines.append(
            f"{key}\t{args.replay}\t{res['calls']}\t{res['modes']}\t"
            f"{len(bad_calls) if res['errors'] else 0}\t{first}"
        )
        print(lines[-1], flush=True)
    head = [f"# FlexMESA ADIPLS oracle check: {args.label} (#171, package A0)"]
    head += [f"# {k}: {v}" for k, v in PROVENANCE.items()]
    head += [
        f"# driver_build: {args.label}",
        f"# replay: {args.replay}",
        f"# extra environment: {' '.join(args.env) or '-'}",
        "# reference 1: pulse_tap/modes.tsv of the mesa_histories bundle (MESA's own run): "
        "el, order, cyclic_freq, inertia",
        "# reference 2: glue tap modes.tsv of the overlay MESA run: "
        "el, order, em, cyclic_freq, inertia, obs_st(1:10), csummm(1:50)",
    ]
    (args.out / f"verify_{args.replay}.tsv").write_text("\n".join(head + lines) + "\n")
    print(f"total modes compared: {total_modes}; mismatches: {total_bad}")
    return 1 if total_bad else 0


def select_calls(case: Case, max_calls: int | None) -> list[dict[str, str]]:
    """All calls, or an evenly spaced sample (first and last included) that
    keeps whole stored models together, so every sampled call sees the arrays
    MESA passed."""
    rows = case.calls
    if max_calls is None or len(rows) <= max_calls:
        return rows
    n = len(rows)
    picks = sorted({(i * (n - 1)) // (max_calls - 1) for i in range(max_calls)})
    return [rows[i] for i in picks]


def _gcov_json(objdir: pathlib.Path, gcov: str) -> list[dict]:
    docs = []
    gcdas = sorted(objdir.glob("*.gcda"))
    for chunk in range(0, len(gcdas), 40):
        res = subprocess.run(
            [gcov, "--json-format", "--stdout", "--branch-probabilities", "-o", str(objdir)]
            + [str(p) for p in gcdas[chunk : chunk + 40]],
            check=True,
            capture_output=True,
            cwd=objdir,
        )
        for line in res.stdout.decode().splitlines():
            line = line.strip()
            if line.startswith("{"):
                docs.append(json.loads(line))
    return docs


def summarise_coverage(docs: list[dict]) -> dict:
    files: dict[str, dict] = {}
    for doc in docs:
        for f in doc.get("files", []):
            key = re.sub(r"^(\.\./)+", "", f["file"])
            key = key.replace("adipack.c/", "")
            ent = files.setdefault(key, {"lines": {}, "functions": {}})
            for ln in f.get("lines", []):
                n = ln["line_number"]
                ent["lines"][n] = ent["lines"].get(n, 0) + ln["count"]
            for fn in f.get("functions", []):
                name = fn.get("demangled_name") or fn["name"]
                prev = ent["functions"].get(name)
                if prev is None:
                    ent["functions"][name] = {
                        "count": fn["execution_count"],
                        "start": fn["start_line"],
                        "end": fn["end_line"],
                        "blocks": fn.get("blocks", 0),
                        "blocks_executed": fn.get("blocks_executed", 0),
                    }
                else:
                    prev["count"] += fn["execution_count"]
                    prev["blocks_executed"] = max(
                        prev["blocks_executed"], fn.get("blocks_executed", 0)
                    )
    return files


def _coverage_pass(
    args: argparse.Namespace, cases: list[Case], out: pathlib.Path, reset: bool
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if reset:
        for p in args.adipls_obj.glob("*.gcda"):
            p.unlink()
    sel: dict[str, list[int]] = {}
    if not args.summarise_only:
        for case in cases:
            rows = select_calls(case, args.max_calls)
            sel[case.key] = [int(r["call_id"]) for r in rows]
            o = out / f"{case.key.replace('/', '__')}.cov.tsv"
            run_driver(
                args.driver,
                case,
                rows,
                "chain" if len(rows) == len(case.calls) else "direct",
                o,
                16,
            )
            o.unlink()
    cov = summarise_coverage(_gcov_json(args.adipls_obj, args.gcov))
    (out / "coverage_selection.json").write_text(json.dumps(sel, indent=1))
    (out / "adipls_coverage.json").write_text(json.dumps(cov, indent=0, sort_keys=True))
    print(f"wrote {out}/adipls_coverage.json ({len(cov)} files)")


def cmd_coverage(args: argparse.Namespace) -> int:
    cases = [Case(k, args) for k in args.cases]
    if args.per_case:
        for case in cases:
            _coverage_pass(args, [case], args.out / case.key.replace("/", "__"), True)
    else:
        _coverage_pass(args, cases, args.out, True)
    return 0


def header_lines(
    kind: str, case: Case, row: dict[str, str], variant: str, eig_stride: int
) -> list[str]:
    amdl = case.tap / row["amdl_file"]
    suite, name = case.key.split("/")
    lines = [f"# FlexMESA ADIPLS oracle: {kind} (#171, package A0)"]
    lines += [f"# {k}: {v}" for k, v in PROVENANCE.items()]
    lines += [
        f"# adipls_build: {variant}",
        f"# case: {case.key}",
        f"# adipls_call_id: {row['call_id']} "
        f"(model_number {row['model_number']}, el {hex_to_float(row['para_el']):g})",
        f"# input_amdl: a0tap/{row['amdl_file']} of the overlay MESA run "
        "(layout in astero_glue_tap.inc)",
        f"# input_amdl_sha256: {sha256(amdl)}",
        f"# adipls_c_pruned_in: {suite}/test_suite/{name}/adipls.c.pruned.in",
        f"# adipls_c_pruned_in_sha256: {sha256(case.adipls_in)}",
        f"# controls (set_adipls_controls, hex): sig1 {row['para_sig1']} "
        f"sig2 {row['para_sig2']} dfsig {row['para_dfsig']} iscan {row['ipara_iscan']} "
        f"nsig {row['ipara_nsig']} itrsig {row['ipara_itrsig']}",
        f"# eigenfunction_stride: {eig_stride}",
        "# replay: isolated (fresh process: setups_adi, parameter read, then this call; "
        "arrays as passed by MESA)",
        "# values: IEEE-754 binary64 bit patterns in hex",
    ]
    return lines


def _stamp(path: pathlib.Path, head: list[str]) -> None:
    path.write_text("\n".join(head) + "\n" + path.read_text())


def cmd_oracles(args: argparse.Namespace) -> int:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    bad = 0
    summary = ["case\tcall_id\tmodes\tmatch_reference\tmodes_bytes\ttaps_bytes"]
    for key, cid, stride in REPRESENTATIVE:
        if key not in args.cases:
            continue
        case = Case(key, args)
        row = next(r for r in case.calls if int(r["call_id"]) == cid)
        stem = f"{key.replace('/', '__')}__c{cid:06d}"
        modes = out / f"{stem}.modes.tsv"
        taps = out / f"{stem}.taps.tsv"
        run_driver(
            args.driver, case, [row], "direct", modes, stride, {"ADIPLS_TAP_FILE": str(taps)}
        )
        m, _ = parse_driver(modes)
        ref = case.tap_modes.get(cid, [])
        errs = []
        if len(m.get(cid, [])) != len(ref):
            errs.append("mode count")
        for r, g in zip(ref, m.get(cid, []), strict=False):
            if r["cyclic_freq_uHz"].lower() != g["freq"] or r["inertia"].lower() != g["inertia"]:
                errs.append(f"mode order {g['order']}")
            if (
                r["csummm(1:50)"].lower().split() != g["csum"]
                or r["obs_st(1:10)"].lower().split() != g["obs"]
            ):
                errs.append(f"mode order {g['order']} csummm/obs_st")
        bad += bool(errs)
        _stamp(
            modes,
            header_lines(
                "modes, obs_st, csummm, eigenfunction samples",
                case,
                row,
                "taps overlay (MESA flags)",
                stride,
            ),
        )
        _stamp(
            taps,
            header_lines(
                "intermediate taps (apply_adipls_taps.py)",
                case,
                row,
                "taps overlay (MESA flags)",
                stride,
            ),
        )
        summary.append(
            f"{key}\t{cid}\t{len(ref)}\t{'yes' if not errs else 'NO: ' + errs[0]}\t"
            f"{modes.stat().st_size}\t{taps.stat().st_size}"
        )
    text = "\n".join(summary) + "\n"
    (out / "oracle_summary.tsv").write_text(text)
    sys.stdout.write(text)
    return 1 if bad else 0


def ulps(a: str, b: str) -> int:
    def key(h: str) -> int:
        u = int(h, 16)
        return u if u < (1 << 63) else (1 << 63) - u

    return abs(key(a) - key(b))


def cmd_libm(args: argparse.Namespace) -> int:
    """glibc vs crlibm: argument-level mismatch counts and effect on modes."""
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    site_rows: dict[tuple[str, str], list[int]] = collections.defaultdict(lambda: [0, 0, 0])
    samples: dict[str, set[str]] = collections.defaultdict(set)
    effect = [
        "case\tcalls\tmodes_glibc\tmodes_crlibm\tmode_set_changed\tfreq_differ\tfreq_max_ulp"
        "\tinertia_differ\tinertia_max_ulp\tcsummm_modes_differ\tglibc_run_matches_reference"
    ]
    for key in args.cases:
        case = Case(key, args)
        rows = select_calls(case, args.max_calls)
        replay = "chain" if len(rows) == len(case.calls) else "direct"
        res = {}
        for mode in ("glibc", "crlibm"):
            o = out / f"{key.replace('/', '__')}.{mode}.tsv"
            log = out / f"{key.replace('/', '__')}.{mode}.libm.tsv"
            run_driver(
                args.driver,
                case,
                rows,
                replay,
                o,
                0,
                {
                    "ADIPLS_LIBM_RETURN": mode,
                    "ADIPLS_LIBM_LOG": str(log),
                    "ADIPLS_LIBM_SAMPLES": "64",
                    **env_opts(args),
                },
            )
            res[mode] = parse_driver(o)[0]
            if mode == "glibc":
                with log.open() as fh:
                    for line in fh:
                        f = line.rstrip("\n").split("\t")
                        if f[0] == "SITE" and f[1] != "function":
                            sr = site_rows[(f[1], f[2])]
                            sr[0] += int(f[4])
                            sr[1] += int(f[5])
                            sr[2] = max(sr[2], int(f[6]))
                        elif f[0] == "SAMPLE":
                            samples[f[1]].add(f[2])
            o.unlink()
        g_all = [m for r in rows for m in res["glibc"].get(int(r["call_id"]), [])]
        c_all = [m for r in rows for m in res["crlibm"].get(int(r["call_id"]), [])]
        # the glibc run must reproduce the glue-tap reference exactly
        ref = [m for r in rows for m in case.tap_modes.get(int(r["call_id"]), [])]
        ok = len(ref) == len(g_all) and all(
            a["cyclic_freq_uHz"].lower() == b["freq"] and a["inertia"].lower() == b["inertia"]
            for a, b in zip(ref, g_all, strict=False)
        )
        changed = len(g_all) != len(c_all) or any(
            (a["el"], a["order"]) != (b["el"], b["order"])
            for a, b in zip(g_all, c_all, strict=False)
        )
        fd = [ulps(a["freq"], b["freq"]) for a, b in zip(g_all, c_all, strict=False)]
        idf = [ulps(a["inertia"], b["inertia"]) for a, b in zip(g_all, c_all, strict=False)]
        cs = sum(a["csum"] != b["csum"] for a, b in zip(g_all, c_all, strict=False))
        effect.append(
            f"{key}\t{len(rows)}\t{len(g_all)}\t{len(c_all)}\t{'yes' if changed else 'no'}\t"
            f"{sum(d > 0 for d in fd)}\t{max(fd, default=0)}\t{sum(d > 0 for d in idf)}\t"
            f"{max(idf, default=0)}\t{cs}\t{'yes' if ok else 'NO'}"
        )
        print(effect[-1], flush=True)
    glibc = subprocess.run(["ldd", "--version"], capture_output=True, text=True).stdout
    glibc = glibc.splitlines()[0]
    sample = "all" if args.max_calls is None else f"evenly spaced sample of {args.max_calls}"
    head = [
        "# FlexMESA ADIPLS libm measurement: glibc vs crlibm 1.0beta4 round-to-nearest "
        "(#171, A0, DECISIONS #60)"
    ]
    head += [f"# {k}: {v}" for k, v in PROVENANCE.items()]
    head += [
        f"# glibc: {glibc}",
        "# crlibm: vendored crates/flexmesa-crlibm-sys sources and flags; "
        "pow_rn, exp_rn, log_rn, log10_rn, sin_rn, cos_rn",
        f"# calls per case: {sample}",
        f"# extra environment: {' '.join(args.env) or '-'}",
    ]
    sites = ["function\troutine\tcalls\tmismatches\tmismatch_fraction\tmax_ulp"]
    for (fn, routine), (n, m, u) in sorted(site_rows.items()):
        sites.append(f"{fn}\t{routine}\t{n}\t{m}\t{m / n if n else 0:.3e}\t{u}")
    (out / "libm_sites.tsv").write_text("\n".join(head + sites) + "\n")
    (out / "libm_effect.tsv").write_text("\n".join(head + effect) + "\n")
    smp = ["function\targs_and_results_hex (x [y] glibc crlibm)"]
    for fn in sorted(samples):
        smp += [f"{fn}\t{v}" for v in sorted(samples[fn])]
    (out / "libm_mismatch_samples.tsv").write_text("\n".join(head + smp) + "\n")
    sys.stdout.write("\n".join(sites) + "\n")
    return 0


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
    ap.add_argument(
        "--captures",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get("ADIPLS_ORACLE_WORK", ".")) / "mesa_runs",
        help="directory holding <case>/a0tap from run_mesa_captures.sh",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--driver", type=pathlib.Path, required=True)
        p.add_argument("--out", type=pathlib.Path, required=True)
        p.add_argument("--cases", nargs="*", default=list(CASES))
        p.add_argument("--jobs", type=int, default=4)

    p = sub.add_parser("verify")
    common(p)
    p.add_argument("--replay", choices=["chain", "direct", "isolated"], default="chain")
    p.add_argument("--keep", action="store_true")
    p.add_argument("--label", default="pristine MESA libadipls.a")
    p.add_argument(
        "--env", nargs="*", default=[], help="KEY=VALUE pairs for the driver environment"
    )
    p = sub.add_parser("coverage")
    common(p)
    p.add_argument("--adipls-obj", type=pathlib.Path, required=True, help="coverage build make dir")
    p.add_argument("--max-calls", type=int, default=None)
    p.add_argument("--per-case", action="store_true")
    p.add_argument("--summarise-only", action="store_true")
    p.add_argument("--gcov", default="gcov", help="gcov of the SDK compiler (GCC 15.2.0)")
    p = sub.add_parser("oracles")
    common(p)
    p = sub.add_parser("libm")
    common(p)
    p.add_argument("--max-calls", type=int, default=None)
    p.add_argument(
        "--env", nargs="*", default=[], help="KEY=VALUE pairs for the driver environment"
    )
    args = ap.parse_args()
    return {
        "verify": cmd_verify,
        "coverage": cmd_coverage,
        "oracles": cmd_oracles,
        "libm": cmd_libm,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
