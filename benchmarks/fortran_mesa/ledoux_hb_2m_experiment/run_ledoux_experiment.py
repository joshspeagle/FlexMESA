# SPDX-License-Identifier: GPL-3.0-or-later
"""Run ``star/test_suite/hb_2M`` with ``use_Ledoux_criterion`` on and off.

The experiment is a 2x2 grid of {MESA 26.4.1, FlexMESA} by
{``use_Ledoux_criterion = .true.`` (upstream inlist), ``.false.``}. Each run
copies the upstream case directory from ``$MESA_DIR`` into a scratch work
root and changes only the ``use_Ledoux_criterion`` line of ``inlist_hb_2M``
in the copy (the ``.true.`` variant is left byte-identical to upstream).
Both engines skip the optional early stages (``MESA_SKIP_OPTIONAL=t``), so
the run starts from upstream's stored ``standard_2.0_ZACHeB.mod``.

Subcommands:

* ``run --engine {mesa,flexmesa} --ledoux {true,false} --work-root DIR``
  executes one cell of the grid and writes ``run_meta.json`` in its cell
  directory. Wrap each call in ``flock`` so heavy runs never overlap.
* ``summarize --work-root DIR --out-dir DIR`` reads the four cells and writes
  ``results.json`` and ``results.csv`` with portable content only.

Environment:

* ``MESA_DIR`` and ``MESASDK_ROOT`` select the MESA build and SDK; the MESA
  environment is initialized by ``scripts/mesa_reference/env_mesa.sh``.
* ``PYTHONPATH`` must import the FlexMESA build under test; the FlexMESA
  commit is read from the git checkout that contains the imported package.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import pathlib
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

CASE = "hb_2M"
STAGE_INLIST = "inlist_hb_2M"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
ENV_SCRIPT = REPO_ROOT / "scripts" / "mesa_reference" / "env_mesa.sh"

# Upstream termination: xa_central_lower_limit(1) = 0.0001 for he4 in
# star/test_suite/hb_2M/inlist_hb_2M at MESA tag 26.4.1.
HE4_STOP = 1.0e-4
HE4_MILESTONES = (0.9, 0.5, 0.1)
LEDOUX_RE = re.compile(
    r"^(\s*use_Ledoux_criterion\s*=\s*)\.(true|false)\.", re.MULTILINE
)
UNSUPPORTED_RE = re.compile(
    r"unsupported control\(s\) \((\d+)\): (.*?) \(pass --allow"
)
SUMMARY_RE = re.compile(
    r"status=(\S+) term=(\S+) steps=(\d+) retries=(\d+) wall_s=([\d.]+)"
)
ENGINES = ("mesa", "flexmesa")
VARIANTS = ("true", "false")


def cell_dir(work_root: pathlib.Path, engine: str, ledoux: str) -> pathlib.Path:
    """Return the scratch directory of one grid cell."""
    return work_root / f"{engine}_ledoux_{ledoux}"


def copy_case(dest: pathlib.Path, ledoux: str) -> pathlib.Path:
    """Copy the upstream case and set ``use_Ledoux_criterion`` in the copy.

    Args:
        dest: Parent directory receiving ``hb_2M/``; must not exist yet.
        ledoux: ``"true"`` keeps the upstream inlist unchanged, ``"false"``
            rewrites only the ``use_Ledoux_criterion`` assignment.

    Returns:
        Path to the copied case directory.

    Raises:
        FileExistsError: If the copy already exists.
        ValueError: If the inlist does not hold exactly one upstream
            ``use_Ledoux_criterion = .true.`` assignment.
    """
    src = pathlib.Path(os.environ["MESA_DIR"]) / "star" / "test_suite" / CASE
    case_dir = dest / CASE
    if case_dir.exists():
        raise FileExistsError(f"refusing to overwrite {case_dir}")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, case_dir, symlinks=True)
    inlist = case_dir / STAGE_INLIST
    text = inlist.read_text(encoding="utf-8")
    matches = LEDOUX_RE.findall(text)
    if len(matches) != 1 or matches[0][1] != "true":
        raise ValueError(f"unexpected use_Ledoux_criterion lines: {matches}")
    if ledoux == "false":
        text = LEDOUX_RE.sub(r"\1.false.", text)
        inlist.write_text(text, encoding="utf-8")
    return case_dir


def run_bash(script: str, cwd: pathlib.Path, timeout: float | None) -> int:
    """Run ``script`` under bash after sourcing the MESA environment."""
    cmd = f'source "{ENV_SCRIPT}" && {script}'
    proc = subprocess.Popen(
        ["bash", "-c", cmd], cwd=cwd, start_new_session=True
    )
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        raise


def run_mesa(dest: pathlib.Path, ledoux: str, timeout: float) -> dict[str, Any]:
    """Build and run one MESA 26.4.1 cell with ``MESA_SKIP_OPTIONAL=t``."""
    case_dir = copy_case(dest, ledoux)
    mk = run_bash(
        'make -C make MESA_DIR="$MESA_DIR" > mk.log 2>&1', case_dir, None
    )
    if mk != 0:
        raise RuntimeError(f"MESA case build failed (exit {mk}), see mk.log")
    meta: dict[str, Any] = {"timed_out": False}
    start = time.monotonic()
    try:
        meta["exit_code"] = run_bash(
            "MESA_SKIP_OPTIONAL=t ./rn > out.txt 2>&1", case_dir, timeout
        )
    except subprocess.TimeoutExpired:
        meta["timed_out"] = True
        meta["exit_code"] = None
    meta["wall_s"] = time.monotonic() - start
    return meta


def flexmesa_commit() -> str:
    """Return the git commit of the checkout providing ``flexmesa``."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import flexmesa, pathlib; "
            "print(pathlib.Path(flexmesa.__file__).resolve().parent)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    pkg = proc.stdout.strip()
    rev = subprocess.run(
        ["git", "-C", pkg, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return rev.stdout.strip()


def trace(
    case_dir: pathlib.Path, out: pathlib.Path, timeout: float, allow: bool
) -> subprocess.CompletedProcess[str]:
    """Run ``flexmesa trace star <case_dir>`` in strict numerics mode."""
    cmd = [
        sys.executable,
        "-m",
        "flexmesa.cli",
        "trace",
        "star",
        str(case_dir),
        "--out",
        str(out),
        "--numerics-mode",
        "strict",
        "--timeout-sec",
        str(timeout),
    ]
    if allow:
        cmd.append("--allow-unsupported-controls")
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    return subprocess.run(
        cmd, capture_output=True, text=True, env=env, check=False
    )


def run_flexmesa(
    dest: pathlib.Path, ledoux: str, timeout: float
) -> dict[str, Any]:
    """Probe strict control validation, then run with controls ignored."""
    case_dir = copy_case(dest, ledoux)
    probe = trace(case_dir, dest / "probe", timeout, allow=False)
    (dest / "probe.log").write_text(probe.stdout + probe.stderr, "utf-8")
    m_unsup = UNSUPPORTED_RE.search(probe.stderr)
    ignored = m_unsup.group(2).split(", ") if m_unsup else []
    if m_unsup and int(m_unsup.group(1)) != len(ignored):
        raise ValueError("ignored-control count disagrees with its list")
    start = time.monotonic()
    full = trace(case_dir, dest / "out", timeout, allow=True)
    wall = time.monotonic() - start
    (dest / "trace.log").write_text(full.stdout + full.stderr, "utf-8")
    m_sum = SUMMARY_RE.search(full.stdout)
    if m_sum is None:
        raise RuntimeError("no trace summary line, see trace.log")
    return {
        "exit_code": full.returncode,
        "tracer_status": m_sum.group(1),
        "termination_code": m_sum.group(2),
        "steps": int(m_sum.group(3)),
        "retries": int(m_sum.group(4)),
        "wall_s": float(m_sum.group(5)),
        "wall_s_outer": wall,
        "timed_out": m_sum.group(1) == "timeout",
        "ignored_controls": ignored,
        "flexmesa_commit": flexmesa_commit(),
        "numerics_mode": "strict",
    }


def cmd_run(ns: argparse.Namespace) -> int:
    """Execute one grid cell and write its ``run_meta.json``."""
    dest = cell_dir(pathlib.Path(ns.work_root), ns.engine, ns.ledoux)
    if ns.engine == "mesa":
        meta = run_mesa(dest, ns.ledoux, ns.timeout_sec)
    else:
        meta = run_flexmesa(dest, ns.ledoux, ns.timeout_sec)
    meta.update(engine=ns.engine, ledoux=ns.ledoux, timeout_sec=ns.timeout_sec)
    (dest / "run_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta))
    return 0


def read_history(
    path: pathlib.Path,
) -> tuple[dict[str, str], dict[str, list[float]]]:
    """Read a MESA-format ``history.data`` into header and column arrays."""
    lines = path.read_text(encoding="utf-8").splitlines()
    header = dict(zip(lines[1].split(), lines[2].split(), strict=True))
    names = lines[5].split()
    cols: dict[str, list[float]] = {n: [] for n in names}
    for ln in lines[6:]:
        vals = ln.split()
        if len(vals) != len(names):
            continue
        for n, v in zip(names, vals, strict=True):
            cols[n].append(float(v.replace("D", "E")))
    return ({k: v.strip('"') for k, v in header.items()}, cols)


def first_below(
    he4: list[float], other: list[float], level: float
) -> float | None:
    """Return ``other`` at the first row where ``he4 < level``.

    This matches the rule upstream's ``run_star_extras.f90`` uses for
    ``mass_conv_core_y050`` (first step with ``center_he4 < 0.5``).
    """
    for y, v in zip(he4, other, strict=True):
        if y < level:
            return v
    return None


def crossing_age(
    he4: list[float], age: list[float], level: float
) -> float | None:
    """Return the age where ``he4`` first crosses below ``level``.

    The age is linearly interpolated in ``center_he4`` between the two
    history rows that bracket ``level``.
    """
    for i in range(1, len(he4)):
        if he4[i] <= level < he4[i - 1]:
            frac = (he4[i - 1] - level) / (he4[i - 1] - he4[i])
            return age[i - 1] + frac * (age[i] - age[i - 1])
    return None


def mesa_out_fields(out_txt: pathlib.Path) -> dict[str, Any]:
    """Parse termination code, steps and retries from MESA ``out.txt``."""
    text = out_txt.read_text(encoding="utf-8", errors="replace")
    terms = re.findall(r"termination code: (\S+)", text)
    rsr = re.findall(
        r"runtime \(minutes\), retries, steps\s+([\d.]+)\s+(\d+)\s+(\d+)",
        text,
    )
    return {
        "termination_code": terms[-1] if terms else "none",
        "steps": int(rsr[-1][2]) if rsr else None,
        "retries": int(rsr[-1][1]) if rsr else None,
        "mesa_runtime_min": float(rsr[-1][0]) if rsr else None,
    }


def read_cpu_model() -> tuple[str, int]:
    """Read CPU model name and logical processor count."""
    model = platform.processor() or "x86_64"
    logical = os.cpu_count() or 1
    cpuinfo = pathlib.Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for ln in cpuinfo.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if ln.startswith("model name"):
                model = ln.split(":", 1)[1].strip()
                break
    return (model, logical)


def read_os_release() -> str:
    """Read OS description from ``/etc/os-release`` and ``uname``."""
    pretty = platform.platform()
    os_rel = pathlib.Path("/etc/os-release")
    if os_rel.exists():
        for ln in os_rel.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if ln.startswith("PRETTY_NAME="):
                pretty = ln.split("=", 1)[1].strip().strip('"')
                break
    u = platform.uname()
    return f"{pretty} ({u.system} {u.release} {u.machine})"


def collect_provenance() -> dict[str, Any]:
    """Query MESA version, SDK, compiler, and host provenance."""
    mesa_dir = pathlib.Path(os.environ["MESA_DIR"])
    version_str = (
        (mesa_dir / "data" / "version_number")
        .read_text(encoding="utf-8")
        .strip()
    )
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{ENV_SCRIPT}" && "$MESASDK_ROOT/bin/mesasdk_version"'
            " && echo '---' && gfortran --version | head -n 1",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    sdk, _, gfortran = proc.stdout.partition("---")
    cpu_model, logical_cpus = read_cpu_model()
    now_iso = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    return {
        "date": now_iso,
        "mesa": {
            "release": "26.4.1",
            "version_number_file": version_str,
            "git_tag": "26.4.1",
            "git_commit": "c8c202662922a22b3efefa824473ad615c623afd",
            "git_commit_short": "c8c2026",
            "zenodo_record": "19722306",
        },
        "sdk_and_compiler": {
            "mesasdk_version": sdk.strip(),
            "compiler": gfortran.strip(),
            "math_backend": "CRMATH",
        },
        "host": {
            "os": read_os_release(),
            "cpu": cpu_model,
            "logical_threads": logical_cpus,
            "python": platform.python_version(),
            "OMP_NUM_THREADS": int(os.environ.get("OMP_NUM_THREADS", "4")),
        },
    }


def summarize_cell(work_root: pathlib.Path, engine: str, ledoux: str) -> Any:
    """Collect the measured quantities of one grid cell."""
    dest = cell_dir(work_root, engine, ledoux)
    meta = json.loads((dest / "run_meta.json").read_text(encoding="utf-8"))
    row: dict[str, Any] = {
        "engine": engine,
        "use_Ledoux_criterion": f".{ledoux}.",
        "inlist_modified": ledoux == "false",
        "timeout_sec": meta["timeout_sec"],
        "timed_out": meta["timed_out"],
        "wall_s": round(meta["wall_s"], 1),
    }
    if engine == "mesa":
        hist = dest / CASE / "LOGS" / "history.data"
        row.update(mesa_out_fields(dest / CASE / "out.txt"))
        row["status"] = (
            "timeout"
            if meta["timed_out"]
            else ("ok" if meta["exit_code"] == 0 else "error")
        )
        row["label"] = "MESA 26.4.1 run"
    else:
        hist = dest / "out" / "LOGS" / "history.data"
        ign = meta["ignored_controls"]
        row.update(
            status=meta["tracer_status"],
            termination_code=meta["termination_code"],
            steps=meta["steps"],
            retries=meta["retries"],
            flexmesa_commit=meta["flexmesa_commit"],
            numerics_mode=meta["numerics_mode"],
            n_controls_ignored=len(ign),
            ignored_controls=ign,
            label=f"diagnostic ({len(ign)} controls ignored)",
        )
    header, cols = read_history(hist)
    he4, age, mcc = cols["center_he4"], cols["star_age"], cols["mass_conv_core"]
    row.update(
        history_rows=len(age),
        final_model_number=int(cols["model_number"][-1]),
        final_star_age_yr=age[-1],
        final_center_he4=he4[-1],
        he_lifetime_yr=crossing_age(he4, age, HE4_STOP),
        mass_conv_core_max_msun=max(mcc),
    )
    for lvl in HE4_MILESTONES:
        row[f"mass_conv_core_y{round(100 * lvl):03d}_msun"] = first_below(
            he4, mcc, lvl
        )
    if engine == "mesa":
        row["history_header"] = {
            k: header[k]
            for k in (
                "version_number",
                "compiler",
                "build",
                "MESA_SDK_version",
                "math_backend",
                "date",
            )
            if k in header
        }
        row["mesa_version_number"] = header.get("version_number", "")
        row["mesa_compiler"] = header.get("compiler", "")
        row["mesa_compiler_build"] = header.get("build", "")
        row["mesa_sdk_version"] = header.get("MESA_SDK_version", "")
        row["mesa_math_backend"] = header.get("math_backend", "")
    return row


def fmt(val: Any) -> str:
    """Format a value for the CSV table."""
    if val is None:
        return ""
    if isinstance(val, float):
        return "nan" if math.isnan(val) else repr(val)
    if isinstance(val, list):
        return ";".join(str(v) for v in val)
    return str(val)


def cmd_summarize(ns: argparse.Namespace) -> int:
    """Write ``results.json`` and ``results.csv`` from the four cells."""
    work_root = pathlib.Path(ns.work_root)
    out_dir = pathlib.Path(ns.out_dir)
    rows = [
        summarize_cell(work_root, eng, var)
        for eng in ENGINES
        for var in VARIANTS
    ]
    prov = collect_provenance()
    result = {
        "case": f"star/test_suite/{CASE}",
        "changed_control": f"use_Ledoux_criterion in {STAGE_INLIST}",
        "he4_termination_threshold": HE4_STOP,
        "mass_conv_core_milestones_center_he4": list(HE4_MILESTONES),
        "skip_optional_stages": True,
        "provenance": prov,
        "runs": rows,
    }
    (out_dir / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    fields = [k for k in rows[0] if k != "history_header"]
    for r in rows[1:]:
        fields += [k for k in r if k not in fields and k != "history_header"]
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(fields)
        for r in rows:
            wr.writerow([fmt(r.get(k)) for k in fields])
    print(json.dumps(rows, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to a subcommand."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="execute one grid cell")
    p_run.add_argument("--engine", choices=ENGINES, required=True)
    p_run.add_argument("--ledoux", choices=VARIANTS, required=True)
    p_run.add_argument("--work-root", required=True)
    p_run.add_argument("--timeout-sec", type=float, default=5400.0)
    p_sum = sub.add_parser("summarize", help="write results.json and .csv")
    p_sum.add_argument("--work-root", required=True)
    p_sum.add_argument("--out-dir", required=True)
    ns = parser.parse_args(argv)
    return cmd_run(ns) if ns.cmd == "run" else cmd_summarize(ns)


if __name__ == "__main__":
    raise SystemExit(main())
