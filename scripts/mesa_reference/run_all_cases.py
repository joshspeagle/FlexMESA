"""Execute all MESA 26.4.1 test_suite cases (star, binary, astero) in order.

Reads each suite's ``do1_test_source`` directly from ``$MESA_DIR``, orders cases
from shortest to longest, executes ``run_mesa_case_26_4_1.sh`` with bounded
concurrency (default 2 workers x ``OMP_NUM_THREADS=4`` = 8 physical cores on
Ryzen 7 9800X3D) and per-case timeout/RSS guards, extracts run telemetry, runs
``notes/audit/tools/thin_mesa_history.py`` on every produced star history, and
removes only transient build artifacts (``./star``, ``./binary``, ``make/*.o``,
``photos/``, ``LOGS*/profile*.data``, ``.mesa_temp_cache``) while preserving
``out.txt``, all ``LOGS*/history.data``, ``binary_history.data``, final ``.mod``
files, and all ``pulse_tap/`` / ``outputs/`` / ``astero_results/`` files.
"""

from __future__ import annotations

import argparse
from concurrent import futures
import glob
import hashlib
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = pathlib.Path(
    os.environ.get(
        "MESA_GROUND_TRUTH_ROOT",
        str(pathlib.Path.home() / "mesa-ground-truth-26.4.1"),
    )
)
MESA_DIR = pathlib.Path(
    os.environ.get("MESA_DIR", str(pathlib.Path.home() / "mesa-26.04.1"))
)
FLEXMESA_DIR = pathlib.Path(
    os.environ.get("FLEXMESA_DIR", str(SCRIPT_DIR.parents[1]))
)
THIN_SCRIPT = FLEXMESA_DIR / "notes/audit/tools/thin_mesa_history.py"
RUN_SCRIPT = SCRIPT_DIR / "run_mesa_case_26_4_1.sh"

PROVENANCE_NOTE_PREFIX = (
    "MESA 26.4.1 (c8c2026), MESA SDK 26.6.1 (gfortran 15.2.0), "
    "OMP_NUM_THREADS=4, MESA_SKIP_OPTIONAL=t"
)


def sha256_file(path: pathlib.Path) -> str:
    """Compute hex SHA-256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_do1_test_source(suite: str) -> list[dict[str, Any]]:
    """Parse ``do_one`` entries from ``<suite>/test_suite/do1_test_source``."""
    src_path = MESA_DIR / suite / "test_suite" / "do1_test_source"
    lines = src_path.read_text(encoding="utf-8").splitlines()
    entries: list[dict[str, Any]] = []
    in_slow_block = False
    blank_count_after_slow = 0

    for idx, raw in enumerate(lines):
        s = raw.strip()
        if s.lower().startswith("# slow cases"):
            in_slow_block = True
            blank_count_after_slow = 0
            continue
        if not s:
            if in_slow_block:
                blank_count_after_slow += 1
                if blank_count_after_slow >= 2:
                    in_slow_block = False
            continue
        if s.startswith("#"):
            continue
        m = re.match(r'^do_one\s+(\S+)\s+"([^"]*)"\s+(\S+)\s+(\S+)', s)
        if not m:
            raise ValueError(f"Unparsed line in {src_path}:{idx + 1}: {raw}")
        case_name, expected_msg, expected_mod, restart_photo = m.groups()
        expected_mod = expected_mod.strip('"')
        restart_photo = restart_photo.strip('"')
        entries.append(
            {
                "suite": suite,
                "case": case_name,
                "order_in_suite": len(entries) + 1,
                "expected_msg": expected_msg,
                "expected_mod": expected_mod,
                "restart_photo": restart_photo,
                "marked_slow_upstream": in_slow_block,
            }
        )
    return entries


def estimate_cost_score(entry: dict[str, Any]) -> tuple[int, int, str]:
    """Assign a rough priority tuple so fast cases run before multi-hour cases."""
    suite = entry["suite"]
    case = entry["case"]
    case_dir = MESA_DIR / suite / "test_suite" / case

    if case == "1.4M_ms_op_mono":
        return (0, 0, case)

    score = 10
    if entry["marked_slow_upstream"]:
        score += 500

    rn_text = (case_dir / "rn").read_text(encoding="utf-8", errors="replace")
    do_one_calls = len(re.findall(r"^\s*do_one\b", rn_text, re.M))
    score += 25 * max(0, do_one_calls - 1)

    inlist_text = ""
    for p in sorted(case_dir.glob("inlist*")):
        if p.is_file():
            inlist_text += "\n" + p.read_text(
                encoding="utf-8", errors="replace"
            )

    if re.search(
        r"mesa_(80|128|151|204)|approx21|co_burn|cno_extras", inlist_text
    ):
        score += 120
    if re.search(r"do_element_diffusion\s*=\s*\.true\.", inlist_text, re.I):
        score += 80
    if re.search(r"op_mono", inlist_text, re.I):
        score += 80
    if case in {
        "radiative_levitation",
        "simplex_solar_calibration",
        "wd_aic",
        "tzo",
        "make_co_wd",
        "make_o_ne_wd",
        "7M_prems_to_AGB",
        "5M_cepheid_blue_loop",
        "20M_z2m2_high_rotation",
        "rsp_check_2nd_crossing",
        "example_astero",
        "fast_simplex",
        "fast_newuoa",
        "fast_scan_grid",
        "surface_effects",
        "double_bh",
        "wind_fed_bhhmxb",
    }:
        score += 150

    tier = 1 if score < 80 else (2 if score < 300 else 3)
    return (tier, score, f"{suite}/{case}")


def tree_rss_kb(root_pid: int) -> int:
    """Sum VmRSS in kB across ``root_pid`` and its descendants."""
    children_map: dict[int, list[int]] = {}
    rss_map: dict[int, int] = {}
    for proc_dir in glob.glob("/proc/[0-9]*"):
        try:
            pid = int(os.path.basename(proc_dir))
            with open(f"{proc_dir}/status", encoding="utf-8") as fh:
                ppid = 0
                rss = 0
                for ln in fh:
                    if ln.startswith("PPid:"):
                        ppid = int(ln.split()[1])
                    elif ln.startswith("VmRSS:"):
                        rss = int(ln.split()[1])
                children_map.setdefault(ppid, []).append(pid)
                rss_map[pid] = rss
        except OSError:
            continue

    total = 0
    stack = [root_pid]
    seen: set[int] = set()
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        total += rss_map.get(p, 0)
        stack.extend(children_map.get(p, []))
    return total


def parse_history_tail(
    hist_path: pathlib.Path,
) -> tuple[int | None, int | None]:
    """Extract (model_number, num_retries) from the last row of a history file."""
    try:
        lines = hist_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if len(lines) < 7:
            return (None, None)
        names = lines[5].split()
        last_row = None
        for ln in reversed(lines[6:]):
            if ln.strip():
                last_row = ln.split()
                break
        if not last_row:
            return (None, None)
        m_idx = names.index("model_number") if "model_number" in names else None
        r_idx = names.index("num_retries") if "num_retries" in names else None
        m_val = int(float(last_row[m_idx])) if m_idx is not None else None
        r_val = int(float(last_row[r_idx])) if r_idx is not None else None
        return (m_val, r_val)
    except Exception:
        return (None, None)


def parse_case_outputs(
    entry: dict[str, Any],
    dest: pathlib.Path,
    pre_mods: dict[str, str],
    wall_s: float,
    timed_out: bool,
    rss_killed: bool,
    peak_rss_gb: float,
    proc_rc: int,
) -> dict[str, Any]:
    """Extract termination code, steps, retries, histories, mods, and status."""
    suite = entry["suite"]
    case = entry["case"]
    out_path = dest / "out.txt"
    mk_path = dest / "mk.log"
    out_text = (
        out_path.read_text(encoding="utf-8", errors="replace")
        if out_path.exists()
        else ""
    )
    mk_text = (
        mk_path.read_text(encoding="utf-8", errors="replace")
        if mk_path.exists()
        else ""
    )

    rn_exit_match = re.findall(r"^rn exit (\d+)\s*$", out_text, re.M)
    rn_exit = (
        int(rn_exit_match[-1])
        if rn_exit_match
        else (0 if proc_rc == 0 else proc_rc)
    )

    intentionally_skipped = bool(
        re.search(r"this test was intentionally skipped", out_text, re.I)
    )
    compile_failed = (not out_path.exists()) and proc_rc != 0

    # Stage runtime summaries: runtime (minutes), retries, steps
    stage_runs: list[dict[str, Any]] = []
    for m in re.finditer(
        r"runtime \(minutes\), retries, steps\s+([0-9.Ee+-]+)\s+(\d+)\s+(\d+)",
        out_text,
    ):
        stage_runs.append(
            {
                "mesa_runtime_min": float(m.group(1)),
                "retries": int(m.group(2)),
                "steps": int(m.group(3)),
            }
        )

    # Termination codes printed by MESA
    term_codes = [
        m.group(1).strip()
        for m in re.finditer(r"^termination code:\s*(.+?)\s*$", out_text, re.M)
    ]
    if not term_codes:
        for pat in (
            r"^(Terminate\b.+?)\s*$",
            r"^(terminate due to\b.+?)\s*$",
            r"^(stop because\b.+?)\s*$",
            r"^\s*(save_sample_results_to_file\s+\S+)\s*$",
        ):
            found = [
                m.group(1).strip() for m in re.finditer(pat, out_text, re.M)
            ]
            if found:
                term_codes = found
                break

    termination_code = term_codes[-1] if term_codes else None

    # Locate history files
    logs_dirs = sorted(
        [p for p in dest.glob("LOGS*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    history_relpaths: list[str] = []
    for ld in logs_dirs:
        hp = ld / "history.data"
        if hp.exists():
            history_relpaths.append(str(hp.relative_to(dest)))
    bin_hist = dest / "binary_history.data"
    if bin_hist.exists():
        history_relpaths.append("binary_history.data")

    # Determine steps & retries
    final_steps: int | None = stage_runs[-1]["steps"] if stage_runs else None
    final_retries: int | None = (
        stage_runs[-1]["retries"] if stage_runs else None
    )
    total_steps: int | None = (
        sum(x["steps"] for x in stage_runs) if stage_runs else None
    )
    total_retries: int | None = (
        sum(x["retries"] for x in stage_runs) if stage_runs else None
    )

    if final_steps is None and history_relpaths:
        primary_hist = dest / (
            "LOGS1/history.data"
            if (dest / "LOGS1/history.data").exists()
            else history_relpaths[-1]
        )
        if (
            primary_hist.name == "binary_history.data"
            and len(history_relpaths) > 1
        ):
            primary_hist = dest / history_relpaths[0]
        h_steps, h_retries = parse_history_tail(primary_hist)
        final_steps = h_steps
        final_retries = h_retries
        total_steps = h_steps
        total_retries = h_retries

    # Saved .mod files (newly created or modified relative to upstream directory)
    produced_mods: list[str] = []
    for mp in sorted(dest.glob("*.mod")):
        digest = sha256_file(mp)
        if mp.name not in pre_mods or pre_mods[mp.name] != digest:
            produced_mods.append(mp.name)
    for sub in ("outputs", "astero_results"):
        subdir = dest / sub
        if subdir.is_dir():
            for mp in sorted(subdir.glob("*.mod")):
                produced_mods.append(str(mp.relative_to(dest)))

    expected_msg = entry["expected_msg"]
    matched_msg = expected_msg in out_text if expected_msg else True
    expected_mod = entry["expected_mod"]
    has_expected_mod = (
        True if expected_mod == "skip" else (dest / expected_mod).exists()
    )

    # Pulse tap telemetry
    modes_tsv = dest / "pulse_tap/modes.tsv"
    pulse_modes_count = 0
    pulse_models_count = 0
    if modes_tsv.exists():
        with modes_tsv.open(encoding="utf-8") as fh:
            pulse_modes_count = max(0, sum(1 for _ in fh) - 1)
    models_dir = dest / "pulse_tap/models"
    if models_dir.is_dir():
        pulse_models_count = len(list(models_dir.glob("*.mod")))

    # Determine status & failure/skip reason
    status = "ok"
    reason = ""
    if timed_out:
        status = "timeout"
        reason = f"Exceeded per-case wall timeout ({wall_s / 60.0:.1f} min)"
    elif rss_killed:
        status = "rss_exceeded"
        reason = f"Exceeded per-case RSS cap ({peak_rss_gb:.2f} GB)"
    elif compile_failed:
        status = "compile_failed"
        last_mk = (
            mk_text.strip().splitlines()[-1]
            if mk_text.strip()
            else "make failed"
        )
        reason = f"Compilation failed: {last_mk}"
    elif intentionally_skipped:
        status = "skipped_upstream"
        if "op_mono_data_path" in out_text:
            reason = "Intentionally skipped by upstream case: external OP_mono opacity tables (OP4STARS_1.3) not installed"
        else:
            reason = "Intentionally skipped by upstream run_star_extras"
    elif rn_exit != 0:
        status = "failed"
        reason = f"./rn exited with code {rn_exit}"
    elif not has_expected_mod:
        status = "failed"
        reason = f"Expected output model {expected_mod} was not created"
    elif not matched_msg:
        status = "failed"
        reason = f"Expected message {expected_msg!r} not found in out.txt"

    return {
        "suite": suite,
        "case": case,
        "order_in_suite": entry["order_in_suite"],
        "status": status,
        "reason": reason,
        "rn_exit_code": rn_exit,
        "expected_msg": expected_msg,
        "matched_expected_msg": matched_msg,
        "expected_mod": expected_mod,
        "has_expected_mod": has_expected_mod,
        "termination_code": termination_code,
        "all_termination_codes": term_codes,
        "steps": final_steps,
        "retries": final_retries,
        "total_steps_all_stages": total_steps,
        "total_retries_all_stages": total_retries,
        "num_stages_reported": len(stage_runs),
        "stage_runs": stage_runs,
        "wall_seconds": round(wall_s, 2),
        "wall_minutes": round(wall_s / 60.0, 3),
        "peak_rss_gb": round(peak_rss_gb, 3),
        "history_files": history_relpaths,
        "produced_mods": produced_mods,
        "pulse_modes_count": pulse_modes_count,
        "pulse_models_count": pulse_models_count,
    }


def write_thinned_histories(
    entry: dict[str, Any], dest: pathlib.Path, result: dict[str, Any]
) -> list[str]:
    """Run ``thin_mesa_history.py`` on produced star histories."""
    suite = entry["suite"]
    case = entry["case"]
    out_dir = ROOT_DIR / "thinned" / suite
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    logs_dirs = sorted(
        [
            p
            for p in dest.glob("LOGS*")
            if p.is_dir() and (p / "history.data").exists()
        ],
        key=lambda p: p.stat().st_mtime,
    )
    if not logs_dirs:
        return written

    if suite in ("star", "astero") and case != "twin_studies":
        # Primary history = final stage LOGS directory (e.g. LOGS or LOGS_to_wd / LOGS_part5)
        primary_ld = (
            dest / "LOGS"
            if (dest / "LOGS/history.data").exists()
            else logs_dirs[-1]
        )
        stage_label = (
            "single stage"
            if len(logs_dirs) == 1 and result["num_stages_reported"] <= 1
            else f"final stage ({primary_ld.name})"
        )
        note = f"{PROVENANCE_NOTE_PREFIX}, {stage_label}; {suite}/test_suite/{case}"
        dst_tsv = out_dir / f"{case}.tsv"
        subprocess.run(
            [
                sys.executable,
                str(THIN_SCRIPT),
                str(primary_ld / "history.data"),
                str(dst_tsv),
                note,
            ],
            check=True,
        )
        written.append(str(dst_tsv.relative_to(ROOT_DIR)))

        if len(logs_dirs) > 1:
            for ld in logs_dirs:
                st_dst = out_dir / f"{case}__{ld.name}.tsv"
                st_note = f"{PROVENANCE_NOTE_PREFIX}, stage {ld.name}; {suite}/test_suite/{case}"
                subprocess.run(
                    [
                        sys.executable,
                        str(THIN_SCRIPT),
                        str(ld / "history.data"),
                        str(st_dst),
                        st_note,
                    ],
                    check=True,
                )
                written.append(str(st_dst.relative_to(ROOT_DIR)))
    else:
        # binary or twin_studies: LOGS1, LOGS2
        for ld in sorted(logs_dirs, key=lambda p: p.name):
            st_dst = out_dir / f"{case}__{ld.name}.tsv"
            st_note = f"{PROVENANCE_NOTE_PREFIX}, {ld.name}; {suite}/test_suite/{case}"
            subprocess.run(
                [
                    sys.executable,
                    str(THIN_SCRIPT),
                    str(ld / "history.data"),
                    str(st_dst),
                    st_note,
                ],
                check=True,
            )
            written.append(str(st_dst.relative_to(ROOT_DIR)))

        # For binary_history.data, also write a compact tab-separated table keeping key orbital columns
        bin_hist = dest / "binary_history.data"
        if bin_hist.exists():
            bh_lines = bin_hist.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            if len(bh_lines) >= 6:
                names = bh_lines[5].split()
                bh_dst = out_dir / f"{case}__binary_history.tsv"
                with bh_dst.open("w", encoding="utf-8") as fh:
                    fh.write(
                        f"# {PROVENANCE_NOTE_PREFIX}, binary_history.data; {suite}/test_suite/{case}\n"
                    )
                    fh.write("\t".join(names) + "\n")
                    for ln in bh_lines[6:]:
                        cols = ln.split()
                        if cols:
                            fh.write("\t".join(cols) + "\n")
                written.append(str(bh_dst.relative_to(ROOT_DIR)))

    return written


def prune_transient_build_files(
    dest: pathlib.Path, pre_mods: dict[str, str], expected_mod: str
) -> None:
    """Remove compiled executables, .o/.mod build files, photos, and radial profiles."""
    for exe in ("star", "binary"):
        p = dest / exe
        if p.is_file():
            p.unlink()
    for dname in (".mesa_temp_cache", "photos", "png", "pgstar_out"):
        d = dest / dname
        if d.is_dir():
            shutil.rmtree(d)
    make_dir = dest / "make"
    if make_dir.is_dir():
        for pat in ("*.o", "*.mod", "*.smod"):
            for fp in make_dir.glob(pat):
                fp.unlink()
    for ld in dest.glob("LOGS*"):
        if ld.is_dir():
            for prof in ld.glob("profile*.data"):
                prof.unlink()
    # Keep produced .mod files; remove unchanged pre-existing input .mod copies to avoid duplicating upstream inputs
    for mp in list(dest.glob("*.mod")):
        if (
            mp.name != expected_mod
            and mp.name in pre_mods
            and sha256_file(mp) == pre_mods[mp.name]
        ):
            mp.unlink()


def run_single_case(
    entry: dict[str, Any],
    timeout_s: float,
    max_rss_gb: float,
) -> dict[str, Any]:
    """Run one MESA test_suite case with timeout and RSS guard."""
    suite = entry["suite"]
    case = entry["case"]
    suite_out = ROOT_DIR / "runs" / suite
    suite_out.mkdir(parents=True, exist_ok=True)
    dest = suite_out / case
    res_json = dest / "case_result.json"

    if res_json.exists():
        return json.loads(res_json.read_text(encoding="utf-8"))

    if dest.exists():
        shutil.rmtree(dest)

    # Hash upstream .mod files in the source case directory before running
    src_case_dir = MESA_DIR / suite / "test_suite" / case
    pre_mods = {
        mp.name: sha256_file(mp) for mp in sorted(src_case_dir.glob("*.mod"))
    }

    env = os.environ.copy()
    env["OUT"] = str(suite_out)
    env["OMP_NUM_THREADS"] = "4"

    t0 = time.monotonic()
    proc = subprocess.Popen(
        ["bash", str(RUN_SCRIPT), suite, case],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    timed_out = False
    rss_killed = False
    peak_kb = 0
    cap_kb = int(max_rss_gb * 1_000_000)

    while proc.poll() is None:
        rss_now = tree_rss_kb(proc.pid)
        if rss_now > peak_kb:
            peak_kb = rss_now
        elapsed = time.monotonic() - t0
        if elapsed > timeout_s:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            break
        if peak_kb > cap_kb:
            rss_killed = True
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            break
        time.sleep(2.0)

    wall_s = time.monotonic() - t0
    rc = proc.returncode if proc.returncode is not None else -9

    result = parse_case_outputs(
        entry=entry,
        dest=dest,
        pre_mods=pre_mods,
        wall_s=wall_s,
        timed_out=timed_out,
        rss_killed=rss_killed,
        peak_rss_gb=peak_kb / 1_000_000.0,
        proc_rc=rc,
    )

    thinned_files: list[str] = []
    if result["history_files"] and result["status"] == "ok":
        try:
            thinned_files = write_thinned_histories(entry, dest, result)
        except Exception as exc:
            result["status"] = "failed"
            result["reason"] = f"thin_mesa_history failed: {exc}"
            result["thin_error"] = str(exc)
    result["thinned_files"] = thinned_files

    prune_transient_build_files(dest, pre_mods, entry["expected_mod"])
    res_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=14400.0)
    parser.add_argument("--max-rss-gb", type=float, default=20.0)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    all_entries: list[dict[str, Any]] = []
    for suite in ("star", "binary", "astero"):
        all_entries.extend(parse_do1_test_source(suite))

    ranked = sorted(all_entries, key=estimate_cost_score)
    if args.limit > 0:
        ranked = ranked[: args.limit]

    if args.list_only:
        print(f"Total cases queued: {len(ranked)}")
        for idx, e in enumerate(ranked, 1):
            tier, score, _ = estimate_cost_score(e)
            print(
                f"{idx:3d}. [T{tier} score={score:3d}] {e['suite']:6s} {e['case']}"
            )
        return

    ROOT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = ROOT_DIR / "campaign_progress.log"

    def log(msg: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    log(
        f"Starting MESA 26.4.1 reference campaign: {len(ranked)} cases, "
        f"workers={args.workers}, OMP_NUM_THREADS=4, timeout_s={args.timeout_s:.0f}"
    )

    done_count = 0
    total = len(ranked)
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_entry = {
            pool.submit(run_single_case, e, args.timeout_s, args.max_rss_gb): e
            for e in ranked
        }
        for fut in futures.as_completed(future_to_entry):
            e = future_to_entry[fut]
            done_count += 1
            try:
                res = fut.result()
                log(
                    f"[{done_count:3d}/{total}] {res['suite']}/{res['case']}: "
                    f"status={res['status']} term={res['termination_code']!r} "
                    f"steps={res['steps']} retries={res['retries']} "
                    f"wall_min={res['wall_minutes']:.2f} "
                    f"reason={res['reason']!r}"
                )
            except Exception as exc:
                log(
                    f"[{done_count:3d}/{total}] {e['suite']}/{e['case']}: EXCEPTION {exc!r}"
                )

    log("Campaign finished.")


if __name__ == "__main__":
    main()
