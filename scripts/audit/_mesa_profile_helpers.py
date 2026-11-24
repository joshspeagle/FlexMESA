# SPDX-License-Identifier: GPL-3.0-or-later
"""Helpers for MESA `26.4.1` logs, `#150` git A/B runs, and CSV writers."""

from __future__ import annotations

import csv
import math
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any

import numpy as np
import numpy.typing as npt

type ArrF64 = npt.NDArray[np.float64]

FORBIDDEN_PUBLIC_TOKENS: tuple[str, ...] = (
    "/" + "home/",
    "/" + "mnt/",
    "/" + "tmp/",
    "." + "venvs",
    "file" + "://",
    "-" + "desktop",
    "\u2014",
)

_MESA_RUNTIME_RE = re.compile(
    r"runtime\s*\(minutes\),\s*retries,\s*steps\s+"
    r"([0-9.+-EeDd]+)\s+(\d+)\s+(\d+)"
)
_TERM_CODE_RE = re.compile(r"termination code:\s*(\S+)")


def assert_clean_public_text(text: str, label: str) -> None:
    """Raise ``ValueError`` if ``text`` contains machine paths or em-dashes."""
    for tok in FORBIDDEN_PUBLIC_TOKENS:
        if tok in text:
            raise ValueError(f"Forbidden token {tok!r} in {label}")


def parse_mesa_history_file(
    history_path: pathlib.Path,
) -> tuple[dict[str, str], list[str], ArrF64]:
    """Parse header metadata, column names, and float64 matrix from MESA log."""
    lines = history_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 7:
        raise ValueError(f"Truncated MESA history file: {history_path.name}")
    hdr_names = lines[1].split()
    hdr_vals = lines[2].split()
    header = {
        k: v.strip('"') for k, v in zip(hdr_names, hdr_vals, strict=False)
    }
    col_names = lines[5].split()
    rows = [
        [float(tok.replace("D", "E").replace("d", "e")) for tok in ln.split()]
        for ln in lines[6:]
        if ln.strip()
    ]
    return header, col_names, np.asarray(rows, dtype=np.float64)


def _estimate_zone_steps_from_history(
    col_names: list[str], data: ArrF64, total_steps: int
) -> tuple[int, int, int]:
    """Estimate zone-steps, initial zones, and final zones from history."""
    if data.ndim != 2 or data.shape[0] == 0:
        return 0, 0, 0
    m_idx = col_names.index("model_number")
    z_idx = col_names.index("num_zones")
    models = data[:, m_idx]
    zones = data[:, z_idx]
    init_z = round(float(zones[0]))
    final_z = round(float(zones[-1]))
    if len(models) == 1:
        return total_steps * final_z, init_z, final_z

    accum = 0.0
    if models[0] > 1.0:
        accum += (models[0] - 1.0) * zones[0]
    accum += zones[0]
    for i in range(1, len(models)):
        dm = max(0.0, float(models[i] - models[i - 1]))
        accum += 0.5 * (float(zones[i - 1]) + float(zones[i])) * dm
    if float(total_steps) > float(models[-1]):
        accum += (float(total_steps) - float(models[-1])) * float(zones[-1])
    return round(accum), init_z, final_z


def _parse_testhub_yml(testhub_path: pathlib.Path) -> dict[str, Any]:
    """Parse exact MESA `test_suite_testhub_end` counters from `testhub.yml`."""
    if not testhub_path.is_file():
        return {}
    parsed: dict[str, Any] = {}
    int_keys = {
        "model_number",
        "num_retries",
        "steps",
        "retries",
        "redos",
        "solver_calls_made",
        "solver_calls_failed",
        "solver_iterations",
    }
    float_keys = {"runtime_minutes", "star_age", "log_rel_run_E_err"}
    for raw_line in testhub_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if ":" not in line or line.startswith("#") or line.startswith("-"):
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()
        if not val:
            continue
        if key in int_keys:
            parsed[key] = int(val)
        elif key in float_keys:
            parsed[key] = float(val.replace("D", "E").replace("d", "e"))
    return parsed


def _clean_mesa_repeat_outputs(work_dir: pathlib.Path) -> None:
    """Remove per-run MESA outputs before each timed `./rn` repeat."""
    for subdir in ("LOGS", "photos"):
        target = work_dir / subdir
        if target.is_dir():
            shutil.rmtree(target)
    for fname in ("testhub.yml", "final.mod"):
        fpath = work_dir / fname
        if fpath.is_file():
            fpath.unlink()


def run_real_mesa_case(
    case_name: str,
    mesa_dir: pathlib.Path,
    mesasdk_root: pathlib.Path,
    omp_threads: int,
    repeats: int = 3,
) -> tuple[dict[str, Any], list[str], ArrF64]:
    """Compile and time a real Fortran MESA `26.4.1` `star/test_suite` case.

    Compiles once in a temporary copy of ``case_name``, then executes
    unmodified ``./rn`` ``repeats`` times with normal ``LOGS/`` output written
    on every timed repeat, returning the median timing plus min-max spread and
    exact ``testhub.yml`` Newton iteration counts.
    """
    src_case = mesa_dir / "star" / "test_suite" / case_name
    if not src_case.is_dir():
        raise FileNotFoundError(f"Missing MESA case directory: {case_name}")
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")

    sysdeps = pathlib.Path.home() / "mesasdk-sysdeps"
    init_cmd = (
        f'export MESASDK_ROOT="{mesasdk_root}"; '
        f'export MESA_DIR="{mesa_dir}"; '
        f'export PATH="{sysdeps}/bin:{sysdeps}/usr/bin:$PATH"; '
        f'export LIBRARY_PATH="{sysdeps}/usr/lib/x86_64-linux-gnu'
        '${LIBRARY_PATH:+:$LIBRARY_PATH}"; '
        f'export CPATH="{sysdeps}/usr/include:'
        f'{sysdeps}/usr/include/x86_64-linux-gnu${{CPATH:+:$CPATH}}"; '
        'source "$MESASDK_ROOT/bin/mesasdk_init.sh"; '
        "unset DISPLAY || true; "
        f"export OMP_NUM_THREADS={omp_threads}; "
        f"export OPENBLAS_NUM_THREADS={omp_threads}; "
        f"export MKL_NUM_THREADS={omp_threads}; "
        "export MESA_SKIP_OPTIONAL=t; "
    )

    with tempfile.TemporaryDirectory(prefix=f"mesa2641_{case_name}_") as tmp:
        work_dir = pathlib.Path(tmp) / case_name
        shutil.copytree(src_case, work_dir)
        _clean_mesa_repeat_outputs(work_dir)
        build_proc = subprocess.run(
            ["bash", "-lc", init_cmd + 'make -C make MESA_DIR="$MESA_DIR"'],
            cwd=work_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if build_proc.returncode != 0:
            raise RuntimeError(
                f"MESA compile failed for {case_name}:\n{build_proc.stderr}"
            )

        rn_samples: list[float] = []
        evolve_samples: list[float] = []
        last_meta: dict[str, Any] = {}
        cols_out: list[str] = []
        arr_out: ArrF64 = np.empty((0, 0), dtype=np.float64)

        for _ in range(repeats):
            _clean_mesa_repeat_outputs(work_dir)
            t0 = time.perf_counter()
            rn_proc = subprocess.run(
                ["bash", "-lc", init_cmd + "./rn"],
                cwd=work_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            rn_wall_sec = time.perf_counter() - t0
            out_txt = rn_proc.stdout + "\n" + rn_proc.stderr
            if rn_proc.returncode != 0:
                raise RuntimeError(
                    f"MESA ./rn failed for {case_name} ({rn_proc.returncode})"
                )

            matches = _MESA_RUNTIME_RE.findall(out_txt)
            if not matches:
                raise RuntimeError(
                    f"Could not parse MESA runtime summary for {case_name}"
                )
            rt_min_str, retries_str, steps_str = matches[-1]
            mesa_evolve_sec = (
                float(rt_min_str.replace("D", "E").replace("d", "e")) * 60.0
            )
            retries = int(retries_str)
            steps = int(steps_str)
            term_matches = _TERM_CODE_RE.findall(out_txt)
            term_code = term_matches[-1] if term_matches else "unknown"

            testhub = _parse_testhub_yml(work_dir / "testhub.yml")
            hist_path = work_dir / "LOGS" / "history.data"
            hdr, cols, arr = parse_mesa_history_file(hist_path)
            zone_steps, init_z, final_z = _estimate_zone_steps_from_history(
                cols, arr, steps
            )
            if "solver_iterations" in testhub:
                iters_total = int(testhub["solver_iterations"])
                iters_source = (
                    "testhub.yml (exact s% total_num_solver_iterations)"
                )
            elif "num_iters" in cols and len(cols) > 0 and arr.shape[0] > 0:
                it_col = arr[:, cols.index("num_iters")]
                iters_total = round(float(np.mean(it_col)) * steps)
                iters_source = "history.data mean(num_iters) * steps"
            else:
                iters_total = 0
                iters_source = "unavailable"

            rn_samples.append(rn_wall_sec)
            evolve_samples.append(mesa_evolve_sec)
            cols_out = cols
            arr_out = arr
            last_meta = {
                "hdr": hdr,
                "term_code": term_code,
                "steps": steps,
                "retries": retries,
                "redos": int(testhub.get("redos", 0)),
                "solver_calls_made": int(testhub.get("solver_calls_made", 0)),
                "solver_calls_failed": int(
                    testhub.get("solver_calls_failed", 0)
                ),
                "iters_total": iters_total,
                "iters_source": iters_source,
                "zone_steps": zone_steps,
                "init_z": init_z,
                "final_z": final_z,
            }

        rn_sorted = sorted(rn_samples)
        ev_sorted = sorted(evolve_samples)
        rn_med = rn_sorted[len(rn_sorted) // 2]
        ev_med = ev_sorted[len(ev_sorted) // 2]

        hdr = last_meta["hdr"]
        steps = int(last_meta["steps"])
        retries = int(last_meta["retries"])
        iters_total = int(last_meta["iters_total"])
        zone_steps = int(last_meta["zone_steps"])
        col_map = {c: idx for idx, c in enumerate(cols_out)}
        last = arr_out[-1]
        steps_f = max(1.0, float(steps))
        zs_f = max(1.0, float(zone_steps))
        nit_f = max(1.0, float(iters_total))
        rec: dict[str, Any] = {
            "case": case_name,
            "mesa_version_number": hdr.get("version_number", "r26.4.1"),
            "compiler": hdr.get("compiler", "gfortran"),
            "build": hdr.get("build", ""),
            "math_backend": hdr.get("math_backend", "CRMATH"),
            "omp_threads": omp_threads,
            "repeats": repeats,
            "history_output_every_repeat": True,
            "termination_code": last_meta["term_code"],
            "rn_wall_sec": rn_med,
            "rn_wall_sec_samples": rn_sorted,
            "rn_wall_sec_min": rn_sorted[0],
            "rn_wall_sec_max": rn_sorted[-1],
            "rn_wall_sec_range": rn_sorted[-1] - rn_sorted[0],
            "mesa_internal_evolve_sec": ev_med,
            "mesa_internal_evolve_sec_samples": ev_sorted,
            "mesa_internal_evolve_sec_min": ev_sorted[0],
            "mesa_internal_evolve_sec_max": ev_sorted[-1],
            "mesa_internal_evolve_sec_range": ev_sorted[-1] - ev_sorted[0],
            "steps_taken": steps,
            "total_retries": retries,
            "total_redos": int(last_meta["redos"]),
            "total_newton_iterations": iters_total,
            "newton_iterations_source": last_meta["iters_source"],
            "newton_iters_per_step": float(iters_total) / steps_f,
            "solver_calls_made": int(last_meta["solver_calls_made"]),
            "solver_calls_failed": int(last_meta["solver_calls_failed"]),
            "total_zone_steps": zone_steps,
            "initial_zones": int(last_meta["init_z"]),
            "final_zones": int(last_meta["final_z"]),
            "mean_zones": float(zone_steps) / steps_f,
            "rn_wall_ms_per_step": 1000.0 * rn_med / steps_f,
            "rn_wall_us_per_zone_step": 1.0e6 * rn_med / zs_f,
            "rn_wall_ms_per_newton_iter": 1000.0 * rn_med / nit_f,
            "evolve_wall_ms_per_step": 1000.0 * ev_med / steps_f,
            "evolve_wall_us_per_zone_step": 1.0e6 * ev_med / zs_f,
            "evolve_wall_ms_per_newton_iter": 1000.0 * ev_med / nit_f,
            "final_star_age_yr": float(last[col_map["star_age"]]),
            "final_log_l": float(last[col_map["log_L"]]),
            "final_log_teff": float(last[col_map["log_Teff"]]),
            "final_log_r": float(last[col_map["log_R"]]),
            "final_log_center_t": float(last[col_map["log_cntr_T"]]),
            "final_log_center_rho": float(last[col_map["log_cntr_Rho"]]),
            "final_center_h1": float(last[col_map["center_h1"]]),
            "final_center_he4": float(last[col_map["center_he4"]]),
        }
        return rec, cols_out, arr_out


def compare_trajectories_by_age(
    cols_a: list[str],
    arr_a: ArrF64,
    cols_b: list[str],
    arr_b: ArrF64,
    tracked_cols: tuple[str, ...],
) -> dict[str, Any]:
    """Compare final-state and age-interpolated track differences (`b - a`)."""
    map_a = {c: i for i, c in enumerate(cols_a)}
    map_b = {c: i for i, c in enumerate(cols_b)}
    age_a = arr_a[:, map_a["star_age"]]
    age_b = arr_b[:, map_b["star_age"]]
    final_age_a = float(age_a[-1])
    final_age_b = float(age_b[-1])
    rel_age_diff = (final_age_b - final_age_a) / max(abs(final_age_a), 1.0e-30)

    # Monotone grid for age interpolation across overlap interval.
    lo = max(float(age_a[0]), float(age_b[0]))
    hi = min(final_age_a, final_age_b)
    grid = (
        np.linspace(lo, hi, 256, dtype=np.float64)
        if hi > lo
        else np.asarray([lo], dtype=np.float64)
    )
    order_a = np.argsort(age_a)
    order_b = np.argsort(age_b)
    ua, idx_a = np.unique(age_a[order_a], return_index=True)
    ub, idx_b = np.unique(age_b[order_b], return_index=True)

    col_diffs: dict[str, dict[str, float]] = {}
    for col in tracked_cols:
        if col not in map_a or col not in map_b:
            continue
        va = arr_a[:, map_a[col]]
        vb = arr_b[:, map_b[col]]
        fin_diff = float(vb[-1] - va[-1])
        fin_abs = abs(fin_diff)
        if len(ua) >= 2 and len(ub) >= 2 and hi > lo:
            ia = np.interp(grid, ua, va[order_a][idx_a])
            ib = np.interp(grid, ub, vb[order_b][idx_b])
            trk_max_abs = float(np.max(np.abs(ib - ia)))
        else:
            trk_max_abs = fin_abs
        col_diffs[col] = {
            "final_signed_diff": fin_diff,
            "final_abs_diff": fin_abs,
            "track_max_abs_diff": trk_max_abs,
        }
    return {
        "final_age_a_yr": final_age_a,
        "final_age_b_yr": final_age_b,
        "rel_star_age_diff": rel_age_diff,
        "columns": col_diffs,
    }


_BENCH_150_EXAMPLE_RS = """// SPDX-License-Identifier: GPL-3.0-or-later
use std::path::Path;
use std::time::Instant;
use flexmesa_star::controls::{StarControls, StarJobControls};
use flexmesa_star::hydro_vars::set_hydro_vars;
use flexmesa_star::inlist::InlistLoadOptions;
use flexmesa_star::inlist_runner::{
    run_test_suite_case_with_options, StarLogsOptions,
};
use flexmesa_star::zams::load_zams_model;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let case_dir = Path::new(&args[1]);
    let repeats: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let mut warm = load_zams_model(
        StarControls::default(),
        StarJobControls::default(),
    ).expect("warmup zams");
    set_hydro_vars(&mut warm, false, false).expect("warmup hydro");

    let mut secs = Vec::with_capacity(repeats);
    let mut steps = 0_usize;
    for _ in 0..repeats {
        let t0 = Instant::now();
        let rep = run_test_suite_case_with_options(
            case_dir,
            true,
            None,
            false,
            StarLogsOptions::default(),
            InlistLoadOptions { allow_unsupported_controls: true },
        ).expect("case run");
        secs.push(t0.elapsed().as_secs_f64());
        steps = rep.total_steps;
    }
    secs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = secs[secs.len() / 2];
    let q1 = secs[0];
    let q3 = *secs.last().unwrap();
    println!("{med:.9e} {:.9e} {steps}", q3 - q1);
}
"""


def measure_commit_case_wall(
    repo_root: pathlib.Path,
    commit_ref: str,
    case_dir: pathlib.Path,
    rayon_threads: int,
    repeats: int = 3,
    shared_target_dir: pathlib.Path | None = None,
) -> dict[str, Any]:
    """Measure median evolution wall time on ``commit_ref`` in detached tree."""
    full_sha = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", commit_ref], text=True
    ).strip()
    short_sha = full_sha[:8]
    with tempfile.TemporaryDirectory(prefix=f"wt150_{short_sha}_") as tmp:
        wt_dir = pathlib.Path(tmp) / "wt"
        tgt_dir = (
            shared_target_dir
            if shared_target_dir is not None
            else pathlib.Path(tmp) / "target"
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "worktree",
                "add",
                "--detach",
                str(wt_dir),
                full_sha,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        ex_dir = wt_dir / "crates" / "flexmesa-star" / "examples"
        ex_dir_existed = ex_dir.exists()
        ex_file = ex_dir / "bench_150_ab.rs"
        try:
            ex_dir.mkdir(parents=True, exist_ok=True)
            ex_file.write_text(_BENCH_150_EXAMPLE_RS, encoding="utf-8")
            env = dict(os.environ)
            env["CARGO_TARGET_DIR"] = str(tgt_dir)
            env["CARGO_BUILD_JOBS"] = "3"
            env["RAYON_NUM_THREADS"] = str(rayon_threads)
            env["OMP_NUM_THREADS"] = str(rayon_threads)
            env["OPENBLAS_NUM_THREADS"] = str(rayon_threads)
            env["MKL_NUM_THREADS"] = str(rayon_threads)
            env["FLEXMESA_NUMERICS_MODE"] = "strict"
            subprocess.run(
                [
                    "cargo",
                    "build",
                    "--release",
                    "-p",
                    "flexmesa-star",
                    "--example",
                    "bench_150_ab",
                ],
                cwd=wt_dir,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            bin_path = tgt_dir / "release" / "examples" / "bench_150_ab"
            out = subprocess.check_output(
                [str(bin_path), str(case_dir), str(repeats)],
                cwd=wt_dir,
                env=env,
                text=True,
            ).strip()
            parts = out.split()
            med_sec = float(parts[0])
            span_sec = float(parts[1])
            steps = int(parts[2])
        finally:
            if ex_file.exists():
                ex_file.unlink()
            if not ex_dir_existed and ex_dir.exists():
                ex_dir.rmdir()
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo_root),
                    "worktree",
                    "remove",
                    str(wt_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
    return {
        "commit": full_sha,
        "short_commit": short_sha,
        "repeats": repeats,
        "rayon_threads": rayon_threads,
        "median_wall_sec": med_sec,
        "range_wall_sec": span_sec,
        "steps_taken": steps,
        "wall_ms_per_step": 1000.0 * med_sec / max(1.0, float(steps)),
    }


def write_csv_rows(
    path: pathlib.Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
    comment_header: list[str] | None = None,
) -> None:
    """Write deterministic CSV output with optional `#` provenance lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        if comment_header:
            for line in comment_header:
                fh.write(f"# {line}\n")
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted: dict[str, Any] = {}
            for k in fieldnames:
                v = row.get(k, "")
                if isinstance(v, float):
                    formatted[k] = f"{v:.6f}" if not math.isnan(v) else "nan"
                else:
                    formatted[k] = v
            writer.writerow(formatted)
    assert_clean_public_text(path.read_text(encoding="utf-8"), str(path.name))
