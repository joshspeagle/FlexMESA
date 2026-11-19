#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Run the `#168` quiet-host stellar evolution profiling campaign (`26.4.1`).

Executes and records raw JSON/CSV artifacts under
`benchmarks/evolution_profile/26.4.1/`:
1. Four real `star/test_suite` evolutions (`1.3M_ms_high_Z`, `low_z`, `high_z`,
   `make_metals`) in `NumericsMode::Strict` (unperturbed + per-step phase
   breakdown and microphysics call counts per accepted step).
2. Real Fortran MESA `26.4.1` on the same four cases at matched thread count
   (`OMP_NUM_THREADS=2`), recording wall/step, wall/zone-step, steps, retries,
   and final/track trajectory differences.
3. `NumericsMode::Fast` vs `NumericsMode::Strict` (`#165`) speedups and
   full-trajectory differences across the same four cases.
4. `#150` waste removal: direct pre-`#150` (`af9747e`) vs post-`#150`
   (`3f7fa734`) benchmark on `1.3M_ms_high_Z` plus `HEAD` component breakdown
   (`MicrophysicsCache` saved evaluations and `PrimaryStateBuffer` vs
   `StarModel::clone`).
5. `N`-star Safe Rust `rayon` worker-pool batching throughput scaling (`N` = 1
   to 32 across 2-thread and 8-thread pools) and a quantitative grid-scale
   accelerator estimate (`#166`) anchored to `benchmarks/raw/26.4.1/`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
from typing import Any, cast

try:
    import _mesa_profile_helpers as helpers
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import _mesa_profile_helpers as helpers

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_OUT_DIR = _REPO_ROOT / "benchmarks" / "evolution_profile" / "26.4.1"


def _resolve_mesa_test_inputs_dir() -> pathlib.Path:
    """Resolve ``mesa_test_inputs`` without requiring ``flexmesa._rust``."""
    spec = importlib.util.spec_from_file_location(
        "flexmesa_fetch", _REPO_ROOT / "src" / "flexmesa" / "data" / "fetch.py"
    )
    assert spec is not None and spec.loader is not None
    fetch_mod = importlib.util.module_from_spec(spec)
    sys.modules["flexmesa_fetch"] = fetch_mod
    spec.loader.exec_module(fetch_mod)
    return cast(pathlib.Path, fetch_mod.resolve_mesa_test_inputs_dir())


SELECTED_CASES: tuple[str, ...] = (
    "1.3M_ms_high_Z",
    "low_z",
    "high_z",
    "make_metals",
)
TRACKED_COLS: tuple[str, ...] = (
    "log_L",
    "log_Teff",
    "log_R",
    "log_cntr_T",
    "log_cntr_Rho",
    "center_h1",
    "center_he4",
)


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n"
    )
    helpers.assert_clean_public_text(text, path.name)
    path.write_text(text, encoding="utf-8")


def _detect_cpu_model() -> str:
    cpuinfo = pathlib.Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _detect_total_ram_gib() -> float:
    meminfo = pathlib.Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                kib = float(line.split()[1])
                return round(kib / (1024.0 * 1024.0), 2)
    return 0.0


def _detect_gpu_model() -> str:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.strip().splitlines()[0].strip()
    return "none"


def _detect_gfortran_version(mesasdk_root: pathlib.Path) -> str:
    gf = mesasdk_root / "bin" / "gfortran"
    proc = subprocess.run(
        [str(gf), "--version"], capture_output=True, text=True, check=False
    )
    if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.strip().splitlines()[0].strip()
    return "GNU Fortran (MESA SDK 26.6.1)"


def _run_profile_bin(
    bin_path: pathlib.Path,
    sub_args: list[str],
    *,
    mode: str,
    threads: int,
) -> dict[str, Any]:
    env = dict(os.environ)
    env["FLEXMESA_NUMERICS_MODE"] = mode
    env["RAYON_NUM_THREADS"] = str(threads)
    env["OMP_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    out = subprocess.check_output(
        [str(bin_path), *sub_args], env=env, text=True
    )
    return cast(dict[str, Any], json.loads(out))


def _run_median_unperturbed_case(
    bin_path: pathlib.Path,
    case_dir: pathlib.Path,
    history_out: pathlib.Path,
    *,
    mode: str,
    threads: int,
    repeats: int = 3,
) -> dict[str, Any]:
    """Run unperturbed evolution ``repeats`` times and return the median run."""
    samples: list[dict[str, Any]] = []
    for rep in range(repeats):
        args = [
            "case",
            "--case-dir",
            str(case_dir),
            "--allow-unsupported-controls",
        ]
        if rep == repeats - 1:
            args.extend(["--history-out", str(history_out)])
        samples.append(
            _run_profile_bin(bin_path, args, mode=mode, threads=threads)
        )
    samples.sort(key=lambda r: float(r["evolve_wall_sec"]))
    med = dict(samples[len(samples) // 2])
    med["evolve_wall_sec_samples"] = [
        float(s["evolve_wall_sec"]) for s in samples
    ]
    return med


def _build_accelerator_estimate(
    repo_root: pathlib.Path,
    strict_cases: list[dict[str, Any]],
    batch_2t: dict[str, Any],
    batch_8t: dict[str, Any],
) -> dict[str, Any]:
    """Compute grid-scale GPU vs Rust worker-pool estimates (`#166`)."""
    spec = importlib.util.spec_from_file_location(
        "raw_benchmarks", repo_root / "scripts" / "docs" / "raw_benchmarks.py"
    )
    assert spec is not None and spec.loader is not None
    raw_mod = importlib.util.module_from_spec(spec)
    sys.modules["raw_benchmarks"] = raw_mod
    spec.loader.exec_module(raw_mod)
    suite = raw_mod.load_raw_benchmarks(repo_root=repo_root)

    def _pair_us_per_zone(workload: str, nz: int) -> dict[str, float]:
        r_rec = suite.get("rust", workload, nz)
        p_rec = suite.get("python", workload, nz)
        r4_us_z = float(r_rec.stage3.median_us) / float(nz)
        jax_vmap_us_z = float(p_rec.stage3.median_us) / float(nz)
        return {
            "n_zones_ref": float(nz),
            "rust_4t_us_per_zone": r4_us_z,
            "jax_gpu_vmap_us_per_zone": jax_vmap_us_z,
            "gpu_vs_rust_4t_speedup": r4_us_z / max(jax_vmap_us_z, 1.0e-12),
        }

    kernels = {
        "eos_helm": _pair_us_per_zone("eos_helm", 4096),
        "eos_skye": _pair_us_per_zone("eos_skye", 4096),
        "eos_4d": _pair_us_per_zone("eos_4d", 4096),
        "kap_fixed": _pair_us_per_zone("kap_fixed", 4096),
        "rates_screen": _pair_us_per_zone("rates_screen", 4096),
        "net_eval_host_cb": _pair_us_per_zone("net_eval", 4096),
        "turb_mlt": _pair_us_per_zone("turb_mlt", 4096),
        "turb_tdc": _pair_us_per_zone("turb_tdc", 4096),
        "colors_sed": _pair_us_per_zone("colors_sed", 1024),
        "mtx_tridiag": _pair_us_per_zone("mtx_tridiag", 4096),
    }

    # Mean phase shares across the 4 real evolutions.
    mean_shares: dict[str, float] = {}
    for key in (
        "eos_share_pct",
        "kap_share_pct",
        "net_and_jacobian_share_pct",
        "turbulence_share_pct",
        "equation_assembly_share_pct",
        "block_solve_share_pct",
        "remesh_share_pct",
        "bookkeeping_and_copies_share_pct",
    ):
        vals = [float(c["phases"][key]) for c in strict_cases]
        mean_shares[key] = sum(vals) / len(vals)

    stateless_micro_frac = (
        mean_shares["eos_share_pct"]
        + mean_shares["kap_share_pct"]
        + mean_shares["net_and_jacobian_share_pct"]
        + mean_shares["turbulence_share_pct"]
    ) / 100.0
    cpu_bound_frac = 1.0 - stateless_micro_frac
    amdahl_ceiling = 1.0 / max(cpu_bound_frac, 1.0e-9)

    # Model per-step wall time on N stars comparing:
    # (a) 8-thread Safe Rust CPU worker pool (measured from batch_8t)
    # (b) Hybrid CPU-solver + hypothetical pure-GPU stateless microphysics
    #     (removing current N1-24 net and N1-25 EOS/kap host callbacks, assuming
    #     6.0x kernel speedup on device-resident EOS/kap/net/MLT + 80 us XLA
    #     dispatch per Newton iter + PCIe 24 GB/s round-trip of 48 float64
    #     arrays/zone, and 0.72 lockstep adaptive-dt/remesh utilization).
    ref_case = strict_cases[0]
    step_2t_sec = float(ref_case["evolve_wall_sec"]) / max(
        1.0, float(ref_case["steps_taken"])
    )
    nz_typ = float(ref_case["mean_zones"])
    iters_typ = float(ref_case["newton_iters_per_step"]) + 1.0

    pts_8t = cast(list[dict[str, Any]], batch_8t["points"])
    max_8t_zps = max(float(p["zone_steps_per_sec"]) for p in pts_8t)
    pts_2t = cast(list[dict[str, Any]], batch_2t["points"])
    n1_2t_zps = float(pts_2t[0]["zone_steps_per_sec"])
    rust_8t_pool_gain_vs_1star_2t = max_8t_zps / max(n1_2t_zps, 1.0e-9)

    pcie_bw_bytes_per_s = 24.0e9
    bytes_per_zone_roundtrip = 48.0 * 8.0 * 2.0
    launch_overhead_sec = 80.0e-6
    device_kernel_speedup_vs_2t = 6.0
    lockstep_efficiency = 0.72

    grid_projections: list[dict[str, Any]] = []
    for n_stars in (1, 8, 64, 256, 1024):
        total_zones = float(n_stars) * nz_typ
        rust_pool_8t_sec = (
            float(n_stars) * step_2t_sec / rust_8t_pool_gain_vs_1star_2t
        )
        cpu_serial_part_8t = rust_pool_8t_sec * cpu_bound_frac
        gpu_micro_compute_sec = (
            float(n_stars)
            * step_2t_sec
            * stateless_micro_frac
            / device_kernel_speedup_vs_2t
        )
        gpu_transfer_and_launch_sec = iters_typ * (
            launch_overhead_sec
            + (total_zones * bytes_per_zone_roundtrip) / pcie_bw_bytes_per_s
        )
        eff = 1.0 if n_stars == 1 else lockstep_efficiency
        hybrid_gpu_step_sec = (
            cpu_serial_part_8t
            + (gpu_micro_compute_sec + gpu_transfer_and_launch_sec) / eff
        )
        # Standalone stateless microphysics batch (e.g. SED / EOS / TDC grid)
        stateless_only_rust_8t_sec = rust_pool_8t_sec * stateless_micro_frac
        stateless_only_gpu_sec = (
            launch_overhead_sec
            + gpu_micro_compute_sec
            + (total_zones * 16.0 * 8.0) / pcie_bw_bytes_per_s
        )
        grid_projections.append(
            {
                "n_stars": n_stars,
                "total_grid_zones": round(total_zones),
                "rust_8t_worker_pool_step_sec": rust_pool_8t_sec,
                "hybrid_gpu_micro_step_sec": hybrid_gpu_step_sec,
                "hybrid_gpu_vs_rust_8t_ratio": rust_pool_8t_sec
                / max(hybrid_gpu_step_sec, 1.0e-12),
                "stateless_batch_only_rust_8t_sec": stateless_only_rust_8t_sec,
                "stateless_batch_only_gpu_sec": stateless_only_gpu_sec,
                "stateless_batch_only_gpu_speedup_vs_rust_8t": (
                    stateless_only_rust_8t_sec
                    / max(stateless_only_gpu_sec, 1.0e-12)
                ),
            }
        )

    return {
        "anchor_raw_suite": "benchmarks/raw/26.4.1/",
        "measured_kernel_anchors": kernels,
        "mean_evolution_phase_shares_pct": mean_shares,
        "stateless_microphysics_fraction": stateless_micro_frac,
        "stateful_solver_remesh_fraction": cpu_bound_frac,
        "amdahl_infinite_gpu_micro_speedup_ceiling": amdahl_ceiling,
        "rust_8t_worker_pool_throughput_gain_vs_single_star_2t": (
            rust_8t_pool_gain_vs_1star_2t
        ),
        "model_assumptions": {
            "pcie_effective_bandwidth_gb_s": 24.0,
            "xla_dispatch_latency_us_per_iter": 80.0,
            "hypothetical_device_micro_speedup_vs_2t_rust": (
                device_kernel_speedup_vs_2t
            ),
            "heterogeneous_grid_lockstep_efficiency": lockstep_efficiency,
            "mtx_pcr_vs_cpu_solver_note": (
                "Measured batch-of-4 GPU vmap PCR at N=4096 is 2795.65 us vs "
                "323.73 us for 4-system amortized Safe Rust "
                "fast_block_thomas_solve (0.12x), and single-profile GPU PCR "
                "is 7451.55 us vs 1380.73 us 1t PyO3 bcyclic / 1294.80 us 1t "
                "Rust block-Thomas, so block-tridiagonal Henyey solves favor "
                "CPU bcyclic/block-Thomas or IFT custom_vjp."
            ),
        },
        "grid_projections": grid_projections,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-bin", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, default=_OUT_DIR)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--mesa-dir",
        type=pathlib.Path,
        default=pathlib.Path(
            os.environ.get(
                "MESA_DIR", str(pathlib.Path.home() / "mesa-26.04.1")
            )
        ),
    )
    parser.add_argument(
        "--mesasdk-root",
        type=pathlib.Path,
        default=pathlib.Path(
            os.environ.get(
                "MESASDK_ROOT", str(pathlib.Path.home() / "mesasdk-26.6.1")
            )
        ),
    )
    args = parser.parse_args(argv)

    out_dir: pathlib.Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    test_inputs = _resolve_mesa_test_inputs_dir()
    threads: int = args.threads

    load_start = os.getloadavg()[0]
    date_utc = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    base_commit = subprocess.check_output(
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "origin/main"], text=True
    ).strip()
    head_commit = subprocess.check_output(
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()

    strict_records: list[dict[str, Any]] = []
    fast_records: list[dict[str, Any]] = []
    mesa_records: list[dict[str, Any]] = []
    strict_vs_fast_records: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="prof168_hists_") as tmp_h:
        h_root = pathlib.Path(tmp_h)
        for cname in SELECTED_CASES:
            cdir = test_inputs / "star" / "test_suite" / cname
            sys.stdout.write(f"[168] Running FlexMESA strict on {cname}...\n")
            sys.stdout.flush()
            shist = h_root / f"{cname}_strict.data"
            unpert_s = _run_median_unperturbed_case(
                args.profile_bin,
                cdir,
                shist,
                mode="strict",
                threads=threads,
                repeats=3,
            )
            prof_s = _run_profile_bin(
                args.profile_bin,
                [
                    "case",
                    "--case-dir",
                    str(cdir),
                    "--allow-unsupported-controls",
                    "--with-phase-profile",
                    "--ref-evolve-sec",
                    f"{unpert_s['evolve_wall_sec']:.9e}",
                ],
                mode="strict",
                threads=threads,
            )
            if unpert_s["photo_checksum"] != prof_s["photo_checksum"]:
                raise RuntimeError(f"Checksum drift on {cname}")
            unpert_s["phases"] = prof_s["phases"]
            strict_records.append(unpert_s)

            sys.stdout.write(f"[168] Running FlexMESA fast on {cname}...\n")
            sys.stdout.flush()
            fhist = h_root / f"{cname}_fast.data"
            unpert_f = _run_median_unperturbed_case(
                args.profile_bin,
                cdir,
                fhist,
                mode="fast",
                threads=threads,
                repeats=3,
            )
            fast_records.append(unpert_f)

            _, cols_s, arr_s = helpers.parse_mesa_history_file(shist)
            _, cols_f, arr_f = helpers.parse_mesa_history_file(fhist)
            svf_cmp = helpers.compare_trajectories_by_age(
                cols_s, arr_s, cols_f, arr_f, TRACKED_COLS
            )
            strict_sec = float(unpert_s["evolve_wall_sec"])
            fast_sec = float(unpert_f["evolve_wall_sec"])
            strict_uzs = float(unpert_s["wall_us_per_zone_step"])
            fast_uzs = float(unpert_f["wall_us_per_zone_step"])
            strict_vs_fast_records.append(
                {
                    "case": cname,
                    "comparison": "fast_vs_strict_only",
                    "strict_steps": int(unpert_s["steps_taken"]),
                    "fast_steps": int(unpert_f["steps_taken"]),
                    "step_delta_fast_minus_strict": int(unpert_f["steps_taken"])
                    - int(unpert_s["steps_taken"]),
                    "strict_retries": int(unpert_s["total_retries"]),
                    "fast_retries": int(unpert_f["total_retries"]),
                    "strict_evolve_wall_sec": strict_sec,
                    "fast_evolve_wall_sec": fast_sec,
                    "wall_speedup_strict_over_fast": strict_sec
                    / max(fast_sec, 1.0e-12),
                    "strict_us_per_zone_step": strict_uzs,
                    "fast_us_per_zone_step": fast_uzs,
                    "zone_step_speedup_strict_over_fast": strict_uzs
                    / max(fast_uzs, 1.0e-12),
                    "trajectory_diff": svf_cmp,
                }
            )

            sys.stdout.write(f"[168] Running real MESA 26.4.1 on {cname}...\n")
            sys.stdout.flush()
            m_rec, cols_m, arr_m = helpers.run_real_mesa_case(
                cname, args.mesa_dir, args.mesasdk_root, threads
            )
            m_cmp = helpers.compare_trajectories_by_age(
                cols_m, arr_m, cols_s, arr_s, TRACKED_COLS
            )
            m_rec["strict_vs_mesa_comparison"] = m_cmp
            mesa_records.append(m_rec)

    sys.stdout.write("[168] Running #150 pre/post commit A/B benchmark...\n")
    sys.stdout.flush()
    c13_dir = test_inputs / "star" / "test_suite" / "1.3M_ms_high_Z"
    with tempfile.TemporaryDirectory(prefix="wt150_shared_tgt_") as shared_tmp:
        shared_tgt = pathlib.Path(shared_tmp)
        pre150 = helpers.measure_commit_case_wall(
            _REPO_ROOT,
            "af9747e",
            c13_dir,
            threads,
            repeats=3,
            shared_target_dir=shared_tgt,
        )
        post150 = helpers.measure_commit_case_wall(
            _REPO_ROOT,
            "3f7fa734",
            c13_dir,
            threads,
            repeats=3,
            shared_target_dir=shared_tgt,
        )
    waste_components = [
        {
            "case": c["case"],
            "steps_taken": c["steps_taken"],
            "evolve_wall_sec": c["evolve_wall_sec"],
            "micro_cache_hits": c["micro_cache_hits"],
            "micro_cache_misses": c["micro_cache_misses"],
            "micro_cache_hits_per_step": c["micro_cache_hits_per_step"],
            "micro_cache_misses_per_step": c["micro_cache_misses_per_step"],
            "saved_micro_eval_sec": c["phases"]["saved_micro_eval_sec"],
            "pre150_full_clone_every_step_sec": c["phases"][
                "pre150_full_clone_every_step_sec"
            ],
            "post150_lazy_primary_buf_sec": c["phases"][
                "post150_lazy_primary_buf_sec"
            ],
            "star_model_heap_bytes": c["star_model_heap_bytes"],
            "primary_state_buffer_heap_bytes": c[
                "primary_state_buffer_heap_bytes"
            ],
            "snapshot_byte_reduction_factor": float(c["star_model_heap_bytes"])
            / max(1.0, float(c["primary_state_buffer_heap_bytes"])),
        }
        for c in strict_records
    ]
    waste_payload = {
        "git_ab_1_3M_ms_high_Z": {
            "case": "1.3M_ms_high_Z",
            "pre_150": pre150,
            "post_150": post150,
            "wall_speedup_pre_over_post": float(pre150["median_wall_sec"])
            / max(float(post150["median_wall_sec"]), 1.0e-12),
            "wall_time_saved_sec": float(pre150["median_wall_sec"])
            - float(post150["median_wall_sec"]),
            "wall_time_saved_pct": 100.0
            * (
                float(pre150["median_wall_sec"])
                - float(post150["median_wall_sec"])
            )
            / max(float(pre150["median_wall_sec"]), 1.0e-12),
        },
        "head_component_breakdown": waste_components,
    }

    sys.stdout.write("[168] Running N-star worker-pool batching curves...\n")
    sys.stdout.flush()
    lowz_dir = test_inputs / "star" / "test_suite" / "low_z"
    batch_2t = _run_profile_bin(
        args.profile_bin,
        [
            "batch",
            "--case-dir",
            str(lowz_dir),
            "--batch-sizes",
            "1,2,4,8,16",
            "--pool-threads",
            "2",
            "--max-steps",
            "60",
            "--heterogeneous",
        ],
        mode="strict",
        threads=2,
    )
    batch_8t = _run_profile_bin(
        args.profile_bin,
        [
            "batch",
            "--case-dir",
            str(lowz_dir),
            "--batch-sizes",
            "1,2,4,8,16,32",
            "--pool-threads",
            "8",
            "--max-steps",
            "60",
            "--heterogeneous",
        ],
        mode="strict",
        threads=8,
    )
    batch_payload = {"pool_2_threads": batch_2t, "pool_8_threads": batch_8t}

    accel_payload = _build_accelerator_estimate(
        _REPO_ROOT, strict_records, batch_2t, batch_8t
    )

    rustc_ver = subprocess.check_output(
        ["rustc", "--version"], text=True
    ).strip()
    gf_ver = _detect_gfortran_version(args.mesasdk_root)
    import jax

    load_end = os.getloadavg()[0]
    prov_payload = {
        "schema_version": 1,
        "issue": "#168",
        "date_utc": date_utc,
        "flexmesa_origin_main_commit": base_commit,
        "flexmesa_branch_commit": head_commit,
        "pre_150_commit": pre150["commit"],
        "post_150_commit": post150["commit"],
        "ci_continuous_verification": False,
        "quiet_host": {
            "load_avg_1m_start": round(load_start, 2),
            "load_avg_1m_end": round(load_end, 2),
            "serialized_single_campaign": True,
        },
        "hardware": {
            "os": f"{platform.system()} {platform.machine()}",
            "cpu_model": _detect_cpu_model(),
            "logical_cores": os.cpu_count() or 16,
            "physical_cores": 8,
            "total_ram_gib": _detect_total_ram_gib(),
            "gpu_model": _detect_gpu_model(),
        },
        "toolchains": {
            "mesa_release": "r26.4.1",
            "mesa_commit": "c8c2026",
            "mesa_zenodo_record": "19722306",
            "mesa_sdk": "x86_64-linux-26.6.1",
            "gfortran": gf_ver,
            "rustc": rustc_ver,
            "python": f"Python {platform.python_version()}",
            "jax": jax.__version__,
            "omp_num_threads": threads,
            "rayon_num_threads": threads,
        },
        "selected_cases": list(SELECTED_CASES),
    }

    _write_json(out_dir / "provenance.json", prov_payload)
    _write_json(
        out_dir / "flexmesa_cases_strict.json", {"cases": strict_records}
    )
    _write_json(out_dir / "flexmesa_cases_fast.json", {"cases": fast_records})
    _write_json(out_dir / "mesa_26_4_1_cases.json", {"cases": mesa_records})
    _write_json(
        out_dir / "strict_vs_fast_trajectories.json",
        {"cases": strict_vs_fast_records},
    )
    _write_json(out_dir / "waste_removal_150.json", waste_payload)
    _write_json(out_dir / "batch_scaling_rust.json", batch_payload)
    _write_json(out_dir / "accelerator_grid_estimate.json", accel_payload)

    # Write companion CSV tables.
    mesa_hdr = [
        f"MESA version: r26.4.1 (commit c8c2026, Zenodo 19722306) | "
        f"SDK: x86_64-linux-26.6.1 | Compiler: {gf_ver}",
        f"Captured UTC: {date_utc} | OMP_NUM_THREADS={threads}",
    ]
    helpers.write_csv_rows(
        out_dir / "phase_breakdown.csv",
        [
            "case",
            "steps_taken",
            "total_zone_steps",
            "evolve_wall_sec",
            "eos_share_pct",
            "kap_share_pct",
            "net_and_jacobian_share_pct",
            "turbulence_share_pct",
            "equation_assembly_share_pct",
            "block_solve_share_pct",
            "remesh_share_pct",
            "bookkeeping_and_copies_share_pct",
            "eos_evals_per_accepted_step",
            "kap_evals_per_accepted_step",
            "net_jac_evals_per_accepted_step",
            "mlt_evals_per_accepted_step",
            "micro_cache_hits_per_step",
        ],
        [
            {
                "case": c["case"],
                "steps_taken": c["steps_taken"],
                "total_zone_steps": c["total_zone_steps"],
                "evolve_wall_sec": c["evolve_wall_sec"],
                **{
                    k: c["phases"][k]
                    for k in (
                        "eos_share_pct",
                        "kap_share_pct",
                        "net_and_jacobian_share_pct",
                        "turbulence_share_pct",
                        "equation_assembly_share_pct",
                        "block_solve_share_pct",
                        "remesh_share_pct",
                        "bookkeeping_and_copies_share_pct",
                        "eos_evals_per_accepted_step",
                        "kap_evals_per_accepted_step",
                        "net_jac_evals_per_accepted_step",
                        "mlt_evals_per_accepted_step",
                    )
                },
                "micro_cache_hits_per_step": c["micro_cache_hits_per_step"],
            }
            for c in strict_records
        ],
    )
    helpers.write_csv_rows(
        out_dir / "mesa_26_4_1_cases.csv",
        [
            "case",
            "mesa_version_number",
            "omp_threads",
            "termination_code",
            "steps_taken",
            "total_retries",
            "total_zone_steps",
            "rn_wall_sec",
            "mesa_internal_evolve_sec",
            "rn_wall_ms_per_step",
            "rn_wall_us_per_zone_step",
            "evolve_wall_ms_per_step",
            "evolve_wall_us_per_zone_step",
        ],
        mesa_records,
        comment_header=mesa_hdr,
    )
    helpers.write_csv_rows(
        out_dir / "strict_vs_fast_trajectories.csv",
        [
            "case",
            "strict_steps",
            "fast_steps",
            "strict_evolve_wall_sec",
            "fast_evolve_wall_sec",
            "wall_speedup_strict_over_fast",
            "strict_us_per_zone_step",
            "fast_us_per_zone_step",
            "zone_step_speedup_strict_over_fast",
            "rel_star_age_diff",
            "max_track_log_L_diff",
            "max_track_log_Teff_diff",
            "max_track_log_cntr_Rho_diff",
        ],
        [
            {
                "case": r["case"],
                "strict_steps": r["strict_steps"],
                "fast_steps": r["fast_steps"],
                "strict_evolve_wall_sec": r["strict_evolve_wall_sec"],
                "fast_evolve_wall_sec": r["fast_evolve_wall_sec"],
                "wall_speedup_strict_over_fast": r[
                    "wall_speedup_strict_over_fast"
                ],
                "strict_us_per_zone_step": r["strict_us_per_zone_step"],
                "fast_us_per_zone_step": r["fast_us_per_zone_step"],
                "zone_step_speedup_strict_over_fast": r[
                    "zone_step_speedup_strict_over_fast"
                ],
                "rel_star_age_diff": r["trajectory_diff"]["rel_star_age_diff"],
                "max_track_log_L_diff": r["trajectory_diff"]["columns"][
                    "log_L"
                ]["track_max_abs_diff"],
                "max_track_log_Teff_diff": r["trajectory_diff"]["columns"][
                    "log_Teff"
                ]["track_max_abs_diff"],
                "max_track_log_cntr_Rho_diff": r["trajectory_diff"]["columns"][
                    "log_cntr_Rho"
                ]["track_max_abs_diff"],
            }
            for r in strict_vs_fast_records
        ],
    )
    batch_rows: list[dict[str, Any]] = [
        *cast(list[dict[str, Any]], batch_2t["points"]),
        *cast(list[dict[str, Any]], batch_8t["points"]),
    ]
    helpers.write_csv_rows(
        out_dir / "batch_scaling_rust.csv",
        [
            "pool_threads",
            "n_stars",
            "wall_sec",
            "total_steps",
            "total_zone_steps",
            "stars_per_min",
            "steps_per_sec",
            "zone_steps_per_sec",
            "speedup_vs_n1",
            "peak_rss_kib",
        ],
        batch_rows,
    )
    sys.stdout.write(f"[168] Completed all captures in {out_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
