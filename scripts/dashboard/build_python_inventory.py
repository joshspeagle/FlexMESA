#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate or check the Python and JAX port inventory (`P1.5`).

Maps every public function and class in ``src/flexmesa/<pkg>/`` (excluding
``testing/`` and ``plotting/``) to the Rust item it mirrors across
``crates/flexmesa-*``, classifies each Python symbol by execution kind
(``pure-python mirror``, ``PyO3 wrapper``, ``JAX kernel``,
``JAX engine twin (frozen)``), computes staleness from per-span ``git blame``
committer timestamps (``mirror-current``, ``mirror-stale``, ``python-only``,
``rust-only``), and records every JAX test marked ``xfail`` under ``#295``.

Writes ``notes/audit/python_inventory.json`` and
``notes/audit/python_inventory.md``. With ``--check``, regenerates both in
memory and exits non-zero on any drift, broken explicit override, or summary
arithmetic mismatch. Uses zero wall-clock timestamps so outputs are
deterministic across runs.
"""

from __future__ import annotations

import argparse
import ast
import collections
import dataclasses
import datetime
import json
import pathlib
import re
import subprocess
import sys
from typing import Any

DEFAULT_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
JSON_RELPATH = pathlib.Path("notes") / "audit" / "python_inventory.json"
MD_RELPATH = pathlib.Path("notes") / "audit" / "python_inventory.md"

EXCLUDED_PY_SUBPACKAGES: frozenset[str] = frozenset({"testing", "plotting"})
EXCLUDED_CRATES: frozenset[str] = frozenset(
    {"flexmesa-py", "flexmesa-testing", "flexmesa-crlibm-sys"}
)

CRATE_ONLY_GROUPS: dict[str, str] = {
    "flexmesa-adipls": "adipls",
    "flexmesa-gyre": "gyre",
    "flexmesa-replay": "replay",
}

PKG_TO_CRATES: dict[str, tuple[str, ...]] = {
    "astero": ("flexmesa-astero",),
    "atm": ("flexmesa-atm",),
    "auto_diff": ("flexmesa-autodiff",),
    "bench": ("flexmesa-bench",),
    "binary": ("flexmesa-binary",),
    "chem": ("flexmesa-chem",),
    "colors": ("flexmesa-colors",),
    "constants": ("flexmesa-const",),
    "data": ("flexmesa-data", "flexmesa-utils"),
    "eos": ("flexmesa-eos",),
    "interp_1d": ("flexmesa-interp1d",),
    "interp_2d": ("flexmesa-interp2d",),
    "ionization": ("flexmesa-ionization",),
    "kap": ("flexmesa-kap",),
    "math": ("flexmesa-math", "flexmesa-autodiff"),
    "mtx": ("flexmesa-mtx",),
    "net": ("flexmesa-net",),
    "neu": ("flexmesa-neu",),
    "num": ("flexmesa-num",),
    "rates": ("flexmesa-rates",),
    "schema": ("flexmesa-schema",),
    "star": ("flexmesa-star",),
    "turb": ("flexmesa-turb", "flexmesa-autodiff"),
    "utils": ("flexmesa-utils",),
}

FROZEN_TWIN_FILES: frozenset[str] = frozenset(
    {
        "src/flexmesa/astero/jax_kernels.py",
        "src/flexmesa/binary/jax_jdot_tides.py",
        "src/flexmesa/binary/jax_kernels.py",
        "src/flexmesa/binary/jax_orbit_mdot.py",
        "src/flexmesa/star/jax_diffusion_rot.py",
        "src/flexmesa/star/jax_evolve.py",
        "src/flexmesa/star/jax_hydro_pulse.py",
        "src/flexmesa/star/jax_kernels.py",
        "src/flexmesa/star/jax_rsp.py",
        "src/flexmesa/star/jax_winds_mix.py",
    }
)
FROZEN_TWIN_SYMBOLS: frozenset[tuple[str, str]] = frozenset(
    {("src/flexmesa/star/model.py", "JaxStarState")}
)

# Verified (py_relpath, py_name) -> (rs_relpath, rs_symbol_or_scoped_name).
# Every target is validated at runtime against the parsed Rust index.
EXPLICIT_PY_TO_RUST: dict[tuple[str, str], tuple[str, str]] = {
    (
        "src/flexmesa/astero/jax_kernels.py",
        "eval_ratios_r01_r10_jax",
    ): (
        "crates/flexmesa-astero/src/ratios_chi2.rs",
        "eval_frequency_ratios_r01_r10",
    ),
    ("src/flexmesa/astero/jax_kernels.py", "eval_ratios_r02_jax"): (
        "crates/flexmesa-astero/src/ratios_chi2.rs",
        "eval_frequency_ratios_r02",
    ),
    ("src/flexmesa/bench/bicub_fast.py", "locate_uniform_cell_jax"): (
        "crates/flexmesa-bench/src/bicub_fast.rs",
        "locate_and_basis_1d",
    ),
    ("src/flexmesa/bench/bicub_fast.py", "eval_bicub_point_jax"): (
        "crates/flexmesa-bench/src/bicub_fast.rs",
        "eval_bicub_prevalidated",
    ),
    ("src/flexmesa/bench/tdc_fast.py", "eval_tdc_point_jax"): (
        "crates/flexmesa-bench/src/kernels_solver.rs",
        "eval_tdc_zone",
    ),
    ("src/flexmesa/binary/jax_jdot_tides.py", "eval_bh_spin_and_edd_jax"): (
        "crates/flexmesa-binary/src/edd_wind.rs",
        "eval_point_mass_edd_and_retention",
    ),
    (
        "src/flexmesa/binary/jax_orbit_mdot.py",
        "solve_roche_lobe_mdot_jax",
    ): (
        "crates/flexmesa-binary/src/mdot_implicit.rs",
        "ImplicitRlofSolver::solve_roche_lobe_mdot",
    ),
    ("src/flexmesa/chem/catalog.py", "NuclideInfo"): (
        "crates/flexmesa-chem/src/nuclides.rs",
        "NuclideEntry",
    ),
    ("src/flexmesa/chem/catalog.py", "ag89_namsol"): (
        "crates/flexmesa-chem/src/solar_ag89.rs",
        "chem_get_solar",
    ),
    ("src/flexmesa/chem/composition.py", "CompositionInfo"): (
        "crates/flexmesa-chem/src/composition.rs",
        "CompositionResult",
    ),
    ("src/flexmesa/chem/composition.py", "basic_composition_info"): (
        "crates/flexmesa-chem/src/composition.rs",
        "get_composition_info",
    ),
    ("src/flexmesa/chem/composition.py", "composition_info"): (
        "crates/flexmesa-chem/src/composition.rs",
        "get_composition_info",
    ),
    ("src/flexmesa/colors/photometry.py", "eval_colors_jax"): (
        "crates/flexmesa-colors/src/photometry.rs",
        "SedCube::eval_colors",
    ),
    ("src/flexmesa/colors/sed.py", "eval_sed_hermite_jax"): (
        "crates/flexmesa-colors/src/interp.rs",
        "SedCube::eval_sed_hermite_generic",
    ),
    ("src/flexmesa/colors/sed.py", "eval_sed_linear_jax"): (
        "crates/flexmesa-colors/src/interp.rs",
        "SedCube::eval_sed_trilinear",
    ),
    ("src/flexmesa/colors/sed.py", "eval_sed_knn_jax"): (
        "crates/flexmesa-colors/src/interp.rs",
        "SedCube::eval_sed_knn",
    ),
    ("src/flexmesa/colors/synthetic.py", "clean_sed_fluxes_jax"): (
        "crates/flexmesa-colors/src/photometry.rs",
        "cleaned_flux_at",
    ),
    ("src/flexmesa/colors/utils.py", "ColorsBuiltinGrid"): (
        "crates/flexmesa-colors/src/sed_cube.rs",
        "SedCube",
    ),
    ("src/flexmesa/data/fetch.py", "resolve_cache_dir"): (
        "crates/flexmesa-utils/src/lib.rs",
        "resolve_flexmesa_cache_dir",
    ),
    ("src/flexmesa/data/fetch.py", "resolve_data_dir"): (
        "crates/flexmesa-utils/src/lib.rs",
        "resolve_flexmesa_data_dir",
    ),
    ("src/flexmesa/data/loader.py", "LoadedTableBundle"): (
        "crates/flexmesa-data/src/safetensors.rs",
        "SafeTensorBundle",
    ),
    ("src/flexmesa/data/loader.py", "save_safetensors_bundle"): (
        "crates/flexmesa-data/src/safetensors.rs",
        "save_safetensors_file",
    ),
    ("src/flexmesa/data/loader.py", "load_safetensors_bundle"): (
        "crates/flexmesa-data/src/safetensors.rs",
        "load_safetensors_file",
    ),
    ("src/flexmesa/eos/ideal.py", "pack_free_energy_thermo_jax"): (
        "crates/flexmesa-eos/src/free_energy.rs",
        "pack_for_export",
    ),
    ("src/flexmesa/interp_1d/pm.py", "validate_1d_grid"): (
        "crates/flexmesa-interp1d/src/lib.rs",
        "validate_grid_and_widths",
    ),
    ("src/flexmesa/interp_2d/renka790.py", "cs2hes_db"): (
        "crates/flexmesa-interp2d/src/renka790.rs",
        "Cshep2Interpolant::eval_hess",
    ),
    ("src/flexmesa/interp_2d/spline1d.py", "interp_mkspline_db"): (
        "crates/flexmesa-interp2d/src/spline1d.rs",
        "mkspline_db",
    ),
    ("src/flexmesa/ionization/tables.py", "IonizationResult"): (
        "crates/flexmesa-ionization/src/ion_table.rs",
        "eval_ionization",
    ),
    ("src/flexmesa/kap/jax_co_lowt.py", "eval_co_kap_jax"): (
        "crates/flexmesa-kap/src/co.rs",
        "get1_kap_co_results",
    ),
    ("src/flexmesa/kap/jax_co_lowt.py", "eval_aesopus_jax"): (
        "crates/flexmesa-kap/src/low_t.rs",
        "aesopus_get",
    ),
    ("src/flexmesa/kap/jax_co_lowt.py", "eval_kapcn_jax"): (
        "crates/flexmesa-kap/src/low_t.rs",
        "kapcn_get",
    ),
    ("src/flexmesa/kap/jax_interp.py", "interp_evbicub_jax"): (
        "crates/flexmesa-interp2d/src/bicub_eval.rs",
        "interp_evbicub_prevalidated_db",
    ),
    ("src/flexmesa/kap/jax_interp.py", "interp_pm_1d_jax"): (
        "crates/flexmesa-interp1d/src/pm.rs",
        "interp_pm",
    ),
    ("src/flexmesa/kap/jax_interp.py", "interp_pm_4pt_with_partials_jax"): (
        "crates/flexmesa-interp1d/src/pm.rs",
        "interp_4pt_pm",
    ),
    ("src/flexmesa/kap/jax_interp.py", "eval_fixed_metal_jax"): (
        "crates/flexmesa-kap/src/fixed.rs",
        "get1_kap_fixed_metal_results",
    ),
    (
        "src/flexmesa/kap/jax_kernels.py",
        "electron_conduction_potekhin_jax",
    ): (
        "crates/flexmesa-kap/src/conduction.rs",
        "do_electron_conduction_potekhin",
    ),
    ("src/flexmesa/kap/jax_kernels.py", "electron_conduction_jax"): (
        "crates/flexmesa-kap/src/conduction.rs",
        "do_electron_conduction",
    ),
    ("src/flexmesa/kap/jax_kernels.py", "kap_get_results_jax"): (
        "crates/flexmesa-kap/src/eval.rs",
        "get_kap_results",
    ),
    ("src/flexmesa/kap/tables_co_lowt_jax.py", "get_co_meta_jax"): (
        "crates/flexmesa-kap/src/loader.rs",
        "KapCoSet",
    ),
    ("src/flexmesa/kap/tables_co_lowt_jax.py", "get_co_z_coeffs_jax"): (
        "crates/flexmesa-kap/src/loader.rs",
        "KapCoZTable",
    ),
    ("src/flexmesa/kap/tables_co_lowt_jax.py", "get_co_all_z_coeffs_jax"): (
        "crates/flexmesa-kap/src/loader.rs",
        "KapCoSet",
    ),
    ("src/flexmesa/kap/tables_co_lowt_jax.py", "get_aesopus_meta_jax"): (
        "crates/flexmesa-kap/src/loader.rs",
        "AesopusData",
    ),
    ("src/flexmesa/kap/tables_co_lowt_jax.py", "get_aesopus_z_coeffs_jax"): (
        "crates/flexmesa-kap/src/loader.rs",
        "AesopusZSet",
    ),
    (
        "src/flexmesa/kap/tables_co_lowt_jax.py",
        "get_aesopus_all_z_coeffs_jax",
    ): (
        "crates/flexmesa-kap/src/loader.rs",
        "AesopusData",
    ),
    ("src/flexmesa/kap/tables_co_lowt_jax.py", "get_kapcn_jax"): (
        "crates/flexmesa-kap/src/low_t.rs",
        "kapcn_get",
    ),
    ("src/flexmesa/kap/tables_jax.py", "splinck"): (
        "crates/flexmesa-kap/src/spline.rs",
        "check_grid_uniform",
    ),
    ("src/flexmesa/kap/tables_jax.py", "fit_bicubic_batch"): (
        "crates/flexmesa-kap/src/spline.rs",
        "fit_bicubic_not_a_knot",
    ),
    ("src/flexmesa/kap/tables_jax.py", "get_kap_blend_bounds_jax"): (
        "crates/flexmesa-kap/src/loader.rs",
        "KapTables::temp_blend_bounds",
    ),
    ("src/flexmesa/kap/tables_jax.py", "get_fixed_set_jax"): (
        "crates/flexmesa-kap/src/loader.rs",
        "KapFixedSet",
    ),
    ("src/flexmesa/kap/tables_jax.py", "get_potekhin_jax"): (
        "crates/flexmesa-kap/src/conduction.rs",
        "potekhin_table",
    ),
    ("src/flexmesa/mtx/pcr.py", "solve_block_tridiag_pcr"): (
        "crates/flexmesa-mtx/src/block_tridiag.rs",
        "solve_block_tridiagonal",
    ),
    ("src/flexmesa/net/_core.py", "NetInfo"): (
        "crates/flexmesa-net/src/types.rs",
        "NetGeneralInfo",
    ),
    ("src/flexmesa/net/_core.py", "NetEvalOutput"): (
        "crates/flexmesa-net/src/types.rs",
        "NetEvalResult",
    ),
    ("src/flexmesa/neu/driver.py", "neu_pair_plas_jax"): (
        "crates/flexmesa-neu/src/pair_plas.rs",
        "pair_neu",
    ),
    ("src/flexmesa/num/newton.py", "newton_solve_jax"): (
        "crates/flexmesa-num/src/newton.rs",
        "newton_solve_dense",
    ),
    ("src/flexmesa/num/root.py", "safe_root_jax"): (
        "crates/flexmesa-num/src/root.rs",
        "safe_root_dual",
    ),
    ("src/flexmesa/rates/jax_kernels.py", "eval_reaclib_jax"): (
        "crates/flexmesa-rates/src/raw_rates.rs",
        "eval_reaclib_handle",
    ),
    ("src/flexmesa/schema/defaults.py", "parse_assignment_clause"): (
        "crates/flexmesa-schema/src/lib.rs",
        "split_assignment",
    ),
    ("src/flexmesa/star/jax_evolve.py", "eval_h211_timestep_jax"): (
        "crates/flexmesa-star/src/timestep.rs",
        "eval_h211_factor",
    ),
    (
        "src/flexmesa/star/jax_hydro_pulse.py",
        "eval_dziembowski_variables_jax",
    ): (
        "crates/flexmesa-star/src/pulse_utils.rs",
        "make_pulse_point_on_segment",
    ),
    (
        "src/flexmesa/star/jax_hydro_pulse.py",
        "eval_sobolev_feii_5169_tau_jax",
    ): (
        "crates/flexmesa-star/src/stella.rs",
        "eval_sobolev_feii_5169_snapshot",
    ),
    ("src/flexmesa/star/jax_kernels.py", "compute_staggered_grid_jax"): (
        "crates/flexmesa-star/src/mesh.rs",
        "StarModel::sync_mesh_and_pack",
    ),
    ("src/flexmesa/star/jax_kernels.py", "face_weights_jax"): (
        "crates/flexmesa-star/src/mesh.rs",
        "StarModel::get_face_weights",
    ),
    ("src/flexmesa/star/jax_kernels.py", "interp_cell_to_face_jax"): (
        "crates/flexmesa-star/src/mesh.rs",
        "StarModel::get_face_value",
    ),
    ("src/flexmesa/star/jax_kernels.py", "eval_optical_depth_jax"): (
        "crates/flexmesa-star/src/micro.rs",
        "get_tau",
    ),
    ("src/flexmesa/star/jax_kernels.py", "eval_face_mlt_and_brunt_jax"): (
        "crates/flexmesa-star/src/hydro_vars.rs",
        "set_mlt_vars",
    ),
    ("src/flexmesa/star/jax_kernels.py", "eval_hydro_residuals_jax"): (
        "crates/flexmesa-star/src/hydro_eqns.rs",
        "eval_henyey_equations",
    ),
    ("src/flexmesa/star/jax_kernels.py", "solve_henyey_linear_jax"): (
        "crates/flexmesa-star/src/solver.rs",
        "solve_henyey_step",
    ),
    (
        "src/flexmesa/star/jax_rsp.py",
        "eval_rsp2_turbulent_eqn_residual_jax",
    ): (
        "crates/flexmesa-star/src/rsp_eqns_cell.rs",
        "turbulent_energy_eqn",
    ),
    ("src/flexmesa/star/jax_rsp.py", "eval_rsp_hyd_fluxes_jax"): (
        "crates/flexmesa-star/src/rsp_hyd_flux.rs",
        "calc_lt",
    ),
    ("src/flexmesa/star/jax_winds_mix.py", "eval_convection_sigmas_jax"): (
        "crates/flexmesa-star/src/mix_info.rs",
        "compute_convection_sigmas",
    ),
    ("src/flexmesa/star/jax_winds_mix.py", "eval_winds_jax"): (
        "crates/flexmesa-star/src/winds_driver.rs",
        "do_set_mdot",
    ),
    ("src/flexmesa/turb/types.py", "val_of"): (
        "crates/flexmesa-turb/src/lib.rs",
        "TurbScalar::val",
    ),
    ("src/flexmesa/turb/types.py", "const_like"): (
        "crates/flexmesa-turb/src/lib.rs",
        "TurbScalar::from_f64",
    ),
    ("src/flexmesa/turb/types.py", "with_val_like"): (
        "crates/flexmesa-turb/src/lib.rs",
        "TurbScalar::with_val",
    ),
    ("src/flexmesa/turb/types.py", "s_abs"): (
        "crates/flexmesa-turb/src/lib.rs",
        "TurbScalar::abs",
    ),
    ("src/flexmesa/turb/types.py", "s_sqrt"): (
        "crates/flexmesa-turb/src/lib.rs",
        "TurbScalar::sqrt",
    ),
    ("src/flexmesa/turb/types.py", "s_exp"): (
        "crates/flexmesa-turb/src/lib.rs",
        "TurbScalar::exp",
    ),
    ("src/flexmesa/turb/types.py", "s_log"): (
        "crates/flexmesa-turb/src/lib.rs",
        "TurbScalar::log",
    ),
    ("src/flexmesa/turb/types.py", "s_powf"): (
        "crates/flexmesa-turb/src/lib.rs",
        "TurbScalar::powf",
    ),
    ("src/flexmesa/turb/types.py", "s_pow2"): (
        "crates/flexmesa-turb/src/lib.rs",
        "TurbScalar::pow2",
    ),
    ("src/flexmesa/turb/types.py", "s_pow3"): (
        "crates/flexmesa-turb/src/lib.rs",
        "TurbScalar::pow3",
    ),
    ("src/flexmesa/turb/types.py", "s_pow4"): (
        "crates/flexmesa-turb/src/lib.rs",
        "TurbScalar::pow4",
    ),
    ("src/flexmesa/turb/types.py", "s_tanh"): (
        "crates/flexmesa-turb/src/lib.rs",
        "TurbScalar::tanh",
    ),
    ("src/flexmesa/turb/types.py", "s_tan"): (
        "crates/flexmesa-turb/src/lib.rs",
        "TurbScalar::tan",
    ),
    ("src/flexmesa/turb/types.py", "s_atan"): (
        "crates/flexmesa-turb/src/lib.rs",
        "TurbScalar::atan",
    ),
    ("src/flexmesa/utils/__init__.py", "TokenKind"): (
        "crates/flexmesa-utils/src/lib.rs",
        "Token",
    ),
}

# Verified Python-only helpers/utilities with no Rust counterpart.
EXPLICIT_PYTHON_ONLY: frozenset[tuple[str, str]] = frozenset(
    {
        ("src/flexmesa/auto_diff/base.py", "AutoDiffBase"),
        ("src/flexmesa/bench/harness.py", "measure_and_capture"),
        ("src/flexmesa/bench/harness.py", "measure_stage3_batch"),
        ("src/flexmesa/data/converters.py", "convert_helm_table_file"),
        ("src/flexmesa/data/converters.py", "convert_isotopes_file"),
        ("src/flexmesa/data/converters.py", "convert_lodders03_file"),
        ("src/flexmesa/data/fetch.py", "EmbeddedIonizationAsset"),
        ("src/flexmesa/data/fetch.py", "ReleaseBundle"),
        ("src/flexmesa/data/fetch.py", "UpstreamReference"),
        ("src/flexmesa/data/fetch.py", "build_parser"),
        ("src/flexmesa/data/fetch.py", "clean_runtime_cache"),
        ("src/flexmesa/data/fetch.py", "fetch_bundles"),
        ("src/flexmesa/data/fetch.py", "is_bundle_ready"),
        ("src/flexmesa/data/fetch.py", "load_bundles_manifest"),
        ("src/flexmesa/data/fetch.py", "load_embedded_ionization_asset"),
        ("src/flexmesa/data/fetch.py", "load_upstream_reference"),
        ("src/flexmesa/data/fetch.py", "main"),
        ("src/flexmesa/data/fetch.py", "resolve_mesa_astero_tracks_dir"),
        ("src/flexmesa/data/fetch.py", "resolve_mesa_histories_dir"),
        ("src/flexmesa/data/fetch.py", "resolve_mesa_module_hex_dir"),
        ("src/flexmesa/data/fetch.py", "select_bundles"),
        ("src/flexmesa/data/fetch.py", "verify_bundles"),
        ("src/flexmesa/eos/core.py", "nuclide_arrays_jax"),
        ("src/flexmesa/eos/core.py", "pack_dt_jvp"),
        ("src/flexmesa/eos/skye.py", "eval_host_dt_jvp"),
        ("src/flexmesa/interp_2d/spline1d.py", "reshape_grid_2d"),
        ("src/flexmesa/kap/jax_kernels.py", "make_differentiable_kap"),
        ("src/flexmesa/kap/tables_jax.py", "ensure_indexed"),
        ("src/flexmesa/kap/tables_jax.py", "reset_index_cache"),
        ("src/flexmesa/math/intrinsics.py", "as_f64"),
        ("src/flexmesa/mtx/pcr.py", "validate_jax_blocks"),
    }
)

# Supplemental symbol links for indirect #295 xfail tests (fixtures/kwargs).
XFAIL_EXTRA_TARGETS: dict[tuple[str, str], tuple[str, ...]] = {
    ("tests/test_bench.py", "test_wave2_ion_fe56_stages_agree"): (
        "eval_charge_of_fe56_in_he4",
    ),
    ("tests/test_bench.py", "test_wave3a_eos_pc_stages_agree"): (
        "eval_pc_eos",
        "eval_pc_eos_jax",
    ),
    ("tests/test_bench.py", "test_wave3b_atm_table_stages_agree"): (
        "eval_table_atm",
        "eval_table_jax",
    ),
    (
        "tests/test_eos_wave3b1.py",
        "test_helm_ideal_edge_skip_blend_rust_and_jax_agree",
    ): (
        "eos_dt_get",
        "eos_dt_get_jax",
    ),
    (
        "tests/test_star_wave4_1.py",
        "test_star_jax_face_gradr_and_gradt_match_rust",
    ): ("eval_face_mlt_and_brunt_jax",),
    (
        "tests/test_wave4_bench.py",
        "test_wave4_star_micro_vars_rust_matches_frozen_jax",
    ): ("eval_face_mlt_and_brunt_jax",),
    (
        "tests/test_wave4_bench.py",
        "test_wave4_star_hydro_eqns_rust_matches_frozen_jax",
    ): ("eval_hydro_residuals_jax",),
    (
        "tests/test_wave5_bench.py",
        "test_wave5_diff_rot_rust_matches_frozen_jax",
    ): ("eval_caplan_ocp_dstar_jax",),
    (
        "tests/unit/test_kap_wave3b2.py",
        "test_rust_strict_vs_jax_backend_consistency_across_all_regimes",
    ): (
        "kap_get_results",
        "kap_get_results_jax",
    ),
    (
        "tests/unit/test_kap_wave3b2_jax.py",
        "test_kapcn_knot_and_multi_zone_batch_consistency",
    ): (
        "kap_get_kapcn",
        "kap_get_zones",
    ),
}


@dataclasses.dataclass(frozen=True)
class PySymbol:
    """One public Python function or class in ``src/flexmesa/<pkg>/``."""

    subpackage: str
    rel_path: str
    start_line: int
    end_line: int
    py_type: str
    name: str
    kind: str
    rust_calls: tuple[str, ...]
    docstring: str


@dataclasses.dataclass(frozen=True)
class RustItem:
    """One indexed Rust function, struct, enum, trait, type, or method."""

    crate: str
    rel_path: str
    start_line: int
    end_line: int
    visibility: str
    item_kind: str
    name: str
    scope: str | None
    externally_public: bool

    @property
    def qualified(self) -> str:
        """Return scoped ``Type::method`` or bare identifier."""
        return f"{self.scope}::{self.name}" if self.scope else self.name

    @property
    def path_line(self) -> str:
        """Return ``crates/.../file.rs:line`` string."""
        return f"{self.rel_path}:{self.start_line}"


@dataclasses.dataclass(frozen=True)
class XfailEntry:
    """One test marked ``xfail`` under ``#295``."""

    test_path: str
    line: int
    test_name: str
    subpackage: str
    reason: str
    targets: tuple[str, ...]

    @property
    def ref(self) -> str:
        """Return ``tests/...:line`` reference."""
        return f"{self.test_path}:{self.line}"


def _norm(name: str) -> str:
    """Normalize Python/Rust identifiers for candidate matching."""
    low = name.lower()
    for suffix in ("_jax", "_rust", "_f64"):
        if low.endswith(suffix):
            low = low[: -len(suffix)]
    return low.replace("_", "")


def _is_overload(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> bool:
    """Return ``True`` when a definition carries ``@overload``."""
    return any(
        (isinstance(dec, ast.Name) and dec.id == "overload")
        or (isinstance(dec, ast.Attribute) and dec.attr == "overload")
        for dec in node.decorator_list
    )


def _classify_python_kind(rel_path: str, name: str, segment: str) -> str:
    """Classify a Python symbol into the four required P1.5 kinds."""
    has_rust = (
        "_rust." in segment
        or "require_rust" in segment
        or "get_rust" in segment
    )
    has_jax = bool(re.search(r"\b(?:jnp|jax|lax)\.", segment)) or name.endswith(
        "_jax"
    )
    if has_rust and not has_jax:
        return "PyO3 wrapper"
    if rel_path in FROZEN_TWIN_FILES or (rel_path, name) in FROZEN_TWIN_SYMBOLS:
        return "JAX engine twin (frozen)"
    fname = pathlib.Path(rel_path).name
    if (
        fname.startswith("jax_")
        or fname.endswith("_jax.py")
        or name.endswith("_jax")
    ):
        return "JAX kernel"
    if has_rust:
        return "PyO3 wrapper"
    if has_jax:
        return "JAX kernel"
    return "pure-python mirror"


def _parse_module_defs(
    repo_root: pathlib.Path, py_path: pathlib.Path, subpackage: str
) -> dict[str, PySymbol]:
    """Parse top-level non-overload function/class definitions in a file."""
    text = py_path.read_text(encoding="utf-8")
    rel = py_path.relative_to(repo_root).as_posix()
    tree = ast.parse(text)
    defs: dict[str, PySymbol] = {}
    for node in tree.body:
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        if _is_overload(node):
            continue
        seg = ast.get_source_segment(text, node) or ""
        doc = ast.get_docstring(node) or ""
        py_calls = tuple(
            dict.fromkeys(re.findall(r"_rust\.([A-Za-z_][A-Za-z0-9_]*)", seg))
        )
        defs[node.name] = PySymbol(
            subpackage=subpackage,
            rel_path=rel,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            py_type=("class" if isinstance(node, ast.ClassDef) else "function"),
            name=node.name,
            kind=_classify_python_kind(rel, node.name, seg),
            rust_calls=py_calls,
            docstring=doc,
        )
    return defs


def _extract_init_reexports(
    pkg_dir: pathlib.Path, mod_defs: dict[str, dict[str, PySymbol]]
) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Resolve explicit ``__init__.py`` aliases and ``__all__`` exports."""
    init_path = pkg_dir / "__init__.py"
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    alias_map: dict[str, tuple[str, str]] = {}
    mod_aliases: dict[str, str] = {}
    pkg_name = pkg_dir.name

    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            parts = node.module.split(".") if node.module else []
            if (node.level == 1 and len(parts) == 1) or (
                node.level == 0
                and parts[:2] == ["flexmesa", pkg_name]
                and len(parts) == 3
            ):
                target_mod = parts[-1]
                if target_mod in mod_defs:
                    for al in node.names:
                        alias_map[al.asname or al.name] = (target_mod, al.name)
            elif (node.level == 1 and node.module is None) or (
                node.level == 0 and parts == ["flexmesa", pkg_name]
            ):
                for al in node.names:
                    if al.name in mod_defs:
                        mod_aliases[al.asname or al.name] = al.name
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (
                    isinstance(tgt, ast.Name)
                    and isinstance(node.value, ast.Attribute)
                    and isinstance(node.value.value, ast.Name)
                ):
                    mod_stem = mod_aliases.get(node.value.value.id)
                    if mod_stem:
                        alias_map[tgt.id] = (mod_stem, node.value.attr)

    dunder_all: list[str] = []
    for node in tree.body:
        val: ast.expr | None = None
        if isinstance(node, ast.Assign):
            if any(
                isinstance(t, ast.Name) and t.id == "__all__"
                for t in node.targets
            ):
                val = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
        ):
            val = node.value
        if isinstance(val, (ast.List, ast.Tuple)):
            dunder_all = [
                str(elt.value)
                for elt in val.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            ]
    return alias_map, dunder_all


def collect_python_symbols(repo_root: pathlib.Path) -> list[PySymbol]:
    """Collect public Python functions/classes in ``src/flexmesa/<pkg>``."""
    src_root = repo_root / "src" / "flexmesa"
    collected: dict[tuple[str, str], PySymbol] = {}

    for sub in sorted(src_root.iterdir()):
        if (
            not sub.is_dir()
            or sub.name.startswith("_")
            or sub.name in EXCLUDED_PY_SUBPACKAGES
        ):
            continue
        mod_defs = {
            p.stem: _parse_module_defs(repo_root, p, sub.name)
            for p in sorted(sub.glob("*.py"))
            if p.name != "__main__.py"
        }
        alias_map, dunder_all = _extract_init_reexports(sub, mod_defs)
        init_defs = mod_defs.get("__init__", {})

        for name in dunder_all:
            if name in init_defs:
                sym = init_defs[name]
                collected[(sym.rel_path, sym.name)] = sym
            elif name in alias_map:
                mod_stem, orig_name = alias_map[name]
                sym_opt = mod_defs.get(mod_stem, {}).get(orig_name)
                if sym_opt is not None:
                    collected[(sym_opt.rel_path, sym_opt.name)] = sym_opt

        for stem, defs in sorted(mod_defs.items()):
            if stem.startswith("_") and stem != "__init__":
                continue
            for sym in defs.values():
                if not sym.name.startswith("_"):
                    collected[(sym.rel_path, sym.name)] = sym

    return sorted(
        collected.values(),
        key=lambda s: (s.subpackage, s.rel_path, s.start_line, s.name),
    )


def _strip_cfg_test(text: str) -> str:
    """Truncate trailing ``#[cfg(test)]`` unit-test block from Rust source."""
    idx = text.find("\n#[cfg(test)]")
    return text[:idx] if idx != -1 else text


def _strip_line_comments(text: str) -> str:
    """Replace Rust ``//...`` comments with spaces preserving newlines."""
    return re.sub(r"//[^\n]*", "", text)


def _expand_braced_use(expr: str) -> list[str]:
    """Expand nested Rust ``use a::{b, c::{d, e}}`` paths recursively."""
    clean = " ".join(expr.split())
    if "{" not in clean:
        return [clean.strip()] if clean.strip() else []
    lbrace = clean.index("{")
    depth = 0
    rbrace = -1
    for idx in range(lbrace, len(clean)):
        if clean[idx] == "{":
            depth += 1
        elif clean[idx] == "}":
            depth -= 1
            if depth == 0:
                rbrace = idx
                break
    prefix = clean[:lbrace].strip().rstrip(":")
    inner = clean[lbrace + 1 : rbrace]
    parts: list[str] = []
    cur: list[str] = []
    bdepth = 0
    for ch in inner:
        if ch == "{":
            bdepth += 1
        elif ch == "}":
            bdepth -= 1
        if ch == "," and bdepth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))

    expanded: list[str] = []
    for part in parts:
        p = part.strip()
        if not p:
            continue
        if p == "self":
            if prefix:
                expanded.append(prefix)
            continue
        for sub in _expand_braced_use(p):
            if sub == "self":
                if prefix:
                    expanded.append(prefix)
            elif prefix:
                expanded.append(f"{prefix}::{sub}")
            else:
                expanded.append(sub)
    return expanded


def _resolve_mod_file(
    base_dir: pathlib.Path, mod_name: str
) -> tuple[pathlib.Path, pathlib.Path] | tuple[None, None]:
    """Resolve Rust ``mod foo;`` to ``foo.rs`` or ``foo/mod.rs``."""
    f1 = base_dir / f"{mod_name}.rs"
    if f1.is_file():
        return f1, base_dir
    f2 = base_dir / mod_name / "mod.rs"
    if f2.is_file():
        return f2, base_dir / mod_name
    return None, None


def _compute_rust_spans(
    lines: list[str],
) -> list[tuple[int, int, str, str, str, str | None]]:
    """Return ``(start_line, end_line, vis, kind, name, scope)`` from Rust."""
    results: list[tuple[int, int, str, str, str, str | None]] = []
    top_re = re.compile(
        r"^(pub(?:\([^)]+\))?\s+)?(?:unsafe\s+)?(?:const\s+)?(?:async\s+)?"
        r"(fn|struct|enum|trait|type)\s+([A-Za-z_][A-Za-z0-9_]*)\b"
    )
    scope_re = re.compile(
        r"^(?:pub(?:\([^)]+\))?\s+)?(?:unsafe\s+)?"
        r"(trait|impl(?:<[^>]+>)?)\s+"
    )
    meth_re = re.compile(
        r"^\s{1,8}(pub(?:\([^)]+\))?\s+)?(?:unsafe\s+)?(?:const\s+)?"
        r"(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\b"
    )

    def find_block_end(start_idx: int) -> int:
        depth = 0
        seen_open = False
        for k in range(start_idx, len(lines)):
            line_nc = lines[k].split("//", 1)[0]
            for ch in line_nc:
                if ch == "{":
                    depth += 1
                    seen_open = True
                elif ch == "}":
                    depth -= 1
                    if seen_open and depth <= 0:
                        return k + 1
            if not seen_open and ";" in line_nc:
                return k + 1
        return len(lines)

    cur_scope: str | None = None
    cur_scope_end = -1
    for i, raw in enumerate(lines):
        if i + 1 > cur_scope_end:
            cur_scope = None

        tm = top_re.match(raw)
        if tm:
            vis = (tm.group(1) or "private").strip()
            kind, name = tm.group(2), tm.group(3)
            end_l = find_block_end(i)
            results.append((i + 1, end_l, vis, kind, name, None))
            if kind == "trait":
                cur_scope = name
                cur_scope_end = end_l
            continue

        sm = scope_re.match(raw)
        if sm and " for " not in raw:
            rest = raw[sm.end() :]
            type_m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", rest)
            if type_m:
                cur_scope = type_m.group(1)
                cur_scope_end = find_block_end(i)
                continue

        if cur_scope is not None:
            mm = meth_re.match(raw)
            if mm:
                vis = (mm.group(1) or "private").strip()
                mname = mm.group(2)
                end_l = find_block_end(i)
                results.append((i + 1, end_l, vis, "fn", mname, cur_scope))

    return results


def _walk_crate_pub(
    repo_root: pathlib.Path,
    crate_dir: pathlib.Path,
    lib_rs: pathlib.Path,
    rs_path: pathlib.Path,
    mod_dir: pathlib.Path,
    visited: set[str],
    ext_pub_keys: set[tuple[str, str]],
    only_names: set[str] | None = None,
) -> None:
    """Walk externally-reachable ``pub`` items from a crate root."""
    rel = rs_path.relative_to(repo_root).as_posix()
    text = _strip_cfg_test(rs_path.read_text(encoding="utf-8"))
    for _s, _e, vis, _k, nm, scope in _compute_rust_spans(text.splitlines()):
        if scope is None and vis == "pub":
            if only_names is None or nm in only_names or "*" in only_names:
                ext_pub_keys.add((rel, nm))
    if only_names is not None or rel in visited:
        return
    visited.add(rel)
    for mod_m in re.finditer(
        r"^pub\s+mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;",
        text,
        flags=re.MULTILINE,
    ):
        mod_name = mod_m.group(1)
        if (
            mod_name == "tests"
            or mod_name.startswith("tests_")
            or mod_name.endswith("_tests")
        ):
            continue
        sub_f, sub_d = _resolve_mod_file(mod_dir, mod_name)
        if sub_f is not None and sub_d is not None:
            _walk_crate_pub(
                repo_root,
                crate_dir,
                lib_rs,
                sub_f,
                sub_d,
                visited,
                ext_pub_keys,
            )
    if rs_path == lib_rs:
        clean = _strip_line_comments(text)
        for pm in re.finditer(
            r"^pub\s+use\s+(?:crate::|self::)?([^;]+);",
            clean,
            flags=re.MULTILINE | re.DOTALL,
        ):
            for path_str in _expand_braced_use(pm.group(1)):
                cleaned = path_str.split(" as ", 1)[0].strip()
                if "::" not in cleaned:
                    continue
                submod, sym = cleaned.rsplit("::", 1)
                cur_d: pathlib.Path | None = crate_dir / "src"
                target_f: pathlib.Path | None = None
                for part in submod.split("::"):
                    if cur_d is None:
                        break
                    target_f, cur_d = _resolve_mod_file(cur_d, part)
                    if target_f is None:
                        break
                if target_f is not None and cur_d is not None:
                    _walk_crate_pub(
                        repo_root,
                        crate_dir,
                        lib_rs,
                        target_f,
                        cur_d,
                        visited,
                        ext_pub_keys,
                        only_names={sym},
                    )


def collect_rust_items(repo_root: pathlib.Path) -> list[RustItem]:
    """Index all non-test Rust items and track crate-public reachability."""
    crates_dir = repo_root / "crates"
    all_items: dict[tuple[str, int, str], RustItem] = {}
    ext_pub_keys: set[tuple[str, str]] = set()

    for crate_dir in sorted(crates_dir.iterdir()):
        crate = crate_dir.name
        lib_rs = crate_dir / "src" / "lib.rs"
        if not lib_rs.is_file() or crate in EXCLUDED_CRATES:
            continue

        # Pass 1: find externally-reachable col-0 `pub` items.
        visited: set[str] = set()
        _walk_crate_pub(
            repo_root,
            crate_dir,
            lib_rs,
            lib_rs,
            crate_dir / "src",
            visited,
            ext_pub_keys,
        )

        # Pass 2: index all non-test `.rs` files for Python mirror resolution.
        for rs_path in sorted((crate_dir / "src").rglob("*.rs")):
            rel = rs_path.relative_to(repo_root).as_posix()
            if (
                "/bin/" in rel
                or rs_path.name.startswith("tests")
                or "test" in rs_path.stem.split("_")
            ):
                continue
            lines = _strip_cfg_test(
                rs_path.read_text(encoding="utf-8")
            ).splitlines()
            for sl, el, vis, ikind, nm, scope in _compute_rust_spans(lines):
                is_ext = scope is None and (rel, nm) in ext_pub_keys
                item = RustItem(
                    crate=crate,
                    rel_path=rel,
                    start_line=sl,
                    end_line=el,
                    visibility=vis,
                    item_kind=ikind,
                    name=nm,
                    scope=scope,
                    externally_public=is_ext,
                )
                all_items[(rel, sl, item.qualified)] = item

    return sorted(
        all_items.values(),
        key=lambda it: (it.crate, it.rel_path, it.start_line, it.qualified),
    )


def map_pyo3_to_rust(
    repo_root: pathlib.Path,
    rust_by_crate_name: dict[tuple[str, str], list[RustItem]],
) -> tuple[dict[str, list[RustItem]], set[tuple[str, int, str]]]:
    """Map ``_rust.<sym>`` PyO3 bindings to target ``crates/`` items."""
    py_src = repo_root / "crates" / "flexmesa-py" / "src"
    pyo3_targets: dict[str, list[RustItem]] = {}
    pyo3_accessed: set[tuple[str, int, str]] = set()

    for rs in sorted(py_src.glob("*.rs")):
        if rs.name == "lib.rs":
            continue
        text = rs.read_text(encoding="utf-8")
        clean = _strip_line_comments(text)
        imported: dict[str, tuple[str, str]] = {}
        for um in re.finditer(
            r"(?:pub\s+)?use\s+(flexmesa_[a-z0-9_]+)::([^;]+);",
            clean,
            flags=re.DOTALL,
        ):
            cr = um.group(1).replace("_", "-")
            for path_str in _expand_braced_use(um.group(2)):
                if " as " in path_str:
                    orig_p, al = [x.strip() for x in path_str.split(" as ", 1)]
                else:
                    orig_p = path_str.strip()
                    al = orig_p.rsplit("::", 1)[-1]
                sym = orig_p.rsplit("::", 1)[-1]
                if sym and sym != "*":
                    imported[al] = (cr, sym)

        lines = text.splitlines()
        local_helpers: dict[str, list[RustItem]] = {}
        for idx, line in enumerate(lines):
            hm = re.match(
                r"^(?:pub(?:\([^)]*\))?\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)",
                line,
            )
            if hm and not any(
                "#[pyfunction" in lines[k] for k in range(max(0, idx - 6), idx)
            ):
                j = idx + 1
                while j < len(lines) and not (
                    lines[j].startswith("#[py")
                    or re.match(
                        r"^(?:pub(?:\([^)]*\))?\s+)?(?:fn|struct|enum|impl)\b",
                        lines[j],
                    )
                ):
                    j += 1
                hbody = "\n".join(lines[idx:j])
                hh: list[RustItem] = []
                for tok in set(
                    re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", hbody)
                ):
                    if tok in imported:
                        hh.extend(rust_by_crate_name.get(imported[tok], []))
                local_helpers[hm.group(1)] = hh

        # Any symbol imported directly from a non-excluded crate is accessed.
        for cr, sym in imported.values():
            for it in rust_by_crate_name.get((cr, sym), []):
                pyo3_accessed.add((it.rel_path, it.start_line, it.qualified))

        for i, line in enumerate(lines):
            if "#[pyfunction" not in line and "#[pyclass" not in line:
                continue
            def_idx: int | None = None
            py_sym: str | None = None
            is_class = False
            for k in range(i, min(len(lines), i + 45)):
                nm = re.search(r'name\s*=\s*"([^"]+)"', lines[k])
                if nm:
                    py_sym = nm.group(1)
                dm = re.match(
                    r"^\s*(?:pub\s+)?(?:unsafe\s+)?(?:fn|struct|enum)\s+"
                    r"([A-Za-z_][A-Za-z0-9_]*)",
                    lines[k],
                )
                if dm:
                    def_idx = k
                    if py_sym is None:
                        py_sym = dm.group(1)
                    is_class = "struct " in lines[k] or "enum " in lines[k]
                    break
            if def_idx is None or py_sym is None:
                continue
            j = def_idx + 1
            while j < len(lines) and j < def_idx + 360:
                if (
                    lines[j].startswith("#[py")
                    and "#[pymethods" not in lines[j]
                ):
                    break
                if not is_class and re.match(
                    r"^(?:pub(?:\([^)]*\))?\s+)?(?:unsafe\s+)?"
                    r"(?:fn|struct|enum|impl)\b",
                    lines[j],
                ):
                    break
                j += 1
            body = "\n".join(lines[i:j])
            hits: list[RustItem] = []
            for cm in re.finditer(
                r"(flexmesa_[a-z0-9_]+)::(?:[A-Za-z0-9_]+::)*"
                r"([A-Za-z_][A-Za-z0-9_]*)",
                body,
            ):
                cr = cm.group(1).replace("_", "-")
                hits.extend(rust_by_crate_name.get((cr, cm.group(2)), []))
            for tok in set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", body)):
                if tok in imported:
                    hits.extend(rust_by_crate_name.get(imported[tok], []))
                elif tok in local_helpers:
                    hits.extend(local_helpers[tok])
            uniq: dict[tuple[str, int, str], RustItem] = {}
            for h in hits:
                uniq[(h.rel_path, h.start_line, h.qualified)] = h
                pyo3_accessed.add((h.rel_path, h.start_line, h.qualified))
            pyo3_targets[py_sym] = sorted(
                uniq.values(),
                key=lambda it: (it.crate, it.rel_path, it.start_line),
            )

    return pyo3_targets, pyo3_accessed


def _match_python_to_rust(
    py_sym: PySymbol,
    crates: tuple[str, ...],
    rust_by_rel_qual: dict[tuple[str, str], RustItem],
    rust_by_crate_name: dict[tuple[str, str], list[RustItem]],
    rust_by_crate_norm: dict[tuple[str, str], list[RustItem]],
    pyo3_targets: dict[str, list[RustItem]],
) -> tuple[RustItem | None, str]:
    """Pair one Python symbol with its corresponding Rust item."""
    key = (py_sym.rel_path, py_sym.name)
    if key in EXPLICIT_PYTHON_ONLY:
        return None, "explicit-python-only"
    if key in EXPLICIT_PY_TO_RUST:
        rs_rel, rs_qual = EXPLICIT_PY_TO_RUST[key]
        found = rust_by_rel_qual.get((rs_rel, rs_qual))
        if found is None:
            raise KeyError(
                f"Dangling EXPLICIT_PY_TO_RUST target for {key}: "
                f"{rs_rel}::{rs_qual}"
            )
        return found, "explicit"

    # 1. PyO3 call-graph match via _rust.<fn>
    for rcall in py_sym.rust_calls:
        cands = pyo3_targets.get(rcall, [])
        if cands:
            in_crate = [c for c in cands if c.crate in crates]
            pool = in_crate or cands
            norm_p = _norm(py_sym.name)
            same = [c for c in pool if _norm(c.name) == norm_p]
            chosen = _pick_best_candidate(same or pool)
            return chosen, "pyo3-call"

    # 2. Explicit docstring citation `flexmesa_<crate>::...::sym`
    for dm in re.finditer(
        r"flexmesa_[a-z0-9_]+::([A-Za-z0-9_:]+)", py_sym.docstring
    ):
        sym = dm.group(1).rsplit("::", 1)[-1]
        for cr in crates:
            cands = rust_by_crate_name.get((cr, sym), [])
            if cands:
                return _pick_best_candidate(cands), "docstring"

    # 3. Normalized exact identifier match within mapped crate(s)
    norm_p = _norm(py_sym.name)
    for cr in crates:
        cands = rust_by_crate_norm.get((cr, norm_p), [])
        if cands:
            return _pick_best_candidate(cands), "name-match"

    return None, "unmatched"


def _pick_best_candidate(cands: list[RustItem]) -> RustItem:
    """Prefer top-level public items over private or scoped methods."""
    scored = sorted(
        cands,
        key=lambda it: (
            0 if it.externally_public else (1 if it.visibility == "pub" else 2),
            0 if it.scope is None else 1,
            it.rel_path,
            it.start_line,
        ),
    )
    return scored[0]


def _git_blame_file(
    repo_root: pathlib.Path, rel_path: str
) -> dict[int, tuple[int, str]]:
    """Return ``{line_no: (committer_time, commit_sha)}`` via ``git blame``."""
    out = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo_root),
            "blame",
            "--line-porcelain",
            "--",
            rel_path,
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    line_meta: dict[int, tuple[int, str]] = {}
    commit_times: dict[str, int] = {}
    cur_sha = ""
    cur_line = 0
    for raw in out.splitlines():
        if raw.startswith("\t"):
            ts = commit_times.get(cur_sha, 0)
            line_meta[cur_line] = (ts, cur_sha)
            continue
        parts = raw.split()
        if (
            len(parts) >= 3
            and len(parts[0]) == 40
            and re.fullmatch(r"[0-9a-f]{40}", parts[0])
        ):
            cur_sha = parts[0]
            cur_line = int(parts[2])
        elif raw.startswith("committer-time "):
            commit_times[cur_sha] = int(raw.split()[1])
    return line_meta


def _span_last_commit(
    blame_map: dict[int, tuple[int, str]], start_line: int, end_line: int
) -> tuple[int, str, str]:
    """Return ``(max_ts, sha, iso_utc_date)`` across span lines."""
    best_ts = 0
    best_sha = ""
    for ln in range(start_line, end_line + 1):
        ts, sha = blame_map.get(ln, (0, ""))
        if (ts, sha) > (best_ts, best_sha):
            best_ts = ts
            best_sha = sha
    iso = (
        datetime.datetime.fromtimestamp(best_ts, tz=datetime.UTC).strftime(
            "%Y-%m-%dT%H:%MZ"
        )
        if best_ts
        else ""
    )
    return best_ts, best_sha, iso


def collect_xfail_295_tests(
    repo_root: pathlib.Path,
) -> list[XfailEntry]:
    """Collect all ``#295`` ``xfail`` tests under ``tests/``."""
    tests_dir = repo_root / "tests"
    entries: list[XfailEntry] = []

    def infer_pkg_and_targets(
        rel_test: str, func_name: str, seg: str
    ) -> tuple[str, tuple[str, ...]]:
        calls: list[str] = []
        tree_fn = ast.parse(seg)
        for sub in ast.walk(tree_fn):
            if isinstance(sub, ast.Call):
                if isinstance(sub.func, ast.Attribute):
                    calls.append(sub.func.attr)
                elif isinstance(sub.func, ast.Name):
                    calls.append(sub.func.id)
        extra = list(XFAIL_EXTRA_TARGETS.get((rel_test, func_name), ()))
        uniq_targets = tuple(
            dict.fromkeys(
                [
                    *extra,
                    *(
                        c
                        for c in calls
                        if c.endswith("_jax")
                        or c.startswith(("bench_", "eval_", "kap_get"))
                    ),
                ]
            )
        )
        fname = pathlib.Path(rel_test).name
        if "bench" in fname:
            pkg = "bench"
        elif "kap" in fname:
            pkg = "kap"
        elif "eos" in fname:
            pkg = "eos"
        elif "ionization" in fname:
            pkg = "ionization"
        elif "star" in fname:
            pkg = "star"
        else:
            pkg = "star"
        return pkg, uniq_targets

    for py_path in sorted(tests_dir.rglob("*.py")):
        text = py_path.read_text(encoding="utf-8")
        if "#295" not in text:
            continue
        rel = py_path.relative_to(repo_root).as_posix()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                dec_seg = ast.get_source_segment(text, dec) or ""
                if "xfail" not in dec_seg or "#295" not in dec_seg:
                    continue
                fn_seg = ast.get_source_segment(text, node) or ""
                reason = _extract_xfail_reason(dec) or dec_seg
                pkg, targets = infer_pkg_and_targets(rel, node.name, fn_seg)
                entries.append(
                    XfailEntry(
                        test_path=rel,
                        line=node.lineno,
                        test_name=node.name,
                        subpackage=pkg,
                        reason=" ".join(reason.split()),
                        targets=targets,
                    )
                )

    return sorted(entries, key=lambda e: (e.subpackage, e.test_path, e.line))


def _extract_xfail_reason(dec: ast.expr) -> str:
    """Extract constant string ``reason=...`` from an xfail decorator."""
    if isinstance(dec, ast.Call):
        for kw in dec.keywords:
            if kw.arg == "reason" and isinstance(kw.value, ast.Constant):
                if isinstance(kw.value.value, str):
                    return kw.value.value
    return ""


def build_inventory(repo_root: pathlib.Path) -> dict[str, Any]:
    """Build the complete P1.5 inventory dictionary from ``repo_root``."""
    py_syms = collect_python_symbols(repo_root)
    rust_items = collect_rust_items(repo_root)
    xfails = collect_xfail_295_tests(repo_root)

    rust_by_rel_qual = {(it.rel_path, it.qualified): it for it in rust_items}
    rust_by_crate_name: dict[tuple[str, str], list[RustItem]] = (
        collections.defaultdict(list)
    )
    rust_by_crate_norm: dict[tuple[str, str], list[RustItem]] = (
        collections.defaultdict(list)
    )
    for it in rust_items:
        rust_by_crate_name[(it.crate, it.name)].append(it)
        if it.scope is not None:
            rust_by_crate_name[(it.crate, it.qualified)].append(it)
        rust_by_crate_norm[(it.crate, _norm(it.name))].append(it)

    pyo3_targets, pyo3_accessed = map_pyo3_to_rust(
        repo_root, rust_by_crate_name
    )

    # First pair every Python item with Rust (validates EXPLICIT_PY_TO_RUST).
    paired: list[tuple[PySymbol, RustItem | None, str]] = []
    paired_rust_keys: set[tuple[str, int, str]] = set(pyo3_accessed)
    blame_files: set[str] = set()

    for ps in py_syms:
        crates = PKG_TO_CRATES[ps.subpackage]
        rs_item, how = _match_python_to_rust(
            ps,
            crates,
            rust_by_rel_qual,
            rust_by_crate_name,
            rust_by_crate_norm,
            pyo3_targets,
        )
        paired.append((ps, rs_item, how))
        if rs_item is not None:
            blame_files.add(ps.rel_path)
            blame_files.add(rs_item.rel_path)
            paired_rust_keys.add(
                (rs_item.rel_path, rs_item.start_line, rs_item.qualified)
            )
            # Also mark the parent struct/enum if matched to a method/type.
            if rs_item.scope is not None:
                for p_cand in rust_by_crate_name.get(
                    (rs_item.crate, rs_item.scope), []
                ):
                    paired_rust_keys.add(
                        (p_cand.rel_path, p_cand.start_line, p_cand.qualified)
                    )

    # Map #295 target names -> list of test refs
    xfail_by_symbol: dict[str, list[str]] = collections.defaultdict(list)
    for xf in xfails:
        for tgt in xf.targets:
            xfail_by_symbol[tgt].append(f"{xf.ref}::{xf.test_name}")

    # Selective git blame on files in paired (py, rs) rows
    blame_cache: dict[str, dict[int, tuple[int, str]]] = {
        rel: _git_blame_file(repo_root, rel) for rel in sorted(blame_files)
    }

    py_rows: list[dict[str, Any]] = []
    stale_candidates: list[dict[str, Any]] = []
    for ps, rs_item, how in paired:
        xf_refs = sorted(xfail_by_symbol.get(ps.name, []))
        if rs_item is None:
            state = "python-only"
            py_ts, py_sha, py_date = 0, "", ""
            rs_ts, rs_sha, rs_date = 0, "", ""
            rs_path_line = ""
        else:
            py_ts, py_sha, py_date = _span_last_commit(
                blame_cache[ps.rel_path], ps.start_line, ps.end_line
            )
            rs_ts, rs_sha, rs_date = _span_last_commit(
                blame_cache[rs_item.rel_path],
                rs_item.start_line,
                rs_item.end_line,
            )
            rs_path_line = rs_item.path_line
            if ps.kind == "PyO3 wrapper":
                state = "mirror-current"
            elif rs_ts > py_ts and rs_sha != py_sha:
                state = "mirror-stale"
            else:
                state = "mirror-current"

        row: dict[str, Any] = {
            "subpackage": ps.subpackage,
            "python_path": ps.rel_path,
            "python_line": ps.start_line,
            "python_end_line": ps.end_line,
            "python_path_line": f"{ps.rel_path}:{ps.start_line}",
            "python_name": ps.name,
            "python_type": ps.py_type,
            "rust_path": rs_item.rel_path if rs_item else "",
            "rust_line": rs_item.start_line if rs_item else 0,
            "rust_path_line": rs_path_line,
            "rust_name": rs_item.qualified if rs_item else "",
            "rust_crate": rs_item.crate if rs_item else "",
            "match_source": how,
            "kind": ps.kind,
            "state": state,
            "py_last_ts": py_ts,
            "py_last_date": py_date,
            "py_last_sha": py_sha[:8] if py_sha else "",
            "rs_last_ts": rs_ts,
            "rs_last_date": rs_date,
            "rs_last_sha": rs_sha[:8] if rs_sha else "",
            "xfail_295": xf_refs,
        }
        py_rows.append(row)
        if state == "mirror-stale":
            stale_candidates.append(row)

    # Determine crate -> summary group mapping for rust-only attribution
    crate_to_group: dict[str, str] = {}
    for pkg, crs in PKG_TO_CRATES.items():
        # Attribute primary crate cleanly (shared crates go to primary pkg).
        crate_to_group.setdefault(crs[0], pkg)
    for cr, grp in CRATE_ONLY_GROUPS.items():
        crate_to_group[cr] = grp

    rust_only_rows: list[dict[str, Any]] = []
    for it in rust_items:
        if not it.externally_public:
            continue
        rkey = (it.rel_path, it.start_line, it.qualified)
        if rkey in paired_rust_keys:
            continue
        grp = crate_to_group.get(it.crate, it.crate)
        rust_only_rows.append(
            {
                "group": grp,
                "crate": it.crate,
                "rust_path": it.rel_path,
                "rust_line": it.start_line,
                "rust_path_line": it.path_line,
                "item_kind": it.item_kind,
                "rust_name": it.qualified,
                "state": "rust-only",
            }
        )

    all_groups = sorted({*PKG_TO_CRATES.keys(), *CRATE_ONLY_GROUPS.values()})
    summary: list[dict[str, Any]] = []
    for grp in all_groups:
        grp_py = [r for r in py_rows if r["subpackage"] == grp]
        grp_ro = [r for r in rust_only_rows if r["group"] == grp]
        grp_xf = [x for x in xfails if x.subpackage == grp]
        pure_cnt = sum(1 for r in grp_py if r["kind"] == "pure-python mirror")
        pyo3_cnt = sum(1 for r in grp_py if r["kind"] == "PyO3 wrapper")
        jax_cnt = sum(1 for r in grp_py if r["kind"] == "JAX kernel")
        frz_cnt = sum(
            1 for r in grp_py if r["kind"] == "JAX engine twin (frozen)"
        )
        cur_cnt = sum(1 for r in grp_py if r["state"] == "mirror-current")
        stl_cnt = sum(1 for r in grp_py if r["state"] == "mirror-stale")
        po_cnt = sum(1 for r in grp_py if r["state"] == "python-only")
        summary.append(
            {
                "subpackage": grp,
                "python_total": len(grp_py),
                "pure_python_mirror": pure_cnt,
                "pyo3_wrapper": pyo3_cnt,
                "jax_kernel": jax_cnt,
                "jax_engine_twin_frozen": frz_cnt,
                "mirror_current": cur_cnt,
                "mirror_stale": stl_cnt,
                "python_only": po_cnt,
                "rust_only": len(grp_ro),
                "xfail_295": len(grp_xf),
            }
        )

    totals = {
        "subpackage": "TOTAL",
        "python_total": sum(r["python_total"] for r in summary),
        "pure_python_mirror": sum(r["pure_python_mirror"] for r in summary),
        "pyo3_wrapper": sum(r["pyo3_wrapper"] for r in summary),
        "jax_kernel": sum(r["jax_kernel"] for r in summary),
        "jax_engine_twin_frozen": sum(
            r["jax_engine_twin_frozen"] for r in summary
        ),
        "mirror_current": sum(r["mirror_current"] for r in summary),
        "mirror_stale": sum(r["mirror_stale"] for r in summary),
        "python_only": sum(r["python_only"] for r in summary),
        "rust_only": sum(r["rust_only"] for r in summary),
        "xfail_295": sum(r["xfail_295"] for r in summary),
    }

    oldest_stale = sorted(
        stale_candidates,
        key=lambda r: (
            r["py_last_ts"],
            r["rs_last_ts"],
            r["python_path_line"],
        ),
    )

    xfail_dicts = [
        {
            "test_path": x.test_path,
            "line": x.line,
            "test_ref": x.ref,
            "test_name": x.test_name,
            "subpackage": x.subpackage,
            "targets": list(x.targets),
            "reason": x.reason,
        }
        for x in xfails
    ]

    return {
        "header": {
            "generator": "scripts/dashboard/build_python_inventory.py",
            "scope": (
                "src/flexmesa/<pkg>/ excluding testing/ and plotting/; "
                "public Rust items across crates/flexmesa-* excluding "
                "flexmesa-py, flexmesa-testing, flexmesa-crlibm-sys, "
                "and binaries"
            ),
            "staleness_rule": (
                "PyO3 wrappers execute Rust directly and stay mirror-current; "
                "pure-python mirrors, JAX kernels, and frozen JAX twins are "
                "mirror-stale when the paired Rust span has a newer git "
                "committer timestamp and different SHA than the Python span"
            ),
        },
        "summary_by_subpackage": summary,
        "totals": totals,
        "oldest_mirror_stale": oldest_stale[:20],
        "xfail_295_tests": xfail_dicts,
        "python_items": py_rows,
        "rust_only_items": rust_only_rows,
    }


def render_outputs(repo_root: pathlib.Path) -> dict[pathlib.Path, str]:
    """Render deterministic JSON and Markdown inventory files in memory."""
    data = build_inventory(repo_root)
    json_text = json.dumps(data, indent=2, sort_keys=False) + "\n"

    t = data["totals"]
    lines: list[str] = [
        "# Python & JAX Port Inventory (generated)",
        "",
        "<!-- GENERATED VIA scripts/dashboard/build_python_inventory.py "
        "(P1.5). Do not edit by hand. -->",
        "",
        "Scope: "
        "`src/flexmesa/<pkg>/` (excluding `testing/` and `plotting/`) and "
        "externally-reachable top-level public Rust items across library "
        "crates.",
        "",
        "## Classification & Staleness Rules",
        "",
        "* **Kinds**: `PyO3 wrapper` (calls `_rust.*`), `JAX engine twin "
        "(frozen)` (`#166` frozen star/binary/astero modules), `JAX kernel` "
        "(non-frozen `@jax.jit` / `jnp` / `custom_vjp` kernels), and "
        "`pure-python mirror` (pure Python/NumPy implementations).",
        "* **States**: `PyO3 wrapper` items delegate directly to Rust at "
        "runtime and are always `mirror-current`; non-PyO3 mirrors/JAX "
        "kernels are "
        "`mirror-stale` when the paired Rust implementation span has a newer "
        "`git blame` committer timestamp and different commit SHA than the "
        "Python span, otherwise `mirror-current`. Unpaired Python items are "
        "`python-only`; externally-reachable public Rust items with neither a "
        "Python counterpart nor a PyO3 binding in `crates/flexmesa-py` are "
        "`rust-only`.",
        "",
        "## Per-Subpackage Summary",
        "",
        f"Headline totals: **{t['python_total']}** public Python items "
        f"(**{t['pure_python_mirror']}** pure-python mirror, "
        f"**{t['pyo3_wrapper']}** PyO3 wrapper, **{t['jax_kernel']}** JAX "
        f"kernel, **{t['jax_engine_twin_frozen']}** frozen JAX twin; "
        f"**{t['mirror_current']}** `mirror-current`, **{t['mirror_stale']}** "
        f"`mirror-stale`, **{t['python_only']}** `python-only`), plus "
        f"**{t['rust_only']}** `rust-only` public items and "
        f"**{t['xfail_295']}** `#295` JAX `xfail` tests.",
        "",
        "| Subpackage | Python | `pure-python mirror` | `PyO3 wrapper` | "
        "`JAX kernel` | `JAX engine twin (frozen)` | `mirror-current` | "
        "`mirror-stale` | `python-only` | `rust-only` | `#295` xfails |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: |",
    ]

    for r in data["summary_by_subpackage"]:
        lines.append(
            f"| `{r['subpackage']}` | {r['python_total']} | "
            f"{r['pure_python_mirror']} | {r['pyo3_wrapper']} | "
            f"{r['jax_kernel']} | {r['jax_engine_twin_frozen']} | "
            f"{r['mirror_current']} | {r['mirror_stale']} | "
            f"{r['python_only']} | {r['rust_only']} | {r['xfail_295']} |"
        )
    lines.append(
        f"| **TOTAL** | **{t['python_total']}** | "
        f"**{t['pure_python_mirror']}** | **{t['pyo3_wrapper']}** | "
        f"**{t['jax_kernel']}** | **{t['jax_engine_twin_frozen']}** | "
        f"**{t['mirror_current']}** | **{t['mirror_stale']}** | "
        f"**{t['python_only']}** | **{t['rust_only']}** | "
        f"**{t['xfail_295']}** |"
    )

    lines.extend(
        [
            "",
            "## Oldest `mirror-stale` Rows",
            "",
            "| Python path:line | Symbol | Rust path:line | Kind | "
            "Python last | Rust last |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for r in data["oldest_mirror_stale"]:
        lines.append(
            f"| `{r['python_path_line']}` | `{r['python_name']}` | "
            f"`{r['rust_path_line']}` | `{r['kind']}` | "
            f"`{r['py_last_date']}` (`{r['py_last_sha']}`) | "
            f"`{r['rs_last_date']}` (`{r['rs_last_sha']}`) |"
        )

    lines.extend(
        [
            "",
            f"## `#295` JAX `xfail` Tests ({len(data['xfail_295_tests'])})",
            "",
            "| Test path:line | Test | Subpackage | Target symbol(s) | "
            "Reason |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for x in data["xfail_295_tests"]:
        tgts = ", ".join(f"`{sym}`" for sym in x["targets"]) or "-"
        rsn = x["reason"].replace("|", "\\|")
        lines.append(
            f"| `{x['test_ref']}` | `{x['test_name']}` | "
            f"`{x['subpackage']}` | {tgts} | {rsn} |"
        )

    lines.extend(
        [
            "",
            f"## Python & JAX Items ({len(data['python_items'])})",
            "",
            "| Python path:line | Rust path:line | Kind | State | "
            "`#295` xfail |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for r in data["python_items"]:
        rs_cell = f"`{r['rust_path_line']}`" if r["rust_path_line"] else "-"
        xf_cell = (
            ", ".join(f"`{ref}`" for ref in r["xfail_295"])
            if r["xfail_295"]
            else "-"
        )
        lines.append(
            f"| `{r['python_path_line']}` (`{r['python_name']}`) | "
            f"{rs_cell} | `{r['kind']}` | `{r['state']}` | {xf_cell} |"
        )

    lines.extend(
        [
            "",
            f"## Public Rust Items With No Python Access "
            f"({len(data['rust_only_items'])})",
            "",
            "| Crate | Rust path:line | Item kind | Name | State |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for ro in data["rust_only_items"]:
        lines.append(
            f"| `{ro['crate']}` | `{ro['rust_path_line']}` | "
            f"`{ro['item_kind']}` | `{ro['rust_name']}` | `rust-only` |"
        )
    lines.append("")

    md_text = "\n".join(lines)
    return {
        repo_root / JSON_RELPATH: json_text,
        repo_root / MD_RELPATH: md_text,
    }


def main(argv: list[str] | None = None) -> int:
    """Generate or check ``notes/audit/python_inventory.{json,md}``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=pathlib.Path, default=DEFAULT_REPO_ROOT
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify committed files match regeneration byte-for-byte.",
    )
    ns = parser.parse_args(argv)
    root = ns.repo_root.resolve()
    outputs = render_outputs(root)

    if ns.check:
        bad = 0
        for path, expected in outputs.items():
            rel = path.relative_to(root).as_posix()
            if (
                not path.is_file()
                or path.read_text(encoding="utf-8") != expected
            ):
                print(
                    f"ERROR: {rel} differs from regeneration",
                    file=sys.stderr,
                )
                bad += 1
        return 1 if bad else 0

    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"Wrote {path.relative_to(root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
