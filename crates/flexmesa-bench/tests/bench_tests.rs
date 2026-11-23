// SPDX-License-Identifier: LGPL-3.0-or-later
//! Integration tests for the four-stage benchmark harness (`flexmesa-bench`).
//!
//! Checks internal Stage 0..3 consistency ($\le 10^{-12}$ relative difference,
//! $\le 10^{-10}$ on Wave 4) and compares Stage 0 against self-generated
//! `FlexMESA` regression snapshots at $N_{\mathrm{zones}} = 8$. These snapshots
//! are internal regression values, not upstream MESA outputs (tracked in #146).

use flexmesa_bench::{
    BenchRecord, run_all_wave1_benchmarks, run_all_wave2_benchmarks,
    run_all_wave3a_benchmarks, run_all_wave3b_benchmarks,
    run_all_wave4_benchmarks, run_all_wave5_benchmarks,
};

/// `FlexMESA` regression snapshots for $N_{\mathrm{zones}} = 8$ across all six
/// Wave 1 workloads (internal self-consistency check, tracked in #146).
const WAVE1_REGRESSION_SNAPSHOTS_N8: [(&str, f64); 6] = [
    ("auto_diff", 1.089_274_155_553_244e4),
    ("interp_2d", 3.622_775_349_153_983e2),
    ("neu", 9.873_195_638_253_145e12),
    ("mtx_tridiag", 1.065_650_530_823_759e1),
    ("turb_mlt", 3.419_048_546_150_96),
    ("turb_tdc", 3.318_734_224_165_011_5),
];

/// `FlexMESA` regression snapshots for $N_{\mathrm{zones}} = 8$ across all four
/// Wave 2 workloads (internal self-consistency check, tracked in #146).
const WAVE2_REGRESSION_SNAPSHOTS_N8: [(&str, f64); 4] = [
    ("chem_comp", 1.687_001_493_521_061e1),
    ("ion_fe56", 1.683_709_392_958_418e2),
    ("ion_4d", 6.118_846_146_172_181e1),
    ("colors_sed", 3.608_691_732_539_108e1),
];

/// `FlexMESA` regression snapshots for $N_{\mathrm{zones}} = 8$ across all four
/// Wave 3a EOS workloads (internal self-consistency check, tracked in #146).
const WAVE3A_REGRESSION_SNAPSHOTS_N8: [(&str, f64); 4] = [
    ("eos_ideal", 1.438_224_257_481_198e2),
    ("eos_helm", 3.561_688_957_866_044e2),
    ("eos_pc", 4.117_886_894_420_108e2),
    ("eos_skye", 3.561_331_836_722_772e2),
];

/// `FlexMESA` regression snapshots for $N_{\mathrm{zones}} = 8$ across all five
/// Wave 3b microphysics workloads (internal self-consistency check, tracked in
/// #146).
const WAVE3B_REGRESSION_SNAPSHOTS_N8: [(&str, f64); 5] = [
    ("eos_4d", 2.037_364_564_689_326e2),
    ("kap_fixed", 2.909_195_757_111_935),
    ("rates_screen", -1.139_188_914_724_813e2),
    ("net_eval", 9.619_649_848_151_354e1),
    ("atm_table", 8.985_411_175_716_764e1),
];

/// `FlexMESA` regression snapshots for $N_{\mathrm{zones}} = 8$ across all five
/// Wave 4 1D stellar engine workloads (internal self-consistency check, not an
/// upstream MESA output; `star_hydro_eqns`, `star_newton_step`, and
/// `star_evolve_step` shifted under `#141` when `neu` `#193` and `num` `#194`
/// were aligned to MESA `26.4.1`, all five shifted under `#149` when
/// `&controls` / microphysics / mesh-function defaults were aligned to MESA
/// `26.4.1`, `star_remesh` shifted from `5.513_777_703_025_969e2` to
/// `5.108_453_216_631_093e2` under `#153` when `eval_mesh_gval` was ported to
/// `set_mesh_function_data` + `get_gval_info`, `star_micro_vars`
/// (`2.497_139_571_258_105e2` -> `2.500_111_893_102_388e2`), `star_hydro_eqns`
/// (`4.253_179_678_855_851e6` -> `4.243_492_123_572_694e6`), and
/// `star_newton_step` (`1.451_475_459_976_951e2` -> `6.383_737_350_497_265`)
/// shifted under `#155` when `remove_mixing_singletons`
/// (`mix_info.f90:1026-1072`) switched the isolated `k = 2` convective
/// singleton on the coarse `N = 8` grid to radiative (`gradt = gradr`), and
/// `star_hydro_eqns`, `star_newton_step`, and `star_evolve_step` shifted under
/// `#152` when `set_xscale_info` and Newton step/equilibration semantics were
/// aligned to MESA `26.4.1` (`DECISIONS #42`), and `star_micro_vars`
/// (`2.500_111_893_102_388e2` -> `2.499_330_102_887_365e2`), `star_hydro_eqns`
/// (`4.243_485_684_423_352e6` -> `4.262_783_875_690_237e6`) and
/// `star_newton_step` (`6.383_773_435_545_09` -> `2.210_834_729_000_362e5`)
/// shifted under `#291` when the face radiation pressure in `do1_mlt_2` was
/// changed to `crad*T_face^4/3` (`star_utils.f90:3838-3843`). Under `#291`
/// `star_evolve_step` also moved from the 8-zone grid
/// (`1.005_928_065_548_472e2`) to `MIN_EVOLVE_STEP_ZONES = 64` zones
/// (`8.895_997_453_722_767e2`), because the 8-zone fixture no longer
/// converges at the model's timestep; tracked in `#146`). Under `#340`
/// (`eval_face_dm_bar` at `k = nz - 1` matching
/// `star/private/hydro_momentum.f90:191-196` and `dt_limit_ratio_old = 0.0`
/// matching `star/private/init.f90:509, 696`), `star_hydro_eqns`
/// (`4.262_783_875_690_237e6` -> `4.262_780_484_078_014e6`),
/// `star_newton_step` (`2.210_834_729_000_362e5` -> `2.263_803_558_180_566e5`),
/// and `star_evolve_step` (`8.895_997_453_722_767e2` ->
/// `8.895_028_893_306_655e2`) shifted. Under `#342` (ZAMS `interp_q` matching
/// `star/private/star_utils.f90:729-752, 1232-1265`, 8-decimal ZAMS index
/// masses matching `star/private/init_model.f90:210, 254, 444-446`, and
/// initial ZAMS `model_number = 0` matching `star/private/init.f90:367`),
/// `star_micro_vars` (`2.499_330_102_887_365e2` -> `2.253_780_782_165_539e2`),
/// `star_hydro_eqns` (`4.262_780_484_078_014e6` -> `1.742_907_215_943_594e6`),
/// `star_newton_step` (`2.263_803_558_180_566e5` -> `8.653_548_256_546_203e5`),
/// `star_remesh` (`5.108_453_216_631_093e2` -> `7.086_334_744_875_726e2`), and
/// `star_evolve_step` (`8.895_028_893_306_655e2` -> `8.843_428_194_210_911e2`)
/// shifted. Under `#360` (removing non-upstream ZAMS `xa` renormalization and
/// `.max(0.0)` clamping per `star/private/init_model.f90:155-397` and
/// `star_utils.f90:1244-1265`), `star_newton_step`
/// (`8.653_548_256_546_203e5` -> `8.653_548_285_862_873e5`) shifted. Under
/// `#381` (`set_qs` and `normalize_dqs` matching
/// `star/private/star_utils.f90:669-752`, and no renormalization of `dq` in
/// the mesh sync), `star_newton_step` (`8.653_548_285_862_873e5` ->
/// `8.653_548_271_013_098e5`) shifted.
const WAVE4_REGRESSION_SNAPSHOTS_N8: [(&str, f64); 5] = [
    ("star_micro_vars", 2.253_780_782_165_539e2),
    ("star_hydro_eqns", 1.742_907_215_943_594e6),
    ("star_newton_step", 8.653_548_271_013_098e5),
    ("star_remesh", 7.086_334_744_875_726e2),
    ("star_evolve_step", 8.843_428_194_210_911e2),
];

/// `FlexMESA` regression snapshots for $N_{\mathrm{zones}} = 8$ across all four
/// Wave 5 workloads (internal self-consistency check, not an upstream MESA
/// output; `w5_diff_rot` shifted from `3.082380424983001e5` to
/// `2.551374595308515e3` in `642da54`, and to `2.551511752772096e3` under
/// `#156` when `eval_caplan_ocp_dstar` was aligned to MESA `26.4.1`
/// `diffusion_support.f90:get_Dstar_OCP`; tracked in #146).
const WAVE5_REGRESSION_SNAPSHOTS_N8: [(&str, f64); 4] = [
    ("w5_diff_rot", 2.551_511_752_772_096e3),
    ("w5_riemann_rsp2", -2.543_418_211_346_364e5),
    ("w5_binary_rlof", 6.234_702_123_446_103),
    ("w5_astero_seism", 7.563_537_818_023_943e1),
];

fn assert_records_match_snapshots_with_tol(
    records: &[BenchRecord],
    snapshots_n8: &[(&str, f64)],
    max_stage_rel_err: f64,
) {
    assert_eq!(records.len(), snapshots_n8.len() * 2);
    for rec in records {
        assert!(
            rec.max_rel_err <= max_stage_rel_err,
            "workload {} (N={}) cross-stage relative error {:.3e} > {:.1e}",
            rec.workload,
            rec.n_zones,
            rec.max_rel_err,
            max_stage_rel_err
        );
        assert!(rec.stage0.median_us > 0.0);
        assert!(rec.stage1.median_us > 0.0);
        assert!(rec.stage2.median_us > 0.0);
        assert!(rec.stage3.median_us > 0.0);
        assert!(rec.checksum.is_finite());
    }
    for (rec, &(exp_name, exp_sum)) in records[..snapshots_n8.len()]
        .iter()
        .zip(snapshots_n8.iter())
    {
        assert_eq!(rec.workload, exp_name);
        let rel_err = (rec.checksum - exp_sum).abs() / exp_sum.abs();
        assert!(
            rel_err <= 1.0e-12,
            "workload {} (N=8) snapshot {:.15e} != {:.15e} (rel {:.3e})",
            rec.workload,
            rec.checksum,
            exp_sum,
            rel_err
        );
    }
}

fn assert_records_match_snapshots(
    records: &[BenchRecord],
    snapshots_n8: &[(&str, f64)],
) {
    assert_records_match_snapshots_with_tol(records, snapshots_n8, 1.0e-12);
}

/// Checks that all six Wave 1 benchmark workloads execute across Stages 0..3
/// and match their `FlexMESA` regression snapshots ($\le 10^{-12}$).
#[test]
fn test_wave1_four_stage_benchmarks_match_snapshots() {
    let records = run_all_wave1_benchmarks(&[8, 32], 1, 3);
    assert_records_match_snapshots(&records, &WAVE1_REGRESSION_SNAPSHOTS_N8);
}

/// Checks that all four Wave 2 benchmark workloads execute across Stages 0..3
/// and match their `FlexMESA` regression snapshots ($\le 10^{-12}$).
#[test]
fn test_wave2_four_stage_benchmarks_match_snapshots() {
    let records = run_all_wave2_benchmarks(&[8, 32], 1, 3);
    assert_records_match_snapshots(&records, &WAVE2_REGRESSION_SNAPSHOTS_N8);
}

/// Checks that all four Wave 3a EOS benchmark workloads execute across Stages
/// 0..3 and match their `FlexMESA` regression snapshots ($\le 10^{-12}$).
#[test]
fn test_wave3a_four_stage_benchmarks_match_snapshots() {
    let records = run_all_wave3a_benchmarks(&[8, 32], 1, 3);
    assert_records_match_snapshots(&records, &WAVE3A_REGRESSION_SNAPSHOTS_N8);
}

/// Checks that all five Wave 3b microphysics benchmark workloads execute
/// across Stages 0..3 and match their `FlexMESA` regression snapshots ($\le 10^{-12}$).
#[test]
fn test_wave3b_four_stage_benchmarks_match_snapshots() {
    let records = run_all_wave3b_benchmarks(&[8, 32], 1, 3);
    assert_records_match_snapshots(&records, &WAVE3B_REGRESSION_SNAPSHOTS_N8);
}

/// Checks that all five Wave 4 1D stellar engine benchmark workloads execute
/// across Stages 0..3 and match their `FlexMESA` regression snapshots ($\le 10^{-10}$).
#[test]
fn test_wave4_four_stage_benchmarks_match_snapshots() {
    let records = run_all_wave4_benchmarks(&[8, 32], 1, 3);
    assert_records_match_snapshots_with_tol(
        &records,
        &WAVE4_REGRESSION_SNAPSHOTS_N8,
        1.0e-10,
    );
}

/// Checks that all four Wave 5 benchmark workloads execute across Stages 0..3
/// and match their `FlexMESA` regression snapshots ($\le 10^{-12}$).
#[test]
fn test_wave5_four_stage_benchmarks_match_snapshots() {
    let records = run_all_wave5_benchmarks(&[8, 32], 1, 3);
    assert_records_match_snapshots(&records, &WAVE5_REGRESSION_SNAPSHOTS_N8);
}
