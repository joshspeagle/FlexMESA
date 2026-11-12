// SPDX-License-Identifier: LGPL-3.0-or-later
//! Integration tests for the four-stage benchmark harness (`flexmesa-bench`).
//!
//! Verifies that all Wave 1 (`auto_diff`, `interp_2d`, `neu`, `mtx_tridiag`,
//! `turb_mlt`, `turb_tdc`), Wave 2 (`chem_comp`, `ion_fe56`, `ion_4d`,
//! `colors_sed`), Wave 3a (`eos_ideal`, `eos_helm`, `eos_pc`, `eos_skye`), and
//! Wave 3b (`eos_4d`, `kap_fixed`, `rates_screen`, `net_eval`, `atm_table`)
//! workloads maintain machine-precision parity ($\le 10^{-12}$ relative error)
//! across Stages 0..3 and against pinned cross-language checksums.

use flexmesa_bench::{
    run_all_wave1_benchmarks, run_all_wave2_benchmarks,
    run_all_wave3a_benchmarks, run_all_wave3b_benchmarks,
    run_all_wave4_benchmarks, run_all_wave5_benchmarks, BenchRecord,
};

/// Pinned cross-language reference checksums for $N_{\mathrm{zones}} = 8$
/// across all six Wave 1 workloads (shared with `tests/test_bench.py`).
const EXPECTED_CHECKSUMS_N8: [(&str, f64); 6] = [
    ("auto_diff", 1.089_274_155_553_244e4),
    ("interp_2d", 3.622_775_349_153_983e2),
    ("neu", 9.868_900_412_911_318e12),
    ("mtx_tridiag", 1.065_650_530_823_759e1),
    ("turb_mlt", 3.419_048_546_150_96),
    ("turb_tdc", 3.318_734_224_165_011_5),
];

/// Pinned cross-language reference checksums for $N_{\mathrm{zones}} = 8$
/// across all four Wave 2 workloads.
const EXPECTED_WAVE2_CHECKSUMS_N8: [(&str, f64); 4] = [
    ("chem_comp", 1.687_001_493_521_061e1),
    ("ion_fe56", 1.683_709_392_958_418e2),
    ("ion_4d", 6.118_846_146_172_181e1),
    ("colors_sed", 3.608_691_732_539_108e1),
];

/// Pinned cross-language reference checksums for $N_{\mathrm{zones}} = 8$
/// across all four Wave 3a EOS workloads.
const EXPECTED_WAVE3A_CHECKSUMS_N8: [(&str, f64); 4] = [
    ("eos_ideal", 1.438_224_257_481_198e2),
    ("eos_helm", 3.561_688_957_866_044e2),
    ("eos_pc", 4.117_886_894_420_108e2),
    ("eos_skye", 3.561_331_836_722_772e2),
];

/// Pinned cross-language reference checksums for $N_{\mathrm{zones}} = 8$
/// across all five Wave 3b microphysics workloads (shared with
/// `tests/test_bench.py` and `benchmarks/fortran_mesa/run_wave3b_bench.f90`).
const EXPECTED_WAVE3B_CHECKSUMS_N8: [(&str, f64); 5] = [
    ("eos_4d", 2.037_364_564_689_326e2),
    ("kap_fixed", 2.909_195_757_111_935),
    ("rates_screen", -1.139_188_914_724_813e2),
    ("net_eval", 9.619_649_848_151_354e1),
    ("atm_table", 8.985_411_175_716_764e1),
];

/// Pinned cross-language reference checksums for $N_{\mathrm{zones}} = 8$
/// across all five Wave 4 1D stellar engine workloads (shared with
/// `tests/test_wave4_bench.py` and `benchmarks/fortran_mesa/run_wave4_bench.f90`).
const EXPECTED_WAVE4_CHECKSUMS_N8: [(&str, f64); 5] = [
    ("star_micro_vars", 2.497_297_961_588_147e2),
    ("star_hydro_eqns", 4.660_073_228_335_436e6),
    ("star_newton_step", 1.333_277_163_962_19e2),
    ("star_remesh", 4.203_476_959_958_853e2),
    ("star_evolve_step", 9.987_143_083_921_163e1),
];

/// Pinned cross-language reference checksums for $N_{\mathrm{zones}} = 8$
/// across all four Wave 5 workloads (shared with `tests/test_wave5_bench.py`
/// and `benchmarks/fortran_mesa/run_wave5_bench.f90`).
const EXPECTED_WAVE5_CHECKSUMS_N8: [(&str, f64); 4] = [
    ("w5_diff_rot", 3.082_380_424_983_001e5),
    ("w5_riemann_rsp2", -2.543_418_211_346_364e5),
    ("w5_binary_rlof", 6.234_702_123_446_103),
    ("w5_astero_seism", 7.563_537_818_023_943e1),
];

fn assert_records_parity_with_tol(
    records: &[BenchRecord],
    expected_n8: &[(&str, f64)],
    max_stage_rel_err: f64,
) {
    assert_eq!(records.len(), expected_n8.len() * 2);
    for rec in records {
        assert!(
            rec.max_rel_err <= max_stage_rel_err,
            "workload {} (N={}) stage parity error {:.3e} > {:.1e}",
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
    for (rec, &(exp_name, exp_sum)) in
        records[..expected_n8.len()].iter().zip(expected_n8.iter())
    {
        assert_eq!(rec.workload, exp_name);
        let rel_err = (rec.checksum - exp_sum).abs() / exp_sum.abs();
        assert!(
            rel_err <= 1.0e-12,
            "workload {} (N=8) checksum {:.15e} vs {:.15e} (rel {:.3e})",
            rec.workload,
            rec.checksum,
            exp_sum,
            rel_err
        );
    }
}

fn assert_records_parity(records: &[BenchRecord], expected_n8: &[(&str, f64)]) {
    assert_records_parity_with_tol(records, expected_n8, 1.0e-12);
}

fn print_bench_table(records: &[BenchRecord]) {
    println!(
        "| Workload | N_zones | Stage 0 (us) | Stage 1 (us) | \
         Stage 2 (us) | Stage 3 (us) | Max Rel Err | Checksum |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|");
    for rec in records {
        println!(
            "| `{}` | {} | {:.2} +/- {:.2} | {:.2} +/- {:.2} | \
             {:.2} +/- {:.2} | {:.2} +/- {:.2} | {:.3e} | {:.8e} |",
            rec.workload,
            rec.n_zones,
            rec.stage0.median_us,
            rec.stage0.iqr_us,
            rec.stage1.median_us,
            rec.stage1.iqr_us,
            rec.stage2.median_us,
            rec.stage2.iqr_us,
            rec.stage3.median_us,
            rec.stage3.iqr_us,
            rec.max_rel_err,
            rec.checksum
        );
    }
}

/// Verifies that all six Wave 1 benchmark workloads execute across Stages 0..3
/// and maintain machine-precision parity ($\le 10^{-12}$).
#[test]
fn test_wave1_four_stage_benchmarks_parity() {
    let records = run_all_wave1_benchmarks(&[8, 32], 1, 3);
    assert_records_parity(&records, &EXPECTED_CHECKSUMS_N8);
}

/// Prints the full Wave 1 four-stage benchmark table across
/// $N_{\mathrm{zones}} \in \{128, 1024, 4096\}$.
#[test]
#[ignore = "benchmark table generator for --release runs"]
fn print_full_wave1_rust_benchmarks() {
    let records = run_all_wave1_benchmarks(&[128, 1024, 4096], 3, 9);
    print_bench_table(&records);
}

/// Verifies that all four Wave 2 benchmark workloads execute across Stages 0..3
/// and maintain machine-precision parity ($\le 10^{-12}$).
#[test]
fn test_wave2_four_stage_benchmarks_parity() {
    let records = run_all_wave2_benchmarks(&[8, 32], 1, 3);
    assert_records_parity(&records, &EXPECTED_WAVE2_CHECKSUMS_N8);
}

/// Prints the full Wave 2 four-stage benchmark table across
/// $N_{\mathrm{zones}} \in \{128, 1024, 4096\}$.
#[test]
#[ignore = "benchmark table generator for --release runs"]
fn print_full_wave2_rust_benchmarks() {
    let records = run_all_wave2_benchmarks(&[128, 1024, 4096], 3, 9);
    print_bench_table(&records);
}

/// Verifies that all four Wave 3a EOS benchmark workloads execute across Stages
/// 0..3 and maintain machine-precision parity ($\le 10^{-12}$).
#[test]
fn test_wave3a_four_stage_benchmarks_parity() {
    let records = run_all_wave3a_benchmarks(&[8, 32], 1, 3);
    assert_records_parity(&records, &EXPECTED_WAVE3A_CHECKSUMS_N8);
}

/// Prints the full Wave 3a four-stage benchmark table across
/// $N_{\mathrm{zones}} \in \{128, 1024, 4096\}$.
#[test]
#[ignore = "benchmark table generator for --release runs"]
fn print_full_wave3a_rust_benchmarks() {
    let records = run_all_wave3a_benchmarks(&[128, 1024, 4096], 3, 9);
    print_bench_table(&records);
}

/// Verifies that all five Wave 3b microphysics benchmark workloads execute
/// across Stages 0..3 and maintain machine-precision parity ($\le 10^{-12}$).
#[test]
fn test_wave3b_four_stage_benchmarks_parity() {
    let records = run_all_wave3b_benchmarks(&[8, 32], 1, 3);
    assert_records_parity(&records, &EXPECTED_WAVE3B_CHECKSUMS_N8);
}

/// Prints the full Wave 3b four-stage benchmark table across
/// $N_{\mathrm{zones}} \in \{128, 1024, 4096\}$.
#[test]
#[ignore = "benchmark table generator for --release runs"]
fn print_full_wave3b_rust_benchmarks() {
    let records = run_all_wave3b_benchmarks(&[128, 1024, 4096], 3, 9);
    print_bench_table(&records);
}

/// Verifies that all five Wave 4 1D stellar engine benchmark workloads execute
/// across Stages 0..3 and maintain machine-precision parity ($\le 10^{-10}$).
#[test]
fn test_wave4_four_stage_benchmarks_parity() {
    let records = run_all_wave4_benchmarks(&[8, 32], 1, 3);
    assert_records_parity_with_tol(
        &records,
        &EXPECTED_WAVE4_CHECKSUMS_N8,
        1.0e-10,
    );
}

/// Prints the full Wave 4 four-stage benchmark table across
/// $N_{\mathrm{zones}} \in \{128, 512, 1024\}$.
#[test]
#[ignore = "benchmark table generator for --release runs"]
fn print_full_wave4_rust_benchmarks() {
    let records = run_all_wave4_benchmarks(&[128, 512, 1024], 2, 7);
    print_bench_table(&records);
}

/// Verifies that all four Wave 5 benchmark workloads execute across Stages 0..3
/// and maintain machine-precision parity ($\le 10^{-12}$).
#[test]
fn test_wave5_four_stage_benchmarks_parity() {
    let records = run_all_wave5_benchmarks(&[8, 32], 1, 3);
    assert_records_parity(&records, &EXPECTED_WAVE5_CHECKSUMS_N8);
}

/// Prints the full Wave 5 four-stage benchmark table across
/// $N_{\mathrm{zones}} \in \{128, 512, 1024\}$.
#[test]
#[ignore = "benchmark table generator for --release runs"]
fn print_full_wave5_rust_benchmarks() {
    let records = run_all_wave5_benchmarks(&[128, 512, 1024], 3, 11);
    print_bench_table(&records);
}
