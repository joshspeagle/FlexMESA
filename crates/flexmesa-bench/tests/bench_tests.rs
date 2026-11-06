//! Integration tests for the Wave 1 four-stage benchmark harness
//! (`flexmesa-bench`).
//!
//! Verifies that all six representative Wave 1 workloads (`auto_diff`,
//! `interp_2d`, `neu`, `mtx_tridiag`, `turb_mlt`, and `turb_tdc`) produce
//! machine-precision parity ($\le 10^{-12}$ relative error) across Stages 0..3
//! and provides a `--release` Markdown table printer (`--ignored`).

use flexmesa_bench::{run_all_wave1_benchmarks, run_all_wave2_benchmarks};

/// Pinned cross-language reference checksums for $N_{\mathrm{zones}} = 8$
/// across all six Wave 1 workloads (shared with `tests/test_bench.py`).
/// Each entry pairs the canonical workload identifier with the exact Stage 2
/// `f64` primal reduction sum.
const EXPECTED_CHECKSUMS_N8: [(&str, f64); 6] = [
    ("auto_diff", 1.089_274_155_553_244e4),
    ("interp_2d", 3.622_775_349_153_983e2),
    ("neu", 9.868_900_412_911_318e12),
    ("mtx_tridiag", 1.065_650_530_823_759e1),
    ("turb_mlt", 3.419_048_546_150_96),
    ("turb_tdc", 3.318_734_224_165_011_5),
];

/// Pinned cross-language reference checksums for $N_{\mathrm{zones}} = 8$
/// across all four Wave 2 workloads (shared with `tests/test_bench.py` and
/// `benchmarks/fortran_mesa/run_wave2_bench.f90`).
const EXPECTED_WAVE2_CHECKSUMS_N8: [(&str, f64); 4] = [
    ("chem_comp", 1.687_001_493_521_061e1),
    ("ion_fe56", 1.683_709_392_958_418e2),
    ("ion_4d", 6.118_846_146_172_181e1),
    ("colors_sed", 3.608_691_732_539_108e1),
];

/// Verifies that all six Wave 1 benchmark workloads (`auto_diff`, `interp_2d`,
/// `neu`, `mtx_tridiag`, `turb_mlt`, and `turb_tdc`) execute across Stages 0..3
/// and maintain machine-precision parity ($\le 10^{-12}$ relative discrepancy)
/// across all four optimization stages and against pinned cross-language
/// reference checksums.
#[test]
fn test_wave1_four_stage_benchmarks_parity() {
    let records = run_all_wave1_benchmarks(&[8, 32], 1, 3);
    assert_eq!(records.len(), 12);
    for rec in &records {
        assert!(
            rec.max_rel_err <= 1.0e-12,
            "workload {} (N={}) stage parity error {:.3e} > 1e-12",
            rec.workload,
            rec.n_zones,
            rec.max_rel_err
        );
        assert!(rec.stage0.median_us > 0.0);
        assert!(rec.stage1.median_us > 0.0);
        assert!(rec.stage2.median_us > 0.0);
        assert!(rec.stage3.median_us > 0.0);
        assert!(rec.checksum.is_finite());
    }
    for (rec, &(exp_name, exp_sum)) in
        records[..6].iter().zip(EXPECTED_CHECKSUMS_N8.iter())
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

/// Prints the full Wave 1 four-stage benchmark table across
/// $N_{\mathrm{zones}} \in \{128, 1024, 4096\}$ when invoked with
/// `cargo test --release -p flexmesa-bench -- --ignored --nocapture`.
#[test]
#[ignore = "benchmark table generator for --release runs"]
fn print_full_wave1_rust_benchmarks() {
    let records = run_all_wave1_benchmarks(&[128, 1024, 4096], 3, 9);
    println!(
        "| Workload | N_zones | Stage 0 (us) | Stage 1 (us) | \
         Stage 2 (us) | Stage 3 (us) | Max Rel Err | Checksum |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|");
    for rec in &records {
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

/// Verifies that all four Wave 2 benchmark workloads (`chem_comp`, `ion_fe56`,
/// `ion_4d`, and `colors_sed`) execute across Stages 0..3 and maintain
/// machine-precision parity ($\le 10^{-12}$ relative discrepancy) across all
/// four optimization stages.
#[test]
fn test_wave2_four_stage_benchmarks_parity() {
    let records = run_all_wave2_benchmarks(&[8, 32], 1, 3);
    assert_eq!(records.len(), 8);
    for rec in &records {
        assert!(
            rec.max_rel_err <= 1.0e-12,
            "workload {} (N={}) stage parity error {:.3e} > 1e-12",
            rec.workload,
            rec.n_zones,
            rec.max_rel_err
        );
        assert!(rec.stage0.median_us > 0.0);
        assert!(rec.stage1.median_us > 0.0);
        assert!(rec.stage2.median_us > 0.0);
        assert!(rec.stage3.median_us > 0.0);
        assert!(rec.checksum.is_finite());
    }
    for (rec, &(exp_name, exp_sum)) in
        records[..4].iter().zip(EXPECTED_WAVE2_CHECKSUMS_N8.iter())
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

/// Prints the full Wave 2 four-stage benchmark table across
/// $N_{\mathrm{zones}} \in \{128, 1024, 4096\}$ when invoked with
/// `cargo test --release -p flexmesa-bench -- --ignored --nocapture`.
#[test]
#[ignore = "benchmark table generator for --release runs"]
fn print_full_wave2_rust_benchmarks() {
    let records = run_all_wave2_benchmarks(&[128, 1024, 4096], 3, 9);
    println!(
        "| Workload | N_zones | Stage 0 (us) | Stage 1 (us) | \
         Stage 2 (us) | Stage 3 (us) | Max Rel Err | Checksum |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|");
    for rec in &records {
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
