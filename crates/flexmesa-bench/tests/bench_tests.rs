//! Integration tests for the Wave 1 four-stage benchmark harness
//! (`flexmesa-bench`).

use flexmesa_bench::run_all_wave1_benchmarks;

/// Verifies that all six Wave 1 benchmark workloads (`auto_diff`, `interp_2d`,
/// `neu`, `mtx_tridiag`, `turb_mlt`, and `turb_tdc`) execute across Stages 0..3
/// and maintain machine-precision parity ($\le 10^{-12}$ relative discrepancy)
/// across all four optimization stages.
#[test]
fn test_wave1_four_stage_benchmarks_parity() {
    let records = run_all_wave1_benchmarks(&[32, 128], 1, 3);
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
    println!(
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    );
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
