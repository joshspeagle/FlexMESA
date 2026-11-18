// SPDX-License-Identifier: LGPL-3.0-or-later
//! Standalone CLI binary emitting Markdown tables for Safe Rust Wave 1 to
//! Wave 5 four-stage benchmarks (`flexmesa-bench`).
//!
//! Separated from unit/integration tests under `#146` so benchmark runners do
//! not live behind `#[ignore]` test functions.

#![forbid(unsafe_code)]

use flexmesa_bench::{
    run_all_wave1_benchmarks, run_all_wave2_benchmarks,
    run_all_wave3a_benchmarks, run_all_wave3b_benchmarks,
    run_all_wave4_benchmarks, run_all_wave5_benchmarks, BenchRecord,
};

fn print_bench_table(wave_label: &str, records: &[BenchRecord]) {
    println!("## {wave_label}");
    println!();
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
    println!();
}

fn main() {
    let quick = std::env::args().any(|arg| arg == "--quick");
    if quick {
        print_bench_table("Wave 1", &run_all_wave1_benchmarks(&[8], 1, 2));
        print_bench_table("Wave 2", &run_all_wave2_benchmarks(&[8], 1, 2));
        print_bench_table("Wave 3a", &run_all_wave3a_benchmarks(&[8], 1, 2));
        print_bench_table("Wave 3b", &run_all_wave3b_benchmarks(&[8], 1, 2));
        print_bench_table("Wave 4", &run_all_wave4_benchmarks(&[8], 1, 2));
        print_bench_table("Wave 5", &run_all_wave5_benchmarks(&[8], 1, 2));
        return;
    }

    print_bench_table(
        "Wave 1",
        &run_all_wave1_benchmarks(&[128, 1024, 4096], 3, 9),
    );
    print_bench_table(
        "Wave 2",
        &run_all_wave2_benchmarks(&[128, 1024, 4096], 3, 9),
    );
    print_bench_table(
        "Wave 3a",
        &run_all_wave3a_benchmarks(&[128, 1024, 4096], 3, 9),
    );
    print_bench_table(
        "Wave 3b",
        &run_all_wave3b_benchmarks(&[128, 1024, 4096], 3, 9),
    );
    print_bench_table(
        "Wave 4",
        &run_all_wave4_benchmarks(&[128, 512, 1024], 2, 7),
    );
    print_bench_table(
        "Wave 5",
        &run_all_wave5_benchmarks(&[128, 512, 1024], 3, 11),
    );
}
