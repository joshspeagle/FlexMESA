// SPDX-License-Identifier: LGPL-3.0-or-later
//! Runs one upstream `binary/test_suite/<case>` directory to completion (or to
//! an optional step cap) and streams incremental MESA-format
//! `binary_history.data` (plus component `LOGS1/` / `LOGS2/` files when
//! targeting an output directory) on every accepted step (`#163`).
//!
//! Usage:
//! `cargo run --release -p flexmesa-binary --example binary_case_trace -- \
//!    <case_dir> <out_binary_history.data | out_dir/> [max_steps] \
//!    [--respect-intervals] [--logs-dir] [--allow-unsupported-controls]`

use std::path::Path;

use flexmesa_binary::{
    run_binary_test_suite_case_with_options, BinaryLogsOptions,
};
use flexmesa_star::InlistLoadOptions;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: binary_case_trace <case_dir> \
             <out_binary_history.data | out_dir/> [max_steps] \
             [--respect-intervals] [--logs-dir] [--allow-unsupported-controls]"
        );
        std::process::exit(2);
    }
    let cap = args.iter().skip(3).find_map(|s| {
        (!s.starts_with("--"))
            .then(|| s.parse::<usize>().ok())
            .flatten()
    });
    let respect_intervals = args.iter().any(|a| a == "--respect-intervals");
    let allow_unsupported_controls =
        args.iter().any(|a| a == "--allow-unsupported-controls");
    let out_arg = &args[2];
    let out_path = Path::new(out_arg);
    let is_logs_dir = args.iter().any(|a| a == "--logs-dir")
        || out_arg.ends_with('/')
        || out_arg.ends_with(std::path::MAIN_SEPARATOR)
        || out_path.is_dir();

    let logs_opts = if is_logs_dir {
        BinaryLogsOptions {
            output_root: Some(out_path),
            explicit_binary_history_file: None,
            history_interval_override: (!respect_intervals).then_some(1),
            write_star_logs: true,
        }
    } else {
        BinaryLogsOptions {
            output_root: None,
            explicit_binary_history_file: Some(out_path),
            history_interval_override: (!respect_intervals).then_some(1),
            write_star_logs: false,
        }
    };

    let rep = match run_binary_test_suite_case_with_options(
        Path::new(&args[1]),
        cap,
        logs_opts,
        InlistLoadOptions {
            allow_unsupported_controls,
        },
    ) {
        Ok(rep) => rep,
        Err(e) => {
            eprintln!("binary run failed: {e:?}");
            std::process::exit(1);
        }
    };

    println!(
        "case={} passed={} term={} steps={} m1={:.6} m2={:.6} sep={:.6} \
         period_d={:.6} runtime={:.2}s",
        rep.case_name,
        rep.passed,
        rep.termination_code,
        rep.total_steps,
        rep.final_m1_msun,
        rep.final_m2_msun,
        rep.final_separation_rsun,
        rep.final_period_days,
        rep.wall_time_ms / 1000.0
    );
}
