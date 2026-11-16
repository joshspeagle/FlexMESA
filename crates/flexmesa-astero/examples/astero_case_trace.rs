// SPDX-License-Identifier: LGPL-3.0-or-later
//! Runs one upstream `astero/test_suite/<case>` directory to completion (or to
//! an optional per-stage step cap) and streams incremental MESA-format
//! `LOGS/history.data` and asteroseismic search/frequency tables (`#143`).
//!
//! Usage:
//! `cargo run --release -p flexmesa-astero --example astero_case_trace -- \
//!    <case_dir> <out_history.data | out_dir/> [max_steps_per_stage] \
//!    [--full-stages] [--respect-intervals] [--write-profiles] [--logs-dir]`

use std::path::Path;

use flexmesa_astero::run_astero_test_suite_case_with_logs;
use flexmesa_star::inlist_runner::StarLogsOptions;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: astero_case_trace <case_dir> <out_history.data | out_dir/> \
             [max_steps_per_stage] [--full-stages] [--respect-intervals] \
             [--write-profiles] [--logs-dir]"
        );
        std::process::exit(2);
    }
    let cap = args
        .iter()
        .skip(3)
        .find_map(|s| (!s.starts_with("--")).then(|| s.parse::<usize>().ok()).flatten());
    let skip_optional = !args.iter().any(|a| a == "--full-stages");
    let respect_intervals = args.iter().any(|a| a == "--respect-intervals");
    let write_profiles = args.iter().any(|a| a == "--write-profiles");
    let out_arg = &args[2];
    let out_path = Path::new(out_arg);
    let is_logs_dir = args.iter().any(|a| a == "--logs-dir")
        || out_arg.ends_with('/')
        || out_arg.ends_with(std::path::MAIN_SEPARATOR)
        || out_path.is_dir();

    let logs_opts = StarLogsOptions::from_target(
        out_path,
        is_logs_dir,
        respect_intervals,
        write_profiles,
    );

    let rep = match run_astero_test_suite_case_with_logs(
        Path::new(&args[1]),
        skip_optional,
        cap,
        false,
        logs_opts,
    ) {
        Ok(rep) => rep,
        Err(e) => {
            eprintln!("astero run failed: {e:?}");
            std::process::exit(1);
        }
    };

    println!(
        "case={} tier={} passed={} assertions_passed={} term={} steps={} \
         retries={} runtime={:.1}s",
        rep.case_name,
        rep.tier,
        rep.passed,
        rep.assertions_passed,
        rep.termination_code,
        rep.total_steps,
        rep.total_retries,
        rep.total_runtime_sec
    );
}
