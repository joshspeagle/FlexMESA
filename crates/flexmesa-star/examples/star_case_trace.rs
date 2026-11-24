// SPDX-License-Identifier: GPL-3.0-or-later
//! Runs one upstream `star/test_suite/<case>` directory to completion (or to
//! an optional per-stage step cap) and streams incremental MESA-format
//! `history.data` (plus `profileN.data` and `profiles.index` when targeting an
//! output directory) on every accepted step so killed or failed runs preserve
//! partial history on disk (`#163`).
//!
//! Usage:
//! `cargo run --release -p flexmesa-star --example star_case_trace -- \
//!    <case_dir> <out_history.data | out_dir/> [max_steps_per_stage] \
//!    [--full-stages] [--respect-intervals] [--write-profiles] [--logs-dir]`

use std::path::Path;

use flexmesa_star::inlist_runner::{
    run_test_suite_case_with_logs, StarLogsOptions,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: star_case_trace <case_dir> <out_history.data | out_dir/> \
             [max_steps_per_stage] [--full-stages] [--respect-intervals] \
             [--write-profiles] [--logs-dir]"
        );
        std::process::exit(2);
    }
    let cap = args.iter().skip(3).find_map(|s| {
        (!s.starts_with("--"))
            .then(|| s.parse::<usize>().ok())
            .flatten()
    });
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

    let rep = match run_test_suite_case_with_logs(
        Path::new(&args[1]),
        skip_optional,
        cap,
        false,
        logs_opts,
    ) {
        Ok(rep) => rep,
        Err(e) => {
            eprintln!("run failed: {e:?}");
            std::process::exit(1);
        }
    };

    for (i, st) in rep.stages.iter().enumerate() {
        println!(
            "stage {} inlist={} steps={} retries={} newton={} term={} \
             runtime={:.1}s",
            i + 1,
            st.inlist_name,
            st.steps_taken,
            st.total_retries,
            st.total_newton_iterations,
            st.termination_code,
            st.runtime_sec
        );
    }
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
