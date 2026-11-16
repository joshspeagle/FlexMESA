// SPDX-License-Identifier: LGPL-3.0-or-later
//! Runs one upstream `star/test_suite/<case>` directory to completion (or to
//! an optional per-stage step cap) and writes one tab-separated row per
//! accepted step, with column names matching upstream MESA `history.data`, so
//! that the trajectory can be compared against a real MESA run.
//!
//! Usage:
//! `cargo run --release -p flexmesa-star --example star_case_trace -- \
//!    <case_dir> <out.tsv> [max_steps_per_stage] [--full-stages]`

use std::fmt::Write as _;
use std::path::Path;

use flexmesa_star::inlist_runner::run_test_suite_case;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: star_case_trace <case_dir> <out.tsv> \
             [max_steps_per_stage] [--full-stages]"
        );
        std::process::exit(2);
    }
    let cap = args.get(3).and_then(|s| s.parse::<usize>().ok());
    let skip_optional = !args.iter().any(|a| a == "--full-stages");
    let rep = match run_test_suite_case(
        Path::new(&args[1]),
        skip_optional,
        cap,
        false,
    ) {
        Ok(rep) => rep,
        Err(e) => {
            eprintln!("run failed: {e:?}");
            std::process::exit(1);
        }
    };
    let mut out = String::from(
        "stage\tmodel_number\tstar_age\tstar_mass\tlog_dt\tlog_LH\tlog_L\t\
         log_Teff\tlog_R\tlog_g\tlog_center_T\tlog_center_Rho\tcenter_h1\t\
         center_he4\the_core_mass\tnum_zones\tnum_retries\t\
         total_num_solver_iterations\n",
    );
    for (i, st) in rep.stages.iter().enumerate() {
        for h in &st.history_rows {
            let _ = writeln!(
                out,
                "{}\t{}\t{:.16e}\t{:.16e}\t{:.16e}\t{:.16e}\t{:.16e}\t\
                 {:.16e}\t{:.16e}\t{:.16e}\t{:.16e}\t{:.16e}\t{:.16e}\t\
                 {:.16e}\t{:.16e}\t{}\t{}\t{}",
                i + 1,
                h.model_number,
                h.star_age,
                h.star_mass,
                h.log_dt,
                h.log_lh,
                h.log_l,
                h.log_teff,
                h.log_r,
                h.log_g,
                h.log_center_t,
                h.log_center_rho,
                h.center_h1,
                h.center_he4,
                h.he_core_mass,
                h.num_zones,
                h.num_retries,
                h.total_num_solver_iterations,
            );
        }
    }
    if let Err(e) = std::fs::write(&args[2], out) {
        eprintln!("failed to write {}: {e}", args[2]);
        std::process::exit(1);
    }
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
