// SPDX-License-Identifier: GPL-3.0-or-later
//! Command-line replay and comparison of MESA 26.4.1 captures (`#552`).
//!
//! Usage:
//!
//! - `flexmesa-replay <capture> [--all-diffs]`: replays every kernel record
//!   (`TDC`, `MLT`) of a MESA capture through the `FlexMESA` ports. Prints,
//!   per family, the calls replayed and identical, per-output tallies with
//!   their largest ulp distance, every differing component of the first
//!   differing call in file order, and the earliest differing call by
//!   `(run, model, solver_iter, k)`.
//! - `flexmesa-replay compare <MESA capture> <FlexMESA capture>
//!   [--all-diffs]`: matches the records of a MESA capture with those
//!   `FlexMESA`'s tap wrote for the same case and compares every common
//!   field bit for bit. Prints per-family match counts, per-field tallies
//!   with the earliest differing call of each field, and every differing
//!   field of the earliest differing call.
//!
//! Exit status: 0 when everything is identical, 1 on any difference, 2 on
//! a malformed capture or bad usage.

#![forbid(unsafe_code)]

use std::fs::File;
use std::io::BufReader;
use std::process::ExitCode;

use flexmesa_replay::argdiff::{compare_captures, RecordDiff};
use flexmesa_replay::compare::CallDiff;

fn open(path: &str) -> Result<BufReader<File>, ExitCode> {
    File::open(path).map(BufReader::new).map_err(|e| {
        eprintln!("cannot open {path}: {e}");
        ExitCode::from(2)
    })
}

fn print_diff(diff: &CallDiff, all: bool) {
    for d in &diff.discrete {
        println!("    {d}");
    }
    let shown = if all { diff.mismatches.len() } else { 12 };
    for m in diff.mismatches.iter().take(shown) {
        println!("    {}", m.describe());
    }
    if diff.mismatches.len() > shown {
        println!(
            "    ... {} more components (--all-diffs)",
            diff.mismatches.len() - shown
        );
    }
}

fn run_replay(path: &str, all: bool) -> ExitCode {
    let reader = match open(path) {
        Ok(r) => r,
        Err(code) => return code,
    };
    let summary = match flexmesa_replay::replay(reader) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    for h in &summary.header {
        println!("# {h}");
    }
    for (family, f) in &summary.families {
        println!(
            "{family}: {} calls replayed, {} identical, {} differ, {} not replayed",
            f.calls,
            f.identical,
            f.calls - f.identical,
            f.not_replayed.len()
        );
        for (field, t) in &f.fields {
            println!(
                "  {field}: val differs in {} calls (max {} ulp), \
                 partials differ in {} calls (max {} ulp)",
                t.val_calls, t.val_max_ulp, t.d1_calls, t.d1_max_ulp
            );
        }
        for (field, n) in &f.discrete {
            println!("  {field}: differs in {n} calls");
        }
        if let Some((ctx, diff)) = &f.first {
            println!("  first differing call: {ctx}");
            print_diff(diff, all);
        }
        if let Some((_, ctx)) = &f.earliest {
            println!("  earliest differing call: {ctx}");
        }
        for nr in f.not_replayed.iter().take(5) {
            println!("  not replayed: {nr}");
        }
    }
    if summary.argument_records > 0 {
        println!(
            "{} argument records (GETR) not replayed; compare them with \
             `flexmesa-replay compare`",
            summary.argument_records
        );
    }
    if summary.all_identical() {
        println!("all replayed calls identical");
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

fn run_compare(mesa: &str, flex: &str, all: bool) -> ExitCode {
    let (a, b) = match (open(mesa), open(flex)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(code), _) | (_, Err(code)) => return code,
    };
    let s = match compare_captures(a, b) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    for h in &s.mesa_header {
        println!("# MESA: {h}");
    }
    for h in &s.flexmesa_header {
        println!("# FlexMESA: {h}");
    }
    for (family, f) in &s.families {
        println!(
            "{family}: {} matched calls, {} identical, {} differ; \
             {} only in MESA, {} only in FlexMESA",
            f.matched,
            f.identical,
            f.matched - f.identical,
            f.mesa_only.0,
            f.flexmesa_only.0
        );
        if f.beyond_window != (0, 0) {
            println!(
                "  outside the other side's window (not compared): {} MESA, \
                 {} FlexMESA calls",
                f.beyond_window.0, f.beyond_window.1
            );
        }
        if let Some(c) = &f.mesa_only.1 {
            println!("  first only in MESA: {c}");
        }
        if let Some(c) = &f.flexmesa_only.1 {
            println!("  first only in FlexMESA: {c}");
        }
    }
    for n in &s.sweep_notes {
        println!("call counts differ: {n}");
    }
    if s.signed_zero_calls > 0 {
        println!(
            "{} matched calls identical except for the sign of zero \
             components (+0.0 against -0.0; counted as identical)",
            s.signed_zero_calls
        );
    }
    for o in &s.one_sided_fields {
        println!("field written by one side only: {o}");
    }
    for (u, n) in &s.unavailable {
        println!("field not available: {u} in {n} calls");
    }
    if !s.fields.is_empty() {
        println!(
            "differing fields (calls with the value differing / with only \
             partials differing; earliest call):"
        );
    }
    for (field, st) in &s.fields {
        println!(
            "  {field}: value {} (max {} ulp), partials only {}",
            st.val_calls, st.val_max_ulp, st.d1_calls
        );
        if let Some((_, c, d)) = &st.earliest {
            println!("    earliest at {c}: {d}");
        }
    }
    if let Some((
        _,
        c,
        RecordDiff {
            diff,
            unavailable,
            signed_zero,
        },
    )) = &s.first
    {
        println!("earliest differing call: {c}");
        print_diff(diff, all);
        for u in unavailable {
            println!("    not available: {u}");
        }
        if *signed_zero > 0 {
            println!("    (+ {signed_zero} components differing only in the sign of zero)");
        }
    }
    if s.all_identical() {
        println!("all matched calls identical");
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let all = args.iter().any(|a| a == "--all-diffs");
    let pos: Vec<&str> = args
        .iter()
        .filter(|a| !a.starts_with("--"))
        .map(String::as_str)
        .collect();
    match pos.as_slice() {
        ["compare", mesa, flex] => run_compare(mesa, flex, all),
        [path] => run_replay(path, all),
        _ => {
            eprintln!(
                "usage: flexmesa-replay <capture> [--all-diffs]\n       \
                 flexmesa-replay compare <MESA capture> <FlexMESA capture> \
                 [--all-diffs]"
            );
            ExitCode::from(2)
        }
    }
}
