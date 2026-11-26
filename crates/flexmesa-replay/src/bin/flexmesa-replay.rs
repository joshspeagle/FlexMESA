// SPDX-License-Identifier: GPL-3.0-or-later
//! Command-line replay of a MESA 26.4.1 capture (`#552`).
//!
//! Usage: `flexmesa-replay <capture file> [--all-diffs]`
//!
//! Prints, per record family, the number of calls replayed and identical,
//! per-output tallies of differing calls with their largest ulp distance,
//! the first differing call in file order with every differing component,
//! and the earliest differing call by `(run, model, solver_iter, k)`.
//! Exits with status 1 when any call differs or could not be replayed, and
//! 2 on a malformed capture.

#![forbid(unsafe_code)]

use std::fs::File;
use std::io::BufReader;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.iter().find(|a| !a.starts_with("--")) else {
        eprintln!("usage: flexmesa-replay <capture file> [--all-diffs]");
        return ExitCode::from(2);
    };
    let all = args.iter().any(|a| a == "--all-diffs");
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("cannot open {path}: {e}");
            return ExitCode::from(2);
        }
    };
    let summary = match flexmesa_replay::replay(BufReader::new(file)) {
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
            let shown = if all { diff.mismatches.len() } else { 8 };
            for d in &diff.discrete {
                println!("    {d}");
            }
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
        if let Some((_, ctx)) = &f.earliest {
            println!("  earliest differing call: {ctx}");
        }
        for nr in f.not_replayed.iter().take(5) {
            println!("  not replayed: {nr}");
        }
    }
    if summary.all_identical() {
        println!("all replayed calls identical");
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
