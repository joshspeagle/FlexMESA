// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone CLI driver for profiling real stellar evolutions, per-step phase
//! breakdowns, `#150` waste-removal components, and `N`-star worker-pool
//! batching (`#168`).
//!
//! Run-level numerics mode (`strict` vs `fast`, `#165`) is read at process
//! startup from `FLEXMESA_NUMERICS_MODE` so each process runs in a single
//! arithmetic mode (`DECISIONS #23`).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_precision_loss,
    clippy::struct_field_names,
    clippy::too_many_lines
)]

mod batch_bench;
mod json_fmt;
mod step_profiler;

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

use batch_bench::run_batch_scaling_json;
use step_profiler::profile_star_case;

fn usage() -> &'static str {
    "Usage:
  profile_evolution case --case-dir <DIR> [--allow-unsupported-controls] \
[--max-steps <N>] [--with-phase-profile] [--ref-evolve-sec <SEC>] \
[--history-out <PATH>] [--logs-dir <DIR>] [--json-out <PATH>]
  profile_evolution batch --case-dir <DIR> [--batch-sizes 1,2,4,8,16] \
[--pool-threads <T>] [--max-steps <N>] [--heterogeneous] [--json-out <PATH>]
  profile_evolution self-test --case-dir <DIR>"
}

fn parse_case_cmd(args: &[String]) -> Result<(), String> {
    let mut case_dir: Option<PathBuf> = None;
    let mut allow_unsupported = false;
    let mut max_steps: Option<usize> = None;
    let mut with_phase_profile = false;
    let mut ref_evolve_sec: Option<f64> = None;
    let mut history_out: Option<PathBuf> = None;
    let mut logs_dir_out: Option<PathBuf> = None;
    let mut json_out: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--case-dir" => {
                i += 1;
                let val = args.get(i).ok_or("--case-dir requires a path")?;
                case_dir = Some(PathBuf::from(val));
            }
            "--allow-unsupported-controls" => allow_unsupported = true,
            "--with-phase-profile" => with_phase_profile = true,
            "--max-steps" => {
                i += 1;
                let val =
                    args.get(i).ok_or("--max-steps requires an integer")?;
                let parsed: usize = val
                    .parse()
                    .map_err(|e| format!("invalid --max-steps '{val}': {e}"))?;
                max_steps = (parsed > 0).then_some(parsed);
            }
            "--ref-evolve-sec" => {
                i += 1;
                let val =
                    args.get(i).ok_or("--ref-evolve-sec requires a float")?;
                let parsed: f64 = val.parse().map_err(|e| {
                    format!("invalid --ref-evolve-sec '{val}': {e}")
                })?;
                ref_evolve_sec = (parsed > 0.0).then_some(parsed);
            }
            "--history-out" => {
                i += 1;
                let val = args.get(i).ok_or("--history-out requires a path")?;
                history_out = Some(PathBuf::from(val));
            }
            "--logs-dir" => {
                i += 1;
                let val = args.get(i).ok_or("--logs-dir requires a path")?;
                logs_dir_out = Some(PathBuf::from(val));
            }
            "--json-out" => {
                i += 1;
                let val = args.get(i).ok_or("--json-out requires a path")?;
                json_out = Some(PathBuf::from(val));
            }
            other => {
                return Err(format!("unknown flag '{other}'\n{}", usage()));
            }
        }
        i += 1;
    }

    let dir =
        case_dir.ok_or_else(|| format!("missing --case-dir\n{}", usage()))?;
    let rec = profile_star_case(
        &dir,
        allow_unsupported,
        max_steps,
        with_phase_profile,
        ref_evolve_sec,
        history_out.as_deref(),
        logs_dir_out.as_deref(),
    )?;
    let json = rec.to_json();
    if let Some(out) = json_out {
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create dir: {e}"))?;
        }
        fs::write(&out, format!("{json}\n"))
            .map_err(|e| format!("failed to write json: {e}"))?;
    } else {
        println!("{json}");
    }
    Ok(())
}

fn parse_batch_cmd(args: &[String]) -> Result<(), String> {
    let mut case_dir: Option<PathBuf> = None;
    let mut batch_sizes: Vec<usize> = vec![1, 2, 4, 8, 16];
    let mut pool_threads: usize = rayon::current_num_threads();
    let mut max_steps: Option<usize> = None;
    let mut heterogeneous = false;
    let mut json_out: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--case-dir" => {
                i += 1;
                let val = args.get(i).ok_or("--case-dir requires a path")?;
                case_dir = Some(PathBuf::from(val));
            }
            "--batch-sizes" => {
                i += 1;
                let val =
                    args.get(i).ok_or("--batch-sizes requires comma list")?;
                batch_sizes = val
                    .split(',')
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| {
                        s.trim().parse::<usize>().map_err(|e| {
                            format!("invalid batch size '{s}': {e}")
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if batch_sizes.is_empty() {
                    return Err(String::from(
                        "--batch-sizes must not be empty",
                    ));
                }
            }
            "--pool-threads" => {
                i += 1;
                let val =
                    args.get(i).ok_or("--pool-threads requires an integer")?;
                pool_threads = val.parse().map_err(|e| {
                    format!("invalid --pool-threads '{val}': {e}")
                })?;
            }
            "--max-steps" => {
                i += 1;
                let val =
                    args.get(i).ok_or("--max-steps requires an integer")?;
                let parsed: usize = val
                    .parse()
                    .map_err(|e| format!("invalid --max-steps '{val}': {e}"))?;
                max_steps = (parsed > 0).then_some(parsed);
            }
            "--heterogeneous" => heterogeneous = true,
            "--json-out" => {
                i += 1;
                let val = args.get(i).ok_or("--json-out requires a path")?;
                json_out = Some(PathBuf::from(val));
            }
            other => {
                return Err(format!("unknown flag '{other}'\n{}", usage()));
            }
        }
        i += 1;
    }

    let dir =
        case_dir.ok_or_else(|| format!("missing --case-dir\n{}", usage()))?;
    let json = run_batch_scaling_json(
        &dir,
        &batch_sizes,
        pool_threads,
        max_steps,
        heterogeneous,
    )?;
    if let Some(out) = json_out {
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create dir: {e}"))?;
        }
        fs::write(&out, format!("{json}\n"))
            .map_err(|e| format!("failed to write json: {e}"))?;
    } else {
        println!("{json}");
    }
    Ok(())
}

fn parse_self_test_cmd(args: &[String]) -> Result<(), String> {
    let mut case_dir: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--case-dir" => {
                i += 1;
                let val = args.get(i).ok_or("--case-dir requires a path")?;
                case_dir = Some(PathBuf::from(val));
            }
            other => {
                return Err(format!("unknown flag '{other}'\n{}", usage()));
            }
        }
        i += 1;
    }
    let dir =
        case_dir.ok_or_else(|| format!("missing --case-dir\n{}", usage()))?;

    // Verify that `--with-phase-profile` leaves the canonical trajectory
    // bit-identical (`0 ULP` on serialized `.photo` checksum) to unperturbed mode.
    let unpert =
        profile_star_case(&dir, true, Some(2), false, None, None, None)?;
    let prof = profile_star_case(
        &dir,
        true,
        Some(2),
        true,
        Some(unpert.evolve_wall_sec),
        None,
        None,
    )?;
    if unpert.photo_checksum != prof.photo_checksum {
        return Err(format!(
            "phase probe perturbed trajectory checksum: {:#x} != {:#x}",
            unpert.photo_checksum, prof.photo_checksum
        ));
    }
    if unpert.steps_taken != 2 || prof.steps_taken != 2 {
        return Err(format!(
            "expected 2 steps in self-test, got {} and {}",
            unpert.steps_taken, prof.steps_taken
        ));
    }
    let ph = prof
        .phases
        .as_ref()
        .ok_or_else(|| String::from("missing phases in profiled run"))?;
    if ph.eos_eval_calls == 0 || ph.linear_solve_calls == 0 {
        return Err(String::from(
            "expected positive EOS and linear solve counts",
        ));
    }
    println!("{}", prof.to_json());
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let Some((sub, rest)) = args.split_first() else {
        return Err(usage().to_string());
    };
    match sub.as_str() {
        "case" => parse_case_cmd(rest),
        "batch" => parse_batch_cmd(rest),
        "self-test" => parse_self_test_cmd(rest),
        "--help" | "-h" | "help" => {
            println!("{}", usage());
            Ok(())
        }
        other => Err(format!("unknown subcommand '{other}'\n{}", usage())),
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("profile_evolution error: {err}");
        process::exit(1);
    }
}
