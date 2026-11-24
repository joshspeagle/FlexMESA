// SPDX-License-Identifier: GPL-3.0-or-later
//! The step dump (`FLEXMESA_STEPDUMP`, `#372`) of the original binary RLOF
//! fixture's model 1 reproduces the instrumented MESA `26.4.1` tables
//! (`benchmarks/fortran_mesa/binary_rlof_fixture/instrumented/`): the donor's
//! `lnd lnT lnR L` at the first solver call bit for bit in all zones (the
//! `rho = dm/dV` finding of `#363`), and MESA's 44 solver calls with the same
//! `dt` sequence (`#358`, `#363`).
//!
//! The dump is read through its own records, so this also checks the record
//! writer against MESA. Composition is not compared: it still differs from
//! MESA by about 4e-13 relative after ZAMS interpolation (`#360`).
//!
//! This file holds a single test: the dump's settings are read from the
//! environment once per process.

#![forbid(unsafe_code)]

mod common;

use std::path::PathBuf;

use flexmesa_binary::DefaultBinaryExtras;
use flexmesa_const::SECYER;

/// Reads a reference table: rows of hex bit patterns after the `#` header
/// and the column-name line, each row as its tab-separated fields.
fn read_table(name: &str) -> Vec<Vec<String>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/fortran_mesa/binary_rlof_fixture/instrumented")
        .join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("missing MESA reference {}: {e}", path.display())
    });
    text.lines()
        .filter(|l| !l.starts_with('#'))
        .skip(1)
        .map(|l| l.split('\t').map(str::to_string).collect())
        .collect()
}

fn hex(s: &str) -> u64 {
    u64::from_str_radix(s, 16).expect("hex bit pattern")
}

#[test]
fn rlof_fixture_model1_stepdump_matches_mesa_26_4_1_state_and_dt_sequence() {
    let dump = std::env::temp_dir()
        .join(format!("flexmesa-rlof-stepdump-{}.txt", std::process::id()));
    let _ = std::fs::remove_file(&dump);
    std::env::set_var("FLEXMESA_STEPDUMP", &dump);
    std::env::set_var("FLEXMESA_STEPDUMP_STEPS", "1");

    let mut sys = common::build_rlof_binary_original();
    sys.try_step_with_extras(1.0e3 * SECYER, &mut DefaultBinaryExtras)
        .expect("model 1");
    let text = std::fs::read_to_string(&dump).expect("step dump written");
    let _ = std::fs::remove_file(&dump);

    // Donor (star 1) records of its first model.
    let recs: Vec<Vec<&str>> = text
        .lines()
        .map(|l| l.split_whitespace().collect::<Vec<_>>())
        .filter(|p| p.len() > 3 && p[1] == "1")
        .collect();

    let mesa_state = read_table("first_call_state_26_4_1.tsv");
    let state: Vec<&Vec<&str>> =
        recs.iter().filter(|p| p[0] == "STATE").collect();
    assert_eq!(state.len(), mesa_state.len(), "zones in STATE records");
    let names = ["lnd", "lnT", "lnR", "L"];
    for (j, name) in names.iter().enumerate() {
        let diff: Vec<usize> = state
            .iter()
            .zip(&mesa_state)
            .filter(|(p, m)| hex(p[4 + j]) != hex(&m[1 + j]))
            .map(|(p, _)| p[3].parse().expect("zone"))
            .collect();
        assert!(
            diff.is_empty(),
            "{name} at the first call differs from MESA 26.4.1 in {} zones, \
             first at k = {:?}",
            diff.len(),
            diff.first()
        );
    }

    let mesa_calls: Vec<Vec<String>> = read_table("solver_calls_26_4_1.tsv")
        .into_iter()
        .filter(|r| r[1] == "1")
        .collect();
    let calls: Vec<&Vec<&str>> = recs.iter().filter(|p| p[0] == "CALL").collect();
    assert_eq!(calls.len(), 44, "MESA 26.4.1: 44 donor solver calls in model 1");
    assert_eq!(calls.len(), mesa_calls.len());
    for (i, (p, m)) in calls.iter().zip(&mesa_calls).enumerate() {
        assert_eq!(hex(p[3]), hex(&m[2]), "dt of solver call {}", i + 1);
    }
}
