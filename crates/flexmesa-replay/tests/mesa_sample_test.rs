// SPDX-License-Identifier: GPL-3.0-or-later
//! Replay of a committed sample of real MESA 26.4.1 records (`#552`).
//!
//! `tests/data/accreted_material_j_sample.replay` holds 80 `TDC`, 60 `MLT`
//! and 120 `EOS` records (with the run's `CONFIG EOS` line) taken from a
//! capture of `star/accreted_material_j` by the overlay of
//! `benchmarks/fortran_mesa/replay/` (MESA r26.4.1, gfortran 15.2.0, SDK
//! x86_64-linux-26.6.1; first two models, records evenly spaced in file
//! order). Every record replays bit for bit through the `FlexMESA` ports,
//! so CI exercises the replay path, including the EOS configuration built
//! from MESA's own handle, without the full capture bundle. The negative
//! controls check that one flipped bit in a captured EOS output or input is
//! reported, and that a malformed `CONFIG EOS` line is refused.

#![forbid(unsafe_code)]

use std::io::Cursor;

const SAMPLE: &str = include_str!("data/accreted_material_j_sample.replay");

fn replay(text: &str) -> flexmesa_replay::ReplaySummary {
    flexmesa_replay::replay(Cursor::new(text)).expect("sample replays")
}

/// Index of the first `EOS` record line of the sample.
fn first_eos_line(lines: &[&str]) -> usize {
    lines
        .iter()
        .position(|l| l.starts_with("EOS "))
        .expect("sample has EOS records")
}

#[test]
fn mesa_sample_replays_bit_identically_for_tdc_mlt_and_eos() {
    let s = replay(SAMPLE);
    assert!(s.header.iter().any(|h| h == "mesa_version: r26.4.1"));
    assert!(s.header.iter().any(|h| h == "compiler: gfortran 15.2.0"));
    for (family, n) in [("TDC", 80), ("MLT", 60), ("EOS", 120)] {
        let f = &s.families[family];
        assert_eq!(f.calls, n, "{family} records replayed");
        assert_eq!(f.identical, n, "{family}: {:?}", f.first);
        assert!(f.not_replayed.is_empty(), "{family}: {:?}", f.not_replayed);
    }
    assert!(s.all_identical());
}

#[test]
fn flipped_eos_output_bit_is_reported_with_its_result_name() {
    let mut lines: Vec<&str> = SAMPLE.lines().collect();
    let i = first_eos_line(&lines);
    let mut fields: Vec<String> =
        lines[i].split(' ').map(ToString::to_string).collect();
    // Field 12 is `out_res`; element 2 is res(i_lnE).
    let mut res: Vec<String> =
        fields[12].split(',').map(ToString::to_string).collect();
    let bits = u64::from_str_radix(&res[1], 16).unwrap() ^ 1;
    res[1] = format!("{bits:016X}");
    fields[12] = res.join(",");
    let mutated = fields.join(" ");
    lines[i] = &mutated;
    let s = replay(&(lines.join("\n") + "\n"));
    let eos = &s.families["EOS"];
    assert_eq!(eos.identical, eos.calls - 1);
    let (_, diff) = eos.first.as_ref().expect("one differing call");
    assert_eq!(diff.mismatches.len(), 1);
    assert_eq!(diff.mismatches[0].field, "res(lnE)");
}

#[test]
fn perturbed_eos_input_changes_the_replayed_results() {
    let mut lines: Vec<&str> = SAMPLE.lines().collect();
    let i = first_eos_line(&lines);
    let mut fields: Vec<String> =
        lines[i].split(' ').map(ToString::to_string).collect();
    // Field 11 is `logT`.
    let bits = u64::from_str_radix(&fields[11], 16).unwrap() ^ (1 << 20);
    fields[11] = format!("{bits:016X}");
    let mutated = fields.join(" ");
    lines[i] = &mutated;
    let s = replay(&(lines.join("\n") + "\n"));
    assert_eq!(s.families["EOS"].identical, s.families["EOS"].calls - 1);
}

#[test]
fn config_line_with_unknown_missing_or_unsupported_control_is_refused() {
    let lines: Vec<&str> = SAMPLE.lines().collect();
    let i = lines
        .iter()
        .position(|l| l.starts_with("CONFIG EOS "))
        .expect("sample has a CONFIG EOS line");
    let cases = [
        format!("{} not_a_control=T", lines[i]),
        lines[i].replace(" use_PC=F", ""),
        lines[i]
            .replace("use_other_eos_component=F", "use_other_eos_component=T"),
        lines[i]
            .replace("eosDT_file_prefix='mesa'", "eosDT_file_prefix='other'"),
    ];
    for (n, bad) in cases.iter().enumerate() {
        assert_ne!(bad, lines[i], "case {n} must change the line");
        let mut v = lines.clone();
        v[i] = bad;
        let text = v.join("\n") + "\n";
        let err = flexmesa_replay::replay(Cursor::new(text));
        assert!(err.is_err(), "case {n} accepted");
    }
}
