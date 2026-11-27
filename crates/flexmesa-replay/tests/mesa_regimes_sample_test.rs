// SPDX-License-Identifier: GPL-3.0-or-later
//! Replay of committed samples of MESA `26.4.1` captures covering microphysics
//! and MLT regimes omitted from `mesa_sample_test` (`#552`, `#591`).
//!
//! Fixtures under `tests/data/`:
//! - `wd_cool_0p6m_sample.replay`: `72` `EOS` (`22` `OPAL_SCVH`, `26`
//!   `OPAL_SCVH+FreeEOS`, `12` `FreeEOS+Skye`, `12` `Skye` with
//!   `Skye_solid_mixing_rule='Ogata'`) and `80` `MLT` (`MLT_option='ML2'`).
//! - `ns_h_simple_skye_sample.replay`: `60` `EOS` with
//!   `use_simple_Skye_blends=T` and non-default Skye density/temperature
//!   bounds.
//! - `split_burn_henyey_80sp_sample.replay`: `24` `80`-species `EOS` (nonzero
//!   `d_dxa`) and `50` `MLT` (`MLT_option='Henyey'`).
//! - `tzo_cox_mlt_sample.replay`: `40` `MLT` (`MLT_option='Cox'`,
//!   `DualOrder1<33>`).

#![forbid(unsafe_code)]

use std::io::Cursor;

const WD_COOL: &str = include_str!("data/wd_cool_0p6m_sample.replay");
const NS_H: &str = include_str!("data/ns_h_simple_skye_sample.replay");
const SPLIT_BURN: &str =
    include_str!("data/split_burn_henyey_80sp_sample.replay");
const TZO: &str = include_str!("data/tzo_cox_mlt_sample.replay");

fn replay(text: &str) -> flexmesa_replay::ReplaySummary {
    flexmesa_replay::replay(Cursor::new(text)).expect("sample replays")
}

fn first_record_line(lines: &[&str], prefix: &str) -> usize {
    lines
        .iter()
        .position(|line| line.starts_with(prefix))
        .expect("sample has expected record prefix")
}

fn assert_replay_ok(text: &str, expected: &[(&str, usize)]) {
    let summary = replay(text);
    assert!(summary.header.iter().any(|h| h == "mesa_version: r26.4.1"));
    assert!(summary
        .header
        .iter()
        .any(|h| h == "compiler: gfortran 15.2.0"));
    assert!(summary
        .header
        .iter()
        .any(|h| h == "mesa_sdk_version: x86_64-linux-26.6.1"));
    assert_eq!(summary.families.len(), expected.len());
    for (family, count) in expected {
        let fam = &summary.families[*family];
        assert_eq!(fam.calls, *count, "{family} records replayed");
        assert_eq!(fam.identical, *count, "{family}: {:?}", fam.first);
        assert!(
            fam.not_replayed.is_empty(),
            "{family}: {:?}",
            fam.not_replayed
        );
    }
    assert!(summary.all_identical());
}

#[test]
fn wd_cool_sample_replays_bit_identically_for_opal_scvh_ogata_skye_and_ml2() {
    assert_replay_ok(WD_COOL, &[("EOS", 72), ("MLT", 80)]);
}

#[test]
fn ns_h_simple_skye_sample_replays_bit_identically() {
    assert_replay_ok(NS_H, &[("EOS", 60)]);
}

#[test]
fn split_burn_sample_replays_bit_identically_for_80sp_eos_and_henyey_mlt() {
    assert_replay_ok(SPLIT_BURN, &[("EOS", 24), ("MLT", 50)]);
}

#[test]
fn tzo_cox_mlt_sample_replays_bit_identically() {
    assert_replay_ok(TZO, &[("MLT", 40)]);
}

#[test]
fn flipped_eos_d_dxa_bit_is_reported() {
    let mut lines: Vec<&str> = SPLIT_BURN.lines().collect();
    let idx = first_record_line(&lines, "EOS ");
    let mut fields: Vec<String> =
        lines[idx].split(' ').map(ToString::to_string).collect();
    // Field 15 is `out_d_dxa`; element 0 is d_dxa(lnPgas, species 1).
    let mut d_dxa: Vec<String> =
        fields[15].split(',').map(ToString::to_string).collect();
    let bits = u64::from_str_radix(&d_dxa[0], 16).unwrap() ^ 1;
    d_dxa[0] = format!("{bits:016X}");
    fields[15] = d_dxa.join(",");
    let mutated = fields.join(" ");
    lines[idx] = &mutated;
    let summary = replay(&(lines.join("\n") + "\n"));
    let eos = &summary.families["EOS"];
    assert_eq!(eos.identical, eos.calls - 1);
    let (_, diff) = eos.first.as_ref().expect("one differing EOS call");
    assert_eq!(diff.mismatches.len(), 1);
    assert_eq!(diff.mismatches[0].field, "d_dxa(lnPgas, species 1)");
}

#[test]
fn flipped_mlt_output_bit_is_reported() {
    let mut lines: Vec<&str> = TZO.lines().collect();
    let idx = first_record_line(&lines, "MLT ");
    let mut fields: Vec<String> =
        lines[idx].split(' ').map(ToString::to_string).collect();
    // Field 24 is `out_gradT` in sparse `<val>/<lane>:<hex>...` notation.
    fields[24] = if let Some((val_hex, rest)) = fields[24].split_once('/') {
        let bits = u64::from_str_radix(val_hex, 16).unwrap() ^ 1;
        format!("{bits:016X}/{rest}")
    } else {
        let bits = u64::from_str_radix(&fields[24], 16).unwrap() ^ 1;
        format!("{bits:016X}")
    };
    let mutated = fields.join(" ");
    lines[idx] = &mutated;
    let summary = replay(&(lines.join("\n") + "\n"));
    let mlt = &summary.families["MLT"];
    assert_eq!(mlt.identical, mlt.calls - 1);
    let (_, diff) = mlt.first.as_ref().expect("one differing MLT call");
    assert_eq!(diff.mismatches.len(), 1);
    assert_eq!(diff.mismatches[0].field, "gradT");
    assert_eq!(diff.mismatches[0].lane, None);
}

#[test]
fn toggling_simple_skye_blends_config_changes_ns_h_blend_results() {
    let toggled = NS_H.replacen(
        "use_simple_Skye_blends=T",
        "use_simple_Skye_blends=F",
        1,
    );
    assert_ne!(toggled, NS_H);
    let summary = replay(&toggled);
    let eos = &summary.families["EOS"];
    assert!(
        eos.identical < eos.calls,
        "expected simple_Skye_blends toggle to affect ns_h FreeEOS+Skye blends"
    );

    let mut lines: Vec<&str> = NS_H.lines().collect();
    let cfg_idx = first_record_line(&lines, "CONFIG EOS ");
    let bad_cfg = format!("{} not_a_control=T", lines[cfg_idx]);
    lines[cfg_idx] = &bad_cfg;
    let text = lines.join("\n") + "\n";
    let err = flexmesa_replay::replay(Cursor::new(text));
    assert!(err.is_err(), "unknown CONFIG EOS control accepted");
}
