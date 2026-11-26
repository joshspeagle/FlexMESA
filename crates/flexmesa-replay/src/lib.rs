// SPDX-License-Identifier: GPL-3.0-or-later
//! Record/replay routine oracles from an instrumented MESA 26.4.1 (`#552`).
//!
//! An overlay build of MESA 26.4.1 (`benchmarks/fortran_mesa/replay/`)
//! appends one record per call of a chosen routine during a real run: the
//! call context and every input and output as IEEE-754 bit patterns. This
//! crate reads such a capture, calls the matching `FlexMESA` port with
//! exactly the captured inputs, and compares the outputs bit for bit,
//! reporting the first differing call and output component.
//!
//! Families replayed:
//! - `TDC`: `set_TDC` (`turb/public/turb.f90:114-204`) through
//!   [`flexmesa_turb::set_tdc`];
//! - `MLT`: `set_MLT` (`turb/public/turb.f90:275-292`) through
//!   [`flexmesa_turb::set_mlt`];
//! - `EOS`: the cell `get_eos` call of `do_eos_for_cell`
//!   (`star/private/micro.f90:357-361`, `eosDT_get`) through
//!   [`flexmesa_eos::eos_dt_get_zones_with_logs`], configured from the
//!   run's `CONFIG EOS` line ([`eos`]).
//!
//! Records of a caller's arguments (`GETR`, `Get_results`) depend on star
//! state and cannot be replayed stand-alone; [`argdiff`] instead compares a
//! MESA capture with the same records written by `FlexMESA`'s own tap
//! (`flexmesa-star`, `FLEXMESA_REPLAY_TAP`), call by call.
//!
//! The comparison is strict-mode only: replay runs under the default
//! numerics mode of `flexmesa-math`.

#![forbid(unsafe_code)]

pub mod argdiff;
pub mod compare;
pub mod eos;
pub mod record;
pub mod turb;

use std::collections::{BTreeMap, BTreeSet};
use std::io::BufRead;

use compare::FamilySummary;
use record::{CaptureReader, FormatError};
use turb::Replayed;

/// Summary of a replayed capture, by family.
#[derive(Debug, Default)]
pub struct ReplaySummary {
    /// Provenance header of the capture.
    pub header: Vec<String>,
    /// Per-family results.
    pub families: BTreeMap<String, FamilySummary>,
    /// Records compared with [`argdiff::compare_captures`] rather than
    /// replayed (`GETR`).
    pub argument_records: usize,
}

impl ReplaySummary {
    /// True when every record was replayed and all outputs are identical.
    #[must_use]
    pub fn all_identical(&self) -> bool {
        self.families
            .values()
            .all(|f| f.identical == f.calls && f.not_replayed.is_empty())
    }
}

/// Replays every record of a capture.
///
/// # Errors
/// Returns [`FormatError`] for a malformed capture, a schema this tool does
/// not implement, or a record of an unknown family.
pub fn replay<R: BufRead>(reader: R) -> Result<ReplaySummary, FormatError> {
    let mut cap = CaptureReader::new(reader);
    let mut summary = ReplaySummary::default();
    let mut checked: BTreeSet<String> = BTreeSet::new();
    // EOS configuration of the latest `CONFIG EOS` line, by its sequence
    // number, and the HELM table (loaded on the first EOS record).
    let mut eos_cfg: Option<(usize, flexmesa_eos::EosConfig)> = None;
    let mut helm: Option<&'static flexmesa_eos::HelmTable> = None;
    while let Some(rec) = cap.next_record()? {
        if !checked.contains(&rec.family) {
            let expected = match rec.family.as_str() {
                "TDC" => turb::TDC_SCHEMA,
                "MLT" => turb::MLT_SCHEMA,
                "GETR" => turb::GETR_SCHEMA,
                "EOS" => eos::EOS_SCHEMA,
                other => {
                    return Err(FormatError(format!(
                        "line {}: no replay for record family `{other}`",
                        rec.line
                    )))
                }
            };
            if let Some(schema) = cap.schema(&rec.family) {
                schema.check(&rec.family, expected)?;
            }
            checked.insert(rec.family.clone());
        }
        let outcome = match rec.family.as_str() {
            "TDC" => turb::replay_tdc(&rec)?,
            "MLT" => turb::replay_mlt(&rec)?,
            "EOS" => {
                let (seq, line) = cap.config("EOS").ok_or_else(|| {
                    FormatError(format!(
                        "line {}: EOS record before any CONFIG EOS line",
                        rec.line
                    ))
                })?;
                let cfg = match &mut eos_cfg {
                    Some((s, c)) if *s == seq => &*c,
                    slot => {
                        let c = eos::eos_config_from_config(
                            line.split_whitespace(),
                        )?;
                        &slot.insert((seq, c)).1
                    }
                };
                let ht = if let Some(h) = helm {
                    h
                } else {
                    let h = flexmesa_eos::HelmTable::builtin()
                        .map_err(|e| FormatError(format!("HELM table: {e}")))?;
                    helm = Some(h);
                    h
                };
                eos::replay_eos(&rec, cfg, ht)?
            }
            _ => {
                summary.argument_records += 1;
                continue;
            }
        };
        let fam = summary.families.entry(rec.family.clone()).or_default();
        match outcome {
            Replayed::Compared(diff) => fam.add(&rec, diff),
            Replayed::NotReplayed(why) => {
                fam.not_replayed.push(format!("{}: {why}", rec.context()));
            }
        }
    }
    summary.header = cap.header;
    Ok(summary)
}
