// SPDX-License-Identifier: GPL-3.0-or-later
//! Bit-for-bit comparison of replayed outputs with captured MESA outputs.

use std::collections::BTreeMap;

use crate::record::{MesaAd, Record, STAR_AD_LANES};

/// Distance in units in the last place between two doubles, ordering the
/// bit patterns as signed magnitudes (`+0.0` and `-0.0` are one apart).
/// `u64::MAX` when exactly one is NaN.
#[must_use]
pub fn ulp_distance(a: f64, b: f64) -> u64 {
    if a.is_nan() || b.is_nan() {
        return if a.is_nan() && b.is_nan() {
            0
        } else {
            u64::MAX
        };
    }
    let key = |x: f64| -> i128 {
        let bits = x.to_bits();
        let mag = i128::from(bits & !(1_u64 << 63));
        if bits >> 63 == 0 {
            mag
        } else {
            -mag - 1
        }
    };
    u64::try_from((key(a) - key(b)).unsigned_abs()).unwrap_or(u64::MAX)
}

/// One output component that differs between MESA and `FlexMESA`.
#[derive(Debug, Clone, PartialEq)]
pub struct Mismatch {
    /// Output field name (schema name).
    pub field: String,
    /// `None` for the value, `Some(lane)` (1-based, MESA numbering) for a
    /// `d1Array` partial.
    pub lane: Option<usize>,
    /// MESA's value.
    pub mesa: f64,
    /// `FlexMESA`'s value.
    pub flexmesa: f64,
}

impl Mismatch {
    /// One-line description with both bit patterns.
    #[must_use]
    pub fn describe(&self) -> String {
        let comp = self
            .lane
            .map_or_else(|| "val".to_string(), |l| format!("d1Array({l})"));
        format!(
            "{}%{}: MESA {:016X} ({:e}) FlexMESA {:016X} ({:e}), {} ulp",
            self.field,
            comp,
            self.mesa.to_bits(),
            self.mesa,
            self.flexmesa.to_bits(),
            self.flexmesa,
            ulp_distance(self.mesa, self.flexmesa)
        )
    }
}

/// Collects the differing components of one replayed call.
#[derive(Debug, Default)]
pub struct CallDiff {
    /// Every differing output component.
    pub mismatches: Vec<Mismatch>,
    /// Integer or flag outputs that differ, as `field: MESA x FlexMESA y`.
    pub discrete: Vec<String>,
}

impl CallDiff {
    /// Compares a real output bit for bit.
    pub fn real(&mut self, field: &str, mesa: f64, flexmesa: f64) {
        if mesa.to_bits() != flexmesa.to_bits() {
            self.mismatches.push(Mismatch {
                field: field.to_string(),
                lane: None,
                mesa,
                flexmesa,
            });
        }
    }

    /// Compares an auto-diff output's value and all lanes bit for bit.
    pub fn ad(&mut self, field: &str, mesa: &MesaAd, flexmesa: &MesaAd) {
        self.real(field, mesa.val, flexmesa.val);
        for lane in 0..STAR_AD_LANES {
            if mesa.d1[lane].to_bits() != flexmesa.d1[lane].to_bits() {
                self.mismatches.push(Mismatch {
                    field: field.to_string(),
                    lane: Some(lane + 1),
                    mesa: mesa.d1[lane],
                    flexmesa: flexmesa.d1[lane],
                });
            }
        }
    }

    /// Compares an integer or flag output.
    pub fn discrete<T: PartialEq + std::fmt::Debug + ?Sized>(
        &mut self,
        field: &str,
        mesa: &T,
        flexmesa: &T,
    ) {
        if mesa != flexmesa {
            self.discrete
                .push(format!("{field}: MESA {mesa:?} FlexMESA {flexmesa:?}"));
        }
    }

    /// True when every compared output is identical.
    #[must_use]
    pub fn identical(&self) -> bool {
        self.mismatches.is_empty() && self.discrete.is_empty()
    }
}

/// Per-field tally over many calls.
#[derive(Debug, Default, Clone)]
pub struct FieldTally {
    /// Calls in which the value differs.
    pub val_calls: usize,
    /// Largest value difference in ulp.
    pub val_max_ulp: u64,
    /// Calls in which at least one partial differs.
    pub d1_calls: usize,
    /// Largest partial difference in ulp.
    pub d1_max_ulp: u64,
}

/// Running summary of one record family.
#[derive(Debug, Default)]
pub struct FamilySummary {
    /// Records replayed.
    pub calls: usize,
    /// Records whose outputs are all identical.
    pub identical: usize,
    /// Records the kernel could not be run on (context and reason).
    pub not_replayed: Vec<String>,
    /// First differing record in file order (context and differences).
    pub first: Option<(String, CallDiff)>,
    /// Earliest differing record by `(run, model, solver_iter, k)`.
    pub earliest: Option<((i64, i64, i64, i64), String)>,
    /// Tallies per output field.
    pub fields: BTreeMap<String, FieldTally>,
    /// Tallies of discrete outputs (`field` to calls differing).
    pub discrete: BTreeMap<String, usize>,
}

impl FamilySummary {
    /// Adds one replayed call.
    pub fn add(&mut self, rec: &Record, diff: CallDiff) {
        self.calls += 1;
        if diff.identical() {
            self.identical += 1;
            return;
        }
        let mut seen_val = std::collections::BTreeSet::new();
        let mut seen_d1 = std::collections::BTreeSet::new();
        for m in &diff.mismatches {
            let t = self.fields.entry(m.field.clone()).or_default();
            let ulp = ulp_distance(m.mesa, m.flexmesa);
            if m.lane.is_none() {
                if seen_val.insert(m.field.clone()) {
                    t.val_calls += 1;
                }
                t.val_max_ulp = t.val_max_ulp.max(ulp);
            } else {
                if seen_d1.insert(m.field.clone()) {
                    t.d1_calls += 1;
                }
                t.d1_max_ulp = t.d1_max_ulp.max(ulp);
            }
        }
        for d in &diff.discrete {
            let name = d.split(':').next().unwrap_or(d).to_string();
            *self.discrete.entry(name).or_default() += 1;
        }
        let key = (
            i64::try_from(rec.run).unwrap_or(i64::MAX),
            rec.int("model").unwrap_or(i64::MAX),
            rec.int("solver_iter").unwrap_or(i64::MAX),
            rec.int("k").unwrap_or(i64::MAX),
        );
        if self.earliest.as_ref().is_none_or(|(k, _)| key < *k) {
            self.earliest = Some((key, rec.context()));
        }
        if self.first.is_none() {
            self.first = Some((rec.context(), diff));
        }
    }
}
