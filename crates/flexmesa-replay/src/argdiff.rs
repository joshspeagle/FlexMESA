// SPDX-License-Identifier: GPL-3.0-or-later
//! Call-by-call comparison of a MESA capture with the same records written
//! by `FlexMESA` (`#552`).
//!
//! The MESA overlay (`benchmarks/fortran_mesa/replay/`) and `FlexMESA`'s tap
//! (`flexmesa-star`, environment variable `FLEXMESA_REPLAY_TAP`) write the
//! same families in the same format. Records are matched by
//! `(family, model, solver_iter, k, site, occurrence)`, where the occurrence
//! counts earlier records with the same other key fields in file order (so
//! the `n`-th sweep over faces at one `(model, solver_iter)` matches the
//! `n`-th on the other side). Every field present on both sides is compared
//! bit for bit, auto-diff values over all 33 lanes in MESA's layout.
//!
//! The first difference is the one in the earliest sweep (sweeps ordered by
//! their first line in the MESA capture, since thread scheduling orders
//! records inside a sweep), then the lowest face `k`, with caller-argument
//! records (`GETR`) ranked before kernel records (`TDC`, `MLT`) at the same
//! face, because the kernel's arguments are assembled from them.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::BufRead;

use crate::compare::{ulp_distance, CallDiff};
use crate::record::{FormatError, Record, Schema, Value};

/// Record key: family, model, solver iteration, face, site, occurrence.
pub type Key = (String, i64, i64, i64, i64, u32);

/// A capture indexed by record key, lines kept unparsed.
struct Indexed {
    header: Vec<String>,
    schemas: HashMap<String, Schema>,
    /// Key to (line number, run, line text).
    records: HashMap<Key, (usize, usize, String)>,
    /// Keys in file order.
    order: Vec<Key>,
}

fn key_positions(schema: &Schema) -> Result<[usize; 4], FormatError> {
    let mut pos = [0_usize; 4];
    for (slot, name) in ["site", "model", "solver_iter", "k"].iter().enumerate()
    {
        pos[slot] = schema
            .fields
            .iter()
            .position(|(n, _)| n == name)
            .ok_or_else(|| FormatError(format!("schema lacks `{name}`")))?;
    }
    Ok(pos)
}

fn index<R: BufRead>(reader: R) -> Result<Indexed, FormatError> {
    let mut idx = Indexed {
        header: Vec::new(),
        schemas: HashMap::new(),
        records: HashMap::new(),
        order: Vec::new(),
    };
    let mut positions: HashMap<String, [usize; 4]> = HashMap::new();
    let mut occ: HashMap<(String, i64, i64, i64, i64), u32> = HashMap::new();
    let mut run = 0;
    for (n, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| FormatError(format!("read: {e}")))?;
        let line_no = n + 1;
        if let Some(h) = line.strip_prefix('#') {
            idx.header.push(h.trim().to_string());
            continue;
        }
        let mut toks = line.split_whitespace();
        let Some(tag) = toks.next() else { continue };
        match tag {
            "RUN" => run += 1,
            // Configuration lines feed stand-alone replay only.
            "CONFIG" => {}
            "SCHEMA" => {
                let family = toks.next().ok_or_else(|| {
                    FormatError(format!("line {line_no}: bare SCHEMA"))
                })?;
                let schema = Schema::parse(toks)?;
                if let Some(old) = idx.schemas.get(family) {
                    if *old != schema {
                        return Err(FormatError(format!(
                            "line {line_no}: {family} schema changed"
                        )));
                    }
                } else {
                    positions
                        .insert(family.to_string(), key_positions(&schema)?);
                    idx.schemas.insert(family.to_string(), schema);
                }
            }
            family => {
                let pos = positions.get(family).ok_or_else(|| {
                    FormatError(format!(
                        "line {line_no}: record `{family}` before its SCHEMA"
                    ))
                })?;
                let fields: Vec<&str> = toks.collect();
                let get = |slot: usize| -> Result<i64, FormatError> {
                    fields
                        .get(pos[slot])
                        .and_then(|t| t.parse().ok())
                        .ok_or_else(|| {
                            FormatError(format!(
                                "line {line_no}: bad key field"
                            ))
                        })
                };
                let base =
                    (family.to_string(), get(1)?, get(2)?, get(3)?, get(0)?);
                let o = occ.entry(base.clone()).or_insert(0);
                let key = (base.0, base.1, base.2, base.3, base.4, *o);
                *o += 1;
                idx.order.push(key.clone());
                idx.records.insert(key, (line_no, run, line));
            }
        }
    }
    Ok(idx)
}

fn parse(idx: &Indexed, key: &Key) -> Result<Record, FormatError> {
    let (line_no, run, line) = &idx.records[key];
    let mut toks = line.split_whitespace();
    let family = toks.next().unwrap_or_default();
    Record::parse(family, *line_no, *run, &idx.schemas[family], toks)
}

/// Differences of one matched record pair.
#[derive(Debug, Default)]
pub struct RecordDiff {
    /// Differing components, in schema order.
    pub diff: CallDiff,
    /// Fields one side wrote as `NA`, as `field (side)`.
    pub unavailable: Vec<String>,
    /// Components equal as numbers but not as bits: `+0.0` against `-0.0`.
    /// They include `d1Array` lanes `FlexMESA`'s 24-lane `StarAdOrder1`
    /// does not represent (written as `+0.0`) where MESA holds `-0.0`.
    pub signed_zero: usize,
}

fn compare_fields(
    schema: &Schema,
    mesa: &Record,
    flex: &Record,
    flex_schema: &Schema,
) -> RecordDiff {
    let mut out = RecordDiff::default();
    for (name, _) in &schema.fields {
        if !flex_schema.fields.iter().any(|(n, _)| n == name) {
            continue;
        }
        let (Some(a), Some(b)) = (mesa.value(name), flex.value(name)) else {
            continue;
        };
        let field = name.strip_prefix("out_").unwrap_or(name);
        let field = if name.starts_with("out_") {
            format!("out {field}")
        } else {
            field.to_string()
        };
        match (a, b) {
            (Value::Missing, Value::Missing) => {}
            (Value::Missing, _) => {
                out.unavailable.push(format!("{name} (MESA)"));
            }
            (_, Value::Missing) => {
                out.unavailable.push(format!("{name} (FlexMESA)"));
            }
            (Value::Real(x), Value::Real(y)) => out.diff.real(&field, *x, *y),
            (Value::Ad(x), Value::Ad(y)) => out.diff.ad(&field, x, y),
            (Value::Vec(x), Value::Vec(y)) => {
                if x.len() == y.len() {
                    for (i, (a, b)) in x.iter().zip(y).enumerate() {
                        out.diff.real(
                            &crate::eos::element_name(&field, i),
                            *a,
                            *b,
                        );
                    }
                } else {
                    out.diff.discrete(
                        &format!("{field} length"),
                        &x.len(),
                        &y.len(),
                    );
                }
            }
            (Value::Int(x), Value::Int(y)) => out.diff.discrete(&field, x, y),
            (Value::Logical(x), Value::Logical(y)) => {
                out.diff.discrete(&field, x, y);
            }
            (Value::Str(x), Value::Str(y)) => {
                out.diff.discrete(&field, x.as_str(), y.as_str());
            }
            _ => out.diff.discrete(&field, "type", "other type"),
        }
    }
    let before = out.diff.mismatches.len();
    // Numeric (not bitwise) equality on purpose: it holds for +0.0 / -0.0.
    #[allow(clippy::float_cmp)]
    out.diff.mismatches.retain(|m| m.mesa != m.flexmesa);
    out.signed_zero = before - out.diff.mismatches.len();
    out
}

/// Calls of one routine at one face, model and solver iteration, in order:
/// `(family, model, solver_iter, k, site)` to the keys of its occurrences.
fn face_calls(idx: &Indexed) -> HashMap<(String, i64, i64, i64, i64), u32> {
    let mut c = HashMap::new();
    for key in &idx.order {
        let e = c
            .entry((key.0.clone(), key.1, key.2, key.3, key.4))
            .or_insert(0);
        *e = (*e).max(key.5 + 1);
    }
    c
}

/// `(model, solver_iter)` pairs with records on both sides. Pairs reached
/// by one side only lie outside the other capture's window (its run ended
/// or started elsewhere) and are counted apart from differences.
fn shared_groups(m: &Indexed, f: &Indexed) -> HashSet<(i64, i64)> {
    let gm: HashSet<(i64, i64)> = m.order.iter().map(|k| (k.1, k.2)).collect();
    f.order
        .iter()
        .map(|k| (k.1, k.2))
        .filter(|g| gm.contains(g))
        .collect()
}

/// Identical tokens of two record lines (a similarity score).
fn same_tokens(a: &str, b: &str) -> u64 {
    a.split_whitespace()
        .zip(b.split_whitespace())
        .map(|(x, y)| u64::from(x == y))
        .sum()
}

/// Matches the calls of each routine at each face, model and solver
/// iteration between the two captures.
///
/// Where both sides called it equally often, calls are matched in order.
/// Where the counts differ (an extra sweep over faces, or trial calls such
/// as those of predictive mixing at a few faces), the two sequences are
/// aligned in order by maximizing the number of identical tokens over
/// matched calls (weighted longest common subsequence), so an extra call on
/// one side does not shift every later one. Returns the matching `FlexMESA`
/// key of each matched MESA key, and one note per `(model, solver_iter)`
/// with unequal counts.
// Occurrence indices come from `u32` counts, so the casts back are exact.
#[allow(clippy::many_single_char_names, clippy::cast_possible_truncation)]
fn align_calls(m: &Indexed, f: &Indexed) -> (HashMap<Key, Key>, Vec<String>) {
    let cm = face_calls(m);
    let cf = face_calls(f);
    let mut map = HashMap::new();
    // (model, solver_iter) -> (faces, extra FlexMESA calls, missing, first)
    let mut agg: BTreeMap<(i64, i64), (usize, usize, usize, String)> =
        BTreeMap::new();
    for (g, &a) in &cm {
        let b = cf.get(g).copied().unwrap_or(0);
        let key = |occ: u32| (g.0.clone(), g.1, g.2, g.3, g.4, occ);
        if a == b {
            for i in 0..a {
                map.insert(key(i), key(i));
            }
            continue;
        }
        let (a, b) = (a as usize, b as usize);
        let mut score = vec![vec![0_u64; b]; a];
        for (i, row) in score.iter_mut().enumerate() {
            let mline = &m.records[&key(i as u32)].2;
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = same_tokens(mline, &f.records[&key(j as u32)].2);
            }
        }
        let mut dp = vec![vec![0_u64; b + 1]; a + 1];
        for i in 1..=a {
            for j in 1..=b {
                dp[i][j] = dp[i - 1][j]
                    .max(dp[i][j - 1])
                    .max(dp[i - 1][j - 1] + score[i - 1][j - 1]);
            }
        }
        let (mut i, mut j) = (a, b);
        let mut pairs = 0;
        while i > 0 && j > 0 {
            if score[i - 1][j - 1] > 0
                && dp[i][j] == dp[i - 1][j - 1] + score[i - 1][j - 1]
            {
                map.insert(key((i - 1) as u32), key((j - 1) as u32));
                pairs += 1;
                i -= 1;
                j -= 1;
            } else if dp[i][j] == dp[i - 1][j] {
                i -= 1;
            } else {
                j -= 1;
            }
        }
        let e = agg
            .entry((g.1, g.2))
            .or_insert_with(|| (0, 0, 0, String::new()));
        e.0 += 1;
        e.1 += b - pairs;
        e.2 += a - pairs;
        let here = format!("{} k={} ({a} MESA, {b} FlexMESA calls)", g.0, g.3);
        if e.3.is_empty() || here < e.3 {
            e.3 = here;
        }
    }
    let shared = shared_groups(m, f);
    for (g, &b) in &cf {
        if !cm.contains_key(g) && shared.contains(&(g.1, g.2)) {
            let e = agg
                .entry((g.1, g.2))
                .or_insert_with(|| (0, 0, 0, String::new()));
            e.0 += 1;
            e.1 += b as usize;
            if e.3.is_empty() {
                e.3 = format!("{} k={} (0 MESA, {b} FlexMESA calls)", g.0, g.3);
            }
        }
    }
    agg.retain(|g, _| shared.contains(g));
    let notes = agg
        .into_iter()
        .map(|((model, iter), (faces, extra, missing, first))| {
            format!(
                "model {model} solver_iter {iter}: call counts differ at \
                 {faces} routine-faces ({extra} FlexMESA calls and \
                 {missing} MESA calls without a counterpart); e.g. {first}"
            )
        })
        .collect();
    (map, notes)
}

/// First differing record of a field, with its tally.
#[derive(Debug, Clone, Default)]
pub struct FieldStats {
    /// Matched calls in which the value differs.
    pub val_calls: usize,
    /// Matched calls in which a partial differs (but maybe not the value).
    pub d1_calls: usize,
    /// Largest value difference in ulp.
    pub val_max_ulp: u64,
    /// Earliest differing call (rank, context, first component).
    pub earliest: Option<((usize, i64, u8), String, String)>,
}

/// Per-family match counts.
#[derive(Debug, Default)]
pub struct FamilyMatch {
    /// Keys present on both sides.
    pub matched: usize,
    /// Matched keys with every common field identical.
    pub identical: usize,
    /// Keys only in the MESA capture, with the first in MESA file order.
    pub mesa_only: (usize, Option<String>),
    /// Keys only in the `FlexMESA` capture, with the first in its order.
    pub flexmesa_only: (usize, Option<String>),
    /// Calls at `(model, solver_iter)` pairs the other side never reached
    /// (outside its window): `(MESA, FlexMESA)`.
    pub beyond_window: (usize, usize),
}

/// Result of [`compare_captures`].
#[derive(Debug, Default)]
pub struct ArgDiffSummary {
    /// MESA capture header.
    pub mesa_header: Vec<String>,
    /// `FlexMESA` capture header.
    pub flexmesa_header: Vec<String>,
    /// Per-family counts.
    pub families: BTreeMap<String, FamilyMatch>,
    /// Fields written by one side only, as `family.field (side)`.
    pub one_sided_fields: Vec<String>,
    /// Per `family.field` statistics over matched calls.
    pub fields: BTreeMap<String, FieldStats>,
    /// Fields written as `NA` by a side, with the number of calls.
    pub unavailable: BTreeMap<String, usize>,
    /// Earliest differing call: rank, context and its differences.
    pub first: Option<((usize, i64, u8), String, RecordDiff)>,
    /// Groups where the two sides made a different number of sweeps.
    pub sweep_notes: Vec<String>,
    /// Matched calls differing only in the sign of zero components.
    pub signed_zero_calls: usize,
}

impl ArgDiffSummary {
    /// True when every key matched and every common field is identical.
    #[must_use]
    pub fn all_identical(&self) -> bool {
        self.first.is_none()
            && self.sweep_notes.is_empty()
            && self.families.values().all(|f| {
                f.matched == f.identical
                    && f.mesa_only.0 == 0
                    && f.flexmesa_only.0 == 0
            })
    }
}

fn family_rank(family: &str) -> u8 {
    u8::from(family != "GETR")
}

fn ctx(key: &Key) -> String {
    format!(
        "{} site={} model={} solver_iter={} k={} sweep#{}",
        key.0, key.4, key.1, key.2, key.3, key.5
    )
}

/// Compares a MESA capture with a `FlexMESA` capture call by call.
///
/// # Errors
/// Returns [`FormatError`] for a malformed capture or a family whose schema
/// differs between the two files in a field both write with different types.
pub fn compare_captures<A: BufRead, B: BufRead>(
    mesa: A,
    flexmesa: B,
) -> Result<ArgDiffSummary, FormatError> {
    let m = index(mesa)?;
    let f = index(flexmesa)?;
    let mut s = ArgDiffSummary {
        mesa_header: m.header.clone(),
        flexmesa_header: f.header.clone(),
        ..ArgDiffSummary::default()
    };
    for (family, ms) in &m.schemas {
        let Some(fs) = f.schemas.get(family) else {
            continue;
        };
        for (name, ty) in &ms.fields {
            match fs.fields.iter().find(|(n, _)| n == name) {
                None => s.one_sided_fields.push(format!("{family}.{name} (MESA)")),
                Some((_, t)) if t != ty => {
                    return Err(FormatError(format!(
                        "{family}.{name} has type {ty:?} in MESA, {t:?} in FlexMESA"
                    )))
                }
                Some(_) => {}
            }
        }
        for (name, _) in &fs.fields {
            if !ms.fields.iter().any(|(n, _)| n == name) {
                s.one_sided_fields
                    .push(format!("{family}.{name} (FlexMESA)"));
            }
        }
    }
    let (call_map, notes) = align_calls(&m, &f);
    let shared = shared_groups(&m, &f);
    s.sweep_notes = notes;
    let flex_key = |key: &Key| -> Option<Key> { call_map.get(key).cloned() };
    // Sweep rank: first MESA line of each (model, solver_iter, occurrence).
    let mut sweep_line: HashMap<(i64, i64, u32), usize> = HashMap::new();
    for key in &m.order {
        let line = m.records[key].0;
        sweep_line
            .entry((key.1, key.2, key.5))
            .and_modify(|l| *l = (*l).min(line))
            .or_insert(line);
    }
    for key in &m.order {
        let fam = s.families.entry(key.0.clone()).or_default();
        let Some(fkey) = flex_key(key).filter(|k| f.records.contains_key(k))
        else {
            if !shared.contains(&(key.1, key.2)) {
                fam.beyond_window.0 += 1;
                continue;
            }
            fam.mesa_only.0 += 1;
            fam.mesa_only.1.get_or_insert_with(|| ctx(key));
            continue;
        };
        fam.matched += 1;
        let mr = parse(&m, key)?;
        let fr = parse(&f, &fkey)?;
        let rd =
            compare_fields(&m.schemas[&key.0], &mr, &fr, &f.schemas[&key.0]);
        for u in &rd.unavailable {
            *s.unavailable.entry(format!("{}.{u}", key.0)).or_default() += 1;
        }
        if rd.diff.identical() {
            fam.identical += 1;
            if rd.signed_zero > 0 {
                s.signed_zero_calls += 1;
            }
            continue;
        }
        let rank = (
            sweep_line[&(key.1, key.2, key.5)],
            key.3,
            family_rank(&key.0),
        );
        tally(&mut s.fields, key, rank, &rd.diff);
        if s.first.as_ref().is_none_or(|(r, _, _)| rank < *r) {
            s.first = Some((rank, ctx(key), rd));
        }
    }
    let matched_f: HashSet<Key> = m.order.iter().filter_map(flex_key).collect();
    for key in &f.order {
        if !matched_f.contains(key) {
            let fam = s.families.entry(key.0.clone()).or_default();
            if !shared.contains(&(key.1, key.2)) {
                fam.beyond_window.1 += 1;
                continue;
            }
            fam.flexmesa_only.0 += 1;
            fam.flexmesa_only.1.get_or_insert_with(|| ctx(key));
        }
    }
    Ok(s)
}

fn tally(
    fields: &mut BTreeMap<String, FieldStats>,
    key: &Key,
    rank: (usize, i64, u8),
    diff: &CallDiff,
) {
    let mut first_of: BTreeMap<String, String> = BTreeMap::new();
    let mut val: BTreeMap<String, u64> = BTreeMap::new();
    for m in &diff.mismatches {
        first_of
            .entry(m.field.clone())
            .or_insert_with(|| m.describe());
        if m.lane.is_none() {
            val.insert(m.field.clone(), ulp_distance(m.mesa, m.flexmesa));
        }
    }
    for d in &diff.discrete {
        let name = d.split(':').next().unwrap_or(d).to_string();
        first_of.entry(name).or_insert_with(|| d.clone());
    }
    for (field, desc) in first_of {
        let st = fields.entry(format!("{}.{field}", key.0)).or_default();
        if let Some(ulp) = val.get(&field) {
            st.val_calls += 1;
            st.val_max_ulp = st.val_max_ulp.max(*ulp);
        } else {
            st.d1_calls += 1;
        }
        if st.earliest.as_ref().is_none_or(|(r, _, _)| rank < *r) {
            st.earliest = Some((rank, ctx(key), desc));
        }
    }
}
