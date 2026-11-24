// SPDX-License-Identifier: GPL-3.0-or-later
//! Raw unscreened nuclear reaction rate evaluation (`set_raw_rate`,
//! `set_raw_rates`, `eval_reaclib_handle`), `interp_pm` rate-table evaluation,
//! and Fushiki & Lamb (1987) pycnonuclear triple-alpha (`pycno.f90`).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use flexmesa_const::ONE_THIRD;
use flexmesa_interp1d::{interp_pm, interp_value_and_slope, CubicCoeffs};
use flexmesa_utils::{
    is_bad, resolve_flexmesa_data_dir, resolve_mesa_test_inputs_dir,
};

use crate::loader::RatesCatalog;
use crate::ratelib::eval_analytic_rate;
use crate::tfactors::{
    eval_reaclib_inverse_ratio, eval_reaclib_terms, TFactors,
};
use crate::types::{RatesError, ReactionInfo, DEFAULT_REACLIB_MIN_T9};

const SPECIAL_RATE_BUNDLE_MAGIC: &[u8; 8] = b"FMRT0001";
const RAW_SPECIAL_RATE_BUNDLE: &[u8] =
    include_bytes!("../data/special_rate_tables.bin");

struct BuiltinSpecialRateIndex {
    entries: HashMap<&'static str, (usize, usize)>,
}

static BUILTIN_SPECIAL_RATES: OnceLock<BuiltinSpecialRateIndex> =
    OnceLock::new();

fn builtin_special_rate_index() -> &'static BuiltinSpecialRateIndex {
    BUILTIN_SPECIAL_RATES.get_or_init(|| {
        let mut entries = HashMap::new();
        let bytes = RAW_SPECIAL_RATE_BUNDLE;
        if bytes.len() < 12 || &bytes[0..8] != SPECIAL_RATE_BUNDLE_MAGIC {
            return BuiltinSpecialRateIndex { entries };
        }
        let count =
            u32::from_le_bytes(bytes[8..12].try_into().unwrap_or([0; 4]))
                as usize;
        let mut pos = 12_usize;
        for _ in 0..count {
            if pos + 8 > bytes.len() {
                break;
            }
            let nlen = u32::from_le_bytes(
                bytes[pos..pos + 4].try_into().unwrap_or([0; 4]),
            ) as usize;
            let nrows = u32::from_le_bytes(
                bytes[pos + 4..pos + 8].try_into().unwrap_or([0; 4]),
            ) as usize;
            pos += 8;
            let data_bytes = nrows.saturating_mul(16);
            if pos + nlen + data_bytes > bytes.len() {
                break;
            }
            if let Ok(name) = std::str::from_utf8(&bytes[pos..pos + nlen]) {
                entries.insert(name, (nrows, pos + nlen));
            }
            pos += nlen + data_bytes;
        }
        BuiltinSpecialRateIndex { entries }
    })
}

fn load_builtin_special_rate(name: &str) -> Option<(Vec<f64>, Vec<f64>)> {
    let idx = builtin_special_rate_index();
    let &(nrows, offset) = idx.entries.get(name)?;
    let bytes = RAW_SPECIAL_RATE_BUNDLE;
    let t8_end = offset + nrows * 8;
    let rate_end = t8_end + nrows * 8;
    if rate_end > bytes.len() {
        return None;
    }
    let mut t8s = Vec::with_capacity(nrows);
    let mut rates = Vec::with_capacity(nrows);
    for i in 0..nrows {
        let p = offset + i * 8;
        t8s.push(f64::from_le_bytes(
            bytes[p..p + 8].try_into().unwrap_or([0; 8]),
        ));
    }
    for i in 0..nrows {
        let p = t8_end + i * 8;
        rates.push(f64::from_le_bytes(
            bytes[p..p + 8].try_into().unwrap_or([0; 8]),
        ));
    }
    Some((t8s, rates))
}

fn parse_fortran_f64(tok: &str) -> Option<f64> {
    if let Ok(v) = tok.parse::<f64>() {
        return Some(v);
    }
    if tok.contains(['D', 'd']) {
        let normalized = tok.replace(['D', 'd'], "E");
        return normalized.parse::<f64>().ok();
    }
    None
}

fn resolve_special_rate_disk_path(
    trimmed: &str,
    search_dir: Option<&Path>,
) -> Option<PathBuf> {
    let direct = Path::new(trimmed);
    if direct.is_absolute() {
        return direct.is_file().then(|| direct.to_path_buf());
    }
    if let Some(dir) = search_dir {
        let local = dir.join(trimmed);
        if local.is_file() {
            return Some(local);
        }
    }
    if direct.is_file() {
        return Some(direct.to_path_buf());
    }
    let mesa_cand = resolve_mesa_test_inputs_dir()
        .join("data")
        .join("rates_data")
        .join("rate_tables")
        .join(trimmed);
    if mesa_cand.is_file() {
        return Some(mesa_cand);
    }
    let data_cand = resolve_flexmesa_data_dir()
        .join("rates_data")
        .join("rate_tables")
        .join(trimmed);
    if data_cand.is_file() {
        return Some(data_cand);
    }
    None
}

/// 1D piecewise-monotonic (`interp_pm`) tabulated nuclear reaction rate
/// (`rate_table_info` / `get_interp_table` in `rates/private/raw_rates.f90:845-1017`).
#[derive(Debug, Clone)]
pub struct RateTable1D {
    /// Tabulated temperature grid $T_8 = T / 10^8\,\mathrm{K}$ (strictly increasing).
    pub t8s: Vec<f64>,
    /// Fitted `interp_pm` cubic spline coefficients at each $T_8$ knot.
    pub f1: Vec<CubicCoeffs>,
}

impl RateTable1D {
    /// Constructs a [`RateTable1D`] by fitting an `interp_pm` spline to
    /// strictly increasing `t8s` ($T / 10^8\,\mathrm{K}$) and `rates`
    /// (`rates/private/raw_rates.f90:989-993`).
    ///
    /// # Errors
    /// Returns [`RatesError::InvalidTableData`] if the grid has fewer than 2
    /// points, contains non-finite values, is not strictly increasing, or if
    /// `interp_pm` fitting fails.
    pub fn from_points(
        t8s: Vec<f64>,
        rates: &[f64],
    ) -> Result<Self, RatesError> {
        if t8s.len() < 2 || t8s.len() != rates.len() {
            return Err(RatesError::InvalidTableData(format!(
                "rate table requires >= 2 matching (T8, rate) points, got ({}, {})",
                t8s.len(),
                rates.len()
            )));
        }
        for i in 0..t8s.len() {
            if is_bad(t8s[i]) || is_bad(rates[i]) {
                return Err(RatesError::InvalidTableData(format!(
                    "non-finite (T8, rate) at row {}",
                    i + 1
                )));
            }
            if i > 0 && t8s[i] <= t8s[i - 1] {
                return Err(RatesError::InvalidTableData(format!(
                    "non-increasing T8 grid at row {}: {} <= {}",
                    i + 1,
                    t8s[i],
                    t8s[i - 1]
                )));
            }
        }
        let f1 = interp_pm(&t8s, rates).map_err(|e| {
            RatesError::InvalidTableData(format!(
                "interp_pm failed on special rate table: {e}"
            ))
        })?;
        Ok(Self { t8s, f1 })
    }

    /// Parses a MESA ASCII 1D rate table file (`get_interp_table` in
    /// `rates/private/raw_rates.f90:962-995`).
    ///
    /// Skips leading blank and comment lines until a line starting with a
    /// positive integer `nT8s`, then reads `nT8s` rows of `(T8, rate)` pairs
    /// (supporting Fortran `D`/`d` exponents and ignoring trailing columns).
    ///
    /// # Errors
    /// Returns [`RatesError::InvalidTableData`] if the header or rows are
    /// malformed or spline fitting fails.
    pub fn from_str_table(text: &str) -> Result<Self, RatesError> {
        let mut lines = text.lines();
        let mut n_t8s: Option<usize> = None;
        for line in &mut lines {
            let trimmed = line.trim();
            if trimmed.is_empty()
                || trimmed.starts_with('#')
                || trimmed.starts_with('!')
            {
                continue;
            }
            if let Some(first_tok) = trimmed.split_whitespace().next() {
                if let Ok(n) = first_tok.parse::<usize>() {
                    if n > 0 {
                        n_t8s = Some(n);
                        break;
                    }
                }
            }
        }
        let n = n_t8s.ok_or_else(|| {
            RatesError::InvalidTableData(String::from(
                "failed to find positive integer nT8s header in rate table",
            ))
        })?;

        let mut t8s = Vec::with_capacity(n);
        let mut rates = Vec::with_capacity(n);
        for row_idx in 0..n {
            let line = lines.next().ok_or_else(|| {
                RatesError::InvalidTableData(format!(
                    "unexpected EOF in rate table at row {} of {n}",
                    row_idx + 1
                ))
            })?;
            let mut toks = line.split_whitespace();
            let t8_tok = toks.next().ok_or_else(|| {
                RatesError::InvalidTableData(format!(
                    "missing T8 column at row {}",
                    row_idx + 1
                ))
            })?;
            let rate_tok = toks.next().ok_or_else(|| {
                RatesError::InvalidTableData(format!(
                    "missing rate column at row {}",
                    row_idx + 1
                ))
            })?;
            let t8 = parse_fortran_f64(t8_tok).ok_or_else(|| {
                RatesError::InvalidTableData(format!(
                    "invalid T8 value '{t8_tok}' at row {}",
                    row_idx + 1
                ))
            })?;
            let rate = parse_fortran_f64(rate_tok).ok_or_else(|| {
                RatesError::InvalidTableData(format!(
                    "invalid rate value '{rate_tok}' at row {}",
                    row_idx + 1
                ))
            })?;
            t8s.push(t8);
            rates.push(rate);
        }
        Self::from_points(t8s, &rates)
    }

    /// Reads and parses a MESA ASCII 1D rate table from `path`.
    ///
    /// # Errors
    /// Returns [`RatesError::InvalidTableData`] if the file cannot be read or
    /// parsed.
    pub fn from_file(path: &Path) -> Result<Self, RatesError> {
        let text = fs::read_to_string(path).map_err(|e| {
            RatesError::InvalidTableData(format!(
                "cannot open special rate table file '{}': {e}",
                path.display()
            ))
        })?;
        Self::from_str_table(&text).map_err(|e| {
            RatesError::InvalidTableData(format!(
                "failed to parse special rate table '{}': {e}",
                path.display()
            ))
        })
    }

    /// Returns `true` if `filename` resolves to an existing file on disk or a
    /// bundled MESA `26.4.1` built-in rate table.
    #[must_use]
    pub fn exists(filename: &str, search_dir: Option<&Path>) -> bool {
        let trimmed = filename.trim();
        if trimmed.is_empty() {
            return false;
        }
        if resolve_special_rate_disk_path(trimmed, search_dir).is_some() {
            return true;
        }
        !trimmed.contains(['/', '\\'])
            && builtin_special_rate_index().entries.contains_key(trimmed)
    }

    /// Resolves and loads `filename` following upstream MESA precedence
    /// (`rates/private/raw_rates.f90:948-960`): local / `search_dir` file first,
    /// then `$MESA_DIR/data/rates_data/rate_tables/<filename>` (or
    /// `FLEXMESA_DATA_DIR/rates_data/rate_tables/<filename>`), then the
    /// embedded MESA `26.4.1` built-in `rate_tables` bundle for bare filenames.
    ///
    /// # Errors
    /// Returns [`RatesError::InvalidTableData`] if the file cannot be found,
    /// read, or parsed.
    pub fn resolve_and_load(
        filename: &str,
        search_dir: Option<&Path>,
    ) -> Result<Self, RatesError> {
        let trimmed = filename.trim();
        if trimmed.is_empty() {
            return Err(RatesError::InvalidTableData(String::from(
                "empty special rate table filename",
            )));
        }
        if let Some(path) = resolve_special_rate_disk_path(trimmed, search_dir)
        {
            return Self::from_file(&path);
        }
        if !trimmed.contains(['/', '\\']) {
            if let Some((t8s, rates)) = load_builtin_special_rate(trimmed) {
                return Self::from_points(t8s, &rates);
            }
        }
        Err(RatesError::InvalidTableData(format!(
            "cannot open special rate table file '{trimmed}'"
        )))
    }

    /// Evaluates `(rate, drate_dt)` at temperature `temp` ($\mathrm{K}$) via
    /// clamped `interp_pm` interpolation (`eval_table` in
    /// `rates/private/raw_rates.f90:845-877`).
    ///
    /// # Errors
    /// Returns [`RatesError::InvalidTableData`] if interpolation fails.
    pub fn eval(&self, temp: f64) -> Result<(f64, f64), RatesError> {
        eval_interp_rate_table(&self.t8s, &self.f1, temp)
    }

    /// Returns a deterministic FNV-1a 64-bit fingerprint of this table's knot
    /// and spline values for network identity caching.
    #[must_use]
    pub fn content_fingerprint(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325_u64;
        let mut mix = |bytes: &[u8]| {
            for &b in bytes {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x100000001b3);
            }
        };
        mix(&(self.t8s.len() as u64).to_le_bytes());
        for (&t8, c) in self.t8s.iter().zip(&self.f1) {
            mix(&t8.to_bits().to_le_bytes());
            mix(&c[0].to_bits().to_le_bytes());
        }
        h
    }
}

/// Evaluates the raw unscreened nuclear reaction rate
/// `set_raw_rate(ir, temp, tf)` for a 1-based reaction ID `ir` (`1..=316`)
/// matching `rates/private/raw_rates.f90`.
///
/// # Errors
/// Returns [`RatesError::UnknownReaction`] if `ir` is out of bounds.
pub fn set_raw_rate(
    ir: usize,
    temp: f64,
    tf: &TFactors,
) -> Result<f64, RatesError> {
    let cat = RatesCatalog::builtin();
    let info = cat.reaction_by_id(ir)?;
    eval_reaction_raw(cat, info, temp, tf)
}

/// Evaluates `set_raw_rate` across a slice of 1-based reaction IDs `irs`.
///
/// # Errors
/// Returns [`RatesError::UnknownReaction`] if any reaction ID is out of bounds,
/// or [`RatesError::InvalidTableData`] if rate evaluation fails.
pub fn set_raw_rates(
    irs: &[usize],
    temp: f64,
    tf: &TFactors,
) -> Result<Vec<f64>, RatesError> {
    let cat = RatesCatalog::builtin();
    let mut out = Vec::with_capacity(irs.len());
    for &ir in irs {
        let info = cat.reaction_by_id(ir)?;
        out.push(eval_reaction_raw(cat, info, temp, tf)?);
    }
    Ok(out)
}

fn eval_reaclib_with_cat(
    cat: &RatesCatalog,
    handle: &str,
    t9: f64,
    check_e_flag: bool,
) -> Result<(f64, f64, f64, f64), RatesError> {
    let (is_rev, terms) = cat.reaclib_terms(handle)?;
    let (lam, dlam, rlam, drlam) =
        eval_reaclib_terms(terms, t9, DEFAULT_REACLIB_MIN_T9, check_e_flag);
    if is_rev {
        Ok((rlam, drlam, lam, dlam))
    } else {
        Ok((lam, dlam, rlam, drlam))
    }
}

/// Evaluates a REACLIB reaction handle
/// `(forward_rate, dforward_dlnT, reverse_rate, dreverse_dlnT)` at temperature
/// $T_9 = T / 10^9\,\mathrm{K}$.
///
/// # Errors
/// Returns [`RatesError::UnknownReaction`] if `handle` is not in the catalog.
pub fn eval_reaclib_handle(
    handle: &str,
    t9: f64,
    check_e_flag: bool,
) -> Result<(f64, f64, f64, f64), RatesError> {
    eval_reaclib_with_cat(RatesCatalog::builtin(), handle, t9, check_e_flag)
}

/// Evaluates `set_raw_rate` for a [`ReactionInfo`] entry given [`RatesCatalog`],
/// [`TFactors`], and optional file-based special rate overrides
/// (`set_raw_rate`, `eval_table`, and `eval_table_reverse` in
/// `rates/private/raw_rates.f90:84-118,845-920`).
///
/// When `info.name` is present in `overrides`, evaluates its table directly.
/// When `info.name` is not overridden but its paired reaction `reverse_name` is
/// present in `overrides`:
/// - If `info.name` is a reverse reaction (not a forward REACLIB handle),
///   computes detailed balance $\lambda_{\mathrm{rev}} = \mathrm{inv\_lambda}(T_9)
///   \cdot \lambda_{\mathrm{fwd,table}}(T)$.
/// - If `info.name` is a forward reaction, returns an error matching MESA
///   `raw_rates.f90:104-112`.
///
/// # Errors
/// Returns [`RatesError`] if a table evaluation fails, detailed balance is
/// requested for a forward reaction from a reverse-only table, or the forward
/// REACLIB entry required for detailed balance is unavailable.
pub fn eval_reaction_raw_with_overrides<S: std::hash::BuildHasher>(
    cat: &RatesCatalog,
    info: &ReactionInfo,
    temp: f64,
    tf: &TFactors,
    overrides: &HashMap<String, Arc<RateTable1D>, S>,
    reverse_name: Option<&str>,
) -> Result<f64, RatesError> {
    if let Some(tbl) = overrides.get(&info.name) {
        return Ok(tbl.eval(temp)?.0);
    }
    if let Some(rev_name) = reverse_name {
        if let Some(rev_tbl) = overrides.get(rev_name) {
            // `rates/private/raw_rates.f90:95-112`:
            // `do_reaclib_lookup(reaction_name(ir), reaclib_rates% reaction_dict)`:
            // forward reactions have a forward (`is_rev == false`) REACLIB entry,
            // while reverse reactions (`r_o16_ga_c12`, `r_c12_to_he4_he4_he4`) do not.
            let is_forward_reaclib = matches!(
                cat.reaclib_terms(&info.name),
                Ok((false, terms)) if !terms.is_empty()
            );
            if is_forward_reaclib {
                return Err(RatesError::InvalidTableData(format!(
                    "cannot evaluate forward reaction '{}' from detailed balance with reverse rate table '{}'; provide both rates or only '{}'",
                    info.name, rev_name, info.name
                )));
            }
            let (fr_table, _) = rev_tbl.eval(temp)?;
            let (_, terms) = cat.reaclib_terms(rev_name)?;
            let first = terms.first().ok_or_else(|| {
                RatesError::UnknownReaction(format!(
                    "missing REACLIB terms for detailed balance of '{rev_name}'"
                ))
            })?;
            let inv_lambda = eval_reaclib_inverse_ratio(first, tf.t9);
            return Ok(inv_lambda * fr_table);
        }
    }
    eval_reaction_raw(cat, info, temp, tf)
}

/// Evaluates `set_raw_rate` for a [`ReactionInfo`] entry given [`RatesCatalog`]
/// and [`TFactors`] at temperature `temp` ($\mathrm{K}$).
///
/// # Errors
/// Returns [`RatesError::UnknownReaction`] if a required REACLIB handle, rate
/// table, or `op_code` is missing/unsupported, or [`RatesError::InvalidTableData`]
/// if rate-table interpolation fails.
pub fn eval_reaction_raw(
    cat: &RatesCatalog,
    info: &ReactionInfo,
    temp: f64,
    tf: &TFactors,
) -> Result<f64, RatesError> {
    match info.op_code {
        0 => {
            let (f, _, _, _) =
                eval_reaclib_with_cat(cat, &info.target_handle, tf.t9, false)?;
            Ok(f)
        }
        1 => Ok(cat.eval_rate_table(info.id, temp)?.0),
        2 => Ok(flexmesa_math::exp(-99.0)),
        3..=28 => {
            let (fr, rr) = eval_analytic_rate(info.op_code, tf);
            Ok(if info.is_reverse { rr } else { fr })
        }
        29 => {
            let (f, _, r, _) =
                eval_reaclib_with_cat(cat, &info.target_handle, tf.t9, true)?;
            Ok(if info.is_reverse { r } else { f })
        }
        30 => {
            let (fr1, _, _, _) =
                eval_reaclib_with_cat(cat, "r_fe54_ag_ni58", tf.t9, true)?;
            let (_, _, fr2, _) =
                eval_reaclib_with_cat(cat, "r_ni57_na_fe54", tf.t9, true)?;
            let (fr3, _, _, _) =
                eval_reaclib_with_cat(cat, "r_fe54_ap_co57", tf.t9, true)?;
            Ok(fr1 + fr2 + fr3)
        }
        31 => {
            let (f1, _, r1, _) = eval_reaclib_with_cat(
                cat,
                "r_o16_o16_to_he4_si28",
                tf.t9,
                true,
            )?;
            let (f2, _, r2, _) =
                eval_reaclib_with_cat(cat, "r_o16_o16_to_h1_p31", tf.t9, true)?;
            Ok(if info.is_reverse { r1 + r2 } else { f1 + f2 })
        }
        other => Err(RatesError::UnknownReaction(format!(
            "{} (unsupported raw rate op_code {other})",
            info.name
        ))),
    }
}

/// Evaluates an `interp_pm` 1D cubic spline rate table `(t8s, f1)` at
/// temperature `temp` ($\mathrm{K}$), returning `(rate, drate_dt)`.
///
/// # Errors
/// Returns [`RatesError::InvalidTableData`] if the table is empty or
/// interpolation fails.
pub fn eval_interp_rate_table(
    t8s: &[f64],
    f1: &[CubicCoeffs],
    temp: f64,
) -> Result<(f64, f64), RatesError> {
    if t8s.is_empty() {
        return Err(RatesError::InvalidTableData(
            "empty interpolated rate table".to_string(),
        ));
    }
    let t8 = (temp * 1.0e-8).clamp(t8s[0], t8s[t8s.len() - 1]);
    let (val, slope) = interp_value_and_slope(t8s, f1, t8).map_err(|e| {
        RatesError::InvalidTableData(format!(
            "rate table interpolation failed at T={temp:e}: {e}"
        ))
    })?;
    Ok((val, slope * 1.0e-8))
}

/// Evaluates the Fushiki & Lamb (1987, ApJ 317, 368) pycnonuclear + thermally
/// enhanced triple-alpha energy generation rate $\epsilon_{3\alpha}$
/// (`erg g^-1 s^-1`) matching `rates/private/pycno.f90:31-248`.
///
/// # Arguments
/// - `t`: Temperature $T$ ($\mathrm{K}$).
/// - `rho`: Mass density $\rho$ ($\mathrm{g\,cm^{-3}}$).
/// - `y`: Helium-4 mass fraction $Y$ (`X(He4)`).
/// - `ue`: Mean nucleon-to-electron ratio $\mu_e \approx 1 / Y_e$ (typically `2.0`).
#[must_use]
pub fn eval_fl_epsnuc_3alf(t: f64, rho: f64, y: f64, ue: f64) -> f64 {
    eval_fl_epsnuc_3alf_with_derivs(t, rho, y, ue).0
}

/// Evaluates the Fushiki & Lamb (1987, ApJ 317, 368) pycnonuclear + thermally
/// enhanced triple-alpha energy generation rate and its analytical $(T, \rho)$
/// partial derivatives `(r, dr_dt, dr_drho)` in `erg g^-1 s^-1`, matching
/// `FL_epsnuc_3alf` in `rates/private/pycno.f90:31-248` (`0 ULP` parity).
///
/// # Arguments
/// - `t`: Temperature $T$ ($\mathrm{K}$).
/// - `rho`: Mass density $\rho$ ($\mathrm{g\,cm^{-3}}$).
/// - `y`: Helium-4 mass fraction $Y$ (`X(He4)`).
/// - `ue`: Mean nucleon-to-electron ratio $\mu_e \approx 1 / Y_e$ (typically `2.0`).
#[must_use]
pub fn eval_fl_epsnuc_3alf_with_derivs(
    t: f64,
    rho: f64,
    y: f64,
    ue: f64,
) -> (f64, f64, f64) {
    // `rates/private/pycno.f90:74-248`
    let r6 = rho * 1.0e-6;
    let d_r6_drho = 1.0e-6;
    let r6t = 2.0 * r6 / ue;
    let d_r6t_drho = 2.0 * d_r6_drho / ue;

    let r6t16 = flexmesa_math::pow(r6t, 1.0 / 6.0);
    let d_r6t16_drho = (1.0 / 6.0) * d_r6t_drho * r6t16 / r6t;
    let r6t13 = r6t16 * r6t16;
    let d_r6t13_drho = 2.0 * r6t16 * d_r6t16_drho;

    let t6 = t * 1.0e-6;
    let d_t6_dt = 1.0e-6;
    let d_t62_dt = 2.0 * t6 * d_t6_dt;
    let t613 = flexmesa_math::pow(t6, ONE_THIRD);
    let d_t613_dt = ONE_THIRD * d_t6_dt * t613 / t6;
    let t623 = t613 * t613;
    let d_t623_dt = 2.0 * t613 * d_t613_dt;
    let t653 = t623 * t6;
    let d_t653_dt = d_t623_dt * t6 + t623 * d_t6_dt;

    let t62 = t6 * t6;
    let t612 = t6.sqrt();
    let d_t612_dt = 0.5 * d_t6_dt / t612;
    let t632 = t6 * t612;
    let d_t632_dt = 1.5 * t612 * d_t6_dt;

    let u = 1.35 * r6t13 / t623;
    let d_u_dt = -u * d_t623_dt / t623;
    let d_u_drho = u * d_r6t13_drho / r6t13;

    let u32 = u * u.sqrt();
    let u52 = u * u32;

    let ((g1, dg1_dt, dg1_drho), (g2, dg2_dt, dg2_drho)) = if u < 1.0 {
        let a1 = flexmesa_math::pow2(1.0 - 4.222e-2 * t623) + 2.643e-5 * t653;
        let d_a1_dt = -2.0 * 4.222e-2 * d_t623_dt * (1.0 - 4.222e-2 * t623)
            + 2.643e-5 * d_t653_dt;
        let b1_denom = a1 * t623;
        let d_b1_denom_dt = d_a1_dt * t623 + a1 * d_t623_dt;
        let b1_num = 16.16 * flexmesa_math::exp(-134.92 / t613);
        let d_b1_num_dt = b1_num * 134.92 * d_t613_dt / (t613 * t613);
        let mut b1 = b1_num / b1_denom;
        let mut db1_dt = d_b1_num_dt / b1_denom - b1 * d_b1_denom_dt / b1_denom;

        let a2 = flexmesa_math::pow2(1.0 - 2.807e-2 * t623) + 2.704e-6 * t653;
        let d_a2_dt = -2.0 * 2.807e-2 * d_t623_dt * (1.0 - 2.807e-2 * t623)
            + 2.704e-6 * d_t653_dt;
        let b2_denom = a2 * t623;
        let d_b2_denom_dt = d_a2_dt * t623 + a2 * d_t623_dt;
        let b2_num = 244.6
            * flexmesa_math::pow5(1.0 + 3.528e-3 * t623)
            * flexmesa_math::exp(-235.72 / t613);
        let d_b2_num_dt = b2_num
            * (5.0 * 3.528e-3 * d_t623_dt / (1.0 + 3.528e-3 * t623)
                + 235.72 * d_t613_dt / t623);
        let mut b2 = b2_num / b2_denom;
        let mut db2_dt = d_b2_num_dt / b2_denom - b2 * d_b2_denom_dt / b2_denom;

        if 5.458e3 > r6t {
            let e1 = -1065.1 / t6;
            let d_e1_dt = -e1 * d_t6_dt / t6;
            let f1 = flexmesa_math::exp(e1) / t632;
            let d_f1_dt = f1 * (d_e1_dt - d_t632_dt / t632);
            b1 += f1;
            db1_dt += d_f1_dt;
        }
        if 1.836e4 > r6t {
            let e2 = -3336.4 / t6;
            let d_e2_dt = -e2 * d_t6_dt / t6;
            let f2 = flexmesa_math::exp(e2) / t632;
            let d_f2_dt = f2 * (d_e2_dt - d_t632_dt / t632);
            b2 += f2;
            db2_dt += d_f2_dt;
        }
        let g1 = b1 * flexmesa_math::exp(60.492 * r6t13 / t6);
        let dg1_dt = g1 * (db1_dt / b1 - 60.492 * r6t13 * d_t6_dt / (t6 * t6));
        let g2 = b2 * flexmesa_math::exp(106.35 * r6t13 / t6);
        let dg2_dt = g2 * (db2_dt / b2 - 106.35 * r6t13 * d_t6_dt / (t6 * t6));
        // Upstream `pycno.f90:161, 165` sets `dG1dRho = 0`, `dG2dRho = 0` for `U < 1`.
        ((g1, dg1_dt, 0.0), (g2, dg2_dt, 0.0))
    } else {
        let af = 1.0 / u32 + 1.0;
        let d_af_dt = -1.5 * d_u_dt / u52;
        let d_af_drho = -1.5 * d_u_drho / u52;

        let b1_denom = t612
            * (flexmesa_math::pow2(1.0 - 5.680e-2 * r6t13) + 8.815e-7 * t62);
        let d_b1_denom_dt =
            b1_denom * d_t612_dt / t612 + t612 * 8.815e-7 * d_t62_dt;
        let d_b1_denom_drho =
            -2.0 * 5.680e-2 * (1.0 - 5.680e-2 * r6t13) * t612 * d_r6t13_drho;

        let b1 = 1.178 * af * flexmesa_math::exp(-77.554 / r6t16) / b1_denom;
        let db1_dt = b1 * (d_af_dt / af - d_b1_denom_dt / b1_denom);
        let db1_drho = b1
            * (d_af_drho / af + 77.554 * d_r6t16_drho / (r6t16 * r6t16)
                - d_b1_denom_drho / b1_denom);

        let b2_denom = t612
            * (flexmesa_math::pow2(1.0 - 3.791e-2 * r6t13) + 5.162e-8 * t62);
        let d_b2_denom_dt =
            b2_denom * d_t612_dt / t612 + t612 * 5.162e-8 * d_t62_dt;
        let tmp = flexmesa_math::pow(rho / ue, ONE_THIRD);
        let d_b2_denom_drho =
            t612 * (-0.000252733 + 9.58112e-8 * tmp) * tmp / rho;

        let b2 = 13.48
            * af
            * flexmesa_math::pow5(1.0 + 5.070e-3 * r6t13)
            * flexmesa_math::exp(-135.08 / r6t16)
            / b2_denom;
        let db2_dt = b2 * (d_af_dt / af - d_b2_denom_dt / b2_denom);
        let db2_drho = b2
            * (d_af_drho / af + 135.08 * d_r6t16_drho / (r6t16 * r6t16)
                - d_b2_denom_drho / b2_denom);

        let g1_tuple = if 5.458e3 > r6t {
            let e1 = (60.492 * r6t13 - 1065.1) / t6;
            let d_e1_dt = -e1 * d_t6_dt / t6;
            let d_e1_drho = 60.492 * d_r6t13_drho / t6;
            let f1 = flexmesa_math::exp(e1) / t632;
            let d_f1_dt = f1 * (d_e1_dt - d_t632_dt / t632);
            let d_f1_drho = f1 * d_e1_drho;
            (b1 + f1, db1_dt + d_f1_dt, db1_drho + d_f1_drho)
        } else {
            (b1, db1_dt, db1_drho)
        };

        let g2_tuple = if 1.836e4 > r6t {
            let e2 = (106.35 * r6t13 - 3336.4) / t6;
            let d_e2_dt = -e2 * d_t6_dt / t6;
            let d_e2_drho = 106.35 * d_r6t13_drho / t6;
            let f2 = flexmesa_math::exp(e2) / t632;
            let d_f2_dt = f2 * (d_e2_dt - d_t632_dt / t632);
            let d_f2_drho = f2 * d_e2_drho;
            (b2 + f2, db2_dt + d_f2_dt, db2_drho + d_f2_drho)
        } else {
            (b2, db2_dt, db2_drho)
        };
        (g1_tuple, g2_tuple)
    };

    let r = 5.120e29 * g1 * g2 * y * y * y * r6 * r6;
    if r < 1.0e-99 || g1 < 1.0e-99 || g2 < 1.0e-99 {
        (r, 0.0, 0.0)
    } else {
        let dr_dt = r * (dg1_dt / g1 + dg2_dt / g2);
        let dr_drho =
            r * (dg1_drho / g1 + dg2_drho / g2 + 2.0 * d_r6_drho / r6);
        (r, dr_dt, dr_drho)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::{
        eval_reaction_raw, eval_reaction_raw_with_overrides, RateTable1D,
    };
    use crate::loader::RatesCatalog;
    use crate::tfactors::{eval_reaclib_inverse_ratio, TFactors};

    #[test]
    fn test_rate_table_1d_parses_comments_fortran_exponents_and_clamps() {
        let sample = "\
# synthetic rate table
! second comment

   4
  0.5D0   1.0D-10   999.0
  1.0d0   2.0d-10
  2.0E0   8.0E-10
  4.0e0   3.2e-9
";
        let tbl = RateTable1D::from_str_table(sample).expect("parse sample");
        assert_eq!(tbl.t8s, vec![0.5, 1.0, 2.0, 4.0]);
        assert!((tbl.eval(0.5e8).unwrap().0 - 1.0e-10).abs() <= 1.0e-24);
        assert!((tbl.eval(2.0e8).unwrap().0 - 8.0e-10).abs() <= 1.0e-24);
        // Endpoint clamping matches MESA `interp_values` (`interp_1d_misc.f90:144-149`)
        assert!((tbl.eval(0.1e8).unwrap().0 - 1.0e-10).abs() <= 1.0e-24);
        assert!((tbl.eval(10.0e8).unwrap().0 - 3.2e-9).abs() <= 1.0e-23);
        let (mid, slope) = tbl.eval(1.5e8).unwrap();
        assert!(mid > 2.0e-10 && mid < 8.0e-10);
        assert!(slope > 0.0);
    }

    #[test]
    fn test_rate_table_1d_loads_bundled_deboer_c12ag_and_cf88_3a() {
        assert!(RateTable1D::exists(
            "c12ag_deboer_sigma_0p0_2000_Tgrid.dat",
            None
        ));
        assert!(RateTable1D::exists("r_he4_he4_he4_to_c12_cf88.txt", None));
        assert!(!RateTable1D::exists(
            "nonexistent_special_rate_xyz.dat",
            None
        ));
        assert!(RateTable1D::resolve_and_load(
            "nonexistent_special_rate_xyz.dat",
            None
        )
        .is_err());

        let c12ag = RateTable1D::resolve_and_load(
            "c12ag_deboer_sigma_0p0_2000_Tgrid.dat",
            None,
        )
        .expect("load deboer c12ag");
        assert_eq!(c12ag.t8s.len(), 2015);
        assert!((c12ag.t8s[0] - 0.01).abs() <= 1.0e-15);
        assert!((c12ag.t8s[2014] - 100.0).abs() <= 1.0e-12);

        let cf88_3a = RateTable1D::resolve_and_load(
            "r_he4_he4_he4_to_c12_cf88.txt",
            None,
        )
        .expect("load cf88 3a");
        assert_eq!(cf88_3a.t8s.len(), 10001);

        let cat = RatesCatalog::builtin();
        let temp = 1.5e8_f64;
        let tf = TFactors::from_temp(temp);
        let c12_id = cat.reaction_id_by_name("r_c12_ag_o16").unwrap();
        let jina_c12 = eval_reaction_raw(
            cat,
            cat.reaction_by_id(c12_id).unwrap(),
            temp,
            &tf,
        )
        .unwrap();
        let deboer_c12 = c12ag.eval(temp).unwrap().0;
        assert!(deboer_c12 > 0.0 && jina_c12 > 0.0);
        // DeBoer (2017) differs measurably from JINA REACLIB at He-burning T=1.5e8 K
        assert!(
            ((deboer_c12 - jina_c12) / jina_c12).abs() > 1.0e-2,
            "deboer={deboer_c12:e} jina={jina_c12:e}"
        );
    }

    #[test]
    fn test_eval_reaction_raw_with_overrides_forward_reverse_and_error() {
        let cat = RatesCatalog::builtin();
        let c12ag = Arc::new(
            RateTable1D::resolve_and_load(
                "c12ag_deboer_sigma_0p0_2000_Tgrid.dat",
                None,
            )
            .expect("c12ag"),
        );
        let cf88_3a = Arc::new(
            RateTable1D::resolve_and_load(
                "r_he4_he4_he4_to_c12_cf88.txt",
                None,
            )
            .expect("cf88_3a"),
        );
        let mut overrides = HashMap::new();
        overrides.insert(String::from("r_c12_ag_o16"), Arc::clone(&c12ag));
        overrides
            .insert(String::from("r_he4_he4_he4_to_c12"), Arc::clone(&cf88_3a));

        let temp = 3.0e8_f64;
        let tf = TFactors::from_temp(temp);

        // 1. Forward c12(a,g)o16 directly matches table
        let info_c12ag = cat
            .reaction_by_id(cat.reaction_id_by_name("r_c12_ag_o16").unwrap())
            .unwrap();
        let fwd_c12 = eval_reaction_raw_with_overrides(
            cat,
            info_c12ag,
            temp,
            &tf,
            &overrides,
            Some("r_o16_ga_c12"),
        )
        .unwrap();
        assert_eq!(fwd_c12.to_bits(), c12ag.eval(temp).unwrap().0.to_bits());

        // 2. Reverse o16(g,a)c12 uses detailed balance `inv_lambda * fr_table`
        let info_o16ga = cat
            .reaction_by_id(cat.reaction_id_by_name("r_o16_ga_c12").unwrap())
            .unwrap();
        let rev_o16 = eval_reaction_raw_with_overrides(
            cat,
            info_o16ga,
            temp,
            &tf,
            &overrides,
            Some("r_c12_ag_o16"),
        )
        .unwrap();
        let (_, c12_terms) = cat.reaclib_terms("r_c12_ag_o16").unwrap();
        let expected_rev_o16 =
            eval_reaclib_inverse_ratio(&c12_terms[0], tf.t9) * fwd_c12;
        assert_eq!(rev_o16.to_bits(), expected_rev_o16.to_bits());

        // 3. Reverse c12 -> 3 alpha uses detailed balance from cf88 table
        let info_c12_3a = cat
            .reaction_by_id(
                cat.reaction_id_by_name("r_c12_to_he4_he4_he4").unwrap(),
            )
            .unwrap();
        let rev_3a = eval_reaction_raw_with_overrides(
            cat,
            info_c12_3a,
            temp,
            &tf,
            &overrides,
            Some("r_he4_he4_he4_to_c12"),
        )
        .unwrap();
        let (_, t3a_terms) = cat.reaclib_terms("r_he4_he4_he4_to_c12").unwrap();
        let expected_rev_3a = eval_reaclib_inverse_ratio(&t3a_terms[0], tf.t9)
            * cf88_3a.eval(temp).unwrap().0;
        assert_eq!(rev_3a.to_bits(), expected_rev_3a.to_bits());

        // 4. Forward from reverse-only override must fail loudly (`raw_rates.f90:104-112`)
        let mut rev_only = HashMap::new();
        rev_only.insert(String::from("r_o16_ga_c12"), Arc::clone(&c12ag));
        let err = eval_reaction_raw_with_overrides(
            cat,
            info_c12ag,
            temp,
            &tf,
            &rev_only,
            Some("r_o16_ga_c12"),
        )
        .expect_err("forward from reverse-only table must fail");
        assert!(err.to_string().contains("detailed balance"));
    }

    #[test]
    fn test_eval_fl_epsnuc_3alf_with_derivs_mesa_hex_zero_ulp() {
        use super::{eval_fl_epsnuc_3alf, eval_fl_epsnuc_3alf_with_derivs};
        use flexmesa_math::{with_numerics_mode, NumericsMode};

        let cases = [
            (
                7.943_282_347_242_821_8e7_f64,
                3.162_277_691_791_155_8e9_f64,
                0x4428_3AEE_9D5C_6665_u64,
                0x42CA_AACB_9AA9_45FA_u64,
                0x4269_3EE0_FF7E_73DA_u64,
            ),
            (
                1.0e8_f64,
                5.0e8_f64,
                0x4333_B486_40DD_E653_u64,
                0x41D6_EC6D_E5F2_81CB_u64,
                0x4175_2884_50EB_FEF7_u64,
            ),
            (
                flexmesa_math::exp10(8.191366),
                flexmesa_math::exp10(9.893157),
                0x44DE_BCD8_F480_6EAF_u64,
                0x4344_21E3_9F5C_E493_u64,
                0x42E0_E24A_7605_ABC1_u64,
            ),
        ];
        with_numerics_mode(NumericsMode::Strict, || {
            for (i, &(t, rho, exp_r, exp_dt, exp_drho)) in
                cases.iter().enumerate()
            {
                let r_only = eval_fl_epsnuc_3alf(t, rho, 1.0, 2.0);
                let (r, dr_dt, dr_drho) =
                    eval_fl_epsnuc_3alf_with_derivs(t, rho, 1.0, 2.0);
                assert_eq!(r_only.to_bits(), exp_r, "case {i} r_only");
                assert_eq!(r.to_bits(), exp_r, "case {i} r");
                assert_eq!(dr_dt.to_bits(), exp_dt, "case {i} dr_dt");
                assert_eq!(dr_drho.to_bits(), exp_drho, "case {i} dr_drho");
            }
        });
    }
}
