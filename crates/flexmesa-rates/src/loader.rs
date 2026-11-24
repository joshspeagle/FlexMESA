// SPDX-License-Identifier: GPL-3.0-or-later
//! Lazy loader and catalog for the embedded binary nuclear reaction rate table
//! bundle (`crates/flexmesa-rates/data/rates_tables.bin`).

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use flexmesa_chem::{NuclideCatalog, NuclideEntry};
use flexmesa_const::{AVO, BOLTZM, HBAR, MEV_TO_ERGS, PI};
use flexmesa_interp1d::CubicCoeffs;

use crate::raw_rates::eval_interp_rate_table;
use crate::types::{RatesError, ReaclibTerm, ReactionInfo};

fn parse_reaclib_handle<'a>(
    nuclides: &'a NuclideCatalog,
    handle: &str,
) -> Option<(Vec<&'a NuclideEntry>, Vec<&'a NuclideEntry>)> {
    let rest = handle.trim().strip_prefix("r_")?;
    let mut in_isos: Vec<&'a NuclideEntry> = Vec::with_capacity(4);
    let mut out_isos: Vec<&'a NuclideEntry> = Vec::with_capacity(4);
    let mut doing_in = true;

    let light_carrier = |b: u8| -> Option<Option<&'static str>> {
        match b {
            b'p' => Some(Some("h1")),
            b'a' => Some(Some("he4")),
            b'n' => Some(Some("neut")),
            b'g' => Some(None),
            _ => None,
        }
    };

    for tok in rest.split('_') {
        let iso_name = if tok == "prot" { "h1" } else { tok };
        if let Some(entry) = nuclides.get_by_name(iso_name) {
            if doing_in {
                in_isos.push(entry);
            } else {
                out_isos.push(entry);
            }
        } else if doing_in {
            let bytes = tok.as_bytes();
            if bytes.len() == 2 {
                if let (Some(ex_in), Some(ex_out)) =
                    (light_carrier(bytes[0]), light_carrier(bytes[1]))
                {
                    if let Some(in_name) = ex_in {
                        if in_isos.len() != 1 {
                            return None;
                        }
                        let ex = nuclides.get_by_name(in_name)?;
                        if in_isos[0].z >= ex.z {
                            in_isos.insert(0, ex);
                        } else {
                            in_isos.push(ex);
                        }
                    }
                    doing_in = false;
                    if let Some(out_name) = ex_out {
                        out_isos.push(nuclides.get_by_name(out_name)?);
                    }
                    continue;
                }
            }
            doing_in = false;
        } else {
            return None;
        }
    }
    if in_isos.is_empty() || out_isos.is_empty() {
        return None;
    }
    Some((in_isos, out_isos))
}

fn compute_reaclib_rev_coeffs(
    in_isos: &[&NuclideEntry],
    out_isos: &[&NuclideEntry],
    fac: f64,
) -> (f64, f64, [f64; 24]) {
    // `rates/private/reaclib_support.f90:204-262` (`compute_rev_ratio`)
    let m_in: f64 = in_isos.iter().map(|e| e.w).product();
    let m_out: f64 = out_isos.iter().map(|e| e.w).product();
    let g_in: f64 = in_isos.iter().map(|e| 2.0 * e.spin + 1.0).product();
    let g_out: f64 = out_isos.iter().map(|e| 2.0 * e.spin + 1.0).product();
    let tmp = m_in / m_out;
    let mut c1 = flexmesa_math::pow_strict(tmp, 1.5) * (g_in / g_out);
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let inv_exp = (in_isos.len() as i32) - (out_isos.len() as i32);
    if inv_exp != 0 {
        c1 *= flexmesa_math::powi_strict(fac, inv_exp);
    }
    let inv_c1 = flexmesa_math::log_strict(c1);

    let sum1: f64 = in_isos.iter().map(|e| e.binding_energy).sum();
    let sum2: f64 = out_isos.iter().map(|e| e.binding_energy).sum();
    let inv_c2 = (sum1 - sum2) * MEV_TO_ERGS / BOLTZM / 1.0e9;

    let mut inv_part = [1.0_f64; 24];
    for (k, slot) in inv_part.iter_mut().enumerate() {
        let p_in: f64 = in_isos.iter().map(|e| e.pfcn[k]).product();
        let p_out: f64 = out_isos.iter().map(|e| e.pfcn[k]).product();
        *slot = p_in / p_out;
    }
    (inv_c1, inv_c2, inv_part)
}

const RAW_BUNDLE: &[u8] = include_bytes!("../data/rates_tables.bin");
const AIC_STATES_SRC: &str = include_str!("../data/aic.states");
const AIC_TRANSITIONS_SRC: &str = include_str!("../data/aic.transitions");

fn normalize_ecapture_iso(name: &str) -> &str {
    match name.trim() {
        "al-6" => "al26-1",
        "al*6" => "al26-2",
        other => other,
    }
}

fn collect_ecapture_tokens(src: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for line in src.lines() {
        let stripped = line.split('!').next().unwrap_or("");
        for tok in stripped.split_whitespace() {
            tokens.push(tok.to_string());
        }
    }
    tokens
}

fn parse_ecapture_f64(tok: &str, what: &str) -> Result<f64, RatesError> {
    let norm = if tok.contains('d') || tok.contains('D') {
        tok.replace(['d', 'D'], "e")
    } else {
        tok.to_string()
    };
    norm.parse::<f64>().map_err(|_| {
        RatesError::InvalidTableData(format!("invalid {what} value '{tok}'"))
    })
}

/// Nuclear level schemes and weak transition $\log ft$ tables loaded from MESA
/// `.states` / `.transitions` files (`rates/private/load_ecapture.f90`).
///
/// The built-in singleton embeds upstream MESA `26.4.1`'s `wd_aic`
/// (`aic.states` and `aic.transitions`) catalog, which is a strict superset
/// of `rates/test/test_special.{states,transitions}`.
#[derive(Debug, Clone)]
pub struct EcaptureTables {
    states: HashMap<String, Vec<(f64, f64)>>,
    trans: HashMap<(String, String), Vec<(usize, usize, f64)>>,
}

static BUILTIN_ECAPTURE: OnceLock<EcaptureTables> = OnceLock::new();

impl EcaptureTables {
    /// Returns the built-in [`EcaptureTables`] parsed from `aic.states` and
    /// `aic.transitions`.
    #[must_use]
    pub fn builtin() -> &'static Self {
        BUILTIN_ECAPTURE.get_or_init(|| {
            Self::from_str_pair(AIC_STATES_SRC, AIC_TRANSITIONS_SRC)
                .expect("built-in aic.states / aic.transitions must parse")
        })
    }

    /// Parses special weak states and transitions tables from in-memory MESA
    /// `.states` and `.transitions` file text (`load_ecapture.f90`).
    ///
    /// # Errors
    /// Returns [`RatesError::InvalidTableData`] if syntax, counts, numeric
    /// fields, or 1-based state indices are malformed.
    pub fn from_str_pair(
        states_src: &str,
        trans_src: &str,
    ) -> Result<Self, RatesError> {
        let st_toks = collect_ecapture_tokens(states_src);
        let mut states: HashMap<String, Vec<(f64, f64)>> = HashMap::new();
        let mut idx = 0_usize;
        while idx < st_toks.len() {
            let iso = normalize_ecapture_iso(&st_toks[idx]).to_string();
            idx += 1;
            let n_tok = st_toks.get(idx).ok_or_else(|| {
                RatesError::InvalidTableData(format!(
                    "missing state count for '{iso}'"
                ))
            })?;
            idx += 1;
            let nstates = n_tok.parse::<usize>().map_err(|_| {
                RatesError::InvalidTableData(format!(
                    "invalid state count '{n_tok}' for '{iso}'"
                ))
            })?;
            let mut levels = Vec::with_capacity(nstates);
            for k in 0..nstates {
                let e_tok = st_toks.get(idx).ok_or_else(|| {
                    RatesError::InvalidTableData(format!(
                        "missing energy for '{iso}' level {}",
                        k + 1
                    ))
                })?;
                let j_tok = st_toks.get(idx + 1).ok_or_else(|| {
                    RatesError::InvalidTableData(format!(
                        "missing spin J for '{iso}' level {}",
                        k + 1
                    ))
                })?;
                idx += 2;
                let e = parse_ecapture_f64(e_tok, "ecapture level energy")?;
                let j = parse_ecapture_f64(j_tok, "ecapture level spin")?;
                levels.push((e, j));
            }
            states.insert(iso, levels);
        }

        let tr_toks = collect_ecapture_tokens(trans_src);
        let mut trans: HashMap<(String, String), Vec<(usize, usize, f64)>> =
            HashMap::new();
        let mut tidx = 0_usize;
        while tidx < tr_toks.len() {
            let lhs = normalize_ecapture_iso(&tr_toks[tidx]).to_string();
            let rhs = tr_toks
                .get(tidx + 1)
                .ok_or_else(|| {
                    RatesError::InvalidTableData(format!(
                        "missing rhs isotope after '{lhs}'"
                    ))
                })
                .map(|s| normalize_ecapture_iso(s).to_string())?;
            let n_tok = tr_toks.get(tidx + 2).ok_or_else(|| {
                RatesError::InvalidTableData(format!(
                    "missing transition count for '{lhs}->{rhs}'"
                ))
            })?;
            tidx += 3;
            let ntrans = n_tok.parse::<usize>().map_err(|_| {
                RatesError::InvalidTableData(format!(
                    "invalid transition count '{n_tok}' for '{lhs}->{rhs}'"
                ))
            })?;
            let lhs_n = states.get(&lhs).map_or(0, Vec::len);
            let rhs_n = states.get(&rhs).map_or(0, Vec::len);
            let mut items = Vec::with_capacity(ntrans);
            for k in 0..ntrans {
                let si_tok = tr_toks.get(tidx).ok_or_else(|| {
                    RatesError::InvalidTableData(format!(
                        "missing Si for '{lhs}->{rhs}' transition {}",
                        k + 1
                    ))
                })?;
                let sf_tok = tr_toks.get(tidx + 1).ok_or_else(|| {
                    RatesError::InvalidTableData(format!(
                        "missing Sf for '{lhs}->{rhs}' transition {}",
                        k + 1
                    ))
                })?;
                let logft_tok = tr_toks.get(tidx + 2).ok_or_else(|| {
                    RatesError::InvalidTableData(format!(
                        "missing logft for '{lhs}->{rhs}' transition {}",
                        k + 1
                    ))
                })?;
                tidx += 3;
                let si = si_tok.parse::<usize>().map_err(|_| {
                    RatesError::InvalidTableData(format!(
                        "invalid Si '{si_tok}' for '{lhs}->{rhs}'"
                    ))
                })?;
                let sf = sf_tok.parse::<usize>().map_err(|_| {
                    RatesError::InvalidTableData(format!(
                        "invalid Sf '{sf_tok}' for '{lhs}->{rhs}'"
                    ))
                })?;
                if si == 0
                    || sf == 0
                    || (lhs_n > 0 && si > lhs_n)
                    || (rhs_n > 0 && sf > rhs_n)
                {
                    return Err(RatesError::InvalidTableData(format!(
                        "out-of-range state index ({si}, {sf}) for '{lhs}->{rhs}' ({lhs_n}, {rhs_n})"
                    )));
                }
                let logft = parse_ecapture_f64(logft_tok, "ecapture logft")?;
                items.push((si, sf, logft));
            }
            trans.insert((lhs, rhs), items);
        }

        Ok(Self { states, trans })
    }

    /// Loads special weak states and transitions tables from disk files.
    ///
    /// # Errors
    /// Returns [`RatesError::InvalidTableData`] if either file cannot be read
    /// or fails parsing.
    pub fn from_files(
        states_path: impl AsRef<Path>,
        trans_path: impl AsRef<Path>,
    ) -> Result<Self, RatesError> {
        let sp = states_path.as_ref();
        let tp = trans_path.as_ref();
        let states_src = fs::read_to_string(sp).map_err(|e| {
            RatesError::InvalidTableData(format!(
                "failed to read states file {}: {e}",
                sp.display()
            ))
        })?;
        let trans_src = fs::read_to_string(tp).map_err(|e| {
            RatesError::InvalidTableData(format!(
                "failed to read transitions file {}: {e}",
                tp.display()
            ))
        })?;
        Self::from_str_pair(&states_src, &trans_src)
    }

    /// Returns all parsed nuclear level tables keyed by nuclide name.
    #[must_use]
    pub const fn states(&self) -> &HashMap<String, Vec<(f64, f64)>> {
        &self.states
    }

    /// Returns all parsed `(Si, Sf, logft)` transitions keyed by `(lhs, rhs)`.
    #[must_use]
    pub const fn transitions(
        &self,
    ) -> &HashMap<(String, String), Vec<(usize, usize, f64)>> {
        &self.trans
    }

    /// Resolves `(states_lhs, states_rhs, transitions, z_lhs, z_rhs, q_tot)`
    /// for `(lhs, rhs)` using nuclear mass excesses from [`NuclideCatalog`].
    ///
    /// # Errors
    /// Returns [`RatesError::UnknownWeakPair`] if `(lhs, rhs)` is absent from
    /// either table, or [`RatesError::InvalidTableData`] if either isotope is
    /// missing from [`NuclideCatalog`].
    pub fn pair_data(
        &self,
        lhs: &str,
        rhs: &str,
    ) -> Result<
        (
            &[(f64, f64)],
            &[(f64, f64)],
            &[(usize, usize, f64)],
            f64,
            f64,
            f64,
        ),
        RatesError,
    > {
        let l_key = normalize_ecapture_iso(lhs).to_string();
        let r_key = normalize_ecapture_iso(rhs).to_string();
        let key = (l_key.clone(), r_key.clone());
        let Some(trans) = self.trans.get(&key) else {
            return Err(RatesError::UnknownWeakPair(key.0, key.1));
        };
        let Some(s_lhs) = self.states.get(&key.0) else {
            return Err(RatesError::UnknownWeakPair(key.0, key.1));
        };
        let Some(s_rhs) = self.states.get(&key.1) else {
            return Err(RatesError::UnknownWeakPair(key.0, key.1));
        };
        let nuclides = NuclideCatalog::builtin_ref();
        let id_l = nuclides.get_nuclide_index(&l_key).ok_or_else(|| {
            RatesError::InvalidTableData(format!(
                "unknown lhs nuclide '{l_key}'"
            ))
        })?;
        let id_r = nuclides.get_nuclide_index(&r_key).ok_or_else(|| {
            RatesError::InvalidTableData(format!(
                "unknown rhs nuclide '{r_key}'"
            ))
        })?;
        let n_lhs = nuclides.get_by_id(id_l).ok_or_else(|| {
            RatesError::InvalidTableData(format!("unknown lhs_id {id_l}"))
        })?;
        let n_rhs = nuclides.get_by_id(id_r).ok_or_else(|| {
            RatesError::InvalidTableData(format!("unknown rhs_id {id_r}"))
        })?;
        #[allow(clippy::cast_precision_loss)]
        Ok((
            s_lhs,
            s_rhs,
            trans,
            n_lhs.z as f64,
            n_rhs.z as f64,
            n_lhs.mass_excess - n_rhs.mass_excess,
        ))
    }
}

/// A 2D tabulated weak reaction table (`weakreactions.tables` or `S13_r_be7_wk_li7.h5`).
#[derive(Debug, Clone)]
pub struct WeakTable {
    /// 1-based `WeakTable` index (`1..=463`).
    pub id: usize,
    /// 1-based reactant `chem_id`.
    pub lhs_id: u16,
    /// 1-based product `chem_id`.
    pub rhs_id: u16,
    /// Reactant isotope name.
    pub lhs_name: String,
    /// Product isotope name.
    pub rhs_name: String,
    /// Total nuclear $Q$-value $\Delta M_{\mathrm{lhs}} - \Delta M_{\mathrm{rhs}}$ ($\mathrm{MeV}$).
    pub q_tot: f64,
    /// $T_9$ grid (`len = nt9`).
    pub t9s: Vec<f64>,
    /// $\log_{10}(\rho Y_e)$ grid (`len = nlye`).
    pub lyerhos: Vec<f64>,
    /// Flattened `(nt9, nlye)` $\log_{10} \lambda_{\mathrm{decay}}$ table.
    pub ldecay: Vec<f64>,
    /// Flattened `(nt9, nlye)` $\log_{10} \lambda_{\mathrm{capture}}$ table.
    pub lcapture: Vec<f64>,
    /// Flattened `(nt9, nlye)` $\log_{10} \epsilon_\nu$ table.
    pub lneutrino: Vec<f64>,
}

/// A 2D dense Suzuki et al. (2016) `sd`-shell weak reaction table (`Suzuki2016.h5`).
#[derive(Debug, Clone)]
pub struct SuzukiTable {
    /// 1-based reactant `chem_id`.
    pub lhs_id: u16,
    /// 1-based product `chem_id`.
    pub rhs_id: u16,
    /// Total nuclear $Q$-value ($\mathrm{MeV}$).
    pub q_tot: f64,
    /// $\log_{10} T$ grid.
    pub logts: Vec<f64>,
    /// $\log_{10}(\rho Y_e)$ grid.
    pub lyerhos: Vec<f64>,
    /// Whether the `decay` dataset group is present.
    pub has_decay: bool,
    /// Whether the `capture` dataset group is present.
    pub has_capture: bool,
    /// Flattened `(nlogt, nlye)` $\log_{10} \lambda_{\mathrm{decay}}$ table.
    pub decay_rate: Vec<f64>,
    /// Flattened `(nlogt, nlye)` $\log_{10} \epsilon_{\nu,\mathrm{decay}}$ table.
    pub decay_nu: Vec<f64>,
    /// Flattened `(nlogt, nlye)` $\log_{10} \lambda_{\mathrm{capture}}$ table.
    pub capture_rate: Vec<f64>,
    /// Flattened `(nlogt, nlye)` $\log_{10} \epsilon_{\nu,\mathrm{capture}}$ table.
    pub capture_nu: Vec<f64>,
}

/// Complete lazily initialized nuclear rates catalog.
#[derive(Debug)]
pub struct RatesCatalog {
    /// All 316 reaction metadata entries (`1..=316`, 0-indexed `0..316`).
    pub reactions: Vec<ReactionInfo>,
    reaction_by_name: HashMap<String, usize>,
    rate_tables: HashMap<usize, (Vec<f64>, Vec<CubicCoeffs>)>,
    reaclib_handles: HashMap<String, (bool, Vec<ReaclibTerm>)>,
    weak_info: HashMap<(u16, u16), (f64, f64)>,
    /// All 463 weak tables in MESA order (`1..=463`).
    pub weak_tables: Vec<WeakTable>,
    weak_pair_map: HashMap<(String, String), usize>,
    suzuki_pair_map: HashMap<(String, String), SuzukiTable>,
}

static CATALOG: OnceLock<RatesCatalog> = OnceLock::new();

impl RatesCatalog {
    /// Returns a reference to the singleton [`RatesCatalog`].
    #[must_use]
    pub fn builtin() -> &'static Self {
        CATALOG.get_or_init(Self::parse_builtin)
    }

    /// Looks up a [`ReactionInfo`] by 1-based reaction ID `ir` (`1..=316`).
    ///
    /// # Errors
    /// Returns [`RatesError::UnknownReaction`] if `ir` is out of bounds.
    pub fn reaction_by_id(
        &self,
        ir: usize,
    ) -> Result<&ReactionInfo, RatesError> {
        if ir == 0 || ir > self.reactions.len() {
            return Err(RatesError::UnknownReaction(format!("id={ir}")));
        }
        Ok(&self.reactions[ir - 1])
    }

    /// Looks up a 1-based reaction ID `ir` (`1..=316`) by reaction handle name.
    #[must_use]
    pub fn reaction_id_by_name(&self, name: &str) -> Option<usize> {
        self.reaction_by_name.get(name.trim()).copied()
    }

    /// Returns `(is_reverse, &[ReaclibTerm])` for a REACLIB reaction handle.
    ///
    /// # Errors
    /// Returns [`RatesError::UnknownReaction`] if `handle` is not found.
    pub fn reaclib_terms(
        &self,
        handle: &str,
    ) -> Result<(bool, &[ReaclibTerm]), RatesError> {
        self.reaclib_handles
            .get(handle.trim())
            .map(|(rev, v)| (*rev, v.as_slice()))
            .ok_or_else(|| RatesError::UnknownReaction(handle.to_string()))
    }

    /// Evaluates a 1D `interp_pm` rate table for 1-based reaction ID `ir` at `temp` ($\mathrm{K}$).
    ///
    /// # Errors
    /// Returns [`RatesError::UnknownReaction`] if `ir` has no tabulated rate
    /// table or [`RatesError::InvalidTableData`] if interpolation fails.
    pub fn eval_rate_table(
        &self,
        ir: usize,
        temp: f64,
    ) -> Result<(f64, f64), RatesError> {
        let (t8s, f1) = self.rate_tables.get(&ir).ok_or_else(|| {
            RatesError::UnknownReaction(format!("rate_table id={ir}"))
        })?;
        eval_interp_rate_table(t8s, f1, temp)
    }

    /// Looks up `(halflife, qneu)` from `weak_info.list` by `(lhs_id, rhs_id)`.
    #[must_use]
    pub fn weak_info_by_ids(
        &self,
        lhs_id: u16,
        rhs_id: u16,
    ) -> Option<(f64, f64)> {
        self.weak_info.get(&(lhs_id, rhs_id)).copied()
    }

    /// Looks up a [`WeakTable`] by `(lhs_name, rhs_name)`.
    ///
    /// # Errors
    /// Returns [`RatesError::UnknownWeakPair`] if the pair is not in `weak_tables`.
    pub fn weak_by_pair(
        &self,
        lhs: &str,
        rhs: &str,
    ) -> Result<&WeakTable, RatesError> {
        let key = (lhs.trim().to_string(), rhs.trim().to_string());
        let &idx = self
            .weak_pair_map
            .get(&key)
            .ok_or(RatesError::UnknownWeakPair(key.0, key.1))?;
        Ok(&self.weak_tables[idx])
    }

    /// Looks up a [`SuzukiTable`] by `(lhs_name, rhs_name)`.
    #[must_use]
    pub fn suzuki_by_pair(&self, lhs: &str, rhs: &str) -> Option<&SuzukiTable> {
        self.suzuki_pair_map
            .get(&(lhs.trim().to_string(), rhs.trim().to_string()))
    }

    /// Returns `(states_lhs, states_rhs, transitions, z_lhs, z_rhs, q_tot)` for
    /// special `ecapture` transitions (`aic.states` / `aic.transitions`, which
    /// embeds both `wd_aic` and `test_special` level schemes).
    ///
    /// # Errors
    /// Returns [`RatesError::UnknownWeakPair`] if `(lhs, rhs)` is absent from
    /// the ecapture tables, or [`RatesError::InvalidTableData`] if either
    /// isotope is missing from [`NuclideCatalog`].
    pub fn special_ecapture_data(
        &self,
        lhs: &str,
        rhs: &str,
    ) -> Result<
        (
            &[(f64, f64)],
            &[(f64, f64)],
            &[(usize, usize, f64)],
            f64,
            f64,
            f64,
        ),
        RatesError,
    > {
        let _ = self;
        EcaptureTables::builtin().pair_data(lhs, rhs)
    }

    #[allow(clippy::too_many_lines)]
    fn parse_builtin() -> Self {
        let mut pos = 8_usize; // skip b"FXRT\x01\x00\x00\x00"
        let read_u8 = |p: &mut usize| -> u8 {
            let v = RAW_BUNDLE[*p];
            *p += 1;
            v
        };
        let read_i8 = |p: &mut usize| -> i8 { read_u8(p) as i8 };
        let read_u16 = |p: &mut usize| -> u16 {
            let v = u16::from_le_bytes([RAW_BUNDLE[*p], RAW_BUNDLE[*p + 1]]);
            *p += 2;
            v
        };
        let read_u32 = |p: &mut usize| -> u32 {
            let v = u32::from_le_bytes(
                RAW_BUNDLE[*p..*p + 4].try_into().unwrap_or([0; 4]),
            );
            *p += 4;
            v
        };
        let read_f64 = |p: &mut usize| -> f64 {
            let v = f64::from_le_bytes(
                RAW_BUNDLE[*p..*p + 8].try_into().unwrap_or([0; 8]),
            );
            *p += 8;
            v
        };
        let read_str = |p: &mut usize, len: usize| -> String {
            let s =
                String::from_utf8_lossy(&RAW_BUNDLE[*p..*p + len]).into_owned();
            *p += len;
            s
        };
        let read_f64_vec = |p: &mut usize, n: usize| -> Vec<f64> {
            (0..n).map(|_| read_f64(p)).collect()
        };

        let num_rxns = read_u32(&mut pos) as usize;
        let mut reactions = Vec::with_capacity(num_rxns);
        let mut reaction_by_name = HashMap::with_capacity(num_rxns);
        for idx in 0..num_rxns {
            let nlen = read_u16(&mut pos) as usize;
            let op_code = read_u8(&mut pos);
            let is_rev = read_u8(&mut pos) != 0;
            let thlen = read_u16(&mut pos) as usize;
            let name = read_str(&mut pos, nlen);
            let target_handle = read_str(&mut pos, thlen);
            let num_in = read_u8(&mut pos) as usize;
            let in_num =
                [read_u8(&mut pos), read_u8(&mut pos), read_u8(&mut pos)];
            let in_iso =
                [read_u16(&mut pos), read_u16(&mut pos), read_u16(&mut pos)];
            let num_out = read_u8(&mut pos) as usize;
            let out_num =
                [read_u8(&mut pos), read_u8(&mut pos), read_u8(&mut pos)];
            let out_iso =
                [read_u16(&mut pos), read_u16(&mut pos), read_u16(&mut pos)];
            let ye_exp = read_i8(&mut pos);
            let rho_exp = read_i8(&mut pos);
            let category = read_u8(&mut pos);
            let screening_isos =
                [read_u16(&mut pos), read_u16(&mut pos), read_u16(&mut pos)];
            let weak_isos = [read_u16(&mut pos), read_u16(&mut pos)];
            let std_q = read_f64(&mut pos);
            let std_qneu = read_f64(&mut pos);
            let weak_lowt_rate = read_f64(&mut pos);
            reaction_by_name.insert(name.clone(), idx + 1);
            reactions.push(ReactionInfo {
                id: idx + 1,
                name,
                op_code,
                is_reverse: is_rev,
                target_handle,
                num_in,
                in_num,
                in_iso,
                num_out,
                out_num,
                out_iso,
                ye_exp,
                rho_exp,
                category,
                screening_isos,
                weak_isos,
                std_q,
                std_qneu,
                weak_lowt_rate,
            });
        }

        let num_rt = read_u32(&mut pos) as usize;
        let mut rate_tables = HashMap::with_capacity(num_rt);
        for _ in 0..num_rt {
            let rxn_id = read_u16(&mut pos) as usize;
            let nt8 = read_u16(&mut pos) as usize;
            let t8s = read_f64_vec(&mut pos, nt8);
            let f1: Vec<CubicCoeffs> = (0..nt8)
                .map(|_| {
                    [
                        read_f64(&mut pos),
                        read_f64(&mut pos),
                        read_f64(&mut pos),
                        read_f64(&mut pos),
                    ]
                })
                .collect();
            rate_tables.insert(rxn_id, (t8s, f1));
        }

        let nuclides = NuclideCatalog::builtin_ref();
        let fac = flexmesa_math::pow_strict(
            1.0e9 * BOLTZM / (2.0 * PI * HBAR * HBAR * AVO),
            1.5,
        ) / AVO;
        let num_handles = read_u32(&mut pos) as usize;
        let mut reaclib_handles = HashMap::with_capacity(num_handles);
        for _ in 0..num_handles {
            let hlen = read_u16(&mut pos) as usize;
            let is_rev = read_u8(&mut pos) != 0;
            let nterms = read_u8(&mut pos) as usize;
            let h = read_str(&mut pos, hlen);
            let rev_override = if nterms > 0 {
                parse_reaclib_handle(nuclides, &h).map(|(ins, outs)| {
                    let (fwd_in, fwd_out) =
                        if is_rev { (outs, ins) } else { (ins, outs) };
                    compute_reaclib_rev_coeffs(&fwd_in, &fwd_out, fac)
                })
            } else {
                None
            };
            let mut terms = Vec::with_capacity(nterms);
            for _ in 0..nterms {
                let chapter = read_u8(&mut pos);
                let rflag = read_u8(&mut pos);
                let inv_exp = read_i8(&mut pos);
                let coeffs = std::array::from_fn(|_| read_f64(&mut pos));
                let mut inv_c1 = read_f64(&mut pos);
                let mut inv_c2 = read_f64(&mut pos);
                let mut inv_part = std::array::from_fn(|_| read_f64(&mut pos));
                if rflag != b'w' && rflag != b'e' && inv_part[0] != 0.0 {
                    let (c1, c2, ip) = rev_override.unwrap_or_else(|| {
                        panic!(
                            "failed to parse non-weak REACLIB handle '{h}' in rates_tables.bin"
                        )
                    });
                    inv_c1 = c1;
                    inv_c2 = c2;
                    inv_part = ip;
                }
                terms.push(ReaclibTerm {
                    chapter,
                    rflag,
                    inv_exp,
                    coeffs,
                    inv_c1,
                    inv_c2,
                    inv_part,
                });
            }
            reaclib_handles.insert(h, (is_rev, terms));
        }

        let num_wi = read_u32(&mut pos) as usize;
        let mut weak_info = HashMap::with_capacity(num_wi);
        for _ in 0..num_wi {
            let wl_id = read_u16(&mut pos);
            let wr_id = read_u16(&mut pos);
            let hl = read_f64(&mut pos);
            let qneu = read_f64(&mut pos);
            weak_info.insert((wl_id, wr_id), (hl, qneu));
        }

        let num_wt = read_u32(&mut pos) as usize;
        let mut weak_tables = Vec::with_capacity(num_wt);
        let mut weak_pair_map = HashMap::with_capacity(num_wt);
        for idx in 0..num_wt {
            let lhs_id = read_u16(&mut pos);
            let rhs_id = read_u16(&mut pos);
            let l_len = read_u8(&mut pos) as usize;
            let r_len = read_u8(&mut pos) as usize;
            let nt9 = read_u8(&mut pos) as usize;
            let nlye = read_u8(&mut pos) as usize;
            let q_tot = read_f64(&mut pos);
            let lhs_name = read_str(&mut pos, l_len);
            let rhs_name = read_str(&mut pos, r_len);
            let t9s = read_f64_vec(&mut pos, nt9);
            let lyerhos = read_f64_vec(&mut pos, nlye);
            let ldecay = read_f64_vec(&mut pos, nt9 * nlye);
            let lcapture = read_f64_vec(&mut pos, nt9 * nlye);
            let lneutrino = read_f64_vec(&mut pos, nt9 * nlye);
            weak_pair_map.insert((lhs_name.clone(), rhs_name.clone()), idx);
            weak_tables.push(WeakTable {
                id: idx + 1,
                lhs_id,
                rhs_id,
                lhs_name,
                rhs_name,
                q_tot,
                t9s,
                lyerhos,
                ldecay,
                lcapture,
                lneutrino,
            });
        }

        let num_suz = read_u32(&mut pos) as usize;
        let mut suzuki_pair_map = HashMap::with_capacity(num_suz);
        for _ in 0..num_suz {
            let lhs_id = read_u16(&mut pos);
            let rhs_id = read_u16(&mut pos);
            let nlogt = read_u8(&mut pos) as usize;
            let nlye = read_u8(&mut pos) as usize;
            let has_decay = read_u8(&mut pos) != 0;
            let has_capture = read_u8(&mut pos) != 0;
            let logts = read_f64_vec(&mut pos, nlogt);
            let lyerhos = read_f64_vec(&mut pos, nlye);
            let (decay_rate, decay_nu) = if has_decay {
                (
                    read_f64_vec(&mut pos, nlogt * nlye),
                    read_f64_vec(&mut pos, nlogt * nlye),
                )
            } else {
                (Vec::new(), Vec::new())
            };
            let (capture_rate, capture_nu) = if has_capture {
                (
                    read_f64_vec(&mut pos, nlogt * nlye),
                    read_f64_vec(&mut pos, nlogt * nlye),
                )
            } else {
                (Vec::new(), Vec::new())
            };
            let n_l = nuclides.get_by_id(usize::from(lhs_id)).unwrap();
            let n_r = nuclides.get_by_id(usize::from(rhs_id)).unwrap();
            suzuki_pair_map.insert(
                (n_l.name.clone(), n_r.name.clone()),
                SuzukiTable {
                    lhs_id,
                    rhs_id,
                    q_tot: n_l.mass_excess - n_r.mass_excess,
                    logts,
                    lyerhos,
                    has_decay,
                    has_capture,
                    decay_rate,
                    decay_nu,
                    capture_rate,
                    capture_nu,
                },
            );
        }

        Self {
            reactions,
            reaction_by_name,
            rate_tables,
            reaclib_handles,
            weak_info,
            weak_tables,
            weak_pair_map,
            suzuki_pair_map,
        }
    }
}
