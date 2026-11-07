// SPDX-License-Identifier: LGPL-3.0-or-later
//! Lazy loader and catalog for the embedded binary nuclear reaction rate table
//! bundle (`crates/flexmesa-rates/data/rates_tables.bin`).

use std::collections::HashMap;
use std::sync::OnceLock;

use flexmesa_chem::NuclideCatalog;
use flexmesa_interp1d::CubicCoeffs;

use crate::raw_rates::eval_interp_rate_table;
use crate::types::{RatesError, ReaclibTerm, ReactionInfo};

const RAW_BUNDLE: &[u8] = include_bytes!("../data/rates_tables.bin");

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
    special_states: HashMap<String, Vec<(f64, f64)>>,
    special_trans: HashMap<(String, String), Vec<(usize, usize, f64)>>,
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
    #[must_use]
    pub fn eval_rate_table(&self, ir: usize, temp: f64) -> Option<(f64, f64)> {
        self.rate_tables
            .get(&ir)
            .map(|(t8s, f1)| eval_interp_rate_table(t8s, f1, temp))
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
    /// special `ecapture` transitions (`test_special.states` / `test_special.transitions`).
    ///
    /// # Errors
    /// Returns [`RatesError::UnknownWeakPair`] if `(lhs, rhs)` is not in `special_trans`.
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
        let key = (lhs.trim().to_string(), rhs.trim().to_string());
        let Some(trans) = self.special_trans.get(&key) else {
            return Err(RatesError::UnknownWeakPair(key.0, key.1));
        };
        let Some(s_lhs) = self.special_states.get(&key.0) else {
            return Err(RatesError::UnknownWeakPair(key.0, key.1));
        };
        let Some(s_rhs) = self.special_states.get(&key.1) else {
            return Err(RatesError::UnknownWeakPair(key.0, key.1));
        };
        let wtab = self.weak_by_pair(&key.0, &key.1)?;
        let nuclides = NuclideCatalog::builtin_ref();
        let n_lhs =
            nuclides
                .get_by_id(usize::from(wtab.lhs_id))
                .ok_or_else(|| {
                    RatesError::InvalidTableData(format!(
                        "unknown lhs_id {}",
                        wtab.lhs_id
                    ))
                })?;
        let n_rhs =
            nuclides
                .get_by_id(usize::from(wtab.rhs_id))
                .ok_or_else(|| {
                    RatesError::InvalidTableData(format!(
                        "unknown rhs_id {}",
                        wtab.rhs_id
                    ))
                })?;
        #[allow(clippy::cast_precision_loss)]
        Ok((
            s_lhs,
            s_rhs,
            trans,
            n_lhs.z as f64,
            n_rhs.z as f64,
            wtab.q_tot,
        ))
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

        let num_handles = read_u32(&mut pos) as usize;
        let mut reaclib_handles = HashMap::with_capacity(num_handles);
        for _ in 0..num_handles {
            let hlen = read_u16(&mut pos) as usize;
            let is_rev = read_u8(&mut pos) != 0;
            let nterms = read_u8(&mut pos) as usize;
            let h = read_str(&mut pos, hlen);
            let mut terms = Vec::with_capacity(nterms);
            for _ in 0..nterms {
                let chapter = read_u8(&mut pos);
                let rflag = read_u8(&mut pos);
                let inv_exp = read_i8(&mut pos);
                let coeffs = std::array::from_fn(|_| read_f64(&mut pos));
                let inv_c1 = read_f64(&mut pos);
                let inv_c2 = read_f64(&mut pos);
                let inv_part = std::array::from_fn(|_| read_f64(&mut pos));
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

        let nuclides = NuclideCatalog::builtin_ref();
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

        let mut special_states = HashMap::new();
        special_states
            .insert("mg24".to_string(), vec![(0.000, 0.0), (1.369, 2.0)]);
        special_states.insert(
            "na24".to_string(),
            vec![
                (0.000, 4.0),
                (0.472, 1.0),
                (0.563, 2.0),
                (1.341, 2.0),
                (1.345, 3.0),
                (1.347, 1.0),
            ],
        );
        let mut special_trans = HashMap::new();
        special_trans.insert(
            ("mg24".to_string(), "na24".to_string()),
            vec![(1, 2, 4.8), (1, 6, 3.772)],
        );
        special_trans.insert(
            ("na24".to_string(), "mg24".to_string()),
            vec![(2, 1, 5.3), (6, 1, 4.249)],
        );

        Self {
            reactions,
            reaction_by_name,
            rate_tables,
            reaclib_handles,
            weak_info,
            weak_tables,
            weak_pair_map,
            suzuki_pair_map,
            special_states,
            special_trans,
        }
    }
}
