// SPDX-License-Identifier: LGPL-3.0-or-later
//! Network initialization, reaction-kind ordering, forward/reverse adjacency
//! pairing, and cached 10,001-point `interp_m3q` raw rate tables
//! (`net/private/net_initialize.f90` and `rates/private/rates_support.f90`).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{LN10, MEV2GR};
use flexmesa_interp1d::{interp_m3q, CubicCoeffs};
use flexmesa_math::exp10;
use flexmesa_rates::{eval_reaction_raw, RatesCatalog, ReactionInfo, TFactors};
use rayon::prelude::*;

use crate::parser::ParsedNetDef;
use crate::screen::init_screening_tables;
use crate::types::{
    NetError, NetGeneralInfo, ReaclibKind, ReactionKind, ScreenPairParams,
    MAX_SAFE_LOGT_FOR_RATES, NRATTAB, RATTAB_THI, RATTAB_TLO, RATTAB_TSTP,
};

static LOGTTAB: OnceLock<Vec<f64>> = OnceLock::new();
static TTAB: OnceLock<Vec<f64>> = OnceLock::new();
static RATTAB_CACHE: OnceLock<Mutex<HashMap<String, Arc<Vec<CubicCoeffs>>>>> =
    OnceLock::new();

/// Returns the shared 10,001-point $\log_{10} T$ grid `logttab`.
#[must_use]
pub fn global_logttab() -> &'static [f64] {
    LOGTTAB.get_or_init(|| {
        (0..NRATTAB)
            .map(|i| (i as f64).mul_add(RATTAB_TSTP, RATTAB_TLO))
            .collect()
    })
}

/// Returns the shared 10,001-point $T = 10^{\log_{10} T}$ grid `ttab`.
#[must_use]
pub fn global_ttab() -> &'static [f64] {
    TTAB.get_or_init(|| global_logttab().iter().map(|&lt| exp10(lt)).collect())
}

const PREDEFINED_REVERSE_PAIRS: &[(&str, &str)] = &[
    ("r_al27_pa_mg24", "r_mg24_ap_al27"),
    ("r_ar36_ag_ca40", "r_ca40_ga_ar36"),
    ("r_ar36_ga_s32", "r_s32_ag_ar36"),
    ("r_b8_gp_be7", "r_be7_pg_b8"),
    ("r_c12_ag_o16", "r_o16_ga_c12"),
    ("r_c12_ap_n15", "r_n15_pa_c12"),
    ("r_c12_pg_n13", "r_n13_gp_c12"),
    ("r_c12_to_he4_he4_he4", "r_he4_he4_he4_to_c12"),
    ("r_c13_an_o16", "r_n14_gp_c13"),
    ("r_ca40_ag_ti44", "r_ti44_ga_ca40"),
    ("r_cr48_ag_fe52", "r_fe52_ga_cr48"),
    ("r_cr48_ga_ti44", "r_ti44_ag_cr48"),
    ("r_f17_ap_ne20", "r_o16_pg_f17"),
    ("r_f17_gp_o16", "r_o14_ap_f17"),
    ("r_f18_gp_o17", "r_o17_pg_f18"),
    ("r_f17_pg_ne18", "r_ne18_gp_f17"),
    ("r_f18_pg_ne19", "r_ne19_gp_f18"),
    ("r_f18_pa_o15", "r_o15_ap_f18"),
    ("r_f19_gp_o18", "r_o18_pg_f19"),
    ("r_f19_pa_o16", "r_o16_ap_f19"),
    ("r_f19_pg_ne20", "r_ne20_gp_f19"),
    ("r_fe52_ag_ni56", "r_ni56_ga_fe52"),
    ("r_mg24_ag_si28", "r_si28_ga_mg24"),
    ("r_mg24_ga_ne20", "r_ne20_ag_mg24"),
    ("r_n13_pg_o14", "r_o14_gp_n13"),
    ("r_n14_ap_o17", "r_o17_pa_n14"),
    ("r_n14_pg_o15", "r_o15_gp_n14"),
    ("r_n15_ap_o18", "r_o18_pa_n15"),
    ("r_n15_pg_o16", "r_o16_gp_n15"),
    ("r_na23_pa_ne20", "r_ne20_ap_na23"),
    ("r_ne19_ga_o15", "r_o15_ag_ne19"),
    ("r_ne20_ga_o16", "r_o16_ag_ne20"),
    ("r_s32_ga_si28", "r_si28_ag_s32"),
    ("rc12ap_to_o16", "iro16gp_to_c12"),
    ("rc12ap_to_o16", "ro16gp_to_c12"),
    ("ro16ap_to_ne20", "rne20gp_to_o16"),
    ("rne20ap_to_mg24", "rmg24gp_to_ne20"),
    ("rmg24ap_to_si28", "rsi28gp_to_mg24"),
    ("rsi28ap_to_s32", "rs32gp_to_si28"),
    ("rs32ap_to_ar36", "rar36gp_to_s32"),
    ("rar36ap_to_ca40", "rca40gp_to_ar36"),
    ("rca40ap_to_ti44", "rti44gp_to_ca40"),
    ("rti44ap_to_cr48", "rcr48gp_to_ti44"),
    ("rcr48ap_to_fe52", "rfe52gp_to_cr48"),
    ("rfe54ng_to_fe56", "rfe56gn_to_fe54"),
    ("rfe52neut_to_fe54", "rfe54g_to_fe52"),
    ("r_li7_pa_he4", "r_he4_ap_li7"),
    ("rhe4_breakup", "rhe4_rebuild"),
    ("rprot_to_neut", "rneut_to_prot"),
    ("rfe52aprot_to_fe54", "rfe54prot_to_fe52"),
    ("rfe52aprot_to_ni56", "rni56gprot_to_fe52"),
    ("rfe54prot_to_ni56", "rni56gprot_to_fe54"),
];

fn build_reverse_id_map(
    reactions: &[ReactionInfo],
    extra_pairs: &[(String, String)],
) -> HashMap<usize, usize> {
    let by_name: HashMap<&str, usize> =
        reactions.iter().map(|r| (r.name.as_str(), r.id)).collect();
    let mut rev_map = HashMap::new();
    for &(s1, s2) in PREDEFINED_REVERSE_PAIRS {
        if let (Some(&id1), Some(&id2)) = (by_name.get(s1), by_name.get(s2)) {
            rev_map.insert(id1, id2);
            rev_map.insert(id2, id1);
        }
    }
    for (s1, s2) in extra_pairs {
        if let (Some(&id1), Some(&id2)) =
            (by_name.get(s1.as_str()), by_name.get(s2.as_str()))
        {
            rev_map.insert(id1, id2);
            rev_map.insert(id2, id1);
        }
    }
    rev_map
}

fn classify_reaclib_kinds(
    reactions: &[ReactionInfo],
    rev_map: &HashMap<usize, usize>,
) -> (Vec<ReaclibKind>, Vec<usize>) {
    let nuclides = NuclideCatalog::builtin_ref();
    let ineut = nuclides.get_nuclide_index("neut").unwrap_or(1) as u16;
    let ih1 = nuclides.get_nuclide_index("h1").unwrap_or(2) as u16;
    let ihe4 = nuclides.get_nuclide_index("he4").unwrap_or(7) as u16;

    let n = reactions.len();
    let mut kinds = vec![ReaclibKind::Other; n];
    let mut rev_ids = vec![0_usize; n];

    for (i, rxn) in reactions.iter().enumerate() {
        let ir = rxn.id;
        if rxn.weak_isos[0] > 0 || !rxn.name.starts_with("r_") {
            continue;
        }
        let Some(&r_ir) = rev_map.get(&ir) else {
            continue;
        };
        rev_ids[i] = r_ir;
        if rxn.num_in > 2 || rxn.num_out > 2 {
            continue;
        }
        let num_in1 = rxn.in_num[0];
        let iso_in1 = rxn.in_iso[0];
        let num_in2 = rxn.in_num[1];
        let iso_in2 = rxn.in_iso[1];
        let num_out1 = rxn.out_num[0];
        let iso_out1 = rxn.out_iso[0];
        let num_out2 = rxn.out_num[1];
        let iso_out2 = rxn.out_iso[1];

        if iso_in1 == 0 || iso_out1 == 0 || num_in1 > 3 || num_out1 > 3 {
            continue;
        }
        if iso_in2 == 0 {
            if iso_out2 > 0 || (num_in1 == 1 && num_out1 == 1) {
                continue;
            }
            kinds[i] = if r_ir < ir {
                ReaclibKind::Other
            } else {
                ReaclibKind::GeneralOneOne
            };
            continue;
        }
        if iso_out2 == 0 {
            if num_in1 > 1 || num_in2 > 1 || num_out1 > 1 {
                kinds[i] = ReaclibKind::GeneralTwoOne;
                continue;
            }
        } else if num_in1 > 1 || num_in2 > 1 || num_out1 > 1 || num_out2 > 1 {
            kinds[i] = if r_ir > ir {
                ReaclibKind::Other
            } else {
                ReaclibKind::GeneralTwoTwo
            };
            continue;
        }

        let z_in1 = nuclides.get_by_id(usize::from(iso_in1)).map_or(0, |e| e.z);
        let z_in2 = nuclides.get_by_id(usize::from(iso_in2)).map_or(0, |e| e.z);
        let in_a = if z_in1 <= z_in2 { iso_in1 } else { iso_in2 };

        if iso_out2 == 0 {
            kinds[i] = if in_a == ineut {
                ReaclibKind::Ng
            } else if in_a == ihe4 {
                ReaclibKind::Ag
            } else if in_a == ih1 {
                ReaclibKind::Pg
            } else {
                ReaclibKind::GeneralTwoOne
            };
        } else {
            let z_o1 =
                nuclides.get_by_id(usize::from(iso_out1)).map_or(0, |e| e.z);
            let z_o2 =
                nuclides.get_by_id(usize::from(iso_out2)).map_or(0, |e| e.z);
            let out_c = if z_o1 <= z_o2 { iso_out1 } else { iso_out2 };
            if in_a == ihe4 {
                kinds[i] = if out_c == ih1 {
                    ReaclibKind::Ap
                } else if out_c == ineut {
                    ReaclibKind::An
                } else {
                    ReaclibKind::GeneralTwoTwo
                };
            } else if in_a == ih1 {
                if out_c == ineut {
                    kinds[i] = ReaclibKind::Pn;
                } else if out_c == ihe4 {
                    kinds[i] = ReaclibKind::Other;
                    continue;
                } else {
                    kinds[i] = ReaclibKind::GeneralTwoTwo;
                }
            } else if in_a == ineut {
                if out_c == ih1 || out_c == ihe4 {
                    kinds[i] = ReaclibKind::Other;
                    continue;
                }
                kinds[i] = ReaclibKind::GeneralTwoTwo;
            } else {
                kinds[i] = ReaclibKind::GeneralTwoTwo;
            }
            if kinds[i] == ReaclibKind::GeneralTwoTwo && r_ir > ir {
                kinds[i] = ReaclibKind::Other;
            }
        }
    }
    (kinds, rev_ids)
}

impl NetGeneralInfo {
    /// Compiles and initializes a complete [`NetGeneralInfo`] from an embedded
    /// `.net` file name (e.g., `"basic.net"`, `"approx21.net"`).
    ///
    /// # Errors
    /// Returns [`NetError`] if the file is unknown or fails to initialize.
    pub fn from_name(net_filename: &str) -> Result<Self, NetError> {
        let parsed = ParsedNetDef::parse_from_name(net_filename)?;
        Self::from_parsed(parsed)
    }

    /// Compiles and initializes a [`NetGeneralInfo`] from an embedded `.net`
    /// file name or a case-local file in `search_dir`.
    ///
    /// # Errors
    /// Returns [`NetError`] if the file is unknown or fails to initialize.
    pub fn from_name_or_dir(
        net_filename: &str,
        search_dir: Option<&std::path::Path>,
    ) -> Result<Self, NetError> {
        let parsed =
            ParsedNetDef::parse_from_name_or_dir(net_filename, search_dir)?;
        Self::from_parsed(parsed)
    }

    /// Compiles and initializes a [`NetGeneralInfo`] from a [`ParsedNetDef`].
    pub fn from_parsed(parsed: ParsedNetDef) -> Result<Self, NetError> {
        let nuclides = NuclideCatalog::builtin_ref();
        let max_chem_id = 7856_usize;
        let mut net_iso = vec![0_usize; max_chem_id + 1];
        let mut chem_id: Vec<usize> = parsed.active_isos.into_iter().collect();
        chem_id.sort_unstable();
        for (idx, &cid) in chem_id.iter().enumerate() {
            net_iso[cid] = idx + 1;
        }
        let num_isos = chem_id.len();

        let mut rxn_map: HashMap<usize, ReactionInfo> = parsed
            .active_reactions
            .into_values()
            .map(|r| (r.id, r))
            .collect();
        let mut reactions: Vec<ReactionInfo> =
            rxn_map.values().cloned().collect();
        reactions.sort_by(|a, b| a.name.cmp(&b.name));

        let num_reactions = reactions.len();
        let rev_map = build_reverse_id_map(&reactions, &parsed.reverse_pairs);
        let (mut reaction_reaclib_kind, mut reverse_id_for_kind_ne_other) =
            classify_reaclib_kinds(&reactions, &rev_map);
        let mut reaction_id: Vec<usize> =
            reactions.iter().map(|r| r.id).collect();
        let mut rtab: HashMap<usize, usize> = reaction_id
            .iter()
            .enumerate()
            .map(|(i, &ir)| (ir, i + 1))
            .collect();

        // Group by kind = 1..=9 (`other_kind = 10` goes last)
        let mut i = 0_usize;
        'kind_loop: for kind_code in 1..=9_u8 {
            while i < num_reactions
                && (reaction_reaclib_kind[i] as u8) == kind_code
            {
                i += 1;
                if i >= num_reactions {
                    break 'kind_loop;
                }
            }
            loop {
                let mut found_one = false;
                for j in (i + 1)..num_reactions {
                    if (reaction_reaclib_kind[j] as u8) != kind_code {
                        continue;
                    }
                    reaction_reaclib_kind.swap(i, j);
                    reverse_id_for_kind_ne_other.swap(i, j);
                    reaction_id.swap(i, j);
                    rtab.insert(reaction_id[i], i + 1);
                    rtab.insert(reaction_id[j], j + 1);
                    found_one = true;
                    break;
                }
                if !found_one {
                    break;
                }
                i += 1;
                if i >= num_reactions {
                    break 'kind_loop;
                }
            }
        }

        // Reorder so forward + reverse pairs are adjacent
        let mut idx = 0_usize;
        while idx + 1 < num_reactions {
            if reaction_reaclib_kind[idx] == ReaclibKind::Other {
                idx += 1;
                continue;
            }
            let r_ir = reverse_id_for_kind_ne_other[idx];
            let Some(&r_i_1based) = rtab.get(&r_ir) else {
                idx += 1;
                continue;
            };
            let r_idx = r_i_1based - 1;
            let ir = reaction_id[idx];
            idx += 1;
            for k in (idx..r_idx).rev() {
                reaction_reaclib_kind[k + 1] = reaction_reaclib_kind[k];
                reverse_id_for_kind_ne_other[k + 1] =
                    reverse_id_for_kind_ne_other[k];
                reaction_id[k + 1] = reaction_id[k];
                rtab.insert(reaction_id[k + 1], k + 2);
            }
            reaction_reaclib_kind[idx] = ReaclibKind::Other;
            reverse_id_for_kind_ne_other[idx] = ir;
            reaction_id[idx] = r_ir;
            rtab.insert(r_ir, idx + 1);
            idx += 1;
        }

        let ordered_reactions: Vec<ReactionInfo> = reaction_id
            .iter()
            .map(|id| rxn_map.remove(id).unwrap())
            .collect();

        let ineut = nuclides.get_nuclide_index("neut").unwrap_or(1) as u16;
        let iprot = nuclides.get_nuclide_index("prot").unwrap_or(3) as u16;
        let ih1 = nuclides.get_nuclide_index("h1").unwrap_or(2) as u16;
        let rates_cat = RatesCatalog::builtin();

        let mut reaction_kind = Vec::with_capacity(num_reactions);
        let mut weak_reaction_index = vec![0_usize; num_reactions];
        let mut weak_reaction_num = Vec::new();
        let mut weaklib_ids = Vec::new();
        let mut reaction_max_z = Vec::with_capacity(num_reactions);
        let mut reaction_max_z_plus_n = Vec::with_capacity(num_reactions);

        for (k, rxn) in ordered_reactions.iter().enumerate() {
            let mut max_z = 0_i32;
            let mut max_a = 0_i32;
            for &cid in rxn
                .in_iso
                .iter()
                .take(rxn.num_in)
                .chain(rxn.out_iso.iter().take(rxn.num_out))
            {
                if cid > 0 {
                    if let Some(e) = nuclides.get_by_id(usize::from(cid)) {
                        let ez = e.z as i32;
                        let ea = e.z_plus_n as i32;
                        if ez > max_z || (ez == max_z && ea > max_a) {
                            max_z = ez;
                            max_a = ea;
                        }
                    }
                }
            }
            reaction_max_z.push(max_z);
            reaction_max_z_plus_n.push(max_a);

            if rxn.weak_isos[0] > 0 {
                reaction_kind.push(ReactionKind::Weak);
                weak_reaction_num.push(k + 1);
                weak_reaction_index[k] = weak_reaction_num.len();
                let lhs = nuclides
                    .get_by_id(usize::from(rxn.weak_isos[0]))
                    .map_or("", |e| e.name.as_str());
                let rhs = nuclides
                    .get_by_id(usize::from(rxn.weak_isos[1]))
                    .map_or("", |e| e.name.as_str());
                let w_id = rates_cat.weak_by_pair(lhs, rhs).map_or(0, |w| w.id);
                weaklib_ids.push(w_id);
            } else {
                let mut has_neut = false;
                let mut has_prot = false;
                for &cid in rxn
                    .in_iso
                    .iter()
                    .take(rxn.num_in)
                    .chain(rxn.out_iso.iter().take(rxn.num_out))
                {
                    if cid == ineut {
                        has_neut = true;
                    }
                    if cid == iprot || cid == ih1 {
                        has_prot = true;
                    }
                }
                if has_neut && !has_prot {
                    reaction_kind.push(ReactionKind::Neut);
                } else if has_prot {
                    reaction_kind.push(ReactionKind::Prot);
                } else {
                    reaction_kind.push(ReactionKind::OtherStrong);
                }
            }
        }

        let mut z158 = Vec::with_capacity(num_isos);
        let mut z52 = Vec::with_capacity(num_isos);
        let mut mion = Vec::with_capacity(num_isos);
        for &cid in &chem_id {
            let e = nuclides.get_by_id(cid).unwrap();
            let zf = e.z as f64;
            z158.push(zf.powf(1.58));
            z52.push(zf.powf(2.50));
            mion.push(nuclides.get_mass_excess(cid).unwrap() * MEV2GR);
        }

        let net_reaction_by_name = ordered_reactions
            .iter()
            .enumerate()
            .map(|(i, r)| (r.name.clone(), i + 1))
            .collect();
        let net_reaction_by_id = ordered_reactions
            .iter()
            .enumerate()
            .map(|(i, r)| (r.id, i + 1))
            .collect();

        let rattab_splines = build_rate_tables(&ordered_reactions);

        let mut info = Self {
            net_filename: parsed.net_filename,
            num_isos,
            num_reactions,
            num_wk_reactions: weak_reaction_num.len(),
            chem_id,
            net_iso,
            reactions: ordered_reactions,
            net_reaction_by_name,
            net_reaction_by_id,
            reaction_reaclib_kind,
            reaction_kind,
            reverse_id_for_kind_ne_other,
            weak_reaction_index,
            weak_reaction_num,
            weaklib_ids,
            reaction_max_z,
            reaction_max_z_plus_n,
            z158,
            z52,
            mion,
            screen_params: vec![ScreenPairParams::default(); num_reactions],
            rattab_splines,
            log_tcut_lo: RATTAB_TLO,
            log_tcut_lim: RATTAB_TLO + 0.1,
            doing_approx21: parsed.doing_approx21,
            add_co56_to_approx21: parsed.add_co56_to_approx21,
            approx21_ye_iso: parsed.approx21_ye_iso,
            fe56ec_n_neut: parsed.fe56ec_n_neut,
            fe56ec_fake_factor: 1.0e-4,
            min_t_for_fe56ec_fake_factor: 3.0e9,
        };
        init_screening_tables(&mut info);
        Ok(info)
    }

    /// Evaluates raw unscreened reaction rates `(rate_raw, rate_raw_dt, rate_raw_drho)`
    /// across `self.reactions` using the precomputed `interp_m3q` rate tables
    /// (`do_get_raw_rates` in `rates/private/rates_support.f90`).
    pub fn eval_raw_rates_from_tables(
        &self,
        ye: f64,
        log_t_in: f64,
        btemp: f64,
        bden: f64,
        rate_factors: &[f64],
        rate_raw: &mut [f64],
        rate_raw_dt: &mut [f64],
        rate_raw_drho: &mut [f64],
    ) -> Result<(), NetError> {
        let logtemp = log_t_in.min(MAX_SAFE_LOGT_FOR_RATES);
        if logtemp > RATTAB_THI {
            return Err(NetError::IntegrationError(format!(
                "logT={logtemp} exceeds RATTAB_THI={RATTAB_THI}"
            )));
        }
        let logttab = global_logttab();
        let iat0 = ((logtemp - RATTAB_TLO) / RATTAB_TSTP) as isize + 1;
        let iat = iat0 - 1;
        let mut k = iat.clamp(1, (NRATTAB as isize) - 3) as usize;
        while k > 0 && logtemp < logttab[k] {
            k -= 1;
        }
        while k + 1 < NRATTAB && logtemp > logttab[k + 1] {
            k += 1;
        }
        let dt = logtemp - logttab[k];
        let inv_bt_ln10 = 1.0 / (btemp * LN10);

        for (i, rxn) in self.reactions.iter().enumerate() {
            let mut dtab = match rxn.ye_exp {
                1 => ye,
                2 => ye * ye,
                _ => 1.0,
            };
            let irho = rxn.rho_exp;
            match irho {
                1 => dtab *= bden,
                2 => dtab *= bden * bden,
                3 => dtab *= bden * bden * bden,
                4 => dtab *= bden * bden * bden * bden,
                _ => {}
            }
            let ddtab = f64::from(irho) * dtab / bden;
            let c = &self.rattab_splines[i][k];
            let fac = rate_factors.get(i).copied().unwrap_or(1.0);
            let poly =
                c[3].mul_add(dt, c[2]).mul_add(dt, c[1]).mul_add(dt, c[0]);
            let dpoly = (1.5 * c[3]).mul_add(dt, c[2]).mul_add(2.0 * dt, c[1]);
            let r = poly * dtab * fac;
            rate_raw[i] = r;
            rate_raw_drho[i] = poly * ddtab * fac;
            rate_raw_dt[i] = if logtemp >= MAX_SAFE_LOGT_FOR_RATES {
                0.0
            } else {
                dpoly * dtab * inv_bt_ln10 * fac
            };
        }
        Ok(())
    }
}

fn build_rate_tables(reactions: &[ReactionInfo]) -> Vec<Arc<Vec<CubicCoeffs>>> {
    let cache_mutex = RATTAB_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut uncached = Vec::new();
    {
        let guard = cache_mutex.lock().unwrap();
        for rxn in reactions {
            if !guard.contains_key(&rxn.name) {
                uncached.push(rxn.clone());
            }
        }
    }
    if !uncached.is_empty() {
        let logttab = global_logttab();
        let ttab = global_ttab();
        let cat = RatesCatalog::builtin();
        let built: Vec<(String, Arc<Vec<CubicCoeffs>>)> = uncached
            .par_iter()
            .map(|rxn| {
                let mut y = Vec::with_capacity(NRATTAB);
                for idx in 0..NRATTAB {
                    let tf = TFactors::from_log_t(logttab[idx], ttab[idx]);
                    y.push(eval_reaction_raw(cat, rxn, ttab[idx], &tf));
                }
                let coeffs = interp_m3q(logttab, &y).unwrap();
                (rxn.name.clone(), Arc::new(coeffs))
            })
            .collect();
        let mut guard = cache_mutex.lock().unwrap();
        for (name, arc_c) in built {
            guard.insert(name, arc_c);
        }
    }
    let guard = cache_mutex.lock().unwrap();
    reactions
        .iter()
        .map(|rxn| Arc::clone(guard.get(&rxn.name).unwrap()))
        .collect()
}
