// SPDX-License-Identifier: LGPL-3.0-or-later
//! Hardwired `approx21` and `approx21_plus_co56` reaction indices, `(p,a)/(p,g)`
//! branching fractions, and weak rates (`net/private/net_approx21.f90`
//! lines 163–348 and 2788–3159).

use flexmesa_chem::NuclideCatalog;
use flexmesa_rates::eval_ecapnuc_rate;

use crate::derivs_support::eval_single_weak_blended;
use crate::types::NetGeneralInfo;

/// Cached 0-based species and reaction indices for an `approx21` network.
#[derive(Debug, Clone, Copy)]
pub struct Approx21Indices {
    pub plus_co56: bool,
    pub ih1: usize,
    pub ihe3: usize,
    pub ihe4: usize,
    pub ic12: usize,
    pub in14: usize,
    pub io16: usize,
    pub ine20: usize,
    pub img24: usize,
    pub isi28: usize,
    pub is32: usize,
    pub iar36: usize,
    pub ica40: usize,
    pub iti44: usize,
    pub icr48: usize,
    pub ife52: usize,
    pub ife54: usize,
    pub ife56: usize,
    pub ico56: usize,
    pub ini56: usize,
    pub ineut: usize,
    pub iprot: usize,
    pub icrx: usize,
    pub ir3a: usize,
    pub irg3a: usize,
    pub ircag: usize,
    pub ir1212: usize,
    pub ir1216: usize,
    pub ir1616: usize,
    pub iroga: usize,
    pub iroag: usize,
    pub irnega: usize,
    pub irneag: usize,
    pub irmgga: usize,
    pub irmgag: usize,
    pub irsiga: usize,
    pub irmgap: usize,
    pub iralpa: usize,
    pub iralpg: usize,
    pub irsigp: usize,
    pub irsiag: usize,
    pub irsga: usize,
    pub irsiap: usize,
    pub irppa: usize,
    pub irppg: usize,
    pub irsgp: usize,
    pub irsag: usize,
    pub irarga: usize,
    pub irsap: usize,
    pub irclpa: usize,
    pub irclpg: usize,
    pub irargp: usize,
    pub irarag: usize,
    pub ircaga: usize,
    pub irarap: usize,
    pub irkpa: usize,
    pub irkpg: usize,
    pub ircagp: usize,
    pub ircaag: usize,
    pub irtiga: usize,
    pub ircaap: usize,
    pub irscpa: usize,
    pub irscpg: usize,
    pub irtigp: usize,
    pub irtiag: usize,
    pub ircrga: usize,
    pub irtiap: usize,
    pub irvpa: usize,
    pub irvpg: usize,
    pub ircrgp: usize,
    pub ircrag: usize,
    pub irfega: usize,
    pub ircrap: usize,
    pub irmnpa: usize,
    pub irmnpg: usize,
    pub irfegp: usize,
    pub irfeag: usize,
    pub irniga: usize,
    pub irfeap: usize,
    pub ircopa: usize,
    pub ircopg: usize,
    pub irnigp: usize,
    pub ir52ng: usize,
    pub ir53gn: usize,
    pub ir53ng: usize,
    pub ir54gn: usize,
    pub irfepg: usize,
    pub ircogp: usize,
    pub irheng: usize,
    pub irhegn: usize,
    pub irhng: usize,
    pub irdgn: usize,
    pub irdpg: usize,
    pub irhegp: usize,
    pub irpen: usize,
    pub irnep: usize,
    pub irn56ec: usize,
    pub irco56ec: usize,
    pub irpp: usize,
    pub ir33: usize,
    pub irhe3ag: usize,
    pub ir_be7_wk_li7: usize,
    pub ir_be7_pg_b8: usize,
    pub ircpg: usize,
    pub irnpg: usize,
    pub iropg: usize,
    pub irn15pg: usize,
    pub irn15pa: usize,
    pub irnag: usize,
    pub ir54ng: usize,
    pub ir55gn: usize,
    pub ir55ng: usize,
    pub ir56gn: usize,
    pub irfe54ap: usize,
    pub irco57pa: usize,
    pub irfe56pg: usize,
    pub irco57gp: usize,
    pub ifa: usize,
    pub ifg: usize,
    pub irr1: usize,
    pub irs1: usize,
    pub irt1: usize,
    pub iru1: usize,
    pub irv1: usize,
    pub irw1: usize,
    pub irx1: usize,
    pub ir1f54: usize,
    pub ir2f54: usize,
    pub ir3f54: usize,
    pub ir4f54: usize,
    pub ir5f54: usize,
    pub ir6f54: usize,
    pub ir7f54: usize,
    pub ir8f54: usize,
    pub iralf1: usize,
    pub iralf2: usize,
    pub irfe56_aux1: usize,
    pub irfe56_aux2: usize,
    pub irfe56_aux3: usize,
    pub irfe56_aux4: usize,
}

impl Approx21Indices {
    /// Resolves all species and reaction indices from `g`.
    #[must_use]
    pub fn from_net(g: &NetGeneralInfo) -> Self {
        let cat = NuclideCatalog::builtin_ref();
        let iso = |name: &str| -> usize {
            let cid = cat.get_nuclide_index(name).unwrap_or(0);
            g.net_iso.get(cid).copied().unwrap_or(1).saturating_sub(1)
        };
        let rxn = |name: &str| -> usize {
            g.reaction_index_by_name(name)
                .unwrap_or(1)
                .saturating_sub(1)
        };
        let plus_co56 = g.add_co56_to_approx21;
        let (irn56ec, irco56ec) = if plus_co56 {
            (rxn("rni56ec_to_co56"), rxn("rco56ec_to_fe56"))
        } else {
            (rxn("rni56ec_to_fe56"), 0)
        };
        let m = g.num_reactions;
        Self {
            plus_co56,
            ih1: iso("h1"),
            ihe3: iso("he3"),
            ihe4: iso("he4"),
            ic12: iso("c12"),
            in14: iso("n14"),
            io16: iso("o16"),
            ine20: iso("ne20"),
            img24: iso("mg24"),
            isi28: iso("si28"),
            is32: iso("s32"),
            iar36: iso("ar36"),
            ica40: iso("ca40"),
            iti44: iso("ti44"),
            icr48: iso("cr48"),
            ife52: iso("fe52"),
            ife54: iso("fe54"),
            ife56: iso("fe56"),
            ico56: if plus_co56 { iso("co56") } else { 0 },
            ini56: iso("ni56"),
            ineut: iso("neut"),
            iprot: iso("prot"),
            icrx: g.net_iso[g.approx21_ye_iso].saturating_sub(1),
            ir3a: rxn("r_he4_he4_he4_to_c12"),
            irg3a: rxn("r_c12_to_he4_he4_he4"),
            ircag: rxn("r_c12_ag_o16"),
            ir1212: rxn("r1212"),
            ir1216: rxn("r1216"),
            ir1616: rxn("r1616"),
            iroga: rxn("r_o16_ga_c12"),
            iroag: rxn("r_o16_ag_ne20"),
            irnega: rxn("r_ne20_ga_o16"),
            irneag: rxn("r_ne20_ag_mg24"),
            irmgga: rxn("r_mg24_ga_ne20"),
            irmgag: rxn("r_mg24_ag_si28"),
            irsiga: rxn("r_si28_ga_mg24"),
            irmgap: rxn("r_mg24_ap_al27"),
            iralpa: rxn("r_al27_pa_mg24"),
            iralpg: rxn("r_al27_pg_si28"),
            irsigp: rxn("r_si28_gp_al27"),
            irsiag: rxn("r_si28_ag_s32"),
            irsga: rxn("r_s32_ga_si28"),
            irsiap: rxn("r_si28_ap_p31"),
            irppa: rxn("r_p31_pa_si28"),
            irppg: rxn("r_p31_pg_s32"),
            irsgp: rxn("r_s32_gp_p31"),
            irsag: rxn("r_s32_ag_ar36"),
            irarga: rxn("r_ar36_ga_s32"),
            irsap: rxn("r_s32_ap_cl35"),
            irclpa: rxn("r_cl35_pa_s32"),
            irclpg: rxn("r_cl35_pg_ar36"),
            irargp: rxn("r_ar36_gp_cl35"),
            irarag: rxn("r_ar36_ag_ca40"),
            ircaga: rxn("r_ca40_ga_ar36"),
            irarap: rxn("r_ar36_ap_k39"),
            irkpa: rxn("r_k39_pa_ar36"),
            irkpg: rxn("r_k39_pg_ca40"),
            ircagp: rxn("r_ca40_gp_k39"),
            ircaag: rxn("r_ca40_ag_ti44"),
            irtiga: rxn("r_ti44_ga_ca40"),
            ircaap: rxn("r_ca40_ap_sc43"),
            irscpa: rxn("r_sc43_pa_ca40"),
            irscpg: rxn("r_sc43_pg_ti44"),
            irtigp: rxn("r_ti44_gp_sc43"),
            irtiag: rxn("r_ti44_ag_cr48"),
            ircrga: rxn("r_cr48_ga_ti44"),
            irtiap: rxn("r_ti44_ap_v47"),
            irvpa: rxn("r_v47_pa_ti44"),
            irvpg: rxn("r_v47_pg_cr48"),
            ircrgp: rxn("r_cr48_gp_v47"),
            ircrag: rxn("r_cr48_ag_fe52"),
            irfega: rxn("r_fe52_ga_cr48"),
            ircrap: rxn("r_cr48_ap_mn51"),
            irmnpa: rxn("r_mn51_pa_cr48"),
            irmnpg: rxn("r_mn51_pg_fe52"),
            irfegp: rxn("r_fe52_gp_mn51"),
            irfeag: rxn("r_fe52_ag_ni56"),
            irniga: rxn("r_ni56_ga_fe52"),
            irfeap: rxn("r_fe52_ap_co55"),
            ircopa: rxn("r_co55_pa_fe52"),
            ircopg: rxn("r_co55_pg_ni56"),
            irnigp: rxn("r_ni56_gp_co55"),
            ir52ng: rxn("r_fe52_ng_fe53"),
            ir53gn: rxn("r_fe53_gn_fe52"),
            ir53ng: rxn("r_fe53_ng_fe54"),
            ir54gn: rxn("r_fe54_gn_fe53"),
            irfepg: rxn("r_fe54_pg_co55"),
            ircogp: rxn("r_co55_gp_fe54"),
            irheng: rxn("r_he3_ng_he4"),
            irhegn: rxn("r_he4_gn_he3"),
            irhng: rxn("r_h1_ng_h2"),
            irdgn: rxn("r_h2_gn_h1"),
            irdpg: rxn("r_h2_pg_he3"),
            irhegp: rxn("r_he3_gp_h2"),
            irpen: rxn("rprot_to_neut"),
            irnep: rxn("rneut_to_prot"),
            irn56ec,
            irco56ec,
            irpp: rxn("rpp_to_he3"),
            ir33: rxn("r_he3_he3_to_h1_h1_he4"),
            irhe3ag: rxn("r_he3_ag_be7"),
            ir_be7_wk_li7: rxn("r_be7_wk_li7"),
            ir_be7_pg_b8: rxn("r_be7_pg_b8"),
            ircpg: rxn("r_c12_pg_n13"),
            irnpg: rxn("r_n14_pg_o15"),
            iropg: rxn("r_o16_pg_f17"),
            irn15pg: rxn("r_n15_pg_o16"),
            irn15pa: rxn("r_n15_pa_c12"),
            irnag: rxn("r_n14_ag_f18"),
            ir54ng: rxn("r_fe54_ng_fe55"),
            ir55gn: rxn("r_fe55_gn_fe54"),
            ir55ng: rxn("r_fe55_ng_fe56"),
            ir56gn: rxn("r_fe56_gn_fe55"),
            irfe54ap: rxn("r_fe54_ap_co57"),
            irco57pa: rxn("r_co57_pa_fe54"),
            irfe56pg: rxn("r_fe56_pg_co57"),
            irco57gp: rxn("r_co57_gp_fe56"),
            ifa: m,
            ifg: m + 1,
            irr1: m + 2,
            irs1: m + 3,
            irt1: m + 4,
            iru1: m + 5,
            irv1: m + 6,
            irw1: m + 7,
            irx1: m + 8,
            ir1f54: m + 9,
            ir2f54: m + 10,
            ir3f54: m + 11,
            ir4f54: m + 12,
            ir5f54: m + 13,
            ir6f54: m + 14,
            ir7f54: m + 15,
            ir8f54: m + 16,
            iralf1: m + 17,
            iralf2: m + 18,
            irfe56_aux1: m + 19,
            irfe56_aux2: m + 20,
            irfe56_aux3: m + 21,
            irfe56_aux4: m + 22,
        }
    }
}

/// Evaluates `(p,a)/(p,g)` branching fractions in `ratraw`, `dratrawdt`, and
/// `dratrawdd` (`approx21_pa_pg_fractions` in `net/private/net_approx21.f90`).
pub fn approx21_pa_pg_fractions(
    idx: &Approx21Indices,
    ratraw: &mut [f64],
    dratrawdt: &mut [f64],
    dratrawdd: &mut [f64],
) {
    {
        let mut set1 = |ifa: usize, irpg: usize, irpa: usize| {
            let ifg = ifa + 1;
            let ff1 = ratraw[irpg] + ratraw[irpa];
            let dff1dt = dratrawdt[irpg] + dratrawdt[irpa];
            let dff1dd = dratrawdd[irpg] + dratrawdd[irpa];
            if ff1 > 1.0e-30 {
                let ff2 = 1.0 / ff1;
                ratraw[ifa] = ratraw[irpa] * ff2;
                ratraw[ifg] = 1.0 - ratraw[ifa];
                dratrawdt[ifa] =
                    dratrawdt[irpa] * ff2 - ratraw[ifa] * ff2 * dff1dt;
                dratrawdd[ifa] =
                    dratrawdd[irpa] * ff2 - ratraw[ifa] * ff2 * dff1dd;
                dratrawdt[ifg] = -dratrawdt[ifa];
                dratrawdd[ifg] = -dratrawdd[ifa];
            } else {
                ratraw[ifa] = 0.0;
                ratraw[ifg] = 1.0;
                dratrawdt[ifa] = 0.0;
                dratrawdd[ifa] = 0.0;
                dratrawdt[ifg] = 0.0;
                dratrawdd[ifg] = 0.0;
            }
        };
        set1(idx.ifa, idx.irn15pg, idx.irn15pa);
    }
    let mut set2 = |irr1: usize, irpg: usize, irpa: usize| {
        let ff1 = ratraw[irpg] + ratraw[irpa];
        let dff1dt = dratrawdt[irpg] + dratrawdt[irpa];
        let dff1dd = dratrawdd[irpg] + dratrawdd[irpa];
        if ff1 > 1.0e-30 {
            let ff2 = 1.0 / ff1;
            ratraw[irr1] = ratraw[irpa] * ff2;
            dratrawdt[irr1] =
                dratrawdt[irpa] * ff2 - ratraw[irr1] * ff2 * dff1dt;
            dratrawdd[irr1] =
                dratrawdd[irpa] * ff2 - ratraw[irr1] * ff2 * dff1dd;
        } else {
            ratraw[irr1] = 0.0;
            dratrawdt[irr1] = 0.0;
            dratrawdd[irr1] = 0.0;
        }
    };
    set2(idx.irr1, idx.iralpg, idx.iralpa);
    set2(idx.irs1, idx.irppg, idx.irppa);
    set2(idx.irt1, idx.irclpg, idx.irclpa);
    set2(idx.iru1, idx.irkpg, idx.irkpa);
    set2(idx.irv1, idx.irscpg, idx.irscpa);
    set2(idx.irw1, idx.irvpg, idx.irvpa);
    set2(idx.irx1, idx.irmnpg, idx.irmnpa);
}

/// Evaluates `approx21_weak_rates` (`net/private/net_approx21.f90` lines 284–348).
#[allow(clippy::too_many_arguments)]
pub fn approx21_weak_rates(
    idx: &Approx21Indices,
    ratraw: &mut [f64],
    dratrawdt: &mut [f64],
    dratrawdd: &mut [f64],
    temp: f64,
    den: f64,
    ye: f64,
    eta: f64,
    weak_rate_factor: f64,
) {
    let (rpen, rnep, _, _) = eval_ecapnuc_rate(eta, temp, den);
    ratraw[idx.irpen] = rpen;
    dratrawdt[idx.irpen] = 0.0;
    dratrawdd[idx.irpen] = 0.0;
    ratraw[idx.irnep] = rnep;
    dratrawdt[idx.irnep] = 0.0;
    dratrawdd[idx.irnep] = 0.0;

    let t9 = temp * 1.0e-9;
    let ye_rho = ye * den;
    let cat = NuclideCatalog::builtin_ref();
    let cid_ni56 = cat.get_nuclide_index("ni56").unwrap_or(0) as u16;
    let cid_co56 = cat.get_nuclide_index("co56").unwrap_or(0) as u16;
    let cid_fe56 = cat.get_nuclide_index("fe56").unwrap_or(0) as u16;

    let ni_rxn = if idx.plus_co56 {
        "rni56ec_to_co56"
    } else {
        "rni56ec_to_fe56"
    };
    let ni_ev = eval_single_weak_blended(
        ni_rxn,
        [cid_ni56, cid_co56],
        true,
        t9,
        ye_rho,
        temp,
        den,
        1.0,
        1.0,
        1.0,
        eta,
        0.0,
        0.0,
        false,
        false,
    );
    ratraw[idx.irn56ec] = ni_ev.lambda * weak_rate_factor;
    dratrawdt[idx.irn56ec] = 0.0;
    dratrawdd[idx.irn56ec] = 0.0;

    if idx.plus_co56 {
        let co_ev = eval_single_weak_blended(
            "rco56ec_to_fe56",
            [cid_co56, cid_fe56],
            true,
            t9,
            ye_rho,
            temp,
            den,
            1.0,
            1.0,
            1.0,
            eta,
            0.0,
            0.0,
            false,
            false,
        );
        ratraw[idx.irco56ec] = co_ev.lambda * weak_rate_factor;
        dratrawdt[idx.irco56ec] = 0.0;
        dratrawdd[idx.irco56ec] = 0.0;
    }
}
