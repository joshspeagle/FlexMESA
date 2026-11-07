// SPDX-License-Identifier: LGPL-3.0-or-later
//! Steady-state equilibrium links and beta-decay rate caps for `approx21`
//! (`approx21_special_reactions` in `net/private/net_approx21.f90` lines 352–923).

use std::sync::OnceLock;

use flexmesa_chem::NuclideCatalog;

use crate::approx21::Approx21Indices;
use crate::derivs_support::fl_epsnuc_3alf_with_derivs;
use crate::reaction_synth::synthesize_reaction_info;
use crate::types::NetGeneralInfo;

/// Cached standard reaction $Q$ and $Q_\nu$ values used by `approx21_eps_info`.
#[derive(Debug, Clone, Copy)]
pub struct Approx21ReactionQs {
    pub qtotal_rpp: f64,
    pub qneu_rpp: f64,
    pub qr33: f64,
    pub qtotal_rpp2: f64,
    pub qneu_rpp2: f64,
    pub qtotal_rpp3: f64,
    pub qneu_rpp3: f64,
    pub qtotal_rcpg: f64,
    pub qneu_rcpg: f64,
    pub qtotal_rnpg: f64,
    pub qneu_rnpg: f64,
    pub qtotal_ropg: f64,
    pub qneu_ropg: f64,
    pub qrn14_to_o16: f64,
    pub qtotal_rpen: f64,
    pub qneu_rpen: f64,
    pub qtotal_rnep: f64,
    pub qneu_rnep: f64,
    pub qtotal_rni56ec: f64,
    pub qneu_rni56ec: f64,
    pub qtotal_rco56ec: f64,
    pub qneu_rco56ec: f64,
    pub qrn14ag: f64,
    pub qr3alf: f64,
    pub qrc12ag: f64,
    pub qro16ag: f64,
    pub qr1212: f64,
    pub qr1216_to_mg24: f64,
    pub qr1216_to_si28: f64,
    pub qr1616a: f64,
    pub qr1616g: f64,
    pub qrne20ag: f64,
    pub qrmg24ag: f64,
    pub qrsi28ag: f64,
    pub qrs32ag: f64,
    pub qrar36ag: f64,
    pub qrca40ag: f64,
    pub qrti44ag: f64,
    pub qrcr48ag: f64,
    pub qrfe52ag: f64,
    pub qrfe52neut_to_fe54: f64,
    pub qrfe52aprot_to_fe54: f64,
    pub qrfe54ng_to_fe56: f64,
    pub qrfe54aprot_to_fe56: f64,
    pub qrfe52aprot_to_ni56: f64,
    pub qrfe54prot_to_ni56: f64,
    pub qrhe4_breakup: f64,
    pub qrhe4_rebuild: f64,
}

static APPROX21_QS: OnceLock<Approx21ReactionQs> = OnceLock::new();

fn q_pair(name: &str) -> (f64, f64) {
    synthesize_reaction_info(name, 0)
        .map_or((0.0, 0.0), |r| (r.std_q, r.std_qneu))
}

/// Returns the cached [`Approx21ReactionQs`] table.
#[must_use]
pub fn global_approx21_qs() -> &'static Approx21ReactionQs {
    APPROX21_QS.get_or_init(|| {
        let (qtotal_rpp, qneu_rpp) = q_pair("rpp_to_he3");
        let (qr33, _) = q_pair("r_he3_he3_to_h1_h1_he4");
        let (qtotal_rpp2, qneu_rpp2) = q_pair("r34_pp2");
        let (qtotal_rpp3, qneu_rpp3) = q_pair("r34_pp3");
        let (qtotal_rcpg, qneu_rcpg) = q_pair("rc12_to_n14");
        let (qtotal_rnpg, qneu_rnpg) = q_pair("rn14_to_c12");
        let (qtotal_ropg, qneu_ropg) = q_pair("ro16_to_n14");
        let (qrn14_to_o16, _) = q_pair("rn14_to_o16");
        let (qtotal_rpen, qneu_rpen) = q_pair("rprot_to_neut");
        let (qtotal_rnep, qneu_rnep) = q_pair("rneut_to_prot");
        let (qtotal_rni56ec, qneu_rni56ec) = q_pair("rni56ec_to_co56");
        let (qtotal_rco56ec, qneu_rco56ec) = q_pair("rco56ec_to_fe56");
        let (qrn14ag, _) = q_pair("rn14ag_lite");
        let (qr3alf, _) = q_pair("r_he4_he4_he4_to_c12");
        let (qrc12ag, _) = q_pair("r_c12_ag_o16");
        let (qro16ag, _) = q_pair("r_o16_ag_ne20");
        let (qr1212, _) = q_pair("r1212");
        let (qr1216_to_mg24, _) = q_pair("r1216_to_mg24");
        let (qr1216_to_si28, _) = q_pair("r1216_to_si28");
        let (qr1616a, _) = q_pair("r1616a");
        let (qr1616g, _) = q_pair("r1616g");
        let (qrne20ag, _) = q_pair("r_ne20_ag_mg24");
        let (qrmg24ag, _) = q_pair("r_mg24_ag_si28");
        let (qrsi28ag, _) = q_pair("r_si28_ag_s32");
        let (qrs32ag, _) = q_pair("r_s32_ag_ar36");
        let (qrar36ag, _) = q_pair("r_ar36_ag_ca40");
        let (qrca40ag, _) = q_pair("r_ca40_ag_ti44");
        let (qrti44ag, _) = q_pair("r_ti44_ag_cr48");
        let (qrcr48ag, _) = q_pair("r_cr48_ag_fe52");
        let (qrfe52ag, _) = q_pair("r_fe52_ag_ni56");
        let (qrfe52neut_to_fe54, _) = q_pair("rfe52neut_to_fe54");
        let (qrfe52aprot_to_fe54, _) = q_pair("rfe52aprot_to_fe54");
        let (qrfe54ng_to_fe56, _) = q_pair("rfe54ng_to_fe56");
        let (qrfe54aprot_to_fe56, _) = q_pair("rfe54aprot_to_fe56");
        let (qrfe52aprot_to_ni56, _) = q_pair("rfe52aprot_to_ni56");
        let (qrfe54prot_to_ni56, _) = q_pair("rfe54prot_to_ni56");
        let (qrhe4_breakup, _) = q_pair("rhe4_breakup");
        let (qrhe4_rebuild, _) = q_pair("rhe4_rebuild");
        Approx21ReactionQs {
            qtotal_rpp,
            qneu_rpp,
            qr33,
            qtotal_rpp2,
            qneu_rpp2,
            qtotal_rpp3,
            qneu_rpp3,
            qtotal_rcpg,
            qneu_rcpg,
            qtotal_rnpg,
            qneu_rnpg,
            qtotal_ropg,
            qneu_ropg,
            qrn14_to_o16,
            qtotal_rpen,
            qneu_rpen,
            qtotal_rnep,
            qneu_rnep,
            qtotal_rni56ec,
            qneu_rni56ec,
            qtotal_rco56ec,
            qneu_rco56ec,
            qrn14ag,
            qr3alf,
            qrc12ag,
            qro16ag,
            qr1212,
            qr1216_to_mg24,
            qr1216_to_si28,
            qr1616a,
            qr1616g,
            qrne20ag,
            qrmg24ag,
            qrsi28ag,
            qrs32ag,
            qrar36ag,
            qrca40ag,
            qrti44ag,
            qrcr48ag,
            qrfe52ag,
            qrfe52neut_to_fe54,
            qrfe52aprot_to_fe54,
            qrfe54ng_to_fe56,
            qrfe54aprot_to_fe56,
            qrfe52aprot_to_ni56,
            qrfe54prot_to_ni56,
            qrhe4_breakup,
            qrhe4_rebuild,
        }
    })
}

pub(crate) fn get_qs_rfe56ec(g: &NetGeneralInfo) -> (f64, f64) {
    let nuclides = NuclideCatalog::builtin_ref();
    let iso_name = nuclides
        .get_by_id(g.approx21_ye_iso)
        .map_or("fe56", |e| e.name.as_str());
    let rxn_name = match iso_name {
        "mn56" => "rfe56ec_fake_to_mn56",
        "mn57" => "rfe56ec_fake_to_mn57",
        "cr56" => "rfe56ec_fake_to_cr56",
        "cr57" => "rfe56ec_fake_to_cr57",
        "cr58" => "rfe56ec_fake_to_cr58",
        "cr59" => "rfe56ec_fake_to_cr59",
        "cr60" => "rfe56ec_fake_to_cr60",
        "cr61" => "rfe56ec_fake_to_cr61",
        "cr62" => "rfe56ec_fake_to_cr62",
        "cr63" => "rfe56ec_fake_to_cr63",
        "cr64" => "rfe56ec_fake_to_cr64",
        "cr65" => "rfe56ec_fake_to_cr65",
        "cr66" => "rfe56ec_fake_to_cr66",
        _ => "rco56ec_to_fe56",
    };
    if let Some(idx1) = g.reaction_index_by_name(rxn_name) {
        let r = &g.reactions[idx1 - 1];
        (r.std_q, r.std_qneu)
    } else {
        q_pair(rxn_name)
    }
}

/// Evaluates `approx21_special_reactions` (`net/private/net_approx21.f90` lines 352–923).
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn approx21_special_reactions(
    idx: &Approx21Indices,
    btemp: f64,
    bden: f64,
    abar: f64,
    zbar: f64,
    y: &[f64],
    use_3a_fl: bool,
    conv_eps_3a: f64,
    ratdum: &mut [f64],
    dratdumdt: &mut [f64],
    dratdumdd: &mut [f64],
    dratdumdy1: &mut [f64],
    dratdumdy2: &mut [f64],
) {
    const TINY_DENOM: f64 = 1.0e-50;
    const TINY_Y: f64 = 1.0e-30;

    if use_3a_fl {
        if y[idx.ihe4] < TINY_Y {
            ratdum[idx.ir3a] = 0.0;
            dratdumdt[idx.ir3a] = 0.0;
            dratdumdd[idx.ir3a] = 0.0;
        } else {
            let (fl_eps, dfl_dt, dfl_drho) = fl_epsnuc_3alf_with_derivs(
                btemp,
                bden,
                4.0 * y[idx.ihe4],
                abar / zbar,
            );
            let xx =
                conv_eps_3a * y[idx.ihe4] * y[idx.ihe4] * y[idx.ihe4] / 6.0;
            ratdum[idx.ir3a] = fl_eps / xx;
            dratdumdt[idx.ir3a] = dfl_dt / xx;
            dratdumdd[idx.ir3a] = dfl_drho / xx;
        }
    }

    let set_eq_link_2 = |ir: usize,
                         r1: usize,
                         r2: usize,
                         zz: f64,
                         d_dy: f64,
                         denomdt: f64,
                         denomdd: f64,
                         ratdum: &mut [f64],
                         dratdumdt: &mut [f64],
                         dratdumdd: &mut [f64],
                         dratdumdy1: &mut [f64]| {
        ratdum[ir] = ratdum[r1] * ratdum[r2] * zz;
        dratdumdy1[ir] = ratdum[ir] * zz * d_dy;
        dratdumdt[ir] = dratdumdt[r1] * ratdum[r2] * zz
            + ratdum[r1] * dratdumdt[r2] * zz
            - ratdum[ir] * zz * denomdt;
        dratdumdd[ir] = dratdumdd[r1] * ratdum[r2] * zz
            + ratdum[r1] * dratdumdd[r2] * zz
            - ratdum[ir] * zz * denomdd;
    };

    for &ir in &[
        idx.ir1f54,
        idx.ir2f54,
        idx.irfe56_aux1,
        idx.irfe56_aux2,
        idx.irfe56_aux3,
        idx.irfe56_aux4,
        idx.ir3f54,
        idx.ir4f54,
        idx.ir5f54,
        idx.ir6f54,
        idx.ir7f54,
        idx.ir8f54,
    ] {
        ratdum[ir] = 0.0;
        dratdumdy1[ir] = 0.0;
        dratdumdt[ir] = 0.0;
        dratdumdd[ir] = 0.0;
    }

    // fe52(n,g)fe53(n,g)fe54 equilibrium links
    let denom = ratdum[idx.ir53gn] + y[idx.ineut] * ratdum[idx.ir53ng];
    if denom > TINY_DENOM && btemp > 1.5e9 {
        let denomdt =
            dratdumdt[idx.ir53gn] + y[idx.ineut] * dratdumdt[idx.ir53ng];
        let denomdd =
            dratdumdd[idx.ir53gn] + y[idx.ineut] * dratdumdd[idx.ir53ng];
        let zz = 1.0 / denom;
        let d_dy = -ratdum[idx.ir53ng];
        set_eq_link_2(
            idx.ir1f54, idx.ir54gn, idx.ir53gn, zz, d_dy, denomdt, denomdd,
            ratdum, dratdumdt, dratdumdd, dratdumdy1,
        );
        set_eq_link_2(
            idx.ir2f54, idx.ir52ng, idx.ir53ng, zz, d_dy, denomdt, denomdd,
            ratdum, dratdumdt, dratdumdd, dratdumdy1,
        );
    }

    // fe54(n,g)fe55(n,g)fe56 equilibrium links
    let denom = ratdum[idx.ir55gn] + y[idx.ineut] * ratdum[idx.ir55ng];
    if denom > TINY_DENOM && btemp > 1.5e9 {
        let denomdt =
            dratdumdt[idx.ir55gn] + y[idx.ineut] * dratdumdt[idx.ir55ng];
        let denomdd =
            dratdumdd[idx.ir55gn] + y[idx.ineut] * dratdumdd[idx.ir55ng];
        let zz = 1.0 / denom;
        let d_dy = -ratdum[idx.ir55ng];
        set_eq_link_2(
            idx.irfe56_aux1,
            idx.ir56gn,
            idx.ir55gn,
            zz,
            d_dy,
            denomdt,
            denomdd,
            ratdum,
            dratdumdt,
            dratdumdd,
            dratdumdy1,
        );
        set_eq_link_2(
            idx.irfe56_aux2,
            idx.ir54ng,
            idx.ir55ng,
            zz,
            d_dy,
            denomdt,
            denomdd,
            ratdum,
            dratdumdt,
            dratdumdd,
            dratdumdy1,
        );
    }

    // fe54(a,p)co57(g,p)fe56 equilibrium links
    let denom = ratdum[idx.irco57gp] + y[idx.iprot] * ratdum[idx.irco57pa];
    if denom > TINY_DENOM && btemp > 1.5e9 {
        let denomdt =
            dratdumdt[idx.irco57gp] + y[idx.iprot] * dratdumdt[idx.irco57pa];
        let denomdd =
            dratdumdd[idx.irco57gp] + y[idx.iprot] * dratdumdd[idx.irco57pa];
        let zz = 1.0 / denom;
        let d_dy = -ratdum[idx.irco57pa];
        set_eq_link_2(
            idx.irfe56_aux3,
            idx.irfe56pg,
            idx.irco57pa,
            zz,
            d_dy,
            denomdt,
            denomdd,
            ratdum,
            dratdumdt,
            dratdumdd,
            dratdumdy1,
        );
        set_eq_link_2(
            idx.irfe56_aux4,
            idx.irfe54ap,
            idx.irco57gp,
            zz,
            d_dy,
            denomdt,
            denomdd,
            ratdum,
            dratdumdt,
            dratdumdd,
            dratdumdy1,
        );
    }

    // co55 equilibrium links r3f54..r8f54
    let denom = ratdum[idx.ircogp]
        + y[idx.iprot] * (ratdum[idx.ircopg] + ratdum[idx.ircopa]);
    if denom > TINY_DENOM && btemp > 1.5e9 {
        let denomdt = dratdumdt[idx.ircogp]
            + y[idx.iprot] * (dratdumdt[idx.ircopg] + dratdumdt[idx.ircopa]);
        let denomdd = dratdumdd[idx.ircogp]
            + y[idx.iprot] * (dratdumdd[idx.ircopg] + dratdumdd[idx.ircopa]);
        let zz = 1.0 / denom;
        let d_dp = -(ratdum[idx.ircopg] + ratdum[idx.ircopa]);

        for (ir, r1, r2) in [
            (idx.ir3f54, idx.irfepg, idx.ircopg),
            (idx.ir4f54, idx.irnigp, idx.ircogp),
            (idx.ir5f54, idx.irfepg, idx.ircopa),
            (idx.ir6f54, idx.irfeap, idx.ircogp),
            (idx.ir7f54, idx.irfeap, idx.ircopg),
            (idx.ir8f54, idx.irnigp, idx.ircopa),
        ] {
            set_eq_link_2(
                ir, r1, r2, zz, d_dp, denomdt, denomdd, ratdum, dratdumdt,
                dratdumdd, dratdumdy1,
            );
        }
    }

    // Photodisintegrated n and p back to he4 equilibrium links
    for &ir in &[idx.iralf1, idx.iralf2] {
        ratdum[ir] = 0.0;
        dratdumdy1[ir] = 0.0;
        dratdumdy2[ir] = 0.0;
        dratdumdt[ir] = 0.0;
        dratdumdd[ir] = 0.0;
    }

    let denom = ratdum[idx.irhegp] * ratdum[idx.irdgn]
        + y[idx.ineut] * ratdum[idx.irheng] * ratdum[idx.irdgn]
        + y[idx.ineut] * y[idx.iprot] * ratdum[idx.irheng] * ratdum[idx.irdpg];
    if denom > TINY_DENOM && btemp > 1.5e9 {
        let denomdt = dratdumdt[idx.irhegp] * ratdum[idx.irdgn]
            + ratdum[idx.irhegp] * dratdumdt[idx.irdgn]
            + y[idx.ineut]
                * (dratdumdt[idx.irheng] * ratdum[idx.irdgn]
                    + ratdum[idx.irheng] * dratdumdt[idx.irdgn])
            + y[idx.ineut]
                * y[idx.iprot]
                * (dratdumdt[idx.irheng] * ratdum[idx.irdpg]
                    + ratdum[idx.irheng] * dratdumdt[idx.irdpg]);
        let denomdd = dratdumdd[idx.irhegp] * ratdum[idx.irdgn]
            + ratdum[idx.irhegp] * dratdumdd[idx.irdgn]
            + y[idx.ineut]
                * (dratdumdd[idx.irheng] * ratdum[idx.irdgn]
                    + ratdum[idx.irheng] * dratdumdd[idx.irdgn])
            + y[idx.ineut]
                * y[idx.iprot]
                * (dratdumdd[idx.irheng] * ratdum[idx.irdpg]
                    + ratdum[idx.irheng] * dratdumdd[idx.irdpg]);
        let zz = 1.0 / denom;

        ratdum[idx.iralf1] =
            ratdum[idx.irhegn] * ratdum[idx.irhegp] * ratdum[idx.irdgn] * zz;
        dratdumdy1[idx.iralf1] = -ratdum[idx.iralf1]
            * zz
            * (ratdum[idx.irheng] * ratdum[idx.irdgn]
                + y[idx.iprot] * ratdum[idx.irheng] * ratdum[idx.irdpg]);
        dratdumdy2[idx.iralf1] = -ratdum[idx.iralf1]
            * zz
            * y[idx.ineut]
            * ratdum[idx.irheng]
            * ratdum[idx.irdpg];
        dratdumdt[idx.iralf1] =
            dratdumdt[idx.irhegn] * ratdum[idx.irhegp] * ratdum[idx.irdgn] * zz
                + ratdum[idx.irhegn]
                    * dratdumdt[idx.irhegp]
                    * ratdum[idx.irdgn]
                    * zz
                + ratdum[idx.irhegn]
                    * ratdum[idx.irhegp]
                    * dratdumdt[idx.irdgn]
                    * zz
                - ratdum[idx.iralf1] * zz * denomdt;
        dratdumdd[idx.iralf1] =
            dratdumdd[idx.irhegn] * ratdum[idx.irhegp] * ratdum[idx.irdgn] * zz
                + ratdum[idx.irhegn]
                    * dratdumdd[idx.irhegp]
                    * ratdum[idx.irdgn]
                    * zz
                + ratdum[idx.irhegn]
                    * ratdum[idx.irhegp]
                    * dratdumdd[idx.irdgn]
                    * zz
                - ratdum[idx.iralf1] * zz * denomdd;

        ratdum[idx.iralf2] =
            ratdum[idx.irheng] * ratdum[idx.irdpg] * ratdum[idx.irhng] * zz;
        dratdumdy1[idx.iralf2] = -ratdum[idx.iralf2]
            * zz
            * (ratdum[idx.irheng] * ratdum[idx.irdgn]
                + y[idx.iprot] * ratdum[idx.irheng] * ratdum[idx.irdpg]);
        dratdumdy2[idx.iralf2] = -ratdum[idx.iralf2]
            * zz
            * y[idx.ineut]
            * ratdum[idx.irheng]
            * ratdum[idx.irdpg];
        dratdumdt[idx.iralf2] =
            dratdumdt[idx.irheng] * ratdum[idx.irdpg] * ratdum[idx.irhng] * zz
                + ratdum[idx.irheng]
                    * dratdumdt[idx.irdpg]
                    * ratdum[idx.irhng]
                    * zz
                + ratdum[idx.irheng]
                    * ratdum[idx.irdpg]
                    * dratdumdt[idx.irhng]
                    * zz
                - ratdum[idx.iralf2] * zz * denomdt;
        dratdumdd[idx.iralf2] =
            dratdumdd[idx.irheng] * ratdum[idx.irdpg] * ratdum[idx.irhng] * zz
                + ratdum[idx.irheng]
                    * dratdumdd[idx.irdpg]
                    * ratdum[idx.irhng]
                    * zz
                + ratdum[idx.irheng]
                    * ratdum[idx.irdpg]
                    * dratdumdd[idx.irhng]
                    * zz
                - ratdum[idx.iralf2] * zz * denomdd;
    }

    // Beta limits
    let mut apply_beta_cap = |ir: usize, y_iso: f64, xx: f64| {
        ratdum[ir] = ratdum[ir].min(xx);
        if ratdum[ir] == xx {
            dratdumdy1[ir] = -xx / y_iso;
            dratdumdt[ir] = 0.0;
            dratdumdd[ir] = 0.0;
        } else {
            dratdumdy1[ir] = 0.0;
        }
    };
    if y[idx.ihe4] > TINY_Y {
        apply_beta_cap(idx.irhe3ag, y[idx.ihe4], 0.896 / y[idx.ihe4]);
    }
    if y[idx.ih1] > TINY_Y {
        apply_beta_cap(idx.irnpg, y[idx.ih1], 5.68e-3 / (y[idx.ih1] * 1.57));
        apply_beta_cap(idx.iropg, y[idx.ih1], 0.0105 / y[idx.ih1]);
    }
}
