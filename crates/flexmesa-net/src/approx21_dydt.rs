// SPDX-License-Identifier: GPL-3.0-or-later
//! Hardwired `approx21` and `approx21_plus_co56` molar abundance derivatives
//! (`approx21_dydt` with `Quad` (`real(qp)`) accumulation in
//! `net/private/net_approx21.f90` lines 926–1544).

use std::fmt::Write as _;

use crate::approx21::Approx21Indices;
use crate::approx21_dfdy_heavy::approx21_dydt_iron_group;
use crate::types::{
    NetError, APPROX21_CO56_NUM_REACTIONS, APPROX21_NUM_REACTIONS,
};
use flexmesa_math::{safe_log10, Quad};

/// Accumulates `terms` sequentially left-to-right onto `q`
/// (`q = q + a1 + a2 + ...`), preventing Rust operator-precedence grouping
/// (`q += a1 + a2`).
#[inline]
pub(crate) fn add_terms(q: &mut Quad, terms: &[Quad]) {
    for &t in terms {
        *q += t;
    }
}

#[inline]
fn eval_fe56ec_fake_factor(
    fe56ec_fake_factor_in: f64,
    min_t_for_fe56ec_fake_factor: f64,
    temp: f64,
) -> f64 {
    if temp >= min_t_for_fe56ec_fake_factor {
        fe56ec_fake_factor_in
    } else {
        0.0
    }
}

#[cold]
fn reaction_name_at(idx: &Approx21Indices, slot: usize) -> &'static str {
    if idx.plus_co56 && slot == idx.irco56ec {
        return "r_co56_wk_fe56";
    }
    if slot == idx.irn56ec {
        return if idx.plus_co56 {
            "r_ni56_wk_co56"
        } else {
            "rni56ec_to_fe56"
        };
    }
    let pairs = [
        (idx.ir3a, "r_he4_he4_he4_to_c12"),
        (idx.irg3a, "r_c12_to_he4_he4_he4"),
        (idx.ircag, "r_c12_ag_o16"),
        (idx.ir1212, "r1212"),
        (idx.ir1216, "r1216"),
        (idx.ir1616, "r1616"),
        (idx.iroga, "r_o16_ga_c12"),
        (idx.iroag, "r_o16_ag_ne20"),
        (idx.irnega, "r_ne20_ga_o16"),
        (idx.irneag, "r_ne20_ag_mg24"),
        (idx.irmgga, "r_mg24_ga_ne20"),
        (idx.irmgag, "r_mg24_ag_si28"),
        (idx.irsiga, "r_si28_ga_mg24"),
        (idx.irmgap, "r_mg24_ap_al27"),
        (idx.iralpa, "r_al27_pa_mg24"),
        (idx.iralpg, "r_al27_pg_si28"),
        (idx.irsigp, "r_si28_gp_al27"),
        (idx.irsiag, "r_si28_ag_s32"),
        (idx.irsga, "r_s32_ga_si28"),
        (idx.irsiap, "r_si28_ap_p31"),
        (idx.irppa, "r_p31_pa_si28"),
        (idx.irppg, "r_p31_pg_s32"),
        (idx.irsgp, "r_s32_gp_p31"),
        (idx.irsag, "r_s32_ag_ar36"),
        (idx.irarga, "r_ar36_ga_s32"),
        (idx.irsap, "r_s32_ap_cl35"),
        (idx.irclpa, "r_cl35_pa_s32"),
        (idx.irclpg, "r_cl35_pg_ar36"),
        (idx.irargp, "r_ar36_gp_cl35"),
        (idx.irarag, "r_ar36_ag_ca40"),
        (idx.ircaga, "r_ca40_ga_ar36"),
        (idx.irarap, "r_ar36_ap_k39"),
        (idx.irkpa, "r_k39_pa_ar36"),
        (idx.irkpg, "r_k39_pg_ca40"),
        (idx.ircagp, "r_ca40_gp_k39"),
        (idx.ircaag, "r_ca40_ag_ti44"),
        (idx.irtiga, "r_ti44_ga_ca40"),
        (idx.ircaap, "r_ca40_ap_sc43"),
        (idx.irscpa, "r_sc43_pa_ca40"),
        (idx.irscpg, "r_sc43_pg_ti44"),
        (idx.irtigp, "r_ti44_gp_sc43"),
        (idx.irtiag, "r_ti44_ag_cr48"),
        (idx.ircrga, "r_cr48_ga_ti44"),
        (idx.irtiap, "r_ti44_ap_v47"),
        (idx.irvpa, "r_v47_pa_ti44"),
        (idx.irvpg, "r_v47_pg_cr48"),
        (idx.ircrgp, "r_cr48_gp_v47"),
        (idx.ircrag, "r_cr48_ag_fe52"),
        (idx.irfega, "r_fe52_ga_cr48"),
        (idx.ircrap, "r_cr48_ap_mn51"),
        (idx.irmnpa, "r_mn51_pa_cr48"),
        (idx.irmnpg, "r_mn51_pg_fe52"),
        (idx.irfegp, "r_fe52_gp_mn51"),
        (idx.irfeag, "r_fe52_ag_ni56"),
        (idx.irniga, "r_ni56_ga_fe52"),
        (idx.irfeap, "r_fe52_ap_co55"),
        (idx.ircopa, "r_co55_pa_fe52"),
        (idx.ircopg, "r_co55_pg_ni56"),
        (idx.irnigp, "r_ni56_gp_co55"),
        (idx.ir52ng, "r_fe52_ng_fe53"),
        (idx.ir53gn, "r_fe53_gn_fe52"),
        (idx.ir53ng, "r_fe53_ng_fe54"),
        (idx.ir54gn, "r_fe54_gn_fe53"),
        (idx.irfepg, "r_fe54_pg_co55"),
        (idx.ircogp, "r_co55_gp_fe54"),
        (idx.irheng, "r_he3_ng_he4"),
        (idx.irhegn, "r_he4_gn_he3"),
        (idx.irhng, "r_h1_ng_h2"),
        (idx.irdgn, "r_h2_gn_h1"),
        (idx.irdpg, "r_h2_pg_he3"),
        (idx.irhegp, "r_he3_gp_h2"),
        (idx.irpen, "rprot_to_neut"),
        (idx.irnep, "rneut_to_prot"),
        (idx.irpp, "rpp_to_he3"),
        (idx.ir33, "r_he3_he3_to_h1_h1_he4"),
        (idx.irhe3ag, "r_he3_ag_be7"),
        (idx.ir_be7_wk_li7, "r_be7_wk_li7"),
        (idx.ir_be7_pg_b8, "r_be7_pg_b8"),
        (idx.ircpg, "r_c12_pg_n13"),
        (idx.irnpg, "r_n14_pg_o15"),
        (idx.iropg, "r_o16_pg_f17"),
        (idx.irn15pg, "r_n15_pg_o16"),
        (idx.irn15pa, "r_n15_pa_c12"),
        (idx.irnag, "r_n14_ag_f18"),
        (idx.ir54ng, "r_fe54_ng_fe55"),
        (idx.ir55gn, "r_fe55_gn_fe54"),
        (idx.ir55ng, "r_fe55_ng_fe56"),
        (idx.ir56gn, "r_fe56_gn_fe55"),
        (idx.irfe54ap, "r_fe54_ap_co57"),
        (idx.irco57pa, "r_co57_pa_fe54"),
        (idx.irfe56pg, "r_fe56_pg_co57"),
        (idx.irco57gp, "r_co57_gp_fe56"),
        (idx.ifa, "fa"),
        (idx.ifg, "fg"),
        (idx.irr1, "r1"),
        (idx.irs1, "s1"),
        (idx.irt1, "t1"),
        (idx.iru1, "u1"),
        (idx.irv1, "v1"),
        (idx.irw1, "w1"),
        (idx.irx1, "x1"),
        (idx.ir1f54, "r1f54"),
        (idx.ir2f54, "r2f54"),
        (idx.ir3f54, "r3f54"),
        (idx.ir4f54, "r4f54"),
        (idx.ir5f54, "r5f54"),
        (idx.ir6f54, "r6f54"),
        (idx.ir7f54, "r7f54"),
        (idx.ir8f54, "r8f54"),
        (idx.iralf1, "ralf1"),
        (idx.iralf2, "ralf2"),
        (idx.irfe56_aux1, "rfe56aux1"),
        (idx.irfe56_aux2, "rfe56aux2"),
        (idx.irfe56_aux3, "rfe56aux3"),
        (idx.irfe56_aux4, "rfe56aux4"),
    ];
    for &(rxn_idx, name) in &pairs {
        if rxn_idx == slot {
            return name;
        }
    }
    "unknown"
}

#[cold]
#[allow(clippy::too_many_arguments)]
fn format_bad_dydt_error(
    idx: &Approx21Indices,
    species: usize,
    num_reactions: usize,
    y: &[f64],
    ratd: &[f64],
    dydt: &[f64],
    temp: f64,
    den: f64,
) -> String {
    let mut msg = String::from("approx21_dydt: non-finite dydt");
    for i in 0..species {
        if !dydt[i].is_finite() {
            let _ = write!(msg, "\ndydt(i) {} {} {}", i + 1, dydt[i], y[i]);
        }
    }
    let _ = write!(msg, "\nlog10(temp) = {}", safe_log10(temp));
    let _ = write!(msg, "\nlog10(rho) = {}", safe_log10(den));
    let nrat = num_reactions.min(ratd.len());
    for i in 0..nrat {
        let name = reaction_name_at(idx, i);
        let _ = write!(msg, "\n{name} {} {}", i + 1, ratd[i]);
    }
    msg
}

/// Evaluates `approx21_dydt` (`net/private/net_approx21.f90` lines 926–1544).
///
/// Here `rat0` is the un-differentiated screened rate array (`ratdum` in MESA),
/// and `ratd` is either `rat0` (`deriva == false`) or `dratdumdt` / `dratdumdd`
/// (`deriva == true`, `rate` in MESA). Returns [`NetError::IntegrationError`]
/// matching upstream's `is_bad`/`mesa_error` check
/// (`net_approx21.f90:1525-1542`) if any entry of `dydt[..species]` is
/// non-finite.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn approx21_dydt(
    idx: &Approx21Indices,
    y: &[f64],
    rat0: &[f64],
    ratd: &[f64],
    deriva: bool,
    fe56ec_fake_factor_in: f64,
    min_t_for_fe56ec_fake_factor: f64,
    fe56ec_n_neut: i32,
    temp: f64,
    den: f64,
    dydt: &mut [f64],
) -> Result<(), NetError> {
    let fe56ec_fake_factor = if !deriva {
        eval_fe56ec_fake_factor(
            fe56ec_fake_factor_in,
            min_t_for_fe56ec_fake_factor,
            temp,
        )
    } else {
        0.0
    };
    let nf = f64::from(fe56ec_n_neut);
    let species = if idx.plus_co56 { 22 } else { 21 };
    let num_reactions = if idx.plus_co56 {
        APPROX21_CO56_NUM_REACTIONS
    } else {
        APPROX21_NUM_REACTIONS
    };
    dydt[..species].fill(0.0);

    // Helper closures using Quad to match Fortran real(qp)
    // (individual products are evaluated in dp before promoting to qp on sum)
    let m2 = |a: f64, b: f64| Quad::from_f64(a * b);
    let m3 = |a: f64, b: f64, c: f64| Quad::from_f64(a * b * c);
    let m4 = |a: f64, b: f64, c: f64, d: f64| Quad::from_f64(a * b * c * d);
    let m5 = |a: f64, b: f64, c: f64, d: f64, e: f64| {
        Quad::from_f64(a * b * c * d * e)
    };
    let m6 = |a: f64, b: f64, c: f64, d: f64, e: f64, f: f64| {
        Quad::from_f64(a * b * c * d * e * f)
    };

    // h1 (lines 960–969)
    let mut q_h1 = Quad::ZERO;
    add_terms(
        &mut q_h1,
        &[
            m4(-1.5, y[idx.ih1], y[idx.ih1], ratd[idx.irpp]),
            m3(y[idx.ihe3], y[idx.ihe3], ratd[idx.ir33]),
            m3(-y[idx.ihe3], y[idx.ihe4], ratd[idx.irhe3ag]),
            m4(-2.0, y[idx.ic12], y[idx.ih1], ratd[idx.ircpg]),
            m4(-2.0, y[idx.in14], y[idx.ih1], ratd[idx.irnpg]),
            m4(-2.0, y[idx.io16], y[idx.ih1], ratd[idx.iropg]),
            m3(-3.0, y[idx.ih1], ratd[idx.irpen]),
        ],
    );
    dydt[idx.ih1] = q_h1.to_f64();

    // he3 (lines 973–978)
    let mut q_he3 = Quad::ZERO;
    add_terms(
        &mut q_he3,
        &[
            m4(0.5, y[idx.ih1], y[idx.ih1], ratd[idx.irpp]),
            m3(-y[idx.ihe3], y[idx.ihe3], ratd[idx.ir33]),
            m3(-y[idx.ihe3], y[idx.ihe4], ratd[idx.irhe3ag]),
            m2(y[idx.ih1], ratd[idx.irpen]),
        ],
    );
    dydt[idx.ihe3] = q_he3.to_f64();

    // he4 (lines 983–1122)
    let mut q_he4 = Quad::ZERO;
    add_terms(
        &mut q_he4,
        &[
            m4(0.5, y[idx.ic12], y[idx.ic12], ratd[idx.ir1212]),
            m4(0.5, y[idx.ic12], y[idx.io16], ratd[idx.ir1216]),
            m5(0.56, 0.5, y[idx.io16], y[idx.io16], ratd[idx.ir1616]),
            m4(-y[idx.ihe4], y[idx.in14], ratd[idx.irnag], 1.5),
        ],
    );

    add_terms(
        &mut q_he4,
        &[
            m5(-0.5, y[idx.ihe4], y[idx.ihe4], y[idx.ihe4], ratd[idx.ir3a]),
            m3(3.0, y[idx.ic12], ratd[idx.irg3a]),
            m3(-y[idx.ihe4], y[idx.ic12], ratd[idx.ircag]),
            m2(y[idx.io16], ratd[idx.iroga]),
            m3(-y[idx.ihe4], y[idx.io16], ratd[idx.iroag]),
            m2(y[idx.ine20], ratd[idx.irnega]),
            m3(-y[idx.ihe4], y[idx.ine20], ratd[idx.irneag]),
            m2(y[idx.img24], ratd[idx.irmgga]),
            m3(-y[idx.ihe4], y[idx.img24], ratd[idx.irmgag]),
            m2(y[idx.isi28], ratd[idx.irsiga]),
            m3(-y[idx.ihe4], y[idx.isi28], ratd[idx.irsiag]),
            m2(y[idx.is32], ratd[idx.irsga]),
        ],
    );

    add_terms(
        &mut q_he4,
        &[
            m3(-y[idx.ihe4], y[idx.is32], ratd[idx.irsag]),
            m2(y[idx.iar36], ratd[idx.irarga]),
            m3(-y[idx.ihe4], y[idx.iar36], ratd[idx.irarag]),
            m2(y[idx.ica40], ratd[idx.ircaga]),
            m3(-y[idx.ihe4], y[idx.ica40], ratd[idx.ircaag]),
            m2(y[idx.iti44], ratd[idx.irtiga]),
            m3(-y[idx.ihe4], y[idx.iti44], ratd[idx.irtiag]),
            m2(y[idx.icr48], ratd[idx.ircrga]),
            m3(-y[idx.ihe4], y[idx.icr48], ratd[idx.ircrag]),
            m2(y[idx.ife52], ratd[idx.irfega]),
            m3(-y[idx.ihe4], y[idx.ife52], ratd[idx.irfeag]),
            m2(y[idx.ini56], ratd[idx.irniga]),
        ],
    );

    if !deriva {
        q_he4 += m6(
            0.34,
            0.5,
            y[idx.io16],
            y[idx.io16],
            ratd[idx.irs1],
            ratd[idx.ir1616],
        );
        for (y0, yp1, rap, r1, rgp) in [
            (y[idx.img24], y[idx.isi28], idx.irmgap, idx.irr1, idx.irsigp),
            (y[idx.isi28], y[idx.is32], idx.irsiap, idx.irs1, idx.irsgp),
            (y[idx.is32], y[idx.iar36], idx.irsap, idx.irt1, idx.irargp),
            (y[idx.iar36], y[idx.ica40], idx.irarap, idx.iru1, idx.ircagp),
            (y[idx.ica40], y[idx.iti44], idx.ircaap, idx.irv1, idx.irtigp),
            (y[idx.iti44], y[idx.icr48], idx.irtiap, idx.irw1, idx.ircrgp),
            (y[idx.icr48], y[idx.ife52], idx.ircrap, idx.irx1, idx.irfegp),
        ] {
            add_terms(
                &mut q_he4,
                &[
                    m4(-y[idx.ihe4], y0, ratd[rap], 1.0 - ratd[r1]),
                    m3(yp1, ratd[rgp], ratd[r1]),
                ],
            );
        }
    } else {
        add_terms(
            &mut q_he4,
            &[
                m6(
                    0.34,
                    0.5,
                    y[idx.io16],
                    y[idx.io16],
                    rat0[idx.irs1],
                    ratd[idx.ir1616],
                ),
                m6(
                    0.34,
                    0.5,
                    y[idx.io16],
                    y[idx.io16],
                    ratd[idx.irs1],
                    rat0[idx.ir1616],
                ),
            ],
        );
        for (y0, yp1, rap, r1, rgp) in [
            (y[idx.img24], y[idx.isi28], idx.irmgap, idx.irr1, idx.irsigp),
            (y[idx.isi28], y[idx.is32], idx.irsiap, idx.irs1, idx.irsgp),
            (y[idx.is32], y[idx.iar36], idx.irsap, idx.irt1, idx.irargp),
            (y[idx.iar36], y[idx.ica40], idx.irarap, idx.iru1, idx.ircagp),
            (y[idx.ica40], y[idx.iti44], idx.ircaap, idx.irv1, idx.irtigp),
            (y[idx.iti44], y[idx.icr48], idx.irtiap, idx.irw1, idx.ircrgp),
            (y[idx.icr48], y[idx.ife52], idx.ircrap, idx.irx1, idx.irfegp),
        ] {
            add_terms(
                &mut q_he4,
                &[
                    m4(-y[idx.ihe4], y0, ratd[rap], 1.0 - rat0[r1]),
                    m4(y[idx.ihe4], y0, rat0[rap], ratd[r1]),
                    m3(yp1, rat0[rgp], ratd[r1]),
                    m3(yp1, ratd[rgp], rat0[r1]),
                ],
            );
        }
    }

    add_terms(
        &mut q_he4,
        &[
            m4(y[idx.ife54], y[idx.iprot], y[idx.iprot], ratd[idx.ir5f54]),
            m3(-y[idx.ife52], y[idx.ihe4], ratd[idx.ir6f54]),
            m4(-y[idx.ife52], y[idx.ihe4], y[idx.iprot], ratd[idx.ir7f54]),
            m3(y[idx.ini56], y[idx.iprot], ratd[idx.ir8f54]),
            m2(-y[idx.ihe4], ratd[idx.iralf1]),
            m5(
                y[idx.ineut],
                y[idx.ineut],
                y[idx.iprot],
                y[idx.iprot],
                ratd[idx.iralf2],
            ),
            m4(
                y[idx.ife56],
                y[idx.iprot],
                y[idx.iprot],
                ratd[idx.irfe56_aux3],
            ),
            m3(-y[idx.ife54], y[idx.ihe4], ratd[idx.irfe56_aux4]),
        ],
    );

    // Match MESA net_approx21.f90: a2 = y(ihe3)*y(ihe4)*rate(irhe3ag) remains
    // live into the CNO block, so the He4 equation adds it a second time
    // (`net/private/net_approx21.f90:1105, 1110-1113`; deferred upstream bug
    // #526).
    let a2_he3_he4 = m3(y[idx.ihe3], y[idx.ihe4], ratd[idx.irhe3ag]);
    add_terms(
        &mut q_he4,
        &[
            m4(0.5, y[idx.ihe3], y[idx.ihe3], ratd[idx.ir33]),
            a2_he3_he4,
        ],
    );
    add_terms(
        &mut q_he4,
        &[m3(y[idx.io16], y[idx.ih1], ratd[idx.iropg]), a2_he3_he4],
    );
    if !deriva {
        q_he4 += m4(y[idx.in14], y[idx.ih1], ratd[idx.ifa], ratd[idx.irnpg]);
    } else {
        add_terms(
            &mut q_he4,
            &[
                m4(y[idx.in14], y[idx.ih1], ratd[idx.ifa], rat0[idx.irnpg]),
                m4(y[idx.in14], y[idx.ih1], rat0[idx.ifa], ratd[idx.irnpg]),
            ],
        );
    }
    dydt[idx.ihe4] = q_he4.to_f64();

    // c12 (lines 1126–1143)
    let mut q_c12 = Quad::ZERO;
    add_terms(
        &mut q_c12,
        &[
            m3(-y[idx.ic12], y[idx.ic12], ratd[idx.ir1212]),
            m3(-y[idx.ic12], y[idx.io16], ratd[idx.ir1216]),
            m5(
                1.0 / 6.0,
                y[idx.ihe4],
                y[idx.ihe4],
                y[idx.ihe4],
                ratd[idx.ir3a],
            ),
            m2(-y[idx.ic12], ratd[idx.irg3a]),
            m3(-y[idx.ic12], y[idx.ihe4], ratd[idx.ircag]),
            m2(y[idx.io16], ratd[idx.iroga]),
            m3(-y[idx.ic12], y[idx.ih1], ratd[idx.ircpg]),
        ],
    );
    if !deriva {
        q_c12 += m4(y[idx.in14], y[idx.ih1], ratd[idx.ifa], ratd[idx.irnpg]);
    } else {
        add_terms(
            &mut q_c12,
            &[
                m4(y[idx.in14], y[idx.ih1], ratd[idx.ifa], rat0[idx.irnpg]),
                m4(y[idx.in14], y[idx.ih1], rat0[idx.ifa], ratd[idx.irnpg]),
            ],
        );
    }
    dydt[idx.ic12] = q_c12.to_f64();

    // n14 (lines 1147–1152)
    let mut q_n14 = Quad::ZERO;
    add_terms(
        &mut q_n14,
        &[
            m3(y[idx.ic12], y[idx.ih1], ratd[idx.ircpg]),
            m3(-y[idx.in14], y[idx.ih1], ratd[idx.irnpg]),
            m3(y[idx.io16], y[idx.ih1], ratd[idx.iropg]),
            m3(-y[idx.ihe4], y[idx.in14], ratd[idx.irnag]),
        ],
    );
    dydt[idx.in14] = q_n14.to_f64();

    // o16 (lines 1156–1173)
    let mut q_o16 = Quad::ZERO;
    add_terms(
        &mut q_o16,
        &[
            m3(-y[idx.ic12], y[idx.io16], ratd[idx.ir1216]),
            m3(-y[idx.io16], y[idx.io16], ratd[idx.ir1616]),
            m3(y[idx.ic12], y[idx.ihe4], ratd[idx.ircag]),
            m3(-y[idx.io16], y[idx.ihe4], ratd[idx.iroag]),
            m2(-y[idx.io16], ratd[idx.iroga]),
            m2(y[idx.ine20], ratd[idx.irnega]),
            m3(-y[idx.io16], y[idx.ih1], ratd[idx.iropg]),
        ],
    );
    if !deriva {
        q_o16 += m4(y[idx.in14], y[idx.ih1], ratd[idx.ifg], ratd[idx.irnpg]);
    } else {
        add_terms(
            &mut q_o16,
            &[
                m4(y[idx.in14], y[idx.ih1], ratd[idx.ifg], rat0[idx.irnpg]),
                m4(y[idx.in14], y[idx.ih1], rat0[idx.ifg], ratd[idx.irnpg]),
            ],
        );
    }
    dydt[idx.io16] = q_o16.to_f64();

    // ne20 (lines 1177–1184)
    let mut q_ne20 = Quad::ZERO;
    add_terms(
        &mut q_ne20,
        &[
            m4(0.5, y[idx.ic12], y[idx.ic12], ratd[idx.ir1212]),
            m3(y[idx.io16], y[idx.ihe4], ratd[idx.iroag]),
            m3(-y[idx.ine20], y[idx.ihe4], ratd[idx.irneag]),
            m2(-y[idx.ine20], ratd[idx.irnega]),
            m2(y[idx.img24], ratd[idx.irmgga]),
            m3(y[idx.in14], y[idx.ihe4], ratd[idx.irnag]),
        ],
    );
    dydt[idx.ine20] = q_ne20.to_f64();

    // mg24 (lines 1188–1209)
    let mut q_mg24 = Quad::ZERO;
    add_terms(
        &mut q_mg24,
        &[
            m4(0.5, y[idx.ic12], y[idx.io16], ratd[idx.ir1216]),
            m3(y[idx.ine20], y[idx.ihe4], ratd[idx.irneag]),
            m3(-y[idx.img24], y[idx.ihe4], ratd[idx.irmgag]),
            m2(-y[idx.img24], ratd[idx.irmgga]),
            m2(y[idx.isi28], ratd[idx.irsiga]),
        ],
    );
    if !deriva {
        add_terms(
            &mut q_mg24,
            &[
                m4(
                    -y[idx.img24],
                    y[idx.ihe4],
                    ratd[idx.irmgap],
                    1.0 - ratd[idx.irr1],
                ),
                m3(y[idx.isi28], ratd[idx.irr1], ratd[idx.irsigp]),
            ],
        );
    } else {
        add_terms(
            &mut q_mg24,
            &[
                m4(
                    -y[idx.img24],
                    y[idx.ihe4],
                    ratd[idx.irmgap],
                    1.0 - rat0[idx.irr1],
                ),
                m4(y[idx.img24], y[idx.ihe4], rat0[idx.irmgap], ratd[idx.irr1]),
                m3(y[idx.isi28], rat0[idx.irr1], ratd[idx.irsigp]),
                m3(y[idx.isi28], ratd[idx.irr1], rat0[idx.irsigp]),
            ],
        );
    }
    dydt[idx.img24] = q_mg24.to_f64();

    // si28 (lines 1213–1246)
    let mut q_si28 = Quad::ZERO;
    add_terms(
        &mut q_si28,
        &[
            m4(0.5, y[idx.ic12], y[idx.io16], ratd[idx.ir1216]),
            m5(0.56, 0.5, y[idx.io16], y[idx.io16], ratd[idx.ir1616]),
            m3(y[idx.img24], y[idx.ihe4], ratd[idx.irmgag]),
            m3(-y[idx.isi28], y[idx.ihe4], ratd[idx.irsiag]),
            m2(-y[idx.isi28], ratd[idx.irsiga]),
            m2(y[idx.is32], ratd[idx.irsga]),
        ],
    );
    if !deriva {
        add_terms(
            &mut q_si28,
            &[
                m6(
                    0.34,
                    0.5,
                    y[idx.io16],
                    y[idx.io16],
                    ratd[idx.irs1],
                    ratd[idx.ir1616],
                ),
                m4(
                    y[idx.img24],
                    y[idx.ihe4],
                    ratd[idx.irmgap],
                    1.0 - ratd[idx.irr1],
                ),
                m3(-y[idx.isi28], ratd[idx.irr1], ratd[idx.irsigp]),
                m4(
                    -y[idx.isi28],
                    y[idx.ihe4],
                    ratd[idx.irsiap],
                    1.0 - ratd[idx.irs1],
                ),
                m3(y[idx.is32], ratd[idx.irs1], ratd[idx.irsgp]),
            ],
        );
    } else {
        add_terms(
            &mut q_si28,
            &[
                m6(
                    0.34,
                    0.5,
                    y[idx.io16],
                    y[idx.io16],
                    rat0[idx.irs1],
                    ratd[idx.ir1616],
                ),
                m6(
                    0.34,
                    0.5,
                    y[idx.io16],
                    y[idx.io16],
                    ratd[idx.irs1],
                    rat0[idx.ir1616],
                ),
                m4(
                    y[idx.img24],
                    y[idx.ihe4],
                    ratd[idx.irmgap],
                    1.0 - rat0[idx.irr1],
                ),
                m4(
                    -y[idx.img24],
                    y[idx.ihe4],
                    rat0[idx.irmgap],
                    ratd[idx.irr1],
                ),
                m3(-y[idx.isi28], rat0[idx.irr1], ratd[idx.irsigp]),
                m3(-y[idx.isi28], ratd[idx.irr1], rat0[idx.irsigp]),
                m4(
                    -y[idx.isi28],
                    y[idx.ihe4],
                    ratd[idx.irsiap],
                    1.0 - rat0[idx.irs1],
                ),
                m4(y[idx.isi28], y[idx.ihe4], rat0[idx.irsiap], ratd[idx.irs1]),
                m3(y[idx.is32], rat0[idx.irs1], ratd[idx.irsgp]),
                m3(y[idx.is32], ratd[idx.irs1], rat0[idx.irsgp]),
            ],
        );
    }
    dydt[idx.isi28] = q_si28.to_f64();

    // s32 (lines 1250–1282)
    let mut q_s32 = Quad::ZERO;
    add_terms(
        &mut q_s32,
        &[
            m5(0.1, 0.5, y[idx.io16], y[idx.io16], ratd[idx.ir1616]),
            m3(y[idx.isi28], y[idx.ihe4], ratd[idx.irsiag]),
            m3(-y[idx.is32], y[idx.ihe4], ratd[idx.irsag]),
            m2(-y[idx.is32], ratd[idx.irsga]),
            m2(y[idx.iar36], ratd[idx.irarga]),
        ],
    );
    if !deriva {
        add_terms(
            &mut q_s32,
            &[
                m6(
                    0.34,
                    0.5,
                    y[idx.io16],
                    y[idx.io16],
                    ratd[idx.ir1616],
                    1.0 - ratd[idx.irs1],
                ),
                m4(
                    y[idx.isi28],
                    y[idx.ihe4],
                    ratd[idx.irsiap],
                    1.0 - ratd[idx.irs1],
                ),
                m3(-y[idx.is32], ratd[idx.irs1], ratd[idx.irsgp]),
                m4(
                    -y[idx.is32],
                    y[idx.ihe4],
                    ratd[idx.irsap],
                    1.0 - ratd[idx.irt1],
                ),
                m3(y[idx.iar36], ratd[idx.irt1], ratd[idx.irargp]),
            ],
        );
    } else {
        add_terms(
            &mut q_s32,
            &[
                m6(
                    0.34,
                    0.5,
                    y[idx.io16],
                    y[idx.io16],
                    ratd[idx.ir1616],
                    1.0 - rat0[idx.irs1],
                ),
                m6(
                    -0.34,
                    0.5,
                    y[idx.io16],
                    y[idx.io16],
                    rat0[idx.ir1616],
                    ratd[idx.irs1],
                ),
                m4(
                    y[idx.isi28],
                    y[idx.ihe4],
                    ratd[idx.irsiap],
                    1.0 - rat0[idx.irs1],
                ),
                m4(
                    -y[idx.isi28],
                    y[idx.ihe4],
                    rat0[idx.irsiap],
                    ratd[idx.irs1],
                ),
                m3(-y[idx.is32], rat0[idx.irs1], ratd[idx.irsgp]),
                m3(-y[idx.is32], ratd[idx.irs1], rat0[idx.irsgp]),
                m4(
                    -y[idx.is32],
                    y[idx.ihe4],
                    ratd[idx.irsap],
                    1.0 - rat0[idx.irt1],
                ),
                m4(y[idx.is32], y[idx.ihe4], rat0[idx.irsap], ratd[idx.irt1]),
                m3(y[idx.iar36], rat0[idx.irt1], ratd[idx.irargp]),
                m3(y[idx.iar36], ratd[idx.irt1], rat0[idx.irargp]),
            ],
        );
    }
    dydt[idx.is32] = q_s32.to_f64();

    // ar36 (lines 1286–1312; note a4 and deriva a7..a8 use rgp * r1)
    let mut q_ar36 = Quad::ZERO;
    add_terms(
        &mut q_ar36,
        &[
            m3(y[idx.is32], y[idx.ihe4], ratd[idx.irsag]),
            m3(-y[idx.iar36], y[idx.ihe4], ratd[idx.irarag]),
            m2(-y[idx.iar36], ratd[idx.irarga]),
            m2(y[idx.ica40], ratd[idx.ircaga]),
        ],
    );
    if !deriva {
        add_terms(
            &mut q_ar36,
            &[
                m4(
                    y[idx.is32],
                    y[idx.ihe4],
                    ratd[idx.irsap],
                    1.0 - ratd[idx.irt1],
                ),
                m3(-y[idx.iar36], ratd[idx.irt1], ratd[idx.irargp]),
                m4(
                    -y[idx.iar36],
                    y[idx.ihe4],
                    ratd[idx.irarap],
                    1.0 - ratd[idx.iru1],
                ),
                m3(y[idx.ica40], ratd[idx.ircagp], ratd[idx.iru1]),
            ],
        );
    } else {
        add_terms(
            &mut q_ar36,
            &[
                m4(
                    y[idx.is32],
                    y[idx.ihe4],
                    ratd[idx.irsap],
                    1.0 - rat0[idx.irt1],
                ),
                m4(-y[idx.is32], y[idx.ihe4], rat0[idx.irsap], ratd[idx.irt1]),
                m3(-y[idx.iar36], rat0[idx.irt1], ratd[idx.irargp]),
                m3(-y[idx.iar36], ratd[idx.irt1], rat0[idx.irargp]),
                m4(
                    -y[idx.iar36],
                    y[idx.ihe4],
                    ratd[idx.irarap],
                    1.0 - rat0[idx.iru1],
                ),
                m4(y[idx.iar36], y[idx.ihe4], rat0[idx.irarap], ratd[idx.iru1]),
                m3(y[idx.ica40], rat0[idx.ircagp], ratd[idx.iru1]),
                m3(y[idx.ica40], ratd[idx.ircagp], rat0[idx.iru1]),
            ],
        );
    }
    dydt[idx.iar36] = q_ar36.to_f64();

    // ca40 (lines 1316–1343; note a2/a4 and deriva a3..a4,a7..a8 use rgp * r1)
    let mut q_ca40 = Quad::ZERO;
    add_terms(
        &mut q_ca40,
        &[
            m3(y[idx.iar36], y[idx.ihe4], ratd[idx.irarag]),
            m3(-y[idx.ica40], y[idx.ihe4], ratd[idx.ircaag]),
            m2(-y[idx.ica40], ratd[idx.ircaga]),
            m2(y[idx.iti44], ratd[idx.irtiga]),
        ],
    );
    if !deriva {
        add_terms(
            &mut q_ca40,
            &[
                m4(
                    y[idx.iar36],
                    y[idx.ihe4],
                    ratd[idx.irarap],
                    1.0 - ratd[idx.iru1],
                ),
                m3(-y[idx.ica40], ratd[idx.ircagp], ratd[idx.iru1]),
                m4(
                    -y[idx.ica40],
                    y[idx.ihe4],
                    ratd[idx.ircaap],
                    1.0 - ratd[idx.irv1],
                ),
                m3(y[idx.iti44], ratd[idx.irtigp], ratd[idx.irv1]),
            ],
        );
    } else {
        add_terms(
            &mut q_ca40,
            &[
                m4(
                    y[idx.iar36],
                    y[idx.ihe4],
                    ratd[idx.irarap],
                    1.0 - rat0[idx.iru1],
                ),
                m4(
                    -y[idx.iar36],
                    y[idx.ihe4],
                    rat0[idx.irarap],
                    ratd[idx.iru1],
                ),
                m3(-y[idx.ica40], rat0[idx.ircagp], ratd[idx.iru1]),
                m3(-y[idx.ica40], ratd[idx.ircagp], rat0[idx.iru1]),
                m4(
                    -y[idx.ica40],
                    y[idx.ihe4],
                    ratd[idx.ircaap],
                    1.0 - rat0[idx.irv1],
                ),
                m4(y[idx.ica40], y[idx.ihe4], rat0[idx.ircaap], ratd[idx.irv1]),
                m3(y[idx.iti44], rat0[idx.irtigp], ratd[idx.irv1]),
                m3(y[idx.iti44], ratd[idx.irtigp], rat0[idx.irv1]),
            ],
        );
    }
    dydt[idx.ica40] = q_ca40.to_f64();

    // ti44 (lines 1347–1373; uses r1 * rgp)
    let mut q_ti44 = Quad::ZERO;
    add_terms(
        &mut q_ti44,
        &[
            m3(y[idx.ica40], y[idx.ihe4], ratd[idx.ircaag]),
            m3(-y[idx.iti44], y[idx.ihe4], ratd[idx.irtiag]),
            m2(-y[idx.iti44], ratd[idx.irtiga]),
            m2(y[idx.icr48], ratd[idx.ircrga]),
        ],
    );
    if !deriva {
        add_terms(
            &mut q_ti44,
            &[
                m4(
                    y[idx.ica40],
                    y[idx.ihe4],
                    ratd[idx.ircaap],
                    1.0 - ratd[idx.irv1],
                ),
                m3(-y[idx.iti44], ratd[idx.irv1], ratd[idx.irtigp]),
                m4(
                    -y[idx.iti44],
                    y[idx.ihe4],
                    ratd[idx.irtiap],
                    1.0 - ratd[idx.irw1],
                ),
                m3(y[idx.icr48], ratd[idx.irw1], ratd[idx.ircrgp]),
            ],
        );
    } else {
        add_terms(
            &mut q_ti44,
            &[
                m4(
                    y[idx.ica40],
                    y[idx.ihe4],
                    ratd[idx.ircaap],
                    1.0 - rat0[idx.irv1],
                ),
                m4(
                    -y[idx.ica40],
                    y[idx.ihe4],
                    rat0[idx.ircaap],
                    ratd[idx.irv1],
                ),
                m3(-y[idx.iti44], rat0[idx.irv1], ratd[idx.irtigp]),
                m3(-y[idx.iti44], ratd[idx.irv1], rat0[idx.irtigp]),
                m4(
                    -y[idx.iti44],
                    y[idx.ihe4],
                    ratd[idx.irtiap],
                    1.0 - rat0[idx.irw1],
                ),
                m4(y[idx.iti44], y[idx.ihe4], rat0[idx.irtiap], ratd[idx.irw1]),
                m3(y[idx.icr48], rat0[idx.irw1], ratd[idx.ircrgp]),
                m3(y[idx.icr48], ratd[idx.irw1], rat0[idx.ircrgp]),
            ],
        );
    }
    dydt[idx.iti44] = q_ti44.to_f64();

    // cr48 (lines 1377–1403; uses r1 * rgp)
    let mut q_cr48 = Quad::ZERO;
    add_terms(
        &mut q_cr48,
        &[
            m3(y[idx.iti44], y[idx.ihe4], ratd[idx.irtiag]),
            m3(-y[idx.icr48], y[idx.ihe4], ratd[idx.ircrag]),
            m2(-y[idx.icr48], ratd[idx.ircrga]),
            m2(y[idx.ife52], ratd[idx.irfega]),
        ],
    );
    if !deriva {
        add_terms(
            &mut q_cr48,
            &[
                m4(
                    y[idx.iti44],
                    y[idx.ihe4],
                    ratd[idx.irtiap],
                    1.0 - ratd[idx.irw1],
                ),
                m3(-y[idx.icr48], ratd[idx.irw1], ratd[idx.ircrgp]),
                m4(
                    -y[idx.icr48],
                    y[idx.ihe4],
                    ratd[idx.ircrap],
                    1.0 - ratd[idx.irx1],
                ),
                m3(y[idx.ife52], ratd[idx.irx1], ratd[idx.irfegp]),
            ],
        );
    } else {
        add_terms(
            &mut q_cr48,
            &[
                m4(
                    y[idx.iti44],
                    y[idx.ihe4],
                    ratd[idx.irtiap],
                    1.0 - rat0[idx.irw1],
                ),
                m4(
                    -y[idx.iti44],
                    y[idx.ihe4],
                    rat0[idx.irtiap],
                    ratd[idx.irw1],
                ),
                m3(-y[idx.icr48], rat0[idx.irw1], ratd[idx.ircrgp]),
                m3(-y[idx.icr48], ratd[idx.irw1], rat0[idx.ircrgp]),
                m4(
                    -y[idx.icr48],
                    y[idx.ihe4],
                    ratd[idx.ircrap],
                    1.0 - rat0[idx.irx1],
                ),
                m4(y[idx.icr48], y[idx.ihe4], rat0[idx.ircrap], ratd[idx.irx1]),
                m3(y[idx.ife52], rat0[idx.irx1], ratd[idx.irfegp]),
                m3(y[idx.ife52], ratd[idx.irx1], rat0[idx.irfegp]),
            ],
        );
    }
    dydt[idx.icr48] = q_cr48.to_f64();

    approx21_dydt_iron_group(
        idx,
        y,
        rat0,
        ratd,
        deriva,
        fe56ec_fake_factor,
        nf,
        dydt,
    );

    if dydt[..species].iter().any(|v| !v.is_finite()) {
        return Err(NetError::IntegrationError(format_bad_dydt_error(
            idx,
            species,
            num_reactions,
            y,
            ratd,
            dydt,
            temp,
            den,
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::approx21_eval::approx21_dfdt_dfdrho;
    use crate::types::NetGeneralInfo;

    fn load_approx21(name: &str) -> (NetGeneralInfo, Approx21Indices) {
        let g = NetGeneralInfo::from_name(name).expect("load builtin net");
        let idx = Approx21Indices::from_net(&g);
        (g, idx)
    }

    #[test]
    fn approx21_dydt_quad_accumulation_is_left_to_right() {
        let (g, idx) = load_approx21("approx21.net");
        let species = g.num_isos;
        let nrat = g.num_reactions + 23;
        let big = flexmesa_math::powi(2.0, 113);

        // Witness 1 (he4, !deriva): block 2 adds onto block 1's Quad sum.
        let mut y = vec![0.0_f64; species];
        let mut rat = vec![0.0_f64; nrat];
        y[idx.ic12] = 1.0;
        rat[idx.ir1212] = 2.0 * big; // 0.5 * 1 * 1 * 2*big = big
        y[idx.io16] = 1.0;
        rat[idx.iroga] = 1.25;
        y[idx.ine20] = 1.0;
        rat[idx.irnega] = 1.25;
        y[idx.img24] = 1.0;
        rat[idx.irmgga] = -big;

        let mut dydt = vec![0.0_f64; species];
        approx21_dydt(
            &idx, &y, &rat, &rat, false, 0.0, 1.0e10, 0, 1.0e9, 1.0e7,
            &mut dydt,
        )
        .expect("finite he4 dydt");

        let seq_he4 = ((Quad::from_f64(big) + Quad::from_f64(1.25))
            + Quad::from_f64(1.25))
            + Quad::from_f64(-big);
        let grp_he4 = Quad::from_f64(big)
            + ((Quad::from_f64(1.25) + Quad::from_f64(1.25))
                + Quad::from_f64(-big));
        assert_ne!(
            seq_he4.to_f64().to_bits(),
            grp_he4.to_f64().to_bits(),
            "test witness must distinguish sequential vs grouped Quad sums",
        );
        assert_eq!(
            dydt[idx.ihe4].to_bits(),
            seq_he4.to_f64().to_bits(),
            "he4 must accumulate left-to-right sequentially in Quad",
        );

        // Witness 2 (ni56 in approx21_dfdy_heavy.rs): block 2 adds onto block 1
        y.fill(0.0);
        rat.fill(0.0);
        y[idx.ini56] = 1.0;
        rat[idx.irniga] = -big; // block 1: -y(ni56) * irniga = +big
        y[idx.ife54] = 1.0;
        y[idx.iprot] = 1.0;
        rat[idx.ir3f54] = 1.25; // block 2 t1
        y[idx.ife52] = 1.0;
        y[idx.ihe4] = 1.0;
        rat[idx.ir7f54] = 1.25; // block 2 t3
        rat[idx.ir8f54] = big; // block 2 t4 = -big

        dydt.fill(0.0);
        approx21_dydt(
            &idx, &y, &rat, &rat, true, 0.0, 1.0e10, 0, 1.0e9, 1.0e7, &mut dydt,
        )
        .expect("finite ni56 dydt");
        assert_eq!(
            dydt[idx.ini56].to_bits(),
            seq_he4.to_f64().to_bits(),
            "ni56 must accumulate left-to-right sequentially in Quad",
        );
    }

    #[test]
    fn approx21_dydt_ar36_ca40_operand_and_deriva_pair_order() {
        let (g, idx) = load_approx21("approx21.net");
        let species = g.num_isos;
        let nrat = g.num_reactions + 23;

        // Distinct non-associative IEEE-754 product witness:
        // ((y * rgp) * r1) != ((y * r1) * rgp) in f64.
        let y_val = 1.1_f64;
        let r1_val = 2.3_f64;
        let rgp_val = 1.1_f64;
        let y_rgp_r1 = (y_val * rgp_val) * r1_val;
        let y_r1_rgp = (y_val * r1_val) * rgp_val;
        assert_ne!(y_rgp_r1.to_bits(), y_r1_rgp.to_bits());

        // !deriva checks across ar36, ca40, ti44, cr48:
        // ar36 a2 = -y*r1*rgp, a4 = +y*rgp*r1
        // ca40 a2 = -y*rgp*r1, a4 = +y*rgp*r1
        // ti44/cr48 a2 = -y*r1*rgp, a4 = +y*r1*rgp
        let mut y = vec![0.0_f64; species];
        let mut rat = vec![0.0_f64; nrat];
        y[idx.ica40] = y_val;
        rat[idx.ircagp] = rgp_val;
        rat[idx.iru1] = r1_val;
        y[idx.iti44] = y_val;
        rat[idx.irtigp] = rgp_val;
        rat[idx.irv1] = r1_val;
        y[idx.icr48] = y_val;
        rat[idx.ircrgp] = rgp_val;
        rat[idx.irw1] = r1_val;
        y[idx.ife52] = y_val;
        rat[idx.irfegp] = rgp_val;
        rat[idx.irx1] = r1_val;
        let mut dydt = vec![0.0_f64; species];
        approx21_dydt(
            &idx, &y, &rat, &rat, false, 0.0, 1.0e10, 0, 1.0e9, 1.0e7,
            &mut dydt,
        )
        .expect("finite ar36..cr48 !deriva");

        // ar36 receives a4 = y(ca40)*ircagp*iru1 (y*rgp*r1).
        assert_eq!(dydt[idx.iar36].to_bits(), y_rgp_r1.to_bits());
        // ca40 receives a2 = -y(ca40)*ircagp*iru1 and a4 = y(ti44)*irtigp*irv1.
        let exp_ca40 = (Quad::from_f64((-y_val * rgp_val) * r1_val)
            + Quad::from_f64(y_rgp_r1))
        .to_f64();
        assert_eq!(dydt[idx.ica40].to_bits(), exp_ca40.to_bits());
        // ti44 receives a2 = -y(ti44)*irv1*irtigp and a4 = y(cr48)*irw1*ircrgp.
        let exp_ti44 = (Quad::from_f64((-y_val * r1_val) * rgp_val)
            + Quad::from_f64(y_r1_rgp))
        .to_f64();
        assert_eq!(dydt[idx.iti44].to_bits(), exp_ti44.to_bits());
        // cr48 receives a2 = -y(cr48)*irw1*ircrgp and a4 = y(fe52)*irx1*irfegp.
        assert_eq!(dydt[idx.icr48].to_bits(), exp_ti44.to_bits());

        // Check ar36 a2 (-y*r1*rgp) and ca40 a2 (-y*rgp*r1) individually.
        y.fill(0.0);
        rat.fill(0.0);
        y[idx.iar36] = y_val;
        rat[idx.irt1] = r1_val;
        rat[idx.irargp] = rgp_val;
        approx21_dydt(
            &idx, &y, &rat, &rat, false, 0.0, 1.0e10, 0, 1.0e9, 1.0e7,
            &mut dydt,
        )
        .expect("ar36 a2");
        assert_eq!(
            dydt[idx.iar36].to_bits(),
            ((-y_val * r1_val) * rgp_val).to_bits(),
        );

        y.fill(0.0);
        rat.fill(0.0);
        y[idx.ica40] = y_val;
        rat[idx.iru1] = r1_val;
        rat[idx.ircagp] = rgp_val;
        approx21_dydt(
            &idx, &y, &rat, &rat, false, 0.0, 1.0e10, 0, 1.0e9, 1.0e7,
            &mut dydt,
        )
        .expect("ca40 a2");
        assert_eq!(
            dydt[idx.ica40].to_bits(),
            ((-y_val * rgp_val) * r1_val).to_bits(),
        );

        // deriva pair-order witness on ar36 (a3..a4 then a7..a8):
        // ((big + 1.25) - big) = 2.0 whereas ((big - big) + 1.25) = 1.25.
        let big = flexmesa_math::powi(2.0, 113);
        let mut rat0 = vec![0.0_f64; nrat];
        let mut ratd = vec![0.0_f64; nrat];
        y.fill(0.0);
        y[idx.iar36] = 1.0;
        rat0[idx.irt1] = -big;
        ratd[idx.irargp] = 1.0; // a3 = +big; rat0[irargp] = 0 => a4 = 0
        y[idx.ica40] = 1.0;
        rat0[idx.ircagp] = 1.25;
        ratd[idx.iru1] = 1.0; // a7 = y*rat0[ircagp]*ratd[iru1] = 1.25
        ratd[idx.ircagp] = -big;
        rat0[idx.iru1] = 1.0; // a8 = y*ratd[ircagp]*rat0[iru1] = -big
        approx21_dydt(
            &idx, &y, &rat0, &ratd, true, 0.0, 1.0e10, 0, 1.0e9, 1.0e7,
            &mut dydt,
        )
        .expect("ar36 deriva pair order");
        assert_eq!(dydt[idx.iar36], 2.0);
        assert_ne!(dydt[idx.iar36], 1.25);

        // deriva pair-order witness on ti44 (a3 + a7 then a8):
        // a7 = y(cr48)*rat0[irw1]*ratd[ircrgp],
        // a8 = y(cr48)*ratd[irw1]*rat0[ircrgp].
        y.fill(0.0);
        rat0.fill(0.0);
        ratd.fill(0.0);
        y[idx.iti44] = 1.0;
        rat0[idx.irv1] = -big;
        ratd[idx.irtigp] = 1.0; // a3 = +big
        y[idx.icr48] = 1.0;
        rat0[idx.irw1] = 1.25;
        ratd[idx.ircrgp] = 1.0; // a7 = 1.25
        ratd[idx.irw1] = -big;
        rat0[idx.ircrgp] = 1.0; // a8 = -big
        approx21_dydt(
            &idx, &y, &rat0, &ratd, true, 0.0, 1.0e10, 0, 1.0e9, 1.0e7,
            &mut dydt,
        )
        .expect("ti44 deriva pair order");
        assert_eq!(dydt[idx.iti44], 2.0);
        assert_ne!(dydt[idx.iti44], 1.25);
    }

    #[test]
    fn approx21_dydt_is_bad_reports_diagnostics_and_zeroes_only_species() {
        for (net_name, exp_species, ec_name) in [
            ("approx21.net", 21_usize, "rni56ec_to_fe56"),
            ("approx21_plus_co56.net", 22_usize, "r_ni56_wk_co56"),
        ] {
            let (g, idx) = load_approx21(net_name);
            let nrat = g.num_reactions + 23;
            let y = vec![1.0e-3_f64; exp_species];
            let mut rat = vec![1.0_f64; nrat];
            let mut dydt = vec![42.0_f64; exp_species + 1];

            // Finite call zeroes only 0..species and preserves sentinel.
            approx21_dydt(
                &idx, &y, &rat, &rat, false, 0.0, 1.0e10, 0, 1.0e9, 1.0e6,
                &mut dydt,
            )
            .expect("finite call succeeds");
            assert_eq!(dydt[exp_species].to_bits(), 42.0_f64.to_bits());

            // NaN and Inf both return IntegrationError with full diagnostics.
            for bad_val in [f64::NAN, f64::INFINITY] {
                rat[idx.ir3a] = bad_val;
                let err = approx21_dydt(
                    &idx, &y, &rat, &rat, false, 0.0, 1.0e10, 0, 1.0e9, 1.0e6,
                    &mut dydt,
                )
                .expect_err("non-finite dydt must error");
                let msg = err.to_string();
                assert!(matches!(err, NetError::IntegrationError(_)));
                assert!(msg.contains("dydt(i) "), "missing species in: {msg}");
                assert!(msg.contains("log10(temp) = 9"), "missing T: {msg}");
                assert!(msg.contains("log10(rho) = 6"), "missing rho: {msg}");
                assert!(msg.contains("r_he4_he4_he4_to_c12"));
                assert!(msg.contains(ec_name));
                assert!(msg.contains("rfe56aux4"));
                if idx.plus_co56 {
                    assert!(msg.contains("r_co56_wk_fe56"));
                }
            }
        }
    }

    #[test]
    fn approx21_dfdt_dfdrho_propagates_bad_dydt_error() {
        let (g, idx) = load_approx21("approx21.net");
        let species = g.num_isos;
        let nrat = g.num_reactions + 23;
        let y = vec![1.0e-3_f64; species];
        let rat0 = vec![1.0_f64; nrat];
        let mut dratdt = vec![0.0_f64; nrat];
        let mut dratdd = vec![0.0_f64; nrat];
        let mion = vec![1.0_f64; species];
        let dfdy = vec![0.0_f64; species * species];
        let mut dfdt = vec![0.0_f64; species];
        let mut dfdrho = vec![42.0_f64; species];
        let mut d_epstotal_dy = vec![42.0_f64; species];

        dratdt[idx.ir3a] = f64::NAN;
        let err_dt = approx21_dfdt_dfdrho(
            &idx,
            species,
            &y,
            &rat0,
            &dratdt,
            &dratdd,
            &mion,
            0.0,
            1.0e10,
            0,
            1.0e9,
            1.0e6,
            &dfdy,
            &mut dfdt,
            &mut dfdrho,
            &mut d_epstotal_dy,
        )
        .expect_err("NaN dratdumdt must abort");
        assert!(matches!(err_dt, NetError::IntegrationError(_)));
        assert_eq!(dfdrho[0].to_bits(), 42.0_f64.to_bits());
        assert_eq!(d_epstotal_dy[0].to_bits(), 42.0_f64.to_bits());

        dratdt[idx.ir3a] = 0.0;
        dratdd[idx.ir3a] = f64::INFINITY;
        let err_dd = approx21_dfdt_dfdrho(
            &idx,
            species,
            &y,
            &rat0,
            &dratdt,
            &dratdd,
            &mion,
            0.0,
            1.0e10,
            0,
            1.0e9,
            1.0e6,
            &dfdy,
            &mut dfdt,
            &mut dfdrho,
            &mut d_epstotal_dy,
        )
        .expect_err("Inf dratdumdd must abort");
        assert!(matches!(err_dd, NetError::IntegrationError(_)));
        assert_eq!(d_epstotal_dy[0].to_bits(), 42.0_f64.to_bits());
    }
}
