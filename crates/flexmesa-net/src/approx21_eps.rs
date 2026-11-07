// SPDX-License-Identifier: LGPL-3.0-or-later
//! Energy generation, category decomposition, and neutrino loss derivatives for
//! `approx21` and `approx21_plus_co56` (`net/private/net_approx21.f90`
//! lines 1547–2030 and `net/private/net_eval.f90` lines 443–552).

use flexmesa_const::{AVO, CLIGHT, QCONV};

use crate::approx21::Approx21Indices;
use crate::approx21_special::get_qs_rfe56ec;
pub use crate::approx21_special::{global_approx21_qs, Approx21ReactionQs};
use crate::f64x2::F64x2;
use crate::types::{NetGeneralInfo, NUM_CATEGORIES};

const CAT_PP: usize = 0;
const CAT_CNO: usize = 1;
const CAT_3ALF: usize = 2;
const CAT_BURN_C: usize = 3;
const CAT_BURN_N: usize = 4;
const CAT_BURN_O: usize = 5;
const CAT_BURN_NE: usize = 6;
const CAT_BURN_MG: usize = 8;
const CAT_BURN_SI: usize = 9;
const CAT_BURN_S: usize = 10;
const CAT_BURN_AR: usize = 11;
const CAT_BURN_CA: usize = 12;
const CAT_BURN_TI: usize = 13;
const CAT_BURN_CR: usize = 14;
const CAT_BURN_FE: usize = 15;
const CAT_CC: usize = 16;
const CAT_CO: usize = 17;
const CAT_OO: usize = 18;
const CAT_PHOTO: usize = 19;
const CAT_PNHE4: usize = 20;
const CAT_NI56_CO56: usize = 21;
const CAT_CO56_FE56: usize = 22;
const CAT_OTHER: usize = 23;

/// Evaluates the PP-II branch fraction `fII` (`approx21_eval_PPII_fraction`
/// in `net/private/net_approx21.f90` lines 1547–1559).
#[must_use]
pub fn approx21_eval_ppii_fraction(
    idx: &Approx21Indices,
    y: &[f64],
    rate: &[f64],
) -> f64 {
    let r2 = rate[idx.ir_be7_wk_li7];
    let r3 = y[idx.ih1] * rate[idx.ir_be7_pg_b8];
    let rsum = r2 + r3;
    if rsum < 1.0e-50 {
        0.5
    } else {
        r2 / rsum
    }
}

/// Evaluates `approx21_eps_info` (`net/private/net_approx21.f90` lines 1562–1982),
/// returning `(eps_total, eps_neu)`.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn approx21_eps_info(
    g: &NetGeneralInfo,
    idx: &Approx21Indices,
    y: &[f64],
    dydt: &[f64],
    rate: &[f64],
    f_ii: f64,
    temp: f64,
    do_eps_nuc_categories: bool,
    eps_nuc_categories: &mut [f64; NUM_CATEGORIES],
) -> (f64, f64) {
    let qs = global_approx21_qs();
    let (qtotal_rfe56ec, qneu_rfe56ec) = get_qs_rfe56ec(g);

    let mut xx = F64x2::ZERO;
    for i in 0..g.num_isos {
        xx += F64x2::from_f64(dydt[i]) * F64x2::from_f64(g.mion[i]);
    }
    let avo_c2 = F64x2::from_f64(AVO)
        * F64x2::from_f64(CLIGHT)
        * F64x2::from_f64(CLIGHT);
    let eps_total = (-avo_c2 * xx).to_f64();

    let fe56ec_fake_factor = if temp >= g.min_t_for_fe56ec_fake_factor {
        g.fe56ec_fake_factor
    } else {
        0.0
    };

    let m2 = |a1: f64, a2: f64| F64x2::from_f64(a1) * F64x2::from_f64(a2);
    let m3 = |a1: f64, a2: f64, a3: f64| m2(a1, a2) * F64x2::from_f64(a3);
    let m4 = |a1: f64, a2: f64, a3: f64, a4: f64| {
        m3(a1, a2, a3) * F64x2::from_f64(a4)
    };
    let m5 = |a1: f64, a2: f64, a3: f64, a4: f64, a5: f64| {
        m4(a1, a2, a3, a4) * F64x2::from_f64(a5)
    };

    let mut eps_neu_q =
        m5(qs.qneu_rpp, 0.5, y[idx.ih1], y[idx.ih1], rate[idx.irpp])
            + m5(
                qs.qneu_rpp2,
                y[idx.ihe3],
                y[idx.ihe4],
                rate[idx.irhe3ag],
                f_ii,
            )
            + m5(
                qs.qneu_rpp3,
                y[idx.ihe3],
                y[idx.ihe4],
                rate[idx.irhe3ag],
                1.0 - f_ii,
            )
            + m4(qs.qneu_rcpg, y[idx.ic12], y[idx.ih1], rate[idx.ircpg])
            + m5(
                qs.qneu_rnpg,
                y[idx.in14],
                y[idx.ih1],
                rate[idx.irnpg],
                rate[idx.ifa],
            )
            + m4(qs.qneu_ropg, y[idx.io16], y[idx.ih1], rate[idx.iropg])
            + m3(qs.qneu_rpen, y[idx.ih1], rate[idx.irpen])
            + m3(qs.qneu_rpen, y[idx.iprot], rate[idx.irpen])
            + m3(qs.qneu_rnep, y[idx.ineut], rate[idx.irnep])
            + m4(
                qneu_rfe56ec,
                y[idx.ife56],
                fe56ec_fake_factor,
                rate[idx.irn56ec],
            );

    if idx.plus_co56 {
        eps_neu_q += m3(qs.qneu_rni56ec, y[idx.ini56], rate[idx.irn56ec])
            + m3(qs.qneu_rco56ec, y[idx.ico56], rate[idx.irco56ec]);
    } else {
        eps_neu_q += m3(
            qs.qneu_rni56ec + qs.qneu_rco56ec,
            y[idx.ini56],
            rate[idx.irn56ec],
        );
    }
    let eps_neu = (eps_neu_q * F64x2::from_f64(QCONV)).to_f64();

    if !do_eps_nuc_categories {
        return (eps_total, eps_neu);
    }

    let mut cat_q = [F64x2::ZERO; NUM_CATEGORIES];
    let mut add = |q: f64, ys: &[f64], rat_id: usize, eps_id: usize| {
        let mut prod = F64x2::from_f64(1.0);
        for &v in ys {
            prod *= F64x2::from_f64(v);
        }
        let val2 = prod * F64x2::from_f64(rate[rat_id]);
        cat_q[eps_id] += F64x2::from_f64(q) * val2;
    };

    add(
        qs.qtotal_rpp - qs.qneu_rpp,
        &[0.5, y[idx.ih1], y[idx.ih1]],
        idx.irpp,
        CAT_PP,
    );
    add(qs.qr33, &[0.5, y[idx.ihe3], y[idx.ihe3]], idx.ir33, CAT_PP);
    add(
        (qs.qtotal_rpp2 - qs.qneu_rpp2) * f_ii
            + (qs.qtotal_rpp3 - qs.qneu_rpp3) * (1.0 - f_ii),
        &[y[idx.ihe3], y[idx.ihe4]],
        idx.irhe3ag,
        CAT_PP,
    );

    add(
        qs.qtotal_rcpg - qs.qneu_rcpg,
        &[y[idx.ic12], y[idx.ih1]],
        idx.ircpg,
        CAT_CNO,
    );
    add(
        qs.qtotal_rcpg - qs.qneu_rnpg,
        &[y[idx.in14], y[idx.ih1]],
        idx.irnpg,
        CAT_CNO,
    );
    add(
        qs.qtotal_ropg - qs.qneu_ropg,
        &[y[idx.io16], y[idx.ih1], rate[idx.ifa]],
        idx.iropg,
        CAT_CNO,
    );

    add(
        qs.qr3alf,
        &[1.0 / 6.0, y[idx.ihe4], y[idx.ihe4], y[idx.ihe4]],
        idx.ir3a,
        CAT_3ALF,
    );
    add(
        qs.qrc12ag,
        &[y[idx.ic12], y[idx.ihe4]],
        idx.ircag,
        CAT_BURN_C,
    );
    add(
        qs.qrn14ag,
        &[y[idx.ihe4], y[idx.in14]],
        idx.irnag,
        CAT_BURN_N,
    );
    add(
        qs.qrn14_to_o16,
        &[y[idx.in14], y[idx.ih1], rate[idx.ifg]],
        idx.irnpg,
        CAT_BURN_N,
    );
    add(
        qs.qro16ag,
        &[y[idx.io16], y[idx.ihe4]],
        idx.iroag,
        CAT_BURN_O,
    );
    add(
        qs.qr1212,
        &[0.5, y[idx.ic12], y[idx.ic12]],
        idx.ir1212,
        CAT_CC,
    );
    add(
        0.5 * (qs.qr1216_to_mg24 + qs.qr1216_to_si28),
        &[y[idx.ic12], y[idx.io16]],
        idx.ir1216,
        CAT_CO,
    );
    add(
        qs.qr1616a * (0.56 + 0.34 * rate[idx.irs1]),
        &[0.5, y[idx.io16], y[idx.io16]],
        idx.ir1616,
        CAT_OO,
    );
    add(
        qs.qr1616g * (0.1 + 0.34 * (1.0 - rate[idx.irs1])),
        &[0.5, y[idx.io16], y[idx.io16]],
        idx.ir1616,
        CAT_OO,
    );

    add(
        qs.qrne20ag,
        &[y[idx.ihe4], y[idx.ine20]],
        idx.irneag,
        CAT_BURN_NE,
    );
    for (qval, y_iso, rag, rap, r1, cat) in [
        (
            qs.qrmg24ag,
            y[idx.img24],
            idx.irmgag,
            idx.irmgap,
            idx.irr1,
            CAT_BURN_MG,
        ),
        (
            qs.qrsi28ag,
            y[idx.isi28],
            idx.irsiag,
            idx.irsiap,
            idx.irs1,
            CAT_BURN_SI,
        ),
        (
            qs.qrs32ag,
            y[idx.is32],
            idx.irsag,
            idx.irsap,
            idx.irt1,
            CAT_BURN_S,
        ),
        (
            qs.qrar36ag,
            y[idx.iar36],
            idx.irarag,
            idx.irarap,
            idx.iru1,
            CAT_BURN_AR,
        ),
        (
            qs.qrca40ag,
            y[idx.ica40],
            idx.ircaag,
            idx.ircaap,
            idx.irv1,
            CAT_BURN_CA,
        ),
        (
            qs.qrti44ag,
            y[idx.iti44],
            idx.irtiag,
            idx.irtiap,
            idx.irw1,
            CAT_BURN_TI,
        ),
        (
            qs.qrcr48ag,
            y[idx.icr48],
            idx.ircrag,
            idx.ircrap,
            idx.irx1,
            CAT_BURN_CR,
        ),
    ] {
        add(qval, &[y[idx.ihe4], y_iso], rag, cat);
        add(qval, &[y[idx.ihe4], y_iso, 1.0 - rate[r1]], rap, cat);
    }

    add(
        qs.qrfe52ag,
        &[y[idx.ihe4], y[idx.ife52]],
        idx.irfeag,
        CAT_BURN_FE,
    );
    add(
        qs.qrfe52aprot_to_ni56,
        &[y[idx.ife52], y[idx.ihe4], y[idx.iprot]],
        idx.ir7f54,
        CAT_BURN_FE,
    );
    add(
        qs.qrfe52neut_to_fe54,
        &[y[idx.ife52], y[idx.ineut], y[idx.ineut]],
        idx.ir2f54,
        CAT_BURN_FE,
    );
    add(
        qs.qrfe54ng_to_fe56,
        &[y[idx.ife54], y[idx.ineut], y[idx.ineut]],
        idx.irfe56_aux2,
        CAT_BURN_FE,
    );
    add(
        qs.qrfe52aprot_to_fe54,
        &[y[idx.ife52], y[idx.ihe4]],
        idx.ir6f54,
        CAT_BURN_FE,
    );
    add(
        qs.qrfe54aprot_to_fe56,
        &[y[idx.ife54], y[idx.ihe4]],
        idx.irfe56_aux4,
        CAT_BURN_FE,
    );
    add(
        qs.qrfe54prot_to_ni56,
        &[y[idx.ife54], y[idx.iprot], y[idx.iprot]],
        idx.ir3f54,
        CAT_BURN_FE,
    );
    add(
        qtotal_rfe56ec - qneu_rfe56ec,
        &[y[idx.ife56], fe56ec_fake_factor],
        idx.irn56ec,
        CAT_BURN_FE,
    );

    add(
        qs.qrhe4_rebuild,
        &[y[idx.ineut], y[idx.ineut], y[idx.iprot], y[idx.iprot]],
        idx.iralf2,
        CAT_PNHE4,
    );
    add(qs.qtotal_rpen, &[y[idx.ih1]], idx.irpen, CAT_OTHER);
    add(qs.qtotal_rpen, &[y[idx.iprot]], idx.irpen, CAT_OTHER);
    add(qs.qtotal_rnep, &[y[idx.ineut]], idx.irnep, CAT_OTHER);

    add(qs.qrhe4_breakup, &[y[idx.ihe4]], idx.iralf1, CAT_PHOTO);
    add(-qs.qrc12ag, &[y[idx.io16]], idx.iroga, CAT_PHOTO);
    add(-qs.qr3alf, &[y[idx.ic12]], idx.irg3a, CAT_PHOTO);
    add(-qs.qro16ag, &[y[idx.ine20]], idx.irnega, CAT_PHOTO);
    add(-qs.qrne20ag, &[y[idx.img24]], idx.irmgga, CAT_PHOTO);

    for (qval, y_iso, rga, rgp, r1) in [
        (-qs.qrmg24ag, y[idx.isi28], idx.irsiga, idx.irsigp, idx.irr1),
        (-qs.qrsi28ag, y[idx.is32], idx.irsga, idx.irsgp, idx.irs1),
        (-qs.qrs32ag, y[idx.iar36], idx.irarga, idx.irargp, idx.irt1),
        (-qs.qrar36ag, y[idx.ica40], idx.ircaga, idx.ircagp, idx.iru1),
        (-qs.qrca40ag, y[idx.iti44], idx.irtiga, idx.irtigp, idx.irv1),
        (-qs.qrti44ag, y[idx.icr48], idx.ircrga, idx.ircrgp, idx.irw1),
        (-qs.qrcr48ag, y[idx.ife52], idx.irfega, idx.irfegp, idx.irx1),
    ] {
        add(qval, &[y_iso], rga, CAT_PHOTO);
        add(qval, &[y_iso, rate[r1]], rgp, CAT_PHOTO);
    }

    add(
        -qs.qrfe52aprot_to_ni56,
        &[y[idx.ini56], y[idx.iprot]],
        idx.ir8f54,
        CAT_PHOTO,
    );
    add(
        -qs.qrfe52aprot_to_fe54,
        &[y[idx.ife54], y[idx.iprot], y[idx.iprot]],
        idx.ir5f54,
        CAT_PHOTO,
    );
    add(-qs.qrfe52ag, &[y[idx.ini56]], idx.irniga, CAT_PHOTO);
    add(
        -qs.qrfe52neut_to_fe54,
        &[y[idx.ife54]],
        idx.ir1f54,
        CAT_PHOTO,
    );
    add(
        -qs.qrfe54ng_to_fe56,
        &[y[idx.ife56]],
        idx.irfe56_aux1,
        CAT_PHOTO,
    );
    add(
        -qs.qrfe54aprot_to_fe56,
        &[y[idx.ife56], y[idx.iprot], y[idx.iprot]],
        idx.irfe56_aux3,
        CAT_PHOTO,
    );
    add(
        -qs.qrfe54prot_to_ni56,
        &[y[idx.ini56]],
        idx.ir4f54,
        CAT_PHOTO,
    );

    add(
        qs.qtotal_rni56ec - qs.qneu_rni56ec,
        &[y[idx.ini56]],
        idx.irn56ec,
        CAT_NI56_CO56,
    );
    if idx.plus_co56 {
        add(
            qs.qtotal_rco56ec - qs.qneu_rco56ec,
            &[y[idx.ico56]],
            idx.irco56ec,
            CAT_CO56_FE56,
        );
    } else {
        add(
            qs.qtotal_rco56ec - qs.qneu_rco56ec,
            &[y[idx.ini56]],
            idx.irn56ec,
            CAT_CO56_FE56,
        );
    }

    for k in 0..NUM_CATEGORIES {
        eps_nuc_categories[k] = (cat_q[k] * F64x2::from_f64(QCONV)).to_f64();
    }

    (eps_total, eps_neu)
}

/// Evaluates `approx21_d_epsneu_dy` (`net/private/net_approx21.f90` lines 1985–2030).
///
/// Mirrors upstream MESA's approximate `fII = 0.5d0` formula (`net_approx21.f90:1985`)
/// for exact bit-parity with MESA's `approx21` analytical derivatives.
pub fn approx21_d_epsneu_dy(
    idx: &Approx21Indices,
    y: &[f64],
    rate: &[f64],
    d_epsneu_dy: &mut [f64],
) {
    let qs = global_approx21_qs();
    let f_ii = 0.5_f64;
    d_epsneu_dy.fill(0.0);
    d_epsneu_dy[idx.ih1] = QCONV
        * (qs.qneu_rpp * y[idx.ih1] * rate[idx.irpp]
            + qs.qneu_rcpg * y[idx.ic12] * rate[idx.ircpg]
            + qs.qneu_rnpg * y[idx.in14] * rate[idx.irnpg]
            + qs.qneu_ropg * y[idx.io16] * rate[idx.iropg]);
    d_epsneu_dy[idx.ihe3] = QCONV
        * (qs.qneu_rpp2 * y[idx.ihe4] * rate[idx.irhe3ag] * f_ii
            + qs.qneu_rpp3 * y[idx.ihe4] * rate[idx.irhe3ag] * (1.0 - f_ii));
    d_epsneu_dy[idx.ihe4] = QCONV
        * (qs.qneu_rpp2 * y[idx.ihe3] * rate[idx.irhe3ag] * f_ii
            + qs.qneu_rpp3 * y[idx.ihe3] * rate[idx.irhe3ag] * (1.0 - f_ii));
    d_epsneu_dy[idx.ic12] = QCONV * qs.qneu_rcpg * y[idx.ih1] * rate[idx.ircpg];
    d_epsneu_dy[idx.in14] = QCONV * qs.qneu_rnpg * y[idx.ih1] * rate[idx.irnpg];
    d_epsneu_dy[idx.io16] = QCONV * qs.qneu_ropg * y[idx.ih1] * rate[idx.iropg];
}
