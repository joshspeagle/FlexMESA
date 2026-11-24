// SPDX-License-Identifier: GPL-3.0-or-later
//! Chugunov, DeWitt & Yakovlev (2007, PhRvD 76, 025028) one-component plasma
//! electrostatic screening ported from `rates/private/screen_chugunov.f90`.

use flexmesa_const::{
    AMU, FOUR_THIRDS, HBAR, KERG, ONE_THIRD, PI, PI2, PI4, QE, TWO_THIRDS,
};
use flexmesa_math::{cospi, exp, pow, sinpi};

use crate::types::ScreenEval;

/// Evaluates Chugunov, DeWitt & Yakovlev (2007) plasma screening (`eval_screen_chugunov`
/// in `rates/private/screen_chugunov.f90` and `rates/private/screen.f90:101-102`)
/// with analytical $(T, \rho)$ partial derivatives.
#[must_use]
#[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
pub fn screen_chugunov(
    temp: f64,
    den: f64,
    zbar: f64,
    abar: f64,
    a1: f64,
    z1: f64,
    a2: f64,
    z2: f64,
) -> ScreenEval {
    const C_A1: f64 = 2.7822_f64;
    const C_A2: f64 = 98.34_f64;
    const C_B1: f64 = -1.7476_f64;
    const C_B2: f64 = 66.07_f64;
    const C_B3: f64 = 1.12_f64;
    const C_B4: f64 = 65.0_f64;
    const GAMMA_1: f64 = 590.0_f64;
    const GAMMA_2: f64 = 600.0_f64;
    const DELTAGAMMA: f64 = GAMMA_2 - GAMMA_1;
    const TP1: f64 = 0.1_f64;
    const TP2: f64 = 0.2_f64;
    const DELTATP: f64 = TP2 - TP1;

    // Upstream `eval_screen_chugunov` (`rates/private/screen_chugunov.f90:115-120`)
    // returns unscreened unity (`scor = 1, dscor_dt = 0, dscor_dd = 0`) for
    // neutral species (`z1 <= 0` or `z2 <= 0`) or hydrogen-dominated plasmas
    // (`zbar <= 1`).
    if z1 <= 0.0 || z2 <= 0.0 || zbar <= 1.0 || temp <= 0.0 || den <= 0.0 {
        return ScreenEval::UNSCREENED;
    }

    // `screen.f90:101-102`
    let ntot = den / (AMU * abar);
    let a_e = pow(3.0 / (PI4 * zbar * ntot), ONE_THIRD);

    // `screen_chugunov.f90:136-171`
    let z1z2 = f64::from((z1 as i32) * (z2 as i32));
    let m1 = f64::from(a1.round() as i32) * AMU;
    let m2 = f64::from(a2.round() as i32) * AMU;
    let _ = (m1, m2);
    let mav = abar * AMU;

    let a_1 = a_e * pow(f64::from(z1 as i32), ONE_THIRD);
    let a_2 = a_e * pow(f64::from(z2 as i32), ONE_THIRD);
    let da_1dd = -ONE_THIRD * a_1 / den;
    let da_2dd = -ONE_THIRD * a_2 / den;
    let a_av = 0.5 * (a_1 + a_2);
    let da_avdd = 0.5 * (da_1dd + da_2dd);

    let wp = (PI4 * z1z2 * QE * QE * ntot / mav).sqrt();
    let dwpdd = 0.5 * wp / den;
    let tp = HBAR * wp / KERG;
    let dtpdd = (HBAR / KERG) * dwpdd;
    let tn = temp / tp;
    let dtndt = 1.0 / tp;
    let dtndd = -tn / tp * dtpdd;

    // `screen_chugunov.f90:173-195`
    let (tk, dtkdt, dtkdd, denom_zeta) = if tn > TP2 {
        let tk_v = temp;
        let dtk_dt = 1.0;
        let dtk_dd = 0.0;
        let den_v = tp * (tk_v * dtpdd - tp * dtk_dd) / (tk_v * tk_v * tk_v);
        (tk_v, dtk_dt, dtk_dd, den_v)
    } else if tn <= TP1 {
        let tk_v = TP1 * tp;
        let dtk_dt = 0.0;
        let dtk_dd = TP1 * dtpdd;
        (tk_v, dtk_dt, dtk_dd, 0.0)
    } else {
        let alpha = 0.5 * (1.0 - cospi((tn - TP1) / DELTATP));
        let dalphadtn = 0.5 * (PI / DELTATP) * sinpi((tn - TP1) / DELTATP);
        let beta = 1.0 - alpha;
        let dbetadtn = -dalphadtn;
        let dtk_dt =
            (dbetadtn * dtndt * TP1 * tp) + (dalphadtn * dtndt * TP2 * tp);
        let dtk_dd = (beta * TP1 * dtpdd)
            + (dbetadtn * dtndd * TP1 * tp)
            + (alpha * TP2 * dtpdd)
            + (dalphadtn * dtndd * TP2 * tp);
        let tk_v = beta * (TP1 * tp) + alpha * (TP2 * tp);
        let den_v = tp * (tk_v * dtpdd - tp * dtk_dd) / (tk_v * tk_v * tk_v);
        (tk_v, dtk_dt, dtk_dd, den_v)
    };

    // `screen_chugunov.f90:197-226`
    let u_zeta = FOUR_THIRDS * (tp * tp) / (PI2 * tk * tk);
    let zeta = pow(u_zeta, ONE_THIRD);
    let dzetadt = -TWO_THIRDS * (zeta / tk) * dtkdt;
    let dzetadd =
        ONE_THIRD * (zeta / u_zeta) * (FOUR_THIRDS / PI2) * 2.0 * denom_zeta;

    let mut gam = z1z2 * QE * QE / (a_av * tk * KERG);
    let mut dgamdt = -(gam / tk) * dtkdt;
    let mut dgamdd = -(gam / (a_av * tk)) * (tk * da_avdd + a_av * dtkdd);

    if gam >= GAMMA_2 {
        gam = GAMMA_2;
        dgamdt = 0.0;
        dgamdd = 0.0;
    } else if gam >= GAMMA_1 {
        let alpha = 0.5 * (1.0 - cospi((gam - GAMMA_1) / DELTAGAMMA));
        let dalphadgam =
            0.5 * PI * sinpi((gam - GAMMA_1) / DELTAGAMMA) * (1.0 / DELTAGAMMA);
        let beta = 1.0 - alpha;
        let dbetadgam = -dalphadgam;
        let dgamdt_new = beta * dgamdt
            + (dbetadgam * dgamdt) * gam
            + (dalphadgam * dgamdt) * GAMMA_2;
        let dgamdd_new = beta * dgamdd
            + (dbetadgam * dgamdd) * gam
            + (dalphadgam * dgamdd) * GAMMA_2;
        gam = beta * gam + alpha * GAMMA_2;
        dgamdt = dgamdt_new;
        dgamdd = dgamdd_new;
    }

    // `screen_chugunov.f90:228-252`
    let alfa = 0.022;
    let beta = 0.41 - 0.6 / gam;
    let dbetadt = 0.6 / (gam * gam) * dgamdt;
    let dbetadd = 0.6 / (gam * gam) * dgamdd;
    let gama = 0.06 + 2.2 / gam;
    let dgamadt = -2.2 / (gam * gam) * dgamdt;
    let dgamadd = -2.2 / (gam * gam) * dgamdd;

    let zeta2 = zeta * zeta;
    let zeta3 = zeta2 * zeta;
    let u = 1.0 + alfa * zeta + beta * zeta2 + gama * zeta3;
    let denom = pow(u, -ONE_THIRD);
    let denom2 = denom / u;
    let ddenomdt = -ONE_THIRD
        * denom2
        * ((alfa * dzetadt)
            + (zeta2 * dbetadt + 2.0 * dzetadt * beta * zeta)
            + (zeta3 * dgamadt + 3.0 * dzetadt * gama * zeta2));
    let ddenomdd = -ONE_THIRD
        * denom2
        * ((alfa * dzetadd)
            + (zeta2 * dbetadd + 2.0 * dzetadd * beta * zeta)
            + (zeta3 * dgamadd + 3.0 * dzetadd * gama * zeta2));

    let gamtild = gam * denom;
    let dgamtilddt = ddenomdt * gam + dgamdt * denom;
    let dgamtilddd = ddenomdd * gam + dgamdd * denom;

    // `screen_chugunov.f90:254-305`
    let c_a3 = 3.0_f64.sqrt() - C_A1 / C_A2.sqrt();
    let gamtild2 = gamtild * gamtild;

    let a_big = gamtild * gamtild.sqrt();
    let da_dt = 1.5 * (a_big / gamtild) * dgamtilddt;
    let da_dd = 1.5 * (a_big / gamtild) * dgamtilddd;

    let b_big = C_A1 / (C_A2 + gamtild).sqrt();
    let db_dt = -0.5 * b_big / (C_A2 + gamtild) * dgamtilddt;
    let db_dd = -0.5 * b_big / (C_A2 + gamtild) * dgamtilddd;

    let c_big = c_a3 / (1.0 + gamtild);
    let dc_dt = -c_big / (1.0 + gamtild) * dgamtilddt;
    let dc_dd = -c_big / (1.0 + gamtild) * dgamtilddd;

    let t1 = a_big * (b_big + c_big);
    let dt1dt = a_big * (db_dt + dc_dt) + da_dt * (b_big + c_big);
    let dt1dd = a_big * (db_dd + dc_dd) + da_dd * (b_big + c_big);

    let denom_t2 = C_B2 + gamtild;
    let ddenom_t2_dt = dgamtilddt;
    let ddenom_t2_dd = dgamtilddd;
    let t2 = C_B1 * gamtild2 / denom_t2;
    let dt2dt = ((denom_t2 * C_B1 * 2.0 * gamtild * dgamtilddt)
        - (C_B1 * gamtild2 * ddenom_t2_dt))
        / (denom_t2 * denom_t2);
    let dt2dd = ((denom_t2 * C_B1 * 2.0 * gamtild * dgamtilddd)
        - (C_B1 * gamtild2 * ddenom_t2_dd))
        / (denom_t2 * denom_t2);

    let denom_t3 = C_B4 + gamtild2;
    let ddenom_t3_dt = 2.0 * gamtild * dgamtilddt;
    let ddenom_t3_dd = 2.0 * gamtild * dgamtilddd;
    let t3 = C_B3 * gamtild2 / denom_t3;
    let dt3dt = ((denom_t3 * C_B3 * 2.0 * gamtild * dgamtilddt)
        - (C_B3 * gamtild2 * ddenom_t3_dt))
        / (denom_t3 * denom_t3);
    let dt3dd = ((denom_t3 * C_B3 * 2.0 * gamtild * dgamtilddd)
        - (C_B3 * gamtild2 * ddenom_t3_dd))
        / (denom_t3 * denom_t3);

    let mut h0 = t1 + t2 + t3;
    let mut dh0dt = dt1dt + dt2dt + dt3dt;
    let mut dh0dd = dt1dd + dt2dd + dt3dd;

    if h0 > 300.0 {
        h0 = 300.0;
        dh0dt = 0.0;
        dh0dd = 0.0;
    }
    let scor = exp(h0);
    ScreenEval {
        scor,
        dscor_dt: dh0dt * scor,
        dscor_dd: dh0dd * scor,
    }
}
