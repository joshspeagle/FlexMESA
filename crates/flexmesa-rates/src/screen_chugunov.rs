// SPDX-License-Identifier: LGPL-3.0-or-later
//! Chugunov, DeWitt & Yakovlev (2007, PhRvD 76, 025028) one-component plasma
//! electrostatic screening ported from `rates/private/screen_chugunov.f90`.

use flexmesa_const::{AMU, BOLTZM, HBAR, ONE_THIRD, PI, QE};

use crate::types::ScreenEval;

/// Evaluates Chugunov, DeWitt & Yakovlev (2007) plasma screening (`chugunov_screening`)
/// with analytical $(T, \rho)$ partial derivatives (`screen_chugunov.f90`).
#[must_use]
#[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
pub fn screen_chugunov(
    temp: f64,
    den: f64,
    zbar: f64,
    abar: f64,
    _a1: f64,
    z1: f64,
    _a2: f64,
    z2: f64,
) -> ScreenEval {
    let mav = abar * AMU;
    let ntot = den / mav;
    let ae = (3.0 / (4.0 * PI * ntot * zbar)).powf(ONE_THIRD);
    let r1 = ae * f64::from(z1 as i32).powf(ONE_THIRD);
    let r2 = ae * f64::from(z2 as i32).powf(ONE_THIRD);
    let rav = 0.5 * (r1 + r2);
    let dravdd = -rav / (3.0 * den);

    let w = (4.0 * PI * z1 * z2 * QE * QE * ntot / mav).sqrt();
    let dwdd = 0.5 * w / den;
    let tp = HBAR * w / BOLTZM;
    let dtpdd = (HBAR / BOLTZM) * dwdd;
    let tn = temp / tp;
    let dtndt = 1.0 / tp;
    let dtndd = -tn * dtpdd / tp;

    let (tk, dtkdt, dtkdd, denom) = if tn > 0.2 {
        (temp, 1.0, 0.0, tp * temp * dtpdd / temp.powi(3))
    } else if tn <= 0.1 {
        (0.1 * tp, 0.0, 0.1 * dtpdd, 0.0)
    } else {
        let alfa = 0.5 * (1.0 - (PI * (tn - 0.1) / 0.1).cos());
        let dalfadtn = 0.5 * PI * (PI * (tn - 0.1) / 0.1).sin() / 0.1;
        let dalfadt = dalfadtn * dtndt;
        let dalfadd = dalfadtn * dtndd;
        let beta = 1.0 - alfa;
        let dbetadt = -dalfadt;
        let dbetadd = -dalfadd;
        let tk_v = (beta * 0.1 + alfa * 0.2) * tp;
        let dtk_dt = (dbetadt * 0.1 + dalfadt * 0.2) * tp;
        let dtk_dd = (dbetadd * 0.1 + dalfadd * 0.2) * tp
            + (beta * 0.1 + alfa * 0.2) * dtpdd;
        let den_v = tp * (tk_v * dtpdd - tp * dtk_dd) / tk_v.powi(3);
        (tk_v, dtk_dt, dtk_dd, den_v)
    };

    let u_zeta = 4.0 * tp * tp / (3.0 * PI * PI * tk * tk);
    let zeta = u_zeta.powf(ONE_THIRD);
    let dzetadt = -2.0 * zeta * dtkdt / (3.0 * tk);
    let dzetadd = (zeta / (3.0 * u_zeta)) * (8.0 / (3.0 * PI * PI)) * denom;

    let mut gam = z1 * z2 * QE * QE / (rav * BOLTZM * tk);
    let mut dgamdt = -gam * dtkdt / tk;
    let mut dgamdd = -gam * (tk * dravdd + rav * dtkdd) / (rav * tk);

    if gam >= 600.0 {
        gam = 600.0;
        dgamdt = 0.0;
        dgamdd = 0.0;
    } else if gam >= 590.0 {
        let alfa = 0.5 * (1.0 - (PI * (gam - 590.0) / 10.0).cos());
        let dalfadgam = 0.5 * PI * (PI * (gam - 590.0) / 10.0).sin() / 10.0;
        let dalfadt = dalfadgam * dgamdt;
        let dalfadd = dalfadgam * dgamdd;
        let beta = 1.0 - alfa;
        let dbetadt = -dalfadt;
        let dbetadd = -dalfadd;
        let gam_orig = gam;
        let dgamdt_orig = dgamdt;
        let dgamdd_orig = dgamdd;
        gam = beta * gam_orig + alfa * 600.0;
        dgamdt = dbetadt * gam_orig + beta * dgamdt_orig + dalfadt * 600.0;
        dgamdd = dbetadd * gam_orig + beta * dgamdd_orig + dalfadd * 600.0;
    }

    let a_z = 0.022;
    let b_z = 0.41 - 0.6 / gam;
    let db_z_dt = 0.6 * dgamdt / (gam * gam);
    let db_z_dd = 0.6 * dgamdd / (gam * gam);
    let g_z = 0.06 + 2.2 / gam;
    let dg_z_dt = -2.2 * dgamdt / (gam * gam);
    let dg_z_dd = -2.2 * dgamdd / (gam * gam);

    let u = 1.0 + a_z * zeta + b_z * zeta * zeta + g_z * zeta.powi(3);
    let dudt = a_z * dzetadt
        + db_z_dt * zeta * zeta
        + 2.0 * b_z * zeta * dzetadt
        + dg_z_dt * zeta.powi(3)
        + 3.0 * g_z * zeta * zeta * dzetadt;
    let dudd = a_z * dzetadd
        + db_z_dd * zeta * zeta
        + 2.0 * b_z * zeta * dzetadd
        + dg_z_dd * zeta.powi(3)
        + 3.0 * g_z * zeta * zeta * dzetadd;

    let u13 = u.powf(ONE_THIRD);
    let tgam = gam / u13;
    let dtgamdt = (dgamdt - ONE_THIRD * gam * dudt / u) / u13;
    let dtgamdd = (dgamdd - ONE_THIRD * gam * dudd / u) / u13;

    let ca1 = 2.7822_f64;
    let ca2 = 98.34_f64;
    let ca3 = 3.0_f64.sqrt() - ca1 / ca2.sqrt();
    let cb1 = -1.7476_f64;
    let cb2 = 66.07_f64;
    let cb3 = 1.12_f64;
    let cb4 = 65.0_f64;

    let t1_g = ca1 / (ca2 + tgam).sqrt();
    let dt1g_dt = -0.5 * ca1 * dtgamdt / (ca2 + tgam).powf(1.5);
    let dt1g_dd = -0.5 * ca1 * dtgamdd / (ca2 + tgam).powf(1.5);
    let t2_g = ca3 / (1.0 + tgam);
    let dt2g_dt = -ca3 * dtgamdt / ((1.0 + tgam) * (1.0 + tgam));
    let dt2g_dd = -ca3 * dtgamdd / ((1.0 + tgam) * (1.0 + tgam));

    let tg32 = tgam.powf(1.5);
    let dtg32_dt = 1.5 * tgam.sqrt() * dtgamdt;
    let dtg32_dd = 1.5 * tgam.sqrt() * dtgamdd;
    let t3_g = tg32 * (t1_g + t2_g);
    let dt3g_dt = dtg32_dt * (t1_g + t2_g) + tg32 * (dt1g_dt + dt2g_dt);
    let dt3g_dd = dtg32_dd * (t1_g + t2_g) + tg32 * (dt1g_dd + dt2g_dd);

    let tg2 = tgam * tgam;
    let dtg2_dt = 2.0 * tgam * dtgamdt;
    let dtg2_dd = 2.0 * tgam * dtgamdd;

    let t4_g = cb1 * tg2 / (cb2 + tgam);
    let dt4g_dt = cb1 * (dtg2_dt - tg2 * dtgamdt / (cb2 + tgam)) / (cb2 + tgam);
    let dt4g_dd = cb1 * (dtg2_dd - tg2 * dtgamdd / (cb2 + tgam)) / (cb2 + tgam);

    let t5_g = cb3 * tg2 / (cb4 + tg2);
    let dt5g_dt = cb3 * (dtg2_dt - tg2 * dtg2_dt / (cb4 + tg2)) / (cb4 + tg2);
    let dt5g_dd = cb3 * (dtg2_dd - tg2 * dtg2_dd / (cb4 + tg2)) / (cb4 + tg2);

    let mut h0 = t3_g + t4_g + t5_g;
    let mut dh0dt = dt3g_dt + dt4g_dt + dt5g_dt;
    let mut dh0dd = dt3g_dd + dt4g_dd + dt5g_dd;

    if h0 > 300.0 {
        h0 = 300.0;
        dh0dt = 0.0;
        dh0dd = 0.0;
    }
    let scor = h0.exp();
    ScreenEval {
        scor,
        dscor_dt: scor * dh0dt,
        dscor_dd: scor * dh0dd,
    }
}
