// SPDX-License-Identifier: LGPL-3.0-or-later
//! Plasma electrostatic screening (`no_screening`, `salpeter_screening`,
//! `extended_screening`, and `chugunov_screening`) ported from
//! `rates/private/screen.f90` and `rates/private/screen5.f90`.

use flexmesa_const::ONE_THIRD;

use crate::screen_chugunov::screen_chugunov;
use crate::types::{ScreenEval, ScreeningMode};

/// Evaluates plasma electrostatic screening for a reacting pair `(a1, z1)` +
/// `(a2, z2)` at `(temp, den, zbar, abar, z2bar)` under the specified
/// [`ScreeningMode`].
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn screen_pair(
    mode: ScreeningMode,
    temp: f64,
    den: f64,
    zbar: f64,
    abar: f64,
    z2bar: f64,
    a1: f64,
    z1: f64,
    a2: f64,
    z2: f64,
) -> ScreenEval {
    if mode == ScreeningMode::NoScreening
        || zbar == 0.0
        || z1 <= 0.0
        || z2 <= 0.0
    {
        return ScreenEval::UNSCREENED;
    }
    match mode {
        ScreeningMode::NoScreening => ScreenEval::UNSCREENED,
        ScreeningMode::Salpeter => {
            screen_salpeter(temp, den, zbar, abar, z2bar, z1, z2)
        }
        ScreeningMode::Extended => {
            screen_extended(temp, den, zbar, abar, z2bar, a1, z1, a2, z2)
        }
        ScreeningMode::Chugunov => {
            screen_chugunov(temp, den, zbar, abar, a1, z1, a2, z2)
        }
    }
}

/// Evaluates Salpeter (1954) weak plasma screening (`salpeter_screening` in
/// `rates/private/screen.f90`).
#[must_use]
pub fn screen_salpeter(
    temp: f64,
    den: f64,
    zbar: f64,
    abar: f64,
    z2bar: f64,
    z1: f64,
    z2: f64,
) -> ScreenEval {
    let zeta = (z2bar + zbar) / abar;
    let lnf = 1.88e8 * z1 * z2 * (den * zeta / (temp * temp * temp)).sqrt();
    let scor = lnf.exp();
    let dscor_dt = -1.5 * lnf / temp * scor;
    let dscor_dd = 0.5 * lnf / den * scor;
    ScreenEval {
        scor,
        dscor_dt,
        dscor_dd,
    }
}

/// Evaluates Graboske et al. (1973), Alastuey & Jancovici (1978), and Itoh et
/// al. (1979) weak-intermediate-strong plasma screening (`fxt_screen5` in
/// `rates/private/screen5.f90`).
#[must_use]
#[allow(clippy::too_many_arguments, clippy::float_cmp)]
pub fn screen_extended(
    temp: f64,
    den: f64,
    zbar: f64,
    abar: f64,
    z2bar: f64,
    a1: f64,
    z1: f64,
    a2: f64,
    z2: f64,
) -> ScreenEval {
    let ytot = 1.0 / abar;
    let rr = den * ytot;
    let tempi = 1.0 / temp;
    let dtempidt = -tempi * tempi;
    let deni = 1.0 / den;
    let pp = (rr * tempi * (z2bar + zbar)).sqrt();
    let dppdt = 0.5 * rr * dtempidt * (z2bar + zbar) / pp;
    let dppdd = 0.5 * ytot * tempi * (z2bar + zbar) / pp;
    let qlam0z = 1.88e8 * tempi * pp;
    let qlam0zdt = 1.88e8 * (dtempidt * pp + tempi * dppdt);
    let qlam0zdd = 1.88e8 * tempi * dppdd;
    let taufac = 4.248719e3 / 3.0 * tempi.powf(ONE_THIRD);
    let taufacdt = -ONE_THIRD * taufac * tempi;
    let xni = (rr * zbar).powf(ONE_THIRD);
    let dxnidd = ONE_THIRD * xni * deni;
    let aa = 2.27493e5 * tempi * xni;
    let daadt = 2.27493e5 * dtempidt * xni;
    let daadd = 2.27493e5 * tempi * dxnidd;

    let z12 = z1 + z2;
    let zs13 = z12.powf(ONE_THIRD);
    let zs13inv = 1.0 / zs13;
    let zhat = z12.powf(5.0 / 3.0) - z1.powf(5.0 / 3.0) - z2.powf(5.0 / 3.0);
    let zhat2 =
        z12.powf(5.0 / 12.0) - z1.powf(5.0 / 12.0) - z2.powf(5.0 / 12.0);
    let lzav = (5.0 / 3.0) * (z1 * z2 / z12).max(1.0e-99).ln();
    let aznut = (z1 * z1 * z2 * z2 * a1 * a2 / (a1 + a2)).powf(ONE_THIRD);

    let bb = z1 * z2;
    let mut gamp = aa;
    let mut gampdt = daadt;
    let mut gampdd = daadd;
    let qq = 2.0_f64.powf(ONE_THIRD) * bb * zs13inv;
    let mut gamef = qq * gamp;
    let mut gamefdt = qq * gampdt;
    let mut gamefdd = qq * gampdd;
    let tau12 = taufac * aznut;
    let tau12dt = taufacdt * aznut;
    let mut alf12 = gamef / tau12;
    let mut alf12dt = (gamefdt - alf12 * tau12dt) / tau12;
    let mut alf12dd = gamefdd / tau12;

    if alf12 > 1.6 {
        alf12 = 1.6;
        alf12dt = 0.0;
        alf12dd = 0.0;
        gamef = 1.6 * tau12;
        gamefdt = 1.6 * tau12dt;
        gamefdd = 0.0;
        let qq_inv = zs13 / (2.0_f64.powf(ONE_THIRD) * bb);
        gamp = gamef * qq_inv;
        gampdt = gamefdt * qq_inv;
        gampdd = 0.0;
    }

    let h12w = bb * qlam0z;
    let dh12wdt = bb * qlam0zdt;
    let dh12wdd = bb * qlam0zdd;
    let (mut h12, mut dh12dt, mut dh12dd) = (h12w, dh12wdt, dh12wdd);

    if gamef > 0.3 {
        let gamp14 = gamp.powf(0.25);
        let cc = 0.896434 * gamp * zhat
            - 3.44740 * gamp14 * zhat2
            - 0.5551 * (gamp.ln() + lzav)
            - 2.996;
        let dccdgamp = 0.896434 * zhat
            - 0.25 * 3.44740 * gamp14 * zhat2 / gamp
            - 0.5551 / gamp;
        let dccdt = dccdgamp * gampdt;
        let dccdd = dccdgamp * gampdd;

        let a3 = alf12 * alf12 * alf12;
        let da3 = 3.0 * alf12 * alf12;
        let qq_s = 0.014 + 0.0128 * alf12;
        let dqqdalf = 0.0128;
        let rr_s = (5.0 / 32.0) - alf12 * qq_s;
        let drrdalf = -(qq_s + alf12 * dqqdalf);
        let ss = tau12 * rr_s;
        let dssdt = tau12dt * rr_s + tau12 * drrdalf * alf12dt;
        let dssdd = tau12 * drrdalf * alf12dd;

        let tt = -0.0098 + 0.0048 * alf12;
        let dttdalf = 0.0048;
        let uu = 0.0055 + alf12 * tt;
        let duudalf = tt + alf12 * dttdalf;
        let vv = gamef * alf12 * uu;
        let dvvdt = (gamefdt * alf12 + gamef * alf12dt) * uu
            + gamef * alf12 * duudalf * alf12dt;
        let dvvdd = (gamefdd * alf12 + gamef * alf12dd) * uu
            + gamef * alf12 * duudalf * alf12dd;

        let ww = ss + vv;
        let dwwdt = dssdt + dvvdt;
        let dwwdd = dssdd + dvvdd;
        let xx = a3 * ww;
        let dxxdt = da3 * alf12dt * ww + a3 * dwwdt;
        let dxxdd = da3 * alf12dd * ww + a3 * dwwdd;

        let xlgfac = (1.0 - 0.0562 * a3).max(0.77);
        let (dxlgfacdt, dxlgfacdd) = if xlgfac > 0.77 {
            let f = -0.0562 * da3 / xlgfac;
            (f * alf12dt, f * alf12dd)
        } else {
            (0.0, 0.0)
        };

        h12 = cc - xx + xlgfac.ln();
        dh12dt = dccdt - dxxdt + dxlgfacdt;
        dh12dd = dccdd - dxxdd + dxlgfacdd;

        if gamef <= 0.8 {
            let dgamef = 2.0;
            let rr_wt = dgamef * (0.8 - gamef);
            let drrdt = -dgamef * gamefdt;
            let drrdd = -dgamef * gamefdd;
            let ss_wt = dgamef * (gamef - 0.3);
            let dss_dt = dgamef * gamefdt;
            let dss_dd = dgamef * gamefdd;
            let vv_s = h12;
            h12 = h12w * rr_wt + vv_s * ss_wt;
            dh12dt =
                dh12wdt * rr_wt + h12w * drrdt + dh12dt * ss_wt + vv_s * dss_dt;
            dh12dd =
                dh12wdd * rr_wt + h12w * drrdd + dh12dd * ss_wt + vv_s * dss_dd;
        }
    }

    h12 = h12.clamp(0.0, 300.0);
    let scor = h12.exp();
    let (dscor_dt, dscor_dd) = if h12 == 300.0 {
        (0.0, 0.0)
    } else {
        (scor * dh12dt, scor * dh12dd)
    };
    ScreenEval {
        scor,
        dscor_dt,
        dscor_dd,
    }
}
