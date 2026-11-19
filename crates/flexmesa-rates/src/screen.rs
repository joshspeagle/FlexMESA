// SPDX-License-Identifier: LGPL-3.0-or-later
//! Plasma electrostatic screening (`no_screening`, `salpeter_screening`,
//! `extended_screening`, and `chugunov_screening`) ported from
//! `rates/private/screen.f90` and `rates/private/screen5.f90`.

use flexmesa_const::{ONE_THIRD, TWO_13};

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

/// Evaluates Salpeter (1954) weak plasma screening (`rates_salpeter_screening`
/// in `rates/public/rates_lib.f90:705-727`).
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
    let dlnf_dd = lnf / (2.0 * den);
    let dlnf_dt = (-lnf * 3.0) / (2.0 * temp);
    let scor = flexmesa_math::exp(lnf);
    let dscor_dd = scor * dlnf_dd;
    let dscor_dt = scor * dlnf_dt;
    ScreenEval {
        scor,
        dscor_dt,
        dscor_dd,
    }
}

/// Evaluates Graboske et al. (1973), Alastuey & Jancovici (1978), and Itoh et
/// al. (1979) weak-intermediate-strong plasma screening (`fxt_screen5` in
/// `rates/private/screen.f90:46-99` and `rates/private/screen5.f90:61-386`).
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
    const CO2: f64 = ONE_THIRD * 4.248_719e3;
    const FACT: f64 = TWO_13;

    let ytot = 1.0 / abar;
    let rr = den * ytot;
    let tempi = 1.0 / temp;
    let dtempidt = -tempi * tempi;
    let deni = 1.0 / den;
    let pp = (rr * tempi * (z2bar + zbar)).sqrt();
    let qq_p = (0.5 / pp) * (z2bar + zbar);
    let dppdt = qq_p * rr * dtempidt;
    let dppdd = qq_p * ytot * tempi;
    let qlam0z = 1.88e8 * tempi * pp;
    let qlam0zdt = 1.88e8 * (dtempidt * pp + tempi * dppdt);
    let qlam0zdd = 1.88e8 * tempi * dppdd;
    let taufac = CO2 * flexmesa_math::pow(tempi, ONE_THIRD);
    let taufacdt = -ONE_THIRD * taufac * tempi;
    let xni = flexmesa_math::pow(rr * zbar, ONE_THIRD);
    let dxnidd = ONE_THIRD * xni * deni;
    let aa = 2.27493e5 * tempi * xni;
    let daadt = 2.27493e5 * dtempidt * xni;
    let daadd = 2.27493e5 * tempi * dxnidd;

    let z12 = z1 + z2;
    let zs13 = flexmesa_math::pow(z12, ONE_THIRD);
    let zs13inv = 1.0 / zs13;
    let zhat = flexmesa_math::pow(z12, 5.0 / 3.0)
        - flexmesa_math::pow(z1, 5.0 / 3.0)
        - flexmesa_math::pow(z2, 5.0 / 3.0);
    let zhat2 = flexmesa_math::pow(z12, 5.0 / 12.0)
        - flexmesa_math::pow(z1, 5.0 / 12.0)
        - flexmesa_math::pow(z2, 5.0 / 12.0);
    let lzav = (5.0 / 3.0) * flexmesa_math::log((z1 * z2 / z12).max(1.0e-99));
    let aznut =
        flexmesa_math::pow(z1 * z1 * z2 * z2 * a1 * a2 / (a1 + a2), ONE_THIRD);

    let bb = z1 * z2;
    let mut gamp = aa;
    let mut gampdt = daadt;
    let mut gampdd = daadd;
    let qq = FACT * bb * zs13inv;
    let mut gamef = qq * gamp;
    let mut gamefdt = qq * gampdt;
    let mut gamefdd = qq * gampdd;
    let tau12 = taufac * aznut;
    let tau12dt = taufacdt * aznut;
    let qq_tau = 1.0 / tau12;
    let mut alf12 = gamef * qq_tau;
    let mut alf12dt = (gamefdt - alf12 * tau12dt) * qq_tau;
    let mut alf12dd = gamefdd * qq_tau;

    if alf12 > 1.6 {
        alf12 = 1.6;
        alf12dt = 0.0;
        alf12dd = 0.0;
        gamef = 1.6 * tau12;
        gamefdt = 1.6 * tau12dt;
        gamefdd = 0.0;
        let qq_inv = zs13 / (FACT * bb);
        gamp = gamef * qq_inv;
        gampdt = gamefdt * qq_inv;
        gampdd = 0.0;
    }

    let h12w = bb * qlam0z;
    let dh12wdt = bb * qlam0zdt;
    let dh12wdd = bb * qlam0zdd;
    let (mut h12, mut dh12dt, mut dh12dd) = (h12w, dh12wdt, dh12wdd);

    if gamef > 0.3 {
        let gamp14 = flexmesa_math::pow(gamp, 0.25);
        let rr_g = 1.0 / gamp;
        let qq_g = 0.25 * gamp14 * rr_g;
        let gamp14dt = qq_g * gampdt;
        let gamp14dd = qq_g * gampdd;

        let cc = 0.896434 * gamp * zhat
            - 3.44740 * gamp14 * zhat2
            - 0.5551 * (flexmesa_math::log(gamp) + lzav)
            - 2.996;
        let dccdt = 0.896434 * gampdt * zhat
            - 3.44740 * gamp14dt * zhat2
            - 0.5551 * rr_g * gampdt;
        let dccdd = 0.896434 * gampdd * zhat
            - 3.44740 * gamp14dd * zhat2
            - 0.5551 * rr_g * gampdd;

        let a3 = alf12 * alf12 * alf12;
        let da3 = 3.0 * alf12 * alf12;

        let qq_s = 0.014 + 0.0128 * alf12;
        let dqqdt = 0.0128 * alf12dt;
        let dqqdd = 0.0128 * alf12dd;

        let rr_s = (5.0 / 32.0) - alf12 * qq_s;
        let drrdt = -(alf12dt * qq_s + alf12 * dqqdt);
        let drrdd = -(alf12dd * qq_s + alf12 * dqqdd);

        let ss = tau12 * rr_s;
        let dssdt = tau12dt * rr_s + tau12 * drrdt;
        let dssdd = tau12 * drrdd;

        let tt = -0.0098 + 0.0048 * alf12;
        let dttdt = 0.0048 * alf12dt;
        let dttdd = 0.0048 * alf12dd;

        let uu = 0.0055 + alf12 * tt;
        let duudt = alf12dt * tt + alf12 * dttdt;
        let duudd = alf12dd * tt + alf12 * dttdd;

        let vv = gamef * alf12 * uu;
        let dvvdt = gamefdt * alf12 * uu
            + gamef * alf12dt * uu
            + gamef * alf12 * duudt;
        let dvvdd = gamefdd * alf12 * uu
            + gamef * alf12dd * uu
            + gamef * alf12 * duudd;

        h12 = cc - a3 * (ss + vv);
        let rr_h = da3 * (ss + vv);
        dh12dt = dccdt - rr_h * alf12dt - a3 * (dssdt + dvvdt);
        dh12dd = dccdd - rr_h * alf12dd - a3 * (dssdd + dvvdd);

        let rr_lg = 1.0 - 0.0562 * a3;
        let ss_lg = -0.0562 * da3;
        let drr_lg_dt = ss_lg * alf12dt;
        let drr_lg_dd = ss_lg * alf12dd;
        let (xlgfac, dxlgfacdt, dxlgfacdd) = if rr_lg >= 0.77 {
            (rr_lg, drr_lg_dt, drr_lg_dd)
        } else {
            (0.77, 0.0, 0.0)
        };

        h12 += flexmesa_math::log(xlgfac);
        let rr_xl = 1.0 / xlgfac;
        dh12dt += rr_xl * dxlgfacdt;
        dh12dd += rr_xl * dxlgfacdd;

        if gamef <= 0.8 {
            let dgamma = 1.0 / (0.8 - 0.3);
            let rr_wt = dgamma * (0.8 - gamef);
            let drr_wt_dt = -dgamma * gamefdt;
            let drr_wt_dd = -dgamma * gamefdd;
            let ss_wt = dgamma * (gamef - 0.3);
            let dss_wt_dt = dgamma * gamefdt;
            let dss_wt_dd = dgamma * gamefdd;
            let vv_s = h12;
            h12 = h12w * rr_wt + vv_s * ss_wt;
            dh12dt = dh12wdt * rr_wt
                + h12w * drr_wt_dt
                + dh12dt * ss_wt
                + vv_s * dss_wt_dt;
            dh12dd = dh12wdd * rr_wt
                + h12w * drr_wt_dd
                + dh12dd * ss_wt
                + vv_s * dss_wt_dd;
        }
    }

    h12 = h12.clamp(0.0, 300.0);
    let scor = flexmesa_math::exp(h12);
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
