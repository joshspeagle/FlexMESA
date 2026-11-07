// SPDX-License-Identifier: LGPL-3.0-or-later
//! Reaction-network electrostatic screening (`net_screen.f90` lines 1–529).

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{LN10, ONE_THIRD, PI, TWO_13};
use flexmesa_math::{cospi, sinpi};
use flexmesa_rates::{
    screen_chugunov, screen_salpeter, ScreenEval, ScreeningMode,
};

use crate::screen_combo::set_combo_screen_rates;
use crate::types::{
    NetError, NetGeneralInfo, ScreenPairParams, ICO55, ICO57, IH2,
};

/// Initializes cached `(A, Z)` screening parameters for every reaction in `g`
/// (`make_screening_tables` in `net/private/net_screen.f90`).
pub fn init_screening_tables(g: &mut NetGeneralInfo) {
    let nuclides = NuclideCatalog::builtin_ref();
    for i in 0..g.num_reactions {
        let sinfo = g.reactions[i].screening_isos;
        if sinfo[2] > 0 {
            init_screen_triple(
                nuclides,
                &mut g.screen_params[i],
                sinfo[0] as usize,
                sinfo[1] as usize,
                sinfo[2] as usize,
            );
        } else if sinfo[1] > 0 {
            let e1 = nuclides.get_by_id(sinfo[0] as usize).unwrap();
            let e2 = nuclides.get_by_id(sinfo[1] as usize).unwrap();
            g.screen_params[i] = init_az_info(
                e1.z_plus_n as f64,
                e1.z as f64,
                e2.z_plus_n as f64,
                e2.z as f64,
            );
        }
    }
}

fn init_screen_triple(
    nuclides: &NuclideCatalog,
    slot: &mut ScreenPairParams,
    mut i1: usize,
    mut i2: usize,
    mut i3: usize,
) {
    let mut e1 = nuclides.get_by_id(i1).unwrap();
    let mut e2 = nuclides.get_by_id(i2).unwrap();
    let mut e3 = nuclides.get_by_id(i3).unwrap();
    if e2.z == 0 {
        if e1.z == 0 {
            return;
        }
        std::mem::swap(&mut i1, &mut i2);
        std::mem::swap(&mut e1, &mut e2);
    }
    if e3.z == 0 {
        std::mem::swap(&mut i1, &mut i3);
        std::mem::swap(&mut e1, &mut e3);
    }
    let a1 = e1.z_plus_n as f64;
    let z1 = e1.z as f64;
    let a2 = e2.z_plus_n as f64;
    let z2 = e2.z as f64;
    let a3 = e3.z_plus_n as f64;
    let z3 = e3.z as f64;
    *slot = init_az_info(a2, z2, a3, z3);
    if z1 == 0.0 {
        return;
    }
    *slot = init_az_info(a1, z1, a2 + a3, z2 + z3);
}

#[must_use]
fn init_az_info(a1: f64, z1: f64, a2: f64, z2: f64) -> ScreenPairParams {
    if z1 <= 0.0 || z2 <= 0.0 {
        return ScreenPairParams::default();
    }
    let z12 = z1 + z2;
    let zs13 = z12.powf(ONE_THIRD);
    let zs13inv = 1.0 / zs13;
    let zhat = z12.powf(5.0 / 3.0) - z1.powf(5.0 / 3.0) - z2.powf(5.0 / 3.0);
    let zhat2 =
        z12.powf(5.0 / 12.0) - z1.powf(5.0 / 12.0) - z2.powf(5.0 / 12.0);
    let lzav = (5.0 / 3.0) * (z1 * z2 / z12).max(1.0e-99).ln();
    let aznut = (z1 * z1 * z2 * z2 * a1 * a2 / (a1 + a2)).powf(ONE_THIRD);
    ScreenPairParams {
        zs13,
        zhat,
        zhat2,
        lzav,
        aznut,
        zs13inv,
    }
}

/// Applies plasma electrostatic screening and compound branch weighting across
/// all network reactions (`screen_net` in `net/private/net_screen.f90`).
#[allow(clippy::too_many_arguments)]
pub fn screen_net(
    g: &NetGeneralInfo,
    y: &[f64],
    temp: f64,
    den: f64,
    log_t: f64,
    _log_rho: f64,
    rate_raw: &[f64],
    rate_raw_dt: &[f64],
    rate_raw_drho: &[f64],
    rate_screened: &mut [f64],
    rate_screened_dt: &mut [f64],
    rate_screened_drho: &mut [f64],
    screening_mode: ScreeningMode,
    zbar: f64,
    abar: f64,
    z2bar: f64,
) -> Result<(), NetError> {
    let nuclides = NuclideCatalog::builtin_ref();
    for i in 0..g.num_reactions {
        let sinfo = g.reactions[i].screening_isos;
        if sinfo[2] > 0 {
            let sc = eval_screen_triple(
                g,
                nuclides,
                i,
                sinfo[0] as usize,
                sinfo[1] as usize,
                sinfo[2] as usize,
                temp,
                den,
                log_t,
                zbar,
                abar,
                z2bar,
                screening_mode,
            );
            set_rate_screening(
                i,
                sc,
                rate_raw,
                rate_raw_dt,
                rate_raw_drho,
                rate_screened,
                rate_screened_dt,
                rate_screened_drho,
            );
        } else if sinfo[1] > 0 {
            let e1 = nuclides.get_by_id(sinfo[0] as usize).unwrap();
            let e2 = nuclides.get_by_id(sinfo[1] as usize).unwrap();
            let sc = eval_screening_pair(
                g,
                i,
                sinfo[0] as usize,
                e1.z_plus_n as f64,
                e1.z as f64,
                sinfo[1] as usize,
                e2.z_plus_n as f64,
                e2.z as f64,
                temp,
                den,
                log_t,
                zbar,
                abar,
                z2bar,
                screening_mode,
            );
            set_rate_screening(
                i,
                sc,
                rate_raw,
                rate_raw_dt,
                rate_raw_drho,
                rate_screened,
                rate_screened_dt,
                rate_screened_drho,
            );
        } else {
            rate_screened[i] = rate_raw[i];
            rate_screened_dt[i] = rate_raw_dt[i];
            rate_screened_drho[i] = rate_raw_drho[i];
        }
    }

    set_combo_screen_rates(
        g,
        y,
        rate_raw,
        rate_raw_dt,
        rate_raw_drho,
        rate_screened,
        rate_screened_dt,
        rate_screened_drho,
    )?;

    if log_t < g.log_tcut_lim || log_t <= g.log_tcut_lo {
        if log_t <= g.log_tcut_lo {
            for i in 0..g.num_reactions {
                if g.weak_reaction_index[i] > 0 {
                    continue;
                }
                rate_screened[i] = 0.0;
                rate_screened_dt[i] = 0.0;
                rate_screened_drho[i] = 0.0;
            }
        } else {
            let mut tfactor =
                (log_t - g.log_tcut_lo) / (g.log_tcut_lim - g.log_tcut_lo);
            tfactor = 0.5 * (1.0 - cospi(tfactor * tfactor));
            let dtfactordt = 0.5 * PI * sinpi(tfactor * tfactor) * 2.0
                / ((g.log_tcut_lim - g.log_tcut_lo) * temp * LN10);
            for i in 0..g.num_reactions {
                if g.weak_reaction_index[i] > 0 {
                    continue;
                }
                rate_screened_dt[i] = tfactor * rate_screened_dt[i]
                    + dtfactordt * rate_screened[i];
                rate_screened_drho[i] *= tfactor;
                rate_screened[i] *= tfactor;
            }
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn set_rate_screening(
    i: usize,
    sc: ScreenEval,
    rate_raw: &[f64],
    rate_raw_dt: &[f64],
    rate_raw_drho: &[f64],
    rate_screened: &mut [f64],
    rate_screened_dt: &mut [f64],
    rate_screened_drho: &mut [f64],
) {
    rate_screened[i] = rate_raw[i] * sc.scor;
    rate_screened_dt[i] = rate_raw_dt[i] * sc.scor + rate_raw[i] * sc.dscor_dt;
    rate_screened_drho[i] =
        rate_raw_drho[i] * sc.scor + rate_raw[i] * sc.dscor_dd;
}

#[allow(clippy::too_many_arguments)]
fn eval_screen_triple(
    g: &NetGeneralInfo,
    nuclides: &NuclideCatalog,
    jscr: usize,
    mut i1: usize,
    mut i2: usize,
    mut i3: usize,
    temp: f64,
    den: f64,
    log_t: f64,
    zbar: f64,
    abar: f64,
    z2bar: f64,
    screening_mode: ScreeningMode,
) -> ScreenEval {
    let mut e1 = nuclides.get_by_id(i1).unwrap();
    let mut e2 = nuclides.get_by_id(i2).unwrap();
    let mut e3 = nuclides.get_by_id(i3).unwrap();
    if e2.z == 0 {
        if e1.z == 0 {
            return ScreenEval::UNSCREENED;
        }
        std::mem::swap(&mut i1, &mut i2);
        std::mem::swap(&mut e1, &mut e2);
    }
    if e3.z == 0 {
        std::mem::swap(&mut i1, &mut i3);
        std::mem::swap(&mut e1, &mut e3);
    }
    let a1 = e1.z_plus_n as f64;
    let z1 = e1.z as f64;
    let a2 = e2.z_plus_n as f64;
    let z2 = e2.z as f64;
    let a3 = e3.z_plus_n as f64;
    let z3 = e3.z as f64;

    let sc2 = eval_screening_pair(
        g,
        jscr,
        i2,
        a2,
        z2,
        i3,
        a3,
        z3,
        temp,
        den,
        log_t,
        zbar,
        abar,
        z2bar,
        screening_mode,
    );
    if z1 == 0.0 {
        return sc2;
    }
    let sc1 = eval_screening_pair(
        g,
        jscr,
        i1,
        a1,
        z1,
        0,
        a2 + a3,
        z2 + z3,
        temp,
        den,
        log_t,
        zbar,
        abar,
        z2bar,
        screening_mode,
    );
    ScreenEval {
        scor: sc1.scor * sc2.scor,
        dscor_dt: sc1.scor * sc2.dscor_dt + sc1.dscor_dt * sc2.scor,
        dscor_dd: sc1.scor * sc2.dscor_dd + sc1.dscor_dd * sc2.scor,
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_screening_pair(
    g: &NetGeneralInfo,
    jscr: usize,
    cid1: usize,
    a1: f64,
    z1: f64,
    cid2: usize,
    a2: f64,
    z2: f64,
    temp: f64,
    den: f64,
    log_t: f64,
    zbar: f64,
    abar: f64,
    z2bar: f64,
    screening_mode: ScreeningMode,
) -> ScreenEval {
    if cid1 > 0 && cid2 > 0 {
        let i1 = g.net_iso[cid1];
        let i2 = g.net_iso[cid2];
        if (i1 == 0 || i2 == 0)
            && g.doing_approx21
            && !(cid1 == ICO55
                || cid1 == ICO57
                || cid2 == ICO55
                || cid2 == ICO57
                || cid2 == IH2)
        {
            return ScreenEval::UNSCREENED;
        }
    }
    if log_t < g.log_tcut_lo {
        return ScreenEval {
            scor: 0.0,
            dscor_dt: 0.0,
            dscor_dd: 0.0,
        };
    }
    if screening_mode == ScreeningMode::NoScreening
        || zbar == 0.0
        || z1 <= 0.0
        || z2 <= 0.0
    {
        return ScreenEval::UNSCREENED;
    }
    match screening_mode {
        ScreeningMode::NoScreening => ScreenEval::UNSCREENED,
        ScreeningMode::Salpeter => {
            screen_salpeter(temp, den, zbar, abar, z2bar, z1, z2)
        }
        ScreeningMode::Extended => fxt_screen5_with_params(
            temp,
            den,
            zbar,
            abar,
            z2bar,
            z1,
            z2,
            &g.screen_params[jscr],
        ),
        ScreeningMode::Chugunov => {
            screen_chugunov(temp, den, zbar, abar, a1, z1, a2, z2)
        }
    }
}

#[allow(clippy::too_many_arguments, clippy::similar_names)]
fn fxt_screen5_with_params(
    temp: f64,
    den: f64,
    zbar: f64,
    abar: f64,
    z2bar: f64,
    z1: f64,
    z2: f64,
    p: &ScreenPairParams,
) -> ScreenEval {
    if z1 <= 0.0 || z2 <= 0.0 {
        return ScreenEval::UNSCREENED;
    }
    let ytot = 1.0 / abar;
    let rr_ctx = den * ytot;
    let tempi = 1.0 / temp;
    let dtempidt = -tempi * tempi;
    let deni = 1.0 / den;
    let pp = (rr_ctx * tempi * (z2bar + zbar)).sqrt();
    let qq_ctx = 0.5 / pp * (z2bar + zbar);
    let dppdt = qq_ctx * rr_ctx * dtempidt;
    let dppdd = qq_ctx * ytot * tempi;
    let qlam0z = 1.88e8 * tempi * pp;
    let qlam0zdt = 1.88e8 * (dtempidt * pp + tempi * dppdt);
    let qlam0zdd = 1.88e8 * tempi * dppdd;
    let taufac = (4.248719e3 / 3.0) * tempi.powf(ONE_THIRD);
    let taufacdt = -ONE_THIRD * taufac * tempi;
    let xni = (rr_ctx * zbar).powf(ONE_THIRD);
    let dxnidd = ONE_THIRD * xni * deni;
    let aa = 2.27493e5 * tempi * xni;
    let daadt = 2.27493e5 * dtempidt * xni;
    let daadd = 2.27493e5 * tempi * dxnidd;

    let bb = z1 * z2;
    let mut gamp = aa;
    let mut gampdt = daadt;
    let mut gampdd = daadd;

    let qq = TWO_13 * bb * p.zs13inv;
    let mut gamef = qq * gamp;
    let mut gamefdt = qq * gampdt;
    let mut gamefdd = qq * gampdd;

    let tau12 = taufac * p.aznut;
    let tau12dt = taufacdt * p.aznut;

    let inv_tau = 1.0 / tau12;
    let mut alph12 = gamef * inv_tau;
    let mut alph12dt = (gamefdt - alph12 * tau12dt) * inv_tau;
    let mut alph12dd = gamefdd * inv_tau;

    if alph12 > 1.6 {
        alph12 = 1.6;
        alph12dt = 0.0;
        alph12dd = 0.0;
        gamef = alph12 * tau12;
        gamefdt = alph12 * tau12dt;
        gamefdd = 0.0;
        let q_lim = p.zs13 / (TWO_13 * bb);
        gamp = gamef * q_lim;
        gampdt = gamefdt * q_lim;
        gampdd = 0.0;
    }

    let h12w = bb * qlam0z;
    let dh12wdt = bb * qlam0zdt;
    let dh12wdd = bb * qlam0zdd;

    let mut h12 = h12w;
    let mut dh12dt = dh12wdt;
    let mut dh12dd = dh12wdd;

    if gamef > 0.3 {
        let gamp14 = gamp.powf(0.25);
        let inv_gamp = 1.0 / gamp;
        let q_g14 = 0.25 * gamp14 * inv_gamp;
        let gamp14dt = q_g14 * gampdt;
        let gamp14dd = q_g14 * gampdd;

        let cc = 0.896434 * gamp * p.zhat
            - 3.44740 * gamp14 * p.zhat2
            - 0.5551 * (gamp.ln() + p.lzav)
            - 2.996;
        let dccdt = 0.896434 * gampdt * p.zhat
            - 3.44740 * gamp14dt * p.zhat2
            - 0.5551 * inv_gamp * gampdt;
        let dccdd = 0.896434 * gampdd * p.zhat
            - 3.44740 * gamp14dd * p.zhat2
            - 0.5551 * inv_gamp * gampdd;

        let a3 = alph12 * alph12 * alph12;
        let da3 = 3.0 * alph12 * alph12;

        let qq_a = 0.014 + 0.0128 * alph12;
        let dqqdt = 0.0128 * alph12dt;
        let dqqdd = 0.0128 * alph12dd;

        let rr = 5.0 / 32.0 - alph12 * qq_a;
        let drrdt = -(alph12dt * qq_a + alph12 * dqqdt);
        let drrdd = -(alph12dd * qq_a + alph12 * dqqdd);

        let ss = tau12 * rr;
        let dssdt = tau12dt * rr + tau12 * drrdt;
        let dssdd = tau12 * drrdd;

        let tt = -0.0098 + 0.0048 * alph12;
        let dttdt = 0.0048 * alph12dt;
        let dttdd = 0.0048 * alph12dd;

        let uu = 0.0055 + alph12 * tt;
        let duudt = alph12dt * tt + alph12 * dttdt;
        let duudd = alph12dd * tt + alph12 * dttdd;

        let vv = gamef * alph12 * uu;
        let dvvdt = gamefdt * alph12 * uu
            + gamef * alph12dt * uu
            + gamef * alph12 * duudt;
        let dvvdd = gamefdd * alph12 * uu
            + gamef * alph12dd * uu
            + gamef * alph12 * duudd;

        h12 = cc - a3 * (ss + vv);
        let r_sv = da3 * (ss + vv);
        dh12dt = dccdt - r_sv * alph12dt - a3 * (dssdt + dvvdt);
        dh12dd = dccdd - r_sv * alph12dd - a3 * (dssdd + dvvdd);

        let rr_lg = 1.0 - 0.0562 * a3;
        let ss_lg = -0.0562 * da3;
        let (xlgfac, dxlgfacdt, dxlgfacdd) = if rr_lg >= 0.77 {
            (rr_lg, ss_lg * alph12dt, ss_lg * alph12dd)
        } else {
            (0.77, 0.0, 0.0)
        };
        h12 += xlgfac.ln();
        let inv_xlg = 1.0 / xlgfac;
        dh12dt += inv_xlg * dxlgfacdt;
        dh12dd += inv_xlg * dxlgfacdd;

        if gamef <= 0.8 {
            let dgamma = 1.0 / (0.8 - 0.3);
            let rr_b = dgamma * (0.8 - gamef);
            let drrdt_b = -dgamma * gamefdt;
            let drrdd_b = -dgamma * gamefdd;
            let ss_b = dgamma * (gamef - 0.3);
            let dssdt_b = dgamma * gamefdt;
            let dssdd_b = dgamma * gamefdd;
            let vv_b = h12;
            h12 = h12w * rr_b + vv_b * ss_b;
            dh12dt = dh12wdt * rr_b
                + h12w * drrdt_b
                + dh12dt * ss_b
                + vv_b * dssdt_b;
            dh12dd = dh12wdd * rr_b
                + h12w * drrdd_b
                + dh12dd * ss_b
                + vv_b * dssdd_b;
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
