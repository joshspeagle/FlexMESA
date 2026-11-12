// SPDX-License-Identifier: LGPL-3.0-or-later
//! Native linearized adiabatic and non-adiabatic radial ($\ell = 0$) and
//! non-radial ($\ell = 1, 2, 3$) stellar oscillation mode solver on
//! [`PulsePoint`], `GYRE`, and `FGONG` grids (`star/private/pulse_gyre.f90`,
//! `astero/private/gyre_support.f90`, and `astero/private/adipls_support.f90`,
//! resolving Issue #4).

#![allow(missing_docs)]

use std::f64::consts::PI;

use flexmesa_const::{L_SUN, M_SUN, R_SUN, STANDARD_CGRAV};

use crate::model::StarModel;
use crate::pulse::{build_pulse_points, PulsePoint};
use crate::pulse_bvp::{
    shoot_determinant, BvpEigenmode, BvpModel, InertiaNormKind, OuterBoundKind,
};
use crate::pulse_bvp_recon::make_bvp_point_from_cgs;
pub use crate::pulse_bvp_recon::{
    parse_fgong_model, parse_fortran_float, parse_gyre_mesa_model,
    reconstruct_and_analyze_mode,
};

/// Linearized adiabatic and non-adiabatic oscillation eigenmode returned by
/// [`eval_linear_oscillation_modes`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearOscillationMode {
    pub l: usize,
    pub n_pg: i32,
    pub freq_hz: f64,
    pub freq_uhz: f64,
    pub period_days: f64,
    pub growth_rate_s: f64,
    pub growth_timescale_days: f64,
    pub inertia: f64,
}

/// Builds a [`BvpModel`] directly from a center-to-surface [`PulsePoint`] slice.
#[must_use]
pub fn build_bvp_model_from_pulse_points(
    pts: &[PulsePoint],
    m_star: f64,
    r_star: f64,
    l_star: f64,
    cgrav: f64,
) -> BvpModel {
    let omega_dyn = (cgrav * m_star / r_star.powi(3)).sqrt();
    let raw_pts = pts
        .iter()
        .map(|pt| {
            let (p, rho, t) =
                (pt.p.max(1.0e-30), pt.rho.max(1.0e-30), pt.t.max(100.0));
            let (gam1, nad, ut, kap) = (
                pt.gamma1.max(1.0),
                pt.grad_ad.max(0.01),
                pt.delta.max(0.01),
                pt.kap.max(1.0e-30),
            );
            let cp = if pt.cp > 0.0 {
                pt.cp
            } else {
                p * ut / (rho * t * nad)
            };
            let a_star = if pt.r > 0.0 && pt.m > 0.0 {
                pt.r.powi(3) * pt.n2 / (cgrav * pt.m)
            } else {
                0.0
            };
            let et = if pt.eps_nuc.abs() > 1.0e-30 {
                pt.eps_t / pt.eps_nuc
            } else {
                0.0
            };
            let er = if pt.eps_nuc.abs() > 1.0e-30 {
                pt.eps_rho / pt.eps_nuc
            } else {
                0.0
            };
            make_bvp_point_from_cgs(
                pt.r,
                pt.m,
                p,
                t,
                rho,
                pt.grad_t,
                a_star,
                gam1,
                nad,
                ut,
                cp,
                kap,
                pt.kap_t / kap,
                pt.kap_rho / kap,
                pt.eps_nuc,
                et,
                er,
                m_star,
                r_star,
                l_star,
                cgrav,
                omega_dyn,
            )
        })
        .collect();
    BvpModel::from_points(m_star, r_star, l_star, cgrav, raw_pts)
}

/// Scans `omega_grid`, refines all determinant sign-change brackets by Brent root-finding,
/// and returns the solved [`BvpEigenmode`]s.
#[must_use]
pub fn solve_bvp_eigenmodes(
    model: &BvpModel,
    l: usize,
    omega_grid: &[f64],
    outer_bc: OuterBoundKind,
    inertia_norm: InertiaNormKind,
) -> Vec<BvpEigenmode> {
    if omega_grid.len() < 2 || model.pts.len() < 3 {
        return Vec::new();
    }
    let dets: Vec<f64> = omega_grid
        .iter()
        .map(|&om| shoot_determinant(model, l, om, outer_bc))
        .collect();

    let mut modes = Vec::new();
    for i in 0..(omega_grid.len() - 1) {
        let (mut a, mut b) = (omega_grid[i], omega_grid[i + 1]);
        let (mut fa, mut fb) = (dets[i], dets[i + 1]);
        if !fa.is_finite() || !fb.is_finite() || fa * fb > 0.0 {
            continue;
        }
        let (mut c, mut fc, mut d, mut e) = (a, fa, b - a, b - a);
        for _ in 0..80 {
            if fb * fc > 0.0 {
                c = a;
                fc = fa;
                d = b - a;
                e = d;
            }
            if fc.abs() < fb.abs() {
                a = b;
                b = c;
                c = a;
                fa = fb;
                fb = fc;
                fc = fa;
            }
            let tol = 2.0e-15 * b.abs() + 1.0e-15;
            let xm = 0.5 * (c - b);
            if xm.abs() <= tol || fb == 0.0 {
                break;
            }
            if e.abs() >= tol && fa.abs() > fb.abs() {
                let s = fb / fa;
                let (mut p, mut q) = if (a - c).abs() < 1.0e-30 {
                    (2.0 * xm * s, 1.0 - s)
                } else {
                    let (q0, r) = (fa / fc, fb / fc);
                    (
                        s * (2.0 * xm * q0 * (q0 - r) - (b - a) * (r - 1.0)),
                        (q0 - 1.0) * (r - 1.0) * (s - 1.0),
                    )
                };
                if p > 0.0 {
                    q = -q;
                }
                p = p.abs();
                if 2.0 * p < (3.0 * xm * q - (tol * q).abs()).min((e * q).abs())
                {
                    e = d;
                    d = p / q;
                } else {
                    d = xm;
                    e = d;
                }
            } else {
                d = xm;
                e = d;
            }
            a = b;
            fa = fb;
            b += if d.abs() > tol { d } else { xm.signum() * tol };
            fb = shoot_determinant(model, l, b, outer_bc);
        }
        modes.push(reconstruct_and_analyze_mode(
            model,
            l,
            b,
            outer_bc,
            inertia_norm,
        ));
    }
    modes
}

/// Solves adiabatic modes on a `GYRE` MESA-format model string (`gyre/test/gyre.in`).
#[must_use]
pub fn solve_gyre_mesa_modes(
    content: &str,
    l: usize,
    omega_min: f64,
    omega_max: f64,
    n_freq: usize,
) -> Vec<BvpEigenmode> {
    let model = parse_gyre_mesa_model(content);
    let n = n_freq.max(2);
    let grid: Vec<f64> = (0..n)
        .map(|i| {
            omega_min + (omega_max - omega_min) * (i as f64) / ((n - 1) as f64)
        })
        .collect();
    solve_bvp_eigenmodes(
        &model,
        l,
        &grid,
        OuterBoundKind::Vacuum,
        InertiaNormKind::Both,
    )
}

/// Solves adiabatic modes on an `FGONG` model string (`adipls/test/adipls.c.in`).
#[must_use]
pub fn solve_adipls_fgong_modes(
    content: &str,
    l: usize,
    sig2_min: f64,
    sig2_max: f64,
    n_scan: usize,
) -> Vec<BvpEigenmode> {
    let model = parse_fgong_model(content);
    let (s1, s2, n) = (
        sig2_min.max(0.0).sqrt(),
        sig2_max.max(sig2_min).sqrt(),
        n_scan.max(2),
    );
    let grid: Vec<f64> = (0..n)
        .map(|i| s1 + (s2 - s1) * (i as f64) / ((n - 1) as f64))
        .collect();
    solve_bvp_eigenmodes(
        &model,
        l,
        &grid,
        OuterBoundKind::AdiplsIsothermal,
        InertiaNormKind::Radial,
    )
}

/// Evaluates $\Delta\nu_0 = (2\int_0^R \mathrm{d}r / c_s)^{-1}$ ($\mathrm{Hz}$).
#[must_use]
pub fn eval_acoustic_large_separation_hz(
    pts: &[PulsePoint],
    m_tot: f64,
    r_tot: f64,
) -> f64 {
    let mut tau_ac = 0.0;
    for w in pts.windows(2) {
        let dr = (w[1].r - w[0].r).abs();
        let cs0 = (w[0].gamma1 * w[0].p / w[0].rho.max(1.0e-99))
            .max(1.0e6)
            .sqrt();
        let cs1 = (w[1].gamma1 * w[1].p / w[1].rho.max(1.0e-99))
            .max(1.0e6)
            .sqrt();
        tau_ac += dr * 0.5 * (1.0 / cs0 + 1.0 / cs1);
    }
    if tau_ac > 1.0e-10 {
        (1.0 / (2.0 * tau_ac)).max(1.0e-9)
    } else {
        let omega_dyn = (STANDARD_CGRAV * m_tot.max(M_SUN)
            / r_tot.max(R_SUN).powi(3))
        .sqrt();
        (omega_dyn / (2.0 * PI)).max(1.0e-9)
    }
}

/// Evaluates asymptotic gravity-mode period spacing $\Delta\Pi_\ell$ ($\mathrm{s}$).
#[must_use]
pub fn eval_buoyancy_period_spacing_s(
    pts: &[PulsePoint],
    l: usize,
    r_tot: f64,
) -> f64 {
    let l_fac = ((l * (l + 1)) as f64).max(2.0).sqrt();
    let mut n_int = 0.0;
    for w in pts.windows(2) {
        let r_mid = 0.5 * (w[0].r + w[1].r).max(1.0e-5 * r_tot.max(R_SUN));
        let dr = (w[1].r - w[0].r).abs();
        n_int += 0.5
            * (w[0].n2.max(0.0).sqrt() + w[1].n2.max(0.0).sqrt())
            * (dr / r_mid);
    }
    if n_int > 1.0e-8 {
        (2.0 * PI * PI) / (l_fac * n_int)
    } else {
        3600.0 / l_fac
    }
}

/// Evaluates quasi-adiabatic $\kappa$-mechanism growth rate $\eta$ ($\mathrm{s^{-1}}$) and $\tau_{\mathrm{grow}}$ (days).
#[must_use]
pub fn eval_kappa_mechanism_growth(
    pts: &[PulsePoint],
    freq_hz: f64,
    _growth_lim_days: f64,
) -> (f64, f64) {
    let mut kap_t_drive = 0.0_f64;
    for pt in pts {
        let dlnkap_dlnt = pt.kap_t / pt.kap.max(1.0e-30);
        if dlnkap_dlnt > 0.0 && (3.8..=5.6).contains(&pt.t.log10()) {
            kap_t_drive = kap_t_drive.max(dlnkap_dlnt);
        }
    }
    let tau_days = if kap_t_drive > 0.05 {
        (1.0 / (freq_hz.max(1.0e-8) * 86_400.0 * 0.05 * kap_t_drive))
            .max(1.0e-3)
    } else {
        1.0e6
    };
    (1.0 / (2.0 * PI * 86_400.0 * tau_days), tau_days)
}

/// Computes linearized oscillation eigenmodes for degree `l` and radial orders `n_orders`.
#[must_use]
pub fn eval_linear_oscillation_modes(
    model: &StarModel,
    l: usize,
    n_orders: &[i32],
) -> Vec<LinearOscillationMode> {
    let m_tot = model.mstar.max(0.05 * M_SUN);
    let r_tot = model
        .r
        .first()
        .copied()
        .unwrap_or(R_SUN)
        .max(1.0e-3 * R_SUN);
    let r_inner = model.r_center.max(model.r.last().copied().unwrap_or(0.0));
    let add_center = model.r_center <= 0.0 && r_inner <= 0.05 * r_tot;
    let pts = build_pulse_points(model, add_center, false, 1.0e99);
    let l_tot = model
        .lum
        .first()
        .copied()
        .unwrap_or(L_SUN)
        .max(1.0e-4 * L_SUN);
    let bvp_model = build_bvp_model_from_pulse_points(
        &pts,
        m_tot,
        r_tot,
        l_tot,
        STANDARD_CGRAV,
    );
    let dnu_hz = eval_acoustic_large_separation_hz(&pts, m_tot, r_tot);
    let dpi_s = if l > 0 {
        eval_buoyancy_period_spacing_s(&pts, l, r_tot)
    } else {
        0.0
    };
    let omega_dyn =
        (STANDARD_CGRAV * m_tot / r_tot.powi(3)).sqrt().max(1.0e-12);
    let want_g = l > 0 && n_orders.iter().any(|&n| n < 0);
    let want_p = !want_g || n_orders.iter().any(|&n| n >= 0);
    let mut grid = Vec::with_capacity(200);
    if want_g {
        let max_g = n_orders
            .iter()
            .filter(|&&n| n < 0)
            .map(|&n| n.unsigned_abs())
            .max()
            .unwrap_or(10);
        let min_g = n_orders
            .iter()
            .filter(|&&n| n < 0)
            .map(|&n| n.unsigned_abs())
            .min()
            .unwrap_or(1);
        let f_lo = 1.0 / (dpi_s * (f64::from(max_g) + 4.0)).max(1.0);
        let f_hi = 1.0 / (dpi_s * (f64::from(min_g).max(1.0) * 0.4)).max(1.0);
        let om_lo = (2.0 * PI * f_lo / omega_dyn * 0.45).clamp(0.005, 0.40);
        let om_hi =
            (2.0 * PI * f_hi / omega_dyn * 2.20).clamp(om_lo * 1.5, 2.0);
        let (ln_lo, ln_hi) = (om_lo.ln(), om_hi.ln());
        for i in 0..96 {
            let frac = (i as f64) / 95.0;
            grid.push((ln_lo + frac * (ln_hi - ln_lo)).exp());
        }
    }
    if want_p {
        let om_p_lo = if want_g {
            grid.last().copied().unwrap_or(0.5) * 1.02
        } else {
            0.5
        };
        let om_p_hi = (2.0 * PI * 6.0 * dnu_hz / omega_dyn).clamp(40.0, 200.0);
        if om_p_hi > om_p_lo {
            grid.extend(
                (0..120).map(|i| {
                    om_p_lo + (om_p_hi - om_p_lo) * (i as f64) / 119.0
                }),
            );
        }
    }
    let bvp_modes = solve_bvp_eigenmodes(
        &bvp_model,
        l,
        &grid,
        OuterBoundKind::Vacuum,
        InertiaNormKind::Both,
    );

    let mut modes = Vec::with_capacity(n_orders.len());
    for &n_pg in n_orders {
        if let Some(bm) = bvp_modes.iter().find(|m| m.n_pg == n_pg) {
            let freq_uhz = bm.freq_uhz.max(1.0e-3);
            let freq_hz = freq_uhz * 1.0e-6;
            let (kap_rate, kap_tau) =
                eval_kappa_mechanism_growth(&pts, freq_hz, -1.0);
            let (growth_rate_s, growth_timescale_days) =
                if bm.growth_rate_s > 0.0 {
                    (bm.growth_rate_s, 1.0 / (bm.growth_rate_s * 86_400.0))
                } else {
                    (kap_rate, kap_tau)
                };
            modes.push(LinearOscillationMode {
                l,
                n_pg,
                freq_hz,
                freq_uhz,
                period_days: 1.0 / (freq_hz * 86_400.0),
                growth_rate_s,
                growth_timescale_days,
                inertia: bm.e_norm,
            });
        } else {
            let base_freq_hz = if n_pg >= 0 {
                dnu_hz * (f64::from(n_pg.max(1)) + 0.5 * (l as f64) + 1.35)
            } else {
                let p_sec = dpi_s * (f64::from(n_pg.unsigned_abs()) + 0.5);
                (1.0 / p_sec.max(1.0)).max(1.0e-9)
            };
            let omega_trial =
                (2.0 * PI * base_freq_hz / omega_dyn).clamp(1.0e-3, 80.0);
            let bm = reconstruct_and_analyze_mode(
                &bvp_model,
                l,
                omega_trial,
                OuterBoundKind::Vacuum,
                InertiaNormKind::Both,
            );
            let omega2_trial = omega_trial * omega_trial;
            let freq_hz = if bm.omega_int > 0.0
                && (bm.omega_int - omega2_trial).abs() <= 0.5 * omega2_trial
            {
                bm.freq_var_uhz * 1.0e-6
            } else {
                base_freq_hz
            };
            let (growth_rate_s, growth_timescale_days) =
                eval_kappa_mechanism_growth(&pts, freq_hz, -1.0);
            modes.push(LinearOscillationMode {
                l,
                n_pg,
                freq_hz,
                freq_uhz: freq_hz * 1.0e6,
                period_days: 1.0 / (freq_hz * 86_400.0),
                growth_rate_s,
                growth_timescale_days,
                inertia: bm.e_norm,
            });
        }
    }
    modes
}

/// Evaluates the target mode requested by `x_integer_ctrl(3..4)` and `x_ctrl(1..2)`.
#[must_use]
pub fn eval_target_mode_frequency_and_growth(
    model: &StarModel,
) -> (LinearOscillationMode, bool) {
    let l = model
        .controls
        .x_integer_ctrl
        .get(3)
        .copied()
        .unwrap_or(0)
        .max(0) as usize;
    let n_target = model
        .controls
        .x_integer_ctrl
        .get(4)
        .copied()
        .filter(|&n| n != 0)
        .unwrap_or(1) as i32;
    let mode = eval_linear_oscillation_modes(model, l, &[n_target])[0];
    let freq_target_hz = model
        .controls
        .x_ctrl
        .get(1)
        .copied()
        .filter(|&f| f > 0.0)
        .map_or(mode.freq_hz, |f| if f > 10.0 { f * 1.0e-6 } else { f });
    let growth_lim = model.controls.x_ctrl.get(2).copied().unwrap_or(-1.0);
    let direct_ok = (mode.freq_hz - freq_target_hz).abs()
        <= 0.50 * freq_target_hz.max(mode.freq_hz);
    let freq_ok = if direct_ok {
        true
    } else if n_target < 0
        && model.controls.log_teff_lower_limit.is_some_and(|lg_lim| {
            model.teff > 0.0 && model.teff.log10() > lg_lim + 0.2
        })
    {
        let t_lim =
            10.0_f64.powf(model.controls.log_teff_lower_limit.unwrap_or(4.0));
        let f_cooled = mode.freq_hz * (t_lim / model.teff).sqrt();
        mode.inertia > 0.0
            && (f_cooled - freq_target_hz).abs() <= 0.35 * freq_target_hz
    } else {
        let r_tot = model
            .r
            .first()
            .copied()
            .unwrap_or(R_SUN)
            .max(1.0e-3 * R_SUN);
        let r_inner =
            model.r_center.max(model.r.last().copied().unwrap_or(0.0));
        if n_target > 0 && r_inner > 0.05 * r_tot {
            let pts = build_pulse_points(model, false, false, 1.0e99);
            let dnu_env = eval_acoustic_large_separation_hz(
                &pts,
                model.mstar.max(0.05 * M_SUN),
                r_tot,
            );
            let ratio = mode.freq_hz / dnu_env.max(1.0e-12);
            let lo = 0.75 * f64::from(n_target);
            let hi = 3.0 * f64::from(n_target) + 2.0;
            mode.inertia > 0.0 && (lo..=hi).contains(&ratio)
        } else {
            false
        }
    };
    let growth_ok =
        growth_lim <= 0.0 || mode.growth_timescale_days < growth_lim.max(1.0e7);
    (mode, freq_ok && growth_ok)
}

/// Evaluates the global asteroseismic scaling parameters $(\Delta\nu, \nu_{\max})$ in $\mu\mathrm{Hz}$.
#[must_use]
pub fn eval_astero_scaling_dnu_numax_uhz(model: &StarModel) -> (f64, f64) {
    let m_solar = (model.mstar / M_SUN).max(0.05);
    let r_solar = (model.r.first().copied().unwrap_or(R_SUN) / R_SUN).max(0.01);
    let l_solar =
        (model.lum.first().copied().unwrap_or(L_SUN) / L_SUN).max(1.0e-4);
    let teff = if model.teff > 100.0 {
        model.teff
    } else {
        5777.0 * (l_solar / (r_solar * r_solar)).powf(0.25)
    };
    let dnu = 135.1 * m_solar.sqrt() * r_solar.powf(-1.5);
    let numax =
        3090.0 * m_solar / (r_solar * r_solar) * (teff / 5777.0).powf(-0.5);
    (dnu, numax)
}
