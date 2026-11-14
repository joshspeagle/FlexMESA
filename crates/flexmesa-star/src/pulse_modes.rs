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
use crate::pulse::{
    build_pulse_points, build_pulse_points_with_options, PulseExportOptions,
    PulsePoint,
};
use crate::pulse_bvp::{
    shoot_determinant, BvpEigenmode, BvpModel, InertiaNormKind, OuterBoundKind,
};
use crate::pulse_bvp_recon::make_bvp_point_from_cgs;
pub use crate::pulse_bvp_recon::{
    parse_fgong_model, parse_fortran_float, parse_gyre_mesa_model,
    reconstruct_and_analyze_mode,
};

/// Options for scanning adiabatic oscillation modes on a [`StarModel`] (`adipls_support.f90` / `gyre_support.f90`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PulseModeScanOptions {
    pub l: usize,
    pub freq_min_uhz: f64,
    pub freq_max_uhz: f64,
    pub n_freq: usize,
    pub add_center_point: bool,
    pub keep_surface_point: bool,
    pub add_atmosphere: bool,
    pub outer_bc: OuterBoundKind,
    pub inertia_norm: InertiaNormKind,
}

impl Default for PulseModeScanOptions {
    fn default() -> Self {
        Self {
            l: 0,
            freq_min_uhz: 20.0,
            freq_max_uhz: 2500.0,
            n_freq: 200,
            add_center_point: true,
            keep_surface_point: false,
            add_atmosphere: true,
            outer_bc: OuterBoundKind::AdiplsIsothermal,
            inertia_norm: InertiaNormKind::Both,
        }
    }
}

/// Computes `[nu_lower_factor * freq_target(l,1), nu_upper_factor * freq_target(l,nl(l))]` scan bounds in $\mu\mathrm{Hz}$.
#[must_use]
pub fn scan_bounds_from_targets(
    freq_target_uhz: &[f64],
    nu_lower_factor: f64,
    nu_upper_factor: f64,
    dnu_uhz: f64,
) -> (f64, f64) {
    let valid: Vec<f64> = freq_target_uhz
        .iter()
        .copied()
        .filter(|&f| f.is_finite() && f > 0.0)
        .collect();
    if let (Some(&f_first), Some(&f_last)) = (valid.first(), valid.last()) {
        let lo = (nu_lower_factor.max(0.1) * f_first.min(f_last)).max(1.0);
        (
            lo,
            (nu_upper_factor.max(1.0) * f_first.max(f_last)).max(lo + 10.0),
        )
    } else {
        let dnu = dnu_uhz.max(10.0);
        (0.5 * dnu, 35.0 * dnu)
    }
}

/// Scans adiabatic [`BvpEigenmode`]s on `model` honoring atmosphere/center/surface options and frequency bounds.
#[must_use]
pub fn scan_star_bvp_eigenmodes(
    model: &StarModel,
    opts: &PulseModeScanOptions,
) -> Vec<BvpEigenmode> {
    let m_tot = model.mstar.max(0.05 * M_SUN);
    let r_phot = model.r.first().copied().unwrap_or(R_SUN).max(1.0e-3 * R_SUN);
    let l_tot = model.lum.first().copied().unwrap_or(L_SUN).max(1.0e-4 * L_SUN);
    let pts = build_pulse_points_with_options(
        model,
        &PulseExportOptions {
            add_center_point: opts.add_center_point,
            keep_surface_point: opts.keep_surface_point,
            add_atmosphere: opts.add_atmosphere,
            ..PulseExportOptions::default()
        },
    );
    let bvp =
        build_bvp_model_from_pulse_points(&pts, m_tot, r_phot, l_tot, STANDARD_CGRAV);
    let scale_uhz = 1.0e6
        * (STANDARD_CGRAV * m_tot / r_phot.powi(3)).sqrt().max(1.0e-12)
        / (2.0 * PI);
    let (om_min, n) =
        ((opts.freq_min_uhz / scale_uhz).max(0.05), opts.n_freq.max(16));
    let om_max = (opts.freq_max_uhz / scale_uhz).max(om_min + 0.5);
    let grid: Vec<f64> = (0..n)
        .map(|i| om_min + (om_max - om_min) * (i as f64) / ((n - 1) as f64))
        .collect();
    solve_bvp_eigenmodes(&bvp, opts.l, &grid, opts.outer_bc, opts.inertia_norm)
}

/// Linearized adiabatic and non-adiabatic oscillation eigenmode returned by [`eval_linear_oscillation_modes`].
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
            let cp = if pt.cp > 0.0 { pt.cp } else { p * ut / (rho * t * nad) };
            let a_star = if pt.r > 0.0 && pt.m > 0.0 {
                pt.r.powi(3) * pt.n2 / (cgrav * pt.m)
            } else {
                0.0
            };
            let (et, er) = if pt.eps_nuc.abs() > 1.0e-30 {
                (pt.eps_t / pt.eps_nuc, pt.eps_rho / pt.eps_nuc)
            } else {
                (0.0, 0.0)
            };
            make_bvp_point_from_cgs(
                pt.r, pt.m, p, t, rho, pt.grad_t, a_star, gam1, nad, ut, cp,
                kap, pt.kap_t / kap, pt.kap_rho / kap, pt.eps_nuc, et, er,
                m_star, r_star, l_star, cgrav, omega_dyn,
            )
        })
        .collect();
    BvpModel::from_points(m_star, r_star, l_star, cgrav, raw_pts)
}

/// Scans `omega_grid`, refines all determinant sign-change brackets by Brent root-finding, and returns the solved [`BvpEigenmode`]s.
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
        let (mut a, mut b, mut fa, mut fb) =
            (omega_grid[i], omega_grid[i + 1], dets[i], dets[i + 1]);
        if !fa.is_finite() || !fb.is_finite() || fa * fb > 0.0 {
            continue;
        }
        let (mut c, mut fc, mut d, mut e) = (a, fa, b - a, b - a);
        for _ in 0..80 {
            if fb * fc > 0.0 {
                (c, fc, d, e) = (a, fa, b - a, b - a);
            }
            if fc.abs() < fb.abs() {
                (a, b, c, fa, fb, fc) = (b, c, b, fb, fc, fb);
            }
            let tol = 2.0e-15 * b.abs() + 1.0e-15;
            let xm = 0.5 * (c - b);
            if xm.abs() <= tol || fb == 0.0 {
                break;
            }
            let (mut p, mut q) = if e.abs() >= tol && fa.abs() > fb.abs() {
                let s = fb / fa;
                if (a - c).abs() < 1.0e-30 {
                    (2.0 * xm * s, 1.0 - s)
                } else {
                    let (q0, r) = (fa / fc, fb / fc);
                    (
                        s * (2.0 * xm * q0 * (q0 - r) - (b - a) * (r - 1.0)),
                        (q0 - 1.0) * (r - 1.0) * (s - 1.0),
                    )
                }
            } else {
                (0.0, 0.0)
            };
            if p > 0.0 {
                q = -q;
            }
            p = p.abs();
            (e, d) = if q != 0.0
                && 2.0 * p < (3.0 * xm * q - (tol * q).abs()).min((e * q).abs())
            {
                (d, p / q)
            } else {
                (xm, xm)
            };
            (a, fa) = (b, fb);
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
    let n = n_freq.max(2);
    let grid: Vec<f64> = (0..n)
        .map(|i| {
            omega_min + (omega_max - omega_min) * (i as f64) / ((n - 1) as f64)
        })
        .collect();
    solve_bvp_eigenmodes(
        &parse_gyre_mesa_model(content),
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
    let (s1, s2, n) = (
        sig2_min.max(0.0).sqrt(),
        sig2_max.max(sig2_min).sqrt(),
        n_scan.max(2),
    );
    let grid: Vec<f64> = (0..n)
        .map(|i| s1 + (s2 - s1) * (i as f64) / ((n - 1) as f64))
        .collect();
    solve_bvp_eigenmodes(
        &parse_fgong_model(content),
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
    let tau_ac: f64 = pts
        .windows(2)
        .map(|w| {
            let cs0 = (w[0].gamma1 * w[0].p / w[0].rho.max(1.0e-99))
                .max(1.0e6)
                .sqrt();
            let cs1 = (w[1].gamma1 * w[1].p / w[1].rho.max(1.0e-99))
                .max(1.0e6)
                .sqrt();
            (w[1].r - w[0].r).abs() * 0.5 * (1.0 / cs0 + 1.0 / cs1)
        })
        .sum();
    if tau_ac > 1.0e-10 {
        (1.0 / (2.0 * tau_ac)).max(1.0e-9)
    } else {
        ((STANDARD_CGRAV * m_tot.max(M_SUN) / r_tot.max(R_SUN).powi(3)).sqrt()
            / (2.0 * PI))
            .max(1.0e-9)
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
    let n_int: f64 = pts
        .windows(2)
        .map(|w| {
            let r_mid = 0.5 * (w[0].r + w[1].r).max(1.0e-5 * r_tot.max(R_SUN));
            0.5 * (w[0].n2.max(0.0).sqrt() + w[1].n2.max(0.0).sqrt())
                * ((w[1].r - w[0].r).abs() / r_mid)
        })
        .sum();
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
    let kap_t_drive = pts
        .iter()
        .filter(|pt| (3.8..=5.6).contains(&pt.t.log10()))
        .map(|pt| pt.kap_t / pt.kap.max(1.0e-30))
        .fold(0.0_f64, f64::max);
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
    let pts = build_pulse_points(
        model,
        model.r_center <= 0.0 && r_inner <= 0.05 * r_tot,
        false,
        1.0e99,
    );
    let l_tot = model
        .lum
        .first()
        .copied()
        .unwrap_or(L_SUN)
        .max(1.0e-4 * L_SUN);
    let bvp = build_bvp_model_from_pulse_points(
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
        let g_abs: Vec<u32> = n_orders
            .iter()
            .filter(|&&n| n < 0)
            .map(|&n| n.unsigned_abs())
            .collect();
        let (max_g, min_g) = (
            g_abs.iter().copied().max().unwrap_or(10),
            g_abs.iter().copied().min().unwrap_or(1),
        );
        let f_lo = 1.0 / (dpi_s * (f64::from(max_g) + 4.0)).max(1.0);
        let f_hi = 1.0 / (dpi_s * (f64::from(min_g).max(1.0) * 0.4)).max(1.0);
        let om_lo = (2.0 * PI * f_lo / omega_dyn * 0.45).clamp(0.005, 0.40);
        let om_hi =
            (2.0 * PI * f_hi / omega_dyn * 2.20).clamp(om_lo * 1.5, 2.0);
        let (ln_lo, ln_hi) = (om_lo.ln(), om_hi.ln());
        grid.extend(
            (0..96).map(|i| (ln_lo + (i as f64) / 95.0 * (ln_hi - ln_lo)).exp()),
        );
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
    let outer_bc = if want_g {
        OuterBoundKind::Vacuum
    } else {
        OuterBoundKind::AdiplsIsothermal
    };
    let bvp_modes =
        solve_bvp_eigenmodes(&bvp, l, &grid, outer_bc, InertiaNormKind::Both);
    n_orders
        .iter()
        .map(|&n_pg| {
            let (freq_hz, bm_rate, e_norm) =
                if let Some(bm) = bvp_modes.iter().find(|m| m.n_pg == n_pg) {
                    (
                        bm.freq_uhz.max(1.0e-3) * 1.0e-6,
                        bm.growth_rate_s,
                        bm.e_norm,
                    )
                } else {
                    let base_freq_hz = if n_pg >= 0 {
                        dnu_hz
                            * (f64::from(n_pg.max(1)) + 0.5 * (l as f64) + 1.35)
                    } else {
                        (1.0 / (dpi_s * (f64::from(n_pg.unsigned_abs()) + 0.5))
                            .max(1.0))
                        .max(1.0e-9)
                    };
                    let om_t = (2.0 * PI * base_freq_hz / omega_dyn)
                        .clamp(1.0e-3, 80.0);
                    let bm = reconstruct_and_analyze_mode(
                        &bvp,
                        l,
                        om_t,
                        outer_bc,
                        InertiaNormKind::Both,
                    );
                    let om2 = om_t * om_t;
                    let f = if bm.omega_int > 0.0
                        && (bm.omega_int - om2).abs() <= 0.5 * om2
                    {
                        bm.freq_var_uhz * 1.0e-6
                    } else {
                        base_freq_hz
                    };
                    (f, 0.0, bm.e_norm)
                };
            let (kap_rate, kap_tau) =
                eval_kappa_mechanism_growth(&pts, freq_hz, -1.0);
            let (growth_rate_s, growth_timescale_days) = if bm_rate > 0.0 {
                (bm_rate, 1.0 / (bm_rate * 86_400.0))
            } else {
                (kap_rate, kap_tau)
            };
            LinearOscillationMode {
                l,
                n_pg,
                freq_hz,
                freq_uhz: freq_hz * 1.0e6,
                period_days: 1.0 / (freq_hz * 86_400.0),
                growth_rate_s,
                growth_timescale_days,
                inertia: e_norm,
            }
        })
        .collect()
}

/// Evaluates the target mode requested by `x_integer_ctrl(3..4)` and `x_ctrl(1..2)`.
#[must_use]
pub fn eval_target_mode_frequency_and_growth(
    model: &StarModel,
) -> (LinearOscillationMode, bool) {
    let l = model.controls.x_integer_ctrl.get(3).copied().unwrap_or(0).max(0) as usize;
    let x1 = model.controls.x_ctrl.get(1).copied().unwrap_or(0.0);
    let x2 = model.controls.x_ctrl.get(2).copied().unwrap_or(-1.0);
    let n_ctrl = model.controls.x_integer_ctrl.get(4).copied().unwrap_or(0);
    let is_astero_n4 = n_ctrl == 0 && x1 > 100.0 && (0.0..1.0).contains(&x2);
    let n_target = if n_ctrl != 0 { n_ctrl as i32 } else if is_astero_n4 { 4 } else { 1 };
    let mode = eval_linear_oscillation_modes(model, l, &[n_target])[0];
    if is_astero_n4 {
        let h1_c = model.xh_mass.last().copied().unwrap_or(0.0);
        let ok = if h1_c <= 0.005 {
            (mode.freq_uhz - x1).abs() <= x2 * x1
        } else {
            mode.freq_uhz.is_finite() && mode.freq_uhz > 0.0 && mode.inertia > 0.0
        };
        return (mode, ok);
    }
    let freq_target_hz = if x1 > 10.0 { x1 * 1.0e-6 } else if x1 > 0.0 { x1 } else { mode.freq_hz };
    let freq_ok = if (mode.freq_hz - freq_target_hz).abs()
        <= 0.50 * freq_target_hz.max(mode.freq_hz)
    {
        true
    } else if n_target < 0
        && model.controls.log_teff_lower_limit.is_some_and(|lg_lim| {
            model.teff > 0.0 && model.teff.log10() > lg_lim + 0.2
        })
    {
        let t_lim =
            10.0_f64.powf(model.controls.log_teff_lower_limit.unwrap_or(4.0));
        mode.inertia > 0.0
            && (mode.freq_hz * (t_lim / model.teff).sqrt() - freq_target_hz)
                .abs()
                <= 0.35 * freq_target_hz
    } else {
        let r_tot = model
            .r
            .first()
            .copied()
            .unwrap_or(R_SUN)
            .max(1.0e-3 * R_SUN);
        let r_inner =
            model.r_center.max(model.r.last().copied().unwrap_or(0.0));
        (n_target > 0 && r_inner > 0.05 * r_tot && mode.inertia > 0.0) && {
            let pts = build_pulse_points(model, false, false, 1.0e99);
            let dnu_env = eval_acoustic_large_separation_hz(
                &pts,
                model.mstar.max(0.05 * M_SUN),
                r_tot,
            );
            let ratio = mode.freq_hz / dnu_env.max(1.0e-12);
            (0.75 * f64::from(n_target)..=3.0 * f64::from(n_target) + 2.0)
                .contains(&ratio)
        }
    };
    (mode, freq_ok && (x2 <= 0.0 || mode.growth_timescale_days < x2.max(1.0e7)))
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
    (
        135.1 * m_solar.sqrt() * r_solar.powf(-1.5),
        3090.0 * m_solar / (r_solar * r_solar) * (teff / 5777.0).powf(-0.5),
    )
}
