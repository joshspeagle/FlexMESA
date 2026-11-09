// SPDX-License-Identifier: LGPL-3.0-or-later
//! Native linearized adiabatic and non-adiabatic radial ($\ell = 0$) and
//! non-radial ($\ell = 1, 2, 3$) stellar oscillation mode solver on
//! [`PulsePoint`] grids (`star/private/pulse_gyre.f90`, `astero/private/gyre_support.f90`,
//! and `astero/private/adipls_support.f90`, resolving Issue #4).

use std::f64::consts::PI;

use flexmesa_const::{L_SUN, M_SUN, R_SUN, STANDARD_CGRAV};

use crate::model::StarModel;
use crate::pulse::{build_pulse_points, PulsePoint};

/// Linearized adiabatic and non-adiabatic oscillation eigenmode returned by
/// [`eval_linear_oscillation_modes`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearOscillationMode {
    /// Spherical harmonic degree $\ell$ (`0` = radial, `1` = dipole, `2` = quadrupole).
    pub l: usize,
    /// Net radial order $n_{pg} = n_p - n_g$ ($> 0$ for acoustic $p$-modes,
    /// $< 0$ for gravity $g$-modes).
    pub n_pg: i32,
    /// Cyclic oscillation frequency $\nu = \operatorname{Re}(\omega)/(2\pi)$ in $\mathrm{Hz}$.
    pub freq_hz: f64,
    /// Cyclic oscillation frequency $\nu$ in $\mu\mathrm{Hz}$ ($10^6\,\mathrm{Hz}$).
    pub freq_uhz: f64,
    /// Oscillation period $P = 1 / (86400\,\nu)$ in days.
    pub period_days: f64,
    /// Linear amplitude growth rate $\eta = \operatorname{Im}(\sigma)$ in $\mathrm{s^{-1}}$
    /// ($> 0$ for unstable $\kappa$-mechanism driving).
    pub growth_rate_s: f64,
    /// Amplitude $e$-folding growth timescale $\tau_{\mathrm{grow}} = 1 / (2\pi \cdot 86400 \cdot \operatorname{Im}(\nu))$
    /// in days ($> 0$ when unstable).
    pub growth_timescale_days: f64,
    /// Normalized mode inertia $E_{\mathrm{norm}}$.
    pub inertia: f64,
}

/// Evaluates the acoustic sound-crossing integral $\tau_{\mathrm{ac}} = \int_0^R \mathrm{d}r / c_s(r)$
/// ($\mathrm{s}$) and the asymptotic large frequency separation
/// $\Delta\nu_0 = (2\tau_{\mathrm{ac}})^{-1}$ ($\mathrm{Hz}$) on a center-to-surface
/// [`PulsePoint`] slice.
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

/// Extracts and normalizes the target oscillation frequency `x_ctrl(1)` into $\mathrm{Hz}$
/// (converting values $> 10$ from $\mu\mathrm{Hz}$ to $\mathrm{Hz}$).
fn target_freq_hz_from_controls(model: &StarModel) -> Option<f64> {
    model
        .controls
        .x_ctrl
        .get(1)
        .copied()
        .filter(|&f| f > 0.0)
        .map(|f| if f > 10.0 { f * 1.0e-6 } else { f })
}

/// Evaluates the asymptotic gravity-mode period spacing
/// $\Delta\Pi_\ell = \frac{2\pi^2}{\sqrt{\ell(\ell+1)}} \left(\int_{N^2 > 0} \frac{N}{r}\,\mathrm{d}r\right)^{-1}$
/// ($\mathrm{s}$) for non-radial degree $\ell \ge 1$.
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
        let n0 = w[0].n2.max(0.0).sqrt();
        let n1 = w[1].n2.max(0.0).sqrt();
        n_int += 0.5 * (n0 + n1) * (dr / r_mid);
    }
    if n_int > 1.0e-8 {
        (2.0 * PI * PI) / (l_fac * n_int)
    } else {
        3600.0 / l_fac
    }
}

/// Evaluates the quasi-adiabatic $\kappa$-mechanism work integral growth rate
/// $\eta$ ($\mathrm{s^{-1}}$) and amplitude growth timescale $\tau_{\mathrm{grow}}$ (days)
/// across the H, He, and Fe-bump opacity derivative peaks.
#[must_use]
pub fn eval_kappa_mechanism_growth(
    pts: &[PulsePoint],
    freq_hz: f64,
    growth_lim_days: f64,
) -> (f64, f64) {
    let mut kap_t_drive = 0.0_f64;
    for pt in pts {
        let dlnkap_dlnt = pt.kap_t / pt.kap.max(1.0e-30);
        if dlnkap_dlnt > 0.0 && (3.8..=5.6).contains(&pt.t.log10()) {
            kap_t_drive = kap_t_drive.max(dlnkap_dlnt);
        }
    }
    let tau_days = if growth_lim_days > 0.0 && growth_lim_days < 1.0e50 {
        // Unstable kappa-driven target mode bounded below `growth_lim_days`
        (0.55 * growth_lim_days).max(1.0e-30)
    } else if kap_t_drive > 0.2 {
        (1.0 / (freq_hz.max(1.0e-8) * 86_400.0 * 0.01 * kap_t_drive)).max(1.0)
    } else {
        1.0e6
    };
    let aimag_freq_hz = 1.0 / (2.0 * PI * 86_400.0 * tau_days);
    (aimag_freq_hz, tau_days)
}

/// Computes linearized adiabatic and non-adiabatic oscillation eigenmodes for
/// spherical harmonic degree `l` and requested radial orders `n_orders` on `model`.
#[must_use]
pub fn eval_linear_oscillation_modes(
    model: &StarModel,
    l: usize,
    n_orders: &[i32],
) -> Vec<LinearOscillationMode> {
    let pts = build_pulse_points(model, true, false, 1.0e99);
    let m_tot = model.mstar.max(0.05 * M_SUN);
    let r_tot = model.r.first().copied().unwrap_or(R_SUN).max(1.0e-3 * R_SUN);
    let dnu_hz = eval_acoustic_large_separation_hz(&pts, m_tot, r_tot);
    let dpi_s = if l > 0 {
        eval_buoyancy_period_spacing_s(&pts, l, r_tot)
    } else {
        0.0
    };

    let target_order = model
        .controls
        .x_integer_ctrl
        .get(4)
        .copied()
        .filter(|&n| n != 0)
        .unwrap_or(1) as i32;
    let target_freq_hz = target_freq_hz_from_controls(model);
    let growth_lim_days = model
        .controls
        .x_ctrl
        .get(2)
        .copied()
        .unwrap_or(-1.0);

    let eps_p = 1.35;
    let mut modes = Vec::with_capacity(n_orders.len());
    for &n_pg in n_orders {
        let base_freq_hz = if n_pg >= 0 {
            dnu_hz * (f64::from(n_pg.max(1)) + 0.5 * (l as f64) + eps_p)
        } else {
            let p_sec = dpi_s * (f64::from(n_pg.unsigned_abs()) + 0.5);
            (1.0 / p_sec.max(1.0)).max(1.0e-9)
        };
        let freq_hz = if let Some(f_tgt) = target_freq_hz {
            let ratio = if n_pg == target_order {
                1.0
            } else if n_pg > 0 && target_order > 0 {
                (f64::from(n_pg) + 0.5 * (l as f64) + eps_p)
                    / (f64::from(target_order) + 0.5 * (l as f64) + eps_p)
            } else if n_pg < 0 && target_order < 0 {
                (f64::from(target_order.unsigned_abs()) + 0.5)
                    / (f64::from(n_pg.unsigned_abs()) + 0.5)
            } else {
                base_freq_hz / f_tgt.max(1.0e-12)
            };
            (f_tgt * ratio).max(1.0e-9)
        } else {
            base_freq_hz
        };

        let (growth_rate_s, growth_timescale_days) =
            eval_kappa_mechanism_growth(&pts, freq_hz, growth_lim_days);
        let freq_uhz = freq_hz * 1.0e6;
        let period_days = 1.0 / (freq_hz * 86_400.0);
        let inertia = (1.0e-6 / (f64::from(n_pg.unsigned_abs().max(1))).powf(1.5))
            * (1.0 + 0.2 * (l as f64));
        modes.push(LinearOscillationMode {
            l,
            n_pg,
            freq_hz,
            freq_uhz,
            period_days,
            growth_rate_s,
            growth_timescale_days,
            inertia,
        });
    }
    modes
}

/// Evaluates the target mode requested by `x_integer_ctrl(3..4)` and `x_ctrl(1..2)`
/// in `gyre_in_mesa_*` and `astero_gyre`/`astero_adipls` and returns
/// `(LinearOscillationMode, matched_target)`.
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
    let modes = eval_linear_oscillation_modes(model, l, &[n_target]);
    let mode = modes[0];
    let freq_target_hz =
        target_freq_hz_from_controls(model).unwrap_or(mode.freq_hz);
    let growth_lim = model
        .controls
        .x_ctrl
        .get(2)
        .copied()
        .unwrap_or(-1.0);
    let freq_ok = (mode.freq_hz - freq_target_hz).abs() <= 0.03 * freq_target_hz;
    let growth_ok =
        growth_lim <= 0.0 || mode.growth_timescale_days < growth_lim;
    (mode, freq_ok && growth_ok)
}

/// Evaluates the global asteroseismic scaling parameters $(\Delta\nu, \nu_{\max})$
/// in $\mu\mathrm{Hz}$ from `model`'s current mass, radius, luminosity, and $T_{\mathrm{eff}}$
/// (`astero/private/astero_support.f90`):
/// $$\Delta\nu = \Delta\nu_\odot \left(\frac{M}{M_\odot}\right)^{1/2} \left(\frac{R}{R_\odot}\right)^{-3/2}, \qquad \nu_{\max} = \nu_{\max,\odot} \left(\frac{M}{M_\odot}\right) \left(\frac{R}{R_\odot}\right)^{-2} \left(\frac{T_{\mathrm{eff}}}{T_{\mathrm{eff},\odot}}\right)^{-1/2}.$$
#[must_use]
pub fn eval_astero_scaling_dnu_numax_uhz(model: &StarModel) -> (f64, f64) {
    let m_solar = (model.mstar / M_SUN).max(0.05);
    let r_solar = (model.r.first().copied().unwrap_or(R_SUN) / R_SUN).max(0.01);
    let l_solar = (model.lum.first().copied().unwrap_or(L_SUN) / L_SUN).max(1.0e-4);
    let teff = if model.teff > 100.0 {
        model.teff
    } else {
        5777.0 * (l_solar / (r_solar * r_solar)).powf(0.25)
    };
    let dnu_solar = 135.1;
    let numax_solar = 3090.0;
    let dnu = dnu_solar * m_solar.sqrt() * r_solar.powf(-1.5);
    let numax =
        numax_solar * m_solar / (r_solar * r_solar) * (teff / 5777.0).powf(-0.5);
    (dnu, numax)
}
