// SPDX-License-Identifier: LGPL-3.0-or-later
//! Global asteroseismic scaling relations, acoustic transit integrals,
//! gravity-mode period spacings $\Delta\Pi_1$, and acoustic glitch depths.
//!
//! Corresponds to `star/private/report.f90:286-363`,
//! `star/private/star_utils.f90:920-1022`,
//! `star/private/profile_getval.f90:671-683`, and
//! `astero/private/astero_support.f90:186-207, 624-643`.

use flexmesa_const::{LSUN, MSUN, PI, RSUN};
use flexmesa_math::pow;

/// Solar reference frequency of maximum power $\nu_{\max,\odot}$ ($\mu\mathrm{Hz}$)
/// (`star/defaults/controls.defaults:7770`, Lund et al. 2017).
pub const NU_MAX_SUN: f64 = 3078.0;

/// Solar reference large frequency separation $\Delta\nu_\odot$ ($\mu\mathrm{Hz}$)
/// (`star/defaults/controls.defaults:7771`).
pub const DELTA_NU_SUN: f64 = 134.91;

/// Solar reference effective temperature $T_{\mathrm{eff},\odot}^{\mathrm{astero}}$ (K)
/// (`star/defaults/controls.defaults:7772`, IAU 2015 Resolution B3).
pub const TEFF_SUN_ASTERO: f64 = 5772.0;

/// Evaluates the cumulative acoustic radius profile $\tau(r_k)$ (s), cumulative
/// acoustic depth profile $\tau_{\mathrm{depth}}(r_k)$ (s) from the surface, and
/// total photospheric acoustic radius $\tau_0 = \int_0^{R_{\mathrm{phot}}} dr / c_s$ (s).
///
/// Input slices `r` (cm) and `csound` ($\mathrm{cm\,s^{-1}}$) are ordered
/// surface-to-center (`0..nz`).
///
/// Corresponds to `report.f90:286-340` and `profile_getval.f90:671-683`.
#[must_use]
pub fn eval_acoustic_profile(
    r: &[f64],
    csound: &[f64],
    r_center: f64,
    r_phot: f64,
) -> (Vec<f64>, Vec<f64>, f64) {
    let nz = r.len();
    if nz == 0 {
        return (Vec::new(), Vec::new(), 0.0);
    }
    let mut dr_div_cs = vec![0.0_f64; nz];
    for k in 0..nz {
        let dr = if k + 1 < nz {
            (r[k] - r[k + 1]).max(0.0)
        } else {
            (r[k] - r_center).max(0.0)
        };
        dr_div_cs[k] = dr / csound[k].max(1.0);
    }

    let mut acoustic_radius = vec![0.0_f64; nz];
    let mut running_r = 0.0_f64;
    for k in (0..nz).rev() {
        running_r += dr_div_cs[k];
        acoustic_radius[k] = running_r;
    }

    let mut acoustic_depth = vec![0.0_f64; nz];
    let mut running_d = 0.0_f64;
    for k in 0..nz {
        acoustic_depth[k] = running_d;
        running_d += dr_div_cs[k];
    }

    let tau_0 = if r_phot >= r[0] {
        acoustic_radius[0] + (r_phot - r[0]) / csound[0].max(1.0)
    } else {
        let mut k_phot = 1usize;
        while k_phot < nz && r_phot <= r[k_phot] {
            k_phot += 1;
        }
        if k_phot >= nz {
            (r_phot - r_center).max(0.0) / csound[nz - 1].max(1.0)
        } else {
            acoustic_radius[k_phot]
                + (r_phot - r[k_phot]) / csound[k_phot - 1].max(1.0)
        }
    };

    (acoustic_radius, acoustic_depth, tau_0)
}

/// Evaluates the asymptotic large frequency separation $\Delta\nu_{\mathrm{asymp}}$
/// ($\mu\mathrm{Hz}$) from the photospheric acoustic radius $\tau_0$ (s):
///
/// $$\Delta\nu_{\mathrm{asymp}} = \frac{10^6}{2\,\tau_0}$$
///
/// Corresponds to `report.f90:343`.
#[inline]
#[must_use]
pub fn eval_delta_nu_asymptotic(photosphere_acoustic_r: f64) -> f64 {
    1.0e6 / (2.0 * photosphere_acoustic_r.max(1.0e-30))
}

/// Evaluates the solar-scaled large frequency separation $\Delta\nu_{\mathrm{scale}}$
/// ($\mu\mathrm{Hz}$) from stellar mass $M$ (g), effective temperature $T_{\mathrm{eff}}$ (K),
/// and photospheric luminosity $L_{\mathrm{phot}}$ ($\mathrm{erg\,s^{-1}}$):
///
/// $$\Delta\nu_{\mathrm{scale}} = \Delta\nu_\odot \left(\frac{M}{M_\odot}\right)^{1/2}
///   \left(\frac{T_{\mathrm{eff}}}{T_{\mathrm{eff},\odot}}\right)^3
///   \left(\frac{L_{\mathrm{phot}}}{L_\odot}\right)^{-3/4}$$
///
/// Corresponds to `report.f90:345-347`.
#[must_use]
pub fn eval_delta_nu_scaling(m_g: f64, teff_k: f64, l_phot_erg_s: f64) -> f64 {
    DELTA_NU_SUN
        * (m_g / MSUN).sqrt()
        * pow(teff_k / TEFF_SUN_ASTERO, 3.0)
        * pow(l_phot_erg_s / LSUN, -0.75)
}

/// Evaluates the solar-scaled large frequency separation $\Delta\nu$ ($\mu\mathrm{Hz}$)
/// directly from mass $M$ (g) and photospheric radius $R_{\mathrm{phot}}$ (cm):
///
/// $$\Delta\nu = \Delta\nu_\odot \left(\frac{M}{M_\odot}\right)^{1/2}
///   \left(\frac{R_{\mathrm{phot}}}{R_\odot}\right)^{-3/2}$$
#[must_use]
pub fn eval_delta_nu_radius_scaling(m_g: f64, r_phot_cm: f64) -> f64 {
    DELTA_NU_SUN * (m_g / MSUN).sqrt() * pow(r_phot_cm / RSUN, -1.5)
}

/// Evaluates the solar-scaled frequency of maximum oscillation power $\nu_{\max}$
/// ($\mu\mathrm{Hz}$) from mass $M$ (g), radius $R_{\mathrm{phot}}$ (cm), and
/// effective temperature $T_{\mathrm{eff}}$ (K):
///
/// $$\nu_{\max} = \nu_{\max,\odot} \frac{M / M_\odot}{(R_{\mathrm{phot}} / R_\odot)^2
///   \sqrt{T_{\mathrm{eff}} / T_{\mathrm{eff},\odot}}}$$
///
/// Corresponds to `report.f90:353-354`.
#[must_use]
pub fn eval_nu_max(m_g: f64, r_phot_cm: f64, teff_k: f64) -> f64 {
    let r_rsun = r_phot_cm / RSUN;
    NU_MAX_SUN * (m_g / MSUN)
        / (r_rsun * r_rsun * (teff_k / TEFF_SUN_ASTERO).sqrt())
}

/// Evaluates the isothermal surface acoustic cutoff frequency $\nu_{\mathrm{ac}}$
/// ($\mu\mathrm{Hz}$) at the outer boundary cell:
///
/// $$\nu_{\mathrm{ac}} = \frac{0.25 \times 10^6}{\pi} g_1 \sqrt{\frac{\Gamma_{1,1} \rho_1}{P_1}}$$
///
/// Corresponds to `report.f90:355-356`.
#[must_use]
pub fn eval_nu_ac(
    g_surf_cgs: f64,
    gamma1_surf: f64,
    rho_surf: f64,
    peos_surf: f64,
) -> f64 {
    (0.25e6 / PI)
        * g_surf_cgs
        * (gamma1_surf * rho_surf / peos_surf.max(1.0e-30)).sqrt()
}

/// Evaluates the traditional asymptotic dipole ($l=1$) gravity-mode period
/// spacing $\Delta\Pi_1$ (s) integrated across all radiative zones with
/// $N^2 > 0$:
///
/// $$\Delta\Pi_1 = \frac{\sqrt{2}\,\pi^2}{\displaystyle\sum_{k=2,\,N_k^2 > 0}^{N_z}
///   \sqrt{N_k^2}\,\frac{r_{\mathrm{mid},k-1} - r_{\mathrm{mid},k}}{r_k}}$$
///
/// Corresponds to `get_delta_Pg_traditional` in `star_utils.f90:920-958`.
#[must_use]
pub fn eval_delta_pg_traditional(
    r: &[f64],
    r_mid: &[f64],
    brunt_n2: &[f64],
) -> f64 {
    let nz = r.len();
    if nz < 2 {
        return 0.0;
    }
    let mut integral = 0.0_f64;
    for k in 1..nz {
        if brunt_n2[k] > 0.0 {
            let dr = (r_mid[k - 1] - r_mid[k]).max(0.0);
            integral += brunt_n2[k].sqrt() * dr / r[k].max(1.0);
        }
    }
    if integral <= 0.0 {
        0.0
    } else {
        2.0_f64.sqrt() * PI * PI / integral
    }
}

/// Evaluates the Bildsten et al. (2012) frequency-dependent dipole ($l=1$)
/// gravity-mode period spacing $\Delta\Pi_1$ (s) with inner WKB tunneling
/// truncation ($I_{\mathrm{decay}} > 0.5$).
///
/// Corresponds to `get_delta_Pg_bildsten2012` in `star_utils.f90:960-1022`.
#[must_use]
pub fn eval_delta_pg_bildsten2012(
    r: &[f64],
    r_mid: &[f64],
    brunt_n2: &[f64],
    csound_face: &[f64],
    mode_freq_uhz: f64,
) -> f64 {
    let nz = r.len();
    if nz < 2 || mode_freq_uhz <= 0.0 {
        return 0.0;
    }
    let omega2 = (2.0 * PI * mode_freq_uhz * 1.0e-6).powi(2);
    let mut entered_g_cavity = false;
    let mut integral_g = 0.0_f64;
    let mut integral_decay = 0.0_f64;

    for k in 1..nz {
        let rk = r[k].max(1.0);
        let cs2 = csound_face[k] * csound_face[k];
        let sl2 = 2.0 * cs2 / (rk * rk);
        if !entered_g_cavity && omega2 >= sl2 {
            continue;
        }
        let dr = (r_mid[k - 1] - r_mid[k]).max(0.0);
        let n2 = brunt_n2[k];
        if n2 > omega2 && omega2 < sl2 {
            entered_g_cavity = true;
            integral_g += n2.sqrt() * dr / rk;
        } else if entered_g_cavity {
            let kr2 = (1.0 - n2 / omega2) * (1.0 - sl2 / omega2) * omega2
                / cs2.max(1.0);
            if kr2 < 0.0 {
                integral_decay += (-kr2).sqrt() * dr;
                if integral_decay > 0.5 {
                    break;
                }
            }
        }
    }
    if integral_g <= 0.0 {
        0.0
    } else {
        2.0_f64.sqrt() * PI * PI / integral_g
    }
}

/// Locates the base of the convective envelope $R_{\mathrm{BCZ}} / R_\odot$,
/// its acoustic depth $\tau_{\mathrm{BCZ}}$ (s), and the acoustic depth of the
/// He II ionization $\Gamma_1$-minimum $\tau_{\mathrm{HeII}}$ (s).
#[must_use]
pub fn eval_glitch_acoustic_depths(
    r: &[f64],
    csound: &[f64],
    gradr: &[f64],
    grada: &[f64],
    gamma1: &[f64],
    t: &[f64],
    r_center: f64,
) -> (f64, f64, f64) {
    let nz = r.len();
    if nz < 2 {
        return (0.0, 0.0, 0.0);
    }
    let (_, tau_depth, _) = eval_acoustic_profile(r, csound, r_center, r[0]);

    let mut r_bcz_rsun = 0.0_f64;
    let mut tau_bcz = 0.0_f64;
    for k in 0..(nz - 1) {
        let is_conv_k = gradr[k] > grada[k];
        let is_conv_next = gradr[k + 1] > grada[k + 1];
        if is_conv_k
            && !is_conv_next
            && r[k + 1] > 0.20 * RSUN
            && r[k] < 0.98 * r[0]
        {
            r_bcz_rsun = r[k] / RSUN;
            tau_bcz = tau_depth[k];
            break;
        }
    }

    let mut min_g1 = f64::INFINITY;
    let mut tau_heii = 0.0_f64;
    for k in 0..nz {
        if t[k] >= 2.5e4 && t[k] <= 8.0e4 && gamma1[k] < min_g1 {
            min_g1 = gamma1[k];
            tau_heii = tau_depth[k];
        }
    }
    (r_bcz_rsun, tau_bcz, tau_heii)
}

/// Evaluates unweighted ordinary least-squares radial $\Delta\nu_{\mathrm{model}}$
/// from radial orders $n_i$ and model radial frequencies $\nu_{0,i}^{\mathrm{model}}$
/// (`astero_support.f90:196-207`).
#[must_use]
pub fn eval_radial_delta_nu_ols(n_orders: &[f64], model_freqs: &[f64]) -> f64 {
    let m = n_orders.len().min(model_freqs.len());
    if m < 2 {
        return 0.0;
    }
    let mean_n = n_orders[..m].iter().sum::<f64>() / (m as f64);
    let mean_f = model_freqs[..m].iter().sum::<f64>() / (m as f64);
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for i in 0..m {
        let dn = n_orders[i] - mean_n;
        num += (model_freqs[i] - mean_f) * dn;
        den += dn * dn;
    }
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}
