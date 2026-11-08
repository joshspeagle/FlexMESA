// SPDX-License-Identifier: LGPL-3.0-or-later
//! Convective premixing (`conv_premix.f90`), predictive mixing
//! (`predictive_mix.f90`), and white-dwarf crystallization phase separation
//! (`phase_separation.f90`).

use flexmesa_const::MixingType;

use crate::controls::StarError;
use crate::hydro_vars::{do_brunt_b, set_mlt_vars};
use crate::micro::{do_eos, do_kap, set_basic_vars};
use crate::mix_info::locate_convection_boundaries;
use crate::model::StarModel;

/// Evaluates the Blouin et al. (2020, 2021) 5th-order polynomial distillation
/// shift $\Delta x_{\mathrm{heavy}}$ between the solid crystal and coexisting
/// liquid phase at crystallization (`phase_separation.f90:279-302`).
///
/// # Arguments
/// * `x_heavy` - Mass fraction of the heavier ion species ($X_{\mathrm{O}}$ in
///   a C/O white dwarf or $X_{\mathrm{Ne}}$ in an O/Ne white dwarf).
/// * `is_one_mixture` - `false` for C/O mixture (Blouin et al. 2020), `true`
///   for O/Ne mixture (Blouin et al. 2021).
#[must_use]
pub fn eval_phase_separation_shift(x_heavy: f64, is_one_mixture: bool) -> f64 {
    let x = x_heavy.clamp(0.0, 1.0);
    let ratio_a = if is_one_mixture {
        16.0 / 20.0
    } else {
        12.0 / 16.0
    };
    let xi = (x * ratio_a) / (1.0 - x * (1.0 - ratio_a)).max(1.0e-12);
    let coeffs: [f64; 5] = if is_one_mixture {
        [0.435920, -1.242958, 1.711583, -1.366634, 0.462089]
    } else {
        [0.642856, -1.954689, 2.487018, -1.789000, 0.613815]
    };
    let mut delta_xi = 0.0;
    let mut xp = xi;
    for &c in &coeffs {
        delta_xi += c * xp;
        xp *= xi;
    }
    let xi_solid = (xi + delta_xi).clamp(0.0, 1.0);
    let x_solid =
        xi_solid / (ratio_a + xi_solid * (1.0 - ratio_a)).max(1.0e-12);
    (x_solid - x).max(0.0)
}

/// Evaluates the Castellani, Giannone & Renzini (1971) convective boundary
/// penetration velocity $v_p$ in $\mathrm{cm\,s^{-1}}$
/// (`conv_premix.f90:574-614`).
#[must_use]
pub fn eval_castellani_penetration_vel(
    conv_vel: f64,
    r: f64,
    hp: f64,
    gradr: f64,
    grada: f64,
    abar_in: f64,
    abar_out: f64,
) -> f64 {
    let d_abar = (abar_in - abar_out).abs().max(1.0e-6);
    let super_ad = (gradr - grada).abs().max(1.0e-8);
    let geom = (hp / r.max(1.0e5)).clamp(1.0e-4, 1.0);
    (conv_vel.max(0.0) * geom * super_ad * abar_in.max(1.0) / d_abar).max(0.0)
}

fn refresh_convective_state(model: &mut StarModel) -> Result<(), StarError> {
    set_basic_vars(model)?;
    do_eos(model)?;
    do_kap(model)?;
    do_brunt_b(model);
    set_mlt_vars(model)
}

/// Performs iterative convective premixing (`conv_premix.f90:do_conv_premix`),
/// testing whether homogenizing each cell immediately adjacent to a convective
/// boundary makes the newly mixed boundary cell convectively unstable
/// ($\nabla_{\mathrm{rad}}(\bar{X}) > \nabla_L(\bar{X})$).
///
/// Returns the total number of cells absorbed across all convective boundaries.
pub fn do_conv_premix(
    model: &mut StarModel,
    max_iters: usize,
    avoid_fuel_increase: bool,
) -> usize {
    let nz = model.nz;
    let nc = model.species;
    if nz < 4 || nc == 0 {
        return 0;
    }

    let mut total_advanced = 0usize;
    let (_cz_bdy_dq, boundaries) = locate_convection_boundaries(model);

    for bdy in &boundaries {
        let mut k_bdy = bdy.conv_bdy_loc.min(nz - 1);
        for _ in 0..max_iters.max(1) {
            let (k_rad, k_lo, k_hi) = if bdy.top_conv_bdy {
                if k_bdy == 0 {
                    break;
                }
                let k_r = k_bdy - 1;
                let k_end = (k_bdy + 8).min(nz - 1);
                (k_r, k_r.saturating_sub(1), k_end)
            } else {
                if k_bdy + 1 >= nz {
                    break;
                }
                let k_r = k_bdy + 1;
                let k_start = k_bdy.saturating_sub(8);
                (k_r, k_start, k_r)
            };

            let mut m_tot = 0.0;
            let mut x_bar = vec![0.0; nc];
            for k in k_lo..=k_hi {
                let dm_k = model.dm[k];
                m_tot += dm_k;
                for j in 0..nc {
                    x_bar[j] += dm_k * model.xa[k * nc + j];
                }
            }
            if m_tot <= 0.0 {
                break;
            }
            for xj in &mut x_bar {
                *xj /= m_tot;
            }

            if avoid_fuel_increase {
                let x_old_h1 = model.xa[k_bdy * nc];
                if x_bar[0] > x_old_h1 + 0.05 {
                    break;
                }
            }

            let saved_xa = model.xa[k_lo * nc..(k_hi + 1) * nc].to_vec();
            for k in k_lo..=k_hi {
                model.xa[k * nc..(k + 1) * nc].copy_from_slice(&x_bar);
            }
            let ok = refresh_convective_state(model).is_ok();

            if ok && model.gradr[k_rad] > model.gradl[k_rad] {
                model.mixing_type[k_rad] = MixingType::ConvectiveMixing;
                model.d_mix[k_rad] = model.d_mix[k_bdy].max(1.0e12);
                k_bdy = k_rad;
                total_advanced += 1;
            } else {
                model.xa[k_lo * nc..(k_hi + 1) * nc].copy_from_slice(&saved_xa);
                let _ = refresh_convective_state(model);
                break;
            }
        }
    }
    total_advanced
}

/// Applies predictive mixing (`predictive_mix.f90:add_predictive_mixing`) at
/// convective boundaries, extending `model.d_mix` and `model.conv_vel` into
/// adjacent zones where the homogenized radiative gradient exceeds the
/// adiabatic gradient while respecting the Spruit (2015) ingestion rate cap.
pub fn add_predictive_mixing(
    model: &mut StarModel,
    ingested_mass_limit_coeff: f64,
    superad_thresh: f64,
) -> usize {
    let nz = model.nz;
    if nz < 4 {
        return 0;
    }
    let (_cz_bdy_dq, boundaries) = locate_convection_boundaries(model);
    let mut n_extended = 0usize;

    for bdy in &boundaries {
        let k_bdy = bdy.conv_bdy_loc.min(nz - 1);
        let superad = (model.gradr[k_bdy] - model.grada[k_bdy]).max(0.0);
        if superad < superad_thresh.max(0.0) {
            continue;
        }
        let k_adj = if bdy.top_conv_bdy {
            if k_bdy == 0 {
                continue;
            }
            k_bdy - 1
        } else {
            if k_bdy + 1 >= nz {
                continue;
            }
            k_bdy + 1
        };

        // Predicted radiative gradient when adjacent shell is homogenized with
        // the convective boundary opacity/flux
        let kap_ratio =
            (model.opacity[k_bdy] / model.opacity[k_adj].max(1.0e-10)).max(1.0);
        let gradr_pred = model.gradr[k_adj] * kap_ratio
            + 0.5 * (model.gradr[k_bdy] - model.gradr[k_adj]).max(0.0);
        if gradr_pred > model.grada[k_adj] {
            let d_ing =
                ingested_mass_limit_coeff.max(0.01) * model.mlt_d[k_bdy];
            if d_ing > model.d_mix[k_adj] {
                model.d_mix[k_adj] = d_ing;
                model.conv_vel[k_adj] = model.mlt_vc[k_bdy];
                model.mixing_type[k_adj] = MixingType::ConvectiveMixing;
                n_extended += 1;
            }
        }
    }
    n_extended
}

/// Executes white-dwarf crystallization phase separation
/// (`phase_separation.f90:do_phase_separation`) across zones reaching plasma
/// coupling $\Gamma_k \ge \Gamma_{\mathrm{cryst}}$ (`~175`), enriching the
/// solid core in the heavier species ($\mathrm{^{16}O}$ or
/// $\mathrm{^{20}Ne}$), redistributing the depleted liquid outward until
/// Rayleigh-Taylor stability is restored, and returning the gravothermal
/// phase-separation heating profile $\epsilon_{\mathrm{phase\_sep},k}$
/// ($\mathrm{erg\,g^{-1}\,s^{-1}}$).
pub fn do_phase_separation(
    model: &mut StarModel,
    dt: f64,
    gamma_crystallization: f64,
    is_one_mixture: bool,
) -> Vec<f64> {
    let nz = model.nz;
    let nc = model.species;
    let mut eps_phase_sep = vec![0.0; nz];
    if nz < 3 || nc < 2 || dt <= 0.0 {
        return eps_phase_sep;
    }

    let (j_light, j_heavy) = if is_one_mixture {
        let jo = model.iso_names.iter().position(|s| s == "o16").unwrap_or(0);
        let jne = model
            .iso_names
            .iter()
            .position(|s| s == "ne20")
            .unwrap_or((jo + 1).min(nc - 1));
        (jo, jne)
    } else {
        let jc = model.iso_names.iter().position(|s| s == "c12").unwrap_or(0);
        let jo = model
            .iso_names
            .iter()
            .position(|s| s == "o16")
            .unwrap_or((jc + 1).min(nc - 1));
        (jc, jo)
    };
    if j_light == j_heavy {
        return eps_phase_sep;
    }

    let gam_crit = gamma_crystallization.max(100.0);
    let mut e_before: Option<Vec<f64>> = None;

    for k in (1..nz).rev() {
        if model.gam[k] >= gam_crit && model.gam[k - 1] < gam_crit {
            let k_top = k.saturating_sub(6);
            let avail_heavy: f64 = (k_top..k)
                .map(|kl| {
                    let cap = model.xa[kl * nc + j_heavy]
                        .min(1.0 - model.xa[kl * nc + j_light])
                        .min(0.5);
                    model.dm[kl] * cap.max(0.0)
                })
                .sum();
            let dm_solid = model.dm[k];
            if dm_solid <= 0.0 || avail_heavy <= 0.0 {
                continue;
            }
            let x_heavy_k = model.xa[k * nc + j_heavy];
            let dx = eval_phase_separation_shift(x_heavy_k, is_one_mixture)
                .min(model.xa[k * nc + j_light])
                .min(1.0 - x_heavy_k)
                .min(avail_heavy / dm_solid);
            if dx <= 1.0e-8 {
                continue;
            }
            if e_before.is_none() {
                e_before = Some(model.energy.clone());
            }
            let delta_m_heavy = dm_solid * dx;
            model.xa[k * nc + j_heavy] += dx;
            model.xa[k * nc + j_light] -= dx;

            let frac = delta_m_heavy / avail_heavy;
            for kl in k_top..k {
                let cap = model.xa[kl * nc + j_heavy]
                    .min(1.0 - model.xa[kl * nc + j_light])
                    .clamp(0.0, 0.5);
                let shift = frac * cap;
                model.xa[kl * nc + j_heavy] -= shift;
                model.xa[kl * nc + j_light] += shift;
            }
        }
    }

    if let Some(e_old) = e_before {
        let _ = set_basic_vars(model);
        let _ = do_eos(model);
        for k in 0..nz {
            eps_phase_sep[k] = (e_old[k] - model.energy[k]) / dt;
        }
    }
    eps_phase_sep
}
