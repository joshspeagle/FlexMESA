// SPDX-License-Identifier: LGPL-3.0-or-later
//! Convective overshooting (`star/private/overshoot.f90`,
//! `star/private/overshoot_utils.f90`, `star/private/overshoot_exp.f90`, and
//! `star/private/overshoot_step.f90`).

use flexmesa_const::{MixingType, FOUR_PI, M_SUN};

use crate::controls::StarError;
use crate::mix_info::ConvBoundary;
use crate::model::StarModel;

/// Functional form of convective overshooting (`overshoot_scheme` in MESA
/// `26.4.1`, `star/defaults/controls.defaults:2702`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OvershootScheme {
    /// Herwig (2000) exponential decay (`'exponential'`).
    Exponential,
    /// Constant step overshoot (`'step'`).
    Step,
}

/// Configuration for a single explicit convective boundary overshooting rule
/// used by standalone [`apply_overshooting`] callers.
#[derive(Debug, Clone, PartialEq)]
pub struct OvershootConfig {
    /// Functional decay profile (`Exponential` or `Step`).
    pub scheme: OvershootScheme,
    /// True to apply at top (outward) convective boundaries; false for bottom
    /// (inward) convective boundaries.
    pub top_conv_bdy: bool,
    /// Primary overshoot scale-height fraction $f$.
    pub f: f64,
    /// Anchor step-back scale-height fraction $f_0$ inside the CZ.
    pub f0: f64,
    /// Additive diffusivity floor $D_0$ ($\mathrm{cm^2\,s^{-1}}$) at $r_{\mathrm{ob}}$.
    pub d0: f64,
    /// Multiplicative diffusivity scale $\Delta_0$ at $r_{\mathrm{ob}}$.
    pub delta0: f64,
    /// Minimum overshoot diffusivity threshold $D_{\mathrm{ov,min}}$
    /// ($\mathrm{cm^2\,s^{-1}}$).
    pub d_min: f64,
}

impl Default for OvershootConfig {
    fn default() -> Self {
        Self {
            scheme: OvershootScheme::Exponential,
            top_conv_bdy: true,
            f: 0.014,
            f0: 0.004,
            d0: 0.0,
            delta0: 1.0,
            d_min: 1.0e-2,
        }
    }
}

impl OvershootConfig {
    /// Evaluates the dimensionless overshoot decay factor at distance `dr` from
    /// $r_{\mathrm{ob}}$ given boundary pressure scale height `hp_cb`.
    #[must_use]
    #[inline]
    pub fn eval_factor(&self, dr: f64, hp_cb: f64) -> f64 {
        if self.f <= 0.0 || hp_cb <= 0.0 {
            return 0.0;
        }
        let dr_step = self.f * hp_cb;
        match self.scheme {
            OvershootScheme::Exponential => {
                flexmesa_math::exp(-2.0 * dr / dr_step)
            }
            OvershootScheme::Step => {
                if dr < dr_step {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

/// Returns the 0-based index `k` of the cell containing the `i`-th convective
/// boundary (`eval_conv_bdy_k` in `star/private/overshoot_utils.f90:35-61`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if `i` is out of range or `k` falls
/// outside `0..nz-1`.
pub fn eval_conv_bdy_k(model: &StarModel, i: usize) -> Result<usize, StarError> {
    let mx = &model.aux.mixing;
    if i >= mx.conv_bdy_loc.len() || i >= mx.top_conv_bdy.len() {
        return Err(StarError::InvalidInput(format!(
            "eval_conv_bdy_k: boundary index {i} out of range (len={})",
            mx.conv_bdy_loc.len()
        )));
    }
    let loc = mx.conv_bdy_loc[i];
    let k = if mx.top_conv_bdy[i] {
        loc
    } else if loc == 0 {
        return Err(StarError::InvalidInput(format!(
            "eval_conv_bdy_k: bottom boundary {i} has conv_bdy_loc == 0"
        )));
    } else {
        loc - 1
    };
    if k + 1 >= model.nz {
        return Err(StarError::InvalidInput(format!(
            "Invalid cell for convective boundary: i={i}, k={k}, nz={}",
            model.nz
        )));
    }
    Ok(k)
}

/// Evaluates the sub-cell radius $r_{\mathrm{cb}}$ in cm at the `i`-th
/// convective boundary (`eval_conv_bdy_r` in
/// `star/private/overshoot_utils.f90:64-104`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if `eval_conv_bdy_k` fails or
/// `w = cz_bdy_dq[k] / dq[k]` lies outside `[0, 1]`.
pub fn eval_conv_bdy_r(model: &StarModel, i: usize) -> Result<f64, StarError> {
    let k = eval_conv_bdy_k(model, i)?;
    let w = model.aux.mixing.cz_bdy_dq[k] / model.dq[k];
    if !(0.0..=1.0).contains(&w) {
        return Err(StarError::InvalidInput(format!(
            "Invalid weight for convective boundary: i={i}, w={w}"
        )));
    }
    let r_o = model.r[k];
    let r_i = model.r[k + 1];
    Ok(flexmesa_math::pow(
        w * r_i * r_i * r_i + (1.0 - w) * r_o * r_o * r_o,
        1.0 / 3.0,
    ))
}

/// Evaluates the sub-cell pressure scale height $H_{P,\mathrm{cb}}$ in cm at
/// the `i`-th convective boundary via 3-cell piecewise-linear log interpolation
/// (`eval_conv_bdy_Hp` in `star/private/overshoot_utils.f90:107-226`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if boundary geometry or 3-cell
/// interpolation fails.
pub fn eval_conv_bdy_hp(model: &StarModel, i: usize) -> Result<f64, StarError> {
    let k = eval_conv_bdy_k(model, i)?;
    let r = eval_conv_bdy_r(model, i)?;
    if k == 0 || k + 1 >= model.nz {
        return Err(StarError::InvalidInput(format!(
            "eval_conv_bdy_hp: boundary cell k={k} lacks 3-cell stencil (nz={})",
            model.nz
        )));
    }

    let k_o = k - 1;
    let k_m = k;
    let k_i = k + 1;

    let x0 = model.dq[k_o] / 2.0;
    let x1 = model.dq[k_o] + model.dq[k_m] / 2.0;
    let x2 = model.dq[k_o] + model.dq[k_m] + model.dq[k_i] / 2.0;
    let x = model.dq[k_o] + model.aux.mixing.cz_bdy_dq[k];

    let (a0, a1, a2) =
        flexmesa_num::two_piece_linear_coeffs(x, x0, x1, x2).map_err(|e| {
            StarError::InvalidInput(format!(
                "eval_conv_bdy_hp two_piece_linear_coeffs failed at i={i}, k={k}: {e}"
            ))
        })?;

    let p = flexmesa_math::exp(
        a0 * model.ln_peos[k_o]
            + a1 * model.ln_peos[k_m]
            + a2 * model.ln_peos[k_i],
    );
    let rho = flexmesa_math::exp(
        a0 * model.lnd[k_o] + a1 * model.lnd[k_m] + a2 * model.lnd[k_i],
    );
    let mut hp = p
        / (rho
            * model.cgrav[k_m]
            * (model.m_center + model.xmstar * model.aux.mixing.conv_bdy_q[i])
            / (r * r));

    let mw = &model.controls.mix_wind;
    if mw.limit_overshoot_hp_using_size_of_convection_zone {
        let mx = &model.aux.mixing;
        let (r_top, r_bot) = if mx.top_conv_bdy[i] {
            let rb = if i == 0 {
                model.r_center
            } else {
                if mx.top_conv_bdy[i - 1] {
                    return Err(StarError::InvalidInput(format!(
                        "Double top boundary in overshoot; i={i}"
                    )));
                }
                model.r[mx.conv_bdy_loc[i - 1]]
            };
            (model.r[k], rb)
        } else {
            let rb = model.r[k + 1];
            let rt = if i + 1 == mx.conv_bdy_loc.len() {
                model.r[0]
            } else {
                if !mx.top_conv_bdy[i + 1] {
                    return Err(StarError::InvalidInput(format!(
                        "Double bottom boundary in overshoot; i={i}"
                    )));
                }
                model.r[mx.conv_bdy_loc[i + 1]]
            };
            (rt, rb)
        };
        let dr = r_top - r_bot;
        if mw.overshoot_alpha > 0.0 {
            if mw.overshoot_alpha * hp > dr {
                hp = dr / mw.overshoot_alpha;
            }
        } else {
            let alpha_k = if mx.alpha_mlt.get(k).copied().unwrap_or(0.0) > 0.0 {
                mx.alpha_mlt[k]
            } else {
                model.controls.mixing_length_alpha
            };
            // Verbatim `star/private/overshoot_utils.f90:219`.
            if alpha_k * hp > dr {
                hp = dr / model.controls.mixing_length_alpha;
            }
        }
    }

    Ok(hp)
}

/// Evaluates overshoot-boundary anchor parameters `(k_ob, r_ob, d_ob, vc_ob)`
/// stepped back by `f0 * Hp_cb` into the convection zone
/// (`eval_over_bdy_params` in `star/private/overshoot_utils.f90:229-374`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if boundary parameters or radius
/// bracketing fail.
pub fn eval_over_bdy_params(
    model: &StarModel,
    i: usize,
    f0: f64,
) -> Result<(usize, f64, f64, f64), StarError> {
    let nz = model.nz;
    let k_cb = eval_conv_bdy_k(model, i)?;
    let r_cb = eval_conv_bdy_r(model, i)?;
    let hp_cb = eval_conv_bdy_hp(model, i)?;

    let (k, r) = if model.aux.mixing.top_conv_bdy[i] {
        let mut r_target = r_cb - f0 * hp_cb;
        if r_target <= model.r[nz - 1] {
            r_target = model.r[nz - 1];
            (nz - 2, r_target)
        } else {
            let mut k_found = nz - 2;
            for kk in k_cb..=(nz - 2) {
                if model.r[kk + 1] <= r_target {
                    k_found = kk;
                    break;
                }
            }
            (k_found, r_target)
        }
    } else {
        let mut r_target = r_cb + f0 * hp_cb;
        if r_target >= model.r[0] {
            r_target = model.r[0];
            (0, r_target)
        } else {
            let mut k_found = 0;
            for kk in (0..=k_cb).rev() {
                if model.r[kk] > r_target {
                    k_found = kk;
                    break;
                }
            }
            (k_found, r_target)
        }
    };

    if !(model.r[k + 1] <= r && model.r[k] >= r) {
        return Err(StarError::InvalidInput(format!(
            "r_ob not correctly bracketed: r(k+1)={}, r={r}, r(k)={}",
            model.r[k + 1],
            model.r[k]
        )));
    }

    let rk = model.r[k];
    let rkp1 = model.r[k + 1];
    let rk3 = rk * rk * rk;
    let rkp1_3 = rkp1 * rkp1 * rkp1;
    let r3 = r * r * r;
    let w = (rk3 - r3) / (rk3 - rkp1_3);

    let lam_k = model
        .aux
        .mixing
        .mlt_mixing_length
        .get(k)
        .copied()
        .filter(|&v| v > 0.0)
        .unwrap_or_else(|| {
            model.controls.mixing_length_alpha * model.scale_height[k]
        });
    let lam_kp1 = model
        .aux
        .mixing
        .mlt_mixing_length
        .get(k + 1)
        .copied()
        .filter(|&v| v > 0.0)
        .unwrap_or_else(|| {
            model.controls.mixing_length_alpha * model.scale_height[k + 1]
        });
    let lambda = (1.0 - w) * lam_k + w * lam_kp1;

    let vc_k = model.conv_vel[k];
    let vc_kp1 = model.conv_vel[k + 1];
    let vc = if vc_k != 0.0 && vc_kp1 != 0.0 {
        (1.0 - w) * vc_k + w * vc_kp1
    } else if vc_k != 0.0 && vc_kp1 == 0.0 {
        let w2 = if rk == r_cb {
            0.0
        } else {
            (rk3 - r3) / (rk3 - r_cb * r_cb * r_cb)
        };
        (1.0 - w2) * vc_k
    } else if vc_k == 0.0 && vc_kp1 != 0.0 {
        let w2 = if rkp1 == r_cb {
            0.0
        } else {
            (r_cb * r_cb * r_cb - r3) / (r_cb * r_cb * r_cb - rkp1_3)
        };
        w2 * vc_kp1
    } else {
        0.0
    };

    let d = vc * lambda / 3.0;
    Ok((k, r, d, vc))
}

#[allow(clippy::too_many_arguments)]
fn eval_overshoot_profile(
    model: &StarModel,
    i: usize,
    scheme: OvershootScheme,
    mut f: f64,
    mut f0: f64,
    d0: f64,
    delta0: f64,
    mass_full_on: f64,
    mass_full_off: f64,
    d_min: f64,
    d_out: &mut [f64],
    vc_out: &mut [f64],
) -> Result<(usize, usize), StarError> {
    if f <= 0.0 || f0 <= 0.0 {
        return Err(StarError::InvalidInput(String::from(
            "for exponential/step overshooting, must set f and f0 > 0",
        )));
    }

    let star_mass = model.mstar / M_SUN;
    if star_mass < mass_full_on {
        if star_mass > mass_full_off {
            let w = (star_mass - mass_full_off) / (mass_full_on - mass_full_off);
            let factor = 0.5 * (1.0 - flexmesa_math::cospi(w));
            f *= factor;
            f0 *= factor;
        } else {
            f = 0.0;
            f0 = 0.0;
        }
    }

    let hp_cb = eval_conv_bdy_hp(model, i)?;
    let (k_ob, r_ob, d_ob, vc_ob) = eval_over_bdy_params(model, i, f0)?;

    let outward = model.aux.mixing.top_conv_bdy[i];
    let k_a = if outward { k_ob } else { k_ob + 1 };
    let mut k_b = if outward { 0 } else { model.nz - 1 };
    let dk: isize = if outward { -1 } else { 1 };

    let mut k_i = k_a as isize;
    let k_end_i = k_b as isize;
    while (dk < 0 && k_i >= k_end_i) || (dk > 0 && k_i <= k_end_i) {
        let k = k_i as usize;
        let r = model.r[k];
        let dr = if outward { r - r_ob } else { r_ob - r };
        let factor = match scheme {
            OvershootScheme::Exponential => {
                if f > 0.0 {
                    flexmesa_math::exp(-2.0 * dr / (f * hp_cb))
                } else {
                    0.0
                }
            }
            OvershootScheme::Step => {
                if dr < f * hp_cb {
                    1.0
                } else {
                    0.0
                }
            }
        };
        d_out[k] = (d0 + delta0 * d_ob) * factor;
        vc_out[k] = if d_ob == 0.0 {
            0.0
        } else {
            (d0 / d_ob + delta0) * vc_ob * factor
        };
        if d_out[k] < d_min {
            k_b = k;
            break;
        }
        k_i += dk;
    }

    Ok((k_a, k_b))
}

#[allow(clippy::too_many_arguments)]
fn apply_evaluated_overshoot_to_model(
    model: &mut StarModel,
    i: usize,
    k_cb: usize,
    k_a: usize,
    k_b: usize,
    d_arr: &[f64],
    vc_arr: &[f64],
    d_min: f64,
) -> Result<(), StarError> {
    let nz = model.nz;
    let outward = model.aux.mixing.top_conv_bdy[i];
    let dk: isize = if outward { -1 } else { 1 };
    let b_max = model.controls.mix_wind.overshoot_brunt_b_max;
    let calc_b = model.controls.calculate_brunt_b;
    let calc_n2 = model.controls.calculate_brunt_n2;

    let mut k_i = k_a as isize;
    let k_end_i = k_b as isize;
    while (dk < 0 && k_i >= k_end_i) || (dk > 0 && k_i <= k_end_i) {
        let k = k_i as usize;

        if b_max > 0.0 && calc_b {
            if !calc_n2 {
                return Err(StarError::InvalidInput(String::from(
                    "add_overshooting: when overshoot_brunt_B_max > 0, must have calculate_Brunt_N2 = .true.",
                )));
            }
            let b_val = if k > 0 && k + 1 < nz {
                model.aux.mixing.unsmoothed_brunt_b[(k_i + dk) as usize]
            } else {
                model.aux.mixing.unsmoothed_brunt_b[k]
            };
            if b_val > b_max {
                break;
            }
        }

        if d_arr[k] < d_min {
            if outward {
                if k + 1 < nz && model.d_mix[k + 1] > d_min {
                    let x0 = flexmesa_num::find0(
                        0.0,
                        d_arr[k] - d_min,
                        model.dq[k],
                        model.d_mix[k + 1] - d_min,
                    )
                    .map_err(|e| {
                        StarError::InvalidInput(format!(
                            "add_overshooting find0 failed at k={k}: {e}"
                        ))
                    })?;
                    if !(0.0..=model.dq[k]).contains(&x0) {
                        return Err(StarError::InvalidInput(format!(
                            "Invalid location for overshoot boundary: cz_bdy_dq={x0}, dq={}",
                            model.dq[k]
                        )));
                    }
                    model.aux.mixing.cz_bdy_dq[k] = x0;
                }
            } else if k > 0 && model.d_mix[k - 1] > d_min {
                let x0 = flexmesa_num::find0(
                    0.0,
                    model.d_mix[k - 1] - d_min,
                    model.dq[k - 1],
                    d_arr[k] - d_min,
                )
                .map_err(|e| {
                    StarError::InvalidInput(format!(
                        "add_overshooting find0 failed at k={k}: {e}"
                    ))
                })?;
                if !(0.0..=model.dq[k - 1]).contains(&x0) {
                    return Err(StarError::InvalidInput(format!(
                        "Invalid location for overshoot boundary: cz_bdy_dq={x0}, dq={}",
                        model.dq[k - 1]
                    )));
                }
                model.aux.mixing.cz_bdy_dq[k - 1] = x0;
            }
            break;
        }

        let rho = if k > 0 {
            (model.dq[k - 1] * model.rho[k] + model.dq[k] * model.rho[k - 1])
                / (model.dq[k - 1] + model.dq[k])
        } else {
            model.rho[k]
        };
        let area_rho = FOUR_PI * model.r[k] * model.r[k] * rho;
        let cdc = area_rho * area_rho * d_arr[k];

        let r_cb = eval_conv_bdy_r(model, i)?;
        if model.r[k] * (dk as f64) >= r_cb * (dk as f64) {
            model.aux.mixing.cdc[k] = cdc;
            model.d_mix[k] = d_arr[k];
            model.conv_vel[k] = vc_arr[k];
        } else if d_arr[k] > model.d_mix[k] {
            model.aux.mixing.cdc[k] = cdc;
            model.d_mix[k] = d_arr[k];
            model.conv_vel[k] = vc_arr[k];
            model.mixing_type[k] = MixingType::OvershootMixing;
        }

        k_i += dk;
    }

    // Post-loop cleanup: if the face loop terminated while still inside the CZ,
    // zero out the remainder from `k_i` to `k_cb` (`overshoot.f90:294-297`).
    let k_cb_i = k_cb as isize;
    let mut kz = k_i;
    while (dk < 0 && kz >= k_cb_i) || (dk > 0 && kz <= k_cb_i) {
        let ku = kz as usize;
        model.aux.mixing.cdc[ku] = 0.0;
        model.d_mix[ku] = 0.0;
        model.conv_vel[ku] = 0.0;
        model.mixing_type[ku] = MixingType::NoMixing;
        kz += dk;
    }

    Ok(())
}

/// Applies convective overshooting across all located convective boundaries
/// using `model.controls.mix_wind.overshoot_sets` (`add_overshooting` in
/// `star/private/overshoot.f90:36-327`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if any matching overshoot set is invalid
/// (including `'other'` without a registered hook, matching
/// `null_other_overshooting_scheme`), or if non-finite `D_mix` is produced.
pub fn add_overshooting(model: &mut StarModel) -> Result<(), StarError> {
    let nz = model.nz;
    let n_bdy = model.aux.mixing.conv_bdy_loc.len();
    if nz < 3 || n_bdy == 0 {
        return Ok(());
    }
    let sets = model.controls.mix_wind.overshoot_sets.clone();
    if !sets.iter().any(|s| !s.scheme.trim().is_empty()) {
        return Ok(());
    }
    let min_q = model.controls.mix_wind.min_overshoot_q;
    let d_min = model.controls.mix_wind.overshoot_d_min;

    let mut d_arr = vec![0.0_f64; nz];
    let mut vc_arr = vec![0.0_f64; nz];

    for i in 0..n_bdy {
        if model.aux.mixing.conv_bdy_q[i] < min_q {
            continue;
        }
        if model.aux.mixing.conv_bdy_loc[i] == 0 {
            continue;
        }

        let bh = model.aux.mixing.burn_h_conv_region[i];
        let bhe = model.aux.mixing.burn_he_conv_region[i];
        let bz = model.aux.mixing.burn_z_conv_region[i];
        let top = model.aux.mixing.top_conv_bdy[i];
        let is_core = i == 0 && model.r_center == 0.0 && top;

        for (j, set) in sets.iter().enumerate() {
            let scheme_str = set.scheme.trim();
            if scheme_str.is_empty() {
                continue;
            }

            let match_zone_type =
                match set.zone_type.trim().to_ascii_lowercase().as_str() {
                    "burn_h" => bh,
                    "burn_he" => bhe,
                    "burn_z" => bz,
                    "nonburn" => !(bh || bhe || bz),
                    "any" => true,
                    other => {
                        return Err(StarError::InvalidInput(format!(
                            "Invalid overshoot_zone_type({}): '{}'",
                            j + 1,
                            other
                        )));
                    }
                };
            let match_zone_loc =
                match set.zone_loc.trim().to_ascii_lowercase().as_str() {
                    "core" => is_core,
                    "shell" => !is_core,
                    "any" => true,
                    other => {
                        return Err(StarError::InvalidInput(format!(
                            "Invalid overshoot_zone_loc({}): '{}'",
                            j + 1,
                            other
                        )));
                    }
                };
            let match_bdy_loc =
                match set.bdy_loc.trim().to_ascii_lowercase().as_str() {
                    "bottom" => !top,
                    "top" => top,
                    "any" => true,
                    other => {
                        return Err(StarError::InvalidInput(format!(
                            "Invalid overshoot_bdy_loc({}): '{}'",
                            j + 1,
                            other
                        )));
                    }
                };

            if !(match_zone_type && match_zone_loc && match_bdy_loc) {
                continue;
            }

            if scheme_str.eq_ignore_ascii_case("none") {
                break;
            }

            let k_cb = eval_conv_bdy_k(model, i)?;
            d_arr.fill(0.0);
            vc_arr.fill(0.0);

            let scheme = match scheme_str.to_ascii_lowercase().as_str() {
                "exponential" => OvershootScheme::Exponential,
                "step" => OvershootScheme::Step,
                "other" => {
                    return Err(StarError::InvalidInput(format!(
                        "overshoot_scheme({}) = 'other' requires a custom other_overshooting_scheme hook (default null_other_overshooting_scheme returns ierr = -1)",
                        j + 1
                    )));
                }
                other => {
                    return Err(StarError::InvalidInput(format!(
                        "Invalid overshoot_scheme({}): '{other}'",
                        j + 1
                    )));
                }
            };

            let (k_a, k_b) = eval_overshoot_profile(
                model,
                i,
                scheme,
                set.f,
                set.f0,
                set.d0,
                set.delta0,
                set.mass_full_on,
                set.mass_full_off,
                d_min,
                &mut d_arr,
                &mut vc_arr,
            )?;

            apply_evaluated_overshoot_to_model(
                model, i, k_cb, k_a, k_b, &d_arr, &vc_arr, d_min,
            )?;
            break;
        }
    }

    for k in 0..nz {
        if !model.d_mix[k].is_finite() {
            return Err(StarError::MicrophysicsError(format!(
                "add_overshooting produced non-finite D_mix at face {k}: {}",
                model.d_mix[k]
            )));
        }
    }
    Ok(())
}

/// Applies convective overshooting profiles across all detected convective
/// boundaries (`add_overshooting` in `star/private/overshoot.f90:36-327`).
///
/// When `configs` is empty, delegates to [`add_overshooting`] using
/// `model.controls.mix_wind.overshoot_sets`. When `configs` is non-empty,
/// populates boundary state if absent and applies each matching explicit
/// [`OvershootConfig`].
pub fn apply_overshooting(
    model: &mut StarModel,
    boundaries: &[ConvBoundary],
    configs: &[OvershootConfig],
) {
    if model.nz < 3 {
        return;
    }
    if configs.is_empty() {
        if model.aux.mixing.conv_bdy_loc.is_empty() && !boundaries.is_empty() {
            sync_aux_boundaries_from_slice(model, boundaries);
        }
        let _ = add_overshooting(model);
        return;
    }
    if boundaries.is_empty() {
        return;
    }
    sync_aux_boundaries_from_slice(model, boundaries);
    let nz = model.nz;
    let min_q = model.controls.mix_wind.min_overshoot_q;
    let mut d_arr = vec![0.0_f64; nz];
    let mut vc_arr = vec![0.0_f64; nz];

    for i in 0..model.aux.mixing.conv_bdy_loc.len() {
        if model.aux.mixing.conv_bdy_q[i] < min_q
            || model.aux.mixing.conv_bdy_loc[i] == 0
        {
            continue;
        }
        let top = model.aux.mixing.top_conv_bdy[i];
        let Ok(k_cb) = eval_conv_bdy_k(model, i) else {
            continue;
        };
        for cfg in configs {
            if cfg.top_conv_bdy != top || cfg.f <= 0.0 || cfg.f0 <= 0.0 {
                continue;
            }
            d_arr.fill(0.0);
            vc_arr.fill(0.0);
            if let Ok((k_a, k_b)) = eval_overshoot_profile(
                model,
                i,
                cfg.scheme,
                cfg.f,
                cfg.f0,
                cfg.d0,
                cfg.delta0,
                0.0,
                0.0,
                cfg.d_min,
                &mut d_arr,
                &mut vc_arr,
            ) {
                let _ = apply_evaluated_overshoot_to_model(
                    model, i, k_cb, k_a, k_b, &d_arr, &vc_arr, cfg.d_min,
                );
            }
        }
    }
}

fn sync_aux_boundaries_from_slice(
    model: &mut StarModel,
    boundaries: &[ConvBoundary],
) {
    let nz = model.nz;
    if model.aux.mixing.cz_bdy_dq.len() != nz {
        model.aux.mixing.resize(nz);
    }
    model.aux.mixing.conv_bdy_loc.clear();
    model.aux.mixing.conv_bdy_q.clear();
    model.aux.mixing.top_conv_bdy.clear();
    model.aux.mixing.burn_h_conv_region.clear();
    model.aux.mixing.burn_he_conv_region.clear();
    model.aux.mixing.burn_z_conv_region.clear();
    for b in boundaries {
        let loc = if b.top_conv_bdy {
            b.cell_index
        } else {
            (b.cell_index + 1).min(nz - 1)
        };
        model.aux.mixing.conv_bdy_loc.push(loc);
        model.aux.mixing.conv_bdy_q.push(b.conv_bdy_q);
        model.aux.mixing.top_conv_bdy.push(b.top_conv_bdy);
        model.aux.mixing.burn_h_conv_region.push(false);
        model.aux.mixing.burn_he_conv_region.push(false);
        model.aux.mixing.burn_z_conv_region.push(false);
        if b.cell_index < nz {
            model.aux.mixing.cz_bdy_dq[b.cell_index] = b.cz_bdy_dq;
        }
    }
}
