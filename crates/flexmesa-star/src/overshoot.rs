// SPDX-License-Identifier: LGPL-3.0-or-later
//! Step, exponential, and `step+exponential` convective overshooting profiles,
//! 16-set convective-zone/location/boundary matching, mass ramps, Péclet number
//! scaling, and Brunt-composition stabilization (`overshoot.f90`,
//! `overshoot_exp.f90`, `overshoot_step.f90`, and `overshoot_utils.f90`).

use std::f64::consts::PI;

use flexmesa_const::MixingType;

use crate::controls_mix_wind::OvershootParamSet;
use crate::mix_info::{cdc_factor, ConvBoundary};
use crate::model::StarModel;

/// Functional form of convective overshooting (`overshoot_scheme` in MESA).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OvershootScheme {
    /// Herwig (2000) exponential decay (`'exponential'`).
    Exponential,
    /// Step overshoot with optional secondary exponential tail (`'step'` or
    /// `'step+exponential'`).
    Step,
}

/// Configuration for a single convective boundary overshooting rule.
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
    /// Secondary exponential tail fraction $f_2$ for `step+exponential`.
    pub f2: f64,
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
            f2: 0.0,
            d0: 0.0,
            delta0: 1.0,
            d_min: 1.0e2,
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
            OvershootScheme::Exponential => (-2.0 * dr / dr_step).exp(),
            OvershootScheme::Step => {
                if dr < dr_step {
                    1.0
                } else if self.f2 > 0.0 {
                    (-2.0 * (dr - dr_step) / (self.f2 * hp_cb)).exp()
                } else {
                    0.0
                }
            }
        }
    }
}

fn cz_cell_span(
    model: &StarModel,
    boundaries: &[ConvBoundary],
    bdy_idx: usize,
) -> (usize, usize, bool) {
    let nz = model.nz;
    let bdy = &boundaries[bdy_idx];
    if bdy.top_conv_bdy {
        let k_top = bdy.cell_index.min(nz - 1);
        let (k_bot, is_core) = if bdy_idx == 0 {
            (nz - 1, model.r_center == 0.0)
        } else {
            (boundaries[bdy_idx - 1].cell_index.min(nz - 1), false)
        };
        (k_top.min(k_bot), k_bot.max(k_top), is_core)
    } else {
        let k_bot = bdy.cell_index.min(nz - 1);
        let k_top = if bdy_idx + 1 < boundaries.len() {
            boundaries[bdy_idx + 1].cell_index.min(nz - 1)
        } else {
            0
        };
        (k_top.min(k_bot), k_bot.max(k_top), false)
    }
}

fn classify_cz_burning(
    model: &StarModel,
    k_top: usize,
    k_bot: usize,
) -> (bool, bool, bool) {
    let mw = &model.controls.mix_wind;
    let mut max_log_t = f64::NEG_INFINITY;
    let mut max_x = 0.0_f64;
    let mut max_y = 0.0_f64;
    let k_hi = k_bot.min(model.nz.saturating_sub(1));
    for k in k_top..=k_hi {
        max_log_t = max_log_t.max(model.ln_t[k] / std::f64::consts::LN_10);
        max_x = max_x.max(model.xh_mass[k]);
        max_y = max_y.max(model.yhe_mass[k]);
    }
    if max_log_t > mw.burn_z_mix_region_logt
        && max_y < mw.max_y_for_burn_z_mix_region
    {
        (false, false, true)
    } else if max_log_t > mw.burn_he_mix_region_logt
        && max_x < mw.max_x_for_burn_he_mix_region
    {
        (false, true, false)
    } else if max_log_t > mw.burn_h_mix_region_logt {
        (true, false, false)
    } else {
        (false, false, false)
    }
}

fn matches_param_set(
    set: &OvershootParamSet,
    top_conv_bdy: bool,
    is_core: bool,
    burn_flags: (bool, bool, bool),
) -> bool {
    let (bh, bhe, bz) = burn_flags;
    let zt_ok = match set.zone_type.trim().to_ascii_lowercase().as_str() {
        "" | "any" => true,
        "burn_h" => bh,
        "burn_he" => bhe,
        "burn_z" => bz,
        "nonburn" => !(bh || bhe || bz),
        _ => false,
    };
    let zl_ok = match set.zone_loc.trim().to_ascii_lowercase().as_str() {
        "" | "any" => true,
        "core" => is_core,
        "shell" => !is_core,
        _ => false,
    };
    let bl_ok = match set.bdy_loc.trim().to_ascii_lowercase().as_str() {
        "" | "any" => true,
        "top" => top_conv_bdy,
        "bottom" => !top_conv_bdy,
        _ => false,
    };
    zt_ok && zl_ok && bl_ok
}

fn config_from_param_set(
    model: &StarModel,
    bdy: &ConvBoundary,
    set: &OvershootParamSet,
) -> Option<OvershootConfig> {
    let s_name = set.scheme.trim().to_ascii_lowercase();
    if s_name.is_empty() || s_name == "none" {
        return None;
    }
    let (scheme, mut f2) = match s_name.as_str() {
        "exponential" => (OvershootScheme::Exponential, 0.0),
        "step" => (OvershootScheme::Step, 0.0),
        "step+exponential" => (OvershootScheme::Step, set.f2),
        _ => return None,
    };

    let mut f = set.f;
    let mut f0 = set.f0;
    let m_msun = model.mstar / flexmesa_const::M_SUN;
    if set.mass_full_on > set.mass_full_off && m_msun < set.mass_full_on {
        if m_msun > set.mass_full_off {
            let w = (m_msun - set.mass_full_off)
                / (set.mass_full_on - set.mass_full_off);
            let fac = 0.5 * (1.0 - (PI * w).cos());
            f *= fac;
            f0 *= fac;
            f2 *= fac;
        } else {
            return None;
        }
    }
    if f <= 0.0 || f0 <= 0.0 {
        return None;
    }
    Some(OvershootConfig {
        scheme,
        top_conv_bdy: bdy.top_conv_bdy,
        f,
        f0,
        f2,
        d0: set.d0,
        delta0: set.delta0,
        d_min: model.controls.mix_wind.overshoot_d_min.max(0.0),
    })
}

/// Applies convective overshooting profiles across all detected convective
/// boundaries (`add_overshooting` in `star/private/overshoot.f90:36-325`).
///
/// When `configs` is non-empty, applies those explicit configurations; when
/// `configs` is empty, evaluates the 16 `model.controls.mix_wind.overshoot_sets`
/// rules in MESA order (applying at most the first matching rule per boundary,
/// with `'none'` short-circuiting).
pub fn apply_overshooting(
    model: &mut StarModel,
    boundaries: &[ConvBoundary],
    configs: &[OvershootConfig],
) {
    if model.nz < 3 || boundaries.is_empty() {
        return;
    }
    let min_q = model.controls.mix_wind.min_overshoot_q;

    if !configs.is_empty() {
        for (i, bdy) in boundaries.iter().enumerate() {
            if bdy.conv_bdy_q < min_q || bdy.conv_bdy_loc == 0 {
                continue;
            }
            for cfg in configs {
                if cfg.top_conv_bdy == bdy.top_conv_bdy
                    && cfg.f > 0.0
                    && cfg.f0 > 0.0
                {
                    apply_single_boundary_overshoot(
                        model, boundaries, i, cfg, None,
                    );
                }
            }
        }
        return;
    }

    let sets = model.controls.mix_wind.overshoot_sets.clone();
    if !sets.iter().any(|s| !s.scheme.trim().is_empty()) {
        return;
    }
    for (i, bdy) in boundaries.iter().enumerate() {
        if bdy.conv_bdy_q < min_q || bdy.conv_bdy_loc == 0 {
            continue;
        }
        let (k_top, k_bot, is_core) = cz_cell_span(model, boundaries, i);
        let burn_flags = classify_cz_burning(model, k_top, k_bot);
        for set in &sets {
            if set.scheme.trim().is_empty() {
                continue;
            }
            if !matches_param_set(set, bdy.top_conv_bdy, is_core, burn_flags) {
                continue;
            }
            if set.scheme.trim().eq_ignore_ascii_case("none") {
                break;
            }
            if let Some(cfg) = config_from_param_set(model, bdy, set) {
                apply_single_boundary_overshoot(
                    model,
                    boundaries,
                    i,
                    &cfg,
                    Some(set),
                );
            }
            break;
        }
    }
}

fn apply_single_boundary_overshoot(
    model: &mut StarModel,
    boundaries: &[ConvBoundary],
    bdy_idx: usize,
    cfg: &OvershootConfig,
    param_set: Option<&OvershootParamSet>,
) {
    let nz = model.nz;
    let bdy = &boundaries[bdy_idx];
    let r_cb = bdy.r_cb;
    let mut hp_cb = bdy.hp_cb;
    if hp_cb <= 0.0 || r_cb <= 0.0 {
        return;
    }

    // Optional CZ radial-size cap on `Hp_cb` (`overshoot_utils.f90:174-222`).
    if model
        .controls
        .mix_wind
        .limit_overshoot_hp_using_size_of_convection_zone
    {
        let (k_top, k_bot, _) = cz_cell_span(model, boundaries, bdy_idx);
        let r_bot = if k_bot + 1 < nz {
            model.r[k_bot + 1]
        } else {
            model.r_center
        };
        let dr_cz = (model.r[k_top] - r_bot).max(0.0);
        let alpha = if model.controls.mix_wind.overshoot_alpha > 0.0 {
            model.controls.mix_wind.overshoot_alpha
        } else {
            model.controls.mixing_length_alpha.max(1.0e-6)
        };
        if dr_cz > 0.0 && alpha * hp_cb > dr_cz {
            hp_cb = dr_cz / alpha;
        }
    }

    let r_ob = if bdy.top_conv_bdy {
        (r_cb - cfg.f0 * hp_cb).max(model.r[nz - 1])
    } else {
        (r_cb + cfg.f0 * hp_cb).min(model.r[0])
    };

    let mut k_ob = bdy.cell_index;
    if bdy.top_conv_bdy {
        while k_ob + 1 < nz && model.r[k_ob] > r_ob {
            if model.r[k_ob + 1] <= r_ob {
                break;
            }
            k_ob += 1;
        }
    } else {
        while k_ob > 0 && model.r[k_ob] < r_ob {
            k_ob -= 1;
        }
    }
    let k_ob_next = (k_ob + 1).min(nz - 1);
    let r0_3 = model.r[k_ob].powi(3);
    let r1_3 = model.r[k_ob_next].powi(3);
    let denom_3 = (r0_3 - r1_3).abs().max(1.0e-99);
    let w = ((r0_3 - r_ob.powi(3)).abs() / denom_3).clamp(0.0, 1.0);

    let vc_ob =
        ((1.0 - w) * model.mlt_vc[k_ob] + w * model.mlt_vc[k_ob_next]).max(0.0);
    let hp_ob = (1.0 - w) * model.scale_height[k_ob]
        + w * model.scale_height[k_ob_next];
    let lambda_ob = model.controls.mixing_length_alpha * hp_ob;
    let d_ob = (vc_ob * lambda_ob / 3.0).max(model.mlt_d[bdy.conv_bdy_loc]);
    if d_ob <= 0.0 && cfg.d0 <= 0.0 {
        return;
    }

    let b_max = model.controls.mix_wind.overshoot_brunt_b_max;
    let check_b = b_max > 0.0 && model.controls.calculate_brunt_n2;

    if bdy.top_conv_bdy {
        for k in (1..=k_ob).rev() {
            if check_b {
                let k_probe = k.saturating_sub(1);
                if model.brunt_b[k_probe] > b_max {
                    break;
                }
            }
            let dr = (model.r[k] - r_ob).max(0.0);
            let inside_cz = model.r[k] <= r_cb;
            if !update_overshoot_face(
                model, cfg, param_set, k, dr, hp_cb, d_ob, vc_ob, inside_cz,
            ) {
                break;
            }
        }
    } else {
        for k in (k_ob + 1)..nz {
            if check_b {
                let k_probe = (k + 1).min(nz - 1);
                if model.brunt_b[k_probe] > b_max {
                    break;
                }
            }
            let dr = (r_ob - model.r[k]).max(0.0);
            let inside_cz = model.r[k] >= r_cb;
            if !update_overshoot_face(
                model, cfg, param_set, k, dr, hp_cb, d_ob, vc_ob, inside_cz,
            ) {
                break;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn update_overshoot_face(
    model: &mut StarModel,
    cfg: &OvershootConfig,
    param_set: Option<&OvershootParamSet>,
    k: usize,
    dr: f64,
    hp_cb: f64,
    d_ob: f64,
    vc_ob: f64,
    inside_cz: bool,
) -> bool {
    let mut factor = cfg.eval_factor(dr, hp_cb);
    let peclet_on = model.controls.mix_wind.peclet_function_full_on;
    let peclet_off = model.controls.mix_wind.peclet_function_full_off;
    if param_set.is_some() && peclet_on > peclet_off && peclet_on > 0.0 {
        let k_rad = (16.0
            * flexmesa_const::CRAD
            * flexmesa_const::CLIGHT
            * model.t[k].powi(3))
            / (9.0
                * model.opacity[k].max(1.0e-10)
                * model.rho[k].powi(2).max(1.0e-30)
                * model.cp[k].max(1.0));
        let d_trial = (cfg.d0 + cfg.delta0 * d_ob) * factor;
        let peclet = d_trial / k_rad.max(1.0e-30);
        if peclet <= peclet_off {
            factor = 0.0;
        } else if peclet < peclet_on {
            let w = (peclet - peclet_off) / (peclet_on - peclet_off);
            factor *= w.clamp(0.0, 1.0);
        }
    }

    let d_ov = (cfg.d0 + cfg.delta0 * d_ob) * factor;
    let vc_ov = if d_ob > 0.0 {
        (cfg.d0 / d_ob + cfg.delta0) * vc_ob * factor
    } else {
        0.0
    };
    if inside_cz {
        model.d_mix[k] = d_ov;
        model.conv_vel[k] = vc_ov;
        model.mlt_cdc[k] = cdc_factor(model, k) * d_ov;
        true
    } else {
        if d_ov < cfg.d_min {
            return false;
        }
        if d_ov > model.d_mix[k] {
            model.d_mix[k] = d_ov;
            model.conv_vel[k] = vc_ov;
            model.mlt_cdc[k] = cdc_factor(model, k) * d_ov;
            model.mixing_type[k] = MixingType::OvershootMixing;
        }
        true
    }
}
