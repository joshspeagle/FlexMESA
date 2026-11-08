// SPDX-License-Identifier: LGPL-3.0-or-later
//! Step and exponential convective overshooting profiles (`overshoot.f90`,
//! `overshoot_exp.f90`, `overshoot_step.f90`, and `overshoot_utils.f90`).
//!
//! # Overshoot Anchor and Decay Profile
//! From a sub-cell convective boundary at radius $r_{\mathrm{cb}}$ with
//! pressure scale height $H_{P,\mathrm{cb}}$, the overshoot anchor steps a
//! distance $f_0 H_{P,\mathrm{cb}}$ **into** the convection zone:
//! $$r_{\mathrm{ob}} = \begin{cases}
//!   \max(r_{\mathrm{nz}-1},\, r_{\mathrm{cb}} - f_0 H_{P,\mathrm{cb}})
//!     & \text{if outward (`top_conv_bdy`)} \\
//!   \min(r_0,\, r_{\mathrm{cb}} + f_0 H_{P,\mathrm{cb}})
//!     & \text{if inward (`!top_conv_bdy`)}
//! \end{cases}$$
//! At $r_{\mathrm{ob}}$, the convective velocity $v_{c,\mathrm{ob}}$ and
//! mixing length $\Lambda_{\mathrm{ob}} = \alpha_{\mathrm{MLT}} H_{P,\mathrm{ob}}$
//! yield the anchor diffusivity
//! $D_{\mathrm{ob}} = \tfrac{1}{3} v_{c,\mathrm{ob}} \Lambda_{\mathrm{ob}}$.
//! Moving across the boundary with radial distance
//! $\Delta r(k) = |r_k - r_{\mathrm{ob}}|$, the decay factor is:
//! - **Exponential (`OvershootScheme::Exponential`)**:
//!   $$\mathrm{factor}(k) = \exp\!\left(-\frac{2\,\Delta r(k)}{f\,H_{P,\mathrm{cb}}}\right)$$
//! - **Step (`OvershootScheme::Step`)**:
//!   $$\mathrm{factor}(k) = \begin{cases} 1 & \Delta r(k) < f\,H_{P,\mathrm{cb}} \\ \exp\!\left(-\frac{2(\Delta r(k) - f\,H_{P,\mathrm{cb}})}{f_2\,H_{P,\mathrm{cb}}}\right) & \Delta r(k) \ge f\,H_{P,\mathrm{cb}} \land f_2 > 0 \\ 0 & \text{otherwise} \end{cases}$$

use flexmesa_const::MixingType;

use crate::mix_info::{cdc_factor, ConvBoundary};
use crate::model::StarModel;

/// Functional form of convective overshooting (`overshoot_scheme` in MESA).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OvershootScheme {
    /// Herwig (2000) exponential decay (`'exponential'`).
    Exponential,
    /// Step overshoot with optional secondary exponential tail (`'step'`).
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
    /// Secondary exponential tail fraction $f_2$ for step overshoot.
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

/// Applies convective overshooting profiles across all detected convective
/// boundaries (`add_overshooting` in `star/private/overshoot.f90:200-468`).
pub fn apply_overshooting(
    model: &mut StarModel,
    boundaries: &[ConvBoundary],
    configs: &[OvershootConfig],
) {
    if model.nz < 3 || boundaries.is_empty() || configs.is_empty() {
        return;
    }

    for bdy in boundaries {
        for cfg in configs {
            if cfg.top_conv_bdy == bdy.top_conv_bdy
                && cfg.f > 0.0
                && cfg.f0 > 0.0
            {
                apply_single_boundary_overshoot(model, bdy, cfg);
            }
        }
    }
}

fn apply_single_boundary_overshoot(
    model: &mut StarModel,
    bdy: &ConvBoundary,
    cfg: &OvershootConfig,
) {
    let nz = model.nz;
    let r_cb = bdy.r_cb;
    let hp_cb = bdy.hp_cb;
    if hp_cb <= 0.0 || r_cb <= 0.0 {
        return;
    }

    // 1. Anchor radius `r_ob` inside the convection zone (`eval_over_bdy_params`).
    let r_ob = if bdy.top_conv_bdy {
        (r_cb - cfg.f0 * hp_cb).max(model.r[nz - 1])
    } else {
        (r_cb + cfg.f0 * hp_cb).min(model.r[0])
    };

    // Locate cell `k_ob` containing `r_ob` (`r[k_ob + 1] <= r_ob <= r[k_ob]`).
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

    // 2. Traverse from `k_ob` across `r_cb` into the adjacent radiative zone.
    if bdy.top_conv_bdy {
        for k in (1..=k_ob).rev() {
            let dr = (model.r[k] - r_ob).max(0.0);
            let inside_cz = model.r[k] <= r_cb;
            if !update_overshoot_face(
                model, cfg, k, dr, hp_cb, d_ob, vc_ob, inside_cz,
            ) {
                break;
            }
        }
    } else {
        for k in (k_ob + 1)..nz {
            let dr = (r_ob - model.r[k]).max(0.0);
            let inside_cz = model.r[k] >= r_cb;
            if !update_overshoot_face(
                model, cfg, k, dr, hp_cb, d_ob, vc_ob, inside_cz,
            ) {
                break;
            }
        }
    }
}

fn update_overshoot_face(
    model: &mut StarModel,
    cfg: &OvershootConfig,
    k: usize,
    dr: f64,
    hp_cb: f64,
    d_ob: f64,
    vc_ob: f64,
    inside_cz: bool,
) -> bool {
    let factor = cfg.eval_factor(dr, hp_cb);
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
