// SPDX-License-Identifier: LGPL-3.0-or-later
//! Adaptive timestep controller (`varcontrol`, nuclear burning depletion
//! limits, and Söderlind & Wang 2006 H211PI digital control filter)
//! corresponding to `star/private/timestep.f90`.
//!
//! # Söderlind & Wang (2006) H211 Digital Filter
//! Given the step-variation ratio $w_n = \max_m (\text{metric}_m / \text{target}_m)$
//! at steps $n$ and $n-1$, MESA (`filter_dt` in `timestep.f90:2360-2412`)
//! updates the timestep via the second-order digital filter
//! ($\beta_1 = \beta_2 = 1/4$, $\alpha_2 = 1/4$):
//! $$r_n = \mathcal{L}(1 / w_n), \qquad r_{n-1} = \mathcal{L}(1 / w_{n-1}),$$
//! $$\frac{\Delta t_{n+1}}{\Delta t_n} =
//!   \mathcal{L}\!\left((r_n r_{n-1} \Delta t_{n-1} / \Delta t_n)^{1/4}\right),$$
//! where $\mathcal{L}(x) = 1 + \kappa \arctan((x - 1)/\kappa)$ with
//! $\kappa = 10$ is the smooth arctangent limiter (`filter_limit` in
//! `timestep.f90:2398`).

use flexmesa_const::L_SUN;

use crate::model::StarModel;

/// Dominant constraint determining the next proposed timestep
/// $\Delta t_{\mathrm{next}}$.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimestepLimitReason {
    /// Structural variation target `varcontrol_target` (`w_vc`).
    Varcontrol,
    /// Nuclear fuel depletion limit (`dX_nuc_drop_limit`).
    NuclearBurn,
    /// Clamped by `max_timestep_factor` (`1.2` default).
    MaxTimestepFactor,
    /// Clamped by `min_timestep_factor` (`0.5` default).
    MinTimestepFactor,
}

/// Summary of the timestep selection pass (`pick_next_timestep` in
/// `timestep.f90`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimestepReport {
    /// Proposed next timestep $\Delta t_{\mathrm{next}}$ in seconds (`s`).
    pub dt_next: f64,
    /// Measured normalized structural variation `varcontrol`.
    pub varcontrol: f64,
    /// Effective limiter ratio $w_n = \max(w_{\mathrm{vc}}, w_{\mathrm{burn}})$.
    pub dt_limit_ratio: f64,
    /// Dominant constraint determining $\Delta t_{\mathrm{next}}$.
    pub reason: TimestepLimitReason,
}

/// Smooth arctangent step-ratio limiter (`filter_limit` in
/// `timestep.f90:2398`):
/// $$\mathcal{L}(x) = 1 + \kappa \arctan\!\left(\frac{x - 1}{\kappa}\right),
///   \qquad \kappa = 10.$$
#[must_use]
#[inline]
pub fn filter_limit(x: f64) -> f64 {
    const KAPPA: f64 = 10.0;
    1.0 + KAPPA * ((x - 1.0) / KAPPA).atan()
}

/// Evaluates the normalized structural variation `varcontrol` between the
/// current state `(\ln\rho, \ln T, \ln R, L)` and the previous step's `*_old`
/// snapshot (`eval_varcontrol` in `timestep.f90:2024-2136`).
#[must_use]
pub fn eval_varcontrol(model: &StarModel) -> f64 {
    let nz = model.nz;
    if nz == 0 || model.lnd_old.len() != nz {
        return 0.0;
    }
    let l_surf_scale = model.lum[0].abs().max(L_SUN);
    let mut sum = 0.0_f64;

    for k in 0..nz {
        let dlnd = (model.lnd[k] - model.lnd_old[k]).abs()
            / (1.0 + model.lnd[k].abs());
        let dlnt = (model.ln_t[k] - model.ln_t_old[k]).abs()
            / (1.0 + model.ln_t[k].abs());
        let dlnr = (model.ln_r[k] - model.ln_r_old[k]).abs()
            / (1.0 + model.ln_r[k].abs());
        let dlum = (model.lum[k] - model.lum_old[k]).abs()
            / (l_surf_scale + model.lum[k].abs());
        sum += dlnd + dlnt + dlnr + dlum;
    }

    sum / ((4 * nz) as f64)
}

/// Evaluates the maximum fractional nuclear fuel depletion ratio
/// $w_{\mathrm{burn}} = \max_{k, j} \max(X_{\mathrm{old},j,k} - X_{j,k}, 0)
/// / \Delta X_{\mathrm{limit}}$ across zones where
/// $X_{\mathrm{old},j,k} \ge X_{\min}$ (`check_dX_nuc_drop` in
/// `timestep.f90:360-520`).
#[must_use]
pub fn eval_dx_nuc_drop_ratio(
    model: &StarModel,
    dx_nuc_drop_limit: f64,
    dx_nuc_drop_min_x_limit: f64,
) -> f64 {
    let nz = model.nz;
    let species = model.species;
    if nz == 0
        || species == 0
        || dx_nuc_drop_limit <= 0.0
        || model.xa_old.len() != nz * species
    {
        return 0.0;
    }
    // Check H1 (`j = 0`) and He4 (`j = 1` or `2` in `basic.net`).
    let mut max_drop = 0.0_f64;
    for k in 0..nz {
        for j in 0..species.min(3) {
            let x_old = model.xa_old[k * species + j];
            if x_old >= dx_nuc_drop_min_x_limit {
                let drop = (x_old - model.xa[k * species + j]).max(0.0);
                max_drop = max_drop.max(drop);
            }
        }
    }
    max_drop / dx_nuc_drop_limit
}

/// Evaluates the Söderlind & Wang (2006) H211PI digital control timestep
/// multiplier $f_{\mathrm{dt}} = \Delta t_{n+1} / \Delta t_n$ (`filter_dt` in
/// `timestep.f90:2360-2412`).
#[must_use]
pub fn eval_h211_factor(
    dt_limit_ratio: f64,
    dt_limit_ratio_old: f64,
    dt: f64,
    dt_old: f64,
    max_timestep_factor: f64,
    min_timestep_factor: f64,
) -> (f64, bool, bool) {
    let ratio_curr = dt_limit_ratio.max(1.0e-10);
    let ratio_prev = if dt_limit_ratio_old > 0.0 {
        dt_limit_ratio_old.max(1.0e-10)
    } else {
        ratio_curr
    };
    let dt_ratio = if dt_old > 0.0 && dt > 0.0 {
        (dt / dt_old).max(1.0e-10)
    } else {
        1.0
    };

    let r_n = filter_limit(1.0 / ratio_curr).max(1.0e-4);
    let r_nm1 = filter_limit(1.0 / ratio_prev).max(1.0e-4);
    let raw_factor = filter_limit((r_n * r_nm1 / dt_ratio).powf(0.25));

    let hit_max = raw_factor > max_timestep_factor;
    let hit_min = raw_factor < min_timestep_factor;
    let clamped = raw_factor.clamp(min_timestep_factor, max_timestep_factor);
    (clamped, hit_max, hit_min)
}

/// Computes `varcontrol`, nuclear depletion limits, and the Söderlind & Wang
/// (2006) H211 filtered next timestep $\Delta t_{\mathrm{next}}$, updating
/// `model.dt_next` and `model.dt_limit_ratio_old` (`pick_next_timestep` in
/// `timestep.f90:102-320`).
pub fn pick_next_timestep(
    model: &mut StarModel,
    max_timestep_factor: f64,
    min_timestep_factor: f64,
    dx_nuc_drop_limit: f64,
) -> TimestepReport {
    let varcontrol = eval_varcontrol(model);
    let vc_target = model.controls.varcontrol_target.max(1.0e-12);
    let ratio_vc = varcontrol / vc_target;
    let ratio_burn = eval_dx_nuc_drop_ratio(model, dx_nuc_drop_limit, 1.0e-4);

    let (dt_limit_ratio, base_reason) = if ratio_burn > ratio_vc {
        (ratio_burn, TimestepLimitReason::NuclearBurn)
    } else {
        (ratio_vc, TimestepLimitReason::Varcontrol)
    };

    let (factor, hit_max, hit_min) = eval_h211_factor(
        dt_limit_ratio,
        model.dt_limit_ratio_old,
        model.dt,
        model.dt_old,
        max_timestep_factor,
        min_timestep_factor,
    );

    let reason = if hit_max {
        TimestepLimitReason::MaxTimestepFactor
    } else if hit_min {
        TimestepLimitReason::MinTimestepFactor
    } else {
        base_reason
    };

    let dt_next = model.dt * factor;
    model.dt_next = dt_next;
    model.dt_limit_ratio_old = dt_limit_ratio.max(1.0e-10);

    TimestepReport {
        dt_next,
        varcontrol,
        dt_limit_ratio,
        reason,
    }
}
