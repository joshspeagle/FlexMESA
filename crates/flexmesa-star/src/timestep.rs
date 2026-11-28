// SPDX-License-Identifier: GPL-3.0-or-later
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
//! $\kappa = 10$ is the smooth arctangent limiter (`limiter` in
//! `timestep.f90:2417-2424`).

use crate::controls::StarError;
use crate::model::StarModel;

/// Outcome of [`crate::timestep_limits::timestep_controller`] on a completed
/// candidate step (`keep_going`, `retry`, or `terminate` in `timestep.f90:39`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimestepAction {
    /// Step satisfies all hard limits; proceed with `report.dt_next`.
    KeepGoing,
    /// Step violated a hard limit; revert to pre-step state and retry with smaller $\Delta t$.
    Retry(String),
    /// Fatal timestep configuration or unrecoverable $\Delta t$ collapse.
    Terminate(String),
}

/// Dominant constraint determining the next proposed timestep
/// $\Delta t_{\mathrm{next}}$.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimestepLimitReason {
    /// Structural variation target `varcontrol_target` (`Tlim_struc`).
    Varcontrol,
    /// Nuclear fuel depletion limit (`dX_nuc_drop_limit`).
    NuclearBurn,
    /// Non-convective abundance change limit (`dX_limit` / `dX_div_X_limit`).
    AbundanceChange,
    /// Structural variable change limit (`delta_lgP/Rho/T/E/R`, `dL_div_L`, `*_cntr`, `delta_lgT_max`).
    StructuralChange,
    /// Luminosity or HR-diagram change limit (`delta_lgL_*`, `delta_lgTeff`, `delta_HR`).
    LuminosityChange,
    /// Total mass change limit (`delta_lg_star_mass_limit`).
    MassChange,
    /// Per-step relative energy conservation error (`limit_for_rel_error_in_energy_conservation`).
    EnergyConservation,
    /// Acoustic Courant or cell-collapse limit (`dt_div_min_dr_div_cs`, `dt_div_dt_cell_collapse`).
    CourantOrCollapse,
    /// Solver iteration or operator-split substep limit (`solver_iters`, `burn_steps`, `diffusion_*`).
    SolverOrSubsteps,
    /// Post-retry timestep growth hold (`timestep_hold > model_number`).
    TimestepHold,
    /// Explicit maximum or forced timestep ceiling (`max_timestep`, `max_years_for_timestep`, `force_timestep`).
    MaxTimestep,
    /// Clamped by `max_timestep_factor` (`1.2` default).
    MaxTimestepFactor,
    /// Clamped by `min_timestep_factor` (`0.8` default).
    MinTimestepFactor,
}

/// Timestep limit identifiers (`Tlim_*` in `star_data/public/star_data_def.inc`,
/// MESA `26.4.1`), 1-based as upstream. `dt_limit_ratio(Tlim_*)` slots are
/// stored at index `Tlim_* - 1`.
///
/// The index chain is copied from upstream, including the two collisions
/// `LG_XSI_CNTR == LG_XNE_CNTR` and `XH_CNTR == LG_XO_CNTR`
/// (`star_data_def.inc` defines `Tlim_lg_XSi_cntr = Tlim_lg_XC_cntr + 1`),
/// recorded in issue #617. Both pairs share one `dt_limit_ratio` slot.
pub(crate) mod tlim {
    pub const STRUC: usize = 1;
    pub const MAX_TIMESTEP_FACTOR: usize = STRUC + 1;
    pub const MIN_TIMESTEP_FACTOR: usize = MAX_TIMESTEP_FACTOR + 1;
    pub const SOLVER: usize = MIN_TIMESTEP_FACTOR + 1;
    pub const NUM_BURN_STEPS: usize = SOLVER + 1;
    pub const NUM_DIFF_SOLVER_STEPS: usize = NUM_BURN_STEPS + 1;
    pub const DX: usize = NUM_DIFF_SOLVER_STEPS + 1;
    pub const DX_DIV_X: usize = DX + 1;
    pub const DL_DIV_L: usize = DX_DIV_X + 1;
    pub const DLGP: usize = DL_DIV_L + 1;
    pub const DLGRHO: usize = DLGP + 1;
    pub const DLGE: usize = DLGRHO + 1;
    pub const DLGT: usize = DLGE + 1;
    pub const DLGR: usize = DLGT + 1;
    pub const DLGL_NUC_CAT: usize = DLGR + 1;
    pub const DLGL_H: usize = DLGL_NUC_CAT + 1;
    pub const DLGL_HE: usize = DLGL_H + 1;
    pub const DLGL_Z: usize = DLGL_HE + 1;
    pub const DLGL_NUC: usize = DLGL_Z + 1;
    pub const DLGTEFF: usize = DLGL_NUC + 1;
    pub const DLGRHO_CNTR: usize = DLGTEFF + 1;
    pub const DLGP_CNTR: usize = DLGRHO_CNTR + 1;
    pub const DLGT_CNTR: usize = DLGP_CNTR + 1;
    pub const DLGT_MAX: usize = DLGT_CNTR + 1;
    pub const DLGT_MAX_AT_HIGH_T: usize = DLGT_MAX + 1;
    pub const LG_XH_CNTR: usize = DLGT_MAX_AT_HIGH_T + 1;
    pub const DMSTAR: usize = LG_XH_CNTR + 1;
    pub const DT_DIV_DT_CELL_COLLAPSE: usize = DMSTAR + 1;
    pub const DT_DIV_MIN_DR_DIV_CS: usize = DT_DIV_DT_CELL_COLLAPSE + 1;
    pub const LGL: usize = DT_DIV_MIN_DR_DIV_CS + 1;
    pub const FORCE_TIMESTEP: usize = LGL + 1;
    pub const MAX_TIMESTEP: usize = FORCE_TIMESTEP + 1;
    pub const TIMESTEP_HOLD: usize = MAX_TIMESTEP + 1;
    pub const DX_DIV_X_CNTR: usize = TIMESTEP_HOLD + 1;
    pub const LG_XHE_CNTR: usize = DX_DIV_X_CNTR + 1;
    pub const LG_XC_CNTR: usize = LG_XHE_CNTR + 1;
    pub const LG_XNE_CNTR: usize = LG_XC_CNTR + 1;
    pub const LG_XO_CNTR: usize = LG_XNE_CNTR + 1;
    /// Upstream defines this as `Tlim_lg_XC_cntr + 1`, which equals
    /// [`LG_XNE_CNTR`] (issue #617).
    pub const LG_XSI_CNTR: usize = LG_XC_CNTR + 1;
    /// Equals [`LG_XO_CNTR`] because `LG_XSI_CNTR` collides (issue #617).
    pub const XH_CNTR: usize = LG_XSI_CNTR + 1;
    pub const XHE_CNTR: usize = XH_CNTR + 1;
    pub const XC_CNTR: usize = XHE_CNTR + 1;
    pub const XNE_CNTR: usize = XC_CNTR + 1;
    pub const XO_CNTR: usize = XNE_CNTR + 1;
    pub const XSI_CNTR: usize = XO_CNTR + 1;
    pub const DX_NUC_DROP: usize = XSI_CNTR + 1;
    pub const NEG_X: usize = DX_NUC_DROP + 1;
    pub const BAD_XSUM: usize = NEG_X + 1;
    pub const DLGL_POWER_PHOTO: usize = BAD_XSUM + 1;
    pub const DELTA_HR: usize = DLGL_POWER_PHOTO + 1;
    pub const DLOG_EPS_NUC: usize = DELTA_HR + 1;
    pub const NUM_DIFF_SOLVER_ITERS: usize = DLOG_EPS_NUC + 1;
    pub const DEL_MDOT: usize = NUM_DIFF_SOLVER_ITERS + 1;
    pub const ADJUST_J_Q: usize = DEL_MDOT + 1;
    pub const DELTA_YE_HIGH_T: usize = ADJUST_J_Q + 1;
    pub const ERROR_IN_ENERGY_CONSERVATION: usize = DELTA_YE_HIGH_T + 1;
    pub const RETRY: usize = ERROR_IN_ENERGY_CONSERVATION + 1;
    pub const BINARY: usize = RETRY + 1;
    pub const ERROR_OTHER: usize = BINARY + 1;
    pub const OTHER_TIMESTEP_LIMIT: usize = ERROR_OTHER + 1;
    /// `numTlim`.
    pub const NUM_TLIM: usize = OTHER_TIMESTEP_LIMIT;
}

/// Binary timestep limit reason code (`Tlim_binary`, 1-based;
/// `star_data/public/star_data_def.inc:100`).
pub const TLIM_BINARY: usize = tlim::BINARY;

/// Negative mass fraction hard limit reason code (`Tlim_neg_X`, 1-based;
/// `star_data/public/star_data_def.inc:84`).
pub const TLIM_NEG_X: usize = tlim::NEG_X;

/// The `dt_limit_ratio(1:numTlim)` array of `do_timestep_limits`.
pub(crate) type DtLimitRatios = [f64; tlim::NUM_TLIM];

/// Coarse [`TimestepLimitReason`] for an upstream `Tlim_*` index.
#[must_use]
pub(crate) fn reason_for_tlim(i_limit: usize) -> TimestepLimitReason {
    #[allow(clippy::wildcard_imports)]
    use tlim::*;
    use TimestepLimitReason as R;
    match i_limit {
        MAX_TIMESTEP_FACTOR => R::MaxTimestepFactor,
        MIN_TIMESTEP_FACTOR => R::MinTimestepFactor,
        SOLVER
        | NUM_BURN_STEPS
        | NUM_DIFF_SOLVER_STEPS
        | NUM_DIFF_SOLVER_ITERS
        | OTHER_TIMESTEP_LIMIT => R::SolverOrSubsteps,
        DX | DX_DIV_X | DX_DIV_X_CNTR | LG_XH_CNTR | LG_XHE_CNTR
        | LG_XC_CNTR | LG_XNE_CNTR | LG_XO_CNTR | XHE_CNTR | XC_CNTR
        | XNE_CNTR | XO_CNTR | XSI_CNTR => R::AbundanceChange,
        DL_DIV_L | DLGP | DLGRHO | DLGE | DLGT | DLGR | DLGRHO_CNTR
        | DLGP_CNTR | DLGT_CNTR | DLGT_MAX | DLGT_MAX_AT_HIGH_T
        | DLOG_EPS_NUC | ADJUST_J_Q | DELTA_YE_HIGH_T => R::StructuralChange,
        DLGL_NUC_CAT | DLGL_H | DLGL_HE | DLGL_Z | DLGL_NUC | DLGTEFF | LGL
        | DLGL_POWER_PHOTO | DELTA_HR => R::LuminosityChange,
        DMSTAR | DEL_MDOT => R::MassChange,
        DT_DIV_DT_CELL_COLLAPSE | DT_DIV_MIN_DR_DIV_CS => R::CourantOrCollapse,
        DX_NUC_DROP => R::NuclearBurn,
        ERROR_IN_ENERGY_CONSERVATION => R::EnergyConservation,
        FORCE_TIMESTEP | MAX_TIMESTEP => R::MaxTimestep,
        TIMESTEP_HOLD => R::TimestepHold,
        _ => R::Varcontrol,
    }
}

/// Summary of the timestep selection pass (`pick_next_timestep` and
/// [`crate::timestep_limits::timestep_controller`] in `timestep.f90`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimestepReport {
    /// Proposed next timestep $\Delta t_{\mathrm{next}}$ in seconds (`s`).
    pub dt_next: f64,
    /// Measured normalized structural variation `varcontrol`.
    pub varcontrol: f64,
    /// Effective limiter ratio $w_n = \max_m w_m$.
    pub dt_limit_ratio: f64,
    /// Dominant constraint determining $\Delta t_{\mathrm{next}}$.
    pub reason: TimestepLimitReason,
    /// Upstream `s% why_Tlim` (1-based `Tlim_*` index) behind `reason`.
    pub why_tlim: usize,
}

/// Smooth arctangent step-ratio limiter (`limiter` in
/// `timestep.f90:2417-2424`):
/// $$\mathcal{L}(x) = 1 + \kappa \arctan\!\left(\frac{x - 1}{\kappa}\right),
///   \qquad \kappa = 10.$$
#[must_use]
#[inline]
pub fn filter_limit(x: f64) -> f64 {
    const KAPPA: f64 = 10.0;
    1.0 + KAPPA * flexmesa_math::atan((x - 1.0) / KAPPA)
}

/// Evaluates the upstream 5-point boxcar variation sum for a single structural
/// variable (`eval_varcontrol` in `star/private/timestep.f90:2332-2347`).
///
/// Note on upstream Fortran MESA (release `26.4.1`) parity (`DECISIONS #9`):
/// `star/private/timestep.f90:2342-2343` computes the 3-point sum at `k = nz - 1`
/// (`0`-based `nz - 2`) into `sumj` without accumulating `sumterm(j) = sumterm(j) + sumj`
/// (an upstream dead-store), whereas `k = 2` (`0`-based `1`) IS accumulated at
/// `star/private/timestep.f90:2339-2341`. Both Safe Rust and `eval_varcontrol_jax` preserve
/// this exact upstream expression for 16-digit cross-backend parity.
fn smoothed_var_sum(v: &[f64], v_old: &[f64]) -> f64 {
    // Callers guarantee `nz >= 3` and equal lengths (`eval_varcontrol`).
    let nz = v.len();
    let mut sum = 0.0_f64;
    for k in 2..nz.saturating_sub(2) {
        let s_new: f64 = v[k - 2..=k + 2].iter().sum();
        let s_old: f64 = v_old[k - 2..=k + 2].iter().sum();
        sum += (s_new - s_old).abs() / 5.0;
    }
    sum += ((2.0 * v[0] + v[1]) - (2.0 * v_old[0] + v_old[1])).abs() / 3.0;
    sum += ((2.0 * v[nz - 1] + v[nz - 2])
        - (2.0 * v_old[nz - 1] + v_old[nz - 2]))
        .abs()
        / 3.0;
    let s1_new = v[0] + v[1] + v[2];
    let s1_old = v_old[0] + v_old[1] + v_old[2];
    sum += (s1_new - s1_old).abs() / 3.0;
    sum
}

/// Evaluates the normalized structural variation `varcontrol` between the
/// current state `(\ln\rho, \ln T, \ln R)` and the previous step's `*_old`
/// snapshot using a 5-point boxcar filter (`eval_varcontrol` in
/// `timestep.f90:2299-2364`; lines `2326-2335` explicitly exclude `L`, `v`,
/// `u`, `w`, `Hp`, `j_rot`, `w_div_wc`, `alpha_RTI`, and `Et_RSP`).
///
/// # Errors
///
/// Upstream stops with `mesa_error` when `xh_old` is not associated
/// (`timestep.f90:2312-2314`) and reads outside the arrays for `nz < 3`,
/// which has no defined behaviour to reproduce, so a missing or mismatched
/// `*_old` snapshot or `nz < 3` returns an error.
pub fn eval_varcontrol(model: &StarModel) -> Result<f64, StarError> {
    let nz = model.nz;
    if nz < 3
        || model.lnd.len() < nz
        || model.ln_t.len() < nz
        || model.ln_r.len() < nz
        || model.lnd_old.len() != nz
        || model.ln_t_old.len() != nz
        || model.ln_r_old.len() != nz
    {
        return Err(StarError::InvalidInput(format!(
            "eval_varcontrol: needs nz >= 3 and matching *_old arrays \
             (nz = {nz}, lnd_old = {}, ln_t_old = {}, ln_r_old = {})",
            model.lnd_old.len(),
            model.ln_t_old.len(),
            model.ln_r_old.len()
        )));
    }
    let sum_lnd = smoothed_var_sum(&model.lnd[..nz], &model.lnd_old) / 3.0;
    let sum_lnt = smoothed_var_sum(&model.ln_t[..nz], &model.ln_t_old);
    let sum_lnr = smoothed_var_sum(&model.ln_r[..nz], &model.ln_r_old);

    let sumscales = model.lnd_old[0].abs().max(1.0)
        + model.ln_t_old[0].abs().max(1.0)
        + model.ln_r_old[0].abs().max(1.0);
    let nterms = (3 * nz) as f64;
    // `sumvar = sumvar/sumscales; varcontrol = sumvar/nterms`
    // (`star/private/timestep.f90:2355-2357`).
    Ok((sum_lnd + sum_lnt + sum_lnr) / sumscales / nterms)
}

/// The low-pass controller's limited step ratio `limtr`, with `order = 1`
/// (`filter_dt_next` in `star/private/timestep.f90:2374-2390`).
/// `dt_limit_ratio` must already be floored at `1e-10`.
#[must_use]
pub fn h211_limtr(
    dt_limit_ratio: f64,
    dt_limit_ratio_old: f64,
    dt: f64,
    dt_old: f64,
) -> f64 {
    let ratio = filter_limit(1.0 / dt_limit_ratio);
    let ratio_prev = filter_limit(1.0 / dt_limit_ratio_old);
    filter_limit(
        flexmesa_math::pow(ratio, 0.25)
            * flexmesa_math::pow(ratio_prev, 0.25)
            * flexmesa_math::pow(dt / dt_old, -0.25),
    )
}

/// Evaluates the Söderlind & Wang (2006) H211PI digital control timestep
/// multiplier $f_{\mathrm{dt}} = \Delta t_{n+1} / \Delta t_n$ (`filter_dt_next`
/// in `timestep.f90:2362-2427`).
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
    let raw_factor = if dt_limit_ratio_old > 0.0 && dt_old > 0.0 && dt > 0.0 {
        h211_limtr(ratio_curr, dt_limit_ratio_old, dt, dt_old)
    } else {
        1.0 / ratio_curr
    };

    let hit_max = max_timestep_factor > 0.0 && raw_factor > max_timestep_factor;
    let hit_min = min_timestep_factor > 0.0 && raw_factor < min_timestep_factor;
    let mut clamped = raw_factor;
    if max_timestep_factor > 0.0 {
        clamped = clamped.min(max_timestep_factor);
    }
    if min_timestep_factor > 0.0 {
        clamped = clamped.max(min_timestep_factor);
    }
    (clamped, hit_max, hit_min)
}

/// Computes `varcontrol`, nuclear depletion limits, and the Söderlind & Wang
/// (2006) H211 filtered next timestep $\Delta t_{\mathrm{next}}$, updating
/// `model.dt_next` and `model.dt_limit_ratio_old` (`pick_next_timestep` in
/// `timestep.f90:102-320`).
///
/// # Errors
///
/// Propagates [`eval_varcontrol`] errors.
pub fn pick_next_timestep(
    model: &mut StarModel,
    max_timestep_factor: f64,
    min_timestep_factor: f64,
) -> Result<TimestepReport, StarError> {
    let varcontrol = eval_varcontrol(model)?;
    let vc_target = model.controls.varcontrol_target.max(1.0e-99);
    let ratio_vc = varcontrol / vc_target;
    // The `check_dX_nuc_drop` port (`timestep.f90:2100-2241`). This helper
    // reports ratios only, so the hard-limit retry that belongs to
    // `timestep_controller` is not applied here.
    let log_max_t = model
        .ln_t
        .iter()
        .take(model.nz)
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        / flexmesa_const::LN10;
    let ratio_burn =
        crate::timestep_checks::check_dx_nuc_drop_full(model, true, log_max_t)
            .expect("hard limit is skipped, so the check cannot fail");

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

    Ok(TimestepReport {
        dt_next,
        varcontrol,
        dt_limit_ratio,
        reason,
        why_tlim: if reason == TimestepLimitReason::NuclearBurn {
            tlim::DX_NUC_DROP
        } else {
            tlim::STRUC
        },
    })
}
