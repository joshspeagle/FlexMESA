// SPDX-License-Identifier: GPL-3.0-or-later
//! Substepped implicit rotational angular momentum transport (`do_solve_omega_mix`)
//! ported from `star/private/solve_omega_mix.f90:20-669` in IEEE-754 `binary128`
//! ([`flexmesa_math::Quad`]).

use flexmesa_const::TWO_THIRDS;
use flexmesa_math::Quad;

use crate::controls::StarError;
use crate::inlist_runner::{DefaultStarExtras, StarExtras};
use crate::model::StarModel;

/// Maximum adaptive substeps per `do_solve_omega_mix` call (`max_steps = 20`).
const MAX_STEPS: usize = 20;
/// Default minimum substeps per `do_solve_omega_mix` call (`min_steps = 4`).
const DEFAULT_MIN_STEPS: usize = 4;
/// Maximum Newton iterations per substep (`max_iters_per_substep = 4`).
const MAX_ITERS_PER_SUBSTEP: usize = 4;
/// Maximum cumulative Newton iterations across all substeps (`max_iters_total = 40`).
const MAX_ITERS_TOTAL: usize = 40;
/// Infinity-norm Newton correction tolerance (`tol_correction_max = 1d-4`).
const TOL_CORRECTION_MAX: f64 = 1.0e-4;
/// Mean absolute Newton correction tolerance (`tol_correction_norm = 1d-7`).
const TOL_CORRECTION_NORM: f64 = 1.0e-7;
/// Remaining-time relative exit tolerance (`1d-10`).
const REMAINING_TIME_REL_TOL: f64 = 1.0e-10;
/// Minimum substep fraction of remaining time (`1d-6`).
const MIN_SUBSTEP_REMAINING_FRAC: f64 = 1.0e-6;
/// Sanity upper bound on `|omega|` (`1d50`, `solve_omega_mix.f90:112,278`).
const MAX_ABS_OMEGA: f64 = 1.0e50;

/// Return status of [`do_solve_omega_mix`] / [`do_solve_omega_mix_with_extras`],
/// mirroring upstream `keep_going` / `retry` / `terminate` (`solve_omega_mix.f90:32-313`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OmegaMixOutcome {
    /// Angular momentum transport succeeded and passed conservation checks.
    KeepGoing,
    /// Substep convergence, tridiagonal solve, implicit torque hook, or total
    /// angular momentum conservation failed; caller should retry the outer step.
    Retry(String),
    /// Non-retriable failure in `update_rotation_mixing_info` or
    /// `get_rotation_sigmas` (`t_solve_omega_mix` in `solve_omega_mix.f90:136,153`).
    Terminate(String),
}

#[inline]
fn q_min(a: Quad, b: Quad) -> Quad {
    if a < b { a } else { b }
}

#[inline]
fn q_max(a: Quad, b: Quad) -> Quad {
    if a > b { a } else { b }
}

/// Solves rotational angular momentum transport (`do_solve_omega_mix` in
/// `star/private/solve_omega_mix.f90:32-667`) with default (no-op) extras hooks.
///
/// # Errors
/// Returns [`StarError`] if `other_torque` / `binary_other_torque` fails or if
/// non-finite / out-of-range `omega` values are encountered.
pub fn do_solve_omega_mix(
    model: &mut StarModel,
    dt_total: f64,
) -> Result<OmegaMixOutcome, StarError> {
    let mut extras = DefaultStarExtras;
    do_solve_omega_mix_with_extras(model, dt_total, &mut extras)
}

/// Solves rotational angular momentum transport (`do_solve_omega_mix` in
/// `star/private/solve_omega_mix.f90:32-667`) in [`Quad`] (`binary128`) precision
/// with custom [`StarExtras`] torque hooks (`other_torque`, `other_torque_implicit`)
/// and optional binary torque hooks (`binary_other_torque`,
/// `binary_other_torque_implicit`).
///
/// Updates `model.aux.i_rot` and `model.omega` first (`set_i_rot`, `set_omega`),
/// returns immediately when `dt_total <= 0.0`, integrates the Heger, Langer &
/// Woosley (2000) Eq. 46 diffusion equation including both $\Omega$-curvature
/// (`am_sig_omega`) and $i_{\mathrm{rot}}$-curvature (`am_sig_j`) terms via
/// adaptive substepping (`min_steps = 4..max_steps = 20`) and Newton iterations
/// solved with a `Quad` Thomas tridiagonal solver, updates `model.j_rot`, and
/// checks total angular momentum conservation against
/// `angular_momentum_error_retry` / `angular_momentum_error_warn` when no
/// external torque hooks are active.
///
/// # Errors
/// Returns [`StarError`] if `other_torque` / `binary_other_torque` fails or if
/// non-finite / out-of-range `omega` values are encountered.
pub fn do_solve_omega_mix_with_extras(
    model: &mut StarModel,
    dt_total: f64,
    extras: &mut dyn StarExtras,
) -> Result<OmegaMixOutcome, StarError> {
    let nz = model.nz;
    if nz == 0 {
        return Ok(OmegaMixOutcome::KeepGoing);
    }

    // Update `omega` for new `i_rot` and previous `j_rot` to conserve angular
    // momentum (`solve_omega_mix.f90:70-72`).
    crate::hydro_rotation::set_i_rot(model, false);
    crate::hydro_rotation::set_omega(model);

    if dt_total <= 0.0 || nz < 2 {
        return Ok(OmegaMixOutcome::KeepGoing);
    }

    prepare_torque_arrays(model, nz);

    if model.controls.use_other_torque {
        extras.apply_other_torque(model)?;
    }
    if let Some(hook) = model.aux.binary_other_torque.clone() {
        hook.apply(model)?;
    }

    let recalc = model.controls.mix_wind.recalc_mixing_info_each_substep;
    run_solve_omega_mix_quad(
        model,
        dt_total,
        extras,
        DEFAULT_MIN_STEPS,
        recalc,
        false,
    )
}

/// Standalone rotational angular momentum diffusion driver with caller-supplied
/// viscosity profile `nu_omega` ($\mathrm{cm^2\,s^{-1}}$) and minimum substep
/// count `n_substeps`.
///
/// Delegates directly to the same [`Quad`] tridiagonal Newton solver core as
/// [`do_solve_omega_mix_with_extras`]. When `model.aux.i_rot` is uninitialized
/// (`<= 0.0`), fills spherical shells $i_{\mathrm{rot},k} = \tfrac{2}{3} r_k^2$
/// and initializes $j_{\mathrm{rot},k} = i_{\mathrm{rot},k} \Omega_k$ before
/// integrating.
pub fn solve_omega_mix(
    model: &mut StarModel,
    dt: f64,
    nu_omega: &[f64],
    n_substeps: usize,
) {
    let nz = model.nz;
    if nz < 2 || dt <= 0.0 || nu_omega.len() != nz {
        return;
    }

    if model.aux.i_rot.len() != nz {
        model.aux.i_rot.resize(nz, 0.0);
    }
    if model.j_rot.len() != nz {
        model.j_rot.resize(nz, 0.0);
    }
    for k in 0..nz {
        if model.aux.i_rot[k] <= 0.0 {
            model.aux.i_rot[k] = TWO_THIRDS * model.r[k] * model.r[k];
        }
        model.j_rot[k] = model.aux.i_rot[k] * model.omega[k];
    }

    model.aux.am_nu_omega.resize(nz, 0.0);
    model.aux.am_nu_omega[..nz].copy_from_slice(nu_omega);
    model.aux.am_nu_j.resize(nz, 0.0);
    model.aux.am_nu_j[..nz].fill(0.0);
    prepare_torque_arrays(model, nz);

    let min_steps = n_substeps.clamp(1, MAX_STEPS);
    let mut extras = DefaultStarExtras;
    let _ = run_solve_omega_mix_quad(
        model,
        dt,
        &mut extras,
        min_steps,
        false,
        true,
    );
}

fn prepare_torque_arrays(model: &mut StarModel, nz: usize) {
    model.aux.extra_jdot.resize(nz, 0.0);
    model.aux.extra_omegadot.resize(nz, 0.0);
    model.aux.extra_jdot[..nz].fill(0.0);
    model.aux.extra_omegadot[..nz].fill(0.0);
    model.aux.d_extra_jdot_domega_m1.resize(nz, 0.0);
    model.aux.d_extra_jdot_domega_00.resize(nz, 0.0);
    model.aux.d_extra_jdot_domega_p1.resize(nz, 0.0);
    model.aux.d_extra_omegadot_domega_m1.resize(nz, 0.0);
    model.aux.d_extra_omegadot_domega_00.resize(nz, 0.0);
    model.aux.d_extra_omegadot_domega_p1.resize(nz, 0.0);
    model.aux.d_extra_jdot_domega_m1[..nz].fill(0.0);
    model.aux.d_extra_jdot_domega_00[..nz].fill(0.0);
    model.aux.d_extra_jdot_domega_p1[..nz].fill(0.0);
    model.aux.d_extra_omegadot_domega_m1[..nz].fill(0.0);
    model.aux.d_extra_omegadot_domega_00[..nz].fill(0.0);
    model.aux.d_extra_omegadot_domega_p1[..nz].fill(0.0);
}

fn eval_j_tot(model: &StarModel, standalone_mode: bool) -> f64 {
    if standalone_mode {
        (0..model.nz)
            .map(|k| model.dm_bar[k] * model.j_rot[k])
            .sum()
    } else {
        crate::hydro_rotation::total_angular_momentum(model)
    }
}

fn validate_omega_profile(
    model: &StarModel,
    stage: &str,
) -> Result<(), StarError> {
    for k in 0..model.nz {
        let w = model.omega[k];
        if !w.is_finite() || w.abs() > MAX_ABS_OMEGA {
            return Err(StarError::ConvergenceFailure(format!(
                "{stage}: bad omega[{k}] = {w:.6e} at model {}",
                model.model_number
            )));
        }
    }
    Ok(())
}

fn run_solve_omega_mix_quad(
    model: &mut StarModel,
    dt_total: f64,
    extras: &mut dyn StarExtras,
    min_steps: usize,
    recalc_mixing_info_each_substep: bool,
    standalone_mode: bool,
) -> Result<OmegaMixOutcome, StarError> {
    let nz = model.nz;
    let j_tot0 = Quad::from_f64(eval_j_tot(model, standalone_mode));
    validate_omega_profile(model, "start solve omega")?;

    let mut am_sig_omega_prev = vec![0.0_f64; nz];
    let mut am_sig_j_prev = vec![0.0_f64; nz];
    let mut du = vec![Quad::ZERO; nz];
    let mut d = vec![Quad::ZERO; nz];
    let mut dl = vec![Quad::ZERO; nz];
    let mut bp = vec![Quad::ZERO; nz];
    let mut vp = vec![Quad::ZERO; nz];
    let mut xp = vec![Quad::ZERO; nz];
    let mut d_x = vec![Quad::ZERO; nz];
    let mut x_0 = vec![Quad::ZERO; nz];
    let mut x_1 = vec![Quad::ZERO; nz];
    let mut rhs = vec![Quad::ZERO; nz];
    let mut del = vec![Quad::ZERO; nz];

    let total_time = Quad::from_f64(dt_total);
    let tol_time = total_time * REMAINING_TIME_REL_TOL;
    let tol_correction_max = Quad::from_f64(TOL_CORRECTION_MAX);
    let tol_correction_norm = Quad::from_f64(TOL_CORRECTION_NORM);
    let dt_cap_min_steps = Quad::from_f64(dt_total / (min_steps as f64));
    let nz_q = Quad::from_i64(i64::try_from(nz).unwrap_or(i64::MAX));

    let mut time = Quad::ZERO;
    let mut steps_used = 0_usize;
    let mut total_num_iters = 0_usize;
    let mut outcome = OmegaMixOutcome::KeepGoing;

    'step_loop: while total_time - time > tol_time && steps_used < MAX_STEPS {
        if steps_used > 0 && recalc_mixing_info_each_substep {
            if let Err(e) =
                crate::rotation_mix::update_rotation_mixing_info(model)
            {
                outcome = OmegaMixOutcome::Terminate(format!(
                    "update_rotation_mixing_info failed in do_solve_omega_mix: {e}"
                ));
                break 'step_loop;
            }
        }

        if steps_used == 0 || recalc_mixing_info_each_substep {
            if steps_used > 0 {
                am_sig_omega_prev
                    .copy_from_slice(&model.aux.am_sig_omega[..nz]);
                am_sig_j_prev.copy_from_slice(&model.aux.am_sig_j[..nz]);
            }
            crate::hydro_rotation::get_rotation_sigmas(
                model, 0, nz, dt_total,
            );
            if !model.aux.am_sig_omega[..nz]
                .iter()
                .chain(model.aux.am_sig_j[..nz].iter())
                .all(|v| v.is_finite())
            {
                outcome = OmegaMixOutcome::Terminate(String::from(
                    "get_rotation_sigmas produced non-finite sigma in do_solve_omega_mix",
                ));
                break 'step_loop;
            }
            if steps_used > 0 {
                for k in 0..nz {
                    model.aux.am_sig_omega[k] = 0.5
                        * (model.aux.am_sig_omega[k] + am_sig_omega_prev[k]);
                    model.aux.am_sig_j[k] =
                        0.5 * (model.aux.am_sig_j[k] + am_sig_j_prev[k]);
                }
            }
        }

        steps_used += 1;

        let mut dt = Quad::from_f64(0.5 * min_mixing_timescale(model));
        let remaining_time = total_time - time;
        dt = q_max(dt, remaining_time * MIN_SUBSTEP_REMAINING_FRAC);
        dt = q_min(dt, dt_cap_min_steps);
        if dt >= remaining_time {
            dt = remaining_time;
        } else {
            dt = q_min(dt, remaining_time * 0.5);
        }
        if steps_used >= MAX_STEPS {
            dt = remaining_time;
        }

        for k in 0..nz {
            let w0 = Quad::from_f64(model.omega[k]);
            x_0[k] = w0;
            x_1[k] = w0;
            d_x[k] = Quad::ZERO;
        }

        'solve_loop: for num_iters in 1..=MAX_ITERS_PER_SUBSTEP {
            if total_num_iters >= MAX_ITERS_TOTAL {
                outcome = OmegaMixOutcome::Retry(String::from(
                    "solve omega mix failed to converge in allowed number of steps",
                ));
                break 'step_loop;
            }
            total_num_iters += 1;

            if !standalone_mode {
                if model.controls.use_other_torque_implicit {
                    if let Err(e) = extras.other_torque_implicit(model) {
                        outcome = OmegaMixOutcome::Retry(format!(
                            "other_torque_implicit returned error: {e}"
                        ));
                        break 'step_loop;
                    }
                }
                if let Some(hook) = model.aux.binary_other_torque_implicit.clone()
                {
                    if let Err(e) = hook.apply(model) {
                        outcome = OmegaMixOutcome::Retry(format!(
                            "binary_other_torque_implicit returned error: {e}"
                        ));
                        break 'step_loop;
                    }
                }
            }

            let use_implicit_derivs = !standalone_mode
                && (model.controls.use_other_torque_implicit
                    || model.aux.binary_other_torque_implicit.is_some());

            create_matrix_and_rhs(
                model,
                dt,
                &d_x,
                use_implicit_derivs,
                &mut dl,
                &mut d,
                &mut du,
                &mut rhs,
            );

            if solve_tridiag(
                &dl, &d, &du, &rhs, &mut del, &mut bp, &mut vp, &mut xp, nz,
            )
            .is_err()
            {
                outcome = OmegaMixOutcome::Retry(String::from(
                    "matrix solve failed in solve mix",
                ));
                break 'step_loop;
            }

            for k in 1..nz {
                d_x[k] += del[k];
                x_1[k] = x_0[k] + d_x[k];
                model.omega[k] = x_1[k].to_f64();
            }
            model.omega[0] = model.omega[1];

            let mut max_del = Quad::ZERO;
            let mut sum_del = Quad::ZERO;
            for &dk in &del[..nz] {
                let ad = dk.abs();
                if ad > max_del {
                    max_del = ad;
                }
                sum_del += ad;
            }
            let avg_del = sum_del / nz_q;
            if max_del <= tol_correction_max && avg_del <= tol_correction_norm {
                break 'solve_loop;
            }

            if num_iters == MAX_ITERS_PER_SUBSTEP {
                outcome = OmegaMixOutcome::Retry(String::from(
                    "num_iters == max_iters_per_substep in solve mix",
                ));
                break 'step_loop;
            }
        }

        time += dt;
    }

    model.aux.step_hist.num_rotation_solver_steps =
        steps_used.max(model.aux.step_hist.num_rotation_solver_steps);

    if matches!(outcome, OmegaMixOutcome::KeepGoing)
        && total_time - time > tol_time
    {
        outcome = OmegaMixOutcome::Retry(String::from(
            "failed in mixing angular momentum",
        ));
    }

    if matches!(outcome, OmegaMixOutcome::KeepGoing) {
        validate_omega_profile(model, "end solve omega")?;
        for k in 0..nz {
            model.j_rot[k] = model.aux.i_rot[k] * model.omega[k];
        }

        let has_external_torque = !standalone_mode
            && (model.controls.use_other_torque
                || model.controls.use_other_torque_implicit
                || model.aux.binary_other_torque.is_some()
                || model.aux.binary_other_torque_implicit.is_some());
        if !has_external_torque {
            let j_tot1 = Quad::from_f64(eval_j_tot(model, standalone_mode));
            let err_j = (j_tot0 - j_tot1).abs();
            let abs_j0 = j_tot0.abs();
            let retry_tol =
                Quad::from_f64(model.controls.mix_wind.angular_momentum_error_retry);
            let warn_tol =
                Quad::from_f64(model.controls.mix_wind.angular_momentum_error_warn);
            if err_j > retry_tol * abs_j0 {
                let rel_err = if abs_j0 > Quad::ZERO {
                    (err_j / abs_j0).to_f64()
                } else {
                    err_j.to_f64()
                };
                outcome = OmegaMixOutcome::Retry(format!(
                    "retry: failed to conserve angular momentum in mixing (rel_err={rel_err:.3e} > {:.3e})",
                    model.controls.mix_wind.angular_momentum_error_retry
                ));
            } else if err_j > warn_tol * abs_j0
                && crate::step_trace::enabled(model.model_number)
            {
                let rel_err = if abs_j0 > Quad::ZERO {
                    (err_j / abs_j0).to_f64()
                } else {
                    err_j.to_f64()
                };
                let mut jl = crate::step_trace::JsonLine::new("omega_mix_warn");
                jl.field_usize("model", model.model_number);
                jl.field_f64("rel_j_err", rel_err);
                jl.emit();
            }
        }
    }

    Ok(outcome)
}

/// Thomas tridiagonal solver in [`Quad`] (`solve_tridiag` in
/// `star/private/solve_omega_mix.f90:398-433`).
///
/// Layout matches upstream 1-based `dl(k-1)` shifted to 0-based: `sub[i - 1]`
/// is the sub-diagonal entry for row `i` (`i = 1..n`), `diag[i]` is the main
/// diagonal, and `sup[i]` is the super-diagonal for row `i` (`i = 0..n - 1`).
fn solve_tridiag(
    sub: &[Quad],
    diag: &[Quad],
    sup: &[Quad],
    rhs: &[Quad],
    x: &mut [Quad],
    bp: &mut [Quad],
    vp: &mut [Quad],
    xp: &mut [Quad],
    n: usize,
) -> Result<(), ()> {
    if n == 0 {
        return Ok(());
    }
    bp[0] = diag[0];
    vp[0] = rhs[0];
    if !bp[0].is_finite() || bp[0] == Quad::ZERO {
        return Err(());
    }

    for i in 1..n {
        let m = sub[i - 1] / bp[i - 1];
        bp[i] = diag[i] - m * sup[i - 1];
        vp[i] = rhs[i] - m * vp[i - 1];
        if !bp[i].is_finite() || bp[i] == Quad::ZERO {
            return Err(());
        }
    }

    xp[n - 1] = vp[n - 1] / bp[n - 1];
    x[n - 1] = xp[n - 1];
    if !x[n - 1].is_finite() {
        return Err(());
    }
    for i in (0..(n - 1)).rev() {
        xp[i] = (vp[i] - sup[i] * xp[i + 1]) / bp[i];
        x[i] = xp[i];
        if !x[i].is_finite() {
            return Err(());
        }
    }
    Ok(())
}

/// Estimates the minimum local angular-momentum mixing timescale across all
/// faces in `f64` (`min_mixing_timescale` in
/// `star/private/solve_omega_mix.f90:436-521`).
fn min_mixing_timescale(model: &StarModel) -> f64 {
    let nz = model.nz;
    let mut dt = 1.0e99_f64;

    for k in 0..nz {
        let omega = model.omega[k];
        let irot = model.aux.i_rot[k];

        let (c_omega_00, del00_omega, c_irot_00, del00_irot) = if k + 1 < nz {
            let irot_mid_00 = 0.5 * (irot + model.aux.i_rot[k + 1]);
            let am_sig_omega_00 =
                model.aux.am_sig_omega[k] + model.aux.am_sig_j[k];
            let c_om = am_sig_omega_00 * irot_mid_00;
            let del_om = omega - model.omega[k + 1];

            let omega_mid_00 = 0.5 * (omega + model.omega[k + 1]);
            let am_sig_irot_00 = model.aux.am_sig_j[k];
            let c_ir = am_sig_irot_00 * omega_mid_00;
            let del_ir = irot - model.aux.i_rot[k + 1];
            (c_om, del_om, c_ir, del_ir)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };

        let (dmbar, c_omega_m1, delm1_omega, c_irot_m1, delm1_irot) = if k > 0 {
            let dmb = if k + 1 < nz {
                0.5 * (model.dm[k - 1] + model.dm[k])
            } else {
                0.5 * model.dm[k - 1] + model.dm[k]
            };
            let irot_mid_m1 = 0.5 * (model.aux.i_rot[k - 1] + irot);
            let am_sig_omega_m1 =
                model.aux.am_sig_omega[k - 1] + model.aux.am_sig_j[k - 1];
            let c_om = am_sig_omega_m1 * irot_mid_m1;
            let del_om = model.omega[k - 1] - omega;

            let omega_mid_m1 = 0.5 * (model.omega[k - 1] + omega);
            let am_sig_irot_m1 = model.aux.am_sig_j[k - 1];
            let c_ir = am_sig_irot_m1 * omega_mid_m1;
            let del_ir = model.aux.i_rot[k - 1] - irot;
            (dmb, c_om, del_om, c_ir, del_ir)
        } else {
            (0.5 * model.dm[k], 0.0, 0.0, 0.0, 0.0)
        };

        let (d2omega, d2irot) = if k == 0 {
            (-c_omega_00 * del00_omega, -c_irot_00 * del00_irot)
        } else if k + 1 == nz {
            (c_omega_m1 * delm1_omega, c_irot_m1 * delm1_irot)
        } else {
            (
                c_omega_m1 * delm1_omega - c_omega_00 * del00_omega,
                c_irot_m1 * delm1_irot - c_irot_00 * del00_irot,
            )
        };

        let rate = ((d2omega + d2irot) / dmbar
            + model.aux.extra_omegadot[k] * irot
            + model.aux.extra_jdot[k])
            .abs();
        let dt00 = omega.abs().max(1.0e-12) * irot / rate.max(1.0e-50);
        if dt00 < dt {
            dt = dt00;
        }
    }

    dt
}

/// Assembles the tridiagonal Jacobian (`dl`, `d`, `du`) and Newton right-hand
/// side (`rhs = -residual`) in [`Quad`] (`create_matrix_and_rhs` in
/// `star/private/solve_omega_mix.f90:524-665`).
fn create_matrix_and_rhs(
    model: &StarModel,
    dt: Quad,
    d_x: &[Quad],
    use_implicit_derivs: bool,
    dl: &mut [Quad],
    d: &mut [Quad],
    du: &mut [Quad],
    rhs: &mut [Quad],
) {
    let nz = model.nz;

    for k in 0..nz {
        let omega = Quad::from_f64(model.omega[k]);
        let irot = Quad::from_f64(model.aux.i_rot[k]);

        let (
            c_omega_00,
            del00_omega,
            c_irot_00,
            d_c_irot_00_domega_p1,
            d_c_irot_00_domega_00,
            del00_irot,
        ) = if k + 1 < nz {
            let irot_p1 = Quad::from_f64(model.aux.i_rot[k + 1]);
            let omega_p1 = Quad::from_f64(model.omega[k + 1]);
            let irot_mid_00 = (irot + irot_p1) * 0.5;
            // Upstream adds `am_sig_omega(k) + am_sig_j(k)` in `dp` before
            // promoting to `qp` (`solve_omega_mix.f90:553`).
            let am_sig_omega_00 =
                Quad::from_f64(model.aux.am_sig_omega[k] + model.aux.am_sig_j[k]);
            let c_om = am_sig_omega_00 * irot_mid_00;
            let del_om = omega - omega_p1;

            let omega_mid_00 = (omega + omega_p1) * 0.5;
            let am_sig_irot_00 = Quad::from_f64(model.aux.am_sig_j[k]);
            let c_ir = am_sig_irot_00 * omega_mid_00;
            let dc_p1 = am_sig_irot_00 * 0.5;
            let dc_00 = am_sig_irot_00 * 0.5;
            let del_ir = irot - irot_p1;
            (c_om, del_om, c_ir, dc_p1, dc_00, del_ir)
        } else {
            (
                Quad::ZERO,
                Quad::ZERO,
                Quad::ZERO,
                Quad::ZERO,
                Quad::ZERO,
                Quad::ZERO,
            )
        };

        let (
            dmbar,
            c_omega_m1,
            delm1_omega,
            c_irot_m1,
            d_c_irot_m1_domega_00,
            d_c_irot_m1_domega_m1,
            delm1_irot,
        ) = if k > 0 {
            let dmb = Quad::from_f64(if k + 1 < nz {
                0.5 * (model.dm[k - 1] + model.dm[k])
            } else {
                0.5 * model.dm[k - 1] + model.dm[k]
            });
            let irot_m1 = Quad::from_f64(model.aux.i_rot[k - 1]);
            let omega_m1 = Quad::from_f64(model.omega[k - 1]);
            let irot_mid_m1 = (irot_m1 + irot) * 0.5;
            let am_sig_omega_m1 = Quad::from_f64(
                model.aux.am_sig_omega[k - 1] + model.aux.am_sig_j[k - 1],
            );
            let c_om = am_sig_omega_m1 * irot_mid_m1;
            let del_om = omega_m1 - omega;

            let omega_mid_m1 = (omega_m1 + omega) * 0.5;
            let am_sig_irot_m1 = Quad::from_f64(model.aux.am_sig_j[k - 1]);
            let c_ir = am_sig_irot_m1 * omega_mid_m1;
            let dc_00 = am_sig_irot_m1 * 0.5;
            let dc_m1 = am_sig_irot_m1 * 0.5;
            let del_ir = irot_m1 - irot;
            (dmb, c_om, del_om, c_ir, dc_00, dc_m1, del_ir)
        } else {
            (
                Quad::from_f64(0.5 * model.dm[k]),
                Quad::ZERO,
                Quad::ZERO,
                Quad::ZERO,
                Quad::ZERO,
                Quad::ZERO,
                Quad::ZERO,
            )
        };

        let (d2omega, d2irot) = if k == 0 {
            (-c_omega_00 * del00_omega, -c_irot_00 * del00_irot)
        } else if k + 1 == nz {
            (c_omega_m1 * delm1_omega, c_irot_m1 * delm1_irot)
        } else {
            (
                c_omega_m1 * delm1_omega - c_omega_00 * del00_omega,
                c_irot_m1 * delm1_irot - c_irot_00 * del00_irot,
            )
        };
        let d_d2omega_domega_00 = -(c_omega_m1 + c_omega_00);
        let d_d2irot_domega_00 = d_c_irot_m1_domega_00 * delm1_irot
            - d_c_irot_00_domega_00 * del00_irot;

        let extra_jdot = Quad::from_f64(model.aux.extra_jdot[k]);
        let extra_omegadot = Quad::from_f64(model.aux.extra_omegadot[k]);
        rhs[k] = -d_x[k]
            + dt * (((d2omega + d2irot) / dmbar + extra_jdot) / irot
                + extra_omegadot);

        let f = dt / (dmbar * irot);
        d[k] = Quad::ONE - (d_d2omega_domega_00 + d_d2irot_domega_00) * f;

        if k + 1 < nz {
            let d_d2omega_domega_p1 = c_omega_00;
            let d_d2irot_domega_p1 = -d_c_irot_00_domega_p1 * del00_irot;
            du[k] = -(d_d2omega_domega_p1 + d_d2irot_domega_p1) * f;
        }

        if k > 0 {
            let d_d2omega_domega_m1 = c_omega_m1;
            let d_d2irot_domega_m1 = d_c_irot_m1_domega_m1 * delm1_irot;
            dl[k - 1] = -(d_d2omega_domega_m1 + d_d2irot_domega_m1) * f;
        }

        if use_implicit_derivs {
            d[k] -= dt
                * (Quad::from_f64(model.aux.d_extra_jdot_domega_00[k]) / irot
                    + Quad::from_f64(model.aux.d_extra_omegadot_domega_00[k]));
            if k + 1 < nz {
                du[k] -= dt
                    * (Quad::from_f64(model.aux.d_extra_jdot_domega_p1[k])
                        / irot
                        + Quad::from_f64(
                            model.aux.d_extra_omegadot_domega_p1[k],
                        ));
            }
            if k > 0 {
                dl[k - 1] -= dt
                    * (Quad::from_f64(model.aux.d_extra_jdot_domega_m1[k])
                        / irot
                        + Quad::from_f64(
                            model.aux.d_extra_omegadot_domega_m1[k],
                        ));
            }
        }
    }
}
