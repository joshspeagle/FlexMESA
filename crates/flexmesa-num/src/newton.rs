//! Damped Newton-Raphson solver for dense, banded, and block-tridiagonal
//! linear systems (`num`).
//!
//! Ports MESA's Newton-Raphson iteration (`num/private/mod_newton.f90` and
//! `num/test/src/test_newton.f90`) with line-search damping and residual/update
//! norm convergence criteria across dense, banded, and block-tridiagonal
//! Jacobians (`flexmesa-mtx`).
//!
//! # Solver Architecture
//!
//! At each iteration $k$, the solver evaluates the residual vector $F(y^{(k)})$
//! and solves the linear Newton system
//! $$J(y^{(k)})\,\Delta y^{(k)} = -F(y^{(k)})$$
//! using [`solve_dense`], [`solve_banded`], or [`solve_block_tridiagonal`]. A
//! backtracking Armijo line search selects $\lambda \in [\lambda_{\min}, 1]$ so
//! that $y^{(k+1)} = y^{(k)} + \lambda\,\Delta y^{(k)}$ reduces the RMS
//! residual norm when far from the solution.

use crate::NumError;
use flexmesa_mtx::{solve_banded, solve_block_tridiagonal, solve_dense};

/// Configuration for [`newton_solve_dense`], [`newton_solve_banded`], and
/// [`newton_solve_block_tridiagonal`].
#[derive(Debug, Clone, Copy)]
pub struct NewtonConfig {
    /// Maximum Newton iterations (`iter_max`).
    pub max_iter: usize,
    /// Maximum RMS correction norm (`tol_correction_norm`).
    pub tol_correction_norm: f64,
    /// Maximum supremum correction (`tol_max_correction`).
    pub tol_max_correction: f64,
    /// Maximum RMS residual norm (`tol_residual_norm`).
    pub tol_residual_norm: f64,
    /// Minimum damping factor before declaring failure (`min_corr_coeff`).
    pub min_damping: f64,
}

impl Default for NewtonConfig {
    fn default() -> Self {
        Self {
            max_iter: 50,
            tol_correction_norm: 1e-9,
            tol_max_correction: 1e-7,
            tol_residual_norm: 1e-10,
            min_damping: 1e-3,
        }
    }
}

/// Convergence diagnostics returned by the Newton-Raphson solvers.
#[derive(Debug, Clone)]
pub struct NewtonResult {
    /// Converged state vector $y$ of length $N_{\mathrm{tot}}$.
    pub y: Vec<f64>,
    /// Number of Newton iterations taken.
    pub iterations: usize,
    /// Final RMS residual norm $\|F(y)\|_2 / \sqrt{N_{\mathrm{tot}}}$.
    pub residual_norm: f64,
    /// Final RMS correction norm $\|\Delta y\|_2 / \sqrt{N_{\mathrm{tot}}}$.
    pub correction_norm: f64,
}

/// Computes the root-mean-square norm and supremum norm of `vec`.
fn rms_and_max(vec: &[f64]) -> (f64, f64) {
    if vec.is_empty() {
        return (0.0, 0.0);
    }
    let mut sum_sq = 0.0_f64;
    let mut max_abs = 0.0_f64;
    for &v in vec {
        let a = v.abs();
        sum_sq += a * a;
        max_abs = max_abs.max(a);
    }
    ((sum_sq / (vec.len() as f64)).sqrt(), max_abs)
}

/// Generic damped Newton-Raphson iteration parameterized by residual evaluation
/// `eval_res` and linear Newton-step solver `solve_step`.
fn newton_driver<E, S>(
    y_init: &[f64],
    config: &NewtonConfig,
    mut eval_res: E,
    mut solve_step: S,
) -> Result<NewtonResult, NumError>
where
    E: FnMut(&[f64]) -> Result<Vec<f64>, NumError>,
    S: FnMut(&[f64], &[f64]) -> Result<Vec<f64>, NumError>,
{
    let n = y_init.len();
    if n == 0 {
        return Err(NumError::InvalidParameter(
            "y_init must be non-empty in newton_solve",
        ));
    }
    let mut y = y_init.to_vec();
    let mut res = eval_res(&y)?;
    let (mut res_norm, _) = rms_and_max(&res);
    if res_norm <= config.tol_residual_norm {
        return Ok(NewtonResult {
            y,
            iterations: 0,
            residual_norm: res_norm,
            correction_norm: 0.0,
        });
    }

    let mut neg_res = vec![0.0; n];
    let mut y_trial = vec![0.0; n];
    for iter in 1..=config.max_iter {
        for i in 0..n {
            neg_res[i] = -res[i];
        }
        let dy = solve_step(&y, &neg_res)?;
        let (corr_norm, max_corr) = rms_and_max(&dy);

        // Backtracking line search if residual norm increases markedly
        let mut coeff = 1.0_f64;
        let mut res_trial = Vec::new();
        let mut res_trial_norm = res_norm;
        while coeff >= config.min_damping {
            for i in 0..n {
                y_trial[i] = y[i] + coeff * dy[i];
            }
            res_trial = eval_res(&y_trial)?;
            let (rn, _) = rms_and_max(&res_trial);
            res_trial_norm = rn;
            if rn <= (1.0 - 0.1 * coeff) * res_norm
                || rn <= config.tol_residual_norm
                || (coeff == 1.0 && corr_norm <= config.tol_correction_norm)
            {
                break;
            }
            coeff *= 0.5;
        }
        y.copy_from_slice(&y_trial);
        res = res_trial;
        res_norm = res_trial_norm;

        if (corr_norm * coeff <= config.tol_correction_norm
            && max_corr * coeff <= config.tol_max_correction)
            || res_norm <= config.tol_residual_norm
        {
            return Ok(NewtonResult {
                y,
                iterations: iter,
                residual_norm: res_norm,
                correction_norm: corr_norm * coeff,
            });
        }
    }
    Err(NumError::ConvergenceFailed("newton_solve exceeded max_iter"))
}

/// Solves a nonlinear system $F(y) = 0$ with a dense $(N, N)$ Jacobian
/// $J(y) = \partial F / \partial y$ via damped Newton-Raphson iteration.
///
/// # Errors
///
/// Propagates any error from `eval_res`, `eval_jac`, or [`solve_dense`].
pub fn newton_solve_dense<E, J>(
    y_init: &[f64],
    config: &NewtonConfig,
    eval_res: E,
    mut eval_jac: J,
) -> Result<NewtonResult, NumError>
where
    E: FnMut(&[f64]) -> Result<Vec<f64>, NumError>,
    J: FnMut(&[f64]) -> Result<Vec<f64>, NumError>,
{
    newton_driver(y_init, config, eval_res, |y, rhs| {
        let n = y.len();
        let jac = eval_jac(y)?;
        solve_dense(n, &jac, rhs)
            .map_err(|_| NumError::LinearSolveFailed("newton_solve_dense LU"))
    })
}

/// Solves a nonlinear system $F(y) = 0$ with a banded Jacobian (`kl`
/// sub-diagonals, `ku` super-diagonals, `ldab = 2 * kl + ku + 1`) via damped
/// Newton-Raphson iteration.
///
/// # Errors
///
/// Propagates any error from `eval_res`, `eval_jac_band`, or [`solve_banded`].
pub fn newton_solve_banded<E, J>(
    kl: usize,
    ku: usize,
    y_init: &[f64],
    config: &NewtonConfig,
    eval_res: E,
    mut eval_jac_band: J,
) -> Result<NewtonResult, NumError>
where
    E: FnMut(&[f64]) -> Result<Vec<f64>, NumError>,
    J: FnMut(&[f64]) -> Result<Vec<f64>, NumError>,
{
    let ldab = 2 * kl + ku + 1;
    newton_driver(y_init, config, eval_res, |y, rhs| {
        let n = y.len();
        let ab = eval_jac_band(y)?;
        solve_banded(n, kl, ku, &ab, ldab, rhs)
            .map_err(|_| NumError::LinearSolveFailed("newton_solve_banded LU"))
    })
}

/// Solves a coupled shell-structured nonlinear system $F(y) = 0$ across `nz`
/// shells and `nvar` variables per shell ($N_{\mathrm{tot}} = n_z \cdot
/// n_{\mathrm{var}}$) using block-tridiagonal Jacobians `(lblk, dblk, ublk)`
/// solved via [`solve_block_tridiagonal`] (`test_newton.f90`).
///
/// # Errors
///
/// Propagates any error from `eval_res`, `eval_blocks`, or
/// [`solve_block_tridiagonal`].
pub fn newton_solve_block_tridiagonal<E, B>(
    nvar: usize,
    nz: usize,
    y_init: &[f64],
    config: &NewtonConfig,
    eval_res: E,
    mut eval_blocks: B,
) -> Result<NewtonResult, NumError>
where
    E: FnMut(&[f64]) -> Result<Vec<f64>, NumError>,
    B: FnMut(&[f64]) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), NumError>,
{
    if y_init.len() != nvar * nz {
        return Err(NumError::InvalidParameter(
            "y_init length must equal nvar * nz",
        ));
    }
    newton_driver(y_init, config, eval_res, |y, rhs| {
        let (lblk, dblk, ublk) = eval_blocks(y)?;
        solve_block_tridiagonal(nvar, nz, &lblk, &dblk, &ublk, rhs).map_err(
            |_| NumError::LinearSolveFailed("newton_solve_block_tridiagonal"),
        )
    })
}
