//! L-stable Rosenbrock-Wanner stiff ODE/DAE integrators (`num`).
//!
//! Ports Hairer & Wanner's Rosenbrock family (`num/private/mod_rosenbrock.f`)
//! supporting all 7 MESA Rosenbrock variants (`Ros2`, `Rose2`, `Ros3p`,
//! `Ros3pl`, `Rodas3`, `Rodas4`, `Rodasp`) with dense and banded Jacobians
//! solved via `flexmesa-mtx` (`solve_dense`, `solve_banded`).
//!
//! # Stiff Integration Strategy
//!
//! Unlike explicit Runge-Kutta methods whose stability domains are bounded,
//! L-stable Rosenbrock-Wanner methods damp stiff eigenmodes $(\lambda h \to
//! -\infty)$ in a single step by incorporating the exact Jacobian $J =
//! \partial f / \partial y$ directly into the linear stage operator
//! $E = (\gamma h)^{-1} I - J$.
//!
//! - [`rosenbrock_dense`] forms the dense $(N, N)$ matrix $E$, computes its
//!   LU factorization once per step via [`dgetrf`], and solves each stage in
//!   $O(N^2)$ forward/back substitutions via [`dgetrs`].
//! - [`rosenbrock_banded`] packs $E$ into LAPACK banded storage `(2 * ml + mu
//!   + 1, N)`, factors it once per step via [`dgbtrf`], and solves each stage
//!   in $O(N (m_l + m_u))$ substitutions via [`dgbtrs`].
//!
//! # Step Rejection & Error Norms
//!
//! - At each step, the embedded error vector $e = \sum_j e_j g_j$ is scaled by
//!   $\mathrm{atol} + \mathrm{rtol} \max(|y_i|, |y_{\mathrm{new}, i}|)$ and
//!   measured in the root-mean-square norm.
//! - Non-finite error norms, mismatched callback vector lengths, and step-size
//!   underflow (`t + h == t`) return explicit [`NumError`] variants rather than
//!   propagating `NaN` states silently.
//! - Zero-span calls (`t0 == t_end`) return immediately with zero evaluations.
//! - Stage right-hand sides include the non-autonomous temporal term
//!   $h \gamma_{t,i} (\partial f / \partial t)$ alongside the stage couplings.

use crate::rosenbrock_coeffs::{coeffs_for, RosenbrockVariant};
use crate::NumError;
use flexmesa_mtx::{dense_to_band, dgbtrf, dgbtrs, dgetrf, dgetrs};

/// Builds the dense stage matrix $E = (\gamma h)^{-1} I - J$ in row-major order
/// across all $N \times N$ entries by negating $J$ and adding $(\gamma h)^{-1}$
/// along the main diagonal.
fn build_stage_matrix(
    n: usize,
    h_gamma: f64,
    j_mat: &[f64],
) -> Result<Vec<f64>, NumError> {
    if j_mat.len() != n * n {
        return Err(NumError::InvalidParameter(
            "Jacobian length must equal n * n in rosenbrock",
        ));
    }
    let mut e_mat = j_mat.iter().map(|&v| -v).collect::<Vec<f64>>();
    let inv_hg = 1.0 / h_gamma;
    for i in 0..n {
        e_mat[i * n + i] += inv_hg;
    }
    Ok(e_mat)
}

/// Summary of an adaptive Rosenbrock integration.
#[derive(Debug, Clone)]
pub struct RosenbrockResult {
    /// Final state vector $y(t_{\mathrm{end}})$ of length $n$.
    pub y: Vec<f64>,
    /// Number of right-hand side evaluations (`nfcn`).
    pub nfcn: usize,
    /// Number of Jacobian evaluations (`njac`).
    pub njac: usize,
    /// Number of accepted steps (`naccpt`).
    pub naccpt: usize,
    /// Number of rejected steps (`nrejct`).
    pub nrejct: usize,
}

/// Integrates a stiff ODE system $y'(t) = f(t, y)$ with dense $(N, N)$
/// Jacobian $J = \partial f / \partial y$ from `t0` to `t_end` using the
/// selected [`RosenbrockVariant`].
///
/// # Errors
///
/// Returns [`NumError::InvalidParameter`] on invalid tolerances, or
/// [`NumError::ConvergenceFailed`] / [`NumError::LinearSolveFailed`] if the
/// step budget or linear solver fails.
pub fn rosenbrock_dense<F, J>(
    variant: RosenbrockVariant,
    t0: f64,
    t_end: f64,
    y0: &[f64],
    h0: f64,
    hmax_in: f64,
    rtol: f64,
    atol: f64,
    max_steps: usize,
    mut fcn: F,
    mut jac: J,
) -> Result<RosenbrockResult, NumError>
where
    F: FnMut(f64, &[f64]) -> Result<Vec<f64>, NumError>,
    J: FnMut(f64, &[f64]) -> Result<Vec<f64>, NumError>,
{
    rosenbrock_core(
        variant,
        t0,
        t_end,
        y0,
        h0,
        hmax_in,
        rtol,
        atol,
        max_steps,
        &mut fcn,
        |t, y, h_gamma| {
            let n = y.len();
            let j_mat = jac(t, y)?;
            let mut e_mat = build_stage_matrix(n, h_gamma, &j_mat)?;
            let mut ipiv = vec![0usize; n];
            dgetrf(n, &mut e_mat, &mut ipiv).map_err(|_| {
                NumError::LinearSolveFailed("rosenbrock_dense LU")
            })?;
            Ok((e_mat, ipiv))
        },
        |n, lu, ipiv, rhs| {
            let mut x = rhs.to_vec();
            dgetrs(n, lu, ipiv, &mut x).map_err(|_| {
                NumError::LinearSolveFailed("rosenbrock_dense dgetrs")
            })?;
            Ok(x)
        },
    )
}

/// Integrates a stiff ODE system $y'(t) = f(t, y)$ with banded Jacobian
/// (`ml` sub-diagonals, `mu` super-diagonals) from `t0` to `t_end` using the
/// selected [`RosenbrockVariant`].
///
/// The callback `jac_dense(t, y)` returns the $(N, N)$ row-major Jacobian,
/// which is packed into banded storage `(2 * ml + mu + 1, N)` and factored
/// once per step via [`dgbtrf`].
///
/// # Errors
///
/// Propagates any error from [`rosenbrock_dense`].
pub fn rosenbrock_banded<F, J>(
    variant: RosenbrockVariant,
    ml: usize,
    mu: usize,
    t0: f64,
    t_end: f64,
    y0: &[f64],
    h0: f64,
    hmax_in: f64,
    rtol: f64,
    atol: f64,
    max_steps: usize,
    mut fcn: F,
    mut jac_dense: J,
) -> Result<RosenbrockResult, NumError>
where
    F: FnMut(f64, &[f64]) -> Result<Vec<f64>, NumError>,
    J: FnMut(f64, &[f64]) -> Result<Vec<f64>, NumError>,
{
    let ldab = 2 * ml + mu + 1;
    rosenbrock_core(
        variant,
        t0,
        t_end,
        y0,
        h0,
        hmax_in,
        rtol,
        atol,
        max_steps,
        &mut fcn,
        |t, y, h_gamma| {
            let n = y.len();
            let j_mat = jac_dense(t, y)?;
            let e_mat = build_stage_matrix(n, h_gamma, &j_mat)?;
            let mut ab = vec![0.0; ldab * n];
            dense_to_band(n, n, &e_mat, ml, mu, &mut ab, ldab).map_err(
                |_| NumError::LinearSolveFailed("dense_to_band failed"),
            )?;
            let mut ipiv = vec![0usize; n];
            dgbtrf(n, ml, mu, &mut ab, ldab, &mut ipiv).map_err(|_| {
                NumError::LinearSolveFailed("rosenbrock_banded LU")
            })?;
            Ok((ab, ipiv))
        },
        |n, lu_ab, ipiv, rhs| {
            let mut x = rhs.to_vec();
            dgbtrs(n, ml, mu, lu_ab, ldab, ipiv, &mut x).map_err(|_| {
                NumError::LinearSolveFailed("rosenbrock_banded dgbtrs")
            })?;
            Ok(x)
        },
    )
}

/// Shared adaptive step-size loop for [`rosenbrock_dense`] and
/// [`rosenbrock_banded`].
///
/// Evaluates the temporal partial derivative $\partial f / \partial t$ via
/// forward difference and factors $E = (\gamma h)^{-1} I - J$ once per step,
/// then solves the $s$ stage linear systems using the factored operator.
fn rosenbrock_core<F, P, S>(
    variant: RosenbrockVariant,
    t0: f64,
    t_end: f64,
    y0: &[f64],
    h0: f64,
    hmax_in: f64,
    rtol: f64,
    atol: f64,
    max_steps: usize,
    fcn: &mut F,
    mut factor_step: P,
    mut solve_stage: S,
) -> Result<RosenbrockResult, NumError>
where
    F: FnMut(f64, &[f64]) -> Result<Vec<f64>, NumError>,
    P: FnMut(f64, &[f64], f64) -> Result<(Vec<f64>, Vec<usize>), NumError>,
    S: FnMut(usize, &[f64], &[usize], &[f64]) -> Result<Vec<f64>, NumError>,
{
    let n = y0.len();
    if n == 0 || rtol <= 0.0 || atol <= 0.0 {
        return Err(NumError::InvalidParameter(
            "require y0.len() >= 1 and rtol, atol > 0 in rosenbrock",
        ));
    }
    if t0 == t_end {
        return Ok(RosenbrockResult {
            y: y0.to_vec(),
            nfcn: 0,
            njac: 0,
            naccpt: 0,
            nrejct: 0,
        });
    }
    let rc = coeffs_for(variant);
    let posneg = (t_end - t0).signum();
    let hmax = if hmax_in <= 0.0 {
        (t_end - t0).abs()
    } else {
        hmax_in.abs()
    };
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut h = if h0 == 0.0 {
        1e-5 * posneg
    } else {
        h0.abs().min(hmax) * posneg
    };
    let nmax = if max_steps == 0 { 100_000 } else { max_steps };
    let mut nfcn = 0usize;
    let mut naccpt = 0usize;
    let mut nrejct = 0usize;

    for (njac, _step) in (0..nmax).enumerate() {
        if (t - t_end).abs() <= 1e-15 * t_end.abs().max(1.0) {
            return Ok(RosenbrockResult {
                y,
                nfcn,
                njac,
                naccpt,
                nrejct,
            });
        }
        if (t + 1.01 * h - t_end) * posneg > 0.0 {
            h = t_end - t;
        }
        if t + h == t {
            return Err(NumError::ConvergenceFailed(
                "step size underflow in rosenbrock",
            ));
        }
        let f0 = fcn(t, &y)?;
        let dt_eps = 1e-8 * t.abs().max(1.0);
        let f_dt = fcn(t + dt_eps, &y)?;
        if f0.len() != n || f_dt.len() != n {
            return Err(NumError::InvalidParameter(
                "derivative vector length mismatch in rosenbrock",
            ));
        }
        nfcn += 2;
        let df_dt: Vec<f64> =
            (0..n).map(|i| (f_dt[i] - f0[i]) / dt_eps).collect();

        let (lu, ipiv) = factor_step(t, &y, h * rc.gamma)?;
        // Solve linear stage equations E * g_stg = rhs_stg for stg = 0..s
        let mut g_stages: Vec<Vec<f64>> = Vec::with_capacity(rc.s);
        for stg in 0..rc.s {
            let mut y_stg = y.clone();
            for j in 0..stg {
                let a_ij = rc.a[stg][j];
                if a_ij != 0.0 {
                    for i in 0..n {
                        y_stg[i] += a_ij * g_stages[j][i];
                    }
                }
            }
            let f_stg = if stg == 0 {
                f0.clone()
            } else {
                nfcn += 1;
                let fs = fcn(t + rc.alpha[stg] * h, &y_stg)?;
                if fs.len() != n {
                    return Err(NumError::InvalidParameter(
                        "derivative vector length mismatch in rosenbrock",
                    ));
                }
                fs
            };
            let mut rhs = f_stg;
            for i in 0..n {
                rhs[i] += h * rc.gamma_t[stg] * df_dt[i];
            }
            for j in 0..stg {
                let c_ij_over_h = rc.c[stg][j] / h;
                for i in 0..n {
                    rhs[i] += c_ij_over_h * g_stages[j][i];
                }
            }
            let g_i = solve_stage(n, &lu, &ipiv, &rhs)?;
            g_stages.push(g_i);
        }

        // Combine stage increments into candidate state y_new and error err_vec
        let mut y_new = y.clone();
        let mut err_vec = vec![0.0; n];
        for j in 0..rc.s {
            for i in 0..n {
                y_new[i] += rc.m[j] * g_stages[j][i];
                err_vec[i] += rc.e[j] * g_stages[j][i];
            }
        }

        // Scale local error vector by mixed absolute + relative tolerance sk
        let mut err_sq = 0.0_f64;
        for i in 0..n {
            let sk = atol + rtol * y[i].abs().max(y_new[i].abs());
            let r = err_vec[i] / sk;
            err_sq += r * r;
        }
        let err = (err_sq / (n as f64)).sqrt();
        if !err.is_finite() {
            return Err(NumError::NonFiniteValue(
                "non-finite local error in rosenbrock",
            ));
        }
        // Accept step if RMS error <= 1.0 and propose next step size h
        // using safety factor 0.9 and clamping h_new / h to [0.1, 5.0]
        let fac = (0.9 * err.max(1e-10).powf(-1.0 / rc.order)).clamp(0.1, 5.0);
        if err <= 1.0 {
            t += h;
            y = y_new;
            naccpt += 1;
            h = posneg * (h.abs() * fac).min(hmax);
        } else {
            nrejct += 1;
            h = posneg * (h.abs() * fac.min(0.5));
        }
    }
    Err(NumError::ConvergenceFailed("rosenbrock exceeded max_steps"))
}
