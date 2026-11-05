//! Explicit Runge-Kutta (`rk2`, `rk4`, `dopri5`, `dop853`) ODE integrators.
//!
//! Ports Hairer & Wanner's Dormand-Prince 5(4) (`num/private/mod_dopri5.f`)
//! with Shampine 4th-order dense output (`contd5`) and high-order adaptive
//! Runge-Kutta integration into Safe Rust in `f64` precision.
//!
//! # Integrator Family
//!
//! - [`rk2`]: Explicit midpoint 2nd-order single-step Runge-Kutta integrator.
//! - [`rk4`]: Classical 4-stage 4th-order single-step Runge-Kutta integrator.
//! - [`dopri5`]: 7-stage First-Same-As-Last (FSAL) Dormand-Prince 5(4) adaptive
//!   integrator with proportional-integral (PI) step-size control ($\beta =
//!   0.04$) and continuous 4th-order dense polynomial output (`contd5`).
//! - [`dop853`]: High-precision adaptive Dormand-Prince integration API
//!   compatible with [`DopriResult`].

use crate::rk_tableau::{A21, A3, A4, A5, A6, A7, C, D, E};
use crate::NumError;

pub use crate::rk_tableau::DopriResult;

/// Performs a single midpoint 2nd-order Runge-Kutta step $y(t + h)$ for
/// $y' = f(t, y)$.
///
/// Evaluates $k_1 = f(t, y)$ and $k_2 = f(t + \tfrac{1}{2} h, y + \tfrac{1}{2}
/// h k_1)$ and returns $y + h k_2$.
///
/// # Errors
///
/// Propagates any error returned by `derivs`.
pub fn rk2<F>(
    t: f64,
    y: &[f64],
    h: f64,
    mut derivs: F,
) -> Result<Vec<f64>, NumError>
where
    F: FnMut(f64, &[f64]) -> Result<Vec<f64>, NumError>,
{
    let n = y.len();
    let k1 = derivs(t, y)?;
    let y_mid: Vec<f64> = (0..n).map(|i| y[i] + 0.5 * h * k1[i]).collect();
    let k2 = derivs(t + 0.5 * h, &y_mid)?;
    Ok((0..n).map(|i| y[i] + h * k2[i]).collect())
}

/// Performs a single classical 4th-order Runge-Kutta step $y(t + h)$ for
/// $y' = f(t, y)$.
///
/// Evaluates stages $k_1, k_2, k_3, k_4$ at $t, t + \tfrac{1}{2} h, t +
/// \tfrac{1}{2} h, t + h$ and combines them with weights $(1, 2, 2, 1)/6$.
///
/// # Errors
///
/// Propagates any error returned by `derivs`.
pub fn rk4<F>(
    t: f64,
    y: &[f64],
    h: f64,
    mut derivs: F,
) -> Result<Vec<f64>, NumError>
where
    F: FnMut(f64, &[f64]) -> Result<Vec<f64>, NumError>,
{
    let n = y.len();
    let k1 = derivs(t, y)?;
    let y2: Vec<f64> = (0..n).map(|i| y[i] + 0.5 * h * k1[i]).collect();
    let k2 = derivs(t + 0.5 * h, &y2)?;
    let y3: Vec<f64> = (0..n).map(|i| y[i] + 0.5 * h * k2[i]).collect();
    let k3 = derivs(t + 0.5 * h, &y3)?;
    let y4: Vec<f64> = (0..n).map(|i| y[i] + h * k3[i]).collect();
    let k4 = derivs(t + h, &y4)?;
    Ok((0..n)
        .map(|i| {
            y[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
        })
        .collect())
}

/// Integrates the ODE system $y'(x) = f(x, y)$ from `x0` to `x_end` using
/// Dormand-Prince 5(4) (`dopri5` in `num/private/mod_dopri5.f`) with PI
/// step-size control and Shampine 4th-order dense output (`contd5`).
///
/// # Arguments
///
/// - `x0`, `x_end`: Initial and final independent variable values.
/// - `y0`: Initial state slice of length $n \ge 1$.
/// - `h0`: Initial trial step size (`0.0` selects an automatic default).
/// - `hmax_in`: Maximum step size magnitude (`0.0` defaults to
///   $|x_{\mathrm{end}} - x_0|$).
/// - `rtol`, `atol`: Scalar relative and absolute local error tolerances.
/// - `max_steps`: Maximum number of integration steps (`0` defaults to
///   `100_000`).
///
/// # Errors
///
/// Returns [`NumError::InvalidParameter`] on invalid tolerances, or
/// [`NumError::ConvergenceFailed`] if `max_steps` is exceeded.
pub fn dopri5<F>(
    x0: f64,
    x_end: f64,
    y0: &[f64],
    h0: f64,
    hmax_in: f64,
    rtol: f64,
    atol: f64,
    max_steps: usize,
    mut fcn: F,
) -> Result<DopriResult, NumError>
where
    F: FnMut(f64, &[f64]) -> Result<Vec<f64>, NumError>,
{
    let n = y0.len();
    if n == 0 || rtol <= 0.0 || atol <= 0.0 {
        return Err(NumError::InvalidParameter(
            "require y0.len() >= 1 and rtol, atol > 0 in dopri5",
        ));
    }
    if x0 == x_end {
        return Ok(DopriResult {
            y: y0.to_vec(),
            nfcn: 0,
            nstep: 0,
            naccpt: 0,
            nrejct: 0,
            history: Vec::new(),
        });
    }

    let posneg = (x_end - x0).signum();
    let hmax = if hmax_in <= 0.0 {
        (x_end - x0).abs()
    } else {
        hmax_in.abs()
    };
    let mut x = x0;
    let mut y = y0.to_vec();
    let mut k1 = fcn(x, &y)?;
    let mut nfcn = 1usize;
    let mut h = if h0 == 0.0 {
        1e-4 * posneg
    } else {
        h0.abs().min(hmax) * posneg
    };

    let nmax = if max_steps == 0 { 100_000 } else { max_steps };
    let mut nstep = 0usize;
    let mut naccpt = 0usize;
    let mut nrejct = 0usize;
    let mut facold = 1e-4_f64;
    let mut reject = false;
    let mut history = Vec::new();
    let mut y_stg = vec![0.0; n];
    let mut y_new = vec![0.0; n];

    loop {
        if nstep > nmax {
            return Err(NumError::ConvergenceFailed(
                "dopri5 exceeded max_steps",
            ));
        }
        if (x - x_end).abs() <= 1e-15 * x_end.abs().max(1.0) {
            break;
        }
        if (x + 1.01 * h - x_end) * posneg > 0.0 {
            h = x_end - x;
        }
        nstep += 1;

        // Evaluate stages k2..k7 using First-Same-As-Last (FSAL) stage k1
        for i in 0..n {
            y_stg[i] = y[i] + h * A21 * k1[i];
        }
        let k2 = fcn(x + C[0] * h, &y_stg)?;
        for i in 0..n {
            y_stg[i] = y[i] + h * (A3[0] * k1[i] + A3[1] * k2[i]);
        }
        let k3 = fcn(x + C[1] * h, &y_stg)?;
        for i in 0..n {
            y_stg[i] =
                y[i] + h * (A4[0] * k1[i] + A4[1] * k2[i] + A4[2] * k3[i]);
        }
        let k4 = fcn(x + C[2] * h, &y_stg)?;
        for i in 0..n {
            y_stg[i] = y[i]
                + h * (A5[0] * k1[i]
                    + A5[1] * k2[i]
                    + A5[2] * k3[i]
                    + A5[3] * k4[i]);
        }
        let k5 = fcn(x + C[3] * h, &y_stg)?;
        for i in 0..n {
            y_stg[i] = y[i]
                + h * (A6[0] * k1[i]
                    + A6[1] * k2[i]
                    + A6[2] * k3[i]
                    + A6[3] * k4[i]
                    + A6[4] * k5[i]);
        }
        let xph = x + h;
        let k6 = fcn(xph, &y_stg)?;
        for i in 0..n {
            y_new[i] = y[i]
                + h * (A7[0] * k1[i]
                    + A7[2] * k3[i]
                    + A7[3] * k4[i]
                    + A7[4] * k5[i]
                    + A7[5] * k6[i]);
        }
        let k7 = fcn(xph, &y_new)?;
        nfcn += 6;

        // Compute scaled RMS error norm across components
        let mut err_sq = 0.0_f64;
        for i in 0..n {
            let err_i = h
                * (E[0] * k1[i]
                    + E[2] * k3[i]
                    + E[3] * k4[i]
                    + E[4] * k5[i]
                    + E[5] * k6[i]
                    + E[6] * k7[i]);
            let sk = atol + rtol * y[i].abs().max(y_new[i].abs());
            let ratio = err_i / sk;
            err_sq += ratio * ratio;
        }
        let err = (err_sq / (n as f64)).sqrt();
        // Lund PI step-size controller (beta = 0.04, expo1 = 0.2 - 0.75 * beta)
        let fac11 = err.powf(0.17);
        let fac = fac11 / facold.powf(0.04);
        let fac_clamped = (fac / 0.9).clamp(0.1, 5.0);
        let mut h_new = h / fac_clamped;

        if err <= 1.0 {
            facold = err.max(1e-4);
            naccpt += 1;
            // Record 4th-order dense output polynomial coefficients rcont[0..5]
            let mut rc = [
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n],
            ];
            for i in 0..n {
                let ydiff = y_new[i] - y[i];
                let bspl = h * k1[i] - ydiff;
                rc[0][i] = y[i];
                rc[1][i] = ydiff;
                rc[2][i] = bspl;
                rc[3][i] = -h * k7[i] + ydiff - bspl;
                rc[4][i] = h
                    * (D[0] * k1[i]
                        + D[2] * k3[i]
                        + D[3] * k4[i]
                        + D[4] * k5[i]
                        + D[5] * k6[i]
                        + D[6] * k7[i]);
            }
            history.push((x, h, rc));
            k1 = k7;
            y.copy_from_slice(&y_new);
            x = xph;
            if h_new.abs() > hmax {
                h_new = posneg * hmax;
            }
            if reject {
                h_new = posneg * h_new.abs().min(h.abs());
            }
            reject = false;
        } else {
            h_new = h / (fac11 / 0.9).min(5.0);
            reject = true;
            if naccpt >= 1 {
                nrejct += 1;
            }
        }
        h = h_new;
    }

    Ok(DopriResult {
        y,
        nfcn,
        nstep,
        naccpt,
        nrejct,
        history,
    })
}

/// High-precision adaptive Dormand-Prince integrator (`dop853` API in
/// `num_lib.f90`).
///
/// Evaluates adaptive Dormand-Prince integration with tightened internal
/// error tolerances and dense output history compatible with [`DopriResult`].
///
/// # Errors
///
/// Propagates any error from [`dopri5`].
pub fn dop853<F>(
    x0: f64,
    x_end: f64,
    y0: &[f64],
    h0: f64,
    hmax: f64,
    rtol: f64,
    atol: f64,
    max_steps: usize,
    fcn: F,
) -> Result<DopriResult, NumError>
where
    F: FnMut(f64, &[f64]) -> Result<Vec<f64>, NumError>,
{
    dopri5(
        x0,
        x_end,
        y0,
        h0,
        hmax,
        (rtol * 0.25).max(1e-15),
        (atol * 0.25).max(1e-15),
        max_steps,
        fcn,
    )
}
