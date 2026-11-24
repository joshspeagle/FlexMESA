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
//!
//! # Step-Size Control & Robustness
//!
//! - Step sizes are adapted via Lund's proportional-integral (PI) controller
//!   with safety factor `0.9` and per-step ratio clamping to `[0.1, 5.0]`.
//! - Every stage callback is checked for state-vector length consistency and
//!   finite local error norms, trial-stage callback errors shrink the step size
//!   by $1/\mathrm{facc}_1$, and step-size underflow (`0.1 * |h| <= |x| * uround`)
//!   is detected and reported via [`NumError::ConvergenceFailed`].

use crate::rk_tableau::dop853 as d8;
use crate::rk_tableau::{A21, A3, A4, A5, A6, A7, C, D, E};
use crate::NumError;

pub use crate::rk_tableau::DopriResult;

/// Evaluates `fcn(t, y)` and verifies that the returned derivative vector has
/// length `n`.
fn eval_stage<F>(
    n: usize,
    t: f64,
    y: &[f64],
    fcn: &mut F,
) -> Result<Vec<f64>, NumError>
where
    F: FnMut(f64, &[f64]) -> Result<Vec<f64>, NumError>,
{
    let k = fcn(t, y)?;
    if k.len() != n {
        return Err(NumError::InvalidParameter(
            "derivative vector length mismatch in rk",
        ));
    }
    Ok(k)
}

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
    let k1 = eval_stage(n, t, y, &mut derivs)?;
    let y_mid: Vec<f64> = (0..n).map(|i| y[i] + 0.5 * h * k1[i]).collect();
    let k2 = eval_stage(n, t + 0.5 * h, &y_mid, &mut derivs)?;
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
    let k1 = eval_stage(n, t, y, &mut derivs)?;
    let y2: Vec<f64> = (0..n).map(|i| y[i] + 0.5 * h * k1[i]).collect();
    let k2 = eval_stage(n, t + 0.5 * h, &y2, &mut derivs)?;
    let y3: Vec<f64> = (0..n).map(|i| y[i] + 0.5 * h * k2[i]).collect();
    let k3 = eval_stage(n, t + 0.5 * h, &y3, &mut derivs)?;
    let y4: Vec<f64> = (0..n).map(|i| y[i] + h * k3[i]).collect();
    let k4 = eval_stage(n, t + h, &y4, &mut derivs)?;
    Ok((0..n)
        .map(|i| y[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
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
    const UROUND: f64 = 2.3e-16;
    const SAFE: f64 = 0.9;
    const FAC1: f64 = 0.2;
    const FAC2: f64 = 10.0;

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
    let mut k1 = eval_stage(n, x, &y, &mut fcn)?;
    let mut nfcn = 1usize;
    let mut h = if h0 == 0.0 {
        nfcn += 1;
        hinit_step(n, x, &y, posneg, &k1, hmax, atol, rtol, 5, &mut fcn)?
    } else {
        h0.abs().min(hmax) * posneg
    };

    let facc1 = 1.0 / FAC1;
    let facc2 = 1.0 / FAC2;

    let nmax = if max_steps == 0 { 100_000 } else { max_steps };
    let mut nstep = 0usize;
    let mut naccpt = 0usize;
    let mut nrejct = 0usize;
    let mut facold = 1e-4_f64;
    let mut reject = false;
    let mut last = false;
    let mut history = Vec::new();
    let mut y_stg = vec![0.0; n];
    let mut y_new = vec![0.0; n];

    macro_rules! eval_trial_stage {
        ($t:expr, $ys:expr) => {
            match eval_stage(n, $t, $ys, &mut fcn) {
                Ok(k) => k,
                Err(e @ NumError::InvalidParameter(_)) => return Err(e),
                Err(_) => {
                    h /= facc1;
                    continue;
                }
            }
        };
    }

    loop {
        if nstep > nmax {
            return Err(NumError::ConvergenceFailed(
                "dopri5 exceeded max_steps",
            ));
        }
        if 0.1 * h.abs() <= x.abs() * UROUND {
            return Err(NumError::ConvergenceFailed(
                "step size underflow in dopri5",
            ));
        }
        if (x + 1.01 * h - x_end) * posneg > 0.0 {
            h = x_end - x;
            last = true;
        }
        nstep += 1;

        // Evaluate stages k2..k7 using First-Same-As-Last (FSAL) stage k1,
        // halving step size by facc1 (= 5.0) on callback error matching
        // num/private/mod_dopri5.f:309-339.
        for i in 0..n {
            y_stg[i] = y[i] + h * A21 * k1[i];
        }
        let k2 = eval_trial_stage!(x + C[0] * h, &y_stg);
        for i in 0..n {
            y_stg[i] = y[i] + h * (A3[0] * k1[i] + A3[1] * k2[i]);
        }
        let k3 = eval_trial_stage!(x + C[1] * h, &y_stg);
        for i in 0..n {
            y_stg[i] =
                y[i] + h * (A4[0] * k1[i] + A4[1] * k2[i] + A4[2] * k3[i]);
        }
        let k4 = eval_trial_stage!(x + C[2] * h, &y_stg);
        for i in 0..n {
            y_stg[i] = y[i]
                + h * (A5[0] * k1[i]
                    + A5[1] * k2[i]
                    + A5[2] * k3[i]
                    + A5[3] * k4[i]);
        }
        let k5 = eval_trial_stage!(x + C[3] * h, &y_stg);
        for i in 0..n {
            y_stg[i] = y[i]
                + h * (A6[0] * k1[i]
                    + A6[1] * k2[i]
                    + A6[2] * k3[i]
                    + A6[3] * k4[i]
                    + A6[4] * k5[i]);
        }
        let xph = x + h;
        let k6 = eval_trial_stage!(xph, &y_stg);
        for i in 0..n {
            y_new[i] = y[i]
                + h * (A7[0] * k1[i]
                    + A7[2] * k3[i]
                    + A7[3] * k4[i]
                    + A7[4] * k5[i]
                    + A7[5] * k6[i]);
        }
        let k7 = eval_trial_stage!(xph, &y_new);
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
        if !err.is_finite() {
            return Err(NumError::NonFiniteValue(
                "non-finite local error in dopri5",
            ));
        }
        // Lund PI step-size controller (beta = 0.04, expo1 = 0.2 - 0.75 * beta)
        let fac11 = flexmesa_math::pow(err, 0.17);
        let fac = fac11 / flexmesa_math::pow(facold, 0.04);
        let fac_clamped = (fac / SAFE).clamp(facc2, facc1);
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
            if last {
                break;
            }
            if h_new.abs() > hmax {
                h_new = posneg * hmax;
            }
            if reject {
                h_new = posneg * h_new.abs().min(h.abs());
            }
            reject = false;
        } else {
            h_new = h / (fac11 / SAFE).min(facc1);
            reject = true;
            if naccpt >= 1 {
                nrejct += 1;
            }
            last = false;
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

/// Estimates initial step size `h` for [`dopri5`] (`iord = 5`,
/// `num/private/mod_dopri5.f:464-542`) and [`dop853`] (`iord = 8`,
/// `num/private/mod_dop853.f:708-791`).
#[allow(clippy::cast_precision_loss, clippy::too_many_arguments)]
fn hinit_step<F>(
    n: usize,
    x: f64,
    y: &[f64],
    posneg: f64,
    f0: &[f64],
    hmax: f64,
    atol: f64,
    rtol: f64,
    iord: u32,
    fcn: &mut F,
) -> Result<f64, NumError>
where
    F: FnMut(f64, &[f64]) -> Result<Vec<f64>, NumError>,
{
    let mut dnf = 0.0_f64;
    let mut dny = 0.0_f64;
    for i in 0..n {
        let sk = atol + rtol * y[i].abs();
        let rf = f0[i] / sk;
        let ry = y[i] / sk;
        dnf += rf * rf;
        dny += ry * ry;
    }
    let h0 = if dnf <= 1.0e-10 || dny <= 1.0e-10 {
        1.0e-6
    } else {
        (dny / dnf).sqrt() * 0.01
    };
    let h = h0.min(hmax) * posneg;
    let y1: Vec<f64> = (0..n).map(|i| y[i] + h * f0[i]).collect();
    let f1 = eval_stage(n, x + h, &y1, fcn)?;
    let mut der2 = 0.0_f64;
    for i in 0..n {
        let sk = atol + rtol * y[i].abs();
        let d = (f1[i] - f0[i]) / sk;
        der2 += d * d;
    }
    der2 = der2.sqrt() / h;
    let der12 = der2.abs().max(dnf.sqrt());
    let h1 = if der12 <= 1.0e-15 {
        1.0e-6_f64.max(h.abs() * 1.0e-3)
    } else {
        flexmesa_math::pow(0.01 / der12, 1.0 / f64::from(iord))
    };
    Ok((100.0 * h.abs()).min(h1).min(hmax) * posneg)
}

/// Integrates the ODE system $y'(x) = f(x, y)$ from `x0` to `x_end` using
/// Hairer & Wanner's 12-stage Dormand-Prince 8(5,3) method (`mod_dop853.f`).
///
/// Matches MESA `num/private/mod_dop853.f` step acceptance, embedded 8th/5th/3rd
/// order error norm with Lund stabilization (`beta = 0.0`, `safe = 0.9`,
/// `fac1 = 0.333`, `fac2 = 6.0`), and records 4th-order Hermite dense output
/// history on accepted steps.
///
/// # Errors
///
/// Returns [`NumError::InvalidParameter`] on invalid tolerances, or
/// [`NumError::ConvergenceFailed`] if `max_steps` is exceeded or step size
/// underflows.
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
pub fn dop853<F>(
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
    const UROUND: f64 = 2.3e-16;
    const SAFE: f64 = 0.9;
    const FAC1: f64 = 0.333;
    const FAC2: f64 = 6.0;

    let n = y0.len();
    if n == 0 || rtol <= 0.0 || atol <= 0.0 {
        return Err(NumError::InvalidParameter(
            "require y0.len() >= 1 and rtol, atol > 0 in dop853",
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

    let facc1 = 1.0 / FAC1;
    let facc2 = 1.0 / FAC2;
    let expo1 = 1.0 / 8.0;
    let posneg = (x_end - x0).signum();
    let hmax = if hmax_in <= 0.0 {
        (x_end - x0).abs()
    } else {
        hmax_in.abs()
    };
    let mut x = x0;
    let mut y = y0.to_vec();
    let mut k1 = eval_stage(n, x, &y, &mut fcn)?;
    let mut nfcn = 1usize;
    let mut h = if h0 == 0.0 {
        nfcn += 1;
        hinit_step(n, x, &y, posneg, &k1, hmax, atol, rtol, 8, &mut fcn)?
    } else {
        h0.abs() * posneg
    };

    let nmax = if max_steps == 0 { 100_000 } else { max_steps };
    let mut nstep = 0usize;
    let mut naccpt = 0usize;
    let mut nrejct = 0usize;
    let mut reject = false;
    let mut last = false;
    let mut history = Vec::new();
    let mut y1 = vec![0.0_f64; n];
    let mut k4 = vec![0.0_f64; n];
    let mut k5 = vec![0.0_f64; n];

    macro_rules! eval_trial_stage8 {
        ($t:expr, $ys:expr) => {
            match eval_stage(n, $t, $ys, &mut fcn) {
                Ok(k) => k,
                Err(e @ NumError::InvalidParameter(_)) => return Err(e),
                Err(_) => {
                    h /= facc1;
                    continue;
                }
            }
        };
    }

    loop {
        if nstep > nmax {
            return Err(NumError::ConvergenceFailed(
                "dop853 exceeded max_steps",
            ));
        }
        if 0.1 * h.abs() <= x.abs() * UROUND {
            return Err(NumError::ConvergenceFailed(
                "step size underflow in dop853",
            ));
        }
        if (x + 1.01 * h - x_end) * posneg > 0.0 {
            h = x_end - x;
            last = true;
        }
        nstep += 1;

        for i in 0..n {
            y1[i] = y[i] + h * d8::A21 * k1[i];
        }
        let k2 = eval_trial_stage8!(x + d8::C2 * h, &y1);
        for i in 0..n {
            y1[i] = y[i] + h * (d8::A31 * k1[i] + d8::A32 * k2[i]);
        }
        let k3 = eval_trial_stage8!(x + d8::C3 * h, &y1);
        for i in 0..n {
            y1[i] = y[i] + h * (d8::A41 * k1[i] + d8::A43 * k3[i]);
        }
        let stg4 = eval_trial_stage8!(x + d8::C4 * h, &y1);
        for i in 0..n {
            y1[i] = y[i]
                + h * (d8::A51 * k1[i] + d8::A53 * k3[i] + d8::A54 * stg4[i]);
        }
        let stg5 = eval_trial_stage8!(x + d8::C5 * h, &y1);
        for i in 0..n {
            y1[i] = y[i]
                + h * (d8::A61 * k1[i] + d8::A64 * stg4[i] + d8::A65 * stg5[i]);
        }
        let k6 = eval_trial_stage8!(x + d8::C6 * h, &y1);
        for i in 0..n {
            y1[i] = y[i]
                + h * (d8::A71 * k1[i]
                    + d8::A74 * stg4[i]
                    + d8::A75 * stg5[i]
                    + d8::A76 * k6[i]);
        }
        let k7 = eval_trial_stage8!(x + d8::C7 * h, &y1);
        for i in 0..n {
            y1[i] = y[i]
                + h * (d8::A81 * k1[i]
                    + d8::A84 * stg4[i]
                    + d8::A85 * stg5[i]
                    + d8::A86 * k6[i]
                    + d8::A87 * k7[i]);
        }
        let k8 = eval_trial_stage8!(x + d8::C8 * h, &y1);
        for i in 0..n {
            y1[i] = y[i]
                + h * (d8::A91 * k1[i]
                    + d8::A94 * stg4[i]
                    + d8::A95 * stg5[i]
                    + d8::A96 * k6[i]
                    + d8::A97 * k7[i]
                    + d8::A98 * k8[i]);
        }
        let k9 = eval_trial_stage8!(x + d8::C9 * h, &y1);
        for i in 0..n {
            y1[i] = y[i]
                + h * (d8::A101 * k1[i]
                    + d8::A104 * stg4[i]
                    + d8::A105 * stg5[i]
                    + d8::A106 * k6[i]
                    + d8::A107 * k7[i]
                    + d8::A108 * k8[i]
                    + d8::A109 * k9[i]);
        }
        let k10 = eval_trial_stage8!(x + d8::C10 * h, &y1);
        for i in 0..n {
            y1[i] = y[i]
                + h * (d8::A111 * k1[i]
                    + d8::A114 * stg4[i]
                    + d8::A115 * stg5[i]
                    + d8::A116 * k6[i]
                    + d8::A117 * k7[i]
                    + d8::A118 * k8[i]
                    + d8::A119 * k9[i]
                    + d8::A1110 * k10[i]);
        }
        let k11 = eval_trial_stage8!(x + d8::C11 * h, &y1);
        let xph = x + h;
        for i in 0..n {
            y1[i] = y[i]
                + h * (d8::A121 * k1[i]
                    + d8::A124 * stg4[i]
                    + d8::A125 * stg5[i]
                    + d8::A126 * k6[i]
                    + d8::A127 * k7[i]
                    + d8::A128 * k8[i]
                    + d8::A129 * k9[i]
                    + d8::A1210 * k10[i]
                    + d8::A1211 * k11[i]);
        }
        let k12 = eval_trial_stage8!(xph, &y1);
        nfcn += 11;

        for i in 0..n {
            k4[i] = d8::B1 * k1[i]
                + d8::B6 * k6[i]
                + d8::B7 * k7[i]
                + d8::B8 * k8[i]
                + d8::B9 * k9[i]
                + d8::B10 * k10[i]
                + d8::B11 * k11[i]
                + d8::B12 * k12[i];
            k5[i] = y[i] + h * k4[i];
        }

        let mut err = 0.0_f64;
        let mut err2 = 0.0_f64;
        for i in 0..n {
            let sk = atol + rtol * y[i].abs().max(k5[i].abs());
            let erri2 =
                k4[i] - d8::BHH1 * k1[i] - d8::BHH2 * k9[i] - d8::BHH3 * k12[i];
            let r2 = erri2 / sk;
            err2 += r2 * r2;
            let erri = d8::ER1 * k1[i]
                + d8::ER6 * k6[i]
                + d8::ER7 * k7[i]
                + d8::ER8 * k8[i]
                + d8::ER9 * k9[i]
                + d8::ER10 * k10[i]
                + d8::ER11 * k11[i]
                + d8::ER12 * k12[i];
            let r = erri / sk;
            err += r * r;
        }
        let mut deno = err + 0.01 * err2;
        if deno <= 0.0 {
            deno = 1.0;
        }
        let err = h.abs() * err * (1.0 / ((n as f64) * deno)).sqrt();
        if !err.is_finite() {
            return Err(NumError::NonFiniteValue(
                "non-finite local error in dop853",
            ));
        }

        let fac11 = flexmesa_math::pow(err, expo1);
        let fac = facc2.max(facc1.min(fac11 / SAFE));
        let mut h_new = h / fac;

        if err <= 1.0 {
            naccpt += 1;
            let k_next = eval_trial_stage8!(xph, &k5);
            nfcn += 1;
            let mut rc = [
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n],
            ];
            for i in 0..n {
                let ydiff = k5[i] - y[i];
                let bspl = h * k1[i] - ydiff;
                rc[0][i] = y[i];
                rc[1][i] = ydiff;
                rc[2][i] = bspl;
                rc[3][i] = ydiff - h * k_next[i] - bspl;
            }
            history.push((x, h, rc));
            k1 = k_next;
            y.copy_from_slice(&k5);
            x = xph;
            if last {
                break;
            }
            if h_new.abs() > hmax {
                h_new = posneg * hmax;
            }
            if reject {
                h_new = posneg * h_new.abs().min(h.abs());
            }
            reject = false;
        } else {
            h_new = h / facc1.min(fac11 / SAFE);
            reject = true;
            if naccpt >= 1 {
                nrejct += 1;
            }
            last = false;
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
