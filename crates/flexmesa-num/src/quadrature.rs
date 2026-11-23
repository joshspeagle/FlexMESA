//! Numerical differentiation (`dfridr`) and adaptive quadrature (`integrate`).
//!
//! Ports Ridders' Neville-tableau polynomial extrapolation derivative
//! (`dfridr` in `num/public/num_lib.f90`) and recursive adaptive Simpson 3/8
//! quadrature (`simp38` in `num/private/mod_integrate.f90`) into Safe Rust in
//! `f64` precision.
//! - [`dfridr`] adapts step sizes geometrically and stops when higher-order
//!   extapolation columns cease reducing the estimated truncation error.
//! - [`integrate`] subdivides intervals recursively on a stack until the
//!   4-point vs. 7-point Newton-Cotes 3/8 difference satisfies the tolerance.

use crate::NumError;

/// Estimates $f'(0)$ and its truncation error via Ridders' (1982) Neville
/// polynomial extrapolation (`dfridr` in `num_lib.f90`).
///
/// Evaluates symmetric central differences $(f(h_i) - f(-h_i)) / (2 h_i)$ for
/// successively geometrically shrinking step sizes $h_i = h_x / 1.4^i$ ($i = 0,
/// \dots, 9$) and extrapolates to $h \to 0$.
///
/// # Errors
///
/// Returns [`NumError::InvalidParameter`] if `hx == 0.0` or `hx` is non-finite.
pub fn dfridr<F>(hx: f64, mut func: F) -> Result<(f64, f64), NumError>
where
    F: FnMut(f64) -> f64,
{
    const NTAB: usize = 10;
    const CON: f64 = 1.4;
    const CON2: f64 = CON * CON;
    const SAFE: f64 = 2.0;

    if hx == 0.0 || !hx.is_finite() {
        return Err(NumError::InvalidParameter(
            "step hx must be finite and non-zero in dfridr",
        ));
    }

    let mut a = [[0.0_f64; NTAB]; NTAB];
    let mut hh = hx;
    a[0][0] = (func(hh) - func(-hh)) / (2.0 * hh);
    let mut err = f64::MAX;
    let mut ans = a[0][0];

    // Build Neville tableau column by column and track minimum error estimate
    for i in 1..NTAB {
        hh /= CON;
        a[0][i] = (func(hh) - func(-hh)) / (2.0 * hh);
        let mut fac = CON2;
        for j in 1..=i {
            a[j][i] = (a[j - 1][i] * fac - a[j - 1][i - 1]) / (fac - 1.0);
            fac *= CON2;
            let errt = (a[j][i] - a[j - 1][i])
                .abs()
                .max((a[j][i] - a[j - 1][i - 1]).abs());
            if errt <= err {
                err = errt;
                ans = a[j][i];
            }
        }
        // Stop early when higher-order diagonal begins amplifying roundoff
        if (a[i][i] - a[i - 1][i - 1]).abs() >= SAFE * err {
            break;
        }
    }
    Ok((ans, err))
}

/// Evaluates $\int_{\min_x}^{\max_x} f(x)\,\mathrm{d}x$ via adaptive Simpson's
/// 3/8 rule matching upstream `integrator` / `simp38`
/// (`num/private/mod_integrate.f90`).
///
/// Recursively bisects each interval $[a, b]$ at $(a + b)/2$ until
/// $|S_1 - S_2| < \epsilon$ or $|S_1 - S_2| / S_1 < \epsilon$ (with `atol =
/// rtol = eps`), returning the two-panel Simpson 3/8 sum $S_2$.
///
/// # Errors
///
/// Returns [`NumError::InvalidParameter`] on invalid tolerances or bounds
/// (`min_x > max_x`), or [`NumError::ConvergenceFailed`] if `max_steps` is
/// exceeded.
pub fn integrate<F>(
    mut func: F,
    min_x: f64,
    max_x: f64,
    eps: f64,
    max_steps: usize,
) -> Result<(f64, usize), NumError>
where
    F: FnMut(f64) -> f64,
{
    if !min_x.is_finite() || !max_x.is_finite() || min_x > max_x || eps <= 0.0 {
        return Err(NumError::InvalidParameter(
            "require finite min_x <= max_x and eps > 0 in integrate",
        ));
    }
    if min_x == max_x {
        return Ok((0.0, 0));
    }
    let mut ncalls = 0usize;
    let val = integrator_rec(
        &mut func,
        min_x,
        max_x,
        eps,
        eps,
        0,
        &mut ncalls,
        max_steps,
    )?;
    Ok((val, ncalls))
}

/// Evaluates the 4-point Simpson 3/8 rule on `[minx, maxx]` matching upstream
/// `simp38` (`num/private/mod_integrate.f90:82-111`).
fn simp38<F: FnMut(f64) -> f64>(
    f: &mut F,
    minx: f64,
    maxx: f64,
    ncalls: &mut usize,
) -> f64 {
    let mut s = f(minx);
    let x1 = (2.0 * minx + maxx) / 3.0;
    s += 3.0 * f(x1);
    let x2 = (minx + 2.0 * maxx) / 3.0;
    s += 3.0 * f(x2);
    s += f(maxx);
    *ncalls += 4;
    ((maxx - minx) / 8.0) * s
}

/// Recursive midpoint bisection driver matching upstream `integrator`
/// (`num/private/mod_integrate.f90:30-79`).
#[allow(clippy::manual_midpoint, clippy::too_many_arguments)]
fn integrator_rec<F: FnMut(f64) -> f64>(
    f: &mut F,
    xlow: f64,
    xhigh: f64,
    atol: f64,
    rtol: f64,
    depth: usize,
    ncalls: &mut usize,
    max_steps: usize,
) -> Result<f64, NumError> {
    if depth >= 55 || (max_steps > 0 && *ncalls > max_steps) {
        return Err(NumError::ConvergenceFailed(
            "max_steps exceeded in integrate",
        ));
    }
    let xmid = (xhigh + xlow) / 2.0;
    let val1 = simp38(f, xlow, xhigh, ncalls);
    let val2 = simp38(f, xlow, xmid, ncalls) + simp38(f, xmid, xhigh, ncalls);
    if val1 == 0.0 || val2 == 0.0 {
        return Ok(val2);
    }
    let diff = (val1 - val2).abs();
    if diff < atol || diff / val1 < rtol {
        Ok(val2)
    } else {
        let left = integrator_rec(
            f,
            xlow,
            xmid,
            atol,
            rtol,
            depth + 1,
            ncalls,
            max_steps,
        )?;
        let right = integrator_rec(
            f,
            xmid,
            xhigh,
            atol,
            rtol,
            depth + 1,
            ncalls,
            max_steps,
        )?;
        Ok(left + right)
    }
}
