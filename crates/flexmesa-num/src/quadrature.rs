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
/// 3/8 rule matching upstream `simp38` (`num/private/mod_integrate.f90`).
///
/// Subdivides each interval $[a, b]$ into three equal sub-intervals of width
/// $h / 3$ and applies Richardson extrapolation $(S_{\mathrm{fine}} -
/// S_{\mathrm{coarse}}) / 15$ once $|S_{\mathrm{fine}} - S_{\mathrm{coarse}}|
/// \le 15\,\epsilon$.
///
/// # Errors
///
/// Returns [`NumError::InvalidParameter`] on invalid tolerances or
/// [`NumError::ConvergenceFailed`] if `max_steps` is exceeded.
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
    if !min_x.is_finite() || !max_x.is_finite() || eps <= 0.0 {
        return Err(NumError::InvalidParameter(
            "bounds must be finite and eps > 0 in integrate",
        ));
    }
    if min_x == max_x {
        return Ok((0.0, 0));
    }
    let mut ncalls = 2usize;
    let fa = func(min_x);
    let fb = func(max_x);
    let h = max_x - min_x;
    let x1 = min_x + h / 3.0;
    let x2 = min_x + 2.0 * h / 3.0;
    let f1 = func(x1);
    let f2 = func(x2);
    ncalls += 2;
    let s_whole = (h / 8.0) * (fa + 3.0 * f1 + 3.0 * f2 + fb);

    let val = simp_rec(
        &mut func,
        min_x,
        max_x,
        fa,
        f1,
        f2,
        fb,
        s_whole,
        eps,
        0,
        &mut ncalls,
        max_steps,
    )?;
    Ok((val, ncalls))
}

/// Recursive trisection step for [`integrate`] reusing parent nodal
/// evaluations at $a, m_1, m_2, b$ and evaluating six new interior abscissas.
///
/// Applies Richardson extrapolation $(S_{\mathrm{new}} - S_{\mathrm{prev}})/15$
/// once the fine-coarse difference falls within $15\,\mathrm{tol}$.
fn simp_rec<F: FnMut(f64) -> f64>(
    f: &mut F,
    a: f64,
    b: f64,
    fa: f64,
    f1: f64,
    f2: f64,
    fb: f64,
    s_prev: f64,
    tol: f64,
    depth: usize,
    ncalls: &mut usize,
    max_steps: usize,
) -> Result<f64, NumError> {
    let h = b - a;
    let m1 = a + h / 3.0;
    let m2 = a + 2.0 * h / 3.0;
    let a1 = a + h / 9.0;
    let a2 = a + 2.0 * h / 9.0;
    let b1 = m1 + h / 9.0;
    let b2 = m1 + 2.0 * h / 9.0;
    let c1 = m2 + h / 9.0;
    let c2 = m2 + 2.0 * h / 9.0;
    let fa1 = f(a1);
    let fa2 = f(a2);
    let fb1 = f(b1);
    let fb2 = f(b2);
    let fc1 = f(c1);
    let fc2 = f(c2);
    *ncalls += 6;
    let s1 = ((m1 - a) / 8.0) * (fa + 3.0 * fa1 + 3.0 * fa2 + f1);
    let s2 = ((m2 - m1) / 8.0) * (f1 + 3.0 * fb1 + 3.0 * fb2 + f2);
    let s3 = ((b - m2) / 8.0) * (f2 + 3.0 * fc1 + 3.0 * fc2 + fb);
    let s_new = s1 + s2 + s3;
    if (depth >= 2 && (s_new - s_prev).abs() <= tol) || depth >= 28 {
        return Ok(s_new + (s_new - s_prev) / 15.0);
    }
    if max_steps > 0 && *ncalls > max_steps {
        return Err(NumError::ConvergenceFailed(
            "max_steps exceeded in integrate",
        ));
    }
    let t3 = tol / 3.0;
    let r1 = simp_rec(
        f,
        a,
        m1,
        fa,
        fa1,
        fa2,
        f1,
        s1,
        t3,
        depth + 1,
        ncalls,
        max_steps,
    )?;
    let r2 = simp_rec(
        f,
        m1,
        m2,
        f1,
        fb1,
        fb2,
        f2,
        s2,
        t3,
        depth + 1,
        ncalls,
        max_steps,
    )?;
    let r3 = simp_rec(
        f,
        m2,
        b,
        f2,
        fc1,
        fc2,
        fb,
        s3,
        t3,
        depth + 1,
        ncalls,
        max_steps,
    )?;
    Ok(r1 + r2 + r3)
}
