//! Brent (1973) root-finding and univariate local/global minimization (`num`).
//!
//! Ports `num/private/mod_brent.f90` (`eval_brent_safe_zero`,
//! `eval_brent_local_min`, and `eval_brent_global_min`) into Safe Rust in `f64`
//! precision.
//!
//! # Algorithmic Overview
//!
//! - [`brent_safe_zero`] combines bisection, linear secant interpolation, and
//!   inverse quadratic interpolation so that the bracket width $[b, c]$
//!   contracts superlinearly on smooth functions while never converging slower
//!   than bisection.
//! - [`brent_local_min`] combines golden-section search with successive
//!   parabolic interpolation on $[a, b]$ without requiring first derivatives.
//! - [`brent_global_min`] partitions $[a, b]$ into sub-intervals scaled by the
//!   second-derivative bound $m$ and refines the lowest minimum via
//!   [`brent_local_min`].
//!
//! # Finiteness & Bracketing Guarantees
//!
//! - All routines validate that input bounds and evaluated function values are
//!   finite IEEE-754 `f64` numbers (`is_finite()`), returning
//!   [`NumError::NonFiniteValue`] or [`NumError::InvalidParameter`] immediately
//!   if `NaN` or `Inf` is encountered.
//! - [`brent_safe_zero`] maintains an enclosing sign-change bracket at every
//!   iteration and falls back to bisection whenever a secant or inverse
//!   quadratic step jumps outside the bracket or contracts too slowly.

use crate::NumError;

/// Finds a root $x \in [a, b]$ of $f(x) = 0$ via Brent's (1973) inverse
/// quadratic, secant, and bisection algorithm (`eval_brent_safe_zero`).
///
/// Stops when the bracket half-width satisfies $|m| \le 2 \epsilon |x| + t$ or
/// $|f(x)| \le \epsilon_y$. Optional endpoint function values `df_a` and `df_b`
/// avoid redundant evaluations when already known by the caller.
///
/// # Errors
///
/// Returns [`NumError::BracketingFailed`] if $f(a)$ and $f(b)$ have the same
/// strict sign, or [`NumError::ConvergenceFailed`] if `max_iter` is exceeded.
pub fn brent_safe_zero<F>(
    a: f64,
    b: f64,
    eps: f64,
    t: f64,
    epsy: f64,
    mut func: F,
    df_a: Option<f64>,
    df_b: Option<f64>,
    max_iter: usize,
) -> Result<(f64, f64), NumError>
where
    F: FnMut(f64) -> Result<f64, NumError>,
{
    let mut sa = a;
    let mut sb = b;
    let mut fa = match df_a {
        Some(v) => v,
        None => func(sa)?,
    };
    let mut fb = match df_b {
        Some(v) => v,
        None => func(sb)?,
    };
    if !sa.is_finite() || !sb.is_finite() || !fa.is_finite() || !fb.is_finite()
    {
        return Err(NumError::NonFiniteValue(
            "non-finite endpoint or value in brent_safe_zero",
        ));
    }
    if fa == 0.0 {
        return Ok((sa, 0.0));
    }
    if fb == 0.0 {
        return Ok((sb, 0.0));
    }
    if (fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0) {
        return Err(NumError::BracketingFailed(
            "f(a) and f(b) must have opposite signs in brent_safe_zero",
        ));
    }

    let mut c = sa;
    let mut fc = fa;
    let mut e = sb - sa;
    let mut d = e;
    let iter_limit = if max_iter == 0 { 200 } else { max_iter };

    for _iter in 0..iter_limit {
        // Ensure sb and c always bracket the zero with |f(sb)| <= |f(c)|
        if (fc > 0.0 && fb > 0.0) || (fc < 0.0 && fb < 0.0) {
            c = sa;
            fc = fa;
            e = sb - sa;
            d = e;
        }
        if fc.abs() < fb.abs() {
            sa = sb;
            sb = c;
            c = sa;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let tol = 2.0 * eps * sb.abs() + t;
        let m = 0.5 * (c - sb);
        if m.abs() <= tol || fb == 0.0 || (epsy > 0.0 && fb.abs() <= epsy) {
            return Ok((sb, fb));
        }
        // Choose bisection if previous step was too small or did not improve
        if e.abs() < tol || fa.abs() <= fb.abs() {
            e = m;
            d = e;
        } else {
            let s = fb / fa;
            // Secant step when only two distinct points are available;
            // inverse quadratic interpolation when sa, sb, c are all distinct
            let (mut p, mut q) = if sa == c {
                (2.0 * m * s, 1.0 - s)
            } else {
                let q0 = fa / fc;
                let r = fb / fc;
                (
                    s * (2.0 * m * q0 * (q0 - r) - (sb - sa) * (r - 1.0)),
                    (q0 - 1.0) * (r - 1.0) * (s - 1.0),
                )
            };
            if p > 0.0 {
                q = -q;
            } else {
                p = -p;
            }
            let s_old = e;
            e = d;
            if 2.0 * p < 3.0 * m * q - (tol * q).abs()
                && p < (0.5 * s_old * q).abs()
            {
                d = p / q;
            } else {
                e = m;
                d = e;
            }
        }
        sa = sb;
        fa = fb;
        if d.abs() > tol {
            sb += d;
        } else if m > 0.0 {
            sb += tol;
        } else {
            sb -= tol;
        }
        fb = func(sb)?;
        if !fb.is_finite() {
            return Err(NumError::NonFiniteValue(
                "non-finite function value in brent_safe_zero",
            ));
        }
    }
    Err(NumError::ConvergenceFailed(
        "brent_safe_zero exceeded max_iter",
    ))
}

/// Finds a local minimum $(x_{\min}, f(x_{\min}))$ of $f$ on $[a, b]$ via
/// Brent's (1973) golden-section and parabolic interpolation method
/// (`eval_brent_local_min`).
///
/// # Errors
///
/// Propagates any error returned by `func` or [`NumError::InvalidParameter`]
/// if $a \ge b$ or tolerances are invalid.
pub fn brent_local_min<F>(
    max_tries: usize,
    a: f64,
    b: f64,
    eps: f64,
    t: f64,
    mut func: F,
) -> Result<(f64, f64), NumError>
where
    F: FnMut(f64) -> Result<f64, NumError>,
{
    if !a.is_finite() || !b.is_finite() || a >= b || eps <= 0.0 || t < 0.0 {
        return Err(NumError::InvalidParameter(
            "require finite a < b, eps > 0, t >= 0 in brent_local_min",
        ));
    }
    // Golden ratio conjugate (3 - sqrt(5)) / 2 ~= 0.3819660112501051
    let c_gold = 0.5 * (3.0 - 5.0_f64.sqrt());
    let mut sa = a;
    let mut sb = b;
    let mut x = sa + c_gold * (sb - sa);
    let mut w = x;
    let mut v = w;
    let mut e = 0.0_f64;
    let mut d = 0.0_f64;
    let mut fx = func(x)?;
    if !fx.is_finite() {
        return Err(NumError::NonFiniteValue(
            "non-finite objective value in brent_local_min",
        ));
    }
    let mut fw = fx;
    let mut fv = fx;
    let limit = if max_tries == 0 { 200 } else { max_tries };

    for _iter in 0..limit {
        let m = f64::midpoint(sa, sb);
        let tol = eps * x.abs() + t;
        let t2 = 2.0 * tol;
        if (x - m).abs() <= t2 - 0.5 * (sb - sa) {
            return Ok((x, fx));
        }
        let mut r = 0.0;
        let mut q = 0.0;
        let mut p = 0.0;
        if e.abs() > tol {
            // Fit trial parabola through (x, fx), (w, fw), and (v, fv)
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }
            r = e;
            e = d;
        }
        if p.abs() < (0.5 * q * r).abs() && p > q * (sa - x) && p < q * (sb - x)
        {
            d = p / q;
            let u = x + d;
            if (u - sa) < t2 || (sb - u) < t2 {
                d = if x < m { tol } else { -tol };
            }
        } else {
            // Fall back to golden-section step into the larger sub-interval
            e = if x < m { sb - x } else { sa - x };
            d = c_gold * e;
        }
        let u = if d.abs() >= tol {
            x + d
        } else if d > 0.0 {
            x + tol
        } else {
            x - tol
        };
        let fu = func(u)?;
        if !fu.is_finite() {
            return Err(NumError::NonFiniteValue(
                "non-finite objective value in brent_local_min",
            ));
        }
        // Update bracket [sa, sb] and lowest-three points x, w, v
        if fu <= fx {
            if u < x {
                sb = x;
            } else {
                sa = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                sa = u;
            } else {
                sb = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }
    Ok((x, fx))
}

/// Alias for [`brent_local_min`] matching upstream `test_brent.f90` (`local_min`).
///
/// # Errors
///
/// Propagates any error from [`brent_local_min`].
pub fn local_min<F>(
    max_tries: usize,
    a: f64,
    b: f64,
    eps: f64,
    t: f64,
    func: F,
) -> Result<(f64, f64), NumError>
where
    F: FnMut(f64) -> Result<f64, NumError>,
{
    brent_local_min(max_tries, a, b, eps, t, func)
}

/// Seeks the global minimum $(x_{\min}, f(x_{\min}))$ of $f$ on $[a, b]$ given
/// an upper bound `m` on $f''(x)$ (`eval_global_min` / `glomin` in
/// `mod_brent.f90`).
///
/// Implements Brent's (1973) global minimization algorithm using parabolic
/// under-estimators with second-derivative bound $m$ to skip regions where
/// $f(x)$ cannot fall below the best value found so far.
///
/// # Arguments
///
/// - `max_tries`: Maximum number of outer interval-advance iterations.
/// - `a`, `b`: Lower and upper search bounds ($a < b$).
/// - `c`: Initial candidate abscissa in $[a, b]$ (uses midpoint if outside).
/// - `m`: Upper bound on $f''(x)$ for $x \in [a, b]$.
/// - `e`: Positive bound on the absolute evaluation error of $f(x)$.
/// - `t`: Positive absolute abscissa tolerance.
///
/// # Errors
///
/// Returns [`NumError::InvalidParameter`] if $a \ge b$ or inputs are non-finite,
/// or propagates any error from `func`.
#[allow(clippy::manual_midpoint, clippy::too_many_lines)]
pub fn brent_global_min<F>(
    max_tries: usize,
    a: f64,
    b: f64,
    c: f64,
    m: f64,
    e: f64,
    t: f64,
    mut func: F,
) -> Result<(f64, f64), NumError>
where
    F: FnMut(f64) -> Result<f64, NumError>,
{
    if !a.is_finite()
        || !b.is_finite()
        || !c.is_finite()
        || !m.is_finite()
        || !e.is_finite()
        || !t.is_finite()
        || a >= b
    {
        return Err(NumError::InvalidParameter(
            "require finite a < b and finite parameters in brent_global_min",
        ));
    }
    let machep = f64::EPSILON;
    let mut a0 = b;
    let mut x = a0;
    let mut a2 = a;
    let mut y0 = func(b)?;
    let yb = y0;
    let mut y2 = func(a)?;
    let mut y = y2;
    if !y0.is_finite() || !y2.is_finite() {
        return Err(NumError::NonFiniteValue(
            "non-finite endpoint value in brent_global_min",
        ));
    }
    if y0 < y {
        y = y0;
    } else {
        x = a;
    }
    if m <= 0.0 || b <= a {
        return Ok((x, y));
    }

    let m2 = 0.5 * (1.0 + 16.0 * machep) * m;
    let mut sc = if c <= a || b <= c {
        0.5 * (a + b)
    } else {
        c
    };
    let mut y1 = func(sc)?;
    if !y1.is_finite() {
        return Err(NumError::NonFiniteValue(
            "non-finite interior value in brent_global_min",
        ));
    }
    let mut k: i64 = 3;
    let mut d0 = a2 - sc;
    let mut h = 9.0 / 11.0;
    if y1 < y {
        x = sc;
        y = y1;
    }

    let limit = if max_tries == 0 { 100_000 } else { max_tries };
    for _iter in 1..=limit {
        let d1 = a2 - a0;
        let d2 = sc - a0;
        let z2 = b - a2;
        let z0 = y2 - y1;
        let z1 = y2 - y0;
        let mut r = d1 * d1 * z0 - d0 * d0 * z1;
        let mut p = r;
        let qs = 2.0 * (d0 * z1 - d1 * z0);
        let mut q = qs;

        if k < 1_000_000 || y2 <= y {
            loop {
                if q * (r * (yb - y2) + z2 * q * ((y2 - y) + t))
                    < z2 * m2 * r * (z2 * q - r)
                {
                    let a3 = a2 + r / q;
                    let y3 = func(a3)?;
                    if !y3.is_finite() {
                        return Err(NumError::NonFiniteValue(
                            "non-finite objective value in brent_global_min",
                        ));
                    }
                    if y3 < y {
                        x = a3;
                        y = y3;
                    }
                }
                k = (1611 * k) % 1_048_576;
                q = 1.0;
                r = (b - a) * 0.00001 * (k as f64);
                if z2 <= r {
                    break;
                }
            }
        } else {
            k = (1611 * k) % 1_048_576;
            q = 1.0;
            r = (b - a) * 0.00001 * (k as f64);
            while r < z2 {
                if q * (r * (yb - y2) + z2 * q * ((y2 - y) + t))
                    < z2 * m2 * r * (z2 * q - r)
                {
                    let a3 = a2 + r / q;
                    let y3 = func(a3)?;
                    if !y3.is_finite() {
                        return Err(NumError::NonFiniteValue(
                            "non-finite objective value in brent_global_min",
                        ));
                    }
                    if y3 < y {
                        x = a3;
                        y = y3;
                    }
                }
                k = (1611 * k) % 1_048_576;
                q = 1.0;
                r = (b - a) * 0.00001 * (k as f64);
            }
        }

        r = m2 * d0 * d1 * d2;
        let s = (((y2 - y) + t) / m2).sqrt();
        h = 0.5 * (1.0 + h);
        p = h * (p + 2.0 * r * s);
        q += 0.5 * qs;
        r = -0.5 * (d0 + (z0 + 2.01 * e) / (d0 * m2));
        r = if r < s || d0 < 0.0 { a2 + s } else { a2 + r };
        let mut a3 = if p * q > 0.0 { a2 + p / q } else { r };

        let y3 = loop {
            a3 = a3.max(r);
            let y3_val = if b <= a3 {
                a3 = b;
                yb
            } else {
                let v = func(a3)?;
                if !v.is_finite() {
                    return Err(NumError::NonFiniteValue(
                        "non-finite objective value in brent_global_min",
                    ));
                }
                v
            };
            if y3_val < y {
                x = a3;
                y = y3_val;
            }
            d0 = a3 - a2;
            if a3 <= r {
                break y3_val;
            }
            p = 2.0 * (y2 - y3_val) / (m * d0);
            if (1.0 + 9.0 * machep) * d0 <= p.abs() {
                break y3_val;
            }
            if 0.5 * m2 * (d0 * d0 + p * p)
                <= (y2 - y) + (y3_val - y) + 2.0 * t
            {
                break y3_val;
            }
            a3 = 0.5 * (a2 + a3);
            h *= 0.9;
        };

        if b <= a3 {
            break;
        }
        a0 = sc;
        sc = a2;
        a2 = a3;
        y0 = y1;
        y1 = y2;
        y2 = y3;
    }
    Ok((x, y))
}

/// Alias for [`brent_global_min`] matching Burkardt/MESA `glomin`.
///
/// # Errors
///
/// Propagates any error from [`brent_global_min`].
pub fn glomin<F>(
    max_tries: usize,
    a: f64,
    b: f64,
    c: f64,
    m: f64,
    e: f64,
    t: f64,
    func: F,
) -> Result<(f64, f64), NumError>
where
    F: FnMut(f64) -> Result<f64, NumError>,
{
    brent_global_min(max_tries, a, b, c, m, e, t, func)
}
