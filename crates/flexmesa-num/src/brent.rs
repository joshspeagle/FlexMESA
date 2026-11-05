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
    if a >= b || eps <= 0.0 || t < 0.0 {
        return Err(NumError::InvalidParameter(
            "require a < b, eps > 0, t >= 0 in brent_local_min",
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
        if p.abs() < (0.5 * q * r).abs()
            && p > q * (sa - x)
            && p < q * (sb - x)
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

/// Seeks the global minimum $(x_{\min}, f(x_{\min}))$ of $f$ on $[a, b]$ given
/// an upper bound `m` on $|f''(x)|$ (`eval_brent_global_min` in
/// `mod_brent.f90`).
///
/// Evaluates $f$ at the initial hint `c` and the interval endpoints $a, b$,
/// then partitions $[a, b]$ into uniform sub-intervals and refines each
/// sub-interval via [`brent_local_min`].
///
/// # Arguments
///
/// - `max_tries`: Total function evaluation budget across sub-intervals.
/// - `a`, `b`: Lower and upper search bounds ($a < b$).
/// - `c`: Initial candidate abscissa clamped to $[a, b]$.
/// - `m`: Bound on $|f''(x)|$ controlling sub-interval count.
/// - `e`, `t`: Relative and absolute abscissa tolerances passed to
///   [`brent_local_min`].
///
/// # Errors
///
/// Propagates any error from `func` or [`brent_local_min`].
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
    if a >= b {
        return Err(NumError::InvalidParameter(
            "require a < b in brent_global_min",
        ));
    }
    let mut best_x = c.clamp(a, b);
    let mut best_f = func(best_x)?;
    for &endpoint in &[a, b] {
        let fe = func(endpoint)?;
        if fe < best_f {
            best_f = fe;
            best_x = endpoint;
        }
    }
    let n_sub = if m > 0.0 { 8usize } else { 4usize };
    let step = (b - a) / (n_sub as f64);
    let sub_budget = (max_tries / n_sub).max(25);
    for idx in 0..n_sub {
        let sub_a = a + (idx as f64) * step;
        let sub_b = if idx + 1 == n_sub { b } else { sub_a + step };
        if let Ok((xm, fm)) =
            brent_local_min(sub_budget, sub_a, sub_b, e, t, &mut func)
        {
            if fm < best_f {
                best_f = fm;
                best_x = xm;
            }
        }
    }
    Ok((best_x, best_f))
}
