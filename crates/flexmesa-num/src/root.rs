//! Bracketed and derivative-assisted root-finding with implicit autodiff.
//!
//! Ports `num/private/mod_root.f90` (`do_look_for_brackets`,
//! `do_safe_root_with_brackets`, `do_safe_root_with_guess`) into Safe Rust in
//! `f64` precision and adds [`safe_root_dual`] for implicit differentiation of
//! $f(x^*(p), p) = 0$ via the implicit function theorem:
//! $$\frac{\partial x^*}{\partial p_m}
//!   = -\left(\frac{\partial f}{\partial x}\right)^{-1}
//!     \frac{\partial f}{\partial p_m}$$

use crate::NumError;
use flexmesa_autodiff::DualOrder1;

/// Bracket pair `((x1, x3), (y1, y3))` returned by [`look_for_brackets`].
pub type BracketPair = ((f64, f64), (f64, f64));

/// Searches outward from `x` in steps of $\pm 2^k \mathrm{dx}$ up to
/// `[xmin, xmax]` to find a sign-change bracket $((x_1, x_3), (y_1, y_3))$ for
/// $f(x) = 0$.
///
/// # Errors
///
/// Returns [`NumError::BracketingFailed`] if no sign change is found within
/// `[xmin, xmax]` or `ntry` steps.
pub fn look_for_brackets<F>(
    x: f64,
    dx: f64,
    xmin: f64,
    xmax: f64,
    mut func: F,
    ntry: usize,
) -> Result<BracketPair, NumError>
where
    F: FnMut(f64) -> Result<(f64, f64), NumError>,
{
    if dx == 0.0 || xmin >= xmax {
        return Err(NumError::InvalidParameter(
            "require dx != 0 and xmin < xmax in look_for_brackets",
        ));
    }
    let mut x1 = x.clamp(xmin, xmax);
    let mut x3 = x1;
    let (y0, _) = func(x1)?;
    if y0 == 0.0 {
        return Ok(((x1, x1), (0.0, 0.0)));
    }
    let mut y1 = y0;
    let mut y3 = y0;
    let mut step = dx.abs();
    let max_tries = if ntry == 0 { 50 } else { ntry };

    // Expand bracket geometrically in both directions until a sign flip occurs
    for _iter in 0..max_tries {
        let can_move_left = x1 > xmin;
        let can_move_right = x3 < xmax;
        if !can_move_left && !can_move_right {
            break;
        }
        if can_move_left {
            let x_new = (x1 - step).max(xmin);
            let (y_new, _) = func(x_new)?;
            if y_new == 0.0 || (y_new > 0.0) != (y1 > 0.0) {
                return Ok(((x_new, x1), (y_new, y1)));
            }
            x1 = x_new;
            y1 = y_new;
        }
        if can_move_right {
            let x_new = (x3 + step).min(xmax);
            let (y_new, _) = func(x_new)?;
            if y_new == 0.0 || (y_new > 0.0) != (y3 > 0.0) {
                return Ok(((x3, x_new), (y3, y_new)));
            }
            x3 = x_new;
            y3 = y_new;
        }
        step *= 2.0;
    }
    Err(NumError::BracketingFailed(
        "failed to bracket root in look_for_brackets",
    ))
}

/// Finds a root $x \in [\min(x_1, x_3), \max(x_1, x_3)]$ of $f(x) = 0$ via
/// safeguarded Newton-Raphson and bisection (`do_safe_root_with_brackets`).
///
/// The callback `func(x)` returns `Ok((f(x), df_dx(x)))`; when `df_dx == 0.0`
/// or a Newton step would exit the current bracket or contract slower than
/// bisection, a bisection step is taken automatically.
///
/// # Errors
///
/// Returns [`NumError::BracketingFailed`] if `y1` and `y3` have the same strict
/// sign, or [`NumError::ConvergenceFailed`] if `imax` is exceeded.
pub fn safe_root_with_brackets<F>(
    mut func: F,
    x1_in: f64,
    x3_in: f64,
    y1_in: f64,
    y3_in: f64,
    imax: usize,
    epsx: f64,
    epsy: f64,
) -> Result<(f64, f64), NumError>
where
    F: FnMut(f64) -> Result<(f64, f64), NumError>,
{
    if y1_in == 0.0 {
        return Ok((x1_in, 0.0));
    }
    if y3_in == 0.0 {
        return Ok((x3_in, 0.0));
    }
    if (y1_in > 0.0 && y3_in > 0.0) || (y1_in < 0.0 && y3_in < 0.0) {
        return Err(NumError::BracketingFailed(
            "y1 and y3 must have opposite signs in safe_root_with_brackets",
        ));
    }
    // Orient the search so that f(xl) < 0 and f(xh) > 0
    let (mut xl, mut xh, mut fl, mut fh) = if y1_in < 0.0 {
        (x1_in, x3_in, y1_in, y3_in)
    } else {
        (x3_in, x1_in, y3_in, y1_in)
    };
    let mut rts = f64::midpoint(x1_in, x3_in);
    let mut dxold = (x3_in - x1_in).abs();
    let mut dx = dxold;
    let (mut f, mut df) = func(rts)?;
    let max_iter = if imax == 0 { 100 } else { imax };

    for _iter in 0..max_iter {
        if f.abs() <= epsy || dx.abs() <= epsx || (xh - xl).abs() <= epsx {
            return Ok((rts, f));
        }
        // Bisect if Newton step jumps out of (xl, xh) or decreases too slowly
        let out_of_bracket =
            ((rts - xh) * df - f) * ((rts - xl) * df - f) > 0.0;
        let slow_decrease = (2.0 * f).abs() > (dxold * df).abs();
        if df == 0.0 || !df.is_finite() || out_of_bracket || slow_decrease {
            dxold = dx;
            dx = 0.5 * (xh - xl);
            rts = xl + dx;
            if xl == rts {
                return Ok((rts, f));
            }
        } else {
            dxold = dx;
            dx = f / df;
            let temp = rts;
            rts -= dx;
            if temp == rts {
                return Ok((rts, f));
            }
        }
        if dx.abs() <= epsx {
            let (f_final, _) = func(rts)?;
            return Ok((rts, f_final));
        }
        let (f_new, df_new) = func(rts)?;
        f = f_new;
        df = df_new;
        if f < 0.0 {
            xl = rts;
            fl = f;
        } else {
            xh = rts;
            fh = f;
        }
        let _ = (fl, fh);
    }
    Err(NumError::ConvergenceFailed(
        "safe_root_with_brackets exceeded imax",
    ))
}

/// Finds a root of $f(x) = 0$ starting from initial guess `x_guess` with
/// optional bracket bounds `(x1, x3, y1, y3)` (`do_safe_root_with_guess`).
///
/// Attempts up to `newt_imax` pure Newton steps inside `[xmin, xmax]`. If not
/// yet converged, uses the supplied bracket `(x1, x3)` when valid or calls
/// [`look_for_brackets`] and finishes via [`safe_root_with_brackets`].
///
/// # Errors
///
/// Propagates any error from [`look_for_brackets`] or
/// [`safe_root_with_brackets`].
pub fn safe_root_with_guess<F>(
    mut func: F,
    x_guess: f64,
    dx: f64,
    x1_opt: Option<f64>,
    x3_opt: Option<f64>,
    y1_opt: Option<f64>,
    y3_opt: Option<f64>,
    newt_imax: usize,
    imax: usize,
    xmin: f64,
    xmax: f64,
    epsx: f64,
    epsy: f64,
) -> Result<(f64, f64), NumError>
where
    F: FnMut(f64) -> Result<(f64, f64), NumError>,
{
    // Try up to newt_imax pure Newton steps from x_guess if within [xmin, xmax]
    let mut x = x_guess.clamp(xmin, xmax);
    for _ in 0..newt_imax {
        let (fx, dfx) = func(x)?;
        if fx.abs() <= epsy {
            return Ok((x, fx));
        }
        if dfx == 0.0 || !dfx.is_finite() {
            break;
        }
        let step = fx / dfx;
        let x_next = x - step;
        if x_next < xmin || x_next > xmax {
            break;
        }
        x = x_next;
        if step.abs() <= epsx {
            let (fx_end, _) = func(x)?;
            if fx_end.abs() <= epsy.max(1e-12) {
                return Ok((x, fx_end));
            }
            break;
        }
    }

    let ((bx1, bx3), (by1, by3)) =
        if let (Some(x1), Some(x3), Some(y1), Some(y3)) =
            (x1_opt, x3_opt, y1_opt, y3_opt)
        {
            if (y1 <= 0.0 && y3 >= 0.0) || (y1 >= 0.0 && y3 <= 0.0) {
                ((x1, x3), (y1, y3))
            } else {
                look_for_brackets(x_guess, dx, xmin, xmax, &mut func, imax)?
            }
        } else {
            look_for_brackets(x_guess, dx, xmin, xmax, &mut func, imax)?
        };

    safe_root_with_brackets(func, bx1, bx3, by1, by3, imax, epsx, epsy)
}

/// Convenience wrapper for [`safe_root_with_guess`] when no initial brackets
/// are known (`safe_root_without_brackets`).
///
/// # Errors
///
/// Propagates any error from [`safe_root_with_guess`].
pub fn safe_root_without_brackets<F>(
    func: F,
    x_guess: f64,
    dx: f64,
    newt_imax: usize,
    imax: usize,
    xmin: f64,
    xmax: f64,
    epsx: f64,
    epsy: f64,
) -> Result<(f64, f64), NumError>
where
    F: FnMut(f64) -> Result<(f64, f64), NumError>,
{
    safe_root_with_guess(
        func, x_guess, dx, None, None, None, None, newt_imax, imax, xmin,
        xmax, epsx, epsy,
    )
}

/// Solves for the root $x^*(p) \in [x_1, x_3]$ of $f(x, p) = 0$ and propagates
/// first-order forward-mode dual numbers [`DualOrder1<K>`] via the implicit
/// function theorem:
/// $$\frac{\partial x^*}{\partial p_m}
///   = -\left(\frac{\partial f}{\partial x}(x^*, p)\right)^{-1}
///     \frac{\partial f}{\partial p_m}(x^*, p)$$
///
/// Here:
/// - `f_primal(x)` evaluates $(f(x, p), \partial f/\partial x(x, p))$ at the
///   primal parameter vector $p$.
/// - `f_param_dual(x_star)` evaluates $f(x^*, p)$ as a [`DualOrder1<K>`] at
///   fixed $x = x^*$ with respect to the $K$ parameters $p$.
///
/// # Errors
///
/// Propagates any root-finding error or returns [`NumError::NonFiniteValue`]
/// if $\partial f/\partial x(x^*, p) = 0$.
pub fn safe_root_dual<const K: usize, FPrimal, FDual>(
    mut f_primal: FPrimal,
    mut f_param_dual: FDual,
    x1: f64,
    x3: f64,
    imax: usize,
    epsx: f64,
    epsy: f64,
) -> Result<DualOrder1<K>, NumError>
where
    FPrimal: FnMut(f64) -> Result<(f64, f64), NumError>,
    FDual: FnMut(f64) -> DualOrder1<K>,
{
    let (y1, _) = f_primal(x1)?;
    let (y3, _) = f_primal(x3)?;
    let (x_star, _) = safe_root_with_brackets(
        &mut f_primal,
        x1,
        x3,
        y1,
        y3,
        imax,
        epsx,
        epsy,
    )?;
    let (_, df_dx) = f_primal(x_star)?;
    if df_dx == 0.0 || !df_dx.is_finite() {
        return Err(NumError::NonFiniteValue(
            "zero or non-finite df/dx at root in safe_root_dual",
        ));
    }
    let f_p = f_param_dual(x_star);
    let mut d1 = [0.0_f64; K];
    for m in 0..K {
        d1[m] = -f_p.d1[m] / df_dx;
    }
    Ok(DualOrder1::new(x_star, d1))
}
