//! Bracketed and derivative-assisted root-finding with implicit autodiff.
//!
//! Ports `num/private/mod_root.f90` (`do_look_for_brackets`,
//! `do_safe_root_with_brackets`, `do_safe_root_with_guess`) into Safe Rust in
//! `f64` precision and adds [`safe_root_dual`] for implicit differentiation of
//! $f(x^*(p), p) = 0$ via the implicit function theorem:
//! $$\frac{\partial x^*}{\partial p_m}
//!   = -\left(\frac{\partial f}{\partial x}\right)^{-1}
//!     \frac{\partial f}{\partial p_m}$$
//!
//! # Safeguards & Error Handling
//!
//! - [`look_for_brackets`] expands outward geometrically from `x` inside
//!   `[xmin, xmax]` while verifying finite function evaluations at every step.
//! - [`safe_root_with_brackets`] delegates to [`brent_safe_zero`] so the
//!   iterate never leaves the sign-change bracket (`mod_root.f90:27-45`).
//! - [`safe_root_with_guess`] tries guarded Newton steps before bracketing.

use crate::brent::brent_safe_zero;
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
    // Validate finiteness and domain ordering before clamping x to [xmin, xmax]
    if !(x.is_finite()
        && dx.is_finite()
        && xmin.is_finite()
        && xmax.is_finite())
        || dx == 0.0
        || xmin >= xmax
    {
        return Err(NumError::InvalidParameter(
            "require finite arguments, dx != 0, and xmin < xmax",
        ));
    }
    let mut x1 = x.clamp(xmin, xmax);
    let mut x3 = x1;
    let mut y1 = 0.0_f64;
    let mut y3 = 0.0_f64;
    let step0 = dx.abs();
    let max_tries = if ntry == 0 { 50 } else { ntry };

    // Matches `do_look_for_brackets` in `num/private/mod_root.f90:197-252`,
    // clamped to `[xmin, xmax]`.
    for i in 1..=max_tries {
        if x1 <= xmin && x3 >= xmax && i > 1 {
            break;
        }
        let jump = ((1_u64 << i.min(50)) as f64) * step0;
        let (move_x1, move_x3) = if i == 1 {
            x1 = (x1 - jump).max(xmin);
            x3 = (x3 + jump).min(xmax);
            (true, true)
        } else {
            let prefer_left = if y1 > 0.0 { y1 < y3 } else { y1 > y3 };
            if (prefer_left && x1 > xmin) || x3 >= xmax {
                x1 = (x1 - jump).max(xmin);
                (true, false)
            } else {
                x3 = (x3 + jump).min(xmax);
                (false, true)
            }
        };

        if move_x1 {
            let (f1, _) = func(x1)?;
            if !f1.is_finite() {
                return Err(NumError::NonFiniteValue(
                    "non-finite f(x) in look_for_brackets",
                ));
            }
            y1 = f1;
        }
        if move_x3 {
            let (f3, _) = func(x3)?;
            if !f3.is_finite() {
                return Err(NumError::NonFiniteValue(
                    "non-finite f(x) in look_for_brackets",
                ));
            }
            y3 = f3;
        }
        if y1 * y3 <= 0.0 {
            return Ok(((x1, x3), (y1, y3)));
        }
    }
    Err(NumError::BracketingFailed(
        "failed to bracket root in look_for_brackets",
    ))
}

/// Finds a root $x \in [\min(x_1, x_3), \max(x_1, x_3)]$ of $f(x) = 0$ by
/// delegating to [`brent_safe_zero`] with relative tolerance `1e-14`, matching
/// `do_safe_root_with_brackets` in `num/private/mod_root.f90:27-45`.
///
/// # Errors
///
/// Returns [`NumError::BracketingFailed`] if `y1_in` and `y3_in` have the same
/// strict sign, or [`NumError::ConvergenceFailed`] if iteration fails.
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
    let max_iter = imax.max(200);
    brent_safe_zero(
        x1_in,
        x3_in,
        1.0e-14,
        epsx,
        epsy,
        |x| Ok(func(x)?.0),
        Some(y1_in),
        Some(y3_in),
        max_iter,
    )
}

/// Finds a root of $f(x) = 0$ starting from initial guess `x_guess` with
/// optional bracket bounds `(x1, x3, y1, y3)` (`do_safe_root_with_guess` in
/// `num/private/mod_root.f90:47-151`).
///
/// Attempts up to `newt_imax` guarded Newton steps inside `[xmin, xmax]`,
/// recording sign-change brackets encountered along the trajectory. If not yet
/// converged, uses any bracket accumulated (or supplied) or calls
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
    if xmin >= xmax
        || !(x_guess.is_finite() && xmin.is_finite() && xmax.is_finite())
    {
        return Err(NumError::InvalidParameter(
            "require finite x_guess and xmin < xmax in safe_root_with_guess",
        ));
    }
    let mut x1 = x1_opt;
    let mut x3 = x3_opt;
    let mut y1 = y1_opt;
    let mut y3 = y3_opt;
    let mut x = x_guess.clamp(xmin, xmax);
    let mut absy_prev = 0.0_f64;

    for i in 0..newt_imax {
        let Ok((y, dydx)) = func(x) else {
            break;
        };
        if !y.is_finite() {
            break;
        }
        let absy = y.abs();
        if absy == 0.0 || absy < epsy.max(dydx.abs() * epsx) {
            return Ok((x, y));
        }
        if dydx == 0.0 || !dydx.is_finite() {
            break;
        }
        if i > 0 && absy > absy_prev {
            break;
        }
        absy_prev = absy;
        if y < 0.0 {
            x1 = Some(x);
            y1 = Some(y);
        } else {
            x3 = Some(x);
            y3 = Some(y);
        }
        let x_next = x - y / dydx;
        if x_next < xmin || x_next > xmax {
            break;
        }
        x = x_next;
    }

    let ((bx1, bx3), (by1, by3)) = if let (Some(xv1), Some(xv3)) = (x1, x3) {
        let yv1 = match y1 {
            Some(v) => v,
            None => func(xv1)?.0,
        };
        let yv3 = match y3 {
            Some(v) => v,
            None => func(xv3)?.0,
        };
        if (yv1 <= 0.0 && yv3 >= 0.0) || (yv1 >= 0.0 && yv3 <= 0.0) {
            ((xv1, xv3), (yv1, yv3))
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
        func, x_guess, dx, None, None, None, None, newt_imax, imax, xmin, xmax,
        epsx, epsy,
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
