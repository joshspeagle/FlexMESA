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
//! - [`look_for_brackets`] is `do_look_for_brackets` as upstream writes it:
//!   no bounds and no finiteness tests, with an explicit error where
//!   upstream's integer `2**i` overflows (#669).
//! - [`safe_root_with_brackets`] delegates to
//!   [`crate::brent::eval_brent_safe_zero`], which has no iteration cap
//!   (`mod_root.f90:27-45`, #667).
//! - [`safe_root_with_guess`] tries guarded Newton steps before bracketing.

use crate::brent::{brent_safe_zero, eval_brent_safe_zero};
use crate::NumError;
use flexmesa_autodiff::DualOrder1;

/// Bracket pair `((x1, x3), (y1, y3))` returned by [`look_for_brackets`].
pub type BracketPair = ((f64, f64), (f64, f64));

/// `do_look_for_brackets` as upstream writes it
/// (`num/private/mod_root.f90:168-256`): starting from `x1 = x3 = x`, widen
/// by `jump = 2**i*dx` for `i = 1..imax`, moving both ends at `i = 1` and
/// then the end whose value is nearer zero, until `y1*y3 <= 0`. No bounds,
/// no finiteness test (a NaN value just fails `y1*y3 <= 0`), and `dx` is
/// used with its sign.
///
/// # Errors
///
/// Propagates an error from `func` (upstream returns on `ierr /= 0`), and
/// returns [`NumError::BracketingFailed`] after `imax` tries
/// (`ierr = -1`). Upstream evaluates `2**i` in default 32-bit integer, which
/// overflows at `i = 31` (#669); there is no defined jump to reproduce, so
/// reaching `i = 31` is an explicit error (`DECISIONS #43`).
pub fn look_for_brackets<F>(
    x: f64,
    dx: f64,
    mut func: F,
    imax: usize,
) -> Result<BracketPair, NumError>
where
    F: FnMut(f64) -> Result<(f64, f64), NumError>,
{
    let mut y1 = 0.0_f64;
    let mut y3 = 0.0_f64;
    let mut x1 = x;
    let mut x3 = x;
    for i in 1..=imax {
        // `jump = (2**i)*dx` (`mod_root.f90:199`): `2**i` overflows the
        // default integer for `i >= 31` (#669, `DECISIONS #43`).
        if i >= 31 {
            return Err(NumError::ConvergenceFailed(
                "look_for_brackets reached i = 31, where upstream's integer \
                 2**i overflows (num/private/mod_root.f90:199, #669)",
            ));
        }
        let jump = f64::from(1_u32 << i) * dx;
        let (move_x1, move_x3);
        if i == 1 {
            x1 -= jump;
            x3 += jump;
            move_x1 = true;
            move_x3 = true;
        } else if y1 > 0.0 {
            if y1 < y3 {
                x1 -= jump;
                move_x1 = true;
                move_x3 = false;
            } else {
                x3 += jump;
                move_x3 = true;
                move_x1 = false;
            }
        } else if y1 > y3 {
            x1 -= jump;
            move_x1 = true;
            move_x3 = false;
        } else {
            x3 += jump;
            move_x3 = true;
            move_x1 = false;
        }
        if move_x1 {
            y1 = func(x1)?.0;
        }
        if move_x3 {
            y3 = func(x3)?.0;
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
    // `do_safe_root_with_brackets` ignores `imax`
    // (`num/private/mod_root.f90:42-43`).
    let _ = imax;
    eval_brent_safe_zero(
        x1_in,
        x3_in,
        1.0e-14,
        epsx,
        epsy,
        |x| Ok(func(x)?.0),
        y1_in,
        y3_in,
    )
}

/// `do_safe_root_with_guess` (`num/private/mod_root.f90:47-164`): up to
/// `newt_imax` Newton steps from `x_guess`, then Brent on a bracket.
///
/// As upstream (#663): no bounds or clamp on `x`; the only convergence test
/// is `abs(y) < max(epsy, abs(dydx)*epsx)` (`:102`); Newton stops on an
/// `f` error, on `dydx == 0` or when `abs(y)` grows (`:89-108`), and
/// continues through NaN values as the Fortran comparisons do. Newton
/// records `x1, y1` (for `y < 0`) or `x3, y3` and marks only `have_x1` or
/// `have_x3` (`:109-114`). Without both `x1` and `x3`, [`look_for_brackets`]
/// starts from `x_guess` (`:118-127`); otherwise `f(x1)` and `f(x3)` are
/// evaluated again unless the caller supplied `y1`/`y3` (`:131-147`), and the
/// pair goes to [`safe_root_with_brackets`] with no sign check (`:150`).
/// Returns `(x, f(x))` where upstream returns `x`.
///
/// # Errors
///
/// Propagates an error from `func` outside the Newton steps, from
/// [`look_for_brackets`] or from [`safe_root_with_brackets`], where upstream
/// returns with `ierr /= 0`.
#[allow(clippy::too_many_arguments)]
pub fn safe_root_with_guess<F>(
    mut func: F,
    x_guess: f64,
    dx: f64,
    x1_in: Option<f64>,
    x3_in: Option<f64>,
    y1_in: Option<f64>,
    y3_in: Option<f64>,
    newt_imax: usize,
    imax: usize,
    epsx: f64,
    epsy: f64,
) -> Result<(f64, f64), NumError>
where
    F: FnMut(f64) -> Result<(f64, f64), NumError>,
{
    // `have_x1 = (x1 /= arg_not_provided)` etc. (`:80-83`).
    let mut have_x1 = x1_in.is_some();
    let mut have_x3 = x3_in.is_some();
    let have_y1 = y1_in.is_some();
    let have_y3 = y3_in.is_some();
    let mut x1 = x1_in.unwrap_or(0.0);
    let mut x3 = x3_in.unwrap_or(0.0);
    let mut y1 = y1_in.unwrap_or(0.0);
    let mut y3 = y3_in.unwrap_or(0.0);
    let mut x = x_guess;
    let mut absy_prev = 0.0_f64;

    for i in 1..=newt_imax {
        // `if (ierr /= 0) ... exit ! try safe_root` (`:89-96`).
        let Ok((y, dydx)) = func(x) else {
            break;
        };
        let absy = y.abs();
        if absy < epsy.max(dydx.abs() * epsx) {
            return Ok((x, y));
        }
        if dydx == 0.0 {
            break;
        }
        if i > 1 && absy > absy_prev {
            break;
        }
        absy_prev = absy;
        if y < 0.0 {
            x1 = x;
            y1 = y;
            have_x1 = true;
        } else {
            x3 = x;
            y3 = y;
            have_x3 = true;
        }
        x -= y / dydx;
    }

    if !(have_x1 && have_x3) {
        ((x1, x3), (y1, y3)) = look_for_brackets(x_guess, dx, &mut func, imax)?;
    } else {
        if !have_y1 {
            y1 = func(x1)?.0;
        }
        if !have_y3 {
            y3 = func(x3)?.0;
        }
    }

    safe_root_with_brackets(func, x1, x3, y1, y3, imax, epsx, epsy)
}

/// `safe_root_without_brackets` (`num/public/num_safe_root.dek:120-144`):
/// [`safe_root_with_guess`] with no brackets and
/// `newt_imax = min(imax, newt_imax)` (`:143`).
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
    epsx: f64,
    epsy: f64,
) -> Result<(f64, f64), NumError>
where
    F: FnMut(f64) -> Result<(f64, f64), NumError>,
{
    safe_root_with_guess(
        func,
        x_guess,
        dx,
        None,
        None,
        None,
        None,
        imax.min(newt_imax),
        imax,
        epsx,
        epsy,
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
    // FlexMESA-internal (no upstream counterpart), so it keeps the bounded
    // Brent with its iteration cap.
    let (x_star, _) = brent_safe_zero(
        x1,
        x3,
        1.0e-14,
        epsx,
        epsy,
        |x| Ok(f_primal(x)?.0),
        Some(y1),
        Some(y3),
        imax.max(200),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// With no sign change, the search widens until `i = 31`, where
    /// upstream's default-integer `2**i` overflows, and stops there with an
    /// explicit error (`num/private/mod_root.f90:199`, #669).
    #[test]
    fn look_for_brackets_errors_at_i_31_where_upstream_overflows() {
        let mut calls = 0_usize;
        let err = look_for_brackets(
            0.0,
            0.01,
            |x| {
                calls += 1;
                Ok((1.0 + x * x, 0.0))
            },
            100,
        )
        .expect_err("no sign change");
        assert!(err.to_string().contains("#669"), "{err}");
        // `i = 1` evaluates both ends, `i = 2..30` one end each.
        assert_eq!(calls, 2 + 29);
    }

    /// After Newton has found both `x1` and `x3`, upstream evaluates `f(x1)`
    /// and `f(x3)` again, since Newton never sets `have_y1`/`have_y3`
    /// (`num/private/mod_root.f90:109-114, 128-147`, #663).
    #[test]
    fn safe_root_with_guess_reevaluates_newton_brackets() {
        let mut calls = Vec::new();
        let (root, _) = safe_root_with_guess(
            |x| {
                calls.push(x);
                Ok((x - 0.3, 0.5))
            },
            0.0,
            0.1,
            None,
            None,
            None,
            None,
            3,
            50,
            1e-14,
            1e-14,
        )
        .expect("root");
        // Newton: 0 (y < 0), 0.6 (y > 0), 0 again; then f(x1), f(x3).
        assert_eq!(&calls[..5], &[0.0, 0.6, 0.0, 0.0, 0.6]);
        assert!((root - 0.3).abs() < 1e-12);
    }

    /// A Newton step is taken wherever `x - y/dydx` lands, with no range
    /// test (`mod_root.f90:115`, #663).
    #[test]
    fn safe_root_with_guess_newton_is_unbounded() {
        let (root, _) = safe_root_with_guess(
            |x| Ok((x - 1.0e40, 1.0)),
            0.0,
            1.0,
            None,
            None,
            None,
            None,
            5,
            50,
            1e-12,
            1e-12,
        )
        .expect("one Newton step reaches the root");
        assert_eq!(root, 1.0e40);
    }

    /// The upstream search moves both ends at `i = 1` by `2*dx`, then the end
    /// whose value is nearer zero, with `dx` used with its sign.
    #[test]
    fn look_for_brackets_follows_upstream_steps() {
        let ((x1, x3), (y1, y3)) =
            look_for_brackets(0.0, 0.5, |x| Ok((x - 2.5, 1.0)), 10)
                .expect("bracket");
        // i = 1: jump = 1, x1 = -1, x3 = 1, both values negative and
        // y1 < y3, so i = 2 moves x3 by jump = 2 to 3.
        assert_eq!((x1, x3), (-1.0, 3.0));
        assert!(y1 < 0.0 && y3 > 0.0);
    }
}
