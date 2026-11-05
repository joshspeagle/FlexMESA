//! Compensated summation, binary search, sorting, and quadratic interpolation.
//!
//! Ports utility solvers from `num/public/num_lib.f90`, including Knuth
//! `two_sum`, Neumaier compensated summation, ascending and descending
//! `binary_search`, index permutation sorting `qsort_index`, linear
//! root-finding `find0`, quadratic root-finding `find0_quadratic`, and
//! quadratic maximum interpolation `find_max_quadratic` (with the upstream
//! concavity check bug fixed).

use crate::NumError;

/// Exact Knuth 2Sum transformation returning `(sum, error)` such that
/// $s + e = a + b$ in exact real arithmetic without extended-precision
/// hardware.
#[must_use]
pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    (s, e)
}

/// Evaluates the Neumaier compensated sum of `values` in `f64`.
///
/// Tracks the low-order rounding residual in a separate accumulator `comp`
/// regardless of the relative magnitudes of the running sum and the next term.
#[must_use]
pub fn neumaier_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut comp = 0.0_f64;
    for &x in values {
        let t = sum + x;
        if sum.abs() >= x.abs() {
            comp += (sum - t) + x;
        } else {
            comp += (x - t) + sum;
        }
        sum = t;
    }
    sum + comp
}

/// Locates the 0-based interval index `k` in a monotonic slice `vec`:
/// - For strictly or weakly **increasing** `vec`: returns `None` if
///   `val < vec[0]`, `Some(n - 1)` if `val >= vec[n - 1]`, or `Some(k)` with
///   `vec[k] <= val < vec[k + 1]`.
/// - For **decreasing** `vec`: returns `None` if `val > vec[0]`, `Some(n - 1)`
///   if `val <= vec[n - 1]`, or `Some(k)` with `vec[k] >= val > vec[k + 1]`.
///
/// When `hint` is `Some(h)`, probes the three adjacent intervals `[h - 1, h,
/// h + 1]` in $O(1)$ time before falling back to bisection.
#[must_use]
pub fn binary_search(
    vec: &[f64],
    val: f64,
    hint: Option<usize>,
) -> Option<usize> {
    let n = vec.len();
    if n == 0 {
        return None;
    }
    if n == 1 {
        return Some(0);
    }
    let increasing = vec[n - 1] >= vec[0];
    if increasing {
        if val < vec[0] {
            return None;
        }
        if val >= vec[n - 1] {
            return Some(n - 1);
        }
        if let Some(h) = hint {
            for cand in [h.saturating_sub(1), h, h + 1] {
                if cand + 1 < n && vec[cand] <= val && val < vec[cand + 1] {
                    return Some(cand);
                }
            }
        }
        let mut lo = 0usize;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = usize::midpoint(lo, hi);
            if val >= vec[mid] {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Some(lo)
    } else {
        if val > vec[0] {
            return None;
        }
        if val <= vec[n - 1] {
            return Some(n - 1);
        }
        if let Some(h) = hint {
            for cand in [h.saturating_sub(1), h, h + 1] {
                if cand + 1 < n && vec[cand] >= val && val > vec[cand + 1] {
                    return Some(cand);
                }
            }
        }
        let mut lo = 0usize;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = usize::midpoint(lo, hi);
            if val <= vec[mid] {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Some(lo)
    }
}

/// 1-based wrapper around [`binary_search`] matching upstream Fortran
/// `binary_search`: returns `0` when `val` lies below the first knot
/// (increasing) or above the first knot (decreasing), and `1..=n` otherwise.
#[must_use]
pub fn binary_search_1based(vec: &[f64], val: f64, hint: usize) -> usize {
    let hint_0 = hint.checked_sub(1);
    binary_search(vec, val, hint_0).map_or(0, |k| k + 1)
}

/// Returns the 0-based permutation indices sorting `vals` in non-decreasing
/// order (`vals[idx[k]] <= vals[idx[k + 1]]`).
#[must_use]
pub fn qsort_index(vals: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..vals.len()).collect();
    idx.sort_by(|&a, &b| vals[a].total_cmp(&vals[b]));
    idx
}

/// Evaluates linear interpolation between `(x1, y1)` and `(x2, y2)` at `x`.
#[must_use]
pub fn linear_interp(x1: f64, y1: f64, x2: f64, y2: f64, x: f64) -> f64 {
    if x2 == x1 {
        return f64::midpoint(y1, y2);
    }
    y1 + (y2 - y1) * (x - x1) / (x2 - x1)
}

/// Finds the linear root $x \in [\min(x_1, x_2), \max(x_1, x_2)]$ where the
/// secant through $(x_1, y_1)$ and $(x_2, y_2)$ crosses zero.
///
/// # Errors
///
/// Returns [`NumError::InvalidParameter`] if $y_1$ and $y_2$ have the same
/// strict sign.
pub fn find0(x1: f64, y1: f64, x2: f64, y2: f64) -> Result<f64, NumError> {
    if y1 == 0.0 {
        return Ok(x1);
    }
    if y2 == 0.0 {
        return Ok(x2);
    }
    if (y1 > 0.0 && y2 > 0.0) || (y1 < 0.0 && y2 < 0.0) || y1 == y2 {
        return Err(NumError::InvalidParameter(
            "y1 and y2 must bracket zero in find0",
        ));
    }
    Ok(x1 - y1 * (x2 - x1) / (y2 - y1))
}

/// Finds the root of the quadratic polynomial passing through $(x_1, y_1)$,
/// $(x_2, y_2)$, and $(x_3, y_3)$ ($x_1 < x_2 < x_3$ or $x_1 > x_2 > x_3$)
/// that lies within the interval $[x_1, x_3]$ when $y_1$ and $y_3$ bracket
/// zero.
///
/// # Errors
///
/// Returns [`NumError::InvalidParameter`] if the abscissas are not strictly
/// ordered or $y_1, y_3$ do not bracket zero.
pub fn find0_quadratic(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    x3: f64,
    y3: f64,
) -> Result<(f64, f64), NumError> {
    if !((x1 < x2 && x2 < x3) || (x1 > x2 && x2 > x3)) {
        return Err(NumError::InvalidParameter(
            "x1, x2, x3 must be strictly monotonic in find0_quadratic",
        ));
    }
    if y1 == 0.0 {
        return Ok((x1, 0.0));
    }
    if y2 == 0.0 {
        return Ok((x2, 0.0));
    }
    if y3 == 0.0 {
        return Ok((x3, 0.0));
    }
    if (y1 > 0.0 && y3 > 0.0) || (y1 < 0.0 && y3 < 0.0) {
        return Err(NumError::InvalidParameter(
            "y1 and y3 must bracket zero in find0_quadratic",
        ));
    }
    let dx12 = x2 - x1;
    let dx23 = x3 - x2;
    let dx13 = x3 - x1;
    let b = (y2 - y1) / dx12;
    let c = ((y3 - y2) / dx23 - b) / dx13;
    // Newton form: p(x) = y1 + b*(x - x1) + c*(x - x1)*(x - x2)
    // Expanded in dx = x - x1: c*dx^2 + (b - c*dx12)*dx + y1 = 0
    let a_q = c;
    let b_q = b - c * dx12;
    let c_q = y1;
    if a_q.abs() < 1e-15 * b_q.abs() {
        let root = find0(x1, y1, x3, y3)?;
        return Ok((root, 0.0));
    }
    let disc = b_q * b_q - 4.0 * a_q * c_q;
    if disc < 0.0 {
        return Err(NumError::InvalidParameter("negative discriminant"));
    }
    let s = disc.sqrt();
    let xmin = x1.min(x3);
    let xmax = x1.max(x3);
    for sign in [1.0_f64, -1.0_f64] {
        let dx = (-b_q + sign * s) / (2.0 * a_q);
        let cand = x1 + dx;
        if cand >= xmin - 1e-12 && cand <= xmax + 1e-12 {
            let y_cand = y1 + b * (cand - x1) + c * (cand - x1) * (cand - x2);
            return Ok((cand.clamp(xmin, xmax), y_cand));
        }
    }
    Err(NumError::BracketingFailed(
        "no quadratic root inside [x1, x3]",
    ))
}

/// Finds the maximum $(x_{\max}, y_{\max})$ of the concave quadratic passing
/// through $(x_1, y_1)$, $(x_2, y_2)$, and $(x_3, y_3)$.
///
/// Fixes the upstream `num_lib.f90` line 581 typo (`if (y2 < max(y1, y2))`) by
/// enforcing $y_2 \ge \max(y_1, y_3)$ and strict concavity $c < 0$.
///
/// # Errors
///
/// Returns [`NumError::InvalidParameter`] if abscissas are not strictly
/// ordered, $y_2 < \max(y_1, y_3)$, or the parabola is not strictly concave.
pub fn find_max_quadratic(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    x3: f64,
    y3: f64,
) -> Result<(f64, f64), NumError> {
    if !((x1 < x2 && x2 < x3) || (x1 > x2 && x2 > x3)) {
        return Err(NumError::InvalidParameter(
            "x1, x2, x3 must be strictly monotonic in find_max_quadratic",
        ));
    }
    if y2 < y1.max(y3) {
        return Err(NumError::InvalidParameter(
            "interior value y2 must be >= max(y1, y3)",
        ));
    }
    let dx1 = x2 - x1;
    let dx2 = x3 - x2;
    let dx12 = x3 - x1;
    let b = (y2 - y1) / dx1;
    let c = ((y3 - y2) / dx2 - b) / dx12;
    if c >= 0.0 {
        return Err(NumError::InvalidParameter(
            "quadratic is not strictly concave (c >= 0)",
        ));
    }
    // Stationary point of p(x) = y1 + b*(x - x1) + c*(x - x1)*(x - x2):
    // dp/dx = b + c*(2*(x - x1) - dx1) = 0 => x - x1 = 0.5*(dx1 - b/c)
    let dxmax = 0.5 * (dx1 - b / c);
    let xmax = x1 + dxmax;
    let ymax = y1 + dxmax * b + dxmax * (xmax - x2) * c;
    Ok((xmax, ymax))
}

/// Solves for the two-piece linear expansion coefficients $(a, b, c)$ at $x$
/// in a monotonic 3-point grid $(x_0, x_1, x_2)$ such that
/// $f(x) = a f(x_0) + b f(x_1) + c f(x_2)$.
///
/// # Errors
///
/// Returns [`NumError::InvalidParameter`] if $x$ lies outside $[x_0, x_2]$.
pub fn two_piece_linear_coeffs(
    x: f64,
    x0: f64,
    x1: f64,
    x2: f64,
) -> Result<(f64, f64, f64), NumError> {
    if !((x0 < x1 && x1 < x2 && x >= x0 && x <= x2)
        || (x0 > x1 && x1 > x2 && x <= x0 && x >= x2))
    {
        return Err(NumError::InvalidParameter(
            "x outside monotonic (x0, x1, x2) range",
        ));
    }
    if x == x1 {
        return Ok((0.0, 1.0, 0.0));
    }
    if (x0 < x1 && x < x1) || (x0 > x1 && x > x1) {
        let a = (x1 - x) / (x1 - x0);
        Ok((a, 1.0 - a, 0.0))
    } else {
        let c = (x - x1) / (x2 - x1);
        Ok((0.0, 1.0 - c, c))
    }
}
