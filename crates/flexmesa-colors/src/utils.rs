// SPDX-License-Identifier: LGPL-3.0-or-later
//! Photometric quadrature, 1D linear interpolation, and grid search utilities
//! (`colors/private/colors_utils.f90` and `colors/private/bolometric.f90`).
//!
//! Provides non-uniform composite Simpson integration (`simpson_integration`,
//! `simpson_integration_fn`), composite trapezoidal integration
//! (`trapezoidal_integration`), geometric flux dilution (`dilute_flux`),
//! piecewise-linear filter resampling (`linear_interpolate`,
//! `interpolate_array`), and grid cell/radius helpers (`find_interval`,
//! `find_bracket_index`, `find_nearest_point`, `compute_interp_radius`) for
//! both `f64` and forward-mode `DualOrder1<N>` scalars.

use crate::types::{ColorsError, RealScalar};

/// Strips the file extension starting at the first `.` character
/// (`colors_utils.f90:remove_dat`).
#[must_use]
pub fn remove_dat(path: &str) -> &str {
    let trimmed = path.trim();
    match trimmed.find('.') {
        Some(idx) => &trimmed[..idx],
        None => trimmed,
    }
}

/// Applies the geometric dilution factor $(R / d)^2$ to convert surface flux
/// into observed flux at distance $d$ (`colors_utils.f90:dilute_flux`).
#[must_use]
pub fn dilute_flux<T: RealScalar>(surface_flux: &[T], r: T, d: T) -> Vec<T> {
    let scale = (r / d) * (r / d);
    surface_flux.iter().map(|&f| f * scale).collect()
}

/// Composite trapezoidal quadrature over non-uniform grid `x`
/// (`colors_utils.f90:trapezoidal_integration`).
///
/// # Errors
/// Returns [`ColorsError::InvalidArrayLength`] if `x.len() != y.len()` or
/// `x.len() < 2`.
pub fn trapezoidal_integration<T: RealScalar>(
    x: &[f64],
    y: &[T],
) -> Result<T, ColorsError> {
    let n = x.len();
    if y.len() != n {
        return Err(ColorsError::InvalidArrayLength {
            context: "trapezoidal_integration",
            got: y.len(),
            expected: n,
        });
    }
    if n < 2 {
        return Err(ColorsError::InvalidArrayLength {
            context: "trapezoidal_integration",
            got: n,
            expected: 2,
        });
    }
    let half = T::from_f64(0.5);
    let mut sum = T::zero();
    for i in 0..(n - 1) {
        let dx = T::from_f64(x[i + 1] - x[i]);
        sum = sum + half * dx * (y[i + 1] + y[i]);
    }
    Ok(sum)
}

/// Zero-allocation composite non-uniform Simpson quadrature evaluated via
/// closure `y_at(i)` (`colors_utils.f90:simpson_integration`).
///
/// # Errors
/// Returns [`ColorsError::InvalidArrayLength`] if `x.len() != y_len` or
/// `x.len() < 2`.
pub fn simpson_integration_fn<T, F>(
    x: &[f64],
    y_len: usize,
    y_at: F,
) -> Result<T, ColorsError>
where
    T: RealScalar,
    F: Fn(usize) -> T,
{
    let n = x.len();
    if y_len != n {
        return Err(ColorsError::InvalidArrayLength {
            context: "simpson_integration",
            got: y_len,
            expected: n,
        });
    }
    if n < 2 {
        return Err(ColorsError::InvalidArrayLength {
            context: "simpson_integration",
            got: n,
            expected: 2,
        });
    }
    let mut sum = T::zero();
    let mut i = 0usize;
    while i + 2 < n {
        let h1 = x[i + 1] - x[i];
        let h2 = x[i + 2] - x[i + 1];
        let h12 = h1 + h2;
        let w_outer = h12 / 6.0;
        let c0 = T::from_f64(w_outer * (2.0 * h1 - h2) / h1);
        let c1 = T::from_f64(w_outer * (h12 * h12) / (h1 * h2));
        let c2 = T::from_f64(w_outer * (2.0 * h2 - h1) / h2);
        sum = sum + y_at(i) * c0 + y_at(i + 1) * c1 + y_at(i + 2) * c2;
        i += 2;
    }
    if n.is_multiple_of(2) {
        let dx = T::from_f64(0.5 * (x[n - 1] - x[n - 2]));
        sum = sum + dx * (y_at(n - 1) + y_at(n - 2));
    }
    Ok(sum)
}

/// Composite non-uniform Simpson's rule with a trapezoidal tail interval when
/// `x.len()` is even (`colors_utils.f90:simpson_integration`).
///
/// # Errors
/// Returns [`ColorsError::InvalidArrayLength`] if `x.len() != y.len()` or
/// `x.len() < 2`.
pub fn simpson_integration<T: RealScalar>(
    x: &[f64],
    y: &[T],
) -> Result<T, ColorsError> {
    simpson_integration_fn(x, y.len(), |i| y[i])
}

/// 1D piecewise-linear interpolation at `x_val` with flat clamping outside
/// `[x[0], x[n - 1]]` (`knn_interp.f90:linear_interpolate`).
///
/// # Errors
/// Returns [`ColorsError::InvalidArrayLength`] if `x.len() != y.len()` or
/// `x.len() < 2`.
pub fn linear_interpolate<T: RealScalar>(
    x: &[f64],
    y: &[T],
    x_val: f64,
) -> Result<T, ColorsError> {
    let n = x.len();
    if y.len() != n {
        return Err(ColorsError::InvalidArrayLength {
            context: "linear_interpolate",
            got: y.len(),
            expected: n,
        });
    }
    if n < 2 {
        return Err(ColorsError::InvalidArrayLength {
            context: "linear_interpolate",
            got: n,
            expected: 2,
        });
    }
    if x_val <= x[0] {
        return Ok(y[0]);
    }
    if x_val >= x[n - 1] {
        return Ok(y[n - 1]);
    }
    let (low, t) = find_interval(x, x_val);
    Ok(y[low] + (y[low + 1] - y[low]) * T::from_f64(t))
}

/// Resamples a 1D curve `(x_in, y_in)` onto target grid `x_out` using
/// [`linear_interpolate`] (`knn_interp.f90:interpolate_array`).
///
/// # Errors
/// Returns [`ColorsError::InvalidArrayLength`] if `x_in.len() != y_in.len()`,
/// `x_in.len() < 2`, or `x_out.is_empty()`.
pub fn interpolate_array(
    x_in: &[f64],
    y_in: &[f64],
    x_out: &[f64],
) -> Result<Vec<f64>, ColorsError> {
    if x_out.is_empty() {
        return Err(ColorsError::InvalidArrayLength {
            context: "interpolate_array",
            got: 0,
            expected: 1,
        });
    }
    x_out
        .iter()
        .map(|&xv| linear_interpolate(x_in, y_in, xv))
        .collect()
}

/// Checks whether a coordinate axis is a dummy axis (`all == 0`, `999`, or
/// `-999`).
#[must_use]
pub fn is_dummy_axis(x: &[f64]) -> bool {
    x.is_empty()
        || x.iter().all(|&v| v.abs() < 1.0e-30)
        || x.iter().all(|&v| (v - 999.0).abs() < 1.0e-12)
        || x.iter().all(|&v| (v + 999.0).abs() < 1.0e-12)
}

/// Locates the interval `i` (`0 <= i < x.len() - 1`) and fractional coordinate
/// $t \in [0, 1]$ along sorted axis `x` (`colors_utils.f90:find_interval`).
#[must_use]
pub fn find_interval<T: RealScalar>(x: &[f64], val: T) -> (usize, T) {
    let n = x.len();
    if n < 2 || is_dummy_axis(x) {
        return (0, T::zero());
    }
    let v = val.val();
    if v < x[0] {
        return (0, T::zero());
    }
    if v > x[n - 1] {
        return (n - 2, T::one());
    }
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = usize::midpoint(lo, hi);
        if v >= x[mid] {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let dx = x[lo + 1] - x[lo];
    if dx.abs() < 1.0e-30 {
        (lo, T::zero())
    } else {
        let t = (val - T::from_f64(x[lo])) / T::from_f64(dx);
        (lo, t)
    }
}

/// Locates the bracketing index `idx` (`0 <= idx < grid.len() - 1`) in sorted
/// `grid` (`colors_utils.f90:find_bracket_index`).
#[must_use]
pub fn find_bracket_index(grid: &[f64], val: f64) -> usize {
    find_interval(grid, val).0
}

/// Finds the nearest grid index along 1D axis `grid`.
#[inline]
#[must_use]
fn nearest_1d_idx(grid: &[f64], v: f64) -> usize {
    grid.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| ((*a - v).abs()).total_cmp(&((*b - v).abs())))
        .map_or(0, |(idx, _)| idx)
}

/// Finds the nearest grid indices `(i_x, i_y, i_z)` along each axis
/// (`colors_utils.f90:find_nearest_point`).
#[must_use]
pub fn find_nearest_point(
    x_val: f64,
    y_val: f64,
    z_val: f64,
    x_grid: &[f64],
    y_grid: &[f64],
    z_grid: &[f64],
) -> (usize, usize, usize) {
    (
        nearest_1d_idx(x_grid, x_val),
        nearest_1d_idx(y_grid, y_val),
        nearest_1d_idx(z_grid, z_val),
    )
}

/// Computes the normalized Euclidean distance from `(teff, log_g, metallicity)`
/// to the nearest grid point in `(t_grid, g_grid, m_grid)`
/// (`bolometric.f90:compute_interp_radius`).
#[must_use]
pub fn compute_interp_radius(
    teff: f64,
    log_g: f64,
    metallicity: f64,
    t_grid: &[f64],
    g_grid: &[f64],
    m_grid: &[f64],
) -> f64 {
    let axis_dist_sq = |grid: &[f64], val: f64| -> f64 {
        if is_dummy_axis(grid) {
            return 0.0;
        }
        let min_v = grid.iter().copied().fold(f64::INFINITY, f64::min);
        let max_v = grid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let range = (max_v - min_v).max(1.0e-12);
        let nearest = grid[nearest_1d_idx(grid, val)];
        let d = (val - nearest) / range;
        d * d
    };
    (axis_dist_sq(t_grid, teff)
        + axis_dist_sq(g_grid, log_g)
        + axis_dist_sq(m_grid, metallicity))
    .sqrt()
}
