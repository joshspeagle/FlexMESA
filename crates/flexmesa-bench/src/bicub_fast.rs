//! Pre-validated 2D bicubic spline evaluation (`interp_2d` Stage 2 & 3).
//!
//! Clamps out-of-range coordinates onto the table domain and delegates to
//! [`flexmesa_interp2d::interp_evbicub_prevalidated_db`], hoisting grid and
//! coefficient validation outside the radial shell loop for Stage 2 and Stage 3
//! benchmarks while honoring the active `NumericsMode`.

use flexmesa_interp2d::interp_evbicub_prevalidated_db;

/// Evaluates `[f, df_dx, df_dy, d2f_dx2, d2f_dy2, d2f_dxdy]` on a
/// pre-validated uniform 2D bicubic spline grid `(x_grid, y_grid, coeffs)`.
///
/// # Arguments
/// * `x_query` - Query coordinate along the first axis ($\log_{10}\rho$).
/// * `y_query` - Query coordinate along the second axis ($\log_{10}T$).
/// * `x_grid` - Strictly ascending uniform 1D grid of length `nx`.
/// * `y_grid` - Strictly ascending uniform 1D grid of length `ny`.
/// * `coeffs` - Flattened bicubic spline coefficients `[f, f_xx, f_yy, f_xxyy]`
///   of length `nx * ny` in row-major order (`i * ny + j`).
///
/// # Returns
/// Six-element array `[f, df_dx, df_dy, d2f_dx2, d2f_dy2, d2f_dxdy]`.
///
/// # Panics
/// Panics if `x_grid` or `y_grid` is empty or the query is non-finite.
#[inline]
#[must_use]
pub fn eval_bicub_prevalidated(
    x_query: f64,
    y_query: f64,
    x_grid: &[f64],
    y_grid: &[f64],
    coeffs: &[[f64; 4]],
) -> [f64; 4 + 2] {
    let xq = if x_query.is_finite() {
        x_query.clamp(x_grid[0], x_grid[x_grid.len() - 1])
    } else {
        x_query
    };
    let yq = if y_query.is_finite() {
        y_query.clamp(y_grid[0], y_grid[y_grid.len() - 1])
    } else {
        y_query
    };
    let ev = interp_evbicub_prevalidated_db(
        xq, yq, x_grid, y_grid, true, true, coeffs,
    )
    .expect("valid prevalidated uniform bicubic grid and finite query");
    [
        ev.f,
        ev.df_dx,
        ev.df_dy,
        ev.d2f_dx2,
        ev.d2f_dy2,
        ev.d2f_dxdy,
    ]
}
