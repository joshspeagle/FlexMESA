//! Fast pre-validated 2D bicubic spline evaluation (`interp_2d` Stage 2 & 3).
//!
//! `interp_evbicub_db` in `flexmesa-interp2d` validates grid monotonicity
//! (`splinck_db`, $O(n_x + n_y)$) and scans all $4 n_x n_y$ coefficients for
//! finiteness on every scalar call. Hoisting table validation outside the
//! radial shell loop isolates the 16-term cubic Hermite evaluation kernel per
//! zone for Stage 2 (single-thread) and Stage 3 (4-thread scoped) benchmarks.

/// Pre-computed 1D cell index, local grid spacing, and cubic Hermite basis
/// polynomials for a single coordinate axis.
#[derive(Debug, Clone, Copy)]
struct AxisHermite {
    /// Lower cell index $i \in \{0, \dots, n - 2\}$.
    idx: usize,
    /// Normalized local coordinate $p = (q - x_i) / h \in [0, 1]$.
    p: f64,
    /// Complementary coordinate $1 - p$.
    pi: f64,
    /// Cell width $h = x_{i+1} - x_i$.
    h: f64,
    /// Squared cell width $h^2$.
    h2: f64,
    /// Reciprocal cell width $1 / h$.
    hi: f64,
    /// Cubic basis term $p(p^2 - 1)$.
    c: f64,
    /// Complementary cubic basis term $(1 - p)((1 - p)^2 - 1)$.
    ci: f64,
    /// Cubic derivative basis term $3 p^2 - 1$.
    cd: f64,
    /// Complementary cubic derivative basis term $-3(1 - p)^2 + 1$.
    cdi: f64,
}

/// Locates the containing interval on a strictly ascending uniform 1D `grid`
/// and evaluates the 1D cubic Hermite basis polynomials at `query`.
#[inline]
fn locate_and_basis_1d(query: f64, grid: &[f64]) -> AxisHermite {
    let n = grid.len();
    let zq = query.clamp(grid[0], grid[n - 1]);
    let d_avg = (grid[n - 1] - grid[0]) / ((n - 1) as f64);
    let mut idx = ((zq - grid[0]) / d_avg).floor() as usize;
    if idx >= n - 1 {
        idx = n - 2;
    }
    if zq < grid[idx] && idx > 0 {
        idx -= 1;
    } else if zq > grid[idx + 1] && idx + 1 < n - 1 {
        idx += 1;
    }
    let h = grid[idx + 1] - grid[idx];
    let hi = 1.0 / h;
    let p = (zq - grid[idx]) * hi;
    let pi = 1.0 - p;
    let p2 = p * p;
    let pi2 = pi * pi;
    AxisHermite {
        idx,
        p,
        pi,
        h,
        h2: h * h,
        hi,
        c: p * (p2 - 1.0),
        ci: pi * (pi2 - 1.0),
        cd: 3.0 * p2 - 1.0,
        cdi: -3.0 * pi2 + 1.0,
    }
}

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
#[inline]
pub fn eval_bicub_prevalidated(
    x_query: f64,
    y_query: f64,
    x_grid: &[f64],
    y_grid: &[f64],
    coeffs: &[[f64; 4]],
) -> [f64; 4 + 2] {
    let ny = y_grid.len();
    let ax = locate_and_basis_1d(x_query, x_grid);
    let ay = locate_and_basis_1d(y_query, y_grid);

    let c00 = &coeffs[ax.idx * ny + ay.idx];
    let c01 = &coeffs[ax.idx * ny + ay.idx + 1];
    let c10 = &coeffs[(ax.idx + 1) * ny + ay.idx];
    let c11 = &coeffs[(ax.idx + 1) * ny + ay.idx + 1];

    let sixth = 1.0 / 6.0;
    let z36th = 1.0 / 36.0;

    let y0_0 = ay.pi * c00[0] + ay.p * c01[0];
    let y1_0 = ay.pi * c10[0] + ay.p * c11[0];
    let y0_1 = ay.pi * c00[1] + ay.p * c01[1];
    let y1_1 = ay.pi * c10[1] + ay.p * c11[1];
    let y0_2 = ay.pi * c00[2] + ay.p * c01[2];
    let y1_2 = ay.pi * c10[2] + ay.p * c11[2];
    let y0_3 = ay.pi * c00[3] + ay.p * c01[3];
    let y1_3 = ay.pi * c10[3] + ay.p * c11[3];

    let cy0_2 = ay.ci * c00[2] + ay.c * c01[2];
    let cy1_2 = ay.ci * c10[2] + ay.c * c11[2];
    let cy0_3 = ay.ci * c00[3] + ay.c * c01[3];
    let cy1_3 = ay.ci * c10[3] + ay.c * c11[3];

    let f = ax.pi * y0_0
        + ax.p * y1_0
        + sixth * ax.h2 * (ax.ci * y0_1 + ax.c * y1_1)
        + sixth * ay.h2 * (ax.pi * cy0_2 + ax.p * cy1_2)
        + z36th * ax.h2 * ay.h2 * (ax.ci * cy0_3 + ax.c * cy1_3);

    let df_dx = ax.hi * (y1_0 - y0_0)
        + sixth * ax.h * (ax.cdi * y0_1 + ax.cd * y1_1)
        + sixth * ax.hi * ay.h2 * (cy1_2 - cy0_2)
        + z36th * ax.h * ay.h2 * (ax.cdi * cy0_3 + ax.cd * cy1_3);

    let cyd0_2 = ay.cdi * c00[2] + ay.cd * c01[2];
    let cyd1_2 = ay.cdi * c10[2] + ay.cd * c11[2];
    let cyd0_3 = ay.cdi * c00[3] + ay.cd * c01[3];
    let cyd1_3 = ay.cdi * c10[3] + ay.cd * c11[3];
    let df0_0 = c01[0] - c00[0];
    let df1_0 = c11[0] - c10[0];
    let df0_1 = c01[1] - c00[1];
    let df1_1 = c11[1] - c10[1];

    let df_dy = ay.hi * (ax.pi * df0_0 + ax.p * df1_0)
        + sixth * ax.h2 * ay.hi * (ax.ci * df0_1 + ax.c * df1_1)
        + sixth * ay.h * (ax.pi * cyd0_2 + ax.p * cyd1_2)
        + z36th * ax.h2 * ay.h * (ax.ci * cyd0_3 + ax.c * cyd1_3);

    let d2f_dx2 = ax.pi * y0_1
        + ax.p * y1_1
        + sixth * ay.h2 * (ax.pi * cy0_3 + ax.p * cy1_3);
    let d2f_dy2 = ax.pi * y0_2
        + ax.p * y1_2
        + sixth * ax.h2 * (ax.ci * y0_3 + ax.c * y1_3);
    let d2f_dxdy = ay.hi * ax.hi * (df1_0 - df0_0)
        + sixth * ax.h * ay.hi * (ax.cdi * df0_1 + ax.cd * df1_1)
        + sixth * ax.hi * ay.h * (cyd1_2 - cyd0_2)
        + z36th * ax.h * ay.h * (ax.cdi * cyd0_3 + ax.cd * cyd1_3);

    [f, df_dx, df_dy, d2f_dx2, d2f_dy2, d2f_dxdy]
}
