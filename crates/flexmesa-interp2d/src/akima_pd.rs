//! Partial-derivative estimation (`rgpd3p_db`) on rectangular grids for Akima
//! (1996, ACM Algorithm 760) bivariate interpolation.
//!
//! Estimates $(z_x, z_y, z_{xy})$ at every node $(x_{d,i}, y_{d,j})$ of a
//! rectangular grid by forming volatility-weighted combinations of up to four
//! 4-point 1D Lagrange stencils along $x$ and $y$ and up to sixteen
//! $4 \times 4$ tensor-product stencils in 2D.
//!
//! # Algorithm Overview
//!
//! 1. **1D Stencil Sweep (`eval_1d_stencils`)**: At each grid point
//!    $(x_0, y_0)$, evaluates up to four 4-point 1D Lagrange stencils (`IDLT`)
//!    along $x$ and along $y$. Each stencil yields a third-order derivative
//!    estimate `pez` and a volatility measure `volf` computed as the sum of
//!    squared residuals from a 4-point linear least-squares fit
//!    $z \approx b_{00} + b_1 (x - x_0)$.
//! 2. **2D Cross-Derivative Sweep**: Pairs every valid $x$-stencil with every
//!    valid $y$-stencil to form a $4 \times 4$ 2D patch, estimates $z_{xy}$ via
//!    tensor-product Lagrange weights (`pezxy`), and weights each patch by the
//!    reciprocal of its 16-point bilinear least-squares residual variance.
//! 3. **Small-Grid Boundary Extrapolation (`supplement_z_grid`)**: When the
//!    input grid has only $N = 2$ or $N = 3$ points along an axis, synthetic
//!    exterior stencil values are extrapolated via linear (`z2f`) or quadratic
//!    (`z3f`) polynomials so the 4-point stencil formulas remain well-defined.
//!
//! Reference: Akima, H. (1996), "Algorithm 760: Rectangular-Grid-Data Surface
//! Fitting that Has the Accuracy of a Bicubic Polynomial", ACM Transactions on
//! Mathematical Software, 22(3), 357-361.

use crate::spline1d::splinck_db;
use crate::Interp2dError;
use flexmesa_utils::is_bad;

/// Relative index offsets for the 4 candidate 4-point stencils around `i0`.
const IDLT: [[isize; 3]; 4] =
    [[-3, -2, -1], [-2, -1, 1], [-1, 1, 2], [1, 2, 3]];

/// Linear extrapolation helper for 2-point grids in `RGPD3P_db`.
///
/// Extrapolates the line through `(0, zz0)` and `(xx1, zz1)` to offset `xx2`.
#[inline]
fn z2f(xx1: f64, xx2: f64, zz0: f64, zz1: f64) -> f64 {
    (zz1 - zz0) * xx2 / xx1 + zz0
}

/// Quadratic extrapolation helper for 3-point grids in `RGPD3P_db`.
///
/// Extrapolates the parabola through `(0, zz0)`, `(xx1, zz1)`, and `(xx2, zz2)`
/// to offset `xx3`.
#[inline]
fn z3f(xx1: f64, xx2: f64, xx3: f64, zz0: f64, zz1: f64, zz2: f64) -> f64 {
    ((zz2 - zz0) * (xx3 - xx1) / xx2 - (zz1 - zz0) * (xx3 - xx2) / xx1)
        * (xx3 / (xx2 - xx1))
        + zz0
}

/// Cached 1D stencil quantities and weighted first derivative along one axis.
#[derive(Clone, Copy, Default)]
struct AxisStencils {
    /// Coordinate displacements `[d1, d2, d3]` for each of the 4 stencils.
    da: [[f64; 3]; 4],
    /// Stencil values `[z1, z2, z3]` for each of the 4 stencils.
    za: [[f64; 3]; 4],
    /// Lagrange derivative weights `[c1, c2, c3]` for each of the 4 stencils.
    ca: [[f64; 3]; 4],
    /// Sum of displacements $d_1 + d_2 + d_3$ per stencil.
    sa: [f64; 4],
    /// Sum of squared displacements $d_1^2 + d_2^2 + d_3^2$ per stencil.
    ssa: [f64; 4],
    /// Linear least-squares intercept $b_{00}$ per stencil.
    b00a: [f64; 4],
    /// Linear least-squares slope $b_1$ per stencil.
    b1a: [f64; 4],
    /// Volatility-weighted 1D derivative estimate at node `i0`.
    deriv: f64,
}

/// Evaluates the 4 candidate 1D stencils along a single grid axis around `i0`.
///
/// Computes Lagrange 4-point cubic derivative estimates `pez` and linear
/// least-squares volatility weights `1 / (volf * ss)` for each valid stencil.
fn eval_1d_stencils<F>(
    grid: &[f64],
    i0: usize,
    z00: f64,
    eps_scale: f64,
    get_z: F,
) -> AxisStencils
where
    F: Fn(usize) -> f64,
{
    let n = grid.len();
    let n0 = n.max(4) as isize;
    let c0 = grid[i0];
    let mut out = AxisStencils::default();
    let (mut smpef, mut smwtf, mut smpei, mut smwti) = (0.0, 0.0, 0.0, 0.0);

    for ipe in 0..4 {
        let [o1, o2, o3] = IDLT[ipe];
        let i0s = i0 as isize;
        let (i1, i2, i3) = (i0s + o1, i0s + o2, i0s + o3);
        if i1.min(i2).min(i3) < 0 || i1.max(i2).max(i3) >= n0 {
            continue;
        }
        let (u1, u2, u3) = (i1 as usize, i2 as usize, i3 as usize);
        let (d1, z1) = (grid[u1] - c0, get_z(u1));
        // Supplement stencil coordinates when grid has fewer than 4 points.
        let (d2, d3, z2, z3) = if n >= 4 {
            (grid[u2] - c0, grid[u3] - c0, get_z(u2), get_z(u3))
        } else if n == 3 {
            let (d2_v, z2_v) = (grid[u2] - c0, get_z(u2));
            let d3_v = 2.0 * grid[2] - grid[1] - c0;
            (d2_v, d3_v, z2_v, z3f(d1, d2_v, d3_v, z00, z1, z2_v))
        } else {
            let d2_v = 2.0 * grid[1] - grid[0] - c0;
            let d3_v = 2.0 * grid[0] - grid[1] - c0;
            (d2_v, d3_v, z2f(d1, d2_v, z00, z1), z2f(d1, d3_v, z00, z1))
        };
        let c1 = d2 * d3 / ((d1 - d2) * (d1 - d3));
        let c2 = d3 * d1 / ((d2 - d3) * (d2 - d1));
        let c3 = d1 * d2 / ((d3 - d1) * (d3 - d2));
        let pez =
            c1 * (z1 - z00) / d1 + c2 * (z2 - z00) / d2 + c3 * (z3 - z00) / d3;

        // Volatility factor from 4-point linear least-squares fit residuals.
        let s = d1 + d2 + d3;
        let sz = z00 + z1 + z2 + z3;
        let ss = d1 * d1 + d2 * d2 + d3 * d3;
        let szd = d1 * z1 + d2 * z2 + d3 * z3;
        let dnm = 4.0 * ss - s * s;
        let b00 = (ss * sz - s * szd) / dnm;
        let b1 = (4.0 * szd - s * sz) / dnm;
        let volf = (z00 - b00).powi(2)
            + (z1 - (b00 + b1 * d1)).powi(2)
            + (z2 - (b00 + b1 * d2)).powi(2)
            + (z3 - (b00 + b1 * d3)).powi(2);
        let epsln = (z00 * z00 + z1 * z1 + z2 * z2 + z3 * z3) * eps_scale;
        if volf > epsln {
            let wt = 1.0 / (volf * ss);
            smpef += wt * pez;
            smwtf += wt;
        } else {
            smpei += pez;
            smwti += 1.0;
        }
        out.da[ipe] = [d1, d2, d3];
        out.za[ipe] = [z1, z2, z3];
        out.ca[ipe] = [c1, c2, c3];
        out.sa[ipe] = s;
        out.ssa[ipe] = ss;
        out.b00a[ipe] = b00;
        out.b1a[ipe] = b1;
    }
    out.deriv = if smwti < 0.5 { smpef / smwtf } else { smpei / smwti };
    out
}

/// Estimates the partial derivatives $(z_x, z_y, z_{xy})$ at every point of a
/// rectangular grid `(xd, yd)` of shape `(nxd, nyd)` (`nxd >= 2`, `nyd >= 2`)
/// using the revised Akima (1996) volatility-weighted four-point stencil
/// (`RGPD3P_db`).
///
/// Input `zd` has row-major shape `(nxd, nyd)`. Returns a vector of length
/// `nxd * nyd` containing `[zx, zy, zxy]` at each node `(ix, iy)`.
///
/// # Errors
///
/// Returns [`Interp2dError`] if `nxd < 2`, `nyd < 2`, `xd` or `yd` is not
/// strictly ascending, or `zd` length/finiteness is invalid.
pub fn rgpd3p_db(
    xd: &[f64],
    yd: &[f64],
    zd: &[f64],
) -> Result<Vec<[f64; 3]>, Interp2dError> {
    let (nxd, nyd) = (xd.len(), yd.len());
    if nxd < 2 {
        return Err(Interp2dError::InsufficientPoints {
            axis: "x",
            min_points: 2,
            actual: nxd,
        });
    }
    if nyd < 2 {
        return Err(Interp2dError::InsufficientPoints {
            axis: "y",
            min_points: 2,
            actual: nyd,
        });
    }
    if zd.len() != nxd * nyd {
        return Err(Interp2dError::DimensionMismatch {
            expected: nxd * nyd,
            actual: zd.len(),
        });
    }
    if zd.iter().any(|&v| is_bad(v)) {
        return Err(Interp2dError::NonFiniteInput);
    }
    splinck_db(xd, 1.0e-3, "x")?;
    splinck_db(yd, 1.0e-3, "y")?;

    let (nx0, ny0) = (nxd.max(4) as isize, nyd.max(4) as isize);
    let eps_scale = f64::from(1.0e-12_f32);
    let mut pdd = vec![[0.0_f64; 3]; nxd * nyd];

    for iy0 in 0..nyd {
        for ix0 in 0..nxd {
            let z00 = zd[ix0 * nyd + iy0];
            let xs = eval_1d_stencils(xd, ix0, z00, eps_scale, |u| {
                zd[u * nyd + iy0]
            });
            let ys = eval_1d_stencils(yd, iy0, z00, eps_scale, |v| {
                zd[ix0 * nyd + v]
            });

            // Combine valid x and y stencils to estimate cross derivative z_xy.
            // If any stencil patch has near-zero residual variance (smwti > 0),
            // return the unweighted average across exact-fit patches; otherwise
            // return the volatility-weighted average smpef / smwtf.
            let (mut smpef, mut smwtf, mut smpei, mut smwti) =
                (0.0, 0.0, 0.0, 0.0);

            for ipex in 0..4 {
                let [ox1, ox2, ox3] = IDLT[ipex];
                let ix0s = ix0 as isize;
                let (ix1, ix2, ix3) = (ix0s + ox1, ix0s + ox2, ix0s + ox3);
                if ix1.min(ix2).min(ix3) < 0 || ix1.max(ix2).max(ix3) >= nx0 {
                    continue;
                }
                let (xs_loc, zs_x, [cx1, cx2, cx3]) =
                    (xs.da[ipex], xs.za[ipex], xs.ca[ipex]);
                let (sx, sxx, b00x, b10) =
                    (xs.sa[ipex], xs.ssa[ipex], xs.b00a[ipex], xs.b1a[ipex]);

                for ipey in 0..4 {
                    let [oy1, oy2, oy3] = IDLT[ipey];
                    let (iy1, iy2, iy3) = (
                        iy0 as isize + oy1,
                        iy0 as isize + oy2,
                        iy0 as isize + oy3,
                    );
                    if iy1.min(iy2).min(iy3) < 0
                        || iy1.max(iy2).max(iy3) >= ny0
                    {
                        continue;
                    }
                    let (ys_loc, zs_y, cys) =
                        (ys.da[ipey], ys.za[ipey], ys.ca[ipey]);
                    let (sy, syy, b00y, b01) = (
                        ys.sa[ipey],
                        ys.ssa[ipey],
                        ys.b00a[ipey],
                        ys.b1a[ipey],
                    );

                    let z_grid = supplement_z_grid(
                        nxd,
                        nyd,
                        zd,
                        [ix1 as usize, ix2 as usize, ix3 as usize],
                        [iy1 as usize, iy2 as usize, iy3 as usize],
                        xs_loc,
                        ys_loc,
                        zs_x,
                        zs_y,
                    );

                    // Primary 2D cross-derivative estimate pezxy.
                    let mut pe_row = [0.0_f64; 3];
                    for r in 0..3 {
                        for c in 0..3 {
                            let dzxy = (z_grid[r][c] - zs_x[r] - zs_y[c] + z00)
                                / (xs_loc[r] * ys_loc[c]);
                            pe_row[r] += cys[c] * dzxy;
                        }
                    }
                    let pezxy =
                        cx1 * pe_row[0] + cx2 * pe_row[1] + cx3 * pe_row[2];

                    // Least-squares bilinear fit and 16-point 2D volatility.
                    let b00 = f64::midpoint(b00x, b00y);
                    let sxxyy = sxx * syy;
                    let mut sxyz = 0.0_f64;
                    for r in 0..3 {
                        let row_sum = ys_loc[0] * z_grid[r][0]
                            + ys_loc[1] * z_grid[r][1]
                            + ys_loc[2] * z_grid[r][2];
                        sxyz += xs_loc[r] * row_sum;
                    }
                    let b11 = (sxyz
                        - b00 * sx * sy
                        - b10 * sxx * sy
                        - b01 * sx * syy)
                        / sxxyy;

                    let mut volf = (z00 - b00).powi(2);
                    let mut z_sq_sum = z00 * z00;
                    for c in 0..3 {
                        volf += (zs_y[c] - (b00 + b01 * ys_loc[c])).powi(2);
                        z_sq_sum += zs_y[c] * zs_y[c];
                    }
                    for r in 0..3 {
                        volf += (zs_x[r] - (b00 + b10 * xs_loc[r])).powi(2);
                        z_sq_sum += zs_x[r] * zs_x[r];
                        for c in 0..3 {
                            let pred = b00
                                + b01 * ys_loc[c]
                                + xs_loc[r] * (b10 + b11 * ys_loc[c]);
                            volf += (z_grid[r][c] - pred).powi(2);
                            z_sq_sum += z_grid[r][c] * z_grid[r][c];
                        }
                    }
                    if volf > z_sq_sum * eps_scale {
                        let wt = 1.0 / (volf * sxxyy);
                        smpef += wt * pezxy;
                        smwtf += wt;
                    } else {
                        smpei += pezxy;
                        smwti += 1.0;
                    }
                }
            }
            let zxydi = if smwti < 0.5 { smpef / smwtf } else { smpei / smwti };
            pdd[ix0 * nyd + iy0] = [xs.deriv, ys.deriv, zxydi];
        }
    }

    Ok(pdd)
}

/// Populates the 3x3 stencil value matrix `z` around `(ix0, iy0)`,
/// extrapolating missing rows or columns via `z2f` or `z3f` when `nxd < 4` or
/// `nyd < 4`.
///
/// Copies all valid interior nodes `(ixs[r], iys[c])` from `zd`, then
/// extrapolates along `x` for small `nxd` (`2` or `3`) and along `y` for small
/// `nyd` (`2` or `3`) matching upstream `RGPD3P_db`.
#[allow(clippy::too_many_arguments)]
fn supplement_z_grid(
    nxd: usize,
    nyd: usize,
    zd: &[f64],
    ixs: [usize; 3],
    iys: [usize; 3],
    xs: [f64; 3],
    ys: [f64; 3],
    zs_x: [f64; 3],
    zs_y: [f64; 3],
) -> [[f64; 3]; 3] {
    let ([x1, x2, x3], [y1, y2, y3]) = (xs, ys);
    let mut z = [[0.0_f64; 3]; 3];
    let (nx_in, ny_in) = ((nxd - 1).min(3), (nyd - 1).min(3));
    for r in 0..nx_in {
        for c in 0..ny_in {
            z[r][c] = zd[ixs[r] * nyd + iys[c]];
        }
    }
    for c in 0..ny_in {
        if nxd == 3 {
            z[2][c] = z3f(x1, x2, x3, zs_y[c], z[0][c], z[1][c]);
        } else if nxd == 2 {
            z[1][c] = z2f(x1, x2, zs_y[c], z[0][c]);
            z[2][c] = z2f(x1, x3, zs_y[c], z[0][c]);
        }
    }
    for r in 0..3 {
        if nyd == 3 {
            z[r][2] = z3f(y1, y2, y3, zs_x[r], z[r][0], z[r][1]);
        } else if nyd == 2 {
            z[r][1] = z2f(y1, y2, zs_x[r], z[r][0]);
            z[r][2] = z2f(y1, y3, zs_x[r], z[r][0]);
        }
    }
    z
}
