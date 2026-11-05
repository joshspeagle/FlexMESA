//! Partial-derivative estimation (`rgpd3p_db`) on rectangular grids for Akima
//! (1996, ACM Algorithm 760) bivariate interpolation.

use crate::spline1d::splinck_db;
use crate::Interp2dError;
use flexmesa_utils::is_bad;

const IDLT: [[isize; 3]; 4] =
    [[-3, -2, -1], [-2, -1, 1], [-1, 1, 2], [1, 2, 3]];

#[inline]
fn z2f(xx1: f64, xx2: f64, zz0: f64, zz1: f64) -> f64 {
    (zz1 - zz0) * xx2 / xx1 + zz0
}

#[inline]
fn z3f(xx1: f64, xx2: f64, xx3: f64, zz0: f64, zz1: f64, zz2: f64) -> f64 {
    ((zz2 - zz0) * (xx3 - xx1) / xx2 - (zz1 - zz0) * (xx3 - xx2) / xx1)
        * (xx3 / (xx2 - xx1))
        + zz0
}

#[derive(Clone, Copy, Default)]
struct AxisStencils {
    da: [[f64; 3]; 4],
    za: [[f64; 3]; 4],
    ca: [[f64; 3]; 4],
    sa: [f64; 4],
    ssa: [f64; 4],
    b00a: [f64; 4],
    b1a: [f64; 4],
    deriv: f64,
}

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
    let mut smpef = 0.0_f64;
    let mut smwtf = 0.0_f64;
    let mut smpei = 0.0_f64;
    let mut smwti = 0.0_f64;

    for ipe in 0..4 {
        let i1 = (i0 as isize) + IDLT[ipe][0];
        let i2 = (i0 as isize) + IDLT[ipe][1];
        let i3 = (i0 as isize) + IDLT[ipe][2];
        if i1 < 0 || i2 < 0 || i3 < 0 || i1 >= n0 || i2 >= n0 || i3 >= n0 {
            continue;
        }
        let u1 = i1 as usize;
        let u2 = i2 as usize;
        let u3 = i3 as usize;
        let d1 = grid[u1] - c0;
        let z1 = get_z(u1);
        let (d2, d3, z2, z3) = if n >= 4 {
            (grid[u2] - c0, grid[u3] - c0, get_z(u2), get_z(u3))
        } else if n == 3 {
            let d2_v = grid[u2] - c0;
            let z2_v = get_z(u2);
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
    out.deriv = if smwti < 0.5 {
        smpef / smwtf
    } else {
        smpei / smwti
    };
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
    let nxd = xd.len();
    let nyd = yd.len();
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

    let nx0 = nxd.max(4) as isize;
    let ny0 = nyd.max(4) as isize;
    let eps_scale = f64::from(1.0e-12_f32);
    let mut pdd = vec![[0.0_f64; 3]; nxd * nyd];

    for iy0 in 0..nyd {
        for ix0 in 0..nxd {
            let z00 = zd[ix0 * nyd + iy0];
            let xs =
                eval_1d_stencils(xd, ix0, z00, eps_scale, |u| zd[u * nyd + iy0]);
            let ys =
                eval_1d_stencils(yd, iy0, z00, eps_scale, |v| zd[ix0 * nyd + v]);

            let mut smpef = 0.0_f64;
            let mut smwtf = 0.0_f64;
            let mut smpei = 0.0_f64;
            let mut smwti = 0.0_f64;

            for ipex in 0..4 {
                let ix1 = (ix0 as isize) + IDLT[ipex][0];
                let ix2 = (ix0 as isize) + IDLT[ipex][1];
                let ix3 = (ix0 as isize) + IDLT[ipex][2];
                if ix1 < 0
                    || ix2 < 0
                    || ix3 < 0
                    || ix1 >= nx0
                    || ix2 >= nx0
                    || ix3 >= nx0
                {
                    continue;
                }
                let xs_loc = xs.da[ipex];
                let zs_x = xs.za[ipex];
                let [cx1, cx2, cx3] = xs.ca[ipex];
                let sx = xs.sa[ipex];
                let sxx = xs.ssa[ipex];
                let b00x = xs.b00a[ipex];
                let b10 = xs.b1a[ipex];

                for ipey in 0..4 {
                    let iy1 = (iy0 as isize) + IDLT[ipey][0];
                    let iy2 = (iy0 as isize) + IDLT[ipey][1];
                    let iy3 = (iy0 as isize) + IDLT[ipey][2];
                    if iy1 < 0
                        || iy2 < 0
                        || iy3 < 0
                        || iy1 >= ny0
                        || iy2 >= ny0
                        || iy3 >= ny0
                    {
                        continue;
                    }
                    let ys_loc = ys.da[ipey];
                    let zs_y = ys.za[ipey];
                    let cys = ys.ca[ipey];
                    let sy = ys.sa[ipey];
                    let syy = ys.ssa[ipey];
                    let b00y = ys.b00a[ipey];
                    let b01 = ys.b1a[ipey];

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

                    let b00 = f64::midpoint(b00x, b00y);
                    let sxxyy = sxx * syy;
                    let mut sxyz = 0.0_f64;
                    for r in 0..3 {
                        let mut row_sum = 0.0_f64;
                        for c in 0..3 {
                            row_sum += ys_loc[c] * z_grid[r][c];
                        }
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
                    let epsln = z_sq_sum * eps_scale;
                    if volf > epsln {
                        let wt = 1.0 / (volf * sxxyy);
                        smpef += wt * pezxy;
                        smwtf += wt;
                    } else {
                        smpei += pezxy;
                        smwti += 1.0;
                    }
                }
            }
            let zxydi = if smwti < 0.5 {
                smpef / smwtf
            } else {
                smpei / smwti
            };
            pdd[ix0 * nyd + iy0] = [xs.deriv, ys.deriv, zxydi];
        }
    }

    Ok(pdd)
}

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
    let [x1, x2, x3] = xs;
    let [y1, y2, y3] = ys;
    let mut z = [[0.0_f64; 3]; 3];
    let nx_in = (nxd - 1).min(3);
    let ny_in = (nyd - 1).min(3);
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
