//! Akima (1996, ACM Algorithm 760) rectangular-grid bivariate interpolation and
//! surface fitting (`interp_rgbi3p_db`, `interp_rgsf3p_db`, and `rgpd3p_db`).
//!
//! Reference: Akima, H. (1996), "Algorithm 760: Rectangular-Grid-Data Surface
//! Fitting that Has the Accuracy of a Bicubic Polynomial", ACM Transactions on
//! Mathematical Software, 22(3), 357-361.

pub use crate::akima_pd::rgpd3p_db;
use crate::Interp2dError;
use flexmesa_utils::is_bad;

#[inline]
fn locate_akima_interval(grid: &[f64], val: f64) -> usize {
    let n = grid.len();
    if val <= grid[0] {
        0
    } else if val < grid[n - 1] {
        grid.partition_point(|&g| g <= val)
    } else {
        n
    }
}

/// Evaluates the Akima (1996) rectangular-grid bivariate polynomial at a single
/// point `(xii, yii)` given precomputed partial derivatives `pdd` (`RGPLNL_db`).
#[must_use]
pub fn eval_akima_point(
    xd: &[f64],
    yd: &[f64],
    zd: &[f64],
    pdd: &[[f64; 3]],
    xii: f64,
    yii: f64,
) -> f64 {
    let nxd = xd.len();
    let nyd = yd.len();
    let ixdi = locate_akima_interval(xd, xii);
    let iydi = locate_akima_interval(yd, yii);
    let ixd0 = ixdi.max(1) - 1;
    let iyd0 = iydi.max(1) - 1;

    let x0 = xd[ixd0];
    let y0 = yd[iyd0];
    let z00 = zd[ixd0 * nyd + iyd0];
    let [zx00, zy00, zxy00] = pdd[ixd0 * nyd + iyd0];
    let u = xii - x0;
    let v = yii - y0;

    if (ixdi > 0 && ixdi < nxd) && (iydi > 0 && iydi < nyd) {
        let ixd1 = ixd0 + 1;
        let iyd1 = iyd0 + 1;
        let dx = xd[ixd1] - x0;
        let dxsq = dx * dx;
        let dy = yd[iyd1] - y0;
        let dysq = dy * dy;

        let z10 = zd[ixd1 * nyd + iyd0];
        let z01 = zd[ixd0 * nyd + iyd1];
        let z11 = zd[ixd1 * nyd + iyd1];
        let [zx10, zy10, zxy10] = pdd[ixd1 * nyd + iyd0];
        let [zx01, zy01, zxy01] = pdd[ixd0 * nyd + iyd1];
        let [zx11, zy11, zxy11] = pdd[ixd1 * nyd + iyd1];

        let z0dx = (z10 - z00) / dx;
        let z0dy = (z01 - z00) / dy;
        let z1dy = (z11 - z10) / dy;
        let zx0dy = (zx01 - zx00) / dy;
        let zx1dy = (zx11 - zx10) / dy;
        let zy0dx = (zy10 - zy00) / dx;
        let zy1dx = (zy11 - zy01) / dx;
        let zdxdy = (z1dy - z0dy) / dx;

        let a = zdxdy - zx0dy - zy0dx + zxy00;
        let b = zx1dy - zx0dy - zxy10 + zxy00;
        let c = zy1dx - zy0dx - zxy01 + zxy00;
        let d = zxy11 - zxy10 - zxy01 + zxy00;

        let p00 = z00;
        let p01 = zy00;
        let p02 = (2.0 * (z0dy - zy00) + z0dy - zy01) / dy;
        let p03 = (-2.0 * z0dy + zy01 + zy00) / dysq;
        let p10 = zx00;
        let p11 = zxy00;
        let p12 = (2.0 * (zx0dy - zxy00) + zx0dy - zxy01) / dy;
        let p13 = (-2.0 * zx0dy + zxy01 + zxy00) / dysq;
        let p20 = (2.0 * (z0dx - zx00) + z0dx - zx10) / dx;
        let p21 = (2.0 * (zy0dx - zxy00) + zy0dx - zxy10) / dx;
        let p22 = (3.0 * (3.0 * a - b - c) + d) / (dx * dy);
        let p23 = (-6.0 * a + 2.0 * b + 3.0 * c - d) / (dx * dysq);
        let p30 = (-2.0 * z0dx + zx10 + zx00) / dxsq;
        let p31 = (-2.0 * zy0dx + zxy10 + zxy00) / dxsq;
        let p32 = (-6.0 * a + 3.0 * b + 2.0 * c - d) / (dxsq * dy);
        let p33 = (2.0 * (2.0 * a - b - c) + d) / (dxsq * dysq);

        let q0 = p00 + v * (p01 + v * (p02 + v * p03));
        let q1 = p10 + v * (p11 + v * (p12 + v * p13));
        let q2 = p20 + v * (p21 + v * (p22 + v * p23));
        let q3 = p30 + v * (p31 + v * (p32 + v * p33));
        q0 + u * (q1 + u * (q2 + u * q3))
    } else if ixdi > 0 && ixdi < nxd {
        let ixd1 = ixd0 + 1;
        let dx = xd[ixd1] - x0;
        let dxsq = dx * dx;
        let z10 = zd[ixd1 * nyd + iyd0];
        let [zx10, zy10, zxy10] = pdd[ixd1 * nyd + iyd0];
        let z0dx = (z10 - z00) / dx;
        let zy0dx = (zy10 - zy00) / dx;
        let p20 = (2.0 * (z0dx - zx00) + z0dx - zx10) / dx;
        let p21 = (2.0 * (zy0dx - zxy00) + zy0dx - zxy10) / dx;
        let p30 = (-2.0 * z0dx + zx10 + zx00) / dxsq;
        let p31 = (-2.0 * zy0dx + zxy10 + zxy00) / dxsq;
        let q0 = z00 + v * zy00;
        let q1 = zx00 + v * zxy00;
        let q2 = p20 + v * p21;
        let q3 = p30 + v * p31;
        q0 + u * (q1 + u * (q2 + u * q3))
    } else if iydi > 0 && iydi < nyd {
        let iyd1 = iyd0 + 1;
        let dy = yd[iyd1] - y0;
        let dysq = dy * dy;
        let z01 = zd[ixd0 * nyd + iyd1];
        let [zx01, zy01, zxy01] = pdd[ixd0 * nyd + iyd1];
        let z0dy = (z01 - z00) / dy;
        let zx0dy = (zx01 - zx00) / dy;
        let p02 = (2.0 * (z0dy - zy00) + z0dy - zy01) / dy;
        let p03 = (-2.0 * z0dy + zy01 + zy00) / dysq;
        let p12 = (2.0 * (zx0dy - zxy00) + zx0dy - zxy01) / dy;
        let p13 = (-2.0 * zx0dy + zxy01 + zxy00) / dysq;
        let q0 = z00 + v * (zy00 + v * (p02 + v * p03));
        let q1 = zx00 + v * (zxy00 + v * (p12 + v * p13));
        q0 + u * q1
    } else {
        (z00 + v * zy00) + u * (zx00 + v * zxy00)
    }
}

/// Evaluates Akima (1996) bivariate interpolation at query points `(xi, yi)` on
/// rectangular grid `(xd, yd)` with row-major values `zd` (`interp_rgbi3p_db`).
///
/// # Errors
///
/// Returns [`Interp2dError`] if grid dimensions or query slices are invalid.
pub fn interp_rgbi3p_db(
    xd: &[f64],
    yd: &[f64],
    zd: &[f64],
    xi: &[f64],
    yi: &[f64],
) -> Result<Vec<f64>, Interp2dError> {
    if xi.len() != yi.len() {
        return Err(Interp2dError::DimensionMismatch {
            expected: xi.len(),
            actual: yi.len(),
        });
    }
    if xi.is_empty() {
        return Err(Interp2dError::InsufficientPoints {
            axis: "query",
            min_points: 1,
            actual: 0,
        });
    }
    if xi.iter().any(|&v| is_bad(v)) || yi.iter().any(|&v| is_bad(v)) {
        return Err(Interp2dError::NonFiniteInput);
    }
    let pdd = rgpd3p_db(xd, yd, zd)?;
    Ok(xi
        .iter()
        .zip(yi.iter())
        .map(|(&x, &y)| eval_akima_point(xd, yd, zd, &pdd, x, y))
        .collect())
}

/// Evaluates Akima (1996) bivariate surface fitting on target rectilinear grid
/// `(xi, yi)` of shape `(nxi, nyi)`, returning row-major `zi` of length
/// `nxi * nyi` (`interp_rgsf3p_db`).
///
/// # Errors
///
/// Returns [`Interp2dError`] if input or output grids are invalid.
pub fn interp_rgsf3p_db(
    xd: &[f64],
    yd: &[f64],
    zd: &[f64],
    xi: &[f64],
    yi: &[f64],
) -> Result<Vec<f64>, Interp2dError> {
    if xi.is_empty() {
        return Err(Interp2dError::InsufficientPoints {
            axis: "x",
            min_points: 1,
            actual: 0,
        });
    }
    if yi.is_empty() {
        return Err(Interp2dError::InsufficientPoints {
            axis: "y",
            min_points: 1,
            actual: 0,
        });
    }
    if xi.iter().any(|&v| is_bad(v)) || yi.iter().any(|&v| is_bad(v)) {
        return Err(Interp2dError::NonFiniteInput);
    }
    let pdd = rgpd3p_db(xd, yd, zd)?;
    let nxi = xi.len();
    let nyi = yi.len();
    let mut zi = vec![0.0_f64; nxi * nyi];
    for (ixi, &xval) in xi.iter().enumerate() {
        for (iyi, &yval) in yi.iter().enumerate() {
            zi[ixi * nyi + iyi] =
                eval_akima_point(xd, yd, zd, &pdd, xval, yval);
        }
    }
    Ok(zi)
}
