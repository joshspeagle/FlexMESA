//! One-dimensional $C^2$ cubic spline solver (`mkspline_db`, `cspline_db`,
//! `v_spline_db`, and `splinck_db`) backing 2D bicubic splines.

use crate::{BicubicBc, Interp2dError};
use flexmesa_utils::is_bad;

/// Checks whether grid `x` is strictly ascending and whether its intervals are
/// uniformly spaced within relative tolerance `ztol`.
///
/// Returns `Ok(true)` (`ilinx = 1`) when `x` is uniformly spaced within `ztol`,
/// and `Ok(false)` (`ilinx = 2`) when `x` is strictly ascending but non-uniform.
///
/// # Errors
///
/// Returns [`Interp2dError`] if `x` contains non-finite values or is not
/// strictly ascending.
pub fn splinck_db(
    x: &[f64],
    ztol: f64,
    axis: &'static str,
) -> Result<bool, Interp2dError> {
    if x.iter().any(|&v| is_bad(v)) {
        return Err(Interp2dError::NonFiniteInput);
    }
    let n = x.len();
    if n <= 1 {
        return Ok(true);
    }
    let dxavg = (x[n - 1] - x[0]) / ((n - 1) as f64);
    let zeps = (ztol * dxavg).abs();
    let mut uniform = true;
    for i in 1..n {
        let zdiffx = x[i] - x[i - 1];
        if zdiffx <= 0.0 {
            return Err(Interp2dError::NonAscendingGrid { axis });
        }
        if (zdiffx - dxavg).abs() > zeps {
            uniform = false;
        }
    }
    Ok(uniform)
}

/// Computes the second derivatives $f''(x_i)$ of a 1D $C^2$ cubic spline at
/// knots `x` (`n >= 4`) with knot values `y` and boundary conditions
/// `(ibcxmin, bcxmin)` at `x[0]` and `(ibcxmax, bcxmax)` at `x[n - 1]`.
///
/// Returns `(fxx, uniform)` where `fxx[i] = f''(x[i])` and `uniform` indicates
/// whether `x` is evenly spaced within `1e-3`.
///
/// # Errors
///
/// Returns [`Interp2dError`] if `x.len() < 4`, lengths mismatch, `x` is not
/// strictly ascending, or boundary conditions are invalid.
pub fn mkspline_db(
    x: &[f64],
    y: &[f64],
    ibcxmin: BicubicBc,
    bcxmin: f64,
    ibcxmax: BicubicBc,
    bcxmax: f64,
) -> Result<(Vec<f64>, bool), Interp2dError> {
    let n = x.len();
    if n != y.len() {
        return Err(Interp2dError::DimensionMismatch {
            expected: n,
            actual: y.len(),
        });
    }
    if n < 4 {
        return Err(Interp2dError::InsufficientPoints {
            axis: "x",
            min_points: 4,
            actual: n,
        });
    }
    if y.iter().any(|&v| is_bad(v)) || is_bad(bcxmin) || is_bad(bcxmax) {
        return Err(Interp2dError::NonFiniteInput);
    }
    if ibcxmin != BicubicBc::Periodic && ibcxmax == BicubicBc::Periodic {
        return Err(Interp2dError::InvalidBoundaryCondition(-1));
    }
    let uniform = splinck_db(x, 1.0e-3, "x")?;

    let mut fspl4 = vec![[0.0_f64; 4]; n];
    for i in 0..n {
        fspl4[i][0] = y[i];
    }
    if ibcxmin == BicubicBc::FirstDeriv {
        fspl4[0][1] = bcxmin;
    } else if ibcxmin == BicubicBc::SecondDeriv {
        fspl4[0][2] = bcxmin;
    }
    if ibcxmax == BicubicBc::FirstDeriv {
        fspl4[n - 1][1] = bcxmax;
    } else if ibcxmax == BicubicBc::SecondDeriv {
        fspl4[n - 1][2] = bcxmax;
    }

    v_spline_db(ibcxmin as i32, ibcxmax as i32, x, &mut fspl4);

    for item in &mut fspl4 {
        item[2] *= 0.5;
        item[3] /= 6.0;
    }

    let mut fxx = vec![0.0_f64; n];
    for i in 0..(n - 1) {
        fxx[i] = 2.0 * fspl4[i][2];
    }
    fxx[n - 1] = 2.0 * fspl4[n - 2][2]
        + (x[n - 1] - x[n - 2]) * 6.0 * fspl4[n - 2][3];

    Ok((fxx, uniform))
}

/// Exact port of `V_SPLINE_db` from upstream PSPLINE / MESA `bicub_db.f`.
#[allow(clippy::too_many_lines)]
fn v_spline_db(k_bc1: i32, k_bcn: i32, x: &[f64], f: &mut [[f64; 4]]) {
    let n = x.len();
    let mut imin = 0usize;
    let mut imax = n - 1;

    let a1 = match k_bc1 {
        1 => f[0][1],
        5 => (f[1][0] - f[0][0]) / (x[1] - x[0]),
        _ => 0.0,
    };
    let b1 = match k_bc1 {
        2 => f[0][2],
        6 => {
            2.0 * ((f[2][0] - f[1][0]) / (x[2] - x[1])
                - (f[1][0] - f[0][0]) / (x[1] - x[0]))
                / (x[2] - x[0])
        }
        _ => 0.0,
    };
    let an = match k_bcn {
        1 => f[n - 1][1],
        5 => (f[n - 1][0] - f[n - 2][0]) / (x[n - 1] - x[n - 2]),
        _ => 0.0,
    };
    let bn = match k_bcn {
        2 => f[n - 1][2],
        6 => {
            2.0 * ((f[n - 1][0] - f[n - 2][0]) / (x[n - 1] - x[n - 2])
                - (f[n - 2][0] - f[n - 3][0]) / (x[n - 2] - x[n - 3]))
                / (x[n - 1] - x[n - 3])
        }
        _ => 0.0,
    };

    f[n - 1][1] = 0.0;
    f[n - 1][2] = 0.0;
    f[n - 1][3] = 0.0;

    f[0][3] = x[1] - x[0];
    f[1][2] = (f[1][0] - f[0][0]) / f[0][3];
    for i in 1..(n - 1) {
        f[i][3] = x[i + 1] - x[i];
        f[i][1] = 2.0 * (f[i - 1][3] + f[i][3]);
        f[i + 1][2] = (f[i + 1][0] - f[i][0]) / f[i][3];
        f[i][2] = f[i + 1][2] - f[i][2];
    }

    let elem21 = f[0][3];
    let mut elemnn1 = f[n - 2][3];
    let mut wk = vec![0.0_f64; n];

    if k_bc1 == -1 {
        f[0][1] = 2.0 * (f[0][3] + f[n - 2][3]);
        f[0][2] = (f[1][0] - f[0][0]) / f[0][3]
            - (f[n - 1][0] - f[n - 2][0]) / f[n - 2][3];
        wk[0] = f[n - 2][3];
        for item in wk.iter_mut().take(n - 3).skip(1) {
            *item = 0.0;
        }
        wk[n - 3] = f[n - 3][3];
        wk[n - 2] = f[n - 2][3];
    } else if k_bc1 == 1 || k_bc1 == 3 || k_bc1 == 5 {
        f[0][1] = 2.0 * f[0][3];
        f[0][2] = (f[1][0] - f[0][0]) / f[0][3] - a1;
    } else if k_bc1 == 2 || k_bc1 == 4 || k_bc1 == 6 {
        f[0][1] = 2.0 * f[0][3];
        f[0][2] = f[0][3] * b1 / 3.0;
        f[0][3] = 0.0;
    } else if k_bc1 == 7 {
        f[0][1] = -f[0][3];
        f[0][2] = f[2][2] / (x[3] - x[1]) - f[1][2] / (x[2] - x[0]);
        f[0][2] = f[0][2] * f[0][3] * f[0][3] / (x[3] - x[0]);
    } else {
        imin = 1;
        f[1][1] = f[0][3] + 2.0 * f[1][3];
        f[1][2] = f[1][2] * f[1][3] / (f[0][3] + f[1][3]);
    }

    if k_bcn == 1 || k_bcn == 3 || k_bcn == 5 {
        f[n - 1][1] = 2.0 * f[n - 2][3];
        f[n - 1][2] = -(f[n - 1][0] - f[n - 2][0]) / f[n - 2][3] + an;
    } else if k_bcn == 2 || k_bcn == 4 || k_bcn == 6 {
        f[n - 1][1] = 2.0 * f[n - 2][3];
        f[n - 1][2] = f[n - 2][3] * bn / 3.0;
        elemnn1 = 0.0;
    } else if k_bcn == 7 {
        f[n - 1][1] = -f[n - 2][3];
        f[n - 1][2] = f[n - 2][2] / (x[n - 1] - x[n - 3])
            - f[n - 3][2] / (x[n - 2] - x[n - 4]);
        f[n - 1][2] =
            -f[n - 1][2] * f[n - 2][3] * f[n - 2][3] / (x[n - 1] - x[n - 4]);
    } else if k_bc1 != -1 {
        imax = n - 2;
        f[n - 2][1] = 2.0 * f[n - 3][3] + f[n - 2][3];
        f[n - 2][2] =
            f[n - 2][2] * f[n - 3][3] / (f[n - 2][3] + f[n - 3][3]);
    }

    if k_bc1 == -1 {
        for i in 1..(n - 2) {
            let t = f[i - 1][3] / f[i - 1][1];
            f[i][1] -= t * f[i - 1][3];
            f[i][2] -= t * f[i - 1][2];
            wk[i] -= t * wk[i - 1];
            let q = wk[n - 2] / f[i - 1][1];
            wk[n - 2] = -q * f[i - 1][3];
            f[n - 2][1] -= q * wk[i - 1];
            f[n - 2][2] -= q * f[i - 1][2];
        }
        wk[n - 2] += f[n - 3][3];
        let t = wk[n - 2] / f[n - 3][1];
        f[n - 2][1] -= t * wk[n - 3];
        f[n - 2][2] -= t * f[n - 3][2];
        f[n - 2][2] /= f[n - 2][1];
        f[n - 3][2] = (f[n - 3][2] - wk[n - 3] * f[n - 2][2]) / f[n - 3][1];
        for ib in 3..n {
            let i = n - 1 - ib;
            f[i][2] = (f[i][2] - f[i][3] * f[i + 1][2] - wk[i] * f[n - 2][2])
                / f[i][1];
        }
        f[n - 1][2] = f[0][2];
    } else {
        for i in (imin + 1)..=imax {
            let t = if i == n - 2 && imax == n - 2 {
                (f[i - 1][3] - f[i][3]) / f[i - 1][1]
            } else if i == 1 {
                elem21 / f[i - 1][1]
            } else if i == n - 1 {
                elemnn1 / f[i - 1][1]
            } else {
                f[i - 1][3] / f[i - 1][1]
            };
            if i == imin + 1 && imin == 1 {
                f[i][1] -= t * (f[i - 1][3] - f[i - 2][3]);
            } else {
                f[i][1] -= t * f[i - 1][3];
            }
            f[i][2] -= t * f[i - 1][2];
        }
        f[imax][2] /= f[imax][1];
        for ib in 1..=(imax - imin) {
            let i = imax - ib;
            if i == 1 && imin == 1 {
                f[i][2] = (f[i][2]
                    - (f[i][3] - f[i - 1][3]) * f[i + 1][2])
                    / f[i][1];
            } else {
                f[i][2] = (f[i][2] - f[i][3] * f[i + 1][2]) / f[i][1];
            }
        }
        f[0][3] = x[1] - x[0];
        f[n - 2][3] = x[n - 1] - x[n - 2];
        if !(1..=7).contains(&k_bc1) {
            f[0][2] = (f[1][2] * (f[0][3] + f[1][3]) - f[2][2] * f[0][3])
                / f[1][3];
        }
        if !(1..=7).contains(&k_bcn) {
            f[n - 1][2] = f[n - 2][2]
                + (f[n - 2][2] - f[n - 3][2]) * f[n - 2][3] / f[n - 3][3];
        }
    }

    for i in 0..(n - 1) {
        let h_i = f[i][3];
        let sig_i = f[i][2];
        let sig_next = f[i + 1][2];
        f[i][1] = (f[i + 1][0] - f[i][0]) / h_i - h_i * (sig_next + 2.0 * sig_i);
        f[i][3] = 6.0 * (sig_next - sig_i) / h_i;
        f[i][2] = 6.0 * sig_i;
    }

    if k_bc1 == -1 {
        f[n - 1][1] = f[0][1];
        f[n - 1][2] = f[0][2];
        f[n - 1][3] = f[0][3];
    } else {
        let hn = x[n - 1] - x[n - 2];
        f[n - 1][1] =
            f[n - 2][1] + hn * (f[n - 2][2] + 0.5 * hn * f[n - 2][3]);
        f[n - 1][2] = f[n - 2][2] + hn * f[n - 2][3];
        f[n - 1][3] = f[n - 2][3];
        if k_bcn == 1 || k_bcn == 3 || k_bcn == 5 {
            f[n - 1][1] = an;
        } else if k_bcn == 2 || k_bcn == 4 || k_bcn == 6 {
            f[n - 1][2] = bn;
        }
    }
}
