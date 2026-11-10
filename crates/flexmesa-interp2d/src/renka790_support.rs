// SPDX-License-Identifier: LGPL-3.0-or-later
//! Spatial cell-grid and QR least-squares helpers for ACM TOMS 790 (`CSHEP2D`).
//!
//! Direct port of `STORE2_db`, `GETNP2_db`, `SETUP2_db`, `GIVENS_db`, and
//! `ROTATE_db` from upstream MESA `interp_2d/private/renka790_db.f`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]

use crate::Interp2dError;

pub(crate) fn store2(
    x: &[f64],
    y: &[f64],
    nr: usize,
) -> Result<(Vec<usize>, Vec<isize>, f64, f64, f64, f64), Interp2dError> {
    let n = x.len();
    if n < 2 || nr < 1 {
        return Err(Interp2dError::InsufficientPoints {
            axis: "n/nr",
            min_points: 2,
            actual: n,
        });
    }
    let mut xmn = x[0];
    let mut xmx = x[0];
    let mut ymn = y[0];
    let mut ymx = y[0];
    for k in 1..n {
        xmn = xmn.min(x[k]);
        xmx = xmx.max(x[k]);
        ymn = ymn.min(y[k]);
        ymx = ymx.max(y[k]);
    }
    let delx = (xmx - xmn) / (nr as f64);
    let dely = (ymx - ymn) / (nr as f64);
    if delx == 0.0 || dely == 0.0 {
        return Err(Interp2dError::DuplicateNodes);
    }
    let mut lcell = vec![0_usize; nr * nr];
    let mut lnext = vec![0_isize; n];
    for k in (1..=n).rev() {
        let i = (((x[k - 1] - xmn) / delx) as usize + 1).min(nr);
        let j = (((y[k - 1] - ymn) / dely) as usize + 1).min(nr);
        let idx = (j - 1) * nr + (i - 1);
        let l = lcell[idx];
        lnext[k - 1] = if l != 0 { l as isize } else { k as isize };
        lcell[idx] = k;
    }
    Ok((lcell, lnext, xmn, ymn, delx, dely))
}

pub(crate) fn getnp2(
    px: f64,
    py: f64,
    x: &[f64],
    y: &[f64],
    nr: usize,
    lcell: &[usize],
    lnext: &mut [isize],
    xmin: f64,
    ymin: f64,
    dx: f64,
    dy: f64,
) -> Option<(usize, f64)> {
    if nr < 1 || dx <= 0.0 || dy <= 0.0 {
        return None;
    }
    let nr_i = nr as isize;
    let mut first = true;
    let mut imin = 1_isize;
    let mut imax = nr_i;
    let mut jmin = 1_isize;
    let mut jmax = nr_i;
    let delx = px - xmin;
    let dely = py - ymin;
    let i0 = ((delx / dx) as isize + 1).clamp(1, nr_i);
    let j0 = ((dely / dy) as isize + 1).clamp(1, nr_i);
    let mut i1 = i0;
    let mut i2 = i0;
    let mut j1 = j0;
    let mut j2 = j0;
    let mut lmin = 0_usize;
    let mut rsmin = 0.0_f64;

    loop {
        'j_loop: for j in j1..=j2 {
            if j > jmax {
                break 'j_loop;
            }
            if j < jmin {
                continue 'j_loop;
            }
            'i_loop: for i in i1..=i2 {
                if i > imax {
                    continue 'j_loop;
                }
                if i < imin {
                    continue 'i_loop;
                }
                if j != j1 && j != j2 && i != i1 && i != i2 {
                    continue 'i_loop;
                }
                let mut l = lcell[(j as usize - 1) * nr + (i as usize - 1)];
                if l == 0 {
                    continue 'i_loop;
                }
                loop {
                    let ln = lnext[l - 1];
                    if ln >= 0 {
                        let rsq =
                            (x[l - 1] - px).powi(2) + (y[l - 1] - py).powi(2);
                        if first {
                            lmin = l;
                            rsmin = rsq;
                            let r = rsmin.sqrt();
                            imin = (((delx - r) / dx) as isize + 1).max(1);
                            imax = (((delx + r) / dx) as isize + 1).min(nr_i);
                            jmin = (((dely - r) / dy) as isize + 1).max(1);
                            jmax = (((dely + r) / dy) as isize + 1).min(nr_i);
                            first = false;
                        } else if rsq < rsmin {
                            lmin = l;
                            rsmin = rsq;
                        }
                    }
                    let next_l = ln.unsigned_abs();
                    if next_l == l {
                        break;
                    }
                    l = next_l;
                }
            }
        }
        if i1 <= imin && i2 >= imax && j1 <= jmin && j2 >= jmax {
            break;
        }
        i1 -= 1;
        i2 += 1;
        j1 -= 1;
        j2 += 1;
    }

    if first {
        return None;
    }
    lnext[lmin - 1] = -lnext[lmin - 1];
    Some((lmin, rsmin))
}

#[inline]
pub(crate) fn setup2(
    xk: f64,
    yk: f64,
    zk: f64,
    xi: f64,
    yi: f64,
    zi: f64,
    s1: f64,
    s2: f64,
    s3: f64,
    r: f64,
) -> [f64; 10] {
    let dx = xi - xk;
    let dy = yi - yk;
    let dxsq = dx * dx;
    let dysq = dy * dy;
    let d = (dxsq + dysq).sqrt();
    if d <= 0.0 || d >= r {
        return [0.0; 10];
    }
    let w = (r - d) / r / d;
    let w1 = s1 * w;
    let w2 = s2 * w;
    let w3 = s3 * w;
    [
        dxsq * dx * w3,
        dxsq * dy * w3,
        dx * dysq * w3,
        dysq * dy * w3,
        dxsq * w2,
        dx * dy * w2,
        dysq * w2,
        dx * w1,
        dy * w1,
        (zi - zk) * w,
    ]
}

#[inline]
pub(crate) fn givens(a: f64, b: f64) -> (f64, f64, f64, f64) {
    let aa = a;
    let bb = b;
    if aa.abs() > bb.abs() {
        let u = aa + aa;
        let v = bb / u;
        let r = (0.25 + v * v).sqrt() * u;
        let c = aa / r;
        let s = v * (c + c);
        (r, s, c, s)
    } else if bb != 0.0 {
        let u = bb + bb;
        let v = aa / u;
        let r = (0.25 + v * v).sqrt() * u;
        let s = bb / r;
        let c = v * (s + s);
        let b_out = if c != 0.0 { 1.0 / c } else { 1.0 };
        (r, b_out, c, s)
    } else {
        (aa, bb, 1.0, 0.0)
    }
}

#[inline]
pub(crate) fn rotate(c: f64, s: f64, x: &mut [f64], y: &mut [f64]) {
    for (xi, yi) in x.iter_mut().zip(y.iter_mut()) {
        let xk = *xi;
        let yk = *yi;
        *xi = c * xk + s * yk;
        *yi = -s * xk + c * yk;
    }
}
