// SPDX-License-Identifier: LGPL-3.0-or-later
//! Scattered 2D cubic Shepard interpolation (ACM TOMS Algorithm 790).
//!
//! Direct port of Robert J. Renka's `CSHEP2D` package (`renka790_db.f` in
//! upstream MESA `interp_2d/private/renka790_db.f`):
//! - [`Cshep2Interpolant::fit`] (`interp_cshep2_db` / `do_cshep2_db`)
//! - [`Cshep2Interpolant::eval`] (`interp_cs2val_db` / `do_cs2val_db`)
//! - [`Cshep2Interpolant::eval_grad`] (`interp_cs2grd_db` / `do_cs2grd_db`)
//! - [`Cshep2Interpolant::eval_hess`] (`cs2hes_db`)

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

use crate::renka790_support::{getnp2, givens, rotate, setup2, store2};
use crate::Interp2dError;

const DTOL: f64 = 0.01;
const RTOL: f64 = 1.0e-5;

/// Scattered 2D cubic Shepard interpolant (ACM TOMS Algorithm 790).
#[derive(Clone, Debug)]
pub struct Cshep2Interpolant {
    n: usize,
    x: Vec<f64>,
    y: Vec<f64>,
    f: Vec<f64>,
    nr: usize,
    lcell: Vec<usize>,
    lnext: Vec<isize>,
    xmin: f64,
    ymin: f64,
    dx: f64,
    dy: f64,
    rmax: f64,
    rw: Vec<f64>,
    a: Vec<[f64; 9]>,
}

/// Alias for [`Cshep2Interpolant`] for callers referencing `Qshep2Interpolant`.
pub type Qshep2Interpolant = Cshep2Interpolant;

impl Cshep2Interpolant {
    /// Fit a scattered 2D cubic Shepard interpolant (`interp_cshep2_db`).
    ///
    /// # Errors
    /// Returns [`Interp2dError`] if `n < 10`, `nc < 9`, `nw < 1`, `nr < 1`, or
    /// if all nodes are collinear or duplicate.
    pub fn fit(
        x: &[f64],
        y: &[f64],
        f: &[f64],
        nc: usize,
        nw: usize,
        nr: usize,
    ) -> Result<Self, Interp2dError> {
        let n = x.len();
        let lmax = 40.min(n.saturating_sub(1));
        let ncwmax = nc.max(nw);
        if n < 10
            || y.len() != n
            || f.len() != n
            || nc < 9
            || nw < 1
            || ncwmax > lmax
            || nr < 1
        {
            return Err(Interp2dError::InsufficientPoints {
                axis: "cshep2(n,nc,nw,nr)",
                min_points: 10,
                actual: n,
            });
        }
        let (lcell, mut lnext, xmin, ymin, dx, dy) = store2(x, y, nr)?;
        let mut npts = vec![0_usize; lmax];
        let mut rw = vec![0.0_f64; n];
        let mut a = vec![[0.0_f64; 9]; n];
        let mut rsmx = 0.0_f64;

        for k in 1..=n {
            let xk = x[k - 1];
            let yk = y[k - 1];
            let fk = f[k - 1];
            lnext[k - 1] = -lnext[k - 1];

            let mut rs = 0.0_f64;
            let mut sum = 0.0_f64;
            let mut rws = 0.0_f64;
            let mut rc = 0.0_f64;
            let mut lnp = 0_usize;
            let mut neq = 0_usize;
            let mut sfs = 0.0_f64;

            loop {
                sum += rs;
                if lnp == lmax {
                    if rws == 0.0 {
                        rws = 1.1 * rs;
                    }
                    if rc == 0.0 {
                        neq = lmax;
                        rc = (1.1 * rs).sqrt();
                        sfs = (neq as f64) / sum;
                    }
                    break;
                }
                lnp += 1;
                let rsold = rs;
                let (np, dsq) = getnp2(
                    xk, yk, x, y, nr, &lcell, &mut lnext, xmin, ymin, dx, dy,
                )
                .ok_or(Interp2dError::DuplicateNodes)?;
                rs = dsq;
                if rs == 0.0 {
                    return Err(Interp2dError::DuplicateNodes);
                }
                npts[lnp - 1] = np;
                if (rs - rsold) / rs < RTOL {
                    continue;
                }
                if rws == 0.0 && lnp > nw {
                    rws = rs;
                }
                if rc == 0.0 && lnp > nc {
                    neq = lnp - 1;
                    rc = rs.sqrt();
                    sfs = (neq as f64) / sum;
                }
                if lnp > ncwmax {
                    break;
                }
            }

            rw[k - 1] = rws.sqrt();
            if rws > rsmx {
                rsmx = rws;
            }
            let sf = sfs.sqrt();
            let sfc = sf * sfs;

            let mut b = [[0.0_f64; 10]; 10];
            let mut i_eq = 0_usize;

            'qr_loop: loop {
                while i_eq < neq {
                    i_eq += 1;
                    let np = npts[i_eq - 1];
                    let irow = i_eq.min(10) - 1;
                    b[irow] = setup2(
                        xk,
                        yk,
                        fk,
                        x[np - 1],
                        y[np - 1],
                        f[np - 1],
                        sf,
                        sfs,
                        sfc,
                        rc,
                    );
                    if i_eq == 1 {
                        continue;
                    }
                    for j in 0..irow {
                        let (new_bj, new_bi, c, s) =
                            givens(b[j][j], b[irow][j]);
                        b[j][j] = new_bj;
                        b[irow][j] = new_bi;
                        let (left, right) = b.split_at_mut(irow);
                        let row_j = &mut left[j][(j + 1)..10];
                        let row_i = &mut right[0][(j + 1)..10];
                        rotate(c, s, row_j, row_i);
                    }
                }

                let mut dmin = b[0][0].abs();
                for d_idx in 1..9 {
                    dmin = dmin.min(b[d_idx][d_idx].abs());
                }
                if dmin * rc >= DTOL {
                    break 'qr_loop;
                }
                if neq == lmax {
                    // Damping third partials (Label 7)
                    let stf = 1.0 / rc;
                    for i_d in 0..4 {
                        b[9] = [0.0; 10];
                        b[9][i_d] = stf;
                        for j in i_d..9 {
                            let (new_bj, new_b9, c, s) =
                                givens(b[j][j], b[9][j]);
                            b[j][j] = new_bj;
                            b[9][j] = new_b9;
                            let (left, right) = b.split_at_mut(9);
                            let row_j = &mut left[j][(j + 1)..10];
                            let row_9 = &mut right[0][(j + 1)..10];
                            rotate(c, s, row_j, row_9);
                        }
                    }
                    let mut dmin_damped = b[4][4].abs();
                    for d_idx in 5..9 {
                        dmin_damped = dmin_damped.min(b[d_idx][d_idx].abs());
                    }
                    if dmin_damped * rc < DTOL {
                        return Err(Interp2dError::DuplicateNodes);
                    }
                    break 'qr_loop;
                }

                // Label 6: increase RC and NEQ
                loop {
                    let rsold = rs;
                    neq += 1;
                    if neq == lmax {
                        rc = (1.1 * rs).sqrt();
                        break;
                    }
                    if neq < lnp {
                        let np = npts[neq];
                        rs =
                            (x[np - 1] - xk).powi(2) + (y[np - 1] - yk).powi(2);
                        if (rs - rsold) / rs < RTOL {
                            continue;
                        }
                        rc = rs.sqrt();
                        break;
                    }
                    lnp += 1;
                    let (np, dsq) = getnp2(
                        xk, yk, x, y, nr, &lcell, &mut lnext, xmin, ymin, dx,
                        dy,
                    )
                    .ok_or(Interp2dError::DuplicateNodes)?;
                    rs = dsq;
                    npts[lnp - 1] = np;
                    if (rs - rsold) / rs < RTOL {
                        continue;
                    }
                    rc = rs.sqrt();
                    break;
                }
            }

            let mut sol = [0.0_f64; 9];
            for i in (0..9).rev() {
                let mut t = 0.0_f64;
                for j in (i + 1)..9 {
                    t += b[i][j] * sol[j];
                }
                sol[i] = (b[i][9] - t) / b[i][i];
            }
            for i in 0..4 {
                sol[i] *= sfc;
            }
            for i in 4..7 {
                sol[i] *= sfs;
            }
            sol[7] *= sf;
            sol[8] *= sf;
            a[k - 1] = sol;

            lnext[k - 1] = -lnext[k - 1];
            for &np in &npts[..lnp] {
                lnext[np - 1] = -lnext[np - 1];
            }
        }

        Ok(Self {
            n,
            x: x.to_vec(),
            y: y.to_vec(),
            f: f.to_vec(),
            nr,
            lcell,
            lnext,
            xmin,
            ymin,
            dx,
            dy,
            rmax: rsmx.sqrt(),
            rw,
            a,
        })
    }

    /// Evaluate the cubic Shepard interpolant value $C(px, py)$ (`interp_cs2val_db`).
    #[must_use]
    pub fn eval(&self, px: f64, py: f64) -> f64 {
        self.eval_hess(px, py).0
    }

    /// Evaluate $(C, \partial C / \partial x, \partial C / \partial y)$ (`interp_cs2grd_db`).
    #[must_use]
    pub fn eval_grad(&self, px: f64, py: f64) -> (f64, f64, f64) {
        let (c, cx, cy, _, _, _) = self.eval_hess(px, py);
        (c, cx, cy)
    }

    /// Evaluate $(C, C_x, C_y, C_{xx}, C_{xy}, C_{yy})$ (`cs2hes_db`).
    #[must_use]
    pub fn eval_hess(
        &self,
        px: f64,
        py: f64,
    ) -> (f64, f64, f64, f64, f64, f64) {
        let nr_i = self.nr as isize;
        let imin =
            (((px - self.xmin - self.rmax) / self.dx) as isize + 1).max(1);
        let imax =
            (((px - self.xmin + self.rmax) / self.dx) as isize + 1).min(nr_i);
        let jmin =
            (((py - self.ymin - self.rmax) / self.dy) as isize + 1).max(1);
        let jmax =
            (((py - self.ymin + self.rmax) / self.dy) as isize + 1).min(nr_i);

        let mut sw = 0.0_f64;
        let mut swc = 0.0_f64;
        let mut swx = 0.0_f64;
        let mut swy = 0.0_f64;
        let mut swcx = 0.0_f64;
        let mut swcy = 0.0_f64;
        let mut swxx = 0.0_f64;
        let mut swxy = 0.0_f64;
        let mut swyy = 0.0_f64;
        let mut swcxx = 0.0_f64;
        let mut swcxy = 0.0_f64;
        let mut swcyy = 0.0_f64;

        for j in jmin..=jmax {
            for i in imin..=imax {
                let mut k =
                    self.lcell[(j as usize - 1) * self.nr + (i as usize - 1)];
                if k == 0 {
                    continue;
                }
                loop {
                    let delx = px - self.x[k - 1];
                    let dely = py - self.y[k - 1];
                    let dxsq = delx * delx;
                    let dysq = dely * dely;
                    let d = (dxsq + dysq).sqrt();
                    let r = self.rw[k - 1];
                    if d < r {
                        let ak = &self.a[k - 1];
                        if d == 0.0 {
                            return (
                                self.f[k - 1],
                                ak[7],
                                ak[8],
                                2.0 * ak[4],
                                ak[5],
                                2.0 * ak[6],
                            );
                        }
                        let t1 = 1.0 / d - 1.0 / r;
                        let w = t1 * t1 * t1;
                        let t2 = -3.0 * t1 * t1 / (d * d * d);
                        let wx = delx * t2;
                        let wy = dely * t2;
                        let t3 = 3.0 * t1 * (2.0 + 3.0 * d * t1) / d.powi(6);
                        let wxx = t3 * dxsq + t2;
                        let wxy = t3 * delx * dely;
                        let wyy = t3 * dysq + t2;

                        let t1_c = ak[0] * delx + ak[1] * dely + ak[4];
                        let t2_c = t1_c + t1_c + ak[0] * delx;
                        let t3_c = ak[3] * dely + ak[2] * delx + ak[6];
                        let t4_c = t3_c + t3_c + ak[3] * dely;
                        let ck = (t1_c * delx + ak[5] * dely + ak[7]) * delx
                            + (t3_c * dely + ak[8]) * dely
                            + self.f[k - 1];
                        let ckx =
                            t2_c * delx + (ak[2] * dely + ak[5]) * dely + ak[7];
                        let cky =
                            t4_c * dely + (ak[1] * delx + ak[5]) * delx + ak[8];
                        let ckxx = t2_c + 3.0 * ak[0] * delx;
                        let ckxy = 2.0 * (ak[1] * delx + ak[2] * dely) + ak[5];
                        let ckyy = t4_c + 3.0 * ak[3] * dely;

                        sw += w;
                        swc += w * ck;
                        swx += wx;
                        swy += wy;
                        swcx += wx * ck + w * ckx;
                        swcy += wy * ck + w * cky;
                        swxx += wxx;
                        swxy += wxy;
                        swyy += wyy;
                        swcxx += w * ckxx + 2.0 * wx * ckx + ck * wxx;
                        swcxy += w * ckxy + wx * cky + wy * ckx + ck * wxy;
                        swcyy += w * ckyy + 2.0 * wy * cky + ck * wyy;
                    }
                    let next_k = self.lnext[k - 1].unsigned_abs();
                    if next_k == k {
                        break;
                    }
                    k = next_k;
                }
            }
        }

        if sw <= 0.0 {
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
        let c = swc / sw;
        let cx = (swcx - swx * c) / sw;
        let cy = (swcy - swy * c) / sw;
        let cxx = (swcxx - 2.0 * swx * cx - c * swxx) / sw;
        let cxy = (swcxy - swx * cy - swy * cx - c * swxy) / sw;
        let cyy = (swcyy - 2.0 * swy * cy - c * swyy) / sw;
        (c, cx, cy, cxx, cxy, cyy)
    }

    /// Number of scattered nodes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.n
    }

    /// Whether the interpolant has zero nodes.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.n == 0
    }
}

/// Construct a scattered 2D cubic Shepard interpolant (`interp_cshep2_db`).
///
/// # Errors
/// Returns [`Interp2dError`] if input dimensions or nodal geometry are invalid.
pub fn interp_cshep2_db(
    x: &[f64],
    y: &[f64],
    f: &[f64],
    nc: usize,
    nw: usize,
    nr: usize,
) -> Result<Cshep2Interpolant, Interp2dError> {
    Cshep2Interpolant::fit(x, y, f, nc, nw, nr)
}

/// Evaluate a scattered 2D cubic Shepard interpolant (`interp_cs2val_db`).
#[must_use]
pub fn interp_cs2val_db(interp: &Cshep2Interpolant, px: f64, py: f64) -> f64 {
    interp.eval(px, py)
}

/// Evaluate value and gradient of a scattered 2D cubic Shepard interpolant (`interp_cs2grd_db`).
#[must_use]
pub fn interp_cs2grd_db(
    interp: &Cshep2Interpolant,
    px: f64,
    py: f64,
) -> (f64, f64, f64) {
    interp.eval_grad(px, py)
}

/// Alias for [`interp_cshep2_db`].
///
/// # Errors
/// Returns [`Interp2dError`] if input dimensions or nodal geometry are invalid.
pub fn qshep2(
    x: &[f64],
    y: &[f64],
    f: &[f64],
    nc: usize,
    nw: usize,
    nr: usize,
) -> Result<Cshep2Interpolant, Interp2dError> {
    Cshep2Interpolant::fit(x, y, f, nc, nw, nr)
}

/// Alias for [`interp_cs2val_db`].
#[must_use]
pub fn qs2val(interp: &Cshep2Interpolant, px: f64, py: f64) -> f64 {
    interp.eval(px, py)
}

/// Alias for [`interp_cs2grd_db`].
#[must_use]
pub fn qs2grd(interp: &Cshep2Interpolant, px: f64, py: f64) -> (f64, f64, f64) {
    interp.eval_grad(px, py)
}
