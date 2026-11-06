//! Four-dimensional $(Z, X, \log Q, \log T)$ 3-stage ionization table
//! evaluator (`ion_tables_eval.f90`).
//!
//! Evaluates the 28 ionization output channels (`logPgas`, `logpp_H..Fe`,
//! `Z_H..Fe`, `fneut_H..Fe`) on an [`IonTableGrid`] in three stages:
//! 1. **Stage 1 ($(\log Q, \log T)$ bicubic spline)**: Evaluates precomputed
//!    not-a-knot bicubic splines (`Do_ion_Interpolations`) with silent boundary
//!    clamping (`Locate_logQ`, `Locate_logT` with $+10^{-4}$ index bias).
//! 2. **Stage 2 ($X$ interpolation)**: 3-point quadratic Lagrange near edges
//!    (`ix_hi - ix_lo == 2`) or 4-point blended quadratic-cubic on interior
//!    intervals (`Get_ion_for_X`).
//! 3. **Stage 3 ($Z$ interpolation)**: 3-point quadratic Lagrange on $Z \in
//!    [0, 0.04]$ (`ion_Zs(1..3)`) or cosine-smoothed `0.5 * (1 - cos(pi *
//!    alfa))` blending on $Z \in (0.04, 1.0]$ (`Get_ion_ZResults`).
//!
//! # Output Channel Ordering (`[T; 28]`)
//! - `0`: `logPgas` ($\log_{10} P_{\mathrm{gas}}$ in $\mathrm{dyn\,cm^{-2}}$).
//! - `1..10`: `logpp_H..Fe` (log partial pressures for H, He, C, N, O, Ne,
//!   Mg, Si, Fe).
//! - `10..19`: `Z_H..Fe` (mean ionic charges for H, He, C, N, O, Ne, Mg,
//!   Si, Fe).
//! - `19..28`: `fneut_H..Fe` (neutral fractions for H, He, C, N, O, Ne,
//!   Mg, Si, Fe).

use flexmesa_const::ONE_SIXTH;
use flexmesa_utils::is_bad;
use rayon::prelude::*;

use crate::fe56_he4::PARALLEL_ZONE_THRESHOLD;
use crate::ion_table_load::{locate_axis, resolve_pair, IonTableGrid};
use crate::{
    IonError, IonScalar, ION_XS, ION_ZS, NUM_ION_VALS, NUM_ION_XS_FOR_Z,
};

/// Precomputed Stage 1 $(\log Q, \log T)$ bicubic cell indices and basis
/// weights shared across all $(Z, X)$ tables for a single stellar zone.
#[derive(Clone, Copy)]
struct QtBasis<T: IonScalar> {
    /// Flat base offsets `(idx00, idx01, idx10, idx11)` into each channel's
    /// spline array.
    idx: [usize; 4],
    /// Linear and cubic basis weights `(xp, xpi, cx, cxi)` along $\log Q$.
    qx: [T; 4],
    /// Linear and cubic basis weights `(yp, ypi, cy, cyi)` along $\log_{10} T$.
    ty: [T; 4],
    /// Scaled step factors `(hx^2 / 6, hy^2 / 6, hx^2 hy^2 / 36)`.
    scales: [T; 3],
}

impl IonTableGrid {
    /// Evaluate all 28 ionization channels generically over `T: IonScalar`
    /// (`f64` or `DualOrder1<N>`) at $(Z, X, \log_{10}\rho, \log_{10} T)$.
    ///
    /// Implements `Get_ion_ZResults` in `ion_tables_eval.f90`:
    /// - For $Z < 10^{-6}$, returns the $Z = 0$ slice (`iz = 0`).
    /// - For $Z \in [10^{-6}, 0.04]$, evaluates the 3-point uniform quadratic
    ///   Lagrange interpolant across `iz = 0, 1, 2` ($Z = 0.00, 0.02, 0.04$).
    /// - For $Z \in (0.04, 0.20]$ and $Z \in (0.20, 1.00 - 10^{-6}]$, blends
    ///   adjacent $Z$ slices via cosine weights $\alpha = \tfrac{1}{2}[1 -
    ///   \cos(\pi \Delta Z)]$.
    /// - For $Z > 1.00 - 10^{-6}$, returns the pure-metal slice (`iz = 4`).
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn eval_generic<T: IonScalar>(
        &self,
        z_in: T,
        x_in: T,
        log10_rho: T,
        log10_t: T,
    ) -> [T; NUM_ION_VALS] {
        let zero = T::from_f64(0.0);
        let x = if x_in.val() < 1.0e-20 { zero } else { x_in };
        let z = if z_in.val() < 1.0e-20 { zero } else { z_in };
        let qt = self.prepare_qt_basis(log10_rho, log10_t);
        let tiny = 1.0e-6;

        if z.val() < tiny {
            return self.eval_for_x(0, x, &qt);
        }
        if z.val() > ION_ZS[2] {
            if z.val() <= ION_ZS[3] {
                return self.interp2_z(2, 3, z, x, &qt);
            } else if z.val() <= ION_ZS[4] - tiny {
                return self.interp2_z(3, 4, z, x, &qt);
            }
            return self.eval_for_x(4, x, &qt);
        }

        let r0 = self.eval_for_x(0, x, &qt);
        let r1 = self.eval_for_x(1, x, &qt);
        let r2 = self.eval_for_x(2, x, &qt);
        quad_3pt(T::from_f64(ION_ZS[1] - ION_ZS[0]), z, &r0, &r1, &r2)
    }

    /// Cosine-smoothed 2-point interpolation in $Z$ (`do_interp2`).
    ///
    /// Computes the $C^1$-smooth transition weight
    /// $\alpha = \tfrac{1}{2}[1 - \cos(\pi (Z - Z_1) / (Z_2 - Z_1))]$ and
    /// blends $(1 - \alpha) \mathbf{r}(Z_1) + \alpha \mathbf{r}(Z_2)$.
    fn interp2_z<T: IonScalar>(
        &self,
        iz1: usize,
        iz2: usize,
        z: T,
        x: T,
        qt: &QtBasis<T>,
    ) -> [T; NUM_ION_VALS] {
        let z1 = T::from_f64(ION_ZS[iz1]);
        let z2 = T::from_f64(ION_ZS[iz2]);
        let one = T::from_f64(1.0);
        let half = T::from_f64(0.5);
        let raw_alfa = (z - z1) / (z2 - z1);
        let alfa = (one - raw_alfa.cospi()) * half;
        let beta = one - alfa;
        let r1 = self.eval_for_x(iz1, x, qt);
        let r2 = self.eval_for_x(iz2, x, qt);
        let mut out = [T::from_f64(0.0); NUM_ION_VALS];
        for k in 0..NUM_ION_VALS {
            out[k] = beta * r1[k] + alfa * r2[k];
        }
        out
    }

    /// Stage 2 interpolation in $X$ at fixed `iz` (`Get_ion_for_X`).
    ///
    /// Uses a 3-point uniform quadratic Lagrange polynomial near the low-$X$
    /// (`X <= 0.2`) and high-$X$ (`X >= X[num_xs - 2]`) boundaries, and a
    /// 4-point blended quadratic-cubic stencil across interior intervals.
    fn eval_for_x<T: IonScalar>(
        &self,
        iz: usize,
        x: T,
        qt: &QtBasis<T>,
    ) -> [T; NUM_ION_VALS] {
        let num_xs = NUM_ION_XS_FOR_Z[iz];
        if x.val() < 1.0e-6 || num_xs == 1 {
            return self.eval_xtable(0, iz, qt);
        }
        let (ix_lo, ix_hi) = if x.val() <= ION_XS[1] {
            (0, 2)
        } else if x.val() >= ION_XS[num_xs - 2] {
            (num_xs - 3, num_xs - 1)
        } else {
            let mut bracket = (0, 3);
            for (ix, &x_grid) in
                ION_XS.iter().enumerate().take(num_xs - 1).skip(2)
            {
                if x.val() <= x_grid {
                    bracket = (ix - 2, ix + 1);
                    break;
                }
            }
            bracket
        };

        let dx = T::from_f64(ION_XS[1] - ION_XS[0]);
        if ix_hi - ix_lo == 2 {
            let r0 = self.eval_xtable(ix_lo, iz, qt);
            let r1 = self.eval_xtable(ix_lo + 1, iz, qt);
            let r2 = self.eval_xtable(ix_lo + 2, iz, qt);
            return quad_3pt(dx, x - T::from_f64(ION_XS[ix_lo]), &r0, &r1, &r2);
        }
        let r0 = self.eval_xtable(ix_lo, iz, qt);
        let r1 = self.eval_xtable(ix_lo + 1, iz, qt);
        let r2 = self.eval_xtable(ix_lo + 2, iz, qt);
        let r3 = self.eval_xtable(ix_lo + 3, iz, qt);
        let coef = (x - T::from_f64(ION_XS[ix_lo + 1])) / dx;
        let zero = T::from_f64(0.0);
        let one = T::from_f64(1.0);
        let two = T::from_f64(2.0);
        let three = T::from_f64(3.0);
        let four = T::from_f64(4.0);
        let five = T::from_f64(5.0);
        let c1 = (zero - coef) * (coef - one) * (coef - one) / two;
        let c2 = (two + coef * coef * (three * coef - five)) / two;
        let c3 = (coef + coef * coef * (four - three * coef)) / two;
        let c4 = coef * coef * (coef - one) / two;
        let mut out = [zero; NUM_ION_VALS];
        for k in 0..NUM_ION_VALS {
            out[k] = c1 * r0[k] + c2 * r1[k] + c3 * r2[k] + c4 * r3[k];
        }
        out
    }

    /// Precompute the $(\log Q, \log T)$ cell indices and cubic basis weights.
    fn prepare_qt_basis<T: IonScalar>(
        &self,
        log10_rho: T,
        log10_t: T,
    ) -> QtBasis<T> {
        let raw_q = log10_rho - T::from_f64(2.0) * log10_t + T::from_f64(12.0);
        let (iq, q0, q1, qc) =
            locate_axis(raw_q, self.log_q_min, self.del_log_q, self.num_log_qs);
        let (jt, t0, t1, tc) = locate_axis(
            log10_t,
            self.log_t_min,
            self.del_log_t,
            self.num_log_ts,
        );
        let (hx, hy, one) = (q1 - q0, t1 - t0, T::from_f64(1.0));
        let (hx2, hy2) = (hx * hx, hy * hy);
        let xp = (qc - T::from_f64(q0)) * T::from_f64(1.0 / hx);
        let xpi = one - xp;
        let yp = (tc - T::from_f64(t0)) * T::from_f64(1.0 / hy);
        let ypi = one - yp;
        let nt = self.num_log_ts;
        QtBasis {
            idx: [
                (iq * nt + jt) * 4,
                (iq * nt + jt + 1) * 4,
                ((iq + 1) * nt + jt) * 4,
                ((iq + 1) * nt + jt + 1) * 4,
            ],
            qx: [xp, xpi, xp * (xp * xp - one), xpi * (xpi * xpi - one)],
            ty: [yp, ypi, yp * (yp * yp - one), ypi * (ypi * ypi - one)],
            scales: [
                T::from_f64(hx2 * ONE_SIXTH),
                T::from_f64(hy2 * ONE_SIXTH),
                T::from_f64(hx2 * hy2 * (ONE_SIXTH * ONE_SIXTH)),
            ],
        }
    }

    /// Stage 1 bicubic spline evaluation in $(\log Q, \log T)$ at fixed `(ix,
    /// iz)` using precomputed [`QtBasis`].
    fn eval_xtable<T: IonScalar>(
        &self,
        ix: usize,
        iz: usize,
        qt: &QtBasis<T>,
    ) -> [T; NUM_ION_VALS] {
        let [idx00, idx01, idx10, idx11] = qt.idx;
        let [xp, xpi, cx, cxi] = qt.qx;
        let [yp, ypi, cy, cyi] = qt.ty;
        let [sixth_hx2, sixth_hy2, z36th] = qt.scales;
        let tbl_zx = &self.coeffs[iz][ix];
        let mut out = [T::from_f64(0.0); NUM_ION_VALS];

        for k in 0..NUM_ION_VALS {
            let fin = &tbl_zx[k];
            let f00 = |c: usize| T::from_f64(fin[idx00 + c]);
            let f01 = |c: usize| T::from_f64(fin[idx01 + c]);
            let f10 = |c: usize| T::from_f64(fin[idx10 + c]);
            let f11 = |c: usize| T::from_f64(fin[idx11 + c]);
            out[k] = xpi * (ypi * f00(0) + yp * f01(0))
                + xp * (ypi * f10(0) + yp * f11(0))
                + sixth_hx2
                    * (cxi * (ypi * f00(1) + yp * f01(1))
                        + cx * (ypi * f10(1) + yp * f11(1)))
                + sixth_hy2
                    * (xpi * (cyi * f00(2) + cy * f01(2))
                        + xp * (cyi * f10(2) + cy * f11(2)))
                + z36th
                    * (cxi * (cyi * f00(3) + cy * f01(3))
                        + cx * (cyi * f10(3) + cy * f11(3)));
        }
        out
    }

    /// Evaluate `eval_ionization` at $(Z, X, \rho, \log_{10}\rho, T,
    /// \log_{10} T)$, matching `Get_ion_Results` in `ion_tables_eval.f90`.
    ///
    /// # Errors
    /// Returns [`IonError::InvalidInput`] if inputs are non-finite or
    /// non-positive.
    pub fn eval(
        &self,
        z: f64,
        x: f64,
        rho: Option<f64>,
        log10_rho: Option<f64>,
        t: Option<f64>,
        log10_t: Option<f64>,
    ) -> Result<[f64; NUM_ION_VALS], IonError> {
        if is_bad(z) || is_bad(x) {
            return Err(IonError::InvalidInput {
                context: "eval_ionization",
                detail: "Z and X must be finite".to_string(),
            });
        }
        let (t_val, lt_val) = resolve_pair(t, log10_t, "T")?;
        let (rho_val, lrho_val) = resolve_pair(rho, log10_rho, "Rho")?;
        if t_val <= 0.0 || rho_val <= 0.0 || is_bad(t_val) || is_bad(rho_val) {
            return Err(IonError::InvalidInput {
                context: "eval_ionization",
                detail: "Rho and T must be positive and finite".to_string(),
            });
        }
        Ok(self.eval_generic(z, x, lrho_val, lt_val))
    }

    /// Evaluate `eval_ionization` across `n_zones` stellar zones, parallelizing
    /// over `rayon` when `n_zones >= 64`.
    ///
    /// # Errors
    /// Returns [`IonError::InvalidInput`] if input slice lengths differ.
    pub fn eval_zones(
        &self,
        z: &[f64],
        x: &[f64],
        log10_rho: &[f64],
        log10_t: &[f64],
    ) -> Result<Vec<[f64; NUM_ION_VALS]>, IonError> {
        let n = z.len();
        if x.len() != n || log10_rho.len() != n || log10_t.len() != n {
            return Err(IonError::InvalidInput {
                context: "IonTableGrid::eval_zones",
                detail: "z, x, log10_rho, and log10_t must have equal length"
                    .to_string(),
            });
        }
        let eval_k =
            |k: usize| self.eval_generic(z[k], x[k], log10_rho[k], log10_t[k]);
        if n >= PARALLEL_ZONE_THRESHOLD {
            Ok((0..n).into_par_iter().map(eval_k).collect())
        } else {
            Ok((0..n).map(eval_k).collect())
        }
    }
}

/// Evaluate `eval_ionization` using [`IonTableGrid::builtin`].
///
/// # Errors
/// Returns [`IonError::InvalidInput`] if inputs are invalid.
pub fn eval_ionization(
    z: f64,
    x: f64,
    rho: Option<f64>,
    log10_rho: Option<f64>,
    t: Option<f64>,
    log10_t: Option<f64>,
) -> Result<[f64; NUM_ION_VALS], IonError> {
    IonTableGrid::builtin().eval(z, x, rho, log10_rho, t, log10_t)
}

/// Evaluate a 3-point uniform-grid quadratic Lagrange polynomial across 28
/// channels.
fn quad_3pt<T: IonScalar>(
    dh: T,
    delta: T,
    r0: &[T; NUM_ION_VALS],
    r1: &[T; NUM_ION_VALS],
    r2: &[T; NUM_ION_VALS],
) -> [T; NUM_ION_VALS] {
    let two = T::from_f64(2.0);
    let three = T::from_f64(3.0);
    let denom = two * dh * dh;
    let c1 = (two * dh * dh - three * dh * delta + delta * delta) / denom;
    let c2 = two * (two * dh - delta) * delta / denom;
    let c3 = delta * (delta - dh) / denom;
    let mut out = [T::from_f64(0.0); NUM_ION_VALS];
    for k in 0..NUM_ION_VALS {
        out[k] = c1 * r0[k] + c2 * r1[k] + c3 * r2[k];
    }
    out
}
