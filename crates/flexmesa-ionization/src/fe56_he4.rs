//! Effective ionic charge of ${}^{56}\mathrm{Fe}$ in ${}^4\mathrm{He}$
//! (`eval_charge_of_Fe56_in_He4` in `mod_ionization.f90`).
//!
//! Evaluates a 2D not-a-knot bicubic spline (`interp_mkbicub_db` /
//! `interp_evbicub_db`) over the $105 \times 30$ Thomas-Fermi-Dirac-Hubbard
//! $(\log_{10} n_e, \log_{10} T)$ grid:
//! - $\log_{10} n_e \in [24.0, 31.0]$ (`105` points).
//! - $\log_{10} T \in [6.0, 8.0]$ (`30` points).
//!
//! Provides both the built-in singleton table ([`Fe56He4Table::builtin`]) and
//! an external ASCII loader ([`Fe56He4Table::from_files`]), along with
//! forward-mode [`DualOrder1<N>`] evaluation and `rayon`-parallel multi-zone
//! evaluation.

use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use flexmesa_autodiff::DualOrder1;
use flexmesa_interp2d::{
    herm2xy_prevalidated_db, interp_evbicub_prevalidated_db, interp_mkbicub_db,
    BicubicBc, BicubicEval,
};
use flexmesa_math::{numerics_mode, str_to_vector, NumericsMode};
use rayon::prelude::*;

use crate::fe56_he4_data_a::{
    LOG_NE_FE56_HE4, LOG_T_FE56_HE4, NUM_LOG_NE_FE56_HE4, NUM_LOG_T_FE56_HE4,
    Z_FE56_HE4_A,
};
use crate::fe56_he4_data_b::Z_FE56_HE4_B;
use crate::fe56_he4_data_c::Z_FE56_HE4_C;
use crate::IonError;

/// Minimum zone count for switching from sequential to `rayon` parallel loop.
pub const PARALLEL_ZONE_THRESHOLD: usize = 64;

/// Precomputed 2D bicubic spline interpolant for $Z(\mathrm{Fe56})$ in
/// ${}^4\mathrm{He}$.
#[derive(Debug, Clone)]
pub struct Fe56He4Table {
    /// Strictly ascending $\log_{10} n_e$ grid (`105` points).
    pub log_ne: Vec<f64>,
    /// Strictly ascending $\log_{10} T$ grid (`30` points).
    pub log_t: Vec<f64>,
    /// Precomputed bicubic spline coefficients `[f, f_xx, f_yy, f_xxyy]` of
    /// length `num_log_ne * num_log_t`.
    pub coeffs: Vec<[f64; 4]>,
    /// Uniform spacing flag along $\log_{10} n_e$.
    pub ilinx: bool,
    /// Uniform spacing flag along $\log_{10} T$.
    pub iliny: bool,
}

impl Fe56He4Table {
    /// Return a reference to the lazily initialized built-in $105 \times 30$
    /// [`Fe56He4Table`].
    ///
    /// # Panics
    /// Never panics for the validated built-in table data.
    #[must_use]
    pub fn builtin() -> &'static Self {
        static BUILTIN: OnceLock<Fe56He4Table> = OnceLock::new();
        BUILTIN.get_or_init(|| {
            let mut z_flat =
                Vec::with_capacity(NUM_LOG_NE_FE56_HE4 * NUM_LOG_T_FE56_HE4);
            z_flat.extend_from_slice(&Z_FE56_HE4_A);
            z_flat.extend_from_slice(&Z_FE56_HE4_B);
            z_flat.extend_from_slice(&Z_FE56_HE4_C);
            Self::from_grid(
                LOG_NE_FE56_HE4.to_vec(),
                LOG_T_FE56_HE4.to_vec(),
                &z_flat,
            )
            .expect("built-in Fe56-in-He4 spline table must be valid")
        })
    }

    /// Construct a [`Fe56He4Table`] from explicit grid vectors and row-major
    /// `(num_log_ne, num_log_t)` effective charges `z_flat`.
    ///
    /// # Errors
    /// Returns [`IonError::Interp`] if spline construction fails.
    pub fn from_grid(
        log_ne: Vec<f64>,
        log_t: Vec<f64>,
        z_flat: &[f64],
    ) -> Result<Self, IonError> {
        let (coeffs, ilinx, iliny) = interp_mkbicub_db(
            &log_ne,
            &log_t,
            z_flat,
            BicubicBc::NotAKnot,
            &[],
            BicubicBc::NotAKnot,
            &[],
            BicubicBc::NotAKnot,
            &[],
            BicubicBc::NotAKnot,
            &[],
        )
        .map_err(|e| IonError::Interp(e.to_string()))?;
        Ok(Self {
            log_ne,
            log_t,
            coeffs,
            ilinx,
            iliny,
        })
    }

    /// Load a [`Fe56He4Table`] from the three upstream MESA ASCII files
    /// (`log_ne_fe56_he4.data`, `logT_fe56_he4.data`, `z_fe56_he4.data`).
    ///
    /// # Errors
    /// Returns [`IonError`] if any file cannot be read or parsed.
    pub fn from_files(
        log_ne_path: &Path,
        log_t_path: &Path,
        z_path: &Path,
    ) -> Result<Self, IonError> {
        let log_ne = parse_float_tokens(log_ne_path)?;
        let log_t = parse_float_tokens(log_t_path)?;
        let raw_z = parse_float_tokens(z_path)?;
        let nx = log_ne.len();
        let ny = log_t.len();
        if raw_z.len() != nx * ny {
            return Err(IonError::TableParse {
                path: z_path.display().to_string(),
                detail: format!(
                    "expected {} values ({nx} x {ny}), got {}",
                    nx * ny,
                    raw_z.len()
                ),
            });
        }
        // Upstream z_fe56_he4.data is stored with outer loop i = 1..num_logT
        // and inner loop j = 1..num_log_ne; transpose to row-major (nx, ny).
        let mut z_flat = vec![0.0_f64; nx * ny];
        for iy in 0..ny {
            for ix in 0..nx {
                z_flat[ix * ny + iy] = raw_z[iy * nx + ix];
            }
        }
        Self::from_grid(log_ne, log_t, &z_flat)
    }

    /// Evaluate the full [`BicubicEval`] (value, first partials, and second
    /// partials) at $(\log_{10} n_e, \log_{10} T)$.
    ///
    /// # Errors
    /// Returns [`IonError::Interp`] if coordinates are out of range or `NaN`.
    pub fn eval_full(
        &self,
        log10_ne: f64,
        log10_t: f64,
    ) -> Result<BicubicEval, IonError> {
        let mut ev = interp_evbicub_prevalidated_db(
            log10_ne,
            log10_t,
            &self.log_ne,
            &self.log_t,
            self.ilinx,
            self.iliny,
            &self.coeffs,
        )
        .map_err(|e| IonError::Interp(e.to_string()))?;
        if numerics_mode() == NumericsMode::Strict {
            let (i, j, xparam, yparam, hx, hxi, hy, hyi) =
                herm2xy_prevalidated_db(
                    log10_ne,
                    log10_t,
                    &self.log_ne,
                    &self.log_t,
                    self.ilinx,
                    self.iliny,
                )
                .map_err(|e| IonError::Interp(e.to_string()))?;
            let ny = self.log_t.len();
            let c00 = self.coeffs[i * ny + j];
            let c01 = self.coeffs[i * ny + (j + 1)];
            let c10 = self.coeffs[(i + 1) * ny + j];
            let c11 = self.coeffs[(i + 1) * ny + (j + 1)];
            let (f, df_dx, df_dy) = eval_fvbicub_strict(
                xparam, yparam, hx, hxi, hy, hyi, c00, c01, c10, c11,
            );
            ev.f = f;
            ev.df_dx = df_dx;
            ev.df_dy = df_dy;
        }
        Ok(ev)
    }

    /// Evaluate the effective charge $Z(\mathrm{Fe56})$ at
    /// $(\log_{10} n_e, \log_{10} T)$.
    ///
    /// # Errors
    /// Returns [`IonError::Interp`] if coordinates are out of range or `NaN`.
    pub fn eval(&self, log10_ne: f64, log10_t: f64) -> Result<f64, IonError> {
        Ok(self.eval_full(log10_ne, log10_t)?.f)
    }

    /// Evaluate $Z(\mathrm{Fe56})$ and propagate first-order forward-mode dual
    /// derivatives (`DualOrder1<N>`) via the analytical spline partial
    /// derivatives $(\partial Z / \partial \log_{10} n_e, \partial Z / \partial
    /// \log_{10} T)$.
    ///
    /// # Errors
    /// Returns [`IonError::Interp`] if the primal point is out of range.
    pub fn eval_dual<const N: usize>(
        &self,
        log10_ne: DualOrder1<N>,
        log10_t: DualOrder1<N>,
    ) -> Result<DualOrder1<N>, IonError> {
        let ev = self.eval_full(log10_ne.val, log10_t.val)?;
        let mut d1 = [0.0_f64; N];
        for ((slot, &d_ne), &d_t) in
            d1.iter_mut().zip(&log10_ne.d1).zip(&log10_t.d1)
        {
            *slot = ev.df_dx * d_ne + ev.df_dy * d_t;
        }
        Ok(DualOrder1 { val: ev.f, d1 })
    }

    /// Evaluate `[z, dz_dlog10_ne, dz_dlog10_t]` across `n_zones` zones.
    ///
    /// # Errors
    /// Returns [`IonError`] if `log10_ne.len() != log10_t.len()` or any zone is
    /// out of range.
    pub fn eval_zones(
        &self,
        log10_ne: &[f64],
        log10_t: &[f64],
    ) -> Result<Vec<[f64; 3]>, IonError> {
        if log10_ne.len() != log10_t.len() {
            return Err(IonError::InvalidInput {
                context: "Fe56He4Table::eval_zones",
                detail: format!(
                    "log10_ne length ({}) != log10_t length ({})",
                    log10_ne.len(),
                    log10_t.len()
                ),
            });
        }
        let n = log10_ne.len();
        if n >= PARALLEL_ZONE_THRESHOLD {
            (0..n)
                .into_par_iter()
                .map(|k| {
                    let ev = self.eval_full(log10_ne[k], log10_t[k])?;
                    Ok([ev.f, ev.df_dx, ev.df_dy])
                })
                .collect()
        } else {
            (0..n)
                .map(|k| {
                    let ev = self.eval_full(log10_ne[k], log10_t[k])?;
                    Ok([ev.f, ev.df_dx, ev.df_dy])
                })
                .collect()
        }
    }
}

/// Evaluate `eval_charge_of_Fe56_in_He4` at $(\log_{10} n_e, \log_{10} T)$
/// using the built-in $105 \times 30$ TFDH spline table.
///
/// # Errors
/// Returns [`IonError::Interp`] if `log10_ne` or `log10_t` is non-finite or
/// outside the table domain.
pub fn eval_charge_of_fe56_in_he4(
    log10_ne: f64,
    log10_t: f64,
) -> Result<f64, IonError> {
    Fe56He4Table::builtin().eval(log10_ne, log10_t)
}

/// Evaluates `(f, df_dx, df_dy)` matching `fvbicub_db`
/// (`interp_2d/private/bicub_db.f:1420-1575`), which omits `implicit none`
/// so local basis scalars are implicit `REAL` (`f32`).
#[allow(clippy::cast_possible_truncation, clippy::too_many_arguments)]
fn eval_fvbicub_strict(
    xparam: f64,
    yparam: f64,
    hx: f64,
    hxi: f64,
    hy: f64,
    hyi: f64,
    c00: [f64; 4],
    c01: [f64; 4],
    c10: [f64; 4],
    c11: [f64; 4],
) -> (f64, f64, f64) {
    let sixth = 1.0_f64 / 6.0_f64;
    let z36th = (sixth * sixth) as f32;

    let xp = xparam as f32;
    let xpi = (1.0_f64 - f64::from(xp)) as f32;
    let xp2 = xp * xp;
    let xpi2 = xpi * xpi;
    let cx = (f64::from(xp) * (f64::from(xp2) - 1.0_f64)) as f32;
    let cxi = (f64::from(xpi) * (f64::from(xpi2) - 1.0_f64)) as f32;
    let hx2 = (hx * hx) as f32;

    let yp = yparam as f32;
    let ypi = (1.0_f64 - f64::from(yp)) as f32;
    let yp2 = yp * yp;
    let ypi2 = ypi * ypi;
    let cy = (f64::from(yp) * (f64::from(yp2) - 1.0_f64)) as f32;
    let cyi = (f64::from(ypi) * (f64::from(ypi2) - 1.0_f64)) as f32;
    let hy2 = (hy * hy) as f32;

    let cxd = (3.0_f64 * f64::from(xp2) - 1.0_f64) as f32;
    let cxdi = (-3.0_f64 * f64::from(xpi2) + 1.0_f64) as f32;
    let cyd = (3.0_f64 * f64::from(yp2) - 1.0_f64) as f32;
    let cydi = (-3.0_f64 * f64::from(ypi2) + 1.0_f64) as f32;

    let z36th_hx2_hy2_d = f64::from((z36th * hx2) * hy2);
    let z36th_hx2_d = f64::from(z36th * hx2);
    let (z36th_d, hx2_d, hy2_d) =
        (f64::from(z36th), f64::from(hx2), f64::from(hy2));
    let (xp_d, xpi_d, cx_d, cxi_d) =
        (f64::from(xp), f64::from(xpi), f64::from(cx), f64::from(cxi));
    let (yp_d, ypi_d, cy_d, cyi_d) =
        (f64::from(yp), f64::from(ypi), f64::from(cy), f64::from(cyi));
    let (cxd_d, cxdi_d, cyd_d, cydi_d) = (
        f64::from(cxd),
        f64::from(cxdi),
        f64::from(cyd),
        f64::from(cydi),
    );

    let mut f = xpi_d * (ypi_d * c00[0] + yp_d * c01[0])
        + xp_d * (ypi_d * c10[0] + yp_d * c11[0]);
    f += sixth
        * hx2_d
        * (cxi_d * (ypi_d * c00[1] + yp_d * c01[1])
            + cx_d * (ypi_d * c10[1] + yp_d * c11[1]));
    f += sixth
        * hy2_d
        * (xpi_d * (cyi_d * c00[2] + cy_d * c01[2])
            + xp_d * (cyi_d * c10[2] + cy_d * c11[2]));
    f += z36th_hx2_hy2_d
        * (cxi_d * (cyi_d * c00[3] + cy_d * c01[3])
            + cx_d * (cyi_d * c10[3] + cy_d * c11[3]));

    let mut df_dx = hxi
        * (-(ypi_d * c00[0] + yp_d * c01[0])
            + (ypi_d * c10[0] + yp_d * c11[0]));
    df_dx += sixth
        * hx
        * (cxdi_d * (ypi_d * c00[1] + yp_d * c01[1])
            + cxd_d * (ypi_d * c10[1] + yp_d * c11[1]));
    df_dx += sixth
        * hxi
        * hy2_d
        * (-(cyi_d * c00[2] + cy_d * c01[2])
            + (cyi_d * c10[2] + cy_d * c11[2]));
    df_dx += z36th_d
        * hx
        * hy2_d
        * (cxdi_d * (cyi_d * c00[3] + cy_d * c01[3])
            + cxd_d * (cyi_d * c10[3] + cy_d * c11[3]));

    let mut df_dy =
        hyi * (xpi_d * (-c00[0] + c01[0]) + xp_d * (-c10[0] + c11[0]));
    df_dy += sixth
        * hx2_d
        * hyi
        * (cxi_d * (-c00[1] + c01[1]) + cx_d * (-c10[1] + c11[1]));
    df_dy += sixth
        * hy
        * (xpi_d * (cydi_d * c00[2] + cyd_d * c01[2])
            + xp_d * (cydi_d * c10[2] + cyd_d * c11[2]));
    df_dy += z36th_hx2_d
        * hy
        * (cxi_d * (cydi_d * c00[3] + cyd_d * c01[3])
            + cx_d * (cydi_d * c10[3] + cyd_d * c11[3]));

    (f, df_dx, df_dy)
}

/// Parse whitespace-separated `f64` values from an ASCII file via
/// [`flexmesa_math::str_to_vector`].
fn parse_float_tokens(path: &Path) -> Result<Vec<f64>, IonError> {
    let text =
        fs::read_to_string(path).map_err(|e| IonError::Io(e.to_string()))?;
    str_to_vector(&text).map_err(|e| IonError::TableParse {
        path: path.display().to_string(),
        detail: e.to_string(),
    })
}
