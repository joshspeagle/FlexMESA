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
    interp_evbicub_db, interp_mkbicub_db, BicubicBc, BicubicEval,
};
use flexmesa_math::str_to_vector;
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
        interp_evbicub_db(
            log10_ne,
            log10_t,
            &self.log_ne,
            &self.log_t,
            self.ilinx,
            self.iliny,
            &self.coeffs,
        )
        .map_err(|e| IonError::Interp(e.to_string()))
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
