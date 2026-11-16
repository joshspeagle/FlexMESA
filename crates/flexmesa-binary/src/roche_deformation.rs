// SPDX-License-Identifier: LGPL-3.0-or-later
//! Roche-potential tidal distortion tables $(f_P, f_T, i_{\mathrm{rot}}, \psi)$
//! and synchronicity blending switch for binary stars (`use_tidal_deformation`).
//!
//! Corresponds to `binary/private/binary_roche_deformation.f90` (Fabry,
//! Marchant & Sana 2022, A&A 661, A123), interpolating 2D piecewise-monotonic
//! Steffen cubic splines over $\log_{10}(M_{\mathrm{other}}/M_{\mathrm{this}})$
//! and fractional Roche-lobe radius $a_r = r / R_L$ from
//! `$MESA_DIR/data/roche_data/{fp,ft,irot,psi}_data.txt`.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use flexmesa_const::PI;
use flexmesa_star::{
    TidalDeformationHandle, TidalDeformationHook, TidalShellDeformation,
};

use crate::orbit::eval_rlobe;

/// Finite-difference step $\Delta(r / R_L) = 10^{-4}$ used for logarithmic
/// radius derivatives (`nudge` in `binary_roche_deformation.f90:43`).
pub const ROCHE_NUDGE: f64 = 1.0e-4;

/// 2D rectilinear grid table loaded from `$MESA_DIR/data/roche_data/*_data.txt`.
#[derive(Debug, Clone)]
pub struct RocheTable2D {
    /// Grid coordinates $x_i = \log_{10}(M_{\mathrm{other}} / M_{\mathrm{this}})$.
    pub xs: Vec<f64>,
    /// Grid coordinates $y_j = r / R_L$.
    pub ys: Vec<f64>,
    /// Row-major table values `values[i * ny + j]` of length `xs.len() * ys.len()`.
    pub values: Vec<f64>,
}

impl RocheTable2D {
    /// Parses a single MESA `roche_data/*_data.txt` file (`setup_interpolator`
    /// in `binary_roche_deformation.f90:89-129`).
    pub fn parse_ascii(text: &str) -> Option<Self> {
        let mut tokens = text.split_whitespace();
        let nx: usize = tokens.next()?.parse().ok()?;
        if nx < 2 {
            return None;
        }
        let mut xs = Vec::with_capacity(nx);
        for _ in 0..nx {
            xs.push(parse_float_token(tokens.next()?)?);
        }
        let ny: usize = tokens.next()?.parse().ok()?;
        if ny < 2 {
            return None;
        }
        let mut ys = Vec::with_capacity(ny);
        for _ in 0..ny {
            ys.push(parse_float_token(tokens.next()?)?);
        }
        let mut values = Vec::with_capacity(nx * ny);
        for _ in 0..(nx * ny) {
            values.push(parse_float_token(tokens.next()?)?);
        }
        Some(Self { xs, ys, values })
    }

    /// Evaluates the 2D Steffen piecewise-monotonic cubic interpolant at
    /// `(lq, ar)` (`interp_evbipm_db` in `binary_roche_deformation.f90:133-197`).
    #[must_use]
    pub fn eval(&self, lq: f64, ar_in: f64) -> f64 {
        let (nx, ny) = (self.xs.len(), self.ys.len());
        if nx < 2 || ny < 2 {
            return f64::NAN;
        }
        let x = lq.clamp(self.xs[0], self.xs[nx - 1] - 1.0e-12);
        let y = ar_in.clamp(self.ys[0], self.ys[ny - 1] - 2.0 * ROCHE_NUDGE);
        let ix = self
            .xs
            .partition_point(|&xv| xv <= x)
            .saturating_sub(1)
            .min(nx - 2);
        let iy = self
            .ys
            .partition_point(|&yv| yv <= y)
            .saturating_sub(1)
            .min(ny - 2);
        let (ix0, ix1) = (ix.saturating_sub(1), (ix + 2).min(nx - 1));
        let (iy0, iy1) = (iy.saturating_sub(1), (iy + 2).min(ny - 1));
        let (x_sub, y_sub) = (&self.xs[ix0..=ix1], &self.ys[iy0..=iy1]);
        let mut col_vals = [0.0_f64; 4];
        for (ky, jy) in (iy0..=iy1).enumerate() {
            let mut row_vals = [0.0_f64; 4];
            for (kx, jx) in (ix0..=ix1).enumerate() {
                let v = self.values[jx * ny + jy];
                if !v.is_finite() {
                    return f64::NAN;
                }
                row_vals[kx] = v;
            }
            col_vals[ky] =
                steffen_1d_local(x_sub, &row_vals[..x_sub.len()], ix - ix0, x);
        }
        steffen_1d_local(y_sub, &col_vals[..y_sub.len()], iy - iy0, y)
    }
}

fn parse_float_token(tok: &str) -> Option<f64> {
    if tok.eq_ignore_ascii_case("-inf") || tok.eq_ignore_ascii_case("-infinity")
    {
        return Some(f64::NEG_INFINITY);
    }
    if tok.eq_ignore_ascii_case("inf")
        || tok.eq_ignore_ascii_case("+inf")
        || tok.eq_ignore_ascii_case("infinity")
    {
        return Some(f64::INFINITY);
    }
    if tok.eq_ignore_ascii_case("nan") {
        return Some(f64::NAN);
    }
    tok.parse::<f64>().ok()
}

fn steffen_slope_at(xs: &[f64], ys: &[f64], k: usize) -> f64 {
    let n = xs.len();
    if n < 2 {
        return 0.0;
    }
    if k == 0 {
        return (ys[1] - ys[0]) / (xs[1] - xs[0]).max(1.0e-30);
    }
    if k + 1 >= n {
        return (ys[n - 1] - ys[n - 2]) / (xs[n - 1] - xs[n - 2]).max(1.0e-30);
    }
    let (h0, h1) = (
        (xs[k] - xs[k - 1]).max(1.0e-30),
        (xs[k + 1] - xs[k]).max(1.0e-30),
    );
    let (d0, d1) = ((ys[k] - ys[k - 1]) / h0, (ys[k + 1] - ys[k]) / h1);
    if d0 * d1 <= 0.0 {
        return 0.0;
    }
    let p = (d0 * h1 + d1 * h0) / (h0 + h1);
    d0.signum() * p.abs().min(2.0 * d0.abs()).min(2.0 * d1.abs())
}

fn steffen_1d_local(xs: &[f64], ys: &[f64], k: usize, x: f64) -> f64 {
    let h = (xs[k + 1] - xs[k]).max(1.0e-30);
    let t = ((x - xs[k]) / h).clamp(0.0, 1.0);
    let (s0, s1, omt) = (
        steffen_slope_at(xs, ys, k),
        steffen_slope_at(xs, ys, k + 1),
        1.0 - t,
    );
    (1.0 + 2.0 * t) * omt * omt * ys[k]
        + t * omt * omt * h * s0
        + t * t * (3.0 - 2.0 * t) * ys[k + 1]
        + t * t * (t - 1.0) * h * s1
}

/// Complete collection of Fabry et al. (2022) Roche-potential tables.
#[derive(Debug, Clone)]
pub struct RocheDeformationTables {
    /// Pressure distortion table $f_P(\log_{10} q, r / R_L)$.
    pub fp: RocheTable2D,
    /// Temperature distortion table $f_T(\log_{10} q, r / R_L)$.
    pub ft: RocheTable2D,
    /// Dimensionless moment-of-inertia table $i_{\mathrm{rot}} / (\tfrac{2}{3} r^2)$.
    pub irot: RocheTable2D,
    /// Dimensionless Roche potential table $\psi(\log_{10} q, r / R_L)$.
    pub psi: RocheTable2D,
}

static GLOBAL_ROCHE_TABLES: OnceLock<Option<RocheDeformationTables>> =
    OnceLock::new();

fn resolve_roche_data_dir() -> Option<PathBuf> {
    if let Some(mesa_dir) = std::env::var_os("MESA_DIR") {
        let p = PathBuf::from(mesa_dir).join("data").join("roche_data");
        if p.is_dir() {
            return Some(p);
        }
    }
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
        let p = PathBuf::from(xdg)
            .join("flexmesa")
            .join("mesa_test_inputs")
            .join("data")
            .join("roche_data");
        if p.is_dir() {
            return Some(p);
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let p = PathBuf::from(home)
            .join(".cache")
            .join("flexmesa")
            .join("mesa_test_inputs")
            .join("data")
            .join("roche_data");
        if p.is_dir() {
            return Some(p);
        }
    }
    None
}

impl RocheDeformationTables {
    /// Loads the four Roche distortion tables from `roche_data_dir`.
    pub fn from_dir(roche_data_dir: &Path) -> Option<Self> {
        let load = |name: &str| -> Option<RocheTable2D> {
            let text = fs::read_to_string(roche_data_dir.join(name)).ok()?;
            RocheTable2D::parse_ascii(&text)
        };
        Some(Self {
            fp: load("fp_data.txt")?,
            ft: load("ft_data.txt")?,
            irot: load("irot_data.txt")?,
            psi: load("psi_data.txt")?,
        })
    }

    /// Returns the lazily initialized global [`RocheDeformationTables`].
    pub fn global() -> Option<&'static Self> {
        GLOBAL_ROCHE_TABLES
            .get_or_init(|| {
                let dir = resolve_roche_data_dir()?;
                Self::from_dir(&dir)
            })
            .as_ref()
    }

    /// Evaluates $f_P(\log_{10} q, a_r)$ (`eval_fp` in `binary_roche_deformation.f90:133-146`).
    #[must_use]
    pub fn eval_fp(&self, lq: f64, ar: f64) -> f64 {
        self.fp.eval(lq, ar)
    }

    /// Evaluates $f_T(\log_{10} q, a_r)$ (`eval_ft` in `binary_roche_deformation.f90:148-162`).
    #[must_use]
    pub fn eval_ft(&self, lq: f64, ar: f64) -> f64 {
        self.ft.eval(lq, ar)
    }

    /// Evaluates $i_{\mathrm{rot}} / (\tfrac{2}{3} r^2)$ (`eval_irot` in `binary_roche_deformation.f90:164-179`).
    #[must_use]
    pub fn eval_irot(&self, lq: f64, ar: f64) -> f64 {
        self.irot.eval(lq, ar)
    }

    /// Evaluates dimensionless Roche potential $\psi(\log_{10} q, a_r)$, mapping
    /// core `NaN`/`-inf` values to `-9.0e99` (`roche_psi` in `binary_roche_deformation.f90:181-234`).
    #[must_use]
    pub fn eval_psi(&self, lq: f64, ar: f64) -> f64 {
        let v = self.psi.eval(lq, ar);
        if v.is_finite() {
            v
        } else {
            -9.0e99
        }
    }

    /// Evaluates $(f_P, \partial f_P/\partial\ln r, f_T, \partial f_T/\partial\ln r)$
    /// (`roche_fp_ft` in `binary_roche_deformation.f90:236-290`).
    #[must_use]
    pub fn roche_fp_ft(&self, lq: f64, ar: f64) -> (f64, f64, f64, f64) {
        let fp = self.eval_fp(lq, ar);
        let ft = self.eval_ft(lq, ar);
        let d_fp_dlnr =
            (self.eval_fp(lq, ar + ROCHE_NUDGE) - fp) / ROCHE_NUDGE * ar;
        let d_ft_dlnr =
            (self.eval_ft(lq, ar + ROCHE_NUDGE) - ft) / ROCHE_NUDGE * ar;
        (fp, d_fp_dlnr, ft, d_ft_dlnr)
    }

    /// Evaluates $(i_{\mathrm{rot}}, \partial i_{\mathrm{rot}}/\partial\ln r)$ in
    /// $\mathrm{cm^2}$ at shell radius `r_cm` (`roche_irot` in `binary_roche_deformation.f90:292-336`).
    #[must_use]
    pub fn roche_irot(&self, lq: f64, ar: f64, r_cm: f64) -> (f64, f64) {
        let eval = self.eval_irot(lq, ar);
        let d_eval_d_ar =
            (self.eval_irot(lq, ar + ROCHE_NUDGE) - eval) / ROCHE_NUDGE;
        let sph_moi = (2.0 / 3.0) * r_cm * r_cm;
        let i_rot = eval * sph_moi;
        let d_irot_dlnr = sph_moi * (ar * d_eval_d_ar + 2.0 * eval);
        (i_rot, d_irot_dlnr)
    }
}

/// Evaluates the synchronicity blending factor $f_{\mathrm{switch}} \in [0, 1]$
/// between Roche tidal deformation (`1`) and single-star rotational distortion (`0`)
/// (`synchronicity` in `binary/private/binary_roche_deformation.f90:368-415`).
#[must_use]
pub fn eval_synchronicity_switch(
    omega_in: f64,
    period_sec: f64,
    f_sync_switch_from_rot_defor: f64,
    f_sync_switch_width: f64,
    f_sync_switch_lim: f64,
) -> f64 {
    if period_sec <= 0.0 || !period_sec.is_finite() {
        return 0.0;
    }
    let omega_sync = 2.0 * PI / period_sec;
    let omega = if omega_in.is_nan() {
        0.0
    } else {
        omega_in.abs()
    };
    let f_sync = if omega <= 0.0 || omega_sync <= 0.0 {
        0.0
    } else {
        let ratio = omega / omega_sync;
        ratio.min(1.0 / ratio)
    };
    let mut f_switch = 1.0
        / (1.0
            + flexmesa_math::exp(
                -f_sync_switch_width * (f_sync - f_sync_switch_from_rot_defor),
            ));
    if f_switch < f_sync_switch_lim {
        f_switch = 0.0;
    } else if 1.0 - f_switch < f_sync_switch_lim {
        f_switch = 1.0;
    }
    f_switch.clamp(0.0, 1.0)
}

/// Binary orbital parameters bound to a stellar component for [`TidalDeformationHook`].
#[derive(Debug, Clone)]
pub struct BinaryTidalDeformationHook {
    /// Log mass ratio $\log_{10}(M_{\mathrm{other}} / M_{\mathrm{this}})$.
    pub lq: f64,
    /// Roche-lobe radius $R_L$ of this star in $\mathrm{cm}$.
    pub r_roche_cm: f64,
    /// Orbital period $P$ in $\mathrm{s}$.
    pub period_sec: f64,
    /// Synchronicity sigmoid midpoint (`f_sync_switch_from_rot_defor`, default `0.8`).
    pub f_sync_switch_from_rot_defor: f64,
    /// Synchronicity sigmoid width (`f_sync_switch_width`, default `0.1`).
    pub f_sync_switch_width: f64,
    /// Synchronicity sigmoid cutoff threshold (`f_sync_switch_lim`, default `0.01`).
    pub f_sync_switch_lim: f64,
}

impl BinaryTidalDeformationHook {
    /// Constructs a [`TidalDeformationHandle`] for a binary component when
    /// `$MESA_DIR/data/roche_data` tables are available.
    #[must_use]
    pub fn new_handle(
        m_this_g: f64,
        m_other_g: f64,
        separation_cm: f64,
        rl_this_cm: f64,
        period_sec: f64,
        f_sync_switch_from_rot_defor: f64,
        f_sync_switch_width: f64,
        f_sync_switch_lim: f64,
    ) -> Option<TidalDeformationHandle> {
        let _ = RocheDeformationTables::global()?;
        let m1 = m_this_g.max(1.0e20);
        let m2 = m_other_g.max(1.0e20);
        let r_roche_cm = if rl_this_cm > 0.0 {
            rl_this_cm
        } else {
            eval_rlobe(m1, m2, separation_cm.max(1.0e5))
        };
        let lq = flexmesa_math::log10(m2 / m1);
        Some(TidalDeformationHandle(Arc::new(Self {
            lq,
            r_roche_cm: r_roche_cm.max(1.0e5),
            period_sec,
            f_sync_switch_from_rot_defor,
            f_sync_switch_width,
            f_sync_switch_lim,
        })))
    }
}

impl TidalDeformationHook for BinaryTidalDeformationHook {
    fn eval_shell(
        &self,
        _k: usize,
        r_cm: f64,
        omega: f64,
    ) -> TidalShellDeformation {
        let Some(tables) = RocheDeformationTables::global() else {
            return TidalShellDeformation {
                fp: 1.0,
                d_fp_dlnr: 0.0,
                ft: 1.0,
                d_ft_dlnr: 0.0,
                i_rot: (2.0 / 3.0) * r_cm * r_cm,
                d_irot_dlnr: (4.0 / 3.0) * r_cm * r_cm,
                f_switch: 0.0,
            };
        };
        let ar = (r_cm / self.r_roche_cm).max(0.0);
        let (fp, d_fp_dlnr, ft, d_ft_dlnr) = tables.roche_fp_ft(self.lq, ar);
        let (i_rot, d_irot_dlnr) = tables.roche_irot(self.lq, ar, r_cm);
        let f_switch = eval_synchronicity_switch(
            omega,
            self.period_sec,
            self.f_sync_switch_from_rot_defor,
            self.f_sync_switch_width,
            self.f_sync_switch_lim,
        );
        TidalShellDeformation {
            fp,
            d_fp_dlnr,
            ft,
            d_ft_dlnr,
            i_rot,
            d_irot_dlnr,
            f_switch,
        }
    }
}
