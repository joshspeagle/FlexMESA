// SPDX-License-Identifier: GPL-3.0-or-later
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
    StarError, TidalDeformationHandle, TidalDeformationHook,
    TidalShellDeformation,
};
use flexmesa_utils::resolve_mesa_test_inputs_dir;

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
    pub fn parse_ascii(text: &str) -> Result<Self, StarError> {
        let mut tokens = text.split_whitespace();
        let nx_tok = tokens.next().ok_or_else(|| {
            StarError::IoError(String::from("empty Roche deformation table"))
        })?;
        let nx: usize = nx_tok.parse().map_err(|e| {
            StarError::IoError(format!(
                "invalid nx '{nx_tok}' in Roche table: {e}"
            ))
        })?;
        if nx < 2 {
            return Err(StarError::IoError(format!(
                "invalid nx={nx} in Roche table (must be >= 2)"
            )));
        }
        let mut xs = Vec::with_capacity(nx);
        for _ in 0..nx {
            let tok = tokens.next().ok_or_else(|| {
                StarError::IoError(String::from(
                    "truncated x-grid in Roche table",
                ))
            })?;
            xs.push(parse_float_token(tok)?);
        }
        let ny_tok = tokens.next().ok_or_else(|| {
            StarError::IoError(String::from("missing ny in Roche table"))
        })?;
        let ny: usize = ny_tok.parse().map_err(|e| {
            StarError::IoError(format!(
                "invalid ny '{ny_tok}' in Roche table: {e}"
            ))
        })?;
        if ny < 2 {
            return Err(StarError::IoError(format!(
                "invalid ny={ny} in Roche table (must be >= 2)"
            )));
        }
        let mut ys = Vec::with_capacity(ny);
        for _ in 0..ny {
            let tok = tokens.next().ok_or_else(|| {
                StarError::IoError(String::from(
                    "truncated y-grid in Roche table",
                ))
            })?;
            ys.push(parse_float_token(tok)?);
        }
        let mut values = Vec::with_capacity(nx * ny);
        for _ in 0..(nx * ny) {
            let tok = tokens.next().ok_or_else(|| {
                StarError::IoError(String::from(
                    "truncated values grid in Roche table",
                ))
            })?;
            values.push(parse_float_token(tok)?);
        }
        Ok(Self { xs, ys, values })
    }

    /// Evaluates the 2D Steffen piecewise-monotonic cubic interpolant at
    /// `(lq, ar_in)` (`eval_fp`/`interp_evbipm_db` in
    /// `binary_roche_deformation.f90:133-197` and `bipm_db.f90:51-118`).
    #[must_use]
    pub fn eval(&self, lq: f64, ar_in: f64) -> f64 {
        let (nx, ny) = (self.xs.len(), self.ys.len());
        if nx < 2 || ny < 2 || !lq.is_finite() || !ar_in.is_finite() {
            return f64::NAN;
        }
        let x_top = (self.xs[nx - 1] - 1.0e-12).max(self.xs[0]);
        let x = if lq >= self.xs[nx - 1] {
            x_top
        } else if lq < self.xs[0] {
            self.xs[0]
        } else {
            lq
        };
        let y_top = (self.ys[ny - 1] - 2.0 * ROCHE_NUDGE).max(self.ys[0]);
        let y = if ar_in >= self.ys[ny - 1] {
            y_top
        } else if ar_in < self.ys[0] {
            self.ys[0]
        } else {
            ar_in
        };
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
        let (iy0, iy1) = if ny >= 4 {
            let jlo = if iy == 0 {
                0
            } else if iy >= ny - 2 {
                ny - 4
            } else {
                iy - 1
            };
            (jlo, jlo + 3)
        } else {
            (0, ny - 1)
        };
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

fn parse_float_token(tok: &str) -> Result<f64, StarError> {
    if tok.eq_ignore_ascii_case("-inf") || tok.eq_ignore_ascii_case("-infinity")
    {
        return Ok(f64::NEG_INFINITY);
    }
    if tok.eq_ignore_ascii_case("inf")
        || tok.eq_ignore_ascii_case("+inf")
        || tok.eq_ignore_ascii_case("infinity")
    {
        return Ok(f64::INFINITY);
    }
    if tok.eq_ignore_ascii_case("nan") {
        return Ok(f64::NAN);
    }
    tok.parse::<f64>().map_err(|e| {
        StarError::IoError(format!("invalid float '{tok}' in Roche table: {e}"))
    })
}

fn fortran_sign1(v: f64) -> f64 {
    if v < 0.0 {
        -1.0
    } else {
        1.0
    }
}

fn steffen_1d_local(xs: &[f64], ys: &[f64], k: usize, x: f64) -> f64 {
    let n = xs.len();
    if n < 2 || k + 1 >= n {
        return f64::NAN;
    }
    if n == 2 {
        let h0 = (xs[1] - xs[0]).max(1.0e-30);
        let s0 = (ys[1] - ys[0]) / h0;
        let dx = x - xs[0];
        return ys[0] + dx * s0;
    }
    let mut h = [0.0_f64; 3];
    let mut s = [0.0_f64; 3];
    let mut d = [0.0_f64; 4];
    for i in 0..(n - 1) {
        h[i] = (xs[i + 1] - xs[i]).max(1.0e-30);
        s[i] = (ys[i + 1] - ys[i]) / h[i];
    }
    let p0 = s[0] * (1.0 + h[0] / (h[0] + h[1])) - s[1] * h[0] / (h[0] + h[1]);
    d[0] = if p0 * s[0] <= 0.0 {
        0.0
    } else if p0.abs() > 2.0 * s[0].abs() {
        2.0 * s[0]
    } else {
        p0
    };
    for i in 1..(n - 1) {
        let p_i = (s[i - 1] * h[i] + s[i] * h[i - 1]) / (h[i - 1] + h[i]);
        d[i] = (fortran_sign1(s[i - 1]) + fortran_sign1(s[i]))
            * s[i - 1].abs().min(s[i].abs()).min(0.5 * p_i.abs());
    }
    let m = n - 1;
    let pn = s[m - 1] * (1.0 + h[m - 1] / (h[m - 1] + h[m - 2]))
        - s[m - 2] * h[m - 1] / (h[m - 1] + h[m - 2]);
    d[m] = if pn * s[m - 1] <= 0.0 {
        0.0
    } else if pn.abs() > 2.0 * s[m - 1].abs() {
        2.0 * s[m - 1]
    } else {
        pn
    };
    let c0 = ys[k];
    let c1 = d[k];
    let c2 = (3.0 * s[k] - 2.0 * d[k] - d[k + 1]) / h[k];
    let c3 = (d[k] + d[k + 1] - 2.0 * s[k]) / (h[k] * h[k]);
    let dx = x - xs[k];
    c0 + dx * (c1 + dx * (c2 + dx * c3))
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

static GLOBAL_ROCHE_TABLES: OnceLock<Result<RocheDeformationTables, String>> =
    OnceLock::new();

fn resolve_roche_data_dir_from(mesa_root: &Path) -> Result<PathBuf, StarError> {
    let dir = mesa_root.join("data").join("roche_data");
    if !dir.is_dir() {
        return Err(StarError::IoError(format!(
            "roche_data directory not found at '{}' (set MESA_DIR or run \
             `flexmesa data fetch --include-test-inputs`)",
            dir.display()
        )));
    }
    Ok(dir)
}

fn resolve_roche_data_dir() -> Result<PathBuf, StarError> {
    resolve_roche_data_dir_from(&resolve_mesa_test_inputs_dir())
}

impl RocheDeformationTables {
    /// Loads the four Roche distortion tables from `roche_data_dir`.
    pub fn from_dir(roche_data_dir: &Path) -> Result<Self, StarError> {
        if !roche_data_dir.is_dir() {
            return Err(StarError::IoError(format!(
                "roche_data directory not found at '{}'",
                roche_data_dir.display()
            )));
        }
        let load = |name: &str| -> Result<RocheTable2D, StarError> {
            let path = roche_data_dir.join(name);
            let text = fs::read_to_string(&path).map_err(|e| {
                StarError::IoError(format!(
                    "failed to read Roche table '{}': {e}",
                    path.display()
                ))
            })?;
            RocheTable2D::parse_ascii(&text).map_err(|e| {
                StarError::IoError(format!(
                    "failed to parse Roche table '{}': {e}",
                    path.display()
                ))
            })
        };
        Ok(Self {
            fp: load("fp_data.txt")?,
            ft: load("ft_data.txt")?,
            irot: load("irot_data.txt")?,
            psi: load("psi_data.txt")?,
        })
    }

    /// Returns the lazily initialized global [`RocheDeformationTables`],
    /// loaded from `data/roche_data` under [`resolve_mesa_test_inputs_dir`].
    pub fn global() -> Result<&'static Self, StarError> {
        let res = GLOBAL_ROCHE_TABLES.get_or_init(|| {
            let dir = resolve_roche_data_dir().map_err(|e| e.to_string())?;
            Self::from_dir(&dir).map_err(|e| e.to_string())
        });
        res.as_ref().map_err(|msg| StarError::IoError(msg.clone()))
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
    tables: &'static RocheDeformationTables,
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
    pub fn new_handle(
        m_this_g: f64,
        m_other_g: f64,
        separation_cm: f64,
        rl_this_cm: f64,
        period_sec: f64,
        f_sync_switch_from_rot_defor: f64,
        f_sync_switch_width: f64,
        f_sync_switch_lim: f64,
    ) -> Result<TidalDeformationHandle, StarError> {
        let tables = RocheDeformationTables::global()?;
        let m1 = m_this_g.max(1.0e20);
        let m2 = m_other_g.max(1.0e20);
        let r_roche_cm = if rl_this_cm > 0.0 {
            rl_this_cm
        } else {
            eval_rlobe(m1, m2, separation_cm.max(1.0e5))
        };
        let lq = flexmesa_math::log10(m2 / m1);
        Ok(TidalDeformationHandle(Arc::new(Self {
            tables,
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
        let tables = self.tables;
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

#[cfg(test)]
mod tests {
    use std::process::Command;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    struct TempDirGuard {
        root: PathBuf,
    }

    impl TempDirGuard {
        fn new(tag: &str) -> Self {
            static SEQ: AtomicU64 = AtomicU64::new(0);
            let id = SEQ.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "flexmesa_roche_{tag}_{}_{id}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&root);
            fs::create_dir_all(&root).expect("create temp roche dir");
            Self { root }
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn constant_table_text(val: f64) -> String {
        format!("2\n-1.0 1.0\n2\n0.0 1.0\n{val} {val}\n{val} {val}\n")
    }

    fn write_minimal_roche_tables(roche_dir: &Path, fp_val: f64) {
        fs::create_dir_all(roche_dir).expect("mkdir roche_data");
        let files = [
            ("fp_data.txt", fp_val),
            ("ft_data.txt", 0.8),
            ("irot_data.txt", 1.25),
            ("psi_data.txt", -1.5),
        ];
        for (name, val) in files {
            fs::write(roche_dir.join(name), constant_table_text(val))
                .expect("write minimal roche table");
        }
    }

    #[test]
    fn test_resolve_roche_data_dir_from_rejects_missing_or_file() {
        let tmp = TempDirGuard::new("root");
        let missing_root = tmp.root.join("missing_mesa");
        fs::create_dir_all(&missing_root).expect("mkdir missing_mesa");
        let expected_missing = missing_root.join("data").join("roche_data");
        let err = resolve_roche_data_dir_from(&missing_root)
            .expect_err("missing data/roche_data must fail");
        assert!(matches!(err, StarError::IoError(_)));
        let msg = err.to_string();
        assert!(
            msg.contains(&expected_missing.display().to_string()),
            "expected missing path in error: {msg}"
        );

        let file_root = tmp.root.join("file_mesa");
        fs::create_dir_all(file_root.join("data")).expect("mkdir data");
        let file_roche = file_root.join("data").join("roche_data");
        fs::write(&file_roche, b"not a dir").expect("write marker file");
        let err_file = resolve_roche_data_dir_from(&file_root)
            .expect_err("non-directory roche_data must fail");
        assert!(
            err_file
                .to_string()
                .contains(&file_roche.display().to_string()),
            "expected non-dir path in error: {err_file}"
        );

        let valid_root = tmp.root.join("valid_mesa");
        let valid_roche = valid_root.join("data").join("roche_data");
        fs::create_dir_all(&valid_roche).expect("mkdir valid roche_data");
        let resolved = resolve_roche_data_dir_from(&valid_root)
            .expect("valid roche_data directory should resolve");
        assert_eq!(resolved, valid_roche);
    }

    #[test]
    fn test_roche_tables_from_dir_requires_dir_and_all_four_files() {
        let tmp = TempDirGuard::new("tables");
        let absent = tmp.root.join("absent_roche_data");
        let err_dir = RocheDeformationTables::from_dir(&absent)
            .expect_err("absent roche_data dir must fail");
        assert!(
            err_dir.to_string().contains(&absent.display().to_string()),
            "expected missing dir in error: {err_dir}"
        );

        let roche_dir = tmp.root.join("roche_data");
        write_minimal_roche_tables(&roche_dir, 0.65);
        let tables = RocheDeformationTables::from_dir(&roche_dir)
            .expect("all four tables present should load");
        assert!((tables.eval_fp(0.0, 0.5) - 0.65).abs() < 1.0e-14);
        assert!((tables.eval_ft(0.0, 0.5) - 0.8).abs() < 1.0e-14);
        assert!((tables.eval_irot(0.0, 0.5) - 1.25).abs() < 1.0e-14);
        assert!((tables.eval_psi(0.0, 0.5) + 1.5).abs() < 1.0e-14);

        for missing_name in [
            "fp_data.txt",
            "ft_data.txt",
            "irot_data.txt",
            "psi_data.txt",
        ] {
            let case_dir = tmp.root.join(format!("missing_{missing_name}"));
            write_minimal_roche_tables(&case_dir, 0.65);
            let missing_file = case_dir.join(missing_name);
            fs::remove_file(&missing_file).expect("remove table file");
            let err = RocheDeformationTables::from_dir(&case_dir)
                .expect_err("incomplete roche_data directory must fail");
            let msg = err.to_string();
            assert!(
                msg.contains(&missing_file.display().to_string()),
                "expected missing file '{missing_name}' in error: {msg}"
            );
        }
    }

    fn spawn_env_probe(
        mode: &str,
        mesa_dir: Option<&Path>,
        xdg_cache_home: Option<&Path>,
        home: Option<&Path>,
        expected_substr: &str,
    ) {
        let exe = std::env::current_exe().expect("current test executable");
        let mut cmd = Command::new(exe);
        cmd.args([
            "--exact",
            "roche_deformation::tests::roche_env_probe_worker",
            "--nocapture",
        ])
        .env("FLEXMESA_ROCHE_ENV_PROBE", mode)
        .env("FLEXMESA_ROCHE_EXPECTED", expected_substr);
        match mesa_dir {
            Some(p) => {
                cmd.env("MESA_DIR", p);
            }
            None => {
                cmd.env_remove("MESA_DIR");
            }
        }
        match xdg_cache_home {
            Some(p) => {
                cmd.env("XDG_CACHE_HOME", p);
            }
            None => {
                cmd.env_remove("XDG_CACHE_HOME");
            }
        }
        match home {
            Some(p) => {
                cmd.env("HOME", p);
            }
            None => {
                cmd.env_remove("HOME");
            }
        }
        let output = cmd.output().expect("spawn roche env probe process");
        assert!(
            output.status.success(),
            "probe mode '{mode}' failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn roche_env_probe_worker() {
        let Ok(mode) = std::env::var("FLEXMESA_ROCHE_ENV_PROBE") else {
            return;
        };
        let expected =
            std::env::var("FLEXMESA_ROCHE_EXPECTED").unwrap_or_default();
        match mode.as_str() {
            "expect_dir_error" => {
                let dir_err = resolve_roche_data_dir()
                    .expect_err("resolve_roche_data_dir must fail loudly");
                assert!(
                    dir_err.to_string().contains(&expected),
                    "missing expected path '{expected}' in dir error: {dir_err}"
                );
                let global_err = RocheDeformationTables::global()
                    .expect_err("global() must fail loudly");
                assert!(
                    global_err.to_string().contains(&expected),
                    "missing expected path '{expected}' in global error: \
                     {global_err}"
                );
            }
            "expect_global_error" => {
                let global_err = RocheDeformationTables::global()
                    .expect_err("partial roche_data must fail loudly");
                assert!(
                    global_err.to_string().contains(&expected),
                    "missing expected path '{expected}' in global error: \
                     {global_err}"
                );
            }
            "expect_fp_val" => {
                let want: f64 = expected.parse().expect("expected fp float");
                let tables = RocheDeformationTables::global()
                    .expect("expected valid roche tables");
                let got = tables.eval_fp(0.0, 0.5);
                assert!(
                    (got - want).abs() < 1.0e-14,
                    "loaded wrong roche root: got fp={got}, want fp={want}"
                );
            }
            other => {
                panic!("unexpected FLEXMESA_ROCHE_ENV_PROBE mode: {other}")
            }
        }
    }

    #[test]
    fn test_resolve_roche_data_dir_uses_single_root_without_fallback() {
        let tmp = TempDirGuard::new("env_probe");
        let mesa_empty = tmp.root.join("mesa_empty");
        let mesa_partial = tmp.root.join("mesa_partial");
        let mesa_valid = tmp.root.join("mesa_valid");
        let xdg_empty = tmp.root.join("xdg_empty");
        let xdg_valid = tmp.root.join("xdg_valid");
        let home_valid = tmp.root.join("home_valid");

        fs::create_dir_all(&mesa_empty).expect("mkdir mesa_empty");
        fs::create_dir_all(&xdg_empty).expect("mkdir xdg_empty");

        write_minimal_roche_tables(
            &mesa_partial.join("data").join("roche_data"),
            0.31,
        );
        fs::remove_file(
            mesa_partial
                .join("data")
                .join("roche_data")
                .join("psi_data.txt"),
        )
        .expect("remove psi_data.txt");

        write_minimal_roche_tables(
            &mesa_valid.join("data").join("roche_data"),
            0.42,
        );
        write_minimal_roche_tables(
            &xdg_valid
                .join("flexmesa")
                .join("mesa_test_inputs")
                .join("data")
                .join("roche_data"),
            0.61,
        );
        write_minimal_roche_tables(
            &home_valid
                .join(".cache")
                .join("flexmesa")
                .join("mesa_test_inputs")
                .join("data")
                .join("roche_data"),
            0.77,
        );

        // 1. MESA_DIR set without data/roche_data must NOT fall back to XDG or HOME.
        let missing_mesa_roche = mesa_empty.join("data").join("roche_data");
        spawn_env_probe(
            "expect_dir_error",
            Some(&mesa_empty),
            Some(&xdg_valid),
            Some(&home_valid),
            &missing_mesa_roche.display().to_string(),
        );

        // 2. XDG_CACHE_HOME set without mesa_test_inputs/data/roche_data must NOT fall back to HOME.
        let missing_xdg_roche = xdg_empty
            .join("flexmesa")
            .join("mesa_test_inputs")
            .join("data")
            .join("roche_data");
        spawn_env_probe(
            "expect_dir_error",
            None,
            Some(&xdg_empty),
            Some(&home_valid),
            &missing_xdg_roche.display().to_string(),
        );

        // 3. Partial MESA_DIR/data/roche_data must fail naming the missing file without fallback.
        let missing_psi = mesa_partial
            .join("data")
            .join("roche_data")
            .join("psi_data.txt");
        spawn_env_probe(
            "expect_global_error",
            Some(&mesa_partial),
            Some(&xdg_valid),
            Some(&home_valid),
            &missing_psi.display().to_string(),
        );

        // 4. Valid MESA_DIR takes precedence over XDG_CACHE_HOME and HOME.
        spawn_env_probe(
            "expect_fp_val",
            Some(&mesa_valid),
            Some(&xdg_valid),
            Some(&home_valid),
            "0.42",
        );

        // 5. Valid XDG_CACHE_HOME takes precedence over HOME when MESA_DIR is unset.
        spawn_env_probe(
            "expect_fp_val",
            None,
            Some(&xdg_valid),
            Some(&home_valid),
            "0.61",
        );

        // 6. Valid HOME fallback resolves when MESA_DIR and XDG_CACHE_HOME are unset.
        spawn_env_probe("expect_fp_val", None, None, Some(&home_valid), "0.77");
    }
}
