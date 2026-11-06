//! Four-dimensional $(Z, X, \log Q, \log T)$ ionization table loader and
//! built-in solar/stellar reference cell (`ion_tables_load.f90`).
//!
//! Provides [`IonTableGrid`], including:
//! - [`IonTableGrid::builtin`]: Lazily initialized reference $(\log Q, \log T)$
//!   cell (`iQ = 88..89`, $\log Q \in [-2.08, -1.99]$; `jT = 40..41`,
//!   $\log T \in [4.50, 4.56]$) containing exact upstream MESA not-a-knot
//!   bicubic spline coefficients across $(Z, X)$.
//! - [`IonTableGrid::from_data_dir`]: Full 22-table ASCII loader (`ion_*.data`)
//!   that builds global $175 \times 102$ not-a-knot bicubic splines via
//!   [`flexmesa_interp2d::interp_mkbicub_db`] and sanitizes non-finite
//!   higher-order derivatives (`Check_ion_Interpolation_Data`).

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use flexmesa_interp2d::{interp_mkbicub_db, BicubicBc};
use flexmesa_math::{exp10, log10};
use flexmesa_utils::is_bad;

use crate::ion_table_data_a::{ION_CELL_Z00_X40, ION_CELL_Z00_X60};
use crate::ion_table_data_b::{ION_CELL_Z00_X100, ION_CELL_Z00_X80};
use crate::ion_table_data_c::{ION_CELL_Z02_X40, ION_CELL_Z02_X60};
use crate::ion_table_data_d::{ION_CELL_Z02_X80, ION_CELL_Z04_X40};
use crate::ion_table_data_e::{ION_CELL_Z04_X60, ION_CELL_Z04_X80};
use crate::{
    IonError, IonScalar, ION_XS, ION_ZS, NUM_ION_VALS, NUM_ION_XS,
    NUM_ION_XS_FOR_Z, NUM_ION_ZS,
};

/// Precomputed 4D $(Z, X, \log Q, \log T)$ ionization spline grid.
#[derive(Debug, Clone)]
pub struct IonTableGrid {
    /// Minimum $\log Q = \log_{10}\rho - 2\log_{10} T + 12$ on the grid.
    pub log_q_min: f64,
    /// Uniform grid spacing $\Delta \log Q$.
    pub del_log_q: f64,
    /// Number of $\log Q$ grid nodes (`>= 2`).
    pub num_log_qs: usize,
    /// Minimum $\log_{10} T$ on the grid.
    pub log_t_min: f64,
    /// Uniform grid spacing $\Delta \log_{10} T$.
    pub del_log_t: f64,
    /// Number of $\log_{10} T$ grid nodes (`>= 2`).
    pub num_log_ts: usize,
    /// Bicubic spline coefficients laid out as
    /// `[iz][ix][v][(iQ * num_log_ts + jT) * 4 + c]`.
    pub coeffs: Vec<Vec<Vec<Vec<f64>>>>,
}

impl IonTableGrid {
    /// Return the built-in solar/stellar reference cell grid (`iQ = 88..89`,
    /// $\log Q \in [-2.08, -1.99]$; `jT = 40..41`, $\log T \in [4.50, 4.56]$)
    /// containing exact upstream MESA spline coefficients across $(Z, X)$.
    #[must_use]
    pub fn builtin() -> &'static Self {
        static BUILTIN: OnceLock<IonTableGrid> = OnceLock::new();
        BUILTIN.get_or_init(Self::build_builtin_cell)
    }

    /// Construct the $2 \times 2$ $(\log Q, \log T)$ reference cell from the
    /// embedded upstream MESA spline patches.
    fn build_builtin_cell() -> Self {
        let mut coeffs = vec![vec![Vec::new(); NUM_ION_XS]; NUM_ION_ZS];
        for (iz, z_slots) in coeffs.iter_mut().enumerate() {
            for (ix, slot) in z_slots.iter_mut().enumerate() {
                let patch = select_builtin_patch(iz, ix);
                let mut vars = Vec::with_capacity(NUM_ION_VALS);
                for ch in patch {
                    vars.push(ch.to_vec());
                }
                *slot = vars;
            }
        }
        Self {
            log_q_min: -2.08,
            del_log_q: 0.09,
            num_log_qs: 2,
            log_t_min: 4.50,
            del_log_t: 0.06,
            num_log_ts: 2,
            coeffs,
        }
    }

    /// Load the full 22-table $(Z, X, \log Q, \log T)$ grid from an upstream
    /// MESA `ionization_data` directory (such as `data/ionization_data`).
    ///
    /// # Errors
    /// Returns [`IonError`] if any table file is missing or malformed.
    pub fn from_data_dir(
        data_dir: &Path,
        file_prefix: &str,
        z1_suffix: &str,
    ) -> Result<Self, IonError> {
        let first_path =
            ion_table_filename(data_dir, file_prefix, z1_suffix, 0.0, 0.0);
        let (nq, lq_min, dq, nt, lt_min, dt, _) =
            parse_ion_ascii_table(&first_path)?;
        let mut coeffs = vec![vec![Vec::new(); NUM_ION_XS]; NUM_ION_ZS];
        for iz in 0..NUM_ION_ZS {
            for ix in 0..NUM_ION_XS_FOR_Z[iz] {
                let path = ion_table_filename(
                    data_dir,
                    file_prefix,
                    z1_suffix,
                    ION_ZS[iz],
                    ION_XS[ix],
                );
                let (_, _, _, _, _, _, tbl) = parse_ion_ascii_table(&path)?;
                coeffs[iz][ix] = tbl;
            }
        }
        Ok(Self {
            log_q_min: lq_min,
            del_log_q: dq,
            num_log_qs: nq,
            log_t_min: lt_min,
            del_log_t: dt,
            num_log_ts: nt,
            coeffs,
        })
    }
}

/// Select the embedded reference cell for `(iz, ix)`.
///
/// Maps each $(Z, X)$ grid index pair `(iz, ix)` to its precomputed $2 \times
/// 2 \times 4$ bicubic spline patch at `iQ = 88..89` and `jT = 40..41`.
fn select_builtin_patch(iz: usize, ix: usize) -> &'static [[f64; 16]; 28] {
    match (iz, ix) {
        (0, 0..=2) => &ION_CELL_Z00_X40,
        (0, 3) => &ION_CELL_Z00_X60,
        (0, 4) => &ION_CELL_Z00_X80,
        (0, _) => &ION_CELL_Z00_X100,
        (1, 0..=2) => &ION_CELL_Z02_X40,
        (1, 3) => &ION_CELL_Z02_X60,
        (1, _) => &ION_CELL_Z02_X80,
        (_, 0..=2) => &ION_CELL_Z04_X40,
        (_, 3) => &ION_CELL_Z04_X60,
        (_, _) => &ION_CELL_Z04_X80,
    }
}

/// Format an upstream MESA `ion_*.data` file path (`Get_ion_Table_Filenames`).
///
/// Converts mass fractions `(z, x)` to integer percentages (`00`, `02`, `04`,
/// `20`, `100`) and appends `z1_suffix` when `z == 1.0`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn ion_table_filename(
    data_dir: &Path,
    prefix: &str,
    z1_suffix: &str,
    z: f64,
    x: f64,
) -> PathBuf {
    let setstr = |v: f64| -> String {
        if v > 0.99999 {
            "100".to_string()
        } else {
            let pct = (100.0 * v + 0.5).floor() as usize;
            format!("{pct:02}")
        }
    };
    let z_str = setstr(z);
    let x_str = setstr(x);
    let suffix = if z_str == "100" { z1_suffix } else { "" };
    data_dir.join(format!("{prefix}_{z_str}z{x_str}x{suffix}.data"))
}

/// Parse an upstream MESA `ion_*.data` ASCII file and construct its 28
/// not-a-knot bicubic splines (`Load1_ion_Table` +
/// `Make_ion_Interpolation_Data`).
///
/// Reads the header row `(nt, lt_min, dt, nq, lq_min, dq)`, parses all
/// `nq * nt` rows across the 28 output channels, fits a 2D not-a-knot bicubic
/// spline to each channel via [`interp_mkbicub_db`], and replaces any
/// non-finite spline derivative entries with `0.0`
/// (`Check_ion_Interpolation_Data`).
///
/// # Errors
/// Returns [`IonError`] if the file cannot be read, has an incomplete header,
/// or fails bicubic spline construction.
#[allow(clippy::cast_precision_loss, clippy::type_complexity)]
fn parse_ion_ascii_table(
    path: &Path,
) -> Result<(usize, f64, f64, usize, f64, f64, Vec<Vec<f64>>), IonError> {
    let text =
        fs::read_to_string(path).map_err(|e| IonError::Io(e.to_string()))?;
    let mut lines = text.lines();
    let _ = lines.next();
    let hdr_line = lines.next().ok_or_else(|| IonError::TableParse {
        path: path.display().to_string(),
        detail: "missing header row".to_string(),
    })?;
    let hdr: Vec<&str> = hdr_line.split_whitespace().collect();
    if hdr.len() < 11 {
        return Err(IonError::TableParse {
            path: path.display().to_string(),
            detail: "incomplete header row".to_string(),
        });
    }
    let nt: usize = hdr[3].parse().unwrap_or(0);
    let lt_min: f64 = hdr[4].parse().unwrap_or(0.0);
    let dt: f64 = hdr[6].parse().unwrap_or(0.0);
    let nq: usize = hdr[7].parse().unwrap_or(0);
    let lq_min: f64 = hdr[8].parse().unwrap_or(0.0);
    let dq: f64 = hdr[10].parse().unwrap_or(0.0);
    let _ = lines.next();

    let mut raw = vec![vec![0.0_f64; nq * nt]; NUM_ION_VALS];
    for iq in 0..nq {
        for _ in 0..4 {
            let _ = lines.next();
        }
        for jt in 0..nt {
            let row_str = lines.next().ok_or_else(|| IonError::TableParse {
                path: path.display().to_string(),
                detail: format!("unexpected EOF at iQ={iq}, jT={jt}"),
            })?;
            let mut toks = row_str.split_whitespace();
            let _ = toks.next();
            for ch in raw.iter_mut().take(NUM_ION_VALS) {
                let val_str =
                    toks.next().ok_or_else(|| IonError::TableParse {
                        path: path.display().to_string(),
                        detail: format!("short row at iQ={iq}, jT={jt}"),
                    })?;
                ch[iq * nt + jt] = val_str.parse::<f64>().unwrap_or(0.0);
            }
        }
        if iq + 1 < nq {
            let _ = lines.next();
            let _ = lines.next();
        }
    }

    let log_qs: Vec<f64> = (0..nq).map(|i| lq_min + (i as f64) * dq).collect();
    let log_ts: Vec<f64> = (0..nt).map(|j| lt_min + (j as f64) * dt).collect();
    let mut coeffs = Vec::with_capacity(NUM_ION_VALS);
    for raw_v in raw.iter().take(NUM_ION_VALS) {
        let (c, _, _) = interp_mkbicub_db(
            &log_qs,
            &log_ts,
            raw_v,
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
        let flat: Vec<f64> = c
            .into_iter()
            .flatten()
            .map(|val| if is_bad(val) { 0.0 } else { val })
            .collect();
        coeffs.push(flat);
    }
    Ok((nq, lq_min, dq, nt, lt_min, dt, coeffs))
}

/// Locate the 0-based cell index and clamp out-of-range coordinates
/// (`Locate_logQ` / `Locate_logT`).
///
/// Applies the upstream MESA $+10^{-4}$ index bias (`INT((val - min) /
/// delta + 1d-4)`) and clamps queries outside `[min_val, max_val]` to the
/// boundary coordinate while preserving dual derivative tangents inside the
/// interior.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub(crate) fn locate_axis<T: IonScalar>(
    val: T,
    min_val: f64,
    delta: f64,
    num_pts: usize,
) -> (usize, f64, f64, T) {
    let raw_idx = ((val.val() - min_val) / delta + 1.0e-4).floor() as isize;
    if raw_idx < 0 {
        (0, min_val, min_val + delta, T::from_f64(min_val))
    } else if (raw_idx as usize) >= num_pts - 1 {
        let idx = num_pts - 2;
        let x0 = min_val + (idx as f64) * delta;
        (idx, x0, x0 + delta, T::from_f64(x0 + delta))
    } else {
        let idx = raw_idx as usize;
        let x0 = min_val + (idx as f64) * delta;
        (idx, x0, x0 + delta, val)
    }
}

/// Resolve `(linear, log10)` argument pairs (`Get_ion_Results`).
///
/// Computes `log10(v)` when only the linear argument is supplied, or
/// `exp10(lv)` when only the base-10 logarithm is supplied.
///
/// # Errors
/// Returns [`IonError::InvalidInput`] if both `lin` and `log10_val` are `None`.
pub(crate) fn resolve_pair(
    lin: Option<f64>,
    log10_val: Option<f64>,
    name: &'static str,
) -> Result<(f64, f64), IonError> {
    match (lin, log10_val) {
        (Some(v), Some(lv)) => Ok((v, lv)),
        (Some(v), None) => Ok((v, log10(v))),
        (None, Some(lv)) => Ok((exp10(lv), lv)),
        (None, None) => Err(IonError::InvalidInput {
            context: "eval_ionization",
            detail: format!("either {name} or log10_{name} must be provided"),
        }),
    }
}
