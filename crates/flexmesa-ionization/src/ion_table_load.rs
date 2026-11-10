//! Four-dimensional $(Z, X, \log Q, \log T)$ ionization table loader and
//! full built-in $175 \times 102$ bicubic spline grid (`ion_tables_load.f90`).
//!
//! Provides [`IonTableGrid`], including:
//! - [`IonTableGrid::builtin`]: Lazily initialized full $175 \times 102$
//!   $(\log Q, \log T)$ grid (`iQ = 0..174`, $\log Q \in [-10.0, 5.66]$;
//!   `jT = 0..101`, $\log T \in [2.10, 8.16]$) across all 22 $(Z, X)$
//!   tables (`ion_tables_full.bin`), containing exact upstream MESA not-a-knot
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
use rayon::prelude::*;

use crate::{
    IonError, IonScalar, ION_XS, ION_ZS, NUM_ION_VALS, NUM_ION_XS,
    NUM_ION_XS_FOR_Z, NUM_ION_ZS,
};

const ION_TABLES_FULL_BIN: &[u8] =
    include_bytes!("../data/ion_tables_full.bin");
const ION_CACHE_MAGIC: &[u8; 8] = b"FMIONCA1";

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
    /// Return the full $175 \times 102$ $(\log Q, \log T)$ ionization spline
    /// grid across all $(Z, X)$ compositions (`iQ = 0..174`, $\log Q \in
    /// [-10.0, 5.66]$; `jT = 0..101`, $\log T \in [2.10, 8.16]$).
    #[must_use]
    pub fn builtin() -> &'static Self {
        static BUILTIN: OnceLock<IonTableGrid> = OnceLock::new();
        BUILTIN.get_or_init(Self::build_builtin_full)
    }

    fn build_builtin_full() -> Self {
        let cache_path = default_ion_cache_path();
        if let Some(ref cp) = cache_path {
            if let Ok(grid) = Self::from_binary_cache(cp) {
                return grid;
            }
        }
        let grid = if let Some(dir) = resolve_ion_data_dir() {
            Self::from_data_dir(&dir, "ion", "")
                .unwrap_or_else(|_| Self::from_embedded_full())
        } else {
            Self::from_embedded_full()
        };
        if let Some(ref cp) = cache_path {
            let _ = grid.write_binary_cache(cp);
        }
        grid
    }

    /// Construct the full $175 \times 102$ $(\log Q, \log T)$ spline grid from
    /// the embedded compact binary `ion_tables_full.bin`.
    #[must_use]
    #[allow(
        clippy::assigning_clones,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::missing_panics_doc,
        clippy::needless_range_loop
    )]
    pub fn from_embedded_full() -> Self {
        assert_eq!(&ION_TABLES_FULL_BIN[0..8], b"FMION001");
        let n_tables =
            u32::from_le_bytes(ION_TABLES_FULL_BIN[8..12].try_into().unwrap())
                as usize;
        let n_vals =
            u32::from_le_bytes(ION_TABLES_FULL_BIN[12..16].try_into().unwrap())
                as usize;
        let nq =
            u32::from_le_bytes(ION_TABLES_FULL_BIN[16..20].try_into().unwrap())
                as usize;
        let nt =
            u32::from_le_bytes(ION_TABLES_FULL_BIN[20..24].try_into().unwrap())
                as usize;
        let (lq_min, dq, lt_min, dt) =
            (-10.0_f64, 0.09_f64, 2.10_f64, 0.06_f64);

        let mut pos = 24usize;
        let mut raw_tables: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_tables);
        for _ in 0..n_tables {
            let mut raw = vec![vec![0.0_f64; nq * nt]; n_vals];
            for ch in raw.iter_mut().take(n_vals) {
                for jt in 0..nt {
                    let flag = ION_TABLES_FULL_BIN[pos];
                    let m0 = i32::from_le_bytes(
                        ION_TABLES_FULL_BIN[pos + 1..pos + 5]
                            .try_into()
                            .unwrap(),
                    );
                    pos += 5;
                    let mut cur = m0;
                    ch[jt] = f64::from(cur) / 1000.0;
                    if flag == 0 {
                        let val = f64::from(cur) / 1000.0;
                        for iq in 1..nq {
                            ch[iq * nt + jt] = val;
                        }
                    } else if flag == 2 {
                        let d0 = i32::from(i16::from_le_bytes(
                            ION_TABLES_FULL_BIN[pos..pos + 2]
                                .try_into()
                                .unwrap(),
                        ));
                        pos += 2;
                        for iq in 1..nq {
                            cur += d0;
                            ch[iq * nt + jt] = f64::from(cur) / 1000.0;
                        }
                    } else {
                        for iq in 1..nq {
                            let d = i32::from(i16::from_le_bytes(
                                ION_TABLES_FULL_BIN[pos..pos + 2]
                                    .try_into()
                                    .unwrap(),
                            ));
                            pos += 2;
                            cur += d;
                            ch[iq * nt + jt] = f64::from(cur) / 1000.0;
                        }
                    }
                }
            }
            raw_tables.push(raw);
        }

        let log_qs: Vec<f64> =
            (0..nq).map(|i| lq_min + (i as f64) * dq).collect();
        let log_ts: Vec<f64> =
            (0..nt).map(|j| lt_min + (j as f64) * dt).collect();
        let spline_tables: Vec<Vec<Vec<f64>>> = raw_tables
            .into_par_iter()
            .map(|raw| {
                build_table_splines(&log_qs, &log_ts, &raw)
                    .expect("embedded ionization bicubic spline")
            })
            .collect();

        let mut coeffs = vec![vec![Vec::new(); NUM_ION_XS]; NUM_ION_ZS];
        let mut t_idx = 0usize;
        for iz in 0..NUM_ION_ZS {
            let n_xs = NUM_ION_XS_FOR_Z[iz];
            for ix in 0..n_xs {
                coeffs[iz][ix] = spline_tables[t_idx].clone();
                t_idx += 1;
            }
            for ix in n_xs..NUM_ION_XS {
                coeffs[iz][ix] = coeffs[iz][n_xs - 1].clone();
            }
        }
        Self {
            log_q_min: lq_min,
            del_log_q: dq,
            num_log_qs: nq,
            log_t_min: lt_min,
            del_log_t: dt,
            num_log_ts: nt,
            coeffs,
        }
    }

    /// Load the full 22-table $(Z, X, \log Q, \log T)$ grid from an upstream
    /// MESA `ionization_data` directory (such as `data/ionization_data`).
    ///
    /// # Errors
    /// Returns [`IonError`] if any table file is missing or malformed.
    #[allow(
        clippy::assigning_clones,
        clippy::needless_range_loop,
        clippy::type_complexity
    )]
    pub fn from_data_dir(
        data_dir: &Path,
        file_prefix: &str,
        z1_suffix: &str,
    ) -> Result<Self, IonError> {
        let mut pairs = Vec::with_capacity(22);
        for iz in 0..NUM_ION_ZS {
            for ix in 0..NUM_ION_XS_FOR_Z[iz] {
                let path = ion_table_filename(
                    data_dir,
                    file_prefix,
                    z1_suffix,
                    ION_ZS[iz],
                    ION_XS[ix],
                );
                pairs.push((iz, ix, path));
            }
        }
        let parsed: Result<
            Vec<(
                usize,
                usize,
                usize,
                f64,
                f64,
                usize,
                f64,
                f64,
                Vec<Vec<f64>>,
            )>,
            IonError,
        > = pairs
            .into_par_iter()
            .map(|(iz, ix, path)| {
                let (nq, lq_min, dq, nt, lt_min, dt, tbl) =
                    parse_ion_ascii_table(&path)?;
                Ok((iz, ix, nq, lq_min, dq, nt, lt_min, dt, tbl))
            })
            .collect();
        let parsed = parsed?;
        let (_, _, nq, lq_min, dq, nt, lt_min, dt, _) = parsed[0];
        let mut coeffs = vec![vec![Vec::new(); NUM_ION_XS]; NUM_ION_ZS];
        for (iz, ix, _, _, _, _, _, _, tbl) in parsed {
            coeffs[iz][ix] = tbl;
        }
        for iz in 0..NUM_ION_ZS {
            let n_xs = NUM_ION_XS_FOR_Z[iz];
            for ix in n_xs..NUM_ION_XS {
                coeffs[iz][ix] = coeffs[iz][n_xs - 1].clone();
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

    #[allow(clippy::cast_possible_truncation)]
    fn from_binary_cache(path: &Path) -> Result<Self, IonError> {
        let bytes = fs::read(path).map_err(|e| IonError::Io(e.to_string()))?;
        if bytes.len() < 56 || &bytes[0..8] != ION_CACHE_MAGIC {
            return Err(IonError::Io("invalid ion cache header".to_string()));
        }
        let lq_min = f64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let dq = f64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let nq = u64::from_le_bytes(bytes[24..32].try_into().unwrap()) as usize;
        let lt_min = f64::from_le_bytes(bytes[32..40].try_into().unwrap());
        let dt = f64::from_le_bytes(bytes[40..48].try_into().unwrap());
        let nt = u64::from_le_bytes(bytes[48..56].try_into().unwrap()) as usize;
        let cell_len = nq * nt * 4;
        let expected_bytes =
            56 + NUM_ION_ZS * NUM_ION_XS * NUM_ION_VALS * cell_len * 8;
        if bytes.len() != expected_bytes {
            return Err(IonError::Io("ion cache size mismatch".to_string()));
        }
        let mut pos = 56usize;
        let mut coeffs = vec![vec![Vec::new(); NUM_ION_XS]; NUM_ION_ZS];
        for z_slots in coeffs.iter_mut().take(NUM_ION_ZS) {
            for slot in z_slots.iter_mut().take(NUM_ION_XS) {
                let mut vars = Vec::with_capacity(NUM_ION_VALS);
                for _ in 0..NUM_ION_VALS {
                    let mut ch = Vec::with_capacity(cell_len);
                    for _ in 0..cell_len {
                        ch.push(f64::from_le_bytes(
                            bytes[pos..pos + 8].try_into().unwrap(),
                        ));
                        pos += 8;
                    }
                    vars.push(ch);
                }
                *slot = vars;
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

    fn write_binary_cache(&self, path: &Path) -> Result<(), IonError> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| IonError::Io(e.to_string()))?;
        }
        let cell_len = self.num_log_qs * self.num_log_ts * 4;
        let mut out = Vec::with_capacity(
            56 + NUM_ION_ZS * NUM_ION_XS * NUM_ION_VALS * cell_len * 8,
        );
        out.extend_from_slice(ION_CACHE_MAGIC);
        out.extend_from_slice(&self.log_q_min.to_le_bytes());
        out.extend_from_slice(&self.del_log_q.to_le_bytes());
        out.extend_from_slice(&(self.num_log_qs as u64).to_le_bytes());
        out.extend_from_slice(&self.log_t_min.to_le_bytes());
        out.extend_from_slice(&self.del_log_t.to_le_bytes());
        out.extend_from_slice(&(self.num_log_ts as u64).to_le_bytes());
        for iz in 0..NUM_ION_ZS {
            for ix in 0..NUM_ION_XS {
                for v in 0..NUM_ION_VALS {
                    for &val in &self.coeffs[iz][ix][v] {
                        out.extend_from_slice(&val.to_le_bytes());
                    }
                }
            }
        }
        fs::write(path, out).map_err(|e| IonError::Io(e.to_string()))
    }
}

fn resolve_ion_data_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("FLEXMESA_DATA_DIR") {
        let p = PathBuf::from(&d).join("ionization_data");
        if p.is_dir() {
            return Some(p);
        }
        let p2 = PathBuf::from(&d);
        if p2.join("ion_00z00x.data").is_file() {
            return Some(p2);
        }
    }
    if let Ok(d) = std::env::var("MESA_DIR") {
        let p = PathBuf::from(d).join("data/ionization_data");
        if p.is_dir() {
            return Some(p);
        }
    }
    let rel = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../.mesa_temp_cache/mesa_data/ionization_data");
    if rel.is_dir() {
        return Some(rel);
    }
    None
}

fn default_ion_cache_path() -> Option<PathBuf> {
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.mesa_temp_cache");
    if root.is_dir() {
        Some(root.join("bin_cache/ionization_full.bin"))
    } else {
        None
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
    let coeffs = build_table_splines(&log_qs, &log_ts, &raw)?;
    Ok((nq, lq_min, dq, nt, lt_min, dt, coeffs))
}

fn build_table_splines(
    log_qs: &[f64],
    log_ts: &[f64],
    raw: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, IonError> {
    let mut coeffs = Vec::with_capacity(NUM_ION_VALS);
    for raw_v in raw.iter().take(NUM_ION_VALS) {
        let (c, _, _) = interp_mkbicub_db(
            log_qs,
            log_ts,
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
    Ok(coeffs)
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
