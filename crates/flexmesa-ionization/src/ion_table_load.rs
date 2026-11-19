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
use flexmesa_utils::cache_envelope::{
    read_verified_cache, write_verified_cache_atomic,
};
use flexmesa_utils::sha256::{compute_sha256_bytes, Sha256Hasher};
use flexmesa_utils::{
    is_bad, resolve_flexmesa_cache_dir, resolve_flexmesa_data_dir,
};
use rayon::prelude::*;

use crate::{
    IonError, IonScalar, ION_XS, ION_ZS, NUM_ION_VALS, NUM_ION_XS,
    NUM_ION_XS_FOR_Z, NUM_ION_ZS,
};

const ION_TABLES_FULL_BIN: &[u8] =
    include_bytes!("../data/ion_tables_full.bin");
const ION_CACHE_MAGIC: &[u8; 8] = b"FMIONCA1";
const ION_CACHE_BUILDER_VERSION: u32 = 1;

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
    /// [-10.0, 5.66]$; `jT = 0..101`, $\log T \in [2.10, 8.16]$), returning an
    /// error if cache verification or table loading fails.
    ///
    /// # Errors
    /// Returns [`IonError`] if a corrupt runtime cache is encountered or ASCII
    /// source tables fail validation.
    pub fn builtin_checked() -> Result<&'static Self, IonError> {
        static BUILTIN: OnceLock<Result<IonTableGrid, IonError>> =
            OnceLock::new();
        BUILTIN
            .get_or_init(Self::build_builtin_checked)
            .as_ref()
            .map_err(Clone::clone)
    }

    /// Return the full $175 \times 102$ $(\log Q, \log T)$ ionization spline
    /// grid across all $(Z, X)$ compositions.
    ///
    /// Prefer [`Self::builtin_checked`] on fallible paths (`PyO3`).
    ///
    /// # Panics
    /// Panics if [`Self::builtin_checked`] fails due to corrupted cache or
    /// malformed external ionization tables.
    #[must_use]
    pub fn builtin() -> &'static Self {
        Self::builtin_checked()
            .expect("verified IonTableGrid builtin load failed")
    }

    fn build_builtin_checked() -> Result<Self, IonError> {
        let dir_opt = resolve_ion_data_dir();
        Self::load_or_verify_cache(dir_opt.as_deref())
    }

    /// Load `IonTableGrid` from `ascii_dir` (when provided) or embedded bytes,
    /// using the versioned `FMCENV01` runtime cache under
    /// [`resolve_flexmesa_cache_dir`] without writing inside `FLEXMESA_DATA_DIR`.
    ///
    /// # Errors
    /// Returns [`IonError`] on corrupt runtime caches or malformed inputs.
    pub fn load_or_verify_cache(
        ascii_dir: Option<&Path>,
    ) -> Result<Self, IonError> {
        let cache_root = resolve_flexmesa_cache_dir();
        Self::load_or_verify_cache_in(ascii_dir, cache_root.as_deref())
    }

    pub(crate) fn load_or_verify_cache_in(
        ascii_dir: Option<&Path>,
        cache_root: Option<&Path>,
    ) -> Result<Self, IonError> {
        let source_sha = if let Some(dir) = ascii_dir {
            compute_ion_dir_sha256(dir, "ion", "")?
        } else {
            compute_sha256_bytes(ION_TABLES_FULL_BIN)
        };
        let cache_file =
            cache_root.map(|root| ion_cache_path(root, &source_sha));
        if let Some(ref cp) = cache_file {
            if cp.is_file() {
                let payload = read_verified_cache(
                    cp,
                    ION_CACHE_BUILDER_VERSION,
                    ION_CACHE_MAGIC,
                    &source_sha,
                )
                .map_err(IonError::Io)?;
                return Self::decode_cache_payload(&payload);
            }
        }
        let grid = if let Some(dir) = ascii_dir {
            Self::from_data_dir(dir, "ion", "")?
        } else {
            Self::try_from_embedded_full()?
        };
        if let Some(ref cp) = cache_file {
            let payload = grid.encode_cache_payload()?;
            write_verified_cache_atomic(
                cp,
                ION_CACHE_BUILDER_VERSION,
                ION_CACHE_MAGIC,
                &source_sha,
                &payload,
            )
            .map_err(IonError::Io)?;
        }
        Ok(grid)
    }

    /// Construct the full $175 \times 102$ spline grid from the embedded compact
    /// binary `ion_tables_full.bin`, returning typed errors on corruption.
    ///
    /// # Errors
    /// Returns [`IonError`] if the embedded payload is malformed.
    #[allow(
        clippy::assigning_clones,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    pub fn try_from_embedded_full() -> Result<Self, IonError> {
        let (n_tables, n_vals, nq, nt) = parse_embedded_ion_header()?;
        let (lq_min, dq, lt_min, dt) =
            (-10.0_f64, 0.09_f64, 2.10_f64, 0.06_f64);

        let mut pos = 24usize;
        let mut raw_tables: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_tables);
        for _ in 0..n_tables {
            raw_tables
                .push(decode_embedded_ion_table(&mut pos, n_vals, nq, nt)?);
        }

        let log_qs: Vec<f64> =
            (0..nq).map(|i| lq_min + (i as f64) * dq).collect();
        let log_ts: Vec<f64> =
            (0..nt).map(|j| lt_min + (j as f64) * dt).collect();
        let spline_tables: Vec<Vec<Vec<f64>>> = raw_tables
            .into_par_iter()
            .map(|raw| build_table_splines(&log_qs, &log_ts, &raw))
            .collect::<Result<Vec<_>, _>>()?;

        let mut coeffs = vec![vec![Vec::new(); NUM_ION_XS]; NUM_ION_ZS];
        let mut t_idx = 0usize;
        for (z_slot, &n_xs) in coeffs.iter_mut().zip(NUM_ION_XS_FOR_Z.iter()) {
            for x_slot in z_slot.iter_mut().take(n_xs) {
                *x_slot = spline_tables[t_idx].clone();
                t_idx += 1;
            }
            for ix in n_xs..NUM_ION_XS {
                z_slot[ix] = z_slot[n_xs - 1].clone();
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

    /// Construct the full $175 \times 102$ $(\log Q, \log T)$ spline grid from
    /// the embedded compact binary `ion_tables_full.bin`.
    ///
    /// # Panics
    /// Panics if the embedded `ion_tables_full.bin` bytes are corrupted.
    #[must_use]
    pub fn from_embedded_full() -> Self {
        Self::try_from_embedded_full()
            .expect("valid embedded ion_tables_full.bin")
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

    fn decode_cache_payload(bytes: &[u8]) -> Result<Self, IonError> {
        let (lq_min, dq, nq, lt_min, dt, nt) = read_ion_cache_header(bytes)?;
        let cell_len = nq
            .checked_mul(nt)
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| {
                IonError::Io("ion cache cell_len overflow".to_string())
            })?;
        let expected_body = NUM_ION_ZS
            .checked_mul(NUM_ION_XS)
            .and_then(|v| v.checked_mul(NUM_ION_VALS))
            .and_then(|v| v.checked_mul(cell_len))
            .and_then(|v| v.checked_mul(8))
            .ok_or_else(|| {
                IonError::Io("ion cache size overflow".to_string())
            })?;
        let expected_bytes =
            56usize.checked_add(expected_body).ok_or_else(|| {
                IonError::Io("ion cache byte size overflow".to_string())
            })?;
        if bytes.len() != expected_bytes {
            return Err(IonError::Io("ion cache size mismatch".to_string()));
        }
        let mut pos = 56usize;
        let mut coeffs = vec![vec![Vec::new(); NUM_ION_XS]; NUM_ION_ZS];
        for (iz, z_slots) in coeffs.iter_mut().enumerate().take(NUM_ION_ZS) {
            for (ix, slot) in z_slots.iter_mut().enumerate().take(NUM_ION_XS) {
                let mut vars = Vec::with_capacity(NUM_ION_VALS);
                for v in 0..NUM_ION_VALS {
                    let mut ch = Vec::with_capacity(cell_len);
                    for idx in 0..cell_len {
                        let s: [u8; 8] = bytes
                            .get(pos..pos + 8)
                            .and_then(|sl| sl.try_into().ok())
                            .ok_or_else(|| {
                                IonError::Io(
                                    "unexpected EOF in ion cache payload"
                                        .to_string(),
                                )
                            })?;
                        let val = f64::from_le_bytes(s);
                        if is_bad(val) {
                            return Err(IonError::Io(format!(
                                "non-finite ion cache coefficient ({val}) at (iz={iz}, ix={ix}, var={v}, idx={idx})"
                            )));
                        }
                        ch.push(val);
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

    fn encode_cache_payload(&self) -> Result<Vec<u8>, IonError> {
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
        for z_plane in self.coeffs.iter().take(NUM_ION_ZS) {
            for x_row in z_plane.iter().take(NUM_ION_XS) {
                for vals in x_row.iter().take(NUM_ION_VALS) {
                    if vals.len() != cell_len {
                        return Err(IonError::Io(
                            "inconsistent ion grid cell_len during cache encode"
                                .to_string(),
                        ));
                    }
                    for &val in vals {
                        out.extend_from_slice(&val.to_le_bytes());
                    }
                }
            }
        }
        Ok(out)
    }
}

fn read_ion_cache_header(
    bytes: &[u8],
) -> Result<(f64, f64, usize, f64, f64, usize), IonError> {
    if bytes.len() < 56 || &bytes[0..8] != ION_CACHE_MAGIC {
        return Err(IonError::Io("invalid ion cache header".to_string()));
    }
    let read_f64 = |off: usize| -> Result<f64, IonError> {
        let s: [u8; 8] = bytes
            .get(off..off + 8)
            .and_then(|v| v.try_into().ok())
            .ok_or_else(|| {
                IonError::Io("truncated ion cache header".to_string())
            })?;
        let val = f64::from_le_bytes(s);
        if is_bad(val) {
            return Err(IonError::Io(format!(
                "non-finite ion cache header f64 at offset {off}"
            )));
        }
        Ok(val)
    };
    let read_u64 = |off: usize| -> Result<usize, IonError> {
        let s: [u8; 8] = bytes
            .get(off..off + 8)
            .and_then(|v| v.try_into().ok())
            .ok_or_else(|| {
                IonError::Io("truncated ion cache header".to_string())
            })?;
        usize::try_from(u64::from_le_bytes(s)).map_err(|_| {
            IonError::Io("ion cache dimension overflow".to_string())
        })
    };
    let lq_min = read_f64(8)?;
    let dq = read_f64(16)?;
    let nq = read_u64(24)?;
    let lt_min = read_f64(32)?;
    let dt = read_f64(40)?;
    let nt = read_u64(48)?;
    if nq < 2 || nt < 2 || dq <= 0.0 || dt <= 0.0 {
        return Err(IonError::Io("invalid ion cache grid bounds".to_string()));
    }
    Ok((lq_min, dq, nq, lt_min, dt, nt))
}

fn resolve_ion_data_dir() -> Option<PathBuf> {
    let d = resolve_flexmesa_data_dir();
    let sub = d.join("ionization_data");
    if sub.is_dir() && sub.join("ion_00z00x.data").is_file() {
        return Some(sub);
    }
    if d.join("ion_00z00x.data").is_file() {
        return Some(d);
    }
    if let Ok(m) = std::env::var("MESA_DIR") {
        let p = PathBuf::from(m).join("data").join("ionization_data");
        if p.is_dir() && p.join("ion_00z00x.data").is_file() {
            return Some(p);
        }
    }
    None
}

fn ion_cache_path(cache_root: &Path, source_sha: &str) -> PathBuf {
    cache_root
        .join("ionization")
        .join(format!("ionization_full.{}.bin", &source_sha[..16]))
}

fn parse_embedded_ion_header() -> Result<(usize, usize, usize, usize), IonError>
{
    if ION_TABLES_FULL_BIN.get(..8) != Some(b"FMION001".as_slice())
        || ION_TABLES_FULL_BIN.len() < 24
    {
        return Err(IonError::TableParse {
            path: "ion_tables_full.bin".to_string(),
            detail: "invalid FMION001 header".to_string(),
        });
    }
    let u32_hdr = |off: usize| -> Result<usize, IonError> {
        let b: [u8; 4] = ION_TABLES_FULL_BIN
            .get(off..off + 4)
            .and_then(|s| s.try_into().ok())
            .ok_or_else(|| IonError::TableParse {
                path: "ion_tables_full.bin".to_string(),
                detail: format!("truncated header at {off}"),
            })?;
        Ok(u32::from_le_bytes(b) as usize)
    };
    Ok((u32_hdr(8)?, u32_hdr(12)?, u32_hdr(16)?, u32_hdr(20)?))
}

fn decode_embedded_ion_table(
    pos: &mut usize,
    n_vals: usize,
    nq: usize,
    nt: usize,
) -> Result<Vec<Vec<f64>>, IonError> {
    let read_i32 = |p: &mut usize| -> Result<i32, IonError> {
        let b: [u8; 4] = ION_TABLES_FULL_BIN
            .get(*p..*p + 4)
            .and_then(|s| s.try_into().ok())
            .ok_or_else(|| IonError::TableParse {
                path: "ion_tables_full.bin".to_string(),
                detail: "unexpected EOF reading i32".to_string(),
            })?;
        *p += 4;
        Ok(i32::from_le_bytes(b))
    };
    let read_i16 = |p: &mut usize| -> Result<i16, IonError> {
        let b: [u8; 2] = ION_TABLES_FULL_BIN
            .get(*p..*p + 2)
            .and_then(|s| s.try_into().ok())
            .ok_or_else(|| IonError::TableParse {
                path: "ion_tables_full.bin".to_string(),
                detail: "unexpected EOF reading i16".to_string(),
            })?;
        *p += 2;
        Ok(i16::from_le_bytes(b))
    };

    let mut raw = vec![vec![0.0_f64; nq * nt]; n_vals];
    for ch in raw.iter_mut().take(n_vals) {
        for jt in 0..nt {
            let flag = *ION_TABLES_FULL_BIN.get(*pos).ok_or_else(|| {
                IonError::TableParse {
                    path: "ion_tables_full.bin".to_string(),
                    detail: "unexpected EOF reading flag".to_string(),
                }
            })?;
            *pos += 1;
            let mut cur = read_i32(pos)?;
            ch[jt] = f64::from(cur) / 1000.0;
            if flag == 0 {
                let val = ch[jt];
                for iq in 1..nq {
                    ch[iq * nt + jt] = val;
                }
            } else if flag == 2 {
                let d0 = i32::from(read_i16(pos)?);
                for iq in 1..nq {
                    cur += d0;
                    ch[iq * nt + jt] = f64::from(cur) / 1000.0;
                }
            } else {
                for iq in 1..nq {
                    let d = i32::from(read_i16(pos)?);
                    cur += d;
                    ch[iq * nt + jt] = f64::from(cur) / 1000.0;
                }
            }
        }
    }
    Ok(raw)
}

fn compute_ion_dir_sha256(
    data_dir: &Path,
    file_prefix: &str,
    z1_suffix: &str,
) -> Result<String, IonError> {
    let mut hasher = Sha256Hasher::new();
    for (&z, &n_xs) in ION_ZS.iter().zip(NUM_ION_XS_FOR_Z.iter()) {
        for &x in ION_XS.iter().take(n_xs) {
            let p = ion_table_filename(data_dir, file_prefix, z1_suffix, z, x);
            let bytes = fs::read(&p).map_err(|e| {
                IonError::Io(format!("failed to read {}: {e}", p.display()))
            })?;
            hasher.update(&(bytes.len() as u64).to_le_bytes());
            hasher.update(&bytes);
        }
    }
    Ok(hasher.finalize_hex())
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
    let parse_hdr_usize =
        |idx: usize, label: &str| -> Result<usize, IonError> {
            hdr[idx].parse::<usize>().map_err(|e| IonError::TableParse {
                path: path.display().to_string(),
                detail: format!("invalid {label} '{}': {e}", hdr[idx]),
            })
        };
    let parse_hdr_f64 = |idx: usize, label: &str| -> Result<f64, IonError> {
        let val =
            hdr[idx].parse::<f64>().map_err(|e| IonError::TableParse {
                path: path.display().to_string(),
                detail: format!("invalid {label} '{}': {e}", hdr[idx]),
            })?;
        if is_bad(val) {
            return Err(IonError::TableParse {
                path: path.display().to_string(),
                detail: format!("non-finite {label} '{val}'"),
            });
        }
        Ok(val)
    };
    let nt = parse_hdr_usize(3, "num_logTs")?;
    let lt_min = parse_hdr_f64(4, "logT_min")?;
    let dt = parse_hdr_f64(6, "del_logT")?;
    let nq = parse_hdr_usize(7, "num_logQs")?;
    let lq_min = parse_hdr_f64(8, "logQ_min")?;
    let dq = parse_hdr_f64(10, "del_logQ")?;
    if nt < 2 || nq < 2 || dt <= 0.0 || dq <= 0.0 {
        return Err(IonError::TableParse {
            path: path.display().to_string(),
            detail: format!("invalid grid ({nq} x {nt}, dq={dq}, dt={dt})"),
        });
    }
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
            for (v, ch) in raw.iter_mut().enumerate().take(NUM_ION_VALS) {
                let val_str =
                    toks.next().ok_or_else(|| IonError::TableParse {
                        path: path.display().to_string(),
                        detail: format!("short row at iQ={iq}, jT={jt}"),
                    })?;
                let parsed = val_str.parse::<f64>().map_err(|e| {
                    IonError::TableParse {
                        path: path.display().to_string(),
                        detail: format!(
                            "invalid float '{val_str}' var={v} at iQ={iq}, jT={jt}: {e}"
                        ),
                    }
                })?;
                if is_bad(parsed) {
                    return Err(IonError::TableParse {
                        path: path.display().to_string(),
                        detail: format!(
                            "non-finite value ({parsed}) var={v} at iQ={iq}, jT={jt}"
                        ),
                    });
                }
                ch[iq * nt + jt] = parsed;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ion_cache_isolates_from_data_dir_and_rejects_corruption() {
        let dir = std::env::temp_dir()
            .join(format!("flexmesa_ion_cache_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let data_dir = dir.join("data");
        let cache_dir = dir.join("cache");
        let cache_v1 = cache_dir.join("v1");
        fs::create_dir_all(&data_dir).expect("mkdir data");

        let g_uncached = IonTableGrid::load_or_verify_cache_in(None, None)
            .expect("build embedded ion grid with cache disabled");
        assert!(!cache_dir.exists());
        assert!(!data_dir.join("ionization_full.bin").exists());

        let g1 = IonTableGrid::load_or_verify_cache_in(None, Some(&cache_v1))
            .expect("build and cache embedded ion grid");
        assert_eq!(g_uncached.num_log_qs, g1.num_log_qs);
        assert!(!data_dir.join("ionization_full.bin").exists());

        let ion_cache_dir = cache_v1.join("ionization");
        let entries: Vec<PathBuf> = fs::read_dir(&ion_cache_dir)
            .expect("read ion cache dir")
            .filter_map(|e| e.ok().map(|x| x.path()))
            .collect();
        assert_eq!(entries.len(), 1);

        let g2 = IonTableGrid::load_or_verify_cache_in(None, Some(&cache_v1))
            .expect("reload verified ion cache");
        assert_eq!(g1.num_log_qs, g2.num_log_qs);

        // Corrupt 1 byte in runtime cache -> must error instead of panicking
        let mut corrupt = fs::read(&entries[0]).expect("read ion cache");
        let last = corrupt.len() - 1;
        corrupt[last] ^= 0x01;
        fs::write(&entries[0], &corrupt).expect("write corrupt cache");
        let err = IonTableGrid::load_or_verify_cache_in(None, Some(&cache_v1))
            .expect_err("corrupt ion cache must error");
        assert!(
            matches!(err, IonError::Io(_)),
            "expected IonError::Io, got {err:?}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_ion_decode_cache_payload_rejects_malformed_inputs_without_panic() {
        assert!(matches!(
            IonTableGrid::decode_cache_payload(&[]),
            Err(IonError::Io(_))
        ));
        assert!(matches!(
            IonTableGrid::decode_cache_payload(b"BADMAGIC_PADDING"),
            Err(IonError::Io(_))
        ));
    }
}
