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
//!   that validates every table header against upstream `ion_read_sizes` and
//!   `Load1_ion_Table` checks, builds global $175 \times 102$ not-a-knot
//!   bicubic splines via [`flexmesa_interp2d::interp_mkbicub_db`], and
//!   sanitizes non-finite higher-order derivatives
//!   (`Check_ion_Interpolation_Data`).
//!
//! The embedded `ion_tables_full.bin` (`FMION001`) is rebuilt from the MESA
//! `26.4.1` ASCII tables by `scripts/data/build_ion_tables.py`, which also
//! writes the provenance manifest `ion_tables_full.toml` (SHA-256, source
//! tables, `Z = 1` suffix, and grid bounds) read by
//! [`IonTableGrid::try_from_embedded_full`].

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use flexmesa_interp2d::{BicubicBc, interp_mkbicub_db};
use flexmesa_math::{exp10, log10};
use flexmesa_utils::cache_envelope::{
    read_verified_cache, write_verified_cache_atomic,
};
use flexmesa_utils::sha256::{Sha256Hasher, compute_sha256_bytes};
use flexmesa_utils::{
    is_bad, resolve_flexmesa_cache_dir, resolve_flexmesa_data_dir,
};
use rayon::prelude::*;

use crate::{
    ION_XS, ION_ZS, IonError, IonScalar, NUM_ION_VALS, NUM_ION_XS,
    NUM_ION_XS_FOR_Z, NUM_ION_ZS,
};

const ION_TABLES_FULL_BIN: &[u8] =
    include_bytes!("../data/ion_tables_full.bin");
const ION_TABLES_FULL_MANIFEST: &str =
    include_str!("../data/ion_tables_full.toml");
const ION_TABLES_FULL_NAME: &str = "ion_tables_full.bin";
const ION_TABLES_MANIFEST_NAME: &str = "ion_tables_full.toml";
const FMION_MAGIC: &[u8; 8] = b"FMION001";
const FMION_HEADER_BYTES: usize = 24;
const ION_CACHE_MAGIC: &[u8; 8] = b"FMIONCA1";
const ION_CACHE_BUILDER_VERSION: u32 = 1;

/// Minimum accepted `ion_*.data` format version (`min_version = 49`,
/// `ionization/public/ionization_def.f90:91`).
pub const ION_MIN_VERSION: i64 = 49;

/// Header comparison tolerance of `Load1_ion_Table`.
///
/// Upstream declares `real(dp), parameter :: tiny = 1e-6`
/// (`ionization/private/ion_tables_load.f90:236`); the default-real literal
/// `1e-6` is rounded to single precision before widening, giving
/// `f64::from(1.0e-6_f32)`.
pub const ION_HEADER_TINY: f64 = 9.999_999_974_752_427e-7;

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
        let dir_opt = resolve_ion_data_dir()?;
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

    #[doc(hidden)]
    pub fn load_or_verify_cache_in(
        ascii_dir: Option<&Path>,
        cache_root: Option<&Path>,
    ) -> Result<Self, IonError> {
        let source_sha = if let Some(dir) = ascii_dir {
            compute_ion_dir_sha256(dir, "ion", "")?
        } else {
            let sha = compute_sha256_bytes(ION_TABLES_FULL_BIN);
            let manifest = embedded_ion_manifest()?;
            if sha != manifest.sha256 {
                return Err(IonError::TableParse {
                    path: ION_TABLES_FULL_NAME.to_string(),
                    detail: format!(
                        "SHA-256 {sha} differs from {} recorded in {}",
                        manifest.sha256, ION_TABLES_MANIFEST_NAME
                    ),
                });
            }
            sha
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
    /// Grid sizes and bounds come from the provenance manifest
    /// `ion_tables_full.toml` written by `scripts/data/build_ion_tables.py`
    /// from the MESA `26.4.1` ASCII table headers, and are cross-checked
    /// against the `FMION001` header.
    ///
    /// # Errors
    /// Returns [`IonError`] if the embedded payload or manifest is malformed.
    pub fn try_from_embedded_full() -> Result<Self, IonError> {
        Self::from_fmion001_bytes(ION_TABLES_FULL_BIN, embedded_ion_manifest()?)
    }

    #[allow(
        clippy::assigning_clones,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    fn from_fmion001_bytes(
        bytes: &[u8],
        manifest: &EmbeddedIonManifest,
    ) -> Result<Self, IonError> {
        let (n_tables, n_vals, nq, nt) = parse_fmion001_header(bytes)?;
        manifest.check_fmion001(bytes.len(), n_tables, n_vals, nq, nt)?;
        let (lq_min, dq, lt_min, dt) = (
            manifest.log_q_min,
            manifest.del_log_q,
            manifest.log_t_min,
            manifest.del_log_t,
        );
        let scale = f64::from(manifest.value_scale);

        let mut pos = FMION_HEADER_BYTES;
        let mut raw_tables: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_tables);
        for _ in 0..n_tables {
            raw_tables.push(decode_fmion001_table(
                bytes, &mut pos, n_vals, nq, nt, scale,
            )?);
        }
        if pos != bytes.len() {
            return Err(fmion_err(format!(
                "{} trailing bytes after the last table",
                bytes.len() - pos
            )));
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
    /// Before reading any table body, every header is checked as upstream
    /// does: the first table (`ion_00z00x.data`) must have `version >=
    /// min_version` (`49`, `ionization/private/ion_tables_load.f90:54-69`),
    /// and each table must match the expected `(X, Z)` of its file name and
    /// the first table's version and grid (`num_logQs`, `num_logTs`,
    /// `logT_min`, `logT_max`, `del_logT`, `logQ_min`, `logQ_max`,
    /// `del_logQ`) within `tiny` (`ion_tables_load.f90:236, 259-290`).
    ///
    /// # Errors
    /// Returns [`IonError::TableParse`] naming the file and header field on
    /// any header violation, and [`IonError`] if any table file is missing or
    /// malformed.
    #[allow(clippy::assigning_clones, clippy::needless_range_loop)]
    pub fn from_data_dir(
        data_dir: &Path,
        file_prefix: &str,
        z1_suffix: &str,
    ) -> Result<Self, IonError> {
        validate_ion_data_dir(data_dir, file_prefix, z1_suffix)?;
        let mut entries = Vec::with_capacity(22);
        for iz in 0..NUM_ION_ZS {
            for ix in 0..NUM_ION_XS_FOR_Z[iz] {
                let path = ion_table_filename(
                    data_dir,
                    file_prefix,
                    z1_suffix,
                    ION_ZS[iz],
                    ION_XS[ix],
                );
                let hdr = read_ion_ascii_header(&path)?;
                entries.push((iz, ix, path, hdr));
            }
        }
        let headers: Vec<(&Path, IonAsciiHeader, f64, f64)> = entries
            .iter()
            .map(|(iz, ix, p, h)| (p.as_path(), *h, ION_ZS[*iz], ION_XS[*ix]))
            .collect();
        let grid = validate_ion_headers(&headers)?;
        let parsed: Vec<(usize, usize, Vec<Vec<f64>>)> = entries
            .into_par_iter()
            .map(|(iz, ix, path, _)| {
                Ok((iz, ix, parse_ion_ascii_table(&path, &grid)?))
            })
            .collect::<Result<_, IonError>>()?;
        let mut coeffs = vec![vec![Vec::new(); NUM_ION_XS]; NUM_ION_ZS];
        for (iz, ix, tbl) in parsed {
            coeffs[iz][ix] = tbl;
        }
        for iz in 0..NUM_ION_ZS {
            let n_xs = NUM_ION_XS_FOR_Z[iz];
            for ix in n_xs..NUM_ION_XS {
                coeffs[iz][ix] = coeffs[iz][n_xs - 1].clone();
            }
        }
        Ok(Self {
            log_q_min: grid.log_q_min,
            del_log_q: grid.del_log_q,
            num_log_qs: grid.num_log_qs,
            log_t_min: grid.log_t_min,
            del_log_t: grid.del_log_t,
            num_log_ts: grid.num_log_ts,
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

    #[doc(hidden)]
    pub fn encode_cache_payload(&self) -> Result<Vec<u8>, IonError> {
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

fn validate_ion_data_dir(
    data_dir: &Path,
    file_prefix: &str,
    z1_suffix: &str,
) -> Result<(), IonError> {
    if !data_dir.is_dir() {
        return Err(IonError::Io(format!(
            "ionization data path {} is not a directory",
            data_dir.display()
        )));
    }
    for (&z, &n_xs) in ION_ZS.iter().zip(NUM_ION_XS_FOR_Z.iter()) {
        for &x in ION_XS.iter().take(n_xs) {
            let expected =
                ion_table_filename(data_dir, file_prefix, z1_suffix, z, x);
            if !expected.is_file() {
                return Err(IonError::Io(format!(
                    "incomplete ionization data directory {}: missing expected table file {}",
                    data_dir.display(),
                    expected.display()
                )));
            }
        }
    }
    Ok(())
}

fn has_any_ion_table_entry(
    data_dir: &Path,
    file_prefix: &str,
    z1_suffix: &str,
) -> bool {
    for (&z, &n_xs) in ION_ZS.iter().zip(NUM_ION_XS_FOR_Z.iter()) {
        for &x in ION_XS.iter().take(n_xs) {
            let p = ion_table_filename(data_dir, file_prefix, z1_suffix, z, x);
            if fs::symlink_metadata(&p).is_ok() {
                return true;
            }
        }
    }
    false
}

/// Resolve an external `ionization_data` directory from `flex_data_dir` or
/// `mesa_dir`, returning a loud error (`IonError::Io`) naming the directory
/// and missing table file whenever an `ionization_data` directory exists on
/// disk in an incomplete state (`ionization/private/ion_tables_load.f90:54-65`,
/// `89-104`, `146-175`).
pub(crate) fn resolve_ion_data_dir_from(
    flex_data_dir: &Path,
    mesa_dir: Option<&Path>,
) -> Result<Option<PathBuf>, IonError> {
    let sub = flex_data_dir.join("ionization_data");
    if fs::symlink_metadata(&sub).is_ok() {
        validate_ion_data_dir(&sub, "ion", "")?;
        return Ok(Some(sub));
    }
    let named_ion_dir = flex_data_dir.file_name().and_then(|s| s.to_str())
        == Some("ionization_data")
        && fs::symlink_metadata(flex_data_dir).is_ok();
    if named_ion_dir || has_any_ion_table_entry(flex_data_dir, "ion", "") {
        validate_ion_data_dir(flex_data_dir, "ion", "")?;
        return Ok(Some(flex_data_dir.to_path_buf()));
    }
    if let Some(m) = mesa_dir.filter(|p| !p.as_os_str().is_empty()) {
        let p = m.join("data").join("ionization_data");
        if fs::symlink_metadata(&p).is_ok() {
            validate_ion_data_dir(&p, "ion", "")?;
            return Ok(Some(p));
        }
    }
    Ok(None)
}

fn resolve_ion_data_dir() -> Result<Option<PathBuf>, IonError> {
    let d = resolve_flexmesa_data_dir();
    let mesa = std::env::var_os("MESA_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from);
    resolve_ion_data_dir_from(&d, mesa.as_deref())
}

fn ion_cache_path(cache_root: &Path, source_sha: &str) -> PathBuf {
    cache_root
        .join("ionization")
        .join(format!("ionization_full.{}.bin", &source_sha[..16]))
}

fn fmion_err(detail: String) -> IonError {
    IonError::TableParse {
        path: ION_TABLES_FULL_NAME.to_string(),
        detail,
    }
}

/// Parse the 24-byte `FMION001` header `(num_tables, num_vals, num_logQs,
/// num_logTs)`.
fn parse_fmion001_header(
    bytes: &[u8],
) -> Result<(usize, usize, usize, usize), IonError> {
    if bytes.len() < FMION_HEADER_BYTES
        || bytes.get(..8) != Some(FMION_MAGIC.as_slice())
    {
        return Err(fmion_err("invalid FMION001 header".to_string()));
    }
    let u32_hdr = |off: usize| -> usize {
        let b: [u8; 4] =
            [bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]];
        u32::from_le_bytes(b) as usize
    };
    Ok((u32_hdr(8), u32_hdr(12), u32_hdr(16), u32_hdr(20)))
}

/// Decode one `(28, nq, nt)` table of signed integer thousandths.
///
/// Each `logT` column starts with a flag byte and the `i32` value at
/// `iQ = 0`: flag `0` repeats it for every `logQ`, flag `1` is followed by
/// `nq - 1` `i16` successive differences, and flag `2` by a single `i16`
/// constant difference. Values are `f64::from(n) / scale`, which for the
/// MESA `%.3f` tables (`scale = 1000`) equals the correctly rounded decimal
/// parse of the ASCII token.
fn decode_fmion001_table(
    bytes: &[u8],
    pos: &mut usize,
    n_vals: usize,
    nq: usize,
    nt: usize,
    scale: f64,
) -> Result<Vec<Vec<f64>>, IonError> {
    fn take<const N: usize>(
        bytes: &[u8],
        p: &mut usize,
    ) -> Result<[u8; N], IonError> {
        let b: [u8; N] = bytes
            .get(*p..*p + N)
            .and_then(|s| s.try_into().ok())
            .ok_or_else(|| {
                fmion_err(format!("unexpected EOF at byte offset {p}"))
            })?;
        *p += N;
        Ok(b)
    }
    let step = |cur: i32, d: i16| -> Result<i32, IonError> {
        cur.checked_add(i32::from(d)).ok_or_else(|| {
            fmion_err("i32 overflow while summing differences".to_string())
        })
    };

    let mut raw = vec![vec![0.0_f64; nq * nt]; n_vals];
    for ch in &mut raw {
        for jt in 0..nt {
            let [flag] = take::<1>(bytes, pos)?;
            let mut cur = i32::from_le_bytes(take::<4>(bytes, pos)?);
            ch[jt] = f64::from(cur) / scale;
            match flag {
                0 => {
                    let val = ch[jt];
                    for iq in 1..nq {
                        ch[iq * nt + jt] = val;
                    }
                }
                1 => {
                    for iq in 1..nq {
                        let d = i16::from_le_bytes(take::<2>(bytes, pos)?);
                        cur = step(cur, d)?;
                        ch[iq * nt + jt] = f64::from(cur) / scale;
                    }
                }
                2 => {
                    let d0 = i16::from_le_bytes(take::<2>(bytes, pos)?);
                    for iq in 1..nq {
                        cur = step(cur, d0)?;
                        ch[iq * nt + jt] = f64::from(cur) / scale;
                    }
                }
                other => {
                    return Err(fmion_err(format!(
                        "unknown column flag {other} at byte offset {}",
                        *pos - 5
                    )));
                }
            }
        }
    }
    Ok(raw)
}

/// Flat top-level fields of the `ion_tables_full.toml` provenance manifest
/// written by `scripts/data/build_ion_tables.py`.
#[derive(Debug, Clone, PartialEq)]
struct EmbeddedIonManifest {
    /// SHA-256 of `ion_tables_full.bin`.
    sha256: String,
    /// Byte length of `ion_tables_full.bin`.
    size_bytes: usize,
    /// `Z = 1` table suffix used for the build (`""`, `_CO_0`, `_CO_1`).
    z1_suffix: String,
    /// Upstream `ion_version` of the source tables.
    ion_version: i64,
    num_tables: usize,
    num_vals: usize,
    num_log_qs: usize,
    log_q_min: f64,
    log_q_max: f64,
    del_log_q: f64,
    num_log_ts: usize,
    log_t_min: f64,
    log_t_max: f64,
    del_log_t: f64,
    /// Integer scale of the stored values (`1000` for `%.3f` tables).
    value_scale: u32,
}

impl EmbeddedIonManifest {
    /// Parse the flat `key = value` lines ahead of the first `[table]`.
    fn parse(text: &str) -> Result<Self, IonError> {
        let err = |detail: String| IonError::TableParse {
            path: ION_TABLES_MANIFEST_NAME.to_string(),
            detail,
        };
        let mut kv: Vec<(&str, &str)> = Vec::new();
        for line in text.lines().map(str::trim) {
            if line.starts_with('[') {
                break;
            }
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (k, v) = line
                .split_once('=')
                .ok_or_else(|| err(format!("malformed line '{line}'")))?;
            kv.push((k.trim(), v.trim()));
        }
        let get = |key: &str| -> Result<&str, IonError> {
            kv.iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| *v)
                .ok_or_else(|| err(format!("missing key '{key}'")))
        };
        let get_str = |key: &str| -> Result<String, IonError> {
            let v = get(key)?;
            v.strip_prefix('"')
                .and_then(|s| s.strip_suffix('"'))
                .map(str::to_string)
                .ok_or_else(|| err(format!("key '{key}' is not a string")))
        };
        let get_num = |key: &str| -> Result<f64, IonError> {
            let v = get(key)?;
            let val = v.parse::<f64>().map_err(|e| {
                err(format!("invalid number for '{key}' ({v}): {e}"))
            })?;
            if is_bad(val) {
                return Err(err(format!("non-finite '{key}' ({v})")));
            }
            Ok(val)
        };
        let get_int = |key: &str| -> Result<i64, IonError> {
            let v = get(key)?;
            v.parse::<i64>().map_err(|e| {
                err(format!("invalid integer for '{key}' ({v}): {e}"))
            })
        };
        let get_usize = |key: &str| -> Result<usize, IonError> {
            usize::try_from(get_int(key)?)
                .map_err(|_| err(format!("negative '{key}'")))
        };
        let format = get_str("format")?;
        if format != "FMION001" {
            return Err(err(format!("unsupported format '{format}'")));
        }
        let value_scale = u32::try_from(get_int("value_scale")?)
            .ok()
            .filter(|&s| s > 0)
            .ok_or_else(|| err("value_scale must be positive".to_string()))?;
        Ok(Self {
            sha256: get_str("sha256")?,
            size_bytes: get_usize("size_bytes")?,
            z1_suffix: get_str("z1_suffix")?,
            ion_version: get_int("ion_version")?,
            num_tables: get_usize("num_tables")?,
            num_vals: get_usize("num_vals")?,
            num_log_qs: get_usize("num_log_qs")?,
            log_q_min: get_num("log_q_min")?,
            log_q_max: get_num("log_q_max")?,
            del_log_q: get_num("del_log_q")?,
            num_log_ts: get_usize("num_log_ts")?,
            log_t_min: get_num("log_t_min")?,
            log_t_max: get_num("log_t_max")?,
            del_log_t: get_num("del_log_t")?,
            value_scale,
        })
    }

    /// Cross-check an `FMION001` payload against this manifest.
    fn check_fmion001(
        &self,
        len: usize,
        n_tables: usize,
        n_vals: usize,
        nq: usize,
        nt: usize,
    ) -> Result<(), IonError> {
        let expected_tables: usize = NUM_ION_XS_FOR_Z.iter().sum();
        let checks = [
            ("size_bytes", self.size_bytes, len),
            ("num_tables", self.num_tables, n_tables),
            ("num_tables (ion_Zs x ion_Xs)", expected_tables, n_tables),
            ("num_vals", self.num_vals, n_vals),
            ("num_vals (NUM_ION_VALS)", NUM_ION_VALS, n_vals),
            ("num_log_qs", self.num_log_qs, nq),
            ("num_log_ts", self.num_log_ts, nt),
        ];
        for (field, want, got) in checks {
            if want != got {
                return Err(fmion_err(format!(
                    "{field} = {got} differs from {want} in \
                     {ION_TABLES_MANIFEST_NAME}"
                )));
            }
        }
        if nq < 2 || nt < 2 || self.del_log_q <= 0.0 || self.del_log_t <= 0.0 {
            return Err(fmion_err(format!(
                "invalid grid ({nq} x {nt}, del_log_q={}, del_log_t={})",
                self.del_log_q, self.del_log_t
            )));
        }
        if self.ion_version < ION_MIN_VERSION {
            return Err(fmion_err(format!(
                "ion_version = {} is below min_version = {ION_MIN_VERSION}",
                self.ion_version
            )));
        }
        Ok(())
    }
}

/// Return the parsed embedded `ion_tables_full.toml` manifest.
fn embedded_ion_manifest() -> Result<&'static EmbeddedIonManifest, IonError> {
    static MANIFEST: OnceLock<Result<EmbeddedIonManifest, IonError>> =
        OnceLock::new();
    MANIFEST
        .get_or_init(|| EmbeddedIonManifest::parse(ION_TABLES_FULL_MANIFEST))
        .as_ref()
        .map_err(Clone::clone)
}

/// Second header row of an upstream `ion_*.data` table (`Load1_ion_Table`,
/// `ionization/private/ion_tables_load.f90:250-252`).
#[derive(Debug, Clone, Copy, PartialEq)]
struct IonAsciiHeader {
    version: i64,
    x: f64,
    z: f64,
    num_log_ts: usize,
    log_t_min: f64,
    log_t_max: f64,
    del_log_t: f64,
    num_log_qs: usize,
    log_q_min: f64,
    log_q_max: f64,
    del_log_q: f64,
}

/// Read the 11-field header row (second line) of an `ion_*.data` table.
fn read_ion_ascii_header(path: &Path) -> Result<IonAsciiHeader, IonError> {
    let file = fs::File::open(path).map_err(|e| {
        IonError::Io(format!("failed to open {}: {e}", path.display()))
    })?;
    let mut lines = BufReader::new(file).lines();
    let _ = lines.next();
    let line = lines
        .next()
        .transpose()
        .map_err(|e| {
            IonError::Io(format!("failed to read {}: {e}", path.display()))
        })?
        .ok_or_else(|| IonError::TableParse {
            path: path.display().to_string(),
            detail: "missing header row".to_string(),
        })?;
    parse_ion_header_line(path, &line)
}

/// Parse `version X Z num_logTs logT_min logT_max del_logT num_logQs
/// logQ_min logQ_max del_logQ`, rejecting non-finite values.
fn parse_ion_header_line(
    path: &Path,
    line: &str,
) -> Result<IonAsciiHeader, IonError> {
    let err = |detail: String| IonError::TableParse {
        path: path.display().to_string(),
        detail,
    };
    let hdr: Vec<&str> = line.split_whitespace().collect();
    if hdr.len() < 11 {
        return Err(err("incomplete header row".to_string()));
    }
    let int = |idx: usize, label: &str| -> Result<i64, IonError> {
        hdr[idx].parse::<i64>().map_err(|e| {
            err(format!("invalid header field {label} '{}': {e}", hdr[idx]))
        })
    };
    let count = |idx: usize, label: &str| -> Result<usize, IonError> {
        usize::try_from(int(idx, label)?)
            .map_err(|_| err(format!("negative header field {label}")))
    };
    let real = |idx: usize, label: &str| -> Result<f64, IonError> {
        let val = hdr[idx].parse::<f64>().map_err(|e| {
            err(format!("invalid header field {label} '{}': {e}", hdr[idx]))
        })?;
        if is_bad(val) {
            return Err(err(format!(
                "non-finite header field {label} '{val}'"
            )));
        }
        Ok(val)
    };
    let h = IonAsciiHeader {
        version: int(0, "version")?,
        x: real(1, "X")?,
        z: real(2, "Z")?,
        num_log_ts: count(3, "num_logTs")?,
        log_t_min: real(4, "logT_min")?,
        log_t_max: real(5, "logT_max")?,
        del_log_t: real(6, "del_logT")?,
        num_log_qs: count(7, "num_logQs")?,
        log_q_min: real(8, "logQ_min")?,
        log_q_max: real(9, "logQ_max")?,
        del_log_q: real(10, "del_logQ")?,
    };
    if h.num_log_ts < 2
        || h.num_log_qs < 2
        || h.del_log_t <= 0.0
        || h.del_log_q <= 0.0
    {
        return Err(err(format!(
            "invalid grid ({} x {}, del_logQ={}, del_logT={})",
            h.num_log_qs, h.num_log_ts, h.del_log_q, h.del_log_t
        )));
    }
    Ok(h)
}

/// Apply the upstream header checks to the 22 tables in load order.
///
/// `entries` holds `(path, header, expected_Z, expected_X)`. The first
/// table supplies the reference version and grid (`ion_read_sizes`,
/// `ionization/private/ion_tables_load.f90:54-69`), whose version must be
/// at least [`ION_MIN_VERSION`]. Every table is then compared with the
/// reference and its expected composition exactly as in `Load1_ion_Table`
/// (`ion_tables_load.f90:259-270`), using [`ION_HEADER_TINY`]. Returns the
/// reference header.
fn validate_ion_headers(
    entries: &[(&Path, IonAsciiHeader, f64, f64)],
) -> Result<IonAsciiHeader, IonError> {
    let (ref_path, reference, _, _) = *entries.first().ok_or_else(|| {
        IonError::Io("no ionization table headers to validate".to_string())
    })?;
    let fail = |path: &Path, detail: String| IonError::TableParse {
        path: path.display().to_string(),
        detail,
    };
    if reference.version < ION_MIN_VERSION {
        return Err(fail(
            ref_path,
            format!(
                "header field version = {} is below min_version = \
                 {ION_MIN_VERSION} (reinstall ionization_data)",
                reference.version
            ),
        ));
    }
    let ref_name = ref_path.display();
    for &(path, h, z_exp, x_exp) in entries {
        if reference.version > h.version {
            return Err(fail(
                path,
                format!(
                    "header field version = {} is older than version = {} \
                     in {ref_name}",
                    h.version, reference.version
                ),
            ));
        }
        for (field, got, want) in [
            ("num_logQs", h.num_log_qs, reference.num_log_qs),
            ("num_logTs", h.num_log_ts, reference.num_log_ts),
        ] {
            if got != want {
                return Err(fail(
                    path,
                    format!(
                        "header field {field} = {got} differs from {want} \
                         in {ref_name}"
                    ),
                ));
            }
        }
        for (field, got, want) in [("X", h.x, x_exp), ("Z", h.z, z_exp)] {
            if (want - got).abs() > ION_HEADER_TINY {
                return Err(fail(
                    path,
                    format!(
                        "header field {field} = {got} differs from the \
                         expected table mass fraction {want}"
                    ),
                ));
            }
        }
        for (field, got, want) in [
            ("logT_min", h.log_t_min, reference.log_t_min),
            ("logT_max", h.log_t_max, reference.log_t_max),
            ("del_logT", h.del_log_t, reference.del_log_t),
            ("logQ_min", h.log_q_min, reference.log_q_min),
            ("logQ_max", h.log_q_max, reference.log_q_max),
            ("del_logQ", h.del_log_q, reference.del_log_q),
        ] {
            if (want - got).abs() > ION_HEADER_TINY {
                return Err(fail(
                    path,
                    format!(
                        "header field {field} = {got} differs from {want} \
                         in {ref_name}"
                    ),
                ));
            }
        }
    }
    Ok(reference)
}

fn compute_ion_dir_sha256(
    data_dir: &Path,
    file_prefix: &str,
    z1_suffix: &str,
) -> Result<String, IonError> {
    validate_ion_data_dir(data_dir, file_prefix, z1_suffix)?;
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
/// Uses the grid of the already validated reference header `grid`, parses
/// all `nq * nt` rows across the 28 output channels (rejecting non-finite
/// raw inputs loudly), fits a 2D not-a-knot bicubic spline to each channel
/// via [`interp_mkbicub_db`], and replaces non-finite fitted spline
/// coefficients with `0.0` matching upstream `Check_ion_Interpolation_Data`
/// (`ionization/private/ion_tables_load.f90:362-382`).
///
/// # Errors
/// Returns [`IonError`] if the file cannot be read, is truncated, or fails
/// bicubic spline construction.
#[allow(clippy::cast_precision_loss)]
fn parse_ion_ascii_table(
    path: &Path,
    grid: &IonAsciiHeader,
) -> Result<Vec<Vec<f64>>, IonError> {
    let text =
        fs::read_to_string(path).map_err(|e| IonError::Io(e.to_string()))?;
    let (nq, lq_min, dq) = (grid.num_log_qs, grid.log_q_min, grid.del_log_q);
    let (nt, lt_min, dt) = (grid.num_log_ts, grid.log_t_min, grid.del_log_t);
    let mut lines = text.lines().skip(3);

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
    build_table_splines(&log_qs, &log_ts, &raw)
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
        // Upstream MESA 26.4.1 Check_ion_Interpolation_Data
        // (ionization/private/ion_tables_load.f90:362-382) replaces NaN and
        // infinite fitted spline entries with 0.0.
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
    use std::fmt::Write as _;

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

    fn populate_all_22_ion_tables(dir: &Path) {
        fs::create_dir_all(dir).expect("mkdir ion dir");
        for (&z, &n_xs) in ION_ZS.iter().zip(NUM_ION_XS_FOR_Z.iter()) {
            for &x in ION_XS.iter().take(n_xs) {
                let p = ion_table_filename(dir, "ion", "", z, x);
                fs::write(p, b"placeholder\n").expect("write ion stub");
            }
        }
    }

    /// Header row of every MESA 26.4.1 `data/ionization_data/ion_*.data`
    /// table, with the `(X, Z)` fields substituted per file.
    fn mesa_26_4_1_header_row(x: f64, z: f64) -> String {
        format!(
            "            51        {x:.4}        {z:.4}       102            \
             2.1000        8.1600        0.0600       175          \
             -10.0000        5.6600        0.0900"
        )
    }

    /// Write 22 small `4 x 4` tables whose header rows come from `header`
    /// (`(file_name, X, Z) -> row`), following the upstream line layout.
    #[allow(clippy::cast_precision_loss)]
    fn write_synthetic_ion_tables(
        dir: &Path,
        header: impl Fn(&str, f64, f64) -> String,
    ) {
        fs::create_dir_all(dir).expect("mkdir ion dir");
        for (&z, &n_xs) in ION_ZS.iter().zip(NUM_ION_XS_FOR_Z.iter()) {
            for &x in ION_XS.iter().take(n_xs) {
                let p = ion_table_filename(dir, "ion", "", z, x);
                let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                let mut out = String::from("version X Z ...\n");
                out.push_str(&header(name, x, z));
                out.push_str("\n\n");
                for iq in 0..4_u32 {
                    out.push_str("\n      logQ\n\n logT vals\n");
                    for jt in 0..4_u32 {
                        let _ =
                            write!(out, "{:.2}", 3.0 + 0.06 * f64::from(jt));
                        for v in 0..NUM_ION_VALS {
                            let val = 0.25
                                + 0.011 * f64::from(iq * iq)
                                + 0.007 * f64::from(jt)
                                + 0.001 * (v as f64);
                            let _ = write!(out, " {val:.3}");
                        }
                        out.push('\n');
                    }
                    if iq + 1 < 4 {
                        out.push_str("\n\n");
                    }
                }
                fs::write(&p, out).expect("write synthetic ion table");
            }
        }
    }

    fn small_header_row(version: i64, x: f64, z: f64) -> String {
        format!("{version} {x:.4} {z:.4} 4 3.00 3.18 0.06 4 -6.00 -5.73 0.09")
    }

    fn assert_bad_header_case(
        dir: &Path,
        target_file: &str,
        override_row: &str,
        expected_detail: &str,
    ) {
        write_synthetic_ion_tables(dir, |name, x, z| {
            if name == target_file {
                override_row.to_string()
            } else {
                small_header_row(51, x, z)
            }
        });
        let err = IonTableGrid::from_data_dir(dir, "ion", "")
            .expect_err("bad header must be rejected");
        let (path, detail) = table_parse_detail(&err);
        assert!(path.ends_with(target_file), "got path {path}");
        assert!(
            detail.contains(expected_detail),
            "expected '{expected_detail}' in '{detail}'"
        );
    }

    fn table_parse_detail(err: &IonError) -> (String, String) {
        match err {
            IonError::TableParse { path, detail } => {
                (path.clone(), detail.clone())
            }
            other => panic!("expected IonError::TableParse, got {other:?}"),
        }
    }

    #[test]
    fn test_header_tiny_equals_single_precision_literal_widened_to_double() {
        assert_eq!(ION_HEADER_TINY.to_bits(), f64::from(1.0e-6_f32).to_bits());
    }

    #[test]
    fn test_valid_mesa_26_4_1_header_rows_pass_upstream_header_checks() {
        let dir = Path::new("ionization_data");
        let mut paths = Vec::new();
        let mut headers = Vec::new();
        for (&z, &n_xs) in ION_ZS.iter().zip(NUM_ION_XS_FOR_Z.iter()) {
            for &x in ION_XS.iter().take(n_xs) {
                let p = ion_table_filename(dir, "ion", "", z, x);
                let h =
                    parse_ion_header_line(&p, &mesa_26_4_1_header_row(x, z))
                        .expect("MESA 26.4.1 header row parses");
                paths.push(p);
                headers.push((h, z, x));
            }
        }
        let entries: Vec<(&Path, IonAsciiHeader, f64, f64)> = paths
            .iter()
            .zip(headers.iter())
            .map(|(p, &(h, z, x))| (p.as_path(), h, z, x))
            .collect();
        let reference =
            validate_ion_headers(&entries).expect("valid 26.4.1 headers");
        assert_eq!(reference.version, 51);
        assert_eq!((reference.num_log_qs, reference.num_log_ts), (175, 102));
        assert_eq!(reference.log_q_min.to_bits(), (-10.0_f64).to_bits());
        assert_eq!(reference.del_log_q.to_bits(), 0.09_f64.to_bits());
        assert_eq!(reference.log_t_min.to_bits(), 2.1_f64.to_bits());
        assert_eq!(reference.del_log_t.to_bits(), 0.06_f64.to_bits());
        assert_eq!(reference.log_q_max.to_bits(), 5.66_f64.to_bits());
        assert_eq!(reference.log_t_max.to_bits(), 8.16_f64.to_bits());
    }

    #[test]
    fn test_from_data_dir_accepts_valid_and_sub_tiny_headers() {
        let root = std::env::temp_dir()
            .join(format!("flexmesa_ion_hdr_ok_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);

        let ok_dir = root.join("ok");
        write_synthetic_ion_tables(&ok_dir, |_, x, z| {
            small_header_row(51, x, z)
        });
        let grid = IonTableGrid::from_data_dir(&ok_dir, "ion", "")
            .expect("valid synthetic headers load");
        assert_eq!((grid.num_log_qs, grid.num_log_ts), (4, 4));

        let near_dir = root.join("near");
        write_synthetic_ion_tables(&near_dir, |name, x, z| {
            if name == "ion_02z40x.data" {
                format!(
                    "51 {x:.4} {z:.4} 4 3.00 3.18 0.0600005 4 -6.00 -5.73 0.09"
                )
            } else {
                small_header_row(51, x, z)
            }
        });
        IonTableGrid::from_data_dir(&near_dir, "ion", "")
            .expect("5e-7 grid offset is within tiny");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_from_data_dir_rejects_bad_headers_naming_file_and_field() {
        let root = std::env::temp_dir()
            .join(format!("flexmesa_ion_headers_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);

        // All tables stale (version < min_version = 49) fails on ion_00z00x.
        let stale_dir = root.join("stale");
        write_synthetic_ion_tables(&stale_dir, |_, x, z| {
            small_header_row(48, x, z)
        });
        let (p, d) = table_parse_detail(
            &IonTableGrid::from_data_dir(&stale_dir, "ion", "")
                .expect_err("stale version must fail"),
        );
        assert!(p.ends_with("ion_00z00x.data"), "got {p}");
        assert!(d.contains("version = 48"), "got {d}");

        let cases = [
            (
                "ion_04z20x.data",
                "50 0.2000 0.0400 4 3.00 3.18 0.06 4 -6.00 -5.73 0.09",
                "version = 50",
            ),
            (
                "ion_02z40x.data",
                "51 0.6000 0.0200 4 3.00 3.18 0.06 4 -6.00 -5.73 0.09",
                "field X = 0.6",
            ),
            (
                "ion_20z60x.data",
                "51 0.6000 0.0200 4 3.00 3.18 0.06 4 -6.00 -5.73 0.09",
                "field Z = 0.02",
            ),
            (
                "ion_00z20x.data",
                "51 0.2000 0.0000 5 3.00 3.24 0.06 4 -6.00 -5.73 0.09",
                "field num_logTs = 5",
            ),
            (
                "ion_00z40x.data",
                "51 0.4000 0.0000 4 3.01 3.18 0.06 4 -6.00 -5.73 0.09",
                "field logT_min = 3.01",
            ),
            (
                "ion_20z60x.data",
                "51 0.6000 0.2000 4 3.00 3.18002 0.06 4 -6.00 -5.73 0.09",
                "field logT_max = 3.18002",
            ),
            (
                "ion_02z20x.data",
                "51 0.2000 0.0200 4 3.00 3.18 0.07 4 -6.00 -5.73 0.09",
                "field del_logT = 0.07",
            ),
            (
                "ion_00z80x.data",
                "51 0.8000 0.0000 4 3.00 3.18 0.06 5 -6.00 -5.64 0.09",
                "field num_logQs = 5",
            ),
            (
                "ion_04z40x.data",
                "51 0.4000 0.0400 4 3.00 3.18 0.06 4 -5.99 -5.73 0.09",
                "field logQ_min = -5.99",
            ),
            (
                "ion_100z00x.data",
                "51 0.0000 1.0000 4 3.00 3.18 0.06 4 -6.00 -5.70 0.09",
                "field logQ_max = -5.7",
            ),
            (
                "ion_20z80x.data",
                "51 0.8000 0.2000 4 3.00 3.18 0.06 4 -6.00 -5.73 0.08",
                "field del_logQ = 0.08",
            ),
            (
                "ion_04z60x.data",
                "51 0.6000 0.0400 4 3.00 NaN 0.06 4 -6.00 -5.73 0.09",
                "non-finite header field logT_max",
            ),
        ];
        for (i, (file, row, needle)) in cases.iter().enumerate() {
            assert_bad_header_case(
                &root.join(format!("bad_{i}")),
                file,
                row,
                needle,
            );
        }
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_embedded_manifest_matches_fmion001_payload_and_sha256() {
        let m = embedded_ion_manifest().expect("embedded manifest parses");
        assert_eq!(m.sha256, compute_sha256_bytes(ION_TABLES_FULL_BIN));
        assert_eq!(m.size_bytes, ION_TABLES_FULL_BIN.len());
        assert_eq!(
            parse_fmion001_header(ION_TABLES_FULL_BIN).expect("header"),
            (m.num_tables, m.num_vals, m.num_log_qs, m.num_log_ts)
        );
        // MESA star_job.defaults:2813 sets ionization_Z1_suffix = '', the
        // suffix used by resolve_ion_data_dir_from for external tables.
        assert_eq!(m.z1_suffix, "");
        assert!(m.ion_version >= ION_MIN_VERSION);
        assert_eq!(m.value_scale, 1000);
        // Grid nodes implied by min, spacing and count reach the recorded
        // maxima of the MESA 26.4.1 headers within upstream tiny.
        #[allow(clippy::cast_precision_loss)]
        let (q_end, t_end) = (
            m.log_q_min + (m.num_log_qs - 1) as f64 * m.del_log_q,
            m.log_t_min + (m.num_log_ts - 1) as f64 * m.del_log_t,
        );
        assert!((q_end - m.log_q_max).abs() <= ION_HEADER_TINY);
        assert!((t_end - m.log_t_max).abs() <= ION_HEADER_TINY);
        let grid = IonTableGrid::try_from_embedded_full().expect("embedded");
        assert_eq!(grid.log_q_min.to_bits(), m.log_q_min.to_bits());
        assert_eq!(grid.del_log_q.to_bits(), m.del_log_q.to_bits());
        assert_eq!(grid.log_t_min.to_bits(), m.log_t_min.to_bits());
        assert_eq!(grid.del_log_t.to_bits(), m.del_log_t.to_bits());
    }

    #[test]
    fn test_manifest_parser_rejects_missing_or_non_finite_fields() {
        let good = ION_TABLES_FULL_MANIFEST;
        assert!(EmbeddedIonManifest::parse(good).is_ok());
        for (from, to) in [
            ("format = \"FMION001\"", "format = \"FMION999\""),
            ("del_log_q = 0.09", "del_log_q = nan"),
            ("log_t_min = 2.1", "log_t_min = inf"),
            ("value_scale = 1000", "value_scale = 0"),
            ("num_log_qs = 175", "num_log_qs = -175"),
            ("z1_suffix = \"\"", "z1_suffix = ''"),
            ("ion_version = 51\n", ""),
        ] {
            assert!(good.contains(from), "fixture lacks '{from}'");
            let bad = good.replacen(from, to, 1);
            let err = EmbeddedIonManifest::parse(&bad)
                .expect_err("malformed manifest must be rejected");
            let (path, _) = table_parse_detail(&err);
            assert_eq!(path, ION_TABLES_MANIFEST_NAME);
        }
    }

    #[test]
    fn test_fmion001_decoder_rejects_corrupt_payloads_without_panic() {
        let m = embedded_ion_manifest().expect("manifest").clone();
        let bin = ION_TABLES_FULL_BIN;
        let mut bad_flag = bin.to_vec();
        bad_flag[FMION_HEADER_BYTES] = 7;
        let mut trailing = bin.to_vec();
        trailing.push(0);
        let mut trailing_m = m.clone();
        trailing_m.size_bytes += 1;
        let mut bad_dims = bin.to_vec();
        bad_dims[16] = 174;
        let cases: [(&[u8], &EmbeddedIonManifest, &str); 6] = [
            (&bin[..10], &m, "invalid FMION001 header"),
            (b"FMION002_PADDING_PADDING", &m, "invalid FMION001 header"),
            (&bin[..bin.len() - 3], &m, "size_bytes"),
            (&bad_flag, &m, "unknown column flag 7"),
            (&trailing, &trailing_m, "trailing bytes"),
            (&bad_dims, &m, "num_log_qs = 174"),
        ];
        for (bytes, manifest, needle) in cases {
            let err = IonTableGrid::from_fmion001_bytes(bytes, manifest)
                .expect_err("corrupt FMION001 must be rejected");
            let (path, detail) = table_parse_detail(&err);
            assert_eq!(path, ION_TABLES_FULL_NAME);
            assert!(detail.contains(needle), "'{needle}' not in '{detail}'");
        }
        // Truncation inside the body is reported as EOF, not a panic.
        let mut trunc_m = m;
        trunc_m.size_bytes = bin.len() - 3;
        let err =
            IonTableGrid::from_fmion001_bytes(&bin[..bin.len() - 3], &trunc_m)
                .expect_err("truncated body must be rejected");
        assert!(table_parse_detail(&err).1.contains("unexpected EOF"));
    }

    /// `f64::from(n) / 1000.0` (embedded decode) equals the correctly rounded
    /// parse of the `%.3f` token printed for `n` thousandths (ASCII decode)
    /// for every value range spanned by the MESA 26.4.1 tables
    /// (`[-59.829, 28.403]`, checked here over `[-100, 100]`).
    #[test]
    fn test_thousandths_division_matches_decimal_parse_bit_for_bit() {
        for n in -100_000_i32..=100_000 {
            let sign = if n < 0 { "-" } else { "" };
            let a = n.unsigned_abs();
            let token = format!("{sign}{}.{:03}", a / 1000, a % 1000);
            let parsed: f64 = token.parse().expect("parse %.3f token");
            let divided = f64::from(n) / 1000.0;
            assert_eq!(
                parsed.to_bits(),
                divided.to_bits(),
                "n = {n}, token {token}"
            );
        }
    }

    #[test]
    fn test_build_table_splines_zeros_non_finite_coefficients_matching_upstream()
     {
        let log_qs = [-6.0_f64, -5.9, -5.8, -5.7];
        let log_ts = [3.0_f64, 3.1, 3.2, 3.3];
        let mut raw = vec![vec![0.0_f64; 16]; NUM_ION_VALS];
        // Choose finite vertex values whose first-pass f_xx (~1e307) stays
        // finite while the second-pass cross-derivative f_xxyy (~1e309)
        // overflows to +/-Inf inside interp_mkbicub_db.
        raw[0][0] = 1.0e305;
        raw[0][1] = -1.0e305;
        // Confirm raw interp_mkbicub_db actually produces non-finite derivative
        // terms on this grid, then verify build_table_splines zeroes them per
        // Check_ion_Interpolation_Data (ionization/private/ion_tables_load.f90:362-382).
        let (raw_c, _, _) = interp_mkbicub_db(
            &log_qs,
            &log_ts,
            &raw[0],
            BicubicBc::NotAKnot,
            &[],
            BicubicBc::NotAKnot,
            &[],
            BicubicBc::NotAKnot,
            &[],
            BicubicBc::NotAKnot,
            &[],
        )
        .expect("bicubic fit returns array with overflowed derivative terms");
        let bad_indices: Vec<usize> = raw_c
            .iter()
            .flatten()
            .enumerate()
            .filter_map(|(idx, &val)| is_bad(val).then_some(idx))
            .collect();
        assert!(
            !bad_indices.is_empty(),
            "test input must trigger non-finite spline derivatives"
        );

        let coeffs = build_table_splines(&log_qs, &log_ts, &raw).expect(
            "upstream Check_ion_Interpolation_Data zeroes non-finite entries",
        );
        assert!(
            coeffs[0].iter().all(|&v| !is_bad(v)),
            "all output spline coefficients must be finite"
        );
        for idx in bad_indices {
            assert_eq!(
                coeffs[0][idx], 0.0,
                "non-finite spline entry at idx={idx} must be replaced with 0.0"
            );
        }
    }

    #[test]
    fn test_resolve_ion_data_dir_errors_on_incomplete_dirs_without_fallback() {
        let root = std::env::temp_dir()
            .join(format!("flexmesa_ion_resolve_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let flex_data = root.join("data");
        let mesa_test_only = root.join("mesa_test_inputs");
        fs::create_dir_all(&flex_data).expect("mkdir flex_data");
        fs::create_dir_all(mesa_test_only.join("data").join("star_data"))
            .expect("mkdir star_data");

        // 1. Standard data-v1 + mesa_test_inputs layout (no ionization_data) -> Ok(None)
        assert_eq!(
            resolve_ion_data_dir_from(&flex_data, Some(&mesa_test_only))
                .expect("absent ionization_data resolves to None"),
            None
        );

        // 2. Empty $FLEXMESA_DATA_DIR/ionization_data/ -> loud error naming dir + file
        let sub = flex_data.join("ionization_data");
        fs::create_dir_all(&sub).expect("mkdir empty ionization_data");
        let err_empty =
            resolve_ion_data_dir_from(&flex_data, Some(&mesa_test_only))
                .expect_err("empty ionization_data must error");
        let msg_empty = err_empty.to_string();
        assert!(msg_empty.contains("ionization_data"), "got: {msg_empty}");
        assert!(msg_empty.contains("ion_00z00x.data"), "got: {msg_empty}");

        // 3. Partial $FLEXMESA_DATA_DIR/ionization_data/ (missing non-sentinel ion_20z80x.data)
        populate_all_22_ion_tables(&sub);
        let missing = sub.join("ion_20z80x.data");
        fs::remove_file(&missing).expect("remove one non-sentinel ion file");
        let err_partial =
            resolve_ion_data_dir_from(&flex_data, Some(&mesa_test_only))
                .expect_err("partial ionization_data must error");
        let msg_partial = err_partial.to_string();
        assert!(
            msg_partial.contains("ionization_data"),
            "got: {msg_partial}"
        );
        assert!(
            msg_partial.contains("ion_20z80x.data"),
            "got: {msg_partial}"
        );

        // Restore missing file -> complete 22-file directory resolves cleanly
        fs::write(&missing, b"placeholder\n").expect("restore missing");
        assert_eq!(
            resolve_ion_data_dir_from(&flex_data, Some(&mesa_test_only))
                .expect("complete ionization_data resolves"),
            Some(sub.clone())
        );
        fs::remove_dir_all(&sub).expect("remove sub");

        // 4. Incomplete $MESA_DIR/data/ionization_data/ -> loud error naming dir + file
        let mesa_full = root.join("mesa_full");
        let mesa_ion = mesa_full.join("data").join("ionization_data");
        populate_all_22_ion_tables(&mesa_ion);
        fs::remove_file(mesa_ion.join("ion_04z40x.data"))
            .expect("remove mesa ion_04z40x.data");
        let err_mesa = resolve_ion_data_dir_from(&flex_data, Some(&mesa_full))
            .expect_err("incomplete MESA_DIR ionization_data must error");
        let msg_mesa = err_mesa.to_string();
        assert!(msg_mesa.contains("ionization_data"), "got: {msg_mesa}");
        assert!(msg_mesa.contains("ion_04z40x.data"), "got: {msg_mesa}");

        // 5. Partial ion_*.data placed directly in $FLEXMESA_DATA_DIR (missing ion_00z00x.data)
        fs::write(flex_data.join("ion_02z60x.data"), b"placeholder\n")
            .expect("write direct stub");
        let err_direct = resolve_ion_data_dir_from(&flex_data, None)
            .expect_err("partial direct ion_*.data must error");
        assert!(
            err_direct.to_string().contains("ion_00z00x.data"),
            "got: {err_direct}"
        );
        fs::remove_file(flex_data.join("ion_02z60x.data")).expect("rm direct");

        // 6. Non-directory file at $FLEXMESA_DATA_DIR/ionization_data -> loud error
        fs::write(&sub, b"not a directory").expect("write file at sub");
        let err_file = resolve_ion_data_dir_from(&flex_data, None)
            .expect_err("file at ionization_data must error");
        assert!(
            err_file.to_string().contains("not a directory"),
            "got: {err_file}"
        );

        let _ = fs::remove_dir_all(&root);
    }
}
