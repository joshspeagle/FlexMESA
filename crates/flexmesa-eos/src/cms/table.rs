// SPDX-License-Identifier: GPL-3.0-or-later
//! Table loader and lazy-loaded slice collection for `CMS` (`eoscms_load_tables.f90`).

pub use crate::cms::config::{
    get_cms_alfa, CmsConfig, CMS_NUM_XS, CMS_XSTR, CMS_XVALS,
};

use crate::helm::table_io::resolve_mesa_data_dir;
use crate::tables_4d::slice_cache::load_or_build_cms_slice;
use crate::tables_4d::spline::{
    build_4d_slice_vertices, validate_4d_grid_header, EosDtVertex,
    NUM_EOS_TABLE_VALS,
};
use crate::types::{
    EosError, I_CHI_RHO, I_CHI_T, I_CP, I_CV, I_DE_DRHO, I_DS_DRHO, I_DS_DT,
    I_ETA, I_GAMMA1, I_GAMMA3, I_GRAD_AD, I_LN_E, I_LN_FREE_E, I_LN_PGAS,
    I_LN_S, I_MU,
};
use flexmesa_const::LN10;
use flexmesa_math::safe_log;
use flexmesa_utils::is_bad;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

/// Magic header for compact raw `(logT, logRho)` `CMS` binary slices.
pub const CMS_RAW_BIN_MAGIC: &[u8; 8] = b"FMCMSR01";

/// Single fixed-$X$ `CMS` bicubic spline table (`eosCMS_X_Info`).
#[derive(Debug, Clone)]
pub struct CmsXTable {
    /// Table format version (`1`).
    pub version: i32,
    /// Hydrogen mass fraction $X$.
    pub x: f64,
    /// Number of $\log_{10} T$ grid points (`121`).
    pub num_log_ts: usize,
    /// Minimum $\log_{10} T$ (`2.0`).
    pub log_t_min: f64,
    /// Maximum $\log_{10} T$ (`8.0`).
    pub log_t_max: f64,
    /// Grid step $\Delta \log_{10} T$ (`0.05`).
    pub delta_log_t: f64,
    /// Number of $\log_{10} \rho$ grid points (`281`).
    pub num_log_rhos: usize,
    /// Minimum $\log_{10} \rho$ (`-8.0`).
    pub log_rho_min: f64,
    /// Maximum $\log_{10} \rho$ (`6.0`).
    pub log_rho_max: f64,
    /// Grid step $\Delta \log_{10} \rho$ (`0.05`).
    pub delta_log_rho: f64,
    /// Full dense grid of `num_log_ts * num_log_rhos` vertices.
    pub dense: Option<Vec<EosDtVertex>>,
}

impl CmsXTable {
    /// Retrieves a 1-based `(i_t, j_rho)` vertex.
    #[must_use]
    pub fn get_vertex(&self, i_t: usize, j_rho: usize) -> Option<&EosDtVertex> {
        let dense = self.dense.as_ref()?;
        let idx = (i_t - 1) * self.num_log_rhos + (j_rho - 1);
        dense.get(idx)
    }

    #[inline]
    fn store_mapped_cell(
        raw_vars: &mut [f64],
        ngrid: usize,
        cell: usize,
        vals: &[f64; 16],
    ) {
        raw_vars[I_LN_PGAS * ngrid + cell] = vals[0] * LN10;
        raw_vars[I_LN_E * ngrid + cell] = vals[1] * LN10;
        raw_vars[I_LN_S * ngrid + cell] = vals[2] * LN10;
        raw_vars[I_CHI_RHO * ngrid + cell] = vals[3];
        raw_vars[I_CHI_T * ngrid + cell] = vals[4];
        raw_vars[I_CP * ngrid + cell] = safe_log(vals[5]);
        raw_vars[I_CV * ngrid + cell] = safe_log(vals[6]);
        raw_vars[I_DE_DRHO * ngrid + cell] = vals[7];
        raw_vars[I_DS_DT * ngrid + cell] = vals[8];
        raw_vars[I_DS_DRHO * ngrid + cell] = vals[9];
        raw_vars[I_MU * ngrid + cell] = vals[10];
        raw_vars[I_LN_FREE_E * ngrid + cell] = vals[11];
        raw_vars[I_GAMMA1 * ngrid + cell] = vals[12];
        raw_vars[I_GAMMA3 * ngrid + cell] = vals[13];
        raw_vars[I_GRAD_AD * ngrid + cell] = vals[14];
        raw_vars[I_ETA * ngrid + cell] = vals[15];
    }

    #[allow(clippy::too_many_arguments)]
    fn build_from_raw_vars(
        table_label: &str,
        version: i32,
        x: f64,
        num_log_ts: usize,
        log_t_min: f64,
        log_t_max: f64,
        delta_log_t: f64,
        num_log_rhos: usize,
        log_rho_min: f64,
        log_rho_max: f64,
        delta_log_rho: f64,
        raw_vars: &[f64],
    ) -> Result<Self, EosError> {
        let mut log_ts = Vec::with_capacity(num_log_ts);
        let mut t_cur = log_t_min;
        for i in 0..num_log_ts {
            if i > 0 {
                t_cur += delta_log_t;
            }
            log_ts.push(t_cur);
        }
        let mut log_rhos = Vec::with_capacity(num_log_rhos);
        let mut rho_cur = log_rho_min;
        for j in 0..num_log_rhos {
            if j > 0 {
                rho_cur += delta_log_rho;
            }
            log_rhos.push(rho_cur);
        }
        let dense = build_4d_slice_vertices(
            "CmsXTable::build_from_raw_vars",
            table_label,
            &log_ts,
            &log_rhos,
            raw_vars,
        )?;

        Ok(Self {
            version,
            x,
            num_log_ts,
            log_t_min,
            log_t_max,
            delta_log_t,
            num_log_rhos,
            log_rho_min,
            log_rho_max,
            delta_log_rho,
            dense: Some(dense),
        })
    }

    /// Reads a compact `FMCMSR01` raw binary slice (`eos_4d_raw/mesa-CMS_*.raw`) and fits its 16 bicubic splines.
    pub fn from_raw_bin(path: &Path) -> Result<Self, EosError> {
        let table_label = path.display().to_string();
        let err = |msg: String| EosError::TableIo {
            context: "CmsXTable::from_raw_bin",
            detail: format!("{table_label}: {msg}"),
        };
        let file = std::fs::File::open(path).map_err(|e| err(e.to_string()))?;
        let mut reader = BufReader::with_capacity(256 * 1024, file);
        let mut hdr = [0u8; 76];
        reader
            .read_exact(&mut hdr)
            .map_err(|e| err(e.to_string()))?;
        if &hdr[0..8] != CMS_RAW_BIN_MAGIC {
            return Err(err("invalid FMCMSR01 magic".to_string()));
        }
        let f64_at = |off: usize| -> Result<f64, EosError> {
            let bytes: [u8; 8] = hdr[off..off + 8]
                .try_into()
                .map_err(|_| err("short FMCMSR01 header slice".to_string()))?;
            Ok(f64::from_le_bytes(bytes))
        };
        let i32_at = |off: usize| -> Result<i32, EosError> {
            let bytes: [u8; 4] = hdr[off..off + 4]
                .try_into()
                .map_err(|_| err("short FMCMSR01 header slice".to_string()))?;
            Ok(i32::from_le_bytes(bytes))
        };
        let u32_at = |off: usize| -> Result<usize, EosError> {
            let bytes: [u8; 4] = hdr[off..off + 4]
                .try_into()
                .map_err(|_| err("short FMCMSR01 header slice".to_string()))?;
            Ok(u32::from_le_bytes(bytes) as usize)
        };
        let version = i32_at(8)?;
        let x = f64_at(12)?;
        let num_log_ts = u32_at(20)?;
        let log_t_min = f64_at(24)?;
        let log_t_max = f64_at(32)?;
        let delta_log_t = f64_at(40)?;
        let num_log_rhos = u32_at(48)?;
        let log_rho_min = f64_at(52)?;
        let log_rho_max = f64_at(60)?;
        let delta_log_rho = f64_at(68)?;

        let ngrid = validate_4d_grid_header(
            "CmsXTable::from_raw_bin",
            &table_label,
            x,
            0.0,
            num_log_ts,
            log_t_min,
            log_t_max,
            delta_log_t,
            num_log_rhos,
            log_rho_min,
            log_rho_max,
            delta_log_rho,
        )?;
        let mut raw_vars = vec![0.0_f64; NUM_EOS_TABLE_VALS * ngrid];
        let mut row_bytes = [0u8; NUM_EOS_TABLE_VALS * 8];
        for cell in 0..ngrid {
            reader
                .read_exact(&mut row_bytes)
                .map_err(|e| err(e.to_string()))?;
            let it = cell / num_log_rhos;
            let jrho = cell % num_log_rhos;
            let mut vals = [0.0_f64; 16];
            for (k, slot) in vals.iter_mut().enumerate() {
                let s = k * 8;
                let bytes: [u8; 8] =
                    row_bytes[s..s + 8].try_into().map_err(|_| {
                        err("short FMCMSR01 cell slice".to_string())
                    })?;
                let val = f64::from_le_bytes(bytes);
                if is_bad(val) {
                    return Err(err(format!(
                        "non-finite raw column {k} ({val}) at cell=({it}, {jrho})"
                    )));
                }
                *slot = val;
            }
            Self::store_mapped_cell(&mut raw_vars, ngrid, cell, &vals);
        }
        let mut trailing = [0_u8; 1];
        if reader.read(&mut trailing).map_err(|e| err(e.to_string()))? != 0 {
            return Err(err(
                "unexpected trailing bytes after FMCMSR01 payload".to_string(),
            ));
        }

        Self::build_from_raw_vars(
            &table_label,
            version,
            x,
            num_log_ts,
            log_t_min,
            log_t_max,
            delta_log_t,
            num_log_rhos,
            log_rho_min,
            log_rho_max,
            delta_log_rho,
            &raw_vars,
        )
    }

    /// Parses a `mesa-CMS_*.data` ASCII file and builds its 16 bicubic splines (`load_eosCMS_table`).
    pub fn from_ascii_reader<R: BufRead>(
        mut reader: R,
    ) -> Result<Self, EosError> {
        let mut line_buf = String::with_capacity(512);
        let mut next_non_empty =
            |buf: &mut String| -> Result<String, EosError> {
                loop {
                    buf.clear();
                    let n = reader.read_line(buf).map_err(|e| {
                        EosError::InvalidState {
                            context: "CmsXTable::from_ascii_reader",
                            detail: e.to_string(),
                        }
                    })?;
                    if n == 0 {
                        return Err(EosError::InvalidState {
                            context: "CmsXTable::from_ascii_reader",
                            detail: "unexpected EOF".to_string(),
                        });
                    }
                    if !buf.trim().is_empty() {
                        return Ok(buf.clone());
                    }
                }
            };

        let _hdr_names = next_non_empty(&mut line_buf)?;
        let hdr_vals_line = next_non_empty(&mut line_buf)?;
        let hdr: Vec<f64> = hdr_vals_line
            .split_whitespace()
            .map(|t| {
                t.parse::<f64>().map_err(|e| EosError::InvalidState {
                    context: "CmsXTable::from_ascii_reader",
                    detail: format!("bad header float '{t}': {e}"),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if hdr.len() < 11 {
            return Err(EosError::InvalidState {
                context: "CmsXTable::from_ascii_reader",
                detail: "fewer than 11 header values".to_string(),
            });
        }

        let version = hdr[0] as i32;
        let x = hdr[1];
        let num_log_ts = hdr[3] as usize;
        let log_t_min = hdr[4];
        let log_t_max = hdr[5];
        let delta_log_t = hdr[6];
        let num_log_rhos = hdr[7] as usize;
        let log_rho_min = hdr[8];
        let log_rho_max = hdr[9];
        let delta_log_rho = hdr[10];

        let _col_names = next_non_empty(&mut line_buf)?;

        let ngrid = validate_4d_grid_header(
            "CmsXTable::from_ascii_reader",
            "CMS ascii",
            x,
            0.0,
            num_log_ts,
            log_t_min,
            log_t_max,
            delta_log_t,
            num_log_rhos,
            log_rho_min,
            log_rho_max,
            delta_log_rho,
        )?;
        let mut raw_vars = vec![0.0_f64; NUM_EOS_TABLE_VALS * ngrid];

        for it in 0..num_log_ts {
            for jrho in 0..num_log_rhos {
                let row = next_non_empty(&mut line_buf)?;
                let mut toks = row.split_whitespace();
                let _lt = toks.next();
                let _lr = toks.next();
                let mut vals = [0.0_f64; 16];
                for (k, v) in vals.iter_mut().enumerate() {
                    let s =
                        toks.next().ok_or_else(|| EosError::InvalidState {
                            context: "CmsXTable::from_ascii_reader",
                            detail: format!(
                                "fewer than 18 columns at ({it}, {jrho})"
                            ),
                        })?;
                    let val = s.parse::<f64>().map_err(|e| {
                        EosError::InvalidState {
                            context: "CmsXTable::from_ascii_reader",
                            detail: format!("bad float '{s}': {e}"),
                        }
                    })?;
                    if is_bad(val) {
                        return Err(EosError::InvalidState {
                            context: "CmsXTable::from_ascii_reader",
                            detail: format!(
                                "non-finite ASCII column {k} ({val}) in CMS(X={x}) cell=({it}, {jrho})"
                            ),
                        });
                    }
                    *v = val;
                }
                let cell = it * num_log_rhos + jrho;
                Self::store_mapped_cell(&mut raw_vars, ngrid, cell, &vals);
            }
        }

        let label = format!("CMS(X={x})");
        Self::build_from_raw_vars(
            &label,
            version,
            x,
            num_log_ts,
            log_t_min,
            log_t_max,
            delta_log_t,
            num_log_rhos,
            log_rho_min,
            log_rho_max,
            delta_log_rho,
            &raw_vars,
        )
    }

    /// Loads a `CMS` `.data` file from disk.
    pub fn from_data_file(path: &Path) -> Result<Self, EosError> {
        let file =
            std::fs::File::open(path).map_err(|e| EosError::TableIo {
                context: "CmsXTable::from_data_file",
                detail: format!("{}: {e}", path.display()),
            })?;
        Self::from_ascii_reader(std::io::BufReader::new(file))
    }
}

/// Collection of the 11 `CMS` tables (`ix_1based in 1..=11`, `~187 MB` total) with lock-free lazy caching.
#[derive(Debug)]
pub struct CmsTableSet {
    /// Optional root MESA `data` directory for raw ASCII `.data` loading.
    pub data_dir: Option<PathBuf>,
    tables: [OnceLock<Arc<CmsXTable>>; CMS_NUM_XS],
}

static BUILTIN_CMS_TABLES: OnceLock<CmsTableSet> = OnceLock::new();

impl CmsTableSet {
    /// Returns a reference to the lazily initialized `CmsTableSet`.
    #[must_use]
    pub fn builtin() -> &'static Self {
        BUILTIN_CMS_TABLES.get_or_init(Self::load_builtin)
    }

    /// Creates a `CmsTableSet` with an explicit MESA `data` directory.
    #[must_use]
    pub fn with_data_dir(data_dir: impl Into<PathBuf>) -> Self {
        let mut set = Self::load_builtin();
        set.data_dir = Some(data_dir.into());
        set
    }

    fn load_builtin() -> Self {
        Self {
            data_dir: resolve_mesa_data_dir(),
            tables: std::array::from_fn(|_| OnceLock::new()),
        }
    }

    /// Retrieves or lazily loads the 1-based `ix` (`1..=11`) `CMS` table.
    pub fn get_table(
        &self,
        ix_1based: usize,
    ) -> Result<Arc<CmsXTable>, EosError> {
        if !(1..=CMS_NUM_XS).contains(&ix_1based) {
            return Err(EosError::InvalidState {
                context: "CmsTableSet::get_table",
                detail: format!("invalid CMS ix_1based={ix_1based}"),
            });
        }
        let slot = &self.tables[ix_1based - 1];
        if let Some(tbl) = slot.get() {
            return Ok(Arc::clone(tbl));
        }
        let stem = format!("mesa-CMS_{}x", CMS_XSTR[ix_1based - 1]);
        let loaded =
            Arc::new(load_or_build_cms_slice(self.data_dir.as_deref(), &stem)?);
        let _ = slot.set(Arc::clone(&loaded));
        Ok(slot.get().map_or(loaded, Arc::clone))
    }
}
