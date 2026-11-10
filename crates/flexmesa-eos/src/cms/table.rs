// SPDX-License-Identifier: LGPL-3.0-or-later
//! Table loader and lazy-loaded slice collection for `CMS` (`eoscms_load_tables.f90`).

pub use crate::cms::config::{
    get_cms_alfa, CmsConfig, CMS_NUM_XS, CMS_XSTR, CMS_XVALS,
};

use crate::helm::table_io::resolve_mesa_data_dir;
use crate::tables_4d::slice_cache::load_or_build_cms_slice;
use crate::tables_4d::spline::{EosDtVertex, NUM_EOS_TABLE_VALS};
use crate::tables_4d::table_set::BUILTIN_4D_PATCHES;
use crate::types::{
    EosError, I_CHI_RHO, I_CHI_T, I_CP, I_CV, I_DE_DRHO, I_DS_DRHO, I_DS_DT,
    I_ETA, I_GAMMA1, I_GAMMA3, I_GRAD_AD, I_LN_E, I_LN_FREE_E, I_LN_PGAS,
    I_LN_S, I_MU,
};
use flexmesa_const::LN10;
use flexmesa_interp2d::{interp_mkbicub_db, BicubicBc};
use flexmesa_math::safe_log;
use flexmesa_utils::is_bad;
use std::collections::HashMap;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};

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
    /// Sparse `(i_t, j_rho)` (1-based) vertices for built-in test patches.
    pub patches: HashMap<(u16, u16), EosDtVertex>,
    /// Optional full dense grid of `num_log_ts * num_log_rhos` vertices.
    pub dense: Option<Vec<EosDtVertex>>,
}

impl CmsXTable {
    /// Creates an empty `CMS` table header with standard dimensions (`121 x 281`).
    #[must_use]
    pub fn new_standard_header(x: f64) -> Self {
        Self {
            version: 1,
            x,
            num_log_ts: 121,
            log_t_min: 2.0,
            log_t_max: 8.0,
            delta_log_t: 0.05,
            num_log_rhos: 281,
            log_rho_min: -8.0,
            log_rho_max: 6.0,
            delta_log_rho: 0.05,
            patches: HashMap::new(),
            dense: None,
        }
    }

    /// Retrieves a 1-based `(i_t, j_rho)` vertex.
    #[must_use]
    pub fn get_vertex(&self, i_t: usize, j_rho: usize) -> Option<&EosDtVertex> {
        if let Some(dense) = &self.dense {
            let idx = (i_t - 1) * self.num_log_rhos + (j_rho - 1);
            dense.get(idx)
        } else {
            self.patches.get(&(i_t as u16, j_rho as u16))
        }
    }

    /// Parses a `mesa-CMS_*.data` ASCII file and builds its 16 bicubic splines (`load_eosCMS_table`).
    pub fn from_ascii_reader<R: BufRead>(mut reader: R) -> Result<Self, EosError> {
        let mut line_buf = String::with_capacity(512);
        let mut next_non_empty = |buf: &mut String| -> Result<String, EosError> {
            loop {
                buf.clear();
                let n = reader.read_line(buf).map_err(|e| EosError::InvalidState {
                    context: "CmsXTable::from_ascii_reader",
                    detail: e.to_string(),
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

        let ngrid = num_log_ts * num_log_rhos;
        let mut raw_vars = vec![0.0_f64; NUM_EOS_TABLE_VALS * ngrid];

        for it in 0..num_log_ts {
            for jrho in 0..num_log_rhos {
                let row = next_non_empty(&mut line_buf)?;
                let mut toks = row.split_whitespace();
                let _lt = toks.next();
                let _lr = toks.next();
                let mut vals = [0.0_f64; 16];
                for v in &mut vals {
                    let s =
                        toks.next().ok_or_else(|| EosError::InvalidState {
                            context: "CmsXTable::from_ascii_reader",
                            detail: format!("fewer than 18 columns at ({it}, {jrho})"),
                        })?;
                    *v = s.parse::<f64>().map_err(|e| EosError::InvalidState {
                        context: "CmsXTable::from_ascii_reader",
                        detail: format!("bad float '{s}': {e}"),
                    })?;
                }
                let cell = it * num_log_rhos + jrho;
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
        }

        let log_ts: Vec<f64> = (0..num_log_ts)
            .map(|i| log_t_min + (i as f64) * delta_log_t)
            .collect();
        let log_rhos: Vec<f64> = (0..num_log_rhos)
            .map(|j| log_rho_min + (j as f64) * delta_log_rho)
            .collect();

        let mut dense = vec![EosDtVertex::default(); ngrid];
        for v in 0..NUM_EOS_TABLE_VALS {
            let z_slice = &raw_vars[v * ngrid..(v + 1) * ngrid];
            let (coeffs, _, _) = interp_mkbicub_db(
                &log_ts,
                &log_rhos,
                z_slice,
                BicubicBc::NotAKnot,
                &[],
                BicubicBc::NotAKnot,
                &[],
                BicubicBc::NotAKnot,
                &[],
                BicubicBc::NotAKnot,
                &[],
            )
            .map_err(|e| EosError::InvalidState {
                context: "CmsXTable::from_ascii_reader",
                detail: format!("interp_mkbicub_db failed for var {v}: {e}"),
            })?;
            for (cell, c4) in coeffs.iter().enumerate() {
                for (m, &val) in c4.iter().enumerate() {
                    dense[cell].coeffs[v * 4 + m] =
                        if is_bad(val) { 0.0 } else { val };
                }
            }
        }

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
            patches: HashMap::new(),
            dense: Some(dense),
        })
    }

    /// Loads a `CMS` `.data` file from disk.
    pub fn from_data_file(path: &Path) -> Result<Self, EosError> {
        let file =
            std::fs::File::open(path).map_err(|e| EosError::InvalidState {
                context: "CmsXTable::from_data_file",
                detail: format!("{}: {e}", path.display()),
            })?;
        Self::from_ascii_reader(std::io::BufReader::new(file))
    }
}

/// Collection of the 11 `CMS` tables (`ix_1based in 1..=11`) with lazy loading.
#[derive(Debug)]
pub struct CmsTableSet {
    /// Optional root MESA `data` directory.
    pub data_dir: Option<PathBuf>,
    dense_slots: [OnceLock<Arc<CmsXTable>>; CMS_NUM_XS],
    tables: RwLock<HashMap<usize, Arc<CmsXTable>>>,
}

static BUILTIN_CMS_TABLES: OnceLock<CmsTableSet> = OnceLock::new();

impl CmsTableSet {
    /// Returns a reference to the lazily initialized built-in `CmsTableSet`.
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
        let mut map: HashMap<usize, CmsXTable> = HashMap::new();
        if BUILTIN_4D_PATCHES.len() >= 4 {
            let dt_count = u32::from_le_bytes(
                BUILTIN_4D_PATCHES[0..4].try_into().unwrap_or([0; 4]),
            ) as usize;
            let cms_offset = 4 + dt_count * (8 + 64 * 8);
            if cms_offset + 4 <= BUILTIN_4D_PATCHES.len() {
                let cms_count = u32::from_le_bytes(
                    BUILTIN_4D_PATCHES[cms_offset..cms_offset + 4]
                        .try_into()
                        .unwrap_or([0; 4]),
                ) as usize;
                let mut pos = cms_offset + 4;
                for _ in 0..cms_count {
                    if pos + 8 + 64 * 8 > BUILTIN_4D_PATCHES.len() {
                        break;
                    }
                    let ix = u16::from_le_bytes([
                        BUILTIN_4D_PATCHES[pos],
                        BUILTIN_4D_PATCHES[pos + 1],
                    ]) as usize;
                    let it = u16::from_le_bytes([
                        BUILTIN_4D_PATCHES[pos + 2],
                        BUILTIN_4D_PATCHES[pos + 3],
                    ]);
                    let jrho = u16::from_le_bytes([
                        BUILTIN_4D_PATCHES[pos + 4],
                        BUILTIN_4D_PATCHES[pos + 5],
                    ]);
                    pos += 8;
                    let coeffs: [f64; 64] = std::array::from_fn(|k| {
                        let s = pos + 8 * k;
                        f64::from_le_bytes(
                            BUILTIN_4D_PATCHES[s..s + 8]
                                .try_into()
                                .unwrap_or([0; 8]),
                        )
                    });
                    pos += 64 * 8;
                    let x = CMS_XVALS.get(ix - 1).copied().unwrap_or(0.0);
                    let entry = map
                        .entry(ix)
                        .or_insert_with(|| CmsXTable::new_standard_header(x));
                    entry.patches.insert((it, jrho), EosDtVertex { coeffs });
                }
            }
        }
        let arc_map = map.into_iter().map(|(k, v)| (k, Arc::new(v))).collect();
        Self {
            data_dir: resolve_mesa_data_dir(),
            dense_slots: std::array::from_fn(|_| OnceLock::new()),
            tables: RwLock::new(arc_map),
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
        if let Some(tbl) = self.dense_slots[ix_1based - 1].get() {
            return Ok(Arc::clone(tbl));
        }
        if let Ok(guard) = self.tables.read() {
            if let Some(tbl) = guard.get(&ix_1based) {
                if tbl.dense.is_some() || self.data_dir.is_none() {
                    return Ok(Arc::clone(tbl));
                }
            }
        }
        if let Some(dir) = &self.data_dir {
            let stem = format!("mesa-CMS_{}x", CMS_XSTR[ix_1based - 1]);
            if let Ok(loaded_tbl) = load_or_build_cms_slice(dir, &stem) {
                let loaded = Arc::new(loaded_tbl);
                let _ =
                    self.dense_slots[ix_1based - 1].set(Arc::clone(&loaded));
                if let Ok(mut guard) = self.tables.write() {
                    guard.insert(ix_1based, Arc::clone(&loaded));
                }
                return Ok(loaded);
            }
        }
        if let Ok(guard) = self.tables.read() {
            if let Some(tbl) = guard.get(&ix_1based) {
                return Ok(Arc::clone(tbl));
            }
        }
        Err(EosError::InvalidState {
            context: "CmsTableSet::get_table",
            detail: format!("CMS table ix={ix_1based} is not loaded"),
        })
    }
}
