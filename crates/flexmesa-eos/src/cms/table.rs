// SPDX-License-Identifier: LGPL-3.0-or-later
//! Configuration, blend-fraction (`get_cms_alfa`), and table loader for `CMS` (`eoscms_load_tables.f90`).

use crate::tables_4d::spline::{EosDtVertex, NUM_EOS_TABLE_VALS};
use crate::tables_4d::table_set::BUILTIN_4D_PATCHES;
use crate::types::{
    EosError, I_CHI_RHO, I_CHI_T, I_CP, I_CV, I_DE_DRHO, I_DS_DRHO, I_DS_DT,
    I_ETA, I_GAMMA1, I_GAMMA3, I_GRAD_AD, I_LN_E, I_LN_FREE_E, I_LN_PGAS,
    I_LN_S, I_MU,
};
use flexmesa_autodiff::AutoDiffReal2VarOrder1;
use flexmesa_const::LN10;
use flexmesa_interp2d::{interp_mkbicub_db, BicubicBc};
use flexmesa_math::safe_log;
use flexmesa_utils::is_bad;
use std::collections::HashMap;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};

/// Number of hydrogen mass fraction tables in `CMS` (`num_eosCMS_Xs = 11`).
pub const CMS_NUM_XS: usize = 11;

/// Hydrogen mass fractions for the 11 `CMS` tables (`0.00, 0.10, ..., 1.00`).
pub const CMS_XVALS: [f64; CMS_NUM_XS] =
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

/// Three-digit filename tags for the 11 `CMS` tables (`000` through `100`).
pub const CMS_XSTR: [&str; CMS_NUM_XS] = [
    "000", "010", "020", "030", "040", "050", "060", "070", "080", "090", "100",
];

/// Configuration for the `CMS` equation of state (`eos.defaults`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CmsConfig {
    /// Use a fixed composition index (`0..=10`) instead of interpolating in $X$ (default `false`).
    pub use_fixed_composition: bool,
    /// Fixed composition index (`0..=10`) when `use_fixed_composition` is `true` (default `-1`).
    pub fixed_composition_index: isize,
    /// Use 4-point piecewise monotonic cubic interpolation in $X$ (default `false`).
    pub cubic_in_x: bool,
    /// Include blackbody radiation pressure/energy/entropy in `CMS` (default `false`).
    pub include_radiation: bool,
    /// Metallicity $Z$ above which `CMS` has zero weight (default `-1.0`, disabled).
    pub max_z_for_any_cms: f64,
    /// Metallicity $Z$ below which `CMS` has full weight (default `-1.0`, disabled).
    pub max_z_for_all_cms: f64,
    /// Lower $\log_{10} Q$ cutoff for any `CMS` weight (default `-100.0`).
    pub logq_min_for_any_cms: f64,
    /// Lower $\log_{10} Q$ threshold for full `CMS` weight (default `-100.0`).
    pub logq_min_for_all_cms: f64,
    /// Upper $\log_{10} Q$ threshold for full `CMS` weight (default `100.0`).
    pub logq_max_for_all_cms: f64,
    /// Upper $\log_{10} Q$ cutoff for any `CMS` weight (default `100.0`).
    pub logq_max_for_any_cms: f64,
    /// Lower $\log_{10} \rho$ cutoff for any `CMS` weight (default `-8.0`).
    pub logrho_min_for_any_cms: f64,
    /// Lower $\log_{10} \rho$ threshold for full `CMS` weight (default `-7.9`).
    pub logrho_min_for_all_cms: f64,
    /// Upper $\log_{10} \rho$ threshold for full `CMS` weight (default `5.9`).
    pub logrho_max_for_all_cms: f64,
    /// Upper $\log_{10} \rho$ cutoff for any `CMS` weight (default `6.0`).
    pub logrho_max_for_any_cms: f64,
    /// Lower $\log_{10} T$ cutoff for any `CMS` weight (default `2.0`).
    pub logt_min_for_any_cms: f64,
    /// Lower $\log_{10} T$ threshold for full `CMS` weight (default `2.1`).
    pub logt_min_for_all_cms: f64,
    /// Upper $\log_{10} T$ threshold for full `CMS` weight (default `7.9`).
    pub logt_max_for_all_cms: f64,
    /// Upper $\log_{10} T$ cutoff for any `CMS` weight (default `8.0`).
    pub logt_max_for_any_cms: f64,
    /// Upper $\log_{10} T$ threshold for full `CMS` in the lower-right corner (default `5.5`).
    pub logt_low_max_for_all_cms: f64,
    /// Upper $\log_{10} T$ cutoff for any `CMS` in the lower-right corner (default `5.6`).
    pub logt_low_max_for_any_cms: f64,
    /// Lower $\log_{10} \rho$ threshold for the high-$T$ region (default `2.2`).
    pub logrho_high_min_for_all_cms: f64,
    /// Lower $\log_{10} \rho$ cutoff for the high-$T$ region (default `2.1`).
    pub logrho_high_min_for_any_cms: f64,
}

impl Default for CmsConfig {
    fn default() -> Self {
        Self {
            use_fixed_composition: false,
            fixed_composition_index: -1,
            cubic_in_x: false,
            include_radiation: false,
            max_z_for_any_cms: -1.0,
            max_z_for_all_cms: -1.0,
            logq_min_for_any_cms: -100.0,
            logq_min_for_all_cms: -100.0,
            logq_max_for_all_cms: 100.0,
            logq_max_for_any_cms: 100.0,
            logrho_min_for_any_cms: -8.0,
            logrho_min_for_all_cms: -7.9,
            logrho_max_for_all_cms: 5.9,
            logrho_max_for_any_cms: 6.0,
            logt_min_for_any_cms: 2.0,
            logt_min_for_all_cms: 2.1,
            logt_max_for_all_cms: 7.9,
            logt_max_for_any_cms: 8.0,
            logt_low_max_for_all_cms: 5.5,
            logt_low_max_for_any_cms: 5.6,
            logrho_high_min_for_all_cms: 2.2,
            logrho_high_min_for_any_cms: 2.1,
        }
    }
}

fn cms_trapezoid(
    x: AutoDiffReal2VarOrder1,
    min_any: f64,
    min_all: f64,
    max_all: f64,
    max_any: f64,
) -> AutoDiffReal2VarOrder1 {
    if x < min_any {
        AutoDiffReal2VarOrder1::zero()
    } else if x <= min_all {
        (x - min_any) / (min_all - min_any)
    } else if x <= max_all {
        AutoDiffReal2VarOrder1::constant(1.0)
    } else if x < max_any {
        (x - max_any) / (max_all - max_any)
    } else {
        AutoDiffReal2VarOrder1::zero()
    }
}

/// Computes the raw `CMS` blend fraction `(alfa, d_alfa_dlogT, d_alfa_dlogRho)`.
///
/// Mirrors `Get_CMS_alfa` in `eos/private/eoscms_eval.f90` (lines 38..149).
#[must_use]
pub fn get_cms_alfa(
    cfg: &CmsConfig,
    log_rho: f64,
    log_t: f64,
    z: f64,
) -> (f64, f64, f64) {
    let log_rho_ad = AutoDiffReal2VarOrder1::var1(log_rho);
    let log_t_ad = AutoDiffReal2VarOrder1::var2(log_t);
    let log_q_ad = log_rho_ad - 2.0 * log_t_ad + 12.0;

    let blend_log_t = cms_trapezoid(
        log_t_ad,
        cfg.logt_min_for_any_cms,
        cfg.logt_min_for_all_cms,
        cfg.logt_max_for_all_cms,
        cfg.logt_max_for_any_cms,
    );
    let blend_log_rho = cms_trapezoid(
        log_rho_ad,
        cfg.logrho_min_for_any_cms,
        cfg.logrho_min_for_all_cms,
        cfg.logrho_max_for_all_cms,
        cfg.logrho_max_for_any_cms,
    );
    let blend_log_q = cms_trapezoid(
        log_q_ad,
        cfg.logq_min_for_any_cms,
        cfg.logq_min_for_all_cms,
        cfg.logq_max_for_all_cms,
        cfg.logq_max_for_any_cms,
    );

    let blend_z = if cfg.max_z_for_any_cms <= 0.0
        || cfg.max_z_for_all_cms <= 0.0
        || z <= cfg.max_z_for_all_cms
    {
        AutoDiffReal2VarOrder1::constant(1.0)
    } else if z >= cfg.max_z_for_any_cms {
        AutoDiffReal2VarOrder1::zero()
    } else {
        AutoDiffReal2VarOrder1::constant(
            (z - cfg.max_z_for_any_cms)
                / (cfg.max_z_for_all_cms - cfg.max_z_for_any_cms),
        )
    };

    let blend = blend_z * blend_log_t * blend_log_rho * blend_log_q;
    (1.0 - blend.val, -blend.d1[1], -blend.d1[0])
}

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
    pub fn from_ascii_reader<R: BufRead>(reader: R) -> Result<Self, EosError> {
        let mut lines = reader
            .lines()
            .map(|l| {
                l.map_err(|e| EosError::InvalidState {
                    context: "CmsXTable::from_ascii_reader",
                    detail: e.to_string(),
                })
            })
            .filter(|res| match res {
                Ok(s) => !s.trim().is_empty(),
                Err(_) => true,
            });

        let _hdr_names =
            lines.next().ok_or_else(|| EosError::InvalidState {
                context: "CmsXTable::from_ascii_reader",
                detail: "missing header names".to_string(),
            })??;
        let hdr_vals_line =
            lines.next().ok_or_else(|| EosError::InvalidState {
                context: "CmsXTable::from_ascii_reader",
                detail: "missing header values".to_string(),
            })??;
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

        let _col_names =
            lines.next().ok_or_else(|| EosError::InvalidState {
                context: "CmsXTable::from_ascii_reader",
                detail: "missing column names".to_string(),
            })??;

        let ngrid = num_log_ts * num_log_rhos;
        let mut raw_vars = vec![0.0_f64; NUM_EOS_TABLE_VALS * ngrid];

        for it in 0..num_log_ts {
            for jrho in 0..num_log_rhos {
                let row =
                    lines.next().ok_or_else(|| EosError::InvalidState {
                        context: "CmsXTable::from_ascii_reader",
                        detail: format!("missing row at it={it}, jrho={jrho}"),
                    })??;
                let mut toks = row.split_whitespace();
                let _lt = toks.next();
                let _lr = toks.next();
                let mut vals = [0.0_f64; 16];
                for v in &mut vals {
                    let s =
                        toks.next().ok_or_else(|| EosError::InvalidState {
                            context: "CmsXTable::from_ascii_reader",
                            detail: "fewer than 18 columns in CMS row"
                                .to_string(),
                        })?;
                    *v = s.parse::<f64>().map_err(|e| {
                        EosError::InvalidState {
                            context: "CmsXTable::from_ascii_reader",
                            detail: format!("bad float '{s}': {e}"),
                        }
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
        let env_dir =
            std::env::var_os("FLEXMESA_EOS_DATA_DIR").map(PathBuf::from);
        Self {
            data_dir: env_dir,
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
        if let Ok(guard) = self.tables.read() {
            if let Some(tbl) = guard.get(&ix_1based) {
                if tbl.dense.is_some() || self.data_dir.is_none() {
                    return Ok(Arc::clone(tbl));
                }
            }
        }
        if let Some(dir) = &self.data_dir {
            let rel = format!(
                "eosCMS_data/mesa-CMS_{}x.data",
                CMS_XSTR[ix_1based - 1]
            );
            let full_path = dir.join(rel);
            if full_path.exists() {
                let loaded = Arc::new(CmsXTable::from_data_file(&full_path)?);
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
