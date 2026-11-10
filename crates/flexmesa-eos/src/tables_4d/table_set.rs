// SPDX-License-Identifier: LGPL-3.0-or-later
//! Composition grid metadata (`DtXzInfo`) and lazy-loaded 4D table collection (`EosDtTableSet`).

use crate::helm::table_io::resolve_mesa_data_dir;
use crate::tables_4d::slice_cache::load_or_build_dt_slice;
use crate::tables_4d::spline::{EosDtTableKind, EosDtVertex, EosDtXzTable};
use crate::types::EosError;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock, RwLock};

/// Built-in binary patch table for `OPAL/SCVH`, `FreeEOS`, and `CMS` test coverage.
pub(crate) const BUILTIN_4D_PATCHES: &[u8] =
    include_bytes!("../../data/eos_tables_4d_patches.bin");

/// Composition $(X, Z)$ grid layout for a 4D tabulated EOS (`DT_xz_Info`).
#[derive(Debug, Clone)]
pub struct DtXzInfo {
    /// Metallicities $Z$ (`nZs`).
    pub zs: Vec<f64>,
    /// Hydrogen mass fractions $X$ available for each $Z$ index (`Xs_for_Z`).
    pub xs_for_z: Vec<Vec<f64>>,
}

impl DtXzInfo {
    /// Standard `OPAL/SCVH` composition grid (`nZs = 3`, `nXs = [6, 5, 5]`).
    #[must_use]
    pub fn opal_scvh() -> Self {
        let zs = vec![0.00, 0.02, 0.04];
        let xs_z0 = vec![0.00, 0.20, 0.40, 0.60, 0.80, 1.00];
        let xs_zpos = vec![0.00, 0.20, 0.40, 0.60, 0.80];
        let xs_for_z = vec![xs_z0, xs_zpos.clone(), xs_zpos];
        Self { zs, xs_for_z }
    }

    /// Standard `FreeEOS` composition grid (`nZs = 15`).
    #[must_use]
    pub fn free_eos() -> Self {
        let zs = vec![
            0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60,
            0.70, 0.80, 0.90, 1.00,
        ];
        let mut xs_for_z = Vec::with_capacity(zs.len());
        for (i, &z) in zs.iter().enumerate() {
            if i < 5 {
                xs_for_z.push(vec![
                    0.0,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0 - z,
                ]);
            } else if i == 5 {
                xs_for_z.push(vec![0.0, 0.1, 0.4, 0.9]);
            } else if i < 13 {
                xs_for_z.push(vec![0.0, 0.1, 1.0 - z]);
            } else if i == 13 {
                xs_for_z.push(vec![0.0, 0.1]);
            } else {
                xs_for_z.push(vec![0.0]);
            }
        }
        Self { zs, xs_for_z }
    }
}

/// Formats a mass fraction $v \in [0, 1]$ into a two- or three-digit string (`setstr`).
#[must_use]
pub fn format_xz_str(v: f64) -> String {
    if v > 0.99999 {
        "100".to_string()
    } else if v > 0.09999 {
        format!("{:02}", (100.0 * v + 0.5).floor() as i32)
    } else {
        format!("0{}", (100.0 * v + 0.5).floor() as i32)
    }
}

/// Returns the table filename stem (e.g. `mesa-eosDT_02z60x`) for `(kind, ix_1based, iz_1based)`.
#[must_use]
pub fn eosdt_table_stem(
    kind: EosDtTableKind,
    xz: &DtXzInfo,
    ix_1based: usize,
    iz_1based: usize,
) -> String {
    let z = xz.zs[iz_1based - 1];
    let x = xz.xs_for_z[iz_1based - 1][ix_1based - 1];
    let z_str = format_xz_str(z);
    let x_str = format_xz_str(x);
    match kind {
        EosDtTableKind::OpalScvh => format!("mesa-eosDT_{z_str}z{x_str}x"),
        EosDtTableKind::FreeEos => {
            let suffix = if iz_1based >= 13 { "_CO" } else { "" };
            format!("mesa-FreeEOS_{z_str}z{x_str}x{suffix}")
        }
    }
}

/// Returns the relative `.data` file path under a MESA `data` directory (`Get_eosDT_Table_Filenames`).
#[must_use]
pub fn eosdt_relative_data_path(
    kind: EosDtTableKind,
    xz: &DtXzInfo,
    ix_1based: usize,
    iz_1based: usize,
) -> String {
    let stem = eosdt_table_stem(kind, xz, ix_1based, iz_1based);
    match kind {
        EosDtTableKind::OpalScvh => format!("eosDT_data/{stem}.data"),
        EosDtTableKind::FreeEos => format!("eosFreeEOS_data/{stem}.data"),
    }
}

/// Collection of `OPAL/SCVH` and `FreeEOS` 4D tables with lazy per-`(ix, iz)` loading.
#[derive(Debug)]
pub struct EosDtTableSet {
    /// Composition grid for `OPAL/SCVH`.
    pub opal_xz: DtXzInfo,
    /// Composition grid for `FreeEOS`.
    pub free_xz: DtXzInfo,
    /// Optional root MESA `data` directory for on-demand `.data` loading.
    pub data_dir: Option<PathBuf>,
    /// Lock-free built-in `OPAL/SCVH` patch tables indexed by `[iz - 1][ix - 1]`.
    builtin_opal: Vec<Vec<Option<Arc<EosDtXzTable>>>>,
    /// Lock-free built-in `FreeEOS` patch tables indexed by `[iz - 1][ix - 1]`.
    builtin_free: Vec<Vec<Option<Arc<EosDtXzTable>>>>,
    /// Lock-free lazily loaded full `OPAL/SCVH` slices indexed by `[iz - 1][ix - 1]`.
    dense_opal: Vec<Vec<OnceLock<Arc<EosDtXzTable>>>>,
    /// Lock-free lazily loaded full `FreeEOS` slices indexed by `[iz - 1][ix - 1]`.
    dense_free: Vec<Vec<OnceLock<Arc<EosDtXzTable>>>>,
    /// Loaded or sparse-patched `(kind, ix_1based, iz_1based)` tables.
    tables: RwLock<HashMap<(EosDtTableKind, usize, usize), Arc<EosDtXzTable>>>,
}

static BUILTIN_DT_TABLES: OnceLock<EosDtTableSet> = OnceLock::new();

impl EosDtTableSet {
    /// Returns a reference to the lazily initialized built-in `EosDtTableSet`.
    #[must_use]
    pub fn builtin() -> &'static Self {
        BUILTIN_DT_TABLES.get_or_init(Self::load_builtin)
    }

    /// Creates a new `EosDtTableSet` backed by the built-in patches and an explicit `data_dir`.
    #[must_use]
    pub fn with_data_dir(data_dir: impl Into<PathBuf>) -> Self {
        let mut set = Self::load_builtin();
        set.data_dir = Some(data_dir.into());
        set
    }

    fn load_builtin() -> Self {
        let opal_xz = DtXzInfo::opal_scvh();
        let free_xz = DtXzInfo::free_eos();
        let mut map: HashMap<(EosDtTableKind, usize, usize), EosDtXzTable> =
            HashMap::new();

        if BUILTIN_4D_PATCHES.len() >= 4 {
            let count = u32::from_le_bytes(
                BUILTIN_4D_PATCHES[0..4].try_into().unwrap_or([0; 4]),
            ) as usize;
            let mut pos = 4;
            for _ in 0..count {
                if pos + 8 + 64 * 8 > BUILTIN_4D_PATCHES.len() {
                    break;
                }
                let kind = if BUILTIN_4D_PATCHES[pos] == 0 {
                    EosDtTableKind::OpalScvh
                } else {
                    EosDtTableKind::FreeEos
                };
                let ix = BUILTIN_4D_PATCHES[pos + 1] as usize;
                let iz = BUILTIN_4D_PATCHES[pos + 2] as usize;
                let iq = u16::from_le_bytes([
                    BUILTIN_4D_PATCHES[pos + 4],
                    BUILTIN_4D_PATCHES[pos + 5],
                ]);
                let jt = u16::from_le_bytes([
                    BUILTIN_4D_PATCHES[pos + 6],
                    BUILTIN_4D_PATCHES[pos + 7],
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

                let xz = match kind {
                    EosDtTableKind::OpalScvh => &opal_xz,
                    EosDtTableKind::FreeEos => &free_xz,
                };
                let entry = map.entry((kind, ix, iz)).or_insert_with(|| {
                    let z = xz.zs.get(iz - 1).copied().unwrap_or(0.0);
                    let x = xz
                        .xs_for_z
                        .get(iz - 1)
                        .and_then(|v| v.get(ix - 1))
                        .copied()
                        .unwrap_or(0.0);
                    EosDtXzTable::new_standard_header(x, z)
                });
                entry.patches.insert((iq, jt), EosDtVertex { coeffs });
            }
        }

        let mut builtin_opal: Vec<Vec<Option<Arc<EosDtXzTable>>>> = opal_xz
            .xs_for_z
            .iter()
            .map(|xs| vec![None; xs.len()])
            .collect();
        let mut builtin_free: Vec<Vec<Option<Arc<EosDtXzTable>>>> = free_xz
            .xs_for_z
            .iter()
            .map(|xs| vec![None; xs.len()])
            .collect();
        let dense_opal: Vec<Vec<OnceLock<Arc<EosDtXzTable>>>> = opal_xz
            .xs_for_z
            .iter()
            .map(|xs| (0..xs.len()).map(|_| OnceLock::new()).collect())
            .collect();
        let dense_free: Vec<Vec<OnceLock<Arc<EosDtXzTable>>>> = free_xz
            .xs_for_z
            .iter()
            .map(|xs| (0..xs.len()).map(|_| OnceLock::new()).collect())
            .collect();

        let mut arc_map = HashMap::with_capacity(map.len());
        for ((kind, ix, iz), mut tbl) in map {
            tbl.finalize_patches();
            let arc_tbl = Arc::new(tbl);
            let grid = match kind {
                EosDtTableKind::OpalScvh => &mut builtin_opal,
                EosDtTableKind::FreeEos => &mut builtin_free,
            };
            if let Some(row) = grid.get_mut(iz.wrapping_sub(1)) {
                if let Some(slot) = row.get_mut(ix.wrapping_sub(1)) {
                    *slot = Some(Arc::clone(&arc_tbl));
                }
            }
            arc_map.insert((kind, ix, iz), arc_tbl);
        }

        Self {
            opal_xz,
            free_xz,
            data_dir: resolve_mesa_data_dir(),
            builtin_opal,
            builtin_free,
            dense_opal,
            dense_free,
            tables: RwLock::new(arc_map),
        }
    }

    #[inline]
    fn dense_slot(
        &self,
        kind: EosDtTableKind,
        ix_1based: usize,
        iz_1based: usize,
    ) -> Option<&OnceLock<Arc<EosDtXzTable>>> {
        let grid = match kind {
            EosDtTableKind::OpalScvh => &self.dense_opal,
            EosDtTableKind::FreeEos => &self.dense_free,
        };
        grid.get(iz_1based.wrapping_sub(1))
            .and_then(|row| row.get(ix_1based.wrapping_sub(1)))
    }

    #[inline]
    fn builtin_arc(
        &self,
        kind: EosDtTableKind,
        ix_1based: usize,
        iz_1based: usize,
    ) -> Option<&Arc<EosDtXzTable>> {
        if let Some(slot) = self.dense_slot(kind, ix_1based, iz_1based) {
            if let Some(tbl) = slot.get() {
                return Some(tbl);
            }
        }
        if self.data_dir.is_some() {
            return None;
        }
        let grid = match kind {
            EosDtTableKind::OpalScvh => &self.builtin_opal,
            EosDtTableKind::FreeEos => &self.builtin_free,
        };
        grid.get(iz_1based.wrapping_sub(1))
            .and_then(|row| row.get(ix_1based.wrapping_sub(1)))
            .and_then(Option::as_ref)
    }

    /// Fast lock-free reference lookup for loaded full slices or built-in patch tables.
    #[inline]
    #[must_use]
    pub fn get_table_fast(
        &self,
        kind: EosDtTableKind,
        ix_1based: usize,
        iz_1based: usize,
    ) -> Option<&EosDtXzTable> {
        self.builtin_arc(kind, ix_1based, iz_1based)
            .map(Arc::as_ref)
    }

    /// Retrieves or lazily loads the `(ix_1based, iz_1based)` table for `kind`.
    pub fn get_table(
        &self,
        kind: EosDtTableKind,
        ix_1based: usize,
        iz_1based: usize,
    ) -> Result<Arc<EosDtXzTable>, EosError> {
        if let Some(tbl) = self.builtin_arc(kind, ix_1based, iz_1based) {
            return Ok(Arc::clone(tbl));
        }
        let key = (kind, ix_1based, iz_1based);
        if let Ok(guard) = self.tables.read() {
            if let Some(tbl) = guard.get(&key) {
                if tbl.dense.is_some() || self.data_dir.is_none() {
                    return Ok(Arc::clone(tbl));
                }
            }
        }

        if let Some(dir) = &self.data_dir {
            let xz = match kind {
                EosDtTableKind::OpalScvh => &self.opal_xz,
                EosDtTableKind::FreeEos => &self.free_xz,
            };
            let stem = eosdt_table_stem(kind, xz, ix_1based, iz_1based);
            if let Ok(loaded_tbl) = load_or_build_dt_slice(dir, kind, &stem) {
                let loaded = Arc::new(loaded_tbl);
                if let Some(slot) = self.dense_slot(kind, ix_1based, iz_1based)
                {
                    let _ = slot.set(Arc::clone(&loaded));
                }
                if let Ok(mut guard) = self.tables.write() {
                    guard.insert(key, Arc::clone(&loaded));
                }
                return Ok(loaded);
            }
        }

        if let Ok(guard) = self.tables.read() {
            if let Some(tbl) = guard.get(&key) {
                return Ok(Arc::clone(tbl));
            }
        }

        Err(EosError::InvalidState {
            context: "EosDtTableSet::get_table",
            detail: format!(
                "table ({kind:?}, ix={ix_1based}, iz={iz_1based}) is not loaded"
            ),
        })
    }
}
