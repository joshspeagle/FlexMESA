// SPDX-License-Identifier: LGPL-3.0-or-later
//! Composition grid metadata (`DtXzInfo`) and lazy-loaded 4D table collection (`EosDtTableSet`).

use crate::helm::table_io::resolve_mesa_data_dir;
use crate::tables_4d::slice_cache::load_or_build_dt_slice;
use crate::tables_4d::spline::{EosDtTableKind, EosDtXzTable};
use crate::types::EosError;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// Maximum number of `FreeEOS` `(ix, iz)` slices retained simultaneously in RAM.
///
/// Each full `(logQ, logT)` slice occupies `~78.75 MiB`: caching all 16
/// `OPAL/SCVH` slices requires `~1.26 GiB`, while capping `FreeEOS` at 24 slices
/// (`~1.89 GiB` max) keeps peak resident memory comfortably below subprocess
/// RSS limits when high-$Z$ C/O core slices are loaded alongside `OPAL/SCVH`.
const MAX_CACHED_FREE_TABLES: usize = 24;

/// Thread-local `FreeEOS` fast-path capacity (`12` weak handles covering one
/// full `(2 Z x 4 X)` cubic stencil plus adjacent interior cells).
const TL_FREE_CACHE_CAP: usize = 12;

static NEXT_SET_ID: AtomicU64 = AtomicU64::new(1);

type TlFreeKey = (u64, usize, usize);

thread_local! {
    static TL_FREE_CACHE: RefCell<VecDeque<(TlFreeKey, Weak<EosDtXzTable>)>> =
        RefCell::new(VecDeque::with_capacity(TL_FREE_CACHE_CAP));
}

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

/// Collection of `OPAL/SCVH` and `FreeEOS` 4D tables with lock-free `OPAL` and LRU `FreeEOS` caching.
#[derive(Debug)]
pub struct EosDtTableSet {
    set_id: u64,
    /// Composition grid for `OPAL/SCVH`.
    pub opal_xz: DtXzInfo,
    /// Composition grid for `FreeEOS`.
    pub free_xz: DtXzInfo,
    /// Optional root MESA `data` directory for raw ASCII `.data` loading.
    pub data_dir: Option<PathBuf>,
    /// Lock-free `[iz][ix]` slots for all 16 `OPAL/SCVH` slices (`~1.26 GiB` total).
    opal_tables: Vec<Vec<OnceLock<Arc<EosDtXzTable>>>>,
    /// True-LRU cache of `(ix_1based, iz_1based)` `FreeEOS` slices (`MAX_CACHED_FREE_TABLES`).
    free_tables: Mutex<VecDeque<((usize, usize), Arc<EosDtXzTable>)>>,
    /// Serializes slice cache misses across Rayon workers so duplicate slices
    /// are never decoded simultaneously in RAM.
    load_lock: Mutex<()>,
}

static BUILTIN_DT_TABLES: OnceLock<EosDtTableSet> = OnceLock::new();

impl EosDtTableSet {
    /// Returns a reference to the lazily initialized `EosDtTableSet`.
    #[must_use]
    pub fn builtin() -> &'static Self {
        BUILTIN_DT_TABLES.get_or_init(Self::load_builtin)
    }

    /// Creates a new `EosDtTableSet` backed by an explicit MESA `data_dir`.
    #[must_use]
    pub fn with_data_dir(data_dir: impl Into<PathBuf>) -> Self {
        let mut set = Self::load_builtin();
        set.data_dir = Some(data_dir.into());
        set
    }

    fn load_builtin() -> Self {
        let opal_xz = DtXzInfo::opal_scvh();
        let opal_tables = opal_xz
            .xs_for_z
            .iter()
            .map(|xs| (0..xs.len()).map(|_| OnceLock::new()).collect())
            .collect();
        Self {
            set_id: NEXT_SET_ID.fetch_add(1, Ordering::Relaxed),
            opal_xz,
            free_xz: DtXzInfo::free_eos(),
            data_dir: resolve_mesa_data_dir(),
            opal_tables,
            free_tables: Mutex::new(VecDeque::with_capacity(
                MAX_CACHED_FREE_TABLES,
            )),
            load_lock: Mutex::new(()),
        }
    }

    /// Retrieves or lazily loads the `(ix_1based, iz_1based)` table for `kind`.
    pub fn get_table(
        &self,
        kind: EosDtTableKind,
        ix_1based: usize,
        iz_1based: usize,
    ) -> Result<Arc<EosDtXzTable>, EosError> {
        if kind == EosDtTableKind::OpalScvh {
            let slot = self
                .opal_tables
                .get(iz_1based.wrapping_sub(1))
                .and_then(|row| row.get(ix_1based.wrapping_sub(1)))
                .ok_or_else(|| EosError::InvalidState {
                    context: "EosDtTableSet::get_table",
                    detail: format!(
                        "invalid OPAL/SCVH indices (ix={ix_1based}, iz={iz_1based})"
                    ),
                })?;
            if let Some(tbl) = slot.get() {
                return Ok(Arc::clone(tbl));
            }
            let _load_guard = self
                .load_lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(tbl) = slot.get() {
                return Ok(Arc::clone(tbl));
            }
            let stem =
                eosdt_table_stem(kind, &self.opal_xz, ix_1based, iz_1based);
            let loaded = Arc::new(load_or_build_dt_slice(
                self.data_dir.as_deref(),
                kind,
                &stem,
            )?);
            let _ = slot.set(Arc::clone(&loaded));
            return Ok(slot.get().map_or(loaded, Arc::clone));
        }

        let tl_key = (self.set_id, ix_1based, iz_1based);
        if let Some(hit) = Self::lookup_tl_free(tl_key) {
            return Ok(hit);
        }

        let key = (ix_1based, iz_1based);
        if let Some(hit) = self.lookup_free_lru(key) {
            Self::record_tl_free(tl_key, &hit);
            return Ok(hit);
        }

        let load_guard = self
            .load_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(hit) = self.lookup_free_lru(key) {
            drop(load_guard);
            Self::record_tl_free(tl_key, &hit);
            return Ok(hit);
        }

        let stem = eosdt_table_stem(kind, &self.free_xz, ix_1based, iz_1based);
        let loaded = Arc::new(load_or_build_dt_slice(
            self.data_dir.as_deref(),
            kind,
            &stem,
        )?);
        {
            let mut guard = self
                .free_tables
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(pos) = guard.iter().position(|(k, _)| *k == key) {
                if let Some(entry) = guard.remove(pos) {
                    let arc = Arc::clone(&entry.1);
                    guard.push_front(entry);
                    drop(guard);
                    drop(load_guard);
                    Self::record_tl_free(tl_key, &arc);
                    return Ok(arc);
                }
            }
            guard.push_front((key, Arc::clone(&loaded)));
            while guard.len() > MAX_CACHED_FREE_TABLES {
                let _ = guard.pop_back();
            }
        }
        drop(load_guard);
        Self::record_tl_free(tl_key, &loaded);
        Ok(loaded)
    }

    fn lookup_tl_free(tl_key: TlFreeKey) -> Option<Arc<EosDtXzTable>> {
        TL_FREE_CACHE.with(|tl| {
            let mut cache = tl.borrow_mut();
            let pos = cache.iter().position(|(k, _)| *k == tl_key)?;
            let entry = cache.remove(pos)?;
            let arc = entry.1.upgrade()?;
            cache.push_front((tl_key, entry.1));
            Some(arc)
        })
    }

    fn lookup_free_lru(&self, key: (usize, usize)) -> Option<Arc<EosDtXzTable>> {
        let mut guard = self
            .free_tables
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let pos = guard.iter().position(|(k, _)| *k == key)?;
        let entry = guard.remove(pos)?;
        let arc = Arc::clone(&entry.1);
        guard.push_front(entry);
        Some(arc)
    }

    fn record_tl_free(tl_key: TlFreeKey, tbl: &Arc<EosDtXzTable>) {
        TL_FREE_CACHE.with(|tl| {
            let mut cache = tl.borrow_mut();
            if let Some(pos) = cache.iter().position(|(k, _)| *k == tl_key) {
                let _ = cache.remove(pos);
            }
            cache.push_front((tl_key, Arc::downgrade(tbl)));
            while cache.len() > TL_FREE_CACHE_CAP {
                let _ = cache.pop_back();
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_table() -> Arc<EosDtXzTable> {
        Arc::new(EosDtXzTable {
            version: 51,
            x: 0.70,
            z: 0.02,
            num_log_ts: 0,
            log_t_min: 0.0,
            log_t_max: 0.0,
            del_log_t: 0.02,
            num_log_qs: 0,
            log_q_min: 0.0,
            log_q_max: 0.0,
            del_log_q: 0.03,
            dense: None,
        })
    }

    #[test]
    fn test_tl_free_cache_uses_weak_refs_and_deduplicates_keys() {
        let set_id = u64::MAX - 1;
        let key = (set_id, 1, 1);
        let strong = dummy_table();
        let probe = Arc::downgrade(&strong);

        EosDtTableSet::record_tl_free(key, &strong);
        EosDtTableSet::record_tl_free(key, &strong);
        let count_for_key = TL_FREE_CACHE.with(|tl| {
            tl.borrow().iter().filter(|(k, _)| *k == key).count()
        });
        assert_eq!(count_for_key, 1);

        let hit = EosDtTableSet::lookup_tl_free(key);
        assert!(hit.is_some());
        drop(hit);
        drop(strong);

        // Once all strong Arc owners are dropped, the thread-local Weak
        // handle must not keep the table alive and lookup must return None.
        assert_eq!(probe.strong_count(), 0);
        assert!(EosDtTableSet::lookup_tl_free(key).is_none());
    }
}
