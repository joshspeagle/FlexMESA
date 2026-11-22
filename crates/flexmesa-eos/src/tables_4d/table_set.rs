// SPDX-License-Identifier: LGPL-3.0-or-later
//! Composition grid metadata (`DtXzInfo`) and lazy-loaded 4D table collection (`EosDtTableSet`).

use crate::helm::table_io::resolve_mesa_data_dir;
use crate::tables_4d::slice_cache::{
    load_or_build_dt_slice_with_cache_path, reload_verified_dt_slice_streaming,
};
use crate::tables_4d::spline::{EosDtTableKind, EosDtXzTable};
use crate::types::EosError;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// Maximum number of 4D EOS `(kind, ix, iz)` slices (`OPAL/SCVH` and `FreeEOS`
/// combined) retained simultaneously in RAM.
///
/// Each full `(logQ, logT)` slice occupies `~78.75 MiB`: capping the unified
/// Clock-LRU pool at `52` slices limits resident 4D table memory to `~4.00 GiB`
/// regardless of whether a run uses `OPAL/SCVH`, `FreeEOS`, or both, fitting
/// cleanly inside the `5.5-6.0 GiB` budget alongside `HELM` and solver buffers
/// while holding the full active `fix_d_eos_dxa_partials` working set.
const MAX_CACHED_4D_TABLES: usize = 52;

/// Thread-local fast-path capacity (`16` weak handles covering stacked
/// `FreeEOS` and `OPAL/SCVH` `(2 Z x 4 X)` cubic stencils).
const TL_SLICE_CACHE_CAP: usize = 16;

static NEXT_SET_ID: AtomicU64 = AtomicU64::new(1);

type SliceKey = (EosDtTableKind, usize, usize);
type TlSliceKey = (u64, EosDtTableKind, usize, usize);

#[derive(Debug)]
struct CachedDtEntry {
    table: Arc<EosDtXzTable>,
    referenced: AtomicBool,
}

impl CachedDtEntry {
    fn new(table: Arc<EosDtXzTable>) -> Arc<Self> {
        Arc::new(Self {
            table,
            referenced: AtomicBool::new(true),
        })
    }
}

thread_local! {
    static TL_SLICE_CACHE: RefCell<VecDeque<(TlSliceKey, Weak<CachedDtEntry>)>> =
        RefCell::new(VecDeque::with_capacity(TL_SLICE_CACHE_CAP));
}

fn evict_clock_lru(
    queue: &mut VecDeque<(SliceKey, Arc<CachedDtEntry>)>,
    target_len: usize,
) {
    while queue.len() > target_len {
        let Some((k, entry)) = queue.pop_back() else {
            break;
        };
        if entry.referenced.swap(false, Ordering::Relaxed) {
            queue.push_front((k, entry));
        }
    }
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

/// Collection of `OPAL/SCVH` and `FreeEOS` 4D tables backed by a unified
/// Clock-LRU slice cache and weak thread-local fast-path handles.
#[derive(Debug)]
pub struct EosDtTableSet {
    set_id: u64,
    /// Composition grid for `OPAL/SCVH`.
    pub opal_xz: DtXzInfo,
    /// Composition grid for `FreeEOS`.
    pub free_xz: DtXzInfo,
    /// Optional root MESA `data` directory for raw ASCII `.data` loading.
    pub data_dir: Option<PathBuf>,
    /// Unified Clock-LRU cache of `(kind, ix_1based, iz_1based)` slices
    /// capped at `MAX_CACHED_4D_TABLES`.
    slices: Mutex<VecDeque<(SliceKey, Arc<CachedDtEntry>)>>,
    /// Derived cache paths already SHA-256-verified in this table set for
    /// low-overhead streaming reload after Clock-LRU eviction.
    verified_cache_paths: Mutex<HashMap<SliceKey, PathBuf>>,
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
        Self {
            set_id: NEXT_SET_ID.fetch_add(1, Ordering::Relaxed),
            opal_xz: DtXzInfo::opal_scvh(),
            free_xz: DtXzInfo::free_eos(),
            data_dir: resolve_mesa_data_dir(),
            slices: Mutex::new(VecDeque::with_capacity(MAX_CACHED_4D_TABLES)),
            verified_cache_paths: Mutex::new(HashMap::new()),
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
        let xz = match kind {
            EosDtTableKind::OpalScvh => &self.opal_xz,
            EosDtTableKind::FreeEos => &self.free_xz,
        };
        let valid_idx = ix_1based >= 1
            && iz_1based >= 1
            && xz
                .xs_for_z
                .get(iz_1based - 1)
                .is_some_and(|row| ix_1based <= row.len());
        if !valid_idx {
            return Err(EosError::InvalidState {
                context: "EosDtTableSet::get_table",
                detail: format!(
                    "invalid {kind:?} indices (ix={ix_1based}, iz={iz_1based})"
                ),
            });
        }

        let tl_key = (self.set_id, kind, ix_1based, iz_1based);
        if let Some(hit) = Self::lookup_tl_slice(tl_key) {
            return Ok(Arc::clone(&hit.table));
        }

        let key = (kind, ix_1based, iz_1based);
        if let Some(hit) = self.lookup_slice_cache(key) {
            Self::record_tl_slice(tl_key, &hit);
            return Ok(Arc::clone(&hit.table));
        }

        let load_guard = self
            .load_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(hit) = self.lookup_slice_cache(key) {
            drop(load_guard);
            Self::record_tl_slice(tl_key, &hit);
            return Ok(Arc::clone(&hit.table));
        }

        let verified_path = self
            .verified_cache_paths
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&key)
            .cloned();
        let reloaded = if let Some(ref cache_file) = verified_path {
            if let Ok(tbl) = reload_verified_dt_slice_streaming(cache_file) {
                Some(tbl)
            } else {
                let _ = self
                    .verified_cache_paths
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .remove(&key);
                None
            }
        } else {
            None
        };

        let loaded = if let Some(tbl) = reloaded {
            tbl
        } else {
            let stem = eosdt_table_stem(kind, xz, ix_1based, iz_1based);
            let (tbl, new_cache_path) = load_or_build_dt_slice_with_cache_path(
                self.data_dir.as_deref(),
                kind,
                &stem,
            )?;
            if let Some(cache_file) = new_cache_path {
                let _ = self
                    .verified_cache_paths
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .insert(key, cache_file);
            }
            tbl
        };

        let entry = CachedDtEntry::new(Arc::new(loaded));
        {
            let mut guard = self
                .slices
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(pos) = guard.iter().position(|(k, _)| *k == key) {
                if let Some((k, existing)) = guard.remove(pos) {
                    existing.referenced.store(true, Ordering::Relaxed);
                    guard.push_front((k, Arc::clone(&existing)));
                    drop(guard);
                    drop(load_guard);
                    Self::record_tl_slice(tl_key, &existing);
                    return Ok(Arc::clone(&existing.table));
                }
            }
            evict_clock_lru(&mut guard, MAX_CACHED_4D_TABLES.saturating_sub(1));
            guard.push_front((key, Arc::clone(&entry)));
        }
        drop(load_guard);
        Self::record_tl_slice(tl_key, &entry);
        Ok(Arc::clone(&entry.table))
    }

    fn lookup_tl_slice(tl_key: TlSliceKey) -> Option<Arc<CachedDtEntry>> {
        TL_SLICE_CACHE.with(|tl| {
            let mut cache = tl.borrow_mut();
            let pos = cache.iter().position(|(k, _)| *k == tl_key)?;
            let (_, weak) = cache.remove(pos)?;
            let entry = weak.upgrade()?;
            if !entry.referenced.load(Ordering::Relaxed) {
                entry.referenced.store(true, Ordering::Relaxed);
            }
            cache.push_front((tl_key, weak));
            Some(entry)
        })
    }

    fn lookup_slice_cache(&self, key: SliceKey) -> Option<Arc<CachedDtEntry>> {
        let mut guard = self
            .slices
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let pos = guard.iter().position(|(k, _)| *k == key)?;
        let (k, entry) = guard.remove(pos)?;
        if !entry.referenced.load(Ordering::Relaxed) {
            entry.referenced.store(true, Ordering::Relaxed);
        }
        let cloned = Arc::clone(&entry);
        guard.push_front((k, entry));
        Some(cloned)
    }

    fn record_tl_slice(tl_key: TlSliceKey, entry: &Arc<CachedDtEntry>) {
        TL_SLICE_CACHE.with(|tl| {
            let mut cache = tl.borrow_mut();
            if let Some(pos) = cache.iter().position(|(k, _)| *k == tl_key) {
                let _ = cache.remove(pos);
            }
            cache.push_front((tl_key, Arc::downgrade(entry)));
            while cache.len() > TL_SLICE_CACHE_CAP {
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
    fn test_tl_slice_cache_uses_weak_refs_marks_referenced_and_deduplicates_keys(
    ) {
        let set_id = u64::MAX - 1;
        let key = (set_id, EosDtTableKind::FreeEos, 1, 1);
        let entry = CachedDtEntry::new(dummy_table());
        entry.referenced.store(false, Ordering::Relaxed);
        let probe = Arc::downgrade(&entry);

        EosDtTableSet::record_tl_slice(key, &entry);
        EosDtTableSet::record_tl_slice(key, &entry);
        let count_for_key = TL_SLICE_CACHE
            .with(|tl| tl.borrow().iter().filter(|(k, _)| *k == key).count());
        assert_eq!(count_for_key, 1);

        let hit = EosDtTableSet::lookup_tl_slice(key);
        assert!(hit.is_some());
        assert!(entry.referenced.load(Ordering::Relaxed));
        drop(hit);
        drop(entry);

        // Once all strong Arc owners are dropped, the thread-local Weak
        // handle must not keep the entry alive and lookup must return None.
        assert_eq!(probe.strong_count(), 0);
        assert!(EosDtTableSet::lookup_tl_slice(key).is_none());
    }

    #[test]
    fn test_clock_lru_protects_tl_referenced_hot_slice_from_eviction() {
        let mut queue: VecDeque<(SliceKey, Arc<CachedDtEntry>)> =
            VecDeque::new();
        let k_cold_1 = (EosDtTableKind::FreeEos, 1, 1);
        let k_cold_2 = (EosDtTableKind::FreeEos, 2, 1);
        let k_hot = (EosDtTableKind::FreeEos, 3, 1);

        let e_cold_1 = CachedDtEntry::new(dummy_table());
        let e_cold_2 = CachedDtEntry::new(dummy_table());
        let e_hot = CachedDtEntry::new(dummy_table());

        // Clear initial referenced flags so only a subsequent TLS hit marks e_hot.
        e_cold_1.referenced.store(false, Ordering::Relaxed);
        e_cold_2.referenced.store(false, Ordering::Relaxed);
        e_hot.referenced.store(false, Ordering::Relaxed);

        // Place e_hot at the back of the global queue (oldest position).
        queue.push_front((k_hot, Arc::clone(&e_hot)));
        queue.push_front((k_cold_1, e_cold_1));
        queue.push_front((k_cold_2, e_cold_2));

        // Simulate a thread-local fast-path hit setting referenced without
        // mutating queue order.
        let tl_key = (u64::MAX - 2, EosDtTableKind::FreeEos, 3, 1);
        EosDtTableSet::record_tl_slice(tl_key, &e_hot);
        assert!(EosDtTableSet::lookup_tl_slice(tl_key).is_some());

        evict_clock_lru(&mut queue, 2);
        let remaining: Vec<SliceKey> = queue.iter().map(|(k, _)| *k).collect();
        assert_eq!(remaining.len(), 2);
        assert!(remaining.contains(&k_hot));
        assert!(!remaining.contains(&k_cold_1));

        // Clock sweep clears the second-chance bit on the surviving hot slice.
        assert!(!e_hot.referenced.load(Ordering::Relaxed));
    }
}
