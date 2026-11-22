// SPDX-License-Identifier: LGPL-3.0-or-later
//! Pure Safe Rust HDF5 loader and cache for MESA `AESOPUS` low-temperature
//! opacity tables (`kap/private/kap_aesopus.f90:41-325` and
//! `kap/private/load_kap.f90:121`).
//!
//! Reads standard MESA/HDF5 `AESOPUS_*.h5` files (superblock v0/v1/v2/v3,
//! object headers v1/v2 with continuations, v1 symbol-table and v2 link groups,
//! contiguous/compact/chunked layouts with DEFLATE/shuffle/fletcher32 filters,
//! and IEEE-754 `f32`/`f64` datasets) without external C libraries or `unsafe`.

use std::collections::HashMap;
use std::fs;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use flexmesa_math::safe_log10;
use flexmesa_utils::{
    is_bad, resolve_flexmesa_data_dir, resolve_mesa_test_inputs_dir,
};

use crate::deflate::zlib_decompress;
use crate::loader::{kap_tables, AesopusData, AesopusSlice, AesopusZSet};
use crate::types::KapError;

const HDF5_MAGIC: &[u8; 8] = b"\x89HDF\r\n\x1a\n";
const BUNDLED_AESOPUS_NAME: &str = "AESOPUS_AGSS09.h5";
const MAX_CACHED_CUSTOM_AESOPUS: usize = 16;

fn io_err(msg: impl Into<String>) -> KapError {
    KapError::TableIo(msg.into())
}

/// Borrowed or shared handle to a loaded [`AesopusData`] table.
pub enum AesopusRef {
    /// Reference to the bundled `AESOPUS_AGSS09.h5` table in [`crate::loader::KapTables`].
    Bundled(&'static AesopusData),
    /// Shared reference to a dynamically loaded custom HDF5 `AESOPUS` table.
    Custom(Arc<AesopusData>),
}

impl Deref for AesopusRef {
    type Target = AesopusData;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Bundled(r) => r,
            Self::Custom(a) => a.as_ref(),
        }
    }
}

static CUSTOM_AESOPUS_CACHE: OnceLock<
    Mutex<HashMap<PathBuf, Arc<AesopusData>>>,
> = OnceLock::new();

fn resolve_aesopus_disk_path(trimmed: &str) -> Option<PathBuf> {
    let direct = Path::new(trimmed);
    if direct.is_file() {
        return Some(direct.to_path_buf());
    }
    if direct.is_absolute() {
        return None;
    }
    let mesa_cand = resolve_mesa_test_inputs_dir()
        .join("data")
        .join("kap_data")
        .join(trimmed);
    if mesa_cand.is_file() {
        return Some(mesa_cand);
    }
    let data_cand = resolve_flexmesa_data_dir().join("kap_data").join(trimmed);
    if data_cand.is_file() {
        return Some(data_cand);
    }
    None
}

/// Returns `true` when `filename` resolves either to the bundled
/// `AESOPUS_AGSS09.h5` table (bare filename) or to an existing file on disk.
#[must_use]
pub fn aesopus_source_exists(filename: &str) -> bool {
    let trimmed = filename.trim();
    if trimmed.is_empty() {
        return false;
    }
    if resolve_aesopus_disk_path(trimmed).is_some() {
        return true;
    }
    !trimmed.contains(['/', '\\']) && trimmed == BUNDLED_AESOPUS_NAME
}

/// Resolves and returns the [`AesopusData`] table for `&kap::AESOPUS_filename`
/// (`kap/private/kap_aesopus.f90:91-110`).
///
/// Precedence:
/// 1. Direct file path (case-relative resolved path or absolute path) if it
///    exists on disk.
/// 2. Bundled `AESOPUS_AGSS09.h5` fast path when `filename` is the bare
///    filename `"AESOPUS_AGSS09.h5"`.
/// 3. `$MESA_DIR/data/kap_data/<filename>` or
///    `FLEXMESA_DATA_DIR/kap_data/<filename>`.
///
/// # Errors
/// Returns [`KapError::InvalidOption`] if `filename` is empty, or
/// [`KapError::TableIo`] if the HDF5 file is missing, unreadable, or malformed.
pub fn resolve_aesopus_data(filename: &str) -> Result<AesopusRef, KapError> {
    let trimmed = filename.trim();
    if trimmed.is_empty() {
        return Ok(AesopusRef::Bundled(&kap_tables()?.aesopus));
    }
    let Some(disk_path) = resolve_aesopus_disk_path(trimmed) else {
        if !trimmed.contains(['/', '\\']) && trimmed == BUNDLED_AESOPUS_NAME {
            return Ok(AesopusRef::Bundled(&kap_tables()?.aesopus));
        }
        return Err(io_err(format!(
            "cannot open AESOPUS_filename '{trimmed}'"
        )));
    };
    let key = fs::canonicalize(&disk_path).unwrap_or(disk_path.clone());
    let cache = CUSTOM_AESOPUS_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(guard) = cache.lock() {
        if let Some(hit) = guard.get(&key) {
            return Ok(AesopusRef::Custom(Arc::clone(hit)));
        }
    }
    let parsed = Arc::new(load_aesopus_hdf5_file(&disk_path)?);
    if let Ok(mut guard) = cache.lock() {
        if guard.len() >= MAX_CACHED_CUSTOM_AESOPUS {
            guard.clear();
        }
        guard.insert(key, Arc::clone(&parsed));
    }
    Ok(AesopusRef::Custom(parsed))
}

/// Loads an `AESOPUS` low-temperature opacity HDF5 file from disk.
///
/// # Errors
/// Returns [`KapError::TableIo`] if reading or decoding the HDF5 file fails.
pub fn load_aesopus_hdf5_file(path: &Path) -> Result<AesopusData, KapError> {
    let bytes = fs::read(path).map_err(|e| {
        io_err(format!(
            "failed to read AESOPUS HDF5 '{}': {e}",
            path.display()
        ))
    })?;
    load_aesopus_hdf5_bytes(&bytes).map_err(|e| {
        io_err(format!(
            "failed to parse AESOPUS HDF5 '{}': {e}",
            path.display()
        ))
    })
}

/// Parses an `AESOPUS` low-temperature opacity HDF5 payload from raw bytes
/// (`kap/private/kap_aesopus.f90:100-325`).
///
/// # Errors
/// Returns [`KapError::TableIo`] if the HDF5 structure, datasets, dimensions,
/// or floating-point values are invalid.
pub fn load_aesopus_hdf5_bytes(bytes: &[u8]) -> Result<AesopusData, KapError> {
    let h5 = Hdf5File::parse(bytes)?;
    let root = h5.list_group_children(h5.root_oh_addr, h5.root_symtab)?;

    let read_root_scalar = |name: &str| -> Result<f64, KapError> {
        let child = root
            .get(name)
            .ok_or_else(|| io_err(format!("missing root dataset '{name}'")))?;
        let ds = h5.read_dataset_f64(child.oh_addr)?;
        if ds.vals.len() != 1 || is_bad(ds.vals[0]) {
            return Err(io_err(format!(
                "expected finite scalar dataset '{name}', got len={}",
                ds.vals.len()
            )));
        }
        Ok(ds.vals[0])
    };
    let read_1d = |children: &HashMap<String, GroupChild>,
                   name: &str,
                   context: &str|
     -> Result<Vec<f64>, KapError> {
        let child = children.get(name).ok_or_else(|| {
            io_err(format!("missing dataset '{context}/{name}'"))
        })?;
        let ds = h5.read_dataset_f64(child.oh_addr)?;
        if ds.dims.len() != 1 || ds.vals.is_empty() {
            return Err(io_err(format!(
                "expected non-empty 1D dataset '{context}/{name}', got dims={:?}",
                ds.dims
            )));
        }
        for (idx, &v) in ds.vals.iter().enumerate() {
            if is_bad(v) {
                return Err(io_err(format!(
                    "non-finite value in '{context}/{name}'[{idx}]"
                )));
            }
        }
        Ok(ds.vals)
    };

    let zsun = read_root_scalar("Zsun")?;
    let fco_ref = read_root_scalar("fCO_ref")?;
    let fc_ref = read_root_scalar("fC_ref")?;
    let fn_ref = read_root_scalar("fN_ref")?;

    let zs = read_1d(&root, "Zs", "")?;
    let log_rs = read_1d(&root, "logRs", "")?;
    let log_ts = read_1d(&root, "logTs", "")?;
    for (label, grid) in [("Zs", &zs), ("logRs", &log_rs), ("logTs", &log_ts)]
    {
        if grid.windows(2).any(|w| w[1] <= w[0]) {
            return Err(io_err(format!(
                "AESOPUS grid '{label}' must be strictly increasing"
            )));
        }
    }
    let log_zs: Vec<f64> = zs.iter().map(|&z| safe_log10(z)).collect();

    let n_lt = log_ts.len();
    let n_lr = log_rs.len();
    let mut ts = Vec::with_capacity(zs.len());

    for &z in &zs {
        // `kap/private/kap_aesopus.f90:157`: `write(Zname, '(f8.6)') ka% Zs(iZ)`
        let z_name = format!("{z:.6}");
        let z_child = root.get(&z_name).ok_or_else(|| {
            io_err(format!("missing AESOPUS metallicity group '{z_name}'"))
        })?;
        let sub = h5.list_group_children(z_child.oh_addr, z_child.symtab)?;
        let xs = read_1d(&sub, "Xs", &z_name)?;
        let fcos = read_1d(&sub, "fCOs", &z_name)?;
        let fcs = read_1d(&sub, "fCs", &z_name)?;
        let fns = read_1d(&sub, "fNs", &z_name)?;

        let kap_child = sub.get("kap").ok_or_else(|| {
            io_err(format!("missing dataset '{z_name}/kap'"))
        })?;
        let kap_ds = h5.read_dataset_f32(kap_child.oh_addr)?;
        let expected_dims =
            [n_lt, n_lr, xs.len(), fcos.len(), fcs.len(), fns.len()];
        if kap_ds.dims.as_slice() != expected_dims {
            return Err(io_err(format!(
                "unexpected '{z_name}/kap' shape {:?}, expected {:?}",
                kap_ds.dims, expected_dims
            )));
        }
        for (idx, &v) in kap_ds.vals.iter().enumerate() {
            if !v.is_finite() {
                return Err(io_err(format!(
                    "non-finite opacity in '{z_name}/kap' at flat index {idx}"
                )));
            }
        }

        // Reorder C-order `(n_lt, n_lr, n_x, n_fco, n_fc, n_fn)` into per-composition
        // slices `(n_lt, n_lr)` matching `scripts/data/build_physics_bundles.py:451`
        // (`np.moveaxis(kap, (0, 1), (-2, -1))`) and `low_t.rs:187`.
        let n_x = xs.len();
        let n_fco = fcos.len();
        let n_fc = fcs.len();
        let n_fn = fns.len();
        let num_slices = n_x * n_fco * n_fc * n_fn;
        let slice_len = n_lt * n_lr;
        let mut slices = Vec::with_capacity(num_slices);
        for ix in 0..n_x {
            for ico in 0..n_fco {
                for ic in 0..n_fc {
                    for in_ in 0..n_fn {
                        let comp_off =
                            ((ix * n_fco + ico) * n_fc + ic) * n_fn + in_;
                        let mut raw = Vec::with_capacity(slice_len);
                        for i_lt in 0..n_lt {
                            for i_lr in 0..n_lr {
                                let src =
                                    (i_lt * n_lr + i_lr) * num_slices
                                        + comp_off;
                                raw.push(kap_ds.vals[src]);
                            }
                        }
                        slices.push(AesopusSlice {
                            raw_f32: Some(raw),
                            spline: OnceLock::new(),
                        });
                    }
                }
            }
        }

        ts.push(AesopusZSet {
            xs,
            fcos,
            fcs,
            fns,
            slices,
        });
    }

    Ok(AesopusData {
        zsun,
        fco_ref,
        fc_ref,
        fn_ref,
        log_rs,
        log_ts,
        zs,
        log_zs,
        ts,
    })
}

// ---------------------------------------------------------------------------
// Minimal Safe Rust HDF5 reader (v0/v1/v2/v3 superblocks, v1/v2 object headers)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct GroupChild {
    oh_addr: u64,
    symtab: Option<(u64, u64)>,
}

struct Hdf5File<'a> {
    bytes: &'a [u8],
    off_size: usize,
    len_size: usize,
    base_addr: u64,
    undef_addr: u64,
    root_oh_addr: u64,
    root_symtab: Option<(u64, u64)>,
}

#[derive(Debug, Clone)]
struct FilterSpec {
    id: u16,
    flags: u16,
    client_data: Vec<u32>,
}

#[derive(Debug, Clone, Copy)]
struct FloatType {
    elem_size: usize,
    big_endian: bool,
}

#[derive(Debug, Clone)]
enum DataLayout {
    Compact(Vec<u8>),
    Contiguous {
        addr: u64,
        size: u64,
    },
    ChunkedV1Btree {
        ndims: usize,
        btree_addr: u64,
        chunk_dims: Vec<u32>,
    },
    ChunkedV4Single {
        chunk_dims: Vec<u32>,
        chunk_addr: u64,
        filtered_size: Option<u64>,
        filter_mask: u32,
    },
}

struct DatasetF64 {
    dims: Vec<usize>,
    vals: Vec<f64>,
}

struct DatasetF32 {
    dims: Vec<usize>,
    vals: Vec<f32>,
}

impl<'a> Hdf5File<'a> {
    fn parse(bytes: &'a [u8]) -> Result<Self, KapError> {
        if bytes.len() < 48 || &bytes[0..8] != HDF5_MAGIC {
            return Err(io_err("not an HDF5 file (invalid magic signature)"));
        }
        let sb_ver = bytes[8];
        match sb_ver {
            0 | 1 => {
                let off_size = usize::from(bytes[13]);
                let len_size = usize::from(bytes[14]);
                if !matches!(off_size, 4 | 8) || !matches!(len_size, 4 | 8) {
                    return Err(io_err(format!(
                        "unsupported HDF5 offset/length sizes ({off_size}, {len_size})"
                    )));
                }
                let undef_addr = if off_size == 4 {
                    u64::from(u32::MAX)
                } else {
                    u64::MAX
                };
                let mut pos = if sb_ver == 0 { 24_usize } else { 28_usize };
                let base_addr = read_uint(bytes, &mut pos, off_size)?;
                let _free_addr = read_uint(bytes, &mut pos, off_size)?;
                let _eof_addr = read_uint(bytes, &mut pos, off_size)?;
                let _driver_addr = read_uint(bytes, &mut pos, off_size)?;
                // Root group symbol table entry
                let _link_off = read_uint(bytes, &mut pos, off_size)?;
                let root_oh_addr = read_uint(bytes, &mut pos, off_size)?;
                let cache_type = read_u32_le(bytes, &mut pos)?;
                let _reserved = read_u32_le(bytes, &mut pos)?;
                let root_symtab = if cache_type == 1 {
                    let bt = read_uint(bytes, &mut pos, off_size)?;
                    let hp = read_uint(bytes, &mut pos, off_size)?;
                    Some((bt, hp))
                } else {
                    None
                };
                Ok(Self {
                    bytes,
                    off_size,
                    len_size,
                    base_addr,
                    undef_addr,
                    root_oh_addr,
                    root_symtab,
                })
            }
            2 | 3 => {
                let off_size = usize::from(bytes[9]);
                let len_size = usize::from(bytes[10]);
                if !matches!(off_size, 4 | 8) || !matches!(len_size, 4 | 8) {
                    return Err(io_err(format!(
                        "unsupported HDF5 v{sb_ver} offset/length sizes ({off_size}, {len_size})"
                    )));
                }
                let undef_addr = if off_size == 4 {
                    u64::from(u32::MAX)
                } else {
                    u64::MAX
                };
                let mut pos = 12_usize;
                let base_addr = read_uint(bytes, &mut pos, off_size)?;
                let _ext_addr = read_uint(bytes, &mut pos, off_size)?;
                let _eof_addr = read_uint(bytes, &mut pos, off_size)?;
                let root_oh_addr = read_uint(bytes, &mut pos, off_size)?;
                Ok(Self {
                    bytes,
                    off_size,
                    len_size,
                    base_addr,
                    undef_addr,
                    root_oh_addr,
                    root_symtab: None,
                })
            }
            other => Err(io_err(format!(
                "unsupported HDF5 superblock version {other}"
            ))),
        }
    }

    fn resolve_addr(&self, addr: u64) -> Result<usize, KapError> {
        let abs = addr.checked_add(self.base_addr).ok_or_else(|| {
            io_err("HDF5 address overflow adding base address")
        })?;
        let off = usize::try_from(abs)
            .map_err(|_| io_err("HDF5 address exceeds usize"))?;
        if off >= self.bytes.len() {
            return Err(io_err(format!(
                "HDF5 address {off:#x} beyond EOF ({:#x})",
                self.bytes.len()
            )));
        }
        Ok(off)
    }

    fn read_offset(&self, pos: &mut usize) -> Result<u64, KapError> {
        read_uint(self.bytes, pos, self.off_size)
    }

    fn read_length(&self, pos: &mut usize) -> Result<u64, KapError> {
        read_uint(self.bytes, pos, self.len_size)
    }

    fn read_object_messages(
        &self,
        oh_addr: u64,
    ) -> Result<Vec<(u16, &'a [u8])>, KapError> {
        let off = self.resolve_addr(oh_addr)?;
        let mut out = Vec::new();
        if off + 4 <= self.bytes.len() && &self.bytes[off..off + 4] == b"OHDR" {
            self.read_oh_v2_chunk(off, true, &mut out, 0)?;
        } else if self.bytes[off] == 1 {
            if off + 16 > self.bytes.len() {
                return Err(io_err("truncated HDF5 v1 object header"));
            }
            let hdr_size = u32::from_le_bytes(
                self.bytes[off + 8..off + 12].try_into().unwrap_or([0; 4]),
            ) as usize;
            let start = off + 16;
            let end = start.checked_add(hdr_size).ok_or_else(|| {
                io_err("HDF5 v1 object header size overflow")
            })?;
            if end > self.bytes.len() {
                return Err(io_err("HDF5 v1 object header exceeds EOF"));
            }
            self.parse_oh_v1_messages(start, end, &mut out, 0)?;
        } else {
            return Err(io_err(format!(
                "unsupported HDF5 object header version {} at {oh_addr:#x}",
                self.bytes[off]
            )));
        }
        Ok(out)
    }

    fn parse_oh_v1_messages(
        &self,
        start: usize,
        end: usize,
        out: &mut Vec<(u16, &'a [u8])>,
        depth: usize,
    ) -> Result<(), KapError> {
        if depth > 16 {
            return Err(io_err("HDF5 object header continuation depth exceeded"));
        }
        let mut pos = start;
        while pos + 8 <= end {
            let msg_type =
                u16::from_le_bytes([self.bytes[pos], self.bytes[pos + 1]]);
            let msg_size =
                u16::from_le_bytes([self.bytes[pos + 2], self.bytes[pos + 3]])
                    as usize;
            let body_start = pos + 8;
            let body_end = body_start + msg_size;
            if body_end > end {
                return Err(io_err("truncated HDF5 v1 header message"));
            }
            let body = &self.bytes[body_start..body_end];
            if msg_type == 0x0010 {
                let mut cpos = body_start;
                let cont_addr = self.read_offset(&mut cpos)?;
                let cont_len = self.read_length(&mut cpos)? as usize;
                let cont_off = self.resolve_addr(cont_addr)?;
                let cont_end = cont_off
                    .checked_add(cont_len)
                    .ok_or_else(|| io_err("HDF5 continuation overflow"))?;
                if cont_end > self.bytes.len() {
                    return Err(io_err("HDF5 v1 continuation exceeds EOF"));
                }
                self.parse_oh_v1_messages(cont_off, cont_end, out, depth + 1)?;
            } else if msg_type != 0 {
                out.push((msg_type, body));
            }
            let padded = (msg_size + 7) & !7;
            pos = body_start + padded;
        }
        Ok(())
    }

    fn read_oh_v2_chunk(
        &self,
        off: usize,
        is_root: bool,
        out: &mut Vec<(u16, &'a [u8])>,
        depth: usize,
    ) -> Result<(), KapError> {
        if depth > 16 {
            return Err(io_err("HDF5 v2 continuation depth exceeded"));
        }
        let (msg_start, msg_end, has_creation_order) = if is_root {
            if off + 7 > self.bytes.len() {
                return Err(io_err("truncated HDF5 v2 OHDR"));
            }
            let ver = self.bytes[off + 4];
            if ver != 2 {
                return Err(io_err(format!("unexpected OHDR version {ver}")));
            }
            let flags = self.bytes[off + 5];
            let mut pos = off + 6;
            if (flags & 0x20) != 0 {
                pos += 16; // access, mod, change, birth timestamps
            }
            if (flags & 0x10) != 0 {
                pos += 4; // max compact / min dense phase change
            }
            let chunk0_sz_bytes = 1_usize << usize::from(flags & 0x03);
            let chunk0_sz =
                read_uint(self.bytes, &mut pos, chunk0_sz_bytes)? as usize;
            let end = pos + chunk0_sz;
            if end > self.bytes.len() {
                return Err(io_err("HDF5 v2 OHDR chunk exceeds EOF"));
            }
            (pos, end.saturating_sub(4), (flags & 0x04) != 0)
        } else {
            return Err(io_err("internal OHDR chunk dispatch error"));
        };
        self.parse_oh_v2_messages(
            msg_start,
            msg_end,
            has_creation_order,
            out,
            depth,
        )
    }

    fn parse_oh_v2_messages(
        &self,
        start: usize,
        end: usize,
        has_creation_order: bool,
        out: &mut Vec<(u16, &'a [u8])>,
        depth: usize,
    ) -> Result<(), KapError> {
        if depth > 16 {
            return Err(io_err("HDF5 v2 continuation depth exceeded"));
        }
        let mut pos = start;
        let hdr_len = if has_creation_order { 6 } else { 4 };
        while pos + hdr_len <= end {
            let msg_type = u16::from(self.bytes[pos]);
            let msg_size =
                u16::from_le_bytes([self.bytes[pos + 1], self.bytes[pos + 2]])
                    as usize;
            let body_start = pos + hdr_len;
            let body_end = body_start + msg_size;
            if body_end > end {
                break;
            }
            let body = &self.bytes[body_start..body_end];
            if msg_type == 0x0010 {
                let mut cpos = body_start;
                let cont_addr = self.read_offset(&mut cpos)?;
                let cont_len = self.read_length(&mut cpos)? as usize;
                let cont_off = self.resolve_addr(cont_addr)?;
                let cont_end = cont_off + cont_len;
                if cont_len < 8
                    || cont_end > self.bytes.len()
                    || &self.bytes[cont_off..cont_off + 4] != b"OCHK"
                {
                    return Err(io_err("invalid HDF5 v2 OCHK continuation"));
                }
                self.parse_oh_v2_messages(
                    cont_off + 4,
                    cont_end - 4,
                    has_creation_order,
                    out,
                    depth + 1,
                )?;
            } else if msg_type != 0 {
                out.push((msg_type, body));
            }
            pos = body_end;
        }
        Ok(())
    }

    fn list_group_children(
        &self,
        oh_addr: u64,
        cached_symtab: Option<(u64, u64)>,
    ) -> Result<HashMap<String, GroupChild>, KapError> {
        let mut children = HashMap::new();
        if let Some((bt, hp)) = cached_symtab {
            self.collect_v1_symtab_children(bt, hp, &mut children)?;
            return Ok(children);
        }
        let msgs = self.read_object_messages(oh_addr)?;
        for (mtype, body) in &msgs {
            if *mtype == 0x0011 {
                let mut pos = 0_usize;
                let bt = read_uint(body, &mut pos, self.off_size)?;
                let hp = read_uint(body, &mut pos, self.off_size)?;
                self.collect_v1_symtab_children(bt, hp, &mut children)?;
                return Ok(children);
            }
        }
        let mut found_link = false;
        for (mtype, body) in &msgs {
            if *mtype == 0x0006 {
                if let Some((name, target_oh)) = self.parse_v2_link(body)? {
                    children.insert(
                        name,
                        GroupChild {
                            oh_addr: target_oh,
                            symtab: None,
                        },
                    );
                    found_link = true;
                }
            }
        }
        if !found_link {
            return Err(io_err(format!(
                "HDF5 group at {oh_addr:#x} uses neither v1 symbol table nor compact v2 links"
            )));
        }
        Ok(children)
    }

    fn parse_v2_link(
        &self,
        body: &[u8],
    ) -> Result<Option<(String, u64)>, KapError> {
        if body.len() < 2 || body[0] != 1 {
            return Err(io_err("unsupported HDF5 v2 link message version"));
        }
        let flags = body[1];
        let mut pos = 2_usize;
        let link_type = if (flags & 0x08) != 0 {
            let t = body[pos];
            pos += 1;
            t
        } else {
            0 // hard link
        };
        if (flags & 0x04) != 0 {
            pos += 8; // creation order
        }
        if (flags & 0x10) != 0 {
            pos += 1; // charset
        }
        let name_len_sz = 1_usize << usize::from(flags & 0x03);
        let name_len = read_uint(body, &mut pos, name_len_sz)? as usize;
        if pos + name_len > body.len() {
            return Err(io_err("truncated HDF5 v2 link name"));
        }
        let name = String::from_utf8(body[pos..pos + name_len].to_vec())
            .map_err(|_| io_err("non-UTF8 HDF5 link name"))?;
        pos += name_len;
        if link_type != 0 {
            return Ok(None);
        }
        let target_oh = read_uint(body, &mut pos, self.off_size)?;
        Ok(Some((name, target_oh)))
    }

    fn collect_v1_symtab_children(
        &self,
        btree_addr: u64,
        heap_addr: u64,
        out: &mut HashMap<String, GroupChild>,
    ) -> Result<(), KapError> {
        let heap_off = self.resolve_addr(heap_addr)?;
        if heap_off + 8 > self.bytes.len()
            || &self.bytes[heap_off..heap_off + 4] != b"HEAP"
        {
            return Err(io_err(format!(
                "invalid HDF5 local heap signature at {heap_addr:#x}"
            )));
        }
        let mut hpos = heap_off + 8;
        let _data_seg_sz = self.read_length(&mut hpos)?;
        let _free_list_off = self.read_length(&mut hpos)?;
        let data_seg_addr = self.read_offset(&mut hpos)?;
        let data_seg_off = self.resolve_addr(data_seg_addr)?;
        self.walk_group_btree(btree_addr, data_seg_off, out, 0)
    }

    fn walk_group_btree(
        &self,
        bt_addr: u64,
        heap_seg_off: usize,
        out: &mut HashMap<String, GroupChild>,
        depth: usize,
    ) -> Result<(), KapError> {
        if depth > 32 {
            return Err(io_err("HDF5 group B-tree depth exceeded"));
        }
        let off = self.resolve_addr(bt_addr)?;
        if off + 8 + 2 * self.off_size > self.bytes.len()
            || &self.bytes[off..off + 4] != b"TREE"
        {
            return Err(io_err(format!(
                "invalid HDF5 group B-tree signature at {bt_addr:#x}"
            )));
        }
        let node_type = self.bytes[off + 4];
        let node_level = self.bytes[off + 5];
        let used =
            u16::from_le_bytes([self.bytes[off + 6], self.bytes[off + 7]])
                as usize;
        if node_type != 0 {
            return Err(io_err(format!(
                "expected HDF5 group B-tree type 0, got {node_type}"
            )));
        }
        let mut pos = off + 8 + 2 * self.off_size;
        for _ in 0..used {
            let _key = self.read_length(&mut pos)?;
            let child = self.read_offset(&mut pos)?;
            if node_level == 0 {
                self.read_snod(child, heap_seg_off, out)?;
            } else {
                self.walk_group_btree(child, heap_seg_off, out, depth + 1)?;
            }
        }
        Ok(())
    }

    fn read_snod(
        &self,
        snod_addr: u64,
        heap_seg_off: usize,
        out: &mut HashMap<String, GroupChild>,
    ) -> Result<(), KapError> {
        let off = self.resolve_addr(snod_addr)?;
        if off + 8 > self.bytes.len()
            || &self.bytes[off..off + 4] != b"SNOD"
        {
            return Err(io_err(format!(
                "invalid HDF5 SNOD signature at {snod_addr:#x}"
            )));
        }
        let nsyms =
            u16::from_le_bytes([self.bytes[off + 6], self.bytes[off + 7]])
                as usize;
        let mut pos = off + 8;
        for _ in 0..nsyms {
            let link_off = self.read_offset(&mut pos)? as usize;
            let obj_oh = self.read_offset(&mut pos)?;
            let cache_type = read_u32_le(self.bytes, &mut pos)?;
            let _reserved = read_u32_le(self.bytes, &mut pos)?;
            let scratch_start = pos;
            let symtab = if cache_type == 1 {
                let mut spos = scratch_start;
                let bt = self.read_offset(&mut spos)?;
                let hp = self.read_offset(&mut spos)?;
                Some((bt, hp))
            } else {
                None
            };
            pos = scratch_start + 16;

            let name_start = heap_seg_off
                .checked_add(link_off)
                .ok_or_else(|| io_err("HDF5 heap string offset overflow"))?;
            let tail = self.bytes.get(name_start..).ok_or_else(|| {
                io_err("HDF5 heap string offset out of bounds")
            })?;
            let nul = tail
                .iter()
                .position(|&b| b == 0)
                .ok_or_else(|| io_err("unterminated HDF5 local heap string"))?;
            let name = String::from_utf8(tail[..nul].to_vec())
                .map_err(|_| io_err("non-UTF8 HDF5 symbol name"))?;
            out.insert(
                name,
                GroupChild {
                    oh_addr: obj_oh,
                    symtab,
                },
            );
        }
        Ok(())
    }

    fn read_dataset_raw(
        &self,
        oh_addr: u64,
    ) -> Result<(Vec<usize>, FloatType, Vec<u8>), KapError> {
        let msgs = self.read_object_messages(oh_addr)?;
        let mut dims: Option<Vec<usize>> = None;
        let mut dtype: Option<FloatType> = None;
        let mut layout: Option<DataLayout> = None;
        let mut filters: Vec<FilterSpec> = Vec::new();

        for (mtype, body) in msgs {
            match mtype {
                0x0001 => dims = Some(self.parse_dataspace(body)?),
                0x0003 => dtype = Some(parse_float_datatype(body)?),
                0x0008 => layout = Some(self.parse_data_layout(body)?),
                0x000b => filters = parse_filter_pipeline(body)?,
                _ => {}
            }
        }
        let dims = dims.ok_or_else(|| {
            io_err(format!("missing dataspace message at {oh_addr:#x}"))
        })?;
        let dtype = dtype.ok_or_else(|| {
            io_err(format!("missing datatype message at {oh_addr:#x}"))
        })?;
        let layout = layout.ok_or_else(|| {
            io_err(format!("missing data layout message at {oh_addr:#x}"))
        })?;

        let total_elems: usize = dims.iter().copied().try_fold(1_usize, |a, b| {
            a.checked_mul(b)
                .ok_or_else(|| io_err("HDF5 dataset element count overflow"))
        })?;
        let total_bytes = total_elems
            .checked_mul(dtype.elem_size)
            .ok_or_else(|| io_err("HDF5 dataset byte size overflow"))?;

        let raw = match layout {
            DataLayout::Compact(buf) => {
                if buf.len() < total_bytes {
                    return Err(io_err("HDF5 compact dataset truncated"));
                }
                buf[..total_bytes].to_vec()
            }
            DataLayout::Contiguous { addr, size } => {
                if addr == self.undef_addr {
                    vec![0_u8; total_bytes]
                } else {
                    let off = self.resolve_addr(addr)?;
                    let sz = usize::try_from(size)
                        .map_err(|_| io_err("HDF5 contiguous size overflow"))?;
                    if sz < total_bytes || off + total_bytes > self.bytes.len()
                    {
                        return Err(io_err(
                            "HDF5 contiguous dataset out of bounds",
                        ));
                    }
                    self.bytes[off..off + total_bytes].to_vec()
                }
            }
            DataLayout::ChunkedV1Btree {
                ndims,
                btree_addr,
                chunk_dims,
            } => {
                let mut out = vec![0_u8; total_bytes];
                if btree_addr != self.undef_addr {
                    let rank = dims.len();
                    if ndims != rank + 1 {
                        return Err(io_err(format!(
                            "HDF5 chunk ndims {ndims} != rank+1 ({})",
                            rank + 1
                        )));
                    }
                    let cdims: Vec<usize> = chunk_dims[..rank]
                        .iter()
                        .map(|&d| d as usize)
                        .collect();
                    self.walk_chunk_btree(
                        btree_addr,
                        ndims,
                        &dims,
                        &cdims,
                        dtype.elem_size,
                        &filters,
                        &mut out,
                        0,
                    )?;
                }
                out
            }
            DataLayout::ChunkedV4Single {
                chunk_dims,
                chunk_addr,
                filtered_size,
                filter_mask,
            } => {
                let mut out = vec![0_u8; total_bytes];
                if chunk_addr != self.undef_addr {
                    let cdims: Vec<usize> =
                        chunk_dims.iter().map(|&d| d as usize).collect();
                    let expected_chunk_bytes = cdims
                        .iter()
                        .copied()
                        .product::<usize>()
                        * dtype.elem_size;
                    let on_disk_sz = filtered_size
                        .map_or(expected_chunk_bytes, |v| v as usize);
                    let off = self.resolve_addr(chunk_addr)?;
                    if off + on_disk_sz > self.bytes.len() {
                        return Err(io_err("HDF5 v4 single chunk out of bounds"));
                    }
                    let decoded = apply_filters_reverse(
                        &self.bytes[off..off + on_disk_sz],
                        &filters,
                        filter_mask,
                        expected_chunk_bytes,
                    )?;
                    let origin = vec![0_usize; dims.len()];
                    scatter_chunk(
                        &decoded,
                        &cdims,
                        &origin,
                        &dims,
                        dtype.elem_size,
                        &mut out,
                    )?;
                }
                out
            }
        };

        Ok((dims, dtype, raw))
    }

    fn parse_dataspace(&self, body: &[u8]) -> Result<Vec<usize>, KapError> {
        if body.len() < 4 {
            return Err(io_err("truncated HDF5 dataspace message"));
        }
        let ver = body[0];
        let rank = usize::from(body[1]);
        let mut pos = match ver {
            1 => 8_usize,
            2 => 4_usize,
            other => {
                return Err(io_err(format!(
                    "unsupported HDF5 dataspace version {other}"
                )));
            }
        };
        let mut dims = Vec::with_capacity(rank);
        for _ in 0..rank {
            let d = read_uint(body, &mut pos, self.len_size)?;
            dims.push(
                usize::try_from(d)
                    .map_err(|_| io_err("HDF5 dimension exceeds usize"))?,
            );
        }
        Ok(dims)
    }

    fn parse_data_layout(&self, body: &[u8]) -> Result<DataLayout, KapError> {
        if body.len() < 2 {
            return Err(io_err("truncated HDF5 data layout message"));
        }
        let ver = body[0];
        let cls = body[1];
        match (ver, cls) {
            (3 | 4, 0) => {
                if body.len() < 4 {
                    return Err(io_err("truncated HDF5 compact layout"));
                }
                let sz = u16::from_le_bytes([body[2], body[3]]) as usize;
                if 4 + sz > body.len() {
                    return Err(io_err("truncated HDF5 compact payload"));
                }
                Ok(DataLayout::Compact(body[4..4 + sz].to_vec()))
            }
            (3 | 4, 1) => {
                let mut pos = 2_usize;
                let addr = read_uint(body, &mut pos, self.off_size)?;
                let size = read_uint(body, &mut pos, self.len_size)?;
                Ok(DataLayout::Contiguous { addr, size })
            }
            (3, 2) => {
                if body.len() < 3 {
                    return Err(io_err("truncated HDF5 v3 chunked layout"));
                }
                let ndims = usize::from(body[2]);
                let mut pos = 3_usize;
                let btree_addr = read_uint(body, &mut pos, self.off_size)?;
                let mut chunk_dims = Vec::with_capacity(ndims);
                for _ in 0..ndims {
                    chunk_dims.push(read_u32_le(body, &mut pos)?);
                }
                Ok(DataLayout::ChunkedV1Btree {
                    ndims,
                    btree_addr,
                    chunk_dims,
                })
            }
            (4, 2) => {
                if body.len() < 5 {
                    return Err(io_err("truncated HDF5 v4 chunked layout"));
                }
                let flags = body[2];
                let rank = usize::from(body[3]);
                let dim_sz_enc = usize::from(body[4]);
                let mut pos = 5_usize;
                let mut chunk_dims = Vec::with_capacity(rank);
                for _ in 0..rank {
                    chunk_dims
                        .push(read_uint(body, &mut pos, dim_sz_enc)? as u32);
                }
                if pos >= body.len() {
                    return Err(io_err("missing HDF5 v4 chunk indexing type"));
                }
                let idx_type = body[pos];
                pos += 1;
                if idx_type == 1 {
                    let (filtered_size, filter_mask) = if (flags & 0x02) != 0 {
                        let fsz = read_uint(body, &mut pos, self.len_size)?;
                        let fmask = read_u32_le(body, &mut pos)?;
                        (Some(fsz), fmask)
                    } else {
                        (None, 0)
                    };
                    let chunk_addr = read_uint(body, &mut pos, self.off_size)?;
                    Ok(DataLayout::ChunkedV4Single {
                        chunk_dims,
                        chunk_addr,
                        filtered_size,
                        filter_mask,
                    })
                } else {
                    Err(io_err(format!(
                        "unsupported HDF5 v4 chunk indexing type {idx_type}"
                    )))
                }
            }
            _ => Err(io_err(format!(
                "unsupported HDF5 data layout version={ver} class={cls}"
            ))),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn walk_chunk_btree(
        &self,
        bt_addr: u64,
        ndims: usize,
        ds_dims: &[usize],
        chunk_dims: &[usize],
        elem_size: usize,
        filters: &[FilterSpec],
        out: &mut [u8],
        depth: usize,
    ) -> Result<(), KapError> {
        if depth > 32 {
            return Err(io_err("HDF5 chunk B-tree depth exceeded"));
        }
        let off = self.resolve_addr(bt_addr)?;
        if off + 8 + 2 * self.off_size > self.bytes.len()
            || &self.bytes[off..off + 4] != b"TREE"
        {
            return Err(io_err(format!(
                "invalid HDF5 chunk B-tree signature at {bt_addr:#x}"
            )));
        }
        let node_type = self.bytes[off + 4];
        let node_level = self.bytes[off + 5];
        let used =
            u16::from_le_bytes([self.bytes[off + 6], self.bytes[off + 7]])
                as usize;
        if node_type != 1 {
            return Err(io_err(format!(
                "expected HDF5 chunk B-tree type 1, got {node_type}"
            )));
        }
        let rank = ds_dims.len();
        let expected_chunk_bytes = chunk_dims
            .iter()
            .copied()
            .try_fold(elem_size, |a, b| {
                a.checked_mul(b).ok_or_else(|| io_err("chunk size overflow"))
            })?;

        let mut pos = off + 8 + 2 * self.off_size;
        let mut origin = vec![0_usize; rank];
        for _ in 0..used {
            let chunk_size = read_u32_le(self.bytes, &mut pos)? as usize;
            let filter_mask = read_u32_le(self.bytes, &mut pos)?;
            for slot in &mut origin {
                let coord = read_u64_le(self.bytes, &mut pos)?;
                *slot = usize::try_from(coord)
                    .map_err(|_| io_err("chunk coordinate exceeds usize"))?;
            }
            for _ in rank..ndims {
                let _ = read_u64_le(self.bytes, &mut pos)?;
            }
            let child = self.read_offset(&mut pos)?;
            if node_level == 0 {
                let coff = self.resolve_addr(child)?;
                if coff + chunk_size > self.bytes.len() {
                    return Err(io_err("HDF5 chunk bytes exceed EOF"));
                }
                let decoded = apply_filters_reverse(
                    &self.bytes[coff..coff + chunk_size],
                    filters,
                    filter_mask,
                    expected_chunk_bytes,
                )?;
                scatter_chunk(
                    &decoded, chunk_dims, &origin, ds_dims, elem_size, out,
                )?;
            } else {
                self.walk_chunk_btree(
                    child,
                    ndims,
                    ds_dims,
                    chunk_dims,
                    elem_size,
                    filters,
                    out,
                    depth + 1,
                )?;
            }
        }
        Ok(())
    }

    fn read_dataset_f64(&self, oh_addr: u64) -> Result<DatasetF64, KapError> {
        let (dims, dtype, raw) = self.read_dataset_raw(oh_addr)?;
        let n = raw.len() / dtype.elem_size;
        let mut vals = Vec::with_capacity(n);
        match (dtype.elem_size, dtype.big_endian) {
            (4, false) => {
                for &c in raw.as_chunks::<4>().0 {
                    vals.push(f64::from(f32::from_le_bytes(c)));
                }
            }
            (4, true) => {
                for &c in raw.as_chunks::<4>().0 {
                    vals.push(f64::from(f32::from_be_bytes(c)));
                }
            }
            (8, false) => {
                for &c in raw.as_chunks::<8>().0 {
                    vals.push(f64::from_le_bytes(c));
                }
            }
            (8, true) => {
                for &c in raw.as_chunks::<8>().0 {
                    vals.push(f64::from_be_bytes(c));
                }
            }
            _ => {
                return Err(io_err(format!(
                    "unsupported float element size {}",
                    dtype.elem_size
                )));
            }
        }
        Ok(DatasetF64 { dims, vals })
    }

    fn read_dataset_f32(&self, oh_addr: u64) -> Result<DatasetF32, KapError> {
        let (dims, dtype, raw) = self.read_dataset_raw(oh_addr)?;
        let n = raw.len() / dtype.elem_size;
        let mut vals = Vec::with_capacity(n);
        match (dtype.elem_size, dtype.big_endian) {
            (4, false) => {
                for &c in raw.as_chunks::<4>().0 {
                    vals.push(f32::from_le_bytes(c));
                }
            }
            (4, true) => {
                for &c in raw.as_chunks::<4>().0 {
                    vals.push(f32::from_be_bytes(c));
                }
            }
            (8, false) => {
                for &c in raw.as_chunks::<8>().0 {
                    vals.push(f64::from_le_bytes(c) as f32);
                }
            }
            (8, true) => {
                for &c in raw.as_chunks::<8>().0 {
                    vals.push(f64::from_be_bytes(c) as f32);
                }
            }
            _ => {
                return Err(io_err(format!(
                    "unsupported float element size {}",
                    dtype.elem_size
                )));
            }
        }
        Ok(DatasetF32 { dims, vals })
    }
}

fn parse_float_datatype(body: &[u8]) -> Result<FloatType, KapError> {
    if body.len() < 8 {
        return Err(io_err("truncated HDF5 datatype message"));
    }
    let class = body[0] & 0x0f;
    if class != 1 {
        return Err(io_err(format!(
            "expected IEEE floating-point HDF5 datatype (class 1), got class {class}"
        )));
    }
    let big_endian = (body[1] & 0x01) != 0;
    let elem_size =
        u32::from_le_bytes([body[4], body[5], body[6], body[7]]) as usize;
    if !matches!(elem_size, 4 | 8) {
        return Err(io_err(format!(
            "unsupported HDF5 float byte size {elem_size}"
        )));
    }
    Ok(FloatType {
        elem_size,
        big_endian,
    })
}

fn parse_filter_pipeline(body: &[u8]) -> Result<Vec<FilterSpec>, KapError> {
    if body.len() < 2 {
        return Err(io_err("truncated HDF5 filter pipeline message"));
    }
    let ver = body[0];
    let n_filters = usize::from(body[1]);
    let mut pos = match ver {
        1 => 8_usize,
        2 => 2_usize,
        other => {
            return Err(io_err(format!(
                "unsupported HDF5 filter pipeline version {other}"
            )));
        }
    };
    let mut filters = Vec::with_capacity(n_filters);
    for _ in 0..n_filters {
        let id = read_u16_le(body, &mut pos)?;
        let name_len = if ver == 1 || id >= 256 {
            usize::from(read_u16_le(body, &mut pos)?)
        } else {
            0
        };
        let flags = read_u16_le(body, &mut pos)?;
        let n_cd = usize::from(read_u16_le(body, &mut pos)?);
        let padded_name = if ver == 1 {
            (name_len + 7) & !7
        } else {
            name_len
        };
        if pos + padded_name > body.len() {
            return Err(io_err("truncated HDF5 filter name"));
        }
        pos += padded_name;
        let mut client_data = Vec::with_capacity(n_cd);
        for _ in 0..n_cd {
            client_data.push(read_u32_le(body, &mut pos)?);
        }
        if ver == 1 && (n_cd & 1) == 1 {
            pos = pos.saturating_add(4);
        }
        filters.push(FilterSpec {
            id,
            flags,
            client_data,
        });
    }
    Ok(filters)
}

fn apply_filters_reverse(
    chunk: &[u8],
    filters: &[FilterSpec],
    filter_mask: u32,
    expected_bytes: usize,
) -> Result<Vec<u8>, KapError> {
    let mut buf = chunk.to_vec();
    for (idx, filt) in filters.iter().enumerate().rev() {
        if idx < 32 && (filter_mask & (1_u32 << idx)) != 0 {
            continue;
        }
        match filt.id {
            1 => {
                buf = zlib_decompress(&buf, expected_bytes)?;
            }
            2 => {
                let elem_sz = filt
                    .client_data
                    .first()
                    .copied()
                    .unwrap_or(1) as usize;
                if elem_sz > 1 && buf.len().is_multiple_of(elem_sz) {
                    let n = buf.len() / elem_sz;
                    let mut unshuffled = vec![0_u8; buf.len()];
                    for b in 0..elem_sz {
                        for i in 0..n {
                            unshuffled[i * elem_sz + b] = buf[b * n + i];
                        }
                    }
                    buf = unshuffled;
                }
            }
            3 => {
                if buf.len() < 4 {
                    return Err(io_err("truncated fletcher32 chunk"));
                }
                buf.truncate(buf.len() - 4);
            }
            other => {
                if (filt.flags & 0x01) == 0 {
                    return Err(io_err(format!(
                        "unsupported mandatory HDF5 filter id {other}"
                    )));
                }
            }
        }
    }
    Ok(buf)
}

fn scatter_chunk(
    chunk: &[u8],
    chunk_dims: &[usize],
    origin: &[usize],
    ds_dims: &[usize],
    elem_size: usize,
    out: &mut [u8],
) -> Result<(), KapError> {
    let rank = ds_dims.len();
    if rank == 0 {
        if chunk.len() < elem_size || out.len() < elem_size {
            return Err(io_err("scalar HDF5 chunk size mismatch"));
        }
        out[..elem_size].copy_from_slice(&chunk[..elem_size]);
        return Ok(());
    }
    let expected_chunk_bytes =
        chunk_dims.iter().copied().product::<usize>() * elem_size;
    if chunk.len() < expected_chunk_bytes {
        return Err(io_err(format!(
            "decoded HDF5 chunk size {} < expected {}",
            chunk.len(),
            expected_chunk_bytes
        )));
    }
    let mut ds_strides = vec![1_usize; rank];
    let mut ch_strides = vec![1_usize; rank];
    for d in (0..rank - 1).rev() {
        ds_strides[d] = ds_strides[d + 1] * ds_dims[d + 1];
        ch_strides[d] = ch_strides[d + 1] * chunk_dims[d + 1];
    }
    let copy_lens: Vec<usize> = (0..rank)
        .map(|d| ds_dims[d].saturating_sub(origin[d]).min(chunk_dims[d]))
        .collect();
    if copy_lens.contains(&0) {
        return Ok(());
    }
    let inner_bytes = copy_lens[rank - 1] * elem_size;
    let outer_dims = &copy_lens[..rank - 1];
    let mut idx = vec![0_usize; rank - 1];
    loop {
        let mut src_elem = 0_usize;
        let mut dst_elem = origin[rank - 1];
        for d in 0..(rank - 1) {
            src_elem += idx[d] * ch_strides[d];
            dst_elem += (origin[d] + idx[d]) * ds_strides[d];
        }
        let sb = src_elem * elem_size;
        let db = dst_elem * elem_size;
        out[db..db + inner_bytes]
            .copy_from_slice(&chunk[sb..sb + inner_bytes]);

        let mut carry = true;
        for d in (0..(rank - 1)).rev() {
            idx[d] += 1;
            if idx[d] < outer_dims[d] {
                carry = false;
                break;
            }
            idx[d] = 0;
        }
        if carry {
            break;
        }
    }
    Ok(())
}

fn read_u16_le(bytes: &[u8], pos: &mut usize) -> Result<u16, KapError> {
    if *pos + 2 > bytes.len() {
        return Err(io_err("unexpected EOF reading HDF5 u16"));
    }
    let v = u16::from_le_bytes([bytes[*pos], bytes[*pos + 1]]);
    *pos += 2;
    Ok(v)
}

fn read_u32_le(bytes: &[u8], pos: &mut usize) -> Result<u32, KapError> {
    if *pos + 4 > bytes.len() {
        return Err(io_err("unexpected EOF reading HDF5 u32"));
    }
    let v = u32::from_le_bytes([
        bytes[*pos],
        bytes[*pos + 1],
        bytes[*pos + 2],
        bytes[*pos + 3],
    ]);
    *pos += 4;
    Ok(v)
}

fn read_u64_le(bytes: &[u8], pos: &mut usize) -> Result<u64, KapError> {
    if *pos + 8 > bytes.len() {
        return Err(io_err("unexpected EOF reading HDF5 u64"));
    }
    let v = u64::from_le_bytes(
        bytes[*pos..*pos + 8].try_into().unwrap_or([0; 8]),
    );
    *pos += 8;
    Ok(v)
}

fn read_uint(bytes: &[u8], pos: &mut usize, width: usize) -> Result<u64, KapError> {
    if *pos + width > bytes.len() {
        return Err(io_err("unexpected EOF reading HDF5 integer"));
    }
    let v = match width {
        1 => u64::from(bytes[*pos]),
        2 => u64::from(u16::from_le_bytes([bytes[*pos], bytes[*pos + 1]])),
        4 => u64::from(u32::from_le_bytes([
            bytes[*pos],
            bytes[*pos + 1],
            bytes[*pos + 2],
            bytes[*pos + 3],
        ])),
        8 => u64::from_le_bytes(
            bytes[*pos..*pos + 8].try_into().unwrap_or([0; 8]),
        ),
        other => {
            return Err(io_err(format!("unsupported HDF5 integer width {other}")));
        }
    };
    *pos += width;
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::{
        aesopus_source_exists, load_aesopus_hdf5_bytes, resolve_aesopus_data,
    };

    #[test]
    fn test_aesopus_h5_rejects_corrupt_or_missing_sources() {
        assert!(aesopus_source_exists("AESOPUS_AGSS09.h5"));
        assert!(!aesopus_source_exists(""));
        assert!(!aesopus_source_exists("AESOPUS_DOES_NOT_EXIST_XYZ.h5"));
        assert!(!aesopus_source_exists("/nonexistent/AESOPUS_AGSS09.h5"));
        assert!(resolve_aesopus_data("AESOPUS_DOES_NOT_EXIST_XYZ.h5").is_err());
        assert!(load_aesopus_hdf5_bytes(b"not an hdf5 payload").is_err());
    }
}
