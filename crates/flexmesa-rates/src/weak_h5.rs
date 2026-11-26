// SPDX-License-Identifier: GPL-3.0-or-later
//! Pure-Rust loader for custom MESA `rate_tables_dir` manifests
//! (`rate_list.txt` and `weak_rate_list.txt`) and HDF5 weak-reaction tables
//! (`T9s`, `lYeRhos`, `ldecay`, `lcapture`, `lneutrino`).
//!
//! Ported from upstream MESA `rates/private/rates_initialize.f90:538-576`,
//! `rates/private/load_weak.f90:491-634`, and `rates/private/weak_support.f90:26-82`.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use flexmesa_chem::NuclideCatalog;

use crate::loader::WeakTable;
use crate::raw_rates::RateTable1D;
use crate::types::RatesError;

const HDF5_MAGIC: &[u8; 8] = b"\x89HDF\r\n\x1a\n";

/// Custom nuclear reaction tables loaded from a MESA `rate_tables_dir`
/// directory containing optional `rate_list.txt` and `weak_rate_list.txt`
/// manifests (`rates/private/rates_initialize.f90:538-576`,
/// `rates/private/load_weak.f90:491-579`).
#[derive(Debug, Clone, Default)]
pub struct LoadedRateTables {
    /// Custom 1D rate tables (`rate_list.txt`) keyed by MESA reaction handle
    /// (e.g., `"r_c14_ag_o18"`).
    pub raw_rates: HashMap<String, Arc<RateTable1D>>,
    /// Custom 2D HDF5 weak reaction tables (`weak_rate_list.txt`) keyed by
    /// `(lhs_name, rhs_name)` isotope pair (e.g., `("n14", "c14")`).
    pub weak_tables: HashMap<(String, String), WeakTable>,
}

/// Parses a MESA weak reaction handle of the form `r_<lhs>_wk_<rhs>` or
/// `r_<lhs>_wk-minus_<rhs>` (`rates/private/weak_support.f90:26-82`).
///
/// Returns `(lhs_name, rhs_name, is_beta_minus)`, validating that both
/// nuclides exist in [`NuclideCatalog::builtin_ref`].
///
/// # Errors
/// Returns [`RatesError::InvalidOption`] if `rate_name` does not match the
/// `r_<lhs>_wk[_|-minus_]<rhs>` syntax, or [`RatesError::UnknownWeakPair`] if
/// either isotope name is missing from the nuclide catalog.
pub fn parse_weak_rate_name(
    rate_name: &str,
) -> Result<(String, String, bool), RatesError> {
    let trimmed = rate_name.trim();
    let Some(rest) = trimmed.strip_prefix("r_") else {
        return Err(RatesError::InvalidOption(format!(
            "weak rate name '{trimmed}' must start with 'r_'"
        )));
    };
    let (lhs, rhs, is_beta_minus) =
        if let Some((l, r)) = rest.split_once("_wk-minus_") {
            (l, r, true)
        } else if let Some((l, r)) = rest.split_once("_wk_") {
            (l, r, false)
        } else {
            return Err(RatesError::InvalidOption(format!(
                "weak rate name '{trimmed}' must contain '_wk_' or '_wk-minus_'"
            )));
        };

    if lhs.is_empty() || rhs.is_empty() {
        return Err(RatesError::InvalidOption(format!(
            "weak rate name '{trimmed}' has empty reactant or product isotope"
        )));
    }
    let nuclides = NuclideCatalog::builtin_ref();
    if nuclides.get_nuclide_index(lhs).is_none()
        || nuclides.get_nuclide_index(rhs).is_none()
    {
        return Err(RatesError::UnknownWeakPair(
            lhs.to_string(),
            rhs.to_string(),
        ));
    }
    Ok((lhs.to_string(), rhs.to_string(), is_beta_minus))
}

/// Loads custom 1D strong/raw rate tables (`rate_list.txt`) and 2D HDF5 weak
/// tables (`weak_rate_list.txt`) from `dir`
/// (`rates/private/rates_initialize.f90:538-576`,
/// `rates/private/load_weak.f90:491-579`).
///
/// Missing `rate_list.txt` or `weak_rate_list.txt` files are treated as empty
/// lists (matching MESA's per-manifest lookup), whereas an invalid `dir` path or
/// malformed/missing referenced table returns an error.
///
/// # Errors
/// Returns [`RatesError::InvalidOption`] if `dir` is not an existing directory,
/// or [`RatesError::InvalidTableData`] / [`RatesError::UnknownWeakPair`] if any
/// manifest entry or referenced file is invalid.
pub fn load_rate_tables_dir(
    dir: &Path,
) -> Result<LoadedRateTables, RatesError> {
    if !dir.is_dir() {
        return Err(RatesError::InvalidOption(format!(
            "rate_tables_dir '{}' does not exist or is not a directory",
            dir.display()
        )));
    }

    let mut out = LoadedRateTables::default();

    let raw_manifest = dir.join("rate_list.txt");
    if raw_manifest.is_file() {
        let src = fs::read_to_string(&raw_manifest).map_err(|e| {
            RatesError::InvalidTableData(format!(
                "cannot read '{}': {e}",
                raw_manifest.display()
            ))
        })?;
        for (rate_name, fname) in parse_list_manifest(&src, "rate_list.txt")? {
            let candidate = dir.join(&fname);
            let tbl = if candidate.is_file() {
                RateTable1D::from_file(&candidate)?
            } else {
                RateTable1D::resolve_and_load(&fname, Some(dir))?
            };
            out.raw_rates.insert(rate_name, Arc::new(tbl));
        }
    }

    let weak_manifest = dir.join("weak_rate_list.txt");
    if weak_manifest.is_file() {
        let src = fs::read_to_string(&weak_manifest).map_err(|e| {
            RatesError::InvalidTableData(format!(
                "cannot read '{}': {e}",
                weak_manifest.display()
            ))
        })?;
        for (idx, (rate_name, fname)) in
            parse_list_manifest(&src, "weak_rate_list.txt")?
                .into_iter()
                .enumerate()
        {
            let (lhs, rhs, _) = parse_weak_rate_name(&rate_name)?;
            let mut tbl = load_weak_table_hdf5(&dir.join(&fname), &lhs, &rhs)?;
            tbl.id = idx + 1;
            out.weak_tables.insert((lhs, rhs), tbl);
        }
    }

    Ok(out)
}

/// Reads and parses a MESA custom weak HDF5 table (`T9s`, `lYeRhos`, `ldecay`,
/// `lcapture`, `lneutrino`) from `path` (`rates/private/load_weak.f90:585-634`).
///
/// # Errors
/// Returns [`RatesError::UnknownWeakPair`] if `(lhs_name, rhs_name)` is not in
/// [`NuclideCatalog::builtin_ref`], or [`RatesError::InvalidTableData`] if the
/// file cannot be read, fails HDF5 parsing, or has mismatched/non-ascending grids.
pub fn load_weak_table_hdf5(
    path: &Path,
    lhs_name: &str,
    rhs_name: &str,
) -> Result<WeakTable, RatesError> {
    let bytes = fs::read(path).map_err(|e| {
        RatesError::InvalidTableData(format!(
            "cannot read weak HDF5 file '{}': {e}",
            path.display()
        ))
    })?;
    load_weak_table_hdf5_bytes(&bytes, lhs_name, rhs_name).map_err(|e| {
        RatesError::InvalidTableData(format!(
            "invalid weak HDF5 file '{}': {e}",
            path.display()
        ))
    })
}

/// Parses an in-memory MESA custom weak HDF5 payload into a [`WeakTable`] for
/// `(lhs_name, rhs_name)` (`rates/private/load_weak.f90:585-634`).
///
/// HDF5 stores 2D arrays in C-order `(n_lye, n_t9)` (`lYeRhos` outer, `T9s`
/// inner, matching Fortran column-major `data(1:num_T9, 1:num_lYeRho)`), which
/// this loader transposes into FlexMESA's row-major `(nt9, nlye)` layout.
///
/// # Errors
/// Returns [`RatesError::UnknownWeakPair`] if `(lhs_name, rhs_name)` is not in
/// [`NuclideCatalog::builtin_ref`], or [`RatesError::InvalidTableData`] if HDF5
/// parsing or shape/monotonicity validation fails.
pub fn load_weak_table_hdf5_bytes(
    bytes: &[u8],
    lhs_name: &str,
    rhs_name: &str,
) -> Result<WeakTable, RatesError> {
    let nuclides = NuclideCatalog::builtin_ref();
    let lhs_idx = nuclides.get_nuclide_index(lhs_name).ok_or_else(|| {
        RatesError::UnknownWeakPair(lhs_name.to_string(), rhs_name.to_string())
    })?;
    let rhs_idx = nuclides.get_nuclide_index(rhs_name).ok_or_else(|| {
        RatesError::UnknownWeakPair(lhs_name.to_string(), rhs_name.to_string())
    })?;
    let lhs_nuc = nuclides.get_by_id(lhs_idx).ok_or_else(|| {
        RatesError::UnknownWeakPair(lhs_name.to_string(), rhs_name.to_string())
    })?;
    let rhs_nuc = nuclides.get_by_id(rhs_idx).ok_or_else(|| {
        RatesError::UnknownWeakPair(lhs_name.to_string(), rhs_name.to_string())
    })?;
    let lhs_id = u16::try_from(lhs_idx).map_err(|_| {
        RatesError::InvalidTableData(format!(
            "nuclide id overflow for {lhs_name}"
        ))
    })?;
    let rhs_id = u16::try_from(rhs_idx).map_err(|_| {
        RatesError::InvalidTableData(format!(
            "nuclide id overflow for {rhs_name}"
        ))
    })?;
    let q_tot = lhs_nuc.mass_excess - rhs_nuc.mass_excess;

    let h5 = Hdf5File::parse(bytes)?;
    let root = h5.list_root_children()?;

    let t9_ds = read_required_dataset(&h5, &root, "T9s")?;
    let lye_ds = read_required_dataset(&h5, &root, "lYeRhos")?;
    let ldec_ds = read_required_dataset(&h5, &root, "ldecay")?;
    let lcap_ds = read_required_dataset(&h5, &root, "lcapture")?;
    let lneu_ds = read_required_dataset(&h5, &root, "lneutrino")?;

    validate_axis("T9s", &t9_ds)?;
    validate_axis("lYeRhos", &lye_ds)?;
    let n_t9 = t9_ds.vals.len();
    let n_lye = lye_ds.vals.len();

    let ldecay = transpose_c_to_t9_lye("ldecay", &ldec_ds, n_lye, n_t9)?;
    let lcapture = transpose_c_to_t9_lye("lcapture", &lcap_ds, n_lye, n_t9)?;
    let lneutrino = transpose_c_to_t9_lye("lneutrino", &lneu_ds, n_lye, n_t9)?;

    Ok(WeakTable {
        id: 0,
        lhs_id,
        rhs_id,
        lhs_name: lhs_nuc.name.clone(),
        rhs_name: rhs_nuc.name.clone(),
        q_tot,
        t9s: t9_ds.vals,
        lyerhos: lye_ds.vals,
        ldecay,
        lcapture,
        lneutrino,
    })
}

fn read_required_dataset(
    h5: &Hdf5File<'_>,
    root: &HashMap<String, GroupChild>,
    name: &str,
) -> Result<DatasetF64, RatesError> {
    let child = root.get(name).ok_or_else(|| {
        RatesError::InvalidTableData(format!(
            "missing required HDF5 weak dataset '{name}'"
        ))
    })?;
    h5.read_dataset_f64(child.oh_addr)
}

fn validate_axis(name: &str, ds: &DatasetF64) -> Result<(), RatesError> {
    if ds.dims.len() != 1 || ds.vals.len() < 2 {
        return Err(RatesError::InvalidTableData(format!(
            "HDF5 dataset '{name}' must be 1D with at least 2 points (got {:?})",
            ds.dims
        )));
    }
    if !ds.vals.iter().all(|v| v.is_finite()) {
        return Err(RatesError::InvalidTableData(format!(
            "HDF5 dataset '{name}' contains non-finite coordinate values"
        )));
    }
    if !ds.vals.windows(2).all(|w| w[0] < w[1]) {
        return Err(RatesError::InvalidTableData(format!(
            "HDF5 dataset '{name}' must be strictly increasing"
        )));
    }
    Ok(())
}

fn transpose_c_to_t9_lye(
    name: &str,
    ds: &DatasetF64,
    n_lye: usize,
    n_t9: usize,
) -> Result<Vec<f64>, RatesError> {
    if ds.dims != [n_lye, n_t9] {
        return Err(RatesError::InvalidTableData(format!(
            "HDF5 dataset '{name}' shape {:?} does not match expected [{n_lye}, {n_t9}]",
            ds.dims
        )));
    }
    if !ds.vals.iter().all(|v| v.is_finite()) {
        return Err(RatesError::InvalidTableData(format!(
            "HDF5 dataset '{name}' contains non-finite values"
        )));
    }
    let mut out = vec![0.0_f64; n_t9 * n_lye];
    for ily in 0..n_lye {
        for it in 0..n_t9 {
            out[it * n_lye + ily] = ds.vals[ily * n_t9 + it];
        }
    }
    Ok(out)
}

fn strip_manifest_comment(line: &str) -> &str {
    let mut quote: Option<char> = None;
    for (idx, ch) in line.char_indices() {
        match quote {
            Some(q) if ch == q => quote = None,
            None if ch == '\'' || ch == '"' => quote = Some(ch),
            None if ch == '!' => return &line[..idx],
            Some(_) | None => {}
        }
    }
    line
}

fn parse_list_manifest(
    src: &str,
    manifest_name: &str,
) -> Result<Vec<(String, String)>, RatesError> {
    let mut entries = Vec::new();
    for (line_idx, raw_line) in src.lines().enumerate() {
        let trimmed = strip_manifest_comment(raw_line).trim();
        if trimmed.is_empty() {
            continue;
        }
        let Some(split_at) = trimmed.find(char::is_whitespace) else {
            return Err(RatesError::InvalidTableData(format!(
                "{manifest_name} line {}: missing table filename for '{}'",
                line_idx + 1,
                trimmed
            )));
        };
        let rate_name = &trimmed[..split_at];
        let rest = trimmed[split_at..].trim();
        let (fname, trailing) = if let Some(after_open) =
            rest.strip_prefix('\'')
        {
            let Some(end_q) = after_open.find('\'') else {
                return Err(RatesError::InvalidTableData(format!(
                        "{manifest_name} line {}: unterminated single-quoted filename",
                        line_idx + 1
                    )));
            };
            (&after_open[..end_q], after_open[end_q + 1..].trim())
        } else if let Some(after_open) = rest.strip_prefix('"') {
            let Some(end_q) = after_open.find('"') else {
                return Err(RatesError::InvalidTableData(format!(
                        "{manifest_name} line {}: unterminated double-quoted filename",
                        line_idx + 1
                    )));
            };
            (&after_open[..end_q], after_open[end_q + 1..].trim())
        } else {
            match rest.find(char::is_whitespace) {
                Some(end_tok) => (&rest[..end_tok], rest[end_tok..].trim()),
                None => (rest, ""),
            }
        };

        if fname.is_empty() || !trailing.is_empty() {
            return Err(RatesError::InvalidTableData(format!(
                "{manifest_name} line {}: malformed entry for '{rate_name}'",
                line_idx + 1
            )));
        }
        entries.push((rate_name.to_string(), fname.to_string()));
    }
    Ok(entries)
}

#[derive(Debug, Clone, Copy)]
struct GroupChild {
    oh_addr: u64,
}

#[derive(Debug, Clone, Copy)]
struct FloatType {
    elem_size: usize,
    big_endian: bool,
}

#[derive(Debug, Clone)]
enum DataLayout {
    Compact(Vec<u8>),
    Contiguous { addr: u64, size: u64 },
}

struct DatasetF64 {
    dims: Vec<usize>,
    vals: Vec<f64>,
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

impl<'a> Hdf5File<'a> {
    fn parse(bytes: &'a [u8]) -> Result<Self, RatesError> {
        if bytes.len() < 48 || &bytes[0..8] != HDF5_MAGIC {
            return Err(h5_err(
                "not an HDF5 payload (invalid magic signature)",
            ));
        }
        let sb_ver = bytes[8];
        match sb_ver {
            0 | 1 => {
                let off_size = usize::from(bytes[13]);
                let len_size = usize::from(bytes[14]);
                if !matches!(off_size, 4 | 8) || !matches!(len_size, 4 | 8) {
                    return Err(h5_err(format!(
                        "unsupported HDF5 offset/length sizes ({off_size}, {len_size})"
                    )));
                }
                let undef_addr = undef_sentinel(off_size);
                let mut pos = if sb_ver == 0 { 24_usize } else { 28_usize };
                let base_addr = read_uint(bytes, &mut pos, off_size)?;
                let _free_addr = read_uint(bytes, &mut pos, off_size)?;
                let _eof_addr = read_uint(bytes, &mut pos, off_size)?;
                let _driver_addr = read_uint(bytes, &mut pos, off_size)?;
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
                    return Err(h5_err(format!(
                        "unsupported HDF5 v{sb_ver} offset/length sizes ({off_size}, {len_size})"
                    )));
                }
                let undef_addr = undef_sentinel(off_size);
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
            other => Err(h5_err(format!(
                "unsupported HDF5 superblock version {other}"
            ))),
        }
    }

    fn resolve_addr(&self, addr: u64) -> Result<usize, RatesError> {
        let abs = addr.checked_add(self.base_addr).ok_or_else(|| {
            h5_err("HDF5 address overflow adding base address")
        })?;
        let off = usize::try_from(abs)
            .map_err(|_| h5_err("HDF5 address exceeds usize"))?;
        if off >= self.bytes.len() {
            return Err(h5_err(format!(
                "HDF5 address {off:#x} beyond EOF ({:#x})",
                self.bytes.len()
            )));
        }
        Ok(off)
    }

    fn read_offset(&self, pos: &mut usize) -> Result<u64, RatesError> {
        read_uint(self.bytes, pos, self.off_size)
    }

    fn read_length(&self, pos: &mut usize) -> Result<u64, RatesError> {
        read_uint(self.bytes, pos, self.len_size)
    }

    fn list_root_children(
        &self,
    ) -> Result<HashMap<String, GroupChild>, RatesError> {
        let mut children = HashMap::new();
        if let Some((bt, hp)) = self.root_symtab {
            self.collect_v1_symtab_children(bt, hp, &mut children)?;
            return Ok(children);
        }
        let msgs = self.read_object_messages(self.root_oh_addr)?;
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
                    children.insert(name, GroupChild { oh_addr: target_oh });
                    found_link = true;
                }
            }
        }
        if !found_link {
            return Err(h5_err(
                "HDF5 root group uses neither v1 symbol table nor compact v2 links",
            ));
        }
        Ok(children)
    }

    fn read_object_messages(
        &self,
        oh_addr: u64,
    ) -> Result<Vec<(u16, &'a [u8])>, RatesError> {
        let off = self.resolve_addr(oh_addr)?;
        let mut out = Vec::new();
        if off + 4 <= self.bytes.len() && &self.bytes[off..off + 4] == b"OHDR" {
            self.read_oh_v2_chunk(off, &mut out, 0)?;
        } else if self.bytes[off] == 1 {
            if off + 16 > self.bytes.len() {
                return Err(h5_err("truncated HDF5 v1 object header"));
            }
            let hdr_size = u32::from_le_bytes(
                self.bytes[off + 8..off + 12].try_into().unwrap_or([0; 4]),
            ) as usize;
            let start = off + 16;
            let end = start
                .checked_add(hdr_size)
                .ok_or_else(|| h5_err("HDF5 v1 object header size overflow"))?;
            if end > self.bytes.len() {
                return Err(h5_err("HDF5 v1 object header exceeds EOF"));
            }
            self.parse_oh_v1_messages(start, end, &mut out, 0)?;
        } else {
            return Err(h5_err(format!(
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
    ) -> Result<(), RatesError> {
        if depth > 16 {
            return Err(h5_err(
                "HDF5 object header continuation depth exceeded",
            ));
        }
        let mut pos = start;
        while pos + 8 <= end {
            let msg_type =
                u16::from_le_bytes([self.bytes[pos], self.bytes[pos + 1]]);
            let msg_size =
                u16::from_le_bytes([self.bytes[pos + 2], self.bytes[pos + 3]])
                    as usize;
            let body_start = pos + 8;
            let body_end =
                body_start.checked_add(msg_size).ok_or_else(|| {
                    h5_err("HDF5 v1 header message size overflow")
                })?;
            if body_end > end {
                return Err(h5_err("truncated HDF5 v1 header message"));
            }
            let body = &self.bytes[body_start..body_end];
            if msg_type == 0x0010 {
                let mut cpos = body_start;
                let cont_addr = self.read_offset(&mut cpos)?;
                let cont_len = self.read_length(&mut cpos)? as usize;
                let cont_off = self.resolve_addr(cont_addr)?;
                let cont_end = cont_off
                    .checked_add(cont_len)
                    .ok_or_else(|| h5_err("HDF5 v1 continuation overflow"))?;
                if cont_end > self.bytes.len() {
                    return Err(h5_err("HDF5 v1 continuation exceeds EOF"));
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
        out: &mut Vec<(u16, &'a [u8])>,
        depth: usize,
    ) -> Result<(), RatesError> {
        if off + 7 > self.bytes.len() {
            return Err(h5_err("truncated HDF5 v2 OHDR"));
        }
        let ver = self.bytes[off + 4];
        if ver != 2 {
            return Err(h5_err(format!("unexpected OHDR version {ver}")));
        }
        let flags = self.bytes[off + 5];
        let mut pos = off + 6;
        if (flags & 0x20) != 0 {
            pos += 16;
        }
        if (flags & 0x10) != 0 {
            pos += 4;
        }
        let chunk0_sz_bytes = 1_usize << usize::from(flags & 0x03);
        let chunk0_sz =
            read_uint(self.bytes, &mut pos, chunk0_sz_bytes)? as usize;
        let end = pos
            .checked_add(chunk0_sz)
            .ok_or_else(|| h5_err("HDF5 v2 OHDR chunk overflow"))?;
        if end > self.bytes.len() || chunk0_sz < 4 {
            return Err(h5_err("HDF5 v2 OHDR chunk exceeds EOF"));
        }
        self.parse_oh_v2_messages(pos, end - 4, (flags & 0x04) != 0, out, depth)
    }

    fn parse_oh_v2_messages(
        &self,
        start: usize,
        end: usize,
        has_creation_order: bool,
        out: &mut Vec<(u16, &'a [u8])>,
        depth: usize,
    ) -> Result<(), RatesError> {
        if depth > 16 {
            return Err(h5_err("HDF5 v2 continuation depth exceeded"));
        }
        let mut pos = start;
        let hdr_len = if has_creation_order { 6 } else { 4 };
        while pos + hdr_len <= end {
            let msg_type = u16::from(self.bytes[pos]);
            let msg_size =
                u16::from_le_bytes([self.bytes[pos + 1], self.bytes[pos + 2]])
                    as usize;
            let body_start = pos + hdr_len;
            let body_end =
                body_start.checked_add(msg_size).ok_or_else(|| {
                    h5_err("HDF5 v2 header message size overflow")
                })?;
            if body_end > end {
                break;
            }
            let body = &self.bytes[body_start..body_end];
            if msg_type == 0x0010 {
                let mut cpos = body_start;
                let cont_addr = self.read_offset(&mut cpos)?;
                let cont_len = self.read_length(&mut cpos)? as usize;
                let cont_off = self.resolve_addr(cont_addr)?;
                let cont_end = cont_off
                    .checked_add(cont_len)
                    .ok_or_else(|| h5_err("HDF5 v2 continuation overflow"))?;
                if cont_len < 8
                    || cont_end > self.bytes.len()
                    || &self.bytes[cont_off..cont_off + 4] != b"OCHK"
                {
                    return Err(h5_err("invalid HDF5 v2 OCHK continuation"));
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

    fn parse_v2_link(
        &self,
        body: &[u8],
    ) -> Result<Option<(String, u64)>, RatesError> {
        if body.len() < 2 || body[0] != 1 {
            return Err(h5_err("unsupported HDF5 v2 link message version"));
        }
        let flags = body[1];
        let mut pos = 2_usize;
        let link_type = if (flags & 0x08) != 0 {
            if pos >= body.len() {
                return Err(h5_err("truncated HDF5 v2 link type"));
            }
            let t = body[pos];
            pos += 1;
            t
        } else {
            0
        };
        if (flags & 0x04) != 0 {
            pos += 8;
        }
        if (flags & 0x10) != 0 {
            pos += 1;
        }
        let name_len_sz = 1_usize << usize::from(flags & 0x03);
        let name_len = read_uint(body, &mut pos, name_len_sz)? as usize;
        if pos + name_len > body.len() {
            return Err(h5_err("truncated HDF5 v2 link name"));
        }
        let name = String::from_utf8(body[pos..pos + name_len].to_vec())
            .map_err(|_| h5_err("non-UTF8 HDF5 link name"))?;
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
    ) -> Result<(), RatesError> {
        let heap_off = self.resolve_addr(heap_addr)?;
        if heap_off + 8 > self.bytes.len()
            || &self.bytes[heap_off..heap_off + 4] != b"HEAP"
        {
            return Err(h5_err(format!(
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
    ) -> Result<(), RatesError> {
        if depth > 32 {
            return Err(h5_err("HDF5 group B-tree depth exceeded"));
        }
        let off = self.resolve_addr(bt_addr)?;
        if off + 8 + 2 * self.off_size > self.bytes.len()
            || &self.bytes[off..off + 4] != b"TREE"
        {
            return Err(h5_err(format!(
                "invalid HDF5 group B-tree signature at {bt_addr:#x}"
            )));
        }
        let node_type = self.bytes[off + 4];
        let node_level = self.bytes[off + 5];
        let used =
            u16::from_le_bytes([self.bytes[off + 6], self.bytes[off + 7]])
                as usize;
        if node_type != 0 {
            return Err(h5_err(format!(
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
    ) -> Result<(), RatesError> {
        let off = self.resolve_addr(snod_addr)?;
        if off + 8 > self.bytes.len() || &self.bytes[off..off + 4] != b"SNOD" {
            return Err(h5_err(format!(
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
            let _cache_type = read_u32_le(self.bytes, &mut pos)?;
            let _reserved = read_u32_le(self.bytes, &mut pos)?;
            if pos + 16 > self.bytes.len() {
                return Err(h5_err("truncated HDF5 SNOD symbol entry"));
            }
            pos += 16;

            let name_start = heap_seg_off
                .checked_add(link_off)
                .ok_or_else(|| h5_err("HDF5 heap string offset overflow"))?;
            let tail = self.bytes.get(name_start..).ok_or_else(|| {
                h5_err("HDF5 heap string offset out of bounds")
            })?;
            let nul = tail
                .iter()
                .position(|&b| b == 0)
                .ok_or_else(|| h5_err("unterminated HDF5 local heap string"))?;
            let name = String::from_utf8(tail[..nul].to_vec())
                .map_err(|_| h5_err("non-UTF8 HDF5 symbol name"))?;
            out.insert(name, GroupChild { oh_addr: obj_oh });
        }
        Ok(())
    }

    fn read_dataset_f64(&self, oh_addr: u64) -> Result<DatasetF64, RatesError> {
        let msgs = self.read_object_messages(oh_addr)?;
        let mut dims: Option<Vec<usize>> = None;
        let mut dtype: Option<FloatType> = None;
        let mut layout: Option<DataLayout> = None;
        let mut has_filters = false;

        for (mtype, body) in msgs {
            match mtype {
                0x0001 => dims = Some(self.parse_dataspace(body)?),
                0x0003 => dtype = Some(parse_float_datatype(body)?),
                0x0008 => layout = Some(self.parse_data_layout(body)?),
                0x000b if body.len() >= 2 && body[1] > 0 => {
                    has_filters = true;
                }
                _ => {}
            }
        }
        if has_filters {
            return Err(h5_err(
                "filtered/compressed HDF5 weak table datasets are not supported",
            ));
        }
        let dims = dims.ok_or_else(|| {
            h5_err(format!("missing dataspace message at {oh_addr:#x}"))
        })?;
        let dtype = dtype.ok_or_else(|| {
            h5_err(format!("missing datatype message at {oh_addr:#x}"))
        })?;
        let layout = layout.ok_or_else(|| {
            h5_err(format!("missing data layout message at {oh_addr:#x}"))
        })?;

        let total_elems = dims.iter().copied().try_fold(1_usize, |a, b| {
            a.checked_mul(b)
                .ok_or_else(|| h5_err("HDF5 dataset element count overflow"))
        })?;
        let total_bytes = total_elems
            .checked_mul(dtype.elem_size)
            .ok_or_else(|| h5_err("HDF5 dataset byte size overflow"))?;

        let raw: &[u8] = match &layout {
            DataLayout::Compact(buf) => {
                if buf.len() < total_bytes {
                    return Err(h5_err("HDF5 compact dataset truncated"));
                }
                &buf[..total_bytes]
            }
            DataLayout::Contiguous { addr, size } => {
                if *addr == self.undef_addr {
                    return Ok(DatasetF64 {
                        dims,
                        vals: vec![0.0_f64; total_elems],
                    });
                }
                let off = self.resolve_addr(*addr)?;
                let sz = usize::try_from(*size)
                    .map_err(|_| h5_err("HDF5 contiguous size overflow"))?;
                let end = off
                    .checked_add(total_bytes)
                    .ok_or_else(|| h5_err("HDF5 contiguous slice overflow"))?;
                if sz < total_bytes || end > self.bytes.len() {
                    return Err(h5_err(
                        "HDF5 contiguous dataset out of bounds",
                    ));
                }
                &self.bytes[off..end]
            }
        };

        let mut vals = Vec::with_capacity(total_elems);
        match (dtype.elem_size, dtype.big_endian) {
            (8, false) => {
                for &b in raw.as_chunks::<8>().0 {
                    vals.push(f64::from_le_bytes(b));
                }
            }
            (8, true) => {
                for &b in raw.as_chunks::<8>().0 {
                    vals.push(f64::from_be_bytes(b));
                }
            }
            (4, false) => {
                for &b in raw.as_chunks::<4>().0 {
                    vals.push(f64::from(f32::from_le_bytes(b)));
                }
            }
            (4, true) => {
                for &b in raw.as_chunks::<4>().0 {
                    vals.push(f64::from(f32::from_be_bytes(b)));
                }
            }
            _ => {
                return Err(h5_err(format!(
                    "unsupported float byte width {}",
                    dtype.elem_size
                )));
            }
        }
        Ok(DatasetF64 { dims, vals })
    }

    fn parse_dataspace(&self, body: &[u8]) -> Result<Vec<usize>, RatesError> {
        if body.len() < 4 {
            return Err(h5_err("truncated HDF5 dataspace message"));
        }
        let ver = body[0];
        let rank = usize::from(body[1]);
        let mut pos = match ver {
            1 => 8_usize,
            2 => 4_usize,
            other => {
                return Err(h5_err(format!(
                    "unsupported HDF5 dataspace version {other}"
                )));
            }
        };
        let mut dims = Vec::with_capacity(rank);
        for _ in 0..rank {
            let d = read_uint(body, &mut pos, self.len_size)?;
            dims.push(
                usize::try_from(d)
                    .map_err(|_| h5_err("HDF5 dimension exceeds usize"))?,
            );
        }
        Ok(dims)
    }

    fn parse_data_layout(&self, body: &[u8]) -> Result<DataLayout, RatesError> {
        if body.len() < 2 {
            return Err(h5_err("truncated HDF5 data layout message"));
        }
        let ver = body[0];
        let cls = body[1];
        match (ver, cls) {
            (3 | 4, 0) => {
                if body.len() < 4 {
                    return Err(h5_err("truncated HDF5 compact layout"));
                }
                let sz = u16::from_le_bytes([body[2], body[3]]) as usize;
                if 4 + sz > body.len() {
                    return Err(h5_err("truncated HDF5 compact payload"));
                }
                Ok(DataLayout::Compact(body[4..4 + sz].to_vec()))
            }
            (3 | 4, 1) => {
                let mut pos = 2_usize;
                let addr = read_uint(body, &mut pos, self.off_size)?;
                let size = read_uint(body, &mut pos, self.len_size)?;
                Ok(DataLayout::Contiguous { addr, size })
            }
            (3 | 4, 2) => Err(h5_err(
                "chunked HDF5 weak table datasets are not supported",
            )),
            _ => Err(h5_err(format!(
                "unsupported HDF5 data layout version={ver} class={cls}"
            ))),
        }
    }
}

fn parse_float_datatype(body: &[u8]) -> Result<FloatType, RatesError> {
    if body.len() < 8 {
        return Err(h5_err("truncated HDF5 datatype message"));
    }
    let class = body[0] & 0x0f;
    if class != 1 {
        return Err(h5_err(format!(
            "expected IEEE floating-point HDF5 datatype (class 1), got class {class}"
        )));
    }
    let big_endian = (body[1] & 0x01) != 0;
    let elem_size =
        u32::from_le_bytes([body[4], body[5], body[6], body[7]]) as usize;
    if !matches!(elem_size, 4 | 8) {
        return Err(h5_err(format!(
            "unsupported HDF5 float byte size {elem_size}"
        )));
    }
    Ok(FloatType {
        elem_size,
        big_endian,
    })
}

fn undef_sentinel(off_size: usize) -> u64 {
    if off_size == 4 {
        u64::from(u32::MAX)
    } else {
        u64::MAX
    }
}

fn h5_err(msg: impl Into<String>) -> RatesError {
    RatesError::InvalidTableData(msg.into())
}

fn read_u32_le(bytes: &[u8], pos: &mut usize) -> Result<u32, RatesError> {
    if *pos + 4 > bytes.len() {
        return Err(h5_err("unexpected EOF reading HDF5 u32"));
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

fn read_uint(
    bytes: &[u8],
    pos: &mut usize,
    width: usize,
) -> Result<u64, RatesError> {
    if *pos + width > bytes.len() {
        return Err(h5_err("unexpected EOF reading HDF5 integer"));
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
            return Err(h5_err(format!(
                "unsupported HDF5 integer width {other}"
            )));
        }
    };
    *pos += width;
    Ok(v)
}
