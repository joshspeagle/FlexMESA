// SPDX-License-Identifier: LGPL-3.0-or-later
//! Full opacity table builder and verified runtime cache manager (`FMKAP001`).
//!
//! Loads the verified immutable `$FLEXMESA_DATA_DIR/kap_full/kap_tables_full.bin`
//! installed by `flexmesa data fetch`, or builds the `FMKAP001` payload from
//! upstream `MESA_DIR/data/kap_data` ASCII files plus a pre-converted
//! `aesopus_section.bin` (caching the `FMCENV01` output strictly under
//! [`flexmesa_utils::resolve_flexmesa_cache_dir`]). Never invokes Python at runtime.

use crate::loader::KapTables;
use crate::loader_parse::parse_kap_tables;
use crate::spline::get_output_string;
use crate::types::KapError;
use flexmesa_utils::cache_envelope::{
    read_verified_cache, write_verified_cache_atomic,
};
use flexmesa_utils::sha256::{compute_sha256_file, Sha256Hasher};
use flexmesa_utils::{
    is_bad, resolve_flexmesa_cache_dir, resolve_flexmesa_data_dir,
};
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};

/// Magic header for `FMKAP001` opacity binary payload.
pub const KAP_FULL_BIN_MAGIC: &[u8; 8] = b"FMKAP001";

/// Builder version stamped into `FMCENV01` runtime caches for `kap`.
const KAP_CACHE_BUILDER_VERSION: u32 = 1;

const FIXED_PREFIXES: &[&str] = &[
    "gs98",
    "gn93",
    "a09",
    "OP_gs98",
    "OP_a09_nans_removed_by_hand",
    "lowT_fa05_gs98",
    "lowT_fa05_gn93",
    "lowT_fa05_a09p",
];

const FIXED_ZS: [f64; 13] = [
    0.0, 0.0001, 0.0003, 0.001, 0.002, 0.004, 0.01, 0.02, 0.03, 0.04, 0.06,
    0.08, 0.100,
];
const FIXED_XS: [f64; 10] =
    [0.00, 0.10, 0.20, 0.35, 0.50, 0.70, 0.80, 0.90, 0.95, 1.0];
const FIXED_NUM_XS: [usize; 13] =
    [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 8];

// Matches `kap_lowT_Freedman11_Zs` in `kap/public/kap_def.f90:616-622` at tag 26.4.1.
const FREEDMAN_ZS: [f64; 7] = [0.01, 0.02, 0.04, 0.10, 0.20, 0.63, 1.00];

const CO_PREFIXES: &[&str] = &["gn93_co", "gs98_co", "a09_co"];
const CO_ZS: [f64; 8] =
    [0.000, 0.001, 0.004, 0.010, 0.020, 0.030, 0.050, 0.100];
const CO_XS: [f64; 5] = [0.00, 0.03, 0.10, 0.35, 0.70];

/// Resolve an optional upstream MESA `kap_data` directory when raw ASCII tables are present.
#[must_use]
pub fn find_kap_data_dir() -> Option<PathBuf> {
    if let Ok(val) = std::env::var("FLEXMESA_DATA_DIR") {
        let p = PathBuf::from(&val);
        if p.join("kap_data").is_dir() {
            return Some(p.join("kap_data"));
        }
        if p.join("gs98_z0.0_x0.0.data").is_file() {
            return Some(p);
        }
    }
    if let Ok(val) = std::env::var("MESA_DIR") {
        let p = PathBuf::from(val).join("data").join("kap_data");
        if p.is_dir() {
            return Some(p);
        }
    }
    None
}

/// Resolve the immutable `kap_full` bundle path under `$FLEXMESA_DATA_DIR`.
#[must_use]
pub fn find_kap_bundle_dir() -> PathBuf {
    resolve_flexmesa_data_dir().join("kap_full")
}

/// Enumerate every relative `.data` file under `kap_dir` consumed by
/// [`build_full_kap_tables_bytes`] in deterministic order (`1,150` files in
/// MESA `26.4.1`: `1,008` fixed/low-T, `7` `Freedman11`, `120` Type-2 CO,
/// `kR_Z_fCN.data`, and `14` `kapCN` `kR_Z*.data` files).
pub(crate) fn kap_source_relpaths(
    kap_dir: &Path,
) -> Result<Vec<String>, KapError> {
    let mut rels = Vec::with_capacity(1152);
    for &prefix in FIXED_PREFIXES {
        for (iz, &z) in FIXED_ZS.iter().enumerate() {
            for &x_raw in &FIXED_XS[..FIXED_NUM_XS[iz]] {
                let x = if x_raw + z > 1.0 { 1.0 - z } else { x_raw };
                rels.push(format!(
                    "{prefix}_z{}_x{}.data",
                    get_output_string(z)?,
                    get_output_string(x)?
                ));
            }
        }
    }
    for &z in &FREEDMAN_ZS {
        rels.push(format!("lowT_Freedman11_z{}.data", get_output_string(z)?));
    }
    for &prefix in CO_PREFIXES {
        for &z in &CO_ZS {
            for &x in &CO_XS {
                rels.push(format!(
                    "{prefix}_z{}_x{}.data",
                    get_output_string(z)?,
                    get_output_string(x)?
                ));
            }
        }
    }
    rels.push("kR_Z_fCN.data".to_string());
    let fcn_path = kap_dir.join("kR_Z_fCN.data");
    let fcn_text = fs::read_to_string(&fcn_path).map_err(|e| {
        KapError::TableIo(format!("failed to read {}: {e}", fcn_path.display()))
    })?;
    let mut cn_fnames = Vec::new();
    for line in fcn_text.lines().skip(1) {
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() < 11 {
            continue;
        }
        let z = parse_f64_str(toks[0], &fcn_path, "kapCN Z")?;
        let fname = kapcn_filename(z);
        if fname.is_empty() {
            return Err(KapError::TableIo(format!(
                "unsupported kapCN metallicity Z={z}"
            )));
        }
        if !cn_fnames.contains(&fname) {
            cn_fnames.push(fname);
        }
    }
    cn_fnames.reverse();
    for fname in cn_fnames {
        rels.push(fname.to_string());
    }
    Ok(rels)
}

/// Compute a combined `SHA-256` fingerprint over every raw `kap_data` `.data`
/// input consumed by [`build_full_kap_tables_bytes`] plus `aesopus_section.bin`.
fn compute_kap_source_sha256(
    kap_dir: &Path,
    aesopus_section_path: &Path,
) -> Result<String, KapError> {
    let aes_len = fs::metadata(aesopus_section_path)
        .map_err(|e| {
            KapError::TableIo(format!(
                "failed to stat {}: {e}",
                aesopus_section_path.display()
            ))
        })?
        .len();
    let aes_sha = compute_sha256_file(aesopus_section_path).map_err(|e| {
        KapError::TableIo(format!(
            "failed to read {}: {e}",
            aesopus_section_path.display()
        ))
    })?;

    let relpaths = kap_source_relpaths(kap_dir)?;
    let file_digests: Vec<(String, u64, String)> = relpaths
        .into_par_iter()
        .map(|rel| {
            let p = kap_dir.join(&rel);
            let len = fs::metadata(&p)
                .map_err(|e| {
                    KapError::TableIo(format!(
                        "failed to stat {}: {e}",
                        p.display()
                    ))
                })?
                .len();
            let digest = compute_sha256_file(&p).map_err(|e| {
                KapError::TableIo(format!(
                    "failed to read {}: {e}",
                    p.display()
                ))
            })?;
            Ok((rel, len, digest))
        })
        .collect::<Result<Vec<_>, KapError>>()?;

    let mut hasher = Sha256Hasher::new();
    hasher.update(b"aesopus_section.bin\0");
    hasher.update(&aes_len.to_le_bytes());
    hasher.update(aes_sha.as_bytes());
    for (rel, len, digest) in file_digests {
        hasher.update(rel.as_bytes());
        hasher.update(&[0_u8]);
        hasher.update(&len.to_le_bytes());
        hasher.update(digest.as_bytes());
    }
    Ok(hasher.finalize_hex())
}

/// Load [`KapTables`] from `$FLEXMESA_DATA_DIR/kap_full/kap_tables_full.bin`, or
/// build from upstream `MESA_DIR/data/kap_data` + `aesopus_section.bin` with
/// verified runtime caching under [`resolve_flexmesa_cache_dir`].
///
/// # Errors
/// Returns [`KapError::TableIo`] if neither the verified bundle nor upstream
/// tables + `aesopus_section.bin` can be found, read, or parsed, or if a
/// runtime cache file fails verification.
pub(crate) fn load_or_build_kap_tables() -> Result<KapTables, KapError> {
    let full_bin_path = find_kap_bundle_dir().join("kap_tables_full.bin");
    if full_bin_path.is_file() {
        let bytes = fs::read(&full_bin_path).map_err(|e| {
            KapError::TableIo(format!(
                "failed to read {}: {e}",
                full_bin_path.display()
            ))
        })?;
        return parse_kap_tables(&bytes);
    }

    if let Some(kap_dir) = find_kap_data_dir() {
        let aes_path = kap_dir.join("aesopus_section.bin");
        if !aes_path.is_file() {
            return Err(KapError::TableIo(format!(
                "FlexMESA opacity bundle not found at {}, and raw source directory {} \
                 lacks pre-converted aesopus_section.bin. Run 'flexmesa data fetch' or \
                 'make data-physics' to build the verified FMKAP001 bundle offline.",
                full_bin_path.display(),
                kap_dir.display()
            )));
        }
        let source_sha = compute_kap_source_sha256(&kap_dir, &aes_path)?;
        if let Some(cache_root) = resolve_flexmesa_cache_dir() {
            let cache_path = cache_root
                .join("kap")
                .join(format!("kap_tables_full.{}.bin", &source_sha[..16]));
            if cache_path.is_file() {
                let payload = read_verified_cache(
                    &cache_path,
                    KAP_CACHE_BUILDER_VERSION,
                    KAP_FULL_BIN_MAGIC,
                    &source_sha,
                )
                .map_err(KapError::TableIo)?;
                return parse_kap_tables(&payload);
            }
            let bytes = build_full_kap_tables_bytes(&kap_dir, &aes_path)?;
            let tables = parse_kap_tables(&bytes)?;
            write_verified_cache_atomic(
                &cache_path,
                KAP_CACHE_BUILDER_VERSION,
                KAP_FULL_BIN_MAGIC,
                &source_sha,
                &bytes,
            )
            .map_err(KapError::TableIo)?;
            return Ok(tables);
        }
        let bytes = build_full_kap_tables_bytes(&kap_dir, &aes_path)?;
        return parse_kap_tables(&bytes);
    }

    Err(KapError::TableIo(format!(
        "FlexMESA opacity tables not found at {}. \
         Run 'flexmesa data fetch' (or set FLEXMESA_DATA_DIR) to download \
         the verified data-v1 physics table bundle.",
        full_bin_path.display()
    )))
}

fn push_u8(buf: &mut Vec<u8>, v: u8) {
    buf.push(v);
}
fn push_u32(buf: &mut Vec<u8>, v: usize) {
    buf.extend_from_slice(&(v as u32).to_le_bytes());
}
fn push_f64(buf: &mut Vec<u8>, v: f64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_str(buf: &mut Vec<u8>, s: &str) {
    push_u32(buf, s.len());
    buf.extend_from_slice(s.as_bytes());
}

fn parse_f64_tok<'a>(
    toks: &mut impl Iterator<Item = &'a str>,
    path: &Path,
    label: &str,
) -> Result<f64, KapError> {
    let raw = toks.next().ok_or_else(|| {
        KapError::TableIo(format!(
            "unexpected EOF reading {label} in {}",
            path.display()
        ))
    })?;
    let val = raw.parse::<f64>().map_err(|e| {
        KapError::TableIo(format!(
            "invalid float '{raw}' for {label} in {}: {e}",
            path.display()
        ))
    })?;
    if is_bad(val) {
        return Err(KapError::TableIo(format!(
            "non-finite float '{raw}' for {label} in {}",
            path.display()
        )));
    }
    Ok(val)
}

fn parse_usize_str(
    raw: &str,
    path: &Path,
    label: &str,
) -> Result<usize, KapError> {
    raw.parse::<usize>().map_err(|e| {
        KapError::TableIo(format!(
            "invalid integer '{raw}' for {label} in {}: {e}",
            path.display()
        ))
    })
}

fn parse_f64_str(raw: &str, path: &Path, label: &str) -> Result<f64, KapError> {
    let val = raw.parse::<f64>().map_err(|e| {
        KapError::TableIo(format!(
            "invalid float '{raw}' for {label} in {}: {e}",
            path.display()
        ))
    })?;
    if is_bad(val) {
        return Err(KapError::TableIo(format!(
            "non-finite float '{raw}' for {label} in {}",
            path.display()
        )));
    }
    Ok(val)
}

fn parse_fixed_file_bytes(
    path: &Path,
    x: f64,
    z: f64,
) -> Result<Vec<u8>, KapError> {
    let text = fs::read_to_string(path).map_err(|e| {
        KapError::TableIo(format!("failed to read {}: {e}", path.display()))
    })?;
    let mut lines = text.lines();
    let eof_err =
        || KapError::TableIo(format!("unexpected EOF in {}", path.display()));
    lines.next().ok_or_else(eof_err)?;
    lines.next().ok_or_else(eof_err)?;
    let hdr_line = lines.next().ok_or_else(eof_err)?;
    let hdr: Vec<&str> = hdr_line.split_whitespace().collect();
    if hdr.len() < 10 {
        return Err(KapError::TableIo(format!(
            "short header ({} cols < 10) in {}",
            hdr.len(),
            path.display()
        )));
    }
    let num_lr = parse_usize_str(hdr[4], path, "num_lr")?;
    let log_r_min = parse_f64_str(hdr[5], path, "log_r_min")?;
    let log_r_max = parse_f64_str(hdr[6], path, "log_r_max")?;
    let num_lt = parse_usize_str(hdr[7], path, "num_lt")?;
    let log_t_min = parse_f64_str(hdr[8], path, "log_t_min")?;
    let log_t_max = parse_f64_str(hdr[9], path, "log_t_max")?;
    lines.next().ok_or_else(eof_err)?;
    lines.next().ok_or_else(eof_err)?;

    let mut toks = lines.flat_map(str::split_whitespace);
    let mut log_rs = Vec::with_capacity(num_lr);
    for _ in 0..num_lr {
        log_rs.push(parse_f64_tok(&mut toks, path, "logR")?);
    }
    let mut log_ts = Vec::with_capacity(num_lt);
    let mut raw_i16 = Vec::with_capacity(num_lr * num_lt);
    for _ in 0..num_lt {
        log_ts.push(parse_f64_tok(&mut toks, path, "logT")?);
        for _ in 0..num_lr {
            let v = parse_f64_tok(&mut toks, path, "log_kap")?;
            raw_i16.push((v * 1000.0).round() as i16);
        }
    }

    let mut out =
        Vec::with_capacity(64 + (num_lr + num_lt) * 8 + raw_i16.len() * 2);
    push_u8(&mut out, 1);
    push_f64(&mut out, x);
    push_f64(&mut out, z);
    push_f64(&mut out, log_r_min);
    push_f64(&mut out, log_r_max);
    push_f64(&mut out, log_t_min);
    push_f64(&mut out, log_t_max);
    push_u32(&mut out, num_lr);
    push_u32(&mut out, num_lt);
    for v in log_rs {
        push_f64(&mut out, v);
    }
    for v in log_ts {
        push_f64(&mut out, v);
    }
    for v in raw_i16 {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(out)
}

fn parse_co_file_bytes(
    path: &Path,
    x: f64,
    z: f64,
) -> Result<([f64; 4], Vec<u8>), KapError> {
    let text = fs::read_to_string(path).map_err(|e| {
        KapError::TableIo(format!("failed to read {}: {e}", path.display()))
    })?;
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < 10 {
        return Err(KapError::TableIo(format!(
            "short CO kap file {}",
            path.display()
        )));
    }
    let hdr: Vec<&str> = lines[2].split_whitespace().collect();
    if hdr.len() < 15 {
        return Err(KapError::TableIo(format!(
            "short CO header ({} cols < 15) in {}",
            hdr.len(),
            path.display()
        )));
    }
    let num_co_tables = parse_usize_str(hdr[2], path, "num_co_tables")?;
    let zfracs = [
        parse_f64_str(hdr[5], path, "zfrac_C")?,
        parse_f64_str(hdr[6], path, "zfrac_N")?,
        parse_f64_str(hdr[7], path, "zfrac_O")?,
        parse_f64_str(hdr[8], path, "zfrac_Ne")?,
    ];
    let num_lr = parse_usize_str(hdr[9], path, "num_lr")?;
    let log_r_min = parse_f64_str(hdr[10], path, "log_r_min")?;
    let log_r_max = parse_f64_str(hdr[11], path, "log_r_max")?;
    let num_lt = parse_usize_str(hdr[12], path, "num_lt")?;
    let log_t_min = parse_f64_str(hdr[13], path, "log_t_min")?;
    let log_t_max = parse_f64_str(hdr[14], path, "log_t_max")?;

    let num_dxc_gt_dxo_str =
        lines[4].split_whitespace().next().ok_or_else(|| {
            KapError::TableIo(format!(
                "missing num_dXC_gt_dXO in {}",
                path.display()
            ))
        })?;
    let num_dxc_gt_dxo = parse_usize_str(num_dxc_gt_dxo_str, path, "num_dxc")?;
    let mut dxc_dxo = Vec::with_capacity(num_co_tables);
    let mut line_idx = 6;
    while line_idx < lines.len() && dxc_dxo.len() < num_co_tables {
        let row: Vec<&str> = lines[line_idx].split_whitespace().collect();
        if row.len() == 4 && row[0].parse::<usize>().is_ok() {
            let dxc = parse_f64_str(row[2], path, "dxc")?;
            let dxo = parse_f64_str(row[3], path, "dxo")?;
            dxc_dxo.push((dxc, dxo));
        }
        line_idx += 1;
    }

    let mut log_rs = Vec::new();
    let mut log_ts = Vec::new();
    let mut sub_grids: Vec<Vec<i16>> = Vec::with_capacity(num_co_tables);
    while line_idx < lines.len() && sub_grids.len() < num_co_tables {
        if lines[line_idx].trim_start().starts_with("logT") {
            let mut toks = lines[line_idx + 1..]
                .iter()
                .flat_map(|l| l.split_whitespace());
            let mut lrs = Vec::with_capacity(num_lr);
            for _ in 0..num_lr {
                lrs.push(parse_f64_tok(&mut toks, path, "CO logR")?);
            }
            let mut lts = Vec::with_capacity(num_lt);
            let mut grid = Vec::with_capacity(num_lr * num_lt);
            for _ in 0..num_lt {
                lts.push(parse_f64_tok(&mut toks, path, "CO logT")?);
                for _ in 0..num_lr {
                    let v = parse_f64_tok(&mut toks, path, "CO log_kap")?;
                    grid.push((v * 1000.0).round() as i16);
                }
            }
            if log_rs.is_empty() {
                log_rs = lrs;
                log_ts = lts;
            }
            sub_grids.push(grid);
            line_idx += 2 + num_lt;
        } else {
            line_idx += 1;
        }
    }
    if sub_grids.len() != num_co_tables {
        return Err(KapError::TableIo(format!(
            "expected {num_co_tables} CO subtables, found {} in {}",
            sub_grids.len(),
            path.display()
        )));
    }

    let mut out = Vec::new();
    push_u8(&mut out, 1);
    push_f64(&mut out, x);
    push_f64(&mut out, z);
    push_u32(&mut out, num_co_tables);
    push_u32(&mut out, num_dxc_gt_dxo);
    push_f64(&mut out, log_r_min);
    push_f64(&mut out, log_r_max);
    push_f64(&mut out, log_t_min);
    push_f64(&mut out, log_t_max);
    push_u32(&mut out, num_lr);
    push_u32(&mut out, num_lt);
    for v in log_rs {
        push_f64(&mut out, v);
    }
    for v in log_ts {
        push_f64(&mut out, v);
    }
    for ((dxc, dxo), grid) in dxc_dxo.into_iter().zip(sub_grids) {
        push_f64(&mut out, dxc);
        push_f64(&mut out, dxo);
        push_u8(&mut out, 1);
        for v in grid {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    Ok((zfracs, out))
}

fn read_aesopus_section_bytes(
    aesopus_section_path: &Path,
) -> Result<Vec<u8>, KapError> {
    let bytes = fs::read(aesopus_section_path).map_err(|e| {
        KapError::TableIo(format!(
            "failed to read AESOPUS section {}: {e}",
            aesopus_section_path.display()
        ))
    })?;
    if bytes.len() < 44 {
        return Err(KapError::TableIo(format!(
            "truncated AESOPUS section {} ({} bytes)",
            aesopus_section_path.display(),
            bytes.len()
        )));
    }
    Ok(bytes)
}

fn kapcn_filename(z: f64) -> &'static str {
    match (z * 1.0e5).round() as i64 {
        1 => "kR_Z1E-5.data",
        3 => "kR_Z3E-5.data",
        10 => "kR_Z1E-4.data",
        30 => "kR_Z3E-4.data",
        100 => "kR_Z1E-3.data",
        200 => "kR_Z2E-3.data",
        300 => "kR_Z3E-3.data",
        400 => "kR_Z4E-3.data",
        500 => "kR_Z5E-3.data",
        600 => "kR_Z6E-3.data",
        800 => "kR_Z8E-3.data",
        1000 => "kR_Z1E-2.data",
        2000 => "kR_Z2E-2.data",
        4000 => "kR_Z4E-2.data",
        _ => "",
    }
}

fn build_kapcn_section_bytes(kap_dir: &Path) -> Result<Vec<u8>, KapError> {
    let fcn_path = kap_dir.join("kR_Z_fCN.data");
    let fcn_text = fs::read_to_string(&fcn_path).map_err(|e| {
        KapError::TableIo(format!("failed to read {}: {e}", fcn_path.display()))
    })?;
    let mut rows: Vec<(f64, [f64; 7], [f64; 3])> = Vec::new();
    for line in fcn_text.lines().skip(1) {
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() < 11 {
            continue;
        }
        let z = parse_f64_str(toks[0], &fcn_path, "kapCN Z")?;
        let mut fcs = [0.0; 7];
        for i in 0..7 {
            fcs[i] = parse_f64_str(toks[1 + i], &fcn_path, "kapCN fC")?;
        }
        let mut fns = [0.0; 3];
        for i in 0..3 {
            fns[i] = parse_f64_str(toks[8 + i], &fcn_path, "kapCN fN")?;
        }
        rows.push((z, fcs, fns));
    }
    rows.reverse();

    let mut log_rs = Vec::new();
    let mut log_ts = Vec::new();
    let mut z_payloads = Vec::with_capacity(rows.len());
    for &(z, fcs, fns) in &rows {
        let fname = kapcn_filename(z);
        if fname.is_empty() {
            return Err(KapError::TableIo(format!(
                "unsupported kapCN metallicity Z={z}"
            )));
        }
        let cn_path = kap_dir.join(fname);
        let text = fs::read_to_string(&cn_path).map_err(|e| {
            KapError::TableIo(format!(
                "failed to read {}: {e}",
                cn_path.display()
            ))
        })?;
        let lines: Vec<&str> = text.lines().collect();
        let mut sub_grids = Vec::with_capacity(63);
        let mut idx = 0;
        while idx < lines.len() && sub_grids.len() < 63 {
            let trimmed = lines[idx].trim_start();
            if trimmed.starts_with("log T") {
                if log_rs.is_empty() {
                    for tok in trimmed.split_whitespace().skip(2) {
                        log_rs.push(parse_f64_str(
                            tok,
                            &cn_path,
                            "kapCN logR",
                        )?);
                    }
                }
                let num_lr = log_rs.len();
                if idx + 18 >= lines.len() {
                    return Err(KapError::TableIo(format!(
                        "truncated kapCN block in {}",
                        cn_path.display()
                    )));
                }
                let mut grid = Vec::with_capacity(num_lr * 18);
                let mut lts = Vec::with_capacity(18);
                for j in 0..18 {
                    let mut toks = lines[idx + 1 + j].split_whitespace();
                    lts.push(parse_f64_tok(&mut toks, &cn_path, "kapCN logT")?);
                    for _ in 0..num_lr {
                        let v = parse_f64_tok(
                            &mut toks,
                            &cn_path,
                            "kapCN log_kap",
                        )?;
                        grid.push((v * 1000.0).round() as i16);
                    }
                }
                if log_ts.is_empty() {
                    log_ts = lts;
                }
                sub_grids.push(grid);
                idx += 19;
            } else {
                idx += 1;
            }
        }
        if sub_grids.len() != 63 {
            return Err(KapError::TableIo(format!(
                "expected 63 kapCN subgrids, found {} in {}",
                sub_grids.len(),
                cn_path.display()
            )));
        }
        z_payloads.push((z, fcs, fns, sub_grids));
    }

    let mut out = Vec::new();
    push_u32(&mut out, z_payloads.len());
    push_u32(&mut out, log_rs.len());
    push_u32(&mut out, log_ts.len());
    for v in log_rs {
        push_f64(&mut out, v);
    }
    for v in log_ts {
        push_f64(&mut out, v);
    }
    for (z, fcs, fns, grids) in z_payloads {
        push_f64(&mut out, z);
        for v in fcs {
            push_f64(&mut out, v);
        }
        for v in fns {
            push_f64(&mut out, v);
        }
        for g in grids {
            for v in g {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    Ok(out)
}

/// Assemble an `FMKAP001` payload buffer from upstream `kap_data` ASCII files
/// and a pre-converted `aesopus_section.bin`.
///
/// # Errors
/// Returns [`KapError::TableIo`] if any upstream table or `aesopus_section.bin`
/// is missing, malformed, or fails validation.
pub fn build_full_kap_tables_bytes(
    kap_dir: &Path,
    aesopus_section_path: &Path,
) -> Result<Vec<u8>, KapError> {
    let mut out = Vec::with_capacity(120_000_000);
    out.extend_from_slice(KAP_FULL_BIN_MAGIC);

    // 1. Fixed-metal & low-T sets (8 standard + Freedman11 = 9 sets)
    push_u32(&mut out, FIXED_PREFIXES.len() + 1);
    for &prefix in FIXED_PREFIXES {
        push_str(&mut out, prefix);
        push_u32(&mut out, FIXED_ZS.len());
        let mut tasks = Vec::with_capacity(126);
        for (iz, &z) in FIXED_ZS.iter().enumerate() {
            for &x_raw in &FIXED_XS[..FIXED_NUM_XS[iz]] {
                let x = if x_raw + z > 1.0 { 1.0 - z } else { x_raw };
                let fname = format!(
                    "{prefix}_z{}_x{}.data",
                    get_output_string(z)?,
                    get_output_string(x)?
                );
                tasks.push((iz, x, z, kap_dir.join(fname)));
            }
        }
        let parsed: Vec<(usize, Vec<u8>)> = tasks
            .into_par_iter()
            .map(|(iz, x, z, p)| Ok((iz, parse_fixed_file_bytes(&p, x, z)?)))
            .collect::<Result<Vec<_>, KapError>>()?;
        let mut pos = 0;
        for (iz, &z) in FIXED_ZS.iter().enumerate() {
            let n_x = FIXED_NUM_XS[iz];
            push_f64(&mut out, z);
            push_u32(&mut out, n_x);
            for _ in 0..n_x {
                out.extend_from_slice(&parsed[pos].1);
                pos += 1;
            }
        }
    }
    // Freedman11
    push_str(&mut out, "lowT_Freedman11");
    push_u32(&mut out, FREEDMAN_ZS.len());
    for &z in &FREEDMAN_ZS {
        let fname = format!("lowT_Freedman11_z{}.data", get_output_string(z)?);
        let x_bytes = parse_fixed_file_bytes(&kap_dir.join(fname), 0.0, z)?;
        push_f64(&mut out, z);
        push_u32(&mut out, 1);
        out.extend_from_slice(&x_bytes);
    }

    // 2. Type 2 CO sets (gn93_co, gs98_co, a09_co)
    push_u32(&mut out, CO_PREFIXES.len());
    for &prefix in CO_PREFIXES {
        push_str(&mut out, prefix);
        push_u32(&mut out, CO_ZS.len());
        let mut tasks = Vec::with_capacity(40);
        for (iz, &z) in CO_ZS.iter().enumerate() {
            for &x in &CO_XS {
                let fname = format!(
                    "{prefix}_z{}_x{}.data",
                    get_output_string(z)?,
                    get_output_string(x)?
                );
                tasks.push((iz, x, z, kap_dir.join(fname)));
            }
        }
        let parsed: Vec<(usize, [f64; 4], Vec<u8>)> = tasks
            .into_par_iter()
            .map(|(iz, x, z, p)| {
                let (zf, b) = parse_co_file_bytes(&p, x, z)?;
                Ok((iz, zf, b))
            })
            .collect::<Result<Vec<_>, KapError>>()?;
        let mut pos = 0;
        for &z in &CO_ZS {
            let zf = parsed[pos].1;
            push_f64(&mut out, z);
            for v in zf {
                push_f64(&mut out, v);
            }
            push_u32(&mut out, CO_XS.len());
            for _ in 0..CO_XS.len() {
                out.extend_from_slice(&parsed[pos].2);
                pos += 1;
            }
        }
    }

    // 3. AESOPUS_AGSS09 (pre-converted offline from native float32 HDF5)
    let aes_bytes = read_aesopus_section_bytes(aesopus_section_path)?;
    out.extend_from_slice(&aes_bytes);

    // 4. kapCN
    let cn_bytes = build_kapcn_section_bytes(kap_dir)?;
    out.extend_from_slice(&cn_bytes);

    let _validated = parse_kap_tables(&out)?;
    Ok(out)
}

/// Build and atomically write a validated `FMKAP001` file at `out_path`.
///
/// # Errors
/// Returns [`KapError::TableIo`] if building, validating, or writing fails.
pub fn build_full_kap_tables_file(
    kap_dir: &Path,
    aesopus_section_path: &Path,
    out_path: &Path,
) -> Result<(), KapError> {
    let bytes = build_full_kap_tables_bytes(kap_dir, aesopus_section_path)?;
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            KapError::TableIo(format!(
                "failed to create output dir {}: {e}",
                parent.display()
            ))
        })?;
    }
    let tmp =
        out_path.with_extension(format!("bin.tmp.{}", std::process::id()));
    fs::write(&tmp, &bytes).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        KapError::TableIo(format!("failed to write {}: {e}", tmp.display()))
    })?;
    fs::rename(&tmp, out_path).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        KapError::TableIo(format!(
            "failed to move {} into place: {e}",
            out_path.display()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::{compute_kap_source_sha256, kap_source_relpaths};
    use crate::types::KapError;
    use std::fs;

    const SAMPLE_FCN: &str = "\
Z fC1 fC2 fC3 fC4 fC5 fC6 fC7 fN1 fN2 fN3
0.04000 1 1 1 1 1 1 1 1 1 1
0.02000 1 1 1 1 1 1 1 1 1 1
0.01000 1 1 1 1 1 1 1 1 1 1
0.00800 1 1 1 1 1 1 1 1 1 1
0.00600 1 1 1 1 1 1 1 1 1 1
0.00500 1 1 1 1 1 1 1 1 1 1
0.00400 1 1 1 1 1 1 1 1 1 1
0.00300 1 1 1 1 1 1 1 1 1 1
0.00200 1 1 1 1 1 1 1 1 1 1
0.00100 1 1 1 1 1 1 1 1 1 1
0.00030 1 1 1 1 1 1 1 1 1 1
0.00010 1 1 1 1 1 1 1 1 1 1
0.00003 1 1 1 1 1 1 1 1 1 1
0.00001 1 1 1 1 1 1 1 1 1 1
";

    #[test]
    fn test_kap_source_sha256_covers_every_table_file() {
        let dir = std::env::temp_dir()
            .join(format!("flexmesa_kap_src_sha_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("mkdir");
        fs::write(dir.join("kR_Z_fCN.data"), SAMPLE_FCN).expect("write fcn");

        let rels = kap_source_relpaths(&dir).expect("relpaths");
        assert_eq!(rels.len(), 1150);

        for (i, rel) in rels.iter().enumerate() {
            if rel != "kR_Z_fCN.data" {
                fs::write(dir.join(rel), format!("payload_{i}\n"))
                    .expect("write data");
            }
        }
        let aes = dir.join("aesopus_section.bin");
        fs::write(&aes, [0_u8; 48]).expect("write aesopus");

        let base_sha =
            compute_kap_source_sha256(&dir, &aes).expect("base sha256");
        assert_eq!(
            compute_kap_source_sha256(&dir, &aes).expect("repeat sha256"),
            base_sha
        );

        // Mutating any non-sentinel fixed, Freedman11, CO, kapCN, or AESOPUS file
        // must change the composite source digest.
        for target in [
            "a09_z0.02_x0.7.data",
            "lowT_Freedman11_z0.02.data",
            "a09_co_z0.02_x0.7.data",
            "kR_Z2E-2.data",
        ] {
            let p = dir.join(target);
            let orig = fs::read(&p).expect("read orig");
            fs::write(&p, b"mutated_table_payload\n").expect("mutate");
            let mutated_sha =
                compute_kap_source_sha256(&dir, &aes).expect("mutated sha256");
            assert_ne!(mutated_sha, base_sha, "{target} mutation missed");
            fs::write(&p, orig).expect("restore");
        }
        fs::write(&aes, [1_u8; 48]).expect("mutate aes");
        assert_ne!(
            compute_kap_source_sha256(&dir, &aes).expect("aes sha"),
            base_sha
        );

        // Deleting a non-sentinel file must fail loudly
        fs::remove_file(dir.join("OP_gs98_z0.01_x0.5.data")).expect("rm");
        assert!(matches!(
            compute_kap_source_sha256(&dir, &aes),
            Err(KapError::TableIo(_))
        ));

        let _ = fs::remove_dir_all(&dir);
    }
}
