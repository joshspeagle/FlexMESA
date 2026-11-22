// SPDX-License-Identifier: LGPL-3.0-or-later
//! Verified runtime spline cache reader/writer (`FMCENV01` wrapping
//! `FMDT4D01` / `FMCMS001`) and upstream MESA Fortran `.bin` cache reader for
//! `OPAL/SCVH`, `FreeEOS`, and `CMS` tables (`N1-11`, `N1-12`).
//!
//! Runtime caches live strictly under [`flexmesa_utils::resolve_flexmesa_cache_dir`]
//! (`~/.cache/flexmesa/derived/v1/` or `$FLEXMESA_CACHE_DIR/v1/`), never inside
//! the verified `FLEXMESA_DATA_DIR`, and carry both the source file `SHA-256`
//! and payload `SHA-256`. Corrupted cache files fail loudly rather than falling
//! back silently.

use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use flexmesa_math::{numerics_mode, NumericsMode};
use flexmesa_utils::cache_envelope::{
    read_verified_cache, write_verified_cache_atomic,
    CACHE_ENVELOPE_HEADER_BYTES, CACHE_ENVELOPE_MAGIC,
};
use flexmesa_utils::sha256::compute_sha256_file;
use flexmesa_utils::{
    is_bad, resolve_flexmesa_cache_dir, RUNTIME_CACHE_SCHEMA_VERSION,
};

use crate::cms::table::CmsXTable;
use crate::helm::table_io::resolve_flexmesa_data_dir;
use crate::tables_4d::spline::{
    validate_4d_grid_header, validate_dense_vertices, EosDtTableKind,
    EosDtVertex, EosDtXzTable, NUM_EOS_TABLE_VALS,
};
use crate::types::EosError;

/// Magic header for cached `(logQ, logT)` slices (`OPAL/SCVH` and `FreeEOS`).
pub const DT_SLICE_BIN_MAGIC: &[u8; 8] = b"FMDT4D01";

/// Magic header for cached `(logT, logRho)` `CMS` slices.
pub const CMS_SLICE_BIN_MAGIC: &[u8; 8] = b"FMCMS001";

/// Builder version stamped into `FMCENV01` headers for 4D EOS slice caches.
pub const SLICE_CACHE_BUILDER_VERSION: u32 = 1;

const VERTEX_BYTES: usize = NUM_EOS_TABLE_VALS * 4 * 8;

fn map_cache_err(context: &'static str, detail: String) -> EosError {
    EosError::TableIo { context, detail }
}

fn header_f64(bytes: &[u8], off: usize, label: &str) -> Result<f64, String> {
    let slice: [u8; 8] = bytes
        .get(off..off + 8)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| format!("{label}: truncated f64 at byte {off}"))?;
    Ok(f64::from_le_bytes(slice))
}

fn header_i32(bytes: &[u8], off: usize, label: &str) -> Result<i32, String> {
    let slice: [u8; 4] = bytes
        .get(off..off + 4)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| format!("{label}: truncated i32 at byte {off}"))?;
    Ok(i32::from_le_bytes(slice))
}

fn header_u32(bytes: &[u8], off: usize, label: &str) -> Result<usize, String> {
    let slice: [u8; 4] = bytes
        .get(off..off + 4)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| format!("{label}: truncated u32 at byte {off}"))?;
    Ok(u32::from_le_bytes(slice) as usize)
}

#[inline]
fn decode_dt_vertex(bytes: &[u8; VERTEX_BYTES]) -> EosDtVertex {
    let coeffs = std::array::from_fn(|k| {
        let s = k * 8;
        let mut buf = [0_u8; 8];
        buf.copy_from_slice(&bytes[s..s + 8]);
        f64::from_le_bytes(buf)
    });
    EosDtVertex { coeffs }
}

#[inline]
fn encode_dt_vertex(v: &EosDtVertex, out: &mut [u8; VERTEX_BYTES]) {
    for (k, &val) in v.coeffs.iter().enumerate() {
        out[k * 8..(k + 1) * 8].copy_from_slice(&val.to_le_bytes());
    }
}

/// Cache file for one slice, one source file and one numerics mode.
///
/// The fitted coefficients depend on the numerics mode: the `CMS` build takes
/// `safe_log` of `Cp` and `Cv` (`cms/table.rs`), and `flexmesa_math::log` is
/// correctly rounded in strict mode and the platform routine in fast mode.
/// A slice built in one mode must therefore never be served to the other, so
/// the mode is part of the file name.
fn slice_cache_path(
    cache_root: &Path,
    stem: &str,
    source_sha256: &str,
    mode: NumericsMode,
) -> PathBuf {
    cache_root.join("eos_4d_slices").join(format!(
        "{stem}.{}.{}.bin",
        &source_sha256[..16],
        mode.as_str()
    ))
}

#[allow(clippy::too_many_arguments)]
fn load_or_populate_slice_cache_in<T>(
    context: &'static str,
    stem: &str,
    source_path: &Path,
    payload_tag: [u8; 8],
    cache_root: Option<&Path>,
    build_from_source: impl FnOnce() -> Result<T, EosError>,
    encode_payload: impl FnOnce(&T) -> Result<Vec<u8>, String>,
    decode_payload: impl FnOnce(&[u8], &str) -> Result<T, EosError>,
) -> Result<T, EosError> {
    let (loaded, _cache_path) = load_or_populate_slice_cache_in_with_path(
        context,
        stem,
        source_path,
        payload_tag,
        cache_root,
        build_from_source,
        encode_payload,
        decode_payload,
    )?;
    Ok(loaded)
}

#[allow(clippy::too_many_arguments)]
fn load_or_populate_slice_cache_in_with_path<T>(
    context: &'static str,
    stem: &str,
    source_path: &Path,
    payload_tag: [u8; 8],
    cache_root: Option<&Path>,
    build_from_source: impl FnOnce() -> Result<T, EosError>,
    encode_payload: impl FnOnce(&T) -> Result<Vec<u8>, String>,
    decode_payload: impl FnOnce(&[u8], &str) -> Result<T, EosError>,
) -> Result<(T, Option<PathBuf>), EosError> {
    let source_sha = compute_sha256_file(source_path).map_err(|e| {
        map_cache_err(
            context,
            format!("failed to hash source {}: {e}", source_path.display()),
        )
    })?;

    if let Some(cache_root) = cache_root {
        let cache_file =
            slice_cache_path(cache_root, stem, &source_sha, numerics_mode());
        if cache_file.is_file() {
            let label = cache_file.display().to_string();
            let payload = read_verified_cache(
                &cache_file,
                SLICE_CACHE_BUILDER_VERSION,
                &payload_tag,
                &source_sha,
            )
            .map_err(|detail| map_cache_err(context, detail))?;
            let decoded = decode_payload(&payload, &label)?;
            return Ok((decoded, Some(cache_file)));
        }

        let built = build_from_source()?;
        let payload = encode_payload(&built).map_err(|detail| {
            map_cache_err(context, format!("{stem}: {detail}"))
        })?;
        write_verified_cache_atomic(
            &cache_file,
            SLICE_CACHE_BUILDER_VERSION,
            &payload_tag,
            &source_sha,
            &payload,
        )
        .map_err(|detail| map_cache_err(context, detail))?;
        return Ok((built, Some(cache_file)));
    }

    Ok((build_from_source()?, None))
}

/// Loads or builds a full `(Z, X)` `EosDtXzTable` slice, caching in
/// [`resolve_flexmesa_cache_dir`] keyed by source `SHA-256` and numerics mode.
pub fn load_or_build_dt_slice(
    mesa_data_dir: Option<&Path>,
    kind: EosDtTableKind,
    stem: &str,
) -> Result<EosDtXzTable, EosError> {
    let (tbl, _cache_path) =
        load_or_build_dt_slice_with_cache_path(mesa_data_dir, kind, stem)?;
    Ok(tbl)
}

/// Loads or builds a full `(Z, X)` `EosDtXzTable` slice via the verified cache
/// path and returns `(table, Some(cache_file_path))` when disk caching is
/// enabled.
pub(crate) fn load_or_build_dt_slice_with_cache_path(
    mesa_data_dir: Option<&Path>,
    kind: EosDtTableKind,
    stem: &str,
) -> Result<(EosDtXzTable, Option<PathBuf>), EosError> {
    let flex_dir = resolve_flexmesa_data_dir();
    let cache_root = resolve_flexmesa_cache_dir();
    load_or_build_dt_slice_with_cache_path_in(
        &flex_dir,
        mesa_data_dir,
        cache_root.as_deref(),
        kind,
        stem,
    )
}

#[doc(hidden)]
pub fn load_or_build_dt_slice_in(
    flex_dir: &Path,
    mesa_data_dir: Option<&Path>,
    cache_root: Option<&Path>,
    kind: EosDtTableKind,
    stem: &str,
) -> Result<EosDtXzTable, EosError> {
    let (tbl, _cache_path) = load_or_build_dt_slice_with_cache_path_in(
        flex_dir,
        mesa_data_dir,
        cache_root,
        kind,
        stem,
    )?;
    Ok(tbl)
}

pub(crate) fn load_or_build_dt_slice_with_cache_path_in(
    flex_dir: &Path,
    mesa_data_dir: Option<&Path>,
    cache_root: Option<&Path>,
    kind: EosDtTableKind,
    stem: &str,
) -> Result<(EosDtXzTable, Option<PathBuf>), EosError> {
    let raw_bin = flex_dir.join("eos_4d_raw").join(format!("{stem}.raw"));
    if raw_bin.is_file() {
        return load_or_populate_slice_cache_in_with_path(
            "load_or_build_dt_slice",
            stem,
            &raw_bin,
            *DT_SLICE_BIN_MAGIC,
            cache_root,
            || EosDtXzTable::from_raw_bin(&raw_bin),
            encode_dt_slice_bytes,
            decode_dt_slice_bytes,
        );
    }

    if let Some(data_dir) = mesa_data_dir {
        let subdir = match kind {
            EosDtTableKind::OpalScvh => "eosDT_data",
            EosDtTableKind::FreeEos => "eosFreeEOS_data",
        };
        let mesa_cache_1 = data_dir
            .join("eosDT_data")
            .join("cache")
            .join(format!("{stem}.bin"));
        let mesa_cache_2 = data_dir
            .join(subdir)
            .join("cache")
            .join(format!("{stem}.bin"));
        let ascii_path = data_dir.join(subdir).join(format!("{stem}.data"));

        let source_path = if mesa_cache_1.is_file() {
            mesa_cache_1
        } else if mesa_cache_2.is_file() {
            mesa_cache_2
        } else if ascii_path.is_file() {
            ascii_path
        } else {
            return Err(map_cache_err(
                "load_or_build_dt_slice",
                format!(
                    "FlexMESA 4D EOS slice '{stem}' not found in {} or {}. \
                     Run 'flexmesa data fetch' (or set FLEXMESA_DATA_DIR).",
                    flex_dir.display(),
                    data_dir.display()
                ),
            ));
        };

        return load_or_populate_slice_cache_in_with_path(
            "load_or_build_dt_slice",
            stem,
            &source_path,
            *DT_SLICE_BIN_MAGIC,
            cache_root,
            || {
                if source_path.extension().and_then(|s| s.to_str())
                    == Some("bin")
                {
                    read_mesa_eosdt_cache_bin(&source_path)
                } else {
                    EosDtXzTable::from_data_file(&source_path)
                }
            },
            encode_dt_slice_bytes,
            decode_dt_slice_bytes,
        );
    }

    Err(EosError::TableIo {
        context: "load_or_build_dt_slice",
        detail: format!(
            "FlexMESA 4D EOS slice '{stem}' not found in {}. \
             Run 'flexmesa data fetch' (or set FLEXMESA_DATA_DIR) to download \
             the verified data-v1 physics table bundle.",
            flex_dir.display()
        ),
    })
}

/// Loads or builds a full `CMS` `CmsXTable` slice, caching in
/// [`resolve_flexmesa_cache_dir`] keyed by source `SHA-256` and numerics mode.
pub fn load_or_build_cms_slice(
    mesa_data_dir: Option<&Path>,
    stem: &str,
) -> Result<CmsXTable, EosError> {
    let flex_dir = resolve_flexmesa_data_dir();
    let cache_root = resolve_flexmesa_cache_dir();
    load_or_build_cms_slice_in(
        &flex_dir,
        mesa_data_dir,
        cache_root.as_deref(),
        stem,
    )
}

#[doc(hidden)]
pub fn load_or_build_cms_slice_in(
    flex_dir: &Path,
    mesa_data_dir: Option<&Path>,
    cache_root: Option<&Path>,
    stem: &str,
) -> Result<CmsXTable, EosError> {
    let raw_bin = flex_dir.join("eos_4d_raw").join(format!("{stem}.raw"));
    if raw_bin.is_file() {
        return load_or_populate_slice_cache_in(
            "load_or_build_cms_slice",
            stem,
            &raw_bin,
            *CMS_SLICE_BIN_MAGIC,
            cache_root,
            || CmsXTable::from_raw_bin(&raw_bin),
            encode_cms_slice_bytes,
            decode_cms_slice_bytes,
        );
    }

    if let Some(data_dir) = mesa_data_dir {
        let ascii_path =
            data_dir.join("eosCMS_data").join(format!("{stem}.data"));
        return load_or_populate_slice_cache_in(
            "load_or_build_cms_slice",
            stem,
            &ascii_path,
            *CMS_SLICE_BIN_MAGIC,
            cache_root,
            || CmsXTable::from_data_file(&ascii_path),
            encode_cms_slice_bytes,
            decode_cms_slice_bytes,
        );
    }

    Err(EosError::TableIo {
        context: "load_or_build_cms_slice",
        detail: format!(
            "FlexMESA CMS EOS slice '{stem}' not found in {}. \
             Run 'flexmesa data fetch' (or set FLEXMESA_DATA_DIR) to download \
             the verified data-v1 physics table bundle.",
            flex_dir.display()
        ),
    })
}

const DT_SLICE_HEADER_BYTES: usize = 84;
const STREAMING_CHUNK_VERTICES: usize = 64;

fn parse_dt_slice_header(
    context: &'static str,
    header: &[u8],
    table_label: &str,
) -> Result<(EosDtXzTable, usize, usize), EosError> {
    let err = |msg: String| EosError::TableIo {
        context,
        detail: format!("{table_label}: {msg}"),
    };
    if header.len() < DT_SLICE_HEADER_BYTES
        || &header[0..8] != DT_SLICE_BIN_MAGIC
    {
        return Err(err(
            "invalid FMDT4D01 magic or truncated header".to_string()
        ));
    }
    let version = header_i32(header, 8, table_label).map_err(&err)?;
    let x = header_f64(header, 12, table_label).map_err(&err)?;
    let z = header_f64(header, 20, table_label).map_err(&err)?;
    let num_log_ts = header_u32(header, 28, table_label).map_err(&err)?;
    let log_t_min = header_f64(header, 32, table_label).map_err(&err)?;
    let log_t_max = header_f64(header, 40, table_label).map_err(&err)?;
    let del_log_t = header_f64(header, 48, table_label).map_err(&err)?;
    let num_log_qs = header_u32(header, 56, table_label).map_err(&err)?;
    let log_q_min = header_f64(header, 60, table_label).map_err(&err)?;
    let log_q_max = header_f64(header, 68, table_label).map_err(&err)?;
    let del_log_q = header_f64(header, 76, table_label).map_err(&err)?;

    let ngrid = validate_4d_grid_header(
        context,
        table_label,
        x,
        z,
        num_log_qs,
        log_q_min,
        log_q_max,
        del_log_q,
        num_log_ts,
        log_t_min,
        log_t_max,
        del_log_t,
    )?;
    let expected_body = ngrid
        .checked_mul(VERTEX_BYTES)
        .ok_or_else(|| err("FMDT4D01 body byte size overflow".to_string()))?;

    let stub = EosDtXzTable {
        version,
        x,
        z,
        num_log_ts,
        log_t_min,
        log_t_max,
        del_log_t,
        num_log_qs,
        log_q_min,
        log_q_max,
        del_log_q,
        dense: None,
    };
    Ok((stub, ngrid, expected_body))
}

/// Decodes a validated `FMDT4D01` payload buffer into an [`EosDtXzTable`].
pub fn decode_dt_slice_bytes(
    bytes: &[u8],
    table_label: &str,
) -> Result<EosDtXzTable, EosError> {
    let err = |msg: String| EosError::TableIo {
        context: "decode_dt_slice_bytes",
        detail: format!("{table_label}: {msg}"),
    };
    let (mut tbl, _ngrid, expected_body) =
        parse_dt_slice_header("decode_dt_slice_bytes", bytes, table_label)?;
    let body = &bytes[DT_SLICE_HEADER_BYTES..];
    if body.len() != expected_body {
        return Err(err(format!(
            "FMDT4D01 body size mismatch: expected {expected_body} bytes, got {}",
            body.len()
        )));
    }
    let dense: Vec<EosDtVertex> = body
        .as_chunks::<VERTEX_BYTES>()
        .0
        .iter()
        .map(decode_dt_vertex)
        .collect();
    validate_dense_vertices(
        "decode_dt_slice_bytes",
        table_label,
        tbl.num_log_qs,
        tbl.num_log_ts,
        &dense,
    )?;
    tbl.dense = Some(dense);
    Ok(tbl)
}

/// Reloads a previously SHA-256-verified `FMCENV01`/`FMDT4D01` cache file
/// directly into a single `Vec<EosDtVertex>` without buffering the raw bytes
/// or re-hashing the `.raw` source on eviction reload.
pub(crate) fn reload_verified_dt_slice_streaming(
    cache_file: &Path,
) -> Result<EosDtXzTable, EosError> {
    let label = cache_file.display().to_string();
    let err = |msg: String| EosError::TableIo {
        context: "reload_verified_dt_slice_streaming",
        detail: format!("{label}: {msg}"),
    };
    let file = std::fs::File::open(cache_file)
        .map_err(|e| err(format!("failed to open cache file: {e}")))?;
    let actual_len = file
        .metadata()
        .map_err(|e| err(format!("failed to stat cache file: {e}")))?
        .len();
    let mut reader = BufReader::with_capacity(256 * 1024, file);

    let mut env_hdr = [0_u8; CACHE_ENVELOPE_HEADER_BYTES];
    reader
        .read_exact(&mut env_hdr)
        .map_err(|e| err(format!("truncated FMCENV01 header: {e}")))?;
    if &env_hdr[0..8] != CACHE_ENVELOPE_MAGIC {
        return Err(err("invalid FMCENV01 magic".to_string()));
    }
    let schema_ver = u32::from_le_bytes(
        env_hdr[8..12]
            .try_into()
            .map_err(|_| err("short schema version slice".to_string()))?,
    );
    if schema_ver != RUNTIME_CACHE_SCHEMA_VERSION {
        return Err(err(format!(
            "runtime cache schema version mismatch: expected \
             {RUNTIME_CACHE_SCHEMA_VERSION}, got {schema_ver}"
        )));
    }
    let builder_ver = u32::from_le_bytes(
        env_hdr[12..16]
            .try_into()
            .map_err(|_| err("short builder version slice".to_string()))?,
    );
    if builder_ver != SLICE_CACHE_BUILDER_VERSION {
        return Err(err(format!(
            "runtime cache builder version mismatch: expected {SLICE_CACHE_BUILDER_VERSION}, got {builder_ver}"
        )));
    }
    if &env_hdr[16..24] != DT_SLICE_BIN_MAGIC {
        return Err(err("runtime cache payload tag mismatch".to_string()));
    }
    let payload_len = u64::from_le_bytes(
        env_hdr[88..96]
            .try_into()
            .map_err(|_| err("short payload len slice".to_string()))?,
    );
    let expected_total = u64::try_from(CACHE_ENVELOPE_HEADER_BYTES)
        .ok()
        .and_then(|hdr| hdr.checked_add(payload_len))
        .ok_or_else(|| err("runtime cache size overflow".to_string()))?;
    if actual_len != expected_total {
        return Err(err(format!(
            "runtime cache length mismatch: expected {expected_total} bytes, got {actual_len}"
        )));
    }

    let mut dt_hdr = [0_u8; DT_SLICE_HEADER_BYTES];
    reader
        .read_exact(&mut dt_hdr)
        .map_err(|e| err(format!("truncated FMDT4D01 header: {e}")))?;
    let (mut tbl, ngrid, expected_body) = parse_dt_slice_header(
        "reload_verified_dt_slice_streaming",
        &dt_hdr,
        &label,
    )?;
    let expected_payload = u64::try_from(DT_SLICE_HEADER_BYTES + expected_body)
        .map_err(|_| err("FMDT4D01 payload byte size overflow".to_string()))?;
    if payload_len != expected_payload {
        return Err(err(format!(
            "FMDT4D01 payload length mismatch: expected {expected_payload} bytes, got {payload_len}"
        )));
    }

    let mut dense = Vec::with_capacity(ngrid);
    let mut chunk_buf = vec![0_u8; STREAMING_CHUNK_VERTICES * VERTEX_BYTES];
    let mut remaining = ngrid;
    while remaining > 0 {
        let count = remaining.min(STREAMING_CHUNK_VERTICES);
        let chunk = &mut chunk_buf[..count * VERTEX_BYTES];
        reader
            .read_exact(chunk)
            .map_err(|e| err(format!("truncated FMDT4D01 body: {e}")))?;
        let (verts, _) = chunk.as_chunks::<VERTEX_BYTES>();
        for raw_v in verts {
            dense.push(decode_dt_vertex(raw_v));
        }
        remaining -= count;
    }

    let mut trailing = [0_u8; 1];
    if reader
        .read(&mut trailing)
        .map_err(|e| err(format!("trailing byte check failed: {e}")))?
        != 0
    {
        return Err(err("unexpected trailing bytes in cache file".to_string()));
    }

    validate_dense_vertices(
        "reload_verified_dt_slice_streaming",
        &label,
        tbl.num_log_qs,
        tbl.num_log_ts,
        &dense,
    )?;
    tbl.dense = Some(dense);
    Ok(tbl)
}

/// Reads a `FMDT4D01` binary slice file into an [`EosDtXzTable`].
pub fn read_dt_slice_bin(path: &Path) -> Result<EosDtXzTable, EosError> {
    let label = path.display().to_string();
    let bytes = std::fs::read(path).map_err(|e| EosError::TableIo {
        context: "read_dt_slice_bin",
        detail: format!("{label}: {e}"),
    })?;
    decode_dt_slice_bytes(&bytes, &label)
}

/// Encodes an [`EosDtXzTable`] as a `FMDT4D01` payload buffer.
pub fn encode_dt_slice_bytes(tbl: &EosDtXzTable) -> Result<Vec<u8>, String> {
    let dense = tbl.dense.as_ref().ok_or("missing dense vertices")?;
    let mut out = Vec::with_capacity(84 + dense.len() * VERTEX_BYTES);
    out.extend_from_slice(DT_SLICE_BIN_MAGIC);
    out.extend_from_slice(&tbl.version.to_le_bytes());
    out.extend_from_slice(&tbl.x.to_le_bytes());
    out.extend_from_slice(&tbl.z.to_le_bytes());
    out.extend_from_slice(&(tbl.num_log_ts as u32).to_le_bytes());
    out.extend_from_slice(&tbl.log_t_min.to_le_bytes());
    out.extend_from_slice(&tbl.log_t_max.to_le_bytes());
    out.extend_from_slice(&tbl.del_log_t.to_le_bytes());
    out.extend_from_slice(&(tbl.num_log_qs as u32).to_le_bytes());
    out.extend_from_slice(&tbl.log_q_min.to_le_bytes());
    out.extend_from_slice(&tbl.log_q_max.to_le_bytes());
    out.extend_from_slice(&tbl.del_log_q.to_le_bytes());
    let mut vbuf = [0u8; VERTEX_BYTES];
    for v in dense {
        encode_dt_vertex(v, &mut vbuf);
        out.extend_from_slice(&vbuf);
    }
    Ok(out)
}

/// Writes an [`EosDtXzTable`] to a `FMDT4D01` binary slice file atomically.
pub fn write_dt_slice_bin_atomic(
    path: &Path,
    tbl: &EosDtXzTable,
) -> Result<(), String> {
    let payload = encode_dt_slice_bytes(tbl)?;
    write_atomic_raw(path, &payload)
}

/// Reads upstream MESA's unformatted Fortran cache (`Read_EoS_Cache` in `eosdt_load_tables.f90`).
pub fn read_mesa_eosdt_cache_bin(
    path: &Path,
) -> Result<EosDtXzTable, EosError> {
    let table_label = path.display().to_string();
    let err = |msg: String| EosError::TableIo {
        context: "read_mesa_eosdt_cache_bin",
        detail: format!("{table_label}: {msg}"),
    };
    let file = std::fs::File::open(path).map_err(|e| err(e.to_string()))?;
    let mut reader = BufReader::with_capacity(512 * 1024, file);
    let mut r1_hdr = [0u8; 84];
    reader
        .read_exact(&mut r1_hdr)
        .map_err(|e| err(e.to_string()))?;
    let r1_len = header_i32(&r1_hdr, 0, &table_label).map_err(&err)? as u32;
    let r1_tail = header_i32(&r1_hdr, 80, &table_label).map_err(&err)? as u32;
    if r1_len != 76 || r1_tail != 76 {
        return Err(err(format!("unexpected MESA header record len {r1_len}")));
    }
    let x = header_f64(&r1_hdr, 4, &table_label).map_err(&err)?;
    let z = header_f64(&r1_hdr, 12, &table_label).map_err(&err)?;
    let num_log_ts_i = header_i32(&r1_hdr, 20, &table_label).map_err(&err)?;
    let log_t_min = header_f64(&r1_hdr, 24, &table_label).map_err(&err)?;
    let log_t_max = header_f64(&r1_hdr, 32, &table_label).map_err(&err)?;
    let del_log_t = header_f64(&r1_hdr, 40, &table_label).map_err(&err)?;
    let num_log_qs_i = header_i32(&r1_hdr, 48, &table_label).map_err(&err)?;
    let log_q_min = header_f64(&r1_hdr, 52, &table_label).map_err(&err)?;
    let log_q_max = header_f64(&r1_hdr, 60, &table_label).map_err(&err)?;
    let del_log_q = header_f64(&r1_hdr, 68, &table_label).map_err(&err)?;
    let version = header_i32(&r1_hdr, 76, &table_label).map_err(&err)?;

    let num_log_ts = usize::try_from(num_log_ts_i)
        .map_err(|_| err(format!("negative num_log_ts={num_log_ts_i}")))?;
    let num_log_qs = usize::try_from(num_log_qs_i)
        .map_err(|_| err(format!("negative num_log_qs={num_log_qs_i}")))?;
    let ngrid = validate_4d_grid_header(
        "read_mesa_eosdt_cache_bin",
        &table_label,
        x,
        z,
        num_log_qs,
        log_q_min,
        log_q_max,
        del_log_q,
        num_log_ts,
        log_t_min,
        log_t_max,
        del_log_t,
    )?;

    let mut r2_marker = [0u8; 4];
    reader
        .read_exact(&mut r2_marker)
        .map_err(|e| err(e.to_string()))?;
    let r2_len = u32::from_le_bytes(r2_marker) as usize;
    let nv_mesa = 26usize;
    let expected_r2 = 4usize
        .checked_mul(nv_mesa)
        .and_then(|v| v.checked_mul(ngrid))
        .and_then(|v| v.checked_mul(8))
        .ok_or_else(|| err("MESA cache record size overflow".to_string()))?;
    if r2_len != expected_r2 {
        return Err(err(format!("unexpected MESA tbl1 record len {r2_len}")));
    }

    let mut dense = vec![EosDtVertex::default(); ngrid];
    let mut cell_buf = [0u8; 4 * 26 * 8];
    for jt in 0..num_log_ts {
        for iq in 0..num_log_qs {
            reader
                .read_exact(&mut cell_buf)
                .map_err(|e| err(e.to_string()))?;
            let dst = &mut dense[iq * num_log_ts + jt].coeffs;
            for (k, slot) in dst.iter_mut().enumerate() {
                let s = k * 8;
                let mut fb = [0_u8; 8];
                fb.copy_from_slice(&cell_buf[s..s + 8]);
                let val = f64::from_le_bytes(fb);
                if is_bad(val) {
                    return Err(err(format!(
                        "non-finite spline entry ({val}) at cell=({iq}, {jt}) slot={k}"
                    )));
                }
                *slot = val;
            }
        }
    }
    Ok(EosDtXzTable {
        version,
        x,
        z,
        num_log_ts,
        log_t_min,
        log_t_max,
        del_log_t,
        num_log_qs,
        log_q_min,
        log_q_max,
        del_log_q,
        dense: Some(dense),
    })
}

/// Decodes a validated `FMCMS001` payload buffer into a [`CmsXTable`].
pub fn decode_cms_slice_bytes(
    bytes: &[u8],
    table_label: &str,
) -> Result<CmsXTable, EosError> {
    let err = |msg: String| EosError::TableIo {
        context: "decode_cms_slice_bytes",
        detail: format!("{table_label}: {msg}"),
    };
    if bytes.len() < 76 || &bytes[0..8] != CMS_SLICE_BIN_MAGIC {
        return Err(err(
            "invalid FMCMS001 magic or truncated header".to_string()
        ));
    }
    let version = header_i32(bytes, 8, table_label).map_err(&err)?;
    let x = header_f64(bytes, 12, table_label).map_err(&err)?;
    let num_log_ts = header_u32(bytes, 20, table_label).map_err(&err)?;
    let log_t_min = header_f64(bytes, 24, table_label).map_err(&err)?;
    let log_t_max = header_f64(bytes, 32, table_label).map_err(&err)?;
    let delta_log_t = header_f64(bytes, 40, table_label).map_err(&err)?;
    let num_log_rhos = header_u32(bytes, 48, table_label).map_err(&err)?;
    let log_rho_min = header_f64(bytes, 52, table_label).map_err(&err)?;
    let log_rho_max = header_f64(bytes, 60, table_label).map_err(&err)?;
    let delta_log_rho = header_f64(bytes, 68, table_label).map_err(&err)?;

    let ngrid = validate_4d_grid_header(
        "decode_cms_slice_bytes",
        table_label,
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
    let expected_body = ngrid
        .checked_mul(VERTEX_BYTES)
        .ok_or_else(|| err("FMCMS001 body byte size overflow".to_string()))?;
    let body = &bytes[76..];
    if body.len() != expected_body {
        return Err(err(format!(
            "FMCMS001 body size mismatch: expected {expected_body} bytes, got {}",
            body.len()
        )));
    }
    let dense: Vec<EosDtVertex> = body
        .as_chunks::<VERTEX_BYTES>()
        .0
        .iter()
        .map(decode_dt_vertex)
        .collect();
    validate_dense_vertices(
        "decode_cms_slice_bytes",
        table_label,
        num_log_ts,
        num_log_rhos,
        &dense,
    )?;
    Ok(CmsXTable {
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

/// Reads a `FMCMS001` binary slice file into a [`CmsXTable`].
pub fn read_cms_slice_bin(path: &Path) -> Result<CmsXTable, EosError> {
    let label = path.display().to_string();
    let bytes = std::fs::read(path).map_err(|e| EosError::TableIo {
        context: "read_cms_slice_bin",
        detail: format!("{label}: {e}"),
    })?;
    decode_cms_slice_bytes(&bytes, &label)
}

/// Encodes a [`CmsXTable`] as a `FMCMS001` payload buffer.
pub fn encode_cms_slice_bytes(tbl: &CmsXTable) -> Result<Vec<u8>, String> {
    let dense = tbl.dense.as_ref().ok_or("missing dense CMS vertices")?;
    let mut out = Vec::with_capacity(76 + dense.len() * VERTEX_BYTES);
    out.extend_from_slice(CMS_SLICE_BIN_MAGIC);
    out.extend_from_slice(&tbl.version.to_le_bytes());
    out.extend_from_slice(&tbl.x.to_le_bytes());
    out.extend_from_slice(&(tbl.num_log_ts as u32).to_le_bytes());
    out.extend_from_slice(&tbl.log_t_min.to_le_bytes());
    out.extend_from_slice(&tbl.log_t_max.to_le_bytes());
    out.extend_from_slice(&tbl.delta_log_t.to_le_bytes());
    out.extend_from_slice(&(tbl.num_log_rhos as u32).to_le_bytes());
    out.extend_from_slice(&tbl.log_rho_min.to_le_bytes());
    out.extend_from_slice(&tbl.log_rho_max.to_le_bytes());
    out.extend_from_slice(&tbl.delta_log_rho.to_le_bytes());
    let mut vbuf = [0u8; VERTEX_BYTES];
    for v in dense {
        encode_dt_vertex(v, &mut vbuf);
        out.extend_from_slice(&vbuf);
    }
    Ok(out)
}

/// Writes a [`CmsXTable`] to a `FMCMS001` binary slice file atomically.
pub fn write_cms_slice_bin_atomic(
    path: &Path,
    tbl: &CmsXTable,
) -> Result<(), String> {
    let payload = encode_cms_slice_bytes(tbl)?;
    write_atomic_raw(path, &payload)
}

fn write_atomic_raw(path: &Path, payload: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let tmp = path.with_extension(format!("bin.tmp.{}", std::process::id()));
    let write_res = std::fs::write(&tmp, payload).map_err(|e| e.to_string());
    if let Err(err) = write_res {
        let _ = std::fs::remove_file(&tmp);
        return Err(err);
    }
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        e.to_string()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cms::table::CMS_RAW_BIN_MAGIC;
    use crate::tables_4d::table_io::DT_RAW_BIN_MAGIC;

    fn make_temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("flexmesa_eos_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn build_synthetic_dt_raw_bytes(inject_nan: bool) -> Vec<u8> {
        let n_q = 4_usize;
        let n_t = 4_usize;
        let mut out = Vec::with_capacity(84 + n_q * n_t * 16 * 8);
        out.extend_from_slice(DT_RAW_BIN_MAGIC);
        out.extend_from_slice(&51_i32.to_le_bytes());
        out.extend_from_slice(&0.70_f64.to_le_bytes());
        out.extend_from_slice(&0.02_f64.to_le_bytes());
        out.extend_from_slice(&(n_t as u32).to_le_bytes());
        out.extend_from_slice(&4.0_f64.to_le_bytes());
        out.extend_from_slice(&4.3_f64.to_le_bytes());
        out.extend_from_slice(&0.1_f64.to_le_bytes());
        out.extend_from_slice(&(n_q as u32).to_le_bytes());
        out.extend_from_slice(&(-5.0_f64).to_le_bytes());
        out.extend_from_slice(&(-4.4_f64).to_le_bytes());
        out.extend_from_slice(&0.2_f64.to_le_bytes());
        for iq in 0..n_q {
            for jt in 0..n_t {
                for k in 0..16 {
                    let val = if inject_nan && iq == 2 && jt == 1 && k == 3 {
                        f64::NAN
                    } else {
                        10.0 + 0.05 * (iq as f64)
                            + 0.1 * (jt as f64)
                            + 0.01 * (k as f64)
                    };
                    out.extend_from_slice(&val.to_le_bytes());
                }
            }
        }
        out
    }

    fn build_synthetic_cms_raw_bytes(inject_inf: bool) -> Vec<u8> {
        let n_t = 4_usize;
        let n_rho = 4_usize;
        let mut out = Vec::with_capacity(76 + n_t * n_rho * 16 * 8);
        out.extend_from_slice(CMS_RAW_BIN_MAGIC);
        out.extend_from_slice(&1_i32.to_le_bytes());
        out.extend_from_slice(&0.70_f64.to_le_bytes());
        out.extend_from_slice(&(n_t as u32).to_le_bytes());
        out.extend_from_slice(&3.0_f64.to_le_bytes());
        out.extend_from_slice(&3.3_f64.to_le_bytes());
        out.extend_from_slice(&0.1_f64.to_le_bytes());
        out.extend_from_slice(&(n_rho as u32).to_le_bytes());
        out.extend_from_slice(&(-4.0_f64).to_le_bytes());
        out.extend_from_slice(&(-3.7_f64).to_le_bytes());
        out.extend_from_slice(&0.1_f64.to_le_bytes());
        for it in 0..n_t {
            for jrho in 0..n_rho {
                for k in 0..16 {
                    let val = if inject_inf && it == 1 && jrho == 2 && k == 4 {
                        f64::INFINITY
                    } else {
                        5.0 + 0.03 * (it as f64)
                            + 0.07 * (jrho as f64)
                            + 0.01 * (k as f64)
                    };
                    out.extend_from_slice(&val.to_le_bytes());
                }
            }
        }
        out
    }

    #[test]
    fn test_raw_table_non_finite_entries_return_named_error() {
        let dir = make_temp_dir("nonfinite");
        let dt_bad = dir.join("bad_dt.raw");
        std::fs::write(&dt_bad, build_synthetic_dt_raw_bytes(true))
            .expect("write bad dt raw");
        let err = EosDtXzTable::from_raw_bin(&dt_bad)
            .expect_err("NaN raw input must fail");
        let msg = err.to_string();
        assert!(msg.contains("bad_dt.raw"), "unexpected err: {msg}");
        assert!(msg.contains("(2, 1)"), "missing cell in err: {msg}");

        let cms_bad = dir.join("bad_cms.raw");
        std::fs::write(&cms_bad, build_synthetic_cms_raw_bytes(true))
            .expect("write bad cms raw");
        let err_cms = CmsXTable::from_raw_bin(&cms_bad)
            .expect_err("Inf CMS raw input must fail");
        let msg_cms = err_cms.to_string();
        assert!(msg_cms.contains("bad_cms.raw"), "unexpected err: {msg_cms}");
        assert!(msg_cms.contains("(1, 2)"), "missing cell in err: {msg_cms}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_slice_cache_path_separates_numerics_modes() {
        let root = Path::new("cache_root");
        let sha = "0123456789abcdef0123456789abcdef";
        let strict =
            slice_cache_path(root, "mesa-CMS_000x", sha, NumericsMode::Strict);
        let fast =
            slice_cache_path(root, "mesa-CMS_000x", sha, NumericsMode::Fast);
        assert_ne!(strict, fast);
        assert_eq!(
            strict,
            root.join("eos_4d_slices")
                .join("mesa-CMS_000x.0123456789abcdef.strict.bin")
        );
        assert_eq!(
            fast,
            root.join("eos_4d_slices")
                .join("mesa-CMS_000x.0123456789abcdef.fast.bin")
        );
    }

    #[test]
    fn test_slice_cache_isolates_from_data_dir_and_detects_corruption() {
        let dir = make_temp_dir("cache_iso");
        let data_dir = dir.join("data");
        let cache_dir = dir.join("cache");
        let cache_v1 = cache_dir.join("v1");
        let raw_dir = data_dir.join("eos_4d_raw");
        std::fs::create_dir_all(&raw_dir).expect("mkdir raw");

        let stem = "mesa-eosDT_synthetic";
        let raw_path = raw_dir.join(format!("{stem}.raw"));
        std::fs::write(&raw_path, build_synthetic_dt_raw_bytes(false))
            .expect("write raw");
        let cms_stem = "mesa-CMS_synthetic";
        let cms_raw_path = raw_dir.join(format!("{cms_stem}.raw"));
        std::fs::write(&cms_raw_path, build_synthetic_cms_raw_bytes(false))
            .expect("write cms raw");

        // 1. Disabled mode (cache_root = None) builds cleanly with zero cache files written
        let _ = load_or_populate_slice_cache_in(
            "test_slice_cache",
            stem,
            &raw_path,
            *DT_SLICE_BIN_MAGIC,
            None,
            || EosDtXzTable::from_raw_bin(&raw_path),
            encode_dt_slice_bytes,
            decode_dt_slice_bytes,
        )
        .expect("load dt with cache disabled");
        assert!(!cache_dir.exists());
        assert!(!data_dir.join("eos_4d_slices").exists());

        // 2. Enabled mode populates v1/eos_4d_slices/*.bin without touching data_dir
        let dt1 = load_or_populate_slice_cache_in(
            "test_slice_cache",
            stem,
            &raw_path,
            *DT_SLICE_BIN_MAGIC,
            Some(&cache_v1),
            || EosDtXzTable::from_raw_bin(&raw_path),
            encode_dt_slice_bytes,
            decode_dt_slice_bytes,
        )
        .expect("populate dt cache");
        let cms1 = load_or_populate_slice_cache_in(
            "test_slice_cache",
            cms_stem,
            &cms_raw_path,
            *CMS_SLICE_BIN_MAGIC,
            Some(&cache_v1),
            || CmsXTable::from_raw_bin(&cms_raw_path),
            encode_cms_slice_bytes,
            decode_cms_slice_bytes,
        )
        .expect("populate cms cache");
        assert!(!data_dir.join("eos_4d_slices").exists());
        let cached_slices_dir = cache_v1.join("eos_4d_slices");
        let cached_files: Vec<PathBuf> = std::fs::read_dir(&cached_slices_dir)
            .expect("read cache dir")
            .filter_map(|e| e.ok().map(|x| x.path()))
            .collect();
        assert_eq!(cached_files.len(), 2);

        // Reload from verified cache
        let dt2 = load_or_populate_slice_cache_in(
            "test_slice_cache",
            stem,
            &raw_path,
            *DT_SLICE_BIN_MAGIC,
            Some(&cache_v1),
            || EosDtXzTable::from_raw_bin(&raw_path),
            encode_dt_slice_bytes,
            decode_dt_slice_bytes,
        )
        .expect("reload dt cache");
        assert_eq!(dt1.dense, dt2.dense);
        let cms2 = load_or_populate_slice_cache_in(
            "test_slice_cache",
            cms_stem,
            &cms_raw_path,
            *CMS_SLICE_BIN_MAGIC,
            Some(&cache_v1),
            || CmsXTable::from_raw_bin(&cms_raw_path),
            encode_cms_slice_bytes,
            decode_cms_slice_bytes,
        )
        .expect("reload cms cache");
        assert_eq!(cms1.dense, cms2.dense);

        // 3. Corrupting 1 byte in cached file must fail loudly (not fall back)
        let dt_cache = cached_files
            .iter()
            .find(|p| {
                p.file_name()
                    .is_some_and(|n| n.to_string_lossy().contains("eosDT"))
            })
            .expect("find dt cache file");
        let mut bytes = std::fs::read(dt_cache).expect("read cached dt");
        let tail = bytes.len() - 1;
        bytes[tail] ^= 0x01;
        std::fs::write(dt_cache, &bytes).expect("write corrupt cache");
        let err = load_or_populate_slice_cache_in(
            "test_slice_cache",
            stem,
            &raw_path,
            *DT_SLICE_BIN_MAGIC,
            Some(&cache_v1),
            || EosDtXzTable::from_raw_bin(&raw_path),
            encode_dt_slice_bytes,
            decode_dt_slice_bytes,
        )
        .expect_err("corrupt cache must fail loudly");
        assert!(
            err.to_string().contains("SHA-256 mismatch"),
            "unexpected err: {err}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_reload_verified_dt_slice_streaming_matches_full_decode() {
        let dir = make_temp_dir("streaming_reload");
        let data_dir = dir.join("data");
        let cache_v1 = dir.join("cache").join("v1");
        let raw_dir = data_dir.join("eos_4d_raw");
        std::fs::create_dir_all(&raw_dir).expect("mkdir raw");

        let stem = "mesa-eosDT_stream_test";
        let raw_path = raw_dir.join(format!("{stem}.raw"));
        std::fs::write(&raw_path, build_synthetic_dt_raw_bytes(false))
            .expect("write raw");

        let (full_tbl, maybe_cache) =
            load_or_build_dt_slice_with_cache_path_in(
                &data_dir,
                None,
                Some(&cache_v1),
                EosDtTableKind::OpalScvh,
                stem,
            )
            .expect("build and cache dt slice");
        let cache_file = maybe_cache.expect("expected cache path");
        assert!(cache_file.is_file());

        let streamed = reload_verified_dt_slice_streaming(&cache_file)
            .expect("streaming reload");
        assert_eq!(streamed.version, full_tbl.version);
        assert_eq!(streamed.x, full_tbl.x);
        assert_eq!(streamed.z, full_tbl.z);
        assert_eq!(streamed.num_log_ts, full_tbl.num_log_ts);
        assert_eq!(streamed.num_log_qs, full_tbl.num_log_qs);
        assert_eq!(streamed.dense, full_tbl.dense);

        let orig = std::fs::read(&cache_file).expect("read cache bytes");
        // Bad magic rejected.
        let mut bad_magic = orig.clone();
        bad_magic[0] = b'X';
        std::fs::write(&cache_file, &bad_magic).expect("write bad magic");
        let err = reload_verified_dt_slice_streaming(&cache_file)
            .expect_err("bad magic must fail");
        assert!(err.to_string().contains("magic"), "unexpected err: {err}");

        // Truncated file rejected.
        std::fs::write(&cache_file, &orig[..orig.len() - 16])
            .expect("write truncated");
        let err = reload_verified_dt_slice_streaming(&cache_file)
            .expect_err("truncated cache must fail");
        assert!(
            err.to_string().contains("length mismatch"),
            "unexpected err: {err}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
