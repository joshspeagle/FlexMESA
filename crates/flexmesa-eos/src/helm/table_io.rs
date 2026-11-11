// SPDX-License-Identifier: LGPL-3.0-or-later
//! Full `HELM` table parsers, binary cache I/O, and shared MESA data-directory
//! resolution (`eos/private/helm_alloc.f90`).

use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::helm::table::{HelmVertex, HELM_IMAX, HELM_JMAX};

/// Magic header for the full dense `HELM` binary cache (`2701 x 1001 x 21` `f64`).
pub(crate) const HELM_FULL_BIN_MAGIC: &[u8; 8] = b"FMHELM01";

/// Number of `f64` values stored per [`HelmVertex`] (`9 + 4 + 4 + 4 = 21`).
const FLOATS_PER_VERTEX: usize = 21;

/// Byte size of one serialized [`HelmVertex`] (`21 * 8 = 168` bytes).
const BYTES_PER_VERTEX: usize = FLOATS_PER_VERTEX * 8;

pub use flexmesa_utils::resolve_flexmesa_data_dir;

/// Resolves an optional upstream MESA `data` directory containing raw ASCII `eosDT_data/`.
#[must_use]
pub(crate) fn resolve_mesa_data_dir() -> Option<PathBuf> {
    for var in ["FLEXMESA_EOS_DATA_DIR", "FLEXMESA_DATA_DIR", "MESA_DIR"] {
        if let Some(val) = std::env::var_os(var) {
            if val.is_empty() {
                continue;
            }
            let p = PathBuf::from(val);
            if p.join("eosDT_data").is_dir() {
                return Some(p);
            }
            let sub = p.join("data");
            if sub.join("eosDT_data").is_dir() {
                return Some(sub);
            }
        }
    }
    None
}

/// Loads or builds the full dense `2701 x 1001` `HELM` vertex grid (`FMHELM01`).
///
/// # Errors
/// Returns an actionable error when neither `helm_table_full.bin` in the
/// FlexMESA data directory nor upstream `MESA_DIR/data/eosDT_data` is present.
pub(crate) fn load_or_build_full_helm_vertices() -> Result<Vec<HelmVertex>, String>
{
    let data_root = resolve_flexmesa_data_dir();
    let full_bin = data_root.join("helm_table_full.bin");
    if full_bin.is_file() {
        return read_helm_full_bin(&full_bin);
    }

    if let Some(mesa_data) = resolve_mesa_data_dir() {
        let eosdt_dir = mesa_data.join("eosDT_data");
        let mesa_cache = eosdt_dir.join("cache").join("helm_table.bin");
        let ascii_dat = eosdt_dir.join("helm_table.dat");
        let parsed = if mesa_cache.is_file() {
            read_mesa_helm_cache_bin(&mesa_cache).or_else(|_| {
                let file = std::fs::File::open(&ascii_dat)
                    .map_err(|e| e.to_string())?;
                parse_helm_ascii_reader(BufReader::with_capacity(
                    1024 * 1024,
                    file,
                ))
            })
        } else if ascii_dat.is_file() {
            let file =
                std::fs::File::open(&ascii_dat).map_err(|e| e.to_string())?;
            parse_helm_ascii_reader(BufReader::with_capacity(1024 * 1024, file))
        } else {
            Err("eosDT_data missing helm_table.dat".to_string())
        };
        if let Ok(vertices) = parsed {
            let _ = write_helm_full_bin_atomic(&full_bin, &vertices);
            return Ok(vertices);
        }
    }

    Err(format!(
        "FlexMESA Helmholtz EOS table not found at {}. \
         Run 'flexmesa data fetch' (or set FLEXMESA_DATA_DIR) to download \
         the verified data-v1 physics table bundle.",
        full_bin.display()
    ))
}

#[inline]
fn decode_vertex(bytes: &[u8]) -> HelmVertex {
    let f64_at = |k: usize| -> f64 {
        let s = k * 8;
        f64::from_le_bytes(bytes[s..s + 8].try_into().unwrap_or([0; 8]))
    };
    HelmVertex {
        f: std::array::from_fn(f64_at),
        dpdf: std::array::from_fn(|k| f64_at(9 + k)),
        ef: std::array::from_fn(|k| f64_at(13 + k)),
        xf: std::array::from_fn(|k| f64_at(17 + k)),
    }
}

#[inline]
fn encode_vertex(v: &HelmVertex, out: &mut [u8; BYTES_PER_VERTEX]) {
    for (k, &val) in v.f.iter().enumerate() {
        out[k * 8..(k + 1) * 8].copy_from_slice(&val.to_le_bytes());
    }
    for (k, &val) in v.dpdf.iter().enumerate() {
        let idx = 9 + k;
        out[idx * 8..(idx + 1) * 8].copy_from_slice(&val.to_le_bytes());
    }
    for (k, &val) in v.ef.iter().enumerate() {
        let idx = 13 + k;
        out[idx * 8..(idx + 1) * 8].copy_from_slice(&val.to_le_bytes());
    }
    for (k, &val) in v.xf.iter().enumerate() {
        let idx = 17 + k;
        out[idx * 8..(idx + 1) * 8].copy_from_slice(&val.to_le_bytes());
    }
}

/// Reads the raw binary cache `helm_table_full.bin` (`FMHELM01` format).
pub fn read_helm_full_bin(path: &Path) -> Result<Vec<HelmVertex>, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut hdr = [0u8; 16];
    reader.read_exact(&mut hdr).map_err(|e| e.to_string())?;
    if &hdr[0..8] != HELM_FULL_BIN_MAGIC {
        return Err("invalid helm_table_full.bin magic header".to_string());
    }
    let imax =
        u32::from_le_bytes(hdr[8..12].try_into().unwrap_or([0; 4])) as usize;
    let jmax =
        u32::from_le_bytes(hdr[12..16].try_into().unwrap_or([0; 4])) as usize;
    if imax != HELM_IMAX || jmax != HELM_JMAX {
        return Err(format!("unexpected HELM dimensions ({imax}, {jmax})"));
    }
    let n_total = imax * jmax;
    let mut vertices = Vec::with_capacity(n_total);
    let mut buf = [0u8; BYTES_PER_VERTEX];
    for _ in 0..n_total {
        reader.read_exact(&mut buf).map_err(|e| e.to_string())?;
        vertices.push(decode_vertex(&buf));
    }
    Ok(vertices)
}

/// Atomically writes `helm_table_full.bin` via a process-unique temporary file.
pub fn write_helm_full_bin_atomic(
    path: &Path,
    vertices: &[HelmVertex],
) -> Result<(), String> {
    if vertices.len() != HELM_IMAX * HELM_JMAX {
        return Err(format!("expected {} vertices", HELM_IMAX * HELM_JMAX));
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let tmp_path =
        path.with_extension(format!("bin.tmp.{}", std::process::id()));
    {
        let file =
            std::fs::File::create(&tmp_path).map_err(|e| e.to_string())?;
        let mut writer = BufWriter::with_capacity(1024 * 1024, file);
        writer
            .write_all(HELM_FULL_BIN_MAGIC)
            .map_err(|e| e.to_string())?;
        writer
            .write_all(&(HELM_IMAX as u32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        writer
            .write_all(&(HELM_JMAX as u32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        let mut v_bytes = [0u8; BYTES_PER_VERTEX];
        for v in vertices {
            encode_vertex(v, &mut v_bytes);
            writer.write_all(&v_bytes).map_err(|e| e.to_string())?;
        }
        writer.flush().map_err(|e| e.to_string())?;
    }
    std::fs::rename(&tmp_path, path).map_err(|e| e.to_string())
}

/// Reads upstream MESA's unformatted Fortran binary cache (`eosDT_data/cache/helm_table.bin`).
pub fn read_mesa_helm_cache_bin(
    path: &Path,
) -> Result<Vec<HelmVertex>, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut hdr = [0u8; 24];
    reader.read_exact(&mut hdr).map_err(|e| e.to_string())?;
    let imax =
        i32::from_le_bytes(hdr[4..8].try_into().unwrap_or([0; 4])) as usize;
    let jmax =
        i32::from_le_bytes(hdr[16..20].try_into().unwrap_or([0; 4])) as usize;
    if imax != HELM_IMAX || jmax != HELM_JMAX {
        return Err(format!(
            "unexpected MESA helm_table.bin dims ({imax}, {jmax})"
        ));
    }
    let n_total = imax * jmax;
    let expected_rec_bytes = (n_total * 8) as u32;
    let mut vertices = vec![HelmVertex::default(); n_total];
    let mut marker = [0u8; 4];
    let mut f_buf = [0u8; 8];

    for col in 0..FLOATS_PER_VERTEX {
        reader.read_exact(&mut marker).map_err(|e| e.to_string())?;
        let rec_len = u32::from_le_bytes(marker);
        if rec_len != expected_rec_bytes {
            return Err(format!("bad Fortran record length {rec_len}"));
        }
        for v in &mut vertices {
            reader.read_exact(&mut f_buf).map_err(|e| e.to_string())?;
            let val = f64::from_le_bytes(f_buf);
            if col < 9 {
                v.f[col] = val;
            } else if col < 13 {
                v.dpdf[col - 9] = val;
            } else if col < 17 {
                v.ef[col - 13] = val;
            } else {
                v.xf[col - 17] = val;
            }
        }
        reader.read_exact(&mut marker).map_err(|e| e.to_string())?;
    }
    Ok(vertices)
}

/// Parses `helm_table.dat` (`10,814,804` lines) into the full `2701 x 1001` vertex grid.
pub fn parse_helm_ascii_reader<R: BufRead>(
    mut reader: R,
) -> Result<Vec<HelmVertex>, String> {
    let n_total = HELM_IMAX * HELM_JMAX;
    let mut dense = vec![HelmVertex::default(); n_total];
    let mut line = String::with_capacity(256);

    let mut read_row = |expected_cols: usize,
                        out: &mut [f64]|
     -> Result<(), String> {
        loop {
            line.clear();
            let n = reader.read_line(&mut line).map_err(|e| e.to_string())?;
            if n == 0 {
                return Err("unexpected EOF in helm_table.dat".to_string());
            }
            if !line.trim().is_empty() {
                break;
            }
        }
        let mut iter = line.split_whitespace();
        for slot in out.iter_mut().take(expected_cols) {
            let tok = iter.next().ok_or_else(|| {
                format!("expected {expected_cols} columns in helm_table.dat")
            })?;
            *slot = tok
                .parse::<f64>()
                .map_err(|e| format!("invalid float '{tok}': {e}"))?;
        }
        Ok(())
    };

    for v in &mut dense {
        read_row(9, &mut v.f)?;
    }
    for v in &mut dense {
        read_row(4, &mut v.dpdf)?;
    }
    for v in &mut dense {
        read_row(4, &mut v.ef)?;
    }
    for v in &mut dense {
        read_row(4, &mut v.xf)?;
    }
    Ok(dense)
}
