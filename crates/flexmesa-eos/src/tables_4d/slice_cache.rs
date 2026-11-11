// SPDX-License-Identifier: LGPL-3.0-or-later
//! Binary slice cache reader/writer (`eos_4d_slices/*.bin`) and upstream MESA
//! Fortran `.bin` cache reader for `OPAL/SCVH`, `FreeEOS`, and `CMS` tables.

use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use flexmesa_utils::is_bad;

use crate::cms::table::CmsXTable;
use crate::helm::table_io::resolve_flexmesa_data_dir;
use crate::tables_4d::spline::{
    EosDtTableKind, EosDtVertex, EosDtXzTable, NUM_EOS_TABLE_VALS,
};
use crate::types::EosError;

/// Magic header for cached `(logQ, logT)` slices (`OPAL/SCVH` and `FreeEOS`).
pub const DT_SLICE_BIN_MAGIC: &[u8; 8] = b"FMDT4D01";

/// Magic header for cached `(logT, logRho)` `CMS` slices.
pub const CMS_SLICE_BIN_MAGIC: &[u8; 8] = b"FMCMS001";

const VERTEX_BYTES: usize = NUM_EOS_TABLE_VALS * 4 * 8;

#[inline]
fn decode_dt_vertex(bytes: &[u8; VERTEX_BYTES]) -> EosDtVertex {
    let coeffs = std::array::from_fn(|k| {
        let s = k * 8;
        f64::from_le_bytes(bytes[s..s + 8].try_into().unwrap_or([0; 8]))
    });
    EosDtVertex { coeffs }
}

#[inline]
fn encode_dt_vertex(v: &EosDtVertex, out: &mut [u8; VERTEX_BYTES]) {
    for (k, &val) in v.coeffs.iter().enumerate() {
        out[k * 8..(k + 1) * 8].copy_from_slice(&val.to_le_bytes());
    }
}

/// Loads or builds a full `(Z, X)` `EosDtXzTable` slice, caching it in `eos_4d_slices/`.
pub fn load_or_build_dt_slice(
    mesa_data_dir: Option<&Path>,
    kind: EosDtTableKind,
    stem: &str,
) -> Result<EosDtXzTable, EosError> {
    let flex_dir = resolve_flexmesa_data_dir();
    let slice_bin = flex_dir.join("eos_4d_slices").join(format!("{stem}.bin"));
    if slice_bin.is_file() {
        if let Ok(tbl) = read_dt_slice_bin(&slice_bin) {
            return Ok(tbl);
        }
    }

    let raw_bin = flex_dir.join("eos_4d_raw").join(format!("{stem}.raw"));
    if raw_bin.is_file() {
        let tbl = EosDtXzTable::from_raw_bin(&raw_bin)?;
        let _ = write_dt_slice_bin_atomic(&slice_bin, &tbl);
        return Ok(tbl);
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

        let tbl = if mesa_cache_1.is_file() {
            read_mesa_eosdt_cache_bin(&mesa_cache_1)
                .or_else(|_| EosDtXzTable::from_data_file(&ascii_path))?
        } else if mesa_cache_2.is_file() {
            read_mesa_eosdt_cache_bin(&mesa_cache_2)
                .or_else(|_| EosDtXzTable::from_data_file(&ascii_path))?
        } else {
            EosDtXzTable::from_data_file(&ascii_path)?
        };
        let _ = write_dt_slice_bin_atomic(&slice_bin, &tbl);
        return Ok(tbl);
    }

    Err(EosError::InvalidState {
        context: "load_or_build_dt_slice",
        detail: format!(
            "FlexMESA 4D EOS slice '{stem}' not found in {}. \
             Run 'flexmesa data fetch' (or set FLEXMESA_DATA_DIR) to download \
             the verified data-v1 physics table bundle.",
            flex_dir.display()
        ),
    })
}

/// Loads or builds a full `CMS` `CmsXTable` slice, caching it in `eos_4d_slices/`.
pub fn load_or_build_cms_slice(
    mesa_data_dir: Option<&Path>,
    stem: &str,
) -> Result<CmsXTable, EosError> {
    let flex_dir = resolve_flexmesa_data_dir();
    let slice_bin = flex_dir.join("eos_4d_slices").join(format!("{stem}.bin"));
    if slice_bin.is_file() {
        if let Ok(tbl) = read_cms_slice_bin(&slice_bin) {
            return Ok(tbl);
        }
    }

    let raw_bin = flex_dir.join("eos_4d_raw").join(format!("{stem}.raw"));
    if raw_bin.is_file() {
        let tbl = CmsXTable::from_raw_bin(&raw_bin)?;
        let _ = write_cms_slice_bin_atomic(&slice_bin, &tbl);
        return Ok(tbl);
    }

    if let Some(data_dir) = mesa_data_dir {
        let ascii_path =
            data_dir.join("eosCMS_data").join(format!("{stem}.data"));
        let tbl = CmsXTable::from_data_file(&ascii_path)?;
        let _ = write_cms_slice_bin_atomic(&slice_bin, &tbl);
        return Ok(tbl);
    }

    Err(EosError::InvalidState {
        context: "load_or_build_cms_slice",
        detail: format!(
            "FlexMESA CMS EOS slice '{stem}' not found in {}. \
             Run 'flexmesa data fetch' (or set FLEXMESA_DATA_DIR) to download \
             the verified data-v1 physics table bundle.",
            flex_dir.display()
        ),
    })
}

/// Reads a cached `FMDT4D01` binary slice into an [`EosDtXzTable`].
pub fn read_dt_slice_bin(path: &Path) -> Result<EosDtXzTable, EosError> {
    let err = |msg: String| EosError::InvalidState {
        context: "read_dt_slice_bin",
        detail: msg,
    };
    let bytes = std::fs::read(path).map_err(|e| err(e.to_string()))?;
    if bytes.len() < 84 || &bytes[0..8] != DT_SLICE_BIN_MAGIC {
        return Err(err("invalid FMDT4D01 magic or truncated header".to_string()));
    }
    let f64_at = |off: usize| -> f64 {
        f64::from_le_bytes(bytes[off..off + 8].try_into().unwrap_or([0; 8]))
    };
    let version = i32::from_le_bytes(bytes[8..12].try_into().unwrap_or([0; 4]));
    let x = f64_at(12);
    let z = f64_at(20);
    let num_log_ts =
        u32::from_le_bytes(bytes[28..32].try_into().unwrap_or([0; 4])) as usize;
    let log_t_min = f64_at(32);
    let log_t_max = f64_at(40);
    let del_log_t = f64_at(48);
    let num_log_qs =
        u32::from_le_bytes(bytes[56..60].try_into().unwrap_or([0; 4])) as usize;
    let log_q_min = f64_at(60);
    let log_q_max = f64_at(68);
    let del_log_q = f64_at(76);

    let ngrid = num_log_qs * num_log_ts;
    let body = &bytes[84..];
    if body.len() < ngrid * VERTEX_BYTES {
        return Err(err("truncated FMDT4D01 body".to_string()));
    }
    let dense: Vec<EosDtVertex> = body[..ngrid * VERTEX_BYTES]
        .as_chunks::<VERTEX_BYTES>()
        .0
        .iter()
        .map(decode_dt_vertex)
        .collect();
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

/// Writes an [`EosDtXzTable`] to a `FMDT4D01` binary slice file atomically.
pub fn write_dt_slice_bin_atomic(
    path: &Path,
    tbl: &EosDtXzTable,
) -> Result<(), String> {
    let dense = tbl.dense.as_ref().ok_or("missing dense vertices")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let tmp = path.with_extension(format!("bin.tmp.{}", std::process::id()));
    {
        let file = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
        let mut w = BufWriter::with_capacity(512 * 1024, file);
        let mut hdr = Vec::with_capacity(84);
        hdr.extend_from_slice(DT_SLICE_BIN_MAGIC);
        hdr.extend_from_slice(&tbl.version.to_le_bytes());
        hdr.extend_from_slice(&tbl.x.to_le_bytes());
        hdr.extend_from_slice(&tbl.z.to_le_bytes());
        hdr.extend_from_slice(&(tbl.num_log_ts as u32).to_le_bytes());
        hdr.extend_from_slice(&tbl.log_t_min.to_le_bytes());
        hdr.extend_from_slice(&tbl.log_t_max.to_le_bytes());
        hdr.extend_from_slice(&tbl.del_log_t.to_le_bytes());
        hdr.extend_from_slice(&(tbl.num_log_qs as u32).to_le_bytes());
        hdr.extend_from_slice(&tbl.log_q_min.to_le_bytes());
        hdr.extend_from_slice(&tbl.log_q_max.to_le_bytes());
        hdr.extend_from_slice(&tbl.del_log_q.to_le_bytes());
        w.write_all(&hdr).map_err(|e| e.to_string())?;
        let mut vbuf = [0u8; VERTEX_BYTES];
        for v in dense {
            encode_dt_vertex(v, &mut vbuf);
            w.write_all(&vbuf).map_err(|e| e.to_string())?;
        }
        w.flush().map_err(|e| e.to_string())?;
    }
    std::fs::rename(&tmp, path).map_err(|e| e.to_string())
}

/// Reads upstream MESA's unformatted Fortran cache (`Read_EoS_Cache` in `eosdt_load_tables.f90`).
pub fn read_mesa_eosdt_cache_bin(
    path: &Path,
) -> Result<EosDtXzTable, EosError> {
    let err = |msg: String| EosError::InvalidState {
        context: "read_mesa_eosdt_cache_bin",
        detail: msg,
    };
    let file = std::fs::File::open(path).map_err(|e| err(e.to_string()))?;
    let mut reader = BufReader::with_capacity(512 * 1024, file);
    let mut r1_hdr = [0u8; 84];
    reader
        .read_exact(&mut r1_hdr)
        .map_err(|e| err(e.to_string()))?;
    let r1_len = u32::from_le_bytes(r1_hdr[0..4].try_into().unwrap_or([0; 4]));
    let r1_tail =
        u32::from_le_bytes(r1_hdr[80..84].try_into().unwrap_or([0; 4]));
    if r1_len != 76 || r1_tail != 76 {
        return Err(err(format!("unexpected MESA header record len {r1_len}")));
    }
    let f64_at = |off: usize| -> f64 {
        f64::from_le_bytes(r1_hdr[off..off + 8].try_into().unwrap_or([0; 8]))
    };
    let x = f64_at(4);
    let z = f64_at(12);
    let num_log_ts =
        i32::from_le_bytes(r1_hdr[20..24].try_into().unwrap_or([0; 4]))
            as usize;
    let log_t_min = f64_at(24);
    let log_t_max = f64_at(32);
    let del_log_t = f64_at(40);
    let num_log_qs =
        i32::from_le_bytes(r1_hdr[48..52].try_into().unwrap_or([0; 4]))
            as usize;
    let log_q_min = f64_at(52);
    let log_q_max = f64_at(60);
    let del_log_q = f64_at(68);
    let version =
        i32::from_le_bytes(r1_hdr[76..80].try_into().unwrap_or([0; 4]));

    let mut r2_marker = [0u8; 4];
    reader
        .read_exact(&mut r2_marker)
        .map_err(|e| err(e.to_string()))?;
    let r2_len = u32::from_le_bytes(r2_marker) as usize;
    let ngrid = num_log_qs * num_log_ts;
    let nv_mesa = 26usize;
    if r2_len != 4 * nv_mesa * ngrid * 8 {
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
                let val = f64::from_le_bytes(
                    cell_buf[s..s + 8].try_into().unwrap_or([0; 8]),
                );
                *slot = if is_bad(val) { 0.0 } else { val };
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

/// Reads a cached `FMCMS001` binary slice into a [`CmsXTable`].
pub fn read_cms_slice_bin(path: &Path) -> Result<CmsXTable, EosError> {
    let err = |msg: String| EosError::InvalidState {
        context: "read_cms_slice_bin",
        detail: msg,
    };
    let bytes = std::fs::read(path).map_err(|e| err(e.to_string()))?;
    if bytes.len() < 76 || &bytes[0..8] != CMS_SLICE_BIN_MAGIC {
        return Err(err("invalid FMCMS001 magic or truncated header".to_string()));
    }
    let f64_at = |off: usize| -> f64 {
        f64::from_le_bytes(bytes[off..off + 8].try_into().unwrap_or([0; 8]))
    };
    let version = i32::from_le_bytes(bytes[8..12].try_into().unwrap_or([0; 4]));
    let x = f64_at(12);
    let num_log_ts =
        u32::from_le_bytes(bytes[20..24].try_into().unwrap_or([0; 4])) as usize;
    let log_t_min = f64_at(24);
    let log_t_max = f64_at(32);
    let delta_log_t = f64_at(40);
    let num_log_rhos =
        u32::from_le_bytes(bytes[48..52].try_into().unwrap_or([0; 4])) as usize;
    let log_rho_min = f64_at(52);
    let log_rho_max = f64_at(60);
    let delta_log_rho = f64_at(68);

    let ngrid = num_log_ts * num_log_rhos;
    let body = &bytes[76..];
    if body.len() < ngrid * VERTEX_BYTES {
        return Err(err("truncated FMCMS001 body".to_string()));
    }
    let dense: Vec<EosDtVertex> = body[..ngrid * VERTEX_BYTES]
        .as_chunks::<VERTEX_BYTES>()
        .0
        .iter()
        .map(decode_dt_vertex)
        .collect();
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

/// Writes a [`CmsXTable`] to a `FMCMS001` binary slice file atomically.
pub fn write_cms_slice_bin_atomic(
    path: &Path,
    tbl: &CmsXTable,
) -> Result<(), String> {
    let dense = tbl.dense.as_ref().ok_or("missing dense CMS vertices")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let tmp = path.with_extension(format!("bin.tmp.{}", std::process::id()));
    {
        let file = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
        let mut w = BufWriter::with_capacity(256 * 1024, file);
        let mut hdr = Vec::with_capacity(76);
        hdr.extend_from_slice(CMS_SLICE_BIN_MAGIC);
        hdr.extend_from_slice(&tbl.version.to_le_bytes());
        hdr.extend_from_slice(&tbl.x.to_le_bytes());
        hdr.extend_from_slice(&(tbl.num_log_ts as u32).to_le_bytes());
        hdr.extend_from_slice(&tbl.log_t_min.to_le_bytes());
        hdr.extend_from_slice(&tbl.log_t_max.to_le_bytes());
        hdr.extend_from_slice(&tbl.delta_log_t.to_le_bytes());
        hdr.extend_from_slice(&(tbl.num_log_rhos as u32).to_le_bytes());
        hdr.extend_from_slice(&tbl.log_rho_min.to_le_bytes());
        hdr.extend_from_slice(&tbl.log_rho_max.to_le_bytes());
        hdr.extend_from_slice(&tbl.delta_log_rho.to_le_bytes());
        w.write_all(&hdr).map_err(|e| e.to_string())?;
        let mut vbuf = [0u8; VERTEX_BYTES];
        for v in dense {
            encode_dt_vertex(v, &mut vbuf);
            w.write_all(&vbuf).map_err(|e| e.to_string())?;
        }
        w.flush().map_err(|e| e.to_string())?;
    }
    std::fs::rename(&tmp, path).map_err(|e| e.to_string())
}
