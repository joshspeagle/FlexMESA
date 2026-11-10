// SPDX-License-Identifier: LGPL-3.0-or-later
//! Full opacity table builder and binary cache manager (`FMKAP001` format).
//!
//! Resolves upstream MESA `kap_data` via `FLEXMESA_DATA_DIR`, `MESA_DIR`, or
//! relative `.mesa_temp_cache/mesa_data/kap_data`, parses all fixed-metal,
//! Type 2 C/O-enhanced, `kapCN`, and `AESOPUS_AGSS09.h5` tables in full, and
//! caches the resulting `FMKAP001` binary payload in
//! `.mesa_temp_cache/bin_cache/kap_full/kap_tables_full.bin`.

use crate::loader::KapTables;
use crate::loader_parse::parse_kap_tables;
use crate::spline::get_output_string;
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

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

const FREEDMAN_ZS: [f64; 7] = [
    0.007452, 0.014760, 0.029024, 0.069840, 0.130292, 0.321456, 0.428844,
];

const CO_PREFIXES: &[&str] = &["gn93_co", "gs98_co", "a09_co"];
const CO_ZS: [f64; 8] =
    [0.000, 0.001, 0.004, 0.010, 0.020, 0.030, 0.050, 0.100];
const CO_XS: [f64; 5] = [0.00, 0.03, 0.10, 0.35, 0.70];

fn find_relative_dir(rel: &str) -> Option<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd);
    }
    roots.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    for root in roots {
        for anc in root.ancestors() {
            let candidate = anc.join(rel);
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Resolve the upstream `kap_data` directory if available.
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
    find_relative_dir(".mesa_temp_cache/mesa_data/kap_data")
}

/// Resolve the binary cache directory `.mesa_temp_cache/bin_cache/kap_full`.
#[must_use]
pub fn find_kap_cache_dir() -> Option<PathBuf> {
    if let Ok(val) = std::env::var("FLEXMESA_CACHE_DIR") {
        return Some(PathBuf::from(val).join("kap_full"));
    }
    if let Some(bin_cache) = find_relative_dir(".mesa_temp_cache/bin_cache") {
        return Some(bin_cache.join("kap_full"));
    }
    None
}

/// Load [`KapTables`] from the full binary cache, or build and cache it from
/// `kap_data` when available, falling back to `embedded_bin` otherwise.
#[must_use]
pub(crate) fn load_or_build_kap_tables(embedded_bin: &[u8]) -> KapTables {
    let cache_dir = find_kap_cache_dir();
    if let Some(ref cdir) = cache_dir {
        let full_bin_path = cdir.join("kap_tables_full.bin");
        if let Ok(bytes) = fs::read(&full_bin_path) {
            if bytes.len() > 50_000_000 && bytes.starts_with(b"FMKAP001") {
                return parse_kap_tables(&bytes);
            }
        }
    }
    if let Some(kap_dir) = find_kap_data_dir() {
        if let Some(bytes) =
            build_full_kap_tables_bytes(&kap_dir, cache_dir.as_deref())
        {
            if let Some(ref cdir) = cache_dir {
                let _ = fs::create_dir_all(cdir);
                let tmp_path = cdir
                    .join(format!("kap_tables_full.{}.tmp", std::process::id()));
                let dst_path = cdir.join("kap_tables_full.bin");
                if fs::write(&tmp_path, &bytes).is_ok() {
                    let _ = fs::rename(&tmp_path, &dst_path);
                }
            }
            return parse_kap_tables(&bytes);
        }
    }
    parse_kap_tables(embedded_bin)
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

fn parse_fixed_file_bytes(path: &Path, x: f64, z: f64) -> Option<Vec<u8>> {
    let text = fs::read_to_string(path).ok()?;
    let mut lines = text.lines();
    lines.next()?;
    lines.next()?;
    let hdr: Vec<&str> = lines.next()?.split_whitespace().collect();
    if hdr.len() < 10 {
        return None;
    }
    let num_lr: usize = hdr[4].parse().ok()?;
    let log_r_min: f64 = hdr[5].parse().ok()?;
    let log_r_max: f64 = hdr[6].parse().ok()?;
    let num_lt: usize = hdr[7].parse().ok()?;
    let log_t_min: f64 = hdr[8].parse().ok()?;
    let log_t_max: f64 = hdr[9].parse().ok()?;
    lines.next()?;
    lines.next()?;

    let mut toks = lines.flat_map(str::split_whitespace);
    let mut log_rs = Vec::with_capacity(num_lr);
    for _ in 0..num_lr {
        log_rs.push(toks.next()?.parse::<f64>().ok()?);
    }
    let mut log_ts = Vec::with_capacity(num_lt);
    let mut raw_i16 = Vec::with_capacity(num_lr * num_lt);
    for _ in 0..num_lt {
        log_ts.push(toks.next()?.parse::<f64>().ok()?);
        for _ in 0..num_lr {
            let v: f64 = toks.next()?.parse().ok()?;
            raw_i16.push((v * 1000.0).round() as i16);
        }
    }

    let mut out = Vec::with_capacity(64 + (num_lr + num_lt) * 8 + raw_i16.len() * 2);
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
    Some(out)
}

fn parse_co_file_bytes(
    path: &Path,
    x: f64,
    z: f64,
) -> Option<([f64; 4], Vec<u8>)> {
    let text = fs::read_to_string(path).ok()?;
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < 10 {
        return None;
    }
    let hdr: Vec<&str> = lines[2].split_whitespace().collect();
    if hdr.len() < 15 {
        return None;
    }
    let num_co_tables: usize = hdr[2].parse().ok()?;
    let zfracs = [
        hdr[5].parse::<f64>().ok()?,
        hdr[6].parse::<f64>().ok()?,
        hdr[7].parse::<f64>().ok()?,
        hdr[8].parse::<f64>().ok()?,
    ];
    let num_lr: usize = hdr[9].parse().ok()?;
    let log_r_min: f64 = hdr[10].parse().ok()?;
    let log_r_max: f64 = hdr[11].parse().ok()?;
    let num_lt: usize = hdr[12].parse().ok()?;
    let log_t_min: f64 = hdr[13].parse().ok()?;
    let log_t_max: f64 = hdr[14].parse().ok()?;

    let num_dxc_gt_dxo: usize =
        lines[4].split_whitespace().next()?.parse().ok()?;
    let mut dxc_dxo = Vec::with_capacity(num_co_tables);
    let mut line_idx = 6;
    while line_idx < lines.len() && dxc_dxo.len() < num_co_tables {
        let row: Vec<&str> = lines[line_idx].split_whitespace().collect();
        if row.len() == 4 && row[0].parse::<usize>().is_ok() {
            let dxc: f64 = row[2].parse().ok()?;
            let dxo: f64 = row[3].parse().ok()?;
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
                lrs.push(toks.next()?.parse::<f64>().ok()?);
            }
            let mut lts = Vec::with_capacity(num_lt);
            let mut grid = Vec::with_capacity(num_lr * num_lt);
            for _ in 0..num_lt {
                lts.push(toks.next()?.parse::<f64>().ok()?);
                for _ in 0..num_lr {
                    let v: f64 = toks.next()?.parse().ok()?;
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
        return None;
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
    Some((zfracs, out))
}

fn build_aesopus_section_bytes(
    kap_dir: &Path,
    cache_dir: Option<&Path>,
) -> Option<Vec<u8>> {
    if let Some(cdir) = cache_dir {
        let p = cdir.join("aesopus_section.bin");
        if let Ok(b) = fs::read(&p) {
            if b.len() > 46_000_000 {
                return Some(b);
            }
        }
    }
    let h5_path = kap_dir.join("AESOPUS_AGSS09.h5");
    if !h5_path.is_file() {
        return None;
    }
    let py_script = r#"
import sys, struct, h5py, numpy as np
h5_path, out_path = sys.argv[1], sys.argv[2]
with h5py.File(h5_path, "r") as f, open(out_path, "wb") as out:
    for k in ["Zsun", "fCO_ref", "fC_ref", "fN_ref"]:
        out.write(struct.pack("<d", float(f[k][()])))
    log_rs = np.asarray(f["logRs"][:], dtype=np.float64)
    log_ts = np.asarray(f["logTs"][:], dtype=np.float64)
    zs = np.asarray(f["Zs"][:], dtype=np.float64)
    out.write(struct.pack("<III", len(log_rs), len(log_ts), len(zs)))
    out.write(log_rs.tobytes())
    out.write(log_ts.tobytes())
    out.write(zs.tobytes())
    for z in zs:
        g = f[f"{z:.6f}"]
        xs = np.asarray(g["Xs"][:], dtype=np.float64)
        fcos = np.asarray(g["fCOs"][:], dtype=np.float64)
        fcs = np.asarray(g["fCs"][:], dtype=np.float64)
        fns = np.asarray(g["fNs"][:], dtype=np.float64)
        out.write(struct.pack("<IIII", len(xs), len(fcos), len(fcs), len(fns)))
        out.write(xs.tobytes()); out.write(fcos.tobytes())
        out.write(fcs.tobytes()); out.write(fns.tobytes())
        out.write(b"\x01")
        kap = np.asarray(g["kap"][:], dtype=np.float32)
        slices = np.moveaxis(kap, (0, 1), (-2, -1)).reshape(-1, len(log_ts) * len(log_rs))
        for s in slices:
            out.write(b"\x01")
            out.write(s.tobytes())
"#;
    let out_dir = cache_dir.map_or_else(std::env::temp_dir, Path::to_path_buf);
    let _ = fs::create_dir_all(&out_dir);
    let tmp_file =
        out_dir.join(format!("aesopus_section.{}.tmp", std::process::id()));
    let dst_file = out_dir.join("aesopus_section.bin");
    let mut py_bins: Vec<PathBuf> = Vec::new();
    if let Ok(py) = std::env::var("PYTHON") {
        py_bins.push(PathBuf::from(py));
    }
    if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
        py_bins.push(PathBuf::from(venv).join("bin").join("python"));
    }
    py_bins.push(PathBuf::from("python3"));
    let ok = py_bins.iter().any(|bin| {
        Command::new(bin)
            .args([
                "-c",
                py_script,
                h5_path.to_str().unwrap_or(""),
                tmp_file.to_str().unwrap_or(""),
            ])
            .status()
            .is_ok_and(|s| s.success())
    });
    if !ok {
        return None;
    }
    let _ = fs::rename(&tmp_file, &dst_file);
    fs::read(&dst_file).ok()
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

fn build_kapcn_section_bytes(kap_dir: &Path) -> Option<Vec<u8>> {
    let fcn_text = fs::read_to_string(kap_dir.join("kR_Z_fCN.data")).ok()?;
    let mut rows: Vec<(f64, [f64; 7], [f64; 3])> = Vec::new();
    for line in fcn_text.lines().skip(1) {
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() < 11 {
            continue;
        }
        let z: f64 = toks[0].parse().ok()?;
        let mut fcs = [0.0; 7];
        for i in 0..7 {
            fcs[i] = toks[1 + i].parse().ok()?;
        }
        let mut fns = [0.0; 3];
        for i in 0..3 {
            fns[i] = toks[8 + i].parse().ok()?;
        }
        rows.push((z, fcs, fns));
    }
    rows.reverse();

    let mut log_rs = Vec::new();
    let mut log_ts = Vec::new();
    let mut z_payloads = Vec::with_capacity(rows.len());
    for &(z, fcs, fns) in &rows {
        let text = fs::read_to_string(kap_dir.join(kapcn_filename(z))).ok()?;
        let lines: Vec<&str> = text.lines().collect();
        let mut sub_grids = Vec::with_capacity(63);
        let mut idx = 0;
        while idx < lines.len() && sub_grids.len() < 63 {
            let trimmed = lines[idx].trim_start();
            if trimmed.starts_with("log T") {
                if log_rs.is_empty() {
                    for tok in trimmed.split_whitespace().skip(2) {
                        log_rs.push(tok.parse::<f64>().ok()?);
                    }
                }
                let num_lr = log_rs.len();
                let mut grid = Vec::with_capacity(num_lr * 18);
                let mut lts = Vec::with_capacity(18);
                for j in 0..18 {
                    let mut toks = lines[idx + 1 + j].split_whitespace();
                    lts.push(toks.next()?.parse::<f64>().ok()?);
                    for _ in 0..num_lr {
                        let v: f64 = toks.next()?.parse().ok()?;
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
            return None;
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
    Some(out)
}

fn build_full_kap_tables_bytes(
    kap_dir: &Path,
    cache_dir: Option<&Path>,
) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(120_000_000);
    out.extend_from_slice(b"FMKAP001");

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
                    get_output_string(z).ok()?,
                    get_output_string(x).ok()?
                );
                tasks.push((iz, x, z, kap_dir.join(fname)));
            }
        }
        let parsed: Vec<(usize, Vec<u8>)> = tasks
            .into_par_iter()
            .map(|(iz, x, z, p)| Some((iz, parse_fixed_file_bytes(&p, x, z)?)))
            .collect::<Option<Vec<_>>>()?;
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
        let fname =
            format!("lowT_Freedman11_z{}.data", get_output_string(z).ok()?);
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
                    get_output_string(z).ok()?,
                    get_output_string(x).ok()?
                );
                tasks.push((iz, x, z, kap_dir.join(fname)));
            }
        }
        let parsed: Vec<(usize, [f64; 4], Vec<u8>)> = tasks
            .into_par_iter()
            .map(|(iz, x, z, p)| {
                let (zf, b) = parse_co_file_bytes(&p, x, z)?;
                Some((iz, zf, b))
            })
            .collect::<Option<Vec<_>>>()?;
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

    // 3. AESOPUS_AGSS09
    let aes_bytes = build_aesopus_section_bytes(kap_dir, cache_dir)?;
    out.extend_from_slice(&aes_bytes);

    // 4. kapCN
    let cn_bytes = build_kapcn_section_bytes(kap_dir)?;
    out.extend_from_slice(&cn_bytes);

    Some(out)
}
