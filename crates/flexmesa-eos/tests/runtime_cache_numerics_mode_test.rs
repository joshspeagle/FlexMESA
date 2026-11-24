// SPDX-License-Identifier: GPL-3.0-or-later
//! Numerics-mode isolation verification for `eos_4d_slices` and `helm`
//! `FMCENV01` runtime caches (`#164`, `DECISIONS #33`).
//!
//! Verifies:
//! 1. `CMS` 4D slice cache (`load_or_build_cms_slice_in`): `CmsXTable` calls
//!    `flexmesa_math::safe_log` on `Cp` and `Cv`, producing distinct IEEE-754
//!    bits between `Strict` (`crlibm` `log_rz`) and `Fast` (`f64::ln`). Both
//!    modes write separate `*.strict.bin` / `*.fast.bin` `FMCENV01` files in
//!    the same `cache_root`, and a `Strict` reload from a `Fast`-populated
//!    cache directory never receives `Fast` coefficients.
//! 2. `DT` (`OPAL/SCVH` / `FreeEOS`) 4D slice cache (`load_or_build_dt_slice_in`):
//!    writes mode-separated `*.strict.bin` / `*.fast.bin` files while producing
//!    bit-identical payloads (`flexmesa_const::LN10` + `interp_mkbicub_db`).
//! 3. `HELM` `FMCENV01` cache (`load_or_build_helm_vertices_in`): parses IEEE-754
//!    table vertices with zero `flexmesa-math` calls, producing bit-for-bit
//!    identical `FMHELM01` payloads and `FMCENV01` envelopes in `Strict` and
//!    `Fast` modes.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]

use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use flexmesa_eos::cms::table::{CMS_RAW_BIN_MAGIC, CmsXTable};
use flexmesa_eos::helm::table::{HELM_IMAX, HELM_JMAX};
use flexmesa_eos::helm::table_io::{
    encode_helm_full_bytes, load_or_build_helm_vertices_in,
};
use flexmesa_eos::tables_4d::slice_cache::{
    encode_cms_slice_bytes, encode_dt_slice_bytes, load_or_build_cms_slice_in,
    load_or_build_dt_slice_in,
};
use flexmesa_eos::tables_4d::spline::EosDtTableKind;
use flexmesa_eos::tables_4d::table_io::DT_RAW_BIN_MAGIC;
use flexmesa_math::{NumericsMode, safe_log, with_numerics_mode};

fn find_ulp_divergent_log_input() -> f64 {
    for k in 1..20_000_u64 {
        let candidate = 1.0e7_f64 + (k as f64) * 0.137_931_034_482_758_62;
        let s =
            with_numerics_mode(NumericsMode::Strict, || safe_log(candidate));
        let f = with_numerics_mode(NumericsMode::Fast, || safe_log(candidate));
        if s.to_bits() != f.to_bits() {
            return candidate;
        }
    }
    panic!(
        "expected at least one ULP difference between crlibm log_rz and std ln"
    );
}

fn write_cms_raw_slice(path: &Path, cp_val: f64) {
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
            for col in 0..16_u32 {
                let val = if col == 5 || col == 6 {
                    cp_val
                } else {
                    4.5 + 0.03 * (it as f64)
                        + 0.07 * (jrho as f64)
                        + 0.01 * f64::from(col)
                };
                out.extend_from_slice(&val.to_le_bytes());
            }
        }
    }
    fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    fs::write(path, out).expect("write cms raw");
}

fn write_dt_raw_slice(path: &Path) {
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
            for col in 0..16_u32 {
                let val = 8.0
                    + 0.05 * (iq as f64)
                    + 0.11 * (jt as f64)
                    + 0.013 * f64::from(col);
                out.extend_from_slice(&val.to_le_bytes());
            }
        }
    }
    fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    fs::write(path, out).expect("write dt raw");
}

fn write_mesa_helm_fortran_cache(path: &Path) {
    fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    let file = fs::File::create(path).expect("create helm_table.bin");
    let mut w = BufWriter::with_capacity(4 * 1024 * 1024, file);

    // 24-byte header record: imax at [4..8], jmax at [16..20]
    let mut hdr = [0_u8; 24];
    hdr[0..4].copy_from_slice(&4_u32.to_le_bytes());
    hdr[4..8].copy_from_slice(&(HELM_IMAX as i32).to_le_bytes());
    hdr[8..12].copy_from_slice(&4_u32.to_le_bytes());
    hdr[12..16].copy_from_slice(&4_u32.to_le_bytes());
    hdr[16..20].copy_from_slice(&(HELM_JMAX as i32).to_le_bytes());
    hdr[20..24].copy_from_slice(&4_u32.to_le_bytes());
    w.write_all(&hdr).expect("write hdr");

    let n_total = HELM_IMAX * HELM_JMAX;
    let rec_len = ((n_total * 8) as u32).to_le_bytes();
    for col in 0..21_usize {
        w.write_all(&rec_len).expect("write lead marker");
        let val = 1.25_f64 + 0.0625 * (col as f64);
        let chunk_val = val.to_le_bytes();
        let mut block = Vec::with_capacity(8192);
        for _ in 0..1024 {
            block.extend_from_slice(&chunk_val);
        }
        let full_blocks = n_total / 1024;
        let rem = n_total % 1024;
        for _ in 0..full_blocks {
            w.write_all(&block).expect("write block");
        }
        for _ in 0..rem {
            w.write_all(&chunk_val).expect("write tail");
        }
        w.write_all(&rec_len).expect("write trail marker");
    }
    w.flush().expect("flush helm_table.bin");
}

#[test]
fn test_eos_4d_slices_runtime_cache_isolates_numerics_modes() {
    let root = std::env::temp_dir()
        .join(format!("flexmesa_eos_4d_mode_iso_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);

    let data_dir = root.join("data");
    let cache_v1 = root.join("cache").join("v1");
    let cp_val = find_ulp_divergent_log_input();
    let cms_raw = data_dir.join("eos_4d_raw").join("mesa-CMS_070x.raw");
    let dt_raw = data_dir.join("eos_4d_raw").join("mesa-eosDT_02z60x.raw");
    write_cms_raw_slice(&cms_raw, cp_val);
    write_dt_raw_slice(&dt_raw);

    // 1. CMS cache: populate in Fast mode first, then load in Strict mode.
    let cms_strict_expected = with_numerics_mode(NumericsMode::Strict, || {
        CmsXTable::from_raw_bin(&cms_raw).expect("uncached strict CMS")
    });
    let cms_fast_cached = with_numerics_mode(NumericsMode::Fast, || {
        load_or_build_cms_slice_in(
            &data_dir,
            None,
            Some(&cache_v1),
            "mesa-CMS_070x",
        )
        .expect("cached fast CMS")
    });
    let cms_strict_cached = with_numerics_mode(NumericsMode::Strict, || {
        load_or_build_cms_slice_in(
            &data_dir,
            None,
            Some(&cache_v1),
            "mesa-CMS_070x",
        )
        .expect("cached strict CMS")
    });

    let bytes_fast =
        encode_cms_slice_bytes(&cms_fast_cached).expect("enc fast");
    let bytes_strict =
        encode_cms_slice_bytes(&cms_strict_cached).expect("enc strict");
    let bytes_strict_exp =
        encode_cms_slice_bytes(&cms_strict_expected).expect("enc strict exp");
    assert_ne!(
        bytes_fast, bytes_strict,
        "CMS Cp/Cv safe_log must differ between Fast and Strict on ULP input"
    );
    assert_eq!(
        bytes_strict, bytes_strict_exp,
        "Strict load after Fast cache population must serve exact Strict bytes"
    );

    // 2. DT cache: separate .fast.bin and .strict.bin files, bit-identical spline bytes.
    let dt_fast = with_numerics_mode(NumericsMode::Fast, || {
        load_or_build_dt_slice_in(
            &data_dir,
            None,
            Some(&cache_v1),
            EosDtTableKind::OpalScvh,
            "mesa-eosDT_02z60x",
        )
        .expect("cached fast DT")
    });
    let dt_strict = with_numerics_mode(NumericsMode::Strict, || {
        load_or_build_dt_slice_in(
            &data_dir,
            None,
            Some(&cache_v1),
            EosDtTableKind::OpalScvh,
            "mesa-eosDT_02z60x",
        )
        .expect("cached strict DT")
    });
    assert_eq!(
        encode_dt_slice_bytes(&dt_fast).expect("enc dt fast"),
        encode_dt_slice_bytes(&dt_strict).expect("enc dt strict")
    );

    let mut slice_files: Vec<String> =
        fs::read_dir(cache_v1.join("eos_4d_slices"))
            .expect("read eos_4d_slices")
            .map(|e| {
                e.expect("entry").file_name().to_string_lossy().into_owned()
            })
            .collect();
    slice_files.sort();
    assert_eq!(slice_files.len(), 4);
    assert!(slice_files.iter().any(|f| f.ends_with(".fast.bin")));
    assert!(slice_files.iter().any(|f| f.ends_with(".strict.bin")));

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn test_helm_runtime_cache_is_bit_identical_between_strict_and_fast_modes() {
    let root = std::env::temp_dir()
        .join(format!("flexmesa_helm_mode_iso_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);

    let empty_data = root.join("empty_helm_data");
    fs::create_dir_all(&empty_data).expect("mkdir empty_helm_data");
    let mesa_data = root.join("mesa_data");
    write_mesa_helm_fortran_cache(
        &mesa_data
            .join("eosDT_data")
            .join("cache")
            .join("helm_table.bin"),
    );

    let helm_cache_fast = root.join("helm_fast").join("v1");
    let helm_cache_strict = root.join("helm_strict").join("v1");
    let v_fast = with_numerics_mode(NumericsMode::Fast, || {
        load_or_build_helm_vertices_in(
            &empty_data,
            Some(&mesa_data),
            Some(&helm_cache_fast),
        )
        .expect("fast HELM cache build")
    });
    let v_strict = with_numerics_mode(NumericsMode::Strict, || {
        load_or_build_helm_vertices_in(
            &empty_data,
            Some(&mesa_data),
            Some(&helm_cache_strict),
        )
        .expect("strict HELM cache build")
    });
    assert_eq!(
        encode_helm_full_bytes(&v_fast).expect("enc helm fast"),
        encode_helm_full_bytes(&v_strict).expect("enc helm strict"),
        "HELM FMHELM01 payload must be bit-identical across Strict and Fast"
    );

    let helm_fast_files: Vec<PathBuf> =
        fs::read_dir(helm_cache_fast.join("helm"))
            .expect("read helm fast cache")
            .map(|e| e.expect("entry").path())
            .collect();
    let helm_strict_files: Vec<PathBuf> =
        fs::read_dir(helm_cache_strict.join("helm"))
            .expect("read helm strict cache")
            .map(|e| e.expect("entry").path())
            .collect();
    assert_eq!(helm_fast_files.len(), 1);
    assert_eq!(helm_strict_files.len(), 1);
    assert_eq!(
        fs::read(&helm_fast_files[0]).expect("read fast helm bin"),
        fs::read(&helm_strict_files[0]).expect("read strict helm bin"),
        "on-disk FMCENV01 HELM cache envelope must match bit-for-bit"
    );

    let v_reloaded = with_numerics_mode(NumericsMode::Strict, || {
        load_or_build_helm_vertices_in(
            &empty_data,
            Some(&mesa_data),
            Some(&helm_cache_fast),
        )
        .expect("reload HELM cache")
    });
    assert_eq!(v_reloaded.len(), HELM_IMAX * HELM_JMAX);
    assert_eq!(v_reloaded[0], v_strict[0]);

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn test_excor7_table_cache_isolates_strict_and_fast_modes() {
    use flexmesa_eos::pc::excor_table::Excor7Table;

    let fast_ptr: *const Excor7Table =
        with_numerics_mode(NumericsMode::Fast, || {
            std::ptr::from_ref(Excor7Table::global())
        });
    let strict_ptr: *const Excor7Table =
        with_numerics_mode(NumericsMode::Strict, || {
            std::ptr::from_ref(Excor7Table::global())
        });
    assert_ne!(
        fast_ptr, strict_ptr,
        "Strict and Fast EXCOR7 OnceLock caches must remain isolated"
    );
}
