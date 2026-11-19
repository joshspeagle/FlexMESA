// SPDX-License-Identifier: LGPL-3.0-or-later
//! Numerics-mode isolation verification for the `ionization` `FMCENV01` cache
//! (`#164`).
//!
//! `IonTableGrid` spline construction (`interp_mkbicub_db`) uses only IEEE-754
//! arithmetic with zero `flexmesa-math` calls, so its `FMIONCA1` payload and
//! `FMCENV01` cache envelope must be bit-for-bit identical in `Strict` and
//! `Fast` modes for both embedded `FMION001` tables and external ASCII
//! `ionization_data/` tables.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use flexmesa_ionization::{
    ION_XS, ION_ZS, IonTableGrid, NUM_ION_VALS, NUM_ION_XS_FOR_Z,
};
use flexmesa_math::{NumericsMode, with_numerics_mode};

#[allow(clippy::cast_precision_loss)]
fn write_minimal_ion_ascii_table(path: &Path, x: f64, z: f64) {
    let mut out = String::from(
        "version X Z num_logTs logT_min logT_max del_logT num_logQs logQ_min logQ_max del_logQ\n",
    );
    let _ =
        writeln!(out, "1 {x:.2} {z:.2} 4 3.00 3.18 0.06 4 -6.00 -5.73 0.09\n");
    for iq in 0..4_u32 {
        let lq = -6.00 + f64::from(iq) * 0.09;
        out.push_str("logQ\n");
        let _ = writeln!(out, "{lq:.4}\n");
        out.push_str("logT vals\n");
        for jt in 0..4_u32 {
            let lt = 3.00 + f64::from(jt) * 0.06;
            let _ = write!(out, "{lt:.4}");
            for v in 0..NUM_ION_VALS {
                let val = 0.35
                    + 0.017 * f64::from(iq) * f64::from(iq)
                    + 0.023 * f64::from(jt)
                    + 0.0013 * (v as f64);
                let _ = write!(out, " {val:.12e}");
            }
            out.push('\n');
        }
        if iq + 1 < 4 {
            out.push_str("\n\n");
        }
    }
    fs::write(path, out).expect("write minimal ion table");
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn ion_filename(dir: &Path, z: f64, x: f64) -> PathBuf {
    let setstr = |v: f64| -> String {
        if v > 0.99999 {
            "100".to_string()
        } else {
            let pct = (100.0 * v + 0.5).floor() as usize;
            format!("{pct:02}")
        }
    };
    dir.join(format!("ion_{}z{}x.data", setstr(z), setstr(x)))
}

#[test]
fn test_ionization_runtime_cache_is_bit_identical_between_strict_and_fast_modes()
 {
    let root = std::env::temp_dir()
        .join(format!("flexmesa_ion_mode_iso_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);

    // 1. Embedded full 175 x 102 x 28 x 30 spline grid
    let cache_strict = root.join("cache_strict").join("v1");
    let cache_fast = root.join("cache_fast").join("v1");
    let g_strict = with_numerics_mode(NumericsMode::Strict, || {
        IonTableGrid::load_or_verify_cache_in(None, Some(&cache_strict))
    })
    .expect("strict embedded ion cache build");
    let g_fast = with_numerics_mode(NumericsMode::Fast, || {
        IonTableGrid::load_or_verify_cache_in(None, Some(&cache_fast))
    })
    .expect("fast embedded ion cache build");
    assert_eq!(
        g_strict.encode_cache_payload().expect("encode strict"),
        g_fast.encode_cache_payload().expect("encode fast"),
        "embedded IonTableGrid spline payload must not depend on NumericsMode"
    );

    // 2. External 22-table ASCII ionization_data/ directory
    let ascii_dir = root.join("ionization_data");
    fs::create_dir_all(&ascii_dir).expect("mkdir ionization_data");
    for (&z, &n_xs) in ION_ZS.iter().zip(NUM_ION_XS_FOR_Z.iter()) {
        for &x in ION_XS.iter().take(n_xs) {
            write_minimal_ion_ascii_table(
                &ion_filename(&ascii_dir, z, x),
                x,
                z,
            );
        }
    }
    let ext_strict_cache = root.join("ext_strict").join("v1");
    let ext_fast_cache = root.join("ext_fast").join("v1");
    let ext_strict = with_numerics_mode(NumericsMode::Strict, || {
        IonTableGrid::load_or_verify_cache_in(
            Some(&ascii_dir),
            Some(&ext_strict_cache),
        )
    })
    .expect("strict ascii ion cache build");
    let ext_fast = with_numerics_mode(NumericsMode::Fast, || {
        IonTableGrid::load_or_verify_cache_in(
            Some(&ascii_dir),
            Some(&ext_fast_cache),
        )
    })
    .expect("fast ascii ion cache build");
    assert_eq!(
        ext_strict
            .encode_cache_payload()
            .expect("encode ext strict"),
        ext_fast.encode_cache_payload().expect("encode ext fast"),
        "ASCII IonTableGrid spline payload must not depend on NumericsMode"
    );

    let strict_files: Vec<PathBuf> =
        fs::read_dir(ext_strict_cache.join("ionization"))
            .expect("read strict cache dir")
            .map(|e| e.expect("entry").path())
            .collect();
    let fast_files: Vec<PathBuf> =
        fs::read_dir(ext_fast_cache.join("ionization"))
            .expect("read fast cache dir")
            .map(|e| e.expect("entry").path())
            .collect();
    assert_eq!(strict_files.len(), 1);
    assert_eq!(fast_files.len(), 1);
    assert_eq!(
        fs::read(&strict_files[0]).expect("read strict cache bytes"),
        fs::read(&fast_files[0]).expect("read fast cache bytes"),
        "on-disk FMCENV01 ionization cache envelope must match bit-for-bit"
    );

    let _ = fs::remove_dir_all(&root);
}
