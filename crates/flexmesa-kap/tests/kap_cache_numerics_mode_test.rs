// SPDX-License-Identifier: GPL-3.0-or-later
//! Numerics-mode isolation verification for the `kap` `FMCENV01` runtime cache
//! (`#164`, `DECISIONS #33`).
//!
//! `build_full_kap_tables_bytes` (`FMKAP001`) parses upstream ASCII `.data`
//! tables and `aesopus_section.bin` with zero `flexmesa-math` calls, while
//! `parse_kap_tables` computes `log_zs` and `dxc_lookup` in memory after
//! reading `FMKAP001` bytes from disk or `FMCENV01` cache.

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use flexmesa_kap::loader_ascii::{
    build_full_kap_tables_bytes, kap_source_relpaths,
    load_or_build_kap_tables_in,
};
use flexmesa_kap::{
    get_kap_results, KapCoOption, KapConfig, KapLowTOption, KapOption,
};
use flexmesa_math::{log10, with_numerics_mode, NumericsMode};

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

fn sample_fixed_table() -> String {
    let mut s = String::new();
    s.push_str("form version X Z num_logRs logR_min logR_max num_logTs logT_min logT_max\n\n");
    s.push_str("1 38 0.70 0.02 4 -8.0 -5.0 4 3.5 5.0\n\n");
    s.push_str("logT                       logR =\n");
    s.push_str("       -8.0 -7.0 -6.0 -5.0\n");
    for jt in 0..4_u32 {
        let lt = 3.5 + 0.5 * f64::from(jt);
        let _ = writeln!(
            s,
            "{lt:.2} {:.3} {:.3} {:.3} {:.3}",
            -1.2 + 0.1 * f64::from(jt),
            -0.9 + 0.1 * f64::from(jt),
            -0.6 + 0.1 * f64::from(jt),
            -0.3 + 0.1 * f64::from(jt),
        );
    }
    s
}

fn sample_co_table() -> String {
    let mut s = String::new();
    s.push_str("header1\nheader2\n");
    s.push_str(
        "1 38 1 0.70 0.02 0.17 0.05 0.48 0.10 4 -8.0 -5.0 4 3.75 5.25\n",
    );
    s.push_str("num_dXC_gt_dXO\n0\ntable_summary\n");
    s.push_str("1 1 0.00 0.00\n\n");
    s.push_str("logT  logR =\n");
    s.push_str("  -8.0 -7.0 -6.0 -5.0\n");
    for jt in 0..4_u32 {
        let lt = 3.75 + 0.5 * f64::from(jt);
        let _ = writeln!(s, "{lt:.2} -0.500 -0.400 -0.300 -0.200");
    }
    s
}

fn sample_kapcn_table() -> String {
    let mut block = String::from("log T   -7.0 -6.0 -5.0 -4.0\n");
    for j in 0..18_u32 {
        let lt = 3.20 + 0.05 * f64::from(j);
        let _ = writeln!(block, "{lt:.2} -2.100 -1.900 -1.700 -1.500");
    }
    block.push('\n');
    block.repeat(63)
}

#[allow(clippy::cast_precision_loss)]
fn sample_aesopus_section(z_test: f64) -> Vec<u8> {
    let mut out = Vec::new();
    for v in [0.01337_f64, -0.3845_f64, -0.7523_f64, -1.2855_f64] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&4_u32.to_le_bytes()); // num_lr
    out.extend_from_slice(&4_u32.to_le_bytes()); // num_lt
    out.extend_from_slice(&1_u32.to_le_bytes()); // num_zs
    for lr in [-8.0_f64, -7.0, -6.0, -5.0] {
        out.extend_from_slice(&lr.to_le_bytes());
    }
    for lt in [3.2_f64, 3.3, 3.4, 3.5] {
        out.extend_from_slice(&lt.to_le_bytes());
    }
    out.extend_from_slice(&z_test.to_le_bytes());
    for n in [1_u32, 1, 1, 1] {
        out.extend_from_slice(&n.to_le_bytes());
    }
    for grid_v in [0.70_f64, 0.0, 0.0, 0.0] {
        out.extend_from_slice(&grid_v.to_le_bytes());
    }
    out.push(1_u8); // z_present
    out.push(1_u8); // slice_present
    for k in 0..16_u32 {
        let v = -2.25_f32 + 0.05_f32 * (k as f32);
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[allow(clippy::cast_precision_loss)]
fn find_ulp_divergent_log10_input() -> f64 {
    for k in 1..20_000_u64 {
        let candidate = 0.01_f64 + (k as f64) * 1.379_310_344_827_586_2e-6;
        let s = with_numerics_mode(NumericsMode::Strict, || log10(candidate));
        let f = with_numerics_mode(NumericsMode::Fast, || log10(candidate));
        if s.to_bits() != f.to_bits() {
            return candidate;
        }
    }
    panic!(
        "expected at least one ULP difference between crlibm log10_rn and std log10"
    );
}

#[test]
fn test_kap_runtime_cache_is_bit_identical_between_strict_and_fast_modes() {
    let root = std::env::temp_dir()
        .join(format!("flexmesa_kap_mode_iso_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);

    let kap_dir = root.join("kap_data");
    fs::create_dir_all(&kap_dir).expect("mkdir kap_data");
    fs::write(kap_dir.join("kR_Z_fCN.data"), SAMPLE_FCN).expect("write fcn");

    let rels = kap_source_relpaths(&kap_dir).expect("kap_source_relpaths");
    let fixed_txt = sample_fixed_table();
    let co_txt = sample_co_table();
    let kapcn_txt = sample_kapcn_table();
    for rel in &rels {
        if rel == "kR_Z_fCN.data" {
            continue;
        }
        let body = if rel.starts_with("kR_Z") {
            &kapcn_txt
        } else if rel.contains("_co_") {
            &co_txt
        } else {
            &fixed_txt
        };
        fs::write(kap_dir.join(rel), body).expect("write synthetic kap table");
    }

    let z_test = find_ulp_divergent_log10_input();
    let aes_path = kap_dir.join("aesopus_section.bin");
    fs::write(&aes_path, sample_aesopus_section(z_test)).expect("write aes");

    let bytes_fast = with_numerics_mode(NumericsMode::Fast, || {
        build_full_kap_tables_bytes(&kap_dir, &aes_path).expect("fast build")
    });
    let bytes_strict = with_numerics_mode(NumericsMode::Strict, || {
        build_full_kap_tables_bytes(&kap_dir, &aes_path).expect("strict build")
    });
    assert_eq!(
        bytes_fast, bytes_strict,
        "FMKAP001 payload bytes must be bit-identical across Strict and Fast"
    );

    let empty_bundle = root.join("empty_bundle");
    fs::create_dir_all(&empty_bundle).expect("mkdir empty_bundle");
    let cache_fast = root.join("cache_fast").join("v1");
    let cache_strict = root.join("cache_strict").join("v1");

    let tbl_fast = with_numerics_mode(NumericsMode::Fast, || {
        load_or_build_kap_tables_in(
            &empty_bundle,
            Some(&kap_dir),
            Some(&cache_fast),
        )
        .expect("fast kap cache populate")
    });
    let tbl_strict_from_fast_cache =
        with_numerics_mode(NumericsMode::Strict, || {
            load_or_build_kap_tables_in(
                &empty_bundle,
                Some(&kap_dir),
                Some(&cache_fast),
            )
            .expect("strict reload from fast kap cache")
        });
    let _tbl_strict = with_numerics_mode(NumericsMode::Strict, || {
        load_or_build_kap_tables_in(
            &empty_bundle,
            Some(&kap_dir),
            Some(&cache_strict),
        )
        .expect("strict kap cache populate")
    });

    // Verify on-disk FMCENV01 envelopes match bit-for-bit across modes.
    let f_files: Vec<PathBuf> = fs::read_dir(cache_fast.join("kap"))
        .expect("read fast kap cache")
        .map(|e| e.expect("entry").path())
        .collect();
    let s_files: Vec<PathBuf> = fs::read_dir(cache_strict.join("kap"))
        .expect("read strict kap cache")
        .map(|e| e.expect("entry").path())
        .collect();
    assert_eq!(f_files.len(), 1);
    assert_eq!(s_files.len(), 1);
    assert_eq!(
        fs::read(&f_files[0]).expect("read fast kap bin"),
        fs::read(&s_files[0]).expect("read strict kap bin"),
        "on-disk FMCENV01 kap cache envelope must match bit-for-bit"
    );

    // Verify in-memory parse_kap_tables evaluates log_zs in the active mode.
    let exp_fast_logz =
        with_numerics_mode(NumericsMode::Fast, || log10(z_test));
    let exp_strict_logz =
        with_numerics_mode(NumericsMode::Strict, || log10(z_test));
    assert_ne!(exp_fast_logz.to_bits(), exp_strict_logz.to_bits());
    assert_eq!(
        tbl_fast.aesopus.log_zs[0].to_bits(),
        exp_fast_logz.to_bits()
    );
    assert_eq!(
        tbl_strict_from_fast_cache.aesopus.log_zs[0].to_bits(),
        exp_strict_logz.to_bits()
    );

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn test_aesopus_strict_vs_fast_bicubic_mode_isolation() {
    let cfg = KapConfig {
        zbase: f64::from_bits(0x3f8c_ac08_3126_e979),
        kap_option: KapOption::A09,
        kap_co_option: KapCoOption::A09Co,
        kap_low_t_option: KapLowTOption::Aesopus,
        cubic_interpolation_in_x: false,
        cubic_interpolation_in_z: false,
        include_electron_conduction: true,
        use_zbase_for_type1: true,
        use_type2_opacities: true,
        ..KapConfig::default()
    };
    let eval = |mode: NumericsMode| {
        with_numerics_mode(mode, || {
            get_kap_results(
                &cfg,
                f64::from_bits(0x3ff1_9c8c_58b7_9d17),
                f64::from_bits(0x3fe5_b970_71a1_ff29),
                f64::from_bits(0x3f94_8e4f_6fd7_6f00),
                f64::from_bits(0x3f68_8cd7_6039_7678),
                f64::from_bits(0x3f59_0514_6aeb_0d98),
                f64::from_bits(0x3f83_1f9a_fd78_f97f),
                0.0,
                f64::from_bits(0xc020_7fd1_3769_ccb8),
                f64::from_bits(0x400c_750b_4c2c_8ffb),
                f64::from_bits(0xc022_6bb2_01fe_476f),
                0.0,
                0.0,
                -20.0,
                0.0,
                0.0,
            )
            .expect("aesopus evaluation")
        })
    };

    let out_strict = eval(NumericsMode::Strict);
    let out_fast = eval(NumericsMode::Fast);
    assert_eq!(out_strict.kap.to_bits(), 0x3f68_8c2b_f0e4_eb01);
    assert_ne!(out_fast.kap.to_bits(), out_strict.kap.to_bits());
    let rel_diff = (out_fast.kap - out_strict.kap).abs() / out_strict.kap;
    assert!(
        rel_diff < 1.0e-5,
        "Fast f64 Hermite diverged unexpectedly: {rel_diff:.3e}"
    );
}
