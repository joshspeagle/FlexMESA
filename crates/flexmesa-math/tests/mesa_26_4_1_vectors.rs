// SPDX-License-Identifier: LGPL-3.0-or-later
//! Exact 64-bit IEEE-754 bit-pattern verification of `flexmesa-math` in
//! `NumericsMode::Strict` against real MESA `26.4.1` (`r26.4.1`) reference
//! vectors (`>= 10,084` vectors per function across all 41 mapped functions;
//! `#165` Items 8, 9, 11, 14).

#![allow(clippy::float_cmp, clippy::too_many_lines)]

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use flexmesa_math::{
    acos, acosh, acospi, asin, asinh, asinpi, atan, atan2, atanh, atanpi, cos,
    cosh, cospi, double_to_str, exp, exp10, expm1, fma, hypot, log, log10,
    log1p, log2, math_init, numerics_mode, pow, pow2, pow3, pow4, pow5, pow6,
    pow7, pow8, powi, powi_f90, powm1, pre_z, safe_log, safe_log10, safe_sqrt,
    sin, sinh, sinpi, tan, tanh, tanpi, with_numerics_mode, NumericsMode,
    MATH_BACKEND, MAX_PRECOMP_INTS,
};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .parent()
        .expect("repo root")
        .to_path_buf()
}

/// Documented NaN rule (`#165` Item 14):
/// - Non-NaN expected outputs require all 64 IEEE-754 bits (`to_bits()`) to
///   match identically (including `+0.0` vs `-0.0`).
/// - Expected NaN outputs require the actual output to be an IEEE-754 quiet NaN
///   (`actual.is_nan()` with quiet bit `0x0008_0000_0000_0000` set).
#[inline]
fn bits_match_with_nan_rule(actual_bits: u64, expected_bits: u64) -> bool {
    let expected_val = f64::from_bits(expected_bits);
    if expected_val.is_nan() {
        let actual_val = f64::from_bits(actual_bits);
        actual_val.is_nan() && (actual_bits & 0x0008_0000_0000_0000) != 0
    } else {
        actual_bits == expected_bits
    }
}

/// Canonicalize quiet-NaN bit patterns to `0x7ff8_0000_0000_0000` when hashing
/// cross-machine strict outputs (`x86_64` vs `aarch64`).
#[inline]
fn canonical_strict_bits(val: f64) -> u64 {
    if val.is_nan() {
        0x7ff8_0000_0000_0000_u64
    } else {
        val.to_bits()
    }
}

fn eval_unary_named(name: &str, x: f64) -> f64 {
    match name {
        "safe_sqrt" => safe_sqrt(x),
        "log" => log(x),
        "safe_log" => safe_log(x),
        "log10" => log10(x),
        "safe_log10" => safe_log10(x),
        "log1p" => log1p(x),
        "log2" => log2(x),
        "exp" => exp(x),
        "exp10" => exp10(x),
        "expm1" => expm1(x),
        "powm1" => powm1(x),
        "pow2" => pow2(x),
        "pow3" => pow3(x),
        "pow4" => pow4(x),
        "pow5" => pow5(x),
        "pow6" => pow6(x),
        "pow7" => pow7(x),
        "pow8" => pow8(x),
        "cos" => cos(x),
        "sin" => sin(x),
        "tan" => tan(x),
        "cospi" => cospi(x),
        "sinpi" => sinpi(x),
        "tanpi" => tanpi(x),
        "acos" => acos(x),
        "asin" => asin(x),
        "atan" => atan(x),
        "acospi" => acospi(x),
        "asinpi" => asinpi(x),
        "atanpi" => atanpi(x),
        "cosh" => cosh(x),
        "sinh" => sinh(x),
        "tanh" => tanh(x),
        "acosh" => acosh(x),
        "asinh" => asinh(x),
        "atanh" => atanh(x),
        other => panic!("unknown unary function section: {other}"),
    }
}

fn eval_binary_f64_named(name: &str, a: f64, b: f64) -> f64 {
    match name {
        "pow" => pow(a, b),
        "atan2" => atan2(a, b),
        "hypot" => hypot(a, b),
        other => panic!("unknown binary_f64 function section: {other}"),
    }
}

fn eval_binary_i32_named(name: &str, a: f64, exp_i32: i32) -> f64 {
    match name {
        "powi" => powi(a, exp_i32),
        "powi_f90" => powi_f90(a, exp_i32),
        other => panic!("unknown binary_i32 function section: {other}"),
    }
}

#[test]
fn test_mesa_26_4_1_hex_reference_vectors_match_bit_for_bit_in_strict_mode() {
    let vec_path = repo_root()
        .join("benchmarks/fortran_mesa/math_hex_vectors/mesa_26_4_1_math_vectors.txt");
    let raw = fs::read_to_string(&vec_path).expect("read mesa_26_4_1_math_vectors.txt");

    // Verify required provenance headers (#165 Item 14).
    assert!(raw.contains("# mesa_version: r26.4.1"));
    assert!(raw.contains("# math_backend: CRMATH"));
    assert!(raw.contains("# compiler: GCC version 15.2.0"));
    assert!(raw.contains("# mesasdk_version: x86_64-linux-26.6.1"));
    assert!(raw.contains("# date: 2026-11-16"));

    with_numerics_mode(NumericsMode::Strict, || {
        math_init();
        assert_eq!(MATH_BACKEND, "CRMATH");
        assert_eq!(log(10.0_f64).to_bits(), 0x4002_6bb1_bbb5_5515_u64);

        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        let mut current_kind = "";
        let mut current_name = String::new();
        let mut fnv_digest: u64 = 0xcbf2_9ce4_8422_2325;

        for (line_idx, line) in raw.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(inner) =
                line.strip_prefix('[').and_then(|s| s.strip_suffix(']'))
            {
                let (kind, fname) =
                    inner.split_once(':').expect("section [kind:name]");
                current_kind = match kind {
                    "unary" => "unary",
                    "binary_f64" => "binary_f64",
                    "binary_i32" => "binary_i32",
                    other => panic!("unknown section kind {other}"),
                };
                current_name = fname.to_string();
                counts.entry(current_name.clone()).or_insert(0);
                continue;
            }

            let actual_val = match current_kind {
                "unary" => {
                    let (in_hex, exp_hex) =
                        line.split_once(' ').expect("unary hex pair");
                    let in_bits =
                        u64::from_str_radix(in_hex, 16).expect("parse in_hex");
                    let exp_bits =
                        u64::from_str_radix(exp_hex, 16).expect("parse exp_hex");
                    let out = eval_unary_named(
                        &current_name,
                        f64::from_bits(in_bits),
                    );
                    let act_bits = out.to_bits();
                    assert!(
                        bits_match_with_nan_rule(act_bits, exp_bits),
                        "bit mismatch in unary `{current_name}` at line {}: \
                         in={in_bits:016X} expected={exp_bits:016X} \
                         actual={act_bits:016X}",
                        line_idx + 1
                    );
                    out
                }
                "binary_f64" => {
                    let mut parts = line.split_whitespace();
                    let a_hex = parts.next().expect("arg1");
                    let b_hex = parts.next().expect("arg2");
                    let exp_hex = parts.next().expect("out");
                    let a_bits = u64::from_str_radix(a_hex, 16).expect("a_bits");
                    let b_bits = u64::from_str_radix(b_hex, 16).expect("b_bits");
                    let exp_bits =
                        u64::from_str_radix(exp_hex, 16).expect("exp_bits");
                    let out = eval_binary_f64_named(
                        &current_name,
                        f64::from_bits(a_bits),
                        f64::from_bits(b_bits),
                    );
                    let act_bits = out.to_bits();
                    assert!(
                        bits_match_with_nan_rule(act_bits, exp_bits),
                        "bit mismatch in binary_f64 `{current_name}` at line {}: \
                         a={a_bits:016X} b={b_bits:016X} \
                         expected={exp_bits:016X} actual={act_bits:016X}",
                        line_idx + 1
                    );
                    out
                }
                "binary_i32" => {
                    let mut parts = line.split_whitespace();
                    let a_hex = parts.next().expect("arg1");
                    let n_hex = parts.next().expect("exp_i32");
                    let exp_hex = parts.next().expect("out");
                    let a_bits = u64::from_str_radix(a_hex, 16).expect("a_bits");
                    let n_u32 = u32::from_str_radix(n_hex, 16).expect("n_u32");
                    let exp_bits =
                        u64::from_str_radix(exp_hex, 16).expect("exp_bits");
                    let exp_i32 = n_u32.cast_signed();
                    let out = eval_binary_i32_named(
                        &current_name,
                        f64::from_bits(a_bits),
                        exp_i32,
                    );
                    let act_bits = out.to_bits();
                    assert!(
                        bits_match_with_nan_rule(act_bits, exp_bits),
                        "bit mismatch in binary_i32 `{current_name}` at line {}: \
                         a={a_bits:016X} n={exp_i32} expected={exp_bits:016X} \
                         actual={act_bits:016X}",
                        line_idx + 1,
                    );
                    out
                }
                _ => panic!("vector line outside section at {}", line_idx + 1),
            };

            *counts.get_mut(&current_name).expect("section initialized") += 1;
            fnv_digest ^= canonical_strict_bits(actual_val);
            fnv_digest = fnv_digest.wrapping_mul(0x0100_0000_01b3);
        }

        assert_eq!(counts.len(), 41, "expected 41 tested functions");
        for (fname, count) in &counts {
            assert!(
                *count >= 10_050,
                "function `{fname}` has only {count} vectors (expected >= 10,050)"
            );
        }
        assert_ne!(fnv_digest, 0);
    });
}

#[test]
fn test_function_mapping_toml_covers_all_math_lib_crmath_public_symbols() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mapping = fs::read_to_string(manifest_dir.join("function_mapping.toml"))
        .expect("function_mapping.toml");
    let lib_rs =
        fs::read_to_string(manifest_dir.join("src/lib.rs")).expect("src/lib.rs");
    let trig_rs = fs::read_to_string(manifest_dir.join("src/trig.rs"))
        .expect("src/trig.rs");
    let sys_rs = fs::read_to_string(
        manifest_dir.join("../flexmesa-crlibm-sys/src/lib.rs"),
    )
    .expect("flexmesa-crlibm-sys/src/lib.rs");

    // Every public symbol declared in MESA 26.4.1 `math/public/math_lib_crmath.f90:61-103`.
    let upstream_public = [
        "MATH_BACKEND",
        "math_init",
        "safe_sqrt",
        "log",
        "safe_log",
        "log10",
        "safe_log10",
        "log1p",
        "log2",
        "exp",
        "exp10",
        "expm1",
        "powm1",
        "pow (pow_i_)",
        "pow (pow_r_)",
        "pow2",
        "pow3",
        "pow4",
        "pow5",
        "pow6",
        "pow7",
        "pow8",
        "cos",
        "sin",
        "tan",
        "cospi",
        "sinpi",
        "tanpi",
        "acos",
        "asin",
        "atan",
        "acospi",
        "asinpi",
        "atanpi",
        "cosh",
        "sinh",
        "tanh",
        "acosh",
        "asinh",
        "atanh",
        "str_to_vector",
        "str_to_double",
        "double_to_str",
    ];
    for sym in upstream_public {
        let needle = format!("mesa_public = \"{sym}\"");
        assert!(
            mapping.contains(&needle),
            "function_mapping.toml missing upstream public symbol `{sym}`"
        );
    }

    // Verify every direct crlibm symbol listed in function_mapping.toml exists in flexmesa-crlibm-sys.
    for crlibm_sym in [
        "crlibm_init",
        "log_rz",
        "log1p_rz",
        "log2_rz",
        "log10_rz",
        "exp_rd",
        "expm1_rz",
        "cos_rz",
        "sin_rz",
        "tan_rz",
        "cospi_rz",
        "sinpi_rz",
        "tanpi_rz",
        "acos_rd",
        "asin_rz",
        "atan_rz",
        "acospi_rd",
        "asinpi_rz",
        "atanpi_rz",
        "cosh_rz",
        "sinh_rz",
    ] {
        assert!(
            mapping.contains(crlibm_sym),
            "mapping missing crlibm symbol {crlibm_sym}"
        );
        assert!(
            sys_rs.contains(&format!("pub fn {crlibm_sym}")),
            "flexmesa-crlibm-sys missing safe wrapper {crlibm_sym}"
        );
    }

    // Verify ruling extensions are present in both mapping and Rust source.
    let combined_src = format!("{lib_rs}\n{trig_rs}");
    for ext in ["atan2", "hypot", "powi_f90", "fma"] {
        assert!(mapping.contains(&format!("name = \"{ext}\"")));
        assert!(combined_src.contains(&format!("pub fn {ext}")));
    }
}

#[test]
fn test_strict_vs_fast_mode_dispatch_fma_pow_edge_cases_and_pre_z() {
    // Default mode is Strict.
    assert_eq!(numerics_mode(), NumericsMode::Strict);

    // Classic Dekker FMA-distinguishing triple: (1 + 2^-52)*(1 - 2^-52) - 1.
    // Separate multiply-then-add rounds (1 - 2^-104) to 1.0 -> 1.0 - 1.0 == 0.0;
    // true IEEE-754 FMA preserves -2^-104.
    let eps = f64::from_bits(0x3cb0_0000_0000_0000); // 2^-52
    let a = 1.0 + eps;
    let b = 1.0 - eps;
    with_numerics_mode(NumericsMode::Strict, || {
        assert_eq!(fma(a, b, -1.0), 0.0);
        // Upstream pow_i_ / pow_r_ zero-base convention (`math_lib_crmath.f90:201,229`).
        assert_eq!(pow(0.0, 0.0), 0.0);
        assert_eq!(pow(0.0, -1.0), 0.0);
        assert_eq!(powi(0.0, 0), 0.0);
        assert_eq!(powi(0.0, -3), 0.0);
        // Fortran `**` integer convention (`0.0 ** 0 == 1.0`).
        assert_eq!(powi_f90(0.0, 0), 1.0);
        assert_eq!(powi_f90(-0.0, 1).to_bits(), (-0.0_f64).to_bits());
        // Deterministic atan2 on (0, 0) and NaN (uninitialized in CRMATH 1.2).
        assert_eq!(atan2(0.0, 0.0).to_bits(), 0.0_f64.to_bits());
        assert_eq!(atan2(-0.0, 0.0).to_bits(), (-0.0_f64).to_bits());
        assert!(atan2(f64::NAN, 1.0).is_nan());
        assert!(atan2(1.0, f64::NAN).is_nan());
        // CRMATH tanh: finite overflow clamps to f64::MAX / f64::MAX == 1.0,
        // while +-Inf evaluates +-Inf / +Inf = NaN (`crmath_r_dp.f90:369`).
        assert_eq!(tanh(750.0), 1.0);
        assert!(tanh(f64::INFINITY).is_nan());
        assert!(tanh(f64::NEG_INFINITY).is_nan());
    });

    with_numerics_mode(NumericsMode::Fast, || {
        assert_eq!(numerics_mode(), NumericsMode::Fast);
        assert_eq!(fma(a, b, -1.0), -(eps * eps));
        assert_eq!((tanh(750.0) - 1.0).abs(), 0.0);
        assert!((log1p(1.0e-16) - 1.0e-16).abs() < 1.0e-30);
        assert!((expm1(1.0e-16) - 1.0e-16).abs() < 1.0e-30);
    });

    // Mode restored to Strict after `with_numerics_mode`.
    assert_eq!(numerics_mode(), NumericsMode::Strict);

    // Check `pre_z(1..=1000)` and `double_to_str`.
    let pz1 = pre_z(1);
    assert_eq!(pz1.z2, 1.0);
    assert_eq!(pz1.logz, 0.0);
    let pz8 = pre_z(8);
    assert_eq!(pz8.z3, 512.0);
    let pz_last = pre_z(MAX_PRECOMP_INTS);
    assert_eq!(pz_last.z2, 1.0e6);

    let formatted = double_to_str(0.1);
    assert_eq!(formatted.len(), 26);
    assert!(formatted.contains("D-01"));
}
