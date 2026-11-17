// SPDX-License-Identifier: LGPL-3.0-or-later
//! Exact 64-bit IEEE-754 bit-pattern verification of `flexmesa-math` in
//! `NumericsMode::Strict` against real MESA `26.4.1` (`r26.4.1`) reference
//! vectors (`>= 10,084` vectors per function across all 41 mapped functions;
//! `#165` Items 8, 9, 11, 14).

#![allow(clippy::float_cmp, clippy::too_many_lines)]

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use flexmesa_math::vector_eval::{
    canonical_strict_bits, eval_binary_f64_section, eval_binary_i32_section,
    eval_unary_section,
};
use flexmesa_math::{
    atan, atan2, cos, double_to_str, exp, expm1, fma, intrinsic_pow, log,
    log10, log1p, math_init, numerics_mode, pow, powi, powi_f90, pre_z, sin,
    sin_cos, str_to_double, str_to_vector, tanh, with_numerics_mode,
    NumericsMode, MATH_BACKEND, MAX_PRECOMP_INTS,
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

fn eval_unary_named(name: &str, x: f64) -> f64 {
    eval_unary_section(name, x).unwrap_or_else(|e| panic!("{e}"))
}

fn eval_binary_f64_named(name: &str, a: f64, b: f64) -> f64 {
    eval_binary_f64_section(name, a, b).unwrap_or_else(|e| panic!("{e}"))
}

fn is_platform_intrinsic_section(name: &str) -> bool {
    matches!(
        name,
        "dlog"
            | "dlog10"
            | "intrinsic_log10"
            | "intrinsic_exp"
            | "atan2"
            | "datan2"
            | "intrinsic_pow"
    )
}

fn eval_rz_counterpart(name: &str, a: f64, b: f64) -> f64 {
    match name {
        "dlog" => log(a),
        "dlog10" | "intrinsic_log10" => log10(a),
        "intrinsic_exp" => exp(a),
        "intrinsic_pow" => pow(a, b),
        "atan2" | "datan2" => {
            const PI_RZ: f64 = f64::from_bits(0x4009_21fb_5444_2d18);
            const HALFPI_RZ: f64 = f64::from_bits(0x3ff9_21fb_5444_2d18);
            const Q1_RZ: f64 = f64::from_bits(0x3fe9_21fb_5444_2d18);
            const Q3_RZ: f64 = f64::from_bits(0x4002_d97c_7f33_21d2);
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else if a == 0.0 && b == 0.0 {
                if !b.is_sign_negative() {
                    if a.is_sign_negative() {
                        -0.0
                    } else {
                        0.0
                    }
                } else if a.is_sign_negative() {
                    -PI_RZ
                } else {
                    PI_RZ
                }
            } else if a.is_infinite() && b.is_infinite() {
                let m = if b > 0.0 { Q1_RZ } else { Q3_RZ };
                if a < 0.0 {
                    -m
                } else {
                    m
                }
            } else if b > 0.0 {
                atan(a / b)
            } else if a > 0.0 {
                HALFPI_RZ - atan(b / a)
            } else if a < 0.0 {
                -HALFPI_RZ - atan(b / a)
            } else if a.is_sign_negative() {
                atan(a / b) - PI_RZ
            } else {
                atan(a / b) + PI_RZ
            }
        }
        other => panic!("not a platform intrinsic section: {other}"),
    }
}

#[inline]
fn ulp_distance(act_bits: u64, exp_bits: u64) -> u64 {
    if act_bits == exp_bits {
        return 0;
    }
    let act_val = f64::from_bits(act_bits);
    let exp_val = f64::from_bits(exp_bits);
    if !act_val.is_finite()
        || !exp_val.is_finite()
        || act_val.is_sign_negative() != exp_val.is_sign_negative()
    {
        return u64::MAX;
    }
    act_bits.abs_diff(exp_bits)
}

fn eval_binary_i32_named(name: &str, a: f64, exp_i32: i32) -> f64 {
    eval_binary_i32_section(name, a, exp_i32).unwrap_or_else(|e| panic!("{e}"))
}

#[test]
fn test_mesa_26_4_1_hex_reference_vectors_match_bit_for_bit_in_strict_mode() {
    let vec_path = repo_root().join(
        "benchmarks/fortran_mesa/math_hex_vectors/mesa_26_4_1_math_vectors.txt",
    );
    let raw = fs::read_to_string(&vec_path)
        .expect("read mesa_26_4_1_math_vectors.txt");

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
        let mut nearest_matches: BTreeMap<String, usize> = BTreeMap::new();
        let mut trunc_matches: BTreeMap<String, usize> = BTreeMap::new();
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
                    let exp_bits = u64::from_str_radix(exp_hex, 16)
                        .expect("parse exp_hex");
                    let x = f64::from_bits(in_bits);
                    let out = eval_unary_named(&current_name, x);
                    let act_bits = out.to_bits();
                    if is_platform_intrinsic_section(&current_name) {
                        if bits_match_with_nan_rule(act_bits, exp_bits) {
                            *nearest_matches
                                .entry(current_name.clone())
                                .or_insert(0) += 1;
                        } else {
                            let u = ulp_distance(act_bits, exp_bits);
                            assert!(
                                u <= 1,
                                "RN >1 ULP in `{current_name}` at line {}: \
                                 in={in_bits:016X} expected={exp_bits:016X} \
                                 actual={act_bits:016X} ulp={u}",
                                line_idx + 1
                            );
                        }
                        let trunc_out =
                            eval_rz_counterpart(&current_name, x, 0.0);
                        if bits_match_with_nan_rule(
                            trunc_out.to_bits(),
                            exp_bits,
                        ) {
                            *trunc_matches
                                .entry(current_name.clone())
                                .or_insert(0) += 1;
                        }
                    } else {
                        assert!(
                            bits_match_with_nan_rule(act_bits, exp_bits),
                            "bit mismatch in unary `{current_name}` at line {}: \
                             in={in_bits:016X} expected={exp_bits:016X} \
                             actual={act_bits:016X}",
                            line_idx + 1
                        );
                    }
                    out
                }
                "binary_f64" => {
                    let mut parts = line.split_whitespace();
                    let a_hex = parts.next().expect("arg1");
                    let b_hex = parts.next().expect("arg2");
                    let exp_hex = parts.next().expect("out");
                    let a_bits =
                        u64::from_str_radix(a_hex, 16).expect("a_bits");
                    let b_bits =
                        u64::from_str_radix(b_hex, 16).expect("b_bits");
                    let exp_bits =
                        u64::from_str_radix(exp_hex, 16).expect("exp_bits");
                    let (a, b) =
                        (f64::from_bits(a_bits), f64::from_bits(b_bits));
                    let out = eval_binary_f64_named(&current_name, a, b);
                    let act_bits = out.to_bits();
                    if is_platform_intrinsic_section(&current_name) {
                        if bits_match_with_nan_rule(act_bits, exp_bits) {
                            *nearest_matches
                                .entry(current_name.clone())
                                .or_insert(0) += 1;
                        } else {
                            let u = ulp_distance(act_bits, exp_bits);
                            assert!(
                                u <= 1,
                                "RN >1 ULP in `{current_name}` at line {}: \
                                 a={a_bits:016X} b={b_bits:016X} \
                                 expected={exp_bits:016X} actual={act_bits:016X} ulp={u}",
                                line_idx + 1
                            );
                        }
                        let trunc_out =
                            eval_rz_counterpart(&current_name, a, b);
                        if bits_match_with_nan_rule(
                            trunc_out.to_bits(),
                            exp_bits,
                        ) {
                            *trunc_matches
                                .entry(current_name.clone())
                                .or_insert(0) += 1;
                        }
                    } else {
                        assert!(
                            bits_match_with_nan_rule(act_bits, exp_bits),
                            "bit mismatch in binary_f64 `{current_name}` at line {}: \
                             a={a_bits:016X} b={b_bits:016X} \
                             expected={exp_bits:016X} actual={act_bits:016X}",
                            line_idx + 1
                        );
                    }
                    out
                }
                "binary_i32" => {
                    let mut parts = line.split_whitespace();
                    let a_hex = parts.next().expect("arg1");
                    let n_hex = parts.next().expect("exp_i32");
                    let exp_hex = parts.next().expect("out");
                    let a_bits =
                        u64::from_str_radix(a_hex, 16).expect("a_bits");
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

        assert_eq!(counts.len(), 47, "expected 47 tested sections");
        let total_vectors: usize = counts.values().sum();
        assert_eq!(
            total_vectors, 474_062,
            "expected 474,062 total reference vectors"
        );
        for (fname, count) in &counts {
            assert!(
                *count >= 10_050,
                "function `{fname}` has only {count} vectors (expected >= 10,050)"
            );
        }
        // Pin the measured RN vs RZ bit-match counts on all 7 platform-intrinsic
        // sections (`#196`, `#198`).
        for (sec, exp_total, expected_trunc, expected_nearest) in [
            ("dlog", 10_067_usize, 6_233_usize, 10_067_usize),
            ("dlog10", 10_067, 6_326, 10_051),
            ("intrinsic_log10", 10_067, 6_186, 10_043),
            ("intrinsic_exp", 10_067, 5_091, 10_067),
            ("atan2", 10_121, 7_011, 9_455),
            ("datan2", 10_121, 6_985, 9_473),
            ("intrinsic_pow", 10_294, 4_248, 10_291),
        ] {
            assert_eq!(counts[sec], exp_total, "{sec} total count");
            assert_eq!(
                trunc_matches[sec], expected_trunc,
                "{sec} RZ bit-match count"
            );
            assert_eq!(
                nearest_matches[sec], expected_nearest,
                "{sec} RN bit-match count"
            );
            assert!(nearest_matches[sec] > trunc_matches[sec]);
        }
        assert_ne!(fnv_digest, 0);
    });
}

#[test]
fn test_function_mapping_toml_covers_all_math_lib_crmath_public_symbols() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mapping =
        fs::read_to_string(manifest_dir.join("function_mapping.toml"))
            .expect("function_mapping.toml");
    let lib_rs = fs::read_to_string(manifest_dir.join("src/lib.rs"))
        .expect("src/lib.rs");
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
        "log_rn",
        "log10_rn",
        "exp_rn",
        "atan_rn",
        "pow_rn",
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
    for ext in [
        "atan2",
        "intrinsic_log",
        "intrinsic_log10",
        "intrinsic_exp",
        "intrinsic_pow",
        "hypot",
        "powi_f90",
        "fma",
    ] {
        assert!(mapping.contains(&format!("name = \"{ext}\"")));
        assert!(combined_src.contains(&format!("pub fn {ext}")));
    }

    // Bidirectional check: every `pub fn` in `flexmesa-math/src/` (excluding
    // mode-control and `auto_diff` sign/step helpers) must be documented in
    // `function_mapping.toml`.
    let mode_and_support_fns = [
        "numerics_mode",
        "set_numerics_mode",
        "with_numerics_mode",
        "sgn",
        "heaviside",
    ];
    for src_file in ["src/lib.rs", "src/trig.rs", "src/io.rs", "src/pre_z.rs"] {
        let src_txt =
            fs::read_to_string(manifest_dir.join(src_file)).expect(src_file);
        for line in src_txt.lines() {
            let trimmed = line.trim_start();
            if let Some(rest) = trimmed.strip_prefix("pub fn ") {
                let fn_name: String = rest
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect();
                if mode_and_support_fns.contains(&fn_name.as_str()) {
                    continue;
                }
                assert!(
                    mapping.contains(&format!("flexmesa_math::{fn_name}\"")),
                    "public function `flexmesa_math::{fn_name}` in `{src_file}` is missing from `function_mapping.toml`"
                );
            }
        }
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
    assert_eq!(formatted, "    1.0000000000000001D-01");
}

#[test]
fn test_intrinsic_pow_exact_rn_undecided_midpoint_pairs_never_return_sentinel()
{
    // Regression test (`#165` R1): upstream `crlibm 1.0beta4` `pow.c:763` returned
    // `-5.0` when both `pow_round_and_check_rn` and `pow_exact_case` remained
    // undecided near `x^y ~ 1 + 2^-59`. MESA SDK 26.6.1 `x ** y` evaluates to `1.0`.
    let pairs: [(u64, u64); 4] = [
        (0x3ff0_0000_0c00_0000, 0x3e25_5555_5d55_5554),
        (0x3ff0_0000_1800_0000, 0x3e15_5555_6555_5551),
        (0x3ff0_0000_2400_0000, 0x3e0c_71c7_3c71_c710),
        (0x3ff0_0000_3000_0000, 0x3e05_5555_7555_5545),
    ];
    with_numerics_mode(NumericsMode::Strict, || {
        for (xb, yb) in pairs {
            let x = f64::from_bits(xb);
            let y = f64::from_bits(yb);
            let got = intrinsic_pow(x, y);
            assert!(
                got > 0.0,
                "intrinsic_pow({xb:#018x}, {yb:#018x}) returned negative sentinel {got}"
            );
            assert_eq!(
                got.to_bits(),
                1.0_f64.to_bits(),
                "expected 1.0 (0x3ff0000000000000), got {got} ({:#018x})",
                got.to_bits()
            );
        }
    });
}

#[test]
fn test_math_io_double_to_str_str_to_double_and_sin_cos() {
    // Exact MESA SDK 26.6.1 `1PD26.16` formatting and bit-exact round-trip (`#165` R4).
    assert_eq!(double_to_str(0.0), "    0.0000000000000000D+00");
    assert_eq!(double_to_str(-0.0), "   -0.0000000000000000D+00");
    assert_eq!(double_to_str(0.1), "    1.0000000000000001D-01");
    assert_eq!(double_to_str(0.73), "    7.2999999999999998D-01");
    assert_eq!(double_to_str(1.0e-105), "    9.9999999999999997-106");
    assert_eq!(
        double_to_str(-1.234_567_890_123_456_7e120),
        "   -1.2345678901234567+120"
    );
    assert_eq!(double_to_str(f64::MAX), "    1.7976931348623157+308");
    assert_eq!(
        double_to_str(f64::from_bits(1)),
        "    4.9406564584124654-324"
    );
    assert_eq!(double_to_str(f64::INFINITY), "                  Infinity");
    assert_eq!(
        double_to_str(f64::NEG_INFINITY),
        "                 -Infinity"
    );
    assert_eq!(double_to_str(f64::NAN), "                       NaN");

    for x in [
        0.0_f64,
        -0.0_f64,
        0.1,
        0.73,
        std::f64::consts::PI,
        1.0e-105,
        -1.234_567_890_123_456_7e120,
        f64::MAX,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        let s = double_to_str(x);
        assert_eq!(s.len(), 26, "len != 26 for {x:?}: {s:?}");
        let parsed = str_to_double(&s)
            .unwrap_or_else(|e| panic!("parse failed for {s:?}: {e}"));
        assert_eq!(
            parsed.to_bits(),
            x.to_bits(),
            "round-trip bit mismatch on {x:?} via {s:?}"
        );
    }
    assert!(str_to_double(&double_to_str(f64::NAN)).unwrap().is_nan());

    let vec_in = format!(
        "{} {} {}",
        double_to_str(0.73),
        double_to_str(1.0e-105),
        double_to_str(-1.234_567_890_123_456_7e120)
    );
    let vec_out =
        str_to_vector(&vec_in).expect("str_to_vector 3-digit exponents");
    assert_eq!(vec_out.len(), 3);
    assert_eq!(vec_out[0].to_bits(), 0.73_f64.to_bits());
    assert_eq!(vec_out[1].to_bits(), 1.0e-105_f64.to_bits());
    assert_eq!(
        vec_out[2].to_bits(),
        (-1.234_567_890_123_456_7e120_f64).to_bits()
    );

    // Verify `sin_cos` matches `(sin(x), cos(x))` bit-for-bit in `Strict` mode.
    with_numerics_mode(NumericsMode::Strict, || {
        for x in [-3.7_f64, -0.5, 0.0, 0.73, 1.234_567_89, 100.0] {
            let (s, c) = sin_cos(x);
            assert_eq!(s.to_bits(), sin(x).to_bits());
            assert_eq!(c.to_bits(), cos(x).to_bits());
        }
    });
}
