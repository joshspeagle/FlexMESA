// SPDX-License-Identifier: GPL-3.0-or-later
//! Level-2 IEEE-754 module-hex parity tests for `flexmesa-mtx` against MESA
//! `26.4.1` `mtx/test/test_output` (`exact_tested_values.tsv`, 22 outputs).

#![forbid(unsafe_code)]
#![allow(
    clippy::approx_constant,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use std::path::PathBuf;

use flexmesa_math::{with_numerics_mode, NumericsMode};
use flexmesa_mtx::{dgetrf, dgetrs, dgetrs_multi, multiply_xa};
use flexmesa_utils::{require_zero_skips_enabled, resolve_flexmesa_data_dir};

#[derive(Debug, Clone)]
struct HexRow {
    src_loc: String,
    label: String,
    raw_bits: u64,
    value: f64,
}

fn signed_magnitude(bits: u64) -> i128 {
    if (bits >> 63) != 0 {
        -i128::from(bits ^ (1_u64 << 63))
    } else {
        i128::from(bits)
    }
}

fn ulp_distance(actual: f64, expected_bits: u64) -> u64 {
    let expected = f64::from_bits(expected_bits);
    if actual.is_nan() || expected.is_nan() {
        return if actual.is_nan() && expected.is_nan() {
            0
        } else {
            u64::MAX
        };
    }
    let a = signed_magnitude(actual.to_bits());
    let b = signed_magnitude(expected_bits);
    u64::try_from((a - b).unsigned_abs()).unwrap_or(u64::MAX)
}

fn parse_hex_bits(raw: &str) -> u64 {
    let trimmed = raw
        .trim()
        .strip_prefix("0x")
        .or_else(|| raw.trim().strip_prefix("0X"))
        .unwrap_or_else(|| {
            panic!("expected 0x-prefixed hex bits, got {raw:?}")
        });
    u64::from_str_radix(trimmed, 16)
        .unwrap_or_else(|e| panic!("invalid hex bits {raw:?}: {e}"))
}

fn resolve_module_hex_tsv() -> Option<PathBuf> {
    if let Some(explicit) =
        std::env::var_os("FLEXMESA_MODULE_HEX_DIR").filter(|v| !v.is_empty())
    {
        let path = PathBuf::from(explicit)
            .join("mtx")
            .join("exact_tested_values.tsv");
        if path.exists() {
            return Some(path);
        }
        assert!(
            !require_zero_skips_enabled(),
            "missing required MESA 26.4.1 module-hex file `{}` (set \
             FLEXMESA_MODULE_HEX_DIR, run `flexmesa data fetch \
             --include-module-hex`, or set FLEXMESA_REQUIRE_ZERO_SKIPS=0 locally)",
            path.display()
        );
        eprintln!(
            "NOT RUN (FLEXMESA_REQUIRE_ZERO_SKIPS opted out): missing {}; \
             the calling test checks nothing",
            path.display()
        );
        return None;
    }

    let mut candidates = Vec::new();
    if let Some(parent) = resolve_flexmesa_data_dir().parent() {
        candidates.push(parent.join("mesa_module_hex"));
    }
    if let Some(home) = std::env::var_os("HOME").filter(|v| !v.is_empty()) {
        candidates.push(
            PathBuf::from(home)
                .join(".cache")
                .join("flexmesa")
                .join("mesa_module_hex"),
        );
    }
    candidates.push(PathBuf::from(".mesa_module_hex"));

    for root in &candidates {
        let path = root.join("mtx").join("exact_tested_values.tsv");
        if path.exists() {
            return Some(path);
        }
    }

    let primary = candidates[0].join("mtx").join("exact_tested_values.tsv");
    assert!(
        !require_zero_skips_enabled(),
        "missing required MESA 26.4.1 module-hex file `{}` (set \
         FLEXMESA_MODULE_HEX_DIR, run `flexmesa data fetch \
         --include-module-hex`, or set FLEXMESA_REQUIRE_ZERO_SKIPS=0 locally)",
        primary.display()
    );
    eprintln!(
        "NOT RUN (FLEXMESA_REQUIRE_ZERO_SKIPS opted out): missing {}; the \
         calling test checks nothing",
        primary.display()
    );
    None
}

fn load_module_hex_rows(path: &std::path::Path) -> Vec<HexRow> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let mut rows = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with('#')
            || trimmed.starts_with("record_idx\t")
        {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 8, "expected 8 columns in TSV line: {line:?}");
        let raw_bits = parse_hex_bits(cols[7]);
        rows.push(HexRow {
            src_loc: cols[2].trim().to_string(),
            label: cols[3].trim().to_string(),
            raw_bits,
            value: f64::from_bits(raw_bits),
        });
    }
    rows
}

fn transpose_square(n: usize, a: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), n * n);
    let mut out = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            out[j * n + i] = a[i * n + j];
        }
    }
    out
}

#[test]
fn mesa_26_4_1_mtx_module_hex_zero_ulp() {
    let Some(tsv_path) = resolve_module_hex_tsv() else {
        return;
    };
    let rows = load_module_hex_rows(&tsv_path);
    assert_eq!(rows.len(), 22, "expected 22 data rows in mtx hex TSV");

    with_numerics_mode(NumericsMode::Strict, || {
        let mut checked = 0_usize;
        let mut zeros = 0_usize;
        let mut max_ulp = 0_u64;
        let mut failures: Vec<String> = Vec::new();

        let mut record = |row_idx: usize,
                          ctx: &str,
                          expected_src_loc: &str,
                          expected_label: &str,
                          got: f64| {
            let want = &rows[row_idx];
            assert_eq!(
                want.src_loc, expected_src_loc,
                "TSV src_loc mismatch at row[{row_idx:02}] ({ctx})"
            );
            assert_eq!(
                want.label, expected_label,
                "TSV label mismatch at row[{row_idx:02}] ({ctx})"
            );
            let ulp = ulp_distance(got, want.raw_bits);
            checked += 1;
            if ulp == 0 {
                zeros += 1;
            } else {
                max_ulp = max_ulp.max(ulp);
                failures.push(format!(
                    "  row[{row_idx:02}] [{ctx}] {} ({}): got={got:.17e} \
                     ({:#018x}) want={:.17e} ({:#018x}) ulp={ulp}",
                    want.label,
                    want.src_loc,
                    got.to_bits(),
                    want.value,
                    want.raw_bits
                ));
            }
        };

        let mut row_cursor = 0_usize;

        // 1. `test_square1` (`mtx/test/src/test_square.f90:140-171`, 4 rows).
        {
            const N: usize = 4;
            let mut a = [
                1.80_f64, 2.88, 2.05, 0.00, //
                5.25, -2.95, -0.95, -3.80, //
                0.00, 0.00, -2.90, -1.04, //
                -1.11, 0.00, -0.59, 0.80, //
            ];
            let b_init = [4.35_f64, 5.05, 3.04, -2.05];
            let mut b_multi = b_init;
            let mut b_single = b_init;
            let mut ipiv = [0_usize; N];

            dgetrf(N, &mut a, &mut ipiv).expect("test_square1 dgetrf");
            dgetrs_multi(N, 1, &a, &ipiv, &mut b_multi)
                .expect("test_square1 dgetrs_multi");
            dgetrs(N, &a, &ipiv, &mut b_single).expect("test_square1 dgetrs");
            for i in 0..N {
                assert_eq!(
                    b_single[i].to_bits(),
                    b_multi[i].to_bits(),
                    "test_square1 dgetrs vs dgetrs_multi bit mismatch at {i}"
                );
                record(
                    row_cursor,
                    "test_square1 B",
                    "test_square.f90:168",
                    "B",
                    b_multi[i],
                );
                row_cursor += 1;
            }
        }
        assert_eq!(row_cursor, 4);

        // 2. `test_square2` (`mtx/test/src/test_square.f90:95-138`, 6 rows).
        {
            const N: usize = 3;
            let mut a1 = [
                3.14_f64, 7.50, 0.00, //
                4.10, 3.20, 0.30, //
                0.00, 1.00, 4.10, //
            ];
            let mut b1_multi = [1.0_f64, 2.0, 3.0];
            let mut b1_single = b1_multi;
            let mut ipiv1 = [0_usize; N];

            dgetrf(N, &mut a1, &mut ipiv1).expect("test_square2 A1 dgetrf");
            dgetrs_multi(N, 1, &a1, &ipiv1, &mut b1_multi)
                .expect("test_square2 B1 dgetrs_multi");
            dgetrs(N, &a1, &ipiv1, &mut b1_single)
                .expect("test_square2 B1 dgetrs");
            for i in 0..N {
                assert_eq!(
                    b1_single[i].to_bits(),
                    b1_multi[i].to_bits(),
                    "test_square2 B1 dgetrs vs dgetrs_multi bit mismatch at {i}"
                );
                record(
                    row_cursor,
                    "test_square2 B1",
                    "test_square.f90:125",
                    "B1",
                    b1_multi[i],
                );
                row_cursor += 1;
            }

            let mut a2 = [
                4.70_f64, 6.20, 0.00, //
                3.20, 0.00, 0.31, //
                0.00, 3.10, 0.00, //
            ];
            let mut b2_multi = [1.1_f64, 2.1, 3.1];
            let mut b2_single = b2_multi;
            let mut ipiv2 = [0_usize; N];

            dgetrf(N, &mut a2, &mut ipiv2).expect("test_square2 A2 dgetrf");
            dgetrs_multi(N, 1, &a2, &ipiv2, &mut b2_multi)
                .expect("test_square2 B2 dgetrs_multi");
            dgetrs(N, &a2, &ipiv2, &mut b2_single)
                .expect("test_square2 B2 dgetrs");
            for i in 0..N {
                assert_eq!(
                    b2_single[i].to_bits(),
                    b2_multi[i].to_bits(),
                    "test_square2 B2 dgetrs vs dgetrs_multi bit mismatch at {i}"
                );
                record(
                    row_cursor,
                    "test_square2 B2",
                    "test_square.f90:135",
                    "B2",
                    b2_multi[i],
                );
                row_cursor += 1;
            }
        }
        assert_eq!(row_cursor, 10);

        // 3. `test_square_inv` (`mtx/test/src/test_square.f90:36-93`, 12 rows).
        // Upstream calls `dgemv('N', n, n, 1d0, A_init, n, X, 1, 0d0, prod, 1)`
        // (`mtx/test/src/test_square.f90:73, 89`). Passing `A_init^T` to
        // `multiply_xa` computes `x^T * A_init^T = A_init * x` in the exact
        // same column-major accumulation order.
        {
            const N: usize = 3;
            let a1_init = [
                3.14_f64, 7.50, 0.00, //
                4.10, 3.20, 0.30, //
                0.00, 1.00, 4.10, //
            ];
            let mut a1 = a1_init;
            let b1 = [1.0_f64, 2.0, 3.0];
            let mut ipiv1 = [0_usize; N];
            dgetrf(N, &mut a1, &mut ipiv1).expect("test_square_inv A1 dgetrf");
            let mut x1 = b1;
            dgetrs(N, &a1, &ipiv1, &mut x1).expect("test_square_inv A1 dgetrs");

            for &val in &b1 {
                record(
                    row_cursor,
                    "test_square_inv B1",
                    "test_square.f90:71",
                    "B1",
                    val,
                );
                row_cursor += 1;
            }
            let a1_t = transpose_square(N, &a1_init);
            let mut prod1 = [0.0_f64; N];
            multiply_xa(N, &a1_t, &x1, &mut prod1)
                .expect("test_square_inv A1_init*X");
            for &val in &prod1 {
                record(
                    row_cursor,
                    "test_square_inv A1_init*X",
                    "test_square.f90:74",
                    "A1_init*X",
                    val,
                );
                row_cursor += 1;
            }

            let a2_init = [
                0.00_f64, 3.10, 0.00, //
                4.70, 6.20, 0.00, //
                3.20, 0.00, 0.31, //
            ];
            let mut a2 = a2_init;
            let b2 = [1.1_f64, 2.1, 3.1];
            let mut ipiv2 = [0_usize; N];
            dgetrf(N, &mut a2, &mut ipiv2).expect("test_square_inv A2 dgetrf");
            let mut x2 = b2;
            dgetrs(N, &a2, &ipiv2, &mut x2).expect("test_square_inv A2 dgetrs");

            for &val in &b2 {
                record(
                    row_cursor,
                    "test_square_inv B2",
                    "test_square.f90:87",
                    "B2",
                    val,
                );
                row_cursor += 1;
            }
            let a2_t = transpose_square(N, &a2_init);
            let mut prod2 = [0.0_f64; N];
            multiply_xa(N, &a2_t, &x2, &mut prod2)
                .expect("test_square_inv A2_init*X");
            for &val in &prod2 {
                record(
                    row_cursor,
                    "test_square_inv A2_init*X",
                    "test_square.f90:90",
                    "A2_init*X",
                    val,
                );
                row_cursor += 1;
            }
        }
        assert_eq!(row_cursor, 22);
        assert_eq!(checked, 22);

        eprintln!(
            "mesa_26_4_1_mtx_module_hex_zero_ulp: {zeros}/{checked} at 0 ULP \
             (max_ulp={max_ulp})"
        );
        for line in &failures {
            eprintln!("{line}");
        }
        assert_eq!(
            max_ulp, 0,
            "{zeros}/{checked} at 0 ULP (max_ulp={max_ulp})"
        );
    });
}
