// SPDX-License-Identifier: LGPL-3.0-or-later
//! Level-2 IEEE-754 module-hex parity tests for `flexmesa-interp1d` against
//! MESA `26.4.1` `interp_1d/test/test_output` (`exact_tested_values.tsv`,
//! 234 rows: 117 `f64` rows from `interp_1d_support.f90` and 117 unported
//! `f32` `_sg` rows from `interp_1d_support_sg.f90`).

#![forbid(unsafe_code)]
#![allow(
    clippy::approx_constant,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use std::f64::consts::PI;
use std::path::{Path, PathBuf};

use flexmesa_interp1d::{
    integrate_values, interp_3_to_2, interp_4pt_pm, interp_m3q, interp_pm,
    interp_values,
};
use flexmesa_math::{cos, sin, with_numerics_mode, NumericsMode};
use flexmesa_utils::{require_zero_skips_enabled, resolve_flexmesa_data_dir};

#[derive(Debug, Clone)]
struct HexRow {
    src_loc: String,
    label: String,
    kind: String,
    raw_bits: u64,
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
    let root = std::env::var_os("FLEXMESA_MODULE_HEX_DIR")
        .filter(|v| !v.is_empty())
        .map_or_else(
            || {
                resolve_flexmesa_data_dir().parent().map_or_else(
                    || PathBuf::from(".mesa_module_hex"),
                    |p| p.join("mesa_module_hex"),
                )
            },
            PathBuf::from,
        );
    let path = root.join("interp_1d").join("exact_tested_values.tsv");
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
        "NOT RUN (FLEXMESA_REQUIRE_ZERO_SKIPS opted out): missing {}; the \
         calling test checks nothing",
        path.display()
    );
    None
}

fn load_module_hex_rows(path: &Path) -> Vec<HexRow> {
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
        rows.push(HexRow {
            src_loc: cols[2].trim().to_string(),
            label: cols[3].trim().to_string(),
            kind: cols[4].trim().to_string(),
            raw_bits: parse_hex_bits(cols[7]),
        });
    }
    rows
}

/// Exact `U(1:100)` double-precision data from `interp_1d_support.f90:55-154`.
#[rustfmt::skip]
const TEST_U: [f64; 100] = [
    0.3691382918814334, 0.3695899829574669, 0.3699633032242006,
    0.3709860746618878, 0.3724175838188959, 0.3742482756438662,
    0.3764768199547272, 0.3790910321656262, 0.3820871425096383,
    0.3854500131142883, 0.3891727128876379, 0.3932377816410259,
    0.3976355585696444, 0.4023461172079230, 0.4073570798427897,
    0.4126461990878889, 0.4181985492965749, 0.4239897218739848,
    0.4300024071530404, 0.4362102270543894, 0.4425936978527515,
    0.4491247203356548, 0.4557819260544076, 0.4625358769752868,
    0.4693638888826394, 0.4762362700345437, 0.4831315999154387,
    0.4900124769496845, 0.4968509249135122, 0.5036231656397859,
    0.5102978108992456, 0.5168503329341820, 0.5232493058916751,
    0.5294708374154181, 0.5354843215442489, 0.5412671682413143,
    0.5467901937111218, 0.5520326792853251, 0.5569674077513700,
    0.5615760727117703, 0.5658339305000317, 0.5697255981615711,
    0.5732292848999179, 0.5763330504836860, 0.5790185013283228,
    0.5812774415045446, 0.5830949206727917, 0.5844671751637984,
    0.5853828024764719, 0.5858432670435806, 0.5858432670435451,
    0.5853828024764413, 0.5844671751637710, 0.5830949206727681,
    0.5812774415045240, 0.5790185013283052, 0.5763330504836706,
    0.5732292848999045, 0.5697255981615594, 0.5658339305000216,
    0.5615760727117616, 0.5569674077513626, 0.5520326792853185,
    0.5467901937111161, 0.5412671682413093, 0.5354843215442447,
    0.5294708374154145, 0.5232493058916718, 0.5168503329341793,
    0.5102978108992431, 0.5036231656397836, 0.4968509249135103,
    0.4900124769496828, 0.4831315999154373, 0.4762362700345427,
    0.4693638888826384, 0.4625358769752859, 0.4557819260544067,
    0.4491247203356541, 0.4425936978527509, 0.4362102270543888,
    0.4300024071530398, 0.4239897218739841, 0.4181985492965744,
    0.4126461990878886, 0.4073570798427895, 0.4023461172079228,
    0.3976355585696443, 0.3932377816410258, 0.3891727128876379,
    0.3854500131142883, 0.3820871425096385, 0.3790910321656264,
    0.3764768199547278, 0.3742482756438669, 0.3724175838188968,
    0.3709860746618888, 0.3699633032242018, 0.3695899829574566,
    0.3691382918814347,
];

fn assert_row_zero_ulp(row_idx: usize, actual: f64, row: &HexRow, tag: &str) {
    let ulps = ulp_distance(actual, row.raw_bits);
    assert_eq!(
        ulps,
        0,
        "row {row_idx} ({tag}, {} `{}`): got {actual:.17e} (0x{:016x}), \
         expected {:.17e} (0x{:016x}), ulps={ulps}",
        row.src_loc,
        row.label,
        actual.to_bits(),
        f64::from_bits(row.raw_bits),
        row.raw_bits
    );
}

fn check_pm_grid_case(
    increasing: bool,
    xs_eval: &[f64],
    include_f2_3: bool,
    rows: &[HexRow],
    start: usize,
    tag: &str,
) -> usize {
    let dx = 0.594059405940594_f64;
    let init_xs: Vec<f64> = (0..100)
        .map(|i| {
            let v = dx * f64::from(i);
            if increasing {
                v
            } else {
                -v
            }
        })
        .collect();
    let coeffs = interp_pm(&init_xs, &TEST_U).unwrap();
    let xs_q: Vec<f64> = xs_eval
        .iter()
        .map(|&v| if increasing { v } else { -v })
        .collect();
    let vals = interp_values(&init_xs, &coeffs, &xs_q).unwrap();

    let mut actuals = vec![coeffs[0][1], coeffs[1][1]];
    if include_f2_3 {
        actuals.push(coeffs[2][1]);
    }
    actuals.extend_from_slice(&vals);

    for (k, &actual) in actuals.iter().enumerate() {
        let idx = start + k;
        assert_row_zero_ulp(idx, actual, &rows[idx], tag);
    }
    start + actuals.len()
}

fn check_m3q_grid_case(
    increasing: bool,
    rows: &[HexRow],
    start: usize,
    tag: &str,
) -> usize {
    let dx = 0.594059405940594_f64;
    let init_xs: Vec<f64> = (0..100)
        .map(|i| {
            let v = dx * f64::from(i);
            if increasing {
                v
            } else {
                -v
            }
        })
        .collect();
    let coeffs = interp_m3q(&init_xs, &TEST_U).unwrap();
    let xs_raw = [10.0_f64, 10.4, 10.8, 11.0];
    let xs_q = xs_raw.map(|v| if increasing { v } else { -v });
    let vals = interp_values(&init_xs, &coeffs, &xs_q).unwrap();

    let actuals = [
        coeffs[0][1],
        coeffs[1][1],
        coeffs[2][1],
        vals[0],
        vals[1],
        vals[2],
        vals[3],
    ];
    for (k, &actual) in actuals.iter().enumerate() {
        let idx = start + k;
        assert_row_zero_ulp(idx, actual, &rows[idx], tag);
    }
    start + actuals.len()
}

fn check_test2_sin_case(
    increasing: bool,
    rows: &[HexRow],
    start: usize,
    tag: &str,
) -> usize {
    let n = 43usize;
    let dx = 1.0_f64 / ((n - 1) as f64);
    let mut init_xs = Vec::with_capacity(n);
    let mut u = Vec::with_capacity(n);
    for i in 0..n {
        let x0 = 4.0 * dx * (i as f64);
        let x = if increasing { x0 } else { -x0 };
        init_xs.push(x);
        u.push(sin(x));
    }
    let coeffs = interp_m3q(&init_xs, &u).unwrap();

    let xs_base = [
        0.0_f64,
        PI / 4.0,
        PI / 3.0,
        PI / 2.0,
        2.0 * PI / 3.0,
        3.0 * PI / 4.0,
        19.0 * PI / 20.0,
    ];
    let xs = xs_base.map(|v| if increasing { v } else { -v });
    let vals = interp_values(&init_xs, &coeffs, &xs).unwrap();

    let mut idx = start;
    for i in 0..xs.len() {
        assert_row_zero_ulp(idx, xs[i], &rows[idx], tag);
        idx += 1;
        assert_row_zero_ulp(idx, vals[i], &rows[idx], tag);
        idx += 1;
        assert_row_zero_ulp(idx, sin(xs[i]), &rows[idx], tag);
        idx += 1;
    }

    let integrals = integrate_values(&init_xs, &coeffs, &xs).unwrap();
    for i in 1..xs.len() {
        assert_row_zero_ulp(idx, xs[i], &rows[idx], tag);
        idx += 1;
        assert_row_zero_ulp(idx, integrals[i], &rows[idx], tag);
        idx += 1;
        let exact = cos(xs[i - 1]) - cos(xs[i]);
        assert_row_zero_ulp(idx, exact, &rows[idx], tag);
        idx += 1;
    }
    idx
}

#[test]
fn test_interp1d_module_hex_zero_ulp() {
    let Some(path) = resolve_module_hex_tsv() else {
        return;
    };
    let rows = load_module_hex_rows(&path);
    assert_eq!(rows.len(), 234, "unexpected interp_1d TSV row count");

    with_numerics_mode(NumericsMode::Strict, || {
        let mut idx = 0usize;

        // Rows 0..8: test_min(.true.) and test_min(.false.)
        idx = check_pm_grid_case(
            true,
            &[10.0, 10.8],
            false,
            &rows,
            idx,
            "test_min_asc",
        );
        idx = check_pm_grid_case(
            false,
            &[10.0, 10.8],
            false,
            &rows,
            idx,
            "test_min_desc",
        );
        assert_eq!(idx, 8);

        // Rows 8..22: test1(.true.) and test1(.false.)
        idx = check_m3q_grid_case(true, &rows, idx, "test1_asc");
        idx = check_m3q_grid_case(false, &rows, idx, "test1_desc");
        assert_eq!(idx, 22);

        // Rows 22..100: test2(.true.) and test2(.false.)
        idx = check_test2_sin_case(true, &rows, idx, "test2_asc");
        idx = check_test2_sin_case(false, &rows, idx, "test2_desc");
        assert_eq!(idx, 100);

        // Rows 100..112: test3(.true.) and test3(.false.)
        idx = check_pm_grid_case(
            true,
            &[10.0, 10.4, 10.8],
            true,
            &rows,
            idx,
            "test3_asc",
        );
        idx = check_pm_grid_case(
            false,
            &[10.0, 10.4, 10.8],
            true,
            &rows,
            idx,
            "test3_desc",
        );
        assert_eq!(idx, 112);

        // Rows 112..115: test_4pt
        let x6 = [
            PI / 4.0,
            PI / 3.0,
            PI / 2.0,
            2.0 * PI / 3.0,
            3.0 * PI / 4.0,
            19.0 * PI / 20.0,
        ];
        let y6 = x6.map(sin);
        let a = interp_4pt_pm(
            &[x6[1], x6[2], x6[3], x6[4]],
            &[y6[1], y6[2], y6[3], y6[4]],
        )
        .unwrap();
        let dx4 = (x6[3] - x6[2]) / 2.0;
        let res4 = y6[2] + dx4 * (a[0] + dx4 * (a[1] + dx4 * a[2]));
        let exact4 = sin(x6[2] + dx4);
        let err4 = ((res4 - exact4) / exact4).abs();
        assert_row_zero_ulp(idx, res4, &rows[idx], "test_4pt res");
        idx += 1;
        assert_row_zero_ulp(idx, exact4, &rows[idx], "test_4pt exact");
        idx += 1;
        assert_row_zero_ulp(idx, err4, &rows[idx], "test_4pt rel_err");
        idx += 1;
        assert_eq!(idx, 115);

        // Rows 115..117: test_interp_3_to_2
        let (left_val, right_val) = interp_3_to_2(
            2.0114947208182310e-3,
            2.0097307373083619e-3,
            1.5784933404892154e-3,
            1.3270582904786616e-3,
            1.8984458478374221e30,
            1.4579738233866260e30,
            1.1136297079131214e30,
        )
        .unwrap();
        assert_row_zero_ulp(
            idx,
            left_val,
            &rows[idx],
            "test_interp_3_to_2 nf00",
        );
        idx += 1;
        assert_row_zero_ulp(
            idx,
            right_val,
            &rows[idx],
            "test_interp_3_to_2 nfp1",
        );
        idx += 1;
        assert_eq!(idx, 117);

        // Rows 117..234: unported f32 (`_sg`) single-precision block
        let sg_rows = &rows[117..];
        assert_eq!(sg_rows.len(), 117);
        for (offset, row) in sg_rows.iter().enumerate() {
            assert_eq!(
                row.kind,
                "f32",
                "expected f32 at unported sg row {}",
                117 + offset
            );
            assert!(
                row.src_loc.starts_with("interp_1d_support_sg.f90:"),
                "unexpected src_loc {} at row {}",
                row.src_loc,
                117 + offset
            );
        }
    });
}
