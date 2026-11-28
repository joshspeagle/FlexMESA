// SPDX-License-Identifier: LGPL-3.0-or-later
//! Level-2 IEEE-754 module-hex parity tests for `flexmesa-interp2d` against
//! MESA `26.4.1` `interp_2d/test/test_output` (`exact_tested_values.tsv`,
//! 104 rows: 56 `f64`/`f64_arr` rows from `test2D_db` and 48 unported `f32`
//! `_sg` rows from `test2D_sg`).
//!
//! # Upstream `fvbicub_db` Implicit-`REAL` Quirk
//!
//! In MESA `26.4.1` `interp_2d/private/bicub_db.f:24,1371-1698` (deferred
//! upstream bug `#680`), local basis scalars (`xp`, `xpi`, `cx`, `cxi`, `yp`,
//! `ypi`, `cy`, `cyi`, `z36th`, `hx2`, `hy2`, `cxd`, `cxdi`, `cyd`, `cydi`)
//! omit `real*8` declarations and fall under Fortran implicit `REAL` (`f32`),
//! while accumulators remain `f64`. Production [`interp_evbicub_db`] reproduces
//! this implicit-`f32` basis evaluation at `0 ULP` across all 18 `fvbicub_db`
//! output rows `[4..34]` in `NumericsMode::Strict`, while preserving pure `f64`
//! precision (`<= 3e-8` abs, `<= 2.5e8` ULP vs MESA) in `NumericsMode::Fast`.

#![forbid(unsafe_code)]
#![allow(
    clippy::approx_constant,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use std::f64::consts::PI;
use std::path::{Path, PathBuf};

use flexmesa_interp2d::{
    interp_evbicub_db, interp_evbipm_db, interp_mkbicub_db, interp_mkbipm_db,
    BicubicBc,
};
use flexmesa_math::{cos, exp, sin, with_numerics_mode, NumericsMode};
use flexmesa_utils::{require_zero_skips_enabled, resolve_flexmesa_data_dir};

const NUM_XPTS: usize = 31;
const NUM_YPTS: usize = 35;

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
    let path = root.join("interp_2d").join("exact_tested_values.tsv");
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

fn test_fcn_db(x: f64, y: f64) -> f64 {
    let r = (x * x + y * y).sqrt();
    let r0 = PI / 2.0;
    exp(-r / r0) * sin(x) * cos(y)
}

fn build_mesa_2d_test_grid() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let xmin = -PI;
    let xmax = PI;
    let dx = (xmax - xmin) / ((NUM_XPTS - 1) as f64);
    let ymin = -PI;
    let ymax = PI;
    let dy = (ymax - ymin) / ((NUM_YPTS - 1) as f64);

    let mut xpts = vec![0.0_f64; NUM_XPTS];
    let mut ypts = vec![0.0_f64; NUM_YPTS];
    let mut z = vec![0.0_f64; NUM_XPTS * NUM_YPTS];
    for i in 0..NUM_XPTS {
        let x = xmin + (i as f64) * dx;
        xpts[i] = x;
        for j in 0..NUM_YPTS {
            let y = ymin + (j as f64) * dy;
            if i == 0 {
                ypts[j] = y;
            }
            z[i * NUM_YPTS + j] = test_fcn_db(x, y);
        }
    }
    (xpts, ypts, z)
}

#[test]
fn test_interp2d_module_hex_zero_ulp() {
    let Some(path) = resolve_module_hex_tsv() else {
        return;
    };
    let rows = load_module_hex_rows(&path);
    assert_eq!(rows.len(), 104, "unexpected interp_2d TSV row count");

    let (xpts, ypts, z) = build_mesa_2d_test_grid();

    let bcxmin = vec![0.0; NUM_YPTS];
    let bcxmax = vec![0.0; NUM_YPTS];
    let bcymin = vec![0.0; NUM_XPTS];
    let bcymax = vec![0.0; NUM_XPTS];
    let (f_bicub, ilinx, iliny) = interp_mkbicub_db(
        &xpts,
        &ypts,
        &z,
        BicubicBc::NotAKnot,
        &bcxmin,
        BicubicBc::NotAKnot,
        &bcxmax,
        BicubicBc::NotAKnot,
        &bcymin,
        BicubicBc::NotAKnot,
        &bcymax,
    )
    .unwrap();
    let f_bipm = interp_mkbipm_db(&xpts, &ypts, &z).unwrap();

    let x_points = 2usize;
    let y_points = 3usize;
    let x_max = 0.8_f64 * PI;
    let x_min = 0.1_f64;
    let y_max = 0.6_f64 * PI;
    let y_min = 0.2_f64;
    let dx = (x_max - x_min) / ((x_points - 1) as f64);
    let dy = (y_max - y_min) / ((y_points - 1) as f64);

    with_numerics_mode(NumericsMode::Strict, || {
        let mut db_vals: Vec<(&'static str, f64)> = Vec::with_capacity(56);
        let mid_bicub = f_bicub[14 * NUM_YPTS + 16];
        for (k, &v) in mid_bicub.iter().enumerate() {
            db_vals.push((
                match k {
                    0 => "bicub_mid_f",
                    1 => "bicub_mid_fxx",
                    2 => "bicub_mid_fyy",
                    _ => "bicub_mid_fxxyy",
                },
                v,
            ));
        }
        for j in 0..y_points {
            let y = y_min + (j as f64) * dy;
            for i in 0..x_points {
                let x = x_min + (i as f64) * dx;
                let ev = interp_evbicub_db(
                    x, y, &xpts, &ypts, ilinx, iliny, &f_bicub,
                )
                .unwrap();
                db_vals.push(("evbicub_x", x));
                db_vals.push(("evbicub_y", y));
                db_vals.push(("evbicub_z", ev.f));
                db_vals.push(("evbicub_dzdx", ev.df_dx));
                db_vals.push(("evbicub_dzdy", ev.df_dy));
            }
        }

        let mid_bipm = f_bipm[14 * NUM_YPTS + 16];
        for &v in &mid_bipm {
            db_vals.push(("bipm_mid", v));
        }
        for j in 0..y_points {
            let y = y_min + (j as f64) * dy;
            for i in 0..x_points {
                let x = x_min + (i as f64) * dx;
                let val =
                    interp_evbipm_db(x, y, &xpts, &ypts, &f_bipm).unwrap();
                db_vals.push(("evbipm_x", x));
                db_vals.push(("evbipm_y", y));
                db_vals.push(("evbipm_z", val));
            }
        }

        assert_eq!(db_vals.len(), 56);
        for (idx, ((name, actual), row)) in
            db_vals.iter().zip(rows[..56].iter()).enumerate()
        {
            let ulps = ulp_distance(*actual, row.raw_bits);
            assert_eq!(
                ulps,
                0,
                "row {idx} ({name}, {} `{}`): got={actual:.17e} \
                 (0x{:016x}) exp={:.17e} (0x{:016x}) ulps={ulps}",
                row.src_loc,
                row.label,
                actual.to_bits(),
                f64::from_bits(row.raw_bits),
                row.raw_bits
            );
        }
    });

    // Verify NumericsMode::Fast retains pure-f64 basis precision (bounded
    // against MESA's implicit-f32 outputs while differing from Strict).
    with_numerics_mode(NumericsMode::Fast, || {
        let mut fast_differs = 0usize;
        for j in 0..y_points {
            let y = y_min + (j as f64) * dy;
            for i in 0..x_points {
                let x = x_min + (i as f64) * dx;
                let ev = interp_evbicub_db(
                    x, y, &xpts, &ypts, ilinx, iliny, &f_bicub,
                )
                .unwrap();
                let base_idx = 4 + (j * x_points + i) * 5;
                for (off, actual) in [(2, ev.f), (3, ev.df_dx), (4, ev.df_dy)] {
                    let row = &rows[base_idx + off];
                    let ulps = ulp_distance(actual, row.raw_bits);
                    let exp_val = f64::from_bits(row.raw_bits);
                    let abs_err = (actual - exp_val).abs();
                    assert!(
                        abs_err <= 3.0e-8 && ulps <= 250_000_000,
                        "fvbicub fast-mode pure-f64 bound exceeded at row {} \
                         ({}): got={actual:.17e} exp={exp_val:.17e} \
                         abs_err={abs_err:.3e} ulps={ulps}",
                        base_idx + off,
                        row.label
                    );
                    if ulps > 0 {
                        fast_differs += 1;
                    }
                }
            }
        }
        assert!(
            fast_differs > 0,
            "NumericsMode::Fast must evaluate pure-f64 basis rather than Strict f32 rounding"
        );
    });

    // Account for rows 56..104: unported f32 (`_sg`) single-precision block
    assert_eq!(rows[56..].len(), 48);
    for (offset, row) in rows[56..].iter().enumerate() {
        assert_eq!(row.kind, "f32", "expected f32 at row {}", 56 + offset);
        assert!(
            row.src_loc.starts_with("test_interp_2d.f90:"),
            "unexpected src_loc {} at row {}",
            row.src_loc,
            56 + offset
        );
    }
}
