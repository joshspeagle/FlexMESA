// SPDX-License-Identifier: LGPL-3.0-or-later
//! Dedicated `NumericsMode::Fast` verification for the `flexmesa-interp2d`
//! bicubic evaluator (`issue #210`, `issue #672`, `DECISIONS #43`).
//!
//! In `NumericsMode::Strict` (default), `interp_evbicub_db` reproduces MESA
//! `26.4.1` `fvbicub_db`'s implicit-`REAL` (`f32`) basis scalars
//! (`interp_2d/private/bicub_db.f:24,1371-1698`, deferred upstream bug `#680`).
//! `NumericsMode::Fast` keeps the basis in pure `f64`, so exact polynomial
//! calculus holds to machine precision. These checks switch the process-wide
//! mode, so they run in this file, which Cargo builds as its own process.

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines
)]

use std::f64::consts::PI;
use std::path::{Path, PathBuf};

use flexmesa_interp2d::{interp_evbicub_db, interp_mkbicub_db, BicubicBc};
use flexmesa_math::{cos, exp, sin, with_numerics_mode, NumericsMode};
use flexmesa_utils::{require_zero_skips_enabled, resolve_flexmesa_data_dir};
use proptest::prelude::*;

/// Number of x-grid knots in upstream MESA `test2D_db`.
const NUM_XPTS: usize = 31;

/// Number of y-grid knots in upstream MESA `test2D_db`.
const NUM_YPTS: usize = 35;

/// Query points of upstream `test2D_db`: `x_points = 2` over
/// `[0.1, 0.8 pi]` and `y_points = 3` over `[0.2, 0.6 pi]`.
const X_POINTS: usize = 2;
const Y_POINTS: usize = 3;
const X_MIN: f64 = 0.1;
const Y_MIN: f64 = 0.2;

fn x_max() -> f64 {
    0.8 * PI
}

fn y_max() -> f64 {
    0.6 * PI
}

/// Upstream MESA `test2D_db` test function
/// $f(x, y) = \exp(-r / r_0) \sin(x) \cos(y)$ with $r = \sqrt{x^2 + y^2}$ and
/// $r_0 = \pi / 2$.
fn test_fcn_db(x: f64, y: f64) -> f64 {
    let r = (x * x + y * y).sqrt();
    let r0 = PI / 2.0;
    exp(-r / r0) * sin(x) * cos(y)
}

/// The $31 \times 35$ uniform test grid `(xpts, ypts, z)` over $[-\pi, \pi]^2$
/// of upstream `test2D_db`, `z` row-major `[i * NUM_YPTS + j]`.
fn build_mesa_2d_test_grid() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let dx = (2.0 * PI) / ((NUM_XPTS - 1) as f64);
    let dy = (2.0 * PI) / ((NUM_YPTS - 1) as f64);
    let mut xpts = vec![0.0_f64; NUM_XPTS];
    let mut ypts = vec![0.0_f64; NUM_YPTS];
    let mut z = vec![0.0_f64; NUM_XPTS * NUM_YPTS];
    for i in 0..NUM_XPTS {
        let x = -PI + (i as f64) * dx;
        xpts[i] = x;
        for j in 0..NUM_YPTS {
            let y = -PI + (j as f64) * dy;
            if i == 0 {
                ypts[j] = y;
            }
            z[i * NUM_YPTS + j] = test_fcn_db(x, y);
        }
    }
    (xpts, ypts, z)
}

/// Not-a-knot bicubic spline of the `test2D_db` grid.
fn build_mesa_bicub() -> (Vec<f64>, Vec<f64>, Vec<[f64; 4]>, bool, bool) {
    let (xpts, ypts, z) = build_mesa_2d_test_grid();
    let bcx = vec![0.0; NUM_YPTS];
    let bcy = vec![0.0; NUM_XPTS];
    let (f, ilinx, iliny) = interp_mkbicub_db(
        &xpts,
        &ypts,
        &z,
        BicubicBc::NotAKnot,
        &bcx,
        BicubicBc::NotAKnot,
        &bcx,
        BicubicBc::NotAKnot,
        &bcy,
        BicubicBc::NotAKnot,
        &bcy,
    )
    .expect("bicubic setup succeeds");
    (xpts, ypts, f, ilinx, iliny)
}

/// `test2D_db` query point `(i, j)`.
fn query_point(i: usize, j: usize) -> (f64, f64) {
    let dx = (x_max() - X_MIN) / ((X_POINTS - 1) as f64);
    let dy = (y_max() - Y_MIN) / ((Y_POINTS - 1) as f64);
    (X_MIN + (i as f64) * dx, Y_MIN + (j as f64) * dy)
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

/// `(label, raw_bits)` of each data row of `exact_tested_values.tsv`.
fn load_module_hex_rows(path: &Path) -> Vec<(String, u64)> {
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
        rows.push((cols[3].trim().to_string(), parse_hex_bits(cols[7])));
    }
    rows
}

/// The 18 `fvbicub_db` output rows `[4..34]` of MESA `26.4.1`
/// `exact_tested_values.tsv`: in `NumericsMode::Fast`, the pure-`f64` basis
/// stays within `3e-8` absolute (`2.5e8` ULP) of MESA's implicit-`REAL`
/// values and differs from them in at least one row.
#[test]
fn test_fast_mode_fvbicub_rows_within_f64_basis_bound_of_mesa_and_not_strict() {
    let Some(path) = resolve_module_hex_tsv() else {
        return;
    };
    let rows = load_module_hex_rows(&path);
    assert_eq!(rows.len(), 104, "unexpected interp_2d TSV row count");
    let (xpts, ypts, f_bicub, ilinx, iliny) = build_mesa_bicub();

    with_numerics_mode(NumericsMode::Fast, || {
        let mut fast_differs = 0usize;
        for j in 0..Y_POINTS {
            for i in 0..X_POINTS {
                let (x, y) = query_point(i, j);
                let ev = interp_evbicub_db(
                    x, y, &xpts, &ypts, ilinx, iliny, &f_bicub,
                )
                .unwrap();
                let base_idx = 4 + (j * X_POINTS + i) * 5;
                for (off, actual) in [(2, ev.f), (3, ev.df_dx), (4, ev.df_dy)] {
                    let (label, raw_bits) = &rows[base_idx + off];
                    let ulps = ulp_distance(actual, *raw_bits);
                    let exp_val = f64::from_bits(*raw_bits);
                    let abs_err = (actual - exp_val).abs();
                    assert!(
                        abs_err <= 3.0e-8 && ulps <= 250_000_000,
                        "fvbicub fast-mode pure-f64 bound exceeded at row {} \
                         ({label}): got={actual:.17e} exp={exp_val:.17e} \
                         abs_err={abs_err:.3e} ulps={ulps}",
                        base_idx + off,
                    );
                    if ulps > 0 {
                        fast_differs += 1;
                    }
                }
            }
        }
        assert!(
            fast_differs > 0,
            "NumericsMode::Fast must evaluate pure-f64 basis rather than \
             Strict f32 rounding"
        );
    });
}

/// At the six `test2D_db` query points, the `NumericsMode::Fast` analytic
/// first partials equal centred finite differences (`h = 1e-6`) of the
/// value to `1e-10`.
#[test]
fn test_fast_mode_bicubic_first_partials_match_centred_differences() {
    let (xpts, ypts, f, ilinx, iliny) = build_mesa_bicub();
    with_numerics_mode(NumericsMode::Fast, || {
        for j in 0..Y_POINTS {
            for i in 0..X_POINTS {
                let (x, y) = query_point(i, j);
                let ev = |qx: f64, qy: f64| {
                    interp_evbicub_db(qx, qy, &xpts, &ypts, ilinx, iliny, &f)
                        .expect("fast bicubic evaluation succeeds")
                };
                let eval_fast = ev(x, y);
                let h = 1.0e-6;
                let fd_dx = (ev(x + h, y).f - ev(x - h, y).f) / (2.0 * h);
                let fd_dy = (ev(x, y + h).f - ev(x, y - h).f) / (2.0 * h);
                assert!(
                    (eval_fast.df_dx - fd_dx).abs() < 1.0e-10,
                    "i={i}, j={j}: df_dx={} vs fd={fd_dx}",
                    eval_fast.df_dx
                );
                assert!(
                    (eval_fast.df_dy - fd_dy).abs() < 1.0e-10,
                    "i={i}, j={j}: df_dy={} vs fd={fd_dy}",
                    eval_fast.df_dy
                );
            }
        }
    });
}

proptest! {
    /// In `NumericsMode::Fast`, the 2D bicubic spline reproduces any
    /// biquadratic polynomial
    /// $p(x, y) = a_{00} + a_{10} x + a_{01} y + a_{20} x^2 + a_{11} x y
    /// + a_{02} y^2$ and its first and second partial derivatives to machine
    /// precision.
    #[test]
    fn prop_fast_mode_bicubic_spline_reproduces_biquadratic_and_derivatives(
        a00 in -2.0_f64..2.0_f64,
        a10 in -1.5_f64..1.5_f64,
        a01 in -1.5_f64..1.5_f64,
        a20 in -0.5_f64..0.5_f64,
        a11 in -0.5_f64..0.5_f64,
        a02 in -0.5_f64..0.5_f64,
        xq in -0.9_f64..0.9_f64,
        yq in -0.9_f64..0.9_f64,
    ) {
        let nx = 9;
        let ny = 10;
        let xs: Vec<f64> = (0..nx)
            .map(|i| -1.0 + 2.0 * (i as f64) / ((nx - 1) as f64))
            .collect();
        let ys: Vec<f64> = (0..ny)
            .map(|j| -1.0 + 2.0 * (j as f64) / ((ny - 1) as f64))
            .collect();

        let poly = |x: f64, y: f64| -> f64 {
            a00 + a10 * x + a01 * y + a20 * x * x + a11 * x * y + a02 * y * y
        };
        let mut z = vec![0.0_f64; nx * ny];
        for (i, &x) in xs.iter().enumerate() {
            for (j, &y) in ys.iter().enumerate() {
                z[i * ny + j] = poly(x, y);
            }
        }
        let bc_x = vec![0.0; ny];
        let bc_y = vec![0.0; nx];
        let (f, ilinx, iliny) = interp_mkbicub_db(
            &xs,
            &ys,
            &z,
            BicubicBc::NotAKnot,
            &bc_x,
            BicubicBc::NotAKnot,
            &bc_x,
            BicubicBc::NotAKnot,
            &bc_y,
            BicubicBc::NotAKnot,
            &bc_y,
        ).unwrap();

        let ev = with_numerics_mode(NumericsMode::Fast, || {
            interp_evbicub_db(xq, yq, &xs, &ys, ilinx, iliny, &f)
        })
        .unwrap();
        let exact_f = poly(xq, yq);
        let exact_df_dx = a10 + 2.0 * a20 * xq + a11 * yq;
        let exact_df_dy = a01 + a11 * xq + 2.0 * a02 * yq;
        let exact_d2f_dx2 = 2.0 * a20;
        let exact_d2f_dy2 = 2.0 * a02;
        let exact_d2f_dxdy = a11;

        prop_assert!((ev.f - exact_f).abs() < 1.0e-12);
        prop_assert!((ev.df_dx - exact_df_dx).abs() < 1.0e-11);
        prop_assert!((ev.df_dy - exact_df_dy).abs() < 1.0e-11);
        prop_assert!((ev.d2f_dx2 - exact_d2f_dx2).abs() < 1.0e-10);
        prop_assert!((ev.d2f_dy2 - exact_d2f_dy2).abs() < 1.0e-10);
        prop_assert!((ev.d2f_dxdy - exact_d2f_dxdy).abs() < 1.0e-10);
    }
}
