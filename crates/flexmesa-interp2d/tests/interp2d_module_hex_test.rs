// SPDX-License-Identifier: LGPL-3.0-or-later
//! Level-2 IEEE-754 module-hex parity tests for `flexmesa-interp2d` against
//! MESA `26.4.1` `interp_2d/test/test_output` (`exact_tested_values.tsv`,
//! 104 rows: 56 `f64`/`f64_arr` rows from `test2D_db` and 48 unported `f32`
//! `_sg` rows from `test2D_sg`).
//!
//! # Upstream `fvbicub_db` Implicit-`REAL` Quirk
//!
//! In MESA `26.4.1` `interp_2d/private/bicub_db.f:1420-1575`, local basis
//! scalars (`xp`, `xpi`, `cx`, `cxi`, `yp`, `ypi`, `cy`, `cyi`, `z36th`,
//! `hx2`, `hy2`, `cxd`, `cxdi`, `cyd`, `cydi`) omit `real*8` declarations and
//! fall under Fortran implicit `REAL` (`f32`), while accumulators remain
//! `f64`. Production [`interp_evbicub_db`] stays strictly pure `f64` for
//! downstream precision (`<= 3e-8` abs, `<= 2.5e8` ULP vs MESA's 18
//! `fvbicub_db` output rows `[4..34]`), while the test-local helper
//! `eval_fvbicub_db_implicit_f32` verifies `0 ULP` on all 18 rows.

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

fn eval_fvbicub_db_implicit_f32(
    xget: f64,
    yget: f64,
    x: &[f64],
    y: &[f64],
    f: &[[f64; 4]],
) -> (f64, f64, f64) {
    let nx = x.len();
    let ny = y.len();
    // Uniform grid cell lookup matching MESA evbicub_db (ilinx=1, iliny=1)
    let mut i = ((((xget - x[0]) * ((nx - 1) as f64)) / (x[nx - 1] - x[0]))
        as usize)
        .min(nx - 2);
    if xget < x[i] {
        while i > 0 && xget < x[i] {
            i -= 1;
        }
    } else {
        while i + 2 < nx && xget >= x[i + 1] {
            i += 1;
        }
    }
    let mut j = ((((yget - y[0]) * ((ny - 1) as f64)) / (y[ny - 1] - y[0]))
        as usize)
        .min(ny - 2);
    if yget < y[j] {
        while j > 0 && yget < y[j] {
            j -= 1;
        }
    } else {
        while j + 2 < ny && yget >= y[j + 1] {
            j += 1;
        }
    }

    let hx = x[i + 1] - x[i];
    let hxi = 1.0_f64 / hx;
    let hy = y[j + 1] - y[j];
    let hyi = 1.0_f64 / hy;
    let xparam = (xget - x[i]) * hxi;
    let yparam = (yget - y[j]) * hyi;

    let fin = |k: usize, ix: usize, iy: usize| -> f64 { f[ix * ny + iy][k] };

    // Match MESA `fvbicub_db` (bicub_db.f:1420-1575) implicit REAL (`f32`)
    // local variables while keeping accumulators (`sum`) and `sixth` in `f64`.
    let sixth = 1.0_f64 / 6.0_f64;
    let z36th = (sixth * sixth) as f32;

    let xp = xparam as f32;
    let xpi = (1.0_f64 - f64::from(xp)) as f32;
    let xp2 = xp * xp;
    let xpi2 = xpi * xpi;
    let cx = (f64::from(xp) * (f64::from(xp2) - 1.0_f64)) as f32;
    let cxi = (f64::from(xpi) * (f64::from(xpi2) - 1.0_f64)) as f32;
    let hx2 = (hx * hx) as f32;

    let yp = yparam as f32;
    let ypi = (1.0_f64 - f64::from(yp)) as f32;
    let yp2 = yp * yp;
    let ypi2 = ypi * ypi;
    let cy = (f64::from(yp) * (f64::from(yp2) - 1.0_f64)) as f32;
    let cyi = (f64::from(ypi) * (f64::from(ypi2) - 1.0_f64)) as f32;
    let hy2 = (hy * hy) as f32;

    let cxd = (3.0_f64 * f64::from(xp2) - 1.0_f64) as f32;
    let cxdi = (-3.0_f64 * f64::from(xpi2) + 1.0_f64) as f32;
    let cyd = (3.0_f64 * f64::from(yp2) - 1.0_f64) as f32;
    let cydi = (-3.0_f64 * f64::from(ypi2) + 1.0_f64) as f32;

    let z36th_hx2_hy2_d = f64::from((z36th * hx2) * hy2);
    let z36th_hx2_d = f64::from(z36th * hx2);
    let (z36th_d, hx2_d, hy2_d) =
        (f64::from(z36th), f64::from(hx2), f64::from(hy2));
    let (xp_d, xpi_d, cx_d, cxi_d) =
        (f64::from(xp), f64::from(xpi), f64::from(cx), f64::from(cxi));
    let (yp_d, ypi_d, cy_d, cyi_d) =
        (f64::from(yp), f64::from(ypi), f64::from(cy), f64::from(cyi));
    let (cxd_d, cxdi_d, cyd_d, cydi_d) = (
        f64::from(cxd),
        f64::from(cxdi),
        f64::from(cyd),
        f64::from(cydi),
    );

    let mut sum_f = xpi_d * (ypi_d * fin(0, i, j) + yp_d * fin(0, i, j + 1))
        + xp_d * (ypi_d * fin(0, i + 1, j) + yp_d * fin(0, i + 1, j + 1));
    sum_f += sixth
        * hx2_d
        * (cxi_d * (ypi_d * fin(1, i, j) + yp_d * fin(1, i, j + 1))
            + cx_d * (ypi_d * fin(1, i + 1, j) + yp_d * fin(1, i + 1, j + 1)));
    sum_f += sixth
        * hy2_d
        * (xpi_d * (cyi_d * fin(2, i, j) + cy_d * fin(2, i, j + 1))
            + xp_d * (cyi_d * fin(2, i + 1, j) + cy_d * fin(2, i + 1, j + 1)));
    sum_f += z36th_hx2_hy2_d
        * (cxi_d * (cyi_d * fin(3, i, j) + cy_d * fin(3, i, j + 1))
            + cx_d * (cyi_d * fin(3, i + 1, j) + cy_d * fin(3, i + 1, j + 1)));

    let mut sum_fx = hxi
        * (-(ypi_d * fin(0, i, j) + yp_d * fin(0, i, j + 1))
            + (ypi_d * fin(0, i + 1, j) + yp_d * fin(0, i + 1, j + 1)));
    sum_fx += sixth
        * hx
        * (cxdi_d * (ypi_d * fin(1, i, j) + yp_d * fin(1, i, j + 1))
            + cxd_d * (ypi_d * fin(1, i + 1, j) + yp_d * fin(1, i + 1, j + 1)));
    sum_fx += sixth
        * hxi
        * hy2_d
        * (-(cyi_d * fin(2, i, j) + cy_d * fin(2, i, j + 1))
            + (cyi_d * fin(2, i + 1, j) + cy_d * fin(2, i + 1, j + 1)));
    sum_fx += z36th_d
        * hx
        * hy2_d
        * (cxdi_d * (cyi_d * fin(3, i, j) + cy_d * fin(3, i, j + 1))
            + cxd_d * (cyi_d * fin(3, i + 1, j) + cy_d * fin(3, i + 1, j + 1)));

    let mut sum_fy = hyi
        * (xpi_d * (-fin(0, i, j) + fin(0, i, j + 1))
            + xp_d * (-fin(0, i + 1, j) + fin(0, i + 1, j + 1)));
    sum_fy += sixth
        * hx2_d
        * hyi
        * (cxi_d * (-fin(1, i, j) + fin(1, i, j + 1))
            + cx_d * (-fin(1, i + 1, j) + fin(1, i + 1, j + 1)));
    sum_fy += sixth
        * hy
        * (xpi_d * (cydi_d * fin(2, i, j) + cyd_d * fin(2, i, j + 1))
            + xp_d
                * (cydi_d * fin(2, i + 1, j) + cyd_d * fin(2, i + 1, j + 1)));
    sum_fy += z36th_hx2_d
        * hy
        * (cxi_d * (cydi_d * fin(3, i, j) + cyd_d * fin(3, i, j + 1))
            + cx_d
                * (cydi_d * fin(3, i + 1, j) + cyd_d * fin(3, i + 1, j + 1)));

    (sum_f, sum_fx, sum_fy)
}

#[test]
fn test_interp2d_module_hex_zero_ulp() {
    let Some(path) = resolve_module_hex_tsv() else {
        return;
    };
    let rows = load_module_hex_rows(&path);
    assert_eq!(rows.len(), 104, "unexpected interp_2d TSV row count");

    with_numerics_mode(NumericsMode::Strict, || {
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

        let x_points = 2usize;
        let y_points = 3usize;
        let x_max = 0.8_f64 * PI;
        let x_min = 0.1_f64;
        let y_max = 0.6_f64 * PI;
        let y_min = 0.2_f64;
        let dx = (x_max - x_min) / ((x_points - 1) as f64);
        let dy = (y_max - y_min) / ((y_points - 1) as f64);

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
        let mut f32_basis_zero = 0usize;
        for j in 0..y_points {
            let y = y_min + (j as f64) * dy;
            for i in 0..x_points {
                let x = x_min + (i as f64) * dx;
                let ev = interp_evbicub_db(
                    x, y, &xpts, &ypts, ilinx, iliny, &f_bicub,
                )
                .unwrap();
                let (sf, sfx, sfy) =
                    eval_fvbicub_db_implicit_f32(x, y, &xpts, &ypts, &f_bicub);
                let base_idx = 4 + (j * x_points + i) * 5;
                for (off, val_s) in [(2, sf), (3, sfx), (4, sfy)] {
                    let row = &rows[base_idx + off];
                    let u = ulp_distance(val_s, row.raw_bits);
                    assert_eq!(
                        u,
                        0,
                        "fvbicub_db implicit-f32 mismatch row {} ({} `{}`): \
                         got={val_s:.17e} (0x{:016x}) exp={:.17e} (0x{:016x}) \
                         ulps={u}",
                        base_idx + off,
                        row.src_loc,
                        row.label,
                        val_s.to_bits(),
                        f64::from_bits(row.raw_bits),
                        row.raw_bits
                    );
                    f32_basis_zero += 1;
                }
                db_vals.push(("evbicub_x", x));
                db_vals.push(("evbicub_y", y));
                db_vals.push(("evbicub_z", ev.f));
                db_vals.push(("evbicub_dzdx", ev.df_dx));
                db_vals.push(("evbicub_dzdy", ev.df_dy));
            }
        }

        let f_bipm = interp_mkbipm_db(&xpts, &ypts, &z).unwrap();
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
        assert_eq!(
            f32_basis_zero, 18,
            "all 18 fvbicub_db outputs must match implicit-f32 MESA basis at 0 ULP"
        );

        // Assert pure-f64 0 ULP on 38 rows (mkbicub mid, evbicub coords,
        // mkbipm, evbipm) and tight pure-f64 agreement on the 18 fvbicub_db
        // implicit-REAL output rows `[4..34]`.
        let mut pure_f64_zero = 0usize;
        for (idx, ((name, actual), row)) in
            db_vals.iter().zip(rows[..56].iter()).enumerate()
        {
            let ulps = ulp_distance(*actual, row.raw_bits);
            if !(4..34).contains(&idx) || (idx - 4) % 5 < 2 {
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
                pure_f64_zero += 1;
            } else {
                let exp_val = f64::from_bits(row.raw_bits);
                let abs_err = (*actual - exp_val).abs();
                assert!(
                    abs_err <= 3.0e-8 && ulps <= 250_000_000,
                    "fvbicub pure-f64 bound exceeded at row {idx} ({name}): \
                     got={actual:.17e} exp={exp_val:.17e} abs_err={abs_err:.3e} \
                     ulps={ulps}"
                );
            }
        }
        assert_eq!(pure_f64_zero, 38);

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
    });
}
