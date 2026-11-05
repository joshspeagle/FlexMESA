//! Integration and property-based tests for `flexmesa-interp2d`.

#![allow(
    clippy::approx_constant,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use flexmesa_interp2d::{
    interp_evbicub_db, interp_evbicub_mask_db, interp_evbipm_db,
    interp_mkbicub_db, interp_mkbipm_db, interp_rgbi3p_db, interp_rgsf3p_db,
    BicubicBc,
};
use proptest::prelude::*;
use std::f64::consts::PI;

const NUM_XPTS: usize = 31;
const NUM_YPTS: usize = 35;

fn test_fcn_db(x: f64, y: f64) -> f64 {
    let r = (x * x + y * y).sqrt();
    let r0 = 0.5 * PI;
    (-r / r0).exp() * x.sin() * y.cos()
}

fn build_mesa_2d_test_grid() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let xmin = -PI;
    let xmax = PI;
    let dx = (xmax - xmin) / ((NUM_XPTS - 1) as f64);
    let ymin = -PI;
    let ymax = PI;
    let dy = (ymax - ymin) / ((NUM_YPTS - 1) as f64);

    let xpts: Vec<f64> =
        (0..NUM_XPTS).map(|i| xmin + (i as f64) * dx).collect();
    let ypts: Vec<f64> =
        (0..NUM_YPTS).map(|j| ymin + (j as f64) * dy).collect();
    let mut z = vec![0.0_f64; NUM_XPTS * NUM_YPTS];
    for (i, &x) in xpts.iter().enumerate() {
        for (j, &y) in ypts.iter().enumerate() {
            z[i * NUM_YPTS + j] = test_fcn_db(x, y);
        }
    }
    (xpts, ypts, z)
}

#[test]
fn test_mesa_bicubic_spline_matches_golden_output() {
    let (xpts, ypts, z) = build_mesa_2d_test_grid();
    let bcxmin = vec![0.0; NUM_YPTS];
    let bcxmax = vec![0.0; NUM_YPTS];
    let bcymin = vec![0.0; NUM_XPTS];
    let bcymax = vec![0.0; NUM_XPTS];

    let (f, ilinx, iliny) = interp_mkbicub_db(
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
    .expect("bicubic setup succeeds");
    assert!(ilinx);
    assert!(iliny);

    // Fortran 1-based (num_xpts/2 = 15, num_ypts/2 = 17) -> 0-based (14, 16)
    let mid = f[14 * NUM_YPTS + 16];
    let expected_mid = [
        -1.710_786_994_037_610_3e-1,
        1.229_709_813_274_336_5,
        3.211_248_961_929_241e-1,
        4.760_377_855_726_039e-1,
    ];
    for (got, exp) in mid.iter().zip(expected_mid.iter()) {
        assert!((got - exp).abs() < 1.0e-12, "got {got}, expected {exp}");
    }

    let x_min = 0.1_f64;
    let x_max = 0.8 * PI;
    let y_min = 0.2_f64;
    let y_max = 0.6 * PI;
    let dx = x_max - x_min;
    let dy = 0.5 * (y_max - y_min);

    let expected_pts: [(usize, usize, f64, f64, f64); 6] = [
        (
            0,
            0,
            8.412_838_387_532_941e-2,
            8.223_244_982_667_942e-1,
            -5.830_990_551_255_731e-2,
        ),
        (
            1,
            0,
            1.157_193_932_651_681_8e-1,
            -2.327_225_412_452_813e-1,
            -2.931_055_018_688_918_2e-2,
        ),
        (
            0,
            1,
            2.583_313_721_966_36e-2,
            2.559_498_503_555_757_7e-1,
            -6.062_774_870_752_535e-2,
        ),
        (
            1,
            1,
            5.241_368_533_274_961_4e-2,
            -1.029_682_315_081_142_5e-1,
            -1.025_887_303_464_508_2e-1,
        ),
        (
            0,
            2,
            -9.275_811_236_254_504e-3,
            -9.214_212_823_541_054e-2,
            -2.265_013_757_433_625e-2,
        ),
        (
            1,
            2,
            -2.458_149_485_380_903e-2,
            4.635_499_485_797_099e-2,
            -6.625_610_809_426_089e-2,
        ),
    ];

    for (i, j, exp_z, exp_dzdx, exp_dzdy) in expected_pts {
        let x = x_min + (i as f64) * dx;
        let y = y_min + (j as f64) * dy;
        let eval = interp_evbicub_db(x, y, &xpts, &ypts, ilinx, iliny, &f)
            .expect("bicubic evaluation succeeds");
        // Upstream MESA fvbicub_db omits `implicit none`, leaving local basis
        // variables (xp, yp, cx, cy, cxd, cyd, hx2, hy2, z36th) as implicit
        // 32-bit `real`. Our full `f64` evaluator matches the golden output to
        // within single-precision basis truncation (~5e-7) while satisfying
        // exact f64 calculus identities to machine precision.
        assert!(
            (eval.f - exp_z).abs() < 5.0e-9,
            "i={i}, j={j}: eval.f={} vs exp_z={exp_z}",
            eval.f
        );
        assert!(
            (eval.df_dx - exp_dzdx).abs() < 5.0e-7,
            "i={i}, j={j}: eval.df_dx={} vs exp_dzdx={exp_dzdx}",
            eval.df_dx
        );
        assert!(
            (eval.df_dy - exp_dzdy).abs() < 5.0e-7,
            "i={i}, j={j}: eval.df_dy={} vs exp_dzdy={exp_dzdy}",
            eval.df_dy
        );

        // Verify exact f64 consistency between value and analytic first/second
        // derivatives on the cubic polynomial patch.
        let h = 1.0e-6;
        let fx_p = interp_evbicub_db(x + h, y, &xpts, &ypts, ilinx, iliny, &f)
            .unwrap();
        let fx_m = interp_evbicub_db(x - h, y, &xpts, &ypts, ilinx, iliny, &f)
            .unwrap();
        let fy_p = interp_evbicub_db(x, y + h, &xpts, &ypts, ilinx, iliny, &f)
            .unwrap();
        let fy_m = interp_evbicub_db(x, y - h, &xpts, &ypts, ilinx, iliny, &f)
            .unwrap();
        let fd_dx = (fx_p.f - fx_m.f) / (2.0 * h);
        let fd_dy = (fy_p.f - fy_m.f) / (2.0 * h);
        assert!((eval.df_dx - fd_dx).abs() < 1.0e-10);
        assert!((eval.df_dy - fd_dy).abs() < 1.0e-10);

        let masked = interp_evbicub_mask_db(
            x,
            y,
            &xpts,
            &ypts,
            ilinx,
            iliny,
            &f,
            &[1, 1, 1, 0, 0, 0],
        )
        .expect("masked bicubic evaluation succeeds");
        assert!((masked[0] - eval.f).abs() < 1.0e-15);
        assert!((masked[1] - eval.df_dx).abs() < 1.0e-15);
        assert!((masked[2] - eval.df_dy).abs() < 1.0e-15);
        assert_eq!(masked[3], 0.0);
    }
}

#[test]
fn test_mesa_bipm_matches_golden_output() {
    let (xpts, ypts, z) = build_mesa_2d_test_grid();
    let f = interp_mkbipm_db(&xpts, &ypts, &z).expect("bipm setup succeeds");

    let mid = f[14 * NUM_YPTS + 16];
    let expected_mid = [
        -1.710_786_994_037_610_3e-1,
        7.131_561_165_935_149e-1,
        9.901_140_746_866_6e-1,
        -2.363_723_238_168_592_6,
    ];
    for (got, exp) in mid.iter().zip(expected_mid.iter()) {
        assert!((got - exp).abs() < 1.0e-12, "got {got}, expected {exp}");
    }

    let x_min = 0.1_f64;
    let x_max = 0.8 * PI;
    let y_min = 0.2_f64;
    let y_max = 0.6 * PI;
    let dx = x_max - x_min;
    let dy = 0.5 * (y_max - y_min);

    let expected_z: [(usize, usize, f64); 6] = [
        (0, 0, 8.351_309_743_282_355e-2),
        (1, 0, 1.157_230_435_480_164_6e-1),
        (0, 1, 2.580_157_764_222_351e-2),
        (1, 1, 5.240_292_085_218_732_6e-2),
        (0, 2, -9.271_793_796_082_475e-3),
        (1, 2, -2.457_567_139_911_116e-2),
    ];

    for (i, j, exp_z) in expected_z {
        let x = x_min + (i as f64) * dx;
        let y = y_min + (j as f64) * dy;
        let z_val = interp_evbipm_db(x, y, &xpts, &ypts, &f)
            .expect("bipm evaluation succeeds");
        assert!(
            (z_val - exp_z).abs() < 1.0e-12,
            "got {z_val}, expected {exp_z}"
        );
    }
}

#[test]
fn test_akima_rgbi3p_and_rgsf3p_match_published_table() {
    let xd: Vec<f64> = (0..9).map(|i| i as f64).collect();
    let yd: Vec<f64> = (0..11).map(|j| j as f64).collect();

    // Fortran column-major ZD(1:9, 1:11) table from test_akima_db.f
    let zd_col_major: [[f64; 9]; 11] = [
        [0.0; 9],
        [0.0; 9],
        [0.0; 9],
        [3.2, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [7.4, 4.8, 1.4, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        [12.0, 8.0, 5.3, 2.9, 0.6, 0.0, 0.0, 0.0, 0.0],
        [16.8, 14.4, 8.1, 6.9, 6.2, 0.6, 0.1, 0.0, 0.0],
        [21.8, 20.5, 12.8, 17.6, 5.8, 7.6, 0.8, 0.6, 0.6],
        [22.4, 22.5, 14.6, 22.5, 4.7, 7.2, 1.8, 2.1, 2.1],
        [37.2, 40.0, 27.0, 41.3, 14.1, 24.5, 17.3, 20.2, 20.8],
        [58.2, 61.5, 47.9, 62.3, 34.6, 45.5, 38.2, 41.2, 41.7],
    ];
    let mut zd = vec![0.0_f64; 9 * 11];
    for ix in 0..9 {
        for iy in 0..11 {
            zd[ix * 11 + iy] = zd_col_major[iy][ix];
        }
    }

    let xi: Vec<f64> = (0..19).map(|i| -0.5 + 0.5 * (i as f64)).collect();
    let yi: Vec<f64> = (0..23).map(|j| -0.5 + 0.5 * (j as f64)).collect();

    let zi_sf = interp_rgsf3p_db(&xd, &yd, &zd, &xi, &yi)
        .expect("surface fit succeeds");

    let mut xi_pts = Vec::with_capacity(19 * 23);
    let mut yi_pts = Vec::with_capacity(19 * 23);
    for &x in &xi {
        for &y in &yi {
            xi_pts.push(x);
            yi_pts.push(y);
        }
    }
    let zi_bi = interp_rgbi3p_db(&xd, &yd, &zd, &xi_pts, &yi_pts)
        .expect("pointwise Akima interpolation succeeds");

    for (a, b) in zi_sf.iter().zip(zi_bi.iter()) {
        assert!((a - b).abs() < 1.0e-14);
    }

    // Spot-check corner, interior, and extrapolated values from ZIE in test_akima_db.f
    assert!((zi_sf[0] - (-0.847)).abs() < 5.0e-4);
    assert!((zi_sf[1] - 0.0).abs() < 5.0e-4);
    assert!((zi_sf[22] - 58.284).abs() < 5.0e-4);
    assert!((zi_sf[18 * 23 + 22] - 48.268).abs() < 5.0e-4);
}

proptest! {
    #[test]
    fn prop_bicubic_spline_reproduces_biquadratic_and_derivatives(
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

        let eval = interp_evbicub_db(xq, yq, &xs, &ys, ilinx, iliny, &f).unwrap();
        let exact_f = poly(xq, yq);
        let exact_df_dx = a10 + 2.0 * a20 * xq + a11 * yq;
        let exact_df_dy = a01 + a11 * xq + 2.0 * a02 * yq;
        let exact_d2f_dx2 = 2.0 * a20;
        let exact_d2f_dy2 = 2.0 * a02;
        let exact_d2f_dxdy = a11;

        prop_assert!((eval.f - exact_f).abs() < 1.0e-12);
        prop_assert!((eval.df_dx - exact_df_dx).abs() < 1.0e-11);
        prop_assert!((eval.df_dy - exact_df_dy).abs() < 1.0e-11);
        prop_assert!((eval.d2f_dx2 - exact_d2f_dx2).abs() < 1.0e-10);
        prop_assert!((eval.d2f_dy2 - exact_d2f_dy2).abs() < 1.0e-10);
        prop_assert!((eval.d2f_dxdy - exact_d2f_dxdy).abs() < 1.0e-10);
    }
}
