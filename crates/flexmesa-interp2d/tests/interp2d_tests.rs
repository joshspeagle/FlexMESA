//! Integration and property-based tests for `flexmesa-interp2d`.
//!
//! Validates Safe Rust 2D bicubic splines (`interp_mkbicub_db`,
//! `interp_evbicub_db`, `interp_evbicub_mask_db`), 2D Steffen (1990)
//! piecewise-monotonic splines (`interp_mkbipm_db`, `interp_evbipm_db`), and
//! Akima (1996, ACM Algorithm 760) bivariate surface fitting
//! (`interp_rgbi3p_db`, `interp_rgsf3p_db`, `rgpd3p_db`) against upstream MESA
//! golden baselines (`test2D_db`, `test_akima_db`) and exact polynomial
//! calculus identities.

#![allow(
    clippy::approx_constant,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
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

/// Number of x-grid knots in upstream MESA `test2D_db`.
const NUM_XPTS: usize = 31;

/// Number of y-grid knots in upstream MESA `test2D_db`.
const NUM_YPTS: usize = 35;

/// Evaluates the upstream MESA 2D test function
/// $f(x, y) = \exp(-r / r_0) \sin(x) \cos(y)$ with $r = \sqrt{x^2 + y^2}$ and
/// $r_0 = \pi / 2$.
fn test_fcn_db(x: f64, y: f64) -> f64 {
    let r = (x * x + y * y).sqrt();
    let r0 = 0.5 * PI;
    (-r / r0).exp() * x.sin() * y.cos()
}

/// Constructs the $31 \times 35$ uniform rectangular test grid
/// `(xpts, ypts, z)` over $[-\pi, \pi]^2$ matching upstream MESA `test2D_db`.
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

/// Verifies `interp_mkbicub_db`, `interp_evbicub_db`, and
/// `interp_evbicub_mask_db` against upstream MESA `test2D_db(.true.)` and
/// finite-difference derivative identities.
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

    // Fortran 1-based (num_xpts/2 = 15, num_ypts/2 = 17) -> 0-based (14, 16).
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

    // Pinned (i, j, z, dz/dx, dz/dy) values from upstream test2D_db(.true.).
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

/// Verifies 2D Steffen (1990) piecewise-monotonic interpolation
/// (`interp_mkbipm_db`, `interp_evbipm_db`) against `test2D_db(.false.)`.
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

/// Verifies Akima (1996, ACM Algorithm 760) `interp_rgbi3p_db` and
/// `interp_rgsf3p_db` against the published 9x11 table in `test_akima_db.f`.
#[test]
fn test_akima_rgbi3p_and_rgsf3p_match_published_table() {
    let xd: Vec<f64> = (0..9).map(|i| i as f64).collect();
    let yd: Vec<f64> = (0..11).map(|j| j as f64).collect();

    // Fortran column-major ZD(1:9, 1:11) table from test_akima_db.f.
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

    // Spot-check corner, interior, and extrapolated values from ZIE.
    assert!((zi_sf[0] - (-0.847)).abs() < 5.0e-4);
    assert!((zi_sf[1] - 0.0).abs() < 5.0e-4);
    assert!((zi_sf[22] - 58.284).abs() < 5.0e-4);
    assert!((zi_sf[18 * 23 + 22] - 48.268).abs() < 5.0e-4);
}

/// Verifies ACM TOMS 790 (`cshep2` / `cs2val` / `cs2grd` / `cs2hes` and
/// `qshep2` / `qs2val` / `qs2grd`) against `test_renka790_db.f` and scattered
/// 2D interpolation on `test_fcn_db`.
#[test]
fn test_renka790_cshep2_cubic_reproduction_and_scattered_2d_db() {
    use flexmesa_interp2d::{
        interp_cs2grd_db, interp_cs2val_db, interp_cshep2_db, qs2grd, qs2val,
        qshep2,
    };

    // Part 1: Exact cubic polynomial reproduction from test_renka790_db.f
    // FQ(x, y) = ((x + 2y)/3)^3 on 4 x 5 grid (N = 20), NC = 10, NW = 16, NR = 3
    let fq = |xx: f64, yy: f64| ((xx + 2.0 * yy) / 3.0).powi(3);
    let fx = |xx: f64, yy: f64| ((xx + 2.0 * yy) / 3.0).powi(2);
    let fy = |xx: f64, yy: f64| 2.0 * ((xx + 2.0 * yy) / 3.0).powi(2);
    let fxx = |xx: f64, yy: f64| 2.0 * (xx + 2.0 * yy) / 9.0;
    let fxy = |xx: f64, yy: f64| 4.0 * (xx + 2.0 * yy) / 9.0;
    let fyy = |xx: f64, yy: f64| 8.0 * (xx + 2.0 * yy) / 9.0;

    let mut x = Vec::with_capacity(20);
    let mut y = Vec::with_capacity(20);
    let mut f = Vec::with_capacity(20);
    for j in 0..5 {
        let yj = (j as f64) / 4.0;
        for i in 0..4 {
            let xi = (i as f64) / 3.0;
            x.push(xi);
            y.push(yj);
            f.push(fq(xi, yj));
        }
    }

    let cshep = interp_cshep2_db(&x, &y, &f, 10, 16, 3)
        .expect("cshep2 cubic fit succeeds");
    assert_eq!(cshep.len(), 20);
    let qshep = qshep2(&x, &y, &f, 10, 16, 3).expect("qshep2 fit succeeds");

    for j in 0..=10 {
        let py = (j as f64) / 10.0;
        for i in 0..=10 {
            let px = (i as f64) / 10.0;
            let c_val = interp_cs2val_db(&cshep, px, py);
            let q_val = qs2val(&qshep, px, py);
            let (cg, cx, cy) = interp_cs2grd_db(&cshep, px, py);
            let (qg, qx, qy) = qs2grd(&qshep, px, py);
            let (ch, chx, chy, chxx, chxy, chyy) = cshep.eval_hess(px, py);

            assert!((c_val - fq(px, py)).abs() < 1.0e-13);
            assert!((q_val - c_val).abs() < 1.0e-15);
            assert!((cg - c_val).abs() < 1.0e-14);
            assert!((qg - q_val).abs() < 1.0e-14);
            assert!((cx - fx(px, py)).abs() < 1.0e-12);
            assert!((cy - fy(px, py)).abs() < 1.0e-12);
            assert!((qx - cx).abs() < 1.0e-15);
            assert!((qy - cy).abs() < 1.0e-15);
            assert!((ch - c_val).abs() < 1.0e-14);
            assert!((chx - cx).abs() < 1.0e-14);
            assert!((chy - cy).abs() < 1.0e-14);
            assert!((chxx - fxx(px, py)).abs() < 1.0e-11);
            assert!((chxy - fxy(px, py)).abs() < 1.0e-11);
            assert!((chyy - fyy(px, py)).abs() < 1.0e-11);
        }
    }

    // Part 2: Franke's 33-node set (KS = 2, N = 33, NI = 33, NC = 17, NW = 30,
    // NR = 6) from test_renka790_db.f (KF = 1, Franke's exponential surface).
    let x2: [f64; 33] = [
        0.05, 0.00, 0.00, 0.00, 0.10, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.50,
        0.50, 0.55, 0.60, 0.60, 0.60, 0.65, 0.70, 0.70, 0.70, 0.75, 0.75, 0.75,
        0.80, 0.80, 0.85, 0.90, 0.90, 0.95, 1.00, 1.00, 1.00,
    ];
    let y2: [f64; 33] = [
        0.45, 0.50, 1.00, 0.00, 0.15, 0.75, 0.30, 0.10, 0.20, 0.35, 0.85, 0.00,
        1.00, 0.95, 0.25, 0.65, 0.85, 0.70, 0.20, 0.65, 0.90, 0.10, 0.35, 0.85,
        0.40, 0.65, 0.25, 0.35, 0.80, 0.90, 0.00, 0.50, 1.00,
    ];
    let tstfn1_exp = |xx: f64, yy: f64| {
        0.75 * (-((9.0 * xx - 2.0).powi(2) + (9.0 * yy - 2.0).powi(2)) / 4.0)
            .exp()
            + 0.75
                * (-(9.0 * xx + 1.0).powi(2) / 49.0 - (9.0 * yy + 1.0) / 10.0)
                    .exp()
            + 0.5
                * (-((9.0 * xx - 7.0).powi(2) + (9.0 * yy - 3.0).powi(2)) / 4.0)
                    .exp()
            - 0.2 * (-(9.0 * xx - 4.0).powi(2) - (9.0 * yy - 7.0).powi(2)).exp()
    };
    let w2: Vec<f64> = x2
        .iter()
        .zip(y2.iter())
        .map(|(&xx, &yy)| tstfn1_exp(xx, yy))
        .collect();
    let franke_interp = interp_cshep2_db(&x2, &y2, &w2, 17, 30, 6)
        .expect("Franke 33-node fit succeeds");
    let mut ermax = 0.0_f64;
    let mut ermean = 0.0_f64;
    for i in 0..33 {
        let px = (i as f64) / 32.0;
        for j in 0..33 {
            let py = (j as f64) / 32.0;
            let err = (interp_cs2val_db(&franke_interp, px, py)
                - tstfn1_exp(px, py))
            .abs();
            ermax = ermax.max(err);
            ermean += err;
        }
    }
    ermean /= (33 * 33) as f64;
    assert!((ermax - 0.198_516_708_768_601_4).abs() < 1.0e-12);
    assert!((ermean - 0.032_170_250_054_613_99).abs() < 1.0e-12);

    // Part 3: Scattered 2D interpolation on MESA's 31 x 35 (N = 1085) grid of
    // test_fcn_db(x, y) evaluated at the 6 golden query points of test2D_db.
    let (xpts, ypts, z_grid) = build_mesa_2d_test_grid();
    let mut xs = Vec::with_capacity(NUM_XPTS * NUM_YPTS);
    let mut ys = Vec::with_capacity(NUM_XPTS * NUM_YPTS);
    let mut fs = Vec::with_capacity(NUM_XPTS * NUM_YPTS);
    for (i, &xv) in xpts.iter().enumerate() {
        for (j, &yv) in ypts.iter().enumerate() {
            xs.push(xv);
            ys.push(yv);
            fs.push(z_grid[i * NUM_YPTS + j]);
        }
    }
    let scat = interp_cshep2_db(&xs, &ys, &fs, 17, 32, 10)
        .expect("scattered 2D fit on MESA test grid succeeds");
    let x_min = 0.1_f64;
    let x_max = 0.8 * PI;
    let y_min = 0.2_f64;
    let y_max = 0.6 * PI;
    let dx = x_max - x_min;
    let dy = 0.5 * (y_max - y_min);
    let golden_pts: [(usize, usize, f64, f64, f64); 6] = [
        (
            0,
            0,
            0.084_430_315_712_205_27,
            0.842_697_806_864_306_5,
            -0.053_120_165_108_320_69,
        ),
        (
            1,
            0,
            0.115_719_537_154_871_95,
            -0.232_723_725_761_097_34,
            -0.029_288_035_126_348_5,
        ),
        (
            0,
            1,
            0.025_831_072_167_465_56,
            0.256_687_855_404_707_57,
            -0.060_768_993_765_178_29,
        ),
        (
            1,
            1,
            0.052_413_604_264_235_25,
            -0.102_755_936_267_800_94,
            -0.102_571_002_639_484_63,
        ),
        (
            0,
            2,
            -0.009_280_594_758_942_013,
            -0.092_617_455_868_842_03,
            -0.022_669_571_118_670_04,
        ),
        (
            1,
            2,
            -0.024_582_273_200_498_193,
            0.046_259_952_921_016_96,
            -0.066_293_101_429_167_07,
        ),
    ];
    for (i, j, exp_z, exp_dzdx, exp_dzdy) in golden_pts {
        let qx = x_min + (i as f64) * dx;
        let qy = y_min + (j as f64) * dy;
        let (z_scat, dz_dx, dz_dy) = interp_cs2grd_db(&scat, qx, qy);
        let z_exact = test_fcn_db(qx, qy);
        assert!((z_scat - exp_z).abs() < 1.0e-12);
        assert!((dz_dx - exp_dzdx).abs() < 1.0e-12);
        assert!((dz_dy - exp_dzdy).abs() < 1.0e-12);
        assert!((z_scat - z_exact).abs() < 5.0e-4);
    }
}

proptest! {
    /// Property test: 2D bicubic spline reproduces any biquadratic polynomial
    /// $p(x, y) = a_{00} + a_{10} x + a_{01} y + a_{20} x^2 + a_{11} x y
    /// + a_{02} y^2$ and its first and second partial derivatives to machine
    /// precision.
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

        let ev =
            interp_evbicub_db(xq, yq, &xs, &ys, ilinx, iliny, &f).unwrap();
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
