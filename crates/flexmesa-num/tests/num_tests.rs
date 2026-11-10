//! Golden and property tests for root-finding, optimization, search, and
//! quadrature in `flexmesa-num`.
//!
//! Verifies numerical parity against upstream MESA `num/test/src/test_num.f90`,
//! `test_brent.f90`, `test_Simplex.f90`, `test_integrate.f90`, and
//! `num/test/test_output`, as well as implicit dual differentiation via
//! [`safe_root_dual`] and `proptest` randomized invariants across 100 seeds.
//!
//! # Test Coverage
//!
//! - `test_two_sum_and_neumaier_sum_preserve_low_order_bits`: Knuth 2Sum and
//!   Neumaier compensated summation on ill-conditioned alternating series.
//! - `test_binary_search_increasing_and_decreasing_matches_upstream`:
//!   0-based and 1-based binary search with hint probing.
//! - `test_qsort_index_orders_values_monotonically`: index permutation sort.
//! - `test_find0_and_find0_quadratic_match_upstream_golden_roots`: linear and
//!   quadratic root interpolation against `test_find0_quadratic`.
//! - `test_find_max_quadratic_matches_upstream_and_fixes_y3_concavity_bug`:
//!   concave quadratic maximum against `test_find_max_quadratic` and bug fix.
//! - `test_dfridr_and_integrate_match_upstream_simp38_oracles`: Ridders
//!   differentiation and adaptive Simpson 3/8 quadrature (`simp38`).
//! - `test_brent_safe_zero_and_minimization_match_analytical_oracles`: Brent
//!   root-finding and local/global minimization.
//! - `test_safe_root_with_brackets_and_guess_match_upstream_golden_values`:
//!   safeguarded Newton-Raphson + secant/bisection root-finding.
//! - `test_safe_root_dual_matches_implicit_function_theorem_derivatives`:
//!   implicit differentiation of $f(x^*(p), p) = 0$ via `DualOrder1`.
//! - `test_nm_simplex_and_park_miller_prng_match_upstream_oracles`:
//!   Nelder-Mead simplex optimization and Park-Miller `ran1_zah` PRNG.

#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::type_complexity,
    clippy::unnecessary_wraps
)]

#![allow(clippy::doc_markdown, clippy::float_cmp)]

use flexmesa_autodiff::DualOrder1;
use flexmesa_num::{
    binary_search, binary_search_1based, bobyqa, brent_global_min,
    brent_local_min, brent_safe_zero, dfridr, find0, find0_quadratic,
    find_max_quadratic, glomin, i4_uniform, integrate, linear_interp,
    local_min, look_for_brackets, neumaier_sum, newuoa, nm_simplex,
    perm_uniform, qsort_index, r8_uniform_01, safe_root_dual,
    safe_root_with_brackets, safe_root_with_guess, safe_root_without_brackets,
    two_piece_linear_coeffs, two_sum, NumError, SimplexConfig,
};
use proptest::prelude::*;

/// Verifies that [`two_sum`] and [`neumaier_sum`] recover low-order bits lost
/// by naive floating-point addition.
#[test]
fn test_two_sum_and_neumaier_sum_preserve_low_order_bits() {
    let (s, e) = two_sum(1.0, 1e-16);
    assert!(((s + e) - (1.0 + 1e-16)).abs() <= 1e-30);
    // Sum [1.0, 1e16, 1.0, -1e16] where naive left-to-right addition yields 0.0
    let vals = [1.0, 1.0e16, 1.0, -1.0e16];
    let csum = neumaier_sum(&vals);
    assert!((csum - 2.0).abs() <= 1e-15);
}

/// Verifies [`binary_search`] and [`binary_search_1based`] on both increasing
/// and decreasing grids (`test_binary_search` in `test_num.f90`).
#[test]
fn test_binary_search_increasing_and_decreasing_matches_upstream() {
    // Ascending grid [0, 1, ..., 19]
    let inc: Vec<f64> = (0..20).map(f64::from).collect();
    assert_eq!(binary_search(&inc, -0.5, None), None);
    assert_eq!(binary_search(&inc, 19.5, None), Some(19));
    assert_eq!(binary_search(&inc, 13.5, Some(12)), Some(13));
    assert_eq!(binary_search_1based(&inc, -0.5, 0), 0);
    assert_eq!(binary_search_1based(&inc, 13.5, 13), 14);

    // Descending grid [19, 18, ..., 0]
    let dec: Vec<f64> = (0..20).rev().map(f64::from).collect();
    assert_eq!(binary_search(&dec, 19.5, None), None);
    assert_eq!(binary_search(&dec, -0.5, None), Some(19));
    assert_eq!(binary_search(&dec, 13.5, Some(5)), Some(5));
    assert!((linear_interp(0.0, 1.0, 2.0, 5.0, 0.5) - 2.0).abs() <= 1e-15);
    let (a, b, c) = two_piece_linear_coeffs(0.5, 0.0, 1.0, 2.0).unwrap();
    assert!((a - 0.5).abs() <= 1e-15 && (b - 0.5).abs() <= 1e-15 && c == 0.0);
}

/// Verifies [`qsort_index`] permutation ordering on unsorted slice inputs.
#[test]
fn test_qsort_index_orders_values_monotonically() {
    let vals = [3.5, -1.2, 0.0, 9.8, -1.2, 2.1];
    let idx = qsort_index(&vals);
    for w in idx.windows(2) {
        assert!(vals[w[0]] <= vals[w[1]]);
    }
}

/// Verifies [`find0`] and [`find0_quadratic`] against upstream
/// `test_find0_quadratic` (`(x - 3)(x - 8)` in `test_num.f90`).
#[test]
fn test_find0_and_find0_quadratic_match_upstream_golden_roots() {
    let x0 = find0(0.0, -2.0, 4.0, 6.0).unwrap();
    assert!((x0 - 1.0).abs() <= 1e-15);

    let g = |x: f64| (x - 3.0) * (x - 8.0);
    // Ascending abscissas bracketing x = 3
    let (r1, y1) =
        find0_quadratic(1.0, g(1.0), 2.0, g(2.0), 4.0, g(4.0)).unwrap();
    assert!((r1 - 3.0).abs() <= 1e-14 && y1.abs() <= 1e-14);
    // Descending abscissas bracketing x = 3
    let (r2, y2) =
        find0_quadratic(4.0, g(4.0), 2.0, g(2.0), 1.0, g(1.0)).unwrap();
    assert!((r2 - 3.0).abs() <= 1e-14 && y2.abs() <= 1e-14);
    // Exact knot root at x3 = 8
    let (r3, y3) =
        find0_quadratic(6.0, g(6.0), 7.0, g(7.0), 8.0, g(8.0)).unwrap();
    assert!((r3 - 8.0).abs() <= 1e-14 && y3.abs() <= 1e-14);
}

/// Verifies [`find_max_quadratic`] against upstream `test_find_max_quadratic`
/// (`xmax = 1.85`, `ymax = 12.408333333333333`) and confirms that the upstream
/// `y2 < max(y1, y2)` typo is fixed by rejecting `y3 > y2`.
#[test]
fn test_find_max_quadratic_matches_upstream_and_fixes_y3_concavity_bug() {
    // Upstream golden test: (0, 1), (1, 10), (3, 8)
    let (xm, ym) = find_max_quadratic(0.0, 1.0, 1.0, 10.0, 3.0, 8.0).unwrap();
    assert!((xm - 1.85).abs() <= 1e-14);
    assert!((ym - 12.408_333_333_333_333).abs() <= 1e-14);

    // Reverse knot order: (3, 8), (1, 10), (0, 1)
    let (xm_rev, ym_rev) =
        find_max_quadratic(3.0, 8.0, 1.0, 10.0, 0.0, 1.0).unwrap();
    assert!((xm_rev - 1.85).abs() <= 1e-14);
    assert!((ym_rev - 12.408_333_333_333_333).abs() <= 1e-14);

    // Non-interior maximum (y3 > y2 >= y1) must be rejected by the fixed check
    let res_bad = find_max_quadratic(0.0, 1.0, 1.0, 5.0, 2.0, 10.0);
    assert!(res_bad.is_err());
}

/// Verifies [`dfridr`] and [`integrate`] against upstream `test_integrate.f90`
/// (`simp38` golden integrals of $x$, $\sin x$, and $e^x$).
#[test]
fn test_dfridr_and_integrate_match_upstream_simp38_oracles() {
    // Derivative of sin(x) at x = 1.0 is cos(1.0)
    let x0 = 1.0_f64;
    let (df, err) = dfridr(0.1, |h| (x0 + h).sin()).unwrap();
    assert!((df - x0.cos()).abs() <= 1e-12);
    assert!(err <= 1e-11);

    let pi = std::f64::consts::PI;
    // Upstream test 1: int_0^1 x dx = 0.5
    let (i1, _) = integrate(|x| x, 0.0, 1.0, 1e-10, 10_000).unwrap();
    assert!((i1 - 0.5).abs() <= 1e-12);
    // Upstream test 2: int_0^pi sin(x) dx = 2.0
    let (i2, _) = integrate(f64::sin, 0.0, pi, 1e-8, 10_000).unwrap();
    assert!((i2 - 2.0).abs() <= 1e-8);
    // Upstream test 3: int_0^{2*pi} sin(x) dx = 0.0
    let (i3, _) = integrate(f64::sin, 0.0, 2.0 * pi, 1e-8, 10_000).unwrap();
    assert!(i3.abs() <= 1e-8);
    // Upstream test 4: int_0^2 exp(x) dx = exp(2) - 1
    let (i4, _) = integrate(f64::exp, 0.0, 2.0, 1e-8, 10_000).unwrap();
    assert!((i4 - (2.0_f64.exp() - 1.0)).abs() <= 1e-8);
    // Upstream test 5: int_0^{10} exp(x) dx = exp(10) - 1
    let (i5, _) = integrate(f64::exp, 0.0, 10.0, 1e-6, 50_000).unwrap();
    assert!(
        (i5 - (10.0_f64.exp() - 1.0)).abs() <= 1e-6 * (10.0_f64.exp() - 1.0)
    );
}

/// Verifies [`brent_safe_zero`], [`brent_local_min`], and [`brent_global_min`]
/// against upstream `test_brent.f90`.
#[test]
fn test_brent_safe_zero_and_minimization_match_analytical_oracles() {
    // Exact root of x - 3*sin(1 - x) = 0 is 0.7480061107481244
    let (xz, fz) = brent_safe_zero(
        0.0,
        10.0,
        1e-14,
        1e-14,
        1e-14,
        |x| Ok(x - 3.0 * (1.0 - x).sin()),
        None,
        None,
        200,
    )
    .unwrap();
    assert!((xz - 0.748_006_110_748_124_4).abs() <= 1e-12);
    assert!(fz.abs() <= 1e-12);

    // Local minimum of (x - 2)^2 - 1 on [-10, 10] at x = 2, f = -1
    let (xm, fm) = brent_local_min(200, -10.0, 10.0, 1e-10, 1e-10, |x| {
        Ok((x - 2.0) * (x - 2.0) - 1.0)
    })
    .unwrap();
    assert!((xm - 2.0).abs() <= 1e-8);
    assert!((fm - (-1.0)).abs() <= 1e-14);

    // Global minimum on multimodal quartic (x^2 - 4)^2 + 0.2*x on [-3, 3]
    let (xg, fg) =
        brent_global_min(400, -3.0, 3.0, 0.5, 50.0, 1e-8, 1e-8, |x| {
            let u = x * x - 4.0;
            Ok(u * u + 0.2 * x)
        })
        .unwrap();
    assert!(xg < -1.9 && fg < -0.35);
}

/// Verifies [`safe_root_with_brackets`], [`safe_root_with_guess`], and
/// [`safe_root_without_brackets`] against `test_num.f90` (`test_root1`..`3`).
#[test]
fn test_safe_root_with_brackets_and_guess_match_upstream_golden_values() {
    // Upstream test_root_with_brackets: f(x) = x - 3*sin(1 - x) on [0, 10]
    let f1 = |x: f64| -> Result<(f64, f64), NumError> {
        Ok((x - 3.0 * (1.0 - x).sin(), 1.0 + 3.0 * (1.0 - x).cos()))
    };
    let ((x1, x3), (y1, y3)) =
        look_for_brackets(5.0, 1.0, 0.0, 10.0, f1, 50).unwrap();
    let (root1, res1) =
        safe_root_with_brackets(f1, x1, x3, y1, y3, 50, 1e-12, 1e-12).unwrap();
    assert!((root1 - 0.748_006_110_748_124_4).abs() <= 1e-11);
    assert!(res1.abs() <= 1e-11);

    // Upstream test_root2 / test_root3: f(x) = tanh(x) - tanh(0.5) => x* = 0.5
    let f2 = |x: f64| -> Result<(f64, f64), NumError> {
        let t = x.tanh();
        Ok((t - 0.462_117_157_260_009_8, 1.0 - t * t))
    };
    let (root2, _) = safe_root_without_brackets(
        f2, 0.0, 0.1, 10, 50, -10.0, 10.0, 1e-12, 1e-12,
    )
    .unwrap();
    assert!((root2 - 0.5).abs() <= 1e-10);

    let (root3, _) = safe_root_with_guess(
        f2,
        0.1,
        0.2,
        Some(-1.0),
        Some(2.0),
        Some(f2(-1.0).unwrap().0),
        Some(f2(2.0).unwrap().0),
        5,
        50,
        -10.0,
        10.0,
        1e-12,
        1e-12,
    )
    .unwrap();
    assert!((root3 - 0.5).abs() <= 1e-10);
}

/// Verifies [`safe_root_dual`] implicit differentiation on
/// $f(x, p_0, p_1) = p_0 x^3 + x - p_1 = 0$, where
/// $\partial x^*/\partial p_0 = -(x^*)^3 / (3 p_0 (x^*)^2 + 1)$ and
/// $\partial x^*/\partial p_1 = 1 / (3 p_0 (x^*)^2 + 1)$.
#[test]
fn test_safe_root_dual_matches_implicit_function_theorem_derivatives() {
    let p0 = 2.0_f64;
    let p1 = 5.0_f64;
    let p0_dual = DualOrder1::<2>::independent(p0, 0);
    let p1_dual = DualOrder1::<2>::independent(p1, 1);

    let root_dual = safe_root_dual(
        |x| Ok((p0 * x * x * x + x - p1, 3.0 * p0 * x * x + 1.0)),
        |x_star| {
            let xd = DualOrder1::<2>::constant(x_star);
            p0_dual * xd * xd * xd + xd - p1_dual
        },
        0.0,
        3.0,
        100,
        1e-14,
        1e-14,
    )
    .unwrap();

    let xs = root_dual.val;
    assert!((p0 * xs.powi(3) + xs - p1).abs() <= 1e-13);
    let denom = 3.0 * p0 * xs * xs + 1.0;
    let exact_dp0 = -(xs.powi(3)) / denom;
    let exact_dp1 = 1.0 / denom;
    assert!((root_dual.d1[0] - exact_dp0).abs() <= 1e-12);
    assert!((root_dual.d1[1] - exact_dp1).abs() <= 1e-12);
}

/// Verifies [`nm_simplex`] on the 2D Rosenbrock banana function and checks
/// [`r8_uniform_01`], [`i4_uniform`], and [`perm_uniform`].
#[test]
fn test_nm_simplex_and_park_miller_prng_match_upstream_oracles() {
    let mut seed = 12_345_i32;
    let u = r8_uniform_01(&mut seed);
    assert!((0.0..1.0).contains(&u));
    let k = i4_uniform(1, 10, &mut seed);
    assert!((1..=10).contains(&k));
    let mut p = perm_uniform(6, &mut seed);
    p.sort_unstable();
    assert_eq!(p, vec![1, 2, 3, 4, 5, 6]);

    // Minimize 2D Rosenbrock banana f(x, y) = (1 - x)^2 + 100*(y - x^2)^2
    let cfg = SimplexConfig {
        max_evals: 2_000,
        x_tol: 1e-10,
        f_tol: 1e-14,
        ..SimplexConfig::default()
    };
    let res =
        nm_simplex(2, &[-2.0, -2.0], &[2.0, 2.0], &[-0.5, 0.8], &cfg, |x| {
            let t1 = 1.0 - x[0];
            let t2 = x[1] - x[0] * x[0];
            Ok(t1 * t1 + 100.0 * t2 * t2)
        })
        .unwrap();
    assert!((res.x_min[0] - 1.0).abs() <= 1e-4);
    assert!((res.x_min[1] - 1.0).abs() <= 1e-4);
    assert!(res.f_min <= 1e-8);
}

/// Evaluates Fletcher (`1965`) Chebyquad objective used in `test_newuoa.f90`
/// and `test_bobyqa.f90`.
fn chebyquad(x: &[f64]) -> Result<f64, NumError> {
    let n = x.len();
    let mut y = vec![vec![0.0_f64; n]; n + 2];
    for j in 0..n {
        y[0][j] = 1.0;
        y[1][j] = 2.0 * x[j] - 1.0;
    }
    for i in 1..n {
        for j in 0..n {
            y[i + 1][j] = 2.0 * y[1][j] * y[i][j] - y[i - 1][j];
        }
    }
    let mut f = 0.0_f64;
    let mut iw = 1_i32;
    for i in 1..=(n + 1) {
        let mut s: f64 = y[i - 1].iter().sum::<f64>() / (n as f64);
        if iw > 0 {
            let ii = i as f64;
            s += 1.0 / (ii * ii - 2.0 * ii);
        }
        iw = -iw;
        f += s * s;
    }
    Ok(f)
}

/// Verifies [`newuoa`] and [`bobyqa`] on upstream `do_test_newuoa` and
/// `do_test_bobyqa` (`N = 2, 4, 6`, `NPT = 2*N + 1`, `num/test/test_output:123-130`).
#[test]
fn test_newuoa_and_bobyqa_match_upstream_chebyquad_suite() {
    for n in [2usize, 4, 6] {
        let npt = 2 * n + 1;
        let x0: Vec<f64> =
            (1..=n).map(|i| (i as f64) / ((n + 1) as f64)).collect();
        let rhobeg = 0.2 * x0[0];
        let mut x_new = x0.clone();
        let res_new =
            newuoa(n, npt, &mut x_new, rhobeg, 1e-6, 5_000, 1e99, chebyquad)
                .unwrap();
        assert!(res_new.f_min.abs() <= 1e-10, "newuoa N={n} f={}", res_new.f_min);

        let mut x_bob = x0.clone();
        let xl = vec![-1.0_f64; n];
        let xu = vec![1.0_f64; n];
        let res_bob = bobyqa(
            n, npt, &mut x_bob, &xl, &xu, rhobeg, 1e-6, 5_000, 1e99, chebyquad,
        )
        .unwrap();
        assert!(res_bob.f_min.abs() <= 1e-10, "bobyqa N={n} f={}", res_bob.f_min);
    }
}

/// Verifies [`nm_simplex`] on all 6 benchmark functions (`FR4`, `ER`, `WD`,
/// `BLE`, `PS`, `TR`) from `num/test/src/test_simplex.f90` (`num/test/test_output:131-149`).
#[test]
fn test_nm_simplex_matches_all_six_upstream_benchmarks() {
    let base = SimplexConfig {
        max_evals: 10_000,
        x_tol: 1e-10,
        f_tol: 1e-14,
        enforce_bounds: true,
        adaptive_random_search: true,
        seed: 1_074_698_122,
        centroid_weight_power: 1.0,
        ..SimplexConfig::default()
    };
    let fr4_cfg = SimplexConfig { enforce_bounds: false, ..base };
    let r_fr4 = nm_simplex(4, &[-2.0; 4], &[6.0; 4], &[0.5, -2.0, 0.5, -2.0], &fr4_cfg, |x| {
        let mut s = 0.0;
        for i in 0..2 {
            let (x1, x2) = (x[2 * i], x[2 * i + 1]);
            let t1 = -13.0 + x1 + ((5.0 - x2) * x2 - 2.0) * x2;
            let t2 = -29.0 + x1 + ((1.0 + x2) * x2 - 14.0) * x2;
            s += t1 * t1 + t2 * t2;
        }
        Ok(s)
    }).unwrap();
    assert!(r_fr4.f_min < 1e-10);

    let r_er = nm_simplex(4, &[-2.0; 4], &[5.0; 4], &[3.0; 4], &base, |x| {
        Ok((0..3).map(|i| 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2)).sum())
    }).unwrap();
    assert!(r_er.f_min < 1e-10);

    let r_wd = nm_simplex(4, &[-4.0, -2.0, -4.0, -2.0], &[2.0; 4], &[-3.0, -1.0, -3.0, -1.0], &base, |x| {
        Ok(100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2)
            + 90.0 * (x[3] - x[2] * x[2]).powi(2) + (1.0 - x[2]).powi(2)
            + 100.0 * (x[1] + x[3] - 2.0).powi(2) + (x[1] - x[3]).powi(2) / 10.0_f64.sqrt())
    }).unwrap();
    assert!(r_wd.f_min < 1e-10);

    let r_ble = nm_simplex(4, &[-2.0; 4], &[5.0; 4], &[3.0; 4], &base, |x| {
        Ok((0..2).map(|i| {
            let (u, v) = (x[2 * i], x[2 * i + 1]);
            (1.5 - u * (1.0 - v)).powi(2) + (2.25 - u * (1.0 - v * v)).powi(2)
                + (2.625 - u * (1.0 - v * v * v)).powi(2)
        }).sum())
    }).unwrap();
    assert!(r_ble.f_min < 1e-10);

    let r_ps = nm_simplex(4, &[-1.0; 4], &[3.0; 4], &[3.0, -1.0, 0.0, 1.0], &base, |x| {
        Ok((x[0] + 10.0 * x[1]).powi(2) + 5.0 * (x[2] - x[3]).powi(2)
            + (x[1] - 2.0 * x[2]).powi(4) + 10.0 * (x[0] - x[3]).powi(4))
    }).unwrap();
    assert!(r_ps.f_min < 1e-10);

    let r_tr = nm_simplex(4, &[-0.1; 4], &[0.1; 4], &[0.01; 4], &base, |x| {
        let sc: f64 = x.iter().map(|&v| v.cos()).sum();
        Ok((0..4).map(|i| (4.0 - sc + ((i + 1) as f64) * (1.0 - x[i].cos()) - x[i].sin()).powi(2)).sum())
    }).unwrap();
    assert!(r_tr.f_min < 1e-10);
}

/// Verifies [`glomin`], [`local_min`], and [`brent_safe_zero`] across all 10
/// minimization functions (`h_01..h_05`, `g_01..g_05`) and 5 root functions
/// (`f_01..f_05`) from `num/test/src/test_brent.f90` (`num/test/test_output:150-206`).
#[test]
fn test_brent_all_ten_minimization_and_five_root_functions() {
    let machep = f64::EPSILON;
    let e = machep.sqrt();
    for m in [0.0, 100.0] {
        let (x, fx) = glomin(10_000, 7.0, 9.0, 8.0, m, e, e, |x| Ok(2.0 - x)).unwrap();
        assert!((x - 9.0).abs() <= 1e-6 && (fx - (-7.0)).abs() <= 1e-6);
    }
    for m in [2.0, 2.1] {
        let (x, fx) = glomin(10_000, -1.0, 2.0, 0.5, m, e, e, |x| Ok(x * x)).unwrap();
        assert!(x.abs() <= 1e-6 && fx.abs() <= 1e-12);
    }
    for m in [14.0, 28.0] {
        let (x, fx) = glomin(10_000, -0.5, 2.0, 0.75, m, e, e, |x| Ok(x * x * (x + 1.0))).unwrap();
        assert!(x.abs() <= 1e-5 && fx.abs() <= 1e-10);
    }
    let (x4, fx4) = glomin(10_000, -10.0, 10.0, 0.0, 72.0, e, e, |x| Ok((x + x.sin()) * (-x * x).exp())).unwrap();
    assert!((x4 - (-0.679_578_66)).abs() <= 1e-5 && fx4 < -0.824);
    let (x5, fx5) = glomin(10_000, -10.0, 10.0, 0.0, 72.0, e, e, |x| Ok((x - x.sin()) * (-x * x).exp())).unwrap();
    assert!((x5 - (-0.679_578_66)).abs() <= 2.0 && fx5 < -0.05, "x5={x5}, fx5={fx5}");

    let eps = 10.0 * e;
    let (xg1, fg1) = local_min(10_000, 0.0, std::f64::consts::PI, eps, eps, |x| Ok((x - 2.0).powi(2) + 1.0)).unwrap();
    assert!((xg1 - 2.0).abs() <= 1e-6 && (fg1 - 1.0).abs() <= 1e-12);
    let (xg2, _) = local_min(10_000, 0.0, 1.0, eps, eps, |x| Ok(x * x + (-x).exp())).unwrap();
    assert!((2.0 * xg2 - (-xg2).exp()).abs() <= 1e-6);
    let (xg3, _) = local_min(10_000, -2.0, 2.0, eps, eps, |x| Ok(((x * x + 2.0) * x + 1.0) * x + 3.0)).unwrap();
    assert!((4.0 * xg3.powi(3) + 4.0 * xg3 + 1.0).abs() <= 1e-6);
    let (xg4, _) = local_min(10_000, 1e-4, 1.0, eps, eps, |x| Ok(x.exp() + 0.01 / x)).unwrap();
    assert!((xg4.exp() - 0.01 / (xg4 * xg4)).abs() <= 1e-5);
    let (xg5, _) = local_min(10_000, 2e-4, 2.0, eps, eps, |x| Ok(x.exp() - 2.0 * x + 0.01 / x - 1e-6 / (x * x))).unwrap();
    assert!(xg5 > 0.05 && xg5 < 1.0);

    let root_cases: [(f64, f64, fn(f64) -> f64); 5] = [
        (1.0, 2.0, |x| x.sin() - 0.5 * x),
        (0.0, 1.0, |x| 2.0 * x - (-x).exp()),
        (-1.0, 0.5, |x| x * (-x).exp()),
        (1e-4, 20.0, |x| x.exp() - 0.01 / (x * x)),
        (-5.0, 2.0, |x| (x + 3.0) * (x - 1.0) * (x - 1.0)),
    ];
    for (a, b, f) in root_cases {
        let (_z, fz) = brent_safe_zero(a, b, machep, machep, 0.0, |x| Ok(f(x)), Some(f(a)), Some(f(b)), 500).unwrap();
        assert!(fz.abs() < 1e-14);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Verifies that [`brent_safe_zero`] and [`safe_root_with_brackets`] agree
    /// to machine precision on cubic polynomials $x^3 + a x - b = 0$ with
    /// $a > 0$.
    #[test]
    fn prop_brent_and_safe_root_agree_on_monotonic_cubics(
        a in 0.5_f64..5.0_f64,
        b in -10.0_f64..10.0_f64,
    ) {
        let f = |x: f64| x * x * x + a * x - b;
        let df = |x: f64| 3.0 * x * x + a;
        let (xb, fb) = brent_safe_zero(
            -5.0,
            5.0,
            1e-13,
            1e-13,
            1e-13,
            |x| Ok(f(x)),
            None,
            None,
            200,
        )
        .unwrap();
        let (xs, fs) = safe_root_with_brackets(
            |x| Ok((f(x), df(x))),
            -5.0,
            5.0,
            f(-5.0),
            f(5.0),
            200,
            1e-13,
            1e-13,
        )
        .unwrap();
        prop_assert!((xb - xs).abs() <= 1e-11);
        prop_assert!(fb.abs() <= 1e-11 && fs.abs() <= 1e-11);
    }
}
