// SPDX-License-Identifier: GPL-3.0-or-later
//! Self-contained unit and synthetic-model integration tests for Package A2a
//! (`gensr/`, `rhs`, `solve`, `driver`) (`#171`, `#479`, `#147`).

use flexmesa_adipls::{
    adirhs, aramax, danorm, deriv2, derive, derivk, geninh, leq, leqdet, leqsd,
    lininh4, lir, lir1, mchcf1, mchcff, mchsol, notwin, notwni, order,
    parse_adipls_pruned_in, res_adimod, rsnfit, sclasl, set_adipls_controls,
    setbcs, setlog, setups_adi, shtint, sigint, signew, sigscn, sigsol, stevft,
    store, takata, vinta, vintk, xlmult, zero, AdiStatusLogEntry,
    AdiplsControlState, AdiplsError, AdiplsSolver, AdiplsTapEvent, AmdlAa,
    BcsDat, Cincnt, Cnrkbc, Cprcns, Csexpc, Csummm, DerivkCache, LirState,
    NoopTapObserver, RhsDat, RotDat, SignwPgState, SigoutArgs, SigoutHandler,
    SolverContext, TapRecorder, VintkCache, XlmultCache, ADIPLS_DEFAULT_EPSPRC,
    ADIPLS_DEFAULT_EPSUFL, IVARMD,
};
use flexmesa_const::STANDARD_CGRAV;
use flexmesa_math::{cos, sin};

const ADIPLS_PRUNED_IN: &str = "\
2  'amdl.mesa'
9  'ttt.adipls.prt'
11 'agsm.mesa'
15 'ttt.adipls.ssm'
16 'ttt.adipls.fsm'
-1 ''
mod.osc.cst.int.out
,,,,,,,
,,1,,,,,,,,,,,,,,,,,,,,
   0,3,0,1,,,,,,,
    1,  220,   ,    1,10,,,,,,,,
,2,100,1000,,,,,,,,,,,,
,,,,,,,,,,,,,,,,
6.67430e-8
,,0,,
1,,,,,,,,,,,,,,,
5,0,0.99,,,,,,,,,,,,,,
,,,,,-1,,,,,,,,,,,,,,
9,10,,20,1,,,,,,,,
1,,,,,,,,,,,,,,,
0,,,,6,,,,,,
,,,,,-5000,100,,,,,,,,,,,,,,,,,,,,,,,,
";

#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
fn build_polytrope_n3_amdl(nn: usize) -> ([f64; 8], Vec<f64>, AmdlAa) {
    // Solve Lane-Emden n = 3 (`theta'' + 2/xi * theta' + theta^3 = 0`) on a
    // fine RK4 grid so interpolated nodes at `nn` points reach `1e-12` accuracy.
    let sub = 10_000_usize;
    let dxi = 7.0_f64 / (sub as f64);
    let mut xis = Vec::with_capacity(sub + 1);
    let mut thetas = Vec::with_capacity(sub + 1);
    let mut dthetas = Vec::with_capacity(sub + 1);

    let mut xi = 0.0_f64;
    let mut th = 1.0_f64;
    let mut dth = 0.0_f64;
    xis.push(xi);
    thetas.push(th);
    dthetas.push(dth);

    let rhs_le = |r: f64, y1: f64, y2: f64| -> (f64, f64) {
        if r == 0.0_f64 {
            (0.0_f64, -1.0_f64 / 3.0_f64)
        } else {
            (y2, -2.0_f64 * y2 / r - y1 * y1 * y1)
        }
    };

    for step in 0..sub {
        // Use analytic power series across step 0 to avoid `0/0` at origin.
        if step == 0 {
            xi = dxi;
            let xi2 = xi * xi;
            let xi4 = xi2 * xi2;
            th = 1.0_f64 - xi2 / 6.0_f64 + xi4 / 40.0_f64;
            dth = -xi / 3.0_f64 + xi * xi2 / 10.0_f64;
        } else {
            let (k1a, k1b) = rhs_le(xi, th, dth);
            let (k2a, k2b) = rhs_le(
                xi + 0.5_f64 * dxi,
                th + 0.5_f64 * dxi * k1a,
                dth + 0.5_f64 * dxi * k1b,
            );
            let (k3a, k3b) = rhs_le(
                xi + 0.5_f64 * dxi,
                th + 0.5_f64 * dxi * k2a,
                dth + 0.5_f64 * dxi * k2b,
            );
            let (k4a, k4b) = rhs_le(xi + dxi, th + dxi * k3a, dth + dxi * k3b);
            xi += dxi;
            th += (dxi / 6.0_f64) * (k1a + 2.0_f64 * (k2a + k3a) + k4a);
            dth += (dxi / 6.0_f64) * (k1b + 2.0_f64 * (k2b + k3b) + k4b);
        }
        xis.push(xi);
        thetas.push(th);
        dthetas.push(dth);
        if th <= 0.0_f64 {
            break;
        }
    }

    // Refine root `theta(xi1) = 0` via secant / linear interpolation + 2 RK4 steps.
    let m = xis.len();
    assert!(m >= 4 && thetas[m - 1] <= 0.0_f64);
    let xi_a = xis[m - 2];
    let th_a = thetas[m - 2];
    let dth_a = dthetas[m - 2];
    let xi_b = xis[m - 1];
    let th_b = thetas[m - 1];
    let mut xi1 = xi_a - th_a * (xi_b - xi_a) / (th_b - th_a);
    let mut dth1 = dth_a;
    for _ in 0..4 {
        let h = xi1 - xi_a;
        let (k1a, k1b) = rhs_le(xi_a, th_a, dth_a);
        let (k2a, k2b) = rhs_le(
            xi_a + 0.5_f64 * h,
            th_a + 0.5_f64 * h * k1a,
            dth_a + 0.5_f64 * h * k1b,
        );
        let (k3a, k3b) = rhs_le(
            xi_a + 0.5_f64 * h,
            th_a + 0.5_f64 * h * k2a,
            dth_a + 0.5_f64 * h * k2b,
        );
        let (k4a, k4b) = rhs_le(xi1, th_a + h * k3a, dth_a + h * k3b);
        let th_end = th_a + (h / 6.0_f64) * (k1a + 2.0_f64 * (k2a + k3a) + k4a);
        dth1 = dth_a + (h / 6.0_f64) * (k1b + 2.0_f64 * (k2b + k3b) + k4b);
        xi1 -= th_end / dth1;
    }

    let rho_c_ratio = -xi1 / (3.0_f64 * dth1);
    let gamma1 = 5.0_f64 / 3.0_f64;
    let inv_gamma1 = 1.0_f64 / gamma1;

    let mut x = Vec::with_capacity(nn);
    let mut r1 = Vec::with_capacity(nn);
    let mut r2 = Vec::with_capacity(nn);
    let mut r3 = Vec::with_capacity(nn);
    let mut r4 = Vec::with_capacity(nn);
    let mut r5 = Vec::with_capacity(nn);
    let mut r10 = Vec::with_capacity(nn);

    let mut lir_th = LirState::default();
    let mut lir_deriv = LirState::default();
    let mut th_val = [0.0_f64; 1];
    let mut dth_val = [0.0_f64; 1];

    for idx in 0..nn {
        let xn = (idx as f64) / ((nn - 1) as f64);
        x.push(xn);
        r3.push(inv_gamma1);
        r10.push(1.0_f64);
        if idx == 0 {
            r1.push(rho_c_ratio);
            r2.push(0.0_f64);
            r4.push(0.0_f64);
            r5.push(3.0_f64);
        } else if idx == nn - 1 {
            r1.push(1.0_f64);
            r2.push(1.0e12_f64);
            r4.push(0.0_f64);
            r5.push(0.0_f64);
        } else {
            let xi_pt = xn * xi1;
            let _ = lir(
                xi_pt,
                &xis,
                &mut th_val,
                &thetas,
                1,
                1,
                xis.len(),
                1,
                &mut lir_th,
            );
            let _ = lir(
                xi_pt,
                &xis,
                &mut dth_val,
                &dthetas,
                1,
                1,
                xis.len(),
                1,
                &mut lir_deriv,
            );
            let thp = th_val[0];
            let deriv_pt = dth_val[0];
            let a1 = deriv_pt / (xn * dth1);
            let v_factor = -xi_pt * deriv_pt / thp;
            let vg = 4.0_f64 * inv_gamma1 * v_factor;
            let a_star = (3.0_f64 - 4.0_f64 * inv_gamma1) * v_factor;
            let u = -xi_pt * (thp * thp * thp) / deriv_pt;
            r1.push(a1);
            r2.push(vg);
            r4.push(a_star);
            r5.push(u);
        }
    }

    let mut aa = AmdlAa::new_unset(nn);
    aa.set_row(1, r1).unwrap();
    aa.set_row(2, r2).unwrap();
    aa.set_row(3, r3).unwrap();
    aa.set_row(4, r4).unwrap();
    aa.set_row(5, r5).unwrap();
    aa.set_row(10, r10).unwrap();

    let p2 = (4.0_f64 / 3.0_f64) * inv_gamma1 * (xi1 * xi1);
    let rh2 = xi1 * xi1;
    let data = [
        1.989e33_f64,
        6.957e10_f64,
        rho_c_ratio * 3.0_f64 * STANDARD_CGRAV * 1.989e33_f64
            / (6.957e10_f64 * 6.957e10_f64 * 6.957e10_f64),
        0.0_f64,
        p2,
        rh2,
        3.0_f64, // polytropic surface index n = 3 (`singsf = true`)
        0.0_f64, // `iggt = 0` so row 6 is untouched (#479)
    ];
    (data, x, aa)
}

#[test]
#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
fn test_gensr_linear_algebra_quadrature_diff_and_interp_identities() {
    // 1. `zero`, `store`, `notwin`, `notwni`
    let mut buf = [9.0_f64; 5];
    zero(&mut buf, 3);
    assert_eq!(buf, [0.0, 0.0, 0.0, 9.0, 9.0]);
    let mut dst = [-1.0_f64; 4];
    store(&[1.5, 2.5, 3.5], &mut dst, 3);
    assert_eq!(dst, [1.5, 2.5, 3.5, -1.0]);

    assert!(!notwin(1.0, 2.0, 1.5));
    assert!(notwin(1.0, 2.0, 2.1));
    assert!(!notwin(3.0, 1.0, 5.0)); // `xw1 > xw2` disables windowing
    assert!(!notwni(2, 5, 2));
    assert!(notwni(2, 5, 6));
    assert!(!notwni(5, 2, 10));

    // 2. `leq`, `leqdet`, `leqsd` against exact 3x3 closed-form solution and det
    // A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], det(A) = -1, x = [2, 3, -1]
    #[rustfmt::skip]
    let a_orig = [
        2.0_f64, -3.0, -2.0, // col 0
        1.0, -1.0, 1.0,      // col 1
        -1.0, 2.0, 2.0,      // col 2
    ];
    let mut a = a_orig;
    let mut b = [8.0_f64, -11.0, -3.0];
    let det = leq(&mut a, &mut b, 3, 1, 3, 3);
    assert!((det + 1.0_f64).abs() < 1.0e-13_f64);
    assert!((b[0] - 2.0_f64).abs() < 1.0e-13_f64);
    assert!((b[1] - 3.0_f64).abs() < 1.0e-13_f64);
    assert!((b[2] + 1.0_f64).abs() < 1.0e-13_f64);

    let mut a_det = a_orig;
    let mut dummy = [0.0_f64; 3];
    assert!(
        (leqdet(&mut a_det, &mut dummy, 3, 0, 3, 3) + 1.0_f64).abs()
            < 1.0e-13_f64
    );
    let mut a_sd = a_orig;
    let mut b_sd = [8.0_f64, -11.0, -3.0];
    assert!(
        (leqsd(&mut a_sd, &mut b_sd, 3, 1, 3, 3) + 1.0_f64).abs() < 1.0e-13_f64
    );

    // Singular matrix returns 0
    let mut a_sing = [1.0_f64, 2.0, 2.0, 4.0];
    let mut b_sing = [1.0_f64, 2.0];
    assert_eq!(leq(&mut a_sing, &mut b_sing, 2, 1, 2, 2).to_bits(), 0);

    // 3. `lininh4` solving `y1' = y2, y2' = -y1` forward & backward (`O(h^4)`)
    let n_ode = 401_usize;
    let x_ode: Vec<f64> = (0..n_ode).map(|i| (i as f64) / 400.0_f64).collect();
    let mut y_ode = vec![0.0_f64; 2 * n_ode];
    y_ode[0] = 0.0_f64;
    y_ode[1] = 1.0_f64;
    lininh4(
        &x_ode,
        &mut y_ode,
        2,
        2,
        1,
        1,
        1,
        n_ode,
        0,
        0,
        |_xv, fd, finh, _nd| {
            fd[0][0] = 0.0_f64;
            fd[0][1] = 1.0_f64;
            fd[1][0] = -1.0_f64;
            fd[1][1] = 0.0_f64;
            finh[0] = 0.0_f64;
            finh[1] = 0.0_f64;
            Ok(())
        },
    )
    .expect("lininh4 forward");
    let end_base = (n_ode - 1) * 2;
    assert!((y_ode[end_base] - sin(1.0_f64)).abs() < 1.0e-12_f64);
    assert!((y_ode[end_base + 1] - cos(1.0_f64)).abs() < 1.0e-12_f64);

    // Backward integration recovers initial state
    lininh4(
        &x_ode,
        &mut y_ode,
        2,
        2,
        1,
        -1,
        n_ode,
        n_ode,
        n_ode - 1,
        n_ode - 1,
        |_xv, fd, finh, _nd| {
            fd[0][0] = 0.0_f64;
            fd[0][1] = 1.0_f64;
            fd[1][0] = -1.0_f64;
            fd[1][1] = 0.0_f64;
            finh[0] = 0.0_f64;
            finh[1] = 0.0_f64;
            Ok(())
        },
    )
    .expect("lininh4 backward");
    assert!(y_ode[0].abs() < 1.0e-12_f64);
    assert!((y_ode[1] - 1.0_f64).abs() < 1.0e-12_f64);

    // 4. `derive`, `deriv2`, `derivk` on a non-uniform grid
    let xg: Vec<f64> = (0..11)
        .map(|i| {
            let t = 0.1_f64 * f64::from(i);
            t + 0.2_f64 * t * t
        })
        .collect();
    let y_quad: Vec<f64> = xg
        .iter()
        .map(|&x| 3.0_f64 * x * x - 5.0_f64 * x + 2.0_f64)
        .collect();
    let mut dy = vec![0.0_f64; 11];
    let mut d2y = vec![0.0_f64; 11];
    derive(&xg, &y_quad, &mut dy, 11, 1, 1, 1, 1);
    deriv2(&xg, &y_quad, &mut d2y, 11, 1, 1, 1, 1);
    for (&x, (&g1, &g2)) in xg.iter().zip(dy.iter().zip(d2y.iter())) {
        assert!((g1 - (6.0_f64 * x - 5.0_f64)).abs() < 1.0e-12_f64);
        assert!((g2 - 6.0_f64).abs() < 1.0e-11_f64);
    }

    let y_quart: Vec<f64> = xg
        .iter()
        .map(|&x| x * x * x * x - 2.0_f64 * x * x + 3.0_f64 * x)
        .collect();
    let mut dyk = vec![0.0_f64; 11];
    let mut dcache = DerivkCache::default();
    assert_eq!(
        derivk(&xg, &y_quart, &mut dyk, 1, 11, 1, 1, 2, &mut dcache),
        1
    );
    for (&x, &g) in xg.iter().zip(dyk.iter()) {
        let exact = 4.0_f64 * x * x * x - 4.0_f64 * x + 3.0_f64;
        assert!((g - exact).abs() < 1.0e-11_f64);
    }

    // 5. `vinta` and `vintk` on non-uniform grids (both odd and even length)
    for &nn_v in &[10_usize, 11_usize] {
        let xv: Vec<f64> = (0..nn_v)
            .map(|i| {
                let t = 0.1_f64 * (i as f64);
                t + 0.15_f64 * t * t
            })
            .collect();
        let fq: Vec<f64> = xv
            .iter()
            .map(|&x| 3.0_f64 * x * x - 2.0_f64 * x + 1.0_f64)
            .collect();
        let mut int_v = vec![0.0_f64; nn_v];
        vinta(&xv, &fq, &mut int_v, nn_v, 1, 1);
        for (&x, &got) in xv.iter().zip(int_v.iter()) {
            let exact = x * x * x - x * x + x;
            assert!((got - exact).abs() < 1.0e-12_f64);
        }

        let fc: Vec<f64> = xv
            .iter()
            .map(|&x| 4.0_f64 * x * x * x - 3.0_f64 * x * x + 2.0_f64 * x)
            .collect();
        let mut int_k = vec![0.0_f64; nn_v];
        let mut vcache = VintkCache::default();
        assert_eq!(
            vintk(&xv, &fc, &mut int_k, 1, nn_v, 1, 1, 2, &mut vcache),
            1
        );
        for (&x, &got) in xv.iter().zip(int_k.iter()) {
            let exact = x * x * x * x - x * x * x + x * x;
            assert!((got - exact).abs() < 1.0e-12_f64);
        }
        // Cached `karg = 0` reuse produces identical result
        let mut int_k2 = vec![0.0_f64; nn_v];
        assert_eq!(
            vintk(&xv, &fc, &mut int_k2, 1, nn_v, 1, 1, 0, &mut vcache),
            1
        );
        assert_eq!(int_k, int_k2);
    }

    // 6. `lir`, `lir1`, and `aramax`
    let y_cub: Vec<f64> = xg
        .iter()
        .map(|&x| 2.0_f64 * x * x * x - x * x + 3.0_f64 * x - 4.0_f64)
        .collect();
    let mut lstate = LirState::default();
    let mut y_out = [0.0_f64; 1];
    let zp = 0.55_f64;
    assert_eq!(
        lir(zp, &xg, &mut y_out, &y_cub, 1, 1, 11, 1, &mut lstate),
        1
    );
    let exact_cub = 2.0_f64 * zp * zp * zp - zp * zp + 3.0_f64 * zp - 4.0_f64;
    assert!((y_out[0] - exact_cub).abs() < 1.0e-12_f64);
    assert_eq!(
        lir(2.5_f64, &xg, &mut y_out, &y_cub, 1, 1, 11, 1, &mut lstate),
        0
    );

    let y_lin: Vec<f64> = xg.iter().map(|&x| 4.0_f64 * x - 1.0_f64).collect();
    let mut lstate1 = LirState::default();
    assert_eq!(
        lir1(0.42_f64, &xg, &mut y_out, &y_lin, 1, 1, 11, 1, &mut lstate1),
        1
    );
    assert!((y_out[0] - (4.0_f64 * 0.42_f64 - 1.0_f64)).abs() < 1.0e-13_f64);

    let xa = [0.0_f64, 0.2, 0.4, 0.6, 0.8];
    let ya: Vec<f64> = xa
        .iter()
        .map(|&x| {
            let d = x - 0.37_f64;
            10.0_f64 - d * d
        })
        .collect();
    let (xext, aext, d2ext) = aramax(&xa, &ya, 1, xa.len());
    assert!((xext - 0.37_f64).abs() < 1.0e-13_f64);
    assert!((aext - 10.0_f64).abs() < 1.0e-13_f64);
    assert!((d2ext + 2.0_f64).abs() < 1.0e-12_f64);
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_479_unset_amdl_row6_errors_and_res_adimod_adirhs_setbcs() {
    let (data, x, aa) = build_polytrope_n3_amdl(101);
    let mut omgrtp = Vec::new();
    let mut iturpr = 0_i32;

    // `res_adimod` thinning with `in_step = 2` keeps endpoints & sets row 10 = 1
    let thinned = res_adimod(
        &x,
        &aa,
        data,
        x.len(),
        IVARMD,
        2,
        0,
        &mut iturpr,
        0,
        &mut omgrtp,
    )
    .expect("res_adimod with in_step=2 succeeds with rows 6..=9 unset");
    assert_eq!(thinned.nn, 51);
    assert_eq!(thinned.x[0].to_bits(), 0);
    assert!((thinned.x[thinned.nn - 1] - 1.0_f64).abs() < 1.0e-14_f64);
    assert!((thinned.aa.get(10, 10).unwrap() - 1.0_f64).abs() < 1.0e-15_f64);

    // `#479`: `iggt == 1` (`data[7] == 20.0`) reads unset row 6 cleanly with error
    let mut data_iggt = data;
    data_iggt[7] = 20.0_f64;
    let err_res = res_adimod(
        &x,
        &aa,
        data_iggt,
        x.len(),
        IVARMD,
        1,
        0,
        &mut iturpr,
        0,
        &mut omgrtp,
    )
    .expect_err("iggt=1 must fail on unset AMDL row 6 (#479)");
    assert!(matches!(err_res, AdiplsError::UnsetAmdlRow { row: 6 }));

    // `adirhs` analytical evaluation + `#479` `iturpr == 2` row 6 error
    let input = parse_adipls_pruned_in(ADIPLS_PRUNED_IN).unwrap();
    let mut cincnt = Cincnt::from_input(&input);
    let rhs = RhsDat {
        el: 0.0_f64,
        ell: 0.0_f64,
        alb: 1.0_f64,
        els: 0.0_f64,
        el1: 0.0_f64,
        sig: 10.0_f64,
        anres: 0.0_f64,
        perfac: 1.0_f64,
        data,
    };
    let rot = RotDat::default();
    let csexpc = Csexpc::default();
    let mut fd = [[0.0_f64; 4]; 4];
    let mut finh = [0.0_f64; 8];
    let n_test = 50_usize;
    let x_pt = x[n_test - 1];
    adirhs(
        x_pt, &mut fd, &mut finh, n_test, &aa, &rhs, &rot, &cincnt, &csexpc,
    )
    .expect("radial adirhs");
    let vg = aa.get(2, n_test - 1).unwrap();
    let a = aa.get(4, n_test - 1).unwrap();
    let qx = aa.get(1, n_test - 1).unwrap();
    let u = aa.get(5, n_test - 1).unwrap();
    assert!((fd[0][0] - (vg - 2.0_f64) / x_pt).abs() < 1.0e-14_f64);
    assert!((fd[0][1] + vg * rhs.sig / (qx * x_pt * x_pt)).abs() < 1.0e-14_f64);
    assert!(
        (fd[1][0] - (1.0_f64 - (qx * a - qx * u) / rhs.sig)).abs()
            < 1.0e-14_f64
    );

    cincnt.iturpr = 2;
    let rhs_err = adirhs(
        x_pt, &mut fd, &mut finh, n_test, &aa, &rhs, &rot, &cincnt, &csexpc,
    )
    .expect_err("iturpr=2 must fail on unset AMDL row 6 (#479)");
    assert!(matches!(rhs_err, AdiplsError::UnsetAmdlRow { row: 6 }));

    // `setbcs`, `danorm`, `xlmult`, `sclasl`
    cincnt.iturpr = 0;
    let mut y = vec![0.0_f64; 8 * (x.len() + 2)];
    let mut bcs = BcsDat {
        fctsbc: input.int_ctrl.fctsbc,
        fcttbc: input.int_ctrl.fcttbc,
        istsbc: input.int_ctrl.istsbc,
        ibotbc: input.int_ctrl.ibotbc,
    };
    let mut istsb1 = bcs.istsbc;
    let mut nd = 0_usize;
    let mut ny = 0_usize;
    let mut csexpc_mut = Csexpc::default();
    let mut cnrkbc = Cnrkbc::default();
    let mut kdgbcs = 0_i32;
    setbcs(
        &x,
        &mut y,
        8,
        x.len(),
        2,
        1,
        &mut istsb1,
        2,
        1,
        &mut nd,
        &mut ny,
        0,
        &aa,
        &rhs,
        &mut bcs,
        &cincnt,
        &mut csexpc_mut,
        &mut cnrkbc,
        &mut kdgbcs,
    )
    .expect("setbcs radial");
    assert_eq!(nd, x.len() - 1);
    assert_eq!(ny, x.len());
    assert_eq!(y[0].to_bits(), 0);
    assert!((y[1] - 1.0_f64).abs() < 1.0e-14_f64);
    assert!((y[9] - 1.0_f64).abs() < 1.0e-2_f64);

    assert!(
        (danorm(&[3.0, 4.0, 1.0e-25], 3, 1.0e-19_f64) - 5.0_f64).abs()
            < 1.0e-14_f64
    );

    let mut xcache = XlmultCache::default();
    assert_eq!(xlmult(0.0, 2.0, 3.0, 0, &mut xcache), (0.0, 0));
    assert_eq!(xlmult(2.0, 0.0, -1.0, 0, &mut xcache), (0.0, -2));
    assert_eq!(xlmult(2.0, -3.0, 0.5, 0, &mut xcache).1, -1);
    let (v_big, ierr_big) = xlmult(1.0e40, 2.0, 200.0, 1, &mut xcache);
    assert_eq!((v_big.to_bits(), ierr_big), (0, 1));
    let (v_ok, ierr_ok) = xlmult(2.0, 3.0, 4.0, 0, &mut xcache);
    assert_eq!(ierr_ok, 0);
    assert!((v_ok - 162.0_f64).abs() < 1.0e-12_f64);

    let mut y_sc = vec![2.0_f64; 8 * 5];
    let x_sc = [0.0_f64, 0.25, 0.5, 0.75, 1.0];
    sclasl(&x_sc, &mut y_sc, 8, 5, 2, 4, 2, 3.0_f64, 0, &mut xcache);
    let ratio = 0.25_f64 / 0.75_f64;
    assert!((y_sc[8] - 2.0_f64 * (ratio * ratio)).abs() < 1.0e-14_f64);
    assert!((y_sc[24] - 2.0_f64).abs() < 1.0e-14_f64);
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_solve_helpers_matching_rsnfit_f32_stevft_order_takata_signew_setlog() {
    // 1. `mchcf1`, `mchcff`, `mchsol` on an exact nullspace 4x4 system
    // Column vectors yL1, yL2, yR1, yR2 such that 2*yL1 + 1*yL2 = 1*yR1 + 3*yR2
    #[rustfmt::skip]
    let ds1 = [
        1.0_f64, 0.0, 1.0, 2.0, // yL1
        0.0, 1.0, 1.0, -1.0,    // yL2
        2.0, 1.0, 0.0, 0.0,     // yR1
        0.0, 0.0, 1.0, 1.0,     // yR2  (2*yL1 + yL2 = 2*(1,0,1,2)+(0,1,1,-1) = (2,1,3,3) = 1*(2,1,0,0) + 3*(0,0,1,1))
    ];
    let mut c = [0.0_f64; 4];
    let mut is = 1_i32;
    let mut js = 1_i32;
    let mut err = 1.0_f64;
    mchcff(&ds1, &mut c, &mut is, &mut js, 1, &mut err);
    assert!((0.0_f64..1.0e-14_f64).contains(&err));
    let mut err1 = mchcf1(
        &ds1,
        &mut c,
        usize::try_from(is).unwrap(),
        usize::try_from(js).unwrap(),
    );
    assert!(err1 < 1.0e-14_f64);

    let mut y = vec![0.0_f64; 8 * 4];
    for i in 0..4 {
        y[8 + i] = ds1[i];
        y[8 + 4 + i] = ds1[4 + i];
        y[16 + i] = ds1[8 + i];
        y[16 + 4 + i] = ds1[12 + i];
    }
    let mut icry = 0_i32;
    let mut rec = TapRecorder::new();
    mchsol(
        &mut y, 8, 1, 3, 4, 2, 2, &mut is, &mut js, 1, 0, &mut err1, &ds1,
        &mut icry, &mut rec,
    )
    .expect("mchsol");
    assert_eq!(icry, 0);
    assert_eq!(rec.events.len(), 1);
    assert_eq!(rec.events[0].tag(), "MATCH");

    // 2. `rsnfit` `f32` `0.05` behavior and parity preservation
    let x_grid: Vec<f64> =
        (0..=100).map(|i| f64::from(i) / 100.0_f64).collect();
    let mut nfit = 50_usize;
    let mut xfit1 = x_grid[nfit - 1];
    let mut nftcnt = 0_i32;
    rsnfit(&x_grid, 2, 101, &mut nfit, &mut xfit1, &mut nftcnt, 0);
    assert_eq!(nftcnt, 1);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let expected_cand = ((50.0_f32 - (0.05_f32 * 1.0_f32) * 101.0_f32).round()
        as i32)
        .max(10) as usize;
    let expected_nfit = if (expected_cand.cast_signed() - 2).rem_euclid(2) != 0
    {
        expected_cand + 1
    } else {
        expected_cand
    };
    assert_eq!(nfit, expected_nfit);
    assert_eq!(xfit1.to_bits(), x_grid[nfit - 1].to_bits());

    // 3. `stevft`, `order`, `takata`, `signew`, `setlog`
    let (data, x, aa) = build_polytrope_n3_amdl(101);
    let input = parse_adipls_pruned_in(ADIPLS_PRUNED_IN).unwrap();
    let cincnt = Cincnt::from_input(&input);
    let rhs = RhsDat {
        el: 2.0_f64,
        ell: 6.0_f64,
        alb: 1.0_f64,
        els: 6.0_f64 / 15.0_f64,
        el1: 1.0_f64,
        sig: 15.0_f64,
        anres: 0.0_f64,
        perfac: 1.0_f64,
        data,
    };
    let mut nev1 = 2_usize;
    let mut nfit_st = 99_usize;
    let mut xfit_st = x[nfit_st - 1];
    stevft(
        &x,
        x.len(),
        2,
        &mut nfit_st,
        1,
        false,
        true,
        &mut nev1,
        &mut xfit_st,
        &aa,
        &rhs,
        &cincnt,
    )
    .expect("stevft");
    assert!(nev1 >= 2 && nev1 <= x.len() - 30);

    // Radial order counting on a 2-zero synthetic radial eigenfunction
    let mut y_ord = vec![0.0_f64; 8 * 101];
    for n in 0..101 {
        let phase = std::f64::consts::PI * 2.5_f64 * x[n];
        y_ord[n * 8] = x[n] * sin(phase);
        y_ord[n * 8 + 1] = -x[n] * cos(phase);
    }
    let mut iord = 0_i32;
    order(
        &x,
        &y_ord,
        &data,
        &aa,
        0.0_f64,
        20.0_f64,
        0,
        0,
        8,
        &mut iord,
        101,
        1,
        5,
        ADIPLS_DEFAULT_EPSUFL.sqrt(),
        STANDARD_CGRAV,
    )
    .expect("radial order");
    // Two interior zeros at `x = 0.4, 0.8` plus initial `phx2 * phy2 < 0`
    // gives ADIPLS radial order `1 + 2 = 3` (`order.d.f:170`).
    assert_eq!(iord, 3);

    // Dipole `takata` classification executes cleanly with unset rows 6..=9
    let iord_tak =
        takata(&x, &y_ord, 101, 1, &data, &aa, 8, 15.0_f64, STANDARD_CGRAV)
            .expect("takata");
    assert!(iord_tak >= 0);

    // `signew` exact arithmetic and window bounds + `setlog`
    let mut cry = 0_i32;
    assert!(
        (signew(1, 1, 4.0, 0.5, 0.0, 10.0, &mut cry).unwrap() - 4.5_f64).abs()
            < 1.0e-14_f64
    );
    assert_eq!(cry, 1);
    assert!(
        (signew(1, 2, 4.0, 0.5, 0.0, 10.0, &mut cry).unwrap() - 6.25_f64).abs()
            < 1.0e-14_f64
    );
    let _ = signew(1, 1, 9.8, 0.5, 0.0, 10.0, &mut cry).unwrap();
    assert_eq!(cry, -1);

    let mut slog: Vec<AdiStatusLogEntry> = Vec::new();
    setlog(
        3,
        1.0,
        2,
        4.0,
        1.0e-8,
        1.0e-9,
        0.9,
        0.9,
        20.0,
        ADIPLS_DEFAULT_EPSUFL,
        &mut slog,
    );
    assert_eq!(slog.len(), 1);
    let entry = slog[0];
    assert!((entry.per_min - 10.0_f64).abs() < 1.0e-14_f64);
    assert!((entry.nu_uhz - 1666.6667_f64).abs() < 1.0e-12_f64);
    assert_eq!(setups_adi().nnmax, 20_000);
}

#[test]
#[allow(clippy::too_many_lines, clippy::similar_names)]
fn test_147_unsupported_controls_fail_loudly() {
    let (data, x, aa) = build_polytrope_n3_amdl(101);
    let input = parse_adipls_pruned_in(ADIPLS_PRUNED_IN).unwrap();
    let cprcns = Cprcns::default();
    let mut cincnt = Cincnt::from_input(&input);
    let mut rhs = RhsDat {
        el: 1.0,
        ell: 2.0,
        alb: 1.0,
        els: 2.0 / 10.0,
        sig: 10.0,
        data,
        ..RhsDat::default()
    };
    let mut rot = RotDat::default();
    let mut bcs = BcsDat::default();
    let mut csexpc = Csexpc::default();
    let mut cnrkbc = Cnrkbc::default();
    let mut kdgbcs = 0_i32;
    let mut xcache = XlmultCache::default();
    let mut pg_state = SignwPgState::default();
    let mut csummm = Csummm::zeroed();
    let mut slog = Vec::new();
    let mut y = vec![0.0_f64; 8 * 103];
    let mut ds1 = [0.0_f64; 16];
    let mut det = 0.0_f64;
    let mut ddsol = 0.0_f64;
    let mut iord = 0_i32;
    let mut icry = 0_i32;
    let mut istsb1 = 1_i32;
    let mut initnr = 1_i32;
    let mut obs = NoopTapObserver;

    // 1. `geninh` mdintg != 5
    let err_gen = geninh(
        &x, &mut y, 2, 8, 1, 1, 2, 10, 1, 1, 1, &aa, &rhs, &rot, &cincnt,
        &csexpc,
    )
    .expect_err("geninh mdintg=1 must fail loudly (#147)");
    assert!(matches!(
        err_gen,
        AdiplsError::UnsupportedControl {
            control: "mdintg",
            ..
        }
    ));

    // 2. `signew` |itrsig| >= 2
    let mut cry_s = 0_i32;
    let err_sgn = signew(2, 1, 5.0, 0.1, 0.0, 10.0, &mut cry_s)
        .expect_err("signew itrsig=2 must fail loudly (#147)");
    assert!(matches!(
        err_sgn,
        AdiplsError::UnsupportedControl {
            control: "itrsig",
            ..
        }
    ));

    // 3. `mchsol` moddet == 1
    let mut imissl = 1;
    let mut imjssl = 1;
    let err_mch = mchsol(
        &mut y,
        8,
        1,
        101,
        4,
        2,
        50,
        &mut imissl,
        &mut imjssl,
        1,
        1,
        &mut det,
        &[
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0,
        ],
        &mut icry,
        &mut obs,
    )
    .expect_err("mchsol moddet=1 must fail loudly (#147)");
    assert!(matches!(
        err_mch,
        AdiplsError::UnsupportedControl {
            control: "moddet",
            ..
        }
    ));

    // 4. `sigint` mdintg == 3, `sigsol` iriche == 1 & nonradial icow == 1, `sigscn` bad nsig
    {
        let mut ctx = SolverContext {
            rhs: &mut rhs,
            rot: &mut rot,
            bcs: &mut bcs,
            cincnt: &mut cincnt,
            csexpc: &mut csexpc,
            cnrkbc: &mut cnrkbc,
            cprcns: &cprcns,
            kdgbcs: &mut kdgbcs,
            cgrav: STANDARD_CGRAV,
            xlmult_cache: &mut xcache,
            signw_pg_state: &mut pg_state,
            csummm: &mut csummm,
            iordtr: 0,
            status_log: &mut slog,
        };
        // `shtint` stride alignment check returns `icry = -1` cleanly
        shtint(
            &x, &mut y, 8, 1, 2, 101, 3, 100, 101, 2, 2, 1, 51, 5, &mut det,
            &mut ddsol, &mut ds1, &mut iord, &mut icry, &aa, &mut ctx,
        )
        .expect("shtint stride mismatch sets icry=-1");
        assert_eq!(icry, -1);
        let err_int = sigint(
            &x,
            &mut y,
            8,
            1,
            2,
            101,
            3,
            2,
            1,
            2,
            1,
            &mut initnr,
            50,
            &mut istsb1,
            &mut det,
            &mut ddsol,
            &mut ds1,
            &mut iord,
            &mut icry,
            &aa,
            &mut ctx,
            &mut obs,
        )
        .expect_err("sigint mdintg=3 must fail loudly (#147)");
        assert!(matches!(
            err_int,
            AdiplsError::UnsupportedControl {
                control: "mdintg",
                ..
            }
        ));

        ctx.cincnt.iriche = 1;
        let mut sig = 10.0_f64;
        let mut nev1 = 2;
        let mut nfit = 50;
        let mut isolcv = 0;
        let mut isigcv = 0;
        let mut sigtst = 0.0;
        let err_ri = sigsol(
            &mut sig,
            &x,
            &mut y,
            8,
            1,
            2,
            101,
            101,
            5,
            &mut nev1,
            &mut nfit,
            &mut istsb1,
            10,
            &mut iord,
            &mut icry,
            &mut isolcv,
            &mut isigcv,
            &mut sigtst,
            &aa,
            &mut ctx,
            &mut obs,
        )
        .expect_err("sigsol iriche=1 must fail loudly (#147)");
        assert!(matches!(
            err_ri,
            AdiplsError::UnsupportedControl {
                control: "iriche",
                ..
            }
        ));

        ctx.cincnt.iriche = 0;
        ctx.cincnt.icow = 1;
        let err_cow = sigsol(
            &mut sig,
            &x,
            &mut y,
            8,
            1,
            2,
            101,
            101,
            5,
            &mut nev1,
            &mut nfit,
            &mut istsb1,
            10,
            &mut iord,
            &mut icry,
            &mut isolcv,
            &mut isigcv,
            &mut sigtst,
            &aa,
            &mut ctx,
            &mut obs,
        )
        .expect_err("sigsol nonradial icow=1 must fail loudly (#147)");
        assert!(matches!(
            err_cow,
            AdiplsError::UnsupportedControl {
                control: "icow",
                ..
            }
        ));

        ctx.cincnt.icow = 0;
        let mut iscan = 20_i32;
        let mut bad_nsig = 9_i32;
        let mut noop_out = flexmesa_adipls::NoopSigoutHandler;
        let err_nsig = sigscn(
            5.0,
            20.0,
            &mut iscan,
            &mut bad_nsig,
            1,
            &x,
            &mut y,
            8,
            1,
            2,
            101,
            101,
            5,
            &mut nev1,
            &mut nfit,
            &mut istsb1,
            &mut icry,
            0,
            0.0,
            -1.0,
            &aa,
            &mut ctx,
            &mut obs,
            &mut noop_out,
        )
        .expect_err("sigscn bad nsig must fail loudly (#147)");
        assert!(matches!(
            err_nsig,
            AdiplsError::UnsupportedControl {
                control: "nsig",
                ..
            }
        ));
    }

    // 5. `AdiplsSolver::run_call` with `xtrnct > 0`
    let mut bad_input = input.clone();
    bad_input.mod_ctrl.xtrnct = 0.1;
    let controls = AdiplsControlState::from_input(&bad_input);
    let err_trn = AdiplsSolver::new(bad_input)
        .run_call(&controls, &x, &aa, data)
        .expect_err("xtrnct > 0 must fail loudly (#147)");
    assert!(matches!(
        err_trn,
        AdiplsError::UnsupportedControl {
            control: "xtrnct/ntrnsf",
            ..
        }
    ));

    // 6. `AdiplsSolver::run_call` with invalid `istsig >= 2` + `itrsig == 0`
    let controls_ist = AdiplsControlState::from_input(&input);
    let mut bad_c = controls_ist;
    bad_c.cadi_param.istsig = 2;
    bad_c.cadi_param.itrsig = 0;
    bad_c.cadi_param.iscan = 0;
    let err_ist = AdiplsSolver::new(input)
        .run_call(&bad_c, &x, &aa, data)
        .expect_err("istsig=2 itrsig=0 iscan=0 must fail loudly (#147)");
    assert!(matches!(
        err_ist,
        AdiplsError::UnsupportedControl {
            control: "istsig",
            ..
        }
    ));
}

#[derive(Default)]
struct ModeCollector {
    modes: Vec<(i32, f64, i32, i32)>, // (iord, sig, isolcv, isigcv)
}

impl SigoutHandler for ModeCollector {
    fn on_sigout(
        &mut self,
        args: &mut SigoutArgs<'_>,
    ) -> Result<(), AdiplsError> {
        self.modes
            .push((args.iord, args.sig, args.isolcv, args.isigcv));
        Ok(())
    }
}

#[test]
fn test_synthetic_polytrope_radial_and_nonradial_end_to_end_solve() {
    let (data, x, aa) = build_polytrope_n3_amdl(401);
    let input = parse_adipls_pruned_in(ADIPLS_PRUNED_IN).unwrap();
    let solver = AdiplsSolver::new(input.clone());

    for &el in &[0_i32, 1_i32, 2_i32] {
        let mut controls = AdiplsControlState::from_input(&input);
        set_adipls_controls(
            &mut controls,
            el,
            0,
            f64::from(el),
            1.0,
            1,
            80,
            4.0,
            60.0,
            0.0,
            2,
            0,
            0,
            0,
            0,
        );

        let mut recorder = TapRecorder::new();
        let mut collector = ModeCollector::default();
        let res = solver
            .run_call_with_hooks(
                &controls,
                &x,
                &aa,
                data,
                &mut recorder,
                &mut collector,
            )
            .unwrap_or_else(|err| {
                panic!("polytrope solve failed for el={el}: {err}")
            });

        assert_eq!(res.ierr_param, 0);
        assert_eq!(res.nn, 401);
        assert!(
            !collector.modes.is_empty(),
            "expected converged modes for el={el}"
        );
        assert_eq!(recorder.events.first().unwrap().tag(), "ENTER");

        // Verify all converged `Itc` events and post-`order` `SigoutArgs` modes.
        let mut itc_count = 0_usize;
        for ev in &recorder.events {
            if let AdiplsTapEvent::Itc {
                ddsig,
                ddsol,
                isolcv,
                isigcv,
                ..
            } = *ev
            {
                itc_count += 1;
                assert_eq!(
                    isolcv, 1,
                    "expected converged solution for el={el}"
                );
                assert_eq!(
                    isigcv, 1,
                    "expected converged frequency for el={el}"
                );
                assert!(
                    ddsig < ADIPLS_DEFAULT_EPSPRC * 100.0_f64
                        || ddsig < 1.0e-10_f64,
                    "ddsig={ddsig} too large for el={el}"
                );
                assert!(
                    ddsol < 1.0e-8_f64,
                    "ddsol={ddsol} too large for el={el}"
                );
            }
        }
        assert_eq!(itc_count, collector.modes.len());

        let mut prev_sig = 0.0_f64;
        let mut prev_ord = i32::MIN;
        for &(iord, sig, isolcv, isigcv) in &collector.modes {
            assert_eq!(isolcv, 1);
            assert_eq!(isigcv, 1);
            assert!(
                sig > prev_sig,
                "frequencies not strictly increasing for el={el}: prev=({prev_ord},{prev_sig}), cur=({iord},{sig})"
            );
            if el == 0 {
                assert!(
                    iord > prev_ord,
                    "radial orders not strictly increasing for el=0: prev=({prev_ord},{prev_sig}), cur=({iord},{sig}), all={:?}",
                    collector.modes
                );
            } else {
                assert!(
                    iord >= prev_ord,
                    "orders decreased for el={el}: prev=({prev_ord},{prev_sig}), cur=({iord},{sig}), all={:?}",
                    collector.modes
                );
            }
            prev_sig = sig;
            prev_ord = iord;
        }
    }
}

#[test]
fn test_adipls_bounds_guards_return_errors_on_edge_nfit_and_small_nn() {
    let (data, x, aa) = build_polytrope_n3_amdl(401);

    // 1. Singular-surface model with xfit > x[nn - 1] returns InvalidModel rather than underflowing `shtint`.
    let mut input_hi = parse_adipls_pruned_in(ADIPLS_PRUNED_IN).unwrap();
    input_hi.int_ctrl.xfit = 1.05_f64;
    let solver_hi = AdiplsSolver::new(input_hi);
    let ctrl_hi = AdiplsControlState::from_input(solver_hi.input());
    let err_hi = solver_hi
        .run_call(&ctrl_hi, &x, &aa, data)
        .expect_err("xfit > 1 on singular-surface model must fail loudly");
    assert!(matches!(err_hi, AdiplsError::InvalidModel { .. }));

    // 2. Full model (`x[0] == 0.0`, `nibc = 2`) with `0 < xfit <= x[1]` (`nfit = 1 < nibc`).
    let mut input_lo = parse_adipls_pruned_in(ADIPLS_PRUNED_IN).unwrap();
    input_lo.int_ctrl.xfit = 1.0e-4_f64;
    let solver_lo = AdiplsSolver::new(input_lo);
    let ctrl_lo = AdiplsControlState::from_input(solver_lo.input());
    let err_lo = solver_lo
        .run_call(&ctrl_lo, &x, &aa, data)
        .expect_err("nfit < nibc must fail loudly before shtint");
    assert!(matches!(err_lo, AdiplsError::InvalidModel { .. }));

    // 3. `mdintg = 1, iriche = 1` with `nfit == nn` and odd `nfit - nibc` must fail loudly instead of indexing `x[nn]`;
    //    on a non-singular model with `nfit == nn` (`nd_limit == nn`), it reaches `mdintg=1` unsupported error cleanly.
    let mut input_iriche = parse_adipls_pruned_in(ADIPLS_PRUNED_IN).unwrap();
    input_iriche.int_ctrl.mdintg = 1;
    input_iriche.int_ctrl.iriche = 1;
    input_iriche.int_ctrl.xfit = 1.05_f64;
    let solver_iriche = AdiplsSolver::new(input_iriche);
    let ctrl_iriche = AdiplsControlState::from_input(solver_iriche.input());
    let err_iriche = solver_iriche
        .run_call(&ctrl_iriche, &x, &aa, data)
        .expect_err("iriche parity shift past nn must fail loudly");
    assert!(matches!(err_iriche, AdiplsError::InvalidModel { .. }));

    let mut data_nonsing = data;
    data_nonsing[6] = -1.0_f64;
    let err_mdintg = solver_iriche
        .run_call(&ctrl_iriche, &x, &aa, data_nonsing)
        .expect_err(
            "mdintg = 1 with valid nfit == nd_limit must fail loudly cleanly",
        );
    assert!(matches!(
        err_mdintg,
        AdiplsError::UnsupportedControl {
            control: "mdintg",
            ..
        }
    ));

    // 4. Small-model guards in `res_adimod`, `run_call` (`nn == 1`), and `rsnfit`.
    let x_tiny = [0.0_f64];
    let mut aa_tiny = AmdlAa::new_unset(1);
    for row in 1..=5 {
        aa_tiny.set_row(row, vec![1.0_f64]).unwrap();
    }
    let mut iturpr = 0_i32;
    let mut omgrtp = [0.0_f64; 1];
    assert!(matches!(
        res_adimod(
            &x_tiny,
            &aa_tiny,
            data,
            1,
            IVARMD,
            1,
            0,
            &mut iturpr,
            0,
            &mut omgrtp
        ),
        Err(AdiplsError::InvalidModel { .. })
    ));
    let mut data_tiny_nonsing = data;
    data_tiny_nonsing[6] = -1.0_f64;
    assert!(matches!(
        solver_hi.run_call(&ctrl_hi, &x_tiny, &aa_tiny, data_tiny_nonsing),
        Err(AdiplsError::InvalidModel { .. })
    ));

    let x_short = [0.0_f64, 0.25, 0.5, 0.75, 1.0];
    let mut nfit = 3_usize;
    let mut xfit1 = 0.5_f64;
    let mut nftcnt = 0_i32;
    rsnfit(&x_short, 2, 5, &mut nfit, &mut xfit1, &mut nftcnt, 0);
    assert!((2..=5).contains(&nfit));
    rsnfit(&x_short, 2, 5, &mut nfit, &mut xfit1, &mut nftcnt, 1);
    assert!((2..=5).contains(&nfit));
}
