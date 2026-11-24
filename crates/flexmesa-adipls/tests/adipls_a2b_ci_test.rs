// SPDX-License-Identifier: GPL-3.0-or-later
//! Self-contained always-on CI unit and integration tests for Package A2b
//! (`kiner`, `varfrq`, `sigout`, `setobs_st`, `setssm`, and full
//! `AdiplsSolver` post-processing) (`#171`, `#479`, `#147`).

use flexmesa_adipls::{
    kiner, kiner_with_observer, parse_adipls_pruned_in, set_adipls_controls,
    setobs_st, setssm, sigout, spcout_adi, varfrq, vintk, AdiplsControlState,
    AdiplsError, AdiplsSolver, AdiplsTapEvent, AmdlAa, BcsDat, Cincnt, Cprcns,
    CrotSplit, Csummm, ObsStState, RhsDat, RotDat, SetssmState, SigoutArgs,
    SigoutState, TapRecorder, Varcon, VarfrqCache, VintkCache,
    ADIPLS_DEFAULT_EPSUFL,
};
use flexmesa_const::{PI4, STANDARD_CGRAV};
use flexmesa_math::cos;

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
fn build_uniform_sphere_amdl(nn: usize) -> ([f64; 8], Vec<f64>, AmdlAa) {
    // Analytic homogeneous sphere (`rho = const`, `Gamma_1 = 5/3`, `P ~ 1 - x^2`):
    // - `aa(1, n) = q / x^3 = 1`
    // - `aa(2, n) = V_g = (6/5) * x^2 / (1 - x^2)` (clamped huge at surface)
    // - `aa(3, n) = Gamma_1 = 5/3`
    // - `aa(4, n) = A* = -(6/5) * x^2 / (1 - x^2)` (convectively adiabatic `A*`? Wait:
    //   for constant density `d ln rho / dr = 0`, `A* = -(1/Gamma_1) d ln P / d ln r = V_g`)
    // - `aa(5, n) = U = 3`
    // - `aa(10, n) = 1`
    let mut x = Vec::with_capacity(nn);
    let mut r1 = Vec::with_capacity(nn);
    let mut r2 = Vec::with_capacity(nn);
    let mut r3 = Vec::with_capacity(nn);
    let mut r4 = Vec::with_capacity(nn);
    let mut r5 = Vec::with_capacity(nn);
    let mut r10 = Vec::with_capacity(nn);
    let gamma1 = 5.0_f64 / 3.0_f64;
    let inv_gm1 = 1.0_f64 / gamma1;

    for i in 0..nn {
        let xn = (i as f64) / ((nn - 1) as f64);
        x.push(xn);
        r1.push(1.0_f64);
        r3.push(gamma1);
        r5.push(3.0_f64);
        r10.push(1.0_f64);
        if i == 0 {
            r2.push(0.0_f64);
            r4.push(0.0_f64);
        } else if i == nn - 1 {
            r2.push(1.0e12_f64);
            r4.push(0.0_f64);
        } else {
            let x2 = xn * xn;
            let vg = 2.0_f64 * inv_gm1 * x2 / (1.0_f64 - x2);
            r2.push(vg);
            r4.push(-vg);
        }
    }
    let mut aa = AmdlAa::new_unset(nn);
    aa.set_row(1, r1).unwrap();
    aa.set_row(2, r2).unwrap();
    aa.set_row(3, r3).unwrap();
    aa.set_row(4, r4).unwrap();
    aa.set_row(5, r5).unwrap();
    aa.set_row(10, r10).unwrap();

    let data = [
        1.989e33_f64,
        6.957e10_f64,
        1.0e15_f64,
        1.41_f64,
        2.0_f64 * inv_gm1,
        0.0_f64,
        0.0_f64, // polytropic surface index 0 (singular)
        0.0_f64,
    ];
    (data, x, aa)
}

#[allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::similar_names
)]
fn build_polytrope_n3_amdl(nn: usize) -> ([f64; 8], Vec<f64>, AmdlAa) {
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

    let m = xis.len();
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
    let mut l_th = flexmesa_adipls::LirState::default();
    let mut l_dth = flexmesa_adipls::LirState::default();
    let mut th_v = [0.0_f64; 1];
    let mut dth_v = [0.0_f64; 1];

    for idx in 0..nn {
        let xn = (idx as f64) / ((nn - 1) as f64);
        x.push(xn);
        r3.push(gamma1);
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
            let _ = flexmesa_adipls::lir(
                xi_pt,
                &xis,
                &mut th_v,
                &thetas,
                1,
                1,
                xis.len(),
                1,
                &mut l_th,
            );
            let _ = flexmesa_adipls::lir(
                xi_pt,
                &xis,
                &mut dth_v,
                &dthetas,
                1,
                1,
                xis.len(),
                1,
                &mut l_dth,
            );
            let thp = th_v[0];
            let deriv_pt = dth_v[0];
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
        3.0_f64,
        0.0_f64,
    ];
    (data, x, aa)
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn test_kiner_radial_nonradial_plane_parallel_and_iekinr() {
    let nn = 101_usize;
    let (_data, x, aa) = build_uniform_sphere_amdl(nn);
    let eprufl = ADIPLS_DEFAULT_EPSUFL.sqrt();

    // Radial: `y1 = x`, `aa1 = 1, aa5 = 3, aa10 = 1` ->
    // `int_0^1 x^2 * 3 * x^2 dx = 3/5`, `ynorm = y1(1)^2 = 1`, `ekin = 3 / (20 * pi)`.
    let iy = 8_usize;
    let mut y = vec![0.0_f64; iy * nn];
    for (n, &xn) in x.iter().enumerate() {
        y[n * iy] = xn;
        y[n * iy + 1] = 2.0_f64 * xn;
    }

    let mut rec = TapRecorder::new();
    let rad_out = kiner_with_observer(
        &x, &y, &aa, 0.0_f64, iy, nn, 0, 0, eprufl, &mut rec,
    )
    .expect("radial kiner");
    assert!((rad_out.ynorm - 1.0_f64).abs() < 1.0e-14_f64);
    assert!(
        (rad_out.ekin - 3.0_f64 / (5.0_f64 * PI4)).abs() < 5.0e-10_f64,
        "got {}",
        rad_out.ekin
    );
    assert_eq!(rec.events.len(), 1);
    assert_eq!(rec.events[0].tag(), "KINER");

    // Nonradial `el = 2`, `iekinr = 1`: `ell = 6`, `int_0^1 (x^2 + 4x^2 / 6) * 3 * x^2 dx = 1`,
    // `ynorm = 1 + 4/6 = 5/3`, `ekin = 1 / (pi4 * 5/3) = 3 / (5 * pi4)`.
    let nr_out = kiner(&x, &y, &aa, 2.0_f64, iy, nn, 0, 1, eprufl)
        .expect("nonradial kiner");
    assert!((nr_out.ynorm - (5.0_f64 / 3.0_f64)).abs() < 1.0e-13_f64);
    assert!((nr_out.ekin - 3.0_f64 / (5.0_f64 * PI4)).abs() < 5.0e-10_f64);

    // Plane-parallel `iplneq = 1`: `elli = 1/4`, `int_0^1 (x^2 + x^2) * 3 dx = 2`.
    let pp_out = kiner(&x, &y, &aa, 2.0_f64, iy, nn, 1, 0, eprufl)
        .expect("plane-parallel kiner");
    assert!((pp_out.ynorm - 1.0_f64).abs() < 1.0e-14_f64);
    assert!((pp_out.ekin - 2.0_f64 / PI4).abs() < 1.0e-12_f64);
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn test_varfrq_branches_cache_reuse_and_loud_errors() {
    let nn = 101_usize;
    let (data, x, aa) = build_uniform_sphere_amdl(nn);
    let cprcns = Cprcns::default();
    let iy = 8_usize;

    // Exact fundamental radial mode of the homogeneous sphere (`Gamma1 = 5/3`):
    // `y1 = x` (homologous expansion), `sigma^2 = 3*Gamma1 - 4 = 1`.
    // In ADIPLS radial variables (`y1 = xi_r / R`, `y2 = delta P / P`):
    // Wait: in ADIPLS `y1 = x`, `y2` satisfies `y1' + 2 y1 / x = ...`, and
    // for `y1 = x`, `y2(1) = -4/sig`? Let's test `usegam = true` (`aa(3,1) = 0.6 > 0.1`):
    // With `y1 = x`, `z1 = y1 / x = 1`, `dz1/dx = 1`, `d1 = 1 - 1 = 0`,
    // `dgm1 = 0` (`aa(3, :) = const`), so integrand is `(3*gm1 - 4) * 1 * 3 * x^4`,
    // denominator integrand is `3 * x^4`, and `ajs` with `y2(nn) = 0` gives
    // `ajs = 1 * ((3/vg - 1)*1 + 0) * 3 = -3`? Let's set `sig = 1.0`, `y2(nn) = (1 - 3/vg)/sig`
    // so `ajs = 0` and exact `sigv = 3*gm1 - 4 = 1.0`!
    let gm1 = 5.0_f64 / 3.0_f64;
    let vg_surf = aa.get(2, nn - 1).unwrap();
    let mut y = vec![0.0_f64; iy * nn];
    for (n, &xn) in x.iter().enumerate() {
        y[n * iy] = xn;
    }
    y[(nn - 1) * iy + 1] = 1.0_f64 - 3.0_f64 / vg_surf;

    let rhs = RhsDat {
        el: 0.0_f64,
        ell: 0.0_f64,
        alb: 1.0_f64,
        els: 0.0_f64,
        sig: 1.0_f64,
        perfac: 1.0_f64,
        data,
        ..RhsDat::default()
    };
    let mut varcon = Varcon::default();
    let mut cache = VarfrqCache::default();
    let sigv = varfrq(
        &x,
        &y,
        &data,
        &aa,
        1.0_f64,
        iy,
        nn,
        1,
        &rhs,
        &cprcns,
        &mut varcon,
        &mut cache,
    )
    .expect("radial varfrq");
    assert!(
        (sigv - (3.0_f64 * gm1 - 4.0_f64)).abs() < 1.0e-12_f64,
        "expected analytic homogeneous sphere radial eigenvalue 1.0, got {sigv}"
    );

    // Second call on identical mesh reuses cached `kvarfi == 0` bit-exactly.
    let sigv2 = varfrq(
        &x,
        &y,
        &data,
        &aa,
        1.0_f64,
        iy,
        nn,
        1,
        &rhs,
        &cprcns,
        &mut varcon,
        &mut cache,
    )
    .expect("cached radial varfrq");
    assert_eq!(sigv.to_bits(), sigv2.to_bits());

    // Nonradial `el > 1e-6` resets `ivarf = 3 -> 1`, and `ivarf = 99` fails loudly (#147).
    let rhs_nr = RhsDat {
        el: 2.0_f64,
        ell: 6.0_f64,
        els: 0.6_f64,
        sig: 10.0_f64,
        ..rhs
    };
    let mut bad_varcon = Varcon {
        ivarf: 99,
        ..Varcon::default()
    };
    let err = varfrq(
        &x,
        &y,
        &data,
        &aa,
        10.0_f64,
        iy,
        nn,
        1,
        &rhs_nr,
        &cprcns,
        &mut bad_varcon,
        &mut cache,
    )
    .expect_err("invalid ivarf must fail loudly (#147)");
    assert!(matches!(
        err,
        AdiplsError::UnsupportedControl {
            control: "ivarf",
            ..
        }
    ));
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_setobs_st_setssm_and_sigout_windows_and_loud_errors() {
    let mut csum = Csummm::zeroed();
    csum.set_f64(1, 3.0_f64).unwrap();
    csum.set_f64(2, 1.989e33_f64).unwrap();
    csum.set_f64(3, 6.957e10_f64).unwrap();
    csum.set_f64(4, 1.0e17_f64).unwrap();
    csum.set_f64(5, 100.0_f64).unwrap();
    csum.set_f64(18, 1.0_f64).unwrap();
    csum.set_f64(19, 5.0_f64).unwrap();
    csum.set_f64(20, 40.0_f64).unwrap();
    csum.set_f64(21, 41.0_f64).unwrap();
    csum.set_f64(24, 1.5e-6_f64).unwrap();
    csum.set_f64(25, 20.0_f64).unwrap();
    csum.set_f64(27, 0.85_f64).unwrap();
    csum.set_f64(37, 0.86_f64).unwrap();
    csum.set_f64(38, -1.0_f64).unwrap();
    csum.set_icsumm([1, 200, 5, 1, 10010, 3, 1, 0]);

    // `setobs_st` rotational splitting (`irotkr == 21`) + `ispcpr == 6`
    let mut obs_state = ObsStState::default();
    let rot_split = CrotSplit {
        beta: 0.5_f64,
        split: [0.4_f64, 0.01_f64, 0.02_f64, 0.0_f64],
    };
    let obs =
        setobs_st(&csum, 0, 1, 6, 21, 10, 10.0_f64, &rot_split, &mut obs_state)
            .expect("setobs_st");
    assert_eq!(obs_state.icobs_st, 116); // 6 + 10 * 11
    assert_eq!(obs.degree_raw(), 1.0_f64);
    assert_eq!(obs.order_raw(), 5.0_f64);
    assert_eq!(
        obs.cyclic_freq_uhz().to_bits(),
        (41.0_f64.sqrt() / (60.0e-6_f64 * 10.0_f64)).to_bits()
    );
    assert_eq!(obs.rot_beta(), 0.5_f64);
    assert_eq!(obs.rot_split1(), 0.4_f64);
    assert_eq!(obs.rot_split2(), 0.01_f64);
    assert_eq!(obs.rot_split3(), 0.02_f64);

    let mode = spcout_adi(&obs, &csum).expect("spcout_adi");
    assert_eq!(mode.el, 1);
    assert_eq!(mode.order, 5);
    assert_eq!(mode.em, -1);
    assert_eq!(mode.inertia.to_bits(), (1.5e-6_f64 * PI4).to_bits());

    // `nobs_stmx` limit fails loudly
    let mut tiny_state = ObsStState::default();
    let err_max =
        setobs_st(&csum, 0, 1, 6, 0, 0, 10.0_f64, &rot_split, &mut tiny_state)
            .expect_err("nobs_st >= nobs_stmx must error");
    assert!(matches!(
        err_max,
        AdiplsError::Solver {
            routine: "setobs_st",
            ..
        }
    ));

    // `setssm` emits model header on first mode (`irmod == 1`) and suppresses on second (`irmod == 0`)
    let mut ssm_state = SetssmState::default();
    let s1 = setssm(&csum, &mut ssm_state).expect("setssm 1");
    assert_eq!(s1.irmod, 1);
    let mrec = s1.model_record.expect("model header");
    assert_eq!(mrec.header_minus_one, -1.0_f64);
    assert_eq!(mrec.xmod, 3.0_f64);
    assert_eq!(s1.ssummm.ss[2], 41.0_f64);
    assert_eq!(s1.ssummm.ss[4], 0.85_f64);
    assert_eq!(s1.ssummm.iss, [10010, 3]);

    let s2 = setssm(&csum, &mut ssm_state).expect("setssm 2");
    assert_eq!(s2.irmod, 0);
    assert!(s2.model_record.is_none());

    // Fallback when `cs(21) <= 0` and `cs(27) <= 0`
    csum.set_f64(21, 0.0_f64).unwrap();
    csum.set_f64(27, 0.0_f64).unwrap();
    let s3 = setssm(&csum, &mut ssm_state).expect("setssm 3");
    assert_eq!(s3.ssummm.ss[2], 40.0_f64);
    assert_eq!(
        s3.ssummm.ss[4].to_bits(),
        (16.666_666_667_f64 / 20.0_f64).to_bits()
    );

    // `sigout` window filtering (`iorwn1..iorwn2`) & loud errors (`itssol`, `irotkr`, `igm1kr`)
    let (data, x, aa) = build_uniform_sphere_amdl(31);
    let input = parse_adipls_pruned_in(ADIPLS_PRUNED_IN).unwrap();
    let cprcns = Cprcns::default();
    let cincnt = Cincnt::from_input(&input);
    let rhs = RhsDat {
        el: 0.0_f64,
        ell: 0.0_f64,
        alb: 1.0_f64,
        els: 0.0_f64,
        sig: 10.0_f64,
        perfac: 10.0_f64,
        data,
        ..RhsDat::default()
    };
    let rot = RotDat::default();
    let bcs = BcsDat::default();
    let mut y = vec![0.0_f64; 8 * 31];
    for (n, &xn) in x.iter().enumerate() {
        y[n * 8] = xn;
        y[n * 8 + 1] = -cos(xn);
    }

    let mut state = SigoutState::new(&input, 0, 0, 0, 0, 0, 20_000);
    // Reject order `iord = 200 > iorwn2 = 100`
    let mut obs_tap = TapRecorder::new();
    let mut cs_win = Csummm::zeroed();
    {
        let mut args = SigoutArgs {
            sig: 10.0_f64,
            x: &x,
            y: &mut y,
            iy: 8,
            ii: 2,
            nw1: 1,
            nibc: 2,
            nn: 31,
            nnw: 31,
            mdintg: 5,
            nev1: 2,
            nfit: 15,
            istsb1: 1,
            iord: 200,
            isolcv: 1,
            isigcv: 1,
            aa: &aa,
            rhs: &rhs,
            rot: &rot,
            bcs: &bcs,
            cincnt: &cincnt,
            cprcns: &cprcns,
            csummm: &mut cs_win,
            observer: &mut obs_tap,
        };
        sigout(&mut args, &mut state).expect("windowed sigout");
    }
    assert!(state.mode_store.is_empty());
    assert_eq!(state.obs_state.records, [] as [flexmesa_adipls::ObsSt; 0]);
    assert_eq!(state.ssm_records, [] as [flexmesa_adipls::SetssmOutput; 0]);
    assert_eq!(obs_tap.events.len(), 2); // KINER + VARFRQ ran before window filter

    // `irotkr > 0` fails loudly when mode passes windowing
    state.out.irotkr = 11;
    let mut obs_rot = TapRecorder::new();
    let mut args_rot = SigoutArgs {
        sig: 10.0_f64,
        x: &x,
        y: &mut y,
        iy: 8,
        ii: 2,
        nw1: 1,
        nibc: 2,
        nn: 31,
        nnw: 31,
        mdintg: 5,
        nev1: 2,
        nfit: 15,
        istsb1: 1,
        iord: 1,
        isolcv: 1,
        isigcv: 1,
        aa: &aa,
        rhs: &rhs,
        rot: &rot,
        bcs: &bcs,
        cincnt: &cincnt,
        cprcns: &cprcns,
        csummm: &mut cs_win,
        observer: &mut obs_rot,
    };
    let err_rot = sigout(&mut args_rot, &mut state)
        .expect_err("irotkr > 0 must fail loudly (#147)");
    assert!(matches!(
        err_rot,
        AdiplsError::UnsupportedControl {
            control: "irotkr",
            ..
        }
    ));
}

#[test]
fn test_adipls_solver_run_call_returns_complete_modes_obs_st_ssm_and_taps() {
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
        let res = solver
            .run_call_with_observer(&controls, &x, &aa, data, &mut recorder)
            .unwrap_or_else(|err| {
                panic!("polytrope A2b solve failed for el={el}: {err}")
            });

        assert_eq!(res.ierr_param, 0);
        assert_eq!(res.icobs_st, 6);
        assert!(!res.modes.is_empty());
        assert_eq!(res.obs_st.len(), res.modes.len());
        assert_eq!(res.ssm_records.len(), res.modes.len());
        assert_eq!(res.ssm_records[0].irmod, 1);
        for ssm in &res.ssm_records[1..] {
            assert_eq!(ssm.irmod, 0);
        }

        // Check `KINER` and `VARFRQ` tap events match `ModeRecord` values.
        let kiner_events: Vec<(f64, f64, f64)> = recorder
            .events
            .iter()
            .filter_map(|e| match *e {
                AdiplsTapEvent::Kiner { el, ynorm, ekin } => {
                    Some((el, ynorm, ekin))
                }
                _ => None,
            })
            .collect();
        let varfrq_events: Vec<(f64, f64)> = recorder
            .events
            .iter()
            .filter_map(|e| match *e {
                AdiplsTapEvent::Varfrq { sig, sigv } => Some((sig, sigv)),
                _ => None,
            })
            .collect();
        assert_eq!(kiner_events.len(), res.modes.len());
        assert_eq!(varfrq_events.len(), res.modes.len());

        for (idx, rec) in res.modes.records.iter().enumerate() {
            assert_eq!(rec.el, el);
            assert_eq!(rec.em, 0);
            assert_eq!(rec.obs_st, res.obs_st[idx]);
            assert_eq!(
                rec.obs_st.ekin().to_bits(),
                kiner_events[idx].2.to_bits()
            );
            assert_eq!(
                rec.inertia.to_bits(),
                (rec.obs_st.ekin() * PI4).to_bits()
            );

            // Variational eigenvalue agrees well with shooting eigenvalue on n=3 polytrope.
            let (sig, sigv) = varfrq_events[idx];
            assert!(
                (sigv - sig).abs() / sig < 5.0e-3_f64,
                "varfrq mismatch for el={el} mode {idx}: sig={sig}, sigv={sigv}"
            );

            // Check `csummm(21)` and `icsumm` invariants (`icase=10010`, `iorign=3`).
            let sig_corrected = rec.csummm.get_f64(21).unwrap();
            assert_eq!(sig_corrected.to_bits(), sig.to_bits());
            let ics = rec.csummm.icsumm();
            assert_eq!(ics[0], 1);
            assert_eq!(ics[1], 401);
            assert_eq!(ics[2], 5);
            assert_eq!(ics[3], 1);
            assert_eq!(ics[4], 10_010);
            assert_eq!(ics[5], 3);
            assert_eq!(ics[6], 1);
        }
    }
}

#[test]
fn test_vintk_nonuniform_stencil_alignment_on_impulse() {
    // Regression test for `adipls/adipack.c/gensr/vintk.new.d.f:86, 141`:
    // integrating `[x_2, x_3]` (0-based `n = 3`) with `k = 2` (`k2 = 4`) must
    // select stencil `[1, 2, 3, 4]` (`i = n - k = 1`), so a unit impulse at
    // `idx = 4` produces the exact negative Lagrange-cubic integral over `[x_2, x_3]`.
    let x = [0.0_f64, 0.05, 0.12, 0.25, 0.45, 0.70, 1.00];
    let a = [0.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let mut y = [0.0_f64; 7];
    let mut cache = VintkCache::default();
    let ii = vintk(&x, &a, &mut y, 1, x.len(), 1, 1, 2, &mut cache);
    assert_eq!(ii, 1);
    assert_eq!(y[1], 0.0_f64);
    assert_eq!(y[2], 0.0_f64);

    // Exact integral of L_4(x) = (x-x1)(x-x2)(x-x3) / ((x4-x1)(x4-x2)(x4-x3)) on [x2, x3].
    let (x1, x2, x3, x4) = (x[1], x[2], x[3], x[4]);
    let denom = (x4 - x1) * (x4 - x2) * (x4 - x3);
    let antideriv = |t: f64| -> f64 {
        let p0 = x1 * x2 * x3;
        let p1 = x1 * x2 + x1 * x3 + x2 * x3;
        let p2 = x1 + x2 + x3;
        (0.25_f64 * t * t * t * t - (p2 / 3.0_f64) * t * t * t
            + 0.5_f64 * p1 * t * t
            - p0 * t)
            / denom
    };
    let expected_step = antideriv(x3) - antideriv(x2);
    assert!(expected_step < -1.0e-3_f64);
    assert!(
        (y[3] - expected_step).abs() < 1.0e-15_f64,
        "vintk stencil mismatch: got {}, expected {expected_step}",
        y[3]
    );
}
