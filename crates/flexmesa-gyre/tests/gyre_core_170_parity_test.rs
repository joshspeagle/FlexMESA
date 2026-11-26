// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone zero-`$MESA_DIR` unit and algebraic parity tests for the core
//! `flexmesa-gyre` modules (`#170` Wave 12): `mode`, `search`, `prox`, `grid`,
//! `grid_refine`, `sysmtx`, `magnus`, `rad`, `ad`, `nad`, `context`, `model`,
//! and `mesa`.

#![forbid(unsafe_code)]
#![allow(
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::cast_precision_loss
)]

use flexmesa_const as fc;
use flexmesa_gyre::ad::{ad_wave_hom, AdBvp, AdEqns, AdTrans};
use flexmesa_gyre::constants::GyreConstants;
use flexmesa_gyre::context::Context;
use flexmesa_gyre::ext_complex::ExtC;
use flexmesa_gyre::ext_real::ExtR;
use flexmesa_gyre::fcomplex::Cplx;
use flexmesa_gyre::grid::{Grid, Point};
use flexmesa_gyre::grid_refine::refine_grid;
use flexmesa_gyre::magnus::{
    eigen_decompose_2_c, eigen_decompose_2_r, eigen_decompose_c,
    eigen_decompose_r, magnus_build_c, magnus_build_r, magnus_stencil,
};
use flexmesa_gyre::mesa::GyreMesa;
use flexmesa_gyre::mode::{rad_wave_hom, Mode, Wave};
use flexmesa_gyre::model::{idx, EvolModel};
use flexmesa_gyre::nad::{nad_wave_hom, NadBvp, NadDiff, NadEqns};
use flexmesa_gyre::par::{
    GridPar, ModePar, ModelPar, NumPar, OscPar, RotPar, ScanPar,
};
use flexmesa_gyre::prox::prox_search;
use flexmesa_gyre::rad::{NoTap, RadBvp, RadEqns};
use flexmesa_gyre::search::{
    bracket_search, build_scan, check_scan, find_turn, solve_root_r, Stat,
};
use flexmesa_gyre::sysmtx::BlockSysmtx;
use flexmesa_gyre::GyreError;
use flexmesa_math::{cos, exp, sin};

fn standard_constants() -> GyreConstants {
    let mut k = GyreConstants::new();
    k.set_real("G_GRAVITY", fc::STANDARD_CGRAV)
        .expect("G_GRAVITY");
    k.set_real("C_LIGHT", fc::CLIGHT).expect("C_LIGHT");
    k.set_real("A_RADIATION", fc::CRAD).expect("A_RADIATION");
    k.set_real("M_SUN", fc::MSUN).expect("M_SUN");
    k.set_real("R_SUN", fc::RSUN).expect("R_SUN");
    k.set_real("L_SUN", fc::LSUN).expect("L_SUN");
    k.set_char("GYRE_DIR", "inline").expect("GYRE_DIR");
    k
}

/// Builds a synthetic hydrostatic uniform-density compressible sphere model in
/// MESA version-101 18-column format (`P(x) = P_c (1 - 0.99 x^2)`).
fn uniform_sphere_mesa_data(
    n: usize,
    n2_scale: f64,
    ups_t_val: f64,
    omega_rot_val: f64,
) -> (Vec<f64>, Vec<Vec<f64>>, i32) {
    let k = standard_constants();
    let pi = k.math.pi;
    let m_star = fc::MSUN;
    let r_star = fc::RSUN;
    let l_star = fc::LSUN;
    let r3 = r_star * r_star * r_star;
    let rho_0 = 3.0 * m_star / (4.0 * pi * r3);
    let p_c = (2.0 / 3.0) * pi * k.g_gravity * rho_0 * rho_0 * r_star * r_star;
    let eps_0 = l_star / m_star;

    let mut rows = Vec::with_capacity(n);
    let nm1 = (n - 1) as f64;
    for j in 0..n {
        let x = (j as f64) / nm1;
        let x2 = x * x;
        let x3 = x * x2;
        let r = r_star * x;
        let m_r = m_star * x3;
        let l_r = l_star * x3;
        let factor = 1.0 - 0.99 * x2;
        let p = p_c * factor;
        let t = 1.0e6 * factor;
        let rho = rho_0;
        let nabla = 0.25;
        let n2 = n2_scale * x2;
        let gamma_1 = 5.0 / 3.0;
        let nabla_ad = 0.4;
        let ups_t = ups_t_val;
        let kap = 1.0;
        let kap_kap_t = -3.5 * kap;
        let kap_kap_rho = 1.0 * kap;
        let eps = eps_0;
        let eps_eps_t = 4.0 * eps;
        let eps_eps_rho = 1.0 * eps;
        rows.push(vec![
            r,
            m_r,
            l_r,
            p,
            t,
            rho,
            nabla,
            n2,
            gamma_1,
            nabla_ad,
            ups_t,
            kap,
            kap_kap_t,
            kap_kap_rho,
            eps,
            eps_eps_t,
            eps_eps_rho,
            omega_rot_val,
        ]);
    }
    (vec![m_star, r_star, l_star], rows, 101)
}

/// Hydrostatically consistent constant-density Pekeris sphere (`As = -V/Gamma1`)
/// so `U`, `V`, `As` satisfy Poisson + hydrostatic compatibility (`y_4 = -U y_1`).
fn pekeris_sphere_mesa_data(n: usize) -> (Vec<f64>, Vec<Vec<f64>>, i32) {
    let k = standard_constants();
    let pi = k.math.pi;
    let m_star = fc::MSUN;
    let r_star = fc::RSUN;
    let l_star = 1.0e40;
    let r3 = r_star * r_star * r_star;
    let rho_0 = 3.0 * m_star / (4.0 * pi * r3);
    let p_c =
        (2.0 / 3.0) * pi * k.g_gravity * rho_0 * rho_0 * r_star * r_star / 0.99;
    let eps_0 = l_star / m_star;

    let mut rows = Vec::with_capacity(n);
    let nm1 = (n - 1) as f64;
    for j in 0..n {
        let x = (j as f64) / nm1;
        let x2 = x * x;
        let x3 = x * x2;
        let r = r_star * x;
        let m_r = m_star * x3;
        let l_r = l_star * x3;
        let factor = 1.0 - 0.99 * x2;
        let p = p_c * factor;
        let t = 1.0e6 * factor;
        let rho = rho_0;
        let gamma_1 = 5.0 / 3.0;
        let n2 = if j == 0 {
            0.0
        } else {
            let g_loc = k.g_gravity * m_r / (r * r);
            -(g_loc * g_loc) * rho / (gamma_1 * p)
        };
        rows.push(vec![
            r,
            m_r,
            l_r,
            p,
            t,
            rho,
            0.25,
            n2,
            gamma_1,
            0.4,
            1.0,
            1.0,
            -3.5,
            1.0,
            eps_0,
            4.0 * eps_0,
            eps_0,
            0.0,
        ]);
    }
    (vec![m_star, r_star, l_star], rows, 101)
}

#[test]
fn mode_classify_l1_non_cowling_all_evanescent_and_active_windings() {
    let k = standard_constants();
    let (global, rows, version) = uniform_sphere_mesa_data(41, 0.0, 0.0, 0.0);
    let ml =
        EvolModel::from_mesa(&ModelPar::default(), &global, &rows, version, &k)
            .expect("from_mesa");
    let gr = ml.gr.clone();
    let n = gr.n();
    assert_eq!(n, 41);

    let md_p = ModePar {
        l: 1,
        ..ModePar::default()
    };
    let nm_p = NumPar::default();
    let os_p = OscPar::default();
    let rt_p = RotPar::default();
    let cx = Context::new(&ml, &k, &gr, &md_p, &os_p, &rt_p).expect("context");

    // Case 1: completely evanescent l=1 non-Cowling mode where find_turn
    // returns j_turn == n and y_1(n) == 0. Previously panicked with
    // slice start > end (&y_1[n..n - 1]); Fortran produces empty slice (0, 0).
    let (j_turn_ev, _) =
        find_turn(&cx, &gr, 1.0e-4, &nm_p, &os_p).expect("find_turn ev");
    assert_eq!(j_turn_ev, n);

    for x_ref in [1.0, 0.5] {
        let mut os_case = os_p.clone();
        os_case.x_ref = x_ref;
        let mut y_c = vec![Cplx::from_real(0.0); 6 * n];
        for j in 0..n - 1 {
            y_c[6 * j] = Cplx::from_real(1.0);
        }
        let wv = Wave::new(
            Cplx::from_real(1.0e-4),
            y_c,
            ExtC::from_real(0.0),
            &cx,
            &gr,
            &md_p,
            &nm_p,
            &os_case,
            1,
        )
        .expect("wave");
        let md = Mode::new(wv).expect("mode evanescent");
        assert_eq!(md.n_p, 0);
        assert_eq!(md.n_g, 0);
        assert_eq!(md.n_pg, 1);
    }

    // Case 2: turning point near origin (j_turn == 3, x in (0.05, 0.075]) and
    // four anticlockwise Takata (Yt_1, Yt_2) windings over x in [0.1, 1.0].
    let (j_turn_prop, x_turn_prop) =
        find_turn(&cx, &gr, 20.0, &nm_p, &os_p).expect("find_turn prop");
    assert_eq!(j_turn_prop, 3);
    assert!(x_turn_prop > gr.pt[2].x && x_turn_prop <= gr.pt[3].x);

    let mut y_c = vec![Cplx::from_real(0.0); 6 * n];
    for j in 0..n {
        let theta = 4.0 * k.math.pi * gr.pt[j].x;
        y_c[6 * j] = Cplx::from_real(1.0);
        y_c[6 * j + 1] = Cplx::from_real(1.0 + sin(theta));
        y_c[6 * j + 2] = Cplx::from_real(3.0 * cos(theta));
    }
    let wv = Wave::new(
        Cplx::from_real(20.0),
        y_c,
        ExtC::from_real(0.0),
        &cx,
        &gr,
        &md_p,
        &nm_p,
        &os_p,
        2,
    )
    .expect("wave prop");
    let md = Mode::new(wv).expect("mode prop");
    assert_eq!(md.n_p, 4);
    assert_eq!(md.n_g, 0);
    assert_eq!(md.n_pg, 5);
    assert!((md.wave.e() - 1.0).abs() < 1.0e-13);
}

#[test]
fn search_brent_root_scan_and_critical_layer_parity() {
    let nm_p = NumPar::default();

    // Closed-form sqrt(2) root via Brent.
    let (root, stat, n_iter) = solve_root_r(
        |x| Ok((x * x - 2.0, Stat::Ok)),
        1.0,
        2.0,
        0.0,
        &nm_p,
        None,
    )
    .expect("solve_root_r");
    assert_eq!(stat, Stat::Ok);
    assert!(n_iter > 0);
    assert!((root - 2.0_f64.sqrt()).abs() <= 4.0 * f64::EPSILON);

    // Zero iteration cap yields Stat::IterMax on the first step.
    let (_, stat_lim, _) = solve_root_r(
        |x| Ok((x - 1.5, Stat::Ok)),
        1.0,
        2.0,
        0.0,
        &nm_p,
        Some(0),
    )
    .expect("iter_max");
    assert_eq!(stat_lim, Stat::IterMax);

    // Callback Stat::OmegaDomain propagates without erroring.
    let mut calls = 0_i32;
    let (_, stat_dom, _) = solve_root_r(
        |x| {
            calls += 1;
            if calls <= 2 {
                Ok((x - 1.5, Stat::Ok))
            } else {
                Ok((0.0, Stat::OmegaDomain))
            }
        },
        1.0,
        2.0,
        0.0,
        &nm_p,
        None,
    )
    .expect("domain");
    assert_eq!(stat_dom, Stat::OmegaDomain);

    // Unbracketed interval asserts cleanly; invalid solver aborts.
    assert!(matches!(
        solve_root_r(
            |x| Ok((x * x + 1.0, Stat::Ok)),
            1.0,
            2.0,
            0.0,
            &nm_p,
            None
        ),
        Err(GyreError::Assert("root is not bracketed"))
    ));
    let mut bad_nm = nm_p.clone();
    bad_nm.r_root_solver = "BAD".into();
    assert!(matches!(
        solve_root_r(|x| Ok((x - 1.5, Stat::Ok)), 1.0, 2.0, 0.0, &bad_nm, None),
        Err(GyreError::Abort("invalid r_root_solver"))
    ));

    // Scan deduplication, unit conversion, and critical-layer check.
    let k = standard_constants();
    let omega_rot_cgs = 1.0e-4;
    let (global, rows, version) =
        uniform_sphere_mesa_data(11, 0.0, 0.0, omega_rot_cgs);
    let ml =
        EvolModel::from_mesa(&ModelPar::default(), &global, &rows, version, &k)
            .expect("from_mesa");
    let gr = ml.gr.clone();
    let md_p = ModePar {
        l: 1,
        m: 1,
        ..ModePar::default()
    };
    let os_p = OscPar::default();
    let rt_p = RotPar::default();
    let cx = Context::new(&ml, &k, &gr, &md_p, &os_p, &rt_p).expect("context");

    let sc = vec![
        ScanPar {
            freq_min: 1.0,
            freq_max: 3.0,
            n_freq: 3,
            freq_min_units: "NONE".into(),
            freq_max_units: "NONE".into(),
            ..ScanPar::default()
        },
        ScanPar {
            freq_min: 2.0,
            freq_max: 4.0,
            n_freq: 3,
            grid_type: "INVERSE".into(),
            freq_min_units: "NONE".into(),
            freq_max_units: "NONE".into(),
            ..ScanPar::default()
        },
    ];
    let omega = build_scan(&cx, &sc, "REAL").expect("build_scan");
    let expected = [1.0, 2.0, 8.0 / 3.0, 3.0, 4.0];
    assert_eq!(omega.len(), expected.len());
    for (a, b) in omega.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1.0e-15);
    }

    let w_rot = cx.omega_rot(gr.pt[0]).expect("omega_rot");
    assert!(matches!(
        check_scan(&cx, &gr, &[w_rot], &md_p),
        Err(GyreError::Assert("critical layer encountered"))
    ));
    assert!(matches!(
        check_scan(&cx, &gr, &[0.5 * w_rot, 1.5 * w_rot], &md_p),
        Err(GyreError::Assert(
            "transition between prograde and retrograde"
        ))
    ));

    let sc_file = [ScanPar {
        grid_type: "FILE".into(),
        ..ScanPar::default()
    }];
    assert!(matches!(
        build_scan(&cx, &sc_file, "REAL"),
        Err(GyreError::Unported { .. })
    ));
}

#[test]
fn grid_nesting_splitting_and_refinement_parity() {
    let gr =
        Grid::from_x(&[0.0, 0.25, 0.5, 0.5, 0.75, 1.0]).expect("multi-seg");
    assert_eq!(gr.s_i(), 1);
    assert_eq!(gr.s_o(), 2);
    assert_eq!(gr.n(), 6);
    assert_eq!(gr.j_s_i(1), 1);
    assert_eq!(gr.j_s_o(1), 3);
    assert_eq!(gr.j_s_i(2), 4);
    assert_eq!(gr.j_s_o(2), 6);
    assert_eq!(gr.s_x(0.5, false), 1);
    assert_eq!(gr.s_x(0.5, true), 2);
    assert_eq!(gr.pt_x(0.5, true), Point { s: 2, x: 0.5 });

    let nested = Grid::nest(&gr, 0.2, 0.8);
    assert_eq!(
        nested.pt,
        vec![
            Point { s: 1, x: 0.2 },
            Point { s: 1, x: 0.25 },
            Point { s: 1, x: 0.5 },
            Point { s: 2, x: 0.5 },
            Point { s: 2, x: 0.75 },
            Point { s: 2, x: 0.8 },
        ]
    );

    let split =
        Grid::split(&gr, &[true, false, false, false, false]).expect("split");
    assert_eq!(split.n(), 7);
    assert!((split.pt[1].x - 0.125).abs() < 1.0e-15);

    // Splitting a double-point boundary interval must assert.
    assert!(matches!(
        Grid::split(&gr, &[false, false, true, false, false]),
        Err(GyreError::Assert(_))
    ));
    // Single-point trailing segment must assert.
    assert!(matches!(
        Grid::from_x(&[0.0, 0.5, 0.5]),
        Err(GyreError::Assert("degenerate segment"))
    ));

    // Grid refinement mechanical/center weights and unported flags.
    let k = standard_constants();
    let (global, rows, version) = uniform_sphere_mesa_data(9, 1.0e-5, 0.0, 0.0);
    let ml =
        EvolModel::from_mesa(&ModelPar::default(), &global, &rows, version, &k)
            .expect("from_mesa");
    let md_p = ModePar::default();
    let os_p = OscPar::default();
    let rt_p = RotPar::default();
    let mut ref_gr = ml.gr.clone();
    let cx =
        Context::new(&ml, &k, &ref_gr, &md_p, &os_p, &rt_p).expect("context");

    let gr_p_thm = GridPar {
        w_thm: 1.0,
        ..GridPar::default()
    };
    assert!(matches!(
        refine_grid(&mut ref_gr, &cx, &[5.0], &gr_p_thm, &os_p),
        Err(GyreError::Unported { .. })
    ));

    let gr_p_str = GridPar {
        w_str: 1.0,
        ..GridPar::default()
    };
    assert!(matches!(
        refine_grid(&mut ref_gr, &cx, &[5.0], &gr_p_str, &os_p),
        Err(GyreError::Unported { .. })
    ));

    let gr_p = GridPar {
        w_osc: 10.0,
        w_ctr: 5.0,
        n_iter_max: 2,
        ..GridPar::default()
    };
    let n_before = ref_gr.n();
    refine_grid(&mut ref_gr, &cx, &[10.0], &gr_p, &os_p).expect("refine_grid");
    assert!(ref_gr.n() > n_before);
}

#[test]
fn sysmtx_block_det_and_hom_nullspace_parity() {
    // Real singular 2-block system: B_i u_1 = 0, E_l u_k + E_r u_{k+1} = 0,
    // B_o u_3 = 0 with nullspace u_k = c [1, 1].
    let mut sm_r = BlockSysmtx::<f64>::new(2, 2, 1, 1);
    let b_i = [1.0, -1.0];
    let b_o = [1.0, -1.0];
    let e_l = [-1.0, 0.0, 0.0, -1.0];
    let e_r = [1.0, 0.0, 0.0, 1.0];
    sm_r.set_b_i(&b_i, &[1.0]);
    sm_r.set_b_o(&b_o, &[1.0]);
    sm_r.set_e(0, &e_l, &e_r, ExtR::from_real(1.0));
    sm_r.set_e(1, &e_l, &e_r, ExtR::from_real(1.0));
    sm_r.factor().expect("factor real");
    let det_r = sm_r.det().expect("det real");
    assert!(det_r.to_real().abs() < 1.0e-13);
    let u = sm_r.soln_vec_hom().expect("hom real");
    assert_eq!(u.len(), 6);
    let nrm2: f64 = u.iter().map(|v| v * v).sum();
    assert!(nrm2 > 0.5);
    assert!((u[0] - u[1]).abs() < 1.0e-13);
    assert!((u[2] - u[0]).abs() < 1.0e-13);
    assert!((u[3] - u[1]).abs() < 1.0e-13);
    assert!((u[4] - u[2]).abs() < 1.0e-13);
    assert!((u[5] - u[3]).abs() < 1.0e-13);

    // Complex singular 2-block system with phase rotation u_{k+1} = [u_k0, i u_k1]:
    // u_1 = c [1, 1], u_2 = c [1, i], u_3 = c [1, -1], B_o = [1, 1].
    let c = Cplx::from_real;
    let mut sm_c = BlockSysmtx::<Cplx>::new(2, 2, 1, 1);
    let bi_c = [c(1.0), c(-1.0)];
    let bo_c = [c(1.0), c(1.0)];
    let el_c = [c(-1.0), c(0.0), c(0.0), Cplx::new(0.0, -1.0)];
    let er_c = [c(1.0), c(0.0), c(0.0), c(1.0)];
    sm_c.set_b_i(&bi_c, &[c(1.0)]);
    sm_c.set_b_o(&bo_c, &[c(1.0)]);
    sm_c.set_e(0, &el_c, &er_c, ExtC::from_real(1.0));
    sm_c.set_e(1, &el_c, &er_c, ExtC::from_real(1.0));
    sm_c.factor().expect("factor cplx");
    let det_c = sm_c.det().expect("det cplx");
    assert!(det_c.abs().to_real() < 1.0e-13);
    let uc = sm_c.soln_vec_hom().expect("hom cplx");
    assert!((uc[0] - uc[1]).re.abs() < 1.0e-13);
    assert!((uc[0] - uc[1]).im.abs() < 1.0e-13);
    assert!((uc[4] + uc[5]).re.abs() < 1.0e-13);
    assert!((uc[4] + uc[5]).im.abs() < 1.0e-13);
}

#[test]
fn magnus_eigendecomposition_and_gl2_exponential_parity() {
    // 2x2 real eigendecomposition with complex conjugate eigenvalues (rotation).
    let a2 = [0.0, 2.0, -2.0, 0.0];
    let (lam, vl, vr) = eigen_decompose_2_r(&a2);
    for i in 0..2 {
        assert!((lam[i].re).abs() < 1.0e-14);
        for j in 0..2 {
            let dot = vl[i] * vr[2 * j] + vl[i + 2] * vr[2 * j + 1];
            let target = if i == j { 1.0 } else { 0.0 };
            assert!((dot.re - target).abs() < 1.0e-13);
            assert!(dot.im.abs() < 1.0e-13);
        }
    }

    // 2x2 complex eigendecomposition.
    let ac2 = [
        Cplx::new(1.0, 0.5),
        Cplx::new(0.0, 1.0),
        Cplx::new(2.0, -0.5),
        Cplx::new(-1.0, 0.5),
    ];
    let (lam_c, vl_c, vr_c) = eigen_decompose_2_c(&ac2);
    for j in 0..2 {
        let lhs0 = ac2[0] * vr_c[2 * j] + ac2[2] * vr_c[2 * j + 1];
        let lhs1 = ac2[1] * vr_c[2 * j] + ac2[3] * vr_c[2 * j + 1];
        assert!((lhs0 - lam_c[j] * vr_c[2 * j]).re.abs() < 1.0e-13);
        assert!((lhs0 - lam_c[j] * vr_c[2 * j]).im.abs() < 1.0e-13);
        assert!((lhs1 - lam_c[j] * vr_c[2 * j + 1]).re.abs() < 1.0e-13);
        assert!((lhs1 - lam_c[j] * vr_c[2 * j + 1]).im.abs() < 1.0e-13);
        let dot = vl_c[j] * vr_c[2 * j] + vl_c[j + 2] * vr_c[2 * j + 1];
        assert!((dot.re - 1.0).abs() < 1.0e-13);
        assert!(dot.im.abs() < 1.0e-13);
    }

    // 4x4 LAPACK eigendecomposition with real-part sorting.
    let mut a4 = [
        3.0, 0.0, 0.0, 0.0, //
        0.0, -2.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 0.0,
    ];
    let eig4 = eigen_decompose_r(4, &mut a4, true).expect("eig4 real");
    let expected_l = [-2.0, 0.0, 1.0, 3.0];
    for i in 0..4 {
        assert!((eig4.lambda[i].re - expected_l[i]).abs() < 1.0e-13);
    }
    let mut c4: Vec<Cplx> = [
        Cplx::new(2.0, 1.0),
        Cplx::from_real(0.0),
        Cplx::from_real(0.0),
        Cplx::from_real(0.0),
        Cplx::from_real(0.0),
        Cplx::new(-1.0, -2.0),
        Cplx::from_real(0.0),
        Cplx::from_real(0.0),
        Cplx::from_real(0.0),
        Cplx::from_real(0.0),
        Cplx::new(0.5, 0.0),
        Cplx::from_real(0.0),
        Cplx::from_real(0.0),
        Cplx::from_real(0.0),
        Cplx::from_real(0.0),
        Cplx::new(-3.0, 0.5),
    ]
    .to_vec();
    let eig4c = eigen_decompose_c(4, &mut c4, true).expect("eig4 cplx");
    let expected_c = [-3.0, -1.0, 0.5, 2.0];
    for i in 0..4 {
        assert!((eig4c.lambda[i].re - expected_c[i]).abs() < 1.0e-13);
    }

    // Magnus GL2 stencil and exact matrix exponential on diagonal system.
    let (dx, pts) =
        magnus_stencil(Point { s: 1, x: 0.2 }, Point { s: 1, x: 0.4 }, "GL2")
            .expect("magnus_stencil");
    assert!((dx - 0.2).abs() < 1.0e-15);
    assert_eq!(pts.len(), 1);
    assert!((pts[0].x - 0.3).abs() < 1.0e-15);

    // Symmetric matrix A = [[-0.5, 2.5], [2.5, -0.5]] has eigenvalues +2, -3
    // and eigenvectors [1, 1], [1, -1].
    let mut d_r = [-0.5, 2.5, 2.5, -0.5];
    let (el_r, er_r, scl_r) =
        magnus_build_r(2, dx, &mut d_r).expect("magnus_build_r");
    // y_+(x) = exp(2x)[1, 1], y_-(x) = exp(-3x)[1, -1] =>
    // E_l y(0) + E_r y(dx) = 0.
    let ep = exp(2.0 * dx);
    let em = exp(-3.0 * dx);
    assert!(((el_r[0] + el_r[2]) + (er_r[0] + er_r[2]) * ep).abs() < 1.0e-13);
    assert!(((el_r[1] + el_r[3]) + (er_r[1] + er_r[3]) * ep).abs() < 1.0e-13);
    assert!(((el_r[0] - el_r[2]) + (er_r[0] - er_r[2]) * em).abs() < 1.0e-13);
    assert!(((el_r[1] - el_r[3]) + (er_r[1] - er_r[3]) * em).abs() < 1.0e-13);
    assert!((scl_r.to_real() - ep).abs() < 1.0e-13);

    let mut d_c = [
        Cplx::from_real(-0.5),
        Cplx::from_real(2.5),
        Cplx::from_real(2.5),
        Cplx::from_real(-0.5),
    ];
    let (el_c, er_c, scl_c) =
        magnus_build_c(2, dx, &mut d_c).expect("magnus_build_c");
    assert!(
        ((el_c[0] + el_c[2]).re + (er_c[0] + er_c[2]).re * ep).abs() < 1.0e-13
    );
    assert!(
        ((el_c[1] + el_c[3]).re + (er_c[1] + er_c[3]).re * ep).abs() < 1.0e-13
    );
    assert!(
        ((el_c[0] - el_c[2]).re + (er_c[0] - er_c[2]).re * em).abs() < 1.0e-13
    );
    assert!(
        ((el_c[1] - el_c[3]).re + (er_c[1] - er_c[3]).re * em).abs() < 1.0e-13
    );
    assert!((scl_c.to_cplx().re - ep).abs() < 1.0e-13);
}

#[test]
fn equations_bvp_prox_and_gyre_mesa_end_to_end_parity() {
    let k = standard_constants();
    let (global, rows, version) = pekeris_sphere_mesa_data(33);
    let ml =
        EvolModel::from_mesa(&ModelPar::default(), &global, &rows, version, &k)
            .expect("from_mesa");
    let gr = ml.gr.clone();
    let md_p = ModePar::default();
    let nm_p = NumPar::default();
    let os_p = OscPar::default();
    let rt_p = RotPar::default();
    let cx = Context::new(&ml, &k, &gr, &md_p, &os_p, &rt_p).expect("context");

    // 1. Cross-verify RadEqns, AdEqns, and NadEqns (l=0, alpha_grv=1, ups_t=0)
    // at an interior point: substituting y_4 = -U y_1 into AdEqns recovers RadEqns.
    let pt_mid = [Point { s: 1, x: 0.5 }];
    let mut r_eq = RadEqns::new(&os_p).expect("RadEqns");
    r_eq.stencil(&cx, &pt_mid).expect("r stencil");
    let mut a_eq = AdEqns::new(&cx, &os_p).expect("AdEqns");
    a_eq.stencil(&cx, &pt_mid).expect("a stencil");
    let mut n_eq = NadEqns::new(&cx, &os_p).expect("NadEqns");
    n_eq.stencil(&cx, &pt_mid).expect("n stencil");

    let w_test = 3.25;
    let ar = r_eq.a(0, w_test).expect("r a");
    let aa = a_eq.a(&cx, 0, w_test).expect("a a");
    let an = n_eq.a(&cx, 0, Cplx::from_real(w_test)).expect("n a");
    let u_mid = ml.coeff(idx::U, pt_mid[0]).expect("u");
    assert!((ar[0] - aa[0]).abs() < 1.0e-13);
    assert!((ar[2] - aa[4]).abs() < 1.0e-13);
    assert!((ar[1] - (aa[1] - u_mid * aa[13])).abs() < 1.0e-13);
    assert!((ar[3] - aa[5]).abs() < 1.0e-13);
    for col in 0..4 {
        for row in 0..4 {
            let nad_entry = an[row + 6 * col];
            let ad_entry = aa[row + 4 * col];
            assert!((nad_entry.re - ad_entry).abs() < 1.0e-13);
            assert!(nad_entry.im.abs() < 1.0e-13);
        }
    }

    // 2. Context freq_scale parity across unit families (including DELTA units).
    let f_none = cx.freq_scale("NONE").expect("NONE");
    assert_eq!(f_none, 1.0);
    let f_crit = cx.freq_scale("CRITICAL").expect("CRITICAL");
    assert!((f_crit - (27.0_f64 / 8.0).sqrt()).abs() < 1.0e-15);
    let dp = cx.freq_scale("ACOUSTIC_DELTA").expect("ACOUSTIC_DELTA");
    assert!(dp > 0.0 && dp.is_finite());

    // 3. Radial bracket search finds acoustic modes, and AdBvp agrees at w_0.
    let sc_p = [ScanPar {
        freq_min: 0.5,
        freq_max: 10.0,
        n_freq: 100,
        freq_min_units: "NONE".into(),
        freq_max_units: "NONE".into(),
        ..ScanPar::default()
    }];
    let omega_scan = build_scan(&cx, &sc_p, "REAL").expect("build_scan");
    check_scan(&cx, &gr, &omega_scan, &md_p).expect("check_scan");

    let mut rad_bp =
        RadBvp::new(&cx, &gr, &md_p, &nm_p, &os_p).expect("RadBvp");
    let mut ad_modes = Vec::new();
    bracket_search(
        &mut rad_bp,
        &omega_scan,
        omega_scan[0],
        *omega_scan.last().expect("non-empty scan"),
        &nm_p,
        &mut NoTap,
        &mut |md, _n_iter, chi| {
            assert_eq!(md.wave.l, 0);
            assert_eq!(md.n_g, 0);
            assert_eq!(md.n_pg, md.n_p);
            assert!((md.wave.e() - 1.0).abs() < 1.0e-11);
            assert!(chi.to_real().abs() < 1.0e-8);
            ad_modes.push((md.wave.omega.re, md.n_pg, md.wave.id));
            Ok(())
        },
    )
    .expect("bracket_search");
    let (w0, npg0, _id0) = ad_modes.first().copied().expect("radial modes");

    // Verify RadBvp vs AdBvp parity on an envelope grid (x >= 0.09375, ZERO_R),
    // where AdBvp avoids the l=0 x=0 COLLOC_GL2 center-row degeneracy.
    let env_ml_p = ModelPar {
        add_center: false,
        ..ModelPar::default()
    };
    let env_ml =
        EvolModel::from_mesa(&env_ml_p, &global, &rows[3..], version, &k)
            .expect("env from_mesa");
    let env_gr = env_ml.gr.clone();
    let os_zr = OscPar {
        inner_bound: "ZERO_R".into(),
        ..OscPar::default()
    };
    let env_cx =
        Context::new(&env_ml, &k, &env_gr, &md_p, &os_zr, &rt_p).expect("cx");
    let mut env_rad =
        RadBvp::new(&env_cx, &env_gr, &md_p, &nm_p, &os_zr).expect("env rad");
    let mut env_modes = Vec::new();
    bracket_search(
        &mut env_rad,
        &omega_scan,
        omega_scan[0],
        *omega_scan.last().expect("non-empty"),
        &nm_p,
        &mut NoTap,
        &mut |md, _n_iter, chi| {
            assert!(chi.to_real().abs() < 1.0e-8);
            env_modes.push((md.wave.omega.re, md.n_pg));
            Ok(())
        },
    )
    .expect("env rad bracket");
    let (w_env, npg_env) = env_modes.first().copied().expect("env radial mode");

    let mut env_ad =
        AdBvp::new(&env_cx, &env_gr, &md_p, &nm_p, &os_zr).expect("env ad");
    let ad_scan = [ScanPar {
        freq_min: w_env * 0.98,
        freq_max: w_env * 1.02,
        n_freq: 20,
        freq_min_units: "NONE".into(),
        freq_max_units: "NONE".into(),
        ..ScanPar::default()
    }];
    let ad_br = build_scan(&env_cx, &ad_scan, "REAL").expect("ad scan");
    let mut ad_found = Vec::new();
    bracket_search(
        &mut env_ad,
        &ad_br,
        ad_br[0],
        *ad_br.last().expect("non-empty"),
        &nm_p,
        &mut NoTap,
        &mut |md, _n_iter, chi| {
            assert!(chi.to_real().abs() < 1.0e-8);
            ad_found.push((md.wave.omega.re, md.n_pg));
            Ok(())
        },
    )
    .expect("ad bracket_search");
    let (w_env_ad, npg_env_ad) =
        ad_found.first().copied().expect("env ad mode");
    assert!((w_env_ad - w_env).abs() < 1.0e-2);
    assert_eq!(npg_env_ad, npg_env);

    // Also confirm rad_wave_hom works directly on a 2-segment model & grid.
    let mut seg_rows = rows.clone();
    seg_rows.insert(16, rows[16].clone());
    let seg_ml = EvolModel::from_mesa(
        &ModelPar::default(),
        &global,
        &seg_rows,
        version,
        &k,
    )
    .expect("seg from_mesa");
    let seg_gr = seg_ml.gr.clone();
    assert_eq!(seg_gr.s_o(), 2);
    let seg_cx = Context::new(&seg_ml, &k, &seg_gr, &md_p, &os_p, &rt_p)
        .expect("seg cx");
    let mut seg_rad =
        RadBvp::new(&seg_cx, &seg_gr, &md_p, &nm_p, &os_p).expect("seg RadBvp");
    let seg_wv = rad_wave_hom(&mut seg_rad, w0, 1, &mut NoTap)
        .expect("seg rad_wave_hom");
    let seg_md = Mode::new(seg_wv).expect("seg Mode");
    assert!((seg_md.wave.e() - 1.0).abs() < 1.0e-12);

    // 4. Non-radial (l=1) AdBvp -> NadBvp prox_search (COLLOC_GL2 and
    // MAGNUS_GL2) recovers w1_ad identically in the ups_t = 0 decoupled limit.
    let md_p1 = ModePar {
        l: 1,
        ..ModePar::default()
    };
    let cx1 = Context::new(&ml, &k, &gr, &md_p1, &os_p, &rt_p).expect("cx1");
    for scheme in ["COLLOC_GL2", "MAGNUS_GL2"] {
        let mut nm_s = nm_p.clone();
        nm_s.diff_scheme = scheme.into();
        let mut ad_bp1 =
            AdBvp::new(&cx1, &gr, &md_p1, &nm_s, &os_p).expect("AdBvp l=1");
        let mut l1_modes = Vec::new();
        bracket_search(
            &mut ad_bp1,
            &omega_scan,
            omega_scan[0],
            *omega_scan.last().expect("non-empty"),
            &nm_s,
            &mut NoTap,
            &mut |md, _n_iter, chi| {
                assert!(chi.to_real().abs() < 1.0e-8);
                l1_modes.push((md.wave.omega.re, md.n_pg, md.wave.id));
                Ok(())
            },
        )
        .expect("ad l1 bracket");
        let (w1, npg1, id1) = l1_modes.first().copied().expect("l=1 ad mode");
        let wv_ad1 =
            ad_wave_hom(&mut ad_bp1, w1, id1, &mut NoTap).expect("ad1 hom");
        let md_ad1 = Mode::new(wv_ad1).expect("ad1 mode");
        assert_eq!(md_ad1.n_pg, npg1);

        let mut nad_bp =
            NadBvp::new(&cx1, &gr, &md_p1, &nm_s, &os_p).expect("NadBvp");
        let mut nad_found = Vec::new();
        prox_search(
            &mut nad_bp,
            &[Cplx::from_real(w1)],
            &[id1],
            omega_scan[0],
            *omega_scan.last().expect("non-empty"),
            &nm_s,
            &mut NoTap,
            &mut |md, _n_iter, chi| {
                assert!((md.wave.e() - 1.0).abs() < 1.0e-11);
                assert!(chi.to_real().abs() < 1.0e-4);
                nad_found.push((md.wave.omega, md.n_pg));
                Ok(())
            },
        )
        .expect("prox_search");
        let &(w_nad, npg_nad) = nad_found.first().expect("nad mode");
        assert!((w_nad.re - w1).abs() < 1.0e-5);
        assert!(w_nad.im.abs() > 1.0e-6 && w_nad.im.abs() < 1.0e-3);
        assert_eq!(npg_nad, npg1);

        let wv_nad = nad_wave_hom(&mut nad_bp, w_nad, 1, &mut NoTap)
            .expect("nad_wave_hom");
        let md_nad = Mode::new(wv_nad).expect("nad mode");
        assert_eq!(md_nad.n_pg, npg1);
    }

    // 5. End-to-end GyreMesa driver with JCD variables_set & COLLOC_GL4.
    let gyre_in = "\
&model /
&mode l = 0 /
&osc variables_set = 'JCD', outer_bound = 'VACUUM' /
&rot /
&num diff_scheme = 'COLLOC_GL4' /
&scan grid_type = 'LINEAR', freq_min = 0.5, freq_max = 10.0, n_freq = 100 /
&grid /
";
    let mut gm = GyreMesa::init(gyre_in).expect("GyreMesa init");
    gm.set_constant_str("GYRE_DIR", "inline").expect("set dir");
    gm.set_model(&global, &rows, version).expect("set_model");
    let mut gm_modes = Vec::new();
    gm.get_modes(0, &mut |md| {
        gm_modes.push((md.wave.omega.re, md.n_pg));
        Ok(0)
    })
    .expect("get_modes");
    let &(gm0_w, gm0_npg) = gm_modes.first().expect("GyreMesa modes");
    assert_eq!(gm0_npg, npg0);
    assert!((gm0_w - w0).abs() < 1.0e-2);

    // 6. Explicit Unported error contracts remain intact (no silent fallback).
    let mut os_no_grv = os_p.clone();
    os_no_grv.alpha_grv = 0.0;
    assert!(matches!(
        RadBvp::new(&cx, &gr, &md_p, &nm_p, &os_no_grv),
        Err(GyreError::Unported { .. })
    ));

    let mut os_jcd = os_p.clone();
    os_jcd.variables_set = "JCD".into();
    assert!(matches!(
        AdTrans::new(&os_jcd),
        Err(GyreError::Unported { .. })
    ));

    let mut nm_gl4 = nm_p.clone();
    nm_gl4.diff_scheme = "COLLOC_GL4".into();
    assert!(matches!(
        NadDiff::new(&cx, gr.pt[0], gr.pt[1], &nm_gl4, &os_p),
        Err(GyreError::Unported { .. })
    ));

    let mut nm_trapz = nm_p.clone();
    nm_trapz.diff_scheme = "TRAPZ".into();
    assert!(matches!(
        NadDiff::new(&cx, gr.pt[0], gr.pt[1], &nm_trapz, &os_p),
        Err(GyreError::Unported { .. })
    ));

    let mut rt_tar = rt_p.clone();
    rt_tar.coriolis_method = "TAR".into();
    assert!(matches!(
        Context::new(&ml, &k, &gr, &md_p, &os_p, &rt_tar),
        Err(GyreError::Unported { .. })
    ));

    assert!(matches!(
        EvolModel::from_mesa(&ModelPar::default(), &global, &rows, 120, &k),
        Err(GyreError::Unported { .. })
    ));
}
