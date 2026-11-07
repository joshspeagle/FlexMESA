// SPDX-License-Identifier: LGPL-3.0-or-later
//! Golden parity and analytical derivative tests for `flexmesa-kap`.

#![allow(clippy::excessive_precision, clippy::similar_names)]

use flexmesa_kap::{
    compton_opacity, do_electron_conduction, do_electron_conduction_potekhin,
    get_kap_results, kap_get_zones, kapcn_get, KapCoOption, KapConfig,
    KapLowTOption, KapOption, I_FRAC_COMPTON, I_FRAC_HIGH_T, I_FRAC_LOW_T,
    I_FRAC_TYPE2,
};

fn assert_rel_close(actual: f64, expected: f64, rtol: f64, label: &str) {
    let diff = (actual - expected).abs();
    let scale = expected.abs().max(1.0e-30);
    assert!(
        diff <= rtol * scale,
        "{label}: actual={actual:.16e}, expected={expected:.16e}, rel_err={:.3e} > rtol={rtol:.1e}",
        diff / scale
    );
}

#[test]
fn test_kap_fixed_metals_matches_fortran_golden_test_1() {
    let cfg = KapConfig {
        zbase: 0.0,
        use_type2_opacities: false,
        ..KapConfig::default()
    };
    let log_t = 6.0_f64;
    let log_rho = -6.0_f64;
    let z = 1.800_000_000_000_001_6e-2_f64;
    let zbar = 1.101_885_743_409_164_7_f64;
    let x = 0.7_f64;
    let lnfree_e = -1.696_507_362_237_710_5e-1_f64;

    let out = get_kap_results(
        &cfg, zbar, x, z, 0.0, 0.0, 0.0, 0.0, log_rho, log_t, lnfree_e, 0.0,
        0.0, -10.0, 0.0, 0.0,
    )
    .expect("Test 1 evaluation failed");

    assert_rel_close(
        out.log10_kap,
        -4.424_000_000_011_061e-1,
        1.0e-12,
        "Test 1 log10kap",
    );
    assert_rel_close(
        out.dlnkap_dln_rho,
        2.281_674_219_017_579e-2,
        1.0e-11,
        "Test 1 dlnkap_dlnRho",
    );
    assert_rel_close(
        out.dlnkap_dln_t,
        3.538_705_857_187_698e-2,
        1.0e-11,
        "Test 1 dlnkap_dlnT",
    );
    assert_rel_close(
        out.kap,
        3.610_771_450_994_333_4e-1,
        1.0e-12,
        "Test 1 kap",
    );
    assert!((out.kap_fracs[I_FRAC_TYPE2] - 0.0).abs() <= 1.0e-15);
    assert!((out.kap_fracs[I_FRAC_HIGH_T] - 1.0).abs() <= 1.0e-15);
}

#[test]
fn test_kap_co_enhanced_matches_fortran_golden_test_2() {
    let cfg = KapConfig {
        zbase: 0.018,
        kap_co_option: KapCoOption::Gn93Co,
        use_type2_opacities: true,
        ..KapConfig::default()
    };
    let log_t = 6.0_f64;
    let log_rho = -6.0_f64;
    let z = 5.800_000_000_000_005e-2_f64;
    let zbar = 2.074_131_706_506_234_f64;
    let x = 0.0_f64;
    let xc = 2.411_961_6e-2_f64;
    let xn = 9.567_36e-4_f64;
    let xo = 2.768_316_4e-2_f64;
    let xne = 1.776_024e-3_f64;
    let lnfree_e = -6.969_230_350_735_236e-1_f64;

    let out = get_kap_results(
        &cfg, zbar, x, z, xc, xn, xo, xne, log_rho, log_t, lnfree_e, 0.0, 0.0,
        -10.0, 0.0, 0.0,
    )
    .expect("Test 2 evaluation failed");

    assert_rel_close(
        out.log10_kap,
        -6.641_677_699_836_668e-1,
        1.0e-12,
        "Test 2 log10kap",
    );
    assert_rel_close(
        out.dlnkap_dln_rho,
        3.648_862_649_703_165_5e-2,
        1.0e-11,
        "Test 2 dlnkap_dlnRho",
    );
    assert_rel_close(
        out.dlnkap_dln_t,
        -1.908_154_874_225_351e-2,
        1.0e-11,
        "Test 2 dlnkap_dlnT",
    );
    assert_rel_close(
        out.kap,
        2.166_866_872_338_855_7e-1,
        1.0e-12,
        "Test 2 kap",
    );
    assert!((out.kap_fracs[I_FRAC_TYPE2] - 1.0).abs() <= 1.0e-15);
}

#[test]
fn test_kap_aesopus_matches_fortran_golden_test_4() {
    let cfg = KapConfig {
        zbase: 0.014,
        kap_option: KapOption::A09,
        kap_co_option: KapCoOption::A09Co,
        kap_low_t_option: KapLowTOption::Aesopus,
        cubic_interpolation_in_x: false,
        cubic_interpolation_in_z: false,
        include_electron_conduction: true,
        use_zbase_for_type1: true,
        use_type2_opacities: true,
        ..KapConfig::default()
    };
    let log_t = 3.557_150_454_626_023_5_f64;
    let log_rho = -8.249_643_069_901_467_f64;
    let z = 2.007_412_071_348_735_3e-2_f64;
    let zbar = 1.100_719_782_401_773_2_f64;
    let x = 6.788_866_252_318_019e-1_f64;
    let xc = 2.996_845_870_980_643_2e-3_f64;
    let xn = 1.527_090_014_563_059_1e-3_f64;
    let xo = 9.337_626_324_051_438e-3_f64;
    let lnfree_e = -9.210_342_466_630_861_f64;

    let out = get_kap_results(
        &cfg, zbar, x, z, xc, xn, xo, 0.0, log_rho, log_t, lnfree_e, 0.0, 0.0,
        -20.0, 0.0, 0.0,
    )
    .expect("Test 4 evaluation failed");

    // Note: AESOPUS calls interp_evbicub_db where upstream Fortran bicub_db.f omitted
    // `implicit none` on local coordinates (xp, yp, cx, cy in f32), whereas FlexMESA
    // evaluates the Hermite basis in full f64 precision (~5e-7 relative difference).
    assert_rel_close(
        out.log10_kap,
        -2.523_381_870_188_199,
        1.0e-7,
        "Test 4 log10kap",
    );
    assert_rel_close(
        out.dlnkap_dln_rho,
        5.728_409_917_973_51e-1,
        1.0e-6,
        "Test 4 dlnkap_dlnRho",
    );
    assert_rel_close(
        out.dlnkap_dln_t,
        8.462_557_917_457_387,
        1.0e-6,
        "Test 4 dlnkap_dlnT",
    );
    assert_rel_close(out.kap, 2.996_526_548_625_922_3e-3, 2.0e-7, "Test 4 kap");
    assert!((out.kap_fracs[I_FRAC_LOW_T] - 1.0).abs() <= 1.0e-15);
    assert!((out.kap_fracs[I_FRAC_TYPE2] - 0.0).abs() <= 1.0e-15);
}

#[test]
fn test_kap_conduction_and_compton_derivatives_match_finite_differences() {
    let cfg = KapConfig::default();
    for &zbar in &[1.0_f64, 1.5, 2.0, 2.5, 6.0, 26.0] {
        let log_rho = 5.2_f64;
        let log_t = 7.8_f64;
        let (kap, dln_dr, dln_dt) =
            do_electron_conduction(&cfg, zbar, log_rho, log_t)
                .expect("conduction failed");
        assert!(kap > 0.0 && kap.is_finite());

        let eps = 1.0e-6;
        let (kr_p, _, _) =
            do_electron_conduction(&cfg, zbar, log_rho + eps, log_t)
                .expect("fd +rho");
        let (kr_m, _, _) =
            do_electron_conduction(&cfg, zbar, log_rho - eps, log_t)
                .expect("fd -rho");
        let fd_rho = (kr_p.log10() - kr_m.log10()) / (2.0 * eps);
        assert_rel_close(dln_dr, fd_rho, 1.0e-6, "conduction dlnkap_dlnRho");

        let (kt_p, _, _) =
            do_electron_conduction(&cfg, zbar, log_rho, log_t + eps)
                .expect("fd +t");
        let (kt_m, _, _) =
            do_electron_conduction(&cfg, zbar, log_rho, log_t - eps)
                .expect("fd -t");
        let fd_t = (kt_p.log10() - kt_m.log10()) / (2.0 * eps);
        assert_rel_close(dln_dt, fd_t, 1.0e-6, "conduction dlnkap_dlnT");

        let (kp, _, _) = do_electron_conduction_potekhin(zbar, log_rho, log_t)
            .expect("potekhin");
        assert!(kp > 0.0);
    }

    // Compton opacity finite-difference check
    let rho = 10.0_f64;
    let t = 3.0e8_f64;
    let lnfree_e = -0.15_f64;
    let d_lnfree_dr = 0.02_f64;
    let d_lnfree_dt = 0.05_f64;
    let eta = 1.2_f64;
    let d_eta_dr = 0.65_f64;
    let d_eta_dt = -0.95_f64;
    let (kc, dlnkc_dr, dlnkc_dt) = compton_opacity(
        rho,
        t,
        lnfree_e,
        d_lnfree_dr,
        d_lnfree_dt,
        eta,
        d_eta_dr,
        d_eta_dt,
    );
    let eps = 1.0e-6;
    let (kc_rp, _, _) = compton_opacity(
        rho * (1.0 + eps),
        t,
        lnfree_e + d_lnfree_dr * eps,
        d_lnfree_dr,
        d_lnfree_dt,
        eta + d_eta_dr * eps,
        d_eta_dr,
        d_eta_dt,
    );
    let (kc_rm, _, _) = compton_opacity(
        rho * (1.0 - eps),
        t,
        lnfree_e - d_lnfree_dr * eps,
        d_lnfree_dr,
        d_lnfree_dt,
        eta - d_eta_dr * eps,
        d_eta_dr,
        d_eta_dt,
    );
    let fd_dr = (kc_rp.ln() - kc_rm.ln()) / (2.0 * eps);
    assert_rel_close(dlnkc_dr, fd_dr, 1.0e-6, "compton dlnkap_dlnRho");

    let (kc_tp, _, _) = compton_opacity(
        rho,
        t * (1.0 + eps),
        lnfree_e + d_lnfree_dt * eps,
        d_lnfree_dr,
        d_lnfree_dt,
        eta + d_eta_dt * eps,
        d_eta_dr,
        d_eta_dt,
    );
    let (kc_tm, _, _) = compton_opacity(
        rho,
        t * (1.0 - eps),
        lnfree_e - d_lnfree_dt * eps,
        d_lnfree_dr,
        d_lnfree_dt,
        eta - d_eta_dt * eps,
        d_eta_dr,
        d_eta_dt,
    );
    let fd_dt = (kc_tp.ln() - kc_tm.ln()) / (2.0 * eps);
    assert_rel_close(dlnkc_dt, fd_dt, 1.0e-6, "compton dlnkap_dlnT");
    assert!(kc > 0.0);
}

#[test]
fn test_kapcn_and_multi_zone_batch_evaluation() {
    let (kap_cn, dln_dr, dln_dt) = kapcn_get(0.02, 0.70, 1.0, 1.0, -10.2, 3.6)
        .expect("kapCN evaluation failed");
    assert_rel_close(
        kap_cn,
        10.0_f64.powf(-3.361),
        1.0e-12,
        "kapCN exact knot",
    );
    assert!(dln_dr.is_finite() && dln_dt.is_finite());

    let cfg = KapConfig::default();
    let zbar = [1.101_885_743_409_164_7, 1.2, 1.5];
    let x = [0.7, 0.7, 0.7];
    let z = [0.018, 0.02, 0.02];
    let zeros = [0.0; 3];
    let log_rho = [-6.0, -4.0, 2.0];
    let log_t = [6.0, 3.84, 8.5];
    let lnfree_e = [-0.17, -2.0, -0.15];
    let eta = [-10.0, -12.0, -3.0];

    let outs = kap_get_zones(
        &cfg, &zbar, &x, &z, &zeros, &zeros, &zeros, &zeros, &log_rho, &log_t,
        &lnfree_e, &zeros, &zeros, &eta, &zeros, &zeros,
    )
    .expect("kap_get_zones failed");
    assert_eq!(outs.len(), 3);
    // Zone 1 is inside the lowT-highT blend [3.80, 3.88]
    assert!(
        outs[1].kap_fracs[I_FRAC_LOW_T] > 0.0
            && outs[1].kap_fracs[I_FRAC_HIGH_T] > 0.0
    );
    // Zone 2 is inside the Compton blend [8.19, 8.69]
    assert!(
        outs[2].kap_fracs[I_FRAC_COMPTON] > 0.0
            && outs[2].kap_fracs[I_FRAC_COMPTON] < 1.0
    );
}
