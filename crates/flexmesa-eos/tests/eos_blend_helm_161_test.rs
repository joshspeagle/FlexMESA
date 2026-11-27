// SPDX-License-Identifier: GPL-3.0-or-later
#![forbid(unsafe_code)]
//! Issue `#161` invariant tests for `flexmesa-eos` non-`PC`/non-`Skye`
//! modules (`gamma_de`, `helm::components`, `helm::eval`, `cms::config`,
//! `tables_4d::blend_alfa`, `blend_dt`, and `eos_pt`).

#![allow(clippy::float_cmp, clippy::similar_names)]

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{AVO, KERG, LN10};
use flexmesa_eos::helm::{
    eval_ideal_ions, HELM_SIFAC, H_GAM1, H_GAM3, H_PTOT, H_VALID,
};
use flexmesa_eos::{
    eos_dt_get, eos_gamma_de_get, eos_pt_get_with_linear, get_cms_alfa,
    get_opal_scvh_alfa_and_partials, helmeos2, CmsConfig, EosConfig, EosError,
    HelmConfig, HelmTable, OpalScvhConfig, I_CHI_RHO, I_CHI_T, I_CP, I_CV,
    I_ETA, I_FRAC_IDEAL, I_GAMMA1, I_GAMMA3, I_GRAD_AD, I_LN_E, I_LN_FREE_E,
    I_LN_PGAS, I_LN_S, I_MU,
};
use flexmesa_math::{exp, log, log10, pow};

#[test]
fn test_gamma_de_matches_eosde_eval_fraction_log_e_and_entropy_guard() {
    let abar = 1.29;
    let zbar = 1.10;
    let rho = 1.0e-3;
    let gamma = 5.0 / 3.0;

    let avo_k_div_abar = AVO * KERG / abar;
    let expected_cv = avo_k_div_abar / (gamma - 1.0);

    for energy in [1.0e10, 2.5e11, 3.5e12, 7.0e14] {
        let out = eos_gamma_de_get(abar, zbar, rho, energy, gamma);
        let expected_p = (gamma - 1.0) * energy * rho;
        let expected_t = (gamma - 1.0) * energy / avo_k_div_abar;

        // `eos/private/eosde_eval.f90:51,84-99` leaves `i_frac_ideal == 0`.
        assert_eq!(out.res[I_FRAC_IDEAL], 0.0);
        // `eos/private/eosde_eval.f90:85` computes `log10(energy) * ln10`.
        assert_eq!(out.res[I_LN_E].to_bits(), (log10(energy) * LN10).to_bits());
        assert_eq!(out.t.to_bits(), expected_t.to_bits());
        assert_eq!(out.log10_t.to_bits(), log10(expected_t).to_bits());
        assert_eq!(out.res[I_LN_PGAS].to_bits(), log(expected_p).to_bits());
        assert_eq!(out.res[I_CV].to_bits(), expected_cv.to_bits());
        assert_eq!(out.res[I_CP].to_bits(), (gamma * expected_cv).to_bits());
        assert_eq!(
            out.res[I_GRAD_AD].to_bits(),
            ((gamma - 1.0) / gamma).to_bits(),
        );
        assert_eq!(out.res[I_CHI_RHO], 1.0);
        assert_eq!(out.res[I_CHI_T], 1.0);
        assert_eq!(out.res[I_GAMMA1], gamma);
        assert_eq!(out.res[I_GAMMA3], gamma);
        assert_eq!(out.res[I_MU].to_bits(), (1.0 / abar).to_bits());
        assert_eq!(out.res[I_LN_FREE_E], -1.0e99);
        assert_eq!(out.res[I_ETA], 0.0);
    }

    // Negative/non-finite `entropy` clamps to `1e-99` (`eosde_eval.f90:65`).
    let expected_ln_floor = log(1.0e-99_f64).to_bits();
    let neg_s = eos_gamma_de_get(abar, zbar, 1.0e100, 1.0e10, gamma);
    assert_eq!(neg_s.res[I_LN_S].to_bits(), expected_ln_floor);
    let inf_s = eos_gamma_de_get(abar, zbar, 1.0e-300, 1.0e10, gamma);
    assert_eq!(inf_s.res[I_LN_S].to_bits(), expected_ln_floor);
}

fn solve_temp_for_sion_raw(target_sion: f64, den: f64, abar: f64) -> f64 {
    let y = KERG * AVO / abar;
    let coeff = (AVO / abar * den) * HELM_SIFAC * pow(1.0 / (KERG * abar), 1.5);
    let z_s_target = exp(2.5 - target_sion / y);
    pow(coeff / z_s_target, 2.0 / 3.0)
}

#[test]
fn test_eval_ideal_ions_sion0_exponential_rolloff_and_derivatives() {
    let abar = 12.0;
    let sion0 = 1.0e3_f64;

    // 1. Deep degeneracy (`sion_raw < 0`) hits the `1e-20` exponential floor:
    //    `sion = sion0 * 1e-20 = 1e-17`, instead of the old rational pole.
    let (_, _, sion_floor, _, _) = eval_ideal_ions(500.0, 1.0e6, abar);
    assert_eq!(sion_floor.val.to_bits(), (sion0 * 1.0e-20).to_bits());
    assert_eq!(sion_floor.ddt, 0.0);

    // 2. Active rolloff for `0 < sion_raw < 1000` (`target_sion = 500`) and
    //    active negative rolloff (`target_sion = -2000`), verifying exponential
    //    suppression, first derivatives vs finite differences, and `dsionddt = 0`.
    let den = 1.0e4;
    let y = KERG * AVO * (1.0 / abar);
    for target_sion in [500.0, -2000.0] {
        let temp = solve_temp_for_sion_raw(target_sion, den, abar);
        let (pion, eion, sion, _, etaion) = eval_ideal_ions(temp, den, abar);
        let sion_raw =
            (eion.val + pion.val * (1.0 / den)) * (1.0 / temp) - etaion.val * y;
        assert!((sion_raw - target_sion).abs() < 1.0e-5);
        let roll = 1.0 - sion_raw / sion0;
        let expected_factor = exp(-roll * roll);
        let expected_sion = sion0 * expected_factor;
        assert!(sion.val > 0.0 && sion.val < sion0);
        assert!(
            (sion.val - expected_sion).abs() / expected_sion < 1.0e-14,
            "target={target_sion}, got={}, expected={expected_sion}",
            sion.val,
        );
        assert_eq!(sion.ddt, 0.0);

        let dt = temp * 1.0e-8;
        let (_, _, sp_t, _, _) = eval_ideal_ions(temp + dt, den, abar);
        let (_, _, sm_t, _, _) = eval_ideal_ions(temp - dt, den, abar);
        let fd_dt = (sp_t.val - sm_t.val) / (2.0 * dt);
        assert!(
            (sion.dt - fd_dt).abs() / fd_dt.abs() < 1.0e-7,
            "dsion/dT mismatch at target={target_sion}: {} vs {fd_dt}",
            sion.dt,
        );

        let dr = den * 1.0e-8;
        let (_, _, sp_r, _, _) = eval_ideal_ions(temp, den + dr, abar);
        let (_, _, sm_r, _, _) = eval_ideal_ions(temp, den - dr, abar);
        let fd_dd = (sp_r.val - sm_r.val) / (2.0 * dr);
        assert!(
            (sion.dd - fd_dd).abs() / fd_dd.abs() < 1.0e-7,
            "dsion/drho mismatch at target={target_sion}: {} vs {fd_dd}",
            sion.dd,
        );
    }

    // 3. Above rolloff (`sion >> 1000`), verify second derivatives match FD.
    let temp_hi = 2.0e8;
    let den_hi = 100.0;
    let abar_he = 4.0;
    let (_, _, sion_hi, _, _) = eval_ideal_ions(temp_hi, den_hi, abar_he);
    assert!(sion_hi.val > 1.0e7);

    let dr = den_hi * 1.0e-6;
    let (_, _, s_rp, _, _) = eval_ideal_ions(temp_hi, den_hi + dr, abar_he);
    let (_, _, s_rm, _, _) = eval_ideal_ions(temp_hi, den_hi - dr, abar_he);
    let fd_ddd = (s_rp.dd - s_rm.dd) / (2.0 * dr);
    assert!((sion_hi.ddd - fd_ddd).abs() / fd_ddd.abs() < 1.0e-7);

    let da = abar_he * 1.0e-6;
    let (_, _, s_ap, _, _) = eval_ideal_ions(temp_hi, den_hi, abar_he + da);
    let (_, _, s_am, _, _) = eval_ideal_ions(temp_hi, den_hi, abar_he - da);
    let fd_dda = (s_ap.dd - s_am.dd) / (2.0 * da);
    let fd_daa = (s_ap.da - s_am.da) / (2.0 * da);
    assert!((sion_hi.dda - fd_dda).abs() / fd_dda.abs() < 1.0e-7);
    assert!((sion_hi.daa - fd_daa).abs() / fd_daa.abs() < 1.0e-7);
}

#[test]
fn test_cms_trapezoid_strict_boundaries_no_nan_and_ignores_z() {
    // 1. Default config (`-1.0` sentinel bounds) at `x == -1.0` must not NaN.
    let def = CmsConfig::default();
    let (a_def, dt_def, dr_def) = get_cms_alfa(&def, -1.0, -1.0, 0.02);
    assert_eq!(a_def, 1.0);
    assert_eq!(dt_def, 0.0);
    assert_eq!(dr_def, 0.0);

    // 2. Degenerate step (`min_any == min_all == 3.0`) at `log_t == 3.0` gives
    //    `alfa == 0.0` (strict `<` enters `x < max_all`, avoiding `0/0 = NaN`).
    let step_cfg = CmsConfig {
        logt_min_for_any_cms: 3.0,
        logt_min_for_all_cms: 3.0,
        logt_max_for_all_cms: 6.0,
        logt_max_for_any_cms: 6.5,
        logrho_min_for_any_cms: -10.0,
        logrho_min_for_all_cms: -10.0,
        logrho_max_for_all_cms: 10.0,
        logrho_max_for_any_cms: 10.0,
        logq_min_for_any_cms: -20.0,
        logq_min_for_all_cms: -20.0,
        logq_max_for_all_cms: 40.0,
        logq_max_for_any_cms: 40.0,
        ..CmsConfig::default()
    };
    let (a_step, dt_step, dr_step) = get_cms_alfa(&step_cfg, 0.0, 3.0, 0.0);
    assert!(a_step.is_finite() && dt_step.is_finite() && dr_step.is_finite());
    assert_eq!(a_step, 0.0);
    assert_eq!(dt_step, 0.0);

    // 3. Strict `<` boundary behavior at `x == min_all` and `x == max_all`,
    //    plus `Z`-independence matching `eoscms_eval.f90:68-148`.
    let ramp_cfg = CmsConfig {
        max_z_for_all_cms: 0.02,
        max_z_for_any_cms: 0.04,
        logt_min_for_any_cms: 2.0,
        logt_min_for_all_cms: 3.0,
        logt_max_for_all_cms: 5.0,
        logt_max_for_any_cms: 7.0,
        ..step_cfg
    };
    // `log_t == min_all = 3.0` is on the plateau (`blend = 1`, zero slope).
    let (a_lo, dt_lo, dr_lo) = get_cms_alfa(&ramp_cfg, 0.0, 3.0, 0.0);
    assert_eq!((a_lo, dt_lo, dr_lo), (0.0, 0.0, 0.0));

    // `log_t == max_all = 5.0` enters upper ramp with `d(alfa)/dlogT = +0.5`.
    let (a_hi, dt_hi, dr_hi) = get_cms_alfa(&ramp_cfg, 0.0, 5.0, 0.0);
    assert_eq!((a_hi, dt_hi, dr_hi), (0.0, 0.5, 0.0));

    // Non-zero Z within `[max_z_for_all_cms, max_z_for_any_cms]` matches `Z = 0`.
    let mid_z0 = get_cms_alfa(&ramp_cfg, 0.0, 2.5, 0.0);
    let mid_z03 = get_cms_alfa(&ramp_cfg, 0.0, 2.5, 0.03);
    assert_eq!(mid_z0, (0.5, -1.0, 0.0));
    assert_eq!(mid_z03, mid_z0);
}

#[test]
fn test_opal_scvh_z_blend_matches_eosdt_eval_1450_multiplicative_zfactor() {
    let cfg = OpalScvhConfig::default();

    // `BlendInY` at `log_t = 7.55` between `logT2 = 7.5` and `logT1 = 7.6`.
    let (a0, dr0, dt0) = get_opal_scvh_alfa_and_partials(&cfg, 7.55, 0.0, 0.0)
        .expect("valid OPAL alfa at Z=0");
    assert!((a0 - 0.5).abs() < 1.0e-14);
    assert_eq!(dr0, 0.0);
    assert!((dt0 - 10.0).abs() < 1.0e-13);

    // Upstream `eosdt_eval.f90:1450-1456` scales `alfa` by `zfactor = 0.5`.
    let (az, drz, dtz) =
        get_opal_scvh_alfa_and_partials(&cfg, 7.55, 0.0, 0.0375)
            .expect("valid OPAL alfa at Z=0.0375");
    assert!((az - 0.25).abs() < 1.0e-14, "expected 0.25, got {az}");
    assert_eq!(drz, 0.0);
    assert!((dtz - 5.0).abs() < 1.0e-13, "expected 5.0, got {dtz}");

    // `BlendCornerOut` (`log_t = 7.55, log_rho = 3.49`) scales both partials.
    let (ac0, drc0, dtc0) =
        get_opal_scvh_alfa_and_partials(&cfg, 7.55, 3.49, 0.0)
            .expect("corner at Z=0");
    let (acz, drcz, dtcz) =
        get_opal_scvh_alfa_and_partials(&cfg, 7.55, 3.49, 0.0375)
            .expect("corner at Z=0.0375");
    assert!((acz - 0.5 * ac0).abs() < 1.0e-14);
    assert!((drcz - 0.5 * drc0).abs() < 1.0e-12);
    assert!((dtcz - 0.5 * dtc0).abs() < 1.0e-12);

    // Interior `UseAll` stays `0.0` for `Z < Z_all_HELM` and jumps at `0.04`.
    assert_eq!(
        get_opal_scvh_alfa_and_partials(&cfg, 5.0, 0.0, 0.036)
            .expect("interior Z=0.036"),
        (0.0, 0.0, 0.0),
    );
    assert_eq!(
        get_opal_scvh_alfa_and_partials(&cfg, 5.0, 0.0, 0.040)
            .expect("Z >= Z_all_HELM"),
        (1.0, 0.0, 0.0),
    );
}

#[test]
fn test_helmeos2_zbar_zero_no_ep_and_h_valid_zero_match_helm_f90() {
    let ht = HelmTable::builtin().expect("builtin HELM table");

    // 1. Standard `helmeos2` evaluation leaves `res[H_VALID - 1] == 0.0`
    //    (`eos/private/helm.f90:296`).
    let (res, off_table) = helmeos2(
        ht,
        2.0e8,
        log10(2.0e8),
        100.0,
        2.0,
        4.0,
        2.0,
        &HelmConfig::default(),
    )
    .expect("valid HELM state");
    assert!(!off_table);
    assert_eq!(res[H_VALID - 1], 0.0);
    assert!(res[H_PTOT - 1] > 0.0 && res[H_GAM1 - 1] > 1.0);

    // 2. Pure-neutron (`zbar = 0.0`) is rejected when `include_elec_pos = true`
    //    (`din < denlo`), but allowed when `include_elec_pos = false`.
    assert_eq!(
        helmeos2(ht, 1.0e7, 7.0, 100.0, 2.0, 1.0, 0.0, &HelmConfig::default()),
        Err(1),
    );
    let cfg_no_ep_no_rad = HelmConfig {
        include_elec_pos: false,
        include_radiation: false,
        ..HelmConfig::default()
    };
    let (res_n, off_n) =
        helmeos2(ht, 1.0e7, 7.0, 100.0, 2.0, 1.0, 0.0, &cfg_no_ep_no_rad)
            .expect("neutral gas with !include_elec_pos");
    assert!(!off_n);
    assert_eq!(res_n[H_VALID - 1], 0.0);
    assert!((res_n[H_GAM1 - 1] - 5.0 / 3.0).abs() < 1.0e-14);
    assert!((res_n[H_GAM3 - 1] - 5.0 / 3.0).abs() < 1.0e-14);
}

#[test]
fn test_eos_dt_and_eos_pt_reject_non_positive_or_non_finite_state() {
    let ht = HelmTable::builtin().expect("builtin HELM table");
    let cat = NuclideCatalog::builtin_ref();
    let chem_id = [
        cat.get_by_name("h1").expect("h1").chem_id,
        cat.get_by_name("he4").expect("he4").chem_id,
    ];
    let xa = [0.7, 0.3];
    let cfg = EosConfig::default();

    for (rho, t) in [
        (0.0, 1.0e5),
        (-1.0, 1.0e5),
        (1.0, 0.0),
        (1.0, -10.0),
        (f64::NAN, 1.0e5),
        (1.0, f64::INFINITY),
    ] {
        let err = eos_dt_get(ht, rho, t, &chem_id, &xa, &cfg)
            .expect_err("invalid eos_dt_get input must error");
        assert!(matches!(err, EosError::InvalidState { .. }));
    }

    for (pgas, t) in [
        (0.0, 1.0e4),
        (-1.0, 1.0e4),
        (1.0e4, 0.0),
        (1.0e4, -1.0),
        (f64::NAN, 1.0e4),
        (1.0e4, f64::INFINITY),
    ] {
        let err = eos_pt_get_with_linear(
            ht,
            &chem_id,
            &xa,
            4.0,
            4.0,
            Some(pgas),
            Some(t),
            &cfg,
        )
        .expect_err("invalid eos_pt_get_with_linear input must error");
        assert!(matches!(err, EosError::InvalidState { .. }));
    }
}
