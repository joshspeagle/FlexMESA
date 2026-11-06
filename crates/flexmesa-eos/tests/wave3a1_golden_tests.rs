// SPDX-License-Identifier: LGPL-3.0-or-later
//! Golden parity and physical verification tests for Wave 3a.1
//! (`flexmesa-eos`: `blend`, `gauss_fermi`, `ion_offset`, `free_energy`, and
//! `ideal`).

#![allow(
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::too_many_lines
)]

use flexmesa_autodiff::AutoDiffReal2VarOrder1;
use flexmesa_const::{AMU, CLIGHT, CRAD, EV2ERG, KERG, LN10, ME};
use flexmesa_eos::{
    blend_eos_results, compute_ion_offset, compute_ion_offset_from_za, dfermi,
    dfermi_order1, eval_ideal_eos, eval_ideal_eos_from_weights, is_contained,
    min_distance_to_polygon, quintic_blend, EosEvalOutput, I_CHI_RHO, I_CHI_T,
    I_CP, I_CV, I_DE_DRHO, I_DS_DRHO, I_DS_DT, I_FRAC_IDEAL, I_GAMMA1,
    I_GAMMA3, I_GRAD_AD, I_LN_E, I_LN_PGAS, I_LN_S, I_MU,
    NIST_IONIZATION_TABLE_EV, NUM_EOS_BASIC_RESULTS,
};

/// Verifies `is_contained` and `min_distance_to_polygon` against all 5 golden
/// outputs in `eos/test/test_output` (`do_test_eos_blend`, lines 465-474).
#[test]
fn test_eos_blend_matches_upstream_mesa_golden_output() {
    let coords = [[-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0]];

    // Case 1: p = (0, 1) -> T, d = 1.0000000133514320e-10, d1 = 0, d2 = 0
    // Note: in test_eos_blend.f90, p(1) = 0d0 and p(2) = 1d0 are assigned
    // constants (zero partials).
    let p1 = [
        AutoDiffReal2VarOrder1::constant(0.0),
        AutoDiffReal2VarOrder1::constant(1.0),
    ];
    assert!(is_contained(&coords, p1));
    let d1 = min_distance_to_polygon(&coords, p1);
    assert!((d1.val - 1.000_000_013_351_432e-10).abs() < 1.0e-24);
    assert_eq!(d1.d1[0], 0.0);
    assert_eq!(d1.d1[1], 0.0);

    // Case 2: p = (-1, 1) -> F, d = 1.0000000133514320e-10
    let p2 = [
        AutoDiffReal2VarOrder1::constant(-1.0),
        AutoDiffReal2VarOrder1::constant(1.0),
    ];
    assert!(!is_contained(&coords, p2));
    let d2 = min_distance_to_polygon(&coords, p2);
    assert!((d2.val - 1.000_000_013_351_432e-10).abs() < 1.0e-24);

    // Case 3: p = (0, 0) -> T, d = 1.0
    let p3 = [
        AutoDiffReal2VarOrder1::constant(0.0),
        AutoDiffReal2VarOrder1::constant(0.0),
    ];
    assert!(is_contained(&coords, p3));
    let d3 = min_distance_to_polygon(&coords, p3);
    assert!((d3.val - 1.0).abs() < 1.0e-15);

    // Case 4: p = (-2, 0) -> F, d = 1.0
    let p4 = [
        AutoDiffReal2VarOrder1::constant(-2.0),
        AutoDiffReal2VarOrder1::constant(0.0),
    ];
    assert!(!is_contained(&coords, p4));
    let d4 = min_distance_to_polygon(&coords, p4);
    assert!((d4.val - 1.0).abs() < 1.0e-15);

    // Case 5: p = (-1, 3) -> F, d = 2.0
    let p5 = [
        AutoDiffReal2VarOrder1::constant(-1.0),
        AutoDiffReal2VarOrder1::constant(3.0),
    ];
    assert!(!is_contained(&coords, p5));
    let d5 = min_distance_to_polygon(&coords, p5);
    assert!((d5.val - 2.0).abs() < 1.0e-15);

    // Also verify seeded first-order partial derivatives at p = (-2.0, 0.25)
    // (nearest edge is x = -1, so distance is -1 - x = 1.0, dd/dx = -1)
    let p_seeded = [
        AutoDiffReal2VarOrder1::independent(-2.0, 0),
        AutoDiffReal2VarOrder1::independent(0.25, 1),
    ];
    let d_seeded = min_distance_to_polygon(&coords, p_seeded);
    assert!((d_seeded.val - 1.0).abs() < 1.0e-15);
    assert!((d_seeded.d1[0] - (-1.0)).abs() < 1.0e-15);
    assert!(d_seeded.d1[1].abs() < 1.0e-15);
}

/// Verifies `dfermi` against the `gfortran` oracle for `gauss_fermi.f90`
/// (`eta = 1.46722890948893`, `T = 11327678.5183021`, `dk = -0.5` and `0.5`)
/// and checks analytical derivatives against central finite differences.
#[test]
fn test_dfermi_matches_upstream_mesa_dirac_integral_golden() {
    let eta = 1.467_228_909_488_93_f64;
    let t = 11_327_678.518_302_1_f64;
    let theta = (KERG * t) / (ME * CLIGHT * CLIGHT);

    let (fdmh, fdeta_mh, fdtheta_mh) = dfermi(-0.5, eta, theta);
    let (fdph, fdeta_ph, fdtheta_ph) = dfermi(0.5, eta, theta);
    let calc_ratio = fdmh / fdph;

    // Exact gfortran oracle values from eos/private/gauss_fermi.f90
    assert!((fdmh - 2.189_678_684_809_835_3).abs() < 1.0e-14);
    assert!((fdeta_mh - 7.819_242_981_028_8e-1).abs() < 1.0e-14);
    assert!((fdtheta_mh - 4.657_371_805_764_492_5e-1).abs() < 1.0e-14);
    assert!((fdph - 1.866_572_507_339_512_3).abs() < 1.0e-14);
    assert!((fdeta_ph - 1.095_729_024_204_351_1).abs() < 1.0e-14);
    assert!((fdtheta_ph - 9.485_028_387_068_805e-1).abs() < 1.0e-14);
    assert!((calc_ratio - 1.173_101_326_736_488_3).abs() < 1.0e-14);

    // Verify analytical derivatives fdeta and fdtheta across k = -0.5..2.5
    let h_eta = 1.0e-6;
    let h_theta = 1.0e-6 * theta;
    for &dk in &[-0.5, 0.5, 1.5, 2.5] {
        let (_, fdeta, fdtheta) = dfermi(dk, eta, theta);
        let (f_ep, _, _) = dfermi(dk, eta + h_eta, theta);
        let (f_em, _, _) = dfermi(dk, eta - h_eta, theta);
        let fd_eta_num = (f_ep - f_em) / (2.0 * h_eta);
        assert!(
            ((fdeta - fd_eta_num) / fdeta).abs() < 5.0e-10,
            "dk={dk}: fdeta={fdeta:e} vs num={fd_eta_num:e}"
        );

        let (f_tp, _, _) = dfermi(dk, eta, theta + h_theta);
        let (f_tm, _, _) = dfermi(dk, eta, theta - h_theta);
        let fd_theta_num = (f_tp - f_tm) / (2.0 * h_theta);
        assert!(
            ((fdtheta - fd_theta_num) / fdtheta).abs() < 5.0e-7,
            "dk={dk}: fdtheta={fdtheta:e} vs num={fd_theta_num:e}"
        );
    }

    // Verify dual-number wrapper dfermi_order1
    let eta_dual = AutoDiffReal2VarOrder1::independent(eta, 0);
    let theta_dual = AutoDiffReal2VarOrder1::independent(theta, 1);
    let fd_dual = dfermi_order1(0.5, eta_dual, theta_dual);
    assert!((fd_dual.val - fdph).abs() < 1.0e-15);
    assert!((fd_dual.d1[0] - fdeta_ph).abs() < 1.0e-15);
    assert!((fd_dual.d1[1] - fdtheta_ph).abs() < 1.0e-15);
    let _ = (fdeta_mh, fdtheta_mh);
}

/// Verifies `compute_ion_offset` and `compute_ion_offset_from_za` against
/// exact NIST atomic binding energies (`ion_offset.f90`).
#[test]
fn test_ion_offset_matches_nist_binding_table() {
    // Pure hydrogen (Z=1, A=1): 13.598434599702 eV / amu
    let h_offset = compute_ion_offset_from_za(&[1.0], &[1], &[1.0]);
    let expected_h = NIST_IONIZATION_TABLE_EV[0] * EV2ERG / AMU;
    assert!((h_offset - expected_h).abs() / expected_h < 1.0e-15);

    // Solar-like mixture (X=0.70 H1, Y=0.28 He4, Z=0.02 C12)
    let xa = [0.70, 0.28, 0.02];
    let z = [1_usize, 2, 6];
    let a = [1.0, 4.0, 12.0];
    let mix_offset = compute_ion_offset_from_za(&xa, &z, &a);
    let expected_mix = (0.70 * NIST_IONIZATION_TABLE_EV[0] / 1.0
        + 0.28 * NIST_IONIZATION_TABLE_EV[1] / 4.0
        + 0.02 * NIST_IONIZATION_TABLE_EV[5] / 12.0)
        * EV2ERG
        / AMU;
    assert!((mix_offset - expected_mix).abs() / expected_mix < 1.0e-15);

    // Verify catalog-backed compute_ion_offset with H1, He4, C12
    let catalog = flexmesa_chem::NuclideCatalog::builtin();
    let ih1 = catalog.get_by_name("h1").unwrap().chem_id;
    let ihe4 = catalog.get_by_name("he4").unwrap().chem_id;
    let ic12 = catalog.get_by_name("c12").unwrap().chem_id;
    let cat_offset = compute_ion_offset(&xa, &[ih1, ihe4, ic12]).unwrap();
    assert!((cat_offset - expected_mix).abs() / expected_mix < 1.0e-15);
}

/// Verifies `eval_ideal_eos` and `eval_ideal_eos_from_weights` against exact
/// closed-form neutral ideal-gas + radiation laws and Maxwell thermodynamic
/// identities.
#[test]
fn test_ideal_eos_satisfies_exact_laws_and_maxwell_relations() {
    let temp = 1.0e5_f64;
    let rho = 1.0e-2_f64;
    let xa = [0.70, 0.28, 0.02];
    let weights = [1.007_825_032_2, 4.002_603_254_1, 12.0];
    let a = [1.0, 4.0, 12.0];
    let inv_abar = xa[0] / weights[0] + xa[1] / weights[1] + xa[2] / weights[2];
    let abar = 1.0 / inv_abar;
    let zbar = abar
        * (xa[0] * 1.0 / weights[0]
            + xa[1] * 2.0 / weights[1]
            + xa[2] * 6.0 / weights[2]);

    let out =
        eval_ideal_eos_from_weights(temp, rho, abar, zbar, &xa, &weights, &a)
            .unwrap();

    // 1. Ideal ion gas pressure: P_gas = rho * k_B * T / (amu * abar)
    let pgas_exact = rho * KERG * temp / (AMU * abar);
    let pgas_eval = out.res[I_LN_PGAS].exp();
    assert!(
        ((pgas_eval - pgas_exact) / pgas_exact).abs() < 1.0e-14,
        "Pgas mismatch: {pgas_eval:e} vs {pgas_exact:e}"
    );
    assert!((out.d_dlnd[I_LN_PGAS] - 1.0).abs() < 1.0e-14);
    assert!((out.d_dln_t[I_LN_PGAS] - 1.0).abs() < 1.0e-14);

    // 2. Specific internal energy: E = 1.5 * k_B * T / (amu * abar) + a T^4/rho
    let egas_exact = 1.5 * KERG * temp / (AMU * abar);
    let erad_exact = CRAD * temp.powi(4) / rho;
    let etot_exact = egas_exact + erad_exact;
    let etot_eval = out.res[I_LN_E].exp();
    assert!(
        ((etot_eval - etot_exact) / etot_exact).abs() < 1.0e-14,
        "Etot mismatch: {etot_eval:e} vs {etot_exact:e}"
    );

    // 3. Neutral mu = abar and blend fraction ideal = 1.0
    assert!((out.res[I_MU] - abar).abs() < 1.0e-15);
    assert!((out.res[I_FRAC_IDEAL] - 1.0).abs() < 1.0e-15);

    // 4. Maxwell consistency relations (from sample_eos.f90 lines 327-329):
    //    dse = T * (dS/dT)_rho / (dE/dT)_rho - 1 = 0
    //    dpe = (rho^2 * (dE/drho)_T + T * (dP/dT)_rho) / P - 1 = 0
    //    dsp = -(dS/drho)_T * rho^2 / (dP/dT)_rho - 1 = 0
    let prad = CRAD * temp.powi(4) / 3.0;
    let ptot = pgas_eval + prad;
    let dpgas_dt = pgas_eval * out.d_dln_t[I_LN_PGAS] / temp;
    let dprad_dt = 4.0 * prad / temp;
    let dptot_dt = dpgas_dt + dprad_dt;

    let detot_dt = out.res[I_CV];
    let dstot_dt = out.res[I_DS_DT];
    let detot_dd = out.res[I_DE_DRHO];
    let dstot_dd = out.res[I_DS_DRHO];

    let dse = (temp * dstot_dt / detot_dt - 1.0).abs();
    let dpe = ((detot_dd * rho * rho + temp * dptot_dt) / ptot - 1.0).abs();
    let dsp = (-dstot_dd * (rho * rho / dptot_dt) - 1.0).abs();
    assert!(dse < 1.0e-14, "Maxwell dse = {dse:e}");
    assert!(dpe < 1.0e-14, "Maxwell dpe = {dpe:e}");
    assert!(dsp < 1.0e-14, "Maxwell dsp = {dsp:e}");

    // 5. Verify log-derivatives d_dlnd and d_dln_t against symmetric central
    //    differences in (ln T, ln rho)
    let eps = 1.0e-5_f64;
    let out_tp = eval_ideal_eos_from_weights(
        temp * eps.exp(),
        rho,
        abar,
        zbar,
        &xa,
        &weights,
        &a,
    )
    .unwrap();
    let out_tm = eval_ideal_eos_from_weights(
        temp * (-eps).exp(),
        rho,
        abar,
        zbar,
        &xa,
        &weights,
        &a,
    )
    .unwrap();
    let out_rp = eval_ideal_eos_from_weights(
        temp,
        rho * eps.exp(),
        abar,
        zbar,
        &xa,
        &weights,
        &a,
    )
    .unwrap();
    let out_rm = eval_ideal_eos_from_weights(
        temp,
        rho * (-eps).exp(),
        abar,
        zbar,
        &xa,
        &weights,
        &a,
    )
    .unwrap();
    let dln_step = 2.0 * eps;

    for &idx in &[
        I_LN_PGAS, I_LN_E, I_LN_S, I_GRAD_AD, I_CHI_RHO, I_CHI_T, I_CP, I_CV,
        I_DE_DRHO, I_DS_DT, I_DS_DRHO, I_GAMMA1, I_GAMMA3,
    ] {
        let fd_dlnt = (out_tp.res[idx] - out_tm.res[idx]) / dln_step;
        let fd_dlnd = (out_rp.res[idx] - out_rm.res[idx]) / dln_step;
        let scale_t = out.d_dln_t[idx]
            .abs()
            .max(out.res[idx].abs() * 1.0e-3)
            .max(1.0);
        let scale_d = out.d_dlnd[idx]
            .abs()
            .max(out.res[idx].abs() * 1.0e-3)
            .max(1.0);
        assert!(
            (out.d_dln_t[idx] - fd_dlnt).abs() / scale_t < 5.0e-6,
            "channel {idx} d_dln_t: {} vs {fd_dlnt}",
            out.d_dln_t[idx]
        );
        assert!(
            (out.d_dlnd[idx] - fd_dlnd).abs() / scale_d < 5.0e-6,
            "channel {idx} d_dlnd: {} vs {fd_dlnd}",
            out.d_dlnd[idx]
        );
    }

    // Exact 50-digit analytical oracles for dCp/dlnT and d(dE/drho)/dlnRho
    let exact_dcp_dlnt = 15_131.562_343_485_932_f64;
    let exact_dde_drho_dlnd = 2.0 * CRAD * temp.powi(4) / (rho * rho);
    assert!(
        ((out.d_dln_t[I_CP] - exact_dcp_dlnt) / exact_dcp_dlnt).abs() < 1.0e-10
    );
    assert!(
        ((out.d_dlnd[I_DE_DRHO] - exact_dde_drho_dlnd) / exact_dde_drho_dlnd)
            .abs()
            < 1.0e-10
    );

    // Also check catalog-based eval_ideal_eos
    let catalog = flexmesa_chem::NuclideCatalog::builtin();
    let chem_ids = [
        catalog.get_by_name("h1").unwrap().chem_id,
        catalog.get_by_name("he4").unwrap().chem_id,
        catalog.get_by_name("c12").unwrap().chem_id,
    ];
    let out_cat =
        eval_ideal_eos(temp, rho, abar, zbar, &xa, &chem_ids).unwrap();
    assert!((out_cat.res[I_LN_PGAS] - out.res[I_LN_PGAS]).abs() < 1.0e-12);
}

/// Verifies `quintic_blend` and `blend_eos_results` against `Do_Blend` in
/// `eos/private/eosdt_support.f90`.
#[test]
fn test_quintic_blend_and_do_blend_chain_rule() {
    assert_eq!(quintic_blend(0.0), (0.0, 0.0));
    assert_eq!(quintic_blend(1.0), (1.0, 0.0));
    let (s_half, ds_half) = quintic_blend(0.5);
    assert!((s_half - 0.5).abs() < 1.0e-15);
    assert!((ds_half - 1.875).abs() < 1.0e-15);

    let mut r1 = EosEvalOutput::zeros(2);
    let mut r2 = EosEvalOutput::zeros(2);
    for j in 0..NUM_EOS_BASIC_RESULTS {
        r1.res[j] = 10.0 + (j as f64);
        r1.d_dlnd[j] = 1.5;
        r1.d_dln_t[j] = 2.5;
        r2.res[j] = 20.0 + 2.0 * (j as f64);
        r2.d_dlnd[j] = 0.5;
        r2.d_dln_t[j] = 3.5;
    }

    let alpha0 = 0.35;
    let d_alpha_dlog_t = 0.8;
    let d_alpha_dlog_rho = -0.4;
    let blended = blend_eos_results(
        alpha0,
        d_alpha_dlog_t,
        d_alpha_dlog_rho,
        false,
        &r1,
        &r2,
    );

    let (s, ds) = quintic_blend(alpha0);
    let d_alpha_dlnt = ds * d_alpha_dlog_t / LN10;
    let d_alpha_dlnd = ds * d_alpha_dlog_rho / LN10;
    for j in 0..NUM_EOS_BASIC_RESULTS {
        let exp_res = s * r1.res[j] + (1.0 - s) * r2.res[j];
        let exp_dlnd = s * r1.d_dlnd[j]
            + (1.0 - s) * r2.d_dlnd[j]
            + d_alpha_dlnd * (r1.res[j] - r2.res[j]);
        let exp_dlnt = s * r1.d_dln_t[j]
            + (1.0 - s) * r2.d_dln_t[j]
            + d_alpha_dlnt * (r1.res[j] - r2.res[j]);
        assert!((blended.res[j] - exp_res).abs() < 1.0e-14);
        assert!((blended.d_dlnd[j] - exp_dlnd).abs() < 1.0e-14);
        assert!((blended.d_dln_t[j] - exp_dlnt).abs() < 1.0e-14);
    }
}
