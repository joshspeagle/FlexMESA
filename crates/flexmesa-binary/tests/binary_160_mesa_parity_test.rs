// SPDX-License-Identifier: GPL-3.0-or-later
//! MESA `26.4.1` coupled parity pins for `#160` (`jdot_gr_check`,
//! `star_plus_point_mass`, and `star_plus_point_mass_explicit_mdot`).

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use flexmesa_binary::{
    binary_history_row_as_f64_pairs, default_binary_history_specs,
    evaluate_binary_history_row, BinaryInlistControls, BinarySystem,
    DefaultBinaryExtras,
};
use flexmesa_const::SECYER;
use flexmesa_math::log10;
use flexmesa_star::{
    eval_photosphere_info, load_zams_model, StarControls, StarJobControls,
};

fn build_jdot_gr_check_system() -> BinarySystem {
    let d_ctrl = StarControls {
        initial_mass: 0.5,
        zbase: 0.02,
        max_age: 1.4e10,
        use_gold2_tolerances: true,
        ..StarControls::default()
    };
    let donor = load_zams_model(d_ctrl, StarJobControls::default())
        .expect("0.5 Msun ZAMS");
    let mut bctrl = BinaryInlistControls {
        m1_msun: 0.5,
        m2_msun: 0.8,
        initial_period_in_days: -1.0,
        initial_separation_in_rsuns: 2.0,
        mdot_scheme: String::from("roche_lobe"),
        max_tries_to_achieve: 20,
        do_jdot_mb: false,
        fj: 0.001,
        ..BinaryInlistControls::default()
    };
    bctrl.use_radiation_corrected_transfer_rate = false;
    BinarySystem::from_controls_point_mass(donor, bctrl)
        .expect("jdot_gr_check BinarySystem")
}

fn build_star_plus_point_mass_system(
    max_tries_to_achieve: i64,
) -> BinarySystem {
    let d_ctrl = StarControls {
        initial_mass: 5.0,
        zbase: 0.02,
        max_model_number: Some(1000),
        max_number_retries: 80,
        use_gold2_tolerances: true,
        varcontrol_target: 1.0e-3,
        use_ledoux_criterion: true,
        alpha_semiconvection: 1.0,
        ..StarControls::default()
    };
    let donor = load_zams_model(d_ctrl, StarJobControls::default())
        .expect("5.0 Msun ZAMS");
    let mut bctrl = BinaryInlistControls {
        m1_msun: 5.0,
        m2_msun: 10.0,
        initial_period_in_days: 1.0,
        fr: 0.05,
        fr_limit: 1.0e-2,
        limit_retention_by_mdot_edd: true,
        max_tries_to_achieve,
        ..BinaryInlistControls::default()
    };
    bctrl.use_radiation_corrected_transfer_rate = false;
    BinarySystem::from_controls_point_mass(donor, bctrl)
        .expect("star_plus_point_mass BinarySystem")
}

fn advance_to_model(sys: &mut BinarySystem, target_model: usize) {
    let mut extras = DefaultBinaryExtras;
    let mut dt = sys.donor.dt_next;
    while sys.model_number < target_model {
        let (next_dt, _) = sys
            .try_step_with_extras(dt, &mut extras)
            .expect("binary step succeeds");
        dt = next_dt;
    }
    assert_eq!(sys.model_number, target_model);
}

fn row_map(sys: &BinarySystem) -> BTreeMap<String, f64> {
    let specs = default_binary_history_specs();
    let row = evaluate_binary_history_row(sys, &specs);
    binary_history_row_as_f64_pairs(&row).into_iter().collect()
}

fn assert_rel_close(actual: f64, expected: f64, rel_tol: f64, label: &str) {
    let denom = expected.abs().max(1.0e-99);
    let rel = (actual - expected).abs() / denom;
    assert!(
        rel <= rel_tol,
        "{label}: actual={actual:.16e}, expected={expected:.16e}, \
         rel={rel:.3e} > tol={rel_tol:.3e}"
    );
}

fn assert_abs_close(actual: f64, expected: f64, abs_tol: f64, label: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= abs_tol,
        "{label}: actual={actual:.16e}, expected={expected:.16e}, \
         diff={diff:.3e} > tol={abs_tol:.3e}"
    );
}

#[test]
fn jdot_gr_check_initial_dt_and_model5_match_mesa_26_4_1() {
    let mut sys = build_jdot_gr_check_system();
    // MESA `26.4.1` `star/private/star_utils.f90:2900-2907` &
    // `jdot_gr_check.stepdump` model-1 CALL (`0x4286F60FA1600000`).
    assert_eq!(sys.donor.dt_next, 1.0e5 * SECYER);
    assert_eq!(sys.donor.dt_next.to_bits(), 0x4286_F60F_A160_0000);

    advance_to_model(&mut sys, 5);
    assert_eq!(sys.donor.nz, 790);

    let cols = row_map(&sys);
    assert_rel_close(cols["age"], 7.4416e5, 1.0e-12, "jdot_gr m5 age");
    assert_rel_close(
        cols["period_days"],
        2.873_362_413_960_604_5e-1,
        1.0e-11,
        "jdot_gr m5 period_days",
    );
    assert_rel_close(
        cols["binary_separation"],
        1.999_919_488_726_777_7,
        1.0e-11,
        "jdot_gr m5 binary_separation",
    );
    assert_rel_close(
        cols["rl_1"],
        6.780_581_886_470_01e-1,
        1.0e-11,
        "jdot_gr m5 rl_1",
    );
    assert_rel_close(
        cols["rl_2"],
        8.403_047_374_065_568e-1,
        1.0e-11,
        "jdot_gr m5 rl_2",
    );
    assert_abs_close(
        cols["rl_relative_overflow_1"],
        -3.430_032_941_570_881_3e-1,
        1.0e-10,
        "jdot_gr m5 rl_relative_overflow_1",
    );
    assert_rel_close(
        cols["J_orb"],
        2.997_555_966_815_422_6e51,
        1.0e-11,
        "jdot_gr m5 J_orb",
    );
    assert_rel_close(
        cols["jdot_gr"],
        -2.569_371_427_414_080_7e33,
        1.0e-10,
        "jdot_gr m5 jdot_gr",
    );

    let phot = eval_photosphere_info(&sys.donor);
    assert_abs_close(
        log10(phot.r_rsun),
        -3.511_698_428_298_629e-1,
        1.0e-8,
        "jdot_gr m5 log_R",
    );
    assert_abs_close(
        log10(phot.l_lsun),
        -1.404_470_030_416_498_3,
        1.0e-8,
        "jdot_gr m5 log_L",
    );
    assert_abs_close(
        log10(sys.donor.teff),
        3.585_793_994_243_055_2,
        1.0e-8,
        "jdot_gr m5 log_Teff",
    );
}

#[test]
fn star_plus_point_mass_model5_and_model115_match_mesa_26_4_1() {
    let mut sys = build_star_plus_point_mass_system(20);
    advance_to_model(&mut sys, 5);
    let m5 = row_map(&sys);
    assert_rel_close(
        m5["age"],
        1.331_193_876_908_994_6e4,
        1.0e-11,
        "spm m5 age",
    );
    assert_rel_close(
        m5["period_days"],
        9.999_978_516_146_46e-1,
        1.0e-11,
        "spm m5 period_days",
    );
    assert_rel_close(
        m5["J_orb"],
        2.512_836_497_924_668e53,
        1.0e-11,
        "spm m5 J_orb",
    );
    assert_rel_close(
        m5["jdot_gr"],
        -4.283_622_744_690_115e35,
        1.0e-11,
        "spm m5 jdot_gr",
    );

    // Model 115 is 35 steps into active implicit RLOF.
    advance_to_model(&mut sys, 115);
    let m115 = row_map(&sys);
    assert_rel_close(
        m115["age"],
        3.838_849_143_529_478_5e7,
        1.0e-9,
        "spm m115 age",
    );
    assert_rel_close(
        m115["period_days"],
        9.999_347_222_446_818e-1,
        1.0e-9,
        "spm m115 period_days",
    );
    assert_rel_close(
        m115["binary_separation"],
        1.037_805_520_662_648_3e1,
        1.0e-9,
        "spm m115 binary_separation",
    );
    assert_rel_close(
        m115["star_1_mass"],
        4.979_505_257_805_243,
        1.0e-9,
        "spm m115 star_1_mass",
    );
    assert_rel_close(
        m115["star_2_mass"],
        1.002_049_474_219_476e1,
        1.0e-9,
        "spm m115 star_2_mass",
    );
    assert_abs_close(
        m115["rl_relative_overflow_1"],
        1.810_766_337_720_462_4e-3,
        1.0e-9,
        "spm m115 rl_relative_overflow_1",
    );
    assert_abs_close(
        m115["lg_mtransfer_rate"],
        -7.488_485_785_440_727,
        1.0e-9,
        "spm m115 lg_mtransfer_rate",
    );
    assert_rel_close(
        m115["J_orb"],
        2.507_612_624_410_114e53,
        1.0e-9,
        "spm m115 J_orb",
    );
}

#[test]
fn star_plus_point_mass_explicit_mdot_model5_and_model110_match_mesa_26_4_1() {
    let mut sys = build_star_plus_point_mass_system(0);
    advance_to_model(&mut sys, 5);
    let m5 = row_map(&sys);
    assert_rel_close(
        m5["age"],
        1.331_193_876_908_994_6e4,
        1.0e-11,
        "exp m5 age",
    );
    assert_rel_close(
        m5["period_days"],
        9.999_978_516_146_46e-1,
        1.0e-11,
        "exp m5 period_days",
    );

    // Model 110 is 30 steps into explicit Ritter RLOF with `cur_mdot_frac = 0.5`.
    advance_to_model(&mut sys, 110);
    assert_eq!(sys.donor.nz, 947);
    let m110 = row_map(&sys);
    assert_rel_close(
        m110["age"],
        3.767_193_250_351_380_6e7,
        1.0e-9,
        "exp m110 age",
    );
    assert_rel_close(
        m110["period_days"],
        9.941_087_242_937_153e-1,
        1.0e-9,
        "exp m110 period_days",
    );
    assert_rel_close(
        m110["binary_separation"],
        1.033_770_497_675_283_2e1,
        1.0e-9,
        "exp m110 binary_separation",
    );
    assert_rel_close(
        m110["star_1_mass"],
        4.999_267_169_440_302,
        1.0e-9,
        "exp m110 star_1_mass",
    );
    assert_rel_close(
        m110["star_2_mass"],
        1.000_073_283_055_97e1,
        1.0e-9,
        "exp m110 star_2_mass",
    );
    assert_abs_close(
        m110["rl_relative_overflow_1"],
        9.759_693_607_434_231e-4,
        1.0e-9,
        "exp m110 rl_relative_overflow_1",
    );
    assert_abs_close(
        m110["lg_mtransfer_rate"],
        -8.351_605_960_023_617,
        1.0e-9,
        "exp m110 lg_mtransfer_rate",
    );
    assert_rel_close(
        m110["J_orb"],
        2.507_710_154_412_364_7e53,
        1.0e-9,
        "exp m110 J_orb",
    );
}
