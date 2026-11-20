// SPDX-License-Identifier: LGPL-3.0-or-later
//! Shared test fixtures for `flexmesa-binary` counter, determinism, and
//! child-process checkpoint integration tests.

use flexmesa_binary::{
    BinaryInlistControls, BinarySystem, DefaultBinaryExtras,
};
use flexmesa_const::SECYER;
use flexmesa_star::{StarControls, StarJobControls, load_zams_model};

#[allow(dead_code)]
pub fn build_rlof_binary() -> BinarySystem {
    // Use a 2.0 Msun ZAMS donor (radiative envelope and well-developed
    // convective core, avoiding the unrelaxed 1.5 Msun ZAMS convective-core
    // Newton limit cycle in MESA 26.4.1 under step 2's artificial fr_hard = 1e-5)
    // near Roche-lobe contact (P = 0.5045 d, q = 4/3) with default StarControls
    // so the fixture tests binary RLOF, rollback counter, determinism, and
    // checkpoint mechanics cleanly.
    let donor_ctrl = StarControls {
        initial_mass: 2.0,
        zbase: 0.02,
        ..StarControls::default()
    };
    let mut donor = load_zams_model(donor_ctrl, StarJobControls::default())
        .expect("2.0 Msun donor ZAMS");
    donor.dt = 1.0e3 * SECYER;
    donor.dt_next = 1.0e3 * SECYER;

    let mut bctrl = BinaryInlistControls {
        m1_msun: 2.0,
        m2_msun: 1.5,
        initial_period_in_days: 0.5045,
        mdot_scheme: String::from("Ritter"),
        max_tries_to_achieve: 20,
        terminate_if_initial_overflow: false,
        ..BinaryInlistControls::default()
    };
    bctrl.use_radiation_corrected_transfer_rate = false;
    BinarySystem::from_controls_point_mass(donor, bctrl)
        .expect("RLOF point-mass binary")
}

#[allow(dead_code)]
pub fn advance_binary_steps_one_to_three(sys: &mut BinarySystem) {
    let mut extras = DefaultBinaryExtras;
    let (next_dt1, _) = sys
        .try_step_with_extras(1.0e3 * SECYER, &mut extras)
        .expect("binary step 1");
    sys.controls.fr_hard = 1.0e-5;
    let (next_dt2, _) = sys
        .try_step_with_extras(next_dt1.max(1.0e4 * SECYER), &mut extras)
        .expect("binary step 2");
    let _ = sys
        .try_step_with_extras(next_dt2, &mut extras)
        .expect("binary step 3");
    assert_eq!(sys.model_number, 3);
}
