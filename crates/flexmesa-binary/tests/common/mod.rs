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
    // Unrelaxed 1.5 Msun ZAMS at step 2's artificial fr_hard = 1e-5 enters a
    // 2-cycle convective-core Newton limit cycle in MESA 26.4.1 (zone 2227 at
    // dt = 1.505e5 s); enable upstream make_gradr_sticky_in_solver_iters
    // (controls.defaults:2450) so the RLOF fixture tests binary mechanics
    // rather than that unrelaxed ZAMS oscillation.
    let mut donor_ctrl = StarControls {
        initial_mass: 1.5,
        zbase: 0.02,
        ..StarControls::default()
    };
    donor_ctrl.mix_wind.make_gradr_sticky_in_solver_iters = true;
    let mut donor = load_zams_model(donor_ctrl, StarJobControls::default())
        .expect("1.5 Msun donor ZAMS");
    donor.dt = 1.0e3 * SECYER;
    donor.dt_next = 1.0e3 * SECYER;

    let mut bctrl = BinaryInlistControls {
        m1_msun: 1.5,
        m2_msun: 1.0,
        initial_period_in_days: 0.45,
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
