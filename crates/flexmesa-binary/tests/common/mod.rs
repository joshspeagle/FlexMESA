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
    // Use a 2.0 Msun ZAMS donor near Roche-lobe contact (m2 = 1.5 Msun,
    // P = 0.5045 d, q = 4/3) with default StarControls. Reproducible MESA
    // 26.4.1 verification in `benchmarks/fortran_mesa/binary_rlof_fixture/`
    // shows that the original unrelaxed (1.5 Msun, 1.0 Msun, P = 0.45 d)
    // fixture fails step 1's initial 1e3 yr solve (`logT >
    // hydro_mtx_max_allowed_logT`, 3 retries -> 125 yr) and enters a 24-solver-
    // retry rejection cascade on step 2 under `fr_hard = 1e-5` (terminating at
    // `max_number_retries = 40`), whereas the 2.0 Msun donor converges step 1
    // cleanly in MESA 26.4.1 with 0 retries and 2 solver iterations.
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
