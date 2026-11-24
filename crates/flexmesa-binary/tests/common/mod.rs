// SPDX-License-Identifier: GPL-3.0-or-later
//! Shared test fixtures for `flexmesa-binary` counter, determinism, and
//! child-process checkpoint integration tests.

use flexmesa_binary::{
    BinaryInlistControls, BinarySystem, DefaultBinaryExtras,
};
use flexmesa_const::SECYER;
use flexmesa_star::{StarControls, StarJobControls, load_zams_model};

/// MESA `26.4.1` results for the two fixtures below, from
/// `benchmarks/fortran_mesa/binary_rlof_fixture/results.json` (reference
/// header: version `r26.4.1`, `gfortran 15.2.0`, SDK `x86_64-linux-26.6.1`).
/// In both, MESA completes model 1 and never completes model 2 once
/// `fr_hard = 1e-5` is set: the donor's radius changes by more than that in
/// the once-per-step remesh, which no timestep cut can undo.
#[allow(dead_code)]
pub fn build_rlof_binary_with(
    m1_msun: f64,
    m2_msun: f64,
    initial_period_in_days: f64,
) -> BinarySystem {
    let donor_ctrl = StarControls {
        initial_mass: m1_msun,
        zbase: 0.02,
        ..StarControls::default()
    };
    let mut donor = load_zams_model(donor_ctrl, StarJobControls::default())
        .expect("donor ZAMS");
    donor.dt = 1.0e3 * SECYER;
    donor.dt_next = 1.0e3 * SECYER;

    let mut bctrl = BinaryInlistControls {
        m1_msun,
        m2_msun,
        initial_period_in_days,
        mdot_scheme: String::from("Ritter"),
        max_tries_to_achieve: 20,
        terminate_if_initial_overflow: false,
        ..BinaryInlistControls::default()
    };
    bctrl.use_radiation_corrected_transfer_rate = false;
    BinarySystem::from_controls_point_mass(donor, bctrl)
        .expect("RLOF point-mass binary")
}

/// The `(2.0 Msun, 1.5 Msun, P = 0.5045 d)` fixture. MESA `26.4.1` takes
/// model 1 at the full `1e3 yr` with no retries.
#[allow(dead_code)]
pub fn build_rlof_binary() -> BinarySystem {
    build_rlof_binary_with(2.0, 1.5, 0.5045)
}

/// The original `(1.5 Msun, 1.0 Msun, P = 0.45 d)` fixture. MESA `26.4.1`
/// cuts model 1 to `125 yr`.
#[allow(dead_code)]
pub fn build_rlof_binary_original() -> BinarySystem {
    build_rlof_binary_with(1.5, 1.0, 0.45)
}

/// Advance three steps with unmodified controls, each at the timestep the
/// previous step proposed. This sequence has no MESA counterpart on record;
/// it is used only by tests that compare `FlexMESA` with itself (thread
/// determinism and checkpoint restarts).
#[allow(dead_code)]
pub fn advance_binary_steps_one_to_three(sys: &mut BinarySystem) {
    let mut extras = DefaultBinaryExtras;
    let (next_dt1, _) = sys
        .try_step_with_extras(1.0e3 * SECYER, &mut extras)
        .expect("binary step 1");
    let (next_dt2, _) = sys
        .try_step_with_extras(next_dt1, &mut extras)
        .expect("binary step 2");
    let _ = sys
        .try_step_with_extras(next_dt2, &mut extras)
        .expect("binary step 3");
    assert_eq!(sys.model_number, 3);
}
