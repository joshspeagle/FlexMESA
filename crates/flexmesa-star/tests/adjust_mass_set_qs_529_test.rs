// SPDX-License-Identifier: GPL-3.0-or-later
//! `set_qs` in the `set_vars` that follows `do_adjust_mass` (`#529`).
//!
//! Upstream's step driver calls `set_vars_if_needed(s, dt, 'after
//! do_adjust_mass')` (`star/private/evolve.f90:593`); with `need_to_setvars`
//! set at `adjust_mass.f90:234` it runs `unpack_xh`, which rebuilds `q` with
//! `set_qs` and then `m`, `dm` and `dm_bar` (`hydro_vars.f90:359-365`,
//! `star_utils.f90:712-752`). `do_adjust_mass` itself returns `q` as
//! `revise_q_and_dq` built it (`adjust_mass.f90:812-821`).

#![allow(clippy::manual_midpoint)]

use flexmesa_const::{M_SUN, SECYER};
use flexmesa_star::{
    do_adjust_mass, do_adjust_mass_step, load_zams_model, set_hydro_vars,
    StarControls, StarJobControls, StarModel,
};

fn zams_model() -> StarModel {
    let ctrl = StarControls {
        initial_mass: 2.0,
        zbase: 0.02,
        initial_z: 0.02,
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(ctrl, StarJobControls::default()).expect("load zams");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    model.save_old_values();
    model
}

/// `q(1) = 1`, `q(k) = q(k-1) - dq(k-1)` (`set_qs`, `star_utils.f90:729-737`).
fn set_qs_reference(dq: &[f64]) -> Vec<f64> {
    let mut q = vec![1.0; dq.len()];
    for k in 1..dq.len() {
        q[k] = q[k - 1] - dq[k - 1];
    }
    q
}

#[test]
fn step_driver_post_adjust_set_vars_rebuilds_q_m_dm_from_set_qs() {
    let mut model = zams_model();
    model.mstar_dot = -1.0e-3 * M_SUN / SECYER;
    do_adjust_mass_step(&mut model, SECYER).expect("do_adjust_mass_step");
    let nz = model.nz;
    let q_ref = set_qs_reference(&model.dq[..nz]);
    for k in 0..nz {
        assert_eq!(model.q[k].to_bits(), q_ref[k].to_bits(), "q at k={k}");
        assert_eq!(
            model.m[k].to_bits(),
            (model.m_center + q_ref[k] * model.xmstar).to_bits(),
            "m at k={k}"
        );
        assert_eq!(
            model.dm[k].to_bits(),
            (model.dq[k] * model.xmstar).to_bits(),
            "dm at k={k}"
        );
    }
    assert_eq!(
        model.dm_bar[0].to_bits(),
        (0.5 * model.dm[0]).to_bits(),
        "dm_bar at the centre-most face"
    );
    for k in 1..nz - 1 {
        assert_eq!(
            model.dm_bar[k].to_bits(),
            (0.5 * (model.dm[k - 1] + model.dm[k])).to_bits(),
            "dm_bar at k={k}"
        );
    }
}

#[test]
fn do_adjust_mass_entry_returns_q_as_revise_q_and_dq_built_it() {
    // No `set_vars` runs inside `do_adjust_mass`: the innermost cell has
    // `q(nz) = dq(nz)` (`adjust_mass.f90:812-821`), not `set_qs`'s
    // `q(nz-1) - dq(nz-1)`.
    let mut model = zams_model();
    let delta_m = -1.0e-3 * M_SUN;
    do_adjust_mass(&mut model, delta_m, None, 0.99, 0.95)
        .expect("do_adjust_mass");
    let nz = model.nz;
    assert_eq!(model.q[nz - 1].to_bits(), model.dq[nz - 1].to_bits());
    let q_ref = set_qs_reference(&model.dq[..nz]);
    // `set_qs` would change `q` at rounding level, so this entry point has not
    // run it.
    let moved = (0..nz)
        .filter(|&k| model.q[k].to_bits() != q_ref[k].to_bits())
        .count();
    assert!(moved > 0, "q was rebuilt by set_qs inside do_adjust_mass");
}
