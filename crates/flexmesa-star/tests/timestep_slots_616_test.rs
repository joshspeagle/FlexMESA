// SPDX-License-Identifier: GPL-3.0-or-later
//! Invariant tests for the `dt_limit_ratio(Tlim_*)` slots of
//! `do_timestep_limits` (`star/private/timestep.f90:128-332`, issue #616).
//!
//! The expected slot indices come from `star_data/public/star_data_def.inc`
//! (MESA `26.4.1`), including the index collisions recorded in issue #617.
//! These are checks of the slot and tie-break rules against closed-form
//! ratios, not comparisons with MESA output.

#![forbid(unsafe_code)]
#![allow(clippy::float_cmp)]

use flexmesa_star::{
    load_zams_model, set_hydro_vars, timestep_controller, StarControls,
    StarJobControls, StarModel, TimestepAction,
};

// Upstream `Tlim_*` values (1-based).
const TLIM_MAX_TIMESTEP_FACTOR: usize = 2;
const TLIM_LG_XH_CNTR: usize = 26;
const TLIM_LG_XNE_CNTR: usize = 37;
const TLIM_LG_XO_CNTR: usize = 38;
const TLIM_XHE_CNTR: usize = 39;
const TLIM_XC_CNTR: usize = 40;

/// ZAMS model with every center-abundance limit switched off and the
/// isotope `mg24` renamed `si28` so that all six `lg_X*_cntr` checks exist.
fn base_model() -> StarModel {
    let controls = StarControls {
        initial_mass: 1.0,
        initial_z: 0.02,
        zbase: 0.02,
        ..StarControls::default()
    };
    let mut m =
        load_zams_model(controls, StarJobControls::default()).expect("zams");
    set_hydro_vars(&mut m, false, false).expect("set_hydro_vars");
    m.save_start_values().expect("save_start_values");
    m.save_old_values();
    m.doing_first_model_of_run = false;
    let last = m.iso_names.len() - 1;
    assert_eq!(m.iso_names[last], "mg24");
    m.iso_names[last] = String::from("si28");
    let ts = &mut m.controls.ts_stop;
    ts.delta_dx_div_x_cntr_limit = -1.0;
    ts.delta_lg_xh_cntr_limit = -1.0;
    ts.delta_lg_xhe_cntr_limit = -1.0;
    ts.delta_lg_xc_cntr_limit = -1.0;
    ts.delta_lg_xne_cntr_limit = -1.0;
    ts.delta_lg_xo_cntr_limit = -1.0;
    ts.delta_lg_xsi_cntr_limit = -1.0;
    ts.delta_xh_cntr_limit = -1.0;
    ts.delta_xhe_cntr_limit = -1.0;
    ts.delta_xc_cntr_limit = -1.0;
    ts.delta_xne_cntr_limit = -1.0;
    ts.delta_xo_cntr_limit = -1.0;
    ts.delta_xsi_cntr_limit = -1.0;
    m
}

/// Sets the center mass fraction of `name` (current and previous step).
fn set_center(m: &mut StarModel, name: &str, x: f64, x_old: f64) {
    let j = m
        .iso_names
        .iter()
        .position(|n| n == name)
        .expect("isotope present");
    let off = (m.nz - 1) * m.species + j;
    m.xa[off] = x;
    m.xa_old[off] = x_old;
}

/// Center with only `ne20` (and `si28`) above the tiny `h1`, `he4`, `c12`.
fn neon_silicon_center(m: &mut StarModel) {
    for iso in ["h1", "he4", "c12"] {
        set_center(m, iso, 1.0e-12, 1.0e-12);
    }
    set_center(m, "ne20", 0.3, 0.1);
    set_center(m, "si28", 0.2, 0.2);
    let ts = &mut m.controls.ts_stop;
    for (min, max) in [
        (&mut ts.delta_lg_xne_cntr_min, &mut ts.delta_lg_xne_cntr_max),
        (&mut ts.delta_lg_xsi_cntr_min, &mut ts.delta_lg_xsi_cntr_max),
    ] {
        *min = -99.0;
        *max = 99.0;
    }
    ts.delta_lg_xne_cntr_limit = 0.125;
    ts.delta_lg_xne_cntr_hard_limit = -1.0;
    ts.delta_lg_xsi_cntr_hard_limit = -1.0;
}

#[test]
fn test_silicon_check_assigns_neon_slot_so_zero_silicon_ratio_hides_neon_limit()
{
    // Upstream `Tlim_lg_XSi_cntr = Tlim_lg_XC_cntr + 1 == Tlim_lg_XNe_cntr`
    // (`star_data_def.inc`; issue #617), and `check_change` assigns the slot
    // whenever the soft limit is positive. A silicon change of zero therefore
    // overwrites the neon ratio with 0.
    let mut m = base_model();
    neon_silicon_center(&mut m);
    m.controls.ts_stop.delta_lg_xsi_cntr_limit = 0.5;
    let (act, rep) = timestep_controller(&mut m);
    assert_eq!(act, TimestepAction::KeepGoing);
    assert_eq!(rep.dt_limit_ratio, 0.0);
    // All slots are 0, so `maxloc` picks `Tlim_struc` and the `max_timestep_factor`
    // clamp relabels it (`timestep.f90:337-347`).
    assert_eq!(rep.why_tlim, TLIM_MAX_TIMESTEP_FACTOR);
}

#[test]
fn test_neon_limit_controls_when_silicon_check_leaves_shared_slot_unassigned() {
    // With the silicon soft limit off, `check_change` returns before it
    // assigns the slot, so the neon ratio survives and wins the `maxloc`.
    let mut m = base_model();
    neon_silicon_center(&mut m);
    m.controls.ts_stop.delta_lg_xsi_cntr_limit = -1.0;
    let (act, rep) = timestep_controller(&mut m);
    assert_eq!(act, TimestepAction::KeepGoing);
    let expected = flexmesa_math::log10(0.3 / 0.1) / 0.125;
    assert!(
        (rep.dt_limit_ratio - expected).abs() <= 1.0e-14 * expected,
        "expected {expected:.16e}, got {:.16e}",
        rep.dt_limit_ratio
    );
    assert_eq!(rep.why_tlim, TLIM_LG_XNE_CNTR);
}

/// Center with `o16` dominating and rising, `h1`, `he4`, `c12` tiny.
fn oxygen_center(m: &mut StarModel) {
    for iso in ["h1", "he4", "c12"] {
        set_center(m, iso, 1.0e-12, 1.0e-12);
    }
    set_center(m, "o16", 0.5, 0.1);
    let ts = &mut m.controls.ts_stop;
    ts.delta_lg_xo_cntr_min = -99.0;
    ts.delta_lg_xo_cntr_max = 99.0;
    ts.delta_lg_xo_cntr_limit = 0.25;
    ts.delta_lg_xo_cntr_hard_limit = -1.0;
    ts.delta_xh_cntr_hard_limit = -1.0;
}

#[test]
fn test_hydrogen_center_check_assigns_oxygen_slot_when_its_soft_limit_is_on() {
    // `Tlim_XH_cntr = Tlim_lg_XSi_cntr + 1 == Tlim_lg_XO_cntr`: an unchanged
    // central hydrogen with a positive soft limit writes 0 over the oxygen
    // ratio.
    let mut m = base_model();
    oxygen_center(&mut m);
    m.controls.ts_stop.delta_xh_cntr_limit = 0.5;
    let (_, rep) = timestep_controller(&mut m);
    assert_eq!(rep.dt_limit_ratio, 0.0);
    // All slots are 0, so `maxloc` picks `Tlim_struc` and the `max_timestep_factor`
    // clamp relabels it (`timestep.f90:337-347`).
    assert_eq!(rep.why_tlim, TLIM_MAX_TIMESTEP_FACTOR);

    m.controls.ts_stop.delta_xh_cntr_limit = -1.0;
    let (_, rep) = timestep_controller(&mut m);
    let expected = flexmesa_math::log10(0.5 / 0.1) / 0.25;
    assert!((rep.dt_limit_ratio - expected).abs() <= 1.0e-14 * expected);
    assert_eq!(rep.why_tlim, TLIM_LG_XO_CNTR);
}

#[test]
fn test_equal_ratios_select_the_lowest_tlim_index() {
    // `maxloc` returns the first maximum. XHe (39) and XC (40) both give the
    // exact ratio 0.25 / 0.125 = 2; so do lg_XH (26) and XH (38) below.
    let mut m = base_model();
    set_center(&mut m, "he4", 0.25, 0.5);
    set_center(&mut m, "c12", 0.25, 0.5);
    let ts = &mut m.controls.ts_stop;
    ts.delta_xhe_cntr_limit = 0.125;
    ts.delta_xc_cntr_limit = 0.125;
    ts.delta_xhe_cntr_hard_limit = -1.0;
    ts.delta_xc_cntr_hard_limit = -1.0;
    let (_, rep) = timestep_controller(&mut m);
    assert_eq!(rep.dt_limit_ratio, 2.0);
    assert_eq!(rep.why_tlim, TLIM_XHE_CNTR);
    const { assert!(TLIM_XHE_CNTR < TLIM_XC_CNTR) };

    // Slot 26 (`lg_XH_cntr`) precedes slot 38 (`XH_cntr`): lg change
    // log10(1/2) per limit log10(2)/2 -> 2, and X change 0.25 per 0.125.
    let mut m = base_model();
    set_center(&mut m, "h1", 0.25, 0.5);
    let ts = &mut m.controls.ts_stop;
    ts.delta_xh_cntr_limit = 0.125;
    ts.delta_xh_cntr_hard_limit = -1.0;
    ts.delta_lg_xh_cntr_min = -99.0;
    ts.delta_lg_xh_cntr_max = 99.0;
    ts.delta_lg_xh_cntr_limit = flexmesa_math::log10(2.0) / 2.0;
    ts.delta_lg_xh_cntr_hard_limit = -1.0;
    let (_, rep) = timestep_controller(&mut m);
    assert!((rep.dt_limit_ratio - 2.0).abs() <= 1.0e-14);
    assert_eq!(rep.why_tlim, TLIM_LG_XH_CNTR);
}
