// SPDX-License-Identifier: GPL-3.0-or-later
//! `have_mlt_vc` / `okay_to_set_mlt_vc` after an accepted step
//! (`star/private/evolve.f90:694-701`): `set_final_vars` is followed by
//! `if (okay_to_set_mlt_vc .and. .not. have_mlt_vc) have_mlt_vc = .true.`
//! and `okay_to_set_mlt_vc = .false.`, so `mlt_vc` keeps its
//! `set_final_vars` value until the next step's `set_current_to_old` /
//! `new_generation` (`star/private/evolve_support.f90:193-195`), even when
//! `set_hydro_vars` runs between steps at a changed state (a relax
//! `check_model` that moves `lnR`, `star/private/relax.f90:2178-2192`).

#![forbid(unsafe_code)]

use flexmesa_star::evolve::star_evolve_step;
use flexmesa_star::{
    load_zams_model, set_hydro_vars, StarControls, StarJobControls, StarModel,
};

fn stepped_4msun_zams() -> StarModel {
    let controls = StarControls {
        initial_mass: 4.0,
        initial_z: 0.02,
        zbase: 0.02,
        ..StarControls::default()
    };
    let mut model = load_zams_model(controls, StarJobControls::default())
        .expect("4 Msun ZAMS");
    set_hydro_vars(&mut model, false, false).expect("init hydro");
    model.aux.have_mlt_vc = false;
    star_evolve_step(&mut model, false, false, true).expect("step 1");
    model
}

/// Moves every `lnR` by a relative `1e-4` and refreshes the packed mesh
/// variables, as a post-step state edit would.
fn shift_radii(model: &mut StarModel) {
    for ln_r in &mut model.ln_r {
        *ln_r += 1.0e-4;
    }
    model.sync_mesh_and_pack().expect("sync mesh");
}

#[test]
fn accepted_step_clears_okay_to_set_mlt_vc_and_sets_have_mlt_vc_from_false() {
    let model = stepped_4msun_zams();
    assert!(
        model.aux.have_mlt_vc,
        "have_mlt_vc is set after set_final_vars when okay_to_set_mlt_vc was \
         true (evolve.f90:697-699)"
    );
    assert!(
        !model.aux.okay_to_set_mlt_vc,
        "okay_to_set_mlt_vc is false after the step (evolve.f90:700)"
    );
}

#[test]
fn set_hydro_vars_after_accepted_step_at_shifted_radii_leaves_mlt_vc_unchanged()
{
    let mut model = stepped_4msun_zams();
    let vc_final = model.mlt_vc.clone();
    assert!(
        vc_final.iter().any(|&v| v > 0.0),
        "4 Msun ZAMS has a convective core"
    );

    // Control: with the flag forced back on, the same edit does change
    // `mlt_vc`, so the invariance below is not vacuous.
    let mut control = model.clone();
    control.aux.okay_to_set_mlt_vc = true;
    shift_radii(&mut control);
    set_hydro_vars(&mut control, false, false).expect("control hydro");
    assert_ne!(
        control
            .mlt_vc
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        vc_final.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "recomputing MLT at shifted radii changes mlt_vc"
    );

    shift_radii(&mut model);
    set_hydro_vars(&mut model, false, false).expect("post-step hydro");
    assert_eq!(
        model.mlt_vc.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        vc_final.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "mlt_vc keeps its set_final_vars value between steps"
    );
}
