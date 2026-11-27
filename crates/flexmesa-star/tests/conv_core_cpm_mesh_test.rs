// SPDX-License-Identifier: GPL-3.0-or-later
//! Invariant tests for Issue #444 covering MESA `26.4.1` logic in
//! convective premixing (`conv_premix`), Ledoux composition-gradient smoothing
//! (`mixing_state`), burning-gradient mesh refinement (`adjust_mesh_support`),
//! and Lagrangian remesh mass conservation (`mesh_adjust` / `mesh_remap`).

#![forbid(unsafe_code)]
#![allow(clippy::too_many_lines)]

use flexmesa_const::{LN10, M_SUN, SECYER};
use flexmesa_star::{
    conv_premix::do_conv_premix as run_conv_premix, do_adjust_mesh,
    get_gval_info, load_zams_model, locate_convection_boundaries,
    mixing_state::set_grads, num_mesh_functions,
    premix::do_conv_premix as run_premix_wrapper, set_hydro_vars,
    try_plan_standard_mesh, MeshFunctionConfig, StarControls, StarJobControls,
    StarModel,
};

fn build_cpm_test_model(mass: f64, zbase: f64) -> StarModel {
    let ctrl = StarControls {
        initial_mass: mass,
        initial_z: zbase,
        zbase,
        mixing_length_alpha: 1.8,
        use_ledoux_criterion: true,
        do_conv_premix: true,
        ..StarControls::default()
    };
    let mut model = load_zams_model(ctrl, StarJobControls::default())
        .expect("load ZAMS model");
    model.dt = 3.418e6 * SECYER;
    model.controls.mix_wind.conv_premix_avoid_increase = true;
    model.controls.mix_wind.conv_premix_time_factor = 1.0;
    model.controls.mix_wind.conv_premix_fix_pgas = true;
    model.aux.mixing.okay_to_set_mixing_info = true;
    set_hydro_vars(&mut model, false, false).expect("initial set_hydro_vars");
    model.sync_mesh_and_pack().expect("sync_mesh_and_pack");
    model
}

fn find_species_index(model: &StarModel, z: usize, z_plus_n: usize) -> usize {
    let cat = flexmesa_chem::NuclideCatalog::builtin_ref();
    model
        .chem_id
        .iter()
        .position(|&cid| {
            cat.get_by_id(cid)
                .is_some_and(|e| e.z == z && e.z_plus_n == z_plus_n)
        })
        .expect("nuclide present in network")
}

#[test]
fn test_conv_premix_sets_need_to_setvars_on_degenerate_and_idle_models() {
    let mut tiny = StarModel::new(
        1,
        vec![2, 7],
        StarJobControls::default(),
        StarControls::default(),
    );
    tiny.aux.need_to_setvars = false;
    let adv_tiny = run_conv_premix(&mut tiny).expect("nz < 2 succeeds");
    assert_eq!(adv_tiny, 0);
    assert!(
        tiny.aux.need_to_setvars,
        "conv_premix must set need_to_setvars = true on return"
    );

    // Exhaust the clock budget so no boundary advances (`adv == 0`).
    let mut idle = build_cpm_test_model(1.5, 0.02);
    idle.controls.mix_wind.conv_premix_time_factor = 1.0e-20;
    idle.aux.need_to_setvars = false;
    let adv_idle = run_conv_premix(&mut idle).expect("idle conv_premix ok");
    assert_eq!(adv_idle, 0);
    assert!(
        idle.aux.need_to_setvars,
        "conv_premix must set need_to_setvars = true even when 0 cells advance \
         (star/private/conv_premix.f90:176)"
    );
}

#[test]
fn test_conv_premix_sets_need_to_setvars_and_leaves_xh_for_unpack_xh() {
    // Construct a Ledoux-stabilized composition gradient inside the
    // Schwarzschild core of a 4 Msun ZAMS model (`gradr >> grada`), mirroring
    // `phase5b_fidelity2_test`, so `do_conv_premix` advances outward across cells.
    let mut model = build_cpm_test_model(4.0, 0.02);
    model
        .controls
        .mix_wind
        .num_cells_for_smooth_gradl_composition_term = 0;
    model.controls.mix_wind.num_cells_for_smooth_brunt_b = 0;
    let n_bdy = locate_convection_boundaries(&mut model)
        .expect("locate_convection_boundaries");
    assert!(n_bdy > 0, "4 Msun ZAMS must have convective core");
    let k_bdy = model.aux.mixing.conv_bdy_loc[0];

    let sp = model.species;
    let i_h1 = find_species_index(&model, 1, 1);
    let i_he4 = find_species_index(&model, 2, 4);
    let k_grad_top = (k_bdy + 30).min(model.nz - 25);
    let k_grad_bot = (k_grad_top + 12).min(model.nz - 10);
    let span = f64::from(
        u32::try_from((k_grad_bot - k_grad_top).max(1)).expect("small span"),
    );
    for k in k_grad_top..model.nz {
        let frac = if k < k_grad_bot {
            f64::from(u32::try_from(k - k_grad_top).expect("small offset"))
                / span
        } else {
            1.0
        };
        let delta = (0.015 * frac).min(model.xa[k * sp + i_h1]);
        model.xa[k * sp + i_h1] -= delta;
        model.xa[k * sp + i_he4] += delta;
    }
    model.aux.mixing.okay_to_set_mixing_info = true;
    set_hydro_vars(&mut model, false, false).expect("refresh hydro_vars");
    model.sync_mesh_and_pack().expect("sync_mesh_and_pack");

    let mut wrapper_model = model.clone();
    let xh_before = model.xh.clone();
    let i_lnd = model.layout.i_lnd;
    let nvar = model.layout.nvar_hydro;
    model.aux.need_to_setvars = false;

    let adv = run_conv_premix(&mut model).expect("do_conv_premix");
    assert!(
        adv > 0,
        "expected advancing convective boundary across Ledoux-stabilized gradient"
    );
    assert!(
        model.aux.need_to_setvars,
        "do_conv_premix must set need_to_setvars = true \
         (star/private/conv_premix.f90:176)"
    );

    // Trial fixed-Pgas updates modify `model.lnd` while preserving pre-premix
    // `xh(i_lnd, k)` so `unpack_xh` recovers original `lnd` bits verbatim.
    assert_eq!(model.xh, xh_before);
    let max_lnd_trial_shift = (0..model.nz)
        .map(|k| (model.lnd[k] - model.xh[k * nvar + i_lnd]).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_lnd_trial_shift > 1.0e-7,
        "expected fixed-Pgas trial density shift during premixing, got \
         {max_lnd_trial_shift}"
    );

    let xa_after_cpm = model.xa.clone();
    model.unpack_xh();
    for k in 0..model.nz {
        assert_eq!(
            model.lnd[k].to_bits(),
            xh_before[k * nvar + i_lnd].to_bits(),
            "unpack_xh must restore pre-premix xh(i_lnd) at zone {k}"
        );
    }
    assert_eq!(model.xa, xa_after_cpm);

    wrapper_model.controls.do_premix_heating = true;
    wrapper_model.aux.need_to_setvars = false;
    let adv_w = run_premix_wrapper(&mut wrapper_model, 0, true)
        .expect("premix wrapper");
    assert_eq!(adv_w, adv);
    assert!(wrapper_model.aux.need_to_setvars);
}

#[test]
fn test_set_grads_smooths_gradl_composition_term_verbatim() {
    let nz = 25_usize;
    let ctrl = StarControls {
        use_ledoux_criterion: true,
        calculate_brunt_b: true,
        ..StarControls::default()
    };
    let mut model =
        StarModel::new(nz, vec![2, 7], StarJobControls::default(), ctrl);
    model
        .controls
        .mix_wind
        .num_cells_for_smooth_gradl_composition_term = 10;
    model
        .controls
        .mix_wind
        .threshold_for_smooth_gradl_composition_term = 0.02;

    for (k, val) in model.aux.mixing.unsmoothed_brunt_b[..nz]
        .iter_mut()
        .enumerate()
    {
        *val = if (8..=16).contains(&k) {
            let offset = i32::try_from(k).expect("zone index fits i32") - 12;
            0.05 + 0.15 * (1.0 - (f64::from(offset) / 5.0).abs())
        } else {
            0.005
        };
    }
    let raw = model.aux.mixing.unsmoothed_brunt_b.clone();
    set_grads(&mut model);

    for k in (0..8).chain(17..nz) {
        assert_eq!(
            model.aux.mixing.gradl_composition_term[k].to_bits(),
            raw[k].to_bits(),
            "sub-threshold entry {k} must remain untouched"
        );
    }
    for k in [8_usize, 16] {
        assert_eq!(
            model.aux.mixing.gradl_composition_term[k].to_bits(),
            raw[k].to_bits(),
            "threshold region endpoint {k} must remain untouched"
        );
    }
    assert!(
        (model.aux.mixing.gradl_composition_term[12] - raw[12]).abs() > 1.0e-3,
        "interior of threshold region must be smoothed"
    );

    model.controls.use_ledoux_criterion = false;
    set_grads(&mut model);
    assert!(model
        .aux
        .mixing
        .gradl_composition_term
        .iter()
        .all(|&v| v == 0.0));
}

#[test]
fn test_dlog_eps_dlogp_disables_extra_refinement_on_pressure_inversion() {
    let ctrl = StarControls {
        initial_mass: 1.3,
        initial_z: 0.04,
        zbase: 0.04,
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(ctrl, StarJobControls::default()).expect("1.3M ZAMS");
    model.aux.mixing.okay_to_set_mixing_info = true;
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");

    model.controls.mix_wind.mesh_dlog_eps_min_for_extra = 2.0;
    model.controls.mix_wind.mesh_dlog_eps_dlogp_full_off = 0.25;
    model.controls.mix_wind.mesh_dlog_eps_dlogp_full_on = 0.25;
    model.controls.mix_wind.mesh_dlog_pp_dlogp_extra = 0.25;
    model.controls.mix_wind.mesh_dlog_cno_dlogp_extra = 1.0;
    model.controls.mix_wind.mesh_dlog_3alf_dlogp_extra = 1.0;
    model.controls.mix_wind.mesh_delta_coeff_factor_smooth_iters = 0;

    let nz = model.nz;
    for cats in &mut model.aux.eps_nuc_categories[..nz] {
        cats.fill(0.0);
    }
    let k_step = nz / 2;
    for cats in &mut model.aux.eps_nuc_categories[(k_step - 3)..k_step] {
        cats[0] = 1.0e6;
    }
    for cats in &mut model.aux.eps_nuc_categories[k_step..=(k_step + 3)] {
        cats[0] = 1.0e3;
    }

    let nfcns = num_mesh_functions(&model).expect("nfcns");
    let g_mono = get_gval_info(&model, nfcns).expect("monotonic gval_info");
    assert!(
        (g_mono.delta_gval_max[k_step] - 0.25).abs() < 1.0e-12,
        "monotonic dlogP > 0 must refine across pp step: got {}",
        g_mono.delta_gval_max[k_step]
    );

    // Invert lnPeos at `k_step` so `dlogP < 0` across `(k_step - 1, k_step)`,
    // replicating the sensitivity at tiny shells in `1.3M_ms_high_Z`.
    model.ln_peos[k_step] = model.ln_peos[k_step - 1] - 1.0e-7 * LN10;
    model.ln_peos[k_step + 1] = model.ln_peos[k_step - 1] + 5.0e-7 * LN10;
    let g_inv = get_gval_info(&model, nfcns).expect("inverted gval_info");
    assert!(
        (g_inv.delta_gval_max[k_step - 1] - 1.0).abs() < 1.0e-12,
        "negative dlogP must skip extra pp refinement at k_step - 1"
    );
}

#[test]
fn test_do_adjust_mesh_steep_he_step_conserves_species_mass_and_splits() {
    let ctrl = StarControls {
        initial_mass: 1.3,
        initial_z: 0.04,
        zbase: 0.04,
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(ctrl, StarJobControls::default()).expect("1.3M ZAMS");
    model.aux.mixing.okay_to_set_mixing_info = true;
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");

    let sp = model.species;
    let i_h1 = find_species_index(&model, 1, 1);
    let i_he4 = find_species_index(&model, 2, 4);
    let k_step = model.nz / 2;
    for k in k_step..model.nz {
        let shift = 0.20_f64.min(model.xa[k * sp + i_h1] - 0.05);
        if shift > 0.0 {
            model.xa[k * sp + i_h1] -= shift;
            model.xa[k * sp + i_he4] += shift;
        }
    }
    set_hydro_vars(&mut model, false, false).expect("refresh hydro_vars");

    let mass_sum = |m: &StarModel, iso: usize| -> f64 {
        (0..m.nz).map(|k| m.dm[k] * m.xa[k * sp + iso]).sum()
    };
    let h1_before = mass_sum(&model, i_h1);
    let he4_before = mass_sum(&model, i_he4);

    let cfg = MeshFunctionConfig::from_controls(&model.controls);
    let plan = try_plan_standard_mesh(&model, &cfg)
        .expect("plan")
        .expect("some plan");
    assert!(
        plan.num_splits > 0,
        "expected mesh splits across steep He step"
    );
    let (splits, _merges) =
        do_adjust_mesh(&mut model, &cfg).expect("remesh across He step");
    assert!(
        splits > 0,
        "expected do_adjust_mesh splits across steep He composition step"
    );
    assert!(
        (model.dq[..model.nz].iter().sum::<f64>() - 1.0).abs() < 1.0e-13,
        "dq must sum to 1 after remesh"
    );
    for k in 0..(model.nz - 1) {
        assert!(
            model.xq[k] < model.xq[k + 1],
            "xq must stay strictly increasing at zone {k}"
        );
    }
    for &dm in &model.dm[..model.nz] {
        assert!(dm > 0.0);
    }

    let h1_after = mass_sum(&model, i_h1);
    let he4_after = mass_sum(&model, i_he4);
    assert!(
        (h1_after - h1_before).abs() / (h1_before + 1.0e-30 * M_SUN) < 1.0e-12,
        "h1 mass conservation drift: {h1_before} -> {h1_after}"
    );
    assert!(
        (he4_after - he4_before).abs() / (he4_before + 1.0e-30 * M_SUN)
            < 1.0e-12,
        "he4 mass conservation drift: {he4_before} -> {he4_after}"
    );
}
