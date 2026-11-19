// SPDX-License-Identifier: LGPL-3.0-or-later
//! Integration tests for `PrimaryStateBuffer` rollback preservation and
//! `MicrophysicsCache` 0-ULP hit/miss behavior (`#150`).

#![forbid(unsafe_code)]

use flexmesa_star::{
    load_zams_model, set_hydro_vars, star_evolve_step, PrimaryStateBuffer,
    StarControls, StarError, StarJobControls,
};

fn assert_bits_eq(label: &str, actual: &[f64], expected_bits: &[u64]) {
    assert_eq!(actual.len(), expected_bits.len(), "{label} length mismatch");
    for (idx, (&val, &want)) in actual.iter().zip(expected_bits).enumerate() {
        assert_eq!(val.to_bits(), want, "{label} mismatch at idx {idx}");
    }
}

#[test]
fn test_primary_state_buffer_zero_ulp_round_trip_memory_and_counter_survival() {
    let ctrl = StarControls {
        initial_mass: 1.0,
        rotation_flag: true,
        zbase: 0.02,
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(ctrl, StarJobControls::default()).expect("1 Msun ZAMS");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    model.save_start_values().expect("save_start_values");
    model.save_old_values();

    let buf = PrimaryStateBuffer::from_model(&model);
    let model_bytes = model.heap_bytes();
    let buf_bytes = buf.heap_bytes();
    assert!(
        buf_bytes * 4 < model_bytes,
        "PrimaryStateBuffer ({buf_bytes} B) should be >4x smaller than StarModel ({model_bytes} B)"
    );

    let ref_lnd: Vec<u64> = model.lnd.iter().map(|v| v.to_bits()).collect();
    let ref_peos: Vec<u64> = model.peos.iter().map(|v| v.to_bits()).collect();
    let ref_kap: Vec<u64> = model.opacity.iter().map(|v| v.to_bits()).collect();
    let ref_eps: Vec<u64> = model.eps_nuc.iter().map(|v| v.to_bits()).collect();
    let ref_jac: Vec<u64> =
        model.d_dxdt_nuc_dx.iter().map(|v| v.to_bits()).collect();
    let ref_teff = model.teff.to_bits();

    // Mutate state and advance all 6 diagnostic counters as a failed attempt would.
    model.model_number += 1;
    model.star_age += 123.0;
    model.lnd[0] += 0.25;
    model.ln_t[0] -= 0.15;
    model.d_dxdt_nuc_dx[0] += 999.0;
    model.num_retries += 3;
    model.num_redos += 2;
    model.total_num_solver_iterations += 17;
    model.total_num_solver_calls += 5;
    model.micro_cache_hits += 11;
    model.micro_cache_misses += 4;
    buf.restore_into(&mut model).expect("restore_into");

    assert_eq!(
        (
            model.num_retries,
            model.num_redos,
            model.total_num_solver_iterations,
            model.total_num_solver_calls,
        ),
        (3, 2, 17, 5),
        "rollback must preserve solver/retry/redo counters"
    );
    assert_eq!(
        (model.micro_cache_hits, model.micro_cache_misses),
        (12, 5),
        "restore_into preserves cache and reuses warm microphysics entry"
    );
    assert_eq!(model.teff.to_bits(), ref_teff);
    assert_bits_eq("lnd", &model.lnd, &ref_lnd);
    assert_bits_eq("peos", &model.peos, &ref_peos);
    assert_bits_eq("opacity", &model.opacity, &ref_kap);
    assert_bits_eq("eps_nuc", &model.eps_nuc, &ref_eps);
    assert_bits_eq("d_dxdt_nuc_dx", &model.d_dxdt_nuc_dx, &ref_jac);
}

#[test]
fn test_microphysics_cache_hit_exactness_and_one_ulp_invalidation() {
    let ctrl = StarControls {
        initial_mass: 1.0,
        zbase: 0.02,
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(ctrl, StarJobControls::default()).expect("1 Msun ZAMS");
    model.aux.micro_cache.clear();
    model.micro_cache_hits = 0;
    model.micro_cache_misses = 0;

    set_hydro_vars(&mut model, false, false).expect("cold eval");
    assert_eq!(model.micro_cache_misses, 1);
    assert_eq!(model.micro_cache_hits, 0);

    let peos0: Vec<u64> = model.peos.iter().map(|v| v.to_bits()).collect();
    let kap0: Vec<u64> = model.opacity.iter().map(|v| v.to_bits()).collect();
    let eps0: Vec<u64> = model.eps_nuc.iter().map(|v| v.to_bits()).collect();

    // Exact unchanged repeat hits cache with 0 ULP diff.
    set_hydro_vars(&mut model, false, false).expect("cached eval");
    assert_eq!(model.micro_cache_misses, 1);
    assert_eq!(model.micro_cache_hits, 1);
    assert_bits_eq("peos", &model.peos, &peos0);
    assert_bits_eq("opacity", &model.opacity, &kap0);
    assert_bits_eq("eps_nuc", &model.eps_nuc, &eps0);

    // 1-ULP perturbation of lnd invalidates cache.
    let orig_dens0 = model.lnd[0];
    model.lnd[0] = f64::from_bits(orig_dens0.to_bits() + 1);
    set_hydro_vars(&mut model, false, false).expect("1-ULP lnd");
    assert_eq!(model.micro_cache_misses, 2);

    // Restoring orig_dens0 hits slot 0 (2-slot LRU), then 1-ULP ln_t misses.
    model.lnd[0] = orig_dens0;
    set_hydro_vars(&mut model, false, false).expect("back to baseline");
    assert_eq!(model.micro_cache_hits, 2);

    let orig_temp0 = model.ln_t[0];
    model.ln_t[0] = f64::from_bits(orig_temp0.to_bits() + 1);
    set_hydro_vars(&mut model, false, false).expect("1-ULP ln_t");
    assert_eq!(model.micro_cache_misses, 3);

    model.ln_t[0] = orig_temp0;
    model.xa[0] = f64::from_bits(model.xa[0].to_bits() + 1);
    set_hydro_vars(&mut model, false, false).expect("1-ULP xa");
    assert_eq!(model.micro_cache_misses, 4);
}

#[test]
fn test_star_evolve_step_populates_solver_calls_and_cache_hits() {
    let ctrl = StarControls {
        initial_mass: 1.0,
        zbase: 0.02,
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(ctrl, StarJobControls::default()).expect("1 Msun ZAMS");
    model.dt = 1.0e11;
    model.dt_next = 1.0e11;
    let rep = star_evolve_step(&mut model, true, false, true).expect("step 1");
    assert!(rep.converged);
    assert!(model.total_num_solver_calls >= 1);
    assert!(model.total_num_solver_iterations >= rep.iterations);
    assert!(model.micro_cache_hits > 0);
}

#[test]
fn test_non_retriable_star_error_fails_immediately_with_zero_retries() {
    let ctrl = StarControls {
        initial_mass: 1.0,
        do_phase_separation: true,
        phase_separation_option: String::from("bogus_phase_sep"),
        zbase: 0.02,
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(ctrl, StarJobControls::default()).expect("1 Msun ZAMS");
    let orig_dt = 1.0e11;
    let orig_nz = model.nz;
    model.dt = orig_dt;
    model.dt_next = orig_dt;

    let err = star_evolve_step(&mut model, true, false, true)
        .expect_err("invalid phase_separation_option must fail loudly");
    assert!(matches!(err, StarError::InvalidInput(_)));
    assert_eq!(model.num_retries, 0, "non-retriable errors must not retry");
    assert_eq!(model.dt.to_bits(), orig_dt.to_bits());
    assert_eq!(model.nz, orig_nz);
}

#[test]
fn test_primary_state_buffer_preserves_omega_when_solver_mlt_eval_count_nonzero(
) {
    let ctrl = StarControls {
        initial_mass: 1.5,
        rotation_flag: true,
        zbase: 0.02,
        ..StarControls::default()
    };
    let mut model = load_zams_model(ctrl, StarJobControls::default())
        .expect("1.5 Msun ZAMS");
    for (k, om) in model.omega.iter_mut().enumerate() {
        let kf = f64::from(u32::try_from(k).unwrap_or(0));
        *om = 1.0e-5 * (1.0 + 0.01 * kf);
    }
    flexmesa_star::update_rotation_distortion_factors(&mut model);
    set_hydro_vars(&mut model, false, false).expect("initial set_hydro_vars");
    // Apply shellular angular-momentum / distortion update so `fp_rot` scales `grav`.
    flexmesa_star::sync_post_henyey_rotation(&mut model, 1.0);

    let snap = PrimaryStateBuffer::from_model(&model);
    eprintln!(
        "1.5 Msun ZAMS nz={}: StarModel::heap_bytes={} B, PrimaryStateBuffer::heap_bytes={} B",
        model.nz,
        model.heap_bytes(),
        snap.heap_bytes(),
    );
    let want_omega: Vec<u64> =
        model.omega.iter().map(|v| v.to_bits()).collect();
    let want_j_rot: Vec<u64> =
        model.j_rot.iter().map(|v| v.to_bits()).collect();
    let want_grav: Vec<u64> = model.grav.iter().map(|v| v.to_bits()).collect();
    let expected_rot_p: Vec<u64> =
        model.aux.fp_rot.iter().map(|v| v.to_bits()).collect();
    let expected_rot_t: Vec<u64> =
        model.aux.ft_rot.iter().map(|v| v.to_bits()).collect();
    let want_gradt: Vec<u64> =
        model.gradt.iter().map(|v| v.to_bits()).collect();
    let want_p_surf = model.ln_peos_surf_ad.val.to_bits();

    // Simulate a rejected Newton trial that perturbs thermodynamics, omega,
    // grav, and leaves solver_mlt_eval_count > 0.
    model.lnd[0] += 0.05;
    model.ln_t[0] -= 0.03;
    for om in &mut model.omega {
        *om *= 1.15;
    }
    for g in &mut model.grav {
        *g *= 1.07;
    }
    model.aux.solver_mlt_eval_count = 4;
    snap.restore_into(&mut model).expect("restore_into");

    assert_bits_eq("omega", &model.omega, &want_omega);
    assert_bits_eq("j_rot", &model.j_rot, &want_j_rot);
    assert_bits_eq("grav", &model.grav, &want_grav);
    assert_bits_eq("fp_rot", &model.aux.fp_rot, &expected_rot_p);
    assert_bits_eq("ft_rot", &model.aux.ft_rot, &expected_rot_t);
    assert_bits_eq("gradt", &model.gradt, &want_gradt);
    assert_eq!(model.ln_peos_surf_ad.val.to_bits(), want_p_surf);
    assert_eq!(model.aux.solver_mlt_eval_count, 0);
}

#[test]
fn test_evolve_step_rolls_back_remesh_on_terminate_and_attempt0_exhaustion() {
    // 1. Remesh + `TimestepAction::Terminate` must restore original pre-remesh mesh.
    let mut term_model = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            mesh_delta_coeff: 0.25,
            okay_to_remesh: true,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("1 Msun ZAMS");
    term_model.controls.ts_stop.min_allowed_varcontrol_target = 1.0;
    let term_nz = term_model.nz;
    let term_dq: Vec<u64> = term_model.dq.iter().map(|v| v.to_bits()).collect();
    let dt0 = term_model.dt;

    let term_err = star_evolve_step(&mut term_model, true, false, true)
        .expect_err("min_allowed_varcontrol_target violation must terminate");
    assert!(
        matches!(term_err, StarError::Terminated { .. }),
        "expected StarError::Terminated, got {term_err:?}"
    );
    assert_eq!(term_model.nz, term_nz, "Terminate must undo remesh nz");
    assert_bits_eq("Terminate dq", &term_model.dq, &term_dq);
    assert_eq!(term_model.dt.to_bits(), dt0.to_bits());

    // 2. Remesh + attempt-0 retry exhaustion must also restore original mesh.
    let mut exh_model = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            mesh_delta_coeff: 0.25,
            okay_to_remesh: true,
            max_newton_iters: 1,
            tol_correction_norm: 1.0e-15,
            tol_max_correction: 1.0e-15,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("1 Msun ZAMS");
    exh_model.controls.min_timestep_limit = 0.6 * exh_model.dt;
    exh_model.controls.ts_stop.timestep_factor_for_retries = 0.5;
    let exh_nz = exh_model.nz;
    let exh_dq: Vec<u64> = exh_model.dq.iter().map(|v| v.to_bits()).collect();

    let exh_err = star_evolve_step(&mut exh_model, true, false, true)
        .expect_err("attempt-0 exhaustion must return StepFailure");
    assert!(
        matches!(exh_err, StarError::StepFailure(_)),
        "expected StarError::StepFailure, got {exh_err:?}"
    );
    assert_eq!(exh_model.nz, exh_nz, "exhaustion must undo attempt-0 remesh");
    assert_bits_eq("exhaustion dq", &exh_model.dq, &exh_dq);
}

#[test]
fn test_var_layout_paired_variable_names_hydrostatic_and_hydro() {
    let iso = vec![String::from("h1"), String::from("he4")];
    let hse = flexmesa_star::VarLayout::new(2, false, false);
    assert_eq!(hse.paired_variable_name(hse.i_equ_l, &iso), "lnT");
    assert_eq!(hse.paired_variable_name(hse.i_dv_dt, &iso), "lnd");
    assert_eq!(hse.paired_variable_name(hse.i_dlnd_dt, &iso), "lnR");
    assert_eq!(hse.paired_variable_name(hse.i_dlne_dt, &iso), "L");
    assert_eq!(hse.paired_variable_name(hse.chem_var_index(1), &iso), "he4");

    let vel = flexmesa_star::VarLayout::new(2, true, false);
    assert_eq!(vel.paired_variable_name(vel.i_equ_l, &iso), "L");
    assert_eq!(vel.paired_variable_name(vel.i_dv_dt, &iso), "v");
    assert_eq!(vel.paired_variable_name(vel.i_dlnd_dt, &iso), "lnd");
    assert_eq!(vel.paired_variable_name(vel.i_dlne_dt, &iso), "lnT");
    assert_eq!(
        vel.paired_variable_name(vel.i_dlnr_dt.expect("i_dlnr_dt"), &iso),
        "lnR"
    );
}
