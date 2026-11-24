// SPDX-License-Identifier: GPL-3.0-or-later
//! Verification suite for `#152` Newton iteration control, `Bdomain`,
//! backtracking `adjust_correction`, `sizequ`/`sizeB` (`#177`, `#251`),
//! `restore_mesh_on_retry`, `select_mdot_action` solver-failure retry (`#139`),
//! `min_timestep_limit` / `max_number_retries` / `redo_limit` termination, and
//! `timestep.f90` limiters (`#177`, `#253`).

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{FOUR_THIRDS_PI, M_SUN, SECYER};
use flexmesa_star::evolve::{
    restore_from_old, snapshot_step_old_extras, star_evolve_step_with_extras,
};
use flexmesa_star::hydro_eqns::HenyeySystem;
use flexmesa_star::inlist_runner::{ExtrasDirective, StarExtras};
use flexmesa_star::solver_norms::{
    active_correction_tolerances, active_max_tries,
    compute_model_correction_norms, compute_residual_norms,
};
use flexmesa_star::solver_support::{apply_bdomain, set_vars_for_solver};
use flexmesa_star::{
    StarControls, StarError, StarJobControls, StarModel, TimestepAction,
    load_zams_model, set_hydro_vars, solve_henyey_step, star_evolve_step,
    timestep_controller,
};

#[test]
fn test_sizequ_double_counts_chemistry_residuals_and_sizeb_matches_upstream_251()
 {
    let mut m = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            zbase: 0.02,
            correction_xa_limit: 0.01,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("zams");
    m.resize_zones(2);
    let nvar_h = m.layout.nvar_hydro;
    let sp = m.species;
    let nvar = nvar_h + sp;
    let mut sys = HenyeySystem::new(2, nvar, vec![1.0; 2 * nvar]);
    // Set hydro residuals to 0.2 and chem residual slot 0 to 0.6 on both zones.
    for k in 0..2 {
        for i in 0..nvar_h {
            sys.equ[k * nvar + i] = 0.2;
        }
        sys.equ[k * nvar + nvar_h] = 0.6;
    }

    // `do_chem_in_sizequ = false`: only `2 * nvar_h` hydro equations counted.
    m.controls.do_burn = false;
    m.controls.do_mix = false;
    let (rnorm_no_chem, rmax_no_chem) =
        compute_residual_norms(&m, nvar, &sys.equ);
    assert!((rnorm_no_chem - 0.2).abs() <= 1.0e-15);
    assert!((rmax_no_chem - 0.2).abs() <= 1.0e-15);

    // `do_chem_in_sizequ = true`: `#177` counts chem residuals twice (`2 * (nvar_h + 2 * sp)`).
    m.controls.do_burn = true;
    let (rnorm_chem, rmax_chem) = compute_residual_norms(&m, nvar, &sys.equ);
    let nvar_h_f = f64::from(u32::try_from(nvar_h).expect("nvar_h fits u32"));
    let expected_sum = 2.0 * (nvar_h_f * 0.2 + 2.0 * 0.6);
    let expected_cnt =
        f64::from(u32::try_from(2 * (nvar_h + 2 * sp)).expect("cnt fits u32"));
    assert!((rnorm_chem - expected_sum / expected_cnt).abs() <= 1.0e-15);
    assert!((rmax_chem - 0.6).abs() <= 1.0e-15);

    // `sizeB` (`#251`): only h1 (`sp=0`) and he4 (`sp=1`) exceed `correction_xa_limit = 0.01`.
    m.xa.fill(0.0);
    for k in 0..2 {
        m.xa[k * sp] = 0.70;
        m.xa[k * sp + 1] = 0.28;
        m.xa[k * sp + 2] = 0.002;
    }
    m.xa_start.clone_from(&m.xa);
    m.controls.include_l_in_correction_limits = true;
    m.lum = vec![0.9, 0.9];
    m.lum_start = vec![1.0, 1.0];
    let mut dx = vec![0.0; 2 * nvar];
    let x_scale = vec![1.0; 2 * nvar];
    // Zone 0: hydro corrections 0.1, h1=0.2, he4=0.4; Zone 1: hydro 0.1, h1=0.6, he4=0.8.
    for k in 0..2 {
        for j in 0..nvar_h {
            dx[k * nvar + j] = 0.1;
        }
    }
    dx[nvar_h] = 0.2;
    dx[nvar_h + 1] = 0.4;
    dx[nvar + nvar_h] = 0.6;
    dx[nvar + nvar_h + 1] = 0.8;

    let (cnorm, cmax) =
        compute_model_correction_norms(&m, 2, nvar, &dx, &x_scale);
    // Note: `0.1 * lum_surf_start + |lum[k]| = 0.1 * 1.0 + 0.9 = 1.0`, so `i_lum` weight = 1.0.
    // Zone 0: `nvar_h` hydro terms (0.1 each) + cumulative mean `(0.2+0.4)/2 = 0.3` + species_loop `0.2 + 0.4`.
    // Zone 1: `nvar_h` hydro terms (0.1 each) + cumulative mean `(0.2+0.4+0.6+0.8)/4 = 0.5` + species_loop `0.6 + 0.8`.
    let exp_sum_b = nvar_h_f * 0.1 + 0.3 + 0.6 + nvar_h_f * 0.1 + 0.5 + 1.4;
    let exp_terms_b =
        f64::from(u32::try_from(2 * (nvar_h + 1 + 2)).expect("terms fits u32"));
    assert!((cnorm - exp_sum_b / exp_terms_b).abs() <= 1.0e-15);
    assert!((cmax - 0.8).abs() <= 1.0e-15);
}

#[test]
fn test_bdomain_damps_neg_xa_clips_rti_alpha_and_edit_lnr_projects_radii() {
    let mut m = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            rti_flag: true,
            zbase: 0.02,
            corr_coeff_limit: 1.0e-6,
            min_xa_hard_limit: -1.0e-4,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("zams");
    m.resize_zones(2);
    m.r_center = 1.0e9;
    m.aux.alpha_rti = vec![0.05, 0.05];
    m.aux.alpha_rti_start = vec![0.05, 0.05];
    m.pack_xh();
    let xh_start = m.xh.clone();
    m.xh_start.clone_from(&xh_start);
    m.xa_start.clone_from(&m.xa);

    let nvar = m.layout.nvar(true);
    let x_scale = vec![1.0; 2 * nvar];
    let mut solver_dx = vec![0.0; 2 * nvar];
    let mut soln = vec![0.0; 2 * nvar];
    let mut corr_fac = 1.0_f64;

    // 1. Negative `alpha_rti` step larger than `alpha_rti[0] = 0.05` must clip in `soln` to `-0.05`.
    let ia = m.layout.i_alpha_rti.expect("rti slot");
    soln[ia] = -0.20;
    // 2. Large negative species step on `h1` (`old_xa = m.xa_start[0]`) damps `corr_fac`.
    let ih1_var = m.layout.chem_var_index(0);
    let old_h1 = m.xa_start[0];
    soln[ih1_var] = -2.0 * old_h1;

    apply_bdomain(
        &m,
        nvar,
        &xh_start,
        &solver_dx,
        &x_scale,
        &mut soln,
        &mut corr_fac,
    );
    assert!((soln[ia] + 0.05).abs() <= 1.0e-15);
    let eps = -0.5 * m.controls.min_xa_hard_limit;
    let expected_alpha = (old_h1 + eps) / (2.0 * old_h1);
    assert!((corr_fac - expected_alpha).abs() <= 1.0e-15);

    // 3. `set_vars_for_solver` projects `ln_r` consistently from `R_center`, `dm`, and `rho` (`edit_lnR`).
    solver_dx.fill(0.0);
    set_vars_for_solver(&mut m, true, &xh_start, &mut solver_dx)
        .expect("set_vars_for_solver edit_lnR");
    let vol1 =
        FOUR_THIRDS_PI * flexmesa_math::pow3(m.r_center) + m.dm[1] / m.rho[1];
    let expected_lnr1 = flexmesa_math::log(vol1 / FOUR_THIRDS_PI) / 3.0;
    assert!((m.ln_r[1] - expected_lnr1).abs() <= 1.0e-14);
}

#[test]
fn test_active_max_tries_schedule_and_high_t_correction_tolerance_relaxation() {
    let mut m = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            zbase: 0.02,
            max_tries1: 13,
            solver_max_tries_before_reject: 21,
            max_tries_for_retry: 29,
            max_tries_after_5_retries: 37,
            max_tries_after_10_retries: 45,
            max_tries_after_20_retries: 53,
            tol_correction_norm: 3.0e-3,
            tol_max_correction: 3.0e-2,
            tol_correction_high_t_limit: 1.0e9,
            tol_correction_norm_high_t: 3.0e-2,
            tol_max_correction_high_t: 3.0e-1,
            tol_correction_extreme_t_limit: 3.0e9,
            tol_correction_norm_extreme_t: 1.0e-1,
            tol_max_correction_extreme_t: 5.0e-1,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("zams");

    m.doing_first_model_of_run = true;
    m.solver_first_try_of_step = true;
    m.step_retry_cnt = 0;
    assert_eq!(active_max_tries(&m), 13);

    m.doing_first_model_of_run = false;
    assert_eq!(active_max_tries(&m), 21);

    m.solver_first_try_of_step = false;
    m.step_retry_cnt = 1;
    assert_eq!(active_max_tries(&m), 29);
    m.step_retry_cnt = 6;
    assert_eq!(active_max_tries(&m), 37);
    m.step_retry_cnt = 11;
    assert_eq!(active_max_tries(&m), 45);
    m.step_retry_cnt = 21;
    assert_eq!(active_max_tries(&m), 53);

    assert_eq!(active_correction_tolerances(&m), (3.0e-3, 3.0e-2));
    let nz = m.nz;
    m.ln_t[nz - 1] = flexmesa_math::log(1.5e9);
    assert_eq!(active_correction_tolerances(&m), (3.0e-2, 3.0e-1));
    m.ln_t[nz - 1] = flexmesa_math::log(4.0e9);
    assert_eq!(active_correction_tolerances(&m), (1.0e-1, 5.0e-1));
}

#[test]
fn test_solver_itermin_and_adjust_correction_backtracking_converge() {
    let mut m = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            zbase: 0.02,
            solver_itermin: 4,
            corr_coeff_limit: 0.05,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("zams");
    m.doing_first_model_of_run = false;
    m.solver_skip_global_corr_coeff_limit = false;
    set_hydro_vars(&mut m, false, false).expect("hydro");
    m.save_start_values().expect("start");
    m.lnd[3] += 0.02;
    m.pack_xh();

    let rep = solve_henyey_step(&mut m, true, true).expect("henyey converges");
    assert!(rep.converged);
    assert!(
        rep.iterations >= 4,
        "solver_itermin=4 must enforce at least 4 iterations, got {}",
        rep.iterations
    );
}

#[test]
fn test_min_xa_and_sum_xa_hard_limits_reject_without_silent_clamping() {
    let mut m = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            zbase: 0.02,
            min_xa_hard_limit: -1.0e-4,
            sum_xa_hard_limit: 1.0e-3,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("zams");
    m.doing_first_model_of_run = false;
    set_hydro_vars(&mut m, false, false).expect("hydro");
    m.save_start_values().expect("start");
    m.pack_xh();
    let xh_start = m.xh.clone();
    let nvar = m.layout.nvar(true);

    // 1. Negative abundance below `min_xa_hard_limit` must fail and flag `why_tlim_neg_x`.
    let mut m_neg = m.clone();
    let mut dx_neg = vec![0.0; m.nz * nvar];
    dx_neg[m.layout.chem_var_index(0)] = -(m.xa_start[0] + 1.0e-3);
    dx_neg[m.layout.chem_var_index(1)] = m.xa_start[0] + 1.0e-3;
    let err_neg = set_vars_for_solver(&mut m_neg, true, &xh_start, &mut dx_neg)
        .expect_err("min_xa_hard_limit must reject");
    assert!(matches!(err_neg, StarError::ConvergenceFailure(_)));
    assert!(m_neg.why_tlim_neg_x);

    // 2. Sum deviation above `sum_xa_hard_limit` must fail instead of renormalizing.
    let mut m_sum = m;
    let mut dx_sum = vec![0.0; m_sum.nz * nvar];
    dx_sum[m_sum.layout.chem_var_index(0)] = 5.0e-3;
    let err_sum = set_vars_for_solver(&mut m_sum, true, &xh_start, &mut dx_sum)
        .expect_err("sum_xa_hard_limit must reject");
    assert!(matches!(err_sum, StarError::ConvergenceFailure(_)));
}

#[test]
fn test_restore_mesh_on_retry_false_keeps_remeshed_grid_and_true_restores() {
    let run_case = |restore_mesh: bool| -> (usize, usize, usize, usize) {
        let mut ctrl = StarControls {
            initial_mass: 1.0,
            zbase: 0.02,
            mesh_delta_coeff: 0.25,
            okay_to_remesh: true,
            max_tries1: 1,
            solver_max_tries_before_reject: 1,
            max_tries_for_retry: 20,
            solver_itermin: 2,
            ..StarControls::default()
        };
        ctrl.mix_wind.restore_mesh_on_retry = restore_mesh;
        let mut m =
            load_zams_model(ctrl, StarJobControls::default()).expect("zams");
        m.doing_first_model_of_run = false;
        let nz_init = m.nz;
        let rep = star_evolve_step(&mut m, true, false, true)
            .expect("retry 1 converges with max_tries_for_retry=20");
        assert_eq!(rep.retries, 1);
        (nz_init, m.nz, rep.num_splits, rep.num_merges)
    };

    let (init_keep, final_keep, sp_keep, mg_keep) = run_case(false);
    assert_ne!(
        final_keep, init_keep,
        "restore_mesh_on_retry=false must keep remeshed grid across retry"
    );
    assert!(sp_keep + mg_keep > 0);

    let (init_rest, final_rest, sp_rest, mg_rest) = run_case(true);
    assert_eq!(
        final_rest, init_rest,
        "restore_mesh_on_retry=true must restore pre-remesh grid on retry"
    );
    assert_eq!(sp_rest + mg_rest, 0);
}

#[test]
fn test_restore_from_old_restores_center_scalars_time_and_mlt_vc() {
    let mut m = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("zams");
    m.m_center = 1.0e30;
    m.r_center = 2.0e8;
    m.l_center = 3.0e31;
    m.v_center = -1.5e4;
    m.time_sec = 1.25e10;
    m.star_age = m.time_sec / SECYER;
    for (k, vc) in m.mlt_vc.iter_mut().enumerate() {
        let idx = f64::from(u32::try_from(k + 1).expect("zone index fits u32"));
        *vc = 1.0e3 * idx;
    }
    snapshot_step_old_extras(&mut m);
    m.save_old_values();

    let want_vc: Vec<u64> = m.mlt_vc.iter().map(|v| v.to_bits()).collect();
    m.m_center = 9.0e30;
    m.r_center = 7.0e8;
    m.l_center = 8.0e31;
    m.v_center = 5.0e5;
    m.time_sec = 9.0e10;
    m.mlt_vc.fill(0.0);

    restore_from_old(&mut m).expect("restore_from_old");
    assert_eq!(m.m_center.to_bits(), 1.0e30_f64.to_bits());
    assert_eq!(m.r_center.to_bits(), 2.0e8_f64.to_bits());
    assert_eq!(m.l_center.to_bits(), 3.0e31_f64.to_bits());
    assert_eq!(m.v_center.to_bits(), (-1.5e4_f64).to_bits());
    assert_eq!(m.time_sec.to_bits(), 1.25e10_f64.to_bits());
    let got_vc: Vec<u64> = m.mlt_vc.iter().map(|v| v.to_bits()).collect();
    assert_eq!(got_vc, want_vc);
}

#[test]
fn test_select_mdot_action_solver_failure_retries_without_force_accept_139() {
    struct FailOnBoostedWindExtras {
        base_mdot_msun_yr: f64,
    }
    impl StarExtras for FailOnBoostedWindExtras {
        fn other_wind(
            &self,
            _m: &StarModel,
            _l: f64,
            _ms: f64,
            _r: f64,
            _t: f64,
        ) -> Result<Option<f64>, StarError> {
            Ok(Some(self.base_mdot_msun_yr))
        }
    }

    let mut m = load_zams_model(
        StarControls {
            initial_mass: 1.5,
            rotation_flag: true,
            use_other_wind: true,
            surf_omega_div_omega_crit_limit: 0.05,
            max_number_retries: 0,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("1.5 Msun ZAMS");
    m.doing_first_model_of_run = false;
    m.controls.mix_wind.max_mdot_redo_cnt = 5;
    m.controls.mix_wind.min_years_dt_for_redo_mdot = 0.0;
    m.controls.mix_wind.mdot_revise_factor = 1.0e12;
    m.controls.mix_wind.rotational_mdot_boost_fac = 1.0e20;
    m.controls.mix_wind.rotational_mdot_kh_fac = 1.0e20;
    m.omega.fill(2.0e-4);
    flexmesa_star::update_rotation_distortion_factors(&mut m);
    set_hydro_vars(&mut m, false, false).expect("hydro");
    flexmesa_star::sync_post_henyey_rotation(&mut m, 1.0);
    m.save_start_values().expect("start");
    m.save_old_values();

    let mut extras = FailOnBoostedWindExtras {
        base_mdot_msun_yr: -1.0e-10,
    };
    let err = star_evolve_step_with_extras(
        &mut m,
        false,
        false,
        true,
        &mut extras,
    )
    .expect_err(
        "boosted mdot failure must retry and hit max_number_retries=0 (#139)",
    );
    assert!(
        matches!(
            err,
            StarError::Terminated { ref reason, .. } if reason == "max_number_retries"
        ),
        "expected Terminated(max_number_retries), got {err:?}"
    );
}

#[test]
fn test_redo_limit_terminates_and_delta_mdot_hard_limit_retries() {
    struct AlwaysRedoExtras;
    impl StarExtras for AlwaysRedoExtras {
        fn extras_check_model(
            &mut self,
            _m: &mut StarModel,
        ) -> ExtrasDirective {
            ExtrasDirective::Redo
        }
    }

    let mut redo_model = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("zams");
    redo_model.controls.ts_stop.redo_limit = 2;
    let mut extras = AlwaysRedoExtras;
    let redo_err = star_evolve_step_with_extras(
        &mut redo_model,
        false,
        false,
        true,
        &mut extras,
    )
    .expect_err("redo_limit=2 must terminate after 3rd redo");
    assert!(
        matches!(
            redo_err,
            StarError::Terminated { ref reason, .. } if reason == "redo_limit"
        ),
        "expected Terminated(redo_limit), got {redo_err:?}"
    );
    assert_eq!(redo_model.num_redos, 3);

    // Verify `check_delta_mdot` moved into `timestep_controller`.
    let mut mdot_model = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("zams");
    mdot_model.doing_first_model_of_run = false;
    mdot_model.aux.step_hist.mstar_dot_old = -1.0e-8 * M_SUN / SECYER;
    mdot_model.mstar_dot = -1.0e-6 * M_SUN / SECYER;
    mdot_model.controls.mix_wind.delta_mdot_hard_limit = 0.5;
    mdot_model.controls.mix_wind.delta_mdot_rtol = 0.5;
    mdot_model.controls.mix_wind.delta_mdot_atol = 1.0e-12;
    let (act_mdot, _) = timestep_controller(&mut mdot_model);
    assert_eq!(
        act_mdot,
        TimestepAction::Retry(String::from("delta_mdot_hard_limit"))
    );
}

#[test]
fn test_timestep_checks_match_177_and_253() {
    // Verify `#177` `'X'`/`'Y'` tautological cycle vs `'Z'` binding, and
    // `use_dt_low_pass_controller = false` first-order `dt / dt_limit_ratio`.
    let mut ts_model = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("zams");
    ts_model.doing_first_model_of_run = false;
    ts_model.d_mix.fill(0.0);
    ts_model
        .mixing_type
        .fill(flexmesa_const::MixingType::NoMixing);
    ts_model.save_old_values();
    ts_model.dt = 100.0;
    ts_model.controls.ts_stop.use_dt_low_pass_controller = false;
    ts_model.controls.ts_stop.min_timestep_factor = 0.0;
    ts_model.controls.ts_stop.max_timestep_factor = 0.0;

    // Perturb `h1` (`X`), `he4` (`Y`), and `c12` (`Z`) in radiative zone 0.
    let ih1 = ts_model
        .iso_names
        .iter()
        .position(|s| s == "h1")
        .expect("h1");
    let ihe4 = ts_model
        .iso_names
        .iter()
        .position(|s| s == "he4")
        .expect("he4");
    let ic12 = ts_model
        .iso_names
        .iter()
        .position(|s| s == "c12")
        .expect("c12");
    ts_model.xa_old[ih1] = 0.70;
    ts_model.xa[ih1] = 0.50;
    ts_model.xa_old[ihe4] = 0.28;
    ts_model.xa[ihe4] = 0.10;
    ts_model.xa_old[ic12] = 0.02;
    ts_model.xa[ic12] = 0.005;

    // `'X'` and `'Y'` rules must NOT bind (`#177`).
    ts_model.controls.ts_stop.dx_limit_species[0] = String::from("X");
    ts_model.controls.ts_stop.dx_hard_limit[0] = 0.01;
    ts_model.controls.ts_stop.dx_limit_species[1] = String::from("Y");
    ts_model.controls.ts_stop.dx_hard_limit[1] = 0.01;
    let (act_xy, _) = timestep_controller(&mut ts_model);
    assert_eq!(act_xy, TimestepAction::KeepGoing);

    // `'Z'` soft limit `0.005` on `delta_X = 0.015` gives `dt_limit_ratio = 3.0`,
    // and first-order controller sets `dt_next = 100.0 / 3.0` without `filter_limit`.
    ts_model.controls.ts_stop.dx_limit_species[2] = String::from("Z");
    ts_model.controls.ts_stop.dx_limit[2] = 0.005;
    ts_model.controls.ts_stop.dx_limit_min_x[2] = 0.001;
    let (act_z, rep_z) = timestep_controller(&mut ts_model);
    assert_eq!(act_z, TimestepAction::KeepGoing);
    assert!((rep_z.dt_limit_ratio - 3.0).abs() <= 1.0e-12);
    assert!((rep_z.dt_next - 100.0 / 3.0).abs() <= 1.0e-12);

    // Verify `#253` out-of-bounds guard when matching species index `j >= 100`.
    let cat = NuclideCatalog::builtin_ref();
    let chem_ids_101: Vec<usize> = (1..=101).collect();
    let iso_101 = cat.get_by_id(101).expect("chem_id 101").name.clone();
    let mut big_net = StarModel::new(
        2,
        chem_ids_101,
        StarJobControls::default(),
        StarControls {
            zbase: 0.02,
            ..StarControls::default()
        },
    );
    big_net.doing_first_model_of_run = false;
    big_net.xa[100] = 0.05;
    big_net.xa_old.clone_from(&big_net.xa);
    big_net.xa[100] = 0.06;
    big_net.controls.ts_stop.dx_limit_species[0] = iso_101;
    big_net.controls.ts_stop.dx_limit[0] = 0.001;
    let (act_oob, _) = timestep_controller(&mut big_net);
    assert!(
        matches!(act_oob, TimestepAction::Terminate(ref msg) if msg.contains("#253")),
        "expected #253 OOB guard, got {act_oob:?}"
    );
}

#[test]
fn test_sizeb_flailing_check_reproduces_upstream_260_never_fires() {
    let mut tracker = flexmesa_star::solver_norms::FlailingTracker::default();
    let r1 = tracker.observe(1, 1.0e-3, 1.0e-2, Some((4, 1)));
    let r2 = tracker.observe(2, 1.0e-3, 1.0e-1, Some((4, 1)));
    let r3 = tracker.observe(3, 1.0e-3, 1.0, Some((4, 1)));
    assert_eq!(r1, 1.0e-2);
    assert_eq!(r2, 1.0e-1);
    assert_eq!(
        r3, 1.0,
        "upstream #260 overwrites abs_max_corr1 before comparison so flailing never fires"
    );
}

/// MESA 26.4.1 keeps the `max_timestep_hi_T_limit` switch of a failed try
/// (`star/private/evolve.f90:705-711`): `prepare_to_retry` recomputes
/// `max_timestep` from the switched `max_years_for_timestep` and caps the
/// retry `dt` with it (`:2250-2255`), and `set_current_to_old`
/// (`star/private/evolve_support.f90:147-240`) restores neither (#510).
#[test]
fn test_retry_keeps_max_timestep_hi_t_limit_switch_of_failed_try() {
    struct RejectOnceExtras {
        calls: usize,
        saw_second_try_max_timestep: f64,
        saw_second_try_dt: f64,
    }
    impl StarExtras for RejectOnceExtras {
        fn extras_check_model(
            &mut self,
            model: &mut StarModel,
        ) -> ExtrasDirective {
            self.calls += 1;
            if self.calls == 1 {
                // First trial crossed `max_timestep_hi_t_limit` and mutated both
                // `max_years_for_timestep` and `ts_stop.max_timestep`; disable
                // the trigger and request a Retry so `prepare_to_retry` runs.
                assert!(
                    (model.controls.max_years_for_timestep - 0.01).abs()
                        < 1.0e-12
                );
                assert!(
                    (model.controls.ts_stop.max_timestep - 0.01 * SECYER).abs()
                        < 1.0e-6
                );
                model.controls.ts_stop.max_timestep_hi_t_limit = 1.0e30;
                ExtrasDirective::Retry
            } else {
                self.saw_second_try_max_timestep =
                    model.controls.ts_stop.max_timestep;
                self.saw_second_try_dt = model.dt;
                ExtrasDirective::KeepGoing
            }
        }
    }

    let mut model = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("1.0 Msun ZAMS should load");
    let orig_max_years = 1.0e6;
    let orig_max_dt = 0.0;
    model.controls.max_years_for_timestep = orig_max_years;
    model.controls.ts_stop.max_timestep = orig_max_dt;
    model.controls.ts_stop.max_timestep_hi_t_limit = 1.0e5;
    model.controls.ts_stop.hi_t_max_years_for_timestep = 0.01;
    model.controls.max_number_retries = 10;
    model.dt_next = 10.0 * SECYER;

    let mut extras = RejectOnceExtras {
        calls: 0,
        saw_second_try_max_timestep: -1.0,
        saw_second_try_dt: -1.0,
    };
    let rep = star_evolve_step_with_extras(
        &mut model,
        false,
        false,
        true,
        &mut extras,
    )
    .expect("retry step should succeed");
    assert!(rep.converged);
    assert_eq!(extras.calls, 2);
    assert_eq!(
        model.controls.max_years_for_timestep, 0.01,
        "the hi_T switch of the failed try stays after the step"
    );
    assert_eq!(
        extras.saw_second_try_max_timestep,
        SECYER * 0.01,
        "prepare_to_retry recomputes max_timestep from the switched \
         max_years_for_timestep"
    );
    assert!(
        extras.saw_second_try_dt <= SECYER * 0.01,
        "retry dt is capped by the switched max_timestep, got {} s",
        extras.saw_second_try_dt
    );
}

/// `prepare_to_redo` (`star/private/evolve.f90:2148-2172`) and
/// `set_current_to_old` keep the `max_timestep_hi_T_limit` switch of the
/// redone try (`:705-711`), so the redo and the accepted step see it (#510).
#[test]
fn test_redo_keeps_max_timestep_hi_t_limit_switch() {
    struct RedoOnceExtras {
        calls: usize,
        saw_second_try_max_timestep: f64,
    }
    impl StarExtras for RedoOnceExtras {
        fn extras_check_model(
            &mut self,
            model: &mut StarModel,
        ) -> ExtrasDirective {
            self.calls += 1;
            if self.calls == 1 {
                assert!(
                    (model.controls.ts_stop.max_timestep - 0.01 * SECYER).abs()
                        < 1.0e-6
                );
                model.controls.ts_stop.max_timestep_hi_t_limit = 1.0e30;
                ExtrasDirective::Redo
            } else {
                self.saw_second_try_max_timestep =
                    model.controls.ts_stop.max_timestep;
                ExtrasDirective::KeepGoing
            }
        }
    }
    let mut redo_model = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("1.0 Msun ZAMS should load");
    redo_model.controls.max_years_for_timestep = -1.0;
    redo_model.controls.ts_stop.max_timestep = 0.0;
    redo_model.controls.ts_stop.max_timestep_hi_t_limit = 1.0e5;
    redo_model.controls.ts_stop.hi_t_max_years_for_timestep = 0.01;
    let mut redo_extras = RedoOnceExtras {
        calls: 0,
        saw_second_try_max_timestep: -1.0,
    };
    let redo_rep = star_evolve_step_with_extras(
        &mut redo_model,
        false,
        false,
        true,
        &mut redo_extras,
    )
    .expect("redo step should succeed");
    assert!(redo_rep.converged);
    assert_eq!(redo_extras.calls, 2);
    assert_eq!(redo_extras.saw_second_try_max_timestep, SECYER * 0.01);
    assert_eq!(redo_model.controls.max_years_for_timestep, 0.01);
    assert_eq!(redo_model.controls.ts_stop.max_timestep, SECYER * 0.01);
}

#[test]
fn test_relax_uniform_omega_restores_all_timestamps_and_applies_ramped_omega_on_early_exit()
 {
    let mut model = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("1.0 Msun ZAMS should load");
    let target_omega = 1.0e-5;
    // Start within 5e-5 relative of `target_omega` so `done` triggers on step 1
    // even with `num_steps = 5`.
    let start_omega = target_omega * (1.0 - 5.0e-5);
    model.controls.rotation_flag = true;
    model.controls.okay_to_remesh = false;
    model.omega.fill(start_omega);
    model.star_age = 1.0e8;
    model.star_age_old = 9.999_9e7;
    model.time_sec = model.star_age * SECYER;
    model.time_old_sec = model.star_age_old * SECYER;
    model.aux.step_hist.time_sec = model.time_sec;
    model.aux.step_hist.time_old_sec = model.time_old_sec;

    let saved_time_sec = model.time_sec;
    let saved_time_old_sec = model.time_old_sec;

    flexmesa_star::relax_uniform_omega(&mut model, target_omega, 5, 1.0e-3)
        .expect("relax_uniform_omega should succeed");

    assert_eq!(model.time_sec, saved_time_sec);
    assert_eq!(model.time_old_sec, saved_time_old_sec);
    assert_eq!(model.aux.step_hist.time_sec, saved_time_sec);
    assert_eq!(model.aux.step_hist.time_old_sec, saved_time_old_sec);
    // Upstream applies `this_step_omega = frac*new_omega + (1-frac)*starting_omega`
    // with `frac = (model_number - starting_model_number)/num_steps`, capped at
    // 1, on the terminating step too (`star/private/relax.f90:1355-1356, 1374,
    // 1383-1389`). Which step terminates depends on how far the solve moves the
    // surface omega, so any step of the ramp is accepted; the result must be a
    // uniform omega on the ramp, never a substitute value.
    // `set_uniform_omega` ends with `set_rotation_info`, whose `set_omega`
    // recomputes `omega = j_rot/i_rot` (`hydro_rotation.f90:532, 549-550`),
    // so the profile is uniform to round-off rather than bit for bit.
    let on_ramp = (1_u32..=5 + 51).any(|k| {
        let frac = 1.0_f64.min(f64::from(k) / 5.0_f64);
        let e = frac * target_omega + (1.0 - frac) * start_omega;
        model.omega.iter().all(|&w| (w - e).abs() <= 1.0e-14 * e.abs())
    });
    assert!(on_ramp, "the terminating step applies the ramped omega, as upstream");
}

#[test]
fn test_make_gradr_sticky_tracks_solver_iter_not_line_search_eval_count() {
    let mut ctrl = StarControls {
        initial_mass: 1.0,
        zbase: 0.02,
        ..StarControls::default()
    };
    ctrl.mix_wind.make_gradr_sticky_in_solver_iters = true;
    let mut model =
        load_zams_model(ctrl, StarJobControls::default()).expect("1 Msun ZAMS");
    model.aux.doing_struct_burn_mix = true;
    model.aux.solver_mlt_eval_count = 1;
    model.aux.fixed_gradr_sticky.fill(false);

    // Repeated line-search evaluations during Newton iteration 1 must NOT
    // advance `solver_iter` past 3 or latch `fixed_gradr_sticky`
    // (`star_solver.f90:284, 610`, `turb_info.f90:56`).
    for _ in 0..5 {
        set_hydro_vars(&mut model, false, false)
            .expect("set_hydro_vars iter 1");
    }
    assert_eq!(model.aux.solver_mlt_eval_count, 1);
    assert!(
        model.aux.fixed_gradr_sticky.iter().all(|&b| !b),
        "repeated line-search trials at solver_iter=1 must not latch fixed_gradr_sticky"
    );

    // At Newton iteration 4 (`solver_iter > 3`), radiative faces latch sticky.
    model.aux.solver_mlt_eval_count = 4;
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars iter 4");
    assert_eq!(model.aux.solver_mlt_eval_count, 4);
    assert!(
        model.aux.fixed_gradr_sticky.iter().any(|&b| b),
        "solver_iter=4 (> 3) must latch radiative faces into fixed_gradr_sticky"
    );
}

#[test]
fn test_check_dx_hard_limit_retries_without_positivity_guard() {
    // Upstream `star/private/timestep.f90:580-581, 603-605` tests
    // `delta_X > dX_hard_limit(i)` and `delta_X_div_X > dX_div_X_hard_limit(i)`
    // without requiring `dX_hard_limit > 0` or `X > 0`.
    let mut m = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("1 Msun ZAMS");
    m.doing_first_model_of_run = false;
    m.d_mix.fill(0.0);
    m.mixing_type.fill(flexmesa_const::MixingType::NoMixing);
    m.save_old_values();
    m.dt = 100.0;

    let ic12 = m.iso_names.iter().position(|s| s == "c12").expect("c12");
    m.xa_old[ic12] = 0.02;
    m.xa[ic12] = 0.019; // delta_X = 0.001 > 0.0

    // `dX_hard_limit = 0.0` (< 1.0 so slot is active) must trigger retry.
    m.controls.ts_stop.dx_limit_species[0] = String::from("c12");
    m.controls.ts_stop.dx_limit_min_x[0] = 0.001;
    m.controls.ts_stop.dx_hard_limit[0] = 0.0;
    let (act_dx, _) = timestep_controller(&mut m);
    assert!(
        matches!(act_dx, TimestepAction::Retry(ref msg) if msg == "dX c12 hard limit"),
        "expected dX hard-limit retry when dx_hard_limit == 0.0, got {act_dx:?}"
    );

    // Disable dX hard limit (`>= 1.0`) and set `dX_div_X_hard_limit = 0.0`.
    m.controls.ts_stop.dx_hard_limit[0] = 1.0e99;
    m.controls.ts_stop.dx_div_x_limit_min_x[0] = 0.001;
    m.controls.ts_stop.dx_div_x_hard_limit[0] = 0.0;
    let (act_dxx, _) = timestep_controller(&mut m);
    assert!(
        matches!(act_dxx, TimestepAction::Retry(ref msg) if msg == "dX_div_X c12 hard limit"),
        "expected dX_div_X hard-limit retry when dx_div_x_hard_limit == 0.0, got {act_dxx:?}"
    );
}

#[test]
fn test_gold_solver_iters_timestep_limit_overrides_relax_limit() {
    // Upstream `star/private/timestep.f90:436-440` and `relax.f90:3999-4000`:
    // during relaxation `relax_solver_iters_timestep_limit` replaces
    // `solver_iters_timestep_limit`, and `using_gold_tolerances` overrides that
    // with `gold_solver_iters_timestep_limit`.
    let mut m = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("1 Msun ZAMS");
    m.doing_first_model_of_run = false;
    m.save_old_values();
    m.dt = 100.0;
    m.controls.ts_stop.use_dt_low_pass_controller = false;
    m.controls.ts_stop.min_timestep_factor = 0.0;
    m.controls.ts_stop.max_timestep_factor = 0.0;
    m.controls.ts_stop.solver_iters_timestep_limit = 20;
    m.controls.relax_solver_iters_timestep_limit = 50;
    m.controls.ts_stop.gold_solver_iters_timestep_limit = 5;
    m.aux.doing_relax = true;
    m.aux.step_hist.last_solver_iterations = 10;

    // Without gold tolerances: effective limit = 50 -> ratio = 10/50 - 0.05 = 0.15 <= 1 -> 0.
    m.using_gold_tolerances = false;
    let (act_relax, rep_relax) = timestep_controller(&mut m);
    assert_eq!(act_relax, TimestepAction::KeepGoing);
    assert_eq!(rep_relax.dt_limit_ratio, 0.0);

    // With gold tolerances: gold overrides relax -> effective limit = 5
    // -> ratio = 10/5 - 0.05 = 1.95 (exact closed-form upstream `check_integer_limit`).
    m.using_gold_tolerances = true;
    let (act_gold, rep_gold) = timestep_controller(&mut m);
    assert_eq!(act_gold, TimestepAction::KeepGoing);
    assert!(
        (rep_gold.dt_limit_ratio - 1.95).abs() <= 1.0e-14,
        "expected dt_limit_ratio = 1.95, got {}",
        rep_gold.dt_limit_ratio
    );
    assert!(
        (rep_gold.dt_next - 100.0 * 1.0 / 1.95).abs() <= 1.0e-12,
        "expected dt_next = 100/1.95, got {}",
        rep_gold.dt_next
    );
}
