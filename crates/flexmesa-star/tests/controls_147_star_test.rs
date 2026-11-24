// SPDX-License-Identifier: GPL-3.0-or-later
//! Issue `#147` strict parsing, `.photo` roundtrip, and MESA `26.4.1`
//! behavioral tests for `&controls` and `&star_job` parity controls.

#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use flexmesa_const::{L_SUN, M_SUN, SECDAY, SECYER};
use flexmesa_star::adjust_mesh_support::{
    get_gval_info, set_delta_gval_max_with_controls,
    OtherMeshDeltaCoeffFactorHandle, OtherMeshDeltaCoeffFactorHook,
};
use flexmesa_star::controls::{StarControls, StarError, StarJobControls};
use flexmesa_star::controls_output::OutputControls;
use flexmesa_star::inlist::InlistBundle;
use flexmesa_star::inlist_inventory::{
    CONTROLS_SUPPORTED, CONTROLS_UNUSED, STAR_JOB_SUPPORTED, STAR_JOB_UNUSED,
};
use flexmesa_star::inlist_runner::{
    install_star_aux_hooks, virial_thm_rel_err_exceeds_warn_limit,
    DefaultStarExtras, StarExtras,
};
use flexmesa_star::mesh_functions::{
    num_mesh_functions, set_mesh_function_data, OtherMeshFunctionColumn,
    OtherMeshFunctionsHandle, OtherMeshFunctionsHook,
};
use flexmesa_star::micro::check_tau_with_report_ierr;
use flexmesa_star::micro_net::high_temp_rate_warning_zone;
use flexmesa_star::model_io::{read_mod_str, write_mod_string};
use flexmesa_star::photo_io::{load_photo_bytes, write_photo_bytes};
use flexmesa_star::pulse::export_fgong;
use flexmesa_star::pulse_exporters::export_pulse_by_format;
use flexmesa_star::star_job_init::init_or_update_model_from_bundle;
use flexmesa_star::{
    apply_step_pre_physics, load_zams_model,
    record_start_of_step_energy_integrals, set_hydro_vars, star_evolve_step,
    yrs_for_init_timestep, StarModel,
};

struct TempCaseGuard {
    root: PathBuf,
}

impl TempCaseGuard {
    fn new(tag: &str) -> Self {
        static SEQ: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        let id = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "flexmesa_star_147_controls_{tag}_{}_{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("create temp dir");
        Self { root }
    }
}

impl Drop for TempCaseGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn default_zams() -> StarModel {
    load_zams_model(
        StarControls {
            initial_mass: 1.0,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("load 1 Msun ZAMS")
}

struct TestMeshFunctionsHook;

impl OtherMeshFunctionsHook for TestMeshFunctionsHook {
    fn eval_other_mesh_functions(
        &self,
        model: &StarModel,
    ) -> Result<Vec<OtherMeshFunctionColumn>, StarError> {
        Ok(vec![OtherMeshFunctionColumn {
            name: String::from("test_q_fcn"),
            is_xa_function: false,
            vals: model.q.iter().map(|&q| 12.0 * q).collect(),
        }])
    }
}

struct TestDeltaFactorHook(f64);

impl OtherMeshDeltaCoeffFactorHook for TestDeltaFactorHook {
    fn eval_mesh_delta_coeff_factor(
        &self,
        _model: &StarModel,
        mesh_delta_coeff_factor: &mut [f64],
    ) -> Result<(), StarError> {
        for v in mesh_delta_coeff_factor.iter_mut() {
            *v *= self.0;
        }
        Ok(())
    }
}

struct TestMeshExtras(f64);

impl StarExtras for TestMeshExtras {
    fn other_mesh_functions_handle(&self) -> Option<OtherMeshFunctionsHandle> {
        Some(OtherMeshFunctionsHandle(Arc::new(TestMeshFunctionsHook)))
    }

    fn other_mesh_delta_coeff_factor_handle(
        &self,
    ) -> Option<OtherMeshDeltaCoeffFactorHandle> {
        Some(OtherMeshDeltaCoeffFactorHandle(Arc::new(
            TestDeltaFactorHook(self.0),
        )))
    }
}

#[test]
fn test_147_mesh_hooks_parsing_extras_installation_and_failures() {
    let tmp = TempCaseGuard::new("mesh_hooks");
    let path = tmp.root.join("inlist_mesh_hooks");
    fs::write(
        &path,
        "&controls\n  initial_mass = 1.0d0\n  initial_z = 0.02d0\n  \
         use_other_mesh_functions = .true.\n  \
         use_other_mesh_delta_coeff_factor = .true.\n/\n",
    )
    .expect("write inlist");
    let bundle = InlistBundle::from_file(&path).expect("strict parse");
    assert!(bundle.controls.mix_wind.use_other_mesh_functions);
    assert!(bundle.controls.mix_wind.use_other_mesh_delta_coeff_factor);

    let base_model = default_zams();
    let n_base =
        num_mesh_functions(&base_model).expect("base mesh function count");

    let mut no_hook =
        load_zams_model(bundle.controls.clone(), bundle.job.clone())
            .expect("zams");
    assert!(matches!(
        num_mesh_functions(&no_hook),
        Err(StarError::InvalidInput(_))
    ));
    assert!(matches!(
        set_delta_gval_max_with_controls(
            &no_hook,
            &no_hook.controls.mix_wind,
            &no_hook.xa,
        ),
        Err(StarError::InvalidInput(_))
    ));

    let extras = TestMeshExtras(0.5);
    install_star_aux_hooks(&extras, &mut no_hook);
    let n_with = num_mesh_functions(&no_hook).expect("count with hook");
    assert_eq!(n_with, n_base + 1);
    let data = set_mesh_function_data(&no_hook, n_with).expect("mesh data");
    assert!(data.names.iter().any(|n| n == "test_q_fcn"));
    let col_idx = data
        .names
        .iter()
        .position(|n| n == "test_q_fcn")
        .expect("custom column");
    assert!((data.vals[col_idx] - 12.0 * no_hook.q[0]).abs() < 1.0e-12);

    let ginfo = get_gval_info(&no_hook, n_with).expect("get_gval_info");
    assert_eq!(ginfo.fcns.vals.len(), n_with * no_hook.nz);
    let dg = set_delta_gval_max_with_controls(
        &no_hook,
        &no_hook.controls.mix_wind,
        &no_hook.xa,
    )
    .expect("delta_gval_max with factor hook");
    let mut base_check = base_model;
    base_check
        .controls
        .mix_wind
        .use_other_mesh_delta_coeff_factor = false;
    let dg_base = set_delta_gval_max_with_controls(
        &base_check,
        &base_check.controls.mix_wind,
        &base_check.xa,
    )
    .expect("base delta_gval_max");
    assert!((dg[0] - 0.5 * dg_base[0]).abs() < 1.0e-12);

    let photo = write_photo_bytes(&no_hook);
    let reloaded = load_photo_bytes(&photo).expect("load photo");
    assert!(reloaded.controls.mix_wind.use_other_mesh_functions);
    assert!(reloaded.controls.mix_wind.use_other_mesh_delta_coeff_factor);

    let bad_path = tmp.root.join("inlist_bad_mesh_bool");
    fs::write(
        &bad_path,
        "&controls\n  use_other_mesh_functions = 'yes'\n/\n",
    )
    .expect("write bad bool");
    assert!(matches!(
        InlistBundle::from_file(&bad_path),
        Err(StarError::InvalidInput(_))
    ));
}

#[test]
fn test_147_virial_thm_rel_err_warning_gates_and_roundtrip() {
    assert!(
        (OutputControls::default().warn_when_large_virial_thm_rel_err - 0.01)
            .abs()
            < 1.0e-15
    );

    let tmp = TempCaseGuard::new("virial");
    let path = tmp.root.join("inlist_virial");
    fs::write(
        &path,
        "&controls\n  initial_mass = 1.0d0\n  initial_z = 0.02d0\n  \
         warn_when_large_virial_thm_rel_err = 0.05d0\n/\n",
    )
    .expect("write inlist");
    let bundle = InlistBundle::from_file(&path).expect("strict parse");
    assert!(
        (bundle.controls.output.warn_when_large_virial_thm_rel_err - 0.05)
            .abs()
            < 1.0e-15
    );

    let mut model =
        load_zams_model(bundle.controls.clone(), bundle.job.clone())
            .expect("zams");
    model.aux.step_hist.virial_thm_p_avg = 1.0e48;
    model.aux.step_hist.total_gravitational_energy_end = -1.1e48;
    let err = virial_thm_rel_err_exceeds_warn_limit(&model)
        .expect("0.10 > 0.05 threshold");
    assert!((err - 0.1).abs() < 1.0e-12);

    model.controls.output.warn_when_large_virial_thm_rel_err = 0.2;
    assert_eq!(virial_thm_rel_err_exceeds_warn_limit(&model), None);

    model.controls.output.warn_when_large_virial_thm_rel_err = 0.05;
    // Guard each mutating flag matches `run_star_support.f90:707-715`.
    let mut rot = model.clone();
    rot.controls.rotation_flag = true;
    assert_eq!(virial_thm_rel_err_exceeds_warn_limit(&rot), None);

    let mut u_on = model.clone();
    u_on.controls.u_flag = true;
    assert_eq!(virial_thm_rel_err_exceeds_warn_limit(&u_on), None);

    let mut v_on = model.clone();
    v_on.controls.v_flag = true;
    assert_eq!(virial_thm_rel_err_exceeds_warn_limit(&v_on), None);

    let mut mc_on = model.clone();
    mc_on.controls.use_mass_corrections = true;
    assert_eq!(virial_thm_rel_err_exceeds_warn_limit(&mc_on), None);

    let mut core_m = model.clone();
    core_m.m_center = 0.1 * M_SUN;
    assert_eq!(virial_thm_rel_err_exceeds_warn_limit(&core_m), None);

    let mut wind = model.clone();
    wind.mstar_dot = -1.0e20;
    assert_eq!(virial_thm_rel_err_exceeds_warn_limit(&wind), None);

    let photo = write_photo_bytes(&model);
    let reloaded = load_photo_bytes(&photo).expect("load photo");
    assert!(
        (reloaded.controls.output.warn_when_large_virial_thm_rel_err - 0.05)
            .abs()
            < 1.0e-15
    );

    let bad = tmp.root.join("inlist_bad_virial");
    fs::write(
        &bad,
        "&controls\n  warn_when_large_virial_thm_rel_err = 'huge'\n/\n",
    )
    .expect("write bad");
    assert!(matches!(
        InlistBundle::from_file(&bad),
        Err(StarError::InvalidInput(_))
    ));
}

#[test]
fn test_147_limit_initial_dt_and_set_initial_center_behavior() {
    assert!(!StarJobControls::default().limit_initial_dt);
    let tmp = TempCaseGuard::new("init_dt");

    let cap_inlist = tmp.root.join("inlist_cap_dt");
    fs::write(
        &cap_inlist,
        "&star_job\n  limit_initial_dt = .true.\n  \
         years_for_initial_dt = 0.5d0\n  \
         set_initial_v_center = .true.\n  new_v_center = 1.25d6\n  \
         set_initial_l_center = .true.\n  new_l_center = 2.5d0\n/\n\
         &kap\n  Zbase = 0.02d0\n/\n\
         &controls\n  initial_mass = 1.0d0\n  initial_z = 0.02d0\n/\n",
    )
    .expect("write cap inlist");
    let bundle_cap = InlistBundle::from_file(&cap_inlist).expect("parse cap");
    assert!(bundle_cap.job.limit_initial_dt);
    let mut extras = DefaultStarExtras;
    let mut large_dt_model = default_zams();
    large_dt_model.dt_next = 100.0 * SECYER;
    let capped = init_or_update_model_from_bundle(
        &bundle_cap,
        Some(large_dt_model),
        &mut extras,
    )
    .expect("init capped model");
    assert!((capped.dt_next - 0.5 * SECYER).abs() < 1.0e-6);
    assert!((capped.v_center - 1.25e6).abs() < 1.0e-6);
    assert!((capped.l_center - 2.5 * L_SUN).abs() < 1.0e-6 * L_SUN);

    // Unlike `set_initial_dt`, `limit_initial_dt` must never enlarge a smaller `dt_next`.
    let big_cap_inlist = tmp.root.join("inlist_big_cap_dt");
    fs::write(
        &big_cap_inlist,
        "&star_job\n  limit_initial_dt = .true.\n  \
         seconds_for_initial_dt = 1.0d12\n/\n\
         &kap\n  Zbase = 0.02d0\n/\n\
         &controls\n  initial_mass = 1.0d0\n  initial_z = 0.02d0\n/\n",
    )
    .expect("write big cap");
    let bundle_big = InlistBundle::from_file(&big_cap_inlist).expect("parse");
    let mut small_dt_model = default_zams();
    small_dt_model.dt_next = 500.0;
    let preserved = init_or_update_model_from_bundle(
        &bundle_big,
        Some(small_dt_model),
        &mut extras,
    )
    .expect("init preserved dt");
    assert!((preserved.dt_next - 500.0).abs() < 1.0e-12);

    let photo = write_photo_bytes(&capped);
    let reloaded = load_photo_bytes(&photo).expect("load photo");
    assert!(reloaded.job.limit_initial_dt);

    let bad = tmp.root.join("inlist_bad_limit_dt");
    fs::write(&bad, "&star_job\n  limit_initial_dt = 7\n/\n")
        .expect("write bad");
    assert!(matches!(
        InlistBundle::from_file(&bad),
        Err(StarError::InvalidInput(_))
    ));
}

#[test]
fn test_147_inject_extra_ergs_sec_analytic_envelope_and_limits() {
    let mw_def = StarControls::default().mix_wind;
    assert_eq!(mw_def.inject_extra_ergs_sec, 0.0);
    assert_eq!(mw_def.base_of_inject_extra_ergs_sec, 0.0);
    assert_eq!(mw_def.total_mass_for_inject_extra_ergs_sec, 0.0);
    assert_eq!(mw_def.start_time_for_inject_extra_ergs_sec, -1.0e99);
    assert_eq!(mw_def.duration_for_inject_extra_ergs_sec, -1.0);
    assert_eq!(mw_def.inject_until_reach_model_with_total_energy, 1.0e99);

    let tmp = TempCaseGuard::new("inject_ergs");
    let path = tmp.root.join("inlist_inject");
    fs::write(
        &path,
        "&kap\n  Zbase = 0.02d0\n/\n\
         &controls\n  initial_mass = 1.0d0\n  initial_z = 0.02d0\n  \
         inject_extra_ergs_sec = 0.0d0\n  \
         base_of_inject_extra_ergs_sec = -1.0d0\n  \
         total_mass_for_inject_extra_ergs_sec = 0.25d0\n  \
         start_time_for_inject_extra_ergs_sec = 10.0d0\n  \
         duration_for_inject_extra_ergs_sec = 100.0d0\n  \
         inject_until_reach_model_with_total_energy = 1.0d50\n/\n",
    )
    .expect("write inject inlist");
    let bundle = InlistBundle::from_file(&path).expect("strict parse");
    let mut model =
        load_zams_model(bundle.controls.clone(), bundle.job.clone())
            .expect("zams");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    model.aux.need_to_setvars = false;
    record_start_of_step_energy_integrals(&mut model);
    let e_init = model.aux.step_hist.total_energy_initial;
    let target_ergs = 1.0e50 - e_init;

    // 1. Before `start_time`: zero heating.
    model.time_old_sec = 0.0;
    model.time_sec = 5.0;
    apply_step_pre_physics(&mut model, 5.0).expect("before start");
    assert!(model.aux.extra_heat.iter().all(|&x| x == 0.0));

    // 2. Partial overlap step across `start_time = 10.0`: dt = 15, [5, 20].
    model.time_old_sec = 5.0;
    model.time_sec = 20.0;
    apply_step_pre_physics(&mut model, 15.0).expect("partial overlap");
    let expected_rate = target_ergs / 100.0;
    assert!(
        ((model.controls.mix_wind.inject_extra_ergs_sec - expected_rate)
            / expected_rate)
            .abs()
            < 1.0e-12
    );
    let total_lum: f64 = model
        .dm
        .iter()
        .zip(&model.aux.extra_heat)
        .map(|(&dm, &eps)| dm * eps)
        .sum();
    let expected_step_lum = expected_rate * (10.0 / 15.0);
    assert!(
        ((total_lum - expected_step_lum) / expected_step_lum).abs() < 1.0e-12,
        "got {total_lum:.12e} vs {expected_step_lum:.12e}"
    );
    // Cells wholly outside `qmax = 0.25` stay zero.
    assert!((0..model.nz - 1)
        .all(|k| model.q[k + 1] < 0.25 || model.aux.extra_heat[k] == 0.0));

    // 3. Remaining-energy cap: `left_to_inject = 2.0e48 < rate * dt`.
    model.time_old_sec = 20.0;
    model.time_sec = 30.0;
    model.aux.step_hist.total_energy_old = 1.0e50 - 2.0e48;
    apply_step_pre_physics(&mut model, 10.0).expect("left_to_inject cap");
    let capped_lum: f64 = model
        .dm
        .iter()
        .zip(&model.aux.extra_heat)
        .map(|(&dm, &eps)| dm * eps)
        .sum();
    assert!(((capped_lum - 2.0e47) / 2.0e47).abs() < 1.0e-12);

    // 4. After `end_time = 110`: leftover heating clears cleanly to 0.
    model.time_old_sec = 120.0;
    model.time_sec = 130.0;
    apply_step_pre_physics(&mut model, 10.0).expect("after end_time");
    assert!(model.aux.extra_heat.iter().all(|&x| x == 0.0));
}

#[test]
fn test_147_inject_extra_ergs_sec_fallback_region_error_and_roundtrip() {
    let mut model = default_zams();
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    model.aux.need_to_setvars = false;
    record_start_of_step_energy_integrals(&mut model);
    let e_init = model.aux.step_hist.total_energy_initial;
    model
        .controls
        .mix_wind
        .inject_until_reach_model_with_total_energy = 1.0e50;
    model.controls.mix_wind.total_mass_for_inject_extra_ergs_sec = 0.25;
    model.controls.mix_wind.base_of_inject_extra_ergs_sec = -1.0;
    model.controls.mix_wind.duration_for_inject_extra_ergs_sec = -1.0;
    model.controls.max_age_in_seconds = None;
    model.controls.ts_stop.max_age_in_days = 10.0;
    model.controls.mix_wind.start_time_for_inject_extra_ergs_sec = 0.0;
    model.aux.step_hist.total_energy_old = e_init;
    model.time_old_sec = 0.0;
    model.time_sec = SECDAY;
    apply_step_pre_physics(&mut model, SECDAY).expect("max_age_in_days");
    let target_ergs = 1.0e50 - e_init;
    let expected_days_rate = target_ergs / (10.0 * SECDAY);
    assert!(
        ((model.controls.mix_wind.inject_extra_ergs_sec - expected_days_rate)
            / expected_days_rate)
            .abs()
            < 1.0e-12
    );

    // Empty injection region (`base_of_inject_extra_ergs_sec > M_star`) fails.
    model.controls.mix_wind.base_of_inject_extra_ergs_sec = 5.0;
    assert!(matches!(
        apply_step_pre_physics(&mut model, SECDAY),
        Err(StarError::MicrophysicsError(_))
    ));

    let photo = write_photo_bytes(&model);
    let reloaded = load_photo_bytes(&photo).expect("load photo");
    assert!(
        (reloaded.controls.mix_wind.base_of_inject_extra_ergs_sec - 5.0).abs()
            < 1.0e-15
    );
    assert!(
        (reloaded
            .controls
            .mix_wind
            .total_mass_for_inject_extra_ergs_sec
            - 0.25)
            .abs()
            < 1.0e-15
    );
}

#[test]
fn test_147_add_atmosphere_to_pulse_data_fgong_and_roundtrip() {
    assert!(!StarControls::default().add_atmosphere_to_pulse_data);
    let tmp = TempCaseGuard::new("pulse_atm");
    let path = tmp.root.join("inlist_pulse_atm");
    fs::write(
        &path,
        "&controls\n  initial_mass = 1.0d0\n  initial_z = 0.02d0\n  \
         atm_option = 'T_tau'\n  atm_T_tau_relation = 'Trampedach_solar'\n  \
         atm_T_tau_opacity = 'varying'\n  pulse_data_format = 'FGONG'\n  \
         add_atmosphere_to_pulse_data = .true.\n/\n",
    )
    .expect("write inlist");
    let bundle = InlistBundle::from_file(&path).expect("strict parse");
    assert!(bundle.controls.add_atmosphere_to_pulse_data);

    let model = load_zams_model(bundle.controls.clone(), bundle.job.clone())
        .expect("zams");
    let with_atm =
        export_pulse_by_format(&model, "FGONG", true).expect("fgong via ctrl");
    let direct_atm =
        export_fgong(&model, true, false, true).expect("fgong direct atm");
    assert_eq!(with_atm, direct_atm);

    let mut no_atm_model = model.clone();
    no_atm_model.controls.add_atmosphere_to_pulse_data = false;
    let without_atm =
        export_pulse_by_format(&no_atm_model, "FGONG", true).expect("no atm");
    assert!(with_atm.len() > without_atm.len());

    // Enabling `add_atmosphere_to_pulse_data` on unported GYRE/OSC atm must fail loudly.
    assert!(matches!(
        export_pulse_by_format(&model, "GYRE", true),
        Err(StarError::InvalidInput(_))
    ));

    let photo = write_photo_bytes(&model);
    let reloaded = load_photo_bytes(&photo).expect("load photo");
    assert!(reloaded.controls.add_atmosphere_to_pulse_data);
}

#[test]
fn test_147_report_ierr_and_warn_rates_for_high_temp_behavior() {
    let def_out = OutputControls::default();
    assert!(!def_out.report_ierr);
    assert!(def_out.warn_rates_for_high_temp);

    let tmp = TempCaseGuard::new("ierr_rates");
    let path = tmp.root.join("inlist_ierr_rates");
    fs::write(
        &path,
        "&controls\n  initial_mass = 1.0d0\n  initial_z = 0.02d0\n  \
         report_ierr = .true.\n  warn_rates_for_high_temp = .false.\n/\n",
    )
    .expect("write inlist");
    let bundle = InlistBundle::from_file(&path).expect("strict parse");
    assert!(bundle.controls.output.report_ierr);
    assert!(!bundle.controls.output.warn_rates_for_high_temp);

    let mut model =
        load_zams_model(bundle.controls.clone(), bundle.job.clone())
            .expect("zams");
    check_tau_with_report_ierr(&model).expect("valid tau passes");

    let mut bad_tau = model.clone();
    bad_tau.rmid[0] = 0.0;
    assert!(matches!(
        check_tau_with_report_ierr(&bad_tau),
        Err(StarError::MicrophysicsError(_))
    ));
    bad_tau.controls.output.report_ierr = false;
    assert!(matches!(
        check_tau_with_report_ierr(&bad_tau),
        Err(StarError::MicrophysicsError(_))
    ));

    // `warn_rates_for_high_temp`: respects threshold, disable flag, `max_logT_for_net`,
    // and `doing_struct_burn_mix` suppression.
    model.controls.max_logt_for_net = 11.0;
    model.ln_t[0] =
        (flexmesa_net::MAX_SAFE_LOGT_FOR_RATES + 0.05) * flexmesa_const::LN10;
    model.t[0] = flexmesa_math::exp(model.ln_t[0]);
    assert_eq!(high_temp_rate_warning_zone(&model), None);

    model.controls.output.warn_rates_for_high_temp = true;
    let hit = high_temp_rate_warning_zone(&model)
        .expect("zone above threshold warns");
    assert_eq!(hit.0, 0);
    assert!(
        (hit.1 - (flexmesa_net::MAX_SAFE_LOGT_FOR_RATES + 0.05)).abs()
            < 1.0e-12
    );

    model.aux.doing_struct_burn_mix = true;
    assert_eq!(high_temp_rate_warning_zone(&model), None);
    model.aux.doing_struct_burn_mix = false;

    model.controls.max_logt_for_net = 10.2;
    assert_eq!(high_temp_rate_warning_zone(&model), None);

    let photo = write_photo_bytes(&model);
    let reloaded = load_photo_bytes(&photo).expect("load photo");
    assert!(reloaded.controls.output.report_ierr);
    assert!(reloaded.controls.output.warn_rates_for_high_temp);
}

#[test]
fn test_147_inventory_parity_and_excluded_keys_still_rejected() {
    for key in [
        "limit_initial_dt",
        "set_initial_l_center",
        "set_initial_v_center",
    ] {
        assert!(STAR_JOB_SUPPORTED.contains(&key), "{key} missing");
        assert!(!STAR_JOB_UNUSED.contains(&key), "{key} still in unused");
    }
    for key in [
        "add_atmosphere_to_pulse_data",
        "base_of_inject_extra_ergs_sec",
        "duration_for_inject_extra_ergs_sec",
        "inject_extra_ergs_sec",
        "inject_until_reach_model_with_total_energy",
        "report_ierr",
        "start_time_for_inject_extra_ergs_sec",
        "total_mass_for_inject_extra_ergs_sec",
        "use_other_mesh_delta_coeff_factor",
        "use_other_mesh_functions",
        "warn_rates_for_high_temp",
        "warn_when_large_virial_thm_rel_err",
    ] {
        assert!(CONTROLS_SUPPORTED.contains(&key), "{key} missing");
        assert!(!CONTROLS_UNUSED.contains(&key), "{key} still in unused");
    }

    // Keys outside our crate/file scope must remain strictly rejected.
    let tmp = TempCaseGuard::new("excluded");
    for (idx, body) in [
        "&controls\n  max_safe_logT_for_rates = 10.5d0\n/\n",
        "&star_job\n  electron_coulomb_corrections = 'Itoh2002'\n/\n",
        "&star_job\n  ion_coulomb_corrections = 'PCR2009'\n/\n",
        "&star_job\n  new_w_div_wc_flag = .true.\n/\n",
    ]
    .iter()
    .enumerate()
    {
        let p = tmp.root.join(format!("inlist_excl_{idx}"));
        fs::write(&p, body).expect("write");
        assert!(
            InlistBundle::from_file(&p).is_err(),
            "excluded key must fail strict validation: {body}"
        );
    }
}

#[test]
fn test_147_dt_next_unclipped_mod_save_and_init_fallback() {
    let mut model = default_zams();
    assert_eq!(model.aux.step_hist.dt_next_unclipped, -1.0);

    // Take one step landing exactly on `max_age` so `dt_next` clips to `0`
    // while `dt_next_unclipped` retains the pre-clip timestep (`evolve.f90:2077`).
    let step_yrs = 100.0;
    model.dt_next = step_yrs * SECYER;
    model.controls.max_age = model.star_age + step_yrs;
    let rep = star_evolve_step(&mut model, false, false, true)
        .expect("step to max_age");
    assert_eq!(rep.dt_next, 0.0);
    assert_eq!(model.dt_next, 0.0);
    let unclipped = model.aux.step_hist.dt_next_unclipped;
    assert!(
        unclipped > 0.0,
        "expected positive dt_next_unclipped before max_age clip, got {unclipped}"
    );

    // 1. `.mod` export writes `dt_next_unclipped` rather than clipped `0`
    // (`write_model.f90:229`).
    let mod_text = write_mod_string(&model).expect("write .mod");
    let from_mod =
        read_mod_str(&mod_text, model.controls.clone(), model.job.clone())
            .expect("read .mod");
    assert!(
        (from_mod.dt_next - unclipped).abs() <= 1.0e-12 * unclipped,
        "expected saved .mod dt_next={unclipped:.12e}, got {:.12e}",
        from_mod.dt_next
    );

    // 2. `.photo` roundtrip preserves `dt_next_unclipped`.
    let photo = write_photo_bytes(&model);
    let reloaded = load_photo_bytes(&photo).expect("load photo");
    assert!(
        (reloaded.aux.step_hist.dt_next_unclipped - unclipped).abs() <= 1.0e-15
    );

    // 3. Multi-stage in-memory handoff restores `dt_next_unclipped` and resets
    // `dt_next_unclipped` to `-1` (`init.f90:558`).
    let tmp = TempCaseGuard::new("dt_unclipped");
    let s2_inlist = tmp.root.join("inlist_stage2");
    fs::write(
        &s2_inlist,
        "&kap\n  Zbase = 0.02d0\n/\n&controls\n  initial_mass = 1.0d0\n  initial_z = 0.02d0\n/\n",
    )
    .expect("write s2");
    let s2_bundle = InlistBundle::from_file(&s2_inlist).expect("parse s2");
    let mut extras = DefaultStarExtras;
    let carried = init_or_update_model_from_bundle(
        &s2_bundle,
        Some(model.clone()),
        &mut extras,
    )
    .expect("carry over clipped model");
    assert!((carried.dt_next - unclipped).abs() <= 1.0e-12 * unclipped);
    assert_eq!(carried.aux.step_hist.dt_next_unclipped, -1.0);

    // 4. Saved `.mod` with `dt_next <= 0` falls back to `yrs_for_init_timestep`
    // (`init.f90:932, 1015-1016`).
    let mut replaced_zero = false;
    let zero_mod: String = mod_text
        .lines()
        .map(|line| {
            if line.trim_start().starts_with("dt_next (seconds)") {
                replaced_zero = true;
                "    dt_next (seconds)       0.0000000000000000E+00"
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    assert!(replaced_zero, "dt_next (seconds) line missing in .mod");
    fs::write(tmp.root.join("zero_dt.mod"), zero_mod).expect("write zero mod");
    let load_zero_inlist = tmp.root.join("inlist_load_zero");
    fs::write(
        &load_zero_inlist,
        "&star_job\n  load_saved_model = .true.\n  load_model_filename = 'zero_dt.mod'\n/\n\
         &kap\n  Zbase = 0.02d0\n/\n\
         &controls\n  initial_mass = 1.0d0\n  initial_z = 0.02d0\n/\n",
    )
    .expect("write zero load inlist");
    let zero_bundle =
        InlistBundle::from_file(&load_zero_inlist).expect("parse zero load");
    let from_zero_mod =
        init_or_update_model_from_bundle(&zero_bundle, None, &mut extras)
            .expect("init zero dt mod");
    let expected_fallback =
        yrs_for_init_timestep(from_zero_mod.controls.initial_mass) * SECYER;
    assert!(
        (from_zero_mod.dt_next - expected_fallback).abs()
            <= 1.0e-15 * expected_fallback
    );

    // 5. Existing model with both `dt_next <= 0` and `dt_next_unclipped <= 0`
    // also applies `yrs_for_init_timestep` fallback.
    let mut no_dt_model = model;
    no_dt_model.dt_next = 0.0;
    no_dt_model.aux.step_hist.dt_next_unclipped = -1.0;
    let fallback = init_or_update_model_from_bundle(
        &s2_bundle,
        Some(no_dt_model),
        &mut extras,
    )
    .expect("init fallback existing model");
    assert!(
        (fallback.dt_next - expected_fallback).abs()
            <= 1.0e-15 * expected_fallback
    );
}
