// SPDX-License-Identifier: GPL-3.0-or-later
//! Integration tests for `#231` `StarExtras` hooks (`other_timestep_limit`,
//! `other_mesh_delta_coeff_factor`, `other_mesh_functions`,
//! `other_build_initial_model`, and `termination_code_string_okay`).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines
)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use flexmesa_const::{FOUR_THIRDS_PI, PI, SECYER};
use flexmesa_math::{exp, log, pow};
use flexmesa_star::adjust_mesh_support::{
    set_delta_gval_max_with_controls, OtherMeshDeltaCoeffFactorHandle,
    OtherMeshDeltaCoeffFactorHook,
};
use flexmesa_star::inlist::InlistBundle;
use flexmesa_star::inlist_runner::{
    install_star_aux_hooks, termination_code_string_okay, ExtrasDirective,
    StarExtras,
};
use flexmesa_star::mesh_functions::{
    num_mesh_functions, OtherMeshFunctionColumn, OtherMeshFunctionsHandle,
    OtherMeshFunctionsHook,
};
use flexmesa_star::{
    get_gval_info, get_or_load_net, init_or_update_model_from_bundle,
    load_zams_model, set_hydro_vars, star_evolve_step_with_extras,
    StarControls, StarError, StarJobControls, StarModel,
};

struct TempTestDir(PathBuf);

impl TempTestDir {
    fn new(label: &str) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "flexmesa_231_hooks_{label}_{}_{n}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("create temp test dir");
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempTestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct ForcedRetryAndLimitExtras {
    calls: AtomicUsize,
}

impl StarExtras for ForcedRetryAndLimitExtras {
    fn other_timestep_limit(
        &mut self,
        model: &mut StarModel,
        _skip_hard_limit: bool,
        _dt: f64,
    ) -> (ExtrasDirective, f64) {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        if n == 0 && model.num_retries == 0 {
            return (ExtrasDirective::Retry, 0.0);
        }
        (ExtrasDirective::KeepGoing, 4.0)
    }
}

#[test]
fn test_other_timestep_limit_retry_and_ratio_scaling() {
    let mut controls = StarControls {
        initial_mass: 1.0,
        initial_z: 0.02,
        zbase: 0.02,
        ..StarControls::default()
    };
    controls.ts_stop.use_other_timestep_limit = true;
    controls.ts_stop.never_skip_hard_limits = true;
    controls.ts_stop.use_dt_low_pass_controller = false;

    let mut model =
        load_zams_model(controls, StarJobControls::default()).expect("zams");
    model.dt = 1.0e4 * SECYER;
    model.dt_next = model.dt;
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    model.save_start_values().expect("save_start_values");
    model.save_old_values();

    let mut extras = ForcedRetryAndLimitExtras {
        calls: AtomicUsize::new(0),
    };

    // Step 1 (`doing_first_model_of_run = true`) bypasses `do_timestep_limits`.
    let rep1 = star_evolve_step_with_extras(
        &mut model,
        false,
        false,
        true,
        &mut extras,
    )
    .expect("step 1");
    assert_eq!(extras.calls.load(Ordering::SeqCst), 0);
    assert_eq!(rep1.retries, 0);

    // Step 2 invokes `other_timestep_limit`: forces 1 retry on try 0, then
    // returns `dt_limit_ratio = 4.0` to clamp `dt_next` below `dt`.
    let rep2 = star_evolve_step_with_extras(
        &mut model,
        false,
        false,
        true,
        &mut extras,
    )
    .expect("step 2");
    assert!(
        extras.calls.load(Ordering::SeqCst) >= 2,
        "expected retry + follow-up call to other_timestep_limit"
    );
    assert!(
        rep2.retries >= 1,
        "forced Retry directive must increment step retries"
    );
    assert!(
        (model.dt_limit_ratio_old - 4.0).abs() < 1.0e-12,
        "expected dt_limit_ratio_old = 4.0 from other_timestep_limit, got {}",
        model.dt_limit_ratio_old
    );
    assert!(
        rep2.dt_next < rep2.dt,
        "dt_limit_ratio = 4.0 must shrink dt_next ({}) below dt ({})",
        rep2.dt_next,
        rep2.dt
    );
}

struct ScaleByQuarterHook;

impl OtherMeshDeltaCoeffFactorHook for ScaleByQuarterHook {
    fn eval_mesh_delta_coeff_factor(
        &self,
        model: &StarModel,
        mesh_delta_coeff_factor: &mut [f64],
    ) -> Result<(), StarError> {
        mesh_delta_coeff_factor[..model.nz].fill(0.75);
        Ok(())
    }
}

struct EmptyOtherMeshFcnsHook;

impl OtherMeshFunctionsHook for EmptyOtherMeshFcnsHook {
    fn eval_other_mesh_functions(
        &self,
        _model: &StarModel,
    ) -> Result<Vec<OtherMeshFunctionColumn>, StarError> {
        Ok(Vec::new())
    }
}

struct AuxMeshExtras;

impl StarExtras for AuxMeshExtras {
    fn extras_controls(
        &mut self,
        model: &mut StarModel,
    ) -> Result<(), StarError> {
        model.aux.other_mesh_delta_coeff_factor = Some(
            OtherMeshDeltaCoeffFactorHandle(Arc::new(ScaleByQuarterHook)),
        );
        model.aux.other_mesh_functions =
            Some(OtherMeshFunctionsHandle(Arc::new(EmptyOtherMeshFcnsHook)));
        install_star_aux_hooks(self, model);
        Ok(())
    }
}

#[test]
fn test_extras_controls_mesh_hooks_survive_install_and_scale_gval() {
    let controls = StarControls {
        initial_mass: 1.0,
        initial_z: 0.02,
        zbase: 0.02,
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(controls, StarJobControls::default()).expect("zams");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");

    let base_n =
        num_mesh_functions(&model).expect("baseline num_mesh_functions");
    let base_gval = set_delta_gval_max_with_controls(
        &model,
        &model.controls.mix_wind,
        &model.xa,
    )
    .expect("baseline delta_gval_max");

    let mut extras = AuxMeshExtras;
    extras.extras_controls(&mut model).expect("extras_controls");
    model.controls.mix_wind.use_other_mesh_functions = true;
    model.controls.mix_wind.use_other_mesh_delta_coeff_factor = true;

    assert!(model.aux.other_mesh_functions.is_some());
    assert!(model.aux.other_mesh_delta_coeff_factor.is_some());
    assert_eq!(
        num_mesh_functions(&model).expect("hook num_mesh_functions"),
        base_n
    );

    let ginfo = get_gval_info(&model, base_n).expect("hook get_gval_info");
    assert_eq!(ginfo.delta_gval_max.len(), model.nz);
    assert!(
        (ginfo.delta_gval_max[0] - 0.75).abs() < 1.0e-12,
        "expected surface delta_gval_max = 0.75, got {}",
        ginfo.delta_gval_max[0]
    );
    for (got, base) in ginfo.delta_gval_max.iter().zip(base_gval.iter()) {
        assert!(*got <= 0.75 + 1.0e-12, "expected <= 0.75, got {got}");
        assert!(*got <= *base + 1.0e-12, "expected {got} <= baseline {base}");
    }
}

struct IsothermalBallExtras;

impl StarExtras for IsothermalBallExtras {
    fn other_build_initial_model(
        &mut self,
        bundle: &InlistBundle,
    ) -> Result<Option<StarModel>, StarError> {
        let net = get_or_load_net("basic.net", Some(&bundle.case_dir))?;
        let nz = usize::try_from(bundle.controls.x_integer_ctrl[0])
            .map_err(|_| StarError::InvalidInput(String::from("bad nz")))?;
        let mstar = bundle.controls.x_ctrl[0];
        let rho_c = bundle.controls.x_ctrl[1];
        let t_bg = bundle.controls.x_ctrl[2];
        let q_bump = bundle.controls.x_ctrl[3];
        let t_bump = bundle.controls.x_ctrl[4];
        let x_c12 = bundle.controls.x_ctrl[5];
        let x_o16 = bundle.controls.x_ctrl[6];

        let mut job = bundle.job.clone();
        job.net_name.clone_from(&net.net_filename);
        let mut controls = bundle.controls.clone();
        controls.initial_y = 0.0;
        controls.v_flag = false;
        controls.u_flag = false;
        controls.rotation_flag = false;

        let mut model = StarModel::new(nz, net.chem_id.clone(), job, controls);
        model.eos_config = bundle.eos_config;
        model.kap_config = bundle.kap_config.clone();
        model.mstar = mstar;
        model.xmstar = mstar;
        model.mstar_start = mstar;
        model.mstar_old = mstar;

        let r0 = pow(3.0 / (4.0 * PI) * mstar / rho_c, 1.0 / 3.0);
        let dr = r0 / (nz as f64);
        let lnd_val = log(rho_c);
        let lnt_bg = log(t_bg);
        let lnt_bump = log(t_bump);
        for k in 0..nz {
            model.lnd[k] = lnd_val;
            model.ln_t[k] = lnt_bg;
            model.ln_r[k] = log(((nz - k) as f64) * dr);
            model.lum[k] = 0.0;
        }
        let mut q = vec![0.0_f64; nz];
        model.dq[nz - 1] = FOUR_THIRDS_PI * dr * dr * dr * rho_c / mstar;
        q[nz - 1] = model.dq[nz - 1];
        for k in (0..nz - 1).rev() {
            let rp = exp(model.ln_r[k]);
            let rm = exp(model.ln_r[k + 1]);
            model.dq[k] =
                FOUR_THIRDS_PI * (rp * rp * rp - rm * rm * rm) * rho_c / mstar;
            q[k] = q[k + 1] + model.dq[k];
        }
        for k in (0..nz).rev() {
            if q[k] < q_bump {
                model.ln_t[k] = lnt_bump;
            }
        }
        let mut xa_uni = vec![0.0_f64; model.species];
        for (j, name) in model.iso_names.iter().enumerate() {
            match name.as_str() {
                "c12" => xa_uni[j] = x_c12,
                "o16" => xa_uni[j] = x_o16,
                _ => {}
            }
        }
        for k in 0..nz {
            model.cell_xa_mut(k).copy_from_slice(&xa_uni);
        }
        model.dt = 1.0e-15;
        model.dt_next = 1.0e-15;
        model.finish_init_model()?;
        Ok(Some(model))
    }
}

#[test]
fn test_other_build_initial_model_precedes_create_initial_and_changes_net() {
    let dir = TempTestDir::new("build_init");
    let inlist_path = dir.path().join("inlist_custom");
    let body = "\
&star_job
  create_initial_model = .true.
  change_net = .true.
  new_net_name = 'approx21.net'
  set_initial_dt = .true.
  seconds_for_initial_dt = 1d-15
/
&eos
  use_Skye = .false.
  use_PC = .false.
/
&kap
  Zbase = 0.02
/
&controls
  use_other_build_initial_model = .true.
  use_zero_pgas_outer_bc = .true.
  use_rsp_l_eqn_outer_bc = .true.
  x_integer_ctrl(1) = 40
  x_ctrl(1) = 10.0
  x_ctrl(2) = 1d10
  x_ctrl(3) = 2d8
  x_ctrl(4) = 0.05
  x_ctrl(5) = 5d9
  x_ctrl(6) = 0.5
  x_ctrl(7) = 0.5
/
";
    fs::write(&inlist_path, body).expect("write custom inlist");
    let bundle =
        InlistBundle::from_file_in_case_dir(dir.path(), "inlist_custom")
            .expect("parse inlist");
    let mut extras = IsothermalBallExtras;
    let model = init_or_update_model_from_bundle(&bundle, None, &mut extras)
        .expect("build custom initial model + change_net");
    assert_eq!(model.nz, 40);
    assert_eq!(model.job.net_name, "approx21.net");
    assert_eq!(model.species, 21);
    assert!(exp(model.ln_t[model.nz - 1]) > 4.0e9);
    assert!((exp(model.ln_t[0]) - 2.0e8).abs() < 1.0e2);
}

#[test]
fn test_cepheid_stale_required_termination_contract() {
    let req = [String::from("star_species_mass_min_limit")];
    // Upstream `star/test_suite/5M_cepheid_blue_loop/src/run_star_extras.f90:125-128` sets `extras_check_model = terminate`
    // with `s% termination_code == 0`, so `termination_code_string_okay` returns
    // `true` for `""` (`run_star_support.f90:831`) and `false` for explicit
    // `"extras_check_model"`.
    assert!(termination_code_string_okay(&req, ""));
    assert!(!termination_code_string_okay(&req, "extras_check_model"));
    let req_extras = [String::from("extras_check_model")];
    assert!(termination_code_string_okay(
        &req_extras,
        "extras_check_model"
    ));
}
