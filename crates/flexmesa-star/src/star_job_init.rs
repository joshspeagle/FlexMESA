// SPDX-License-Identifier: GPL-3.0-or-later
//! `&star_job` model initialization, network switching, composition/mass/core
//! relaxation, and rotational spin-up handlers (`init_or_update_model_from_bundle`,
//! corresponding to `star/private/init.f90` and `star/private/relax.f90`).

use std::fs;
use std::path::Path;

use flexmesa_const::{M_SUN, R_SUN, SECYER, STANDARD_CGRAV};

use crate::controls::{StarControls, StarError, StarJobControls, VarLayout};
use crate::job_controls::{AtmOption, AtmTTauOpacity, AtmTTauRelation};
use crate::hydro_vars::set_hydro_vars;
use crate::inlist::InlistBundle;
use crate::inlist_runner::StarExtras;
use crate::micro::{resolve_model_net, NetSearchDirGuard};
use crate::model::StarModel;
use crate::model_io::read_mod_str_with_dir;
use crate::relax::{
    apply_uniform_rotation, excise_core_by_temperature,
    set_uniform_initial_composition,
};
use crate::relax_comp::{parse_composition_profile_str, set_y, set_z};
use crate::relax_driver::{
    RelaxOmegaKind, do_relax_num_steps, do_relax_to_radiative_core,
    do_relax_uniform_omega,
};
use crate::relax_mass_comp::{
    do_relax_composition, do_relax_core, do_relax_irradiation,
    do_relax_m_center, do_relax_mass, do_relax_mass_scale, do_relax_tau_factor,
    do_relax_y, do_relax_z,
};
use crate::zams::load_zams_model;

/// Resolves a `.mod` model file path against `case_dir`.
pub(crate) fn resolve_model_path(
    case_dir: &Path,
    model_name: &str,
) -> Option<std::path::PathBuf> {
    let direct = case_dir.join(model_name);
    if direct.exists() {
        return Some(direct);
    }
    if let Some(fname) = Path::new(model_name).file_name() {
        let local_fname = case_dir.join(fname);
        if local_fname.exists() {
            return Some(local_fname);
        }
    }
    None
}

/// Loads `model_name` from `case_dir`, failing loudly if missing, unreadable,
/// or malformed.
fn load_saved_mod(
    case_dir: &Path,
    model_name: &str,
    controls: StarControls,
    job: StarJobControls,
) -> Result<StarModel, StarError> {
    let trimmed = model_name.trim();
    if trimmed.is_empty() {
        return Err(StarError::InvalidInput(String::from(
            "load_saved_model = true requires a non-empty load_model_filename",
        )));
    }
    let mod_path = resolve_model_path(case_dir, trimmed).ok_or_else(|| {
        StarError::IoError(format!(
            "missing saved model '{trimmed}' in '{}'",
            case_dir.display()
        ))
    })?;
    let text = fs::read_to_string(&mod_path).map_err(|e| {
        StarError::IoError(format!(
            "failed to read saved model {}: {e}",
            mod_path.display()
        ))
    })?;
    read_mod_str_with_dir(&text, controls, job, Some(case_dir))
}

/// `finish_load_model` (`star/private/read_model.f90:69-205`) for a model
/// read from a `.mod` file (`restart = .false.`).
///
/// Mesh quantities (`set_qs`, `set_m_and_dm`, `set_m_grav_and_grav`,
/// `set_dm_bar`) come from `sync_mesh_and_pack`; the arrays upstream clears at
/// `read_model.f90:131-149` are freshly zeroed by the reader. Upstream's
/// `s% total_angular_momentum` (`:170`) and `do_report` (`:197`) are derived
/// quantities that FlexMESA evaluates on demand from the model arrays
/// (`report_utils`), so no stored state corresponds to them.
///
/// # Errors
/// Returns [`StarError::InvalidInput`] on paths whose callees are not ported
/// here (`RSP_setup_part1` / `RSP_setup_part2`, `set_viscosity_vars_TDC`),
/// and propagates `set_vars` and brunt errors.
fn finish_load_model(
    model: &mut StarModel,
    extras: &mut dyn StarExtras,
) -> Result<(), StarError> {
    // `if (.not. restart) s% have_ST_start_info = .false.` (`read_model.f90:139`).
    model.aux.st_start.have_st_start_info = false;
    // `RSP_setup_part1` / `RSP_setup_part2` (`read_model.f90:150-156,
    // 175-181`): an `RSP` model is set up by `rsp_driver` instead.
    if model.controls.rsp_flag {
        return Err(StarError::InvalidInput(String::from(
            "finish_load_model: RSP_setup_part1 is not ported on the load_saved_model path (star/private/read_model.f90:150-156)",
        )));
    }
    // `set_viscosity_vars_TDC` (`star/private/tdc_hydro.f90:40-89`), called
    // at `read_model.f90:164-167`, is not ported.
    if model.controls.tdc_alpha_m > 0.0
        && model.controls.mlt_option == flexmesa_turb::MltOption::Tdc
        && !(model.controls.rsp2_flag || model.controls.rsp_flag)
    {
        return Err(StarError::InvalidInput(String::from(
            "finish_load_model: set_viscosity_vars_TDC (star/private/tdc_hydro.f90:40-89) is not ported; required when TDC_alpha_M > 0 and MLT_option = 'TDC' (star/private/read_model.f90:164-167)",
        )));
    }
    model.layout = VarLayout::with_flags(
        model.species,
        model.controls.v_flag,
        model.controls.u_flag,
        model.controls.rsp2_flag,
        model.controls.rti_flag,
    );
    model.sync_mesh_and_pack()?;

    // `read_model.f90:102-129`.
    if model.controls.rotation_flag {
        if model.aux.have_j_rot {
            crate::hydro_rotation::use_xh_to_update_i_rot(model);
            for k in 0..model.nz {
                model.omega[k] = model.j_rot[k] / model.aux.i_rot[k];
            }
        } else {
            crate::hydro_rotation::use_xh_to_update_i_rot_and_j_rot(model);
        }
        model.pack_xh();
    }

    // `read_model.f90:157-159`: `okay_to_set_mlt_vc` is set by the reader
    // from `have_mlt_vc` (`model_io::read_mod_str_with_dir`).
    if model.controls.use_other_cgrav {
        extras.other_cgrav(model)?;
    }
    // `set_vars` with `doing_finish_load_model = .true.`
    // (`read_model.f90:161-168`): under `D_omega_flag` / `am_nu_rot_flag`
    // it zeroes `D_omega` and `am_nu_rot`
    // (`star/private/rotation_mix_info.f90:255-258`,
    // `star/private/mix_info.f90:1810-1813`). The `RSP` `unpack_xh` branch
    // (`star/private/hydro_vars.f90:225`) is excluded above.
    model.aux.doing_finish_load_model = true;
    if let Err(e) = set_hydro_vars(model, false, false) {
        model.aux.doing_finish_load_model = false;
        return Err(e);
    }
    if model.controls.rsp2_flag {
        // `set_RSP2_vars` (`star/private/hydro_rsp2.f90:67-111`): fill
        // `Hp_face(k) <= 0` from the face scale height, then the RSP2 face
        // and cell variables.
        for k in 0..model.nz {
            if model.aux.rsp.hp_face[k] <= 0.0 {
                model.aux.rsp.hp_face[k] =
                    crate::rsp2_eqns::eval_rsp2_hp_face_zone(model, k);
            }
        }
        model.pack_xh();
        crate::rsp2_ad::update_rsp2_vars_ad(model);
    }
    model.aux.doing_finish_load_model = false;

    // `read_model.f90:184-199`, again with `doing_finish_load_model`, which
    // nothing in the Brunt-Vaisala update reads.
    if model.controls.calculate_brunt_b {
        crate::hydro_brunt_surf::do_brunt_b(model)?;
    }
    if model.controls.calculate_brunt_n2 {
        crate::hydro_brunt_surf::do_brunt_n2(model)?;
    }
    Ok(())
}

/// Brings a newly built model into the state the step loop starts from, so
/// `model_builder` can relax it (`finish_load_model` and the energy integrals,
/// `star/private/init.f90:1031-1075`; `doing_first_model_of_run = .true.`,
/// `star/private/init.f90:921`).
fn prepare_new_model_for_relax(
    model: &mut StarModel,
    extras: &mut dyn StarExtras,
) -> Result<(), StarError> {
    model.layout = VarLayout::with_flags(
        model.species,
        model.controls.v_flag,
        model.controls.u_flag,
        model.controls.rsp2_flag,
        model.controls.rti_flag,
    );
    model.time_sec = model.star_age * SECYER;
    model.time_old_sec = model.star_age_old * SECYER;
    model.init_model_number = model.model_number as isize;
    model.doing_first_model_of_run = true;
    model.sync_mesh_and_pack()?;
    if model.controls.use_other_cgrav {
        extras.other_cgrav(model)?;
    }
    // `finish_load_model` runs `set_vars` with `doing_finish_load_model`
    // (`star/private/init.f90:1045`, `star/private/read_model.f90:161-168`).
    model.aux.doing_finish_load_model = true;
    let set_vars_result = set_hydro_vars(model, false, false);
    model.aux.doing_finish_load_model = false;
    set_vars_result?;
    model.save_start_values()?;
    model.save_old_values();
    model.m_center_old = model.m_center;
    model.r_center_old = model.r_center;
    model.l_center_old = model.l_center;
    model.v_center_old = model.v_center;
    Ok(())
}

/// `star_relax_mass_to_remove_H_env` (`star/public/star_lib.f90:1271-1287`):
/// relaxes the mass to the He core mass (`get_mass_info`) plus
/// `extra_mass_retained_by_remove_H_env`.
///
/// # Errors
/// Returns the `do_relax_mass` error.
fn do_relax_mass_to_remove_h_env(
    model: &mut StarModel,
    job: &StarJobControls,
    extras: &mut dyn StarExtras,
) -> Result<(), StarError> {
    let he_core_mass =
        crate::core_boundaries::compute_core_masses(model).he_core_mass;
    do_relax_mass(
        model,
        he_core_mass + job.extra_mass_retained_by_remove_h_env,
        job.lg_max_abs_mdot,
        extras,
    )
}

/// `do_relax_initial_composition` (`star/job/run_star_support.f90:2793-2845`):
/// reads `relax_composition_filename` (`num_pts num_species`, then
/// `xq xa(1:num_species)` per line) and relaxes to it with
/// `star_relax_composition` over `num_steps_to_relax_composition` steps.
///
/// # Errors
/// Returns [`StarError::IoError`] when the file cannot be read,
/// [`StarError::InvalidInput`] when it is malformed or its species count
/// differs from the model, and the relax error.
fn do_relax_initial_composition(
    model: &mut StarModel,
    job: &StarJobControls,
    case_dir: &Path,
    extras: &mut dyn StarExtras,
) -> Result<(), StarError> {
    let path = case_dir.join(job.relax_composition_filename.trim());
    let text = fs::read_to_string(&path).map_err(|e| {
        StarError::IoError(format!(
            "failed to open {} (star/job/run_star_support.f90:2803-2809): {e}",
            path.display()
        ))
    })?;
    let profile = parse_composition_profile_str(&text, Some(model.species))?;
    do_relax_composition(
        model,
        job.num_steps_to_relax_composition,
        &profile,
        extras,
    )
}

/// The relaxes `model_builder` applies to a new pre-main-sequence model
/// (`star/private/init.f90:1077-1109`), with the atmosphere set by
/// `setup_for_relax_after_create_pre_ms_model` and restored by
/// `done_relax_after_create_pre_ms_model` (`star/private/init.f90:1126-1142`).
fn relax_after_create_pre_ms_model(
    model: &mut StarModel,
    extras: &mut dyn StarExtras,
) -> Result<(), StarError> {
    let c = &mut model.controls;
    let saved_atm = (
        c.atm_option,
        c.atm_t_tau_relation,
        c.atm_t_tau_opacity,
        c.pextra_factor,
    );
    c.atm_option = AtmOption::TTau;
    c.atm_t_tau_relation = AtmTTauRelation::Eddington;
    c.atm_t_tau_opacity = AtmTTauOpacity::Fixed;
    c.pextra_factor = 2.0;

    // `lg_max_abs_mdot = -1000` picks the default rate by mass
    // (`star/private/init.f90:874`).
    let target_mstar = model.controls.initial_mass * M_SUN;
    if model.mstar > target_mstar {
        let initial_mass = model.controls.initial_mass;
        do_relax_mass(model, initial_mass, -1000.0, extras)?;
    } else if model.mstar < target_mstar {
        let (initial_mass, dlgm, years) = (
            model.controls.initial_mass,
            model.job.dlgm_per_step,
            model.job.change_mass_years_for_dt,
        );
        do_relax_mass_scale(model, initial_mass, dlgm, years, extras)?;
    }
    let (num_steps, dt_next) = (model.job.pre_ms_relax_num_steps, model.dt_next);
    do_relax_num_steps(model, num_steps, dt_next, extras)?;
    if model.job.pre_ms_relax_to_start_radiative_core {
        do_relax_to_radiative_core(model, extras)?;
    }

    let c = &mut model.controls;
    (
        c.atm_option,
        c.atm_t_tau_relation,
        c.atm_t_tau_opacity,
        c.pextra_factor,
    ) = saved_atm;
    Ok(())
}

/// Outcome of model initialization (`Terminate` stops cleanly at 0 steps).
pub(crate) enum InitModelOutcome {
    Model(Box<StarModel>),
    Terminate(String),
}

/// Initializes a [`StarModel`] from `bundle` (or updates `existing_model` for
/// subsequent multi-inlist stages) and executes all `&star_job` startup and
/// relaxation directives (`star/private/star_job_controls_io.f90` &
/// `star/private/init.f90`).
///
/// # Errors
/// Returns [`StarError`] if model loading, network switching, microphysics
/// evaluation, or an `other_rsp_build_model` termination occurs.
pub fn init_or_update_model_from_bundle(
    bundle: &InlistBundle,
    existing_model: Option<StarModel>,
    extras: &mut dyn StarExtras,
) -> Result<StarModel, StarError> {
    match init_or_update_model_from_bundle_outcome(
        bundle,
        existing_model,
        extras,
    )? {
        InitModelOutcome::Model(m) => Ok(*m),
        InitModelOutcome::Terminate(reason) => Err(StarError::InvalidInput(
            format!("stage terminated during initialization: {reason}"),
        )),
    }
}

/// Copies the hooks `extras_controls` installed on the pre-load star onto the
/// loaded model (upstream installs them on the one `s`, so they persist).
fn carry_extras_hooks(
    src: &crate::hydro_state::HydroAuxState,
    model: &mut StarModel,
) {
    let aux = &mut model.aux;
    if src.other_overshooting_scheme.is_some() {
        aux.other_overshooting_scheme
            .clone_from(&src.other_overshooting_scheme);
    }
    if src.other_adjust_mlt_gradt_fraction.is_some() {
        aux.other_adjust_mlt_gradt_fraction
            .clone_from(&src.other_adjust_mlt_gradt_fraction);
    }
    if src.other_mesh_functions.is_some() {
        aux.other_mesh_functions
            .clone_from(&src.other_mesh_functions);
    }
    if src.other_mesh_delta_coeff_factor.is_some() {
        aux.other_mesh_delta_coeff_factor
            .clone_from(&src.other_mesh_delta_coeff_factor);
    }
}

pub(crate) fn init_or_update_model_from_bundle_outcome(
    bundle: &InlistBundle,
    mut existing_model: Option<StarModel>,
    extras: &mut dyn StarExtras,
) -> Result<InitModelOutcome, StarError> {
    let _net_dir_guard = NetSearchDirGuard::new(&bundle.case_dir);
    // `extras_controls` runs after `star_setup` reads the inlists and before
    // `do_star_job_controls_before` and the model load
    // (`star/job/run_star_support.f90:429-459`): the star has its controls
    // but no model yet, so the hook sees a model with no cells. The controls
    // and `&star_job` values it changes, and the hooks it installs, carry
    // over to the loaded model.
    let (bundle, hooks) = {
        let mut shell = StarModel::new(
            0,
            Vec::new(),
            bundle.job.clone(),
            bundle.controls.clone(),
        );
        shell.eos_config = bundle.eos_config;
        shell.kap_config = bundle.kap_config.clone();
        extras.extras_controls(&mut shell)?;
        let mut b = bundle.clone();
        b.controls = shell.controls;
        b.job = shell.job;
        (b, shell.aux)
    };
    let bundle = &bundle;
    match crate::rsp_driver::init_rsp_model_from_bundle(
        bundle,
        &mut existing_model,
        extras,
    )? {
        crate::rsp_driver::InitRspOutcome::Model(mut m) => {
            carry_extras_hooks(&hooks, &mut m);
            return Ok(InitModelOutcome::Model(m));
        }
        crate::rsp_driver::InitRspOutcome::Terminate(reason) => {
            return Ok(InitModelOutcome::Terminate(reason));
        }
        crate::rsp_driver::InitRspOutcome::NotHandled => {}
    }
    let job = bundle.job.clone();
    let mut controls = bundle.controls.clone();
    // `model_builder` starts with `if (s% initial_y < 0) s% initial_y =
    // max(0d0, min(1d0, 0.24d0 + 2*initial_z))` from the inlist's `initial_z`,
    // before any model is loaded or built (`star/private/init.f90:891-894`);
    // a loaded `.mod` file does not set `initial_y` (`read_model.f90:283,
    // 366`).
    if controls.initial_y < 0.0 {
        controls.initial_y = (0.24 + 2.0 * controls.initial_z).clamp(0.0, 1.0);
    }
    let initial_y = controls.initial_y;
    let mut did_create_pre_ms = false;
    let mut did_create_initial_model = false;
    let mut did_load_saved_model = false;
    // Upstream's `restart` is a photo restart only
    // (`star/job/run_star_support.f90:433-442`). A model carried over from an
    // earlier `./rn` stage stands for the `.mod` file that stage saved and
    // the next one loads, and a binary seed is a freshly loaded model, so
    // neither is a restart: every `.not. restart` control applies.
    let restart = false;
    let mut model = if let Some(prev) = existing_model {
        prev
    } else if job.load_saved_model {
        did_load_saved_model = true;
        load_saved_mod(
            &bundle.case_dir,
            &job.saved_model_name,
            controls.clone(),
            job.clone(),
        )?
    } else if job.create_pre_ms_model {
        did_create_pre_ms = true;
        crate::pre_ms::build_pre_ms_model_with_configs(
            controls.clone(),
            job.clone(),
            bundle.eos_config,
            bundle.kap_config.clone(),
            Some(&bundle.case_dir),
        )?
    } else if job.use_other_build_initial_model {
        extras.other_build_initial_model(bundle)?.ok_or_else(|| {
            StarError::InvalidInput(String::from(
                "use_other_build_initial_model = .true. requires an other_build_initial_model hook",
            ))
        })?
    } else if job.create_initial_model {
        // `do_load1_star` (`star/job/run_star_support.f90:1677-1699`) and
        // `model_builder` (`star/private/init.f90:950-966, 1110-1117`).
        if job.create_pre_ms_model {
            return Err(StarError::InvalidInput(String::from(
                "you have both create_initial_model and create_pre_main_sequence_model set true",
            )));
        }
        did_create_initial_model = true;
        crate::create_initial_model::build_initial_model(
            controls.clone(),
            job.clone(),
            bundle.eos_config,
            bundle.kap_config.clone(),
        )?
    } else {
        if !job.changes_z() && (controls.initial_z - 0.02).abs() > 1.0e-4 {
            return Err(StarError::InvalidInput(format!(
                "built-in ZAMS catalog zams_z2m2_y28 has initial_z = 0.02, got {} without relax_z/change_initial_z",
                controls.initial_z
            )));
        }
        if !job.changes_y()
            && controls.initial_y >= 0.0
            && (controls.initial_y - 0.28).abs() > 1.0e-4
        {
            return Err(StarError::InvalidInput(format!(
                "built-in ZAMS catalog zams_z2m2_y28 has initial_y = 0.28, got {} without relax_y/change_initial_y",
                controls.initial_y
            )));
        }
        load_zams_model(controls.clone(), job.clone())?
    };

    // Preserve active model state across stages / `.mod` loads unless overridden
    let preserved_net = model.job.net_name.clone();
    let (preserved_v, preserved_u, preserved_rot, preserved_rti) = (
        model.controls.v_flag,
        model.controls.u_flag,
        model.controls.rotation_flag,
        model.controls.rti_flag,
    );
    let (preserved_tau, preserved_m, preserved_z) = (
        model.controls.tau_factor,
        model.controls.initial_mass,
        model.controls.initial_z,
    );

    model.job = job.clone();
    model.job.net_name = preserved_net;
    model.controls = controls;
    (
        model.controls.v_flag,
        model.controls.u_flag,
        model.controls.rotation_flag,
    ) = (preserved_v, preserved_u, preserved_rot);
    if !job.change_initial_rti_flag {
        model.controls.rti_flag = preserved_rti;
    }
    (model.controls.initial_mass, model.controls.initial_z) =
        (preserved_m, preserved_z);
    // The model's `tau_factor` (as saved in a `.mod` file) replaces the
    // inlist value unless the inlist just set it
    // (`star/private/read_model.f90:307-315`).
    if (preserved_tau - model.controls.tau_factor).abs() > preserved_tau * 1.0e-9
        && model.controls.tau_factor != job.set_to_this_tau_factor
    {
        model.controls.tau_factor = preserved_tau;
    }
    model.controls.initial_y = initial_y;
    model.eos_config = bundle.eos_config;
    model.kap_config = bundle.kap_config.clone();

    crate::inlist_runner::install_star_aux_hooks(&*extras, &mut model);
    carry_extras_hooks(&hooks, &mut model);

    // `do_star_job_controls_before` sets `s% tau_base` from the atmosphere
    // controls (`star/job/run_star_support.f90:1546`).
    model.tau_base =
        crate::hydro_brunt_surf::get_atm_tau_base(&model.controls)?;

    // `star_read_model` ends in `finish_load_model`
    // (`star/private/read_model.f90:69-205`), before
    // `do_star_job_controls_after` changes the net
    // (`star/job/run_star_support.f90:1912-1916`); its `set_vars` sets the
    // `mlt_D` that `set_new_abundances` reads in `get_test_lgT`
    // (`star/private/adjust_xyz.f90:617-626`).
    if did_load_saved_model {
        finish_load_model(&mut model, extras)?;
    }

    // `model_builder` relaxes a newly built model before the `&star_job`
    // controls applied after loading (`star/private/init.f90:1077-1118`,
    // `star/job/run_star_support.f90:1653-1699`).
    if did_create_pre_ms {
        prepare_new_model_for_relax(&mut model, extras)?;
        relax_after_create_pre_ms_model(&mut model, extras)?;
    } else if did_create_initial_model
        && model.job.initial_model_relax_num_steps > 0
    {
        prepare_new_model_for_relax(&mut model, extras)?;
        let (num_steps, dt_next) =
            (model.job.initial_model_relax_num_steps, model.dt_next);
        do_relax_num_steps(&mut model, num_steps, dt_next, extras)?;
    }

    // 1. Reaction network switching (`change_net` / `change_initial_net`)
    // A pre-MS model is already built on `new_net_name` (`pre_ms_change_net`,
    // `star/job/run_star_support.f90:1661-1667`), and `change_net` returns
    // when the new name equals `s% net_name` (`star/private/adjust_xyz.f90:93`).
    let pre_ms_on_new_net = did_create_pre_ms
        && model.job.net_name.trim() == job.new_net_name.trim();
    if job.change_net && !job.new_net_name.is_empty() && !pre_ms_on_new_net {
        // `star_change_to_new_net(id, s% job% adjust_abundances_for_new_isos,
        // ...)` (`star/job/run_star_support.f90:1912-1916`).
        crate::micro::change_model_network_with_adjust(
            &mut model,
            &job.new_net_name,
            Some(&bundle.case_dir),
            job.adjust_abundances_for_new_isos,
        )?;
    }

    // `model_builder` ends with `doing_first_model_of_run = .true.`
    // (`star/private/init.f90:1120`), before `do_star_job_controls_after`;
    // a relax there clears it again in `finish_step`
    // (`star/private/evolve.f90:2366`).
    if !restart {
        model.doing_first_model_of_run = true;
    }

    // 2. Age, model number and initial timestep, which upstream sets before
    // the flag changes and relaxes (`star/job/run_star_support.f90:1957-2012`).
    // `set_tau_factor` (`star/job/run_star_support.f90:1949-1953`).
    if job.set_tau_factor || (job.set_initial_tau_factor && !restart) {
        model.controls.tau_factor = job.set_to_this_tau_factor;
    }
    if job.set_initial_age && !restart {
        model.star_age = job.initial_age;
        model.star_age_old = job.initial_age;
    }
    model.time_sec = model.star_age * SECYER;
    model.time_old_sec = model.star_age_old * SECYER;
    if job.set_initial_model_number && !restart {
        model.model_number = job.initial_model_number;
        model.init_model_number = model.model_number as isize;
    } else if !restart {
        model.init_model_number = model.model_number as isize;
    }
    if job.set_initial_number_retries && !restart {
        model.num_retries = job.initial_number_retries;
    }
    // `set_initial_dt` sets `dt_next` only
    // (`star/job/run_star_support.f90:1967-1975`).
    if job.set_initial_dt && !restart {
        if job.years_for_initial_dt > 0.0 {
            model.dt_next = job.years_for_initial_dt * SECYER;
        } else if job.seconds_for_initial_dt > 0.0 {
            model.dt_next = job.seconds_for_initial_dt;
        }
    }

    // 3. Velocity and rotation flags (`change_v_flag`, `change_u_flag`,
    // `change_rotation_flag`; `star/job/run_star_support.f90:2058-2187`).
    if job.change_v_flag {
        model.controls.v_flag = job.new_v_flag;
        if job.new_v_flag && model.v.len() != model.nz {
            model.v.resize(model.nz, 0.0);
        }
    }
    if job.change_u_flag {
        model.controls.u_flag = job.new_u_flag;
        if job.new_u_flag && model.u.len() != model.nz {
            model.u.resize(model.nz, 0.0);
        }
    }
    // `star_set_D_omega_flag` and `star_set_am_nu_rot_flag`
    // (`star/job/run_star_support.f90:2119-2129`,
    // `star/private/set_flags.f90:718-745`): a change of flag zeroes the
    // array.
    if job.change_d_omega_flag
        && model.controls.d_omega_flag != job.new_d_omega_flag
    {
        model.controls.d_omega_flag = job.new_d_omega_flag;
        let nz = model.nz;
        model.aux.rotmix.d_omega.resize(nz, 0.0);
        model.aux.rotmix.d_omega[..nz].fill(0.0);
    }
    if job.change_am_nu_rot_flag
        && model.controls.am_nu_rot_flag != job.new_am_nu_rot_flag
    {
        model.controls.am_nu_rot_flag = job.new_am_nu_rot_flag;
        let nz = model.nz;
        model.aux.am_nu_rot.resize(nz, 0.0);
        model.aux.am_nu_rot[..nz].fill(0.0);
    }
    if job.change_rotation_flag {
        model.controls.rotation_flag = job.new_rotation_flag;
        if job.new_rotation_flag && model.omega.len() != model.nz {
            model.omega.resize(model.nz, 0.0);
            model.j_rot.resize(model.nz, 0.0);
        }
    }

    if job.change_initial_rti_flag {
        model.controls.rti_flag = job.new_rti_flag;
    }

    if model.controls.rsp2.change_rsp_flag {
        model.controls.rsp_flag = model.controls.rsp2.new_rsp_flag;
    }
    if model.controls.rsp2.change_rsp2_flag {
        model.controls.rsp2_flag = model.controls.rsp2.new_rsp2_flag;
    }
    if model.controls.rsp2_flag {
        model.controls.rsp_flag = false;
    }

    model.layout = VarLayout::with_flags(
        model.species,
        model.controls.v_flag,
        model.controls.u_flag,
        model.controls.rsp2_flag,
        model.controls.rti_flag,
    );

    if model.controls.v_flag && model.v.len() != model.nz {
        model.v.resize(model.nz, 0.0);
        model.v_start.resize(model.nz, 0.0);
    }
    if model.controls.u_flag && model.u.len() != model.nz {
        model.u.resize(model.nz, 0.0);
        model.u_start.resize(model.nz, 0.0);
    }
    if model.controls.rti_flag {
        if model.aux.alpha_rti.len() != model.nz {
            model.aux.alpha_rti.resize(model.nz, 0.0);
        }
        if model.aux.alpha_rti_start.len() != model.nz {
            model.aux.alpha_rti_start.resize(model.nz, 0.0);
        }
        if model.aux.alpha_rti_old.len() != model.nz {
            model.aux.alpha_rti_old.resize(model.nz, 0.0);
        }
    }
    model.layout = VarLayout::with_flags(
        model.species,
        model.controls.v_flag,
        model.controls.u_flag,
        model.controls.rsp2_flag,
        model.controls.rti_flag,
    );

    // Uniform rotation (`star_set_uniform_omega`), after the flag changes and
    // before the composition changes and relaxes, in upstream's order
    // (`star/job/run_star_support.f90:2138-2187`).
    if model.controls.rotation_flag && job.set_omega {
        apply_uniform_rotation(&mut model, job.new_omega);
    }
    if model.controls.rotation_flag && job.set_initial_omega && !restart {
        apply_uniform_rotation(&mut model, job.new_omega);
    }
    if model.controls.rotation_flag
        && (job.set_surface_rotation_v
            || (job.set_initial_surface_rotation_v && !restart))
    {
        let new_omega = job.new_surface_rotation_v * 1.0e5 / model.r[0];
        apply_uniform_rotation(&mut model, new_omega);
    }
    if model.controls.rotation_flag
        && (job.set_omega_div_omega_crit
            || (job.set_initial_omega_div_omega_crit && !restart))
    {
        let r_surf = model.r.first().copied().unwrap_or(R_SUN).max(1.0);
        let g_surf = model
            .cgrav
            .first()
            .copied()
            .filter(|&g| g > 0.0)
            .unwrap_or(STANDARD_CGRAV);
        let omega_crit =
            (g_surf * model.mstar / flexmesa_math::pow3(r_surf)).sqrt();
        apply_uniform_rotation(
            &mut model,
            job.new_omega_div_omega_crit * omega_crit,
        );
    }

    // 4. Composition and mass, in upstream's order
    // (`star/job/run_star_support.f90:2198-2487`), with `restart = false`.
    let mut relax_ready = did_create_pre_ms;
    let mut ready_for_relax =
        |m: &mut StarModel, ex: &mut dyn StarExtras| -> Result<(), StarError> {
            if !relax_ready {
                prepare_new_model_for_relax(m, ex)?;
                relax_ready = true;
            }
            Ok(())
        };
    if job.set_uniform_initial_composition {
        set_uniform_initial_composition(&mut model, &job)?;
    }
    if job.relax_initial_composition {
        ready_for_relax(&mut model, extras)?;
        do_relax_initial_composition(
            &mut model,
            &job,
            &bundle.case_dir,
            extras,
        )?;
    }
    if job.relax_initial_to_xaccrete && !restart {
        return Err(StarError::InvalidInput(String::from(
            "relax_initial_to_xaccrete (star_relax_to_xaccrete, \
             star/job/run_star_support.f90:2240-2243) is not ported (#436)",
        )));
    }
    if (job.relax_initial_angular_momentum || job.relax_initial_entropy)
        && !restart
    {
        return Err(StarError::InvalidInput(String::from(
            "relax_initial_angular_momentum / relax_initial_entropy \
             (star/job/run_star_support.f90:2251-2259) are not ported (#436)",
        )));
    }
    // Change Z before Y, since changing Z can change Y
    // (`star/job/run_star_support.f90:2289-2308`).
    if job.change_z || job.change_initial_z {
        let nz = model.nz;
        set_z(&mut model, job.new_z, 0, nz - 1)?;
    }
    if job.change_y || job.change_initial_y {
        let nz = model.nz;
        set_y(&mut model, job.new_y, 0, nz - 1)?;
    }
    // `star/job/run_star_support.f90:2355-2380`.
    if job.set_irradiation || (job.set_initial_irradiation && !restart) {
        model.controls.irradiation_flux = job.set_to_this_irrad_flux;
        model.controls.column_depth_for_irradiation =
            job.column_depth_for_irradiation;
    }
    if job.set_v_center || (job.set_initial_v_center && !restart) {
        model.v_center = job.new_v_center;
    }
    if job.set_l_center || (job.set_initial_l_center && !restart) {
        model.l_center = job.new_l_center * flexmesa_const::L_SUN;
    }
    // `star_relax_Z` / `star_relax_Y` (`star/job/run_star_support.f90:2386-2420`).
    for _ in 0..(usize::from(job.relax_initial_z) + usize::from(job.relax_z)) {
        ready_for_relax(&mut model, extras)?;
        let c = &model.controls;
        let (dlnz, minq, maxq) =
            (c.relax_dlnz, job.relax_z_minq, job.relax_z_maxq);
        do_relax_z(&mut model, job.new_z, dlnz, minq, maxq, extras)?;
    }
    for _ in 0..(usize::from(job.relax_initial_y) + usize::from(job.relax_y)) {
        ready_for_relax(&mut model, extras)?;
        let c = &model.controls;
        let (dy, minq, maxq) = (c.relax_dy, job.relax_y_minq, job.relax_y_maxq);
        do_relax_y(&mut model, job.new_y, dy, minq, maxq, extras)?;
    }
    if job.relax_mass {
        ready_for_relax(&mut model, extras)?;
        do_relax_mass(&mut model, job.new_mass, job.lg_max_abs_mdot, extras)?;
    }
    if job.relax_mass_to_remove_h_env {
        ready_for_relax(&mut model, extras)?;
        do_relax_mass_to_remove_h_env(&mut model, &job, extras)?;
    }
    if job.relax_initial_mass {
        ready_for_relax(&mut model, extras)?;
        do_relax_mass(&mut model, job.new_mass, job.lg_max_abs_mdot, extras)?;
    }
    if job.relax_initial_mass_to_remove_h_env {
        ready_for_relax(&mut model, extras)?;
        do_relax_mass_to_remove_h_env(&mut model, &job, extras)?;
    }
    if job.relax_mass_scale || job.relax_initial_mass_scale {
        ready_for_relax(&mut model, extras)?;
        do_relax_mass_scale(
            &mut model,
            job.new_mass,
            job.dlgm_per_step,
            job.change_mass_years_for_dt,
            extras,
        )?;
    }

    // 5. Inner boundary, `tau_factor` and irradiation relaxes, in upstream's
    // order (`star/job/run_star_support.f90:2489-2567`).
    if job.relax_core || (job.relax_initial_core && !restart) {
        ready_for_relax(&mut model, extras)?;
        do_relax_core(
            &mut model,
            job.new_core_mass,
            job.dlg_core_mass_per_step,
            job.relax_core_years_for_dt,
            job.core_avg_rho,
            job.core_avg_eps,
            extras,
        )?;
    }
    if job.relax_m_center || (job.relax_initial_m_center && !restart) {
        ready_for_relax(&mut model, extras)?;
        do_relax_m_center(
            &mut model,
            job.new_mass,
            job.dlgm_per_step,
            job.relax_m_center_dt,
            extras,
        )?;
    }
    if job.relax_r_center || job.relax_l_center {
        return Err(StarError::InvalidInput(String::from(
            "relax_R_center / relax_L_center (star_relax_R_center, \
             star_relax_L_center; star/private/relax.f90:2490-2889) are not \
             ported (#436)",
        )));
    }
    if job.relax_tau_factor || (job.relax_initial_tau_factor && !restart) {
        ready_for_relax(&mut model, extras)?;
        do_relax_tau_factor(
            &mut model,
            job.relax_to_this_tau_factor,
            job.dlogtau_factor,
            extras,
        )?;
    }
    if job.relax_irradiation || (job.relax_initial_irradiation && !restart) {
        ready_for_relax(&mut model, extras)?;
        do_relax_irradiation(
            &mut model,
            job.relax_irradiation_min_steps,
            job.new_irradiation_flux,
            job.column_depth_for_irradiation,
            job.relax_irradiation_max_yrs_dt,
            extras,
        )?;
    }
    if job.remove_surface_by_relax_to_star_cut {
        // `do_remove_surface` with `remove_surface_by_relax_to_star_cut`
        // calls `do_relax_to_star_cut` (`star/private/remove_shells.f90:
        // 1024-1035, 1156-1404`), which is not ported.
        return Err(StarError::InvalidInput(String::from(
            "remove_surface_by_relax_to_star_cut (do_relax_to_star_cut, \
             star/private/remove_shells.f90:1156-1404) is not ported (#436)",
        )));
    }

    // Rotation relaxes, after `do_remove_surface`
    // (`star/job/run_star_support.f90:2593-2652`).
    {
        let rot = model.controls.rotation_flag;
        let (n, dt_yrs) =
            (job.num_steps_to_relax_rotation, job.relax_omega_max_yrs_dt);
        let relaxes = [
            (job.relax_omega, RelaxOmegaKind::NewOmega, job.new_omega),
            (
                job.relax_initial_omega && !restart,
                RelaxOmegaKind::NewOmega,
                job.new_omega,
            ),
            (
                job.relax_omega_div_omega_crit,
                RelaxOmegaKind::NewOmegaDivOmegaCrit,
                job.new_omega_div_omega_crit,
            ),
            (
                job.relax_initial_omega_div_omega_crit && !restart,
                RelaxOmegaKind::NewOmegaDivOmegaCrit,
                job.new_omega_div_omega_crit,
            ),
            (
                job.relax_surface_rotation_v,
                RelaxOmegaKind::NewSurfaceRotationV,
                job.new_surface_rotation_v,
            ),
            (
                job.relax_initial_surface_rotation_v && !restart,
                RelaxOmegaKind::NewSurfaceRotationV,
                job.new_surface_rotation_v,
            ),
        ];
        for (i, (wanted, kind, target)) in relaxes.into_iter().enumerate() {
            if !(rot && wanted) {
                continue;
            }
            ready_for_relax(&mut model, extras)?;
            do_relax_uniform_omega(&mut model, kind, target, n, dt_yrs, extras)?;
            if i == 4 {
                // `s% job% new_omega = new_surface_rotation_v*1d5/s% r(1)`
                // after `relax_surface_rotation_v` only (`:2633`).
                model.job.new_omega =
                    job.new_surface_rotation_v * 1.0e5 / model.r[0];
            }
        }
    }

    if job.remove_initial_center_by_temperature > 0.0 {
        excise_core_by_temperature(
            &mut model,
            job.remove_initial_center_by_temperature,
        )?;
    }
    if model.controls.rsp2_flag {
        crate::rsp2_remesh::seed_rsp2_w_from_mlt_vc_if_needed(&mut model)?;
        if model.controls.rsp2.remesh_when_load
            && model.controls.rsp2.nz >= 2
            && model.nz != model.controls.rsp2.nz
        {
            crate::rsp2_remesh::remesh_for_rsp2(&mut model)?;
        } else {
            for k in 0..model.nz {
                if model.aux.rsp.hp_face[k] <= 0.0 {
                    model.aux.rsp.hp_face[k] =
                        crate::rsp2_eqns::eval_rsp2_hp_face_zone(&model, k);
                }
            }
            model.pack_xh();
        }
    }
    model.sync_mesh_and_pack()?;
    if model.controls.use_other_cgrav {
        extras.other_cgrav(&mut model)?;
    }

    set_hydro_vars(&mut model, false, false)?;
    if model.controls.rsp2_flag {
        let (e0, _) = crate::rsp_driver::rsp_total_energy_integrals(&mut model);
        model.aux.total_energy = e0;
        model.aux.rsp.total_energy_old = e0;
        model.aux.rsp.time_sec = model.star_age * SECYER;
        model.aux.rsp.time_old_sec = model.aux.rsp.time_sec;
    }
    model.save_start_values()?;
    model.save_old_values();
    model.m_center_old = model.m_center;
    model.r_center_old = model.r_center;
    model.l_center_old = model.l_center;
    model.v_center_old = model.v_center;
    if job.set_initial_cumulative_energy_error && !restart {
        model.aux.cumulative_energy_error = job.new_cumulative_energy_error;
        model.aux.step_hist.cumulative_energy_error_old =
            job.new_cumulative_energy_error;
    }
    extras.extras_startup(&mut model, false)?;
    if !model.job.net_name.trim().is_empty() {
        let _ = resolve_model_net(&model)?;
    }

    Ok(InitModelOutcome::Model(Box::new(model)))
}
