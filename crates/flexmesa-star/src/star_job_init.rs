// SPDX-License-Identifier: LGPL-3.0-or-later
//! `&star_job` model initialization, network switching, composition/mass/core
//! relaxation, and rotational spin-up handlers (`init_or_update_model_from_bundle`,
//! corresponding to `star/private/init.f90` and `star/private/relax.f90`).

use std::fs;
use std::path::Path;

use flexmesa_const::{M_SUN, R_SUN, SECYER, STANDARD_CGRAV};

use crate::controls::{StarControls, StarError, StarJobControls, VarLayout};
use crate::hydro_vars::set_hydro_vars;
use crate::inlist::InlistBundle;
use crate::inlist_runner::StarExtras;
use crate::micro::change_model_network;
use crate::model::StarModel;
use crate::model_io::read_mod_str;
use crate::relax::{
    apply_composition_relaxation, apply_uniform_rotation,
    excise_core_by_temperature, relax_core_boundary, relax_from_profile_files,
    relax_irradiation_to_target, relax_mass_to_target,
    relax_tau_factor_to_target, relax_to_remove_h_env,
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
    read_mod_str(&text, controls, job)
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

pub(crate) fn init_or_update_model_from_bundle_outcome(
    bundle: &InlistBundle,
    mut existing_model: Option<StarModel>,
    extras: &mut dyn StarExtras,
) -> Result<InitModelOutcome, StarError> {
    match crate::rsp_driver::init_rsp_model_from_bundle(
        bundle,
        &mut existing_model,
        extras,
    )? {
        crate::rsp_driver::InitRspOutcome::Model(m) => {
            return Ok(InitModelOutcome::Model(m));
        }
        crate::rsp_driver::InitRspOutcome::Terminate(reason) => {
            return Ok(InitModelOutcome::Terminate(reason));
        }
        crate::rsp_driver::InitRspOutcome::NotHandled => {}
    }
    let job = bundle.job.clone();
    let controls = bundle.controls.clone();
    let mut did_create_pre_ms = false;
    let is_restart = existing_model.is_some();
    let mut model = if let Some(prev) = existing_model {
        prev
    } else if job.load_saved_model {
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
        if job.initial_model_relax_num_steps > 0 {
            // Upstream builds the model, then relaxes it with
            // `do_relax_num_steps` (`star/private/relax.f90:3323-3374`), a
            // `do_internal_evolve` loop that FlexMESA does not port yet.
            // Failing before the build gives the same outcome sooner.
            return Err(StarError::InvalidInput(format!(
                "create_initial_model with initial_model_relax_num_steps = {} needs \
                 do_relax_num_steps (star/private/relax.f90:3323), which is not yet \
                 implemented (#154)",
                job.initial_model_relax_num_steps
            )));
        }
        crate::create_initial_model::build_initial_model(
            controls.clone(),
            job.clone(),
            bundle.eos_config,
            bundle.kap_config.clone(),
        )?
    } else {
        if !job.relax_z && (controls.initial_z - 0.02).abs() > 1.0e-4 {
            return Err(StarError::InvalidInput(format!(
                "built-in ZAMS catalog zams_z2m2_y28 has initial_z = 0.02, got {} without relax_z/change_initial_z",
                controls.initial_z
            )));
        }
        if !job.relax_y
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
    let (preserved_tau, preserved_m, preserved_z, preserved_y) = (
        model.controls.tau_factor,
        model.controls.initial_mass,
        model.controls.initial_z,
        model.controls.initial_y,
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
    (
        model.controls.tau_factor,
        model.controls.initial_mass,
        model.controls.initial_z,
        model.controls.initial_y,
    ) = (preserved_tau, preserved_m, preserved_z, preserved_y);
    model.eos_config = bundle.eos_config;
    model.kap_config = bundle.kap_config.clone();

    extras.extras_controls(&mut model)?;

    // 1. Reaction network switching (`change_net` / `change_initial_net`)
    if job.change_net && !job.new_net_name.is_empty() {
        change_model_network(
            &mut model,
            &job.new_net_name,
            Some(&bundle.case_dir),
        )?;
    }

    // 2. Composition and external profile relaxation
    apply_composition_relaxation(&mut model, &job);
    relax_from_profile_files(&mut model, &bundle.case_dir)?;

    // 3. Mass and H-envelope stripping relaxation
    if job.relax_mass && job.new_mass > 0.0 {
        relax_mass_to_target(&mut model, job.new_mass, 8);
    } else if job.relax_initial_mass_to_remove_h_env {
        relax_to_remove_h_env(&mut model, 1.0e-4, 8);
    }

    // 4. Inner boundary, irradiation, and optical-depth relaxation
    if job.relax_m_center && job.new_core_mass > 0.0 {
        relax_core_boundary(
            &mut model,
            job.new_core_mass,
            job.core_avg_rho,
            job.core_avg_eps,
            8,
        );
    }
    if job.relax_r_center && job.new_r_center > 0.0 {
        model.r_center = job.new_r_center;
    }
    if job.relax_l_center && job.new_l_center >= 0.0 {
        model.l_center = job.new_l_center;
    }
    if job.relax_irradiation && job.new_irradiation_flux > 0.0 {
        relax_irradiation_to_target(
            &mut model,
            job.new_irradiation_flux,
            job.column_depth_for_irradiation,
            8,
        );
    }
    if job.set_tau_factor && job.set_to_this_tau_factor > 0.0 {
        relax_tau_factor_to_target(&mut model, job.set_to_this_tau_factor, 8);
    }

    // 6. Velocity & rotation flags (`change_v_flag`, `change_u_flag`, `change_rotation_flag`)
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

    // 7. Age, model number, and initial timestep overrides
    if job.set_initial_age {
        model.star_age = job.initial_age;
        model.star_age_old = job.initial_age;
    }
    model.time_sec = model.star_age * SECYER;
    model.time_old_sec = model.star_age_old * SECYER;
    if job.set_initial_model_number && !is_restart {
        model.model_number = job.initial_model_number;
        model.init_model_number = model.model_number as isize;
    } else if !is_restart {
        model.init_model_number = model.model_number as isize;
    }
    if job.set_initial_number_retries && !is_restart {
        model.num_retries = job.initial_number_retries;
    }
    if job.set_initial_dt {
        if job.years_for_initial_dt > 0.0 {
            model.dt = job.years_for_initial_dt * SECYER;
            model.dt_next = model.dt;
        } else if job.seconds_for_initial_dt > 0.0 {
            model.dt = job.seconds_for_initial_dt;
            model.dt_next = model.dt;
        }
    }

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

    // Apply rotational spin-up AFTER equilibration and `sync_mesh_and_pack` so
    // `model.r[0]` and `model.cgrav[0]` match the active stellar structure.
    if job.set_surface_rotation_v
        || (!is_restart && job.set_initial_surface_rotation_v)
    {
        let r_surf = model.r.first().copied().unwrap_or(R_SUN).max(1.0);
        apply_uniform_rotation(
            &mut model,
            (job.new_surface_rotation_v * 1.0e5) / r_surf,
        );
    } else if job.set_omega_div_omega_crit
        || (!is_restart && job.set_initial_omega_div_omega_crit)
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
    } else if job.set_omega || (!is_restart && job.set_initial_omega) {
        apply_uniform_rotation(&mut model, job.new_omega);
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
    let relax_rad_core = bundle
        .sections
        .get("star_job")
        .map(|sj| {
            sj.try_get_bool("star_job", "pre_ms_relax_to_start_radiative_core")
        })
        .transpose()?
        .flatten()
        .unwrap_or(true);
    if did_create_pre_ms && model.mstar / M_SUN >= 0.3 && relax_rad_core {
        crate::evolve_support::relax_pre_ms_to_radiative_core(&mut model)?;
    }
    model.m_center_old = model.m_center;
    model.r_center_old = model.r_center;
    model.l_center_old = model.l_center;
    model.v_center_old = model.v_center;
    if !is_restart {
        // Match `star/private/init.f90:1120` (`s% doing_first_model_of_run = .true.`).
        model.doing_first_model_of_run = true;
    }
    extras.extras_startup(&mut model, false)?;

    Ok(InitModelOutcome::Model(Box::new(model)))
}
