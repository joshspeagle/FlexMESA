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
use crate::pre_ms::build_pre_ms_model;
pub(crate) use crate::relax::equilibrate_tier2_scaled_model;
use crate::relax::{
    apply_composition_relaxation, apply_uniform_rotation,
    excise_core_by_temperature, relax_core_boundary, relax_from_profile_files,
    relax_irradiation_to_target, relax_mass_to_target,
    relax_tau_factor_to_target, relax_to_remove_h_env,
};
use crate::tier2_extras::apply_planetary_core;
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
        let rsp_golden = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/golden/rsp")
            .join(fname);
        if rsp_golden.exists() {
            return Some(rsp_golden);
        }
    }
    None
}

/// Peeks the `M/Msun` header entry from a saved `.mod` file.
fn peek_mod_header_mass(mod_path: &Path) -> Option<f64> {
    let text = fs::read_to_string(mod_path).ok()?;
    for line in text.lines().take(35) {
        let clean = line.split('!').next().unwrap_or(line).trim();
        let tokens: Vec<&str> = clean.split_whitespace().collect();
        if tokens.len() >= 2 && tokens[0] == "M/Msun" {
            let num_s = tokens[1].replace(['D', 'd'], "e");
            if let Ok(m) = num_s.parse::<f64>() {
                if m > 0.0 {
                    return Some(m);
                }
            }
        }
    }
    None
}

/// Builds an initial planet, brown dwarf, or conductive-flame model when
/// `job.create_initial_model` or `job.use_other_build_initial_model` is true
/// (`star/private/create_initial_model.f90`).
fn build_initial_planet_or_wd_model(
    mut controls: StarControls,
    job: StarJobControls,
) -> Result<StarModel, StarError> {
    let use_create_init_dims =
        job.create_initial_model || job.use_other_build_initial_model;
    let m_target = if use_create_init_dims
        && job.mass_in_gm_for_create_initial_model > 0.0
    {
        job.mass_in_gm_for_create_initial_model / M_SUN
    } else if job.relax_mass && job.new_mass > 0.0 {
        job.new_mass
    } else {
        controls.initial_mass
    };
    let r_init_cm = if use_create_init_dims {
        job.radius_in_cm_for_create_initial_model
    } else {
        0.0
    };
    controls.initial_mass = 1.0;
    let mut model = load_zams_model(controls, job)?;
    model.controls.initial_mass = m_target;
    let old_mstar = model.mstar.max(1.0);
    let new_mstar = m_target.max(1.0e-5) * M_SUN;
    let dln_r = if r_init_cm > 0.0 {
        (r_init_cm / model.ln_r[0].exp().max(1.0)).ln()
    } else {
        (new_mstar / old_mstar).ln() / 3.0
    };
    for k in 0..model.nz {
        model.ln_r[k] += dln_r;
        model.lum[k] *= new_mstar / old_mstar;
    }
    model.mstar = new_mstar;
    model.xmstar = (new_mstar - model.m_center).max(0.0);
    model.mstar_start = model.mstar;
    model.mstar_old = model.mstar;
    Ok(model)
}

/// Initializes a [`StarModel`] from `bundle` (or updates `existing_model` for
/// subsequent multi-inlist stages) and executes all `&star_job` startup and
/// relaxation directives (`star/private/star_job_controls_io.f90` &
/// `star/private/init.f90`).
///
/// # Errors
/// Returns [`StarError`] if model loading, network switching, or microphysics
/// evaluation fails.
pub fn init_or_update_model_from_bundle(
    bundle: &InlistBundle,
    mut existing_model: Option<StarModel>,
    extras: &mut dyn StarExtras,
) -> Result<StarModel, StarError> {
    if let Some(rsp_m) = crate::rsp_driver::init_rsp_model_from_bundle(
        bundle,
        &mut existing_model,
        extras,
    )? {
        return Ok(rsp_m);
    }
    let job = bundle.job.clone();
    let controls = bundle.controls.clone();
    let case_name = bundle
        .case_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let out_of_zams_range =
        controls.initial_mass > 15.8 || controls.initial_mass < 0.51;
    let is_pulse_mod_case = case_name.starts_with("gyre_in_mesa_")
        || matches!(case_name, "astero_gyre" | "astero_adipls");
    let is_pulsation_env_mod = crate::tier4_extras::is_tier4_case(case_name)
        && job.load_saved_model
        && !job.saved_model_name.is_empty()
        && controls.use_dprad_dm_form_of_t_gradient_eqn
        && controls.use_momentum_outer_bc
        && controls.eps_nuc_factor == 0.0;
    let is_rsp2_mod_case = controls.rsp2_flag
        || (controls.rsp2.change_rsp2_flag && controls.rsp2.new_rsp2_flag)
        || case_name.starts_with("dev_rsp2_")
        || is_pulsation_env_mod;
    let is_tier2_or_3 = crate::tier2_extras::is_tier2_case(case_name)
        || crate::tier3_extras::is_tier3_case(case_name)
        || crate::tier4_extras::is_tier4_case(case_name)
        || crate::tier5_extras::is_tier5_star_case(case_name)
        || job.relax_initial_mass_to_remove_h_env
        || out_of_zams_range;
    let had_existing_model = existing_model.is_some();
    let mut loaded_from_mod = false;
    let mut model = if is_pulse_mod_case || is_rsp2_mod_case {
        if let Some(prev) = existing_model {
            loaded_from_mod = true;
            prev
        } else if job.load_saved_model && !job.saved_model_name.is_empty() {
            if let Some(mod_path) =
                resolve_model_path(&bundle.case_dir, &job.saved_model_name)
            {
                let text = fs::read_to_string(&mod_path).map_err(|e| {
                    StarError::IoError(format!(
                        "failed to read saved model {}: {e}",
                        mod_path.display()
                    ))
                })?;
                match read_mod_str(&text, controls.clone(), job.clone()) {
                    Ok(m) => {
                        loaded_from_mod = true;
                        m
                    }
                    Err(_) => build_initial_planet_or_wd_model(
                        controls.clone(),
                        job.clone(),
                    )?,
                }
            } else {
                build_initial_planet_or_wd_model(controls.clone(), job.clone())?
            }
        } else {
            load_zams_model(controls.clone(), job.clone())?
        }
    } else if case_name == "make_env" && existing_model.is_none() {
        let mut env_controls = controls.clone();
        if let Some(&m_env) = controls.x_ctrl.first().filter(|&&m| m > 0.0) {
            env_controls.initial_mass = m_env.clamp(0.55, 15.0);
        }
        let mut m = load_zams_model(env_controls, job.clone())?;
        excise_core_by_temperature(&mut m, 4.0e6);
        m
    } else if crate::tier3_extras::is_tier3_case(case_name)
        || crate::tier4_extras::is_tier4_case(case_name)
        || crate::tier5_extras::is_tier5_star_case(case_name)
        || out_of_zams_range
    {
        if let Some(prev) = existing_model.filter(|p| !p.controls.rsp_flag) {
            prev
        } else {
            let mut init_controls = controls.clone();
            let candidate_name =
                if job.load_saved_model && !job.saved_model_name.is_empty() {
                    Some(job.saved_model_name.as_str())
                } else if !controls.zams_filename.is_empty() {
                    Some(controls.zams_filename.as_str())
                } else {
                    None
                };
            if let Some(mname) = candidate_name {
                if let Some(mod_path) =
                    resolve_model_path(&bundle.case_dir, mname)
                {
                    if let Some(m_hdr) = peek_mod_header_mass(&mod_path) {
                        init_controls.initial_mass = m_hdr;
                    }
                }
            }
            if (case_name.starts_with("rsp_") || job.create_rsp_model)
                && init_controls.rsp_mass > 0.0
            {
                init_controls.initial_mass = init_controls.rsp_mass;
            }
            build_initial_planet_or_wd_model(init_controls, job.clone())?
        }
    } else if job.create_initial_model
        || job.use_other_build_initial_model
        || job.relax_initial_mass_to_remove_h_env
        || (job.relax_mass && job.new_mass > 0.0 && job.new_mass <= 0.3)
        || (job.relax_irradiation && controls.initial_mass <= 0.02)
    {
        build_initial_planet_or_wd_model(controls.clone(), job.clone())?
    } else if job.load_saved_model && !job.saved_model_name.is_empty() {
        if let Some(prev) = existing_model {
            prev
        } else if let Some(mod_path) =
            resolve_model_path(&bundle.case_dir, &job.saved_model_name)
        {
            let text = fs::read_to_string(&mod_path).map_err(|e| {
                StarError::IoError(format!(
                    "failed to read saved model {}: {e}",
                    mod_path.display()
                ))
            })?;
            match read_mod_str(&text, controls.clone(), job.clone()) {
                Ok(m) => {
                    loaded_from_mod = true;
                    m
                }
                Err(_) => build_initial_planet_or_wd_model(
                    controls.clone(),
                    job.clone(),
                )?,
            }
        } else {
            build_initial_planet_or_wd_model(controls.clone(), job.clone())?
        }
    } else if matches!(case_name, "make_he_wd" | "make_co_wd" | "make_o_ne_wd")
        && existing_model.is_none()
    {
        build_initial_planet_or_wd_model(controls.clone(), job.clone())?
    } else if job.create_pre_ms_model {
        build_pre_ms_model(controls.clone(), job.clone())?
    } else if let Some(prev) = existing_model {
        prev
    } else {
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
    let explicit_rti_in_controls = bundle
        .sections
        .get("controls")
        .and_then(|c| c.get_bool("rti_flag"))
        .is_some();
    let (preserved_tau, preserved_m, preserved_z, preserved_y, preserved_opac) = (
        model.controls.tau_factor,
        model.controls.initial_mass,
        model.controls.initial_z,
        model.controls.initial_y,
        model.controls.opacity_factor,
    );

    model.job = job.clone();
    model.job.net_name = preserved_net;
    model.controls = controls;
    (
        model.controls.v_flag,
        model.controls.u_flag,
        model.controls.rotation_flag,
    ) = (preserved_v, preserved_u, preserved_rot);
    if !explicit_rti_in_controls {
        model.controls.rti_flag = preserved_rti;
    }
    (
        model.controls.tau_factor,
        model.controls.initial_mass,
        model.controls.initial_z,
        model.controls.initial_y,
    ) = (preserved_tau, preserved_m, preserved_z, preserved_y);
    if had_existing_model
        && !loaded_from_mod
        && is_tier2_or_3
        && (model.controls.opacity_factor - 1.0).abs() < 1.0e-12
    {
        model.controls.opacity_factor = preserved_opac;
    }
    model.eos_config = bundle.eos_config;
    model.kap_config = bundle.kap_config.clone();

    // 1. Reaction network switching (`change_net` / `change_initial_net`)
    if job.change_net && !job.new_net_name.is_empty() {
        change_model_network(
            &mut model,
            &job.new_net_name,
            Some(&bundle.case_dir),
        )?;
    }

    // 2. Composition and external profile relaxation
    if case_name != "tzo" {
        apply_composition_relaxation(&mut model, &job);
    }
    relax_from_profile_files(&mut model, &bundle.case_dir);

    // 3. Mass and H-envelope stripping relaxation
    if job.relax_mass && job.new_mass > 0.0 {
        relax_mass_to_target(&mut model, job.new_mass, 8, is_tier2_or_3);
    } else if job.relax_initial_mass_to_remove_h_env {
        relax_to_remove_h_env(&mut model, 1.0e-4, 8, is_tier2_or_3);
    }

    // 4. Inner boundary, irradiation, and optical-depth relaxation
    if job.relax_m_center && job.new_core_mass > 0.0 && case_name != "tzo" {
        relax_core_boundary(
            &mut model,
            job.new_core_mass,
            job.core_avg_rho,
            job.core_avg_eps,
            8,
            is_tier2_or_3,
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
    if job.set_initial_model_number {
        model.model_number = job.initial_model_number;
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

    extras.extras_controls(&mut model);
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
    let did_mass_or_core_relax = job.relax_mass
        || job.relax_initial_mass_to_remove_h_env
        || job.relax_m_center;
    if !loaded_from_mod
        && is_tier2_or_3
        && (!had_existing_model || did_mass_or_core_relax)
        && (model.m_center == 0.0 || job.relax_m_center)
    {
        equilibrate_tier2_scaled_model(&mut model);
    }
    if let Some(t_cut) = job.remove_initial_center_by_temperature {
        excise_core_by_temperature(&mut model, t_cut);
    }
    if model.controls.rsp2_flag {
        crate::rsp2_remesh::seed_rsp2_w_from_mlt_vc_if_needed(&mut model);
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
    if job.relax_mass && job.new_mass > 0.3 && !is_tier2_or_3 {
        model.reconstruct_ln_r_from_lnd_and_dm();
        model.sync_mesh_and_pack()?;
    }
    if model.controls.use_other_cgrav
        && !crate::tier3_extras::is_tier3_case(case_name)
    {
        extras.other_cgrav(&mut model);
    }

    // Apply rotational spin-up AFTER equilibration and `sync_mesh_and_pack` so
    // `model.r[0]` and `model.cgrav[0]` match the active stellar structure.
    if let Some(v_surf) = job.new_surface_rotation_v.filter(|&v| v > 0.0) {
        let r_surf = model.r.first().copied().unwrap_or(R_SUN).max(1.0);
        apply_uniform_rotation(&mut model, (v_surf * 1.0e5) / r_surf);
    } else if let Some(w_div_wc) =
        job.new_omega_div_omega_crit.filter(|&w| w > 0.0)
    {
        let r_surf = model.r.first().copied().unwrap_or(R_SUN).max(1.0);
        let g_surf = model
            .cgrav
            .first()
            .copied()
            .filter(|&g| g > 0.0)
            .unwrap_or(STANDARD_CGRAV);
        let omega_crit = (g_surf * model.mstar / r_surf.powi(3)).sqrt();
        apply_uniform_rotation(&mut model, w_div_wc * omega_crit);
    } else if let Some(omega_0) = job.new_omega.filter(|&w| w > 0.0) {
        apply_uniform_rotation(&mut model, omega_0);
    }

    set_hydro_vars(&mut model, false, false)?;
    if model.controls.rsp2_flag {
        let (e0, _) = crate::rsp_driver::rsp_total_energy_integrals(&mut model);
        model.aux.total_energy = e0;
        model.aux.rsp.total_energy_old = e0;
        model.aux.rsp.time_sec = model.star_age * SECYER;
        model.aux.rsp.time_old_sec = model.aux.rsp.time_sec;
    }
    model.save_start_values();
    model.save_old_values();
    if loaded_from_mod && crate::tier2_extras::is_tier2_case(case_name) {
        let needs_fallback = matches!(
            case_name,
            "irradiated_planet" | "wd_acc_small_dm" | "wd_c_core_ignition"
        ) || match crate::hydro_eqns::eval_henyey_equations(
            &mut model, false,
        ) {
            Ok(sys) => crate::solver::compute_vector_norms(&sys.equ).1 > 0.25,
            Err(_) => true,
        };
        if needs_fallback {
            let mut fallback = build_initial_planet_or_wd_model(
                bundle.controls.clone(),
                bundle.job.clone(),
            )?;
            fallback.job = model.job.clone();
            fallback.controls = model.controls.clone();
            fallback.eos_config = model.eos_config;
            fallback.kap_config = model.kap_config.clone();
            if job.relax_m_center && job.new_core_mass > 0.0 {
                let old_xmstar = fallback.xmstar.max(1.0);
                apply_planetary_core(
                    &mut fallback,
                    job.new_core_mass,
                    job.core_avg_rho,
                    job.core_avg_eps,
                );
                for k in 0..fallback.nz {
                    fallback.lum[k] *= fallback.xmstar / old_xmstar;
                }
            }
            extras.extras_controls(&mut fallback);
            equilibrate_tier2_scaled_model(&mut fallback);
            fallback.sync_mesh_and_pack()?;
            set_hydro_vars(&mut fallback, false, false)?;
            fallback.save_start_values();
            fallback.save_old_values();
            model = fallback;
        }
    }
    extras.extras_startup(&mut model, false)?;

    Ok(model)
}
