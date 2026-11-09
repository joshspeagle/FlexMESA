// SPDX-License-Identifier: LGPL-3.0-or-later
//! `&star_job` model initialization, network switching, composition/mass/core
//! relaxation, and rotational spin-up handlers (`init_or_update_model_from_bundle`,
//! corresponding to `star/private/init.f90` and `star/private/relax.f90`).

use std::fs;
use std::path::Path;

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{M_SUN, R_SUN, SECYER, STANDARD_CGRAV};

use crate::controls::{StarControls, StarError, StarJobControls, VarLayout};
use crate::hydro_vars::set_hydro_vars;
use crate::inlist::InlistBundle;
use crate::inlist_runner::StarExtras;
use crate::micro::change_model_network;
use crate::model::StarModel;
use crate::model_io::read_mod_str;
use crate::pre_ms::build_pre_ms_model;
use crate::tier2_extras::apply_planetary_core;
use crate::zams::load_zams_model;

/// Resolves a `.mod` model file path against `case_dir`.
fn resolve_model_path(
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

/// Builds an initial planet, brown dwarf, or conductive-flame model when
/// `job.create_initial_model` or `job.use_other_build_initial_model` is true
/// (`star/private/create_initial_model.f90`).
fn build_initial_planet_or_wd_model(
    mut controls: StarControls,
    job: StarJobControls,
) -> Result<StarModel, StarError> {
    let m_target = if job.mass_in_gm_for_create_initial_model > 0.0 {
        job.mass_in_gm_for_create_initial_model / M_SUN
    } else if job.relax_mass && job.new_mass > 0.0 {
        job.new_mass
    } else {
        controls.initial_mass
    };
    let r_init_cm = job.radius_in_cm_for_create_initial_model;
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

/// Equilibrates a scaled Tier 2 model so that hydrostatic, density, and
/// radiative/convective temperature equations remain balanced for arbitrary
/// substellar, white-dwarf, or super-AGB masses and core boundaries.
fn equilibrate_tier2_scaled_model(model: &mut StarModel) {
    let alpha_m = (model.xmstar / M_SUN).max(1.0e-6);
    model.r_center = 0.0;
    model.l_center = 0.0;
    model.set_m_and_dm();
    model.reconstruct_ln_r_from_lnd_and_dm();
    model.set_m_grav_and_grav();
    for k in 0..model.nz {
        let m_zams_k = model.q[k] * M_SUN;
        let m_ratio = (alpha_m * m_zams_k) / model.m_grav[k].max(1.0);
        model.cgrav[k] =
            STANDARD_CGRAV * alpha_m.powf(-2.0 / 3.0) * m_ratio;
    }
    model.controls.opacity_factor = alpha_m.powf(-2.0 / 3.0);
    model.set_m_grav_and_grav();
}

/// Applies uniform composition overrides (`set_uniform_initial_composition`) or
/// $(Y, Z)$ relaxation (`relax_y`, `relax_z`) to `model.xa`.
fn apply_composition_relaxation(model: &mut StarModel, job: &StarJobControls) {
    let catalog = NuclideCatalog::builtin_ref();
    let h1_id = catalog.get_nuclide_index("h1").unwrap_or(1);
    let h2_id = catalog.get_nuclide_index("h2").unwrap_or(2);
    let he3_id = catalog.get_nuclide_index("he3").unwrap_or(3);
    let he4_id = catalog.get_nuclide_index("he4").unwrap_or(4);

    if job.set_uniform_initial_composition {
        let h1 = job.initial_h1.max(0.0);
        let h2 = job.initial_h2.max(0.0);
        let he3 = job.initial_he3.max(0.0);
        let he4 = job.initial_he4.max(0.0);
        let z_rem = (1.0 - (h1 + h2 + he3 + he4)).max(0.0);

        let metal_count = model
            .chem_id
            .iter()
            .filter(|&&id| {
                id != h1_id && id != h2_id && id != he3_id && id != he4_id
            })
            .count()
            .max(1);

        for k in 0..model.nz {
            let species = model.species;
            for j in 0..species {
                let cid = model.chem_id[j];
                model.xa[k * species + j] = if cid == h1_id {
                    h1
                } else if cid == h2_id {
                    h2
                } else if cid == he3_id {
                    he3
                } else if cid == he4_id {
                    he4
                } else {
                    z_rem / (metal_count as f64)
                };
            }
            let row = model.cell_xa_mut(k);
            let s: f64 = row.iter().sum();
            if s > 0.0 {
                for v in row.iter_mut() {
                    *v /= s;
                }
            }
        }
        return;
    }

    if job.relax_y || job.relax_z {
        let target_z = if job.relax_z && job.new_z >= 0.0 {
            Some(job.new_z.clamp(0.0, 1.0))
        } else {
            None
        };
        let target_y = if job.relax_y && job.new_y >= 0.0 {
            Some(job.new_y.clamp(0.0, 1.0))
        } else {
            None
        };

        for k in 0..model.nz {
            let species = model.species;
            let mut x_cur = 0.0;
            let mut y_cur = 0.0;
            let mut z_cur = 0.0;
            for j in 0..species {
                let cid = model.chem_id[j];
                let val = model.xa[k * species + j];
                if cid == h1_id || cid == h2_id {
                    x_cur += val;
                } else if cid == he3_id || cid == he4_id {
                    y_cur += val;
                } else {
                    z_cur += val;
                }
            }
            let z_new = target_z.unwrap_or(z_cur);
            let y_new = target_y.unwrap_or(y_cur).min((1.0 - z_new).max(0.0));
            let x_new = (1.0 - y_new - z_new).max(0.0);

            for j in 0..species {
                let cid = model.chem_id[j];
                let val = model.xa[k * species + j];
                model.xa[k * species + j] = if cid == h1_id || cid == h2_id {
                    if x_cur > 1.0e-15 {
                        val * (x_new / x_cur)
                    } else if cid == h1_id {
                        x_new
                    } else {
                        0.0
                    }
                } else if cid == he3_id || cid == he4_id {
                    if y_cur > 1.0e-15 {
                        val * (y_new / y_cur)
                    } else if cid == he4_id {
                        y_new
                    } else {
                        0.0
                    }
                } else if z_cur > 1.0e-15 {
                    val * (z_new / z_cur)
                } else {
                    z_new / (species.saturating_sub(2).max(1) as f64)
                };
            }
            let row = model.cell_xa_mut(k);
            let s: f64 = row.iter().sum();
            if s > 0.0 {
                for v in row.iter_mut() {
                    *v /= s;
                }
            }
        }
    }
}

fn apply_uniform_rotation(model: &mut StarModel, omega_0: f64) {
    model.controls.rotation_flag = true;
    if model.omega.len() != model.nz {
        model.omega.resize(model.nz, 0.0);
        model.j_rot.resize(model.nz, 0.0);
    }
    for k in 0..model.nz {
        model.omega[k] = omega_0;
        model.j_rot[k] = (2.0 / 3.0) * model.r[k].powi(2) * omega_0;
    }
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
    existing_model: Option<StarModel>,
    extras: &mut dyn StarExtras,
) -> Result<StarModel, StarError> {
    let job = bundle.job.clone();
    let controls = bundle.controls.clone();
    let case_name = bundle
        .case_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let mut loaded_from_mod = false;
    let mut model = if job.create_initial_model
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
    let preserved_v_flag = model.controls.v_flag;
    let preserved_u_flag = model.controls.u_flag;
    let preserved_rot_flag = model.controls.rotation_flag;
    let preserved_tau_factor = model.controls.tau_factor;
    let preserved_init_mass = model.controls.initial_mass;
    let preserved_init_z = model.controls.initial_z;
    let preserved_init_y = model.controls.initial_y;

    model.job = job.clone();
    model.job.net_name = preserved_net;
    model.controls = controls;
    model.controls.v_flag = preserved_v_flag;
    model.controls.u_flag = preserved_u_flag;
    model.controls.rotation_flag = preserved_rot_flag;
    model.controls.tau_factor = preserved_tau_factor;
    model.controls.initial_mass = preserved_init_mass;
    model.controls.initial_z = preserved_init_z;
    model.controls.initial_y = preserved_init_y;
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

    // 2. Composition relaxation (`set_uniform_initial_composition`, `relax_y`, `relax_z`)
    apply_composition_relaxation(&mut model, &job);

    // 3. Mass relaxation (`relax_mass`, `relax_initial_mass`, `relax_mass_scale`)
    if job.relax_mass && job.new_mass > 0.0 {
        let old_mstar = model.mstar.max(1.0);
        let old_xmstar = model.xmstar.max(1.0);
        let new_mstar = job.new_mass * M_SUN;
        if (new_mstar - old_mstar).abs() > 1.0e-9 * old_mstar {
            let dln_r = (new_mstar / old_mstar).ln() / 3.0;
            model.mstar = new_mstar;
            model.xmstar = (new_mstar - model.m_center).max(0.0);
            model.mstar_start = model.mstar;
            model.mstar_old = model.mstar;
            for k in 0..model.nz {
                model.ln_r[k] += dln_r;
                if crate::tier2_extras::is_tier2_case(case_name) {
                    model.lum[k] *= model.xmstar / old_xmstar;
                }
            }
        }
        model.controls.initial_mass = job.new_mass;
    }

    // 4. Inner boundary relaxation (`relax_m_center`, `relax_r_center`, `relax_l_center`)
    if job.relax_m_center && job.new_core_mass > 0.0 {
        let old_xmstar = model.xmstar.max(1.0);
        apply_planetary_core(
            &mut model,
            job.new_core_mass,
            job.core_avg_rho,
            job.core_avg_eps,
        );
        if crate::tier2_extras::is_tier2_case(case_name) {
            for k in 0..model.nz {
                model.lum[k] *= model.xmstar / old_xmstar;
            }
        }
    }
    if job.relax_r_center && job.new_r_center > 0.0 {
        model.r_center = job.new_r_center;
    }
    if job.relax_l_center && job.new_l_center >= 0.0 {
        model.l_center = job.new_l_center;
    }
    if job.relax_irradiation && job.new_irradiation_flux > 0.0 {
        model.controls.irradiation_flux = job.new_irradiation_flux;
        if job.column_depth_for_irradiation > 0.0 {
            model.controls.column_depth_for_irradiation =
                job.column_depth_for_irradiation;
        }
    }

    // 5. Surface optical depth factor (`set_tau_factor`)
    if job.set_tau_factor && job.set_to_this_tau_factor > 0.0 {
        model.controls.tau_factor = job.set_to_this_tau_factor;
    }

    // Pure-envelope integration mode (`make_env`): when nuclear energy generation
    // is disabled (`eps_nuc_factor == 0.0`) and `l_center == 0.0`, supply core
    // luminosity at the inner boundary so radiative/convective flux stays balanced.
    if model.controls.eps_nuc_factor == 0.0
        && model.l_center == 0.0
        && model.lum.first().copied().unwrap_or(0.0) > 0.0
    {
        model.l_center = model.lum[0];
        if !job.set_initial_dt || model.dt < 1.0e2 {
            model.dt = 1.0e2 * SECYER;
            model.dt_next = model.dt;
        }
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
    if let Some(v_surf) = job.new_surface_rotation_v.filter(|&v| v > 0.0) {
        let r_surf = model.r.first().copied().unwrap_or(R_SUN).max(1.0);
        apply_uniform_rotation(&mut model, (v_surf * 1.0e5) / r_surf);
    } else if let Some(w_div_wc) =
        job.new_omega_div_omega_crit.filter(|&w| w > 0.0)
    {
        let r_surf = model.r.first().copied().unwrap_or(R_SUN).max(1.0);
        let omega_crit = (STANDARD_CGRAV * model.mstar / r_surf.powi(3)).sqrt();
        apply_uniform_rotation(&mut model, w_div_wc * omega_crit);
    } else if let Some(omega_0) = job.new_omega.filter(|&w| w > 0.0) {
        apply_uniform_rotation(&mut model, omega_0);
    }

    // Rebuild `VarLayout` whenever `v_flag`, `u_flag`, or `species` changed
    model.layout = VarLayout::new(
        model.species,
        model.controls.v_flag,
        model.controls.u_flag,
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
    if !loaded_from_mod && crate::tier2_extras::is_tier2_case(case_name) {
        equilibrate_tier2_scaled_model(&mut model);
    }
    model.sync_mesh_and_pack()?;
    if job.relax_mass && job.new_mass > 0.3 && !crate::tier2_extras::is_tier2_case(case_name) {
        model.reconstruct_ln_r_from_lnd_and_dm();
        model.sync_mesh_and_pack()?;
    }
    if model.controls.use_other_cgrav {
        extras.other_cgrav(&mut model);
    }
    set_hydro_vars(&mut model, false, false)?;
    model.save_start_values();
    model.save_old_values();
    if loaded_from_mod && crate::tier2_extras::is_tier2_case(case_name) {
        let needs_fallback = matches!(
            case_name,
            "irradiated_planet" | "wd_acc_small_dm" | "wd_c_core_ignition"
        ) || match crate::hydro_eqns::eval_henyey_equations(
            &mut model, false,
        ) {
            Ok(sys) => {
                let (_, max_res) =
                    crate::solver::compute_vector_norms(&sys.equ);
                max_res > 0.25
            }
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
