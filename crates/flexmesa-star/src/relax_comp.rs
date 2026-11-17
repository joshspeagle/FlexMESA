// SPDX-License-Identifier: LGPL-3.0-or-later
//! Multi-step Henyey composition equilibration (`do_relax_Z`, `do_relax_Y`, and
//! `do_internal_evolve` in `star/private/relax.f90:3411-3700, 3859-4161`).

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{LSUN, R_SUN, SECYER, STANDARD_CGRAV};
use flexmesa_turb::MltOption;

use crate::controls::{EnergyEqnOption, StarError, StarJobControls};
use crate::core_boundaries::{
    compute_power_categories, snapshot_step_history_old,
};
use crate::evolve_support::eval_total_energy_integrals;
use crate::hydro_vars::set_hydro_vars;
use crate::micro::set_basic_vars;
use crate::model::StarModel;
use crate::relax::apply_uniform_rotation;
use crate::rotation::update_rotation_distortion_factors;
use crate::struct_burn_mix::do_struct_burn_mix;

fn mass_weighted_mean_yz(model: &StarModel) -> (f64, f64) {
    let cat = NuclideCatalog::builtin_ref();
    let h1 = cat.get_nuclide_index("h1").unwrap_or(1);
    let h2 = cat.get_nuclide_index("h2").unwrap_or(2);
    let he3 = cat.get_nuclide_index("he3").unwrap_or(3);
    let he4 = cat.get_nuclide_index("he4").unwrap_or(4);
    let (mut y_sum, mut z_sum, mut dm_tot) = (0.0, 0.0, 0.0);
    for k in 0..model.nz {
        let dm = model.dm[k].max(0.0);
        dm_tot += dm;
        for j in 0..model.species {
            let cid = model.chem_id[j];
            let x = model.xa[k * model.species + j];
            if cid == he3 || cid == he4 {
                y_sum += dm * x;
            } else if cid != h1 && cid != h2 {
                z_sum += dm * x;
            }
        }
    }
    if dm_tot > 0.0 {
        (y_sum / dm_tot, z_sum / dm_tot)
    } else {
        (0.28, 0.02)
    }
}

fn set_profile_yz(
    model: &mut StarModel,
    target_y: Option<f64>,
    target_z: Option<f64>,
) {
    let cat = NuclideCatalog::builtin_ref();
    let h1 = cat.get_nuclide_index("h1").unwrap_or(1);
    let h2 = cat.get_nuclide_index("h2").unwrap_or(2);
    let he3 = cat.get_nuclide_index("he3").unwrap_or(3);
    let he4 = cat.get_nuclide_index("he4").unwrap_or(4);
    let sp = model.species;
    for k in 0..model.nz {
        let (mut xc, mut yc, mut zc) = (0.0, 0.0, 0.0);
        for j in 0..sp {
            let cid = model.chem_id[j];
            let v = model.xa[k * sp + j];
            if cid == h1 || cid == h2 {
                xc += v;
            } else if cid == he3 || cid == he4 {
                yc += v;
            } else {
                zc += v;
            }
        }
        let zn = target_z.unwrap_or(zc).clamp(0.0, 1.0);
        let yn = target_y.unwrap_or(yc).clamp(0.0, (1.0 - zn).max(0.0));
        let xn = (1.0 - yn - zn).max(0.0);
        let n_met = model
            .chem_id
            .iter()
            .filter(|&&c| c != h1 && c != h2 && c != he3 && c != he4)
            .count()
            .max(1);
        for j in 0..sp {
            let cid = model.chem_id[j];
            let v = model.xa[k * sp + j];
            model.xa[k * sp + j] = if cid == h1 || cid == h2 {
                if xc > 1.0e-15 {
                    v * (xn / xc)
                } else if cid == h1 {
                    xn
                } else {
                    0.0
                }
            } else if cid == he3 || cid == he4 {
                if yc > 1.0e-15 {
                    v * (yn / yc)
                } else if cid == he4 {
                    yn
                } else {
                    0.0
                }
            } else if zc > 1.0e-15 {
                v * (zn / zc)
            } else {
                zn / (n_met as f64)
            };
        }
        let row = model.cell_xa_mut(k);
        let s: f64 = row.iter().sum();
        if s > 0.0 && (s - 1.0).abs() > 1.0e-12 {
            for v in row.iter_mut() {
                *v /= s;
            }
        }
    }
}

fn step_relax_henyey(
    model: &mut StarModel,
    dt_s: f64,
) -> Result<bool, StarError> {
    let backup = model.clone();
    set_basic_vars(model)?;
    set_hydro_vars(model, false, false)?;
    model.save_old_values();
    model.save_start_values()?;
    match do_struct_burn_mix(model, dt_s, false, true, &[]) {
        Ok(rep) if rep.converged => Ok(true),
        _ => {
            *model = backup;
            Ok(false)
        }
    }
}

/// Ramps $(Y, Z)$ across steps with `doing_relax = true`, `dxdt_nuc_factor = 0.0`,
/// `MltOption::Cox`, and winds/mass-change disabled, solving coupled Henyey
/// structure to hydrostatic and thermal equilibrium before resetting counters
/// (`star/private/relax.f90:3411-3700, 3859-4161`).
pub fn relax_composition_with_henyey(
    model: &mut StarModel,
    job: &StarJobControls,
) -> Result<(), StarError> {
    if !(job.relax_z || job.relax_y) || model.nz == 0 || model.species == 0 {
        return Ok(());
    }
    let (y0, z0) = mass_weighted_mean_yz(model);
    let tz = job
        .relax_z
        .then_some(job.new_z)
        .filter(|&z| (0.0..=1.0).contains(&z));
    let ty = job
        .relax_y
        .then_some(job.new_y)
        .filter(|&y| (0.0..=1.0).contains(&y));

    let saved_controls = model.controls.clone();
    let omega_0 = if saved_controls.rotation_flag {
        model.omega.first().copied().unwrap_or(0.0)
    } else {
        0.0
    };

    model.aux.doing_relax = true;
    apply_uniform_rotation(model, 0.0);
    model.controls.rotation_flag = false;
    update_rotation_distortion_factors(model);

    model.controls.dxdt_nuc_factor = 0.0;
    model.controls.op_split_burn = false;
    model.controls.do_burn = true;
    model.controls.eps_nuc_factor = 1.0;
    model.controls.do_mix = false;
    model.controls.mix_factor = 0.0;
    model.controls.mlt_option = MltOption::Cox;
    model.controls.use_ledoux_criterion = false;
    model.controls.energy_eqn_option = EnergyEqnOption::EpsGrav;
    model.controls.mass_change = 0.0;
    model.controls.use_other_wind = false;
    model.controls.use_other_adjust_mdot = false;
    model.controls.reimers_scaling_factor = 0.0;
    model.controls.blocker_scaling_factor = 0.0;
    model.controls.vink_scaling_factor = 0.0;
    model.controls.dutch_scaling_factor = 0.0;
    model.controls.cool_wind_rgb_scheme.clear();
    model.controls.cool_wind_agb_scheme.clear();
    model.controls.mix_wind.hot_wind_scheme.clear();
    model.controls.use_gold_tolerances = false;
    model.controls.use_gold2_tolerances = false;
    model.controls.steps_before_use_gold_tolerances = -1;
    model.controls.steps_before_use_gold2_tolerances = -1;
    model.controls.max_newton_iters = saved_controls.max_newton_iters.max(60);
    model.mstar_dot = 0.0;

    if let Some(z_end) = tz {
        let z_lo = z0.max(1.0e-8);
        let z_hi = z_end.max(1.0e-8);
        let n_z = ((flexmesa_math::log(z_hi / z_lo).abs() / 0.25).ceil()
            as usize)
            .max(1);
        for s in 1..=n_z {
            let z_s = if s == n_z {
                z_end
            } else {
                z_lo * flexmesa_math::exp(
                    flexmesa_math::log(z_hi / z_lo) * (s as f64) / (n_z as f64),
                )
            };
            set_profile_yz(model, None, Some(z_s));
            let _ = step_relax_henyey(model, SECYER)?;
        }
    }
    if let Some(y_end) = ty {
        let n_y = (((y_end - y0).abs() / 0.01).ceil() as usize).max(1);
        for s in 1..=n_y {
            let y_s = if s == n_y {
                y_end
            } else {
                y0 + (y_end - y0) * (s as f64) / (n_y as f64)
            };
            set_profile_yz(model, Some(y_s), tz);
            let _ = step_relax_henyey(model, SECYER)?;
        }
    }

    let mut dt_yr = 50.0_f64;
    for step in 0..30_usize {
        set_hydro_vars(model, false, false)?;
        let lnuc = (compute_power_categories(model).power_nuc_burn * LSUN)
            .abs()
            .max(1.0);
        let lsurf = model.lum.first().copied().unwrap_or(1.0).abs().max(1.0);
        if step >= 2 && flexmesa_math::log10(lnuc / lsurf).abs() < 0.003 {
            break;
        }
        if step_relax_henyey(model, dt_yr * SECYER)? {
            dt_yr = (dt_yr * 1.8).min(5.0e4);
        } else {
            dt_yr = (dt_yr * 0.5).max(1.0);
        }
    }

    model.controls = saved_controls;
    if let Some(z_end) = tz {
        model.controls.initial_z = z_end;
    }
    if let Some(y_end) = ty {
        model.controls.initial_y = y_end;
    }
    if model.controls.rotation_flag {
        let r_s = model.r.first().copied().unwrap_or(R_SUN).max(1.0);
        let om_t = if let Some(v_s) =
            job.new_surface_rotation_v.filter(|&v| v > 0.0)
        {
            (v_s * 1.0e5) / r_s
        } else if let Some(w_wc) =
            job.new_omega_div_omega_crit.filter(|&w| w > 0.0)
        {
            let g_s = model
                .cgrav
                .first()
                .copied()
                .filter(|&g| g > 0.0)
                .unwrap_or(STANDARD_CGRAV);
            w_wc * (g_s * model.mstar / flexmesa_math::pow3(r_s)).sqrt()
        } else {
            omega_0
        };
        if om_t > 0.0 {
            apply_uniform_rotation(model, om_t);
        }
    }
    update_rotation_distortion_factors(model);

    model.aux.doing_relax = false;
    model.star_age = 0.0;
    model.star_age_old = 0.0;
    model.model_number = 1;
    model.num_retries = 0;
    model.total_num_solver_iterations = 0;
    model.mstar_dot = 0.0;
    model.mstar_start = model.mstar;
    model.mstar_old = model.mstar;
    model.dt = SECYER;
    model.dt_next = SECYER;
    model.dt_old = 0.0;
    model.dt_limit_ratio_old = 0.0;
    model.aux.step_hist.timestep_hold = 0;
    model.aux.step_hist.model_number_for_last_retry = None;

    model.pack_xh();
    set_basic_vars(model)?;
    set_hydro_vars(model, false, false)?;
    let etot = eval_total_energy_integrals(model).5;
    model.aux.total_energy = etot;
    model.aux.cumulative_energy_error = 0.0;
    snapshot_step_history_old(model);
    model.aux.step_hist.total_energy_old = etot;
    model.aux.step_hist.total_energy_end = etot;
    model.aux.step_hist.error_in_energy_conservation = 0.0;
    model.aux.step_hist.cumulative_energy_error_old = 0.0;
    model.save_start_values()?;
    model.save_old_values();
    Ok(())
}
