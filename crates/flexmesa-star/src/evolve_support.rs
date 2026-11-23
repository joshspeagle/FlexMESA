// SPDX-License-Identifier: LGPL-3.0-or-later
//! Pre-step operator-split physics hooks (`apply_step_pre_physics`),
//! domain total-energy integration (`eval_total_energy_integrals`), and
//! discrete energy-conservation residual accounting
//! (`update_step_energy_conservation`) corresponding to
//! `star/private/evolve.f90:605-665,1101-1373,1646-1681` and
//! `star/private/star_utils.f90:2452-2612`.

use flexmesa_const::MixingType;
use flexmesa_turb::MltOption;

use crate::cell_energy::{
    cell_rsp2_eturb, cell_specific_ke, cell_specific_pe,
    cell_specific_rotational_energy, cell_specific_total_energy,
};
use crate::controls::StarError;
use crate::conv_premix::do_conv_premix;
use crate::element_diffusion::{
    do_element_diffusion_step, finish_element_diffusion,
};
use crate::hydro_vars::set_hydro_vars;
use crate::mesh_adjust::{MeshFunctionConfig, do_adjust_mesh};
use crate::model::StarModel;
use crate::phase_separation::do_phase_separation_step;
use crate::primary_state::PrimaryStateBuffer;
use crate::rotation::conserve_shell_j_rot_and_distortion;

/// Evaluates domain-integrated `(e_int, e_grav, e_kin, e_rot, e_turb, e_tot)`
/// over zero-based zones `klo..khi` with optional mass cap `delta_m`, optional
/// TDC turbulent energy inclusion, and optional per-cell profile recording
/// (`eval_deltaM_total_energy_integrals` and `eval_total_energy_profile` in
/// `star/private/star_utils.f90:2439-2560`).
#[must_use]
pub(crate) fn eval_energy_integrals_kernel(
    model: &StarModel,
    klo: usize,
    khi: usize,
    delta_m: Option<f64>,
    include_tdc_eturb: bool,
    save_profile: bool,
) -> (Vec<f64>, (f64, f64, f64, f64, f64, f64)) {
    let nz = model.nz;
    if khi > nz || klo >= khi {
        return (Vec::new(), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
    }
    let mut profile = if save_profile {
        vec![0.0_f64; nz]
    } else {
        Vec::new()
    };
    let (
        mut total_internal_energy,
        mut total_gravitational_energy,
        mut total_radial_kinetic_energy,
        mut total_rotational_kinetic_energy,
        mut total_turbulent_energy,
    ) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);

    let tdc_turb = include_tdc_eturb
        && model.controls.mlt_option == MltOption::Tdc
        && model.controls.tdc_include_eturb_in_energy_equation;

    // Matches `star/private/star_utils.f90:2467-2514` verbatim (upstream tests
    // `sum_dm` against `deltaM` and clips `dm` if `sum_dm + dm > deltaM`
    // without incrementing `sum_dm`).
    let sum_dm = 0.0_f64;
    for k in klo..khi {
        let mut dm = model.dm[k];
        if let Some(cap) = delta_m {
            if sum_dm >= cap {
                break;
            }
            if sum_dm + dm > cap {
                dm = cap - sum_dm;
            }
        }
        let mut cell_total = 0.0_f64;
        let mut cell1 = dm * model.energy[k];
        cell_total += cell1;
        total_internal_energy += cell1;

        if model.controls.v_flag || model.controls.u_flag {
            cell1 = dm * cell_specific_ke(model, k);
            cell_total += cell1;
            total_radial_kinetic_energy += cell1;
        }

        cell1 = dm * cell_specific_pe(model, k);
        cell_total += cell1;
        total_gravitational_energy += cell1;

        if model.controls.rotation_flag {
            cell1 = dm * cell_specific_rotational_energy(model, k);
            total_rotational_kinetic_energy += cell1;
            if model.controls.include_rotation_in_total_energy {
                cell_total += cell1;
            }
        }

        if model.controls.rsp2_flag {
            cell1 = dm * cell_rsp2_eturb(model, k);
            cell_total += cell1;
            total_turbulent_energy += cell1;
        } else if tdc_turb {
            let vc_00 = model.mlt_vc.get(k).copied().unwrap_or(0.0);
            let tdc_eturb_cell = if k + 1 < nz {
                let vc_p1 = model.mlt_vc.get(k + 1).copied().unwrap_or(0.0);
                0.75 * (flexmesa_math::pow2(vc_00) + flexmesa_math::pow2(vc_p1))
            } else {
                0.75 * flexmesa_math::pow2(vc_00)
            };
            cell1 = dm * tdc_eturb_cell;
            cell_total += cell1;
            total_turbulent_energy += cell1;
        }

        if model.controls.rsp_flag {
            cell1 = dm * model.aux.rsp.et_rsp.get(k).copied().unwrap_or(0.0);
            cell_total += cell1;
            total_turbulent_energy += cell1;
        }

        if save_profile {
            profile[k] = cell_total;
        }
    }

    let mut sum_total = total_internal_energy
        + total_gravitational_energy
        + total_radial_kinetic_energy
        + total_turbulent_energy;
    if model.controls.include_rotation_in_total_energy {
        sum_total += total_rotational_kinetic_energy;
    }

    (
        profile,
        (
            total_internal_energy,
            total_gravitational_energy,
            total_radial_kinetic_energy,
            total_rotational_kinetic_energy,
            total_turbulent_energy,
            sum_total,
        ),
    )
}

/// Evaluates `eval_deltaM_total_energy_integrals`
/// (`star/private/star_utils.f90:2439-2522`) over zero-based zones
/// `klo..khi` up to mass `delta_m` ($\mathrm{g}$), returning
/// `(profile, (e_int, e_grav, e_kin, e_rot, e_turb, e_tot))` in $\mathrm{erg}$.
///
/// Uses [`crate::cell_energy::cell_specific_pe`] and
/// [`crate::cell_energy::cell_specific_ke`] (`real(qp)` arithmetic via
/// [`flexmesa_math::Quad`]) and matches upstream's exact addition order and
/// `include_rotation_in_total_energy`, `RSP2`, `TDC`, and `RSP` rules.
#[must_use]
pub fn eval_delta_m_total_energy_integrals(
    model: &StarModel,
    klo: usize,
    khi: usize,
    delta_m: f64,
    save_profile: bool,
) -> (Vec<f64>, (f64, f64, f64, f64, f64, f64)) {
    eval_energy_integrals_kernel(
        model,
        klo,
        khi,
        Some(delta_m),
        true,
        save_profile,
    )
}

/// Evaluates domain-integrated `(e_int, e_grav, e_kin, e_rot, e_turb, e_tot)`
/// in $\mathrm{erg}$ (`eval_total_energy_integrals` in
/// `star/private/star_utils.f90:2582-2599`).
#[must_use]
pub fn eval_total_energy_integrals(
    model: &StarModel,
) -> (f64, f64, f64, f64, f64, f64) {
    eval_delta_m_total_energy_integrals(model, 0, model.nz, model.mstar, false)
        .1
}

/// Accumulates per-step `error_in_energy_conservation` into
/// `cumulative_energy_error` (`star/private/evolve.f90:1359-1365`).
#[must_use]
pub(crate) fn next_cumulative_energy_error(
    model: &StarModel,
    err_e: f64,
) -> f64 {
    if model.controls.absolute_cumulative_energy_err {
        model.aux.step_hist.cumulative_energy_error_old + err_e.abs()
    } else {
        model.aux.step_hist.cumulative_energy_error_old + err_e
    }
}

/// Captures beginning-of-Phase-1 (post-remesh) total-energy integrals,
/// surface cell specific total energy, and initial-model energy integral,
/// while zeroing per-attempt energy accumulators
/// (`do_step_part1` in `star/private/evolve.f90:430-453`).
pub fn record_start_of_step_energy_integrals(model: &mut StarModel) {
    let (_, _, _, _, _, e_tot) = eval_total_energy_integrals(model);
    let surf_e = if model.nz > 0 {
        cell_specific_total_energy(model, 0)
    } else {
        0.0
    };
    model.aux.total_energy = e_tot;
    model.aux.non_epsnuc_energy_change_from_split_burn = 0.0;
    let sh = &mut model.aux.step_hist;
    sh.total_energy_old = e_tot;
    sh.surface_cell_specific_total_energy_old = surf_e;
    if !sh.have_initial_energy_integrals {
        sh.total_energy_initial = e_tot;
        sh.have_initial_energy_integrals = true;
    }
    sh.clear_adjust_mass_energy_fields();
    sh.work_outward_at_surface = 0.0;
    sh.work_inward_at_center = 0.0;
    sh.tot_e_equ_err = 0.0;
    sh.used_op_split_burn = false;
}

/// Applies pre-solver operator-split physics (`do_conv_premix`,
/// `do_phase_separation`, `do_element_diffusion`, uniform extra heating, and
/// `do_solve_omega_mix`) corresponding to `do_step_part2` and
/// `check_for_extra_heat` in `star/private/evolve.f90:605-665,1646-1681`.
///
/// # Errors
/// Returns [`StarError`] if refreshing hydrodynamic/microphysics state or
/// pre-step `premix_omega` transport fails.
pub fn apply_step_pre_physics(
    model: &mut StarModel,
    dt: f64,
) -> Result<(), StarError> {
    let mut extras = crate::inlist_runner::DefaultStarExtras;
    apply_step_pre_physics_with_extras(model, dt, &mut extras)
}

/// Runs pre-step operator-split physics hooks with custom
/// [`crate::inlist_runner::StarExtras`] callbacks (`other_torque`, `other_torque_implicit`) forwarded into
/// `do_solve_omega_mix_with_extras` when `premix_omega` is enabled
/// (`star/private/evolve.f90:648-655`).
///
/// # Errors
/// Returns [`StarError`] if refreshing hydrodynamic/microphysics state or
/// pre-step `premix_omega` transport fails.
pub fn apply_step_pre_physics_with_extras(
    model: &mut StarModel,
    dt: f64,
    extras: &mut dyn crate::inlist_runner::StarExtras,
) -> Result<(), StarError> {
    let nz = model.nz;
    model.aux.mixing.okay_to_set_mixing_info = true;
    model.aux.mixing.have_new_cz_bdy_info = false;
    model.aux.phase_sep_mixing_mass = -1.0;
    model.aux.mixing.gradt_excess_alpha_old =
        model.aux.mixing.gradt_excess_alpha;
    model.aux.mixing.crystal_core_boundary_mass_old =
        model.aux.crystal_core_boundary_mass;
    if !model.controls.rsp_flag {
        crate::hydro_vars::set_gradt_excess_alpha(model);
    }

    // Refresh hydro and mixing info before operator-split pre-physics
    // (`evolve.f90:569,606`).
    set_hydro_vars(model, false, false)?;

    if model.controls.do_conv_premix {
        if model.aux.eps_pre_mix.len() != nz {
            model.aux.eps_pre_mix.resize(nz, 0.0);
        }
        model.aux.eps_pre_mix.copy_from_slice(&model.energy[..nz]);
        let saved_dt = model.dt;
        if dt > 0.0 {
            model.dt = dt;
        }
        let pm_res = do_conv_premix(model);
        model.dt = saved_dt;
        let _ = pm_res?;
        set_hydro_vars(model, false, false)?;
        if dt > 0.0 {
            for k in 0..nz {
                model.aux.eps_pre_mix[k] =
                    (model.aux.eps_pre_mix[k] - model.energy[k]) / dt;
            }
        } else {
            model.aux.eps_pre_mix.fill(0.0);
        }
    } else {
        model.aux.eps_pre_mix.fill(0.0);
    }

    if model.controls.do_phase_separation {
        do_phase_separation_step(model, dt)?;
        set_hydro_vars(model, false, false)?;
    } else {
        model.aux.crystal_core_boundary_mass = -1.0;
        model.aux.phase_sep_mixing_mass = -1.0;
        model.aux.eps_phase_separation.fill(0.0);
    }

    // Lock mixing info before element diffusion and `do_struct_burn_mix`
    // (`star/private/evolve.f90:632`).
    model.aux.mixing.okay_to_set_mixing_info = false;

    if model.controls.do_element_diffusion {
        do_element_diffusion_step(model, dt)?;
        set_hydro_vars(model, false, false)?;
        finish_element_diffusion(model, dt);
    } else {
        model.aux.step_hist.num_diffusion_solver_steps = 0;
        model.aux.step_hist.num_diffusion_solver_iters = 0;
        model.aux.eps_diffusion.fill(0.0);
        model.aux.eps_wd_sedimentation.fill(0.0);
    }

    let (extra_src, inject_heat, qmin_raw, qmax_raw) = {
        let mw = &model.controls.mix_wind;
        (
            mw.extra_power_source,
            mw.inject_uniform_extra_heat,
            mw.min_q_for_uniform_extra_heat,
            mw.max_q_for_uniform_extra_heat,
        )
    };
    if extra_src != 0.0 || inject_heat != 0.0 {
        if model.aux.extra_heat.len() != nz {
            model.aux.extra_heat.resize(nz, 0.0);
            model.aux.d_extra_heat_dlnd.resize(nz, 0.0);
            model.aux.d_extra_heat_dlnt.resize(nz, 0.0);
        }
        model.aux.extra_heat.fill(extra_src);
        model.aux.d_extra_heat_dlnd.fill(0.0);
        model.aux.d_extra_heat_dlnt.fill(0.0);
        let qmin = qmin_raw.clamp(0.0, 1.0);
        let qmax = qmax_raw.clamp(0.0, 1.0);
        if inject_heat != 0.0 && qmax > qmin && nz > 0 {
            let mut qp1 = 0.0_f64;
            for k in (0..nz).rev() {
                let q00 = model.q[k];
                let q_top = q00.min(qmax);
                let q_bot = qp1.max(qmin);
                if q_top > q_bot {
                    model.aux.extra_heat[k] += inject_heat * (q_top - q_bot)
                        / model.dq[k].max(1.0e-30);
                }
                qp1 = q00;
            }
        }
    }

    if model.controls.rotation_flag {
        conserve_shell_j_rot_and_distortion(model);
        if model.controls.mix_wind.premix_omega && dt > 0.0 {
            match crate::omega_mix::do_solve_omega_mix_with_extras(
                model,
                0.5 * dt,
                extras,
            )? {
                crate::omega_mix::OmegaMixOutcome::KeepGoing => {}
                crate::omega_mix::OmegaMixOutcome::Retry(msg) => {
                    return Err(StarError::ConvergenceFailure(msg));
                }
                crate::omega_mix::OmegaMixOutcome::Terminate(msg) => {
                    return Err(StarError::Terminated {
                        model_number: model.model_number,
                        reason: msg,
                    });
                }
            }
            crate::hydro_rotation::set_rotation_info(model, false);
            set_hydro_vars(model, false, false)?;
        }
    }
    Ok(())
}

/// Evaluates end-of-step total-energy integrals and updates
/// `error_in_energy_conservation` and `cumulative_energy_error` for both
/// `dedt` and `eps_grav` energy formulations (`okay_energy_conservation` in
/// `star/private/evolve.f90:1101-1373`).
///
/// Computes the global total-energy balance:
/// $$\Delta E_{\mathrm{err}} = E_{\mathrm{end}} - \left(E_{\mathrm{old}} + S_{\mathrm{phase1}} + S_{\mathrm{phase2}}\right)$$
/// across Phase 1 operator-split updates (`adjust_mass`, `conv_premix`,
/// `phase_separation`, `element_diffusion`, `op_split_burn`) and Phase 2
/// solver sources, sinks, radiation, and surface/center mechanical work.
pub fn update_step_energy_conservation(model: &mut StarModel, dt: f64) {
    if model.controls.rsp_flag {
        return;
    }
    let nz = model.nz;
    let (e_int, e_grav, e_kin, e_rot, e_turb, e_end) =
        eval_total_energy_integrals(model);

    model.aux.total_energy = e_end;
    model.aux.step_hist.total_internal_energy_end = e_int;
    model.aux.step_hist.total_gravitational_energy_end = e_grav;
    model.aux.step_hist.total_radial_kinetic_energy_end = e_kin;
    model.aux.step_hist.total_rotational_kinetic_energy_end = e_rot;
    model.aux.step_hist.total_turbulent_energy_end = e_turb;
    model.aux.step_hist.total_energy_end = e_end;

    if dt <= 0.0 || nz == 0 {
        model.aux.step_hist.total_energy_sources_and_sinks = 0.0;
        model.aux.step_hist.error_in_energy_conservation = 0.0;
        model.aux.cumulative_energy_error =
            model.aux.step_hist.cumulative_energy_error_old;
        return;
    }

    let dot_dm = |vals: &[f64]| -> f64 {
        let mut s = 0.0_f64;
        for k in 0..nz {
            s += model.dm[k] * vals.get(k).copied().unwrap_or(0.0);
        }
        s
    };

    // `evolve.f90:1173-1174`
    let mut virial = 0.0_f64;
    for k in 0..nz {
        virial += model.dm[k] * model.peos[k] / model.rho[k];
    }
    let virial_thm_p_avg = 3.0 * virial;

    // `evolve.f90:1176-1181`
    let total_eps_grav = if model.controls.u_flag {
        0.0
    } else {
        let mut s = 0.0_f64;
        for k in 0..nz {
            s += model.dm[k] * model.eps_grav_ad[k].val;
        }
        dt * s
    };

    // `evolve.f90:1220-1235`
    let using_op_split_burn =
        model.controls.op_split_burn || model.aux.step_hist.used_op_split_burn;
    let total_nuclear_heating = if model.controls.eps_nuc_factor == 0.0
        || model.controls.nonlocal_nico_decay_heat
    {
        0.0
    } else if using_op_split_burn {
        let mut sum = 0.0_f64;
        for k in 0..nz {
            let t_start_k = flexmesa_math::exp(
                model.ln_t_start.get(k).copied().unwrap_or(model.ln_t[k]),
            );
            if t_start_k >= model.controls.op_split_burn_min_t {
                let burn_eps = model
                    .aux
                    .burn_avg_epsnuc
                    .get(k)
                    .copied()
                    .unwrap_or(model.eps_nuc[k]);
                sum += dt * model.dm[k] * burn_eps;
            } else {
                sum += dt * model.dm[k] * model.eps_nuc[k];
            }
        }
        sum
    } else {
        dt * dot_dm(&model.eps_nuc)
    };

    // `evolve.f90:1241-1244`
    let mut neu_sum = 0.0_f64;
    for k in 0..nz {
        let neu_start = model
            .non_nuc_neu_start
            .get(k)
            .copied()
            .unwrap_or(model.non_nuc_neu[k]);
        neu_sum += (model.non_nuc_neu[k] + neu_start) * model.dm[k];
    }
    let total_non_nuc_neu_cooling = dt * 0.5 * neu_sum;
    let total_irradiation_heating = dt * dot_dm(&model.aux.irradiation_heat);

    // `evolve.f90:1247-1269`
    let total_wd_sedimentation_heating = if model.controls.do_element_diffusion
        && model.controls.diffusion.do_wd_sedimentation_heating
    {
        dt * dot_dm(&model.aux.eps_wd_sedimentation)
    } else {
        0.0
    };

    let total_energy_from_diffusion = if model.controls.do_element_diffusion
        && model.controls.diffusion.do_diffusion_heating
    {
        dt * dot_dm(&model.aux.eps_diffusion)
    } else {
        0.0
    };

    let total_energy_from_pre_mixing = if model.controls.do_conv_premix {
        dt * dot_dm(&model.aux.eps_pre_mix)
    } else {
        0.0
    };

    let total_energy_from_phase_separation =
        if model.controls.do_phase_separation {
            dt * dot_dm(&model.aux.eps_phase_separation)
        } else {
            0.0
        };

    // `evolve.f90:1271-1288`
    let phase2_total_energy_from_mdot = dt * dot_dm(&model.eps_mdot);
    let total_extra_heating = dt * dot_dm(&model.aux.extra_heat);
    let (work_surf, work_center) =
        crate::hydro_energy::eval_surface_and_center_work(model);
    let phase2_work = dt * (work_surf - work_center);

    let l_theta = crate::hydro_energy::eval_l_theta(model, 0);
    let l_start_0 = model.lum_start.first().copied().unwrap_or(model.lum[0]);
    let l_surf = l_theta * model.lum[0] + (1.0 - l_theta) * l_start_0;
    let total_radiation = dt * (l_surf - model.l_center);

    // `evolve.f90:1310-1336` (exact upstream term order, without algebraic
    // cancellation of Phase 1 / Phase 2 operator-split terms).
    let phase1_total_energy_from_mdot = model
        .aux
        .step_hist
        .energy_change_from_do_adjust_mass_and_calculate_eps_mdot
        + model.aux.step_hist.mdot_adiabatic_surface;

    let phase1_sources_and_sinks = phase1_total_energy_from_mdot
        + total_energy_from_pre_mixing
        + total_energy_from_phase_separation
        + total_wd_sedimentation_heating
        + total_energy_from_diffusion
        + model.aux.non_epsnuc_energy_change_from_split_burn;

    let phase2_sources_and_sinks = -total_energy_from_pre_mixing
        - total_energy_from_phase_separation
        - total_wd_sedimentation_heating
        - total_energy_from_diffusion
        + phase2_total_energy_from_mdot
        + total_nuclear_heating
        - total_non_nuc_neu_cooling
        + total_irradiation_heating
        + total_extra_heating
        - total_radiation
        - phase2_work;

    let total_energy_sources_and_sinks =
        phase1_sources_and_sinks + phase2_sources_and_sinks;

    // `evolve.f90:1356-1365`
    let error_in_energy_conservation = e_end
        - (model.aux.step_hist.total_energy_old
            + total_energy_sources_and_sinks);

    model.aux.cumulative_energy_error =
        next_cumulative_energy_error(model, error_in_energy_conservation);
    let sh = &mut model.aux.step_hist;
    sh.work_outward_at_surface = work_surf;
    sh.work_inward_at_center = work_center;
    sh.virial_thm_p_avg = virial_thm_p_avg;
    sh.total_eps_grav = total_eps_grav;
    sh.total_nuclear_heating = total_nuclear_heating;
    sh.total_non_nuc_neu_cooling = total_non_nuc_neu_cooling;
    sh.total_irradiation_heating = total_irradiation_heating;
    sh.total_wd_sedimentation_heating = total_wd_sedimentation_heating;
    sh.total_extra_heating = total_extra_heating;
    sh.total_energy_sources_and_sinks = total_energy_sources_and_sinks;
    sh.error_in_energy_conservation = error_in_energy_conservation;
}

/// Evaluates enclosed nuclear luminosity by reaction category (`erg/s`) from
/// center outward (`set_luminosity_by_category` in `star/private/star_utils.f90:4112-4132`).
#[must_use]
pub fn eval_lum_by_category_erg_s(
    model: &StarModel,
) -> Vec<[f64; flexmesa_chem::NUM_CATEGORIES]> {
    let nz = model.nz;
    let mut out = vec![[0.0_f64; flexmesa_chem::NUM_CATEGORIES]; nz];
    if nz == 0 {
        return out;
    }
    let mut running = [0.0_f64; flexmesa_chem::NUM_CATEGORIES];
    for k in (0..nz).rev() {
        let dm_k = model.dm[k];
        if let Some(cats) = model.aux.eps_nuc_categories.get(k) {
            for (c, &val) in cats.iter().enumerate() {
                running[c] += dm_k * val;
            }
        }
        out[k] = running;
    }
    out
}

/// Snapshots start-of-step `ye_start`, `eps_nuc_start`, and
/// `lum_by_category_start` (`set_start_of_step_info` in
/// `star/private/evolve.f90:1748-1808` and `struct_burn_mix.f90:268-356`).
pub fn snapshot_step_start_extras(model: &mut StarModel) {
    model.ye_start.clone_from(&model.ye);
    model.eps_nuc_start.clone_from(&model.eps_nuc);
    model.lum_by_category_start = eval_lum_by_category_erg_s(model);
}

/// Snapshots inner-boundary and elapsed-time `*_old` scalars (`new_generation`
/// in `star/private/evolve_support.f90:82-90`).
pub fn snapshot_step_old_extras(model: &mut StarModel) {
    if model.time_sec == 0.0 && model.star_age != 0.0 {
        model.time_sec = model.star_age * flexmesa_const::SECYER;
    }
    model.time_old_sec = model.time_sec;
    model.m_center_old = model.m_center;
    model.r_center_old = model.r_center;
    model.l_center_old = model.l_center;
    model.v_center_old = model.v_center;
}

/// Executes pre-step adaptive remeshing (`do_mesh` in
/// `star/private/evolve.f90:1882-1889, 1961-1989`), returning
/// `(num_splits, num_merges, pre_remesh_snapshot)`.
///
/// # Errors
/// Returns [`StarError`] if microphysics or mesh adjustment fails.
pub fn remesh_star_before_step(
    model: &mut StarModel,
    dt_prev: f64,
) -> Result<(usize, usize, Option<PrimaryStateBuffer>), StarError> {
    let mw = &model.controls.mix_wind;
    let hold_after_retry = mw.restore_mesh_on_retry
        && mw.num_steps_to_hold_mesh_after_retry > 0
        && model.aux.step_hist.model_number_for_last_retry.is_some_and(
            |m_ret| {
                (m_ret as isize) + mw.num_steps_to_hold_mesh_after_retry
                    > (model.model_number as isize)
            },
        );
    let min_remesh_dt = mw.remesh_dt_limit;
    let dt_remesh_ok = dt_prev <= 0.0 || dt_prev >= min_remesh_dt;
    let allow_first =
        model.controls.rsp_flag || !model.doing_first_model_of_run;
    if model.controls.okay_to_remesh
        && (!model.controls.rsp_flag || mw.use_split_merge_amr)
        && allow_first
        && !hold_after_retry
        && dt_remesh_ok
    {
        model.aux.mixing.okay_to_set_mixing_info = true;
        set_hydro_vars(model, false, false)?;
        let snap = PrimaryStateBuffer::from_model(model);
        let mesh_cfg = MeshFunctionConfig::from_controls(&model.controls);
        let (ns, nm) = do_adjust_mesh(model, &mesh_cfg)?;
        model.aux.mixing.okay_to_set_mixing_info = true;
        set_hydro_vars(model, false, false)?;
        model.save_start_values()?;
        snapshot_step_start_extras(model);
        let pre_snap = (ns + nm > 0).then_some(snap);
        Ok((ns, nm, pre_snap))
    } else {
        Ok((0, 0, None))
    }
}

/// Bottom fractional mass `q` of the largest convective mixing region
/// (`conv_mx1_bot` in `star/private/report.f90:620-638`).
pub(crate) fn eval_conv_mx1_bot(model: &StarModel) -> f64 {
    let nz = model.nz;
    let (mut best_dq, mut best_bot, mut k) = (0.0_f64, 0.0_f64, 0usize);
    while k < nz {
        if model.mixing_type[k] == MixingType::ConvectiveMixing {
            let (mut sum_dq, mut kbot) = (0.0, k);
            while kbot < nz
                && model.mixing_type[kbot] == MixingType::ConvectiveMixing
            {
                sum_dq += model.dq[kbot];
                kbot += 1;
            }
            if sum_dq > best_dq {
                best_dq = sum_dq;
                best_bot = (model.q[kbot - 1] - model.dq[kbot - 1]).max(0.0);
            }
            k = kbot;
        } else {
            k += 1;
        }
    }
    best_bot
}

/// Upstream `do_relax_to_radiative_core` (`star/private/init.f90:1103` &
/// `star/private/relax.f90:3295-3408`): contracts a pre-MS model with frozen
/// nuclear composition until a radiative core forms (`conv_mx1_bot > 1e-3` after
/// being fully convective) or `L_nuc / L_phot > 0.1`, then resets age and step.
pub(crate) fn relax_pre_ms_to_radiative_core(
    model: &mut StarModel,
) -> Result<(), StarError> {
    let saved_ctrl = model.controls.clone();
    model.aux.doing_relax = true;
    model.model_number = 0;
    let c = &mut model.controls;
    if c.mlt_option == MltOption::Tdc {
        c.mlt_option = MltOption::Cox;
    }
    (c.atm_option, c.atm_t_tau_relation, c.pextra_factor) = (
        crate::job_controls::AtmOption::TTau,
        crate::job_controls::AtmTTauRelation::Eddington,
        2.0,
    );
    (
        c.dxdt_nuc_factor,
        c.ts_stop.max_timestep_factor,
        c.mass_change,
        c.use_other_wind,
    ) = (0.0, 2.0, 0.0, false);
    c.max_years_for_timestep = 0.0;
    c.max_age = 1.0e36;
    c.max_model_number = None;
    (
        c.reimers_scaling_factor,
        c.blocker_scaling_factor,
        c.vink_scaling_factor,
        c.dutch_scaling_factor,
    ) = (0.0, 0.0, 0.0, 0.0);
    let mut min_conv_mx1_bot = 1.0e99_f64;
    for step in 0..600usize {
        // Upstream pre-MS relaxation (`star/private/init.f90:1095-1108` &
        // `star/private/relax.f90:4068`) runs with remeshing enabled so the
        // initial `dq[0]` grid refines to `max_surface_cell_dq` under
        // `doing_relax = true` before evolution starts.
        crate::evolve::star_evolve_step(
            model,
            model.controls.okay_to_remesh,
            model.controls.op_split_burn,
            true,
        )?;
        let conv_bot = eval_conv_mx1_bot(model);
        min_conv_mx1_bot = min_conv_mx1_bot.min(conv_bot);
        let l_nuc: f64 =
            (0..model.nz).map(|k| model.eps_nuc[k] * model.dm[k]).sum();
        let l_phot = model.lum.first().copied().unwrap_or(0.0);
        if (l_phot > 0.0 && l_nuc / l_phot > 0.1)
            || (step >= 1 && min_conv_mx1_bot < 1.0e-6 && conv_bot > 1.0e-3)
        {
            break;
        }
    }
    model.controls = saved_ctrl;
    model.aux.doing_relax = false;
    model.doing_first_model_of_run = true;
    (
        model.star_age,
        model.star_age_old,
        model.time_sec,
        model.time_old_sec,
        model.aux.step_hist.time_sec,
        model.aux.step_hist.time_old_sec,
        model.model_number,
        model.num_retries,
        model.num_redos,
        model.total_num_solver_iterations,
        model.total_num_solver_calls,
        model.micro_cache_hits,
        model.micro_cache_misses,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0, 0, 0, 0, 0, 0);
    model.aux.cumulative_energy_error = 0.0;
    set_hydro_vars(model, false, false)?;
    model.save_start_values()?;
    snapshot_step_start_extras(model);
    model.save_old_values();
    snapshot_step_old_extras(model);
    Ok(())
}

/// Relaxes a rotating stellar model smoothly to a uniform angular velocity
/// `target_omega` ($\mathrm{rad\,s^{-1}}$) over `num_steps` internal
/// relaxation steps (`do_relax_uniform_omega` and `do_internal_evolve` in
/// `star/private/relax.f90:1234-1391, 3878-4286`).
///
/// # Errors
/// Returns [`StarError`] if `num_steps == 0` or an internal relaxation step
/// fails to converge.
pub fn relax_uniform_omega(
    model: &mut StarModel,
    target_omega: f64,
    num_steps: usize,
    max_yrs_dt: f64,
) -> Result<(), StarError> {
    if !model.controls.rotation_flag || model.nz == 0 {
        return Ok(());
    }
    if model.omega.iter().all(|&w| w == target_omega) {
        return Ok(());
    }
    if num_steps == 0 {
        return Err(StarError::InvalidInput(String::from(
            "invalid num_steps_to_relax_rotation (must be > 0)",
        )));
    }

    let saved_ctrl = model.controls.clone();
    let saved_age = model.star_age;
    let saved_age_old = model.star_age_old;
    let saved_time_sec = model.time_sec;
    let saved_time_old_sec = model.time_old_sec;
    let saved_hist_time_sec = model.aux.step_hist.time_sec;
    let saved_hist_time_old_sec = model.aux.step_hist.time_old_sec;
    let saved_model_number = model.model_number;
    let saved_num_retries = model.num_retries;
    let saved_num_redos = model.num_redos;
    let saved_iters = model.total_num_solver_iterations;
    let saved_calls = model.total_num_solver_calls;
    let saved_cache_hits = model.micro_cache_hits;
    let saved_cache_misses = model.micro_cache_misses;
    let saved_dt_next = model.dt_next;
    let saved_first = model.doing_first_model_of_run;
    let saved_cum_err = model.aux.cumulative_energy_error;

    let starting_omega = model.omega.first().copied().unwrap_or(0.0);
    model.aux.doing_relax = true;
    {
        let c = &mut model.controls;
        c.dxdt_nuc_factor = 0.0;
        c.mix_factor = 0.0;
        c.mix_wind.am_d_mix_factor = 0.0;
        c.max_age = 1.0e50;
        c.ts_stop.max_age_in_days = 1.0e50;
        c.max_age_in_seconds = None;
        c.ts_stop.max_timestep_factor = 2.0;
        c.max_model_number = None;
        c.max_years_for_timestep = max_yrs_dt;
        c.ts_stop.max_timestep = 0.0;
        c.mass_change = 0.0;
        c.use_other_wind = false;
        c.reimers_scaling_factor = 0.0;
        c.blocker_scaling_factor = 0.0;
        c.vink_scaling_factor = 0.0;
        c.dutch_scaling_factor = 0.0;
        c.mix_wind.de_jager_scaling_factor = 0.0;
        c.mix_wind.van_loon_scaling_factor = 0.0;
        c.mix_wind.nieuwenhuijzen_scaling_factor = 0.0;
        c.mix_wind.bjorklund_scaling_factor = 0.0;
        c.use_gold2_tolerances = false;
        c.steps_before_use_gold2_tolerances = -1;
        c.use_gold_tolerances = c.relax_use_gold_tolerances;
        c.steps_before_use_gold_tolerances = -1;
        if c.mlt_option == MltOption::Tdc {
            c.mlt_option = MltOption::Cox;
        }
    }
    if model.dt_next <= 0.0 {
        model.dt_next = flexmesa_const::SECYER;
    }

    let max_iters = num_steps + 51;
    let mut step_err = Ok(());
    for step_idx in 1..=max_iters {
        if let Err(e) = crate::evolve::star_evolve_step(
            model,
            model.controls.okay_to_remesh,
            model.controls.op_split_burn,
            true,
        ) {
            step_err = Err(e);
            break;
        }
        let frac = ((step_idx as f64) / (num_steps as f64)).clamp(0.0, 1.0);
        let cur_omega = model.omega.first().copied().unwrap_or(0.0);
        let done = step_idx > num_steps + 50
            || (cur_omega - target_omega).abs()
                < 1.0e-4 * target_omega.abs().max(1.0e-10);
        let this_step_omega = if done {
            target_omega
        } else {
            frac * target_omega + (1.0 - frac) * starting_omega
        };
        crate::relax::apply_uniform_rotation(model, this_step_omega);
        crate::rotation::update_rotation_distortion_factors(model);
        if done {
            break;
        }
    }

    model.controls = saved_ctrl;
    model.aux.doing_relax = false;
    model.doing_first_model_of_run = saved_first;
    model.star_age = saved_age;
    model.star_age_old = saved_age_old;
    model.time_sec = saved_time_sec;
    model.time_old_sec = saved_time_old_sec;
    model.aux.step_hist.time_sec = saved_hist_time_sec;
    model.aux.step_hist.time_old_sec = saved_hist_time_old_sec;
    model.model_number = saved_model_number;
    model.num_retries = saved_num_retries;
    model.num_redos = saved_num_redos;
    model.total_num_solver_iterations = saved_iters;
    model.total_num_solver_calls = saved_calls;
    model.micro_cache_hits = saved_cache_hits;
    model.micro_cache_misses = saved_cache_misses;
    model.dt = 0.0;
    model.dt_old = 0.0;
    model.dt_next = saved_dt_next;
    model.aux.cumulative_energy_error = saved_cum_err;
    model.aux.step_hist.timestep_hold = 0;
    model.aux.step_hist.model_number_for_last_retry = None;

    step_err?;
    set_hydro_vars(model, false, false)?;
    model.save_start_values()?;
    snapshot_step_start_extras(model);
    model.save_old_values();
    snapshot_step_old_extras(model);
    // Restore pre-relaxation `*_old` timestamps after `save_old_values`
    // (`restore_stuff` in `star/private/relax.f90:4280-4281`).
    model.star_age_old = saved_age_old;
    model.time_old_sec = saved_time_old_sec;
    model.aux.step_hist.time_old_sec = saved_hist_time_old_sec;
    Ok(())
}
