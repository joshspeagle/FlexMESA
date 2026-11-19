// SPDX-License-Identifier: LGPL-3.0-or-later
//! Pre-step operator-split physics hooks (`apply_step_pre_physics`),
//! domain total-energy integration (`eval_total_energy_integrals`), and
//! discrete energy-conservation residual accounting
//! (`update_step_energy_conservation`) corresponding to
//! `star/private/evolve.f90:605-665,1101-1373,1646-1681` and
//! `star/private/star_utils.f90:2452-2612`.

use flexmesa_const::MixingType;
use flexmesa_turb::MltOption;

use crate::controls::{EnergyEqnOption, StarError};
use crate::diffusion::{DiffusionScheme, do_element_diffusion};
use crate::hydro_energy::{
    setup_d_turbulent_energy_dt, setup_dl_dm, setup_sources_and_others,
};
use crate::hydro_vars::set_hydro_vars;
use crate::hydro_work::{
    eval_cell_dpe_dt_ad, eval_v_flag_dke_dt_ad, eval_v_flag_dwork_dm_ad,
    eval_v_flag_simple_pdv_dm_ad,
};
use crate::mesh_adjust::{MeshFunctionConfig, do_adjust_mesh};
use crate::model::StarModel;
use crate::premix::{do_conv_premix, do_phase_separation};
use crate::primary_state::PrimaryStateBuffer;
use crate::riemann_ad::eval_u_flag_total_energy_rates_ad;
use crate::rotation::{
    conserve_shell_j_rot_and_distortion, set_rotation_mixing_info,
};
use crate::struct_burn_mix::refresh_mixing_state;

/// Evaluates domain-integrated `(e_int, e_grav, e_kin, e_rot, e_turb, e_tot)`
/// in $\mathrm{erg}$ (`eval_total_energy_integrals` in
/// `star/private/star_utils.f90:2452-2612`).
#[must_use]
pub fn eval_total_energy_integrals(
    model: &StarModel,
) -> (f64, f64, f64, f64, f64, f64) {
    let nz = model.nz;
    let (mut e_int, mut e_grav, mut e_kin, mut e_rot, mut e_turb) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let use_mc = model.controls.use_mass_corrections;
    let tdc_turb = model.controls.mlt_option == MltOption::Tdc
        && model.controls.tdc_include_eturb_in_energy_equation;

    for k in 0..nz {
        let dm = model.dm[k];
        let mc = if use_mc {
            model.aux.mass_correction.get(k).copied().unwrap_or(1.0)
        } else {
            1.0
        };
        e_int += dm * model.energy[k];
        let grav_00 = -model.cgrav[k] * model.m_grav[k] / model.r[k].max(1.0);
        let grav_p1 = if k + 1 < nz {
            -model.cgrav[k + 1] * model.m_grav[k + 1] / model.r[k + 1].max(1.0)
        } else if model.r_center > 0.0 {
            -model.cgrav[nz - 1] * model.m_center / model.r_center
        } else {
            0.0
        };
        e_grav += dm * 0.5 * mc * (grav_00 + grav_p1);
        if model.controls.u_flag {
            e_kin += dm * 0.5 * mc * model.u[k] * model.u[k];
        } else if model.controls.v_flag {
            let v_p1 = if k + 1 < nz {
                model.v[k + 1]
            } else {
                model.v_center
            };
            e_kin += dm * 0.25 * mc * (model.v[k] * model.v[k] + v_p1 * v_p1);
        }
        if model.controls.rotation_flag {
            let i_rot = model.aux.i_rot.get(k).copied().unwrap_or(0.0);
            let omega = model.omega.get(k).copied().unwrap_or(0.0);
            e_rot += dm * 0.5 * i_rot * omega * omega;
        }
        if model.controls.rsp2_flag {
            let et_k = if model.layout.i_w.is_some() {
                model.aux.rsp.et_rsp.get(k).copied().unwrap_or(0.0)
            } else {
                model.aux.etrb.get(k).copied().unwrap_or(0.0)
            };
            e_turb += dm * et_k.max(0.0);
        } else if tdc_turb {
            let vc_00 = model.mlt_vc.get(k).copied().unwrap_or(0.0);
            let vc_p1 = if k + 1 < nz {
                model.mlt_vc.get(k + 1).copied().unwrap_or(0.0)
            } else {
                0.0
            };
            e_turb += dm * 0.75 * (vc_00 * vc_00 + vc_p1 * vc_p1);
        } else if model.controls.rsp_flag {
            e_turb += dm * model.aux.rsp.et_rsp.get(k).copied().unwrap_or(0.0);
        }
    }
    (
        e_int,
        e_grav,
        e_kin,
        e_rot,
        e_turb,
        e_int + e_grav + e_kin + e_turb,
    )
}

/// Applies pre-solver operator-split physics (`do_conv_premix`,
/// `do_phase_separation`, `do_element_diffusion`, uniform extra heating, and
/// `solve_omega_mix`) corresponding to `do_step_part2` and `check_for_extra_heat`
/// in `star/private/evolve.f90:605-665,1646-1681`.
///
/// # Errors
/// Returns [`StarError`] if refreshing hydrodynamic/microphysics state fails.
pub fn apply_step_pre_physics(
    model: &mut StarModel,
    dt: f64,
) -> Result<(), StarError> {
    let nz = model.nz;
    let mut need_hydro_refresh = false;

    if model.controls.do_conv_premix {
        do_conv_premix(
            model,
            5,
            model.controls.mix_wind.conv_premix_avoid_increase,
        )?;
        need_hydro_refresh = true;
    } else {
        model.aux.eps_pre_mix.fill(0.0);
    }

    if model.controls.do_phase_separation {
        let is_one = match model
            .controls
            .phase_separation_option
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "co" => false,
            "one" => true,
            other => {
                return Err(StarError::InvalidInput(format!(
                    "unrecognized '&controls::phase_separation_option' = '{other}' (expected 'CO' or 'ONe')"
                )));
            }
        };
        model.aux.eps_phase_separation =
            do_phase_separation(model, dt, 175.0, is_one)?;
        need_hydro_refresh = true;
    } else {
        model.aux.eps_phase_separation.fill(0.0);
    }

    if model.controls.do_element_diffusion {
        let (_, eps_wd) =
            do_element_diffusion(model, dt, DiffusionScheme::Paquette)?;
        model.aux.eps_wd_sedimentation = eps_wd;
        need_hydro_refresh = true;
    } else {
        model.aux.step_hist.num_diffusion_solver_steps = 0;
        model.aux.step_hist.num_diffusion_solver_iters = 0;
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
            crate::rotation::sync_post_henyey_rotation(model, dt);
        }
        let _ = set_rotation_mixing_info(model);
        need_hydro_refresh = true;
    }

    if need_hydro_refresh {
        set_hydro_vars(model, false, false)?;
    }
    Ok(())
}

/// Evaluates end-of-step total energy integrals and updates
/// `error_in_energy_conservation` and `cumulative_energy_error`
/// (`okay_energy_conservation` in `star/private/evolve.f90:1101-1373`).
pub fn update_step_energy_conservation(model: &mut StarModel, dt: f64) {
    if model.controls.rsp_flag || model.controls.rsp2_flag {
        return;
    }
    let (_, _, _, _, _, e_tot) = eval_total_energy_integrals(model);
    model.aux.total_energy = e_tot;
    model.aux.step_hist.total_energy_end = e_tot;

    if model.controls.energy_eqn_option == EnergyEqnOption::EpsGrav || dt <= 0.0
    {
        model.aux.step_hist.error_in_energy_conservation = 0.0;
        model.aux.cumulative_energy_error =
            model.aux.step_hist.cumulative_energy_error_old;
        return;
    }

    let nz = model.nz;
    let v_flag_only = model.controls.v_flag && !model.controls.u_flag;
    let u_flag_only = model.controls.u_flag && !model.controls.v_flag;
    let vel_tc = model.controls.is_velocity_time_centered(model.model_number)
        && model.dt > 0.0
        && !model.ln_r_start.is_empty();
    let simple_pdv = vel_tc
        && model
            .controls
            .use_p_d_1_div_rho_form_of_work_when_time_centering_velocity;

    let mut err_erg = 0.0_f64;
    for k in 0..nz {
        let dl_dm = setup_dl_dm(model, k).val;
        let sources = setup_sources_and_others(model, k).0.val;
        let d_eturb_dt = setup_d_turbulent_energy_dt(model, k).val;
        let e_start_k = model
            .energy_start
            .get(k)
            .copied()
            .unwrap_or(model.energy[k]);

        let d_mech_int_dt = if u_flag_only {
            eval_u_flag_total_energy_rates_ad(model, k, dt).val
        } else if v_flag_only {
            let de_dt = (model.energy[k] - e_start_k) / dt;
            if simple_pdv {
                eval_v_flag_simple_pdv_dm_ad(model, k, false, vel_tc).0.val
                    + de_dt
            } else {
                eval_v_flag_dwork_dm_ad(model, k, false, vel_tc).0.val
                    + eval_v_flag_dke_dt_ad(model, k, dt).val
                    + eval_cell_dpe_dt_ad(model, k, dt).val
                    + de_dt
            }
        } else {
            let de_dt = (model.energy[k] - e_start_k) / dt;
            if simple_pdv {
                eval_v_flag_simple_pdv_dm_ad(model, k, false, vel_tc).0.val
                    + de_dt
            } else {
                eval_v_flag_dwork_dm_ad(model, k, false, vel_tc).0.val
                    + eval_cell_dpe_dt_ad(model, k, dt).val
                    + de_dt
            }
        };

        err_erg +=
            dt * model.dm[k] * (d_mech_int_dt + d_eturb_dt + dl_dm - sources);
    }

    model.aux.step_hist.error_in_energy_conservation = err_erg;
    model.aux.cumulative_energy_error =
        model.aux.step_hist.cumulative_energy_error_old + err_erg;
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
        set_hydro_vars(model, false, false)?;
        let dt_curr = dt_prev.max(1.0);
        refresh_mixing_state(model, dt_curr, &[]);
        let snap = PrimaryStateBuffer::from_model(model);
        let mesh_cfg = MeshFunctionConfig::from_controls(&model.controls);
        let (ns, nm) = do_adjust_mesh(model, &mesh_cfg)?;
        set_hydro_vars(model, false, false)?;
        model.save_start_values()?;
        snapshot_step_start_extras(model);
        let pre_snap = (ns + nm > 0).then_some(snap);
        Ok((ns, nm, pre_snap))
    } else {
        Ok((0, 0, None))
    }
}

/// Updates sticky-`gradr` solver MLT state and zero-initialized `sig`
/// immediately after face MLT evaluation (`turb_info.f90:56-61, 213-223`).
pub(crate) fn update_mlt_sticky_and_sig_after_hydro(model: &mut StarModel) {
    let nz = model.nz;
    if !model.aux.doing_struct_burn_mix {
        for k in 0..nz {
            model.aux.mlt_prev_conv[k] =
                u8::from(model.mixing_type[k] == MixingType::ConvectiveMixing);
        }
        if model.sig.iter().all(|&s| s == 0.0) {
            crate::hydro_vars::update_sig_from_d_mix(model);
        }
    } else if !model.controls.rsp2_flag {
        model.aux.solver_mlt_eval_count += 1;
        let mw = &model.controls.mix_wind;
        let sticky_ctrl = mw.make_gradr_sticky_in_solver_iters
            || (mw.min_logt_for_make_gradr_sticky_in_solver_iters < 1.0e20
                && model
                    .ln_t_start
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max)
                    / flexmesa_const::LN10
                    >= mw.min_logt_for_make_gradr_sticky_in_solver_iters);
        // `eval_cnt == 1` is the pre-iteration residual evaluation (`solver_iter == 0`),
        // so `solver_iter > 3` (`turb_info.f90:56`) corresponds to `eval_cnt > 4`.
        if sticky_ctrl && model.aux.solver_mlt_eval_count > 4 {
            for k in 0..nz {
                if !model.aux.fixed_gradr_sticky[k]
                    && model.mixing_type[k] == MixingType::NoMixing
                {
                    model.aux.fixed_gradr_sticky[k] = true;
                }
            }
        }
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
        model.model_number,
        model.num_retries,
        model.num_redos,
        model.total_num_solver_iterations,
        model.total_num_solver_calls,
        model.micro_cache_hits,
        model.micro_cache_misses,
    ) = (0.0, 0.0, 1, 0, 0, 0, 0, 0, 0);
    model.aux.cumulative_energy_error = 0.0;
    set_hydro_vars(model, false, false)?;
    model.save_start_values()?;
    model.save_old_values();
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
        c.max_age_in_days = 1.0e50;
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
        let this_step_omega =
            frac * target_omega + (1.0 - frac) * starting_omega;
        let cur_omega = model.omega.first().copied().unwrap_or(0.0);
        let done = step_idx > num_steps + 50
            || (cur_omega - target_omega).abs()
                < 1.0e-4 * target_omega.abs().max(1.0e-10);
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
    model.save_old_values();
    Ok(())
}
