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
use crate::diffusion::{do_element_diffusion, DiffusionScheme};
use crate::hydro_energy::{
    setup_d_turbulent_energy_dt, setup_dl_dm, setup_sources_and_others,
};
use crate::hydro_vars::set_hydro_vars;
use crate::mesh_adjust::{do_adjust_mesh, MeshFunctionConfig};
use crate::hydro_work::{
    eval_cell_dpe_dt_ad, eval_v_flag_dke_dt_ad, eval_v_flag_dwork_dm_ad,
    eval_v_flag_simple_pdv_dm_ad,
};
use crate::model::StarModel;
use crate::premix::{do_conv_premix, do_phase_separation};
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
        let _ = do_conv_premix(
            model,
            5,
            model.controls.mix_wind.conv_premix_avoid_increase,
        );
        need_hydro_refresh = true;
    } else {
        model.aux.eps_pre_mix.fill(0.0);
    }

    if model.controls.do_phase_separation {
        let is_one = model
            .controls
            .phase_separation_option
            .trim()
            .eq_ignore_ascii_case("ONe");
        model.aux.eps_phase_separation =
            do_phase_separation(model, dt.max(1.0), 175.0, is_one);
        need_hydro_refresh = true;
    } else {
        model.aux.eps_phase_separation.fill(0.0);
    }

    if model.controls.do_element_diffusion {
        let (_, eps_wd) = do_element_diffusion(
            model,
            dt.clamp(1.0, 1.0e13),
            DiffusionScheme::Paquette,
        );
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

/// Executes pre-step adaptive remeshing (`do_adjust_mesh` + hydro/rotation
/// smoothing) once before a step or coupled binary RLOF trial loop, returning
/// `(num_splits, num_merges, pre_remesh_snapshot)`.
///
/// # Errors
/// Returns [`StarError`] if microphysics or mesh adjustment fails.
pub fn remesh_star_before_step(
    model: &mut StarModel,
    dt_try: f64,
) -> Result<(usize, usize, Option<StarModel>), StarError> {
    let recent_retry = model
        .aux
        .step_hist
        .model_number_for_last_retry
        .is_some_and(|m_ret| model.model_number <= m_ret + 1);
    let sharp_dt_drop = !model.controls.rsp_flag
        && dt_try > 0.0
        && model.dt_old > 0.0
        && dt_try < 0.25 * model.dt_old;
    let min_remesh_dt = model.controls.mix_wind.remesh_dt_limit;
    let dt_remesh_ok =
        model.controls.rsp_flag || dt_try <= 0.0 || dt_try >= min_remesh_dt;
    if model.controls.okay_to_remesh
        && !model.controls.rsp2_flag
        && !recent_retry
        && !sharp_dt_drop
        && dt_remesh_ok
    {
        set_hydro_vars(model, false, false)?;
        let dt_curr = dt_try.max(1.0);
        refresh_mixing_state(model, dt_curr, &[]);
        let mesh_cfg = MeshFunctionConfig::from_controls(&model.controls);
        let snap = model.clone();
        let (ns, nm) = do_adjust_mesh(model, &mesh_cfg)?;
        if ns + nm > 0 {
            crate::hydro_state::smooth_post_remesh_hydro_and_rotation(model, &snap);
            set_hydro_vars(model, false, false)?;
            model.save_start_values();
            return Ok((ns, nm, Some(snap)));
        }
        Ok((ns, nm, None))
    } else {
        Ok((0, 0, None))
    }
}

/// Updates sticky-`gradr` solver MLT flip-tracking and zero-initialized `sig`
/// immediately after face MLT evaluation (`mlt_info.f90` solver iteration hysteresis).
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
                    / std::f64::consts::LN_10
                    >= mw.min_logt_for_make_gradr_sticky_in_solver_iters);
        let eval_cnt = model.aux.solver_mlt_eval_count;
        for k in 0..nz {
            if model.aux.fixed_gradr_sticky[k] {
                continue;
            }
            let is_conv =
                u8::from(model.mixing_type[k] == MixingType::ConvectiveMixing);
            if is_conv != model.aux.mlt_prev_conv[k] {
                model.aux.mlt_prev_conv[k] = is_conv;
                model.aux.mlt_flip_count[k] =
                    model.aux.mlt_flip_count[k].saturating_add(1);
                if model.aux.mlt_flip_count[k] >= 2 {
                    model.aux.solver_had_mlt_flip = true;
                }
            }
        }
        if sticky_ctrl && eval_cnt > 3 {
            for k in 0..nz {
                if model.mixing_type[k] != MixingType::ConvectiveMixing {
                    model.aux.fixed_gradr_sticky[k] = true;
                }
            }
        } else if model.aux.solver_had_mlt_flip && eval_cnt >= 3 {
            for k in 0..nz {
                if model.t[k] > 1.0e6
                    && model.mixing_type[k] != MixingType::ConvectiveMixing
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
    (c.max_years_for_timestep, c.max_age, c.max_model_number) =
        (None, None, None);
    (
        c.reimers_scaling_factor,
        c.blocker_scaling_factor,
        c.vink_scaling_factor,
        c.dutch_scaling_factor,
    ) = (0.0, 0.0, 0.0, 0.0);
    let mut min_conv_mx1_bot = 1.0e99_f64;
    for step in 0..600usize {
        if crate::evolve::star_evolve_step(
            model,
            false,
            model.controls.op_split_burn,
            true,
        )
        .is_err()
        {
            break;
        }
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
    (
        model.star_age,
        model.star_age_old,
        model.model_number,
        model.num_retries,
    ) = (0.0, 0.0, 1, 0);
    model.aux.cumulative_energy_error = 0.0;
    set_hydro_vars(model, false, false)?;
    model.save_start_values();
    model.save_old_values();
    Ok(())
}
