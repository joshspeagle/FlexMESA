// SPDX-License-Identifier: LGPL-3.0-or-later
//! Pre-step operator-split physics hooks (`apply_step_pre_physics`),
//! domain total-energy integration (`eval_total_energy_integrals`), and
//! discrete energy-conservation residual accounting
//! (`update_step_energy_conservation`) corresponding to
//! `star/private/evolve.f90:605-665,1101-1373,1646-1681` and
//! `star/private/star_utils.f90:2452-2612`.

use flexmesa_turb::MltOption;

use crate::controls::{EnergyEqnOption, StarError};
use crate::diffusion::{do_element_diffusion, DiffusionScheme};
use crate::hydro_energy::{
    eval_cell_pressure_for_work_ad, setup_d_turbulent_energy_dt, setup_dl_dm,
    setup_sources_and_others,
};
use crate::hydro_vars::set_hydro_vars;
use crate::hydro_work::{
    eval_cell_dpe_dt_ad, eval_v_flag_dke_dt_ad, eval_v_flag_dwork_dm_ad,
    eval_v_flag_simple_pdv_dm_ad,
};
use crate::model::StarModel;
use crate::omega_mix::solve_omega_mix;
use crate::premix::{do_conv_premix, do_phase_separation};
use crate::riemann_ad::eval_u_flag_total_energy_rates_ad;
use crate::rotation::{
    set_rotation_mixing_info, update_rotation_distortion_factors,
};

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
        update_rotation_distortion_factors(model);
        let zones = set_rotation_mixing_info(model);
        let nu_omega: Vec<f64> = zones.iter().map(|z| z.nu_st).collect();
        solve_omega_mix(model, dt, &nu_omega, 2);
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
            let (p_ad, _) = eval_cell_pressure_for_work_ad(model, k);
            let lnd_s = model.lnd_start.get(k).copied().unwrap_or(model.lnd[k]);
            let pdv_dt =
                p_ad.val * ((-model.lnd[k]).exp() - (-lnd_s).exp()) / dt;
            de_dt + pdv_dt
        };

        err_erg +=
            dt * model.dm[k] * (d_mech_int_dt + d_eturb_dt + dl_dm - sources);
    }

    model.aux.step_hist.error_in_energy_conservation = err_erg;
    model.aux.cumulative_energy_error =
        model.aux.step_hist.cumulative_energy_error_old + err_erg;
}
