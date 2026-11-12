// SPDX-License-Identifier: LGPL-3.0-or-later
//! Face-centered hydrostatic/hydrodynamic momentum equation (`do1_momentum_eqn`)
//! and kinematic radius-velocity equation (`do1_radius_eqn`) with 3-cell
//! stencil [`StarAdOrder1`] automatic differentiation
//! (`star/private/hydro_momentum.f90`).
//!
//! Implements full line-by-line fidelity with upstream `hydro_momentum.f90`:
//! - **Rotational gravity correction (`expected_HSE_grav_term`,
//!   `hydro_momentum.f90:354-358`)**: Multiplies $g_{\mathrm{HSE},k} = -G_k m_{\mathrm{grav},k}/r_k^2$
//!   by $f_{P,\mathrm{rot},k}$ when `rotation_flag && use_gravity_rotation_correction`.
//! - **Baryonic mass correction (`hydro_momentum.f90:187-199`)**: Scales face
//!   control-volume mass $\Delta m_{\mathrm{face},k}$ by `mass_correction` when
//!   `use_mass_corrections` is active (`0.5 * dm[0] * mc[0]` at $k = 0$).
//! - **Non-HSE hydrodynamic terms (`expected_non_HSE_term`,
//!   `hydro_momentum.f90:374-461`)**:
//!   - `extra_grav[k]` extra face gravitational acceleration,
//!   - Lagrangian acceleration $-(v_k - v_{\mathrm{start},k})/\Delta t$,
//!   - Velocity drag $-c_{\mathrm{drag}} v_k / \Delta t$ for $q_k > q_{\mathrm{drag,min}}$,
//!   - Turbulent viscosity $U_{q,k}$ (`uq_ad[k]`),
//!   - Artificial viscosity gradient $-(A_k / \Delta m_{\mathrm{face},k})(P_{\mathrm{vsc},k-1} - P_{\mathrm{vsc},k})$,
//!   - MLT turbulent pressure difference $-(A_k / \Delta m_{\mathrm{face},k})(P_{\mathrm{turb},k-1} - P_{\mathrm{turb},k})$,
//!   - Rayleigh-Taylor instability (`RTI`) momentum diffusion and source kick
//!     (`setup_RTI_terms`, `hydro_momentum.f90:250-284`).
//! - **Time-centered pressure $\theta_P$ and composition partials
//!   `dln_peos_dxa_for_partials` (`hydro_momentum.f90:312-343`)**.
//! - **Radius equation (`do1_radius_eqn`, `hydro_momentum.f90:552-608`)** with
//!   `force_zero_v` bounds (`velocity_q_upper_bound`, `velocity_tau_lower_bound`,
//!   `velocity_logt_lower_bound`) and optional velocity time-centering.

use flexmesa_const::LN10;

use crate::hydro_energy::{eval_face_area_time_centered_ad, eval_pvsc_cell_ad};
use crate::hydro_eqns::HenyeySystem;
use crate::model::StarModel;
use crate::star_ad::{
    StarAdExt, StarAdOrder1, I_LND_00, I_LND_M1, I_LNR_00, I_V_00, I_V_M1,
    I_V_P1,
};

/// Evaluates total EOS pressure $P_{\mathrm{eos}}$ in cell `k` with optional
/// $\theta_P$ velocity time-centering (`hydro_momentum.f90:151-162`).
#[must_use]
pub fn eval_cell_peos_time_centered_ad(
    model: &StarModel,
    k: usize,
    shift_m1: bool,
) -> (StarAdOrder1, f64) {
    let logt = model.ln_t[k] / LN10;
    let p_theta =
        if model.controls.is_velocity_time_centered(model.model_number)
            && model.controls.include_p_in_velocity_time_centering
            && logt
                <= model
                    .controls
                    .max_logt_for_include_p_and_l_in_velocity_time_centering
        {
            model.controls.p_theta_for_velocity_time_centering
        } else {
            1.0
        };
    let peos_ad = if shift_m1 {
        model.ln_peos_ad[k].shift_m1().exp()
    } else {
        model.ln_peos_ad[k].exp()
    };
    let peos_start = model.peos_start.get(k).copied().unwrap_or(model.peos[k]);
    (peos_ad * p_theta + peos_start * (1.0 - p_theta), p_theta)
}

/// Evaluates face control-volume mass $\Delta m_{\mathrm{face},k}$ including
/// optional baryonic `mass_correction` (`hydro_momentum.f90:187-199`).
#[must_use]
pub fn eval_face_dm_bar(model: &StarModel, k: usize) -> f64 {
    if !model.controls.use_mass_corrections {
        return if k == 0 {
            0.5 * model.dm[0]
        } else {
            model.dm_bar[k]
        };
    }
    let mc_00 = model.aux.mass_correction.get(k).copied().unwrap_or(1.0);
    if k == 0 {
        0.5 * model.dm[0] * mc_00
    } else {
        let mc_m1 =
            model.aux.mass_correction.get(k - 1).copied().unwrap_or(1.0);
        0.5 * (model.dm[k - 1] * mc_m1 + model.dm[k] * mc_00)
    }
}

/// Evaluates the hydrostatic gravitational acceleration
/// $g_{\mathrm{HSE},k} = -f_{P,\mathrm{rot},k} G_k m_{\mathrm{grav},k} / r_k^2$
/// (`expected_HSE_grav_term` in `star/private/hydro_momentum.f90:348-360`).
#[must_use]
pub fn expected_hse_grav_term(
    model: &StarModel,
    k: usize,
    r2: StarAdOrder1,
) -> StarAdOrder1 {
    let fp = if model.controls.rotation_flag
        && model.controls.use_gravity_rotation_correction
    {
        model.aux.fp_rot.get(k).copied().unwrap_or(1.0)
    } else {
        1.0
    };
    -(model.cgrav[k] * model.m_grav[k] * fp) / r2
}

/// Evaluates Rayleigh-Taylor instability (`RTI`) momentum diffusion and source
/// terms at interior face `1 <= k < nz - 1` (`setup_RTI_terms` in
/// `star/private/hydro_momentum.f90:250-284`).
#[must_use]
fn setup_rti_momentum_terms(
    model: &StarModel,
    k: usize,
    area: StarAdOrder1,
    dm_face: f64,
) -> StarAdOrder1 {
    if k == 0
        || k + 1 >= model.nz
        || !model.controls.rti_flag
        || !(model.controls.v_flag || model.controls.u_flag)
    {
        return StarAdOrder1::ZERO;
    }
    let dvdt_diffusion = if model.controls.dudt_rti_diffusion_factor > 0.0 {
        let fac = model.controls.dudt_rti_diffusion_factor;
        let sig_m1 =
            model.aux.sigmid_rti.get(k - 1).copied().unwrap_or(0.0) * fac;
        let sig_00 = model.aux.sigmid_rti.get(k).copied().unwrap_or(0.0) * fac;
        let v_val = |idx: usize| {
            if model.controls.v_flag {
                model.v[idx]
            } else {
                model.u[idx]
            }
        };
        let v_00 = StarAdOrder1::variable(v_val(k), I_V_00);
        let v_m1 = StarAdOrder1::variable(v_val(k - 1), I_V_M1);
        let v_p1 = StarAdOrder1::variable(v_val(k + 1), I_V_P1);
        ((v_p1 - v_00) * sig_00 - (v_00 - v_m1) * sig_m1)
            / model.dm_bar[k].max(1.0e-30)
    } else {
        StarAdOrder1::ZERO
    };
    let eta_k = model.aux.eta_rti.get(k).copied().unwrap_or(0.0);
    let dvdt_kick = if eta_k > 0.0
        && model.controls.dlnddt_rti_diffusion_factor > 0.0
        && model.dt > 0.0
    {
        let f = area
            * (model.controls.dlnddt_rti_diffusion_factor * eta_k
                / dm_face.max(1.0e-30));
        let rho_00 = StarAdOrder1::variable(model.lnd[k], I_LND_00).exp();
        let rho_m1 = StarAdOrder1::variable(model.lnd[k - 1], I_LND_M1).exp();
        f * (rho_00 - rho_m1) / model.dt
    } else {
        StarAdOrder1::ZERO
    };
    dvdt_diffusion + dvdt_kick
}

/// Evaluates all non-HSE acceleration and momentum-flux terms at face `k`
/// (`expected_non_HSE_term` in `star/private/hydro_momentum.f90:374-461`).
#[must_use]
pub fn expected_non_hse_term(
    model: &StarModel,
    k: usize,
    area: StarAdOrder1,
    dm_face: f64,
) -> StarAdOrder1 {
    let mut non_hse = model
        .aux
        .extra_grav
        .get(k)
        .copied()
        .unwrap_or(StarAdOrder1::ZERO);

    let dt = model.dt;
    let has_vel = (model.controls.v_flag || model.controls.u_flag) && dt > 0.0;
    if has_vel {
        let (v_curr, v_st) = if model.controls.v_flag {
            (model.v[k], model.v_start[k])
        } else {
            (model.u[k], model.u_start[k])
        };
        let v_00 = StarAdOrder1::variable(v_curr, I_V_00);
        let dv_dt = (v_00 - v_st) / dt;
        non_hse = non_hse - dv_dt;

        if model.controls.v_flag
            && model.controls.drag_coefficient > 0.0
            && model.q[k] > model.controls.min_q_for_drag
        {
            non_hse = non_hse - v_00 * (model.controls.drag_coefficient / dt);
        }
    }

    if let Some(&uq) = model.aux.uq_ad.get(k) {
        if uq.val != 0.0 || uq.d1.iter().any(|&d| d != 0.0) {
            non_hse = non_hse + uq;
        }
    }

    if k > 0 && model.controls.use_pvsc_art_visc && model.controls.v_flag {
        let pvsc_00 = eval_pvsc_cell_ad(model, k);
        let pvsc_m1 = eval_pvsc_cell_ad(model, k - 1).shift_m1();
        non_hse = non_hse - (area / dm_face.max(1.0e-30)) * (pvsc_m1 - pvsc_00);
    }

    if k > 0 && model.controls.rsp2_flag && model.controls.rsp2.alfap != 0.0 {
        let ptrb_00 = crate::rsp2_ad::eval_rsp2_ptrb_cell_ad(model, k, false);
        let ptrb_m1 =
            crate::rsp2_ad::eval_rsp2_ptrb_cell_ad(model, k - 1, true);
        non_hse = non_hse - (area / dm_face.max(1.0e-30)) * (ptrb_m1 - ptrb_00);
    }

    if k > 0 && model.controls.mlt_pturb_factor > 0.0 {
        let fac = (1.0 / 3.0) * model.controls.mlt_pturb_factor;
        let vc_00 = model.aux.mlt_vc_old.get(k).copied().unwrap_or(0.0);
        let vc_m1 = model.aux.mlt_vc_old.get(k - 1).copied().unwrap_or(0.0);
        if vc_00 > 0.0 || vc_m1 > 0.0 {
            let rho_00 = StarAdOrder1::variable(model.lnd[k], I_LND_00).exp();
            let rho_m1 =
                StarAdOrder1::variable(model.lnd[k - 1], I_LND_M1).exp();
            let pturb_00 = rho_00 * (fac * vc_00 * vc_00);
            let pturb_m1 = rho_m1 * (fac * vc_m1 * vc_m1);
            non_hse =
                non_hse - (area / dm_face.max(1.0e-30)) * (pturb_m1 - pturb_00);
        }
    }

    if k > 0 && model.controls.rti_flag {
        non_hse = non_hse + setup_rti_momentum_terms(model, k, area, dm_face);
    }

    non_hse
}

/// Hydrostatic / hydrodynamic momentum equation at face `k` (`do1_momentum_eqn`
/// in `star/private/hydro_momentum.f90:79-436`).
pub fn do1_momentum_eqn(model: &StarModel, k: usize, sys: &mut HenyeySystem) {
    let use_surf_momentum_bc = model.controls.use_momentum_outer_bc
        || model.controls.use_zero_pgas_outer_bc
        || model.controls.use_fixed_psurf_outer_bc;
    if k == 0 && !use_surf_momentum_bc {
        return;
    }
    let lnr_k = StarAdOrder1::variable(model.ln_r[k], I_LNR_00);
    let r_k = lnr_k.exp();
    let using_v_tc =
        model.controls.is_velocity_time_centered(model.model_number)
            && (model.controls.v_flag || model.controls.u_flag)
            && model.dt > 0.0
            && !model.ln_r_start.is_empty();
    let lnr_start_k = model.ln_r_start.get(k).copied().unwrap_or(model.ln_r[k]);
    let r2 = if using_v_tc {
        let r_start = lnr_start_k.exp().max(1.0);
        (r_k * r_start).max(StarAdOrder1::ONE)
    } else {
        (r_k * r_k).max(StarAdOrder1::ONE)
    };
    let area = eval_face_area_time_centered_ad(lnr_k, lnr_start_k, using_v_tc)
        .max(StarAdOrder1::ONE);
    let dm_face = eval_face_dm_bar(model, k);
    let g_hse = expected_hse_grav_term(model, k, r2);
    let non_hse = expected_non_hse_term(model, k, area, dm_face);
    let dp_expected = (g_hse + non_hse) * (dm_face / area);

    let i_equ = model.layout.i_dv_dt;
    if k == 0 {
        let (p_0, p_theta_0) = eval_cell_peos_time_centered_ad(model, 0, false);
        let mut p_tot_0 = p_0;
        if model.controls.use_pvsc_art_visc && model.controls.v_flag {
            p_tot_0 = p_tot_0 + eval_pvsc_cell_ad(model, 0);
        }
        if model.controls.rsp2_flag && model.controls.rsp2.alfap != 0.0 {
            p_tot_0 = p_tot_0
                + crate::rsp2_ad::eval_rsp2_ptrb_cell_ad(model, 0, false);
        }
        let p_surf = if model.controls.use_zero_pgas_outer_bc {
            let t_s0 = model
                .ln_t_start
                .first()
                .copied()
                .unwrap_or(model.ln_t[0])
                .exp();
            StarAdOrder1::constant((flexmesa_const::CRAD / 3.0) * t_s0.powi(4))
        } else if model.controls.use_fixed_psurf_outer_bc {
            StarAdOrder1::constant(model.controls.fixed_psurf)
        } else {
            model.ln_peos_surf_ad.exp()
        };
        let resid = (dp_expected - (p_surf - p_tot_0)) / p_0;
        sys.save_eqn_residual(&model.layout, 0, i_equ, &resid);
        if sys.nvar > model.layout.nvar_hydro && model.species > 0 {
            let p_0_val = p_0.val.max(1.0e-30);
            let resid_val = resid.val;
            for j in 0..model.species {
                let dlnp_0 = model
                    .aux
                    .dln_peos_dxa_for_partials
                    .get(j)
                    .copied()
                    .unwrap_or(0.0);
                if dlnp_0 != 0.0 {
                    let dp_0 = p_theta_0 * model.peos[0] * dlnp_0;
                    let d_res_0 = (dp_0 - resid_val * dp_0) / p_0_val;
                    sys.add_partial(
                        0,
                        i_equ,
                        0,
                        model.layout.chem_var_index(j),
                        d_res_0,
                    );
                }
            }
        }
    } else {
        let (p_00, p_theta_00) =
            eval_cell_peos_time_centered_ad(model, k, false);
        let (p_m1, p_theta_m1) =
            eval_cell_peos_time_centered_ad(model, k - 1, true);
        let (alfa, beta) = model.get_face_weights(k);
        let p_avg = p_00 * alfa + p_m1 * beta;
        let resid = (dp_expected - (p_m1 - p_00)) / p_avg;
        sys.save_eqn_residual(&model.layout, k, i_equ, &resid);

        if sys.nvar > model.layout.nvar_hydro && model.species > 0 {
            let p_avg_val = p_avg.val.max(1.0e-30);
            let resid_val = resid.val;
            let base_00 = k * model.species;
            let base_m1 = (k - 1) * model.species;
            for j in 0..model.species {
                let dlnp_00 = model
                    .aux
                    .dln_peos_dxa_for_partials
                    .get(base_00 + j)
                    .copied()
                    .unwrap_or(0.0);
                let dlnp_m1 = model
                    .aux
                    .dln_peos_dxa_for_partials
                    .get(base_m1 + j)
                    .copied()
                    .unwrap_or(0.0);
                if dlnp_00 != 0.0 {
                    let dp_00 = p_theta_00 * model.peos[k] * dlnp_00;
                    let d_res_00 =
                        (dp_00 - resid_val * alfa * dp_00) / p_avg_val;
                    sys.add_partial(
                        k,
                        i_equ,
                        k,
                        model.layout.chem_var_index(j),
                        d_res_00,
                    );
                }
                if dlnp_m1 != 0.0 {
                    let dp_m1 = p_theta_m1 * model.peos[k - 1] * dlnp_m1;
                    let d_res_m1 =
                        (-dp_m1 - resid_val * beta * dp_m1) / p_avg_val;
                    sys.add_partial(
                        k,
                        i_equ,
                        k - 1,
                        model.layout.chem_var_index(j),
                        d_res_m1,
                    );
                }
            }
        }
    }
}

/// Optional kinematic radius-velocity equation for face `k` (`do1_radius_eqn`
/// in `star/private/hydro_momentum.f90:552-608`).
pub fn do1_radius_eqn(model: &StarModel, k: usize, sys: &mut HenyeySystem) {
    let Some(i_dlnr_dt) = model.layout.i_dlnr_dt else {
        return;
    };
    let ln_t_ref = model.ln_t_start.get(k).copied().unwrap_or(model.ln_t[k]);
    let logt = ln_t_ref / LN10;
    let force_zero_v = model.q[k] > model.controls.velocity_q_upper_bound
        || model.tau[k] < model.controls.velocity_tau_lower_bound
        || logt < model.controls.velocity_logt_lower_bound;

    let (vel_curr, vel_start) = if model.controls.v_flag {
        (
            model.v[k],
            model.v_start.get(k).copied().unwrap_or(model.v[k]),
        )
    } else {
        (
            model.u[k],
            model.u_start.get(k).copied().unwrap_or(model.u[k]),
        )
    };
    let v_var = StarAdOrder1::variable(vel_curr, I_V_00);
    let resid = if force_zero_v {
        let cs = model.csound[k].max(1.0);
        -v_var / cs
    } else {
        let vel_tc =
            model.controls.is_velocity_time_centered(model.model_number);
        let v_ad = if model.controls.u_flag && !model.controls.v_flag {
            let uf_ad = model.aux.u_face_ad.get(k).copied().unwrap_or(v_var);
            if vel_tc {
                let uf_s =
                    model.aux.u_face_start.get(k).copied().unwrap_or(vel_start);
                (uf_ad + uf_s) * 0.5
            } else {
                uf_ad
            }
        } else if vel_tc {
            (v_var + vel_start) * 0.5
        } else {
            v_var
        };
        let lnr_ad = StarAdOrder1::variable(model.ln_r[k], I_LNR_00);
        let r_start = model.ln_r_start[k].exp().max(1.0);
        v_ad * (model.dt / r_start)
            - ((lnr_ad - model.ln_r_start[k]).exp() - 1.0)
    };
    sys.save_eqn_residual(&model.layout, k, i_dlnr_dt, &resid);
}
