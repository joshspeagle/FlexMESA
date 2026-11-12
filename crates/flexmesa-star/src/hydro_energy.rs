// SPDX-License-Identifier: LGPL-3.0-or-later
//! Cell-centered energy conservation equation (`do1_energy_eqn`) with 3-cell
//! stencil [`StarAdOrder1`] automatic differentiation and composition partial
//! derivatives (`star/private/hydro_energy.f90` and `star_utils.f90`).
//!
//! Implements full line-by-line fidelity with upstream `hydro_energy.f90`:
//! - **Luminosity divergence (`setup_dL_dm`, `hydro_energy.f90:211-228`)** with
//!   optional velocity time-centering weight $\theta_L$.
//! - **Nuclear, neutrino, and non-nuclear energy sources
//!   (`setup_sources_and_others`, `hydro_energy.f90:230-330`)**:
//!   - Operator-split `burn_avg_epsnuc` and `nonlocal_nico_decay_heat` gating,
//!   - Time-centered thermal neutrino losses
//!     $\tfrac{1}{2}(\epsilon_{\nu,\mathrm{start},k} + \epsilon_{\nu,k})$,
//!   - `extra_heat` with $(\ln\rho, \ln T)$ partials,
//!   - `eps_wd_sedimentation`, `eps_diffusion`, `eps_pre_mix`,
//!     `eps_phase_separation`, `irradiation_heat`, and `eps_mdot`,
//!   - `RSP2` and `TDC` turbulent work heating `eq_ad`,
//!   - Rayleigh-Taylor instability (`RTI`) internal energy diffusion
//!     (`setup_RTI_diffusion`, `hydro_energy.f90:333-373`),
//!   - Velocity drag heating $\tfrac{1}{2} c_{\mathrm{drag}}(v_k^2 + v_{k+1}^2)/\Delta t$.
//! - **Turbulent energy time derivative (`setup_d_turbulent_energy_dt`,
//!   `hydro_energy.f90:375-408`)** for `RSP2` and `TDC`.
//! - **Internal energy rate and mechanical work (`setup_de_dt_and_friends`,
//!   `eval_simple_PdV_work`, `eval_dwork`, `hydro_energy.f90:109-209, 441-902`)**
//!   including time-centered pressure $\theta_P$, artificial viscosity
//!   $P_{\mathrm{vsc}}$, MLT turbulent pressure $P_{\mathrm{turb,MLT}}$, and
//!   time-centered spherical face area $A_k = \tfrac{4\pi}{3}(r_k^2 + r_k r_{\mathrm{start},k} + r_{\mathrm{start},k}^2)$.
//! - **Composition partial derivatives (`unpack_res18`, `hydro_energy.f90:490-553`)**
//!   including `d_epsnuc_dx`, `dln_e_dxa_for_partials`, `d_eps_grav_dx`, and
//!   `dln_peos_dxa_for_partials`.

use flexmesa_const::{FOUR_PI, LN10};
use flexmesa_turb::MltOption;

use crate::controls::EnergyEqnOption;
use crate::hydro_eqns::HenyeySystem;
use crate::model::StarModel;
use crate::star_ad::{
    StarAdExt, StarAdOrder1, I_LND_00, I_L_00, I_L_P1, I_V_00, I_V_P1,
};

/// Evaluates artificial viscosity pressure $P_{\mathrm{vsc},k}$ in cell `k`
/// (`get_Pvsc_ad` in `star/private/star_utils.f90:3454-3482`).
#[must_use]
pub fn eval_pvsc_cell_ad(model: &StarModel, k: usize) -> StarAdOrder1 {
    if !model.controls.use_pvsc_art_visc
        || !model.controls.v_flag
        || model.controls.pvsc_cq <= 0.0
    {
        return StarAdOrder1::ZERO;
    }
    let v_00 = StarAdOrder1::variable(model.v[k], I_V_00);
    let v_p1 = if k + 1 < model.nz {
        StarAdOrder1::variable(model.v[k + 1], I_V_P1)
    } else {
        StarAdOrder1::constant(model.v_center)
    };
    let peos_00 = model.ln_peos_ad[k].exp();
    let rho_00 = StarAdOrder1::variable(model.lnd[k], I_LND_00).exp();
    let peos_div_rho = (peos_00 / rho_00).max(StarAdOrder1::constant(1.0e-30));
    let dv = (v_p1 - v_00) - model.controls.pvsc_zshock * peos_div_rho.sqrt();
    if dv.val <= 0.0 {
        return StarAdOrder1::ZERO;
    }
    rho_00 * model.controls.pvsc_cq * (dv * dv)
}

/// Evaluates effective cell pressure $P_{\mathrm{cell},k}$ including optional
/// velocity time-centering $\theta_P$, artificial viscosity $P_{\mathrm{vsc},k}$,
/// and MLT turbulent pressure $P_{\mathrm{turb,MLT},k}$ (`get_P_cell_for_dwork`
/// in `star/private/hydro_energy.f90:709-747`).
#[must_use]
pub fn eval_cell_pressure_for_work_ad(
    model: &StarModel,
    k: usize,
) -> (StarAdOrder1, f64) {
    let (mut p_ad, p_theta) =
        crate::hydro_momentum::eval_cell_peos_time_centered_ad(model, k, false);

    if model.controls.use_pvsc_art_visc && model.controls.v_flag {
        p_ad = p_ad + eval_pvsc_cell_ad(model, k);
    }
    if model.controls.rsp2_flag && model.controls.rsp2.alfap != 0.0 {
        p_ad = p_ad + crate::rsp2_ad::eval_rsp2_ptrb_cell_ad(model, k, false);
    }
    if model.controls.mlt_pturb_factor > 0.0 {
        let vc_old = model.aux.mlt_vc_old.get(k).copied().unwrap_or(0.0);
        if vc_old > 0.0 {
            let rho_00 = StarAdOrder1::variable(model.lnd[k], I_LND_00).exp();
            let pturb = rho_00
                * ((1.0 / 3.0)
                    * model.controls.mlt_pturb_factor
                    * vc_old
                    * vc_old);
            p_ad = p_ad + pturb;
        }
    }
    (p_ad, p_theta)
}

/// Evaluates the luminosity divergence $dL/dm = (L_k - L_{k+1}) / \Delta m_k$
/// with optional $\theta_L$ velocity time-centering (`setup_dL_dm` in
/// `star/private/hydro_energy.f90:211-228`).
#[must_use]
pub fn setup_dl_dm(model: &StarModel, k: usize) -> StarAdOrder1 {
    let logt = model.ln_t[k] / LN10;
    let l_theta =
        if model.controls.is_velocity_time_centered(model.model_number)
            && model.controls.include_l_in_velocity_time_centering
            && logt
                <= model
                    .controls
                    .max_logt_for_include_p_and_l_in_velocity_time_centering
        {
            model.controls.l_theta_for_velocity_time_centering
        } else {
            1.0
        };
    let l_start_00 = model.lum_start.get(k).copied().unwrap_or(model.lum[k]);
    let lum_00 = StarAdOrder1::variable(model.lum[k], I_L_00) * l_theta
        + l_start_00 * (1.0 - l_theta);
    let lum_p1 = if k + 1 < model.nz {
        let l_start_p1 = model
            .lum_start
            .get(k + 1)
            .copied()
            .unwrap_or(model.lum[k + 1]);
        StarAdOrder1::variable(model.lum[k + 1], I_L_P1) * l_theta
            + l_start_p1 * (1.0 - l_theta)
    } else {
        StarAdOrder1::constant(model.l_center)
    };
    (lum_00 - lum_p1) / model.dm[k].max(1.0e-30)
}

/// Evaluates Rayleigh-Taylor instability (`RTI`) internal energy diffusion
/// (`setup_RTI_diffusion` in `star/private/hydro_energy.f90:333-373`).
#[must_use]
fn setup_rti_diffusion(model: &StarModel, k: usize) -> StarAdOrder1 {
    if !model.controls.rti_flag
        || model.controls.dedt_rti_diffusion_factor <= 0.0
    {
        return StarAdOrder1::ZERO;
    }
    let dm_k = model.dm[k].max(1.0e-30);
    let fac = model.controls.dedt_rti_diffusion_factor;
    let e_00 = model.ln_e_ad[k].exp();
    let term_00 = if k > 0 {
        let sig_00 = model.aux.sig_rti.get(k).copied().unwrap_or(0.0) * fac;
        if sig_00 > 0.0 {
            let e_m1 = model.ln_e_ad[k - 1].shift_m1().exp();
            (e_00 - e_m1) * (sig_00 / dm_k)
        } else {
            StarAdOrder1::ZERO
        }
    } else {
        StarAdOrder1::ZERO
    };
    let term_p1 = if k + 1 < model.nz {
        let sig_p1 = model.aux.sig_rti.get(k + 1).copied().unwrap_or(0.0) * fac;
        if sig_p1 > 0.0 {
            let e_p1 = model.ln_e_ad[k + 1].shift_p1().exp();
            (e_p1 - e_00) * (sig_p1 / dm_k)
        } else {
            StarAdOrder1::ZERO
        }
    } else {
        StarAdOrder1::ZERO
    };
    term_p1 - term_00
}

/// Evaluates all nuclear, neutrino, extra, diffusion, premix, phase-separation,
/// turbulent work, RTI diffusion, drag, and irradiation heating terms for cell
/// `k` (`setup_sources_and_others` in `star/private/hydro_energy.f90:230-330`).
#[must_use]
pub fn setup_sources_and_others(
    model: &StarModel,
    k: usize,
) -> (StarAdOrder1, bool) {
    let t_start_k = model
        .ln_t_start
        .get(k)
        .copied()
        .unwrap_or(model.ln_t[k])
        .exp();
    let doing_op_split_burn = model.controls.op_split_burn
        && t_start_k >= model.controls.op_split_burn_min_t;

    let eps_nuc_ad = if model.controls.eps_nuc_factor == 0.0
        || model.controls.nonlocal_nico_decay_heat
    {
        StarAdOrder1::ZERO
    } else if doing_op_split_burn {
        let avg_eps = model
            .aux
            .burn_avg_epsnuc
            .get(k)
            .copied()
            .filter(|&v| v != 0.0)
            .unwrap_or(model.eps_nuc[k]);
        StarAdOrder1::constant(avg_eps)
    } else {
        StarAdOrder1::from_dlnd_dlnt_00(
            model.eps_nuc[k],
            model.d_epsnuc_dlnd[k],
            model.d_epsnuc_dlnt[k],
        )
    };

    let non_nuc_neu_start_k = model
        .non_nuc_neu_start
        .get(k)
        .copied()
        .unwrap_or(model.non_nuc_neu[k]);
    let non_nuc_neu_ad = StarAdOrder1::from_dlnd_dlnt_00(
        0.5 * (non_nuc_neu_start_k + model.non_nuc_neu[k]),
        0.5 * model.d_non_nuc_neu_dlnd[k],
        0.5 * model.d_non_nuc_neu_dlnt[k],
    );

    let extra_heat_ad = StarAdOrder1::from_dlnd_dlnt_00(
        model.aux.extra_heat.get(k).copied().unwrap_or(0.0),
        model.aux.d_extra_heat_dlnd.get(k).copied().unwrap_or(0.0),
        model.aux.d_extra_heat_dlnt.get(k).copied().unwrap_or(0.0),
    );

    let eps_wd = model
        .aux
        .eps_wd_sedimentation
        .get(k)
        .copied()
        .unwrap_or(0.0);
    let eps_diff = model.aux.eps_diffusion.get(k).copied().unwrap_or(0.0);
    let diff_heat = if eps_wd == 0.0 { eps_diff } else { eps_wd };
    let premix_heat =
        if model.controls.do_conv_premix && model.controls.do_premix_heating {
            model.aux.eps_pre_mix.get(k).copied().unwrap_or(0.0)
        } else {
            0.0
        };
    let phase_heat = if model.controls.do_phase_separation
        && model.controls.do_phase_separation_heating
    {
        model
            .aux
            .eps_phase_separation
            .get(k)
            .copied()
            .unwrap_or(0.0)
    } else {
        0.0
    };
    let irrad_heat = model.aux.irradiation_heat.get(k).copied().unwrap_or(0.0);
    let others_scalar =
        diff_heat + premix_heat + phase_heat + irrad_heat + model.eps_mdot[k];

    let eq_ad = if model.controls.rsp2_flag
        || (model.controls.tdc_alpha_m > 0.0
            && model.controls.mlt_option == MltOption::Tdc
            && model.controls.tdc_include_eturb_in_energy_equation
            && (model.controls.v_flag || model.controls.u_flag))
    {
        model
            .aux
            .eq_ad
            .get(k)
            .copied()
            .unwrap_or(StarAdOrder1::ZERO)
    } else {
        StarAdOrder1::ZERO
    };

    let rti_diff_ad = setup_rti_diffusion(model, k);

    let dt = model.dt.max(1.0e-30);
    let drag_energy = if model.controls.v_flag
        && model.controls.use_drag_energy
        && model.controls.drag_coefficient > 0.0
        && model.q[k] > model.controls.min_q_for_drag
    {
        let v_00 = StarAdOrder1::variable(model.v[k], I_V_00);
        let v_p1 = if k + 1 < model.nz {
            StarAdOrder1::variable(model.v[k + 1], I_V_P1)
        } else {
            StarAdOrder1::constant(model.v_center)
        };
        (v_00 * v_00 + v_p1 * v_p1)
            * (0.5 * model.controls.drag_coefficient / dt)
    } else {
        StarAdOrder1::ZERO
    };

    let total_sources = eps_nuc_ad - non_nuc_neu_ad
        + extra_heat_ad
        + eq_ad
        + rti_diff_ad
        + drag_energy
        + others_scalar;
    (total_sources, doing_op_split_burn)
}

/// Evaluates the turbulent energy time derivative $de_{\mathrm{turb}}/dt$ for
/// cell `k` (`setup_d_turbulent_energy_dt` in
/// `star/private/hydro_energy.f90:375-408`).
#[must_use]
pub fn setup_d_turbulent_energy_dt(
    model: &StarModel,
    k: usize,
) -> StarAdOrder1 {
    let dt = model.dt.max(1.0e-30);
    if model.controls.rsp2_flag {
        let etrb = if model.layout.i_w.is_some() {
            model.aux.rsp.et_rsp.get(k).copied().unwrap_or(0.0).max(0.0)
        } else {
            model.aux.etrb.get(k).copied().unwrap_or(0.0).max(0.0)
        };
        let etrb_s = model.aux.etrb_start.get(k).copied().unwrap_or(etrb);
        let et_ad = if model.layout.i_w.is_some() {
            let w_00 =
                StarAdOrder1::variable(etrb.sqrt(), crate::star_ad::I_W_00);
            w_00 * w_00
        } else {
            StarAdOrder1::constant(etrb)
        };
        return (et_ad - etrb_s) / dt;
    }
    if model.controls.mlt_option == MltOption::Tdc
        && model.controls.tdc_include_eturb_in_energy_equation
    {
        let vc_00 = model.mlt_vc_ad[k];
        let vc_p1 = if k + 1 < model.nz {
            model.mlt_vc_ad[k + 1].shift_p1()
        } else {
            StarAdOrder1::ZERO
        };
        let vc_old_00 = model.aux.mlt_vc_old.get(k).copied().unwrap_or(0.0);
        let vc_old_p1 = if k + 1 < model.nz {
            model.aux.mlt_vc_old.get(k + 1).copied().unwrap_or(0.0)
        } else {
            0.0
        };
        let ke_new = vc_00 * vc_00 + vc_p1 * vc_p1;
        let ke_old = vc_old_00 * vc_old_00 + vc_old_p1 * vc_old_p1;
        return (ke_new - ke_old) * (0.75 / dt);
    }
    StarAdOrder1::ZERO
}

/// Evaluates the time-centered face area
/// $A_k = \tfrac{4\pi}{3}(r_k^2 + r_k r_{\mathrm{start},k} + r_{\mathrm{start},k}^2)$
/// (`get_area_info_opt_time_center` in `star/private/star_utils.f90:3671-3688`).
#[must_use]
pub fn eval_face_area_time_centered_ad(
    lnr_ad: StarAdOrder1,
    lnr_start: f64,
    time_center: bool,
) -> StarAdOrder1 {
    let r = lnr_ad.exp();
    if time_center {
        let rs = lnr_start.exp();
        (r * r + r * rs + rs * rs) * (FOUR_PI / 3.0)
    } else {
        r * r * FOUR_PI
    }
}

/// Evaluates the inverse specific-power equation scale factor `scal` in
/// $(\mathrm{erg\,g^{-1}\,s^{-1}})^{-1}$ for cell `k` (`set_energy_eqn_scal` in
/// `star/private/star_utils.f90:3691-3710`).
///
/// In thin envelope zones where $e_{\mathrm{start},k}\Delta m_k /
/// E_{\mathrm{int,start}} < \mathrm{dedt\_eqn\_r\_scale}^{-1}$, caps the dimensionless
/// prefactor at `cell_energy_fraction_start * dedt_eqn_r_scale` so IEEE-754 ULP
/// differences in $L_k - L_{k+1}$ are measured relative to
/// $E_{\mathrm{int,start}} / \mathrm{dedt\_eqn\_r\_scale}$ rather than amplified
/// by $1 / \Delta m_k$.
#[must_use]
pub fn set_energy_eqn_scal(model: &StarModel, k: usize) -> f64 {
    let dt = model.dt.max(1.0e-30);
    let mut scal: f64 = if k > 0 { 1.0 } else { 1.0e-6 };
    if model.controls.dedt_eqn_r_scale > 0.0 {
        let total_e = if model.aux.total_internal_energy_start > 0.0 {
            model.aux.total_internal_energy_start
        } else {
            (0..model.nz)
                .map(|j| model.energy_start[j].max(0.0) * model.dm[j])
                .sum::<f64>()
        };
        if total_e > 0.0 {
            let cell_energy_fraction_start =
                (model.energy_start[k].max(0.0) * model.dm[k]) / total_e;
            scal = scal.min(
                cell_energy_fraction_start * model.controls.dedt_eqn_r_scale,
            );
        }
    }
    scal * dt / model.energy_start[k].max(1.0)
}

/// Energy conservation equation for cell `k` (`do1_energy_eqn` in
/// `star/private/hydro_energy.f90:36-905`).
pub fn do1_energy_eqn(
    model: &StarModel,
    k: usize,
    include_chem: bool,
    sys: &mut HenyeySystem,
) {
    let dl_dm = setup_dl_dm(model, k);
    let (sources, doing_op_split_burn) = setup_sources_and_others(model, k);
    let d_eturb_dt = setup_d_turbulent_energy_dt(model, k);
    let dt = model.dt.max(1.0e-30);

    let v_flag_only = model.controls.v_flag && !model.controls.u_flag;
    let u_flag_only = model.controls.u_flag && !model.controls.v_flag;
    let vel_tc = model.controls.is_velocity_time_centered(model.model_number)
        && model.dt > 0.0
        && !model.ln_r_start.is_empty();

    let mut dwork_fac_m1 = 0.0_f64;
    let mut dwork_fac_00 = 0.0_f64;
    let mut dwork_fac_p1 = 0.0_f64;
    let mut has_dedt_dxa = false;

    let esum = if u_flag_only {
        has_dedt_dxa = true;
        let total_rates =
            crate::riemann_ad::eval_u_flag_total_energy_rates_ad(model, k, dt);
        -dl_dm + sources - d_eturb_dt - total_rates
    } else if v_flag_only {
        let simple_pdv = vel_tc
            && model
                .controls
                .use_p_d_1_div_rho_form_of_work_when_time_centering_velocity;
        match model.controls.energy_eqn_option {
            EnergyEqnOption::EpsGrav => {
                let dwork_dm = if simple_pdv {
                    crate::hydro_work::eval_v_flag_simple_pdv_dm_ad(
                        model, k, true, vel_tc,
                    )
                    .0
                } else {
                    crate::hydro_work::eval_v_flag_dwork_dm_ad(
                        model, k, true, vel_tc,
                    )
                    .0
                };
                -dl_dm + sources - d_eturb_dt - dwork_dm + model.eps_grav_ad[k]
            }
            EnergyEqnOption::Dedt => {
                has_dedt_dxa = true;
                let de_dt =
                    (model.ln_e_ad[k].exp() - model.energy_start[k]) / dt;
                if simple_pdv {
                    let (dwork_dm, fac_00) =
                        crate::hydro_work::eval_v_flag_simple_pdv_dm_ad(
                            model, k, false, vel_tc,
                        );
                    dwork_fac_00 = fac_00;
                    -dl_dm + sources - d_eturb_dt - dwork_dm - de_dt
                } else {
                    let (dwork_dm, f_m1, f_00, f_p1) =
                        crate::hydro_work::eval_v_flag_dwork_dm_ad(
                            model, k, false, vel_tc,
                        );
                    dwork_fac_m1 = f_m1;
                    dwork_fac_00 = f_00;
                    dwork_fac_p1 = f_p1;
                    let dke_dt =
                        crate::hydro_work::eval_v_flag_dke_dt_ad(model, k, dt);
                    let dpe_dt =
                        crate::hydro_work::eval_cell_dpe_dt_ad(model, k, dt);
                    -dl_dm + sources
                        - d_eturb_dt
                        - dwork_dm
                        - dke_dt
                        - dpe_dt
                        - de_dt
                }
            }
        }
    } else {
        match model.controls.energy_eqn_option {
            EnergyEqnOption::EpsGrav => {
                -dl_dm + sources - d_eturb_dt + model.eps_grav_ad[k]
            }
            EnergyEqnOption::Dedt => {
                has_dedt_dxa = true;
                let e_ad = model.ln_e_ad[k].exp();
                let de_dt = (e_ad - model.energy_start[k]) / dt;
                let (p_ad, p_th) = eval_cell_pressure_for_work_ad(model, k);
                let inv_rho_ad =
                    (-StarAdOrder1::variable(model.lnd[k], I_LND_00)).exp();
                let inv_rho_start = (-model.lnd_start[k]).exp();
                let d_inv_rho_dt = (inv_rho_ad.val - inv_rho_start) / dt;
                dwork_fac_00 = p_th * model.peos[k] * d_inv_rho_dt;
                let pdv_dt = p_ad * (inv_rho_ad - inv_rho_start) / dt;
                -dl_dm + sources - d_eturb_dt - de_dt - pdv_dt
            }
        }
    };

    let scal = set_energy_eqn_scal(model, k);
    let resid = esum * scal;
    let i_equ = model.layout.i_dlne_dt;
    sys.save_eqn_residual(&model.layout, k, i_equ, &resid);

    if include_chem {
        let sp = model.species;
        let base = k * sp;
        let active_nuc_dxa = !doing_op_split_burn
            && model.controls.eps_nuc_factor != 0.0
            && !model.controls.nonlocal_nico_decay_heat;
        let dlnp = |cell: usize, j: usize| {
            model
                .aux
                .dln_peos_dxa_for_partials
                .get(cell * sp + j)
                .copied()
                .unwrap_or(0.0)
        };
        for j in 0..sp {
            let ivar = model.layout.chem_var_index(j);
            let mut d_equ_dx = if active_nuc_dxa {
                scal * model.d_epsnuc_dx[base + j]
            } else {
                0.0
            };
            if has_dedt_dxa {
                let dlne_dx = model
                    .aux
                    .dln_e_dxa_for_partials
                    .get(base + j)
                    .copied()
                    .unwrap_or(0.0);
                d_equ_dx -= scal
                    * ((model.energy[k] / dt) * dlne_dx
                        + dwork_fac_00 * dlnp(k, j));
                if k > 0 && dwork_fac_m1 != 0.0 {
                    sys.add_partial(
                        k,
                        i_equ,
                        k - 1,
                        ivar,
                        -scal * dwork_fac_m1 * dlnp(k - 1, j),
                    );
                }
                if k + 1 < model.nz && dwork_fac_p1 != 0.0 {
                    sys.add_partial(
                        k,
                        i_equ,
                        k + 1,
                        ivar,
                        -scal * dwork_fac_p1 * dlnp(k + 1, j),
                    );
                }
            } else if model.controls.include_composition_in_eps_grav {
                let deg_dx = model
                    .aux
                    .d_eps_grav_dx
                    .get(base + j)
                    .copied()
                    .unwrap_or(0.0);
                d_equ_dx += scal * deg_dx;
            }
            sys.add_partial(k, i_equ, k, ivar, d_equ_dx);
        }
    }
}
