// SPDX-License-Identifier: GPL-3.0-or-later
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
//!     (`setup_RTI_diffusion`, `star/private/hydro_energy.f90:312-352`),
//!   - Velocity drag heating $\tfrac{1}{2} c_{\mathrm{drag}}(v_k^2 + v_{k+1}^2)/\Delta t$.
//! - **Turbulent energy time derivative (`setup_d_turbulent_energy_dt`,
//!   `star/private/hydro_energy.f90:354-387`)** for `RSP2` and `TDC`.
//! - **Internal energy rate and mechanical work (`setup_de_dt_and_friends`,
//!   `eval_simple_PdV_work`, `eval_dwork`, `star/private/hydro_energy.f90:109-209, 420-877`)**
//!   including time-centered pressure $\theta_P$, artificial viscosity
//!   $P_{\mathrm{vsc}}$, MLT turbulent pressure $P_{\mathrm{turb,MLT}}$, and
//!   time-centered spherical face area $A_k = \tfrac{4\pi}{3}(r_k^2 + r_k r_{\mathrm{start},k} + r_{\mathrm{start},k}^2)$.
//! - **Composition partial derivatives (`unpack_res18`, `star/private/hydro_energy.f90:469-532`)**
//!   including `d_epsnuc_dx`, `dln_e_dxa_for_partials`, `d_eps_grav_dx`, and
//!   `dln_peos_dxa_for_partials`.

use flexmesa_const::{FOUR_PI, LN10};
use flexmesa_eos::I_LN_E;
use flexmesa_turb::MltOption;

use crate::controls::{EnergyEqnOption, StarError};
use crate::hydro_eqns::HenyeySystem;
use crate::model::StarModel;
use crate::star_ad::{
    StarAdExt, StarAdOrder1, I_LND_00, I_LNT_00, I_L_00, I_L_P1, I_V_00, I_V_P1,
};

/// Evaluates artificial viscosity pressure $P_{\mathrm{vsc},k}$ in cell `k`
/// (`get_Pvsc_ad` in `star/private/star_utils.f90:3441-3469`). The
/// `s% Pvsc_start` side effect is `capture_pvsc_start`.
#[must_use]
pub fn eval_pvsc_cell_ad(model: &StarModel, k: usize) -> StarAdOrder1 {
    if !model.controls.use_pvsc_art_visc
        || !model.controls.v_flag
        || model.controls.pvsc_cq == 0.0
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
    let peos_div_rho = peos_00 / rho_00;
    let dv = (v_p1 - v_00) - model.controls.pvsc_zshock * peos_div_rho.sqrt();
    if dv.val <= 0.0 {
        return StarAdOrder1::ZERO;
    }
    rho_00 * model.controls.pvsc_cq * dv.pow2()
}

/// `s% Pvsc_start(k)` as `get_Pvsc_ad` leaves it once it has evaluated
/// `Pvsc(k) = pvsc` (`star/private/star_utils.f90:3452-3468`): the value
/// captured earlier in the step, or `pvsc` itself while the
/// [`crate::hydro_state::PVSC_START_UNSET`] sentinel is still in place
/// (`pvsc` is zero when `get_Pvsc_ad` returns early).
#[must_use]
pub(crate) fn pvsc_start_after_get(
    model: &StarModel,
    k: usize,
    pvsc: f64,
) -> f64 {
    match model.aux.pvsc_start.get(k) {
        Some(&start) if start < 0.0 => pvsc,
        Some(&start) => start,
        None => pvsc,
    }
}

/// Stores the `s% Pvsc_start` side effect of the first `get_Pvsc_ad` call of
/// a step (`star/private/star_utils.f90:3452-3468`): a cell still holding the
/// `-1d99` sentinel set by `save_start_values`
/// (`star/private/struct_burn_mix.f90:313`) takes `Pvsc` at the current
/// state, zero when `get_Pvsc_ad` returns early, and keeps it for the rest of
/// the step. Upstream calls `get_Pvsc_ad` only under `use_Pvsc_art_visc`
/// (`star/private/hydro_energy.f90:710`, `star/private/star_utils.f90:3411`),
/// and every call inside one `eval_equations` sees the same state, so a pass
/// over all cells before the equations are assembled stores the same values.
pub(crate) fn capture_pvsc_start(model: &mut StarModel) {
    if !model.controls.use_pvsc_art_visc {
        return;
    }
    let nz = model.nz;
    if model.aux.pvsc_start.len() != nz {
        model
            .aux
            .pvsc_start
            .resize(nz, crate::hydro_state::PVSC_START_UNSET);
    }
    for k in 0..nz {
        if model.aux.pvsc_start[k] < 0.0 {
            model.aux.pvsc_start[k] = eval_pvsc_cell_ad(model, k).val;
        }
    }
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
    p_ad =
        p_ad + crate::hydro_momentum::eval_mlt_pturb_cell_ad(model, k, false);
    (p_ad, p_theta)
}

/// Evaluates luminosity velocity time-centering weight $\theta_L$ for cell `k`
/// (`setup_dL_dm`, `star/private/hydro_energy.f90:211-228` and
/// `okay_energy_conservation`, `star/private/evolve.f90:1279-1285`).
#[must_use]
pub(crate) fn eval_l_theta(model: &StarModel, k: usize) -> f64 {
    let logt = model.ln_t[k] / LN10;
    if model
        .controls
        .is_velocity_time_centered(model.model_number_old)
        && model.controls.include_l_in_velocity_time_centering
        && logt
            <= model
                .controls
                .max_logt_for_include_p_and_l_in_velocity_time_centering
    {
        model.controls.l_theta_for_velocity_time_centering
    } else {
        1.0
    }
}

/// Evaluates the luminosity divergence $dL/dm = (L_k - L_{k+1}) / \Delta m_k$
/// with optional $\theta_L$ velocity time-centering (`setup_dL_dm` in
/// `star/private/hydro_energy.f90:211-228`).
#[must_use]
pub fn setup_dl_dm(model: &StarModel, k: usize) -> StarAdOrder1 {
    let l_theta = eval_l_theta(model, k);
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
    // `dL_dm_ad = (L00_ad - Lp1_ad)/dm` with `dm = s% dm(k)`
    // (`star/private/hydro_energy.f90:157, 213`).
    (lum_00 - lum_p1) / model.dm[k]
}

/// `wrap_e_00` (`star/private/auto_diff_support.f90:528-536`): `energy(k)`
/// with `lnd` and `lnT` partials `dE_dRho_for_partials*rho` and
/// `Cv_for_partials*T`, where `dE_dRho_for_partials = d_dlnd(i_lnE)*energy/rho`
/// and `Cv_for_partials = d_dlnT(i_lnE)*energy/T`
/// (`star/private/micro.f90:499-500`).
fn wrap_e_cell_00(model: &StarModel, k: usize) -> StarAdOrder1 {
    let e = model.energy[k];
    let de_drho_for_partials = model.eos_d_dlnd[k][I_LN_E] * e / model.rho[k];
    let cv_for_partials = model.eos_d_dlnt[k][I_LN_E] * e / model.t[k];
    let mut out = StarAdOrder1::constant(e);
    out.d1[I_LND_00] = de_drho_for_partials * model.rho[k];
    out.d1[I_LNT_00] = cv_for_partials * model.t[k];
    out
}

/// Evaluates Rayleigh-Taylor instability (`RTI`) internal energy diffusion
/// (`setup_RTI_diffusion` in `star/private/hydro_energy.f90:312-352`).
#[must_use]
fn setup_rti_diffusion(model: &StarModel, k: usize) -> StarAdOrder1 {
    let mut diffusion_factor = model.controls.dedt_rti_diffusion_factor;
    let do_diffusion = model.controls.rti_flag && diffusion_factor > 0.0;
    if !do_diffusion {
        return StarAdOrder1::ZERO;
    }
    let nz = model.nz;
    let dm = model.dm[k];
    let (sigp1, e_p1) = if k + 1 < nz {
        // `star/private/hydro_energy.f90:325-332`.
        if model.aux.alpha_rti[k] > 1.0e-10 && k > 0 {
            let emin_start = model.energy_start[k + 1]
                .min(model.energy_start[k])
                .min(model.energy_start[k - 1]);
            let floor = model.controls.rti_energy_floor;
            if emin_start < 5.0 * floor {
                diffusion_factor *=
                    1.0 + (5.0 * floor - emin_start) / emin_start;
            }
        }
        (
            diffusion_factor * model.aux.sig_rti[k + 1],
            wrap_e_cell_00(model, k + 1).shift_p1(),
        )
    } else {
        (0.0, StarAdOrder1::ZERO)
    };
    let (sig00, e_m1) = if k > 0 {
        (
            diffusion_factor * model.aux.sig_rti[k],
            wrap_e_cell_00(model, k - 1).shift_m1(),
        )
    } else {
        (0.0, StarAdOrder1::ZERO)
    };
    let e_00 = wrap_e_cell_00(model, k);
    // `star/private/hydro_energy.f90:347-349`.
    let diffusion_eps_in = (e_p1 - e_00) * sigp1 / dm;
    let diffusion_eps_out = (e_00 - e_m1) * sig00 / dm;
    diffusion_eps_in - diffusion_eps_out
}

/// Evaluates all nuclear, neutrino, extra, diffusion, premix, phase-separation,
/// turbulent work, RTI diffusion, drag, and irradiation heating terms for cell
/// `k` (`setup_sources_and_others` in `star/private/hydro_energy.f90:230-330`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] when the TDC eddy-viscous heating term
/// is active (`star/private/hydro_energy.f90:268-277`): it needs
/// `compute_tdc_Eq_div_w_face`, which is not ported.
pub fn setup_sources_and_others(
    model: &StarModel,
    k: usize,
) -> Result<(StarAdOrder1, StarAdOrder1, bool), StarError> {
    let t_start_k = flexmesa_math::exp(
        model.ln_t_start.get(k).copied().unwrap_or(model.ln_t[k]),
    );
    let doing_op_split_burn = model.controls.op_split_burn
        && t_start_k >= model.controls.op_split_burn_min_t;

    let eps_nuc_ad = if model.controls.eps_nuc_factor == 0.0
        || model.controls.nonlocal_nico_decay_heat
    {
        StarAdOrder1::ZERO
    } else if doing_op_split_burn {
        let avg_eps = model.aux.burn_avg_epsnuc.get(k).copied().unwrap_or(0.0);
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

    // `others_ad`: `eps_WD_sedimentation` or `eps_diffusion`, `eps_pre_mix`
    // and `eps_phase_separation`, added in that order to a zero value with no
    // partials (`star/private/hydro_energy.f90:251-262`).
    let mut others = StarAdOrder1::ZERO;
    if model.controls.do_element_diffusion {
        if model.controls.diffusion.do_wd_sedimentation_heating {
            others.val += model.aux.eps_wd_sedimentation[k];
        } else if model.controls.diffusion.do_diffusion_heating {
            others.val += model.aux.eps_diffusion[k];
        }
    }
    if model.controls.do_conv_premix && model.controls.do_premix_heating {
        others.val += model.aux.eps_pre_mix[k];
    }
    if model.controls.do_phase_separation
        && model.controls.do_phase_separation_heating
    {
        others.val += model.aux.eps_phase_separation[k];
    }

    // `star/private/hydro_energy.f90:264-277`: `RSP2` reads `s% Eq_ad(k)`;
    // the TDC branch builds `Eq_ad` from `compute_tdc_Eq_div_w_face`.
    let eq_ad = if model.controls.rsp2_flag {
        model
            .aux
            .eq_ad
            .get(k)
            .copied()
            .unwrap_or(StarAdOrder1::ZERO)
    } else if model.controls.tdc_alpha_m > 0.0
        && crate::tdc_hydro::effective_mlt_option(model) == MltOption::Tdc
        && model.controls.tdc_include_eturb_in_energy_equation
        && (model.controls.v_flag || model.controls.u_flag)
    {
        crate::tdc_hydro::tdc_energy_eq_ad(model, k)
    } else {
        StarAdOrder1::ZERO
    };

    let rti_diff_ad = setup_rti_diffusion(model, k);

    // `drag_energy` (`star/private/hydro_energy.f90:281-302`): outer and inner
    // half-cell terms `0.5*v*(drag_coefficient*v/dt)`, both only for
    // `k /= nz`, each gated by its own face's `q`. `wrap_v_00`/`wrap_v_p1`
    // read `s% v`, which is zero without `v_flag` (`hydro_vars.f90:354`).
    let c = &model.controls;
    let mut drag_energy = StarAdOrder1::ZERO;
    if k + 1 < model.nz {
        let v_of = |j: usize| {
            if model.layout.i_v.is_some() {
                model.v[j]
            } else {
                0.0
            }
        };
        // `wrap_v_00` and `wrap_v_p1` return the Riemann face velocity
        // `u_face_ad` when `u_flag` is set (`auto_diff_support.f90:1060-1073,
        // 1089-1111`), and `s% v` otherwise.
        let v_00_ad = if c.u_flag {
            model.aux.u_face_ad[k]
        } else {
            StarAdOrder1::variable(v_of(k), I_V_00)
        };
        let v_p1_ad = if c.u_flag {
            model.aux.u_face_ad[k + 1].shift_p1()
        } else {
            StarAdOrder1::variable(v_of(k + 1), I_V_P1)
        };
        if model.q[k] > c.min_q_for_drag
            && c.drag_coefficient > 0.0
            && c.use_drag_energy
        {
            let v_00 = v_00_ad;
            let drag_force = v_00 * c.drag_coefficient / model.dt;
            drag_energy = v_00 * 0.5 * drag_force;
        }
        if model.q[k + 1] > c.min_q_for_drag
            && c.drag_coefficient > 0.0
            && c.use_drag_energy
        {
            let v_p1 = v_p1_ad;
            let drag_force = v_p1 * c.drag_coefficient / model.dt;
            drag_energy = drag_energy + v_p1 * 0.5 * drag_force;
        }
    }

    // `star/private/hydro_energy.f90:304-308`.
    let mut sources = eps_nuc_ad - non_nuc_neu_ad
        + extra_heat_ad
        + eq_ad
        + rti_diff_ad
        + drag_energy;
    sources.val += model.aux.irradiation_heat[k];
    if model.mstar_dot != 0.0 {
        sources.val += model.eps_mdot[k];
    }
    Ok((sources, others, doing_op_split_burn))
}

/// Evaluates the turbulent energy time derivative $de_{\mathrm{turb}}/dt$ for
/// cell `k` (`setup_d_turbulent_energy_dt` in
/// `star/private/hydro_energy.f90:375-408`).
#[must_use]
pub fn setup_d_turbulent_energy_dt(
    model: &StarModel,
    k: usize,
) -> StarAdOrder1 {
    let dt = model.dt;
    if model.controls.rsp2_flag {
        // `(wrap_etrb_00 - get_etrb_start)/dt` with `etrb = pow2(w)` and
        // `get_etrb_start = pow2(w_start(k))` (`hydro_energy.f90:358-359`,
        // `auto_diff_support.f90:396-400,434-439`).
        let etrb_ad =
            crate::rsp2_ad::wrap_w(model, k, crate::star_ad::I_W_00).pow2();
        let etrb_s = flexmesa_math::pow2(
            model.aux.rsp.w_start.get(k).copied().unwrap_or(0.0),
        );
        return (etrb_ad - etrb_s) / dt;
    }
    if model.controls.mlt_option == MltOption::Tdc
        && model.controls.tdc_include_eturb_in_energy_equation
    {
        // `hydro_energy.f90:360-382`; the start-of-step value is zero until
        // `mlt_vc_old` has been set (`okay_to_set_mlt_vc`).
        let have_old = model.aux.okay_to_set_mlt_vc;
        let vc_old =
            |j: usize| model.aux.mlt_vc_old.get(j).copied().unwrap_or(0.0);
        let vc_00 = model.mlt_vc_ad[k];
        let (eturb_cell, eturb_cell_start) = if k + 1 < model.nz {
            let vc_p1 = model.mlt_vc_ad[k + 1].shift_p1();
            let start = if have_old {
                0.75 * (flexmesa_math::pow2(vc_old(k))
                    + flexmesa_math::pow2(vc_old(k + 1)))
            } else {
                0.0
            };
            (0.75 * (vc_00.pow2() + vc_p1.pow2()), start)
        } else {
            let start = if have_old {
                0.75 * flexmesa_math::pow2(vc_old(k))
            } else {
                0.0
            };
            (vc_00.pow2() * 0.75, start)
        };
        return (eturb_cell - eturb_cell_start) / dt;
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
        let rs = flexmesa_math::exp(lnr_start);
        // `area = 4d0*pi*(r2_00 + r_00*s% r_start(k) + s% r_start(k)**2)/3d0`
        // (`star/private/star_utils.f90:3669`).
        FOUR_PI * (r * r + r * rs + rs * rs) / 3.0
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
///
/// # Errors
/// Returns [`StarError::InvalidInput`] when `total_internal_energy_old` has
/// not been recorded for the step: upstream would read it undefined
/// (DECISIONS #43).
pub fn set_energy_eqn_scal(
    model: &StarModel,
    k: usize,
) -> Result<f64, StarError> {
    // `set_energy_eqn_scal` (`star/private/star_utils.f90:3677-3697`), with
    // `total_internal_energy_old` from the start of the step
    // (`evolve.f90:420-431`) and no clamps.
    let mut scal: f64 = if k > 0 { 1.0 } else { 1.0e-6 };
    if model.controls.dedt_eqn_r_scale > 0.0 {
        let total_internal_energy_old =
            model.aux.total_internal_energy_start.ok_or_else(|| {
                StarError::InvalidInput(String::from(
                    "set_energy_eqn_scal: total_internal_energy_old is not set for this step; upstream reads it undefined (star_utils.f90:3693, evolve.f90:427-436; DECISIONS #43)",
                ))
            })?;
        let cell_energy_fraction_start =
            model.energy_start[k] * model.dm[k] / total_internal_energy_old;
        scal = scal
            .min(cell_energy_fraction_start * model.controls.dedt_eqn_r_scale);
    }
    Ok(scal * model.dt / model.energy_start[k])
}

/// Evaluates `(work_outward_at_surface, work_inward_at_center)` in
/// $\mathrm{erg\,s^{-1}}$ matching `setup_dwork_dm`, `eval1_work`, and
/// `eval_simple_PdV_work` (`star/private/hydro_energy.f90:176-177, 640-655,
/// 769, 872`).
#[must_use]
pub fn eval_surface_and_center_work(model: &StarModel) -> (f64, f64) {
    let nz = model.nz;
    if nz == 0 {
        return (0.0, 0.0);
    }
    let u_flag_only = model.controls.u_flag && !model.controls.v_flag;
    let vel_tc = model
        .controls
        .is_velocity_time_centered(model.model_number_old)
        && model.dt > 0.0
        && !model.ln_r_start.is_empty();
    let simple_pdv = !u_flag_only
        && vel_tc
        && model
            .controls
            .use_p_d_1_div_rho_form_of_work_when_time_centering_velocity;
    // `skip_P = eps_grav_form` (`star/private/hydro_energy.f90:166`).
    let skip_peos = eps_grav_form(model);

    let p_in = model.peos_start.get(nz - 1).copied().unwrap_or(0.0);
    let work_inward_at_center =
        FOUR_PI * flexmesa_math::pow2(model.r_center) * p_in * model.v_center;

    let work_outward_at_surface = if u_flag_only {
        crate::riemann_eqns::eval1_work_u_flag_ad(model, 0).val
    } else if simple_pdv {
        let av_00 =
            crate::hydro_work::eval1_a_times_v_face_v_flag_ad(model, 0, vel_tc)
                .val;
        let (p_tot_ad, _) =
            crate::hydro_work::eval_simple_pdv_p_tot_ad(model, 0, skip_peos);
        p_tot_ad.val * av_00
    } else if model.dt <= 0.0
        && !(model.controls.v_flag || model.controls.u_flag)
    {
        0.0
    } else {
        crate::hydro_work::eval1_work_v_flag_ad(model, 0, skip_peos, vel_tc)
            .0
            .val
    };

    (work_outward_at_surface, work_inward_at_center)
}

/// Energy conservation equation for cell `k` (`do1_energy_eqn` in
/// `star/private/hydro_energy.f90:36-879`).
pub fn do1_energy_eqn(
    model: &StarModel,
    k: usize,
    include_chem: bool,
    sys: &mut HenyeySystem,
) -> Result<(), StarError> {
    do1_energy_eqn_with_ergs_error(model, k, include_chem, sys).map(|_| ())
}

/// Evaluates [`do1_energy_eqn`] and returns the cell unscaled energy residual
/// `ergs_error(k) = -dm(k) * dt * resid_ad%val` in $\mathrm{erg}$
/// (`star/private/hydro_energy.f90:114`).
/// `eps_grav_form` of `setup_eps_grav` (`star/private/hydro_energy.f90:394-403`):
/// never with `u_flag`; otherwise `energy_eqn_option = 'eps_grav'`
/// (`star/private/hydro_eqns.f90:117-123`), or while relaxing with
/// `no_dedt_form_during_relax`.
#[must_use]
pub(crate) fn eps_grav_form(model: &StarModel) -> bool {
    if model.controls.u_flag {
        return false;
    }
    model.controls.energy_eqn_option == EnergyEqnOption::EpsGrav
        || (model.aux.doing_relax && model.controls.no_dedt_form_during_relax)
}

/// `de_dt_ad` of `setup_de_dt_and_friends`
/// (`star/private/hydro_energy.f90:434-441`): value
/// `(energy - energy_start)/dt`, with `lnd` and `lnT` partials
/// `dE_dRho_for_partials*rho/dt` and `Cv_for_partials*T/dt`, where
/// `dE_dRho_for_partials = d_dlnd(i_lnE)*energy/rho` and
/// `Cv_for_partials = d_dlnT(i_lnE)*energy/T` (`star/private/micro.f90:499-500`).
fn setup_de_dt(model: &StarModel, k: usize, dt: f64) -> StarAdOrder1 {
    let e = model.energy[k];
    let de_drho_for_partials = model.eos_d_dlnd[k][I_LN_E] * e / model.rho[k];
    let cv_for_partials = model.eos_d_dlnt[k][I_LN_E] * e / model.t[k];
    let mut de_dt = StarAdOrder1::constant((e - model.energy_start[k]) / dt);
    de_dt.d1[I_LND_00] = de_drho_for_partials * model.rho[k] / dt;
    de_dt.d1[I_LNT_00] = cv_for_partials * model.t[k] / dt;
    de_dt
}

pub(crate) fn do1_energy_eqn_with_ergs_error(
    model: &StarModel,
    k: usize,
    include_chem: bool,
    sys: &mut HenyeySystem,
) -> Result<f64, StarError> {
    let dl_dm = setup_dl_dm(model, k);
    let (sources, others, doing_op_split_burn) =
        setup_sources_and_others(model, k)?;
    // Every rate in the energy equation divides by `s% dt`
    // (`dt = s% dt`, `star/private/hydro_energy.f90:156`); a solve always has `dt > 0`.
    // Without `v_flag` or `u_flag`, `eval1_work` returns early for
    // `dt <= 0` with `work` unset for `k <= nz`
    // (`star/private/hydro_energy.f90:640-655`), so that path has no defined
    // value either and stops here.
    // `dm = s% dm(k)` (`star/private/hydro_energy.f90:157`); `set_m_and_dm`
    // flags `dm <= 0` as bad (`star/private/star_utils.f90:642-646`).
    let dm_k = model.dm[k];
    if dm_k.is_nan() || dm_k <= 0.0 {
        return Err(StarError::InvalidInput(format!(
            "energy equation at cell {k}: dm = {dm_k} is not positive (star/private/star_utils.f90:642-646)"
        )));
    }
    let dt = model.dt;
    if dt.is_nan() || dt <= 0.0 {
        return Err(StarError::InvalidInput(format!(
            "energy equation at cell {k}: dt = {dt} is not positive (star/private/hydro_energy.f90:156,640-655)"
        )));
    }
    // `setup_eps_grav` stops for `RSP2_flag` in the `eps_grav` form
    // (`star/private/hydro_energy.f90:407-409`).
    if model.controls.rsp2_flag && eps_grav_form(model) {
        return Err(StarError::InvalidInput(format!(
            "energy equation at cell {k}: cannot use eps_grav with RSP2 yet (star/private/hydro_energy.f90:407-409)"
        )));
    }
    let d_eturb_dt = setup_d_turbulent_energy_dt(model, k);

    let v_flag_only = model.controls.v_flag && !model.controls.u_flag;
    let u_flag_only = model.controls.u_flag && !model.controls.v_flag;
    let vel_tc = model
        .controls
        .is_velocity_time_centered(model.model_number_old)
        && model.dt > 0.0
        && !model.ln_r_start.is_empty();

    // `d_dwork_dxam1`, `d_dwork_dxa00`, `d_dwork_dxap1` (zero unless a work
    // term with `Peos` is evaluated; `hydro_energy.f90:163-190`).
    let mut d_dwork = crate::hydro_work::DworkDxa {
        dxam1: Vec::new(),
        dxa00: Vec::new(),
        dxap1: Vec::new(),
    };
    let mut has_dedt_dxa = false;

    let esum = if u_flag_only {
        has_dedt_dxa = true;
        if vel_tc
            && model
                .controls
                .use_p_d_1_div_rho_form_of_work_when_time_centering_velocity
        {
            // Upstream switches to `eval_simple_PdV_work` here
            // (`hydro_energy.f90:170-175`), which is not ported for the
            // Riemann face velocity `u_face_ad`.
            return Err(StarError::InvalidInput(format!(
                "energy equation at cell {k}: use_P_d_1_div_rho_form_of_work_when_time_centering_velocity with u_flag (star/private/hydro_energy.f90:170-175)"
            )));
        }
        // `esum_ad` terms in upstream order (`hydro_energy.f90:107-108`):
        // `dwork_dm` from `eval_dwork` divided by `dm`, then `dke_dt`,
        // `dpe_dt` and `de_dt` (`setup_de_dt_and_friends`).
        let work_00 = crate::riemann_eqns::eval1_work_u_flag_ad(model, k);
        let work_p1 = if k + 1 < model.nz {
            crate::riemann_eqns::eval1_work_u_flag_ad(model, k + 1).shift_p1()
        } else {
            crate::riemann_eqns::eval1_work_u_flag_ad(model, model.nz)
        };
        let dwork_dm = (work_00 - work_p1) / dm_k;
        let de_dt = setup_de_dt(model, k, dt);
        let dke_dt = crate::hydro_work::eval_dke_dt_ad(model, k, dt);
        let dpe_dt = crate::hydro_work::eval_cell_dpe_dt_ad(model, k, dt);
        -dl_dm + sources + others
            - d_eturb_dt
            - dwork_dm
            - dke_dt
            - dpe_dt
            - de_dt
    } else if v_flag_only || eps_grav_form(model) {
        // The hydrostatic `eps_grav` form reaches here too: upstream always
        // subtracts `dwork_dm` evaluated with `skip_P = eps_grav_form`
        // (`star/private/hydro_energy.f90:102-104,166-192`), which keeps
        // the non-`Peos` work terms (`Ptrb`, `mlt_Pturb`, `extra_P`) and the
        // surface and centre work.
        let simple_pdv = vel_tc
            && model
                .controls
                .use_p_d_1_div_rho_form_of_work_when_time_centering_velocity;
        if eps_grav_form(model) {
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
            -dl_dm + sources + others - d_eturb_dt - dwork_dm
                + model.eps_grav_ad[k]
        } else {
            has_dedt_dxa = true;
            let de_dt = setup_de_dt(model, k, dt);
            if simple_pdv {
                let (dwork_dm, dxa00) =
                    crate::hydro_work::eval_v_flag_simple_pdv_dm_ad(
                        model, k, false, vel_tc,
                    );
                d_dwork.dxa00 = dxa00;
                -dl_dm + sources + others - d_eturb_dt - dwork_dm - de_dt
            } else {
                let (dwork_dm, dxa) =
                    crate::hydro_work::eval_v_flag_dwork_dm_ad(
                        model, k, false, vel_tc,
                    );
                d_dwork = dxa;
                let dke_dt =
                    crate::hydro_work::eval_v_flag_dke_dt_ad(model, k, dt);
                let dpe_dt =
                    crate::hydro_work::eval_cell_dpe_dt_ad(model, k, dt);
                -dl_dm + sources + others
                    - d_eturb_dt
                    - dwork_dm
                    - dke_dt
                    - dpe_dt
                    - de_dt
            }
        }
    } else {
        has_dedt_dxa = true;
        let de_dt = setup_de_dt(model, k, dt);
        let simple_pdv = vel_tc
            && model
                .controls
                .use_p_d_1_div_rho_form_of_work_when_time_centering_velocity;
        if simple_pdv {
            let (dwork_dm, dxa00) =
                crate::hydro_work::eval_v_flag_simple_pdv_dm_ad(
                    model, k, false, vel_tc,
                );
            d_dwork.dxa00 = dxa00;
            -dl_dm + sources + others - d_eturb_dt - dwork_dm - de_dt
        } else {
            let (dwork_dm, dxa) = crate::hydro_work::eval_v_flag_dwork_dm_ad(
                model, k, false, vel_tc,
            );
            d_dwork = dxa;
            let dpe_dt = crate::hydro_work::eval_cell_dpe_dt_ad(model, k, dt);
            -dl_dm + sources + others - d_eturb_dt - dwork_dm - dpe_dt - de_dt
        }
    };

    let ergs_error_k = -model.dm[k] * model.dt * esum.val;
    let scal = set_energy_eqn_scal(model, k)?;
    let resid = esum * scal;
    let i_equ = model.layout.i_dlne_dt;
    sys.save_eqn_residual(&model.layout, k, i_equ, &resid);

    if include_chem {
        let sp = model.species;
        let base = k * sp;
        let dm = model.dm[k];
        // `unpack_res18` (`star/private/hydro_energy.f90:469-527`): each
        // species partial is accumulated term by term into `dxa00`,
        // `dxam1` and `dxap1`, starting from zero.
        let active_nuc_dxa = !(model.controls.nonlocal_nico_decay_heat
            || doing_op_split_burn)
            && model.controls.dxdt_nuc_factor > 0.0;
        let dw = |v: &Vec<f64>, j: usize| v.get(j).copied().unwrap_or(0.0);
        for j in 0..sp {
            let ivar = model.layout.chem_var_index(j);
            let mut dxa00 = 0.0_f64;
            if active_nuc_dxa {
                dxa00 += scal * model.d_epsnuc_dx[base + j];
            }
            if has_dedt_dxa {
                // `-scal*(s%energy(k)/dt)*s% dlnE_dxa_for_partials(j,k)`
                // (`hydro_energy.f90:491-495`).
                let dlne_dx = model
                    .aux
                    .dln_e_dxa_for_partials
                    .get(base + j)
                    .copied()
                    .unwrap_or(0.0);
                dxa00 += -(scal * (model.energy[k] / dt) * dlne_dx);
            } else if !doing_op_split_burn
                && (model.controls.dxdt_nuc_factor > 0.0
                    || model.controls.mix_factor > 0.0)
            {
                // `star/private/hydro_energy.f90:497-503`.
                let deg_dx = model
                    .aux
                    .d_eps_grav_dx
                    .get(base + j)
                    .copied()
                    .unwrap_or(0.0);
                dxa00 += scal * deg_dx;
            }
            // `-scal*d_dwork_dxa00(j)/dm` (`hydro_energy.f90:506-510`).
            dxa00 += -(scal * dw(&d_dwork.dxa00, j) / dm);
            sys.add_partial(k, i_equ, k, ivar, dxa00);
            if k > 0 {
                // `hydro_energy.f90:511-517`.
                let dxam1 = -(scal * dw(&d_dwork.dxam1, j) / dm);
                sys.add_partial(k, i_equ, k - 1, ivar, dxam1);
            }
            if k + 1 < model.nz {
                // `hydro_energy.f90:518-524`.
                let dxap1 = -(scal * dw(&d_dwork.dxap1, j) / dm);
                sys.add_partial(k, i_equ, k + 1, ivar, dxap1);
            }
        }
    }
    Ok(ergs_error_k)
}
