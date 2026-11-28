// SPDX-License-Identifier: GPL-3.0-or-later
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
//!   - `extra_grav[k]` extra face gravitational acceleration when
//!     `use_other_momentum || use_other_momentum_implicit`,
//!   - Lagrangian acceleration $-\mathrm{dxh\_v}_k/\Delta t$ (or
//!     $-2\,v_k/\Delta t$ below `velocity_logT_lower_bound`),
//!   - Velocity drag $-c_{\mathrm{drag}} v_k / \Delta t$ for $q_k > q_{\mathrm{drag,min}}$,
//!   - Turbulent viscosity $U_{q,k}$ (`uq_ad[k]`),
//!   - Artificial viscosity gradient $-(A_k / \Delta m_{\mathrm{face},k})(P_{\mathrm{vsc},k-1} - P_{\mathrm{vsc},k})$,
//!   - MLT turbulent pressure difference $-(A_k / \Delta m_{\mathrm{face},k})(P_{\mathrm{turb},k-1} - P_{\mathrm{turb},k})$,
//!   - Rayleigh-Taylor instability (`RTI`) momentum diffusion and source kick
//!     (`setup_RTI_terms`, `hydro_momentum.f90:250-284`).
//! - **Time-centered pressure $\theta_P$ and composition partials
//!   `dln_peos_dxa_for_partials` (`hydro_momentum.f90:312-343`)**.
//! - **Radius equation (`do1_radius_eqn`, `star/private/hydro_momentum.f90:545-601`)** with
//!   `force_zero_v` bounds (`velocity_q_upper_bound`, `velocity_tau_lower_bound`,
//!   `velocity_logt_lower_bound`, `max_dt_yrs_for_velocity_logt_lower_bound`) and
//!   optional velocity time-centering.

use flexmesa_const::{FOUR_PI, LN10, SECYER};

use crate::controls::StarError;
use crate::hydro_energy::{eval_face_area_time_centered_ad, eval_pvsc_cell_ad};
use crate::hydro_eqns::HenyeySystem;
use crate::model::StarModel;
use crate::star_ad::{
    StarAdExt, StarAdOrder1, I_LND_00, I_LND_M1, I_LNR_00, I_LNT_00, I_V_00,
    I_V_M1, I_V_P1,
};

/// Evaluates the pressure time-centering weight $\theta_P \in [0, 1]$ at cell
/// or face `k` (`hydro_momentum.f90:151-162`).
#[must_use]
#[inline]
pub fn eval_p_theta(model: &StarModel, k: usize) -> f64 {
    let logt = model.ln_t[k] / LN10;
    if model.controls.using_velocity_time_centering
        && model.controls.include_p_in_velocity_time_centering
        && logt
            <= model
                .controls
                .max_logt_for_include_p_and_l_in_velocity_time_centering
    {
        model.controls.p_theta_for_velocity_time_centering
    } else {
        1.0
    }
}

/// Evaluates cell MLT turbulent pressure
/// $P_{\mathrm{turb,MLT},k} = \tfrac{1}{3}\,\mathrm{mlt\_pturb\_factor}\,\rho_k v_{\mathrm{conv,old},k}^2$
/// (`star/private/star_utils.f90`).
#[must_use]
#[inline]
pub fn eval_mlt_pturb_cell_ad(
    model: &StarModel,
    k: usize,
    shift_m1: bool,
) -> StarAdOrder1 {
    if model.controls.mlt_pturb_factor <= 0.0 {
        return StarAdOrder1::ZERO;
    }
    let vc_old = model.aux.mlt_vc_old[k];
    if vc_old <= 0.0 {
        return StarAdOrder1::ZERO;
    }
    let rho_slot = if shift_m1 { I_LND_M1 } else { I_LND_00 };
    let rho_ad = StarAdOrder1::variable(model.lnd[k], rho_slot).exp();
    // `star/private/star_utils.f90:3426`:
    // `s% mlt_Pturb_factor * pow2(s% mlt_vc_old(k))`
    // `* get_rho_face(s, k) / 3d0`.
    (model.controls.mlt_pturb_factor * flexmesa_math::pow2(vc_old) * rho_ad)
        / 3.0
}

/// Evaluates time-centered face geometry `(r2, area)` at face `k` for HSE
/// momentum and temperature-gradient equations (`hydro_momentum.f90:167-182`
/// and `hydro_temperature.f90:157-194`).
#[must_use]
pub fn eval_time_centered_hse_geom_ad(
    model: &StarModel,
    k: usize,
) -> (StarAdOrder1, StarAdOrder1) {
    let lnr_k = StarAdOrder1::variable(model.ln_r[k], I_LNR_00);
    let r_k = lnr_k.exp();
    let using_v_tc = model.controls.using_velocity_time_centering
        && (model.controls.v_flag || model.controls.u_flag)
        && model.dt > 0.0;
    let lnr_start_k = model.ln_r_start[k];
    let r2 = if using_v_tc {
        let r_start = flexmesa_math::exp(lnr_start_k).max(1.0);
        (r_k * r_start).max(StarAdOrder1::ONE)
    } else {
        (r_k * r_k).max(StarAdOrder1::ONE)
    };
    let area = eval_face_area_time_centered_ad(lnr_k, lnr_start_k, using_v_tc)
        .max(StarAdOrder1::ONE);
    (r2, area)
}

/// Returns `true` when an explicit outer-boundary momentum equation is active
/// at `k = 0` (`use_momentum_outer_bc`, `use_zero_pgas_outer_bc`, or
/// `use_fixed_psurf_outer_bc`).
#[must_use]
#[inline]
pub const fn has_surf_momentum_bc(c: &crate::controls::StarControls) -> bool {
    c.use_momentum_outer_bc
        || c.use_zero_pgas_outer_bc
        || c.use_fixed_psurf_outer_bc
}

/// Evaluates the outer-boundary surface pressure $P_{\mathrm{surf}}$ for
/// momentum outer boundary conditions (`use_zero_pgas_outer_bc`,
/// `use_fixed_psurf_outer_bc`, or `use_momentum_outer_bc`).
#[must_use]
pub fn eval_surf_momentum_bc_pressure_ad(model: &StarModel) -> StarAdOrder1 {
    if model.controls.use_zero_pgas_outer_bc {
        let t_s0 =
            flexmesa_math::exp(
                model.ln_t_start.first().copied().unwrap_or_else(|| {
                    model.ln_t.first().copied().unwrap_or(0.0)
                }),
            );
        // `P_bc_ad = Radiation_Pressure(s% T_start(1))`
        // (`star/private/hydro_eqns.f90:761`, `eos/public/eos_lib.f90:91`:
        // `crad*T*T*T*T/3d0`).
        StarAdOrder1::constant(
            flexmesa_const::CRAD * t_s0 * t_s0 * t_s0 * t_s0 / 3.0,
        )
    } else if model.controls.use_fixed_psurf_outer_bc {
        StarAdOrder1::constant(model.controls.fixed_psurf)
    } else {
        // `P_bc_ad` from `get_PT_bc_ad` for `use_momentum_outer_BC`
        // (`star/private/hydro_eqns.f90:788-798`).
        crate::hydro_eqns::get_pt_bc_ad(model).p_bc
    }
}

/// `wrap_Peos_00` (`star/private/auto_diff_support.f90:562-570`): value
/// `Peos(k)` with partials `Peos*chiRho_for_partials` and
/// `Peos*chiT_for_partials` (`star/private/micro.f90:497-498`), which
/// `ln_peos_ad` carries as its `lnd_00` and `lnT_00` partials.
#[must_use]
#[inline]
pub fn wrap_peos_00(model: &StarModel, k: usize) -> StarAdOrder1 {
    let p = model.peos[k];
    let lp = &model.ln_peos_ad[k];
    StarAdOrder1::from_dlnd_dlnt_00(p, p * lp.d1[I_LND_00], p * lp.d1[I_LNT_00])
}

/// `wrap_Peos_m1` (`star/private/auto_diff_support.f90:550-560`) for face
/// `k >= 1`: cell `k - 1` in the `m1` slots.
#[must_use]
#[inline]
pub fn wrap_peos_m1(model: &StarModel, k: usize) -> StarAdOrder1 {
    let p = model.peos[k - 1];
    let lp = &model.ln_peos_ad[k - 1];
    StarAdOrder1::from_dlnd_dlnt_m1(p, p * lp.d1[I_LND_00], p * lp.d1[I_LNT_00])
}

/// `wrap_r_00` (`star/private/auto_diff_support.f90:923-930`): value `r(k)`
/// with partial `r(k)` on `lnR(k)`.
#[must_use]
#[inline]
pub fn wrap_r_00(model: &StarModel, k: usize) -> StarAdOrder1 {
    let r = model.r[k];
    let mut out = StarAdOrder1::constant(r);
    out.d1[I_LNR_00] = r;
    out
}

/// `get_area_info_opt_time_center` (`star/private/star_utils.f90:3658-3675`):
/// face area and `1/r^2`, time-centered with `r_start(k)` when
/// `using_velocity_time_centering`.
#[must_use]
pub fn get_area_info_opt_time_center(
    model: &StarModel,
    k: usize,
) -> (StarAdOrder1, StarAdOrder1) {
    let r_00 = if model.r[k] > 0.0 {
        wrap_r_00(model, k)
    } else {
        StarAdOrder1::variable(model.ln_r[k], I_LNR_00).exp()
    };
    let r2_00 = r_00.pow2();
    if model.controls.using_velocity_time_centering {
        // `r_start(k)` is saved from `r(k) = exp(lnR(k))` at the start of the
        // step; `ln_r_start` holds that `lnR(k)`.
        let r_start = flexmesa_math::exp(model.ln_r_start[k]);
        let area = FOUR_PI * (r2_00 + r_00 * r_start + r_start * r_start) / 3.0;
        let inv_r2 = 1.0 / (r_00 * r_start);
        (area, inv_r2)
    } else {
        let area = FOUR_PI * r2_00;
        let inv_r2 = 1.0 / r2_00;
        (area, inv_r2)
    }
}

/// `expected_HSE_grav_term` (`star/private/hydro_momentum.f90:327-357`):
/// returns `(grav, area)` with `grav = -cgrav*m_grav*inv_R2[*fp_rot]`.
#[must_use]
pub fn expected_hse_grav_and_area(
    model: &StarModel,
    k: usize,
) -> (StarAdOrder1, StarAdOrder1) {
    let (area, inv_r2) = get_area_info_opt_time_center(model, k);
    let grav = if model.controls.rotation_flag
        && model.controls.use_gravity_rotation_correction
    {
        let fp = model.aux.fp_rot.get(k).copied().unwrap_or(1.0);
        -(model.cgrav[k] * model.m_grav[k] * inv_r2 * fp)
    } else {
        -(model.cgrav[k] * model.m_grav[k] * inv_r2)
    };
    (grav, area)
}

/// Evaluates total EOS pressure $P_{\mathrm{eos}}$ in cell `k` with optional
/// $\theta_P$ velocity time-centering (`hydro_momentum.f90:151-162`).
#[must_use]
pub fn eval_cell_peos_time_centered_ad(
    model: &StarModel,
    k: usize,
    shift_m1: bool,
) -> (StarAdOrder1, f64) {
    let p_theta = eval_p_theta(model, k);
    // `Peos_ad = wrap_peos_00(s, k)`; `Peos_ad = alfa*Peos_ad +
    // beta*s% Peos_start(k)` (`star/private/star_utils.f90:3400-3403`).
    let peos_ad = if shift_m1 {
        wrap_peos_m1(model, k + 1)
    } else {
        wrap_peos_00(model, k)
    };
    let peos_start = model.peos_start.get(k).copied().unwrap_or(model.peos[k]);
    (p_theta * peos_ad + (1.0 - p_theta) * peos_start, p_theta)
}

/// Evaluates face control-volume mass $\Delta m_{\mathrm{face},k}$ including
/// optional baryonic `mass_correction` (`hydro_momentum.f90:187-199`).
#[must_use]
pub fn eval_face_dm_bar(model: &StarModel, k: usize) -> f64 {
    if !model.controls.use_mass_corrections {
        return if k == 0 {
            0.5 * model.dm[0]
        } else {
            0.5 * (model.dm[k - 1] + model.dm[k])
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

/// `setup_RTI_terms` (`star/private/hydro_momentum.f90:247-281`): `RTI`
/// momentum diffusion and density-kick terms at face `k`, in $\mathrm{cm\,s^{-2}}$.
#[must_use]
fn setup_rti_momentum_terms(
    model: &StarModel,
    k: usize,
    dm_div_a: StarAdOrder1,
) -> StarAdOrder1 {
    // `hydro_momentum.f90:255-256`: runs whenever `RTI_flag` and interior face.
    if !model.controls.rti_flag || k == 0 || k + 1 >= model.nz {
        return StarAdOrder1::ZERO;
    }
    let dvdt_diffusion = if model.controls.dudt_rti_diffusion_factor > 0.0 {
        let fac = model.controls.dudt_rti_diffusion_factor;
        let sigm1 = fac * model.aux.sigmid_rti[k - 1];
        let sig00 = fac * model.aux.sigmid_rti[k];
        // `wrap_v_{m1,00,p1}` (`star/private/auto_diff_support.f90:1052-1119`):
        // delegates to `wrap_u_face_{m1,00,p1}` when `u_flag` is set.
        let (v_m1, v_00, v_p1) = if model.controls.u_flag {
            (
                model.aux.u_face_ad[k - 1].shift_m1(),
                model.aux.u_face_ad[k],
                model.aux.u_face_ad[k + 1].shift_p1(),
            )
        } else {
            (
                StarAdOrder1::variable(model.v[k - 1], I_V_M1),
                StarAdOrder1::variable(model.v[k], I_V_00),
                StarAdOrder1::variable(model.v[k + 1], I_V_P1),
            )
        };
        (sig00 * (v_p1 - v_00) - sigm1 * (v_00 - v_m1)) / model.dm_bar[k]
    } else {
        StarAdOrder1::ZERO
    };
    let eta_k = model.aux.eta_rti[k];
    let dvdt_kick = if eta_k > 0.0
        && model.controls.dlnddt_rti_diffusion_factor > 0.0
        && model.dt > 0.0
    {
        let f = model.controls.dlnddt_rti_diffusion_factor * eta_k / dm_div_a;
        let rho_00 = wrap_d_00(model, k);
        let rho_m1 = wrap_d_m1(model, k);
        f * (rho_00 - rho_m1) / model.dt
    } else {
        StarAdOrder1::ZERO
    };
    dvdt_diffusion + dvdt_kick
}

/// `wrap_d_00` (`star/private/auto_diff_support.f90:268-276`).
#[must_use]
#[inline]
pub fn wrap_d_00(model: &StarModel, k: usize) -> StarAdOrder1 {
    StarAdOrder1::from_dlnd_dlnt_00(model.rho[k], model.rho[k], 0.0)
}

/// `wrap_d_m1` (`star/private/auto_diff_support.f90:257-266`): zero at the
/// surface face.
#[must_use]
#[inline]
pub fn wrap_d_m1(model: &StarModel, k: usize) -> StarAdOrder1 {
    if k == 0 {
        return StarAdOrder1::ZERO;
    }
    StarAdOrder1::from_dlnd_dlnt_m1(model.rho[k - 1], model.rho[k - 1], 0.0)
}

/// `other_ad = extra_ad - accel_ad + drag + Uq_ad` at face `k`
/// (`expected_non_HSE_term` in `star/private/hydro_momentum.f90:363-433`).
/// Artificial viscosity and `RSP2` turbulent pressure enter `Ptot`
/// ([`calc_ptot_ad_tw`]); MLT turbulent pressure and the `RTI` terms are
/// separate terms of the residual (`hydro_momentum.f90:110-111`).
#[must_use]
pub fn expected_non_hse_term(model: &StarModel, k: usize) -> StarAdOrder1 {
    // Upstream reads `extra_grav` only when `use_other_momentum` or
    // `use_other_momentum_implicit` is set (`hydro_momentum.f90:392-395`).
    let mut non_hse = if model.controls.use_other_momentum
        || model.controls.use_other_momentum_implicit
    {
        model.aux.extra_grav[k]
    } else {
        StarAdOrder1::ZERO
    };

    let dt = model.dt;
    // Upstream applies Lagrangian acceleration, drag, and `Uq_ad` only when
    // `v_flag` is active (`hydro_momentum.f90:401-438`).
    if model.controls.v_flag && dt > 0.0 {
        let ln_t_old_k =
            model.ln_t_old.get(k).copied().unwrap_or(model.ln_t[k]);
        let local_v_flag =
            ln_t_old_k / LN10 >= model.controls.velocity_logt_lower_bound;
        let (accel, d_accel_dv) = if local_v_flag {
            // `hydro_momentum.f90:410-412`.
            (model.aux.dxh_v[k] / dt, 1.0 / dt)
        } else {
            // `hydro_momentum.f90:413-420`: assume `v_start = 0` and
            // constant acceleration over `dt`.
            (2.0 * model.v[k] / dt, 2.0 / dt)
        };
        let mut accel_ad = StarAdOrder1::ZERO;
        accel_ad.val = accel;
        accel_ad.d1[I_V_00] = d_accel_dv;
        non_hse = non_hse - accel_ad;

        if model.q[k] > model.controls.min_q_for_drag
            && model.controls.drag_coefficient > 0.0
        {
            // `drag = -s% drag_coefficient*v_00/s% dt` (`hydro_momentum.f90:424-428`).
            let v_00 = StarAdOrder1::variable(model.v[k], I_V_00);
            non_hse = non_hse - model.controls.drag_coefficient * v_00 / dt;
        }

        if model.controls.rsp2_flag {
            if let Some(&uq) = model.aux.uq_ad.get(k) {
                if uq.val != 0.0 || uq.d1.iter().any(|&d| d != 0.0) {
                    non_hse = non_hse + uq;
                }
            }
        } else if model.controls.tdc_alpha_m > 0.0
            && model.controls.mlt_option == flexmesa_turb::MltOption::Tdc
        {
            // `Uq_ad = compute_tdc_Uq_face(s, k, ierr)`
            // (`star/private/hydro_momentum.f90:433-436`).
            non_hse = non_hse + crate::tdc_hydro::compute_tdc_uq_face(model, k);
        }
    }

    non_hse
}

/// `calc_Ptot_ad_tw` with `skip_mlt_Pturb = .true.`
/// (`star/private/star_utils.f90:3375-3438`, called from
/// `get_dPtot_face_info`, `hydro_momentum.f90:470-478`) for cell `cell`,
/// placed in the `m1` slots when `shift_m1`.
#[must_use]
pub fn calc_ptot_ad_tw(
    model: &StarModel,
    cell: usize,
    shift_m1: bool,
) -> StarAdOrder1 {
    let (peos_ad, _) = eval_cell_peos_time_centered_ad(model, cell, shift_m1);
    let mut pvsc_ad = StarAdOrder1::ZERO;
    if model.controls.use_pvsc_art_visc && model.controls.v_flag {
        pvsc_ad = eval_pvsc_cell_ad(model, cell);
        if shift_m1 {
            pvsc_ad = pvsc_ad.shift_m1();
        }
    }
    let mut ptrb_ad = StarAdOrder1::ZERO;
    if model.controls.rsp2_flag {
        ptrb_ad = crate::rsp2_ad::eval_rsp2_ptrb_cell_ad(model, cell, shift_m1);
    }
    peos_ad + pvsc_ad + ptrb_ad
}

/// Hydrostatic / hydrodynamic momentum equation at face `k` (`do1_momentum_eqn`
/// in `star/private/hydro_momentum.f90:79-436`).
pub fn do1_momentum_eqn(model: &StarModel, k: usize, sys: &mut HenyeySystem) {
    if k == 0 && !has_surf_momentum_bc(&model.controls) {
        return;
    }
    // `get1_momentum_eqn` (`star/private/hydro_momentum.f90:62-118`).
    let dm_face = eval_face_dm_bar(model, k);
    // `setup_HSE`
    let (grav_ad, area_ad) = expected_hse_grav_and_area(model, k);
    let dm_div_a_ad = dm_face / area_ad;
    // `setup_non_HSE`
    let other_ad = expected_non_hse_term(model, k);
    // `setup_dPtot` (`get_dPtot_face_info`, `hydro_momentum.f90:446-526`)
    let ptot00_ad = calc_ptot_ad_tw(model, k, false);
    let ptotm1_ad = if k == 0 {
        eval_surf_momentum_bc_pressure_ad(model)
    } else {
        calc_ptot_ad_tw(model, k - 1, true)
    };
    let dptot_ad = ptotm1_ad - ptot00_ad;
    let (alfa, beta) = if k == 0 {
        (1.0, 0.0)
    } else {
        model.get_face_weights(k)
    };
    let ptotavg_ad = if k == 0 {
        ptot00_ad
    } else {
        alfa * ptot00_ad + beta * ptotm1_ad
    };
    let iptotavg_ad = 1.0 / ptotavg_ad;
    // `setup_d_mlt_Pturb`
    let vc_old = model.aux.mlt_vc_old[k];
    let d_mlt_pturb_ad =
        if model.controls.mlt_pturb_factor > 0.0 && vc_old > 0.0 {
            let rho_00 = wrap_d_00(model, k);
            let rho_m1 = wrap_d_m1(model, k);
            model.controls.mlt_pturb_factor
                * flexmesa_math::pow2(vc_old)
                * (rho_m1 - rho_00)
                / 3.0
        } else {
            StarAdOrder1::ZERO
        };
    // `setup_RTI_terms`
    let rti_terms_ad = setup_rti_momentum_terms(model, k, dm_div_a_ad);

    let other_dm_div_a_ad = other_ad * dm_div_a_ad;
    let grav_dm_div_a_ad = grav_ad * dm_div_a_ad;
    let rti_terms_dm_div_a_ad = rti_terms_ad * dm_div_a_ad;
    // `residual_sum_ad = other_dm_div_A_ad + grav_dm_div_A_ad - dPtot_ad -
    // d_mlt_Pturb_ad + RTI_terms_dm_div_A_ad` is evaluated in
    // `auto_diff_real_star_order1` and only then assigned to the accurate
    // type, whose compensator is then zero (`set_acc_adr`,
    // `num/public/accurate_sum_auto_diff_star_order1.f90`).
    let resid1_ad =
        other_dm_div_a_ad + grav_dm_div_a_ad - dptot_ad - d_mlt_pturb_ad
            + rti_terms_dm_div_a_ad;
    let resid = resid1_ad * iptotavg_ad;

    let i_equ = model.layout.i_dv_dt;
    sys.save_eqn_residual(&model.layout, k, i_equ, &resid);

    // Composition partials: `get_dPtot_face_info` (`hydro_momentum.f90:470-526`,
    // `d_Ptot_dxa(j) = alfa*Peos(k)*dlnPeos_dxa_for_partials(j,k)` from
    // `calc_Ptot_ad_tw`, `star_utils.f90:3400-3406`) and `unpack_res18`
    // (`hydro_momentum.f90:126-152`).
    if sys.nvar > model.layout.nvar_hydro && model.species > 0 {
        let sp = model.species;
        let resid1 = resid1_ad.val;
        let ptotavg = ptotavg_ad.val;
        let iptotavg = 1.0 / ptotavg;
        let d_ptot_dxa = |cell: usize, j: usize| {
            let d = model.peos[cell]
                * model
                    .aux
                    .dln_peos_dxa_for_partials
                    .get(cell * sp + j)
                    .copied()
                    .unwrap_or(0.0);
            eval_p_theta(model, cell) * d
        };
        for j in 0..sp {
            let ivar = model.layout.chem_var_index(j);
            let d_ptot00_dxa00 = d_ptot_dxa(k, j);
            let d_dptot_dxa00 = -d_ptot00_dxa00;
            let d_ptotavg_dxa00 = if k == 0 {
                d_ptot00_dxa00
            } else {
                alfa * d_ptot00_dxa00
            };
            let d_iptotavg_dxa00 = -iptotavg * d_ptotavg_dxa00 / ptotavg;
            let d_residual_dxa00 =
                resid1 * d_iptotavg_dxa00 - iptotavg * d_dptot_dxa00;
            sys.add_partial(k, i_equ, k, ivar, d_residual_dxa00);
            if k > 0 {
                let d_ptotm1_dxam1 = d_ptot_dxa(k - 1, j);
                let d_dptot_dxam1 = d_ptotm1_dxam1;
                let d_ptotavg_dxam1 = beta * d_ptotm1_dxam1;
                let d_iptotavg_dxam1 = -iptotavg * d_ptotavg_dxam1 / ptotavg;
                let d_residual_dxam1 =
                    resid1 * d_iptotavg_dxam1 - iptotavg * d_dptot_dxam1;
                sys.add_partial(k, i_equ, k - 1, ivar, d_residual_dxam1);
            }
        }
    }
}

/// Optional kinematic radius-velocity equation for face `k` (`do1_radius_eqn`
/// in `star/private/hydro_momentum.f90:545-601`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if neither `u_flag` nor `v_flag` is
/// set (`star/private/hydro_momentum.f90:559`) or if `layout.i_dlnr_dt` is
/// absent.
pub fn do1_radius_eqn(
    model: &StarModel,
    k: usize,
    sys: &mut HenyeySystem,
) -> Result<(), StarError> {
    if !(model.controls.u_flag || model.controls.v_flag) {
        return Err(StarError::InvalidInput(String::from(
            "must have either v or u for do1_radius_eqn",
        )));
    }
    let i_dlnr_dt = model.layout.i_dlnr_dt.ok_or_else(|| {
        StarError::InvalidInput(String::from(
            "do1_radius_eqn requires layout.i_dlnr_dt",
        ))
    })?;
    let logt = model.ln_t_start[k] / LN10;
    // `star/private/hydro_momentum.f90:561-564`.
    let force_zero_v = model.q[k] > model.controls.velocity_q_upper_bound
        || model.tau[k] < model.controls.velocity_tau_lower_bound
        || (logt < model.controls.velocity_logt_lower_bound
            && model.dt
                < SECYER
                    * model.controls.max_dt_yrs_for_velocity_logt_lower_bound);

    let resid = if force_zero_v {
        // `resid_ad = v00/s% csound_start(k)` (`hydro_momentum.f90:565-574`),
        // checking `u_flag` before `v_flag`.
        let vel_curr = if model.controls.u_flag {
            model.u[k]
        } else {
            model.v[k]
        };
        let v00 = StarAdOrder1::variable(vel_curr, I_V_00);
        v00 / model.aux.csound_start[k]
    } else {
        let vel_tc = model.controls.using_velocity_time_centering;
        // `wrap_opt_time_center_v_00` (`star/private/auto_diff_support.f90:1090-1101,1171-1185`).
        let v_ad = if model.controls.v_flag {
            let mut v_tc = if model.controls.u_flag {
                model.aux.u_face_ad[k]
            } else {
                StarAdOrder1::variable(model.v[k], I_V_00)
            };
            if vel_tc {
                v_tc = 0.5 * (v_tc + model.v_start[k]);
            }
            v_tc
        } else {
            let mut v_tc = model.aux.u_face_ad[k];
            if vel_tc {
                v_tc = 0.5 * (v_tc + model.aux.u_face_start[k]);
            }
            v_tc
        };
        // `dxh_lnR = wrap_dxh_lnR(s,k)` carries the Newton change of `lnR`
        // before `edit_lnR` (`auto_diff_support.f90:977-984`), and
        // `dr_div_r0_actual = expm1(dxh_lnR)`
        // (`star/private/hydro_momentum.f90:582-583`).
        let dxh_lnr = StarAdOrder1::variable(model.aux.dxh_lnr[k], I_LNR_00);
        let dr_div_r0_actual = dxh_lnr.expm1();
        // `dr_div_r0_expected = v00*s% dt/s% r_start(k)` (`:585-586`), with
        // `r_start = exp(lnR_start)` as `set_rv_info` sets `r`.
        let r_start = flexmesa_math::exp(model.ln_r_start[k]);
        let dr_div_r0_expected = v_ad * model.dt / r_start;
        dr_div_r0_expected - dr_div_r0_actual
    };
    sys.save_eqn_residual(&model.layout, k, i_dlnr_dt, &resid);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use flexmesa_const::LN10;

    use super::{
        do1_radius_eqn, expected_non_hse_term, setup_rti_momentum_terms,
        wrap_d_00, wrap_d_m1, StarAdExt, StarAdOrder1, I_LND_00, I_LND_M1,
        I_LNR_00, I_V_00,
    };
    use crate::controls::{StarControls, StarError, StarJobControls};
    use crate::hydro_eqns::HenyeySystem;
    use crate::hydro_vars::set_hydro_vars;
    use crate::inlist::{InlistBundle, InlistLoadOptions};
    use crate::model::StarModel;
    use crate::solver_support::set_vars_for_solver;
    use crate::zams::load_zams_model;

    struct TempInlistDir(PathBuf);

    impl Drop for TempInlistDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn test_expected_non_hse_term_uses_dxh_v_above_logt_bound() {
        let ctrl = StarControls {
            v_flag: true,
            velocity_logt_lower_bound: 4.0,
            ..StarControls::default()
        };
        let mut model =
            StarModel::new(3, vec![1], StarJobControls::default(), ctrl);
        model.dt = 10.0;
        model.ln_t = vec![5.0 * LN10; 3];
        model.ln_t_old = vec![5.0 * LN10, 5.0 * LN10, 3.0 * LN10];
        model.v = vec![3.0e5, 4.0e5, 5.0e5];
        model.v_start = vec![1.0e5, 2.0e5, 3.0e5];
        // Intentionally distinct from `v - v_start = 2.0e5` so any fallback is caught.
        model.aux.dxh_v = vec![0.5e5, 0.5e5, 0.5e5];

        let term_above = expected_non_hse_term(&model, 1);
        assert!((term_above.val - (-0.5e5 / 10.0)).abs() < 1.0e-12);
        assert!((term_above.d1[I_V_00] - (-1.0 / 10.0)).abs() < 1.0e-12);

        let term_below = expected_non_hse_term(&model, 2);
        assert!((term_below.val - (-2.0 * 5.0e5 / 10.0)).abs() < 1.0e-12);
        assert!((term_below.d1[I_V_00] - (-2.0 / 10.0)).abs() < 1.0e-12);
    }

    #[test]
    fn test_expected_non_hse_term_gates_extra_grav_on_use_other_momentum() {
        let ctrl = StarControls::default();
        let mut model =
            StarModel::new(3, vec![1], StarJobControls::default(), ctrl);
        model.aux.extra_grav = vec![StarAdOrder1::constant(-125.0); 3];

        assert!(expected_non_hse_term(&model, 1).val.abs() < 1.0e-15);

        model.controls.use_other_momentum = true;
        assert!(
            (expected_non_hse_term(&model, 1).val - (-125.0)).abs() < 1.0e-12
        );

        model.controls.use_other_momentum = false;
        model.controls.use_other_momentum_implicit = true;
        assert!(
            (expected_non_hse_term(&model, 1).val - (-125.0)).abs() < 1.0e-12
        );
    }

    #[test]
    fn test_unpack_xh_clears_and_set_vars_for_solver_stores_dxh_v() {
        let ctrl = StarControls {
            v_flag: true,
            zbase: 0.02,
            initial_mass: 1.0,
            ..StarControls::default()
        };
        let mut model = load_zams_model(ctrl, StarJobControls::default())
            .expect("load zams");
        model.pack_xh();
        model.aux.dxh_v = vec![123.0; model.nz];
        model.unpack_xh();
        assert!(model.aux.dxh_v.iter().all(|&x| x == 0.0));

        let xh_start = model.xh.clone();
        let nvar = model.layout.nvar(true);
        let iv = model.layout.i_v.expect("i_v with v_flag");
        let mut solver_dx = vec![0.0; model.nz * nvar];
        for k in 0..model.nz {
            let k_f64 = f64::from(u32::try_from(k).expect("nz fits u32"));
            solver_dx[k * nvar + iv] = 42.5 + k_f64;
        }
        set_vars_for_solver(&mut model, true, &xh_start, &mut solver_dx)
            .expect("set_vars_for_solver");
        for k in 0..model.nz {
            let expected =
                42.5 + f64::from(u32::try_from(k).expect("nz fits u32"));
            assert!((model.aux.dxh_v[k] - expected).abs() < 1.0e-12);
        }
    }

    #[test]
    fn test_use_other_momentum_inlist_and_set_hydro_vars_error() {
        let dir = std::env::temp_dir()
            .join(format!("flexmesa_other_mom_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp dir");
        let _guard = TempInlistDir(dir.clone());
        let inlist_path = dir.join("inlist_project");
        fs::write(
            &inlist_path,
            "&controls\n  use_other_momentum = .true.\n/\n",
        )
        .expect("write inlist");

        assert!(matches!(
            InlistBundle::from_file(&inlist_path),
            Err(StarError::InvalidInput(_))
        ));

        let bundle = InlistBundle::from_file_with_options(
            &inlist_path,
            InlistLoadOptions {
                allow_unsupported_controls: true,
            },
        )
        .expect("parse with allow_unsupported_controls");
        assert!(bundle.controls.use_other_momentum);

        let ctrl = StarControls {
            zbase: 0.02,
            initial_mass: 1.0,
            ..StarControls::default()
        };
        let mut model = load_zams_model(ctrl, StarJobControls::default())
            .expect("load zams");
        model.controls.use_other_momentum = true;
        let res = set_hydro_vars(&mut model, false, false);
        assert!(matches!(
            res,
            Err(StarError::InvalidInput(ref msg)) if msg.contains("other_momentum")
        ));
    }

    #[test]
    fn test_setup_rti_momentum_terms_runs_without_vel_flags_and_uses_u_face_ad()
    {
        let ctrl = StarControls {
            rti_flag: true,
            dlnddt_rti_diffusion_factor: 0.75,
            dudt_rti_diffusion_factor: 0.0,
            ..StarControls::default()
        };
        let mut model =
            StarModel::new(4, vec![1], StarJobControls::default(), ctrl);
        model.dt = 5.0;
        model.aux.eta_rti[1] = 2.0e14;
        model.rho = vec![1.0e-3, 2.5e-3, 3.0e-3, 4.0e-3];
        let dm_div_a = StarAdOrder1::constant(1.0e4);

        let kick = setup_rti_momentum_terms(&model, 1, dm_div_a);
        let exp_kick = 0.75 * 2.0e14 / dm_div_a
            * (wrap_d_00(&model, 1) - wrap_d_m1(&model, 1))
            / 5.0;
        assert!(kick.val.abs() > 0.0);
        assert!((kick.val - exp_kick.val).abs() < 1.0e-12);
        assert!((kick.d1[I_LND_00] - exp_kick.d1[I_LND_00]).abs() < 1.0e-12);
        assert!((kick.d1[I_LND_M1] - exp_kick.d1[I_LND_M1]).abs() < 1.0e-12);

        model.controls.u_flag = true;
        model.controls.dlnddt_rti_diffusion_factor = 0.0;
        model.controls.dudt_rti_diffusion_factor = 1.2;
        model.dm_bar[1] = 1.0e28;
        model.aux.sigmid_rti[0] = 1.5e22;
        model.aux.sigmid_rti[1] = 2.5e22;
        model.u = vec![1.0e4, 2.0e4, 3.0e4, 4.0e4];
        model.aux.u_face_ad[0] =
            StarAdOrder1::from_dlnd_dlnt_00(1.0e5, 3.0e2, 0.0);
        model.aux.u_face_ad[1] = StarAdOrder1::variable(3.0e5, I_V_00);
        model.aux.u_face_ad[2] = StarAdOrder1::variable(7.0e5, I_V_00);

        let diff = setup_rti_momentum_terms(&model, 1, dm_div_a);
        let exp_diff = (1.2
            * 2.5e22
            * (model.aux.u_face_ad[2].shift_p1() - model.aux.u_face_ad[1])
            - 1.2
                * 1.5e22
                * (model.aux.u_face_ad[1] - model.aux.u_face_ad[0].shift_m1()))
            / model.dm_bar[1];
        assert!((diff.val - exp_diff.val).abs() < 1.0e-12);
        for (a, b) in diff.d1.iter().zip(exp_diff.d1.iter()) {
            assert!((a - b).abs() < 1.0e-15);
        }
    }

    #[test]
    fn test_do1_radius_eqn_errors_without_vel_flag_and_uses_start_arrays() {
        let ctrl_none = StarControls::default();
        let m_none =
            StarModel::new(3, vec![1], StarJobControls::default(), ctrl_none);
        let nvar_none = m_none.layout.nvar(false);
        let mut sys_none =
            HenyeySystem::new(3, nvar_none, vec![1.0; 3 * nvar_none]);
        let res = do1_radius_eqn(&m_none, 0, &mut sys_none);
        assert!(matches!(
            res,
            Err(StarError::InvalidInput(ref msg))
                if msg.contains("must have either v or u")
        ));

        let ctrl_u = StarControls {
            u_flag: true,
            using_velocity_time_centering: true,
            velocity_q_upper_bound: 2.0,
            velocity_tau_lower_bound: -1.0,
            velocity_logt_lower_bound: 4.0,
            ..StarControls::default()
        };
        let mut mu =
            StarModel::new(3, vec![1], StarJobControls::default(), ctrl_u);
        mu.dt = 10.0;
        mu.q = vec![1.0, 0.5, 0.1];
        mu.tau = vec![1.0e5; 3];
        mu.ln_t = vec![15.0; 3];
        mu.ln_t_start = vec![15.0; 3];
        mu.ln_r_start = vec![25.0; 3];
        mu.aux.dxh_lnr = vec![1.0e-3; 3];
        mu.u = vec![1.0e5; 3];
        mu.u_start = vec![9.0e5; 3];
        let mut uf = StarAdOrder1::variable(4.0e5, I_V_00);
        uf.d1[I_LND_00] = 7.0;
        mu.aux.u_face_ad[1] = uf;
        mu.aux.u_face_start[1] = 2.0e5;

        let nvar_u = mu.layout.nvar(false);
        let mut sys_u = HenyeySystem::new(3, nvar_u, vec![1.0; 3 * nvar_u]);
        do1_radius_eqn(&mu, 1, &mut sys_u).expect("do1_radius_eqn u_flag");
        let i_equ_u = mu.layout.i_dlnr_dt.expect("i_dlnr_dt");
        let iu = mu.layout.i_u.expect("i_u");
        let r_start = flexmesa_math::exp(25.0);
        let exp_u = 0.5 * (uf + 2.0e5) * 10.0 / r_start
            - StarAdOrder1::variable(1.0e-3, I_LNR_00).expm1();
        assert!((sys_u.equ[nvar_u + i_equ_u] - exp_u.val).abs() < 1.0e-12);
        assert!(
            (sys_u.dblk[nvar_u * nvar_u + i_equ_u * nvar_u + iu]
                - exp_u.d1[I_V_00])
                .abs()
                < 1.0e-15
        );
        assert!(
            (sys_u.dblk[nvar_u * nvar_u + i_equ_u * nvar_u + mu.layout.i_lnd]
                - exp_u.d1[I_LND_00])
                .abs()
                < 1.0e-15
        );
        assert!(
            (sys_u.dblk[nvar_u * nvar_u + i_equ_u * nvar_u + mu.layout.i_lnr]
                - exp_u.d1[I_LNR_00])
                .abs()
                < 1.0e-15
        );

        let ctrl_v = StarControls {
            v_flag: true,
            using_velocity_time_centering: true,
            velocity_q_upper_bound: 2.0,
            velocity_tau_lower_bound: -1.0,
            velocity_logt_lower_bound: 4.0,
            ..StarControls::default()
        };
        let mut mv =
            StarModel::new(3, vec![1], StarJobControls::default(), ctrl_v);
        mv.dt = 10.0;
        mv.q = vec![1.0, 0.5, 0.1];
        mv.tau = vec![1.0e5; 3];
        mv.ln_t = vec![15.0; 3];
        mv.ln_t_start = vec![15.0; 3];
        mv.ln_r_start = vec![25.0; 3];
        mv.aux.dxh_lnr = vec![1.0e-3; 3];
        mv.v = vec![6.0e5; 3];
        mv.v_start = vec![2.0e5; 3];

        let nvar_v = mv.layout.nvar(false);
        let mut sys_v = HenyeySystem::new(3, nvar_v, vec![1.0; 3 * nvar_v]);
        do1_radius_eqn(&mv, 1, &mut sys_v).expect("do1_radius_eqn v_flag");
        let i_equ_v = mv.layout.i_dlnr_dt.expect("i_dlnr_dt");
        let iv = mv.layout.i_v.expect("i_v");
        let exp_v =
            0.5 * (StarAdOrder1::variable(6.0e5, I_V_00) + 2.0e5) * 10.0
                / r_start
                - StarAdOrder1::variable(1.0e-3, I_LNR_00).expm1();
        assert!((sys_v.equ[nvar_v + i_equ_v] - exp_v.val).abs() < 1.0e-12);
        assert!(
            (sys_v.dblk[nvar_v * nvar_v + i_equ_v * nvar_v + iv]
                - exp_v.d1[I_V_00])
                .abs()
                < 1.0e-15
        );
    }
}
