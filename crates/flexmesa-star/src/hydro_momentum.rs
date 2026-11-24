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
//! - **Radius equation (`do1_radius_eqn`, `star/private/hydro_momentum.f90:545-601`)** with
//!   `force_zero_v` bounds (`velocity_q_upper_bound`, `velocity_tau_lower_bound`,
//!   `velocity_logt_lower_bound`) and optional velocity time-centering.

use flexmesa_const::{FOUR_PI, LN10};

use crate::hydro_energy::{eval_face_area_time_centered_ad, eval_pvsc_cell_ad};
use crate::hydro_eqns::HenyeySystem;
use crate::model::StarModel;
use crate::star_ad::{
    StarAdExt, StarAdOrder1, I_LND_00, I_LND_M1, I_LNR_00, I_LNT_00, I_V_00, I_V_M1,
    I_V_P1,
};

/// Evaluates the pressure time-centering weight $\theta_P \in [0, 1]$ at cell
/// or face `k` (`hydro_momentum.f90:151-162`).
#[must_use]
#[inline]
pub fn eval_p_theta(model: &StarModel, k: usize) -> f64 {
    let logt = model.ln_t[k] / LN10;
    if model.controls.is_velocity_time_centered(model.model_number_old)
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
    let vc_old = model.aux.mlt_vc_old.get(k).copied().unwrap_or(0.0);
    if vc_old <= 0.0 {
        return StarAdOrder1::ZERO;
    }
    let rho_slot = if shift_m1 { I_LND_M1 } else { I_LND_00 };
    let rho_ad = StarAdOrder1::variable(model.lnd[k], rho_slot).exp();
    rho_ad * ((1.0 / 3.0) * model.controls.mlt_pturb_factor * vc_old * vc_old)
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
    let using_v_tc =
        model.controls.is_velocity_time_centered(model.model_number_old)
            && (model.controls.v_flag || model.controls.u_flag)
            && model.dt > 0.0
            && !model.ln_r_start.is_empty();
    let lnr_start_k = model.ln_r_start.get(k).copied().unwrap_or(model.ln_r[k]);
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
        model.ln_peos_surf_ad.exp()
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
    let r_00 = wrap_r_00(model, k);
    let r2_00 = r_00.pow2();
    if model.controls.is_velocity_time_centered(model.model_number_old) {
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

/// `setup_RTI_terms` (`star/private/hydro_momentum.f90:250-277`): `RTI`
/// momentum diffusion and density-kick terms at face `k`, in $\mathrm{cm\,s^{-2}}$.
#[must_use]
fn setup_rti_momentum_terms(
    model: &StarModel,
    k: usize,
    dm_div_a: StarAdOrder1,
) -> StarAdOrder1 {
    if !model.controls.rti_flag
        || k == 0
        || k + 1 >= model.nz
        || !(model.controls.v_flag || model.controls.u_flag)
    {
        return StarAdOrder1::ZERO;
    }
    let dvdt_diffusion = if model.controls.dudt_rti_diffusion_factor > 0.0 {
        let fac = model.controls.dudt_rti_diffusion_factor;
        let sigm1 = fac * model.aux.sigmid_rti.get(k - 1).copied().unwrap_or(0.0);
        let sig00 = fac * model.aux.sigmid_rti.get(k).copied().unwrap_or(0.0);
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
        (sig00 * (v_p1 - v_00) - sigm1 * (v_00 - v_m1)) / model.dm_bar[k]
    } else {
        StarAdOrder1::ZERO
    };
    let eta_k = model.aux.eta_rti.get(k).copied().unwrap_or(0.0);
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
            // `drag = -s% drag_coefficient*v_00/s% dt` (`hydro_momentum.f90:410`).
            non_hse = non_hse + -(model.controls.drag_coefficient * v_00 / dt);
        }
    }

    if let Some(&uq) = model.aux.uq_ad.get(k) {
        if uq.val != 0.0 || uq.d1.iter().any(|&d| d != 0.0) {
            non_hse = non_hse + uq;
        }
    }

    non_hse
}

/// `calc_Ptot_ad_tw` with `skip_mlt_Pturb = .true.`
/// (`star/private/star_utils.f90:3375-3438`, called from
/// `get_dPtot_face_info`, `hydro_momentum.f90:470-478`) for cell `cell`,
/// placed in the `m1` slots when `shift_m1`.
#[must_use]
pub fn calc_ptot_ad_tw(model: &StarModel, cell: usize, shift_m1: bool) -> StarAdOrder1 {
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
    let (alfa, beta) = if k == 0 { (1.0, 0.0) } else { model.get_face_weights(k) };
    let ptotavg_ad = if k == 0 {
        ptot00_ad
    } else {
        alfa * ptot00_ad + beta * ptotm1_ad
    };
    let iptotavg_ad = 1.0 / ptotavg_ad;
    // `setup_d_mlt_Pturb`
    let vc_old = model.aux.mlt_vc_old.get(k).copied().unwrap_or(0.0);
    let d_mlt_pturb_ad = if model.controls.mlt_pturb_factor > 0.0 && vc_old > 0.0 {
        let rho_00 = wrap_d_00(model, k);
        let rho_m1 = wrap_d_m1(model, k);
        model.controls.mlt_pturb_factor * flexmesa_math::pow2(vc_old) * (rho_m1 - rho_00)
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
    let resid1_ad = other_dm_div_a_ad + grav_dm_div_a_ad - dptot_ad - d_mlt_pturb_ad
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
            let d_ptotavg_dxa00 =
                if k == 0 { d_ptot00_dxa00 } else { alfa * d_ptot00_dxa00 };
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
            model.controls.is_velocity_time_centered(model.model_number_old);
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
        let r_start = flexmesa_math::exp(model.ln_r_start[k]).max(1.0);
        v_ad * (model.dt / r_start)
            - ((lnr_ad - model.ln_r_start[k]).exp() - 1.0)
    };
    sys.save_eqn_residual(&model.layout, k, i_dlnr_dt, &resid);
}
