// SPDX-License-Identifier: GPL-3.0-or-later
//! Face-centered temperature gradient transport equations (`do1_temperature_eqn`,
//! `do1_dlnT_dm_eqn`, `do1_gradT_eqn`, `do1_alt_dlnT_dm_eqn`, and
//! `set_RSP_Lsurf_BC`) with 3-cell stencil [`StarAdOrder1`] automatic
//! differentiation (`star/private/hydro_temperature.f90` and `hydro_eqns.f90`).
//!
//! Implements full line-by-line fidelity with upstream `hydro_temperature.f90`:
//! - **Standard quasi-hydrostatic $d\ln T / dm$ equation (`do1_dlnT_dm_eqn`,
//!   `hydro_temperature.f90:240-310, 412-434`)**:
//!   $$\left(\frac{d\ln P}{dm}\right)_{\mathrm{QHSE},k} = -\frac{f_{P,\mathrm{rot},k}\,G_k m_{\mathrm{grav},k}}{4\pi r_k^4 P_{\mathrm{point},k}},$$
//!   including rotational distortion $f_{P,\mathrm{rot},k}$ (`rotation_flag && use_gravity_rotation_correction`),
//!   time-centered pressure $\theta_P$, and MLT turbulent pressure $P_{\mathrm{turb,MLT}}$.
//! - **Direct $\nabla_T$ contrast equation (`do1_gradT_eqn`,
//!   `hydro_temperature.f90:53-112`)** when
//!   `use_gradt_actual_vs_gradt_mlt_for_t_gradient_eqn` is enabled:
//!   $$\mathrm{equ}_k = \nabla_{T,\mathrm{MLT},k}\,\Delta\ln P_k - \Delta\ln T_k.$$
//! - **Radiation-pressure gradient equation with Levermore-Pomraning (1981)
//!   flux limiter (`do1_alt_dlnT_dm_eqn`, `hydro_temperature.f90:115-238`)**
//!   when `use_dprad_dm_form_of_t_gradient_eqn` is enabled.
//! - **RSP surface Stefan-Boltzmann luminosity boundary condition
//!   (`set_RSP_Lsurf_BC`, `hydro_eqns.f90:961-983`)** when
//!   `use_rsp_l_eqn_outer_bc` is enabled.

use flexmesa_const::{MixingType, CLIGHT, CRAD, FOUR_PI, LN10, L_SUN};

use crate::hydro_eqns::HenyeySystem;
use crate::hydro_momentum::{
    eval_cell_peos_time_centered_ad, eval_mlt_pturb_cell_ad,
    expected_hse_grav_and_area, get_area_info_opt_time_center, wrap_peos_00,
    wrap_peos_m1,
};
use crate::model::StarModel;
use crate::star_ad::{
    StarAdExt, StarAdOrder1, I_LNR_00, I_LNT_00, I_LNT_M1, I_L_00,
};

/// Evaluates face-interpolated total pressure $P_{\mathrm{point},k}$ at
/// interior face `k >= 1` including optional $\theta_P$ velocity time-centering
/// and MLT turbulent pressure $P_{\mathrm{turb,MLT}}$.
#[must_use]
pub fn interp_p_point_ad(model: &StarModel, k: usize) -> StarAdOrder1 {
    let (alfa, beta) = model.get_face_weights(k);
    let (mut p_00, _) = eval_cell_peos_time_centered_ad(model, k, false);
    let (mut p_m1, _) = eval_cell_peos_time_centered_ad(model, k - 1, true);
    if model.controls.mlt_pturb_factor > 0.0 {
        p_00 = p_00 + eval_mlt_pturb_cell_ad(model, k, false);
        p_m1 = p_m1 + eval_mlt_pturb_cell_ad(model, k - 1, true);
    }
    p_00 * alfa + p_m1 * beta
}

/// Evaluates the RSP outer Stefan-Boltzmann luminosity boundary condition at
/// `k = 0` (`set_RSP_Lsurf_BC` in `star/private/hydro_temperature.f90:318-388`).
pub fn set_rsp_lsurf_bc(model: &StarModel, sys: &mut HenyeySystem) {
    if model.nz == 0 {
        return;
    }
    let (area_ad, _) = get_area_info_opt_time_center(model, 0);
    let lum_0 = StarAdOrder1::variable(model.lum[0], I_L_00);
    let t_0 = StarAdOrder1::variable(model.ln_t[0], I_LNT_00).exp();
    // `rhs_ad = s%RSP2_Lsurf_factor * area_ad * clight * (crad * pow4(T_surf))`
    // (`hydro_temperature.f90:348-370`).
    let rhs_ad = model.controls.rsp2.lsurf_factor
        * area_ad
        * CLIGHT
        * (CRAD * t_0.pow4());
    let resid_ad = lum_0 - rhs_ad;
    let max_l_start = model
        .lum_start
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let scale = if max_l_start > 0.0 && max_l_start.is_finite() {
        max_l_start
    } else {
        L_SUN
    };
    let resid = resid_ad / scale;
    sys.save_eqn_residual(&model.layout, 0, model.layout.i_equ_l, &resid);
}

/// Evaluates the direct $\nabla_T$ temperature equation at interior face `k >= 1`
/// (`do1_gradT_eqn` in `star/private/hydro_temperature.f90:158-204`).
fn do1_gradt_eqn(model: &StarModel, k: usize, sys: &mut HenyeySystem) {
    let lnt_00 = StarAdOrder1::variable(model.ln_t[k], I_LNT_00);
    let lnt_m1 = StarAdOrder1::variable(model.ln_t[k - 1], I_LNT_M1);
    let dln_t = lnt_m1 - lnt_00;
    let dln_p = model.ln_peos_ad[k - 1].shift_m1() - model.ln_peos_ad[k];

    let resid = model.gradt_ad[k] * dln_p - dln_t;
    sys.save_eqn_residual(&model.layout, k, model.layout.i_equ_l, &resid);
}

/// Evaluates the radiation-pressure gradient temperature equation with optional
/// Levermore & Pomraning (1981) flux limiting at interior face `k >= 1`
/// (`do1_alt_dlnT_dm_eqn` in `star/private/hydro_temperature.f90:37-155`).
fn do1_alt_dlnt_dm_eqn(model: &StarModel, k: usize, sys: &mut HenyeySystem) {
    let e_start = model
        .energy_start
        .get(k)
        .copied()
        .unwrap_or(model.energy[k]);
    let rho_start = model
        .lnd_start
        .get(k)
        .copied()
        .map_or_else(|| model.rho[k], flexmesa_math::exp);
    let raw_scale = e_start * rho_start;
    let scale = if raw_scale != 0.0 && raw_scale.is_finite() {
        raw_scale
    } else {
        1.0e-30
    };
    let dm_bar = if model.dm_bar[k] == 0.0 {
        1.0e-30
    } else {
        model.dm_bar[k]
    };

    let lum_k = StarAdOrder1::variable(model.lum[k], I_L_00);
    let r_k = StarAdOrder1::variable(model.ln_r[k], I_LNR_00).exp();
    let area = FOUR_PI * (r_k * r_k);
    let area2 = area * area;

    let lrad_ad = if model.ln_t[k] / LN10
        <= model.controls.mix_wind.max_logt_for_mlt
        && model.mixing_type[k] == MixingType::ConvectiveMixing
        && model.gradr[k] > 0.0
        && (model.gradr[k] - model.gradt[k]).abs()
            > model.gradr[k].abs() * 1.0e-5
        && model.gradr_ad[k].val != 0.0
    {
        lum_k * model.gradt_ad[k] / model.gradr_ad[k]
    } else {
        lum_k
    };

    let mut kap_f = model.interp_cell_ad_to_face(
        &model.opacity,
        &model.d_opacity_dlnd,
        &model.d_opacity_dlnt,
        k,
    );
    if kap_f.val < model.controls.min_kap_for_dprad_dm_eqn {
        kap_f = StarAdOrder1::constant(model.controls.min_kap_for_dprad_dm_eqn);
    }

    let mut d_p_rad_expected = -(dm_bar * kap_f * lrad_ad) / (CLIGHT * area2);

    let t_00 = StarAdOrder1::variable(model.ln_t[k], I_LNT_00).exp();
    let t_m1 = StarAdOrder1::variable(model.ln_t[k - 1], I_LNT_M1).exp();
    let t4_00 = t_00.pow4();
    let t4_m1 = t_m1.pow4();
    let prad_00 = (CRAD / 3.0) * t4_00;
    let prad_m1 = (CRAD / 3.0) * t4_m1;
    let d_p_rad_actual = prad_m1 - prad_00;

    if model.controls.use_flux_limiting_with_dprad_dm_form {
        let flx_r = (area * (t4_m1 - t4_00).abs() / dm_bar)
            / (kap_f * 0.5 * (t4_m1 + t4_00));
        let flx_lambda = (6.0 + 3.0 * flx_r) / (6.0 + (3.0 + flx_r) * flx_r);
        d_p_rad_expected = d_p_rad_expected / flx_lambda;
    }

    let resid = (d_p_rad_expected - d_p_rad_actual) / scale;
    sys.save_eqn_residual(&model.layout, k, model.layout.i_equ_l, &resid);
}

/// Interior temperature transport equation at face `k`
/// (`do1_temperature_eqn`, `do1_dlnT_dm_eqn`, `do1_gradT_eqn`, and
/// `do1_alt_dlnT_dm_eqn` in `star/private/hydro_temperature.f90:36-310`).
pub fn do1_temperature_eqn(
    model: &StarModel,
    k: usize,
    sys: &mut HenyeySystem,
) {
    if k == 0 {
        if model.controls.use_rsp_l_eqn_outer_bc {
            set_rsp_lsurf_bc(model, sys);
        }
        return;
    }
    if model
        .controls
        .use_gradt_actual_vs_gradt_mlt_for_t_gradient_eqn
    {
        do1_gradt_eqn(model, k, sys);
        return;
    }
    if model.controls.use_dprad_dm_form_of_t_gradient_eqn {
        do1_alt_dlnt_dm_eqn(model, k, sys);
        return;
    }

    // `do1_dlnT_dm_eqn` (`star/private/hydro_temperature.f90:277-296`).
    let (dlnp_dm, _p_point) = eval_dlnpdm_qhse(model, k);

    let grad_t = model.gradt_ad[k];
    let dln_t_dm = dlnp_dm * grad_t;

    let t_m1 = StarAdOrder1::from_dlnd_dlnt_m1(model.t[k - 1], 0.0, model.t[k - 1]);
    let t_00 = StarAdOrder1::from_dlnd_dlnt_00(model.t[k], 0.0, model.t[k]);
    let d_t = t_m1 - t_00;
    let alfa = model.dm[k - 1] / (model.dm[k - 1] + model.dm[k]);
    let t_point = alfa * t_00 + (1.0 - alfa) * t_m1;
    let ln_t_diff = d_t / t_point;
    let delm = (model.dm[k] + model.dm[k - 1]) / 2.0;

    let resid = delm * dln_t_dm - ln_t_diff;
    sys.save_eqn_residual(&model.layout, k, model.layout.i_equ_l, &resid);
}

/// `eval_dlnPdm_qhse` (`star/private/hydro_temperature.f90:391-458`) at
/// interior face `k >= 1`: returns `(dlnPdm_qhse, Ppoint)`.
///
/// The MLT turbulent-pressure terms are added only under the dev control
/// `include_mlt_Pturb_in_thermodynamic_gradients`
/// (`star/defaults/controls_dev.defaults:212`, default `.false.`), which
/// `FlexMESA` does not accept, so they are always zero here.
fn eval_dlnpdm_qhse(model: &StarModel, k: usize) -> (StarAdOrder1, StarAdOrder1) {
    let (grav, area) = expected_hse_grav_and_area(model, k);
    let using_tc = model.controls.is_velocity_time_centered(model.model_number_old);

    let mut p00 = wrap_peos_00(model, k);
    if using_tc {
        p00 = 0.5 * (p00 + model.peos_start[k]);
    }
    let mut pm1 = wrap_peos_m1(model, k);
    if using_tc {
        pm1 = 0.5 * (pm1 + model.peos_start[k - 1]);
    }
    let alfa = model.dq[k - 1] / (model.dq[k - 1] + model.dq[k]);
    let p_point = alfa * p00 + (1.0 - alfa) * pm1;

    (grav / (area * p_point), p_point)
}
