// SPDX-License-Identifier: LGPL-3.0-or-later
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

use flexmesa_const::{CLIGHT, CRAD, FOUR_PI, L_SUN};

use crate::hydro_eqns::HenyeySystem;
use crate::hydro_momentum::eval_cell_peos_time_centered_ad;
use crate::model::StarModel;
use crate::star_ad::{
    StarAdExt, StarAdOrder1, I_LND_00, I_LND_M1, I_LNR_00, I_LNT_00, I_LNT_M1,
    I_L_00,
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
        let fac = (1.0 / 3.0) * model.controls.mlt_pturb_factor;
        let vc_00 = model.aux.mlt_vc_old.get(k).copied().unwrap_or(0.0);
        let vc_m1 = model.aux.mlt_vc_old.get(k - 1).copied().unwrap_or(0.0);
        if vc_00 > 0.0 {
            let rho_00 = StarAdOrder1::variable(model.lnd[k], I_LND_00).exp();
            p_00 = p_00 + rho_00 * (fac * vc_00 * vc_00);
        }
        if vc_m1 > 0.0 {
            let rho_m1 =
                StarAdOrder1::variable(model.lnd[k - 1], I_LND_M1).exp();
            p_m1 = p_m1 + rho_m1 * (fac * vc_m1 * vc_m1);
        }
    }
    p_00 * alfa + p_m1 * beta
}

/// Evaluates the RSP outer Stefan-Boltzmann luminosity boundary condition at
/// `k = 0` (`set_RSP_Lsurf_BC` in `star/private/hydro_eqns.f90:961-983`).
pub fn set_rsp_lsurf_bc(model: &StarModel, sys: &mut HenyeySystem) {
    if model.nz == 0 {
        return;
    }
    let lum_0 = StarAdOrder1::variable(model.lum[0], I_L_00);
    let r_0 = StarAdOrder1::variable(model.ln_r[0], I_LNR_00).exp();
    let t_0 = StarAdOrder1::variable(model.ln_t[0], I_LNT_00).exp();
    // Upstream `set_RSP_Lsurf_BC`: `L1 = 0.5 * 4 * pi * r(1)^2 * (c * a / 4) * T(1)^4`
    // (`0.5` Eddington outer boundary radiation flux factor in RSP).
    let sigma_sb = 0.25 * CLIGHT * CRAD;
    let l_rad_surf = (0.5 * FOUR_PI * sigma_sb) * (r_0 * r_0) * t_0.pow4();
    let l_scale = model
        .lum_start
        .first()
        .copied()
        .unwrap_or(L_SUN)
        .abs()
        .max(L_SUN);
    let resid = (lum_0 - l_rad_surf) / l_scale;
    sys.save_eqn_residual(&model.layout, 0, model.layout.i_equ_l, &resid);
}

/// Evaluates the direct $\nabla_T$ temperature equation at interior face `k >= 1`
/// (`do1_gradT_eqn` in `star/private/hydro_temperature.f90:53-112`).
fn do1_gradt_eqn(model: &StarModel, k: usize, sys: &mut HenyeySystem) {
    let (alfa, beta) = model.get_face_weights(k);
    let (p_00, _) = eval_cell_peos_time_centered_ad(model, k, false);
    let (p_m1, _) = eval_cell_peos_time_centered_ad(model, k - 1, true);
    let p_face = p_00 * alfa + p_m1 * beta;
    let dln_p = (p_m1 - p_00) / p_face;

    let t_00 = StarAdOrder1::variable(model.ln_t[k], I_LNT_00).exp();
    let t_m1 = StarAdOrder1::variable(model.ln_t[k - 1], I_LNT_M1).exp();
    let t_face = t_00 * alfa + t_m1 * beta;
    let dln_t = (t_m1 - t_00) / t_face;

    let resid = model.gradt_ad[k] * dln_p - dln_t;
    sys.save_eqn_residual(&model.layout, k, model.layout.i_equ_l, &resid);
}

/// Evaluates the radiation-pressure gradient temperature equation with the
/// Levermore & Pomraning (1981) flux limiter at interior face `k >= 1`
/// (`do1_alt_dlnT_dm_eqn` in `star/private/hydro_temperature.f90:115-238`).
fn do1_alt_dlnt_dm_eqn(model: &StarModel, k: usize, sys: &mut HenyeySystem) {
    let r_k = StarAdOrder1::variable(model.ln_r[k], I_LNR_00).exp();
    let area = FOUR_PI * r_k * r_k;
    let dm_bar = model.dm_bar[k].max(1.0e-30);
    let kap_f = model.interp_cell_ad_to_face(
        &model.opacity,
        &model.d_opacity_dlnd,
        &model.d_opacity_dlnt,
        k,
    );
    let t_00 = StarAdOrder1::variable(model.ln_t[k], I_LNT_00).exp();
    let t_m1 = StarAdOrder1::variable(model.ln_t[k - 1], I_LNT_M1).exp();
    let prad_00 = (CRAD / 3.0) * t_00.pow4();
    let prad_m1 = (CRAD / 3.0) * t_m1.pow4();
    let (alfa, beta) = model.get_face_weights(k);
    let prad_f = prad_00 * alfa + prad_m1 * beta;

    // Levermore & Pomraning (1981) dimensionless radiation gradient R_LP and
    // flux limiter lambda_LP = (2 + R) / (6 + 3*R + R^2), f_LP = 1 / (3 * lambda_LP)
    // evaluated as full 3-cell `StarAdOrder1` expressions (`hydro_temperature.f90:129-140`).
    let dprad_actual = prad_m1 - prad_00;
    let dtau_f = (kap_f * dm_bar / area.max(StarAdOrder1::ONE))
        .max(StarAdOrder1::constant(1.0e-12));
    let r_lp = (dprad_actual.abs()
        / (prad_f.max(StarAdOrder1::constant(1.0e-30)) * dtau_f))
        .min(StarAdOrder1::constant(1.0e4));
    let lambda_lp = (r_lp + 2.0) / (r_lp * r_lp + r_lp * 3.0 + 6.0);
    let f_lp = (StarAdOrder1::ONE / (lambda_lp * 3.0)).max(StarAdOrder1::ONE);

    let lum_k = StarAdOrder1::variable(model.lum[k], I_L_00);
    // Multiply by (gradT / gradr) so convective transport reduces the required
    // radiation-pressure drop identically in convective zones.
    let grad_ratio = if model.gradr_ad[k].val.abs() > 1.0e-30 {
        model.gradt_ad[k] / model.gradr_ad[k]
    } else {
        StarAdOrder1::ONE
    };
    let dprad_dm_expected =
        -(kap_f * lum_k / (area * CLIGHT)) * grad_ratio * f_lp;
    let dprad_dm_actual = dprad_actual / dm_bar;
    let scale = (prad_f / dm_bar).max(StarAdOrder1::constant(1.0e-30));
    let resid = (dprad_dm_expected - dprad_dm_actual) / scale;
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

    let (alfa, beta) = model.get_face_weights(k);
    let p_point = interp_p_point_ad(model, k);
    let fp = if model.controls.rotation_flag
        && model.controls.use_gravity_rotation_correction
    {
        model.aux.fp_rot.get(k).copied().unwrap_or(1.0)
    } else {
        1.0
    };

    let r_k = StarAdOrder1::variable(model.ln_r[k], I_LNR_00).exp();
    let dlnp_dm_qhse = -(model.cgrav[k] * model.m_grav[k] * fp)
        / (FOUR_PI * r_k.pow4() * p_point);
    let dm_k = 0.5 * (model.dm[k - 1] + model.dm[k]);
    let dln_t_theory = dlnp_dm_qhse * model.gradt_ad[k] * dm_k;

    let t_00 = StarAdOrder1::variable(model.ln_t[k], I_LNT_00).exp();
    let t_m1 = StarAdOrder1::variable(model.ln_t[k - 1], I_LNT_M1).exp();
    let t_point = t_00 * alfa + t_m1 * beta;
    let dln_t_actual = (t_m1 - t_00) / t_point;

    let resid = dln_t_theory - dln_t_actual;
    sys.save_eqn_residual(&model.layout, k, model.layout.i_equ_l, &resid);
}
