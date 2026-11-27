// SPDX-License-Identifier: GPL-3.0-or-later
//! TDC eddy-viscosity terms for `v_flag` and `u_flag` hydrodynamics
//! (`star/private/tdc_hydro.f90`): the face eddy-viscosity energy divided by
//! the convective velocity `Chi/w` (`compute_Chi_div_w_face`), the turbulent
//! dissipation `Eq/w` (`compute_tdc_Eq_div_w_face`), the turbulent viscous
//! accelerations `Uq` (`compute_tdc_Uq_face` for `v_flag`,
//! `compute_tdc_Uq_dm_cell` for `u_flag`), and `set_viscosity_vars_TDC`.
//!
//! Faces and cells use 0-based `k` here; upstream's 1-based `k` is `k + 1`.
//!
//! Only the default branches of four controls that this port does not parse
//! are reproduced; setting any of them in an inlist is rejected as an
//! unsupported control when the inlist is read:
//! - `TDC_use_density_form_for_eddy_viscosity = .false.`: the `d(v/r)/dr`
//!   forms, not the density forms (`tdc_hydro.f90:179-185,252-258,326-331`).
//! - `TDC_alpha_M_use_explicit_mlt_vc_in_momentum_equation = .false.`:
//!   `w = mlt_vc_ad/sqrt(2/3)`, not `mlt_vc_old` (`tdc_hydro.f90:189-203,
//!   282-288,339-345,415-428`).
//! - `TDC_hydro_use_mass_interp_face_values` is read only by
//!   `get_TDC_alfa_beta_face_weights`, which only the density forms call.
//! - `use_rsp_form_of_scale_height = .false.` in `get_scale_height_face`
//!   (`star/private/star_utils.f90:4002`).

use flexmesa_const::{PI, SQRT_2_DIV_3};
use flexmesa_turb::MltOption;

use crate::controls::StarError;
use crate::hydro_momentum::{wrap_d_00, wrap_d_m1, wrap_peos_00, wrap_peos_m1};
use crate::model::StarModel;
use crate::star_ad::{
    StarAdExt, StarAdOrder1, I_LNR_00, I_LNR_M1, I_LNR_P1, I_V_00, I_V_M1,
    I_V_P1,
};

/// `s% MLT_option` for the current step: with `steps_before_use_TDC > 0`
/// outside a relax, `do_step_part1` sets it to `TDC` once
/// `model_number >= steps_before_use_TDC` and to `Cox` before
/// (`star/private/evolve.f90:390-396`), tested before `model_number`
/// advances, as in `do1_mlt_2`.
#[must_use]
pub(crate) fn effective_mlt_option(model: &StarModel) -> MltOption {
    if !model.aux.doing_relax && model.controls.steps_before_use_tdc > 0 {
        if (model.model_number_old as isize)
            >= model.controls.steps_before_use_tdc
        {
            MltOption::Tdc
        } else {
            MltOption::Cox
        }
    } else {
        model.controls.mlt_option
    }
}

/// Upstream's `k > s% nz - s% TDC_num_innermost_cells_forced_nonturbulent`
/// for 0-based `k`.
fn in_forced_inner(model: &StarModel, k: usize) -> bool {
    k + 1
        + model
            .controls
            .mix_wind
            .tdc_num_innermost_cells_forced_nonturbulent
        > model.nz
}

/// Upstream's `k <= s% TDC_num_outermost_cells_forced_nonturbulent` for
/// 0-based `k`.
fn in_forced_outer(model: &StarModel, k: usize) -> bool {
    k < model
        .controls
        .mix_wind
        .tdc_num_outermost_cells_forced_nonturbulent
}

/// `ALFAM_ALFA` (`star/private/tdc_hydro.f90:166-171,249-254`).
fn alfam_alfa(model: &StarModel) -> f64 {
    if effective_mlt_option(model) == MltOption::Tdc
        && !model.controls.rsp2_flag
    {
        model.controls.tdc_alpha_m * model.controls.mixing_length_alpha
    } else {
        0.0
    }
}

/// `velocity time centering` as the `wrap_opt_time_center_*` wrappers test
/// it (`s% using_velocity_time_centering`,
/// `star/private/auto_diff_support.f90:1121-1256`).
fn time_centered(model: &StarModel) -> bool {
    model
        .controls
        .is_velocity_time_centered(model.model_number_old)
}

/// `s% r_start(k)`.
fn r_start(model: &StarModel, k: usize) -> f64 {
    flexmesa_math::exp(
        model.ln_r_start.get(k).copied().unwrap_or(model.ln_r[k]),
    )
}

/// `wrap_r_m1` (`star/private/auto_diff_support.f90:911-920`).
fn wrap_r_m1(model: &StarModel, k: usize) -> StarAdOrder1 {
    let mut out = StarAdOrder1::ZERO;
    if k > 0 {
        out.val = model.r[k - 1];
        out.d1[I_LNR_M1] = model.r[k - 1];
    }
    out
}

/// `wrap_r_00` (`star/private/auto_diff_support.f90:922-929`).
fn wrap_r_00(model: &StarModel, k: usize) -> StarAdOrder1 {
    let mut out = StarAdOrder1::ZERO;
    out.val = model.r[k];
    out.d1[I_LNR_00] = model.r[k];
    out
}

/// `wrap_r_p1` (`star/private/auto_diff_support.f90:931-942`).
fn wrap_r_p1(model: &StarModel, k: usize) -> StarAdOrder1 {
    let mut out = StarAdOrder1::ZERO;
    if k + 1 < model.nz {
        out.val = model.r[k + 1];
        out.d1[I_LNR_P1] = model.r[k + 1];
    } else {
        out.val = model.r_center;
    }
    out
}

/// `wrap_v_00` for `v_flag` (`star/private/auto_diff_support.f90:1090-1101`).
fn wrap_v_00(model: &StarModel, k: usize) -> StarAdOrder1 {
    StarAdOrder1::variable(model.v[k], I_V_00)
}

/// `wrap_v_m1` for `v_flag` (`star/private/auto_diff_support.f90:1075-1088`).
fn wrap_v_m1(model: &StarModel, k: usize) -> StarAdOrder1 {
    if k > 0 {
        StarAdOrder1::variable(model.v[k - 1], I_V_M1)
    } else {
        StarAdOrder1::ZERO
    }
}

/// `wrap_v_p1` for `v_flag` (`star/private/auto_diff_support.f90:1103-1119`).
fn wrap_v_p1(model: &StarModel, k: usize) -> StarAdOrder1 {
    if k + 1 < model.nz {
        StarAdOrder1::variable(model.v[k + 1], I_V_P1)
    } else {
        StarAdOrder1::constant(model.v_center)
    }
}

/// `wrap_u_00` (`star/private/auto_diff_support.f90:1215-1222`).
fn wrap_u_00(model: &StarModel, k: usize) -> StarAdOrder1 {
    StarAdOrder1::variable(model.u[k], I_V_00)
}

/// `wrap_u_m1` (`star/private/auto_diff_support.f90:1204-1213`).
fn wrap_u_m1(model: &StarModel, k: usize) -> StarAdOrder1 {
    if k > 0 {
        StarAdOrder1::variable(model.u[k - 1], I_V_M1)
    } else {
        StarAdOrder1::ZERO
    }
}

/// `wrap_opt_time_center_r_m1` (`auto_diff_support.f90:1121-1129`).
fn otc_r_m1(model: &StarModel, k: usize) -> StarAdOrder1 {
    let r_tc = wrap_r_m1(model, k);
    if time_centered(model) && k > 0 {
        0.5 * (r_tc + r_start(model, k - 1))
    } else {
        r_tc
    }
}

/// `wrap_opt_time_center_r_00` (`auto_diff_support.f90:1131-1138`).
fn otc_r_00(model: &StarModel, k: usize) -> StarAdOrder1 {
    let r_tc = wrap_r_00(model, k);
    if time_centered(model) {
        0.5 * (r_tc + r_start(model, k))
    } else {
        r_tc
    }
}

/// `wrap_opt_time_center_r_p1` (`auto_diff_support.f90:1140-1152`).
fn otc_r_p1(model: &StarModel, k: usize) -> StarAdOrder1 {
    let r_tc = wrap_r_p1(model, k);
    if !time_centered(model) {
        r_tc
    } else if k + 1 < model.nz {
        0.5 * (r_tc + r_start(model, k + 1))
    } else {
        0.5 * (r_tc + model.r_center)
    }
}

/// `wrap_opt_time_center_v_m1` for `v_flag` (`auto_diff_support.f90:1154-1169`).
fn otc_v_m1(model: &StarModel, k: usize) -> StarAdOrder1 {
    if k == 0 {
        return StarAdOrder1::ZERO;
    }
    let v_tc = wrap_v_m1(model, k);
    if time_centered(model) {
        0.5 * (v_tc + model.v_start[k - 1])
    } else {
        v_tc
    }
}

/// `wrap_opt_time_center_v_00` for `v_flag` (`auto_diff_support.f90:1171-1185`).
fn otc_v_00(model: &StarModel, k: usize) -> StarAdOrder1 {
    let v_tc = wrap_v_00(model, k);
    if time_centered(model) {
        0.5 * (v_tc + model.v_start[k])
    } else {
        v_tc
    }
}

/// `wrap_opt_time_center_v_p1` for `v_flag` (`auto_diff_support.f90:1187-1202`):
/// zero at the innermost cell, not `v_center`.
fn otc_v_p1(model: &StarModel, k: usize) -> StarAdOrder1 {
    if k + 1 == model.nz {
        return StarAdOrder1::ZERO;
    }
    let v_tc = wrap_v_p1(model, k);
    if time_centered(model) {
        0.5 * (v_tc + model.v_start[k + 1])
    } else {
        v_tc
    }
}

/// `wrap_opt_time_center_u_m1` (`auto_diff_support.f90:1238-1247`).
fn otc_u_m1(model: &StarModel, k: usize) -> StarAdOrder1 {
    if k == 0 {
        return StarAdOrder1::ZERO;
    }
    let v_tc = wrap_u_m1(model, k);
    if time_centered(model) {
        0.5 * (v_tc + model.u_start[k - 1])
    } else {
        v_tc
    }
}

/// `wrap_opt_time_center_u_00` (`auto_diff_support.f90:1249-1256`).
fn otc_u_00(model: &StarModel, k: usize) -> StarAdOrder1 {
    let v_tc = wrap_u_00(model, k);
    if time_centered(model) {
        0.5 * (v_tc + model.u_start[k])
    } else {
        v_tc
    }
}

/// `get_TDC_alfa_beta_face_weights` (`star/private/tdc_hydro.f90:91-104`):
/// `face_value = alfa*cell_value(k) + beta*cell_value(k-1)`, with
/// `TDC_hydro_use_mass_interp_face_values = .false.` (an unparsed control),
/// so `alfa = beta = 0.5`. Upstream reads it only in the density forms of
/// `d(v/r)/dr`, which are not reproduced here.
///
/// # Errors
/// Returns [`StarError::InvalidInput`] at the surface face, where upstream
/// stops with `mesa_error` (`:96`).
pub fn get_tdc_alfa_beta_face_weights(
    k: usize,
) -> Result<(f64, f64), StarError> {
    if k == 0 {
        return Err(StarError::InvalidInput(String::from(
            "bad k==1 for get_TDC_alfa_beta_face_weights (star/private/tdc_hydro.f90:96)",
        )));
    }
    Ok((0.5, 0.5))
}

/// `get_rho_face` (`star/private/star_utils.f90:3785-3796`).
fn get_rho_face(model: &StarModel, k: usize) -> StarAdOrder1 {
    if k == 0 {
        return wrap_d_00(model, k);
    }
    let (alfa, beta) = model.get_face_weights(k);
    alfa * wrap_d_00(model, k) + beta * wrap_d_m1(model, k)
}

/// `get_Peos_face` (`star/private/star_utils.f90:3846-3857`).
fn get_peos_face(model: &StarModel, k: usize) -> StarAdOrder1 {
    if k == 0 {
        return wrap_peos_00(model, k);
    }
    let (alfa, beta) = model.get_face_weights(k);
    alfa * wrap_peos_00(model, k) + beta * wrap_peos_m1(model, k)
}

/// `get_scale_height_face` (`star/private/star_utils.f90:3993-4018`), with
/// `use_rsp_form_of_scale_height = .false.`.
pub(crate) fn get_scale_height_face(
    model: &StarModel,
    k: usize,
) -> StarAdOrder1 {
    let g = model.cgrav[k];
    let grav = g * model.m_grav[k] / wrap_r_00(model, k).pow2();
    let p = get_peos_face(model, k);
    let rho = get_rho_face(model, k);
    let mut scale_height = p / (grav * rho);
    if model.controls.mix_wind.alt_scale_height_flag {
        let scale_height2 = (p / g).sqrt() / rho;
        if scale_height2.val < scale_height.val {
            scale_height = scale_height2;
        }
    }
    scale_height
}

/// `wrap_Hp_cell` (`star/private/tdc_hydro.f90:107-118`), which is all of
/// `Hp_cell_for_Chi` (`:120-148` returns before its other branch). Upstream
/// tests `k+1 < s% nz` with 1-based `k`, so the cell next to the innermost
/// has no `p1` term; that is reproduced.
fn wrap_hp_cell(model: &StarModel, k: usize) -> StarAdOrder1 {
    let hp0 = get_scale_height_face(model, k);
    let hp1 = if k + 2 < model.nz {
        get_scale_height_face(model, k + 1).shift_p1()
    } else {
        StarAdOrder1::ZERO
    };
    0.5 * (hp0 + hp1)
}

/// `compute_d_v_div_r` for `v_flag` (`star/private/tdc_hydro.f90:449-473`).
fn compute_d_v_div_r(model: &StarModel, k: usize) -> StarAdOrder1 {
    let v_00 = wrap_v_00(model, k);
    let v_p1 = wrap_v_p1(model, k);
    let r_00 = wrap_r_00(model, k);
    let mut r_p1 = wrap_r_p1(model, k);
    if r_p1.val == 0.0 {
        r_p1 = StarAdOrder1::constant(1.0);
    }
    v_00 / r_00 - v_p1 / r_p1
}

/// `compute_d_v_div_r_face` (`star/private/tdc_hydro.f90:578-616`).
fn compute_d_v_div_r_face(model: &StarModel, k: usize) -> StarAdOrder1 {
    let (v_00, v_m1, mut r_00, mut r_m1) = if model.controls.v_flag {
        (
            0.5 * (wrap_v_00(model, k) + wrap_v_p1(model, k)),
            0.5 * (wrap_v_00(model, k) + wrap_v_m1(model, k)),
            0.5 * (wrap_r_00(model, k) + wrap_r_p1(model, k)),
            0.5 * (wrap_r_00(model, k) + wrap_r_m1(model, k)),
        )
    } else {
        (
            wrap_u_00(model, k),
            wrap_u_m1(model, k),
            wrap_r_00(model, k),
            wrap_r_m1(model, k),
        )
    };
    if r_00.val == 0.0 {
        r_00 = StarAdOrder1::constant(1.0);
    }
    if r_m1.val == 0.0 {
        r_m1 = StarAdOrder1::constant(1.0);
    }
    v_m1 / r_m1 - v_00 / r_00
}

/// `compute_d_v_div_r_opt_time_center_face`
/// (`star/private/tdc_hydro.f90:618-656`).
fn compute_d_v_div_r_opt_time_center_face(
    model: &StarModel,
    k: usize,
) -> StarAdOrder1 {
    let (v_00, v_m1, mut r_00, mut r_m1) = if model.controls.v_flag {
        (
            0.5 * (otc_v_00(model, k) + otc_v_p1(model, k)),
            0.5 * (otc_v_00(model, k) + otc_v_m1(model, k)),
            0.5 * (otc_r_00(model, k) + otc_r_p1(model, k)),
            0.5 * (otc_r_00(model, k) + otc_r_m1(model, k)),
        )
    } else {
        (
            otc_u_00(model, k),
            otc_u_m1(model, k),
            otc_r_00(model, k),
            otc_r_m1(model, k),
        )
    };
    if r_00.val == 0.0 {
        r_00 = StarAdOrder1::constant(1.0);
    }
    if r_m1.val == 0.0 {
        r_m1 = StarAdOrder1::constant(1.0);
    }
    v_m1 / r_m1 - v_00 / r_00
}

/// `w = mlt_vc_ad(k)/sqrt_2_div_3` at face `k`
/// (`star/private/tdc_hydro.f90:286-287,343-344`).
fn w_face(model: &StarModel, k: usize) -> StarAdOrder1 {
    model.mlt_vc_ad[k] / SQRT_2_DIV_3
}

/// `0.5*(dm(k) + dm(k-1))`, or `0.5*dm(k)` at the surface face
/// (`star/private/tdc_hydro.f90:261-265,333-337`).
fn dmbar(model: &StarModel, k: usize) -> f64 {
    if k >= 1 {
        0.5 * (model.dm[k] + model.dm[k - 1])
    } else {
        0.5 * model.dm[k]
    }
}

/// `compute_Chi_cell` for `v_flag` (`star/private/tdc_hydro.f90:150-231`):
/// cell eddy-viscosity energy in $\mathrm{erg}$. Upstream stores nothing
/// here.
fn compute_chi_cell(model: &StarModel, k: usize) -> StarAdOrder1 {
    let alfam_alfa = alfam_alfa(model);
    if alfam_alfa == 0.0
        || in_forced_outer(model, k)
        || in_forced_inner(model, k)
    {
        return StarAdOrder1::ZERO;
    }
    let hp_cell = wrap_hp_cell(model, k);
    let d_v_div_r = compute_d_v_div_r(model, k);
    let w_00 = if k + 1 < model.nz {
        0.5 * (model.mlt_vc_ad[k] + model.mlt_vc_ad[k + 1].shift_p1())
            / SQRT_2_DIV_3
    } else {
        0.5 * model.mlt_vc_ad[k] / SQRT_2_DIV_3
    };
    let d_00 = wrap_d_00(model, k);
    let f = (16.0 / 3.0) * PI * alfam_alfa / model.dm[k];
    let rho2 = d_00.pow2();
    let r_00 = wrap_r_00(model, k);
    let r_p1 = wrap_r_p1(model, k);
    let r6_cell = 0.5 * (r_00.pow6() + r_p1.pow6());
    f * rho2 * r6_cell * d_v_div_r * hp_cell * w_00
}

/// `compute_Chi_div_w_face` (`star/private/tdc_hydro.f90:234-309`): face
/// eddy-viscosity energy divided by the convective velocity,
/// $\mathrm{erg\,s\,cm^{-1}}$. Also returns `Chi_face*w_00`, which upstream
/// stores in `s% Chi_ad(k)` (and its value in `s% Chi(k)`).
fn compute_chi_div_w_face(
    model: &StarModel,
    k: usize,
) -> (StarAdOrder1, StarAdOrder1) {
    let alfam_alfa = alfam_alfa(model);
    let chi_face = if alfam_alfa == 0.0 || in_forced_inner(model, k) {
        StarAdOrder1::ZERO
    } else {
        let hp_face = get_scale_height_face(model, k);
        let d_v_div_r = compute_d_v_div_r_face(model, k);
        let dmbar = dmbar(model, k);
        let d_00 = get_rho_face(model, k);
        let f = (16.0 / 3.0) * PI * alfam_alfa / dmbar;
        let rho2 = d_00.pow2();
        let r6_face = wrap_r_00(model, k).pow6();
        f * rho2 * r6_face * d_v_div_r * hp_face
    };
    let chi_out = chi_face * w_face(model, k);
    (chi_face, chi_out)
}

/// `compute_tdc_Eq_div_w_face` (`star/private/tdc_hydro.f90:311-355`):
/// turbulent dissipation divided by the convective velocity at face `k`,
/// $\mathrm{erg\,g^{-1}\,s^{-1}}\,(\mathrm{cm^{-1}\,s})$. Also returns
/// `Eq_face*w_00`, which upstream stores in `s% Eq_ad(k)` (and its value
/// in `s% Eq(k)`) for output.
#[must_use]
pub fn compute_tdc_eq_div_w_face(
    model: &StarModel,
    k: usize,
) -> (StarAdOrder1, StarAdOrder1) {
    let eq_face = if model.controls.mixing_length_alpha == 0.0
        || in_forced_inner(model, k)
    {
        StarAdOrder1::ZERO
    } else {
        let (chi_face, _) = compute_chi_div_w_face(model, k);
        let d_v_div_r = compute_d_v_div_r_opt_time_center_face(model, k);
        let dmbar = dmbar(model, k);
        4.0 * PI * chi_face * d_v_div_r / dmbar
    };
    let eq_out = eq_face * w_face(model, k);
    (eq_face, eq_out)
}

/// `compute_tdc_Uq_face` for `v_flag` (`star/private/tdc_hydro.f90:358-392`):
/// turbulent viscous acceleration at face `k`, $\mathrm{cm\,s^{-2}}$.
#[must_use]
pub fn compute_tdc_uq_face(model: &StarModel, k: usize) -> StarAdOrder1 {
    if model.controls.mixing_length_alpha == 0.0
        || in_forced_outer(model, k)
        || in_forced_inner(model, k)
    {
        return StarAdOrder1::ZERO;
    }
    let r_00 = otc_r_00(model, k);
    let chi_00 = compute_chi_cell(model, k);
    let chi_m1 = if k > 0 {
        compute_chi_cell(model, k - 1).shift_m1()
    } else {
        StarAdOrder1::ZERO
    };
    4.0 * PI * (chi_m1 - chi_00) / (r_00 * model.dm_bar[k])
}

/// `compute_tdc_Uq_dm_cell` for `u_flag` (`star/private/tdc_hydro.f90:395-445`):
/// `Uq*dm` in cell `k`, a force in dyn; the momentum equation divides by
/// `dm` (`star/private/hydro_riemann.f90:127-136`).
#[must_use]
pub fn compute_tdc_uq_dm_cell(model: &StarModel, k: usize) -> StarAdOrder1 {
    if model.controls.mixing_length_alpha == 0.0
        || in_forced_outer(model, k)
        || in_forced_inner(model, k)
    {
        return StarAdOrder1::ZERO;
    }
    let r_00 = otc_r_00(model, k);
    let r_p1 = otc_r_p1(model, k);
    let r_cell = 0.5 * (r_00 + r_p1);
    let w_00 = w_face(model, k);
    let chi_00 = compute_chi_div_w_face(model, k).0 * w_00;
    let chi_p1 = if k + 1 < model.nz {
        let w_p1 = model.mlt_vc_ad[k + 1].shift_p1() / SQRT_2_DIV_3;
        compute_chi_div_w_face(model, k + 1).0.shift_p1() * w_p1
    } else {
        StarAdOrder1::ZERO
    };
    4.0 * PI * (chi_00 - chi_p1) / r_cell
}

/// `Eq_ad` of the energy equation's TDC branch
/// (`star/private/hydro_energy.f90:268-277`):
/// `0.5*(Eq_div_w(k)*mlt_vc_ad(k) + shift_p1(Eq_div_w(k+1))*shift_p1(mlt_vc_ad(k+1)))/sqrt_2_div_3`,
/// with only the first term at the innermost cell.
#[must_use]
pub fn tdc_energy_eq_ad(model: &StarModel, k: usize) -> StarAdOrder1 {
    let eq_00 = compute_tdc_eq_div_w_face(model, k).0;
    if k + 1 < model.nz {
        let eq_p1 = compute_tdc_eq_div_w_face(model, k + 1).0.shift_p1();
        0.5 * (eq_00 * model.mlt_vc_ad[k]
            + eq_p1 * model.mlt_vc_ad[k + 1].shift_p1())
            / SQRT_2_DIV_3
    } else {
        0.5 * eq_00 * model.mlt_vc_ad[k] / SQRT_2_DIV_3
    }
}

/// `get_scale_height_face_val` (`star/private/star_utils.f90:4021-4046`),
/// with `use_rsp_form_of_scale_height = .false.`.
fn get_scale_height_face_val(model: &StarModel, k: usize) -> f64 {
    let g = model.cgrav[k];
    let grav = g * model.m_grav[k] / wrap_r_00(model, k).pow2();
    let p = get_peos_face(model, k).val;
    let rho = get_rho_face(model, k).val;
    let mut scale_height = p / (grav.val * rho);
    if model.controls.mix_wind.alt_scale_height_flag {
        let scale_height2 = (p / g).sqrt() / rho;
        if scale_height2 < scale_height {
            scale_height = scale_height2;
        }
    }
    scale_height
}

/// `set_viscosity_vars_TDC` (`star/private/tdc_hydro.f90:40-89`), called
/// from `set_hydro_vars` when `TDC_alpha_M > 0`, `MLT_option = 'TDC'` and
/// neither `RSP2_flag` nor `RSP_flag` (`star/private/hydro_vars.f90:602-610`).
///
/// It fills `Hp_face` where it is not positive, and the output arrays
/// `Chi`, `Eq`, `Eq_ad` and `Uq`. The equations do not read them for TDC
/// (they call the `compute_*` routines directly); `Chi_ad` has no store in
/// this port and is not kept.
pub(crate) fn set_viscosity_vars_tdc(
    model: &mut StarModel,
) -> Result<(), StarError> {
    let nz = model.nz;
    model.aux.rsp.resize(nz);
    if model.aux.eq_ad.len() != nz {
        model.aux.eq_ad.resize(nz, StarAdOrder1::ZERO);
    }
    if !(model.controls.v_flag || model.controls.u_flag) {
        for k in 0..nz {
            model.aux.rsp.eq_rsp[k] = 0.0;
            model.aux.eq_ad[k] = StarAdOrder1::ZERO;
            model.aux.rsp.chi_rsp[k] = 0.0;
            model.aux.rsp.uq_rsp[k] = 0.0;
        }
        return Ok(());
    }
    if model.mlt_vc_ad.len() != nz {
        return Err(StarError::MicrophysicsError(format!(
            "set_viscosity_vars_TDC: mlt_vc_ad has {} entries for nz = {nz}",
            model.mlt_vc_ad.len()
        )));
    }
    for k in 0..nz {
        if model.aux.rsp.hp_face[k] <= 0.0 {
            model.aux.rsp.hp_face[k] = get_scale_height_face_val(model, k);
        }
    }
    for k in 0..nz {
        let (_, chi_out) = compute_chi_div_w_face(model, k);
        model.aux.rsp.chi_rsp[k] = chi_out.val;
        let (_, eq_out) = compute_tdc_eq_div_w_face(model, k);
        model.aux.rsp.eq_rsp[k] = eq_out.val;
        model.aux.eq_ad[k] = eq_out;
        if model.controls.v_flag {
            model.aux.rsp.uq_rsp[k] = compute_tdc_uq_face(model, k).val;
        } else if model.controls.u_flag {
            model.aux.rsp.uq_rsp[k] =
                compute_tdc_uq_dm_cell(model, k).val / model.dm[k];
        }
    }
    Ok(())
}
