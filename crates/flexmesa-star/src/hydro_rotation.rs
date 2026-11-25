// SPDX-License-Identifier: GPL-3.0-or-later
//! Roche-lobe rotational distortion helpers (`w_div_w_roche_omega`,
//! `w_div_w_roche_jrot`, `eval_i_rot`, and the Fabry, Marchant & Sana 2022
//! polynomial fits `rpsi_from_re_factor`, `re_from_rpsi_factor`, `A`, `B`, `C`,
//! and `sigmoid`) matching `star/private/hydro_rotation.f90:51-226, 874-920`.

use flexmesa_autodiff::{AutoDiffReal1VarOrder1, AutoDiffReal2VarOrder1};
use flexmesa_const::{FOUR_PI, ONE_SIXTH, ONE_THIRD, TWO_THIRDS};

use crate::model::StarModel;

/// Exponent in `log(1 - w^5.626)` (`log_term_power` in `hydro_rotation.f90:51`).
pub const LOG_TERM_POWER: f64 = 5.626;

/// Upstream default for `w_div_wcrit_max2` (`star/defaults/controls.defaults:4188`).
pub const DEFAULT_W_DIV_WCRIT_MAX2: f64 = 0.90;

/// Volume-equivalent radius ratio $r_\psi / r_e$ (`rpsi_from_re_factor` in
/// `star/private/hydro_rotation.f90:874-880`; Fabry et al. 2022, Eq. A.2).
#[must_use]
#[inline]
pub fn rpsi_from_re_factor(o2: f64, o4: f64, o6: f64) -> f64 {
    1.0 - ONE_SIXTH * o2 + 0.02025 * o4 - 0.03870 * o6
}

/// Equatorial radius ratio $r_e / r_\psi$ (`re_from_rpsi_factor` in
/// `star/private/hydro_rotation.f90:882-888`; Fabry et al. 2022, Eq. A.3).
#[must_use]
#[inline]
pub fn re_from_rpsi_factor(o2: f64, o4: f64, o6: f64) -> f64 {
    1.0 + ONE_SIXTH * o2 - 0.005124 * o4 + 0.06562 * o6
}

/// Roche surface-integral factor `A` (`star/private/hydro_rotation.f90:890-896`;
/// Fabry et al. 2022, Eq. A.6).
#[must_use]
#[inline]
pub fn roche_a(o4: f64, o6: f64, o_log_term: f64) -> f64 {
    1.0 + 0.3293 * o4 - 0.4926 * o6 - 0.5560 * o_log_term
}

/// Roche moment-of-inertia factor `B` (`star/private/hydro_rotation.f90:898-905`;
/// Fabry et al. 2022, Eq. A.7).
#[must_use]
#[inline]
pub fn roche_b(o2: f64, o4: f64, o6: f64, o_log_term: f64) -> f64 {
    1.0 + o2 / 5.0 + 0.4140 * o4 - 0.8650 * o6 - 0.8370 * o_log_term
}

/// Roche specific-angular-momentum factor `C`
/// (`star/private/hydro_rotation.f90:907-914`; Fabry et al. 2022, Eq. A.12).
#[must_use]
#[inline]
pub fn roche_c(o2: f64, o4: f64, o6: f64, o_log_term: f64) -> f64 {
    1.0 + (17.0 / 60.0) * o2 + 0.4010 * o4 - 0.8606 * o6 - 0.9305 * o_log_term
}

/// Smoothly maps $|x| \in [x_{\max 1}, \infty)$ onto $[x_{\max 1}, x_{\max 2})$
/// (`sigmoid` in `star/private/hydro_rotation.f90:916-920`).
#[must_use]
#[inline]
pub fn roche_sigmoid(x: f64, xmax1: f64, xmax2: f64) -> f64 {
    2.0 * (xmax2 - xmax1)
        / (1.0 + flexmesa_math::exp(-2.0 * (x.abs() - xmax1) / (xmax2 - xmax1)))
        - xmax2
        + 2.0 * xmax1
}

/// Computes `w_div_w_roche` for a known angular frequency `omega`
/// ($\mathrm{rad\,s^{-1}}$), volume-equivalent radius `rpsi` ($\mathrm{cm}$),
/// and enclosed mass `m_phi` ($\mathrm{g}$) matching `w_div_w_roche_omega` in
/// `star/private/hydro_rotation.f90:55-108`.
#[must_use]
pub fn w_div_w_roche_omega(
    rpsi: f64,
    m_phi: f64,
    omega: f64,
    cgrav: f64,
    max_w: f64,
    max_w2: f64,
    w_div_wc_flag: bool,
) -> f64 {
    if omega == 0.0 {
        return 0.0;
    }

    let mut dimless_rpsi = rpsi * flexmesa_math::pow(omega.abs(), TWO_THIRDS)
        / flexmesa_math::pow(cgrav * m_phi, ONE_THIRD);

    let mut wr;
    if w_div_wc_flag {
        wr = max_w;
        let rpsi_lim1 = flexmesa_math::pow(wr, TWO_THIRDS)
            * rpsi_from_re_factor(
                flexmesa_math::pow2(wr),
                flexmesa_math::pow4(wr),
                flexmesa_math::pow6(wr),
            );

        wr = max_w2;
        let rpsi_lim2 = flexmesa_math::pow(wr, TWO_THIRDS)
            * rpsi_from_re_factor(
                flexmesa_math::pow2(wr),
                flexmesa_math::pow4(wr),
                flexmesa_math::pow6(wr),
            );

        if dimless_rpsi.abs() > rpsi_lim1 {
            dimless_rpsi =
                roche_sigmoid(dimless_rpsi.abs(), rpsi_lim1, rpsi_lim2);
        }
    } else {
        wr = max_w;
        let new_dimless_rpsi = flexmesa_math::pow(wr, TWO_THIRDS)
            * rpsi_from_re_factor(
                flexmesa_math::pow2(wr),
                flexmesa_math::pow4(wr),
                flexmesa_math::pow6(wr),
            );
        if dimless_rpsi > new_dimless_rpsi {
            return wr;
        }
    }

    let mut wr_high = wr;
    let mut wr_low = 0.0_f64;
    while wr_high - wr_low > 1.0e-6 {
        wr = 0.5 * (wr_high + wr_low);
        let new_dimless_rpsi = flexmesa_math::pow(wr, TWO_THIRDS)
            * rpsi_from_re_factor(
                flexmesa_math::pow2(wr),
                flexmesa_math::pow4(wr),
                flexmesa_math::pow6(wr),
            );
        if dimless_rpsi > new_dimless_rpsi {
            wr_low = wr;
        } else {
            wr_high = wr;
        }
    }
    let mut w_roche = 0.5 * (wr_high + wr_low);
    if omega < 0.0 {
        w_roche = -w_roche;
    }
    w_roche
}

/// Computes `w_div_w_roche` for a known specific angular momentum `jrot`
/// ($\mathrm{cm^2\,s^{-1}}$), volume-equivalent radius `rpsi` ($\mathrm{cm}$),
/// and enclosed mass `m_phi` ($\mathrm{g}$) matching `w_div_w_roche_jrot` in
/// `star/private/hydro_rotation.f90:111-179`.
#[must_use]
pub fn w_div_w_roche_jrot(
    rpsi: f64,
    m_phi: f64,
    jrot: f64,
    cgrav: f64,
    max_w: f64,
    max_w2: f64,
    w_div_wc_flag: bool,
) -> f64 {
    if jrot == 0.0 {
        return 0.0;
    }

    let mut dimless_factor = jrot.abs() / (cgrav * m_phi * rpsi).sqrt();

    let mut wr;
    if w_div_wc_flag {
        wr = max_w;
        let w4 = flexmesa_math::pow4(wr);
        let w6 = flexmesa_math::pow6(wr);
        let w_log_term =
            flexmesa_math::log(1.0 - flexmesa_math::pow(wr, LOG_TERM_POWER));
        let jr_lim1 = TWO_THIRDS
            * wr
            * roche_c(flexmesa_math::pow2(wr), w4, w6, w_log_term)
            / roche_a(w4, w6, w_log_term);

        wr = max_w2;
        let w4 = flexmesa_math::pow4(wr);
        let w6 = flexmesa_math::pow6(wr);
        let w_log_term =
            flexmesa_math::log(1.0 - flexmesa_math::pow(wr, LOG_TERM_POWER));
        let jr_lim2 = TWO_THIRDS
            * wr
            * roche_c(flexmesa_math::pow2(wr), w4, w6, w_log_term)
            / roche_a(w4, w6, w_log_term);

        if dimless_factor.abs() > jr_lim1 {
            dimless_factor =
                roche_sigmoid(dimless_factor.abs(), jr_lim1, jr_lim2);
        }
    } else {
        wr = max_w;
        let w2 = flexmesa_math::pow2(wr);
        let w4 = flexmesa_math::pow4(wr);
        let w6 = flexmesa_math::pow6(wr);
        let w_log_term =
            flexmesa_math::log(1.0 - flexmesa_math::pow(wr, LOG_TERM_POWER));
        let new_dimless_factor =
            TWO_THIRDS * wr * roche_c(w2, w4, w6, w_log_term)
                / roche_a(w4, w6, w_log_term);
        if dimless_factor > new_dimless_factor {
            return wr;
        }
    }

    let mut wr_high = wr;
    let mut wr_low = 0.0_f64;
    while wr_high - wr_low > 1.0e-6 {
        wr = 0.5 * (wr_high + wr_low);
        let w2 = flexmesa_math::pow2(wr);
        let w4 = flexmesa_math::pow4(wr);
        let w6 = flexmesa_math::pow6(wr);
        let w_log_term =
            flexmesa_math::log(1.0 - flexmesa_math::pow(wr, LOG_TERM_POWER));
        let new_dimless_factor =
            TWO_THIRDS * wr * roche_c(w2, w4, w6, w_log_term)
                / roche_a(w4, w6, w_log_term);
        if dimless_factor > new_dimless_factor {
            wr_low = wr;
        } else {
            wr_high = wr;
        }
    }
    let mut w_roche = 0.5 * (wr_high + wr_low);
    if jrot < 0.0 {
        w_roche = -w_roche;
    }
    w_roche
}

/// Evaluates the shell specific moment of inertia $i_{\mathrm{rot}}$
/// ($\mathrm{cm^2}$) together with its exact first-order partial derivatives
/// $\partial i_{\mathrm{rot}} / \partial (\omega/\omega_{\mathrm{crit}})$
/// (`d1[0]`, `i_w_div_wc_00`) and $\partial i_{\mathrm{rot}} / \partial \ln r$
/// (`d1[1]`, `i_lnR_00`), mirroring `eval_i_rot` in
/// `star/private/hydro_rotation.f90:181-226` using [`AutoDiffReal2VarOrder1`]
/// so division via `powm1` matches upstream bit-for-bit.
#[must_use]
pub fn eval_i_rot_dual(
    r00: f64,
    w_div_w_crit_roche: f64,
    simple_i_rot_flag: bool,
) -> AutoDiffReal2VarOrder1 {
    if simple_i_rot_flag {
        let val = TWO_THIRDS * r00 * r00;
        AutoDiffReal2VarOrder1::new(val, [0.0, 2.0 * val])
    } else {
        // Compute i_rot following Paxton et al. 2019 (ApJS, 243, 10) in
        // upstream's exact `auto_diff_real_2var_order1` operation order
        // (`star/private/hydro_rotation.f90:204-224`).
        let w = AutoDiffReal2VarOrder1::new(w_div_w_crit_roche, [1.0, 0.0]);
        let r = AutoDiffReal2VarOrder1::new(r00, [0.0, r00]);
        let w2 = w.pow2();
        let w4 = w.pow4();
        let w6 = w.pow6();
        let lg_one_sub_w4 = (1.0 - w4).log();
        let re = r * (1.0 + w2 / 6.0 - 0.0002507 * w4 + 0.06075 * w6);
        let b = 1.0 + w2 / 5.0
            - 0.2735 * w4
            - 0.4327 * w6
            - ((3.0 / 2.0) * 0.5583) * lg_one_sub_w4;
        let a = 1.0 - 0.1076 * w4 - 0.2336 * w6 - 0.5583 * lg_one_sub_w4;
        TWO_THIRDS * re.pow2() * b / a
    }
}

/// Evaluates the shell specific moment of inertia $i_{\mathrm{rot}}$
/// ($\mathrm{cm^2}$) following Paxton et al. (2019, ApJS 243, 10), matching
/// `eval_i_rot` in `star/private/hydro_rotation.f90:181-226`.
#[must_use]
pub fn eval_i_rot(
    r00: f64,
    w_div_w_crit_roche: f64,
    simple_i_rot_flag: bool,
) -> f64 {
    eval_i_rot_dual(r00, w_div_w_crit_roche, simple_i_rot_flag).val
}

/// Evaluates single-star Roche distortion factors $(f_P, f_T)$ and their
/// derivatives with respect to $|\omega/\omega_{\mathrm{crit}}|$
/// (`d1[0]`, `i_w_div_wc_00`), matching `eval_fp_ft` in
/// `star/private/hydro_rotation.f90:775-804` using [`AutoDiffReal1VarOrder1`]
/// (Fabry, Marchant & Sana 2022, Eqs. A.10 and A.11).
#[must_use]
pub fn eval_fp_ft_single_dual(
    w_div_w_crit_roche: f64,
) -> (AutoDiffReal1VarOrder1, AutoDiffReal1VarOrder1) {
    let w = AutoDiffReal1VarOrder1::new(w_div_w_crit_roche.abs(), [1.0]);
    let w2 = w.pow2();
    let w4 = w.pow4();
    let w6 = w.pow6();
    let w_log_term = (1.0 - w.powf(LOG_TERM_POWER)).log();
    let a_omega = 1.0 + 0.3293 * w4 - 0.4926 * w6 - 0.5560 * w_log_term;
    let fp = (1.0 - TWO_THIRDS * w2 - 0.2133 * w4 - 0.1068 * w6) / a_omega;
    let ft = (1.0 - 0.07955 * w4 - 0.2322 * w6) / a_omega;
    (fp, ft)
}

/// Evaluates single-star Roche distortion factors $(f_P, f_T)$ without
/// artificial floors (`eval_fp_ft` in `star/private/hydro_rotation.f90:775-788`;
/// Fabry, Marchant & Sana 2022, Eqs. A.10 and A.11).
#[must_use]
pub fn eval_fp_ft_single(w_div_w_crit_roche: f64) -> (f64, f64) {
    let (fp, ft) = eval_fp_ft_single_dual(w_div_w_crit_roche);
    (fp.val, ft.val)
}

/// Evaluates single-star Roche distortion factors and deformed radii
/// `(fp_rot, ft_rot, r_polar, r_equatorial)` matching `eval_fp_ft` in
/// `star/private/hydro_rotation.f90:748-806` (Fabry, Marchant & Sana 2022).
#[must_use]
pub fn eval_fp_ft(r: f64, w_div_w_crit_roche: f64) -> (f64, f64, f64, f64) {
    let w = w_div_w_crit_roche.abs();
    let w2 = w * w;
    let w4 = w2 * w2;
    let w6 = w4 * w2;
    let (fp, ft) = eval_fp_ft_single_dual(w);
    let r_equatorial = (r * re_from_rpsi_factor(w2, w4, w6)).max(r);
    let r_polar = (r_equatorial / (1.0 + 0.5 * w2)).min(r);
    (fp.val, ft.val, r_polar, r_equatorial)
}

/// Updates `model.aux.i_rot[k]` and `model.aux.d_i_rot_dlnr[k]` at face `k`
/// for volume-equivalent radius `r` ($\mathrm{cm}$), blending optional binary
/// tidal deformation (`set1_i_rot` in `star/private/hydro_rotation.f90:317-336`).
pub fn set1_i_rot(model: &mut StarModel, k: usize, r: f64) {
    let w = model.aux.w_div_wc.get(k).copied().unwrap_or(0.0);
    let ir_single =
        eval_i_rot_dual(r, w, model.controls.mix_wind.simple_i_rot_flag);
    if let Some(ref hook) = model.aux.tidal_deformation {
        let td = hook.0.eval_shell(k, r, model.omega[k]);
        let f_sw = td.f_switch;
        let f_1m = 1.0 - f_sw;
        model.aux.i_rot[k] = f_sw * td.i_rot + f_1m * ir_single.val;
        model.aux.d_i_rot_dlnr[k] =
            f_sw * td.d_irot_dlnr + f_1m * ir_single.d1[1];
    } else {
        model.aux.i_rot[k] = ir_single.val;
        model.aux.d_i_rot_dlnr[k] = ir_single.d1[1];
    }
}

/// Updates `model.aux.i_rot[k]` using `r00 = exp(ln_r[k])`
/// (`update1_i_rot_from_xh` in `star/private/hydro_rotation.f90:308-315` and
/// `get_r_from_xh` in `star/private/star_utils.f90:429-440`).
pub fn update1_i_rot_from_xh(model: &mut StarModel, k: usize) {
    let r00 = flexmesa_math::exp(model.ln_r[k]);
    set1_i_rot(model, k, r00);
}

/// Updates `w_div_wc` (unless `skip_w_div_w_crit_roche`) from `j_rot` via
/// [`w_div_w_roche_jrot`] and recomputes `i_rot` across all faces
/// (`set_i_rot` in `star/private/hydro_rotation.f90:229-245`).
pub fn set_i_rot(model: &mut StarModel, skip_w_div_w_crit_roche: bool) {
    let nz = model.nz;
    if nz == 0 {
        return;
    }
    if model.aux.w_div_wc.len() != nz
        || model.aux.i_rot.len() != nz
        || model.aux.d_i_rot_dlnr.len() != nz
    {
        model.aux.resize(nz, model.species);
    }
    let max_w = model.controls.mix_wind.w_div_wcrit_max;
    let max_w2 = model.controls.mix_wind.w_div_wcrit_max2;
    for k in 0..nz {
        if !skip_w_div_w_crit_roche {
            model.aux.w_div_wc[k] = w_div_w_roche_jrot(
                model.r[k],
                model.m[k],
                model.j_rot[k],
                model.cgrav[k],
                max_w,
                max_w2,
                false,
            );
        }
        let r_k = model.r[k];
        set1_i_rot(model, k, r_k);
    }
}

/// Computes `i_rot` from `j_rot / omega` when `omega != 0`, falling back to
/// [`update1_i_rot_from_xh`] at non-rotating faces
/// (`set_i_rot_from_omega_and_j_rot` in
/// `star/private/hydro_rotation.f90:247-264`).
pub fn set_i_rot_from_omega_and_j_rot(model: &mut StarModel) {
    let nz = model.nz;
    if nz == 0 {
        return;
    }
    if model.aux.i_rot.len() != nz || model.aux.d_i_rot_dlnr.len() != nz {
        model.aux.resize(nz, model.species);
    }
    for k in 0..nz {
        if model.omega[k] == 0.0 {
            update1_i_rot_from_xh(model, k);
        } else {
            model.aux.i_rot[k] = model.j_rot[k] / model.omega[k];
            model.aux.d_i_rot_dlnr[k] = 0.0;
        }
    }
}

/// Sets specific angular momentum `j_rot[k] = i_rot[k] * omega[k]` across all
/// faces (`set_j_rot` in `star/private/hydro_rotation.f90:266-275`).
pub fn set_j_rot(model: &mut StarModel) {
    let n = model
        .nz
        .min(model.j_rot.len())
        .min(model.aux.i_rot.len())
        .min(model.omega.len());
    for k in 0..n {
        model.j_rot[k] = model.aux.i_rot[k] * model.omega[k];
    }
}

/// Sets face angular velocity `omega[k] = j_rot[k] / i_rot[k]` across all
/// faces (`set_omega` in `star/private/hydro_rotation.f90:277-287`).
pub fn set_omega(model: &mut StarModel) {
    let n = model
        .nz
        .min(model.omega.len())
        .min(model.j_rot.len())
        .min(model.aux.i_rot.len());
    for k in 0..n {
        model.omega[k] = model.j_rot[k] / model.aux.i_rot[k];
    }
}

/// Updates `w_div_wc` from `j_rot` (zero where `j_rot == 0`) and `i_rot`
/// from `ln_r` (`use_xh_to_update_i_rot` in
/// `star/private/hydro_rotation.f90:362-380`). `w_div_wc_flag` is `.false.`
/// on this path (`star/private/init.f90:531`).
pub fn use_xh_to_update_i_rot(model: &mut StarModel) {
    let nz = model.nz;
    if nz == 0 {
        return;
    }
    if model.aux.w_div_wc.len() != nz || model.aux.i_rot.len() != nz {
        model.aux.resize(nz, model.species);
    }
    let max_w = model.controls.mix_wind.w_div_wcrit_max;
    let max_w2 = model.controls.mix_wind.w_div_wcrit_max2;
    for k in 0..nz {
        if model.j_rot[k] == 0.0 {
            model.aux.w_div_wc[k] = 0.0;
        } else {
            let r00 = flexmesa_math::exp(model.ln_r[k]);
            model.aux.w_div_wc[k] = w_div_w_roche_jrot(
                r00,
                model.m[k],
                model.j_rot[k],
                model.cgrav[k],
                max_w,
                max_w2,
                false,
            );
        }
        update1_i_rot_from_xh(model, k);
    }
}

/// Updates `w_div_wc`, `i_rot`, and `j_rot = i_rot * omega` from `ln_r` and
/// `omega` (`use_xh_to_update_i_rot_and_j_rot` in
/// `star/private/hydro_rotation.f90:382-397`).
pub fn use_xh_to_update_i_rot_and_j_rot(model: &mut StarModel) {
    let nz = model.nz;
    if nz == 0 {
        return;
    }
    if model.aux.w_div_wc.len() != nz || model.aux.i_rot.len() != nz {
        model.aux.resize(nz, model.species);
    }
    let max_w = model.controls.mix_wind.w_div_wcrit_max;
    let max_w2 = model.controls.mix_wind.w_div_wcrit_max2;
    for k in 0..nz {
        let r00 = flexmesa_math::exp(model.ln_r[k]);
        model.aux.w_div_wc[k] = w_div_w_roche_omega(
            r00,
            model.m[k],
            model.omega[k],
            model.cgrav[k],
            max_w,
            max_w2,
            false,
        );
        update1_i_rot_from_xh(model, k);
    }
    set_j_rot(model);
}

pub(crate) fn apply_face_fp_ft(
    model: &mut StarModel,
    k: usize,
    r_k: f64,
    om_k: f64,
    tidal: Option<&crate::hydro_state::TidalDeformationHandle>,
) {
    let (fp_s, ft_s) = eval_fp_ft_single(model.aux.w_div_wc[k]);
    if let Some(hook) = tidal {
        let td = hook.0.eval_shell(k, r_k, om_k);
        let f_sw = td.f_switch;
        let f_1m = 1.0 - f_sw;
        model.aux.fp_rot[k] = f_sw * td.fp + f_1m * fp_s;
        model.aux.d_fp_rot_dlnr[k] = f_sw * td.d_fp_dlnr;
        model.aux.ft_rot[k] = f_sw * td.ft + f_1m * ft_s;
        model.aux.d_ft_rot_dlnr[k] = f_sw * td.d_ft_dlnr;
    } else {
        model.aux.fp_rot[k] = fp_s;
        model.aux.d_fp_rot_dlnr[k] = 0.0;
        model.aux.ft_rot[k] = ft_s;
        model.aux.d_ft_rot_dlnr[k] = 0.0;
    }
}

/// Updates `i_rot`, `omega` (from `j_rot`), and Roche distortion factors
/// `fp_rot` and `ft_rot` across all faces (`set_rotation_info` in
/// `star/private/hydro_rotation.f90:537-586`).
pub fn set_rotation_info(model: &mut StarModel, skip_w_div_w_crit_roche: bool) {
    let nz = model.nz;
    if !model.controls.rotation_flag || nz == 0 {
        return;
    }
    if model.aux.fp_rot.len() != nz
        || model.aux.d_fp_rot_dlnr.len() != nz
        || model.aux.ft_rot.len() != nz
        || model.aux.d_ft_rot_dlnr.len() != nz
    {
        model.aux.resize(nz, model.species);
    }
    set_i_rot(model, skip_w_div_w_crit_roche);
    set_omega(model);
    let tidal = model.aux.tidal_deformation.clone();
    for k in 0..nz {
        apply_face_fp_ft(model, k, model.r[k], model.omega[k], tidal.as_ref());
    }
}

/// Computes cell-centered angular momentum diffusion coefficients `am_sig`
/// ($\mathrm{g\,s^{-1}}$) from face viscosities `am_nu` ($\mathrm{cm^2\,s^{-1}}$)
/// over 0-based range `nzlo..nzhi`, applying upstream's `am_sig_term_limit`
/// stiffness bound without artificial caps (`get1_am_sig` in
/// `star/private/hydro_rotation.f90:437-490`).
pub fn get1_am_sig(
    model: &StarModel,
    nzlo: usize,
    nzhi: usize,
    am_nu: &[f64],
    am_sig: &mut [f64],
    dt: f64,
) {
    let nz = model.nz;
    if nz == 0 || nzlo >= nzhi || nzhi > nz {
        return;
    }
    let xmstar = model.xmstar;
    let sig_term_limit = model.controls.mix_wind.am_sig_term_limit;

    let mut nz2 = nzhi;
    if nzhi == nz {
        let k = nz - 1;
        let d_val = 0.0_f64;
        let r = 0.5 * model.r[k];
        let s1 = FOUR_PI * r * r * model.rho[k];
        am_sig[k] = s1 * s1 * d_val / model.dm[k];
        nz2 = nz - 1;
    }
    for k in nzlo..nz2 {
        let am_nu_e00 = am_nu[k].max(0.0);
        let am_nu_ep1 = am_nu[k + 1].max(0.0);
        // Meynet, Maeder, & Mowlavi, A&A 416, 1023-1036, 2004, eqn 51 with f = 1/2.
        let d_val = 2.0 * (am_nu_e00 * am_nu_ep1)
            / (am_nu_e00 + am_nu_ep1).max(1.0e-99);
        let r = 0.5 * (model.r[k] + model.r[k + 1]);
        let s1 = FOUR_PI * r * r * model.rho[k];
        am_sig[k] = s1 * s1 * d_val / model.dm[k];
    }

    for k in nzlo..nzhi {
        let dmbar = if k + 1 < nz {
            xmstar * model.dq[k].min(model.dq[k + 1])
        } else {
            xmstar * model.dq[k]
        };
        let siglim = sig_term_limit * dmbar / dt;
        if am_sig[k] > siglim {
            am_sig[k] = siglim;
        }
    }
}

/// Computes `model.aux.am_sig_j` and `model.aux.am_sig_omega` over 0-based
/// range `nzlo..nzhi`, limiting the sum `am_nu_j + am_nu_omega` before
/// splitting (`get_rotation_sigmas` in `star/private/hydro_rotation.f90:399-435`).
pub fn get_rotation_sigmas(
    model: &mut StarModel,
    nzlo: usize,
    nzhi: usize,
    dt: f64,
) {
    let nz = model.nz;
    if nz == 0 || nzlo >= nzhi || nzhi > nz {
        return;
    }
    if model.aux.am_nu_j.len() != nz
        || model.aux.am_nu_omega.len() != nz
        || model.aux.am_sig_j.len() != nz
        || model.aux.am_sig_omega.len() != nz
    {
        model.aux.resize(nz, model.species);
    }

    let am_nu_j = model.aux.am_nu_j.clone();
    let mut am_sig_j = model.aux.am_sig_j.clone();
    get1_am_sig(model, nzlo, nzhi, &am_nu_j, &mut am_sig_j, dt);

    let am_nu: Vec<f64> = (0..nz)
        .map(|k| model.aux.am_nu_j[k] + model.aux.am_nu_omega[k])
        .collect();
    let mut am_sig = vec![0.0_f64; nz];
    get1_am_sig(model, nzlo, nzhi, &am_nu, &mut am_sig, dt);

    for k in 0..nz {
        model.aux.am_sig_j[k] = am_sig_j[k];
        model.aux.am_sig_omega[k] = (am_sig[k] - am_sig_j[k]).max(0.0);
    }
}

/// Returns total stellar angular momentum $\sum_{k=0}^{\mathrm{nz}-1} \overline{\Delta m}_k j_{\mathrm{rot},k}$
/// in $\mathrm{g\,cm^2\,s^{-1}}$ (`total_angular_momentum` in
/// `star/private/star_utils.f90:1838-1846`).
#[must_use]
pub fn total_angular_momentum(model: &StarModel) -> f64 {
    if !model.controls.rotation_flag {
        return 0.0;
    }
    let mut j_tot = 0.0_f64;
    for k in 0..model.nz {
        j_tot += model.dm_bar[k] * model.j_rot[k];
    }
    j_tot
}

#[cfg(test)]
mod tests {
    use flexmesa_const::{M_SUN, R_SUN, STANDARD_CGRAV, TWO_THIRDS};

    use super::{eval_i_rot, w_div_w_roche_jrot, w_div_w_roche_omega};

    // Expected values below were evaluated independently in Python directly
    // from the upstream formulas in `star/private/hydro_rotation.f90:55-226,
    // 874-920` using MESA `26.4.1` `const_def.f90` constants:
    //   w = 0.6 -> j_exact = (2/3)*w*C/A * sqrt(G*Msun*Rsun)
    //                      = 1.3493170707419899e18 cm^2/s
    //   w = 0.6 -> om_exact = w * rpsi_from_re_factor(w)^(3/2) * sqrt(G*Msun/Rsun^3)
    //                       = 3.4374235810367103e-4 rad/s
    //   eval_i_rot(Rsun, 0.6, false) = 3.9215030601732740e21 cm^2

    #[test]
    fn test_w_div_w_roche_zero_inputs_return_exact_zero() {
        assert_eq!(
            w_div_w_roche_jrot(
                R_SUN,
                M_SUN,
                0.0,
                STANDARD_CGRAV,
                0.89,
                0.90,
                false
            ),
            0.0
        );
        assert_eq!(
            w_div_w_roche_omega(
                R_SUN,
                M_SUN,
                0.0,
                STANDARD_CGRAV,
                0.89,
                0.90,
                false
            ),
            0.0
        );
    }

    #[test]
    fn test_w_div_w_roche_bisection_matches_independent_upstream_formulas() {
        let j_06 = 1.349_317_070_741_989_9e18;
        let w_j = w_div_w_roche_jrot(
            R_SUN,
            M_SUN,
            j_06,
            STANDARD_CGRAV,
            0.89,
            0.90,
            false,
        );
        assert!(
            (w_j - 0.6).abs() <= 5.0e-7,
            "w_div_w_roche_jrot expected 0.6 within bisection tol 5e-7, got {w_j:.16e}"
        );
        // Sign symmetry for negative jrot below cap (`hydro_rotation.f90:175-177`).
        let w_j_neg = w_div_w_roche_jrot(
            R_SUN,
            M_SUN,
            -j_06,
            STANDARD_CGRAV,
            0.89,
            0.90,
            false,
        );
        assert_eq!(w_j_neg, -w_j);

        let om_06 = 3.437_423_581_036_710_3e-4;
        let w_om = w_div_w_roche_omega(
            R_SUN,
            M_SUN,
            om_06,
            STANDARD_CGRAV,
            0.89,
            0.90,
            false,
        );
        assert!(
            (w_om - 0.6).abs() <= 5.0e-7,
            "w_div_w_roche_omega expected 0.6 within bisection tol 5e-7, got {w_om:.16e}"
        );
        let w_om_neg = w_div_w_roche_omega(
            R_SUN,
            M_SUN,
            -om_06,
            STANDARD_CGRAV,
            0.89,
            0.90,
            false,
        );
        assert_eq!(w_om_neg, -w_om);
    }

    #[test]
    fn test_w_div_w_roche_hard_cap_and_sigmoid_asymptote_match_upstream() {
        // Hard cap when `w_div_wc_flag = false` returns exact `max_w`.
        assert_eq!(
            w_div_w_roche_jrot(
                R_SUN,
                M_SUN,
                1.0e22,
                STANDARD_CGRAV,
                0.89,
                0.90,
                false
            ),
            0.89
        );
        assert_eq!(
            w_div_w_roche_omega(
                R_SUN,
                M_SUN,
                1.0e-1,
                STANDARD_CGRAV,
                0.89,
                0.90,
                false
            ),
            0.89
        );

        // Smooth sigmoid cap when `w_div_wc_flag = true` approaches `max_w2 = 0.90`.
        let w_sig_j = w_div_w_roche_jrot(
            R_SUN,
            M_SUN,
            1.0e22,
            STANDARD_CGRAV,
            0.89,
            0.90,
            true,
        );
        assert!(
            (w_sig_j - 0.90).abs() <= 1.0e-6,
            "sigmoid jrot asymptote expected 0.90, got {w_sig_j:.16e}"
        );
        let w_sig_om = w_div_w_roche_omega(
            R_SUN,
            M_SUN,
            1.0e-1,
            STANDARD_CGRAV,
            0.89,
            0.90,
            true,
        );
        assert!(
            (w_sig_om - 0.90).abs() <= 1.0e-6,
            "sigmoid omega asymptote expected 0.90, got {w_sig_om:.16e}"
        );
    }

    #[test]
    fn test_eval_i_rot_closed_forms_and_paxton_2019_polynomial() {
        assert_eq!(eval_i_rot(R_SUN, 0.6, true), TWO_THIRDS * R_SUN * R_SUN);
        assert_eq!(eval_i_rot(R_SUN, 0.0, false), TWO_THIRDS * (R_SUN * R_SUN));

        let expected_06 = 3.921_503_060_173_274e21;
        let got_06 = eval_i_rot(R_SUN, 0.6, false);
        let rel = (got_06 - expected_06).abs() / expected_06;
        assert!(
            rel < 1.0e-14,
            "eval_i_rot(R_SUN, 0.6) mismatch: got {got_06:.16e}, expected {expected_06:.16e}"
        );
    }
}
