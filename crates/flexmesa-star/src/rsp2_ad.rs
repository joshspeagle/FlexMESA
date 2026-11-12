// SPDX-License-Identifier: LGPL-3.0-or-later
//! 24-slot 3-cell stencil [`StarAdOrder1`] automatic differentiation for
//! coupled `RSP2` hydrodynamic variables and Henyey equations (`set_RSP2_vars`,
//! `do1_rsp2_L_eqn`, `do1_rsp2_Hp_eqn`, and `do1_turbulent_energy_eqn` in
//! `star/private/hydro_rsp2.f90:67-1172`).

use flexmesa_const::{BOLTZ_SIGMA, CLIGHT, CRAD, LN10, PI, PI4, SQRT_2_DIV_3};
use flexmesa_eos::{I_CHI_RHO, I_CHI_T, I_CP, I_GRAD_AD, I_LN_E};

use crate::hydro_eqns::HenyeySystem;
use crate::model::StarModel;
use crate::rsp2::{
    rsp2_flux_limiter, X_ALFAC, X_ALFAP, X_ALFAS, X_CEDE, X_GAMMAR,
};
use crate::rsp2_eqns::{is_nonturbulent_cell, rsp2_face_weights};
use crate::star_ad::{
    StarAdExt, StarAdOrder1, I_HP_00, I_HP_P1, I_LND_00, I_LND_M1, I_LNR_00,
    I_LNR_P1, I_LNT_00, I_LNT_M1, I_L_00, I_V_00, I_V_P1, I_W_00, I_W_M1,
};

#[inline]
fn start_or(slice: &[f64], k: usize, fallback: f64) -> f64 {
    slice.get(k).copied().unwrap_or(fallback)
}

#[inline]
fn start_exp(slice: &[f64], k: usize, fallback_exp: f64) -> f64 {
    slice.get(k).copied().map_or(fallback_exp, f64::exp)
}

#[inline]
fn wrap_w(model: &StarModel, k: usize, slot: usize) -> StarAdOrder1 {
    let w = model.aux.rsp.et_rsp[k].max(0.0).sqrt();
    if model.layout.i_w.is_some() {
        StarAdOrder1::variable(w, slot)
    } else {
        StarAdOrder1::constant(w)
    }
}

#[inline]
fn wrap_hp(model: &StarModel, k: usize, slot: usize) -> StarAdOrder1 {
    let hp = model.aux.rsp.hp_face[k].max(1.0);
    if model.layout.i_hp.is_some() {
        StarAdOrder1::variable(hp, slot)
    } else {
        StarAdOrder1::constant(hp)
    }
}

#[inline]
fn wrap_hp_cell(model: &StarModel, k: usize) -> StarAdOrder1 {
    let hp_00 = wrap_hp(model, k, I_HP_00);
    if k + 1 < model.nz {
        (hp_00 + wrap_hp(model, k + 1, I_HP_P1)) * 0.5
    } else {
        hp_00
    }
}

#[inline]
fn wrap_eos_00(
    model: &StarModel,
    k: usize,
    val: f64,
    slot: usize,
) -> StarAdOrder1 {
    StarAdOrder1::from_dlnd_dlnt_00(
        val,
        model.eos_d_dlnd[k][slot],
        model.eos_d_dlnt[k][slot],
    )
}

#[inline]
fn wrap_eos_m1(
    model: &StarModel,
    k: usize,
    val: f64,
    slot: usize,
) -> StarAdOrder1 {
    StarAdOrder1::from_dlnd_dlnt_m1(
        val,
        model.eos_d_dlnd[k][slot],
        model.eos_d_dlnt[k][slot],
    )
}

#[inline]
fn wrap_peos_cell(model: &StarModel, k: usize, shift_m1: bool) -> StarAdOrder1 {
    if shift_m1 {
        model.ln_peos_ad[k].shift_m1().exp()
    } else {
        model.ln_peos_ad[k].exp()
    }
}

/// Evaluates `RSP2` cell turbulent pressure $P_{\mathrm{trb},k} = \frac{2}{3}\alpha_p \rho_k w_k^2$
/// with optional $\theta_P$ velocity time-centering (`calc_Ptrb_ad_tw` in
/// `star/private/star_utils.f90`).
#[must_use]
pub fn eval_rsp2_ptrb_cell_ad(
    model: &StarModel,
    k: usize,
    shift_m1: bool,
) -> StarAdOrder1 {
    let coeff = X_ALFAP * model.controls.rsp2.alfap;
    if coeff == 0.0
        || !model.controls.rsp2_flag
        || is_nonturbulent_cell(model, k)
    {
        return StarAdOrder1::ZERO;
    }
    let (lnd_slot, w_slot) = if shift_m1 {
        (I_LND_M1, I_W_M1)
    } else {
        (I_LND_00, I_W_00)
    };
    let rho_ad = StarAdOrder1::variable(model.lnd[k], lnd_slot).exp();
    let w_ad = wrap_w(model, k, w_slot);
    let ptrb_now = rho_ad * w_ad * w_ad * coeff;

    let logt = model.ln_t[k] / LN10;
    let p_theta = if model.controls.using_velocity_time_centering
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
    if p_theta < 1.0 {
        let pt_s = start_or(&model.aux.rsp.ptrb_start, k, ptrb_now.val);
        ptrb_now * p_theta + (1.0 - p_theta) * pt_s
    } else {
        ptrb_now
    }
}

/// Evaluates `Hp_face_for_rsp2_eqn` (`hydro_rsp2.f90:230-285`) with 24-slot partials.
#[must_use]
pub fn eval_rsp2_hp_face_ad(model: &StarModel, k: usize) -> StarAdOrder1 {
    let r_00 = StarAdOrder1::variable(model.ln_r[k], I_LNR_00).exp();
    let d_00 = StarAdOrder1::variable(model.lnd[k], I_LND_00).exp();
    let peos_00 = wrap_peos_cell(model, k, false);

    if k > 0 && !model.controls.rsp2.assume_hse {
        let (alfa, beta) = rsp2_face_weights(model, k);
        let d_m1 = StarAdOrder1::variable(model.lnd[k - 1], I_LND_M1).exp();
        let rho_face = d_00 * alfa + d_m1 * beta;
        let area = r_00 * r_00 * PI4;
        let dlnp = model.ln_peos_ad[k - 1].shift_m1() - model.ln_peos_ad[k];
        if dlnp.val.abs() > 1.0e-30 {
            let hp = StarAdOrder1::constant(-model.dm_bar[k])
                / (area * rho_face * dlnp);
            if hp.val > 1.0 {
                return hp;
            }
        }
    }

    let peos_div_rho = if k == 0 {
        peos_00 / d_00
    } else {
        let (alfa, beta) = rsp2_face_weights(model, k);
        let d_m1 = StarAdOrder1::variable(model.lnd[k - 1], I_LND_M1).exp();
        let peos_m1 = wrap_peos_cell(model, k - 1, true);
        (peos_00 / d_00) * alfa + (peos_m1 / d_m1) * beta
    };
    let gm = (model.cgrav[k] * model.m[k]).max(1.0e-30);
    let hp = r_00 * r_00 * peos_div_rho * (1.0 / gm);
    if hp.val >= 1.0 {
        hp
    } else {
        StarAdOrder1::ONE
    }
}

pub(crate) struct Rsp2AdWork {
    pub(crate) coupl: Vec<StarAdOrder1>,
    pub(crate) lr: Vec<StarAdOrder1>,
    pub(crate) lc: Vec<StarAdOrder1>,
    pub(crate) lt: Vec<StarAdOrder1>,
}

pub(crate) fn build_rsp2_ad_work(model: &mut StarModel) -> Rsp2AdWork {
    let nz = model.nz;
    model.aux.rsp.resize(nz);
    let alpha = model.controls.mixing_length_alpha;
    let rsp2 = model.controls.rsp2.clone();

    for k in 0..nz {
        if model.aux.rsp.hp_face[k] <= 0.0 {
            model.aux.rsp.hp_face[k] =
                eval_rsp2_hp_face_ad(model, k).val.max(1.0);
        }
    }

    let mut pii_ad = vec![StarAdOrder1::ZERO; nz];
    for (k, pii_slot) in pii_ad.iter_mut().enumerate() {
        if k == 0 || alpha == 0.0 {
            model.aux.rsp.y_face_rsp[k] = 0.0;
            model.aux.rsp.pii_rsp[k] = 0.0;
            continue;
        }
        let (alfa, beta) = rsp2_face_weights(model, k);
        let d_00 = StarAdOrder1::variable(model.lnd[k], I_LND_00).exp();
        let d_m1 = StarAdOrder1::variable(model.lnd[k - 1], I_LND_M1).exp();
        let lnt_00 = StarAdOrder1::variable(model.ln_t[k], I_LNT_00);
        let lnt_m1 = StarAdOrder1::variable(model.ln_t[k - 1], I_LNT_M1);
        let t_00 = lnt_00.exp();
        let t_m1 = lnt_m1.exp();
        let peos_00 = wrap_peos_cell(model, k, false);
        let peos_m1 = wrap_peos_cell(model, k - 1, true);
        let cp_00 = wrap_eos_00(model, k, model.cp[k], I_CP);
        let cp_m1 = wrap_eos_m1(model, k - 1, model.cp[k - 1], I_CP);

        let y_face = if rsp2.use_rsp_eqn_for_y_face {
            let ct_00 = wrap_eos_00(model, k, model.chi_t[k], I_CHI_T);
            let cr_00 = wrap_eos_00(model, k, model.chi_rho[k], I_CHI_RHO);
            let ct_m1 = wrap_eos_m1(model, k - 1, model.chi_t[k - 1], I_CHI_T);
            let cr_m1 =
                wrap_eos_m1(model, k - 1, model.chi_rho[k - 1], I_CHI_RHO);
            let qq_00 = ct_00 / (d_00 * t_00 * cr_00);
            let qq_m1 = ct_m1 / (d_m1 * t_m1 * cr_m1);
            let qq_div_cp = (qq_00 / cp_00) * alfa + (qq_m1 / cp_m1) * beta;
            let y1 = qq_div_cp * (peos_m1 - peos_00) - (lnt_m1 - lnt_00);
            let r_00 = StarAdOrder1::variable(model.ln_r[k], I_LNR_00).exp();
            let hp_00 = wrap_hp(model, k, I_HP_00);
            let inv_rho_sum =
                StarAdOrder1::ONE / d_00 + StarAdOrder1::ONE / d_m1;
            let y2 = r_00
                * r_00
                * hp_00
                * (2.0 * PI4 / model.dm_bar[k].max(1.0e-99))
                / inv_rho_sum;
            y1 * y2
        } else {
            let ga_00 = wrap_eos_00(model, k, model.grada[k], I_GRAD_AD);
            let ga_m1 =
                wrap_eos_m1(model, k - 1, model.grada[k - 1], I_GRAD_AD);
            let grad_ad_f = ga_00 * alfa + ga_m1 * beta;
            let dlnt = lnt_m1 - lnt_00;
            let dlnp = model.ln_peos_ad[k - 1].shift_m1() - model.ln_peos_ad[k];
            if dlnp.val.abs() > 1.0e-30 {
                dlnt / dlnp - grad_ad_f
            } else {
                StarAdOrder1::ZERO
            }
        };
        model.aux.rsp.y_face_rsp[k] = y_face.val;
        if k + 1 < nz {
            let cp_face = cp_00 * alfa + cp_m1 * beta;
            let alfas_alfa = X_ALFAS * alpha;
            let pii_raw = cp_face * y_face * alfas_alfa;
            let pii = if y_face.val > 0.0 && rsp2.use_tdc_enthalpy_flux_limiter
            {
                let t_face = t_00 * alfa + t_m1 * beta;
                let rho_face = d_00 * alfa + d_m1 * beta;
                let peos_face = peos_00 * alfa + peos_m1 * beta;
                let e_00 = wrap_eos_00(model, k, model.ln_e[k], I_LN_E).exp();
                let e_m1 =
                    wrap_eos_m1(model, k - 1, model.ln_e[k - 1], I_LN_E).exp();
                let e_face = e_00 * alfa + e_m1 * beta;
                let h_face = e_face + peos_face / rho_face;
                let x_ad = (cp_face * t_face / h_face)
                    * y_face
                    * (alfas_alfa / SQRT_2_DIV_3);
                if x_ad.val.abs() >= 0.95 {
                    let (fl_v, dfl) = rsp2_flux_limiter(x_ad.val);
                    let fl_ad = x_ad.make_unary_operator(fl_v, dfl);
                    pii_raw * (fl_ad / x_ad)
                } else {
                    pii_raw
                }
            } else {
                pii_raw
            };
            model.aux.rsp.pii_rsp[k] = pii.val;
            *pii_slot = pii;
        } else {
            model.aux.rsp.pii_rsp[k] = 0.0;
        }
    }

    let mut chi_ad = vec![StarAdOrder1::ZERO; nz];
    let mut coupl_ad = vec![StarAdOrder1::ZERO; nz];
    for k in 0..nz {
        let w_00 = wrap_w(model, k, I_W_00);
        let d_00 = StarAdOrder1::variable(model.lnd[k], I_LND_00).exp();
        model.aux.rsp.ptrb_rsp[k] =
            X_ALFAP * rsp2.alfap * d_00.val * w_00.val.powi(2);
        if is_nonturbulent_cell(model, k) {
            model.aux.rsp.chi_rsp[k] = 0.0;
            model.aux.rsp.eq_rsp[k] = 0.0;
            model.aux.rsp.source_rsp[k] = 0.0;
            model.aux.rsp.damp_rsp[k] = 0.0;
            model.aux.rsp.dampr_rsp[k] = 0.0;
            model.aux.rsp.coupl_rsp[k] = 0.0;
            model.aux.eq_ad[k] = StarAdOrder1::ZERO;
            continue;
        }
        let r_00 = StarAdOrder1::variable(model.ln_r[k], I_LNR_00).exp();
        let r_p1 = if k + 1 < nz {
            StarAdOrder1::variable(model.ln_r[k + 1], I_LNR_P1).exp()
        } else {
            StarAdOrder1::constant(model.r_center)
        };
        let v_00 = if model.controls.v_flag {
            StarAdOrder1::variable(model.v[k], I_V_00)
        } else {
            StarAdOrder1::ZERO
        };
        let v_p1 = if !model.controls.v_flag {
            StarAdOrder1::ZERO
        } else if k + 1 < nz {
            StarAdOrder1::variable(model.v[k + 1], I_V_P1)
        } else {
            StarAdOrder1::constant(model.v_center)
        };
        let dv_div_r = v_00 / r_00
            - if r_p1.val > 0.0 {
                v_p1 / r_p1
            } else {
                StarAdOrder1::ZERO
            };
        let dv_div_r_tc = if model.controls.using_velocity_time_centering
            && model.controls.v_flag
            && !model.v_start.is_empty()
            && !model.ln_r_start.is_empty()
        {
            let v_tc_00 = (v_00 + start_or(&model.v_start, k, v_00.val)) * 0.5;
            let r_tc_00 =
                (r_00 + start_exp(&model.ln_r_start, k, r_00.val)) * 0.5;
            let (v_tc_p1, r_tc_p1) = if k + 1 < nz {
                let v_s1 = start_or(&model.v_start, k + 1, v_p1.val);
                let r_s1 = start_exp(&model.ln_r_start, k + 1, r_p1.val);
                ((v_p1 + v_s1) * 0.5, (r_p1 + r_s1) * 0.5)
            } else {
                (
                    StarAdOrder1::constant(model.v_center),
                    StarAdOrder1::constant(model.r_center),
                )
            };
            let in_term = if r_tc_p1.val > 0.0 {
                v_tc_p1 / r_tc_p1
            } else {
                StarAdOrder1::ZERO
            };
            v_tc_00 / r_tc_00 - in_term
        } else {
            dv_div_r
        };
        let hp_cell = wrap_hp_cell(model, k);
        let f_chi =
            (16.0 / 3.0) * PI * (rsp2.alfam * alpha) / model.dm[k].max(1.0e-99);
        let r_00_3 = r_00 * r_00 * r_00;
        let r_p1_3 = r_p1 * r_p1 * r_p1;
        let r6 = (r_00_3 * r_00_3 + r_p1_3 * r_p1_3) * 0.5;
        let chi = d_00 * d_00 * r6 * dv_div_r * hp_cell * w_00 * f_chi;
        let eq = chi * dv_div_r_tc * (PI4 / model.dm[k].max(1.0e-99));
        chi_ad[k] = chi;
        model.aux.eq_ad[k] = eq;

        let hp_00 = wrap_hp(model, k, I_HP_00);
        let pii_div_hp = if k + 1 == nz {
            pii_ad[k] / hp_00
        } else {
            let hp_p1 = wrap_hp(model, k + 1, I_HP_P1);
            (pii_ad[k] / hp_00 + pii_ad[k + 1].shift_p1() / hp_p1) * 0.5
        };
        let t_00 = StarAdOrder1::variable(model.ln_t[k], I_LNT_00).exp();
        let peos_00 = wrap_peos_cell(model, k, false);
        let cp_00 = wrap_eos_00(model, k, model.cp[k], I_CP);
        let ct_00 = wrap_eos_00(model, k, model.chi_t[k], I_CHI_T);
        let cr_00 = wrap_eos_00(model, k, model.chi_rho[k], I_CHI_RHO);
        let qq_00 = ct_00 / (d_00 * t_00 * cr_00);
        let src = (w_00 + rsp2.source_seed)
            * pii_div_hp
            * t_00
            * (peos_00 * qq_00 / cp_00);
        let dw3 = w_00 * w_00 * w_00 - rsp2.w_min_for_damping.powi(3);
        let damp = dw3 / hp_cell * (rsp2.alfad * X_CEDE / alpha);
        let dampr = if rsp2.alfar == 0.0 {
            StarAdOrder1::ZERO
        } else {
            let kap_00 = StarAdOrder1::from_dlnd_dlnt_00(
                model.opacity[k].max(1.0e-30),
                model.d_opacity_dlnd[k],
                model.d_opacity_dlnt[k],
            );
            let pom =
                4.0 * BOLTZ_SIGMA * (rsp2.alfar * X_GAMMAR / alpha).powi(2);
            let pom2 = (t_00 * t_00 * t_00) / (d_00 * d_00 * cp_00 * kap_00);
            (w_00 * w_00 * pom) * pom2 / (hp_cell * hp_cell)
        };
        let coupl = src - damp - dampr;
        coupl_ad[k] = coupl;
        model.aux.rsp.chi_rsp[k] = chi.val;
        model.aux.rsp.eq_rsp[k] = eq.val;
        model.aux.rsp.source_rsp[k] = src.val;
        model.aux.rsp.damp_rsp[k] = damp.val;
        model.aux.rsp.dampr_rsp[k] = dampr.val;
        model.aux.rsp.coupl_rsp[k] = coupl.val;
    }

    let mut lr_ad = vec![StarAdOrder1::ZERO; nz];
    let mut lc_ad = vec![StarAdOrder1::ZERO; nz];
    let mut lt_ad = vec![StarAdOrder1::ZERO; nz];
    for k in 0..nz {
        let r_00 = StarAdOrder1::variable(model.ln_r[k], I_LNR_00).exp();
        let area = r_00 * r_00 * PI4;
        let t_00 = StarAdOrder1::variable(model.ln_t[k], I_LNT_00).exp();
        let t4_00 = t_00.pow4();
        lr_ad[k] = if k == 0 {
            area * t4_00 * (rsp2.lsurf_factor * CLIGHT * CRAD)
        } else {
            let t_m1 =
                StarAdOrder1::variable(model.ln_t[k - 1], I_LNT_M1).exp();
            let t4_m1 = t_m1.pow4();
            let a_dq =
                model.dq[k - 1] / (model.dq[k - 1] + model.dq[k]).max(1.0e-99);
            let kap_00 = StarAdOrder1::from_dlnd_dlnt_00(
                model.opacity[k].max(1.0e-30),
                model.d_opacity_dlnd[k],
                model.d_opacity_dlnt[k],
            );
            let kap_m1 = StarAdOrder1::from_dlnd_dlnt_m1(
                model.opacity[k - 1].max(1.0e-30),
                model.d_opacity_dlnd[k - 1],
                model.d_opacity_dlnt[k - 1],
            );
            let kap_f = kap_00 * a_dq + kap_m1 * (1.0 - a_dq);
            let mut diff = (t4_m1 - t4_00) / kap_f;
            if rsp2.use_stellingwerf_lr {
                let bw = (t4_m1 / t4_00).ln();
                if bw.val.abs() > 1.0e-20 {
                    let bk = (kap_m1 / kap_00).ln();
                    if (1.0 - bk.val / bw.val).abs() > 1.0e-15
                        && (bw.val - bk.val).abs() > 1.0e-15
                    {
                        diff = (t4_m1 / kap_m1 - t4_00 / kap_00)
                            / (StarAdOrder1::ONE - bk / bw);
                    }
                }
            }
            diff * area
                * area
                * (-(CRAD * CLIGHT / 3.0) / model.dm_bar[k].max(1.0e-99))
        };
        model.aux.rsp.fr_rsp[k] = (lr_ad[k] / area).val;

        if k > 0 && !is_nonturbulent_cell(model, k) {
            let (alfa, beta) = rsp2_face_weights(model, k);
            let d_00 = StarAdOrder1::variable(model.lnd[k], I_LND_00).exp();
            let d_m1 = StarAdOrder1::variable(model.lnd[k - 1], I_LND_M1).exp();
            let t_m1 =
                StarAdOrder1::variable(model.ln_t[k - 1], I_LNT_M1).exp();
            let w_00 = wrap_w(model, k, I_W_00);
            let w_m1 = wrap_w(model, k - 1, I_W_M1);
            let t_rho_f = t_00 * d_00 * alfa + t_m1 * d_m1 * beta;
            let w_f = w_00 * alfa + w_m1 * beta;
            lc_ad[k] = area * t_rho_f * pii_ad[k] * w_f * (X_ALFAC / X_ALFAS);
            if alpha * rsp2.alfat != 0.0 {
                let rho2_f = d_00 * d_00 * alfa + d_m1 * d_m1 * beta;
                let hp_00 = wrap_hp(model, k, I_HP_00);
                let detrb = w_m1 * w_m1 - w_00 * w_00;
                let c_lt = -(alpha * rsp2.alfat) / model.dm_bar[k].max(1.0e-99);
                lt_ad[k] =
                    (area * area * rho2_f) * (hp_00 * w_f * detrb) * c_lt;
            }
        }
        model.aux.rsp.lc_rsp[k] = lc_ad[k].val;
        model.aux.rsp.lt_rsp[k] = lt_ad[k].val;

        model.aux.uq_ad[k] = if is_nonturbulent_cell(model, k) {
            StarAdOrder1::ZERO
        } else {
            let chi_m1 = if k > 0 {
                chi_ad[k - 1].shift_m1()
            } else {
                StarAdOrder1::ZERO
            };
            let r_tc = if model.controls.using_velocity_time_centering {
                (r_00 + start_exp(&model.ln_r_start, k, r_00.val)) * 0.5
            } else {
                r_00
            };
            (chi_m1 - chi_ad[k]) / r_tc * (PI4 / model.dm_bar[k].max(1.0e-99))
        };
    }

    Rsp2AdWork {
        coupl: coupl_ad,
        lr: lr_ad,
        lc: lc_ad,
        lt: lt_ad,
    }
}

/// Evaluates all `RSP2` cell and face AD variables (`Eq_ad`, `Uq_ad`, and
/// `model.aux.rsp.*`) during `set_hydro_vars` (`hydro_rsp2.f90:67-111`).
pub fn update_rsp2_vars_ad(model: &mut StarModel) {
    let _ = build_rsp2_ad_work(model);
}

/// Assembles `do1_rsp2_L_eqn`, `do1_rsp2_Hp_eqn`, and `do1_turbulent_energy_eqn`
/// into `sys` across all zones `0..nz` (`hydro_rsp2.f90:114-417`).
pub fn eval_and_save_rsp2_eqns(model: &mut StarModel, sys: &mut HenyeySystem) {
    let work = build_rsp2_ad_work(model);
    let nz = model.nz;
    let l_start_max = model.lum_start.iter().copied().fold(0.0_f64, f64::max);
    let l_scale = if l_start_max > 0.0 {
        l_start_max
    } else {
        model.lum.first().copied().unwrap_or(1.0).abs().max(1.0)
    };
    let inv_l_scale = 1.0 / l_scale;
    let dt = model.dt.max(1.0e-30);
    let k_l_start = usize::from(!model.controls.rsp2.use_l_eqn_at_surface);

    for k in k_l_start..nz {
        let l_exp = work.lr[k] + work.lc[k] + work.lt[k];
        let l_act = StarAdOrder1::variable(model.lum[k], I_L_00);
        let resid_l = (l_exp - l_act) * inv_l_scale;
        sys.save_eqn_residual(&model.layout, k, model.layout.i_equ_l, &resid_l);
    }

    if let Some(i_equ_hp) = model.layout.i_equ_hp {
        for k in 0..nz {
            let hp_exp = eval_rsp2_hp_face_ad(model, k);
            let hp_act = wrap_hp(model, k, I_HP_00);
            let denom = if hp_exp.val >= hp_act.val {
                hp_exp
            } else {
                hp_act
            };
            let resid_hp = (hp_exp - hp_act) / denom;
            sys.save_eqn_residual(&model.layout, k, i_equ_hp, &resid_hp);
        }
    }

    if let Some(i_etrb) = model.layout.i_turbulent_energy {
        for k in 0..nz {
            let w_00 = wrap_w(model, k, I_W_00);
            let resid_w = if is_nonturbulent_cell(model, k) {
                let cs_s = start_or(&model.aux.rsp.csound_start_rsp, k, 0.0);
                let cs =
                    (if cs_s > 0.0 { cs_s } else { model.csound[k] }).max(1.0);
                w_00 * (1.0 / cs)
            } else {
                let d_et = w_00 * w_00 - model.aux.rsp.w_start[k].powi(2);
                let ptrb_ad = eval_rsp2_ptrb_cell_ad(model, k, false);
                let d_00 = StarAdOrder1::variable(model.lnd[k], I_LND_00).exp();
                let rho_s = start_or(&model.lnd_start, k, model.lnd[k])
                    .exp()
                    .max(1.0e-99);
                let ptrb_dv =
                    ptrb_ad * (StarAdOrder1::ONE / d_00 - 1.0 / rho_s);

                let logt = model.ln_t[k] / LN10;
                let l_theta = if model.controls.using_velocity_time_centering
                    && model.controls.include_l_in_velocity_time_centering
                    && logt <= model
                        .controls
                        .max_logt_for_include_p_and_l_in_velocity_time_centering
                {
                    model.controls.l_theta_for_velocity_time_centering
                } else {
                    1.0
                };
                let lt_s0 = start_or(&model.aux.rsp.lt_start, k, 0.0);
                let lt_00 = work.lt[k] * l_theta + (1.0 - l_theta) * lt_s0;
                let lt_p1 = if k + 1 < nz {
                    let lt_s1 = start_or(&model.aux.rsp.lt_start, k + 1, 0.0);
                    work.lt[k + 1].shift_p1() * l_theta
                        + (1.0 - l_theta) * lt_s1
                } else {
                    StarAdOrder1::ZERO
                };
                let dt_dlt_dm =
                    (lt_00 - lt_p1) * (dt / model.dm[k].max(1.0e-99));
                let esum = d_et + ptrb_dv + dt_dlt_dm
                    - work.coupl[k] * dt
                    - model.aux.eq_ad[k] * dt;
                let e_s = start_or(&model.energy_start, k, 0.0);
                let e_start =
                    (if e_s > 0.0 { e_s } else { model.energy[k] }).max(1.0);
                let scal_div_dt = (if k == 0 { 1.0e-6 } else { 1.0 }) / e_start;
                esum * scal_div_dt
            };
            sys.save_eqn_residual(&model.layout, k, i_etrb, &resid_w);
        }
    }
}
