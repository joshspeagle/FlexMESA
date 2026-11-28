// SPDX-License-Identifier: GPL-3.0-or-later
//! Spherical face mechanical work (`eval_dwork`, `eval1_work`,
//! `eval_simple_PdV_work`) and cell-specific kinetic/gravitational-potential
//! energy rates (`cell_specific_KE_qp`, `cell_specific_PE_qp`) for `v_flag` and
//! `u_flag` hydrodynamics (`star/private/hydro_energy.f90:582-879` and
//! `star/private/star_utils.f90:2197-2338`).

use flexmesa_const::{FOUR_PI, LN10};

use crate::hydro_energy::{
    eval_face_area_time_centered_ad, eval_pvsc_cell_ad, pvsc_start_after_get,
};
use crate::model::StarModel;
use crate::star_ad::{
    StarAdExt, StarAdOrder1, I_LND_00, I_LND_M1, I_LNR_00, I_LNR_P1, I_V_00,
    I_V_P1,
};

/// Evaluates the cell-specific gravitational potential energy rate
/// $de_{\mathrm{pot}}/dt$ in $\mathrm{erg\,g^{-1}\,s^{-1}}$ for cell `k` as
/// `dpe_dt_ad` (`setup_de_dt_and_friends`, `star/private/hydro_energy.f90:455-458`),
/// with the value and partials from [`crate::cell_energy::get_dke_dt_dpe_dt`].
#[must_use]
pub fn eval_cell_dpe_dt_ad(
    model: &StarModel,
    k: usize,
    dt: f64,
) -> StarAdOrder1 {
    let rates = crate::cell_energy::get_dke_dt_dpe_dt(model, k, dt);
    let mut dpe_dt_ad = StarAdOrder1::constant(rates.dpe_dt);
    dpe_dt_ad.d1[I_LNR_00] = rates.d_dpedt_dlnr00;
    dpe_dt_ad.d1[I_LNR_P1] = rates.d_dpedt_dlnrp1;
    dpe_dt_ad
}

/// Evaluates the kinetic energy rate $de_K/dt$ in
/// $\mathrm{erg\,g^{-1}\,s^{-1}}$ for cell `k` as `dke_dt_ad`
/// (`setup_de_dt_and_friends`, `star/private/hydro_energy.f90:450-453`), for
/// either `v_flag` or `u_flag`, with the value and partials from
/// [`crate::cell_energy::get_dke_dt_dpe_dt`].
#[must_use]
pub fn eval_dke_dt_ad(model: &StarModel, k: usize, dt: f64) -> StarAdOrder1 {
    let rates = crate::cell_energy::get_dke_dt_dpe_dt(model, k, dt);
    let mut dke_dt_ad = StarAdOrder1::constant(rates.dke_dt);
    dke_dt_ad.d1[I_V_00] = rates.d_dkedt_dv00;
    dke_dt_ad.d1[I_V_P1] = rates.d_dkedt_dvp1;
    dke_dt_ad
}

/// `v_flag` name for [`eval_dke_dt_ad`], kept for existing callers.
#[must_use]
pub fn eval_v_flag_dke_dt_ad(
    model: &StarModel,
    k: usize,
    dt: f64,
) -> StarAdOrder1 {
    eval_dke_dt_ad(model, k, dt)
}

/// Evaluates spherical volume-flux rate $A_k v_{c,k}$ in $\mathrm{cm^3\,s^{-1}}$
/// at face `f_idx` in face `f_idx`'s local stencil (`eval1_A_times_v_face_ad`
/// in `star/private/hydro_energy.f90:816-847`).
#[must_use]
pub fn eval1_a_times_v_face_v_flag_ad(
    model: &StarModel,
    f_idx: usize,
    vel_tc: bool,
) -> StarAdOrder1 {
    if f_idx >= model.nz {
        // `star/private/hydro_energy.f90:856`:
        // `4*pi*pow2(s% r_center)*s% v_center`.
        return StarAdOrder1::constant(
            FOUR_PI * flexmesa_math::pow2(model.r_center) * model.v_center,
        );
    }
    let lnr_f = StarAdOrder1::variable(model.ln_r[f_idx], I_LNR_00);
    let lnr_s_f = model.ln_r_start[f_idx];
    let area_f = eval_face_area_time_centered_ad(lnr_f, lnr_s_f, vel_tc);
    let vc_f = if model.controls.v_flag {
        let v_f = StarAdOrder1::variable(model.v[f_idx], I_V_00);
        if vel_tc {
            let vs_f =
                model.v_start.get(f_idx).copied().unwrap_or(model.v[f_idx]);
            (v_f + vs_f) * 0.5
        } else {
            v_f
        }
    } else {
        // `u_face_ad%val = (s% r(k) - s% r_start(k))/s% dt` with
        // `d1Array(i_lnR_00) = s% r(k)/s% dt`, both halved first when time
        // centering (`star/private/hydro_energy.f90:810-816`).
        let dt = model.dt;
        let r_f = model.r[f_idx];
        let rs_f = flexmesa_math::exp(lnr_s_f);
        let mut u_face = StarAdOrder1::ZERO;
        if vel_tc {
            u_face.val = 0.5 * (r_f - rs_f) / dt;
            u_face.d1[I_LNR_00] = 0.5 * r_f / dt;
        } else {
            u_face.val = (r_f - rs_f) / dt;
            u_face.d1[I_LNR_00] = r_f / dt;
        }
        u_face
    };
    area_f * vc_f
}

/// `s% dlnPeos_dxa_for_partials(j, cell)` (0 when the partials are not set).
fn dlnp_dxa(model: &StarModel, cell: usize, j: usize) -> f64 {
    model
        .aux
        .dln_peos_dxa_for_partials
        .get(cell * model.species + j)
        .copied()
        .unwrap_or(0.0)
}

/// Evaluates face mechanical power $W_f = A_f v_{c,f} P_{\mathrm{face},f}$ in
/// $\mathrm{erg\,s^{-1}}$ and its composition partials
/// `(d_work_dxa00, d_work_dxam1)`, one entry per species (`eval1_work` in
/// `star/private/hydro_energy.f90:634-813`).
#[must_use]
pub fn eval1_work_v_flag_ad(
    model: &StarModel,
    f_idx: usize,
    skip_peos: bool,
    vel_tc: bool,
) -> (StarAdOrder1, Vec<f64>, Vec<f64>) {
    let sp = model.species;
    if f_idx >= model.nz {
        let p_in = model.peos_start.last().copied().unwrap_or(0.0);
        // `star/private/hydro_energy.f90:643`:
        // `pi4*pow2(s% r_center)*s% Peos_start(nz)*s% v_center`.
        let work_center = FOUR_PI
            * flexmesa_math::pow2(model.r_center)
            * p_in
            * model.v_center;
        return (
            StarAdOrder1::constant(work_center),
            vec![0.0; sp],
            vec![0.0; sp],
        );
    }
    let av_ad = eval1_a_times_v_face_v_flag_ad(model, f_idx, vel_tc);
    let (alfa, beta) = if f_idx > 0 {
        model.get_face_weights(f_idx)
    } else {
        (1.0, 0.0)
    };
    let p_theta = if vel_tc {
        crate::hydro_momentum::eval_p_theta(model, f_idx)
    } else {
        1.0
    };

    // `d_Pface_dxa00` and `d_Pface_dxam1` (`hydro_energy.f90:674-700`).
    let mut d_pface_dxa00 = vec![0.0; sp];
    let mut d_pface_dxam1 = vec![0.0; sp];
    let peos_ad = if skip_peos {
        StarAdOrder1::ZERO
    } else {
        let p_s_l = model
            .peos_start
            .get(f_idx)
            .copied()
            .unwrap_or(model.peos[f_idx]);
        // `PL_ad = P_theta*wrap_Peos_00(s,k) + (1d0-P_theta)*s% Peos_start(k)`
        // and `PR_ad` with `wrap_Peos_m1` (`star/private/hydro_energy.f90:685-691`):
        // value `Peos`, partials `Peos*chi*_for_partials`.
        let pl_ad = crate::hydro_momentum::wrap_peos_00(model, f_idx) * p_theta
            + p_s_l * (1.0 - p_theta);
        let pr_ad = if f_idx > 0 {
            let p_s_r = model
                .peos_start
                .get(f_idx - 1)
                .copied()
                .unwrap_or(model.peos[f_idx - 1]);
            crate::hydro_momentum::wrap_peos_m1(model, f_idx) * p_theta
                + p_s_r * (1.0 - p_theta)
        } else {
            StarAdOrder1::ZERO
        };
        if f_idx > 0 {
            for j in 0..sp {
                d_pface_dxa00[j] = alfa
                    * dlnp_dxa(model, f_idx, j)
                    * p_theta
                    * model.peos[f_idx];
            }
            for j in 0..sp {
                d_pface_dxam1[j] = beta
                    * dlnp_dxa(model, f_idx - 1, j)
                    * p_theta
                    * model.peos[f_idx - 1];
            }
        } else {
            for j in 0..sp {
                d_pface_dxa00[j] =
                    dlnp_dxa(model, f_idx, j) * p_theta * model.peos[f_idx];
            }
        }
        pl_ad * alfa + pr_ad * beta
    };

    // `set Pvsc_ad` (`star/private/hydro_energy.f90:709-728`): both sides are
    // averaged with `Pvsc_start` when `include_P_in_velocity_time_centering`
    // holds and the face cell's `lnT(k)/ln10` is at most
    // `max_logT_for_include_P_and_L_in_velocity_time_centering`; upstream
    // does not test `using_velocity_time_centering` here.
    let pvsc_ad = if model.controls.use_pvsc_art_visc {
        let pvsc_tc = model.controls.include_p_in_velocity_time_centering
            && model.ln_t[f_idx] / LN10
                <= model
                    .controls
                    .max_logt_for_include_p_and_l_in_velocity_time_centering;
        let pvsc_r = if f_idx > 0 {
            let mut pvsc_r = eval_pvsc_cell_ad(model, f_idx - 1).shift_m1();
            if pvsc_tc {
                let start = pvsc_start_after_get(model, f_idx - 1, pvsc_r.val);
                pvsc_r = 0.5 * (pvsc_r + start);
            }
            pvsc_r
        } else {
            StarAdOrder1::ZERO
        };
        let mut pvsc_l = eval_pvsc_cell_ad(model, f_idx);
        if pvsc_tc {
            let start = pvsc_start_after_get(model, f_idx, pvsc_l.val);
            pvsc_l = 0.5 * (pvsc_l + start);
        }
        pvsc_l * alfa + pvsc_r * beta
    } else {
        StarAdOrder1::ZERO
    };

    let ptrb_ad =
        if model.controls.rsp2_flag && model.controls.rsp2.alfap != 0.0 {
            let ptrb_l =
                crate::rsp2_ad::eval_rsp2_ptrb_cell_ad(model, f_idx, false);
            let ptrb_r = if f_idx > 0 {
                crate::rsp2_ad::eval_rsp2_ptrb_cell_ad(model, f_idx - 1, true)
            } else {
                StarAdOrder1::ZERO
            };
            ptrb_l * alfa + ptrb_r * beta
        } else {
            StarAdOrder1::ZERO
        };

    let mlt_pturb_ad = if model.controls.mlt_pturb_factor > 0.0 {
        let vc_old = model.aux.mlt_vc_old[f_idx];
        if vc_old > 0.0 {
            let rho_00 =
                StarAdOrder1::variable(model.lnd[f_idx], I_LND_00).exp();
            let rho_face = if f_idx == 0 {
                rho_00
            } else {
                let rho_m1 =
                    StarAdOrder1::variable(model.lnd[f_idx - 1], I_LND_M1)
                        .exp();
                rho_00 * alfa + rho_m1 * beta
            };
            // `mlt_Pturb_factor*pow2(mlt_vc_old(k))*get_rho_face(s,k)/3d0`
            // (`star/private/hydro_energy.f90:760`).
            rho_face
                * (model.controls.mlt_pturb_factor
                    * flexmesa_math::pow2(vc_old))
                / 3.0
        } else {
            StarAdOrder1::ZERO
        }
    } else {
        StarAdOrder1::ZERO
    };

    let p_face_ad = peos_ad + pvsc_ad + ptrb_ad + mlt_pturb_ad;
    // `d_work_dxa00(j) = Av_face*d_Pface_dxa00(j)` (`hydro_energy.f90:771-775`).
    let av_face = av_ad.val;
    let d_work_dxa00 = d_pface_dxa00.iter().map(|d| av_face * d).collect();
    let d_work_dxam1 = d_pface_dxam1.iter().map(|d| av_face * d).collect();
    (av_ad * p_face_ad, d_work_dxa00, d_work_dxam1)
}

/// Composition partials of the work difference across cell `k`
/// (`d_dwork_dxam1`, `d_dwork_dxa00`, `d_dwork_dxap1` of `eval_dwork`), one
/// entry per species, not yet divided by `dm`.
pub struct DworkDxa {
    /// `d_dwork_dxam1`.
    pub dxam1: Vec<f64>,
    /// `d_dwork_dxa00`.
    pub dxa00: Vec<f64>,
    /// `d_dwork_dxap1`.
    pub dxap1: Vec<f64>,
}

/// Evaluates spherical mechanical work divergence
/// $dW/dm = (W_k - W_{k+1}) / \Delta m_k$ (`eval_dwork` and
/// `setup_dwork_dm` in `star/private/hydro_energy.f90:163-190, 561-610`) and
/// the composition partials of $W_k - W_{k+1}$.
#[must_use]
pub fn eval_v_flag_dwork_dm_ad(
    model: &StarModel,
    k: usize,
    skip_peos: bool,
    vel_tc: bool,
) -> (StarAdOrder1, DworkDxa) {
    let dm_k = model.dm[k];
    let (work_00, d_work_00_dxa00, d_work_00_dxam1) =
        eval1_work_v_flag_ad(model, k, skip_peos, vel_tc);
    let (work_p1_raw, d_work_p1_dxap1, d_work_p1_dxa00) =
        eval1_work_v_flag_ad(model, k + 1, skip_peos, vel_tc);
    let work_p1 = if k + 1 < model.nz {
        work_p1_raw.shift_p1()
    } else {
        work_p1_raw
    };
    // `hydro_energy.f90:588-592`.
    let dxa00 = d_work_00_dxa00
        .iter()
        .zip(&d_work_p1_dxa00)
        .map(|(a, b)| a - b)
        .collect();
    let dxap1 = d_work_p1_dxap1.iter().map(|d| -d).collect();
    (
        (work_00 - work_p1) / dm_k,
        DworkDxa {
            dxam1: d_work_00_dxam1,
            dxa00,
            dxap1,
        },
    )
}

/// Evaluates time-centered total pressure `(P_tot, P_theta)` for `simple_pdv`
/// work (`calc_Ptot_ad_tw` / `eval_simple_PdV_work`,
/// `star/private/hydro_energy.f90:862-866`).
#[must_use]
pub(crate) fn eval_simple_pdv_p_tot_ad(
    model: &StarModel,
    k: usize,
    skip_peos: bool,
) -> (StarAdOrder1, f64) {
    let (mut p_tot_ad, p_theta) = if skip_peos {
        (StarAdOrder1::ZERO, 0.0)
    } else {
        crate::hydro_momentum::eval_cell_peos_time_centered_ad(model, k, false)
    };
    if model.controls.use_pvsc_art_visc {
        p_tot_ad = p_tot_ad + eval_pvsc_cell_ad(model, k);
    }
    if model.controls.rsp2_flag && model.controls.rsp2.alfap != 0.0 {
        p_tot_ad =
            p_tot_ad + crate::rsp2_ad::eval_rsp2_ptrb_cell_ad(model, k, false);
    }
    if k > 0 && model.controls.mlt_pturb_factor > 0.0 {
        p_tot_ad = p_tot_ad + eval_simple_pdv_mlt_pturb_ad(model, k);
    }
    (p_tot_ad, p_theta)
}

/// `mlt_Pturb_ad` of `calc_Ptot_ad_tw` (`star/private/star_utils.f90:3424-3432`)
/// at cell `k > 0`: `mlt_Pturb_factor*pow2(mlt_vc_old(k))*get_rho_face(s,k)/3`,
/// with the face density `alfa*rho(k) + beta*rho(k-1)`; when time centering
/// applies (`:3392-3394`) it is blended with the start-of-step value
/// `mlt_Pturb_factor*pow2(mlt_vc_old(k))*(rho_start(k-1) + rho_start(k))/6`
/// using `alfa = P_theta_for_velocity_time_centering`.
fn eval_simple_pdv_mlt_pturb_ad(model: &StarModel, k: usize) -> StarAdOrder1 {
    let vc_old = model.aux.mlt_vc_old[k];
    if vc_old <= 0.0 {
        return StarAdOrder1::ZERO;
    }
    let (alfa_f, beta_f) = model.get_face_weights(k);
    let rho_00 = StarAdOrder1::variable(model.lnd[k], I_LND_00).exp();
    let rho_m1 = StarAdOrder1::variable(model.lnd[k - 1], I_LND_M1).exp();
    let rho_face = rho_00 * alfa_f + rho_m1 * beta_f;
    let coef = model.controls.mlt_pturb_factor * flexmesa_math::pow2(vc_old);
    let mlt_pturb = (rho_face * coef) / 3.0;
    let alfa = crate::hydro_momentum::eval_p_theta(model, k);
    let time_center = model.controls.using_velocity_time_centering
        && model.controls.include_p_in_velocity_time_centering
        && model.ln_t[k] / flexmesa_const::LN10
            <= model
                .controls
                .max_logt_for_include_p_and_l_in_velocity_time_centering;
    if !time_center {
        return mlt_pturb;
    }
    let rho_start_m1 = flexmesa_math::exp(model.lnd_start[k - 1]);
    let rho_start_00 = flexmesa_math::exp(model.lnd_start[k]);
    let mlt_pturb_start = coef * (rho_start_m1 + rho_start_00) / 6.0;
    mlt_pturb * alfa + mlt_pturb_start * (1.0 - alfa)
}

/// Evaluates `eval_simple_PdV_work` divided by $\Delta m_k$, `(P_tot * dV)/dm_k`
/// with `dV = A_k v_{c,k} - A_{k+1} v_{c,k+1}` (`setup_dwork_dm` and
/// `eval_simple_PdV_work`, `star/private/hydro_energy.f90:163-190, 824-877`),
/// and `d_dwork_dxa00`, one entry per species, not yet divided by `dm`.
#[must_use]
pub fn eval_v_flag_simple_pdv_dm_ad(
    model: &StarModel,
    k: usize,
    skip_peos: bool,
    vel_tc: bool,
) -> (StarAdOrder1, Vec<f64>) {
    let dm_k = model.dm[k];
    let av_00 = eval1_a_times_v_face_v_flag_ad(model, k, vel_tc);
    let av_p1 = if k + 1 < model.nz {
        eval1_a_times_v_face_v_flag_ad(model, k + 1, vel_tc).shift_p1()
    } else {
        eval1_a_times_v_face_v_flag_ad(model, model.nz, vel_tc)
    };
    let dv = av_00 - av_p1;

    let (p_tot_ad, p_theta) = eval_simple_pdv_p_tot_ad(model, k, skip_peos);
    // `d_Ptot_dxa(j) = alfa*(Peos(k)*dlnPeos_dxa_for_partials(j,k))`
    // (`star/private/star_utils.f90:3403-3406`) and
    // `d_dwork_dxa00(j) = d_Ptot_dxa(j)*(Av_face00 - Av_facep1)`
    // (`hydro_energy.f90:867-869`).
    let dav = av_00.val - av_p1.val;
    let d_dwork_dxa00 = (0..model.species)
        .map(|j| {
            if skip_peos {
                0.0
            } else {
                p_theta * (model.peos[k] * dlnp_dxa(model, k, j)) * dav
            }
        })
        .collect();
    ((p_tot_ad * dv) / dm_k, d_dwork_dxa00)
}
