// SPDX-License-Identifier: LGPL-3.0-or-later
//! Array-level Heger, Langer & Woosley (2000) and Spruit (2002) rotational
//! instability diffusivities (`rotation_mix_info.f90:493-1345`) and stiffness-capped
//! substepped angular-momentum transport (`solve_omega_mix.f90`).

use flexmesa_const::{CLIGHT, CRAD, FOUR_PI, L_SUN, M_SUN, MixingType, PI, R_SUN};

use crate::hydro_state::{smooth_triangular_signed, use_smoothed_brunt_b};
use crate::model::StarModel;
use crate::rotation::{
    RI_CRIT_DSI, conserve_shell_j_rot_and_distortion, eval_spruit_tayler_dynamo,
};
use crate::struct_burn_mix::ScalarTridiagFactor;

fn for_each_unstable_block(unstable: &[bool], mut body: impl FnMut(usize, usize)) {
    let nz = unstable.len();
    let mut k = 0_usize;
    while k < nz {
        if !unstable[k] {
            k += 1;
            continue;
        }
        let ktop = k;
        while k + 1 < nz && unstable[k + 1] {
            k += 1;
        }
        if ktop < k {
            body(ktop, k);
        }
        k += 1;
    }
}

fn fill_velocity_scale_heights(
    r: &[f64], dr: &[f64], v: &[f64], ktop: usize, kbot: usize, h_out: &mut [f64],
) {
    h_out[kbot] = v[kbot] * (r[kbot - 1] - r[kbot]).max(1.0)
        / (v[kbot - 1] - v[kbot]).abs().max(1.0e-99);
    for k in (ktop + 1)..kbot {
        h_out[k] = v[k] * dr[k] / (0.5 * (v[k + 1] - v[k - 1]).abs()).max(1.0e-99);
    }
    h_out[ktop] = v[ktop] * (r[ktop] - r[ktop + 1]).max(1.0)
        / (v[ktop] - v[ktop + 1]).abs().max(1.0e-99);
}

struct RotDiffArrays {
    d_dsi: Vec<f64>, d_sh: Vec<f64>, d_ssi: Vec<f64>, d_es: Vec<f64>,
    d_gsf: Vec<f64>, d_st: Vec<f64>, nu_st: Vec<f64>, d_visc: Vec<f64>,
}

fn eval_heger_rot_diffusivities(model: &StarModel) -> RotDiffArrays {
    let nz = model.nz;
    let empty = || vec![0.0; nz];
    if !model.controls.rotation_flag || nz < 3 {
        return RotDiffArrays {
            d_dsi: empty(), d_sh: empty(), d_ssi: empty(), d_es: empty(),
            d_gsf: empty(), d_st: empty(), nu_st: empty(), d_visc: empty(),
        };
    }
    let mw = &model.controls.mix_wind;
    let f_mu = mw.am_gradmu_factor.max(0.0);
    let b_cow: std::borrow::Cow<'_, [f64]> = if use_smoothed_brunt_b(model) {
        std::borrow::Cow::Borrowed(&model.brunt_b)
    } else {
        std::borrow::Cow::Owned(smooth_triangular_signed(
            &model.brunt_b, mw.num_cells_for_smooth_brunt_b,
        ))
    };

    let (mut rho, mut t, mut p, mut cp, mut kap, mut cs, mut eps, mut abar, mut zbar) =
        (empty(), empty(), empty(), empty(), empty(), empty(), empty(), empty(), empty());
    let (mut grada, mut delta, mut hp, mut dgrad, mut dr, mut visc, mut is_rad) =
        (empty(), empty(), empty(), empty(), empty(), empty(), vec![false; nz]);
    for k in 0..nz {
        let (a, b) = if k == 0 {
            (1.0, 0.0)
        } else {
            let s = (model.dq[k - 1] + model.dq[k]).max(1.0e-99);
            (model.dq[k - 1] / s, model.dq[k] / s)
        };
        let km1 = k.saturating_sub(1);
        rho[k] = (a * model.rho[k] + b * model.rho[km1]).max(1.0e-30);
        t[k] = (a * model.t[k] + b * model.t[km1]).max(10.0);
        p[k] = (a * model.peos[k] + b * model.peos[km1]).max(1.0);
        cp[k] = (a * model.cp[k] + b * model.cp[km1]).max(1.0);
        kap[k] = (a * model.opacity[k] + b * model.opacity[km1]).max(1.0e-10);
        cs[k] = (a * model.csound[k] + b * model.csound[km1]).max(1.0e3);
        eps[k] = a * model.eps_nuc[k] + b * model.eps_nuc[km1];
        abar[k] = (a * model.abar[k] + b * model.abar[km1]).max(1.0);
        zbar[k] = (a * model.zbar[k] + b * model.zbar[km1]).max(1.0);
        grada[k] = a * model.grada[k] + b * model.grada[km1];
        delta[k] = (a * (model.chi_t[k] / model.chi_rho[k].max(1.0e-12))
            + b * (model.chi_t[km1] / model.chi_rho[km1].max(1.0e-12)))
        .max(1.0e-12);
        let (r_k, m_k, cg) = (model.r[k].max(1.0), model.m_grav[k].max(1.0), model.cgrav[k].max(1.0e-20));
        hp[k] = (p[k] * r_k * r_k / (cg * m_k * rho[k])).min((p[k] / cg).sqrt() / rho[k]).max(1.0);
        let dg = model.gradt[k] - grada[k];
        dgrad[k] = dg.abs().max(1.0e-99).copysign(if dg == 0.0 { -1.0 } else { dg });
        dr[k] = if k == 0 {
            (model.r[0] - model.r[1]).max(1.0)
        } else if k + 1 == nz {
            (model.r[nz - 1] - model.r_center).max(1.0)
        } else {
            (0.5 * (model.r[k - 1] - model.r[k + 1])).max(1.0)
        };
        is_rad[k] = !mw.skip_rotation_in_convection_zones
            || model.mixing_type[k] != MixingType::ConvectiveMixing;
        let rho6 = (rho[k] * 1.0e-6).max(1.0e-30);
        let gam = (0.2275 * zbar[k] * zbar[k] * (rho6 / abar[k]).cbrt() * 1.0e8 / t[k]).max(1.0e-20);
        let rm23 = (rho6 * zbar[k] / abar[k]).powf(2.0 / 3.0);
        let xi2 = ((PI / 3.0).sqrt() * zbar[k].ln() / 3.0
            + 2.0 * (1.32 + 2.33 / gam.sqrt()).ln()
            - (0.475 * (1.0 + 2.036 * rm23) - 0.276 * rm23) / (1.0 + 1.018 * rm23))
            .max(1.0e-6);
        let eta_s = (-0.016321227 + 1.0198850 * gam.powf(-1.9217970) + 0.024113535 * gam.powf(0.49999098)).max(1.0e-12);
        let dyn_v = 5.53e3 * zbar[k] * rho6.powf(5.0 / 6.0) * eta_s / abar[k].cbrt()
            + 4.0 * CRAD * t[k].powi(4) / (15.0 * CLIGHT * kap[k] * rho[k])
            + 1.893e6 * rm23.powf(2.5) / (zbar[k] * eta_s * xi2);
        visc[k] = if is_rad[k] { (dyn_v / rho[k]).clamp(0.0, hp[k] * cs[k]) } else { 0.0 };
    }

    let (mut t_dyn, mut v_es, mut v_gsf, mut dom_dlnr, mut ri_all, mut ris_max) =
        (vec![1.0; nz], empty(), empty(), empty(), vec![1.0; nz], vec![1.0; nz]);
    let (mut un_dsi, mut un_ssi, mut un_es, mut un_gsf) =
        (vec![false; nz], vec![false; nz], vec![false; nz], vec![false; nz]);
    let (mut raw_st, mut raw_nu_st) = (empty(), empty());

    for k in 0..nz {
        let (r_k, m_k, l_k) = (model.r[k].max(1.0e5), model.m_grav[k].max(1.0e-3 * M_SUN), model.lum[k]);
        let (g_k, cg, om_k) = (model.grav[k].max(1.0), model.cgrav[k].max(1.0e-20), model.omega[k]);
        t_dyn[k] = (r_k.powi(3) / (cg * m_k)).sqrt().max(1.0e-6);
        if !is_rad[k] {
            continue;
        }
        let t_kh = (cg * m_k * m_k / (r_k * l_k.abs().max(1.0))).max(1.0);
        let b_pos = b_cow[k].max(0.0);
        let brk = 2.0 * r_k * r_k * (eps[k] / l_k.abs().max(1.0e-3 * L_SUN) - 1.0 / m_k)
            - 3.0 / (FOUR_PI * rho[k] * r_k.max(1.0e-3 * R_SUN));
        let den_es = -dgrad[k] * delta[k] * (cg * m_k).powi(2);
        let ve0 = if den_es.abs() >= 1.0e-50 { grada[k] * om_k * om_k * r_k.powi(3) * l_k * brk / den_es } else { 0.0 };
        let ve_mu = (hp[k] / t_kh) * (f_mu * b_pos) / dgrad[k];
        let ves_k = ve0.abs() - ve_mu.abs();
        if ves_k > 0.0 && ves_k.is_finite() {
            un_es[k] = true;
            v_es[k] = ves_k;
        }
        if k == 0 || k + 1 == nz {
            continue;
        }

        dom_dlnr[k] = 0.5 * (model.omega[k - 1] - model.omega[k + 1]) * r_k / dr[k];
        let dlnr_dom = (1.0 / dom_dlnr[k].abs().max(1.0e-30)).copysign(dom_dlnr[k]);
        let dj = model.j_rot[k - 1] - model.j_rot[k];
        let hj = if dj.abs() > 0.0 { model.j_rot[k] * dr[k] / dj } else { 0.0 };
        let h_t = if model.gradt[k] > 1.0e-20 { hp[k] / model.gradt[k] } else { hp[k] };
        let den_gsf = 1.0 + 2.0 * om_k * dlnr_dom;
        if hj.abs() > 0.0 && den_gsf.abs() > 1.0e-30 {
            let vgsf_k = (ve0 * 2.0 * h_t * r_k / (hj * hj * den_gsf)).abs() - ve_mu.abs();
            if vgsf_k > 0.0 && vgsf_k.is_finite() {
                un_gsf[k] = true;
                v_gsf[k] = vgsf_k;
            }
        }

        let n2_mu = (g_k / hp[k]) * delta[k] * f_mu * b_pos;
        let n2_t = (g_k / hp[k]) * delta[k] * (-dgrad[k]);
        if n2_t + n2_mu <= 0.0 {
            continue;
        }
        if dgrad[k] < 0.0 && dom_dlnr[k].abs() > 1.0e-20 {
            let ri0 = (rho[k] * delta[k] / p[k]) * (dlnr_dom * g_k).powi(2);
            let (ri_t, ri_mu) = (ri0 * (-dgrad[k]).max(0.0), ri0 * f_mu * b_pos);
            ri_all[k] = ri_t + ri_mu;
            if ri_all[k] > 0.0 && ri_all[k] < RI_CRIT_DSI {
                un_dsi[k] = true;
            }
            let radcon = (4.0 * CRAD * CLIGHT * t[k].powi(3) / (3.0 * kap[k] * rho[k])).max(1.0e-30);
            let prandtl = visc[k] * rho[k] * cp[k] / radcon;
            let ris_1 = 0.125 * 2500.0 * prandtl * ri_t;
            if ris_1 <= RI_CRIT_DSI && ri_mu <= RI_CRIT_DSI {
                un_ssi[k] = true;
                ris_max[k] = ris_1.max(ri_mu);
            }
        }
        let q_st = if om_k.abs() > 1.0e-30 && (model.r[k - 1] - model.r[k + 1]).abs() > 0.0 {
            ((model.omega[k - 1] - model.omega[k + 1]) / (model.r[k - 1] - model.r[k + 1]) * (r_k / om_k)).abs()
        } else {
            0.0
        };
        if !matches!(model.mixing_type[k], MixingType::ConvectiveMixing | MixingType::OvershootMixing) {
            let (dst, nust, _, _) = eval_spruit_tayler_dynamo(
                r_k, rho[k], t[k], cp[k], kap[k], n2_t, n2_mu, om_k, q_st, abar[k], zbar[k], hp[k] * cs[k],
            );
            raw_st[k] = dst;
            raw_nu_st[k] = nust;
        }
    }
    raw_st[0] = raw_st[1];
    raw_st[nz - 1] = raw_st[nz - 2];
    raw_nu_st[0] = raw_nu_st[1];
    raw_nu_st[nz - 1] = raw_nu_st[nz - 2];

    let (mut raw_dsi, mut raw_ssi, mut raw_es, mut raw_gsf, mut v_tmp, mut h_buf) =
        (empty(), empty(), empty(), empty(), empty(), empty());
    for_each_unstable_block(&un_dsi, |ktop, kbot| {
        let inst_h = (model.r[ktop] - model.r[kbot]).max(0.0);
        for k in ktop..=kbot {
            v_tmp[k] = cs[k].min((hp[k] / t_dyn[k]) * (1.0 - (ri_all[k] / RI_CRIT_DSI).max(0.0)).powi(2));
        }
        fill_velocity_scale_heights(&model.r, &dr, &v_tmp, ktop, kbot, &mut h_buf);
        for k in ktop..=kbot {
            raw_dsi[k] = (inst_h.min(h_buf[k]).min(hp[k]) * v_tmp[k]).min(hp[k] * cs[k]);
        }
    });
    for_each_unstable_block(&un_ssi, |ktop, kbot| {
        for k in ktop..=kbot {
            v_tmp[k] = (visc[k] / 2500.0 * dom_dlnr[k].abs()).sqrt();
        }
        fill_velocity_scale_heights(&model.r, &dr, &v_tmp, ktop, kbot, &mut h_buf);
        for k in ktop..=kbot {
            let f = (1.0 - (ris_max[k] / RI_CRIT_DSI).max(0.0)).powi(2);
            raw_ssi[k] = (h_buf[k].min(hp[k]) * v_tmp[k].min(cs[k]) * f).min(hp[k] * cs[k]);
        }
    });
    for_each_unstable_block(&un_es, |ktop, kbot| {
        let inst_h = (model.r[ktop] - model.r[kbot]).max(0.0);
        fill_velocity_scale_heights(&model.r, &dr, &v_es, ktop, kbot, &mut h_buf);
        for k in ktop..=kbot {
            raw_es[k] = (inst_h.min(h_buf[k]).min(hp[k]) * v_es[k].min(cs[k])).min(hp[k] * cs[k]);
        }
    });
    for_each_unstable_block(&un_gsf, |ktop, kbot| {
        let inst_h = (model.r[ktop] - model.r[kbot]).max(0.0);
        fill_velocity_scale_heights(&model.r, &dr, &v_gsf, ktop, kbot, &mut h_buf);
        for k in ktop..=kbot {
            raw_gsf[k] = (inst_h.min(h_buf[k]).min(hp[k]) * v_gsf[k].min(cs[k])).min(hp[k] * cs[k]);
        }
    });

    let sm3 = |arr: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; nz];
        if is_rad[0] { out[0] = arr[0]; }
        if is_rad[nz - 1] { out[nz - 1] = arr[nz - 1]; }
        for j in 1..(nz - 1) {
            if is_rad[j] {
                out[j] = 0.25 * arr[j - 1] + 0.5 * arr[j] + 0.25 * arr[j + 1];
            }
        }
        out
    };
    RotDiffArrays {
        d_dsi: sm3(&raw_dsi), d_sh: empty(), d_ssi: sm3(&raw_ssi), d_es: sm3(&raw_es),
        d_gsf: sm3(&raw_gsf), d_st: sm3(&raw_st), nu_st: sm3(&raw_nu_st), d_visc: visc,
    }
}

pub(crate) fn augment_d_mix_with_rotation(model: &mut StarModel) {
    let mw = model.controls.mix_wind.clone();
    if !model.controls.rotation_flag || (mw.am_d_mix_factor <= 0.0 && mw.d_visc_factor <= 0.0) {
        return;
    }
    let diffs = eval_heger_rot_diffusivities(model);
    for k in 0..model.nz {
        let d_rot_k = mw.d_visc_factor.max(0.0) * diffs.d_visc[k]
            + mw.am_d_mix_factor.max(0.0)
                * (mw.d_dsi_factor * diffs.d_dsi[k] + mw.d_sh_factor * diffs.d_sh[k]
                    + mw.d_ssi_factor * diffs.d_ssi[k] + mw.d_es_factor * diffs.d_es[k]
                    + mw.d_gsf_factor * diffs.d_gsf[k] + mw.d_st_factor * diffs.d_st[k]);
        if d_rot_k > 0.0 {
            if model.mixing_type[k] != MixingType::ConvectiveMixing && d_rot_k > model.d_mix[k] {
                model.mixing_type[k] = MixingType::RotationMixing;
            }
            model.d_mix[k] += d_rot_k;
        }
    }
}

pub(crate) fn solve_omega_transport(model: &mut StarModel, dt_transport: f64) {
    let nz = model.nz;
    if !model.controls.rotation_flag || nz < 2 || dt_transport <= 0.0 {
        return;
    }
    let mw = model.controls.mix_wind.clone();
    let diffs = eval_heger_rot_diffusivities(model);
    let f_nu = |am: f64, d: f64| if am < 0.0 { d.max(0.0) } else { am.max(0.0) };
    let nu_omega: Vec<f64> = (0..nz)
        .map(|k| {
            let d_rot_chem = mw.d_visc_factor.max(0.0) * diffs.d_visc[k]
                + mw.am_d_mix_factor.max(0.0)
                    * (mw.d_dsi_factor * diffs.d_dsi[k] + mw.d_sh_factor * diffs.d_sh[k]
                        + mw.d_ssi_factor * diffs.d_ssi[k] + mw.d_es_factor * diffs.d_es[k]
                        + mw.d_gsf_factor * diffs.d_gsf[k] + mw.d_st_factor * diffs.d_st[k]);
            let d_non_rot = match model.mixing_type[k] {
                MixingType::NoMixing => 0.0,
                _ => (model.d_mix[k] - d_rot_chem).max(0.0),
            };
            let nu_rot = f_nu(mw.am_nu_dsi_factor, mw.d_dsi_factor) * diffs.d_dsi[k]
                + f_nu(mw.am_nu_sh_factor, mw.d_sh_factor) * diffs.d_sh[k]
                + f_nu(mw.am_nu_ssi_factor, mw.d_ssi_factor) * diffs.d_ssi[k]
                + f_nu(mw.am_nu_es_factor, mw.d_es_factor) * diffs.d_es[k]
                + f_nu(mw.am_nu_gsf_factor, mw.d_gsf_factor) * diffs.d_gsf[k]
                + f_nu(mw.am_nu_st_factor, mw.d_st_factor) * diffs.nu_st[k]
                + f_nu(mw.am_nu_visc_factor, mw.d_visc_factor) * diffs.d_visc[k];
            (mw.am_nu_factor.max(0.0) * (mw.am_nu_non_rotation_factor.max(0.0) * d_non_rot + nu_rot)).max(0.0)
        })
        .collect();
    if nu_omega.iter().all(|&v| v <= 0.0) {
        return;
    }

    let i_rot: Vec<f64> = (0..nz)
        .map(|k| {
            model.aux.i_rot.get(k).copied().filter(|&v| v > 0.0)
                .unwrap_or_else(|| (2.0 / 3.0) * model.r[k].max(1.0).powi(2))
        })
        .collect();
    let j_tot_before: f64 = (0..nz).map(|k| model.dm_bar[k] * i_rot[k] * model.omega[k]).sum();
    let sig_cap = mw.am_sig_term_limit.clamp(1.0e4, 1.0e10);
    let dm_eff = |k: usize| if model.dm_bar[k] > 0.0 { model.dm_bar[k] } else { 0.5 * model.dm[k].max(1.0e-30) };

    let mut c_omega = vec![0.0; nz - 1];
    for (k, c_om_k) in c_omega.iter_mut().enumerate() {
        let (nu_0, nu_1) = (nu_omega[k].max(0.0), nu_omega[k + 1].max(0.0));
        let nu_cell = 2.0 * nu_0 * nu_1 / (nu_0 + nu_1).max(1.0e-99);
        let r_mid = 0.5 * (model.r[k] + model.r[k + 1]);
        let area_rho = FOUR_PI * r_mid * r_mid * model.rho[k];
        let am_sig_raw = (area_rho * area_rho * nu_cell) / model.dm[k].max(1.0e-99);
        let am_sig_max = sig_cap * model.xmstar.max(1.0) * model.dq[k].min(model.dq[k + 1]).max(1.0e-30) / dt_transport;
        let denom_min = (dm_eff(k) * i_rot[k]).min(dm_eff(k + 1) * i_rot[k + 1]).max(1.0e-99);
        *c_om_k = (am_sig_raw.min(am_sig_max) * 0.5 * (i_rot[k] + i_rot[k + 1])).min(sig_cap * denom_min / dt_transport);
    }

    let dt_min = (0..nz).fold(1.0e99_f64, |acc, k| {
        let t_m1 = if k > 0 { c_omega[k - 1] * (model.omega[k - 1] - model.omega[k]) } else { 0.0 };
        let t_00 = if k + 1 < nz { c_omega[k] * (model.omega[k] - model.omega[k + 1]) } else { 0.0 };
        let d2om = (t_m1 - t_00).abs() / dm_eff(k).max(1.0e-99);
        acc.min(model.omega[k].abs().max(1.0e-12) * i_rot[k] / d2om.max(1.0e-50))
    });
    let steps = ((dt_transport / (0.5 * dt_min).max(1.0e-6 * dt_transport)).ceil() as usize).clamp(4, 20);
    let dt_sub = dt_transport / (steps as f64);

    let (mut dl, mut d, mut du) = (vec![0.0; nz], vec![1.0; nz], vec![0.0; nz]);
    for k in 0..nz {
        let f_k = dt_sub / (dm_eff(k) * i_rot[k]).max(1.0e-99);
        let c_m1 = if k > 0 { c_omega[k - 1] } else { 0.0 };
        let c_00 = if k + 1 < nz { c_omega[k] } else { 0.0 };
        dl[k] = -f_k * c_m1;
        d[k] = 1.0 + f_k * (c_m1 + c_00);
        du[k] = -f_k * c_00;
    }

    let factor = ScalarTridiagFactor::factor(&dl, &d, &du);
    let (mut rhs, mut d_omega, mut d_prime) = (vec![0.0; nz], vec![0.0; nz], vec![0.0; nz]);
    for _ in 0..steps {
        for (k, r_k) in rhs.iter_mut().enumerate() {
            let f_k = dt_sub / (dm_eff(k) * i_rot[k]).max(1.0e-99);
            let t_m1 = if k > 0 { c_omega[k - 1] * (model.omega[k - 1] - model.omega[k]) } else { 0.0 };
            let t_00 = if k + 1 < nz { c_omega[k] * (model.omega[k] - model.omega[k + 1]) } else { 0.0 };
            *r_k = f_k * (t_m1 - t_00);
        }
        factor.solve_into(&dl, &rhs, &mut d_omega, &mut d_prime);
        if d_omega.iter().all(|dw| dw.is_finite()) {
            for (k, &dw) in d_omega.iter().enumerate().skip(1) {
                model.omega[k] += dw;
            }
            model.omega[0] = model.omega[1];
        } else {
            break;
        }
    }
    let j_tot_after: f64 = (0..nz).map(|k| model.dm_bar[k] * i_rot[k] * model.omega[k]).sum();
    let scale = if j_tot_before.abs() > 0.0 && j_tot_after.abs() > 1.0e-30 && j_tot_before * j_tot_after > 0.0 {
        j_tot_before / j_tot_after
    } else {
        1.0
    };
    for (k, &ir) in i_rot.iter().enumerate() {
        model.omega[k] *= scale;
        model.j_rot[k] = ir * model.omega[k];
    }
}

pub(crate) fn sync_post_henyey_rotation(model: &mut StarModel, dt: f64) {
    let nz = model.nz;
    if !model.controls.rotation_flag || nz == 0 || model.j_rot.len() != nz {
        return;
    }
    if dt <= 0.0 {
        conserve_shell_j_rot_and_distortion(model);
        return;
    }
    conserve_shell_j_rot_and_distortion(model);
    if nz >= 2 && model.controls.mix_wind.am_nu_factor > 0.0 {
        let dt_tr = if model.controls.mix_wind.premix_omega { 0.5 * dt } else { dt };
        solve_omega_transport(model, dt_tr);
        conserve_shell_j_rot_and_distortion(model);
    }
}
