// SPDX-License-Identifier: LGPL-3.0-or-later
//! Array-level Heger, Langer & Woosley (2000) and Spruit (2002) rotational
//! instability diffusivities (`rotation_mix_info.f90:493-1345`) and
//! `update_rotation_mixing_info` (`mix_info.f90:1561-1740`).

use flexmesa_const::{
    MixingType, CLIGHT, CRAD, FOUR_PI, L_SUN, M_SUN, PI, R_SUN,
};

use crate::controls::StarError;
use crate::model::StarModel;
use crate::rotation::{
    conserve_shell_j_rot_and_distortion, eval_spruit_tayler_dynamo, RI_CRIT_DSI,
};

fn for_each_unstable_block(
    unstable: &[bool],
    mut body: impl FnMut(usize, usize),
) {
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
    r: &[f64],
    dr: &[f64],
    v: &[f64],
    ktop: usize,
    kbot: usize,
    h_out: &mut [f64],
) {
    h_out[kbot] = v[kbot] * (r[kbot - 1] - r[kbot]).max(1.0)
        / (v[kbot - 1] - v[kbot]).abs().max(1.0e-99);
    for k in (ktop + 1)..kbot {
        h_out[k] =
            v[k] * dr[k] / (0.5 * (v[k + 1] - v[k - 1]).abs()).max(1.0e-99);
    }
    h_out[ktop] = v[ktop] * (r[ktop] - r[ktop + 1]).max(1.0)
        / (v[ktop] - v[ktop + 1]).abs().max(1.0e-99);
}

struct RotDiffArrays {
    d_dsi: Vec<f64>,
    d_sh: Vec<f64>,
    d_ssi: Vec<f64>,
    d_es: Vec<f64>,
    d_gsf: Vec<f64>,
    d_st: Vec<f64>,
    nu_st: Vec<f64>,
    d_visc: Vec<f64>,
}

fn eval_heger_rot_diffusivities(model: &StarModel) -> RotDiffArrays {
    let nz = model.nz;
    let empty = || vec![0.0; nz];
    if !model.controls.rotation_flag || nz < 3 {
        return RotDiffArrays {
            d_dsi: empty(),
            d_sh: empty(),
            d_ssi: empty(),
            d_es: empty(),
            d_gsf: empty(),
            d_st: empty(),
            nu_st: empty(),
            d_visc: empty(),
        };
    }
    let mw = &model.controls.mix_wind;
    let f_mu = mw.am_gradmu_factor.max(0.0);
    let b_cow: std::borrow::Cow<'_, [f64]> =
        std::borrow::Cow::Borrowed(&model.brunt_b);

    let (
        mut rho,
        mut t,
        mut p,
        mut cp,
        mut kap,
        mut cs,
        mut eps,
        mut abar,
        mut zbar,
    ) = (
        empty(),
        empty(),
        empty(),
        empty(),
        empty(),
        empty(),
        empty(),
        empty(),
        empty(),
    );
    let (mut grada, mut delta, mut hp, mut dgrad, mut dr, mut visc, mut is_rad) = (
        empty(),
        empty(),
        empty(),
        empty(),
        empty(),
        empty(),
        vec![false; nz],
    );
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
        let (r_k, m_k, cg) = (
            model.r[k].max(1.0),
            model.m_grav[k].max(1.0),
            model.cgrav[k].max(1.0e-20),
        );
        hp[k] = (p[k] * r_k * r_k / (cg * m_k * rho[k]))
            .min((p[k] / cg).sqrt() / rho[k])
            .max(1.0);
        let dg = model.gradt[k] - grada[k];
        dgrad[k] =
            dg.abs()
                .max(1.0e-99)
                .copysign(if dg == 0.0 { -1.0 } else { dg });
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
        let gam = (0.2275
            * zbar[k]
            * zbar[k]
            * flexmesa_math::pow(rho6 / abar[k], 1.0 / 3.0)
            * 1.0e8
            / t[k])
            .max(1.0e-20);
        let rm23 = flexmesa_math::pow(rho6 * zbar[k] / abar[k], 2.0 / 3.0);
        let xi2 = ((PI / 3.0).sqrt() * flexmesa_math::log(zbar[k]) / 3.0
            + 2.0 * flexmesa_math::log(1.32 + 2.33 / gam.sqrt())
            - (0.475 * (1.0 + 2.036 * rm23) - 0.276 * rm23)
                / (1.0 + 1.018 * rm23))
            .max(1.0e-6);
        let eta_s = (-0.016321227
            + 1.0198850 * flexmesa_math::pow(gam, -1.9217970)
            + 0.024113535 * flexmesa_math::pow(gam, 0.49999098))
        .max(1.0e-12);
        let dyn_v =
            5.53e3 * zbar[k] * flexmesa_math::pow(rho6, 5.0 / 6.0) * eta_s
                / flexmesa_math::pow(abar[k], 1.0 / 3.0)
                + 4.0 * CRAD * flexmesa_math::pow4(t[k])
                    / (15.0 * CLIGHT * kap[k] * rho[k])
                + 1.893e6 * flexmesa_math::pow(rm23, 2.5)
                    / (zbar[k] * eta_s * xi2);
        visc[k] = if is_rad[k] {
            (dyn_v / rho[k]).clamp(0.0, hp[k] * cs[k])
        } else {
            0.0
        };
    }

    let (mut t_dyn, mut v_es, mut v_gsf, mut dom_dlnr, mut ri_all, mut ris_max) = (
        vec![1.0; nz],
        empty(),
        empty(),
        empty(),
        vec![1.0; nz],
        vec![1.0; nz],
    );
    let (mut un_dsi, mut un_ssi, mut un_es, mut un_gsf) = (
        vec![false; nz],
        vec![false; nz],
        vec![false; nz],
        vec![false; nz],
    );
    let (mut raw_st, mut raw_nu_st) = (empty(), empty());

    for k in 0..nz {
        let (r_k, m_k, l_k) = (
            model.r[k].max(1.0e5),
            model.m_grav[k].max(1.0e-3 * M_SUN),
            model.lum[k],
        );
        let (g_k, cg, om_k) = (
            model.grav[k].max(1.0),
            model.cgrav[k].max(1.0e-20),
            model.omega[k],
        );
        t_dyn[k] = (flexmesa_math::pow3(r_k) / (cg * m_k)).sqrt().max(1.0e-6);
        if !is_rad[k] {
            continue;
        }
        let t_kh = (cg * m_k * m_k / (r_k * l_k.abs().max(1.0))).max(1.0);
        let b_pos = b_cow[k].max(0.0);
        let brk = 2.0
            * r_k
            * r_k
            * (eps[k] / l_k.abs().max(1.0e-3 * L_SUN) - 1.0 / m_k)
            - 3.0 / (FOUR_PI * rho[k] * r_k.max(1.0e-3 * R_SUN));
        let den_es = -dgrad[k] * delta[k] * flexmesa_math::pow2(cg * m_k);
        let ve0 = if den_es.abs() >= 1.0e-50 {
            grada[k] * om_k * om_k * flexmesa_math::pow3(r_k) * l_k * brk
                / den_es
        } else {
            0.0
        };
        let ve_mu = (hp[k] / t_kh) * (f_mu * b_pos) / dgrad[k];
        let ves_k = ve0.abs() - ve_mu.abs();
        if ves_k > 0.0 && ves_k.is_finite() {
            un_es[k] = true;
            v_es[k] = ves_k;
        }
        if k == 0 || k + 1 == nz {
            continue;
        }

        dom_dlnr[k] =
            0.5 * (model.omega[k - 1] - model.omega[k + 1]) * r_k / dr[k];
        let dlnr_dom =
            (1.0 / dom_dlnr[k].abs().max(1.0e-30)).copysign(dom_dlnr[k]);
        let dj = model.j_rot[k - 1] - model.j_rot[k];
        let hj = if dj.abs() > 0.0 {
            model.j_rot[k] * dr[k] / dj
        } else {
            0.0
        };
        let h_t = if model.gradt[k] > 1.0e-20 {
            hp[k] / model.gradt[k]
        } else {
            hp[k]
        };
        let den_gsf = 1.0 + 2.0 * om_k * dlnr_dom;
        if hj.abs() > 0.0 && den_gsf.abs() > 1.0e-30 {
            let vgsf_k = (ve0 * 2.0 * h_t * r_k / (hj * hj * den_gsf)).abs()
                - ve_mu.abs();
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
            let ri0 = (rho[k] * delta[k] / p[k])
                * flexmesa_math::pow2(dlnr_dom * g_k);
            let (ri_t, ri_mu) =
                (ri0 * (-dgrad[k]).max(0.0), ri0 * f_mu * b_pos);
            ri_all[k] = ri_t + ri_mu;
            if ri_all[k] > 0.0 && ri_all[k] < RI_CRIT_DSI {
                un_dsi[k] = true;
            }
            let radcon = (4.0 * CRAD * CLIGHT * flexmesa_math::pow3(t[k])
                / (3.0 * kap[k] * rho[k]))
                .max(1.0e-30);
            let prandtl = visc[k] * rho[k] * cp[k] / radcon;
            let ris_1 = 0.125 * 2500.0 * prandtl * ri_t;
            if ris_1 <= RI_CRIT_DSI && ri_mu <= RI_CRIT_DSI {
                un_ssi[k] = true;
                ris_max[k] = ris_1.max(ri_mu);
            }
        }
        let q_st = if om_k.abs() > 1.0e-30
            && (model.r[k - 1] - model.r[k + 1]).abs() > 0.0
        {
            ((model.omega[k - 1] - model.omega[k + 1])
                / (model.r[k - 1] - model.r[k + 1])
                * (r_k / om_k))
                .abs()
        } else {
            0.0
        };
        if !matches!(
            model.mixing_type[k],
            MixingType::ConvectiveMixing | MixingType::OvershootMixing
        ) {
            let (dst, nust, _, _) = eval_spruit_tayler_dynamo(
                r_k,
                rho[k],
                t[k],
                cp[k],
                kap[k],
                n2_t,
                n2_mu,
                om_k,
                q_st,
                abar[k],
                zbar[k],
                hp[k] * cs[k],
            );
            raw_st[k] = dst;
            raw_nu_st[k] = nust;
        }
    }
    raw_st[0] = raw_st[1];
    raw_st[nz - 1] = raw_st[nz - 2];
    raw_nu_st[0] = raw_nu_st[1];
    raw_nu_st[nz - 1] = raw_nu_st[nz - 2];

    let (
        mut raw_dsi,
        mut raw_ssi,
        mut raw_es,
        mut raw_gsf,
        mut v_tmp,
        mut h_buf,
    ) = (empty(), empty(), empty(), empty(), empty(), empty());
    for_each_unstable_block(&un_dsi, |ktop, kbot| {
        let inst_h = (model.r[ktop] - model.r[kbot]).max(0.0);
        for k in ktop..=kbot {
            v_tmp[k] = cs[k].min(
                (hp[k] / t_dyn[k])
                    * flexmesa_math::pow2(
                        1.0 - (ri_all[k] / RI_CRIT_DSI).max(0.0),
                    ),
            );
        }
        fill_velocity_scale_heights(
            &model.r, &dr, &v_tmp, ktop, kbot, &mut h_buf,
        );
        for k in ktop..=kbot {
            raw_dsi[k] =
                (inst_h.min(h_buf[k]).min(hp[k]) * v_tmp[k]).min(hp[k] * cs[k]);
        }
    });
    for_each_unstable_block(&un_ssi, |ktop, kbot| {
        for k in ktop..=kbot {
            v_tmp[k] = (visc[k] / 2500.0 * dom_dlnr[k].abs()).sqrt();
        }
        fill_velocity_scale_heights(
            &model.r, &dr, &v_tmp, ktop, kbot, &mut h_buf,
        );
        for k in ktop..=kbot {
            let f =
                flexmesa_math::pow2(1.0 - (ris_max[k] / RI_CRIT_DSI).max(0.0));
            raw_ssi[k] = (h_buf[k].min(hp[k]) * v_tmp[k].min(cs[k]) * f)
                .min(hp[k] * cs[k]);
        }
    });
    for_each_unstable_block(&un_es, |ktop, kbot| {
        let inst_h = (model.r[ktop] - model.r[kbot]).max(0.0);
        fill_velocity_scale_heights(
            &model.r, &dr, &v_es, ktop, kbot, &mut h_buf,
        );
        for k in ktop..=kbot {
            raw_es[k] = (inst_h.min(h_buf[k]).min(hp[k]) * v_es[k].min(cs[k]))
                .min(hp[k] * cs[k]);
        }
    });
    for_each_unstable_block(&un_gsf, |ktop, kbot| {
        let inst_h = (model.r[ktop] - model.r[kbot]).max(0.0);
        fill_velocity_scale_heights(
            &model.r, &dr, &v_gsf, ktop, kbot, &mut h_buf,
        );
        for k in ktop..=kbot {
            raw_gsf[k] = (inst_h.min(h_buf[k]).min(hp[k])
                * v_gsf[k].min(cs[k]))
            .min(hp[k] * cs[k]);
        }
    });

    let sm3 = |arr: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; nz];
        if is_rad[0] {
            out[0] = arr[0];
        }
        if is_rad[nz - 1] {
            out[nz - 1] = arr[nz - 1];
        }
        for j in 1..(nz - 1) {
            if is_rad[j] {
                out[j] = 0.25 * arr[j - 1] + 0.5 * arr[j] + 0.25 * arr[j + 1];
            }
        }
        out
    };
    RotDiffArrays {
        d_dsi: sm3(&raw_dsi),
        d_sh: empty(),
        d_ssi: sm3(&raw_ssi),
        d_es: sm3(&raw_es),
        d_gsf: sm3(&raw_gsf),
        d_st: sm3(&raw_st),
        nu_st: sm3(&raw_nu_st),
        d_visc: visc,
    }
}

pub(crate) fn eval_d_mix_non_rotation_vec(model: &StarModel) -> Vec<f64> {
    let nz = model.nz;
    if model.aux.mixing.have_mixing_info
        && model.aux.mixing.d_mix_non_rotation.len() == nz
    {
        return model.aux.mixing.d_mix_non_rotation.clone();
    }
    let mw = &model.controls.mix_wind;
    if !model.controls.rotation_flag
        || (mw.am_d_mix_factor <= 0.0 && mw.d_visc_factor <= 0.0)
    {
        return (0..nz)
            .map(|k| match model.mixing_type.get(k).copied() {
                Some(MixingType::NoMixing | MixingType::RotationMixing)
                | None => 0.0,
                Some(_) => model.d_mix.get(k).copied().unwrap_or(0.0),
            })
            .collect();
    }
    let diffs = eval_heger_rot_diffusivities(model);
    (0..nz)
        .map(|k| match model.mixing_type.get(k).copied() {
            Some(MixingType::NoMixing | MixingType::RotationMixing) | None => {
                0.0
            }
            Some(_) => {
                let d_rot_k = mw.d_visc_factor.max(0.0) * diffs.d_visc[k]
                    + mw.am_d_mix_factor.max(0.0)
                        * (mw.d_dsi_factor * diffs.d_dsi[k]
                            + mw.d_sh_factor * diffs.d_sh[k]
                            + mw.d_ssi_factor * diffs.d_ssi[k]
                            + mw.d_es_factor * diffs.d_es[k]
                            + mw.d_gsf_factor * diffs.d_gsf[k]
                            + mw.d_st_factor * diffs.d_st[k]);
                (model.d_mix[k] - d_rot_k).max(0.0)
            }
        })
        .collect()
}

/// Evaluates rotational chemical diffusivities and angular-momentum viscosities
/// (`am_nu_non_rot`, `am_nu_rot`, `am_nu_omega`, `am_nu_j`), sets
/// `model.d_mix[k] = d_mix_non_rotation[k] + d_mix_rotation[k]`, and returns
/// `d_mix_rotation` (`update_rotation_mixing_info` in
/// `star/private/mix_info.f90:1561-1740`).
///
/// # Errors
/// Returns [`StarError::ConvergenceFailure`] if any face yields a negative or
/// non-finite `am_nu_omega` or `am_nu_j`.
pub(crate) fn update_rotation_mixing_info(
    model: &mut StarModel,
) -> Result<Vec<f64>, StarError> {
    let nz = model.nz;
    let mut d_mix_rotation = vec![0.0; nz];
    model.aux.am_nu_non_rot.resize(nz, 0.0);
    model.aux.am_nu_rot.resize(nz, 0.0);
    model.aux.am_nu_omega.resize(nz, 0.0);
    model.aux.am_nu_j.resize(nz, 0.0);
    model.aux.am_nu_non_rot[..nz].fill(0.0);
    model.aux.am_nu_rot[..nz].fill(0.0);
    model.aux.am_nu_omega[..nz].fill(0.0);
    model.aux.am_nu_j[..nz].fill(0.0);
    if !model.controls.rotation_flag || nz == 0 {
        return Ok(d_mix_rotation);
    }
    if model.aux.mixing.d_mix_non_rotation.len() != nz {
        model.aux.mixing.d_mix_non_rotation =
            eval_d_mix_non_rotation_vec(model);
    }

    let mw = model.controls.mix_wind.clone();
    let diffs = eval_heger_rot_diffusivities(model);

    for k in 0..nz {
        let d_rot_k = if mw.am_d_mix_factor > 0.0 || mw.d_visc_factor > 0.0 {
            mw.d_visc_factor.max(0.0) * diffs.d_visc[k]
                + mw.am_d_mix_factor.max(0.0)
                    * (mw.d_dsi_factor * diffs.d_dsi[k]
                        + mw.d_sh_factor * diffs.d_sh[k]
                        + mw.d_ssi_factor * diffs.d_ssi[k]
                        + mw.d_es_factor * diffs.d_es[k]
                        + mw.d_gsf_factor * diffs.d_gsf[k]
                        + mw.d_st_factor * diffs.d_st[k])
        } else {
            0.0
        };
        d_mix_rotation[k] = d_rot_k;
        model.d_mix[k] = model.aux.mixing.d_mix_non_rotation[k] + d_rot_k;
    }

    let am_nu_dsi = if mw.am_nu_dsi_factor < 0.0 {
        mw.d_dsi_factor
    } else {
        mw.am_nu_dsi_factor
    };
    let am_nu_sh = if mw.am_nu_sh_factor < 0.0 {
        mw.d_sh_factor
    } else {
        mw.am_nu_sh_factor
    };
    let am_nu_ssi = if mw.am_nu_ssi_factor < 0.0 {
        mw.d_ssi_factor
    } else {
        mw.am_nu_ssi_factor
    };
    let am_nu_es = if mw.am_nu_es_factor < 0.0 {
        mw.d_es_factor
    } else {
        mw.am_nu_es_factor
    };
    let am_nu_gsf = if mw.am_nu_gsf_factor < 0.0 {
        mw.d_gsf_factor
    } else {
        mw.am_nu_gsf_factor
    };
    let am_nu_st = if mw.am_nu_st_factor < 0.0 {
        mw.d_st_factor
    } else {
        mw.am_nu_st_factor
    };
    let am_nu_visc = if mw.am_nu_visc_factor < 0.0 {
        mw.d_visc_factor
    } else {
        mw.am_nu_visc_factor
    };

    for k in 0..nz {
        let non_rot = mw.am_nu_factor
            * mw.am_nu_non_rotation_factor
            * model.aux.mixing.d_mix_non_rotation[k];
        let rot = (mw.am_nu_factor
            * (am_nu_visc * diffs.d_visc[k]
                + am_nu_dsi * diffs.d_dsi[k]
                + am_nu_sh * diffs.d_sh[k]
                + am_nu_ssi * diffs.d_ssi[k]
                + am_nu_es * diffs.d_es[k]
                + am_nu_gsf * diffs.d_gsf[k]
                + am_nu_st * diffs.nu_st[k]))
            .max(0.0);
        let nu_omega = mw.am_nu_omega_non_rot_factor * non_rot
            + mw.am_nu_omega_rot_factor * rot;
        if nu_omega < 0.0 || !nu_omega.is_finite() {
            return Err(StarError::ConvergenceFailure(format!(
                "update_rotation_mixing_info: invalid am_nu_omega[{k}] = {nu_omega:.6e}"
            )));
        }
        let nu_j = mw.am_nu_j_non_rot_factor * non_rot
            + mw.am_nu_j_rot_factor * rot;
        if nu_j < 0.0 || !nu_j.is_finite() {
            return Err(StarError::ConvergenceFailure(format!(
                "update_rotation_mixing_info: invalid am_nu_j[{k}] = {nu_j:.6e}"
            )));
        }
        model.aux.am_nu_non_rot[k] = non_rot;
        model.aux.am_nu_rot[k] = rot;
        model.aux.am_nu_omega[k] = nu_omega;
        model.aux.am_nu_j[k] = nu_j;
    }

    Ok(d_mix_rotation)
}

pub(crate) fn augment_d_mix_with_rotation(model: &mut StarModel) {
    let nz = model.nz;
    model.aux.mixing.d_mix_non_rotation.resize(nz, 0.0);
    model
        .aux
        .mixing
        .d_mix_non_rotation
        .copy_from_slice(&model.d_mix[..nz]);
    let Ok(d_mix_rotation) = update_rotation_mixing_info(model) else {
        return;
    };
    for k in 0..nz {
        if model.d_mix[k] != 0.0
            && model.aux.mixing.d_mix_non_rotation[k] < d_mix_rotation[k]
        {
            model.mixing_type[k] = MixingType::RotationMixing;
        }
    }
}

pub(crate) fn sync_post_henyey_rotation(model: &mut StarModel, dt: f64) {
    let nz = model.nz;
    if !model.controls.rotation_flag || nz == 0 || model.j_rot.len() != nz {
        return;
    }
    conserve_shell_j_rot_and_distortion(model);
    if dt <= 0.0 || nz < 2 {
        return;
    }
    let mw = &model.controls.mix_wind;
    let d_mix_changed = model.aux.mixing.d_mix_non_rotation.len() != nz
        || (mw.am_d_mix_factor <= 0.0
            && mw.d_visc_factor <= 0.0
            && model.d_mix[..nz] != model.aux.mixing.d_mix_non_rotation[..nz]);
    let nu_unpopulated = model.aux.am_nu_omega.len() != nz
        || (model.aux.am_nu_omega.iter().all(|&v| v == 0.0)
            && model.aux.am_nu_j.iter().all(|&v| v == 0.0)
            && model.d_mix.iter().any(|&v| v > 0.0));
    if d_mix_changed || nu_unpopulated {
        model.aux.mixing.d_mix_non_rotation.resize(nz, 0.0);
        model
            .aux
            .mixing
            .d_mix_non_rotation
            .copy_from_slice(&model.d_mix[..nz]);
        let _ = update_rotation_mixing_info(model);
    }
    let dt_tr = if model.controls.mix_wind.premix_omega {
        0.5 * dt
    } else {
        dt
    };
    let _ = crate::omega_mix::do_solve_omega_mix(model, dt_tr);
    crate::hydro_rotation::set_rotation_info(model, false);
}
