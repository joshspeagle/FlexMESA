// SPDX-License-Identifier: LGPL-3.0-or-later
//! Per-channel soft and hard timestep limit evaluators (`check_change`,
//! `check_integer_limit`, `check_dX`, `check_dX_nuc_drop`, structural
//! `delta_lg*`, nuclear `check_lgL`, HR-diagram, Courant, and energy-error
//! checks in `star/private/timestep.f90:384-2246`).

use std::f64::consts::LN_10;

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{MixingType, L_SUN, M_SUN, SECYER};

use crate::controls_timestep_stop::MAX_DX_LIMIT_CTRLS;
use crate::core_boundaries::compute_power_categories;
use crate::model::StarModel;

#[inline]
pub(crate) fn check_change(
    delta_value: f64,
    lim_in: f64,
    hard_lim_in: f64,
    time_delta_coeff: f64,
    msg: &str,
    skip_hard_limit: bool,
) -> Result<f64, String> {
    if !delta_value.is_finite() {
        return Err(format!("{msg} non-finite delta"));
    }
    let abs_change = delta_value.abs();
    let lim = lim_in * time_delta_coeff;
    let hard_lim = hard_lim_in * time_delta_coeff;
    if hard_lim > 0.0 && abs_change > hard_lim && !skip_hard_limit {
        return Err(format!("{msg} hard limit"));
    }
    if lim <= 0.0 {
        return Ok(0.0);
    }
    let r = abs_change / lim;
    Ok(if r <= 1.0 { 0.0 } else { r })
}

#[inline]
pub(crate) fn check_integer_limit(
    limit: isize,
    hard_limit: isize,
    value: usize,
    msg: &str,
    skip_hard_limit: bool,
) -> Result<f64, String> {
    let val_i = value as isize;
    if hard_limit > 0 && val_i > hard_limit && !skip_hard_limit {
        return Err(format!("{msg} hard limit"));
    }
    if value == 0 || limit <= 0 {
        return Ok(0.0);
    }
    let r = (value as f64) / (limit as f64) - 0.05;
    Ok(if r <= 1.0 { 0.0 } else { r })
}

pub(crate) fn collect_mixing_boundary_qs(model: &StarModel) -> Vec<f64> {
    let mut qs = Vec::new();
    for k in 1..model.nz {
        let m_prev = model.mixing_type[k - 1] != MixingType::NoMixing;
        let m_curr = model.mixing_type[k] != MixingType::NoMixing;
        if m_prev != m_curr {
            qs.push(model.q[k]);
        }
    }
    qs
}

pub(crate) fn check_dx_limits(
    model: &StarModel,
    skip_hard_limit: bool,
    mix_bdy_q: &[f64],
) -> Result<f64, String> {
    let c = &model.controls;
    let ts = &c.ts_stop;
    let (nz, sp) = (model.nz, model.species);
    if nz == 0
        || sp == 0
        || model.xa_old.len() != nz * sp
        || (c.mix_factor == 0.0 && c.dxdt_nuc_factor == 0.0)
    {
        return Ok(0.0);
    }
    let catalog = NuclideCatalog::builtin_ref();
    let cz_dist_limit = ts.dx_mix_dist_limit * M_SUN;
    let mut best_ratio = 0.0_f64;

    for i in 0..MAX_DX_LIMIT_CTRLS {
        let sp_name = ts.dx_limit_species[i].trim();
        if sp_name.is_empty()
            || (ts.dx_limit[i] >= 1.0
                && ts.dx_hard_limit[i] >= 1.0
                && ts.dx_div_x_limit[i] >= 1.0
                && ts.dx_div_x_hard_limit[i] >= 1.0)
        {
            continue;
        }
        let dx_lim = ts.dx_limit[i] * ts.time_delta_coeff;
        let dx_hard = ts.dx_hard_limit[i] * ts.time_delta_coeff;
        let dxx_lim = ts.dx_div_x_limit[i] * ts.time_delta_coeff;
        let dxx_hard = ts.dx_div_x_hard_limit[i] * ts.time_delta_coeff;
        let (mut max_dx, mut max_dxx) = (-1.0_f64, -1.0_f64);

        for k in 0..nz {
            if model.d_mix[k] > 0.0 || (k + 1 < nz && model.d_mix[k + 1] > 0.0)
            {
                continue;
            }
            if cz_dist_limit > 0.0
                && mix_bdy_q.iter().any(|&qb| {
                    model.xmstar * (model.q[k] - qb).abs() < cz_dist_limit
                })
            {
                continue;
            }
            for j in 0..sp {
                let Some(nuc) = catalog.get_by_id(model.chem_id[j]) else {
                    continue;
                };
                let matches = match sp_name {
                    "X" => nuc.z == 1,
                    "Y" => nuc.z == 2,
                    "Z" => nuc.z > 2,
                    other => nuc.name.eq_ignore_ascii_case(other),
                };
                if !matches {
                    continue;
                }
                let x = model.xa[k * sp + j];
                let mut delta_x = model.xa_old[k * sp + j] - x;
                if !ts.dx_decreases_only[i] && delta_x < 0.0 {
                    delta_x = -delta_x;
                }
                if x >= ts.dx_limit_min_x[i] {
                    if !skip_hard_limit && dx_hard > 0.0 && delta_x > dx_hard {
                        return Err(format!("dX {} hard limit", nuc.name));
                    }
                    max_dx = max_dx.max(delta_x);
                }
                if x >= ts.dx_div_x_limit_min_x[i] && x > 0.0 {
                    let dxx = delta_x / x;
                    if !skip_hard_limit && dxx_hard > 0.0 && dxx > dxx_hard {
                        return Err(format!(
                            "dX_div_X {} hard limit",
                            nuc.name
                        ));
                    }
                    max_dxx = max_dxx.max(dxx);
                }
            }
        }
        if dx_lim > 0.0 && max_dx > dx_lim {
            best_ratio = best_ratio.max(max_dx / dx_lim);
        }
        if dxx_lim > 0.0 && max_dxx > dxx_lim {
            best_ratio = best_ratio.max(max_dxx / dxx_lim);
        }
    }
    Ok(best_ratio)
}

pub(crate) fn check_dx_nuc_drop_full(
    model: &StarModel,
    skip_hard_limit: bool,
    log_max_t: f64,
) -> Result<f64, String> {
    let c = &model.controls;
    let ts = &c.ts_stop;
    let (nz, sp, dt) = (model.nz, model.species, model.dt);
    if nz == 0
        || sp == 0
        || dt <= 0.0
        || model.xa_old.len() != nz * sp
        || model.dxdt_nuc.len() != nz * sp
        || (c.mix_factor == 0.0 && c.dxdt_nuc_factor == 0.0)
    {
        return Ok(0.0);
    }
    let catalog = NuclideCatalog::builtin_ref();
    let mut max_drop = 0.0_f64;

    for k in 0..nz {
        let dt_dm = dt / model.dm[k].max(1.0e-30);
        let sig00 = model.sig.get(k).copied().unwrap_or(0.0);
        let sigp1 = model.sig.get(k + 1).copied().unwrap_or(0.0);
        for j in 0..sp {
            let Some(nuc) = catalog.get_by_id(model.chem_id[j]) else {
                continue;
            };
            if nuc.z <= 2
                || ((nuc.z + nuc.n) as f64) > ts.dx_nuc_drop_max_a_limit
            {
                continue;
            }
            let x = model.xa[k * sp + j];
            let dxdt = model.dxdt_nuc[k * sp + j];
            if x < ts.dx_nuc_drop_min_x_limit || dxdt >= 0.0 {
                continue;
            }
            let flux00 = if k > 0 {
                -sig00 * (model.xa[(k - 1) * sp + j] - x)
            } else {
                0.0
            };
            let fluxp1 = if k + 1 < nz {
                -sigp1 * (x - model.xa[(k + 1) * sp + j])
            } else {
                0.0
            };
            let dx_inflow = 0.0_f64.max(fluxp1).max(-flux00) * dt_dm;
            let dx_drop =
                (-(dxdt * dt + dx_inflow)).min(model.xa_old[k * sp + j] - x);
            max_drop = max_drop.max(dx_drop);
        }
    }

    let hard_lim = ts.dx_nuc_drop_hard_limit * ts.time_delta_coeff;
    if hard_lim > 0.0 && !skip_hard_limit && max_drop > hard_lim {
        return Err(String::from("dX_nuc_drop_hard_limit"));
    }
    let mut lim = ts.dx_nuc_drop_limit * ts.time_delta_coeff;
    if log_max_t >= 9.45 && ts.dx_nuc_drop_limit_at_high_t > 0.0 {
        lim = ts.dx_nuc_drop_limit_at_high_t;
    }
    let min_dt = SECYER * ts.dx_nuc_drop_min_yrs_for_dt;
    if lim <= 0.0 || max_drop <= 0.0 || dt < min_dt {
        return Ok(0.0);
    }
    let r = (max_drop / lim).min(if min_dt > 0.0 {
        dt / min_dt
    } else {
        f64::INFINITY
    });
    Ok(if r <= 1.0 { 0.0 } else { r })
}

pub(crate) fn check_structural_limits(
    model: &StarModel,
    skip_hard_limit: bool,
) -> Result<f64, String> {
    let ts = &model.controls.ts_stop;
    let coeff = ts.time_delta_coeff;
    let nz = model.nz;
    let mut ratio = 0.0_f64;

    if ts.dl_div_l_limit_min_l > 0.0
        && (ts.dl_div_l_limit > 0.0 || ts.dl_div_l_hard_limit > 0.0)
        && model.lum_start.len() == nz
    {
        let min_l = L_SUN * ts.dl_div_l_limit_min_l;
        let max_dl = (0..nz)
            .filter(|&k| model.lum[k] >= min_l)
            .map(|k| (model.lum[k] - model.lum_start[k]).abs() / model.lum[k])
            .fold(0.0_f64, f64::max);
        ratio = ratio.max(check_change(
            max_dl,
            ts.dl_div_l_limit,
            ts.dl_div_l_hard_limit,
            coeff,
            "dL_div_L",
            skip_hard_limit,
        )?);
    }

    let max_zone_diff = |cur: &[f64], start: &[f64], min_lg: f64| -> f64 {
        if start.len() != nz {
            return 0.0;
        }
        let lim_ln = LN_10 * min_lg;
        (0..nz)
            .filter(|&k| cur[k] >= lim_ln)
            .map(|k| (cur[k] - start[k]).abs())
            .fold(0.0_f64, f64::max)
            / LN_10
    };

    for (val, lim, hard, msg) in [
        (
            max_zone_diff(
                &model.ln_peos,
                &model.ln_peos_start,
                ts.delta_lgp_limit_min_lgp,
            ),
            ts.delta_lgp_limit,
            ts.delta_lgp_hard_limit,
            "delta_lgP",
        ),
        (
            max_zone_diff(
                &model.lnd,
                &model.lnd_start,
                ts.delta_lgrho_limit_min_lgrho,
            ),
            ts.delta_lgrho_limit,
            ts.delta_lgrho_hard_limit,
            "delta_lgRho",
        ),
        (
            max_zone_diff(
                &model.ln_t,
                &model.ln_t_start,
                ts.delta_lgt_limit_min_lgt,
            ),
            ts.delta_lgt_limit,
            ts.delta_lgt_hard_limit,
            "delta_lgT",
        ),
        (
            max_zone_diff(
                &model.ln_r,
                &model.ln_r_start,
                ts.delta_lgr_limit_min_lgr,
            ),
            ts.delta_lgr_limit,
            ts.delta_lgr_hard_limit,
            "delta_lgR",
        ),
    ] {
        ratio = ratio.max(check_change(
            val,
            lim,
            hard,
            coeff,
            msg,
            skip_hard_limit,
        )?);
    }

    if model.energy_start.len() == nz {
        let lim_lne = LN_10 * ts.delta_lge_limit_min_lge;
        let max_dlne = (0..nz)
            .filter(|&k| {
                model.ln_e[k] >= lim_lne && model.energy[k].abs() > 1.0e-30
            })
            .map(|k| {
                (model.energy[k] - model.energy_start[k]).abs()
                    / model.energy[k].abs()
            })
            .fold(0.0_f64, f64::max);
        ratio = ratio.max(check_change(
            max_dlne / LN_10,
            ts.delta_lge_limit,
            ts.delta_lge_hard_limit,
            coeff,
            "delta_lgE",
            skip_hard_limit,
        )?);
    }

    if !model.aux.doing_relax && nz > 0 {
        let c_idx = nz - 1;
        if model.ln_t_start.len() == nz {
            let p_cat = compute_power_categories(model);
            let l_surf_lsun = model.surface_l_lsun();
            let before_near_zams = model.xh_mass[c_idx] > 0.1
                && l_surf_lsun > 0.0
                && (p_cat.power_nuc_burn / l_surf_lsun)
                    < model.controls.lnuc_div_l_zams_limit;
            if !before_near_zams {
                ratio = ratio.max(check_change(
                    (model.ln_t[c_idx] - model.ln_t_start[c_idx]) / LN_10,
                    ts.delta_lgt_cntr_limit,
                    ts.delta_lgt_cntr_hard_limit,
                    coeff,
                    "delta_lgT_cntr",
                    skip_hard_limit,
                )?);
            }
            let lnt_max =
                model.ln_t.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let lnt_max_s = model
                .ln_t_start
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            ratio = ratio.max(check_change(
                (lnt_max - lnt_max_s) / LN_10,
                ts.delta_lgt_max_limit,
                ts.delta_lgt_max_hard_limit,
                coeff,
                "delta_lgT_max",
                skip_hard_limit,
            )?);
        }
        if model.lnd_start.len() == nz {
            ratio = ratio.max(check_change(
                (model.lnd[c_idx] - model.lnd_start[c_idx]) / LN_10,
                ts.delta_lgrho_cntr_limit,
                ts.delta_lgrho_cntr_hard_limit,
                coeff,
                "delta_lgRho_cntr",
                skip_hard_limit,
            )?);
        }
        if model.ln_peos_start.len() == nz {
            ratio = ratio.max(check_change(
                (model.ln_peos[c_idx] - model.ln_peos_start[c_idx]) / LN_10,
                ts.delta_lgp_cntr_limit,
                ts.delta_lgp_cntr_hard_limit,
                coeff,
                "delta_lgP_cntr",
                skip_hard_limit,
            )?);
        }
    }
    Ok(ratio)
}

#[allow(clippy::too_many_arguments)]
fn check_one_lgl_burn(
    new_l: f64,
    old_l: f64,
    lim_in: f64,
    hard_in: f64,
    lgl_min: f64,
    drop_factor: f64,
    coeff: f64,
    dtf: f64,
    msg: &str,
    skip_hard: bool,
) -> Result<f64, String> {
    if new_l <= 0.0 || old_l <= 0.0 {
        return Ok(0.0);
    }
    let (mut lim, mut hard_lim) = (lim_in * coeff, hard_in * coeff);
    if new_l < old_l {
        lim *= drop_factor;
        hard_lim *= drop_factor;
    }
    let lgl = new_l.log10();
    if lgl < lgl_min {
        return Ok(0.0);
    }
    let abs_change = (lgl - old_l.log10()).abs();
    if hard_lim > 0.0 && abs_change > hard_lim && !skip_hard {
        return Err(format!("{msg} hard limit"));
    }
    if lim <= 0.0 {
        return Ok(0.0);
    }
    let r = 1.0 / dtf.clamp(0.1, 0.999).powf((abs_change - lim) / lim);
    Ok(if r <= 1.0 { 0.0 } else { r })
}

pub(crate) fn check_luminosity_and_hr_limits(
    model: &StarModel,
    skip_hard_limit: bool,
    log_max_t: f64,
) -> Result<f64, String> {
    let ts = &model.controls.ts_stop;
    let sh = &model.aux.step_hist;
    let (coeff, dtf) = (ts.time_delta_coeff, ts.timestep_dt_factor);
    let p = compute_power_categories(model);
    let mut ratio = 0.0_f64;

    for (nl, ol, lim, hard, lmin, df, msg) in [
        (
            p.power_h_burn,
            sh.power_h_burn_old,
            ts.delta_lgl_h_limit,
            ts.delta_lgl_h_hard_limit,
            ts.lgl_h_burn_min,
            ts.lgl_h_drop_factor,
            "check_lgL_H_change",
        ),
        (
            p.power_he_burn,
            sh.power_he_burn_old,
            ts.delta_lgl_he_limit,
            ts.delta_lgl_he_hard_limit,
            ts.lgl_he_burn_min,
            ts.lgl_he_drop_factor,
            "check_lgL_He_change",
        ),
        (
            p.power_z_burn,
            sh.power_z_burn_old,
            ts.delta_lgl_z_limit,
            ts.delta_lgl_z_hard_limit,
            ts.lgl_z_burn_min,
            ts.lgl_z_drop_factor,
            "check_lgL_z_change",
        ),
        (
            p.power_nuc_burn,
            sh.power_nuc_burn_old,
            ts.delta_lgl_nuc_limit,
            ts.delta_lgl_nuc_hard_limit,
            ts.lgl_nuc_burn_min,
            ts.lgl_nuc_drop_factor,
            "check_lgL_nuc_change",
        ),
    ] {
        ratio = ratio.max(check_one_lgl_burn(
            nl,
            ol,
            lim,
            hard,
            lmin,
            df,
            coeff,
            dtf,
            msg,
            skip_hard_limit,
        )?);
    }
    if log_max_t >= ts.min_lgt_for_lgl_power_photo_limit {
        ratio = ratio.max(check_one_lgl_burn(
            p.power_photo.abs(),
            sh.power_photo_old.abs(),
            ts.delta_lgl_power_photo_limit,
            ts.delta_lgl_power_photo_hard_limit,
            ts.lgl_power_photo_burn_min,
            ts.lgl_power_photo_drop_factor,
            coeff,
            dtf,
            "check_lgL_power_photo_change",
            skip_hard_limit,
        )?);
    }

    if !model.aux.doing_relax {
        let dlgteff = if model.teff > 0.0 && sh.teff_old > 0.0 {
            Some((model.teff / sh.teff_old).log10())
        } else {
            None
        };
        if let Some(dt_eff) = dlgteff {
            ratio = ratio.max(check_change(
                dt_eff,
                ts.delta_lgteff_limit,
                ts.delta_lgteff_hard_limit,
                coeff,
                "delta_lgTeff",
                skip_hard_limit,
            )?);
        }
        let l_surf = model.surface_l_lsun();
        if l_surf >= ts.delta_lgl_limit_l_min
            && l_surf > 0.0
            && sh.l_surf_old > 0.0
        {
            let dlgl = (l_surf / sh.l_surf_old).log10();
            ratio = ratio.max(check_change(
                dlgl,
                ts.delta_lgl_limit,
                ts.delta_lgl_hard_limit,
                coeff,
                "delta_lgL",
                skip_hard_limit,
            )?);
            if let Some(dt_eff) = dlgteff {
                let dhr = ((ts.delta_hr_ds_l * dlgl).powi(2)
                    + (ts.delta_hr_ds_teff * dt_eff).powi(2))
                .sqrt();
                ratio = ratio.max(check_change(
                    dhr,
                    ts.delta_hr_limit,
                    ts.delta_hr_hard_limit,
                    coeff,
                    "delta_HR",
                    skip_hard_limit,
                )?);
            }
        }
    }
    Ok(ratio)
}
