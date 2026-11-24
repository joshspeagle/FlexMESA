// SPDX-License-Identifier: GPL-3.0-or-later
//! Per-channel soft and hard timestep limit evaluators (`check_change`,
//! `check_integer_limit`, `check_dX`, `check_dX_nuc_drop`, structural
//! `delta_lg*`, nuclear `check_lgL`, `check_lgL_nuc_cat_change`, center
//! abundance `check_lg_X*_cntr` / `check_X*_cntr` / `check_dX_div_X_cntr`,
//! `check_dYe_highT_change`, `check_dlog_eps_nuc_change`, `check_delta_mdot`,
//! `check_adjust_J_q`, HR-diagram, Courant, and energy-error checks in
//! `star/private/timestep.f90:384-2246`).

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{L_SUN, LN10 as LN_10, M_SUN, MixingType, SECYER};
use flexmesa_num::binary_search_1based;

use crate::controls::StarError;
use crate::controls_timestep_stop::MAX_DX_LIMIT_CTRLS;
use crate::core_boundaries::compute_power_categories;
use crate::evolve_support::eval_lum_by_category_erg_s;
use crate::model::StarModel;

#[derive(Debug)]
pub(crate) enum CheckFailure {
    Retry(String),
    Fatal(StarError),
}

impl From<String> for CheckFailure {
    fn from(msg: String) -> Self {
        Self::Retry(msg)
    }
}

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

/// Collects mixing-region boundary mass fractions `mix_bdy_q` in increasing `q`
/// order from center outward (`locate_mixing_boundaries` in
/// `star/private/mix_info.f90:799-855`).
pub(crate) fn collect_mixing_boundary_qs(model: &StarModel) -> Vec<f64> {
    let nz = model.nz;
    if nz < 2 {
        return Vec::new();
    }
    let mut qs = Vec::new();
    let mut in_mix = model.mixing_type[nz - 1] != MixingType::NoMixing;
    for k in (1..(nz - 1)).rev() {
        let is_mix = model.mixing_type[k] != MixingType::NoMixing;
        if in_mix {
            if !is_mix {
                qs.push(model.q[k]);
                in_mix = false;
            }
        } else if is_mix {
            qs.push(model.q[k]);
            in_mix = true;
        }
    }
    if in_mix {
        qs.push(1.0);
    }
    qs
}

/// Collects 0-based zone indices of convective-region boundaries from center
/// outward (`locate_convection_boundaries` in `star/private/mix_info.f90:580-624`).
fn collect_conv_boundary_locs(model: &StarModel) -> Vec<usize> {
    let nz = model.nz;
    if nz < 2 {
        return Vec::new();
    }
    let mut locs = Vec::new();
    let mut in_conv = model.mixing_type[nz - 1] == MixingType::ConvectiveMixing;
    for k in (1..(nz - 1)).rev() {
        let is_conv = model.mixing_type[k] == MixingType::ConvectiveMixing;
        if in_conv {
            if !is_conv {
                locs.push(k);
                in_conv = false;
            }
        } else if is_conv {
            locs.push(k + 1);
            in_conv = true;
        }
    }
    if in_conv {
        locs.push(0);
    }
    locs
}

#[allow(clippy::too_many_lines)]
pub(crate) fn check_dx_limits(
    model: &StarModel,
    skip_hard_limit: bool,
    mix_bdy_q: &[f64],
    log_max_t: f64,
) -> Result<(f64, f64), CheckFailure> {
    let c = &model.controls;
    let ts = &c.ts_stop;
    let (nz, sp) = (model.nz, model.species);
    if nz == 0
        || sp == 0
        || model.xa_old.len() != nz * sp
        || (c.mix_factor == 0.0 && c.dxdt_nuc_factor == 0.0)
    {
        return Ok((0.0, 0.0));
    }
    let catalog = NuclideCatalog::builtin_ref();
    let cz_dist_limit = ts.dx_mix_dist_limit * M_SUN;
    let d_mix_cutoff = if c.mix_wind.set_min_d_mix
        && model.ye[nz - 1] >= c.mix_wind.min_center_ye_for_min_d_mix
    {
        c.min_d_mix
    } else {
        0.0
    };
    let (mut dx_ratio, mut dxx_ratio) = (0.0_f64, 0.0_f64);

    for i in 0..MAX_DX_LIMIT_CTRLS {
        if ts.dx_limit[i] >= 1.0
            && ts.dx_hard_limit[i] >= 1.0
            && ts.dx_div_x_limit[i] >= 1.0
            && ts.dx_div_x_hard_limit[i] >= 1.0
        {
            continue;
        }
        let sp_name = ts.dx_limit_species[i].trim();
        if sp_name.is_empty() {
            continue;
        }
        let dx_lim = ts.dx_limit[i] * ts.time_delta_coeff;
        let dx_hard = ts.dx_hard_limit[i] * ts.time_delta_coeff;
        let (dxx_raw, dxx_hard_raw) =
            if log_max_t > ts.dx_div_x_at_high_t_limit_lgt_min[i] {
                (
                    ts.dx_div_x_at_high_t_limit[i],
                    ts.dx_div_x_at_high_t_hard_limit[i],
                )
            } else {
                (ts.dx_div_x_limit[i], ts.dx_div_x_hard_limit[i])
            };
        let dxx_lim = dxx_raw * ts.time_delta_coeff;
        let dxx_hard = dxx_hard_raw * ts.time_delta_coeff;
        let (mut max_dx, mut max_dxx) = (-1.0_f64, -1.0_f64);
        let mut bdy = 0_usize;

        for k in 0..nz {
            if model.d_mix[k] > d_mix_cutoff
                || (k + 1 < nz && model.d_mix[k + 1] > d_mix_cutoff)
            {
                continue;
            }
            bdy = binary_search_1based(mix_bdy_q, model.q[k], bdy);
            if bdy > 0 && bdy < mix_bdy_q.len() {
                let bdy_dist_dm =
                    model.xmstar * (model.q[k] - mix_bdy_q[bdy - 1]).abs();
                if bdy_dist_dm < cz_dist_limit {
                    continue;
                }
            }

            for j in 0..sp {
                let Some(nuc) = catalog.get_by_id(model.chem_id[j]) else {
                    continue;
                };
                // Upstream MESA `26.4.1` bug (`#177`, `star/private/timestep.f90:564-567`):
                // `if (cid /= ih1 .or. cid /= ih2 .or. cid /= ih3) cycle` and
                // `if (cid /= ihe4 .or. cid /= ihe3) cycle` evaluate `.true.` for
                // every nuclide `cid`, so `'X'` and `'Y'` rules always `cycle` and
                // never bind. Reproduced unconditionally for `26.4.1` parity.
                if sp_name == "X" || sp_name == "Y" {
                    continue;
                }
                if sp_name == "Z" {
                    if nuc.z <= 2 {
                        continue;
                    }
                } else if !nuc.name.eq_ignore_ascii_case(sp_name) {
                    continue;
                }

                // Upstream MESA `26.4.1` bug (`#253` under `#169`, `star/private/timestep.f90:578`):
                // `s% dX_decreases_only(j)` subscripts `dX_decreases_only(1:100)`
                // by 1-based species index `j` instead of control slot `i`.
                let decreases_only = if j < MAX_DX_LIMIT_CTRLS {
                    ts.dx_decreases_only[j]
                } else {
                    return Err(CheckFailure::Fatal(StarError::InvalidInput(
                        format!(
                            "check_dX matched species '{}' at 1-based index {} > {} (`dX_decreases_only(j)` out-of-bounds read in `star/private/timestep.f90:578`, tracked under #253 / #169)",
                            nuc.name,
                            j + 1,
                            MAX_DX_LIMIT_CTRLS
                        ),
                    )));
                };

                let x = model.xa[k * sp + j];
                let mut delta_x = model.xa_old[k * sp + j] - x;
                if !decreases_only && delta_x < 0.0 {
                    delta_x = -delta_x;
                }
                // `star/private/timestep.f90:580-581, 603-605`: the hard
                // limits are compared without a positivity guard, so a hard
                // limit `<= 0` retries on any positive change, and
                // `delta_X / X` is formed without an `X > 0` guard.
                if x >= ts.dx_limit_min_x[i] {
                    if !skip_hard_limit && delta_x > dx_hard {
                        return Err(CheckFailure::Retry(format!(
                            "dX {} hard limit",
                            nuc.name
                        )));
                    }
                    if delta_x > max_dx {
                        max_dx = delta_x;
                    }
                }
                if x >= ts.dx_div_x_limit_min_x[i] {
                    let dxx = delta_x / x;
                    if !skip_hard_limit && dxx > dxx_hard {
                        return Err(CheckFailure::Retry(format!(
                            "dX_div_X {} hard limit",
                            nuc.name
                        )));
                    }
                    if dxx > max_dxx {
                        max_dxx = dxx;
                    }
                }
            }
        }
        if dx_lim > 0.0 {
            let r_tmp = max_dx / dx_lim;
            if r_tmp > dx_ratio {
                dx_ratio = if r_tmp <= 1.0 { 0.0 } else { r_tmp };
            }
        }
        if dxx_lim > 0.0 {
            let r_tmp = max_dxx / dxx_lim;
            if r_tmp > dxx_ratio {
                dxx_ratio = if r_tmp <= 1.0 { 0.0 } else { r_tmp };
            }
        }
    }
    Ok((dx_ratio, dxx_ratio))
}

#[allow(clippy::too_many_lines)]
pub(crate) fn check_cntr_abundance_limits(
    model: &StarModel,
    skip_hard_limit: bool,
) -> Result<f64, String> {
    let c = &model.controls;
    let ts = &c.ts_stop;
    let (nz, sp, coeff) = (model.nz, model.species, ts.time_delta_coeff);
    if nz == 0
        || sp == 0
        || model.xa_old.len() != nz * sp
        || (c.mix_factor == 0.0 && c.dxdt_nuc_factor == 0.0)
    {
        return Ok(0.0);
    }
    let base = (nz - 1) * sp;
    let mut ratio = 0.0_f64;

    // `check_dX_div_X_cntr` (`star/private/timestep.f90:1484-1516`)
    let mut max_abs_dxx = -1.0_f64;
    for j in 0..sp {
        let x = model.xa[base + j];
        if x > ts.delta_dx_div_x_cntr_max
            || x < ts.delta_dx_div_x_cntr_min
            || x <= 0.0
        {
            continue;
        }
        let dx = x - model.xa_old[base + j];
        if ts.delta_dx_div_x_drop_only && dx > 0.0 {
            continue;
        }
        let abs_dxx = (dx / x).abs();
        if abs_dxx > max_abs_dxx {
            max_abs_dxx = abs_dxx;
        }
    }
    if max_abs_dxx > 0.0 {
        ratio = ratio.max(check_change(
            max_abs_dxx,
            ts.delta_dx_div_x_cntr_limit,
            ts.delta_dx_div_x_cntr_hard_limit,
            coeff,
            "delta_dX_div_X_cntr",
            skip_hard_limit,
        )?);
    }

    let find_iso = |name: &str| {
        model
            .iso_names
            .iter()
            .position(|n| n.eq_ignore_ascii_case(name))
    };
    let h1 = find_iso("h1");
    let he4 = find_iso("he4");
    let c12 = find_iso("c12");
    let ne20 = find_iso("ne20");
    let o16 = find_iso("o16");
    let si28 = find_iso("si28");

    let eval_lg = |iso: usize,
                   min_x: f64,
                   lg_min: f64,
                   lg_max: f64,
                   lim: f64,
                   hard: f64,
                   drop_only: bool,
                   msg: &str|
     -> Result<f64, String> {
        let x = model.xa[base + iso];
        if x < min_x || x < 1.0e-10 {
            return Ok(0.0);
        }
        let lg_x = flexmesa_math::log10(x);
        if lg_x < lg_min || lg_x > lg_max {
            return Ok(0.0);
        }
        let lg_old =
            flexmesa_math::log10(model.xa_old[base + iso].max(1.0e-99));
        if drop_only && lg_x >= lg_old {
            return Ok(0.0);
        }
        check_change(lg_x - lg_old, lim, hard, coeff, msg, skip_hard_limit)
    };

    // `check_lg_X{H,He,C,Ne,O,Si}_cntr` (`timestep.f90:1519-1725`)
    if let Some(ih1) = h1 {
        let xh1 = model.xa[base + ih1];
        ratio = ratio.max(eval_lg(
            ih1,
            1.0e-10,
            ts.delta_lg_xh_cntr_min,
            ts.delta_lg_xh_cntr_max,
            ts.delta_lg_xh_cntr_limit,
            ts.delta_lg_xh_cntr_hard_limit,
            ts.delta_lg_xh_drop_only,
            "delta_lg_XH_cntr",
        )?);
        if let Some(ihe4) = he4 {
            let xhe4 = model.xa[base + ihe4];
            ratio = ratio.max(eval_lg(
                ihe4,
                xh1.max(1.0e-10),
                ts.delta_lg_xhe_cntr_min,
                ts.delta_lg_xhe_cntr_max,
                ts.delta_lg_xhe_cntr_limit,
                ts.delta_lg_xhe_cntr_hard_limit,
                ts.delta_lg_xhe_drop_only,
                "delta_lg_XHe_cntr",
            )?);
            if let Some(ic12) = c12 {
                let xc12 = model.xa[base + ic12];
                let min_h_he = xh1.max(xhe4).max(1.0e-10);
                ratio = ratio.max(eval_lg(
                    ic12,
                    min_h_he,
                    ts.delta_lg_xc_cntr_min,
                    ts.delta_lg_xc_cntr_max,
                    ts.delta_lg_xc_cntr_limit,
                    ts.delta_lg_xc_cntr_hard_limit,
                    ts.delta_lg_xc_drop_only,
                    "delta_lg_XC_cntr",
                )?);
                let min_h_he_c = min_h_he.max(xc12);
                if let Some(ine20) = ne20 {
                    ratio = ratio.max(eval_lg(
                        ine20,
                        min_h_he_c,
                        ts.delta_lg_xne_cntr_min,
                        ts.delta_lg_xne_cntr_max,
                        ts.delta_lg_xne_cntr_limit,
                        ts.delta_lg_xne_cntr_hard_limit,
                        ts.delta_lg_xne_drop_only,
                        "delta_lg_XNe_cntr",
                    )?);
                }
                if let Some(io16) = o16 {
                    ratio = ratio.max(eval_lg(
                        io16,
                        min_h_he_c,
                        ts.delta_lg_xo_cntr_min,
                        ts.delta_lg_xo_cntr_max,
                        ts.delta_lg_xo_cntr_limit,
                        ts.delta_lg_xo_cntr_hard_limit,
                        ts.delta_lg_xo_drop_only,
                        "delta_lg_XO_cntr",
                    )?);
                }
                if let Some(isi28) = si28 {
                    ratio = ratio.max(eval_lg(
                        isi28,
                        min_h_he_c,
                        ts.delta_lg_xsi_cntr_min,
                        ts.delta_lg_xsi_cntr_max,
                        ts.delta_lg_xsi_cntr_limit,
                        ts.delta_lg_xsi_cntr_hard_limit,
                        ts.delta_lg_xsi_drop_only,
                        "delta_lg_XSi_cntr",
                    )?);
                }
            }
        }
    }

    // `check_X{H,He,C,Ne,O,Si}_cntr` (`timestep.f90:1728-1881`)
    for (iso_opt, lim, hard, drop_only, msg) in [
        (
            h1,
            ts.delta_xh_cntr_limit,
            ts.delta_xh_cntr_hard_limit,
            ts.delta_xh_drop_only,
            "delta_XH_cntr",
        ),
        (
            he4,
            ts.delta_xhe_cntr_limit,
            ts.delta_xhe_cntr_hard_limit,
            ts.delta_xhe_drop_only,
            "delta_XHe_cntr",
        ),
        (
            c12,
            ts.delta_xc_cntr_limit,
            ts.delta_xc_cntr_hard_limit,
            ts.delta_xc_drop_only,
            "delta_XC_cntr",
        ),
        (
            ne20,
            ts.delta_xne_cntr_limit,
            ts.delta_xne_cntr_hard_limit,
            ts.delta_xne_drop_only,
            "delta_XNe_cntr",
        ),
        (
            o16,
            ts.delta_xo_cntr_limit,
            ts.delta_xo_cntr_hard_limit,
            ts.delta_xo_drop_only,
            "delta_XO_cntr",
        ),
        (
            si28,
            ts.delta_xsi_cntr_limit,
            ts.delta_xsi_cntr_hard_limit,
            ts.delta_xsi_drop_only,
            "delta_XSi_cntr",
        ),
    ] {
        if let Some(iso) = iso_opt {
            let x = model.xa[base + iso];
            let x_old = model.xa_old[base + iso];
            if !(drop_only && x >= x_old) {
                ratio = ratio.max(check_change(
                    x - x_old,
                    lim,
                    hard,
                    coeff,
                    msg,
                    skip_hard_limit,
                )?);
            }
        }
    }

    Ok(ratio)
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
        let sigp1 = if k + 1 < nz {
            model.sig.get(k + 1).copied().unwrap_or(0.0)
        } else {
            0.0
        };
        for j in 0..sp {
            let Some(nuc) = catalog.get_by_id(model.chem_id[j]) else {
                continue;
            };
            if nuc.w > ts.dx_nuc_drop_max_a_limit || nuc.z <= 2 {
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

#[allow(clippy::too_many_lines)]
pub(crate) fn check_structural_limits(
    model: &StarModel,
    skip_hard_limit: bool,
) -> Result<f64, String> {
    let ts = &model.controls.ts_stop;
    let coeff = ts.time_delta_coeff;
    let nz = model.nz;
    let mut ratio = 0.0_f64;

    // `check_dL_div_L` (`star/private/timestep.f90:676-729`)
    let dl_min_l = L_SUN * ts.dl_div_l_limit_min_l;
    let dl_lim = ts.dl_div_l_limit * coeff;
    let dl_hard = ts.dl_div_l_hard_limit * coeff;
    if dl_min_l > 0.0
        && (dl_lim > 0.0 || dl_hard > 0.0)
        && model.lum_start.len() == nz
    {
        let mut max_dl = -1.0_f64;
        for k in 0..nz {
            let l = model.lum[k];
            if l >= dl_min_l {
                let abs_dl_div_l = (l - model.lum_start[k]).abs() / l;
                if dl_hard > 0.0 && !skip_hard_limit && abs_dl_div_l > dl_hard {
                    return Err(String::from("dL_div_L hard limit"));
                }
                if abs_dl_div_l > max_dl {
                    max_dl = abs_dl_div_l;
                }
            }
        }
        if dl_lim > 0.0 && max_dl >= 0.0 {
            ratio = ratio.max(max_dl / dl_lim);
        }
    }

    let max_zone_diff =
        |cur: &[f64], start: &[f64], min_lg: f64| -> Option<f64> {
            if start.len() != nz {
                return None;
            }
            let lim_ln = LN_10 * min_lg;
            let mut found = false;
            let mut best = 0.0_f64;
            for k in 0..nz {
                if cur[k] < lim_ln {
                    continue;
                }
                let d = (cur[k] - start[k]).abs();
                if d > best {
                    best = d;
                    found = true;
                }
            }
            found.then_some(best / LN_10)
        };

    for (val_opt, lim, hard, msg) in [
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
        if let Some(val) = val_opt {
            ratio = ratio.max(check_change(
                val,
                lim,
                hard,
                coeff,
                msg,
                skip_hard_limit,
            )?);
        }
    }

    if model.energy_start.len() == nz {
        let lim_lne = LN_10 * ts.delta_lge_limit_min_lge;
        let mut found_e = false;
        let mut max_dlne = 0.0_f64;
        for k in 0..nz {
            if model.ln_e[k] >= lim_lne && model.energy[k].abs() > 1.0e-30 {
                let d = (model.energy[k] - model.energy_start[k]).abs()
                    / model.energy[k].abs();
                if d > max_dlne {
                    max_dlne = d;
                    found_e = true;
                }
            }
        }
        if found_e {
            ratio = ratio.max(check_change(
                max_dlne / LN_10,
                ts.delta_lge_limit,
                ts.delta_lge_hard_limit,
                coeff,
                "delta_lgE",
                skip_hard_limit,
            )?);
        }
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

            if !(ts.delta_lgt_cntr_limit_only_after_near_zams
                && before_near_zams)
            {
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
            if ts.delta_lgt_max_limit_lgt_min >= 0.0
                && !(ts.delta_lgt_max_limit_only_after_near_zams
                    && before_near_zams)
                && lnt_max >= ts.delta_lgt_max_limit_lgt_min * LN_10
            {
                ratio = ratio.max(check_change(
                    (lnt_max - lnt_max_s) / LN_10,
                    ts.delta_lgt_max_limit,
                    ts.delta_lgt_max_hard_limit,
                    coeff,
                    "delta_lgT_max",
                    skip_hard_limit,
                )?);
            }
            if ts.delta_lgt_max_at_high_t_limit_lgt_min >= 0.0
                && lnt_max >= ts.delta_lgt_max_at_high_t_limit_lgt_min * LN_10
            {
                ratio = ratio.max(check_change(
                    (lnt_max - lnt_max_s) / LN_10,
                    ts.delta_lgt_max_at_high_t_limit,
                    ts.delta_lgt_max_at_high_t_hard_limit,
                    coeff,
                    "delta_lgT_max_at_high_T",
                    skip_hard_limit,
                )?);
            }
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

        // `check_dYe_highT_change` (`star/private/timestep.f90:1281-1308`)
        if model.ye_start.len() == nz {
            let t_lim = ts.mint_for_hight_ye_limit;
            let mut max_ye_diff = 0.0_f64;
            for k in 0..nz {
                if model.t[k] >= t_lim {
                    max_ye_diff = max_ye_diff
                        .max((model.ye[k] - model.ye_start[k]).abs());
                }
            }
            ratio = ratio.max(check_change(
                max_ye_diff,
                ts.delta_ye_hight_limit,
                ts.delta_ye_hight_hard_limit,
                coeff,
                "delta_Ye_highT",
                false,
            )?);
        }
    }

    // `check_dlog_eps_nuc_change` (`star/private/timestep.f90:1438-1481`)
    if nz > 0
        && model.eps_nuc_start.len() == nz
        && (ts.delta_log_eps_nuc_limit > 0.0
            || ts.delta_log_eps_nuc_hard_limit > 0.0)
    {
        let conv_bdy_locs = collect_conv_boundary_locs(model);
        let limit_ratio = if ts.delta_log_eps_nuc_limit > 0.0 {
            flexmesa_math::exp10(ts.delta_log_eps_nuc_limit)
        } else {
            1.0
        };
        let mut max_eps_ratio = limit_ratio;
        let mut found_eps = false;
        'zoneloop: for k in 0..nz {
            if model.eps_nuc_start[k] < 1.0 {
                continue;
            }
            let r_eps = model.eps_nuc[k] / model.eps_nuc_start[k];
            let k_next = (k + 1).min(nz - 1);
            if model.mixing_type[k] != MixingType::ConvectiveMixing
                && model.mixing_type[k_next] != MixingType::ConvectiveMixing
                && r_eps > max_eps_ratio
            {
                for &b_loc in &conv_bdy_locs {
                    if (model.r[b_loc] - model.r[k]).abs()
                        <= model.scale_height[k]
                    {
                        continue 'zoneloop;
                    }
                }
                max_eps_ratio = r_eps;
                found_eps = true;
            }
        }
        let delta_eps = if found_eps {
            flexmesa_math::log10(max_eps_ratio)
        } else {
            0.0
        };
        ratio = ratio.max(check_change(
            delta_eps,
            ts.delta_log_eps_nuc_limit,
            ts.delta_log_eps_nuc_hard_limit,
            coeff,
            "delta_log_eps_nuc",
            skip_hard_limit,
        )?);
    }

    Ok(ratio)
}

#[allow(clippy::too_many_arguments)]
fn check_one_lgl_burn(
    new_l: f64,
    old_l: f64,
    max_other_l: f64,
    relative_limit: f64,
    lim_in: f64,
    hard_in: f64,
    lgl_min: f64,
    drop_factor: f64,
    coeff: f64,
    dtf: f64,
    msg: &str,
    skip_hard: bool,
) -> Result<f64, String> {
    if old_l < 0.0 || new_l <= 0.0 || old_l <= 0.0 {
        return Ok(0.0);
    }
    let (mut lim, mut hard_lim) = (lim_in * coeff, hard_in * coeff);
    if new_l < old_l {
        lim *= drop_factor;
        hard_lim *= drop_factor;
    }
    let lgl = flexmesa_math::log10(new_l);
    if lgl < lgl_min {
        return Ok(0.0);
    }
    if max_other_l > 0.0
        && flexmesa_math::log10(max_other_l) - relative_limit > lgl
    {
        return Ok(0.0);
    }
    let abs_change = (lgl - flexmesa_math::log10(old_l)).abs();
    if hard_lim > 0.0 && abs_change > hard_lim && !skip_hard {
        return Err(format!("{msg} hard limit"));
    }
    if lim <= 0.0 {
        return Ok(0.0);
    }
    let r = 1.0 / flexmesa_math::pow(dtf, (abs_change - lim) / lim);
    Ok(if r <= 1.0 { 0.0 } else { r })
}

#[allow(clippy::too_many_lines)]
pub(crate) fn check_luminosity_and_hr_limits(
    model: &StarModel,
    skip_hard_limit: bool,
    log_max_t: f64,
    mix_bdy_q: &[f64],
) -> Result<f64, String> {
    let ts = &model.controls.ts_stop;
    let sh = &model.aux.step_hist;
    let (coeff, dtf) = (ts.time_delta_coeff, ts.timestep_dt_factor);
    let p = compute_power_categories(model);
    let mut ratio = 0.0_f64;

    // `check_lgL_nuc_cat_change` (`star/private/timestep.f90:1169-1258`)
    if (ts.delta_lgl_nuc_cat_limit > 0.0
        || ts.delta_lgl_nuc_cat_hard_limit > 0.0)
        && model.lum_by_category_start.len() == model.nz
    {
        let cat_burn_min = flexmesa_math::exp10(ts.lgl_nuc_cat_burn_min);
        let lum_cat = eval_lum_by_category_erg_s(model);
        let mut max_diff = 0.0_f64;
        let mut bdy = 0_usize;
        for k in 0..model.nz {
            bdy = binary_search_1based(mix_bdy_q, model.q[k], bdy);
            if bdy > 0
                && (model.q[k] - mix_bdy_q[bdy - 1]).abs()
                    < ts.lgl_nuc_mix_dist_limit
            {
                continue;
            }
            for cat_j in 0..19 {
                let l_now = lum_cat[k][cat_j];
                let l_beg = model.lum_by_category_start[k][cat_j];
                if l_now >= cat_burn_min && l_beg >= cat_burn_min {
                    let abs_diff = (l_now - l_beg).abs();
                    if abs_diff > max_diff {
                        max_diff = abs_diff;
                    }
                }
            }
        }
        if max_diff > 0.0 {
            let max_lgl_diff = flexmesa_math::log10(max_diff / L_SUN);
            ratio = ratio.max(check_change(
                max_lgl_diff,
                ts.delta_lgl_nuc_cat_limit,
                ts.delta_lgl_nuc_cat_hard_limit,
                coeff,
                "delta_lgL_nuc_cat",
                skip_hard_limit,
            )?);
        }
    }

    for (nl, ol, max_other, rel_lim, lim, hard, lmin, df, msg) in [
        (
            p.power_h_burn,
            sh.power_h_burn_old,
            p.power_he_burn.max(p.power_z_burn),
            ts.lgl_h_burn_relative_limit,
            ts.delta_lgl_h_limit,
            ts.delta_lgl_h_hard_limit,
            ts.lgl_h_burn_min,
            ts.lgl_h_drop_factor,
            "check_lgL_H_change",
        ),
        (
            p.power_he_burn,
            sh.power_he_burn_old,
            p.power_h_burn.max(p.power_z_burn),
            ts.lgl_he_burn_relative_limit,
            ts.delta_lgl_he_limit,
            ts.delta_lgl_he_hard_limit,
            ts.lgl_he_burn_min,
            ts.lgl_he_drop_factor,
            "check_lgL_He_change",
        ),
        (
            p.power_z_burn,
            sh.power_z_burn_old,
            p.power_h_burn.max(p.power_he_burn),
            ts.lgl_z_burn_relative_limit,
            ts.delta_lgl_z_limit,
            ts.delta_lgl_z_hard_limit,
            ts.lgl_z_burn_min,
            ts.lgl_z_drop_factor,
            "check_lgL_z_change",
        ),
    ] {
        ratio = ratio.max(check_one_lgl_burn(
            nl,
            ol,
            max_other,
            rel_lim,
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
            0.0,
            0.0,
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

    if log_max_t <= ts.max_lgt_for_lgl_nuc_limit {
        let (nuc_lim, nuc_hard) =
            if log_max_t > ts.delta_lgl_nuc_at_high_t_limit_lgt_min {
                (
                    ts.delta_lgl_nuc_at_high_t_limit,
                    ts.delta_lgl_nuc_at_high_t_hard_limit,
                )
            } else {
                (ts.delta_lgl_nuc_limit, ts.delta_lgl_nuc_hard_limit)
            };
        ratio = ratio.max(check_one_lgl_burn(
            p.power_nuc_burn,
            sh.power_nuc_burn_old,
            0.0,
            0.0,
            nuc_lim,
            nuc_hard,
            ts.lgl_nuc_burn_min,
            ts.lgl_nuc_drop_factor,
            coeff,
            dtf,
            "check_lgL_nuc_change",
            skip_hard_limit,
        )?);
    }

    let l_surf = model.surface_l_lsun();
    if !model.aux.doing_relax {
        if model.teff > 0.0 && sh.teff_old > 0.0 {
            let dt_eff = flexmesa_math::log10(model.teff / sh.teff_old);
            ratio = ratio.max(check_change(
                dt_eff,
                ts.delta_lgteff_limit,
                ts.delta_lgteff_hard_limit,
                coeff,
                "delta_lgTeff",
                skip_hard_limit,
            )?);
        }
        let dlgl = if l_surf < ts.delta_lgl_limit_l_min
            || l_surf <= 0.0
            || sh.l_surf_old <= 0.0
        {
            0.0
        } else {
            flexmesa_math::log10(l_surf / sh.l_surf_old)
        };
        ratio = ratio.max(check_change(
            dlgl,
            ts.delta_lgl_limit,
            ts.delta_lgl_hard_limit,
            coeff,
            "delta_lgL",
            skip_hard_limit,
        )?);
    }

    // `check_delta_HR` (`star/private/timestep.f90:1990-2020`, ungated by `doing_relax` or `delta_lgL_limit_L_min`)
    if l_surf > 0.0
        && sh.l_surf_old > 0.0
        && model.teff > 0.0
        && sh.teff_old > 0.0
    {
        let dlgl = flexmesa_math::log10(l_surf / sh.l_surf_old);
        let dlgteff = flexmesa_math::log10(model.teff / sh.teff_old);
        let dhr = (flexmesa_math::pow2(ts.delta_hr_ds_l * dlgl)
            + flexmesa_math::pow2(ts.delta_hr_ds_teff * dlgteff))
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

    Ok(ratio)
}
