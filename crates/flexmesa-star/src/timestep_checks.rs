// SPDX-License-Identifier: GPL-3.0-or-later
//! Per-channel soft and hard timestep limit evaluators (`check_change`,
//! `check_integer_limit`, `check_dX`, `check_dX_nuc_drop`, structural
//! `delta_lg*`, nuclear `check_lgL`, `check_lgL_nuc_cat_change`, center
//! abundance `check_lg_X*_cntr` / `check_X*_cntr` / `check_dX_div_X_cntr`,
//! `check_dYe_highT_change`, `check_dlog_eps_nuc_change`, `check_delta_mdot`,
//! `check_adjust_J_q`, HR-diagram, Courant, and energy-error checks in
//! `star/private/timestep.f90:384-2246`).

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{MixingType, LN10 as LN_10, L_SUN, M_SUN, SECYER};
use flexmesa_num::binary_search_1based;

use crate::controls::StarError;
use crate::controls_timestep_stop::MAX_DX_LIMIT_CTRLS;
use crate::core_boundaries::{compute_power_categories, PowerCategories};
use crate::evolve_support::eval_lum_by_category_erg_s;
use crate::mesa_history::{eval_min_cell_collapse_time, eval_min_dr_div_cs};
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

/// `check_change` (`star/private/timestep.f90:732-766`). Like upstream, the
/// `dt_limit_ratio` slot is `intent(inout)` and is assigned only when the
/// soft limit is positive; with `lim <= 0` the slot keeps its earlier value.
#[inline]
pub(crate) fn check_change(
    delta_value: f64,
    lim_in: f64,
    hard_lim_in: f64,
    time_delta_coeff: f64,
    msg: &str,
    skip_hard_limit: bool,
    slot: &mut f64,
) -> Result<(), CheckFailure> {
    // Upstream stops with `mesa_error` on a non-finite change
    // (`timestep.f90:744-747`); there is no retry on this path.
    if !delta_value.is_finite() {
        return Err(CheckFailure::Fatal(StarError::InvalidInput(format!(
            "check_change: {msg} delta_value is not finite"
        ))));
    }
    let abs_change = delta_value.abs();
    let lim = lim_in * time_delta_coeff;
    let hard_lim = hard_lim_in * time_delta_coeff;
    if hard_lim > 0.0 && abs_change > hard_lim && !skip_hard_limit {
        return Err(CheckFailure::Retry(format!("{msg} hard limit")));
    }
    if lim <= 0.0 {
        return Ok(());
    }
    let r = abs_change / lim;
    // `timestep.f90:760-763`.
    if !r.is_finite() {
        return Err(CheckFailure::Fatal(StarError::InvalidInput(format!(
            "check_change: {msg} dt_limit_ratio is not finite"
        ))));
    }
    *slot = if r <= 1.0 { 0.0 } else { r };
    Ok(())
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
                let dq_off =
                    model.aux.mixing.cz_bdy_dq.get(k).copied().unwrap_or(0.0);
                qs.push(model.q[k] - dq_off);
                in_mix = false;
            }
        } else if is_mix {
            let dq_off =
                model.aux.mixing.cz_bdy_dq.get(k).copied().unwrap_or(0.0);
            qs.push(model.q[k] - dq_off);
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

/// Inputs shared by the per-limit checks of one `do_timestep_limits` call
/// (`star/private/timestep.f90:83-127`), computed once.
pub(crate) struct LimitCtx<'a> {
    pub model: &'a StarModel,
    /// `skip_hard_limit` (`timestep.f90:90-97`).
    pub skip: bool,
    /// `s% log_max_temperature`.
    pub log_max_t: f64,
    /// `s% mix_bdy_q`.
    pub mix_bdy_q: Vec<f64>,
    /// Burning power categories from the report routine (`report.f90:76-113`).
    pub power: PowerCategories,
}

impl<'a> LimitCtx<'a> {
    pub(crate) fn new(model: &'a StarModel, skip: bool) -> Self {
        let log_max_t = model
            .ln_t
            .iter()
            .take(model.nz)
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            / LN_10;
        Self {
            model,
            skip,
            log_max_t,
            mix_bdy_q: collect_mixing_boundary_qs(model),
            power: compute_power_categories(model),
        }
    }

    /// `X(nz) > 0.1 .and. power_nuc_burn / L_phot < Lnuc_div_L_zams_limit`
    /// (`timestep.f90:1324-1325, 1380-1381`).
    fn before_near_zams(&self) -> bool {
        let m = self.model;
        let l_phot_lsun = crate::report_utils::eval_photosphere_info(m).l_lsun;
        m.xh_mass[m.nz - 1] > 0.1
            && l_phot_lsun > 0.0
            && (self.power.power_nuc_burn / l_phot_lsun)
                < m.controls.lnuc_div_l_zams_limit
    }
}

type CheckResult = Result<(), CheckFailure>;

/// `check_dL_div_L` (`star/private/timestep.f90:676-729`).
pub(crate) fn check_dl_div_l(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let model = ctx.model;
    let ts = &model.controls.ts_stop;
    let coeff = ts.time_delta_coeff;
    let nz = model.nz;
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
                if dl_hard > 0.0 && !ctx.skip && abs_dl_div_l > dl_hard {
                    return Err(CheckFailure::Retry(String::from(
                        "dL_div_L hard limit",
                    )));
                }
                if abs_dl_div_l > max_dl {
                    max_dl = abs_dl_div_l;
                }
            }
        }
        if dl_lim > 0.0 && max_dl >= 0.0 {
            *slot = max_dl / dl_lim;
        }
    }
    Ok(())
}

/// Shared body of `get_dlg{P,Rho,T,R}_info` + `check_dlg*_change`
/// (`star/private/timestep.f90:769-986`).
#[allow(clippy::too_many_arguments)]
fn check_zone_dlg_change(
    ctx: &LimitCtx<'_>,
    cur: &[f64],
    start: &[f64],
    min_lg: f64,
    lim: f64,
    hard: f64,
    msg: &str,
    slot: &mut f64,
) -> CheckResult {
    let nz = ctx.model.nz;
    if start.len() != nz {
        return Ok(());
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
    if found {
        check_change(
            best / LN_10,
            lim,
            hard,
            ctx.model.controls.ts_stop.time_delta_coeff,
            msg,
            ctx.skip,
            slot,
        )?;
    }
    Ok(())
}

/// `check_dlgP_change` (`star/private/timestep.f90:790-809`).
pub(crate) fn check_dlg_p_change(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let (m, ts) = (ctx.model, &ctx.model.controls.ts_stop);
    check_zone_dlg_change(
        ctx,
        &m.ln_peos,
        &m.ln_peos_start,
        ts.delta_lgp_limit_min_lgp,
        ts.delta_lgp_limit,
        ts.delta_lgp_hard_limit,
        "delta_lgP",
        slot,
    )
}

/// `check_dlgRho_change` (`star/private/timestep.f90:834-854`).
pub(crate) fn check_dlg_rho_change(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let (m, ts) = (ctx.model, &ctx.model.controls.ts_stop);
    check_zone_dlg_change(
        ctx,
        &m.lnd,
        &m.lnd_start,
        ts.delta_lgrho_limit_min_lgrho,
        ts.delta_lgrho_limit,
        ts.delta_lgrho_hard_limit,
        "delta_lgRho",
        slot,
    )
}

/// `check_dlgT_change` (`star/private/timestep.f90:879-899`).
pub(crate) fn check_dlg_t_change(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let (m, ts) = (ctx.model, &ctx.model.controls.ts_stop);
    check_zone_dlg_change(
        ctx,
        &m.ln_t,
        &m.ln_t_start,
        ts.delta_lgt_limit_min_lgt,
        ts.delta_lgt_limit,
        ts.delta_lgt_hard_limit,
        "delta_lgT",
        slot,
    )
}

/// `check_dlgR_change` (`star/private/timestep.f90:967-986`).
pub(crate) fn check_dlg_r_change(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let (m, ts) = (ctx.model, &ctx.model.controls.ts_stop);
    check_zone_dlg_change(
        ctx,
        &m.ln_r,
        &m.ln_r_start,
        ts.delta_lgr_limit_min_lgr,
        ts.delta_lgr_limit,
        ts.delta_lgr_hard_limit,
        "delta_lgR",
        slot,
    )
}

/// `get_dlgE_info` + `check_dlgE_change` (`star/private/timestep.f90:902-943`).
pub(crate) fn check_dlg_e_change(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let model = ctx.model;
    let ts = &model.controls.ts_stop;
    let nz = model.nz;
    if model.energy_start.len() == nz {
        let lim_lne = LN_10 * ts.delta_lge_limit_min_lge;
        let mut found_e = false;
        let mut max_dlne = 0.0_f64;
        for k in 0..nz {
            if model.ln_e[k] >= lim_lne {
                // Signed denominator, as upstream
                // (`abs(energy - energy_start)/energy`, `timestep.f90:918`).
                let d = (model.energy[k] - model.energy_start[k]).abs()
                    / model.energy[k];
                if d > max_dlne {
                    max_dlne = d;
                    found_e = true;
                }
            }
        }
        if found_e {
            check_change(
                max_dlne / LN_10,
                ts.delta_lge_limit,
                ts.delta_lge_hard_limit,
                ts.time_delta_coeff,
                "delta_lgE",
                ctx.skip,
                slot,
            )?;
        }
    }
    Ok(())
}

/// `check_lgL_nuc_cat_change` (`star/private/timestep.f90:1169-1258`).
pub(crate) fn check_lgl_nuc_cat_change(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let model = ctx.model;
    let ts = &model.controls.ts_stop;
    let mix_bdy_q = &ctx.mix_bdy_q;
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
            check_change(
                max_lgl_diff,
                ts.delta_lgl_nuc_cat_limit,
                ts.delta_lgl_nuc_cat_hard_limit,
                ts.time_delta_coeff,
                "delta_lgL_nuc_cat",
                ctx.skip,
                slot,
            )?;
        }
    }
    Ok(())
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
    let lgl = flexmesa_math::safe_log10(new_l);
    if lgl < lgl_min {
        return Ok(0.0);
    }
    if max_other_l > 0.0
        && flexmesa_math::safe_log10(max_other_l) - relative_limit > lgl
    {
        return Ok(0.0);
    }
    let abs_change = (lgl - flexmesa_math::safe_log10(old_l)).abs();
    if hard_lim > 0.0 && abs_change > hard_lim && !skip_hard {
        return Err(format!("{msg} hard limit"));
    }
    if lim <= 0.0 {
        return Ok(0.0);
    }
    let r = 1.0 / flexmesa_math::pow(dtf, (abs_change - lim) / lim);
    Ok(if r <= 1.0 { 0.0 } else { r })
}

/// `check_lgL_H_change` (`star/private/timestep.f90:1118-1125`).
pub(crate) fn check_lgl_h_change(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let ts = &ctx.model.controls.ts_stop;
    let (p, sh) = (&ctx.power, &ctx.model.aux.step_hist);
    *slot = check_one_lgl_burn(
        p.power_h_burn,
        sh.power_h_burn_old,
        p.power_he_burn.max(p.power_z_burn),
        ts.lgl_h_burn_relative_limit,
        ts.delta_lgl_h_limit,
        ts.delta_lgl_h_hard_limit,
        ts.lgl_h_burn_min,
        ts.lgl_h_drop_factor,
        ts.time_delta_coeff,
        ts.timestep_dt_factor,
        "check_lgL_H_change",
        ctx.skip,
    )?;
    Ok(())
}

/// `check_lgL_He_change` (`star/private/timestep.f90:1128-1135`).
pub(crate) fn check_lgl_he_change(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let ts = &ctx.model.controls.ts_stop;
    let (p, sh) = (&ctx.power, &ctx.model.aux.step_hist);
    *slot = check_one_lgl_burn(
        p.power_he_burn,
        sh.power_he_burn_old,
        p.power_h_burn.max(p.power_z_burn),
        ts.lgl_he_burn_relative_limit,
        ts.delta_lgl_he_limit,
        ts.delta_lgl_he_hard_limit,
        ts.lgl_he_burn_min,
        ts.lgl_he_drop_factor,
        ts.time_delta_coeff,
        ts.timestep_dt_factor,
        "check_lgL_He_change",
        ctx.skip,
    )?;
    Ok(())
}

/// `check_lgL_z_change` (`star/private/timestep.f90:1138-1145`).
pub(crate) fn check_lgl_z_change(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let ts = &ctx.model.controls.ts_stop;
    let (p, sh) = (&ctx.power, &ctx.model.aux.step_hist);
    *slot = check_one_lgl_burn(
        p.power_z_burn,
        sh.power_z_burn_old,
        p.power_h_burn.max(p.power_he_burn),
        ts.lgl_z_burn_relative_limit,
        ts.delta_lgl_z_limit,
        ts.delta_lgl_z_hard_limit,
        ts.lgl_z_burn_min,
        ts.lgl_z_drop_factor,
        ts.time_delta_coeff,
        ts.timestep_dt_factor,
        "check_lgL_z_change",
        ctx.skip,
    )?;
    Ok(())
}

/// `check_lgL_power_photo_change` (`star/private/timestep.f90:1148-1155`).
pub(crate) fn check_lgl_power_photo_change(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let ts = &ctx.model.controls.ts_stop;
    let (p, sh) = (&ctx.power, &ctx.model.aux.step_hist);
    if ctx.log_max_t >= ts.min_lgt_for_lgl_power_photo_limit {
        *slot = check_one_lgl_burn(
            p.power_photo.abs(),
            sh.power_photo_old.abs(),
            0.0,
            0.0,
            ts.delta_lgl_power_photo_limit,
            ts.delta_lgl_power_photo_hard_limit,
            ts.lgl_power_photo_burn_min,
            ts.lgl_power_photo_drop_factor,
            ts.time_delta_coeff,
            ts.timestep_dt_factor,
            "check_lgL_power_photo_change",
            ctx.skip,
        )?;
    }
    Ok(())
}

/// `check_lgL_nuc_change` (`star/private/timestep.f90:1158-1166`).
pub(crate) fn check_lgl_nuc_change(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let ts = &ctx.model.controls.ts_stop;
    let (p, sh) = (&ctx.power, &ctx.model.aux.step_hist);
    if ctx.log_max_t <= ts.max_lgt_for_lgl_nuc_limit {
        let (nuc_lim, nuc_hard) =
            if ctx.log_max_t > ts.delta_lgl_nuc_at_high_t_limit_lgt_min {
                (
                    ts.delta_lgl_nuc_at_high_t_limit,
                    ts.delta_lgl_nuc_at_high_t_hard_limit,
                )
            } else {
                (ts.delta_lgl_nuc_limit, ts.delta_lgl_nuc_hard_limit)
            };
        *slot = check_one_lgl_burn(
            p.power_nuc_burn,
            sh.power_nuc_burn_old,
            0.0,
            0.0,
            nuc_lim,
            nuc_hard,
            ts.lgl_nuc_burn_min,
            ts.lgl_nuc_drop_factor,
            ts.time_delta_coeff,
            ts.timestep_dt_factor,
            "check_lgL_nuc_change",
            ctx.skip,
        )?;
    }
    Ok(())
}

/// `check_dlgTeff_change` (`star/private/timestep.f90:1261-1278`).
pub(crate) fn check_dlg_teff_change(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let model = ctx.model;
    let ts = &model.controls.ts_stop;
    let sh = &model.aux.step_hist;
    if !model.aux.doing_relax && model.teff > 0.0 && sh.teff_old > 0.0 {
        let dt_eff = flexmesa_math::safe_log10(model.teff / sh.teff_old);
        check_change(
            dt_eff,
            ts.delta_lgteff_limit,
            ts.delta_lgteff_hard_limit,
            ts.time_delta_coeff,
            "delta_lgTeff",
            ctx.skip,
            slot,
        )?;
    }
    Ok(())
}

/// `check_dlgRho_cntr_change` (`star/private/timestep.f90:1415-1435`).
pub(crate) fn check_dlg_rho_cntr_change(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let (m, ts) = (ctx.model, &ctx.model.controls.ts_stop);
    let nz = m.nz;
    if !m.aux.doing_relax && nz > 0 && m.lnd_start.len() == nz {
        check_change(
            (m.lnd[nz - 1] - m.lnd_start[nz - 1]) / LN_10,
            ts.delta_lgrho_cntr_limit,
            ts.delta_lgrho_cntr_hard_limit,
            ts.time_delta_coeff,
            "delta_lgRho_cntr",
            ctx.skip,
            slot,
        )?;
    }
    Ok(())
}

/// `check_dlgP_cntr_change` (`star/private/timestep.f90:1394-1412`).
pub(crate) fn check_dlg_p_cntr_change(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let (m, ts) = (ctx.model, &ctx.model.controls.ts_stop);
    let nz = m.nz;
    if !m.aux.doing_relax && nz > 0 && m.ln_peos_start.len() == nz {
        check_change(
            (m.ln_peos[nz - 1] - m.ln_peos_start[nz - 1]) / LN_10,
            ts.delta_lgp_cntr_limit,
            ts.delta_lgp_cntr_hard_limit,
            ts.time_delta_coeff,
            "delta_lgP_cntr",
            ctx.skip,
            slot,
        )?;
    }
    Ok(())
}

/// `check_dlgT_cntr_change` (`star/private/timestep.f90:1369-1391`).
pub(crate) fn check_dlg_t_cntr_change(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let (m, ts) = (ctx.model, &ctx.model.controls.ts_stop);
    let nz = m.nz;
    if !m.aux.doing_relax
        && nz > 0
        && m.ln_t_start.len() == nz
        && !(ts.delta_lgt_cntr_limit_only_after_near_zams
            && ctx.before_near_zams())
    {
        check_change(
            (m.ln_t[nz - 1] - m.ln_t_start[nz - 1]) / LN_10,
            ts.delta_lgt_cntr_limit,
            ts.delta_lgt_cntr_hard_limit,
            ts.time_delta_coeff,
            "delta_lgT_cntr",
            ctx.skip,
            slot,
        )?;
    }
    Ok(())
}

fn max_ln_t(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// `check_dlgT_max_change` (`star/private/timestep.f90:1311-1339`).
pub(crate) fn check_dlg_t_max_change(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let (m, ts) = (ctx.model, &ctx.model.controls.ts_stop);
    let nz = m.nz;
    if !m.aux.doing_relax && nz > 0 && m.ln_t_start.len() == nz {
        let lnt_max = max_ln_t(&m.ln_t);
        if ts.delta_lgt_max_limit_lgt_min >= 0.0
            && !(ts.delta_lgt_max_limit_only_after_near_zams
                && ctx.before_near_zams())
            && lnt_max >= ts.delta_lgt_max_limit_lgt_min * LN_10
        {
            check_change(
                (lnt_max - max_ln_t(&m.ln_t_start)) / LN_10,
                ts.delta_lgt_max_limit,
                ts.delta_lgt_max_hard_limit,
                ts.time_delta_coeff,
                "delta_lgT_max",
                ctx.skip,
                slot,
            )?;
        }
    }
    Ok(())
}

/// `check_dlgT_max_at_high_T_change` (`star/private/timestep.f90:1342-1366`).
pub(crate) fn check_dlg_t_max_at_high_t_change(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let (m, ts) = (ctx.model, &ctx.model.controls.ts_stop);
    let nz = m.nz;
    if !m.aux.doing_relax && nz > 0 && m.ln_t_start.len() == nz {
        let lnt_max = max_ln_t(&m.ln_t);
        if ts.delta_lgt_max_at_high_t_limit_lgt_min >= 0.0
            && lnt_max >= ts.delta_lgt_max_at_high_t_limit_lgt_min * LN_10
        {
            check_change(
                (lnt_max - max_ln_t(&m.ln_t_start)) / LN_10,
                ts.delta_lgt_max_at_high_t_limit,
                ts.delta_lgt_max_at_high_t_hard_limit,
                ts.time_delta_coeff,
                "delta_lgT_max_at_high_T",
                ctx.skip,
                slot,
            )?;
        }
    }
    Ok(())
}

/// `check_dYe_highT_change` (`star/private/timestep.f90:1281-1308`).
pub(crate) fn check_dye_hight_change(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let (m, ts) = (ctx.model, &ctx.model.controls.ts_stop);
    let nz = m.nz;
    if !m.aux.doing_relax && nz > 0 && m.ye_start.len() == nz {
        let t_lim = ts.mint_for_hight_ye_limit;
        let mut max_ye_diff = 0.0_f64;
        for k in 0..nz {
            if m.t[k] >= t_lim {
                max_ye_diff = max_ye_diff.max((m.ye[k] - m.ye_start[k]).abs());
            }
        }
        check_change(
            max_ye_diff,
            ts.delta_ye_hight_limit,
            ts.delta_ye_hight_hard_limit,
            ts.time_delta_coeff,
            "delta_Ye_highT",
            false,
            slot,
        )?;
    }
    Ok(())
}

/// `check_dlog_eps_nuc_change` (`star/private/timestep.f90:1438-1481`).
pub(crate) fn check_dlog_eps_nuc_change(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let (model, ts) = (ctx.model, &ctx.model.controls.ts_stop);
    let nz = model.nz;
    if nz > 0
        && model.eps_nuc_start.len() == nz
        && (ts.delta_log_eps_nuc_limit > 0.0
            || ts.delta_log_eps_nuc_hard_limit > 0.0)
    {
        let conv_bdy_locs = collect_conv_boundary_locs(model);
        // `star/private/timestep.f90:1449-1450`.
        let limit_ratio = flexmesa_math::exp10(ts.delta_log_eps_nuc_limit);
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
        check_change(
            delta_eps,
            ts.delta_log_eps_nuc_limit,
            ts.delta_log_eps_nuc_hard_limit,
            ts.time_delta_coeff,
            "delta_log_eps_nuc",
            ctx.skip,
            slot,
        )?;
    }
    Ok(())
}

/// Index of the center zone when the center-abundance checks run
/// (`mix_factor == 0 .and. dxdt_nuc_factor == 0` returns early upstream).
fn cntr_base(model: &StarModel) -> Option<usize> {
    let c = &model.controls;
    let (nz, sp) = (model.nz, model.species);
    if nz == 0
        || sp == 0
        || model.xa_old.len() != nz * sp
        || (c.mix_factor == 0.0 && c.dxdt_nuc_factor == 0.0)
    {
        None
    } else {
        Some((nz - 1) * sp)
    }
}

fn find_iso(model: &StarModel, name: &str) -> Option<usize> {
    model
        .iso_names
        .iter()
        .position(|n| n.eq_ignore_ascii_case(name))
}

/// `check_dX_div_X_cntr` (`star/private/timestep.f90:1484-1516`).
pub(crate) fn check_dx_div_x_cntr(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let (model, ts) = (ctx.model, &ctx.model.controls.ts_stop);
    let Some(base) = cntr_base(model) else {
        return Ok(());
    };
    let mut max_abs_dxx = -1.0_f64;
    for j in 0..model.species {
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
        check_change(
            max_abs_dxx,
            ts.delta_dx_div_x_cntr_limit,
            ts.delta_dx_div_x_cntr_hard_limit,
            ts.time_delta_coeff,
            "delta_dX_div_X_cntr",
            ctx.skip,
            slot,
        )?;
    }
    Ok(())
}

/// Which center abundances `check_lg_X*_cntr` compares against
/// (`star/private/timestep.f90:1519-1725`): the isotope itself, then the
/// lighter fuels whose abundance must not exceed it.
#[derive(Clone, Copy)]
enum LgX {
    H,
    He,
    C,
    Ne,
    O,
    Si,
}

/// Body of `check_lg_X{H,He,C,Ne,O,Si}_cntr`
/// (`star/private/timestep.f90:1519-1725`).
fn check_lg_x_cntr(
    ctx: &LimitCtx<'_>,
    which: LgX,
    slot: &mut f64,
) -> CheckResult {
    let (model, ts) = (ctx.model, &ctx.model.controls.ts_stop);
    let Some(base) = cntr_base(model) else {
        return Ok(());
    };
    let x_of =
        |name: &str| find_iso(model, name).map(|i| (i, model.xa[base + i]));
    let Some((ih1, xh1)) = x_of("h1") else {
        return Ok(());
    };
    // (isotope index, lower bound on X, controls, message)
    let (iso, min_x, lg_min, lg_max, lim, hard, drop_only, msg) = match which {
        LgX::H => (
            ih1,
            1.0e-10,
            ts.delta_lg_xh_cntr_min,
            ts.delta_lg_xh_cntr_max,
            ts.delta_lg_xh_cntr_limit,
            ts.delta_lg_xh_cntr_hard_limit,
            ts.delta_lg_xh_drop_only,
            "delta_lg_XH_cntr",
        ),
        LgX::He => {
            let Some((i, _)) = x_of("he4") else {
                return Ok(());
            };
            (
                i,
                xh1.max(1.0e-10),
                ts.delta_lg_xhe_cntr_min,
                ts.delta_lg_xhe_cntr_max,
                ts.delta_lg_xhe_cntr_limit,
                ts.delta_lg_xhe_cntr_hard_limit,
                ts.delta_lg_xhe_drop_only,
                "delta_lg_XHe_cntr",
            )
        }
        LgX::C | LgX::Ne | LgX::O | LgX::Si => {
            let (Some((_, xhe4)), Some((ic12, xc12))) =
                (x_of("he4"), x_of("c12"))
            else {
                return Ok(());
            };
            let min_h_he = xh1.max(xhe4).max(1.0e-10);
            match which {
                LgX::C => (
                    ic12,
                    min_h_he,
                    ts.delta_lg_xc_cntr_min,
                    ts.delta_lg_xc_cntr_max,
                    ts.delta_lg_xc_cntr_limit,
                    ts.delta_lg_xc_cntr_hard_limit,
                    ts.delta_lg_xc_drop_only,
                    "delta_lg_XC_cntr",
                ),
                LgX::Ne => {
                    let Some((i, _)) = x_of("ne20") else {
                        return Ok(());
                    };
                    (
                        i,
                        min_h_he.max(xc12),
                        ts.delta_lg_xne_cntr_min,
                        ts.delta_lg_xne_cntr_max,
                        ts.delta_lg_xne_cntr_limit,
                        ts.delta_lg_xne_cntr_hard_limit,
                        ts.delta_lg_xne_drop_only,
                        "delta_lg_XNe_cntr",
                    )
                }
                LgX::O => {
                    let Some((i, _)) = x_of("o16") else {
                        return Ok(());
                    };
                    (
                        i,
                        min_h_he.max(xc12),
                        ts.delta_lg_xo_cntr_min,
                        ts.delta_lg_xo_cntr_max,
                        ts.delta_lg_xo_cntr_limit,
                        ts.delta_lg_xo_cntr_hard_limit,
                        ts.delta_lg_xo_drop_only,
                        "delta_lg_XO_cntr",
                    )
                }
                _ => {
                    let Some((i, _)) = x_of("si28") else {
                        return Ok(());
                    };
                    (
                        i,
                        min_h_he.max(xc12),
                        ts.delta_lg_xsi_cntr_min,
                        ts.delta_lg_xsi_cntr_max,
                        ts.delta_lg_xsi_cntr_limit,
                        ts.delta_lg_xsi_cntr_hard_limit,
                        ts.delta_lg_xsi_drop_only,
                        "delta_lg_XSi_cntr",
                    )
                }
            }
        }
    };
    let x = model.xa[base + iso];
    if x < min_x || x < 1.0e-10 {
        return Ok(());
    }
    let lg_x = flexmesa_math::log10(x);
    if lg_x > lg_max || lg_x < lg_min {
        return Ok(());
    }
    let lg_old = flexmesa_math::log10(model.xa_old[base + iso].max(1.0e-99));
    if drop_only && lg_x >= lg_old {
        return Ok(());
    }
    check_change(
        lg_x - lg_old,
        lim,
        hard,
        ts.time_delta_coeff,
        msg,
        ctx.skip,
        slot,
    )?;
    Ok(())
}

macro_rules! lg_x_check {
    ($name:ident, $which:expr, $up:literal) => {
        #[doc = concat!(
            "`check_lg_", $up, "_cntr` (`star/private/timestep.f90:1519-1725`)."
        )]
        pub(crate) fn $name(
            ctx: &LimitCtx<'_>,
            slot: &mut f64,
        ) -> CheckResult {
            check_lg_x_cntr(ctx, $which, slot)
        }
    };
}
lg_x_check!(check_lg_xh_cntr, LgX::H, "XH");
lg_x_check!(check_lg_xhe_cntr, LgX::He, "XHe");
lg_x_check!(check_lg_xc_cntr, LgX::C, "XC");
lg_x_check!(check_lg_xne_cntr, LgX::Ne, "XNe");
lg_x_check!(check_lg_xo_cntr, LgX::O, "XO");
lg_x_check!(check_lg_xsi_cntr, LgX::Si, "XSi");

/// Body of `check_X{H,He,C,Ne,O,Si}_cntr`
/// (`star/private/timestep.f90:1728-1881`).
fn check_x_cntr(
    ctx: &LimitCtx<'_>,
    name: &str,
    lim: f64,
    hard: f64,
    drop_only: bool,
    msg: &str,
    slot: &mut f64,
) -> CheckResult {
    let model = ctx.model;
    let Some(base) = cntr_base(model) else {
        return Ok(());
    };
    if let Some(iso) = find_iso(model, name) {
        let x = model.xa[base + iso];
        let x_old = model.xa_old[base + iso];
        if !(drop_only && x >= x_old) {
            check_change(
                x - x_old,
                lim,
                hard,
                model.controls.ts_stop.time_delta_coeff,
                msg,
                ctx.skip,
                slot,
            )?;
        }
    }
    Ok(())
}

macro_rules! x_check {
    ($name:ident, $iso:literal, $lim:ident, $hard:ident, $drop:ident,
     $msg:literal, $up:literal) => {
        #[doc = concat!(
            "`check_", $up, "_cntr` (`star/private/timestep.f90:1728-1881`)."
        )]
        pub(crate) fn $name(
            ctx: &LimitCtx<'_>,
            slot: &mut f64,
        ) -> CheckResult {
            let ts = &ctx.model.controls.ts_stop;
            check_x_cntr(
                ctx, $iso, ts.$lim, ts.$hard, ts.$drop, $msg, slot,
            )
        }
    };
}
x_check!(
    check_xh_cntr,
    "h1",
    delta_xh_cntr_limit,
    delta_xh_cntr_hard_limit,
    delta_xh_drop_only,
    "delta_XH_cntr",
    "XH"
);
x_check!(
    check_xhe_cntr,
    "he4",
    delta_xhe_cntr_limit,
    delta_xhe_cntr_hard_limit,
    delta_xhe_drop_only,
    "delta_XHe_cntr",
    "XHe"
);
x_check!(
    check_xc_cntr,
    "c12",
    delta_xc_cntr_limit,
    delta_xc_cntr_hard_limit,
    delta_xc_drop_only,
    "delta_XC_cntr",
    "XC"
);
x_check!(
    check_xne_cntr,
    "ne20",
    delta_xne_cntr_limit,
    delta_xne_cntr_hard_limit,
    delta_xne_drop_only,
    "delta_XNe_cntr",
    "XNe"
);
x_check!(
    check_xo_cntr,
    "o16",
    delta_xo_cntr_limit,
    delta_xo_cntr_hard_limit,
    delta_xo_drop_only,
    "delta_XO_cntr",
    "XO"
);
x_check!(
    check_xsi_cntr,
    "si28",
    delta_xsi_cntr_limit,
    delta_xsi_cntr_hard_limit,
    delta_xsi_drop_only,
    "delta_XSi_cntr",
    "XSi"
);

/// `check_delta_mstar` (`star/private/timestep.f90:1915-1938`).
pub(crate) fn check_delta_mstar(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let model = ctx.model;
    let ts = &model.controls.ts_stop;
    if model.mstar > 0.0 && model.mstar_old > 0.0 {
        check_change(
            flexmesa_math::log10(model.mstar / model.mstar_old).abs(),
            ts.delta_lg_star_mass_limit,
            ts.delta_lg_star_mass_hard_limit,
            ts.time_delta_coeff,
            "delta_lg_star_mass_hard_limit",
            ctx.skip,
            slot,
        )?;
    }
    Ok(())
}

/// `check_delta_mdot` (`star/private/timestep.f90:1884-1912`).
pub(crate) fn check_delta_mdot(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let model = ctx.model;
    let c = &model.controls;
    let mw = &c.mix_wind;
    let coeff = c.ts_stop.time_delta_coeff;
    let (mdot, mdot_old) = (model.mstar_dot, model.aux.step_hist.mstar_dot_old);
    // Evaluate `star/private/timestep.f90:1893-1911` without a `denom > 0.0`
    // guard: `0.0 / 0.0` yields `NaN` (`delta_mdot == 0` is false,
    // `delta_mdot > hard_lim` is false, and `r_mdot <= 1.0` is false so `NaN`
    // remains in `*slot`), while `x / 0.0` yields `+inf` (triggering retry if
    // `hard_lim > 0.0 && !ctx.skip`, else writing `+inf` when `lim > 0.0`).
    let denom = (mw.delta_mdot_atol * M_SUN) / SECYER
        + mw.delta_mdot_rtol * mdot.abs().max(mdot_old.abs());
    let delta_mdot = (mdot - mdot_old).abs() / denom;
    if delta_mdot == 0.0 {
        return Ok(());
    }
    let lim = mw.delta_mdot_limit * coeff;
    let hard_lim = mw.delta_mdot_hard_limit * coeff;
    if hard_lim > 0.0 && !ctx.skip && delta_mdot > hard_lim {
        return Err(CheckFailure::Retry(String::from("delta_mdot_hard_limit")));
    }
    if lim <= 0.0 {
        return Ok(());
    }
    let r_mdot = delta_mdot / lim;
    *slot = if r_mdot <= 1.0 { 0.0 } else { r_mdot };
    Ok(())
}

/// `check_adjust_J_q` (`star/private/timestep.f90:1941-1958`).
pub(crate) fn check_adjust_j_q(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let model = ctx.model;
    let (c, ts) = (&model.controls, &model.controls.ts_stop);
    if !model.aux.doing_relax
        && c.rotation_flag
        && c.mix_wind.do_adjust_j_lost
        && model.mstar_dot < 0.0
    {
        check_change(
            1.0 - model.adjust_j_q,
            1.0 - ts.adjust_j_q_limit,
            1.0 - ts.adjust_j_q_hard_limit,
            ts.time_delta_coeff,
            "check_adjust_J_q",
            false,
            slot,
        )?;
    }
    Ok(())
}

/// `check_delta_lgL` (`star/private/timestep.f90:1961-1987`).
pub(crate) fn check_delta_lgl(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let model = ctx.model;
    let ts = &model.controls.ts_stop;
    let sh = &model.aux.step_hist;
    if model.aux.doing_relax {
        return Ok(());
    }
    let l_surf = model.surface_l_lsun();
    *slot = 0.0;
    let dlgl = if l_surf < ts.delta_lgl_limit_l_min || sh.l_surf_old <= 0.0 {
        0.0
    } else {
        flexmesa_math::log10(l_surf / sh.l_surf_old)
    };
    // Upstream stops with `mesa_error` on a non-finite `dlgL`
    // (`timestep.f90:1972-1976`), e.g. `L_surf <= 0` above a negative
    // `delta_lgL_limit_L_min`.
    if !dlgl.is_finite() {
        return Err(CheckFailure::Fatal(StarError::InvalidInput(format!(
            "check_delta_lgL: dlgL is not finite (L_surf = {l_surf}, \
             L_surf_old = {})",
            sh.l_surf_old
        ))));
    }
    check_change(
        dlgl,
        ts.delta_lgl_limit,
        ts.delta_lgl_hard_limit,
        ts.time_delta_coeff,
        "delta_lgL",
        ctx.skip,
        slot,
    )?;
    Ok(())
}

/// `check_dt_div_dt_cell_collapse` (`star/private/timestep.f90:2046-2068`).
pub(crate) fn check_dt_div_dt_cell_collapse(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let model = ctx.model;
    let ts = &model.controls.ts_stop;
    if model.aux.doing_relax {
        return Ok(());
    }
    let dt_coll = eval_min_cell_collapse_time(model);
    if dt_coll >= 1.0e-30 {
        check_change(
            model.dt / dt_coll,
            ts.dt_div_dt_cell_collapse_limit,
            ts.dt_div_dt_cell_collapse_hard_limit,
            ts.time_delta_coeff,
            "dt_div_dt_cell_collapse",
            ctx.skip,
            slot,
        )?;
    }
    Ok(())
}

/// `check_dt_div_min_dr_div_cs` (`star/private/timestep.f90:2071-2097`).
pub(crate) fn check_dt_div_min_dr_div_cs(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let model = ctx.model;
    let ts = &model.controls.ts_stop;
    if model.aux.doing_relax || ts.dt_div_min_dr_div_cs_limit <= 0.0 {
        return Ok(());
    }
    let (dt_x, _) = eval_min_dr_div_cs(model);
    if dt_x > 0.0 {
        check_change(
            model.dt / dt_x,
            ts.dt_div_min_dr_div_cs_limit,
            ts.dt_div_min_dr_div_cs_hard_limit,
            ts.time_delta_coeff,
            "dt_div_min_dr_div_cs",
            ctx.skip,
            slot,
        )?;
    }
    Ok(())
}

/// `check_delta_HR` (`star/private/timestep.f90:1990-2020`), ungated by
/// `doing_relax`, using `L_phot` for the current step; `l_surf_old` stands
/// in for `L_phot_old`.
pub(crate) fn check_delta_hr(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let model = ctx.model;
    let ts = &model.controls.ts_stop;
    let sh = &model.aux.step_hist;
    let l_phot = crate::report_utils::eval_photosphere_info(model).l_lsun;
    if l_phot > 0.0
        && sh.l_surf_old > 0.0
        && model.teff > 0.0
        && sh.teff_old > 0.0
    {
        let dlgl = flexmesa_math::log10(l_phot / sh.l_surf_old);
        let dlgteff = flexmesa_math::log10(model.teff / sh.teff_old);
        let dhr = (flexmesa_math::pow2(ts.delta_hr_ds_l * dlgl)
            + flexmesa_math::pow2(ts.delta_hr_ds_teff * dlgteff))
        .sqrt();
        check_change(
            dhr,
            ts.delta_hr_limit,
            ts.delta_hr_hard_limit,
            ts.time_delta_coeff,
            "delta_HR",
            ctx.skip,
            slot,
        )?;
    }
    Ok(())
}

/// `check_rel_error_in_energy` (`star/private/timestep.f90:2023-2043`).
pub(crate) fn check_rel_error_in_energy(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    let model = ctx.model;
    let ts = &model.controls.ts_stop;
    let sh = &model.aux.step_hist;
    if !model.aux.doing_relax && sh.total_energy_end.abs() > 1.0e-30 {
        check_change(
            (sh.error_in_energy_conservation / sh.total_energy_end).abs(),
            ts.limit_for_rel_error_in_energy_conservation,
            ts.hard_limit_for_rel_error_in_energy_conservation,
            ts.time_delta_coeff,
            "hard_limit_for_rel_error_in_energy_conservation",
            false,
            slot,
        )?;
    }
    Ok(())
}

/// `check_dX_nuc_drop` (`star/private/timestep.f90:2100-2241`).
pub(crate) fn check_dx_nuc_drop(
    ctx: &LimitCtx<'_>,
    slot: &mut f64,
) -> CheckResult {
    *slot = check_dx_nuc_drop_full(ctx.model, ctx.skip, ctx.log_max_t)
        .map_err(CheckFailure::Retry)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controls::StarControls;
    use crate::job_controls::StarJobControls;

    fn delta_mdot_model(
        mdot: f64,
        mdot_old: f64,
        atol: f64,
        rtol: f64,
        lim: f64,
        hard_lim: f64,
    ) -> StarModel {
        let mut controls = StarControls::default();
        controls.ts_stop.time_delta_coeff = 1.0;
        controls.mix_wind.delta_mdot_atol = atol;
        controls.mix_wind.delta_mdot_rtol = rtol;
        controls.mix_wind.delta_mdot_limit = lim;
        controls.mix_wind.delta_mdot_hard_limit = hard_lim;
        let mut model =
            StarModel::new(0, vec![1], StarJobControls::default(), controls);
        model.mstar_dot = mdot;
        model.aux.step_hist.mstar_dot_old = mdot_old;
        model
    }

    #[test]
    fn test_check_change_non_finite_delta_is_fatal_not_retry() {
        let mut slot = 0.0;
        let r = check_change(f64::NAN, 1.0, 2.0, 1.0, "x", false, &mut slot);
        assert!(matches!(r, Err(CheckFailure::Fatal(_))));
    }

    #[test]
    fn test_check_change_overflowed_ratio_is_fatal() {
        let mut slot = 0.0;
        let r =
            check_change(1.0e300, 1.0e-300, 0.0, 1.0, "x", false, &mut slot);
        assert!(matches!(r, Err(CheckFailure::Fatal(_))));
    }

    #[test]
    fn test_check_change_hard_limit_retries_and_soft_ratio_is_zeroed_below_one()
    {
        let mut slot = 7.0;
        let r = check_change(3.0, 1.0, 2.0, 1.0, "x", false, &mut slot);
        assert!(matches!(r, Err(CheckFailure::Retry(_))));
        let r = check_change(0.5, 1.0, 2.0, 1.0, "x", false, &mut slot);
        assert!(r.is_ok());
        assert_eq!(slot, 0.0);
        let r = check_change(1.5, 1.0, 2.0, 1.0, "x", false, &mut slot);
        assert!(r.is_ok());
        assert_eq!(slot, 1.5);
    }

    #[test]
    fn test_check_delta_mdot_zero_over_zero_leaves_nan_in_slot() {
        let model = delta_mdot_model(0.0, 0.0, 0.0, 0.0, 0.1, 0.2);
        let ctx = LimitCtx::new(&model, false);
        let mut slot = 0.0;
        assert!(check_delta_mdot(&ctx, &mut slot).is_ok());
        assert!(slot.is_nan());
    }

    #[test]
    fn test_check_delta_mdot_nonzero_over_zero_retries_or_writes_infinity() {
        let mdot = -1.0e-5 * M_SUN / SECYER;
        let m_retry = delta_mdot_model(mdot, 0.0, 0.0, 0.0, 0.1, 0.2);
        let ctx_retry = LimitCtx::new(&m_retry, false);
        let mut slot = 0.0;
        let err = check_delta_mdot(&ctx_retry, &mut slot).unwrap_err();
        assert!(matches!(
            err,
            CheckFailure::Retry(ref msg) if msg == "delta_mdot_hard_limit"
        ));

        let ctx_skip = LimitCtx::new(&m_retry, true);
        let mut slot_skip = 0.0;
        assert!(check_delta_mdot(&ctx_skip, &mut slot_skip).is_ok());
        assert_eq!(slot_skip, f64::INFINITY);

        let m_nohard = delta_mdot_model(mdot, 0.0, 0.0, 0.0, 0.1, -1.0);
        let ctx_nohard = LimitCtx::new(&m_nohard, false);
        let mut slot_nohard = 0.0;
        assert!(check_delta_mdot(&ctx_nohard, &mut slot_nohard).is_ok());
        assert_eq!(slot_nohard, f64::INFINITY);
    }

    #[test]
    fn test_check_delta_mdot_negative_denom_clamps_ratio_to_zero() {
        let mdot = -2.0e-3 * M_SUN / SECYER;
        let model = delta_mdot_model(mdot, 0.0, -1.0e-3, 0.0, 0.5, 0.2);
        let ctx = LimitCtx::new(&model, false);
        let mut slot = 7.0;
        assert!(check_delta_mdot(&ctx, &mut slot).is_ok());
        assert_eq!(slot, 0.0);
    }

    #[test]
    fn test_check_delta_mdot_equal_mdot_positive_denom_preserves_slot() {
        let mdot = -1.0e-5 * M_SUN / SECYER;
        let model = delta_mdot_model(mdot, mdot, 1.0e-3, 0.5, 0.1, 0.2);
        let ctx = LimitCtx::new(&model, false);
        let mut slot = 7.0;
        assert!(check_delta_mdot(&ctx, &mut slot).is_ok());
        assert_eq!(slot, 7.0);
    }

    #[test]
    fn test_check_delta_mdot_positive_denom_soft_and_hard_limits() {
        let atol = 1.0e-3;
        let rtol = 0.5;
        let mdot_old = -1.0e-3 * M_SUN / SECYER;
        let mdot = -2.0e-3 * M_SUN / SECYER;
        let expected_denom =
            (atol * M_SUN) / SECYER + rtol * mdot.abs().max(mdot_old.abs());
        let expected_delta = (mdot - mdot_old).abs() / expected_denom;

        // Soft ratio <= 1.0 clamps to 0.0.
        let m_low = delta_mdot_model(mdot, mdot_old, atol, rtol, 1.0, -1.0);
        let ctx_low = LimitCtx::new(&m_low, false);
        let mut slot_low = 7.0;
        assert!(check_delta_mdot(&ctx_low, &mut slot_low).is_ok());
        assert_eq!(slot_low, 0.0);

        // Soft ratio > 1.0 reports `delta_mdot / lim`.
        let m_soft = delta_mdot_model(mdot, mdot_old, atol, rtol, 0.25, -1.0);
        let ctx_soft = LimitCtx::new(&m_soft, false);
        let mut slot_soft = 0.0;
        assert!(check_delta_mdot(&ctx_soft, &mut slot_soft).is_ok());
        assert_eq!(slot_soft, expected_delta / 0.25);

        // Hard limit exceeded triggers retry unless skipped.
        let m_hard = delta_mdot_model(mdot, mdot_old, atol, rtol, 0.25, 0.4);
        let ctx_hard = LimitCtx::new(&m_hard, false);
        let mut slot_hard = 0.0;
        let err = check_delta_mdot(&ctx_hard, &mut slot_hard).unwrap_err();
        assert!(matches!(
            err,
            CheckFailure::Retry(ref msg) if msg == "delta_mdot_hard_limit"
        ));

        let ctx_skip = LimitCtx::new(&m_hard, true);
        assert!(check_delta_mdot(&ctx_skip, &mut slot_hard).is_ok());
        assert_eq!(slot_hard, expected_delta / 0.25);
    }
}
