// SPDX-License-Identifier: LGPL-3.0-or-later
//! Upstream `do_check_limits` (`star/private/do_one_utils.f90:565-1465`),
//! `set_phase_of_evolution` (`star/private/star_utils.f90:2910-2969`),
//! ejecta/remnant/shock diagnostics (`star/private/star_utils.f90:1328-1485,
//! 1918-1966`), and `check_termination_limits` compatibility wrappers.

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{
    AMU, CLIGHT, KERG, LN10, M_SUN, R_SUN, SECDAY, SECYER,
};
use flexmesa_turb::MixingType;

use crate::controls::StarError;
use crate::core_boundaries::{compute_core_masses, CoreMasses};
use crate::history::HistoryRow;
use crate::model::StarModel;
use crate::report_utils::{
    center_avg_x_slot, center_value, eval_astero_report_scalars,
    eval_upstream_omega_crit, safe_log10, surface_avg_x_slot,
    StepReportContext,
};

/// Evolutionary phase integer codes (`star_data/public/star_data_def.inc:26-40`).
pub const PHASE_RELAX: i32 = -1;
/// Initial/unclassified starting phase (`phase_starting = 1`).
pub const PHASE_STARTING: i32 = 1;
/// Pre-main-sequence contraction (`phase_PreMS = 2`).
pub const PHASE_PREMS: i32 = 2;
/// Zero-age main sequence (`phase_ZAMS = 3`).
pub const PHASE_ZAMS: i32 = 3;
/// Intermediate-age main sequence (`phase_IAMS = 4`).
pub const PHASE_IAMS: i32 = 4;
/// Terminal-age main sequence (`phase_TAMS = 5`).
pub const PHASE_TAMS: i32 = 5;
/// Helium burning (`phase_He_Burn = 6`).
pub const PHASE_HE_BURN: i32 = 6;
/// Zero-age core helium burning (`phase_ZACHeB = 7`).
pub const PHASE_ZACHEB: i32 = 7;
/// Terminal-age core helium burning (`phase_TACHeB = 8`).
pub const PHASE_TACHEB: i32 = 8;
/// Thermally pulsing AGB (`phase_TP_AGB = 9`).
pub const PHASE_TP_AGB: i32 = 9;
/// Carbon burning (`phase_C_Burn = 10`).
pub const PHASE_C_BURN: i32 = 10;
/// Neon burning (`phase_Ne_Burn = 11`).
pub const PHASE_NE_BURN: i32 = 11;
/// Oxygen burning (`phase_O_Burn = 12`).
pub const PHASE_O_BURN: i32 = 12;
/// Silicon burning (`phase_Si_Burn = 13`).
pub const PHASE_SI_BURN: i32 = 13;
/// White-dwarf cooling sequence (`phase_WDCS = 14`).
pub const PHASE_WDCS: i32 = 14;

/// 0-based indices into `eps_nuc_categories` / `L_by_category` matching
/// `chem/public/chem_def.f90:414-435` and `flexmesa_chem::CATEGORY_NAME`.
const CAT_3ALF: usize = 2;
const CAT_BURN_NE: usize = 6;
const CAT_BURN_SI: usize = 9;
const CAT_CC: usize = 16;
const CAT_CO: usize = 17;
const CAT_OO: usize = 18;

/// Outcome returned by [`do_check_limits`] (`star_data_def.inc:17-23`).
#[derive(Debug, Clone, PartialEq)]
pub enum LimitCheckOutcome {
    /// Continue evolution normally (`keep_going`).
    KeepGoing,
    /// Repeat the current step from start-of-step state with reduced `dt_new`
    /// to land closer to a stopping target (`redo`, `when_to_stop_rtol`/`atol`).
    Redo {
        /// Halved timestep in seconds (`0.5 * s% dt`).
        dt_new: f64,
        /// Human-readable description of the target that triggered `redo`.
        reason: String,
    },
    /// Repeat the current step through the retry cascade (`retry`, e.g.
    /// `retry_for_v_above_clight`).
    Retry {
        /// Human-readable description of the retry trigger.
        reason: String,
    },
    /// Terminate evolution cleanly with upstream MESA `termination_code_str`.
    Terminate {
        /// Canonical MESA `termination_code_str` (`star_private_def.f90:85-210`).
        code: String,
        /// Human-readable description of the stopping condition.
        reason: String,
    },
}

/// Computes `(he_core_mass, co_core_mass, one_core_mass)` in solar masses
/// ($M_\odot$) via [`compute_core_masses`] (`report.f90:1029-1259`).
#[must_use]
pub fn compute_core_masses_msun(model: &StarModel) -> (f64, f64, f64) {
    let cores = compute_core_masses(model);
    (cores.he_core_mass, cores.co_core_mass, cores.one_core_mass)
}

#[inline]
fn find0(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let a = x1 * y2 - x2 * y1;
    let b = y2 - y1;
    if b == 0.0 {
        0.5 * (x1 + x2)
    } else {
        let x = a / b;
        if (x1 <= x2 && (x < x1 || x > x2)) || (x2 < x1 && (x < x2 || x > x1)) {
            0.5 * (x1 + x2)
        } else {
            x
        }
    }
}

#[inline]
fn csound_face_at(model: &StarModel, k: usize) -> f64 {
    if k == 0 {
        model.csound[0]
    } else {
        let denom = model.dm[k - 1] + model.dm[k];
        if denom > 0.0 {
            (model.dm[k - 1] * model.csound[k]
                + model.dm[k] * model.csound[k - 1])
                / denom
        } else {
            0.5 * (model.csound[k - 1] + model.csound[k])
        }
    }
}

/// Evaluates unbound ejecta mass in grams matching `get_ejecta_mass` in
/// `star/private/star_utils.f90:1924-1966`.
#[must_use]
pub fn get_ejecta_mass(model: &StarModel) -> f64 {
    let nz = model.nz;
    if nz == 0 || !(model.controls.u_flag || model.controls.v_flag) {
        return 0.0;
    }
    let mut v_div_vesc_prev = 0.0_f64;
    for k in 0..nz {
        let v = if model.controls.u_flag {
            if k == 0 {
                model.u[0]
            } else {
                0.5 * (model.u[k - 1] + model.u[k])
            }
        } else {
            model.v[k]
        };
        let vesc = if model.r[k] > 0.0 {
            (2.0 * model.cgrav[k] * model.m[k] / model.r[k])
                .max(0.0)
                .sqrt()
        } else {
            0.0
        };
        let v_div_vesc = if vesc > 0.0 { v / vesc } else { 0.0 };
        if v_div_vesc < 1.0 {
            if k == 0 {
                return 0.0;
            }
            let dm = find0(
                0.0,
                v_div_vesc_prev - 1.0,
                model.dm[k - 1],
                v_div_vesc - 1.0,
            )
            .max(0.0);
            return if k == 1 {
                dm
            } else {
                model.dm[..(k - 1)].iter().sum::<f64>() + dm
            };
        }
        v_div_vesc_prev = v_div_vesc;
    }
    0.0
}

/// Evaluates bound remnant mass in grams (`m(1) - get_ejecta_mass(s)`) matching
/// `get_remnant_mass` in `star/private/star_utils.f90:1918-1921`.
#[must_use]
pub fn get_remnant_mass(model: &StarModel) -> f64 {
    if model.nz == 0 {
        return 0.0;
    }
    model.m[0] - get_ejecta_mass(model)
}

/// Evaluates shock mass coordinate in $M_\odot$ matching `get_shock_info` and
/// `get_shock_location_info` in `star/private/star_utils.f90:1328-1485`.
#[must_use]
pub fn get_shock_mass_msun(model: &StarModel) -> f64 {
    let nz = model.nz;
    if nz < 3 || !(model.controls.u_flag || model.controls.v_flag) {
        return 0.0;
    }
    let v = if model.controls.u_flag {
        &model.u
    } else {
        &model.v
    };
    if v.len() < nz || model.csound.len() < nz {
        return 0.0;
    }
    let mut shock_radius = -1.0_f64;
    let mut k_shock = 0_usize;
    let mut v_div_cs_00 = if model.csound[0] == 0.0 {
        0.0
    } else {
        v[0] / model.csound[0]
    };
    for k in 1..(nz - 1) {
        let v_div_cs_m1 = v_div_cs_00;
        v_div_cs_00 = if model.csound[k] == 0.0 {
            0.0
        } else {
            v[k] / model.csound[k]
        };
        let v_div_cs_max = v_div_cs_00.max(v_div_cs_m1);
        let v_div_cs_min = v_div_cs_00.min(v_div_cs_m1);
        if v_div_cs_max >= 1.0
            && v_div_cs_min < 1.0
            && v[k + 1] > model.csound[k + 1]
        {
            shock_radius = find0(
                model.r[k],
                v_div_cs_00 - 1.0,
                model.r[k - 1],
                v_div_cs_m1 - 1.0,
            );
            k_shock = k - 1;
            break;
        }
        if v_div_cs_min <= -1.0
            && v_div_cs_max > -1.0
            && v[k + 1] < -model.csound[k + 1]
        {
            shock_radius = find0(
                model.r[k],
                v_div_cs_00 + 1.0,
                model.r[k - 1],
                v_div_cs_m1 + 1.0,
            );
            k_shock = k - 1;
            break;
        }
    }
    if shock_radius <= 0.0
        || shock_radius < model.r_center
        || shock_radius > model.r[0]
    {
        return 0.0;
    }
    if k_shock + 1 < nz {
        let denom = model.r[k_shock + 1] - model.r[k_shock];
        if denom == 0.0 {
            return model.m[k_shock] / M_SUN;
        }
        let alfa = (shock_radius - model.r[k_shock]) / denom;
        let beta = 1.0 - alfa;
        (alfa * model.m[k_shock + 1] + beta * model.m[k_shock]) / M_SUN
    } else {
        model.m[k_shock] / M_SUN
    }
}

/// Evaluates `s% min_gamma1` matching `set_min_gamma1` in
/// `star/private/report.f90:35-51`.
#[must_use]
pub fn eval_min_gamma1(model: &StarModel) -> f64 {
    let nz = model.nz;
    let o = &model.controls.output;
    let mut min_g1 = 1.0e99_f64;
    for k in (0..nz).rev() {
        if model.q[k] > o.gamma1_limit_max_q {
            break;
        }
        let vesc = if model.r[k] > 0.0 {
            (2.0 * model.cgrav[k] * model.m[k] / model.r[k])
                .max(0.0)
                .sqrt()
        } else {
            0.0
        };
        if model.controls.u_flag {
            if model.u[k] > vesc * o.gamma1_limit_max_v_div_vesc {
                break;
            }
        } else if model.controls.v_flag
            && model.v[k] > vesc * o.gamma1_limit_max_v_div_vesc
        {
            break;
        }
        if model.gamma1[k] < min_g1 {
            min_g1 = model.gamma1[k];
        }
    }
    min_g1
}

/// Evaluates `(fe_core_infall, non_fe_core_infall, non_fe_core_rebound)` in
/// $\mathrm{cm\,s^{-1}}$ matching `star/private/report.f90:378-441`.
#[must_use]
pub fn eval_core_infall_and_rebound_cms(
    model: &StarModel,
    cores: &CoreMasses,
) -> (f64, f64, f64) {
    let nz = model.nz;
    let has_vel = model.controls.u_flag || model.controls.v_flag;
    if !has_vel || nz < 2 {
        return (0.0, 0.0, 0.0);
    }
    let vel = if model.controls.u_flag {
        &model.u
    } else {
        &model.v
    };
    if vel.len() < nz {
        return (0.0, 0.0, 0.0);
    }
    let ts = &model.controls.ts_stop;
    let k_min = (0..nz)
        .min_by(|&a, &b| vel[a].total_cmp(&vel[b]))
        .unwrap_or(0);

    let mut fe_infall = 0.0_f64;
    if cores.fe_core_mass > 0.0 {
        let mut mass_sum = 0.0_f64;
        let mut k = nz - 1;
        while k > 0 {
            if model.m[k] > M_SUN * cores.fe_core_mass {
                break;
            }
            if -vel[k] > fe_infall {
                mass_sum += model.dm[k];
            }
            k -= 1;
        }
        let k_fe = k;
        if mass_sum > ts.fe_core_infall_mass * M_SUN {
            if model.controls.output.report_max_infall_inside_fe_core {
                // Reproduces `report.f90:412` (`s% v(k_fe:nz)` even when `u_flag`).
                let min_v = if model.controls.v_flag && model.v.len() >= nz {
                    model.v[k_fe..nz]
                        .iter()
                        .copied()
                        .fold(f64::INFINITY, f64::min)
                } else {
                    0.0
                };
                fe_infall = -min_v;
            } else {
                fe_infall = -vel[k_min];
            }
        }
    }

    let mut non_fe_infall = 0.0_f64;
    let mut non_fe_rebound = 0.0_f64;
    if cores.he_core_mass > 0.0 {
        let mut mass_sum = 0.0_f64;
        for k in 0..nz {
            if model.m[k] > M_SUN * cores.he_core_mass {
                continue;
            }
            if model.m[k] < M_SUN * cores.fe_core_mass {
                break;
            }
            if -vel[k] > non_fe_infall {
                mass_sum += model.dm[k];
            }
        }
        if mass_sum > ts.non_fe_core_infall_mass * M_SUN
            && model.m[k_min] <= cores.he_core_mass * M_SUN
        {
            non_fe_infall = -vel[k_min];
        }
        // `report.f90:436`: `maxval(velocity(s% he_core_k : s% fe_core_k))`.
        // In Fortran, an empty slice (`he_core_k > fe_core_k` or `fe_core_k == 0`)
        // evaluates to `-huge(1d0)`.
        if let (Some(k_he), Some(k_fe)) = (cores.he_core_k, cores.fe_core_k) {
            let k_lo = k_he.min(nz - 1);
            let k_hi = k_fe.min(nz - 1);
            if k_lo <= k_hi {
                non_fe_rebound = vel[k_lo..=k_hi]
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
            } else {
                non_fe_rebound = -f64::MAX;
            }
        } else {
            non_fe_rebound = -f64::MAX;
        }
    }

    (fe_infall, non_fe_infall, non_fe_rebound)
}

/// Checks whether any convective region qualifies as a He-burning convective
/// region (`burn_he_conv_region` in `star/private/mix_info.f90:702-736`).
fn has_burn_he_conv_region(model: &StarModel) -> bool {
    let nz = model.nz;
    if nz == 0 || model.mixing_type.len() < nz {
        return false;
    }
    let mw = &model.controls.mix_wind;
    let mut k = 0_usize;
    while k < nz {
        if model.mixing_type[k] != MixingType::ConvectiveMixing {
            k += 1;
            continue;
        }
        let k_top = k;
        while k + 1 < nz
            && model.mixing_type[k + 1] == MixingType::ConvectiveMixing
        {
            k += 1;
        }
        let k_bot = k;
        let kk_end = (k_bot + 1).min(nz - 1);
        let mut max_log_t = -1.0e99_f64;
        let mut max_x = 0.0_f64;
        let mut max_y = 0.0_f64;
        for kk in k_top..=kk_end {
            let log_t = model.ln_t[kk] / LN10;
            if log_t > max_log_t {
                max_log_t = log_t;
            }
            let x_k = model.xh_mass.get(kk).copied().unwrap_or(0.0);
            let y_k = model.yhe_mass.get(kk).copied().unwrap_or(0.0);
            if x_k > max_x {
                max_x = x_k;
            }
            if y_k > max_y {
                max_y = y_k;
            }
        }
        let is_burn_z = max_log_t > mw.burn_z_mix_region_logt
            && max_y < mw.max_y_for_burn_z_mix_region;
        if !is_burn_z
            && max_log_t > mw.burn_he_mix_region_logt
            && max_x < mw.max_x_for_burn_he_mix_region
        {
            return true;
        }
        k += 1;
    }
    false
}

/// Evaluates `phase_of_evolution` matching `set_phase_of_evolution` in
/// `star/private/star_utils.f90:2910-2969`.
#[must_use]
pub fn eval_phase_of_evolution(
    model: &StarModel,
    ctx: &StepReportContext,
    doing_first_model_of_run: bool,
) -> i32 {
    if model.nz == 0 || doing_first_model_of_run {
        return PHASE_STARTING;
    }
    let catalog = NuclideCatalog::builtin_ref();
    let center_h1 = species_col_index(model, catalog, "h1")
        .map_or(1.0e99, |j| center_avg_x_slot(model, Some(j)));
    let center_he4 = species_col_index(model, catalog, "he4")
        .map_or(1.0e99, |j| center_avg_x_slot(model, Some(j)));

    let log_center_t = center_value(model, &model.ln_t) / LN10;
    if model.aux.doing_relax {
        PHASE_RELAX
    } else if ctx.phot.logg > 6.0 {
        PHASE_WDCS
    } else if ctx.cat_lum_lsun[CAT_BURN_SI] > 1.0e2 {
        PHASE_SI_BURN
    } else if (ctx.cat_lum_lsun[CAT_OO] + ctx.cat_lum_lsun[CAT_CO]) > 1.0e2
        && center_he4 < 1.0e-4
    {
        PHASE_O_BURN
    } else if ctx.cat_lum_lsun[CAT_BURN_NE] > 1.0e2 && center_he4 < 1.0e-4 {
        PHASE_NE_BURN
    } else if ctx.cat_lum_lsun[CAT_CC] > 1.0e2 && center_he4 < 1.0e-4 {
        PHASE_C_BURN
    } else if center_he4 < 1.0e-4
        && (ctx.cores.he_core_mass - ctx.cores.co_core_mass) <= 0.1
        && has_burn_he_conv_region(model)
    {
        PHASE_TP_AGB
    } else if center_he4 <= 1.0e-4 {
        PHASE_TACHEB
    } else if ctx.center_eps_burn[CAT_3ALF] > 1.0e2 {
        PHASE_ZACHEB
    } else if ctx.cat_lum_lsun[CAT_3ALF] > 1.0e2 {
        PHASE_HE_BURN
    } else if center_h1 <= 1.0e-6 {
        PHASE_TAMS
    } else if center_h1 <= 0.3 {
        PHASE_IAMS
    } else if ctx.powers.power_nuc_burn
        >= ctx.phot.l_lsun * model.controls.lnuc_div_l_zams_limit
    {
        PHASE_ZAMS
    } else if log_center_t > 5.0 {
        PHASE_PREMS
    } else {
        PHASE_STARTING
    }
}

/// Evaluates and stores `model.aux.step_hist.phase_of_evolution` matching
/// `set_phase_of_evolution` in `star/private/star_utils.f90:2910-2969`.
pub fn set_phase_of_evolution(
    model: &mut StarModel,
    doing_first_model_of_run: bool,
) -> i32 {
    let ctx = StepReportContext::compute(model);
    let phase = eval_phase_of_evolution(model, &ctx, doing_first_model_of_run);
    model.aux.step_hist.phase_of_evolution = phase;
    phase
}

fn species_col_index(
    model: &StarModel,
    catalog: &NuclideCatalog,
    iso_name: &str,
) -> Option<usize> {
    let trimmed = iso_name.trim();
    if trimmed.is_empty() {
        return None;
    }
    let cid = catalog.get_nuclide_index(trimmed)?;
    model.chem_id.iter().position(|&id| id == cid)
}

fn get_species_mass_msun(
    model: &StarModel,
    catalog: &NuclideCatalog,
    iso_name: &str,
) -> f64 {
    let Some(j) = species_col_index(model, catalog, iso_name) else {
        return -1.0;
    };
    let mut iso_mass_g = 0.0_f64;
    for k in 0..model.nz {
        iso_mass_g += model.dm[k] * model.cell_xa(k)[j];
    }
    iso_mass_g / M_SUN
}

#[inline]
fn compare_to_target(
    model: &StarModel,
    desc: &str,
    value: f64,
    target_value: f64,
    code: &str,
) -> LimitCheckOutcome {
    let ts = &model.controls.ts_stop;
    let denom = ts.when_to_stop_atol
        + ts.when_to_stop_rtol * value.abs().max(target_value.abs());
    let err = if denom > 0.0 {
        (value - target_value).abs() / denom
    } else if value == target_value {
        0.0
    } else {
        f64::INFINITY
    };
    if err > 1.0 {
        LimitCheckOutcome::Redo {
            dt_new: 0.5 * model.dt,
            reason: format!(
                "redo with smaller timestep to get closer to stopping target {desc} (value={value:.6e}, target={target_value:.6e})"
            ),
        }
    } else {
        LimitCheckOutcome::Terminate {
            code: code.to_owned(),
            reason: desc.to_owned(),
        }
    }
}

#[inline]
fn model_time_seconds(model: &StarModel) -> f64 {
    let t_hist = model.aux.step_hist.time_sec;
    let t_age = model.star_age * SECYER;
    if (t_hist == 0.0 && model.star_age != 0.0)
        || (t_hist / SECYER - model.star_age).abs()
            > 1.0e-12 * model.star_age.abs().max(1.0)
    {
        t_age
    } else {
        t_hist
    }
}

fn check_xa_limits_block(
    model: &StarModel,
    catalog: &NuclideCatalog,
    species_names: &[String],
    limits: &[f64],
    is_upper: bool,
    region: &str,
    code: &str,
    eval_avg: impl Fn(&StarModel, usize) -> f64,
) -> Option<LimitCheckOutcome> {
    let n = species_names.len().min(limits.len());
    for j in 0..n {
        let lim = limits[j];
        if lim <= 0.0 || (is_upper && lim >= 1.0) {
            continue;
        }
        let name = species_names[j].trim();
        if name.is_empty() {
            continue;
        }
        let Some(cid) = catalog.get_nuclide_index(name) else {
            return Some(LimitCheckOutcome::Terminate {
                code: code.to_owned(),
                reason: format!("invalid species '{name}' for {code}"),
            });
        };
        let Some(i) = model.chem_id.iter().position(|&id| id == cid) else {
            continue;
        };
        let avg_x = eval_avg(model, i);
        let hit = if is_upper { avg_x > lim } else { avg_x < lim };
        if hit {
            let desc = if is_upper {
                format!("have risen above {region} upper limit for {name}")
            } else {
                format!("have dropped below {region} lower limit for {name}")
            };
            return Some(compare_to_target(model, &desc, avg_x, lim, code));
        }
    }
    None
}

fn eval_check_limits_core(
    model: &StarModel,
    ctx: &StepReportContext,
    phase: i32,
) -> Result<LimitCheckOutcome, StarError> {
    let c = &model.controls;
    let ts = &c.ts_stop;
    let nz = model.nz;
    if nz == 0 {
        return Ok(LimitCheckOutcome::KeepGoing);
    }
    let catalog = NuclideCatalog::builtin_ref();

    let (max_period_number, period_number, grekm_avg_abs) = if c.rsp_flag {
        (
            c.rsp_max_num_periods,
            model.aux.rsp.num_periods as isize,
            model.aux.rsp.grekm_avg_abs,
        )
    } else {
        (0_isize, -1_isize, 1.0e99)
    };

    let species_mass_for_min_limit =
        get_species_mass_msun(model, catalog, &c.star_species_mass_min_limit_iso);
    let species_mass_for_max_limit =
        get_species_mass_msun(model, catalog, &c.star_species_mass_max_limit_iso);

    let csound_surf = csound_face_at(model, 0);
    let (v_surf, v_div_csound_max, max_abs_vel) =
        if c.u_flag && model.u.len() >= nz {
            let mut max_vc = 0.0_f64;
            let mut max_v = 0.0_f64;
            for k in 0..nz {
                let cs = model.csound[k];
                if cs > 0.0 {
                    max_vc = max_vc.max((model.u[k] / cs).abs());
                }
                if model.u[k].abs() > max_v.abs() {
                    max_v = model.u[k];
                }
            }
            (model.u[0], max_vc, max_v)
        } else if c.v_flag && model.v.len() >= nz {
            let mut max_vc = 0.0_f64;
            let mut max_v = 0.0_f64;
            for k in 0..nz {
                let cs = csound_face_at(model, k);
                if cs > 0.0 {
                    max_vc = max_vc.max((model.v[k] / cs).abs());
                }
                if model.v[k].abs() > max_v.abs() {
                    max_v = model.v[k];
                }
            }
            (model.v[0], max_vc, max_v)
        } else {
            (0.0, 0.0, 0.0)
        };

    let remnant_mass = get_remnant_mass(model) / M_SUN;
    let ejecta_mass = get_ejecta_mass(model) / M_SUN;

    let (v_surf_div_v_kh, v_surf_div_v_esc) = if ctx.phot.r_rsun > 0.0 {
        // Reproduces `star/private/do_one_utils.f90:635` verbatim (`#265` under `#169`):
        // mixed units `abs(v_surf) [cm/s] * kh_timescale [yr] / photosphere_r [Rsun]`.
        let v_kh = v_surf.abs() * ctx.kh_timescale_yr / ctx.phot.r_rsun;
        let v_esc = if model.cgrav[0] <= 0.0 {
            0.0
        } else {
            v_surf
                / (2.0 * model.cgrav[0] * model.m[0] / (ctx.phot.r_rsun * R_SUN))
                    .max(1.0e-99)
                    .sqrt()
        };
        (v_kh, v_esc)
    } else {
        (0.0, 0.0)
    };

    let log_surface_gravity = safe_log10(model.grav[0]);
    let log_surface_temperature = model.ln_t[0] / LN10;
    let log_surface_density = model.lnd[0] / LN10;
    let log_surface_pressure = model.ln_peos[0] / LN10;
    let log_surface_radius = safe_log10(model.r[0] / R_SUN);
    let log_surface_luminosity = safe_log10(ctx.phot.l_lsun);

    let center_gamma = center_value(model, &model.gam);
    let center_degeneracy = center_value(model, &model.eta);
    let center_ye = center_value(model, &model.ye);
    let log_center_temperature = center_value(model, &model.ln_t) / LN10;
    let log_center_density = center_value(model, &model.lnd) / LN10;
    let center_entropy = ctx.center_entropy(model);
    let max_entropy = model
        .ln_s
        .iter()
        .take(nz)
        .map(|&lns| flexmesa_math::exp(lns) * AMU / KERG)
        .fold(f64::NEG_INFINITY, f64::max);
    let log_max_temperature = model
        .ln_t
        .iter()
        .take(nz)
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        / LN10;

    let center_h1 = species_col_index(model, catalog, "h1")
        .map_or(0.0, |j| center_avg_x_slot(model, Some(j)));
    let center_he4 = species_col_index(model, catalog, "he4")
        .map_or(0.0, |j| center_avg_x_slot(model, Some(j)));

    let power_nuc_burn = ctx.powers.power_nuc_burn;
    let power_h_burn = ctx.powers.power_h_burn;
    let power_he_burn = ctx.powers.power_he_burn;
    let power_z_burn = ctx.powers.power_z_burn;
    let teff = if model.teff > 0.0 {
        model.teff
    } else {
        ctx.phot.black_body_t
    };
    let log_teff = safe_log10(teff);
    let lnuc_div_l = if ctx.phot.l_lsun > 0.0 {
        power_nuc_burn / ctx.phot.l_lsun
    } else {
        0.0
    };

    let need_astero = ts.delta_nu_lower_limit > 0.0
        || ts.delta_nu_upper_limit > 0.0
        || ts.delta_pg_lower_limit > 0.0
        || ts.delta_pg_upper_limit > 0.0;
    let (delta_nu, delta_pg) = if need_astero && ctx.phot.l_lsun > 0.0 {
        let (_, dnu, _, dpg) = eval_astero_report_scalars(model, &ctx.phot);
        (dnu, dpg)
    } else {
        (0.0, 0.0)
    };

    let peak_burn_vconv_div_cs = if c.dxdt_nuc_factor > 0.0 {
        let mut k_peak = 0_usize;
        let mut eps_peak = f64::NEG_INFINITY;
        for k in 0..nz {
            if model.eps_nuc[k] > eps_peak {
                eps_peak = model.eps_nuc[k];
                k_peak = k;
            }
        }
        let cs = model.csound[k_peak];
        if cs > 0.0 {
            model.conv_vel[k_peak] / cs
        } else {
            0.0
        }
    } else {
        0.0
    };

    let star_mass = model.mstar / M_SUN;
    let envelope_fraction_left = if c.initial_mass > ctx.cores.he_core_mass {
        (star_mass - ctx.cores.he_core_mass)
            / (c.initial_mass - ctx.cores.he_core_mass)
    } else {
        1.0
    };

    let mut min_pgas_div_p = 1.0e99_f64;
    for k in (0..nz).rev() {
        if model.q[k] > ts.pgas_div_p_limit_max_q {
            break;
        }
        let peos_k = model.peos[k];
        if peos_k > 0.0 {
            let ratio = flexmesa_math::exp(model.ln_pgas[k]) / peos_k;
            if ratio < min_pgas_div_p {
                min_pgas_div_p = ratio;
            }
        }
    }

    let min_gamma1 = eval_min_gamma1(model);

    let mut max_omega_div_omega_crit = 0.0_f64;
    if c.rotation_flag && ts.omega_div_omega_crit_limit > 0.0 {
        for k in 0..nz {
            let wc = eval_upstream_omega_crit(model, k);
            if wc > 0.0 {
                let ratio = model.omega[k].abs() / wc;
                if ratio > max_omega_div_omega_crit {
                    max_omega_div_omega_crit = ratio;
                }
            }
        }
    }

    let (fe_core_infall, non_fe_core_infall, non_fe_core_rebound) =
        eval_core_infall_and_rebound_cms(model, &ctx.cores);

    // --- Pre-cascade checks (`do_one_utils.f90:713-745`) ---
    if ts.max_abs_rel_run_e_err > 0.0 && !model.aux.doing_relax {
        let e_tot = model.aux.total_energy;
        if e_tot != 0.0
            && (model.aux.cumulative_energy_error / e_tot).abs()
                > ts.max_abs_rel_run_e_err
        {
            return Ok(LimitCheckOutcome::Terminate {
                code: String::from("exceeded max abs rel_run_E_err"),
                reason: String::from(
                    "abs rel_run_E_err exceeds max_abs_rel_run_E_err",
                ),
            });
        }
    }

    if max_abs_vel.abs() > CLIGHT && ts.retry_for_v_above_clight {
        return Ok(LimitCheckOutcome::Retry {
            reason: format!(
                "maximum velocity ({max_abs_vel:.3e} cm/s) exceeds speed of light"
            ),
        });
    }

    // Reproduces `star/private/do_one_utils.f90:738-745` verbatim (`#264` under `#169`):
    // upstream aborts via `mesa_error` at `0.75 * peak_burn_vconv_div_cs_limit`
    // before the clean `>= limit` stop at line 790 can be reached.
    if peak_burn_vconv_div_cs > 0.75 * ts.peak_burn_vconv_div_cs_limit {
        return Err(StarError::InvalidInput(format!(
            "peak_burn_vconv_div_cs ({peak_burn_vconv_div_cs:.6e}) > 0.75 * peak_burn_vconv_div_cs_limit ({:.6e}) (`do_one_utils.f90:738-745`, #264)",
            ts.peak_burn_vconv_div_cs_limit
        )));
    }

    let time_sec = model_time_seconds(model);

    // --- Main `if / else if` stopping cascade (`do_one_utils.f90:769-1258`) ---
    let mut outcome = LimitCheckOutcome::KeepGoing;

    if c.max_age > 0.0 && model.star_age >= c.max_age {
        outcome = compare_to_target(
            model,
            "star_age >= max_age",
            model.star_age,
            c.max_age,
            "max_age",
        );
    } else if ts.max_age_in_days > 0.0
        && time_sec >= ts.max_age_in_days * SECDAY
    {
        outcome = compare_to_target(
            model,
            "time >= max_age_in_days",
            time_sec / SECDAY,
            ts.max_age_in_days,
            "max_age",
        );
    } else if c
        .max_age_in_seconds
        .is_some_and(|lim| lim > 0.0 && time_sec >= lim)
    {
        let lim = c.max_age_in_seconds.unwrap_or(0.0);
        outcome = compare_to_target(
            model,
            "time >= max_age_in_seconds",
            time_sec,
            lim,
            "max_age",
        );
    } else if ts.omega_div_omega_crit_limit > 0.0
        && max_omega_div_omega_crit >= ts.omega_div_omega_crit_limit
    {
        outcome = LimitCheckOutcome::Terminate {
            code: String::from("max_omega_div_omega_crit"),
            reason: String::from(
                "max_omega_div_omega_crit >= omega_div_omega_crit_limit",
            ),
        };
    } else if peak_burn_vconv_div_cs >= ts.peak_burn_vconv_div_cs_limit {
        outcome = LimitCheckOutcome::Terminate {
            code: String::from("peak_burn_vconv_div_cs_limit"),
            reason: String::from(
                "peak_burn_vconv_div_cs >= peak_burn_vconv_div_cs_limit",
            ),
        };
    } else if c
        .max_model_number
        .is_some_and(|max_mod| model.model_number >= max_mod)
    {
        outcome = LimitCheckOutcome::Terminate {
            code: String::from("max_model_number"),
            reason: String::from("model_number >= max_model_number"),
        };
    } else if max_period_number >= 0 && period_number >= max_period_number {
        outcome = LimitCheckOutcome::Terminate {
            code: String::from("reached max number of periods"),
            reason: String::from("period_number >= max_period_number"),
        };
    } else if c.rsp.grekm_avg_abs_limit >= 0.0
        && period_number >= 10
        && grekm_avg_abs < c.rsp.grekm_avg_abs_limit
    {
        outcome = LimitCheckOutcome::Terminate {
            code: String::from("RSP_GREKM_avg_abs_limit"),
            reason: String::from("GREKM_avg_abs < RSP_GREKM_avg_abs_limit"),
        };
    } else if let Some(lim) =
        ts.eta_center_limit.filter(|&lim| center_degeneracy >= lim)
    {
        outcome = compare_to_target(
            model,
            "center_degeneracy >= eta_center_limit",
            center_degeneracy,
            lim,
            "eta_center_limit",
        );
    } else if log_center_temperature >= c.log_center_temp_upper_limit {
        outcome = compare_to_target(
            model,
            "log_center_temperature >= log_center_temp_upper_limit",
            log_center_temperature,
            c.log_center_temp_upper_limit,
            "log_center_temp_upper_limit",
        );
    } else if let Some(lim) = c
        .log_center_temp_lower_limit
        .filter(|&lim| log_center_temperature <= lim)
    {
        outcome = compare_to_target(
            model,
            "log_center_temperature <= log_center_temp_lower_limit",
            log_center_temperature,
            lim,
            "log_center_temp_lower_limit",
        );
    } else if max_entropy >= ts.max_entropy_upper_limit {
        outcome = compare_to_target(
            model,
            "max_entropy >= max_entropy_upper_limit",
            max_entropy,
            ts.max_entropy_upper_limit,
            "max_entropy_upper_limit",
        );
    } else if max_entropy <= ts.max_entropy_lower_limit {
        outcome = compare_to_target(
            model,
            "max_entropy <= max_entropy_lower_limit",
            max_entropy,
            ts.max_entropy_lower_limit,
            "max_entropy_lower_limit",
        );
    } else if center_entropy >= ts.center_entropy_upper_limit {
        outcome = compare_to_target(
            model,
            "center_entropy >= center_entropy_upper_limit",
            center_entropy,
            ts.center_entropy_upper_limit,
            "center_entropy_upper_limit",
        );
    } else if center_entropy <= ts.center_entropy_lower_limit {
        outcome = compare_to_target(
            model,
            "center_entropy <= center_entropy_lower_limit",
            center_entropy,
            ts.center_entropy_lower_limit,
            "center_entropy_lower_limit",
        );
    } else if let Some(lim) = c
        .log_center_density_lower_limit
        .filter(|&lim| log_center_density <= lim)
    {
        outcome = compare_to_target(
            model,
            "log_center_density <= log_center_density_lower_limit",
            log_center_density,
            lim,
            "log_center_density_lower_limit",
        );
    } else if log_center_density >= c.log_center_density_upper_limit {
        outcome = compare_to_target(
            model,
            "log_center_density >= log_center_density_upper_limit",
            log_center_density,
            c.log_center_density_upper_limit,
            "log_center_density_upper_limit",
        );
    } else if let Some(lim) =
        c.gamma_center_limit.filter(|&lim| center_gamma > lim)
    {
        outcome = compare_to_target(
            model,
            "center_gamma > gamma_center_limit",
            center_gamma,
            lim,
            "gamma_center_limit",
        );
    } else if log_max_temperature >= c.log_max_temp_upper_limit {
        outcome = compare_to_target(
            model,
            "log_max_temperature >= log_max_temp_upper_limit",
            log_max_temperature,
            c.log_max_temp_upper_limit,
            "log_max_temp_upper_limit",
        );
    } else if log_max_temperature <= ts.log_max_temp_lower_limit {
        outcome = compare_to_target(
            model,
            "log_max_temperature <= log_max_temp_lower_limit",
            log_max_temperature,
            ts.log_max_temp_lower_limit,
            "log_max_temp_lower_limit",
        );
    } else if center_he4 < ts.hb_limit && center_h1 < 1.0e-4 {
        outcome = compare_to_target(
            model,
            "center he4 < HB_limit",
            center_he4,
            ts.hb_limit,
            "HB_limit",
        );
    } else if c.star_mass_min_limit > 0.0 && star_mass <= c.star_mass_min_limit
    {
        outcome = compare_to_target(
            model,
            "star_mass <= star_mass_min_limit",
            star_mass,
            c.star_mass_min_limit,
            "star_mass_min_limit",
        );
    } else if c.star_mass_max_limit > 0.0 && star_mass >= c.star_mass_max_limit
    {
        outcome = compare_to_target(
            model,
            "star_mass >= star_mass_max_limit",
            star_mass,
            c.star_mass_max_limit,
            "star_mass_max_limit",
        );
    } else if c.remnant_mass_min_limit > 0.0
        && remnant_mass <= c.remnant_mass_min_limit
    {
        outcome = compare_to_target(
            model,
            "remnant_mass <= remnant_mass_min_limit",
            remnant_mass,
            c.remnant_mass_min_limit,
            "remnant_mass_min_limit",
        );
    } else if ts.ejecta_mass_max_limit > 0.0
        && ejecta_mass >= ts.ejecta_mass_max_limit
    {
        outcome = compare_to_target(
            model,
            "ejecta_mass >= ejecta_mass_max_limit",
            ejecta_mass,
            ts.ejecta_mass_max_limit,
            "ejecta_mass_max_limit",
        );
    } else if species_mass_for_min_limit >= 0.0
        && species_mass_for_min_limit <= c.star_species_mass_min_limit
    {
        outcome = compare_to_target(
            model,
            "species total mass <= star_species_mass_min_limit",
            species_mass_for_min_limit,
            c.star_species_mass_min_limit,
            "star_species_mass_min_limit",
        );
    } else if species_mass_for_max_limit >= 0.0
        && species_mass_for_max_limit >= c.star_species_mass_max_limit
    {
        outcome = compare_to_target(
            model,
            "species total mass >= star_species_mass_max_limit",
            species_mass_for_max_limit,
            c.star_species_mass_max_limit,
            "star_species_mass_max_limit",
        );
    } else if ts.xmstar_min_limit > 0.0 && model.xmstar <= ts.xmstar_min_limit {
        outcome = compare_to_target(
            model,
            "xmstar <= xmstar_min_limit",
            model.xmstar,
            ts.xmstar_min_limit,
            "xmstar_min_limit",
        );
    } else if ts.xmstar_max_limit > 0.0 && model.xmstar >= ts.xmstar_max_limit {
        outcome = compare_to_target(
            model,
            "xmstar >= xmstar_max_limit",
            model.xmstar,
            ts.xmstar_max_limit,
            "xmstar_max_limit",
        );
    } else if star_mass - ctx.cores.he_core_mass < c.envelope_mass_limit {
        outcome = compare_to_target(
            model,
            "envelope mass < envelope_mass_limit",
            star_mass - ctx.cores.he_core_mass,
            c.envelope_mass_limit,
            "envelope_mass_limit",
        );
    } else if envelope_fraction_left < ts.envelope_fraction_left_limit {
        outcome = compare_to_target(
            model,
            "envelope_fraction_left < limit",
            envelope_fraction_left,
            ts.envelope_fraction_left_limit,
            "envelope_fraction_left_limit",
        );
    } else if let Some(lim) = c
        .he_core_mass_limit
        .filter(|&lim| ctx.cores.he_core_mass >= lim)
    {
        outcome = compare_to_target(
            model,
            "he_core_mass >= he_core_mass_limit",
            ctx.cores.he_core_mass,
            lim,
            "he_core_mass_limit",
        );
    } else if let Some(lim) = c
        .co_core_mass_limit
        .filter(|&lim| ctx.cores.co_core_mass >= lim)
    {
        outcome = compare_to_target(
            model,
            "co_core_mass >= co_core_mass_limit",
            ctx.cores.co_core_mass,
            lim,
            "co_core_mass_limit",
        );
    } else if let Some(lim) = c
        .one_core_mass_limit
        .filter(|&lim| ctx.cores.one_core_mass >= lim)
    {
        outcome = compare_to_target(
            model,
            "one_core_mass >= one_core_mass_limit",
            ctx.cores.one_core_mass,
            lim,
            "one_core_mass_limit",
        );
    } else if ctx.cores.fe_core_mass >= ts.fe_core_mass_limit {
        outcome = compare_to_target(
            model,
            "fe_core_mass >= fe_core_mass_limit",
            ctx.cores.fe_core_mass,
            ts.fe_core_mass_limit,
            "fe_core_mass_limit",
        );
    } else if ctx.cores.neutron_rich_core_mass >= ts.neutron_rich_core_mass_limit
    {
        outcome = compare_to_target(
            model,
            "neutron_rich_core_mass >= neutron_rich_core_mass_limit",
            ctx.cores.neutron_rich_core_mass,
            ts.neutron_rich_core_mass_limit,
            "neutron_rich_core_mass_limit",
        );
    } else if ctx.cores.he_core_mass >= ctx.cores.co_core_mass
        && ctx.cores.co_core_mass > 0.0
        && center_he4 < 1.0e-4
        && (ctx.cores.he_core_mass - ctx.cores.co_core_mass)
            < c.he_layer_mass_lower_limit
    {
        outcome = compare_to_target(
            model,
            "he layer mass < he_layer_mass_lower_limit",
            ctx.cores.he_core_mass - ctx.cores.co_core_mass,
            c.he_layer_mass_lower_limit,
            "he_layer_mass_lower_limit",
        );
    } else if ts.abs_diff_lg_lh_lg_ls_limit > 0.0
        && (safe_log10(power_h_burn) - log_surface_luminosity).abs()
            <= ts.abs_diff_lg_lh_lg_ls_limit
    {
        let diff = (safe_log10(power_h_burn) - log_surface_luminosity).abs();
        outcome = compare_to_target(
            model,
            "abs(lg_LH - lg_Ls) <= limit",
            diff,
            ts.abs_diff_lg_lh_lg_ls_limit,
            "abs_diff_lg_LH_lg_Ls_limit",
        );
    } else if let Some(lim) = c.teff_lower_limit.filter(|&lim| teff <= lim) {
        outcome = compare_to_target(
            model,
            "Teff <= Teff_lower_limit",
            teff,
            lim,
            "Teff_lower_limit",
        );
    } else if let Some(lim) = c.teff_upper_limit.filter(|&lim| teff >= lim) {
        outcome = compare_to_target(
            model,
            "Teff >= Teff_upper_limit",
            teff,
            lim,
            "Teff_upper_limit",
        );
    } else if ts.delta_nu_lower_limit > 0.0
        && delta_nu <= ts.delta_nu_lower_limit
    {
        outcome = compare_to_target(
            model,
            "delta_nu <= delta_nu_lower_limit",
            delta_nu,
            ts.delta_nu_lower_limit,
            "delta_nu_lower_limit",
        );
    } else if ts.delta_nu_upper_limit > 0.0
        && delta_nu >= ts.delta_nu_upper_limit
    {
        outcome = compare_to_target(
            model,
            "delta_nu >= delta_nu_upper_limit",
            delta_nu,
            ts.delta_nu_upper_limit,
            "delta_nu_upper_limit",
        );
    } else if ts.delta_pg_lower_limit > 0.0
        && delta_pg <= ts.delta_pg_lower_limit
    {
        outcome = compare_to_target(
            model,
            "delta_Pg <= delta_Pg_lower_limit",
            delta_pg,
            ts.delta_pg_lower_limit,
            "delta_Pg_lower_limit",
        );
    } else if ts.delta_pg_upper_limit > 0.0
        && delta_pg >= ts.delta_pg_upper_limit
    {
        outcome = compare_to_target(
            model,
            "delta_Pg >= delta_Pg_upper_limit",
            delta_pg,
            ts.delta_pg_upper_limit,
            "delta_Pg_upper_limit",
        );
    } else if ctx.phot.m_msun - model.m_center / M_SUN
        <= ts.photosphere_m_sub_m_center_limit
    {
        outcome = compare_to_target(
            model,
            "photosphere_m - M_center/Msun <= photosphere_m_sub_M_center_limit",
            ctx.phot.m_msun - model.m_center / M_SUN,
            ts.photosphere_m_sub_m_center_limit,
            "photosphere_m_sub_M_center_limit",
        );
    } else if ctx.phot.m_msun <= ts.photosphere_m_lower_limit {
        outcome = compare_to_target(
            model,
            "photosphere_m <= photosphere_m_lower_limit",
            ctx.phot.m_msun,
            ts.photosphere_m_lower_limit,
            "photosphere_m_lower_limit",
        );
    } else if ctx.phot.m_msun >= ts.photosphere_m_upper_limit {
        outcome = compare_to_target(
            model,
            "photosphere_m >= photosphere_m_upper_limit",
            ctx.phot.m_msun,
            ts.photosphere_m_upper_limit,
            "photosphere_m_upper_limit",
        );
    } else if ctx.phot.r_rsun <= ts.photosphere_r_lower_limit {
        outcome = compare_to_target(
            model,
            "photosphere_r <= photosphere_r_lower_limit",
            ctx.phot.r_rsun,
            ts.photosphere_r_lower_limit,
            "photosphere_r_lower_limit",
        );
    } else if ctx.phot.r_rsun >= ts.photosphere_r_upper_limit {
        outcome = compare_to_target(
            model,
            "photosphere_r >= photosphere_r_upper_limit",
            ctx.phot.r_rsun,
            ts.photosphere_r_upper_limit,
            "photosphere_r_upper_limit",
        );
    } else if let Some(lim) =
        c.log_teff_lower_limit.filter(|&lim| log_teff <= lim)
    {
        outcome = compare_to_target(
            model,
            "log_Teff <= log_Teff_lower_limit",
            log_teff,
            lim,
            "log_Teff_lower_limit",
        );
    } else if let Some(lim) =
        c.log_teff_upper_limit.filter(|&lim| log_teff >= lim)
    {
        outcome = compare_to_target(
            model,
            "log_Teff >= log_Teff_upper_limit",
            log_teff,
            lim,
            "log_Teff_upper_limit",
        );
    } else if log_surface_temperature <= ts.log_tsurf_lower_limit {
        outcome = compare_to_target(
            model,
            "log_surface_temperature <= log_Tsurf_lower_limit",
            log_surface_temperature,
            ts.log_tsurf_lower_limit,
            "log_Tsurf_lower_limit",
        );
    } else if log_surface_temperature >= ts.log_tsurf_upper_limit {
        outcome = compare_to_target(
            model,
            "log_surface_temperature >= log_Tsurf_upper_limit",
            log_surface_temperature,
            ts.log_tsurf_upper_limit,
            "log_Tsurf_upper_limit",
        );
    } else if log_surface_radius <= ts.log_rsurf_lower_limit {
        outcome = compare_to_target(
            model,
            "log_surface_radius <= log_Rsurf_lower_limit",
            log_surface_radius,
            ts.log_rsurf_lower_limit,
            "log_Rsurf_lower_limit",
        );
    } else if let Some(lim) = c
        .log_rsurf_upper_limit
        .filter(|&lim| log_surface_radius >= lim)
    {
        outcome = compare_to_target(
            model,
            "log_surface_radius >= log_Rsurf_upper_limit",
            log_surface_radius,
            lim,
            "log_Rsurf_upper_limit",
        );
    } else if log_surface_pressure <= ts.log_psurf_lower_limit {
        outcome = compare_to_target(
            model,
            "log_surface_pressure <= log_Psurf_lower_limit",
            log_surface_pressure,
            ts.log_psurf_lower_limit,
            "log_Psurf_lower_limit",
        );
    } else if let Some(lim) = c
        .log_psurf_upper_limit
        .filter(|&lim| log_surface_pressure >= lim)
    {
        outcome = compare_to_target(
            model,
            "log_surface_pressure >= log_Psurf_upper_limit",
            log_surface_pressure,
            lim,
            "log_Psurf_upper_limit",
        );
    } else if log_surface_density <= ts.log_dsurf_lower_limit {
        outcome = compare_to_target(
            model,
            "log_surface_density <= log_Dsurf_lower_limit",
            log_surface_density,
            ts.log_dsurf_lower_limit,
            "log_Dsurf_lower_limit",
        );
    } else if log_surface_density >= ts.log_dsurf_upper_limit {
        outcome = compare_to_target(
            model,
            "log_surface_density >= log_Dsurf_upper_limit",
            log_surface_density,
            ts.log_dsurf_upper_limit,
            "log_Dsurf_upper_limit",
        );
    } else if let Some(lim) = c
        .log_l_lower_limit
        .filter(|&lim| log_surface_luminosity <= lim)
    {
        outcome = compare_to_target(
            model,
            "log_surface_luminosity <= log_L_lower_limit",
            log_surface_luminosity,
            lim,
            "log_L_lower_limit",
        );
    } else if let Some(lim) = c
        .log_l_upper_limit
        .filter(|&lim| log_surface_luminosity >= lim)
    {
        outcome = compare_to_target(
            model,
            "log_surface_luminosity >= log_L_upper_limit",
            log_surface_luminosity,
            lim,
            "log_L_upper_limit",
        );
    } else if let Some(lim) =
        c.log_g_lower_limit.filter(|&lim| log_surface_gravity <= lim)
    {
        outcome = compare_to_target(
            model,
            "log_surface_gravity <= log_g_lower_limit",
            log_surface_gravity,
            lim,
            "log_g_lower_limit",
        );
    } else if let Some(lim) =
        c.log_g_upper_limit.filter(|&lim| log_surface_gravity >= lim)
    {
        outcome = compare_to_target(
            model,
            "log_surface_gravity >= log_g_upper_limit",
            log_surface_gravity,
            lim,
            "log_g_upper_limit",
        );
    } else if let Some(lim) = c
        .power_nuc_burn_upper_limit
        .filter(|&lim| power_nuc_burn >= lim)
    {
        outcome = compare_to_target(
            model,
            "power_nuc_burn >= power_nuc_burn_upper_limit",
            power_nuc_burn,
            lim,
            "power_nuc_burn_upper_limit",
        );
    } else if let Some(lim) =
        ts.power_h_burn_upper_limit.filter(|&lim| power_h_burn >= lim)
    {
        outcome = compare_to_target(
            model,
            "power_h_burn >= power_h_burn_upper_limit",
            power_h_burn,
            lim,
            "power_h_burn_upper_limit",
        );
    } else if let Some(lim) = c
        .power_he_burn_upper_limit
        .filter(|&lim| power_he_burn >= lim)
    {
        outcome = compare_to_target(
            model,
            "power_he_burn >= power_he_burn_upper_limit",
            power_he_burn,
            lim,
            "power_he_burn_upper_limit",
        );
    } else if let Some(lim) =
        ts.power_z_burn_upper_limit.filter(|&lim| power_z_burn >= lim)
    {
        outcome = compare_to_target(
            model,
            "power_z_burn >= power_z_burn_upper_limit",
            power_z_burn,
            lim,
            "power_z_burn_upper_limit",
        );
    } else if let Some(lim) = c
        .power_nuc_burn_lower_limit
        .filter(|&lim| power_nuc_burn < lim)
    {
        outcome = compare_to_target(
            model,
            "power_nuc_burn < power_nuc_burn_lower_limit",
            power_nuc_burn,
            lim,
            "power_nuc_burn_lower_limit",
        );
    } else if let Some(lim) =
        ts.power_h_burn_lower_limit.filter(|&lim| power_h_burn < lim)
    {
        outcome = compare_to_target(
            model,
            "power_h_burn < power_h_burn_lower_limit",
            power_h_burn,
            lim,
            "power_h_burn_lower_limit",
        );
    } else if let Some(lim) = ts
        .power_he_burn_lower_limit
        .filter(|&lim| power_he_burn < lim)
    {
        outcome = compare_to_target(
            model,
            "power_he_burn < power_he_burn_lower_limit",
            power_he_burn,
            lim,
            "power_he_burn_lower_limit",
        );
    } else if let Some(lim) =
        ts.power_z_burn_lower_limit.filter(|&lim| power_z_burn < lim)
    {
        outcome = compare_to_target(
            model,
            "power_z_burn < power_z_burn_lower_limit",
            power_z_burn,
            lim,
            "power_z_burn_lower_limit",
        );
    } else if center_ye < ts.center_ye_lower_limit {
        outcome = compare_to_target(
            model,
            "center_Ye < center_Ye_lower_limit",
            center_ye,
            ts.center_ye_lower_limit,
            "center_Ye_lower_limit",
        );
    } else if let Some(lim) =
        c.center_r_lower_limit.filter(|&lim| model.r_center < lim)
    {
        outcome = compare_to_target(
            model,
            "R_center < center_R_lower_limit",
            model.r_center,
            lim,
            "center_R_lower_limit",
        );
    } else if fe_core_infall > c.fe_core_infall_limit {
        let e_end = model.aux.step_hist.total_energy_end;
        let e_err = model.aux.step_hist.error_in_energy_conservation;
        let rel_ok = if e_end == 0.0 && e_err == 0.0 {
            true
        } else if e_end != 0.0 {
            (e_err / e_end).abs()
                < ts.hard_limit_for_rel_error_in_energy_conservation
        } else {
            false
        };
        if rel_ok {
            outcome = LimitCheckOutcome::Terminate {
                code: String::from("fe_core_infall_limit"),
                reason: String::from("fe_core_infall > fe_core_infall_limit"),
            };
        }
    } else if let Some(lim) = ts
        .non_fe_core_infall_limit
        .filter(|&lim| non_fe_core_infall > lim)
    {
        outcome = compare_to_target(
            model,
            "non_fe_core_infall > non_fe_core_infall_limit",
            non_fe_core_infall,
            lim,
            "non_fe_core_infall_limit",
        );
    } else if non_fe_core_rebound > ts.non_fe_core_rebound_limit {
        outcome = compare_to_target(
            model,
            "non_fe_core_rebound > non_fe_core_rebound_limit",
            non_fe_core_rebound,
            ts.non_fe_core_rebound_limit,
            "non_fe_core_rebound_limit",
        );
    } else if csound_surf > 0.0
        && (v_surf / csound_surf) > ts.v_div_csound_surf_limit
    {
        outcome = compare_to_target(
            model,
            "v_surf/csound_surf > v_div_csound_surf_limit",
            v_surf / csound_surf,
            ts.v_div_csound_surf_limit,
            "v_div_csound_surf_limit",
        );
    } else if v_div_csound_max > ts.v_div_csound_max_limit {
        outcome = compare_to_target(
            model,
            "v_div_csound_max > v_div_csound_max_limit",
            v_div_csound_max,
            ts.v_div_csound_max_limit,
            "v_div_csound_max_limit",
        );
    } else if min_gamma1 < ts.gamma1_limit {
        outcome = compare_to_target(
            model,
            "min_gamma1 < gamma1_limit",
            min_gamma1,
            ts.gamma1_limit,
            "gamma1_limit",
        );
    } else if min_pgas_div_p < ts.pgas_div_p_limit {
        outcome = compare_to_target(
            model,
            "min_pgas_div_p < Pgas_div_P_limit",
            min_pgas_div_p,
            ts.pgas_div_p_limit,
            "Pgas_div_P_limit",
        );
    } else if lnuc_div_l <= ts.lnuc_div_l_lower_limit {
        outcome = compare_to_target(
            model,
            "Lnuc_div_L <= Lnuc_div_L_lower_limit",
            lnuc_div_l,
            ts.lnuc_div_l_lower_limit,
            "Lnuc_div_L_lower_limit",
        );
    } else if lnuc_div_l >= ts.lnuc_div_l_upper_limit {
        outcome = compare_to_target(
            model,
            "Lnuc_div_L >= Lnuc_div_L_upper_limit",
            lnuc_div_l,
            ts.lnuc_div_l_upper_limit,
            "Lnuc_div_L_upper_limit",
        );
    } else if v_surf_div_v_kh <= ts.v_surf_div_v_kh_lower_limit {
        outcome = compare_to_target(
            model,
            "v_surf_div_v_kh <= v_surf_div_v_kh_lower_limit",
            v_surf_div_v_kh,
            ts.v_surf_div_v_kh_lower_limit,
            "v_surf_div_v_kh_lower_limit",
        );
    } else if let Some(lim) = c
        .v_surf_div_v_kh_upper_limit
        .filter(|&lim| v_surf_div_v_kh >= lim)
    {
        outcome = compare_to_target(
            model,
            "v_surf_div_v_kh >= v_surf_div_v_kh_upper_limit",
            v_surf_div_v_kh,
            lim,
            "v_surf_div_v_kh_upper_limit",
        );
    } else if let Some(lim) = c
        .v_surf_div_v_esc_limit
        .filter(|&lim| v_surf_div_v_esc >= lim)
    {
        outcome = compare_to_target(
            model,
            "v_surf_div_v_esc >= v_surf_div_v_esc_limit",
            v_surf_div_v_esc,
            lim,
            "v_surf_div_v_esc_limit",
        );
    } else if v_surf * 1.0e-5 >= ts.v_surf_kms_limit {
        outcome = compare_to_target(
            model,
            "v_surf_kms >= v_surf_kms_limit",
            v_surf * 1.0e-5,
            ts.v_surf_kms_limit,
            "v_surf_kms_limit",
        );
    } else if ts.stop_when_reach_this_cumulative_extra_heating > 0.0
        && model.aux.step_hist.cumulative_extra_heating
            >= ts.stop_when_reach_this_cumulative_extra_heating
    {
        outcome = compare_to_target(
            model,
            "cumulative_extra_heating >= limit",
            model.aux.step_hist.cumulative_extra_heating,
            ts.stop_when_reach_this_cumulative_extra_heating,
            "cumulative extra heating",
        );
    } else if c.stop_near_zams && lnuc_div_l >= c.lnuc_div_l_zams_limit {
        outcome = LimitCheckOutcome::Terminate {
            code: String::from("Lnuc_div_L_zams_limit"),
            reason: String::from("Lnuc_div_L >= Lnuc_div_L_zams_limit"),
        };
    } else if let Some(phase_code) = match_stop_at_phase(model, phase) {
        outcome = LimitCheckOutcome::Terminate {
            code: String::from(phase_code),
            reason: format!("phase_of_evolution == {phase_code}"),
        };
    }

    // Hydro shock mass check (`do_one_utils.f90:1260-1265`, runs after `end if`
    // in upstream, so overrides `outcome` if triggered).
    if (c.u_flag || c.v_flag) && ts.shock_mass_upper_limit > 0.0 {
        let shock_mass = get_shock_mass_msun(model);
        if shock_mass >= ts.shock_mass_upper_limit {
            outcome = compare_to_target(
                model,
                "shock_mass >= shock_mass_upper_limit",
                shock_mass,
                ts.shock_mass_upper_limit,
                "shock_mass_upper_limit",
            );
        }
    }

    if outcome != LimitCheckOutcome::KeepGoing {
        return Ok(outcome);
    }

    // Abundance limits (`do_one_utils.f90:1269-1422`).
    if let Some(res) = check_xa_limits_block(
        model,
        catalog,
        &c.xa_central_lower_limit_species,
        &c.xa_central_lower_limit,
        false,
        "central",
        "xa_central_lower_limit",
        |m, j| center_avg_x_slot(m, Some(j)),
    ) {
        return Ok(res);
    }
    if let Some(res) = check_xa_limits_block(
        model,
        catalog,
        &c.xa_central_upper_limit_species,
        &c.xa_central_upper_limit,
        true,
        "central",
        "xa_central_upper_limit",
        |m, j| center_avg_x_slot(m, Some(j)),
    ) {
        return Ok(res);
    }
    if let Some(res) = check_xa_limits_block(
        model,
        catalog,
        &ts.xa_surface_lower_limit_species,
        &ts.xa_surface_lower_limit,
        false,
        "surface",
        "xa_surface_lower_limit",
        |m, j| surface_avg_x_slot(m, Some(j)),
    ) {
        return Ok(res);
    }
    if let Some(res) = check_xa_limits_block(
        model,
        catalog,
        &ts.xa_surface_upper_limit_species,
        &ts.xa_surface_upper_limit,
        true,
        "surface",
        "xa_surface_upper_limit",
        |m, j| surface_avg_x_slot(m, Some(j)),
    ) {
        return Ok(res);
    }
    let avg_dot_dq = |m: &StarModel, i: usize| -> f64 {
        (0..m.nz).map(|k| m.dq[k] * m.cell_xa(k)[i]).sum()
    };
    if let Some(res) = check_xa_limits_block(
        model,
        catalog,
        &ts.xa_average_lower_limit_species,
        &ts.xa_average_lower_limit,
        false,
        "average",
        "xa_average_lower_limit",
        avg_dot_dq,
    ) {
        return Ok(res);
    }
    if let Some(res) = check_xa_limits_block(
        model,
        catalog,
        &ts.xa_average_upper_limit_species,
        &ts.xa_average_upper_limit,
        true,
        "average",
        "xa_average_upper_limit",
        avg_dot_dq,
    ) {
        return Ok(res);
    }

    Ok(LimitCheckOutcome::KeepGoing)
}

fn match_stop_at_phase(model: &StarModel, phase: i32) -> Option<&'static str> {
    let c = &model.controls;
    let ts = &c.ts_stop;
    if ts.stop_at_phase_prems && phase == PHASE_PREMS {
        Some("phase_PreMS")
    } else if ts.stop_at_phase_zams && phase == PHASE_ZAMS {
        Some("phase_ZAMS")
    } else if ts.stop_at_phase_iams && phase == PHASE_IAMS {
        Some("phase_IAMS")
    } else if ts.stop_at_phase_tams && phase == PHASE_TAMS {
        Some("phase_TAMS")
    } else if ts.stop_at_phase_he_burn && phase == PHASE_HE_BURN {
        Some("phase_He_Burn")
    } else if ts.stop_at_phase_zacheb && phase == PHASE_ZACHEB {
        Some("phase_ZACHeB")
    } else if ts.stop_at_phase_tacheb && phase == PHASE_TACHEB {
        Some("phase_TACHeB")
    } else if c.stop_at_phase_tp_agb && phase == PHASE_TP_AGB {
        Some("phase_TP_AGB")
    } else if c.stop_at_phase_c_burn && phase == PHASE_C_BURN {
        Some("phase_C_Burn")
    } else if c.stop_at_phase_ne_burn && phase == PHASE_NE_BURN {
        Some("phase_Ne_Burn")
    } else if c.stop_at_phase_o_burn && phase == PHASE_O_BURN {
        Some("phase_O_Burn")
    } else if ts.stop_at_phase_si_burn && phase == PHASE_SI_BURN {
        Some("phase_Si_Burn")
    } else if ts.stop_at_phase_wdcs && phase == PHASE_WDCS {
        Some("phase_WDCS")
    } else {
        None
    }
}

/// Evaluates all MESA `26.4.1` stopping limits, target-hitting redos, velocity
/// retries, and evolutionary phase stops matching `do_check_limits` in
/// `star/private/do_one_utils.f90:565-1465`, updating
/// `model.aux.step_hist.phase_of_evolution`.
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if `peak_burn_vconv_div_cs` exceeds
/// `0.75 * peak_burn_vconv_div_cs_limit` (reproducing upstream `mesa_error` at
/// `do_one_utils.f90:738-745`, `#264`).
pub fn do_check_limits(
    model: &mut StarModel,
    ctx: Option<&StepReportContext>,
    doing_first_model_of_run: bool,
) -> Result<LimitCheckOutcome, StarError> {
    let owned_ctx;
    let report_ctx = if let Some(existing) = ctx {
        existing
    } else {
        owned_ctx = StepReportContext::compute(model);
        &owned_ctx
    };
    let phase =
        eval_phase_of_evolution(model, report_ctx, doing_first_model_of_run);
    model.aux.step_hist.phase_of_evolution = phase;
    eval_check_limits_core(model, report_ctx, phase)
}

/// Evaluates all built-in `&controls` termination conditions on `model`
/// (`star/private/do_one_utils.f90`). Returns `Some(termination_code)` if any
/// active limit requests termination.
#[must_use]
pub fn check_termination_limits(model: &StarModel) -> Option<String> {
    check_termination_limits_with_history(model, None)
}

/// Evaluates all built-in `&controls` termination conditions on `model`,
/// delegating to the canonical `do_check_limits` cascade.
#[must_use]
pub fn check_termination_limits_with_history(
    model: &StarModel,
    _cached_hr: Option<&HistoryRow>,
) -> Option<String> {
    if model.nz == 0 {
        return None;
    }
    let ctx = StepReportContext::compute(model);
    let phase = eval_phase_of_evolution(model, &ctx, false);
    match eval_check_limits_core(model, &ctx, phase) {
        Ok(LimitCheckOutcome::Terminate { code, .. }) => Some(code),
        _ => None,
    }
}
