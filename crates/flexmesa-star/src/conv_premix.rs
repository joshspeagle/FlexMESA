// SPDX-License-Identifier: LGPL-3.0-or-later
//! Verbatim MESA `26.4.1` convective premixing (`star/private/conv_premix.f90`).
//!
//! Advances convective boundaries cell-by-cell prior to the Henyey solve,
//! tracking per-zone clocks via Castellani (1971) penetration velocities,
//! adaptive subcell refinement (`n_sub = 1..=1024`) to resolve interior
//! convective zone splitting cleanly, fixed-`(Pgas, T)` or fixed-`(rho, T)`
//! EOS updates, opacity and MLT re-evaluation with $\nabla_{L,\mathrm{comp}} = 0$
//! on interior faces plus the advancing face, and optional fuel-increase
//! avoidance (`conv_premix_avoid_increase`).

use std::fmt::Write as _;
use std::fs;

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{MixingType, M_SUN};

use crate::controls::StarError;
use crate::hydro_brunt_surf::do_brunt_b;
use crate::hydro_vars::set_mlt_vars_range;
use crate::mixing_micro::{
    eval_eos_kap_range_by_mode, CellRangeSnapshot, FaceMltSnapshot,
};
use crate::mixing_state::set_grads;
use crate::model::StarModel;

const N_SUB_INI: usize = 1;
const N_SUB_MAX: usize = 1024;

/// Threshold specific nuclear energy generation rate ($\mathrm{erg\,g^{-1}\,s^{-1}}$)
/// classifying a convective zone as nuclear burning (`EPS_THRESHOLD` in
/// `star/private/conv_premix.f90:1024`).
const EPS_THRESHOLD: f64 = 1.0;

/// 0-based indices into `eps_nuc_categories` matching `chem_def.f90:414-419`:
/// `ipp = 1`, `icno = 2`, `i3alf = 3`, `i_burn_c = 4`.
const CAT_PP: usize = 0;
const CAT_CNO: usize = 1;
const CAT_3ALF: usize = 2;
const CAT_BURN_C: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BurnType {
    None,
    H,
    He,
    Z,
}

#[derive(Debug, Clone)]
struct ZoneInfo {
    /// 0-based cell index of top boundary (`kc_t`, inclusive).
    kc_t: usize,
    /// 0-based cell index of bottom boundary (`kc_b`, inclusive).
    kc_b: usize,
    /// Convective velocity near top boundary (`mlt_vc[kc_t + 1]`).
    vc_t: f64,
    /// Convective velocity near bottom boundary (`mlt_vc[kc_b]`).
    vc_b: f64,
    /// Castellani (1971) penetration velocity at top boundary.
    vp_t: f64,
    /// Castellani (1971) penetration velocity at bottom boundary.
    vp_b: f64,
    /// Accumulated clock for top boundary advance.
    dt_t: f64,
    /// Accumulated clock for bottom boundary advance.
    dt_b: f64,
    /// Principal burning classification of the zone.
    burn_type: BurnType,
    /// Whether top boundary remains eligible to advance.
    sel_t: bool,
    /// Whether bottom boundary remains eligible to advance.
    sel_b: bool,
    /// Initial mass-weighted average abundances over `kc_t..=kc_b`.
    avg_xa: Vec<f64>,
    /// Initial mass-weighted average nuclear burning rates over `kc_t..=kc_b`.
    davg_xa_dt: Vec<f64>,
    /// 0-based species index whose abundance increase is forbidden, if any.
    avoid_inc_iso: Option<usize>,
}

impl ZoneInfo {
    fn new(kc_t: usize, kc_b: usize, vc_t: f64, vc_b: f64) -> Self {
        Self {
            kc_t,
            kc_b,
            vc_t,
            vc_b,
            vp_t: 0.0,
            vp_b: 0.0,
            dt_t: 0.0,
            dt_b: 0.0,
            burn_type: BurnType::None,
            sel_t: false,
            sel_b: false,
            avg_xa: Vec::new(),
            davg_xa_dt: Vec::new(),
            avoid_inc_iso: None,
        }
    }
}

struct SavedData {
    cell_snap: CellRangeSnapshot,
    face_snap: Option<FaceMltSnapshot>,
    gradl_comp_lo: usize,
    gradl_comp: Vec<f64>,
    fix_pgas_lo: usize,
    fix_pgas: Vec<bool>,
}

/// Runs verbatim MESA `26.4.1` convective premixing (`do_conv_premix` in
/// `star/private/conv_premix.f90:82-182`).
///
/// Returns the total number of cells absorbed across all convective boundaries.
///
/// # Errors
/// Returns [`StarError`] if zone-table validation, EOS/opacity/MLT updates, or
/// optional diagnostic snapshot I/O fails.
pub fn do_conv_premix(model: &mut StarModel) -> Result<usize, StarError> {
    let nz = model.nz;
    if nz < 2 || model.species == 0 {
        model.aux.need_to_setvars = true;
        return Ok(0);
    }

    let mut fix_pgas = vec![false; nz];
    let mut zi = init_zone_info(model)?;
    let mut total_advanced = 0_usize;
    let mut j_it = 0_usize;

    loop {
        j_it += 1;
        validate_zone_info(model, &zi)?;

        if !zi.iter().any(|z| z.sel_t || z.sel_b) {
            break;
        }

        // Select the eligible boundary with the shortest mixing timescale
        // (`conv_premix.f90:148-159`).
        let mut best_i = 0_usize;
        let mut best_t_bdy = true;
        let mut best_dt = f64::INFINITY;

        for (i, z) in zi.iter().enumerate() {
            let dr = model.r[z.kc_t + 1] - model.r[z.kc_b];
            let dt_t = if z.sel_t { dr / z.vc_t } else { f64::MAX };
            let dt_b = if z.sel_b { dr / z.vc_b } else { f64::MAX };
            let dt_min = dt_t.min(dt_b);
            if dt_min < best_dt {
                best_dt = dt_min;
                best_i = i;
                best_t_bdy = dt_t < dt_b;
            }
        }

        let mut i_bdy = best_i;
        total_advanced += advance_bdy(
            model,
            &mut fix_pgas,
            &mut zi,
            &mut i_bdy,
            best_t_bdy,
            j_it,
        )?;
    }

    // Match `star/private/conv_premix.f90:176`: downstream `set_vars_if_needed`
    // must unpack `xh` and refresh microphysics after premixing trial updates.
    model.aux.need_to_setvars = true;

    Ok(total_advanced)
}

fn advance_bdy(
    model: &mut StarModel,
    fix_pgas: &mut [bool],
    zi: &mut Vec<ZoneInfo>,
    i_bdy: &mut usize,
    t_bdy: bool,
    j_it: usize,
) -> Result<usize, StarError> {
    let nz = model.nz;
    let dump = model.controls.mix_wind.conv_premix_dump_snapshots;
    if dump {
        let filename = format!(
            "cpm-snap.{:05}.it_{j_it:05}.ad_{:05}.dat",
            model.model_number, 0
        );
        dump_snapshot(model, &filename)?;
    }

    let mut j_ad = 0_usize;
    let mut n_advanced = 0_usize;

    loop {
        j_ad += 1;

        if zi.is_empty() || *i_bdy >= zi.len() {
            break;
        }
        if t_bdy && zi[*i_bdy].kc_t == 0 {
            zi[*i_bdy].sel_t = false;
        } else if !t_bdy && zi[*i_bdy].kc_b == nz - 1 {
            zi[*i_bdy].sel_b = false;
        }

        if (t_bdy && !zi[*i_bdy].sel_t) || (!t_bdy && !zi[*i_bdy].sel_b) {
            break;
        }

        let accepted =
            advance_bdy_by_one_cell(model, fix_pgas, zi, i_bdy, t_bdy)?;
        if accepted {
            n_advanced += 1;
        }

        if dump && j_ad.is_multiple_of(50) {
            let filename = format!(
                "cpm-snap.{:05}.it_{j_it:05}.ad_{j_ad:05}.dat",
                model.model_number
            );
            dump_snapshot(model, &filename)?;
        }
    }

    if dump {
        let filename = format!(
            "cpm-snap.{:05}.it_{j_it:05}.ad_{:05}.dat",
            model.model_number,
            j_ad.saturating_sub(1)
        );
        dump_snapshot(model, &filename)?;
    }

    Ok(n_advanced)
}

#[allow(clippy::too_many_lines)]
fn advance_bdy_by_one_cell(
    model: &mut StarModel,
    fix_pgas: &mut [bool],
    zi: &mut Vec<ZoneInfo>,
    i_bdy: &mut usize,
    t_bdy: bool,
) -> Result<bool, StarError> {
    let nz = model.nz;
    let sp = model.species;

    // Clock increment for adding the adjacent cell (`conv_premix.f90:312-372`).
    let (dr, vp) = if t_bdy {
        let kc_t = zi[*i_bdy].kc_t;
        (model.r[kc_t] - model.r[kc_t + 1], zi[*i_bdy].vp_t)
    } else {
        let kc_b = zi[*i_bdy].kc_b;
        let dr = if kc_b < nz - 1 {
            model.r[kc_b] - model.r[kc_b + 1]
        } else {
            model.r[kc_b]
        };
        (dr, zi[*i_bdy].vp_b)
    };
    let delta_dt = dr / vp;

    let time_fac = model.controls.mix_wind.conv_premix_time_factor;
    let dt_limit = if time_fac > 0.0 {
        time_fac * model.dt
    } else {
        f64::MAX
    };

    if t_bdy {
        if zi[*i_bdy].dt_t + delta_dt > dt_limit {
            zi[*i_bdy].sel_t = false;
            return Ok(false);
        }
        zi[*i_bdy].dt_t += delta_dt;
    } else {
        if zi[*i_bdy].dt_b + delta_dt > dt_limit {
            zi[*i_bdy].sel_b = false;
            return Ok(false);
        }
        zi[*i_bdy].dt_b += delta_dt;
    }

    let kc_new = if t_bdy {
        zi[*i_bdy].kc_t - 1
    } else {
        zi[*i_bdy].kc_b + 1
    };

    let (save_t, save_b) = if t_bdy {
        (kc_new, zi[*i_bdy].kc_b)
    } else {
        (zi[*i_bdy].kc_t, kc_new)
    };
    let sd = save_model(model, fix_pgas, save_t, save_b);

    let mode_for_mix = model.controls.mix_wind.conv_premix_fix_pgas;

    let mut n_sub = N_SUB_INI;
    let mut kc_t;
    let mut kc_b;
    let mut kf_t;
    let mut kf_b;
    let mut has_split;
    let mut avg_xa = vec![0.0_f64; sp];

    'refine_loop: loop {
        kc_t = zi[*i_bdy].kc_t;
        kc_b = zi[*i_bdy].kc_b;
        kf_t = kc_t;
        kf_b = kc_b + 1;

        fix_pgas[kc_t..=kc_b].fill(mode_for_mix);

        // Zero `gradL_composition_term` on interior faces plus the advancing
        // face (`conv_premix.f90:422-426`).
        if t_bdy {
            model.aux.mixing.gradl_composition_term[kf_t..kf_b].fill(0.0);
        } else {
            model.aux.mixing.gradl_composition_term[(kf_t + 1)..=kf_b]
                .fill(0.0);
        }

        // Divide `kc_new` into `n_sub` virtual subcells with uniform initial
        // abundances (`conv_premix.f90:431-438`).
        let n_sub_cur = n_sub;
        let mut xa_sub = vec![0.0_f64; sp * n_sub_cur];
        for l in 0..sp {
            let x_init = model.xa[kc_new * sp + l];
            for s_idx in 0..n_sub_cur {
                xa_sub[l * n_sub_cur + s_idx] = x_init;
            }
        }
        let m_sub = model.dm[kc_new] / (n_sub_cur as f64);

        has_split = false;

        for kc_sub in 1..=n_sub_cur {
            let m_zone: f64 = model.dm[kc_t..=kc_b].iter().sum();
            let denom = m_zone + m_sub * (kc_sub as f64);
            for l in 0..sp {
                let mut s_zone = 0.0;
                for k in kc_t..=kc_b {
                    s_zone += model.dm[k] * model.xa[k * sp + l];
                }
                let mut s_sub = 0.0;
                for s_idx in 0..kc_sub {
                    s_sub += m_sub * xa_sub[l * n_sub_cur + s_idx];
                }
                avg_xa[l] = ((s_zone + s_sub) / denom).clamp(0.0, 1.0);
            }
            let sum_x: f64 = avg_xa.iter().sum();
            if sum_x > 0.0 {
                for v in &mut avg_xa {
                    *v /= sum_x;
                }
            }

            for k in kc_t..=kc_b {
                model.xa[k * sp..(k + 1) * sp].copy_from_slice(&avg_xa);
            }
            for l in 0..sp {
                let val = avg_xa[l];
                for s_idx in 0..kc_sub {
                    xa_sub[l * n_sub_cur + s_idx] = val;
                }
            }

            update_model(model, fix_pgas, kc_t, kc_b)?;

            // Count interior faces that became non-convective (`conv_premix.f90:474`).
            let n_nc = ((kf_t + 1)..kf_b)
                .filter(|&kf| {
                    model.aux.mixing.mlt_mixing_type[kf]
                        != MixingType::ConvectiveMixing
                })
                .count();

            if n_nc > 1 && n_sub_cur < N_SUB_MAX {
                n_sub = 2 * n_sub_cur;
                restore_model(model, fix_pgas, &sd);
                continue 'refine_loop;
            } else if n_nc > 0 {
                has_split = true;
                if t_bdy {
                    let mut kf_hit = kf_b - 1;
                    for kf in (kf_t + 1)..kf_b {
                        if model.aux.mixing.mlt_mixing_type[kf]
                            != MixingType::ConvectiveMixing
                        {
                            kf_hit = kf;
                            break;
                        }
                    }
                    kc_b = kf_hit - 1;
                    kf_b = kc_b + 1;
                } else {
                    let mut kf_hit = kf_t + 1;
                    for kf in ((kf_t + 1)..kf_b).rev() {
                        if model.aux.mixing.mlt_mixing_type[kf]
                            != MixingType::ConvectiveMixing
                        {
                            kf_hit = kf;
                            break;
                        }
                    }
                    kc_t = kf_hit;
                    kf_t = kc_t;
                }
            }
        }

        break 'refine_loop;
    }

    // Write final averaged composition into `kc_new` and update the extended
    // zone (`conv_premix.f90:541-557`).
    model.xa[kc_new * sp..(kc_new + 1) * sp].copy_from_slice(&avg_xa);
    fix_pgas[kc_new] = mode_for_mix;

    if t_bdy {
        kc_t = kc_new;
        kf_t = kc_t;
    } else {
        kc_b = kc_new;
        kf_b = kc_b + 1;
    }

    update_model(model, fix_pgas, kc_t, kc_b)?;

    // Check fuel abundance increase avoidance (`conv_premix.f90:562-594`).
    // Upstream divides by `s%dt` with no guard (`conv_premix.f90:566`); the
    // IEEE quotient is reproduced as is.
    if let Some(iso) = zi[*i_bdy].avoid_inc_iso {
        let davg = (avg_xa[iso] - zi[*i_bdy].avg_xa[iso]) / model.dt;
        if davg > (-zi[*i_bdy].davg_xa_dt[iso]).max(0.0) {
            restore_model(model, fix_pgas, &sd);
            if t_bdy {
                zi[*i_bdy].sel_t = false;
            } else {
                zi[*i_bdy].sel_b = false;
            }
            return Ok(false);
        }
    }

    // Check whether the face just inside the advancing boundary is convective
    // (`conv_premix.f90:600-632`).
    if t_bdy {
        if model.aux.mixing.mlt_mixing_type[kf_t + 1]
            != MixingType::ConvectiveMixing
        {
            restore_model(model, fix_pgas, &sd);
            zi[*i_bdy].sel_t = false;
            return Ok(false);
        }
    } else if model.aux.mixing.mlt_mixing_type[kf_b - 1]
        != MixingType::ConvectiveMixing
    {
        restore_model(model, fix_pgas, &sd);
        zi[*i_bdy].sel_b = false;
        return Ok(false);
    }

    update_zone_info(model, i_bdy, t_bdy, has_split, zi)?;
    Ok(true)
}

fn init_zone_info(model: &StarModel) -> Result<Vec<ZoneInfo>, StarError> {
    let nz = model.nz;
    let mut zi = create_zone_info(model, 0, nz - 1)?;

    for z in &mut zi {
        set_penetration_velocities(model, z)?;
        set_burn_data(model, z);
        set_abund_data(model, z)?;
        z.dt_t = 0.0;
        z.dt_b = 0.0;
        z.sel_t = z.kc_t != 0;
        z.sel_b = z.kc_b != nz - 1;
    }

    Ok(zi)
}

fn create_zone_info(
    model: &StarModel,
    kc_t: usize,
    kc_b: usize,
) -> Result<Vec<ZoneInfo>, StarError> {
    if kc_t >= kc_b || kc_b >= model.nz {
        return Err(StarError::InvalidInput(format!(
            "conv_premix: Invalid cell range in create_zone_info: kc_t={kc_t}, kc_b={kc_b}"
        )));
    }

    let mut zi = Vec::new();
    let mlt_mt = &model.aux.mixing.mlt_mixing_type;

    let mut in_conv = mlt_mt[kc_b] == MixingType::ConvectiveMixing;
    let mut cur_kc_b = kc_b;
    let mut cur_vc_b = if in_conv { model.mlt_vc[kc_b] } else { 0.0 };

    for kf in ((kc_t + 1)..kc_b).rev() {
        if in_conv && mlt_mt[kf] != MixingType::ConvectiveMixing {
            zi.push(ZoneInfo::new(
                kf,
                cur_kc_b,
                model.mlt_vc[kf + 1],
                cur_vc_b,
            ));
            in_conv = false;
        } else if !in_conv && mlt_mt[kf] == MixingType::ConvectiveMixing {
            cur_kc_b = kf;
            cur_vc_b = model.mlt_vc[kf];
            in_conv = true;
        }
    }

    if in_conv {
        zi.push(ZoneInfo::new(
            kc_t,
            cur_kc_b,
            model.mlt_vc[kc_t + 1],
            cur_vc_b,
        ));
    }

    Ok(zi)
}

fn update_zone_info(
    model: &StarModel,
    i_bdy: &mut usize,
    t_bdy: bool,
    has_split: bool,
    zi: &mut Vec<ZoneInfo>,
) -> Result<(), StarError> {
    if t_bdy {
        zi[*i_bdy].kc_t -= 1;
        let kc_t = zi[*i_bdy].kc_t;
        zi[*i_bdy].vc_t = model.mlt_vc[kc_t + 1];
    } else {
        zi[*i_bdy].kc_b += 1;
        let kc_b = zi[*i_bdy].kc_b;
        zi[*i_bdy].vc_b = model.mlt_vc[kc_b];
    }

    set_penetration_velocities(model, &mut zi[*i_bdy])?;

    // Truncate or delete an adjacent zone we encroached into (`conv_premix.f90:804-868`).
    if t_bdy && *i_bdy + 1 < zi.len() {
        if zi[*i_bdy].kc_t == zi[*i_bdy + 1].kc_b {
            if zi[*i_bdy + 1].kc_b - zi[*i_bdy + 1].kc_t > 1 {
                zi[*i_bdy + 1].kc_b -= 1;
                let nb = zi[*i_bdy + 1].kc_b;
                zi[*i_bdy + 1].vc_b = model.mlt_vc[nb];
            } else {
                zi.remove(*i_bdy + 1);
            }
        }
    } else if !t_bdy && *i_bdy > 0 && zi[*i_bdy].kc_b == zi[*i_bdy - 1].kc_t {
        if zi[*i_bdy - 1].kc_b - zi[*i_bdy - 1].kc_t > 1 {
            zi[*i_bdy - 1].kc_t += 1;
            let nt = zi[*i_bdy - 1].kc_t;
            zi[*i_bdy - 1].vc_t = model.mlt_vc[nt + 1];
        } else {
            zi.remove(*i_bdy - 1);
            *i_bdy -= 1;
        }
    }

    // If the zone split (or its trailing boundary retreated), rebuild sub-zones
    // (`conv_premix.f90:873-933`).
    if has_split {
        let old = zi[*i_bdy].clone();
        let mut zi_new = create_zone_info(model, old.kc_t, old.kc_b)?;

        for zn in &mut zi_new {
            if zn.kc_t == old.kc_t {
                zn.dt_t = old.dt_t;
                zn.sel_t = old.sel_t;
            } else {
                zn.dt_t = if t_bdy { old.dt_t } else { old.dt_b };
                zn.sel_t = false;
            }

            if zn.kc_b == old.kc_b {
                zn.dt_b = old.dt_b;
                zn.sel_b = old.sel_b;
            } else {
                zn.dt_b = if t_bdy { old.dt_t } else { old.dt_b };
                zn.sel_b = false;
            }

            set_penetration_velocities(model, zn)?;
            set_burn_data(model, zn);
            zn.avg_xa.clone_from(&old.avg_xa);
            zn.davg_xa_dt.clone_from(&old.davg_xa_dt);
        }

        zi.splice(*i_bdy..=*i_bdy, zi_new);
    }

    Ok(())
}

/// Computes Castellani (1971) convective boundary penetration velocities
/// (`set_penetration_velocities_` in `star/private/conv_premix.f90:948-1016`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] when the top boundary cell is the
/// innermost cell, where upstream reads `mlt_mixing_length(nz+1)` past the end
/// of the array (`conv_premix.f90:996`).
fn set_penetration_velocities(
    model: &StarModel,
    zi: &mut ZoneInfo,
) -> Result<(), StarError> {
    let nz = model.nz;
    let mixing_length = &model.aux.mixing.mlt_mixing_length;

    // Bottom boundary (`conv_premix.f90:966-981`).
    if zi.kc_b < nz - 1 {
        let mu_i = model.mu[zi.kc_b];
        let mu_e = model.mu[zi.kc_b + 1];
        let kf = zi.kc_b;
        let z_s = if mu_e - mu_i > f64::EPSILON {
            (zi.vc_b * zi.vc_b / (2.0 * model.grav[kf] * (1.0 - mu_i / mu_e)))
                .min(mixing_length[kf])
        } else {
            mixing_length[kf]
        };
        zi.vp_b = zi.vc_b * z_s / mixing_length[kf];
    } else {
        zi.vp_b = zi.vc_b;
    }

    // Top boundary (`conv_premix.f90:991-1006`).
    if zi.kc_t > 0 {
        let mu_i = model.mu[zi.kc_t];
        let mu_e = model.mu[zi.kc_t - 1];
        let kf = zi.kc_t + 1;
        if kf >= nz {
            return Err(StarError::InvalidInput(format!(
                "conv_premix: top boundary cell {} is the innermost cell; \
                 upstream reads mlt_mixing_length past the array here",
                zi.kc_t
            )));
        }
        let z_s = if mu_i - mu_e > f64::EPSILON {
            (zi.vc_t * zi.vc_t / (2.0 * model.grav[kf] * (1.0 - mu_e / mu_i)))
                .min(mixing_length[kf])
        } else {
            mixing_length[kf]
        };
        zi.vp_t = zi.vc_t * z_s / mixing_length[kf];
    } else {
        zi.vp_t = zi.vc_t;
    }
    Ok(())
}

fn set_burn_data(model: &StarModel, zi: &mut ZoneInfo) {
    let catalog = NuclideCatalog::builtin_ref();
    let iso_h1 = catalog
        .get_by_name("h1")
        .and_then(|e| model.net_iso.get(e.chem_id).copied().flatten());
    let iso_he4 = catalog
        .get_by_name("he4")
        .and_then(|e| model.net_iso.get(e.chem_id).copied().flatten());

    let mut eps_max = -f64::MAX;
    let mut eps_h_max = -f64::MAX;
    let mut eps_he_max = -f64::MAX;
    let has_cats = model.aux.eps_nuc_categories.len() == model.nz;

    for kc in zi.kc_t..=zi.kc_b {
        if model.eps_nuc[kc] > eps_max {
            eps_max = model.eps_nuc[kc];
            if has_cats {
                let cats = &model.aux.eps_nuc_categories[kc];
                eps_h_max = cats[CAT_PP] + cats[CAT_CNO];
                eps_he_max = cats[CAT_3ALF] + cats[CAT_BURN_C];
            } else {
                eps_h_max = 0.0;
                eps_he_max = 0.0;
            }
        }
    }

    zi.burn_type = if eps_max > EPS_THRESHOLD {
        if eps_h_max > 0.5 * eps_max {
            BurnType::H
        } else if eps_he_max > 0.5 * eps_max {
            BurnType::He
        } else {
            BurnType::Z
        }
    } else {
        BurnType::None
    };

    if model.controls.mix_wind.conv_premix_avoid_increase {
        zi.avoid_inc_iso = match zi.burn_type {
            BurnType::H => iso_h1,
            BurnType::He => iso_he4,
            BurnType::None | BurnType::Z => None,
        };
    } else {
        zi.avoid_inc_iso = None;
    }
}

fn set_abund_data(
    model: &StarModel,
    zi: &mut ZoneInfo,
) -> Result<(), StarError> {
    let sp = model.species;
    // Upstream reads `s% dxdt_nuc` unconditionally
    // (`star/private/conv_premix.f90:1141`); a model without the array
    // cannot supply the burning term.
    if model.dxdt_nuc.len() != model.nz * sp {
        return Err(StarError::InvalidInput(format!(
            "conv_premix set_abund_data: dxdt_nuc has length {}, expected nz*species = {}",
            model.dxdt_nuc.len(),
            model.nz * sp
        )));
    }
    zi.avg_xa.resize(sp, 0.0);
    zi.davg_xa_dt.resize(sp, 0.0);

    let m_zone: f64 = model.dm[zi.kc_t..=zi.kc_b].iter().sum();
    if m_zone <= 0.0 {
        zi.avg_xa.fill(0.0);
        zi.davg_xa_dt.fill(0.0);
        return Ok(());
    }
    for l in 0..sp {
        let mut s_x = 0.0;
        let mut s_dx = 0.0;
        for k in zi.kc_t..=zi.kc_b {
            let dm_k = model.dm[k];
            let idx = k * sp + l;
            s_x += dm_k * model.xa[idx];
            s_dx += dm_k * model.dxdt_nuc[idx];
        }
        zi.avg_xa[l] = (s_x / m_zone).clamp(0.0, 1.0);
        zi.davg_xa_dt[l] = s_dx / m_zone;
    }
    let sum_x: f64 = zi.avg_xa.iter().sum();
    if sum_x > 0.0 {
        for v in &mut zi.avg_xa {
            *v /= sum_x;
        }
    }
    Ok(())
}

fn validate_zone_info(
    model: &StarModel,
    zi: &[ZoneInfo],
) -> Result<(), StarError> {
    let nz = model.nz;
    let n_zi = zi.len();
    for (i, z) in zi.iter().enumerate() {
        if z.kc_t >= z.kc_b {
            return Err(StarError::ConvergenceFailure(format!(
                "conv_premix: Degenerate zone {i}: kc_t={}, kc_b={}",
                z.kc_t, z.kc_b
            )));
        }
        if z.kc_b >= nz {
            return Err(StarError::ConvergenceFailure(format!(
                "conv_premix: Zone {i} bottom outside grid: kc_b={}, nz={nz}",
                z.kc_b
            )));
        }
        if i > 0 && z.kc_b >= zi[i - 1].kc_t {
            return Err(StarError::ConvergenceFailure(format!(
                "conv_premix: Zone {i} bottom {} overlaps previous zone top {}",
                z.kc_b,
                zi[i - 1].kc_t
            )));
        }
        if i + 1 < n_zi && z.kc_t <= zi[i + 1].kc_b {
            return Err(StarError::ConvergenceFailure(format!(
                "conv_premix: Zone {i} top {} overlaps next zone bottom {}",
                z.kc_t,
                zi[i + 1].kc_b
            )));
        }
        if z.vc_t <= 0.0 || z.vc_b <= 0.0 {
            return Err(StarError::ConvergenceFailure(format!(
                "conv_premix: Non-positive convective velocity in zone {i}: vc_t={}, vc_b={}",
                z.vc_t, z.vc_b
            )));
        }
        for kf in (z.kc_t + 1)..=z.kc_b {
            if model.aux.mixing.mlt_mixing_type[kf]
                != MixingType::ConvectiveMixing
            {
                return Err(StarError::ConvergenceFailure(format!(
                    "conv_premix: Zone {i} contains non-convective face kf={kf}: {:?}",
                    model.aux.mixing.mlt_mixing_type[kf]
                )));
            }
        }
    }
    Ok(())
}

fn update_model(
    model: &mut StarModel,
    fix_pgas: &[bool],
    kc_t: usize,
    kc_b: usize,
) -> Result<(), StarError> {
    // Pass 1/2: FIXED_PT_MODE / FIXED_DT_MODE EOS + opacity in parallel
    // (`conv_premix.f90:1311-1335`).
    eval_eos_kap_range_by_mode(model, kc_t, kc_b, fix_pgas)?;

    // Update MLT on interior faces `(kc_t + 1)..=kc_b` (`conv_premix.f90:1342`).
    if kc_t < kc_b {
        set_mlt_vars_range(model, kc_t + 1, kc_b)?;
    }
    Ok(())
}

fn save_model(
    model: &StarModel,
    fix_pgas: &[bool],
    kc_t: usize,
    kc_b: usize,
) -> SavedData {
    let kf_lo = kc_t + 1;
    let kf_hi = kc_b;
    let face_snap = if kf_lo <= kf_hi {
        Some(FaceMltSnapshot::capture(model, kf_lo, kf_hi))
    } else {
        None
    };
    let gradl_comp = if kf_lo <= kf_hi {
        model.aux.mixing.gradl_composition_term[kf_lo..=kf_hi].to_vec()
    } else {
        Vec::new()
    };
    SavedData {
        cell_snap: CellRangeSnapshot::capture(model, kc_t, kc_b),
        face_snap,
        gradl_comp_lo: kf_lo,
        gradl_comp,
        fix_pgas_lo: kc_t,
        fix_pgas: fix_pgas[kc_t..=kc_b].to_vec(),
    }
}

fn restore_model(model: &mut StarModel, fix_pgas: &mut [bool], sd: &SavedData) {
    let len_u = sd.fix_pgas.len();
    fix_pgas[sd.fix_pgas_lo..(sd.fix_pgas_lo + len_u)]
        .copy_from_slice(&sd.fix_pgas);
    sd.cell_snap.restore(model);
    if !sd.gradl_comp.is_empty() {
        let len_g = sd.gradl_comp.len();
        model.aux.mixing.gradl_composition_term
            [sd.gradl_comp_lo..(sd.gradl_comp_lo + len_g)]
            .copy_from_slice(&sd.gradl_comp);
    }
    if let Some(ref fs) = sd.face_snap {
        fs.restore(model);
    }
}

fn dump_snapshot(
    model: &mut StarModel,
    filename: &str,
) -> Result<(), StarError> {
    let nz = model.nz;
    let saved_gradl = model.aux.mixing.gradl_composition_term.clone();
    do_brunt_b(model)?;
    set_grads(model);

    let catalog = NuclideCatalog::builtin_ref();
    let iso_h1 = catalog
        .get_by_name("h1")
        .and_then(|e| model.net_iso.get(e.chem_id).copied().flatten());
    let iso_he4 = catalog
        .get_by_name("he4")
        .and_then(|e| model.net_iso.get(e.chem_id).copied().flatten());

    let mut out = String::new();
    let _ = writeln!(out, "1 2");
    let _ = writeln!(out, "star_age model_number");
    let _ = writeln!(out, "{:24.16e} {}", model.star_age, model.model_number);
    let _ = writeln!(out);
    let _ = writeln!(out, "1 2 3 4 5 6 7 8");
    let _ = writeln!(
        out,
        "k mass gradr grada gradL_composition_term mixing_type x y"
    );
    let sp = model.species;
    for k in 0..nz {
        let xh = iso_h1.map_or(0.0, |i| model.xa[k * sp + i]);
        let yhe = iso_he4.map_or(0.0, |i| model.xa[k * sp + i]);
        let _ = writeln!(
            out,
            "{} {:24.16e} {:24.16e} {:24.16e} {:24.16e} {} {:24.16e} {:24.16e}",
            k + 1,
            model.m[k] / M_SUN,
            model.gradr[k],
            model.aux.mixing.grada_face[k],
            model.aux.mixing.gradl_composition_term[k],
            model.aux.mixing.mlt_mixing_type[k] as i32,
            xh,
            yhe,
        );
    }

    model.aux.mixing.gradl_composition_term = saved_gradl;
    fs::write(filename, out).map_err(|e| {
        StarError::IoError(format!(
            "conv_premix failed writing snapshot {filename}: {e}"
        ))
    })
}
