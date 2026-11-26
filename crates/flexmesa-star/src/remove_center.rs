// SPDX-License-Identifier: GPL-3.0-or-later
//! Core excision (`do_remove_center`, `do_remove_inner_fraction_q`, and
//! `do_remove_center_by_entropy`), porting
//! `star/private/remove_shells.f90:452-470` and
//! `star/private/remove_shells.f90:609-698`.

use flexmesa_const::{AVO, KERG};
use flexmesa_math::exp;
use flexmesa_utils::is_bad;

use crate::controls::StarError;
use crate::inlist_runner::StarExtras;
use crate::model::StarModel;
use crate::star_job_init::finish_load_model;

/// Removes inner cells `k_cut..model.nz` (keeping outer cells `0..k_cut`),
/// updating inner boundary conditions and calling `finish_load_model`.
///
/// Mirrors `do_remove_center` (`star/private/remove_shells.f90:638-698`) and
/// `star/private/star_utils.f90:635-752`. Upstream checks
/// `s% do_normalize_dqs_as_part_of_set_qs` before calling `normalize_dqs`;
/// that `&controls` flag is not yet exposed on `OutputControls` and defaults
/// to `.false.`, so `normalize_dqs` runs unconditionally before `set_qs`.
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if `new_xmstar <= 0`, if the active
/// velocity array has a non-finite boundary value at `k_cut`, if zone arrays
/// are shorter than `model.nz`, or if [`StarModel::normalize_dqs`] or
/// `finish_load_model` fails.
pub fn do_remove_center(
    model: &mut StarModel,
    k_cut: usize,
    extras: &mut dyn StarExtras,
) -> Result<(), StarError> {
    if k_cut == 0 || k_cut >= model.nz {
        return Ok(());
    }
    if model.m.len() < model.nz
        || model.r.len() < model.nz
        || model.lum.len() < model.nz
        || model.dm.len() < model.nz
        || model.dq.len() < model.nz
    {
        return Err(StarError::InvalidInput(String::from(
            "do_remove_center: model mesh arrays are shorter than nz",
        )));
    }

    model.m_center = model.m[k_cut];
    let new_xmstar = model.m[0] - model.m_center;
    if !new_xmstar.is_finite() || new_xmstar <= 0.0 {
        return Err(StarError::InvalidInput(format!(
            "do_remove_center: non-positive new_xmstar = {new_xmstar} at k_cut = {k_cut}"
        )));
    }
    model.xmstar = new_xmstar;
    model.r_center = model.r[k_cut];
    if model.job.remove_center_adjust_l_center {
        model.l_center = model.lum[k_cut];
    }

    let mut v_center = if model.controls.u_flag {
        let u_k = *model.u.get(k_cut).ok_or_else(|| {
            StarError::InvalidInput(format!(
                "do_remove_center: u length {} <= k_cut {k_cut}",
                model.u.len()
            ))
        })?;
        if is_bad(u_k) {
            return Err(StarError::InvalidInput(format!(
                "do_remove_center: bad u({k_cut}) = {u_k} (star/private/remove_shells.f90:662-666)"
            )));
        }
        u_k
    } else if model.controls.v_flag {
        let v_k = *model.v.get(k_cut).ok_or_else(|| {
            StarError::InvalidInput(format!(
                "do_remove_center: v length {} <= k_cut {k_cut}",
                model.v.len()
            ))
        })?;
        if is_bad(v_k) {
            return Err(StarError::InvalidInput(format!(
                "do_remove_center: bad v({k_cut}) = {v_k} (star/private/remove_shells.f90:668-672)"
            )));
        }
        v_k
    } else {
        0.0
    };
    if model.job.remove_center_set_zero_v_center {
        v_center = 0.0;
    }
    model.v_center = v_center;

    for kk in 0..k_cut {
        model.dq[kk] = model.dm[kk] / new_xmstar;
    }

    model.resize_zones(k_cut);
    if model.ye_start.len() > k_cut {
        model.ye_start.truncate(k_cut);
    }
    if model.eps_nuc_start.len() > k_cut {
        model.eps_nuc_start.truncate(k_cut);
    }
    if model.lum_by_category_start.len() > k_cut {
        model.lum_by_category_start.truncate(k_cut);
    }
    model.aux.need_to_setvars = true;
    model.generations = 1;

    model.normalize_dqs()?;
    model.set_qs();
    finish_load_model(model, extras)?;
    Ok(())
}

/// Removes cells inward from the first zone with `q(k) <= q`, mirroring
/// `do_remove_inner_fraction_q` (`star/private/remove_shells.f90:609-635`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if `q` is outside `[0.0, 1.0]` or
/// non-finite, if `model.q.len() < model.nz`, or if [`do_remove_center`]
/// fails.
pub fn do_remove_inner_fraction_q(
    model: &mut StarModel,
    q: f64,
    extras: &mut dyn StarExtras,
) -> Result<(), StarError> {
    if !q.is_finite() || !(0.0..=1.0).contains(&q) {
        return Err(StarError::InvalidInput(format!(
            "do_remove_inner_fraction_q: q must be in [0, 1], got {q} (star/private/remove_shells.f90:609-635)"
        )));
    }
    if model.q.len() < model.nz {
        return Err(StarError::InvalidInput(String::from(
            "do_remove_inner_fraction_q: model.q shorter than nz",
        )));
    }
    let Some(k_cut) = (0..model.nz).find(|&k| model.q[k] <= q) else {
        return Ok(());
    };
    do_remove_center(model, k_cut, extras)
}

/// Scans outward-to-inward (`k = 0..model.nz`) for the first zone whose
/// specific entropy `exp(ln_s[k]) / (AVO * KERG)` (in
/// $k_{\mathrm{B}}/\mathrm{baryon}$, `star/private/report.f90:278`) satisfies
/// `<= entropy_cut`, then removes the interior via
/// [`do_remove_inner_fraction_q`].
///
/// Mirrors `do_remove_center_by_entropy`
/// (`star/private/remove_shells.f90:452-470`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if `entropy_cut` is non-finite, if
/// `model.ln_s` or `model.q` is shorter than `model.nz`, if any scanned `ln_s`
/// yields a non-finite entropy, or if no zone satisfies `s_kb <= entropy_cut`
/// (upstream `ierr = -1`).
pub fn do_remove_center_by_entropy(
    model: &mut StarModel,
    entropy_cut: f64,
    extras: &mut dyn StarExtras,
) -> Result<(), StarError> {
    if !entropy_cut.is_finite() {
        return Err(StarError::InvalidInput(format!(
            "do_remove_center_by_entropy: non-finite entropy_cut = {entropy_cut}"
        )));
    }
    if model.ln_s.len() < model.nz || model.q.len() < model.nz {
        return Err(StarError::InvalidInput(String::from(
            "do_remove_center_by_entropy: model arrays shorter than nz",
        )));
    }
    let avo_kerg = AVO * KERG;
    for k in 0..model.nz {
        let s_kb = exp(model.ln_s[k]) / avo_kerg;
        if !s_kb.is_finite() {
            return Err(StarError::InvalidInput(format!(
                "do_remove_center_by_entropy: non-finite entropy at zone {k}"
            )));
        }
        if s_kb <= entropy_cut {
            return do_remove_inner_fraction_q(model, model.q[k], extras);
        }
    }
    Err(StarError::InvalidInput(format!(
        "do_remove_center_by_entropy: no zone satisfies entropy <= {entropy_cut} k_B/baryon (star/private/remove_shells.f90:452-470)"
    )))
}
