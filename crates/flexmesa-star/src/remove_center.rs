// SPDX-License-Identifier: GPL-3.0-or-later
//! Core excision (`do_remove_center`, `do_remove_inner_fraction_q`,
//! `do_remove_center_by_entropy`, and `do_remove_fallback`), porting
//! `star/private/remove_shells.f90:268-367`, `:452-470`, and `:609-698`.

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

/// Removes the inner fallback region, mirroring `do_remove_fallback`
/// (`star/private/remove_shells.f90:268-367`), called before each step when
/// `remove_fallback_at_each_step` (`star/job/run_star_support.f90:596-599`).
///
/// The velocity used is `u` when `u_flag`, else `v` when `v_flag`; with
/// neither, nothing is removed (`remove_shells.f90:285-291`). The outermost
/// removed cell `k0` (0-based here, so `nz - 1` is the innermost cell) is
/// chosen by one of two criteria:
///
/// * `fallback_check_total_energy` (`remove_shells.f90:297-347`): sum the
///   cell energy $e_k\,\Delta m_k + \tfrac12 v_k^2 \Delta m_k -
///   G_k (m_k - \tfrac12\Delta m_k)\Delta m_k / r_{C,k}$ outward from the
///   center, with $r_{C,k} = (r_k + r_{k+1})/2$ (`R_center` below the
///   innermost cell). If the running sum never goes negative there is no bound
///   inner region and nothing is removed. Otherwise continue outward from the
///   cell where it went negative until a cell whose own energy is positive,
///   and remove everything below that cell.
/// * otherwise (`remove_shells.f90:349-356`): if the innermost cell's infall
///   speed `-v` exceeds `remove_fallback_speed_limit * csound`, remove the
///   contiguous inner run of cells with `-v >= remove_fallback_speed_limit *
///   csound`.
///
/// Cells `k0..nz` are then removed through [`do_remove_inner_fraction_q`] at
/// `q(k0)` (`remove_shells.f90:362`), and `v_center` is zeroed when
/// `remove_center_set_zero_v_center` (`remove_shells.f90:364`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if a zone array is shorter than
/// `model.nz`, if a cell energy or the running energy sum is not finite
/// (upstream `mesa_error`, `remove_shells.f90:313-326`), or if
/// [`do_remove_inner_fraction_q`] fails.
pub fn do_remove_fallback(
    model: &mut StarModel,
    extras: &mut dyn StarExtras,
) -> Result<(), StarError> {
    // `remove_shells.f90:285-291`.
    let use_u = if model.controls.u_flag {
        true
    } else if model.controls.v_flag {
        false
    } else {
        return Ok(());
    };
    let nz = model.nz;
    if nz == 0 {
        return Ok(());
    }
    let vel_len = if use_u { model.u.len() } else { model.v.len() };
    if vel_len < nz
        || model.q.len() < nz
        || model.m.len() < nz
        || model.dm.len() < nz
        || model.r.len() < nz
        || (model.job.fallback_check_total_energy
            && (model.energy.len() < nz || model.cgrav.len() < nz))
        || (!model.job.fallback_check_total_energy && model.csound.len() < nz)
    {
        return Err(StarError::InvalidInput(String::from(
            "do_remove_fallback: model arrays are shorter than nz (star/private/remove_shells.f90:268-367)",
        )));
    }
    let vel: &[f64] = if use_u { &model.u } else { &model.v };

    // Fortran `k0 = nz` (`remove_shells.f90:296`); 0-based innermost cell.
    let mut k0 = nz - 1;
    if model.job.fallback_check_total_energy {
        // Cell total energy `ie + ke + pe` (`remove_shells.f90:302-312`).
        let cell_energy = |k: usize, r_l: f64| -> (f64, f64, f64) {
            let ie = model.energy[k] * model.dm[k];
            let ke = 0.5 * vel[k] * vel[k] * model.dm[k];
            let r_c = 0.5 * (model.r[k] + r_l);
            let m_cntr = model.m[k] - 0.5 * model.dm[k];
            let pe = -model.cgrav[k] * m_cntr * model.dm[k] / r_c;
            (ie, ke, pe)
        };
        let mut sum_total_energy = 0.0_f64;
        for k in (0..nz).rev() {
            let r_l = if k == nz - 1 {
                model.r_center
            } else {
                model.r[k + 1]
            };
            let (ie, ke, pe) = cell_energy(k, r_l);
            if is_bad(ie + ke + pe) {
                return Err(StarError::InvalidInput(format!(
                    "do_remove_fallback: bad cell energy at k = {} (ie = {ie}, ke = {ke}, pe = {pe}) (star/private/remove_shells.f90:313-318)",
                    k + 1
                )));
            }
            sum_total_energy += ie + ke + pe;
            if is_bad(sum_total_energy) {
                return Err(StarError::InvalidInput(format!(
                    "do_remove_fallback: bad sum_total_energy = {sum_total_energy} at k = {} (star/private/remove_shells.f90:320-326)",
                    k + 1
                )));
            }
            if sum_total_energy < 0.0 {
                break;
            }
            k0 = k;
        }
        // `remove_shells.f90:330-333`: no bound inner region.
        if sum_total_energy >= 0.0 {
            return Ok(());
        }
        // `do k = k0-1, 1, -1` (`remove_shells.f90:334-346`): `rL = r(k+1)`.
        for k in (0..k0).rev() {
            let (ie, ke, pe) = cell_energy(k, model.r[k + 1]);
            if ie + ke + pe > 0.0 {
                k0 = k + 1;
                break;
            }
        }
    } else {
        // `remove_shells.f90:349-356`.
        let speed_limit = model.job.remove_fallback_speed_limit;
        if -vel[nz - 1] <= speed_limit * model.csound[nz - 1] {
            return Ok(());
        }
        for k in (0..nz - 1).rev() {
            k0 = k + 1;
            if -vel[k] < speed_limit * model.csound[k] {
                break;
            }
        }
    }

    // Remove cells k0..nz (`remove_shells.f90:362`).
    let q_k0 = model.q[k0];
    do_remove_inner_fraction_q(model, q_k0, extras)?;

    // `remove_shells.f90:364`.
    if model.job.remove_center_set_zero_v_center {
        model.v_center = 0.0;
    }
    Ok(())
}
