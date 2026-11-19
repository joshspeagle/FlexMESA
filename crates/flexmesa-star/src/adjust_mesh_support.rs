// SPDX-License-Identifier: LGPL-3.0-or-later
//! Local mesh-spacing multiplier evaluation, smoothing, and pre-remesh grid
//! validity checks (`star/private/adjust_mesh_support.f90:1-457`).

use flexmesa_const::LN10;

use crate::controls::StarError;
use crate::controls_mix_wind::{MixWindControls, NUM_MESH_LOGX};
use crate::mesh_functions::{
    MeshFunctionData, get_net_iso_0based, set_mesh_function_data_with_controls,
};
use crate::model::StarModel;

/// Callback trait for `use_other_mesh_delta_coeff_factor`
/// (`star/other/other_mesh_delta_coeff_factor.f90` and
/// `star/private/adjust_mesh_support.f90:106-118`).
pub trait OtherMeshDeltaCoeffFactorHook: Send + Sync {
    /// Updates per-zone `mesh_delta_coeff_factor` (`len = nz`) in place.
    fn eval_mesh_delta_coeff_factor(
        &self,
        model: &StarModel,
        mesh_delta_coeff_factor: &mut [f64],
    ) -> Result<(), StarError>;
}

/// Cloneable reference-counted handle wrapping an [`OtherMeshDeltaCoeffFactorHook`].
#[derive(Clone)]
pub struct OtherMeshDeltaCoeffFactorHandle(
    pub std::sync::Arc<dyn OtherMeshDeltaCoeffFactorHook>,
);

impl std::fmt::Debug for OtherMeshDeltaCoeffFactorHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OtherMeshDeltaCoeffFactorHandle(..)")
    }
}

/// Evaluated mesh functions together with smoothed local `delta_gval_max`
/// (`get_gval_info` in `star/private/adjust_mesh_support.f90:34-295`).
#[derive(Debug, Clone)]
pub struct GvalInfo {
    /// Per-zone local multiplier `delta_gval_max(1:nz)` before global `delta_coeff` scaling.
    pub delta_gval_max: Vec<f64>,
    /// Smoothed mesh spacing function profiles and metadata (`gvals(1:nz, 1:num_gvals)`).
    pub fcns: MeshFunctionData,
}

#[inline]
pub(crate) fn fortran_maxloc_0based(slice: &[f64]) -> usize {
    let mut best_idx = 0_usize;
    let mut best_val = f64::NEG_INFINITY;
    for (idx, &val) in slice.iter().enumerate() {
        if idx == 0 || val > best_val {
            best_val = val;
            best_idx = idx;
        }
    }
    best_idx
}

#[inline]
pub(crate) fn fortran_minloc_0based(slice: &[f64]) -> usize {
    let mut best_idx = 0_usize;
    let mut best_val = f64::INFINITY;
    for (idx, &val) in slice.iter().enumerate() {
        if idx == 0 || val < best_val {
            best_val = val;
            best_idx = idx;
        }
    }
    best_idx
}

fn smooth_3pt_step(src: &[f64], dst: &mut [f64]) {
    let nz = src.len();
    if nz == 0 {
        return;
    }
    if nz == 1 {
        dst[0] = src[0];
        return;
    }
    dst[0] = (2.0 * src[0] + src[1]) / 3.0;
    for k in 1..(nz - 1) {
        dst[k] = (src[k - 1] + src[k] + src[k + 1]) / 3.0;
    }
    dst[nz - 1] = (2.0 * src[nz - 1] + src[nz - 2]) / 3.0;
}

/// Applies 3-pass boxcar smoothing to each mesh function column (`smooth_gvals`
/// in `star/private/adjust_mesh_support.f90:366-397`).
pub fn smooth_gvals(nz: usize, num_gvals: usize, gvals: &mut [f64]) {
    let mut src = vec![0.0_f64; nz];
    for i in 0..num_gvals {
        let col = &mut gvals[i * nz..(i + 1) * nz];
        src.copy_from_slice(col);
        smooth_3pt_step(&src, col);
        smooth_3pt_step(col, &mut src);
        smooth_3pt_step(&src, col);
    }
}

fn do_mesh_dlogx_dlogp_coef(
    model: &StarModel,
    mw: &MixWindControls,
    which: usize,
    delta_gval_max: &mut [f64],
) {
    let sp = mw.mesh_logx_species.get(which).map_or("", |s| s.trim());
    let Some(j0) = get_net_iso_0based(model, sp) else {
        return;
    };
    let logx_min_for_extra = mw
        .mesh_logx_min_for_extra
        .get(which)
        .copied()
        .unwrap_or(0.0);
    let dlogx_dlogp_extra =
        mw.mesh_dlogx_dlogp_extra.get(which).copied().unwrap_or(0.0);
    let dlogx_dlogp_full_on = mw
        .mesh_dlogx_dlogp_full_on
        .get(which)
        .copied()
        .unwrap_or(0.0);
    let dlogx_dlogp_full_off = mw
        .mesh_dlogx_dlogp_full_off
        .get(which)
        .copied()
        .unwrap_or(0.0);
    let x_min_for_extra =
        flexmesa_math::exp10((-50.0_f64).max(logx_min_for_extra));
    let nz = model.nz;
    let species = model.species;

    for k in 1..nz {
        let x_k = model.xa[k * species + j0];
        let x_km1 = model.xa[(k - 1) * species + j0];
        if x_k < x_min_for_extra || x_km1 < x_min_for_extra {
            continue;
        }
        let dlogx = flexmesa_math::log10(x_k / x_km1).abs();
        // Upstream `star/private/adjust_mesh_support.f90:199` multiplies
        // `abs(lnPeos(k) - lnPeos(k-1))` by `ln10` instead of dividing by `ln10`
        // (deferred bug #243); reproduced verbatim for MESA 26.4.1 fidelity.
        let dlogp = (1.0e-10_f64)
            .max((model.ln_peos[k] - model.ln_peos[k - 1]).abs() * LN10);
        let dlogx_dlogp = dlogx / dlogp;
        if dlogx_dlogp <= dlogx_dlogp_full_off {
            continue;
        }
        let coef = if dlogx_dlogp >= dlogx_dlogp_full_on {
            dlogx_dlogp_extra
        } else {
            1.0 - (1.0 - dlogx_dlogp_extra)
                * (dlogx_dlogp - dlogx_dlogp_full_off)
                / (dlogx_dlogp_full_on - dlogx_dlogp_full_off)
        };
        if coef < delta_gval_max[k] {
            delta_gval_max[k] = coef;
        }
        if coef < delta_gval_max[k - 1] {
            delta_gval_max[k - 1] = coef;
        }
        if k + 1 < nz && coef < delta_gval_max[k + 1] {
            delta_gval_max[k + 1] = coef;
        }
    }
}

fn do1_dlog_eps_dlogp_coef(
    model: &StarModel,
    dlog_eps_dlogp_extra: f64,
    cat: usize,
    eps_min_for_delta: f64,
    dlog_eps_dlogp_full_off: f64,
    dlog_eps_dlogp_full_on: f64,
    delta_gval_max: &mut [f64],
) -> Result<(), StarError> {
    if dlog_eps_dlogp_extra <= 0.0 || dlog_eps_dlogp_extra >= 1.0 {
        return Ok(());
    }
    let nz = model.nz;
    if model.aux.eps_nuc_categories.len() < nz {
        return Ok(());
    }
    let maxv = (0..nz)
        .map(|k| model.aux.eps_nuc_categories[k][cat])
        .fold(f64::NEG_INFINITY, f64::max);
    if maxv < eps_min_for_delta {
        return Ok(());
    }

    for k in 1..nz {
        let eps = model.aux.eps_nuc_categories[k][cat];
        if eps < eps_min_for_delta {
            continue;
        }
        let epsm1 = model.aux.eps_nuc_categories[k - 1][cat];
        if epsm1 < eps_min_for_delta {
            continue;
        }
        let maxv_k = model.aux.eps_nuc_categories[k]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if maxv_k != eps {
            continue;
        }
        let dlogp = (model.ln_peos[k] - model.ln_peos[k - 1]) / LN10;
        if dlogp.abs() < 1.0e-50 {
            continue;
        }
        let dlog_eps = flexmesa_math::log10(eps / epsm1).abs();
        if !dlog_eps.is_finite() {
            return Err(StarError::MicrophysicsError(format!(
                "non-finite dlog_eps in adjust_mesh_support for category {cat} at zone {k}"
            )));
        }
        let dlog_eps_dlogp = dlog_eps / dlogp;
        if dlog_eps_dlogp <= dlog_eps_dlogp_full_off {
            continue;
        }
        let extra = if dlog_eps_dlogp >= dlog_eps_dlogp_full_on {
            dlog_eps_dlogp_extra
        } else {
            1.0 - (1.0 - dlog_eps_dlogp_extra)
                * (dlog_eps_dlogp - dlog_eps_dlogp_full_off)
                / (dlog_eps_dlogp_full_on - dlog_eps_dlogp_full_off)
        };
        if extra < delta_gval_max[k] {
            delta_gval_max[k] = extra;
        }
        if extra < delta_gval_max[k - 1] {
            delta_gval_max[k - 1] = extra;
        }
        if k + 1 < nz && extra < delta_gval_max[k + 1] {
            delta_gval_max[k + 1] = extra;
        }
    }
    Ok(())
}

/// Evaluates `delta_gval_max` (`set_delta_gval_max` in
/// `star/private/adjust_mesh_support.f90:86-174`).
///
/// # Errors
/// Returns [`StarError`] if `use_other_mesh_delta_coeff_factor` is enabled
/// without a hook or if nuclear category ratios produce non-finite values.
pub fn set_delta_gval_max_with_controls(
    model: &StarModel,
    mw: &MixWindControls,
) -> Result<Vec<f64>, StarError> {
    let nz = model.nz;
    let mut delta_gval_max = vec![1.0_f64; nz];
    if nz == 0 {
        return Ok(delta_gval_max);
    }

    if mw.mesh_pgas_div_p_exponent != 0.0 {
        let p_exp = mw.mesh_pgas_div_p_exponent;
        for k in 0..nz {
            let beta = model.pgas[k] / model.peos[k];
            delta_gval_max[k] *= flexmesa_math::pow(beta, p_exp);
        }
    }

    if mw.use_other_mesh_delta_coeff_factor {
        let Some(ref hook) = model.aux.other_mesh_delta_coeff_factor else {
            return Err(StarError::InvalidInput(String::from(
                "use_other_mesh_delta_coeff_factor = .true. requires an other_mesh_delta_coeff_factor hook",
            )));
        };
        hook.0
            .eval_mesh_delta_coeff_factor(model, &mut delta_gval_max)?;
    }

    for j in 0..NUM_MESH_LOGX {
        let extra = mw.mesh_dlogx_dlogp_extra.get(j).copied().unwrap_or(0.0);
        if extra > 0.0 && extra < 1.0 {
            do_mesh_dlogx_dlogp_coef(model, mw, j, &mut delta_gval_max);
        }
    }

    let eps_min_for_delta =
        flexmesa_math::exp10(mw.mesh_dlog_eps_min_for_extra);
    let full_off = mw.mesh_dlog_eps_dlogp_full_off;
    let full_on = mw.mesh_dlog_eps_dlogp_full_on;

    let category_extras: [(f64, usize); 21] = [
        (mw.mesh_dlog_pp_dlogp_extra, 0),
        (mw.mesh_dlog_cno_dlogp_extra, 1),
        (mw.mesh_dlog_3alf_dlogp_extra, 2),
        (mw.mesh_dlog_burn_c_dlogp_extra, 3),
        (mw.mesh_dlog_burn_n_dlogp_extra, 4),
        (mw.mesh_dlog_burn_o_dlogp_extra, 5),
        (mw.mesh_dlog_burn_ne_dlogp_extra, 6),
        (mw.mesh_dlog_burn_na_dlogp_extra, 7),
        (mw.mesh_dlog_burn_mg_dlogp_extra, 8),
        (mw.mesh_dlog_burn_si_dlogp_extra, 9),
        (mw.mesh_dlog_burn_s_dlogp_extra, 10),
        (mw.mesh_dlog_burn_ar_dlogp_extra, 11),
        (mw.mesh_dlog_burn_ca_dlogp_extra, 12),
        (mw.mesh_dlog_burn_ti_dlogp_extra, 13),
        (mw.mesh_dlog_burn_cr_dlogp_extra, 14),
        (mw.mesh_dlog_burn_fe_dlogp_extra, 15),
        (mw.mesh_dlog_cc_dlogp_extra, 16),
        (mw.mesh_dlog_co_dlogp_extra, 17),
        (mw.mesh_dlog_oo_dlogp_extra, 18),
        (mw.mesh_dlog_pnhe4_dlogp_extra, 20),
        (mw.mesh_dlog_photo_dlogp_extra, 19),
    ];
    for (extra, cat) in category_extras {
        do1_dlog_eps_dlogp_coef(
            model,
            extra,
            cat,
            eps_min_for_delta,
            full_off,
            full_on,
            &mut delta_gval_max,
        )?;
    }
    do1_dlog_eps_dlogp_coef(
        model,
        mw.mesh_dlog_other_dlogp_extra,
        23,
        eps_min_for_delta,
        full_off,
        full_on,
        &mut delta_gval_max,
    )?;

    if mw.mesh_delta_coeff_factor_smooth_iters > 0 && nz >= 2 {
        let mut src = delta_gval_max.clone();
        for j in 1..=mw.mesh_delta_coeff_factor_smooth_iters {
            smooth_3pt_step(&src, &mut delta_gval_max);
            if j == 3 {
                break;
            }
            smooth_3pt_step(&delta_gval_max, &mut src);
        }
    }

    Ok(delta_gval_max)
}

/// Evaluates mesh spacing functions, local `delta_gval_max` multipliers, and
/// 3-pass `gvals` smoothing (`get_gval_info` in
/// `star/private/adjust_mesh_support.f90:34-295`).
///
/// # Errors
/// Returns [`StarError`] if function evaluation or multiplier setup fails.
pub fn get_gval_info_with_controls(
    model: &StarModel,
    mw: &MixWindControls,
    num_gvals: usize,
) -> Result<GvalInfo, StarError> {
    let nz = model.nz;
    let mut fcns = set_mesh_function_data_with_controls(model, mw, num_gvals)?;
    let delta_gval_max = set_delta_gval_max_with_controls(model, mw)?;
    smooth_gvals(nz, num_gvals, &mut fcns.vals);
    Ok(GvalInfo {
        delta_gval_max,
        fcns,
    })
}

/// Evaluates `get_gval_info` using `model.controls.mix_wind`.
///
/// # Errors
/// Returns [`StarError`] if function evaluation or multiplier setup fails.
pub fn get_gval_info(
    model: &StarModel,
    num_gvals: usize,
) -> Result<GvalInfo, StarError> {
    get_gval_info_with_controls(model, &model.controls.mix_wind, num_gvals)
}

/// Piecewise linear blend `blend_coef` (`adjust_mesh_support.f90:276-292`).
#[must_use]
pub fn blend_coef(coef_start: f64, alfa: f64) -> f64 {
    let beta = 2.0 * (alfa - 0.5);
    if beta < 0.0 {
        coef_start
    } else {
        coef_start * (1.0 - beta) + beta
    }
}

/// Enforces unimodality around `maxloc(src)` (`single_peak` in
/// `adjust_mesh_support.f90:298-318`).
pub fn single_peak(src: &mut [f64]) {
    let nz = src.len();
    if nz < 2 {
        return;
    }
    let kmax = fortran_maxloc_0based(src);
    let mut val = src[kmax];
    for item in src.iter_mut().skip(kmax + 1) {
        let prev = val;
        val = *item;
        if val > prev {
            val = prev;
        }
        *item = val;
    }
    val = src[kmax];
    for k in (0..kmax).rev() {
        let prev = val;
        val = src[k];
        if val > prev {
            val = prev;
        }
        src[k] = val;
    }
}

/// Enforces monotone non-decreasing order from surface to center
/// (`increasing_inward` in `adjust_mesh_support.f90:321-333`).
pub fn increasing_inward(src: &mut [f64]) {
    if src.len() < 2 {
        return;
    }
    let mut val = src[0];
    for item in src.iter_mut().skip(1) {
        let prev = val;
        val = *item;
        if val < prev {
            val = prev;
        }
        *item = val;
    }
}

/// Enforces monotone non-increasing order from center to surface
/// (`decreasing_outward` in `adjust_mesh_support.f90:336-348`).
pub fn decreasing_outward(src: &mut [f64]) {
    let nz = src.len();
    if nz < 2 {
        return;
    }
    let mut val = src[nz - 1];
    for k in (0..(nz - 1)).rev() {
        let prev = val;
        val = src[k];
        if val > prev {
            val = prev;
        }
        src[k] = val;
    }
}

/// Enforces monotone non-decreasing order from center to surface
/// (`increasing_outward` in `adjust_mesh_support.f90:351-363`).
pub fn increasing_outward(src: &mut [f64]) {
    let nz = src.len();
    if nz < 2 {
        return;
    }
    let mut val = src[nz - 1];
    for k in (0..(nz - 1)).rev() {
        let prev = val;
        val = src[k];
        if val < prev {
            val = prev;
        }
        src[k] = val;
    }
}

/// Validates strictly decreasing outer face $\ln R$ and positive $\Delta q$ at
/// entry to remeshing (`check_validity` in `adjust_mesh_support.f90:413-455`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if any zone has non-positive volume or mass.
pub fn check_validity(model: &StarModel) -> Result<(), StarError> {
    let nz = model.nz;
    for k in 0..nz.saturating_sub(1) {
        if model.ln_r[k] <= model.ln_r[k + 1] {
            return Err(StarError::InvalidInput(format!(
                "at start of remesh: negative cell volume for cell {} (lnR[k]={}, lnR[k+1]={})",
                k + 1,
                model.ln_r[k],
                model.ln_r[k + 1]
            )));
        }
        if model.dq[k] <= 0.0 {
            return Err(StarError::InvalidInput(format!(
                "at start of remesh: non-positive cell mass for cell {} (dq={})",
                k + 1,
                model.dq[k]
            )));
        }
    }
    Ok(())
}
