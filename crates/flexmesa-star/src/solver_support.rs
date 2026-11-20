// SPDX-License-Identifier: LGPL-3.0-or-later
//! Henyey solver domain enforcement (`Bdomain`), trial-state variable
//! synchronization (`set_vars_for_solver`, `check1_chem`, `edit_lnR`),
//! minimum-iteration / accretion surface-entropy revision gating
//! (`force_another_iteration`), and post-convergence sanity checks
//! (`check_after_converge`, `RTI_check_after_converge`) corresponding to
//! `star/private/solver_support.f90`, `star/private/star_utils.f90`, and
//! `star/private/struct_burn_mix.f90` at MESA `26.4.1`.

use flexmesa_const::{FOUR_THIRDS_PI, LN10};

use crate::controls::StarError;
use crate::model::StarModel;

/// Evaluates the active negative mass-fraction hard limit interpolated by
/// central $\log_{10} T_c$ (`current_min_xa_hard_limit` in
/// `star/private/star_utils.f90:3162-3176`).
#[must_use]
pub fn current_min_xa_hard_limit(model: &StarModel) -> f64 {
    let c = &model.controls;
    if model.nz == 0 {
        return c.min_xa_hard_limit;
    }
    let log_tc = model.ln_t[model.nz - 1] / LN10;
    if log_tc <= c.logt_max_for_min_xa_hard_limit {
        c.min_xa_hard_limit
    } else if log_tc >= c.logt_min_for_min_xa_hard_limit_for_hight {
        c.min_xa_hard_limit_for_hight
    } else {
        let alfa = (log_tc - c.logt_max_for_min_xa_hard_limit)
            / (c.logt_min_for_min_xa_hard_limit_for_hight
                - c.logt_max_for_min_xa_hard_limit);
        alfa * c.min_xa_hard_limit_for_hight
            + (1.0 - alfa) * c.min_xa_hard_limit
    }
}

/// Evaluates the active mass-fraction sum deviation hard limit interpolated by
/// central $\log_{10} T_c$ (`current_sum_xa_hard_limit` in
/// `star/private/star_utils.f90:3179-3195`).
#[must_use]
pub fn current_sum_xa_hard_limit(model: &StarModel) -> f64 {
    let c = &model.controls;
    if model.nz == 0 {
        return c.sum_xa_hard_limit;
    }
    let log_tc = model.ln_t[model.nz - 1] / LN10;
    if log_tc <= c.logt_max_for_sum_xa_hard_limit {
        c.sum_xa_hard_limit
    } else if log_tc >= c.logt_min_for_sum_xa_hard_limit_for_hight {
        c.sum_xa_hard_limit_for_hight
    } else {
        let alfa = (log_tc - c.logt_max_for_sum_xa_hard_limit)
            / (c.logt_min_for_sum_xa_hard_limit_for_hight
                - c.logt_max_for_sum_xa_hard_limit);
        alfa * c.sum_xa_hard_limit_for_hight
            + (1.0 - alfa) * c.sum_xa_hard_limit
    }
}

fn clip_so_non_negative(
    nz: usize,
    nvar: usize,
    nvar_hydro: usize,
    ivar: usize,
    min_val: f64,
    xh_start: &[f64],
    solver_dx: &[f64],
    x_scale: &[f64],
    soln: &mut [f64],
    correction_factor: f64,
) {
    if correction_factor == 0.0 {
        return;
    }
    for k in 0..nz {
        let s_idx = k * nvar + ivar;
        let h_idx = k * nvar_hydro + ivar;
        let dval = soln[s_idx] * x_scale[s_idx] * correction_factor;
        let old_val = xh_start[h_idx] + solver_dx[s_idx];
        let new_val = old_val + dval;
        if dval >= 0.0 || new_val >= 0.0 {
            continue;
        }
        let dval_clipped = min_val - old_val;
        let denom = x_scale[s_idx] * correction_factor;
        if denom != 0.0 {
            soln[s_idx] = dval_clipped / denom;
        }
    }
}

/// Enforces physical domain bounds on the scaled Newton step `soln` and global
/// `correction_factor` (`Bdomain` in `star/private/solver_support.f90:623-730`).
pub fn apply_bdomain(
    model: &StarModel,
    nvar: usize,
    xh_start: &[f64],
    solver_dx: &[f64],
    x_scale: &[f64],
    soln: &mut [f64],
    correction_factor: &mut f64,
) {
    let nz = model.nz;
    if nz == 0 || nvar == 0 {
        return;
    }
    let layout = &model.layout;
    let nvar_hydro = layout.nvar_hydro.min(nvar);
    let c = &model.controls;

    if c.rsp2_flag {
        if let Some(iw) = layout.i_w.filter(|&i| i < nvar) {
            clip_so_non_negative(
                nz,
                nvar,
                layout.nvar_hydro,
                iw,
                0.0,
                xh_start,
                solver_dx,
                x_scale,
                soln,
                *correction_factor,
            );
        }
    }

    if c.rti_flag {
        if let Some(ia) = layout.i_alpha_rti.filter(|&i| i < nvar) {
            clip_so_non_negative(
                nz,
                nvar,
                layout.nvar_hydro,
                ia,
                0.0,
                xh_start,
                solver_dx,
                x_scale,
                soln,
                *correction_factor,
            );
        }
    }

    if layout.i_lum < nvar && c.scale_max_correction_for_negative_surf_lum {
        let ilum = layout.i_lum;
        let dlum_surf = soln[ilum] * x_scale[ilum];
        let old_lum_surf = xh_start[ilum] + solver_dx[ilum];
        let new_lum_surf = old_lum_surf + dlum_surf;
        if new_lum_surf < 0.0 && old_lum_surf > 0.0 {
            *correction_factor = (*correction_factor).min(
                -c.max_frac_for_negative_surf_lum * old_lum_surf / dlum_surf,
            );
        }
    }

    let mut min_alpha = 1.0_f64;
    if !c.do_solver_damping_for_neg_xa || !(c.do_mix || c.do_burn) {
        if min_alpha < 1.0 {
            min_alpha = min_alpha.max(c.corr_coeff_limit);
            *correction_factor *= min_alpha;
        }
        return;
    }

    if nvar > nvar_hydro && model.species > 0 {
        let n_chem = (nvar - nvar_hydro).min(model.species);
        let min_xa_hard_limit = current_min_xa_hard_limit(model);
        let eps = -0.5 * min_xa_hard_limit;
        for k in 0..nz {
            let c_base = k * model.species;
            let s_base = k * nvar;
            for j in 0..n_chem {
                let ivar = nvar_hydro + j;
                let old_xa = model
                    .xa_start
                    .get(c_base + j)
                    .copied()
                    .unwrap_or_else(|| model.xa[c_base + j])
                    + solver_dx[s_base + ivar];
                if old_xa <= 1.0e-90 {
                    continue;
                }
                let dxa = soln[s_base + ivar]
                    * x_scale[s_base + ivar]
                    * (*correction_factor);
                let new_xa = old_xa + dxa;
                if new_xa >= 0.0 {
                    continue;
                }
                let alpha = -(old_xa + eps) / dxa;
                if alpha < min_alpha {
                    min_alpha = alpha;
                }
            }
        }
    }

    min_alpha = min_alpha.max(c.corr_coeff_limit);
    *correction_factor *= min_alpha;
}

/// Checks whether another Newton iteration must be forced after tolerance
/// convergence (`force_another_iteration` in
/// `star/private/solver_support.f90:772-796`).
pub fn force_another_iteration(
    model: &mut StarModel,
    iter: usize,
    itermin: usize,
) -> bool {
    if iter < itermin {
        return true;
    }
    if model.k_below_just_added > 1
        && model.num_surf_revisions < model.controls.max_num_surf_revisions
    {
        let lns_0 = model.ln_s.first().copied().unwrap_or(0.0);
        if (lns_0 - model.surf_lns).abs()
            > model.controls.max_abs_rel_change_surf_lns
                * lns_0.max(model.surf_lns)
        {
            model.surf_lns = lns_0;
            model.num_surf_revisions += 1;
            model.used_extra_iter_in_solver_for_accretion = true;
            return true;
        }
    }
    false
}

/// Synchronizes `model` structure and composition arrays from
/// `xh_start + solver_dx` and `xa_start + solver_dx`, enforces `hydro_mtx_*`
/// and `check1_chem` hard limits, projects outer face radii via `edit_lnR`, and
/// updates `grav` and `rmid` (`set_vars_for_solver`, `check1_chem`, and
/// `edit_lnR` in `star/private/solver_support.f90:809-1534`).
///
/// # Errors
/// Returns [`StarError::ConvergenceFailure`] if any state variable is
/// non-finite or violates `hydro_mtx_*`, `min_xa_hard_limit`, or
/// `sum_xa_hard_limit`.
pub fn set_vars_for_solver(
    model: &mut StarModel,
    include_chem: bool,
    xh_start: &[f64],
    solver_dx: &mut [f64],
) -> Result<(), StarError> {
    let nz = model.nz;
    if nz == 0 {
        return Ok(());
    }
    let nvar_hydro = model.layout.nvar_hydro;
    let nvar = model.layout.nvar(include_chem);
    let min_xa_hard_limit = current_min_xa_hard_limit(model);
    let sum_xa_hard_limit = current_sum_xa_hard_limit(model);

    if include_chem && model.species > 0 {
        let species = model.species;
        model.xa_sub_xa_start.resize(nz * species, 0.0);
        for k in 0..nz {
            let s_base = k * nvar + nvar_hydro;
            let c_base = k * species;
            for j in 0..species {
                let dx_kj = solver_dx[s_base + j];
                model.xa_sub_xa_start[c_base + j] = dx_kj;
                let xs = model
                    .xa_start
                    .get(c_base + j)
                    .copied()
                    .unwrap_or(model.xa[c_base + j]);
                model.xa[c_base + j] = xs + dx_kj;
            }
        }
    }

    model.xh.resize(nz * nvar_hydro, 0.0);
    let i_lnd = model.layout.i_lnd;
    let i_lnt = model.layout.i_lnt;
    let i_lnr = model.layout.i_lnr;
    let i_lum = model.layout.i_lum;
    let i_v = model.layout.i_v;
    let i_u = model.layout.i_u;
    let i_w = model.layout.i_w;
    let i_hp = model.layout.i_hp;
    let i_alpha_rti = model.layout.i_alpha_rti;

    for k in 0..nz {
        let h_base = k * nvar_hydro;
        let s_base = k * nvar;
        for j in 0..nvar_hydro {
            model.xh[h_base + j] = xh_start[h_base + j] + solver_dx[s_base + j];
        }

        // `lnT` update and `hydro_mtx_*_logT` domain checks (`solver_support.f90:1138-1187`).
        let lnt_k = model.xh[h_base + i_lnt];
        let lnt_start_k = model.ln_t_start.get(k).copied().unwrap_or(lnt_k);
        if !lnt_k.is_finite() {
            return Err(StarError::ConvergenceFailure(format!(
                "bad num for lnT at zone {k}"
            )));
        }
        // Upstream MESA `26.4.1` `set_vars_for_solver` (`solver_support.f90:1145-1146,
        // 1157-1158, 1339-1340, 1351-1352`, `#252` under `#169`): compares
        // `min_logT_for_hydro_mtx_max_allowed < ln10 * min(lnT(k), lnT_start(k))`
        // (multiplying `lnT` by `ln10` instead of dividing by `ln10`). Reproduced
        // verbatim for `26.4.1` parity (`CLAUDE.md`, `#252`).
        let high_t_mtx_gate = model.controls.min_logt_for_hydro_mtx_max_allowed
            < LN10 * lnt_k.min(lnt_start_k);
        if (lnt_k - lnt_start_k).abs()
            > LN10 * model.controls.hydro_mtx_max_allowed_abs_dlogt
            && high_t_mtx_gate
        {
            return Err(StarError::ConvergenceFailure(format!(
                "abs(dlogT) > hydro_mtx_max_allowed_abs_dlogT at zone {k}"
            )));
        }
        if lnt_k > LN10 * model.controls.hydro_mtx_max_allowed_logt
            && high_t_mtx_gate
        {
            return Err(StarError::ConvergenceFailure(format!(
                "logT > hydro_mtx_max_allowed_logT at zone {k}"
            )));
        }
        if lnt_k < LN10 * model.controls.hydro_mtx_min_allowed_logt {
            return Err(StarError::ConvergenceFailure(format!(
                "logT < hydro_mtx_min_allowed_logT at zone {k}"
            )));
        }
        let t_k = flexmesa_math::exp(lnt_k);
        if !t_k.is_finite() {
            return Err(StarError::ConvergenceFailure(format!(
                "bad num for T at zone {k}"
            )));
        }
        model.ln_t[k] = lnt_k;
        model.t[k] = t_k;

        // `L` check (`solver_support.f90:1189-1200, 1033-1049`).
        let lum_k = model.xh[h_base + i_lum];
        if !lum_k.is_finite() {
            return Err(StarError::ConvergenceFailure(format!(
                "bad num for L at zone {k}"
            )));
        }
        model.lum[k] = lum_k;

        // Optional `RSP2` `w` and `Hp_face` (`solver_support.f90:1202-1228`).
        if let Some(iw) = i_w {
            let mut w_k = model.xh[h_base + iw];
            if w_k < 0.0 {
                let w_fix = model.controls.rsp2.w_fix_if_neg;
                w_k = if w_fix < 0.0 { 0.0 } else { w_fix };
                model.xh[h_base + iw] = w_k;
            }
            if !w_k.is_finite() {
                return Err(StarError::ConvergenceFailure(format!(
                    "bad num for w at zone {k}"
                )));
            }
            let et = w_k * w_k;
            model.aux.rsp.et_rsp[k] = et;
            model.aux.etrb[k] = et;
        }
        if let Some(ihp) = i_hp {
            let hp_k = model.xh[h_base + ihp];
            if !hp_k.is_finite() {
                return Err(StarError::ConvergenceFailure(format!(
                    "bad num for Hp_face at zone {k}"
                )));
            }
            model.aux.rsp.hp_face[k] = hp_k;
        }

        // Optional `v` and `u` (`solver_support.f90:1230-1256`).
        if let Some(iv) = i_v {
            let v_k = model.xh[h_base + iv];
            if !v_k.is_finite() {
                return Err(StarError::ConvergenceFailure(format!(
                    "bad num for v at zone {k}"
                )));
            }
            model.v[k] = v_k;
        }
        if let Some(iu) = i_u {
            let u_k = model.xh[h_base + iu];
            if !u_k.is_finite() {
                return Err(StarError::ConvergenceFailure(format!(
                    "bad num for u at zone {k}"
                )));
            }
            model.u[k] = u_k;
        }

        // Optional `alpha_RTI` (`solver_support.f90:1258-1274`).
        if let Some(ia) = i_alpha_rti {
            let raw_a = model.xh[h_base + ia];
            if !raw_a.is_finite() {
                return Err(StarError::ConvergenceFailure(format!(
                    "bad num for alpha_RTI at zone {k}"
                )));
            }
            let a_k = raw_a.max(0.0);
            model.xh[h_base + ia] = a_k;
            model.aux.alpha_rti[k] = a_k;
        }

        // `lnR` check (`solver_support.f90:1308-1323`).
        let lnr_k = model.xh[h_base + i_lnr];
        if !lnr_k.is_finite() {
            return Err(StarError::ConvergenceFailure(format!(
                "bad num for lnR at zone {k}"
            )));
        }
        model.ln_r[k] = lnr_k;
        model.r[k] = flexmesa_math::exp(lnr_k);

        // `lnd` update and `hydro_mtx_*_logRho` domain checks (`solver_support.f90:1325-1372`).
        let lnd_k = model.xh[h_base + i_lnd];
        let lnd_start_k = model.lnd_start.get(k).copied().unwrap_or(lnd_k);
        if !lnd_k.is_finite() {
            return Err(StarError::ConvergenceFailure(format!(
                "bad num for lnd at zone {k}"
            )));
        }
        if lnd_k < LN10 * model.controls.hydro_mtx_min_allowed_logrho {
            return Err(StarError::ConvergenceFailure(format!(
                "logRho < hydro_mtx_min_allowed_logRho at zone {k}"
            )));
        }
        if lnd_k > LN10 * model.controls.hydro_mtx_max_allowed_logrho
            && high_t_mtx_gate
        {
            return Err(StarError::ConvergenceFailure(format!(
                "logRho > hydro_mtx_max_allowed_logRho at zone {k}"
            )));
        }
        if (lnd_k - lnd_start_k).abs()
            > LN10 * model.controls.hydro_mtx_max_allowed_abs_dlogrho
            && high_t_mtx_gate
        {
            return Err(StarError::ConvergenceFailure(format!(
                "abs(dlogRho) > hydro_mtx_max_allowed_abs_dlogRho at zone {k}"
            )));
        }
        let rho_k = flexmesa_math::exp(lnd_k);
        if !rho_k.is_finite() || rho_k <= 0.0 {
            return Err(StarError::ConvergenceFailure(format!(
                "bad num for rho at zone {k}"
            )));
        }
        model.lnd[k] = lnd_k;
        model.rho[k] = rho_k;

        // `check1_chem` (`solver_support.f90:1374-1477`).
        if include_chem && model.species > 0 {
            let species = model.species;
            let c_base = k * species;
            if min_xa_hard_limit > -1.0e50 {
                for j in 0..species {
                    if model.xa[c_base + j] < min_xa_hard_limit {
                        model.why_tlim_neg_x = true;
                        return Err(StarError::ConvergenceFailure(format!(
                            "some xa < min_xa_hard_limit at zone {k}"
                        )));
                    }
                }
            }
            let mut sum_xa = 0.0_f64;
            for j in 0..species {
                let x = model.xa[c_base + j];
                if !x.is_finite() {
                    return Err(StarError::ConvergenceFailure(format!(
                        "bad num for mass fractions at zone {k}"
                    )));
                }
                let x_clamped = x.clamp(0.0, 1.0);
                model.xa[c_base + j] = x_clamped;
                sum_xa += x_clamped;
            }
            if !sum_xa.is_finite() {
                return Err(StarError::ConvergenceFailure(format!(
                    "bad num for mass fractions at zone {k}"
                )));
            }
            if (sum_xa - 1.0).abs() > sum_xa_hard_limit {
                return Err(StarError::ConvergenceFailure(format!(
                    "mass fractions > sum_xa_hard_limit at zone {k}: sum_xa = {sum_xa}"
                )));
            }
            if sum_xa > 0.0 && (sum_xa - 1.0).abs() > 1.0e-12 {
                for j in 0..species {
                    model.xa[c_base + j] /= sum_xa;
                }
            }
            let clip_limit = model.controls.xa_clip_limit;
            if clip_limit > 0.0 {
                for j in 0..species {
                    if model.xa[c_base + j] < clip_limit {
                        model.xa[c_base + j] = 0.0;
                    }
                }
            }
        }
    }

    // `edit_lnR` (`solver_support.f90:1075-1090, 1516-1534`): project `ln_r`
    // from `R_center`, `dm`, and `rho`, and hold `m_grav` constant during
    // Newton iterations.
    let mut vol00 = FOUR_THIRDS_PI * flexmesa_math::pow3(model.r_center);
    for k in (0..nz).rev() {
        let cell_vol = model.dm[k] / model.rho[k];
        vol00 += cell_vol;
        if vol00 <= 0.0 || !vol00.is_finite() {
            return Err(StarError::ConvergenceFailure(format!(
                "non-positive volume in edit_lnR at zone {k}: {vol00}"
            )));
        }
        let lnr_k = flexmesa_math::log(vol00 / FOUR_THIRDS_PI) / 3.0;
        model.ln_r[k] = lnr_k;
        model.xh[k * nvar_hydro + i_lnr] = lnr_k;
        solver_dx[k * nvar + i_lnr] = lnr_k - xh_start[k * nvar_hydro + i_lnr];
    }

    for k in 0..nz {
        let r_k = flexmesa_math::exp(model.ln_r[k]);
        if !r_k.is_finite() || r_k <= 0.0 {
            return Err(StarError::ConvergenceFailure(format!(
                "bad radius in set_vars_for_solver at zone {k}: {r_k}"
            )));
        }
        model.r[k] = r_k;
        model.grav[k] = model.cgrav[k] * model.m_grav[k] / (r_k * r_k);
    }

    for k in 0..nz {
        let r_inner = if k + 1 < nz {
            model.r[k + 1]
        } else {
            model.r_center
        };
        if model.r[k] < r_inner {
            return Err(StarError::ConvergenceFailure(format!(
                "set_rmid r[k] < r_inner at zone {k}"
            )));
        }
        model.rmid[k] = if model.controls.rsp_flag {
            0.5 * (model.r[k] + r_inner)
        } else {
            flexmesa_math::pow(
                0.5 * (flexmesa_math::pow3(model.r[k])
                    + flexmesa_math::pow3(r_inner)),
                1.0 / 3.0,
            )
        };
    }

    Ok(())
}

/// Validates converged thermodynamic and geometric bounds (`check_after_converge`
/// in `star/private/struct_burn_mix.f90:679-721`).
///
/// # Errors
/// Returns a diagnostic string when `R_center > r[nz-1]`, `ln_r[k] <= ln_r[k+1]`,
/// $\log_{10} T \notin [1, 12]$, or $\log_{10} \rho \notin [-30, 12]$.
pub fn check_after_converge(model: &StarModel) -> Result<(), String> {
    let nz = model.nz;
    if nz == 0 {
        return Ok(());
    }
    if model.r_center > 0.0
        && model.r_center > flexmesa_math::exp(model.ln_r[nz - 1])
    {
        return Err(format!("volume < 0 in cell {}", nz - 1));
    }
    for k in 0..nz.saturating_sub(1) {
        if model.ln_r[k] <= model.ln_r[k + 1] {
            return Err(format!(
                "after hydro, negative cell volume in cell {k}"
            ));
        }
        if model.ln_t[k] > 12.0 * LN10 {
            return Err(format!("after hydro, logT > 12 in cell {k}"));
        }
        if model.ln_t[k] < LN10 {
            return Err(format!("after hydro, logT < 1 in cell {k}"));
        }
        if model.lnd[k] > 12.0 * LN10 {
            return Err(format!("after hydro, logRho > 12 in cell {k}"));
        }
        if model.lnd[k] < -30.0 * LN10 {
            return Err(format!("after hydro, logRho < -30 in cell {k}"));
        }
    }
    Ok(())
}

/// Enforces `rti_energy_floor` after Henyey convergence when `rti_flag` is
/// active (`RTI_check_after_converge` in
/// `star/private/struct_burn_mix.f90:619-676`).
///
/// # Errors
/// Returns [`StarError`] if EOS temperature inversion fails.
pub fn rti_check_after_converge(
    model: &mut StarModel,
) -> Result<(), StarError> {
    if !model.controls.rti_flag || model.nz == 0 {
        return Ok(());
    }
    let nz = model.nz;
    let species = model.species;
    let floor = model.controls.rti_energy_floor;
    let nvar_hydro = model.layout.nvar_hydro;
    let i_lnt = model.layout.i_lnt;

    for k in 0..nz {
        if k + 1 < nz
            && model.aux.alpha_rti.get(k).copied().unwrap_or(0.0) < 1.0e-10
        {
            continue;
        }
        let old_energy = model.energy[k];
        if old_energy >= floor {
            continue;
        }
        model.energy[k] = floor;
        if floor > 0.0 {
            model.ln_e[k] = flexmesa_math::log(floor);
        }
        let log_rho = model.lnd[k] / LN10;
        let xa_k = model.xa[k * species..(k + 1) * species].to_vec();
        let (new_lnt, _rev_e) = crate::mesh_remap::solve_cell_lnt_for_energy(
            model,
            &xa_k,
            log_rho,
            model.energy[k],
            model.ln_t[k],
            1.0e-11,
        )?;
        model.ln_t[k] = new_lnt;
        model.t[k] = flexmesa_math::exp(new_lnt);
        if model.xh.len() == nz * nvar_hydro {
            model.xh[k * nvar_hydro + i_lnt] = new_lnt;
        }
        let old_ie = old_energy * model.dm[k];
        let new_ie = model.energy[k] * model.dm[k];
        if model.controls.u_flag {
            let old_ke = 0.5 * model.dm[k] * model.u[k] * model.u[k];
            let new_ke = (old_ke + old_ie - new_ie).max(0.0);
            let new_u = (new_ke / (0.5 * model.dm[k].max(1.0e-99))).sqrt();
            model.u[k] = if model.u[k] > 0.0 { new_u } else { -new_u };
            if let Some(iu) = model
                .layout
                .i_u
                .filter(|_| model.xh.len() == nz * nvar_hydro)
            {
                model.xh[k * nvar_hydro + iu] = model.u[k];
            }
        } else if model.controls.v_flag {
            let old_ke = 0.5 * model.dm_bar[k] * model.v[k] * model.v[k];
            let new_ke = (old_ke + old_ie - new_ie).max(0.0);
            let new_v = (new_ke / (0.5 * model.dm_bar[k].max(1.0e-99))).sqrt();
            model.v[k] = if model.v[k] > 0.0 { new_v } else { -new_v };
            if let Some(iv) = model
                .layout
                .i_v
                .filter(|_| model.xh.len() == nz * nvar_hydro)
            {
                model.xh[k * nvar_hydro + iv] = model.v[k];
            }
        }
    }
    Ok(())
}
