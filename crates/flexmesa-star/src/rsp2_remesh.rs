// SPDX-License-Identifier: GPL-3.0-or-later
//! Two-zone anchor-temperature remeshing and quasi-hydrostatic `lnT` revision
//! for `RSP2` (`remesh_for_RSP2` in `star/private/hydro_rsp2_support.f90:36-442`).

use flexmesa_const::{FOUR_THIRDS_PI, LN10, PI4, SQRT_2_DIV_3};
use flexmesa_eos::{eos_dt_get_t, WHICH_OTHER_LOG_PTOT};
use flexmesa_interp1d::interpolate_vector_pm;

use crate::controls::StarError;
use crate::hydro_vars::set_hydro_vars;
use crate::model::StarModel;
use crate::rsp2_eqns::eval_rsp2_hp_face_zone;

/// Seeds `RSP2` turbulent energy $e_t = w^2$ from face `mlt_vc` when enabling
/// `RSP2_flag` on a model without existing turbulent energy (`set_flags.f90:361-370`).
///
/// # Errors
/// Returns [`StarError`] if microphysics or convection evaluation fails.
pub fn seed_rsp2_w_from_mlt_vc_if_needed(
    model: &mut StarModel,
) -> Result<(), StarError> {
    let nz = model.nz;
    model.aux.rsp.resize(nz);
    if model.aux.rsp.et_rsp.iter().any(|&et| et > 0.0) {
        return Ok(());
    }
    if model.mlt_vc.len() < nz || model.mlt_vc.iter().all(|&v| v <= 0.0) {
        crate::micro::set_basic_vars(model)?;
        crate::micro::do_eos(model)?;
        crate::micro::do_kap(model)?;
        crate::hydro_brunt_surf::do_brunt_b(model)?;
        crate::hydro_vars::set_mlt_vars(model)?;
    }
    if model.mlt_vc.len() < nz {
        return Ok(());
    }
    for k in 0..nz {
        let vc0 = model.mlt_vc[k].max(0.0);
        let vc1 = model.mlt_vc.get(k + 1).copied().unwrap_or(0.0).max(0.0);
        let wk = 0.5 * (vc0 + vc1) / SQRT_2_DIV_3;
        model.aux.rsp.et_rsp[k] = wk * wk;
        model.aux.rsp.w_start[k] = wk;
    }
    Ok(())
}

/// Remeshes `model` onto the `RSP2` two-region anchor-temperature Lagrangian
/// mass grid (`RSP2_nz`, `RSP2_nz_outer`, `RSP2_T_anchor`, `RSP2_dq_1_factor`)
/// using Steffen (1990) piecewise-monotonic cubic interpolation and revises
/// $\ln T_k$ for quasi-hydrostatic equilibrium (`hydro_rsp2_support.f90:36-442`).
///
/// # Errors
/// Returns [`StarError`] if `RSP2_nz > model.nz`, `T_anchor` lies above the
/// surface temperature, spline interpolation fails, or QHSE `eosDT` inversion
/// fails to converge.
pub fn remesh_for_rsp2(model: &mut StarModel) -> Result<(), StarError> {
    let nz_old = model.nz;
    let nz = model.controls.rsp2.nz;
    if nz == nz_old {
        return Ok(());
    }
    if nz > nz_old {
        return Err(StarError::InvalidInput(format!(
            "remesh_for_rsp2 cannot increase nz ({nz} > {nz_old})"
        )));
    }
    let nz_outer = model.controls.rsp2.nz_outer;
    if nz_outer == 0 || nz_outer >= nz {
        return Err(StarError::InvalidInput(format!(
            "invalid RSP2_nz_outer ({nz_outer}) for RSP2_nz ({nz})"
        )));
    }

    set_hydro_vars(model, true, true)?;
    seed_rsp2_w_from_mlt_vc_if_needed(model)?;
    crate::hydro_brunt_surf::eval_surf_bc(model)?;
    let p_surf = flexmesa_math::exp(model.surf_pt.ln_p);

    // 1. Old exterior mass coordinates `xm_old` (`0..=nz_old`) and `xm_mid_old`.
    let mut xm_old = vec![0.0_f64; nz_old + 1];
    let mut xm_mid_old = vec![0.0_f64; nz_old];
    for k in 1..nz_old {
        xm_old[k] = xm_old[k - 1] + model.dm[k - 1];
    }
    xm_old[nz_old] = model.xmstar;
    for k in 0..nz_old {
        xm_mid_old[k] = xm_old[k] + 0.5 * model.dm[k];
    }

    // 2. Locate `xm_anchor` where `ln_t == ln(RSP2_T_anchor)` (`164-188`).
    let lnt_anchor = flexmesa_math::log(model.controls.rsp2.t_anchor);
    if lnt_anchor <= model.ln_t[0] {
        return Err(StarError::InvalidInput(format!(
            "remesh_for_rsp2: T_anchor ({}) <= T_surf ({})",
            model.controls.rsp2.t_anchor,
            flexmesa_math::exp(model.ln_t[0])
        )));
    }
    let mut xm_anchor = xm_old[nz_old - 1];
    for k in 1..nz_old {
        if model.ln_t[k] >= lnt_anchor {
            let (xmm1, xm00) = (xm_old[k - 1], xm_old[k]);
            let (lntm1, lnt00) = (model.ln_t[k - 1], model.ln_t[k]);
            let denom = (lnt00 - lntm1).max(1.0e-30);
            xm_anchor = xmm1 + (xm00 - xmm1) * (lnt_anchor - lntm1) / denom;
            break;
        }
    }
    if !xm_anchor.is_finite() || xm_anchor <= 0.0 || xm_anchor >= model.xmstar {
        return Err(StarError::ConvergenceFailure(format!(
            "remesh_for_rsp2: invalid xm_anchor = {xm_anchor}"
        )));
    }

    // 3. Build new `xm` (`0..=nz`), `dm`, and `xm_mid` (`190-230`).
    let dq_1_factor = model.controls.rsp2.dq_1_factor;
    let dxm_outer = xm_anchor / ((nz_outer as f64) - 1.0 + dq_1_factor);
    let mut xm = vec![0.0_f64; nz + 1];
    let mut dm_new = vec![0.0_f64; nz];
    xm[0] = 0.0;
    xm[1] = dxm_outer * dq_1_factor;
    dm_new[0] = xm[1];
    for k in 2..=nz_outer {
        xm[k] = xm[k - 1] + dxm_outer;
        dm_new[k - 1] = dxm_outer;
    }
    let mut lnx = flexmesa_math::log(xm[nz_outer]);
    let dlnx =
        (flexmesa_math::log(model.xmstar) - lnx) / ((nz - nz_outer) as f64);
    for k in (nz_outer + 1)..nz {
        lnx += dlnx;
        xm[k] = flexmesa_math::exp(lnx);
        dm_new[k - 1] = xm[k] - xm[k - 1];
    }
    xm[nz] = model.xmstar;
    dm_new[nz - 1] = model.xmstar - xm[nz - 1];

    let mut xm_mid = vec![0.0_f64; nz];
    for k in 0..nz {
        xm_mid[k] = 0.5 * (xm[k] + xm[k + 1]);
    }

    // 4. Interpolate face `(lnR, L, v)` and cell `(lnT, w, xa)` via `interp_pm`.
    let pm_err = |e: flexmesa_interp1d::Interp1dError| {
        StarError::ConvergenceFailure(format!("remesh_for_rsp2 interp_pm: {e}"))
    };
    let mut lnr_old = model.ln_r.clone();
    lnr_old.push(flexmesa_math::log(model.r_center.max(1.0)));
    let lnr_new =
        interpolate_vector_pm(&xm_old, &lnr_old, &xm).map_err(pm_err)?;

    let mut lum_old_f = model.lum.clone();
    lum_old_f.push(model.l_center);
    let lum_new =
        interpolate_vector_pm(&xm_old, &lum_old_f, &xm).map_err(pm_err)?;

    let v_new = if model.controls.v_flag {
        let mut v_old_f = model.v.clone();
        v_old_f.push(model.v_center);
        interpolate_vector_pm(&xm_old, &v_old_f, &xm).map_err(pm_err)?
    } else {
        vec![0.0; nz + 1]
    };

    let lnt_new = interpolate_vector_pm(&xm_mid_old, &model.ln_t, &xm_mid)
        .map_err(pm_err)?;
    let w_old: Vec<f64> = (0..nz_old)
        .map(|k| {
            model
                .aux
                .rsp
                .et_rsp
                .get(k)
                .copied()
                .unwrap_or(0.0)
                .max(0.0)
                .sqrt()
        })
        .collect();
    let w_new =
        interpolate_vector_pm(&xm_mid_old, &w_old, &xm_mid).map_err(pm_err)?;

    let species = model.species;
    let mut xa_new = vec![0.0_f64; nz * species];
    for j in 0..species {
        let col_old: Vec<f64> =
            (0..nz_old).map(|k| model.xa[k * species + j]).collect();
        let col_new = interpolate_vector_pm(&xm_mid_old, &col_old, &xm_mid)
            .map_err(pm_err)?;
        for k in 0..nz {
            xa_new[k * species + j] = col_new[k];
        }
    }
    for k in 0..nz {
        let row = &mut xa_new[k * species..(k + 1) * species];
        let sum_xa: f64 = row.iter().sum();
        if sum_xa > 0.0 {
            for x in row {
                *x /= sum_xa;
            }
        }
    }

    // 5. Resize cloned `StarModel` to preserve scalar and history state (`253-290`).
    let mut next = model.clone();
    next.resize_zones(nz);

    for k in 0..nz {
        next.dm[k] = dm_new[k];
        next.dq[k] = dm_new[k] / next.xmstar;
        next.ln_r[k] = lnr_new[k];
        next.r[k] = flexmesa_math::exp(lnr_new[k]);
        next.lum[k] = lum_new[k];
        next.v[k] = v_new[k];
        next.ln_t[k] = lnt_new[k];
        next.t[k] = flexmesa_math::exp(lnt_new[k]);
        let wk = w_new[k].max(0.0);
        next.aux.rsp.et_rsp[k] = wk * wk;
        next.aux.rsp.w_start[k] = wk;
    }
    next.xa = xa_new;
    next.m[0] = next.mstar;
    next.q[0] = 1.0;
    next.dq[0] = next.dm[0] / next.xmstar;
    for k in 1..nz {
        next.m[k] = next.m[k - 1] - next.dm[k - 1];
        next.dq[k] = next.dm[k] / next.xmstar;
        next.q[k] = next.q[k - 1] - next.dq[k - 1];
    }
    next.set_xqs();
    next.set_dm_bar();
    next.set_m_grav_and_grav();

    for k in 0..nz {
        let r3_00 = flexmesa_math::pow3(next.r[k]);
        let r3_p1 = if k + 1 < nz {
            flexmesa_math::pow3(next.r[k + 1])
        } else {
            flexmesa_math::pow3(next.r_center)
        };
        if r3_00 <= r3_p1 {
            return Err(StarError::ConvergenceFailure(format!(
                "remesh_for_rsp2: non-monotonic radius at zone {k}"
            )));
        }
        let vol = FOUR_THIRDS_PI * (r3_00 - r3_p1);
        next.rho[k] = next.dm[k] / vol;
        next.lnd[k] = flexmesa_math::log(next.rho[k]);
    }

    // 6. Revise `ln_t` for quasi-hydrostatic equilibrium (`327-418`) using pre-remesh `P_surf`.
    crate::micro::set_basic_vars(&mut next)?;
    crate::micro::do_eos(&mut next)?;
    crate::micro::do_kap(&mut next)?;
    let mut p_m1 = p_surf;
    for k in 0..nz {
        let dm_face = if k + 1 < nz {
            next.dm_bar[k]
        } else {
            0.5 * (next.dm[k - 1] + next.dm[k])
        };
        let p_00 = p_m1
            + next.cgrav[k] * next.m[k] * dm_face
                / (PI4 * flexmesa_math::pow4(next.r[k]));
        next.peos[k] = p_00;
        next.ln_peos[k] = flexmesa_math::log(p_00);
        let log_rho = next.lnd[k] / LN10;
        let log_p = flexmesa_math::log10(p_00);
        let log_t_guess = next.ln_t[k] / LN10;
        let inv = eos_dt_get_t(
            crate::micro::builtin_helm_table()?,
            &next.chem_id,
            next.cell_xa(k),
            log_rho,
            WHICH_OTHER_LOG_PTOT,
            log_p,
            log_t_guess,
            1.0e-11,
            1.0e-11,
            100,
            &next.eos_config,
        )
        .map_err(|e| {
            StarError::MicrophysicsError(format!(
                "remesh_for_rsp2 eos_dt_get_t at zone {k}: {e}"
            ))
        })?;
        next.ln_t[k] = inv.log10_result * LN10;
        next.t[k] = flexmesa_math::exp(next.ln_t[k]);
        p_m1 = p_00;
    }

    set_hydro_vars(&mut next, false, false)?;
    for k in 0..nz {
        let hp = eval_rsp2_hp_face_zone(&next, k);
        next.aux.rsp.hp_face[k] = hp;
    }
    next.pack_xh();
    next.save_start_values()?;
    next.save_old_values();
    *model = next;
    Ok(())
}
