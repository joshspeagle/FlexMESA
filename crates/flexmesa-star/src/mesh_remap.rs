// SPDX-License-Identifier: LGPL-3.0-or-later
//! Conservative species, volume, kinetic-energy, angular-momentum, and internal
//! energy remapping (`do_mesh_adjust`, `get1_lpp`, `adjust_omega`, and
//! `set_lnT_for_energy` in `star/private/mesh_adjust.f90`).

use flexmesa_eos::{eos_dt_get_zones, HelmTable, I_LN_E};

use crate::controls::StarError;
use crate::hydro_vars::set_hydro_vars;
use crate::model::StarModel;

pub(crate) fn normalize_and_assign_remapped_xa(
    dst: &mut [f64],
    accum: &[f64],
    dx: f64,
) {
    let mut sum_x = 0.0;
    for (d, &acc) in dst.iter_mut().zip(accum.iter()) {
        let xj = (acc / dx).clamp(0.0, 1.0);
        *d = xj;
        sum_x += xj;
    }
    if sum_x > 0.0 {
        for d in dst {
            *d /= sum_x;
        }
    }
}

/// Computes monotone, sum-preserving linear slopes `xa_c1` (`(nz_old, species)`)
/// in exterior mass coordinate `xq` (`get1_lpp` in `mesh_adjust.f90:266-299`).
#[must_use]
pub fn compute_monotone_xa_slopes(
    dq_old: &[f64],
    xa_old: &[f64],
    nz_old: usize,
    species: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut xa_c0 = vec![0.0; nz_old * species];
    let mut xa_c1 = vec![0.0; nz_old * species];
    if nz_old == 0 || species == 0 {
        return (xa_c0, xa_c1);
    }

    for k in 0..nz_old {
        let base = k * species;
        let sum_x: f64 = xa_old[base..base + species].iter().sum();
        if sum_x > 0.0 {
            for j in 0..species {
                xa_c0[base + j] = (xa_old[base + j] / sum_x).clamp(0.0, 1.0);
            }
        }
    }
    if nz_old < 3 {
        return (xa_c0, xa_c1);
    }

    for k in 1..(nz_old - 1) {
        let base = k * species;
        let dm_lo = 0.5 * (dq_old[k - 1] + dq_old[k]).max(1.0e-99);
        let dm_hi = 0.5 * (dq_old[k] + dq_old[k + 1]).max(1.0e-99);
        for j in 0..species {
            let x_m1 = xa_c0[(k - 1) * species + j];
            let x_00 = xa_c0[base + j];
            let x_p1 = xa_c0[(k + 1) * species + j];
            let s_lo = (x_00 - x_m1) / dm_lo;
            let s_hi = (x_p1 - x_00) / dm_hi;
            if s_lo * s_hi > 0.0 {
                let s_min = s_lo.abs().min(s_hi.abs());
                xa_c1[base + j] = s_lo.signum() * s_min;
            }
        }

        // Enforce `sum_j xa_c1[k, j] == 0` by adjusting the dominant-slope
        // species (`mesh_adjust.f90:283-290`).
        let slope_sum: f64 = xa_c1[base..base + species].iter().sum();
        if slope_sum.abs() > 0.0 {
            let mut j_piv = 0_usize;
            if slope_sum > 0.0 {
                for j in 1..species {
                    if xa_c1[base + j] > xa_c1[base + j_piv] {
                        j_piv = j;
                    }
                }
            } else {
                for j in 1..species {
                    if xa_c1[base + j] < xa_c1[base + j_piv] {
                        j_piv = j;
                    }
                }
            }
            xa_c1[base + j_piv] -= slope_sum;
        }

        // Zero all slopes in cell `k` if any species would leave `[0, 1]` at
        // either cell edge (`mesh_adjust.f90:291-297`).
        let half_dq = 0.5 * dq_old[k];
        let mut invalid = false;
        for j in 0..species {
            let dxa = xa_c1[base + j].abs() * half_dq;
            if xa_c0[base + j] + dxa > 1.0 || xa_c0[base + j] - dxa < 0.0 {
                invalid = true;
                break;
            }
        }
        if invalid {
            xa_c1[base..base + species].fill(0.0);
        }
    }
    (xa_c0, xa_c1)
}

/// Inverts EOS $\ln e(\rho_k, \ln T_k, \mathbf{X}_k) = \ln e_{\mathrm{target},k}$
/// across all modified zones via damped Newton-Raphson (`set_lnT_for_energy` in
/// `mesh_adjust.f90:345-385, 2550-2680`).
///
/// # Errors
/// Returns [`StarError`] if EOS evaluation fails during temperature inversion.
pub fn set_lnt_for_energy(
    model: &mut StarModel,
    e_target: &[f64],
    unchanged: &[bool],
) -> Result<f64, StarError> {
    let nz = model.nz;
    if nz == 0 || e_target.len() != nz {
        return Ok(0.0);
    }
    let ht = HelmTable::builtin();

    for _iter in 0..20 {
        for k in 0..nz {
            model.rho[k] = model.lnd[k].exp();
            model.t[k] = model.ln_t[k].exp();
        }
        let outs = eos_dt_get_zones(
            ht,
            &model.rho,
            &model.t,
            &model.chem_id,
            &model.xa,
            &model.eos_config,
        )
        .map_err(|e| {
            StarError::MicrophysicsError(format!(
                "set_lnt_for_energy EOS failed: {e}"
            ))
        })?;

        let mut max_rel = 0.0_f64;
        for (k, out) in outs.iter().enumerate() {
            let e_cur = out.res[I_LN_E].exp();
            model.energy[k] = e_cur;
            if unchanged.get(k).copied().unwrap_or(false) || e_target[k] <= 0.0
            {
                continue;
            }
            let ln_e_tar = e_target[k].ln();
            let resid = out.res[I_LN_E] - ln_e_tar;
            let rel = ((e_cur - e_target[k]) / e_target[k]).abs();
            max_rel = max_rel.max(rel);
            if rel > 1.0e-13 {
                let d_lne_dlnt = out.d_dln_t[I_LN_E].max(1.0e-4);
                let step = (-resid / d_lne_dlnt).clamp(-0.15, 0.15);
                model.ln_t[k] = (model.ln_t[k] + step).clamp(3.0, 26.0);
            }
        }
        if max_rel <= 1.0e-12 {
            break;
        }
    }

    for k in 0..nz {
        model.rho[k] = model.lnd[k].exp();
        model.t[k] = model.ln_t[k].exp();
    }
    let outs = eos_dt_get_zones(
        ht,
        &model.rho,
        &model.t,
        &model.chem_id,
        &model.xa,
        &model.eos_config,
    )
    .map_err(|e| {
        StarError::MicrophysicsError(format!(
            "set_lnt_for_energy final EOS failed: {e}"
        ))
    })?;
    for (k, out) in outs.iter().enumerate() {
        model.energy[k] = out.res[I_LN_E].exp();
    }

    let e_before: f64 = model
        .dq
        .iter()
        .zip(e_target.iter())
        .map(|(dq, e)| dq * e)
        .sum();
    let e_after: f64 = model
        .dq
        .iter()
        .zip(model.energy.iter())
        .map(|(dq, e)| dq * e)
        .sum();
    let rel_err =
        (e_before - e_after).abs() / e_before.abs().max(e_after.abs()).max(1.0);
    Ok(rel_err)
}

fn dqbar_profile(dq: &[f64]) -> Vec<f64> {
    let n = dq.len();
    let mut dqbar = vec![0.0; n];
    if n == 0 {
        return dqbar;
    }
    dqbar[0] = 0.5 * dq[0];
    for k in 1..(n - 1) {
        dqbar[k] = 0.5 * (dq[k - 1] + dq[k]);
    }
    if n > 1 {
        dqbar[n - 1] = 0.5 * dq[n - 2] + dq[n - 1];
    } else {
        dqbar[0] = dq[0];
    }
    dqbar
}

/// Conservatively projects `model` onto a new mass grid `dq_new` (`mesh_adjust`
/// in `star/private/mesh_adjust.f90`).
///
/// Conserves species mass $\sum_k \Delta m_k X_{j,k}$ via monotone linear
/// overlap integration, cell volume $\sum_k \Delta m_k / \rho_k$, total
/// internal energy $\sum_k \Delta m_k e_k$ via EOS $\ln T$ inversion, specific
/// kinetic/turbulent energy, and total rotational angular momentum
/// $J_{\mathrm{tot}} = \sum_k \overline{\Delta m}_k j_{\mathrm{rot},k}$.
///
/// # Errors
/// Returns [`StarError`] if `dq_new` has fewer than 2 zones or EOS fails.
pub fn remap_onto_dq(
    model: &mut StarModel,
    dq_new: &[f64],
) -> Result<(), StarError> {
    let nz_old = model.nz;
    let nz_new = dq_new.len();
    let species = model.species;
    if nz_new < 2 {
        return Err(StarError::InvalidInput(String::from(
            "dq_new must have at least 2 zones",
        )));
    }

    let dq_old = model.dq.clone();
    let xq_old = model.xq.clone();
    let xa_old = model.xa.clone();
    let e_old = model.energy.clone();
    let lnd_old = model.lnd.clone();
    let lnt_old = model.ln_t.clone();
    let lum_old = model.lum.clone();
    let v_old = model.v.clone();
    let u_old = model.u.clone();
    let omega_old = model.omega.clone();
    let j_rot_old = model.j_rot.clone();
    let et_rsp_old = model.aux.rsp.et_rsp.clone();
    let hp_face_old = model.aux.rsp.hp_face.clone();
    let alpha_rti_old = model.aux.alpha_rti.clone();

    let (xa_c0, xa_c1) =
        compute_monotone_xa_slopes(&dq_old, &xa_old, nz_old, species);
    let dqbar_old = dqbar_profile(&dq_old);

    model.resize_zones(nz_new);
    model.dq.copy_from_slice(dq_new);
    model.normalize_dqs()?;
    model.set_qs();
    model.set_xqs();
    model.set_m_and_dm();

    let mut e_target = vec![0.0; nz_new];
    let mut unchanged = vec![false; nz_new];
    let mut accum = vec![0.0; species];
    let mut kk_start = 0_usize;

    for k in 0..nz_new {
        let a0 = model.xq[k];
        let a1 = if k + 1 < nz_new { model.xq[k + 1] } else { 1.0 };
        let dx = (a1 - a0).max(1.0e-99);
        accum.fill(0.0);
        let mut e_acc = 0.0;
        let mut inv_rho_acc = 0.0;
        let mut lnt_acc = 0.0;
        let mut lum_acc = 0.0;
        let mut v_acc = 0.0;
        let mut v2_acc = 0.0;
        let mut u_acc = 0.0;
        let mut u2_acc = 0.0;
        let mut et_acc = 0.0;
        let mut hp_acc = 0.0;
        let mut omega_acc = 0.0;
        let mut alpha_rti_acc = 0.0;

        while kk_start + 1 < nz_old && xq_old[kk_start + 1] <= a0 {
            kk_start += 1;
        }
        let exact_match = (kk_start..(kk_start + 2).min(nz_old)).find(|&kk| {
            let b0 = xq_old[kk];
            let b1 = if kk + 1 < nz_old { xq_old[kk + 1] } else { 1.0 };
            (a0 - b0).abs() <= 1.0e-15 * dx && (a1 - b1).abs() <= 1.0e-15 * dx
        });
        if let Some(kk) = exact_match {
            unchanged[k] = true;
            model.xa[k * species..(k + 1) * species]
                .copy_from_slice(&xa_old[kk * species..(kk + 1) * species]);
            e_target[k] = e_old[kk];
            model.energy[k] = e_old[kk];
            model.lnd[k] = lnd_old[kk];
            model.ln_t[k] = lnt_old[kk];
            model.lum[k] = lum_old[kk];
            model.v[k] = v_old[kk];
            model.u[k] = u_old[kk];
            model.aux.rsp.et_rsp[k] =
                et_rsp_old.get(kk).copied().unwrap_or(0.0);
            model.aux.rsp.hp_face[k] =
                hp_face_old.get(kk).copied().unwrap_or(0.0);
            model.omega[k] = omega_old[kk];
            model.aux.alpha_rti[k] =
                alpha_rti_old.get(kk).copied().unwrap_or(0.0);
            continue;
        }
        for kk in kk_start..nz_old {
            let b0 = xq_old[kk];
            if b0 >= a1 {
                break;
            }
            let b1 = if kk + 1 < nz_old { xq_old[kk + 1] } else { 1.0 };
            let lo = a0.max(b0);
            let hi = a1.min(b1);
            let ov = (hi - lo).max(0.0);
            if ov > 0.0 {
                let x_mid = 0.5 * (lo + hi);
                let x_cen = b0 + 0.5 * dq_old[kk];
                let dx_cen = x_mid - x_cen;
                let base = kk * species;
                for j in 0..species {
                    accum[j] +=
                        ov * (xa_c0[base + j] + xa_c1[base + j] * dx_cen);
                }
                e_acc += ov * e_old[kk];
                inv_rho_acc += ov * (-lnd_old[kk]).exp();
                lnt_acc += ov * lnt_old[kk];
                lum_acc += ov * lum_old[kk];
                v_acc += ov * v_old[kk];
                v2_acc += ov * v_old[kk] * v_old[kk];
                u_acc += ov * u_old[kk];
                u2_acc += ov * u_old[kk] * u_old[kk];
                et_acc += ov * et_rsp_old.get(kk).copied().unwrap_or(0.0);
                hp_acc += ov * hp_face_old.get(kk).copied().unwrap_or(0.0);
                omega_acc += ov * omega_old[kk];
                alpha_rti_acc +=
                    ov * alpha_rti_old.get(kk).copied().unwrap_or(0.0);
            }
        }

        normalize_and_assign_remapped_xa(
            &mut model.xa[k * species..(k + 1) * species],
            &accum,
            dx,
        );
        e_target[k] = e_acc / dx;
        model.energy[k] = e_target[k];
        let inv_rho = (inv_rho_acc / dx).max(1.0e-99);
        model.lnd[k] = -(inv_rho.ln());
        model.ln_t[k] = lnt_acc / dx;
        model.lum[k] = lum_acc / dx;
        let v_mean = v_acc / dx;
        model.v[k] = if model.controls.v_flag && v_mean != 0.0 {
            v_mean.signum() * (v2_acc / dx).max(0.0).sqrt()
        } else {
            v_mean
        };
        let u_mean = u_acc / dx;
        model.u[k] = if model.controls.u_flag && u_mean != 0.0 {
            u_mean.signum() * (u2_acc / dx).max(0.0).sqrt()
        } else {
            u_mean
        };
        model.aux.rsp.et_rsp[k] = (et_acc / dx).max(0.0);
        model.aux.rsp.hp_face[k] = (hp_acc / dx).max(0.0);
        model.omega[k] = omega_acc / dx;
        model.aux.alpha_rti[k] = (alpha_rti_acc / dx).max(0.0);
    }

    model.reconstruct_ln_r_from_lnd_and_dm();

    // Conserve total rotational angular momentum `sum_k dm_bar[k] * j_rot[k]`
    // across face-centered control volumes (`adjust_omega`,
    // `mesh_adjust.f90:232-248, 1280-1360`).
    if model.controls.rotation_flag {
        let dqbar_new = dqbar_profile(&model.dq);
        let mut xbar_old = vec![0.0; nz_old + 1];
        for kk in 0..nz_old {
            xbar_old[kk + 1] = xbar_old[kk] + dqbar_old[kk];
        }
        xbar_old[nz_old] = 1.0;

        let mut xbar_cur = 0.0;
        let mut kk_b = 0_usize;
        for k in 0..nz_new {
            let a0 = xbar_cur;
            let a1 = if k + 1 < nz_new {
                a0 + dqbar_new[k]
            } else {
                1.0
            };
            xbar_cur = a1;
            let dx = (a1 - a0).max(1.0e-99);
            while kk_b + 1 < nz_old && xbar_old[kk_b + 1] <= a0 {
                kk_b += 1;
            }
            let mut j_acc = 0.0;
            for kk in kk_b..nz_old {
                let b0 = xbar_old[kk];
                if b0 >= a1 {
                    break;
                }
                let b1 = xbar_old[kk + 1];
                let ov = (a1.min(b1) - a0.max(b0)).max(0.0);
                if ov > 0.0 {
                    let jr = j_rot_old.get(kk).copied().unwrap_or_else(|| {
                        (2.0 / 3.0)
                            * model.r[k.min(nz_new - 1)].powi(2)
                            * omega_old[kk]
                    });
                    j_acc += ov * jr;
                }
            }
            let jr_new = j_acc / dx;
            let i_rot_k = ((2.0 / 3.0) * model.r[k] * model.r[k]).max(1.0e-99);
            model.j_rot[k] = jr_new;
            model.aux.i_rot[k] = i_rot_k;
            model.omega[k] = jr_new / i_rot_k;
        }
    }

    model.set_m_grav_and_grav();
    let _ie_err = set_lnt_for_energy(model, &e_target, &unchanged)?;

    let (age_old, dt_old, mstar_old) =
        (model.star_age_old, model.dt_old, model.mstar_old);
    model.pack_xh();
    set_hydro_vars(model, false, false)?;
    model.save_start_values();
    model.save_old_values();
    model.star_age_old = age_old;
    model.dt_old = dt_old;
    model.mstar_old = mstar_old;
    Ok(())
}
