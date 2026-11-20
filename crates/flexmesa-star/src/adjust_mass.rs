// SPDX-License-Identifier: LGPL-3.0-or-later
//! Three-region pseudo-Lagrangian mass adjustment (`do_adjust_mass`,
//! `compute_delta_m`, `revise_q_and_dq`, `set_omega_adjust_mass`,
//! `adjust_J_lost`, and `smooth_outer_xa`) matching
//! `star/private/adjust_mass.f90`.

use flexmesa_const::{LN10, M_SUN, SECYER};
use flexmesa_num::neumaier_sum;

pub use crate::accretion_xa::{build_accreted_xa, get_xa_for_accretion};
use crate::controls::StarError;
use crate::core_boundaries::dominant_species_index;
use crate::eps_mdot::{
    PrevMassState, calculate_eps_mdot_with_change,
    eval_specific_kinetic_energy_profile, eval_specific_total_energy_profile,
};
use crate::hydro_vars::set_hydro_vars;
use crate::mesh_remap::normalize_and_assign_remapped_xa;
use crate::micro::{do_eos, set_basic_vars};
use crate::model::StarModel;
use crate::rotation::{
    conserve_shell_j_rot_and_distortion, eval_omega_crit, eval_rotation_fp_ft,
};

fn sync_enclosed_mass_from_dm(model: &mut StarModel, dm_target: &[f64]) {
    model.dm.copy_from_slice(dm_target);
    let mut m_acc = model.m_center;
    for k in (1..model.nz).rev() {
        m_acc += model.dm[k];
        model.m[k] = m_acc;
    }
    model.m[0] = model.mstar;
}

/// Evaluates `delta_m = dt * mstar_dot` with the upstream super-Eddington
/// override and `min_star_mass_for_loss` / `max_star_mass_for_gain` gates
/// (`compute_delta_m` in `star/private/adjust_mass.f90:36-73`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if `dt`, `mstar_dot`, or `delta_m` is
/// non-finite.
pub fn compute_delta_m(
    model: &mut StarModel,
    dt: f64,
) -> Result<f64, StarError> {
    let mut delta_m = dt * model.mstar_dot;
    let sedd = model.aux.step_hist.super_eddington_wind_mdot;
    let mw = &model.controls.mix_wind;
    let star_mass = model.mstar / M_SUN;

    if sedd != 0.0 && sedd > -model.mstar_dot {
        if mw.max_wind > 0.0 && mw.max_wind * M_SUN / SECYER < sedd {
            model.mstar_dot = -mw.max_wind * M_SUN / SECYER;
        } else {
            model.mstar_dot = -sedd;
        }
    } else if delta_m == 0.0
        || (delta_m < 0.0 && star_mass <= mw.min_star_mass_for_loss)
        || (delta_m > 0.0
            && mw.max_star_mass_for_gain > 0.0
            && star_mass >= mw.max_star_mass_for_gain)
    {
        model.mstar_dot = 0.0;
    }

    delta_m = dt * model.mstar_dot;
    if !dt.is_finite() || !model.mstar_dot.is_finite() || !delta_m.is_finite() {
        return Err(StarError::InvalidInput(format!(
            "compute_delta_m: non-finite dt={dt}, mstar_dot={}, delta_m={delta_m}",
            model.mstar_dot
        )));
    }
    Ok(delta_m)
}

/// Applies the 3-region pseudo-Lagrangian mass adjustment (`do_adjust_mass` in
/// `star/private/adjust_mass.f90:73-1131`).
///
/// # Errors
/// Returns [`StarError`] if `xm_new <= 0`, accretion composition is invalid,
/// or microphysics fails.
pub fn do_adjust_mass(
    model: &mut StarModel,
    delta_m: f64,
    x_accreted: Option<&[f64]>,
    q_below_const_q: f64,
    q_const_mass: f64,
) -> Result<(), StarError> {
    do_adjust_mass_with_dt(
        model,
        delta_m,
        0.0,
        x_accreted,
        q_below_const_q,
        q_const_mass,
    )
}

/// Executes 3-region mass adjustment and, when `dt > 0`, evaluates
/// `model.eps_mdot` via [`crate::eps_mdot::calculate_eps_mdot_with_change`].
///
/// # Errors
/// Returns [`StarError`] if `xm_new <= 0`, accretion composition is invalid,
/// or microphysics fails.
pub fn do_adjust_mass_with_dt(
    model: &mut StarModel,
    delta_m: f64,
    dt: f64,
    x_accreted: Option<&[f64]>,
    q_below_const_q: f64,
    q_const_mass: f64,
) -> Result<(), StarError> {
    let nz = model.nz;
    if delta_m == 0.0 || nz < 3 {
        model.eps_mdot.fill(0.0);
        return Ok(());
    }
    let old_xmstar = model.xmstar;
    let new_xmstar = old_xmstar + delta_m;
    let old_mstar = model.mstar;
    let new_mstar = old_mstar + delta_m;

    let mut k_a = 1_usize;
    while k_a + 2 < nz && model.q[k_a] > q_below_const_q {
        k_a += 1;
    }
    let mut k_b = (k_a + 1).min(nz - 1);
    while k_b + 1 < nz && model.q[k_b] > q_const_mass {
        k_b += 1;
    }
    do_adjust_mass_core(
        model,
        old_mstar,
        new_mstar,
        old_xmstar,
        new_xmstar,
        delta_m,
        dt,
        x_accreted,
        Some((k_a, k_b)),
    )
}

/// Applies control-driven mass adjustment (`model.mstar_dot * dt`) using
/// upstream `compute_delta_m` (`adjust_mass.f90:36-73`), mass-limit clamps
/// (`adjust_mass.f90:247-265`), and `revise_q_and_dq`
/// (`adjust_mass.f90:607-832`).
///
/// # Errors
/// Returns [`StarError`] if envelope mass becomes non-positive, accretion
/// composition is invalid, or microphysics fails.
pub fn do_adjust_mass_step(
    model: &mut StarModel,
    dt: f64,
) -> Result<(), StarError> {
    let nz = model.nz;
    if dt <= 0.0 || nz < 3 {
        model.aux.k_const_mass = 0;
        model.eps_mdot.fill(0.0);
        return Ok(());
    }
    let mut delta_m = compute_delta_m(model, dt)?;
    if delta_m == 0.0 {
        model.aux.k_const_mass = 0;
        model.eps_mdot.fill(0.0);
        return Ok(());
    }

    let old_mstar = model.mstar;
    let old_xmstar = model.xmstar;
    let mut new_mstar = old_mstar + delta_m;
    let new_xmstar = old_xmstar + delta_m;
    let mw = &model.controls.mix_wind;

    if delta_m > 0.0
        && mw.max_star_mass_for_gain > 0.0
        && new_mstar > M_SUN * mw.max_star_mass_for_gain
    {
        new_mstar = M_SUN * mw.max_star_mass_for_gain;
        delta_m = new_mstar - old_mstar;
        model.mstar_dot = delta_m / dt;
        // Reproduces `star/private/adjust_mass.f90:247-265` verbatim (`#263`
        // under `#169`): upstream omits updating `new_xmstar` on the crossing
        // step, leaving `xmstar` and `revise_q_and_dq` with the unclamped value.
    } else if delta_m < 0.0 && new_mstar < M_SUN * mw.min_star_mass_for_loss {
        new_mstar = M_SUN * mw.min_star_mass_for_loss;
        delta_m = new_mstar - old_mstar;
        model.mstar_dot = delta_m / dt;
        // Reproduces `star/private/adjust_mass.f90:247-265` verbatim (`#263`
        // under `#169`): upstream omits updating `new_xmstar` on the crossing
        // step, leaving `xmstar` and `revise_q_and_dq` with the unclamped value.
    }

    do_adjust_mass_core(
        model, old_mstar, new_mstar, old_xmstar, new_xmstar, delta_m, dt, None,
        None,
    )
}

fn revise_q_and_dq_upstream(
    model: &mut StarModel,
    old_xmstar: f64,
    new_xmstar: f64,
    delta_m: f64,
    explicit_ka_kb: Option<(usize, usize)>,
) -> Result<(usize, usize, Vec<f64>), StarError> {
    let nz = model.nz;
    let mw = &model.controls.mix_wind;

    let mold_o_mnew = old_xmstar / new_xmstar;
    let dq_acc = delta_m / new_xmstar;

    let sum_dq_init = neumaier_sum(&model.dq);
    if !sum_dq_init.is_finite() || sum_dq_init <= 0.0 {
        return Err(StarError::ConvergenceFailure(format!(
            "revise_q_and_dq: invalid initial sum(dq)={sum_dq_init}"
        )));
    }
    let inv_sum_init = 1.0 / sum_dq_init;
    let mut dq: Vec<f64> = model.dq.iter().map(|&d| d * inv_sum_init).collect();

    let mut xq = vec![0.0_f64; nz];
    for k in 1..nz {
        xq[k] = xq[k - 1] + dq[k - 1];
    }

    let (k_a, k_b, qfrac) = if let Some((ka_in, kb_in)) = explicit_ka_kb {
        let ka = ka_in.clamp(1, nz.saturating_sub(2));
        let mut kb = kb_in.clamp(ka + 1, nz - 1);
        let xq_a = xq[ka];
        let mut qf;
        loop {
            let xq_b_old = xq[kb];
            let xq_b_new = dq_acc + xq_b_old * mold_o_mnew;
            qf = (xq_b_new - xq_a) / (xq_b_old - xq_a).max(1.0e-99);
            if qf > 0.0 || kb + 1 >= nz {
                break;
            }
            kb += 1;
        }
        (ka, kb, qf)
    } else {
        let lnt_lim_a = LN10 * mw.max_logt_for_k_below_const_q;
        let lnt_lim_b = LN10 * mw.max_logt_for_k_const_mass;
        let lnt_max =
            model.ln_t.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lnt_a = lnt_max.min(lnt_lim_a);
        let lnt_b = lnt_max.min(lnt_lim_b);
        let max_q_a = mold_o_mnew;

        let min_k_a = 4.min(nz.saturating_sub(2)).max(1);
        let mut ka = min_k_a;
        for k in min_k_a..(nz - 1) {
            ka = k;
            if 1.0 - xq[k] > max_q_a {
                continue;
            }
            if 1.0 - xq[k] <= mw.min_q_for_k_below_const_q
                || model.ln_t[k] >= lnt_a
            {
                break;
            }
        }
        let xq_a = xq[ka];

        let kb_start = (ka + 1).min(nz - 1);
        let mut kb = kb_start;
        for k in kb_start..nz {
            kb = k;
            if 1.0 - xq[k] > mw.max_q_for_k_const_mass {
                continue;
            }
            if 1.0 - xq[k] <= mw.min_q_for_k_const_mass
                || model.ln_t[k] >= lnt_b
            {
                break;
            }
        }

        let mut xq_b_old = xq[kb];
        let mut xq_b_new = dq_acc + xq_b_old * mold_o_mnew;
        let qf = loop {
            let qf_cur = (xq_b_new - xq_a) / (xq_b_old - xq_a).max(1.0e-99);
            if kb + 1 >= nz {
                break qf_cur;
            }
            if kb - ka > 10 {
                if qf_cur > 0.67 && qf_cur < 1.50 {
                    break qf_cur;
                }
                if qf_cur > 0.50
                    && qf_cur < 2.00
                    && dominant_species_index(model.cell_xa(kb))
                        != dominant_species_index(model.cell_xa(kb + 1))
                {
                    // `adjust_mass.f90:753`: `kB = max(1, kB-5)` (0-based `max(0, kb-5)`).
                    kb = kb.saturating_sub(5);
                    break qf_cur;
                }
            }
            kb += 1;
            xq_b_old = xq[kb];
            xq_b_new = dq_acc + xq_b_old * mold_o_mnew;
        };
        (ka, kb, qf)
    };

    if !qfrac.is_finite() || qfrac <= 0.0 {
        return Err(StarError::ConvergenceFailure(format!(
            "revise_q_and_dq: non-positive qfrac={qfrac} at kA={k_a}, kB={k_b}"
        )));
    }

    // `adjust_mass.f90:764-772`:
    // dq[0..k_a] unchanged; dq[k_a..k_b] *= qfrac; dq[k_b..nz] *= mold_o_mnew.
    for val in &mut dq[k_a..k_b] {
        *val *= qfrac;
    }
    for val in &mut dq[k_b..nz] {
        *val *= mold_o_mnew;
    }

    // Renormalize `dq` (`adjust_mass.f90:791-806`).
    let sum_dq = neumaier_sum(&dq);
    if !sum_dq.is_finite() || sum_dq <= 0.0 {
        return Err(StarError::ConvergenceFailure(format!(
            "revise_q_and_dq: invalid renorm sum(dq)={sum_dq}"
        )));
    }
    let inv_sum = 1.0 / sum_dq;
    for val in &mut dq {
        *val *= inv_sum;
    }

    // Set `q` from `dq` and close `dq[nz-1] = q[nz-1]` (`adjust_mass.f90:813-826`).
    // Accumulate with Neumaier compensation so `dq[nz - 1]` closes to machine
    // precision (`sum(dq) == 1.0`) without dumping naive-sum drift into cell `nz - 1`.
    model.dq = dq;
    model.q.resize(nz, 0.0);
    model.q[0] = 1.0;
    let mut sum_hi = model.dq[0];
    let mut sum_lo = 0.0_f64;
    for k in 1..(nz - 1) {
        model.q[k] = 1.0 - (sum_hi + sum_lo);
        let d = model.dq[k];
        let t = sum_hi + d;
        if sum_hi.abs() >= d.abs() {
            sum_lo += (sum_hi - t) + d;
        } else {
            sum_lo += (d - t) + sum_hi;
        }
        sum_hi = t;
    }
    model.q[nz - 1] = 1.0 - (sum_hi + sum_lo);
    model.dq[nz - 1] = model.q[nz - 1];
    model.set_xqs();

    let dm_target: Vec<f64> =
        model.dq.iter().map(|&d| d * new_xmstar).collect();
    Ok((k_a, k_b, dm_target))
}

#[allow(clippy::too_many_arguments)]
fn do_adjust_mass_core(
    model: &mut StarModel,
    old_mstar: f64,
    new_mstar: f64,
    old_xmstar: f64,
    new_xmstar_in: f64,
    delta_m: f64,
    dt: f64,
    x_accreted: Option<&[f64]>,
    explicit_ka_kb: Option<(usize, usize)>,
) -> Result<(), StarError> {
    let (nz, species) = (model.nz, model.species);
    if let Some(x_acc) = x_accreted {
        if x_acc.len() < species {
            return Err(StarError::InvalidInput(format!(
                "x_accreted length {} < species {species}",
                x_acc.len()
            )));
        }
    }
    if delta_m == 0.0 || nz < 3 {
        model.aux.k_const_mass = 0;
        model.eps_mdot.fill(0.0);
        return Ok(());
    }

    let frac = old_xmstar / new_xmstar_in;
    let new_xmstar = old_xmstar / frac;
    if !new_xmstar.is_finite() || new_xmstar <= 0.0 {
        return Err(StarError::ConvergenceFailure(format!(
            "do_adjust_mass produced non-positive envelope mass xm_new={new_xmstar}"
        )));
    }

    let need_eps_mdot =
        dt > 0.0 && model.controls.mix_wind.eps_mdot_factor != 0.0;
    if need_eps_mdot {
        model.reconstruct_ln_r_from_lnd_and_dm();
        model.set_m_grav_and_grav();
        set_basic_vars(model)?;
        do_eos(model)?;
    }
    let prev_dm_raw = model.dm.clone();
    let (
        prev_r,
        prev_m_grav,
        prev_energy,
        prev_peos,
        prev_rho,
        prev_te,
        prev_ke,
    ) = if need_eps_mdot {
        (
            model.r.clone(),
            model.m_grav.clone(),
            model.energy.clone(),
            model.peos.clone(),
            model.rho.clone(),
            eval_specific_total_energy_profile(model),
            eval_specific_kinetic_energy_profile(model),
        )
    } else {
        (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
    };

    let xa_old = model.xa.clone();
    let starting_j_rot_surf = model.j_rot.first().copied().unwrap_or(0.0);

    // Update `xmstar` and `mstar` (`adjust_mass.f90:264-265`).
    model.xmstar = new_xmstar;
    model.mstar = model.xmstar + model.m_center;

    let (_k_a, k_b, dm_target) = revise_q_and_dq_upstream(
        model,
        old_xmstar,
        new_xmstar,
        delta_m,
        explicit_ka_kb,
    )?;
    model.aux.k_const_mass = k_b;

    // `k_below_just_added` (`adjust_mass.f90:360-370`, 0-based).
    let k_below_just_added = if delta_m > 0.0 {
        let q_for_just_added = old_xmstar / new_xmstar;
        let mut kbja = nz - 1;
        for k in 0..nz {
            if model.q[k] <= q_for_just_added {
                kbja = k;
                break;
            }
        }
        kbja
    } else {
        0
    };
    model.k_below_just_added = k_below_just_added + 1;

    // Build exterior mass coordinates `rxm_old`, `rxm_new` (`adjust_mass.f90:397-419`).
    let mmax = old_mstar.max(new_mstar);
    let mut new_cell_mass = dm_target.clone();
    let mut rxm_old = vec![0.0_f64; nz];
    let mut rxm_new = vec![0.0_f64; nz];
    if delta_m < 0.0 {
        rxm_old[0] = 0.0;
        rxm_new[0] = -delta_m;
    } else {
        rxm_old[0] = delta_m;
        rxm_new[0] = 0.0;
    }
    for k in 1..nz {
        rxm_old[k] = rxm_old[k - 1] + prev_dm_raw[k - 1];
        if k >= k_b {
            rxm_new[k] = rxm_old[k];
            new_cell_mass[k] = prev_dm_raw[k];
        } else {
            rxm_new[k] = rxm_new[k - 1] + new_cell_mass[k - 1];
        }
    }

    let mut change_in_dm = vec![0.0_f64; nz];
    let mut prev_mesh_dm = vec![0.0_f64; nz];
    for k in 0..nz {
        change_in_dm[k] = if k < k_b {
            new_cell_mass[k] - prev_dm_raw[k]
        } else {
            0.0
        };
        prev_mesh_dm[k] = prev_dm_raw[k].max(1.0e-99);
    }

    sync_enclosed_mass_from_dm(model, &dm_target);
    model.set_dm_bar();

    // Accretion composition (`adjust_mass.f90:379-395`).
    let built_acc = if delta_m > 0.0 && x_accreted.is_none() {
        get_xa_for_accretion(model)?
    } else {
        vec![0.0_f64; species]
    };
    let x_acc = x_accreted.unwrap_or(&built_acc);

    // Conservative composition remap on `0..k_b` (`set_xa`, `adjust_mass.f90:421`).
    remap_xa_cells(
        model,
        &xa_old,
        x_acc,
        &rxm_old,
        &rxm_new,
        &prev_dm_raw,
        &new_cell_mass,
        k_b,
        mmax,
    );

    // Rotation remap (`set_omega_adjust_mass` + `adjust_J_lost`, `adjust_mass.f90:295-300, 430-452`).
    if model.controls.rotation_flag && model.j_rot.len() == nz {
        // Compute `angular_momentum_removed` from entry `model.j_rot` before
        // `set_omega_adjust_mass` overwrites `model.j_rot` (`adjust_mass.f90:295, 527-552`).
        let j_removed = if delta_m < 0.0 {
            compute_angular_momentum_removed(model, &prev_dm_raw, -delta_m)
        } else {
            0.0
        };
        set_omega_adjust_mass(
            model,
            k_b,
            k_below_just_added,
            delta_m,
            &rxm_old,
            &rxm_new,
            mmax,
            &prev_dm_raw,
            &new_cell_mass,
        )?;
        if model.controls.mix_wind.do_adjust_j_lost {
            adjust_j_lost(
                model,
                old_mstar,
                &new_cell_mass,
                starting_j_rot_surf,
                j_removed,
            )?;
        }
    }
    let j_rot_after_remap = if model.controls.rotation_flag {
        model.j_rot.clone()
    } else {
        Vec::new()
    };

    // Soften outer `xa` (`adjust_mass.f90:480-510`).
    smooth_outer_xa(model);

    model.reconstruct_ln_r_from_lnd_and_dm();
    if model.ln_r_start.len() == nz {
        model.ln_r_start.clone_from(&model.ln_r);
    }
    model.set_m_grav_and_grav();
    model.pack_xh();
    if model.xh_start.len() == model.xh.len() {
        let nv = model.layout.nvar_hydro;
        let i_lnr = model.layout.i_lnr;
        for k in 0..nz {
            model.xh_start[k * nv + i_lnr] = model.xh[k * nv + i_lnr];
        }
    }
    if model.controls.rotation_flag {
        conserve_shell_j_rot_and_distortion(model);
    }

    set_basic_vars(model)?;
    sync_enclosed_mass_from_dm(model, &dm_target);
    model.set_dm_bar();
    model.set_m_grav_and_grav();
    if need_eps_mdot {
        set_hydro_vars(model, false, false)?;
        sync_enclosed_mass_from_dm(model, &dm_target);
        model.set_dm_bar();
        if model.controls.rotation_flag {
            model.j_rot.clone_from(&j_rot_after_remap);
            conserve_shell_j_rot_and_distortion(model);
        }
        let prev = PrevMassState {
            dm: &prev_mesh_dm,
            r: &prev_r,
            m_grav: &prev_m_grav,
            energy: &prev_energy,
            peos: &prev_peos,
            rho: &prev_rho,
            te: &prev_te,
            ke: &prev_ke,
        };
        calculate_eps_mdot_with_change(model, &prev, &change_in_dm, dt);
    } else {
        model.eps_mdot.fill(0.0);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn remap_xa_cells(
    model: &mut StarModel,
    xa_old: &[f64],
    x_acc: &[f64],
    rxm_old: &[f64],
    rxm_new: &[f64],
    old_cell_mass: &[f64],
    new_cell_mass: &[f64],
    k_const_mass: usize,
    mmax: f64,
) {
    let (nz, species) = (model.nz, model.species);
    if species == 0 {
        return;
    }
    let mut accum = vec![0.0_f64; species];
    let mut kk_start = 0_usize;
    let m_top = rxm_old[0];

    for k in 0..k_const_mass {
        let a0 = rxm_new[k];
        let new_dm = new_cell_mass[k];
        let a1 = a0 + new_dm;
        if a1 <= m_top {
            model.cell_xa_mut(k).copy_from_slice(&x_acc[..species]);
            continue;
        }
        accum.fill(0.0);
        if a0 < m_top {
            let w_acc = (m_top - a0).min(new_dm);
            for j in 0..species {
                accum[j] += w_acc * x_acc[j];
            }
        }
        let a_eff = a0.max(m_top);
        while kk_start + 1 < nz && rxm_old[kk_start + 1] <= a_eff {
            kk_start += 1;
        }
        for kk in kk_start..nz {
            let b0 = rxm_old[kk];
            if b0 >= a1 {
                break;
            }
            let b1 = if kk + 1 < nz {
                rxm_old[kk + 1]
            } else {
                (mmax - model.m_center).max(b0 + old_cell_mass[kk])
            };
            let ov = (a1.min(b1) - a_eff.max(b0)).max(0.0);
            if ov > 0.0 {
                for j in 0..species {
                    accum[j] += ov * xa_old[kk * species + j];
                }
            }
        }
        normalize_and_assign_remapped_xa(
            &mut model.xa[k * species..(k + 1) * species],
            &accum,
            new_dm.max(1.0e-99),
        );
    }
}

fn set1_irot(model: &mut StarModel, k: usize, jrot_known: bool) {
    let r00 = model.r[k].max(1.0e5);
    let m_k = model.m[k].max(1.0e20);
    let cgrav_k = model.cgrav[k];
    let max_w = model.controls.mix_wind.w_div_wcrit_max.clamp(0.0, 0.99);
    let om_c = eval_omega_crit(cgrav_k, m_k, r00, 0.0, 0.0, 0.99);

    let i_rot_k = if jrot_known {
        let j_target = model.j_rot[k];
        let mut ir = model
            .aux
            .i_rot
            .get(k)
            .copied()
            .filter(|&v| v > 0.0)
            .unwrap_or_else(|| (2.0 / 3.0) * r00 * r00);
        for _ in 0..12 {
            let om = j_target / ir.max(1.0e-30);
            let w = if om_c > 0.0 {
                (om.abs() / om_c).min(max_w)
            } else {
                0.0
            };
            let (_, _, i_r2) = eval_rotation_fp_ft(w, 0.75, 0.95);
            let ir_new = (i_r2 * r00 * r00).max(1.0e-30);
            let rel = (ir_new - ir).abs() / ir_new;
            ir = ir_new;
            if rel < 1.0e-14 {
                break;
            }
        }
        ir
    } else {
        let om = model.omega[k];
        let w = if om_c > 0.0 {
            (om.abs() / om_c).min(max_w)
        } else {
            0.0
        };
        let (_, _, i_r2) = eval_rotation_fp_ft(w, 0.75, 0.95);
        (i_r2 * r00 * r00).max(1.0e-30)
    };

    if k < model.aux.i_rot.len() {
        model.aux.i_rot[k] = i_rot_k;
    }
}

#[allow(clippy::too_many_arguments)]
fn set_omega_adjust_mass(
    model: &mut StarModel,
    k_const_mass: usize,
    k_below_just_added: usize,
    delta_m: f64,
    old_cell_xbdy: &[f64],
    new_cell_xbdy: &[f64],
    mmax: f64,
    old_cell_mass: &[f64],
    new_cell_mass: &[f64],
) -> Result<(), StarError> {
    let nz = model.nz;
    let mut old_xout = vec![0.0_f64; nz];
    let mut new_xout = vec![0.0_f64; nz];
    let mut old_dmbar = vec![0.0_f64; nz];
    let mut new_dmbar = vec![0.0_f64; nz];
    let old_j_rot = model.j_rot.clone();

    old_xout[0] = old_cell_xbdy[0];
    new_xout[0] = new_cell_xbdy[0];
    old_dmbar[0] = 0.5 * old_cell_mass[0];
    new_dmbar[0] = 0.5 * new_cell_mass[0];
    for k in 1..nz {
        old_xout[k] = old_xout[k - 1] + old_dmbar[k - 1];
        new_xout[k] = new_xout[k - 1] + new_dmbar[k - 1];
        old_dmbar[k] = 0.5 * (old_cell_mass[k - 1] + old_cell_mass[k]);
        new_dmbar[k] = 0.5 * (new_cell_mass[k - 1] + new_cell_mass[k]);
    }
    old_dmbar[nz - 1] = 0.5 * old_cell_mass[nz - 2] + old_cell_mass[nz - 1];
    new_dmbar[nz - 1] = 0.5 * new_cell_mass[nz - 2] + new_cell_mass[nz - 1];

    let old_j_tot: f64 = old_j_rot
        .iter()
        .zip(&old_dmbar)
        .map(|(&j, &dm)| j * dm)
        .sum();

    // `adjust_mass.f90:1156-1171`: 0-based `k_below_just_added == 0` corresponds
    // to 1-based `k_below_just_added == 1`.
    let k0 = if k_below_just_added == 0 {
        0
    } else {
        k_below_just_added + 1
    };
    let k_end = k_const_mass.min(nz - 1);
    for k in k0..=k_end {
        set1_omega(
            model, k, &old_xout, &new_xout, mmax, &old_dmbar, &new_dmbar,
            &old_j_rot,
        )?;
    }

    // `adjust_mass.f90:1173-1206`: set omega / j_rot in newly added outer shells.
    if k_below_just_added > 0 {
        let mw = &model.controls.mix_wind;
        if mw.use_accreted_material_j {
            let acc_j = mw.accreted_material_j;
            let outer_end = k_below_just_added.saturating_sub(1);
            for k in 0..outer_end {
                model.j_rot[k] = acc_j;
                set1_irot(model, k, true);
                model.omega[k] =
                    model.j_rot[k] / model.aux.i_rot[k].max(1.0e-30);
            }
            let goal_total = old_j_tot + delta_m * acc_j;
            let inner_total: f64 = ((k_below_just_added + 1)..nz)
                .map(|k| model.j_rot[k] * new_dmbar[k])
                .sum();
            let outer_total: f64 =
                (0..outer_end).map(|k| model.j_rot[k] * new_dmbar[k]).sum();
            let bdy_total = goal_total - (inner_total + outer_total);
            let bdy_dm = new_dmbar[k_below_just_added - 1]
                + new_dmbar[k_below_just_added];
            let bdy_j = bdy_total / bdy_dm;
            if !bdy_j.is_finite() || bdy_j <= 0.0 {
                return Err(StarError::ConvergenceFailure(format!(
                    "set_omega_adjust_mass: non-positive boundary j_rot={bdy_j}"
                )));
            }
            for k in (k_below_just_added - 1)..=k_below_just_added {
                model.j_rot[k] = bdy_j;
                set1_irot(model, k, true);
                model.omega[k] =
                    model.j_rot[k] / model.aux.i_rot[k].max(1.0e-30);
            }
        } else {
            // Default (`adjust_mass.f90:1199-1205`): assign old surface angular
            // velocity `omega(k_below_just_added)` to all newly added shells and
            // compute `i_rot` from `(r, m, omega)`.
            let om_ref = model.omega[k_below_just_added];
            for k in 0..k_below_just_added {
                model.omega[k] = om_ref;
                set1_irot(model, k, false);
                model.j_rot[k] = model.omega[k] * model.aux.i_rot[k];
            }
        }
    }

    for (k, &om) in model.omega.iter().enumerate() {
        if !om.is_finite() || om.abs() > 1.0e50 {
            return Err(StarError::ConvergenceFailure(format!(
                "set_omega_adjust_mass: invalid omega[{k}]={om}"
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn set1_omega(
    model: &mut StarModel,
    k: usize,
    old_xout: &[f64],
    new_xout: &[f64],
    mmax: f64,
    old_dmbar: &[f64],
    new_dmbar: &[f64],
    old_j_rot: &[f64],
) -> Result<(), StarError> {
    let nz = model.nz;
    let mut xm_outer = new_xout[k];
    let new_point_dmbar = if k + 1 == nz {
        mmax - xm_outer - model.m_center
    } else {
        new_dmbar[k]
    };
    let xm_inner = xm_outer + new_point_dmbar;
    let mut dm_sum = 0.0_f64;
    let mut j_tot = 0.0_f64;

    let k_outer = if xm_outer < old_xout[0] {
        if xm_inner <= old_xout[0] {
            model.omega[k] = 0.0;
            return Ok(());
        }
        let dm = new_point_dmbar.min(old_xout[0] - xm_outer);
        dm_sum = dm;
        xm_outer = old_xout[0];
        0
    } else if xm_outer >= old_xout[nz - 1] {
        nz - 1
    } else {
        // Largest `idx` with `old_xout[idx] <= xm_outer` and `old_xout[idx+1] > xm_outer`.
        old_xout
            .partition_point(|&x| x <= xm_outer)
            .saturating_sub(1)
            .min(nz - 1)
    };

    for kk in k_outer..nz {
        let xm0 = old_xout[kk];
        if xm0 >= xm_inner {
            if dm_sum < new_point_dmbar && kk > 0 {
                let dm = new_point_dmbar - dm_sum;
                dm_sum = new_point_dmbar;
                j_tot += old_j_rot[kk - 1] * dm;
            }
            break;
        }
        let xm1 = if kk + 1 == nz {
            mmax - model.m_center
        } else {
            old_xout[kk + 1]
        };
        if xm1 < xm_outer {
            return Err(StarError::ConvergenceFailure(String::from(
                "set1_omega: xm1 < xm_outer",
            )));
        }
        let dm = if xm0 >= xm_outer && xm1 <= xm_inner {
            let mut d = old_dmbar[kk];
            dm_sum += d;
            if dm_sum > new_point_dmbar {
                // Preserves `adjust_mass.f90:1387` (`#244` pattern).
                d -= new_point_dmbar - dm_sum;
                dm_sum = new_point_dmbar;
            }
            d
        } else if xm0 <= xm_outer && xm1 >= xm_inner {
            let d = new_dmbar[k];
            dm_sum += d;
            d
        } else if xm_inner <= xm1 {
            let d = new_point_dmbar - dm_sum;
            dm_sum = new_point_dmbar;
            d
        } else {
            let mut d = (xm1 - xm_outer).max(0.0);
            if dm_sum + d > new_point_dmbar {
                d = new_point_dmbar - dm_sum;
            }
            dm_sum += d;
            d
        };
        j_tot += old_j_rot[kk] * dm;
        if dm_sum >= new_point_dmbar {
            dm_sum = new_point_dmbar;
            break;
        }
    }

    if dm_sum != new_point_dmbar {
        if k + 1 == nz {
            j_tot += old_j_rot[nz - 1] * (new_point_dmbar - dm_sum);
        } else {
            return Err(StarError::ConvergenceFailure(format!(
                "set1_omega: dm_sum ({dm_sum}) != new_point_dmbar ({new_point_dmbar}) at k={k}"
            )));
        }
    }

    model.j_rot[k] = j_tot / new_point_dmbar;
    set1_irot(model, k, true);
    model.omega[k] = model.j_rot[k] / model.aux.i_rot[k].max(1.0e-30);
    Ok(())
}

/// Integrates angular momentum removed by mass loss using entry `model.j_rot`
/// (`angular_momentum_removed` in `star/private/adjust_mass.f90:527-552`).
fn compute_angular_momentum_removed(
    model: &StarModel,
    old_cell_mass: &[f64],
    dm_lost: f64,
) -> f64 {
    if !model.controls.rotation_flag || dm_lost <= 0.0 {
        return 0.0;
    }
    let nz = model.nz;
    let mut dm00 = 0.0_f64;
    let mut dm_sum = 0.0_f64;
    let mut j_tot = 0.0_f64;
    for k in 0..nz {
        let dmm1 = dm00;
        dm00 = old_cell_mass[k];
        let mut dm_k = 0.5 * (dmm1 + dm00);
        if dm_sum + dm_k > dm_lost {
            dm_k = dm_lost - dm_sum;
            dm_sum = dm_lost;
        } else {
            dm_sum += dm_k;
        }
        j_tot += dm_k * model.j_rot[k];
        if dm_sum == dm_lost {
            break;
        }
    }
    j_tot
}

fn adjust_j_lost(
    model: &mut StarModel,
    old_mstar: f64,
    new_dm: &[f64],
    starting_j_rot_surf: f64,
    j_removed: f64,
) -> Result<(), StarError> {
    let nz = model.nz;
    let mass_lost = old_mstar - model.mstar;
    if mass_lost <= 0.0 {
        return Ok(());
    }

    let mw = &model.controls.mix_wind;
    let actual_j_lost = mw.adjust_j_fraction * mass_lost * starting_j_rot_surf
        + (1.0 - mw.adjust_j_fraction) * j_removed;
    let delta_j = actual_j_lost - j_removed;
    if delta_j == 0.0 {
        return Ok(());
    }

    let mut dm00 = 0.0_f64;
    let mut j_sum = 0.0_f64;
    let mut last_k = nz - 1;
    for k in 0..nz {
        let dmm1 = dm00;
        dm00 = new_dm[k];
        let dm_f = 0.5 * (dmm1 + dm00);
        j_sum += dm_f * model.j_rot[k];
        if j_sum < mw.min_j_div_delta_j * delta_j.abs()
            || model.q[k] > mw.min_q_for_adjust_j_lost
        {
            continue;
        }
        last_k = k;
        break;
    }
    if last_k + 1 >= nz {
        return Err(StarError::ConvergenceFailure(String::from(
            "adjust_J_lost: last_k == nz",
        )));
    }
    let qlast = model.q[last_k];
    model.adjust_j_q = qlast;
    let mut dm00 = 0.0_f64;
    let mut denom = 0.0_f64;
    for k in 0..last_k {
        let dmm1 = dm00;
        dm00 = new_dm[k];
        let dm_f = 0.5 * (dmm1 + dm00);
        denom += (model.q[k] - qlast) * model.j_rot[k] * dm_f;
    }
    if denom.abs() > 1.0e-99 {
        let frac = -delta_j / denom;
        for k in 0..last_k {
            model.j_rot[k] *= frac * (model.q[k] - qlast) + 1.0;
            set1_irot(model, k, true);
            model.omega[k] = model.j_rot[k] / model.aux.i_rot[k].max(1.0e-30);
        }
    }
    Ok(())
}

/// Outer-envelope composition smoothing (`smooth_outer_xa_big` /
/// `smooth_outer_xa_small` in `star/private/adjust_mass.f90:480-510`).
fn smooth_outer_xa(model: &mut StarModel) {
    let mw = &model.controls.mix_wind;
    let big = mw.smooth_outer_xa_big;
    let small = mw.smooth_outer_xa_small;
    let (nz, species) = (model.nz, model.species);
    if big <= 0.0 || small <= 0.0 || nz == 0 || species == 0 {
        return;
    }
    let m1 = model.m[0];
    let mut m_end = 0_usize;
    for k in 0..nz {
        if m1 - model.m[k] > big * m1 {
            break;
        }
        let mut region_total_mass = 0.0_f64;
        m_end = k;
        let limit = small * m1 * (1.0 - (m1 - model.m[k]) / (big * m1));
        for l in k..nz {
            if model.m[k] - model.m[l] > limit {
                break;
            }
            m_end = l;
            region_total_mass += model.dm[l];
        }
        if m_end == k || region_total_mass <= 0.0 {
            continue;
        }
        for j in 0..species {
            let mut partial_xa_mass = 0.0_f64;
            for l in k..=m_end {
                partial_xa_mass += model.dm[l] * model.xa[l * species + j];
            }
            let avg = partial_xa_mass / region_total_mass;
            for l in k..=m_end {
                model.xa[l * species + j] = avg;
            }
        }
    }
    for k in 0..=m_end.min(nz - 1) {
        let cell = &mut model.xa[k * species..(k + 1) * species];
        let sum_x: f64 = cell.iter().sum();
        if sum_x > 0.0 {
            let inv = 1.0 / sum_x;
            for x in cell.iter_mut() {
                *x *= inv;
            }
        }
    }
}
