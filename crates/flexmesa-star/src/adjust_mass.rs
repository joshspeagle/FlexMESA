// SPDX-License-Identifier: LGPL-3.0-or-later
//! Three-region pseudo-Lagrangian mass adjustment (`do_adjust_mass` in
//! `star/private/adjust_mass.f90`), accreted composition construction
//! (`accrete_given_mass_fractions`), angular momentum accretion, and coupled
//! $\epsilon_{\dot{M}}$ evaluation.

use flexmesa_chem::NuclideCatalog;
use flexmesa_num::neumaier_sum;

use crate::controls::StarError;
use crate::core_boundaries::dominant_species_index;
use crate::eps_mdot::{
    calculate_eps_mdot_with_change, eval_specific_kinetic_energy_profile,
    eval_specific_total_energy_profile, PrevMassState,
};
use crate::hydro_vars::set_hydro_vars;
use crate::mesh_remap::normalize_and_assign_remapped_xa;
use crate::micro::{do_eos, set_basic_vars};
use crate::model::StarModel;
use crate::rotation::conserve_shell_j_rot_and_distortion;

fn sync_enclosed_mass_from_dm(model: &mut StarModel, dm_target: &[f64]) {
    model.dm.copy_from_slice(dm_target);
    let mut m_acc = model.m_center;
    for k in (1..model.nz).rev() {
        m_acc += model.dm[k];
        model.m[k] = m_acc;
    }
    model.m[0] = model.mstar;
}

fn normalize_or_surface(mut xa: Vec<f64>, model: &StarModel) -> Vec<f64> {
    let sum: f64 = xa.iter().map(|&x| x.max(0.0)).sum();
    if sum > 0.0 {
        for x in &mut xa {
            *x = x.max(0.0) / sum;
        }
        xa
    } else {
        model.cell_xa(0).to_vec()
    }
}

/// Builds the normalized accreted mass-fraction vector `x_accreted` (`len =
/// model.species`) from `model.controls` (`set_accretion_composition` in
/// `adjust_mass.f90:382-395` and `get_xa_for_accretion` in `adjust_xyz.f90`).
#[must_use]
pub fn build_accreted_xa(model: &StarModel) -> Vec<f64> {
    let species = model.species;
    if species == 0 {
        return Vec::new();
    }
    if model.controls.accrete_same_as_surface {
        return model.cell_xa(0).to_vec();
    }

    if model.controls.accrete_given_mass_fractions {
        let has_custom = model
            .controls
            .accretion_species_id
            .iter()
            .zip(&model.controls.accretion_species_xa)
            .any(|(iso, &val)| {
                val >= 0.0
                    && !iso.is_empty()
                    && model
                        .iso_names
                        .iter()
                        .any(|n| n.eq_ignore_ascii_case(iso))
            });
        let mut xa = if has_custom {
            vec![0.0; species]
        } else {
            model.cell_xa(0).to_vec()
        };
        for (iso, &val) in model
            .controls
            .accretion_species_id
            .iter()
            .zip(&model.controls.accretion_species_xa)
        {
            if val >= 0.0 && !iso.is_empty() {
                if let Some(j) = model
                    .iso_names
                    .iter()
                    .position(|n| n.eq_ignore_ascii_case(iso))
                {
                    xa[j] = val;
                }
            }
        }
        return normalize_or_surface(xa, model);
    }

    let (h1, he4) = (
        model.controls.accretion_h1.max(0.0),
        model.controls.accretion_he4.max(0.0),
    );
    let z_rem = (1.0 - h1 - he4).max(0.0);
    let catalog = NuclideCatalog::builtin_ref();
    let surf_xa = model.cell_xa(0);

    let mut is_metal = vec![false; species];
    let (mut metal_count, mut surf_z_sum) = (0usize, 0.0_f64);
    for j in 0..species {
        let z = model
            .chem_id
            .get(j)
            .and_then(|&cid| catalog.get_by_id(cid))
            .map_or(0, |e| e.z);
        if z > 2 {
            is_metal[j] = true;
            metal_count += 1;
            surf_z_sum += surf_xa[j].max(0.0);
        }
    }

    let mut xa = vec![0.0_f64; species];
    for j in 0..species {
        let name = model.iso_names.get(j).map_or("", String::as_str);
        if name.eq_ignore_ascii_case("h1") {
            xa[j] = h1;
        } else if name.eq_ignore_ascii_case("he4") {
            xa[j] = he4;
        } else if name.eq_ignore_ascii_case("h2")
            || name.eq_ignore_ascii_case("he3")
        {
            xa[j] = 0.0;
        } else if is_metal[j] {
            xa[j] = if surf_z_sum > 1.0e-30 {
                surf_xa[j].max(0.0) * (z_rem / surf_z_sum)
            } else {
                z_rem / (metal_count.max(1) as f64)
            };
        }
    }
    normalize_or_surface(xa, model)
}

/// Applies the 3-region pseudo-Lagrangian mass adjustment (`do_adjust_mass` in
/// `star/private/adjust_mass.f90:73-1131`).
///
/// # Errors
/// Returns [`StarError`] if `xm_new <= 0` or microphysics fails.
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
/// Returns [`StarError`] if `xm_new <= 0` or microphysics fails.
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
    let mut k_a = 1_usize;
    while k_a + 2 < nz && model.q[k_a] > q_below_const_q {
        k_a += 1;
    }
    let mut k_b = (k_a + 1).min(nz - 1);
    while k_b + 1 < nz && model.q[k_b] > q_const_mass {
        k_b += 1;
    }
    do_adjust_mass_regions(model, delta_m, dt, x_accreted, k_a, k_b)
}

fn do_adjust_mass_regions(
    model: &mut StarModel,
    delta_m: f64,
    dt: f64,
    x_accreted: Option<&[f64]>,
    k_a: usize,
    k_b: usize,
) -> Result<(), StarError> {
    model.aux.k_const_mass = k_b;
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
        model.eps_mdot.fill(0.0);
        return Ok(());
    }
    let xm_old = model.xmstar;
    let xm_new = xm_old + delta_m;
    if xm_new <= 0.0 {
        // Upstream sets `ierr = -1` here (`adjust_mass.f90:260-263`), which
        // the step driver turns into a retry with a smaller timestep
        // (`evolve.f90:591`). The error is therefore a retriable one.
        return Err(StarError::ConvergenceFailure(format!(
            "do_adjust_mass produced non-positive envelope mass xm_new={xm_new}"
        )));
    }
    let k_a = k_a.clamp(1, nz.saturating_sub(2));
    let k_b = k_b.clamp(k_a + 1, nz - 1);

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

    let sum_prev_dm = neumaier_sum(&prev_dm_raw).max(1.0e-99);
    let dq_old: Vec<f64> =
        prev_dm_raw.iter().map(|&d| d / sum_prev_dm).collect();
    let mut xq_old = vec![0.0_f64; nz];
    for k in 1..nz {
        xq_old[k] = xq_old[k - 1] + dq_old[k - 1];
    }
    let xa_old = model.xa.clone();
    let j_rot_old = model.j_rot.clone();
    let starting_j_rot_surf = j_rot_old.first().copied().unwrap_or(0.0);

    let dq_acc_raw = delta_m / xm_new;
    let dxq_ii = (xq_old[k_b] - xq_old[k_a]).max(1.0e-99);
    let qfrac = 1.0 + dq_acc_raw * (1.0 - xq_old[k_b]) / dxq_ii;
    let uniform_fallback = qfrac <= 0.0;
    let k_b_remap = if uniform_fallback { nz } else { k_b };

    let mut dm_target = prev_dm_raw.clone();
    let old_outer_mass = neumaier_sum(&prev_dm_raw[..k_b_remap]);
    let new_outer_mass = (old_outer_mass + delta_m).max(1.0e-99);
    let scale_i = xm_new / sum_prev_dm;
    if uniform_fallback {
        for k in 0..nz {
            dm_target[k] = (prev_dm_raw[k] * scale_i).max(1.0e-99);
        }
    } else {
        for k in 0..k_a {
            dm_target[k] = (prev_dm_raw[k] * scale_i).max(1.0e-99);
        }
        let sum_i = neumaier_sum(&dm_target[..k_a]);
        let target_ii = (new_outer_mass - sum_i).max(1.0e-99);
        let old_ii = neumaier_sum(&prev_dm_raw[k_a..k_b]).max(1.0e-99);
        let scale_ii = target_ii / old_ii;
        for k in k_a..k_b {
            dm_target[k] = (prev_dm_raw[k] * scale_ii).max(1.0e-99);
        }
    }
    let dm_resid = new_outer_mass - neumaier_sum(&dm_target[..k_b_remap]);
    if dm_target[k_b_remap - 1] + dm_resid > 1.0e-99 {
        dm_target[k_b_remap - 1] += dm_resid;
    }

    let mut dq_new: Vec<f64> = dm_target
        .iter()
        .map(|&d| (d / xm_new).max(1.0e-99))
        .collect();
    let dq_resid = 1.0 - neumaier_sum(&dq_new);
    if dq_new[k_b_remap - 1] + dq_resid > 1.0e-99 {
        dq_new[k_b_remap - 1] += dq_resid;
    }

    let mut change_in_dm = vec![0.0_f64; nz];
    let mut prev_mesh_dm = vec![0.0_f64; nz];
    for k in 0..nz {
        change_in_dm[k] = if k < k_b_remap {
            dm_target[k] - prev_dm_raw[k]
        } else {
            0.0
        };
        prev_mesh_dm[k] = prev_dm_raw[k].max(1.0e-99);
    }

    model.mstar += delta_m;
    model.xmstar = xm_new;
    model.dq = dq_new;
    model.set_qs();
    model.set_xqs();
    sync_enclosed_mass_from_dm(model, &dm_target);
    model.set_dm_bar();

    let built_acc = build_accreted_xa(model);
    let x_acc = x_accreted.unwrap_or(&built_acc);
    let j_acc = if model.controls.mix_wind.use_accreted_material_j {
        model.controls.mix_wind.accreted_material_j
    } else {
        starting_j_rot_surf
    };

    let mut cum_old = vec![0.0_f64; k_b_remap + 1];
    let mut cum_new = vec![0.0_f64; k_b_remap + 1];
    for k in 0..k_b_remap {
        cum_old[k + 1] = cum_old[k] + prev_dm_raw[k];
        cum_new[k + 1] = cum_new[k] + dm_target[k];
    }
    let bdy_resid = (cum_old[k_b_remap] + delta_m) - cum_new[k_b_remap];
    if dm_target[k_b_remap - 1] + bdy_resid > 1.0e-99 {
        dm_target[k_b_remap - 1] += bdy_resid;
        cum_new[k_b_remap] = cum_old[k_b_remap] + delta_m;
        sync_enclosed_mass_from_dm(model, &dm_target);
        model.set_dm_bar();
    }

    let mut accum = vec![0.0; species];
    let mut kk_start = 0_usize;
    for k in 0..k_b_remap {
        let x_00 = cum_new[k] - delta_m;
        let x_p1 = cum_new[k + 1] - delta_m;
        let dx = (x_p1 - x_00).max(1.0e-99);

        if x_p1 <= 0.0 {
            model.cell_xa_mut(k).copy_from_slice(&x_acc[..species]);
            if model.controls.rotation_flag && delta_m > 0.0 {
                model.j_rot[k] = j_acc;
            }
        } else {
            accum.fill(0.0);
            let mut jr_acc = 0.0;
            if x_00 < 0.0 {
                let w_acc = -x_00;
                for j in 0..species {
                    accum[j] += w_acc * x_acc[j];
                }
                jr_acc += w_acc * j_acc;
            }
            let a_start = x_00.max(0.0);
            while kk_start + 1 < k_b_remap && cum_old[kk_start + 1] <= a_start {
                kk_start += 1;
            }
            for kk in kk_start..k_b_remap {
                let b0 = cum_old[kk];
                if b0 >= x_p1 {
                    break;
                }
                let b1 = cum_old[kk + 1];
                let ov = (x_p1.min(b1) - a_start.max(b0)).max(0.0);
                if ov > 0.0 {
                    for j in 0..species {
                        accum[j] += ov * xa_old[kk * species + j];
                    }
                    jr_acc += ov * j_rot_old.get(kk).copied().unwrap_or(0.0);
                }
            }
            normalize_and_assign_remapped_xa(
                &mut model.xa[k * species..(k + 1) * species],
                &accum,
                dx,
            );
            if model.controls.rotation_flag && delta_m > 0.0 {
                model.j_rot[k] = jr_acc / dx;
            }
        }
    }

    if model.controls.rotation_flag && delta_m < 0.0 && j_rot_old.len() == nz {
        remap_j_rot_and_adjust_j_lost(
            model,
            &prev_dm_raw,
            &dm_target,
            &j_rot_old,
            -delta_m,
            starting_j_rot_surf,
            k_b_remap,
        )?;
    }

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
        if model.controls.rotation_flag {
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

fn remap_j_rot_and_adjust_j_lost(
    model: &mut StarModel,
    old_dm: &[f64],
    new_dm: &[f64],
    j_rot_old: &[f64],
    mass_lost: f64,
    starting_j_rot_surf: f64,
    k_b_remap: usize,
) -> Result<(), StarError> {
    let nz = model.nz;
    let build_dmbar = |dm: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0_f64; nz];
        out[0] = 0.5 * dm[0];
        for k in 1..(nz - 1) {
            out[k] = 0.5 * (dm[k - 1] + dm[k]);
        }
        out[nz - 1] = 0.5 * dm[nz - 2] + dm[nz - 1];
        out
    };
    let old_dmbar = build_dmbar(old_dm);
    let new_dmbar = build_dmbar(new_dm);

    let (mut j_removed, mut dm_sum) = (0.0_f64, 0.0_f64);
    for k in 0..nz {
        let mut dm_k = old_dmbar[k];
        if dm_sum + dm_k >= mass_lost {
            dm_k = (mass_lost - dm_sum).max(0.0);
            j_removed += dm_k * j_rot_old[k];
            break;
        }
        dm_sum += dm_k;
        j_removed += dm_k * j_rot_old[k];
    }

    let mut old_xout = vec![0.0_f64; nz];
    let mut new_xout = vec![0.0_f64; nz];
    new_xout[0] = mass_lost;
    for k in 1..nz {
        old_xout[k] = old_xout[k - 1] + old_dmbar[k - 1];
        new_xout[k] = if k >= k_b_remap {
            old_xout[k]
        } else {
            new_xout[k - 1] + new_dmbar[k - 1]
        };
    }

    let mut kk_start = 0_usize;
    for k in 0..k_b_remap {
        let a0 = new_xout[k];
        let a1 = a0 + new_dmbar[k];
        let dx = new_dmbar[k].max(1.0e-99);
        while kk_start + 1 < nz
            && old_xout[kk_start] + old_dmbar[kk_start] <= a0
        {
            kk_start += 1;
        }
        let mut jr_acc = 0.0_f64;
        for kk in kk_start..nz {
            let b0 = old_xout[kk];
            if b0 >= a1 {
                break;
            }
            let b1 = b0 + old_dmbar[kk];
            let ov = (a1.min(b1) - a0.max(b0)).max(0.0);
            jr_acc += ov * j_rot_old[kk];
        }
        model.j_rot[k] = jr_acc / dx;
    }

    let mw = &model.controls.mix_wind;
    if !mw.do_adjust_j_lost {
        return Ok(());
    }
    let f_adj = mw.adjust_j_fraction.clamp(0.0, 1.0);
    let actual_j_lost =
        f_adj * mass_lost * starting_j_rot_surf + (1.0 - f_adj) * j_removed;
    let delta_j = actual_j_lost - j_removed;
    if delta_j == 0.0 {
        return Ok(());
    }

    let (mut j_sum, mut last_k) = (0.0_f64, nz);
    for k in 0..nz {
        let dm_f = if k == 0 {
            0.5 * new_dm[0]
        } else {
            0.5 * (new_dm[k - 1] + new_dm[k])
        };
        j_sum += dm_f * model.j_rot[k];
        if j_sum >= mw.min_j_div_delta_j * delta_j.abs()
            && model.q[k] <= mw.min_q_for_adjust_j_lost
        {
            last_k = k;
            break;
        }
    }
    if last_k >= nz - 1 {
        return Err(StarError::ConvergenceFailure(String::from(
            "adjust_J_lost: last_k == nz",
        )));
    }
    let qlast = model.q[last_k];
    let denom: f64 = (0..last_k)
        .map(|k| {
            let dm_f = if k == 0 {
                0.5 * new_dm[0]
            } else {
                0.5 * (new_dm[k - 1] + new_dm[k])
            };
            (model.q[k] - qlast) * model.j_rot[k] * dm_f
        })
        .sum();
    if denom.abs() > 1.0e-99 {
        let frac = -delta_j / denom;
        for k in 0..last_k {
            model.j_rot[k] *= 1.0 + frac * (model.q[k] - qlast);
        }
    }
    Ok(())
}

/// Applies control-driven mass adjustment (`model.mstar_dot * dt`) using
/// upstream `revise_q_and_dq` (`adjust_mass.f90:650-761`) region selection.
///
/// # Errors
/// Returns [`StarError`] if envelope mass or EOS fails.
pub fn do_adjust_mass_step(
    model: &mut StarModel,
    dt: f64,
) -> Result<(), StarError> {
    let mdot = model.mstar_dot;
    let nz = model.nz;
    let delta_m = mdot * dt;
    if delta_m == 0.0 || dt <= 0.0 || nz < 3 {
        model.aux.k_const_mass = 0;
        model.eps_mdot.fill(0.0);
        return Ok(());
    }
    let xm_old = model.xmstar;
    let xm_new = xm_old + delta_m;
    if xm_new <= 0.0 {
        // Upstream sets `ierr = -1` here (`adjust_mass.f90:260-263`), which
        // the step driver turns into a retry with a smaller timestep
        // (`evolve.f90:591`). The error is therefore a retriable one.
        return Err(StarError::ConvergenceFailure(format!(
            "do_adjust_mass_step produced non-positive envelope mass xm_new={xm_new}"
        )));
    }

    let mw = &model.controls.mix_wind;
    let sum_dq = model.dq.iter().sum::<f64>().max(1.0e-99);
    let dq_old: Vec<f64> = model.dq.iter().map(|&d| d / sum_dq).collect();
    let mut xq_old = vec![0.0_f64; nz];
    for k in 1..nz {
        xq_old[k] = xq_old[k - 1] + dq_old[k - 1];
    }

    let dq_acc = delta_m / xm_new;
    let mold_o_mnew = xm_old / xm_new;
    let lnt_max = model.ln_t.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let lnt_a =
        lnt_max.min(flexmesa_const::LN10 * mw.max_logt_for_k_below_const_q);
    let lnt_b =
        lnt_max.min(flexmesa_const::LN10 * mw.max_logt_for_k_const_mass);
    let max_q_a = mw.max_q_for_k_below_const_q.min(mold_o_mnew);

    let min_k_a = 4.min(nz.saturating_sub(2)).max(1);
    let mut k_a = min_k_a;
    for k in min_k_a..(nz - 1) {
        k_a = k;
        if 1.0 - xq_old[k] > max_q_a {
            continue;
        }
        if 1.0 - xq_old[k] <= mw.min_q_for_k_below_const_q
            || model.ln_t[k] >= lnt_a
        {
            break;
        }
    }

    let min_k_b = (k_a + 1).min(nz - 1);
    let mut k_b = min_k_b;
    for k in min_k_b..nz {
        k_b = k;
        if 1.0 - xq_old[k] > mw.max_q_for_k_const_mass {
            continue;
        }
        if 1.0 - xq_old[k] <= mw.min_q_for_k_const_mass
            || model.ln_t[k] >= lnt_b
        {
            break;
        }
    }

    let xq_a = xq_old[k_a];
    loop {
        let dxq = (xq_old[k_b] - xq_a).max(1.0e-99);
        let qfrac = 1.0 + dq_acc * (1.0 - xq_old[k_b]) / dxq;
        if k_b + 1 >= nz {
            break;
        }
        if k_b - k_a > 10 {
            if qfrac > 0.67 && qfrac < 1.50 {
                break;
            }
            if qfrac > 0.50
                && qfrac < 2.00
                && dominant_species_index(model.cell_xa(k_b))
                    != dominant_species_index(model.cell_xa(k_b + 1))
            {
                k_b = k_b.saturating_sub(5).max(k_a + 1);
                break;
            }
        }
        k_b += 1;
    }

    do_adjust_mass_regions(model, delta_m, dt, None, k_a, k_b)
}
