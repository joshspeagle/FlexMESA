// SPDX-License-Identifier: GPL-3.0-or-later
//! Three-region pseudo-Lagrangian mass adjustment (`do_adjust_mass`,
//! `compute_delta_m`, `revise_q_and_dq`, `set_omega_adjust_mass`,
//! `adjust_J_lost`, and `smooth_outer_xa`) matching
//! `star/private/adjust_mass.f90`.

use flexmesa_const::{FOUR_THIRDS_PI, LN10, M_SUN, ONE_THIRD, SECYER};

pub use crate::accretion_xa::{build_accreted_xa, get_xa_for_accretion};
use flexmesa_math::Quad;

use crate::controls::StarError;
use crate::core_boundaries::dominant_species_index;
use crate::eps_mdot::{AdjustMassRecord, calculate_eps_mdot};
use crate::hydro_rotation::{
    eval_i_rot, w_div_w_roche_jrot, w_div_w_roche_omega,
};
use crate::hydro_vars::set_hydro_vars;
use flexmesa_num::search::binary_search_1based;
use crate::micro::set_basic_vars;
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

/// Executes 3-region mass adjustment for an explicit `delta_m` and, when
/// `dt > 0`, evaluates `model.eps_mdot` via
/// [`crate::eps_mdot::calculate_eps_mdot`].
///
/// This entry point has no upstream counterpart: it takes the mass change
/// and the two region boundaries directly. `calculate_eps_mdot` reads
/// `mstar_dot` as upstream does (`eps_mdot.f90:510`), so `model.mstar_dot`
/// is set to `delta_m / dt` here.
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
        model.aux.step_hist.clear_adjust_mass_energy_fields();
        return Ok(());
    }
    if dt > 0.0 {
        model.mstar_dot = delta_m / dt;
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
        model.aux.k_below_const_q = 1;
        model.aux.k_const_mass = 0;
        model.eps_mdot.fill(0.0);
        model.aux.step_hist.clear_adjust_mass_energy_fields();
        return Ok(());
    }
    let mut delta_m = compute_delta_m(model, dt)?;
    if delta_m == 0.0 {
        model.aux.k_below_const_q = 1;
        model.aux.k_const_mass = 0;
        model.eps_mdot.fill(0.0);
        model.aux.step_hist.clear_adjust_mass_energy_fields();
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

/// `q1/q2` with `q1 = xqB_new - xqA` and `q2 = max(1d-99, xqB_old - xqA)`,
/// both formed in `real(dp)` and divided in `real(qp)`
/// (`adjust_mass.f90:740-743`).
fn qfrac_qp(xq_b_new: f64, xq_a: f64, xq_b_old: f64) -> Quad {
    Quad::from(xq_b_new - xq_a) / Quad::from((xq_b_old - xq_a).max(1.0e-99))
}

/// Result of [`revise_q_and_dq_upstream`].
struct RevisedMesh {
    /// Zero-based first cell of the middle region (upstream `kA - 1`).
    k_a: usize,
    /// Zero-based first cell of the constant-mass region (upstream `kB - 1`).
    k_b: usize,
    /// New cell masses `dq * new_xmstar` in grams.
    dm_target: Vec<f64>,
    /// `adjust_mass_{outer,mid,inner}_frac_sub1`.
    frac_sub1: [f64; 3],
}

fn revise_q_and_dq_upstream(
    model: &mut StarModel,
    old_xmstar: f64,
    new_xmstar: f64,
    delta_m: f64,
    explicit_ka_kb: Option<(usize, usize)>,
) -> Result<RevisedMesh, StarError> {
    let nz = model.nz;
    let mw = &model.controls.mix_wind;

    // `mold_o_mnew_qp = q1/q2`, `mold_o_mnew = mold_o_mnew_qp`
    // (`adjust_mass.f90:653-656`).
    let mold_o_mnew_qp = Quad::from(old_xmstar) / Quad::from(new_xmstar);
    let mold_o_mnew = mold_o_mnew_qp.to_f64();
    let dq_acc = delta_m / new_xmstar;

    // Initial `dq` renormalization (`adjust_mass.f90:661-676`): plain
    // left-to-right sum matching upstream strict-mode arithmetic.
    let mut sum_dq_init = 0.0_f64;
    for &d in &model.dq {
        sum_dq_init += d;
    }
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

    let (k_a, k_b, qfrac_qp_final) = if let Some((ka_in, kb_in)) = explicit_ka_kb {
        let ka = ka_in.clamp(1, nz.saturating_sub(2));
        let mut kb = kb_in.clamp(ka + 1, nz - 1);
        let xq_a = xq[ka];
        let mut qf;
        loop {
            let xq_b_old = xq[kb];
            let xq_b_new = dq_acc + xq_b_old * mold_o_mnew;
            qf = qfrac_qp(xq_b_new, xq_a, xq_b_old);
            if qf.to_f64() > 0.0 || kb + 1 >= nz {
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
            let qf_qp = qfrac_qp(xq_b_new, xq_a, xq_b_old);
            let qf_cur = qf_qp.to_f64();
            if kb + 1 >= nz {
                break qf_qp;
            }
            if kb - ka > 10 {
                if qf_cur > 0.67 && qf_cur < 1.50 {
                    break qf_qp;
                }
                if qf_cur > 0.50
                    && qf_cur < 2.00
                    && dominant_species_index(model.cell_xa(kb))
                        != dominant_species_index(model.cell_xa(kb + 1))
                {
                    // `adjust_mass.f90:753`: `kB = max(1, kB-5)` (0-based `max(0, kb-5)`).
                    kb = kb.saturating_sub(5);
                    break qf_qp;
                }
            }
            kb += 1;
            xq_b_old = xq[kb];
            xq_b_new = dq_acc + xq_b_old * mold_o_mnew;
        };
        (ka, kb, qf)
    };

    // `qfrac = qfrac_qp` (`adjust_mass.f90:743-744`).
    let qfrac = qfrac_qp_final.to_f64();
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

    // Renormalize `dq` (`adjust_mass.f90:791-806`): plain left-to-right sum
    // matching upstream strict-mode arithmetic.
    let mut sum_dq = 0.0_f64;
    for &d in &dq {
        sum_dq += d;
    }
    if !sum_dq.is_finite() || sum_dq <= 0.0 {
        return Err(StarError::ConvergenceFailure(format!(
            "revise_q_and_dq: invalid renorm sum(dq)={sum_dq}"
        )));
    }
    // `frac_qp = q1/q2` with `q1 = 1d0`, `q2 = sumdq`; `frac = frac_qp`
    // (`adjust_mass.f90:796-799`).
    let frac_qp = Quad::ONE / Quad::from(sum_dq);
    let inv_sum = frac_qp.to_f64();
    for val in &mut dq {
        *val *= inv_sum;
    }

    // `adjust_mass_{outer,mid,inner}_frac_sub1` (`adjust_mass.f90:773-776,
    // 808-810`): `frac_qp*adjust_mass_*_frac*new_xmstar/old_xmstar - 1.0_qp`,
    // stored in `real(dp)` fields.
    let sub1 = |adjust: Quad| {
        (frac_qp * adjust * new_xmstar / old_xmstar - Quad::ONE).to_f64()
    };
    let frac_sub1 = [
        sub1(Quad::ONE),
        sub1(qfrac_qp_final),
        sub1(mold_o_mnew_qp),
    ];

    // Set `q` from `dq` and close `dq[nz-1] = q[nz-1]` (`adjust_mass.f90:813-826`)
    // using upstream's plain left-to-right running sum.
    model.dq = dq;
    model.q.resize(nz, 0.0);
    model.q[0] = 1.0;
    let mut sumdq = model.dq[0];
    for k in 1..(nz - 1) {
        model.q[k] = 1.0 - sumdq;
        sumdq += model.dq[k];
    }
    model.q[nz - 1] = 1.0 - sumdq;
    model.dq[nz - 1] = model.q[nz - 1];
    model.set_xqs();

    let dm_target: Vec<f64> =
        model.dq.iter().map(|&d| d * new_xmstar).collect();
    Ok(RevisedMesh {
        k_a,
        k_b,
        dm_target,
        frac_sub1,
    })
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
        model.aux.k_below_const_q = 1;
        model.aux.k_const_mass = 0;
        model.eps_mdot.fill(0.0);
        model.aux.step_hist.clear_adjust_mass_energy_fields();
        return Ok(());
    }

    let frac = old_xmstar / new_xmstar_in;
    let new_xmstar = old_xmstar / frac;
    if !new_xmstar.is_finite() || new_xmstar <= 0.0 {
        return Err(StarError::ConvergenceFailure(format!(
            "do_adjust_mass produced non-positive envelope mass xm_new={new_xmstar}"
        )));
    }

    // Upstream runs `save_for_eps_mdot` unconditionally (`adjust_mass.f90:204`)
    // and `calculate_eps_mdot` whenever `mstar_dot != 0 && dt > 0`
    // (`eps_mdot.f90:494-678`), scaling `eps_mdot` by `eps_mdot_factor` at
    // the end (`eps_mdot.f90:666`).
    let need_eps_mdot = dt > 0.0 && model.mstar_dot != 0.0;
    // Upstream has no fallback here: `save_for_eps_mdot` reads `s% r` and
    // `s% energy` as `set_vars` left them (`adjust_mass.f90:75-95`).
    if need_eps_mdot {
        if let Some(k) = model.r[..nz].iter().position(|&v| v <= 0.0) {
            // `grav00 = -G00*m00/r00` divides by `r(k)`
            // (`star/private/star_utils.f90:2266`; DECISIONS #43, deferred #488).
            return Err(StarError::InvalidInput(format!(
                "do_adjust_mass: r({}) = {:e} <= 0 before save_for_eps_mdot; upstream divides by it in cell_specific_PE (star_utils.f90:2266; deferred #488)",
                k + 1,
                model.r[k]
            )));
        }
        if model.energy[..nz].iter().all(|&v| v == 0.0) {
            // `s% energy` is always set by `set_vars` before this point
            // upstream (`star/private/micro.f90:469`); an all-zero array here
            // means it was never evaluated (DECISIONS #43).
            return Err(StarError::InvalidInput(String::from(
                "do_adjust_mass: cell energies are unset before save_for_eps_mdot (star/private/adjust_mass.f90:75-95, 204)",
            )));
        }
    }
    let prev_dm_raw = model.dm.clone();
    // `save_for_eps_mdot` (`adjust_mass.f90:75-95, 204`).
    let total_energy_profile_before = if need_eps_mdot {
        let (profile, (_, _, _, _, _, e_before)) =
            crate::evolve_support::eval_delta_m_total_energy_integrals(
                model,
                0,
                nz,
                model.mstar,
                true,
            );
        model.aux.step_hist.total_energy_before_adjust_mass = e_before;
        profile
    } else {
        Vec::new()
    };

    let starting_j_rot_surf = model.j_rot.first().copied().unwrap_or(0.0);

    // Update `xmstar` and `mstar` (`adjust_mass.f90:264-265`).
    model.xmstar = new_xmstar;
    model.mstar = model.xmstar + model.m_center;

    let RevisedMesh {
        k_a,
        k_b,
        dm_target,
        frac_sub1,
    } = revise_q_and_dq_upstream(
        model,
        old_xmstar,
        new_xmstar,
        delta_m,
        explicit_ka_kb,
    )?;
    // `star/private/adjust_mass.f90:828-829` (1-based `k_below_const_q`;
    // `k_const_mass` remains 0-based on `HydroAuxState` for `hydro_vars`).
    model.aux.k_below_const_q = k_a + 1;
    model.aux.k_const_mass = k_b;

    // Save the composition down to `k_const_mass`, clamped to [0, 1] and
    // renormalized cell by cell, before the remap reads it
    // (`adjust_mass.f90:326-337`; `xa_old` is copied afterwards, `:372-376`).
    // `mtot_init` is accumulated upstream but never read, so it is not kept.
    let species = model.species;
    for k in 0..=k_b.min(nz - 1) {
        let cell = &mut model.xa[k * species..(k + 1) * species];
        let mut sumx = 0.0_f64;
        for x in cell.iter_mut() {
            *x = x.clamp(0.0, 1.0);
            sumx += *x;
        }
        for x in cell.iter_mut() {
            *x /= sumx;
        }
    }
    let xa_old = model.xa.clone();

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

    sync_enclosed_mass_from_dm(model, &dm_target);
    model.set_dm_bar();

    // Accretion composition (`adjust_mass.f90:379-395`).
    let built_acc = if delta_m > 0.0 && x_accreted.is_none() {
        get_xa_for_accretion(model)?
    } else {
        vec![0.0_f64; species]
    };
    let x_acc = x_accreted.unwrap_or(&built_acc);

    // Composition remap of zones 1..k_const_mass (`set_xa`,
    // `adjust_mass.f90:421`, upstream `k_const_mass = kB`, 0-based `k_b`).
    set_xa(
        model,
        k_b + 1,
        &xa_old,
        x_acc,
        &rxm_old,
        &rxm_new,
        &prev_dm_raw,
        &new_cell_mass,
        mmax,
    )?;

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

    update_radius(model);
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
        // `calculate_eps_mdot` (`evolve.f90:596`), with what
        // `save_for_eps_mdot` and `revise_q_and_dq` left for it. The indices
        // are upstream's one-based `kA` and `kB`.
        let record = AdjustMassRecord {
            dm_before: prev_dm_raw.clone(),
            total_energy_profile_before,
            outer_frac_sub1: frac_sub1[0],
            mid_frac_sub1: frac_sub1[1],
            inner_frac_sub1: frac_sub1[2],
            k_below_const_q: k_a + 1,
            k_const_mass: k_b + 1,
        };
        let surface = calculate_eps_mdot(model, &record, dt)?;
        // `eps_mdot.f90:669-677` and `evolve.f90:599-604`
        let (e_int, e_grav, e_kin, e_rot, e_turb, e_after) =
            crate::evolve_support::eval_total_energy_integrals(model);
        let sh = &mut model.aux.step_hist;
        sh.mdot_acoustic_surface = surface.mdot_acoustic_surface;
        sh.mdot_adiabatic_surface = surface.mdot_adiabatic_surface;
        sh.total_internal_energy_after_adjust_mass = e_int;
        sh.total_gravitational_energy_after_adjust_mass = e_grav;
        sh.total_radial_kinetic_energy_after_adjust_mass = e_kin;
        sh.total_rotational_kinetic_energy_after_adjust_mass = e_rot;
        sh.total_turbulent_energy_after_adjust_mass = e_turb;
        sh.total_energy_after_adjust_mass = e_after;
        sh.energy_change_from_do_adjust_mass_and_calculate_eps_mdot =
            e_after - sh.total_energy_before_adjust_mass;
    } else {
        model.eps_mdot.fill(0.0);
        model.aux.step_hist.clear_adjust_mass_energy_fields();
    }
    Ok(())
}

/// `update_radius` (`star/private/adjust_mass.f90:137-167`): rebuilds the
/// radii from the centre outwards with each cell's volume `dm/rho`, using
/// the cell densities `rho` as they stand, and stores `lnR = log(r_new)`,
/// `r = exp(lnR)`, `rmid = r_new` and `lnR_start = lnR`.
fn update_radius(model: &mut StarModel) {
    let nz = model.nz;
    let mut vol00 =
        FOUR_THIRDS_PI * model.r_center * model.r_center * model.r_center;
    for j in (0..nz).rev() {
        let volp1 = vol00;
        let cell_vol = model.dm[j] / model.rho[j];
        vol00 = volp1 + cell_vol;
        let r_new = flexmesa_math::exp(
            flexmesa_math::log(vol00 / FOUR_THIRDS_PI) * ONE_THIRD,
        );
        model.rmid[j] = r_new;
        // `store_r_in_xh` then `get_r_and_lnR_from_xh`
        // (`star/private/star_utils.f90:413-426, 456-468`).
        model.ln_r[j] = flexmesa_math::log(r_new);
        model.r[j] = flexmesa_math::exp(model.ln_r[j]);
        if model.ln_r_start.len() == nz {
            model.ln_r_start[j] = model.ln_r[j];
        }
    }
}

/// Remaps the composition of zones `1..k_const_mass` onto the revised mesh
/// (`set_xa`, `star/private/adjust_mass.f90`); deeper zones keep `xa_old`.
///
/// `k_const_mass` is upstream's 1-based count (0-based zones
/// `0..k_const_mass`). `old_xbdy` and `new_xbdy` are upstream's
/// `old_cell_xbdy` and `new_cell_xbdy` (exterior mass at each cell's outer
/// face), here `rxm_old` and `rxm_new`.
///
/// # Errors
///
/// Returns [`StarError::ConvergenceFailure`] where upstream's `set1_xa`
/// sets `ierr = -1`.
#[allow(clippy::too_many_arguments)]
fn set_xa(
    model: &mut StarModel,
    k_const_mass: usize,
    xa_old: &[f64],
    xaccrete: &[f64],
    old_xbdy: &[f64],
    new_xbdy: &[f64],
    old_cell_mass: &[f64],
    new_cell_mass: &[f64],
    mmax: f64,
) -> Result<(), StarError> {
    let (nz, species) = (model.nz, model.species);
    if species == 0 {
        return Ok(());
    }
    let k_const_mass = k_const_mass.min(nz);
    for k in k_const_mass..nz {
        model.xa[k * species..(k + 1) * species]
            .copy_from_slice(&xa_old[k * species..(k + 1) * species]);
    }
    let mut first_err: Option<StarError> = None;
    for k in 0..k_const_mass {
        if let Err(e) = set1_xa(
            model,
            k,
            xa_old,
            xaccrete,
            old_xbdy,
            new_xbdy,
            old_cell_mass,
            new_cell_mass,
            mmax,
        ) {
            // Upstream records the error and continues with the next cell.
            first_err.get_or_insert(e);
        }
    }
    first_err.map_or(Ok(()), Err)
}

/// One cell of [`set_xa`]: line-by-line port of `set1_xa`
/// (`star/private/adjust_mass.f90`), with 0-based `k` and `kk`.
#[allow(clippy::too_many_arguments)]
fn set1_xa(
    model: &mut StarModel,
    k: usize,
    xa_old: &[f64],
    xaccrete: &[f64],
    old_xbdy: &[f64],
    new_xbdy: &[f64],
    old_cell_mass: &[f64],
    new_cell_mass: &[f64],
    mmax: f64,
) -> Result<(), StarError> {
    let (nz, species) = (model.nz, model.species);
    let fail = || {
        StarError::ConvergenceFailure(format!(
            "set1_xa failed for cell {}",
            k + 1
        ))
    };
    let mut msum = vec![-1.0_f64; species];
    let mut xm_outer = new_xbdy[k];
    let new_cell_dm = if k == nz - 1 {
        mmax - xm_outer - model.m_center
    } else {
        new_cell_mass[k]
    };
    let xm_inner = xm_outer + new_cell_dm;
    let mut dm_sum = 0.0_f64;
    let mut partially_accreted_cell = false;
    let k_outer;
    if xm_outer < old_xbdy[0] {
        // Some accreted material in the new cell.
        if xm_inner <= old_xbdy[0] {
            model.xa[k * species..(k + 1) * species]
                .copy_from_slice(&xaccrete[..species]);
            return Ok(());
        }
        let dm = new_cell_dm.min(old_xbdy[0] - xm_outer);
        dm_sum = dm;
        for j in 0..species {
            msum[j] = xaccrete[j] * dm;
        }
        partially_accreted_cell = true;
        xm_outer = old_xbdy[0];
        k_outer = 0;
    } else {
        // New cell entirely composed of old material.
        msum.fill(0.0);
        if xm_outer >= old_xbdy[nz - 1] {
            k_outer = nz - 1;
        } else {
            let loc = binary_search_1based(&old_xbdy[..nz], xm_outer, 0);
            if loc == 0 || loc > nz {
                return Err(fail());
            }
            k_outer = loc - 1;
            if xm_outer < old_xbdy[k_outer] {
                return Err(fail());
            }
            if k_outer < nz - 1 && old_xbdy[k_outer + 1] <= xm_outer {
                return Err(fail());
            }
        }
    }

    for kk in k_outer..nz {
        // Loop until reaching m_inner.
        let xm0 = old_xbdy[kk];
        if xm0 >= xm_inner {
            if dm_sum < new_cell_dm && kk > 0 {
                // Upstream also sets `dm_sum = new_cell_dm` here; it is not
                // read after this exit.
                let dm = new_cell_dm - dm_sum;
                for j in 0..species {
                    msum[j] += xa_old[(kk - 1) * species + j] * dm;
                }
            }
            break;
        }
        let xm1 = if kk == nz - 1 {
            mmax - model.m_center
        } else {
            old_xbdy[kk + 1]
        };
        if xm1 < xm_outer {
            return Err(fail());
        }
        if xm0 >= xm_outer && xm1 <= xm_inner {
            let mut dm = old_cell_mass[kk];
            dm_sum += dm;
            if dm_sum > new_cell_dm {
                // Reproduced as written (deferred upstream bug #386): with
                // `dm_sum > new_cell_dm` this increases `dm`.
                dm -= new_cell_dm - dm_sum;
                dm_sum = new_cell_dm;
            }
            for j in 0..species {
                msum[j] += xa_old[kk * species + j] * dm;
            }
        } else if xm0 <= xm_outer && xm1 >= xm_inner {
            let dm = if partially_accreted_cell {
                xm_inner - xm_outer
            } else {
                new_cell_mass[k]
            };
            dm_sum += dm;
            for j in 0..species {
                msum[j] += xa_old[kk * species + j] * dm;
            }
        } else {
            // Only the part of old cell kk that lies in new cell k.
            let dm = if xm_inner <= xm1 {
                let dm = new_cell_dm - dm_sum;
                dm_sum = new_cell_dm;
                dm
            } else {
                let mut dm = (xm1 - xm_outer).max(0.0);
                if dm_sum + dm > new_cell_dm {
                    dm = new_cell_dm - dm_sum;
                }
                dm_sum += dm;
                dm
            };
            for j in 0..species {
                msum[j] += xa_old[kk * species + j] * dm;
            }
            if dm <= 0.0 {
                return Err(fail());
            }
        }
        if dm_sum >= new_cell_dm {
            break;
        }
    }

    let cell = &mut model.xa[k * species..(k + 1) * species];
    for j in 0..species {
        cell[j] = msum[j] / new_cell_mass[k];
    }
    let mut sumx = 0.0_f64;
    for &x in cell.iter() {
        sumx += x;
    }
    if sumx <= 0.0 {
        return Err(fail());
    }
    for x in cell.iter_mut() {
        *x /= sumx;
    }
    Ok(())
}

fn set1_irot(model: &mut StarModel, k: usize, jrot_known: bool) {
    // Verbatim port of `set1_irot` (`star/private/adjust_mass.f90:1232-1257`):
    // `r00 = get_r_from_xh(s, k)` (`star/private/star_utils.f90:429-440`),
    // `w_div_w_roche_{jrot,omega}`, and `eval_i_rot`
    // (`star/private/hydro_rotation.f90:55-226, 874-920`).
    let r00 = flexmesa_math::exp(model.ln_r[k]);
    let max_w = model.controls.mix_wind.w_div_wcrit_max;
    let max_w2 = model.controls.mix_wind.w_div_wcrit_max2;
    let w_div_wcrit_roche = if jrot_known {
        w_div_w_roche_jrot(
            r00,
            model.m[k],
            model.j_rot[k],
            model.cgrav[k],
            max_w,
            max_w2,
            false,
        )
    } else {
        w_div_w_roche_omega(
            r00,
            model.m[k],
            model.omega[k],
            model.cgrav[k],
            max_w,
            max_w2,
            false,
        )
    };
    if k < model.aux.w_div_wc.len() {
        model.aux.w_div_wc[k] = w_div_wcrit_roche;
    }
    let i_rot_k = eval_i_rot(
        r00,
        w_div_wcrit_roche,
        model.controls.mix_wind.simple_i_rot_flag,
    );
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
                model.omega[k] = model.j_rot[k] / model.aux.i_rot[k];
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
                model.omega[k] = model.j_rot[k] / model.aux.i_rot[k];
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
    model.omega[k] = model.j_rot[k] / model.aux.i_rot[k];
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
            model.omega[k] = model.j_rot[k] / model.aux.i_rot[k];
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
