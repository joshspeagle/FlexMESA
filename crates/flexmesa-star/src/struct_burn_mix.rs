// SPDX-License-Identifier: LGPL-3.0-or-later
//! Coupled and thresholded operator-split structure-burn-mix driver
//! (`do_struct_burn_mix`), implicit tridiagonal species diffusion
//! (`solve_composition_diffusion`), and substepped rotational angular momentum
//! diffusion (`solve_omega_mix`) corresponding to
//! `star/private/struct_burn_mix.f90` and `star/private/solve_omega_mix.f90`.
//!
//! # Thresholded Operator-Split vs. Coupled Burn + Mix
//! Following upstream `struct_burn_mix.f90:33-224`:
//! 1. **Optional High-$T$ 1-Zone Burn (`op_split_burn_zones`)**: When
//!    `op_split_burn && do_burn`, zones with $T_{\mathrm{start},k} \ge
//!    \mathrm{op\_split\_burn\_min\_T}$ (`2e9` K default) integrate the reaction
//!    network over $[0, \Delta t]$ in parallel (`burn_1_zone` or variable-$T$
//!    `burn_1_zone_const_density`), updating `xa` and `burn_avg_epsnuc` and
//!    refreshing EOS/opacity (`non_epsnuc_energy_change_from_split_burn`), while
//!    cooler zones ($T_{\mathrm{start},k} < \mathrm{op\_split\_burn\_min\_T}$)
//!    remain coupled in the Henyey system.
//! 2. **Fixed Start-of-Step Convective Coupling (`sig`)**: Evaluates glitch
//!    cleanup, sub-cell convective boundaries, overshooting, and Lagrangian face
//!    coupling $\sigma_k$ once before `save_start_values`.
//! 3. **Coupled Henyey Solve**: Solves the `(nvar_hydro + species)` block-tridiagonal
//!    system when `(do_burn || do_mix) && species > 0`, or `nvar_hydro` when
//!    neither burning nor mixing is active.

use flexmesa_chem::NUM_CATEGORIES;
use flexmesa_const::LN10;
use flexmesa_eos::HelmTable;
use flexmesa_net::{
    burn_1_zone, burn_1_zone_const_density, Burn1ZoneInput,
    BurnConstDensityInput,
};
use rayon::prelude::*;

use crate::controls::StarError;
use crate::hydro_vars::set_hydro_vars;
use crate::micro::{do_eos, do_kap, resolve_model_net, set_basic_vars};
use crate::mix_info::{
    clean_mixing_glitches, compute_convection_sigmas,
    locate_convection_boundaries,
};
use crate::model::StarModel;
pub use crate::omega_mix::solve_omega_mix;
use crate::overshoot::{apply_overshooting, OvershootConfig};
use crate::solver::{solve_henyey_step, HenyeySolverReport};

struct SplitZoneOutcome {
    k: usize,
    xa_after: Vec<f64>,
    dxdt_split: Vec<f64>,
    avg_epsnuc: f64,
    eps_neu_total: f64,
    eps_cats: [f64; NUM_CATEGORIES],
    nstep: usize,
}

/// Integrates the 1-zone nuclear reaction network over $[0, \Delta t]$ in
/// parallel across all zones with $T_{\mathrm{start},k} \ge
/// \mathrm{op\_split\_burn\_min\_T}$ (`do_burn` and `burn1_zone` in
/// `star/private/struct_burn_mix.f90:819-1151`).
///
/// Cooler zones ($T_{\mathrm{start},k} < \mathrm{op\_split\_burn\_min\_T}$) set
/// `burn_avg_epsnuc[k] = 0` and retain their coupled `do_net` rates and
/// Jacobians for the subsequent Henyey solve.
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if 1-zone integration fails.
pub fn op_split_burn_zones(
    model: &mut StarModel,
    dt: f64,
) -> Result<(), StarError> {
    if dt <= 0.0 || model.nz == 0 || model.species == 0 {
        return Ok(());
    }
    let nz = model.nz;
    let species = model.species;
    let Some(net_info) = resolve_model_net(model) else {
        return Ok(());
    };
    if model.aux.burn_avg_epsnuc.len() != nz {
        model.aux.burn_avg_epsnuc.resize(nz, 0.0);
    }
    if model.aux.eps_nuc_categories.len() != nz {
        model
            .aux
            .eps_nuc_categories
            .resize(nz, [0.0; NUM_CATEGORIES]);
    }

    let min_t_split = model.controls.op_split_burn_min_t;
    let min_t_var = model.controls.op_split_burn_min_t_for_variable_t_solver;
    let min_logt_net = model.controls.min_logt_for_net;
    let max_abar = model.controls.max_abar_for_burning;
    let eps_nuc_factor = model.controls.eps_nuc_factor;
    let dxdt_nuc_factor = model.controls.dxdt_nuc_factor;
    let screening_mode = model.controls.screening_mode;
    let burn_eps = model.controls.op_split_burn_eps;
    let burn_odescal = model.controls.op_split_burn_odescal;
    let max_steps = model.controls.burn_steps_hard_limit.max(1);

    let min_v_start = model.v_start.iter().copied().fold(0.0_f64, f64::min);
    let skip_burn =
        min_v_start < -model.controls.op_split_burn_eps_nuc_infall_limit;

    let mut split_indices = Vec::new();
    for k in 0..nz {
        let t_start_k = model
            .ln_t_start
            .get(k)
            .copied()
            .unwrap_or(model.ln_t[k])
            .exp();
        if t_start_k < min_t_split {
            model.aux.burn_avg_epsnuc[k] = 0.0;
        } else {
            split_indices.push(k);
        }
    }
    if split_indices.is_empty() {
        return Ok(());
    }

    let ht = HelmTable::builtin();
    let rate_factors = vec![1.0_f64; net_info.num_reactions];
    let eos_cfg = &model.eos_config;

    let outcomes: Result<Vec<SplitZoneOutcome>, StarError> = split_indices
        .into_par_iter()
        .map(|k| {
            let start_x = &model.xa[k * species..(k + 1) * species];
            let log_t = model.ln_t[k] / LN10;
            let log_rho = model.lnd[k] / LN10;
            if skip_burn
                || (eps_nuc_factor == 0.0 && dxdt_nuc_factor == 0.0)
                || model.abar[k] > max_abar
                || log_t < min_logt_net
            {
                return Ok(SplitZoneOutcome {
                    k,
                    xa_after: start_x.to_vec(),
                    dxdt_split: vec![0.0; species],
                    avg_epsnuc: 0.0,
                    eps_neu_total: 0.0,
                    eps_cats: [0.0; NUM_CATEGORIES],
                    nstep: 0,
                });
            }

            let (ending_x, avg_raw, eps_neu, mut cats, nstep_zone) =
                if model.t[k] >= min_t_var {
                    let input = BurnConstDensityInput {
                        t_start: 0.0,
                        t_end: dt,
                        starting_x: start_x,
                        starting_log10_t: log_t,
                        log10_rho: log_rho,
                        rate_factors: &rate_factors,
                        weak_rate_factor: 1.0,
                        screening_mode,
                        stptry_in: 0.0,
                        max_steps,
                        eps: burn_eps,
                        odescal: burn_odescal,
                    };
                    let out = burn_1_zone_const_density(
                        &net_info, ht, eos_cfg, &input,
                    )
                    .map_err(|e| {
                        StarError::MicrophysicsError(format!(
                            "burn_1_zone_const_density failed at zone {k}: {e}"
                        ))
                    })?;
                    (
                        out.ending_x,
                        out.avg_eps_nuc,
                        out.eps_neu_total,
                        out.eps_nuc_categories,
                        out.nstep,
                    )
                } else {
                    let times = [0.0, dt];
                    let log10_ts = [log_t, log_t];
                    let log10_rhos = [log_rho, log_rho];
                    let etas = [model.eta[k], model.eta[k]];
                    let input = Burn1ZoneInput {
                        t_start: 0.0,
                        t_end: dt,
                        starting_x: start_x,
                        times: &times,
                        log10_ts: &log10_ts,
                        log10_rhos: &log10_rhos,
                        etas: &etas,
                        dxdt_source_term: None,
                        rate_factors: &rate_factors,
                        weak_rate_factor: 1.0,
                        reaction_qs: &[],
                        reaction_neu_qs: &[],
                        screening_mode,
                        stptry_in: 0.0,
                        max_steps,
                        eps: burn_eps,
                        odescal: burn_odescal,
                    };
                    let out = burn_1_zone(&net_info, ht, eos_cfg, &input)
                        .map_err(|e| {
                            StarError::MicrophysicsError(format!(
                                "op_split_burn failed at zone {k}: {e}"
                            ))
                        })?;
                    (
                        out.ending_x,
                        out.avg_eps_nuc,
                        out.eps_neu_total,
                        out.eps_nuc_categories,
                        out.nstep,
                    )
                };

            let mut xa_after = vec![0.0_f64; species];
            let mut x_sum = 0.0_f64;
            for (j, &xj) in ending_x.iter().enumerate() {
                let xc = xj.clamp(0.0, 1.0);
                xa_after[j] = xc;
                x_sum += xc;
            }
            if x_sum > 0.0 {
                for x in &mut xa_after {
                    *x /= x_sum;
                }
            }
            let dxdt_split: Vec<f64> = (0..species)
                .map(|j| (xa_after[j] - start_x[j]) / dt)
                .collect();
            for c in &mut cats {
                *c *= eps_nuc_factor;
            }

            Ok(SplitZoneOutcome {
                k,
                xa_after,
                dxdt_split,
                avg_epsnuc: avg_raw * eps_nuc_factor,
                eps_neu_total: eps_neu * eps_nuc_factor,
                eps_cats: cats,
                nstep: nstep_zone,
            })
        })
        .collect();

    let mut max_nstep = 0_usize;
    for out in outcomes? {
        max_nstep = max_nstep.max(out.nstep);
        let k = out.k;
        let base_s = k * species;
        let base_ss = k * species * species;
        model.xa[base_s..base_s + species].copy_from_slice(&out.xa_after);
        model.xa_start[base_s..base_s + species].copy_from_slice(&out.xa_after);
        model.eps_nuc[k] = out.avg_epsnuc;
        model.aux.burn_avg_epsnuc[k] = out.avg_epsnuc;
        model.aux.eps_nuc_categories[k] = out.eps_cats;
        model.eps_nuc_neu_total[k] = out.eps_neu_total;
        model.d_epsnuc_dlnd[k] = 0.0;
        model.d_epsnuc_dlnt[k] = 0.0;
        model.d_epsnuc_dx[base_s..base_s + species].fill(0.0);
        model.dxdt_nuc[base_s..base_s + species]
            .copy_from_slice(&out.dxdt_split);
        model.d_dxdt_nuc_dlnd[base_s..base_s + species].fill(0.0);
        model.d_dxdt_nuc_dlnt[base_s..base_s + species].fill(0.0);
        model.d_dxdt_nuc_dx[base_ss..base_ss + species * species].fill(0.0);
    }
    model.aux.step_hist.burn_max_num_substeps =
        model.aux.step_hist.burn_max_num_substeps.max(max_nstep);
    Ok(())
}

/// Solves the backward-Euler tridiagonal species diffusion system
/// $$\frac{X_{j,k}^{(n+1)} - X_{j,k}^{(n)}}{\Delta t} = \frac{\sigma_k (X_{j,k-1}^{(n+1)} - X_{j,k}^{(n+1)}) - \sigma_{k+1} (X_{j,k}^{(n+1)} - X_{j,k+1}^{(n+1)})}{\Delta m_k}$$
/// across all species $j \in \{0, \dots, \mathrm{species}-1\}$ via the Thomas
/// algorithm, conserving $\sum_{k=0}^{\mathrm{nz}-1} \Delta m_k X_{j,k}$ to
/// machine precision.
pub fn solve_composition_diffusion(model: &mut StarModel, dt: f64) {
    let nz = model.nz;
    let species = model.species;
    if nz < 2 || species == 0 || dt <= 0.0 {
        return;
    }

    let mut dl = vec![0.0; nz];
    let mut d = vec![1.0; nz];
    let mut du = vec![0.0; nz];
    for k in 0..nz {
        let dm_k = model.dm[k].max(1.0e-99);
        let a_m1 = if k > 0 { dt * model.sig[k] / dm_k } else { 0.0 };
        let a_p1 = if k + 1 < nz {
            dt * model.sig[k + 1] / dm_k
        } else {
            0.0
        };
        dl[k] = -a_m1;
        d[k] = 1.0 + a_m1 + a_p1;
        du[k] = -a_p1;
    }

    let factor = ScalarTridiagFactor::factor(&dl, &d, &du);
    let mut rhs = vec![0.0; nz];
    let mut sol = vec![0.0; nz];
    let mut d_prime = vec![0.0; nz];
    for j in 0..species {
        for (k, r_k) in rhs.iter_mut().enumerate() {
            *r_k = model.xa[k * species + j];
        }
        factor.solve_into(&dl, &rhs, &mut sol, &mut d_prime);
        for (k, &val) in sol.iter().enumerate() {
            model.xa[k * species + j] = val.clamp(0.0, 1.0);
        }
    }

    for k in 0..nz {
        let sum_x: f64 = model.cell_xa(k).iter().sum();
        if sum_x > 0.0 {
            for x in model.cell_xa_mut(k) {
                *x /= sum_x;
            }
        }
    }
}

pub(crate) struct ScalarTridiagFactor {
    denom: Vec<f64>,
    c_prime: Vec<f64>,
}

impl ScalarTridiagFactor {
    pub(crate) fn factor(dl: &[f64], d: &[f64], du: &[f64]) -> Self {
        let n = d.len();
        if n == 0 {
            return Self {
                denom: Vec::new(),
                c_prime: Vec::new(),
            };
        }
        let mut denom = vec![0.0; n];
        let mut c_prime = vec![0.0; n];
        denom[0] = d[0];
        c_prime[0] = if n > 1 { du[0] / denom[0] } else { 0.0 };
        for i in 1..n {
            denom[i] = d[i] - dl[i] * c_prime[i - 1];
            c_prime[i] = if i + 1 < n { du[i] / denom[i] } else { 0.0 };
        }
        Self { denom, c_prime }
    }

    pub(crate) fn solve_into(
        &self,
        dl: &[f64],
        rhs: &[f64],
        x: &mut [f64],
        d_prime: &mut [f64],
    ) {
        let n = self.denom.len();
        if n == 0 {
            return;
        }
        d_prime[0] = rhs[0] / self.denom[0];
        for i in 1..n {
            d_prime[i] = (rhs[i] - dl[i] * d_prime[i - 1]) / self.denom[i];
        }
        x[n - 1] = d_prime[n - 1];
        for i in (0..(n - 1)).rev() {
            x[i] = d_prime[i] - self.c_prime[i] * x[i + 1];
        }
    }
}

/// Restores time-averaged split-burn `eps_nuc` for zones above
/// `op_split_burn_min_t` after synchronous EOS/network evaluation.
pub(crate) fn apply_burn_avg_epsnuc(model: &mut StarModel) {
    let min_t_split = model.controls.op_split_burn_min_t;
    for k in 0..model.nz {
        let t_start_k = model
            .ln_t_start
            .get(k)
            .copied()
            .unwrap_or(model.ln_t[k])
            .exp();
        if t_start_k >= min_t_split {
            if let Some(&avg_eps) = model.aux.burn_avg_epsnuc.get(k) {
                model.eps_nuc[k] = avg_eps;
            }
        }
    }
}

/// Applies glitch cleanup, predictive mixing, convective boundary detection,
/// overshooting, and Lagrangian coupling factor (`sig`) evaluation.
pub(crate) fn refresh_mixing_state(
    model: &mut StarModel,
    dt: f64,
    overshoot_configs: &[OvershootConfig],
) {
    if model.controls.remove_mixing_glitches {
        clean_mixing_glitches(model, 1.0, 0.1);
    }
    if model.controls.do_mix && model.controls.mix_factor > 0.0 {
        crate::premix::apply_predictive_mixing_from_controls(model);
    }
    let (_cz_bdy_dq, boundaries) = locate_convection_boundaries(model);
    apply_overshooting(model, &boundaries, overshoot_configs);
    compute_convection_sigmas(model, dt, 1.0, 1.0e10);
    if model.controls.rti_flag
        && model.controls.do_mix
        && model.controls.mix_factor > 0.0
    {
        crate::hydro_rti::set_rti_mixing_info(model);
        crate::hydro_rti::augment_mixing_with_rti(model);
    }
}

fn run_struct_burn_mix_core(
    model: &mut StarModel,
    dt: f64,
    op_split_burn: bool,
    do_burn: bool,
    do_mix: bool,
    use_bcyclic: bool,
    overshoot_configs: &[OvershootConfig],
) -> Result<HenyeySolverReport, StarError> {
    if op_split_burn && do_burn {
        let e_before: f64 =
            (0..model.nz).map(|k| model.energy[k] * model.dm[k]).sum();
        op_split_burn_zones(model, dt)?;
        set_basic_vars(model)?;
        do_eos(model)?;
        do_kap(model)?;
        let e_after: f64 =
            (0..model.nz).map(|k| model.energy[k] * model.dm[k]).sum();
        model.aux.non_epsnuc_energy_change_from_split_burn = e_after - e_before;
    }

    if do_mix {
        refresh_mixing_state(model, dt, overshoot_configs);
    } else {
        model.sig.fill(0.0);
    }

    if model.controls.rsp2_flag {
        crate::rsp2_eqns::rsp2_adjust_vars_before_call_solver(model);
        model.pack_xh();
    }
    model.save_start_values();

    let do_chem = (do_burn || do_mix) && model.species > 0;
    solve_henyey_step(model, do_chem, use_bcyclic)
}

/// Executes a full structure + nuclear burn + convective mixing step
/// (`do_struct_burn_mix` in `star/private/struct_burn_mix.f90:33-224`).
///
/// # Errors
/// Returns [`StarError`] if microphysics or Newton-Raphson convergence fails.
pub fn do_struct_burn_mix(
    model: &mut StarModel,
    dt: f64,
    op_split_burn: bool,
    use_bcyclic: bool,
    overshoot_configs: &[OvershootConfig],
) -> Result<HenyeySolverReport, StarError> {
    model.dt = dt;
    let saved_op_split = model.controls.op_split_burn;
    model.controls.op_split_burn = op_split_burn;
    model.aux.doing_struct_burn_mix = false;

    if let Err(e) = set_hydro_vars(model, false, false) {
        model.controls.op_split_burn = saved_op_split;
        return Err(e);
    }
    model.ln_t_start.clone_from(&model.ln_t);
    model.v_start.clone_from(&model.v);
    model.aux.doing_struct_burn_mix = true;
    model.aux.non_epsnuc_energy_change_from_split_burn = 0.0;
    model.aux.step_hist.burn_max_num_substeps = 0;

    let do_burn =
        model.controls.do_burn && model.controls.dxdt_nuc_factor > 0.0;
    let do_mix = model.controls.do_mix && model.controls.mix_factor > 0.0;

    let step_res = run_struct_burn_mix_core(
        model,
        dt,
        op_split_burn,
        do_burn,
        do_mix,
        use_bcyclic,
        overshoot_configs,
    );

    model.aux.doing_struct_burn_mix = false;
    let report = match step_res {
        Ok(rep) => rep,
        Err(e) => {
            model.controls.op_split_burn = saved_op_split;
            return Err(e);
        }
    };
    model.aux.step_hist.last_solver_iterations = report.iterations;

    let final_res = set_hydro_vars(model, false, false);
    if op_split_burn {
        apply_burn_avg_epsnuc(model);
    }
    model.controls.op_split_burn = saved_op_split;
    final_res?;

    if do_mix {
        refresh_mixing_state(model, dt, overshoot_configs);
    }
    Ok(report)
}
