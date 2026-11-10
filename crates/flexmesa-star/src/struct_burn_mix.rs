// SPDX-License-Identifier: LGPL-3.0-or-later
//! Operator-split and coupled structure-burn-mix driver (`do_struct_burn_mix`),
//! implicit tridiagonal species diffusion (`solve_composition_diffusion`), and
//! substepped rotational angular momentum diffusion (`solve_omega_mix`)
//! corresponding to `star/private/struct_burn_mix.f90` and
//! `star/private/solve_omega_mix.f90`.
//!
//! # Operator-Split vs. Coupled Burn + Mix
//! In `do_struct_burn_mix`, each evolutionary timestep $\Delta t$ proceeds via:
//! 1. **Hydro & Mixing Setup**: Evaluates `set_hydro_vars`, cleans mixing
//!    glitches (`clean_mixing_glitches`), locates sub-cell convective
//!    boundaries (`locate_convection_boundaries`), applies convective
//!    overshooting (`apply_overshooting`), and computes Lagrangian face coupling
//!    factors $\sigma_k$ (`compute_convection_sigmas`).
//! 2. **Optional 1-Zone Operator-Split Burn (`op_split_burn_zones`)**: For zones
//!    with $\log_{10}(T/\mathrm{K}) \ge \mathrm{min\_logt\_for\_net}$, integrates
//!    the reaction network over $[0, \Delta t]$ via `burn_1_zone` at constant
//!    $(\rho_k, T_k)$, updating `xa`, `xa_start`, and `eps_nuc` while zeroing
//!    `dxdt_nuc` for the subsequent structure solve.
//! 3. **Henyey Structure + Diffusion Solve**: Runs `solve_henyey_step` (coupled
//!    with species when `!op_split_burn`, or followed by implicit tridiagonal
//!    species diffusion `solve_composition_diffusion` when `op_split_burn`).

use flexmesa_const::{FOUR_PI, LN10};
use flexmesa_eos::HelmTable;
use flexmesa_net::{burn_1_zone, Burn1ZoneInput};

use crate::controls::StarError;
use crate::hydro_vars::set_hydro_vars;
use crate::micro::resolve_model_net;
use crate::mix_info::{
    clean_mixing_glitches, compute_convection_sigmas,
    locate_convection_boundaries,
};
use crate::model::StarModel;
use crate::overshoot::{apply_overshooting, OvershootConfig};
use crate::solver::{solve_henyey_step, HenyeySolverReport};

/// Integrates the 1-zone nuclear reaction network over $[0, \Delta t]$ in all
/// active burning zones (`burn_zones` in `struct_burn_mix.f90:819-1224`).
///
/// Updates `xa` and `xa_start` with the post-burn mass fractions, stores the
/// time-averaged nuclear energy generation rate in `eps_nuc`, and zeroes
/// `dxdt_nuc` and its partials so the subsequent step does not double-count
/// nuclear reactions.
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if 1-zone integration fails.
pub fn op_split_burn_zones(
    model: &mut StarModel,
    dt: f64,
) -> Result<(), StarError> {
    if dt <= 0.0 {
        return Ok(());
    }
    let nz = model.nz;
    let species = model.species;
    let Some(net_info) = resolve_model_net(model) else {
        return Ok(());
    };
    let ht = HelmTable::builtin();
    let num_reactions = net_info.num_reactions;
    let rate_factors = vec![1.0; num_reactions];
    let reaction_qs = vec![0.0; num_reactions];
    let reaction_neu_qs = vec![0.0; num_reactions];
    let min_logt = if species > 12 {
        model.controls.min_logt_for_net.max(8.3)
    } else {
        model.controls.min_logt_for_net
    };
    let eps_nuc_factor = model.controls.eps_nuc_factor;
    let screening_mode = model.controls.screening_mode;

    for k in 0..nz {
        let log_t = model.ln_t[k] / LN10;
        let log_rho = model.lnd[k] / LN10;
        if log_t < min_logt {
            for j in 0..species {
                model.dxdt_nuc[k * species + j] = 0.0;
                model.d_dxdt_nuc_dlnd[k * species + j] = 0.0;
                model.d_dxdt_nuc_dlnt[k * species + j] = 0.0;
            }
            continue;
        }
        let start_x = model.xa[k * species..(k + 1) * species].to_vec();
        let times = [0.0, dt];
        let log10_ts = [log_t, log_t];
        let log10_rhos = [log_rho, log_rho];
        let etas = [model.eta[k], model.eta[k]];
        let input = Burn1ZoneInput {
            t_start: 0.0,
            t_end: dt,
            starting_x: &start_x,
            times: &times,
            log10_ts: &log10_ts,
            log10_rhos: &log10_rhos,
            etas: &etas,
            dxdt_source_term: None,
            rate_factors: &rate_factors,
            weak_rate_factor: 1.0,
            reaction_qs: &reaction_qs,
            reaction_neu_qs: &reaction_neu_qs,
            screening_mode,
            stptry_in: 0.0,
            max_steps: 500,
            eps: 1.0e-6,
            odescal: 1.0e-12,
        };
        let out = burn_1_zone(&net_info, ht, &model.eos_config, &input)
            .map_err(|e| {
                StarError::MicrophysicsError(format!(
                    "op_split_burn failed at zone {k}: {e}"
                ))
            })?;
        let mut x_sum = 0.0;
        for (j, &xj) in out.ending_x.iter().enumerate() {
            let x_clamped = xj.clamp(0.0, 1.0);
            model.xa[k * species + j] = x_clamped;
            x_sum += x_clamped;
        }
        if x_sum > 0.0 {
            for j in 0..species {
                let x_norm = model.xa[k * species + j] / x_sum;
                model.xa[k * species + j] = x_norm;
                model.xa_start[k * species + j] = x_norm;
            }
        }
        model.eps_nuc[k] = out.avg_eps_nuc * eps_nuc_factor;
        model.aux.burn_avg_epsnuc[k] = out.avg_eps_nuc * eps_nuc_factor;
        model.eps_nuc_neu_total[k] = out.eps_neu_total;
        model.d_epsnuc_dlnd[k] = 0.0;
        model.d_epsnuc_dlnt[k] = 0.0;
        for j in 0..species {
            model.d_epsnuc_dx[k * species + j] = 0.0;
            model.dxdt_nuc[k * species + j] = 0.0;
            model.d_dxdt_nuc_dlnd[k * species + j] = 0.0;
            model.d_dxdt_nuc_dlnt[k * species + j] = 0.0;
            for m in 0..species {
                model.d_dxdt_nuc_dx[(k * species + j) * species + m] = 0.0;
            }
        }
    }
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

    let mut rhs = vec![0.0; nz];
    for j in 0..species {
        for (k, r_k) in rhs.iter_mut().enumerate() {
            *r_k = model.xa[k * species + j];
        }
        let sol = solve_scalar_tridiag(&dl, &d, &du, &rhs);
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

pub(crate) fn solve_scalar_tridiag(
    dl: &[f64],
    d: &[f64],
    du: &[f64],
    rhs: &[f64],
) -> Vec<f64> {
    let n = d.len();
    if n == 0 {
        return Vec::new();
    }
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];
    let mut x = vec![0.0; n];

    let b0 = d[0];
    c_prime[0] = if n > 1 { du[0] / b0 } else { 0.0 };
    d_prime[0] = rhs[0] / b0;
    for i in 1..n {
        let denom = d[i] - dl[i] * c_prime[i - 1];
        c_prime[i] = if i + 1 < n { du[i] / denom } else { 0.0 };
        d_prime[i] = (rhs[i] - dl[i] * d_prime[i - 1]) / denom;
    }
    x[n - 1] = d_prime[n - 1];
    for i in (0..(n - 1)).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
    x
}

/// Solves rotational angular momentum transport (`do_solve_omega_mix` in
/// `star/private/solve_omega_mix.f90:32-667`).
///
/// Evolves face angular velocity $\Omega_k$ and specific angular momentum
/// $j_{\mathrm{rot},k} = i_{\mathrm{rot},k} \Omega_k$ ($i_{\mathrm{rot},k} = \tfrac{2}{3} r_k^2$)
/// over `dt` using `n_substeps` implicit backward-Euler tridiagonal steps,
/// conserving total angular momentum $J_{\mathrm{tot}} = \sum_{k=0}^{\mathrm{nz}-1} \overline{\Delta m}_k j_{\mathrm{rot},k}$
/// to machine precision.
pub fn solve_omega_mix(
    model: &mut StarModel,
    dt: f64,
    nu_omega: &[f64],
    n_substeps: usize,
) {
    let nz = model.nz;
    if nz < 2 || dt <= 0.0 || nu_omega.len() != nz {
        return;
    }
    let steps = n_substeps.max(1);
    let dt_sub = dt / (steps as f64);

    // Specific moment of inertia `i_rot[k]` on faces (`model.aux.i_rot[k]` when
    // populated by `update_rotation_distortion_factors`, or `(2/3) * r[k]^2`).
    let i_rot: Vec<f64> = (0..nz)
        .map(|k| {
            let ir = model.aux.i_rot.get(k).copied().unwrap_or(0.0);
            if ir > 0.0 {
                ir
            } else {
                (2.0 / 3.0) * model.r[k] * model.r[k]
            }
        })
        .collect();

    // Cell-centered harmonic-mean rotational coupling `am_sig_omega[k]` for
    // cell `k` (`0..nz-1`, `hydro_rotation.f90:399-490`).
    let mut c_omega = vec![0.0; nz - 1];
    for (k, c_om_k) in c_omega.iter_mut().enumerate() {
        let nu0 = nu_omega[k].max(0.0);
        let nu1 = nu_omega[k + 1].max(0.0);
        let d_harm = if nu0 + nu1 > 1.0e-99 {
            2.0 * nu0 * nu1 / (nu0 + nu1)
        } else {
            0.0
        };
        let r_mid = 0.5 * (model.r[k] + model.r[k + 1]);
        let area_rho = FOUR_PI * r_mid * r_mid * model.rho[k];
        let am_sig_raw =
            (area_rho * area_rho * d_harm) / model.dm[k].max(1.0e-99);
        let i_mid = 0.5 * (i_rot[k] + i_rot[k + 1]);
        *c_om_k = am_sig_raw * i_mid;
    }

    let mut dl = vec![0.0; nz];
    let mut d = vec![1.0; nz];
    let mut du = vec![0.0; nz];
    for k in 0..nz {
        let denom = (model.dm_bar[k] * i_rot[k]).max(1.0e-99);
        let f_k = dt_sub / denom;
        let c_m1 = if k > 0 { c_omega[k - 1] } else { 0.0 };
        let c_00 = if k + 1 < nz { c_omega[k] } else { 0.0 };
        dl[k] = -f_k * c_m1;
        d[k] = 1.0 + f_k * (c_m1 + c_00);
        du[k] = -f_k * c_00;
    }

    for _ in 0..steps {
        let mut rhs = vec![0.0; nz];
        for (k, r_k) in rhs.iter_mut().enumerate() {
            let denom = (model.dm_bar[k] * i_rot[k]).max(1.0e-99);
            let f_k = dt_sub / denom;
            let torque_m1 = if k > 0 {
                c_omega[k - 1] * (model.omega[k - 1] - model.omega[k])
            } else {
                0.0
            };
            let torque_00 = if k + 1 < nz {
                c_omega[k] * (model.omega[k] - model.omega[k + 1])
            } else {
                0.0
            };
            *r_k = f_k * (torque_m1 - torque_00);
        }
        let d_omega = solve_scalar_tridiag(&dl, &d, &du, &rhs);
        for (k, &dw) in d_omega.iter().enumerate() {
            model.omega[k] += dw;
        }
        if model.dm_bar[0] <= 0.0 && nz >= 2 {
            model.omega[0] = model.omega[1];
        }
    }
    for (k, &ir) in i_rot.iter().enumerate() {
        model.j_rot[k] = ir * model.omega[k];
    }
}

/// Applies glitch cleanup, convective boundary detection, overshooting, and
/// Lagrangian coupling factor (`sig`) evaluation.
pub(crate) fn refresh_mixing_state(
    model: &mut StarModel,
    dt: f64,
    overshoot_configs: &[OvershootConfig],
) {
    if model.controls.remove_mixing_glitches {
        clean_mixing_glitches(model, 1.0, 0.1);
    }
    let (_cz_bdy_dq, boundaries) = locate_convection_boundaries(model);
    if !overshoot_configs.is_empty() {
        apply_overshooting(model, &boundaries, overshoot_configs);
    }
    compute_convection_sigmas(model, dt, 1.0, 1.0e10);
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
    set_hydro_vars(model, false, false)?;
    model.save_start_values();
    refresh_mixing_state(model, dt, overshoot_configs);

    let report = if op_split_burn {
        op_split_burn_zones(model, dt)?;
        let res = solve_henyey_step(model, false, use_bcyclic);
        model.controls.op_split_burn = saved_op_split;
        let rep = res?;
        solve_composition_diffusion(model, dt);
        set_hydro_vars(model, false, false)?;
        rep
    } else {
        let res = solve_henyey_step(model, true, use_bcyclic);
        model.controls.op_split_burn = saved_op_split;
        let rep = res?;
        set_hydro_vars(model, false, false)?;
        rep
    };
    refresh_mixing_state(model, dt, overshoot_configs);
    Ok(report)
}
