// SPDX-License-Identifier: LGPL-3.0-or-later
//! Parallel thread-local nuclear reaction network evaluation (`do_net`)
//! corresponding to `star/private/net.f90:62-436` and
//! `star/job/run_star_support.f90:1493-1524`.

use flexmesa_chem::{NuclideCatalog, NUM_CATEGORIES};
use flexmesa_const::{LN10, QCONV};
use flexmesa_eos::I_ETA;
use flexmesa_math::exp10;
use flexmesa_net::derivs::{eval_net_with_work, with_thread_local_work};
use flexmesa_net::f64x2::F64x2;
use flexmesa_net::types::NUM_RVS;
use flexmesa_net::NetGeneralInfo;
use flexmesa_utils::is_bad;
use rayon::prelude::*;

use crate::controls::{StarError, StarJobControls};
use crate::model::StarModel;

/// Builds the 0-based `rate_factors` vector (`len = net_info.num_reactions`)
/// from `&star_job` special rate factor controls (`run_star_support.f90:1493-1524`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if a requested reaction name is absent
/// from the active network (unless a `change_net` transition to a different
/// network is still pending).
pub fn build_rate_factors(
    job: &StarJobControls,
    net_info: &NetGeneralInfo,
) -> Result<Vec<f64>, StarError> {
    let mut factors = vec![1.0_f64; net_info.num_reactions];
    if job.num_special_rate_factors == 0 {
        return Ok(factors);
    }
    if job.change_net
        && !job.new_net_name.trim().is_empty()
        && job.new_net_name.trim() != job.net_name.trim()
    {
        return Ok(factors);
    }
    for i in 0..job.num_special_rate_factors {
        let name = job
            .reaction_for_special_factor
            .get(i)
            .map_or("", |s| s.trim());
        if name.is_empty() {
            continue;
        }
        let Some(idx1) = net_info.reaction_index_by_name(name) else {
            return Err(StarError::InvalidInput(format!(
                "reaction_for_special_factor({}) = '{name}' not found in active network '{}'",
                i + 1,
                job.net_name
            )));
        };
        let fac = job.special_rate_factor.get(i).copied().unwrap_or(1.0);
        factors[idx1 - 1] = fac;
    }
    Ok(factors)
}

/// Evaluates the nuclear reaction network across all `nz` zones in parallel
/// using thread-local `NetWorkState` workspaces (`do_net` in
/// `star/private/net.f90:62-436`).
///
/// Zones handled by operator-split 1-zone burning during `do_struct_burn_mix`
/// (`doing_struct_burn_mix && op_split_burn && T_start[k] >= op_split_burn_min_T`,
/// `net.f90:160-164`) are left untouched so `burn_avg_epsnuc` and post-burn
/// categories remain intact across Newton iterations.
///
/// # Errors
/// Returns [`StarError`] if `rate_factors` lookup or reaction evaluation fails.
pub fn do_net(
    model: &mut StarModel,
    net_info: &NetGeneralInfo,
    skip_jacobian: bool,
) -> Result<(), StarError> {
    let nz = model.nz;
    let species = model.species;
    if nz == 0 || species == 0 {
        return Ok(());
    }
    if model.aux.eps_nuc_categories.len() != nz {
        model
            .aux
            .eps_nuc_categories
            .resize(nz, [0.0; NUM_CATEGORIES]);
    }

    let rate_factors = build_rate_factors(&model.job, net_info)?;
    let use_3a_fl87 = model.job.use_3a_fl87;
    let use_suzuki = model.job.use_suzuki_weak_rates;

    let max_logt = model.controls.max_logt_for_net;
    let max_abar = model.controls.max_abar_for_burning;
    let eps_fac = model.controls.eps_nuc_factor;
    let dxdt_fac = model.controls.dxdt_nuc_factor;
    let weak_rate_factor = model.controls.weak_rate_factor;
    let max_abs_eps_nuc = model.controls.max_abs_eps_nuc;
    let screening_mode = model.controls.screening_mode;
    let doing_split =
        model.aux.doing_struct_burn_mix && model.controls.op_split_burn;
    let split_min_t = model.controls.op_split_burn_min_t;

    let catalog = NuclideCatalog::builtin_ref();
    let aion: Vec<f64> = net_info
        .chem_id
        .iter()
        .map(|&cid| catalog.get_by_id(cid).map_or(1.0, |n| n.z_plus_n as f64))
        .collect();

    let t_slice = &model.t;
    let lnt_slice = &model.ln_t;
    let rho_slice = &model.rho;
    let lnd_slice = &model.lnd;
    let lnt_start_slice = &model.ln_t_start;
    let xa_slice = &model.xa;
    let abar_slice = &model.abar;
    let zbar_slice = &model.zbar;
    let z2bar_slice = &model.z2bar;
    let ye_slice = &model.ye;
    let eta_slice = &model.eta;
    let eos_dlnt_slice = &model.eos_d_dlnt;
    let eos_dlnd_slice = &model.eos_d_dlnd;

    let eps_nuc = &mut model.eps_nuc[..nz];
    let d_eps_dlnd = &mut model.d_epsnuc_dlnd[..nz];
    let d_eps_dlnt = &mut model.d_epsnuc_dlnt[..nz];
    let eps_neu = &mut model.eps_nuc_neu_total[..nz];
    let eps_cats = &mut model.aux.eps_nuc_categories[..nz];
    let d_eps_dx = &mut model.d_epsnuc_dx[..nz * species];
    let dxdt_nuc = &mut model.dxdt_nuc[..nz * species];
    let d_dxdt_dlnd = &mut model.d_dxdt_nuc_dlnd[..nz * species];
    let d_dxdt_dlnt = &mut model.d_dxdt_nuc_dlnt[..nz * species];
    let d_dxdt_dx = &mut model.d_dxdt_nuc_dx[..nz * species * species];

    (
        eps_nuc.par_iter_mut(),
        d_eps_dlnd.par_iter_mut(),
        d_eps_dlnt.par_iter_mut(),
        eps_neu.par_iter_mut(),
        eps_cats.par_iter_mut(),
        d_eps_dx.par_chunks_mut(species),
        dxdt_nuc.par_chunks_mut(species),
        d_dxdt_dlnd.par_chunks_mut(species),
        d_dxdt_dlnt.par_chunks_mut(species),
        d_dxdt_dx.par_chunks_mut(species * species),
    )
        .into_par_iter()
        .enumerate()
        .try_for_each(
            |(
                k,
                (
                    eps_k,
                    deps_d_k,
                    deps_t_k,
                    neu_k,
                    cats_k,
                    deps_x_k,
                    dxdt_k,
                    ddxdt_d_k,
                    ddxdt_t_k,
                    ddxdt_x_k,
                ),
            )| {
                if doing_split {
                    let t_start_k = lnt_start_slice
                        .get(k)
                        .copied()
                        .map_or(t_slice[k], flexmesa_math::exp);
                    if t_start_k >= split_min_t {
                        return Ok(());
                    }
                }

                if (eps_fac == 0.0 && dxdt_fac == 0.0)
                    || abar_slice[k] > max_abar
                {
                    *eps_k = 0.0;
                    *deps_d_k = 0.0;
                    *deps_t_k = 0.0;
                    *neu_k = 0.0;
                    cats_k.fill(0.0);
                    deps_x_k.fill(0.0);
                    dxdt_k.fill(0.0);
                    ddxdt_d_k.fill(0.0);
                    ddxdt_t_k.fill(0.0);
                    ddxdt_x_k.fill(0.0);
                    return Ok(());
                }

                // `star/private/net.f90:185-193`: `log10_rho = lnd/ln10`,
                // `log10_T = lnT/ln10`, with `max_logT_for_net` clipping.
                let mut t_k = t_slice[k];
                let mut log_t = lnt_slice[k] / LN10;
                let clipped_t = max_logt > 0.0 && log_t > max_logt;
                if clipped_t {
                    log_t = max_logt;
                    t_k = exp10(log_t);
                }

                let rho_k = rho_slice[k];
                let log_rho = lnd_slice[k] / LN10;
                let xa_k = &xa_slice[k * species..(k + 1) * species];

                let mut stack_dydt = [[F64x2::ZERO; NUM_RVS]; 128];
                let mut heap_dydt = Vec::new();
                let dydt: &mut [[F64x2; NUM_RVS]] = if species <= 128 {
                    &mut stack_dydt[..species]
                } else {
                    heap_dydt.resize(species, [F64x2::ZERO; NUM_RVS]);
                    &mut heap_dydt[..]
                };

                with_thread_local_work(net_info, |w| {
                    let (raw_eps, raw_deps_dt, raw_deps_drho, raw_neu) =
                        eval_net_with_work(
                            net_info,
                            w,
                            dydt,
                            skip_jacobian,
                            xa_k,
                            t_k,
                            log_t,
                            rho_k,
                            log_rho,
                            abar_slice[k],
                            zbar_slice[k],
                            z2bar_slice[k],
                            ye_slice[k],
                            eta_slice[k],
                            eos_dlnt_slice[k][I_ETA],
                            eos_dlnd_slice[k][I_ETA],
                            &rate_factors,
                            weak_rate_factor,
                            screening_mode,
                            use_3a_fl87,
                            use_suzuki,
                            false,
                        )
                        .map_err(|e| {
                            StarError::MicrophysicsError(format!(
                                "do_net failed at zone {k}: {e}"
                            ))
                        })?;

                    if is_bad(raw_eps) {
                        return Err(StarError::MicrophysicsError(format!(
                            "do1_net: non-finite eps_nuc ({raw_eps}) at zone {k}"
                        )));
                    }

                    // `star/private/net.f90:296-331`: clip T derivatives,
                    // clamp `|eps_nuc| <= max_abs_eps_nuc` (zeroing partials),
                    // and renormalize `eps_nuc_categories` before `eps_nuc_factor`.
                    let mut eps_val = raw_eps;
                    let mut deps_drho = raw_deps_drho;
                    let mut deps_dt = if clipped_t { 0.0 } else { raw_deps_dt };
                    let clamped_eps = eps_val.abs() > max_abs_eps_nuc;
                    if clamped_eps {
                        eps_val = max_abs_eps_nuc.copysign(eps_val);
                        deps_drho = 0.0;
                        deps_dt = 0.0;
                    }

                    let mut eps_cat_sum = 0.0_f64;
                    for c in 0..NUM_CATEGORIES {
                        let c_val = QCONV * w.eps_nuc_categories[c];
                        cats_k[c] = c_val;
                        eps_cat_sum += c_val;
                    }
                    let alfa = if eps_cat_sum.abs() < 1.0e-10 {
                        1.0
                    } else {
                        eps_val / eps_cat_sum
                    };
                    for c_val in cats_k.iter_mut() {
                        *c_val *= alfa * eps_fac;
                    }

                    *eps_k = eps_val * eps_fac;
                    *deps_d_k = deps_drho * rho_k * eps_fac;
                    *deps_t_k = deps_dt * t_k * eps_fac;
                    // `star/private/net.f90:255,365-372`: `eps_nuc_neu_total`
                    // is not multiplied by `eps_nuc_factor`.
                    *neu_k = raw_neu;

                    for i in 0..species {
                        let a_i = aion[i];
                        dxdt_k[i] = a_i * dydt[i][0].to_f64() * dxdt_fac;
                    }
                    if skip_jacobian {
                        deps_x_k.fill(0.0);
                        ddxdt_d_k.fill(0.0);
                        ddxdt_t_k.fill(0.0);
                        ddxdt_x_k.fill(0.0);
                    } else {
                        for i in 0..species {
                            let a_i = aion[i];
                            deps_x_k[i] = if clamped_eps {
                                0.0
                            } else {
                                (QCONV * w.d_eps_nuc_dy[i] / a_i) * eps_fac
                            };
                            ddxdt_d_k[i] =
                                a_i * dydt[i][2].to_f64() * rho_k * dxdt_fac;
                            ddxdt_t_k[i] = if clipped_t {
                                0.0
                            } else {
                                a_i * dydt[i][1].to_f64() * t_k * dxdt_fac
                            };
                            let row_off = i * species;
                            for j in 0..species {
                                ddxdt_x_k[row_off + j] = w.d_dydt_dy
                                    [row_off + j]
                                    * (a_i / aion[j])
                                    * dxdt_fac;
                            }
                        }
                    }
                    Ok(())
                })
            },
        )
}
