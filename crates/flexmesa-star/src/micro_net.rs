// SPDX-License-Identifier: LGPL-3.0-or-later
//! Parallel thread-local nuclear reaction network evaluation (`do_net`)
//! corresponding to `star/private/net.f90:62-436`.

use flexmesa_chem::{NuclideCatalog, NUM_CATEGORIES};
use flexmesa_const::QCONV;
use flexmesa_eos::I_ETA;
use flexmesa_math::exp10;
use flexmesa_net::derivs::{eval_net_with_work, with_thread_local_work};
use flexmesa_net::f64x2::F64x2;
use flexmesa_net::types::NUM_RVS;
use flexmesa_net::NetGeneralInfo;
use rayon::prelude::*;

use crate::controls::StarError;
use crate::model::StarModel;

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
/// Returns [`StarError::MicrophysicsError`] if reaction evaluation fails.
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

    let min_logt = model.controls.min_logt_for_net;
    let max_logt = model.controls.max_logt_for_net;
    let max_abar = model.controls.max_abar_for_burning;
    let eps_fac = model.controls.eps_nuc_factor;
    let dxdt_fac = model.controls.dxdt_nuc_factor;
    let screening_mode = model.controls.screening_mode;
    let doing_split =
        model.aux.doing_struct_burn_mix && model.controls.op_split_burn;
    let split_min_t = model.controls.op_split_burn_min_t;
    let rate_factors = vec![1.0_f64; net_info.num_reactions];

    let catalog = NuclideCatalog::builtin_ref();
    let aion: Vec<f64> = net_info
        .chem_id
        .iter()
        .map(|&cid| catalog.get_by_id(cid).map_or(1.0, |n| n.z_plus_n as f64))
        .collect();

    let t_slice = &model.t;
    let rho_slice = &model.rho;
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

    eps_nuc
        .par_iter_mut()
        .zip(d_eps_dlnd.par_iter_mut())
        .zip(d_eps_dlnt.par_iter_mut())
        .zip(eps_neu.par_iter_mut())
        .zip(eps_cats.par_iter_mut())
        .zip(d_eps_dx.par_chunks_mut(species))
        .zip(dxdt_nuc.par_chunks_mut(species))
        .zip(d_dxdt_dlnd.par_chunks_mut(species))
        .zip(d_dxdt_dlnt.par_chunks_mut(species))
        .zip(d_dxdt_dx.par_chunks_mut(species * species))
        .enumerate()
        .try_for_each(
            |(
                k,
                (
                    (
                        (
                            (
                                (
                                    (
                                        (((eps_k, deps_d_k), deps_t_k), neu_k),
                                        cats_k,
                                    ),
                                    deps_x_k,
                                ),
                                dxdt_k,
                            ),
                            ddxdt_d_k,
                        ),
                        ddxdt_t_k,
                    ),
                    ddxdt_x_k,
                ),
            )| {
                if doing_split {
                    let t_start_k = lnt_start_slice
                        .get(k)
                        .copied()
                        .map_or(t_slice[k], f64::exp);
                    if t_start_k >= split_min_t {
                        return Ok(());
                    }
                }

                let mut t_k = t_slice[k];
                let mut log_t = t_k.log10();
                if (eps_fac == 0.0 && dxdt_fac == 0.0)
                    || abar_slice[k] > max_abar
                    || log_t < min_logt
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

                let clipped_t = max_logt > 0.0 && log_t > max_logt;
                if clipped_t {
                    log_t = max_logt;
                    t_k = exp10(log_t);
                }

                let rho_k = rho_slice[k];
                let log_rho = rho_k.log10();
                let xa_k = &xa_slice[k * species..(k + 1) * species];

                let mut stack_dydt = [[F64x2::ZERO; NUM_RVS]; 32];
                let mut heap_dydt = Vec::new();
                let dydt: &mut [[F64x2; NUM_RVS]] = if species <= 32 {
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
                            1.0,
                            screening_mode,
                            false,
                            false,
                            false,
                        )
                        .map_err(|e| {
                            StarError::MicrophysicsError(format!(
                                "do_net failed at zone {k}: {e}"
                            ))
                        })?;

                    *eps_k = raw_eps * eps_fac;
                    *deps_d_k = raw_deps_drho * rho_k * eps_fac;
                    *deps_t_k = if clipped_t {
                        0.0
                    } else {
                        raw_deps_dt * t_k * eps_fac
                    };
                    *neu_k = raw_neu * eps_fac;
                    for c in 0..NUM_CATEGORIES {
                        cats_k[c] = QCONV * w.eps_nuc_categories[c] * eps_fac;
                    }
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
                            deps_x_k[i] =
                                (QCONV * w.d_eps_nuc_dy[i] / a_i) * eps_fac;
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
