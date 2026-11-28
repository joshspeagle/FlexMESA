// SPDX-License-Identifier: GPL-3.0-or-later
//! Parallel thread-local nuclear reaction network evaluation (`do_net`)
//! corresponding to `star/private/net.f90:62-436` and
//! `star/job/run_star_support.f90:1493-1524`.

use flexmesa_chem::{NuclideCatalog, NUM_CATEGORIES};
use flexmesa_const::{FOUR_PI, LN10, QCONV};
use flexmesa_eos::I_ETA;
use flexmesa_math::exp10;
use flexmesa_math::Quad;
use flexmesa_net::derivs::{eval_net_with_work, with_thread_local_work};
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

pub(crate) fn scale_eps_nuc_categories(
    raw_cats: &[f64; NUM_CATEGORIES],
    eps_val: f64,
    eps_fac: f64,
    cats_already_cgs: bool,
) -> [f64; NUM_CATEGORIES] {
    let mut cats = [0.0_f64; NUM_CATEGORIES];
    let mut eps_cat_sum = 0.0_f64;
    for (dst, &src) in cats.iter_mut().zip(raw_cats.iter()) {
        let c_val = if cats_already_cgs { src } else { QCONV * src };
        *dst = c_val;
        eps_cat_sum += c_val;
    }
    let alfa = if eps_cat_sum.abs() < 1.0e-10 {
        1.0
    } else {
        eps_val / eps_cat_sum
    };
    // `star/private/net.f90:327,366-372`: renormalize by `alfa` first, then
    // scale by `eps_nuc_factor` when `!= 1`.
    for c_val in &mut cats {
        *c_val *= alfa;
    }
    if eps_fac != 1.0 {
        for c_val in &mut cats {
            *c_val *= eps_fac;
        }
    }
    cats
}

/// `tau_gamma` of every zone for the non-local Ni/Co heating attenuation
/// (`star/private/net.f90:301-309`), or empty when it is off
/// (`nonlocal_NiCo_kap_gamma <= 0` or `nonlocal_NiCo_decay_heat`). Upstream
/// sums `dm(kk)/(pi4*rmid(kk)*rmid(kk))` over `kk = 1..k` from zero for each
/// zone and then multiplies by `nonlocal_NiCo_kap_gamma`; a running sum adds
/// the same terms in the same order. Zones that do not burn (split burning,
/// `max_abar_for_burning`) still contribute to the sum.
#[must_use]
pub(crate) fn nonlocal_nico_tau_gamma(model: &StarModel) -> Vec<f64> {
    let kap_gamma = model.controls.nonlocal_nico_kap_gamma;
    let active = kap_gamma > 0.0 && !model.controls.nonlocal_nico_decay_heat;
    if !active {
        return Vec::new();
    }
    let mut acc = 0.0_f64;
    (0..model.nz)
        .map(|kk| {
            acc += model.dm[kk] / (FOUR_PI * model.rmid[kk] * model.rmid[kk]);
            acc * kap_gamma
        })
        .collect()
}

/// `eps_nuc(k)*(1d0 - exp(-tau_gamma))` (`star/private/net.f90:308`),
/// applied before the `max_abs_eps_nuc` clamp (`:311-316`).
#[must_use]
#[inline]
fn attenuate_nonlocal_nico(eps_nuc: f64, tau_gamma: f64) -> f64 {
    eps_nuc * (1.0 - flexmesa_math::exp(-tau_gamma))
}

/// Evaluates `do1_net(..., do_jac = .false.)` on a single zone to populate
/// post-burn `eps_nuc_categories` (`star/private/struct_burn_mix.f90:1112` and
/// `star/private/net.f90:180-372`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_zone_eps_nuc_categories(
    net_info: &NetGeneralInfo,
    zone: usize,
    xa_k: &[f64],
    t_in: f64,
    log_t_in: f64,
    rho_k: f64,
    log_rho: f64,
    abar: f64,
    zbar: f64,
    z2bar: f64,
    ye: f64,
    eta: f64,
    d_eta_dlnt: f64,
    d_eta_dlnd: f64,
    rate_factors: &[f64],
    weak_rate_factor: f64,
    screening_mode: flexmesa_rates::ScreeningMode,
    use_3a_fl87: bool,
    use_suzuki: bool,
    max_logt: f64,
    max_abar: f64,
    eps_fac: f64,
    dxdt_fac: f64,
    max_abs_eps_nuc: f64,
    tau_gamma: Option<f64>,
) -> Result<[f64; NUM_CATEGORIES], StarError> {
    if (eps_fac == 0.0 && dxdt_fac == 0.0) || abar > max_abar {
        return Ok([0.0; NUM_CATEGORIES]);
    }
    let species = xa_k.len();
    let mut t_k = t_in;
    let mut log_t = log_t_in;
    if max_logt > 0.0 && log_t > max_logt {
        log_t = max_logt;
        t_k = exp10(log_t);
    }
    let mut stack_dydt = [[Quad::ZERO; NUM_RVS]; 128];
    let mut heap_dydt = Vec::new();
    let dydt: &mut [[Quad; NUM_RVS]] = if species <= 128 {
        &mut stack_dydt[..species]
    } else {
        heap_dydt.resize(species, [Quad::ZERO; NUM_RVS]);
        &mut heap_dydt[..]
    };
    with_thread_local_work(net_info, |w| {
        let (raw_eps, _, _, _) = eval_net_with_work(
            net_info,
            w,
            dydt,
            // `star/private/struct_burn_mix.f90:1112` reaches `net_get` with
            // `just_dxdt = .false.` (`star/private/net.f90:245-246`); paired
            // reactions skip their energy categories when it is set.
            false,
            xa_k,
            t_k,
            log_t,
            rho_k,
            log_rho,
            abar,
            zbar,
            z2bar,
            ye,
            eta,
            d_eta_dlnt,
            d_eta_dlnd,
            rate_factors,
            weak_rate_factor,
            screening_mode,
            use_3a_fl87,
            use_suzuki,
            false,
        )
        .map_err(|e| {
            StarError::MicrophysicsError(format!(
                "do_net failed at zone {zone}: {e}"
            ))
        })?;
        if is_bad(raw_eps) {
            return Err(StarError::MicrophysicsError(format!(
                "do1_net: non-finite eps_nuc ({raw_eps}) at zone {zone}"
            )));
        }
        // `star/private/net.f90:301-316`: non-local Ni/Co attenuation, then
        // the `max_abs_eps_nuc` clamp.
        let eps_att = match tau_gamma {
            Some(tau) => attenuate_nonlocal_nico(raw_eps, tau),
            None => raw_eps,
        };
        let eps_val = if eps_att.abs() > max_abs_eps_nuc {
            max_abs_eps_nuc.copysign(eps_att)
        } else {
            eps_att
        };
        let raw_cats = if net_info.doing_approx21 {
            &w.direct_eps_nuc_categories
        } else {
            &w.eps_nuc_categories
        };
        Ok(scale_eps_nuc_categories(
            raw_cats,
            eps_val,
            eps_fac,
            net_info.doing_approx21,
        ))
    })
}

/// Rejects networks that lack `h1` or `he4` (`star/private/net.f90:658-663`:
/// "mesa/star requires both h1 and he4 in net isotopes").
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] when either isotope is absent.
pub fn require_h1_and_he4(net_info: &NetGeneralInfo) -> Result<(), StarError> {
    let catalog = NuclideCatalog::builtin_ref();
    let has = |name: &str| {
        catalog
            .get_by_name(name)
            .is_some_and(|iso| net_info.chem_id.contains(&iso.chem_id))
    };
    if has("h1") && has("he4") {
        Ok(())
    } else {
        Err(StarError::MicrophysicsError(
            "mesa/star requires both h1 and he4 in net isotopes".to_string(),
        ))
    }
}

/// Returns the first 0-based active zone and clipped $\log_{10} T$ that would
/// trigger `warn_rates_for_high_temp` during [`do_net`]
/// (`rates/private/rates_support.f90:90-93` and `star/private/struct_burn_mix.f90:784-796`).
///
/// Returns `None` when `warn_rates_for_high_temp` is `.false.` or while
/// `doing_struct_burn_mix` suppresses high-temperature rate warnings.
#[must_use]
pub fn high_temp_rate_warning_zone(model: &StarModel) -> Option<(usize, f64)> {
    if model.aux.doing_struct_burn_mix
        || !model.controls.output.warn_rates_for_high_temp
    {
        return None;
    }
    let max_logt = model.controls.max_logt_for_net;
    let max_abar = model.controls.max_abar_for_burning;
    (0..model.nz).find_map(|k| {
        if model.abar[k] > max_abar {
            return None;
        }
        let mut log_t = model.ln_t[k] / LN10;
        if max_logt > 0.0 && log_t > max_logt {
            log_t = max_logt;
        }
        if log_t >= model.controls.output.max_safe_logt_for_rates {
            Some((k, log_t))
        } else {
            None
        }
    })
}

/// Evaluates the nuclear reaction network across all `nz` zones in parallel
/// using thread-local `NetWorkState` workspaces (`do_net` in
/// `star/private/net.f90:62-436`).
///
/// Zones handled by operator-split 1-zone burning during `do_struct_burn_mix`
/// (`doing_struct_burn_mix && op_split_burn && T_start[k] >= op_split_burn_min_T`,
/// `star/private/net.f90:160-164`) are left untouched so `burn_avg_epsnuc` and post-burn
/// categories remain intact across Newton iterations.
///
/// # Errors
/// Returns [`StarError`] if `rate_factors` lookup or reaction evaluation fails.
pub fn do_net(
    model: &mut StarModel,
    net_info: &NetGeneralInfo,
    skip_jacobian: bool,
) -> Result<(), StarError> {
    require_h1_and_he4(net_info)?;
    if let Some((zone, log_t)) = high_temp_rate_warning_zone(model) {
        eprintln!(
            "WARNING: high temperature for rates (zone {}, logT = {:.6} >= {:.6}); rates have been truncated",
            zone + 1,
            log_t,
            model.controls.output.max_safe_logt_for_rates
        );
    }
    let report = model.controls.output.report_ierr;
    let res = do_net_inner(model, net_info, skip_jacobian);
    crate::controls_output::report_result_ierr(report, res)
}

fn do_net_inner(
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

    let max_logt = model.controls.max_logt_for_net;
    let max_abar = model.controls.max_abar_for_burning;
    let eps_fac = model.controls.eps_nuc_factor;
    let dxdt_fac = model.controls.dxdt_nuc_factor;
    if eps_fac == 0.0 && dxdt_fac == 0.0 {
        // `star/private/net.f90:59-77`: zero all zones (including
        // `burn_avg_epsnuc` and `burn_num_iters` when `op_split_burn`) and
        // return early.
        model.eps_nuc[..nz].fill(0.0);
        model.d_epsnuc_dlnd[..nz].fill(0.0);
        model.d_epsnuc_dlnt[..nz].fill(0.0);
        model.d_epsnuc_dx[..nz * species].fill(0.0);
        model.aux.eps_nuc_categories[..nz].fill([0.0; NUM_CATEGORIES]);
        model.dxdt_nuc[..nz * species].fill(0.0);
        model.d_dxdt_nuc_dlnd[..nz * species].fill(0.0);
        model.d_dxdt_nuc_dlnt[..nz * species].fill(0.0);
        model.d_dxdt_nuc_dx[..nz * species * species].fill(0.0);
        model.eps_nuc_neu_total[..nz].fill(0.0);
        if model.controls.op_split_burn {
            if model.aux.burn_avg_epsnuc.len() >= nz {
                model.aux.burn_avg_epsnuc[..nz].fill(0.0);
            }
            if model.aux.burn_num_iters.len() >= nz {
                model.aux.burn_num_iters[..nz].fill(0);
            }
        }
        return Ok(());
    }

    let rate_factors = build_rate_factors(&model.job, net_info)?;
    let use_3a_fl87 = model.job.use_3a_fl87;
    let use_suzuki = model.job.use_suzuki_weak_rates;
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

    let tau_gamma = nonlocal_nico_tau_gamma(model);

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

                if abar_slice[k] > max_abar {
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

                let mut stack_dydt = [[Quad::ZERO; NUM_RVS]; 128];
                let mut heap_dydt = Vec::new();
                let dydt: &mut [[Quad; NUM_RVS]] = if species <= 128 {
                    &mut stack_dydt[..species]
                } else {
                    heap_dydt.resize(species, [Quad::ZERO; NUM_RVS]);
                    &mut heap_dydt[..]
                };

                with_thread_local_work(net_info, |w| {
                    let (raw_eps, raw_deps_dt, raw_deps_drho, raw_neu) =
                        eval_net_with_work(
                            net_info,
                            w,
                            dydt,
                            // `star/private/net.f90:245-246` always passes
                            // `just_dxdt = .false.`; `skip_jacobian` only
                            // discards the partials below.
                            false,
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
                    if !tau_gamma.is_empty() {
                        eps_val = attenuate_nonlocal_nico(eps_val, tau_gamma[k]);
                    }
                    let mut deps_drho = raw_deps_drho;
                    let mut deps_dt = if clipped_t { 0.0 } else { raw_deps_dt };
                    let clamped_eps = eps_val.abs() > max_abs_eps_nuc;
                    if clamped_eps {
                        eps_val = max_abs_eps_nuc.copysign(eps_val);
                        deps_drho = 0.0;
                        deps_dt = 0.0;
                    }

                    let direct = net_info.doing_approx21;
                    let raw_cats = if direct {
                        &w.direct_eps_nuc_categories
                    } else {
                        &w.eps_nuc_categories
                    };
                    *cats_k = scale_eps_nuc_categories(
                        raw_cats,
                        eps_val,
                        eps_fac,
                        direct,
                    );

                    *eps_k = eps_val * eps_fac;
                    *deps_d_k = deps_drho * rho_k * eps_fac;
                    *deps_t_k = deps_dt * t_k * eps_fac;
                    // `star/private/net.f90:255,365-372`: `eps_nuc_neu_total`
                    // is not multiplied by `eps_nuc_factor`.
                    *neu_k = raw_neu;

                    for i in 0..species {
                        let mut dxdt_i = if direct {
                            w.direct_dxdt[i]
                        } else {
                            (Quad::from_f64(aion[i]) * dydt[i][0]).to_f64()
                        };
                        if dxdt_fac != 1.0 {
                            dxdt_i *= dxdt_fac;
                        }
                        dxdt_k[i] = dxdt_i;
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
                                let mut val = if direct {
                                    w.direct_d_eps_nuc_dx[i]
                                } else {
                                    QCONV * w.d_eps_nuc_dy[i] / a_i
                                };
                                if eps_fac != 1.0 {
                                    val *= eps_fac;
                                }
                                val
                            };
                            let mut d_rho = if direct {
                                w.direct_d_dxdt_drho[i]
                            } else {
                                (Quad::from_f64(a_i) * dydt[i][2]).to_f64()
                            };
                            if dxdt_fac != 1.0 {
                                d_rho *= dxdt_fac;
                            }
                            ddxdt_d_k[i] = d_rho * rho_k;
                            ddxdt_t_k[i] = if clipped_t {
                                0.0
                            } else {
                                let mut d_t = if direct {
                                    w.direct_d_dxdt_dt[i]
                                } else {
                                    (Quad::from_f64(a_i) * dydt[i][1]).to_f64()
                                };
                                if dxdt_fac != 1.0 {
                                    d_t *= dxdt_fac;
                                }
                                d_t * t_k
                            };
                            let row_off = i * species;
                            for j in 0..species {
                                let mut jac = if direct {
                                    w.direct_d_dxdt_dx[row_off + j]
                                } else {
                                    w.d_dydt_dy[row_off + j] * a_i / aion[j]
                                };
                                if dxdt_fac != 1.0 {
                                    jac *= dxdt_fac;
                                }
                                ddxdt_x_k[row_off + j] = jac;
                            }
                        }
                    }
                    Ok(())
                })
            },
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controls::{StarControls, StarJobControls};
    use crate::micro::{do_eos, resolve_model_net, set_basic_vars};
    use flexmesa_eos::EosConfig;

    #[test]
    fn test_eval_zone_eps_nuc_categories_matches_do_net_approx21() {
        let g =
            NetGeneralInfo::from_name("approx21.net").expect("approx21.net");
        let cat = NuclideCatalog::builtin_ref();
        let job = StarJobControls {
            net_name: String::from("approx21.net"),
            ..Default::default()
        };
        let ctrls = StarControls {
            initial_z: 0.02,
            zbase: 0.02,
            ..Default::default()
        };
        let mut model = StarModel::new(1, g.chem_id.clone(), job, ctrls);
        model.eos_config = EosConfig::default();
        model.lnd[0] = 7.6 * LN10;
        model.ln_t[0] = 8.95 * LN10;
        model.ln_r[0] = 20.0;
        model.dq[0] = 1.0;
        model.mstar = 1.988_409_870_698_051e33;
        model.xmstar = model.mstar;
        model.dm[0] = model.mstar;
        model.xa.fill(1.0e-20);
        for (iso, val) in
            [("he4", 0.08), ("c12", 0.42), ("o16", 0.45), ("ni56", 0.05)]
        {
            let p = g
                .chem_id
                .iter()
                .position(|&c| cat.get_by_id(c).is_some_and(|n| n.name == iso))
                .expect("iso in approx21");
            model.xa[p] = val;
        }
        let s: f64 = model.xa.iter().sum();
        for v in &mut model.xa {
            *v /= s;
        }
        set_basic_vars(&mut model).expect("set_basic_vars");
        do_eos(&mut model).expect("do_eos");
        let net_info = resolve_model_net(&model).expect("resolve_model_net");
        do_net(&mut model, &net_info, false).expect("do_net");
        let rf = build_rate_factors(&model.job, &net_info).expect("rf");
        let zone_cats = eval_zone_eps_nuc_categories(
            &net_info,
            0,
            &model.xa,
            model.t[0],
            model.ln_t[0] / LN10,
            model.rho[0],
            model.lnd[0] / LN10,
            model.abar[0],
            model.zbar[0],
            model.z2bar[0],
            model.ye[0],
            model.eta[0],
            model.eos_d_dlnt[0][I_ETA],
            model.eos_d_dlnd[0][I_ETA],
            &rf,
            model.controls.weak_rate_factor,
            model.controls.screening_mode,
            model.job.use_3a_fl87,
            model.job.use_suzuki_weak_rates,
            model.controls.max_logt_for_net,
            model.controls.max_abar_for_burning,
            model.controls.eps_nuc_factor,
            model.controls.dxdt_nuc_factor,
            model.controls.max_abs_eps_nuc,
            None,
        )
        .expect("eval_zone_eps_nuc_categories");
        for c in 0..NUM_CATEGORIES {
            assert_eq!(
                zone_cats[c].to_bits(),
                model.aux.eps_nuc_categories[0][c].to_bits(),
                "cat {c} mismatch between do_net and eval_zone"
            );
        }
    }

    /// General (non-approx21) nets must keep paired-reaction energy categories
    /// in the post-burn `do1_net` call, because upstream reaches `net_get`
    /// with `just_dxdt = .false.` (`star/private/net.f90:245-246`).
    #[test]
    fn test_eval_zone_eps_nuc_categories_matches_do_net_general_net() {
        let g = NetGeneralInfo::from_name("co_burn_plus.net")
            .expect("co_burn_plus.net");
        let cat = NuclideCatalog::builtin_ref();
        let job = StarJobControls {
            net_name: String::from("co_burn_plus.net"),
            ..Default::default()
        };
        let ctrls = StarControls {
            initial_z: 0.02,
            zbase: 0.02,
            ..Default::default()
        };
        let mut model = StarModel::new(1, g.chem_id.clone(), job, ctrls);
        model.eos_config = EosConfig::default();
        model.lnd[0] = 6.0 * LN10;
        model.ln_t[0] = 9.2 * LN10;
        model.ln_r[0] = 20.0;
        model.dq[0] = 1.0;
        model.mstar = 1.988_409_870_698_051e33;
        model.xmstar = model.mstar;
        model.dm[0] = model.mstar;
        model.xa.fill(1.0e-20);
        for (iso, val) in [
            ("he4", 0.05),
            ("c12", 0.25),
            ("o16", 0.45),
            ("ne20", 0.20),
            ("mg24", 0.05),
        ] {
            let p = g
                .chem_id
                .iter()
                .position(|&c| cat.get_by_id(c).is_some_and(|n| n.name == iso))
                .expect("iso in co_burn_plus.net");
            model.xa[p] = val;
        }
        let s: f64 = model.xa.iter().sum();
        for v in &mut model.xa {
            *v /= s;
        }
        set_basic_vars(&mut model).expect("set_basic_vars");
        do_eos(&mut model).expect("do_eos");
        let net_info = resolve_model_net(&model).expect("resolve_model_net");
        do_net(&mut model, &net_info, false).expect("do_net");
        let rf = build_rate_factors(&model.job, &net_info).expect("rf");
        let zone_cats = eval_zone_eps_nuc_categories(
            &net_info,
            0,
            &model.xa,
            model.t[0],
            model.ln_t[0] / LN10,
            model.rho[0],
            model.lnd[0] / LN10,
            model.abar[0],
            model.zbar[0],
            model.z2bar[0],
            model.ye[0],
            model.eta[0],
            model.eos_d_dlnt[0][I_ETA],
            model.eos_d_dlnd[0][I_ETA],
            &rf,
            model.controls.weak_rate_factor,
            model.controls.screening_mode,
            model.job.use_3a_fl87,
            model.job.use_suzuki_weak_rates,
            model.controls.max_logt_for_net,
            model.controls.max_abar_for_burning,
            model.controls.eps_nuc_factor,
            model.controls.dxdt_nuc_factor,
            model.controls.max_abs_eps_nuc,
            None,
        )
        .expect("eval_zone_eps_nuc_categories");
        let total: f64 = zone_cats.iter().map(|c| c.abs()).sum();
        assert!(total > 0.0, "paired categories were dropped");
        for c in 0..NUM_CATEGORIES {
            assert_eq!(
                zone_cats[c].to_bits(),
                model.aux.eps_nuc_categories[0][c].to_bits(),
                "cat {c} mismatch between do_net and eval_zone"
            );
        }

        // `skip_jacobian` only discards partials; categories are unchanged.
        let cats_full = model.aux.eps_nuc_categories[0];
        do_net(&mut model, &net_info, true).expect("do_net skip_jacobian");
        for c in 0..NUM_CATEGORIES {
            assert_eq!(
                cats_full[c].to_bits(),
                model.aux.eps_nuc_categories[0][c].to_bits(),
                "cat {c} changed under skip_jacobian"
            );
        }
    }

    /// Three `basic.net` zones for the non-local Ni/Co tests: zones 0 and 2
    /// burn hydrogen at the same `(rho, T)`, zone 1 is carbon/oxygen with
    /// `abar > max_abar_for_burning = 5`, so `do1_net` skips it.
    fn nonlocal_nico_three_zone_model() -> (StarModel, NetGeneralInfo) {
        let g = NetGeneralInfo::from_name("basic.net").expect("basic.net");
        let cat = NuclideCatalog::builtin_ref();
        let job = StarJobControls {
            net_name: String::from("basic.net"),
            ..Default::default()
        };
        let ctrls = StarControls {
            initial_z: 0.02,
            zbase: 0.02,
            max_abar_for_burning: 5.0,
            ..Default::default()
        };
        let nz = 3;
        let mut model = StarModel::new(nz, g.chem_id.clone(), job, ctrls);
        model.eos_config = EosConfig::default();
        model.mstar = 1.988_409_870_698_051e33;
        model.xmstar = model.mstar;
        let sp = model.species;
        let pos = |iso: &str| {
            g.chem_id
                .iter()
                .position(|&c| cat.get_by_id(c).is_some_and(|n| n.name == iso))
                .expect("iso in basic.net")
        };
        model.xa.fill(1.0e-20);
        for k in 0..nz {
            model.lnd[k] = 2.0 * LN10;
            model.ln_t[k] = 7.4 * LN10;
            model.ln_r[k] = 21.0 - 0.5 * (k as f64);
            let mix: &[(&str, f64)] = if k == 1 {
                &[("c12", 0.5), ("o16", 0.5)]
            } else {
                &[("h1", 0.70), ("he4", 0.28), ("c12", 0.02)]
            };
            for &(iso, val) in mix {
                model.xa[k * sp + pos(iso)] = val;
            }
            let s: f64 = model.xa[k * sp..(k + 1) * sp].iter().sum();
            for v in &mut model.xa[k * sp..(k + 1) * sp] {
                *v /= s;
            }
        }
        model.dq.copy_from_slice(&[0.2, 0.3, 0.5]);
        set_basic_vars(&mut model).expect("set_basic_vars");
        do_eos(&mut model).expect("do_eos");
        // Set the geometry `tau_gamma` reads after the setup routines.
        model.dm.copy_from_slice(&[1.0e30, 2.0e30, 3.0e30]);
        model.rmid.copy_from_slice(&[1.0e10, 8.0e9, 5.0e9]);
        (model, g)
    }

    /// `nonlocal_NiCo_kap_gamma > 0` without `nonlocal_NiCo_decay_heat`
    /// scales `eps_nuc(k)` by `1 - exp(-tau_gamma)`, `tau_gamma` summing
    /// `dm(kk)/(pi4*rmid(kk)*rmid(kk))` from the surface to zone `k`,
    /// skipped zones included, times `nonlocal_NiCo_kap_gamma`
    /// (`star/private/net.f90:301-309`). Order-unity `tau_gamma`.
    #[test]
    fn test_nonlocal_nico_kap_gamma_attenuates_eps_nuc() {
        let (mut model, _) = nonlocal_nico_three_zone_model();
        let net_info = resolve_model_net(&model).expect("resolve_model_net");
        assert!(model.abar[1] > 5.0 && model.abar[0] < 5.0);
        do_net(&mut model, &net_info, false).expect("do_net");
        let plain = [model.eps_nuc[0], model.eps_nuc[1], model.eps_nuc[2]];
        assert!(plain[0] != 0.0 && plain[2] != 0.0);
        assert_eq!(plain[1], 0.0, "zone 1 is skipped (abar)");

        let col = |kk: usize| {
            model.dm[kk] / (FOUR_PI * model.rmid[kk] * model.rmid[kk])
        };
        let (c0, c1, c2) = (col(0), col(1), col(2));
        // `tau_gamma(3) = 1.5` at the innermost zone.
        let kap = 1.5 / (c0 + c1 + c2);
        let tau = [c0 * kap, (c0 + c1) * kap, ((c0 + c1) + c2) * kap];
        assert!(tau[0] > 0.05 && tau[2] > 1.0);

        model.controls.nonlocal_nico_kap_gamma = kap;
        do_net(&mut model, &net_info, false).expect("do_net kap_gamma");
        for k in [0, 2] {
            let want = plain[k] * (1.0 - flexmesa_math::exp(-tau[k]));
            assert_eq!(model.eps_nuc[k].to_bits(), want.to_bits(), "zone {k}");
            assert!(
                model.eps_nuc[k].abs() > 0.0
                    && model.eps_nuc[k].abs() < plain[k].abs(),
                "zone {k}: {} vs {}",
                model.eps_nuc[k],
                plain[k]
            );
            // Categories are renormalized to the attenuated `eps_nuc`
            // (`net.f90:318-327`).
            let cat_sum: f64 = model.aux.eps_nuc_categories[k].iter().sum();
            assert!(
                (cat_sum - model.eps_nuc[k]).abs()
                    <= 1.0e-12 * model.eps_nuc[k].abs()
            );
        }
        assert_eq!(model.eps_nuc[1], 0.0);
        // The skipped zone's column enters zone 2's `tau_gamma`: leaving it
        // out would give a different value.
        let tau_without_skipped = (c0 + c2) * kap;
        let wrong = plain[2] * (1.0 - flexmesa_math::exp(-tau_without_skipped));
        assert_ne!(model.eps_nuc[2].to_bits(), wrong.to_bits());

        model.controls.nonlocal_nico_decay_heat = true;
        do_net(&mut model, &net_info, false).expect("do_net decay_heat");
        for k in 0..3 {
            assert_eq!(model.eps_nuc[k].to_bits(), plain[k].to_bits());
        }
    }

    /// The attenuation comes before the `max_abs_eps_nuc` clamp
    /// (`star/private/net.f90:301-316`): a limit between the attenuated and
    /// the plain value leaves the attenuated value unclamped, and a limit
    /// below it clamps to the limit.
    #[test]
    fn test_nonlocal_nico_attenuation_precedes_max_abs_eps_nuc_clamp() {
        let (mut model, _) = nonlocal_nico_three_zone_model();
        let net_info = resolve_model_net(&model).expect("resolve_model_net");
        do_net(&mut model, &net_info, false).expect("do_net");
        let plain = model.eps_nuc[2];
        let col: f64 = (0..3)
            .map(|kk| {
                model.dm[kk] / (FOUR_PI * model.rmid[kk] * model.rmid[kk])
            })
            .sum();
        model.controls.nonlocal_nico_kap_gamma = 1.0 / col;
        do_net(&mut model, &net_info, false).expect("do_net kap_gamma");
        let att = model.eps_nuc[2];
        assert!(att.abs() < plain.abs());

        let between = 0.5 * (att.abs() + plain.abs());
        model.controls.max_abs_eps_nuc = between;
        do_net(&mut model, &net_info, false).expect("do_net clamp between");
        // Clamping first would give `between*(1 - exp(-tau))`.
        assert_eq!(model.eps_nuc[2].to_bits(), att.to_bits());
        assert!(
            model.d_epsnuc_dlnt[2] != 0.0,
            "partials kept when unclamped"
        );

        let below = 0.5 * att.abs();
        model.controls.max_abs_eps_nuc = below;
        do_net(&mut model, &net_info, false).expect("do_net clamp below");
        assert_eq!(model.eps_nuc[2].to_bits(), below.copysign(att).to_bits());
        assert_eq!(model.d_epsnuc_dlnt[2], 0.0);
    }

    /// The final `do1_net` of an op-split burn
    /// (`star/private/struct_burn_mix.f90:1112`) applies the same
    /// attenuation before the clamp when it rescales the categories.
    #[test]
    fn test_eval_zone_categories_apply_nonlocal_nico_attenuation() {
        let (mut model, _) = nonlocal_nico_three_zone_model();
        let net_info = resolve_model_net(&model).expect("resolve_model_net");
        let col: f64 = (0..3)
            .map(|kk| {
                model.dm[kk] / (FOUR_PI * model.rmid[kk] * model.rmid[kk])
            })
            .sum();
        model.controls.nonlocal_nico_kap_gamma = 1.0 / col;
        do_net(&mut model, &net_info, false).expect("do_net kap_gamma");
        let tau = nonlocal_nico_tau_gamma(&model);
        let rf = crate::micro::build_rate_factors(&model.job, &net_info)
            .expect("rate factors");
        let k = 2;
        let sp = model.species;
        let zone_cats = eval_zone_eps_nuc_categories(
            &net_info,
            k,
            &model.xa[k * sp..(k + 1) * sp],
            model.t[k],
            model.ln_t[k] / LN10,
            model.rho[k],
            model.lnd[k] / LN10,
            model.abar[k],
            model.zbar[k],
            model.z2bar[k],
            model.ye[k],
            model.eta[k],
            model.eos_d_dlnt[k][I_ETA],
            model.eos_d_dlnd[k][I_ETA],
            &rf,
            model.controls.weak_rate_factor,
            model.controls.screening_mode,
            model.job.use_3a_fl87,
            model.job.use_suzuki_weak_rates,
            model.controls.max_logt_for_net,
            model.controls.max_abar_for_burning,
            model.controls.eps_nuc_factor,
            model.controls.dxdt_nuc_factor,
            model.controls.max_abs_eps_nuc,
            tau.get(k).copied(),
        )
        .expect("eval_zone_eps_nuc_categories");
        for c in 0..NUM_CATEGORIES {
            assert_eq!(
                zone_cats[c].to_bits(),
                model.aux.eps_nuc_categories[k][c].to_bits(),
                "cat {c}"
            );
        }
    }

    #[test]
    fn test_require_h1_and_he4_accepts_basic_net_and_rejects_missing_isotope() {
        let mut g = NetGeneralInfo::from_name("basic.net").expect("basic.net");
        assert!(require_h1_and_he4(&g).is_ok());
        let he4 = NuclideCatalog::builtin_ref()
            .get_by_name("he4")
            .expect("he4")
            .chem_id;
        g.chem_id.retain(|&c| c != he4);
        let err = require_h1_and_he4(&g).expect_err("he4 missing");
        assert!(err.to_string().contains("h1 and he4"), "{err}");
    }
}
