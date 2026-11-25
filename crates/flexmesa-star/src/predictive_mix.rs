// SPDX-License-Identifier: GPL-3.0-or-later
//! Verbatim MESA `26.4.1` predictive mixing (`star/private/predictive_mix.f90`).
//!
//! Expands convective boundaries outward or inward cell-by-cell under a
//! trial-mixed homogeneous composition (including the nuclear burning
//! correction $\dot{X}_{\mathrm{nuc}}\Delta t$), solving the EOS at fixed
//! $(P_{\mathrm{gas}}, T)$ and re-evaluating opacity and MLT with
//! $\nabla_{L,\mathrm{comp}} = 0$ (Schwarzschild criterion). Updates
//! `model.d_mix`, `model.conv_vel`, `model.mixing_type`, `model.aux.mixing.cdc`,
//! and `model.aux.mixing.cz_bdy_dq` while leaving cell compositions and EOS
//! state restored to their pre-trial values.

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{MixingType, AMU, BOLTZM, FOUR_PI};
use flexmesa_num::find0;

use crate::controls::StarError;
use crate::hydro_vars::do1_mlt_2;
use crate::micro::eval_kap_for_range;
use crate::mixing_micro::{eval_uniform_xa_eos_kap_range, CellRangeSnapshot};
use crate::model::StarModel;

/// Applies MESA `26.4.1` predictive mixing across all located convective
/// boundaries (`add_predictive_mixing` in `star/private/predictive_mix.f90:35-193`).
///
/// Returns the total number of faces newly converted to
/// [`MixingType::ConvectiveMixing`].
///
/// # Errors
/// Returns [`StarError`] if a predictive mixing parameter set specifies an
/// invalid selector/isotope name, if both `do_conv_premix` and predictive
/// mixing are enabled simultaneously, if EOS/opacity/MLT evaluation fails, or
/// if a non-finite `d_mix` value is produced.
pub fn add_predictive_mixing(
    model: &mut StarModel,
) -> Result<usize, StarError> {
    let nz = model.nz;
    let n_bdy = model.aux.mixing.conv_bdy_loc.len();
    if nz < 2 || n_bdy == 0 {
        return Ok(0);
    }
    if !model
        .controls
        .mix_wind
        .predictive_sets
        .iter()
        .any(|ps| ps.predictive_mix)
    {
        return Ok(0);
    }

    let mut mix_mask = vec![false; nz];
    let mut total_faces = 0_usize;

    let conv_bdy_loc = model.aux.mixing.conv_bdy_loc.clone();
    let conv_bdy_q = model.aux.mixing.conv_bdy_q.clone();
    let top_conv_bdy = model.aux.mixing.top_conv_bdy.clone();
    let burn_h = model.aux.mixing.burn_h_conv_region.clone();
    let burn_he = model.aux.mixing.burn_he_conv_region.clone();
    let burn_z = model.aux.mixing.burn_z_conv_region.clone();
    let predictive_sets = model.controls.mix_wind.predictive_sets.clone();

    for i in 0..n_bdy {
        // Skip surface boundary (`conv_bdy_loc(i) == 1` in 1-based Fortran).
        if conv_bdy_loc[i] == 0 {
            continue;
        }

        for (j, ps) in predictive_sets.iter().enumerate() {
            if !ps.predictive_mix {
                continue;
            }

            let match_zone_type = match ps.zone_type.trim_end() {
                "burn_H" => burn_h[i],
                "burn_He" => burn_he[i],
                "burn_Z" => burn_z[i],
                "nonburn" => !(burn_h[i] || burn_he[i] || burn_z[i]),
                "any" => true,
                other => {
                    return Err(StarError::InvalidInput(format!(
                        "Invalid predictive_zone_type({}): '{other}'",
                        j + 1
                    )));
                }
            };

            let is_core_zone =
                i == 0 && model.r_center == 0.0 && top_conv_bdy[i];
            let is_surf_zone = if top_conv_bdy[i] {
                conv_bdy_loc[i] == 0
            } else {
                conv_bdy_loc.get(i + 1).copied() == Some(0)
            };

            let match_zone_loc = match ps.zone_loc.trim_end() {
                "core" => is_core_zone,
                "shell" => !(is_core_zone || is_surf_zone),
                "surf" => is_surf_zone,
                "any" => true,
                other => {
                    return Err(StarError::InvalidInput(format!(
                        "Invalid predictive_zone_loc({}): '{other}'",
                        j + 1
                    )));
                }
            };

            let match_bdy_loc = match ps.bdy_loc.trim_end() {
                "bottom" => !top_conv_bdy[i],
                "top" => top_conv_bdy[i],
                "any" => true,
                other => {
                    return Err(StarError::InvalidInput(format!(
                        "Invalid predictive_bdy_loc({}): '{other}'",
                        j + 1
                    )));
                }
            };

            if !(match_zone_type && match_zone_loc && match_bdy_loc) {
                continue;
            }

            if conv_bdy_q[i] < ps.bdy_q_min || conv_bdy_q[i] > ps.bdy_q_max {
                continue;
            }

            if model.controls.do_conv_premix {
                return Err(StarError::InvalidInput(String::from(
                    "Predictive mixing and convective premixing cannot be enabled at the same time",
                )));
            }

            total_faces += do_predictive_mixing(model, i, j, &mut mix_mask)?;

            // Apply at most a single predictive mixing set per boundary.
            break;
        }
    }

    for k in 0..nz {
        if !model.d_mix[k].is_finite() {
            return Err(StarError::ConvergenceFailure(format!(
                "add_predictive_mixing produced non-finite d_mix at face {k}: {}",
                model.d_mix[k]
            )));
        }
    }

    Ok(total_faces)
}

#[allow(clippy::too_many_lines)]
fn do_predictive_mixing(
    model: &mut StarModel,
    i: usize,
    j: usize,
    mix_mask: &mut [bool],
) -> Result<usize, StarError> {
    let nz = model.nz;
    let species = model.species;
    let ps = model.controls.mix_wind.predictive_sets[j].clone();

    let superad_thresh = ps.superad_thresh.max(0.0);
    let catalog = NuclideCatalog::builtin_ref();

    let ar_name = ps.avoid_reversal.trim();
    let iso_r: Option<usize> = if ar_name.is_empty() {
        None
    } else if let Some(entry) = catalog.get_by_name(ar_name) {
        model.net_iso.get(entry.chem_id).copied().flatten()
    } else {
        return Err(StarError::InvalidInput(format!(
            "Invalid isotope name in predictive_avoid_reversal: '{}'",
            ps.avoid_reversal
        )));
    };

    let li_name = ps.limit_ingestion.trim();
    let (iso_i, ingest_factor): (Option<usize>, f64) = if li_name.is_empty() {
        (None, 0.0)
    } else if let Some(entry) = catalog.get_by_name(li_name) {
        (
            model.net_iso.get(entry.chem_id).copied().flatten(),
            ps.ingestion_factor,
        )
    } else {
        return Err(StarError::InvalidInput(format!(
            "Invalid isotope name in predictive_limit_ingestion: '{}'",
            ps.limit_ingestion
        )));
    };

    // Determine cell indices spanning the initial convection zone, including
    // the cells containing the actual convective boundaries (`predictive_mix.f90:283-313`).
    let n_bdy = model.aux.mixing.conv_bdy_loc.len();
    let outward = model.aux.mixing.top_conv_bdy[i];
    let (k_top_cz, k_bot_cz) = if outward {
        let k_bot = if i == 0 {
            nz - 1
        } else {
            if model.aux.mixing.top_conv_bdy[i - 1] {
                return Err(StarError::InvalidInput(format!(
                    "Double top boundary in predictive_mix; i={i}"
                )));
            }
            model.aux.mixing.conv_bdy_loc[i - 1].saturating_sub(1)
        };
        let k_top = model.aux.mixing.conv_bdy_loc[i];
        (k_top, k_bot)
    } else {
        let k_bot = model.aux.mixing.conv_bdy_loc[i].saturating_sub(1);
        let k_top = if i + 1 == n_bdy {
            0
        } else {
            if !model.aux.mixing.top_conv_bdy[i + 1] {
                return Err(StarError::InvalidInput(format!(
                    "Double bottom boundary in predictive_mix; i={i}"
                )));
            }
            model.aux.mixing.conv_bdy_loc[i + 1]
        };
        (k_top, k_bot)
    };

    if k_top_cz > k_bot_cz || k_bot_cz >= nz {
        return Ok(0);
    }

    let mut xa_cz = vec![0.0_f64; species];
    let mut xa_cz_burn = vec![0.0_f64; species];
    eval_abundances(model, k_bot_cz, k_top_cz, &mut xa_cz, &mut xa_cz_burn)?;

    let mut ledoux_extension = model.controls.use_ledoux_criterion;

    let mut k_bot_ez = k_bot_cz;
    let mut k_top_ez = k_top_cz;
    let mut xa_ez = vec![0.0_f64; species];
    let mut xa_ez_burn = vec![0.0_f64; species];
    eval_abundances(model, k_bot_ez, k_top_ez, &mut xa_ez, &mut xa_ez_burn)?;

    let mut xa_mz = vec![0.0_f64; species];
    let mut xa_mz_burn = vec![0.0_f64; species];
    let mut d_arr = vec![0.0_f64; nz];
    let mut vc_arr = vec![0.0_f64; nz];
    let mut grada_arr = vec![0.0_f64; nz];
    let mut gradr_arr = vec![0.0_f64; nz];

    // Use signed indices during `search_loop` so boundary exit + back-off
    // mirrors 1-based Fortran `k_top_mz < 1` / `k_bot_mz > s%nz` verbatim.
    let mut k_top_mz_i = k_top_cz as isize;
    let mut k_bot_mz_i = k_bot_cz as isize;
    let nz_i = nz as isize;

    loop {
        if outward {
            k_top_mz_i -= 1;
        } else {
            k_bot_mz_i += 1;
        }

        if (outward && k_top_mz_i < 0) || (!outward && k_bot_mz_i >= nz_i) {
            break;
        }

        let k_top_mz = k_top_mz_i as usize;
        let k_bot_mz = k_bot_mz_i as usize;

        eval_abundances(
            model,
            k_bot_mz,
            k_top_mz,
            &mut xa_mz,
            &mut xa_mz_burn,
        )?;

        let (k_a, k_b) = eval_mixing_coeffs(
            model,
            k_bot_mz,
            k_top_mz,
            &xa_mz_burn,
            &mut d_arr,
            &mut vc_arr,
            &mut grada_arr,
            &mut gradr_arr,
        )?;

        if ledoux_extension {
            // Check unmixed Schwarzschild stability across `k_a..=k_b`.
            let all_schwarzschild = (k_a..=k_b)
                .all(|k| model.gradr[k] > model.aux.mixing.grada_face[k]);
            if all_schwarzschild {
                k_bot_ez = k_bot_mz;
                k_top_ez = k_top_mz;
                eval_abundances(
                    model,
                    k_bot_ez,
                    k_top_ez,
                    &mut xa_ez,
                    &mut xa_ez_burn,
                )?;
            } else {
                ledoux_extension = false;
            }
        }

        if !ledoux_extension {
            if let Some(ir) = iso_r {
                let s_mz = (xa_mz_burn[ir] - xa_ez[ir]) < 0.0;
                let s_ez = (xa_ez_burn[ir] - xa_ez[ir]) < 0.0;
                if s_mz != s_ez {
                    break;
                }
            }

            if let Some(ii) = iso_i {
                let dm_mz: f64 = model.dm[k_top_mz..=k_bot_mz].iter().sum();
                let dm_ez: f64 = model.dm[k_top_ez..=k_bot_ez].iter().sum();
                let m_ingest = xa_mz[ii] * dm_mz - xa_ez[ii] * dm_ez;
                let k_lim = if outward { k_a } else { k_b };
                let m_ingest_limit = eval_ingest_limit(
                    model,
                    &grada_arr,
                    &gradr_arr,
                    ingest_factor,
                    k_lim,
                );
                if m_ingest > m_ingest_limit {
                    break;
                }
            }
        }

        // Check growing boundary face for non-convective exit.
        if (outward && gradr_arr[k_a] < grada_arr[k_a])
            || (!outward && gradr_arr[k_b] < grada_arr[k_b])
        {
            break;
        }

        // Check interior faces for split-zone proximity (`superad_thresh`).
        let superad_min =
            eval_superad_min(&gradr_arr, &grada_arr, outward, k_a, k_b);
        if superad_min <= superad_thresh {
            break;
        }
    }

    // Back off the mixing by one zone (`predictive_mix.f90:497-503`).
    if outward {
        k_top_mz_i += 1;
    } else {
        k_bot_mz_i -= 1;
    }
    let k_top_mz = k_top_mz_i as usize;
    let k_bot_mz = k_bot_mz_i as usize;

    // Verbatim `predictive_mix.f90:507-514`: upstream checks `ANY(mix_mask)`
    // and then writes `.false.` rather than `.true.` in the `else` branch
    // (tracked under #169 / #266).
    if mix_mask[k_top_mz..=k_bot_mz].iter().any(|&v| v) {
        return Err(StarError::ConvergenceFailure(String::from(
            "Double predictive mixing overlap detected",
        )));
    }
    mix_mask[k_top_mz..=k_bot_mz].fill(false);

    if (outward && k_top_mz == k_top_cz) || (!outward && k_bot_mz == k_bot_cz) {
        return Ok(0);
    }

    // Re-evaluate mixed abundances and mixing coefficients at the backed-off extent.
    eval_abundances(model, k_bot_mz, k_top_mz, &mut xa_mz, &mut xa_mz_burn)?;
    let (k_a_eval, k_b_eval) = eval_mixing_coeffs(
        model,
        k_bot_mz,
        k_top_mz,
        &xa_mz_burn,
        &mut d_arr,
        &mut vc_arr,
        &mut grada_arr,
        &mut gradr_arr,
    )?;

    // Update `cdc`, `d_mix`, `conv_vel`, and `mixing_type` across newly mixed faces,
    // stopping if an adjacent convection zone is encountered (`predictive_mix.f90:550-583`).
    let mut updated_faces = 0_usize;
    let mut k_b_final = if outward { k_a_eval } else { k_b_eval };

    if outward {
        // Iterate `k = k_top_cz` down to `k_a_eval` inclusive (`dk = -1`).
        let mut k = k_top_cz;
        loop {
            if model.mixing_type[k] == MixingType::ConvectiveMixing {
                k_b_final = k + 1;
                break;
            }
            apply_predicted_face(model, k, d_arr[k], vc_arr[k]);
            updated_faces += 1;
            if k <= k_a_eval {
                break;
            }
            k -= 1;
        }
    } else {
        // Iterate `k = k_bot_cz + 1` up to `k_b_eval` inclusive (`dk = +1`).
        for k in (k_bot_cz + 1)..=k_b_eval {
            if model.mixing_type[k] == MixingType::ConvectiveMixing {
                k_b_final = k - 1;
                break;
            }
            apply_predicted_face(model, k, d_arr[k], vc_arr[k]);
            updated_faces += 1;
        }
    }

    // Update sub-cell boundary mass fraction `cz_bdy_dq` (`predictive_mix.f90:589-621`).
    if outward {
        model.aux.mixing.cz_bdy_dq[k_top_cz] = 0.0;
        if k_b_final > 0 {
            let dg0 = model.aux.mixing.grada_face[k_b_final - 1]
                - model.gradr[k_b_final - 1];
            let dg1 = grada_arr[k_b_final] - gradr_arr[k_b_final];
            if dg0 * dg1 < 0.0 {
                let x0 = find0(0.0, dg0, model.dq[k_top_mz], dg1).map_err(
                    |e| {
                        StarError::ConvergenceFailure(format!(
                            "find0 failed in do_predictive_mixing: k={k_top_mz}: {e}"
                        ))
                    },
                )?;
                if x0 < 0.0 || x0 > model.dq[k_top_mz] {
                    return Err(StarError::ConvergenceFailure(format!(
                        "bad cz_bdy_dq in do_predictive_mixing: k={k_top_mz}, x0={x0}, dq={}",
                        model.dq[k_top_mz]
                    )));
                }
                model.aux.mixing.cz_bdy_dq[k_top_mz] = x0;
            }
        }
    } else {
        model.aux.mixing.cz_bdy_dq[k_bot_cz] = 0.0;
        if k_b_final + 1 < nz {
            let dg0 = grada_arr[k_b_final] - gradr_arr[k_b_final];
            let dg1 = model.aux.mixing.grada_face[k_b_final + 1]
                - model.gradr[k_b_final + 1];
            if dg0 * dg1 < 0.0 {
                let x0 = find0(0.0, dg0, model.dq[k_bot_mz], dg1).map_err(
                    |e| {
                        StarError::ConvergenceFailure(format!(
                            "find0 failed in do_predictive_mixing: k={k_bot_mz}: {e}"
                        ))
                    },
                )?;
                if x0 < 0.0 || x0 > model.dq[k_bot_mz] {
                    return Err(StarError::ConvergenceFailure(format!(
                        "bad cz_bdy_dq in do_predictive_mixing: k={k_bot_mz}, x0={x0}, dq={}",
                        model.dq[k_bot_mz]
                    )));
                }
                model.aux.mixing.cz_bdy_dq[k_bot_mz] = x0;
            }
        }
    }

    Ok(updated_faces)
}

fn apply_predicted_face(
    model: &mut StarModel,
    k: usize,
    d_val: f64,
    vc_val: f64,
) {
    let rho = if k > 0 {
        (model.dq[k - 1] * model.rho[k] + model.dq[k] * model.rho[k - 1])
            / (model.dq[k - 1] + model.dq[k])
    } else {
        model.rho[k]
    };
    let factor = FOUR_PI * model.r[k] * model.r[k] * rho;
    model.aux.mixing.cdc[k] = factor * factor * d_val;
    model.d_mix[k] = d_val;
    model.conv_vel[k] = vc_val;
    model.mixing_type[k] = MixingType::ConvectiveMixing;
}

fn eval_superad_min(
    gradr: &[f64],
    grada: &[f64],
    outward: bool,
    k_a: usize,
    k_b: usize,
) -> f64 {
    let (lo, hi) = if outward {
        (k_a + 1, k_b)
    } else {
        (k_a, k_b.saturating_sub(1))
    };
    if lo > hi {
        return f64::MAX;
    }
    let mut min_ratio = f64::MAX;
    for k in lo..=hi {
        let ratio = gradr[k] / grada[k];
        if ratio < min_ratio {
            min_ratio = ratio;
        }
    }
    min_ratio - 1.0
}

/// Evaluates mass-weighted mean abundances over cells `k_top..=k_bot` without
/// and with the nuclear burning increment `dxdt_nuc * dt`
/// (`eval_abundances` in `star/private/predictive_mix.f90:633-667`).
pub(crate) fn eval_abundances(
    model: &StarModel,
    k_bot: usize,
    k_top: usize,
    xa: &mut [f64],
    xa_burn: &mut [f64],
) -> Result<(), StarError> {
    let sp = model.species;
    // Upstream reads `s% dxdt_nuc` unconditionally
    // (`star/private/predictive_mix.f90:654`); a model without the array
    // cannot supply the burning correction.
    if model.dxdt_nuc.len() != model.nz * sp {
        return Err(StarError::InvalidInput(format!(
            "predictive_mix eval_abundances: dxdt_nuc has length {}, expected nz*species = {}",
            model.dxdt_nuc.len(),
            model.nz * sp
        )));
    }
    let mc: f64 = model.dm[k_top..=k_bot].iter().sum();
    if mc <= 0.0 || sp == 0 {
        xa.fill(0.0);
        xa_burn.fill(0.0);
        return Ok(());
    }
    let dt = model.dt;

    for l in 0..sp {
        let mut s_xa = 0.0;
        let mut s_burn = 0.0;
        for k in k_top..=k_bot {
            let dm_k = model.dm[k];
            let idx = k * sp + l;
            s_xa += dm_k * model.xa[idx];
            s_burn += dm_k * model.dxdt_nuc[idx] * dt;
        }
        let x_raw = s_xa / mc;
        xa[l] = x_raw.clamp(0.0, 1.0);
        xa_burn[l] = (x_raw + s_burn / mc).clamp(0.0, 1.0);
    }

    let sum_xa: f64 = xa.iter().sum();
    if sum_xa > 0.0 {
        for v in xa.iter_mut() {
            *v /= sum_xa;
        }
    }
    let sum_burn: f64 = xa_burn.iter().sum();
    if sum_burn > 0.0 {
        for v in xa_burn.iter_mut() {
            *v /= sum_burn;
        }
    }
    Ok(())
}

/// Evaluates trial mixing coefficients (`D`, `vc`, `grada`, `gradr`) across
/// faces `k_a..=k_b` (`k_top_mz + 1 ..= k_bot_mz`) under homogenized
/// composition `xa_mx` at fixed `(Pgas, T)` (`eval_mixing_coeffs` in
/// `star/private/predictive_mix.f90:670-859`).
///
/// After the trial, the saved cell EOS state and composition are put back and
/// the cell opacities and face MLT values are recomputed rather than copied
/// back: upstream calls `do_kap_for_cell` on every cell of the mixed region
/// and then `do1_mlt_2` with its default arguments (`mixing_length_alpha` and
/// the `use_Ledoux_criterion` composition term) on every face
/// (`star/private/predictive_mix.f90:813-852`). The recomputed `gradT`,
/// `mlt_*` and `L_conv` therefore reflect the restored state, including any
/// difference between the default MLT arguments and those of the earlier
/// call that produced the values on entry.
#[allow(clippy::too_many_arguments)]
fn eval_mixing_coeffs(
    model: &mut StarModel,
    k_bot_mz: usize,
    k_top_mz: usize,
    xa_mx: &[f64],
    d_out: &mut [f64],
    vc_out: &mut [f64],
    grada_out: &mut [f64],
    gradr_out: &mut [f64],
) -> Result<(usize, usize), StarError> {
    let k_a = k_top_mz + 1;
    let k_b = k_bot_mz;

    let cell_snap = CellRangeSnapshot::capture(model, k_top_mz, k_bot_mz);

    let has_alpha = model.aux.mixing.alpha_mlt.len() == model.nz
        && model.aux.mixing.alpha_mlt.iter().any(|&v| v > 0.0);
    let result = (|| -> Result<(), StarError> {
        eval_uniform_xa_eos_kap_range(model, k_top_mz, k_bot_mz, xa_mx)?;
        // `do1_mlt_2` updates each face `k` in place before reading out MLT results.
        #[allow(clippy::manual_memcpy)]
        for k in k_a..=k_b {
            let alpha_k = if has_alpha {
                model.aux.mixing.alpha_mlt[k]
            } else {
                model.controls.mixing_length_alpha
            };
            // Explicitly pass `gradL_composition_term = 0.0` (Schwarzschild).
            let _ = do1_mlt_2(model, k, Some(alpha_k), Some(0.0))?;
            d_out[k] = model.mlt_d[k];
            // MESA `predictive_mix.f90:806`: `vc(k) = s% mlt_vc(k)`.
            vc_out[k] = model.mlt_vc[k];
            grada_out[k] = model.aux.mixing.grada_face[k];
            gradr_out[k] = model.gradr[k];
        }
        Ok(())
    })();

    // Put back the saved cell EOS state and composition
    // (`restore_cell_loop`, `star/private/predictive_mix.f90:815-841`). The
    // trial only overwrote those fields and the opacities, so restoring the
    // whole snapshot is equivalent.
    cell_snap.restore(model);
    // Upstream returns `ierr` from a failed trial EOS or opacity call before
    // its restore loops (`star/private/predictive_mix.f90:762-779`), and a
    // failed trial `do1_mlt_2` is fatal (`:802`); the error ends the step
    // either way, so the recomputation below runs only after a clean trial.
    result?;

    // Recompute opacity from the restored cell state (`do_kap_for_cell` in
    // `restore_cell_loop`, `star/private/predictive_mix.f90:837-838`).
    eval_kap_for_range(model, k_top_mz, k_bot_mz)?;
    // Recompute face MLT with default arguments (`restore_face_loop`,
    // `star/private/predictive_mix.f90:848-854`).
    for k in k_a..=k_b {
        let _ = do1_mlt_2(model, k, None, None)?;
    }
    Ok((k_a, k_b))
}

/// Evaluates the Spruit (2015) mass ingestion limit at face `k`
/// (`eval_ingest_limit` in `star/private/predictive_mix.f90:930-963`).
pub(crate) fn eval_ingest_limit(
    model: &StarModel,
    grada: &[f64],
    gradr: &[f64],
    ingest_factor: f64,
    k: usize,
) -> f64 {
    let t_face = if k == 0 {
        model.t[0]
    } else {
        let alfa = model.dq[k - 1] / (model.dq[k - 1] + model.dq[k]);
        let beta = 1.0 - alfa;
        alfa * model.t[k] + beta * model.t[k - 1]
    };
    ingest_factor
        * (AMU * model.lum[k] / (BOLTZM * t_face))
        * (1.0 - grada[k] / gradr[k])
        * model.dt
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controls::{StarControls, StarJobControls};
    use crate::hydro_vars::set_hydro_vars;
    use crate::zams::load_zams_model;

    /// The trial in `eval_mixing_coeffs` must end as upstream's does
    /// (`star/private/predictive_mix.f90:813-852`): the cell state is put back
    /// and the opacities and face MLT values are recomputed, the latter with
    /// `do1_mlt_2` default arguments. A non-default `alpha_mlt` used by the
    /// trial and a stale `gradT` on entry must both be replaced by the
    /// default-argument recomputation.
    #[test]
    fn test_eval_mixing_coeffs_recomputes_kap_and_default_mlt_on_reset() {
        let ctrl = StarControls {
            initial_mass: 4.0,
            zbase: 0.02,
            ..StarControls::default()
        };
        let mut model = load_zams_model(ctrl, StarJobControls::default())
            .expect("4 Msun ZAMS");
        set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");

        let nz = model.nz;
        let sp = model.species;
        let k_top_mz = nz - 40;
        let k_bot_mz = nz - 30;
        let (k_a, k_b) = (k_top_mz + 1, k_bot_mz);

        // Trial composition: the cell at the bottom of the window, applied to
        // every cell of the window.
        let xa_mx = model.xa[k_bot_mz * sp..(k_bot_mz + 1) * sp].to_vec();
        // Non-default per-face alpha read only by the trial call, and a
        // stale `gradT` that a copy-back reset would keep.
        let alpha_trial = 2.5 * model.controls.mixing_length_alpha;
        model.aux.mixing.alpha_mlt.fill(alpha_trial);
        for k in k_a..=k_b {
            model.gradt[k] = -1.0;
        }

        // Expected state: what upstream's restore loops compute on the
        // untouched model.
        let mut expected = model.clone();
        eval_kap_for_range(&mut expected, k_top_mz, k_bot_mz)
            .expect("expected kap");
        for k in k_a..=k_b {
            do1_mlt_2(&mut expected, k, None, None).expect("expected mlt");
        }

        let xa_before = model.xa.clone();
        let lnd_before = model.lnd.clone();
        let mut d = vec![0.0; nz];
        let mut vc = vec![0.0; nz];
        let mut grada = vec![0.0; nz];
        let mut gradr = vec![0.0; nz];
        let (ka, kb) = eval_mixing_coeffs(
            &mut model, k_bot_mz, k_top_mz, &xa_mx, &mut d, &mut vc,
            &mut grada, &mut gradr,
        )
        .expect("eval_mixing_coeffs");
        assert_eq!((ka, kb), (k_a, k_b));

        assert_eq!(model.xa, xa_before);
        assert_eq!(model.lnd, lnd_before);
        for k in k_top_mz..=k_bot_mz {
            assert_eq!(
                model.opacity[k].to_bits(),
                expected.opacity[k].to_bits(),
                "opacity at cell {k}"
            );
        }
        let mx = &model.aux.mixing;
        let ex = &expected.aux.mixing;
        for k in k_a..=k_b {
            assert_ne!(model.gradt[k], -1.0, "stale gradT kept at face {k}");
            for (name, got, want) in [
                ("gradT", model.gradt[k], expected.gradt[k]),
                ("gradr", model.gradr[k], expected.gradr[k]),
                ("gradL", model.gradl[k], expected.gradl[k]),
                ("mlt_D", model.mlt_d[k], expected.mlt_d[k]),
                ("mlt_vc", model.mlt_vc[k], expected.mlt_vc[k]),
                ("mlt_cdc", model.mlt_cdc[k], expected.mlt_cdc[k]),
                ("mlt_Gamma", model.mlt_gamma[k], expected.mlt_gamma[k]),
                (
                    "mlt_mixing_length",
                    mx.mlt_mixing_length[k],
                    ex.mlt_mixing_length[k],
                ),
                ("grada_face", mx.grada_face[k], ex.grada_face[k]),
                ("L_conv", mx.l_conv[k], ex.l_conv[k]),
            ] {
                assert_eq!(got.to_bits(), want.to_bits(), "{name} at face {k}");
            }
            assert_eq!(mx.mlt_mixing_type[k], ex.mlt_mixing_type[k]);
            // The reset uses `mixing_length_alpha`, not the trial alpha.
            let lambda_default =
                model.controls.mixing_length_alpha * model.scale_height[k];
            assert_eq!(
                mx.mlt_mixing_length[k].to_bits(),
                lambda_default.to_bits(),
                "mixing length at face {k}"
            );
        }
    }
}
