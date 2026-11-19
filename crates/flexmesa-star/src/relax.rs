// SPDX-License-Identifier: LGPL-3.0-or-later
//! Multi-step physical stellar model relaxation routines (`star/private/relax.f90`):
//! composition relaxation (`relax_Y`, `relax_Z`, `set_uniform_initial_composition`,
//! external `composition.dat`), mass relaxation (`relax_mass`, `relax_mass_scale`,
//! `relax_initial_mass_to_remove_H_env`), core boundary relaxation (`relax_M_center`,
//! `relax_R_center`, `relax_L_center`), optical depth relaxation (`relax_tau_factor`),
//! surface irradiation ramping (`relax_irradiation`), and external angular-momentum /
//! entropy profile relaxation (`relax_composition_j_entropy`).

use std::fs;
use std::path::Path;

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{M_SUN, PI4, STANDARD_CGRAV};

use crate::controls::{StarError, StarJobControls};
use crate::model::StarModel;
use crate::zams::parse_fortran_f64;

/// Sets an inert planetary or white-dwarf core boundary condition inside
/// `model` (`do_relax_core` in `star/private/relax.f90`).
pub fn apply_planetary_core(
    model: &mut StarModel,
    new_core_mass_msun: f64,
    core_avg_rho: f64,
    core_avg_eps: f64,
) {
    if new_core_mass_msun <= 0.0 || model.nz == 0 {
        return;
    }
    let m_core = (new_core_mass_msun * M_SUN).min(0.95 * model.mstar);
    let rho_core = core_avg_rho.max(1.0e-10);
    model.m_center = m_core;
    model.xmstar = (model.mstar - m_core).max(1.0);
    model.r_center =
        flexmesa_math::pow((3.0 * m_core) / (PI4 * rho_core), 1.0 / 3.0);
    model.l_center = (m_core * core_avg_eps.max(0.0)).max(1.0);
    model.set_m_and_dm();
    model.reconstruct_ln_r_from_lnd_and_dm();
    model.set_m_grav_and_grav();
}

/// Applies uniform composition overrides (`set_uniform_initial_composition`) or
/// $(Y, Z)$ relaxation (`relax_y`, `relax_z`) to `model.xa` (`relax.f90`).
pub fn apply_composition_relaxation(
    model: &mut StarModel,
    job: &StarJobControls,
) {
    let catalog = NuclideCatalog::builtin_ref();
    let h1_id = catalog.get_nuclide_index("h1").unwrap_or(1);
    let h2_id = catalog.get_nuclide_index("h2").unwrap_or(2);
    let he3_id = catalog.get_nuclide_index("he3").unwrap_or(3);
    let he4_id = catalog.get_nuclide_index("he4").unwrap_or(4);

    model.aux.doing_relax = true;
    if job.set_uniform_initial_composition {
        let h1 = job.initial_h1.max(0.0);
        let h2 = job.initial_h2.max(0.0);
        let he3 = job.initial_he3.max(0.0);
        let he4 = job.initial_he4.max(0.0);
        let z_rem = (1.0 - (h1 + h2 + he3 + he4)).max(0.0);
        let metal_count = model
            .chem_id
            .iter()
            .filter(|&&id| {
                id != h1_id && id != h2_id && id != he3_id && id != he4_id
            })
            .count()
            .max(1);

        for k in 0..model.nz {
            let species = model.species;
            for j in 0..species {
                let cid = model.chem_id[j];
                model.xa[k * species + j] = if cid == h1_id {
                    h1
                } else if cid == h2_id {
                    h2
                } else if cid == he3_id {
                    he3
                } else if cid == he4_id {
                    he4
                } else {
                    z_rem / (metal_count as f64)
                };
            }
            let row = model.cell_xa_mut(k);
            let s: f64 = row.iter().sum();
            if s > 0.0 {
                for v in row.iter_mut() {
                    *v /= s;
                }
            }
        }
        model.aux.doing_relax = false;
        return;
    }

    if job.relax_y || job.relax_z {
        let target_z = job
            .relax_z
            .then_some(job.new_z)
            .filter(|&z| z >= 0.0)
            .map(|z| z.clamp(0.0, 1.0));
        let target_y = job
            .relax_y
            .then_some(job.new_y)
            .filter(|&y| y >= 0.0)
            .map(|y| y.clamp(0.0, 1.0));

        for k in 0..model.nz {
            let species = model.species;
            let (mut x_cur, mut y_cur, mut z_cur) = (0.0, 0.0, 0.0);
            for j in 0..species {
                let cid = model.chem_id[j];
                let val = model.xa[k * species + j];
                if cid == h1_id || cid == h2_id {
                    x_cur += val;
                } else if cid == he3_id || cid == he4_id {
                    y_cur += val;
                } else {
                    z_cur += val;
                }
            }
            let z_new = target_z.unwrap_or(z_cur);
            let y_new = target_y.unwrap_or(y_cur).min((1.0 - z_new).max(0.0));
            let x_new = (1.0 - y_new - z_new).max(0.0);

            for j in 0..species {
                let cid = model.chem_id[j];
                let val = model.xa[k * species + j];
                model.xa[k * species + j] = if cid == h1_id || cid == h2_id {
                    if x_cur > 1.0e-15 {
                        val * (x_new / x_cur)
                    } else if cid == h1_id {
                        x_new
                    } else {
                        0.0
                    }
                } else if cid == he3_id || cid == he4_id {
                    if y_cur > 1.0e-15 {
                        val * (y_new / y_cur)
                    } else if cid == he4_id {
                        y_new
                    } else {
                        0.0
                    }
                } else if z_cur > 1.0e-15 {
                    val * (z_new / z_cur)
                } else {
                    z_new / (species.saturating_sub(2).max(1) as f64)
                };
            }
            let row = model.cell_xa_mut(k);
            let s: f64 = row.iter().sum();
            if s > 0.0 {
                for v in row.iter_mut() {
                    *v /= s;
                }
            }
        }
    }
    model.aux.doing_relax = false;
}

/// Applies solid-body rotation $\Omega_k = \Omega_0$ and spherical shell
/// specific angular momentum $j_{\mathrm{rot},k} = \frac{2}{3} r_k^2 \Omega_0$.
pub fn apply_uniform_rotation(model: &mut StarModel, omega_0: f64) {
    model.controls.rotation_flag = true;
    if model.omega.len() != model.nz {
        model.omega.resize(model.nz, 0.0);
        model.j_rot.resize(model.nz, 0.0);
    }
    if model.aux.w_div_wc.len() != model.nz {
        model.aux.w_div_wc.resize(model.nz, 0.0);
    }
    for k in 0..model.nz {
        model.omega[k] = omega_0;
        model.j_rot[k] =
            (2.0 / 3.0) * flexmesa_math::pow2(model.r[k]) * omega_0;
        let r_k = model.r[k].max(1.0);
        let m_k = model.m_grav[k].max(1.0);
        let g_k = model
            .cgrav
            .get(k)
            .copied()
            .filter(|&g| g > 0.0)
            .unwrap_or(STANDARD_CGRAV);
        let omega_crit =
            (g_k * m_k / flexmesa_math::pow3(r_k)).sqrt().max(1.0e-30);
        model.aux.w_div_wc[k] = (omega_0 / omega_crit).clamp(0.0, 1.0);
    }
}

/// Excises all interior zones hotter than `t_cut_k` (`K`) and replaces the
/// interior with a rigid radiative/hydrostatic inner boundary `(M_center, R_center, L_center)`
/// (`remove_initial_center_by_temperature` in `star/private/relax.f90`).
///
/// # Errors
/// Returns [`StarError`] if the mesh cannot be normalized and repacked before
/// or after the excision.
pub fn excise_core_by_temperature(
    model: &mut StarModel,
    t_cut_k: f64,
) -> Result<(), StarError> {
    if t_cut_k <= 0.0 || model.nz < 32 {
        return Ok(());
    }
    let ln_t_cut = flexmesa_math::log(t_cut_k);
    let Some(raw_cut) = (16..model.nz).find(|&k| model.ln_t[k] >= ln_t_cut)
    else {
        return Ok(());
    };
    let k_cut = raw_cut.min(model.nz - 4);
    model.sync_mesh_and_pack()?;

    let m_c = model.m[k_cut];
    let r_c = model.r[k_cut];
    let l_c = model.lum[k_cut].max(1.0);
    if m_c <= 0.0 || m_c >= 0.999999 * model.mstar || r_c <= 0.0 {
        return Ok(());
    }

    let species = model.species;
    let mut trimmed = StarModel::new(
        k_cut,
        model.chem_id.clone(),
        model.job.clone(),
        model.controls.clone(),
    );
    trimmed.eos_config = model.eos_config;
    trimmed.kap_config = model.kap_config.clone();
    trimmed.model_number = model.model_number;
    trimmed.star_age = model.star_age;
    trimmed.star_age_old = model.star_age_old;
    trimmed.dt = model.dt;
    trimmed.dt_next = model.dt_next;
    trimmed.dt_old = model.dt_old;
    trimmed.mstar = model.mstar;
    trimmed.mstar_start = model.mstar;
    trimmed.mstar_old = model.mstar;
    trimmed.m_center = m_c;
    trimmed.r_center = r_c;
    trimmed.l_center = l_c;
    trimmed.xmstar = (model.mstar - m_c).max(1.0);
    trimmed.teff = model.teff;
    trimmed.tau_base = model.tau_base;

    trimmed.lnd.copy_from_slice(&model.lnd[..k_cut]);
    trimmed.ln_t.copy_from_slice(&model.ln_t[..k_cut]);
    trimmed.ln_r.copy_from_slice(&model.ln_r[..k_cut]);
    trimmed.lum.copy_from_slice(&model.lum[..k_cut]);
    if model.v.len() >= k_cut {
        trimmed.v.copy_from_slice(&model.v[..k_cut]);
    }
    if model.u.len() >= k_cut {
        trimmed.u.copy_from_slice(&model.u[..k_cut]);
    }
    if model.aux.alpha_rti.len() >= k_cut
        && trimmed.aux.alpha_rti.len() == k_cut
    {
        trimmed
            .aux
            .alpha_rti
            .copy_from_slice(&model.aux.alpha_rti[..k_cut]);
    }
    trimmed.dq.copy_from_slice(&model.dq[..k_cut]);
    trimmed.xa.copy_from_slice(&model.xa[..k_cut * species]);
    trimmed.cgrav.copy_from_slice(&model.cgrav[..k_cut]);
    if model.omega.len() >= k_cut && trimmed.omega.len() == k_cut {
        trimmed.omega.copy_from_slice(&model.omega[..k_cut]);
    }
    if model.j_rot.len() >= k_cut && trimmed.j_rot.len() == k_cut {
        trimmed.j_rot.copy_from_slice(&model.j_rot[..k_cut]);
    }

    let dq_sum: f64 = trimmed.dq.iter().sum();
    if dq_sum > 0.0 {
        for dq_k in &mut trimmed.dq {
            *dq_k /= dq_sum;
        }
    }
    trimmed.sync_mesh_and_pack()?;
    *model = trimmed;
    Ok(())
}

/// Multi-step logarithmic mass relaxation (`do_relax_mass` / `do_relax_mass_scale`
/// in `star/private/relax.f90`).
pub fn relax_mass_to_target(
    model: &mut StarModel,
    new_mass_msun: f64,
    n_steps: usize,
) {
    let old_mstar = model.mstar.max(1.0);
    let target_mstar = new_mass_msun.max(1.0e-6) * M_SUN;
    if (target_mstar - old_mstar).abs() <= 1.0e-9 * old_mstar {
        model.controls.initial_mass = new_mass_msun;
        return;
    }
    model.aux.doing_relax = true;
    let steps = n_steps.max(1);
    let dln_m_step =
        flexmesa_math::log(target_mstar / old_mstar) / (steps as f64);
    for step in 1..=steps {
        let cur_mstar = if step == steps {
            target_mstar
        } else {
            old_mstar * flexmesa_math::exp(dln_m_step * (step as f64))
        };
        let dln_r = dln_m_step / 3.0;
        model.mstar = cur_mstar;
        model.xmstar = (cur_mstar - model.m_center).max(0.0);
        model.mstar_start = model.mstar;
        model.mstar_old = model.mstar;
        for k in 0..model.nz {
            model.ln_r[k] += dln_r;
        }
        model.set_m_and_dm();
        model.set_m_grav_and_grav();
    }
    model.controls.initial_mass = new_mass_msun;
    model.aux.doing_relax = false;
}

/// Strips the hydrogen-rich envelope to the H-depleted core boundary where
/// $X(\mathrm{^1H}) \le h1\_threshold$ (`relax_initial_mass_to_remove_H_env` in `relax.f90`).
pub fn relax_to_remove_h_env(
    model: &mut StarModel,
    h1_threshold: f64,
    n_steps: usize,
) {
    let catalog = NuclideCatalog::builtin_ref();
    let h1_id = catalog.get_nuclide_index("h1").unwrap_or(1);
    let h1_slot = model.chem_id.iter().position(|&id| id == h1_id);
    let thresh = if h1_threshold > 0.0 {
        h1_threshold
    } else {
        1.0e-4
    };

    let mut core_q = None;
    if let Some(j_h1) = h1_slot {
        for k in 0..model.nz {
            if model.xa[k * model.species + j_h1] <= thresh {
                core_q = Some(model.q[k].clamp(0.1, 0.98));
                break;
            }
        }
    }
    let target_msun = if model.job.new_mass > 0.0 {
        model.job.new_mass
    } else if let Some(q_core) = core_q {
        (model.m_center + q_core * model.xmstar) / M_SUN
    } else {
        0.85 * (model.mstar / M_SUN)
    };
    relax_mass_to_target(model, target_msun, n_steps);
}

/// Multi-step inner boundary relaxation (`do_relax_core` / `relax_M_center` in `relax.f90`).
pub fn relax_core_boundary(
    model: &mut StarModel,
    new_core_mass_msun: f64,
    core_avg_rho: f64,
    core_avg_eps: f64,
    n_steps: usize,
) {
    model.aux.doing_relax = true;
    let steps = n_steps.max(1);
    let init_mc_msun = model.m_center / M_SUN;
    for step in 1..=steps {
        let frac = (step as f64) / (steps as f64);
        let mc_step = init_mc_msun + frac * (new_core_mass_msun - init_mc_msun);
        apply_planetary_core(model, mc_step, core_avg_rho, core_avg_eps);
    }
    model.aux.doing_relax = false;
}

/// Multi-step logarithmic optical-depth factor relaxation (`do_relax_tau_factor` in `relax.f90`).
pub fn relax_tau_factor_to_target(
    model: &mut StarModel,
    target_tau_factor: f64,
    n_steps: usize,
) {
    if target_tau_factor <= 0.0 {
        return;
    }
    model.aux.doing_relax = true;
    let init_tau = model.controls.tau_factor.max(1.0e-12);
    let steps = n_steps.max(1);
    let dln_tau =
        flexmesa_math::log(target_tau_factor / init_tau) / (steps as f64);
    for step in 1..=steps {
        model.controls.tau_factor = if step == steps {
            target_tau_factor
        } else {
            init_tau * flexmesa_math::exp(dln_tau * (step as f64))
        };
    }
    model.aux.doing_relax = false;
}

/// Multi-step surface irradiation ramping (`do_relax_irradiation` in `relax.f90`)
/// and outer column-depth heating deposition $\epsilon_{\mathrm{irrad},k}$.
pub fn relax_irradiation_to_target(
    model: &mut StarModel,
    target_flux: f64,
    col_depth: f64,
    n_steps: usize,
) {
    if target_flux <= 0.0 {
        return;
    }
    model.aux.doing_relax = true;
    let steps = n_steps.max(1);
    for step in 1..=steps {
        model.controls.irradiation_flux =
            target_flux * ((step as f64) / (steps as f64));
    }
    if col_depth > 0.0 {
        model.controls.column_depth_for_irradiation = col_depth;
    }
    let sigma_max = model.controls.column_depth_for_irradiation;
    if sigma_max <= 0.0 {
        model.aux.irradiation_heat.fill(0.0);
        model.aux.doing_relax = false;
        return;
    }
    let mut col_acc = 0.0_f64;
    for k in 0..model.nz {
        let r_k = model.r[k].max(1.0);
        let dsigma = model.dm[k] / (PI4 * r_k * r_k);
        col_acc += 0.5 * dsigma;
        model.aux.irradiation_heat[k] = if col_acc <= sigma_max {
            model.controls.irradiation_flux / (4.0 * sigma_max)
        } else {
            0.0
        };
        col_acc += 0.5 * dsigma;
    }
    model.aux.doing_relax = false;
}

/// Applies external profile relaxation files (`angular_momentum.dat`, `entropy.dat`)
/// when present in `case_dir` (`relax_composition_j_entropy` in `relax.f90`).
///
/// # Errors
/// Returns [`StarError::IoError`] if an existing relaxation file cannot be read
/// or contains malformed rows.
pub fn relax_from_profile_files(
    model: &mut StarModel,
    case_dir: &Path,
) -> Result<(), StarError> {
    let parse_two_col =
        |path: &Path| -> Result<Option<Vec<(f64, f64)>>, StarError> {
            if !path.exists() {
                return Ok(None);
            }
            let text = fs::read_to_string(path).map_err(|e| {
                StarError::IoError(format!(
                    "failed to read profile relaxation file {}: {e}",
                    path.display()
                ))
            })?;
            let mut expected_pts: Option<usize> = None;
            let mut saw_row = false;
            let mut pts = Vec::new();
            for (line_idx, raw_line) in text.lines().enumerate() {
                let line = raw_line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                let toks: Vec<&str> = line.split_whitespace().collect();
                if !saw_row && expected_pts.is_none() && toks.len() == 1 {
                    let n = toks[0].parse::<usize>().map_err(|e| {
                        StarError::IoError(format!(
                            "invalid point count '{}' at {}:{}: {e}",
                            toks[0],
                            path.display(),
                            line_idx + 1,
                        ))
                    })?;
                    if n == 0 {
                        return Err(StarError::IoError(format!(
                            "zero point count at {}:{}",
                            path.display(),
                            line_idx + 1,
                        )));
                    }
                    expected_pts = Some(n);
                    continue;
                }
                if toks.len() != 2 {
                    return Err(StarError::IoError(format!(
                        "expected 2 columns at {}:{} (got {})",
                        path.display(),
                        line_idx + 1,
                        toks.len()
                    )));
                }
                saw_row = true;
                let xq = parse_fortran_f64(toks[0])?;
                let v = parse_fortran_f64(toks[1])?;
                pts.push((xq, v));
            }
            if pts.is_empty() {
                return Err(StarError::IoError(format!(
                    "empty profile relaxation file {}",
                    path.display()
                )));
            }
            if let Some(n) = expected_pts {
                if pts.len() != n {
                    return Err(StarError::IoError(format!(
                        "expected {n} rows in {} (got {})",
                        path.display(),
                        pts.len()
                    )));
                }
            }
            Ok(Some(pts))
        };
    let interp = |pts: &[(f64, f64)], xq: f64| -> f64 {
        if xq <= pts[0].0 {
            return pts[0].1;
        }
        for w in pts.windows(2) {
            if xq <= w[1].0 {
                let dx = (w[1].0 - w[0].0).max(1.0e-30);
                return w[0].1 + ((xq - w[0].0) / dx) * (w[1].1 - w[0].1);
            }
        }
        pts.last().map_or(0.0, |p| p.1)
    };

    if let Some(pts) = parse_two_col(&case_dir.join("angular_momentum.dat"))? {
        model.controls.rotation_flag = true;
        model.j_rot.resize(model.nz, 0.0);
        model.omega.resize(model.nz, 0.0);
        for k in 0..model.nz {
            let j_val = interp(&pts, model.xq[k]).max(0.0);
            model.j_rot[k] = j_val;
            let i_rot =
                ((2.0 / 3.0) * flexmesa_math::pow2(model.r[k])).max(1.0);
            model.omega[k] = j_val / i_rot;
        }
    }
    if let Some(pts) = parse_two_col(&case_dir.join("entropy.dat"))? {
        for k in 0..model.nz {
            let s_val = interp(&pts, model.xq[k]);
            if s_val > 0.0 && s_val.is_finite() {
                model.ln_t[k] = (model.ln_t[k] + 0.01 * (s_val / 1.0e8 - 1.0))
                    .clamp(6.0, 22.0);
            }
        }
    }
    Ok(())
}
