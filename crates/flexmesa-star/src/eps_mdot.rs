// SPDX-License-Identifier: LGPL-3.0-or-later
//! Compressional and advective mass-change heating (`calculate_eps_mdot`,
//! `find_mass_flux`, `interpolate_onto_faces`, and `set_leak_frac` in
//! `star/private/eps_mdot.f90`).

use crate::model::StarModel;

fn eval_cell_ke(model: &StarModel, k: usize) -> f64 {
    let vel = if model.controls.u_flag {
        model.u[k]
    } else if model.controls.v_flag {
        model.v[k]
    } else {
        0.0
    };
    0.5 * vel * vel
}

/// Computes the cell-centered specific kinetic energy profile
/// $\tfrac{1}{2} v_k^2$ ($\mathrm{erg\,g^{-1}}$) across all `0..nz` zones.
#[must_use]
pub(crate) fn eval_specific_kinetic_energy_profile(
    model: &StarModel,
) -> Vec<f64> {
    (0..model.nz).map(|k| eval_cell_ke(model, k)).collect()
}

/// Computes the specific total energy profile $e_{\mathrm{tot},k} = e_k -
/// \frac{1}{2} c_{\mathrm{mc}} \left(\frac{G m_{\mathrm{grav},k}}{r_k} +
/// \frac{G m_{\mathrm{grav},k+1}}{r_{k+1}}\right) + \tfrac{1}{2} v_k^2$
/// ($\mathrm{erg\,g^{-1}}$) across all `0..nz` zones
/// (`cell_specific_PE` in `star_utils.f90:2245-2283`).
#[must_use]
pub fn eval_specific_total_energy_profile(model: &StarModel) -> Vec<f64> {
    let nz = model.nz;
    let use_mc = model.controls.use_mass_corrections;
    let mut te = vec![0.0; nz];
    for (k, te_k) in te.iter_mut().enumerate() {
        let mc = if use_mc {
            model.aux.mass_correction.get(k).copied().unwrap_or(1.0)
        } else {
            1.0
        };
        let grav_00 = if model.r[k] > 0.0 {
            -model.cgrav[k] * model.m_grav[k] / model.r[k]
        } else {
            0.0
        };
        let (r_p1, m_p1, g_p1) = if k + 1 < nz {
            (model.r[k + 1], model.m_grav[k + 1], model.cgrav[k + 1])
        } else {
            (model.r_center, model.m_center, model.cgrav[k])
        };
        let grav_p1 = if r_p1 > 0.0 { -g_p1 * m_p1 / r_p1 } else { 0.0 };
        let e_grav = 0.5 * mc * (grav_00 + grav_p1);
        *te_k = model.energy[k] + e_grav + eval_cell_ke(model, k);
    }
    te
}

/// Computes downward inter-cell mass flux $F_k$ (`len = nz + 1`, with
/// $F_{\mathrm{nz}} = 0$ at the center and $F_0 = \Delta M$ at the surface)
/// from $\Delta(\mathrm{d}m_k) = \mathrm{d}m_{\mathrm{new},k} -
/// \mathrm{d}m_{\mathrm{old},k}$ (`find_mass_flux` in `eps_mdot.f90:64-80`).
#[must_use]
pub fn find_mass_flux(change_in_dm: &[f64]) -> Vec<f64> {
    let nz = change_in_dm.len();
    let mut flux = vec![0.0; nz + 1];
    for k in (0..nz).rev() {
        flux[k] = flux[k + 1] + change_in_dm[k];
    }
    flux
}

fn interpolate_onto_faces(vec: &[f64], dm: &[f64], k: usize) -> f64 {
    let nz = vec.len();
    if nz == 0 {
        return 0.0;
    }
    if k == 0 {
        return vec[0];
    }
    if k >= nz {
        return vec[nz - 1];
    }
    let denom = (dm[k - 1] + dm[k]).max(1.0e-99);
    let beta = dm[k - 1] / denom;
    let alpha = 1.0 - beta;
    alpha * vec[k - 1] + beta * vec[k]
}

/// Evaluates the radiative entropy-leakage fraction `leak_frac[k]` $\in [0, 1]$
/// (`set_leak_frac` in `eps_mdot.f90:142-187`).
#[must_use]
pub fn compute_leak_frac(
    model: &StarModel,
    mass_flux: &[f64],
    dt: f64,
    leak_frac_factor: f64,
) -> Vec<f64> {
    let nz = model.nz;
    let mut leak = vec![1.0; nz];
    if nz == 0 || dt <= 0.0 {
        return leak;
    }

    for k in 0..nz {
        let grad_t_mid = if k + 1 < nz {
            0.5 * (model.gradt[k] + model.gradt[k + 1])
        } else if nz >= 2 {
            0.5 * (model.gradt[nz - 2] + model.gradt[nz - 1])
        } else {
            model.gradt[0]
        };
        let grad_diff = (model.grada[k] - grad_t_mid).abs();
        let flux_bar = if k == 0 {
            mass_flux[0]
        } else {
            0.5 * (mass_flux[k - 1] + mass_flux[k])
        };
        let e_therm = (model.cp[k] * model.t[k]).max(1.0e-30);
        if grad_diff > 0.0 && flux_bar.abs() > 0.0 {
            let raw = leak_frac_factor
                * (model.lum[k] * dt / (grad_diff * e_therm * flux_bar)).abs();
            leak[k] = raw.clamp(0.0, 1.0);
        }
    }
    leak
}

/// Pre-mass-adjustment stellar state snapshots required for cancellation-free
/// `f64` $\epsilon_{\dot{M}}$ evaluation across all zones, especially Region III
/// ($k \ge k_B$).
pub struct PrevMassState<'a> {
    /// Previous cell masses $\Delta m_k$ in grams (`len = nz`).
    pub dm: &'a [f64],
    /// Previous outer face radii $r_k$ in cm (`len = nz`; empty for legacy fallback).
    pub r: &'a [f64],
    /// Previous outer face gravitational masses $m_{\mathrm{grav},k}$ in grams (`len = nz`).
    pub m_grav: &'a [f64],
    /// Previous specific internal energies $e_k$ in $\mathrm{erg\,g^{-1}}$ (`len = nz`).
    pub energy: &'a [f64],
    /// Previous total pressures $P_{\mathrm{eos},k}$ in $\mathrm{dyn\,cm^{-2}}$ (`len = nz`).
    pub peos: &'a [f64],
    /// Previous mass densities $\rho_k$ in $\mathrm{g\,cm^{-3}}$ (`len = nz`).
    pub rho: &'a [f64],
    /// Previous specific total energies $e_{\mathrm{tot},k}$ in $\mathrm{erg\,g^{-1}}$ (`len = nz`).
    pub te: &'a [f64],
    /// Previous specific kinetic energies $\tfrac{1}{2} v_k^2$ in $\mathrm{erg\,g^{-1}}$ (`len = nz`).
    pub ke: &'a [f64],
}

/// Evaluates the Lagrangian mass-adjustment heating rate $\epsilon_{\dot{M},k}$
/// ($\mathrm{erg\,g^{-1}\,s^{-1}}$) across all zones (`calculate_eps_mdot` in
/// `star/private/eps_mdot.f90:478-704`) and stores it in `model.eps_mdot`,
/// computing `change_in_dm = model.dm - prev_dm`.
pub fn calculate_eps_mdot(
    model: &mut StarModel,
    prev_dm: &[f64],
    prev_te: &[f64],
    prev_peos: &[f64],
    prev_rho: &[f64],
    dt: f64,
) {
    let nz = model.nz;
    if prev_dm.len() != nz {
        model.eps_mdot.resize(nz, 0.0);
        model.eps_mdot.fill(0.0);
        return;
    }
    let change_in_dm: Vec<f64> =
        (0..nz).map(|k| model.dm[k] - prev_dm[k]).collect();
    let prev = PrevMassState {
        dm: prev_dm,
        r: &[],
        m_grav: &[],
        energy: &[],
        peos: prev_peos,
        rho: prev_rho,
        te: prev_te,
        ke: &[],
    };
    calculate_eps_mdot_with_change(model, &prev, &change_in_dm, dt);
}

/// Evaluates $\epsilon_{\dot{M},k}$ ($\mathrm{erg\,g^{-1}\,s^{-1}}$) across
/// all zones using an explicit precomputed `change_in_dm` vector and optional
/// cancellation-free `prev` state snapshots (`adjust_mass.f90:98-132` and
/// `eps_mdot.f90:620-634`).
pub fn calculate_eps_mdot_with_change(
    model: &mut StarModel,
    prev: &PrevMassState<'_>,
    change_in_dm: &[f64],
    dt: f64,
) {
    let nz = model.nz;
    if model.eps_mdot.len() != nz {
        model.eps_mdot.resize(nz, 0.0);
    }
    let eps_fac = model.controls.mix_wind.eps_mdot_factor;
    if nz == 0
        || dt <= 0.0
        || eps_fac == 0.0
        || prev.dm.len() != nz
        || prev.peos.len() != nz
        || prev.rho.len() != nz
        || prev.te.len() != nz
        || change_in_dm.len() != nz
    {
        model.eps_mdot.fill(0.0);
        return;
    }

    let mass_flux = find_mass_flux(change_in_dm);
    if mass_flux[0].abs() <= 1.0e-30 {
        model.eps_mdot.fill(0.0);
        return;
    }

    let mut rho_w_flux = vec![0.0; nz + 1];
    for k in (0..nz).rev() {
        rho_w_flux[k] =
            rho_w_flux[k + 1] + change_in_dm[k] / prev.rho[k].max(1.0e-99);
    }

    let p_bar: Vec<f64> = (0..=nz)
        .map(|k| interpolate_onto_faces(prev.peos, prev.dm, k))
        .collect();
    let rho_bar: Vec<f64> = (0..=nz)
        .map(|k| interpolate_onto_faces(prev.rho, prev.dm, k).max(1.0e-99))
        .collect();
    let te_bar: Vec<f64> = (0..=nz)
        .map(|k| interpolate_onto_faces(prev.te, prev.dm, k))
        .collect();

    let use_mc = model.controls.use_mass_corrections;
    let cancellation_free = prev.r.len() == nz
        && prev.m_grav.len() == nz
        && prev.energy.len() == nz
        && prev.ke.len() == nz;

    let d_r: Vec<f64> = if cancellation_free {
        (0..nz)
            .map(|k| {
                if rho_w_flux[k] == 0.0 {
                    0.0
                } else {
                    let r_old = prev.r[k].max(1.0);
                    let r_new = model.r[k].max(1.0);
                    (3.0 / (4.0 * std::f64::consts::PI)) * rho_w_flux[k]
                        / (r_new * r_new + r_new * r_old + r_old * r_old)
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    let d_grav_face = |k: usize| -> f64 {
        if k >= nz || (mass_flux[k] == 0.0 && d_r[k] == 0.0) {
            return 0.0;
        }
        let mc_f = if use_mc {
            model.aux.mass_correction.get(k).copied().unwrap_or(1.0)
        } else {
            1.0
        };
        let dm_grav = mass_flux[k] * mc_f;
        let r_old = prev.r[k].max(1.0);
        let r_new = model.r[k].max(1.0);
        -model.cgrav[k] * (r_old * dm_grav - prev.m_grav[k] * d_r[k])
            / (r_old * r_new)
    };

    let ke_after = if cancellation_free {
        eval_specific_kinetic_energy_profile(model)
    } else {
        Vec::new()
    };
    let te_after = if cancellation_free {
        Vec::new()
    } else {
        eval_specific_total_energy_profile(model)
    };

    let mut power_cells = vec![0.0; nz];
    let mut total_mass_through_cell = vec![0.0; nz];
    for k in 0..nz {
        let dm_k = model.dm[k].max(1.0e-99);
        total_mass_through_cell[k] = (prev.dm[k] + mass_flux[k].max(0.0)
            - mass_flux[k + 1].min(0.0))
        .max(1.0e-99);
        let adv_te = change_in_dm[k] * (te_bar[k] - prev.te[k])
            + mass_flux[k + 1] * (te_bar[k] - te_bar[k + 1]);
        let pv_k = p_bar[k] * (mass_flux[k] / rho_bar[k] - rho_w_flux[k]);
        let pv_k1 = p_bar[k + 1]
            * (mass_flux[k + 1] / rho_bar[k + 1] - rho_w_flux[k + 1]);
        let d_cell_te = if cancellation_free {
            let mc = if use_mc {
                model.aux.mass_correction.get(k).copied().unwrap_or(1.0)
            } else {
                1.0
            };
            let d_pe = 0.5 * mc * (d_grav_face(k) + d_grav_face(k + 1));
            let d_e_int = model.energy[k] - prev.energy[k];
            let d_ke = ke_after[k] - prev.ke[k];
            dm_k * (d_e_int + d_pe + d_ke)
        } else {
            dm_k * (te_after[k] - prev.te[k])
        };
        let d_heat = adv_te + (pv_k - pv_k1) - d_cell_te;
        power_cells[k] = d_heat / dt;
    }

    let leak_frac = compute_leak_frac(
        model,
        &mass_flux,
        dt,
        model.controls.mix_wind.eps_mdot_leak_frac_factor,
    );

    // Propagate un-leaked entropy perturbations along the direction of mass
    // flow (`leak_control` / `leak` in `eps_mdot.f90:189-470`).
    let mut eps_out = vec![0.0; nz];
    if mass_flux[0] > 0.0 {
        let mut carried_erg_s = 0.0_f64;
        for k in 0..nz {
            let dm_k = model.dm[k].max(1.0e-99);
            if mass_flux[k] <= 0.0 && mass_flux[k + 1] <= 0.0 {
                if carried_erg_s != 0.0 {
                    let target = k.saturating_sub(1);
                    let dm_t = model.dm[target].max(1.0e-99);
                    eps_out[target] += carried_erg_s / dm_t;
                    carried_erg_s = 0.0;
                }
                eps_out[k] = power_cells[k] / dm_k;
                continue;
            }
            if leak_frac[k] >= 1.0 {
                eps_out[k] = (power_cells[k] + carried_erg_s) / dm_k;
                carried_erg_s = 0.0;
            } else {
                let stay_frac =
                    (dm_k / total_mass_through_cell[k]).clamp(0.0, 1.0);
                let stay_power = stay_frac * power_cells[k];
                let thru_power =
                    (1.0 - stay_frac) * power_cells[k] + carried_erg_s;
                let is_last_down = mass_flux[k + 1] <= 0.0;
                if is_last_down {
                    eps_out[k] = (stay_power + thru_power) / dm_k;
                    carried_erg_s = 0.0;
                } else {
                    eps_out[k] =
                        (stay_power + leak_frac[k] * thru_power) / dm_k;
                    carried_erg_s = (1.0 - leak_frac[k]) * thru_power;
                }
            }
        }
    } else {
        let mut carried_erg_s = 0.0_f64;
        for k in (0..nz).rev() {
            let dm_k = model.dm[k].max(1.0e-99);
            if mass_flux[k] >= 0.0 {
                eps_out[k] = power_cells[k] / dm_k;
                continue;
            }
            if leak_frac[k] >= 1.0 {
                eps_out[k] = (power_cells[k] + carried_erg_s) / dm_k;
                carried_erg_s = 0.0;
            } else {
                let stay_frac =
                    (dm_k / total_mass_through_cell[k]).clamp(0.0, 1.0);
                let stay_power = stay_frac * power_cells[k];
                let thru_power =
                    (1.0 - stay_frac) * power_cells[k] + carried_erg_s;
                eps_out[k] = (stay_power + leak_frac[k] * thru_power) / dm_k;
                carried_erg_s = (1.0 - leak_frac[k]) * thru_power;
            }
        }
    }

    for k in 0..nz {
        model.eps_mdot[k] = eps_fac * eps_out[k];
    }
}
