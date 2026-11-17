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

/// Evaluates the single-zone specific total energy $e_{\mathrm{tot},k} = e_k -
/// \frac{1}{2} c_{\mathrm{mc}} \left(\frac{G m_{\mathrm{grav},k}}{r_k} +
/// \frac{G m_{\mathrm{grav},k+1}}{r_{k+1}}\right) + \tfrac{1}{2} v_k^2$
/// ($\mathrm{erg\,g^{-1}}$) at zone `k` (`eval_cell_section_total_energy` in
/// `star/private/star_utils.f90:2245-2283` and `profile_getval.f90:2022-2023`).
#[must_use]
#[inline]
pub fn eval_cell_specific_total_energy(model: &StarModel, k: usize) -> f64 {
    let nz = model.nz;
    if k >= nz {
        return 0.0;
    }
    let mc = if model.controls.use_mass_corrections {
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
    model.energy[k] + e_grav + eval_cell_ke(model, k)
}

/// Computes the specific total energy profile $e_{\mathrm{tot},k}$
/// ($\mathrm{erg\,g^{-1}}$) across all `0..nz` zones
/// (`cell_specific_PE` in `star_utils.f90:2245-2283`).
#[must_use]
pub fn eval_specific_total_energy_profile(model: &StarModel) -> Vec<f64> {
    (0..model.nz)
        .map(|k| eval_cell_specific_total_energy(model, k))
        .collect()
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
            rho_w_flux[k + 1] + change_in_dm[k] / model.rho[k].max(1.0e-99);
    }

    let p_bar: Vec<f64> = (0..=nz)
        .map(|k| interpolate_onto_faces(&model.peos, prev.dm, k))
        .collect();
    let rho_bar: Vec<f64> = (0..=nz)
        .map(|k| interpolate_onto_faces(&model.rho, prev.dm, k).max(1.0e-99))
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
                    let (r_old, r_new) =
                        (prev.r[k].max(1.0), model.r[k].max(1.0));
                    (3.0 / (4.0 * flexmesa_const::PI)) * rho_w_flux[k]
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
    let eps_out = leak_control_parcel(
        nz,
        &mass_flux,
        &model.dm,
        prev.dm,
        &total_mass_through_cell,
        &power_cells,
        &leak_frac,
    );
    for k in 0..nz {
        model.eps_mdot[k] = eps_fac * eps_out[k];
    }
}

fn make_compressed_intersect(
    dm1: &[f64],
    dm2: &[f64],
    nz: usize,
) -> (Vec<f64>, Vec<[i32; 2]>) {
    let mut intersects = vec![0.0_f64; 2 * nz + 1];
    let mut ranges = vec![[-1_i32, -1_i32]; 2 * nz + 1];
    let mut rem1 = dm1.to_vec();
    let mut rem2 = dm2.to_vec();
    let mut j = nz;
    let mut k = nz;
    let mut counter = 2 * nz;
    let (mut side, mut diff) = if dm1[j] > dm2[j] {
        ranges[counter] = [j as i32, k as i32];
        intersects[counter] = dm2[k];
        counter -= 1;
        rem1[j] -= dm2[k];
        rem2[j] -= dm2[k];
        (1_u8, dm1[j] - dm2[k])
    } else {
        ranges[counter] = [j as i32, k as i32];
        intersects[counter] = dm1[j];
        counter -= 1;
        rem1[j] -= dm1[j];
        rem2[j] -= dm1[j];
        (2_u8, dm2[k] - dm1[j])
    };
    while k > 0 && j > 0 && ((side == 1 && k > 1) || (side == 2 && j > 1)) {
        if side == 1 {
            k -= 1;
            ranges[counter] = [j as i32, k as i32];
            if dm2[k] < diff {
                intersects[counter] = dm2[k];
                rem1[j] -= dm2[k];
                rem2[k] -= dm2[k];
                diff -= dm2[k];
            } else {
                intersects[counter] = diff;
                rem1[j] -= diff;
                rem2[k] -= diff;
                diff = dm2[k] - diff;
                side = 2;
            }
            counter -= 1;
        } else {
            j -= 1;
            ranges[counter] = [j as i32, k as i32];
            if dm1[j] < diff {
                intersects[counter] = dm1[j];
                rem1[j] -= dm1[j];
                rem2[k] -= dm1[j];
                diff -= dm1[j];
            } else {
                intersects[counter] = diff;
                rem1[j] -= diff;
                rem2[k] -= diff;
                diff = dm1[j] - diff;
                side = 1;
            }
            counter -= 1;
        }
    }
    if j > 1 {
        for i in (1..=j).rev() {
            ranges[counter] = [i as i32, 0];
            intersects[counter] = rem1[i];
            counter -= 1;
        }
    } else {
        for i in (1..=k).rev() {
            ranges[counter] = [0, i as i32];
            intersects[counter] = rem2[i];
            counter -= 1;
        }
    }
    (intersects, ranges)
}

fn compute_pass_fraction(
    i: i32,
    j: usize,
    i_min: &[i32],
    i_max: &[i32],
    pf: &[Vec<f64>],
) -> f64 {
    let ji = j as i32;
    if i >= i_min[j] && i <= i_max[j] {
        pf[j][(i - i_min[j]) as usize]
    } else if i > ji.max(i_max[j]) || i < ji.min(i_min[j]) {
        0.0
    } else {
        1.0
    }
}

#[allow(clippy::too_many_arguments)]
fn leak_segment(
    nz: usize,
    i_start: i32,
    i_end: i32,
    i_min: &[i32],
    i_max: &[i32],
    j_min: &[i32],
    j_max: &[i32],
    pf: &[Vec<f64>],
    dm: &[f64],
    delta_m: f64,
    eps_per_tot: &[f64],
    leak_frac: &[f64],
    accumulated: &mut [f64],
) {
    if i_start == i_end {
        return;
    }
    let dir = if i_start < i_end { 1_i32 } else { -1_i32 };
    let mut excess = vec![0.0_f64; nz + 1];
    let mut i = i_start;
    while i != i_end + dir {
        let iu = i as usize;
        if leak_frac[iu] >= 1.0 {
            if i != i_start && leak_frac[(i - dir) as usize] < 1.0 {
                for j in j_min[iu]..=j_max[iu] {
                    if j >= 0 {
                        accumulated[iu] += excess[j as usize];
                        excess[j as usize] = 0.0;
                    }
                }
            }
        } else {
            for j in j_min[iu]..=j_max[iu] {
                if j < 0 {
                    continue;
                }
                let ju = j as usize;
                if (i == i_end && j > 0) || j == i {
                    accumulated[iu] += excess[ju];
                    excess[ju] = 0.0;
                } else if i == i_end && i == 1 && j == 0 {
                    let pf_10 = compute_pass_fraction(1, 0, i_min, i_max, pf);
                    excess[0] -= delta_m * pf_10 * eps_per_tot[1];
                    accumulated[1] += leak_frac[1] * excess[0];
                    excess[0] = 0.0;
                } else {
                    let pf_ij = compute_pass_fraction(i, ju, i_min, i_max, pf);
                    let pass_mass = if j > 0 {
                        pf_ij * dm[ju]
                    } else {
                        -pf_ij * delta_m
                    };
                    let next =
                        excess[ju] + dm[iu] * eps_per_tot[iu] * pass_mass;
                    accumulated[iu] += leak_frac[iu] * next;
                    excess[ju] = (1.0 - leak_frac[iu]) * next;
                }
            }
        }
        i += dir;
    }
}

#[allow(clippy::too_many_arguments)]
fn leak_control_parcel(
    nz: usize,
    mass_flux_0: &[f64],
    dm_0: &[f64],
    prev_dm_0: &[f64],
    tot_mass_0: &[f64],
    power_0: &[f64],
    leak_0: &[f64],
) -> Vec<f64> {
    let mut dm = vec![0.0_f64; nz + 1];
    let mut prev_dm = vec![0.0_f64; nz + 1];
    let mut tot_mass = vec![0.0_f64; nz + 1];
    let mut eps_per_tot = vec![0.0_f64; nz + 1];
    let mut leak_frac = vec![1.0_f64; nz + 1];
    for k in 0..nz {
        dm[k + 1] = dm_0[k].max(1.0e-99);
        prev_dm[k + 1] = prev_dm_0[k].max(1.0e-99);
        tot_mass[k + 1] = tot_mass_0[k].max(1.0e-99);
        eps_per_tot[k + 1] = power_0[k] / (dm[k + 1] * tot_mass[k + 1]);
        leak_frac[k + 1] = leak_0[k];
    }
    let mut mf = vec![0.0_f64; nz + 2];
    mf[1..=(nz + 1)].copy_from_slice(&mass_flux_0[..=nz]);
    let delta_m = if mf[1] == 0.0 { -1.0 } else { mf[1] };

    let (intersects, ranges) = make_compressed_intersect(&dm, &prev_dm, nz);
    let mut i_min = vec![(nz + 1) as i32; nz + 1];
    let mut i_max = vec![-1_i32; nz + 1];
    for &[j, i] in &ranges[1..=(2 * nz)] {
        if j >= 0 {
            let ju = j as usize;
            i_min[ju] = i_min[ju].min(i);
            i_max[ju] = i_max[ju].max(i);
        }
    }
    let mut pf: Vec<Vec<f64>> = (0..=nz)
        .map(|j| vec![0.0_f64; (i_max[j] - i_min[j] + 1).max(0) as usize])
        .collect();
    let mut counter = 1_usize;
    for j in 0..=nz {
        for i in i_min[j]..=i_max[j] {
            pf[j][(i - i_min[j]) as usize] = intersects[counter];
            counter += 1;
        }
        let ji = j as i32;
        for i in (i_min[j] + 1)..=ji.min(i_max[j]) {
            let idx = (i - i_min[j]) as usize;
            pf[j][idx] += pf[j][idx - 1];
        }
        for i in (ji.max(i_min[j])..i_max[j]).rev() {
            let idx = (i - i_min[j]) as usize;
            pf[j][idx] += pf[j][idx + 1];
        }
        let denom = if j > 0 { dm[j] } else { -delta_m };
        for v in &mut pf[j] {
            *v /= denom;
        }
    }

    let mut j_min = vec![(nz + 1) as i32; nz + 1];
    let mut j_max = vec![-1_i32; nz + 1];
    for i in 1..=nz {
        let ii = i as i32;
        if mf[i] >= 0.0 && mf[i + 1] >= 0.0 {
            j_min[i] = ii;
        }
        if mf[i] <= 0.0 && mf[i + 1] <= 0.0 {
            j_max[i] = ii;
        }
        if mf[i] >= 0.0 && mf[i + 1] <= 0.0 {
            j_min[i] = ii;
            j_max[i] = ii;
        }
    }
    for &[j, i] in &ranges[1..=(2 * nz)] {
        if i > 0 {
            let iu = i as usize;
            if mf[iu] >= 0.0 && mf[iu + 1] >= 0.0 {
                j_max[iu] = j_max[iu].max(j);
            }
            if mf[iu] <= 0.0 && mf[iu + 1] <= 0.0 {
                j_min[iu] = j_min[iu].min(j);
            }
            if mf[iu] <= 0.0 && mf[iu + 1] >= 0.0 {
                j_min[iu] = j_min[iu].min(j);
                j_max[iu] = j_max[iu].max(j);
            }
        }
    }

    let mut accumulated = vec![0.0_f64; nz + 1];
    let (mut i_start, mut i_end) = (1_i32, 1_i32);
    let mut sgn = if mf[1] > 0.0 {
        1
    } else if mf[1] < 0.0 {
        -1
    } else {
        0
    };
    let nzi = nz as i32;
    while i_start <= nzi && i_end <= nzi {
        match sgn.cmp(&0) {
            std::cmp::Ordering::Greater => {
                let next_mf = mf[(i_end + 1) as usize];
                if next_mf > 0.0 {
                    i_end += 1;
                } else {
                    leak_segment(
                        nz,
                        i_start,
                        i_end,
                        &i_min,
                        &i_max,
                        &j_min,
                        &j_max,
                        &pf,
                        &dm,
                        delta_m,
                        &eps_per_tot,
                        &leak_frac,
                        &mut accumulated,
                    );
                    i_start = i_end;
                    sgn = if next_mf < 0.0 { -1 } else { 0 };
                }
            }
            std::cmp::Ordering::Less => {
                let next_mf = mf[(i_start + 1) as usize];
                if next_mf < 0.0 {
                    i_start += 1;
                } else {
                    leak_segment(
                        nz,
                        i_start,
                        i_end,
                        &i_min,
                        &i_max,
                        &j_min,
                        &j_max,
                        &pf,
                        &dm,
                        delta_m,
                        &eps_per_tot,
                        &leak_frac,
                        &mut accumulated,
                    );
                    i_end = i_start;
                    sgn = i32::from(next_mf > 0.0);
                }
            }
            std::cmp::Ordering::Equal => {
                let next_mf = mf[(i_start + 1) as usize];
                if next_mf > 0.0 {
                    sgn = 1;
                } else if next_mf < 0.0 {
                    sgn = -1;
                } else {
                    leak_segment(
                        nz,
                        i_start,
                        i_end,
                        &i_min,
                        &i_max,
                        &j_min,
                        &j_max,
                        &pf,
                        &dm,
                        delta_m,
                        &eps_per_tot,
                        &leak_frac,
                        &mut accumulated,
                    );
                }
                i_start += 1;
                i_end += 1;
            }
        }
    }

    let mut out = vec![0.0_f64; nz];
    for j in 1..=nz {
        let mut val = accumulated[j] / dm[j];
        if leak_frac[j] >= 1.0 {
            val += eps_per_tot[j] * tot_mass[j];
        } else {
            let pf_jj = compute_pass_fraction(j as i32, j, &i_min, &i_max, &pf);
            val += eps_per_tot[j] * dm[j] * pf_jj;
        }
        out[j - 1] = val;
    }
    out
}
