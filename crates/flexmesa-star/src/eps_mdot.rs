// SPDX-License-Identifier: LGPL-3.0-or-later
//! Heating from a change of stellar mass (`star/private/eps_mdot.f90`).
//!
//! When mass is added or removed the cells are redefined, and material
//! crossing the new cell boundaries carries energy and does work. Upstream
//! evaluates the balance in `real(qp)` because the changes in cell mass are
//! tiny compared with the cell masses deep in the star. This module follows
//! upstream statement by statement with [`Quad`] and [`AccurateReal`]
//! (`DECISIONS #46`), using the overlap routines in [`crate::mass_utils`].

use flexmesa_math::Quad;
use flexmesa_num::AccurateReal;

use crate::mass_utils::{
    compute_pass_fraction, find_j_ranges, make_compressed_intersect,
    prepare_pass_fraction, zero_slot, CompressedIntersect, PassFraction,
};
use crate::model::StarModel;

/// Evaluates the single-zone specific total energy $e_{\mathrm{tot},k}$
/// ($\mathrm{erg\,g^{-1}}$) at zone `k` (`cell_specific_total_energy` in
/// `star/private/star_utils.f90:2397-2410`).
#[must_use]
#[inline]
pub fn eval_cell_specific_total_energy(model: &StarModel, k: usize) -> f64 {
    crate::cell_energy::cell_specific_total_energy(model, k)
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

/// What `save_for_eps_mdot` and `revise_q_and_dq` leave on the star for
/// `calculate_eps_mdot` (`star/private/adjust_mass.f90:75-95, 808-829`).
#[derive(Clone, Debug)]
pub struct AdjustMassRecord {
    /// `dm_before_adjust_mass(1:nz)` in grams (zero-based here).
    pub dm_before: Vec<f64>,
    /// `total_energy_profile_before_adjust_mass(1:nz)` in erg (zero-based).
    pub total_energy_profile_before: Vec<f64>,
    /// `adjust_mass_outer_frac_sub1`, for cells above `k_below_const_q`.
    pub outer_frac_sub1: f64,
    /// `adjust_mass_mid_frac_sub1`, for `k_below_const_q <= k < k_const_mass`.
    pub mid_frac_sub1: f64,
    /// `adjust_mass_inner_frac_sub1`, for `k >= k_const_mass`.
    pub inner_frac_sub1: f64,
    /// Upstream's one-based `k_below_const_q` (`kA`).
    pub k_below_const_q: usize,
    /// Upstream's one-based `k_const_mass` (`kB`).
    pub k_const_mass: usize,
}

/// The per-cell total energy in erg that both upstream profile routines
/// build, `dm` times the specific energies added in upstream's order
/// (`eval_deltaM_total_energy_integrals` and `eval_total_energy_profile` in
/// `star/private/star_utils.f90:2439-2560`).
///
/// `with_tdc_eturb` selects the extra turbulent term that
/// `eval_deltaM_total_energy_integrals` adds (`star_utils.f90:2494-2505`) and
/// `eval_total_energy_profile` does not (`star_utils.f90:2525-2560`).
fn total_energy_profile_impl(
    model: &StarModel,
    with_tdc_eturb: bool,
) -> Vec<f64> {
    if with_tdc_eturb {
        return crate::evolve_support::eval_delta_m_total_energy_integrals(
            model,
            0,
            model.nz,
            model.mstar,
            true,
        )
        .0;
    }
    let nz = model.nz;
    let mut profile = vec![0.0_f64; nz];
    for k in 0..nz {
        let dm = model.dm[k];
        let mut cell_total = 0.0_f64;
        let mut cell1 = dm * model.energy[k];
        cell_total += cell1;
        if model.controls.v_flag || model.controls.u_flag {
            cell1 = dm * crate::cell_energy::cell_specific_ke(model, k);
            cell_total += cell1;
        }
        cell1 = dm * crate::cell_energy::cell_specific_pe(model, k);
        cell_total += cell1;
        if model.controls.rotation_flag
            && model.controls.include_rotation_in_total_energy
        {
            cell1 = dm
                * crate::cell_energy::cell_specific_rotational_energy(model, k);
            cell_total += cell1;
        }
        if model.controls.rsp2_flag {
            cell1 = dm * crate::cell_energy::cell_rsp2_eturb(model, k);
            cell_total += cell1;
        }
        if model.controls.rsp_flag {
            cell1 = dm * model.aux.rsp.et_rsp.get(k).copied().unwrap_or(0.0);
            cell_total += cell1;
        }
        profile[k] = cell_total;
    }
    profile
}

/// `total_energy_profile_before_adjust_mass` as `save_for_eps_mdot` fills it
/// (`adjust_mass.f90:75-95`, through `eval_deltaM_total_energy_integrals`,
/// `star_utils.f90:2439-2522`, called with `klo = 1`, `khi = nz`).
///
/// Upstream's loop tests `sum_dm` against `deltaM` but never adds to
/// `sum_dm`, so no cell is clipped unless a single cell exceeds `mstar`,
/// which cannot happen; every cell contributes its full `dm`.
///
/// # Errors
///
/// Always returns `Ok`; retains `Result` return type for API compatibility.
pub fn total_energy_profile_before_adjust_mass(
    model: &StarModel,
) -> Result<Vec<f64>, crate::controls::StarError> {
    Ok(total_energy_profile_impl(model, true))
}

/// `eval_total_energy_profile` (`star_utils.f90:2525-2560`), in erg per cell.
///
/// # Errors
///
/// Always returns `Ok`; retains `Result` return type for API compatibility.
pub fn eval_total_energy_profile(
    model: &StarModel,
) -> Result<Vec<f64>, crate::controls::StarError> {
    Ok(total_energy_profile_impl(model, false))
}

#[inline]
fn q(x: f64) -> Quad {
    Quad::from(x)
}

/// `interpolate_onto_faces` (`eps_mdot.f90:48-62`). `vec` is zero-based
/// (`vec[j - 1]` is upstream's `vec(j)`), `dm` is one-based `real(qp)`, and
/// `j` is upstream's face index in `1..=nz + 1`.
fn interpolate_onto_faces(vec: &[f64], dm: &[Quad], nz: usize, j: usize) -> f64 {
    let mut interp = 0.0;
    if j == 1 {
        interp = vec[0];
    }
    if j == nz + 1 {
        interp = vec[nz - 1];
    }
    if j > 1 && j < nz + 1 {
        // `dmbar1`, `beta` and `alpha` are `real(dp)`.
        let dmbar1 = ((dm[j - 1] + dm[j]) / 2.0).to_f64();
        let beta = (dm[j - 1] / (2.0 * dmbar1)).to_f64();
        let alpha = 1.0 - beta;
        interp = alpha * vec[j - 2] + beta * vec[j - 1];
    }
    interp
}

/// `set_leak_frac` (`eps_mdot.f90:142-187`). Returns `leak_frac(1:nz)`,
/// one-based with element `0` unused.
fn set_leak_frac(
    model: &StarModel,
    dt: f64,
    thermal_energy: &[f64],
    grad_r_sub_grad_a: &[f64],
    mass_flux: &[Quad],
    factor: f64,
) -> Vec<f64> {
    let nz = model.nz;
    let mut leak_frac = vec![0.0_f64; nz + 1];
    for k in 1..=nz {
        let mass_flux_bar = if k == 1 {
            mass_flux[1].to_f64()
        } else {
            ((mass_flux[k - 1] + mass_flux[k]) / 2.0).to_f64()
        };
        if grad_r_sub_grad_a[k].abs() == 0.0 || mass_flux_bar == 0.0 {
            leak_frac[k] = 1.0;
        } else {
            leak_frac[k] = factor
                * (model.lum[k - 1] * dt
                    / (grad_r_sub_grad_a[k]
                        * thermal_energy[k]
                        * mass_flux_bar))
                    .abs();
            leak_frac[k] = leak_frac[k].min(1.0);
        }
    }
    leak_frac
}

/// Arrays shared by `leak_control` and `leak`, all with upstream's indices.
struct LeakState<'a> {
    nz: usize,
    pf: &'a PassFraction,
    j_min: &'a [i32],
    j_max: &'a [i32],
    dm: &'a [Quad],
    delta_m: Quad,
    eps_mdot_per_total_mass: &'a [f64],
    leak_frac: &'a [f64],
    accumulated: Vec<f64>,
    mdot_adiabatic_surface: f64,
}

impl LeakState<'_> {
    /// `leak` (`eps_mdot.f90:399-470`).
    fn leak(&mut self, i_start: usize, i_end: usize) {
        if i_start == i_end {
            return;
        }
        let forward = i_start < i_end;
        let step = |i: usize| if forward { i + 1 } else { i - 1 };
        let prev = |i: usize| if forward { i - 1 } else { i + 1 };
        let stop = step(i_end);
        // `excess(0:nz)`
        let mut excess = vec![0.0_f64; self.nz + 1];
        let mut i = i_start;
        while i != stop {
            if self.leak_frac[i] == 1.0 {
                if i != i_start && self.leak_frac[prev(i)] < 1.0 {
                    for j in self.j_min[i]..=self.j_max[i] {
                        let ju = usize::try_from(j).expect("j >= 0");
                        self.accumulated[i] += excess[ju];
                        excess[ju] = 0.0;
                    }
                }
            } else {
                for j in self.j_min[i]..=self.j_max[i] {
                    let ju = usize::try_from(j).expect("j >= 0");
                    if (i == i_end && ju > 0) || ju == i {
                        self.accumulated[i] += excess[ju];
                        excess[ju] = 0.0;
                    } else if i == i_end && i == 1 && ju == 0 {
                        let pass_frac = compute_pass_fraction(1, 0, self.pf);
                        excess[ju] = (q(excess[ju])
                            - self.delta_m
                                * pass_frac
                                * self.eps_mdot_per_total_mass[1])
                        .to_f64();
                        self.mdot_adiabatic_surface +=
                            (1.0 - self.leak_frac[i_end]) * excess[ju];
                        self.accumulated[i] +=
                            self.leak_frac[i_end] * excess[ju];
                        excess[ju] = 0.0;
                    } else {
                        let ii = i32::try_from(i).expect("nz fits in i32");
                        let pass_frac = compute_pass_fraction(ii, ju, self.pf);
                        let pass_mass = if ju > 0 {
                            pass_frac * self.dm[ju]
                        } else {
                            -pass_frac * self.delta_m
                        };
                        let next = q(excess[ju])
                            + self.dm[i]
                                * self.eps_mdot_per_total_mass[i]
                                * pass_mass;
                        self.accumulated[i] = (q(self.accumulated[i])
                            + q(self.leak_frac[i]) * next)
                            .to_f64();
                        excess[ju] =
                            (q(1.0 - self.leak_frac[i]) * next).to_f64();
                    }
                }
            }
            i = step(i);
        }
    }
}

/// `leak_control` (`eps_mdot.f90:189-397`). Returns `accumulated(1:nz)`
/// (one-based) and `mdot_adiabatic_surface`.
#[allow(clippy::too_many_arguments)]
fn leak_control(
    nz: usize,
    mass_flux: &[Quad],
    dm: &[Quad],
    mesh_intersects: &[Quad],
    ranges: &[[i32; 2]],
    total_mass_through_cell: &[Quad],
    eps_mdot_per_total_mass: &[f64],
    leak_frac: &[f64],
) -> (Vec<f64>, f64) {
    let zero = Quad::ZERO;
    let mut delta_m = mass_flux[1];
    if mass_flux[1] == zero {
        delta_m = q(-1.0);
    }
    let pf = prepare_pass_fraction(nz, delta_m, dm, mesh_intersects, ranges);
    let (j_min, j_max) = find_j_ranges(nz, ranges, mass_flux);

    let mut st = LeakState {
        nz,
        pf: &pf,
        j_min: &j_min,
        j_max: &j_max,
        dm,
        delta_m,
        eps_mdot_per_total_mass,
        leak_frac,
        accumulated: vec![0.0_f64; nz + 1],
        mdot_adiabatic_surface: 0.0,
    };

    let (mut i_start, mut i_end) = (1_usize, 1_usize);
    // `sgn = mass_flux(1)/abs(mass_flux(1))`, or 0.
    let mut sgn: i32 = if mass_flux[1] == zero {
        0
    } else if mass_flux[1] > zero {
        1
    } else {
        -1
    };
    while i_start <= nz && i_end <= nz {
        if sgn > 0 {
            if mass_flux[i_end + 1] > zero {
                i_end += 1;
            } else if mass_flux[i_end + 1] < zero {
                st.leak(i_start, i_end);
                i_start = i_end;
                sgn = -1;
            } else {
                st.leak(i_start, i_end);
                i_start = i_end;
                sgn = 0;
            }
        } else if sgn < 0 {
            if mass_flux[i_start + 1] < zero {
                i_start += 1;
            } else if mass_flux[i_start + 1] > zero {
                st.leak(i_start, i_end);
                i_end = i_start;
                sgn = 1;
            } else {
                st.leak(i_start, i_end);
                i_end = i_start;
                sgn = 0;
            }
        } else if mass_flux[i_start + 1] > zero {
            i_start += 1;
            i_end += 1;
            sgn = 1;
        } else if mass_flux[i_start + 1] < zero {
            i_start += 1;
            i_end += 1;
            sgn = -1;
        } else {
            st.leak(i_start, i_end);
            i_start += 1;
            i_end += 1;
        }
    }

    let mdot_adiabatic_surface = st.mdot_adiabatic_surface;
    let mut accumulated = st.accumulated;
    for j in 1..=nz {
        accumulated[j] = (q(accumulated[j]) / dm[j]).to_f64();
    }
    for j in 1..=nz {
        if leak_frac[j] == 1.0 {
            accumulated[j] = (q(accumulated[j])
                + total_mass_through_cell[j] * eps_mdot_per_total_mass[j])
                .to_f64();
        } else {
            let jj = i32::try_from(j).expect("nz fits in i32");
            accumulated[j] = (q(accumulated[j])
                + q(eps_mdot_per_total_mass[j])
                    * dm[j]
                    * compute_pass_fraction(jj, j, &pf))
            .to_f64();
        }
    }
    (accumulated, mdot_adiabatic_surface)
}

/// Surface work and energy rates that `calculate_eps_mdot` also sets.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EpsMdotSurface {
    /// `mdot_acoustic_surface` in erg/s (`eps_mdot.f90:613`).
    pub mdot_acoustic_surface: f64,
    /// `mdot_adiabatic_surface` in erg/s (`eps_mdot.f90:654`).
    pub mdot_adiabatic_surface: f64,
}

/// `calculate_eps_mdot` (`star/private/eps_mdot.f90:472-704`): the heating
/// rate per unit mass in each cell from the change of mass, stored in
/// `model.eps_mdot` in $\mathrm{erg\,g^{-1}\,s^{-1}}$.
///
/// Cell masses, their changes, the mass flux and the mesh overlaps are
/// `real(qp)` ([`Quad`]); the per-cell energy balance is an `accurate_real`
/// sum ([`AccurateReal`]). Each conversion between `real(dp)` and `real(qp)`
/// is where upstream has it.
///
/// `model` must hold the state after `do_adjust_mass` with its variables
/// set (`evolve.f90:591-596`), and `record` what `do_adjust_mass` saved.
///
/// # Errors
///
/// Returns `InvalidInput` for RSP or RSP2 models, and when `record` does not
/// match the model's zone count.
pub fn calculate_eps_mdot(
    model: &mut StarModel,
    record: &AdjustMassRecord,
    dt: f64,
) -> Result<EpsMdotSurface, crate::controls::StarError> {
    let nz = model.nz;
    model.eps_mdot.resize(nz, 0.0);
    if model.mstar_dot == 0.0 || dt <= 0.0 {
        model.eps_mdot.fill(0.0);
        return Ok(EpsMdotSurface::default());
    }
    if record.dm_before.len() != nz
        || record.total_energy_profile_before.len() != nz
        || nz < 2
    {
        return Err(crate::controls::StarError::InvalidInput(format!(
            "calculate_eps_mdot: record sized for {} zones, model has {nz}",
            record.dm_before.len()
        )));
    }
    let delta_m = model.mstar_dot * dt;

    // `compute_prev_mesh_dm` (`adjust_mass.f90:103-132`).
    let mut dm = zero_slot(nz);
    let mut prev_mesh_dm = zero_slot(nz);
    let mut change_in_dm = zero_slot(nz);
    for j in 1..=nz {
        dm[j] = q(model.dm[j - 1]);
        let before = q(record.dm_before[j - 1]);
        let frac_sub1 = if j < record.k_below_const_q {
            record.outer_frac_sub1
        } else if j < record.k_const_mass {
            record.mid_frac_sub1
        } else {
            record.inner_frac_sub1
        };
        change_in_dm[j] = before * frac_sub1;
        prev_mesh_dm[j] = dm[j] - change_in_dm[j];
    }

    // `find_mass_flux` (`eps_mdot.f90:36-46`).
    let mut mass_flux = zero_slot(nz + 1);
    for j in (1..=nz).rev() {
        mass_flux[j] = mass_flux[j + 1] + change_in_dm[j];
    }

    let CompressedIntersect {
        mesh_intersects,
        ranges,
    } = make_compressed_intersect(&dm, &prev_mesh_dm, nz);

    // `density_weighted_flux(1:nz+1)` is `real(dp)`.
    let mut density_weighted_flux = vec![0.0_f64; nz + 2];
    for j in (1..=nz).rev() {
        density_weighted_flux[j] = (q(density_weighted_flux[j + 1])
            + change_in_dm[j] / model.rho[j - 1])
            .to_f64();
    }

    let zero = Quad::ZERO;
    let mut total_mass_through_cell = zero_slot(nz);
    for j in 1..=nz {
        let inflow = if mass_flux[j] > zero { mass_flux[j] } else { zero };
        let outflow = if mass_flux[j + 1] < zero {
            mass_flux[j + 1]
        } else {
            zero
        };
        total_mass_through_cell[j] = prev_mesh_dm[j] + inflow - outflow;
    }

    // `te(1:nz)` is `real(dp)`; zero-based here for `interpolate_onto_faces`.
    let te: Vec<f64> = (1..=nz)
        .map(|j| {
            (q(record.total_energy_profile_before[j - 1]) / prev_mesh_dm[j])
                .to_f64()
        })
        .collect();
    let after = eval_total_energy_profile(model)?;

    // Face values `(1:nz+1)`, one-based.
    let mut p_bar = vec![0.0_f64; nz + 2];
    let mut rho_bar = vec![0.0_f64; nz + 2];
    let mut te_bar = vec![0.0_f64; nz + 2];
    for j in 1..=nz + 1 {
        p_bar[j] = interpolate_onto_faces(&model.peos, &prev_mesh_dm, nz, j);
        rho_bar[j] = interpolate_onto_faces(&model.rho, &prev_mesh_dm, nz, j);
        te_bar[j] = interpolate_onto_faces(&te, &prev_mesh_dm, nz, j);
    }

    let mut grad_r_sub_grad_a = vec![0.0_f64; nz + 1];
    for j in 1..nz {
        let grad_t_mid = 0.5 * (model.gradt[j - 1] + model.gradt[j]);
        grad_r_sub_grad_a[j] = model.grada[j - 1] - grad_t_mid;
    }
    grad_r_sub_grad_a[nz] = grad_r_sub_grad_a[nz - 1];

    let mut thermal_energy = vec![0.0_f64; nz + 1];
    for j in 1..=nz {
        thermal_energy[j] = model.t[j - 1] * model.cp[j - 1];
    }

    let mdot_acoustic_surface = delta_m * p_bar[1] / rho_bar[1] / dt;

    let mut eps_mdot_per_total_mass = vec![0.0_f64; nz + 1];
    for j in 1..=nz {
        let s_dm = model.dm[j - 1];
        let mut sum = AccurateReal::ZERO;
        sum = sum + change_in_dm[j] * te_bar[j];
        sum = sum + mass_flux[j + 1] * (te_bar[j] - te_bar[j + 1]);
        sum = sum - change_in_dm[j] * te[j - 1];
        sum = sum
            + q(p_bar[j])
                * (mass_flux[j] / rho_bar[j] - density_weighted_flux[j]);
        sum = sum
            - q(p_bar[j + 1])
                * (mass_flux[j + 1] / rho_bar[j + 1]
                    - density_weighted_flux[j + 1]);
        sum = sum - q(after[j - 1] - s_dm * te[j - 1]);
        eps_mdot_per_total_mass[j] =
            (sum.value() / (s_dm * dt) / total_mass_through_cell[j]).to_f64();
    }

    let leak_frac = set_leak_frac(
        model,
        dt,
        &thermal_energy,
        &grad_r_sub_grad_a,
        &mass_flux,
        model.controls.mix_wind.eps_mdot_leak_frac_factor,
    );
    let (accumulated, mdot_adiabatic_surface) = leak_control(
        nz,
        &mass_flux,
        &dm,
        &mesh_intersects,
        &ranges,
        &total_mass_through_cell,
        &eps_mdot_per_total_mass,
        &leak_frac,
    );

    let factor = model.controls.mix_wind.eps_mdot_factor;
    for j in 1..=nz {
        model.eps_mdot[j - 1] = factor * accumulated[j];
    }
    Ok(EpsMdotSurface {
        mdot_acoustic_surface,
        mdot_adiabatic_surface: -mdot_adiabatic_surface,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mass_utils::one_based;

    /// Builds the inputs of `leak_control` for a small mesh and returns
    /// `(sum_j accumulated(j)*dm(j) + mdot_adiabatic_surface,
    ///   sum_j eps_mdot_per_total_mass(j)*dm(j)*total_mass_through_cell(j))`.
    ///
    /// Upstream's own debug check compares these two (`eps_mdot.f90:667-674`):
    /// redistributing the heating between cells must not change its total.
    fn leak_totals(
        dm_new: &[f64],
        dm_old: &[f64],
        eps: &[f64],
        leak: &[f64],
    ) -> (f64, f64) {
        let nz = dm_new.len();
        let dm = one_based(dm_new);
        let prev = one_based(dm_old);
        let mut change = zero_slot(nz);
        let mut mass_flux = zero_slot(nz + 1);
        for j in (1..=nz).rev() {
            change[j] = dm[j] - prev[j];
            mass_flux[j] = mass_flux[j + 1] + change[j];
        }
        let zero = Quad::ZERO;
        let mut tmtc = zero_slot(nz);
        for j in 1..=nz {
            let inflow = if mass_flux[j] > zero { mass_flux[j] } else { zero };
            let outflow = if mass_flux[j + 1] < zero {
                mass_flux[j + 1]
            } else {
                zero
            };
            tmtc[j] = prev[j] + inflow - outflow;
        }
        let ci = make_compressed_intersect(&dm, &prev, nz);
        let mut eps1 = vec![0.0];
        eps1.extend_from_slice(eps);
        let mut leak1 = vec![0.0];
        leak1.extend_from_slice(leak);
        let (acc, mdot_ad) = leak_control(
            nz,
            &mass_flux,
            &dm,
            &ci.mesh_intersects,
            &ci.ranges,
            &tmtc,
            &eps1,
            &leak1,
        );
        let redistributed: f64 = (1..=nz)
            .map(|j| acc[j] * dm_new[j - 1])
            .sum::<f64>()
            + mdot_ad;
        let original: f64 = (1..=nz)
            .map(|j| eps1[j] * dm_new[j - 1] * tmtc[j].to_f64())
            .sum();
        (redistributed, original)
    }

    #[test]
    fn test_leak_control_with_full_leak_keeps_each_cells_heating() {
        // `leak_frac == 1` everywhere: nothing is carried along, so each
        // cell keeps `eps * total_mass_through_cell`.
        let (got, want) = leak_totals(
            &[1.0, 1.5, 1.5, 1.0],
            &[2.0, 1.75, 1.5, 1.0],
            &[3.0, -2.0, 0.5, 0.25],
            &[1.0, 1.0, 1.0, 1.0],
        );
        assert!((got - want).abs() <= 1.0e-14 * want.abs(), "{got} {want}");
    }

    #[test]
    fn test_leak_control_conserves_total_heating_for_mass_loss() {
        let (got, want) = leak_totals(
            &[1.0, 1.5, 1.5, 1.0],
            &[2.0, 1.75, 1.5, 1.0],
            &[3.0, -2.0, 0.5, 0.25],
            &[0.25, 0.5, 1.0, 1.0],
        );
        assert!((got - want).abs() <= 1.0e-13 * want.abs(), "{got} {want}");
    }

    #[test]
    fn test_leak_control_conserves_total_heating_for_accretion() {
        let (got, want) = leak_totals(
            &[1.0, 1.0, 1.5, 1.0],
            &[0.25, 0.5, 1.5, 1.0],
            &[3.0, -2.0, 0.5, 0.25],
            &[0.25, 0.5, 0.75, 1.0],
        );
        assert!((got - want).abs() <= 1.0e-13 * want.abs(), "{got} {want}");
    }

    #[test]
    fn test_interpolate_onto_faces_matches_upstream_weights() {
        // Interior face j: beta = dm(j-1)/(dm(j-1)+dm(j)), alpha = 1 - beta,
        // value = alpha*vec(j-1) + beta*vec(j) (`eps_mdot.f90:56-60`).
        let dm = one_based(&[1.0, 3.0, 4.0]);
        let vec = [10.0, 20.0, 40.0];
        assert_eq!(interpolate_onto_faces(&vec, &dm, 3, 1), 10.0);
        assert_eq!(interpolate_onto_faces(&vec, &dm, 3, 4), 40.0);
        assert_eq!(
            interpolate_onto_faces(&vec, &dm, 3, 2),
            0.75 * 10.0 + 0.25 * 20.0
        );
        assert_eq!(
            interpolate_onto_faces(&vec, &dm, 3, 3),
            (1.0 - 3.0 / 7.0) * 20.0 + (3.0 / 7.0) * 40.0
        );
    }
}
