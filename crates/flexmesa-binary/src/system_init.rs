// SPDX-License-Identifier: GPL-3.0-or-later
//! Initial hydrodynamic, rotational-distortion, and energy-accounting setup and
//! evolutionary stage `varcontrol` selector for coupled
//! [`crate::system::BinarySystem`] star models (`run_binary_support.f90` and
//! `binary/private/binary_timestep.f90:290-318`).

use flexmesa_const::{MSUN, SECYER};
use flexmesa_star::{
    StarError, StarModel, apply_uniform_rotation, eval_total_energy_integrals,
    set_hydro_vars, snapshot_step_history_old,
    update_rotation_distortion_factors,
};

use crate::system::{BinaryCompanion, BinarySystem};

/// Synchronizes initial hydrodynamic, rotational, and step-history state of a
/// component [`StarModel`] at `model_number = 0, star_age = 0` before coupled
/// binary evolution begins (`F3-9`: no fabricated pre-evolution settling loops;
/// `&star_job` `relax_initial_Z` / `relax_initial_Y` run during the star's
/// initialization through `do_internal_evolve`).
pub(crate) fn prep_and_settle_binary_star(
    star: &mut StarModel,
) -> Result<(), StarError> {
    if star.controls.rotation_flag {
        let omega_0 = star.omega.first().copied().unwrap_or(0.0);
        if omega_0 > 0.0
            && star.omega.iter().all(|&w| (w - omega_0).abs() == 0.0)
        {
            apply_uniform_rotation(star, omega_0);
        }
    }
    update_rotation_distortion_factors(star);

    star.mstar_start = star.mstar;
    star.mstar_old = star.mstar;
    set_hydro_vars(star, false, false)?;

    let etot = eval_total_energy_integrals(star).5;
    star.aux.total_energy = etot;
    star.aux.cumulative_energy_error = 0.0;
    snapshot_step_history_old(star);
    star.aux.step_hist.total_energy_old = etot;
    star.aux.step_hist.total_energy_end = etot;
    star.aux.step_hist.error_in_energy_conservation = 0.0;
    star.aux.step_hist.cumulative_energy_error_old = 0.0;
    star.save_start_values()?;
    star.save_old_values();
    Ok(())
}

fn center_h1_frac(star: &StarModel) -> f64 {
    if star.nz == 0 || star.species == 0 {
        return 0.7;
    }
    star.iso_names
        .iter()
        .position(|n| n.eq_ignore_ascii_case("h1"))
        .map_or(0.7, |j| star.cell_xa(star.nz - 1)[j])
}

impl BinarySystem {
    pub(crate) fn update_stage_varcontrol(&mut self) {
        let xh_d = center_h1_frac(&self.donor);
        if self.mtransfer_rate.abs() * SECYER / MSUN > 1.0e-20 {
            let vc = if xh_d > 1.0e-12 {
                self.controls.varcontrol_case_a
            } else {
                self.controls.varcontrol_case_b
            };
            if vc > 0.0 {
                self.donor.controls.varcontrol_target = vc;
                if let BinaryCompanion::Star(ref mut s2) = self.companion {
                    s2.controls.varcontrol_target = vc;
                }
            }
        } else {
            let vc_d = if xh_d > 1.0e-12 {
                self.controls.varcontrol_ms
            } else {
                self.controls.varcontrol_post_ms
            };
            if vc_d > 0.0 {
                self.donor.controls.varcontrol_target = vc_d;
            }
            if let BinaryCompanion::Star(ref mut s2) = self.companion {
                let vc_a = if center_h1_frac(s2) > 1.0e-12 {
                    self.controls.varcontrol_ms
                } else {
                    self.controls.varcontrol_post_ms
                };
                if vc_a > 0.0 {
                    s2.controls.varcontrol_target = vc_a;
                }
            }
        }
    }
}
