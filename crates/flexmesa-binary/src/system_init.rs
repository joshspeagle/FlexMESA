// SPDX-License-Identifier: GPL-3.0-or-later
//! Initial hydrodynamic, rotational-distortion, and energy-accounting setup and
//! evolutionary stage `varcontrol` selector for coupled
//! [`crate::system::BinarySystem`] star models (`run_binary_support.f90` and
//! `binary/private/binary_timestep.f90:290-318`).

use flexmesa_const::{MSUN, SECYER};
use flexmesa_star::{
    center_avg_x_by_name, eval_total_energy_integrals, set_hydro_vars,
    snapshot_step_history_old, StarError, StarModel,
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
    // No rotation update here: upstream's only one at this point is
    // `star_relax_uniform_omega(.., 2*pi/period, ..)` under
    // `do_initial_orbit_sync_N .and. .not. doing_restart`
    // (`binary/private/run_binary_support.f90:319-339`), which FlexMESA
    // performs in `apply_initial_orbit_sync` (`system_jdot.rs`).

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

impl BinarySystem {
    /// Sets the stars' `varcontrol_target` for the current evolutionary phase,
    /// the last block of `binary_pick_next_timestep`
    /// (`binary/private/binary_timestep.f90:287-317`). `center_h1` is
    /// `s% center_h1`, the central average set by `do_report`
    /// (`star/private/report.f90:259`, `center_avg_x`).
    pub(crate) fn update_stage_varcontrol(&mut self) -> Result<(), StarError> {
        let c = &self.controls;
        let two_stars = self.point_mass_i == 0;
        let donor_h1 = center_avg_x_by_name(&self.donor, "h1")?;
        if self.mtransfer_rate.abs() / MSUN * SECYER > 1.0e-20 {
            let vc = if donor_h1 > 1.0e-12 && c.varcontrol_case_a > 0.0 {
                Some(c.varcontrol_case_a)
            } else if donor_h1 < 1.0e-12 && c.varcontrol_case_b > 0.0 {
                Some(c.varcontrol_case_b)
            } else {
                None
            };
            if let Some(vc) = vc {
                self.donor.controls.varcontrol_target = vc;
                if two_stars {
                    if let BinaryCompanion::Star(ref mut s2) = self.companion {
                        s2.controls.varcontrol_target = vc;
                    }
                }
            }
        } else {
            let (vc_ms, vc_post_ms) = (c.varcontrol_ms, c.varcontrol_post_ms);
            if donor_h1 > 1.0e-12 {
                if vc_ms > 0.0 {
                    self.donor.controls.varcontrol_target = vc_ms;
                }
            } else if vc_post_ms > 0.0 {
                self.donor.controls.varcontrol_target = vc_post_ms;
            }
            if two_stars {
                if let BinaryCompanion::Star(ref mut s2) = self.companion {
                    if center_avg_x_by_name(s2, "h1")? > 1.0e-12 {
                        if vc_ms > 0.0 {
                            s2.controls.varcontrol_target = vc_ms;
                        }
                    } else if vc_post_ms > 0.0 {
                        s2.controls.varcontrol_target = vc_post_ms;
                    }
                }
            }
        }
        Ok(())
    }
}
