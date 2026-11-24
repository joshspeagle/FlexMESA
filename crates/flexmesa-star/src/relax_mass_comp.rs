// SPDX-License-Identifier: LGPL-3.0-or-later
//! The mass and composition relaxes of `star/private/relax.f90` at MESA
//! `26.4.1`, run through [`do_internal_evolve`]: `do_relax_mass`
//! (`relax.f90:126-313`), `do_relax_mass_scale` (`relax.f90:2195-2336`),
//! `do_relax_composition` (`relax.f90:316-548`), `do_relax_Z`
//! (`relax.f90:3536-3701`) and `do_relax_Y` (`relax.f90:3704-3873`).
//!
//! Each relax keeps upstream's `rpar`/`ipar` values as fields of its hook
//! struct. The composition arithmetic (`set_y`, `set_z`, `eval_current_*`,
//! the `interp_pm` target profile) lives in [`crate::relax_comp`]; this module
//! holds only the evolve loops.

use flexmesa_const::{FOUR_THIRDS_PI, M_SUN, ONE_THIRD, SECYER};

use crate::controls::StarError;
use crate::core_boundaries::k_for_q_1based;
use crate::inlist_runner::StarExtras;
use crate::model::StarModel;
use crate::relax_comp::{
    eval_current_y, eval_current_z, set_y, set_z, CompositionProfile,
    RelaxCompositionInterpolants,
};
use crate::relax_driver::{
    do_internal_evolve, setup_before_relax, turn_off_winds, RelaxHookResult,
    RelaxHooks, RELAX_FINISHED_OKAY,
};

/// `min_z` (`star/private/relax.f90:58`).
const MIN_Z: f64 = 1.0e-12;
/// `min_dlnz` (`star/private/relax.f90:57`).
const MIN_DLNZ: f64 = -12.0;

fn finished() -> RelaxHookResult {
    RelaxHookResult::Terminate(String::from(RELAX_FINISHED_OKAY))
}

/// `s% star_mass` as `do_report` sets it before `check_model`
/// (`star/private/report.f90:231`).
fn star_mass(model: &StarModel) -> f64 {
    model.mstar / M_SUN
}

/// Sets `star_mass`, `mstar` and `xmstar` for `new_mass` in $M_\odot$
/// (`star/private/relax.f90:147-149`).
fn set_star_mass(model: &mut StarModel, new_mass: f64) {
    model.mstar = new_mass * M_SUN;
    model.xmstar = model.mstar - model.m_center;
}

/// `before_evolve_relax_mass` and `relax_mass_check_model`
/// (`star/private/relax.f90:207-313`). Masses are in grams.
struct RelaxMass {
    /// `rpar(1) = new_mass*Msun`.
    new_mass: f64,
    /// `rpar(2) = lg_max_abs_mdot`.
    lg_max_abs_mdot: f64,
}

impl RelaxHooks for RelaxMass {
    fn before_evolve(
        &mut self,
        model: &mut StarModel,
    ) -> Result<(), StarError> {
        setup_before_relax(model);
        model.controls.mass_change = 0.0;
        model.dt_next = model.dt_next.min(1.0e4 * SECYER);
        Ok(())
    }

    fn check_model(
        &mut self,
        model: &mut StarModel,
    ) -> Result<RelaxHookResult, StarError> {
        // `do_bare_bones_check_model` only gathers history values.
        let new_mass = self.new_mass;
        let mut lg_max_abs_mdot = self.lg_max_abs_mdot;
        if lg_max_abs_mdot <= -100.0 {
            let sm = star_mass(model);
            lg_max_abs_mdot = if sm < 0.003 {
                -7.0
            } else if sm < 0.006 {
                -6.3
            } else if sm < 0.01 {
                -6.0
            } else if sm < 0.1 {
                -4.0
            } else {
                -3.0
            };
        }
        let max_abs_mdot =
            flexmesa_math::exp10(lg_max_abs_mdot) * (M_SUN / SECYER);

        // `max_model_number = -1111` while relaxing (`setup_before_relax`),
        // so the `model_number >= max_model_number` test never fires.
        if let Some(max) = model.controls.max_model_number {
            if model.model_number >= max && max > 0 {
                return Ok(RelaxHookResult::Terminate(String::from(
                    "max_model_number",
                )));
            }
        }

        let abs_diff = (model.mstar - new_mass).abs();
        let mut mdot = (new_mass - model.mstar) / model.dt_next;
        if mdot.abs() > max_abs_mdot {
            mdot = max_abs_mdot.copysign(mdot);
        }

        if abs_diff < 1.0e-4 * new_mass || mdot.abs() < 1.0e-50 {
            model.controls.mass_change = 0.0;
            // Upstream sets `s% mstar = new_mass*Msun` with `new_mass =
            // rpar(1)` already in grams (`star/private/relax.f90:280-284`),
            // so `mstar` holds $M\,M_\odot$ until `do_relax_mass` resets it
            // after the loop (`relax.f90:195-197`). Reproduced.
            set_star_mass(model, new_mass);
            return Ok(finished());
        }

        model.controls.ts_stop.max_timestep =
            (new_mass - model.mstar).abs() / mdot;

        let ramp = 12_usize;
        if model.model_number < ramp {
            mdot *= flexmesa_math::pow(
                1.1,
                model.model_number as f64 - ramp as f64,
            );
        }

        if mdot.abs() * model.dt > 0.05 * model.mstar {
            mdot = (0.05 * model.mstar / model.dt).copysign(mdot);
        }

        model.controls.mass_change = mdot / (M_SUN / SECYER);
        Ok(RelaxHookResult::KeepGoing)
    }
}

/// `do_relax_mass` (`star/private/relax.f90:126-204`): changes the total mass
/// to `new_mass` ($M_\odot$) through `mass_change`, at most
/// $10^{\texttt{lg\_max\_abs\_mdot}}\,M_\odot\,\mathrm{yr}^{-1}$, and resets
/// `initial_mass`.
///
/// # Errors
/// Returns [`StarError::InvalidInput`] for `new_mass <= 0` and the relax error
/// when the evolve loop fails.
pub(crate) fn do_relax_mass(
    model: &mut StarModel,
    new_mass: f64,
    lg_max_abs_mdot: f64,
    extras: &mut dyn StarExtras,
) -> Result<(), StarError> {
    if (new_mass - star_mass(model)).abs() < 1.0e-12 * new_mass {
        set_star_mass(model, new_mass);
        return Ok(());
    }
    if new_mass <= 0.0 {
        return Err(StarError::InvalidInput(format!(
            "invalid new mass {new_mass} (star/private/relax.f90:157-161)"
        )));
    }
    let c = &mut model.controls;
    let cool_wind_agb_scheme = std::mem::take(&mut c.cool_wind_agb_scheme);
    let cool_wind_rgb_scheme = std::mem::take(&mut c.cool_wind_rgb_scheme);
    let varcontrol_target = c.varcontrol_target;
    c.varcontrol_target = 1.0e-3;
    let eps_mdot_factor = c.mix_wind.eps_mdot_factor;
    c.mix_wind.eps_mdot_factor = 0.0;
    let max_model_number = c.max_model_number;
    c.max_model_number = None;
    let starting_dt_next = model.dt_next;

    let mut hooks = RelaxMass {
        new_mass: new_mass * M_SUN,
        lg_max_abs_mdot,
    };
    let res = do_internal_evolve(model, &mut hooks, true, extras);

    let c = &mut model.controls;
    c.max_model_number = max_model_number;
    model.dt_next = model.dt_next.min(starting_dt_next) * 1.0e-1;
    model.controls.initial_mass = new_mass;
    let c = &mut model.controls;
    c.cool_wind_agb_scheme = cool_wind_agb_scheme;
    c.cool_wind_rgb_scheme = cool_wind_rgb_scheme;
    c.varcontrol_target = varcontrol_target;
    c.mix_wind.eps_mdot_factor = eps_mdot_factor;
    set_star_mass(model, new_mass);
    res
}

/// `before_evolve_relax_mass_scale` and `relax_mass_scale_check_model`
/// (`star/private/relax.f90:2261-2336`).
struct RelaxMassScale {
    /// `rpar(1)`, $M_\odot$.
    new_mass: f64,
    /// `rpar(2)`.
    dlgm_per_step: f64,
    /// `rpar(3)`, years.
    change_mass_years_for_dt: f64,
}

impl RelaxHooks for RelaxMassScale {
    fn before_evolve(
        &mut self,
        model: &mut StarModel,
    ) -> Result<(), StarError> {
        setup_before_relax(model);
        model.controls.max_model_number = None;
        model.dt_next = SECYER * self.change_mass_years_for_dt;
        Ok(())
    }

    fn check_model(
        &mut self,
        model: &mut StarModel,
    ) -> Result<RelaxHookResult, StarError> {
        let new_mass = self.new_mass;
        let mut dlgm_per_step = self.dlgm_per_step;
        let sm = star_mass(model);
        if sm < 0.01 {
            dlgm_per_step *= 0.1;
        }

        if (sm - new_mass).abs() < 1.0e-12 * new_mass {
            set_star_mass(model, new_mass);
            return Ok(finished());
        }

        let relax_mass_scale_dt = SECYER * self.change_mass_years_for_dt;

        // Give a chance to stabilize.
        if model.dt < relax_mass_scale_dt * 0.9 {
            return Ok(RelaxHookResult::KeepGoing);
        }

        let next = if new_mass < sm {
            let next = flexmesa_math::exp10(
                flexmesa_math::safe_log10(sm) - dlgm_per_step,
            );
            if next < new_mass {
                new_mass
            } else {
                next
            }
        } else {
            let next = flexmesa_math::exp10(
                flexmesa_math::safe_log10(sm) + dlgm_per_step,
            );
            if next > new_mass {
                new_mass
            } else {
                next
            }
        };
        set_star_mass(model, next);
        Ok(RelaxHookResult::KeepGoing)
    }
}

/// `do_relax_mass_scale` (`star/private/relax.f90:2195-2258`): changes the
/// total mass to `new_mass` ($M_\odot$) by rescaling the cell masses, by
/// `dlgm_per_step` in $\log_{10} M$ per step of `change_mass_years_for_dt`
/// years.
///
/// # Errors
/// Returns [`StarError::InvalidInput`] for `new_mass <= 0` and the relax error
/// when the evolve loop fails.
pub(crate) fn do_relax_mass_scale(
    model: &mut StarModel,
    new_mass: f64,
    dlgm_per_step: f64,
    change_mass_years_for_dt: f64,
    extras: &mut dyn StarExtras,
) -> Result<(), StarError> {
    if new_mass <= 0.0 {
        return Err(StarError::InvalidInput(format!(
            "invalid new_mass {new_mass} (star/private/relax.f90:2213-2217)"
        )));
    }
    if (new_mass - star_mass(model)).abs() <= 1.0e-12 * new_mass {
        set_star_mass(model, new_mass);
        return Ok(());
    }
    let max_model_number = model.controls.max_model_number;
    model.controls.max_model_number = None;
    let eps_mdot_factor = model.controls.mix_wind.eps_mdot_factor;
    model.controls.mix_wind.eps_mdot_factor = 0.0;
    let max_years_for_timestep = model.controls.max_years_for_timestep;
    let relax_mass_scale_dt = SECYER * change_mass_years_for_dt;
    model.controls.max_years_for_timestep = relax_mass_scale_dt / SECYER;

    let mut hooks = RelaxMassScale {
        new_mass,
        dlgm_per_step,
        change_mass_years_for_dt,
    };
    let res = do_internal_evolve(model, &mut hooks, true, extras);

    model.controls.max_model_number = max_model_number;
    set_star_mass(model, new_mass);
    model.controls.mix_wind.eps_mdot_factor = eps_mdot_factor;
    model.controls.max_years_for_timestep = max_years_for_timestep;
    res
}

/// 0-based inclusive zone range `klo..=khi` for `minq`/`maxq`
/// (`khi = k_for_q(s, minq)`, `klo = k_for_q(s, maxq)`).
fn zone_range_for_q(model: &StarModel, minq: f64, maxq: f64) -> (usize, usize) {
    let khi = k_for_q_1based(model, minq);
    let klo = k_for_q_1based(model, maxq);
    (klo - 1, khi - 1)
}

/// `before_evolve_relax_Z` and `relax_Z_check_model`
/// (`star/private/relax.f90:3594-3701`).
struct RelaxZ {
    /// `rpar(1) = log(max(min_z, new_z))`.
    lnz_target: f64,
    /// `rpar(2)`.
    new_z: f64,
    /// `rpar(3) = abs(dlnz)`.
    dlnz: f64,
    /// `rpar(4)`.
    minq: f64,
    /// `rpar(5)`.
    maxq: f64,
}

impl RelaxHooks for RelaxZ {
    fn before_evolve(
        &mut self,
        model: &mut StarModel,
    ) -> Result<(), StarError> {
        setup_before_relax(model);
        model.controls.max_model_number = None;
        model.controls.ts_stop.max_timestep = SECYER;
        model.dt_next = model.controls.ts_stop.max_timestep;
        Ok(())
    }

    fn check_model(
        &mut self,
        model: &mut StarModel,
    ) -> Result<RelaxHookResult, StarError> {
        let (lnz_target, new_z) = (self.lnz_target, self.new_z);
        let mut dlnz = self.dlnz;
        let (klo, khi) = zone_range_for_q(model, self.minq, self.maxq);

        let current_z = eval_current_z(model, klo, khi)?;

        if (current_z - new_z).abs() <= 1.0e-6 * new_z {
            return Ok(finished());
        }

        let lnz = flexmesa_math::log(current_z.max(MIN_Z));

        if (lnz - lnz_target).abs() < dlnz {
            dlnz = (lnz - lnz_target).abs();
        }

        let new_lnz = if lnz_target < lnz {
            lnz - dlnz
        } else {
            lnz + dlnz
        };

        let next_z = if new_lnz >= MIN_DLNZ {
            flexmesa_math::exp(new_lnz)
        } else {
            new_z
        };

        // `set_z` failing sets `result_reason = nonzero_ierr` and terminates
        // without a termination code (`star/private/relax.f90:3687-3692`),
        // so `do_internal_evolve` returns `ierr = -1`.
        set_z(model, next_z, klo, khi)?;

        if klo == 0 && khi == model.nz - 1 {
            model.controls.initial_z = next_z;
        }
        model.controls.ts_stop.max_timestep = SECYER * (model.dt / SECYER);
        Ok(RelaxHookResult::KeepGoing)
    }
}

/// `do_relax_Z` (`star/private/relax.f90:3536-3591`): relaxes $Z$ between
/// `minq` and `maxq` to `new_z` by `dlnz` in $\ln Z$ per step.
///
/// # Errors
/// Returns [`StarError::InvalidInput`] for `new_z` outside `[0, 1]`, a
/// `set_z` failure, and the relax error when the evolve loop fails.
pub(crate) fn do_relax_z(
    model: &mut StarModel,
    new_z: f64,
    dlnz: f64,
    minq: f64,
    maxq: f64,
    extras: &mut dyn StarExtras,
) -> Result<(), StarError> {
    if !(0.0..=1.0).contains(&new_z) {
        return Err(StarError::InvalidInput(format!(
            "invalid new_Z {new_z} (star/private/relax.f90:3554-3558)"
        )));
    }
    let nz = model.nz;
    let z = eval_current_z(model, 0, nz - 1)?;
    if (new_z - z).abs() <= 1.0e-6 * z {
        return Ok(());
    }
    if new_z.max(z) > 1.0e-6 && (new_z - z).abs() <= 1.0e-3 * new_z {
        return set_z(model, new_z, 0, nz - 1);
    }
    let max_model_number = model.controls.max_model_number;
    model.controls.max_model_number = None;
    model.controls.initial_z = z;
    let mut hooks = RelaxZ {
        lnz_target: flexmesa_math::log(MIN_Z.max(new_z)),
        new_z,
        dlnz: dlnz.abs(),
        minq,
        maxq,
    };
    let res = do_internal_evolve(model, &mut hooks, true, extras);
    model.controls.max_model_number = max_model_number;
    res
}

/// `before_evolve_relax_Y` and `relax_Y_check_model`
/// (`star/private/relax.f90:3761-3873`).
struct RelaxY {
    /// `rpar(1)`.
    new_y: f64,
    /// `rpar(2) = abs(dY)`.
    dy: f64,
    /// `rpar(3)`.
    minq: f64,
    /// `rpar(4)`.
    maxq: f64,
}

impl RelaxHooks for RelaxY {
    fn before_evolve(
        &mut self,
        model: &mut StarModel,
    ) -> Result<(), StarError> {
        setup_before_relax(model);
        model.controls.max_model_number = None;
        model.controls.ts_stop.max_timestep = SECYER;
        model.dt_next = model.controls.ts_stop.max_timestep;
        Ok(())
    }

    fn check_model(
        &mut self,
        model: &mut StarModel,
    ) -> Result<RelaxHookResult, StarError> {
        let new_y = self.new_y;
        let mut dy = self.dy;
        let mut end_now = false;
        let (klo, khi) = zone_range_for_q(model, self.minq, self.maxq);

        let current_y = eval_current_y(model, klo, khi)?;
        if !current_y.is_finite() {
            return Err(StarError::ConvergenceFailure(format!(
                "relax_y_check_model: current_y = {current_y} \
                 (star/private/relax.f90:3812-3815)"
            )));
        }

        if (current_y - new_y).abs() < 1.0e-10 {
            return Ok(finished());
        }

        if (current_y - new_y).abs() < dy {
            dy = (current_y - new_y).abs();
            end_now = true;
        }

        let next_y = if new_y >= current_y {
            current_y + dy
        } else {
            current_y - dy
        };

        // `set_y` failing terminates without a termination code
        // (`star/private/relax.f90:3851-3856`).
        set_y(model, next_y, klo, khi)?;

        let actual_next_y = eval_current_y(model, klo, khi)?;

        if (actual_next_y - new_y).abs() < 1.0e-10 || end_now {
            return Ok(finished());
        }

        if klo == 0 && khi == model.nz - 1 {
            model.controls.initial_y = next_y;
        }
        model.controls.ts_stop.max_timestep = SECYER * (model.dt / SECYER);
        Ok(RelaxHookResult::KeepGoing)
    }
}

/// `do_relax_Y` (`star/private/relax.f90:3704-3758`): relaxes $Y$ between
/// `minq` and `maxq` to `new_y` by `dy` per step.
///
/// # Errors
/// Returns [`StarError::InvalidInput`] for `new_y` outside `[0, 1]`, a
/// non-finite current $Y$, a `set_y` failure, and the relax error when the
/// evolve loop fails.
pub(crate) fn do_relax_y(
    model: &mut StarModel,
    new_y: f64,
    dy: f64,
    minq: f64,
    maxq: f64,
    extras: &mut dyn StarExtras,
) -> Result<(), StarError> {
    let (klo, khi) = zone_range_for_q(model, minq, maxq);
    let y = eval_current_y(model, klo, khi)?;
    if !y.is_finite() {
        return Err(StarError::ConvergenceFailure(format!(
            "do_relax_Y: y = {y} (star/private/relax.f90:3730-3733)"
        )));
    }
    if (new_y - y).abs() <= 1.0e-6 * new_y {
        return Ok(());
    }
    if !(0.0..=1.0).contains(&new_y) {
        return Err(StarError::InvalidInput(format!(
            "invalid new_Y {new_y} (star/private/relax.f90:3736-3740)"
        )));
    }
    let max_model_number = model.controls.max_model_number;
    model.controls.max_model_number = None;
    model.controls.initial_y = y;
    let mut hooks = RelaxY {
        new_y,
        dy: dy.abs(),
        minq,
        maxq,
    };
    let res = do_internal_evolve(model, &mut hooks, true, extras);
    model.controls.max_model_number = max_model_number;
    res
}

/// `before_evolve_relax_composition`, `relax_composition_adjust_model` and
/// `relax_composition_check_model` (`star/private/relax.f90:424-547`).
struct RelaxComposition {
    interp: RelaxCompositionInterpolants,
    /// `ipar(2)`.
    num_steps_to_use: usize,
    /// `ipar(3)`.
    starting_model_number: usize,
}

impl RelaxHooks for RelaxComposition {
    fn before_evolve(
        &mut self,
        model: &mut StarModel,
    ) -> Result<(), StarError> {
        setup_before_relax(model);
        Ok(())
    }

    fn adjust_model(
        &mut self,
        model: &mut StarModel,
    ) -> Result<RelaxHookResult, StarError> {
        let lambda = RelaxCompositionInterpolants::eval_lambda(
            model.dt,
            model.job.timescale_for_relax_composition,
            model.model_number,
            self.starting_model_number,
            self.num_steps_to_use,
        )?;
        // Upstream sets `terminate` when `adjust_xa` returns `ierr /= 0` and
        // then overwrites it with `keep_going` (`star/private/relax.f90:
        // 469-474`), so a failed `adjust_xa` is ignored. Its only failing
        // path (`allocate(..., stat=ierr)`, `relax.f90:485-486`) returns
        // before `xa` is touched, and `apply_step` also fails only before it
        // changes `xa`, so the step continues with `xa` unchanged.
        // Reproduced; deferred bug #440.
        // `do_evolve_step_part1` starts fresh and sets the variables of the
        // unchanged composition (`star/private/evolve.f90:319, 409`) before
        // `adjust_model` runs.
        crate::hydro_vars::set_hydro_vars(model, false, false)?;
        let _ = self.interp.apply_step(model, lambda);
        // `adjust_xa` changes `xa` without setting `need_to_setvars`, so the
        // solve starts from the microphysics of the unchanged composition
        // (`star/private/evolve.f90:580`,
        // `star/private/solver_support.f90:108-118`).
        model.aux.need_to_setvars = false;
        Ok(RelaxHookResult::KeepGoing)
    }

    fn check_model(
        &mut self,
        model: &mut StarModel,
    ) -> Result<RelaxHookResult, StarError> {
        let timescale = model.job.timescale_for_relax_composition;
        if timescale < 0.0 {
            if model.model_number as i64 - self.starting_model_number as i64
                >= self.num_steps_to_use as i64
            {
                return Ok(finished());
            }
        } else if model.generations > 1
            && model.dt > timescale
            && model.dt_old > timescale
        {
            return Ok(finished());
        }
        Ok(RelaxHookResult::KeepGoing)
    }
}

/// `do_relax_composition` (`star/private/relax.f90:316-421`): relaxes the
/// composition toward `profile` (`xq`, `xa`) over `num_steps_to_use` steps,
/// or on `timescale_for_relax_composition`, with burning, mixing and element
/// diffusion off.
///
/// # Errors
/// Returns [`StarError::InvalidInput`] when the profile's species count does
/// not match the model or its interpolants cannot be built, and the relax
/// error when the evolve loop fails.
pub(crate) fn do_relax_composition(
    model: &mut StarModel,
    num_steps_to_use: usize,
    profile: &CompositionProfile,
    extras: &mut dyn StarExtras,
) -> Result<(), StarError> {
    if profile.species != model.species {
        return Err(StarError::InvalidInput(format!(
            "relax_composition: profile has {} species, model has {} \
             (star/private/relax.f90:345-348)",
            profile.species, model.species
        )));
    }
    let interp = RelaxCompositionInterpolants::from_profile(profile)?;

    let max_model_number = model.controls.max_model_number;
    model.controls.max_model_number =
        Some(num_steps_to_use + model.model_number + 1);

    let c = &mut model.controls;
    let dxdt_nuc_factor = c.dxdt_nuc_factor;
    c.dxdt_nuc_factor = 0.0;
    let mix_factor = c.mix_factor;
    c.mix_factor = 0.0;
    let do_element_diffusion = c.do_element_diffusion;
    c.do_element_diffusion = false;
    let include_composition_in_eps_grav = c.include_composition_in_eps_grav;
    c.include_composition_in_eps_grav = false;
    let starting_dt_next = model.dt_next;

    let mut hooks = RelaxComposition {
        interp,
        num_steps_to_use,
        starting_model_number: model.model_number,
    };
    let res = do_internal_evolve(model, &mut hooks, true, extras);

    model.controls.max_model_number = max_model_number;
    model.dt_next = starting_dt_next;
    let c = &mut model.controls;
    c.dxdt_nuc_factor = dxdt_nuc_factor;
    c.mix_factor = mix_factor;
    c.do_element_diffusion = do_element_diffusion;
    c.include_composition_in_eps_grav = include_composition_in_eps_grav;
    res
}

/// `do1_relax_R_center` (`star/private/relax.f90:2616-2637`): sets
/// `R_center` and rebuilds every `lnR` outward from it so each cell keeps its
/// density, as the first guess for the next model.
fn do1_relax_r_center(model: &mut StarModel, new_r_center: f64) {
    model.r_center = new_r_center;
    let mut rp13 = model.r_center * model.r_center * model.r_center;
    for k in (0..model.nz).rev() {
        let dm = model.dm[k];
        let rho = model.rho[k];
        let dr3 = dm / (rho * FOUR_THIRDS_PI);
        let ln_r = flexmesa_math::log(rp13 + dr3) * ONE_THIRD;
        model.ln_r[k] = ln_r;
        model.r[k] = flexmesa_math::exp(ln_r);
        rp13 += dr3;
    }
}

/// `do1_relax_L_center` (`star/private/relax.f90:2871-2889`): sets
/// `L_center` and shifts every `L` by the change.
fn do1_relax_l_center(model: &mut StarModel, new_l_center: f64) {
    let l_center_prev = model.l_center;
    model.l_center = new_l_center;
    let d_l = new_l_center - l_center_prev;
    for l in &mut model.lum[..model.nz] {
        *l += d_l;
    }
}

/// `do1_relax_core` (`star/private/relax.f90:2178-2192`): `next` in
/// $M_\odot$; `core_avg_rho` in g cm$^{-3}$, `core_avg_eps` in erg g$^{-1}$
/// s$^{-1}$.
fn do1_relax_core(
    model: &mut StarModel,
    next: f64,
    core_avg_rho: f64,
    core_avg_eps: f64,
) {
    model.m_center = next * M_SUN;
    model.xmstar = model.mstar - model.m_center;
    let next_r_center = flexmesa_math::pow(
        model.m_center / (core_avg_rho * FOUR_THIRDS_PI),
        ONE_THIRD,
    );
    do1_relax_r_center(model, next_r_center);
    let next_l_center = model.m_center * core_avg_eps;
    do1_relax_l_center(model, next_l_center);
}

/// `before_evolve_relax_core` and `relax_core_check_model`
/// (`star/private/relax.f90:2063-2176`).
struct RelaxCore {
    new_core_mass: f64,
    dlg_core_mass_per_step: f64,
    years_for_dt: f64,
    core_avg_rho: f64,
    core_avg_eps: f64,
}

impl RelaxHooks for RelaxCore {
    fn before_evolve(&mut self, model: &mut StarModel) -> Result<(), StarError> {
        setup_before_relax(model);
        model.controls.max_model_number = None;
        model.dt_next = SECYER * self.years_for_dt;
        Ok(())
    }

    fn check_model(
        &mut self,
        model: &mut StarModel,
    ) -> Result<RelaxHookResult, StarError> {
        let new_core_mass = self.new_core_mass;
        let current_core_mass = model.m_center / M_SUN;
        if (new_core_mass - current_core_mass).abs() <= 1.0e-12 * new_core_mass
        {
            do1_relax_core(
                model,
                new_core_mass,
                self.core_avg_rho,
                self.core_avg_eps,
            );
            return Ok(finished());
        }

        let relax_core_dt = SECYER * self.years_for_dt;
        // Give a chance to stabilize.
        if model.dt < relax_core_dt * 0.9 {
            return Ok(RelaxHookResult::KeepGoing);
        }

        let mut end_now = false;
        let next = if new_core_mass < current_core_mass {
            let next = flexmesa_math::exp10(
                flexmesa_math::safe_log10(current_core_mass)
                    - self.dlg_core_mass_per_step,
            );
            if next < new_core_mass {
                end_now = true;
                new_core_mass
            } else {
                next
            }
        } else {
            let next = flexmesa_math::exp10(
                flexmesa_math::log10(current_core_mass.max(1.0e-8))
                    + self.dlg_core_mass_per_step,
            );
            if next > new_core_mass {
                end_now = true;
                new_core_mass
            } else {
                next
            }
        };

        do1_relax_core(model, next, self.core_avg_rho, self.core_avg_eps);

        if end_now {
            return Ok(finished());
        }
        Ok(RelaxHookResult::KeepGoing)
    }
}

/// `do_relax_core` (`star/private/relax.f90:1999-2060`): moves the inner
/// boundary mass `M_center` to `new_core_mass` ($M_\odot$) by
/// `dlg_core_mass_per_step` in $\log_{10}$ per step of
/// `relax_core_years_for_dt` years, with `R_center` from `core_avg_rho` and
/// `L_center` from `core_avg_eps`.
///
/// # Errors
/// Returns [`StarError::InvalidInput`] for `new_core_mass <= 0` and the relax
/// error when the evolve loop fails.
pub(crate) fn do_relax_core(
    model: &mut StarModel,
    new_core_mass: f64,
    dlg_core_mass_per_step: f64,
    years_for_dt: f64,
    core_avg_rho: f64,
    core_avg_eps: f64,
    extras: &mut dyn StarExtras,
) -> Result<(), StarError> {
    if new_core_mass <= 0.0 {
        return Err(StarError::InvalidInput(format!(
            "invalid new_core_mass {new_core_mass} \
             (star/private/relax.f90:2022-2026)"
        )));
    }
    let current_core_mass = model.m_center / M_SUN;
    if (new_core_mass - current_core_mass).abs() <= 1.0e-12 * new_core_mass {
        do1_relax_core(model, new_core_mass, core_avg_rho, core_avg_eps);
        return Ok(());
    }
    let max_model_number = model.controls.max_model_number;
    model.controls.max_model_number = None;
    let max_years_for_timestep = model.controls.max_years_for_timestep;
    model.controls.max_years_for_timestep = years_for_dt;
    let mut hooks = RelaxCore {
        new_core_mass,
        dlg_core_mass_per_step,
        years_for_dt,
        core_avg_rho,
        core_avg_eps,
    };
    let res = do_internal_evolve(model, &mut hooks, true, extras);
    model.controls.max_model_number = max_model_number;
    model.controls.max_years_for_timestep = max_years_for_timestep;
    res
}

/// `set_new_mass_for_relax_M_center` (`star/private/relax.f90:2477-2488`).
fn set_new_mass_for_relax_m_center(model: &mut StarModel, new_mass: f64) {
    model.mstar = new_mass * M_SUN;
    model.m_center = model.mstar - model.xmstar;
}

/// `before_evolve_relax_M_center` and `relax_M_center_check_model`
/// (`star/private/relax.f90:2392-2475`).
struct RelaxMCenter {
    new_mass: f64,
    dlgm_per_step: f64,
    dt_seconds: f64,
}

impl RelaxHooks for RelaxMCenter {
    fn before_evolve(&mut self, model: &mut StarModel) -> Result<(), StarError> {
        setup_before_relax(model);
        model.controls.max_model_number = None;
        model.dt_next = self.dt_seconds;
        Ok(())
    }

    fn check_model(
        &mut self,
        model: &mut StarModel,
    ) -> Result<RelaxHookResult, StarError> {
        let new_mass = self.new_mass;
        let sm = star_mass(model);
        let mut end_now = false;
        let next = if new_mass < sm {
            let next = flexmesa_math::exp10(
                flexmesa_math::safe_log10(sm) - self.dlgm_per_step,
            );
            if next < new_mass {
                end_now = true;
                new_mass
            } else {
                next
            }
        } else {
            let next = flexmesa_math::exp10(
                flexmesa_math::safe_log10(sm) + self.dlgm_per_step,
            );
            if next > new_mass {
                end_now = true;
                new_mass
            } else {
                next
            }
        };

        if (sm - new_mass).abs() < 1.0e-15 || end_now {
            set_new_mass_for_relax_m_center(model, new_mass);
            return Ok(finished());
        }

        // Give a chance to stabilize.
        if model.dt < self.dt_seconds * 0.9 {
            return Ok(RelaxHookResult::KeepGoing);
        }

        set_new_mass_for_relax_m_center(model, next);
        Ok(RelaxHookResult::KeepGoing)
    }
}

/// `do_relax_M_center` (`star/private/relax.f90:2339-2389`): changes the
/// total mass to `new_mass` ($M_\odot$) at fixed envelope mass `xmstar` by
/// moving `M_center`, `dlgm_per_step` in $\log_{10}$ per step of
/// `relax_M_center_dt` seconds.
///
/// # Errors
/// Returns [`StarError::InvalidInput`] for `new_mass <= 0` and the relax error
/// when the evolve loop fails.
pub(crate) fn do_relax_m_center(
    model: &mut StarModel,
    new_mass: f64,
    dlgm_per_step: f64,
    dt_seconds: f64,
    extras: &mut dyn StarExtras,
) -> Result<(), StarError> {
    if new_mass <= 0.0 {
        return Err(StarError::InvalidInput(format!(
            "invalid new_mass {new_mass} (star/private/relax.f90:2354-2358)"
        )));
    }
    if (new_mass - star_mass(model)).abs() <= 1.0e-6 {
        set_star_mass(model, new_mass);
        return Ok(());
    }
    let max_model_number = model.controls.max_model_number;
    let max_years_for_timestep = model.controls.max_years_for_timestep;
    model.controls.max_model_number = None;
    model.controls.max_years_for_timestep = dt_seconds / SECYER;
    let mut hooks = RelaxMCenter {
        new_mass,
        dlgm_per_step,
        dt_seconds,
    };
    let res = do_internal_evolve(model, &mut hooks, true, extras);
    model.controls.max_model_number = max_model_number;
    model.controls.max_years_for_timestep = max_years_for_timestep;
    res
}

/// `before_evolve_relax_tau_factor` and `relax_tau_factor_check_model`
/// (`star/private/relax.f90:1442-1502`).
struct RelaxTauFactor {
    new_tau_factor: f64,
    dlogtau_factor: f64,
}

impl RelaxHooks for RelaxTauFactor {
    fn before_evolve(&mut self, model: &mut StarModel) -> Result<(), StarError> {
        turn_off_winds(model);
        model.controls.max_model_number = None;
        Ok(())
    }

    fn check_model(
        &mut self,
        model: &mut StarModel,
    ) -> Result<RelaxHookResult, StarError> {
        let new_tau_factor = self.new_tau_factor;
        let current = model.controls.tau_factor;
        if (current - new_tau_factor).abs() < 1.0e-15 {
            model.controls.tau_factor = new_tau_factor;
            return Ok(finished());
        }
        let next = if new_tau_factor < current {
            let next = flexmesa_math::exp10(
                flexmesa_math::safe_log10(current) - self.dlogtau_factor,
            );
            if next < new_tau_factor { new_tau_factor } else { next }
        } else {
            let next = flexmesa_math::exp10(
                flexmesa_math::safe_log10(current) + self.dlogtau_factor,
            );
            if next > new_tau_factor { new_tau_factor } else { next }
        };
        model.controls.tau_factor = next;
        model.controls.ts_stop.max_timestep = SECYER * (model.dt / SECYER);
        Ok(RelaxHookResult::KeepGoing)
    }
}

/// `do_relax_tau_factor` (`star/private/relax.f90:1394-1439`): relaxes
/// `tau_factor` to `new_tau_factor` by `dlogtau_factor` in $\log_{10}$ per
/// step. Upstream then sets `force_tau_factor`, which only a photo restart
/// reads (`star/private/photo_in.f90:95-98`); this path has none.
///
/// # Errors
/// Returns [`StarError::InvalidInput`] for `new_tau_factor <= 0` and the
/// relax error when the evolve loop fails.
pub(crate) fn do_relax_tau_factor(
    model: &mut StarModel,
    new_tau_factor: f64,
    dlogtau_factor: f64,
    extras: &mut dyn StarExtras,
) -> Result<(), StarError> {
    if new_tau_factor <= 0.0 {
        return Err(StarError::InvalidInput(format!(
            "invalid new_tau_factor {new_tau_factor} \
             (star/private/relax.f90:1410-1414)"
        )));
    }
    if (new_tau_factor - model.controls.tau_factor).abs() <= 1.0e-6 {
        model.controls.tau_factor = new_tau_factor;
        return Ok(());
    }
    let max_model_number = model.controls.max_model_number;
    model.controls.max_model_number = None;
    let mut hooks = RelaxTauFactor {
        new_tau_factor,
        dlogtau_factor,
    };
    let res = do_internal_evolve(model, &mut hooks, true, extras);
    model.controls.max_model_number = max_model_number;
    res
}

/// `before_evolve_relax_irradiation` and `relax_irradiation_check_model`
/// (`star/private/relax.f90:1800-1874`).
struct RelaxIrradiation {
    /// `ipar(1)`, advanced while waiting for `dt`.
    adjust_model: i64,
    /// `ipar(2)`.
    max_num_steps: i64,
    new_irrad_flux: f64,
    new_irrad_col_depth: f64,
    old_irrad_flux: f64,
    old_irrad_col_depth: f64,
}

impl RelaxHooks for RelaxIrradiation {
    fn before_evolve(&mut self, model: &mut StarModel) -> Result<(), StarError> {
        turn_off_winds(model);
        model.controls.max_model_number = None;
        Ok(())
    }

    fn check_model(
        &mut self,
        model: &mut StarModel,
    ) -> Result<RelaxHookResult, StarError> {
        let num_steps = model.model_number as i64 - self.adjust_model;
        let frac = num_steps as f64 / self.max_num_steps as f64;

        if model.dt < model.controls.max_years_for_timestep * SECYER {
            self.adjust_model += 1;
            return Ok(RelaxHookResult::KeepGoing);
        }

        if frac >= 1.0 {
            model.controls.irradiation_flux = self.new_irrad_flux;
            model.controls.column_depth_for_irradiation =
                self.new_irrad_col_depth;
            return Ok(finished());
        }

        model.controls.irradiation_flux = frac * self.new_irrad_flux
            + (1.0 - frac) * self.old_irrad_flux;
        model.controls.column_depth_for_irradiation = frac
            * self.new_irrad_col_depth
            + (1.0 - frac) * self.old_irrad_col_depth;
        Ok(RelaxHookResult::KeepGoing)
    }
}

/// `do_relax_irradiation` (`star/private/relax.f90:1743-1797`): ramps the
/// irradiation flux and column depth linearly to the new values over at
/// least `min_steps` steps taken at `relax_irradiation_max_yrs_dt`.
///
/// # Errors
/// Returns the relax error when the evolve loop fails.
pub(crate) fn do_relax_irradiation(
    model: &mut StarModel,
    min_steps: i64,
    new_irrad_flux: f64,
    new_irrad_col_depth: f64,
    relax_irradiation_max_yrs_dt: f64,
    extras: &mut dyn StarExtras,
) -> Result<(), StarError> {
    let mut hooks = RelaxIrradiation {
        adjust_model: model.model_number as i64,
        max_num_steps: min_steps,
        new_irrad_flux,
        new_irrad_col_depth,
        old_irrad_flux: model.controls.irradiation_flux,
        old_irrad_col_depth: model.controls.column_depth_for_irradiation,
    };
    if (hooks.new_irrad_flux - hooks.old_irrad_flux).abs() <= 1.0e-10
        && (hooks.new_irrad_col_depth - hooks.old_irrad_col_depth).abs()
            <= 1.0e-10
    {
        return Ok(());
    }
    let max_model_number = model.controls.max_model_number;
    model.controls.max_model_number = None;
    let max_years_for_timestep = model.controls.max_years_for_timestep;
    model.controls.max_years_for_timestep = relax_irradiation_max_yrs_dt;
    let res = do_internal_evolve(model, &mut hooks, true, extras);
    model.controls.max_model_number = max_model_number;
    model.controls.max_years_for_timestep = max_years_for_timestep;
    res
}
