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

use flexmesa_const::{M_SUN, SECYER};

use crate::controls::StarError;
use crate::core_boundaries::k_for_q_1based;
use crate::inlist_runner::StarExtras;
use crate::model::StarModel;
use crate::relax_comp::{
    eval_current_y, eval_current_z, set_y, set_z, CompositionProfile,
    RelaxCompositionInterpolants,
};
use crate::relax_driver::{
    do_internal_evolve, setup_before_relax, RelaxHookResult, RelaxHooks,
    RELAX_FINISHED_OKAY,
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
