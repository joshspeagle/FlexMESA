// SPDX-License-Identifier: GPL-3.0-or-later
//! What the run loop does after each accepted step, once
//! `star_finish_step` has returned `keep_going`
//! (`after_step_loop`, `star/job/run_star_support.f90:752-763`):
//! `adjust_tau_factor` (`:913-958`, with `relax_tau_factor` at `:1069-1092`),
//! then `do_rotation_near_zams` (`:994-1067`) when the star has reached the
//! ZAMS without rotation, then `do_rotation` (`:960-992`) when it rotates.

use flexmesa_const::{L_SUN, R_SUN};
use flexmesa_math::{exp10, log10};

use crate::controls::StarError;
use crate::inlist_runner::StarExtras;
use crate::model::StarModel;
use crate::relax_driver::{RelaxOmegaKind, do_relax_uniform_omega};

/// The rotation and `tau_factor` tail of `after_step_loop`
/// (`star/job/run_star_support.f90:752-763`).
///
/// # Errors
/// Returns the [`StarError`] of a failed rotation relax or of an input the
/// port rejects.
pub(crate) fn after_step_rotation_and_tau(
    model: &mut StarModel,
    extras: &mut dyn StarExtras,
) -> Result<(), StarError> {
    adjust_tau_factor(model);
    // `s% L_nuc_burn_total = s% power_nuc_burn` (`report.f90:112`) and
    // `s% L_phot = s% L(1)/Lsun` (`hydro_vars.f90:711`).
    let l_nuc_burn_total =
        crate::core_boundaries::compute_power_categories(model).power_nuc_burn;
    let l_phot = model.lum[0] / L_SUN;
    if l_nuc_burn_total / l_phot >= model.controls.lnuc_div_l_zams_limit
        && !model.controls.rotation_flag
    {
        do_rotation_near_zams(model, extras)?;
    }
    if model.controls.rotation_flag {
        do_rotation(model);
    }
    Ok(())
}

/// Central mass fraction `xa(j, nz)` of a nuclide, or `None` when it is not
/// in the network (`net_iso(i) == 0`).
fn center_xa(model: &StarModel, iso: &str) -> Option<f64> {
    let (_, slot) = crate::report_utils::resolve_iso_slot(model, iso).ok()?;
    slot.map(|j| model.cell_xa(model.nz - 1)[j])
}

/// `after_He_burn` (`star/private/star_utils.f90:2690-2703`).
fn after_he_burn(model: &StarModel, he4_limit: f64) -> bool {
    const SMALL: f64 = 1.0e-4;
    let (Some(h1), Some(he4)) =
        (center_xa(model, "h1"), center_xa(model, "he4"))
    else {
        return false;
    };
    !(h1 > SMALL || he4 > he4_limit)
}

/// `after_C_burn` (`star/private/star_utils.f90:2706-2721`).
fn after_c_burn(model: &StarModel, c12_limit: f64) -> bool {
    const SMALL: f64 = 1.0e-4;
    let (Some(h1), Some(he4), Some(c12)) = (
        center_xa(model, "h1"),
        center_xa(model, "he4"),
        center_xa(model, "c12"),
    ) else {
        return false;
    };
    !(h1 > SMALL || he4 > SMALL || c12 > c12_limit)
}

/// `safe_log10` (`math/public/math_lib_crmath.f90`): `log10(max(1d-99, x))`.
fn safe_log10(x: f64) -> f64 {
    log10(x.max(1.0e-99))
}

/// `relax_tau_factor` (`star/job/run_star_support.f90:1069-1092`): one step
/// of `dlogtau_factor` toward `relax_to_this_tau_factor`.
fn relax_tau_factor(model: &mut StarModel) {
    let job = &model.job;
    let tau = model.controls.tau_factor;
    let target = job.relax_to_this_tau_factor;
    let next = if target < tau {
        let next = exp10(safe_log10(tau) - job.dlogtau_factor);
        if next < target { target } else { next }
    } else {
        let next = exp10(safe_log10(tau) + job.dlogtau_factor);
        if next > target { target } else { next }
    };
    if next != tau {
        model.controls.tau_factor = next;
        model.aux.need_to_setvars = true;
    }
}

/// `abs(a - b) > 1d-6*max(a, b)` as written at
/// `star/job/run_star_support.f90:925-926, 935-936, 945-946, 952-953`.
fn differs(a: f64, b: f64) -> bool {
    (a - b).abs() > 1.0e-6 * a.max(b)
}

/// `adjust_tau_factor` (`star/job/run_star_support.f90:913-958`).
fn adjust_tau_factor(model: &mut StarModel) {
    let job = model.job.clone();
    if job.adjust_tau_factor_to_surf_density
        && job.base_for_adjust_tau_factor_to_surf_density > 0.0
    {
        model.controls.tau_factor =
            model.rho[0] / job.base_for_adjust_tau_factor_to_surf_density;
        model.aux.need_to_setvars = true;
    }
    if job.set_tau_factor_after_core_he_burn > 0.0
        && differs(model.controls.tau_factor, job.set_to_this_tau_factor)
        && after_he_burn(model, job.set_tau_factor_after_core_he_burn)
    {
        model.controls.tau_factor = job.set_to_this_tau_factor;
        model.aux.need_to_setvars = true;
    }
    if job.set_tau_factor_after_core_c_burn > 0.0
        && differs(model.controls.tau_factor, job.set_to_this_tau_factor)
        && after_c_burn(model, job.set_tau_factor_after_core_c_burn)
    {
        model.controls.tau_factor = job.set_to_this_tau_factor;
        model.aux.need_to_setvars = true;
    }
    if job.relax_tau_factor_after_core_he_burn > 0.0
        && differs(model.controls.tau_factor, job.relax_to_this_tau_factor)
        && after_he_burn(model, job.relax_tau_factor_after_core_he_burn)
    {
        relax_tau_factor(model);
    }
    if job.relax_tau_factor_after_core_c_burn > 0.0
        && differs(model.controls.tau_factor, job.relax_to_this_tau_factor)
        && after_c_burn(model, job.relax_tau_factor_after_core_c_burn)
    {
        relax_tau_factor(model);
    }
}

/// `star_set_rotation_flag(id, .true.)` (`star/public/star_lib.f90:894-914`)
/// with `set_rotation_flag` (`star/private/set_flags.f90:748-804`).
///
/// `use_w_div_wc_flag_with_rotation` and `use_j_rot_flag_with_rotation` are
/// not ported (`#505`); they default to `.false.`. `D_omega`, `nu_ST`,
/// `D_ST`, `D_DSI`, `D_SH`, `D_SSI`, `D_ES`, `D_GSF`, `prev_mesh_omega` and
/// `prev_mesh_j_rot` are not stored by FlexMESA, so zeroing them has no
/// counterpart.
fn set_rotation_flag_on(model: &mut StarModel) {
    if model.controls.rotation_flag {
        return;
    }
    let nz = model.nz;
    model.controls.rotation_flag = true;
    model.omega.resize(nz, 0.0);
    model.omega.fill(0.0);
    model.j_rot.resize(nz, 0.0);
    model.j_rot.fill(0.0);
    model.aux.am_nu_rot.fill(0.0);
    // `if (rotation_flag .and. .not. previous_rotation_flag) call
    // set_rotation_info(s, .false., ierr)` (`star_lib.f90:911-913`).
    crate::hydro_rotation::set_rotation_info(model, false);
}

/// `do_rotation_near_zams` (`star/job/run_star_support.f90:994-1067`): the
/// first branch that applies switches rotation on and either sets a step
/// limit for `do_rotation` or relaxes to a uniform $\Omega$.
fn do_rotation_near_zams(
    model: &mut StarModel,
    extras: &mut dyn StarExtras,
) -> Result<(), StarError> {
    let job = model.job.clone();
    let mn = isize::try_from(model.model_number).unwrap_or(isize::MAX);
    let relax = |model: &mut StarModel,
                 extras: &mut dyn StarExtras,
                 kind: RelaxOmegaKind,
                 target: f64|
     -> Result<(), StarError> {
        do_relax_uniform_omega(
            model,
            kind,
            target,
            job.num_steps_to_relax_rotation,
            job.relax_omega_max_yrs_dt,
            extras,
        )
    };
    if job.set_near_zams_surface_rotation_v_steps > 0
        && job.new_surface_rotation_v != 0.0
    {
        model.job.new_rotation_flag = true;
        set_rotation_flag_on(model);
        model.job.set_surf_rotation_v_step_limit =
            mn + job.set_near_zams_surface_rotation_v_steps - 1;
    } else if job.set_near_zams_omega_steps > 0 && job.new_omega != 0.0 {
        model.job.new_rotation_flag = true;
        set_rotation_flag_on(model);
        model.job.set_omega_step_limit = mn + job.set_near_zams_omega_steps - 1;
    } else if job.set_near_zams_omega_div_omega_crit_steps > 0
        && job.new_omega_div_omega_crit != 0.0
    {
        model.job.new_rotation_flag = true;
        set_rotation_flag_on(model);
        model.job.set_omega_div_omega_crit_step_limit =
            mn + job.set_near_zams_omega_div_omega_crit_steps - 1;
    } else if job.near_zams_relax_omega && job.new_omega != 0.0 {
        model.job.new_rotation_flag = true;
        set_rotation_flag_on(model);
        relax(model, extras, RelaxOmegaKind::NewOmega, job.new_omega)?;
    } else if job.near_zams_relax_omega_div_omega_crit
        && job.new_omega_div_omega_crit != 0.0
    {
        model.job.new_rotation_flag = true;
        set_rotation_flag_on(model);
        relax(
            model,
            extras,
            RelaxOmegaKind::NewOmegaDivOmegaCrit,
            job.new_omega_div_omega_crit,
        )?;
    } else if job.near_zams_relax_initial_surface_rotation_v
        && job.new_surface_rotation_v != 0.0
    {
        model.job.new_rotation_flag = true;
        set_rotation_flag_on(model);
        relax(
            model,
            extras,
            RelaxOmegaKind::NewSurfaceRotationV,
            job.new_surface_rotation_v,
        )?;
    }
    Ok(())
}

/// `do_rotation` (`star/job/run_star_support.f90:960-992`): while
/// `model_number` is within a step limit set by `do_rotation_near_zams`, set
/// a uniform $\Omega$ (`star_set_uniform_omega`).
fn do_rotation(model: &mut StarModel) {
    let mn = isize::try_from(model.model_number).unwrap_or(isize::MAX);
    if mn <= model.job.set_surf_rotation_v_step_limit {
        let phot = crate::report_utils::eval_photosphere_info(model);
        model.job.new_omega =
            model.job.new_surface_rotation_v * 1.0e5 / (phot.r_rsun * R_SUN);
        let w = model.job.new_omega;
        crate::relax::apply_uniform_rotation(model, w);
    } else if mn <= model.job.set_omega_step_limit {
        let w = model.job.new_omega;
        crate::relax::apply_uniform_rotation(model, w);
    } else if mn <= model.job.set_omega_div_omega_crit_step_limit {
        // `star_surface_omega_crit` (`star/public/star_lib.f90:2338-2350`,
        // `star/job/run_star_support.f90:982-983`).
        let omega_crit = crate::report_utils::star_surface_omega_crit(model);
        model.job.new_omega = model.job.new_omega_div_omega_crit * omega_crit;
        let w = model.job.new_omega;
        crate::relax::apply_uniform_rotation(model, w);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controls::{StarControls, StarJobControls};

    fn zams() -> StarModel {
        crate::zams::load_zams_model(
            StarControls {
                initial_mass: 1.0,
                zbase: 0.02,
                ..StarControls::default()
            },
            StarJobControls::default(),
        )
        .expect("1 Msun ZAMS")
    }

    #[test]
    fn test_adjust_tau_factor_to_surf_density_sets_rho1_over_base() {
        let mut m = zams();
        m.job.adjust_tau_factor_to_surf_density = true;
        m.job.base_for_adjust_tau_factor_to_surf_density = 1.0e-9;
        adjust_tau_factor(&mut m);
        assert_eq!(
            m.controls.tau_factor.to_bits(),
            (m.rho[0] / 1.0e-9).to_bits()
        );
    }

    #[test]
    fn test_relax_tau_factor_steps_by_dlogtau_and_clamps_at_target() {
        let mut m = zams();
        m.controls.tau_factor = 1.0;
        m.job.relax_to_this_tau_factor = 100.0;
        m.job.dlogtau_factor = 0.5;
        relax_tau_factor(&mut m);
        assert_eq!(
            m.controls.tau_factor.to_bits(),
            exp10(safe_log10(1.0) + 0.5).to_bits()
        );
        m.job.dlogtau_factor = 10.0;
        relax_tau_factor(&mut m);
        assert_eq!(m.controls.tau_factor, 100.0);
    }

    #[test]
    fn test_near_zams_relax_with_zero_target_leaves_rotation_off() {
        // `near_zams_relax_omega_div_omega_crit .and.
        // new_omega_div_omega_crit /= 0d0` (`run_star_support.f90:1040-1041`).
        let mut m = zams();
        m.controls.rotation_flag = false;
        m.job.near_zams_relax_omega_div_omega_crit = true;
        m.job.new_omega_div_omega_crit = 0.0;
        let mut ex = crate::inlist_runner::DefaultStarExtras;
        do_rotation_near_zams(&mut m, &mut ex).expect("no-op");
        assert!(!m.controls.rotation_flag);
    }

    #[test]
    fn test_set_near_zams_omega_steps_switches_rotation_on_and_sets_limit() {
        let mut m = zams();
        m.controls.rotation_flag = false;
        m.model_number = 7;
        m.job.set_near_zams_omega_steps = 3;
        m.job.new_omega = 1.0e-5;
        let mut ex = crate::inlist_runner::DefaultStarExtras;
        do_rotation_near_zams(&mut m, &mut ex).expect("set limit");
        assert!(m.controls.rotation_flag);
        assert!(m.job.new_rotation_flag);
        assert_eq!(m.job.set_omega_step_limit, 9);
        assert!(m.omega.iter().all(|&w| w == 0.0));
        do_rotation(&mut m);
        let ok = m.omega.iter().all(|&w| (w - 1.0e-5).abs() <= 1.0e-19);
        assert!(
            ok,
            "do_rotation sets the uniform new_omega within the limit"
        );
    }
}
