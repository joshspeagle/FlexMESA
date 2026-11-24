// SPDX-License-Identifier: GPL-3.0-or-later
//! Routines from upstream `star/private/star_utils.f90` that act on the
//! whole structure at the start of a step.
//!
//! Corresponds to:
//! - `star/private/star_utils.f90:429-440` (`get_r_from_xh`)
//! - `star/private/star_utils.f90:575-587` (`store_rho_in_xh`)
//! - `star/private/star_utils.f90:606-632` (`use_xh_to_set_rho_to_dm_div_dV`)

use flexmesa_const::FOUR_THIRDS_PI;

use crate::controls::StarError;
use crate::model::StarModel;

/// Cell at which [`use_xh_to_set_rho_to_dm_div_dv`] found a non-positive
/// density.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonPositiveRho {
    /// Zero-based cell index (`k - 1` in upstream's numbering).
    pub k: usize,
    /// Density `dm / dV` in g cm^-3 that failed the `rho > 0` test.
    pub rho: f64,
}

/// Resets every cell's density to its mass divided by its volume.
///
/// The radii are the independent variables of the structure, so this makes
/// `ln rho` consistent with them and with the cell masses before the step
/// is solved. Upstream calls it from `prepare_for_new_step` whenever
/// `set_rho_to_dm_div_dV` is true and neither `u_flag` nor `RSP_flag` is
/// set (`star/private/evolve.f90:1854-1857`).
///
/// The loop runs from the centre outward in upstream's order
/// (`star/private/star_utils.f90:617-630`): `rL` is the previous face
/// radius, starting at `R_center`, `rR = exp(lnR(k))`,
/// `dV = four_thirds_pi*(rR*rR*rR - rL*rL*rL)`, and `lnd(k) = log(dm(k)/dV)`.
///
/// Args:
///     model: Star model whose `ln_r` (`ln(r/cm)`, `len = nz`), `dm` (g,
///         `len = nz`) and `r_center` (cm) are read, and whose `lnd`
///         (`ln(rho / g cm^-3)`, `len = nz`) is overwritten.
///
/// Returns:
///     `Ok(())` when every cell has `rho > 0`.
///
/// Errors:
///     [`NonPositiveRho`] at the first cell, counting from the centre, where
///     `dm / dV <= 0`. Upstream returns `ierr = -1` there, leaving the cells
///     already visited updated, and `prepare_for_new_step` terminates the
///     run with `result_reason = nonzero_ierr`.
pub fn use_xh_to_set_rho_to_dm_div_dv(
    model: &mut StarModel,
) -> Result<(), NonPositiveRho> {
    let nz = model.nz;
    let mut r_r = model.r_center;
    for k in (0..nz).rev() {
        let r_l = r_r;
        r_r = flexmesa_math::exp(model.ln_r[k]);
        let dm = model.dm[k];
        let dv = FOUR_THIRDS_PI * (r_r * r_r * r_r - r_l * r_l * r_l);
        let rho = dm / dv;
        if rho <= 0.0 {
            return Err(NonPositiveRho { k, rho });
        }
        model.lnd[k] = flexmesa_math::log(rho);
    }
    Ok(())
}

/// Applies [`use_xh_to_set_rho_to_dm_div_dv`] as `prepare_for_new_step`
/// does at the start of a new step (`star/private/evolve.f90:1854-1857`).
///
/// It does nothing unless `set_rho_to_dm_div_dV` is set and neither `u_flag`
/// nor `RSP_flag` is. Upstream calls it once per step, on the first try,
/// after the `dt` checks and before the remesh (`evolve.f90:1882-1889`) and
/// `new_generation` (`evolve.f90:1895`), so the state a retry or redo
/// restores already carries the reset density.
///
/// Args:
///     model: Star model at the start of a step, before remeshing.
///
/// Errors:
///     [`StarError::Terminated`] with a `nonzero_ierr` reason when a cell has
///     `dm / dV <= 0`: `prepare_for_new_step`'s `failed` returns `terminate`
///     with `result_reason = nonzero_ierr` (`evolve.f90:1946-1956`).
pub fn set_rho_to_dm_div_dv_for_new_step(
    model: &mut StarModel,
) -> Result<(), StarError> {
    if !model.controls.set_rho_to_dm_div_dv
        || model.controls.u_flag
        || model.controls.rsp_flag
    {
        return Ok(());
    }
    use_xh_to_set_rho_to_dm_div_dv(model).map_err(|e| StarError::Terminated {
        model_number: model.model_number,
        reason: format!(
            "nonzero_ierr: use_xh_to_set_rho_to_dm_div_dV: rho = {:e} <= 0 \
             at k = {}",
            e.rho,
            e.k + 1
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controls::{StarControls, StarJobControls};

    /// Three shells of uniform density `rho0` around a core of radius
    /// `r_center`, with `dm = rho0 * dV` (surface first, as upstream).
    fn uniform_shells(rho0: f64) -> StarModel {
        let mut model = StarModel::new(
            3,
            vec![2, 7],
            StarJobControls::default(),
            StarControls::default(),
        );
        model.r_center = 0.5;
        let r = [4.0_f64, 2.0, 1.0];
        let inner = [2.0_f64, 1.0, 0.5];
        model.ln_r = r.iter().map(|&x| flexmesa_math::log(x)).collect();
        model.dm = r
            .iter()
            .zip(inner)
            .map(|(&ro, ri)| {
                rho0 * FOUR_THIRDS_PI * (ro * ro * ro - ri * ri * ri)
            })
            .collect();
        model.lnd = vec![0.0; 3];
        model
    }

    #[test]
    fn uniform_density_shells_recover_ln_rho0_to_round_off() {
        let rho0 = 1.4_f64;
        let mut model = uniform_shells(rho0);
        use_xh_to_set_rho_to_dm_div_dv(&mut model).expect("rho > 0");
        let expected = flexmesa_math::log(rho0);
        for (k, lnd) in model.lnd.iter().enumerate() {
            assert!(
                (lnd - expected).abs() <= 1.0e-14,
                "cell {k}: lnd = {lnd}, expected ln(rho0) = {expected}"
            );
        }
    }

    #[test]
    fn non_positive_rho_stops_at_first_cell_from_centre_leaving_outer_cells() {
        let mut model = uniform_shells(3.0);
        model.dm[1] = -1.0;
        let err = use_xh_to_set_rho_to_dm_div_dv(&mut model)
            .expect_err("negative dm gives rho <= 0");
        assert_eq!(err.k, 1);
        assert!(err.rho < 0.0);
        // Upstream walks k = nz..1: the innermost cell is already updated,
        // the failing and outer cells are not.
        assert!((model.lnd[2] - flexmesa_math::log(3.0)).abs() <= 1.0e-14);
        assert_eq!(model.lnd[1], 0.0);
        assert_eq!(model.lnd[0], 0.0);
    }

    #[test]
    fn new_step_reset_respects_control_u_flag_and_rsp_flag() {
        for setup in [
            |m: &mut StarModel| m.controls.set_rho_to_dm_div_dv = false,
            |m: &mut StarModel| m.controls.u_flag = true,
            |m: &mut StarModel| m.controls.rsp_flag = true,
        ] {
            let mut model = uniform_shells(2.0);
            setup(&mut model);
            set_rho_to_dm_div_dv_for_new_step(&mut model).expect("skipped");
            assert_eq!(model.lnd, vec![0.0; 3]);
        }
        let mut model = uniform_shells(2.0);
        assert!(
            model.controls.set_rho_to_dm_div_dv,
            "MESA default is .true."
        );
        set_rho_to_dm_div_dv_for_new_step(&mut model).expect("rho > 0");
        assert!(model.lnd.iter().all(|&v| v != 0.0));
    }

    #[test]
    fn new_step_reset_failure_terminates_with_nonzero_ierr() {
        let mut model = uniform_shells(1.0);
        model.dm[2] = 0.0;
        let err = set_rho_to_dm_div_dv_for_new_step(&mut model)
            .expect_err("rho = 0 terminates");
        assert!(
            matches!(&err, StarError::Terminated { reason, .. }
                if reason.starts_with("nonzero_ierr")),
            "got {err:?}"
        );
    }
}
