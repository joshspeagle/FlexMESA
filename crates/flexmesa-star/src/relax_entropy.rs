// SPDX-License-Identifier: GPL-3.0-or-later
//! The entropy relax of `star/private/relax.f90` at MESA `26.4.1`:
//! `do_relax_entropy` (`relax.f90:694-809`), its hooks
//! (`relax.f90:811-917`) and the `other_energy` routine it installs
//! (`entropy_relax_other_energy`, `relax.f90:919-945`), together with the
//! reader of `do_relax_initial_entropy`
//! (`star/job/run_star_support.f90:2891-3009`).
//!
//! The relax adds the heating $\epsilon_k = (1 - S_k/S^{\mathrm{target}}_k)
//! E_k / (\tau\,\mathrm{secyer})$ to every cell and evolves with burning,
//! mixing and diffusion off, so the entropy profile approaches the target on
//! the timescale $\tau$ (`timescale_for_relax_entropy`).

use flexmesa_const::SECYER;
use flexmesa_interp1d::{interp_pm, interp_values, CubicCoeffs};

use crate::controls::StarError;
use crate::inlist_runner::StarExtras;
use crate::model::StarModel;
use crate::relax_driver::{
    do_internal_evolve, setup_before_relax, RelaxHookResult, RelaxHooks,
    RELAX_FINISHED_OKAY,
};

/// Termination code of a failed target interpolation. Upstream sets
/// `terminate` without a `termination_code` (`relax.f90:864`).
const ENTROPY_INTERP_FAILED: &str = "relax_entropy_interp_failed";

/// The target entropy profile that `store_rpar` keeps in `rpar`
/// (`star/private/relax.f90:782-808`): the knots `xq` and the
/// piecewise-monotonic cubic `interp_pm` of the target entropy.
#[derive(Debug, Clone)]
pub struct RelaxEntropyTarget {
    /// Exterior mass fraction $x_q$ of each knot (dimensionless, increasing).
    xq: Vec<f64>,
    /// Cubic coefficients of the target entropy
    /// ($k_B\,m_u^{-1}$, as stored in `lnS`) on each interval.
    coeffs: Vec<CubicCoeffs>,
}

impl RelaxEntropyTarget {
    /// `store_rpar` of `do_relax_entropy` (`star/private/relax.f90:782-808`).
    ///
    /// # Errors
    /// Returns [`StarError::InvalidInput`] when `interp_pm` rejects the knots
    /// (`ierr = op_err`, `relax.f90:803-804`).
    pub fn new(xq: &[f64], entropy: &[f64]) -> Result<Self, StarError> {
        let coeffs = interp_pm(xq, entropy).map_err(|e| {
            StarError::InvalidInput(format!(
                "do_relax_entropy: interp_pm failed \
                 (star/private/relax.f90:803): {e}"
            ))
        })?;
        Ok(Self {
            xq: xq.to_vec(),
            coeffs,
        })
    }

    /// Target entropy at the cell centers of `model`
    /// (`interp_values` at `relax.f90:895-898` and `relax.f90:932-935`).
    ///
    /// # Errors
    /// Returns [`StarError::InvalidInput`] when `interp_values` fails.
    fn at_cell_centers(
        &self,
        model: &StarModel,
    ) -> Result<Vec<f64>, StarError> {
        let nz = model.nz;
        let dq = &model.dq;
        let mut xq = Vec::with_capacity(nz);
        // `xq(1) = dq(1)/2`, `xq(k) = xq(k-1) + (dq(k) + dq(k-1))/2`
        // (`relax.f90:927-930`).
        xq.push(dq[0] / 2.0);
        for k in 1..nz {
            xq.push(xq[k - 1] + (dq[k] + dq[k - 1]) / 2.0);
        }
        interp_values(&self.xq, &self.coeffs, &xq).map_err(|e| {
            StarError::InvalidInput(format!(
                "relax_entropy: interp_values failed \
                 (star/private/relax.f90:935): {e}"
            ))
        })
    }

    /// `entropy_relax_other_energy` (`star/private/relax.f90:919-945`): sets
    /// `extra_heat` from the current `lnS` and `lnE`. Upstream assigns plain
    /// reals to the auto-differentiated `extra_heat`, so the partials of the
    /// heating with respect to $\ln\rho$ and $\ln T$ are zero.
    ///
    /// # Errors
    /// Returns [`StarError::InvalidInput`] when the target cannot be
    /// interpolated onto the model.
    pub(crate) fn apply_other_energy(
        &self,
        model: &mut StarModel,
    ) -> Result<(), StarError> {
        let vals = self.at_cell_centers(model)?;
        let timescale = model.job.timescale_for_relax_entropy;
        let nz = model.nz;
        if model.aux.extra_heat.len() != nz {
            model.aux.extra_heat.resize(nz, 0.0);
        }
        model.aux.d_extra_heat_dlnd.fill(0.0);
        model.aux.d_extra_heat_dlnt.fill(0.0);
        for k in 0..nz {
            model.aux.extra_heat[k] = (1.0
                - flexmesa_math::exp(model.ln_s[k]) / vals[k])
                * flexmesa_math::exp(model.ln_e[k]);
            model.aux.extra_heat[k] /= timescale * SECYER;
        }
        Ok(())
    }
}

/// Reads the entropy profile of `do_relax_initial_entropy` for
/// `get_entropy_for_relax_from_eos = ''`
/// (`star/job/run_star_support.f90:2918-3000`): `num_pts` on the first
/// record, then `num_pts` pairs `xq entropy`.
///
/// Fortran list-directed `read(iounit, *)` takes the first value of the first
/// record and drops the rest, reads a value pair from one record and drops
/// extra values, skips blank records and continues onto the next record when
/// a record is short; values are separated by blanks or commas.
///
/// # Errors
/// Returns [`StarError::InvalidInput`] when the first record is not an
/// integer or the file ends or holds a non-numeric value before `num_pts`
/// pairs are read (`ierr /= 0`, `run_star_support.f90:2924-2930, 2993-3000`).
pub fn parse_entropy_profile_str(
    text: &str,
) -> Result<(Vec<f64>, Vec<f64>), StarError> {
    let mut records = text
        .lines()
        .map(|l| {
            l.split(|c: char| c.is_whitespace() || c == ',')
                .filter(|t| !t.is_empty())
                .collect::<Vec<_>>()
        })
        .filter(|toks| !toks.is_empty());
    let first = records.next().ok_or_else(|| {
        StarError::InvalidInput(String::from(
            "failed while trying to read 1st line of the relax_entropy file \
             (star/job/run_star_support.f90:2924-2930)",
        ))
    })?;
    let num_pts: usize = first[0].parse().map_err(|_| {
        StarError::InvalidInput(String::from(
            "failed while trying to read 1st line of the relax_entropy file: \
             num_pts must be an integer (star/job/run_star_support.f90:2924)",
        ))
    })?;
    let mut xq = Vec::with_capacity(num_pts);
    let mut entropy = Vec::with_capacity(num_pts);
    let mut pending: Vec<&str> = Vec::new();
    for i in 0..num_pts {
        // One list-directed read of two values; leftover values of the last
        // record read are dropped.
        pending.clear();
        while pending.len() < 2 {
            let Some(rec) = records.next() else {
                return Err(StarError::InvalidInput(format!(
                    "failed while trying to read the relax_entropy file at \
                     line {} (perhaps wrong info in 1st line; the 1st line \
                     must have num_pts) (star/job/run_star_support.f90:\
                     2993-3000)",
                    i + 2
                )));
            };
            pending.extend(rec);
        }
        let parse = |tok: &str| {
            crate::zams::parse_fortran_f64(tok).map_err(|_| {
                StarError::InvalidInput(format!(
                    "failed while trying to read the relax_entropy file at \
                     line {}: '{tok}' is not a number \
                     (star/job/run_star_support.f90:2993-3000)",
                    i + 2
                ))
            })
        };
        xq.push(parse(pending[0])?);
        entropy.push(parse(pending[1])?);
    }
    Ok((xq, entropy))
}

/// `before_evolve_relax_entropy`, `relax_entropy_adjust_model` and
/// `relax_entropy_check_model` (`star/private/relax.f90:811-917`).
struct RelaxEntropy {
    target: RelaxEntropyTarget,
    /// `ipar(2)`: `max_steps_to_use + model_number + 1` at the start.
    max_model_number: usize,
}

impl RelaxHooks for RelaxEntropy {
    fn before_evolve(
        &mut self,
        model: &mut StarModel,
    ) -> Result<(), StarError> {
        setup_before_relax(model);
        Ok(())
    }

    fn check_model(
        &mut self,
        model: &mut StarModel,
    ) -> Result<RelaxHookResult, StarError> {
        // `do_bare_bones_check_model` only gathers history values.
        // `adjust_entropy` (`relax.f90:887-915`) interpolates the target
        // onto the cells; its average relative error is only printed
        // (`relax.f90:866-868`), which `FlexMESA` does not write
        // (`DECISIONS #26`). A failed interpolation sets `terminate`, which
        // the two checks below overwrite when they apply.
        let mut result = RelaxHookResult::KeepGoing;
        if self.target.at_cell_centers(model).is_err() {
            result =
                RelaxHookResult::Terminate(String::from(ENTROPY_INTERP_FAILED));
        }
        let job = &model.job;
        if model.star_age
            >= job.timescale_for_relax_entropy
                * job.num_timescales_for_relax_entropy
            || model.model_number >= self.max_model_number
        {
            result =
                RelaxHookResult::Terminate(String::from(RELAX_FINISHED_OKAY));
        }
        Ok(result)
    }
}

/// `do_relax_entropy` (`star/private/relax.f90:694-809`): relaxes the
/// entropy of `model` toward `target` for at most `max_steps_to_use` steps
/// or until `star_age` reaches `timescale_for_relax_entropy *
/// num_timescales_for_relax_entropy` years, with composition change by
/// burning, mixing and element diffusion off and the timestep limited to
/// `max_dt_for_relax_entropy` years. The controls, the clock and
/// `other_energy` are restored afterwards.
///
/// # Errors
/// Returns [`StarError`] when the relax fails.
pub(crate) fn do_relax_entropy(
    model: &mut StarModel,
    max_steps_to_use: usize,
    target: RelaxEntropyTarget,
    extras: &mut dyn StarExtras,
) -> Result<(), StarError> {
    let max_model_number = model.controls.max_model_number;
    let new_max_model_number = max_steps_to_use + model.model_number + 1;
    model.controls.max_model_number = Some(new_max_model_number);

    let c = &mut model.controls;
    let dxdt_nuc_factor = c.dxdt_nuc_factor;
    c.dxdt_nuc_factor = 0.0;
    let mix_factor = c.mix_factor;
    c.mix_factor = 0.0;
    let do_element_diffusion = c.do_element_diffusion;
    c.do_element_diffusion = false;
    let starting_dt_next = model.dt_next;
    let max_years_for_timestep = c.max_years_for_timestep;
    let max_dt = model.job.max_dt_for_relax_entropy;
    c.max_years_for_timestep = max_dt;
    model.dt_next = model.dt_next.min(max_dt * SECYER);
    let use_other_energy = c.use_other_energy;
    c.use_other_energy = true;
    // `s% other_energy => entropy_relax_other_energy`
    // (`relax.f90:761`), applied by `apply_extra_heat_controls`.
    let other_energy = model.aux.relax_entropy.replace(target.clone());
    let time = model.time_sec;
    let hist_time = model.aux.step_hist.time_sec;
    model.time_sec = 0.0;
    model.aux.step_hist.time_sec = 0.0;

    let mut hooks = RelaxEntropy {
        target,
        max_model_number: new_max_model_number,
    };
    let res = do_internal_evolve(model, &mut hooks, true, extras);

    model.controls.max_model_number = max_model_number;
    model.dt_next = starting_dt_next;
    let c = &mut model.controls;
    c.dxdt_nuc_factor = dxdt_nuc_factor;
    c.mix_factor = mix_factor;
    c.do_element_diffusion = do_element_diffusion;
    c.max_years_for_timestep = max_years_for_timestep;
    c.use_other_energy = use_other_energy;
    model.aux.relax_entropy = other_energy;
    model.time_sec = time;
    model.aux.step_hist.time_sec = hist_time;
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_reads_num_pts_then_pairs_ignoring_trailing_values() {
        // The first record's trailing values and the third value of a row
        // are dropped by list-directed reads.
        let text = "3 ignored\n0.0 1.5 9.9\n0.5 2.5\n1.0 3.5\n";
        let (xq, s) = parse_entropy_profile_str(text).unwrap();
        assert_eq!(xq, vec![0.0, 0.5, 1.0]);
        assert_eq!(s, vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn parse_accepts_commas_blank_records_and_split_rows() {
        let text = "2\n\n0.25d0, 4.0\n\n0.75\n 8.0\n";
        let (xq, s) = parse_entropy_profile_str(text).unwrap();
        assert_eq!(xq, vec![0.25, 0.75]);
        assert_eq!(s, vec![4.0, 8.0]);
    }

    #[test]
    fn parse_rejects_truncated_file_and_bad_header_and_bad_number() {
        assert!(parse_entropy_profile_str("").is_err());
        assert!(parse_entropy_profile_str("two\n0 1\n").is_err());
        assert!(parse_entropy_profile_str("3\n0 1\n0.5 2\n").is_err());
        assert!(parse_entropy_profile_str("1\n0 abc\n").is_err());
    }

    #[test]
    fn target_clamps_outside_knots_and_reproduces_knot_values() {
        let xq = [0.0, 0.5, 1.0];
        let s = [1.0, 2.0, 4.0];
        let t = RelaxEntropyTarget::new(&xq, &s).unwrap();
        let v = interp_values(&t.xq, &t.coeffs, &[-1.0, 0.0, 0.5, 1.0, 2.0])
            .unwrap();
        assert_eq!(v, vec![1.0, 1.0, 2.0, 4.0, 4.0]);
    }

    #[test]
    fn target_rejects_non_monotonic_knots() {
        assert!(RelaxEntropyTarget::new(&[0.0, 1.0, 0.5], &[1.0, 2.0, 3.0])
            .is_err());
    }

    /// A 1 Msun ZAMS model and a target entropy profile equal to
    /// `factor * exp(lnS)` at its cell centers (`xq` as in
    /// `entropy_relax_other_energy`, `relax.f90:927-930`).
    fn zams_and_scaled_target(factor: f64) -> (StarModel, RelaxEntropyTarget) {
        let model = crate::zams::load_zams_model(
            crate::controls::StarControls {
                zbase: 0.02,
                ..crate::controls::StarControls::default()
            },
            crate::controls::StarJobControls::default(),
        )
        .expect("1 Msun ZAMS should load");
        let mut xq = vec![model.dq[0] / 2.0];
        for k in 1..model.nz {
            xq.push(xq[k - 1] + (model.dq[k] + model.dq[k - 1]) / 2.0);
        }
        let s: Vec<f64> = model
            .ln_s
            .iter()
            .map(|&ln_s| factor * flexmesa_math::exp(ln_s))
            .collect();
        let target = RelaxEntropyTarget::new(&xq, &s).expect("target");
        (model, target)
    }

    /// Mean over the cells of `model` of $|S_k/S^{\mathrm{target}}_k - 1|$,
    /// the quantity `relax_entropy_check_model` prints as `avg rel err`
    /// (`relax.f90:867, 912-913`). The mesh changes during the relax, so the
    /// target is evaluated on the cells of `model` each time.
    fn mean_relative_error(
        model: &StarModel,
        target: &RelaxEntropyTarget,
    ) -> f64 {
        let vals = target.at_cell_centers(model).expect("target at cells");
        (0..model.nz)
            .map(|k| (flexmesa_math::exp(model.ln_s[k]) / vals[k] - 1.0).abs())
            .sum::<f64>()
            / model.nz as f64
    }

    #[test]
    fn other_energy_is_zero_when_target_equals_current_entropy() {
        let (mut model, target) = zams_and_scaled_target(1.0);
        target.apply_other_energy(&mut model).unwrap();
        // `interp_values` returns each knot value exactly, so
        // `1 - S/S_target = 0` (`relax.f90:940`).
        assert!(model.aux.extra_heat.iter().all(|&h| h == 0.0));
        assert!(model.aux.d_extra_heat_dlnd.iter().all(|&h| h == 0.0));
        assert!(model.aux.d_extra_heat_dlnt.iter().all(|&h| h == 0.0));
    }

    #[test]
    fn other_energy_matches_closed_form_heating_for_scaled_target() {
        // S_target = 1.02 S gives eps = (1 - 1/1.02) E / (tau secyer).
        let (mut model, target) = zams_and_scaled_target(1.02);
        model.job.timescale_for_relax_entropy = 3.0e-7;
        target.apply_other_energy(&mut model).unwrap();
        for k in 0..model.nz {
            let want = (1.0 - 1.0 / 1.02) * flexmesa_math::exp(model.ln_e[k])
                / (3.0e-7 * SECYER);
            let got = model.aux.extra_heat[k];
            assert!(
                (got - want).abs() <= 1.0e-12 * want.abs(),
                "k={k}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn relax_entropy_heats_toward_target_and_restores_state() {
        let factor = 1.01;
        let (mut model, target) = zams_and_scaled_target(factor);
        let err0 = mean_relative_error(&model, &target);
        let target_copy = target.clone();
        model.job.timescale_for_relax_entropy = 1.0e-9;
        model.job.max_dt_for_relax_entropy = 1.0e-9;
        model.job.num_timescales_for_relax_entropy = 100.0;
        let before = (
            model.controls.max_model_number,
            model.controls.dxdt_nuc_factor,
            model.controls.mix_factor,
            model.controls.do_element_diffusion,
            model.controls.max_years_for_timestep,
            model.controls.use_other_energy,
            model.dt_next,
            model.time_sec,
            model.model_number,
        );
        do_relax_entropy(
            &mut model,
            3,
            target,
            &mut crate::inlist_runner::DefaultStarExtras,
        )
        .expect("relax entropy");
        let after = (
            model.controls.max_model_number,
            model.controls.dxdt_nuc_factor,
            model.controls.mix_factor,
            model.controls.do_element_diffusion,
            model.controls.max_years_for_timestep,
            model.controls.use_other_energy,
            model.dt_next,
            model.time_sec,
            model.model_number,
        );
        // `do_internal_evolve` restores `model_number`, the rest is restored
        // by `do_relax_entropy` (`relax.f90:788-799`).
        assert_eq!(before, after);
        assert!(model.aux.relax_entropy.is_none());
        let err1 = mean_relative_error(&model, &target_copy);
        assert!(err1 < err0, "entropy moved away: {err0} -> {err1}");
    }
}
