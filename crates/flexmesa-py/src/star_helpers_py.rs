// SPDX-License-Identifier: LGPL-3.0-or-later
//! Internal implementation helpers for [`crate::star_py::PyStarModel`].
//!
//! Separates the multi-step marshalling, validation, and GIL-releasing logic
//! for ZAMS/pre-MS initialization, primary state synchronization, block
//! Jacobian reshaping, mixing/overshooting updates, stellar winds, and
//! history extraction from the `#[pymethods]` dispatch table in
//! [`crate::star_py`].

use flexmesa_star::{
    apply_overshooting, build_pre_ms_model, clean_mixing_glitches,
    compute_convection_sigmas, do_adjust_mass, do_struct_burn_mix,
    eval_henyey_equations, extract_history_row, load_zams_model,
    locate_convection_boundaries, set_mdot, solve_omega_mix, star_evolve,
    star_evolve_step, DutchLowTScheme, EvolveStepReport, HenyeySolverReport,
    OvershootConfig, OvershootScheme, StarControls, StarError, StarJobControls,
    WindScheme,
};
use numpy::{
    PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArray2,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::star_py::PyStarModel;

/// Return tuple for `eval_equations` (`(equ, lblk, dblk, ublk, x_scale)`).
pub type HenyeyEquationsPyTuple<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray2<f64>>,
);

/// Return tuple for `solve_henyey` and `step_struct_burn_mix`.
pub type SolverReportTuple = (bool, usize, f64, f64, f64, f64);

/// Return tuple for `evolve_step` and `evolve`.
pub type EvolveStepTuple =
    (usize, f64, f64, f64, usize, usize, usize, usize, f64, bool);

/// Return tuple for `history_tuple`.
pub type HistoryTuple = ((usize, usize, usize, usize), [f64; 13]);

/// Maps [`StarError`] to Python `ValueError`.
pub fn to_py_err(e: StarError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Converts [`HenyeySolverReport`] to a Python tuple.
pub const fn report_to_tuple(r: HenyeySolverReport) -> SolverReportTuple {
    (
        r.converged,
        r.iterations,
        r.residual_norm,
        r.max_residual,
        r.correction_norm,
        r.max_correction,
    )
}

/// Converts [`EvolveStepReport`] to a Python tuple.
pub const fn evolve_to_tuple(r: EvolveStepReport) -> EvolveStepTuple {
    (
        r.model_number,
        r.star_age,
        r.dt,
        r.dt_next,
        r.iterations,
        r.retries,
        r.num_splits,
        r.num_merges,
        r.varcontrol,
        r.converged,
    )
}

/// Builds top and bottom [`OvershootConfig`] entries when `f > 0`.
pub fn build_overshoot_configs(
    f: f64,
    f0: f64,
    scheme: &str,
) -> Vec<OvershootConfig> {
    if f <= 0.0 {
        return Vec::new();
    }
    let s = if scheme.eq_ignore_ascii_case("step") {
        OvershootScheme::Step
    } else {
        OvershootScheme::Exponential
    };
    let d = OvershootConfig {
        scheme: s,
        f,
        f0,
        ..OvershootConfig::default()
    };
    vec![
        OvershootConfig {
            top_conv_bdy: true,
            ..d
        },
        OvershootConfig {
            top_conv_bdy: false,
            ..d
        },
    ]
}

/// Parses a wind prescription string into [`WindScheme`].
pub fn parse_wind_scheme(scheme: &str) -> PyResult<WindScheme> {
    WindScheme::parse_name(scheme).ok_or_else(|| {
        PyValueError::new_err(format!("unknown wind scheme: {scheme}"))
    })
}

impl PyStarModel {
    /// Helper for `PyStarModel::from_zams`.
    pub(crate) fn build_zams(
        initial_mass: f64,
        initial_z: f64,
        initial_y: f64,
    ) -> PyResult<Self> {
        let c = StarControls {
            initial_mass,
            initial_z,
            initial_y,
            zbase: initial_z,
            ..StarControls::default()
        };
        let inner = load_zams_model(c, StarJobControls::default())
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Helper for `PyStarModel::from_pre_ms`.
    pub(crate) fn build_pre_ms(
        initial_mass: f64,
        initial_z: f64,
        initial_y: f64,
        t_c: f64,
        d_log10_p: f64,
    ) -> PyResult<Self> {
        let c = StarControls {
            initial_mass,
            initial_z,
            initial_y,
            zbase: initial_z,
            ..StarControls::default()
        };
        let j = StarJobControls {
            create_pre_ms_model: true,
            pre_ms_t_c: t_c,
            pre_ms_d_log10_p: d_log10_p,
            ..StarJobControls::default()
        };
        Ok(Self {
            inner: build_pre_ms_model(c, j).map_err(to_py_err)?,
        })
    }

    /// Helper for `PyStarModel::set_primary_state`.
    pub(crate) fn do_set_primary_state(
        &mut self,
        lnd: PyReadonlyArray1<'_, f64>,
        ln_t: PyReadonlyArray1<'_, f64>,
        ln_r: PyReadonlyArray1<'_, f64>,
        lum: PyReadonlyArray1<'_, f64>,
        dq: PyReadonlyArray1<'_, f64>,
        xa: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<()> {
        let (lnd_s, lnt_s, lnr_s, lum_s, dq_s, xa_s) = (
            lnd.as_slice()?,
            ln_t.as_slice()?,
            ln_r.as_slice()?,
            lum.as_slice()?,
            dq.as_slice()?,
            xa.as_slice()?,
        );
        let (nz, sp) = (lnd_s.len(), self.inner.species);
        if lnt_s.len() != nz
            || lnr_s.len() != nz
            || lum_s.len() != nz
            || dq_s.len() != nz
            || xa_s.len() != nz * sp
        {
            return Err(PyValueError::new_err(
                "mismatched array lengths in set_primary_state",
            ));
        }
        if nz != self.inner.nz && nz >= 2 {
            self.inner.resize_zones(nz);
        }
        self.inner.lnd.copy_from_slice(lnd_s);
        self.inner.ln_t.copy_from_slice(lnt_s);
        self.inner.ln_r.copy_from_slice(lnr_s);
        self.inner.lum.copy_from_slice(lum_s);
        self.inner.dq.copy_from_slice(dq_s);
        self.inner.xa.copy_from_slice(xa_s);
        self.inner.sync_mesh_and_pack().map_err(to_py_err)
    }

    /// Helper for `PyStarModel::eval_equations`.
    pub(crate) fn do_eval_equations<'py>(
        &mut self,
        py: Python<'py>,
        include_chem: bool,
    ) -> PyResult<HenyeyEquationsPyTuple<'py>> {
        let inner = &mut self.inner;
        let sys = py
            .detach(|| eval_henyey_equations(inner, include_chem))
            .map_err(to_py_err)?;
        let (nz, nv) = (sys.nz, sys.nvar);
        Ok((
            PyArray1::from_vec(py, sys.equ).reshape([nz, nv])?,
            PyArray1::from_vec(py, sys.lblk).reshape([nz, nv, nv])?,
            PyArray1::from_vec(py, sys.dblk).reshape([nz, nv, nv])?,
            PyArray1::from_vec(py, sys.ublk).reshape([nz, nv, nv])?,
            PyArray1::from_vec(py, sys.x_scale).reshape([nz, nv])?,
        ))
    }

    /// Helper for `PyStarModel::update_mixing`.
    pub(crate) fn do_update_mixing(
        &mut self,
        dt: f64,
        overshoot_f: f64,
        overshoot_f0: f64,
        scheme: &str,
    ) -> usize {
        clean_mixing_glitches(&mut self.inner, 1.0, 0.1);
        let (_cz_dq, bdy) = locate_convection_boundaries(&self.inner);
        let cfgs = build_overshoot_configs(overshoot_f, overshoot_f0, scheme);
        if !cfgs.is_empty() {
            apply_overshooting(&mut self.inner, &bdy, &cfgs);
        }
        compute_convection_sigmas(&mut self.inner, dt, 1.0, 1.0e10);
        bdy.len()
    }

    /// Helper for `PyStarModel::run_omega_mix`.
    pub(crate) fn do_run_omega_mix(
        &mut self,
        dt: f64,
        omega: PyReadonlyArray1<'_, f64>,
        nu_omega: PyReadonlyArray1<'_, f64>,
        n_substeps: usize,
    ) -> PyResult<()> {
        let (w_s, nu_s) = (omega.as_slice()?, nu_omega.as_slice()?);
        if w_s.len() != self.inner.nz || nu_s.len() != self.inner.nz {
            return Err(PyValueError::new_err(
                "omega and nu_omega must have length nz",
            ));
        }
        self.inner.omega.copy_from_slice(w_s);
        solve_omega_mix(&mut self.inner, dt, nu_s, n_substeps);
        Ok(())
    }

    /// Helper for `PyStarModel::step_struct_burn_mix`.
    pub(crate) fn do_step_struct_burn_mix(
        &mut self,
        py: Python<'_>,
        dt: f64,
        op_split_burn: bool,
        use_bcyclic: bool,
        overshoot_f: f64,
        overshoot_f0: f64,
    ) -> PyResult<SolverReportTuple> {
        let cfgs =
            build_overshoot_configs(overshoot_f, overshoot_f0, "exponential");
        let inner = &mut self.inner;
        let rep = py
            .detach(|| {
                do_struct_burn_mix(inner, dt, op_split_burn, use_bcyclic, &cfgs)
            })
            .map_err(to_py_err)?;
        Ok(report_to_tuple(rep))
    }

    /// Helper for `PyStarModel::apply_wind_and_adjust_mass`.
    pub(crate) fn do_apply_wind(
        &mut self,
        scheme: &str,
        eta: f64,
        dt: f64,
    ) -> PyResult<f64> {
        let ws = parse_wind_scheme(scheme)?;
        let mdot = set_mdot(&mut self.inner, ws, DutchLowTScheme::DeJager, eta);
        if dt > 0.0 && mdot != 0.0 {
            do_adjust_mass(&mut self.inner, mdot * dt, None, 0.99, 0.95)
                .map_err(to_py_err)?;
        }
        Ok(mdot)
    }

    /// Helper for `PyStarModel::evolve_step`.
    pub(crate) fn do_evolve_step(
        &mut self,
        py: Python<'_>,
        do_remesh: bool,
        op_split_burn: bool,
        use_bcyclic: bool,
    ) -> PyResult<EvolveStepTuple> {
        let inner = &mut self.inner;
        let r = py
            .detach(|| {
                star_evolve_step(inner, do_remesh, op_split_burn, use_bcyclic)
            })
            .map_err(to_py_err)?;
        Ok(evolve_to_tuple(r))
    }

    /// Helper for `PyStarModel::evolve`.
    pub(crate) fn do_evolve(
        &mut self,
        py: Python<'_>,
        max_steps: usize,
        do_remesh: bool,
        op_split_burn: bool,
        use_bcyclic: bool,
        max_age: Option<f64>,
    ) -> PyResult<Vec<EvolveStepTuple>> {
        let inner = &mut self.inner;
        if max_age.is_some() {
            inner.controls.max_age = max_age;
        }
        let reps = py
            .detach(|| {
                star_evolve(
                    inner,
                    max_steps,
                    do_remesh,
                    op_split_burn,
                    use_bcyclic,
                )
            })
            .map_err(to_py_err)?;
        Ok(reps.into_iter().map(evolve_to_tuple).collect())
    }

    /// Helper for `PyStarModel::history_tuple`.
    pub(crate) fn do_history_tuple(&self) -> HistoryTuple {
        let h = extract_history_row(&self.inner);
        (
            (
                h.model_number,
                h.num_zones,
                h.num_retries,
                h.total_num_solver_iterations,
            ),
            [
                h.star_age,
                h.star_mass,
                h.log_dt,
                h.log_lh,
                h.log_l,
                h.log_teff,
                h.log_r,
                h.log_g,
                h.log_center_t,
                h.log_center_rho,
                h.center_h1,
                h.center_he4,
                h.he_core_mass,
            ],
        )
    }
}
