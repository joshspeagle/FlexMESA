// SPDX-License-Identifier: LGPL-3.0-or-later
//! `PyO3` callback adapters and RSP/GYRE helpers bridging Python test-side
//! extras (`run_star_extras` / `run_binary_extras` implementations) into
//! `StarExtras` and `BinaryExtras` (`#148`).

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use flexmesa_binary::{
    separation_from_period, BinaryCompanion, BinaryExtras, BinaryExtrasAction,
    BinarySystem, DefaultBinaryExtras,
};
use flexmesa_const::{CLIGHT, CRAD, LSUN, M_SUN, PI, RSUN, SECDAY};
use flexmesa_star::inlist_runner::{
    DefaultStarExtras, ExtrasDirective, RspBuildOutcome, StarExtras,
};
use flexmesa_star::{
    center_avg_x_by_name, compute_core_masses, compute_power_categories,
    do_rsp_build_model, eval_photosphere_info, export_gyre,
    parse_gyre_mesa_model, safe_log10, solve_bvp_eigenmodes, InertiaNormKind,
    InlistBundle, OuterBoundKind, RspChemCache, StarControls, StarError,
    StarModel,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::star_helpers_py::to_py_err;
use crate::star_py::PyStarModel;

/// Context passed into Python `other_rsp_build_model` hooks so test suites can
/// build static `RSP` envelopes and run linear `LINA` instability scans using
/// parsed inlist controls without embedding case logic in Rust (`#148`).
#[pyclass(module = "flexmesa._rust")]
pub struct PyRspBuildContext {
    bundle: InlistBundle,
}

impl PyRspBuildContext {
    /// Constructs a [`PyRspBuildContext`] from an already-loaded [`InlistBundle`].
    #[must_use]
    pub fn from_bundle(bundle: &InlistBundle) -> Self {
        Self {
            bundle: bundle.clone(),
        }
    }
}

#[pymethods]
impl PyRspBuildContext {
    /// Staged case directory path for the active stage.
    #[getter]
    pub fn case_dir(&self) -> String {
        self.bundle.case_dir.to_string_lossy().into_owned()
    }

    /// Envelope total mass (`RSP_mass`, $M_\odot$).
    #[getter]
    pub const fn rsp_mass(&self) -> f64 {
        self.bundle.controls.rsp_mass
    }

    /// Envelope hydrogen mass fraction (`RSP_X`).
    #[getter]
    pub const fn rsp_x(&self) -> f64 {
        self.bundle.controls.rsp_x
    }

    /// Envelope metallicity mass fraction (`RSP_Z`).
    #[getter]
    pub const fn rsp_z(&self) -> f64 {
        self.bundle.controls.rsp_z
    }

    /// Turbulent convective mixing-length parameter `RSP_alfam`.
    #[getter]
    pub const fn rsp_alfam(&self) -> f64 {
        self.bundle.controls.rsp.alfam
    }

    /// Eddy-viscosity parameter `RSP_alfap`.
    #[getter]
    pub const fn rsp_alfap(&self) -> f64 {
        self.bundle.controls.rsp.alfap
    }

    /// Turbulent-flux parameter `RSP_alfat`.
    #[getter]
    pub const fn rsp_alfat(&self) -> f64 {
        self.bundle.controls.rsp.alfat
    }

    /// Radiative-damping parameter `RSP_gammar`.
    #[getter]
    pub const fn rsp_gammar(&self) -> f64 {
        self.bundle.controls.rsp.gammar
    }

    /// Floating-point user control array `x_ctrl`.
    #[getter]
    pub fn x_ctrl(&self) -> Vec<f64> {
        self.bundle.controls.x_ctrl.clone()
    }

    /// Integer user control array `x_integer_ctrl`.
    #[getter]
    pub fn x_integer_ctrl(&self) -> Vec<i64> {
        self.bundle.controls.x_integer_ctrl.clone()
    }

    /// Logical user control array `x_logical_ctrl`.
    #[getter]
    pub fn x_logical_ctrl(&self) -> Vec<bool> {
        self.bundle.controls.x_logical_ctrl.clone()
    }

    /// String user control array `x_character_ctrl`.
    #[getter]
    pub fn x_character_ctrl(&self) -> Vec<String> {
        self.bundle.controls.x_character_ctrl.clone()
    }

    /// Evaluates linear non-adiabatic (`LINA`) periods (days) and growth rates
    /// at `(teff_k, l_lsun)` using the staged bundle's `RSP` parameters.
    #[pyo3(signature = (teff_k, l_lsun, n_modes = 3))]
    pub fn eval_lina(
        &self,
        py: Python<'_>,
        teff_k: f64,
        l_lsun: f64,
        n_modes: usize,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let mut ctrl = self.bundle.controls.clone();
        let mut job = self.bundle.job.clone();
        ctrl.rsp_teff = teff_k;
        ctrl.rsp_l = l_lsun;
        ctrl.rsp.nmodes = n_modes.clamp(1, 15);
        ctrl.rsp.use_other_rsp_build_model = false;
        ctrl.rsp.use_other_rsp_linear_analysis = false;
        ctrl.rsp.use_rsp_new_start_scheme = false;
        job.create_rsp_model = true;
        let eos_cfg = self.bundle.eos_config;
        let kap_cfg = self.bundle.kap_config.clone();

        let (_model, lina_opt) = py
            .detach(|| do_rsp_build_model(ctrl, job, eos_cfg, kap_cfg, None))
            .map_err(to_py_err)?;
        let lina = lina_opt.ok_or_else(|| {
            PyValueError::new_err("do_LINA returned no linear mode output")
        })?;
        let periods_days: Vec<f64> =
            lina.periods_sec.iter().map(|&p| p / SECDAY).collect();
        Ok((periods_days, lina.growth_rates))
    }
}

/// Solves adiabatic `GYRE` MESA modes and returns eigenfunction radial profiles:
/// `(l, n_p, n_g, n_pg, freq_uhz, x_grid, xi_r)` per solved eigenmode.
#[pyfunction]
#[pyo3(signature = (gyre_content, l = 0, omega_min = 0.5, omega_max = 12.0, n_freq = 400))]
#[allow(clippy::type_complexity)]
pub fn star_solve_gyre_mesa_modes_full(
    py: Python<'_>,
    gyre_content: &str,
    l: usize,
    omega_min: f64,
    omega_max: f64,
    n_freq: usize,
) -> PyResult<Vec<(usize, i32, i32, i32, f64, Py<PyList>, Vec<f64>)>> {
    let (x_grid, raw_modes) = py.detach(|| {
        let bvp = parse_gyre_mesa_model(gyre_content);
        let x_grid: Vec<f64> = bvp.pts.iter().map(|pt| pt.x).collect();
        let n = n_freq.max(2);
        let grid: Vec<f64> = (0..n)
            .map(|i| {
                omega_min
                    + (omega_max - omega_min) * (i as f64) / ((n - 1) as f64)
            })
            .collect();
        let modes = solve_bvp_eigenmodes(
            &bvp,
            l,
            &grid,
            OuterBoundKind::Vacuum,
            InertiaNormKind::Both,
        );
        let solved: Vec<(usize, i32, i32, i32, f64, Vec<f64>)> = modes
            .into_iter()
            .map(|m| {
                let xi_r: Vec<f64> = x_grid
                    .iter()
                    .zip(m.y.iter())
                    .map(|(&x, y)| {
                        let scale = match m.l {
                            0 if x == 0.0 => 0.0,
                            0 => 1.0 / x,
                            1 => 1.0,
                            _ => flexmesa_math::powi_f90(x, (m.l as i32) - 1),
                        };
                        scale * y[0]
                    })
                    .collect();
                (m.l, m.n_p, m.n_g, m.n_pg, m.freq_uhz, xi_r)
            })
            .collect();
        (x_grid, solved)
    });
    let shared_x_grid = PyList::new(py, &x_grid)?.unbind();
    Ok(raw_modes
        .into_iter()
        .map(|(m_l, n_p, n_g, n_pg, freq_uhz, xi_r)| {
            (
                m_l,
                n_p,
                n_g,
                n_pg,
                freq_uhz,
                shared_x_grid.clone_ref(py),
                xi_r,
            )
        })
        .collect())
}

fn format_py_err(py: Python<'_>, err: &PyErr) -> String {
    err.value(py)
        .str()
        .map_or_else(|_| err.to_string(), |s| s.to_string_lossy().into_owned())
}

fn lock_unpoisoned<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

#[derive(Default)]
struct PyAdapterState {
    staged_case_dir: Mutex<Option<(PathBuf, String)>>,
    pending_err: Mutex<Option<String>>,
}

impl PyAdapterState {
    fn set_staged_case_dir(&self, case_dir: &Path) {
        let s = case_dir.to_string_lossy().into_owned();
        *lock_unpoisoned(&self.staged_case_dir) =
            Some((case_dir.to_path_buf(), s));
    }

    fn staged_dir_string(&self) -> String {
        lock_unpoisoned(&self.staged_case_dir)
            .as_ref()
            .map_or_else(String::new, |(_, s)| s.clone())
    }

    fn record_py_err(&self, context: &str, err: String) {
        let mut slot = lock_unpoisoned(&self.pending_err);
        if slot.is_none() {
            *slot = Some(format!("{context}: {err}"));
        }
    }

    fn take_pending_msg(&self) -> Option<String> {
        lock_unpoisoned(&self.pending_err).take()
    }

    fn take_pending_py_err(&self) -> PyResult<()> {
        if let Some(msg) = self.take_pending_msg() {
            return Err(PyRuntimeError::new_err(msg));
        }
        Ok(())
    }

    fn take_pending_star_err(&self) -> Result<(), StarError> {
        if let Some(msg) = self.take_pending_msg() {
            return Err(StarError::InvalidInput(msg));
        }
        Ok(())
    }
}

fn set_user_controls(
    d: &Bound<'_, PyDict>,
    controls: &StarControls,
) -> PyResult<()> {
    let py = d.py();
    d.set_item("x_ctrl", PyList::new(py, &controls.x_ctrl)?)?;
    d.set_item("x_integer_ctrl", PyList::new(py, &controls.x_integer_ctrl)?)?;
    d.set_item("x_logical_ctrl", PyList::new(py, &controls.x_logical_ctrl)?)?;
    d.set_item(
        "x_character_ctrl",
        PyList::new(py, &controls.x_character_ctrl)?,
    )?;
    Ok(())
}

/// Surface-cell mass fraction `s% xa(net_iso(i<iso>), 1)`, or `None` when the
/// isotope is not in the net or the model has no cells. Callers leave the
/// snapshot key out in that case so a hook that reads it raises instead of
/// seeing a substituted value.
fn surface_iso_frac(model: &StarModel, iso_name: &str) -> Option<f64> {
    if model.nz == 0 {
        return None;
    }
    model
        .iso_names
        .iter()
        .position(|name| name.eq_ignore_ascii_case(iso_name))
        .and_then(|j| model.cell_xa(0).get(j).copied())
}

fn build_star_snapshot<'py>(
    py: Python<'py>,
    model: &StarModel,
    case_dir: &str,
    include_history: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("model_number", model.model_number)?;
    d.set_item("star_age", model.star_age)?;
    d.set_item("star_age_old", model.star_age_old)?;
    d.set_item("dt", model.dt)?;
    d.set_item("dt_next", model.dt_next)?;
    d.set_item("mstar", model.mstar)?;
    d.set_item("star_mass", model.mstar / M_SUN)?;
    d.set_item("teff", model.teff)?;
    d.set_item("surface_l_lsun", model.surface_l_lsun())?;
    d.set_item("surface_r_rsun", model.surface_r_rsun())?;
    d.set_item("total_energy", model.aux.total_energy)?;
    d.set_item("cumulative_energy_error", model.aux.cumulative_energy_error)?;
    d.set_item("nz", model.nz)?;
    d.set_item("case_dir", case_dir)?;
    d.set_item("rsp_flag", model.controls.rsp_flag)?;
    set_user_controls(&d, &model.controls)?;
    if model.controls.rsp_flag {
        d.set_item("rsp_num_periods", model.aux.rsp.num_periods)?;
        d.set_item("rsp_period", model.aux.rsp.period)?;
        d.set_item("rsp_period_lin", model.aux.rsp.period_lin)?;
        d.set_item("rsp_grekm", model.aux.rsp.grekm)?;
        d.set_item("rsp_grekm_avg_abs", model.aux.rsp.grekm_avg_abs)?;
        d.set_item("rsp_delta_r", model.aux.rsp.delta_r)?;
        d.set_item("rsp_delta_mag", model.aux.rsp.delta_mag)?;
        d.set_item(
            "rsp_lina_periods",
            PyList::new(py, model.aux.rsp.lina_periods)?,
        )?;
        d.set_item(
            "rsp_lina_growth_rates",
            PyList::new(py, model.aux.rsp.lina_growth_rates)?,
        )?;
        d.set_item("rsp_mass", model.controls.rsp_mass)?;
        d.set_item("rsp_l", model.controls.rsp_l)?;
        d.set_item("rsp_teff", model.controls.rsp_teff)?;
        d.set_item("rsp_nmodes", model.controls.rsp.nmodes)?;
    }
    if model.controls.use_other_wind && model.nz > 0 {
        if let Some(x) = surface_iso_frac(model, "h1") {
            d.set_item("surface_h1", x)?;
        }
        if let Some(y) = surface_iso_frac(model, "he4") {
            d.set_item("surface_he4", y)?;
        }
        d.set_item("zbase", model.controls.zbase)?;
    }
    if include_history {
        let cores = compute_core_masses(model);
        d.set_item("center_h1", center_avg_x_by_name(model, "h1"))?;
        d.set_item("center_he4", center_avg_x_by_name(model, "he4"))?;
        d.set_item("center_c12", center_avg_x_by_name(model, "c12"))?;
        d.set_item("he_core_mass", cores.he_core_mass)?;
        d.set_item("co_core_mass", cores.co_core_mass)?;
        d.set_item("log_dt", safe_log10(model.dt / flexmesa_const::SECYER))?;
        d.set_item("log_l", safe_log10(model.surface_l_lsun()))?;
        d.set_item("log_teff", safe_log10(model.teff))?;
        d.set_item("log_r", safe_log10(model.surface_r_rsun()))?;
    }
    Ok(d)
}

fn apply_star_updates(
    model: &mut StarModel,
    updates: &Bound<'_, PyDict>,
) -> PyResult<()> {
    if let Some(val) = updates.get_item("x_ctrl")? {
        let xs: Vec<f64> = val.extract()?;
        let n = xs.len().min(model.controls.x_ctrl.len());
        model.controls.x_ctrl[..n].copy_from_slice(&xs[..n]);
    }
    if let Some(val) = updates.get_item("x_integer_ctrl")? {
        let xs: Vec<i64> = val.extract()?;
        let n = xs.len().min(model.controls.x_integer_ctrl.len());
        model.controls.x_integer_ctrl[..n].copy_from_slice(&xs[..n]);
    }
    if let Some(val) = updates.get_item("x_logical_ctrl")? {
        let xs: Vec<bool> = val.extract()?;
        let n = xs.len().min(model.controls.x_logical_ctrl.len());
        model.controls.x_logical_ctrl[..n].copy_from_slice(&xs[..n]);
    }
    if let Some(val) = updates.get_item("x_character_ctrl")? {
        let xs: Vec<String> = val.extract()?;
        let n = xs.len().min(model.controls.x_character_ctrl.len());
        for (dst, src) in
            model.controls.x_character_ctrl[..n].iter_mut().zip(xs)
        {
            *dst = src;
        }
    }
    if let Some(val) = updates.get_item("dt_next")? {
        model.dt_next = val.extract::<f64>()?;
    }
    Ok(())
}

fn extract_rsp_vec3(
    d: &Bound<'_, PyDict>,
    key: &str,
    dst: &mut [f64; 3],
) -> PyResult<()> {
    if let Some(obj) = d.get_item(key)? {
        let vals: Vec<f64> = obj.extract()?;
        for (slot, val) in dst.iter_mut().zip(vals) {
            *slot = val;
        }
    }
    Ok(())
}

fn apply_rsp_linear_analysis_result(
    d: &Bound<'_, PyDict>,
    model: &mut StarModel,
) -> PyResult<bool> {
    let handled = d
        .get_item("handled")?
        .map_or(Ok(true), |v| v.extract::<bool>())?;
    if !handled {
        return Ok(false);
    }
    if let Some(v_obj) = d.get_item("v")? {
        let v_vec: Vec<f64> = v_obj.extract()?;
        if v_vec.len() != model.nz {
            return Err(PyValueError::new_err(format!(
                "other_rsp_linear_analysis returned v len {} != nz {}",
                v_vec.len(),
                model.nz
            )));
        }
        model.v = v_vec;
        if let Some(i_v) = model.layout.i_v {
            let nv = model.layout.nvar_hydro;
            for k in 0..model.nz {
                model.xh[k * nv + i_v] = model.v[k];
            }
        }
    }
    if let Some(vc) = d.get_item("v_center")? {
        model.v_center = vc.extract::<f64>()?;
    }
    if let Some(p) = d.get_item("rsp_period")? {
        let period_sec = p.extract::<f64>()?;
        model.aux.rsp.period = period_sec;
        model.aux.rsp.period_lin = period_sec;
    }
    extract_rsp_vec3(d, "lina_periods", &mut model.aux.rsp.lina_periods)?;
    extract_rsp_vec3(
        d,
        "lina_growth_rates",
        &mut model.aux.rsp.lina_growth_rates,
    )?;
    model.aux.rsp.have_set_velocities = true;
    Ok(true)
}

/// Converts a star hook's return value into an [`ExtrasDirective`].
///
/// A hook cannot choose its termination code: upstream sets it from the hook
/// that returned `terminate` (`star/job/standard_run_star_extras.inc:83`
/// `t_extras_check_model`, `:228` `t_extras_finish_step`; strings at
/// `star/private/star_private_def.f90:212-213`). `default_terminate_code` is
/// that hook's upstream name and is always the code. A message supplied with
/// `terminate` is diagnostic only (upstream hooks `write(*,*)` theirs) and
/// goes to stderr. An unrecognised directive string is an error.
fn parse_star_directive(
    res: &Bound<'_, PyAny>,
    default_terminate_code: &str,
    model: &mut StarModel,
) -> PyResult<ExtrasDirective> {
    if res.is_none() {
        return Ok(ExtrasDirective::KeepGoing);
    }
    let mut dict_msg: Option<String> = None;
    let directive_str: String = if let Ok(tup) = res.cast::<PyTuple>() {
        if tup.len() != 2 {
            return Err(PyValueError::new_err(
                "expected 2-tuple from star extras hook",
            ));
        }
        let second = tup.get_item(1)?;
        if let Ok(updates) = second.cast::<PyDict>() {
            apply_star_updates(model, updates)?;
            tup.get_item(0)?.extract::<String>()?
        } else {
            let first: String = tup.get_item(0)?.extract()?;
            let msg: String = second.extract()?;
            match first.as_str() {
                "terminate" => dict_msg = Some(msg),
                "error" => return Ok(ExtrasDirective::Error(msg)),
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "unsupported star directive tuple tag '{first}' \
                         from {default_terminate_code}"
                    )));
                }
            }
            first
        }
    } else if let Ok(d) = res.cast::<PyDict>() {
        if let Some(updates) = d.get_item("updates")? {
            if let Ok(udict) = updates.cast::<PyDict>() {
                apply_star_updates(model, udict)?;
            }
        }
        if let Some(msg_obj) = d.get_item("message")? {
            if !msg_obj.is_none() {
                dict_msg = Some(msg_obj.extract::<String>()?);
            }
        }
        if let Some(act) = d.get_item("action")? {
            act.extract::<String>()?
        } else {
            String::from("keep_going")
        }
    } else {
        res.extract::<String>()?
    };

    match directive_str.as_str() {
        "keep_going" => Ok(ExtrasDirective::KeepGoing),
        "redo" => Ok(ExtrasDirective::Redo),
        "retry" => Ok(ExtrasDirective::Retry),
        "error" => Ok(ExtrasDirective::Error(dict_msg.unwrap_or_else(|| {
            String::from("error returned by star extras hook")
        }))),
        "terminate" => {
            if let Some(msg) = dict_msg {
                eprintln!("{default_terminate_code}: {msg}");
            }
            Ok(ExtrasDirective::Terminate(
                default_terminate_code.to_string(),
            ))
        }
        other => Err(PyValueError::new_err(format!(
            "unrecognised directive '{other}' returned by \
             {default_terminate_code}"
        ))),
    }
}

/// Thread-safe adapter forwarding [`StarExtras`] trait callbacks into a Python
/// test-side extras object while avoiding per-step whole-model clones (`#148`).
#[allow(clippy::struct_excessive_bools)]
pub struct PyStarExtrasAdapter {
    py_obj: Py<PyAny>,
    has_controls: bool,
    has_startup: bool,
    has_start_step: bool,
    has_check_model: bool,
    has_finish_step: bool,
    has_after_evolve: bool,
    has_other_cgrav: bool,
    has_other_alpha_mlt: bool,
    has_other_wind: bool,
    has_other_adjust_mdot: bool,
    has_other_torque: bool,
    has_other_energy: bool,
    has_other_build_initial_model: bool,
    has_other_rsp_build_model: bool,
    has_other_rsp_linear_analysis: bool,
    has_extra_history_columns: bool,
    state: PyAdapterState,
}

impl PyStarExtrasAdapter {
    /// Wraps a Python star extras object, caching available hook flags under the GIL.
    ///
    /// # Errors
    /// Returns [`PyErr`] if attribute inspection fails.
    pub fn new(py: Python<'_>, obj: Py<PyAny>) -> PyResult<Self> {
        let b = obj.bind(py);
        Ok(Self {
            has_controls: b.hasattr("extras_controls")?,
            has_startup: b.hasattr("extras_startup")?,
            has_start_step: b.hasattr("extras_start_step")?,
            has_check_model: b.hasattr("extras_check_model")?,
            has_finish_step: b.hasattr("extras_finish_step")?,
            has_after_evolve: b.hasattr("extras_after_evolve")?,
            has_other_cgrav: b.hasattr("other_cgrav")?,
            has_other_alpha_mlt: b.hasattr("other_alpha_mlt")?,
            has_other_wind: b.hasattr("other_wind")?,
            has_other_adjust_mdot: b.hasattr("other_adjust_mdot")?,
            has_other_torque: b.hasattr("other_torque")?,
            has_other_energy: b.hasattr("other_energy")?,
            has_other_build_initial_model: b
                .hasattr("other_build_initial_model")?,
            has_other_rsp_build_model: b.hasattr("other_rsp_build_model")?,
            has_other_rsp_linear_analysis: b
                .hasattr("other_rsp_linear_analysis")?,
            has_extra_history_columns: b.hasattr("extra_history_columns")?,
            py_obj: obj,
            state: PyAdapterState::default(),
        })
    }

    /// Consumes any deferred error raised in non-`Result` [`StarExtras`] hooks.
    ///
    /// # Errors
    /// Returns [`PyRuntimeError`] if any callback previously raised a Python exception.
    pub fn take_pending_py_err(&self) -> PyResult<()> {
        self.state.take_pending_py_err()
    }

    fn call_directive_hook(
        &self,
        enabled: bool,
        method: &str,
        default_terminate: &str,
        model: &mut StarModel,
    ) -> ExtrasDirective {
        if let Some(msg) = self.state.take_pending_msg() {
            return ExtrasDirective::Error(msg);
        }
        if !enabled {
            return ExtrasDirective::KeepGoing;
        }
        let case_dir = self.state.staged_dir_string();
        Python::attach(|py| {
            let snap = match build_star_snapshot(py, model, &case_dir, false) {
                Ok(s) => s,
                Err(e) => {
                    return ExtrasDirective::Error(format!(
                        "{method}: {}",
                        format_py_err(py, &e)
                    ));
                }
            };
            match self.py_obj.bind(py).call_method1(method, (snap,)) {
                Ok(res) => parse_star_directive(&res, default_terminate, model)
                    .unwrap_or_else(|e| {
                        ExtrasDirective::Error(format!(
                            "{method}: {}",
                            format_py_err(py, &e)
                        ))
                    }),
                Err(e) => ExtrasDirective::Error(format!(
                    "{method}: {}",
                    format_py_err(py, &e)
                )),
            }
        })
    }

    fn call_vec_hook(
        &self,
        enabled: bool,
        method: &str,
        model: &StarModel,
    ) -> Option<Vec<f64>> {
        if !enabled {
            return None;
        }
        let case_dir = self.state.staged_dir_string();
        Python::attach(|py| {
            let snap = match build_star_snapshot(py, model, &case_dir, false) {
                Ok(s) => s,
                Err(e) => {
                    self.state.record_py_err(method, format_py_err(py, &e));
                    return None;
                }
            };
            match self.py_obj.bind(py).call_method1(method, (snap,)) {
                Ok(res) if res.is_none() => None,
                Ok(res) => match res.extract::<Vec<f64>>() {
                    Ok(v) => Some(v),
                    Err(e) => {
                        self.state.record_py_err(method, format_py_err(py, &e));
                        None
                    }
                },
                Err(e) => {
                    self.state.record_py_err(method, format_py_err(py, &e));
                    None
                }
            }
        })
    }
}

impl StarExtras for PyStarExtrasAdapter {
    fn set_staged_case_dir(&mut self, case_dir: &Path) {
        self.state.set_staged_case_dir(case_dir);
    }

    fn extras_controls(
        &mut self,
        model: &mut StarModel,
    ) -> Result<(), StarError> {
        self.state.take_pending_star_err()?;
        if !self.has_controls {
            return Ok(());
        }
        let case_dir = self.state.staged_dir_string();
        Python::attach(|py| {
            let snap = build_star_snapshot(py, model, &case_dir, false)
                .map_err(|e| StarError::InvalidInput(format_py_err(py, &e)))?;
            let res = self
                .py_obj
                .bind(py)
                .call_method1("extras_controls", (snap,))
                .map_err(|e| {
                    StarError::InvalidInput(format!(
                        "extras_controls: {}",
                        format_py_err(py, &e)
                    ))
                })?;
            if let Ok(d) = res.cast::<PyDict>() {
                apply_star_updates(model, d).map_err(|e| {
                    StarError::InvalidInput(format_py_err(py, &e))
                })?;
            }
            Ok(())
        })
    }

    fn extras_startup(
        &mut self,
        model: &mut StarModel,
        restart: bool,
    ) -> Result<(), StarError> {
        self.state.take_pending_star_err()?;
        if !self.has_startup {
            return Ok(());
        }
        let case_dir = self.state.staged_dir_string();
        Python::attach(|py| {
            let snap = build_star_snapshot(py, model, &case_dir, false)
                .map_err(|e| StarError::InvalidInput(format_py_err(py, &e)))?;
            let res = self
                .py_obj
                .bind(py)
                .call_method1("extras_startup", (snap, restart))
                .map_err(|e| {
                    StarError::InvalidInput(format!(
                        "extras_startup: {}",
                        format_py_err(py, &e)
                    ))
                })?;
            if let Ok(d) = res.cast::<PyDict>() {
                apply_star_updates(model, d).map_err(|e| {
                    StarError::InvalidInput(format_py_err(py, &e))
                })?;
            }
            Ok(())
        })
    }

    fn extras_start_step(&mut self, model: &mut StarModel) -> ExtrasDirective {
        self.call_directive_hook(
            self.has_start_step,
            "extras_start_step",
            "extras_start_step",
            model,
        )
    }

    fn extras_check_model(&mut self, model: &mut StarModel) -> ExtrasDirective {
        self.call_directive_hook(
            self.has_check_model,
            "extras_check_model",
            "extras_check_model",
            model,
        )
    }

    fn extras_finish_step(&mut self, model: &mut StarModel) -> ExtrasDirective {
        self.call_directive_hook(
            self.has_finish_step,
            "extras_finish_step",
            "extras_finish_step",
            model,
        )
    }

    fn extras_after_evolve(
        &mut self,
        model: &mut StarModel,
    ) -> Result<(), StarError> {
        self.state.take_pending_star_err()?;
        if !self.has_after_evolve {
            return Ok(());
        }
        let case_dir = self.state.staged_dir_string();
        Python::attach(|py| {
            let snap = build_star_snapshot(py, model, &case_dir, true)
                .map_err(|e| StarError::InvalidInput(format_py_err(py, &e)))?;
            self.py_obj
                .bind(py)
                .call_method1("extras_after_evolve", (snap,))
                .map_err(|e| {
                    StarError::InvalidInput(format!(
                        "extras_after_evolve: {}",
                        format_py_err(py, &e)
                    ))
                })?;
            Ok(())
        })
    }

    fn other_cgrav(&self, model: &mut StarModel) -> Result<(), StarError> {
        self.state.take_pending_star_err()?;
        if !self.has_other_cgrav {
            return Ok(());
        }
        let case_dir = self.state.staged_dir_string();
        Python::attach(|py| {
            let snap = build_star_snapshot(py, model, &case_dir, false)
                .map_err(|e| StarError::InvalidInput(format_py_err(py, &e)))?;
            let res = self
                .py_obj
                .bind(py)
                .call_method1("other_cgrav", (snap,))
                .map_err(|e| {
                    StarError::InvalidInput(format!(
                        "other_cgrav: {}",
                        format_py_err(py, &e)
                    ))
                })?;
            if res.is_none() {
                return Ok(());
            }
            if let Ok(vals) = res.extract::<Vec<f64>>() {
                if vals.len() != model.nz {
                    return Err(StarError::InvalidInput(format!(
                        "other_cgrav returned {} values for nz = {}",
                        vals.len(),
                        model.nz
                    )));
                }
                model.cgrav = vals;
            } else {
                let scalar = res.extract::<f64>().map_err(|e| {
                    StarError::InvalidInput(format_py_err(py, &e))
                })?;
                model.cgrav.fill(scalar);
            }
            Ok(())
        })
    }

    fn other_alpha_mlt(&self, model: &mut StarModel) {
        if !self.has_other_alpha_mlt {
            return;
        }
        let case_dir = self.state.staged_dir_string();
        Python::attach(|py| {
            let snap = match build_star_snapshot(py, model, &case_dir, false) {
                Ok(s) => s,
                Err(e) => {
                    self.state.record_py_err(
                        "other_alpha_mlt",
                        format_py_err(py, &e),
                    );
                    return;
                }
            };
            match self
                .py_obj
                .bind(py)
                .call_method1("other_alpha_mlt", (snap,))
            {
                Ok(res) if !res.is_none() => match res.extract::<f64>() {
                    Ok(alpha) => {
                        model.controls.mixing_length_alpha = alpha;
                    }
                    Err(e) => self.state.record_py_err(
                        "other_alpha_mlt",
                        format_py_err(py, &e),
                    ),
                },
                Ok(_) => {}
                Err(e) => self
                    .state
                    .record_py_err("other_alpha_mlt", format_py_err(py, &e)),
            }
        });
    }

    fn other_wind(
        &self,
        model: &StarModel,
        l: f64,
        ms: f64,
        r: f64,
        t: f64,
    ) -> Result<Option<f64>, StarError> {
        self.state.take_pending_star_err()?;
        if !self.has_other_wind {
            return Ok(None);
        }
        let case_dir = self.state.staged_dir_string();
        Python::attach(|py| {
            let snap = build_star_snapshot(py, model, &case_dir, false)
                .map_err(|e| StarError::InvalidInput(format_py_err(py, &e)))?;
            let res = self
                .py_obj
                .bind(py)
                .call_method1("other_wind", (snap, l, ms, r, t))
                .map_err(|e| {
                    StarError::InvalidInput(format!(
                        "other_wind: {}",
                        format_py_err(py, &e)
                    ))
                })?;
            if res.is_none() {
                return Ok(None);
            }
            res.extract::<f64>().map(Some).map_err(|e| {
                StarError::InvalidInput(format!(
                    "other_wind: {}",
                    format_py_err(py, &e)
                ))
            })
        })
    }

    fn other_adjust_mdot(
        &self,
        model: &StarModel,
        current_mdot: f64,
    ) -> Result<Option<f64>, StarError> {
        self.state.take_pending_star_err()?;
        if !self.has_other_adjust_mdot {
            return Ok(None);
        }
        let case_dir = self.state.staged_dir_string();
        Python::attach(|py| {
            let snap = build_star_snapshot(py, model, &case_dir, false)
                .map_err(|e| StarError::InvalidInput(format_py_err(py, &e)))?;
            let res = self
                .py_obj
                .bind(py)
                .call_method1("other_adjust_mdot", (snap, current_mdot))
                .map_err(|e| {
                    StarError::InvalidInput(format!(
                        "other_adjust_mdot: {}",
                        format_py_err(py, &e)
                    ))
                })?;
            if res.is_none() {
                return Ok(None);
            }
            res.extract::<f64>().map(Some).map_err(|e| {
                StarError::InvalidInput(format!(
                    "other_adjust_mdot: {}",
                    format_py_err(py, &e)
                ))
            })
        })
    }

    fn other_torque(&self, model: &StarModel) -> Option<Vec<f64>> {
        self.call_vec_hook(self.has_other_torque, "other_torque", model)
    }

    fn other_energy(&self, model: &StarModel) -> Option<Vec<f64>> {
        self.call_vec_hook(self.has_other_energy, "other_energy", model)
    }

    fn other_build_initial_model(
        &mut self,
        bundle: &InlistBundle,
    ) -> Result<Option<StarModel>, StarError> {
        self.state.take_pending_star_err()?;
        if !self.has_other_build_initial_model {
            return Ok(None);
        }
        Python::attach(|py| {
            let ctx = PyRspBuildContext::from_bundle(bundle);
            let py_ctx = Py::new(py, ctx)
                .map_err(|e| StarError::InvalidInput(format_py_err(py, &e)))?;
            let res = self
                .py_obj
                .bind(py)
                .call_method1("other_build_initial_model", (py_ctx,))
                .map_err(|e| {
                    StarError::InvalidInput(format!(
                        "other_build_initial_model: {}",
                        format_py_err(py, &e)
                    ))
                })?;
            if res.is_none() {
                return Ok(None);
            }
            let pm = res.extract::<PyRef<'_, PyStarModel>>().map_err(|e| {
                StarError::InvalidInput(format_py_err(py, &PyErr::from(e)))
            })?;
            Ok(Some(pm.inner.clone()))
        })
    }

    fn other_rsp_build_model(
        &mut self,
        bundle: &InlistBundle,
    ) -> Result<Option<RspBuildOutcome>, StarError> {
        self.state.take_pending_star_err()?;
        if !self.has_other_rsp_build_model {
            return Ok(None);
        }
        Python::attach(|py| {
            let ctx = PyRspBuildContext::from_bundle(bundle);
            let py_ctx = Py::new(py, ctx)
                .map_err(|e| StarError::InvalidInput(format_py_err(py, &e)))?;
            let res = self
                .py_obj
                .bind(py)
                .call_method1("other_rsp_build_model", (py_ctx,))
                .map_err(|e| {
                    StarError::InvalidInput(format!(
                        "other_rsp_build_model: {}",
                        format_py_err(py, &e)
                    ))
                })?;
            if res.is_none() {
                return Ok(None);
            }
            if let Ok(tup) = res.cast::<PyTuple>() {
                if tup.len() != 2 {
                    return Err(StarError::InvalidInput(
                        "expected 2-tuple from other_rsp_build_model"
                            .to_string(),
                    ));
                }
                let tag: String =
                    tup.get_item(0).and_then(|v| v.extract()).map_err(|e| {
                        StarError::InvalidInput(format_py_err(py, &e))
                    })?;
                let msg: String =
                    tup.get_item(1).and_then(|v| v.extract()).map_err(|e| {
                        StarError::InvalidInput(format_py_err(py, &e))
                    })?;
                // The hook's message is diagnostic only; the code is the
                // hook's name, as for `parse_star_directive`.
                return match tag.as_str() {
                    "terminate" => {
                        eprintln!("other_rsp_build_model: {msg}");
                        Ok(Some(RspBuildOutcome::Terminate(String::from(
                            "other_rsp_build_model",
                        ))))
                    }
                    "error" => Err(StarError::InvalidInput(msg)),
                    _ => Err(StarError::InvalidInput(format!(
                        "unsupported other_rsp_build_model tuple tag '{tag}'"
                    ))),
                };
            }
            if let Ok(code) = res.extract::<String>() {
                if code != "terminate" {
                    return Err(StarError::InvalidInput(format!(
                        "unrecognised directive '{code}' returned by \
                         other_rsp_build_model"
                    )));
                }
                return Ok(Some(RspBuildOutcome::Terminate(String::from(
                    "other_rsp_build_model",
                ))));
            }
            let pm = res.extract::<PyRef<'_, PyStarModel>>().map_err(|e| {
                StarError::InvalidInput(format_py_err(py, &PyErr::from(e)))
            })?;
            Ok(Some(RspBuildOutcome::Model(Box::new(pm.inner.clone()))))
        })
    }

    fn other_rsp_linear_analysis(
        &mut self,
        model: &mut StarModel,
        _chem: &RspChemCache,
        log_dir: Option<&Path>,
    ) -> Result<bool, StarError> {
        self.state.take_pending_star_err()?;
        if !self.has_other_rsp_linear_analysis {
            return Ok(false);
        }
        let case_dir = self.state.staged_dir_string();
        let gyre_str = export_gyre(model, false);
        let log_dir_opt = log_dir.map(|p| p.to_string_lossy().into_owned());
        Python::attach(|py| {
            let snap = build_star_snapshot(py, model, &case_dir, false)
                .map_err(|e| StarError::InvalidInput(format_py_err(py, &e)))?;
            snap.set_item(
                "mode_for_setting_periodlin",
                model.controls.rsp.mode_for_setting_periodlin,
            )
            .map_err(|e| StarError::InvalidInput(format_py_err(py, &e)))?;
            let res = self
                .py_obj
                .bind(py)
                .call_method1(
                    "other_rsp_linear_analysis",
                    (snap, gyre_str, case_dir, log_dir_opt),
                )
                .map_err(|e| {
                    StarError::InvalidInput(format!(
                        "other_rsp_linear_analysis: {}",
                        format_py_err(py, &e)
                    ))
                })?;
            if res.is_none() {
                return Ok(false);
            }
            if let Ok(handled) = res.extract::<bool>() {
                return Ok(handled);
            }
            let d = res.cast::<PyDict>().map_err(|e| {
                StarError::InvalidInput(format_py_err(py, &PyErr::from(e)))
            })?;
            apply_rsp_linear_analysis_result(d, model)
                .map_err(|e| StarError::InvalidInput(format_py_err(py, &e)))
        })
    }

    fn extra_history_columns(&self, model: &StarModel) -> Vec<(String, f64)> {
        if !self.has_extra_history_columns {
            return Vec::new();
        }
        let case_dir = self.state.staged_dir_string();
        Python::attach(|py| {
            let snap = match build_star_snapshot(py, model, &case_dir, false) {
                Ok(s) => s,
                Err(e) => {
                    self.state.record_py_err(
                        "extra_history_columns",
                        format_py_err(py, &e),
                    );
                    return Vec::new();
                }
            };
            match self
                .py_obj
                .bind(py)
                .call_method1("extra_history_columns", (snap,))
            {
                Ok(res) => {
                    res.extract::<Vec<(String, f64)>>().unwrap_or_else(|e| {
                        self.state.record_py_err(
                            "extra_history_columns",
                            format_py_err(py, &e),
                        );
                        Vec::new()
                    })
                }
                Err(e) => {
                    self.state.record_py_err(
                        "extra_history_columns",
                        format_py_err(py, &e),
                    );
                    Vec::new()
                }
            }
        })
    }
}

fn set_binary_star_scalars(
    d: &Bound<'_, PyDict>,
    prefix: &str,
    star: &StarModel,
) -> PyResult<()> {
    let powers = compute_power_categories(star);
    let cores = compute_core_masses(star);
    d.set_item(format!("{prefix}_star_age"), star.star_age)?;
    d.set_item(
        format!("{prefix}_log_lnuc"),
        safe_log10(powers.power_nuc_burn - powers.power_photo),
    )?;
    d.set_item(format!("{prefix}_log_l"), safe_log10(star.surface_l_lsun()))?;
    // `s% L_nuc_burn_total` (`star/private/report.f90:112`, in `Lsun`),
    // `s% L(1)` (`erg s^-1`) and `s% mstar_dot` (`g s^-1`) as read by
    // `binary/test_suite/double_bh/src/run_binary_extras.f90:172,197` and
    // `binary/test_suite/jdot_gr_check/src/run_binary_extras.f90:160`.
    d.set_item(format!("{prefix}_l_nuc_burn_total"), powers.power_nuc_burn)?;
    if let Some(&l_surf) = star.lum.first() {
        d.set_item(format!("{prefix}_l_surf"), l_surf)?;
    }
    d.set_item(format!("{prefix}_mstar_dot"), star.mstar_dot)?;
    d.set_item(
        format!("{prefix}_center_h1"),
        center_avg_x_by_name(star, "h1"),
    )?;
    d.set_item(
        format!("{prefix}_center_he4"),
        center_avg_x_by_name(star, "he4"),
    )?;
    d.set_item(
        format!("{prefix}_center_c12"),
        center_avg_x_by_name(star, "c12"),
    )?;
    if let Some(y) = surface_iso_frac(star, "he4") {
        d.set_item(format!("{prefix}_surface_he4"), y)?;
    }
    d.set_item(format!("{prefix}_he_core_mass"), cores.he_core_mass)?;
    Ok(())
}

fn build_binary_snapshot<'py>(
    py: Python<'py>,
    bin: &BinarySystem,
    case_dir: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let initial_sep_cm = if bin.controls.initial_period_in_days > 0.0 {
        separation_from_period(
            bin.initial_mass_msun[0] * M_SUN,
            bin.initial_mass_msun[1] * M_SUN,
            bin.initial_binary_period_sec,
        )
    } else {
        bin.controls.initial_separation_in_rsuns * RSUN
    };
    let d = PyDict::new(py);
    d.set_item("model_number", bin.model_number)?;
    d.set_item("binary_age_yr", bin.binary_age_yr)?;
    d.set_item("separation", bin.separation)?;
    d.set_item("period", bin.period)?;
    d.set_item("eccentricity", bin.eccentricity)?;
    d.set_item("initial_period_sec", bin.initial_binary_period_sec)?;
    d.set_item("initial_separation_cm", initial_sep_cm)?;
    // The controls `b% initial_period_in_days` and
    // `b% initial_separation_in_Rsuns` exactly as set by the inlist; upstream
    // never rewrites them (`binary/private/binary_evolve.f90:106-108`).
    d.set_item(
        "initial_period_in_days",
        bin.controls.initial_period_in_days,
    )?;
    d.set_item(
        "initial_separation_in_rsuns",
        bin.controls.initial_separation_in_rsuns,
    )?;
    // The controls `b% m1` and `b% m2` (`Msun`) as set by the inlist, which
    // `jdot_gr_check` and `jdot_ml_check` read, as opposed to
    // `initial_mass_msun` (`binary_evolve.f90:73-85`).
    d.set_item("m1", bin.controls.m1_msun)?;
    d.set_item("m2", bin.controls.m2_msun)?;
    // `b% m(1:2)` in grams, ordered by star index.
    d.set_item("m", PyList::new(py, bin.star_masses_g())?)?;
    d.set_item("initial_mass_msun", PyList::new(py, bin.initial_mass_msun)?)?;
    d.set_item("initial_eccentricity", bin.controls.initial_eccentricity)?;
    d.set_item("angular_momentum_j", bin.angular_momentum_j)?;
    d.set_item("mtransfer_rate", bin.mtransfer_rate)?;
    d.set_item("step_mtransfer_rate", bin.step_mtransfer_rate)?;
    d.set_item("mass_transfer_alpha", bin.mass_transfer_alpha)?;
    d.set_item("mass_transfer_beta", bin.mass_transfer_beta)?;
    d.set_item("mass_transfer_delta", bin.mass_transfer_delta)?;
    d.set_item("mass_transfer_gamma", bin.mass_transfer_gamma)?;
    d.set_item("jdot", bin.jdot)?;
    d.set_item("rl", PyList::new(py, bin.rl)?)?;
    d.set_item("rl_relative_gap", PyList::new(py, bin.rl_relative_gap)?)?;
    d.set_item("point_mass_i", bin.point_mass_i)?;
    d.set_item("donor_i", bin.donor_i)?;
    d.set_item("ignore_rlof_flag", bin.ignore_rlof_flag)?;
    d.set_item("donor_mass_msun", bin.donor.mstar / M_SUN)?;
    d.set_item("accretor_mass_msun", bin.accretor_mass_g() / M_SUN)?;
    d.set_item("case_dir", case_dir)?;
    set_user_controls(&d, &bin.donor.controls)?;
    let companion_star = match &bin.companion {
        BinaryCompanion::Star(s2) => Some(s2.as_ref()),
        BinaryCompanion::PointMass { .. } => None,
    };
    let (s1, s2) = if bin.donor_i == 2 {
        (companion_star, Some(&bin.donor))
    } else {
        (Some(&bin.donor), companion_star)
    };
    if let Some(star1) = s1 {
        set_binary_star_scalars(&d, "s1", star1)?;
    }
    if let Some(star2) = s2 {
        set_binary_star_scalars(&d, "s2", star2)?;
    }
    Ok(d)
}

fn apply_binary_updates(
    bin: &mut BinarySystem,
    updates: &Bound<'_, PyDict>,
) -> PyResult<()> {
    if let Some(val) = updates.get_item("ignore_rlof_flag")? {
        bin.set_ignore_rlof_flag(val.extract::<bool>()?);
    }
    if let Some(val) = updates.get_item("point_mass_i")? {
        bin.set_point_mass_i(val.extract::<usize>()?)
            .map_err(to_py_err)?;
    }
    if let Some(val) = updates.get_item("period_eccentricity")? {
        let (p_sec, ecc) = val.extract::<(f64, f64)>()?;
        bin.set_period_eccentricity(p_sec, ecc).map_err(to_py_err)?;
    }
    if let Some(val) = updates.get_item("max_implicit_abs_mdot")? {
        bin.controls.max_implicit_abs_mdot = val.extract::<f64>()?;
    }
    if let Some(val) = updates.get_item("terminate_if_l2_overflow")? {
        bin.controls.terminate_if_l2_overflow = val.extract::<bool>()?;
    }
    if let Some(val) = updates.get_item("delta_mdot_hard_limit")? {
        let lim = val.extract::<f64>()?;
        bin.donor.controls.mix_wind.delta_mdot_hard_limit = lim;
        if let BinaryCompanion::Star(s2) = &mut bin.companion {
            s2.controls.mix_wind.delta_mdot_hard_limit = lim;
        }
    }
    Ok(())
}

/// Converts a binary hook's return value into a [`BinaryExtrasAction`].
///
/// Same contract as [`parse_star_directive`]: the termination code is always
/// `default_terminate_code` (the hook's upstream name), a message supplied
/// with `terminate` is printed to stderr, and an unrecognised action string
/// is an error.
fn parse_binary_action(
    res: &Bound<'_, PyAny>,
    default_terminate_code: &str,
    bin: &mut BinarySystem,
) -> PyResult<BinaryExtrasAction> {
    if res.is_none() {
        return Ok(BinaryExtrasAction::KeepGoing);
    }
    let mut dict_msg: Option<String> = None;
    let action_str: String = if let Ok(tup) = res.cast::<PyTuple>() {
        if tup.len() != 2 {
            return Err(PyValueError::new_err(
                "expected 2-tuple from binary extras hook",
            ));
        }
        let second = tup.get_item(1)?;
        if let Ok(updates) = second.cast::<PyDict>() {
            apply_binary_updates(bin, updates)?;
            tup.get_item(0)?.extract::<String>()?
        } else {
            let first: String = tup.get_item(0)?.extract()?;
            let msg: String = second.extract()?;
            match first.as_str() {
                "terminate" => dict_msg = Some(msg),
                "error" => return Ok(BinaryExtrasAction::Error(msg)),
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "unsupported binary action tuple tag '{first}' \
                         from {default_terminate_code}"
                    )));
                }
            }
            first
        }
    } else if let Ok(d) = res.cast::<PyDict>() {
        if let Some(updates) = d.get_item("updates")? {
            if let Ok(udict) = updates.cast::<PyDict>() {
                apply_binary_updates(bin, udict)?;
            }
        }
        if let Some(msg_obj) = d.get_item("message")? {
            if !msg_obj.is_none() {
                dict_msg = Some(msg_obj.extract::<String>()?);
            }
        }
        if let Some(act) = d.get_item("action")? {
            act.extract::<String>()?
        } else {
            String::from("keep_going")
        }
    } else {
        res.extract::<String>()?
    };

    match action_str.as_str() {
        "keep_going" => Ok(BinaryExtrasAction::KeepGoing),
        "error" => {
            Ok(BinaryExtrasAction::Error(dict_msg.unwrap_or_else(|| {
                String::from("error returned by binary extras hook")
            })))
        }
        "terminate" => {
            if let Some(msg) = dict_msg {
                eprintln!("{default_terminate_code}: {msg}");
            }
            Ok(BinaryExtrasAction::Terminate(
                default_terminate_code.to_string(),
            ))
        }
        other => Err(PyValueError::new_err(format!(
            "unrecognised action '{other}' returned by \
             {default_terminate_code}"
        ))),
    }
}

/// Thread-safe adapter forwarding [`BinaryExtras`] callbacks into a Python
/// test-side extras object while keeping per-step calls clone-free (`#148`).
#[allow(clippy::struct_excessive_bools)]
pub struct PyBinaryExtrasAdapter {
    py_obj: Py<PyAny>,
    has_other_wind_cgs: bool,
    has_check_model: bool,
    has_finish_step: bool,
    has_extra_binary_history_columns: bool,
    has_extra_star_history_columns: bool,
    state: PyAdapterState,
}

impl PyBinaryExtrasAdapter {
    /// Wraps a Python binary extras object, caching available hook flags under the GIL.
    ///
    /// # Errors
    /// Returns [`PyErr`] if attribute inspection fails.
    pub fn new(py: Python<'_>, obj: Py<PyAny>) -> PyResult<Self> {
        let b = obj.bind(py);
        Ok(Self {
            has_other_wind_cgs: b.hasattr("other_wind_cgs")?,
            has_check_model: b.hasattr("check_model")?,
            has_finish_step: b.hasattr("finish_step")?,
            has_extra_binary_history_columns: b
                .hasattr("extra_binary_history_columns")?,
            has_extra_star_history_columns: b
                .hasattr("extra_star_history_columns")?,
            py_obj: obj,
            state: PyAdapterState::default(),
        })
    }

    /// Consumes any deferred error raised in non-`Result` [`BinaryExtras`] hooks.
    ///
    /// # Errors
    /// Returns [`PyRuntimeError`] if any callback previously raised a Python exception.
    pub fn take_pending_py_err(&self) -> PyResult<()> {
        self.state.take_pending_py_err()
    }

    fn call_action_hook(
        &mut self,
        enabled: bool,
        method: &str,
        default_terminate: &str,
        bin: &mut BinarySystem,
    ) -> BinaryExtrasAction {
        if let Some(msg) = self.state.take_pending_msg() {
            return BinaryExtrasAction::Error(msg);
        }
        if !enabled {
            return BinaryExtrasAction::KeepGoing;
        }
        let case_dir = self.state.staged_dir_string();
        Python::attach(|py| {
            let snap = match build_binary_snapshot(py, bin, &case_dir) {
                Ok(s) => s,
                Err(e) => {
                    return BinaryExtrasAction::Error(format!(
                        "{method}: {}",
                        format_py_err(py, &e)
                    ));
                }
            };
            match self.py_obj.bind(py).call_method1(method, (snap,)) {
                Ok(res) => parse_binary_action(&res, default_terminate, bin)
                    .unwrap_or_else(|e| {
                        BinaryExtrasAction::Error(format!(
                            "{method}: {}",
                            format_py_err(py, &e)
                        ))
                    }),
                Err(e) => BinaryExtrasAction::Error(format!(
                    "{method}: {}",
                    format_py_err(py, &e)
                )),
            }
        })
    }
}

impl BinaryExtras for PyBinaryExtrasAdapter {
    fn set_staged_case_dir(&mut self, case_dir: &Path) {
        self.state.set_staged_case_dir(case_dir);
    }

    fn other_wind_cgs(
        &mut self,
        star: &StarModel,
        star_idx: usize,
        ignore_rlof_flag: bool,
    ) -> Result<Option<f64>, StarError> {
        self.state.take_pending_star_err()?;
        if !self.has_other_wind_cgs {
            return Ok(None);
        }
        let case_dir = self.state.staged_dir_string();
        // The four surface arguments of upstream's `other_wind`
        // (`star/private/winds.f90:81-84,345`), from the values `set_mdot`
        // is called with (`star/private/evolve.f90:427`:
        // `s% L_phot*Lsun, s% mstar, s% Teff`). No floors or fallbacks.
        let l1 = eval_photosphere_info(star).l_lsun * LSUN;
        let m1 = star.mstar;
        let t1 = star.teff;
        let r1 = (l1 / (PI * CRAD * CLIGHT * t1 * t1 * t1 * t1)).sqrt();
        Python::attach(|py| {
            let snap = build_star_snapshot(py, star, &case_dir, false)
                .map_err(|e| StarError::InvalidInput(format_py_err(py, &e)))?;
            let res = self
                .py_obj
                .bind(py)
                .call_method1(
                    "other_wind_cgs",
                    (snap, star_idx, ignore_rlof_flag, l1, m1, r1, t1),
                )
                .map_err(|e| {
                    StarError::InvalidInput(format!(
                        "other_wind_cgs: {}",
                        format_py_err(py, &e)
                    ))
                })?;
            if res.is_none() {
                Ok(None)
            } else {
                res.extract::<f64>()
                    .map(Some)
                    .map_err(|e| StarError::InvalidInput(format_py_err(py, &e)))
            }
        })
    }

    fn check_model(&mut self, bin: &mut BinarySystem) -> BinaryExtrasAction {
        self.call_action_hook(
            self.has_check_model,
            "check_model",
            "extras_binary_check_model",
            bin,
        )
    }

    fn finish_step(&mut self, bin: &mut BinarySystem) -> BinaryExtrasAction {
        self.call_action_hook(
            self.has_finish_step,
            "finish_step",
            "extras_binary_finish_step",
            bin,
        )
    }

    fn extra_binary_history_columns(
        &self,
        bin: &BinarySystem,
    ) -> Vec<(String, f64)> {
        if !self.has_extra_binary_history_columns {
            return Vec::new();
        }
        let case_dir = self.state.staged_dir_string();
        Python::attach(|py| {
            let snap = match build_binary_snapshot(py, bin, &case_dir) {
                Ok(s) => s,
                Err(e) => {
                    self.state.record_py_err(
                        "extra_binary_history_columns",
                        format_py_err(py, &e),
                    );
                    return Vec::new();
                }
            };
            match self
                .py_obj
                .bind(py)
                .call_method1("extra_binary_history_columns", (snap,))
            {
                Ok(res) => {
                    res.extract::<Vec<(String, f64)>>().unwrap_or_else(|e| {
                        self.state.record_py_err(
                            "extra_binary_history_columns",
                            format_py_err(py, &e),
                        );
                        Vec::new()
                    })
                }
                Err(e) => {
                    self.state.record_py_err(
                        "extra_binary_history_columns",
                        format_py_err(py, &e),
                    );
                    Vec::new()
                }
            }
        })
    }

    fn extra_star_history_columns(
        &self,
        bin: &BinarySystem,
        star_idx: usize,
    ) -> Vec<(String, f64)> {
        if !self.has_extra_star_history_columns {
            return Vec::new();
        }
        let case_dir = self.state.staged_dir_string();
        Python::attach(|py| {
            let snap = match build_binary_snapshot(py, bin, &case_dir) {
                Ok(s) => s,
                Err(e) => {
                    self.state.record_py_err(
                        "extra_star_history_columns",
                        format_py_err(py, &e),
                    );
                    return Vec::new();
                }
            };
            match self
                .py_obj
                .bind(py)
                .call_method1("extra_star_history_columns", (snap, star_idx))
            {
                Ok(res) => {
                    res.extract::<Vec<(String, f64)>>().unwrap_or_else(|e| {
                        self.state.record_py_err(
                            "extra_star_history_columns",
                            format_py_err(py, &e),
                        );
                        Vec::new()
                    })
                }
                Err(e) => {
                    self.state.record_py_err(
                        "extra_star_history_columns",
                        format_py_err(py, &e),
                    );
                    Vec::new()
                }
            }
        })
    }
}

pub(crate) fn run_with_star_extras<T, F>(
    py: Python<'_>,
    extras: Option<Py<PyAny>>,
    run: F,
) -> PyResult<T>
where
    F: FnOnce(&mut dyn StarExtras) -> Result<T, StarError>,
{
    let Some(obj) = extras else {
        let mut default = DefaultStarExtras;
        return run(&mut default).map_err(to_py_err);
    };
    let mut adapter = PyStarExtrasAdapter::new(py, obj)?;
    match run(&mut adapter) {
        Ok(val) => {
            adapter.take_pending_py_err()?;
            Ok(val)
        }
        Err(err) => {
            adapter.take_pending_py_err()?;
            Err(to_py_err(err))
        }
    }
}

pub(crate) fn run_with_binary_extras<T, F>(
    py: Python<'_>,
    extras: Option<Py<PyAny>>,
    run: F,
) -> PyResult<T>
where
    F: FnOnce(&mut dyn BinaryExtras) -> Result<T, StarError>,
{
    let Some(obj) = extras else {
        let mut default = DefaultBinaryExtras;
        return run(&mut default).map_err(to_py_err);
    };
    let mut adapter = PyBinaryExtrasAdapter::new(py, obj)?;
    match run(&mut adapter) {
        Ok(val) => {
            adapter.take_pending_py_err()?;
            Ok(val)
        }
        Err(err) => {
            adapter.take_pending_py_err()?;
            Err(to_py_err(err))
        }
    }
}

/// Registers test-side extras context types and full `GYRE` mode helpers on `flexmesa._rust`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRspBuildContext>()?;
    m.add_function(wrap_pyfunction!(star_solve_gyre_mesa_modes_full, m)?)?;
    Ok(())
}
