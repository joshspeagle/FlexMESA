// SPDX-License-Identifier: GPL-3.0-or-later
//! `PyO3` callback adapters and RSP/GYRE helpers bridging Python test-side
//! extras (`run_star_extras` / `run_binary_extras` implementations) into
//! `StarExtras` and `BinaryExtras` (`#148`).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use flexmesa_binary::{
    BinaryCompanion, BinaryExtras, BinaryExtrasAction, BinarySystem,
    DefaultBinaryExtras, separation_from_period,
};
use flexmesa_const::{
    CLIGHT, CRAD, LN10, LSUN, M_SUN, PI, RSUN, SECDAY, SECYER,
};
use flexmesa_eos::{
    I_FRAC_CMS, I_FRAC_FREE_EOS, I_FRAC_HELM, I_FRAC_IDEAL, I_FRAC_OPAL_SCVH,
    I_FRAC_PC, I_FRAC_SKYE,
};
use flexmesa_star::hydro_state::{
    OtherAdjustMltGradtFractionHandle, OtherAdjustMltGradtFractionHook,
    OtherOvershootingSchemeHandle, OtherOvershootingSchemeHook,
};
use flexmesa_star::inlist_runner::{
    DefaultStarExtras, ExtrasDirective, RspBuildOutcome, StarExtras,
    install_star_aux_hooks,
};
use flexmesa_star::overshoot::{
    OverBdyAnchorCache, eval_conv_bdy_hp, eval_over_bdy_params,
};
use flexmesa_star::report_utils::surface_avg_x_slot;
use flexmesa_star::{
    AtmTTauRelation, InertiaNormKind, InlistBundle, OuterBoundKind,
    RspChemCache, StarControls, StarError, StarModel, center_avg_x_by_name,
    center_value, compute_core_masses, compute_power_categories,
    do_rsp_build_model, eval_conv_and_mx_summary, eval_photosphere_info,
    eval_surf_avg_rotation_info, export_gyre, parse_gyre_mesa_model,
    safe_log10, solve_bvp_eigenmodes,
};
use numpy::PyArray1;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::binary_py::PyBinarySystem;
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

/// 0-based index of the `c12_c12` (`icc`) burning category in
/// `flexmesa_chem::CATEGORY_NAME` (`CATEGORY_NAME[16] == "c12_c12"`).
const CATEGORY_ICC: usize = 16;

/// Signed $\log_{10}$ matching `star/private/profile_getval.f90:475-477`
/// (`p_net_nuclear_energy`) and `flexmesa_star::mesa_profile::signed_log`.
#[inline]
fn signed_log(val: f64) -> f64 {
    let sgn = if val.is_sign_negative() { -1.0 } else { 1.0 };
    sgn * flexmesa_math::log10(val.abs().max(1.0))
}

/// Canonical upstream MESA string for [`AtmTTauRelation`].
const fn atm_t_tau_relation_name(rel: AtmTTauRelation) -> &'static str {
    match rel {
        AtmTTauRelation::Eddington => "Eddington",
        AtmTTauRelation::SolarHopf => "solar_Hopf",
        AtmTTauRelation::KrishnaSwamy => "Krishna_Swamy",
        AtmTTauRelation::TrampedachSolar => "Trampedach_solar",
    }
}

/// Parses an upstream MESA `atm_T_tau_relation` string, raising `ValueError`
/// on unrecognized names (`inlist_populate.rs:504-511`).
fn parse_atm_t_tau_relation(raw: &str) -> PyResult<AtmTTauRelation> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "eddington" => Ok(AtmTTauRelation::Eddington),
        "solar_hopf" => Ok(AtmTTauRelation::SolarHopf),
        "krishna_swamy" => Ok(AtmTTauRelation::KrishnaSwamy),
        "trampedach_solar" => Ok(AtmTTauRelation::TrampedachSolar),
        other => Err(PyValueError::new_err(format!(
            "unrecognized atm_t_tau_relation '{other}'"
        ))),
    }
}

fn set_star_control_scalars(
    d: &Bound<'_, PyDict>,
    model: &StarModel,
) -> PyResult<()> {
    let c = &model.controls;
    d.set_item("rsp_flag", c.rsp_flag)?;
    d.set_item("u_flag", c.u_flag)?;
    d.set_item("v_flag", c.v_flag)?;
    d.set_item("rotation_flag", c.rotation_flag)?;
    d.set_item(
        "create_pre_main_sequence_model",
        model.job.create_pre_ms_model,
    )?;
    d.set_item(
        "atm_t_tau_relation",
        atm_t_tau_relation_name(c.atm_t_tau_relation),
    )?;
    d.set_item("log_l_upper_limit", c.log_l_upper_limit.unwrap_or(1.0e99))?;
    d.set_item("teff_lower_limit", c.teff_lower_limit.unwrap_or(-1.0e99))?;
    d.set_item("max_age", c.max_age)?;
    d.set_item("mixing_length_alpha", c.mixing_length_alpha)?;
    d.set_item("accretion_h1", c.accretion_h1)?;
    d.set_item("reimers_scaling_factor", c.reimers_scaling_factor)?;
    d.set_item("blocker_scaling_factor", c.blocker_scaling_factor)?;
    d.set_item("do_element_diffusion", c.do_element_diffusion)?;
    d.set_item("mass_change", c.mass_change)?;
    d.set_item("initial_mass", c.initial_mass)?;
    d.set_item("max_timestep", c.ts_stop.max_timestep)?;
    d.set_item("center_avg_value_dq", c.output.center_avg_value_dq)?;
    d.set_item(
        "he_core_boundary_h1_fraction",
        c.ts_stop.he_core_boundary_h1_fraction,
    )?;
    d.set_item("zbase", c.zbase)?;
    d.set_item("use_other_cgrav", c.use_other_cgrav)?;
    d.set_item("use_other_alpha_mlt", c.use_other_alpha_mlt)?;
    d.set_item("use_other_wind", c.use_other_wind)?;
    d.set_item("use_other_adjust_mdot", c.use_other_adjust_mdot)?;
    d.set_item("use_other_torque", c.use_other_torque)?;
    d.set_item("use_other_torque_implicit", c.use_other_torque_implicit)?;
    d.set_item("use_other_energy", c.use_other_energy)?;
    d.set_item("use_other_kap", c.use_other_kap)?;
    d.set_item("use_other_surface_pt", c.use_other_surface_pt)?;
    d.set_item(
        "use_other_overshooting_scheme",
        c.use_other_overshooting_scheme,
    )?;
    d.set_item(
        "use_other_mesh_functions",
        c.mix_wind.use_other_mesh_functions,
    )?;
    d.set_item(
        "use_other_mesh_delta_coeff_factor",
        c.mix_wind.use_other_mesh_delta_coeff_factor,
    )?;
    // Upstream `controls.defaults` sets `use_other_remove_surface = .false.`,
    // and `StarControls` does not yet wire `other_remove_surface` (`#153`).
    d.set_item("use_other_remove_surface", false)?;
    d.set_item("min_d_mix", c.min_d_mix)?;
    d.set_item("overshoot_d_min", c.mix_wind.overshoot_d_min)?;
    let py = d.py();
    let sets = &c.mix_wind.overshoot_sets;
    d.set_item("overshoot_f", PyList::new(py, sets.iter().map(|s| s.f))?)?;
    d.set_item("overshoot_f0", PyList::new(py, sets.iter().map(|s| s.f0))?)?;
    d.set_item("overshoot_d0", PyList::new(py, sets.iter().map(|s| s.d0))?)?;
    d.set_item(
        "overshoot_delta0",
        PyList::new(py, sets.iter().map(|s| s.delta0))?,
    )?;
    d.set_item(
        "overshoot_mass_full_on",
        PyList::new(py, sets.iter().map(|s| s.mass_full_on))?,
    )?;
    d.set_item(
        "overshoot_mass_full_off",
        PyList::new(py, sets.iter().map(|s| s.mass_full_off))?,
    )?;
    Ok(())
}

fn set_star_report_scalars(
    py: Python<'_>,
    d: &Bound<'_, PyDict>,
    model: &StarModel,
) -> PyResult<()> {
    let nz = model.nz;
    let cores = compute_core_masses(model);
    let powers = compute_power_categories(model);
    let phot = eval_photosphere_info(model);
    let mix = eval_conv_and_mx_summary(model);
    let surf_rot = eval_surf_avg_rotation_info(model);

    d.set_item("center_h1", center_avg_x_by_name(model, "h1"))?;
    d.set_item("center_he4", center_avg_x_by_name(model, "he4"))?;
    d.set_item("center_c12", center_avg_x_by_name(model, "c12"))?;
    d.set_item("he_core_mass", cores.he_core_mass)?;
    d.set_item("co_core_mass", cores.co_core_mass)?;
    d.set_item("one_core_mass", cores.one_core_mass)?;
    d.set_item("fe_core_mass", cores.fe_core_mass)?;
    d.set_item("he_core_omega", cores.he_bdy.omega)?;
    d.set_item("power_nuc_burn", powers.power_nuc_burn)?;
    d.set_item("power_h_burn", powers.power_h_burn)?;
    d.set_item("power_he_burn", powers.power_he_burn)?;
    d.set_item("power_z_burn", powers.power_z_burn)?;
    d.set_item("power_photo", powers.power_photo)?;
    d.set_item("power_neutrinos", powers.power_neutrinos)?;
    d.set_item("l_by_category_icc", powers.l_by_category_lsun[CATEGORY_ICC])?;
    d.set_item(
        "l_by_category",
        PyArray1::from_slice(py, &powers.l_by_category_lsun),
    )?;
    d.set_item("l_phot", phot.l_lsun)?;
    d.set_item("photosphere_r", phot.r_rsun)?;
    d.set_item("mass_conv_core", mix.mass_conv_core)?;
    d.set_item("conv_mx1_top", mix.conv_mx_q[0])?;
    d.set_item("conv_mx1_bot", mix.conv_mx_q[1])?;
    d.set_item("w_div_w_crit_avg_surf", surf_rot.w_div_w_crit)?;
    d.set_item("omega_avg_surf", surf_rot.omega)?;
    d.set_item("v_rot_avg_surf", surf_rot.v_rot_cm_s)?;
    d.set_item(
        "log_center_temperature",
        center_value(model, &model.ln_t) / LN10,
    )?;
    d.set_item("log_center_density", center_value(model, &model.lnd) / LN10)?;
    d.set_item("center_omega", center_value(model, &model.omega))?;
    d.set_item("log_dt", safe_log10(model.dt / SECYER))?;
    d.set_item("log_l", safe_log10(model.surface_l_lsun()))?;
    d.set_item("log_teff", safe_log10(model.teff))?;
    d.set_item("log_r", safe_log10(model.surface_r_rsun()))?;

    let total_j: f64 = if model.controls.rotation_flag {
        model
            .dm_bar
            .iter()
            .take(nz)
            .zip(model.j_rot.iter().take(nz))
            .map(|(&dm, &j)| dm * j)
            .sum()
    } else {
        0.0
    };
    d.set_item("total_angular_momentum", total_j)?;

    let net_iso = PyDict::new(py);
    for (j, name) in model.iso_names.iter().enumerate() {
        let iso_lower = name.to_ascii_lowercase();
        let slot_1based = j + 1;
        net_iso.set_item(&iso_lower, slot_1based)?;
        d.set_item(format!("net_iso_{iso_lower}"), slot_1based)?;
        if nz > 0 {
            d.set_item(
                format!("surface_{iso_lower}"),
                surface_avg_x_slot(model, Some(j)),
            )?;
        }
    }
    d.set_item("net_iso", net_iso)?;
    Ok(())
}

fn set_star_eos_frac_arrays(
    py: Python<'_>,
    d: &Bound<'_, PyDict>,
    model: &StarModel,
) -> PyResult<()> {
    let nz = model.nz;
    if nz == 0 || model.eos_res.len() < nz {
        return Ok(());
    }
    let mut opal = Vec::with_capacity(nz);
    let mut helm = Vec::with_capacity(nz);
    let mut skye = Vec::with_capacity(nz);
    let mut pc = Vec::with_capacity(nz);
    let mut free_eos = Vec::with_capacity(nz);
    let mut cms = Vec::with_capacity(nz);
    let mut ideal = Vec::with_capacity(nz);
    for row in &model.eos_res[..nz] {
        opal.push(row[I_FRAC_OPAL_SCVH]);
        helm.push(row[I_FRAC_HELM]);
        skye.push(row[I_FRAC_SKYE]);
        pc.push(row[I_FRAC_PC]);
        free_eos.push(row[I_FRAC_FREE_EOS]);
        cms.push(row[I_FRAC_CMS]);
        ideal.push(row[I_FRAC_IDEAL]);
    }
    d.set_item("eos_frac_opal_scvh", PyArray1::from_vec(py, opal))?;
    d.set_item("eos_frac_helm", PyArray1::from_vec(py, helm))?;
    d.set_item("eos_frac_skye", PyArray1::from_vec(py, skye))?;
    d.set_item("eos_frac_pc", PyArray1::from_vec(py, pc))?;
    d.set_item("eos_frac_free_eos", PyArray1::from_vec(py, free_eos))?;
    d.set_item("eos_frac_cms", PyArray1::from_vec(py, cms))?;
    d.set_item("eos_frac_ideal", PyArray1::from_vec(py, ideal))?;
    Ok(())
}

fn set_star_species_arrays(
    py: Python<'_>,
    d: &Bound<'_, PyDict>,
    model: &StarModel,
) -> PyResult<()> {
    let nz = model.nz;
    let ns = model.species;
    if nz == 0
        || ns == 0
        || model.xa.len() < nz * ns
        || model.iso_names.len() < ns
    {
        return Ok(());
    }
    let mut cols: Vec<Vec<f64>> =
        (0..ns).map(|_| Vec::with_capacity(nz)).collect();
    for k in 0..nz {
        let row = &model.xa[k * ns..(k + 1) * ns];
        for (col, &val) in cols.iter_mut().zip(row.iter()) {
            col.push(val);
        }
    }
    for (name, col) in model.iso_names.iter().take(ns).zip(cols) {
        let key = format!("xa_{}", name.to_ascii_lowercase());
        d.set_item(key, PyArray1::from_vec(py, col))?;
    }
    Ok(())
}

fn set_star_radial_arrays(
    py: Python<'_>,
    d: &Bound<'_, PyDict>,
    model: &StarModel,
) -> PyResult<()> {
    let nz = model.nz;
    if nz == 0 {
        return Ok(());
    }
    let set_f64 = |key: &str, slice: &[f64]| -> PyResult<()> {
        if slice.len() >= nz {
            d.set_item(key, PyArray1::from_slice(py, &slice[..nz]))?;
        }
        Ok(())
    };
    set_f64("r", &model.r)?;
    set_f64("m", &model.m)?;
    set_f64("q", &model.q)?;
    set_f64("dq", &model.dq)?;
    set_f64("dm", &model.dm)?;
    set_f64("dm_bar", &model.dm_bar)?;
    set_f64("t", &model.t)?;
    set_f64("rho", &model.rho)?;
    set_f64("lnd", &model.lnd)?;
    set_f64("lnt", &model.ln_t)?;
    set_f64("peos", &model.peos)?;
    set_f64("ln_peos", &model.ln_peos)?;
    set_f64("pgas", &model.pgas)?;
    set_f64("prad", &model.prad)?;
    set_f64("energy", &model.energy)?;
    set_f64("entropy", &model.entropy)?;
    set_f64("cp", &model.cp)?;
    set_f64("csound", &model.csound)?;
    set_f64("gamma1", &model.gamma1)?;
    set_f64("grada", &model.grada)?;
    set_f64("gradr", &model.gradr)?;
    set_f64("gradt", &model.gradt)?;
    set_f64("abar", &model.abar)?;
    set_f64("zbar", &model.zbar)?;
    set_f64("mu", &model.mu)?;
    set_f64("opacity", &model.opacity)?;
    set_f64("d_opacity_dlnd", &model.d_opacity_dlnd)?;
    set_f64("d_opacity_dlnt", &model.d_opacity_dlnt)?;
    set_f64("tau", &model.tau)?;
    set_f64("v", &model.v)?;
    set_f64("u", &model.u)?;
    set_f64("cgrav", &model.cgrav)?;
    set_f64("grav", &model.grav)?;
    set_f64("omega", &model.omega)?;
    set_f64("j_rot", &model.j_rot)?;
    set_f64("d_mix", &model.d_mix)?;
    set_f64(
        "adjust_mlt_gradt_fraction",
        &model.aux.mixing.adjust_mlt_gradt_fraction,
    )?;
    set_f64("conv_vel", &model.conv_vel)?;
    set_f64("mlt_vc", &model.mlt_vc)?;
    set_f64("eps_nuc", &model.eps_nuc)?;
    set_f64("non_nuc_neu", &model.non_nuc_neu)?;
    set_f64(
        "brunt_n2_composition_term",
        &model.aux.brunt_n2_composition_term,
    )?;
    set_f64("w_div_w_crit_roche", &model.aux.w_div_wc)?;

    if model.eps_nuc.len() >= nz && model.non_nuc_neu.len() >= nz {
        let net_nuc: Vec<f64> = (0..nz)
            .map(|k| signed_log(model.eps_nuc[k] - model.non_nuc_neu[k]))
            .collect();
        d.set_item("net_nuclear_energy", PyArray1::from_vec(py, net_nuc))?;
    }
    if model.mixing_type.len() >= nz {
        let mix_codes: Vec<i32> = model.mixing_type[..nz]
            .iter()
            .map(|&mt| mt as i32)
            .collect();
        d.set_item("mixing_type", PyArray1::from_vec(py, mix_codes))?;
    }
    set_star_eos_frac_arrays(py, d, model)?;
    set_star_species_arrays(py, d, model)?;
    Ok(())
}

fn build_star_snapshot<'py>(
    py: Python<'py>,
    model: &StarModel,
    case_dir: &str,
    include_history: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let surface_l = model.surface_l_lsun();
    d.set_item("model_number", model.model_number)?;
    d.set_item("star_age", model.star_age)?;
    d.set_item("star_age_old", model.star_age_old)?;
    d.set_item("time", model.star_age * SECYER)?;
    d.set_item("dt", model.dt)?;
    d.set_item("dt_next", model.dt_next)?;
    d.set_item("mstar", model.mstar)?;
    d.set_item("m_center", model.m_center)?;
    d.set_item("xmstar", model.xmstar)?;
    d.set_item("num_conv_boundaries", model.aux.mixing.conv_bdy_loc.len())?;
    d.set_item(
        "conv_bdy_loc",
        PyList::new(py, &model.aux.mixing.conv_bdy_loc)?,
    )?;
    d.set_item(
        "top_conv_bdy",
        PyList::new(py, &model.aux.mixing.top_conv_bdy)?,
    )?;
    d.set_item("star_mass", model.mstar / M_SUN)?;
    d.set_item("mstar_dot", model.mstar_dot)?;
    d.set_item("star_mdot", model.mstar_dot / (M_SUN / SECYER))?;
    d.set_item("teff", model.teff)?;
    d.set_item("surface_l_lsun", surface_l)?;
    d.set_item("log_surface_luminosity", safe_log10(surface_l))?;
    d.set_item("surface_r_rsun", model.surface_r_rsun())?;
    d.set_item("total_energy", model.aux.total_energy)?;
    d.set_item("total_energy_end", model.aux.step_hist.total_energy_end)?;
    d.set_item("cumulative_energy_error", model.aux.cumulative_energy_error)?;
    d.set_item("nz", model.nz)?;
    d.set_item("species", model.species)?;
    d.set_item("case_dir", case_dir)?;
    set_star_control_scalars(&d, model)?;
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
    if !model.controls.rsp_flag || include_history {
        set_star_report_scalars(py, &d, model)?;
        set_star_radial_arrays(py, &d, model)?;
    }
    if model.controls.use_other_wind && model.nz > 0 {
        if let Some(x) = surface_iso_frac(model, "h1") {
            d.set_item("surface_h1", x)?;
        }
        if let Some(y) = surface_iso_frac(model, "he4") {
            d.set_item("surface_he4", y)?;
        }
    }
    Ok(d)
}

fn apply_star_updates(
    model: &mut StarModel,
    updates: &Bound<'_, PyDict>,
) -> PyResult<()> {
    for (key_obj, val) in updates.iter() {
        let key: String = key_obj.extract()?;
        match key.as_str() {
            "x_ctrl" => {
                let xs: Vec<f64> = val.extract()?;
                let n = xs.len().min(model.controls.x_ctrl.len());
                model.controls.x_ctrl[..n].copy_from_slice(&xs[..n]);
            }
            "x_integer_ctrl" => {
                let xs: Vec<i64> = val.extract()?;
                let n = xs.len().min(model.controls.x_integer_ctrl.len());
                model.controls.x_integer_ctrl[..n].copy_from_slice(&xs[..n]);
            }
            "x_logical_ctrl" => {
                let xs: Vec<bool> = val.extract()?;
                let n = xs.len().min(model.controls.x_logical_ctrl.len());
                model.controls.x_logical_ctrl[..n].copy_from_slice(&xs[..n]);
            }
            "x_character_ctrl" => {
                let xs: Vec<String> = val.extract()?;
                let n = xs.len().min(model.controls.x_character_ctrl.len());
                for (dst, src) in
                    model.controls.x_character_ctrl[..n].iter_mut().zip(xs)
                {
                    *dst = src;
                }
            }
            "dt_next" => {
                model.dt_next = val.extract::<f64>()?;
            }
            "atm_t_tau_relation" => {
                let raw: String = val.extract()?;
                model.controls.atm_t_tau_relation =
                    parse_atm_t_tau_relation(&raw)?;
            }
            "reimers_scaling_factor" => {
                model.controls.reimers_scaling_factor = val.extract::<f64>()?;
            }
            "blocker_scaling_factor" => {
                model.controls.blocker_scaling_factor = val.extract::<f64>()?;
            }
            "do_element_diffusion" => {
                model.controls.do_element_diffusion = val.extract::<bool>()?;
            }
            "mass_change" => {
                model.controls.mass_change = val.extract::<f64>()?;
            }
            "mixing_length_alpha" => {
                model.controls.mixing_length_alpha = val.extract::<f64>()?;
            }
            "max_timestep" => {
                model.controls.ts_stop.max_timestep = val.extract::<f64>()?;
            }
            "initial_mass" => {
                model.controls.initial_mass = val.extract::<f64>()?;
            }
            "log_l_upper_limit" => {
                model.controls.log_l_upper_limit =
                    val.extract::<Option<f64>>()?;
            }
            "teff_lower_limit" => {
                model.controls.teff_lower_limit =
                    val.extract::<Option<f64>>()?;
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unsupported star extras update key '{other}'"
                )));
            }
        }
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
            } else if !updates.is_none() {
                return Err(PyValueError::new_err(format!(
                    "hook returned 'updates' of type {}; expected a dict",
                    updates.get_type().name()?
                )));
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

/// Callable attached to `other_overshooting_scheme` snapshots so Python hooks
/// can evaluate `eval_over_bdy_params(i_bdy, f0)` directly via
/// `OverBdyAnchorCache` for arbitrary tapered `f0` values.
#[pyclass]
struct PyOverBdyEvaluator {
    anchor: OverBdyAnchorCache,
}

#[pymethods]
impl PyOverBdyEvaluator {
    fn __call__(
        &self,
        i_bdy: usize,
        f0: f64,
    ) -> PyResult<(usize, f64, f64, f64)> {
        if i_bdy != self.anchor.boundary_index() {
            return Err(PyValueError::new_err(format!(
                "eval_over_bdy_params called for boundary {i_bdy} != active boundary {}",
                self.anchor.boundary_index()
            )));
        }
        self.anchor.eval_at(f0).map_err(to_py_err)
    }
}

struct PyOtherOvershootingSchemeHook {
    py_obj: Arc<Py<PyAny>>,
    state: Arc<PyAdapterState>,
}

impl OtherOvershootingSchemeHook for PyOtherOvershootingSchemeHook {
    fn eval_overshooting(
        &self,
        model: &StarModel,
        i: usize,
        j: usize,
        d_out: &mut [f64],
        vc_out: &mut [f64],
    ) -> Result<Option<(usize, usize)>, StarError> {
        self.state.take_pending_star_err()?;
        let anchor = OverBdyAnchorCache::from_model(model, i)?;
        let n_bdy = model.aux.mixing.conv_bdy_loc.len();
        let set_f0 = model
            .controls
            .mix_wind
            .overshoot_sets
            .get(j)
            .map_or(0.0, |s| s.f0);
        let mut conv_bdy_hp = Vec::with_capacity(n_bdy);
        let mut over_bdy_k = Vec::with_capacity(n_bdy);
        let mut over_bdy_r = Vec::with_capacity(n_bdy);
        let mut over_bdy_d = Vec::with_capacity(n_bdy);
        let mut over_bdy_vc = Vec::with_capacity(n_bdy);
        for b in 0..n_bdy {
            let hp = if b == i {
                anchor.hp_cb()
            } else {
                eval_conv_bdy_hp(model, b).unwrap_or(0.0)
            };
            let (k_ob, r_ob, d_ob, vc_ob) = if b == i {
                anchor.eval_at(set_f0)?
            } else {
                eval_over_bdy_params(model, b, set_f0)
                    .unwrap_or((0, 0.0, 0.0, 0.0))
            };
            conv_bdy_hp.push(hp);
            over_bdy_k.push(k_ob);
            over_bdy_r.push(r_ob);
            over_bdy_d.push(d_ob);
            over_bdy_vc.push(vc_ob);
        }

        let nz = model.nz;
        let case_dir = self.state.staged_dir_string();
        Python::attach(|py| {
            let call_res = (|| -> PyResult<Option<(isize, isize, Vec<f64>, Vec<f64>)>> {
                let snap = build_star_snapshot(py, model, &case_dir, false)?;
                snap.set_item("conv_bdy_hp", PyList::new(py, &conv_bdy_hp)?)?;
                snap.set_item("over_bdy_k", PyList::new(py, &over_bdy_k)?)?;
                snap.set_item("over_bdy_r", PyList::new(py, &over_bdy_r)?)?;
                snap.set_item("over_bdy_d", PyList::new(py, &over_bdy_d)?)?;
                snap.set_item("over_bdy_vc", PyList::new(py, &over_bdy_vc)?)?;
                let eval_fn = Py::new(py, PyOverBdyEvaluator { anchor })?;
                snap.set_item("eval_over_bdy_params", eval_fn)?;
                let res = self.py_obj.bind(py).call_method1(
                    "other_overshooting_scheme",
                    (snap, i, j),
                )?;
                if res.is_none() {
                    return Ok(None);
                }
                Ok(Some(res.extract()?))
            })()
            .map_err(|e| {
                StarError::InvalidInput(format!(
                    "other_overshooting_scheme: {}",
                    format_py_err(py, &e)
                ))
            })?;
            let Some((ka, kb, d_vec, vc_vec)) = call_res else {
                return Ok(None);
            };
            if ka < 0 || kb < 0 {
                return Err(StarError::InvalidInput(format!(
                    "other_overshooting_scheme returned negative index bounds (k_a={ka}, k_b={kb})"
                )));
            }
            if d_vec.len() != nz || vc_vec.len() != nz {
                return Err(StarError::InvalidInput(format!(
                    "other_overshooting_scheme returned arrays len ({}, {}) != nz={nz}",
                    d_vec.len(),
                    vc_vec.len()
                )));
            }
            d_out[..nz].copy_from_slice(&d_vec);
            vc_out[..nz].copy_from_slice(&vc_vec);
            Ok(Some((ka as usize, kb as usize)))
        })
    }
}

struct PyOtherAdjustMltGradtFractionHook {
    py_obj: Arc<Py<PyAny>>,
    state: Arc<PyAdapterState>,
}

impl OtherAdjustMltGradtFractionHook for PyOtherAdjustMltGradtFractionHook {
    fn apply(&self, model: &mut StarModel) -> Result<(), StarError> {
        self.state.take_pending_star_err()?;
        let nz = model.nz;
        let case_dir = self.state.staged_dir_string();
        Python::attach(|py| {
            let snap = build_star_snapshot(py, model, &case_dir, false)
                .map_err(|e| StarError::InvalidInput(format_py_err(py, &e)))?;
            let res = self
                .py_obj
                .bind(py)
                .call_method1("other_adjust_mlt_gradt_fraction", (snap,))
                .map_err(|e| {
                    StarError::InvalidInput(format!(
                        "other_adjust_mlt_gradt_fraction: {}",
                        format_py_err(py, &e)
                    ))
                })?;
            if res.is_none() {
                return Ok(());
            }
            let vals: Vec<f64> = res.extract().map_err(|e| {
                StarError::InvalidInput(format!(
                    "other_adjust_mlt_gradt_fraction: {}",
                    format_py_err(py, &e)
                ))
            })?;
            if vals.len() != nz {
                return Err(StarError::InvalidInput(format!(
                    "other_adjust_mlt_gradt_fraction returned {} values for nz = {nz}",
                    vals.len()
                )));
            }
            model.aux.mixing.adjust_mlt_gradt_fraction[..nz]
                .copy_from_slice(&vals);
            Ok(())
        })
    }
}

/// Thread-safe adapter forwarding [`StarExtras`] trait callbacks into a Python
/// test-side extras object while avoiding per-step whole-model clones (`#148`).
#[allow(clippy::struct_excessive_bools)]
pub struct PyStarExtrasAdapter {
    py_obj: Arc<Py<PyAny>>,
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
    has_other_overshooting_scheme: bool,
    has_other_adjust_mlt_gradt_fraction: bool,
    has_other_build_initial_model: bool,
    has_other_rsp_build_model: bool,
    has_other_rsp_linear_analysis: bool,
    has_extra_history_columns: bool,
    has_extra_profile_columns: bool,
    has_extra_history_header_items: bool,
    has_extra_profile_header_items: bool,
    state: Arc<PyAdapterState>,
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
            has_other_overshooting_scheme: b
                .hasattr("other_overshooting_scheme")?,
            has_other_adjust_mlt_gradt_fraction: b
                .hasattr("other_adjust_mlt_gradt_fraction")?,
            has_other_build_initial_model: b
                .hasattr("other_build_initial_model")?,
            has_other_rsp_build_model: b.hasattr("other_rsp_build_model")?,
            has_other_rsp_linear_analysis: b
                .hasattr("other_rsp_linear_analysis")?,
            has_extra_history_columns: b.hasattr("extra_history_columns")?,
            has_extra_profile_columns: b.hasattr("extra_profile_columns")?,
            has_extra_history_header_items: b
                .hasattr("extra_history_header_items")?,
            has_extra_profile_header_items: b
                .hasattr("extra_profile_header_items")?,
            py_obj: Arc::new(obj),
            state: Arc::new(PyAdapterState::default()),
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

    fn call_star_snapshot_extract<T, F>(
        &self,
        enabled: bool,
        method: &str,
        model: &StarModel,
        extract: F,
    ) -> Result<Vec<T>, StarError>
    where
        F: FnOnce(&Bound<'_, PyAny>) -> PyResult<Vec<T>>,
    {
        if !enabled {
            return Ok(Vec::new());
        }
        let case_dir = self.state.staged_dir_string();
        Python::attach(|py| {
            let snap = build_star_snapshot(py, model, &case_dir, false)
                .map_err(|e| StarError::InvalidInput(format_py_err(py, &e)))?;
            let res = self
                .py_obj
                .bind(py)
                .call_method1(method, (snap,))
                .map_err(|e| StarError::InvalidInput(format_py_err(py, &e)))?;
            if res.is_none() {
                return Ok(Vec::new());
            }
            extract(&res)
                .map_err(|e| StarError::InvalidInput(format_py_err(py, &e)))
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
        install_star_aux_hooks(self, model);
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
            } else if !res.is_none() {
                return Err(StarError::InvalidInput(String::from(
                    "extras_controls: hook must return None or a dict of updates",
                )));
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
            } else if !res.is_none() {
                return Err(StarError::InvalidInput(String::from(
                    "extras_startup: hook must return None or a dict of updates",
                )));
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

    fn extra_profile_columns(
        &self,
        model: &StarModel,
    ) -> Result<Vec<(String, Vec<f64>)>, StarError> {
        self.call_star_snapshot_extract(
            self.has_extra_profile_columns,
            "extra_profile_columns",
            model,
            |res| res.extract::<Vec<(String, Vec<f64>)>>(),
        )
    }

    fn extra_history_header_items(
        &self,
        model: &StarModel,
    ) -> Result<Vec<(String, f64)>, StarError> {
        self.call_star_snapshot_extract(
            self.has_extra_history_header_items,
            "extra_history_header_items",
            model,
            |res| res.extract::<Vec<(String, f64)>>(),
        )
    }

    fn extra_profile_header_items(
        &self,
        model: &StarModel,
    ) -> Result<Vec<(String, f64)>, StarError> {
        self.call_star_snapshot_extract(
            self.has_extra_profile_header_items,
            "extra_profile_header_items",
            model,
            |res| res.extract::<Vec<(String, f64)>>(),
        )
    }

    fn other_overshooting_scheme_handle(
        &self,
    ) -> Option<OtherOvershootingSchemeHandle> {
        self.has_other_overshooting_scheme.then(|| {
            OtherOvershootingSchemeHandle::new(PyOtherOvershootingSchemeHook {
                py_obj: Arc::clone(&self.py_obj),
                state: Arc::clone(&self.state),
            })
        })
    }

    fn other_adjust_mlt_gradt_fraction_handle(
        &self,
    ) -> Option<OtherAdjustMltGradtFractionHandle> {
        self.has_other_adjust_mlt_gradt_fraction.then(|| {
            OtherAdjustMltGradtFractionHandle::new(
                PyOtherAdjustMltGradtFractionHook {
                    py_obj: Arc::clone(&self.py_obj),
                    state: Arc::clone(&self.state),
                },
            )
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
    d.set_item(
        format!("{prefix}_omega_avg_surf"),
        eval_surf_avg_rotation_info(star).omega,
    )?;
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
    d.set_item("rl", PyList::new(py, bin.rl_by_star())?)?;
    d.set_item(
        "rl_relative_gap",
        PyList::new(py, bin.rl_relative_gap_by_star())?,
    )?;
    d.set_item("point_mass_i", bin.point_mass_i)?;
    d.set_item("donor_i", bin.donor_i)?;
    d.set_item("ignore_rlof_flag", bin.ignore_rlof_flag)?;
    d.set_item("eq_initial_bh_mass", bin.eq_initial_bh_mass)?;
    let map_by_star = |pair: [f64; 2]| -> [f64; 2] {
        if bin.donor_i == 1 {
            pair
        } else {
            [pair[1], pair[0]]
        }
    };
    let mdot = &bin.last_mdot;
    d.set_item(
        "component_mdot",
        PyList::new(py, map_by_star(mdot.component_mdot))?,
    )?;
    d.set_item(
        "mdot_wind_transfer",
        PyList::new(py, map_by_star(mdot.mdot_wind_transfer))?,
    )?;
    d.set_item(
        "mdot_system_transfer",
        PyList::new(py, map_by_star(mdot.mdot_system_transfer))?,
    )?;
    d.set_item(
        "mdot_system_wind",
        PyList::new(py, map_by_star(mdot.mdot_system_wind))?,
    )?;
    d.set_item(
        "wind_xfer_fraction",
        PyList::new(py, map_by_star(mdot.wind_xfer_fraction))?,
    )?;
    d.set_item("mdot_edd", mdot.mdot_edd)?;
    d.set_item("mdot_edd_eta", mdot.mdot_edd_eta)?;
    d.set_item("accretion_luminosity", mdot.accretion_luminosity)?;
    d.set_item("donor_mass_msun", bin.donor.mstar / M_SUN)?;
    d.set_item("accretor_mass_msun", bin.accretor_mass_g() / M_SUN)?;
    d.set_item("case_dir", case_dir)?;
    set_user_controls(&d, &bin.donor.controls)?;
    if let Some(star1) = bin.star_slot_ref(1) {
        set_binary_star_scalars(&d, "s1", star1)?;
    }
    if let Some(star2) = bin.star_slot_ref(2) {
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
    if let Some(val) = updates.get_item("eq_initial_bh_mass")? {
        bin.set_eq_initial_bh_mass(val.extract::<f64>()?);
    }
    if let Some(val) = updates.get_item("m1")? {
        bin.set_m1(val.extract::<f64>()?).map_err(to_py_err)?;
    }
    if let Some(val) = updates.get_item("m2")? {
        bin.set_m2(val.extract::<f64>()?).map_err(to_py_err)?;
    }
    if let Some(val) = updates.get_item("period_eccentricity")? {
        let (p_sec, ecc) = val.extract::<(f64, f64)>()?;
        bin.set_period_eccentricity(p_sec, ecc).map_err(to_py_err)?;
    }
    if let Some(val) = updates.get_item("separation_eccentricity")? {
        let (sep_cm, ecc) = val.extract::<(f64, f64)>()?;
        bin.set_separation_eccentricity(sep_cm, ecc)
            .map_err(to_py_err)?;
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
            } else if !updates.is_none() {
                return Err(PyValueError::new_err(format!(
                    "hook returned 'updates' of type {}; expected a dict",
                    updates.get_type().name()?
                )));
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
        "redo" => Ok(BinaryExtrasAction::Redo),
        "retry" => Ok(BinaryExtrasAction::Retry),
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
    startup_method: Option<&'static str>,
    after_evolve_method: Option<&'static str>,
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
        let startup_method = if b.hasattr("extras_binary_startup")? {
            Some("extras_binary_startup")
        } else if b.hasattr("startup")? {
            Some("startup")
        } else {
            None
        };
        let after_evolve_method = if b.hasattr("extras_binary_after_evolve")? {
            Some("extras_binary_after_evolve")
        } else if b.hasattr("after_evolve")? {
            Some("after_evolve")
        } else {
            None
        };
        Ok(Self {
            startup_method,
            after_evolve_method,
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

    fn startup(
        &mut self,
        bin: &mut BinarySystem,
        restart: bool,
    ) -> BinaryExtrasAction {
        if let Some(msg) = self.state.take_pending_msg() {
            return BinaryExtrasAction::Error(msg);
        }
        let Some(method) = self.startup_method else {
            return BinaryExtrasAction::KeepGoing;
        };
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
            match self.py_obj.bind(py).call_method1(method, (snap, restart)) {
                Ok(res) => {
                    parse_binary_action(&res, "extras_binary_startup", bin)
                        .unwrap_or_else(|e| {
                            BinaryExtrasAction::Error(format!(
                                "{method}: {}",
                                format_py_err(py, &e)
                            ))
                        })
                }
                Err(e) => BinaryExtrasAction::Error(format!(
                    "{method}: {}",
                    format_py_err(py, &e)
                )),
            }
        })
    }

    fn after_evolve(
        &mut self,
        bin: &mut BinarySystem,
    ) -> Result<(), StarError> {
        self.state.take_pending_star_err()?;
        let Some(method) = self.after_evolve_method else {
            return Ok(());
        };
        let case_dir = self.state.staged_dir_string();
        Python::attach(|py| {
            let snap = build_binary_snapshot(py, bin, &case_dir)
                .map_err(|e| StarError::InvalidInput(format_py_err(py, &e)))?;
            self.py_obj
                .bind(py)
                .call_method1(method, (snap,))
                .map(|_| ())
                .map_err(|e| {
                    StarError::InvalidInput(format!(
                        "{method}: {}",
                        format_py_err(py, &e)
                    ))
                })
        })
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

/// Builds the general MESA `star_info` extras snapshot dictionary for a
/// [`PyStarModel`] without running a full trace (`#148`, `#231`).
///
/// # Errors
/// Returns [`PyErr`] if Python dictionary or NumPy array construction fails.
#[pyfunction]
#[pyo3(signature = (model, case_dir = "", include_history = true))]
pub fn star_extras_snapshot<'py>(
    py: Python<'py>,
    model: PyRef<'_, PyStarModel>,
    case_dir: &str,
    include_history: bool,
) -> PyResult<Bound<'py, PyDict>> {
    build_star_snapshot(py, &model.inner, case_dir, include_history)
}

/// Applies a general MESA star extras update dictionary onto a [`PyStarModel`]
/// in place (`#148`, `#231`).
///
/// # Errors
/// Returns [`PyValueError`] when `updates` contains an unrecognized key or an
/// invalid control value (such as an unknown `atm_t_tau_relation` string).
#[pyfunction]
pub fn star_apply_extras_updates(
    mut model: PyRefMut<'_, PyStarModel>,
    updates: &Bound<'_, PyDict>,
) -> PyResult<()> {
    apply_star_updates(&mut model.inner, updates)
}

/// Builds the general MESA `binary_info` extras snapshot dictionary for a
/// [`PyBinarySystem`] without running a full trace (`#148`, `#231`).
///
/// # Errors
/// Returns [`PyErr`] if Python dictionary construction fails.
#[pyfunction]
#[pyo3(signature = (bin, case_dir = ""))]
pub fn binary_extras_snapshot<'py>(
    py: Python<'py>,
    bin: PyRef<'_, PyBinarySystem>,
    case_dir: &str,
) -> PyResult<Bound<'py, PyDict>> {
    build_binary_snapshot(py, &bin.inner, case_dir)
}

/// Registers test-side extras context types and full `GYRE` mode helpers on `flexmesa._rust`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRspBuildContext>()?;
    m.add_function(wrap_pyfunction!(star_solve_gyre_mesa_modes_full, m)?)?;
    m.add_function(wrap_pyfunction!(star_extras_snapshot, m)?)?;
    m.add_function(wrap_pyfunction!(star_apply_extras_updates, m)?)?;
    m.add_function(wrap_pyfunction!(binary_extras_snapshot, m)?)?;
    Ok(())
}
