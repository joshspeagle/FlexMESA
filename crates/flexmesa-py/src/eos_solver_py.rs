// SPDX-License-Identifier: LGPL-3.0-or-later
//! `PyO3` bindings for `flexmesa-eos` Wave 3a.3 (`eosDT_get` multi-component
//! blend and `PT`/`DE` inverse solvers).

use flexmesa_eos::{
    eos_dt_get, eos_dt_get_rho, eos_dt_get_t, eos_gamma_de_get, eos_pt_get,
    EosConfig, HelmTable,
};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::eos_pc_skye_py::parse_mixing_rule;
use crate::eos_py::{into_py_eos_tuple, PyEosTuple};

type PyPtTuple<'py> = (
    f64,
    f64,
    f64,
    f64,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

type PyInvTuple<'py> = (
    f64,
    f64,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

fn build_eos_config(
    use_skye: bool,
    use_pc: bool,
    pc_use_gamma_limit: bool,
    skye_solid_mixing_rule: &str,
) -> PyResult<EosConfig> {
    let mut cfg = EosConfig {
        use_skye,
        use_pc,
        ..EosConfig::default()
    };
    cfg.pc.pc_use_gamma_limit_instead_of_t = pc_use_gamma_limit;
    cfg.skye.skye_solid_mixing_rule =
        parse_mixing_rule(skye_solid_mixing_rule)?;
    Ok(cfg)
}

/// Evaluates the unified top-level `eosDT_get` (`Skye` / `PC` + `HELM` + `ideal`
/// blend) at `(rho, temp, chem_id, xa)`.
#[pyfunction]
#[pyo3(signature = (
    rho,
    temp,
    chem_id,
    xa,
    use_skye=true,
    use_pc=false,
    pc_use_gamma_limit=true,
    skye_solid_mixing_rule="PC"
))]
pub fn eos_eval_dt_get<'py>(
    py: Python<'py>,
    rho: f64,
    temp: f64,
    chem_id: Vec<usize>,
    xa: PyReadonlyArray1<'py, f64>,
    use_skye: bool,
    use_pc: bool,
    pc_use_gamma_limit: bool,
    skye_solid_mixing_rule: &str,
) -> PyResult<PyEosTuple<'py>> {
    let xa_s = xa
        .as_slice()
        .map_err(|_| PyValueError::new_err("xa must be contiguous"))?;
    let cfg = build_eos_config(
        use_skye,
        use_pc,
        pc_use_gamma_limit,
        skye_solid_mixing_rule,
    )?;
    let out = eos_dt_get(HelmTable::builtin(), rho, temp, &chem_id, xa_s, &cfg)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(into_py_eos_tuple(py, &out))
}

/// Evaluates `eosPT_get` at `(pgas, temp, chem_id, xa)`, returning
/// `(rho, log10_rho, dln_rho_dln_pgas_c_t, dln_rho_dln_t_c_pgas, res, d_dlnd,
/// d_dln_t)`.
#[pyfunction]
#[pyo3(signature = (
    pgas,
    temp,
    chem_id,
    xa,
    use_skye=true,
    use_pc=false,
    pc_use_gamma_limit=true,
    skye_solid_mixing_rule="PC"
))]
pub fn eos_eval_pt_get<'py>(
    py: Python<'py>,
    pgas: f64,
    temp: f64,
    chem_id: Vec<usize>,
    xa: PyReadonlyArray1<'py, f64>,
    use_skye: bool,
    use_pc: bool,
    pc_use_gamma_limit: bool,
    skye_solid_mixing_rule: &str,
) -> PyResult<PyPtTuple<'py>> {
    let xa_s = xa
        .as_slice()
        .map_err(|_| PyValueError::new_err("xa must be contiguous"))?;
    let cfg = build_eos_config(
        use_skye,
        use_pc,
        pc_use_gamma_limit,
        skye_solid_mixing_rule,
    )?;
    let pt = eos_pt_get(
        HelmTable::builtin(),
        &chem_id,
        xa_s,
        pgas.log10(),
        temp.log10(),
        &cfg,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let (res, d_dlnd, d_dln_t) = into_py_eos_tuple(py, &pt.eos);
    Ok((
        pt.rho,
        pt.log10_rho,
        pt.dln_rho_dln_pgas_const_t,
        pt.dln_rho_dln_t_const_pgas,
        res,
        d_dlnd,
        d_dln_t,
    ))
}

/// Evaluates `eosDT_get_T` solving for `T` at fixed `rho` and
/// `res[which_other] == other_value`.
#[pyfunction]
#[pyo3(signature = (
    rho,
    which_other,
    other_value,
    chem_id,
    xa,
    log_t_guess=6.0,
    use_skye=true,
    use_pc=false
))]
pub fn eos_eval_dt_get_t<'py>(
    py: Python<'py>,
    rho: f64,
    which_other: isize,
    other_value: f64,
    chem_id: Vec<usize>,
    xa: PyReadonlyArray1<'py, f64>,
    log_t_guess: f64,
    use_skye: bool,
    use_pc: bool,
) -> PyResult<PyInvTuple<'py>> {
    let xa_s = xa
        .as_slice()
        .map_err(|_| PyValueError::new_err("xa must be contiguous"))?;
    let cfg = build_eos_config(use_skye, use_pc, true, "PC")?;
    let inv = eos_dt_get_t(
        HelmTable::builtin(),
        &chem_id,
        xa_s,
        rho.log10(),
        which_other,
        other_value,
        log_t_guess,
        1.0e-12,
        1.0e-12,
        50,
        &cfg,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let (res, d_dlnd, d_dln_t) = into_py_eos_tuple(py, &inv.eos);
    Ok((inv.linear_result, inv.log10_result, res, d_dlnd, d_dln_t))
}

/// Evaluates `eosDT_get_Rho` solving for `rho` at fixed `temp` and
/// `res[which_other] == other_value`.
#[pyfunction]
#[pyo3(signature = (
    temp,
    which_other,
    other_value,
    chem_id,
    xa,
    log_rho_guess=0.0,
    use_skye=true,
    use_pc=false
))]
pub fn eos_eval_dt_get_rho<'py>(
    py: Python<'py>,
    temp: f64,
    which_other: isize,
    other_value: f64,
    chem_id: Vec<usize>,
    xa: PyReadonlyArray1<'py, f64>,
    log_rho_guess: f64,
    use_skye: bool,
    use_pc: bool,
) -> PyResult<PyInvTuple<'py>> {
    let xa_s = xa
        .as_slice()
        .map_err(|_| PyValueError::new_err("xa must be contiguous"))?;
    let cfg = build_eos_config(use_skye, use_pc, true, "PC")?;
    let inv = eos_dt_get_rho(
        HelmTable::builtin(),
        &chem_id,
        xa_s,
        temp.log10(),
        which_other,
        other_value,
        log_rho_guess,
        1.0e-12,
        1.0e-12,
        50,
        &cfg,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let (res, d_dlnd, d_dln_t) = into_py_eos_tuple(py, &inv.eos);
    Ok((inv.linear_result, inv.log10_result, res, d_dlnd, d_dln_t))
}

/// Evaluates the closed-form ideal $\gamma$-law $(\rho, E)$ inverse solver
/// `eos_gamma_DE_get(abar, zbar, rho, energy, gamma)`.
#[pyfunction]
#[pyo3(signature = (abar, zbar, rho, energy, gamma=1.666_666_666_666_666_7))]
pub fn eos_eval_gamma_de_get<'py>(
    py: Python<'py>,
    abar: f64,
    zbar: f64,
    rho: f64,
    energy: f64,
    gamma: f64,
) -> PyInvTuple<'py> {
    let inv = eos_gamma_de_get(abar, zbar, rho, energy, gamma);
    (
        inv.t,
        inv.log10_t,
        PyArray1::from_slice(py, &inv.res),
        PyArray1::from_slice(py, &inv.d_dlnd),
        PyArray1::from_slice(py, &inv.d_dln_t),
    )
}

/// Registers Wave 3a.3 `eosDT_get` blend and inverse solver bindings on the
/// `flexmesa._rust` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(eos_eval_dt_get, m)?)?;
    m.add_function(wrap_pyfunction!(eos_eval_pt_get, m)?)?;
    m.add_function(wrap_pyfunction!(eos_eval_dt_get_t, m)?)?;
    m.add_function(wrap_pyfunction!(eos_eval_dt_get_rho, m)?)?;
    m.add_function(wrap_pyfunction!(eos_eval_gamma_de_get, m)?)?;
    Ok(())
}
