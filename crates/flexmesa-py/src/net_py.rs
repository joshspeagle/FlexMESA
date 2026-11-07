// SPDX-License-Identifier: LGPL-3.0-or-later
//! Tier B/C `PyO3` bindings for `flexmesa-net` (nuclear reaction networks,
//! analytical Jacobians, `approx21`, and stiff one-zone burners).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use flexmesa_chem::NuclideCatalog;
use flexmesa_eos::{EosConfig, HelmTable};
use flexmesa_net::{
    builtin_net_files, burn_1_zone, burn_1_zone_const_p, net_get,
    Burn1ZoneInput, BurnConstPInput, NetError, NetGeneralInfo,
};
use flexmesa_num::RosenbrockVariant;
use flexmesa_rates::ScreeningMode;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn to_py_err(e: NetError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn cached_net(name: &str) -> PyResult<Arc<NetGeneralInfo>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<NetGeneralInfo>>>> =
        OnceLock::new();
    let map_mutex = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = map_mutex.lock().unwrap();
    if let Some(g) = map.get(name) {
        return Ok(Arc::clone(g));
    }
    let g = Arc::new(NetGeneralInfo::from_name(name).map_err(to_py_err)?);
    map.insert(name.to_string(), Arc::clone(&g));
    Ok(g)
}

fn parse_screening(mode: &str) -> PyResult<ScreeningMode> {
    match mode.to_ascii_lowercase().as_str() {
        "no_screening" | "none" | "off" => Ok(ScreeningMode::NoScreening),
        "extended" | "classic" | "default" => Ok(ScreeningMode::Extended),
        "salpeter" => Ok(ScreeningMode::Salpeter),
        "chugunov" => Ok(ScreeningMode::Chugunov),
        other => Err(PyValueError::new_err(format!(
            "unknown screening_mode: {other}"
        ))),
    }
}

fn parse_rosenbrock_solver(name: &str) -> PyResult<RosenbrockVariant> {
    match name.to_ascii_lowercase().as_str() {
        "rodas4" | "rodas4_solver" | "default" => Ok(RosenbrockVariant::Rodas4),
        "rodasp" | "rodasp_solver" => Ok(RosenbrockVariant::Rodasp),
        "ros3p" | "ros3p_solver" => Ok(RosenbrockVariant::Ros3p),
        "ros3pl" | "ros3pl_solver" => Ok(RosenbrockVariant::Ros3pl),
        other => Err(PyValueError::new_err(format!(
            "unknown Rosenbrock solver: {other}"
        ))),
    }
}

/// Lists all built-in `.net` nuclear network definition filenames.
#[pyfunction]
pub fn net_list_networks() -> Vec<String> {
    let mut names: Vec<String> = builtin_net_files()
        .keys()
        .map(|s| (*s).to_string())
        .collect();
    names.sort();
    names
}

/// Returns `(num_isos, num_reactions, chem_id, iso_names, reaction_names)` for a network.
#[pyfunction]
pub fn net_info(
    net_name: &str,
) -> PyResult<(usize, usize, Vec<usize>, Vec<String>, Vec<String>)> {
    let g = cached_net(net_name)?;
    let cat = NuclideCatalog::builtin_ref();
    let iso_names: Vec<String> = g
        .chem_id
        .iter()
        .map(|&cid| {
            cat.get_by_id(cid)
                .map_or_else(|| format!("cid_{cid}"), |n| n.name.clone())
        })
        .collect();
    let rxn_names: Vec<String> =
        g.reactions.iter().map(|r| r.name.clone()).collect();
    Ok((
        g.num_isos,
        g.num_reactions,
        g.chem_id.clone(),
        iso_names,
        rxn_names,
    ))
}

/// Evaluates `net_get` at a single thermodynamic and composition state.
///
/// Returns `(eps_nuc, d_eps_nuc_drho, d_eps_nuc_dt, d_eps_nuc_dx, dxdt,
/// d_dxdt_drho, d_dxdt_dt, d_dxdt_dx, eps_nuc_categories, eps_neu_total,
/// rate_raw, rate_screened)`.
#[pyfunction]
#[pyo3(signature = (
    net_name, x, temp, log_t, rho, log_rho, abar, zbar, z2bar, ye,
    eta=0.0, d_eta_dlnt=0.0, d_eta_dlnrho=0.0,
    rate_factors=None, weak_rate_factor=1.0,
    screening_mode="extended", skip_jacobian=false
))]
pub fn net_eval<'py>(
    py: Python<'py>,
    net_name: &str,
    x: PyReadonlyArray1<'py, f64>,
    temp: f64,
    log_t: f64,
    rho: f64,
    log_rho: f64,
    abar: f64,
    zbar: f64,
    z2bar: f64,
    ye: f64,
    eta: f64,
    d_eta_dlnt: f64,
    d_eta_dlnrho: f64,
    rate_factors: Option<PyReadonlyArray1<'py, f64>>,
    weak_rate_factor: f64,
    screening_mode: &str,
    skip_jacobian: bool,
) -> PyResult<(
    f64,
    f64,
    f64,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    f64,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let g = cached_net(net_name)?;
    let x_slice = x.as_slice()?;
    let sc = parse_screening(screening_mode)?;
    let rf_vec = match rate_factors {
        Some(rf) => rf.as_slice()?.to_vec(),
        None => vec![1.0; g.num_reactions],
    };
    let res = net_get(
        &g,
        skip_jacobian,
        x_slice,
        temp,
        log_t,
        rho,
        log_rho,
        abar,
        zbar,
        z2bar,
        ye,
        eta,
        d_eta_dlnt,
        d_eta_dlnrho,
        &rf_vec,
        weak_rate_factor,
        sc,
    )
    .map_err(to_py_err)?;
    let s = g.num_isos;
    let d_dxdt_dx = PyArray1::from_vec(py, res.d_dxdt_dx).reshape([s, s])?;
    Ok((
        res.eps_nuc,
        res.d_eps_nuc_drho,
        res.d_eps_nuc_dt,
        PyArray1::from_vec(py, res.d_eps_nuc_dx),
        PyArray1::from_vec(py, res.dxdt),
        PyArray1::from_vec(py, res.d_dxdt_drho),
        PyArray1::from_vec(py, res.d_dxdt_dt),
        d_dxdt_dx,
        PyArray1::from_slice(py, &res.eps_nuc_categories),
        res.eps_neu_total,
        PyArray1::from_vec(py, res.rate_raw),
        PyArray1::from_vec(py, res.rate_screened),
    ))
}

/// Integrates a one-zone burn at constant or prescribed $(T(t), \rho(t))$ (`burn_1_zone`).
#[pyfunction]
#[pyo3(signature = (
    net_name, t_start, t_end, starting_x, times, log10_ts, log10_rhos, etas,
    rate_factors=None, weak_rate_factor=1.0, screening_mode="extended",
    stptry_in=0.0, max_steps=10000, eps=1.0e-8, odescal=1.0e-12
))]
pub fn net_burn_1_zone<'py>(
    py: Python<'py>,
    net_name: &str,
    t_start: f64,
    t_end: f64,
    starting_x: PyReadonlyArray1<'py, f64>,
    times: PyReadonlyArray1<'py, f64>,
    log10_ts: PyReadonlyArray1<'py, f64>,
    log10_rhos: PyReadonlyArray1<'py, f64>,
    etas: PyReadonlyArray1<'py, f64>,
    rate_factors: Option<PyReadonlyArray1<'py, f64>>,
    weak_rate_factor: f64,
    screening_mode: &str,
    stptry_in: f64,
    max_steps: usize,
    eps: f64,
    odescal: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    f64,
    f64,
    usize,
    usize,
    usize,
    usize,
    usize,
)> {
    let g = cached_net(net_name)?;
    let sc = parse_screening(screening_mode)?;
    let rf_vec = match rate_factors {
        Some(rf) => rf.as_slice()?.to_vec(),
        None => vec![1.0; g.num_reactions],
    };
    let input = Burn1ZoneInput {
        t_start,
        t_end,
        starting_x: starting_x.as_slice()?,
        times: times.as_slice()?,
        log10_ts: log10_ts.as_slice()?,
        log10_rhos: log10_rhos.as_slice()?,
        etas: etas.as_slice()?,
        dxdt_source_term: None,
        rate_factors: &rf_vec,
        weak_rate_factor,
        reaction_qs: &[],
        reaction_neu_qs: &[],
        screening_mode: sc,
        stptry_in,
        max_steps,
        eps,
        odescal,
    };
    let ht = HelmTable::builtin();
    let eos_cfg = EosConfig::default();
    let out = burn_1_zone(&g, ht, &eos_cfg, &input).map_err(to_py_err)?;
    Ok((
        PyArray1::from_vec(py, out.ending_x),
        PyArray1::from_slice(py, &out.eps_nuc_categories),
        out.avg_eps_nuc,
        out.eps_neu_total,
        out.nfcn,
        out.njac,
        out.nstep,
        out.naccpt,
        out.nrejct,
    ))
}

/// Integrates a constant-pressure one-zone burn (`burn_1_zone_const_p`).
#[pyfunction]
#[pyo3(signature = (
    net_name, starting_temp, starting_x, t_end, pressure,
    solver="rodas4", clip=false, rate_factors=None, weak_rate_factor=1.0,
    screening_mode="extended", h0=1.0e-6, max_step_size=1.0,
    max_steps=1000, rtol=1.0e-7, atol=1.0e-8, x_min=-1.0e-10
))]
pub fn net_burn_1_zone_const_p<'py>(
    py: Python<'py>,
    net_name: &str,
    starting_temp: f64,
    starting_x: PyReadonlyArray1<'py, f64>,
    t_end: f64,
    pressure: f64,
    solver: &str,
    clip: bool,
    rate_factors: Option<PyReadonlyArray1<'py, f64>>,
    weak_rate_factor: f64,
    screening_mode: &str,
    h0: f64,
    max_step_size: f64,
    max_steps: usize,
    rtol: f64,
    atol: f64,
    x_min: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    f64,
    f64,
    f64,
    f64,
    f64,
    usize,
    usize,
    usize,
    usize,
    usize,
)> {
    let g = cached_net(net_name)?;
    let sc = parse_screening(screening_mode)?;
    let ros = parse_rosenbrock_solver(solver)?;
    let rf_vec = match rate_factors {
        Some(rf) => rf.as_slice()?.to_vec(),
        None => vec![1.0; g.num_reactions],
    };
    let input = BurnConstPInput {
        solver: ros,
        starting_temp,
        starting_x: starting_x.as_slice()?,
        clip,
        t_end,
        pressure,
        rate_factors: &rf_vec,
        weak_rate_factor,
        reaction_qs: &[],
        reaction_neu_qs: &[],
        screening_mode: sc,
        h0,
        max_step_size,
        max_steps,
        rtol,
        atol,
        x_min,
    };
    let ht = HelmTable::builtin();
    let eos_cfg = EosConfig::default();
    let out =
        burn_1_zone_const_p(&g, ht, &eos_cfg, &input).map_err(to_py_err)?;
    Ok((
        PyArray1::from_vec(py, out.ending_x),
        out.ending_temp,
        out.ending_rho,
        out.ending_lns,
        out.initial_rho,
        out.initial_lns,
        out.nfcn,
        out.njac,
        out.nstep,
        out.naccpt,
        out.nrejct,
    ))
}

/// Registers `flexmesa-net` functions on the `flexmesa._rust` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(net_list_networks, m)?)?;
    m.add_function(wrap_pyfunction!(net_info, m)?)?;
    m.add_function(wrap_pyfunction!(net_eval, m)?)?;
    m.add_function(wrap_pyfunction!(net_burn_1_zone, m)?)?;
    m.add_function(wrap_pyfunction!(net_burn_1_zone_const_p, m)?)?;
    Ok(())
}
