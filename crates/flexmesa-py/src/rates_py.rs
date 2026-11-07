// SPDX-License-Identifier: LGPL-3.0-or-later
//! Tier B `PyO3` bindings for `flexmesa-rates` (`rates` nuclear reaction rates,
//! plasma screening, tabulated weak rates, and Fermi-Dirac electron captures).

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::KEV;
use flexmesa_rates::{
    eval_ecapture_pair, eval_fl_epsnuc_3alf, eval_mui_coulomb,
    eval_reaclib_handle, eval_vs_coulomb, eval_weak_reaction_pair,
    eval_weaklib_one, screen_pair, set_raw_rate, set_raw_rates,
    CoulombMuiOption, CoulombVsOption, RatesCatalog, ScreeningMode, TFactors,
    WeakRateEval,
};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn pack_weak(w: &WeakRateEval) -> [f64; 9] {
    [
        w.lambda,
        w.dlambda_dln_t,
        w.dlambda_dln_rho,
        w.q,
        w.dq_dln_t,
        w.dq_dln_rho,
        w.qneu,
        w.dqneu_dln_t,
        w.dqneu_dln_rho,
    ]
}

fn parse_screen_mode(mode: &str) -> PyResult<ScreeningMode> {
    ScreeningMode::from_name(mode)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

fn parse_mui_opt(opt: &str) -> PyResult<CoulombMuiOption> {
    CoulombMuiOption::from_name(opt)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

fn parse_vs_opt(opt: &str) -> PyResult<CoulombVsOption> {
    CoulombVsOption::from_name(opt)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Look up the 1-based reaction ID (`1..=316`) for a reaction name.
#[pyfunction]
pub fn rates_reaction_id_by_name(name: &str) -> PyResult<usize> {
    RatesCatalog::builtin()
        .reaction_id_by_name(name)
        .ok_or_else(|| {
            PyValueError::new_err(format!("unknown reaction: {name}"))
        })
}

/// Look up the reaction name for a 1-based reaction ID (`1..=316`).
#[pyfunction]
pub fn rates_reaction_name_by_id(ir: usize) -> PyResult<String> {
    RatesCatalog::builtin()
        .reaction_by_id(ir)
        .map(|info| info.name.clone())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Return metadata `(name, op_code, is_reverse, target_handle, ye_exp, rho_exp,
/// category, std_q, std_qneu, weak_lowt_rate)` for 1-based reaction ID `ir`.
#[pyfunction]
pub fn rates_reaction_info(
    ir: usize,
) -> PyResult<(String, u8, bool, String, i8, i8, u8, f64, f64, f64)> {
    let info = RatesCatalog::builtin()
        .reaction_by_id(ir)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        info.name.clone(),
        info.op_code,
        info.is_reverse,
        info.target_handle.clone(),
        info.ye_exp,
        info.rho_exp,
        info.category,
        info.std_q,
        info.std_qneu,
        info.weak_lowt_rate,
    ))
}

/// Return `(is_reverse, terms)` for a REACLIB reaction handle, where each term
/// is `(chapter, rflag, inv_exp, coeffs[7], inv_c1, inv_c2, inv_part[24])`.
#[pyfunction]
pub fn rates_reaclib_terms(
    handle: &str,
) -> PyResult<(bool, Vec<(u8, u8, i8, [f64; 7], f64, f64, [f64; 24])>)> {
    let (is_rev, terms) = RatesCatalog::builtin()
        .reaclib_terms(handle)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let out = terms
        .iter()
        .map(|t| {
            (
                t.chapter, t.rflag, t.inv_exp, t.coeffs, t.inv_c1, t.inv_c2,
                t.inv_part,
            )
        })
        .collect();
    Ok((is_rev, out))
}

/// Evaluate a REACLIB handle `(forward_rate, dforward_dlnT, reverse_rate, dreverse_dlnT)`.
#[pyfunction]
#[pyo3(signature = (handle, t9, check_e_flag=true))]
pub fn rates_eval_reaclib_handle(
    handle: &str,
    t9: f64,
    check_e_flag: bool,
) -> PyResult<(f64, f64, f64, f64)> {
    eval_reaclib_handle(handle, t9, check_e_flag)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Evaluate `set_raw_rate` for a single 1-based reaction ID `ir` (`1..=316`).
#[pyfunction]
#[pyo3(signature = (ir, temp, log_t=None))]
pub fn rates_set_raw_rate(
    ir: usize,
    temp: f64,
    log_t: Option<f64>,
) -> PyResult<f64> {
    let tf = match log_t {
        Some(lt) => TFactors::from_log_t(lt, temp),
        None => TFactors::from_temp(temp),
    };
    set_raw_rate(ir, temp, &tf)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Evaluate `set_raw_rates` for a 1D array of 1-based reaction IDs `irs`.
#[pyfunction]
#[pyo3(signature = (irs, temp, log_t=None))]
pub fn rates_set_raw_rates<'py>(
    py: Python<'py>,
    irs: PyReadonlyArray1<'py, usize>,
    temp: f64,
    log_t: Option<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let ir_slice = irs
        .as_slice()
        .map_err(|_| PyValueError::new_err("irs must be C-contiguous"))?;
    let tf = match log_t {
        Some(lt) => TFactors::from_log_t(lt, temp),
        None => TFactors::from_temp(temp),
    };
    let out = set_raw_rates(ir_slice, temp, &tf)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, out))
}

/// Evaluate `set_raw_rates` across `n_zones` temperatures, returning a `(n_zones, n_rxns)` array.
#[pyfunction]
#[pyo3(signature = (irs, temps, log_ts=None))]
pub fn rates_set_raw_rates_zones<'py>(
    py: Python<'py>,
    irs: PyReadonlyArray1<'py, usize>,
    temps: PyReadonlyArray1<'py, f64>,
    log_ts: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let ir_slice = irs
        .as_slice()
        .map_err(|_| PyValueError::new_err("irs must be C-contiguous"))?;
    let t_slice = temps
        .as_slice()
        .map_err(|_| PyValueError::new_err("temps must be C-contiguous"))?;
    let lt_opt = match &log_ts {
        Some(arr) => Some(arr.as_slice().map_err(|_| {
            PyValueError::new_err("log_ts must be C-contiguous")
        })?),
        None => None,
    };
    let nz = t_slice.len();
    if let Some(lts) = lt_opt {
        if lts.len() != nz {
            return Err(PyValueError::new_err(
                "log_ts and temps must have equal length",
            ));
        }
    }
    let nr = ir_slice.len();
    let mut flat = Vec::with_capacity(nz * nr);
    for z in 0..nz {
        let t = t_slice[z];
        let tf = match lt_opt {
            Some(lts) => TFactors::from_log_t(lts[z], t),
            None => TFactors::from_temp(t),
        };
        let row = set_raw_rates(ir_slice, t, &tf)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        flat.extend_from_slice(&row);
    }
    PyArray1::from_vec(py, flat).reshape([nz, nr])
}

/// Evaluate Fushiki & Lamb (1987) pycnonuclear 3-alpha energy generation rate.
#[pyfunction]
pub fn rates_eval_fl_epsnuc_3alf(t: f64, rho: f64, y: f64, ue: f64) -> f64 {
    eval_fl_epsnuc_3alf(t, rho, y, ue)
}

/// Evaluate an `interp_pm` 1D cubic rate table `(rate, drate_dt)` for reaction ID `ir`.
#[pyfunction]
pub fn rates_eval_rate_table(ir: usize, temp: f64) -> PyResult<(f64, f64)> {
    RatesCatalog::builtin()
        .eval_rate_table(ir, temp)
        .ok_or_else(|| {
            PyValueError::new_err(format!("no rate table for reaction ID {ir}"))
        })
}

/// Evaluate plasma electrostatic screening `(scor, dscor_dt, dscor_dd)` for a reacting pair.
#[pyfunction]
pub fn rates_screen_pair(
    mode: &str,
    temp: f64,
    den: f64,
    zbar: f64,
    abar: f64,
    z2bar: f64,
    a1: f64,
    z1: f64,
    a2: f64,
    z2: f64,
) -> PyResult<(f64, f64, f64)> {
    let sm = parse_screen_mode(mode)?;
    let out = screen_pair(sm, temp, den, zbar, abar, z2bar, a1, z1, a2, z2);
    Ok((out.scor, out.dscor_dt, out.dscor_dd))
}

/// Evaluate plasma screening across `n_zones`, returning a `(n_zones, 3)` array.
#[pyfunction]
pub fn rates_screen_pair_zones<'py>(
    py: Python<'py>,
    mode: &str,
    temps: PyReadonlyArray1<'py, f64>,
    dens: PyReadonlyArray1<'py, f64>,
    zbars: PyReadonlyArray1<'py, f64>,
    abars: PyReadonlyArray1<'py, f64>,
    z2bars: PyReadonlyArray1<'py, f64>,
    a1: f64,
    z1: f64,
    a2: f64,
    z2: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let sm = parse_screen_mode(mode)?;
    let t_v = temps.as_slice()?.to_vec();
    let d_v = dens.as_slice()?.to_vec();
    let zb_v = zbars.as_slice()?.to_vec();
    let ab_v = abars.as_slice()?.to_vec();
    let z2b_v = z2bars.as_slice()?.to_vec();
    let nz = t_v.len();
    if d_v.len() != nz
        || zb_v.len() != nz
        || ab_v.len() != nz
        || z2b_v.len() != nz
    {
        return Err(PyValueError::new_err(
            "all zone arrays (temps, dens, zbars, abars, z2bars) must have equal length",
        ));
    }
    crate::eval_batch_2d(py, nz, true, 64, move |k| {
        let s = screen_pair(
            sm, t_v[k], d_v[k], zb_v[k], ab_v[k], z2b_v[k], a1, z1, a2, z2,
        );
        [s.scor, s.dscor_dt, s.dscor_dd]
    })
}

/// Evaluate Coulomb chemical potential and electron screening shift `(mu1, mu2, dmu_eV, vs)`.
#[pyfunction]
pub fn rates_eval_coulomb(
    which_mui: &str,
    which_vs: &str,
    temp: f64,
    den: f64,
    zbar: f64,
    abar: f64,
    z1: f64,
    z2: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    let mui_opt = parse_mui_opt(which_mui)?;
    let vs_opt = parse_vs_opt(which_vs)?;
    let mu1 = eval_mui_coulomb(mui_opt, temp, den, zbar, abar, z1);
    let mu2 = eval_mui_coulomb(mui_opt, temp, den, zbar, abar, z2);
    let dmu_ev = (mu1 - mu2).abs() * KEV * temp;
    let vs = eval_vs_coulomb(vs_opt, temp, den, zbar, abar, z1);
    Ok((mu1, mu2, dmu_ev, vs))
}

/// Evaluate tabulated weak reaction rate (`weaklib` or `Suzuki2016`), returning a 9-element array.
#[pyfunction]
#[pyo3(signature = (lhs_name, rhs_name, t9, ye_rho, use_suzuki=false))]
pub fn rates_eval_weak_pair(
    lhs_name: &str,
    rhs_name: &str,
    t9: f64,
    ye_rho: f64,
    use_suzuki: bool,
) -> PyResult<[f64; 9]> {
    let w = eval_weak_reaction_pair(lhs_name, rhs_name, t9, ye_rho, use_suzuki)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(pack_weak(&w))
}

/// Evaluate on-the-fly relativistic Fermi-Dirac electron-capture / beta-decay rate.
#[pyfunction]
#[pyo3(signature = (
    lhs_name, rhs_name, t9, ye_rho, eta_k,
    which_mui="pcr2009", which_vs="itoh2002",
    zbar=10.909090909090908, abar=21.818181818181817
))]
pub fn rates_eval_ecapture_pair(
    lhs_name: &str,
    rhs_name: &str,
    t9: f64,
    ye_rho: f64,
    eta_k: f64,
    which_mui: &str,
    which_vs: &str,
    zbar: f64,
    abar: f64,
) -> PyResult<[f64; 9]> {
    let mui_opt = parse_mui_opt(which_mui)?;
    let vs_opt = parse_vs_opt(which_vs)?;
    let w = eval_ecapture_pair(
        lhs_name, rhs_name, t9, ye_rho, eta_k, mui_opt, vs_opt, zbar, abar,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(pack_weak(&w))
}

/// Look up `(halflife_sec, qneu_mev)` from `weak_info.list` by isotope names.
#[pyfunction]
pub fn rates_weak_info_by_names(
    lhs_name: &str,
    rhs_name: &str,
) -> PyResult<(f64, f64)> {
    let ncat = NuclideCatalog::builtin_ref();
    let wl = ncat
        .get_nuclide_index(lhs_name)
        .ok_or_else(|| PyValueError::new_err(format!("unknown {lhs_name}")))?
        as u16;
    let wr = ncat
        .get_nuclide_index(rhs_name)
        .ok_or_else(|| PyValueError::new_err(format!("unknown {rhs_name}")))?
        as u16;
    RatesCatalog::builtin()
        .weak_info_by_ids(wl, wr)
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "no weak_info entry for ({lhs_name}, {rhs_name})"
            ))
        })
}

/// Evaluate the `idx`-th (`1..=463`) weak table at `(t9, l_ye_rho)`.
#[pyfunction]
pub fn rates_eval_weak_table_index(
    idx: usize,
    t9: f64,
    l_ye_rho: f64,
) -> PyResult<(String, String, [f64; 9])> {
    let rcat = RatesCatalog::builtin();
    if idx == 0 || idx > rcat.weak_tables.len() {
        return Err(PyValueError::new_err(format!(
            "weak table index {idx} out of range 1..={}",
            rcat.weak_tables.len()
        )));
    }
    let tab = &rcat.weak_tables[idx - 1];
    let w = eval_weaklib_one(tab, t9, l_ye_rho);
    Ok((tab.lhs_name.clone(), tab.rhs_name.clone(), pack_weak(&w)))
}

/// Register `rates` functions on `flexmesa._rust`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rates_reaction_id_by_name, m)?)?;
    m.add_function(wrap_pyfunction!(rates_reaction_name_by_id, m)?)?;
    m.add_function(wrap_pyfunction!(rates_reaction_info, m)?)?;
    m.add_function(wrap_pyfunction!(rates_reaclib_terms, m)?)?;
    m.add_function(wrap_pyfunction!(rates_eval_reaclib_handle, m)?)?;
    m.add_function(wrap_pyfunction!(rates_set_raw_rate, m)?)?;
    m.add_function(wrap_pyfunction!(rates_set_raw_rates, m)?)?;
    m.add_function(wrap_pyfunction!(rates_set_raw_rates_zones, m)?)?;
    m.add_function(wrap_pyfunction!(rates_eval_fl_epsnuc_3alf, m)?)?;
    m.add_function(wrap_pyfunction!(rates_eval_rate_table, m)?)?;
    m.add_function(wrap_pyfunction!(rates_screen_pair, m)?)?;
    m.add_function(wrap_pyfunction!(rates_screen_pair_zones, m)?)?;
    m.add_function(wrap_pyfunction!(rates_eval_coulomb, m)?)?;
    m.add_function(wrap_pyfunction!(rates_eval_weak_pair, m)?)?;
    m.add_function(wrap_pyfunction!(rates_eval_ecapture_pair, m)?)?;
    m.add_function(wrap_pyfunction!(rates_weak_info_by_names, m)?)?;
    m.add_function(wrap_pyfunction!(rates_eval_weak_table_index, m)?)?;
    Ok(())
}
