// SPDX-License-Identifier: LGPL-3.0-or-later
//! Tier B `PyO3` bindings for `flexmesa-atm` (surface boundary conditions,
//! $T(\tau)$ relations, irradiated atmospheres, and atmosphere table interpolation).

use flexmesa_atm::{
    eval_e2, eval_irradiated, eval_paczynski_gradr, eval_t_tau,
    eval_t_tau_dq_dtau, eval_t_tau_uniform, eval_t_tau_varying, eval_table,
    get_t_tau_base, get_table_alfa_beta, get_table_base, radiation_pressure,
    AtmEosCallbackResult, AtmError, AtmEvalOutput, AtmKapCallbackResult,
    AtmTables, ATM_TABLE_PHOTOSPHERE,
};
use flexmesa_const::LN10;
use flexmesa_eos::{eos_pt_get, EosConfig, HelmTable, I_ETA, I_LN_FREE_E};
use flexmesa_kap::{kap_get, KapConfig, KapLowTOption};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type AtmEvalTuple = (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64);
type AtmSplineTuple<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    bool,
    bool,
    Bound<'py, PyArray2<f64>>,
);

fn to_py_err(e: AtmError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn unpack_atm_eval(out: AtmEvalOutput) -> AtmEvalTuple {
    (
        out.ln_t,
        out.dln_t_dl,
        out.dln_t_dln_r,
        out.dln_t_dln_m,
        out.dln_t_dln_kap,
        out.ln_p,
        out.dln_p_dl,
        out.dln_p_dln_r,
        out.dln_p_dln_m,
        out.dln_p_dln_kap,
        out.kap,
    )
}

fn parse_low_t_option(name: &str) -> PyResult<KapLowTOption> {
    if let Some(opt) = KapLowTOption::from_prefix(name) {
        return Ok(opt);
    }
    match name.to_ascii_lowercase().as_str() {
        "low_t_fa05_gs98" | "lowt_fa05_gs98" | "default" => {
            Ok(KapLowTOption::LowTFa05Gs98)
        }
        "low_t_fa05_gn93" | "lowt_fa05_gn93" => Ok(KapLowTOption::LowTFa05Gn93),
        "low_t_fa05_a09p" | "lowt_fa05_a09p" => Ok(KapLowTOption::LowTFa05A09p),
        "low_t_freedman11" | "lowt_freedman11" | "freedman11" => {
            Ok(KapLowTOption::LowTFreedman11)
        }
        "kapcn" | "kap_cn" => Ok(KapLowTOption::KapCn),
        "aesopus" => Ok(KapLowTOption::Aesopus),
        other => Err(PyValueError::new_err(format!(
            "unknown kap low_t_option: {other}"
        ))),
    }
}

fn eval_eos_callback(
    ht: &HelmTable,
    eos_cfg: &EosConfig,
    chem_id: &[usize],
    xa: &[f64],
    ln_p: f64,
    ln_t: f64,
) -> Result<AtmEosCallbackResult, AtmError> {
    let t = ln_t.exp();
    let p = ln_p.exp();
    let prad = radiation_pressure(t);
    let pgas = (p - prad).max(1.0e-99);
    let log_pgas = pgas.log10();
    let log_t = ln_t / LN10;
    let pt =
        eos_pt_get(ht, chem_id, xa, log_pgas, log_t, eos_cfg).map_err(|e| {
            AtmError::CallbackError {
                context: "eos_proc",
                detail: e.to_string(),
            }
        })?;
    Ok(AtmEosCallbackResult {
        ln_rho: pt.log10_rho * LN10,
        res: pt.eos.res,
        dres_dln_rho: pt.eos.d_dlnd,
        dres_dln_t: pt.eos.d_dln_t,
    })
}

fn eval_kap_callback(
    kap_cfg: &KapConfig,
    chem_id: &[usize],
    xa: &[f64],
    ln_rho: f64,
    ln_t: f64,
    eos: &AtmEosCallbackResult,
) -> Result<AtmKapCallbackResult, AtmError> {
    let out = kap_get(
        kap_cfg,
        chem_id,
        xa,
        ln_rho / LN10,
        ln_t / LN10,
        eos.res[I_LN_FREE_E],
        eos.dres_dln_rho[I_LN_FREE_E],
        eos.dres_dln_t[I_LN_FREE_E],
        eos.res[I_ETA],
        eos.dres_dln_rho[I_ETA],
        eos.dres_dln_t[I_ETA],
    )
    .map_err(|e| AtmError::CallbackError {
        context: "kap_proc",
        detail: e.to_string(),
    })?;
    Ok(AtmKapCallbackResult {
        kap: out.kap,
        dlnkap_dln_rho: out.dlnkap_dln_rho,
        dlnkap_dln_t: out.dlnkap_dln_t,
    })
}

/// Evaluates $\ln T(\tau, T_{\mathrm{eff}})$ for a $T(\tau)$ relation ID.
#[pyfunction]
pub fn atm_eval_t_tau(t_tau_id: i32, tau: f64, teff: f64) -> PyResult<f64> {
    eval_t_tau(t_tau_id, tau, teff).map_err(to_py_err)
}

/// Evaluates $dq/d\tau$ for a $T(\tau)$ relation ID at optical depth $\tau$.
#[pyfunction]
pub fn atm_eval_t_tau_dq_dtau(t_tau_id: i32, tau: f64) -> PyResult<f64> {
    eval_t_tau_dq_dtau(t_tau_id, tau).map_err(to_py_err)
}

/// Returns the base optical depth $\tau_{\mathrm{base}}$ of a $T(\tau)$ relation ID.
#[pyfunction]
pub fn atm_get_t_tau_base(t_tau_id: i32) -> PyResult<f64> {
    get_t_tau_base(t_tau_id).map_err(to_py_err)
}

/// Evaluates the second exponential integral $(E_2(x), dE_2/dx)$.
#[pyfunction]
pub fn atm_eval_e2(x: f64) -> PyResult<(f64, f64)> {
    eval_e2(x).map_err(to_py_err)
}

/// Evaluates the Paczynski (1969) optical-depth-diluted radiative gradient $\nabla_{\mathrm{rad}}$.
#[pyfunction]
pub fn atm_eval_paczynski_gradr(
    t: f64,
    p: f64,
    rho: f64,
    tau: f64,
    kap: f64,
    l: f64,
    m: f64,
    r: f64,
    cgrav: f64,
) -> f64 {
    eval_paczynski_gradr(t, p, rho, tau, kap, l, m, r, cgrav)
}

/// Evaluates a tabulated model atmosphere (`eval_table`).
#[pyfunction]
pub fn atm_eval_table(
    l: f64,
    r: f64,
    m: f64,
    cgrav: f64,
    id: i32,
    z: f64,
    skip_partials: bool,
    teff: f64,
) -> PyResult<AtmEvalTuple> {
    let out = eval_table(l, r, m, cgrav, id, z, skip_partials, teff)
        .map_err(to_py_err)?;
    Ok(unpack_atm_eval(out))
}

/// Evaluates blending weights $(\alpha, \beta)$ between grey and table atmospheres.
#[pyfunction]
pub fn atm_get_table_alfa_beta(
    l: f64,
    teff: f64,
    r: f64,
    m: f64,
    cgrav: f64,
    id: i32,
) -> PyResult<(f64, f64)> {
    get_table_alfa_beta(l, teff, r, m, cgrav, id).map_err(to_py_err)
}

/// Returns the base optical depth $\tau_{\mathrm{base}}$ of an atmosphere table ID.
#[pyfunction]
pub fn atm_get_table_base(id: i32) -> PyResult<f64> {
    get_table_base(id).map_err(to_py_err)
}

fn make_kap_cfg(zbase: f64, low_t_option: &str) -> PyResult<KapConfig> {
    let low_t = parse_low_t_option(low_t_option)?;
    Ok(KapConfig {
        zbase,
        kap_low_t_option: low_t,
        ..KapConfig::default()
    })
}

fn unpack_opt_comp(
    chem_id: Option<Vec<usize>>,
    xa: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<(Vec<usize>, Vec<f64>)> {
    let cid_vec = chem_id.unwrap_or_default();
    let xa_vec = xa
        .map(|a| a.as_slice().map(<[f64]>::to_vec))
        .transpose()?
        .unwrap_or_default();
    Ok((cid_vec, xa_vec))
}

/// Evaluates a grey $T(\tau)$ atmosphere with uniform opacity (`eval_t_tau_uniform`).
#[pyfunction]
#[pyo3(signature = (
    tau_surf, l, r, m, cgrav, kap_guess, pextra_factor, t_tau_id,
    errtol, max_iters, skip_partials, teff,
    chem_id=None, xa=None, zbase=0.02, low_t_option="low_t_fa05_gs98"
))]
pub fn atm_eval_t_tau_uniform(
    tau_surf: f64,
    l: f64,
    r: f64,
    m: f64,
    cgrav: f64,
    kap_guess: f64,
    pextra_factor: f64,
    t_tau_id: i32,
    errtol: f64,
    max_iters: usize,
    skip_partials: bool,
    teff: f64,
    chem_id: Option<Vec<usize>>,
    xa: Option<PyReadonlyArray1<'_, f64>>,
    zbase: f64,
    low_t_option: &str,
) -> PyResult<AtmEvalTuple> {
    let (cid_vec, xa_vec) = unpack_opt_comp(chem_id, xa)?;
    let ht = HelmTable::builtin();
    let eos_cfg = EosConfig::default();
    let kap_cfg = make_kap_cfg(zbase, low_t_option)?;
    let out = eval_t_tau_uniform(
        tau_surf,
        l,
        r,
        m,
        cgrav,
        kap_guess,
        pextra_factor,
        t_tau_id,
        |ln_p, ln_t| {
            eval_eos_callback(ht, &eos_cfg, &cid_vec, &xa_vec, ln_p, ln_t)
        },
        |ln_rho, ln_t, eos| {
            eval_kap_callback(&kap_cfg, &cid_vec, &xa_vec, ln_rho, ln_t, eos)
        },
        errtol,
        max_iters,
        skip_partials,
        teff,
    )
    .map_err(to_py_err)?;
    Ok(unpack_atm_eval(out))
}

/// Evaluates a $T(\tau)$ atmosphere with depth-varying opacity (`eval_t_tau_varying`).
#[pyfunction]
#[pyo3(signature = (
    tau_surf, l, r, m, cgrav, t_tau_id, errtol, max_steps,
    skip_partials, teff, chem_id, xa, zbase=0.02, low_t_option="low_t_fa05_gs98"
))]
pub fn atm_eval_t_tau_varying(
    tau_surf: f64,
    l: f64,
    r: f64,
    m: f64,
    cgrav: f64,
    t_tau_id: i32,
    errtol: f64,
    max_steps: usize,
    skip_partials: bool,
    teff: f64,
    chem_id: Vec<usize>,
    xa: PyReadonlyArray1<'_, f64>,
    zbase: f64,
    low_t_option: &str,
) -> PyResult<AtmEvalTuple> {
    let xa_slice = xa.as_slice()?;
    let ht = HelmTable::builtin();
    let eos_cfg = EosConfig::default();
    let kap_cfg = make_kap_cfg(zbase, low_t_option)?;
    let out = eval_t_tau_varying(
        tau_surf,
        l,
        r,
        m,
        cgrav,
        t_tau_id,
        |ln_p, ln_t| {
            eval_eos_callback(ht, &eos_cfg, &chem_id, xa_slice, ln_p, ln_t)
        },
        |ln_rho, ln_t, eos| {
            eval_kap_callback(&kap_cfg, &chem_id, xa_slice, ln_rho, ln_t, eos)
        },
        errtol,
        max_steps,
        skip_partials,
        teff,
    )
    .map_err(to_py_err)?;
    Ok(unpack_atm_eval(out))
}

/// Evaluates an irradiated grey atmosphere (`eval_irradiated`).
#[pyfunction]
#[pyo3(signature = (
    l, r, m, cgrav, t_eq, p_surf, kap_guess, kap_v, gamma,
    errtol, max_iters, skip_partials,
    chem_id=None, xa=None, zbase=0.02, low_t_option="low_t_fa05_gs98"
))]
pub fn atm_eval_irradiated(
    l: f64,
    r: f64,
    m: f64,
    cgrav: f64,
    t_eq: f64,
    p_surf: f64,
    kap_guess: f64,
    kap_v: f64,
    gamma: f64,
    errtol: f64,
    max_iters: usize,
    skip_partials: bool,
    chem_id: Option<Vec<usize>>,
    xa: Option<PyReadonlyArray1<'_, f64>>,
    zbase: f64,
    low_t_option: &str,
) -> PyResult<(f64, f64, f64, f64, f64, f64, f64)> {
    let (cid_vec, xa_vec) = unpack_opt_comp(chem_id, xa)?;
    let ht = HelmTable::builtin();
    let eos_cfg = EosConfig::default();
    let kap_cfg = make_kap_cfg(zbase, low_t_option)?;
    let out = eval_irradiated(
        l,
        r,
        m,
        cgrav,
        t_eq,
        p_surf,
        kap_guess,
        kap_v,
        gamma,
        |ln_p, ln_t| {
            eval_eos_callback(ht, &eos_cfg, &cid_vec, &xa_vec, ln_p, ln_t)
        },
        |ln_rho, ln_t, eos| {
            eval_kap_callback(&kap_cfg, &cid_vec, &xa_vec, ln_rho, ln_t, eos)
        },
        errtol,
        max_iters,
        skip_partials,
    )
    .map_err(to_py_err)?;
    Ok((
        out.kap,
        out.tau,
        out.ln_t,
        out.dln_t_dl,
        out.dln_t_dln_r,
        out.dln_t_dln_m,
        out.dln_t_dln_kap,
    ))
}

/// Evaluates `eval_table` across `n_zones` surface boundary conditions,
/// returning an `(n_zones, 11)` array.
#[pyfunction]
#[pyo3(signature = (l, r, m, cgrav, id, z, skip_partials, teff, parallel=true))]
pub fn atm_eval_table_zones<'py>(
    py: Python<'py>,
    l: PyReadonlyArray1<'py, f64>,
    r: PyReadonlyArray1<'py, f64>,
    m: PyReadonlyArray1<'py, f64>,
    cgrav: f64,
    id: i32,
    z: f64,
    skip_partials: bool,
    teff: PyReadonlyArray1<'py, f64>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let l_v = l.as_slice()?.to_vec();
    let r_v = r.as_slice()?.to_vec();
    let m_v = m.as_slice()?.to_vec();
    let t_v = teff.as_slice()?.to_vec();
    let nz = l_v.len();
    if r_v.len() != nz || m_v.len() != nz || t_v.len() != nz {
        return Err(PyValueError::new_err(
            "l, r, m, and teff must have equal length",
        ));
    }
    let out_arr =
        crate::eval_batch_2d::<11, _>(py, nz, parallel, 16, move |k| {
            match eval_table(
                l_v[k],
                r_v[k],
                m_v[k],
                cgrav,
                id,
                z,
                skip_partials,
                t_v[k],
            ) {
                Ok(out) => [
                    out.ln_t,
                    out.dln_t_dl,
                    out.dln_t_dln_r,
                    out.dln_t_dln_m,
                    out.dln_t_dln_kap,
                    out.ln_p,
                    out.dln_p_dl,
                    out.dln_p_dln_r,
                    out.dln_p_dln_m,
                    out.dln_p_dln_kap,
                    out.kap,
                ],
                Err(_) => [f64::NAN; 11],
            }
        })?;
    if out_arr.readonly().as_slice()?.iter().any(|v| v.is_nan()) {
        return Err(PyValueError::new_err("atm_eval_table_zones failed"));
    }
    Ok(out_arr)
}

/// Returns `(logg_array, teff_array, iling, ilint, pgas_coeffs)` for the solar
/// metallicity (`log_z == 0.0`) slice of `ATM_TABLE_PHOTOSPHERE`.
#[pyfunction]
pub fn atm_table_photosphere_zsol_spline<'py>(
    py: Python<'py>,
) -> PyResult<AtmSplineTuple<'py>> {
    let ai = AtmTables::builtin()
        .get_info(ATM_TABLE_PHOTOSPHERE)
        .map_err(to_py_err)?;
    let iz = ai
        .log_z
        .iter()
        .position(|&lz| lz.abs() < 1.0e-12)
        .ok_or_else(|| PyValueError::new_err("missing log_z == 0.0 slice"))?;
    let n_cells = ai.ng * ai.nt;
    let mut flat = Vec::with_capacity(n_cells * 4);
    for c4 in &ai.pgas_splines[iz] {
        flat.extend_from_slice(c4);
    }
    Ok((
        PyArray1::from_slice(py, &ai.logg_array),
        PyArray1::from_slice(py, &ai.teff_array),
        ai.iling,
        ai.ilint,
        PyArray1::from_vec(py, flat).reshape([n_cells, 4])?,
    ))
}

/// Registers `flexmesa-atm` functions on the `flexmesa._rust` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(atm_eval_t_tau, m)?)?;
    m.add_function(wrap_pyfunction!(atm_eval_t_tau_dq_dtau, m)?)?;
    m.add_function(wrap_pyfunction!(atm_get_t_tau_base, m)?)?;
    m.add_function(wrap_pyfunction!(atm_eval_e2, m)?)?;
    m.add_function(wrap_pyfunction!(atm_eval_paczynski_gradr, m)?)?;
    m.add_function(wrap_pyfunction!(atm_eval_table, m)?)?;
    m.add_function(wrap_pyfunction!(atm_eval_table_zones, m)?)?;
    m.add_function(wrap_pyfunction!(atm_table_photosphere_zsol_spline, m)?)?;
    m.add_function(wrap_pyfunction!(atm_get_table_alfa_beta, m)?)?;
    m.add_function(wrap_pyfunction!(atm_get_table_base, m)?)?;
    m.add_function(wrap_pyfunction!(atm_eval_t_tau_uniform, m)?)?;
    m.add_function(wrap_pyfunction!(atm_eval_t_tau_varying, m)?)?;
    m.add_function(wrap_pyfunction!(atm_eval_irradiated, m)?)?;
    Ok(())
}
