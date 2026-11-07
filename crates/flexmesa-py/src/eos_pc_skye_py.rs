// SPDX-License-Identifier: LGPL-3.0-or-later
//! `PyO3` bindings for `flexmesa-eos` Wave 3a.3 (`PC` and `Skye` equations of
//! state and domain blend weights).

use flexmesa_eos::{
    eval_pc_eos, eval_pc_eos_from_weights, eval_skye_eos,
    eval_skye_eos_from_weights, get_pc_alfa, get_skye_alfa,
    get_skye_alfa_simple, HelmTable, PcConfig, SkyeConfig, SkyeSolidMixingRule,
    NUM_EOS_BASIC_RESULTS,
};
use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::eos_py::{into_py_eos_tuple, PyEosTuple};
use crate::eval_batch_2d;

pub(crate) fn parse_mixing_rule(rule: &str) -> PyResult<SkyeSolidMixingRule> {
    match rule.to_ascii_lowercase().as_str() {
        "ogata" => Ok(SkyeSolidMixingRule::Ogata),
        "pc" => Ok(SkyeSolidMixingRule::Pc),
        other => Err(PyValueError::new_err(format!(
            "unknown skye_solid_mixing_rule '{other}' (expected 'Ogata' or 'PC')"
        ))),
    }
}

/// Evaluates `Get_PC_alfa` (`eospc_eval.f90` lines 31-96), returning
/// `(alfa, d_alfa_dlogT, d_alfa_dlogRho)`.
#[pyfunction]
#[pyo3(signature = (
    log_rho,
    log_t,
    abar,
    zbar,
    use_gamma_limit=true,
    log_rho1=2.999,
    log_rho2=2.8,
    log_t1=7.6,
    log_t2=7.7,
    log_gamma_e_all_helm=-1.8,
    log_gamma_e_all_pc=-1.7
))]
pub fn eos_get_pc_alfa(
    log_rho: f64,
    log_t: f64,
    abar: f64,
    zbar: f64,
    use_gamma_limit: bool,
    log_rho1: f64,
    log_rho2: f64,
    log_t1: f64,
    log_t2: f64,
    log_gamma_e_all_helm: f64,
    log_gamma_e_all_pc: f64,
) -> (f64, f64, f64) {
    let cfg = PcConfig {
        pc_use_gamma_limit_instead_of_t: use_gamma_limit,
        log_rho1_pc_limit: log_rho1,
        log_rho2_pc_limit: log_rho2,
        log_t1_pc_limit: log_t1,
        log_t2_pc_limit: log_t2,
        log_gamma_e_all_helm,
        log_gamma_e_all_pc,
        ..PcConfig::default()
    };
    get_pc_alfa(&cfg, log_rho, log_t, abar, zbar)
}

/// Evaluates the 26-channel Potekhin-Chabrier (`PC`) equation of state at a
/// single zone `(rho, temp, abar, zbar, chem_id, xa)`.
#[pyfunction]
#[pyo3(signature = (
    rho,
    temp,
    abar,
    zbar,
    chem_id,
    xa,
    include_radiation=true,
    pc_gamma_start_crystal=170.0,
    pc_gamma_full_crystal=180.0,
    mass_fraction_limit=1.0e-4
))]
pub fn eos_eval_pc<'py>(
    py: Python<'py>,
    rho: f64,
    temp: f64,
    abar: f64,
    zbar: f64,
    chem_id: Vec<usize>,
    xa: PyReadonlyArray1<'py, f64>,
    include_radiation: bool,
    pc_gamma_start_crystal: f64,
    pc_gamma_full_crystal: f64,
    mass_fraction_limit: f64,
) -> PyResult<PyEosTuple<'py>> {
    let xa_s = xa
        .as_slice()
        .map_err(|_| PyValueError::new_err("xa must be contiguous"))?;
    let cfg = PcConfig {
        include_radiation,
        pc_gamma_start_crystal,
        pc_gamma_full_crystal,
        mass_fraction_limit_for_pc: mass_fraction_limit,
        ..PcConfig::default()
    };
    let out = eval_pc_eos(&cfg, rho, temp, abar, zbar, &chem_id, xa_s)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(into_py_eos_tuple(py, &out))
}

/// Evaluates `PC` across `nz` zones at `(rho, temp, abar, zbar, xa_flat, azion,
/// acmi)`, returning an `(nz, 78)` array packing
/// `[res[0..26], d_dlnd[0..26], d_dln_t[0..26]]`.
#[pyfunction]
#[pyo3(signature = (
    rho,
    temp,
    abar,
    zbar,
    xa_flat,
    azion,
    acmi,
    include_radiation=true,
    pc_gamma_start_crystal=170.0,
    pc_gamma_full_crystal=180.0,
    mass_fraction_limit=1.0e-4,
    parallel=true
))]
pub fn eos_eval_pc_zones<'py>(
    py: Python<'py>,
    rho: PyReadonlyArray1<'py, f64>,
    temp: PyReadonlyArray1<'py, f64>,
    abar: PyReadonlyArray1<'py, f64>,
    zbar: PyReadonlyArray1<'py, f64>,
    xa_flat: PyReadonlyArray1<'py, f64>,
    azion: PyReadonlyArray1<'py, f64>,
    acmi: PyReadonlyArray1<'py, f64>,
    include_radiation: bool,
    pc_gamma_start_crystal: f64,
    pc_gamma_full_crystal: f64,
    mass_fraction_limit: f64,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let r_s = rho
        .as_slice()
        .map_err(|_| PyValueError::new_err("rho must be contiguous"))?
        .to_vec();
    let t_s = temp
        .as_slice()
        .map_err(|_| PyValueError::new_err("temp must be contiguous"))?
        .to_vec();
    let a_s = abar
        .as_slice()
        .map_err(|_| PyValueError::new_err("abar must be contiguous"))?
        .to_vec();
    let z_s = zbar
        .as_slice()
        .map_err(|_| PyValueError::new_err("zbar must be contiguous"))?
        .to_vec();
    let xa_s = xa_flat
        .as_slice()
        .map_err(|_| PyValueError::new_err("xa_flat must be contiguous"))?
        .to_vec();
    let az_s = azion
        .as_slice()
        .map_err(|_| PyValueError::new_err("azion must be contiguous"))?
        .to_vec();
    let ac_s = acmi
        .as_slice()
        .map_err(|_| PyValueError::new_err("acmi must be contiguous"))?
        .to_vec();
    let nz = r_s.len();
    let sp = az_s.len();
    if t_s.len() != nz
        || a_s.len() != nz
        || z_s.len() != nz
        || ac_s.len() != sp
        || xa_s.len() != nz * sp
    {
        return Err(PyValueError::new_err(
            "zone or species slice length mismatch in eos_eval_pc_zones",
        ));
    }
    let cfg = PcConfig {
        include_radiation,
        pc_gamma_start_crystal,
        pc_gamma_full_crystal,
        mass_fraction_limit_for_pc: mass_fraction_limit,
        ..PcConfig::default()
    };
    eval_batch_2d::<78, _>(py, nz, parallel, 32, move |k| {
        let mut row = [f64::NAN; 78];
        let xa_k = &xa_s[k * sp..(k + 1) * sp];
        if let Ok(out) = eval_pc_eos_from_weights(
            &cfg, r_s[k], t_s[k], a_s[k], z_s[k], xa_k, &az_s, &ac_s,
        ) {
            let n = NUM_EOS_BASIC_RESULTS;
            row[0..n].copy_from_slice(&out.res);
            row[n..2 * n].copy_from_slice(&out.d_dlnd);
            row[2 * n..3 * n].copy_from_slice(&out.d_dln_t);
        }
        row
    })
}

/// Evaluates `Get_Skye_alfa` (`skye.f90` lines 52-135), returning
/// `(alfa, d_alfa_dlogT, d_alfa_dlogRho)`.
#[pyfunction]
pub fn eos_get_skye_alfa(
    log_rho: f64,
    log_t: f64,
    abar: f64,
    zbar: f64,
) -> (f64, f64, f64) {
    get_skye_alfa(HelmTable::builtin(), log_rho, log_t, abar, zbar)
}

/// Evaluates `get_Skye_alfa_simple` (`skye.f90` lines 139-227), returning
/// `(alfa, d_alfa_dlogT, d_alfa_dlogRho)`.
#[pyfunction]
pub fn eos_get_skye_alfa_simple(log_rho: f64, log_t: f64) -> (f64, f64, f64) {
    get_skye_alfa_simple(&SkyeConfig::default(), log_rho, log_t)
}

/// Evaluates the 26-channel Jermyn et al. (2021) `Skye` equation of state at a
/// single zone `(rho, temp, abar, zbar, chem_id, xa)`.
#[pyfunction]
#[pyo3(signature = (
    rho,
    temp,
    abar,
    zbar,
    chem_id,
    xa,
    skye_use_ion_offsets=true,
    skye_solid_mixing_rule="PC",
    mass_fraction_limit=1.0e-4
))]
pub fn eos_eval_skye<'py>(
    py: Python<'py>,
    rho: f64,
    temp: f64,
    abar: f64,
    zbar: f64,
    chem_id: Vec<usize>,
    xa: PyReadonlyArray1<'py, f64>,
    skye_use_ion_offsets: bool,
    skye_solid_mixing_rule: &str,
    mass_fraction_limit: f64,
) -> PyResult<PyEosTuple<'py>> {
    let xa_s = xa
        .as_slice()
        .map_err(|_| PyValueError::new_err("xa must be contiguous"))?;
    let cfg = SkyeConfig {
        skye_use_ion_offsets,
        skye_solid_mixing_rule: parse_mixing_rule(skye_solid_mixing_rule)?,
        mass_fraction_limit_for_skye: mass_fraction_limit,
        ..SkyeConfig::default()
    };
    let out = eval_skye_eos(
        &cfg,
        HelmTable::builtin(),
        rho,
        temp,
        abar,
        zbar,
        &chem_id,
        xa_s,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(into_py_eos_tuple(py, &out))
}

/// Evaluates `Skye` across `nz` zones at `(rho, temp, abar, zbar, xa_flat, z,
/// acmi, a)`, returning an `(nz, 78)` array packing
/// `[res[0..26], d_dlnd[0..26], d_dln_t[0..26]]`.
#[pyfunction]
#[pyo3(signature = (
    rho,
    temp,
    abar,
    zbar,
    xa_flat,
    z,
    acmi,
    a,
    skye_use_ion_offsets=true,
    skye_solid_mixing_rule="PC",
    mass_fraction_limit=1.0e-4,
    parallel=true
))]
#[allow(clippy::too_many_arguments)]
pub fn eos_eval_skye_zones<'py>(
    py: Python<'py>,
    rho: PyReadonlyArray1<'py, f64>,
    temp: PyReadonlyArray1<'py, f64>,
    abar: PyReadonlyArray1<'py, f64>,
    zbar: PyReadonlyArray1<'py, f64>,
    xa_flat: PyReadonlyArray1<'py, f64>,
    z: Vec<usize>,
    acmi: PyReadonlyArray1<'py, f64>,
    a: PyReadonlyArray1<'py, f64>,
    skye_use_ion_offsets: bool,
    skye_solid_mixing_rule: &str,
    mass_fraction_limit: f64,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let r_s = rho
        .as_slice()
        .map_err(|_| PyValueError::new_err("rho must be contiguous"))?
        .to_vec();
    let t_s = temp
        .as_slice()
        .map_err(|_| PyValueError::new_err("temp must be contiguous"))?
        .to_vec();
    let ab_s = abar
        .as_slice()
        .map_err(|_| PyValueError::new_err("abar must be contiguous"))?
        .to_vec();
    let zb_s = zbar
        .as_slice()
        .map_err(|_| PyValueError::new_err("zbar must be contiguous"))?
        .to_vec();
    let xa_s = xa_flat
        .as_slice()
        .map_err(|_| PyValueError::new_err("xa_flat must be contiguous"))?
        .to_vec();
    let ac_s = acmi
        .as_slice()
        .map_err(|_| PyValueError::new_err("acmi must be contiguous"))?
        .to_vec();
    let a_s = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("a must be contiguous"))?
        .to_vec();
    let nz = r_s.len();
    let sp = z.len();
    if t_s.len() != nz
        || ab_s.len() != nz
        || zb_s.len() != nz
        || ac_s.len() != sp
        || a_s.len() != sp
        || xa_s.len() != nz * sp
    {
        return Err(PyValueError::new_err(
            "zone or species slice length mismatch in eos_eval_skye_zones",
        ));
    }
    let cfg = SkyeConfig {
        skye_use_ion_offsets,
        skye_solid_mixing_rule: parse_mixing_rule(skye_solid_mixing_rule)?,
        mass_fraction_limit_for_skye: mass_fraction_limit,
        ..SkyeConfig::default()
    };
    let ht = HelmTable::builtin();
    eval_batch_2d::<78, _>(py, nz, parallel, 32, move |k| {
        let mut row = [f64::NAN; 78];
        let xa_k = &xa_s[k * sp..(k + 1) * sp];
        if let Ok(out) = eval_skye_eos_from_weights(
            &cfg, ht, t_s[k], r_s[k], ab_s[k], zb_s[k], xa_k, &z, &ac_s, &a_s,
        ) {
            let n = NUM_EOS_BASIC_RESULTS;
            row[0..n].copy_from_slice(&out.res);
            row[n..2 * n].copy_from_slice(&out.d_dlnd);
            row[2 * n..3 * n].copy_from_slice(&out.d_dln_t);
        }
        row
    })
}

/// Registers Wave 3a.3 `PC` and `Skye` bindings on the `flexmesa._rust` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(eos_get_pc_alfa, m)?)?;
    m.add_function(wrap_pyfunction!(eos_eval_pc, m)?)?;
    m.add_function(wrap_pyfunction!(eos_eval_pc_zones, m)?)?;
    m.add_function(wrap_pyfunction!(eos_get_skye_alfa, m)?)?;
    m.add_function(wrap_pyfunction!(eos_get_skye_alfa_simple, m)?)?;
    m.add_function(wrap_pyfunction!(eos_eval_skye, m)?)?;
    m.add_function(wrap_pyfunction!(eos_eval_skye_zones, m)?)?;
    Ok(())
}
