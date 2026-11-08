// SPDX-License-Identifier: LGPL-3.0-or-later
//! `PyO3` bindings for `flexmesa-eos` Wave 3a.3 and Wave 3b.1 (`eosDT_get`
//! 7-level cascade, 4D tabulated EOS evaluators, and `PT`/`DE` inverse solvers).

use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_const::LN10;
use flexmesa_eos::{
    eos_dt_get, eos_dt_get_rho, eos_dt_get_t, eos_gamma_de_get, eos_pt_get,
    eos_pt_get_pgas, eos_pt_get_t, eval_cms_eos, eval_free_eos,
    eval_opal_scvh_eos, get1_eosdt_results, CmsConfig, CmsTableSet, EosConfig,
    EosDtTableKind, EosDtTableSet, FreeEosConfig, HelmTable, OpalScvhConfig,
    I_FRAC_OPAL_SCVH, NUM_EOS_BASIC_RESULTS,
};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::eos_pc_skye_py::parse_mixing_rule;
use crate::eos_py::{into_py_eos_tuple, PyEosTuple};
use crate::eval_batch_2d;

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

#[allow(clippy::too_many_arguments)]
fn build_eos_config(
    use_skye: bool,
    use_pc: bool,
    pc_use_gamma_limit: bool,
    skye_solid_mixing_rule: &str,
    use_cms: bool,
    use_free_eos: bool,
    use_opal_scvh: bool,
) -> PyResult<EosConfig> {
    let mut cfg = EosConfig {
        use_cms,
        use_skye,
        use_pc,
        use_free_eos,
        use_opal_scvh,
        ..EosConfig::default()
    };
    cfg.pc.pc_use_gamma_limit_instead_of_t = pc_use_gamma_limit;
    cfg.skye.skye_solid_mixing_rule =
        parse_mixing_rule(skye_solid_mixing_rule)?;
    Ok(cfg)
}

/// Evaluates the unified top-level `eosDT_get` 7-level cascade (`CMS -> Skye ->
/// PC -> FreeEOS -> OPAL/SCVH -> HELM -> ideal`) at `(rho, temp, chem_id, xa)`.
#[pyfunction]
#[pyo3(signature = (
    rho,
    temp,
    chem_id,
    xa,
    use_skye=true,
    use_pc=false,
    pc_use_gamma_limit=true,
    skye_solid_mixing_rule="PC",
    use_cms=false,
    use_free_eos=true,
    use_opal_scvh=true
))]
#[allow(clippy::too_many_arguments)]
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
    use_cms: bool,
    use_free_eos: bool,
    use_opal_scvh: bool,
) -> PyResult<PyEosTuple<'py>> {
    let xa_s = xa
        .as_slice()
        .map_err(|_| PyValueError::new_err("xa must be contiguous"))?;
    let cfg = build_eos_config(
        use_skye,
        use_pc,
        pc_use_gamma_limit,
        skye_solid_mixing_rule,
        use_cms,
        use_free_eos,
        use_opal_scvh,
    )?;
    let out = eos_dt_get(HelmTable::builtin(), rho, temp, &chem_id, xa_s, &cfg)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(into_py_eos_tuple(py, &out))
}

/// Evaluates `eosPT_get` at `(pgas, temp, chem_id, xa)`.
#[pyfunction]
#[pyo3(signature = (
    pgas,
    temp,
    chem_id,
    xa,
    use_skye=true,
    use_pc=false,
    pc_use_gamma_limit=true,
    skye_solid_mixing_rule="PC",
    use_cms=false,
    use_free_eos=true,
    use_opal_scvh=true
))]
#[allow(clippy::too_many_arguments)]
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
    use_cms: bool,
    use_free_eos: bool,
    use_opal_scvh: bool,
) -> PyResult<PyPtTuple<'py>> {
    let xa_s = xa
        .as_slice()
        .map_err(|_| PyValueError::new_err("xa must be contiguous"))?;
    let cfg = build_eos_config(
        use_skye,
        use_pc,
        pc_use_gamma_limit,
        skye_solid_mixing_rule,
        use_cms,
        use_free_eos,
        use_opal_scvh,
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

/// Evaluates `OPAL/SCVH` 4D tabulated EOS standalone at `(rho, temp, chem_id, xa)`.
#[pyfunction]
pub fn eos_eval_opal_scvh<'py>(
    py: Python<'py>,
    rho: f64,
    temp: f64,
    chem_id: Vec<usize>,
    xa: PyReadonlyArray1<'py, f64>,
) -> PyResult<PyEosTuple<'py>> {
    let xa_s = xa
        .as_slice()
        .map_err(|_| PyValueError::new_err("xa must be contiguous"))?;
    let out = eval_opal_scvh_eos(
        EosDtTableSet::builtin(),
        rho,
        temp,
        &chem_id,
        xa_s,
        &OpalScvhConfig::default(),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(into_py_eos_tuple(py, &out))
}

/// Evaluates `FreeEOS` 4D tabulated EOS standalone at `(rho, temp, chem_id, xa)`.
#[pyfunction]
pub fn eos_eval_free_eos<'py>(
    py: Python<'py>,
    rho: f64,
    temp: f64,
    chem_id: Vec<usize>,
    xa: PyReadonlyArray1<'py, f64>,
) -> PyResult<PyEosTuple<'py>> {
    let xa_s = xa
        .as_slice()
        .map_err(|_| PyValueError::new_err("xa must be contiguous"))?;
    let out = eval_free_eos(
        EosDtTableSet::builtin(),
        rho,
        temp,
        &chem_id,
        xa_s,
        &FreeEosConfig::default(),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(into_py_eos_tuple(py, &out))
}

/// Evaluates Chabrier, Mazevet & Soubiran (`CMS`) H/He EOS standalone.
#[pyfunction]
pub fn eos_eval_cms<'py>(
    py: Python<'py>,
    rho: f64,
    temp: f64,
    chem_id: Vec<usize>,
    xa: PyReadonlyArray1<'py, f64>,
) -> PyResult<PyEosTuple<'py>> {
    let xa_s = xa
        .as_slice()
        .map_err(|_| PyValueError::new_err("xa must be contiguous"))?;
    let out = eval_cms_eos(
        CmsTableSet::builtin(),
        &CmsConfig::default(),
        rho,
        temp,
        &chem_id,
        xa_s,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(into_py_eos_tuple(py, &out))
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
    use_pc=false,
    use_cms=false,
    use_free_eos=true,
    use_opal_scvh=true
))]
#[allow(clippy::too_many_arguments)]
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
    use_cms: bool,
    use_free_eos: bool,
    use_opal_scvh: bool,
) -> PyResult<PyInvTuple<'py>> {
    let xa_s = xa
        .as_slice()
        .map_err(|_| PyValueError::new_err("xa must be contiguous"))?;
    let cfg = build_eos_config(
        use_skye,
        use_pc,
        true,
        "PC",
        use_cms,
        use_free_eos,
        use_opal_scvh,
    )?;
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
        100,
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
    use_pc=false,
    use_cms=false,
    use_free_eos=true,
    use_opal_scvh=true
))]
#[allow(clippy::too_many_arguments)]
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
    use_cms: bool,
    use_free_eos: bool,
    use_opal_scvh: bool,
) -> PyResult<PyInvTuple<'py>> {
    let xa_s = xa
        .as_slice()
        .map_err(|_| PyValueError::new_err("xa must be contiguous"))?;
    let cfg = build_eos_config(
        use_skye,
        use_pc,
        true,
        "PC",
        use_cms,
        use_free_eos,
        use_opal_scvh,
    )?;
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
        100,
        &cfg,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let (res, d_dlnd, d_dln_t) = into_py_eos_tuple(py, &inv.eos);
    Ok((inv.linear_result, inv.log10_result, res, d_dlnd, d_dln_t))
}

/// Evaluates `eosPT_get_T` solving for `T` at fixed `pgas` and
/// `res[which_other] == other_value`.
#[pyfunction]
#[pyo3(signature = (
    pgas,
    which_other,
    other_value,
    chem_id,
    xa,
    log_t_guess=6.0,
    use_skye=true,
    use_pc=false,
    use_cms=false,
    use_free_eos=true,
    use_opal_scvh=true
))]
#[allow(clippy::too_many_arguments)]
pub fn eos_eval_pt_get_t<'py>(
    py: Python<'py>,
    pgas: f64,
    which_other: usize,
    other_value: f64,
    chem_id: Vec<usize>,
    xa: PyReadonlyArray1<'py, f64>,
    log_t_guess: f64,
    use_skye: bool,
    use_pc: bool,
    use_cms: bool,
    use_free_eos: bool,
    use_opal_scvh: bool,
) -> PyResult<PyInvTuple<'py>> {
    let xa_s = xa
        .as_slice()
        .map_err(|_| PyValueError::new_err("xa must be contiguous"))?;
    let cfg = build_eos_config(
        use_skye,
        use_pc,
        true,
        "PC",
        use_cms,
        use_free_eos,
        use_opal_scvh,
    )?;
    let inv = eos_pt_get_t(
        HelmTable::builtin(),
        &chem_id,
        xa_s,
        pgas.log10(),
        which_other,
        other_value,
        log_t_guess,
        1.0e-8,
        LN10 * 1.0e-8,
        100,
        &cfg,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let (res, d_dlnd, d_dln_t) = into_py_eos_tuple(py, &inv.pt.eos);
    Ok((inv.linear_result, inv.log10_result, res, d_dlnd, d_dln_t))
}

/// Evaluates `eosPT_get_Pgas` solving for `pgas` at fixed `temp` and
/// `res[which_other] == other_value`.
#[pyfunction]
#[pyo3(signature = (
    temp,
    which_other,
    other_value,
    chem_id,
    xa,
    log_pgas_guess=10.0,
    use_skye=true,
    use_pc=false,
    use_cms=false,
    use_free_eos=true,
    use_opal_scvh=true
))]
#[allow(clippy::too_many_arguments)]
pub fn eos_eval_pt_get_pgas<'py>(
    py: Python<'py>,
    temp: f64,
    which_other: usize,
    other_value: f64,
    chem_id: Vec<usize>,
    xa: PyReadonlyArray1<'py, f64>,
    log_pgas_guess: f64,
    use_skye: bool,
    use_pc: bool,
    use_cms: bool,
    use_free_eos: bool,
    use_opal_scvh: bool,
) -> PyResult<PyInvTuple<'py>> {
    let xa_s = xa
        .as_slice()
        .map_err(|_| PyValueError::new_err("xa must be contiguous"))?;
    let cfg = build_eos_config(
        use_skye,
        use_pc,
        true,
        "PC",
        use_cms,
        use_free_eos,
        use_opal_scvh,
    )?;
    let inv = eos_pt_get_pgas(
        HelmTable::builtin(),
        &chem_id,
        xa_s,
        temp.log10(),
        which_other,
        other_value,
        log_pgas_guess,
        1.0e-8,
        LN10 * 1.0e-8,
        100,
        &cfg,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let (res, d_dlnd, d_dln_t) = into_py_eos_tuple(py, &inv.pt.eos);
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

/// Evaluates `OPAL/SCVH` 4D tabulated EOS across `nz` zones at `(chem_id, xa)`,
/// returning an `(nz, 78)` array `[res[0..26], d_dlnd[0..26], d_dln_t[0..26]]`.
#[pyfunction]
#[pyo3(signature = (rho, temp, chem_id, xa, parallel=true))]
pub fn eos_eval_opal_scvh_zones<'py>(
    py: Python<'py>,
    rho: PyReadonlyArray1<'py, f64>,
    temp: PyReadonlyArray1<'py, f64>,
    chem_id: Vec<usize>,
    xa: PyReadonlyArray1<'py, f64>,
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
    let xa_v = xa
        .as_slice()
        .map_err(|_| PyValueError::new_err("xa must be contiguous"))?
        .to_vec();
    let nz = r_s.len();
    if t_s.len() != nz {
        return Err(PyValueError::new_err(
            "rho and temp must have equal length",
        ));
    }
    let cat = NuclideCatalog::builtin_ref();
    let comp = get_composition_info(cat, &chem_id, &xa_v, true)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dt = EosDtTableSet::builtin();
    let cfg = OpalScvhConfig::default();
    let out_arr = eval_batch_2d::<78, _>(py, nz, parallel, 8, move |k| {
        let mut row = [f64::NAN; 78];
        if let Ok((mut out, _, _)) = get1_eosdt_results(
            dt,
            EosDtTableKind::OpalScvh,
            comp.xz,
            comp.xh,
            r_s[k].log10(),
            t_s[k].log10(),
            cfg.use_linear_interp_for_x,
            cfg.tiny_fuzz,
        ) {
            out.set_single_component_fraction(I_FRAC_OPAL_SCVH);
            let n = NUM_EOS_BASIC_RESULTS;
            row[0..n].copy_from_slice(&out.res);
            row[n..2 * n].copy_from_slice(&out.d_dlnd);
            row[2 * n..3 * n].copy_from_slice(&out.d_dln_t);
        }
        row
    })?;
    if out_arr.readonly().as_slice()?.iter().any(|v| v.is_nan()) {
        return Err(PyValueError::new_err(
            "OPAL/SCVH evaluation failed in eos_eval_opal_scvh_zones",
        ));
    }
    Ok(out_arr)
}

/// Registers Wave 3a.3 and Wave 3b.1 `eosDT_get` cascade, 4D table evaluators,
/// and inverse solver bindings on the `flexmesa._rust` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(eos_eval_dt_get, m)?)?;
    m.add_function(wrap_pyfunction!(eos_eval_pt_get, m)?)?;
    m.add_function(wrap_pyfunction!(eos_eval_opal_scvh, m)?)?;
    m.add_function(wrap_pyfunction!(eos_eval_opal_scvh_zones, m)?)?;
    m.add_function(wrap_pyfunction!(eos_eval_free_eos, m)?)?;
    m.add_function(wrap_pyfunction!(eos_eval_cms, m)?)?;
    m.add_function(wrap_pyfunction!(eos_eval_dt_get_t, m)?)?;
    m.add_function(wrap_pyfunction!(eos_eval_dt_get_rho, m)?)?;
    m.add_function(wrap_pyfunction!(eos_eval_pt_get_t, m)?)?;
    m.add_function(wrap_pyfunction!(eos_eval_pt_get_pgas, m)?)?;
    m.add_function(wrap_pyfunction!(eos_eval_gamma_de_get, m)?)?;
    Ok(())
}
