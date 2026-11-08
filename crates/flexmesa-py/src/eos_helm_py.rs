// SPDX-License-Identifier: LGPL-3.0-or-later
//! `PyO3` bindings for `flexmesa-eos` Wave 3a.2 (`HELM` equation of state).

use flexmesa_eos::{
    eval_helm_eos, locate_log_q, locate_log_t, EosDtTableKind, EosDtTableSet,
    HelmTable, NUM_EOS_BASIC_RESULTS,
};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::eval_batch_2d;

type PyHelmTuple<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

type PyHelmCellTuple<'py> = (
    f64,
    f64,
    f64,
    f64,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

type PyHelmCellsBatchTuple<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
);

/// Evaluates the 26-channel `HELM` equation of state at a single zone
/// `(temp, rho, abar, zbar)`, returning
/// `(res, d_dlnd, d_dln_t, d_dabar, d_dzbar)`.
#[pyfunction]
pub fn eos_eval_helm<'py>(
    py: Python<'py>,
    temp: f64,
    rho: f64,
    abar: f64,
    zbar: f64,
) -> PyResult<PyHelmTuple<'py>> {
    let (out, d_dabar, d_dzbar) = eval_helm_eos(temp, rho, abar, zbar)
        .map_err(|code| {
            PyValueError::new_err(format!(
                "eval_helm_eos failed with ierr = {code}"
            ))
        })?;
    Ok((
        PyArray1::from_slice(py, &out.res),
        PyArray1::from_slice(py, &out.d_dlnd),
        PyArray1::from_slice(py, &out.d_dln_t),
        PyArray1::from_slice(py, &d_dabar),
        PyArray1::from_slice(py, &d_dzbar),
    ))
}

/// Evaluates `eval_helm_eos` across `nz` zones, returning an `(nz, 130)` array
/// packing `[res[0..26], d_dlnd[0..26], d_dln_t[0..26], d_dabar[0..26],
/// d_dzbar[0..26]]`.
#[pyfunction]
#[pyo3(signature = (temp, rho, abar, zbar, parallel=true))]
pub fn eos_eval_helm_zones<'py>(
    py: Python<'py>,
    temp: PyReadonlyArray1<'py, f64>,
    rho: PyReadonlyArray1<'py, f64>,
    abar: PyReadonlyArray1<'py, f64>,
    zbar: PyReadonlyArray1<'py, f64>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let t_s = temp
        .as_slice()
        .map_err(|_| PyValueError::new_err("temp must be contiguous"))?
        .to_vec();
    let r_s = rho
        .as_slice()
        .map_err(|_| PyValueError::new_err("rho must be contiguous"))?
        .to_vec();
    let a_s = abar
        .as_slice()
        .map_err(|_| PyValueError::new_err("abar must be contiguous"))?
        .to_vec();
    let z_s = zbar
        .as_slice()
        .map_err(|_| PyValueError::new_err("zbar must be contiguous"))?
        .to_vec();
    let nz = t_s.len();
    if r_s.len() != nz || a_s.len() != nz || z_s.len() != nz {
        return Err(PyValueError::new_err(
            "temp, rho, abar, and zbar must have equal length",
        ));
    }
    eval_batch_2d::<130, _>(py, nz, parallel, 64, move |k| {
        let mut row = [f64::NAN; 130];
        if let Ok((out, d_dabar, d_dzbar)) =
            eval_helm_eos(t_s[k], r_s[k], a_s[k], z_s[k])
        {
            let n = NUM_EOS_BASIC_RESULTS;
            row[0..n].copy_from_slice(&out.res);
            row[n..2 * n].copy_from_slice(&out.d_dlnd);
            row[2 * n..3 * n].copy_from_slice(&out.d_dln_t);
            row[3 * n..4 * n].copy_from_slice(&d_dabar);
            row[4 * n..5 * n].copy_from_slice(&d_dzbar);
        }
        row
    })
}

/// Returns the `HelmCell` data `(t_lo, dth, d_lo, dd, fi_f, fi_dpdf, fi_ef,
/// fi_xf)` for a single `(log_temp, din)` lookup.
#[pyfunction]
pub fn eos_helm_cell_data<'py>(
    py: Python<'py>,
    log_temp: f64,
    din: f64,
) -> PyHelmCellTuple<'py> {
    let c = HelmTable::builtin().cell(log_temp, din);
    (
        c.t_lo,
        c.dth,
        c.d_lo,
        c.dd,
        PyArray1::from_slice(py, &c.fi_f),
        PyArray1::from_slice(py, &c.fi_dpdf),
        PyArray1::from_slice(py, &c.fi_ef),
        PyArray1::from_slice(py, &c.fi_xf),
    )
}

/// Returns batched `HelmCell` arrays `(t_lo, dth, d_lo, dd, fi_f, fi_dpdf,
/// fi_ef, fi_xf)` across `nz` zones for vectorized `@jax.jit` evaluation.
#[pyfunction]
pub fn eos_helm_cells_batch<'py>(
    py: Python<'py>,
    log_temp: PyReadonlyArray1<'py, f64>,
    din: PyReadonlyArray1<'py, f64>,
) -> PyResult<PyHelmCellsBatchTuple<'py>> {
    let lt_s = log_temp
        .as_slice()
        .map_err(|_| PyValueError::new_err("log_temp must be contiguous"))?;
    let din_s = din
        .as_slice()
        .map_err(|_| PyValueError::new_err("din must be contiguous"))?;
    let nz = lt_s.len();
    if din_s.len() != nz {
        return Err(PyValueError::new_err(
            "log_temp and din must have equal length",
        ));
    }
    let table = HelmTable::builtin();
    let mut t_lo = Vec::with_capacity(nz);
    let mut dth = Vec::with_capacity(nz);
    let mut d_lo = Vec::with_capacity(nz);
    let mut dd = Vec::with_capacity(nz);
    let mut fi_f = Vec::with_capacity(nz * 36);
    let mut fi_dpdf = Vec::with_capacity(nz * 16);
    let mut fi_ef = Vec::with_capacity(nz * 16);
    let mut fi_xf = Vec::with_capacity(nz * 16);
    for k in 0..nz {
        let c = table.cell(lt_s[k], din_s[k]);
        t_lo.push(c.t_lo);
        dth.push(c.dth);
        d_lo.push(c.d_lo);
        dd.push(c.dd);
        fi_f.extend_from_slice(&c.fi_f);
        fi_dpdf.extend_from_slice(&c.fi_dpdf);
        fi_ef.extend_from_slice(&c.fi_ef);
        fi_xf.extend_from_slice(&c.fi_xf);
    }
    Ok((
        PyArray1::from_vec(py, t_lo),
        PyArray1::from_vec(py, dth),
        PyArray1::from_vec(py, d_lo),
        PyArray1::from_vec(py, dd),
        PyArray1::from_vec(py, fi_f).reshape([nz, 36])?,
        PyArray1::from_vec(py, fi_dpdf).reshape([nz, 16])?,
        PyArray1::from_vec(py, fi_ef).reshape([nz, 16])?,
        PyArray1::from_vec(py, fi_xf).reshape([nz, 16])?,
    ))
}

/// Returns batched `OPAL/SCVH` cell bounds and 64-coefficient vertex arrays
/// `(q0, q1, t0, t1, v00, v10, v01, v11)` across `nz` zones for `(ix, iz)`.
#[pyfunction]
#[pyo3(signature = (log_rho, log_temp, ix_1based=5, iz_1based=2))]
pub fn eos_opal_scvh_cells_batch<'py>(
    py: Python<'py>,
    log_rho: PyReadonlyArray1<'py, f64>,
    log_temp: PyReadonlyArray1<'py, f64>,
    ix_1based: usize,
    iz_1based: usize,
) -> PyResult<PyHelmCellsBatchTuple<'py>> {
    let lr_s = log_rho
        .as_slice()
        .map_err(|_| PyValueError::new_err("log_rho must be contiguous"))?;
    let lt_s = log_temp
        .as_slice()
        .map_err(|_| PyValueError::new_err("log_temp must be contiguous"))?;
    let nz = lr_s.len();
    if lt_s.len() != nz {
        return Err(PyValueError::new_err(
            "log_rho and log_temp must have equal length",
        ));
    }
    if ix_1based < 3 {
        return Err(PyValueError::new_err("ix_1based must be >= 3"));
    }
    let dt = EosDtTableSet::builtin();
    let eps = [
        dt.get_table(EosDtTableKind::OpalScvh, ix_1based - 2, iz_1based)
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
        dt.get_table(EosDtTableKind::OpalScvh, ix_1based - 1, iz_1based)
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
        dt.get_table(EosDtTableKind::OpalScvh, ix_1based, iz_1based)
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
    ];
    let w = [-0.125_f64, 0.75_f64, 0.375_f64];
    let mut q0 = Vec::with_capacity(nz);
    let mut q1 = Vec::with_capacity(nz);
    let mut t0 = Vec::with_capacity(nz);
    let mut t1 = Vec::with_capacity(nz);
    let mut v00 = Vec::with_capacity(nz * 64);
    let mut v10 = Vec::with_capacity(nz * 64);
    let mut v01 = Vec::with_capacity(nz * 64);
    let mut v11 = Vec::with_capacity(nz * 64);
    let blend_v = |i: usize, j: usize| -> PyResult<[f64; 64]> {
        let a = eps[0]
            .get_vertex(i, j)
            .ok_or_else(|| PyValueError::new_err("missing vertex"))?;
        let b = eps[1]
            .get_vertex(i, j)
            .ok_or_else(|| PyValueError::new_err("missing vertex"))?;
        let c = eps[2]
            .get_vertex(i, j)
            .ok_or_else(|| PyValueError::new_err("missing vertex"))?;
        let mut out = [0.0_f64; 64];
        for (m, item) in out.iter_mut().enumerate() {
            *item =
                w[0] * a.coeffs[m] + w[1] * b.coeffs[m] + w[2] * c.coeffs[m];
        }
        Ok(out)
    };
    for k in 0..nz {
        let lq = lr_s[k] - 2.0 * lt_s[k] + 12.0;
        let (iq, lq0, _, lq1) = locate_log_q(&eps[0], lq, true)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let (jt, lt0, _, lt1) = locate_log_t(&eps[0], lt_s[k], true)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        q0.push(lq0);
        q1.push(lq1);
        t0.push(lt0);
        t1.push(lt1);
        v00.extend_from_slice(&blend_v(iq, jt)?);
        v10.extend_from_slice(&blend_v(iq + 1, jt)?);
        v01.extend_from_slice(&blend_v(iq, jt + 1)?);
        v11.extend_from_slice(&blend_v(iq + 1, jt + 1)?);
    }
    Ok((
        PyArray1::from_vec(py, q0),
        PyArray1::from_vec(py, q1),
        PyArray1::from_vec(py, t0),
        PyArray1::from_vec(py, t1),
        PyArray1::from_vec(py, v00).reshape([nz, 64])?,
        PyArray1::from_vec(py, v10).reshape([nz, 64])?,
        PyArray1::from_vec(py, v01).reshape([nz, 64])?,
        PyArray1::from_vec(py, v11).reshape([nz, 64])?,
    ))
}

/// Registers Wave 3a.2 `HELM` and `OPAL/SCVH` cell batch bindings.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(eos_eval_helm, m)?)?;
    m.add_function(wrap_pyfunction!(eos_eval_helm_zones, m)?)?;
    m.add_function(wrap_pyfunction!(eos_helm_cell_data, m)?)?;
    m.add_function(wrap_pyfunction!(eos_helm_cells_batch, m)?)?;
    m.add_function(wrap_pyfunction!(eos_opal_scvh_cells_batch, m)?)?;
    Ok(())
}
