// SPDX-License-Identifier: LGPL-3.0-or-later
//! Tier B (`neu`) `PyO3` bindings for Itoh et al. (1996) 5-channel thermal
//! neutrino energy-loss rates and analytical logarithmic partial derivatives.
//!
//! Exposes both single-zone (`neu_get`) and contiguous `float64` multi-zone
//! batch (`neu_get_batch`) evaluators with persistent 4-thread `rayon`
//! data parallelism across stellar zones.

use flexmesa_neu::{neu_get as rust_neu_get, NeuFlags};
use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::eval_batch_2d;

/// Parses a 5-boolean array `[pair, plas, phot, brem, reco]` into
/// [`NeuFlags`].
///
/// The boolean order matches upstream MESA `neu_def` channel indices:
/// 0 = pair annihilation, 1 = plasmon decay, 2 = photoneutrino,
/// 3 = bremsstrahlung, and 4 = recombination.
const fn flags_from_arr(f: [bool; 5]) -> NeuFlags {
    NeuFlags {
        pair: f[0],
        plas: f[1],
        phot: f[2],
        brem: f[3],
        reco: f[4],
    }
}

/// Evaluates `neu_get` at a single zone, returning `(loss, sources)`.
///
/// Matches upstream MESA `neu_lib::neu_get` across all 5 Itoh et al. (1996)
/// thermal neutrino channels (`pair`, `plas`, `phot`, `brem`, `reco`).
///
/// # Arguments
/// - `t`, `log10_t`: temperature $T$ in $\mathrm{K}$ and $\log_{10} T$.
/// - `rho`, `log10_rho`: density $\rho$ in $\mathrm{g\,cm^{-3}}$ and
///   $\log_{10} \rho$.
/// - `abar`, `zbar`: mean atomic weight $\bar{A}$ and mean charge $\bar{Z}$.
/// - `log10_tlim`: lower temperature threshold $\log_{10} T_{\mathrm{lim}}$.
/// - `flags`: 5-element boolean channel mask.
///
/// # Errors
/// Raises `ValueError` if neither `t` nor `log10_t` (or neither `rho` nor
/// `log10_rho`) is provided or if any thermodynamic input is non-positive.
#[pyfunction]
#[pyo3(signature = (
    t=None,
    log10_t=None,
    rho=None,
    log10_rho=None,
    abar=4.0,
    zbar=2.0,
    log10_tlim=7.5,
    flags=[true, true, true, true, true]
))]
fn neu_get(
    t: Option<f64>,
    log10_t: Option<f64>,
    rho: Option<f64>,
    log10_rho: Option<f64>,
    abar: f64,
    zbar: f64,
    log10_tlim: f64,
    flags: [bool; 5],
) -> PyResult<([f64; 5], [[f64; 5]; 5])> {
    let out = rust_neu_get(
        t,
        log10_t,
        rho,
        log10_rho,
        abar,
        zbar,
        log10_tlim,
        flags_from_arr(flags),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((out.loss, out.sources))
}

/// Evaluates `neu_get` across 1D arrays `(t, log10_t, rho, log10_rho, abar,
/// zbar)` of shape `(nz,)`, returning a 2D `(nz, 5)` array of `loss`.
///
/// Each row $k$ contains `[sneu, dsneu_dT, dsneu_dRho, dsneu_dAbar,
/// dsneu_dZbar]` in $\mathrm{erg\,g^{-1}\,s^{-1}}$ and corresponding partial
/// derivatives. Dispatches across the persistent 4-thread `rayon` worker pool
/// when `parallel=True` and `nz >= 64`.
#[pyfunction]
#[pyo3(signature = (
    t,
    log10_t,
    rho,
    log10_rho,
    abar,
    zbar,
    log10_tlim=7.5,
    flags=[true, true, true, true, true],
    parallel=false
))]
fn neu_get_batch<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    log10_t: PyReadonlyArray1<'py, f64>,
    rho: PyReadonlyArray1<'py, f64>,
    log10_rho: PyReadonlyArray1<'py, f64>,
    abar: PyReadonlyArray1<'py, f64>,
    zbar: PyReadonlyArray1<'py, f64>,
    log10_tlim: f64,
    flags: [bool; 5],
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let ts = t
        .as_slice()
        .map_err(|_| PyValueError::new_err("t must be C-contiguous"))?;
    let lts = log10_t
        .as_slice()
        .map_err(|_| PyValueError::new_err("log10_t must be C-contiguous"))?;
    let rhos = rho
        .as_slice()
        .map_err(|_| PyValueError::new_err("rho must be C-contiguous"))?;
    let lrhos = log10_rho
        .as_slice()
        .map_err(|_| PyValueError::new_err("log10_rho must be C-contiguous"))?;
    let as_ = abar
        .as_slice()
        .map_err(|_| PyValueError::new_err("abar must be C-contiguous"))?;
    let zs = zbar
        .as_slice()
        .map_err(|_| PyValueError::new_err("zbar must be C-contiguous"))?;
    let nz = ts.len();
    if lts.len() != nz
        || rhos.len() != nz
        || lrhos.len() != nz
        || as_.len() != nz
        || zs.len() != nz
    {
        return Err(PyValueError::new_err(
            "neu_get_batch input length mismatch",
        ));
    }
    let flg = flags_from_arr(flags);
    eval_batch_2d(py, nz, parallel, 64, |k| {
        rust_neu_get(
            Some(ts[k]),
            Some(lts[k]),
            Some(rhos[k]),
            Some(lrhos[k]),
            as_[k],
            zs[k],
            log10_tlim,
            flg,
        )
        .map(|r| r.loss)
        .unwrap_or([f64::NAN; 5])
    })
}

/// Registers Tier B `neu` functions on `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(neu_get, m)?)?;
    m.add_function(wrap_pyfunction!(neu_get_batch, m)?)?;
    Ok(())
}
