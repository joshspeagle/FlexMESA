// SPDX-License-Identifier: LGPL-3.0-or-later
//! Compiled `PyO3` extension bridge (`flexmesa._rust`) for `FlexMESA`.
//!
//! Exposes the canonical Safe Rust (`crates/`) implementations of:
//! - **Tier A (`utils`, `schema`, `data`, `testing`, `auto_diff`)**:
//!   Infrastructure parsers, `.safetensors` table converters and `SHA-256`
//!   hashing, golden baseline parsers, and 1st/2nd-order dual-number kernels.
//! - **Tier B (`math`, `interp_1d`, `interp_2d`, `neu`, `turb`, `chem`,
//!   `ionization`)**:
//!   Domain-safe elementary intrinsics, 1D/2D spline builders and evaluators,
//!   Itoh et al. (1996) 5-channel thermal neutrino losses, and Cox (1968) /
//!   Kuhfuss (1986) convective transport (`MLT` and `TDC`) with persistent
//!   4-thread `rayon` data parallelism.
//! - **Tier C (`mtx`, `num`)**:
//!   Dense, banded, sequential block-Thomas, and parallel `bcyclic` odd-even
//!   cyclic reduction solvers, plus bracketed root-finding and minimization.

#![deny(unsafe_code, missing_docs)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::needless_pass_by_value,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_wraps
)]

/// Tier A and Tier B bindings for `chem` (elements, nuclides, solar scales,
/// composition).
#[allow(unsafe_code)]
pub mod chem_py;
/// Tier B bindings for 1D piecewise-monotonic and MP cubic interpolation.
#[allow(unsafe_code)]
pub mod interp_1d;
/// Tier B bindings for 2D bicubic and piecewise-monotonic interpolation.
#[allow(unsafe_code)]
pub mod interp_2d;
/// Tier B bindings for `ionization` (`Fe56` in `He4`, 4D tables,
/// `typical_charge`).
#[allow(unsafe_code)]
pub mod ionization_py;
/// Tier A/B/C bindings for `auto_diff`, `math`, and `mtx`.
#[allow(unsafe_code)]
pub mod math_solvers;
/// Tier B bindings for `neu` (Itoh et al. 1996 thermal neutrino losses).
#[allow(unsafe_code)]
pub mod neu_py;
/// Tier A bindings for `data` (`SHA-256` hashing and `.safetensors` parsers).
#[allow(unsafe_code)]
pub mod tier_a_data;
/// Tier A bindings for `utils`, `testing`, and `schema`.
#[allow(unsafe_code)]
pub mod tier_a_parse;
/// Tier B bindings for `turb` (`MLT` and `TDC` convective transport).
#[allow(unsafe_code)]
pub mod turb_py;

use flexmesa_bench::stage3_pool;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Evaluates `eval_zone(k)` across `nz` zones into a `(nz, W)` `f64` array.
///
/// Dispatches across the persistent 4-thread `rayon` worker pool when
/// `parallel == true` and `nz >= min_par`, and executes sequentially otherwise.
pub(crate) fn eval_batch_2d<'py, const W: usize, F>(
    py: Python<'py>,
    nz: usize,
    parallel: bool,
    min_par: usize,
    eval_zone: F,
) -> PyResult<Bound<'py, PyArray2<f64>>>
where
    F: Fn(usize) -> [f64; W] + Sync,
{
    let mut out = vec![0.0_f64; nz * W];
    let (rows, _) = out.as_chunks_mut::<W>();
    if parallel && nz >= min_par {
        stage3_pool().install(|| {
            rows.par_iter_mut().enumerate().for_each(|(k, dst)| {
                *dst = eval_zone(k);
            });
        });
    } else {
        for (k, dst) in rows.iter_mut().enumerate() {
            *dst = eval_zone(k);
        }
    }
    PyArray1::from_vec(py, out).reshape([nz, W])
}

/// Registers all Tier A, Tier B, and Tier C functions on the `flexmesa._rust`
/// Python extension module.
#[allow(unsafe_code)]
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    tier_a_parse::register(m)?;
    tier_a_data::register(m)?;
    math_solvers::register(m)?;
    interp_1d::register(m)?;
    interp_2d::register(m)?;
    neu_py::register(m)?;
    turb_py::register(m)?;
    chem_py::register(m)?;
    ionization_py::register(m)?;
    Ok(())
}
