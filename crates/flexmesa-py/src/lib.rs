// SPDX-License-Identifier: GPL-3.0-or-later
//! Compiled `PyO3` extension bridge (`flexmesa._rust`) for `FlexMESA`.
//!
//! Exposes the canonical Safe Rust (`crates/`) implementations of:
//! - **Tier A (`utils`, `schema`, `data`, `testing`, `auto_diff`)**:
//!   Infrastructure parsers, `.safetensors` table converters and `SHA-256`
//!   hashing, golden baseline parsers, and 1st/2nd-order dual-number kernels.
//! - **Tier B (`math`, `interp_1d`, `interp_2d`, `neu`, `turb`, `chem`,
//!   `ionization`, `colors`)**:
//!   Domain-safe elementary intrinsics, 1D/2D spline builders and evaluators,
//!   Itoh et al. (1996) 5-channel thermal neutrino losses, and Cox (1968) /
//!   Kuhfuss (1986) convective transport (`MLT` and `TDC`) with persistent
//!   4-thread `rayon` data parallelism.
//! - **Tier C (`mtx`, `num`)**:
//!   Dense, banded, sequential block-Thomas, and parallel `bcyclic` odd-even
//!   cyclic reduction solvers, plus bracketed root-finding and minimization.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::elidable_lifetime_names,
    clippy::fn_params_excessive_bools,
    clippy::implicit_clone,
    clippy::many_single_char_names,
    clippy::map_unwrap_or,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::needless_pass_by_value,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_wraps,
    clippy::unreadable_literal
)]

/// Tier B/C bindings for Wave 5.4 `astero` (scaling relations, surface effects, optimization).
pub mod astero_py;
/// PyO3 adapter for an `astero` case's own `set_param` (test-side extras).
pub mod astero_case_hooks_py;
/// Tier B bindings for `atm` (surface boundary conditions, $T(\tau)$, tables).
pub mod atm_py;
/// Tier B/C bindings for Wave 5.3 `binary` (orbital mechanics, RLOF, tides, binary evolution).
pub mod binary_py;
/// Tier A and Tier B bindings for `chem` (elements, nuclides, solar scales,
/// composition).
pub mod chem_py;
/// Tier B bindings for `colors` (Kurucz2003 SEDs, bolometric and synthetic
/// photometry).
pub mod colors_py;
/// Tier B bindings for `eos` Wave 3a.2 (`HELM` equation of state).
pub mod eos_helm_py;
/// Tier B bindings for `eos` Wave 3a.3 (`PC` and `Skye` equations of state).
pub mod eos_pc_skye_py;
/// Tier B bindings for `eos` Wave 3a (`ideal`, `gauss_fermi`, `ion_offset`).
pub mod eos_py;
/// Tier B bindings for `eos` Wave 3a.3 (`eosDT_get` blend and inverse solvers).
pub mod eos_solver_py;
/// PyO3 callback adapters and RSP/GYRE helpers for test-side extras.
pub mod extras_py;
/// Tier B bindings for 1D piecewise-monotonic and MP cubic interpolation.
pub mod interp_1d;
/// Tier B bindings for 2D bicubic and piecewise-monotonic interpolation.
pub mod interp_2d;
/// Tier B bindings for `ionization` (`Fe56` in `He4`, 4D tables,
/// `typical_charge`).
pub mod ionization_py;
/// Tier B bindings for `kap` (`kap` radiative, conductive, and Compton opacities).
pub mod kap_py;
/// Level-2 exact-hex test-suite probes for `math` and `auto_diff`.
pub mod math_hex_probe;
/// Tier A/B/C bindings for `auto_diff`, `math`, and `mtx`.
pub mod math_solvers;
/// Reference-vector `Strict` digest, `Fast`-vs-`Strict` audit, and dispatch timing.
pub mod math_vec_audit;
/// Tier B/C bindings for `net` (nuclear networks, analytical Jacobians, burners).
pub mod net_py;
/// Tier B bindings for `neu` (Itoh et al. 1996 thermal neutrino losses).
pub mod neu_py;
/// Tier B bindings for `rates` (nuclear reaction rates, screening, and weak rates).
pub mod rates_py;
/// PyO3 bindings for `star` end-to-end `inlist` parsing and `test_suite` runner.
pub mod star_e2e_py;
/// Internal helper routines for [`star_py::PyStarModel`].
pub mod star_helpers_py;
/// Tier A/B/C bindings for `star` (stellar structure, ZAMS/pre-MS, I/O, microphysics).
pub mod star_py;
/// Tier A/B bindings for Wave 5.2 `star` hydrodynamics and pulsation export.
pub mod star_wave5_2_py;
/// Tier B/C bindings for Wave 5.1 `star` extensions (diffusion, rotation, premix).
pub mod star_wave5_py;
/// Tier A bindings for `data` (`SHA-256` hashing and `.safetensors` parsers).
pub mod tier_a_data;
/// Tier A bindings for `utils`, `testing`, and `schema`.
pub mod tier_a_parse;
/// PyO3 bindings for `flexmesa trace` (`star`, `binary`, `astero`) and MESA log reading.
pub mod trace_py;
/// Tier B bindings for `turb` (`MLT` and `TDC` convective transport).
pub mod turb_py;
/// Batch PyO3 bindings for the Wave 5 benchmark suite.
pub mod wave5_bench_py;

use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, Thread};

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

const STATE_IDLE: u8 = 0;
const STATE_READY: u8 = 1;
const STATE_PARKED: u8 = 2;
const SPIN_LIMIT: u32 = 200_000;

/// Single persistent worker thread slot with adaptive spin-then-park wakeup.
struct WorkerSlot {
    state: AtomicU8,
    task: Mutex<Option<Box<dyn FnOnce() + Send>>>,
    thread: OnceLock<Thread>,
}

/// Returns the 3 persistent background worker slots (forming a 4-thread team
/// together with the calling thread).
fn worker_slots() -> &'static [Arc<WorkerSlot>; 3] {
    static SLOTS: OnceLock<[Arc<WorkerSlot>; 3]> = OnceLock::new();
    SLOTS.get_or_init(|| {
        std::array::from_fn(|_| {
            let slot = Arc::new(WorkerSlot {
                state: AtomicU8::new(STATE_IDLE),
                task: Mutex::new(None),
                thread: OnceLock::new(),
            });
            let slot_c = Arc::clone(&slot);
            let handle = thread::spawn(move || {
                let mut spins = 0_u32;
                loop {
                    if slot_c.state.load(Ordering::Acquire) == STATE_READY {
                        slot_c.state.store(STATE_IDLE, Ordering::Relaxed);
                        let job = slot_c.task.lock().unwrap().take();
                        if let Some(f) = job {
                            f();
                        }
                        spins = 0;
                    } else if spins < SPIN_LIMIT {
                        spins += 1;
                        std::hint::spin_loop();
                    } else if slot_c
                        .state
                        .compare_exchange(
                            STATE_IDLE,
                            STATE_PARKED,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        while slot_c.state.load(Ordering::Acquire)
                            == STATE_PARKED
                        {
                            thread::park();
                        }
                        spins = 0;
                    }
                }
            });
            let _ = slot.thread.set(handle.thread().clone());
            slot
        })
    })
}

/// Returns the global mutex serializing access to `worker_slots()`.
fn team_dispatch_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// Evaluates `eval_zone(k)` across `nz` zones into a `(nz, W)` `f64` array.
///
/// Dispatches across the persistent 4-thread adaptive spin-then-park worker
/// team when `parallel == true` and `nz >= min_par`, and executes sequentially
/// otherwise.
pub(crate) fn eval_batch_2d<'py, const W: usize, F>(
    py: Python<'py>,
    nz: usize,
    parallel: bool,
    min_par: usize,
    eval_zone: F,
) -> PyResult<Bound<'py, PyArray2<f64>>>
where
    F: Fn(usize) -> [f64; W] + Send + Sync + 'static,
{
    let mut out = vec![0.0_f64; nz * W];
    let (rows, _) = out.as_chunks_mut::<W>();
    if parallel && nz >= min_par {
        let eval_arc = Arc::new(eval_zone);
        let slots = worker_slots();
        py.detach(|| {
            let Ok(_guard) = team_dispatch_lock().try_lock() else {
                for (k, dst) in rows.iter_mut().enumerate() {
                    *dst = eval_arc(k);
                }
                return;
            };
            let chunk_z = nz.div_ceil(4).max(1);
            let done = Arc::new(AtomicUsize::new(0));
            let mut results: [Option<Arc<Mutex<Vec<[f64; W]>>>>; 3] =
                [None, None, None];
            let mut active = 0_usize;
            for (w, slot_ref) in slots.iter().enumerate() {
                let z0 = (w + 1) * chunk_z;
                if z0 >= nz {
                    break;
                }
                let z1 = (z0 + chunk_z).min(nz);
                let res_buf = Arc::new(Mutex::new(Vec::new()));
                results[w] = Some(Arc::clone(&res_buf));
                let eval_w = Arc::clone(&eval_arc);
                let done_w = Arc::clone(&done);
                *slot_ref.task.lock().unwrap() = Some(Box::new(move || {
                    let mut local = Vec::with_capacity(z1 - z0);
                    for k in z0..z1 {
                        local.push(eval_w(k));
                    }
                    *res_buf.lock().unwrap() = local;
                    done_w.fetch_add(1, Ordering::Release);
                }));
                if slot_ref.state.swap(STATE_READY, Ordering::AcqRel)
                    == STATE_PARKED
                {
                    if let Some(th) = slot_ref.thread.get() {
                        th.unpark();
                    }
                }
                active += 1;
            }
            let z0_end = chunk_z.min(nz);
            for (k, dst) in rows[..z0_end].iter_mut().enumerate() {
                *dst = eval_arc(k);
            }
            while done.load(Ordering::Acquire) < active {
                std::hint::spin_loop();
            }
            for (w, res_opt) in results.into_iter().enumerate() {
                if let Some(res_arc) = res_opt {
                    let z0 = (w + 1) * chunk_z;
                    let local = res_arc.lock().unwrap();
                    rows[z0..z0 + local.len()].copy_from_slice(&local);
                }
            }
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
#[allow(clippy::used_underscore_items)]
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    if let Ok(raw) = std::env::var("FLEXMESA_NUMERICS_MODE") {
        if !raw.trim().is_empty() {
            let mode =
                raw.parse::<flexmesa_math::NumericsMode>().map_err(|err| {
                    pyo3::exceptions::PyValueError::new_err(err.to_string())
                })?;
            flexmesa_math::set_numerics_mode(mode);
        }
    }
    flexmesa_math::math_init();
    tier_a_parse::register(m)?;
    tier_a_data::register(m)?;
    math_hex_probe::register(m)?;
    math_solvers::register(m)?;
    math_vec_audit::register(m)?;
    interp_1d::register(m)?;
    interp_2d::register(m)?;
    neu_py::register(m)?;
    turb_py::register(m)?;
    chem_py::register(m)?;
    colors_py::register(m)?;
    ionization_py::register(m)?;
    eos_py::register(m)?;
    eos_helm_py::register(m)?;
    eos_pc_skye_py::register(m)?;
    eos_solver_py::register(m)?;
    kap_py::register(m)?;
    rates_py::register(m)?;
    atm_py::register(m)?;
    net_py::register(m)?;
    star_py::register(m)?;
    extras_py::register(m)?;
    star_e2e_py::register(m)?;
    star_wave5_py::register(m)?;
    star_wave5_2_py::register(m)?;
    binary_py::register(m)?;
    astero_py::register(m)?;
    trace_py::register(m)?;
    wave5_bench_py::register(m)?;
    Ok(())
}
