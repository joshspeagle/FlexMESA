// SPDX-License-Identifier: LGPL-3.0-or-later
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

/// Tier B bindings for `atm` (surface boundary conditions, $T(\tau)$, tables).
#[allow(unsafe_code)]
pub mod atm_py;
/// Tier A and Tier B bindings for `chem` (elements, nuclides, solar scales,
/// composition).
#[allow(unsafe_code)]
pub mod chem_py;
/// Tier B bindings for `colors` (Kurucz2003 SEDs, bolometric and synthetic
/// photometry).
#[allow(unsafe_code)]
pub mod colors_py;
/// Tier B bindings for `eos` Wave 3a.2 (`HELM` equation of state).
#[allow(unsafe_code)]
pub mod eos_helm_py;
/// Tier B bindings for `eos` Wave 3a.3 (`PC` and `Skye` equations of state).
#[allow(unsafe_code)]
pub mod eos_pc_skye_py;
/// Tier B bindings for `eos` Wave 3a (`ideal`, `gauss_fermi`, `ion_offset`).
#[allow(unsafe_code)]
pub mod eos_py;
/// Tier B bindings for `eos` Wave 3a.3 (`eosDT_get` blend and inverse solvers).
#[allow(unsafe_code)]
pub mod eos_solver_py;
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
/// Tier B bindings for `kap` (`kap` radiative, conductive, and Compton opacities).
#[allow(unsafe_code)]
pub mod kap_py;
/// Tier A/B/C bindings for `auto_diff`, `math`, and `mtx`.
#[allow(unsafe_code)]
pub mod math_solvers;
/// Tier B/C bindings for `net` (nuclear networks, analytical Jacobians, burners).
#[allow(unsafe_code)]
pub mod net_py;
/// Tier B bindings for `neu` (Itoh et al. 1996 thermal neutrino losses).
#[allow(unsafe_code)]
pub mod neu_py;
/// Tier B bindings for `rates` (nuclear reaction rates, screening, and weak rates).
#[allow(unsafe_code)]
pub mod rates_py;
/// Tier A bindings for `data` (`SHA-256` hashing and `.safetensors` parsers).
#[allow(unsafe_code)]
pub mod tier_a_data;
/// Tier A bindings for `utils`, `testing`, and `schema`.
#[allow(unsafe_code)]
pub mod tier_a_parse;
/// Tier B bindings for `turb` (`MLT` and `TDC` convective transport).
#[allow(unsafe_code)]
pub mod turb_py;

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
    Ok(())
}
