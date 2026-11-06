//! Wave 1 four-stage performance benchmark harness (`flexmesa-bench`).
//!
//! Evaluates the Wave 1 leaf utilities and microphysics kernels (`auto_diff`,
//! `interp_2d`, `mtx`, `neu`, and `turb`) across four progressive optimization
//! stages in Safe Rust:
//! - **Stage 0 (Idiomatic Scalar Baseline)**: Zone-by-zone scalar calls with
//!   per-call output allocation.
//! - **Stage 1 (Contiguous Structure-of-Arrays Layout)**: Pre-allocated
//!   contiguous `f64` slices (`SoA`) across $N_{\mathrm{zones}}$ stellar zones.
//! - **Stage 2 (Fused Kernel & Hoisted Invariants)**: Inlined slice loops with
//!   hoisted table/grid validation and pre-allocated solver workspaces.
//! - **Stage 3 (Multi-Core Data Parallelism)**: Persistent `rayon` worker pool
//!   with lock-free atomic team barrier across 4 worker threads.

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::chunks_exact_to_as_chunks,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::manual_midpoint,
    clippy::many_single_char_names,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]

pub mod bicub_fast;
pub mod kernels_micro;
pub mod kernels_solver;
pub mod kernels_wave2;
pub mod tridiag_fast;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use flexmesa_interp2d::{interp_mkbicub_db, BicubicBc};
use flexmesa_math::{cospi, exp10, sinpi};
use rayon::ThreadPool;

pub use kernels_micro::{bench_autodiff, bench_interp2d, bench_neu};
pub use kernels_solver::{
    bench_mtx_block_tridiag, bench_turb_mlt, bench_turb_tdc,
};
pub use kernels_wave2::{
    bench_chem_comp, bench_colors_sed, bench_ion_4d, bench_ion_fe56,
    run_all_wave2_benchmarks,
};

/// Number of worker threads used in Stage 3 multi-core benchmarks.
pub const STAGE3_THREADS: usize = 4;

/// Returns a reference to the persistent 4-thread `rayon` worker pool used for
/// Stage 3 data-parallel benchmarks.
pub fn stage3_pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(STAGE3_THREADS)
            .build()
            .expect("failed to build persistent Stage 3 rayon thread pool")
    })
}

/// Computes median and interquartile range (`Q75 - Q25`) from raw timing
/// samples in microseconds.
fn stats_from_samples(mut samples: Vec<f64>) -> LatencyStats {
    let repeats = samples.len();
    samples.sort_by(f64::total_cmp);
    let mid = repeats / 2;
    let median_us = if repeats % 2 == 1 {
        samples[mid]
    } else {
        f64::midpoint(samples[mid - 1], samples[mid])
    };
    LatencyStats {
        median_us,
        iqr_us: samples[(3 * repeats) / 4] - samples[repeats / 4],
    }
}

/// Measures a static-chunked Stage 3 multi-core kernel `op_chunk(tid, chunk)`
/// across all `STAGE3_THREADS` workers of the persistent `rayon` pool using a
/// lock-free atomic team barrier.
///
/// # Arguments
/// * `n_warmup` - Number of untimed warmup invocations.
/// * `n_repeats` - Number of timed invocations (clamped to at least 3).
/// * `slice` - Mutable output slice partitioned into at most `STAGE3_THREADS`
///   chunks of length `chunk_len`.
/// * `chunk_len` - Number of elements assigned to each worker thread.
/// * `op_chunk` - Per-thread kernel receiving `(thread_idx, dst_chunk)`.
pub fn measure_stage3_chunks_us<T: Send, F: Fn(usize, &mut [T]) + Sync>(
    n_warmup: usize,
    n_repeats: usize,
    slice: &mut [T],
    chunk_len: usize,
    op_chunk: F,
) -> LatencyStats {
    let repeats = n_repeats.max(3);
    let (step, count) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let wait = || {
        let s = step.load(Ordering::Acquire);
        if count.fetch_add(1, Ordering::AcqRel) + 1 == STAGE3_THREADS {
            count.store(0, Ordering::Release);
            step.fetch_add(1, Ordering::Release);
        } else {
            while step.load(Ordering::Acquire) == s {
                std::hint::spin_loop();
            }
        }
    };
    let chunk_slots: Vec<Mutex<&mut [T]>> =
        slice.chunks_mut(chunk_len).map(Mutex::new).collect();
    assert!(chunk_slots.len() <= STAGE3_THREADS);
    let samples_mutex = Mutex::new(Vec::new());
    stage3_pool().broadcast(|ctx| {
        let tid = ctx.index();
        let mut guard = (tid < chunk_slots.len())
            .then(|| chunk_slots[tid].lock().unwrap());
        wait();
        for _ in 0..n_warmup {
            if let Some(ref mut dst) = guard {
                op_chunk(tid, dst);
            }
        }
        let mut local_samples =
            Vec::with_capacity(if tid == 0 { repeats } else { 0 });
        for _ in 0..repeats {
            wait();
            let t0 = (tid == 0).then(Instant::now);
            if let Some(ref mut dst) = guard {
                op_chunk(tid, dst);
            }
            wait();
            if let Some(start) = t0 {
                local_samples.push(start.elapsed().as_nanos() as f64 * 1.0e-3);
            }
        }
        if tid == 0 {
            *samples_mutex.lock().unwrap() = local_samples;
        }
    });
    stats_from_samples(samples_mutex.into_inner().unwrap())
}

/// Summary latency statistics (median and interquartile range in microseconds)
/// for a single optimization stage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatencyStats {
    /// Median execution latency in microseconds ($\mu\mathrm{s}$).
    pub median_us: f64,
    /// Interquartile range ($Q_{75} - Q_{25}$) in microseconds
    /// ($\mu\mathrm{s}$).
    pub iqr_us: f64,
}

/// Complete four-stage benchmark record for a single workload and zone count
/// $N_{\mathrm{zones}}$.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchRecord {
    /// Workload identifier (`"auto_diff"`, `"interp_2d"`, `"mtx_tridiag"`,
    /// `"neu"`, `"turb_mlt"`, or `"turb_tdc"`).
    pub workload: &'static str,
    /// Number of stellar zones ($N_{\mathrm{zones}}$).
    pub n_zones: usize,
    /// Stage 0 (idiomatic scalar baseline) latency.
    pub stage0: LatencyStats,
    /// Stage 1 (contiguous Structure-of-Arrays layout) latency.
    pub stage1: LatencyStats,
    /// Stage 2 (hoisted invariants and fused slice kernel) latency.
    pub stage2: LatencyStats,
    /// Stage 3 (4-thread data-parallel) latency.
    pub stage3: LatencyStats,
    /// Maximum relative discrepancy across Stages 0, 1, 2, and 3.
    pub max_rel_err: f64,
    /// Checksum scalar (`f64`) used for cross-language parity verification.
    pub checksum: f64,
}

/// Measures the median and interquartile range (in microseconds) of `op` across
/// `n_warmup` untimed warmup calls and `n_repeats` timed calls.
///
/// # Arguments
/// * `n_warmup` - Number of untimed warmup invocations.
/// * `n_repeats` - Number of timed invocations (clamped to at least 3).
/// * `op` - Closure to benchmark.
///
/// # Returns
/// [`LatencyStats`] containing the median and interquartile range in
/// $\mu\mathrm{s}$.
pub fn measure_us<F: FnMut()>(
    n_warmup: usize,
    n_repeats: usize,
    mut op: F,
) -> LatencyStats {
    for _ in 0..n_warmup {
        op();
    }
    let repeats = n_repeats.max(3);
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let t0 = Instant::now();
        op();
        samples.push(t0.elapsed().as_nanos() as f64 * 1.0e-3);
    }
    stats_from_samples(samples)
}

/// Computes the maximum relative difference between two `f64` slices `a` and
/// `b`, propagating `f64::NAN` if any element is non-finite.
///
/// # Arguments
/// * `a` - Reference `f64` slice.
/// * `b` - Candidate `f64` slice of matching length.
///
/// # Returns
/// Maximum elementwise relative discrepancy, or `f64::NAN` if any entry is
/// non-finite.
pub fn max_rel_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "slice length mismatch in max_rel_diff");
    let mut max_err = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if !x.is_finite() || !y.is_finite() {
            return f64::NAN;
        }
        let denom = x.abs().max(y.abs()).max(1.0e-30);
        let rel = (x - y).abs() / denom;
        if rel.is_nan() || rel > max_err {
            max_err = rel;
        }
    }
    max_err
}

/// Returns the maximum of `a` and `b` while propagating `f64::NAN`.
#[inline]
pub fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// Combines three stage relative errors while propagating `f64::NAN`.
///
/// # Arguments
/// * `d1` - Stage 1 relative error.
/// * `d2` - Stage 2 relative error.
/// * `d3` - Stage 3 relative error.
///
/// # Returns
/// Maximum of `(d1, d2, d3)`, or `f64::NAN` if any input is `NaN`.
pub fn max_err3(d1: f64, d2: f64, d3: f64) -> f64 {
    nan_max(nan_max(d1, d2), d3)
}

/// Deterministic stellar thermodynamic and composition profile across
/// $N_{\mathrm{zones}}$ shells.
///
/// Spans $\log_{10} T \in [7.2, 9.8]$ ($\mathrm{K}$), $\log_{10} \rho \in
/// [4.0, 9.0]$ ($\mathrm{g\,cm^{-3}}$), mean atomic weight $\bar{A} \in
/// [4.0, 28.0]$, and mean charge $\bar{Z} \in [2.0, 14.0]$.
#[derive(Debug, Clone)]
pub struct ThermoGrid {
    /// Number of stellar zones ($N_{\mathrm{zones}}$).
    pub n_zones: usize,
    /// Base-10 log temperature $\log_{10}(T/\mathrm{K})$ of length `n_zones`.
    pub log10_t: Vec<f64>,
    /// Temperature $T$ ($\mathrm{K}$) of length `n_zones`.
    pub t: Vec<f64>,
    /// Base-10 log density $\log_{10}(\rho/\mathrm{g\,cm^{-3}})$ of length
    /// `n_zones`.
    pub log10_rho: Vec<f64>,
    /// Density $\rho$ ($\mathrm{g\,cm^{-3}}$) of length `n_zones`.
    pub rho: Vec<f64>,
    /// Mean atomic mass $\bar{A}$ ($\mathrm{amu}$) of length `n_zones`.
    pub abar: Vec<f64>,
    /// Mean ionic charge $\bar{Z}$ ($e$) of length `n_zones`.
    pub zbar: Vec<f64>,
}

impl ThermoGrid {
    /// Constructs a deterministic stellar profile of `n_zones` shells.
    ///
    /// # Arguments
    /// * `n_zones` - Number of radial zones ($N_{\mathrm{zones}} \ge 1$).
    ///
    /// # Returns
    /// Initialized [`ThermoGrid`] across `n_zones` shells.
    pub fn new(n_zones: usize) -> Self {
        let (mut log10_t, mut t) =
            (Vec::with_capacity(n_zones), Vec::with_capacity(n_zones));
        let (mut log10_rho, mut rho) =
            (Vec::with_capacity(n_zones), Vec::with_capacity(n_zones));
        let (mut abar, mut zbar) =
            (Vec::with_capacity(n_zones), Vec::with_capacity(n_zones));
        let denom = (n_zones.saturating_sub(1)).max(1) as f64;
        for k in 0..n_zones {
            let s = k as f64 / denom;
            let (lt, lrho, a) = (7.2 + 2.6 * s, 4.0 + 5.0 * s, 4.0 + 24.0 * s);
            log10_t.push(lt);
            t.push(exp10(lt));
            log10_rho.push(lrho);
            rho.push(exp10(lrho));
            abar.push(a);
            zbar.push(0.5 * a);
        }
        Self { n_zones, log10_t, t, log10_rho, rho, abar, zbar }
    }
}

/// Precomputed $64 \times 64$ 2D bicubic spline table over
/// $(\log_{10}\rho, \log_{10}T)$ representative of stellar EOS and opacity
/// look-ups.
#[derive(Debug, Clone)]
pub struct BicubTable {
    /// Strictly ascending `x` grid ($\log_{10}\rho \in [3.5, 9.5]$), len `64`.
    pub x: Vec<f64>,
    /// Strictly ascending `y` grid ($\log_{10}T \in [7.0, 10.0]$), len `64`.
    pub y: Vec<f64>,
    /// Bicubic spline coefficients `[f, f_xx, f_yy, f_xxyy]`, len `64 * 64`.
    pub coeffs: Vec<[f64; 4]>,
    /// Uniform-spacing flag along `x`.
    pub ilinx: bool,
    /// Uniform-spacing flag along `y`.
    pub iliny: bool,
}

impl BicubTable {
    /// Builds the deterministic $64 \times 64$ bicubic spline table.
    pub fn new() -> Self {
        let (nx, ny) = (64usize, 64usize);
        let x: Vec<f64> = (0..nx)
            .map(|i| 3.5 + 6.0 * (i as f64) / ((nx - 1) as f64))
            .collect();
        let y: Vec<f64> = (0..ny)
            .map(|j| 7.0 + 3.0 * (j as f64) / ((ny - 1) as f64))
            .collect();
        let mut z = vec![0.0_f64; nx * ny];
        for (i, &xi) in x.iter().enumerate() {
            for (j, &yj) in y.iter().enumerate() {
                z[i * ny + j] = 0.25 * xi * xi
                    + 0.15 * xi * yj
                    + 0.35 * yj * yj
                    + 0.1 * sinpi(0.2 * xi) * cospi(0.15 * yj);
            }
        }
        let bc = BicubicBc::NotAKnot;
        let (coeffs, ilinx, iliny) = interp_mkbicub_db(
            &x, &y, &z, bc, &[], bc, &[], bc, &[], bc, &[],
        )
        .expect("valid bicubic table");
        Self { x, y, coeffs, ilinx, iliny }
    }
}

impl Default for BicubTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Runs the complete Wave 1 four-stage benchmark suite across `zone_counts` and
/// returns all [`BenchRecord`] entries.
///
/// # Arguments
/// * `zone_counts` - Slice of radial zone counts $N_{\mathrm{zones}}$.
/// * `n_warmup` - Number of untimed warmup iterations per stage.
/// * `n_repeats` - Number of timed iterations per stage.
///
/// # Returns
/// Vector of [`BenchRecord`] results across all six Wave 1 workloads.
pub fn run_all_wave1_benchmarks(
    zone_counts: &[usize],
    n_warmup: usize,
    n_repeats: usize,
) -> Vec<BenchRecord> {
    let table = BicubTable::new();
    let mut records = Vec::with_capacity(zone_counts.len() * 6);
    for &nz in zone_counts {
        let grid = ThermoGrid::new(nz);
        records.push(bench_autodiff(&grid, n_warmup, n_repeats));
        records.push(bench_interp2d(&grid, &table, n_warmup, n_repeats));
        records.push(bench_neu(&grid, n_warmup, n_repeats));
        records.push(bench_mtx_block_tridiag(nz, n_warmup, n_repeats));
        records.push(bench_turb_mlt(&grid, n_warmup, n_repeats));
        records.push(bench_turb_tdc(&grid, n_warmup, n_repeats));
    }
    records
}
