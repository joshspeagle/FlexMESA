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
//! - **Stage 3 (Multi-Core Data Parallelism)**: Scoped thread parallelism
//!   chunked across 4 worker threads.

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
pub mod tridiag_fast;

use std::time::Instant;

use flexmesa_interp2d::{interp_mkbicub_db, BicubicBc};
use flexmesa_math::{cospi, exp10, sinpi};

pub use kernels_micro::{bench_autodiff, bench_interp2d, bench_neu};
pub use kernels_solver::{
    bench_mtx_block_tridiag, bench_turb_mlt, bench_turb_tdc,
};

/// Number of worker threads used in Stage 3 multi-core benchmarks.
pub const STAGE3_THREADS: usize = 4;

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
    samples.sort_by(f64::total_cmp);
    let mid = repeats / 2;
    let median_us = if repeats % 2 == 1 {
        samples[mid]
    } else {
        f64::midpoint(samples[mid - 1], samples[mid])
    };
    let q25 = samples[repeats / 4];
    let q75 = samples[(3 * repeats) / 4];
    LatencyStats {
        median_us,
        iqr_us: q75 - q25,
    }
}

/// Computes the maximum relative difference between two `f64` slices `a` and
/// `b`.
pub fn max_rel_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "slice length mismatch in max_rel_diff");
    let mut max_err = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let denom = x.abs().max(y.abs()).max(1.0e-30);
        let rel = (x - y).abs() / denom;
        if rel > max_err {
            max_err = rel;
        }
    }
    max_err
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
    pub fn new(n_zones: usize) -> Self {
        let mut log10_t = Vec::with_capacity(n_zones);
        let mut t = Vec::with_capacity(n_zones);
        let mut log10_rho = Vec::with_capacity(n_zones);
        let mut rho = Vec::with_capacity(n_zones);
        let mut abar = Vec::with_capacity(n_zones);
        let mut zbar = Vec::with_capacity(n_zones);
        let denom = (n_zones.saturating_sub(1)).max(1) as f64;
        for k in 0..n_zones {
            let s = k as f64 / denom;
            let lt = 7.2 + 2.6 * s;
            let lrho = 4.0 + 5.0 * s;
            log10_t.push(lt);
            t.push(exp10(lt));
            log10_rho.push(lrho);
            rho.push(exp10(lrho));
            let a = 4.0 + 24.0 * s;
            abar.push(a);
            zbar.push(0.5 * a);
        }
        Self {
            n_zones,
            log10_t,
            t,
            log10_rho,
            rho,
            abar,
            zbar,
        }
    }
}

/// Precomputed $64 \times 64$ 2D bicubic spline table over
/// $(\log_{10}\rho, \log_{10}T)$ representative of stellar EOS and opacity
/// look-ups.
#[derive(Debug, Clone)]
pub struct BicubTable {
    /// Strictly ascending `x` grid ($\log_{10}\rho \in [3.5, 9.5]$) of length
    /// `64`.
    pub x: Vec<f64>,
    /// Strictly ascending `y` grid ($\log_{10}T \in [7.0, 10.0]$) of length
    /// `64`.
    pub y: Vec<f64>,
    /// Precomputed bicubic spline coefficients `[f, f_xx, f_yy, f_xxyy]` of
    /// length `64 * 64`.
    pub coeffs: Vec<[f64; 4]>,
    /// Uniform-spacing flag along `x`.
    pub ilinx: bool,
    /// Uniform-spacing flag along `y`.
    pub iliny: bool,
}

impl BicubTable {
    /// Builds the deterministic $64 \times 64$ bicubic spline table.
    pub fn new() -> Self {
        let nx = 64usize;
        let ny = 64usize;
        let x: Vec<f64> = (0..nx)
            .map(|i| 3.5 + 6.0 * (i as f64) / ((nx - 1) as f64))
            .collect();
        let y: Vec<f64> = (0..ny)
            .map(|j| 7.0 + 3.0 * (j as f64) / ((ny - 1) as f64))
            .collect();
        let mut z = vec![0.0_f64; nx * ny];
        for i in 0..nx {
            for j in 0..ny {
                let xi = x[i];
                let yj = y[j];
                z[i * ny + j] = 0.25 * xi * xi
                    + 0.15 * xi * yj
                    + 0.35 * yj * yj
                    + 0.1 * sinpi(0.2 * xi) * cospi(0.15 * yj);
            }
        }
        let (coeffs, ilinx, iliny) = interp_mkbicub_db(
            &x,
            &y,
            &z,
            BicubicBc::NotAKnot,
            &[],
            BicubicBc::NotAKnot,
            &[],
            BicubicBc::NotAKnot,
            &[],
            BicubicBc::NotAKnot,
            &[],
        )
        .expect("valid bicubic table");
        Self {
            x,
            y,
            coeffs,
            ilinx,
            iliny,
        }
    }
}

impl Default for BicubTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Runs the complete Wave 1 four-stage benchmark suite across `zone_counts` and
/// returns all [`BenchRecord`] entries.
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
