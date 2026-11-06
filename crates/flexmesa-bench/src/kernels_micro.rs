//! Four-stage microphysics and utility kernel benchmarks (`auto_diff`,
//! `interp_2d`, and `neu`).
//!
//! Implements Stages 0..3 for:
//! 1. `bench_autodiff`: Two-variable second-order forward-mode hyperdual
//!    evaluation (`AutoDiffReal2VarOrder2`) of a non-polynomial thermodynamic
//!    expression $f(x, y) = \exp(0.1 x y) + x^{1.5} / (y + 2)$.
//! 2. `bench_interp2d`: 2D bicubic spline evaluation (`interp_evbicub_db`)
//!    over a $64 \times 64$ $(\log_{10}\rho, \log_{10}T)$ table.
//! 3. `bench_neu`: Itoh et al. (1996) pair-production and plasmon-decay
//!    thermal neutrino energy-loss rate and 4-variable Jacobian (`neu_get`).

use flexmesa_autodiff::AutoDiffReal2VarOrder2;
use flexmesa_interp2d::interp_evbicub_db;
use flexmesa_neu::{neu_get, NeuFlags};

use crate::bicub_fast::eval_bicub_prevalidated;
use crate::{
    max_err3, max_rel_diff, measure_stage3_chunks_us, measure_us, BenchRecord,
    BicubTable, ThermoGrid, STAGE3_THREADS,
};

/// Evaluates the non-polynomial bivariate thermodynamic expression
/// $f(x, y) = \exp(0.1 x y) + x^{1.5} / (y + 2)$ using
/// `AutoDiffReal2VarOrder2` and returns the 6-element value/gradient/Hessian
/// array `[val, d1val1, d1val2, d2val1, d1val1_d1val2, d2val2]`.
#[inline]
fn eval_autodiff_kernel(x: f64, y: f64) -> [f64; 6] {
    let xv = AutoDiffReal2VarOrder2::var1(x);
    let yv = AutoDiffReal2VarOrder2::var2(y);
    let term1 = (xv * yv * 0.1).exp();
    let term2 = xv.powf(1.5) / (yv + 2.0);
    let z = term1 + term2;
    [
        z.val,
        z.d1val1,
        z.d1val2,
        z.d2val1,
        z.d1val1_d1val2,
        z.d2val2,
    ]
}

/// Evaluates the expanded closed-form algebraic value, first partials, and
/// second partials of $f(x, y) = \exp(0.1 x y) + x^{1.5} / (y + 2)$ directly
/// in `f64` registers for Stage 2 and Stage 3.
#[inline]
fn eval_autodiff_fused(x: f64, y: f64) -> [f64; 6] {
    let u = 0.1 * x * y;
    let e = u.exp();
    let yp2 = y + 2.0;
    let inv_yp2 = 1.0 / yp2;
    let inv_yp2_2 = inv_yp2 * inv_yp2;
    let inv_yp2_3 = inv_yp2_2 * inv_yp2;
    let x_half = x.sqrt();
    let x_1p5 = x * x_half;
    let x_m0p5 = 1.0 / x_half;

    let val = e + x_1p5 * inv_yp2;
    let d1val1 = 0.1 * y * e + 1.5 * x_half * inv_yp2;
    let d1val2 = 0.1 * x * e - x_1p5 * inv_yp2_2;
    let d2val1 = 0.01 * y * y * e + 0.75 * x_m0p5 * inv_yp2;
    let d1val1_d1val2 = (0.1 + 0.01 * x * y) * e - 1.5 * x_half * inv_yp2_2;
    let d2val2 = 0.01 * x * x * e + 2.0 * x_1p5 * inv_yp2_3;
    [val, d1val1, d1val2, d2val1, d1val1_d1val2, d2val2]
}

/// Shared 4-stage benchmark driver for a 6-output per-zone kernel pair
/// `(eval_base, eval_fast)` over `nz` shells.
///
/// Evaluates `eval_base` in Stages 0 and 1 and `eval_fast` in Stages 2 and 3:
/// - **Stage 0**: Scalar loop allocating a fresh `Vec<[f64; 6]>` per step.
/// - **Stage 1**: Pre-allocated contiguous Structure-of-Arrays (SoA) slice.
/// - **Stage 2**: Fused single-thread slice loop (`chunks_exact_mut(6)`).
/// - **Stage 3**: Scoped 4-thread parallel execution across radial chunks.
///
/// # Arguments
/// * `workload` - Static benchmark name (`"auto_diff"` or `"interp_2d"`).
/// * `nz` - Number of radial zones ($N_{\mathrm{zones}}$).
/// * `n_warmup` - Number of untimed warmup iterations.
/// * `n_repeats` - Number of timed iterations.
/// * `eval_base` - Baseline per-zone evaluator for Stages 0 and 1.
/// * `eval_fast` - Thread-safe fused per-zone evaluator for Stages 2 and 3.
///
/// # Returns
/// `BenchRecord` with Stages 0..3 latencies, maximum relative stage
/// discrepancy, and Stage 2 primal checksum.
fn bench_zone_fn_6<F0, F2>(
    workload: &'static str,
    nz: usize,
    n_warmup: usize,
    n_repeats: usize,
    eval_base: F0,
    eval_fast: F2,
) -> BenchRecord
where
    F0: Fn(usize) -> [f64; 6],
    F2: Fn(usize) -> [f64; 6] + Sync,
{
    // Stage 0: Scalar loop allocating Vec<[f64; 6]> per call.
    let mut ref_flat = vec![0.0_f64; nz * 6];
    let stage0 = measure_us(n_warmup, n_repeats, || {
        let mut rows = Vec::with_capacity(nz);
        for k in 0..nz {
            rows.push(eval_base(k));
        }
        for (k, r) in rows.iter().enumerate() {
            ref_flat[k * 6..(k + 1) * 6].copy_from_slice(r);
        }
    });

    // Stage 1: Contiguous pre-allocated SoA slice buffer.
    let mut out1 = vec![0.0_f64; nz * 6];
    let stage1 = measure_us(n_warmup, n_repeats, || {
        for k in 0..nz {
            out1[k * 6..(k + 1) * 6].copy_from_slice(&eval_base(k));
        }
    });

    // Stage 2: Fused inlined / pre-validated slice kernel.
    let mut out2 = vec![0.0_f64; nz * 6];
    let stage2 = measure_us(n_warmup, n_repeats, || {
        for (k, dst) in out2.chunks_exact_mut(6).enumerate() {
            dst.copy_from_slice(&eval_fast(k));
        }
    });

    // Stage 3: 4-thread persistent rayon parallel execution across zone chunks.
    let mut out3 = vec![0.0_f64; nz * 6];
    let chunk_z = nz.div_ceil(STAGE3_THREADS);
    let fast_ref = &eval_fast;
    let stage3 = measure_stage3_chunks_us(
        n_warmup,
        n_repeats,
        &mut out3,
        chunk_z * 6,
        |idx, dst_chunk| {
            let z0 = idx * chunk_z;
            for (local_k, dst) in dst_chunk.chunks_exact_mut(6).enumerate() {
                dst.copy_from_slice(&fast_ref(z0 + local_k));
            }
        },
    );

    let max_rel_err = max_err3(
        max_rel_diff(&ref_flat, &out1),
        max_rel_diff(&ref_flat, &out2),
        max_rel_diff(&ref_flat, &out3),
    );
    let checksum: f64 = out2.iter().step_by(6).sum();

    BenchRecord {
        workload,
        n_zones: nz,
        stage0,
        stage1,
        stage2,
        stage3,
        max_rel_err,
        checksum,
    }
}

/// Benchmarks two-variable 2nd-order automatic differentiation (`auto_diff`)
/// across Stages 0..3 over `grid.n_zones` stellar shells.
///
/// # Arguments
/// * `grid` - Stellar thermodynamic profile across `n_zones` shells.
/// * `n_warmup` - Number of untimed warmup iterations.
/// * `n_repeats` - Number of timed iterations.
///
/// # Returns
/// [`BenchRecord`] with Stages 0..3 latencies, parity error, and checksum.
pub fn bench_autodiff(
    grid: &ThermoGrid,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    bench_zone_fn_6(
        "auto_diff",
        grid.n_zones,
        n_warmup,
        n_repeats,
        |k| eval_autodiff_kernel(grid.log10_t[k], grid.log10_rho[k]),
        |k| eval_autodiff_fused(grid.log10_t[k], grid.log10_rho[k]),
    )
}

/// Benchmarks 2D bicubic spline evaluation (`interp_2d`) across Stages 0..3.
///
/// # Arguments
/// * `grid` - Stellar thermodynamic profile across `n_zones` shells.
/// * `table` - Precomputed $64 \times 64$ bicubic spline table.
/// * `n_warmup` - Number of untimed warmup iterations.
/// * `n_repeats` - Number of timed iterations.
///
/// # Returns
/// [`BenchRecord`] with Stages 0..3 latencies, parity error, and checksum.
pub fn bench_interp2d(
    grid: &ThermoGrid,
    table: &BicubTable,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    bench_zone_fn_6(
        "interp_2d",
        grid.n_zones,
        n_warmup,
        n_repeats,
        |k| {
            let ev = interp_evbicub_db(
                grid.log10_rho[k],
                grid.log10_t[k],
                &table.x,
                &table.y,
                table.ilinx,
                table.iliny,
                &table.coeffs,
            )
            .unwrap();
            [
                ev.f,
                ev.df_dx,
                ev.df_dy,
                ev.d2f_dx2,
                ev.d2f_dy2,
                ev.d2f_dxdy,
            ]
        },
        |k| {
            eval_bicub_prevalidated(
                grid.log10_rho[k],
                grid.log10_t[k],
                &table.x,
                &table.y,
                &table.coeffs,
            )
        },
    )
}

/// Benchmarks Itoh et al. (1996) pair + plasmon thermal neutrino loss (`neu`)
/// across Stages 0..3.
///
/// # Arguments
/// * `grid` - Stellar thermodynamic profile across `n_zones` shells.
/// * `n_warmup` - Number of untimed warmup iterations.
/// * `n_repeats` - Number of timed iterations.
///
/// # Returns
/// [`BenchRecord`] with Stages 0..3 latencies, parity error, and checksum.
pub fn bench_neu(
    grid: &ThermoGrid,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    let nz = grid.n_zones;
    let flags = NeuFlags {
        pair: true,
        plas: true,
        phot: false,
        brem: false,
        reco: false,
    };

    // Stage 0: Scalar loop recomputing log10(T)/log10(Rho) and allocating Vec.
    let mut ref_flat = vec![0.0_f64; nz * 5];
    let stage0 = measure_us(n_warmup, n_repeats, || {
        let mut rows = Vec::with_capacity(nz);
        for k in 0..nz {
            let out = neu_get(
                Some(grid.t[k]),
                None,
                Some(grid.rho[k]),
                None,
                grid.abar[k],
                grid.zbar[k],
                7.5,
                flags,
            )
            .unwrap();
            rows.push(out.loss);
        }
        for (k, r) in rows.iter().enumerate() {
            ref_flat[k * 5..(k + 1) * 5].copy_from_slice(r);
        }
    });

    // Stage 1: Contiguous SoA output buffer without precomputed log10 inputs.
    let mut out1 = vec![0.0_f64; nz * 5];
    let stage1 = measure_us(n_warmup, n_repeats, || {
        for k in 0..nz {
            let out = neu_get(
                Some(grid.t[k]),
                None,
                Some(grid.rho[k]),
                None,
                grid.abar[k],
                grid.zbar[k],
                7.5,
                flags,
            )
            .unwrap();
            out1[k * 5..(k + 1) * 5].copy_from_slice(&out.loss);
        }
    });

    // Stage 2: Fused slice kernel passing precomputed (T, log10_T, Rho,
    // log10_Rho).
    let mut out2 = vec![0.0_f64; nz * 5];
    let stage2 = measure_us(n_warmup, n_repeats, || {
        for (k, dst) in out2.chunks_exact_mut(5).enumerate() {
            let out = neu_get(
                Some(grid.t[k]),
                Some(grid.log10_t[k]),
                Some(grid.rho[k]),
                Some(grid.log10_rho[k]),
                grid.abar[k],
                grid.zbar[k],
                7.5,
                flags,
            )
            .unwrap();
            dst.copy_from_slice(&out.loss);
        }
    });

    // Stage 3: 4-thread persistent rayon parallel execution across zone chunks.
    let mut out3 = vec![0.0_f64; nz * 5];
    let chunk_z = nz.div_ceil(STAGE3_THREADS);
    let stage3 = measure_stage3_chunks_us(
        n_warmup,
        n_repeats,
        &mut out3,
        chunk_z * 5,
        |idx, dst_chunk| {
            let z0 = idx * chunk_z;
            for (local_k, dst) in dst_chunk.chunks_exact_mut(5).enumerate() {
                let k = z0 + local_k;
                let out = neu_get(
                    Some(grid.t[k]),
                    Some(grid.log10_t[k]),
                    Some(grid.rho[k]),
                    Some(grid.log10_rho[k]),
                    grid.abar[k],
                    grid.zbar[k],
                    7.5,
                    flags,
                )
                .unwrap();
                dst.copy_from_slice(&out.loss);
            }
        },
    );

    let max_rel_err = max_err3(
        max_rel_diff(&ref_flat, &out1),
        max_rel_diff(&ref_flat, &out2),
        max_rel_diff(&ref_flat, &out3),
    );
    let checksum: f64 = out2.iter().step_by(5).sum();

    BenchRecord {
        workload: "neu",
        n_zones: nz,
        stage0,
        stage1,
        stage2,
        stage3,
        max_rel_err,
        checksum,
    }
}
