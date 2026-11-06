//! Four-stage microphysics and utility kernel benchmarks (`auto_diff`,
//! `interp_2d`, and `neu`).
//!
//! Each benchmark evaluates the same mathematical function across
//! $N_{\mathrm{zones}}$ stellar zones through four progressive stages:
//! 1. **Stage 0 (Idiomatic Scalar Baseline)**: Zone-by-zone scalar calls with
//!    per-call output allocation.
//! 2. **Stage 1 (Contiguous Structure-of-Arrays Layout)**: Pre-allocated
//!    contiguous `f64` output buffer (`SoA`).
//! 3. **Stage 2 (Fused Kernel & Hoisted Invariants)**: Inlined slice loops with
//!    hoisted grid/table validation or precomputed logs.
//! 4. **Stage 3 (Multi-Core Data Parallelism)**: 4-thread scoped parallel chunk
//!    execution.

use std::thread;

use flexmesa_autodiff::AutoDiffReal2VarOrder2;
use flexmesa_interp2d::interp_evbicub_db;
use flexmesa_neu::{neu_get, NeuFlags};

use crate::bicub_fast::eval_bicub_prevalidated;
use crate::{
    max_rel_diff, measure_us, BenchRecord, BicubTable, ThermoGrid,
    STAGE3_THREADS,
};

/// Evaluates the benchmark two-variable second-order thermodynamic test
/// function $f(x, y) = \exp(0.1\,x\,y) + x^{1.5} / (y + 2)$ at a single zone
/// using [`AutoDiffReal2VarOrder2`].
#[inline]
pub fn eval_autodiff_zone(x_val: f64, y_val: f64) -> [f64; 6] {
    let x = AutoDiffReal2VarOrder2::var1(x_val);
    let y = AutoDiffReal2VarOrder2::var2(y_val);
    let z = (x * y * 0.1).exp() + x.powf(1.5) / (y + 2.0);
    [
        z.val,
        z.d1val1,
        z.d1val2,
        z.d2val1,
        z.d1val1_d1val2,
        z.d2val2,
    ]
}

/// Evaluates `eval_autodiff_zone` across a contiguous slice of `(x, y)` pairs
/// into `out` (`6 * x.len()`).
pub fn eval_autodiff_slice(x: &[f64], y: &[f64], out: &mut [f64]) {
    for ((&xk, &yk), dst) in x.iter().zip(y.iter()).zip(out.chunks_exact_mut(6))
    {
        dst.copy_from_slice(&eval_autodiff_zone(xk, yk));
    }
}

/// Benchmarks two-variable second-order automatic differentiation
/// (`auto_diff`) across Stages 0..3 for `grid.n_zones`.
pub fn bench_autodiff(
    grid: &ThermoGrid,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    let nz = grid.n_zones;
    let x = &grid.log10_t;
    let y = &grid.log10_rho;

    // Stage 0: Scalar loop allocating per-call Vec<[f64; 6]>.
    let mut ref_flat = vec![0.0_f64; nz * 6];
    let stage0 = measure_us(n_warmup, n_repeats, || {
        let mut rows = Vec::with_capacity(nz);
        for k in 0..nz {
            rows.push(eval_autodiff_zone(x[k], y[k]));
        }
        for (k, r) in rows.iter().enumerate() {
            ref_flat[k * 6..(k + 1) * 6].copy_from_slice(r);
        }
    });

    // Stage 1: Contiguous SoA output slice via operator-overloaded loop.
    let mut out1 = vec![0.0_f64; nz * 6];
    let stage1 = measure_us(n_warmup, n_repeats, || {
        for k in 0..nz {
            let r = eval_autodiff_zone(x[k], y[k]);
            out1[k * 6..(k + 1) * 6].copy_from_slice(&r);
        }
    });

    // Stage 2: Fused chunk slice kernel without bounds re-indexing.
    let mut out2 = vec![0.0_f64; nz * 6];
    let stage2 = measure_us(n_warmup, n_repeats, || {
        eval_autodiff_slice(x, y, &mut out2);
    });

    // Stage 3: 4-thread scoped parallel execution across zone chunks.
    let mut out3 = vec![0.0_f64; nz * 6];
    let chunk_z = nz.div_ceil(STAGE3_THREADS);
    let stage3 = measure_us(n_warmup, n_repeats, || {
        thread::scope(|s| {
            for (idx, dst_chunk) in out3.chunks_mut(chunk_z * 6).enumerate() {
                let z0 = idx * chunk_z;
                let z1 = (z0 + dst_chunk.len() / 6).min(nz);
                let x_sub = &x[z0..z1];
                let y_sub = &y[z0..z1];
                s.spawn(move || {
                    eval_autodiff_slice(x_sub, y_sub, dst_chunk);
                });
            }
        });
    });

    let max_rel_err = max_rel_diff(&ref_flat, &out1)
        .max(max_rel_diff(&ref_flat, &out2))
        .max(max_rel_diff(&ref_flat, &out3));
    let checksum: f64 = out2.iter().step_by(6).sum();

    BenchRecord {
        workload: "auto_diff",
        n_zones: nz,
        stage0,
        stage1,
        stage2,
        stage3,
        max_rel_err,
        checksum,
    }
}

/// Benchmarks 2D bicubic spline evaluation (`interp_2d`) across Stages 0..3.
pub fn bench_interp2d(
    grid: &ThermoGrid,
    table: &BicubTable,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    let nz = grid.n_zones;
    let xq = &grid.log10_rho;
    let yq = &grid.log10_t;

    // Stage 0: Scalar loop allocating Vec<[f64; 6]> via interp_evbicub_db.
    let mut ref_flat = vec![0.0_f64; nz * 6];
    let stage0 = measure_us(n_warmup, n_repeats, || {
        let mut rows = Vec::with_capacity(nz);
        for k in 0..nz {
            let ev = interp_evbicub_db(
                xq[k],
                yq[k],
                &table.x,
                &table.y,
                table.ilinx,
                table.iliny,
                &table.coeffs,
            )
            .unwrap();
            rows.push([
                ev.f,
                ev.df_dx,
                ev.df_dy,
                ev.d2f_dx2,
                ev.d2f_dy2,
                ev.d2f_dxdy,
            ]);
        }
        for (k, r) in rows.iter().enumerate() {
            ref_flat[k * 6..(k + 1) * 6].copy_from_slice(r);
        }
    });

    // Stage 1: Contiguous SoA output slice calling interp_evbicub_db.
    let mut out1 = vec![0.0_f64; nz * 6];
    let stage1 = measure_us(n_warmup, n_repeats, || {
        for k in 0..nz {
            let ev = interp_evbicub_db(
                xq[k],
                yq[k],
                &table.x,
                &table.y,
                table.ilinx,
                table.iliny,
                &table.coeffs,
            )
            .unwrap();
            out1[k * 6..(k + 1) * 6].copy_from_slice(&[
                ev.f,
                ev.df_dx,
                ev.df_dy,
                ev.d2f_dx2,
                ev.d2f_dy2,
                ev.d2f_dxdy,
            ]);
        }
    });

    // Stage 2: Pre-validated table kernel hoisting O(nx * ny) grid checks.
    let mut out2 = vec![0.0_f64; nz * 6];
    let stage2 = measure_us(n_warmup, n_repeats, || {
        for ((&xk, &yk), dst) in
            xq.iter().zip(yq.iter()).zip(out2.chunks_exact_mut(6))
        {
            dst.copy_from_slice(&eval_bicub_prevalidated(
                xk,
                yk,
                &table.x,
                &table.y,
                &table.coeffs,
            ));
        }
    });

    // Stage 3: 4-thread scoped parallel execution over pre-validated table.
    let mut out3 = vec![0.0_f64; nz * 6];
    let chunk_z = nz.div_ceil(STAGE3_THREADS);
    let stage3 = measure_us(n_warmup, n_repeats, || {
        thread::scope(|s| {
            for (idx, dst_chunk) in out3.chunks_mut(chunk_z * 6).enumerate() {
                let z0 = idx * chunk_z;
                let z1 = (z0 + dst_chunk.len() / 6).min(nz);
                let x_sub = &xq[z0..z1];
                let y_sub = &yq[z0..z1];
                s.spawn(move || {
                    for ((&xk, &yk), dst) in x_sub
                        .iter()
                        .zip(y_sub.iter())
                        .zip(dst_chunk.chunks_exact_mut(6))
                    {
                        dst.copy_from_slice(&eval_bicub_prevalidated(
                            xk,
                            yk,
                            &table.x,
                            &table.y,
                            &table.coeffs,
                        ));
                    }
                });
            }
        });
    });

    let max_rel_err = max_rel_diff(&ref_flat, &out1)
        .max(max_rel_diff(&ref_flat, &out2))
        .max(max_rel_diff(&ref_flat, &out3));
    let checksum: f64 = out2.iter().step_by(6).sum();

    BenchRecord {
        workload: "interp_2d",
        n_zones: nz,
        stage0,
        stage1,
        stage2,
        stage3,
        max_rel_err,
        checksum,
    }
}

/// Benchmarks Itoh et al. (1996) pair + plasmon thermal neutrino loss (`neu`)
/// across Stages 0..3.
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

    // Stage 3: 4-thread scoped parallel execution across zone chunks.
    let mut out3 = vec![0.0_f64; nz * 5];
    let chunk_z = nz.div_ceil(STAGE3_THREADS);
    let stage3 = measure_us(n_warmup, n_repeats, || {
        thread::scope(|s| {
            for (idx, dst_chunk) in out3.chunks_mut(chunk_z * 5).enumerate() {
                let z0 = idx * chunk_z;
                s.spawn(move || {
                    for (local_k, dst) in
                        dst_chunk.chunks_exact_mut(5).enumerate()
                    {
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
                });
            }
        });
    });

    let max_rel_err = max_rel_diff(&ref_flat, &out1)
        .max(max_rel_diff(&ref_flat, &out2))
        .max(max_rel_diff(&ref_flat, &out3));
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
