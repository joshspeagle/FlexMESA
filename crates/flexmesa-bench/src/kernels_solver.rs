//! Four-stage solver and convective transport benchmarks (`mtx` block
//! tridiagonal, `turb` MLT, and `turb` TDC).
//!
//! Evaluates block-tridiagonal linear solves (`nvar = 5`), cubic Mixing-Length
//! Theory (`set_mlt`), and iterative Kuhfuß 1D time-dependent convection
//! (`set_tdc`) across Stages 0..3.

use std::thread;

use flexmesa_autodiff::DualOrder1;
use flexmesa_const::{KERG, LSUN, MP, MSUN, RSUN, STANDARD_CGRAV};
use flexmesa_mtx::{
    block_thomas_factor, block_thomas_solve, solve_block_tridiagonal,
};
use flexmesa_turb::{
    set_mlt, set_tdc, MltInputs, MltParams, TdcInputs, TdcParams,
};

use crate::tridiag_fast::{
    fast_block_thomas_solve, make_block_tridiag_fixture,
};
use crate::{
    max_rel_diff, measure_us, BenchRecord, ThermoGrid, STAGE3_THREADS,
};

/// Benchmarks $5 \times 5$ block-tridiagonal factorization and solve (`mtx`)
/// across Stages 0..3 for `nz` stellar zones.
pub fn bench_mtx_block_tridiag(
    nz: usize,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    let nvar = 5usize;
    let (lblk, dblk, ublk, rhs) = make_block_tridiag_fixture(nz);

    // Stage 0: Per-call heap allocation of working blocks and pivot arrays.
    let mut ref_sol = vec![0.0_f64; nz * nvar];
    let stage0 = measure_us(n_warmup, n_repeats, || {
        let sol = solve_block_tridiagonal(nvar, nz, &lblk, &dblk, &ublk, &rhs)
            .unwrap();
        ref_sol.copy_from_slice(&sol);
    });

    // Stage 1: Pre-allocated contiguous workspace buffers.
    let mut d1 = dblk.clone();
    let mut u1 = ublk.clone();
    let mut ipiv1 = vec![0usize; nz * nvar];
    let mut out1 = rhs.clone();
    let stage1 = measure_us(n_warmup, n_repeats, || {
        d1.copy_from_slice(&dblk);
        u1.copy_from_slice(&ublk);
        out1.copy_from_slice(&rhs);
        block_thomas_factor(nvar, nz, &lblk, &mut d1, &mut u1, &mut ipiv1)
            .unwrap();
        block_thomas_solve(nvar, nz, &lblk, &d1, &u1, &ipiv1, &mut out1)
            .unwrap();
    });

    // Stage 2: Pre-validated fast block Thomas kernel without NaN re-scans.
    let mut d2 = dblk.clone();
    let mut u2 = ublk.clone();
    let mut ipiv2 = vec![0usize; nz * nvar];
    let mut out2 = rhs.clone();
    let stage2 = measure_us(n_warmup, n_repeats, || {
        d2.copy_from_slice(&dblk);
        u2.copy_from_slice(&ublk);
        out2.copy_from_slice(&rhs);
        fast_block_thomas_solve(
            nvar, nz, &lblk, &mut d2, &mut u2, &mut ipiv2, &mut out2,
        );
    });

    // Stage 3: 4-thread parallel ensemble solve (per-system amortized latency).
    let mut out3 = rhs.clone();
    let batch_raw = measure_us(n_warmup, n_repeats, || {
        let mut batch_sols = vec![rhs.clone(); STAGE3_THREADS];
        thread::scope(|s| {
            for sol in &mut batch_sols {
                let l_ref = &lblk;
                let d_ref = &dblk;
                let u_ref = &ublk;
                s.spawn(move || {
                    let mut dw = d_ref.clone();
                    let mut uw = u_ref.clone();
                    let mut ipiv = vec![0usize; nz * nvar];
                    fast_block_thomas_solve(
                        nvar, nz, l_ref, &mut dw, &mut uw, &mut ipiv, sol,
                    );
                });
            }
        });
        out3.copy_from_slice(&batch_sols[0]);
    });
    let stage3 = crate::LatencyStats {
        median_us: batch_raw.median_us / (STAGE3_THREADS as f64),
        iqr_us: batch_raw.iqr_us / (STAGE3_THREADS as f64),
    };

    let max_rel_err = max_rel_diff(&ref_sol, &out1)
        .max(max_rel_diff(&ref_sol, &out2))
        .max(max_rel_diff(&ref_sol, &out3));
    let checksum: f64 = out2.iter().sum();

    BenchRecord {
        workload: "mtx_tridiag",
        n_zones: nz,
        stage0,
        stage1,
        stage2,
        stage3,
        max_rel_err,
        checksum,
    }
}

/// Evaluates Cox Mixing-Length Theory (`set_mlt`) with 6-variable first-order
/// automatic differentiation at zone fraction `s_frac`.
#[inline]
pub fn eval_mlt_zone(s_frac: f64) -> [f64; 4] {
    let t_val = 1.0e5 * (1.0 + 0.5 * s_frac);
    let rho_val = 1.0e-5 * (1.0 + 0.5 * s_frac);
    let grav_val = MSUN * STANDARD_CGRAV / (RSUN * RSUN);
    let cp_val = 2.5 * KERG / MP;
    let p_val = rho_val * t_val * KERG / MP;
    let hp_val = p_val / (rho_val * grav_val);
    let c = DualOrder1::<6>::constant;
    let inp = MltInputs {
        chi_t: c(1.0),
        chi_rho: c(1.0),
        cp: c(cp_val),
        grav: c(grav_val),
        lambda: DualOrder1::<6>::independent(2.0 * hp_val, 3),
        rho: DualOrder1::<6>::independent(rho_val, 5),
        p: c(p_val),
        t: DualOrder1::<6>::independent(t_val, 4),
        opacity: c(1.0),
        gradr: DualOrder1::<6>::independent(0.45 + 0.25 * s_frac, 0),
        grada: DualOrder1::<6>::independent(0.40, 1),
        gradl: DualOrder1::<6>::independent(0.40, 2),
    };
    let params = MltParams::cox(2.0);
    let out = set_mlt(&params, &inp).unwrap();
    [
        out.grad_t.val,
        out.d.val,
        out.conv_vel.val,
        out.grad_t.d1[0],
    ]
}

/// Benchmarks Cox Mixing-Length Theory (`set_mlt`) with 6-variable `DualOrder1`
/// sensitivities across Stages 0..3.
pub fn bench_turb_mlt(
    grid: &ThermoGrid,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    let nz = grid.n_zones;
    let denom = (nz.saturating_sub(1)).max(1) as f64;
    let s_vec: Vec<f64> = (0..nz).map(|k| k as f64 / denom).collect();
    let s_slice = s_vec.as_slice();

    // Stage 0: Scalar loop allocating Vec<[f64; 4]>.
    let mut ref_flat = vec![0.0_f64; nz * 4];
    let stage0 = measure_us(n_warmup, n_repeats, || {
        let mut rows = Vec::with_capacity(nz);
        for &s_val in s_slice {
            rows.push(eval_mlt_zone(s_val));
        }
        for (k, r) in rows.iter().enumerate() {
            ref_flat[k * 4..(k + 1) * 4].copy_from_slice(r);
        }
    });

    // Stage 1: Contiguous SoA output buffer.
    let mut out1 = vec![0.0_f64; nz * 4];
    let stage1 = measure_us(n_warmup, n_repeats, || {
        for (k, &s_val) in s_slice.iter().enumerate() {
            out1[k * 4..(k + 1) * 4].copy_from_slice(&eval_mlt_zone(s_val));
        }
    });

    // Stage 2: Fused chunk slice kernel.
    let mut out2 = vec![0.0_f64; nz * 4];
    let stage2 = measure_us(n_warmup, n_repeats, || {
        for (&s_val, dst) in s_slice.iter().zip(out2.chunks_exact_mut(4)) {
            dst.copy_from_slice(&eval_mlt_zone(s_val));
        }
    });

    // Stage 3: 4-thread scoped parallel execution across zone chunks.
    let mut out3 = vec![0.0_f64; nz * 4];
    let chunk_z = nz.div_ceil(STAGE3_THREADS);
    let stage3 = measure_us(n_warmup, n_repeats, || {
        thread::scope(|s| {
            for (idx, dst_chunk) in out3.chunks_mut(chunk_z * 4).enumerate() {
                let z0 = idx * chunk_z;
                let z1 = (z0 + dst_chunk.len() / 4).min(nz);
                let s_sub = &s_slice[z0..z1];
                s.spawn(move || {
                    for (&s_val, dst) in
                        s_sub.iter().zip(dst_chunk.chunks_exact_mut(4))
                    {
                        dst.copy_from_slice(&eval_mlt_zone(s_val));
                    }
                });
            }
        });
    });

    let max_rel_err = max_rel_diff(&ref_flat, &out1)
        .max(max_rel_diff(&ref_flat, &out2))
        .max(max_rel_diff(&ref_flat, &out3));
    let checksum: f64 = out2.iter().step_by(4).sum();

    BenchRecord {
        workload: "turb_mlt",
        n_zones: nz,
        stage0,
        stage1,
        stage2,
        stage3,
        max_rel_err,
        checksum,
    }
}

/// Evaluates Kuhfuß 1D Time-Dependent Convection (`set_tdc`) with 6-variable
/// implicit-function-theorem sensitivities at zone fraction `s_frac`.
#[inline]
pub fn eval_tdc_zone(s_frac: f64) -> [f64; 4] {
    let t_val = 1.0e5 * (1.0 + 0.2 * s_frac);
    let rho_val = 1.0e-5 * (1.0 + 0.2 * s_frac);
    let grav_val = MSUN * STANDARD_CGRAV / (RSUN * RSUN);
    let cp_val = 2.5 * KERG / MP;
    let p_val = rho_val * t_val * KERG / MP;
    let hp_val = p_val / (rho_val * grav_val);
    let c = DualOrder1::<6>::constant;
    let inp = TdcInputs {
        chi_t: c(1.0),
        chi_rho: c(1.0),
        gradr: DualOrder1::<6>::independent(0.42 + 0.08 * s_frac, 0),
        r: c(RSUN),
        p: c(p_val),
        t: DualOrder1::<6>::independent(t_val, 4),
        rho: DualOrder1::<6>::independent(rho_val, 5),
        dv: c(0.0),
        cp: c(cp_val),
        opacity: c(1.0),
        scale_height: DualOrder1::<6>::independent(hp_val, 3),
        gradl: DualOrder1::<6>::independent(0.40, 2),
        grada: DualOrder1::<6>::independent(0.40, 1),
        eq_div_w: c(0.0),
        grav: c(grav_val),
        energy: c(0.0),
    };
    let params = TdcParams {
        conv_vel_start: 1.0e5,
        mixing_length_alpha: 2.0,
        tdc_alpha_d: 1.0,
        tdc_alpha_r: 0.0,
        tdc_alpha_pt: 0.0,
        tdc_alpha_c: 1.0,
        tdc_alpha_s: 1.0,
        dt: 1.0e4,
        cgrav: STANDARD_CGRAV,
        m: MSUN,
        scale: 70.0 * LSUN * 1.0e-3,
        max_conv_vel: 1.0e99,
        include_mlt_corr_to_tdc: true,
        use_tdc_enthalpy_flux_limiter: false,
        y_face_guess: 0.0,
    };
    let out = set_tdc(&params, &inp).unwrap().output;
    [
        out.grad_t.val,
        out.d.val,
        out.conv_vel.val,
        out.grad_t.d1[0],
    ]
}

/// Benchmarks Kuhfuß 1D Time-Dependent Convection (`set_tdc`) across Stages
/// 0..3.
pub fn bench_turb_tdc(
    grid: &ThermoGrid,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    let nz = grid.n_zones;
    let denom = (nz.saturating_sub(1)).max(1) as f64;
    let s_vec: Vec<f64> = (0..nz).map(|k| k as f64 / denom).collect();
    let s_slice = s_vec.as_slice();

    // Stage 0: Scalar loop allocating Vec<[f64; 4]>.
    let mut ref_flat = vec![0.0_f64; nz * 4];
    let stage0 = measure_us(n_warmup, n_repeats, || {
        let mut rows = Vec::with_capacity(nz);
        for &s_val in s_slice {
            rows.push(eval_tdc_zone(s_val));
        }
        for (k, r) in rows.iter().enumerate() {
            ref_flat[k * 4..(k + 1) * 4].copy_from_slice(r);
        }
    });

    // Stage 1: Contiguous SoA output buffer.
    let mut out1 = vec![0.0_f64; nz * 4];
    let stage1 = measure_us(n_warmup, n_repeats, || {
        for (k, &s_val) in s_slice.iter().enumerate() {
            out1[k * 4..(k + 1) * 4].copy_from_slice(&eval_tdc_zone(s_val));
        }
    });

    // Stage 2: Fused chunk slice kernel (with primal scalar bisection/Newton).
    let mut out2 = vec![0.0_f64; nz * 4];
    let stage2 = measure_us(n_warmup, n_repeats, || {
        for (&s_val, dst) in s_slice.iter().zip(out2.chunks_exact_mut(4)) {
            dst.copy_from_slice(&eval_tdc_zone(s_val));
        }
    });

    // Stage 3: 4-thread scoped parallel execution across zone chunks.
    let mut out3 = vec![0.0_f64; nz * 4];
    let chunk_z = nz.div_ceil(STAGE3_THREADS);
    let stage3 = measure_us(n_warmup, n_repeats, || {
        thread::scope(|s| {
            for (idx, dst_chunk) in out3.chunks_mut(chunk_z * 4).enumerate() {
                let z0 = idx * chunk_z;
                let z1 = (z0 + dst_chunk.len() / 4).min(nz);
                let s_sub = &s_slice[z0..z1];
                s.spawn(move || {
                    for (&s_val, dst) in
                        s_sub.iter().zip(dst_chunk.chunks_exact_mut(4))
                    {
                        dst.copy_from_slice(&eval_tdc_zone(s_val));
                    }
                });
            }
        });
    });

    let max_rel_err = max_rel_diff(&ref_flat, &out1)
        .max(max_rel_diff(&ref_flat, &out2))
        .max(max_rel_diff(&ref_flat, &out3));
    let checksum: f64 = out2.iter().step_by(4).sum();

    BenchRecord {
        workload: "turb_tdc",
        n_zones: nz,
        stage0,
        stage1,
        stage2,
        stage3,
        max_rel_err,
        checksum,
    }
}
