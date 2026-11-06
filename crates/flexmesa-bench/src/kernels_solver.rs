//! Four-stage solver and convective transport benchmarks (`mtx` and `turb`).
//!
//! Implements Stages 0..3 for:
//! 1. `bench_mtx_block_tridiag`: $5 \times 5$ block-tridiagonal Henyey system
//!    factorization and solve (`solve_block_tridiagonal`).
//! 2. `bench_turb_mlt`: Cox (1968) Mixing-Length Theory (`set_mlt`) with
//!    6-variable `DualOrder1<6>` sensitivities.
//! 3. `bench_turb_tdc`: Kuhfuss (1986) 1D Time-Dependent Convection (`set_tdc`)
//!    with Implicit Function Theorem `DualOrder1<6>` sensitivities.

use flexmesa_autodiff::DualOrder1;
use flexmesa_const::{KERG, LSUN, MP, MSUN, RSUN, STANDARD_CGRAV};
use flexmesa_mtx::{
    block_thomas_factor, block_thomas_solve, solve_block_tridiagonal,
};
use flexmesa_turb::{
    set_mlt, set_tdc, MltInputs, MltParams, TdcInputs, TdcParams,
};

use crate::tridiag_fast::{fast_block_thomas_solve, make_block_tridiag_fixture};
use crate::{
    max_err3, max_rel_diff, measure_stage3_chunks_us, measure_us, BenchRecord,
    LatencyStats, ThermoGrid, STAGE3_THREADS,
};

/// Solar surface gravitational acceleration $G M_{\odot} / R_{\odot}^2$ in
/// $\mathrm{cm\,s^{-2}}$.
const GRAV_SOLAR: f64 = MSUN * STANDARD_CGRAV / (RSUN * RSUN);

/// Ideal monatomic specific Gas constant $k_{\mathrm{B}} / m_p$ in
/// $\mathrm{erg\,g^{-1}\,K^{-1}}$.
const RGAS_IDEAL: f64 = KERG / MP;

/// Ideal monatomic specific heat at constant pressure $C_P = \frac{5}{2}
/// k_{\mathrm{B}} / m_p$ in $\mathrm{erg\,g^{-1}\,K^{-1}}$.
const CP_IDEAL: f64 = 2.5 * RGAS_IDEAL;

/// Per-thread workspace `(dblk, ublk, ipiv, sol)` for Stage 3 block-tridiagonal
/// benchmarks.
type TridiagWorkspace = (Vec<f64>, Vec<f64>, Vec<usize>, Vec<f64>);

/// Benchmarks $5 \times 5$ block-tridiagonal Henyey solves (`mtx`) across
/// Stages 0..3 for `nz` radial shells.
///
/// # Arguments
/// * `nz` - Number of radial zones ($N_{\mathrm{zones}}$).
/// * `n_warmup` - Number of untimed warmup iterations.
/// * `n_repeats` - Number of timed iterations.
///
/// # Returns
/// [`BenchRecord`] with Stages 0..3 latencies, parity error, and checksum.
pub fn bench_mtx_block_tridiag(
    nz: usize,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    let nvar = 5usize;
    let (lblk, dblk, ublk, rhs) = make_block_tridiag_fixture(nz);

    // Stage 0: High-level `solve_block_tridiagonal` allocating workspace per
    // call.
    let mut ref_sol = vec![0.0_f64; nz * nvar];
    let stage0 = measure_us(n_warmup, n_repeats, || {
        let sol =
            solve_block_tridiagonal(nvar, nz, &lblk, &dblk, &ublk, &rhs)
                .unwrap();
        ref_sol.copy_from_slice(&sol);
    });

    // Stage 1: Pre-allocated workspace buffers reused across calls.
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

    // Stage 2: Zero-allocation `fast_block_thomas_solve` with stack/reused
    // buffers.
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

    // Stage 3: Multi-system throughput across 4 concurrent block-tridiagonal
    // solves with pre-allocated per-thread workspaces (amortized per system).
    let mut workspaces: Vec<TridiagWorkspace> = (0..STAGE3_THREADS)
        .map(|_| {
            (
                dblk.clone(),
                ublk.clone(),
                vec![0usize; nz * nvar],
                rhs.clone(),
            )
        })
        .collect();
    let l_ref = &lblk;
    let d_ref = &dblk;
    let u_ref = &ublk;
    let r_ref = &rhs;
    let stage3_batch = measure_stage3_chunks_us(
        n_warmup,
        n_repeats,
        &mut workspaces,
        1,
        |_tid, ws_slice| {
            let (dw, uw, ipiv, sol) = &mut ws_slice[0];
            dw.copy_from_slice(d_ref);
            uw.copy_from_slice(u_ref);
            sol.copy_from_slice(r_ref);
            fast_block_thomas_solve(nvar, nz, l_ref, dw, uw, ipiv, sol);
        },
    );
    let stage3 = LatencyStats {
        median_us: stage3_batch.median_us / (STAGE3_THREADS as f64),
        iqr_us: stage3_batch.iqr_us / (STAGE3_THREADS as f64),
    };
    let mut d3 = 0.0_f64;
    for (_, _, _, sol) in &workspaces {
        let err = max_rel_diff(&ref_sol, sol);
        d3 = if err.is_nan() || d3.is_nan() {
            f64::NAN
        } else {
            d3.max(err)
        };
    }

    let max_rel_err = max_err3(
        max_rel_diff(&ref_sol, &out1),
        max_rel_diff(&ref_sol, &out2),
        d3,
    );
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

/// Creates a 6-variable constant `DualOrder1<6>` with zero derivatives.
#[inline]
fn c6(val: f64) -> DualOrder1<6> {
    DualOrder1::constant(val)
}

/// Creates the `idx`-th independent `DualOrder1<6>` variable ($dx_i/dx_i = 1$).
#[inline]
fn v6(val: f64, idx: usize) -> DualOrder1<6> {
    DualOrder1::independent(val, idx)
}

/// Evaluates Cox (1968) Mixing-Length Theory (`set_mlt`) with 6-variable
/// forward-mode `DualOrder1<6>` sensitivities at normalized shell coordinate
/// `s_frac` $\in [0, 1]$, returning `[grad_t, d, conv_vel, d_grad_t_d_gradr]`.
#[inline]
fn eval_mlt_zone(s_frac: f64) -> [f64; 4] {
    let t_val = 1.0e5 * (1.0 + 0.5 * s_frac);
    let rho_val = 1.0e-5 * (1.0 + 0.5 * s_frac);
    let p_val = rho_val * t_val * RGAS_IDEAL;
    let hp_val = p_val / (rho_val * GRAV_SOLAR);
    let gradr_val = 0.45 + 0.25 * s_frac;

    let params = MltParams::cox(2.0);
    let inp = MltInputs {
        chi_t: c6(1.0),
        chi_rho: c6(1.0),
        cp: c6(CP_IDEAL),
        grav: c6(GRAV_SOLAR),
        lambda: v6(2.0 * hp_val, 3),
        rho: v6(rho_val, 5),
        p: c6(p_val),
        t: v6(t_val, 4),
        opacity: c6(1.0),
        gradr: v6(gradr_val, 0),
        grada: v6(0.40, 1),
        gradl: v6(0.40, 2),
    };
    let out = set_mlt(&params, &inp).unwrap();
    [out.grad_t.val, out.d.val, out.conv_vel.val, out.grad_t.d1[0]]
}

/// Evaluates Kuhfuss (1986) 1D Time-Dependent Convection (`set_tdc`) with
/// 6-variable Implicit Function Theorem `DualOrder1<6>` sensitivities at
/// normalized shell coordinate `s_frac` $\in [0, 1]$, returning
/// `[grad_t, d, conv_vel, d_grad_t_d_gradr]`.
#[inline]
fn eval_tdc_zone(s_frac: f64) -> [f64; 4] {
    let t_val = 1.0e5 * (1.0 + 0.2 * s_frac);
    let rho_val = 1.0e-5 * (1.0 + 0.2 * s_frac);
    let p_val = rho_val * t_val * RGAS_IDEAL;
    let hp_val = p_val / (rho_val * GRAV_SOLAR);
    let gradr_val = 0.42 + 0.08 * s_frac;

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
    let inp = TdcInputs {
        chi_t: c6(1.0),
        chi_rho: c6(1.0),
        gradr: v6(gradr_val, 0),
        r: c6(RSUN),
        p: c6(p_val),
        t: v6(t_val, 4),
        rho: v6(rho_val, 5),
        dv: c6(0.0),
        cp: c6(CP_IDEAL),
        opacity: c6(1.0),
        scale_height: v6(hp_val, 3),
        gradl: v6(0.40, 2),
        grada: v6(0.40, 1),
        eq_div_w: c6(0.0),
        grav: c6(GRAV_SOLAR),
        energy: c6(0.0),
    };
    let res = set_tdc(&params, &inp).unwrap();
    [
        res.output.grad_t.val,
        res.output.d.val,
        res.output.conv_vel.val,
        res.output.grad_t.d1[0],
    ]
}

/// Shared 4-stage benchmark driver for a 4-output zone kernel `eval_zone(s)`.
///
/// Evaluates `eval_zone` at normalized radial coordinates $s_k = k / (N - 1)$
/// across:
/// - **Stage 0**: Scalar loop allocating a fresh `Vec<[f64; 4]>` per step.
/// - **Stage 1**: Pre-allocated contiguous Structure-of-Arrays (SoA) slice.
/// - **Stage 2**: Fused slice-chunk loop (`chunks_exact_mut(4)`).
/// - **Stage 3**: Scoped 4-thread parallel execution across radial chunks.
///
/// # Arguments
/// * `workload` - Static benchmark name (`"turb_mlt"` or `"turb_tdc"`).
/// * `grid` - Stellar thermodynamic profile across `n_zones` shells.
/// * `n_warmup` - Number of untimed warmup iterations.
/// * `n_repeats` - Number of timed iterations.
/// * `eval_zone` - Thread-safe per-zone evaluator returning `[f64; 4]`.
fn bench_zone_fn_4<F>(
    workload: &'static str,
    grid: &ThermoGrid,
    n_warmup: usize,
    n_repeats: usize,
    eval_zone: F,
) -> BenchRecord
where
    F: Fn(f64) -> [f64; 4] + Sync,
{
    let nz = grid.n_zones;
    let denom = (nz.saturating_sub(1)).max(1) as f64;

    // Stage 0: Scalar loop allocating Vec<[f64; 4]> per call.
    let mut ref_flat = vec![0.0_f64; nz * 4];
    let stage0 = measure_us(n_warmup, n_repeats, || {
        let mut rows = Vec::with_capacity(nz);
        for k in 0..nz {
            rows.push(eval_zone(k as f64 / denom));
        }
        for (k, r) in rows.iter().enumerate() {
            ref_flat[k * 4..(k + 1) * 4].copy_from_slice(r);
        }
    });

    // Stage 1: Contiguous pre-allocated SoA slice buffer.
    let mut out1 = vec![0.0_f64; nz * 4];
    let stage1 = measure_us(n_warmup, n_repeats, || {
        for k in 0..nz {
            let r = eval_zone(k as f64 / denom);
            out1[k * 4..(k + 1) * 4].copy_from_slice(&r);
        }
    });

    // Stage 2: Fused slice chunk loop.
    let mut out2 = vec![0.0_f64; nz * 4];
    let stage2 = measure_us(n_warmup, n_repeats, || {
        for (k, dst) in out2.chunks_exact_mut(4).enumerate() {
            dst.copy_from_slice(&eval_zone(k as f64 / denom));
        }
    });

    // Stage 3: 4-thread persistent rayon parallel execution across zone chunks.
    let mut out3 = vec![0.0_f64; nz * 4];
    let chunk_z = nz.div_ceil(STAGE3_THREADS);
    let eval_ref = &eval_zone;
    let stage3 = measure_stage3_chunks_us(
        n_warmup,
        n_repeats,
        &mut out3,
        chunk_z * 4,
        |idx, dst_chunk| {
            let z0 = idx * chunk_z;
            for (local_k, dst) in dst_chunk.chunks_exact_mut(4).enumerate() {
                let k = z0 + local_k;
                dst.copy_from_slice(&eval_ref(k as f64 / denom));
            }
        },
    );

    let max_rel_err = max_err3(
        max_rel_diff(&ref_flat, &out1),
        max_rel_diff(&ref_flat, &out2),
        max_rel_diff(&ref_flat, &out3),
    );
    let checksum: f64 = out2.iter().step_by(4).sum();

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

/// Benchmarks Cox Mixing-Length Theory (`set_mlt`) with `DualOrder1<6>`
/// sensitivities across Stages 0..3.
///
/// # Arguments
/// * `grid` - Stellar thermodynamic profile across `n_zones` shells.
/// * `n_warmup` - Number of untimed warmup iterations.
/// * `n_repeats` - Number of timed iterations.
///
/// # Returns
/// [`BenchRecord`] with Stages 0..3 latencies, parity error, and checksum.
pub fn bench_turb_mlt(
    grid: &ThermoGrid,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    bench_zone_fn_4("turb_mlt", grid, n_warmup, n_repeats, eval_mlt_zone)
}

/// Benchmarks Kuhfuss 1D Time-Dependent Convection (`set_tdc`) with
/// `DualOrder1<6>` Implicit Function Theorem sensitivities across Stages 0..3.
///
/// # Arguments
/// * `grid` - Stellar thermodynamic profile across `n_zones` shells.
/// * `n_warmup` - Number of untimed warmup iterations.
/// * `n_repeats` - Number of timed iterations.
///
/// # Returns
/// [`BenchRecord`] with Stages 0..3 latencies, parity error, and checksum.
pub fn bench_turb_tdc(
    grid: &ThermoGrid,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    bench_zone_fn_4("turb_tdc", grid, n_warmup, n_repeats, eval_tdc_zone)
}
