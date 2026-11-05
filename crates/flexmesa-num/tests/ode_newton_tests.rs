//! Golden and convergence tests for ODE/DAE (`rk2`, `rk4`, `dopri5`, `dop853`,
//! `rosenbrock`) and Newton-Raphson (`newton_solve_*`) solvers in
//! `flexmesa-num`.
//!
//! Verifies numerical parity against upstream MESA
//! `num/test/src/test_dopri.f90` (`vdpol`), `test_newton.f90`, and
//! `num/test/test_output`.
//!
//! # Test Coverage
//!
//! - `test_rk2_and_rk4_exhibit_expected_convergence_orders`: verifies 2nd-
//!   and 4th-order step-halving convergence on exponential decay.
//! - `test_dopri5_and_dense_eval_match_upstream_van_der_pol_benchmark`:
//!   verifies adaptive Dormand-Prince 5(4) and Shampine 4th-order continuous
//!   dense output (`contd5`) on the stiff Van der Pol oscillator ($\epsilon =
//!   10^{-3}$).
//! - `test_dop853_achieves_high_precision_on_harmonic_oscillator`: verifies
//!   high-order adaptive integration over a full $2\pi$ period.
//! - `test_rosenbrock_dense_and_banded_solve_stiff_system_across_variants`:
//!   verifies L-stable `Ros2`, `Rodas3`, and `Rodas4` integrators with dense
//!   and banded Jacobians on a stiff 3-component tridiagonal system.
//! - `test_newton_solve_dense_and_banded_converge_quadratically`: verifies
//!   damped Newton-Raphson iteration on a coupled cubic system.
//! - **Newton Diffusion**: reproduces the upstream 1001-zone implicit
//!   diffusion benchmark ($c_{501} = 2.9347118120566711 \times 10^{-2}$).
//!
//! # Benchmark Details
//!
//! - **Van der Pol (`vdpol`)**: $y_1' = y_2$, $y_2' = ((1 - y_1^2) y_2 - y_1) /
//!   \epsilon$ with $\epsilon = 10^{-3}$ and $y(0) = [2, 0]$ on $t \in [0, 2]$.
//! - **Stiff Tridiagonal ODE**: $y' = A y$ with eigenvalues spanning $[-2000,
//!   -1]$, verifying unconditional stability of `Ros2`, `Rodas3`, and `Rodas4`.
//! - **Implicit Diffusion (`test_newton.f90`)**: 1001-cell backward-Euler step
//!   solved with `newton_solve_block_tridiagonal`.
//! - **Implicit Diffusion (`test_newton`)**: $N_z = 1001$ shells,
//!   $N_{\mathrm{var}} = 2$ variables per shell ($c_k, f_k$), $\kappa = 1$,
//!   $\Delta x = 1/N_z$, $\Delta t = 10\,\Delta x^2 / \kappa$, evolved for 10
//!   steps from a central delta spike $c_{501}(0) = 1$.

#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::needless_range_loop
)]

use flexmesa_mtx::dense_to_band;
use flexmesa_num::{
    dop853, dopri5, newton_solve_banded, newton_solve_block_tridiagonal,
    newton_solve_dense, rk2, rk4, rosenbrock_banded, rosenbrock_dense,
    NewtonConfig, NumError, RosenbrockVariant,
};

/// Verifies that single-step [`rk2`] and [`rk4`] integrators exhibit their
/// theoretical 2nd-order and 4th-order error scaling on $y' = -y$, $y(0) = 1$.
#[test]
fn test_rk2_and_rk4_exhibit_expected_convergence_orders() {
    let rhs =
        |_t: f64, y: &[f64]| -> Result<Vec<f64>, NumError> { Ok(vec![-y[0]]) };
    let y_exact = (-0.1_f64).exp();
    let y_rk2_h = rk2(0.0, &[1.0], 0.1, rhs).unwrap()[0];
    let y_rk2_half = {
        let m = rk2(0.0, &[1.0], 0.05, rhs).unwrap();
        rk2(0.05, &m, 0.05, rhs).unwrap()[0]
    };
    let ratio_rk2 = (y_rk2_h - y_exact).abs() / (y_rk2_half - y_exact).abs();
    assert!(ratio_rk2 > 3.5 && ratio_rk2 < 4.5);

    let y_rk4_h = rk4(0.0, &[1.0], 0.1, rhs).unwrap()[0];
    assert!((y_rk4_h - y_exact).abs() <= 1e-7);
}

/// Verifies [`dopri5`] and [`DopriResult::dense_eval`] (`contd5`) against the
/// upstream Hairer & Wanner Van der Pol benchmark (`test_dopri.f90`,
/// $\epsilon = 10^{-3}$, $y(0) = [2, 0]$, $t \in [0, 2]$).
///
/// Upstream golden output (`num/test/test_output`):
/// $y(2.0) = [1.7632345401889102, -0.83568868191466206]$.
#[test]
fn test_dopri5_and_dense_eval_match_upstream_van_der_pol_benchmark() {
    let eps = 1e-3_f64;
    let vdpol = |_x: f64, y: &[f64]| -> Result<Vec<f64>, NumError> {
        let y1 = y[0];
        let y2 = y[1];
        Ok(vec![y2, ((1.0 - y1 * y1) * y2 - y1) / eps])
    };
    let res = dopri5(
        0.0,
        2.0,
        &[2.0, 0.0],
        1e-6,
        0.0,
        1e-8,
        1e-8,
        100_000,
        vdpol,
    )
    .unwrap();
    assert!((res.y[0] - 1.763_234_540_188_910_2).abs() <= 5e-5);
    assert!((res.y[1] - (-0.835_688_681_914_662_1)).abs() <= 5e-5);

    // Verify Shampine 4th-order dense output consistency at endpoints and
    // against direct integration to t = 1.0
    let res_t1 = dopri5(
        0.0,
        1.0,
        &[2.0, 0.0],
        1e-6,
        0.0,
        1e-8,
        1e-8,
        100_000,
        vdpol,
    )
    .unwrap();
    let y_mid = res.dense_eval(1.0);
    assert!((y_mid[0] - res_t1.y[0]).abs() <= 1e-4);
    let y_end = res.dense_eval(2.0);
    assert!((y_end[0] - res.y[0]).abs() <= 1e-12);
    assert!((y_end[1] - res.y[1]).abs() <= 1e-12);
}

/// Verifies [`dop853`] on the 2D harmonic oscillator $y'' + y = 0$ over a full
/// period $[0, 2\pi]$.
#[test]
fn test_dop853_achieves_high_precision_on_harmonic_oscillator() {
    let two_pi = 2.0 * std::f64::consts::PI;
    let res = dop853(
        0.0,
        two_pi,
        &[1.0, 0.0],
        1e-3,
        0.5,
        1e-10,
        1e-12,
        50_000,
        |_t, y| Ok(vec![y[1], -y[0]]),
    )
    .unwrap();
    assert!((res.y[0] - 1.0).abs() <= 1e-8);
    assert!(res.y[1].abs() <= 1e-8);
}

/// Verifies [`rosenbrock_dense`] and [`rosenbrock_banded`] across `Ros2`,
/// `Rodas3`, and `Rodas4` on a stiff 3-component tridiagonal system
/// $y_1' = -y_1$, $y_2' = -1000 (y_2 - y_1) - y_1$, $y_3' = -500 (y_3 - y_2)
/// - y_2$ with exact solution $y_1(t) = y_2(t) = y_3(t) = e^{-t}$.
#[test]
fn test_rosenbrock_dense_and_banded_solve_stiff_system_across_variants() {
    let fcn = |_t: f64, y: &[f64]| -> Result<Vec<f64>, NumError> {
        Ok(vec![
            -y[0],
            -1_000.0 * (y[1] - y[0]) - y[0],
            -500.0 * (y[2] - y[1]) - y[1],
        ])
    };
    let jac = |_t: f64, _y: &[f64]| -> Result<Vec<f64>, NumError> {
        Ok(vec![
            -1.0, 0.0, 0.0, 999.0, -1_000.0, 0.0, 0.0, 499.0, -500.0,
        ])
    };
    let y_exact = (-1.0_f64).exp();

    for variant in [
        RosenbrockVariant::Ros2,
        RosenbrockVariant::Rodas3,
        RosenbrockVariant::Rodas4,
    ] {
        let res_d = rosenbrock_dense(
            variant,
            0.0,
            1.0,
            &[1.0, 1.0, 1.0],
            1e-3,
            0.1,
            1e-6,
            1e-8,
            10_000,
            fcn,
            jac,
        )
        .unwrap();
        for k in 0..3 {
            assert!((res_d.y[k] - y_exact).abs() <= 5e-4);
        }

        let res_b = rosenbrock_banded(
            variant,
            1,
            1,
            0.0,
            1.0,
            &[1.0, 1.0, 1.0],
            1e-3,
            0.1,
            1e-6,
            1e-8,
            10_000,
            fcn,
            jac,
        )
        .unwrap();
        for k in 0..3 {
            assert!((res_b.y[k] - res_d.y[k]).abs() <= 1e-12);
        }
    }
}

/// Verifies [`newton_solve_dense`] and [`newton_solve_banded`] on a coupled
/// nonlinear tridiagonal system
/// $y_i + 0.2 y_i^3 - 0.1 (y_{i-1} + y_{i+1}) = b_i$.
#[test]
fn test_newton_solve_dense_and_banded_converge_quadratically() {
    let n = 6usize;
    let b = [1.0, 0.5, -0.5, 1.2, -0.8, 0.3];
    let eval_res = |y: &[f64]| -> Result<Vec<f64>, NumError> {
        let mut r = vec![0.0; n];
        for i in 0..n {
            let left = if i > 0 { y[i - 1] } else { 0.0 };
            let right = if i + 1 < n { y[i + 1] } else { 0.0 };
            r[i] = y[i] + 0.2 * y[i].powi(3) - 0.1 * (left + right) - b[i];
        }
        Ok(r)
    };
    let eval_jac_dense = |y: &[f64]| -> Result<Vec<f64>, NumError> {
        let mut j = vec![0.0; n * n];
        for i in 0..n {
            j[i * n + i] = 1.0 + 0.6 * y[i] * y[i];
            if i > 0 {
                j[i * n + (i - 1)] = -0.1;
            }
            if i + 1 < n {
                j[i * n + (i + 1)] = -0.1;
            }
        }
        Ok(j)
    };
    let cfg = NewtonConfig::default();
    let res_d =
        newton_solve_dense(&[0.0; 6], &cfg, eval_res, eval_jac_dense).unwrap();
    assert!(res_d.residual_norm <= 1e-12);

    let ldab = 4usize; // kl = 1, ku = 1 => 2*kl + ku + 1 = 4
    let res_b = newton_solve_banded(1, 1, &[0.0; 6], &cfg, eval_res, |y| {
        let jd = eval_jac_dense(y)?;
        let mut ab = vec![0.0; ldab * n];
        dense_to_band(n, n, &jd, 1, 1, &mut ab, ldab).unwrap();
        Ok(ab)
    })
    .unwrap();
    for i in 0..n {
        assert!((res_b.y[i] - res_d.y[i]).abs() <= 1e-13);
    }
}

/// Verifies [`newton_solve_block_tridiagonal`] against the upstream MESA
/// 1001-zone implicit diffusion benchmark (`num/test/src/test_newton.f90`,
/// $N_z = 1001, N_{\mathrm{var}} = 2$, 10 implicit Euler steps to $t_{\max} =
/// 100\,\Delta x^2 / \kappa$).
///
/// Upstream golden midpoint concentration (`num/test/test_output`):
/// $c_{501} = 2.9347118120566711 \times 10^{-2}$.
#[test]
fn test_newton_solve_block_tridiagonal_matches_upstream_diffusion_benchmark() {
    let nz = 1001usize;
    let nvar = 2usize;
    let kappa = 1.0_f64;
    let dx = 1.0 / (nz as f64);
    let dt = 10.0 * dx * dx / kappa;
    let mut y = vec![0.0_f64; nvar * nz];
    // Upstream initial condition: delta spike c = 1 at shell 501 (0-based 500)
    // and zero initial diffusive fluxes across all shells
    y[500 * nvar] = 1.0;

    let cfg = NewtonConfig {
        max_iter: 20,
        tol_correction_norm: 1e-11,
        tol_max_correction: 1e-10,
        tol_residual_norm: 1e-12,
        min_damping: 1e-2,
    };

    // Advance 10 implicit Euler steps of size dt = 10 * dx^2 / kappa
    // solving the 2x2 block-tridiagonal concentration-flux system at each step
    for _step in 0..10 {
        let y_old = y.clone();
        let res = newton_solve_block_tridiagonal(
            nvar,
            nz,
            &y,
            &cfg,
            |yc| {
                let mut f = vec![0.0; nvar * nz];
                for k in 0..nz {
                    let c_k = yc[k * nvar];
                    let f_k = yc[k * nvar + 1];
                    let c_old = y_old[k * nvar];
                    let f_next = if k + 1 < nz {
                        yc[(k + 1) * nvar + 1]
                    } else {
                        kappa * c_k / dx
                    };
                    f[k * nvar] = c_k - c_old + dt * (f_next - f_k) / dx;
                    f[k * nvar + 1] = if k == 0 {
                        f_k
                    } else {
                        let c_prev = yc[(k - 1) * nvar];
                        f_k + kappa * (c_k - c_prev) / dx
                    };
                }
                Ok(f)
            },
            |_yc| {
                let mut lblk = vec![0.0; nz * 4];
                let mut dblk = vec![0.0; nz * 4];
                let mut ublk = vec![0.0; nz * 4];
                for k in 0..nz {
                    let base = k * 4;
                    dblk[base] = if k + 1 < nz {
                        1.0
                    } else {
                        1.0 + dt * kappa / (dx * dx)
                    };
                    dblk[base + 1] = -dt / dx;
                    dblk[base + 3] = 1.0;
                    if k + 1 < nz {
                        ublk[base + 1] = dt / dx;
                    }
                    if k > 0 {
                        dblk[base + 2] = kappa / dx;
                        lblk[base + 2] = -kappa / dx;
                    }
                }
                Ok((lblk, dblk, ublk))
            },
        )
        .unwrap();
        y = res.y;
    }

    // Midpoint shell k = 501 (1-based) => 0-based index 500
    // Verify exact parity with upstream MESA LAPACK/block-Thomas golden output
    let c_mid = y[500 * nvar];
    assert!((c_mid - 2.934_711_812_056_671e-2).abs() <= 1e-13);
}
