//! Block-tridiagonal golden, Harwell-Boeing, autodiff, and property tests.
//!
//! Verifies `solve_block_tridiagonal`, `solve_block_tridiag_bcyclic`,
//! `block_dble_mv`, `block_multiply_xa`, `lu_factored_block_dble_mv`,
//! `write_block_tridiagonal_text`, `parse_block_tridiagonal_text`, and
//! `solve_dense_dual`.
//!
//! # Test Coverage
//!
//! 1. **Coupled $25 \times 20$ Block-Tridiagonal System**: Matches the scale of
//!    upstream `test_block_tridiagonal.f90` ($N_{\mathrm{var}} = 25$,
//!    $N_z = 20$), verifying forward-backward block Thomas solve residuals,
//!    LU-factored block matrix-vector multiplication, bilinear duality
//!    $b^{\top}(A x) = (b^{\top} A) x$, and Harwell-Boeing `RUA` text I/O.
//! 2. **Implicit Forward-Mode Autodiff**: Verifies `solve_dense_dual` against
//!    centered finite differences to $< 10^{-9}$.
//! 3. **Property-Based Verification**: Compares `solve_block_tridiagonal` and
//!    `solve_block_tridiag_bcyclic` against full dense LU (`solve_dense`)
//!    across random non-symmetric block systems.
//! 4. **Parallel Block Cyclic Reduction (`bcyclic`) Parity**: Verifies both
//!    serial and `rayon`-parallel Hirshman et al. (2010) odd-even cyclic
//!    reduction (`solve_block_tridiag_bcyclic`) against sequential block
//!    Thomas across odd, prime, and power-of-two grid sizes up to
//!    $N_{\mathrm{zones}} = 4096$ with non-symmetric, non-commutative,
//!    shell-varying blocks.
//!
//! All tests run deterministically in IEEE-754 `f64` precision.

#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names
)]

use flexmesa_autodiff::DualOrder1;
use flexmesa_mtx::{
    block_dble_mv, block_multiply_xa, dgetrf, lu_factored_block_dble_mv,
    parse_block_tridiagonal_text, solve_block_tridiag_bcyclic,
    solve_block_tridiagonal, solve_dense, solve_dense_dual,
    write_block_tridiagonal_text,
};
use proptest::prelude::*;

/// Verifies `solve_block_tridiagonal`, `block_dble_mv`, `block_multiply_xa`,
/// `lu_factored_block_dble_mv`, and Harwell-Boeing text roundtrip on a coupled
/// $N_z = 20, N_{\mathrm{var}} = 25$ system matching upstream
/// `test_block_tridiagonal.f90`.
///
/// Checks that:
/// - The block Thomas solution satisfies
///   $\|A x - b\|_\infty / \|b\|_\infty < 10^{-12}$.
/// - Reconstructing $A x$ from LU-factored diagonal blocks agrees to
///   $< 10^{-12}$.
/// - Left multiplication $b^{\top} A$ satisfies
///   $b^{\top} (A x) = (b^{\top} A) x$.
/// - Harwell-Boeing `RUA` serialization and parsing round-trips all blocks.
#[test]
fn test_block_tridiagonal_25x20_and_hb_roundtrip() {
    let nvar = 25;
    let nz = 20;
    let blk_sz = nvar * nvar;
    let mut lblk = vec![0.0; nz * blk_sz];
    let mut dblk = vec![0.0; nz * blk_sz];
    let mut ublk = vec![0.0; nz * blk_sz];
    let mut rhs = vec![0.0; nz * nvar];

    for k in 0..nz {
        for i in 0..nvar {
            rhs[k * nvar + i] =
                1.1 + 0.05 * (k as f64) - 0.04 * (i as f64);
            for j in 0..nvar {
                let idx = k * blk_sz + i * nvar + j;
                dblk[idx] = if i == j {
                    8.0 + 0.1 * ((k + i) as f64)
                } else {
                    0.02 * (((i + 2 * j + k) % 7) as f64 - 3.0)
                };
                if k > 0 {
                    lblk[idx] = -0.05 * (((2 * i + j + k) % 5) as f64 - 2.0);
                }
                if k + 1 < nz {
                    ublk[idx] = -0.04 * (((i + 3 * j + k) % 5) as f64 - 2.0);
                }
            }
        }
    }

    let x = solve_block_tridiagonal(nvar, nz, &lblk, &dblk, &ublk, &rhs)
        .unwrap();
    let mut prod = vec![0.0; nz * nvar];
    block_dble_mv(nvar, nz, &lblk, &dblk, &ublk, &x, &mut prod).unwrap();
    for idx in 0..(nz * nvar) {
        let rel_err = (prod[idx] - rhs[idx]).abs() / rhs[idx].abs().max(1.0);
        assert!(rel_err < 1e-12, "block residual at {idx}: {rel_err}");
    }

    // Verify lu_factored_block_dble_mv when diagonal blocks are LU-factored
    let mut d_lu = dblk.clone();
    let mut ipiv = vec![0usize; nz * nvar];
    for k in 0..nz {
        dgetrf(
            nvar,
            &mut d_lu[k * blk_sz..(k + 1) * blk_sz],
            &mut ipiv[k * nvar..(k + 1) * nvar],
        )
        .unwrap();
    }
    let mut prod_lu = vec![0.0; nz * nvar];
    lu_factored_block_dble_mv(
        nvar, nz, &lblk, &d_lu, &ublk, &ipiv, &x, &mut prod_lu,
    )
    .unwrap();
    for idx in 0..(nz * nvar) {
        assert!((prod_lu[idx] - prod[idx]).abs() < 1e-12);
    }

    // Verify left product block_multiply_xa: x^T (A y) == (x^T A) y
    let mut xa = vec![0.0; nz * nvar];
    block_multiply_xa(nvar, nz, &lblk, &dblk, &ublk, &rhs, &mut xa).unwrap();
    let dot_1: f64 = rhs.iter().zip(prod.iter()).map(|(u, v)| u * v).sum();
    let dot_2: f64 = xa.iter().zip(x.iter()).map(|(u, v)| u * v).sum();
    assert!((dot_1 - dot_2).abs() < 1e-11);

    // Verify Harwell-Boeing text serialization and parsing roundtrip
    let nvar_s = 3;
    let nz_s = 3;
    let sz_s = nvar_s * nvar_s;
    let l_s = lblk[..(nz_s * sz_s)].to_vec();
    let d_s = dblk[..(nz_s * sz_s)].to_vec();
    let u_s = ublk[..(nz_s * sz_s)].to_vec();
    let text =
        write_block_tridiagonal_text(nvar_s, nz_s, &l_s, &d_s, &u_s).unwrap();
    let (nv_p, nz_p, l_p, d_p, u_p) =
        parse_block_tridiagonal_text(&text).unwrap();
    assert_eq!((nv_p, nz_p), (nvar_s, nz_s));
    for idx in 0..(nz_s * sz_s) {
        assert!((d_p[idx] - d_s[idx]).abs() < 1e-14);
    }
    for k in 1..nz_s {
        for idx in (k * sz_s)..((k + 1) * sz_s) {
            assert!((l_p[idx] - l_s[idx]).abs() < 1e-14);
        }
    }
    for k in 0..(nz_s - 1) {
        for idx in (k * sz_s)..((k + 1) * sz_s) {
            assert!((u_p[idx] - u_s[idx]).abs() < 1e-14);
        }
    }
}

/// Verifies `solve_dense_dual` implicit differentiation against centered finite
/// differences on a parameter-dependent $3 \times 3$ linear system.
///
/// Evaluates $A(p_0, p_1)\,x(p_0, p_1) = b(p_0, p_1)$ with `DualOrder1<2>` at
/// $p = (1.2, -0.7)$ and compares both tangent columns
/// $\partial x / \partial p_0$ and $\partial x / \partial p_1$ against $O(h^2)$
/// centered finite differences with step $h = 10^{-7}$.
///
/// The implicit tangent solve reuses the primal LU factorization $P A = L U$
/// across all parameter directions without differentiating through pivot swaps.
#[test]
fn test_solve_dense_dual_matches_finite_differences() {
    let n = 3;
    let p0 = [1.2_f64, -0.7_f64];
    let eval_system = |p: [DualOrder1<2>; 2]| {
        let a = [
            DualOrder1::constant(4.0) + p[0] * DualOrder1::constant(0.5),
            DualOrder1::constant(-1.0) + p[1] * DualOrder1::constant(0.2),
            DualOrder1::constant(0.3),
            DualOrder1::constant(0.5),
            DualOrder1::constant(3.5) - p[0] * DualOrder1::constant(0.3),
            DualOrder1::constant(-0.4) + p[1],
            DualOrder1::constant(-0.2),
            DualOrder1::constant(0.7) + p[0],
            DualOrder1::constant(5.0) + p[1] * DualOrder1::constant(0.4),
        ];
        let b = [
            DualOrder1::constant(2.0) + p[0] * p[1],
            DualOrder1::constant(-1.0) + p[0] * DualOrder1::constant(1.5),
            DualOrder1::constant(3.0) - p[1] * DualOrder1::constant(0.8),
        ];
        solve_dense_dual(n, &a, &b).unwrap()
    };

    let p_dual = [
        DualOrder1::independent(p0[0], 0),
        DualOrder1::independent(p0[1], 1),
    ];
    let sol = eval_system(p_dual);

    let h = 1e-7;
    for m in 0..2 {
        let mut p_plus = p0;
        let mut p_minus = p0;
        p_plus[m] += h;
        p_minus[m] -= h;
        let sol_plus = eval_system([
            DualOrder1::constant(p_plus[0]),
            DualOrder1::constant(p_plus[1]),
        ]);
        let sol_minus = eval_system([
            DualOrder1::constant(p_minus[0]),
            DualOrder1::constant(p_minus[1]),
        ]);
        for i in 0..n {
            let fd = (sol_plus[i].val - sol_minus[i].val) / (2.0 * h);
            assert!(
                (sol[i].d1[m] - fd).abs() < 1e-9,
                "mismatch at i={i}, m={m}: dual={} vs fd={fd}",
                sol[i].d1[m]
            );
        }
    }
}

proptest! {
    /// Property test verifying that `solve_block_tridiagonal` matches the full
    /// expanded dense LU solution (`solve_dense`) to within $10^{-11}$.
    ///
    /// Constructs a $3$-shell, $2$-variable block-tridiagonal system
    /// ($6 \times 6$ total dimension) with diagonally dominant diagonal blocks
    /// $D_k$, random sub-diagonal blocks $L_1, L_2$, and random super-diagonal
    /// blocks $U_0, U_1$, and verifies numerical equivalence with dense LU.
    #[test]
    fn prop_block_thomas_matches_dense_solve(
        d_pert in prop::collection::vec(-0.3_f64..0.3_f64, 12),
        l_vals in prop::collection::vec(-0.2_f64..0.2_f64, 12),
        u_vals in prop::collection::vec(-0.2_f64..0.2_f64, 12),
        rhs in prop::collection::vec(-5.0_f64..5.0_f64, 6),
    ) {
        let nvar = 2;
        let nz = 3;
        let n_tot = nvar * nz;
        let mut dblk = d_pert;
        for k in 0..nz {
            for i in 0..nvar {
                dblk[k * 4 + i * 2 + i] += 5.0;
            }
        }
        let mut lblk = l_vals;
        let mut ublk = u_vals;
        lblk[0..4].fill(0.0);
        ublk[8..12].fill(0.0);

        let x_blk = solve_block_tridiagonal(
            nvar, nz, &lblk, &dblk, &ublk, &rhs,
        ).unwrap();

        let mut a_dense = vec![0.0; n_tot * n_tot];
        for k in 0..nz {
            for i in 0..nvar {
                for j in 0..nvar {
                    let row = k * nvar + i;
                    let col = k * nvar + j;
                    a_dense[row * n_tot + col] = dblk[k * 4 + i * 2 + j];
                    if k > 0 {
                        let col_m1 = (k - 1) * nvar + j;
                        a_dense[row * n_tot + col_m1] = lblk[k * 4 + i * 2 + j];
                    }
                    if k + 1 < nz {
                        let col_p1 = (k + 1) * nvar + j;
                        a_dense[row * n_tot + col_p1] = ublk[k * 4 + i * 2 + j];
                    }
                }
            }
        }
        let x_dense = solve_dense(n_tot, &a_dense, &rhs).unwrap();
        let x_bcyclic = solve_block_tridiag_bcyclic(
            nvar, nz, &lblk, &dblk, &ublk, &rhs, false,
        ).unwrap();
        for idx in 0..n_tot {
            prop_assert!((x_blk[idx] - x_dense[idx]).abs() < 1e-11);
            prop_assert!((x_bcyclic[idx] - x_dense[idx]).abs() < 1e-11);
        }
    }
}

/// Verifies `solve_block_tridiag_bcyclic` (serial and parallel `rayon`) against
/// `solve_block_tridiagonal` across odd, prime, and power-of-two grid sizes up
/// to $N_{\mathrm{zones}} = 4096$.
///
/// Constructs shell-varying, non-symmetric, non-commutative $4 \times 4$ blocks
/// $D_k \ne D_k^{\top}$, $L_k \ne L_k^{\top}$, and $U_k \ne U_k^{\top}$ so that
/// any block-multiplication order reversal ($B A$ vs. $A B$), transpose error,
/// or shell-index offset on $L_k$ or $U_k$ is immediately detected.
///
/// Checks both `parallel = false` and `parallel = true` across
/// $N_{\mathrm{zones}} \in \{1, 2, 3, 7, 8, 15, 32, 65, 128, 1024, 4096\}$ to
/// relative error $< 10^{-13}$.
#[test]
fn test_bcyclic_matches_block_thomas_across_grid_sizes() {
    let nvar = 4usize;
    let blk_sz = nvar * nvar;
    let grid_sizes = [1usize, 2, 3, 7, 8, 15, 32, 65, 128, 1024, 4096];
    for &nz in &grid_sizes {
        let mut lblk = vec![0.0_f64; nz * blk_sz];
        let mut dblk = vec![0.0_f64; nz * blk_sz];
        let mut ublk = vec![0.0_f64; nz * blk_sz];
        let mut rhs = vec![0.0_f64; nz * nvar];
        for k in 0..nz {
            let kf = k as f64;
            let s = (k + 1) as f64 / (nz as f64);
            for i in 0..nvar {
                rhs[k * nvar + i] = (0.3 * kf + 0.7 * (i as f64)).sin();
                for j in 0..nvar {
                    let idx = k * blk_sz + i * nvar + j;
                    let d_off =
                        0.03 * (((3 * i + j + 2 * k + 1) % 7) as f64 - 3.0);
                    dblk[idx] = if i == j {
                        5.0 + s + 0.15 * (i as f64) + d_off
                    } else {
                        d_off
                    };
                    if k > 0 {
                        lblk[idx] = -0.06
                            * (((2 * i + 5 * j + k + 2) % 9) as f64 - 4.0);
                    }
                    if k + 1 < nz {
                        ublk[idx] = -0.05
                            * (((4 * i + j + 3 * k + 1) % 9) as f64 - 4.0);
                    }
                }
            }
        }
        let x_ref =
            solve_block_tridiagonal(nvar, nz, &lblk, &dblk, &ublk, &rhs)
                .unwrap();
        for &par in &[false, true] {
            let x_bc = solve_block_tridiag_bcyclic(
                nvar, nz, &lblk, &dblk, &ublk, &rhs, par,
            )
            .unwrap();
            for (a, b) in x_ref.iter().zip(x_bc.iter()) {
                let rel = (a - b).abs() / a.abs().max(1.0);
                assert!(
                    rel < 1.0e-13,
                    "nz={nz} par={par} rel={rel:e} a={a} b={b}"
                );
            }
        }
    }
}
