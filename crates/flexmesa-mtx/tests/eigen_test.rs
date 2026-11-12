// SPDX-License-Identifier: LGPL-3.0-or-later
//! Verification tests for `flexmesa_mtx::eigen` (`DGEEV` parity across exact
//! closed-form, companion, isolated-row/col, and $600 \times 600$ multiscale
//! `LINA`-structured nonsymmetric eigensystems).

#![allow(clippy::cast_precision_loss, clippy::float_cmp, clippy::similar_names)]

use flexmesa_mtx::{dgeev_eigenvalues, dgeev_right, EigenResult};

fn max_eigenpair_rel_residual(n: usize, a: &[f64], res: &EigenResult) -> f64 {
    let a_frob = a.iter().map(|&x| x * x).sum::<f64>().sqrt().max(1.0e-30);
    let mut max_rel = 0.0_f64;
    let mut j = 0usize;
    while j < n {
        if res.wi[j] == 0.0 {
            let lam = res.wr[j];
            let mut err2 = 0.0_f64;
            let mut v2 = 0.0_f64;
            for i in 0..n {
                let mut av = 0.0_f64;
                for k in 0..n {
                    av += a[i * n + k] * res.vr[k * n + j];
                }
                let vi = res.vr[i * n + j];
                v2 += vi * vi;
                let diff = av - lam * vi;
                err2 += diff * diff;
            }
            let rel = err2.sqrt() / (a_frob * v2.sqrt().max(1.0e-30));
            max_rel = max_rel.max(rel);
            j += 1;
        } else {
            let (lr, li) = (res.wr[j], res.wi[j]);
            let mut err2 = 0.0_f64;
            let mut v2 = 0.0_f64;
            for i in 0..n {
                let (mut avr, mut avi) = (0.0_f64, 0.0_f64);
                for k in 0..n {
                    avr += a[i * n + k] * res.vr[k * n + j];
                    avi += a[i * n + k] * res.vr[k * n + j + 1];
                }
                let (vr_i, vi_i) = (res.vr[i * n + j], res.vr[i * n + j + 1]);
                v2 += vr_i * vr_i + vi_i * vi_i;
                let dr = avr - (lr * vr_i - li * vi_i);
                let di = avi - (lr * vi_i + li * vr_i);
                err2 += dr * dr + di * di;
            }
            let rel = err2.sqrt() / (a_frob * v2.sqrt().max(1.0e-30));
            max_rel = max_rel.max(rel);
            j += 2;
        }
    }
    max_rel
}

#[test]
fn test_dgeev_right_2x2_companion_and_isolated_permutations_match_oracle() {
    let a2 = [2.0, -5.0, 1.0, 2.0];
    let r2 = dgeev_right(2, &a2).expect("2x2 complex solve");
    assert!((r2.wr[0] - 2.0).abs() < 1.0e-14);
    assert!((r2.wi[0] - 5.0_f64.sqrt()).abs() < 1.0e-14);
    assert!(max_eigenpair_rel_residual(2, &a2, &r2) < 1.0e-14);

    // Polynomial (x - 1)(x - 3)(x - (2 + 4i))(x - (2 - 4i))
    // = x^4 - 8x^3 + 39x^2 - 92x + 60
    let comp = [
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0, //
        -60.0, 92.0, -39.0, 8.0,
    ];
    let rc = dgeev_right(4, &comp).expect("companion 4x4");
    assert!(max_eigenpair_rel_residual(4, &comp, &rc) < 1.0e-13);

    // Matrix with isolated zero-off-diagonal rows and columns (exercises
    // `ilo` and `ihi` permutations in `dgebal` + `dgebak`).
    let iso = [
        7.5, 0.0, 0.0, 0.0, 0.0, //
        2.0, 1.0, -3.0, 0.5, 0.0, //
        -1.0, 3.0, 1.0, -0.2, 0.0, //
        0.0, 0.0, 0.0, -4.25, 0.0, //
        1.5, -0.5, 2.0, 0.3, 9.0,
    ];
    let ri = dgeev_right(5, &iso).expect("isolated 5x5");
    assert!(max_eigenpair_rel_residual(5, &iso, &ri) < 1.0e-13);
}

#[test]
fn test_dgeev_right_multiscale_lina_600x600_eigenpairs_below_1e12() {
    let n_zones = 150usize;
    let n = 4 * n_zones;
    let scales = [1.0e11_f64, 1.0e5, 1.0e4, 1.0e9];
    let mut a = vec![0.0_f64; n * n];

    for iz in 0..n_zones {
        let s = (iz as f64 + 1.0) / (n_zones as f64);
        let omega0 = 1.0e-4 * (1.0 + 3.5 * s);
        let damp = 2.0e-6 * (0.3 - s);
        let b = 4 * iz;
        let blk = [
            [0.0, 1.0, 0.0, 0.0],
            [-omega0 * omega0, damp, 0.15 * omega0, 0.05 * omega0],
            [-0.2 * omega0, -0.08 * omega0, -3.5 * omega0, 0.12 * omega0],
            [0.04 * omega0, 0.02 * omega0, 0.25 * omega0, -1.8 * omega0],
        ];
        for r in 0..4 {
            for c in 0..4 {
                a[(b + r) * n + (b + c)] = blk[r][c] * (scales[r] / scales[c]);
            }
        }
        if iz > 0 {
            let bm = 4 * (iz - 1);
            a[(b + 1) * n + bm] =
                0.35 * omega0 * omega0 * (scales[1] / scales[0]);
            a[(b + 2) * n + (bm + 2)] = 1.2 * omega0;
        }
        if iz + 1 < n_zones {
            let bp = 4 * (iz + 1);
            a[(b + 1) * n + bp] =
                0.35 * omega0 * omega0 * (scales[1] / scales[0]);
            a[(b + 2) * n + (bp + 2)] = 1.1 * omega0;
            a[(b + 3) * n + (bp + 3)] = 0.4 * omega0;
        }
    }

    let (wr_only, wi_only) =
        dgeev_eigenvalues(n, &a).expect("600x600 eigenvalues only");
    let res = dgeev_right(n, &a).expect("600x600 multiscale dgeev_right");
    for j in 0..n {
        assert!(
            (wr_only[j] - res.wr[j]).abs()
                <= 1.0e-14 * res.wr[j].abs().max(1.0)
        );
        assert!(
            (wi_only[j] - res.wi[j]).abs()
                <= 1.0e-14 * res.wi[j].abs().max(1.0)
        );
    }

    let mut a_unscaled = vec![0.0_f64; n * n];
    let mut vr_unscaled = vec![0.0_f64; n * n];
    for i in 0..n {
        let si = scales[i % 4];
        for j in 0..n {
            let sj = scales[j % 4];
            a_unscaled[i * n + j] = a[i * n + j] * (sj / si);
            vr_unscaled[i * n + j] = res.vr[i * n + j] / si;
        }
    }
    let res_unscaled = EigenResult {
        wr: res.wr,
        wi: res.wi,
        vr: vr_unscaled,
    };
    let max_rel = max_eigenpair_rel_residual(n, &a_unscaled, &res_unscaled);
    assert!(
        max_rel < 1.0e-12,
        "600x600 eigenpair relative residual {max_rel:.3e} >= 1e-12"
    );
}
