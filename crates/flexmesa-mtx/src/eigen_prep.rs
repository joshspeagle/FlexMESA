// SPDX-License-Identifier: LGPL-3.0-or-later
//! Parlett-Reinsch radix-2 balancing (`dgebal`) and Householder upper
//! Hessenberg reduction (`dgehrd`) for `flexmesa-mtx::eigen`.

/// Balances a row-major $N \times N$ matrix `a` in-place via exact row/column
/// permutations and radix-2 diagonal similarity $D^{-1} P A P^{\top} D$
/// (`dgebal`). Returns `(ilo, ihi, scale, perm)`.
pub(crate) fn balance_matrix(
    n: usize,
    a: &mut [f64],
) -> (usize, usize, Vec<f64>, Vec<usize>) {
    const SCLFAC: f64 = 2.0;
    const FACTOR: f64 = 0.95;

    let mut scale = vec![1.0_f64; n];
    let mut perm: Vec<usize> = (0..n).collect();
    if n <= 1 {
        return (0, n.saturating_sub(1), scale, perm);
    }

    let mut ilo = 0usize;
    let mut ihi = n - 1;

    // 1. Isolate rows with zero off-diagonal entries into trailing positions.
    'isolate_hi: loop {
        if ihi == 0 {
            break;
        }
        for i in (0..=ihi).rev() {
            let isolated = (0..=ihi).all(|j| j == i || a[i * n + j] == 0.0);
            if isolated {
                perm[ihi] = i;
                if i != ihi {
                    for col in 0..n {
                        a.swap(i * n + col, ihi * n + col);
                    }
                    for row in 0..n {
                        a.swap(row * n + i, row * n + ihi);
                    }
                }
                if ihi == 0 {
                    break 'isolate_hi;
                }
                ihi -= 1;
                continue 'isolate_hi;
            }
        }
        break;
    }

    // 2. Isolate columns with zero off-diagonal entries into leading positions.
    'isolate_lo: loop {
        if ilo >= ihi {
            break;
        }
        let lo = ilo;
        for j in lo..=ihi {
            let isolated = (lo..=ihi).all(|i| i == j || a[i * n + j] == 0.0);
            if isolated {
                perm[lo] = j;
                if j != lo {
                    for col in 0..n {
                        a.swap(j * n + col, lo * n + col);
                    }
                    for row in 0..n {
                        a.swap(row * n + j, row * n + lo);
                    }
                }
                ilo = lo + 1;
                continue 'isolate_lo;
            }
        }
        break;
    }

    // 3. Iterative radix-2 diagonal scaling on `ilo..=ihi`.
    let sfmin1 = f64::MIN_POSITIVE / f64::EPSILON;
    let sfmax1 = 1.0 / sfmin1;
    let sfmin2 = sfmin1 * SCLFAC;
    let sfmax2 = 1.0 / sfmin2;

    let mut converged = false;
    while !converged {
        converged = true;
        for i in ilo..=ihi {
            let mut c = 0.0_f64;
            let mut r = 0.0_f64;
            for j in ilo..=ihi {
                if j != i {
                    c += a[j * n + i].abs();
                    r += a[i * n + j].abs();
                }
            }
            if c == 0.0 || r == 0.0 {
                continue;
            }
            let mut f = 1.0_f64;
            let s = c + r;
            while c < r / SCLFAC
                && f.max(c) < sfmax2
                && r.min(r / SCLFAC) > sfmin2
            {
                f *= SCLFAC;
                c *= SCLFAC;
                r /= SCLFAC;
            }
            while c / SCLFAC >= r
                && r < sfmax2
                && f.min(c).min(c / SCLFAC) > sfmin2
            {
                f /= SCLFAC;
                c /= SCLFAC;
                r *= SCLFAC;
            }
            if (c + r) >= FACTOR * s {
                continue;
            }
            if (f < 1.0 && scale[i] < 1.0 && f * scale[i] <= sfmin1)
                || (f > 1.0 && scale[i] > 1.0 && scale[i] >= sfmax1 / f)
            {
                continue;
            }
            let inv_f = 1.0 / f;
            scale[i] *= f;
            converged = false;
            for col in ilo..n {
                a[i * n + col] *= inv_f;
            }
            for row in 0..=ihi {
                a[row * n + i] *= f;
            }
        }
    }

    (ilo, ihi, scale, perm)
}

/// Reduces `a[ilo..=ihi, ilo..=ihi]` to upper Hessenberg form $H = Q^{\top} A Q$
/// in-place (`dgehrd`) and optionally accumulates $Q$ into row-major `z`.
pub(crate) fn reduce_to_hessenberg(
    n: usize,
    ilo: usize,
    ihi: usize,
    a: &mut [f64],
    mut z: Option<&mut [f64]>,
) {
    if let Some(z_mat) = z.as_deref_mut() {
        z_mat.fill(0.0);
        for i in 0..n {
            z_mat[i * n + i] = 1.0;
        }
    }
    if ihi <= ilo + 1 {
        return;
    }

    let mut v = vec![0.0_f64; n];
    for k in ilo..(ihi - 1) {
        let mut scale = 0.0_f64;
        for i in (k + 1)..=ihi {
            scale += a[i * n + k].abs();
        }
        if scale == 0.0 {
            continue;
        }
        let mut sum_sq = 0.0_f64;
        for i in (k + 1)..=ihi {
            let u = a[i * n + k] / scale;
            v[i] = u;
            sum_sq += u * u;
        }
        let alpha = v[k + 1];
        let norm = sum_sq.sqrt();
        let beta = if alpha >= 0.0 { -norm } else { norm };
        let tau = (beta - alpha) / beta;
        let denom = alpha - beta;
        v[k + 1] = 1.0;
        for val in &mut v[(k + 2)..=ihi] {
            *val /= denom;
        }
        a[(k + 1) * n + k] = scale * beta;
        for i in (k + 2)..=ihi {
            a[i * n + k] = 0.0;
        }

        // Left update: A[k+1..=ihi, k+1..n] -= tau * v * (v^T * A)
        for col in (k + 1)..n {
            let mut dot = 0.0_f64;
            for row in (k + 1)..=ihi {
                dot += v[row] * a[row * n + col];
            }
            let s = tau * dot;
            for row in (k + 1)..=ihi {
                a[row * n + col] -= s * v[row];
            }
        }

        // Right update: A[0..=ihi, k+1..=ihi] -= tau * (A * v) * v^T
        for row in 0..=ihi {
            let mut dot = 0.0_f64;
            for col in (k + 1)..=ihi {
                dot += a[row * n + col] * v[col];
            }
            let s = tau * dot;
            for col in (k + 1)..=ihi {
                a[row * n + col] -= s * v[col];
            }
        }

        // Accumulate orthogonal matrix Z <- Z * H_k
        if let Some(z_mat) = z.as_deref_mut() {
            for row in ilo..=ihi {
                let mut dot = 0.0_f64;
                for col in (k + 1)..=ihi {
                    dot += z_mat[row * n + col] * v[col];
                }
                let s = tau * dot;
                for col in (k + 1)..=ihi {
                    z_mat[row * n + col] -= s * v[col];
                }
            }
        }
    }
}
