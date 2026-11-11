// SPDX-License-Identifier: LGPL-3.0-or-later
//! Right-eigenvector back-substitution (`dtrevc`), balancing back-transform
//! (`dgebak`), and LAPACK `DGEEV` eigenvector normalization for real Schur
//! decompositions (`crates/flexmesa-mtx/src/eigen.rs`).

/// Solves a $4 \times 4$ real linear system $M x = b$ with complete pivoting
/// and a minimum singular floor `smin` (real equivalent of $2 \times 2$ complex
/// shifted Schur solve in LAPACK `dlaln2`).
pub(crate) fn solve_4x4_pivoted(
    mut m: [[f64; 4]; 4],
    mut b: [f64; 4],
    smin: f64,
) -> [f64; 4] {
    let mut col_perm = [0usize, 1, 2, 3];
    for k in 0..4 {
        let mut best = 0.0_f64;
        let (mut pr, mut pc) = (k, k);
        for (r, row) in m.iter().enumerate().skip(k) {
            for (c, &val) in row.iter().enumerate().skip(k) {
                let av = val.abs();
                if av > best {
                    best = av;
                    pr = r;
                    pc = c;
                }
            }
        }
        if pr != k {
            m.swap(k, pr);
            b.swap(k, pr);
        }
        if pc != k {
            for row in &mut m {
                row.swap(k, pc);
            }
            col_perm.swap(k, pc);
        }
        let piv = if m[k][k].abs() < smin {
            if m[k][k] < 0.0 {
                -smin
            } else {
                smin
            }
        } else {
            m[k][k]
        };
        m[k][k] = piv;
        for r in (k + 1)..4 {
            let factor = m[r][k] / piv;
            b[r] -= factor * b[k];
            for c in (k + 1)..4 {
                m[r][c] -= factor * m[k][c];
            }
        }
    }
    let mut x_perm = [0.0_f64; 4];
    for k in (0..4).rev() {
        let mut sum = b[k];
        for c in (k + 1)..4 {
            sum -= m[k][c] * x_perm[c];
        }
        x_perm[k] = sum / m[k][k];
    }
    let mut x = [0.0_f64; 4];
    for (k, &orig_col) in col_perm.iter().enumerate() {
        x[orig_col] = x_perm[k];
    }
    x
}

/// Solves a $2 \times 2$ real linear system with complete pivoting and minimum
/// pivot magnitude `smin`.
fn solve_2x2_pivoted(
    mut m: [[f64; 2]; 2],
    mut b: [f64; 2],
    smin: f64,
) -> [f64; 2] {
    let mut swapped_cols = false;
    let mut best = 0.0_f64;
    let (mut pr, mut pc) = (0usize, 0usize);
    for (r, row) in m.iter().enumerate() {
        for (c, &val) in row.iter().enumerate() {
            if val.abs() > best {
                best = val.abs();
                pr = r;
                pc = c;
            }
        }
    }
    if pr == 1 {
        m.swap(0, 1);
        b.swap(0, 1);
    }
    if pc == 1 {
        m[0].swap(0, 1);
        m[1].swap(0, 1);
        swapped_cols = true;
    }
    let p0 = if m[0][0].abs() < smin {
        smin.copysign(if m[0][0] == 0.0 { 1.0 } else { m[0][0] })
    } else {
        m[0][0]
    };
    let factor = m[1][0] / p0;
    let p1_raw = m[1][1] - factor * m[0][1];
    let p1 = if p1_raw.abs() < smin {
        smin.copysign(if p1_raw == 0.0 { 1.0 } else { p1_raw })
    } else {
        p1_raw
    };
    let b1 = b[1] - factor * b[0];
    let x1 = b1 / p1;
    let x0 = (b[0] - m[0][1] * x1) / p0;
    if swapped_cols {
        [x1, x0]
    } else {
        [x0, x1]
    }
}

/// Safe complex division $(a + i b) / (c + i d)$ with `smin` guard.
#[inline]
fn cdiv_guarded(a: f64, b: f64, c: f64, d: f64, smin: f64) -> (f64, f64) {
    let (mut cr, ci) = (c, d);
    if cr.abs() + ci.abs() < smin {
        cr = smin.copysign(if cr == 0.0 { 1.0 } else { cr });
    }
    if ci.abs() <= cr.abs() {
        let r = ci / cr;
        let den = cr + ci * r;
        ((a + b * r) / den, (b - a * r) / den)
    } else {
        let r = cr / ci;
        let den = ci + cr * r;
        ((a * r + b) / den, (b * r - a) / den)
    }
}

/// Computes right eigenvectors of a real quasi-triangular Schur matrix `t` and
/// accumulates $V = Z Y$ into row-major `vr` (`dtrevc` right-only).
pub(crate) fn compute_schur_right_eigenvectors(
    n: usize,
    t: &[f64],
    z: &[f64],
    wr: &[f64],
    wi: &[f64],
    t_norm: f64,
    vr: &mut [f64],
) {
    let ulp = f64::EPSILON;
    let smlnum = f64::MIN_POSITIVE * (n as f64) / ulp;
    let mut y_r = vec![0.0_f64; n];
    let mut y_i = vec![0.0_f64; n];
    let mut rhs_r = vec![0.0_f64; n];
    let mut rhs_i = vec![0.0_f64; n];

    let mut ki = 0usize;
    while ki < n {
        if wi[ki] == 0.0 {
            let wr_k = wr[ki];
            let smin = (ulp * wr_k.abs().max(t_norm)).max(smlnum);
            y_r.fill(0.0);
            y_r[ki] = 1.0;
            for r in 0..ki {
                rhs_r[r] = -t[r * n + ki];
            }
            let mut j = ki;
            while j > 0 {
                let j_curr = j - 1;
                if j_curr > 0 && t[j_curr * n + (j_curr - 1)] != 0.0 {
                    let j1 = j_curr - 1;
                    let j2 = j_curr;
                    let m2 = [
                        [t[j1 * n + j1] - wr_k, t[j1 * n + j2]],
                        [t[j2 * n + j1], t[j2 * n + j2] - wr_k],
                    ];
                    let sol =
                        solve_2x2_pivoted(m2, [rhs_r[j1], rhs_r[j2]], smin);
                    y_r[j1] = sol[0];
                    y_r[j2] = sol[1];
                    for r in 0..j1 {
                        rhs_r[r] -=
                            t[r * n + j1] * sol[0] + t[r * n + j2] * sol[1];
                    }
                    j -= 2;
                } else {
                    let diag = t[j_curr * n + j_curr] - wr_k;
                    let d_safe = if diag.abs() < smin {
                        smin.copysign(if diag == 0.0 { 1.0 } else { diag })
                    } else {
                        diag
                    };
                    let yj = rhs_r[j_curr] / d_safe;
                    y_r[j_curr] = yj;
                    for r in 0..j_curr {
                        rhs_r[r] -= t[r * n + j_curr] * yj;
                    }
                    j -= 1;
                }
            }
            for row in 0..n {
                let mut acc = 0.0_f64;
                let z_row = &z[(row * n)..=(row * n + ki)];
                for col in 0..=ki {
                    acc += z_row[col] * y_r[col];
                }
                vr[row * n + ki] = acc;
            }
            ki += 1;
        } else {
            let wr_k = wr[ki];
            let wi_k = wi[ki];
            let smin =
                (ulp * (wr_k.abs() + wi_k.abs()).max(t_norm)).max(smlnum);
            y_r.fill(0.0);
            y_i.fill(0.0);
            let b_val = t[ki * n + ki + 1];
            let c_val = t[(ki + 1) * n + ki];
            if b_val.abs() >= c_val.abs() {
                y_r[ki] = 1.0;
                y_i[ki + 1] = wi_k / b_val;
            } else {
                y_i[ki] = wi_k / c_val;
                y_r[ki + 1] = 1.0;
            }
            for r in 0..ki {
                let t1 = t[r * n + ki];
                let t2 = t[r * n + ki + 1];
                rhs_r[r] = -(t1 * y_r[ki] + t2 * y_r[ki + 1]);
                rhs_i[r] = -(t1 * y_i[ki] + t2 * y_i[ki + 1]);
            }

            let mut j = ki;
            while j > 0 {
                let j_curr = j - 1;
                if j_curr > 0 && t[j_curr * n + (j_curr - 1)] != 0.0 {
                    let j1 = j_curr - 1;
                    let j2 = j_curr;
                    let a11 = t[j1 * n + j1] - wr_k;
                    let a12 = t[j1 * n + j2];
                    let a21 = t[j2 * n + j1];
                    let a22 = t[j2 * n + j2] - wr_k;
                    let m4 = [
                        [a11, a12, wi_k, 0.0],
                        [a21, a22, 0.0, wi_k],
                        [-wi_k, 0.0, a11, a12],
                        [0.0, -wi_k, a21, a22],
                    ];
                    let sol = solve_4x4_pivoted(
                        m4,
                        [rhs_r[j1], rhs_r[j2], rhs_i[j1], rhs_i[j2]],
                        smin,
                    );
                    y_r[j1] = sol[0];
                    y_r[j2] = sol[1];
                    y_i[j1] = sol[2];
                    y_i[j2] = sol[3];
                    for r in 0..j1 {
                        let t1 = t[r * n + j1];
                        let t2 = t[r * n + j2];
                        rhs_r[r] -= t1 * sol[0] + t2 * sol[1];
                        rhs_i[r] -= t1 * sol[2] + t2 * sol[3];
                    }
                    j -= 2;
                } else {
                    let (yr, yi) = cdiv_guarded(
                        rhs_r[j_curr],
                        rhs_i[j_curr],
                        t[j_curr * n + j_curr] - wr_k,
                        -wi_k,
                        smin,
                    );
                    y_r[j_curr] = yr;
                    y_i[j_curr] = yi;
                    for r in 0..j_curr {
                        let tj = t[r * n + j_curr];
                        rhs_r[r] -= tj * yr;
                        rhs_i[r] -= tj * yi;
                    }
                    j -= 1;
                }
            }

            for row in 0..n {
                let z_row = &z[row * n..row * n + ki + 2];
                let (mut acc_r, mut acc_i) = (0.0_f64, 0.0_f64);
                for col in 0..=(ki + 1) {
                    acc_r += z_row[col] * y_r[col];
                    acc_i += z_row[col] * y_i[col];
                }
                vr[row * n + ki] = acc_r;
                vr[row * n + ki + 1] = acc_i;
            }
            ki += 2;
        }
    }
}

/// Applies the `dgebal` diagonal scaling and row permutations to right
/// eigenvectors `vr` and normalizes each vector to unit Euclidean norm with
/// largest-magnitude component real and positive (`dgebak` + `DGEEV` finish).
pub(crate) fn backtransform_and_normalize_right_vecs(
    n: usize,
    ilo: usize,
    ihi: usize,
    scale: &[f64],
    perm: &[usize],
    wi: &[f64],
    vr: &mut [f64],
) {
    for row in ilo..=ihi {
        let s = scale[row];
        for col in 0..n {
            vr[row * n + col] *= s;
        }
    }
    for row in (0..ilo).rev() {
        let p = perm[row];
        if p != row {
            for col in 0..n {
                vr.swap(row * n + col, p * n + col);
            }
        }
    }
    for row in (ihi + 1)..n {
        let p = perm[row];
        if p != row {
            for col in 0..n {
                vr.swap(row * n + col, p * n + col);
            }
        }
    }

    let mut j = 0usize;
    while j < n {
        if wi[j] == 0.0 {
            let mut norm2 = 0.0_f64;
            let mut max_abs = -1.0_f64;
            let mut sign = 1.0_f64;
            for i in 0..n {
                let v = vr[i * n + j];
                norm2 += v * v;
                if v.abs() > max_abs {
                    max_abs = v.abs();
                    sign = if v < 0.0 { -1.0 } else { 1.0 };
                }
            }
            let inv = if norm2 > 0.0 {
                sign / norm2.sqrt()
            } else {
                1.0
            };
            for i in 0..n {
                vr[i * n + j] *= inv;
            }
            j += 1;
        } else {
            let mut norm2 = 0.0_f64;
            let mut max_mag2 = -1.0_f64;
            let (mut best_r, mut best_i) = (1.0_f64, 0.0_f64);
            for i in 0..n {
                let r = vr[i * n + j];
                let im = vr[i * n + j + 1];
                let m2 = r * r + im * im;
                norm2 += m2;
                if m2 > max_mag2 {
                    max_mag2 = m2;
                    best_r = r;
                    best_i = im;
                }
            }
            let norm = norm2.sqrt();
            if norm > 0.0 {
                let best_abs = max_mag2.sqrt().max(f64::MIN_POSITIVE);
                let cs = best_r / best_abs;
                let sn = best_i / best_abs;
                let inv = 1.0 / norm;
                for i in 0..n {
                    let r = vr[i * n + j] * inv;
                    let im = vr[i * n + j + 1] * inv;
                    vr[i * n + j] = cs * r + sn * im;
                    vr[i * n + j + 1] = cs * im - sn * r;
                }
            }
            j += 2;
        }
    }
}
