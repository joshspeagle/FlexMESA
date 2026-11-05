//! Dense LU factorization, linear system solvers, and implicit autodiff.
//!
//! Stores dense $N \times N$ matrices in row-major order (`a[i * n + j]`).
//! Provides `dgetrf` (LU factorization with partial pivoting), `dgetrs` and
//! `dgetrs_multi` (forward/backward substitution), 1-norm reciprocal condition
//! number estimation (`rcond_dense`), and implicit first-order forward-mode
//! differentiation (`solve_dense_dual`).
//!
//! # Implicit Differentiation
//!
//! Differentiating $A(p)\,x(p) = b(p)$ with respect to parameter $p_m$ yields
//! the tangent linear system:
//! $$A(p)\,\frac{\partial x}{\partial p_m}
//! = \frac{\partial b}{\partial p_m} - \frac{\partial A}{\partial p_m}\,x(p)$$
//! which reuses the primal LU factorization $P A = L U$ across all $K$ tangent
//! directions in $O(K N^2)$ post-factorization operations.

use crate::MtxError;
use flexmesa_autodiff::DualOrder1;
use flexmesa_utils::is_bad_slice;

/// Computes the in-place LU factorization $P A = L U$ of a row-major
/// $N \times N$ matrix `a` with partial pivoting.
///
/// On exit, the unit lower-triangular factor $L$ is stored strictly below the
/// diagonal of `a`, the upper-triangular factor $U$ is stored on and above the
/// diagonal, and `ipiv[k]` records the 0-based row index swapped with row `k`
/// at step `k`.
///
/// # Errors
///
/// Returns [`MtxError::DimensionMismatch`] if `a.len() != n * n` or
/// `ipiv.len() != n`, [`MtxError::NonFiniteInput`] if `a` contains `NaN` or
/// $\pm\infty$, or [`MtxError::SingularMatrix`] if a pivot is exactly zero.
pub fn dgetrf(
    n: usize,
    a: &mut [f64],
    ipiv: &mut [usize],
) -> Result<(), MtxError> {
    if n == 0 {
        return Err(MtxError::InvalidParameter("n must be >= 1"));
    }
    if a.len() != n * n || ipiv.len() != n {
        return Err(MtxError::DimensionMismatch(
            "a must have length n * n and ipiv must have length n",
        ));
    }
    if is_bad_slice(a) {
        return Err(MtxError::NonFiniteInput("dense matrix a contains NaN/Inf"));
    }

    for k in 0..n {
        let mut pivot_row = k;
        let mut max_val = a[k * n + k].abs();
        for i in (k + 1)..n {
            let val = a[i * n + k].abs();
            if val > max_val {
                max_val = val;
                pivot_row = i;
            }
        }
        ipiv[k] = pivot_row;
        if max_val == 0.0 {
            return Err(MtxError::SingularMatrix(k));
        }
        if pivot_row != k {
            for j in 0..n {
                a.swap(k * n + j, pivot_row * n + j);
            }
        }
        let diag = a[k * n + k];
        for i in (k + 1)..n {
            let mult = a[i * n + k] / diag;
            a[i * n + k] = mult;
            for j in (k + 1)..n {
                a[i * n + j] -= mult * a[k * n + j];
            }
        }
    }
    Ok(())
}

/// Solves $A x = b$ (or $A^{\top} x = b$ when `trans` is `true`) in-place using
/// the LU factors and pivot indices produced by [`dgetrf`].
///
/// # Errors
///
/// Returns [`MtxError::DimensionMismatch`] if slice lengths do not match `n`,
/// [`MtxError::InvalidParameter`] if any pivot index is out of bounds,
/// [`MtxError::NonFiniteInput`] if `lu` or `b` contains `NaN` or $\pm\infty$,
/// or [`MtxError::SingularMatrix`] if a diagonal entry of $U$ is zero.
pub fn dgetrs_trans(
    trans: bool,
    n: usize,
    lu: &[f64],
    ipiv: &[usize],
    rhs: &mut [f64],
) -> Result<(), MtxError> {
    if n == 0 {
        return Err(MtxError::InvalidParameter("n must be >= 1"));
    }
    if lu.len() != n * n || ipiv.len() != n || rhs.len() != n {
        return Err(MtxError::DimensionMismatch(
            "lu must have length n * n and ipiv, rhs must have length n",
        ));
    }
    if is_bad_slice(lu) || is_bad_slice(rhs) {
        return Err(MtxError::NonFiniteInput("lu or rhs contains NaN/Inf"));
    }
    for &piv in ipiv {
        if piv >= n {
            return Err(MtxError::InvalidParameter("pivot index out of bounds"));
        }
    }

    if !trans {
        for (k, &piv) in ipiv.iter().enumerate() {
            if piv != k {
                rhs.swap(k, piv);
            }
        }
        for i in 0..n {
            let mut sum = rhs[i];
            for j in 0..i {
                sum -= lu[i * n + j] * rhs[j];
            }
            rhs[i] = sum;
        }
        for i in (0..n).rev() {
            let diag = lu[i * n + i];
            if diag == 0.0 {
                return Err(MtxError::SingularMatrix(i));
            }
            let mut sum = rhs[i];
            for j in (i + 1)..n {
                sum -= lu[i * n + j] * rhs[j];
            }
            rhs[i] = sum / diag;
        }
    } else {
        for i in 0..n {
            let diag = lu[i * n + i];
            if diag == 0.0 {
                return Err(MtxError::SingularMatrix(i));
            }
            let mut sum = rhs[i];
            for j in 0..i {
                sum -= lu[j * n + i] * rhs[j];
            }
            rhs[i] = sum / diag;
        }
        for i in (0..n).rev() {
            let mut sum = rhs[i];
            for j in (i + 1)..n {
                sum -= lu[j * n + i] * rhs[j];
            }
            rhs[i] = sum;
        }
        for (k, &piv) in ipiv.iter().enumerate().rev() {
            if piv != k {
                rhs.swap(k, piv);
            }
        }
    }
    Ok(())
}

/// Solves $A x = b$ in-place using the LU factors from [`dgetrf`].
///
/// # Errors
///
/// Propagates any error from [`dgetrs_trans`].
pub fn dgetrs(
    n: usize,
    lu: &[f64],
    ipiv: &[usize],
    rhs: &mut [f64],
) -> Result<(), MtxError> {
    dgetrs_trans(false, n, lu, ipiv, rhs)
}

/// Solves $A X = B$ in-place for `nrhs` right-hand sides where `rhs_mat` has
/// row-major shape `(n, nrhs)` (`rhs_mat[i * nrhs + col_idx]`).
///
/// # Errors
///
/// Returns [`MtxError::DimensionMismatch`] if `rhs_mat.len() != n * nrhs`, or
/// propagates any error from [`dgetrf`] / [`dgetrs`].
pub fn dgetrs_multi(
    n: usize,
    nrhs: usize,
    lu: &[f64],
    ipiv: &[usize],
    rhs_mat: &mut [f64],
) -> Result<(), MtxError> {
    if n == 0 || nrhs == 0 {
        return Err(MtxError::InvalidParameter("n and nrhs must be >= 1"));
    }
    if lu.len() != n * n || ipiv.len() != n || rhs_mat.len() != n * nrhs {
        return Err(MtxError::DimensionMismatch(
            "rhs_mat must have length n * nrhs for dgetrs_multi",
        ));
    }
    let mut col = vec![0.0; n];
    for rhs_idx in 0..nrhs {
        for i in 0..n {
            col[i] = rhs_mat[i * nrhs + rhs_idx];
        }
        dgetrs(n, lu, ipiv, &mut col)?;
        for i in 0..n {
            rhs_mat[i * nrhs + rhs_idx] = col[i];
        }
    }
    Ok(())
}

/// Solves the dense linear system $A x = b$ and returns the solution $x$.
///
/// Copies `a` and `rhs` into workspace buffers, factors $P A = L U$ via
/// [`dgetrf`], and solves via [`dgetrs`].
///
/// # Errors
///
/// Propagates any error from [`dgetrf`] or [`dgetrs`].
pub fn solve_dense(
    n: usize,
    a: &[f64],
    rhs: &[f64],
) -> Result<Vec<f64>, MtxError> {
    if a.len() != n * n || rhs.len() != n {
        return Err(MtxError::DimensionMismatch(
            "a must have length n * n and rhs must have length n",
        ));
    }
    let mut lu = a.to_vec();
    let mut ipiv = vec![0usize; n];
    let mut sol = rhs.to_vec();
    dgetrf(n, &mut lu, &mut ipiv)?;
    dgetrs(n, &lu, &ipiv, &mut sol)?;
    Ok(sol)
}

/// Solves the dense multi-RHS linear system $A X = B$ for `B` of row-major
/// shape `(n, nrhs)`.
///
/// # Errors
///
/// Propagates any error from [`dgetrf`] or [`dgetrs_multi`].
pub fn solve_dense_multi(
    n: usize,
    nrhs: usize,
    a: &[f64],
    rhs_mat: &[f64],
) -> Result<Vec<f64>, MtxError> {
    if a.len() != n * n || rhs_mat.len() != n * nrhs {
        return Err(MtxError::DimensionMismatch(
            "a must have length n * n and rhs_mat must have length n * nrhs",
        ));
    }
    let mut lu = a.to_vec();
    let mut ipiv = vec![0usize; n];
    let mut sol = rhs_mat.to_vec();
    dgetrf(n, &mut lu, &mut ipiv)?;
    dgetrs_multi(n, nrhs, &lu, &ipiv, &mut sol)?;
    Ok(sol)
}

/// Estimates the 1-norm reciprocal condition number
/// $\mathrm{rcond} = (\|A\|_1 \|A^{-1}\|_1)^{-1}$ from $A$ and its LU factors
/// using Hager's 1-norm estimator (`dgecon`).
///
/// # Errors
///
/// Propagates any dimension error from [`dgetrs_trans`].
pub fn rcond_dense(
    n: usize,
    a_mat: &[f64],
    lu: &[f64],
    ipiv: &[usize],
) -> Result<f64, MtxError> {
    if n == 0
        || a_mat.len() != n * n
        || lu.len() != n * n
        || ipiv.len() != n
    {
        return Err(MtxError::DimensionMismatch(
            "a_mat and lu must have length n * n and ipiv must have length n",
        ));
    }
    let mut anorm = 0.0_f64;
    for j in 0..n {
        let mut col_sum = 0.0;
        for i in 0..n {
            col_sum += a_mat[i * n + j].abs();
        }
        anorm = anorm.max(col_sum);
    }
    if anorm == 0.0 {
        return Ok(0.0);
    }

    let inv_n = 1.0 / (n as f64);
    let mut trial = vec![inv_n; n];
    let mut ainv_norm = 0.0_f64;
    for _iter in 0..5 {
        let mut sol = trial.clone();
        if dgetrs_trans(false, n, lu, ipiv, &mut sol).is_err() {
            return Ok(0.0);
        }
        let est: f64 = sol.iter().map(|v| v.abs()).sum();
        if est <= ainv_norm {
            break;
        }
        ainv_norm = est;
        let mut dual_vec: Vec<f64> = sol
            .iter()
            .map(|&v| if v >= 0.0 { 1.0 } else { -1.0 })
            .collect();
        if dgetrs_trans(true, n, lu, ipiv, &mut dual_vec).is_err() {
            return Ok(0.0);
        }
        let mut max_idx = 0;
        let mut max_z = dual_vec[0].abs();
        for (i, &zv) in dual_vec.iter().enumerate().skip(1) {
            if zv.abs() > max_z {
                max_z = zv.abs();
                max_idx = i;
            }
        }
        let z_dot_x: f64 = dual_vec
            .iter()
            .zip(trial.iter())
            .map(|(zi, xi)| zi * xi)
            .sum();
        if max_z <= z_dot_x {
            break;
        }
        trial.fill(0.0);
        trial[max_idx] = 1.0;
    }
    if ainv_norm == 0.0 || !ainv_norm.is_finite() {
        return Ok(0.0);
    }
    Ok(1.0 / (anorm * ainv_norm))
}

/// Solves $A(p) x(p) = b(p)$ with first-order forward-mode dual numbers
/// [`DualOrder1<K>`] via implicit differentiation:
/// $$A\,\frac{\partial x}{\partial p_m}
///   = \frac{\partial b}{\partial p_m}
///   - \frac{\partial A}{\partial p_m}\,x$$
///
/// # Errors
///
/// Propagates any error from [`dgetrf`] or [`dgetrs`].
pub fn solve_dense_dual<const K: usize>(
    n: usize,
    a: &[DualOrder1<K>],
    rhs: &[DualOrder1<K>],
) -> Result<Vec<DualOrder1<K>>, MtxError> {
    if a.len() != n * n || rhs.len() != n {
        return Err(MtxError::DimensionMismatch(
            "a must have length n * n and rhs must have length n",
        ));
    }
    let mut lu: Vec<f64> = a.iter().map(|v| v.val).collect();
    let mut ipiv = vec![0usize; n];
    dgetrf(n, &mut lu, &mut ipiv)?;

    let mut x_val: Vec<f64> = rhs.iter().map(|v| v.val).collect();
    dgetrs(n, &lu, &ipiv, &mut x_val)?;

    let mut result = vec![DualOrder1::constant(0.0); n];
    for i in 0..n {
        result[i].val = x_val[i];
    }

    let mut tang_rhs = vec![0.0; n];
    for m in 0..K {
        for i in 0..n {
            let mut da_x = 0.0;
            for j in 0..n {
                da_x += a[i * n + j].d1[m] * x_val[j];
            }
            tang_rhs[i] = rhs[i].d1[m] - da_x;
        }
        dgetrs(n, &lu, &ipiv, &mut tang_rhs)?;
        for i in 0..n {
            result[i].d1[m] = tang_rhs[i];
        }
    }
    Ok(result)
}
