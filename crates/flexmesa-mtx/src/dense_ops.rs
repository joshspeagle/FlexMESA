//! Dense matrix-vector and LU-factored matrix products (`mtx`).
//!
//! Provides left row-vector products (`multiply_xa`, `multiply_xa_plus_c`) and
//! products reconstructed directly from in-place LU factors
//! (`lu_factored_square_mv`, `lu_factored_square_mm`).
//!
//! # LU Reconstruction
//!
//! Given $P A = L U$, any matrix-vector product $y = A x$ can be evaluated
//! without storing an unfactored copy of $A$ by computing $z = U x$, then
//! $w = L z$, and applying the inverse row permutations $P^{-1} w$ in reverse
//! pivot order $k = n - 1 \dots 0$.

use crate::MtxError;

/// Evaluates $b = x^{\top} A$ (row-vector times matrix) matching upstream
/// `multiply_xa`.
///
/// # Errors
///
/// Returns [`MtxError::DimensionMismatch`] if slice lengths do not match `n`.
pub fn multiply_xa(
    n: usize,
    a_mat: &[f64],
    vec_x: &[f64],
    out_b: &mut [f64],
) -> Result<(), MtxError> {
    if a_mat.len() != n * n || vec_x.len() != n || out_b.len() != n {
        return Err(MtxError::DimensionMismatch(
            "a_mat must have length n * n and vec_x, out_b must have length n",
        ));
    }
    for j in 0..n {
        let mut sum = 0.0;
        for i in 0..n {
            sum += vec_x[i] * a_mat[i * n + j];
        }
        out_b[j] = sum;
    }
    Ok(())
}

/// Evaluates $b = x^{\top} A + c$ matching upstream `multiply_xa_plus_c`.
///
/// # Errors
///
/// Returns [`MtxError::DimensionMismatch`] if slice lengths do not match `n`.
pub fn multiply_xa_plus_c(
    n: usize,
    a_mat: &[f64],
    vec_x: &[f64],
    vec_c: &[f64],
    out_b: &mut [f64],
) -> Result<(), MtxError> {
    if vec_c.len() != n {
        return Err(MtxError::DimensionMismatch("vec_c must have length n"));
    }
    multiply_xa(n, a_mat, vec_x, out_b)?;
    for j in 0..n {
        out_b[j] += vec_c[j];
    }
    Ok(())
}

/// Evaluates $y = A x = P^{-1} L U x$ directly from the LU factors and pivot
/// vector produced by [`crate::dense::dgetrf`].
///
/// # Errors
///
/// Returns [`MtxError::DimensionMismatch`] if slice lengths do not match `n`.
pub fn lu_factored_square_mv(
    n: usize,
    lu: &[f64],
    ipiv: &[usize],
    vec_x: &[f64],
    out_y: &mut [f64],
) -> Result<(), MtxError> {
    if lu.len() != n * n
        || ipiv.len() != n
        || vec_x.len() != n
        || out_y.len() != n
    {
        return Err(MtxError::DimensionMismatch(
            "lu must have length n * n and ipiv, vec_x, out_y length n",
        ));
    }
    let mut u_x = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in i..n {
            sum += lu[i * n + j] * vec_x[j];
        }
        u_x[i] = sum;
    }
    for i in 0..n {
        let mut sum = u_x[i];
        for j in 0..i {
            sum += lu[i * n + j] * u_x[j];
        }
        out_y[i] = sum;
    }
    for (k, &piv) in ipiv.iter().enumerate().rev() {
        if piv >= n {
            return Err(MtxError::InvalidParameter(
                "pivot index out of bounds",
            ));
        }
        if piv != k {
            out_y.swap(k, piv);
        }
    }
    Ok(())
}

/// Evaluates $C = A B = P^{-1} L U B$ for $N \times N$ matrices directly from
/// the LU factors of $A$ produced by [`crate::dense::dgetrf`].
///
/// # Errors
///
/// Returns [`MtxError::DimensionMismatch`] if slice lengths do not match
/// `n * n`.
pub fn lu_factored_square_mm(
    n: usize,
    lu: &[f64],
    ipiv: &[usize],
    mat_b: &[f64],
    out_c: &mut [f64],
) -> Result<(), MtxError> {
    if lu.len() != n * n
        || ipiv.len() != n
        || mat_b.len() != n * n
        || out_c.len() != n * n
    {
        return Err(MtxError::DimensionMismatch(
            "lu, mat_b, out_c must have length n * n and ipiv length n",
        ));
    }
    let mut u_b = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in i..n {
                sum += lu[i * n + k] * mat_b[k * n + j];
            }
            u_b[i * n + j] = sum;
        }
    }
    for i in 0..n {
        for j in 0..n {
            let mut sum = u_b[i * n + j];
            for k in 0..i {
                sum += lu[i * n + k] * u_b[k * n + j];
            }
            out_c[i * n + j] = sum;
        }
    }
    for (k, &piv) in ipiv.iter().enumerate().rev() {
        if piv >= n {
            return Err(MtxError::InvalidParameter(
                "pivot index out of bounds",
            ));
        }
        if piv != k {
            for j in 0..n {
                out_c.swap(k * n + j, piv * n + j);
            }
        }
    }
    Ok(())
}
