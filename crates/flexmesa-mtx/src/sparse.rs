//! Dense-to-sparse (CSR/CSC, 1-based and 0-based) conversions (`mtx`).
//!
//! Matches upstream `mtx_formats.dek` and `mtx_support.f90` while fixing the
//! upstream 0-based index bounds bugs in `row_sparse_0_based_to_dense`.
//!
//! # Storage Conventions
//!
//! - **CSR (Compressed Sparse Row)**: `rowptr` has length `n + 1`, `colind`
//!   and `values` have length `nz <= nzmax`. Row `i` spans
//!   `(rowptr[i] - base)..(rowptr[i + 1] - base)`.
//! - **CSC (Compressed Sparse Column)**: `colptr` has length `n + 1`, `rowind`
//!   and `values` have length `nz <= nzmax`. Column `j` spans
//!   `(colptr[j] - base)..(colptr[j + 1] - base)`.

use crate::{MtxError, SparseFormat};

/// Core dense-to-compressed conversion parameterized by orientation
/// (`is_row_major`), diagonal retention (`include_diag`), and index `base`
/// (`0` or `1`).
fn dense_to_compressed_impl(
    n: usize,
    ndim: usize,
    a: &[f64],
    nzmax: usize,
    ptr: &mut [usize],
    ind: &mut [usize],
    values: &mut [f64],
    is_row_major: bool,
    include_diag: bool,
    base: usize,
) -> Result<usize, MtxError> {
    if n == 0 || ndim < n {
        return Err(MtxError::InvalidParameter("n must be >= 1 and ndim >= n"));
    }
    if a.len() < n * ndim
        || ptr.len() < n + 1
        || ind.len() < nzmax
        || values.len() < nzmax
    {
        return Err(MtxError::DimensionMismatch(
            "buffer too small for dense_to_sparse conversion",
        ));
    }
    let mut nz = 0usize;
    for outer in 0..n {
        ptr[outer] = nz + base;
        for inner in 0..n {
            let (i, j) = if is_row_major {
                (outer, inner)
            } else {
                (inner, outer)
            };
            let val = a[i * ndim + j];
            if val == 0.0 && (i != j || !include_diag) {
                continue;
            }
            if nz >= nzmax {
                return Err(MtxError::CapacityExceeded(outer + 1));
            }
            values[nz] = val;
            ind[nz] = inner + base;
            nz += 1;
        }
    }
    ptr[n] = nz + base;
    Ok(nz)
}

/// Core compressed-to-dense conversion parameterized by orientation
/// (`is_row_major`) and index `base` (`0` or `1`).
fn compressed_to_dense_impl(
    n: usize,
    ndim: usize,
    a: &mut [f64],
    nz: usize,
    ptr: &[usize],
    ind: &[usize],
    values: &[f64],
    is_row_major: bool,
    base: usize,
) -> Result<(), MtxError> {
    if n == 0 || ndim < n {
        return Err(MtxError::InvalidParameter("n must be >= 1 and ndim >= n"));
    }
    if a.len() < n * ndim
        || ptr.len() < n + 1
        || ind.len() < nz
        || values.len() < nz
    {
        return Err(MtxError::DimensionMismatch(
            "buffer too small for sparse_to_dense conversion",
        ));
    }
    a[..(n * ndim)].fill(0.0);
    for outer in 0..n {
        let start = ptr[outer]
            .checked_sub(base)
            .ok_or(MtxError::InvalidParameter("ptr index < base"))?;
        let end = ptr[outer + 1]
            .checked_sub(base)
            .ok_or(MtxError::InvalidParameter("ptr index < base"))?;
        if end > nz || start > end {
            return Err(MtxError::InvalidParameter("ptr range out of bounds"));
        }
        for k in start..end {
            let inner = ind[k]
                .checked_sub(base)
                .ok_or(MtxError::InvalidParameter("index < base"))?;
            if inner >= n {
                return Err(MtxError::InvalidParameter("index >= n"));
            }
            let (i, j) = if is_row_major {
                (outer, inner)
            } else {
                (inner, outer)
            };
            a[i * ndim + j] = values[k];
        }
    }
    Ok(())
}

/// Converts a row-major dense matrix `a` into 1-based CSR format.
///
/// # Errors
///
/// Returns [`MtxError::CapacityExceeded`] if non-zeros exceed `nzmax`.
pub fn dense_to_row_sparse(
    n: usize,
    ndim: usize,
    a: &[f64],
    nzmax: usize,
    rowptr: &mut [usize],
    colind: &mut [usize],
    values: &mut [f64],
) -> Result<usize, MtxError> {
    dense_to_compressed_impl(
        n, ndim, a, nzmax, rowptr, colind, values, true, false, 1,
    )
}

/// Converts a dense matrix `a` into 1-based CSR format, retaining diagonals.
///
/// # Errors
///
/// Propagates any error from [`dense_to_row_sparse`].
pub fn dense_to_row_sparse_with_diag(
    n: usize,
    ndim: usize,
    a: &[f64],
    nzmax: usize,
    rowptr: &mut [usize],
    colind: &mut [usize],
    values: &mut [f64],
) -> Result<usize, MtxError> {
    dense_to_compressed_impl(
        n, ndim, a, nzmax, rowptr, colind, values, true, true, 1,
    )
}

/// Converts a dense matrix `a` into 0-based CSR format.
///
/// # Errors
///
/// Propagates any error from [`dense_to_row_sparse`].
pub fn dense_to_row_0_based(
    n: usize,
    ndim: usize,
    a: &[f64],
    nzmax: usize,
    rowptr: &mut [usize],
    colind: &mut [usize],
    values: &mut [f64],
) -> Result<usize, MtxError> {
    dense_to_compressed_impl(
        n, ndim, a, nzmax, rowptr, colind, values, true, false, 0,
    )
}

/// Converts a dense matrix `a` into 0-based CSR format, retaining diagonals.
///
/// # Errors
///
/// Propagates any error from [`dense_to_row_sparse`].
pub fn dense_to_row_with_diag_0_based(
    n: usize,
    ndim: usize,
    a: &[f64],
    nzmax: usize,
    rowptr: &mut [usize],
    colind: &mut [usize],
    values: &mut [f64],
) -> Result<usize, MtxError> {
    dense_to_compressed_impl(
        n, ndim, a, nzmax, rowptr, colind, values, true, true, 0,
    )
}

/// Converts a 1-based CSR matrix into a row-major dense matrix `a`.
///
/// # Errors
///
/// Returns [`MtxError::InvalidParameter`] or [`MtxError::DimensionMismatch`].
pub fn row_sparse_to_dense(
    n: usize,
    ndim: usize,
    a: &mut [f64],
    nz: usize,
    rowptr: &[usize],
    colind: &[usize],
    values: &[f64],
) -> Result<(), MtxError> {
    compressed_to_dense_impl(n, ndim, a, nz, rowptr, colind, values, true, 1)
}

/// Converts a 0-based CSR matrix into a row-major dense matrix `a`.
///
/// # Errors
///
/// Returns [`MtxError::InvalidParameter`] or [`MtxError::DimensionMismatch`].
pub fn row_sparse_0_based_to_dense(
    n: usize,
    ndim: usize,
    a: &mut [f64],
    nz: usize,
    rowptr: &[usize],
    colind: &[usize],
    values: &[f64],
) -> Result<(), MtxError> {
    compressed_to_dense_impl(n, ndim, a, nz, rowptr, colind, values, true, 0)
}

/// Converts a dense matrix `a` into 1-based CSC format.
///
/// # Errors
///
/// Returns [`MtxError::CapacityExceeded`] if non-zeros exceed `nzmax`.
pub fn dense_to_col_sparse(
    n: usize,
    ndim: usize,
    a: &[f64],
    nzmax: usize,
    colptr: &mut [usize],
    rowind: &mut [usize],
    values: &mut [f64],
) -> Result<usize, MtxError> {
    dense_to_compressed_impl(
        n, ndim, a, nzmax, colptr, rowind, values, false, false, 1,
    )
}

/// Converts a dense matrix `a` into 1-based CSC format, retaining diagonals.
///
/// # Errors
///
/// Propagates any error from [`dense_to_col_sparse`].
pub fn dense_to_col_sparse_with_diag(
    n: usize,
    ndim: usize,
    a: &[f64],
    nzmax: usize,
    colptr: &mut [usize],
    rowind: &mut [usize],
    values: &mut [f64],
) -> Result<usize, MtxError> {
    dense_to_compressed_impl(
        n, ndim, a, nzmax, colptr, rowind, values, false, true, 1,
    )
}

/// Converts a dense matrix `a` into 0-based CSC format.
///
/// # Errors
///
/// Propagates any error from [`dense_to_col_sparse`].
pub fn dense_to_col_0_based(
    n: usize,
    ndim: usize,
    a: &[f64],
    nzmax: usize,
    colptr: &mut [usize],
    rowind: &mut [usize],
    values: &mut [f64],
) -> Result<usize, MtxError> {
    dense_to_compressed_impl(
        n, ndim, a, nzmax, colptr, rowind, values, false, false, 0,
    )
}

/// Converts a dense matrix `a` into 0-based CSC format, retaining diagonals.
///
/// # Errors
///
/// Propagates any error from [`dense_to_col_sparse`].
pub fn dense_to_col_with_diag_0_based(
    n: usize,
    ndim: usize,
    a: &[f64],
    nzmax: usize,
    colptr: &mut [usize],
    rowind: &mut [usize],
    values: &mut [f64],
) -> Result<usize, MtxError> {
    dense_to_compressed_impl(
        n, ndim, a, nzmax, colptr, rowind, values, false, true, 0,
    )
}

/// Dispatches dense-to-sparse conversion with diagonal retention across
/// [`SparseFormat`].
///
/// # Errors
///
/// Propagates any error from the underlying CSR/CSC conversion.
pub fn dense_to_sparse_with_diag(
    format: SparseFormat,
    n: usize,
    ndim: usize,
    a: &[f64],
    nzmax: usize,
    ptr: &mut [usize],
    ind: &mut [usize],
    values: &mut [f64],
) -> Result<usize, MtxError> {
    match format {
        SparseFormat::CompressedRowSparse => {
            dense_to_row_sparse_with_diag(n, ndim, a, nzmax, ptr, ind, values)
        }
        SparseFormat::CompressedColumnSparse => {
            dense_to_col_sparse_with_diag(n, ndim, a, nzmax, ptr, ind, values)
        }
        SparseFormat::CompressedRowSparse0Based => {
            dense_to_row_with_diag_0_based(n, ndim, a, nzmax, ptr, ind, values)
        }
        SparseFormat::CompressedColSparse0Based => {
            dense_to_col_with_diag_0_based(n, ndim, a, nzmax, ptr, ind, values)
        }
    }
}

/// Converts a 1-based CSC matrix into a row-major dense matrix `a`.
///
/// # Errors
///
/// Returns [`MtxError::InvalidParameter`] or [`MtxError::DimensionMismatch`].
pub fn column_sparse_to_dense(
    n: usize,
    ndim: usize,
    a: &mut [f64],
    nz: usize,
    colptr: &[usize],
    rowind: &[usize],
    values: &[f64],
) -> Result<(), MtxError> {
    compressed_to_dense_impl(n, ndim, a, nz, colptr, rowind, values, false, 1)
}

/// Converts a 0-based CSC matrix into a row-major dense matrix `a`.
///
/// # Errors
///
/// Returns [`MtxError::InvalidParameter`] or [`MtxError::DimensionMismatch`].
pub fn col_sparse_0_based_to_dense(
    n: usize,
    ndim: usize,
    a: &mut [f64],
    nz: usize,
    colptr: &[usize],
    rowind: &[usize],
    values: &[f64],
) -> Result<(), MtxError> {
    compressed_to_dense_impl(n, ndim, a, nz, colptr, rowind, values, false, 0)
}
