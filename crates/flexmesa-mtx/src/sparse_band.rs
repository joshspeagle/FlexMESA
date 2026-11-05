//! Banded-to-sparse conversions and sparse entry location lookup (`mtx`).
//!
//! Provides conversions between banded storage `(ldab, n)` and 1-based CSR/CSC
//! formats, plus [`find_loc_in_sparse`] across both 1-based and 0-based sparse
//! conventions.
//!
//! # Band Storage Convention
//!
//! In a banded buffer `ab` of shape `(ldab, n)` (`ldab >= ml + mu + 1`),
//! matrix entry $A_{i,j}$ resides at `row = (ldab - 1 - ml) + i - j` and
//! column `j`.

use crate::{MtxError, SparseFormat};

/// Core band-to-compressed conversion parameterized by orientation
/// (`is_row_major`) and diagonal retention (`include_diag`).
///
/// Iterates over the outer index (`i` for CSR, `j` for CSC) and scans only
/// within `[max(0, outer - bw_low)..=min(n - 1, outer + bw_high)]`, writing
/// 1-based pointers and indices into `ptr` and `ind`.
/// Returns the total number of stored non-zero (and retained diagonal) entries.
fn band_to_compressed_impl(
    n: usize,
    ml: usize,
    mu: usize,
    ab: &[f64],
    ldab: usize,
    nzmax: usize,
    ptr: &mut [usize],
    ind: &mut [usize],
    values: &mut [f64],
    is_row_major: bool,
    include_diag: bool,
) -> Result<usize, MtxError> {
    if n == 0 || ml + mu + 1 > n || ldab < ml + mu + 1 {
        return Err(MtxError::InvalidParameter("invalid band dimensions"));
    }
    if ab.len() < ldab * n
        || ptr.len() < n + 1
        || ind.len() < nzmax
        || values.len() < nzmax
    {
        return Err(MtxError::DimensionMismatch("buffer too small"));
    }
    let diag_row = ldab - 1 - ml;
    let mut nz = 0usize;
    for outer in 0..n {
        ptr[outer] = nz + 1;
        let (in_min, in_max) = if is_row_major {
            (outer.saturating_sub(ml), (outer + mu).min(n - 1))
        } else {
            (outer.saturating_sub(mu), (outer + ml).min(n - 1))
        };
        for inner in in_min..=in_max {
            let (i, j) = if is_row_major {
                (outer, inner)
            } else {
                (inner, outer)
            };
            let val = ab[((diag_row + i) - j) * n + j];
            if val == 0.0 && (i != j || !include_diag) {
                continue;
            }
            if nz >= nzmax {
                return Err(MtxError::CapacityExceeded(outer + 1));
            }
            values[nz] = val;
            ind[nz] = inner + 1;
            nz += 1;
        }
    }
    ptr[n] = nz + 1;
    Ok(nz)
}

/// Core 1-based compressed-to-band conversion parameterized by orientation
/// (`is_row_major`).
///
/// Zero-fills `ab[..(ldab * n)]`, validates 1-based pointer and index bounds
/// plus band limits $i \le j + m_l$ and $j \le i + m_u$, and scatters each
/// sparse entry into `ab[((diag_row + i) - j) * n + j]`.
fn compressed_to_band_impl(
    n: usize,
    ml: usize,
    mu: usize,
    ab: &mut [f64],
    ldab: usize,
    nz: usize,
    ptr: &[usize],
    ind: &[usize],
    values: &[f64],
    is_row_major: bool,
) -> Result<(), MtxError> {
    if n == 0 || ml + mu + 1 > n || ldab < ml + mu + 1 {
        return Err(MtxError::InvalidParameter("invalid band dimensions"));
    }
    if ab.len() < ldab * n
        || ptr.len() < n + 1
        || ind.len() < nz
        || values.len() < nz
    {
        return Err(MtxError::DimensionMismatch("buffer too small"));
    }
    ab[..(ldab * n)].fill(0.0);
    let diag_row = ldab - 1 - ml;
    for outer in 0..n {
        let start = ptr[outer]
            .checked_sub(1)
            .ok_or(MtxError::InvalidParameter("ptr < 1"))?;
        let end = ptr[outer + 1]
            .checked_sub(1)
            .ok_or(MtxError::InvalidParameter("ptr < 1"))?;
        if end > nz || start > end {
            return Err(MtxError::InvalidParameter("ptr out of bounds"));
        }
        for k in start..end {
            let inner = ind[k]
                .checked_sub(1)
                .ok_or(MtxError::InvalidParameter("ind < 1"))?;
            let (i, j) = if is_row_major {
                (outer, inner)
            } else {
                (inner, outer)
            };
            if i >= n || j >= n || i > j + ml || j > i + mu {
                return Err(MtxError::InvalidParameter(
                    "entry outside band in sparse_to_band",
                ));
            }
            ab[((diag_row + i) - j) * n + j] = values[k];
        }
    }
    Ok(())
}

/// Converts a banded matrix `ab` into 1-based CSC format.
///
/// # Errors
///
/// Returns [`MtxError::CapacityExceeded`] if non-zeros exceed `nzmax`.
pub fn band_to_column_sparse(
    n: usize,
    ml: usize,
    mu: usize,
    ab: &[f64],
    ldab: usize,
    nzmax: usize,
    colptr: &mut [usize],
    rowind: &mut [usize],
    values: &mut [f64],
) -> Result<usize, MtxError> {
    band_to_compressed_impl(
        n, ml, mu, ab, ldab, nzmax, colptr, rowind, values, false, false,
    )
}

/// Converts a banded matrix `ab` into 1-based CSC format, retaining diagonals.
///
/// # Errors
///
/// Propagates any error from [`band_to_column_sparse`].
pub fn band_to_col_sparse_with_diag(
    n: usize,
    ml: usize,
    mu: usize,
    ab: &[f64],
    ldab: usize,
    nzmax: usize,
    colptr: &mut [usize],
    rowind: &mut [usize],
    values: &mut [f64],
) -> Result<usize, MtxError> {
    band_to_compressed_impl(
        n, ml, mu, ab, ldab, nzmax, colptr, rowind, values, false, true,
    )
}

/// Converts a 1-based CSC matrix into band storage `ab` of shape `(ldab, n)`.
///
/// # Errors
///
/// Returns [`MtxError::InvalidParameter`] if an entry lies outside the band.
pub fn column_sparse_to_band(
    n: usize,
    ml: usize,
    mu: usize,
    ab: &mut [f64],
    ldab: usize,
    nz: usize,
    colptr: &[usize],
    rowind: &[usize],
    values: &[f64],
) -> Result<(), MtxError> {
    compressed_to_band_impl(
        n, ml, mu, ab, ldab, nz, colptr, rowind, values, false,
    )
}

/// Converts a banded matrix `ab` into 1-based CSR format.
///
/// # Errors
///
/// Returns [`MtxError::CapacityExceeded`] if non-zeros exceed `nzmax`.
pub fn band_to_row_sparse(
    n: usize,
    ml: usize,
    mu: usize,
    ab: &[f64],
    ldab: usize,
    nzmax: usize,
    rowptr: &mut [usize],
    colind: &mut [usize],
    values: &mut [f64],
) -> Result<usize, MtxError> {
    band_to_compressed_impl(
        n, ml, mu, ab, ldab, nzmax, rowptr, colind, values, true, false,
    )
}

/// Converts a banded matrix `ab` into 1-based CSR format, retaining diagonals.
///
/// # Errors
///
/// Propagates any error from [`band_to_row_sparse`].
pub fn band_to_row_sparse_with_diag(
    n: usize,
    ml: usize,
    mu: usize,
    ab: &[f64],
    ldab: usize,
    nzmax: usize,
    rowptr: &mut [usize],
    colind: &mut [usize],
    values: &mut [f64],
) -> Result<usize, MtxError> {
    band_to_compressed_impl(
        n, ml, mu, ab, ldab, nzmax, rowptr, colind, values, true, true,
    )
}

/// Converts a 1-based CSR matrix into band storage `ab` of shape `(ldab, n)`.
///
/// # Errors
///
/// Returns [`MtxError::InvalidParameter`] if an entry lies outside the band.
pub fn row_sparse_to_band(
    n: usize,
    ml: usize,
    mu: usize,
    ab: &mut [f64],
    ldab: usize,
    nz: usize,
    rowptr: &[usize],
    colind: &[usize],
    values: &[f64],
) -> Result<(), MtxError> {
    compressed_to_band_impl(
        n, ml, mu, ab, ldab, nz, rowptr, colind, values, true,
    )
}

/// Finds the position of entry `(row_idx, col_idx)` in a sparse matrix (`ia`,
/// `ja`) using the index convention of `format` (1-based for
/// `CompressedColumnSparse` / `CompressedRowSparse`, 0-based for
/// `CompressedColSparse0Based` / `CompressedRowSparse0Based`).
///
/// # Errors
///
/// Returns [`MtxError::DimensionMismatch`] or [`MtxError::InvalidParameter`] if
/// pointer arrays or indices are out of bounds.
pub fn find_loc_in_sparse(
    format: SparseFormat,
    n: usize,
    nz: usize,
    ia: &[usize],
    ja: &[usize],
    row_idx: usize,
    col_idx: usize,
    hint: Option<usize>,
) -> Result<Option<usize>, MtxError> {
    if ia.len() < n + 1 || ja.len() < nz {
        return Err(MtxError::DimensionMismatch(
            "ia or ja too short for find_loc_in_sparse",
        ));
    }
    let (base, outer, inner) = match format {
        SparseFormat::CompressedColumnSparse => (1usize, col_idx, row_idx),
        SparseFormat::CompressedRowSparse => (1usize, row_idx, col_idx),
        SparseFormat::CompressedColSparse0Based => (0usize, col_idx, row_idx),
        SparseFormat::CompressedRowSparse0Based => (0usize, row_idx, col_idx),
    };
    let outer_0 = outer
        .checked_sub(base)
        .ok_or(MtxError::InvalidParameter("row/col index < base"))?;
    let inner_0 = inner
        .checked_sub(base)
        .ok_or(MtxError::InvalidParameter("row/col index < base"))?;
    if outer_0 >= n || inner_0 >= n {
        return Err(MtxError::InvalidParameter("row/col index out of bounds"));
    }
    let start = ia[outer_0]
        .checked_sub(base)
        .ok_or(MtxError::InvalidParameter("ia pointer < base"))?;
    let end = ia[outer_0 + 1]
        .checked_sub(base)
        .ok_or(MtxError::InvalidParameter("ia pointer < base"))?;
    if end > nz || start > end {
        return Err(MtxError::InvalidParameter("ia pointer out of bounds"));
    }
    if let Some(h) = hint {
        let next_0 = (h + 1).saturating_sub(base);
        if next_0 >= start && next_0 < end && ja[next_0] == inner {
            return Ok(Some(next_0 + base));
        }
    }
    for (offset, &val) in ja[start..end].iter().enumerate() {
        if val == inner {
            return Ok(Some(start + offset + base));
        }
    }
    Ok(None)
}
