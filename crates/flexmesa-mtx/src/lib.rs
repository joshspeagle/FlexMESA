//! Dense, banded, sparse, and block-tridiagonal linear algebra (`mtx`).
//!
//! Provides IEEE-754 `f64` LU factorization with partial pivoting for dense
//! (`dgetrf`, `dgetrs`), banded (`dgbtrf`, `dgbtrs`), and block-tridiagonal
//! (`block_thomas_factor`, `block_thomas_solve`) linear systems, forward-mode
//! implicit differentiation for parameter-dependent linear systems
//! (`solve_dense_dual`), and bidirectional conversions across dense, banded,
//! compressed-row sparse (CSR), and compressed-column sparse (CSC) formats.

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::if_not_else,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]

pub mod band;
pub mod block_io;
pub mod block_tridiag;
pub mod dense;
mod dense_ops;
pub mod sparse;
mod sparse_band;

pub use band::{
    band_multiply_xa, band_to_dense, dense_to_band, dgbtrf, dgbtrs,
    rcond_banded, solve_banded,
};
pub use block_io::{
    parse_block_tridiagonal_text, write_block_tridiagonal_text,
    BlockTridiagData,
};
pub use block_tridiag::{
    block_dble_mv, block_multiply_xa, block_thomas_factor,
    block_thomas_solve, clip_block, clip_blocks, lu_factored_block_dble_mv,
    solve_block_tridiagonal,
};
pub use dense::{
    dgetrf, dgetrs, dgetrs_multi, dgetrs_trans, rcond_dense, solve_dense,
    solve_dense_dual, solve_dense_multi,
};
pub use dense_ops::{
    lu_factored_square_mm, lu_factored_square_mv, multiply_xa,
    multiply_xa_plus_c,
};
pub use sparse::{
    col_sparse_0_based_to_dense, column_sparse_to_dense, dense_to_col_0_based,
    dense_to_col_sparse, dense_to_col_sparse_with_diag,
    dense_to_col_with_diag_0_based, dense_to_row_0_based,
    dense_to_row_sparse, dense_to_row_sparse_with_diag,
    dense_to_row_with_diag_0_based, dense_to_sparse_with_diag,
    row_sparse_0_based_to_dense, row_sparse_to_dense,
};
pub use sparse_band::{
    band_to_col_sparse_with_diag, band_to_column_sparse, band_to_row_sparse,
    band_to_row_sparse_with_diag, column_sparse_to_band, find_loc_in_sparse,
    row_sparse_to_band,
};

use std::fmt;

/// Linear solver selection options matching upstream `mtx_def.f90`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum DecsolOption {
    /// Dense or banded LAPACK-compatible LU solver (`lapack = 1`).
    Lapack = 1,
    /// Double-precision block Thomas solver (`block_thomas_dble = 2`).
    BlockThomasDble = 2,
    /// Block Thomas solver with iterative refinement
    /// (`block_thomas_refine = 4`).
    BlockThomasRefine = 4,
    /// Parallel cyclic reduction block-tridiagonal solver (`bcyclic_dble = 5`).
    BcyclicDble = 5,
}

/// Sparse matrix storage convention matching upstream `mtx_def.f90`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum SparseFormat {
    /// 1-based compressed-column sparse format
    /// (`compressed_column_sparse = 0`).
    CompressedColumnSparse = 0,
    /// 1-based compressed-row sparse format (`compressed_row_sparse = 1`).
    CompressedRowSparse = 1,
    /// 0-based compressed-column sparse format
    /// (`compressed_col_sparse_0_based = 2`).
    CompressedColSparse0Based = 2,
    /// 0-based compressed-row sparse format
    /// (`compressed_row_sparse_0_based = 3`).
    CompressedRowSparse0Based = 3,
}

/// Matrix type codes matching upstream `mtx_def.f90`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum MatrixType {
    /// Full square dense matrix (`square_matrix_type = 1`).
    Square = 1,
    /// Banded matrix (`banded_matrix_type = 2`).
    Banded = 2,
    /// Double-precision block-tridiagonal matrix
    /// (`block_tridiag_dble_matrix_type = 3`).
    BlockTridiagDble = 3,
}

/// Parses a solver option name into a [`DecsolOption`] variant.
///
/// # Errors
///
/// Returns [`MtxError::InvalidParameter`] if `name` is not recognized.
pub fn decsol_option(name: &str) -> Result<DecsolOption, MtxError> {
    match name.trim() {
        "lapack" => Ok(DecsolOption::Lapack),
        "block_thomas_dble" => Ok(DecsolOption::BlockThomasDble),
        "block_thomas_refine" => Ok(DecsolOption::BlockThomasRefine),
        "bcyclic_dble" => Ok(DecsolOption::BcyclicDble),
        _ => Err(MtxError::InvalidParameter(
            "unrecognized decsol option name",
        )),
    }
}

/// Returns the canonical solver option string for `option`.
#[must_use]
pub const fn decsol_option_str(option: DecsolOption) -> &'static str {
    match option {
        DecsolOption::Lapack => "lapack",
        DecsolOption::BlockThomasDble => "block_thomas_dble",
        DecsolOption::BlockThomasRefine => "block_thomas_refine",
        DecsolOption::BcyclicDble => "bcyclic_dble",
    }
}

/// Returns `true` if `option` operates on block-tridiagonal matrices.
#[must_use]
pub const fn is_block_tridiagonal_decsol(option: DecsolOption) -> bool {
    matches!(
        option,
        DecsolOption::BlockThomasDble
            | DecsolOption::BlockThomasRefine
            | DecsolOption::BcyclicDble
    )
}

/// Error conditions returned by `flexmesa-mtx` linear algebra routines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MtxError {
    /// Input slices do not match the declared matrix or vector dimensions.
    DimensionMismatch(&'static str),
    /// Matrix bandwidth, index convention, or option parameter is invalid.
    InvalidParameter(&'static str),
    /// An input matrix or vector contains `NaN` or infinite entries.
    NonFiniteInput(&'static str),
    /// A zero pivot was encountered at the given 0-based diagonal index.
    SingularMatrix(usize),
    /// Sparse output buffer capacity `nzmax` was exceeded at the given 1-based
    /// row or column index.
    CapacityExceeded(usize),
    /// Harwell-Boeing or block-tridiagonal text parsing failed.
    ParseError(&'static str),
}

impl fmt::Display for MtxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch(msg) => {
                write!(f, "dimension mismatch: {msg}")
            }
            Self::InvalidParameter(msg) => {
                write!(f, "invalid parameter: {msg}")
            }
            Self::NonFiniteInput(msg) => write!(f, "non-finite input: {msg}"),
            Self::SingularMatrix(idx) => {
                write!(f, "singular matrix at diagonal index {idx}")
            }
            Self::CapacityExceeded(idx) => {
                write!(f, "sparse capacity exceeded at index {idx}")
            }
            Self::ParseError(msg) => write!(f, "parse error: {msg}"),
        }
    }
}

impl std::error::Error for MtxError {}
