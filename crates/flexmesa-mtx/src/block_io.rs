//! Harwell-Boeing text serialization and parsing for block-tridiagonal systems.
//!
//! Matches upstream `read_block_tridiagonal` and `write_block_tridiagonal` in
//! `mtx/private/mtx_support.f90`.
//!
//! # Format Specification
//!
//! - Line 1: `<nvar> <nz>` specifying the block dimension $N_{\mathrm{var}}$
//!   and shell count $N_z$.
//! - Subsequent blocks: for each shell $k \in 0..n_z$, the sub-diagonal block
//!   $L_k$ ($k > 0$), diagonal block $D_k$, and super-diagonal block $U_k$
//!   ($k + 1 < n_z$) are written in 1-based compressed-column Harwell-Boeing
//!   `RUA` format with diagonal entries retained.
//! - Each Harwell-Boeing block consists of a blank separator line, a 3-line
//!   header (`totcrd/ptrcrd/indcrd/valcrd`, `RUA nrow ncol nnz 0`, format
//!   specifiers), followed by `colptr` (`10I8`), `rowind` (`12I6`), and
//!   `values` (`5E27.16`, accepting Fortran `D`/`d` exponents on input).

use crate::sparse::{
    column_sparse_to_dense, dense_to_col_sparse_with_diag,
};
use crate::MtxError;
use std::fmt::Write as _;

/// Block-tridiagonal system parsed from Harwell-Boeing text format:
/// `(nvar, nz, lblk, dblk, ublk)`.
///
/// The three returned `Vec<f64>` buffers have length `nz * nvar * nvar` in
/// row-major `(nz, nvar, nvar)` block order.
pub type BlockTridiagData = (usize, usize, Vec<f64>, Vec<f64>, Vec<f64>);

/// Parses a block-tridiagonal system serialized in upstream MESA Harwell-Boeing
/// text format (`read_block_tridiagonal`), returning
/// `(nvar, nz, lblk, dblk, ublk)`.
///
/// # Arguments
///
/// - `text`: Full ASCII text containing the `<nvar> <nz>` header and
///   $3 N_z - 2$ Harwell-Boeing `RUA` blocks.
///
/// # Errors
///
/// Returns [`MtxError::ParseError`] if the header or any block is malformed,
/// or propagates [`MtxError::InvalidParameter`] if sparse indices are out of
/// bounds.
pub fn parse_block_tridiagonal_text(
    text: &str,
) -> Result<BlockTridiagData, MtxError> {
    let mut lines = text.lines();
    let header = lines
        .find(|l| !l.trim().is_empty())
        .ok_or(MtxError::ParseError("missing nvar/nblk header line"))?;
    let mut header_toks = header.split_whitespace();
    let nvar: usize = header_toks
        .next()
        .ok_or(MtxError::ParseError("missing nvar"))?
        .parse()
        .map_err(|_| MtxError::ParseError("invalid nvar"))?;
    let nz: usize = header_toks
        .next()
        .ok_or(MtxError::ParseError("missing nz"))?
        .parse()
        .map_err(|_| MtxError::ParseError("invalid nz"))?;
    if nvar == 0 || nz == 0 {
        return Err(MtxError::ParseError("nvar and nz must be >= 1"));
    }

    let blk_sz = nvar * nvar;
    let mut lblk = vec![0.0; nz * blk_sz];
    let mut dblk = vec![0.0; nz * blk_sz];
    let mut ublk = vec![0.0; nz * blk_sz];

    for k in 0..nz {
        if k > 0 {
            parse_one_hb_block(
                &mut lines,
                nvar,
                &mut lblk[k * blk_sz..(k + 1) * blk_sz],
            )?;
        }
        parse_one_hb_block(
            &mut lines,
            nvar,
            &mut dblk[k * blk_sz..(k + 1) * blk_sz],
        )?;
        if k + 1 < nz {
            parse_one_hb_block(
                &mut lines,
                nvar,
                &mut ublk[k * blk_sz..(k + 1) * blk_sz],
            )?;
        }
    }
    Ok((nvar, nz, lblk, dblk, ublk))
}

/// Parses a single $N_{\mathrm{var}} \times N_{\mathrm{var}}$ Harwell-Boeing
/// `RUA` block from `line_iter` into row-major `blk`.
///
/// Reads the 4-line Harwell-Boeing block header, extracts `(nrow, ncol, nnz)`,
/// reads `ncol + 1` column pointers, `nnz` row indices, and `nnz` values, and
/// converts to row-major dense layout via [`column_sparse_to_dense`].
fn parse_one_hb_block<'a, I>(
    line_iter: &mut I,
    nvar: usize,
    blk: &mut [f64],
) -> Result<(), MtxError>
where
    I: Iterator<Item = &'a str>,
{
    let _hdr1 = line_iter
        .next()
        .ok_or(MtxError::ParseError("unexpected EOF at HB line 1"))?;
    let _hdr2 = line_iter
        .next()
        .ok_or(MtxError::ParseError("unexpected EOF at HB line 2"))?;
    let hdr3 = line_iter
        .next()
        .ok_or(MtxError::ParseError("unexpected EOF at HB line 3"))?;
    let _hdr4 = line_iter
        .next()
        .ok_or(MtxError::ParseError("unexpected EOF at HB line 4"))?;

    let mut l3 = hdr3.split_whitespace();
    let _mxtype = l3
        .next()
        .ok_or(MtxError::ParseError("missing MXTYPE"))?;
    let nrow: usize = l3
        .next()
        .ok_or(MtxError::ParseError("missing nrow"))?
        .parse()
        .map_err(|_| MtxError::ParseError("invalid nrow"))?;
    let ncol: usize = l3
        .next()
        .ok_or(MtxError::ParseError("missing ncol"))?
        .parse()
        .map_err(|_| MtxError::ParseError("invalid ncol"))?;
    let nnz: usize = l3
        .next()
        .ok_or(MtxError::ParseError("missing nnz"))?
        .parse()
        .map_err(|_| MtxError::ParseError("invalid nnz"))?;
    if nrow != nvar || ncol != nvar {
        return Err(MtxError::ParseError("HB block dimension != nvar"));
    }

    let mut colptr = Vec::with_capacity(ncol + 1);
    while colptr.len() < ncol + 1 {
        let row_str = line_iter
            .next()
            .ok_or(MtxError::ParseError("unexpected EOF in colptr"))?;
        for tok in row_str.split_whitespace() {
            colptr.push(
                tok.parse::<usize>()
                    .map_err(|_| MtxError::ParseError("invalid colptr int"))?,
            );
        }
    }
    let mut rowind = Vec::with_capacity(nnz);
    while rowind.len() < nnz {
        let row_str = line_iter
            .next()
            .ok_or(MtxError::ParseError("unexpected EOF in rowind"))?;
        for tok in row_str.split_whitespace() {
            rowind.push(
                tok.parse::<usize>()
                    .map_err(|_| MtxError::ParseError("invalid rowind int"))?,
            );
        }
    }
    let mut values = Vec::with_capacity(nnz);
    while values.len() < nnz {
        let row_str = line_iter
            .next()
            .ok_or(MtxError::ParseError("unexpected EOF in values"))?;
        for tok in row_str.split_whitespace() {
            let clean = tok.replace('D', "E").replace('d', "e");
            values.push(
                clean
                    .parse::<f64>()
                    .map_err(|_| MtxError::ParseError("invalid HB float"))?,
            );
        }
    }
    column_sparse_to_dense(nvar, nvar, blk, nnz, &colptr, &rowind, &values)
}

/// Serializes a block-tridiagonal system in upstream Harwell-Boeing text format
/// (`write_block_tridiagonal`).
///
/// Writes the `<nvar> <nz>` header line followed by the $3 N_z - 2$
/// Harwell-Boeing `RUA` blocks ($D_0, U_0$, then $L_k, D_k, U_k$ for interior
/// shells, and $L_{n_z-1}, D_{n_z-1}$).
///
/// # Arguments
///
/// - `nvar`: Block dimension $N_{\mathrm{var}} \ge 1$.
/// - `nz`: Shell count $N_z \ge 1$.
/// - `lblk`, `dblk`, `ublk`: Contiguous `(nz, nvar, nvar)` block slices.
///
/// # Errors
///
/// Propagates any dimension or sparse conversion error.
pub fn write_block_tridiagonal_text(
    nvar: usize,
    nz: usize,
    lblk: &[f64],
    dblk: &[f64],
    ublk: &[f64],
) -> Result<String, MtxError> {
    let blk_sz = nvar * nvar;
    if nvar == 0
        || nz == 0
        || lblk.len() != nz * blk_sz
        || dblk.len() != nz * blk_sz
        || ublk.len() != nz * blk_sz
    {
        return Err(MtxError::DimensionMismatch(
            "invalid slice lengths in write_block_tridiagonal_text",
        ));
    }
    let mut out = String::new();
    let _ = writeln!(out, "{nvar} {nz}");
    for k in 0..nz {
        if k > 0 {
            write_one_hb_block(
                &mut out,
                nvar,
                &lblk[k * blk_sz..(k + 1) * blk_sz],
            )?;
        }
        write_one_hb_block(
            &mut out,
            nvar,
            &dblk[k * blk_sz..(k + 1) * blk_sz],
        )?;
        if k + 1 < nz {
            write_one_hb_block(
                &mut out,
                nvar,
                &ublk[k * blk_sz..(k + 1) * blk_sz],
            )?;
        }
    }
    Ok(out)
}

/// Serializes a single $N_{\mathrm{var}} \times N_{\mathrm{var}}$ block `blk`
/// into `out` in Harwell-Boeing `RUA` format with retained diagonal entries.
fn write_one_hb_block(
    out: &mut String,
    nvar: usize,
    blk: &[f64],
) -> Result<(), MtxError> {
    let nzmax = nvar * nvar;
    let mut colptr = vec![0usize; nvar + 1];
    let mut rowind = vec![0usize; nzmax];
    let mut values = vec![0.0; nzmax];
    let nnz = dense_to_col_sparse_with_diag(
        nvar,
        nvar,
        blk,
        nzmax,
        &mut colptr,
        &mut rowind,
        &mut values,
    )?;
    let ptrcrd = (nvar + 1) / 10 + 1;
    let indcrd = nnz / 12 + 1;
    let valcrd = nnz / 5 + 1;
    let totcrd = 3 + ptrcrd + indcrd + valcrd;
    let zero_val = 0usize;
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "{totcrd:14}{ptrcrd:14}{indcrd:14}{valcrd:14}{zero_val:14}"
    );
    let _ = writeln!(
        out,
        "RUA           {nvar:14}{nvar:14}{nnz:14}{zero_val:14}"
    );
    let _ = writeln!(out, "(10I8)          (12I6)          (5E27.16)");
    for chunk in colptr.chunks(10) {
        for &c in chunk {
            let _ = write!(out, "{c:8}");
        }
        let _ = writeln!(out);
    }
    for chunk in rowind[..nnz].chunks(12) {
        for &r in chunk {
            let _ = write!(out, "{r:6}");
        }
        let _ = writeln!(out);
    }
    for chunk in values[..nnz].chunks(5) {
        for &v in chunk {
            let _ = write!(out, "{v:27.16E}");
        }
        let _ = writeln!(out);
    }
    Ok(())
}
