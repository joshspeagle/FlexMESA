// SPDX-License-Identifier: GPL-3.0-or-later
//! Reference LAPACK 3.12.0 band condition-number routines, transcribed in
//! LAPACK's operation order on column-major band storage `AB(LDAB,*)`
//! (`#676`, part 1).
//!
//! # Origin
//!
//! Reference LAPACK / Reference BLAS 3.12.0
//! (<https://github.com/Reference-LAPACK/lapack>, tag `v3.12.0`, modified BSD
//! licence). Every routine's doc comment gives the source file and line
//! ranges at that tag.
//!
//! # Contents
//!
//! - [`dlangb`](fn@dlangb) (`SRC/dlangb.f`): norm of a general band matrix.
//! - [`dgbcon`](fn@dgbcon) (`SRC/dgbcon.f`): reciprocal condition number from the band
//!   LU factors, calling [`dlatbs`](fn@dlatbs), [`drscl`](fn@drscl) and `DLACN2`
//!   ([`crate::lapack::aux::dlacn2`], already a one-to-one transcription of
//!   `SRC/dlacn2.f`, re-checked against it for this part).
//! - [`dlatbs`](fn@dlatbs) (`SRC/dlatbs.f`): scaled banded triangular solve.
//! - [`drscl`](fn@drscl) (`SRC/drscl.f`): `x := x / sa` without overflow.
//! - [`dtbsv`](fn@dtbsv) (`BLAS/SRC/dtbsv.f`): banded triangular solve, needed by
//!   `DLATBS`.
//!
//! `DGBTRF`/`DGBTRS` stay in [`crate::band`]; they are not duplicated here.
//!
//! # Conventions
//!
//! `AB(i,j)` (1-based, as in the Fortran) is `ab[ix(i, j, ldab)]`, and a
//! vector argument such as `AB(KD+1,J)` is the slice starting at that
//! element. `IPIV` is passed 1-based, as `DGBTRF` returns it. Where the
//! Fortran calls `XERBLA`, the routine returns
//! [`LapackError::IllegalArgument`](crate::lapack::LapackError::IllegalArgument) with the same negative `INFO`.

#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::manual_swap,
    clippy::needless_late_init,
    clippy::range_plus_one,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]

pub mod dgbcon;
pub mod dlangb;
pub mod dlatbs;
pub mod drscl;
pub mod dtbsv;

pub use dgbcon::dgbcon;
pub use dlangb::dlangb;
pub use dlatbs::dlatbs;
pub use drscl::drscl;
pub use dtbsv::dtbsv;
