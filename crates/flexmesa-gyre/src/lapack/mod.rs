// SPDX-License-Identifier: GPL-3.0-only
//! The real LAPACK/BLAS 3.12.0 subset GYRE 8.1 reaches through MESA 26.4.1
//! (package G2 of `#170`), reproducing the MESA SDK `x86_64-linux-26.6.1`
//! `crmath` math-slot libraries bit for bit.
//!
//! # Origin
//!
//! Each routine is a transcription of Reference LAPACK / Reference BLAS
//! 3.12.0 (<https://github.com/Reference-LAPACK/lapack>, tag `v3.12.0`,
//! modified BSD licence, Univ. of Tennessee, Univ. of California Berkeley,
//! Univ. of Colorado Denver and NAG Ltd.); each doc comment names the
//! source file. The SDK compiles these sources with `gfortran -O2 -fPIC`
//! (GCC 15.2.0): for every routine here except `dbdsqr` and `dlamch` the
//! SDK object is instruction-for-instruction identical to that build of the
//! netlib file (`benchmarks/fortran_mesa/gyre_g2_vectors/README.md`).
//! The two exceptions:
//!
//! - `dbdsqr`: the SDK's CRMATH-patched build evaluates
//!   `TOLMUL = MAX(TEN, MIN(HNDRD, EPS**MEIGTH))` as
//!   `exp(0 * log(EPS))`, i.e. `TOLMUL = 10` instead of netlib's
//!   `EPS**(-1/8) ~ 98.7` (read from the object's disassembly and confirmed
//!   by the vectors). See [`svd::dbdsqr`].
//! - `dlamch`: built at `-O0`; its values are the IEEE constants in
//!   [`dlamch`].
//!
//! # Scope
//!
//! The ported paths are those GYRE's calls reach (measured by line
//! coverage in package G0) plus the small branches next to them. Paths
//! that GYRE's calls never take and that would need further routines
//! (blocked algorithms for blocks larger than the `ILAENV` block size,
//! `DLASCL` rescaling of tiny or huge matrices, the careful-solve path of
//! `DLATRS`, the `DLASQ1` path of `DBDSQR`) return
//! [`LapackError::UnportedPath`] instead of computing something different.
//!
//! # Conventions
//!
//! Matrices are column-major slices with a leading dimension `ld`; element
//! `A(i,j)` (1-based, as in the Fortran) is `a[(i-1) + (j-1)*ld]`. A
//! sub-matrix argument `A(i,j)` is the slice starting at that element.
//! Character options are ASCII bytes compared case-insensitively
//! (`LSAME`).

// The routines mirror the Fortran one to one: argument lists, local names
// (`ll`/`lll`, `smin`/`sminl`), loop bounds written `a..n + 1` for
// `DO I = A, N`, deferred assignment of locals and routine length all follow
// the source, so these style lints are allowed for the LAPACK modules.
// Matrix sizes are bounded by GYRE's block size (at most a few tens), so
// the `usize` to `isize` index casts in `DBDSQR` cannot wrap.
macro_rules! lapack_mod {
    ($name:ident) => {
        #[allow(
            clippy::similar_names,
            clippy::too_many_arguments,
            clippy::too_many_lines,
            clippy::range_plus_one,
            clippy::needless_late_init,
            clippy::cast_possible_wrap
        )]
        pub mod $name;
    };
}
lapack_mod!(aux);
lapack_mod!(blas);
lapack_mod!(lu);
lapack_mod!(svd);

/// Errors from the LAPACK port.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LapackError {
    /// Upstream would call `XERBLA` (invalid argument number `info`).
    IllegalArgument {
        /// Routine name.
        routine: &'static str,
        /// Negative `INFO` upstream would report.
        info: i32,
    },
    /// A branch GYRE's calls never reach and that is not ported.
    UnportedPath {
        /// Routine name.
        routine: &'static str,
        /// Upstream source lines of the branch.
        lines: &'static str,
    },
}

impl std::fmt::Display for LapackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IllegalArgument { routine, info } => {
                write!(f, "{routine}: illegal argument (INFO = {info})")
            }
            Self::UnportedPath { routine, lines } => write!(
                f,
                "{routine}: branch at lines {lines} of the Reference LAPACK 3.12.0 \
                 source is not ported (never reached by GYRE's calls)"
            ),
        }
    }
}

impl std::error::Error for LapackError {}

/// Column-major index of 1-based `(i, j)` with leading dimension `ld`.
#[inline]
#[must_use]
pub const fn ix(i: usize, j: usize, ld: usize) -> usize {
    (i - 1) + (j - 1) * ld
}

/// `LSAME(ca, cb)`: case-insensitive ASCII comparison.
#[inline]
#[must_use]
pub fn lsame(ca: u8, cb: u8) -> bool {
    ca.eq_ignore_ascii_case(&cb)
}

/// `DLAMCH` constants (`INSTALL/dlamch.f`, binary64, round-to-nearest).
pub mod dlamch {
    /// `DLAMCH('E')`: relative machine epsilon `EPSILON(0d0)*0.5 = 2**-53`.
    pub const EPS: f64 = f64::from_bits(0x3CA0_0000_0000_0000);
    /// `DLAMCH('S')`: safe minimum `TINY(0d0) = 2**-1022`.
    pub const SFMIN: f64 = f64::from_bits(0x0010_0000_0000_0000);
    /// `DLAMCH('B')`: base 2.
    pub const BASE: f64 = 2.0;
    /// `DLAMCH('P')`: `EPS*BASE = 2**-52`.
    pub const PREC: f64 = f64::from_bits(0x3CB0_0000_0000_0000);
    /// `DLAMCH('O')`: `HUGE(0d0)`.
    pub const RMAX: f64 = f64::MAX;
    /// `DLAMCH('U')`: `TINY(0d0)`.
    pub const RMIN: f64 = f64::from_bits(0x0010_0000_0000_0000);
}

/// `ILAENV(1, name, ...)` block sizes for the routines ported here
/// (`SRC/ilaenv.f:269-516`).
#[must_use]
pub const fn ilaenv_nb(name: &str) -> i32 {
    match name.as_bytes() {
        b"DGETRF" => 64,
        b"DGEQRF" | b"DGELQF" | b"DGEBRD" | b"DORGQR" | b"DORGLQ" => 32,
        // ILAENV's default for names it does not list
        _ => 1,
    }
}

/// `ILAENV(6, 'DGESVD', ...)`: `INT(REAL(MIN(N1,N2))*1.6E0)` in `REAL(4)`.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub fn ilaenv_mnthr(n1: usize, n2: usize) -> usize {
    let p = n1.min(n2) as f32 * 1.6_f32;
    p as usize
}
