// SPDX-License-Identifier: GPL-3.0-or-later
//! Reference LAPACK / Reference BLAS 3.12.0 pure real scalar, auxiliary, BLAS,
//! and LU subset (`aux`, `blas`, `lu`), matching the MESA SDK
//! `x86_64-linux-26.6.1` `crmath` math-slot libraries bit for bit.
//!
//! # Origin
//!
//! Each routine is a transcription of Reference LAPACK / Reference BLAS
//! 3.12.0 (<https://github.com/Reference-LAPACK/lapack>, tag `v3.12.0`,
//! modified BSD licence, Univ. of Tennessee, Univ. of California Berkeley,
//! Univ. of Colorado Denver and NAG Ltd.); each doc comment names the
//! source file. The SDK compiles these sources with `gfortran -O2 -fPIC`
//! (GCC 15.2.0): for every routine here except `dlamch`, the SDK object is
//! instruction-for-instruction identical to that build of the netlib file
//! (`benchmarks/fortran_mesa/gyre_g2_vectors/README.md`). `dlamch` is built
//! at `-O0`; its values are the IEEE constants in [`dlamch`].
//!
//! # Scope
//!
//! Contains the pure Reference LAPACK/BLAS real auxiliary (`aux`), BLAS
//! (`blas`), and LU (`lu`: `dgetrf2`, `dgetrf`, `dgetrs`, `dgecon`, `dgerfs`,
//! `dgesvx`) routines shared across `flexmesa-mtx`, `flexmesa-gyre`, and
//! `flexmesa-turb`. Paths that need further unported routines (blocked `DGETRF`
//! when `min(m, n) > 64`, `DLASCL`/`DRSCL` rescaling, or the careful-solve path
//! of `DLATRS`) return [`LapackError::UnportedPath`] instead of computing
//! something different.
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
macro_rules! lapack_mod {
    ($name:ident) => {
        #[allow(
            clippy::similar_names,
            clippy::too_many_arguments,
            clippy::too_many_lines,
            clippy::range_plus_one,
            clippy::needless_late_init,
            clippy::cast_possible_wrap,
            clippy::manual_midpoint,
            clippy::nonminimal_bool,
            clippy::semicolon_if_nothing_returned
        )]
        pub mod $name;
    };
}
lapack_mod!(aux);
lapack_mod!(blas);
lapack_mod!(lu);

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
    /// A branch upstream would take that is not ported.
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
