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
//! The real auxiliary, BLAS, and LU subset (`aux`, `blas`, `lu`,
//! `LapackError`, `ix`, `lsame`, `dlamch`, `ilaenv_nb`, `ilaenv_mnthr`) is
//! hosted in [`flexmesa_mtx::lapack`] and re-exported here; the remaining
//! modules (`svd`, `eig`, `z*`) stay in `flexmesa-gyre`. The ported paths are
//! those GYRE's calls reach (measured by line coverage in package G0) plus the
//! small branches next to them. Paths that GYRE's calls never take and that
//! would need further routines (blocked algorithms for blocks larger than the
//! `ILAENV` block size, `DLASCL` rescaling of tiny or huge matrices, the
//! careful-solve path of `DLATRS`, the `DLASQ1` path of `DBDSQR`) return
//! [`LapackError::UnportedPath`] instead of computing something different.
//!
//! # Conventions
//!
//! Matrices are column-major slices with a leading dimension `ld`; element
//! `A(i,j)` (1-based, as in the Fortran) is `a[(i-1) + (j-1)*ld]`. A
//! sub-matrix argument `A(i,j)` is the slice starting at that element.
//! Character options are ASCII bytes compared case-insensitively
//! (`LSAME`).

pub use flexmesa_mtx::lapack::{
    aux, blas, dlamch, ilaenv_mnthr, ilaenv_nb, ix, lsame, lu, LapackError,
};

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
lapack_mod!(svd);
lapack_mod!(zaux);
lapack_mod!(zblas);
lapack_mod!(zlu);
lapack_mod!(zsvd);
lapack_mod!(eig);
lapack_mod!(zeig);
