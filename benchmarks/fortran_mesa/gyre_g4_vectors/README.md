# GYRE G4 Complex LAPACK/BLAS Reference Vectors (`#170`, Package G4)

This directory produces the vectors that `crates/flexmesa-gyre/tests/g4_zlapack_vectors.rs` checks bit for bit, for the port of the complex LAPACK/BLAS routines that GYRE 8.1's nonadiabatic path reaches in MESA 26.4.1 (`crates/flexmesa-gyre/src/lapack/{zblas,zaux,zlu,zsvd}.rs`).

- **Reference libraries:** the MESA SDK `x86_64-linux-26.6.1` `crmath` math-slot `liblapack`/`libblas`, which MESA links because `utils/makefile_header` sets `USE_CRMATH = YES` and exports `MESASDK_MATH_SLOT = crmath`.
- **Release:** MESA release `26.4.1` (`r26.4.1`, commit `c8c202662922a22b3efefa824473ad615c623afd`, Zenodo record `19722306`).

## Contents

| File | Role |
| :--- | :--- |
| `extract_tapped_inputs.py` | Extracts from the G0 oracle taps (`../gyre_oracles/data/*.taps.tsv`) the complex `ZGESVD` inputs GYRE passed, GYRE's own `ZGESVD` outputs, and the complex blocks `E_l`, `E_r` of each `BUILD`. |
| `gen_gyre_g4_vectors.f90`, `Makefile` | Replays the tapped calls with GYRE's argument patterns and adds synthetic edge cases. |
| `stamp_header.py` | Runs the generator, checks that the replayed `ZGESVD` calls reproduce GYRE's tapped outputs bit for bit (and fails otherwise), and writes `crates/flexmesa-gyre/tests/data/mesa_26_4_1_gyre_g4_vectors.tsv` with a provenance header. |

## Recipe

```bash
export MESASDK_ROOT=/path/to/mesasdk         # x86_64-linux-26.6.1
export MESA_DIR=/path/to/mesa-26.04.1
source "$MESASDK_ROOT/bin/mesasdk_init.sh"
cd benchmarks/fortran_mesa/gyre_g4_vectors
make OUT_DIR=/path/to/scratch generate
cargo test --release -p flexmesa-gyre --test g4_zlapack_vectors
```

## What the Vectors Contain

- **No duplicate of the taps.** The data file does not repeat what the committed G0 taps hold. `stamp_header.py` drops the `gesvd_tap` records once the replay check passes; the test compares the port directly with the tapped `ZGESVD` inputs and outputs. The tapped inputs of `getrf_det_tap`, `factor_tap` and `gemv_tap` are likewise read from the taps by record id, numbered as `extract_tapped_inputs.py` numbers them.
- **Tapped GYRE calls.** The G0 taps hold 23 complex `ZGESVD` calls (`sing_decompose_c_`: `ZGESVD('A', 'A', n, n, ..., LWORK = 3n, RWORK(5n))`, `n = 2 n_e = 12`). Replaying them through the SDK library reproduces GYRE's own tapped `S`, `U`, `VH` and `INFO` in 23 of 23 cases (`stamp_header.py` checks this on every generation).
- **GYRE's LU kernels on GYRE's data.** The taps contain no complex `XGETRF` records: the tap budget of those names went to real calls. The complex LU inputs therefore come from GYRE's data in two ways. The `ZGESVD` boundary matrices are also factored with `ZGETRF(2n_e, 2n_e)`, which is what `det_c_` does with the same matrix (`getrf_det_tap`, 23 records). The first two complex blocks of each tapped `BUILD` go through GYRE's cyclic-reduction step (`factor_c_`: `ZGETRF(2n_e, n_e)`, then `ZLASWP`, `ZTRSM('L','L','N','U')`, `ZGEMM('N','N', ..., -ONE, ..., ONE, ...)` and `ZTRSM('L','U','N','N')` on `M_E` and `M_G`; `factor_tap`, 30 records). The first block pair of each `BUILD` goes through the back-substitution `ZGEMV` pair (`backsub_c_`, `beta = ZERO` then `ONE`; `gemv_tap`, 30 records).
- **Synthetic cases**, each selector branch at least once (the leaf routines 34 cases, LU 39, SVD 14, `ZBDSQR` 18):
  - `ZGETRF`: shapes up to `12 x 12`, including GYRE's `2n x n` and `2n x 2n`, plus zero columns (`INFO > 0`), pivots below `SFMIN` (the division loop of `ZGETRF2`), rank deficiency, real-valued and imaginary matrices, and one rank-deficient `12 x 10` matrix with tiny pivots, the case that separates the recursive order of `ZGETRF2` from the classical one.
  - `ZGESVD` (path 10, `n <= m < 1.6 n`): rank-deficient, diagonal, repeated, scaled and real-valued matrices, a zero column and the zero matrix.
  - `ZBDSQR` called directly with `UPLO = 'U'` and `'L'`, including `ncvt`, `nru` other than `n`, split and zero diagonals.
  - The leaf routines `DZNRM2`, `IZAMAX`, `DLAPY3`, `DLADIV`, `ZLADIV`, `ZLARFG` (including the tiny-`beta` rescaling loop, also with several passes) and `ZLANGE` (`'M'`, `'O'`, `'1'`, `'I'`).
  - `ZGEMV` (`'N'`, `'T'`, `'C'`, increments 1 and 2), `ZGEMM` (`'N','N'`) and left-side `ZTRSM` (every `UPLO`, `TRANSA`, `DIAG`), with `alpha` and `beta` equal to 0, 1 and general values.

## How the SDK Builds These Routines

The object comparison of package G2 (`../gyre_g2_vectors/sdk_object_check.txt`) and the CRMATH symbol references of the SDK objects establish the following.

- **Same object code as netlib `-O2 -fPIC`:** every routine of the set except the six below. None of them references CRMATH or a libm function, so their complex arithmetic is gfortran's inline expansion (`crates/flexmesa-gyre/src/fcomplex.rs`).
- **CRMATH-patched (netlib plus `USE CRMATH`):**
  - `zgetrf2` and `zlange` call `abs_c_dp` for the complex `ABS` (`hypot`, not libm `cabs`).
  - `zdscal`, `zladiv` and `zlarfg` call `dcmplx_r_r_dp`/`dcmplx_r_dp` for `DCMPLX`, a plain constructor.
  - `zbdsqr` calls `log_r_dp` and `exp_r_dp` for `TOLMUL = MAX(TEN, MIN(HNDRD, EPS**MEIGTH))`. As in `dbdsqr`, the SDK object evaluates this as `exp(0*log(EPS))`, so `TOLMUL = 10` rather than netlib's `EPS**(-1/8)`.

The test's negative controls show that the vectors detect three alternatives: netlib's `TOLMUL` in `ZBDSQR`, Smith's division in place of `ZLADIV`, and the classical right-looking unblocked LU order in place of the recursive `ZGETRF2`.
