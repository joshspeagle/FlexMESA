# GYRE G2 Real LAPACK/BLAS Reference Vectors (`#170`, Package G2)

This directory produces the vectors that `crates/flexmesa-gyre/tests/g2_vectors.rs` checks bit for bit, for the port of the real LAPACK/BLAS routines that GYRE 8.1 reaches in MESA 26.4.1 (`crates/flexmesa-gyre/src/lapack/`). It also establishes how the MESA SDK builds those routines.

- **Reference libraries:** the MESA SDK `x86_64-linux-26.6.1` `crmath` math-slot `liblapack`/`libblas`. MESA links these because `utils/makefile_header` sets `USE_CRMATH = YES` and exports `MESASDK_MATH_SLOT = crmath`.
- **Release:** MESA release `26.4.1` (`r26.4.1`, commit `c8c202662922a22b3efefa824473ad615c623afd`, Zenodo record `19722306`).

## Contents

| File | Role |
| :--- | :--- |
| `extract_tapped_inputs.py` | Extracts the real `XGETRF`, `XGESVX` and `DGESVD` inputs GYRE passed in the G0 oracle calls (`../gyre_oracles/data/*.taps.tsv`). |
| `gen_gyre_g2_vectors.f90`, `Makefile`, `stamp_header.py` | Replays those calls with GYRE's exact arguments and adds synthetic edge cases. The calls are `DGETRF(m, n, A, m, ...)`, `DGESVX('E', 'N', ..., work(4n), iwork(n))` and `DGESVD('A', 'A', n, n, ..., lwork = 5n)`. The synthetic cases cover the leaf routines `DNRM2`, `DLARTG`, `DLAS2`, `DLASV2`, `DLAPY2`, `DLARFG`, plus `DLAMCH` and `ILAENV`. Writes `crates/flexmesa-gyre/tests/data/mesa_26_4_1_gyre_g2_vectors.tsv` with a provenance header. |
| `check_sdk_objects.sh` | Disassembles each SDK object and compares it with Reference LAPACK 3.12.0 compiled by the SDK's gfortran. Writes `sdk_object_check.txt`. |

## Recipe

```bash
export MESASDK_ROOT=/path/to/mesasdk         # x86_64-linux-26.6.1
export MESA_DIR=/path/to/mesa-26.04.1
source "$MESASDK_ROOT/bin/mesasdk_init.sh"
cd benchmarks/fortran_mesa/gyre_g2_vectors
make OUT_DIR=/path/to/scratch generate
LAPACK_TARBALL=/path/to/lapack-v3.12.0.tar.gz WORK=/path/to/scratch2 bash check_sdk_objects.sh
cargo test --release -p flexmesa-gyre
```

The tarball is `https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.0.tar.gz`, SHA-256 `eac9570f8e0ad6f30ce4b963f4f033f0f643e7c3912fc9ee6cd99120675ad48b`.

## How the SDK Builds These Routines (`sdk_object_check.txt`)

- **Same object code as netlib `-O2 -fPIC`.** All 51 real routines of the G2 set except `dbdsqr` and `dlamch` are identical, instruction for instruction, to Reference LAPACK 3.12.0 compiled with `gfortran -O2 -fPIC` (GCC 15.2.0). The comparison masks addresses and relocation offsets. None of these objects references CRMATH, and they contain no FMA and no x87 instructions.
- **Complex routines (for G4 to G6).** Of the 20 complex routines and helpers checked, the complex BLAS routines and `zlaswp`, `zlacpy`, `zlarf`, `zgetrf`, `zgetrs`, `zung2r`, `zungqr`, `zgebd2` and `dladiv` match netlib `-O2 -fPIC`. Their complex arithmetic is therefore gfortran's inline expansion at those flags, the same as measured in package G1: Smith-style division with no `__divdc3` call. Three objects differ, and each one replaces a complex intrinsic with a CRMATH module procedure:
  - `zgetrf2`: `ABS` of a complex value becomes `abs_c_dp`.
  - `zlahqr`: `ABS` and `SQRT` of a complex value become `abs_c_dp` and `sqrt_c_dp`.
  - `zladiv`: `DCMPLX` becomes `dcmplx_r_r_dp`.

  The output file's last section lists all 267 SDK objects that reference CRMATH. That set includes `dlanv2`, which calls `log_r_dp` and is reached by `DGEEV`. The complex packages must port these objects against the CRMATH entry points. G1 already provides `abs_c` and `sqrt_c`.
- **`dlamch`** matches `INSTALL/dlamch.f` compiled at `-O0`. Its values are the IEEE constants: `E = 2**-53`, `S = U = 2**-1022`, `P = 2**-52`, `B = 2`, `O = HUGE`.
- **`dbdsqr`** is the SDK's CRMATH-patched build. Its only CRMATH symbols are `log_r_dp` and `exp_r_dp`, called for `TOLMUL = MAX(TEN, MIN(HNDRD, EPS**MEIGTH))`. In the disassembly, the multiplier of `log(EPS)` is a constant zero (a `pxor`-cleared register), so the SDK computes `exp(0*log(EPS)) = 1` and `TOLMUL = 10`; netlib computes `EPS**(-1/8) = 98.70`. The port reproduces this. Running the port with netlib's `TOLMUL` changes 87 of the 235 `DGESVD` records, so the vectors confirm the SDK value.
- **`iparmq`** calls CRMATH `log_r_sp` only for `ISPEC` 12 to 16 (`iparmq.f:271`). GYRE's calls reach `IPARMQ` only from `xHSEQR` in the eigenvalue chains (packages G5 and G6), and only with `ISPEC = INMIN`. That line executed 0 times in all 834 G0 calls. The real G2 set does not call `IPARMQ` at all.

## Evidence in the Vectors

- **Tapped GYRE calls.** The file holds 72 `DGETRF`, 8 `DGESVX` and 35 `DGESVD` records. Replaying them through the SDK library reproduces GYRE's own tapped outputs, including LU factors, solutions and singular values, in 115 of 115 cases.
- **Synthetic cases:**
  - `DGETRF`: rectangular shapes including `2n x n`, zero columns (`INFO > 0`), pivots below `SFMIN`, and rank deficiency.
  - `DGESVX`: `EQUED` values `N`, `R`, `C` and `B`; singular systems; `RCOND < EPS` (`INFO = N+1`).
  - `DGESVD`: rank-deficient, diagonal, repeated and scaled singular values.
- **Line coverage.** A coverage build of netlib 3.12.0 shows that the vectors execute every LAPACK/BLAS source line that GYRE executes in `astero_gyre`, `fast_from_file` and `gyre_in_mesa_wd`: 1,619, 1,694 and 1,119 lines, none missed. Those three cases are the real-only GYRE cases.
