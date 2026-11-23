# GYRE 8.1 Reach Measurement and Hex Oracles from MESA 26.4.1 (`#170`, Package G0)

This directory holds the tools that measure which parts of GYRE MESA 26.4.1 actually runs, and that capture exact reference values for the FlexMESA GYRE port.

- **Code under test:** GYRE 8.1 as bundled with MESA release `26.4.1` (`r26.4.1`, commit `c8c202662922a22b3efefa824473ad615c623afd`, Zenodo record `19722306`), in the configuration MESA builds it with (`gyre/make/makefile_full`: `FRONTENDS=no TOOLS=no IFACES=yes FORUM=no CRMATH=yes SHARED=no`).
- **Toolchain:** MESA SDK `x86_64-linux-26.6.1` (GCC 15.2.0). LAPACK and BLAS are netlib 3.12.0 from the SDK's `crmath` math slot.

The inputs are the `pulse_tap` captures in the `mesa_histories` data bundle (`runs/<suite>/<case>/pulse_tap/`). For each GYRE call that MESA made in the reference runs, a capture holds two things:

- the exact model passed to `set_model`, in `models/gyre_eval_*.gyre`;
- every mode GYRE returned, as IEEE-754 bits, in `modes.tsv`.

The model text is written in `1PE26.16`, which is 17 significant digits. A list-directed read therefore recovers the same bits as the in-memory arrays MESA passed. The `verify` results below confirm this for every captured call.

## Contents

| File | Role |
| :--- | :--- |
| `gyre_oracle_driver.f90` | Stand-alone driver. It mirrors MESA's calls into `gyre_mesa_m` (`init`, the `set_constant` calls, `set_model`, and one `get_modes` per degree). For every mode it writes the integers, `omega`, `freq` in `UHZ`/`HZ`/`RAD_PER_SEC`, and `E_norm`, all as hex. It can also write eigenfunction and model-coefficient samples. |
| `Makefile` | Builds the driver against a chosen GYRE build. The default is the pristine MESA build. |
| `gyre_tap_m.fypp`, `apply_taps.py`, `build_overlay.sh` | Build an overlay copy of GYRE under a scratch directory. The pristine MESA tree is only read. Two variants exist: `taps`, which keeps MESA's flags and adds read-only hex taps, and `coverage`, which builds with `-O0 --coverage`. |
| `build_lapack_coverage.sh` | Builds a line-coverage copy of netlib LAPACK/BLAS 3.12.0 (double and double-complex parts) from the pinned upstream tarball. It is used to measure routine and branch reach only. |
| `run_oracles.py` | `verify`, `coverage` and `oracles` subcommands (see below). |
| `coverage_report.py` | Turns gcov JSON into the reach tables in `results/`. |
| `results/` | Script-generated reach tables and verification summaries, each with a provenance header. |

## Recipe

```bash
export MESASDK_ROOT=/path/to/mesasdk          # x86_64-linux-26.6.1
export MESA_DIR=/path/to/mesa-26.04.1         # built with USE_GYRE = YES
export GYRE_ORACLE_WORK=/path/to/scratch      # outside the repository
export LAPACK_TARBALL=/path/to/lapack-v3.12.0.tar.gz
# https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.0.tar.gz
# sha256 eac9570f8e0ad6f30ce4b963f4f033f0f643e7c3912fc9ee6cd99120675ad48b
source "$MESASDK_ROOT/bin/mesasdk_init.sh"
cd benchmarks/fortran_mesa/gyre_oracles
W=$GYRE_ORACLE_WORK

# Drivers
make OUT=$W/bin/driver_pristine                                   # pristine MESA GYRE
bash build_overlay.sh taps
make OUT=$W/bin/driver_taps GYRE_BUILD=$W/taps/gyre/build
bash build_overlay.sh coverage
bash build_lapack_coverage.sh
make OUT=$W/bin/driver_cov GYRE_BUILD=$W/coverage/gyre/build COVERAGE=yes \
     EXTRA_LIBS=$W/lapack-cov/liblapack_cov.a

# 1. Every captured GYRE call, compared bit for bit with modes.tsv
python3 run_oracles.py verify --driver $W/bin/driver_pristine --out $W/verify_pristine
python3 run_oracles.py verify --driver $W/bin/driver_taps --out $W/verify_taps \
    --label 'tapped overlay, MESA flags'

# 2. Reach: union over all calls, then one pass per case (20 calls each)
python3 run_oracles.py coverage --driver $W/bin/driver_cov --out $W/coverage \
    --gyre-build $W/coverage/gyre/build --lapack-obj $W/lapack-cov/obj
python3 run_oracles.py coverage --per-case --max-evals 20 --driver $W/bin/driver_cov \
    --out $W/coverage/per_case --gyre-build $W/coverage/gyre/build \
    --lapack-obj $W/lapack-cov/obj
python3 coverage_report.py --cov $W/coverage --gyre-src $MESA_DIR/gyre/gyre/src --out results

# 3. Intermediate hex oracles for the representative calls (REPRESENTATIVE in run_oracles.py)
python3 run_oracles.py oracles --driver $W/bin/driver_taps --out $W/oracles
```

All runs use `OMP_NUM_THREADS=1`; set `GYRE_ORACLE_THREADS` to change this. GYRE's OpenMP loops in the reached code have no reductions. The reference runs used more threads, and the `verify` results below match them bit for bit.

## Output Formats

The oracle files from step 3 are committed in `data/` (one `modes.tsv` and one `taps.tsv` per representative call, plus `oracle_summary.tsv`, which records that every mode in them matches the reference).


Every oracle file starts with `#` header lines. These record the MESA version and commit, the GYRE version, the SDK, the compiler, the GYRE flags, the build variant, the case, the call number, the SHA-256 of the input model and of `gyre.in`, the degrees called, and the tap controls. All values are IEEE-754 binary64 bit patterns in hex. Complex values are written as `re im`.

**Driver files (`*.modes.tsv`)** contain these record types:

- `GET_MODES l`;
- `MODE idx l m n_p n_g n_pg n id`;
- `OMEGA`, `FREQ_UHZ`, `FREQ_HZ`, `FREQ_RAD_PER_SEC` (each `idx re im`) and `E_NORM idx`;
- `EIG idx k s x y_1..y_6 xi_r xi_h dW_dx` and `COEFF idx k s <20 model coefficients>` (`V_2, As, U, c_1, Gamma_1, ups_T, nabla_ad, nabla, beta_rad, c_lum, c_rad, c_thn, c_thk, c_eps, c_egv, eps_rho, eps_T, kap_rho, kap_T, Omega_rot`, with `undefined` where the model does not define a coefficient). These two are written every `eig_stride` points plus the last point.

**Tap files (`*.taps.tsv`)** have one record per line, `TAG <tab> name <tab> count <tab> values`. Records appear in execution order:

| Tag | Where (overlay source) | Content |
| :--- | :--- | :--- |
| `MODEPAR` | `lib/gyre_mesa_m.fypp`, start of each `&mode` group matching `l` | `l`, group index |
| `GRID`, `SCAN` | after `refine_grid` | refined grid `x` and segment `s`; scan frequencies `omega_re` |
| `BUILD` | `bvp/bvp_m.fypp` `build` | build index, `n`, flag for builds made to construct a mode (`wave_t_hom_` in `{rad,ad,nad}_bvp_m`); `omega` for dumped builds |
| `BOUND` | same, dumped builds | `B_i`, `scl_i`, `B_o`, `scl_o` |
| `BLOCK` | same, dumped builds | `E_l`, `E_r` (column-major), `scl` (`ext`: fraction, exponent) for evenly spaced intervals `j` |
| `DISCRIM` | `search/discrim_m.fypp` | `omega` and the determinant (`ext` fraction, exponent) of every evaluation: first the scan points in order, then the root-finder iterates |
| `MODE_AD`, `MODE_NAD` | `process_mode_ad`/`_nad` | `omega`; `l, n_p, n_g, n_pg, n` |
| `LAPACK_IN`, `LAPACK_OUT` | `matrix/linalg_m.fypp`, `matrix/block_sysmtx_m.fypp` | Inputs and raw outputs of the first `GYRE_TAP_LAPACK_MAX` calls of `XGETRF` (cyclic reduction and determinant), `DGEEV`/`ZGEEV`, `DGESVD`/`ZGESVD` and `DGESVX`/`ZGESVX` |

Dumped builds are the first `GYRE_TAP_MATRIX_BUILDS` builds of each `&mode` group (the first scan points) plus every mode-construction build.

## Measured Results (Generated in `results/`)

- **Verify (pristine build):** 834 captured GYRE calls across all nine GYRE-calling cases, 7,032 modes, 0 mismatches. `results/verify_pristine.tsv` holds the per-case counts.
- **Verify (tapped build):** the tapped overlay reproduces the same 7,032 modes bit for bit, so the taps do not change any value (`results/verify_taps.tsv`).
- **Reach:** see `results/reach_summary.md`, `results/gyre_reach_files.tsv`, `results/gyre_reach_procedures.tsv` and `results/lapack_reach_routines.tsv`. The last of these lists, for every reached routine, the executable source lines of netlib 3.12.0 that were never executed.
- **Sensitivity checks** (not oracles):
  - GYRE built at `-O0` with the SDK LAPACK reproduces every star case and `astero_gyre`. In `fast_from_file`, 56 calls differ in `E_norm` of `l = 1, 2` modes; the frequencies are equal.
  - At `-O0`, gfortran calls the library `MATMUL` (`_gfortran_matmul_r8`) for the small matrices in `ad_trans_m`. At `-O2` it inlines them.
  - Linking netlib LAPACK/BLAS source instead of the SDK's CRMATH-built library changes frequencies in every Magnus case and `E_norm` in `astero_gyre` (187 calls in total).
  - The SDK library is therefore the reference, and the port must follow its CRMATH calls.

## Limitations

- The driver also evaluates `xi_r`, `xi_h`, `dW_dx` and the model coefficients for every case so that their code is covered. In MESA only `rsp_gyre` uses eigenfunctions.
- The per-case passes sample 20 evenly spaced calls per case. The union pass covers every captured call.
- LAPACK branch reach is measured on the netlib source build. In the calls where that build gives different bits (see Sensitivity checks), the branches taken could in principle differ from the SDK library's. Routine-level reach matches the routines the SDK objects reference.
- `fast_newuoa` makes no GYRE calls (`chi2_seismo_fraction = 0`) and is not covered.
