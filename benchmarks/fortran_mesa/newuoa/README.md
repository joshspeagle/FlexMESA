# MESA 26.4.1 `NEWUOA` Reference Oracles (package G2)

Reference outputs for the transcription of Powell's `NEWUOA` in `crates/flexmesa-num/src/newuoa.rs` (upstream `num/public/num_newuoa.dek:22-34` and `num/private/mod_newuoa.f:7-1581`). They were produced by MESA release `26.4.1` (`r26.4.1`) built with MESA SDK `x86_64-linux-26.6.1` (`gfortran 15.2.0`, `CRMATH`); every file records the version string, compiler and SDK in its header.

## Files

| File | Content |
| :--- | :--- |
| `newuoa_oracle.f90` | Driver adapted from `num/test/src/test_newuoa.f90`. Calls `num_lib`'s `newuoa`; its `CALFUN` writes every `X(1:N)` and `F` as IEEE-754 binary64 hexadecimal words, and the returned `X` follows. |
| `bigden_oracle.f90` | Calls MESA's compiled `BIGDEN` (`mod_newuoa.f:703-1056`) directly on deterministic inputs and writes the inputs and every output array. |
| `Makefile` | Build, reference runs and coverage. |
| `mesa_26_4_1_newuoa_<run>.hex` | One full run each (below). |
| `mesa_26_4_1_newuoa_bigden.hex` | Two `BIGDEN` cases. |
| `mesa_26_4_1_newuoa_gcov.txt` | Lines of each routine executed, per run. |

## Runs

| Run | Problem | `N` | `NPT` | `RHOBEG` | `RHOEND` | `MAXFUN` | `max_valid_value` | `CALFUN` calls |
| :--- | :--- | ---: | ---: | :--- | :--- | ---: | :--- | ---: |
| `chebyquad_n2` | Chebyquad, `test_newuoa.f90` | 2 | 5 | `0.2*X(1)` | `1d-6` | 5000 | `1d99` | 31 |
| `chebyquad_n4` | Chebyquad, `test_newuoa.f90` | 4 | 9 | `0.2*X(1)` | `1d-6` | 5000 | `1d99` | 91 |
| `chebyquad_n6` | Chebyquad, `test_newuoa.f90` | 6 | 13 | `0.2*X(1)` | `1d-6` | 5000 | `1d99` | 188 |
| `chebyquad_n4_npt6` | Chebyquad with `NPT < 2N+1` (`mod_newuoa.f:208-212`) | 4 | 6 | `0.2*X(1)` | `1d-6` | 5000 | `1d99` | 77 |
| `quad_rhoend1` | Quadratic, astero `fast_newuoa` arguments | 3 | 7 | `0.45` | `1.0` | 1000 | `1d6` | 9 |
| `quad_rhoend1em6` | Same with the default `newuoa_rhoend` | 3 | 7 | `0.45` | `1d-6` | 1000 | `1d6` | 36 |

The `fast_newuoa`-shaped runs take their arguments from `astero/private/astero_run_support.f90:663-691` (`npt = 2*nvar+1`, `rhobeg = 0.45d0`, `maxfun = 1000`, `max_value = 1d6`, `X = 0`) with the three varied parameters of `astero/test_suite/fast_newuoa/inlist_astero_search_controls` and its `newuoa_rhoend = 1.0`. The objective is the closed-form quadratic `quad_f` in `newuoa_oracle.f90`, not a stellar-model chi-square.

## Recipe

```bash
export MESASDK_ROOT=/path/to/mesasdk           # SDK x86_64-linux-26.6.1
source $MESASDK_ROOT/bin/mesasdk_init.sh
make -C benchmarks/fortran_mesa/newuoa MESA_DIR=/path/to/mesa-26.4.1 BUILD=/scratch/newuoa runs
make -C benchmarks/fortran_mesa/newuoa MESA_DIR=/path/to/mesa-26.4.1 BUILD=/scratch/newuoa coverage
```

The Makefile includes MESA's `utils/makefile_header` for the compiler flags and libraries and links `$(MESA_DIR)/lib/libnum.a`. `bigden_oracle.f90` reads `$(MESA_DIR)/num/make/mod_newuoa.mod`. On some hosts the link also needs `LIBRARY_PATH` pointing at the system libraries the SDK expects. Nothing under `MESA_DIR` is written.

## Coverage

`make coverage` compiles a private copy of `mod_newuoa.f` with the flags `num/make/makefile_base` uses for it (`COMPILE_XTRA`) plus `--coverage`, links it ahead of `libnum.a`, reruns every case, checks that its output is byte-identical to the reference files, and records per-run `gcov` line counts. `do_newuoa`, `NEWUOB`, `TRSAPP` and `UPDATE` run in every full run and `BIGLAG` in every full run except `quad_rhoend1`. No full run reaches `BIGDEN` (a search over further objectives did not find one), so `BIGDEN` is evidenced by the direct-call cases only (`bigden 100.00% of 223` lines).

The replay test is `crates/flexmesa-num/tests/newuoa_oracle_test.rs`.
