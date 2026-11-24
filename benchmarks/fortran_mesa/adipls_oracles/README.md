# ADIPLS Reach Measurement and Hex Oracles from MESA 26.4.1 (`#171`, Package A0)

This directory holds the tools that measure which parts of ADIPLS MESA 26.4.1 actually runs, that capture exact reference values for the FlexMESA ADIPLS port, and that measure how far a correctly rounded libm moves ADIPLS's results (`DECISIONS #60`).

- **Code under test:** ADIPLS (the 2010 `adipack.c` release with Bill Paxton's MESA changes) as bundled with MESA release `26.4.1` (`r26.4.1`, commit `c8c202662922a22b3efefa824473ad615c623afd`, Zenodo record `19722306`), and the MESA glue in `astero/private/adipls_support.f90` and `adipls_support_procs.f90`.
- **Build:** `adipls/make/makefile_adipls:78-86` compiles the library with `gfortran -std=legacy -w -c` and no other flag, so ADIPLS runs at `-O0` and calls the system libm (glibc) for `pow`, `exp` and `log`. There is no CRMATH in ADIPLS. Integer powers go to libgcc `__powidf2`.
- **Toolchain:** MESA SDK `x86_64-linux-26.6.1` (GCC 15.2.0); glibc 2.39 on `x86_64`.

## Inputs

The `mesa_histories` data bundle records, for every ADIPLS call in the reference runs, the model arrays passed to `adipls` (`pulse_tap/models/adipls_eval_*.amdl.bin`) and every mode MESA stored (`pulse_tap/modes.tsv`). It does not record the controls MESA passes through common blocks for each call (`sig1`, `sig2` and `dfsig` depend on the stellar radius, mass and `delta_nu_model`, and `iscan` on the number of targets), and it does not record the FGONG data from which MESA builds the AMDL arrays. The bundle's `.amdl.bin` files also hold uninitialised memory in rows 6 to 9 of `aa` (see Findings).

The cases are therefore re-run against an overlay of MESA whose astero library carries read-only taps at the glue (`astero_glue_tap.inc`, inserted by `apply_astero_taps.py`). A tapped run writes, per case, into `<case>/a0tap/`:

| File | Content |
| :--- | :--- |
| `calls.tsv` | One line per per-model `adipls` call: call id, model number, stored-model id, whether a model was stored just before the call, `nn`, the array file, and every value of the commons `cadr_param` (12 reals, hex), `cadi_param` (8 integers) and the four `coutpt` entries MESA sets (`set_adipls_controls`). |
| `call_NNNNNN.amdl.bin` | The `(x, aa, data)` passed to `adipls`, written whenever it differs bytewise from the previous call's. |
| `store_NNNNNN.fgong.bin` | `star_get_pulse_data` output (`glob`, `var`) for every stored model: the input of `fgong_amdl`. |
| `store_NNNNNN.amdl_fgong.bin` | `fgong_amdl` output (`data`, `aa(1:6,:)`). |
| `store_NNNNNN.amdl_store.bin` | `store_amdl` output (`data`, `x`, `aa(1:10,:)`). |
| `store_NNNNNN.amdl_redist.bin` | `srdist` output when MESA redistributes the mesh for that call (no test case does; see Findings). |
| `modes.tsv` | For every mode returned through `spcout_adi`: degree, order, `em`, cyclic frequency and inertia as MESA stores them, `obs_st(1:10, nobs_st)` and `csummm(1:50)`, all as hex. |

Binary files are little-endian: `int32` counts followed by `float64` arrays in Fortran (column-major) order. The layouts are documented at the top of `astero_glue_tap.inc`.

The overlay run also links the bundle's `pulse_tap` capture. Its `pulse_tap/modes.tsv` is byte-identical to the bundle's for every case, so the taps change nothing MESA computes.

## Contents

| File | Role |
| :--- | :--- |
| `adipls_oracle_driver.f90` | Stand-alone driver. It repeats MESA's sequence (`run_adipls` with `first_time`: `srdist` parameter read when the inlist sets `do_redistribute_mesh`, `setups_adi`, the parameter-reading `adipls` call; then per call: `adipls.stdout` routing, optional `srdist`, `set_adipls_controls`, `adipls`) and replaces `spcout_adi`, `modmod` and `resdif` as MESA's `adipls_support_procs.f90` defines them. It writes every mode (degree, order, `em`, frequency, inertia, `obs_st`, `csummm`) and optional eigenfunction samples as hex. |
| `Makefile` | Builds the driver against one `libadipls.a` (default: the pristine MESA build). |
| `astero_glue_tap.inc`, `apply_astero_taps.py`, `build_mesa_overlay.sh`, `run_mesa_captures.sh` | MESA overlay with the glue taps, and the case runner. |
| `apply_adipls_taps.py`, `adipls_tap.c`, `build_adipls_overlay.sh` | ADIPLS overlays: `taps` (MESA's flags plus read-only intermediate taps), `coverage` (`--coverage`), and `libm` (vendored crlibm and the libm interposer `adipls_libm_wrap.c`). |
| `run_oracles.py` | `verify`, `coverage`, `oracles` and `libm` subcommands. |
| `coverage_report.py` | Turns gcov JSON into the reach tables in `results/`. |
| `results/` | Script-generated tables, each with a provenance header. |
| `data/` | Intermediate hex oracles for the representative calls. |

## Recipe

```bash
export MESASDK_ROOT=/path/to/mesasdk          # x86_64-linux-26.6.1
export MESA_DIR=/path/to/mesa-26.04.1         # built, USE_ADIPLS = YES
export ADIPLS_ORACLE_WORK=/path/to/scratch    # outside the repository
source "$MESASDK_ROOT/bin/mesasdk_init.sh"    # puts the SDK gfortran and gcov first on PATH
# linking MESA test cases may also need LIBRARY_PATH to point at a libz
cd benchmarks/fortran_mesa/adipls_oracles
W=$ADIPLS_ORACLE_WORK

# 0. Captures: tapped MESA overlay, then the five ADIPLS-calling cases
bash build_mesa_overlay.sh
for c in astero_adipls surface_effects example_astero fast_scan_grid fast_simplex; do
    bash run_mesa_captures.sh $c                # writes $W/mesa_runs/<case>/a0tap
done

# 1. Drivers
make OUT=$W/bin/driver_pristine                                    # MESA's libadipls.a
bash build_adipls_overlay.sh taps
make OUT=$W/bin/driver_taps ADIPLS_LIB=$W/taps/adipls/make/libadipls.a \
     EXTRA_OBJS=$W/taps/adipls_tap.o
bash build_adipls_overlay.sh coverage
make OUT=$W/bin/driver_cov ADIPLS_LIB=$W/coverage/adipls/make/libadipls.a COVERAGE=yes
bash build_adipls_overlay.sh libm
make OUT=$W/bin/driver_libm \
     EXTRA_OBJS="$W/libm/adipls_libm_wrap.o $W/libm/libflexmesa_crlibm.a" \
     LDWRAP="-rdynamic -Wl,--wrap=pow,--wrap=exp,--wrap=log,--wrap=log10,--wrap=sin,--wrap=cos,--wrap=atan2 -ldl"

# 2. Every captured call, bit for bit (three replays; see run_oracles.py)
for r in chain isolated; do
    python3 run_oracles.py verify --driver $W/bin/driver_pristine --out $W/verify --replay $r
done
python3 run_oracles.py verify --driver $W/bin/driver_taps --out $W/verify_taps --label 'tapped overlay, MESA flags'
python3 run_oracles.py verify --driver $W/bin/driver_cov --out $W/verify_cov --label 'coverage build'
# sensitivity checks
python3 run_oracles.py verify --driver $W/bin/driver_pristine --out $W/verify_poison \
    --env ADIPLS_A0_POISON_ROWS=6,7,8,9 --label 'aa rows 6-9 set to NaN'
python3 run_oracles.py verify --driver $W/bin/driver_pristine --out $W/verify_nofma \
    --env GLIBC_TUNABLES=glibc.cpu.hwcaps=-AVX2,-FMA,-AVX512F --label 'glibc without FMA variants'

# 3. Reach: union over all calls, then one pass per case
python3 run_oracles.py coverage --driver $W/bin/driver_cov --out $W/coverage_run \
    --adipls-obj $W/coverage/adipls/make
python3 run_oracles.py coverage --per-case --driver $W/bin/driver_cov --out $W/coverage_run \
    --adipls-obj $W/coverage/adipls/make
python3 coverage_report.py --cov $W/coverage_run --adipls-src $MESA_DIR/adipls/adipack.c --out results

# 4. Intermediate hex oracles (REPRESENTATIVE in run_oracles.py)
python3 run_oracles.py oracles --driver $W/bin/driver_taps --out data

# 5. libm: glibc vs crlibm on ADIPLS's own arguments, and the effect on the modes
python3 run_oracles.py libm --driver $W/bin/driver_libm --out $W/libm
```

`run_oracles.py` reads the bundle from `~/.cache/flexmesa/mesa_histories` (`--histories`) and the captures from `$ADIPLS_ORACLE_WORK/mesa_runs` (`--captures`).

## Output Formats

Driver files (`*.modes.tsv`) contain, per call: `MODE call k el order em freq_uHz inertia`, `OBS call k obs_st(1:10)`, `CSUM call k csummm(1:50)`, optional `EIG call k n x y(1:8,n)` samples (every `eig_stride` points plus the last), and `CALL call ierr_param control_mismatches nn`.

Tap files (`*.taps.tsv`) have one record per line, `TAG <tab> count <tab> values`, in execution order:

| Tag | Where (overlay source) | Content |
| :--- | :--- | :--- |
| `ENTER` | `adipls.c.d.f`, label `10000` | start of a per-model call |
| `DET` | `sigint.n.d.f`, exit | `sig`, `det`, `ddsol`, `iord`, `icry`, `nfit` of every integration: the scan points of `sigscn` in order, then the iterates of `sigitr` |
| `MATCH` | `mchsol.n.d.f`, normal exit | `det`, `y(1:ii, nfit)` after matching |
| `ITC` | `sigitc.c.d.f`, exit | `sig`, `ddsig`, `ddsol`, `xfit1`, `nfit`, `iord`, `icry`, `isolcv`, `isigcv`, `istatus` |
| `KINER` | `kiner.n.d.f`, exit | `el`, `ynorm`, `ekin` |
| `VARFRQ` | `varfrq.n.d.f`, exit | `sig`, `sigv` |
| `GRAVPO` | `gravpo.n.d.f`, normal exit | `sig`, `dsig` |

Every oracle file starts with `#` header lines recording the MESA version and commit, the SDK, the compiler, ADIPLS's flags, the build variant, the case, the call, the SHA-256 of the input arrays and of `adipls.c.pruned.in`, and the call's controls.

## Measured Results

The five ADIPLS-calling cases of MESA 26.4.1 make 2,165 per-model ADIPLS calls and return 27,361 modes. `fast_newuoa` makes no ADIPLS call.

| Case | Calls | Modes | Degrees called |
| :--- | ---: | ---: | :--- |
| `astero_adipls` | 10 | 92 | 0 |
| `surface_effects` | 4 | 110 | 0, 1, 2, 3 |
| `example_astero` | 65 | 1,079 | 0, 1, 2 |
| `fast_scan_grid` | 1,107 | 16,301 | 0, 1, 2 |
| `fast_simplex` | 979 | 9,779 | 0 |

- **Captures:** the overlay MESA runs reproduce the bundle's `pulse_tap/modes.tsv` byte for byte in all five cases. Their AMDL arrays equal the bundle's in `x`, `data` and rows 1 to 5 and 10 of `aa`; rows 6 to 9 differ between the two runs (uninitialised memory, see Findings).
- **Verify (pristine `libadipls.a`):** every mode of every call matches bit for bit in degree, order, `em`, frequency, inertia, all ten `obs_st` entries and all fifty `csummm` entries, against both references. This holds when the calls are replayed in MESA's order in one process (`results/verify_pristine_chain.tsv`) and when each call runs in a fresh process (`results/verify_pristine_isolated.tsv`). Each ADIPLS call therefore depends only on the parameter file, the controls and the model it is given, not on earlier calls.
- **Verify (other builds):** the tapped overlay (`results/verify_taps.tsv`) and the coverage build (`results/verify_coverage_build.tsv`) give the same bits, so the taps and `--coverage` change no value.
- **Reach** (`results/reach_summary.md`, `results/adipls_reach_files.tsv`, `results/adipls_reach_procedures.tsv`): 2,437 of the 7,517 executable lines of the 78 library sources (18,989 source lines) are executed, in 51 procedures (two of them `ENTRY` duplicates) across 43 files. The plan's estimate of 8,000 to 10,000 lines counted whole files. The executed lines include 91 lines of `srdist` (parameter read only), 73 lines of file-name bookkeeping in `ofiles_blstio.f`, and 448 of the 2,094 lines of `adipls_`.
- **Binary32 literals** (`results/adipls_sp_literals_reached.tsv`): 38 default-kind real literals sit on executed lines, 32 of them not exact in binary32 (for example `el.le.1.e-6` in nine routines, `x.gt.0.95` in `adirhs`, `0.1` in `setbcs` and `res_adimod`, `1.e-10` in `varfrq`). No `float`, `sngl` or kind-less `real` conversion is executed.
- **Intermediate oracles** (`data/`, `data/oracle_summary.tsv`): five calls covering `l = 0` to `3`, the model with an added atmosphere (`astero_adipls`), the 1,638-point red-giant model (`surface_effects`) and a coarse 498-point model (`fast_scan_grid`). Each `*.modes.tsv` matches the glue-tap reference bit for bit.

### libm (`DECISIONS #60`)

`results/libm_sites.tsv` lists every libm call ADIPLS makes on the reached path, with the number of arguments on which glibc 2.39 and crlibm 1.0beta4 round-to-nearest differ:

| Function | Calling routine | Calls | glibc ≠ crlibm | Largest difference |
| :--- | :--- | ---: | ---: | ---: |
| `pow` | `varfrq` | 4,692,675 | 2,730 (5.8e-4) | 1 ulp |
| `pow` | `xlmult` | 87,106 | 2 (2.3e-5) | 1 ulp |

No other libm function is called: `exp`, `log`, `log10`, `sin`, `cos` and `atan2` have zero calls, and the `pow` sites in `adirhs`, `setbcs`, `shtint`, `sph` and `uhx` are not reached. Radial calls (`l = 0`, all of `astero_adipls` and `fast_simplex`) make no libm call at all.

Returning crlibm's value instead of glibc's at every call changes no output: all 27,361 modes keep their degree, order, frequency, inertia and all fifty `csummm` entries bit for bit (`results/libm_effect.tsv`). The two `varfrq` sites are `xx**el` and `xx**(-1.d0-el)` (`varfrq.n.d.f:139, 203`); their 1-ulp differences do not reach any stored result on these cases.

glibc's results depend on the CPU. With the FMA variants masked (`GLIBC_TUNABLES=glibc.cpu.hwcaps=-AVX2,-FMA,-AVX512F`), glibc's `pow` disagrees with crlibm on 3,134 of the same arguments instead of 2,732 (`results/libm_sites_glibc_no_fma.tsv`). A separate probe on two million random arguments found the two glibc variants disagreeing on 7.2e-4 of `pow`, 6.8e-4 of `exp` and 1.4e-5 of `log` results (1 ulp each). For these cases the ADIPLS outputs are still identical under either variant (`results/verify_glibc_no_fma.tsv`, `results/libm_effect_glibc_no_fma.tsv`).

The expected agreement level for a port that routes these calls through `flexmesa_math::intrinsic_pow` (`crlibm::pow_rn`) is therefore bit-exact on all 27,361 reference modes, measured with MESA's own arguments. The 1-ULP allowance of `DECISIONS #60` is not needed on these cases, though it remains possible on other models.

### Findings for the Port

1. **Uninitialised rows of `aa`.** `fgong_amdl` (`astero/private/adipls_support.f90:867-883`) allocates `aa(10, nn)` and sets rows 1 to 6. `store_amdl` (`:707-710`) copies rows 1 to 6 into `aa1(10, nn)` and then sets `aa(i, n) = aa1(i+1, n1)` for `i = 1..6` (`:751-769`), so row 6 receives `aa1(7, :)`, which was never set. Rows 7 to 9 of `aa` are never set either. ADIPLS copies row 6 in `res_adimod` but does not read rows 6 to 9 on any reached path: overwriting them with NaN before every call leaves all 27,361 modes unchanged (`results/verify_poison_aa_rows_6_9.tsv`). This is deferred issue `#479` (`DECISIONS #43`): the port does not perform the unset read, stores rows 6 to 9 as explicitly unset, and returns an explicit error on any path that reads them; no 26.4.1 case reaches such a read.
2. **No mesh redistribution on any reached path.** `astero_adipls`'s inlist sets `do_redistribute_mesh = .true.`, which makes MESA read `redistrb.c.pruned.in` at start-up (`run_adipls`, `setup_redist`, 91 lines of `srdist`). Its `run_star_extras.f90:245` passes `do_restribute_mesh = .false.` to `adipls_get_one_el_info`, so no stored model is redistributed and the numerical part of `srdist` and `rseta4` is not reached.
3. **Argument shadowing in the glue.** `store_model_for_adipls` (`adipls_support.f90:217-327`) takes only `add_atmosphere` and `do_redistribute_mesh`; `star_get_pulse_data` there receives `add_center_point` and `keep_surface_point` from `astero_def`, not the values the caller passed to `adipls_get_one_el_info` Deferred issue `#480` (read at the tag); the port reproduces it, and the captures record what MESA used.
4. **Calls are independent.** The isolated replay shows that the common-block and `save` state ADIPLS carries between calls does not change any result. The port can treat one call as a pure function of the parameter file, the controls and the model, after one parameter read.
5. **No FMA.** The pristine `libadipls.a` contains no FMA instructions (`objdump`), as expected at `-O0` without `-march`.

## Limitations

- Reach and the libm counts cover the five MESA 26.4.1 test cases only. Other inlists can reach other branches (`mdintg` other than 5, kernels, mesh redistribution, `l > 3`), and other models can turn a 1-ulp `pow` difference into a different result.
- The glibc results are from one host (glibc 2.39, AMD Zen 5 with FMA). The FMA-masked runs emulate a CPU without FMA through glibc's own dispatch; no other CPU or glibc version was run.
- The glue taps record what MESA passed. Porting `fgong_amdl`, `store_amdl` and the control computation (package A1) still needs its own bit-exact comparison against these captures; this package does not implement or check it.
- `star_get_pulse_data` (the FGONG producer on the star side) is captured but not checked against FlexMESA's `pulse_exporters`.
- The intermediate taps sample five calls. `GRAVPO` is never emitted because `gravpo` is not reached.
- The per-call captures (about 350 MB, mostly FGONG and AMDL arrays) are not committed; they are distributed separately from this repository and regenerated by steps 0 to 2 of the recipe.
