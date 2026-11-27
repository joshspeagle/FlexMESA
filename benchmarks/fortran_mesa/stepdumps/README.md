# MESA 26.4.1 step dumps

Per-iteration records of MESA's star solver, written by `write` statements that
a patch adds to a private copy of `star/private/star_solver.f90`. FlexMESA's
own trace writes the same records, and
`flexmesa trace <suite> <case> --compare <dump>` reports the first record at
which the two differ (record type, model, zone, variable, ULP distance). The
patch adds output only: no expression, branch or order of arithmetic of the
solver changes.

| File | Role |
| :--- | :--- |
| `stepdump_patch.py` | The patch text and the record layout (see its docstring). |
| `run_stepdumps.py` | Builds the instrumented overlay of a compiled MESA tree and runs cases with the dump on. |
| `windows.toml` | Mid-run windows worth keeping (case, stage, occurrence, first model, steps, reason). |
| `regenerate_windows.py` | Runs every window of `windows.toml` through `run_stepdumps.py`, one at a time, resumably. |
| `first_divergence_table.py` | First-divergence table over a directory of dumps. |
| `xtra_patch.py` | Opt-in `XTRA` diagnostics (`--xtra star`, `--xtra binary-rlo`), see below. |
| `flexmesa.testing.stepdump_compare` | The comparison (reads any dump described here). |

The bundle builder is `scripts/data/build_mesa_stepdump_windows.py`.

## Running

```bash
python run_stepdumps.py --mesa-dir "$MESA_DIR" --mesasdk-root "$MESASDK_ROOT" \
    --scratch <empty directory> --out <output directory> \
    [--steps 3] [--first-model N] [--occurrence K] [--stage S] \
    [--keep-running] --cases star/<case> [binary/<case> astero/<case> ...]
```

The first call builds the overlay under `--scratch`; later calls pass
`--reuse-overlay`. Linking may need `LIBRARY_PATH` to point at a `libz` the SDK
can use. Output is `<out>/<suite>/<case>.stepdump`: `#` header lines (MESA
version, compiler, SDK, thread count, the window, the inlist hashes), then the
records.

## Choosing the window

By default the dump holds the first 3 models of each star in every process of
the case. These options place a longer or later window.

| Option | Environment variable of the patch | Meaning |
| :--- | :--- | :--- |
| `--steps N` | `FLEXMESA_STEPDUMP_STEPS` | Distinct consecutive model numbers dumped per star. |
| `--first-model F` | `FLEXMESA_STEPDUMP_FIRST` | The window opens when the model number rises to `F` or past it; models below `F` are never written. |
| `--occurrence K` | `FLEXMESA_STEPDUMP_OCCURRENCE` | The window opens at the K-th such rise (default 1). |
| `--stage S` | `FLEXMESA_STEPDUMP_STAGE` | Dump only the S-th process of the case that calls the solver (default every process). |
| `--keep-running` | (omits `FLEXMESA_STEPDUMP_STOP`) | Let the run continue after the window instead of stopping it. |

Each `./rn` stage is a separate process, and the patch numbers them by the
`RUN` records already in the output file, so `--stage 2` of a case with
two inlists is the second `RUN` section. Processes before the selected one run
to completion (they write a `RUN` line and nothing else); the selected one
stops after its window, unless `--keep-running` is given.

### Restarted model numbers

A case whose model numbers restart inside one process (a relax phase, then the
evolve phase) passes `--first-model F` once per phase. `star/wd_stable_h_burn`
relaxes through models 1..84 and then evolves from model 1 again, so model 70
occurs twice:

```bash
# relax phase, models 70..81 (the first rise to 70)
... --first-model 70 --steps 12 --cases star/wd_stable_h_burn
# evolve phase, models 70..81 (the second rise to 70)
... --first-model 70 --occurrence 2 --steps 12 --cases star/wd_stable_h_burn
```

Before `--occurrence` existed, the first form was the only one, and reaching
the evolve phase meant a window long enough to include the relax models
(`--steps 27`) and separating the two by hand. A window that is not full when
the numbering restarts resumes at the next rise to `F`, so it never holds
models below `F`.

### Later stages

```bash
# second ./rn stage of star/ns_h, models 90..101
... --stage 2 --first-model 90 --steps 12 --cases star/ns_h
# the whole second stage from its first model
... --stage 2 --first-model 1 --steps 22 --cases star/ns_h
```

### Letting the run finish

`--keep-running` leaves the process running after the window, for a run whose
later output is wanted next to the dump (histories, terminal output in the
case's scratch copy). The `DONE` record is written once either way, when the
first star dumps its last model.

A window that ends on the last model of its stage never meets a later model,
so the patch cannot write `DONE` there. With `--stage` (and no
`--keep-running`) `run_stepdumps.py` watches the raw file: when the selected
process has written records and the next stage has started, or the whole run
has exited, it appends `DONE`, and in the first case stops the remaining
stages. A stage that ends before the window opens gets no `DONE`.

## What is recorded

The `#` header carries `steps_per_star`, `first_model`, `occurrence` and
`stage` (`all` when every process is dumped), so a dump says which window it
is. Records are documented in `stepdump_patch.py`. A dump is written complete;
the comparison tolerates record types it does not know (diagnostic records
from a locally patched solver, for example): it skips them, counts them per
type, and prints the counts in its report. Only the first `RUN` section that
holds records is compared.

## Opt-in diagnostics (`XTRA` records)

`xtra_patch.py` adds write statements for scalars the default records do not
hold. Nothing changes unless `--xtra` is given: the default overlay, dump
records and bundles are as before, and the comparators skip the extra record
types (counting them). The overlay under `--scratch` is built once with the
chosen options, so pass the same `--xtra` choices on every run that reuses it.
The full record list is in the docstring of `xtra_patch.py`.

### `--xtra star`: solver scalars

Tagged records in the step-dump file, one value per line:
`XTRA <tag> model iter j k hex decimal` (hex is the binary64 bit pattern,
decimal 17 significant digits). They hold, per equation `j`, the sum and the
maximum (with its zone) of `|equ(j,k)|` at the initial residual evaluation
(`iter` 0) and after each accepted correction, the sum of `|soln(j,k)|` per
correction, and the `M_center`, `R_center`, `L_center`, `xmstar` and `mstar`
values at solver entry. `--xtra-zones` adds per-zone values (the equations and
`lnd lnT lnR L Peos opacity gradT gradr mlt_vc Cp energy` after each
residual evaluation); `--xtra-corr-zones` (default: the same list) adds
`soln` and `dx` at each correction.

```bash
python run_stepdumps.py --mesa-dir "$MESA_DIR" --mesasdk-root "$MESASDK_ROOT" \
    --scratch <empty directory> --out <output directory> \
    --xtra star --xtra-zones 36,37,262,474:481 --xtra-corr-zones 1,262,474:481 \
    --stage 2 --first-model 9 --steps 3 --cases star/make_planets
```

`residual_norm` of the `CALL` and `RESID` records is MESA's `sizequ`
(`star/private/solver_support.f90`), which divides a weighted sum by a term
count that is not simply `nvar * nz`. With `do_burn` or `do_mix` on, the first
block adds all `nvar` equations over `nz` zones, then a second block adds the
`nvar_chem` chemistry equations (`j > nvar_hydro`) to both the sum and the term
count again, so upstream counts the chemistry equations twice:

```
residual_norm = (sum_{j=1..nvar} S_j + sum_{j>nvar_hydro} S_j)
                / ((nvar + nvar_chem) * nz),    S_j = sum_k |equ(j,k) * w(j,k)|
```

(`w` is `residual_weight`, 1 unless an equation sets it.) For
`star/make_planets` stage 2, `nvar = 12` and `nvar_chem = 8`, so the divisor is
`20 * nz`; the printed `RESID_EQUSUM` values reproduce the `RESID` norms to
rounding (`1e-15` relative). With neither `do_burn` nor `do_mix` only the
`nvar_hydro` equations count, and `convergence_ignore_equL_residuals` removes
`j = equL` from the sum and `nz` from the count. `max_residual` is the
maximum of the same weighted terms, which is the `RESID_EQUMAX` maximum while
the weights are 1.

### `--xtra binary-rlo`: implicit mass-transfer iterations

Records `XTRA <tag> model rlo_iter name hex decimal` in
`<out>/binary/<case>.xtra` (cgs; `mdot` is negative for mass loss) for each
iteration of the implicit scheme: current and explicit `mdot`, the function
`f`, the brackets `mdot_lo`/`mdot_hi` and their function values, whether
`pick_mdot_for_implicit_rlo` tries `find0_quadratic` or bisects, the result and
`ierr` of `find0_quadratic` before and after the `find0` fallback, and the next
`mdot` before and after the maximum-rate clamp. The bracket records are
written after the `num_tries == 1` reset, so iteration 1 shows the reset
values; an iteration that ends within tolerance has `RLO_CALL` and
`RLO_RESULT_in_tol` only. `--xtra-binary-models FIRST:LAST` is required; the
run proceeds to model `LAST` (it is not stopped by the step-dump window) and
stops there.

```bash
python run_stepdumps.py --mesa-dir "$MESA_DIR" --mesasdk-root "$MESASDK_ROOT" \
    --scratch <empty directory> --out <output directory> \
    --xtra binary-rlo --xtra-binary-models 102:102 \
    --cases binary/wind_fed_bhhmxb
```

## Comparing

```bash
flexmesa trace star <case> --out <dir> --compare <dump>.stepdump \
    --allow-unsupported-controls
python first_divergence_table.py --dumps <dir of suite/case.stepdump> \
    --work <scratch> --markdown-out table.md --json-out table.json
```

FlexMESA's own dump (`crates/flexmesa-star/src/stepdump.rs`) honours
`FLEXMESA_STEPDUMP_STEPS` and `FLEXMESA_STEPDUMP_FIRST` and has no occurrence
or stage selector. A MESA window with `--occurrence` above 1 or a `--stage`
therefore compares only against a FlexMESA dump that opens at the same place
in its run; until the FlexMESA dump gains the selectors, such a window is read
directly (it is a complete dump) rather than through `--compare`.

## The mid-run window bundle

`windows.toml` lists the windows kept for localizing a difference beyond the
first models of a case. `regenerate_windows.py` runs them:

```bash
python regenerate_windows.py --mesa-dir "$MESA_DIR" --mesasdk-root "$MESASDK_ROOT" \
    --scratch <dir> --out <dir> [--only star/ns_h star/wd_stable_h_burn_s1_k2_m70_n12] \
    [--dry-run]
```

Output is `<out>/<window id>/<suite>/<case>.stepdump`, with the window id
`<case>_s<stage>_k<occurrence>_m<first model>_n<steps>`. A window whose dump
already ends in `DONE` is skipped, so an interrupted run resumes. Every window
is real MESA time, so run it under the machine's compute reservation.
`--dry-run` prints the commands.

`scripts/data/build_mesa_stepdump_windows.py --dumps <dir> [<dir> ...] --out
mesa-stepdump-windows-v1.tar.zst` packs the windows it finds under the given
directories (matching each file's header to a manifest entry) into the
opt-in bundle, with `PROVENANCE.json`, a copy of `windows.toml` and a per-file
SHA-256 manifest.

## Causal checks with a private copy of a case

To test a suspected cause inside MESA itself without editing or rebuilding
`$MESA_DIR`, work in a disposable private copy of the single test-suite case:

1. **What to copy:** Copy only `$MESA_DIR/<suite>/test_suite/<case>` (its
   inlists, `src/`, `make/`, `clean`/`mk`/`rn`/`re` scripts, model/column files,
   and any `photos/` checkpoint needed for `./re`, plus any shared
   `test_suite` inlist include it references) into a private scratch directory
   outside `$MESA_DIR`. Leave `$MESA_DIR` untouched.
2. **Build and run against the unmodified tree:** Register the hypothesized
   perturbation only in the scratch copy's `src/run_star_extras.f90` (for
   example via an `other_*` hook such as `other_gradr_factor` gated on an
   `x_logical_ctrl` switch when the hook runs at the target call site). With
   `MESA_DIR` and `MESASDK_ROOT` pointing at the unmodified MESA tree (or an
   already-built step-dump overlay if step-dump records are needed), rebuild
   only the case executable (`./clean && ./mk`) and run `./rn` or `./re <photo>`
   from that scratch directory.
3. **Diagnostic only:** Any run from a modified `run_star_extras.f90` is a
   one-off causal diagnostic (for example to reproduce an exact ULP shift) and
   is **never** a MESA reference run, baseline, or golden artifact. Delete the
   scratch directory after recording the observation.
