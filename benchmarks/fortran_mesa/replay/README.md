# Record/replay routine oracles from MESA 26.4.1 (`#552`)

A step dump (`../stepdumps/`) says *where* in a run `FlexMESA` first departs from MESA 26.4.1: which model, which solver iteration, which zone. Record/replay says *which routine* is responsible. An instrumented overlay build of MESA 26.4.1 appends one record per call of a chosen routine during a real test-suite run, holding the call context and every input and output as IEEE-754 bit patterns. The `flexmesa-replay` tool then calls the matching `FlexMESA` port with exactly the captured inputs and compares the outputs bit for bit. A routine whose calls all replay identically is cleared, and the divergence lies in the arguments its caller assembled; a routine with a differing call is the place to look, and the report names the call, the output and the bits.

The families are the convection kernel and the cell EOS. The convection kernel is `set_TDC` and `set_MLT` (`turb/public/turb.f90:114-204, 275-292`), captured at their four call sites in `Get_results` (`star/private/turb_support.f90:328-334`, `:349-355` with `gradr_scaled`, `:366-369`, `:384-387` with `gradr_scaled`) and replayed through `flexmesa_turb::set_tdc` and `flexmesa_turb::set_mlt`.

## Files

| File | Role |
| :--- | :--- |
| `replay_patch.py` | Inserts the capture calls into `star/private/turb_support.f90` and `micro.f90`, and the shared writers into `star_utils.f90`. Output only: no MESA expression, branch or order of arithmetic changes. Documents the record format. |
| `run_capture.py` | Builds the overlay (this patch plus the step-dump patch), runs cases with capture on, and writes `<case>.replay` and `<case>.stepdump` with a provenance header. |
| `scripts/data/build_mesa_replay.py` | Packs captures into a `mesa-replay-<version>.tar.zst` bundle with `PROVENANCE.json` and a per-file SHA-256 manifest. |
| `crates/flexmesa-replay` | Reads a capture, replays each record and reports per-family results. |

## Recipe

Capture (the pristine tree is never modified; the overlay holds real copies of `star/`, `lib/` and `include/` and links to the rest):

```bash
export LIBRARY_PATH=<directory holding a libz the SDK can link>
python benchmarks/fortran_mesa/replay/run_capture.py \
    --mesa-dir "$MESA_DIR" --mesasdk-root "$MESASDK_ROOT" \
    --scratch <empty directory> --out <output directory> \
    --models 3 --cases star/conserve_angular_momentum star/T_tau_gradr
```

Add `--reuse-overlay` to run more cases against an overlay already built under `--scratch`, and `--first-model N` to start the window at model number `N` (earlier stages then run to completion first). The step-dump patch stops MESA once `--models` models have been solved, which bounds each run; the capture window itself is left open up to that stop, so it also holds the sweeps made while setting up the loaded model and the first step.

Replay:

```bash
cargo run --release -p flexmesa-replay -- <output directory>/star/<case>.replay
```

The tool prints the capture header, then for each family the number of calls replayed and identical, per-output tallies of differing calls with their largest ulp distance (values and `d1Array` partials separately), every differing component of the first differing call in file order, and the earliest differing call by `(run, model, solver_iter, k)`. Exit status is 0 when every call is identical, 1 when any call differs or could not be replayed, and 2 for a malformed capture or a schema this tool does not implement.

The overlay can also be driven by hand: set `FLEXMESA_REPLAY=<file>` (and optionally `FLEXMESA_REPLAY_MODELS`, default 1, and `FLEXMESA_REPLAY_FIRST`) in the environment of an `./rn` built against the overlay. Each `./rn` stage appends a `RUN` line and the `SCHEMA` lines.

## Record format

One record per line; fields are separated by one space. `SCHEMA <family> <name>:<type> ...` lines, written by the overlay itself, are the single definition of field order, and the replay tool refuses a capture whose schema differs from the one it implements. Types: `i` integer, `l` logical (`T`/`F`), `s` string, `r` a `real(dp)` as its `Z16.16` bit pattern, `a` an `auto_diff_real_star_order1` as the value's bit pattern followed by `/<lane>:<bits>` for each of the 33 `d1Array` lanes (1-based, MESA's layout, `star_data/public/star_data_def.inc:255-288`) whose bits are not `+0.0`. Lanes holding `-0.0` are written, and they occur: a face gravity `cgrav*m/pow2(r)` carries `-0.0` in every lane that `r` does not depend on.

The `turb` kernels are generic over the number of lanes and treat every lane alike, so replay runs them on `DualOrder1<33>` in MESA's own layout. No mapping onto `flexmesa-star`'s 24-lane `StarAdOrder1` (whose lane order also differs) is needed, and all 33 lanes are compared.

Each record carries its context: `site` (1, 2 for the two `set_TDC` calls; 3, 4 for the two `set_MLT` calls), star `id`, `model`, `solver_iter` and face `k`. `Get_results` runs inside the `!$OMP PARALLEL DO` of `set_mlt_vars` (`star/private/turb_info.f90:52`), so the writer holds a named critical section and emits each record in one `write`: records are complete, but their order within one sweep over faces depends on thread scheduling.

Two upstream calling conventions are recorded rather than assumed:

- `get_TDC_solution` leaves its `intent(out)` iteration count unassigned on its radiative exits (`turb/private/tdc.f90:55,113`), so MESA returns whatever the caller held. The record carries that value (`tdc_num_iters_in`); `set_tdc` returns 0 there, and a difference is reported as such.
- `calc_MLT` returns early without assigning its outputs on three paths (`turb/private/mlt.f90:140,145,169`), leaving the caller's values. At site 3 those are `Get_results`' no-mixing initial values, which `set_mlt` reproduces; at site 4 they are site 3's outputs, which a stand-alone `set_mlt` call cannot know. A site-4 difference on such a path points at this convention, not at the kernel arithmetic.

## First result (MESA 26.4.1, SDK `x86_64-linux-26.6.1`, `gfortran 15.2.0`, 4 OpenMP threads, `--models 3`)

| Case | `TDC` calls | `MLT` calls | Capture size | Replay |
| :--- | ---: | ---: | ---: | :--- |
| `star/conserve_angular_momentum` | 19,279 | 0 | 48.7 MB | all identical |
| `star/T_tau_gradr` | 26,922 | 15,898 | 103.9 MB | all identical |
| `star/1M_thermohaline` | 0 | 9,528 | 23.0 MB | all identical |
| `star/tzo` | 0 | 3,132 | 7.5 MB | all identical |

"Identical" means every output value, every one of the 33 partial lanes, `mixing_type`, `tdc_num_iters` and `ierr`. Only sites 1 and 3 occurred in these windows (no `use_superad_reduction` rescaling with `Gamma_factor > 1`, and no `k = 0` calls). The convection kernels are therefore cleared for the first models of these cases: any difference in `gradT`, `Y_face`, `conv_vel` or `D` there comes from the arguments `do1_mlt_2` assembles, not from `set_tdc` or `set_mlt`. A one-bit change to one captured input or output is detected and reported with its field and bits (negative control in `crates/flexmesa-replay/tests/replay_roundtrip_test.rs`, and by hand on a real capture).

## The cell EOS (`EOS`)

After the `get_eos` call of `do_eos_for_cell` (`star/private/micro.f90:357-361`), the overlay writes an `EOS` record (site 6) with that call's inputs (`chem_id`, `xa(1:species)`, `Rho`, `logRho`, `T`, `logT`) and results (`res(1:26)`, `d_dlnd`, `d_dlnT`, `d_dxa(1:2, 1:species)` in column-major order, `ierr`), each vector written as type `v` (bit patterns joined by commas). The EOS handle's controls are not arguments, so before the first `EOS` record of a run the overlay writes one `CONFIG EOS` line with every control of `EoS_General_Info` (`eos/public/eos_def.f90`) read from `s% eos_handle`, as `name=value` entries (reals as bit patterns, strings quoted with blanks written as `~`, arrays element by element).

`flexmesa-replay` builds its `EosConfig` from that line, mapping the controls as `flexmesa-star` maps the `&eos` namelist, and refuses a line with an unknown or missing control, a hook (`use_other_eos_component`, `use_other_eos_results`), `dbg`, or table files other than the bundled ones. It then calls `flexmesa_eos::eos_dt_get_zones_with_logs`, the entry point of `flexmesa-star`'s `do_eos`, on the captured inputs (returning `ierr = -1` for `logRho < -25` as upstream `get_eos` does, `star/private/eos_support.f90:70-76`) and compares all 78 results and the `2 x species` composition partials bit for bit. Reports name each component (`res(lnE)`, `d_dlnd(frac_Skye)`, `d_dxa(lnPgas, species 3)`). `FlexMESA`'s tap writes the same `EOS` record for every cell EOS result `store_eos_for_cell` stores, so `flexmesa-replay compare` also matches the EOS inputs `FlexMESA` passes with MESA's call by call.

### EOS result (MESA 26.4.1, SDK `x86_64-linux-26.6.1`, `gfortran 15.2.0`, 4 OpenMP threads, `--models 2`)

Eighteen star cases (the near-miss and cluster cases of `#546`), first two models of the first `./rn` stage; RSP cases with `--window-models 2` (RSP has its own solver, so the step dump neither stops the run nor records it):

| Case | `EOS` | `TDC` | `MLT` | Replay |
| :--- | ---: | ---: | ---: | :--- |
| `star/12M_pre_ms_to_core_collapse` | 51,137 | 42,174 | 0 | all identical |
| `star/20M_pre_ms_to_core_collapse` | 46,278 | 40,661 | 0 | all identical |
| `star/5M_cepheid_blue_loop` | 32,638 | 33,660 | 0 | all identical |
| `star/accreted_material_j` | 29,099 | 19,563 | 1,074 | all identical |
| `star/check_pulse_atm` | 56,230 | 21,913 | 10,194 | all identical |
| `star/conserve_angular_momentum` | 13,651 | 13,651 | 0 | all identical |
| `star/high_z` | 18,383 | 11,461 | 1,390 | all identical |
| `star/magnetic_braking` | 36,832 | 0 | 68,158 | all identical |
| `star/ns_c` | 94,512 | 39,343 | 20,063 | 15 `EOS` calls differ |
| `star/ns_h` | 28,170 | 27,810 | 0 | all identical |
| `star/ns_he` | 35,524 | 0 | 499 | all identical |
| `star/rsp_Delta_Scuti` | 600 | 0 | 0 | all identical |
| `star/rsp_Type_II_Cepheid` | 800 | 0 | 0 | all identical |
| `star/split_burn_big_net` | 70,952 | 0 | 24,936 | all identical |
| `star/starspots` | 172,752 | 0 | 205,608 | all identical |
| `star/tzo` | 12,182 | 0 | 2,262 | all identical |
| `star/wd_cool_0.6M` | 27,851 | 0 | 175 | all identical |
| `star/zams_to_cc_80` | 43,210 | 35,961 | 0 | all identical |

In total 770,786 of 770,801 `EOS` calls, all 286,197 `TDC` calls and all 334,359 `MLT` calls replay bit for bit. The 15 differing `EOS` calls of `star/ns_c` lie where the FreeEOS and Skye components blend in a carbon-dominated envelope (first at model 1, solver iteration 1, `k = 214`, `logRho = 1.617`, `logT = 7.315`, `frac_FreeEOS = 0.975`, `frac_Skye = 0.025`): MESA returns `d_dlnd(frac_FreeEOS) = d_dlnd(frac_Skye) = 0` where `FlexMESA` returns `+1.17e-30` and `-1.17e-30`, and in four of the calls the blended results and their partials differ by 1 to about 400 ulp (up to 12,032 ulp in `d_dlnd(Cp)`). These are findings for the EOS blend, not changed here.

## Comparing the arguments FlexMESA assembles (`GETR`)

Replay clears a kernel; it cannot check the caller, because `Get_results` reads star state beyond its arguments. For that the overlay also writes, after the `Get_results` call in `do1_mlt_eval` (`star/private/turb_support.f90:188-192`), a `GETR` record with that call's arguments (`r`, `L`, `T`, `P`, `opacity`, `rho`, `dV`, `chiRho`, `chiT`, `Cp`, `gradr`, `grada`, `scale_height`, `iso`, `XH1`, `cgrav`, `m`, `gradL_composition_term`, `mixing_length_alpha`, `alpha_semiconvection`, `thermohaline_coeff`, `energy`) and results (`mixing_type`, `gradT`, `Y_face`, `conv_vel`, `D`, `Gamma`). `FlexMESA` writes the same `GETR`, `TDC` and `MLT` records for its own run from a tap in `crates/flexmesa-star/src/replay_tap.rs`, called from `do1_mlt_2`:

```bash
FLEXMESA_REPLAY_TAP=<file> FLEXMESA_REPLAY_TAP_MODELS=10 \
    flexmesa trace star <case> --out <dir> --max-steps 4 [...]
cargo run --release -p flexmesa-replay -- compare <case>.replay <file>
```

The tap is off unless `FLEXMESA_REPLAY_TAP` is set, costs one initialized `OnceLock` read per face when off, and only reads: a run with the tap on writes the same step dump, byte for byte, as with it off. It writes auto-diff values in MESA's 33-lane layout (`StarAdOrder1`'s 24 slots mapped by name; its unused `omega` slots go to MESA's `w_div_wc` lanes) and `NA` for a value it does not have (`tdc_num_iters_in`, and `dV` and `energy` before a start-of-step density exists).

`compare` matches records by family, model, solver iteration, face, call site and occurrence. Where one side calls a routine at a face more often than the other at the same model and iteration (an extra sweep, or the trial calls of predictive mixing), the two sequences of calls are aligned by content, and the counts are reported. Calls at a model and iteration the other capture never reached (its run stopped earlier) are counted apart and not treated as differences. Components that differ only as `+0.0` against `-0.0` are counted as identical and tallied; they include the lanes `StarAdOrder1` does not represent, where MESA often holds `-0.0`. The report gives per-field tallies with the earliest differing call of each field, and every differing field of the earliest differing call, ordering sweeps by their position in the MESA capture and ranking `GETR` before the kernel records of the same face.

First results (MESA as above, 3 solved models; `FlexMESA` at this branch with `--max-steps 4`):

- `star/conserve_angular_momentum` (19,279 `GETR` and 19,279 `TDC` calls matched): every argument and result is identical (up to the sign of zero) through model 2, solver iteration 0. The first difference is `T` at face 18 after the first Newton iteration of model 2 (4 ulp), followed by every other state-derived argument. The convection code is therefore cleared up to the point where the step dump finds the first divergence (model 2, first solver call, `equ_h1` at zone 535): the residuals differ before any convection argument does. `FlexMESA` also makes one extra sweep over all faces at model 0 that MESA does not make (its values differ from MESA's only sweep; the following sweep matches it).
- `star/5M_cepheid_blue_loop` (55,637 calls of each family matched): the only differing arguments are `XH1` (1 ulp at face 1146 in the first sweep, then wherever hydrogen is present) and the `lnd_m1` partial (lane 1) of `energy`, which differs from the second evaluation at face 1266 onward by about 0.25 % while its value agrees. `do1_mlt_2` passes `X` (`xh_mass`, the summed hydrogen isotopes) where upstream passes `XH1 = xa(h1)` (`turb_support.f90:176`); the network of this case includes `h2`. Neither difference reaches `set_tdc`'s results here (the enthalpy flux limiter that reads `energy` is off, and `XH1` is read only by the thermohaline options of Brown et al. and Traxler et al.). `FlexMESA` makes 3,448 more calls than MESA at the first model's solver iteration 0.

## Data volume

A `TDC` record averages about 2.5 kB (16 auto-diff inputs and 4 auto-diff outputs, typically 4 to 6 nonzero lanes each); an `MLT` record about 1.5 kB. A sweep calls the kernel once per face, and a model takes one sweep per solver iteration plus two or three at the start of the step, so an 800-zone model with 4 solver iterations writes about 6,000 records, or 15 MB. gzip reduces a capture by a factor of about 3. An `EOS` record is 2 to 8 kB, growing with the number of species (`xa`, `d_dxa` and `chem_id` scale with it). Captures stay out of git; `scripts/data/build_mesa_replay.py` packs them into a versioned bundle (`DECISIONS #67`), and `crates/flexmesa-replay/tests/data/accreted_material_j_sample.replay` (80 `TDC`, 60 `MLT` and 120 `EOS` records of a real capture, 0.57 MB) lets CI exercise the replay path without it.

## Extending the pattern

Each new family needs the same three pieces: a capture call after the routine's call site in an overlay patch (one `write` per call, inputs and outputs, under the critical section), a `SCHEMA` line, and a replay function in `flexmesa-replay` that calls the port and compares every output. The constraint that decides where to capture is that replay must be able to call the port without star state, so the record has to hold the routine's complete input, including configuration.

- **kap** (`kap_get`, called from `star/private/kap_support.f90:489`). Inputs (`kap/public/kap_lib.f90:161-166`): `species`, `chem_id`, `xa`, `logRho`, `logT`, `lnfree_e` and its two partials, `eta` and its two partials; outputs: `kap_fracs`, `kap`, `dlnkap_dlnRho`, `dlnkap_dlnT`, `dlnkap_dxa`, `ierr`. The opacity tables and blending controls go in a `CONFIG` line as for EOS. About 1 kB per call.
- **net** (`net_get`, called from `star/private/net.f90:245`). Inputs (`net/public/net_lib.f90:550-558`): `x`, `temp`, `rho` and their logarithms, `abar`, `zbar`, `z2bar`, `ye`, `eta` and its partials, `rate_factors`, `weak_rate_factor`, `reaction_Qs`, `reaction_neuQs`, `screening_mode`; outputs: `eps_nuc` and its partials, `dxdt`, `d_dxdt_dRho`, `d_dxdt_dT`, `d_dxdt_dx`. The Jacobian `d_dxdt_dx` is `species x species`: about 1,000 values for `approx21`, but 240,000 for a 495-isotope network. Write it sparse (nonzero entries only) and restrict the window with a zone list (`FLEXMESA_REPLAY_ZONES`) for large networks. The net handle (network file, rate selections, cached rate tables) is configuration and is recorded per run.

