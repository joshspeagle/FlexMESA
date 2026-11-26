# Record/replay routine oracles from MESA 26.4.1 (`#552`)

A step dump (`../stepdumps/`) says *where* in a run `FlexMESA` first departs from MESA 26.4.1: which model, which solver iteration, which zone. Record/replay says *which routine* is responsible. An instrumented overlay build of MESA 26.4.1 appends one record per call of a chosen routine during a real test-suite run, holding the call context and every input and output as IEEE-754 bit patterns. The `flexmesa-replay` tool then calls the matching `FlexMESA` port with exactly the captured inputs and compares the outputs bit for bit. A routine whose calls all replay identically is cleared, and the divergence lies in the arguments its caller assembled; a routine with a differing call is the place to look, and the report names the call, the output and the bits.

The first family is the convection kernel: `set_TDC` and `set_MLT` (`turb/public/turb.f90:114-204, 275-292`), captured at their four call sites in `Get_results` (`star/private/turb_support.f90:328-334`, `:349-355` with `gradr_scaled`, `:366-369`, `:384-387` with `gradr_scaled`) and replayed through `flexmesa_turb::set_tdc` and `flexmesa_turb::set_mlt`.

## Files

| File | Role |
| :--- | :--- |
| `replay_patch.py` | Inserts the capture calls and their writer into `star/private/turb_support.f90`. Output only: no MESA expression, branch or order of arithmetic changes. Documents the record format. |
| `run_capture.py` | Builds the overlay (this patch plus the step-dump patch), runs cases with capture on, and writes `<case>.replay` and `<case>.stepdump` with a provenance header. |
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

## Data volume

A `TDC` record averages about 2.5 kB (16 auto-diff inputs and 4 auto-diff outputs, typically 4 to 6 nonzero lanes each); an `MLT` record about 1.5 kB. A sweep calls the kernel once per face, and a model takes one sweep per solver iteration plus two or three at the start of the step, so an 800-zone model with 4 solver iterations writes about 6,000 records, or 15 MB. gzip reduces a capture by a factor of about 3. Captures stay out of git; a later versioned data bundle would hold the captures CI replays (`DECISIONS #67`).

## Extending the pattern

Each new family needs the same three pieces: a capture call after the routine's call site in an overlay patch (one `write` per call, inputs and outputs, under the critical section), a `SCHEMA` line, and a replay function in `flexmesa-replay` that calls the port and compares every output. The constraint that decides where to capture is that replay must be able to call the port without star state, so the record has to hold the routine's complete input, including configuration.

- **EOS** (`eosDT_get`, `eos/public/eos_lib.f90:108`, called from `star/private/eos_support.f90:78` for each cell from `micro.f90`). Inputs: `species`, `chem_id`, `xa(species)`, `Rho`, `logRho`, `T`, `logT`; outputs: `res(num_eos_basic_results)`, `d_dlnd`, `d_dlnT`, `d_dxa(:, species)`, `ierr`. The EOS handle's controls (component blends, `Zfrac` limits, `use_*` flags) are not arguments: record them once per run as a `CONFIG` line read from the handle, and have the replay tool build its EOS handle from that line and fail on any control it does not know. An 8-species cell is about 300 values, 5 kB per call, about 15 MB per 800-zone model with 4 iterations.
- **kap** (`kap_get`, called from `star/private/kap_support.f90:489`). Inputs (`kap/public/kap_lib.f90:161-166`): `species`, `chem_id`, `xa`, `logRho`, `logT`, `lnfree_e` and its two partials, `eta` and its two partials; outputs: `kap_fracs`, `kap`, `dlnkap_dlnRho`, `dlnkap_dlnT`, `dlnkap_dxa`, `ierr`. The opacity tables and blending controls go in a `CONFIG` line as for EOS. About 1 kB per call.
- **net** (`net_get`, called from `star/private/net.f90:245`). Inputs (`net/public/net_lib.f90:550-558`): `x`, `temp`, `rho` and their logarithms, `abar`, `zbar`, `z2bar`, `ye`, `eta` and its partials, `rate_factors`, `weak_rate_factor`, `reaction_Qs`, `reaction_neuQs`, `screening_mode`; outputs: `eps_nuc` and its partials, `dxdt`, `d_dxdt_dRho`, `d_dxdt_dT`, `d_dxdt_dx`. The Jacobian `d_dxdt_dx` is `species x species`: about 1,000 values for `approx21`, but 240,000 for a 495-isotope network. Write it sparse (nonzero entries only) and restrict the window with a zone list (`FLEXMESA_REPLAY_ZONES`) for large networks. The net handle (network file, rate selections, cached rate tables) is configuration and is recorded per run.

The same tool can also take records one level up. A capture of `Get_results`' own arguments together with the star fields it reads (`mlt_vc_old(k)`, `L(k)`, `L_start`, `q(k)`, `csound_face(k)` and the controls it uses) would let replay check the caller-side arithmetic of `do1_mlt_2` (face scale, `Ptot`, `grav`, `scale` for the TDC luminosity test, `max_conv_vel`), which is where these four cases place any convection difference.
