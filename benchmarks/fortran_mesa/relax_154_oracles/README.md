# MESA 26.4.1 Initial-Model Captures for the `#154` Oracle Tests

Captures, from compiled MESA release `26.4.1` (`r26.4.1`, `MESA SDK x86_64-linux-26.6.1`, `gfortran 15.2.0`), the zone-by-zone initial model of the three setups that `crates/flexmesa-star/tests/relax_154_parity_test.rs` pins: `create_initial_model` with `new_net_name = 'pp_and_cno_extras.net'`, the `make_brown_dwarf`-style `create_initial_model` with the default `basic.net`, and the `15M_dynamo` `create_pre_main_sequence_model`. The committed samples are the scalars plus the zones the Rust test inspects.

## What is dumped, and where

`relax154_patch.py` adds `write` statements (output only) to `star/private/init.f90` in a private overlay, in `model_builder` (called by `star_create_initial_model` and `star_create_pre_ms_model` from `do_load1_star`, `star/job/run_star_support.f90:1661-1699`). Three points, tagged in every record:

| Tag | Point in `init.f90` (MESA 26.4.1) | Contents |
| :--- | :--- | :--- |
| `BUILD` | line 1043, right after `build_initial_model` (960) or `build_pre_ms_model` (1001) returned, before `finish_load_model` | `xh` unknowns lnd, lnT, lnR, L; `dq`; `xa` |
| `LOADED` | line 1054, after `finish_load_model` (1046) | the above plus `q`, `m`, and the `s% lnd`, `s% lnT`, `s% lnR`, `s% L` arrays |
| `FINAL` | line 1121, end of `model_builder` | same as `LOADED`, after the mass relaxation, `do_relax_num_steps` and, for the pre-main-sequence setup, `do_relax_to_radiative_core` (default `pre_ms_relax_to_start_radiative_core`, which takes `nz` from 1033 to 680) |

The two `create_initial_model` setups agree at all three points. For the 15 Msun setup, `BUILD` and `LOADED` are the freshly built model (`nz = 1033`, as the Rust test asserts); `FINAL` is what a MESA run actually starts evolving from. Right after `FINAL`, with `FLEXMESA_RELAX154_STOP` set, the process executes `stop 0`, so no evolve step runs and the state is never changed by a step or rounded by a profile. That early exit is the only change to control flow.

Per zone `k = 1..nz` (`k = 1` is the surface): `lnd`, `lnT`, `lnR`, `L`, `dq`, `q`, `m`, `xa(1:species, k)`. Scalars: `nz`, `species`, `net_name`, the species names with `chem_id`, `mstar`, `xmstar`, `M_center`, `R_center`, `L_center`, `initial_mass`, `dt_next`. Every float is written as `ES25.16E3` decimal and as 16-digit IEEE-754 hex; `xa` is hex only.

## Setups

The inlists in `inlists/` set exactly what the Rust test sets and leave everything else at MESA defaults, with `pre_ms_relax_num_steps = 0` and `initial_model_relax_num_steps = 0` explicit.

- `create_initial_model_pp_and_cno_extras`: 1.9892e31 g, 3.5e10 cm, `initial_z = 1d-4`, `Zbase = 1d-4`, `kap_lowT_prefix = 'lowT_Freedman11'`, `change_net` to `pp_and_cno_extras.net`.
- `make_brown_dwarf_basic_net`: as above without the net change (`basic.net`).
- `pre_ms_15m_dynamo`: 15 Msun, `initial_z = 0.02`, `initial_y = -1`, `Zbase = 0.02`, `use_Type2_opacities`, `mlt_option = 'Cox'`, `mixing_length_alpha = 1.6`, `max_dq = 4d-3`.

The Rust test's third `create_initial_model` call (1.8986e30 g, 1.39822e10 cm, `job.net_name` instead of `change_net`) only asserts species count and `chem_id`, so it has no capture.

## Rerunning

With `MESA_DIR` a compiled MESA `26.4.1` tree, `MESASDK_ROOT` the SDK, and `LIBRARY_PATH` pointing at a directory with `libz` the SDK linker can use:

```bash
python run_relax154_oracles.py --mesa-dir "$MESA_DIR" \
    --mesasdk-root "$MESASDK_ROOT" --scratch <empty directory> \
    --out <output directory> --sample-out samples
```

The script never modifies `MESA_DIR` and checks that its `libstar.a` carries no dump marker before and after. Each `.dump` header records the MESA version, compiler, SDK, thread count and inlist SHA-256. `samples/*.sample` keep the header, all non-zone records, and the zones named in `SETUPS` in the script (1-based surface, middle, centre).
FINAL samples keep zones 1, 517 and 1033 only where they exist (the 15 Msun `FINAL` model has 680 zones).
