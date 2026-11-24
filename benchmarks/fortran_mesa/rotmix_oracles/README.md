# MESA 26.4.1 `set_rotation_mixing_info` Reference Oracle (`#158`)

This directory holds one call of MESA's rotational-mixing routine captured from a real MESA release `26.4.1` run (`r26.4.1`, commit `c8c202662922a22b3efefa824473ad615c623afd`), built with `MESA SDK x86_64-linux-26.6.1` (`GNU Fortran (GCC) 15.2.0`, `CRMATH` / `crlibm 1.0beta4`) and the default `star` build flags of `utils/makefile_header`.

| File | Role |
| :--- | :--- |
| `mesa_26_4_1_rotmix_20M_model2.txt` | Committed fixture: inputs at entry to `update_rotation_mixing_info` and its outputs, for one call |
| `instrumentation.patch` | Patch to `star/private/star_utils.f90` and `star/private/mix_info.f90` that writes the raw dumps |
| `make_fixture.py` | Trims the raw dumps to the one call kept in the fixture |

The Rust test `crates/flexmesa-star/tests/rotmix_oracle_158_test.rs` loads the fixture into a `StarModel`, runs `set_rotation_mixing_info` (`star/private/rotation_mix_info.f90:47-1123`) and requires every face value of `D_visc`, `D_DSI`, `D_SH`, `D_SSI`, `D_ES`, `D_GSF`, `D_ST`, `nu_ST`, `richardson_number`, `omega_shear` and `domega_dlnR` to be bit-identical to MESA.

## What the call covers

The case is `star/test_suite/20M_z2m2_high_rotation` (inlist `inlist_to_end_core_he_burn_header`), `nz = 1748`. The kept call is the first `update_rotation_mixing_info` call with $\Omega \neq 0$ that is not part of a relax, not the first model of the run, and has `prev_mesh_have_ST_start_info` set, so the Spruit-Tayler time smoothing (`rotation_mix_info.f90:191-215`) runs. The case uses `D_SH`, `D_SSI`, `D_ES`, `D_GSF` and `D_ST` factors of 1, `D_DSI_factor = 0`, and `smooth_D_ST = smooth_nu_ST = 5`.

The fixture also records `D_mix`, `D_mix_rotation` and the `am_nu_*` arrays at exit. The test does not compare them yet, because this case sets `D_omega_flag`, which is ported separately. `kap_cond` (column 36 of `IN`) is MESA's face electron-conduction opacity, recorded for diagnosis; the routine computes its own.

## Format

Lines starting with `#` are the provenance header and column legend. `HDR` gives the call number, model number, `nz`, three logical flags and `R_center`, `mstar_old`, `dt`. Each `IN` row holds the 41 input values of one face or cell `k` (1-based, surface first); each `RMX` row holds the 18 output values of face `k`. Floating-point values are big-endian IEEE-754 binary64 raw bits (`Z16.16`).

## Regenerating

1. Make an overlay of the MESA `26.4.1` tree with real copies of `star/`, `lib/` and `include/` and symlinks for everything else, so the pristine build is not touched. Set `MESA_DIR` to the overlay.
2. Apply the patch from the overlay root: `patch -p1 < instrumentation.patch`. Rebuild and install `star` (`cd star && ./mk && ./export`).
3. Copy `star/test_suite/20M_z2m2_high_rotation` to a work directory, set `max_model_number = 4` in `inlist_to_end_core_he_burn` (line 71) (the dumped call is on model 2, so the cap does not change it), build it against the overlay and run `./rn`. The run writes `rotmix-in.txt` and `rotmix-out.txt` in the work directory.
4. Trim: `python make_fixture.py rotmix-in.txt rotmix-out.txt 1 mesa_26_4_1_rotmix_20M_model2.txt`.

The patch is the instrumentation used to produce the committed fixture with two cosmetic differences: the helper routines are named `rotmix_oracle_*` and the dump files are written to the work directory, where the original build used other routine names and absolute output paths. Neither touches any value MESA computes.
