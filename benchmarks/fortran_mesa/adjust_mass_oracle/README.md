# MESA 26.4.1 `adjust_mass` reference (`#385`)

`adjust_mass_4msun_26_4_1.tsv` records, per zone, `q`, `dq`, `dm` and the H1 mass fraction of a 4 Msun, Z = 0.02 ZAMS model before and after one call of upstream `do_adjust_mass` (`star/private/adjust_mass.f90`). The model carries a composition step (H1 −0.10, He4 +0.10 for 0.95 < q ≤ 0.98), and the mass loss is `mstar_dot = -0x1.0640eab8de93dp+52` g/s over `dt = 0x1.cb39389b80000p+44` s. Values are IEEE-754 binary64 bit patterns in hexadecimal.

The setup matches `tests/test_star_wave4_3.py::test_stellar_winds_rust_jax_consistency_and_mass_adjustment`. That test calls FlexMESA's `do_adjust_mass` with explicit region limits (q ≤ 0.99 for `kA`, q ≤ 0.95 for `kB`). The MESA run reproduces them with `min_q_for_k_below_const_q = 0.99` and `min_q_for_k_const_mass = 0.95`, with the `max_logT_*` limits raised to 99 so that temperature does not end the search (`inlist_project`).

## Recipe

1. Copy `$MESA_DIR/star/work` to a scratch directory, and put `inlist_project` there.
2. Copy `$MESA_DIR/include/standard_run_star_extras.inc` to `src/h1_extras.inc`, replace its `extras_startup` with the one in `run_star_extras_h1.inc`, and make `src/run_star_extras.f90` include `h1_extras.inc`.
3. Build with the private module files on the include path, and run:

   ```bash
   make -C make SPECIAL_FC_FLAGS="-I$MESA_DIR/star/make"
   ./rn
   ```

   `extras_startup` writes `h1_zones.txt` (`BEFORE` and `AFTER` rows) and stops. The table is that file with this header added.

The work directory compiles against the pristine MESA tree, which it only reads. The added code makes the call and writes values; it changes no MESA routine.
