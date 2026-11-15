---
paths:
  - "crates/**/*.rs"
  - "src/flexmesa/**/*.py"
  - "tests/**/*.py"
  - "benchmarks/**/*"
---

# End-to-End (`test_suite`) Porting, Checkpointing & Debugging Protocol

Whenever implementing, auditing, or debugging multi-step stellar, asteroseismic, or binary evolution campaigns (`star/test_suite`, `binary/test_suite`, `astero/test_suite`, `dev_cases_*`), follow these five engineering rules without exception (`DECISIONS #9` & `#17`):

---

## 1. Checkpoint-First Isolation for Long Campaigns (`> 60 s`)

- **At Most One Full Blind Run**: Never execute a campaign taking $> 60\,\mathrm{s}$ end-to-end from step 1 more than once to diagnose a mid-evolution convergence failure, retry cascade, or stopping-target discrepancy.
- **Extract a `< 2 s` Pre-Failure Fixture**: Immediately after the first failing run, serialize a `.photo` / `.mod` state (or `BinarySystem` snapshot) `2` to `5` steps before the anomalous transition and build a `< 2 s` single-step reproduction test that logs:
  1. Exact step number, `dt`, `dt_next`, and retry/limit reason (`why_Tlim` / `TimestepLimitReason`).
  2. Maximum Newton residual and correction `(eqn, zone k, value, eqn_scale)` across iterations.
  3. Pre-solver vs. post-solver `xa - xa_start`, convective boundaries, `d_mix`, `sig / dm`, and $\Omega(r)$ profiles.
- **Verify Every Hypothesis on the `< 2 s` Fixture First**: Re-run the full multi-minute campaign from step 1 **only after** the `< 2 s` single-step transition and crate unit tests pass cleanly.

---

## 2. Upstream Step-Lifecycle Audit Before Local Solver Edits

- **Distinguish Downstream `f64` Quantization Traps from Upstream Retry Triggers**: In stiff convective cores (`sig / dm = sig_term_limit / dt = 1e13 / dt`, `min_chem_eqn_scale = 1.0`), a `1-ULP` IEEE-754 `f64` difference across adjacent cells yields a chemical residual of $(10^{13}/\Delta t)\,\Delta X_{\mathrm{ULP}}$, creating a hard `f64` ceiling around $\Delta t_{\mathrm{crit}} \sim 10^2\,\mathrm{s}$ under `gold2` tolerances (`1e-5`) in both Fortran MESA (`real(dp)`) and Safe Rust (`f64`). Never introduce non-MESA `eqn_scale` floors, row-equilibration hacks, or `dt`-dependent `sig_term_limit` caps to survive inside an unphysical low-`dt` trap; instead, diagnose why `dt` collapsed from evolutionary timescales in the first place.
- **Audit Full Step Ordering Against Upstream Fortran MESA**: Before modifying `hydro_eqns` or `solver`, verify exact ordering parity across:
  1. `evolve.f90` pre-solver hooks (`do_adjust_mesh`, `do_adjust_mass`, `calculate_eps_mdot`, `do_conv_premix`, `do_phase_separation`, `do_element_diffusion`, pre-solver `0.5 * dt` `premix_omega` `do_solve_omega_mix`, `set_rotation_info`, `set_vars_if_needed`).
  2. `struct_burn_mix.f90` (`do_burn` operator-split burning, `get_convection_sigmas`, `j_rot_start`, **then** `save_start_values`, **then** `do_solver_converge`).
  3. `solver_support.f90` (`xa_sub_xa_start` incremental tracking and `abs(sum_xa - 1d0) > 1d-12` renormalization threshold).
  4. Post-solver `do_mix_omega` (`0.5 * dt` when `premix_omega` else `dt`), `set_final_vars`, energy conservation, and outer binary implicit-RLOF / hard-limit retry loops (`binary_evolve.f90`, without clobbering `*_start` arrays mid-trial).

---

## 3. Three-Fix Stop & Incremental GitHub Checkpoints

- **Push Working Baselines Early**: Push clean, compiling milestones to the `<issue>-<slug>` branch and open a Draft PR (`Closes #<issue>`) before starting long uncapped E2E sweeps so working progress is backed up on GitHub and verified continuously by CI.
- **Three-Fix Stop**: If three targeted fixes on the `< 2 s` checkpoint harness fail to resolve the issue or reveal deeper cross-module coupling, stop coding, push any clean prerequisite commits, and document the full root-cause and dependency analysis in a GitHub issue before attempting a fourth fix.

---

## 4. Genuine Multi-Step Henyey Relaxation for `relax_initial_*`

- Whenever an upstream `inlist` requests initial model relaxation (`relax_initial_Z`, `relax_initial_Y`, `relax_initial_mass`, `relax_M_center`, `relax_R_center`, `relax_L_center`, `relax_tau_factor`, `relax_irradiation`), perform multi-step pseudo-evolutionary Henyey equilibration (`doing_relax = true`, `dxdt_nuc_factor = 0`, `Cox` MLT, wind/mass-loss disabled) so $(T, \rho, R, L)$ adjust smoothly to the target opacity, equation of state, and boundary conditions rather than overwriting `model.xa` or `model.mstar` algebraically in a single step without solving the structure equations.

---

## 5. Strict `StarExtras` & `BinaryExtras` Trait Isolation

- Upstream test-case customizations in `src/run_star_extras.f90` and `src/run_binary_extras.f90` (custom wind recipes, step hooks, RLOF switches, bespoke stopping criteria) must be implemented through `StarExtras` and `BinaryExtras` trait hooks in dedicated `e2e` or test-suite modules.
- Core physics and evolution modules (`flexmesa-star` solver/step drivers and `flexmesa-binary` orbital/RLOF step drivers) must never branch on `case_name` strings (`"double_bh"`, `"wind_fed_bhhmxb"`, etc.).
