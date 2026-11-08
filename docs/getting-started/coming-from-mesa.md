# Coming from Fortran MESA

If you have an existing workflow built around upstream Fortran MESA (`r24.08.1` or earlier), `FlexMESA` preserves MESA's physical formulations, control names, `.mod` file formats, and pulsation exports (`GYRE`, `FGONG`, `OSC`) while replacing Fortran namelist executables, 1-based global module state, `run_star_extras.f90` hooks, and X11 `PGSTAR` windows with a modular Safe Rust and Python/JAX library. This guide provides a side-by-side translation reference.

---

## 1. Workflow Comparison at a Glance

| Task | Upstream Fortran MESA | FlexMESA (Safe Rust & Python/JAX) |
| :--- | :--- | :--- |
| **Build a work directory** | Copy `$MESA_DIR/star/work`, edit `makefile`, run `./mk` (`gfortran`) | `import flexmesa` in any Python script or notebook (no per-run compilation) |
| **Configure controls** | Text files `inlist`, `&star_job`, `&controls`, `&binary_controls` | Typed Python methods/dataclasses (`StarModel.from_zams`, `set_evolve_controls`) or validated namelist parsing via `flexmesa.schema.namelist` |
| **Load / save models** | `load_saved_model = .true.`, `save_model_filename = 'foo.mod'` | `StarModel.from_mod_file("foo.mod")` and `model.write_mod_file("foo.mod")` |
| **Binary restarts (`photos/`)** | Unformatted Fortran binary `photos/x100` (compiler-dependent) | Portable deterministic byte buffers via `model.to_photo_bytes()` and `StarModel.from_photo_bytes(data)` |
| **Custom physics hooks** | Edit `src/run_star_extras.f90` (`extras_check_model`, `other_wind`, `other_mlt`) and recompile | Standard Python control loops over `model.evolve_step()` or differentiable `@jax.jit` functions |
| **Live / post-run plotting** | `PGSTAR` (`&pgstar`, X11 server + PGPLOT) | Publication-styled `matplotlib` functions in `flexmesa.plotting` (`plot_hr_diagram`, `plot_kippenhahn`, `plot_echelle_diagram`) |
| **Sensitivity / derivatives** | Finite differences across multiple `./rn` directories or internal 33-slot `auto_diff_real_star_order1` | Direct `jax.grad`, `jax.jacfwd`, `jax.vmap`, or const-generic `DualOrder1<N>` in Safe Rust |

---

## 2. Mapping Fortran `inlist` Namelists to `FlexMESA`

In Fortran MESA, run configuration is split across four namelist blocks (`&star_job`, `&controls`, `&binary_controls`, and `&astero_search_controls`). In `FlexMESA`, you can either parse and validate existing Fortran `inlist` files directly using `flexmesa.schema` or configure models programmatically in Python and Rust.

### Parsing Existing Fortran `inlist` Files (`flexmesa.schema`)
If you have existing MESA `inlist` files, `flexmesa.schema.namelist` parses Fortran `&section ... /` syntax (including `!` comments, `.true.`/`.false.` booleans, `1.0d-4` double-precision literals, and 1-based array assignments such as `xa_central_lower_limit_species(1) = 'h1'`) and validates every key against `.defaults` schemas:

```python
from flexmesa.schema import defaults, namelist

defs = defaults.parse_defaults_text("""
initial_mass = 1.0d0
initial_z = 0.02d0
mixing_length_alpha = 2.0d0
varcontrol_target = 1.0d-4
""")

blocks = namelist.parse_namelist_text("""
&controls
  initial_mass = 4.0d0
  initial_z = 0.014d0
  mixing_length_alpha = 1.82d0
  varcontrol_target = 5.0d-5 ! tighter H211 timestep control
/
""")
validated = namelist.validate_namelist_block(blocks[0], defs)
print(validated)
```

### Side-by-Side Namelist Control Mapping

| Fortran Namelist & Option | FlexMESA Equivalent |
| :--- | :--- |
| **`&star_job`** | |
| `create_pre_main_sequence_model = .true.` | `StarModel.from_pre_ms(initial_mass=..., initial_z=..., t_c=9.0e5)` |
| `load_saved_model = .true.` / `load_model_filename` | `StarModel.from_mod_file(path)` or `StarModel.from_mod_str(text)` |
| `save_model_when_terminate = .true.` | `model.write_mod_file(path)` or `model.to_mod_str()` |
| `change_net = .true.` / `new_net_name = 'basic.net'` | `flexmesa.net.get_net_info("basic.net")` (embedded in `StarModel`) |
| `save_pulse_data_when_terminate = .true.` (`GYRE` / `FGONG` / `OSC`) | `model.to_gyre_str()`, `model.to_fgong_str()`, `model.to_osc_str()` |
| **`&controls`** | |
| `initial_mass`, `initial_z`, `initial_y` | `StarModel.from_zams(initial_mass=..., initial_z=..., initial_y=...)` |
| `max_model_number`, `max_age` | `model.evolve(max_steps=..., max_age=...)` |
| `mesh_delta_coeff`, `varcontrol_target` | `model.set_evolve_controls(mesh_delta_coeff=1.0, varcontrol_target=1.0e-4)` |
| `overshoot_scheme`, `overshoot_f`, `overshoot_f0` | `model.update_mixing(dt, overshoot_f=..., overshoot_f0=..., scheme="exponential")` |
| `cool_wind_RGB_scheme`, `hot_wind_scheme`, `Dutch_scaling_factor` | `model.apply_wind_and_adjust_mass(scheme="Dutch", eta=1.0, dt=dt)` |
| `do_element_diffusion = .true.` | `model.do_element_diffusion(dt, scheme="stanton_murillo")` |
| `do_conv_premix = .true.` | `model.do_conv_premix(max_iters=10, avoid_fuel_increase=True)` |
| `predictive_mix = .true.` | `model.add_predictive_mixing(ingested_mass_limit_coeff=0.1, superad_thresh=0.005)` |
| `do_phase_separation = .true.` | `model.do_phase_separation(dt, gamma_crystallization=175.0)` |
| **`&binary_controls`** | |
| `m1`, `m2`, `initial_period_in_days`, `initial_eccentricity` | `BinarySystem.from_point_mass(donor, companion_mass_msun=..., initial_period_days=..., eccentricity=...)` or `BinarySystem.from_two_stars(donor, accretor, initial_period_days=...)` |
| `mdot_scheme = 'Ritter'` / `'Kolb'` | `binary.eval_ritter_mdot_thin`, `binary.eval_kolb_mdot_thick`, `binary.solve_roche_lobe_mdot_jax` |
| `do_jdot_gr`, `do_jdot_mb`, `do_tidal_sync` | `binary.eval_jdot_gr_jax`, `binary.eval_jdot_mb_jax`, `binary.eval_edot_tidal_hut_jax` |
| **`&astero_search_controls`** | |
| `search_type = 'simplex'` / `'bobyqa'` / `'scan_grid'` | `astero.nelder_mead_minimize`, `astero.powell_quadratic_minimize`, `astero.scan_cartesian_grid` |
| `correction_scheme = 'kjeldsen'` / `'cubic'` / `'combined'` / `'sonoi'` | `astero.eval_surface_correction(scheme, nu_obs, sigma, nu_mod, inertia, nu_max=...)` |

---

## 3. Loading & Saving `.mod` Files and `.photo` Checkpoints

`FlexMESA` reads and writes upstream MESA ASCII `.mod` files directly, allowing you to start a `FlexMESA` calculation from a converged Fortran MESA model or inspect a `FlexMESA` model with external MESA post-processing scripts:

```python
from flexmesa import star

# Export a ZAMS model to an upstream-compatible ASCII .mod string and roundtrip it
model = star.StarModel.from_zams(initial_mass=2.0, initial_z=0.02)
mod_text = model.to_mod_str()
reloaded = star.StarModel.from_mod_str(mod_text, eval_hydro=True)
assert reloaded.nz == model.nz

# Save and restore an exact binary restart checkpoint (.photo)
photo_bytes = model.to_photo_bytes()
restored = star.StarModel.from_photo_bytes(photo_bytes, eval_hydro=True)
```

---

## 4. The Index and Memory-Order Contract (`1..nz` vs. `0..nz-1`)

When porting formulas or `run_star_extras.f90` diagnostics from Fortran MESA to `FlexMESA`, keep two layout rules in mind:

1. **Outer-to-Inner Radial Ordering is Preserved (`surface -> center`)**:
   Both Fortran MESA and `FlexMESA` order radial shells from the outer surface (photosphere) to the stellar center. However, Fortran uses **1-based indexing (`k = 1 .. nz`)** whereas `FlexMESA` uses **0-based indexing (`k = 0 .. nz - 1`)**:
   - **Surface shell**: `k = 1` in Fortran $\Longleftrightarrow$ `k = 0` (`arr[0]`) in `FlexMESA`.
   - **Central shell**: `k = s% nz` in Fortran $\Longleftrightarrow$ `k = model.nz - 1` (`arr[-1]`) in `FlexMESA`.
   - **Interior face $k$**: separates outer zone `k - 1` from inner zone `k` for `k in 1 .. nz - 1` (with `k = 0` at the outer surface boundary).
2. **Column-Major (`F`) vs. C-Contiguous Row-Major (`C`) Memory Layout**:
   Fortran MESA stores multi-dimensional arrays in column-major order with the zone index last (for example, `s% xa(1:species, 1:nz)` and `lblk(1:nvar, 1:nvar, 1:nz)`). `FlexMESA` stores all arrays in **C-contiguous row-major order with the zone index first** so that per-zone slices are contiguous in cache and compatible with `jax.vmap`:

| Array | Upstream Fortran MESA Shape | FlexMESA Shape (`C`-Contiguous) |
| :--- | :--- | :--- |
| Species mass fractions `xa` | `(species, nz)` (`s% xa(j, k)`) | `(nz, species)` (`model.xa[k, j]`) |
| 1D cubic spline coefficients `f` | `(4, n)` (`f(1:4, i)`) | `(n, 4)` (`f[i, 0:4]`) |
| 2D bicubic spline coefficients | `(4, 4, nx, ny)` | `(nx, ny, 4, 4)` |
| Henyey block-tridiagonal `lblk, dblk, ublk` | `(nvar, nvar, nz)` | `(nz, nvar, nvar)` (`lblk[0] = 0`, `ublk[nz-1] = 0`) |
| Henyey residual / RHS vector `equ` | `(nvar, nz)` | `(nz, nvar)` |

---

## 5. Replacing `run_star_extras.f90` with Python & JAX Callbacks

In Fortran MESA, customizing a run requires copying `standard_run_star_extras.inc` into `src/run_star_extras.f90`, assigning procedure pointers (`s% extras_startup`, `s% extras_check_model`, `s% extras_finish_step`, `s% other_wind`), and recompiling the executable.

In `FlexMESA`, the evolutionary loop is a regular Python or Rust loop over `model.evolve_step()`. You can inspect or modify the stellar state before or after any step, apply custom termination criteria, or evaluate differentiable JAX physics hooks directly on `model.to_jax_state()`:

```python
from flexmesa import star

model = star.StarModel.from_zams(initial_mass=4.0, initial_z=0.02)
history_rows = []

for step_idx in range(1, 51):
    # Equivalent to extras_before_struct_burn_mix: apply winds or custom mixing
    mdot_wind = model.apply_wind_and_adjust_mass(scheme="Dutch", eta=0.8, dt=model.dt)

    # Advance one coupled Henyey step
    report = model.evolve_step(do_remesh=False, op_split_burn=False, use_bcyclic=True)

    # Equivalent to extras_finish_step / extras_check_model
    row = model.history_row()
    history_rows.append(row)
    if row.center_h1 < 0.65:
        print(f"Stopping at step {step_idx}: central H1 reached {row.center_h1:.4f}")
        break
```

---

## 6. Replacing `PGSTAR` Windows with `flexmesa.plotting`

Upstream MESA's `PGSTAR` module (`~26,000` lines of Fortran wrapping PGPLOT and X11) renders live diagnostic dashboards during a terminal run. `FlexMESA` replaces `PGSTAR` with [`flexmesa.plotting`](../api/index.md) (`from flexmesa import plotting as plot`), which produces publication-styled `matplotlib` figures (`300 dpi`, Computer Modern math typography, inward ticks on all four sides, and colorblind-safe Okabe-Ito palettes) directly from NumPy or JAX arrays:

| `PGSTAR` Window | `flexmesa.plotting` Function |
| :--- | :--- |
| `HR_Window` (`log_Teff` vs. `log_L`) | `plot.plot_hr_diagram(log_teff, log_l, color_val=center_h1)` |
| `TRho_Profile_Window` / `TRho_Window` (`log_center_Rho` vs. `log_center_T`) | `plot.plot_tc_rhoc_diagram(log_rhoc, log_tc, show_burn_lines=True)` |
| `Kipp_Window` (convective zones & $\epsilon_{\mathrm{nuc}}$ vs. age) | `plot.plot_kippenhahn(age_myr, mass_grid_msun, mix_type_grid, eps_nuc_grid)` |
| `Profile_Panels` ($N^2$ and $S_l^2$ seismic propagation diagram) | `plot.plot_propagation_diagram(r_over_rstar, brunt_n2, csound, r_cm)` |
| Asteroseismic echelle & separation ratios | `plot.plot_echelle_diagram(...)` and `plot.plot_separation_ratios(...)` |
| `pgbinary` orbital & RLOF panels | `plot.plot_binary_evolution(age_myr, period_days, lg_mtransfer_rate, eccentricity)` |
