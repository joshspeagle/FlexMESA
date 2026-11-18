# User Guide: Backends, JAX Contract & Custom Gradients

`FlexMESA` separates its execution stack into a canonical **Safe Rust** engine (exposed to Python through zero-copy `PyO3` bindings) and a derived **64-bit JAX** layer (`float64`) for automatic differentiation and vectorized batching (`#165`, `#166`).

| Route | Role | Validated against | Device scope |
| :--- | :--- | :--- | :--- |
| **Rust, `"strict"` mode** *(default)* | Reference port, identical across `x86_64` / `aarch64` and `1` / `4` threads (`#165`) | Real MESA `26.4.1` | CPU |
| **Rust, `"fast"` mode** | Same Rust source compiled with hardware `std` math and `mul_add` (`#165`) | Rust `"strict"`, within tolerance | CPU |
| **JAX (`backend="jax"`)** | Differentiable and batched kernels (hardware-dependent via XLA) (`#166`) | Rust `"strict"`, within tolerance | Per-module (see Section 4) |

Rust `"strict"` mode is canonical, and JAX is derived from it. Cross-backend consistency between Rust and JAX never serves as evidence of agreement with upstream MESA. The `star`, `binary`, and `astero` JAX helper modules are **frozen and unverified** (`JAX_ENGINE_STATUS = "frozen-unverified-until-#143"`) until the canonical Rust engine passes the `#143` upstream history harness.

---

## 1. Backend Dispatch: `backend="auto" | "rust" | "jax"`

Every public microphysics, spline, binary, and asteroseismic routine with dual implementations accepts a keyword-only `backend` selector (`"auto"`, `"rust"`, or `"jax"`) routed through `flexmesa.resolve_backend`:

- **`backend="auto"` (Default)**: Inspects all numeric, array, and nested `tuple` / `list` / `dict` leaf arguments without copying arrays. If any leaf is a `jax.Array` or an active JAX tracer (inside `@jax.jit`, `jax.grad`, `jax.jacfwd`, or `jax.vmap`), `FlexMESA` dispatches to the derived JAX path. Otherwise, it passes Python floats or C-contiguous NumPy `float64` views to `flexmesa._rust`.
- **`backend="rust"`**: Forces execution on the Safe Rust CPU engine (releasing the Python GIL during multi-zone `parallel=True` `rayon` loops).
- **`backend="jax"`**: Forces execution through the JAX kernel even when given Python floats or NumPy arrays (subject to each package's host-callback and device constraints listed in Section 4).

```python
import jax.numpy as jnp
import flexmesa
from flexmesa import eos

# Verify that 64-bit IEEE-754 precision is active in JAX
assert flexmesa.enable_float64() is True

# 1. NumPy / float inputs route automatically to Safe Rust (`backend="auto"`)
print("Resolved backend for float:", flexmesa.resolve_backend("auto", 1.0e7, 1.0e2))
out_rust, _, _ = eos.eval_helm_eos(1.0e7, 1.0e2, 1.3, 1.1, backend="auto")

# 2. jax.Array inputs route automatically to @jax.jit (`backend="auto"`)
t_jax = jnp.float64(1.0e7)
print("Resolved backend for jax.Array:", flexmesa.resolve_backend("auto", t_jax, 1.0e2))
out_jax, _, _ = eos.eval_helm_eos(t_jax, jnp.float64(1.0e2), 1.3, 1.1, backend="auto")

print(
    f"Rust strict vs JAX ln(P_gas) agreement: "
    f"|diff| = {abs(float(out_rust.res[0]) - float(out_jax.res[0])):.2e}"
)
```

### Rust Numerics Modes: `"strict"` (Default) vs. `"fast"` (`#165`)

Within the canonical Rust engine, all physics crates call `flexmesa-math` and support two run-level arithmetic modes built from the same Rust physics source:

- **`"strict"` (Default)**: Calls vendored `crlibm 1.0beta4` (`CRMATH 1.2`) without fused multiply-add contraction (`a * b + c`) and preserves upstream operation order. `"strict"` produces bit-identical results across `x86_64` and `aarch64` hardware and across `1` and `4` `rayon` threads (Level 1), and is the **only** mode judged against MESA `26.4.1` (Levels 2 and 3).
- **`"fast"`**: Uses platform `std` `f64` intrinsics and hardware `mul_add`. `"fast"` is hardware-dependent, is validated exclusively against `"strict"` (`notes/audit/fast_vs_strict.md`), and is never compared against MESA. Similarly, `backend="jax"` is a derived layer for automatic differentiation and GPU batching validated against Rust `"strict"` (`#166`).

You can inspect or switch the Rust numerics mode in Python (`flexmesa.get_numerics_mode()`, `flexmesa.set_numerics_mode("fast")`, `with flexmesa.numerics_mode_context("fast"): ...`) or on the CLI (`flexmesa trace ... --numerics-mode fast`), and the active mode is recorded in every emitted `LOGS/history.data`, `LOGS/profileN.data`, `binary_history.data`, `.mod`, and `.photo` file.

The numerics mode is one setting for the whole process, so a process runs in one mode at a time. Switching it while another thread is evaluating physics would mix `"strict"` and `"fast"` arithmetic inside that calculation. To compare the two modes, run them one after the other or in separate processes (`DECISIONS #23`).

---

## 2. Multi-Zone CPU Parallelism (`parallel=True` & `RAYON_NUM_THREADS`)

For multi-zone stellar profiles ($N_{\mathrm{zones}} \sim 10^3\text{ to }10^4$), calling scalar Python functions in a `for` loop incurs per-zone interpreter overhead. All `*_zones` evaluators in `FlexMESA` (`eos.eval_helm_eos_zones`, `eos.eval_skye_eos_zones`, `eos.eval_pc_eos_zones`, `kap.kap_get_zones`, `chem.get_composition_info_zones`, `colors.eval_colors_zones`) accept 1D/2D NumPy arrays and a `parallel: bool = True` flag that distributes zones across the persistent Rust `rayon` thread pool with the Python GIL released.

You can control the CPU thread count via environment variables prior to starting Python:

```bash
export RAYON_NUM_THREADS=8
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
```

```python
import numpy as np
from flexmesa import eos

nz = 2048
temps = np.geomspace(1.0e5, 5.0e7, nz, dtype=np.float64)
rhos = np.geomspace(1.0e-3, 1.0e3, nz, dtype=np.float64)
abars = np.full(nz, 1.30, dtype=np.float64)
zbars = np.full(nz, 1.10, dtype=np.float64)

# Evaluate all 2,048 radial shells in parallel across the Rust rayon pool
helm_zones = eos.eval_helm_eos_zones(
    temps, rhos, abars, zbars, parallel=True, backend="rust"
)
print(
    f"Evaluated {nz} HELM zones: res shape = {helm_zones.res.shape}, "
    f"surface ln(P_gas) = {helm_zones.res[0, 0]:.4f}, center ln(P_gas) = {helm_zones.res[-1, 0]:.4f}"
)
```

---

## 3. Choosing Block-Tridiagonal Henyey Solvers (`thomas`, `bcyclic`, `pcr`, `rust_ffi`)

Inverting the block-tridiagonal Henyey Jacobian ($N_{\mathrm{zones}}$ blocks of size $N_{\mathrm{var}} \times N_{\mathrm{var}}$) is the primary linear-algebra bottleneck in 1D stellar evolution. `flexmesa.mtx` and `flexmesa.star` provide four solver strategies tailored to different hardware and differentiation regimes:

| Solver Option | Entry Point | Complexity / Parallelism | Recommended Use Case |
| :--- | :--- | :--- | :--- |
| **`method="thomas"`** | `mtx.solve_block_tridiagonal(..., method="thomas")` | Sequential $\mathcal{O}(N_{\mathrm{zones}} N_{\mathrm{var}}^3)$ forward/backward sweep | Small grids ($N_{\mathrm{zones}} \lesssim 256$) or single-core runs |
| **`method="bcyclic"`** | `mtx.solve_block_tridiagonal(..., method="bcyclic", parallel=True)` / `model.evolve(..., use_bcyclic=True)` | Hirshman et al. (2010) parallel odd-even cyclic reduction on `rayon` | Default multi-core CPU solver for $N_{\mathrm{zones}} \ge 512$ |
| **`solver_mode="pcr"`** | `mtx.solve_block_tridiagonal_jax(..., solver_mode="pcr")` | $\lceil \log_2 N_{\mathrm{zones}} \rceil$ vectorized Parallel Cyclic Reduction in JAX | Batched GPU forward/reverse-mode solves (`jax.vmap`) |
| **`solver_mode="rust_ffi"`** | `mtx.solve_block_tridiagonal_jax(..., solver_mode="rust_ffi")` | `@jax.custom_vjp` calling Rust `bcyclic` for $\mathbf{J}\mathbf{x}=\mathbf{b}$ and $\mathbf{J}^\top\boldsymbol{\lambda}=\bar{\mathbf{x}}$ | CPU-hosted JAX differentiation with minimal memory footprint |

```python
import jax.numpy as jnp
import numpy as np
from flexmesa import mtx

nz, nb = 128, 5
lblk = np.broadcast_to(-0.25 * np.eye(nb), (nz, nb, nb)).copy()
dblk = np.broadcast_to(2.50 * np.eye(nb), (nz, nb, nb)).copy()
ublk = np.broadcast_to(-0.25 * np.eye(nb), (nz, nb, nb)).copy()
rhs = np.ones((nz, nb), dtype=np.float64)

x_thomas = mtx.solve_block_tridiagonal(lblk, dblk, ublk, rhs, method="thomas")
x_bcyclic = mtx.solve_block_tridiagonal(
    lblk, dblk, ublk, rhs, method="bcyclic", parallel=True
)
x_pcr = mtx.solve_block_tridiagonal_jax(
    jnp.asarray(lblk),
    jnp.asarray(dblk),
    jnp.asarray(ublk),
    jnp.asarray(rhs),
    solver_mode="pcr",
)
print(
    f"Max |Thomas - bcyclic| = {np.max(np.abs(x_thomas - x_bcyclic)):.2e}, "
    f"Max |Thomas - JAX PCR| = {np.max(np.abs(x_thomas - np.asarray(x_pcr))):.2e}"
)
```

![Wave 5 four-way runtime comparison against Reference Fortran comparators](../assets/figures/wave5_benchmark_speedup.png)

*Figure 1: MESA `r26.4.1` raw capture Wave 5 kernel runtime comparison (`benchmarks/raw/26.4.1/`) at $N_{\mathrm{zones}} = 1024$ (left) and zone scaling (right) across Safe Rust (`strict`), PyO3, JAX (`float64`), and the `bench_kernels_wave5.f90` Reference Fortran comparator (`w5_diff_rot`, `w5_binary_rlof`, and `w5_astero_seism` hybrid MESA helpers; `w5_riemann_rsp2` non-MESA; `w5_diff_rot` diverges by `99.17%` from Reference Fortran, tracked in `#158`).*

---

## 4. Per-Module JAX Status, Differentiability & Device Support

The table below documents what the JAX side of each package actually implements, which arguments and AD modes are supported, and whether execution is pure XLA or bound to CPU host callbacks into `flexmesa._rust`:

| Package | Implementation on Python/JAX side | Differentiable through & AD modes | Devices |
| :--- | :--- | :--- | :--- |
| `flexmesa.constants` | Scalar constants mirroring `flexmesa-const` | Non-differentiable scalars | CPU |
| `flexmesa.math` | Independent `@jax.jit` `float64` intrinsics and double-`jnp.where` safe guards | All numeric arguments (forward `jvp` and reverse `vjp`) | CPU, GPU/TPU |
| `flexmesa.utils` | Rust `PyO3` tokenizers/parsers + `jnp.where` array guard (`replace_bad_nums`) | `replace_bad_nums` wrt `values` (forward + reverse); parsers non-AD | CPU, GPU/TPU (`replace_bad_nums`), CPU (`_rust`) |
| `flexmesa.auto_diff` | Python registered JAX PyTree dual/hyperdual types (`DualOrder1`, `AutoDiffReal2VarOrder2`); canonical engine uses Rust `flexmesa-autodiff` | Python scalar and JAX array unary/binary ops (forward/reverse) | CPU, GPU/TPU (when driven with JAX arrays) |
| `flexmesa.interp_1d` | JAX-traceable Horner spline evaluation (`interp_value`, `interp_value_and_slope`, `interp_values`, `interp_values_and_slopes` in `eval.py`); knot builders (`pm`, `mp`), `integrate_values`, and `remesh` run host NumPy on CPU | `eval.py` point/array evaluators wrt query `xs` and `coeffs` (forward + reverse); builders, `integrate_values`, and `remesh` non-AD | CPU, GPU/TPU (`eval.py` evaluators), CPU (builders, quadrature, remesh) |
| `flexmesa.interp_2d` | JAX-traceable bicubic evaluators (`bicub_eval.py`); builders (`bicub`, `bipm`, `akima`, `renka790`) run host NumPy on CPU | `interp_evbicub_db` and `interp_evbicub_mask_db` wrt `(xget, yget)` and `coeffs` (forward + reverse); builders non-AD | CPU, GPU/TPU (`bicub_eval`), CPU (builders) |
| `flexmesa.mtx` | Independent JAX `pcr` solver + `@jax.custom_vjp` `rust_ffi` wrapper (`bcyclic` forward and transpose solve) | `(lblk, dblk, ublk, rhs)` (forward + reverse on `pcr`; reverse `vjp`/`grad` only on `rust_ffi`) | CPU, GPU/TPU (`pcr`), CPU (`rust_ffi`) |
| `flexmesa.num` | Python callable utilities (`root`, `rk`, `rosenbrock`, `brent`, `simplex`, `newton`) + eager `@jax.custom_vjp` `safe_root_jax` / `newton_solve_jax`; canonical solver engine uses `flexmesa-num` | `safe_root_jax` and `newton_solve_jax` wrt problem parameters (eager reverse `vjp`/`grad` only; cannot be nested inside `jax.jit` or `jax.vmap`) | CPU (NumPy/SciPy helpers and eager custom-VJP host solvers) |
| `flexmesa.chem` | Independent `@jax.jit` composition moments (`get_composition_info_jax`, `get_composition_info`); catalogs delegate to Rust | Mass fractions `x` (forward + reverse, once cached species properties load from Rust) | CPU, GPU/TPU (kernels), CPU (catalog queries) |
| `flexmesa.ionization` | Independent `@jax.jit` splines/polynomials (`fe56_he4`, `tables`, `chi_info_jax`, `chi_effective_jax`) using tables cached from Rust | `(log10_ne, log10_t)`, `(z, x, log10_rho, log10_t)`, and `(chi, c0..c2, z1, z2)` (forward + reverse); `eval_typical_charge` step has zero AD gradient | CPU, GPU/TPU (after table load) |
| `flexmesa.colors` | Independent `@jax.jit` 3D Hermite/Linear/KNN SED and filter integration using Kurucz/Vega grid cached from Rust | `(teff, log_g, metallicity, r, d)` and flux/filter arrays (forward + reverse) | CPU, GPU/TPU (after grid load) |
| `flexmesa.rates` | Independent `@jax.jit` screening (`Salpeter`, `Extended`, `Chugunov`), `3-alpha`, and REACLIB term evaluator; weak/tabulated rates are Rust-only | `(temp, den, zbar, abar, z2bar)` / `t9` / `y_he4` (forward + reverse); weak tables non-AD | CPU, GPU/TPU (JAX kernels), CPU (weak/raw tables) |
| `flexmesa.neu` | Independent `@jax.jit` 5-channel Itoh (1996) kernels and `@jax.custom_jvp` `ifermi12_jax`; `neu_get(..., backend="jax")` executes `_jax.neu_get_jax` | `(t, rho, abar, zbar)` and chemical potential inputs (forward `jacfwd` + reverse `grad`) | CPU, GPU/TPU (`_jax`, `neu_get(..., backend="jax")`) |
| `flexmesa.eos` | Independent `@jax.jit` `Ideal`, `HELM`, `PC`, `Skye`, and `C^2` blend formulas + `@jax.custom_jvp` inverse wrappers; table cell/vertex stencils call Rust `pure_callback` under `stop_gradient` | `eval_ideal_eos_from_weights`, `eval_helm_eos`, and `blend_eos_results` wrt `(T, rho, abar, zbar)` / `xa` (forward + reverse); `eval_pc_eos`, `eval_skye_eos`, `eos_dt_get`, `eos_pt_get`, and 4D table evaluators intra-cell wrt `(T, rho)` only (zero AD gradient wrt `abar`, `zbar`, `xa`, with concrete `xa` required outside `jax.jit` for `eos_dt_get` / `eos_pt_get` / 4D tables) | CPU (`HELM`, `Skye`, `PC`, 4D `DT`/`PT` host callbacks), CPU + GPU/TPU (`Ideal`, `blend_eos_results`) |
| `flexmesa.kap` | Independent `@jax.jit` Compton, conduction, Type 1/2, AESOPUS, and `kapCN` interpolation kernels; static tables load via Rust | Intra-cell wrt `(z, x, zbase, log_rho, log_t, eta, ...)` (forward + reverse); zero wrt table index | CPU + GPU/TPU (`Compton`, conduction after init), CPU (Rust-backed table paths) |
| `flexmesa.atm` | Independent `@jax.jit` `eval_t_tau_jax` and `eval_paczynski_gradr_jax`; `eval_table_jax` wraps `_rust.atm_eval_table` via `pure_callback` + `@jax.custom_vjp` | `eval_t_tau_jax` / `eval_paczynski_gradr_jax` forward + reverse; `eval_table_jax` **reverse-mode only** (`grad`/`vjp`) wrt `(luminosity, r, m)` | CPU, GPU/TPU (`t_tau`, `paczynski`), CPU (`eval_table_jax`) |
| `flexmesa.net` | Thin `jax.pure_callback` wrapper around `_rust.net_eval` using `@jax.custom_vjp` | **Reverse-mode only** (`grad`/`vjp`; `jvp`/`jacfwd` unsupported) wrt `(x, temp, rho)` at **frozen** composition moments `(abar, zbar, z2bar, ye, eta)` | CPU (Rust host callback) |
| `flexmesa.turb` | Independent `@jax.jit` Cox/MLT kernel (`set_mlt_jax`); Python `DualOrder1` / scalar helpers for `tdc`, `semiconvection`, `thermohaline`; plain floats in `set_mlt` call `_rust.set_mlt_f64` | `set_mlt_jax` forward + reverse wrt local state inputs; `DualOrder1` tangent mode in Python helpers | CPU, GPU/TPU (`set_mlt_jax`), CPU (`_rust` and Python helpers) |
| `flexmesa.star` | **Frozen & unverified (`#166`)**: wraps Rust `StarModel` + standalone fixed-input formula helpers (`jax_kernels`, `jax_winds_mix`, `jax_evolve`, `jax_diffusion_rot`, `jax_hydro_pulse`, `jax_rsp`); no JAX evolution step | Fixed-input formulas only at frozen microphysics (`eval_hydro_residuals_jax` differentiates at fixed `P, E, gradT, eps_nuc`, not the Henyey Jacobian) | Frozen until `#143` |
| `flexmesa.binary` | **Frozen & unverified (`#166`)**: wraps Rust `BinarySystem` + standalone closed-form orbital/`jdot`/tides/RLOF helpers | Standalone analytic formula helpers only; no coupled JAX evolution | Frozen until `#143` |
| `flexmesa.astero` | **Frozen & unverified (`#166`)**: wraps Rust optimization drivers + standalone scaling, surface-correction, ratio, and $\chi^2$ helpers | Standalone formula helpers only | Frozen until `#143` |

---

## 5. Design Note: Gradients Through the Canonical Rust Engine (Post-`#143`)

Because JAX is derived from Rust and frozen on the engine side (`#166`), end-to-end differentiation through stellar structure solves and evolutionary steps will wrap the verified Safe Rust engine with custom JAX differentiation rules rather than maintaining a second full evolution engine in JAX. This section records the interface needed once the `#143` history harness passes:

1. **Derivatives Rust Already Computes**:
   - Single-zone microphysics evaluators return full partial derivatives wrt thermodynamic and composition inputs: `eos` returns $(P_{\mathrm{gas}}, E, S, \dots)$ and partials wrt $(\ln T, \ln \rho, \bar{A}, \bar{Z})$; `kap`, `rates`, `neu`, and `turb` return analytical state partials; `net` returns $(d\mathbf{X}/dt, \epsilon_{\mathrm{nuc}}, \epsilon_\nu)$ and their analytical Jacobian wrt $(X_i, T, \rho)$; `atm` returns $(\ln T_{\mathrm{surf}}, \ln P_{\mathrm{surf}})$ and surface partials wrt $(L, \ln R, \ln M, \ln \kappa)$.
   - `flexmesa-star`'s Henyey assembly (`eval_equations`) already computes and stores the analytical 3-cell Jacobian blocks $\mathbf{J} = (\mathbf{L}_k, \mathbf{D}_k, \mathbf{U}_k)$ of shape `(nz, nvar, nvar)` together with the residual vector $\mathbf{R}(\mathbf{y}_{n+1}; \mathbf{y}_n, \Delta t, \boldsymbol{\theta}) = \mathbf{0}$, and `flexmesa-mtx` already exposes `solve_block_tridiag_bcyclic` with transposed adjoint solve support via `mtx.solve_block_tridiagonal_jax(..., solver_mode="rust_ffi")`.
2. **Implicit-Function-Theorem (IFT) `@jax.custom_vjp` and `@jax.custom_jvp` Step Wrappers**:
   - Let $\mathbf{y}_{n+1}(\mathbf{y}_n, \boldsymbol{\theta})$ denote a converged backward-Euler / implicit step solving $\mathbf{F}(\mathbf{y}_{n+1}, \mathbf{y}_n, \boldsymbol{\theta}) = \mathbf{0}$ with converged block-tridiagonal Jacobian $\mathbf{J} = \partial\mathbf{F}/\partial\mathbf{y}_{n+1}$.
   - **Reverse mode (`@jax.custom_vjp`)**: Given output cotangent $\bar{\mathbf{y}}_{n+1}$, solve the single linear adjoint system $\mathbf{J}^\top \boldsymbol{\lambda} = -\bar{\mathbf{y}}_{n+1}$ via the existing `bcyclic` transposed solver, then accumulate vector-Jacobian products into prior state and parameters:
     $$
     \bar{\mathbf{y}}_n = \left(\frac{\partial\mathbf{F}}{\partial\mathbf{y}_n}\right)^\top \boldsymbol{\lambda}, \qquad \bar{\boldsymbol{\theta}} = \left(\frac{\partial\mathbf{F}}{\partial\boldsymbol{\theta}}\right)^\top \boldsymbol{\lambda}.
     $$
     No Newton solver iterations are differentiated (`jax.lax.stop_gradient` on step count, retries, and line-search factors), and the linear solve reuses the verified `flexmesa.mtx` IFT pattern.
   - **Forward mode (`@jax.custom_jvp`)**: Given input tangents $(\dot{\mathbf{y}}_n, \dot{\boldsymbol{\theta}})$, solve $\mathbf{J}\,\dot{\mathbf{y}}_{n+1} = -\left(\frac{\partial\mathbf{F}}{\partial\mathbf{y}_n}\dot{\mathbf{y}}_n + \frac{\partial\mathbf{F}}{\partial\boldsymbol{\theta}}\dot{\boldsymbol{\theta}}\right)$ with the same factorized or `bcyclic` linear solver to support `jax.jvp` / `jax.jacfwd`.
3. **Requirements for a Differentiable Multi-Step Evolution Trajectory**:
   - **Explicit $\partial\mathbf{F}/\partial\mathbf{y}_n$ and $\partial\mathbf{F}/\partial\boldsymbol{\theta}$ Hooks over `PyO3`**: Beyond the already-exposed $\partial\mathbf{F}/\partial\mathbf{y}_{n+1}$ blocks `(lblk, dblk, ublk)`, Rust must expose lightweight VJP/JVP contractions of time-derivative and boundary terms wrt old-step arrays $(\rho_{\mathrm{start}}, T_{\mathrm{start}}, E_{\mathrm{start}}, X_{i,\mathrm{start}})$ (`_start` coupling in hydro and nuclear burning) and continuous namelist parameters $\boldsymbol{\theta}$ (e.g., $\alpha_{\mathrm{MLT}}$, overshoot $f_{\mathrm{ov}}$, wind efficiency $\eta$, $\Delta t$).
   - **Variable-Mesh Remesh & Discrete Control Handling**: Adaptive remeshing (`adjust_mesh`) changes zone count `nz` and cell mass fractions `dq` discretely. Differentiable evolutionary trajectories must treat split/merge placement as fixed per step (`stop_gradient` on zone indices) while differentiating the conservative remap weights linearly wrt `dq` (or fixed Lagrangian mass coordinates), and isolate discrete stopping conditions with implicit event-time boundary terms.
   - **Checkpointed Multi-Step Adjoints (`#150`)**: Long trajectories (`N_steps ~ 10^3 - 10^4`) cannot retain all intermediate `(nz, nvar, nvar)` Jacobians in memory; reverse-mode trajectory gradients require binomial checkpointing (`revolve`) built directly on `#150`'s full-state `.photo` / `BinarySystem` checkpoint serialization so intermediate steps are replayed in bounded memory before each local IFT linear solve.
