# User Guide: Backends, Solvers & CPU/GPU Performance Tuning

`FlexMESA` is built on a **Three-Tier Hybrid Architecture** that combines memory-safe parallel Rust (`rayon`), zero-copy Python bindings (`PyO3` / `numpy`), and branchless 64-bit JAX (`@jax.jit` + `jax.vmap`). Understanding when to select `backend="rust"` versus `backend="jax"`, how to choose between block-tridiagonal Henyey solvers (`thomas`, `bcyclic`, `pcr`, `rust_ffi`), and how to size batch dimensions for CPU caches and GPU streaming multiprocessors allows you to achieve maximum throughput across single-star evolution, grid sweeps, and gradient-based inference.

---

## 1. Backend Dispatch: `backend="auto" | "rust" | "jax"`

Every public microphysics, binary, and asteroseismic evaluator in `FlexMESA` accepts a keyword-only `backend` parameter (`"auto"`, `"rust"`, or `"jax"`) routed through `flexmesa.resolve_backend`:

- **`backend="auto"` (Default)**: Inspects the primary input arguments without copying data. If any input is a `jax.Array` or an active JAX tracer (inside `@jax.jit`, `jax.grad`, `jax.jacfwd`, or `jax.vmap`), `FlexMESA` dispatches to the pure-JAX kernel. Otherwise, it converts Python floats or NumPy arrays into C-contiguous `float64` views and calls the zero-copy PyO3 Rust binding.
- **`backend="rust"`**: Forces execution on the Safe Rust CPU engine (releasing the Python GIL during multi-zone `parallel=True` `rayon` loops).
- **`backend="jax"`**: Forces execution through the `@jax.jit` XLA kernel even when given standard Python floats or NumPy arrays.

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
