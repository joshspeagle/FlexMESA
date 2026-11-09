# Frequently Asked Questions (FAQ) & Troubleshooting

This page answers common questions regarding floating-point precision, backend routing (`backend="auto" | "rust" | "jax"`), block-tridiagonal Henyey solver selection (`bcyclic` vs. `pcr` vs. `rust_ffi`), legacy Fortran MESA file compatibility, and multi-thread/GPU configuration.

---

## 1. Precision & Backend Dispatch

### Why does importing `flexmesa` enable 64-bit precision in JAX (`jax_enable_x64`)?

Stellar structure, spline table interpolation, and stiff nuclear networks span 15 to 30 orders of magnitude in density, pressure, and reaction timescale. In 32-bit `float32` arithmetic ($\epsilon_{\mathrm{mach}} \approx 1.19 \times 10^{-7}$), hydrostatic pressure differences across thin photospheric shells ($dq_k \sim 10^{-10}$) and second-order Helmholtz free-energy derivatives suffer severe catastrophic cancellation. Importing `flexmesa` calls `flexmesa.enable_float64()`, which sets `jax.config.update("jax_enable_x64", True)` before any arrays or JIT traces are constructed.

If you import `jax.numpy` before `flexmesa` in a custom script, either import `flexmesa` first or set `JAX_ENABLE_X64=1` in your shell environment.

### How does `backend="auto"` decide between Safe Rust and JAX?

Every Tier B and Tier C physics function inspects its positional and keyword arguments via `flexmesa.resolve_backend(backend, *args)`:

- If **any** argument is a `jax.Array` or an active JAX tracer (`jax.core.Tracer` inside `jax.jit`, `jax.grad`, `jax.jacfwd`, or `jax.vmap`), `backend="auto"` dispatches to the native `@jax.jit` kernel.
- Otherwise (for Python `float` scalars and `numpy.ndarray` arrays), `backend="auto"` dispatches zero-copy to the compiled `flexmesa._rust` (`PyO3`) Safe Rust extension.

You can override this behavior at any time by passing `backend="rust"` or `backend="jax"` explicitly.

---

## 2. Linear Solvers & Automatic Differentiation

### When should I use `solver_mode="pcr"` versus `solver_mode="rust_ffi"` in `flexmesa.mtx`?

When solving block-tridiagonal linear systems $A x = b$ of shape `(nz, m, m)` in JAX (`mtx.solve_block_tridiagonal_jax`):

- **`solver_mode="pcr"` (Native GPU Parallel Cyclic Reduction)**: Executes $\lceil \log_2 N_{\mathrm{zones}} \rceil$ vectorized block reduction steps (`12` steps at $N_{\mathrm{zones}} = 4096$) entirely inside XLA without leaving the GPU. Use `"pcr"` whenever you are vectorizing across batches of stars with `jax.vmap` on a GPU or TPU.
- **`solver_mode="rust_ffi"` (Safe Rust `bcyclic` + Analytical `@jax.custom_vjp`)**: Dispatches the forward pass to `flexmesa._rust.solve_block_tridiag_bcyclic` via `jax.pure_callback` and differentiates the reverse pass analytically via the Implicit Function Theorem adjoint system $A^\top \lambda = \bar{x}$ and $\bar{A} = -\lambda x^\top$. Use `"rust_ffi"` for single-star CPU workflows or hybrid CPU/GPU adjoint sensitivities where host cyclic reduction ($2.76\,\mathrm{ms}$ at $N_{\mathrm{zones}} = 4096$) is faster than GPU kernel launch overhead.

```python
import jax.numpy as jnp
import flexmesa.mtx as mtx

# Solve a 4-zone, 2x2 block-tridiagonal system with native PCR
nz, m = 4, 2
sub = jnp.zeros((nz, m, m), dtype=jnp.float64)
diag = jnp.broadcast_to(jnp.eye(m, dtype=jnp.float64) * 2.0, (nz, m, m))
sup = jnp.zeros((nz, m, m), dtype=jnp.float64)
rhs = jnp.ones((nz, m), dtype=jnp.float64)

sol = mtx.solve_block_tridiagonal_jax(sub, diag, sup, rhs, solver_mode="pcr")
print("PCR solution norm:", float(jnp.linalg.norm(sol)))
```

### How does `FlexMESA` differentiate iterative solvers (`set_mlt`, `set_tdc`, `eos_pt_get`) without differentiating through the iteration loop?

Propagating 33-variable or 70-variable dual numbers through every bracket-search and Newton-Raphson iteration is the primary CPU bottleneck in upstream Fortran MESA's `turb_tdc` (`223 ms` per 4096-zone call). `FlexMESA` instead converges the scalar root $y^*$ in plain `f64` and applies the **Implicit Function Theorem** once at the converged root $Q(y^*, \theta) = 0$:

$$\frac{\partial y^*}{\partial \theta} = -\left(\frac{\partial Q}{\partial y}\right)^{-1} \frac{\partial Q}{\partial \theta}$$

This yields the exact analytical tangent (matching infinite-precision unrolled AD to $\le 10^{-15}$) while accelerating `turb_tdc` by $54\times$ on CPU and $146\times$ on GPU.

---

## 3. Fortran MESA Compatibility & Data Tables

### Can I load existing Fortran MESA `.mod` files and `inlist` directories directly?

Yes. `StarModel.from_mod_file("path/to/model.mod")` parses upstream ASCII `.mod` files (converting 1-based Fortran zone indexing and `D` exponents into 0-based C-contiguous `FlexMESA` arrays), and `star.parse_inlist_bundle("path/to/inlist")` recursively resolves `read_extra_star_job_inlist*`, `read_extra_controls_inlist*`, and `&pgstar` namelist chains. See [Inlists, Models & Data Caches](../guides/inlists-and-models.md) for complete examples.

### Why does zone index `0` correspond to the stellar surface instead of the center?

Upstream Fortran MESA indexes stellar zones from `k = 1` at the outer photosphere (`q(1) = 1.0`) to `k = s% nz` at the stellar center (`q(nz) = dq(nz)`). Preserving the surface-to-center ordering (`k = 0` at the photosphere to `k = nz - 1` at the center) ensures that optical-depth integration ($\tau$ increasing inward from `k = 0`), atmospheric boundary conditions at the first block row (`k = 0`), and `GYRE`/`FGONG` reversal conventions map 1-to-1 with upstream MESA without reversing array strides during FFI calls.

---

## 4. Threading, Memory & GPU Environment Variables

### Which environment variables control CPU thread pools and GPU memory allocation?

For reproducible multi-core benchmarks or concurrent multi-process runs, configure the following environment variables before launching Python:

```bash
# Cap Safe Rust rayon and BLAS/OpenMP worker threads
export RAYON_NUM_THREADS=4
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Prevent XLA from pre-allocating 75% of GPU VRAM when sharing a workstation GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.65

# Force CPU-only JAX execution if the GPU is occupied by another job
export JAX_PLATFORMS=cpu
```
