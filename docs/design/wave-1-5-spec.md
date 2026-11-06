# Wave 1.5 Specification: Three-Tier Hybrid Integration, Solver Scaling & Public Documentation Website

This specification defines the engineering contract for **Wave 1.5** ([Issue #28](https://github.com/joshspeagle/FlexMESA/issues/28)), which serves as the inaugural **Phase `X.5`** holistic consolidation, optimization, upstream Fortran MESA verification, and public documentation milestone (`DECISIONS #6`–`#8`).

---

## 1. The Phase `X.5` Post-Wave Verification and Optimization Protocol

While per-pull-request CI checks verify unit correctness, static types, and local golden test baselines for individual modules during Wave `X`, systemic cross-module interactions and comparative performance against upstream Fortran MESA can only be evaluated once all modules in Wave `X` are assembled. Every primary migration wave (`Wave 1`, `Wave 2`, `Wave 3a`, `Wave 3b`, `Wave 4`, `Wave 5`) therefore concludes with an explicit **`Wave X.5`** milestone before any module in Wave `X+1` begins:

1. **Upstream Fortran MESA Compilation, Parity & 4-Stage Timing (`benchmarks/fortran_mesa/`)**:
   - Compile and link the upstream Fortran MESA static archives (`lib*.a`) corresponding to all modules in Wave `X` with `gfortran -O2 -fopenmp`.
   - Verify 16-digit parity against `<pkg>/test/test_output` and cross-language stage checksums across $N_{\mathrm{zones}} \in \{128, 1024, 4096\}$, and measure 4-stage execution latencies across Upstream Fortran MESA, Safe Rust (`crates/`), Python (`backend="rust"`), and JAX (`backend="jax"`).
2. **Holistic Multi-Lens Review & Algorithmic Optimization**:
   - Audit all code written in Wave `X` to remove redundant abstractions, eliminate host-device or FFI copy bottlenecks, replace suboptimal algorithms, and ensure uniform `backend="auto" | "rust" | "jax"` behavior.
3. **Iterative Re-Verification Loop Against `FlexMESA` CI and Upstream Fortran MESA**:
   - After each optimization batch, re-run the full `FlexMESA` test suite (`pytest`, `cargo test`, `check_grads`, `proptest`, `hypothesis`) and re-execute `benchmarks/fortran_mesa/`.
   - **Advancement Gate**: Work does not advance to Wave `X+1` until all `FlexMESA` tests and all compare-to-original-Fortran-MESA parity and performance gates pass and are published to the `Wave X.5` GitHub tracking issue.
4. **Public Documentation Website Sync (`docs/` + `mkdocs build --strict`)**:
   - Author and verify the Wave `X` module explainers, governing equations, publication-styled diagnostic figures (`scripts/build_docs_figures.py`), updated post-`X.5` 4-way benchmark scorecards, and synced API reference pages on the public MkDocs Material website.

---

## 2. Three-Tier Hybrid Architecture Contract (`DECISIONS #6`)

### 2.1 Tier Classification

| Tier | Modules | Safe Rust (`crates/`) | Python / JAX (`src/flexmesa/`) | Default Dispatch (`backend="auto"`) |
| :--- | :--- | :--- | :--- | :--- |
| **Tier A: Infrastructure, I/O & Explicit Dual Types** | `utils`, `schema`, `data`, `testing`, `auto_diff` | Canonical implementation (`#![forbid(unsafe_code)]`, `rayon`, zero-copy `.safetensors` `mmap`) | Thin `PyO3` wrappers over `flexmesa._rust`; zero duplicated Python parser/loader or dual-number loop code | Always executes in Safe Rust via `flexmesa._rust` |
| **Tier B: Differentiable Shell-Local Microphysics & Splines** | `const`, `math`, `interp_1d`, `interp_2d`, `neu`, `turb`, `chem`, `ionization`, `colors`, `eos`, `kap`, `rates`, `atm` | Monomorphized `f64` and `DualOrder1<N>` CPU kernels with Implicit Function Theorem root derivatives and persistent `rayon` parallel zone loops | Branchless `@jax.jit` flat-array kernels (`jnp.where`, `lax.fori_loop`, `jax.custom_jvp`) | Routes `np.ndarray` and float inputs zero-copy to `flexmesa._rust`; routes `jax.Array` and traced inputs to `@jax.jit` |
| **Tier C: Coupled Linear & Stiff Solvers** | `mtx`, `num`, `net`, `star_solver` | Sequential block-Thomas, Hirshman `bcyclic` odd-even cyclic reduction, and stiff Rosenbrock/Newton solvers | Native JAX Parallel Cyclic Reduction (`solver_mode="pcr"`) + `flexmesa._rust` FFI bindings with analytical `jax.custom_vjp` (`solver_mode="rust_ffi"`) | Routes `np.ndarray` to `flexmesa._rust`; routes `jax.Array` to `solver_mode` (`"pcr"` default on GPU/vmap, `"rust_ffi"` optional) |

### 2.2 Zero-Copy `PyO3` Extension Bridge (`crates/flexmesa-py` $\to$ `flexmesa._rust`)

- All 14 core numerical, physics, schema, data, and benchmark crates in `crates/` enforce `#![forbid(unsafe_code)]` via `[workspace.lints.rust]`.
- The thin `PyO3` binding crate `crates/flexmesa-py` (`cdylib` built via `maturin` into `flexmesa._rust`) sets `#![deny(unsafe_code)]` with targeted allowance only for `#[pymodule]` / `#[pyfunction]` macro expansion, containing zero hand-written `unsafe` blocks.
- Array arguments cross the Python-to-Rust boundary via `numpy::PyReadonlyArray1<f64>` / `PyReadonlyArray2<f64>` (`as_slice()`) and return contiguous `PyArray1<f64>` / `PyArray2<f64>` buffers without intermediate copying.

---

## 3. Parallel Block Cyclic Reduction (`bcyclic` & PCR) and Implicit Solver Adjoints (`DECISIONS #7`)

### 3.1 Hirshman et al. (2010) Odd-Even Block Cyclic Reduction (`crates/flexmesa-mtx/src/bcyclic.rs`)

For an $N_z$-shell block-tridiagonal system with $N_{\mathrm{var}} \times N_{\mathrm{var}}$ blocks,
$$L_k x_{k-1} + D_k x_k + U_k x_{k+1} = b_k, \qquad k \in \{0, \dots, N_z - 1\},$$
odd-even block cyclic reduction proceeds across $\lceil \log_2 N_z \rceil$ levels $s = 0, 1, \dots, S-1$ with stride $p = 2^s$:

1. **Odd-Row Normalization**: For every active odd multiple $j = (2m + 1)p < N_z$, factor $D_j^{(s)} = P_j^{-1} L_j^{(D)} U_j^{(D)}$ and normalize in place:
   $$\tilde{L}_j = \left(D_j^{(s)}\right)^{-1} L_j^{(s)}, \qquad \tilde{U}_j = \left(D_j^{(s)}\right)^{-1} U_j^{(s)}, \qquad \tilde{b}_j = \left(D_j^{(s)}\right)^{-1} b_j^{(s)}.$$
2. **Even-Row Schur Update**: For every active even multiple $k = 2m p < N_z$, eliminate $x_{k-p}$ and $x_{k+p}$:
   $$D_k^{(s+1)} = D_k^{(s)} - L_k^{(s)} \tilde{U}_{k-p} - U_k^{(s)} \tilde{L}_{k+p},$$
   $$L_k^{(s+1)} = -L_k^{(s)} \tilde{L}_{k-p}, \qquad U_k^{(s+1)} = -U_k^{(s)} \tilde{U}_{k+p},$$
   $$b_k^{(s+1)} = b_k^{(s)} - L_k^{(s)} \tilde{b}_{k-p} - U_k^{(s)} \tilde{b}_{k+p}.$$
3. **Root Solve & Backward Substitution**: Solve $D_0^{(S)} x_0 = b_0^{(S)}$ at the root shell $k = 0$, then traverse $s = S-1 \dots 0$ in reverse to recover every odd-multiple shell via
   $$x_j = \tilde{b}_j - \tilde{L}_j x_{j-p} - \tilde{U}_j x_{j+p}.$$

### 3.2 Vectorized Parallel Cyclic Reduction (PCR) in JAX (`src/flexmesa/mtx`)

On SIMD/SIMT accelerators (`@jax.jit` and `jax.vmap`), all-row Parallel Cyclic Reduction (PCR) updates all $N_z$ rows simultaneously at each stride $p = 2^s$ ($s = 0 \dots \lceil \log_2 N_z \rceil - 1$) using uniform tensor shapes `(N_z, nvar, nvar)`:
1. Normalize all rows in parallel via `jax.vmap(jnp.linalg.solve)`:
   $$\tilde{L}_k = \left(D_k^{(s)}\right)^{-1} L_k^{(s)}, \qquad \tilde{U}_k = \left(D_k^{(s)}\right)^{-1} U_k^{(s)}, \qquad \tilde{b}_k = \left(D_k^{(s)}\right)^{-1} b_k^{(s)}.$$
2. Shift arrays by $\pm p$ along axis 0 (zero-padded at boundaries) and apply the vectorized Schur update to all $N_z$ rows in a single fused XLA step. After $\lceil \log_2 N_z \rceil$ steps, both $\tilde{L}_k$ and $\tilde{U}_k$ vanish outside the domain and $x_k = \tilde{b}_k^{(S)}$.

### 3.3 Analytical Implicit Differentiation (`custom_jvp` & `custom_vjp`) for Linear Solves

For any block-tridiagonal system $A(\theta) x(\theta) = b(\theta)$:
- **Forward mode (`custom_jvp`)**: Given primal solution $x = A^{-1} b$ and tangents $(\dot{A}, \dot{b})$, the tangent $\dot{x}$ solves the linear system with the identical matrix $A$:
  $$A \dot{x} = \dot{b} - \dot{A} x = \dot{b} - \left(\dot{L}_k x_{k-1} + \dot{D}_k x_k + \dot{U}_k x_{k+1}\right).$$
- **Reverse mode (`custom_vjp`)**: Given cotangent $\bar{x}$, solve the transpose block-tridiagonal system $A^\top \lambda = \bar{x}$ (where the transpose of $(L_k, D_k, U_k)$ has sub-diagonal $U_{k-1}^\top$, diagonal $D_k^\top$, and super-diagonal $L_{k+1}^\top$), yielding cotangents:
  $$\bar{b}_k = \lambda_k, \qquad \bar{D}_k = -\lambda_k x_k^\top, \qquad \bar{L}_k = -\lambda_k x_{k-1}^\top, \qquad \bar{U}_k = -\lambda_k x_{k+1}^\top.$$

---

## 4. Public Documentation Website Architecture (`MkDocs Material`, `DECISIONS #8`)

The public documentation website (`mkdocs.yml`, `docs/`) is organized into five top-level navigation tabs:

1. **Overview & Philosophy (`docs/index.md`, `docs/design-philosophy.md`)**: Project motivation, the 12-tier MESA architecture map, the Three-Tier Hybrid Architecture, and quickstart guides for Python (`backend="rust"` and `backend="jax"`) and Safe Rust.
2. **Architecture & Specifications (`docs/design/`)**: Complete engineering specifications (`scoping-and-architecture.md`, `wave-0-spec.md`, `wave-1-spec.md`, `wave-1-5-spec.md`).
3. **Module Physics & Numerics Guide (`docs/modules/`)**: Deep-dive explainers for Wave 0 (`schema`, `data`, `testing`) and Wave 1 (`const`, `utils`, `math`, `auto_diff`, `interp_1d`, `interp_2d`, `mtx`, `num`, `neu`, `turb`), featuring governing physical equations, numerical derivations, upstream MESA precision notes, and embedded diagnostic figures (`docs/assets/figures/`).
4. **Benchmarks & Parity (`docs/design/wave-1-benchmark-report.md`)**: Four-way parity scorecards and runtime scaling comparisons (`Upstream Fortran MESA` vs. `Safe Rust` vs. `Python PyO3` vs. `Python JAX`).
5. **API Reference (`docs/api/`)**: Auto-generated Python/JAX API reference (`mkdocstrings`) and cross-linked Safe Rust crate documentation (`cargo doc`).
