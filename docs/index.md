# FlexMESA

**FlexMESA** is a modular, memory-safe, and end-to-end differentiable rewrite of the [Modules for Experiments in Stellar Astrophysics (MESA)](https://docs.mesastar.org) codebase, built on an **Integrated Three-Tier Hybrid Architecture** combining **Safe Rust (`#![forbid(unsafe_code)]`)**, zero-copy **Python (`PyO3` / `flexmesa._rust`)**, and **JAX (`float64` `@jax.jit` + `jax.vmap`)**.

---

## Core Design Goals

1. **16-Digit Numerical Parity with Upstream Fortran MESA**:
   - Every ported constant, spline interpolator, linear solver, and microphysics module is verified against upstream Fortran MESA's `<pkg>/test/test_output` golden baselines to $\le 10^{-14}$ relative error in IEEE-754 double precision (`f64` / `jnp.float64`).
   - At the end of every migration wave (`Wave X.5`), `FlexMESA` compiles and benchmarks the original Fortran MESA static archives (`benchmarks/fortran_mesa/`) side-by-side against Safe Rust, Python (`PyO3`), and JAX across radial grid sizes $N_{\mathrm{zones}} \in \{128, 1024, 4096\}$.
2. **Integrated Three-Tier Hybrid Architecture**:
   - **Tier A (`utils`, `schema`, `data`, `testing`, `auto_diff`)**: Canonical implementation in Safe Rust (`crates/`) with zero-copy `.safetensors` memory-mapping (`memmap2`) and a persistent `rayon` thread pool, exposed to Python via `flexmesa._rust`.
   - **Tier B (`const`, `math`, `interp_1d`, `interp_2d`, `neu`, `turb`, plus Waves 2–3 microphysics)**: Monomorphized Safe Rust CPU kernels (`f64` + `DualOrder1<N>`) paired with branchless `@jax.jit` GPU kernels behind a unified `backend="auto" | "rust" | "jax"` Python dispatcher.
   - **Tier C (`mtx`, `num`, `net`, `star_solver`)**: Sequential block-Thomas and parallel Hirshman et al. (2010) `bcyclic` odd-even block cyclic reduction in Safe Rust, paired with $\lceil \log_2 N_{\mathrm{zones}} \rceil$ vectorized Parallel Cyclic Reduction (`solver_mode="pcr"`) and analytical Implicit Function Theorem `@jax.custom_vjp` FFI bindings (`solver_mode="rust_ffi"`) in JAX.
3. **End-to-End Automatic Differentiation via the Implicit Function Theorem**:
   - Rather than propagating a 33-component tangent array through every intermediate root-finding iteration, `FlexMESA` converges nonlinear microphysics (`turb_tdc`, `eos`) and coupled Henyey linear systems (`mtx`) in primal `f64` precision and evaluates exact analytical sensitivities once at the converged root $Y^*$ via the Implicit Function Theorem, delivering $12.6\times\text{--}146\times$ speedups over upstream 4-thread OpenMP Fortran MESA.

![Four-way runtime comparison across Wave 1 workloads at N_zones = 4096](assets/figures/wave1_benchmark_staircase.png)

---

## Quickstart

=== "Python & JAX (`flexmesa`)"

    ```python
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flexmesa import interp_2d, mtx, neu, turb

    # 1. Zero-copy Safe Rust CPU execution (default for float / np.ndarray)
    res_neu = neu.neu_get(
        t=1.0e9, log10_t=9.0, rho=1.0e6, log10_rho=6.0, abar=12.0, zbar=6.0
    )
    print(f"Total neutrino loss: {res_neu.loss[0]:.6e} erg/g/s")

    # 2. End-to-end differentiable JAX GPU execution (auto-routed on jax.Array)
    grad_eps_t = jax.grad(
        lambda t: neu.neu_pair_plas_jax(t, jnp.float64(1.0e6), 12.0, 6.0)
    )(jnp.float64(1.0e9))

    # 3. Block-tridiagonal Henyey solve (CPU bcyclic or GPU log2 PCR)
    rng = np.random.default_rng(0)
    lblk = rng.uniform(-0.1, 0.1, size=(128, 5, 5))
    dblk = np.broadcast_to(4.0 * np.eye(5), (128, 5, 5)).copy()
    ublk = rng.uniform(-0.1, 0.1, size=(128, 5, 5))
    rhs = rng.uniform(-1.0, 1.0, size=(128, 5))

    x_cpu = mtx.solve_block_tridiagonal(lblk, dblk, ublk, rhs, method="bcyclic")
    x_gpu = mtx.solve_block_tridiagonal_jax(
        jnp.asarray(lblk),
        jnp.asarray(dblk),
        jnp.asarray(ublk),
        jnp.asarray(rhs),
        solver_mode="pcr",
    )
    ```

=== "Safe Rust (`crates/`)"

    ```rust
    use flexmesa_autodiff::DualOrder1;
    use flexmesa_neu::{NeuFlags, neu_get};
    use flexmesa_turb::{MltInputs, MltParams, set_mlt};

    fn main() {
        // Evaluate thermal neutrino energy loss and exact 4-variable Jacobian
        let out = neu_get(
            Some(1.0e9), Some(9.0), Some(1.0e6), Some(6.0),
            12.0, 6.0, 7.5, NeuFlags::all(),
        ).unwrap();
        println!("Total loss = {:.6e} erg/g/s", out.loss[0]);

        // Evaluate Cox MLT with const-generic DualOrder1<6> sensitivities
        let gradr = DualOrder1::<6>::independent(0.45, 5);
        let inp = MltInputs {
            chi_t: DualOrder1::constant(1.0),
            chi_rho: DualOrder1::constant(1.0),
            cp: DualOrder1::constant(2.0e8),
            grav: DualOrder1::constant(2.7e4),
            lambda: DualOrder1::constant(1.5e9),
            rho: DualOrder1::constant(1.0e-5),
            p: DualOrder1::constant(1.0e8),
            t: DualOrder1::constant(1.0e5),
            opacity: DualOrder1::constant(1.0),
            gradr,
            grada: DualOrder1::constant(0.40),
            gradl: DualOrder1::constant(0.40),
        };
        let mlt = set_mlt(&MltParams::cox(2.0), &inp).unwrap();
        println!(
            "grad_T = {:.6}, d(grad_T)/d(grad_r) = {:.6}",
            mlt.grad_t.val, mlt.grad_t.d1[5]
        );
    }
    ```

---

## Migration Roadmap and Status

| Wave | Scope | Status | Verification Gate |
| :--- | :--- | :--- | :--- |
| **Wave 0** | Infrastructure: `schema` (TOML/JSON namelists), `data` (`.safetensors` + SHA-256), `testing` (`test_output` parser) | **Complete** | 100% golden parser & binary table roundtrip tests |
| **Wave 1** | Foundation & First Microphysics: `const`, `utils`, `math`, `auto_diff`, `interp_1d`, `interp_2d`, `mtx`, `num`, `neu`, `turb` | **Complete** | 16-digit parity with upstream `<pkg>/test/test_output` |
| **Wave 1.5** | Three-Tier Hybrid Consolidation, `PyO3` Zero-Copy Bridge, `bcyclic` & JAX PCR Solvers, 4-Way Fortran Benchmarks & Docs | **Complete** | 18/18 4-way Fortran MESA parity & performance gates passed |
| **Wave 2** | Atomic & Photometric Microphysics: `chem` (`lodders03`, `gs98`, `asplund09` + 5095 nuclides), `ionization`, `colors` | **Complete** | 16-digit parity with upstream `<pkg>/test/test_output` |
| **Wave 2.5** | Pre-Validated Bicubic Spline Fast Path, Adaptive Spin-Then-Park Worker Team, Precomputed Simpson & 32-Gather Hermite SED Synthesis | **Complete** | 12/12 4-way Fortran MESA parity & performance gates passed |
| **Wave 3a** | Analytical & Single-Table Equations of State: `ideal`, `helm` (Timmes & Swesty 2000), `pc` (Potekhin & Chabrier 2010), `skye` (Jermyn et al. 2021), polygonal `eosDT_get` blending & `PT`/`DE` inversions | **Complete** | 16-digit parity with upstream `eos/test/test_output` |
| **Wave 3a.5** | `OnceLock<NuclideCatalog>`, Stack-Allocated Mixture Buffers, `@jax.custom_jvp` Horner Basis Chain, Direct `(T, rho)` Second-Order JAX Autodiff & 4-Way Fortran Benchmarks | **Complete** | 12/12 4-way Fortran MESA parity & performance gates passed |
| **Wave 3b** | 4D Tabulated EOS (`opal_scvh`, `free_eos`, `cms`, `vh1`), Radiative & Conductive Opacity (`kap`), Nuclear Reaction Rates & Screening (`rates`), Networks (`net`), Stellar Atmospheres (`atm`) | **Complete** | 16-digit parity with upstream `eos`, `kap`, `rates`, `net`, and `atm` `test_output` |
| **Wave 3b.5** | Lock-Free 4D EOS Table Lookup & Sorted Vertex LUTs (`eos_4d`), Pre-Validated Bicubic Spline Fast Paths (`kap`, `atm`), Thread-Local `NetWorkState` (`net`), Pre-Blended 4D EOS Batch Evaluators & 4-Way Fortran Benchmarks | **Complete** | 15/15 4-way Fortran MESA parity & performance gates passed |
| **Wave 4–5** | 1D Hydrostatic/Hydrodynamic Stellar Evolution (`star`), Non-Adiabatic Pulsations (`gyre`), Binary Evolution (`binary`), Asteroseismology (`astero`) | **Next** | Full evolutionary track & eigenfrequency parity |
