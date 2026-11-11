# Python, PyO3 & JAX API Reference

`FlexMESA` exposes a unified Python and JAX (`float64`) public interface backed by compiled `#![forbid(unsafe_code)]` Safe Rust workspace crates (`flexmesa._rust`) and XLA-compiled `@jax.jit` array kernels. Importing `flexmesa` automatically enables 64-bit IEEE-754 double precision in JAX (`jax_enable_x64=True`) so that spline interpolations, thermodynamic partials, nuclear networks, and block-tridiagonal Henyey solves maintain full floating-point precision by default.

Safe Rust crate documentation (`crates/flexmesa-*`) is built via `cargo doc --workspace --no-deps` and published at `/rustdoc/` in the documentation deployment.

---

## Domain API Reference Map

To keep each reference page fast to load and easy to navigate, the Python and JAX API is organized into eight domain-specific pages:

| Domain Reference Page | Python Subpackages | Primary Classes & Functions |
| :--- | :--- | :--- |
| **[1D Stellar Structure & Evolution](star.md)** | `flexmesa.star` | `StarModel`, `HenyeySolverReport`, `EvolveStepReport`, `HistoryRow`, `JaxStarState`, `parse_inlist_bundle`, `run_test_suite_case`, atomic diffusion, shellular rotation, HLLC Riemann hydrodynamics, `RSP2` turbulent pulsation, and `GYRE`/`FGONG`/`OSC` exporters |
| **[Interacting Binaries & Accretors](binary.md)** | `flexmesa.binary` | `BinarySystem`, `eval_rlobe`, `eval_ritter_mdot_thin`, `eval_kolb_mdot_thick`, Eggleton Roche-lobe geometry, Kerr BH ISCO spin & Eddington accretion, orbital $\dot{J}$, and Hut tidal evolution |
| **[Asteroseismology & Optimization](astero.md)** | `flexmesa.astero` | Acoustic radius/depth profiles, $\Delta\nu$ and $\nu_{\max}$ scaling relations, $\Delta\Pi_1$ period spacing, He II/BCZ glitch depths, 5 surface corrections (`BG14`, `Kjeldsen2008`, `Sonoi2015`), separation ratios, $\chi^2$, and Nelder-Mead/Powell/grid optimizers |
| **[Equations of State](eos.md)** | `flexmesa.eos` | 7-component EOS (`Ideal`, `HELM`, `PC`, `Skye`, `FreeEOS`, `OPAL/SCVH`, `CMS`), $C^2$ quintic polygonal blending (`eos_dt_get`), and $(P_{\mathrm{gas}}, T)$ root-finding inversions (`eos_pt_get`) |
| **[Opacity, Atmospheres & Photometry](opacity-atm-colors.md)** | `flexmesa.kap`, `flexmesa.atm`, `flexmesa.colors` | Radiative (`OPAL`/`OP`/`AESOPUS`), conductive, and Compton opacities; Eddington/Krishna-Swamy/irradiated and tabulated model atmospheres; 3D cubic Hermite Kurucz SED synthesis and synthetic filter photometry |
| **[Nuclear Networks, Rates & Neutrinos](nuclear.md)** | `flexmesa.chem`, `flexmesa.ionization`, `flexmesa.neu`, `flexmesa.rates`, `flexmesa.net` | 347 built-in reference nuclides and 10 solar abundance scales; partial ionization and $\mathrm{Fe}^{56}$ in $\mathrm{He}^4$ charge states; Itoh 5-channel thermal neutrinos; 316 REACLIB rates and plasma screening; `basic.net`/`approx21.net` RHS, analytical Jacobians, and 1-zone stiff burners |
| **[Foundational Numerics & Convection](numerics.md)** | `flexmesa.constants`, `flexmesa.math`, `flexmesa.utils`, `flexmesa.auto_diff`, `flexmesa.interp_1d`, `flexmesa.interp_2d`, `flexmesa.mtx`, `flexmesa.num`, `flexmesa.turb` | Physical constants; IEEE-754 elementary intrinsics; dual and hyperdual automatic differentiation; 1D/2D monotonic and bicubic splines; dense, banded, `bcyclic`, and `PCR` block-tridiagonal solvers; root finders and Rosenbrock/RK integrators; `MLT`, `TDC`, semiconvection, and thermohaline mixing |
| **[I/O, Schemas, Plotting & Benchmarks](io-plotting-bench.md)** | `flexmesa.schema`, `flexmesa.data`, `flexmesa.testing`, `flexmesa.plotting`, `flexmesa.bench` | `.defaults` and Fortran `&namelist` parsers; `.safetensors` memory-mapped loaders and `SHA-256` manifest verification; golden `test_output` comparators; HR, Kippenhahn, échelle, and binary RLOF plotting; 4-stage benchmark harnesses |

---

## Top-Level Package Exports (`flexmesa`)

::: flexmesa
    options:
      members:
        - enable_float64
        - __version__

---

## Hybrid Backend Dispatch (`flexmesa._dispatch`)

Every Tier B and Tier C physics function accepts `backend="auto" | "rust" | "jax"`. Under `backend="auto"` (the default), calls passing `numpy.ndarray` or Python float arguments route zero-copy to `flexmesa._rust`, whereas calls passing `jax.Array` or JAX tracer arguments (`jax.grad`, `jax.vmap`, `jax.jit`) route to native `@jax.jit` kernels.

```python
import jax.numpy as jnp
import numpy as np
import flexmesa

# NumPy inputs resolve to the zero-copy Safe Rust backend
print("NumPy dispatch:", flexmesa.resolve_backend("auto", np.array([1.0, 2.0])))

# JAX arrays resolve to the XLA-compiled JAX backend
print("JAX dispatch:  ", flexmesa.resolve_backend("auto", jnp.array([1.0, 2.0])))
```

::: flexmesa._dispatch

---

## Command-Line Interface (`flexmesa.cli`)

The installed `flexmesa` console script dispatches subcommands such as `flexmesa data fetch`, `flexmesa data status`, and `flexmesa --version`.

::: flexmesa.cli
