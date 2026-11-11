# I/O, Schemas, Plotting & Benchmarks (`schema`, `data`, `testing`, `plotting`, `bench`)

This reference page documents the configuration schema and Fortran `&namelist` parser (`flexmesa.schema`), the `.safetensors` memory-mapped data loader and `SHA-256` manifest verifier (`flexmesa.data`), the golden `test_output` oracle comparator (`flexmesa.testing`), the publication-styled diagnostic plotting suite (`flexmesa.plotting`), and the 4-stage benchmark harness (`flexmesa.bench`).

For hands-on guides, see [Inlists, Models & Data Caches](../guides/inlists-and-models.md), [Diagnostic Plotting](../guides/diagnostic-plotting.md), and [Backends & CPU/GPU Performance](../guides/backends-and-performance.md).

---

## Quick Usage Example

```python
from flexmesa.schema import namelist
from flexmesa.testing import golden

# Parse an inline Fortran &namelist block into typed NamelistBlock structures
blocks = namelist.parse_namelist_text(
    """
&star_job
  create_pre_main_sequence_model = .false.
/
&controls
  initial_mass = 1.0d0
  initial_z = 0.02d0
  mixing_length_alpha = 2.0d0
/
"""
)
val = golden.parse_fortran_float("1.98840987D+33")
print(
    f"block={blocks[1].name}, {blocks[1].assignments[0].name}={blocks[1].assignments[0].value}, "
    f"parsed float={val:.6e}"
)
```

---

## Declarative `.defaults` & Fortran `inlist` Schemas (`flexmesa.schema`)

`flexmesa.schema` parses upstream MESA `<pkg>/defaults/*.defaults` and `*_columns.list` files into typed control registries and validates Fortran `&namelist` (`inlist`) files (including `1.0d0` exponents, `.true.`/`.false.` booleans, array subscripts `xa_central_lower_limit_species(1)`, and recursive `read_extra_*_inlist` chains).

::: flexmesa.schema.defaults

::: flexmesa.schema.namelist

---

## `.safetensors` Data Loader, Manifest, Release Bundles & Converters (`flexmesa.data`)

`flexmesa.data` verifies `SHA-256` checksums against `manifest.toml` and `bundles.toml`, downloads and unpacks full-resolution `data-v1` release archives into `$FLEXMESA_DATA_DIR` (`flexmesa data fetch`), memory-maps 64-byte-aligned `.safetensors` physics tables directly from the OS page cache, and converts upstream ASCII/binary MESA data tables (`isotopes.data`, `lodders03.data`, `helm_table.dat`, `eosDT`, `kap`, `rates`, `atm`) into `.safetensors`.

::: flexmesa.data.manifest

::: flexmesa.data.fetch

::: flexmesa.data.loader

::: flexmesa.data.converters

---

## Golden `test_output` Oracle Comparators (`flexmesa.testing`)

`flexmesa.testing` parses upstream MESA `<pkg>/test/test_output` reference files (including Fortran `1pe24.16` and missing-`E` three-digit exponents like `1.2345-102`) and asserts multi-tier relative and absolute numerical tolerances.

::: flexmesa.testing.golden

---

## Publication-Styled Diagnostic Plotting (`flexmesa.plotting`)

`flexmesa.plotting` replaces Fortran MESA's legacy X11/PGPLOT `PGSTAR` and `PGBINARY` subsystems with composable, colorblind-safe (Okabe-Ito) Matplotlib figure builders for Hertzsprung-Russell (`plot_hr_diagram`), $T_c$ versus $\rho_c$ (`plot_tc_rhoc_diagram`), Kippenhahn convective/burning history (`plot_kippenhahn`), asteroseismic propagation diagrams (`plot_propagation_diagram`), binary RLOF evolution (`plot_binary_evolution`), and asteroseismic échelle and separation-ratio diagrams (`plot_echelle_diagram`, `plot_separation_ratios`).

::: flexmesa.plotting.style

::: flexmesa.plotting.stellar_plots

::: flexmesa.plotting.astero_binary_plots

---

## Four-Stage Performance Benchmark Harness (`flexmesa.bench`)

`flexmesa.bench` executes the 4-stage optimization staircase (`Stage 0 Sequential Baseline`, `Stage 1 Multi-Core FFI / SoA`, `Stage 2 @jax.jit`, `Stage 3 @jax.jit(vmap)`) and golden checksum verification across all `FlexMESA` domains.

::: flexmesa.bench.harness

::: flexmesa.bench.kernels_micro

::: flexmesa.bench.kernels_solver

::: flexmesa.bench.kernels_wave2

::: flexmesa.bench.kernels_wave3a

::: flexmesa.bench.kernels_wave3b

::: flexmesa.bench.kernels_wave4

::: flexmesa.bench.kernels_wave5
