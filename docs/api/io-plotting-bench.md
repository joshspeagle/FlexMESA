# I/O, Schemas, Plotting & Benchmarks (`schema`, `data`, `testing`, `plotting`, `bench`)

This reference page documents the configuration schema and Fortran `&namelist` parser (`flexmesa.schema`), the `.safetensors` memory-mapped data loader and `SHA-256` manifest verifier (`flexmesa.data`), the upstream `test_output` reference and MESA `26.4.1` trajectory comparators (`flexmesa.testing`), the publication-styled diagnostic plotting suite (`flexmesa.plotting`), and the 4-stage benchmark harness (`flexmesa.bench`).

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

`flexmesa.data` validates archive and per-file `SHA-256` checksums against `manifest.toml`, `bundles.toml`, and `bundle_hashes/*.sha256.tsv` (`flexmesa data fetch`, `flexmesa data verify`, rejecting unexpected files inside read-only `$FLEXMESA_DATA_DIR`), isolates versioned runtime-derived caches under `$FLEXMESA_CACHE_DIR/v1` (`resolve_cache_dir`, `FLEXMESA_DISABLE_RUNTIME_CACHE`, `flexmesa data clean-cache`), memory-maps 64-byte-aligned `.safetensors` physics tables directly from the OS page cache, and converts built-in ASCII MESA tables (`lodders03.data`, `isotopes.data`, and `helm_table.dat`) into `.safetensors` (while full-grid `data-v1` binary bundles are built offline via `make data-bundles`).

::: flexmesa.data.manifest

::: flexmesa.data.fetch

::: flexmesa.data.loader

::: flexmesa.data.converters

---

## Upstream `test_output` Reference & Trajectory Harness (`flexmesa.testing`)

`flexmesa.testing` parses upstream MESA `<pkg>/test/test_output` reference files (including Fortran `1pe24.16` and missing-`E` three-digit exponents like `1.2345-102`), validates the 106-case MESA `26.4.1` trajectory manifest and declared known-gaps registry (`case_manifest.toml`, `known_gaps.toml`), compares `LOGS/history.data` and `binary_history.data` tracks on a shared `float64` coordinate grid, and drives `flexmesa trace` and `flexmesa harness` (supporting `--numerics-mode {strict,fast}`, with `--gate` restricted to `strict`, and defaulting to strict rejection of unsupported controls unless `--allow-unsupported-controls` is explicitly given).

::: flexmesa.testing.golden

::: flexmesa.testing.case_manifest

::: flexmesa.testing.history_compare

::: flexmesa.testing.harness_runner

::: flexmesa.testing.case_report

::: flexmesa.testing.baseline_regression

`flexmesa.testing.harness_shards` splits a harness campaign into disjoint shards (`flexmesa harness run --shard I/N`) and checks that the shard reports form one complete campaign before `flexmesa harness merge` combines them (`#389`).

::: flexmesa.testing.harness_shards

::: flexmesa.testing.port_manifest

::: flexmesa.testing.dashboard

::: flexmesa.testing.trace_cli

`flexmesa.testing.stepdump_compare` compares a FlexMESA step dump with a MESA `26.4.1` step dump record by record and reports the first difference: the step, solver record, zone and variable, with its ULP distance (`flexmesa trace <suite> <case> --compare <dump>`, `#372`).

::: flexmesa.testing.stepdump_compare

::: flexmesa.testing.known_gaps_gate

::: flexmesa.testing.module_hex_gate

---

## Publication-Styled Diagnostic Plotting (`flexmesa.plotting`)

`flexmesa.plotting` replaces Fortran MESA's legacy X11/PGPLOT `PGSTAR` and `PGBINARY` subsystems with composable, colorblind-safe (Okabe-Ito) Matplotlib figure builders for Hertzsprung-Russell (`plot_hr_diagram`), $T_c$ versus $\rho_c$ (`plot_tc_rhoc_diagram`), Kippenhahn convective/burning history (`plot_kippenhahn`), asteroseismic propagation diagrams (`plot_propagation_diagram`), binary RLOF evolution (`plot_binary_evolution`), and asteroseismic échelle and separation-ratio diagrams (`plot_echelle_diagram`, `plot_separation_ratios`).

::: flexmesa.plotting.style

::: flexmesa.plotting.stellar_plots

::: flexmesa.plotting.astero_binary_plots

::: flexmesa.plotting.compare

::: flexmesa.plotting.kippenhahn

::: flexmesa.plotting.pgstar_panels

::: flexmesa.plotting.pgstar_summary

::: flexmesa.plotting.pgstar_text

::: flexmesa.plotting.pgstar_extra

::: flexmesa.plotting.pgstar_nucleosyn

::: flexmesa.plotting.pgbinary_panels

::: flexmesa.plotting.pgstar_grid

::: flexmesa.plotting.live

::: flexmesa.plotting.pgstar_coverage

::: flexmesa.plotting.cli

---

## Four-Stage Performance Benchmark Harness (`flexmesa.bench`)

`flexmesa.bench` executes the 4-stage optimization staircase (`Stage 0 Sequential Baseline`, `Stage 1 Multi-Core FFI / SoA`, `Stage 2 @jax.jit`, `Stage 3 @jax.jit(vmap)`) and reports regression summary checksums across standalone kernels (post-audit MESA `26.4.1` comparator re-runs and timing provenance are tracked under `#146`).

::: flexmesa.bench.harness

::: flexmesa.bench.kernels_micro

::: flexmesa.bench.kernels_solver

::: flexmesa.bench.kernels_wave2

::: flexmesa.bench.kernels_wave3a

::: flexmesa.bench.kernels_wave3b

::: flexmesa.bench.kernels_wave4

::: flexmesa.bench.kernels_wave5
