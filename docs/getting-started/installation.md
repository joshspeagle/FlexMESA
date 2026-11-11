# Installation & Environment Setup

`FlexMESA` combines a compiled Safe Rust workspace (`crates/flexmesa-*`) with a Python and JAX package (`src/flexmesa/`) bridged via [`PyO3`](https://pyo3.rs) and [`maturin`](https://www.maturin.rs). This guide covers installing prerequisites, building `FlexMESA` from source, configuring multi-core CPU thread pools and 64-bit JAX execution on CPU or GPU, and managing `.safetensors` microphysics data tables.

---

## 1. Prerequisites

Building and running `FlexMESA` requires:

- **Python (`>= 3.12`)**: Required for modern type annotations (`type` aliases and PEP 695 generics) and `tomllib` manifest parsing.
- **Rust Toolchain (`>= 1.80`, `stable`)**: Required to compile the 21 Safe Rust workspace crates (`crates/flexmesa-*`) and the `flexmesa._rust` native extension module via `cargo` and `maturin`.
- **C/C++ Linker (`gcc` or `clang`)**: Standard platform linker used by `rustc` when producing the shared `PyO3` library (`_rust.abi3.so` on Linux/macOS or `_rust.pyd` on Windows). No Fortran compiler (`gfortran`) or MESA SDK installation is required to build or run `FlexMESA`.

To verify your Python and Rust toolchains before installation:

```bash
python3 --version
rustc --version
cargo --version
```

If Rust is not yet installed on your system, install the standard `stable` toolchain via [rustup](https://rustup.rs):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
```

---

## 2. Installing from Source (`uv` or `pip` + `maturin`)

Clone the `FlexMESA` repository and install the package in release mode. Because `pyproject.toml` declares `maturin` as the PEP 517 build backend, installing with `uv pip install` or `pip install` automatically compiles the Rust workspace and installs the Python and JAX modules alongside all required runtime dependencies (`numpy>=2.0`, `scipy>=1.12`, `jax>=0.4.30`, `safetensors>=0.4`, and `matplotlib>=3.9`).

=== "Using `uv` (Recommended)"

    ```bash
    git clone https://github.com/joshspeagle/FlexMESA.git
    cd FlexMESA
    uv pip install -e ".[test,docs]"
    ```

=== "Using `pip`"

    ```bash
    git clone https://github.com/joshspeagle/FlexMESA.git
    cd FlexMESA
    pip install -e ".[test,docs]"
    ```

=== "Direct `maturin develop` (For Active Rust + Python Development)"

    When iterating on the Safe Rust crates in `crates/`, invoke `maturin develop --release` directly to recompile `flexmesa._rust` with full Link-Time Optimization (`lto = "thin"`) and `opt-level = 3` in place:

    ```bash
    maturin develop --release
    ```

### Installing GPU-Accelerated JAX (Optional)

By default, `jax>=0.4.30` installs the CPU backend (`jaxlib`). On Linux workstations or cluster nodes equipped with NVIDIA GPUs, install the matching CUDA 12 plugin so `@jax.jit`, `jax.vmap`, and Parallel Cyclic Reduction (`solver_mode="pcr"`) execute on the GPU:

```bash
pip install -U "jax[cuda12]"
```

---

## 3. Verifying the Installation

Run the following one-liner to verify that the compiled `flexmesa._rust` extension, the built-in `.safetensors` physics tables, and 64-bit JAX precision are active:

```bash
python3 -c "import flexmesa, jax; print('FlexMESA', flexmesa.__version__, '| x64:', jax.config.jax_enable_x64, '| devices:', jax.devices())"
```

To run the fast verification suite across Python/JAX (`pytest`) and the Safe Rust workspace (`cargo test`):

```bash
pytest -q
cargo test --workspace --release
```

---

## 4. Runtime Configuration: Thread Pools & `float64` Precision

### Automatic IEEE-754 `float64` Precision in JAX
Stellar structure equations, bicubic and biquintic thermodynamic splines, and block-tridiagonal Henyey Jacobians require 64-bit IEEE-754 floating-point arithmetic (`f64` / `jnp.float64`) because 32-bit `float32` rounding errors ($\sim 10^{-7}$) exceed the convergence tolerance of hydrostatic equilibrium ($\le 10^{-10}$) and amplify curvature noise in second-order Helmholtz free-energy derivatives. Importing `flexmesa` automatically calls `flexmesa.enable_float64()`, which sets `jax_enable_x64 = True`.

If you import `jax` and allocate arrays before importing `flexmesa`, either set the environment variable before launching Python:

```bash
export JAX_ENABLE_X64=1
```

or call `flexmesa.enable_float64()` at the top of your script. To force CPU execution on a machine that also has a GPU (for example, in deterministic CI jobs or small single-zone scripts), set:

```bash
export JAX_PLATFORMS=cpu
```

### Controlling Safe Rust CPU Parallelism (`RAYON_NUM_THREADS`)
Multi-zone microphysics evaluators (`neu`, `turb`, `chem`, `ionization`, `colors`, `eos`, `kap`, `rates`, `net`, `atm`) and the Hirshman et al. (2010) `bcyclic` block-tridiagonal solver parallelize across radial shells using a persistent [`rayon`](https://docs.rs/rayon) worker pool in `flexmesa._rust`. You can control the number of CPU worker threads via standard environment variables:

```bash
export RAYON_NUM_THREADS=4
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
```

Setting `RAYON_NUM_THREADS` to match the physical core count (typically 4 to 8 threads for a single stellar track of $N_{\mathrm{zones}} \sim 10^3$ to $4 \times 10^3$) avoids thread-contention overhead when running ensembles of stellar tracks in parallel.

---

## 5. Physics Data Tables (`flexmesa data fetch` & `FLEXMESA_DATA_DIR`)

Upstream Fortran MESA distributes over 2 GiB of ASCII text tables paired with platform-dependent Fortran unformatted binary cache files. `FlexMESA` replaces these with content-addressed, zero-copy `.safetensors` and IEEE-754 little-endian binary table bundles verified by `SHA-256` checksums:

- **Full-Resolution `data-v1` Physics Bundles (`flexmesa data fetch`)**: After installing `FlexMESA`, fetch and verify the canonical full-resolution physics tables (`helm_table_full.bin`, full radiative and conductive opacity tables `kap_tables_full.bin`, and the 116 `OPAL`/`SCVH`, `FreeEOS`, and `CMS` 4D EOS composition slices) into `$FLEXMESA_DATA_DIR` (defaulting to `$XDG_CACHE_HOME/flexmesa/data` or `~/.cache/flexmesa/data`):

```bash
flexmesa data fetch
flexmesa data verify
flexmesa data status
```

  To also download the 3,142 upstream MESA `r24.08.1` golden `test_output` and `test_suite` input files into `~/.cache/flexmesa/mesa_test_inputs` for full end-to-end parity verification, pass `--include-test-inputs`:

```bash
flexmesa data fetch --include-test-inputs
```

- **Bundled Core Catalogs**: The reference nuclide catalog (347 built-in reference nuclides plus full `isotopes.data` parser), 286 solar isotopes across 10 solar abundance scales (`AG89` through `AAG21`, including `lodders03`, `gs98`, and `asplund09`), the Kurucz (2003) stellar atmosphere SED atlas (`lcb98cor.dat`), the 316 built-in REACLIB nuclear rates, and the Zero-Age Main Sequence (`ZAMS`) stellar catalog are embedded directly into the repository.
- **Cryptographic Integrity (`manifest.toml` & `bundles.toml`)**: Every `.safetensors` table bundle in `src/flexmesa/data/manifest.toml` and every release archive in `src/flexmesa/data/bundles.toml` carries its exact byte size and 64-character lowercase hexadecimal `SHA-256` digest. You can inspect both manifests from Python:

```python
from flexmesa.data import fetch, manifest

catalog = manifest.load_default_manifest()
for name, artifact in catalog.items():
    print(f"{name}: {artifact.filename} ({artifact.size_bytes} bytes, sha256={artifact.sha256[:12]}...)")

repo, tag, bundles = fetch.load_bundles_manifest()
print(f"Release {repo}@{tag}: {len(bundles)} verified archives")
```

- **Custom External Table Directory (`FLEXMESA_DATA_DIR`)**: When hosting physics table bundles on shared cluster storage, set the `FLEXMESA_DATA_DIR` environment variable (or pass `--dest` / `--data-dir` to `flexmesa data fetch`, `flexmesa data verify`, and `flexmesa data path`), and optionally set `MESA_DIR` to point to the extracted `mesa_test_inputs` directory. `flexmesa data fetch` and `flexmesa data verify` validate archive and unpacked `SHA-256` digests, byte sizes, and sentinel paths, while table loaders memory-map (`mmap`) binary physics tables directly into Safe Rust and JAX/NumPy without intermediate heap copies.
