# Wave 0 Technical Specification & Index/Memory-Order Contract

## 1. Overview & Sub-Issue Decomposition (`Issue #6`)

Wave 0 establishes the shared verification, schema-generation, and binary data-loading infrastructure required by both tracks (`crates/` in Safe Rust and `src/flexmesa/` in Python/JAX `float64`) before Wave 1 numerical and microphysics kernels are implemented.

Following the project's milestone-to-sub-issue workflow (`DECISIONS #5`), Wave 0 ([Issue #6](https://github.com/joshspeagle/FlexMESA/issues/6)) is decomposed into three focused pull requests:

| Sub-Issue | Branch / PR Scope | Primary Modules & Crates | Verification Gate |
| :--- | :--- | :--- | :--- |
| **[#7 (Wave 0A)](https://github.com/joshspeagle/FlexMESA/issues/7)** | Wave 0 specification, Index & Memory-Order Contract, and `<pkg>/test/test_output` golden baseline parser & numerical comparator | `docs/design/wave-0-spec.md`, `src/flexmesa/testing/`, `crates/flexmesa-testing/` | Parse and verify MESA `<pkg>/test/test_output*` files in `pytest`, `hypothesis`, and `cargo test` with $\le 10^{-12}$ relative tolerance. |
| **[#8 (Wave 0B)](https://github.com/joshspeagle/FlexMESA/issues/8)** | Declarative `<pkg>/defaults/*.defaults` and `*_columns.list` parser & Fortran `inlist` validator | `src/flexmesa/schema/`, `crates/flexmesa-schema/` | Parse MESA's `controls.defaults`, `star_job.defaults`, `eos.defaults`, and `kap.defaults` without textual `.inc` or `ctrls_io.f90` boilerplate. |
| **[#9 (Wave 0C)](https://github.com/joshspeagle/FlexMESA/issues/9)** | Physics table `.safetensors` converter (`isotopes.data`, `lodders03.data`, `helm_table`), `SHA-256` manifest (`manifest.toml`), and zero-copy `mmap` loader | `src/flexmesa/data/`, `crates/flexmesa-data/`, `scripts/convert_mesa_data.py` | Bit-exact round-trip parity between MESA ASCII tables and memory-mapped `float64` `.safetensors` tensors in Python/JAX and Safe Rust. |

---

## 2. Mandatory Index & Memory-Order Contract

Every module migrated from upstream MESA Fortran into Safe Rust (`crates/`) and Python/JAX (`src/flexmesa/`) obeys this single index and memory-layout contract so that array shapes, spline axes, and zone stencils never silently transpose or shift by 1.

### 2.1 Base Indexing (`1-Based Fortran` $\to$ `0-Based Rust / Python / JAX`)

- **Rule**: All public and internal array indices in `FlexMESA` are **0-based** (`0 .. n - 1`), translating Fortran's `1 .. n` loop bounds via $i_{\text{target}} = i_{\text{fortran}} - 1$.
- **Half-Open Intervals**: A Fortran slice `a(i_lo:i_hi)` (inclusive at both ends, length `i_hi - i_lo + 1`) maps to `a[i_lo - 1 .. i_hi]` in Rust and `a[i_lo - 1 : i_hi]` in Python/JAX.
- **Stellar Zone Ordering (`k = 0 .. nz - 1`)**:
  - MESA indexes stellar zones from the **surface (`k = 1` in Fortran)** inward to the **center (`k = nz` in Fortran)**.
  - `FlexMESA` preserves this physical **surface-to-center** ordering with 0-based indices:
    - `k = 0`: Surface cell (corresponding to Fortran `k = 1`).
    - `k = nz - 1`: Central core cell (corresponding to Fortran `k = s% nz`).
  - Stencil neighbors for interior cell `k` are `k - 1` (outer / nearer surface) and `k + 1` (inner / nearer center), matching MESA's 3-zone stencil `(k - 1, k, k + 1)`.

### 2.2 Memory Order & Struct-of-Arrays (SoA) Layout

Fortran stores multi-dimensional arrays in **column-major** order (`A(i, j)` has unit stride along index `i`), whereas C, Rust slices/ndarray, NumPy (default), and JAX/XLA store arrays in **row-major (C-contiguous)** order (`A[i, j]` has unit stride along the last index `j`).

To achieve both **cache-contiguous SIMD/GPU vectorization** across zones and **zero-copy interoperability** between Rust and Python/JAX via `.safetensors`, `FlexMESA` enforces the following canonical axis orderings:

| Domain Object | Original Fortran Shape (Column-Major) | `FlexMESA` Canonical Shape (Row-Major / C-Contiguous) | Unit-Stride (Innermost) Axis & Rationale |
| :--- | :--- | :--- | :--- |
| **Per-Zone Scalar Profile** (`lnT`, `lnd`, `lnR`, `L`, `dq`) | `(nz)` (`1:nz`) | `(nz,)` | `k` (`0..nz`): Contiguous 1D vector for SIMD and `vmap`. |
| **Per-Zone Mass Fractions** (`xa`) | `xa(species, nz)` (`1:species, 1:nz`) | `(nz, n_species)` in zone kernels; `(n_species, nz)` SoA views supported via explicit axis tags | In zone-parallel microphysics (`eos`, `kap`, `rates`, `net`), row `xa[k, :]` is contiguous per zone `k`; across `vmap(kernel, in_axes=0)`, the leading batch dimension is `nz`. |
| **1D Cubic Spline Coefficients** (`interp_1d`) | `f(4, nx)` (`f(1:4, i)` = $[f_i, f'_i, f''_i/2, f'''_i/6]$ or polynomial coefficients) | `(nx, 4)` | For grid node `i` (`0..nx`), the 4 cubic polynomial coefficients `[c0, c1, c2, c3]` sit contiguously in a single 32-byte cache line (`f[i, 0..4]`). |
| **2D Bicubic Spline Coefficients** (`interp_2d`) | `f(4, 4, nx, ny)` or `f(16, nx, ny)` | `(nx, ny, 4, 4)` (or flattened `(nx, ny, 16)`) | Evaluating point `(x, y)` in cell `(ix, iy)` reads all 16 bicubic coefficients (`128` bytes = 2 cache lines) contiguously from `f[ix, iy, :, :]`. |
| **Helmholtz Electron-Positron Table** (`helm_table`) | 9 arrays `(imax, jmax)` (`f, fd, ft, fdd, ftt, fdt, fddt, fdtt, fddtt`) + 4 `dpdf` + 4 `ef` + 4 `xf` (`imax=2701, jmax=1001`) | Individual named `(imax, jmax)` (`(2701, 1001)`) `float64` tensors in `helm_table.safetensors` + packed `(imax, jmax, 9)` free-energy jet tensor | Direct 2D index `(i_rho, j_temp)` maps to `[i_rho, j_temp]` in row-major order without transposing `d` (`imax`) and `t` (`jmax`). |
| **4D Tabulated EOS / Opacity Grids** (`eosDT`, `FreeEOS`, `kap`) | `(4, n_vars, n_logQ, n_logT, n_X, n_Z)` | `(n_Z, n_X, n_logT, n_logQ, n_vars, 4)` | Outermost axes are slow-varying composition `(Z, X)`; innermost axes are `(n_vars, 4)` spline coefficients at a given $(\log T, \log Q)$ cell. |
| **3-Zone Stencil Jacobian** (`star` / `turb` 33-var AD) | `d1Array(1:33)` (`11` vars $\times$ `3` zones: $k-1, k, k+1$) | `(nz, 3, n_var)` or `(nz, 33)` | Contiguous per-zone 33-element dual gradient vector `[d/dx_{k-1}, d/dx_k, d/dx_{k+1}]`. |
| **Block-Tridiagonal System** (`mtx/bcyclic`, `star_bcyclic`) | `lblk(nvar, nvar, nz)`, `dblk(nvar, nvar, nz)`, `ublk(nvar, nvar, nz)`, `rhs(nvar, nz)` | `lblk`, `dblk`, `ublk`: `(nz, nvar, nvar)`; `rhs`: `(nz, nvar)` | Each zone `k` owns a contiguous `(nvar, nvar)` dense matrix block and `(nvar,)` right-hand side vector for L1-cache-resident LU/GEMM. |

---

## 3. Golden Baseline Parser & Numerical Comparison Specification (`Wave 0A`)

MESA's 21 module unit-test suites emit 16-digit formatted text (`<pkg>/test/test_output*`) in three recurring patterns:

1. **Key-Value Scalar Lines** (e.g., `const/test/test_output`, `math/test/test_output.*`):
   - Format: `<label_string> <float_token>` (where `<float_token>` uses `E` or Fortran `D` exponent notation, e.g., `3.1415926535897931E+00` or `8.2630874248383526D-04`).
2. **Labeled Multi-Value / Indexed Lines** (e.g., `interp_1d/test/test_output`, `turb/test/test_output`):
   - Format: `<label_string> [int_index] <float_1> <float_2> ... <float_k>`.
3. **Dense Numeric Table Rows** (e.g., `neu/test/test_output` with 10,000 rows $\times$ 34 columns):
   - Format: `<float_1> <float_2> ... <float_m>` with no leading text label.

Both `flexmesa.testing.golden` (Python) and `crates/flexmesa-testing` (Safe Rust) provide:
- `is_fortran_float_token(token)` / `parse_fortran_float(token)`: Parses IEEE-754 `float64` strings with `e`/`E` or Fortran `d`/`D` exponents (as well as bare 2- or 3-digit signed Fortran exponents such as `1.0000000000000000-100`).
- `parse_golden_file(path)` / `parse_golden_text(text)` / `extract_golden_table(records)`: Extracts all numeric records (`GoldenRecord` with `line_number`, `section`, `label`, `indices: tuple[int, ...] / Vec<i64>`, and `values: NDArray[np.float64] / Vec<f64>`) while preserving section headers (`Test MLT`, `test1 T`, `Function Value (x= ...)`), and stacks uniform-width rows into a 2D table.
- `assert_values_match(actual, expected, rtol=1e-12, atol=1e-15)` and `assert_golden_records_match(actual, expected, rtol=1e-12, atol=1e-15)`: Computes elementwise absolute error $|a - e|$ and relative error $|a - e| / \max(|e|, \text{atol})$, checks $|a - e| \le \text{atol} + \text{rtol} \cdot |e|$, returns a `GoldenComparisonSummary`, and raises a diagnostic error pinpointing the first failing record, line number, label, and relative deviation.
