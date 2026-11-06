# Wave 0 Infrastructure: Schemas, Binary Data & Golden Oracles

Wave 0 establishes the Tier A canonical Safe Rust infrastructure crates (`crates/flexmesa-schema`, `crates/flexmesa-data`, `crates/flexmesa-testing`) exposed to Python via zero-copy `PyO3` bindings (`flexmesa._rust`).

---

## 1. Typed Configuration Schemas (`flexmesa.schema`)

Upstream Fortran MESA configures evolutionary runs through untyped Fortran `.defaults` files (`star_job.defaults`, `controls.defaults`) and namelist files (`&star_job`, `&controls`, `&pgstar`) parsed via runtime `read(unit, nml=...)` statements. `flexmesa.schema` replaces untyped namelists with strongly typed, validated schemas (`ControlType`, `ControlDefault`, `ColumnEntry`, `NamelistAssignment`, `NamelistBlock`) backed by `crates/flexmesa-schema`:

- **Defaults and Column Catalog Parsing (`flexmesa.schema.defaults`)**: `parse_defaults_text`, `parse_defaults_file`, `parse_columns_text`, and `parse_columns_file` parse MESA `.defaults` and `columns.list` files into typed `ControlDefault` and `ColumnEntry` records (`ControlType.FLOAT`, `INT`, `BOOL`, `STRING`).
- **Fortran Namelist Ingestion and Validation (`flexmesa.schema.namelist`)**: `parse_namelist_text` and `parse_namelist_file` parse Fortran `&section ... /` blocks, `!` inline comments, `.true.` / `.false.` booleans, Fortran `1.0d-4` double-precision exponents, and 1-based indexed array assignments (`xa_central_lower_limit_species(1) = 'h1'`), while `validate_namelist_block` verifies every assignment against the reference defaults dictionary.

```python
from flexmesa.schema import defaults, namelist

defs = defaults.parse_defaults_text("""
initial_mass = 1.0d0
initial_z = 0.02d0
mixing_length_alpha = 2.0d0
""")
block = namelist.parse_namelist_text("""
&controls
  initial_mass = 1.0d0
  initial_z = 0.014d0
  mixing_length_alpha = 1.82d0
/
""")[0]
validated = namelist.validate_namelist_block(block, defs)
assert validated["initial_z"] == 0.014
```

---

## 2. Content-Addressed `.safetensors` Data Loader (`flexmesa.data`)

Upstream MESA stores opacity (`kap`), equation-of-state (`eosDT`, `eosPT`), and nuclear rate tables in ASCII text files paired with non-portable Fortran unformatted binary cache files. `flexmesa.data` (`crates/flexmesa-data`) replaces these with content-addressed `.safetensors` archives:

1. **Cryptographic Integrity (`flexmesa.data.manifest`)**: Every data table is described by a `TableArtifact` record and verified against its expected 64-character lowercase hexadecimal SHA-256 digest via `compute_sha256_bytes`, `compute_sha256_file`, `verify_file_sha256`, and `load_default_manifest`.
2. **Binary `.safetensors` Table Archives (`flexmesa.data.loader` & `flexmesa.data.converters`)**: `encode_safetensors_bytes` and `save_safetensors_bundle` write named `float64` and `int64` tensors in deterministic lexicographical key order with an 8-byte little-endian JSON header; `load_safetensors_bundle` decodes the archive in Safe Rust (`#![forbid(unsafe_code)]` in `flexmesa-data`) and returns a `LoadedTableBundle` of C-contiguous NumPy arrays via `flexmesa._rust`.

---

## 3. Golden Test Output Parser (`flexmesa.testing.golden`)

Every upstream MESA package includes a `test/test_output` reference file containing named labeled scalar and tabular floating-point dumps. `flexmesa.testing.golden` (`crates/flexmesa-testing`) parses these files into `GoldenRecord` structures via `parse_golden_text`, `parse_golden_file`, and `extract_golden_table` (backed by `flexmesa._rust.parse_golden_text`) and verifies relative and absolute tolerances ($\le 10^{-14}$) via `assert_values_match` and `assert_golden_records_match`, returning a `GoldenComparisonSummary`.
