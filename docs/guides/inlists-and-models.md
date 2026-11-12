# User Guide: Inlist Bundles, `.mod` Files, Binary Photos & Data Caches

Upstream Fortran MESA workflows organize stellar evolution runs around hierarchical Fortran namelists (`inlist` files chained via `read_extra_star_job_inlist(1..5)` and `read_extra_controls_inlist(1..5)`), multi-stage `./rn` shell scripts, ASCII `.mod` structure files, and binary `photos/` checkpoint files. `FlexMESA` provides first-class parsers, validators, and bit-exact I/O routines for every one of these artifacts so you can run or migrate existing MESA `test_suite` directories directly from Python or Rust.

---

## 1. Parsing Recursive `inlist` Bundles and Validating Against `.defaults`

In Fortran MESA, a top-level `inlist` file often delegates shared settings to `inlist_project`, `inlist_common`, or stage-specific namelists via `read_extra_star_job_inlist1 = .true.` and `extra_star_job_inlist1_name = '...'`. `FlexMESA` provides two levels of namelist tooling:

1. **Recursive Bundle Resolution (`star.parse_inlist_bundle`)**: Traverses the entry `inlist` and all nested `read_extra_*_inlist(1..5)` files in upstream precedence order, returning a normalized Python dictionary of `&star_job` and `&controls` parameters (`initial_mass`, `initial_z`, `initial_y`, `create_pre_ms_model`, `load_saved_model`, `saved_model_name`, `save_model_when_terminate`, `save_model_filename`, `change_net`, `new_net_name`, `max_model_number`, `max_age`, `v_flag`, `u_flag`, `rti_flag`, `rotation_flag`, `rsp2_flag`, `okay_to_remesh`, `use_gold_tolerances`, `use_gold2_tolerances`, `op_split_burn`, `op_split_burn_min_t`).
2. **Typed Schema Validation (`schema.defaults`, `schema.namelist`)**: Parses arbitrary `.defaults` specification files and validates raw `&namelist` blocks (`&star_job`, `&controls`, `&binary_controls`, `&astero_search_controls`), catching misspelled parameter names or invalid Fortran literal types (`1.0d-4`, `.true.`, array subscripts `xa_central_lower_limit(1)`) before a run starts.

```python
import pathlib
import tempfile
from flexmesa import schema, star

# 1. Resolve a multi-file recursive MESA inlist bundle
with tempfile.TemporaryDirectory() as tmpdir:
    root = pathlib.Path(tmpdir)
    (root / "inlist").write_text(
        "&star_job\n"
        "  read_extra_star_job_inlist1 = .true.\n"
        "  extra_star_job_inlist1_name = 'inlist_core'\n"
        "/\n"
        "&controls\n"
        "  read_extra_controls_inlist1 = .true.\n"
        "  extra_controls_inlist1_name = 'inlist_core'\n"
        "/\n"
    )
    (root / "inlist_core").write_text(
        "&star_job\n"
        "  change_net = .true.\n"
        "  new_net_name = 'approx21.net'\n"
        "  save_model_when_terminate = .true.\n"
        "  save_model_filename = 'tams.mod'\n"
        "/\n"
        "&controls\n"
        "  initial_mass = 2.5d0\n"
        "  initial_z = 0.014d0\n"
        "  max_model_number = 50\n"
        "  rotation_flag = .true.\n"
        "/\n"
    )
    bundle = star.parse_inlist_bundle(str(root / "inlist"))
    print(
        f"Resolved bundle: M={bundle['initial_mass']} Msun, Z={bundle['initial_z']}, "
        f"net={bundle['new_net_name']!r}, save_to={bundle['save_model_filename']!r}"
    )

# 2. Validate custom namelist assignments against a typed .defaults schema
defaults_spec = schema.defaults.parse_defaults_text("""
initial_mass = 1.0d0
initial_z = 0.02d0
mixing_length_alpha = 2.0d0
use_ledoux_criterion = .false.
""")
blocks = schema.namelist.parse_namelist_text("""
&controls
  initial_mass = 3.0d0
  mixing_length_alpha = 1.85d0
  use_ledoux_criterion = .true.
/
""")
validated = schema.namelist.validate_namelist_block(blocks[0], defaults_spec)
print("Validated controls:", validated)
```

---

## 2. Executing Multi-Stage `./rn` Test-Suite Scripts

Many upstream MESA `test_suite` cases chain multiple evolutionary stages inside a `./rn` bash script using `do_one <inlist> <final_mod>` helper calls, where Stage 1 relaxes a pre-MS or ZAMS model and saves `stage1.mod`, and Stage 2 loads `stage1.mod` (`load_saved_model = .true.`) to evolve through helium burning or mass transfer.

`star.parse_rn_stages` extracts the ordered list of `(stage_kind, inlist_name, output_mod)` stages from a case directory (respecting `MESA_SKIP_OPTIONAL` conditional blocks when `skip_optional=True`), and `star.run_test_suite_case` executes the entire multi-stage pipeline end-to-end with optional `.photo` restart verification. For upstream `star/test_suite` cases that include custom Fortran `src/run_star_extras.f90` hooks, `star.TIER1_CASES`, `star.run_test_suite_case`, and `star.run_tier1_suite` automatically attach the built-in `Tier1CaseExtras` hook registry and verify physical end-of-run invariants:

```python
import pathlib
import tempfile
from flexmesa import star

with tempfile.TemporaryDirectory() as tmpdir:
    case_dir = pathlib.Path(tmpdir) / "two_stage_case"
    case_dir.mkdir()

    # Stage 1: Evolve 1 Msun ZAMS for 2 steps and save stage1.mod
    (case_dir / "inlist_stage1").write_text(
        "&star_job\n"
        "  save_model_when_terminate = .true.\n"
        "  save_model_filename = 'stage1.mod'\n"
        "/\n"
        "&controls\n"
        "  initial_mass = 1.0d0\n"
        "  initial_z = 0.02d0\n"
        "  max_model_number = 2\n"
        "/\n"
    )
    # Stage 2: Load stage1.mod and evolve 2 additional steps
    (case_dir / "inlist_stage2").write_text(
        "&star_job\n"
        "  load_saved_model = .true.\n"
        "  saved_model_name = 'stage1.mod'\n"
        "  save_model_when_terminate = .true.\n"
        "  save_model_filename = 'final.mod'\n"
        "/\n"
        "&controls\n"
        "  max_model_number = 4\n"
        "/\n"
    )
    (case_dir / "rn").write_text(
        "#!/usr/bin/env bash\n"
        "do_one inlist_stage1 stage1.mod\n"
        "do_one inlist_stage2 final.mod\n"
    )

    stages = star.parse_rn_stages(str(case_dir), skip_optional=True)
    summary = star.run_test_suite_case(
        str(case_dir),
        skip_optional=True,
        max_steps_per_stage=2,
        verify_restart=True,
    )
    print(f"Stages parsed: {stages}")
    print(
        f"Case '{summary['case_name']}': total_steps={summary['total_steps']}, "
        f"Newton_iters={summary['total_newton_iterations']}, "
        f"photo_restart_verified={summary['photo_restart_verified']}"
    )
```

---

## 3. ASCII `.mod` Files and Bit-Identical Binary `photos/` Checkpoints

`StarModel` supports two complementary serialization formats:

- **ASCII `.mod` Files (`to_mod_str`, `from_mod_str`, `write_mod_file`, `from_mod_file`)**: Human-readable MESA model files storing header scalars (`M/Msun`, `model_number`, `star_age`, `net_name`, `species`, `n_shells`), per-zone structural variables (`lnd`, `lnT`, `lnR`, `L`, `dq`), and isotope mass fractions $X_{i,k}$ formatted in scientific notation.
- **Binary `photos/` Snapshots (`to_photo_bytes`, `from_photo_bytes`, `verify_photo_restart`)**: Exact IEEE-754 little-endian binary dumps of every primary, old, and start-of-step solver state array plus timestep controller history (`dt`, `dt_old`, `varcontrol`). Restoring from a binary photo guarantees **zero-ULP bit-identical evolution** compared to an uninterrupted run (`verify_photo_restart`).

```python
import pathlib
import tempfile
import numpy as np
from flexmesa import star

model = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
model.evolve(max_steps=2, use_bcyclic=True)

# 1. Round-trip through an ASCII .mod file
mod_text = model.to_mod_str()
reloaded_mod = star.StarModel.from_mod_str(mod_text, eval_hydro=True)
max_dlnt = float(np.max(np.abs(np.asarray(model.ln_t) - np.asarray(reloaded_mod.ln_t))))

with tempfile.TemporaryDirectory() as tmpdir:
    mod_path = pathlib.Path(tmpdir) / "1m_evolved.mod"
    model.write_mod_file(mod_path)
    from_disk = star.StarModel.from_mod_file(mod_path, eval_hydro=True)

# 2. Round-trip through a binary photo and verify bit-identical restart parity
photo_bytes = model.to_photo_bytes()
restored_photo = star.StarModel.from_photo_bytes(photo_bytes, eval_hydro=True)
assert np.array_equal(np.asarray(model.ln_t), np.asarray(restored_photo.ln_t))

hash_cont, hash_rest = model.verify_photo_restart(n_steps=2)
print(
    f"ASCII .mod max |d(lnT)| = {max_dlnt:.2e}, photo size = {len(photo_bytes)} bytes, "
    f"restart state hashes: continuous={hash_cont:#x}, restored={hash_rest:#x}"
)
```

---

## 4. Managing Physics Data Bundles and `.safetensors` Tables (`flexmesa.data`)

Full-grid equation-of-state and opacity tables (`helm_table_full.bin`, `kap_tables_full.bin`, and 4D `OPAL`/`SCvH`/`CMS`/`FreeEOS` binaries) plus upstream MESA verification inputs are distributed via the `data-v1` release manifest (`bundles.toml`) and fetched on demand with `flexmesa data fetch` (or `flexmesa data fetch --include-test-inputs`), which verifies every compressed `.zst` archive against its pinned `SHA-256` digest before unpacking into `$FLEXMESA_DATA_DIR`.

In addition, `flexmesa.data` stores custom tabulated physics payloads in 64-byte-aligned `.safetensors` files verified against `SHA-256` digests in `manifest.toml`. You can inspect the built-in manifest, verify checksums, and round-trip custom `float64` tensor bundles in memory or on disk:

```python
import pathlib
import tempfile
import numpy as np
from flexmesa.data import loader, manifest

# 1. Inspect the default SHA-256 table manifest
default_tables = manifest.load_default_manifest()
for name, art in default_tables.items():
    print(f"{name}: {art.filename} ({art.size_bytes} bytes, tensors={art.tensors})")

# 2. Save, SHA-256 verify, and memory-map a custom float64 .safetensors bundle
with tempfile.TemporaryDirectory() as tmpdir:
    bundle_path = pathlib.Path(tmpdir) / "custom_table.safetensors"
    tensors = {
        "log_t_grid": np.linspace(3.5, 8.5, 51, dtype=np.float64),
        "log_rho_grid": np.linspace(-8.0, 6.0, 29, dtype=np.float64),
    }
    sha256_hex = loader.save_safetensors_bundle(
        bundle_path,
        tensors,
        metadata={"source": "FlexMESA tutorial", "units": "CGS"},
    )
    loaded = loader.load_safetensors_bundle(
        bundle_path,
        expected_sha256=sha256_hex,
        required_tensors=("log_t_grid", "log_rho_grid"),
    )
    print(
        f"Verified SHA-256={sha256_hex[:16]}..., "
        f"loaded shapes={[v.shape for v in loaded.tensors.values()]}"
    )
```
