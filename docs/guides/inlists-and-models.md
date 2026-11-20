# User Guide: Inlist Bundles, `.mod` Files, Binary Photos & Data Caches

Upstream Fortran MESA workflows organize stellar evolution runs around hierarchical Fortran namelists (`inlist` files chained via `read_extra_star_job_inlist(1..5)` and `read_extra_controls_inlist(1..5)`), multi-stage `./rn` shell scripts, ASCII `.mod` structure files, and binary `photos/` checkpoint files. `FlexMESA` provides parsers, schema validators, and ASCII/binary I/O helpers (recording the active `numerics_mode` in `LOGS/`, `.mod`, and internal `.photo` headers). Strict runtime `inlist` loading fails loudly by default on unported control keys (`#147`; pass `allow_unsupported_controls=True` only for diagnostic runs), and `FlexMESA` `.photo` buffers (`FMPHOTO2` v26 for `StarModel` and `FMBINCP1` v2 for `BinarySystem`) are internal full-state checkpoints not binary-compatible with Fortran MESA `photos/` (`#150`, `#149`, `#153`, `#155`, `#156`, `#159`, `#160`, `#161`).

---

## 1. Parsing Recursive `inlist` Bundles and Validating Against `.defaults`

In Fortran MESA, a top-level `inlist` file often delegates shared settings to `inlist_project`, `inlist_common`, or stage-specific namelists via `read_extra_star_job_inlist1 = .true.` and `extra_star_job_inlist1_name = '...'`. `FlexMESA` provides two levels of namelist tooling:

1. **Recursive Bundle Resolution (`star.parse_inlist_bundle`)**: Traverses the entry `inlist` and all nested `read_extra_*_inlist(1..5)` files in upstream precedence order, failing loudly by default on unknown, unread, or unsupported controls unless `allow_unsupported_controls=True` is passed, and returning a normalized Python dictionary of `&star_job` and `&controls` parameters (`initial_mass`, `initial_z`, `initial_y`, `create_pre_ms_model`, `load_saved_model`, `load_model_filename`, `save_model_when_terminate`, `save_model_filename`, `change_net`, `new_net_name`, `max_model_number`, `max_age`, `v_flag`, `u_flag`, `rti_flag`, `rotation_flag`, `rsp2_flag`, `okay_to_remesh`, `use_gold_tolerances`, `use_gold2_tolerances`, `op_split_burn`, `op_split_burn_min_t`), together with 1-based array controls on `MixWindControls` and `TimestepStopControls` (`overshoot_scheme(1..16)`, `overshoot_zone_type(1..16)`, `overshoot_f(1..16)`, `overshoot_f0(1..16)`, `predictive_mix(1..16)`, `predictive_superad_thresh(1..16)`, `hard_limit_for_rel_error_in_energy_conservation`, and `min_allowed_varcontrol_target`). Keys inside `&pgstar`, `&pgbinary`, and `&astero_pgstar_controls` (plus `pgstar_flag` / `pgbinary_flag`) are validated against upstream `.defaults` schemas so misspelled plotting keys still fail loudly while valid plotting keys execute as headless no-ops (matching upstream `USE_PGSTAR = NO`).
2. **Typed Schema Validation (`schema.defaults`, `schema.namelist`)**: Parses arbitrary `.defaults` specification files and validates raw `&namelist` blocks (`&star_job`, `&controls`, `&eos`, `&kap`, `&binary_job`, `&binary_controls`, `&astero_search_controls`), catching misspelled parameter names or invalid Fortran literal types (`1.0d-4`, `.true.`, array subscripts `xa_central_lower_limit(1)`) before a run starts.

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

`star.parse_rn_stages` extracts the ordered list of `(stage_kind, inlist_name, output_mod)` stages from a case directory (respecting `MESA_SKIP_OPTIONAL` conditional blocks when `skip_optional=True`), and `star.run_test_suite_case` executes multi-stage inlists with optional internal `.photo` round-trip checks and caller-supplied Python `extras` hooks (while canonical MESA `26.4.1` trajectory evaluation runs through `flexmesa harness`):

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
        "&kap\n"
        "  zbase = 0.02d0\n"
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
        "  load_model_filename = 'stage1.mod'\n"
        "  save_model_when_terminate = .true.\n"
        "  save_model_filename = 'final.mod'\n"
        "/\n"
        "&kap\n"
        "  zbase = 0.02d0\n"
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

## 3. ASCII `.mod` Files, `LOGS/` History & Profile Output, and Binary `photos/` Checkpoints

`StarModel` and the `inlist` stage runners support upstream MESA model, log, and checkpoint formats:

- **Incremental `LOGS/` History, Radial Profiles & Binary History (`history.data`, `profileN.data`, `profiles.index`, `binary_history.data`)**: In Rust (`run_inlist_stage_with_logs`, `run_test_suite_case_with_logs`, `StarLogsWriter`, `run_binary_test_suite_case_with_logs`, `BinaryLogsWriter`) and the `star_case_trace` / `binary_case_trace` drivers (plus `StarModel.to_profile_str()` in Python, which formats the 6-line-header MESA `profileN.data` snapshot), log emission honors `&star_job` (`history_columns_file`, `profile_columns_file`, `write_profile_when_terminate`, `filename_for_profile_when_terminate`, `profile_starting_model`), `&controls` (`log_directory`, `star_history_name`, `star_history_header_name`, `history_interval`, `write_profiles_flag`, `profile_interval`, `priority_profile_interval`, `profile_data_header_suffix`, `max_num_profile_models`, `max_num_profile_zones`, `burn_min1`, `burn_min2`, `center_avg_value_dq`, `surface_avg_abundance_dq`), `&binary_job` (`binary_history_columns_file`), and `&binary_controls` (`log_directory`, `history_name`, `history_interval`, `append_to_star_history`), flushing each recorded history row immediately and writing `profileN.data` before updating `profiles.index`. (The default Python `run_test_suite_case` helper runs in-memory without writing `LOGS/` unless invoked via the log-emitting Rust drivers.)
- **ASCII `.mod` Files (`to_mod_str`, `from_mod_str`, `write_mod_file`, `from_mod_file`)**: Human-readable MESA model files storing header scalars (`M/Msun`, `model_number`, `star_age`, `net_name`, `species`, `n_shells`), per-zone structural variables (`lnd`, `lnT`, `lnR`, `L`, `dq`), and isotope mass fractions $X_{i,k}$ formatted in scientific notation.
- **Internal Binary `photos/` Checkpoints (`to_photo_bytes`, `from_photo_bytes`, `verify_photo_restart`)**: Deterministic little-endian `serde` byte buffers (`FMPHOTO2` v26 for `StarModel`, `FMBINCP1` v2 for `BinarySystem`) storing primary, old, and start-of-step solver state arrays alongside `controls`, `job`, `rsp`, `BinaryStepCounters`, retry/redo counters, and the active `numerics_mode` wire tag (`strict` or `fast`, rejecting mismatched modes on load) while skipping ephemeral `MicrophysicsCache` hit/miss counters. `from_photo_bytes(bytes)` restores serialized `controls` and `job` (`read_photo_bytes` overrides them when requested), and `verify_photo_restart(n_steps)` verifies that checkpointing and restarting produces bit-identical continuation for `n_steps` under the same `numerics_mode`. `FlexMESA` `.photo` bytes remain internal and are not binary-compatible with Fortran MESA `photos/` files (`#150`, `#149`, `#153`, `#155`, `#156`, `#159`, `#160`, `#161`).

```python
import pathlib
import tempfile
import numpy as np
from flexmesa import star

model = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
model.evolve(max_steps=2, use_bcyclic=True)

# 1. Round-trip through an ASCII .mod file (header does not store &kap::zbase)
mod_text = model.to_mod_str()
reloaded_mod = star.StarModel.from_mod_str(mod_text)
max_dlnt = float(np.max(np.abs(np.asarray(model.ln_t) - np.asarray(reloaded_mod.ln_t))))

with tempfile.TemporaryDirectory() as tmpdir:
    mod_path = pathlib.Path(tmpdir) / "1m_evolved.mod"
    model.write_mod_file(mod_path)
    from_disk = star.StarModel.from_mod_file(mod_path)

# 2. Round-trip through a binary photo and check n_steps continuation state hashes
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

Full-grid equation-of-state and opacity tables (`helm_table_full.bin`, `kap_tables_full.bin`, and 4D `OPAL`/`SCvH`/`CMS`/`FreeEOS` binaries) plus upstream MESA test inputs, module hex outputs, and reference histories are distributed via the `data-v1` release manifest (`bundles.toml`) and fetched on demand with `flexmesa data fetch` (or `flexmesa data fetch --include-test-inputs`). `flexmesa data verify` checks every unpacked file against per-file `SHA-256` manifests (`bundle_hashes/*.sha256.tsv`) and rejects unexpected files inside the read-only `$FLEXMESA_DATA_DIR` directory, while runtime-derived spline caches live separately under `$FLEXMESA_CACHE_DIR/v1` (`FMCENV01` envelope, keyed by `numerics_mode` for 4D EOS slices, controllable via `FLEXMESA_DISABLE_RUNTIME_CACHE=1` and `flexmesa data clean-cache`).

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

---

## 5. Incremental Trajectory Tracing and Harness Comparisons (`flexmesa trace`, `flexmesa harness`)

To run any upstream `star`, `binary`, or `astero` test-suite case and stream MESA-formatted `LOGS/history.data` or `binary_history.data` rows on every accepted step (preserving partial trajectories if a wall-clock or memory guard stops the run), invoke `flexmesa trace` (strict inlist validation fails loudly on unsupported controls unless `--allow-unsupported-controls` is passed):

```bash
flexmesa trace star 1.3M_ms_high_Z --out ./run_out --timeout-sec 600 --max-rss-gb 6.0 --allow-unsupported-controls
flexmesa trace binary jdot_gr_check --out ./bin_out --timeout-sec 600 --max-rss-gb 6.0 --allow-unsupported-controls
flexmesa trace astero surface_effects --out ./astero_out --allow-unsupported-controls
```

Unmodified upstream inlists set controls that are not yet ported. Without `--allow-unsupported-controls` each command stops and prints the full list of unsupported controls. With the flag the same list is printed as a warning and the run continues past them, so the resulting trajectory is exploratory and is never counted as parity evidence.

To regenerate the 106-case MESA `26.4.1` reference-availability table (`notes/audit/case_table.md`) or run guarded trajectory comparisons against `benchmarks/history_harness/case_manifest.toml` and `benchmarks/history_harness/known_gaps.toml`, invoke `flexmesa harness`. `harness run` runs each case under strict control validation first. When that run stops on unsupported controls, the harness runs the case again with those controls ignored and reports the second run in the `Run Mode` column as `diagnostic (N controls ignored)`. A diagnostic run shows where the engine stands and can never give a `pass` verdict (`DECISIONS #26`). Keys that only gate or format MESA's terminal text, such as `terminal_interval` and `trace_history_value_name`, pass strict validation and have no effect, because `FlexMESA` does not print MESA's terminal log. Each uncapped strict run is also compared with the committed baseline run (`benchmarks/history_harness/baseline_run_26_4_1.json`, a regression snapshot of what `FlexMESA` did, not a MESA reference). `--gate` fails when a case that took steps now takes none, no longer reaches upstream's termination condition, or leaves the manifest tolerances; runs cut off by the time or memory budget are not judged. A change that makes a case worse on purpose updates the baseline for that case with `--update-baseline` (`DECISIONS #27`). Each finished case is also appended to `progress.jsonl` in the work directory, so a run that is cancelled part way still leaves what it measured:

```bash
flexmesa harness table --check
flexmesa harness run --preset pr-ci --markdown-out case_table.md --json-out case_table.json --gate
flexmesa harness run --cases star/1.3M_ms_high_Z --update-baseline benchmarks/history_harness/baseline_run_26_4_1.json
```
