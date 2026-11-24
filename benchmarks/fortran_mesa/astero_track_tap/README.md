# Per-Track MESA 26.4.1 Histories for Astero Search Cases (`#430`)

MESA's `astero` search (`astero/private/astero_run_support.f90`) evolves one
stellar track per objective evaluation: `eval1` calls
`run_star_support::run1_star` once per track (`astero_run_support.f90:224`).
Each run reopens `LOGS/history.data` without `position = 'append'` because
`doing_first_model_of_run` is set (`star/private/history.f90:313-321`), so a
finished search leaves only its last track in `LOGS/history.data`. The
`outputs/sample_NNNN.data` files MESA writes are 33-line best-model
summaries, not histories. This recipe records every track's history so that
the harness can compare FlexMESA's track `i` with MESA's track `i`. The
result ships as the `mesa-astero-tracks-v1.tar.zst` data bundle
(`src/flexmesa/data/bundles.toml`, fetched by
`flexmesa data fetch --include-test-inputs`), and the case manifest points at
it with `track_reference_dir = "astero/<case>"`.

## Files

| File | Role |
| :--- | :--- |
| `history_tap_wrap.c` | Linker wrapper for `__run_star_support_MOD_run1_star` (`-Wl,--wrap`). It calls the real `run1_star` and then copies `LOGS/history.data` to `history_tap/track_NNNN.data`, numbering every track from 1 whether or not the search accepted it. It changes no MESA source and no arithmetic. |
| `thin_track_histories.py` | Thins each tapped track with `thin_history_preserving_mesa_header` from `scripts/data/build_mesa_histories.py`, the function that produced the stored thinned references: MESA's 6-line header (`version_number`, `compiler`, `build`, `MESA_SDK_version`, `math_backend`, `date`, ...) is kept verbatim and the columns are the `STAR_KEEP_COLUMNS` subset present in the file (15 columns for these cases). `--last-reference` checks that the last track equals the stored thinned reference of the whole run below the header. |
| `build_bundle.py` | Packages the thinned tracks as `mesa_astero_tracks/astero/<case>/track_NNNN.data` with a `PROVENANCE.json` (MESA version, compiler, build and SDK from the tracks' headers, this recipe, and the per-case last-track check with the stored reference's SHA-256) into a reproducible `tar.zst`, and writes `src/flexmesa/data/bundle_hashes/mesa_astero_tracks.sha256.tsv`. |
| `run_tracks.sh` | Builds the wrapper, runs the five cases one at a time with `scripts/mesa_reference/run_mesa_case_26_4_1.sh` (`HISTORY_TAP_OBJ` adds the wrapper to `LOAD_EXTRAS1` next to the `pulse_tap` objects), thins the tracks, and builds the bundle. |

## Reproduce

```bash
OUT=<scratch dir> STORED_THINNED=<mesa_histories>/thinned/astero \
    bash benchmarks/fortran_mesa/astero_track_tap/run_tracks.sh
```

The build is MESA `26.4.1` with MESA SDK `x86_64-linux-26.6.1`
(`gfortran 15.2.0`, `CRMATH`), with the same `pulse_tap` link line as the
`mesa-histories-v1` campaign and 4 OpenMP threads. A rerun writes its own
date into each MESA header, so it gives a new archive SHA-256 with the same
data rows.

## `mesa-astero-tracks-v1` contents

| Case | Tracks | Samples MESA accepted | Last track equals stored reference below header |
| :--- | ---: | ---: | :--- |
| `example_astero` | 1 | 1 | yes |
| `fast_from_file` | 2 | 2 | yes |
| `fast_newuoa` | 8 | 8 | yes |
| `fast_scan_grid` | 4 | 4 | yes |
| `fast_simplex` | 7 | 6 | yes |

22 tracks, 3,579,117 bytes unpacked, 682,157 bytes compressed.
