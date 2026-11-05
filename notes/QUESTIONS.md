# Open Architectural Questions (`QUESTIONS.md`)

All foundational architectural questions for Wave 0 and Wave 1 have been resolved and recorded in [`notes/DECISIONS.md`](./DECISIONS.md):

1. **Target Language & Backend Architecture (`DECISIONS #1`)**: Wave 1 Dual-Track (`crates/` Safe Rust + `src/flexmesa/` Python/JAX `float64`) covering all 8 foundational numerical modules (`const`, `utils`, `math`, `auto_diff`, `interp_1d`, `interp_2d`, `mtx`, `num`) plus stateless Tier 3/5 microphysics modules `neu` and `turb`.
2. **Legacy & External Bundled Subsystem Exclusions (`DECISIONS #2`)**: Exclude `pgstar`/`pgbinary` (`#3`), external `stella`/`adipls`/`gyre` source ports (`#4`), and legacy `RSP1` (`#5`), replacing `pgstar` with Python/Matplotlib diagnostics, exporting `GYRE`/`FGONG` structures at the pulsation boundary, and porting unified `TDC`/`RSP2`.
3. **Physics Data Tables & 7-Component EOS Staging (`DECISIONS #3`)**: Convert MESA tables in Wave 0 into 64-byte-aligned `float64` `.safetensors` files (`mmap`-ready in `~/.cache/flexmesa/data/`), distributed as modular `.safetensors.zst` bundles on GitHub Release Assets with a committed `SHA-256` manifest, and stage `eos` into Wave 3a (`Ideal + HELM + PC + Skye`) and Wave 3b (`FreeEOS + OPAL/SCVH + CMS`).
4. **Repository License & Portable Configuration (`DECISIONS #4`)**: License `FlexMESA` under `LGPL-3.0-or-later` and keep build/test configuration portable across platforms.

## Upcoming Decision Gates

- **Post-Wave 1 Backend Selection (`Safe Rust` vs. `Python/JAX` vs. `Hybrid`)**: Scheduled upon completion of the 10 dual-track Wave 1 modules and their 4-stage performance and readability evaluation.
