# Open Architectural Questions (`QUESTIONS.md`)

All four foundational Stage 0/1 architectural questions have been resolved with Josh and recorded in [`notes/DECISIONS.md`](./DECISIONS.md):

1. **Target Language & Backend Architecture (`DECISIONS #1`)**: Wave 1 Dual-Track (`crates/` Safe Rust + `src/flexmesa/` Python/JAX `float64`) covering all 8 foundational numerical modules (`const`, `utils`, `math`, `auto_diff`, `interp_1d`, `interp_2d`, `mtx`, `num`) plus stateless Tier 3/5 microphysics modules `neu` and `turb`.
2. **Legacy & External Bundled Subsystem Exclusions (`DECISIONS #3`)**: Exclude `pgstar`/`pgbinary` (`#3`), external `stella`/`adipls`/`gyre` source ports (`#4`), and legacy `RSP1` (`#5`), replacing `pgstar` with Python/Matplotlib diagnostics, exporting `GYRE`/`FGONG` structures at the pulsation boundary, and porting unified `TDC`/`RSP2`.
3. **Physics Data Tables & 7-Component EOS Staging (`DECISIONS #4`)**: Convert MESA tables in Wave 0 into 64-byte-aligned `float64` `.safetensors` files (`mmap`-ready in `~/.cache/flexmesa/data/`), distributed as modular `.safetensors.zst` bundles on GitHub Release Assets with a committed `SHA-256` manifest, and stage `eos` into Wave 3a (`Ideal + HELM + PC + Skye`) and Wave 3b (`FreeEOS + OPAL/SCVH + CMS`).
4. **Stage 0 Toolchain Setup & Repository License (`DECISIONS #5`)**: Install `safetensors` in `~/.venvs/default`, `rustup` in `~/.cargo`, and the user-space MESA SDK in `~/.local/mesasdk`, and license `FlexMESA` under `LGPL-3.0-or-later`.
