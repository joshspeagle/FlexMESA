---
name: Migration Wave Milestone
about: Tracking issue for a multi-PR migration wave across MESA's dependency DAG
labels: phase
---

## Wave Overview

<!-- Topological tier(s) and MESA modules covered in this wave. -->

## Linked Module Tasks

- [ ] #

## Wave Exit Criteria

- [ ] All constituent module unit tests and golden `test_output` parity checks pass (`<= 1e-12`).
- [ ] All files respect the `<= 400` LOC budget and `>= 20%` documentation ratio.
- [ ] 4-stage performance staircase and telemetry (`migration_telemetry.json`) recorded.
- [ ] Final audit (`/code-simplify` and `/code-review`) completed.
