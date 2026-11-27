# Routine-level port table (generated)

<!-- GENERATED VIA scripts/dashboard/build_routine_table.py (#611). Do not edit by hand. -->

Derived from `benchmarks/dashboard/routine_inventory.json` (MESA r26.4.1, commit `c8c2026`), the `routine_map` entries of `benchmarks/dashboard/port_manifest.toml`, the oracle routine lists of `benchmarks/dashboard/test_taxonomy.toml`, and the declared gaps of `benchmarks/history_harness/known_gaps.toml`. Machine-readable copy: `notes/audit/routine_table.json`.

## Evidence levels

Every routine is at one level. Lines are credited to a routine for its own lines only (its span minus the spans of the procedures contained in it), so a mapped parent does not credit its unmapped children, and the own lines of all routines plus the lines outside any routine equal the file's line count.

* `full`: the `routine_map` names the routine, the target passes the `routine_map` gate, and the routine is not listed in `routine_map_partial`.
* `partial`: mapped as above and listed in the file's `routine_map_partial`, because the Rust counterpart implements only part of the upstream routine.
* `N/A`: declared not applicable to a Rust port in `benchmarks/dashboard/port_status_na.toml` (0 declared). Never counted as mapped and never removed from a denominator; both fractions are shown.
* `unmapped`: none of the above.
* `oracle`: the routine is mapped in full and a test file classified `oracle` in the taxonomy lists its `<upstream path>::<routine>` id in its `routines` field. This is the taxonomy's own record of per-routine oracle cover; it is a listing, not a call-site check.
* `exercised` (reached by a strict-pass history case) is not derived.

39 manifest files are not in the inventory (C sources and STELLA `.trf` files) and are excluded from every figure; none of them is in the core engine modules. Contained procedures are separate routines, counted once each.

## Core engine (`star`, `star_data`, `binary`, `astero`)

| Measure | Value |
| :--- | ---: |
| Upstream lines | 144,952 |
| Lines in routines mapped in full | 22,214 |
| Lines in routines mapped in part | 0 |
| Lines in not-applicable routines | 0 |
| Lines in unmapped routines | 101,120 |
| Lines outside routines | 21,618 |
| Lines in routines with oracle evidence | 4,911 |
| Line-weighted full-mapping fraction (all lines) | 15.3% |
| Line-weighted partial-mapping fraction (all lines) | 0.0% |
| Full-mapping fraction excluding not-applicable lines | 15.3% |
| Partial-mapping fraction excluding not-applicable lines | 0.0% |
| Not-applicable fraction of all lines | 0.0% |
| Line-weighted oracle fraction | 3.4% |
| Whole-file ported fraction (existing headline) | 30.7% |
| Routines (full / partial / N/A / unmapped / total) | 261 / 0 / 0 / 2,327 / 2,588 |
| Routines with oracle evidence | 25 |

The fractions answer different questions. The whole-file figure counts every line of a file whose hand-set status is ported; the line-weighted figures count only the own lines of routines that carry the named evidence. Neither is a statement about agreement with MESA.

## Per module

| Module | Files | Routines | Full | Partial | N/A | Unmapped | Oracle | Upstream lines | Lines full | Lines partial | Lines N/A | Lines unmapped | Outside routines | Full | Partial | Full excl. N/A | Oracle | Whole-file ported |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `star` | 173 | 2,176 | 192 | 0 | 0 | 1,984 | 25 | 120,509 | 19,506 | 0 | 0 | 88,345 | 12,658 | 16.2% | 0.0% | 16.2% | 4.1% | 35.1% |
| `star_data` | 12 | 5 | 0 | 0 | 0 | 5 | 0 | 4,870 | 0 | 0 | 0 | 49 | 4,821 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `binary` | 47 | 218 | 69 | 0 | 0 | 149 | 0 | 11,638 | 2,708 | 0 | 0 | 6,173 | 2,757 | 23.3% | 0.0% | 23.3% | 0.0% | 18.7% |
| `astero` | 14 | 189 | 0 | 0 | 0 | 189 | 0 | 7,935 | 0 | 0 | 0 | 6,553 | 1,382 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `eos` | 57 | 326 | 108 | 0 | 0 | 218 | 30 | 28,666 | 6,791 | 0 | 0 | 14,724 | 7,151 | 23.7% | 0.0% | 23.7% | 6.1% | 0.0% |
| `kap` | 38 | 311 | 0 | 0 | 0 | 311 | 0 | 22,430 | 0 | 0 | 0 | 19,985 | 2,445 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `net` | 16 | 195 | 0 | 0 | 0 | 195 | 0 | 13,452 | 0 | 0 | 0 | 12,219 | 1,233 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `rates` | 26 | 477 | 0 | 0 | 0 | 477 | 0 | 17,217 | 0 | 0 | 0 | 14,276 | 2,941 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `neu` | 4 | 15 | 11 | 0 | 0 | 4 | 4 | 2,184 | 1,586 | 0 | 0 | 428 | 170 | 72.6% | 0.0% | 72.6% | 18.5% | 83.9% |
| `atm` | 19 | 93 | 0 | 0 | 0 | 93 | 0 | 6,853 | 0 | 0 | 0 | 5,350 | 1,503 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `turb` | 6 | 25 | 16 | 0 | 0 | 9 | 6 | 2,033 | 968 | 0 | 0 | 459 | 606 | 47.6% | 0.0% | 47.6% | 17.5% | 66.5% |
| `chem` | 7 | 67 | 0 | 0 | 0 | 67 | 0 | 4,195 | 0 | 0 | 0 | 3,330 | 865 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `ionization` | 22 | 67 | 0 | 0 | 0 | 67 | 0 | 3,859 | 0 | 0 | 0 | 3,055 | 804 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `colors` | 10 | 89 | 0 | 0 | 0 | 89 | 0 | 3,823 | 0 | 0 | 0 | 3,155 | 668 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `num` | 66 | 273 | 0 | 0 | 0 | 273 | 0 | 20,045 | 0 | 0 | 0 | 17,448 | 2,597 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `mtx` | 23 | 163 | 0 | 0 | 0 | 163 | 0 | 5,802 | 0 | 0 | 0 | 5,147 | 655 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `interp_1d` | 11 | 111 | 0 | 0 | 0 | 111 | 0 | 4,969 | 0 | 0 | 0 | 4,351 | 618 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `interp_2d` | 10 | 82 | 2 | 0 | 0 | 80 | 2 | 13,565 | 90 | 0 | 0 | 12,965 | 510 | 0.7% | 0.0% | 0.7% | 0.7% | 0.9% |
| `auto_diff` | 10 | 927 | 0 | 0 | 0 | 927 | 0 | 14,353 | 0 | 0 | 0 | 10,306 | 4,047 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `math` | 6 | 35 | 0 | 0 | 0 | 35 | 0 | 936 | 0 | 0 | 0 | 485 | 451 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `const` | 2 | 2 | 0 | 0 | 0 | 2 | 0 | 256 | 0 | 0 | 0 | 48 | 208 | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| `utils` | 11 | 150 | 0 | 0 | 0 | 150 | 0 | 3,398 | 0 | 0 | 0 | 2,614 | 784 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `gyre` | 171 | 983 | 0 | 0 | 0 | 983 | 0 | 55,708 | 0 | 0 | 0 | 40,741 | 14,967 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `adipls` | 133 | 221 | 0 | 0 | 0 | 221 | 0 | 30,923 | 0 | 0 | 0 | 30,764 | 159 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `stella` | 109 | 238 | 0 | 0 | 0 | 238 | 0 | 45,351 | 0 | 0 | 0 | 39,716 | 5,635 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `plotting` | 58 | 559 | 0 | 0 | 0 | 559 | 0 | 35,732 | 0 | 0 | 0 | 25,133 | 10,599 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## Derived status ceiling

The ceiling is the highest status that the routine evidence of a file supports. The rule counts full mappings only, over the file's applicable routines (those not declared not applicable):

* none mapped in full or in part, or no applicable routines: no ceiling claim;
* any applicable routine partial or unmapped: `partial`, and the file is not eligible for a raise;
* all applicable routines mapped in full: `ported_unverified`;
* all mapped in full and each has oracle evidence: `kernel_reference_vectors`.

`case_level_evidence` is never a ceiling because the `exercised` level is not derived. Contained procedures count as routines. A ceiling over the hand-set status is a candidate for review, not a raise: the table reads no status as evidence and changes none, and nothing gates on this section.

### Hand-set status above the ceiling (38 files)

Files whose status needs more than the routine evidence gives. Files with no ceiling claim are not listed; 5 files with a ported status have no ceiling claim. Largest ten:

| File | Status | Ceiling | Lines | Routines | Full | Partial | N/A | Oracle |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `star/private/rsp_step.f90` | ported_unverified | partial | 3,834 | 53 | 27 | 0 | 0 | 0 |
| `star/private/mesh_adjust.f90` | ported_unverified | partial | 2,889 | 39 | 4 | 0 | 0 | 0 |
| `star/private/timestep.f90` | ported_unverified | partial | 2,432 | 64 | 3 | 0 | 0 | 0 |
| `star/private/diffusion.f90` | kernel_reference_vectors | partial | 2,050 | 30 | 2 | 0 | 0 | 1 |
| `star/private/star_solver.f90` | ported_unverified | partial | 1,940 | 35 | 1 | 0 | 0 | 0 |
| `star/private/diffusion_support.f90` | kernel_reference_vectors | partial | 1,800 | 16 | 1 | 0 | 0 | 1 |
| `neu/private/mod_neu.f90` | ported_unverified | partial | 1,715 | 12 | 10 | 0 | 0 | 3 |
| `star/private/conv_premix.f90` | ported_unverified | partial | 1,561 | 16 | 1 | 0 | 0 | 0 |
| `star/private/solver_support.f90` | ported_unverified | partial | 1,547 | 21 | 7 | 0 | 0 | 0 |
| `star/private/adjust_mesh_split_merge.f90` | ported_unverified | partial | 1,461 | 22 | 1 | 0 | 0 | 0 |

### `partial` files with every applicable routine mapped in full (8 files)

Candidates for a reviewed raise to `ported_unverified`. Largest ten:

| File | Status | Ceiling | Lines | Routines | Full | Partial | N/A | Oracle |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `star/private/mass_utils.f90` | partial | ported_unverified | 594 | 9 | 9 | 0 | 0 | 0 |
| `eos/private/skye_coulomb_solid.f90` | partial | ported_unverified | 348 | 8 | 8 | 0 | 0 | 0 |
| `binary/private/binary_timestep.f90` | partial | ported_unverified | 325 | 2 | 2 | 0 | 0 | 0 |
| `eos/private/skye_thermodynamics.f90` | partial | ported_unverified | 241 | 3 | 3 | 0 | 0 | 0 |
| `eos/private/helm_polynomials.f90` | partial | ported_unverified | 185 | 22 | 22 | 0 | 0 | 0 |
| `eos/private/skye_coulomb_liquid.f90` | partial | ported_unverified | 181 | 4 | 4 | 0 | 0 | 0 |
| `eos/private/eosde_eval.f90` | partial | kernel_reference_vectors | 113 | 1 | 1 | 0 | 0 | 1 |
| `eos/private/ion_offset.f90` | partial | kernel_reference_vectors | 84 | 1 | 1 | 0 | 0 | 1 |

## Per file (files with a non-absent status or a mapped routine)

| File | Module | Status | Ceiling | Lines | Routines | Full | Partial | N/A | Oracle | Lines full | Lines partial | Lines N/A | Lines unmapped | Outside |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `astero/private/astero_run_support.f90` | `astero` | partial | - | 1,127 | 22 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,058 | 69 |
| `astero/private/astero_support.f90` | `astero` | partial | - | 1,530 | 43 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,415 | 115 |
| `astero/private/extras_support.f90` | `astero` | partial | - | 1,238 | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,162 | 76 |
| `astero/public/astero_def.f90` | `astero` | partial | - | 1,709 | 31 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,005 | 704 |
| `astero/public/astero_lib.f90` | `astero` | partial | - | 146 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 88 | 58 |
| `atm/private/atm_irradiated.f90` | `atm` | partial | - | 334 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 294 | 40 |
| `atm/private/atm_t_tau_relations.f90` | `atm` | partial | - | 367 | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 311 | 56 |
| `atm/private/atm_t_tau_uniform.f90` | `atm` | partial | - | 702 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 655 | 47 |
| `atm/private/atm_t_tau_varying.f90` | `atm` | partial | - | 824 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 776 | 48 |
| `atm/private/atm_table.f90` | `atm` | partial | - | 447 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 407 | 40 |
| `atm/private/atm_utils.f90` | `atm` | partial | - | 207 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 154 | 53 |
| `atm/private/e2_pairs.dek` | `atm` | partial | - | 581 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 581 |
| `atm/private/table_atm.f90` | `atm` | partial | - | 684 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 644 | 40 |
| `atm/public/atm_def.f90` | `atm` | partial | - | 146 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 146 |
| `atm/public/atm_lib.f90` | `atm` | partial | - | 93 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 16 | 77 |
| `auto_diff/private/auto_diff_real_15var_order1_module.f90` | `auto_diff` | partial | - | 1,132 | 101 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 714 | 418 |
| `auto_diff/private/auto_diff_real_1var_order1_module.f90` | `auto_diff` | partial | - | 1,319 | 116 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 820 | 499 |
| `auto_diff/private/auto_diff_real_2var_order1_module.f90` | `auto_diff` | partial | - | 1,545 | 117 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,039 | 506 |
| `auto_diff/private/auto_diff_real_2var_order2_module.f90` | `auto_diff` | partial | - | 2,009 | 117 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,500 | 509 |
| `auto_diff/private/auto_diff_real_2var_order3_module.f90` | `auto_diff` | partial | - | 3,410 | 117 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2,897 | 513 |
| `auto_diff/private/auto_diff_real_4var_order1_module.f90` | `auto_diff` | partial | - | 1,751 | 119 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,231 | 520 |
| `auto_diff/private/auto_diff_real_star_order1_module.f90` | `auto_diff` | partial | - | 1,307 | 115 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 814 | 493 |
| `auto_diff/private/support_functions.f90` | `auto_diff` | partial | - | 135 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 77 | 58 |
| `auto_diff/public/auto_diff.f90` | `auto_diff` | partial | - | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 30 |
| `binary/job/run_binary.f90` | `binary` | ported_unverified | ported_unverified | 44 | 1 | 1 | 0 | 0 | 0 | 17 | 0 | 0 | 0 | 27 |
| `binary/private/binary_ctrls_io.f90` | `binary` | partial | - | 821 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 568 | 253 |
| `binary/private/binary_disk.f90` | `binary` | ported_unverified | ported_unverified | 318 | 7 | 7 | 0 | 0 | 0 | 265 | 0 | 0 | 0 | 53 |
| `binary/private/binary_edot.f90` | `binary` | ported_unverified | ported_unverified | 231 | 4 | 4 | 0 | 0 | 0 | 186 | 0 | 0 | 0 | 45 |
| `binary/private/binary_evolve.f90` | `binary` | partial | - | 698 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 654 | 44 |
| `binary/private/binary_history.f90` | `binary` | partial | - | 880 | 21 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 835 | 45 |
| `binary/private/binary_history_specs.f90` | `binary` | partial | - | 286 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 244 | 42 |
| `binary/private/binary_irradiation.f90` | `binary` | ported_unverified | ported_unverified | 62 | 1 | 1 | 0 | 0 | 0 | 28 | 0 | 0 | 0 | 34 |
| `binary/private/binary_jdot.f90` | `binary` | ported_unverified | ported_unverified | 330 | 7 | 7 | 0 | 0 | 0 | 290 | 0 | 0 | 0 | 40 |
| `binary/private/binary_job_controls.inc` | `binary` | partial | - | 43 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 43 |
| `binary/private/binary_job_ctrls_io.f90` | `binary` | partial | - | 331 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 244 | 87 |
| `binary/private/binary_mdot.f90` | `binary` | ported_unverified | partial | 1,023 | 14 | 12 | 0 | 0 | 0 | 686 | 0 | 0 | 287 | 50 |
| `binary/private/binary_roche_deformation.f90` | `binary` | partial | partial | 416 | 11 | 7 | 0 | 0 | 0 | 210 | 0 | 0 | 141 | 65 |
| `binary/private/binary_tides.f90` | `binary` | partial | partial | 461 | 8 | 7 | 0 | 0 | 0 | 357 | 0 | 0 | 57 | 47 |
| `binary/private/binary_timestep.f90` | `binary` | partial | ported_unverified | 325 | 2 | 2 | 0 | 0 | 0 | 289 | 0 | 0 | 0 | 36 |
| `binary/private/binary_utils.f90` | `binary` | partial | partial | 215 | 9 | 7 | 0 | 0 | 0 | 130 | 0 | 0 | 42 | 43 |
| `binary/private/binary_wind.f90` | `binary` | ported_unverified | ported_unverified | 173 | 3 | 3 | 0 | 0 | 0 | 137 | 0 | 0 | 0 | 36 |
| `binary/private/run_binary_support.f90` | `binary` | partial | - | 1,016 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 975 | 41 |
| `binary/public/binary_controls.inc` | `binary` | partial | - | 190 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 190 |
| `binary/public/binary_data.inc` | `binary` | partial | - | 326 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 326 |
| `binary/public/binary_def.f90` | `binary` | partial | - | 292 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 25 | 267 |
| `binary/public/binary_lib.f90` | `binary` | partial | partial | 272 | 17 | 11 | 0 | 0 | 0 | 113 | 0 | 0 | 111 | 48 |
| `chem/private/chem_isos_io.f90` | `chem` | partial | - | 233 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 33 |
| `chem/private/lodders_mod.f90` | `chem` | partial | - | 106 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 76 | 30 |
| `chem/private/nuclide_set_mod.f90` | `chem` | partial | - | 112 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 80 | 32 |
| `chem/public/chem_def.f90` | `chem` | partial | - | 2,654 | 22 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2,087 | 567 |
| `chem/public/chem_lib.f90` | `chem` | partial | - | 717 | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 569 | 148 |
| `colors/private/bolometric.f90` | `colors` | partial | - | 191 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 149 | 42 |
| `colors/private/colors_utils.f90` | `colors` | partial | - | 1,111 | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,019 | 92 |
| `colors/private/hermite_interp.f90` | `colors` | partial | - | 443 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 384 | 59 |
| `colors/private/knn_interp.f90` | `colors` | partial | - | 438 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 385 | 53 |
| `colors/private/linear_interp.f90` | `colors` | partial | - | 342 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 290 | 52 |
| `colors/private/synthetic.f90` | `colors` | partial | - | 230 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 183 | 47 |
| `colors/public/colors_def.f90` | `colors` | partial | - | 319 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 176 | 143 |
| `colors/public/colors_lib.f90` | `colors` | partial | - | 321 | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 246 | 75 |
| `const/public/const_def.f90` | `const` | ported_unverified | - | 218 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 42 | 176 |
| `const/public/const_lib.f90` | `const` | ported_unverified | - | 38 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 32 |
| `eos/private/eos_blend.f90` | `eos` | partial | partial | 208 | 5 | 2 | 0 | 0 | 2 | 48 | 0 | 0 | 92 | 68 |
| `eos/private/eos_ctrls_io.f90` | `eos` | partial | - | 697 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 414 | 283 |
| `eos/private/eos_helm_eval.f90` | `eos` | partial | partial | 361 | 4 | 2 | 0 | 0 | 1 | 214 | 0 | 0 | 105 | 42 |
| `eos/private/eos_initialize.f90` | `eos` | partial | - | 75 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 48 | 27 |
| `eos/private/eoscms_eval.f90` | `eos` | partial | partial | 703 | 8 | 4 | 0 | 0 | 1 | 432 | 0 | 0 | 197 | 74 |
| `eos/private/eosde_eval.f90` | `eos` | partial | kernel_reference_vectors | 113 | 1 | 1 | 0 | 0 | 1 | 82 | 0 | 0 | 0 | 31 |
| `eos/private/eosdt_eval.f90` | `eos` | partial | partial | 2,537 | 30 | 10 | 0 | 0 | 4 | 985 | 0 | 0 | 1,430 | 122 |
| `eos/private/eosdt_load_tables.f90` | `eos` | partial | - | 643 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 579 | 64 |
| `eos/private/eosdt_support.f90` | `eos` | partial | partial | 241 | 2 | 1 | 0 | 0 | 0 | 141 | 0 | 0 | 65 | 35 |
| `eos/private/eospc_eval.f90` | `eos` | partial | partial | 365 | 4 | 3 | 0 | 0 | 2 | 164 | 0 | 0 | 165 | 36 |
| `eos/private/eospt_eval.f90` | `eos` | partial | partial | 606 | 8 | 4 | 0 | 0 | 3 | 365 | 0 | 0 | 197 | 44 |
| `eos/private/gauss_fermi.f90` | `eos` | partial | partial | 1,199 | 11 | 1 | 0 | 0 | 1 | 105 | 0 | 0 | 1,036 | 58 |
| `eos/private/helm.f90` | `eos` | partial | partial | 350 | 3 | 1 | 0 | 0 | 1 | 33 | 0 | 0 | 276 | 41 |
| `eos/private/helm_alloc.f90` | `eos` | partial | - | 468 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 432 | 36 |
| `eos/private/helm_coulomb2.dek` | `eos` | partial | - | 420 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 420 |
| `eos/private/helm_declare_local_variables.dek` | `eos` | partial | - | 341 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 341 |
| `eos/private/helm_electron_positron.dek` | `eos` | partial | - | 641 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 641 |
| `eos/private/helm_gammas.dek` | `eos` | partial | - | 271 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 271 |
| `eos/private/helm_ideal_ions.dek` | `eos` | partial | - | 220 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 220 |
| `eos/private/helm_initialize_local_variables.dek` | `eos` | partial | - | 521 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 521 |
| `eos/private/helm_polynomials.f90` | `eos` | partial | ported_unverified | 185 | 22 | 22 | 0 | 0 | 0 | 121 | 0 | 0 | 0 | 64 |
| `eos/private/helm_print_results.dek` | `eos` | partial | - | 247 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 247 |
| `eos/private/helm_radiation.dek` | `eos` | partial | - | 67 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 67 |
| `eos/private/helm_store_results.dek` | `eos` | partial | - | 437 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 437 |
| `eos/private/helm_sum_totals.dek` | `eos` | partial | - | 169 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 169 |
| `eos/private/ideal.f90` | `eos` | partial | partial | 206 | 4 | 2 | 0 | 0 | 2 | 55 | 0 | 0 | 113 | 38 |
| `eos/private/ion_offset.f90` | `eos` | partial | kernel_reference_vectors | 84 | 1 | 1 | 0 | 0 | 1 | 43 | 0 | 0 | 0 | 41 |
| `eos/private/pc_eos.f90` | `eos` | partial | partial | 3,028 | 26 | 22 | 0 | 0 | 0 | 2,679 | 0 | 0 | 111 | 238 |
| `eos/private/pc_support.f90` | `eos` | partial | partial | 1,043 | 13 | 1 | 0 | 0 | 0 | 74 | 0 | 0 | 894 | 75 |
| `eos/private/skye.f90` | `eos` | partial | partial | 402 | 5 | 3 | 0 | 0 | 3 | 167 | 0 | 0 | 170 | 65 |
| `eos/private/skye_coulomb.f90` | `eos` | partial | partial | 486 | 7 | 1 | 0 | 0 | 0 | 72 | 0 | 0 | 282 | 132 |
| `eos/private/skye_coulomb_liquid.f90` | `eos` | partial | ported_unverified | 181 | 4 | 4 | 0 | 0 | 0 | 107 | 0 | 0 | 0 | 74 |
| `eos/private/skye_coulomb_solid.f90` | `eos` | partial | ported_unverified | 348 | 8 | 8 | 0 | 0 | 0 | 236 | 0 | 0 | 0 | 112 |
| `eos/private/skye_ideal.f90` | `eos` | partial | partial | 708 | 4 | 3 | 0 | 0 | 0 | 42 | 0 | 0 | 622 | 44 |
| `eos/private/skye_thermodynamics.f90` | `eos` | partial | ported_unverified | 241 | 3 | 3 | 0 | 0 | 0 | 158 | 0 | 0 | 0 | 83 |
| `eos/public/eos_def.f90` | `eos` | partial | - | 642 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 210 | 432 |
| `eos/public/eos_lib.f90` | `eos` | partial | partial | 910 | 29 | 9 | 0 | 0 | 8 | 468 | 0 | 0 | 335 | 107 |
| `eos/public/helm_def.dek` | `eos` | partial | - | 646 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 646 |
| `interp_1d/private/interp_1d_misc.f90` | `interp_1d` | partial | - | 712 | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 653 | 59 |
| `interp_1d/private/interp_1d_mp.f90` | `interp_1d` | partial | - | 388 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 353 | 35 |
| `interp_1d/private/interp_1d_mp_autodiff.f90` | `interp_1d` | partial | - | 391 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 355 | 36 |
| `interp_1d/private/interp_1d_pm.f90` | `interp_1d` | partial | - | 458 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 410 | 48 |
| `interp_1d/private/interp_1d_pm_autodiff.f90` | `interp_1d` | partial | - | 468 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 419 | 49 |
| `interp_1d/public/interp_1d_def.f90` | `interp_1d` | partial | - | 34 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 34 |
| `interp_1d/public/interp_1d_lib.f90` | `interp_1d` | partial | - | 894 | 41 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 757 | 137 |
| `interp_2d/private/akima760_db.f` | `interp_2d` | partial | - | 1,217 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,209 | 8 |
| `interp_2d/private/bicub_db.f` | `interp_2d` | partial | - | 3,487 | 17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3,363 | 124 |
| `interp_2d/private/bipm_db.f90` | `interp_2d` | ported_unverified | kernel_reference_vectors | 120 | 2 | 2 | 0 | 0 | 2 | 90 | 0 | 0 | 0 | 30 |
| `interp_2d/private/renka790_db.f` | `interp_2d` | partial | - | 1,854 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,852 | 2 |
| `interp_2d/public/interp_2d_lib_db.f90` | `interp_2d` | partial | - | 226 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 116 | 110 |
| `ionization/private/ion_tables_eval.f90` | `ionization` | partial | - | 576 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 530 | 46 |
| `ionization/private/ion_tables_load.f90` | `ionization` | partial | - | 556 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 510 | 46 |
| `ionization/private/mod_ionization.f90` | `ionization` | partial | - | 237 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 191 | 46 |
| `ionization/public/ionization_def.f90` | `ionization` | partial | - | 172 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 51 | 121 |
| `ionization/public/ionization_lib.f90` | `ionization` | partial | - | 92 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 50 | 42 |
| `kap/private/condint.f90` | `kap` | partial | - | 449 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 400 | 49 |
| `kap/private/cubic_interpolator.f90` | `kap` | partial | - | 185 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 130 | 55 |
| `kap/private/kap_aesopus.f90` | `kap` | partial | - | 608 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 552 | 56 |
| `kap/private/kap_ctrls_io.f90` | `kap` | partial | - | 527 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 374 | 153 |
| `kap/private/kap_eval.f90` | `kap` | partial | - | 874 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 826 | 48 |
| `kap/private/kap_eval_co.f90` | `kap` | partial | - | 1,241 | 17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,189 | 52 |
| `kap/private/kap_eval_fixed.f90` | `kap` | partial | - | 762 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 712 | 50 |
| `kap/private/kap_eval_support.f90` | `kap` | partial | - | 311 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 270 | 41 |
| `kap/private/kapcn.f90` | `kap` | partial | - | 441 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 370 | 71 |
| `kap/private/load_co_kap.f90` | `kap` | partial | - | 913 | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 854 | 59 |
| `kap/private/load_kap.f90` | `kap` | partial | - | 765 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 710 | 55 |
| `kap/public/kap_def.f90` | `kap` | partial | - | 834 | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 320 | 514 |
| `kap/public/kap_lib.f90` | `kap` | partial | - | 868 | 26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 697 | 171 |
| `math/private/math_pown.f90` | `math` | partial | - | 143 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 64 | 79 |
| `math/public/math_def.f90` | `math` | partial | - | 49 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 49 |
| `math/public/math_lib_intrinsic.f90` | `math` | partial | - | 324 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 164 | 160 |
| `math/public/precompute_zs.inc` | `math` | partial | - | 52 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 52 | 0 |
| `mtx/private/bcyclic.f90` | `mtx` | partial | - | 603 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 539 | 64 |
| `mtx/private/mtx_support.f90` | `mtx` | partial | - | 1,099 | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 998 | 101 |
| `mtx/private/my_lapack95.f90` | `mtx` | partial | - | 1,414 | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,338 | 76 |
| `mtx/public/mtx_bcyclic_dble_decsol.dek` | `mtx` | partial | - | 45 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 41 | 4 |
| `mtx/public/mtx_dble_decsol.dek` | `mtx` | partial | - | 264 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 244 | 20 |
| `mtx/public/mtx_debug_decsol.dek` | `mtx` | partial | - | 68 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 60 | 8 |
| `mtx/public/mtx_decsol.dek` | `mtx` | partial | - | 21 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 21 | 0 |
| `mtx/public/mtx_decsolblk.dek` | `mtx` | partial | - | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `mtx/public/mtx_decsolblk_dble.dek` | `mtx` | partial | - | 12 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 11 | 1 |
| `mtx/public/mtx_decsolc.dek` | `mtx` | partial | - | 19 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 19 | 0 |
| `mtx/public/mtx_decsolcs.dek` | `mtx` | partial | - | 13 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 12 | 1 |
| `mtx/public/mtx_decsols.dek` | `mtx` | partial | - | 14 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14 | 0 |
| `mtx/public/mtx_def.f90` | `mtx` | partial | - | 39 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 39 |
| `mtx/public/mtx_formats.dek` | `mtx` | partial | - | 429 | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 378 | 51 |
| `mtx/public/mtx_jac.dek` | `mtx` | partial | - | 137 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 133 | 4 |
| `mtx/public/mtx_lapack95.dek` | `mtx` | partial | - | 16 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 1 |
| `mtx/public/mtx_lib.f90` | `mtx` | partial | - | 340 | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 207 | 133 |
| `mtx/public/mtx_null_decsol.dek` | `mtx` | partial | - | 63 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 53 | 10 |
| `mtx/public/mtx_rcond.dek` | `mtx` | partial | - | 62 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 57 | 5 |
| `mtx/public/mtx_solve_routines.inc` | `mtx` | partial | - | 694 | 21 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 650 | 44 |
| `mtx/public/mtx_tridiag_decsol.dek` | `mtx` | partial | - | 19 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 12 | 7 |
| `net/private/burner_derivs.inc` | `net` | partial | - | 7 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 |
| `net/private/burner_jakob.inc` | `net` | partial | - | 8 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 0 |
| `net/private/net_approx21.f90` | `net` | partial | - | 3,374 | 31 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3,108 | 266 |
| `net/private/net_burn.f90` | `net` | partial | - | 475 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 430 | 45 |
| `net/private/net_burn_const_density.f90` | `net` | partial | - | 502 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 457 | 45 |
| `net/private/net_burn_const_p.f90` | `net` | partial | - | 583 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 547 | 36 |
| `net/private/net_burn_support.f90` | `net` | partial | - | 641 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 596 | 45 |
| `net/private/net_derivs.f90` | `net` | partial | - | 1,833 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,789 | 44 |
| `net/private/net_derivs_support.f90` | `net` | partial | - | 454 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 385 | 69 |
| `net/private/net_eval.f90` | `net` | partial | - | 852 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 800 | 52 |
| `net/private/net_initialize.f90` | `net` | partial | - | 2,300 | 41 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2,206 | 94 |
| `net/private/net_screen.f90` | `net` | partial | - | 566 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 525 | 41 |
| `net/public/burner_const_density_get_eos_info.inc` | `net` | partial | - | 15 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 0 |
| `net/public/burner_finish_substep.inc` | `net` | partial | - | 7 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 |
| `net/public/net_def.f90` | `net` | partial | - | 625 | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 276 | 349 |
| `net/public/net_lib.f90` | `net` | partial | - | 1,210 | 45 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,063 | 147 |
| `neu/private/mod_neu.f90` | `neu` | ported_unverified | partial | 1,715 | 12 | 10 | 0 | 0 | 3 | 1,552 | 0 | 0 | 76 | 87 |
| `neu/public/neu_def.f90` | `neu` | ported_unverified | - | 46 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 46 |
| `neu/public/neu_lib.f90` | `neu` | ported_unverified | kernel_reference_vectors | 71 | 1 | 1 | 0 | 0 | 1 | 34 | 0 | 0 | 0 | 37 |
| `num/private/decomc.dek` | `num` | partial | - | 429 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 425 | 4 |
| `num/private/decomr.dek` | `num` | partial | - | 349 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 349 | 0 |
| `num/private/decsol_done.dek` | `num` | partial | - | 138 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 138 | 0 |
| `num/private/mod_bobyqa.f` | `num` | partial | - | 2,243 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2,224 | 19 |
| `num/private/mod_brent.f90` | `num` | partial | - | 707 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 688 | 19 |
| `num/private/mod_dc_decsol.f` | `num` | partial | - | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14 |
| `num/private/mod_dop853.f` | `num` | partial | - | 836 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 802 | 34 |
| `num/private/mod_dopri5.f` | `num` | partial | - | 645 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 606 | 39 |
| `num/private/mod_integrate.f90` | `num` | partial | - | 114 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 80 | 34 |
| `num/private/mod_newton.f90` | `num` | partial | - | 1,536 | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,500 | 36 |
| `num/private/mod_newuoa.f` | `num` | partial | - | 1,583 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,568 | 15 |
| `num/private/mod_qsort.f90` | `num` | partial | - | 160 | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 138 | 22 |
| `num/private/mod_random.f90` | `num` | partial | - | 376 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 341 | 35 |
| `num/private/mod_root.f90` | `num` | partial | - | 277 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 244 | 33 |
| `num/private/mod_rosenbrock.f` | `num` | partial | - | 1,862 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,746 | 116 |
| `num/private/mod_simplex.f90` | `num` | partial | - | 386 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 355 | 31 |
| `num/private/mod_solout_root.f90` | `num` | partial | - | 248 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 219 | 29 |
| `num/private/qsort_inline.inc` | `num` | partial | - | 194 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 194 |
| `num/private/rodas_args.dek` | `num` | partial | - | 55 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 55 |
| `num/private/slvrod.dek` | `num` | partial | - | 395 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 395 | 0 |
| `num/public/accurate_sum.f90` | `num` | partial | - | 285 | 21 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 183 | 102 |
| `num/public/accurate_sum_auto_diff_star_order1.f90` | `num` | partial | - | 325 | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 215 | 110 |
| `num/public/newton_procs.dek` | `num` | partial | - | 260 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 138 | 122 |
| `num/public/num_binary_search.dek` | `num` | partial | - | 259 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 234 | 25 |
| `num/public/num_bobyqa.dek` | `num` | partial | - | 106 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 13 | 93 |
| `num/public/num_bobyqa_proc.dek` | `num` | partial | - | 7 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 |
| `num/public/num_brent.dek` | `num` | partial | - | 225 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 43 | 182 |
| `num/public/num_def.f90` | `num` | partial | - | 220 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 220 |
| `num/public/num_dfridr.dek` | `num` | partial | - | 55 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 55 | 0 |
| `num/public/num_dfx.dek` | `num` | partial | - | 10 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 0 |
| `num/public/num_dop853.dek` | `num` | partial | - | 183 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 133 | 50 |
| `num/public/num_dopri5.dek` | `num` | partial | - | 183 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 133 | 50 |
| `num/public/num_fcn.dek` | `num` | partial | - | 12 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 12 | 0 |
| `num/public/num_fcn_blk_dble.dek` | `num` | partial | - | 11 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 11 | 0 |
| `num/public/num_interp_y.dek` | `num` | partial | - | 11 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 | 2 |
| `num/public/num_isolve.dek` | `num` | partial | - | 357 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 351 | 6 |
| `num/public/num_jac.dek` | `num` | partial | - | 19 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 19 | 0 |
| `num/public/num_jac_blk_dble.dek` | `num` | partial | - | 12 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 12 | 0 |
| `num/public/num_lib.f90` | `num` | partial | - | 626 | 21 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 256 | 370 |
| `num/public/num_mas.dek` | `num` | partial | - | 8 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 0 |
| `num/public/num_newton.dek` | `num` | partial | - | 322 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 283 | 39 |
| `num/public/num_newuoa.dek` | `num` | partial | - | 64 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 13 | 51 |
| `num/public/num_newuoa_proc.dek` | `num` | partial | - | 7 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 |
| `num/public/num_rk2.dek` | `num` | partial | - | 41 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 38 | 3 |
| `num/public/num_rk4.dek` | `num` | partial | - | 53 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 50 | 3 |
| `num/public/num_rodas3.dek` | `num` | partial | - | 347 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 293 | 54 |
| `num/public/num_rodas4.dek` | `num` | partial | - | 344 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 293 | 51 |
| `num/public/num_rodasp.dek` | `num` | partial | - | 347 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 293 | 54 |
| `num/public/num_root_fcn.dek` | `num` | partial | - | 11 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 11 | 0 |
| `num/public/num_ros2.dek` | `num` | partial | - | 341 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 292 | 49 |
| `num/public/num_ros3p.dek` | `num` | partial | - | 347 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 292 | 55 |
| `num/public/num_ros3pl.dek` | `num` | partial | - | 347 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 292 | 55 |
| `num/public/num_rose2.dek` | `num` | partial | - | 344 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 292 | 52 |
| `num/public/num_safe_root.dek` | `num` | partial | - | 144 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 129 | 15 |
| `num/public/num_safe_root_x.dek` | `num` | partial | - | 76 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 65 | 11 |
| `num/public/num_simplex.dek` | `num` | partial | - | 146 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 114 | 32 |
| `num/public/num_simplex_fcn.dek` | `num` | partial | - | 11 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 1 |
| `num/public/num_sjac.dek` | `num` | partial | - | 14 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14 | 0 |
| `num/public/num_solout.dek` | `num` | partial | - | 21 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 21 | 0 |
| `num/public/num_solout_root.dek` | `num` | partial | - | 85 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 76 | 9 |
| `num/public/num_solout_root_fcn.dek` | `num` | partial | - | 16 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 16 | 0 |
| `num/public/num_solvers_options.dek` | `num` | partial | - | 51 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 49 | 2 |
| `rates/private/coulomb.f90` | `rates` | partial | - | 71 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 37 | 34 |
| `rates/private/eval_coulomb.f90` | `rates` | partial | - | 350 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 293 | 57 |
| `rates/private/eval_ecapture.f90` | `rates` | partial | - | 334 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 301 | 33 |
| `rates/private/eval_psi.f90` | `rates` | partial | - | 163 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 127 | 36 |
| `rates/private/eval_weak.f90` | `rates` | partial | - | 346 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 311 | 35 |
| `rates/private/load_ecapture.f90` | `rates` | partial | - | 353 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 315 | 38 |
| `rates/private/load_weak.f90` | `rates` | partial | - | 710 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 668 | 42 |
| `rates/private/pycno.f90` | `rates` | partial | - | 308 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 276 | 32 |
| `rates/private/ratelib.f90` | `rates` | partial | - | 4,619 | 220 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3,775 | 844 |
| `rates/private/rates_initialize.f90` | `rates` | partial | - | 1,422 | 44 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,354 | 68 |
| `rates/private/rates_names.f90` | `rates` | partial | - | 368 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 335 | 33 |
| `rates/private/rates_reverses.f90` | `rates` | partial | - | 173 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 142 | 31 |
| `rates/private/rates_support.f90` | `rates` | partial | - | 706 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 652 | 54 |
| `rates/private/raw_rates.f90` | `rates` | partial | - | 1,140 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,082 | 58 |
| `rates/private/reaclib_eval.f90` | `rates` | partial | - | 292 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 245 | 47 |
| `rates/private/reaclib_input.f90` | `rates` | partial | - | 670 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 628 | 42 |
| `rates/private/reaclib_support.f90` | `rates` | partial | - | 766 | 19 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 724 | 42 |
| `rates/private/screen.f90` | `rates` | partial | - | 108 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 76 | 32 |
| `rates/private/screen5.f90` | `rates` | partial | - | 396 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 364 | 32 |
| `rates/private/screen_chugunov.f90` | `rates` | partial | - | 309 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 239 | 70 |
| `rates/private/suzuki_tables.f90` | `rates` | partial | - | 553 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 460 | 93 |
| `rates/private/weak_support.f90` | `rates` | partial | - | 85 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 57 | 28 |
| `rates/private/weaklib_tables.f90` | `rates` | partial | - | 514 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 456 | 58 |
| `rates/public/rates_def.f90` | `rates` | partial | - | 1,295 | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 377 | 918 |
| `rates/public/rates_lib.f90` | `rates` | partial | - | 974 | 55 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 828 | 146 |
| `star/job/other_extras.f90` | `star` | partial | - | 207 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 144 | 63 |
| `star/job/run_star.f90` | `star` | ported_unverified | ported_unverified | 48 | 1 | 1 | 0 | 0 | 0 | 22 | 0 | 0 | 0 | 26 |
| `star/job/run_star_support.f90` | `star` | partial | - | 3,705 | 49 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3,560 | 145 |
| `star/private/adjust_mass.f90` | `star` | partial | partial | 1,624 | 18 | 3 | 0 | 0 | 0 | 433 | 0 | 0 | 1,122 | 69 |
| `star/private/adjust_mesh.f90` | `star` | ported_unverified | partial | 700 | 14 | 1 | 0 | 0 | 0 | 509 | 0 | 0 | 154 | 37 |
| `star/private/adjust_mesh_split_merge.f90` | `star` | ported_unverified | partial | 1,461 | 22 | 1 | 0 | 0 | 0 | 44 | 0 | 0 | 1,343 | 74 |
| `star/private/adjust_mesh_support.f90` | `star` | ported_unverified | partial | 457 | 12 | 2 | 0 | 0 | 0 | 104 | 0 | 0 | 304 | 49 |
| `star/private/adjust_xyz.f90` | `star` | partial | - | 1,468 | 31 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,373 | 95 |
| `star/private/atm_support.f90` | `star` | partial | - | 1,289 | 19 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,227 | 62 |
| `star/private/auto_diff_support.f90` | `star` | partial | - | 1,460 | 118 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,286 | 174 |
| `star/private/brunt.f90` | `star` | ported_unverified | partial | 335 | 5 | 2 | 0 | 0 | 0 | 133 | 0 | 0 | 157 | 45 |
| `star/private/conv_premix.f90` | `star` | ported_unverified | partial | 1,561 | 16 | 1 | 0 | 0 | 0 | 101 | 0 | 0 | 1,347 | 113 |
| `star/private/ctrls_io.f90` | `star` | partial | - | 4,272 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3,698 | 574 |
| `star/private/diffusion.f90` | `star` | kernel_reference_vectors | partial | 2,050 | 30 | 2 | 0 | 0 | 1 | 462 | 0 | 0 | 1,518 | 70 |
| `star/private/diffusion_procs.f90` | `star` | partial | - | 1,746 | 17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,677 | 69 |
| `star/private/diffusion_support.f90` | `star` | kernel_reference_vectors | partial | 1,800 | 16 | 1 | 0 | 0 | 1 | 201 | 0 | 0 | 1,487 | 112 |
| `star/private/do_one_utils.f90` | `star` | partial | partial | 1,544 | 16 | 1 | 0 | 0 | 0 | 867 | 0 | 0 | 610 | 67 |
| `star/private/element_diffusion.f90` | `star` | ported_unverified | partial | 582 | 6 | 2 | 0 | 0 | 1 | 424 | 0 | 0 | 122 | 36 |
| `star/private/eos_support.f90` | `star` | partial | - | 413 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 336 | 77 |
| `star/private/eps_grav.f90` | `star` | ported_unverified | partial | 408 | 6 | 2 | 0 | 0 | 0 | 40 | 0 | 0 | 321 | 47 |
| `star/private/eps_mdot.f90` | `star` | ported_unverified | ported_unverified | 706 | 6 | 6 | 0 | 0 | 1 | 634 | 0 | 0 | 0 | 72 |
| `star/private/evolve.f90` | `star` | partial | - | 2,403 | 26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2,331 | 72 |
| `star/private/evolve_support.f90` | `star` | partial | - | 305 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 267 | 38 |
| `star/private/history.f90` | `star` | partial | - | 3,449 | 43 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3,389 | 60 |
| `star/private/history_specs.f90` | `star` | partial | - | 783 | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 706 | 77 |
| `star/private/hydro_alpha_rti_eqns.f90` | `star` | ported_unverified | ported_unverified | 195 | 1 | 1 | 0 | 0 | 0 | 161 | 0 | 0 | 0 | 34 |
| `star/private/hydro_chem_eqns.f90` | `star` | ported_unverified | partial | 224 | 3 | 2 | 0 | 0 | 1 | 175 | 0 | 0 | 13 | 36 |
| `star/private/hydro_energy.f90` | `star` | ported_unverified | partial | 879 | 17 | 1 | 0 | 0 | 0 | 19 | 0 | 0 | 812 | 48 |
| `star/private/hydro_eqns.f90` | `star` | partial | partial | 1,164 | 18 | 12 | 0 | 0 | 0 | 893 | 0 | 0 | 220 | 51 |
| `star/private/hydro_momentum.f90` | `star` | ported_unverified | partial | 603 | 15 | 4 | 0 | 0 | 0 | 127 | 0 | 0 | 422 | 54 |
| `star/private/hydro_riemann.f90` | `star` | partial | - | 459 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 396 | 63 |
| `star/private/hydro_rotation.f90` | `star` | partial | - | 922 | 26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 831 | 91 |
| `star/private/hydro_rsp2.f90` | `star` | partial | - | 1,243 | 34 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,135 | 108 |
| `star/private/hydro_rsp2_support.f90` | `star` | partial | - | 445 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 407 | 38 |
| `star/private/hydro_temperature.f90` | `star` | partial | - | 464 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 411 | 53 |
| `star/private/hydro_vars.f90` | `star` | partial | - | 1,163 | 19 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,089 | 74 |
| `star/private/init.f90` | `star` | stub | - | 1,232 | 27 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,130 | 102 |
| `star/private/init_model.f90` | `star` | ported_unverified | partial | 468 | 7 | 1 | 0 | 0 | 1 | 68 | 0 | 0 | 359 | 41 |
| `star/private/ionization_potentials.f90` | `star` | ported_unverified | - | 596 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 558 | 38 |
| `star/private/kap_support.f90` | `star` | partial | - | 526 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 480 | 46 |
| `star/private/magnetic_diffusion.f90` | `star` | partial | - | 137 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 62 | 75 |
| `star/private/mass_utils.f90` | `star` | partial | ported_unverified | 594 | 9 | 9 | 0 | 0 | 0 | 441 | 0 | 0 | 0 | 153 |
| `star/private/mesh_adjust.f90` | `star` | ported_unverified | partial | 2,889 | 39 | 4 | 0 | 0 | 0 | 486 | 0 | 0 | 2,293 | 110 |
| `star/private/mesh_functions.f90` | `star` | ported_unverified | partial | 327 | 6 | 2 | 0 | 0 | 0 | 213 | 0 | 0 | 70 | 44 |
| `star/private/mesh_plan.f90` | `star` | ported_unverified | partial | 1,301 | 14 | 1 | 0 | 0 | 0 | 163 | 0 | 0 | 1,092 | 46 |
| `star/private/micro.f90` | `star` | partial | - | 766 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 710 | 56 |
| `star/private/mix_info.f90` | `star` | partial | partial | 2,438 | 37 | 7 | 0 | 0 | 1 | 987 | 0 | 0 | 1,351 | 100 |
| `star/private/mod_typical_charge.f90` | `star` | ported_unverified | partial | 111 | 4 | 1 | 0 | 0 | 0 | 11 | 0 | 0 | 40 | 60 |
| `star/private/net.f90` | `star` | partial | - | 813 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 754 | 59 |
| `star/private/neu.f90` | `star` | ported_unverified | ported_unverified | 129 | 2 | 2 | 0 | 0 | 1 | 93 | 0 | 0 | 0 | 36 |
| `star/private/overshoot.f90` | `star` | ported_unverified | ported_unverified | 329 | 1 | 1 | 0 | 0 | 0 | 292 | 0 | 0 | 0 | 37 |
| `star/private/overshoot_exp.f90` | `star` | ported_unverified | ported_unverified | 176 | 1 | 1 | 0 | 0 | 0 | 143 | 0 | 0 | 0 | 33 |
| `star/private/overshoot_step.f90` | `star` | ported_unverified | ported_unverified | 162 | 1 | 1 | 0 | 0 | 0 | 129 | 0 | 0 | 0 | 33 |
| `star/private/overshoot_utils.f90` | `star` | ported_unverified | ported_unverified | 376 | 4 | 4 | 0 | 0 | 2 | 334 | 0 | 0 | 0 | 42 |
| `star/private/paquette_coeffs.f90` | `star` | kernel_reference_vectors | kernel_reference_vectors | 721 | 3 | 3 | 0 | 0 | 3 | 656 | 0 | 0 | 0 | 65 |
| `star/private/phase_separation.f90` | `star` | kernel_reference_vectors | partial | 400 | 7 | 1 | 0 | 0 | 1 | 14 | 0 | 0 | 340 | 46 |
| `star/private/photo_in.f90` | `star` | partial | - | 320 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 288 | 32 |
| `star/private/photo_out.f90` | `star` | partial | - | 179 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 147 | 32 |
| `star/private/pre_ms_model.f90` | `star` | ported_unverified | partial | 738 | 6 | 1 | 0 | 0 | 0 | 252 | 0 | 0 | 445 | 41 |
| `star/private/predictive_mix.f90` | `star` | ported_unverified | partial | 965 | 6 | 1 | 0 | 0 | 0 | 159 | 0 | 0 | 760 | 46 |
| `star/private/profile.f90` | `star` | partial | - | 1,186 | 23 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,120 | 66 |
| `star/private/profile_getval.f90` | `star` | partial | - | 2,466 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2,394 | 72 |
| `star/private/pulse.f90` | `star` | partial | - | 185 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 138 | 47 |
| `star/private/pulse_cafein.f90` | `star` | partial | - | 722 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 680 | 42 |
| `star/private/pulse_fgong.f90` | `star` | partial | - | 710 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 660 | 50 |
| `star/private/pulse_gr1d.f90` | `star` | partial | - | 182 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 144 | 38 |
| `star/private/pulse_gyre.f90` | `star` | partial | - | 476 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 435 | 41 |
| `star/private/pulse_osc.f90` | `star` | partial | - | 608 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 559 | 49 |
| `star/private/pulse_saio.f90` | `star` | partial | - | 347 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 307 | 40 |
| `star/private/pulse_utils.f90` | `star` | ported_unverified | ported_unverified | 525 | 9 | 9 | 0 | 0 | 0 | 466 | 0 | 0 | 0 | 59 |
| `star/private/read_model.f90` | `star` | partial | - | 986 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 896 | 90 |
| `star/private/relax.f90` | `star` | stub | - | 4,340 | 128 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4,096 | 244 |
| `star/private/report.f90` | `star` | partial | - | 1,419 | 23 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,359 | 60 |
| `star/private/rotation_mix_info.f90` | `star` | partial | - | 1,371 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,315 | 56 |
| `star/private/rsp.f90` | `star` | ported_unverified | partial | 1,030 | 19 | 6 | 0 | 0 | 1 | 339 | 0 | 0 | 621 | 70 |
| `star/private/rsp_build.f90` | `star` | ported_unverified | partial | 986 | 18 | 1 | 0 | 0 | 1 | 253 | 0 | 0 | 680 | 53 |
| `star/private/rsp_def.f90` | `star` | ported_unverified | partial | 863 | 22 | 17 | 0 | 0 | 3 | 497 | 0 | 0 | 182 | 184 |
| `star/private/rsp_eval_eos_and_kap.f90` | `star` | ported_unverified | partial | 1,047 | 16 | 14 | 0 | 0 | 0 | 860 | 0 | 0 | 112 | 75 |
| `star/private/rsp_lina.f90` | `star` | ported_unverified | ported_unverified | 1,269 | 3 | 3 | 0 | 0 | 1 | 1,225 | 0 | 0 | 0 | 44 |
| `star/private/rsp_relax_env.f90` | `star` | ported_unverified | ported_unverified | 1,335 | 2 | 2 | 0 | 0 | 0 | 1,296 | 0 | 0 | 0 | 39 |
| `star/private/rsp_step.f90` | `star` | ported_unverified | partial | 3,834 | 53 | 27 | 0 | 0 | 0 | 2,057 | 0 | 0 | 1,490 | 287 |
| `star/private/set_flags.f90` | `star` | stub | - | 806 | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 755 | 51 |
| `star/private/solve_omega_mix.f90` | `star` | kernel_reference_vectors | partial | 669 | 7 | 1 | 0 | 0 | 1 | 306 | 0 | 0 | 330 | 33 |
| `star/private/solver_support.f90` | `star` | ported_unverified | partial | 1,547 | 21 | 7 | 0 | 0 | 0 | 634 | 0 | 0 | 839 | 74 |
| `star/private/star_bcyclic.f90` | `star` | ported_unverified | partial | 981 | 15 | 3 | 0 | 0 | 0 | 284 | 0 | 0 | 625 | 72 |
| `star/private/star_job_ctrls_io.f90` | `star` | partial | - | 1,763 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,211 | 552 |
| `star/private/star_solver.f90` | `star` | ported_unverified | partial | 1,940 | 35 | 1 | 0 | 0 | 0 | 52 | 0 | 0 | 1,847 | 41 |
| `star/private/star_utils.f90` | `star` | partial | - | 4,149 | 156 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3,660 | 489 |
| `star/private/struct_burn_mix.f90` | `star` | partial | - | 1,213 | 19 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,150 | 63 |
| `star/private/timestep.f90` | `star` | ported_unverified | partial | 2,432 | 64 | 3 | 0 | 0 | 0 | 112 | 0 | 0 | 2,159 | 161 |
| `star/private/turb_info.f90` | `star` | ported_unverified | partial | 608 | 12 | 5 | 0 | 0 | 1 | 380 | 0 | 0 | 172 | 56 |
| `star/private/turb_support.f90` | `star` | ported_unverified | partial | 512 | 5 | 3 | 0 | 0 | 0 | 387 | 0 | 0 | 71 | 54 |
| `star/private/winds.f90` | `star` | ported_unverified | partial | 706 | 14 | 4 | 0 | 0 | 3 | 361 | 0 | 0 | 302 | 43 |
| `star/private/write_model.f90` | `star` | ported_unverified | partial | 281 | 4 | 1 | 0 | 0 | 0 | 207 | 0 | 0 | 39 | 35 |
| `star/public/star_def.f90` | `star` | ported_unverified | - | 26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 26 |
| `star_data/private/star_controls.inc` | `star_data` | partial | - | 1,412 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,412 |
| `star_data/private/star_controls_dev.inc` | `star_data` | partial | - | 49 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 49 |
| `star_data/private/star_job_controls.inc` | `star_data` | partial | - | 368 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 368 |
| `star_data/private/star_job_controls_dev.inc` | `star_data` | partial | - | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 25 |
| `star_data/private/star_job_controls_params.inc` | `star_data` | partial | - | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| `star_data/public/star_data.inc` | `star_data` | partial | - | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14 |
| `star_data/public/star_data_def.f90` | `star_data` | partial | - | 161 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 42 | 119 |
| `star_data/public/star_data_def.inc` | `star_data` | partial | - | 1,195 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,195 |
| `star_data/public/star_data_procedures.inc` | `star_data` | partial | - | 210 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 210 |
| `star_data/public/star_data_step_input.inc` | `star_data` | partial | - | 208 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 208 |
| `star_data/public/star_data_step_work.inc` | `star_data` | partial | - | 1,192 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,192 |
| `turb/private/mlt.f90` | `turb` | ported_unverified | kernel_reference_vectors | 183 | 1 | 1 | 0 | 0 | 1 | 129 | 0 | 0 | 0 | 54 |
| `turb/private/semiconvection.f90` | `turb` | ported_unverified | kernel_reference_vectors | 131 | 1 | 1 | 0 | 0 | 1 | 75 | 0 | 0 | 0 | 56 |
| `turb/private/tdc.f90` | `turb` | ported_unverified | partial | 381 | 2 | 1 | 0 | 0 | 0 | 167 | 0 | 0 | 150 | 64 |
| `turb/private/tdc_support.f90` | `turb` | partial | partial | 681 | 11 | 8 | 0 | 0 | 0 | 403 | 0 | 0 | 45 | 233 |
| `turb/private/thermohaline.f90` | `turb` | ported_unverified | partial | 363 | 6 | 1 | 0 | 0 | 0 | 43 | 0 | 0 | 264 | 56 |
| `turb/public/turb.f90` | `turb` | ported_unverified | kernel_reference_vectors | 294 | 4 | 4 | 0 | 0 | 4 | 151 | 0 | 0 | 0 | 143 |
| `utils/private/utils_idict.f90` | `utils` | partial | - | 525 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 472 | 53 |
| `utils/private/utils_nan.f90` | `utils` | partial | - | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 32 |
| `utils/private/utils_nan_dp.f90` | `utils` | partial | - | 270 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 191 | 79 |
| `utils/public/utils_def.f90` | `utils` | partial | - | 62 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 62 |
| `utils/public/utils_lib.f90` | `utils` | partial | - | 1,189 | 75 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,015 | 174 |

## Backlog of unmapped core-engine routines

1,894 routines in core-engine files with status `stub`, `partial`, or `absent` have no gate-checked `routine_map` entry (77,407 own lines). Partial and not-applicable routines are not in the backlog. Some may already be implemented without a map entry; the table cannot tell.

### How cases are attributed, and its limits

The 40 declared gaps in `known_gaps.toml` are the non-passing cases. A routine is attributed to a case only when that case's gap `reason` names the routine (an identifier equal to its Fortran name, at least eight characters or containing an underscore), basis `routine`, or names its upstream file, basis `file`, which attributes every unmapped routine of that file. A name found in several backlog files counts only when the reason also names one of the files. No other signal is used: the unsupported-controls report and `extras_status.toml` name controls and cases but no routines, so a missing control or an unported extras hook cannot be tied to a routine mechanically. Most gap reasons name neither a routine nor a file, so most routines are unattributed. Unattributed does not mean unneeded, and the case counts are lower bounds. Rows are sorted by number of cases named, then lines.

1 backlog routines are attributed to at least one case.

### Attributed routines

| # | Upstream file | Routine | Cases | Lines | Basis | Cases named |
| ---: | :--- | :--- | ---: | ---: | :--- | :--- |
| 1 | `binary/work/src/run_binary_extras.f90` | `extras_binary_finish_step` | 1 | 11 | routine | `binary/double_bh` |

### Largest unattributed routines

The 40 largest backlog routines with no case named. The full list is in the JSON file.

| # | Upstream file | Routine | Cases | Lines | Basis | Cases named |
| ---: | :--- | :--- | ---: | ---: | :--- | :--- |
| 1 | `star/private/profile_getval.f90` | `getval_for_profile` | 0 | 2,014 | - | - |
| 2 | `star/private/history.f90` | `history_getval` | 0 | 1,794 | - | - |
| 3 | `star/private/ctrls_io.f90` | `store_controls` | 0 | 1,720 | - | - |
| 4 | `star/private/ctrls_io.f90` | `set_controls_for_writing` | 0 | 1,716 | - | - |
| 5 | `star/private/alloc.f90` | `star_info_arrays` | 0 | 981 | - | - |
| 6 | `star/job/run_star_support.f90` | `do_star_job_controls_after` | 0 | 901 | - | - |
| 7 | `binary/private/run_binary_support.f90` | `do_run1_binary` | 0 | 798 | - | - |
| 8 | `star/private/star_profile_def.f90` | `profile_column_names_init` | 0 | 710 | - | - |
| 9 | `star/private/star_history_def.f90` | `history_column_names_init` | 0 | 690 | - | - |
| 10 | `star/private/star_job_ctrls_io.f90` | `store_star_job_controls` | 0 | 505 | - | - |
| 11 | `star/private/history.f90` | `do_history_info` | 0 | 504 | - | - |
| 12 | `star/private/star_job_ctrls_io.f90` | `set_star_job_controls_for_writing` | 0 | 504 | - | - |
| 13 | `star/private/evolve.f90` | `okay_energy_conservation` | 0 | 498 | - | - |
| 14 | `astero/private/extras_support.f90` | `do_astero_extras_check_model` | 0 | 444 | - | - |
| 15 | `star/private/profile.f90` | `do_profile_info` | 0 | 379 | - | - |
| 16 | `star/private/adjust_xyz.f90` | `set_new_abundances` | 0 | 373 | - | - |
| 17 | `star/private/init.f90` | `set_starting_star_data` | 0 | 348 | - | - |
| 18 | `star/private/history_specs.f90` | `add_history_columns` | 0 | 347 | - | - |
| 19 | `star/private/report.f90` | `do_report` | 0 | 323 | - | - |
| 20 | `star/private/kap_support.f90` | `get_kap` | 0 | 321 | - | - |
| 21 | `star/job/run_star_support.f90` | `do_remove_center` | 0 | 317 | - | - |
| 22 | `star/private/rotation_mix_info.f90` | `set_rotation_mixing_info` | 0 | 304 | - | - |
| 23 | `star/private/rotation_mix_info.f90` | `setup` | 0 | 294 | - | - |
| 24 | `star/private/net.f90` | `do1_net` | 0 | 289 | - | - |
| 25 | `star/private/init.f90` | `model_builder` | 0 | 281 | - | - |
| 26 | `star/private/star_private_def.f90` | `star_private_def_init` | 0 | 277 | - | - |
| 27 | `star/private/evolve.f90` | `select_mdot_action` | 0 | 270 | - | - |
| 28 | `star/private/photo_in.f90` | `read_star_photo` | 0 | 269 | - | - |
| 29 | `binary/private/binary_history.f90` | `do_binary_history_info` | 0 | 264 | - | - |
| 30 | `star/private/relax.f90` | `do_internal_evolve` | 0 | 263 | - | - |
| 31 | `star/private/atm_support.f90` | `get_table` | 0 | 260 | - | - |
| 32 | `star/private/diffusion_procs.f90` | `redistribute_mass` | 0 | 253 | - | - |
| 33 | `star/private/hydro_vars.f90` | `get_surf_PT` | 0 | 252 | - | - |
| 34 | `star/private/remove_shells.f90` | `do_relax_to_star_cut` | 0 | 249 | - | - |
| 35 | `star/private/evolve.f90` | `do_step_part2` | 0 | 245 | - | - |
| 36 | `star/private/pulse_cafein.f90` | `get_cafein_data` | 0 | 245 | - | - |
| 37 | `star/private/read_model.f90` | `do_read_saved_model` | 0 | 245 | - | - |
| 38 | `binary/private/binary_history.f90` | `binary_history_getval` | 0 | 243 | - | - |
| 39 | `star/private/pulse_gyre.f90` | `get_gyre_data` | 0 | 239 | - | - |
| 40 | `star/private/adjust_mass.f90` | `set1_xa` | 0 | 235 | - | - |
