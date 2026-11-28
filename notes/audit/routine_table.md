# Routine-level port table (generated)

<!-- GENERATED VIA scripts/dashboard/build_routine_table.py (#611). Do not edit by hand. -->

Derived from `benchmarks/dashboard/routine_inventory.json` (MESA r26.4.1, commit `c8c2026`), the `routine_map` entries of `benchmarks/dashboard/port_manifest.toml`, the verified oracle entries of `benchmarks/dashboard/test_taxonomy.toml` (see `notes/audit/oracle_provenance.md`), and the declared gaps of `benchmarks/history_harness/known_gaps.toml`. Machine-readable copy: `notes/audit/routine_table.json`.

## Evidence levels

Every routine is at one level. Lines are credited to a routine for its own lines only (its span minus the spans of the procedures contained in it), so a mapped parent does not credit its unmapped children, and the own lines of all routines plus the lines outside any routine equal the file's line count.

* `full`: the `routine_map` names the routine, the target passes the `routine_map` gate, and the routine is not listed in `routine_map_partial`.
* `partial`: mapped as above and listed in the file's `routine_map_partial`, because the Rust counterpart implements only part of the upstream routine.
* `N/A`: declared not applicable to a Rust port in `benchmarks/dashboard/port_status_na.toml` (107 declared). Never counted as mapped and never removed from a denominator; both fractions are shown.
* `unmapped`: none of the above.
* `oracle exact` / `oracle print`: the routine is mapped in full and a verified `[[tests.oracles]]` entry of an `oracle` test compares it with a committed MESA 26.4.1 artifact, at 0 ULP (`exact`) or within a stated tolerance of printed MESA values (`print`). A routine with both is counted as exact. Entries are verified by `flexmesa.testing._oracle_provenance` (DECISIONS #80); claims that fail a check are listed in `notes/audit/oracle_provenance.md` and do not count.
* `exercised` (reached by a strict-pass history case) is not derived.

39 manifest files are not in the inventory (C sources and STELLA `.trf` files) and are excluded from every figure; none of them is in the core engine modules. Contained procedures are separate routines, counted once each.

## Core engine (`star`, `star_data`, `binary`, `astero`)

| Measure | Value |
| :--- | ---: |
| Upstream lines | 144,952 |
| Lines in routines mapped in full | 39,420 |
| Lines in routines mapped in part | 50,921 |
| Lines in not-applicable routines | 2,550 |
| Lines in unmapped routines | 30,443 |
| Lines outside routines | 21,618 |
| Lines in routines with exact oracle evidence | 2,234 |
| Lines in routines with print-precision oracle evidence only | 779 |
| Line-weighted full-mapping fraction (all lines) | 27.2% |
| Line-weighted partial-mapping fraction (all lines) | 35.1% |
| Full-mapping fraction excluding not-applicable lines | 27.7% |
| Partial-mapping fraction excluding not-applicable lines | 35.8% |
| Not-applicable fraction of all lines | 1.8% |
| Line-weighted oracle fraction (exact / print) | 1.5% / 0.5% |
| Whole-file ported fraction (existing headline) | 31.1% |
| Routines (full / partial / N/A / unmapped / total) | 800 / 602 / 107 / 1,079 / 2,588 |
| Routines with oracle evidence (exact / print) | 14 / 4 |

The fractions answer different questions. The whole-file figure counts every line of a file whose hand-set status is ported; the line-weighted figures count only the own lines of routines that carry the named evidence. Neither is a statement about agreement with MESA.

## Per module

| Module | Files | Routines | Full | Partial | N/A | Unmapped | Oracle (exact / print) | Upstream lines | Lines full | Lines partial | Lines N/A | Lines unmapped | Outside routines | Full | Partial | Full excl. N/A | Oracle (exact / print) | Whole-file ported |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `star` | 173 | 2,176 | 721 | 472 | 73 | 910 | 14 / 4 | 120,509 | 37,035 | 43,386 | 1,858 | 25,572 | 12,658 | 30.7% | 36.0% | 31.2% | 1.9% / 0.6% | 35.6% |
| `star_data` | 12 | 5 | 0 | 0 | 3 | 2 | 0 / 0 | 4,870 | 0 | 0 | 23 | 26 | 4,821 | 0.0% | 0.0% | 0.0% | 0.0% / 0.0% | 0.0% |
| `binary` | 47 | 218 | 65 | 51 | 11 | 91 | 0 / 0 | 11,638 | 2,124 | 3,963 | 367 | 2,427 | 2,757 | 18.3% | 34.1% | 18.8% | 0.0% / 0.0% | 18.7% |
| `astero` | 14 | 189 | 14 | 79 | 20 | 76 | 0 / 0 | 7,935 | 261 | 3,572 | 302 | 2,418 | 1,382 | 3.3% | 45.0% | 3.4% | 0.0% / 0.0% | 0.0% |
| `eos` | 57 | 326 | 108 | 0 | 0 | 218 | 18 / 12 | 28,666 | 6,791 | 0 | 0 | 14,724 | 7,151 | 23.7% | 0.0% | 23.7% | 3.3% / 2.8% | 0.0% |
| `kap` | 38 | 311 | 41 | 50 | 0 | 220 | 0 / 0 | 22,430 | 2,257 | 3,911 | 0 | 13,817 | 2,445 | 10.1% | 17.4% | 10.1% | 0.0% / 0.0% | 0.0% |
| `net` | 16 | 195 | 35 | 84 | 0 | 76 | 0 / 0 | 13,452 | 1,702 | 9,068 | 0 | 1,449 | 1,233 | 12.7% | 67.4% | 12.7% | 0.0% / 0.0% | 0.0% |
| `rates` | 26 | 477 | 53 | 143 | 0 | 281 | 0 / 0 | 17,217 | 2,119 | 4,726 | 0 | 7,431 | 2,941 | 12.3% | 27.4% | 12.3% | 0.0% / 0.0% | 0.0% |
| `neu` | 4 | 15 | 12 | 0 | 0 | 3 | 2 / 0 | 2,184 | 1,653 | 0 | 0 | 361 | 170 | 75.7% | 0.0% | 75.7% | 13.3% / 0.0% | 83.9% |
| `atm` | 19 | 93 | 0 | 0 | 0 | 93 | 0 / 0 | 6,853 | 0 | 0 | 0 | 5,350 | 1,503 | 0.0% | 0.0% | 0.0% | 0.0% / 0.0% | 0.0% |
| `turb` | 6 | 25 | 23 | 2 | 0 | 0 | 6 / 0 | 2,033 | 1,191 | 236 | 0 | 0 | 606 | 58.6% | 11.6% | 58.6% | 17.5% / 0.0% | 66.5% |
| `chem` | 7 | 67 | 0 | 0 | 0 | 67 | 0 / 0 | 4,195 | 0 | 0 | 0 | 3,330 | 865 | 0.0% | 0.0% | 0.0% | 0.0% / 0.0% | 0.0% |
| `ionization` | 22 | 67 | 0 | 0 | 0 | 67 | 0 / 0 | 3,859 | 0 | 0 | 0 | 3,055 | 804 | 0.0% | 0.0% | 0.0% | 0.0% / 0.0% | 0.0% |
| `colors` | 10 | 89 | 0 | 0 | 0 | 89 | 0 / 0 | 3,823 | 0 | 0 | 0 | 3,155 | 668 | 0.0% | 0.0% | 0.0% | 0.0% / 0.0% | 0.0% |
| `num` | 66 | 273 | 0 | 0 | 0 | 273 | 0 / 0 | 20,045 | 0 | 0 | 0 | 17,448 | 2,597 | 0.0% | 0.0% | 0.0% | 0.0% / 0.0% | 0.0% |
| `mtx` | 23 | 163 | 0 | 0 | 0 | 163 | 0 / 0 | 5,802 | 0 | 0 | 0 | 5,147 | 655 | 0.0% | 0.0% | 0.0% | 0.0% / 0.0% | 0.0% |
| `interp_1d` | 11 | 111 | 0 | 0 | 0 | 111 | 0 / 0 | 4,969 | 0 | 0 | 0 | 4,351 | 618 | 0.0% | 0.0% | 0.0% | 0.0% / 0.0% | 0.0% |
| `interp_2d` | 10 | 82 | 2 | 0 | 0 | 80 | 2 / 0 | 13,565 | 90 | 0 | 0 | 12,965 | 510 | 0.7% | 0.0% | 0.7% | 0.7% / 0.0% | 0.9% |
| `auto_diff` | 10 | 927 | 0 | 0 | 0 | 927 | 0 / 0 | 14,353 | 0 | 0 | 0 | 10,306 | 4,047 | 0.0% | 0.0% | 0.0% | 0.0% / 0.0% | 0.0% |
| `math` | 6 | 35 | 0 | 0 | 0 | 35 | 0 / 0 | 936 | 0 | 0 | 0 | 485 | 451 | 0.0% | 0.0% | 0.0% | 0.0% / 0.0% | 0.0% |
| `const` | 2 | 2 | 0 | 0 | 0 | 2 | 0 / 0 | 256 | 0 | 0 | 0 | 48 | 208 | 0.0% | 0.0% | 0.0% | 0.0% / 0.0% | 100.0% |
| `utils` | 11 | 150 | 0 | 0 | 0 | 150 | 0 / 0 | 3,398 | 0 | 0 | 0 | 2,614 | 784 | 0.0% | 0.0% | 0.0% | 0.0% / 0.0% | 0.0% |
| `gyre` | 171 | 983 | 198 | 85 | 0 | 700 | 0 / 0 | 55,708 | 7,874 | 4,891 | 0 | 27,976 | 14,967 | 14.1% | 8.8% | 14.1% | 0.0% / 0.0% | 0.0% |
| `adipls` | 133 | 221 | 36 | 10 | 0 | 175 | 0 / 0 | 30,923 | 5,633 | 3,889 | 0 | 21,242 | 159 | 18.2% | 12.6% | 18.2% | 0.0% / 0.0% | 0.0% |
| `stella` | 109 | 238 | 0 | 0 | 0 | 238 | 0 / 0 | 45,351 | 0 | 0 | 0 | 39,716 | 5,635 | 0.0% | 0.0% | 0.0% | 0.0% / 0.0% | 0.0% |
| `plotting` | 58 | 559 | 0 | 0 | 0 | 559 | 0 / 0 | 35,732 | 0 | 0 | 0 | 25,133 | 10,599 | 0.0% | 0.0% | 0.0% | 0.0% / 0.0% | 0.0% |

## Derived status ceiling

The ceiling is the highest status that the routine evidence of a file supports. The rule counts full mappings only, over the file's applicable routines (those not declared not applicable):

* none mapped in full or in part, or no applicable routines: no ceiling claim;
* any applicable routine partial or unmapped: `partial`, and the file is not eligible for a raise;
* all applicable routines mapped in full: `ported_unverified`;
* all mapped in full and each has exact (0-ULP) oracle evidence: `kernel_reference_vectors` (print-precision evidence does not count).

`case_level_evidence` is never a ceiling because the `exercised` level is not derived. Contained procedures count as routines. A ceiling over the hand-set status is a candidate for review, not a raise: the table reads no status as evidence and changes none, and nothing gates on this section.

### Hand-set status above the ceiling (37 files)

Files whose status needs more than the routine evidence gives. Files with no ceiling claim are not listed; 5 files with a ported status have no ceiling claim. Largest ten:

| File | Status | Ceiling | Lines | Routines | Full | Partial | N/A | Oracle (exact / print) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `star/private/rsp_step.f90` | ported_unverified | partial | 3,834 | 53 | 39 | 6 | 0 | 0 / 0 |
| `star/private/mesh_adjust.f90` | ported_unverified | partial | 2,889 | 39 | 16 | 13 | 0 | 0 / 0 |
| `star/private/timestep.f90` | ported_unverified | partial | 2,432 | 64 | 58 | 5 | 0 | 0 / 0 |
| `star/private/diffusion.f90` | kernel_reference_vectors | partial | 2,050 | 30 | 13 | 5 | 0 | 0 / 1 |
| `star/private/star_solver.f90` | ported_unverified | partial | 1,940 | 35 | 4 | 8 | 0 | 0 / 0 |
| `star/private/diffusion_support.f90` | kernel_reference_vectors | partial | 1,800 | 16 | 11 | 5 | 0 | 1 / 0 |
| `neu/private/mod_neu.f90` | ported_unverified | partial | 1,715 | 12 | 11 | 0 | 0 | 1 / 0 |
| `star/private/conv_premix.f90` | ported_unverified | partial | 1,561 | 16 | 14 | 0 | 0 | 0 / 0 |
| `star/private/solver_support.f90` | ported_unverified | partial | 1,547 | 21 | 9 | 3 | 0 | 0 / 0 |
| `star/private/adjust_mesh_split_merge.f90` | ported_unverified | partial | 1,461 | 22 | 12 | 6 | 0 | 0 / 0 |

### `partial` files with every applicable routine mapped in full (59 files)

Candidates for a reviewed raise to `ported_unverified`. Largest ten:

| File | Status | Ceiling | Lines | Routines | Full | Partial | N/A | Oracle (exact / print) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adipls/adipack.c/adipls/setbcs.n.d.f` | partial | ported_unverified | 774 | 1 | 1 | 0 | 0 | 0 / 0 |
| `turb/private/tdc_support.f90` | partial | ported_unverified | 681 | 11 | 11 | 0 | 0 | 0 / 0 |
| `star/private/mass_utils.f90` | partial | ported_unverified | 594 | 9 | 9 | 0 | 0 | 0 / 0 |
| `adipls/adipack.c/adipls/sigscn.c.d.f` | partial | ported_unverified | 480 | 2 | 2 | 0 | 0 | 0 / 0 |
| `gyre/gyre/src/grid/grid_m.fypp` | partial | ported_unverified | 466 | 13 | 13 | 0 | 0 | 0 / 0 |
| `adipls/adipack.c/adipls/varfrq.n.d.f` | partial | ported_unverified | 427 | 1 | 1 | 0 | 0 | 0 / 0 |
| `rates/private/screen5.f90` | partial | ported_unverified | 396 | 2 | 2 | 0 | 0 | 0 / 0 |
| `eos/private/skye_coulomb_solid.f90` | partial | ported_unverified | 348 | 8 | 8 | 0 | 0 | 0 / 0 |
| `rates/private/pycno.f90` | partial | ported_unverified | 308 | 1 | 1 | 0 | 0 | 0 / 0 |
| `adipls/adipack.c/adipls/order.d.f` | partial | ported_unverified | 293 | 1 | 1 | 0 | 0 | 0 / 0 |

## Per file (files with a non-absent status or a mapped routine)

| File | Module | Status | Ceiling | Lines | Routines | Full | Partial | N/A | Oracle (exact / print) | Lines full | Lines partial | Lines N/A | Lines unmapped | Outside |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adipls/adipack.c/adipls/adipls.c.d.f` | `adipls` | partial | partial | 2,369 | 5 | 0 | 2 | 0 | 0 / 0 | 0 | 2,205 | 0 | 164 | 0 |
| `adipls/adipack.c/adipls/adirhs.c.d.f` | `adipls` | partial | ported_unverified | 252 | 1 | 1 | 0 | 0 | 0 / 0 | 252 | 0 | 0 | 0 | 0 |
| `adipls/adipack.c/adipls/danorm.d.f` | `adipls` | partial | ported_unverified | 34 | 1 | 1 | 0 | 0 | 0 / 0 | 32 | 0 | 0 | 0 | 2 |
| `adipls/adipack.c/adipls/geninh.d.f` | `adipls` | partial | partial | 60 | 2 | 0 | 1 | 0 | 0 / 0 | 0 | 51 | 0 | 9 | 0 |
| `adipls/adipack.c/adipls/kiner.n.d.f` | `adipls` | partial | ported_unverified | 87 | 1 | 1 | 0 | 0 | 0 / 0 | 85 | 0 | 0 | 0 | 2 |
| `adipls/adipack.c/adipls/mchcff.d.f` | `adipls` | partial | ported_unverified | 151 | 2 | 2 | 0 | 0 | 0 / 0 | 149 | 0 | 0 | 0 | 2 |
| `adipls/adipack.c/adipls/mchsol.n.d.f` | `adipls` | partial | partial | 166 | 1 | 0 | 1 | 0 | 0 / 0 | 0 | 166 | 0 | 0 | 0 |
| `adipls/adipack.c/adipls/order.d.f` | `adipls` | partial | ported_unverified | 293 | 1 | 1 | 0 | 0 | 0 / 0 | 293 | 0 | 0 | 0 | 0 |
| `adipls/adipack.c/adipls/res_adimod.c.d.f` | `adipls` | partial | ported_unverified | 195 | 1 | 1 | 0 | 0 | 0 / 0 | 195 | 0 | 0 | 0 | 0 |
| `adipls/adipack.c/adipls/rsnfit.n.d.f` | `adipls` | partial | ported_unverified | 68 | 1 | 1 | 0 | 0 | 0 / 0 | 68 | 0 | 0 | 0 | 0 |
| `adipls/adipack.c/adipls/sclasl.c.d.f` | `adipls` | partial | ported_unverified | 45 | 1 | 1 | 0 | 0 | 0 / 0 | 45 | 0 | 0 | 0 | 0 |
| `adipls/adipack.c/adipls/setbcs.n.d.f` | `adipls` | partial | ported_unverified | 774 | 1 | 1 | 0 | 0 | 0 / 0 | 774 | 0 | 0 | 0 | 0 |
| `adipls/adipack.c/adipls/setobs_st.c.d.f` | `adipls` | partial | ported_unverified | 150 | 1 | 1 | 0 | 0 | 0 / 0 | 150 | 0 | 0 | 0 | 0 |
| `adipls/adipack.c/adipls/setssm.n.d.f` | `adipls` | partial | ported_unverified | 59 | 1 | 1 | 0 | 0 | 0 / 0 | 59 | 0 | 0 | 0 | 0 |
| `adipls/adipack.c/adipls/setups_adi.c.d.f` | `adipls` | partial | ported_unverified | 90 | 2 | 2 | 0 | 0 | 0 / 0 | 90 | 0 | 0 | 0 | 0 |
| `adipls/adipack.c/adipls/shtint.c.d.f` | `adipls` | partial | ported_unverified | 235 | 1 | 1 | 0 | 0 | 0 / 0 | 235 | 0 | 0 | 0 | 0 |
| `adipls/adipack.c/adipls/sigint.n.d.f` | `adipls` | partial | partial | 93 | 1 | 0 | 1 | 0 | 0 / 0 | 0 | 93 | 0 | 0 | 0 |
| `adipls/adipack.c/adipls/sigitc.c.d.f` | `adipls` | partial | ported_unverified | 114 | 1 | 1 | 0 | 0 | 0 / 0 | 114 | 0 | 0 | 0 | 0 |
| `adipls/adipack.c/adipls/sigitr.n.d.f` | `adipls` | partial | ported_unverified | 243 | 1 | 1 | 0 | 0 | 0 / 0 | 243 | 0 | 0 | 0 | 0 |
| `adipls/adipack.c/adipls/signew.n.d.f` | `adipls` | partial | partial | 371 | 1 | 0 | 1 | 0 | 0 / 0 | 0 | 371 | 0 | 0 | 0 |
| `adipls/adipack.c/adipls/sigout.c.d.f` | `adipls` | partial | partial | 418 | 1 | 0 | 1 | 0 | 0 / 0 | 0 | 418 | 0 | 0 | 0 |
| `adipls/adipack.c/adipls/sigscn.c.d.f` | `adipls` | partial | ported_unverified | 480 | 2 | 2 | 0 | 0 | 0 / 0 | 480 | 0 | 0 | 0 | 0 |
| `adipls/adipack.c/adipls/sigsol.c.d.f` | `adipls` | partial | partial | 470 | 1 | 0 | 1 | 0 | 0 / 0 | 0 | 470 | 0 | 0 | 0 |
| `adipls/adipack.c/adipls/stevft.n.d.f` | `adipls` | partial | ported_unverified | 152 | 1 | 1 | 0 | 0 | 0 / 0 | 152 | 0 | 0 | 0 | 0 |
| `adipls/adipack.c/adipls/takata.d.f` | `adipls` | partial | ported_unverified | 96 | 1 | 1 | 0 | 0 | 0 / 0 | 96 | 0 | 0 | 0 | 0 |
| `adipls/adipack.c/adipls/varfrq.n.d.f` | `adipls` | partial | ported_unverified | 427 | 1 | 1 | 0 | 0 | 0 / 0 | 427 | 0 | 0 | 0 | 0 |
| `adipls/adipack.c/adipls/xlmult.d.f` | `adipls` | partial | ported_unverified | 124 | 1 | 1 | 0 | 0 | 0 / 0 | 122 | 0 | 0 | 0 | 2 |
| `adipls/adipack.c/gensr/aramax.d.f` | `adipls` | partial | ported_unverified | 96 | 1 | 1 | 0 | 0 | 0 / 0 | 96 | 0 | 0 | 0 | 0 |
| `adipls/adipack.c/gensr/derive.d.f` | `adipls` | partial | ported_unverified | 234 | 1 | 1 | 0 | 0 | 0 / 0 | 234 | 0 | 0 | 0 | 0 |
| `adipls/adipack.c/gensr/derivk.new.d.f` | `adipls` | partial | ported_unverified | 193 | 1 | 1 | 0 | 0 | 0 / 0 | 193 | 0 | 0 | 0 | 0 |
| `adipls/adipack.c/gensr/leq.d.f` | `adipls` | partial | ported_unverified | 230 | 2 | 2 | 0 | 0 | 0 / 0 | 230 | 0 | 0 | 0 | 0 |
| `adipls/adipack.c/gensr/lininh4.d.f` | `adipls` | partial | ported_unverified | 259 | 1 | 1 | 0 | 0 | 0 / 0 | 259 | 0 | 0 | 0 | 0 |
| `adipls/adipack.c/gensr/lir.d.f` | `adipls` | partial | ported_unverified | 167 | 1 | 1 | 0 | 0 | 0 / 0 | 167 | 0 | 0 | 0 | 0 |
| `adipls/adipack.c/gensr/notwin.d.f` | `adipls` | partial | ported_unverified | 21 | 1 | 1 | 0 | 0 | 0 / 0 | 21 | 0 | 0 | 0 | 0 |
| `adipls/adipack.c/gensr/notwni.f` | `adipls` | partial | ported_unverified | 12 | 1 | 1 | 0 | 0 | 0 / 0 | 12 | 0 | 0 | 0 | 0 |
| `adipls/adipack.c/gensr/ofiles.c.f` | `adipls` | partial | partial | 506 | 12 | 0 | 2 | 0 | 0 / 0 | 0 | 115 | 0 | 391 | 0 |
| `adipls/adipack.c/gensr/store.d.f` | `adipls` | partial | ported_unverified | 21 | 1 | 1 | 0 | 0 | 0 / 0 | 21 | 0 | 0 | 0 | 0 |
| `adipls/adipack.c/gensr/vinta.d.f` | `adipls` | partial | ported_unverified | 134 | 1 | 1 | 0 | 0 | 0 / 0 | 134 | 0 | 0 | 0 | 0 |
| `adipls/adipack.c/gensr/vintk.new.d.f` | `adipls` | partial | ported_unverified | 182 | 1 | 1 | 0 | 0 | 0 / 0 | 182 | 0 | 0 | 0 | 0 |
| `adipls/adipack.c/gensr/zero.d.f` | `adipls` | partial | ported_unverified | 23 | 1 | 1 | 0 | 0 | 0 / 0 | 23 | 0 | 0 | 0 | 0 |
| `astero/private/astero_run_support.f90` | `astero` | partial | partial | 1,127 | 22 | 2 | 12 | 0 | 0 / 0 | 66 | 686 | 0 | 306 | 69 |
| `astero/private/astero_support.f90` | `astero` | partial | partial | 1,530 | 43 | 5 | 35 | 2 | 0 / 0 | 111 | 1,280 | 6 | 18 | 115 |
| `astero/private/extras_support.f90` | `astero` | partial | partial | 1,238 | 28 | 2 | 11 | 7 | 0 / 0 | 38 | 771 | 84 | 269 | 76 |
| `astero/public/astero_def.f90` | `astero` | partial | partial | 1,709 | 31 | 4 | 16 | 11 | 0 / 0 | 41 | 752 | 212 | 0 | 704 |
| `astero/public/astero_lib.f90` | `astero` | partial | partial | 146 | 6 | 1 | 5 | 0 | 0 / 0 | 5 | 83 | 0 | 0 | 58 |
| `atm/private/atm_irradiated.f90` | `atm` | partial | - | 334 | 2 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 294 | 40 |
| `atm/private/atm_t_tau_relations.f90` | `atm` | partial | - | 367 | 11 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 311 | 56 |
| `atm/private/atm_t_tau_uniform.f90` | `atm` | partial | - | 702 | 6 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 655 | 47 |
| `atm/private/atm_t_tau_varying.f90` | `atm` | partial | - | 824 | 9 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 776 | 48 |
| `atm/private/atm_table.f90` | `atm` | partial | - | 447 | 3 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 407 | 40 |
| `atm/private/atm_utils.f90` | `atm` | partial | - | 207 | 6 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 154 | 53 |
| `atm/private/e2_pairs.dek` | `atm` | partial | - | 581 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 581 |
| `atm/private/table_atm.f90` | `atm` | partial | - | 684 | 8 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 644 | 40 |
| `atm/public/atm_def.f90` | `atm` | partial | - | 146 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 146 |
| `atm/public/atm_lib.f90` | `atm` | partial | - | 93 | 3 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 16 | 77 |
| `auto_diff/private/auto_diff_real_15var_order1_module.f90` | `auto_diff` | partial | - | 1,132 | 101 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 714 | 418 |
| `auto_diff/private/auto_diff_real_1var_order1_module.f90` | `auto_diff` | partial | - | 1,319 | 116 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 820 | 499 |
| `auto_diff/private/auto_diff_real_2var_order1_module.f90` | `auto_diff` | partial | - | 1,545 | 117 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 1,039 | 506 |
| `auto_diff/private/auto_diff_real_2var_order2_module.f90` | `auto_diff` | partial | - | 2,009 | 117 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 1,500 | 509 |
| `auto_diff/private/auto_diff_real_2var_order3_module.f90` | `auto_diff` | partial | - | 3,410 | 117 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 2,897 | 513 |
| `auto_diff/private/auto_diff_real_4var_order1_module.f90` | `auto_diff` | partial | - | 1,751 | 119 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 1,231 | 520 |
| `auto_diff/private/auto_diff_real_star_order1_module.f90` | `auto_diff` | partial | - | 1,307 | 115 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 814 | 493 |
| `auto_diff/private/support_functions.f90` | `auto_diff` | partial | - | 135 | 9 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 77 | 58 |
| `auto_diff/public/auto_diff.f90` | `auto_diff` | partial | - | 30 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 30 |
| `binary/job/run_binary.f90` | `binary` | ported_unverified | ported_unverified | 44 | 1 | 1 | 0 | 0 | 0 / 0 | 17 | 0 | 0 | 0 | 27 |
| `binary/private/binary_ctrls_io.f90` | `binary` | partial | partial | 821 | 9 | 0 | 5 | 1 | 0 / 0 | 0 | 302 | 197 | 69 | 253 |
| `binary/private/binary_disk.f90` | `binary` | ported_unverified | ported_unverified | 318 | 7 | 7 | 0 | 0 | 0 / 0 | 265 | 0 | 0 | 0 | 53 |
| `binary/private/binary_edot.f90` | `binary` | ported_unverified | ported_unverified | 231 | 4 | 4 | 0 | 0 | 0 / 0 | 186 | 0 | 0 | 0 | 45 |
| `binary/private/binary_evolve.f90` | `binary` | partial | partial | 698 | 9 | 0 | 8 | 1 | 0 / 0 | 0 | 600 | 54 | 0 | 44 |
| `binary/private/binary_history.f90` | `binary` | partial | partial | 880 | 21 | 0 | 14 | 1 | 0 / 0 | 0 | 662 | 6 | 167 | 45 |
| `binary/private/binary_history_specs.f90` | `binary` | partial | partial | 286 | 7 | 0 | 4 | 3 | 0 / 0 | 0 | 224 | 20 | 0 | 42 |
| `binary/private/binary_irradiation.f90` | `binary` | ported_unverified | ported_unverified | 62 | 1 | 1 | 0 | 0 | 0 / 0 | 28 | 0 | 0 | 0 | 34 |
| `binary/private/binary_jdot.f90` | `binary` | ported_unverified | ported_unverified | 330 | 7 | 7 | 0 | 0 | 0 / 0 | 290 | 0 | 0 | 0 | 40 |
| `binary/private/binary_job_controls.inc` | `binary` | partial | - | 43 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 43 |
| `binary/private/binary_job_ctrls_io.f90` | `binary` | partial | partial | 331 | 8 | 0 | 4 | 1 | 0 / 0 | 0 | 118 | 46 | 80 | 87 |
| `binary/private/binary_mdot.f90` | `binary` | ported_unverified | partial | 1,023 | 14 | 7 | 7 | 0 | 0 / 0 | 288 | 685 | 0 | 0 | 50 |
| `binary/private/binary_roche_deformation.f90` | `binary` | partial | partial | 416 | 11 | 9 | 0 | 1 | 0 / 0 | 291 | 0 | 24 | 36 | 65 |
| `binary/private/binary_tides.f90` | `binary` | partial | partial | 461 | 8 | 7 | 1 | 0 | 0 / 0 | 357 | 57 | 0 | 0 | 47 |
| `binary/private/binary_timestep.f90` | `binary` | partial | partial | 325 | 2 | 0 | 2 | 0 | 0 / 0 | 0 | 289 | 0 | 0 | 36 |
| `binary/private/binary_utils.f90` | `binary` | partial | partial | 215 | 9 | 7 | 1 | 0 | 0 / 0 | 130 | 22 | 0 | 20 | 43 |
| `binary/private/binary_wind.f90` | `binary` | ported_unverified | ported_unverified | 173 | 3 | 3 | 0 | 0 | 0 / 0 | 137 | 0 | 0 | 0 | 36 |
| `binary/private/run_binary_support.f90` | `binary` | partial | partial | 1,016 | 4 | 1 | 3 | 0 | 0 / 0 | 22 | 953 | 0 | 0 | 41 |
| `binary/public/binary_controls.inc` | `binary` | partial | - | 190 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 190 |
| `binary/public/binary_data.inc` | `binary` | partial | - | 326 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 326 |
| `binary/public/binary_def.f90` | `binary` | partial | - | 292 | 4 | 0 | 0 | 3 | 0 / 0 | 0 | 0 | 20 | 5 | 267 |
| `binary/public/binary_lib.f90` | `binary` | partial | partial | 272 | 17 | 11 | 2 | 0 | 0 / 0 | 113 | 51 | 0 | 60 | 48 |
| `chem/private/chem_isos_io.f90` | `chem` | partial | - | 233 | 4 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 200 | 33 |
| `chem/private/lodders_mod.f90` | `chem` | partial | - | 106 | 2 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 76 | 30 |
| `chem/private/nuclide_set_mod.f90` | `chem` | partial | - | 112 | 3 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 80 | 32 |
| `chem/public/chem_def.f90` | `chem` | partial | - | 2,654 | 22 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 2,087 | 567 |
| `chem/public/chem_lib.f90` | `chem` | partial | - | 717 | 24 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 569 | 148 |
| `colors/private/bolometric.f90` | `colors` | partial | - | 191 | 4 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 149 | 42 |
| `colors/private/colors_utils.f90` | `colors` | partial | - | 1,111 | 25 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 1,019 | 92 |
| `colors/private/hermite_interp.f90` | `colors` | partial | - | 443 | 10 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 384 | 59 |
| `colors/private/knn_interp.f90` | `colors` | partial | - | 438 | 7 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 385 | 53 |
| `colors/private/linear_interp.f90` | `colors` | partial | - | 342 | 4 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 290 | 52 |
| `colors/private/synthetic.f90` | `colors` | partial | - | 230 | 5 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 183 | 47 |
| `colors/public/colors_def.f90` | `colors` | partial | - | 319 | 6 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 176 | 143 |
| `colors/public/colors_lib.f90` | `colors` | partial | - | 321 | 18 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 246 | 75 |
| `const/public/const_def.f90` | `const` | ported_unverified | - | 218 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 42 | 176 |
| `const/public/const_lib.f90` | `const` | ported_unverified | - | 38 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 6 | 32 |
| `eos/private/eos_blend.f90` | `eos` | partial | partial | 208 | 5 | 2 | 0 | 0 | 2 / 0 | 48 | 0 | 0 | 92 | 68 |
| `eos/private/eos_ctrls_io.f90` | `eos` | partial | - | 697 | 8 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 414 | 283 |
| `eos/private/eos_helm_eval.f90` | `eos` | partial | partial | 361 | 4 | 2 | 0 | 0 | 0 / 1 | 214 | 0 | 0 | 105 | 42 |
| `eos/private/eos_initialize.f90` | `eos` | partial | - | 75 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 48 | 27 |
| `eos/private/eoscms_eval.f90` | `eos` | partial | partial | 703 | 8 | 4 | 0 | 0 | 0 / 1 | 432 | 0 | 0 | 197 | 74 |
| `eos/private/eosde_eval.f90` | `eos` | partial | ported_unverified | 113 | 1 | 1 | 0 | 0 | 0 / 1 | 82 | 0 | 0 | 0 | 31 |
| `eos/private/eosdt_eval.f90` | `eos` | partial | partial | 2,537 | 30 | 10 | 0 | 0 | 3 / 1 | 985 | 0 | 0 | 1,430 | 122 |
| `eos/private/eosdt_load_tables.f90` | `eos` | partial | - | 643 | 12 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 579 | 64 |
| `eos/private/eosdt_support.f90` | `eos` | partial | partial | 241 | 2 | 1 | 0 | 0 | 0 / 0 | 141 | 0 | 0 | 65 | 35 |
| `eos/private/eospc_eval.f90` | `eos` | partial | partial | 365 | 4 | 3 | 0 | 0 | 1 / 1 | 164 | 0 | 0 | 165 | 36 |
| `eos/private/eospt_eval.f90` | `eos` | partial | partial | 606 | 8 | 4 | 0 | 0 | 2 / 1 | 365 | 0 | 0 | 197 | 44 |
| `eos/private/gauss_fermi.f90` | `eos` | partial | partial | 1,199 | 11 | 1 | 0 | 0 | 0 / 1 | 105 | 0 | 0 | 1,036 | 58 |
| `eos/private/helm.f90` | `eos` | partial | partial | 350 | 3 | 1 | 0 | 0 | 1 / 0 | 33 | 0 | 0 | 276 | 41 |
| `eos/private/helm_alloc.f90` | `eos` | partial | - | 468 | 8 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 432 | 36 |
| `eos/private/helm_coulomb2.dek` | `eos` | partial | - | 420 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 420 |
| `eos/private/helm_declare_local_variables.dek` | `eos` | partial | - | 341 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 341 |
| `eos/private/helm_electron_positron.dek` | `eos` | partial | - | 641 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 641 |
| `eos/private/helm_gammas.dek` | `eos` | partial | - | 271 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 271 |
| `eos/private/helm_ideal_ions.dek` | `eos` | partial | - | 220 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 220 |
| `eos/private/helm_initialize_local_variables.dek` | `eos` | partial | - | 521 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 521 |
| `eos/private/helm_polynomials.f90` | `eos` | partial | ported_unverified | 185 | 22 | 22 | 0 | 0 | 0 / 0 | 121 | 0 | 0 | 0 | 64 |
| `eos/private/helm_print_results.dek` | `eos` | partial | - | 247 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 247 |
| `eos/private/helm_radiation.dek` | `eos` | partial | - | 67 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 67 |
| `eos/private/helm_store_results.dek` | `eos` | partial | - | 437 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 437 |
| `eos/private/helm_sum_totals.dek` | `eos` | partial | - | 169 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 169 |
| `eos/private/ideal.f90` | `eos` | partial | partial | 206 | 4 | 2 | 0 | 0 | 2 / 0 | 55 | 0 | 0 | 113 | 38 |
| `eos/private/ion_offset.f90` | `eos` | partial | kernel_reference_vectors | 84 | 1 | 1 | 0 | 0 | 1 / 0 | 43 | 0 | 0 | 0 | 41 |
| `eos/private/pc_eos.f90` | `eos` | partial | partial | 3,028 | 26 | 22 | 0 | 0 | 0 / 0 | 2,679 | 0 | 0 | 111 | 238 |
| `eos/private/pc_support.f90` | `eos` | partial | partial | 1,043 | 13 | 1 | 0 | 0 | 0 / 0 | 74 | 0 | 0 | 894 | 75 |
| `eos/private/skye.f90` | `eos` | partial | partial | 402 | 5 | 3 | 0 | 0 | 1 / 2 | 167 | 0 | 0 | 170 | 65 |
| `eos/private/skye_coulomb.f90` | `eos` | partial | partial | 486 | 7 | 1 | 0 | 0 | 0 / 0 | 72 | 0 | 0 | 282 | 132 |
| `eos/private/skye_coulomb_liquid.f90` | `eos` | partial | ported_unverified | 181 | 4 | 4 | 0 | 0 | 0 / 0 | 107 | 0 | 0 | 0 | 74 |
| `eos/private/skye_coulomb_solid.f90` | `eos` | partial | ported_unverified | 348 | 8 | 8 | 0 | 0 | 0 / 0 | 236 | 0 | 0 | 0 | 112 |
| `eos/private/skye_ideal.f90` | `eos` | partial | partial | 708 | 4 | 3 | 0 | 0 | 0 / 0 | 42 | 0 | 0 | 622 | 44 |
| `eos/private/skye_thermodynamics.f90` | `eos` | partial | ported_unverified | 241 | 3 | 3 | 0 | 0 | 0 / 0 | 158 | 0 | 0 | 0 | 83 |
| `eos/public/eos_def.f90` | `eos` | partial | - | 642 | 8 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 210 | 432 |
| `eos/public/eos_lib.f90` | `eos` | partial | partial | 910 | 29 | 9 | 0 | 0 | 5 / 3 | 468 | 0 | 0 | 335 | 107 |
| `eos/public/helm_def.dek` | `eos` | partial | - | 646 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 646 |
| `gyre/gyre/src/ad/ad_bound_m.fypp` | `gyre` | partial | partial | 903 | 14 | 6 | 3 | 0 | 0 / 0 | 272 | 202 | 0 | 282 | 147 |
| `gyre/gyre/src/ad/ad_bvp_m.fypp` | `gyre` | partial | partial | 262 | 4 | 2 | 1 | 0 | 0 / 0 | 74 | 56 | 0 | 38 | 94 |
| `gyre/gyre/src/ad/ad_diff_m.fypp` | `gyre` | partial | ported_unverified | 134 | 2 | 2 | 0 | 0 | 0 / 0 | 60 | 0 | 0 | 0 | 74 |
| `gyre/gyre/src/ad/ad_eqns_m.fypp` | `gyre` | partial | ported_unverified | 271 | 4 | 4 | 0 | 0 | 0 / 0 | 173 | 0 | 0 | 0 | 98 |
| `gyre/gyre/src/ad/ad_match_m.fypp` | `gyre` | partial | ported_unverified | 222 | 3 | 3 | 0 | 0 | 0 / 0 | 136 | 0 | 0 | 0 | 86 |
| `gyre/gyre/src/ad/ad_trans_m.fypp` | `gyre` | partial | partial | 1,039 | 19 | 4 | 7 | 0 | 0 / 0 | 287 | 262 | 0 | 330 | 160 |
| `gyre/gyre/src/bvp/bvp_m.fypp` | `gyre` | partial | partial | 330 | 8 | 4 | 1 | 0 | 0 / 0 | 92 | 44 | 0 | 84 | 110 |
| `gyre/gyre/src/common/atmos_m.fypp` | `gyre` | partial | partial | 298 | 5 | 2 | 0 | 0 | 0 / 0 | 78 | 0 | 0 | 148 | 72 |
| `gyre/gyre/src/common/constants_m.fypp` | `gyre` | partial | partial | 188 | 3 | 2 | 0 | 0 | 0 / 0 | 48 | 0 | 0 | 41 | 99 |
| `gyre/gyre/src/common/freq_context_m.fypp` | `gyre` | partial | partial | 248 | 4 | 0 | 1 | 0 | 0 / 0 | 0 | 48 | 0 | 126 | 74 |
| `gyre/gyre/src/common/freq_m.fypp` | `gyre` | partial | partial | 160 | 4 | 2 | 1 | 0 | 0 / 0 | 47 | 24 | 0 | 24 | 65 |
| `gyre/gyre/src/common/string_m.fypp` | `gyre` | partial | partial | 393 | 8 | 1 | 0 | 0 | 0 / 0 | 69 | 0 | 0 | 233 | 91 |
| `gyre/gyre/src/common/util_m.fypp` | `gyre` | partial | partial | 265 | 6 | 0 | 2 | 0 | 0 / 0 | 0 | 48 | 0 | 125 | 92 |
| `gyre/gyre/src/context/context_m.fypp` | `gyre` | partial | partial | 571 | 15 | 3 | 6 | 0 | 0 / 0 | 48 | 200 | 0 | 174 | 149 |
| `gyre/gyre/src/diff/colloc_diff_m.fypp` | `gyre` | partial | partial | 391 | 6 | 1 | 4 | 0 | 0 / 0 | 29 | 202 | 0 | 21 | 139 |
| `gyre/gyre/src/diff/diff_factory_m.fypp` | `gyre` | partial | partial | 102 | 1 | 0 | 1 | 0 | 0 / 0 | 0 | 36 | 0 | 0 | 66 |
| `gyre/gyre/src/diff/magnus_diff_m.fypp` | `gyre` | partial | partial | 358 | 5 | 1 | 2 | 0 | 0 / 0 | 23 | 159 | 0 | 66 | 110 |
| `gyre/gyre/src/ext/ext_complex_m.fypp` | `gyre` | partial | partial | 921 | 33 | 24 | 0 | 0 | 0 / 0 | 525 | 0 | 0 | 133 | 263 |
| `gyre/gyre/src/ext/ext_real_m.fypp` | `gyre` | partial | partial | 810 | 27 | 23 | 1 | 0 | 0 / 0 | 520 | 15 | 0 | 44 | 231 |
| `gyre/gyre/src/forum/src/lib/order_m.fypp` | `gyre` | partial | partial | 554 | 6 | 0 | 5 | 0 | 0 / 0 | 0 | 418 | 0 | 42 | 94 |
| `gyre/gyre/src/grid/grid_m.fypp` | `gyre` | partial | ported_unverified | 466 | 13 | 13 | 0 | 0 | 0 / 0 | 358 | 0 | 0 | 0 | 108 |
| `gyre/gyre/src/grid/grid_refine_m.fypp` | `gyre` | partial | partial | 701 | 7 | 3 | 2 | 0 | 0 / 0 | 274 | 202 | 0 | 145 | 80 |
| `gyre/gyre/src/grid/grid_scaffold_m.fypp` | `gyre` | partial | partial | 207 | 2 | 0 | 1 | 0 | 0 / 0 | 0 | 29 | 0 | 120 | 58 |
| `gyre/gyre/src/grid/grid_util_m.fypp` | `gyre` | partial | ported_unverified | 227 | 3 | 3 | 0 | 0 | 0 / 0 | 170 | 0 | 0 | 0 | 57 |
| `gyre/gyre/src/interp/cubic_interp_m.fypp` | `gyre` | partial | partial | 369 | 10 | 2 | 3 | 0 | 0 / 0 | 60 | 105 | 0 | 89 | 115 |
| `gyre/gyre/src/interp/deriv_m.fypp` | `gyre` | partial | partial | 272 | 4 | 1 | 1 | 0 | 0 / 0 | 59 | 29 | 0 | 115 | 69 |
| `gyre/gyre/src/lib/gyre_mesa_m.fypp` | `gyre` | partial | partial | 484 | 9 | 2 | 3 | 0 | 0 / 0 | 48 | 239 | 0 | 96 | 101 |
| `gyre/gyre/src/math/crmath/math_m.fypp` | `gyre` | partial | partial | 153 | 2 | 1 | 1 | 0 | 0 / 0 | 14 | 15 | 0 | 0 | 124 |
| `gyre/gyre/src/math/integ_m.fypp` | `gyre` | partial | partial | 124 | 2 | 0 | 2 | 0 | 0 / 0 | 0 | 57 | 0 | 0 | 67 |
| `gyre/gyre/src/math/root_m.fypp` | `gyre` | partial | partial | 1,030 | 9 | 0 | 2 | 0 | 0 / 0 | 0 | 383 | 0 | 539 | 108 |
| `gyre/gyre/src/matrix/block_sysmtx_m.fypp` | `gyre` | partial | partial | 744 | 13 | 9 | 0 | 0 | 0 / 0 | 451 | 0 | 0 | 166 | 127 |
| `gyre/gyre/src/matrix/linalg_m.fypp` | `gyre` | partial | partial | 872 | 19 | 4 | 2 | 0 | 0 / 0 | 298 | 63 | 0 | 344 | 167 |
| `gyre/gyre/src/matrix/sysmtx_factory_m.fypp` | `gyre` | partial | partial | 87 | 1 | 0 | 1 | 0 | 0 / 0 | 0 | 25 | 0 | 0 | 62 |
| `gyre/gyre/src/mode/mode_m.fypp` | `gyre` | partial | partial | 356 | 4 | 2 | 1 | 0 | 0 / 0 | 97 | 144 | 0 | 28 | 87 |
| `gyre/gyre/src/mode/wave_m.fypp` | `gyre` | partial | partial | 2,522 | 62 | 15 | 2 | 0 | 0 / 0 | 633 | 70 | 0 | 1,465 | 354 |
| `gyre/gyre/src/model/evol_model_m.fypp` | `gyre` | partial | partial | 790 | 17 | 7 | 3 | 0 | 0 / 0 | 337 | 157 | 0 | 157 | 139 |
| `gyre/gyre/src/model/mesa_file_m.fypp` | `gyre` | partial | partial | 513 | 6 | 0 | 2 | 0 | 0 / 0 | 0 | 213 | 0 | 238 | 62 |
| `gyre/gyre/src/model/model_util_m.fypp` | `gyre` | partial | partial | 311 | 6 | 2 | 0 | 0 | 0 / 0 | 100 | 0 | 0 | 143 | 68 |
| `gyre/gyre/src/nad/nad_bound_m.fypp` | `gyre` | partial | partial | 893 | 12 | 5 | 3 | 0 | 0 / 0 | 263 | 195 | 0 | 292 | 143 |
| `gyre/gyre/src/nad/nad_bvp_m.fypp` | `gyre` | partial | partial | 231 | 4 | 2 | 1 | 0 | 0 / 0 | 55 | 49 | 0 | 36 | 91 |
| `gyre/gyre/src/nad/nad_diff_m.fypp` | `gyre` | partial | partial | 200 | 2 | 1 | 1 | 0 | 0 / 0 | 57 | 62 | 0 | 0 | 81 |
| `gyre/gyre/src/nad/nad_eqns_m.fypp` | `gyre` | partial | partial | 413 | 4 | 2 | 2 | 0 | 0 / 0 | 79 | 209 | 0 | 0 | 125 |
| `gyre/gyre/src/nad/nad_match_m.fypp` | `gyre` | partial | partial | 275 | 3 | 2 | 1 | 0 | 0 / 0 | 61 | 127 | 0 | 0 | 87 |
| `gyre/gyre/src/nad/nad_trans_m.fypp` | `gyre` | partial | partial | 1,135 | 16 | 1 | 1 | 0 | 0 / 0 | 44 | 34 | 0 | 909 | 148 |
| `gyre/gyre/src/par/grid_par_m.fypp` | `gyre` | partial | ported_unverified | 165 | 1 | 1 | 0 | 0 | 0 / 0 | 101 | 0 | 0 | 0 | 64 |
| `gyre/gyre/src/par/mode_par_m.fypp` | `gyre` | partial | ported_unverified | 117 | 1 | 1 | 0 | 0 | 0 / 0 | 65 | 0 | 0 | 0 | 52 |
| `gyre/gyre/src/par/model_par_m.fypp` | `gyre` | partial | ported_unverified | 192 | 1 | 1 | 0 | 0 | 0 / 0 | 118 | 0 | 0 | 0 | 74 |
| `gyre/gyre/src/par/num_par_m.fypp` | `gyre` | partial | ported_unverified | 146 | 1 | 1 | 0 | 0 | 0 / 0 | 86 | 0 | 0 | 0 | 60 |
| `gyre/gyre/src/par/osc_par_m.fypp` | `gyre` | partial | ported_unverified | 223 | 1 | 1 | 0 | 0 | 0 / 0 | 143 | 0 | 0 | 0 | 80 |
| `gyre/gyre/src/par/rot_par_m.fypp` | `gyre` | partial | ported_unverified | 128 | 1 | 1 | 0 | 0 | 0 / 0 | 72 | 0 | 0 | 0 | 56 |
| `gyre/gyre/src/par/scan_par_m.fypp` | `gyre` | partial | ported_unverified | 151 | 1 | 1 | 0 | 0 | 0 / 0 | 90 | 0 | 0 | 0 | 61 |
| `gyre/gyre/src/rad/rad_bound_m.fypp` | `gyre` | partial | partial | 569 | 10 | 5 | 3 | 0 | 0 / 0 | 172 | 178 | 0 | 91 | 128 |
| `gyre/gyre/src/rad/rad_bvp_m.fypp` | `gyre` | partial | partial | 341 | 4 | 2 | 1 | 0 | 0 / 0 | 144 | 65 | 0 | 38 | 94 |
| `gyre/gyre/src/rad/rad_diff_m.fypp` | `gyre` | partial | ported_unverified | 133 | 2 | 2 | 0 | 0 | 0 / 0 | 60 | 0 | 0 | 0 | 73 |
| `gyre/gyre/src/rad/rad_eqns_m.fypp` | `gyre` | partial | partial | 225 | 4 | 3 | 1 | 0 | 0 / 0 | 102 | 29 | 0 | 0 | 94 |
| `gyre/gyre/src/rad/rad_match_m.fypp` | `gyre` | partial | ported_unverified | 189 | 3 | 3 | 0 | 0 | 0 / 0 | 107 | 0 | 0 | 0 | 82 |
| `gyre/gyre/src/rad/rad_trans_m.fypp` | `gyre` | partial | partial | 614 | 14 | 4 | 7 | 0 | 0 / 0 | 137 | 256 | 0 | 85 | 136 |
| `gyre/gyre/src/rot/null_rot_m.fypp` | `gyre` | partial | ported_unverified | 123 | 3 | 3 | 0 | 0 | 0 / 0 | 46 | 0 | 0 | 0 | 77 |
| `gyre/gyre/src/rot/rot_factory_m.fypp` | `gyre` | partial | partial | 85 | 1 | 0 | 1 | 0 | 0 / 0 | 0 | 22 | 0 | 0 | 63 |
| `gyre/gyre/src/search/bracket_search_m.fypp` | `gyre` | partial | partial | 320 | 4 | 3 | 0 | 0 | 0 / 0 | 227 | 0 | 0 | 30 | 63 |
| `gyre/gyre/src/search/discrim_m.fypp` | `gyre` | partial | ported_unverified | 77 | 1 | 1 | 0 | 0 | 0 / 0 | 18 | 0 | 0 | 0 | 59 |
| `gyre/gyre/src/search/prox_search_m.fypp` | `gyre` | partial | partial | 352 | 4 | 2 | 1 | 0 | 0 / 0 | 87 | 168 | 0 | 31 | 66 |
| `gyre/gyre/src/search/scan_m.fypp` | `gyre` | partial | partial | 336 | 4 | 2 | 1 | 0 | 0 / 0 | 160 | 52 | 0 | 62 | 62 |
| `interp_1d/private/interp_1d_misc.f90` | `interp_1d` | partial | - | 712 | 18 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 653 | 59 |
| `interp_1d/private/interp_1d_mp.f90` | `interp_1d` | partial | - | 388 | 2 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 353 | 35 |
| `interp_1d/private/interp_1d_mp_autodiff.f90` | `interp_1d` | partial | - | 391 | 2 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 355 | 36 |
| `interp_1d/private/interp_1d_pm.f90` | `interp_1d` | partial | - | 458 | 5 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 410 | 48 |
| `interp_1d/private/interp_1d_pm_autodiff.f90` | `interp_1d` | partial | - | 468 | 5 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 419 | 49 |
| `interp_1d/public/interp_1d_def.f90` | `interp_1d` | partial | - | 34 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 34 |
| `interp_1d/public/interp_1d_lib.f90` | `interp_1d` | partial | - | 894 | 41 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 757 | 137 |
| `interp_2d/private/akima760_db.f` | `interp_2d` | partial | - | 1,217 | 5 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 1,209 | 8 |
| `interp_2d/private/bicub_db.f` | `interp_2d` | partial | - | 3,487 | 17 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 3,363 | 124 |
| `interp_2d/private/bipm_db.f90` | `interp_2d` | ported_unverified | kernel_reference_vectors | 120 | 2 | 2 | 0 | 0 | 2 / 0 | 90 | 0 | 0 | 0 | 30 |
| `interp_2d/private/renka790_db.f` | `interp_2d` | partial | - | 1,854 | 9 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 1,852 | 2 |
| `interp_2d/public/interp_2d_lib_db.f90` | `interp_2d` | partial | - | 226 | 10 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 116 | 110 |
| `ionization/private/ion_tables_eval.f90` | `ionization` | partial | - | 576 | 8 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 530 | 46 |
| `ionization/private/ion_tables_load.f90` | `ionization` | partial | - | 556 | 14 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 510 | 46 |
| `ionization/private/mod_ionization.f90` | `ionization` | partial | - | 237 | 8 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 191 | 46 |
| `ionization/public/ionization_def.f90` | `ionization` | partial | - | 172 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 51 | 121 |
| `ionization/public/ionization_lib.f90` | `ionization` | partial | - | 92 | 5 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 50 | 42 |
| `kap/private/condint.f90` | `kap` | partial | partial | 449 | 5 | 4 | 1 | 0 | 0 / 0 | 303 | 97 | 0 | 0 | 49 |
| `kap/private/cubic_interpolator.f90` | `kap` | partial | - | 185 | 5 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 130 | 55 |
| `kap/private/kap_aesopus.f90` | `kap` | partial | partial | 608 | 9 | 4 | 4 | 0 | 0 / 0 | 208 | 340 | 0 | 4 | 56 |
| `kap/private/kap_ctrls_io.f90` | `kap` | partial | partial | 527 | 8 | 0 | 2 | 0 | 0 / 0 | 0 | 169 | 0 | 205 | 153 |
| `kap/private/kap_eval.f90` | `kap` | partial | partial | 874 | 6 | 2 | 4 | 0 | 0 / 0 | 172 | 654 | 0 | 0 | 48 |
| `kap/private/kap_eval_co.f90` | `kap` | partial | partial | 1,241 | 17 | 12 | 5 | 0 | 0 / 0 | 726 | 463 | 0 | 0 | 52 |
| `kap/private/kap_eval_fixed.f90` | `kap` | partial | partial | 762 | 9 | 7 | 2 | 0 | 0 / 0 | 451 | 261 | 0 | 0 | 50 |
| `kap/private/kap_eval_support.f90` | `kap` | partial | partial | 311 | 4 | 1 | 3 | 0 | 0 / 0 | 138 | 132 | 0 | 0 | 41 |
| `kap/private/kapcn.f90` | `kap` | partial | partial | 441 | 10 | 6 | 3 | 0 | 0 / 0 | 189 | 177 | 0 | 4 | 71 |
| `kap/private/load_co_kap.f90` | `kap` | partial | partial | 913 | 15 | 4 | 10 | 0 | 0 / 0 | 48 | 753 | 0 | 53 | 59 |
| `kap/private/load_kap.f90` | `kap` | partial | partial | 765 | 12 | 0 | 11 | 0 | 0 / 0 | 0 | 674 | 0 | 36 | 55 |
| `kap/public/kap_def.f90` | `kap` | partial | partial | 834 | 11 | 1 | 0 | 0 | 0 / 0 | 22 | 0 | 0 | 298 | 514 |
| `kap/public/kap_lib.f90` | `kap` | partial | partial | 868 | 26 | 0 | 5 | 0 | 0 / 0 | 0 | 191 | 0 | 506 | 171 |
| `math/private/math_pown.f90` | `math` | partial | - | 143 | 8 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 64 | 79 |
| `math/public/math_def.f90` | `math` | partial | - | 49 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 49 |
| `math/public/math_lib_intrinsic.f90` | `math` | partial | - | 324 | 16 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 164 | 160 |
| `math/public/precompute_zs.inc` | `math` | partial | - | 52 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 52 | 0 |
| `mtx/private/bcyclic.f90` | `mtx` | partial | - | 603 | 9 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 539 | 64 |
| `mtx/private/mtx_support.f90` | `mtx` | partial | - | 1,099 | 32 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 998 | 101 |
| `mtx/private/my_lapack95.f90` | `mtx` | partial | - | 1,414 | 24 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 1,338 | 76 |
| `mtx/public/mtx_bcyclic_dble_decsol.dek` | `mtx` | partial | - | 45 | 2 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 41 | 4 |
| `mtx/public/mtx_dble_decsol.dek` | `mtx` | partial | - | 264 | 8 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 244 | 20 |
| `mtx/public/mtx_debug_decsol.dek` | `mtx` | partial | - | 68 | 4 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 60 | 8 |
| `mtx/public/mtx_decsol.dek` | `mtx` | partial | - | 21 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 21 | 0 |
| `mtx/public/mtx_decsolblk.dek` | `mtx` | partial | - | 1 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 1 |
| `mtx/public/mtx_decsolblk_dble.dek` | `mtx` | partial | - | 12 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 11 | 1 |
| `mtx/public/mtx_decsolc.dek` | `mtx` | partial | - | 19 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 19 | 0 |
| `mtx/public/mtx_decsolcs.dek` | `mtx` | partial | - | 13 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 12 | 1 |
| `mtx/public/mtx_decsols.dek` | `mtx` | partial | - | 14 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 14 | 0 |
| `mtx/public/mtx_def.f90` | `mtx` | partial | - | 39 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 39 |
| `mtx/public/mtx_formats.dek` | `mtx` | partial | - | 429 | 24 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 378 | 51 |
| `mtx/public/mtx_jac.dek` | `mtx` | partial | - | 137 | 3 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 133 | 4 |
| `mtx/public/mtx_lapack95.dek` | `mtx` | partial | - | 16 | 2 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 15 | 1 |
| `mtx/public/mtx_lib.f90` | `mtx` | partial | - | 340 | 15 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 207 | 133 |
| `mtx/public/mtx_null_decsol.dek` | `mtx` | partial | - | 63 | 5 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 53 | 10 |
| `mtx/public/mtx_rcond.dek` | `mtx` | partial | - | 62 | 2 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 57 | 5 |
| `mtx/public/mtx_solve_routines.inc` | `mtx` | partial | - | 694 | 21 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 650 | 44 |
| `mtx/public/mtx_tridiag_decsol.dek` | `mtx` | partial | - | 19 | 2 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 12 | 7 |
| `net/private/burner_derivs.inc` | `net` | partial | - | 7 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 7 | 0 |
| `net/private/burner_jakob.inc` | `net` | partial | - | 8 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 8 | 0 |
| `net/private/net_approx21.f90` | `net` | partial | partial | 3,374 | 31 | 15 | 11 | 0 | 0 / 0 | 981 | 1,924 | 0 | 203 | 266 |
| `net/private/net_burn.f90` | `net` | partial | partial | 475 | 5 | 1 | 3 | 0 | 0 / 0 | 21 | 378 | 0 | 31 | 45 |
| `net/private/net_burn_const_density.f90` | `net` | partial | partial | 502 | 5 | 2 | 2 | 0 | 0 / 0 | 43 | 384 | 0 | 30 | 45 |
| `net/private/net_burn_const_p.f90` | `net` | partial | partial | 583 | 6 | 0 | 4 | 0 | 0 / 0 | 0 | 502 | 0 | 45 | 36 |
| `net/private/net_burn_support.f90` | `net` | partial | partial | 641 | 4 | 2 | 2 | 0 | 0 / 0 | 279 | 317 | 0 | 0 | 45 |
| `net/private/net_derivs.f90` | `net` | partial | partial | 1,833 | 13 | 0 | 12 | 0 | 0 / 0 | 0 | 1,776 | 0 | 13 | 44 |
| `net/private/net_derivs_support.f90` | `net` | partial | partial | 454 | 7 | 0 | 5 | 0 | 0 / 0 | 0 | 366 | 0 | 19 | 69 |
| `net/private/net_eval.f90` | `net` | partial | partial | 852 | 13 | 3 | 9 | 0 | 0 / 0 | 90 | 688 | 0 | 22 | 52 |
| `net/private/net_initialize.f90` | `net` | partial | partial | 2,300 | 41 | 7 | 25 | 0 | 0 / 0 | 221 | 1,758 | 0 | 227 | 94 |
| `net/private/net_screen.f90` | `net` | partial | partial | 566 | 10 | 5 | 5 | 0 | 0 / 0 | 67 | 458 | 0 | 0 | 41 |
| `net/public/burner_const_density_get_eos_info.inc` | `net` | partial | - | 15 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 15 | 0 |
| `net/public/burner_finish_substep.inc` | `net` | partial | - | 7 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 7 | 0 |
| `net/public/net_def.f90` | `net` | partial | - | 625 | 11 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 276 | 349 |
| `net/public/net_lib.f90` | `net` | partial | partial | 1,210 | 45 | 0 | 6 | 0 | 0 / 0 | 0 | 517 | 0 | 546 | 147 |
| `neu/private/mod_neu.f90` | `neu` | ported_unverified | partial | 1,715 | 12 | 11 | 0 | 0 | 1 / 0 | 1,619 | 0 | 0 | 9 | 87 |
| `neu/public/neu_def.f90` | `neu` | ported_unverified | - | 46 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 46 |
| `neu/public/neu_lib.f90` | `neu` | ported_unverified | kernel_reference_vectors | 71 | 1 | 1 | 0 | 0 | 1 / 0 | 34 | 0 | 0 | 0 | 37 |
| `num/private/decomc.dek` | `num` | partial | - | 429 | 2 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 425 | 4 |
| `num/private/decomr.dek` | `num` | partial | - | 349 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 349 | 0 |
| `num/private/decsol_done.dek` | `num` | partial | - | 138 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 138 | 0 |
| `num/private/mod_bobyqa.f` | `num` | partial | - | 2,243 | 7 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 2,224 | 19 |
| `num/private/mod_brent.f90` | `num` | partial | - | 707 | 3 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 688 | 19 |
| `num/private/mod_dc_decsol.f` | `num` | partial | - | 14 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 14 |
| `num/private/mod_dop853.f` | `num` | partial | - | 836 | 4 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 802 | 34 |
| `num/private/mod_dopri5.f` | `num` | partial | - | 645 | 5 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 606 | 39 |
| `num/private/mod_integrate.f90` | `num` | partial | - | 114 | 2 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 80 | 34 |
| `num/private/mod_newton.f90` | `num` | partial | - | 1,536 | 24 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 1,500 | 36 |
| `num/private/mod_newuoa.f` | `num` | partial | - | 1,583 | 6 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 1,568 | 15 |
| `num/private/mod_qsort.f90` | `num` | partial | - | 160 | 15 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 138 | 22 |
| `num/private/mod_random.f90` | `num` | partial | - | 376 | 4 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 341 | 35 |
| `num/private/mod_root.f90` | `num` | partial | - | 277 | 4 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 244 | 33 |
| `num/private/mod_rosenbrock.f` | `num` | partial | - | 1,862 | 20 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 1,746 | 116 |
| `num/private/mod_simplex.f90` | `num` | partial | - | 386 | 9 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 355 | 31 |
| `num/private/mod_solout_root.f90` | `num` | partial | - | 248 | 4 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 219 | 29 |
| `num/private/qsort_inline.inc` | `num` | partial | - | 194 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 194 |
| `num/private/rodas_args.dek` | `num` | partial | - | 55 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 55 |
| `num/private/slvrod.dek` | `num` | partial | - | 395 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 395 | 0 |
| `num/public/accurate_sum.f90` | `num` | partial | - | 285 | 21 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 183 | 102 |
| `num/public/accurate_sum_auto_diff_star_order1.f90` | `num` | partial | - | 325 | 25 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 215 | 110 |
| `num/public/newton_procs.dek` | `num` | partial | - | 260 | 13 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 138 | 122 |
| `num/public/num_binary_search.dek` | `num` | partial | - | 259 | 3 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 234 | 25 |
| `num/public/num_bobyqa.dek` | `num` | partial | - | 106 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 13 | 93 |
| `num/public/num_bobyqa_proc.dek` | `num` | partial | - | 7 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 7 | 0 |
| `num/public/num_brent.dek` | `num` | partial | - | 225 | 3 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 43 | 182 |
| `num/public/num_def.f90` | `num` | partial | - | 220 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 220 |
| `num/public/num_dfridr.dek` | `num` | partial | - | 55 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 55 | 0 |
| `num/public/num_dfx.dek` | `num` | partial | - | 10 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 10 | 0 |
| `num/public/num_dop853.dek` | `num` | partial | - | 183 | 2 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 133 | 50 |
| `num/public/num_dopri5.dek` | `num` | partial | - | 183 | 2 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 133 | 50 |
| `num/public/num_fcn.dek` | `num` | partial | - | 12 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 12 | 0 |
| `num/public/num_fcn_blk_dble.dek` | `num` | partial | - | 11 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 11 | 0 |
| `num/public/num_interp_y.dek` | `num` | partial | - | 11 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 9 | 2 |
| `num/public/num_isolve.dek` | `num` | partial | - | 357 | 3 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 351 | 6 |
| `num/public/num_jac.dek` | `num` | partial | - | 19 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 19 | 0 |
| `num/public/num_jac_blk_dble.dek` | `num` | partial | - | 12 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 12 | 0 |
| `num/public/num_lib.f90` | `num` | partial | - | 626 | 21 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 256 | 370 |
| `num/public/num_mas.dek` | `num` | partial | - | 8 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 8 | 0 |
| `num/public/num_newton.dek` | `num` | partial | - | 322 | 14 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 283 | 39 |
| `num/public/num_newuoa.dek` | `num` | partial | - | 64 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 13 | 51 |
| `num/public/num_newuoa_proc.dek` | `num` | partial | - | 7 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 7 | 0 |
| `num/public/num_rk2.dek` | `num` | partial | - | 41 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 38 | 3 |
| `num/public/num_rk4.dek` | `num` | partial | - | 53 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 50 | 3 |
| `num/public/num_rodas3.dek` | `num` | partial | - | 347 | 2 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 293 | 54 |
| `num/public/num_rodas4.dek` | `num` | partial | - | 344 | 2 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 293 | 51 |
| `num/public/num_rodasp.dek` | `num` | partial | - | 347 | 2 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 293 | 54 |
| `num/public/num_root_fcn.dek` | `num` | partial | - | 11 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 11 | 0 |
| `num/public/num_ros2.dek` | `num` | partial | - | 341 | 2 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 292 | 49 |
| `num/public/num_ros3p.dek` | `num` | partial | - | 347 | 2 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 292 | 55 |
| `num/public/num_ros3pl.dek` | `num` | partial | - | 347 | 2 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 292 | 55 |
| `num/public/num_rose2.dek` | `num` | partial | - | 344 | 2 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 292 | 52 |
| `num/public/num_safe_root.dek` | `num` | partial | - | 144 | 6 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 129 | 15 |
| `num/public/num_safe_root_x.dek` | `num` | partial | - | 76 | 3 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 65 | 11 |
| `num/public/num_simplex.dek` | `num` | partial | - | 146 | 3 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 114 | 32 |
| `num/public/num_simplex_fcn.dek` | `num` | partial | - | 11 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 10 | 1 |
| `num/public/num_sjac.dek` | `num` | partial | - | 14 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 14 | 0 |
| `num/public/num_solout.dek` | `num` | partial | - | 21 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 21 | 0 |
| `num/public/num_solout_root.dek` | `num` | partial | - | 85 | 3 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 76 | 9 |
| `num/public/num_solout_root_fcn.dek` | `num` | partial | - | 16 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 16 | 0 |
| `num/public/num_solvers_options.dek` | `num` | partial | - | 51 | 2 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 49 | 2 |
| `rates/private/coulomb.f90` | `rates` | partial | partial | 71 | 1 | 0 | 1 | 0 | 0 / 0 | 0 | 37 | 0 | 0 | 34 |
| `rates/private/eval_coulomb.f90` | `rates` | partial | partial | 350 | 9 | 7 | 2 | 0 | 0 / 0 | 243 | 50 | 0 | 0 | 57 |
| `rates/private/eval_ecapture.f90` | `rates` | partial | partial | 334 | 1 | 0 | 1 | 0 | 0 / 0 | 0 | 301 | 0 | 0 | 33 |
| `rates/private/eval_psi.f90` | `rates` | partial | partial | 163 | 3 | 1 | 2 | 0 | 0 / 0 | 13 | 114 | 0 | 0 | 36 |
| `rates/private/eval_weak.f90` | `rates` | partial | partial | 346 | 3 | 0 | 2 | 0 | 0 / 0 | 0 | 283 | 0 | 28 | 35 |
| `rates/private/load_ecapture.f90` | `rates` | partial | partial | 353 | 5 | 0 | 3 | 0 | 0 / 0 | 0 | 301 | 0 | 14 | 38 |
| `rates/private/load_weak.f90` | `rates` | partial | - | 710 | 16 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 668 | 42 |
| `rates/private/pycno.f90` | `rates` | partial | ported_unverified | 308 | 1 | 1 | 0 | 0 | 0 / 0 | 276 | 0 | 0 | 0 | 32 |
| `rates/private/ratelib.f90` | `rates` | partial | partial | 4,619 | 220 | 31 | 75 | 0 | 0 / 0 | 951 | 871 | 0 | 1,953 | 844 |
| `rates/private/rates_initialize.f90` | `rates` | partial | partial | 1,422 | 44 | 2 | 1 | 0 | 0 / 0 | 39 | 30 | 0 | 1,285 | 68 |
| `rates/private/rates_names.f90` | `rates` | partial | - | 368 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 335 | 33 |
| `rates/private/rates_reverses.f90` | `rates` | partial | - | 173 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 142 | 31 |
| `rates/private/rates_support.f90` | `rates` | partial | partial | 706 | 10 | 1 | 3 | 0 | 0 / 0 | 34 | 282 | 0 | 336 | 54 |
| `rates/private/raw_rates.f90` | `rates` | partial | partial | 1,140 | 16 | 1 | 13 | 0 | 0 / 0 | 22 | 1,044 | 0 | 16 | 58 |
| `rates/private/reaclib_eval.f90` | `rates` | partial | partial | 292 | 6 | 1 | 5 | 0 | 0 / 0 | 56 | 189 | 0 | 0 | 47 |
| `rates/private/reaclib_input.f90` | `rates` | partial | - | 670 | 8 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 628 | 42 |
| `rates/private/reaclib_support.f90` | `rates` | partial | partial | 766 | 19 | 0 | 1 | 0 | 0 / 0 | 0 | 92 | 0 | 632 | 42 |
| `rates/private/screen.f90` | `rates` | partial | partial | 108 | 1 | 0 | 1 | 0 | 0 / 0 | 0 | 76 | 0 | 0 | 32 |
| `rates/private/screen5.f90` | `rates` | partial | ported_unverified | 396 | 2 | 2 | 0 | 0 | 0 / 0 | 364 | 0 | 0 | 0 | 32 |
| `rates/private/screen_chugunov.f90` | `rates` | partial | partial | 309 | 1 | 0 | 1 | 0 | 0 / 0 | 0 | 239 | 0 | 0 | 70 |
| `rates/private/suzuki_tables.f90` | `rates` | partial | partial | 553 | 10 | 1 | 3 | 0 | 0 / 0 | 25 | 230 | 0 | 205 | 93 |
| `rates/private/weak_support.f90` | `rates` | partial | - | 85 | 2 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 57 | 28 |
| `rates/private/weaklib_tables.f90` | `rates` | partial | partial | 514 | 12 | 1 | 3 | 0 | 0 / 0 | 25 | 173 | 0 | 258 | 58 |
| `rates/public/rates_def.f90` | `rates` | partial | - | 1,295 | 24 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 377 | 918 |
| `rates/public/rates_lib.f90` | `rates` | partial | partial | 974 | 55 | 4 | 26 | 0 | 0 / 0 | 71 | 414 | 0 | 343 | 146 |
| `star/job/other_extras.f90` | `star` | partial | partial | 207 | 10 | 5 | 4 | 1 | 0 / 0 | 64 | 56 | 24 | 0 | 63 |
| `star/job/run_star.f90` | `star` | ported_unverified | ported_unverified | 48 | 1 | 1 | 0 | 0 | 0 / 0 | 22 | 0 | 0 | 0 | 26 |
| `star/job/run_star_support.f90` | `star` | partial | partial | 3,705 | 49 | 6 | 13 | 14 | 0 / 0 | 220 | 1,823 | 299 | 1,218 | 145 |
| `star/private/adjust_mass.f90` | `star` | partial | partial | 1,624 | 18 | 5 | 8 | 3 | 0 / 0 | 369 | 1,092 | 22 | 72 | 69 |
| `star/private/adjust_mesh.f90` | `star` | ported_unverified | partial | 700 | 14 | 2 | 1 | 0 | 0 / 0 | 65 | 509 | 0 | 89 | 37 |
| `star/private/adjust_mesh_split_merge.f90` | `star` | ported_unverified | partial | 1,461 | 22 | 12 | 6 | 0 | 0 / 0 | 259 | 1,070 | 0 | 58 | 74 |
| `star/private/adjust_mesh_support.f90` | `star` | ported_unverified | partial | 457 | 12 | 8 | 3 | 0 | 0 / 0 | 190 | 207 | 0 | 11 | 49 |
| `star/private/adjust_xyz.f90` | `star` | partial | partial | 1,468 | 31 | 5 | 14 | 4 | 0 / 0 | 276 | 846 | 18 | 233 | 95 |
| `star/private/atm_support.f90` | `star` | partial | partial | 1,289 | 19 | 6 | 11 | 0 | 0 / 0 | 178 | 904 | 0 | 145 | 62 |
| `star/private/auto_diff_support.f90` | `star` | partial | partial | 1,460 | 118 | 22 | 16 | 0 | 0 / 0 | 208 | 195 | 0 | 883 | 174 |
| `star/private/brunt.f90` | `star` | ported_unverified | partial | 335 | 5 | 3 | 2 | 0 | 0 / 0 | 142 | 148 | 0 | 0 | 45 |
| `star/private/conv_premix.f90` | `star` | ported_unverified | partial | 1,561 | 16 | 14 | 0 | 0 | 0 / 0 | 1,399 | 0 | 0 | 49 | 113 |
| `star/private/ctrls_io.f90` | `star` | partial | partial | 4,272 | 10 | 0 | 6 | 0 | 0 / 0 | 0 | 1,886 | 0 | 1,812 | 574 |
| `star/private/diffusion.f90` | `star` | kernel_reference_vectors | partial | 2,050 | 30 | 13 | 5 | 0 | 0 / 1 | 1,126 | 509 | 0 | 345 | 70 |
| `star/private/diffusion_procs.f90` | `star` | partial | partial | 1,746 | 17 | 4 | 8 | 0 | 0 / 0 | 285 | 898 | 0 | 494 | 69 |
| `star/private/diffusion_support.f90` | `star` | kernel_reference_vectors | partial | 1,800 | 16 | 11 | 5 | 0 | 1 / 0 | 805 | 883 | 0 | 0 | 112 |
| `star/private/do_one_utils.f90` | `star` | partial | partial | 1,544 | 16 | 3 | 2 | 7 | 0 / 0 | 901 | 105 | 437 | 34 | 67 |
| `star/private/element_diffusion.f90` | `star` | ported_unverified | partial | 582 | 6 | 3 | 1 | 0 | 0 / 0 | 38 | 414 | 0 | 94 | 36 |
| `star/private/eos_support.f90` | `star` | partial | partial | 413 | 8 | 2 | 5 | 0 | 0 / 0 | 102 | 196 | 0 | 38 | 77 |
| `star/private/eps_grav.f90` | `star` | ported_unverified | partial | 408 | 6 | 5 | 1 | 0 | 0 / 0 | 223 | 138 | 0 | 0 | 47 |
| `star/private/eps_mdot.f90` | `star` | ported_unverified | ported_unverified | 706 | 6 | 6 | 0 | 0 | 1 / 0 | 634 | 0 | 0 | 0 | 72 |
| `star/private/evolve.f90` | `star` | partial | partial | 2,403 | 26 | 2 | 14 | 9 | 0 / 0 | 68 | 1,936 | 315 | 12 | 72 |
| `star/private/evolve_support.f90` | `star` | partial | partial | 305 | 5 | 0 | 2 | 1 | 0 / 0 | 0 | 189 | 19 | 59 | 38 |
| `star/private/history.f90` | `star` | partial | partial | 3,449 | 43 | 17 | 22 | 2 | 0 / 0 | 307 | 2,920 | 43 | 119 | 60 |
| `star/private/history_specs.f90` | `star` | partial | partial | 783 | 15 | 4 | 5 | 4 | 0 / 0 | 55 | 582 | 34 | 35 | 77 |
| `star/private/hydro_alpha_rti_eqns.f90` | `star` | ported_unverified | ported_unverified | 195 | 1 | 1 | 0 | 0 | 0 / 0 | 161 | 0 | 0 | 0 | 34 |
| `star/private/hydro_chem_eqns.f90` | `star` | ported_unverified | partial | 224 | 3 | 2 | 0 | 0 | 0 / 0 | 175 | 0 | 0 | 13 | 36 |
| `star/private/hydro_energy.f90` | `star` | ported_unverified | partial | 879 | 17 | 4 | 11 | 0 | 0 / 0 | 89 | 720 | 0 | 22 | 48 |
| `star/private/hydro_eqns.f90` | `star` | partial | partial | 1,164 | 18 | 12 | 0 | 2 | 0 / 0 | 893 | 0 | 60 | 160 | 51 |
| `star/private/hydro_momentum.f90` | `star` | ported_unverified | partial | 603 | 15 | 13 | 1 | 0 | 0 / 0 | 451 | 83 | 0 | 15 | 54 |
| `star/private/hydro_riemann.f90` | `star` | partial | partial | 459 | 10 | 0 | 10 | 0 | 0 / 0 | 0 | 396 | 0 | 0 | 63 |
| `star/private/hydro_rotation.f90` | `star` | partial | partial | 922 | 26 | 14 | 10 | 0 | 0 / 0 | 302 | 446 | 0 | 83 | 91 |
| `star/private/hydro_rsp2.f90` | `star` | partial | partial | 1,243 | 34 | 21 | 13 | 0 | 0 / 0 | 624 | 511 | 0 | 0 | 108 |
| `star/private/hydro_rsp2_support.f90` | `star` | partial | partial | 445 | 14 | 9 | 5 | 0 | 0 / 0 | 165 | 242 | 0 | 0 | 38 |
| `star/private/hydro_temperature.f90` | `star` | partial | partial | 464 | 6 | 0 | 5 | 1 | 0 / 0 | 0 | 390 | 21 | 0 | 53 |
| `star/private/hydro_vars.f90` | `star` | partial | partial | 1,163 | 19 | 2 | 12 | 3 | 0 / 0 | 26 | 1,008 | 23 | 32 | 74 |
| `star/private/init.f90` | `star` | stub | partial | 1,232 | 27 | 2 | 9 | 0 | 0 / 0 | 16 | 893 | 0 | 221 | 102 |
| `star/private/init_model.f90` | `star` | ported_unverified | partial | 468 | 7 | 1 | 3 | 0 | 1 / 0 | 68 | 324 | 0 | 35 | 41 |
| `star/private/ionization_potentials.f90` | `star` | ported_unverified | - | 596 | 1 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 558 | 38 |
| `star/private/kap_support.f90` | `star` | partial | partial | 526 | 5 | 0 | 1 | 0 | 0 / 0 | 0 | 321 | 0 | 159 | 46 |
| `star/private/magnetic_diffusion.f90` | `star` | partial | ported_unverified | 137 | 5 | 5 | 0 | 0 | 0 / 0 | 62 | 0 | 0 | 0 | 75 |
| `star/private/mass_utils.f90` | `star` | partial | ported_unverified | 594 | 9 | 9 | 0 | 0 | 0 / 0 | 441 | 0 | 0 | 0 | 153 |
| `star/private/mesh_adjust.f90` | `star` | ported_unverified | partial | 2,889 | 39 | 16 | 13 | 0 | 0 / 0 | 1,118 | 1,500 | 0 | 161 | 110 |
| `star/private/mesh_functions.f90` | `star` | ported_unverified | partial | 327 | 6 | 2 | 4 | 0 | 0 / 0 | 213 | 70 | 0 | 0 | 44 |
| `star/private/mesh_plan.f90` | `star` | ported_unverified | partial | 1,301 | 14 | 10 | 0 | 0 | 0 / 0 | 1,097 | 0 | 0 | 158 | 46 |
| `star/private/micro.f90` | `star` | partial | partial | 766 | 12 | 0 | 9 | 3 | 0 / 0 | 0 | 617 | 93 | 0 | 56 |
| `star/private/mix_info.f90` | `star` | partial | partial | 2,438 | 37 | 14 | 16 | 2 | 1 / 0 | 1,295 | 833 | 74 | 136 | 100 |
| `star/private/mod_typical_charge.f90` | `star` | ported_unverified | partial | 111 | 4 | 1 | 0 | 0 | 0 / 0 | 11 | 0 | 0 | 40 | 60 |
| `star/private/net.f90` | `star` | partial | partial | 813 | 10 | 0 | 7 | 1 | 0 / 0 | 0 | 586 | 137 | 31 | 59 |
| `star/private/neu.f90` | `star` | ported_unverified | ported_unverified | 129 | 2 | 2 | 0 | 0 | 1 / 0 | 93 | 0 | 0 | 0 | 36 |
| `star/private/overshoot.f90` | `star` | ported_unverified | ported_unverified | 329 | 1 | 1 | 0 | 0 | 0 / 0 | 292 | 0 | 0 | 0 | 37 |
| `star/private/overshoot_exp.f90` | `star` | ported_unverified | ported_unverified | 176 | 1 | 1 | 0 | 0 | 0 / 0 | 143 | 0 | 0 | 0 | 33 |
| `star/private/overshoot_step.f90` | `star` | ported_unverified | ported_unverified | 162 | 1 | 1 | 0 | 0 | 0 / 0 | 129 | 0 | 0 | 0 | 33 |
| `star/private/overshoot_utils.f90` | `star` | ported_unverified | ported_unverified | 376 | 4 | 4 | 0 | 0 | 2 / 0 | 334 | 0 | 0 | 0 | 42 |
| `star/private/paquette_coeffs.f90` | `star` | kernel_reference_vectors | kernel_reference_vectors | 721 | 3 | 3 | 0 | 0 | 3 / 0 | 656 | 0 | 0 | 0 | 65 |
| `star/private/phase_separation.f90` | `star` | kernel_reference_vectors | partial | 400 | 7 | 6 | 1 | 0 | 0 / 1 | 288 | 66 | 0 | 0 | 46 |
| `star/private/photo_in.f90` | `star` | partial | partial | 320 | 3 | 0 | 1 | 2 | 0 / 0 | 0 | 269 | 19 | 0 | 32 |
| `star/private/photo_out.f90` | `star` | partial | partial | 179 | 2 | 0 | 1 | 1 | 0 / 0 | 0 | 142 | 5 | 0 | 32 |
| `star/private/pre_ms_model.f90` | `star` | ported_unverified | partial | 738 | 6 | 4 | 1 | 0 | 0 / 0 | 614 | 80 | 0 | 3 | 41 |
| `star/private/predictive_mix.f90` | `star` | ported_unverified | ported_unverified | 965 | 6 | 6 | 0 | 0 | 0 / 0 | 919 | 0 | 0 | 0 | 46 |
| `star/private/profile.f90` | `star` | partial | partial | 1,186 | 23 | 3 | 14 | 4 | 0 / 0 | 75 | 984 | 23 | 38 | 66 |
| `star/private/profile_getval.f90` | `star` | partial | partial | 2,466 | 14 | 0 | 7 | 0 | 0 / 0 | 0 | 2,217 | 0 | 177 | 72 |
| `star/private/pulse.f90` | `star` | partial | partial | 185 | 3 | 0 | 3 | 0 | 0 / 0 | 0 | 138 | 0 | 0 | 47 |
| `star/private/pulse_cafein.f90` | `star` | partial | partial | 722 | 6 | 2 | 3 | 0 | 0 / 0 | 99 | 479 | 0 | 102 | 42 |
| `star/private/pulse_fgong.f90` | `star` | partial | partial | 710 | 5 | 3 | 2 | 0 | 0 / 0 | 321 | 339 | 0 | 0 | 50 |
| `star/private/pulse_gr1d.f90` | `star` | partial | partial | 182 | 3 | 0 | 1 | 0 | 0 / 0 | 0 | 51 | 0 | 93 | 38 |
| `star/private/pulse_gyre.f90` | `star` | partial | partial | 476 | 5 | 1 | 3 | 0 | 0 / 0 | 56 | 335 | 0 | 44 | 41 |
| `star/private/pulse_osc.f90` | `star` | partial | partial | 608 | 5 | 0 | 4 | 0 | 0 / 0 | 0 | 458 | 0 | 101 | 49 |
| `star/private/pulse_saio.f90` | `star` | partial | partial | 347 | 4 | 1 | 3 | 0 | 0 / 0 | 48 | 259 | 0 | 0 | 40 |
| `star/private/pulse_utils.f90` | `star` | ported_unverified | ported_unverified | 525 | 9 | 9 | 0 | 0 | 0 / 0 | 466 | 0 | 0 | 0 | 59 |
| `star/private/read_model.f90` | `star` | partial | partial | 986 | 13 | 0 | 9 | 0 | 0 / 0 | 0 | 786 | 0 | 110 | 90 |
| `star/private/relax.f90` | `star` | stub | partial | 4,340 | 128 | 62 | 12 | 0 | 0 / 0 | 1,959 | 814 | 0 | 1,323 | 244 |
| `star/private/report.f90` | `star` | partial | partial | 1,419 | 23 | 14 | 3 | 1 | 0 / 0 | 693 | 406 | 7 | 253 | 60 |
| `star/private/rotation_mix_info.f90` | `star` | partial | partial | 1,371 | 13 | 9 | 3 | 0 | 0 / 0 | 496 | 808 | 0 | 11 | 56 |
| `star/private/rsp.f90` | `star` | ported_unverified | partial | 1,030 | 19 | 9 | 8 | 0 | 0 / 1 | 397 | 511 | 0 | 52 | 70 |
| `star/private/rsp_build.f90` | `star` | ported_unverified | partial | 986 | 18 | 10 | 7 | 0 | 0 / 1 | 545 | 373 | 0 | 15 | 53 |
| `star/private/rsp_def.f90` | `star` | ported_unverified | partial | 863 | 22 | 18 | 1 | 0 | 0 / 0 | 543 | 51 | 0 | 85 | 184 |
| `star/private/rsp_eval_eos_and_kap.f90` | `star` | ported_unverified | ported_unverified | 1,047 | 16 | 16 | 0 | 0 | 0 / 0 | 972 | 0 | 0 | 0 | 75 |
| `star/private/rsp_lina.f90` | `star` | ported_unverified | ported_unverified | 1,269 | 3 | 3 | 0 | 0 | 0 / 0 | 1,225 | 0 | 0 | 0 | 44 |
| `star/private/rsp_relax_env.f90` | `star` | ported_unverified | ported_unverified | 1,335 | 2 | 2 | 0 | 0 | 0 / 0 | 1,296 | 0 | 0 | 0 | 39 |
| `star/private/rsp_step.f90` | `star` | ported_unverified | partial | 3,834 | 53 | 39 | 6 | 0 | 0 / 0 | 2,769 | 457 | 0 | 321 | 287 |
| `star/private/set_flags.f90` | `star` | stub | partial | 806 | 30 | 2 | 8 | 0 | 0 / 0 | 19 | 433 | 0 | 303 | 51 |
| `star/private/solve_omega_mix.f90` | `star` | kernel_reference_vectors | partial | 669 | 7 | 3 | 1 | 0 | 1 / 0 | 534 | 36 | 0 | 66 | 33 |
| `star/private/solver_support.f90` | `star` | ported_unverified | partial | 1,547 | 21 | 9 | 3 | 0 | 0 / 0 | 667 | 667 | 0 | 139 | 74 |
| `star/private/star_bcyclic.f90` | `star` | ported_unverified | partial | 981 | 15 | 1 | 8 | 0 | 0 / 0 | 14 | 738 | 0 | 157 | 72 |
| `star/private/star_job_ctrls_io.f90` | `star` | partial | partial | 1,763 | 9 | 0 | 4 | 0 | 0 / 0 | 0 | 591 | 0 | 620 | 552 |
| `star/private/star_solver.f90` | `star` | ported_unverified | partial | 1,940 | 35 | 4 | 8 | 0 | 0 / 0 | 115 | 890 | 0 | 894 | 41 |
| `star/private/star_utils.f90` | `star` | partial | partial | 4,149 | 156 | 88 | 32 | 5 | 0 / 0 | 1,964 | 1,032 | 131 | 533 | 489 |
| `star/private/struct_burn_mix.f90` | `star` | partial | partial | 1,213 | 19 | 1 | 13 | 2 | 0 / 0 | 6 | 1,045 | 39 | 60 | 63 |
| `star/private/tdc_hydro.f90` | `star` | ported_unverified | partial | 658 | 16 | 7 | 5 | 1 | 0 / 0 | 229 | 267 | 15 | 84 | 63 |
| `star/private/timestep.f90` | `star` | ported_unverified | partial | 2,432 | 64 | 58 | 5 | 0 | 0 / 0 | 1,687 | 574 | 0 | 10 | 161 |
| `star/private/turb_info.f90` | `star` | ported_unverified | partial | 608 | 12 | 6 | 5 | 0 | 0 / 0 | 305 | 240 | 0 | 7 | 56 |
| `star/private/turb_support.f90` | `star` | ported_unverified | partial | 512 | 5 | 2 | 3 | 0 | 0 / 0 | 71 | 387 | 0 | 0 | 54 |
| `star/private/winds.f90` | `star` | ported_unverified | partial | 706 | 14 | 13 | 1 | 0 | 3 / 0 | 646 | 17 | 0 | 0 | 43 |
| `star/private/write_model.f90` | `star` | ported_unverified | partial | 281 | 4 | 1 | 0 | 0 | 0 / 0 | 207 | 0 | 0 | 39 | 35 |
| `star/public/star_def.f90` | `star` | ported_unverified | - | 26 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 26 |
| `star_data/private/star_controls.inc` | `star_data` | partial | - | 1,412 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 1,412 |
| `star_data/private/star_controls_dev.inc` | `star_data` | partial | - | 49 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 49 |
| `star_data/private/star_job_controls.inc` | `star_data` | partial | - | 368 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 368 |
| `star_data/private/star_job_controls_dev.inc` | `star_data` | partial | - | 25 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 25 |
| `star_data/private/star_job_controls_params.inc` | `star_data` | partial | - | 2 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 2 |
| `star_data/public/star_data.inc` | `star_data` | partial | - | 14 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 14 |
| `star_data/public/star_data_def.f90` | `star_data` | partial | - | 161 | 4 | 0 | 0 | 3 | 0 / 0 | 0 | 0 | 23 | 19 | 119 |
| `star_data/public/star_data_def.inc` | `star_data` | partial | - | 1,195 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 1,195 |
| `star_data/public/star_data_procedures.inc` | `star_data` | partial | - | 210 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 210 |
| `star_data/public/star_data_step_input.inc` | `star_data` | partial | - | 208 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 208 |
| `star_data/public/star_data_step_work.inc` | `star_data` | partial | - | 1,192 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 1,192 |
| `turb/private/mlt.f90` | `turb` | ported_unverified | kernel_reference_vectors | 183 | 1 | 1 | 0 | 0 | 1 / 0 | 129 | 0 | 0 | 0 | 54 |
| `turb/private/semiconvection.f90` | `turb` | ported_unverified | kernel_reference_vectors | 131 | 1 | 1 | 0 | 0 | 1 / 0 | 75 | 0 | 0 | 0 | 56 |
| `turb/private/tdc.f90` | `turb` | ported_unverified | partial | 381 | 2 | 1 | 1 | 0 | 0 / 0 | 167 | 150 | 0 | 0 | 64 |
| `turb/private/tdc_support.f90` | `turb` | partial | ported_unverified | 681 | 11 | 11 | 0 | 0 | 0 / 0 | 448 | 0 | 0 | 0 | 233 |
| `turb/private/thermohaline.f90` | `turb` | ported_unverified | partial | 363 | 6 | 5 | 1 | 0 | 0 / 0 | 221 | 86 | 0 | 0 | 56 |
| `turb/public/turb.f90` | `turb` | ported_unverified | kernel_reference_vectors | 294 | 4 | 4 | 0 | 0 | 4 / 0 | 151 | 0 | 0 | 0 | 143 |
| `utils/private/utils_idict.f90` | `utils` | partial | - | 525 | 20 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 472 | 53 |
| `utils/private/utils_nan.f90` | `utils` | partial | - | 32 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 32 |
| `utils/private/utils_nan_dp.f90` | `utils` | partial | - | 270 | 8 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 191 | 79 |
| `utils/public/utils_def.f90` | `utils` | partial | - | 62 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 62 |
| `utils/public/utils_lib.f90` | `utils` | partial | - | 1,189 | 75 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 1,015 | 174 |

## Backlog of unmapped core-engine routines

958 routines in core-engine files with status `stub`, `partial`, or `absent` have no gate-checked `routine_map` entry (26,923 own lines). Partial and not-applicable routines are not in the backlog. Some may already be implemented without a map entry; the table cannot tell.

### How cases are attributed, and its limits

The 34 declared gaps in `known_gaps.toml` are the non-passing cases. A routine is attributed to a case only when that case's gap `reason` names the routine (an identifier equal to its Fortran name, at least eight characters or containing an underscore), basis `routine`, or names its upstream file, basis `file`, which attributes every unmapped routine of that file. A name found in several backlog files counts only when the reason also names one of the files. No other signal is used: the unsupported-controls report and `extras_status.toml` name controls and cases but no routines, so a missing control or an unported extras hook cannot be tied to a routine mechanically. Most gap reasons name neither a routine nor a file, so most routines are unattributed. Unattributed does not mean unneeded, and the case counts are lower bounds. Rows are sorted by number of cases named, then lines.

1 backlog routines are attributed to at least one case.

### Attributed routines

| # | Upstream file | Routine | Cases | Lines | Basis | Cases named |
| ---: | :--- | :--- | ---: | ---: | :--- | :--- |
| 1 | `binary/work/src/run_binary_extras.f90` | `extras_binary_finish_step` | 1 | 11 | routine | `binary/double_bh` |

### Largest unattributed routines

The 40 largest backlog routines with no case named. The full list is in the JSON file.

| # | Upstream file | Routine | Cases | Lines | Basis | Cases named |
| ---: | :--- | :--- | ---: | ---: | :--- | :--- |
| 1 | `star/private/ctrls_io.f90` | `set_controls_for_writing` | 0 | 1,716 | - | - |
| 2 | `star/private/alloc.f90` | `star_info_arrays` | 0 | 981 | - | - |
| 3 | `star/private/star_profile_def.f90` | `profile_column_names_init` | 0 | 710 | - | - |
| 4 | `star/private/star_history_def.f90` | `history_column_names_init` | 0 | 690 | - | - |
| 5 | `star/private/star_job_ctrls_io.f90` | `set_star_job_controls_for_writing` | 0 | 504 | - | - |
| 6 | `star/job/run_star_support.f90` | `do_remove_center` | 0 | 317 | - | - |
| 7 | `star/private/star_private_def.f90` | `star_private_def_init` | 0 | 277 | - | - |
| 8 | `star/private/remove_shells.f90` | `do_relax_to_star_cut` | 0 | 249 | - | - |
| 9 | `binary/private/binary_do_one_utils.f90` | `output_binary_terminal_summary` | 0 | 234 | - | - |
| 10 | `star/private/adjust_net.f90` | `check_adjust_net` | 0 | 210 | - | - |
| 11 | `star/job/run_star_support.f90` | `do_before_evolve_loop` | 0 | 186 | - | - |
| 12 | `star/private/create_initial_model.f90` | `build_initial_model` | 0 | 176 | - | - |
| 13 | `binary/private/binary_ce.f90` | `CE_init` | 0 | 164 | - | - |
| 14 | `star/private/diffusion_procs.f90` | `calc_g_rad_mombarg` | 0 | 161 | - | - |
| 15 | `star/job/gyre_in_mesa_extras_set_velocities.inc` | `gyre_in_mesa_extras_set_velocities` | 0 | 157 | - | - |
| 16 | `star/private/remove_shells.f90` | `do_remove_surface` | 0 | 156 | - | - |
| 17 | `star/private/star_private_def.f90` | `stardata_init` | 0 | 153 | - | - |
| 18 | `star/other/sample_pgstar_plot.f90` | `plot` | 0 | 150 | - | - |
| 19 | `star/private/create_initial_model.f90` | `PSerrfunc` | 0 | 141 | - | - |
| 20 | `star/private/alloc.f90` | `set_var_info` | 0 | 137 | - | - |
| 21 | `star/private/diffusion_procs.f90` | `calc_g_rad_hu` | 0 | 128 | - | - |
| 22 | `star/job/run_star_support.f90` | `do_remove_initial_surface` | 0 | 123 | - | - |
| 23 | `astero/private/adipls_support.f90` | `do_adipls_get_one_el_info` | 0 | 116 | - | - |
| 24 | `astero/private/adipls_support.f90` | `store_amdl` | 0 | 115 | - | - |
| 25 | `binary/private/binary_photos.f90` | `binary_photo_read` | 0 | 115 | - | - |
| 26 | `star/job/multi_stars_extras.inc` | `do_run_multi_stars` | 0 | 111 | - | - |
| 27 | `star/private/atm_support.f90` | `get_legacy` | 0 | 108 | - | - |
| 28 | `star/job/run_star_support.f90` | `do_remove_surface` | 0 | 107 | - | - |
| 29 | `star/private/tdc_hydro_support.f90` | `set_xm_new2` | 0 | 105 | - | - |
| 30 | `astero/private/adipls_support.f90` | `fgong_amdl` | 0 | 104 | - | - |
| 31 | `star/private/pulse_cafein.f90` | `store_point_data_atm` | 0 | 102 | - | - |
| 32 | `star/private/pulse_osc.f90` | `store_point_data_atm` | 0 | 101 | - | - |
| 33 | `binary/private/binary_ce.f90` | `CE_binary_evolve_step` | 0 | 100 | - | - |
| 34 | `star/private/remove_shells.f90` | `do_remove_fallback` | 0 | 100 | - | - |
| 35 | `binary/private/binary_private_def.f90` | `binary_history_column_names_init` | 0 | 99 | - | - |
| 36 | `star/private/history.f90` | `get_int_k_r_dr` | 0 | 97 | - | - |
| 37 | `star/private/pulse_gsm.f90` | `write_gsm_data` | 0 | 97 | - | - |
| 38 | `star/job/gyre_in_mesa_extras_finish_step.inc` | `process_mode_` | 0 | 95 | - | - |
| 39 | `astero/private/adipls_support.f90` | `run_adipls` | 0 | 94 | - | - |
| 40 | `star/job/run_star_support.f90` | `create_merger_model` | 0 | 94 | - | - |
