# Routine-level port table (generated)

<!-- GENERATED VIA scripts/dashboard/build_routine_table.py (#611). Do not edit by hand. -->

Derived from `benchmarks/dashboard/routine_inventory.json` (MESA r26.4.1, commit `c8c2026`) and the `routine_map` entries of `benchmarks/dashboard/port_manifest.toml`. Evidence level: `mapped` only (the `routine_map` names the routine and the target passes the `routine_map` gate). Oracle and exercised levels are not derived here. Machine-readable copy: `notes/audit/routine_table.json`.

39 manifest files are not in the inventory (C sources and STELLA `.trf` files) and are excluded from every figure; none of them is in the core engine modules. Contained procedures are separate routines whose lines also lie inside their parent, so routine counts include both while line figures count each line once.

## Core engine (`star`, `star_data`, `binary`, `astero`)

| Measure | Value |
| :--- | ---: |
| Upstream lines | 144,952 |
| Lines in mapped routines | 25,353 |
| Line-weighted mapped fraction | 17.5% |
| Whole-file ported fraction (existing headline) | 30.7% |
| Mapped fraction of routine lines only | 20.6% |
| Routines (mapped / total) | 249 / 2,588 |

The two fractions answer different questions. The whole-file figure counts every line of a file whose hand-set status is ported; the line-weighted figure counts only the lines of routines that a gate-checked `routine_map` entry names. Neither is a statement about agreement with MESA.

## Per module

| Module | Files | Routines | Mapped | Upstream lines | In routines | In mapped routines | Outside routines | Lines mapped | Whole-file ported |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `star` | 173 | 2,176 | 180 | 120,509 | 107,851 | 22,645 | 12,658 | 18.8% | 35.1% |
| `star_data` | 12 | 5 | 0 | 4,870 | 49 | 0 | 4,821 | 0.0% | 0.0% |
| `binary` | 47 | 218 | 69 | 11,638 | 8,881 | 2,708 | 2,757 | 23.3% | 18.7% |
| `astero` | 14 | 189 | 0 | 7,935 | 6,553 | 0 | 1,382 | 0.0% | 0.0% |
| `eos` | 57 | 326 | 108 | 28,666 | 21,515 | 7,552 | 7,151 | 26.3% | 0.0% |
| `kap` | 38 | 311 | 0 | 22,430 | 19,985 | 0 | 2,445 | 0.0% | 0.0% |
| `net` | 16 | 195 | 0 | 13,452 | 12,219 | 0 | 1,233 | 0.0% | 0.0% |
| `rates` | 26 | 477 | 0 | 17,217 | 14,276 | 0 | 2,941 | 0.0% | 0.0% |
| `neu` | 4 | 15 | 11 | 2,184 | 2,014 | 1,595 | 170 | 73.0% | 83.9% |
| `atm` | 19 | 93 | 0 | 6,853 | 5,350 | 0 | 1,503 | 0.0% | 0.0% |
| `turb` | 6 | 25 | 18 | 2,033 | 1,427 | 980 | 606 | 48.2% | 66.5% |
| `chem` | 7 | 67 | 0 | 4,195 | 3,330 | 0 | 865 | 0.0% | 0.0% |
| `ionization` | 22 | 67 | 0 | 3,859 | 3,055 | 0 | 804 | 0.0% | 0.0% |
| `colors` | 10 | 89 | 0 | 3,823 | 3,155 | 0 | 668 | 0.0% | 0.0% |
| `num` | 66 | 273 | 0 | 20,045 | 17,448 | 0 | 2,597 | 0.0% | 0.0% |
| `mtx` | 23 | 163 | 0 | 5,802 | 5,147 | 0 | 655 | 0.0% | 0.0% |
| `interp_1d` | 11 | 111 | 0 | 4,969 | 4,351 | 0 | 618 | 0.0% | 0.0% |
| `interp_2d` | 10 | 82 | 2 | 13,565 | 13,055 | 90 | 510 | 0.7% | 0.9% |
| `auto_diff` | 10 | 927 | 0 | 14,353 | 10,306 | 0 | 4,047 | 0.0% | 0.0% |
| `math` | 6 | 35 | 0 | 936 | 485 | 0 | 451 | 0.0% | 0.0% |
| `const` | 2 | 2 | 0 | 256 | 48 | 0 | 208 | 0.0% | 100.0% |
| `utils` | 11 | 150 | 0 | 3,398 | 2,614 | 0 | 784 | 0.0% | 0.0% |
| `gyre` | 171 | 983 | 0 | 55,708 | 40,741 | 0 | 14,967 | 0.0% | 0.0% |
| `adipls` | 133 | 221 | 0 | 30,923 | 30,764 | 0 | 159 | 0.0% | 0.0% |
| `stella` | 109 | 238 | 0 | 45,351 | 39,716 | 0 | 5,635 | 0.0% | 0.0% |
| `plotting` | 58 | 559 | 0 | 35,732 | 25,133 | 0 | 10,599 | 0.0% | 0.0% |

## Per file (files with a non-absent status or a mapped routine)

| File | Module | Status | Lines | Routines | Mapped | In routines | In mapped | Outside |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `astero/private/astero_run_support.f90` | `astero` | partial | 1,127 | 22 | 0 | 1,058 | 0 | 69 |
| `astero/private/astero_support.f90` | `astero` | partial | 1,530 | 43 | 0 | 1,415 | 0 | 115 |
| `astero/private/extras_support.f90` | `astero` | partial | 1,238 | 28 | 0 | 1,162 | 0 | 76 |
| `astero/public/astero_def.f90` | `astero` | partial | 1,709 | 31 | 0 | 1,005 | 0 | 704 |
| `astero/public/astero_lib.f90` | `astero` | partial | 146 | 6 | 0 | 88 | 0 | 58 |
| `atm/private/atm_irradiated.f90` | `atm` | partial | 334 | 2 | 0 | 294 | 0 | 40 |
| `atm/private/atm_t_tau_relations.f90` | `atm` | partial | 367 | 11 | 0 | 311 | 0 | 56 |
| `atm/private/atm_t_tau_uniform.f90` | `atm` | partial | 702 | 6 | 0 | 655 | 0 | 47 |
| `atm/private/atm_t_tau_varying.f90` | `atm` | partial | 824 | 9 | 0 | 776 | 0 | 48 |
| `atm/private/atm_table.f90` | `atm` | partial | 447 | 3 | 0 | 407 | 0 | 40 |
| `atm/private/atm_utils.f90` | `atm` | partial | 207 | 6 | 0 | 154 | 0 | 53 |
| `atm/private/e2_pairs.dek` | `atm` | partial | 581 | 0 | 0 | 0 | 0 | 581 |
| `atm/private/table_atm.f90` | `atm` | partial | 684 | 8 | 0 | 644 | 0 | 40 |
| `atm/public/atm_def.f90` | `atm` | partial | 146 | 0 | 0 | 0 | 0 | 146 |
| `atm/public/atm_lib.f90` | `atm` | partial | 93 | 3 | 0 | 16 | 0 | 77 |
| `auto_diff/private/auto_diff_real_15var_order1_module.f90` | `auto_diff` | partial | 1,132 | 101 | 0 | 714 | 0 | 418 |
| `auto_diff/private/auto_diff_real_1var_order1_module.f90` | `auto_diff` | partial | 1,319 | 116 | 0 | 820 | 0 | 499 |
| `auto_diff/private/auto_diff_real_2var_order1_module.f90` | `auto_diff` | partial | 1,545 | 117 | 0 | 1,039 | 0 | 506 |
| `auto_diff/private/auto_diff_real_2var_order2_module.f90` | `auto_diff` | partial | 2,009 | 117 | 0 | 1,500 | 0 | 509 |
| `auto_diff/private/auto_diff_real_2var_order3_module.f90` | `auto_diff` | partial | 3,410 | 117 | 0 | 2,897 | 0 | 513 |
| `auto_diff/private/auto_diff_real_4var_order1_module.f90` | `auto_diff` | partial | 1,751 | 119 | 0 | 1,231 | 0 | 520 |
| `auto_diff/private/auto_diff_real_star_order1_module.f90` | `auto_diff` | partial | 1,307 | 115 | 0 | 814 | 0 | 493 |
| `auto_diff/private/support_functions.f90` | `auto_diff` | partial | 135 | 9 | 0 | 77 | 0 | 58 |
| `auto_diff/public/auto_diff.f90` | `auto_diff` | partial | 30 | 0 | 0 | 0 | 0 | 30 |
| `binary/job/run_binary.f90` | `binary` | ported_unverified | 44 | 1 | 1 | 17 | 17 | 27 |
| `binary/private/binary_ctrls_io.f90` | `binary` | partial | 821 | 9 | 0 | 568 | 0 | 253 |
| `binary/private/binary_disk.f90` | `binary` | ported_unverified | 318 | 7 | 7 | 265 | 265 | 53 |
| `binary/private/binary_edot.f90` | `binary` | ported_unverified | 231 | 4 | 4 | 186 | 186 | 45 |
| `binary/private/binary_evolve.f90` | `binary` | partial | 698 | 9 | 0 | 654 | 0 | 44 |
| `binary/private/binary_history.f90` | `binary` | partial | 880 | 21 | 0 | 835 | 0 | 45 |
| `binary/private/binary_history_specs.f90` | `binary` | partial | 286 | 7 | 0 | 244 | 0 | 42 |
| `binary/private/binary_irradiation.f90` | `binary` | ported_unverified | 62 | 1 | 1 | 28 | 28 | 34 |
| `binary/private/binary_jdot.f90` | `binary` | ported_unverified | 330 | 7 | 7 | 290 | 290 | 40 |
| `binary/private/binary_job_controls.inc` | `binary` | partial | 43 | 0 | 0 | 0 | 0 | 43 |
| `binary/private/binary_job_ctrls_io.f90` | `binary` | partial | 331 | 8 | 0 | 244 | 0 | 87 |
| `binary/private/binary_mdot.f90` | `binary` | ported_unverified | 1,023 | 14 | 12 | 973 | 686 | 50 |
| `binary/private/binary_roche_deformation.f90` | `binary` | partial | 416 | 11 | 7 | 351 | 210 | 65 |
| `binary/private/binary_tides.f90` | `binary` | partial | 461 | 8 | 7 | 414 | 357 | 47 |
| `binary/private/binary_timestep.f90` | `binary` | partial | 325 | 2 | 2 | 289 | 289 | 36 |
| `binary/private/binary_utils.f90` | `binary` | partial | 215 | 9 | 7 | 172 | 130 | 43 |
| `binary/private/binary_wind.f90` | `binary` | ported_unverified | 173 | 3 | 3 | 137 | 137 | 36 |
| `binary/private/run_binary_support.f90` | `binary` | partial | 1,016 | 4 | 0 | 975 | 0 | 41 |
| `binary/public/binary_controls.inc` | `binary` | partial | 190 | 0 | 0 | 0 | 0 | 190 |
| `binary/public/binary_data.inc` | `binary` | partial | 326 | 0 | 0 | 0 | 0 | 326 |
| `binary/public/binary_def.f90` | `binary` | partial | 292 | 4 | 0 | 25 | 0 | 267 |
| `binary/public/binary_lib.f90` | `binary` | partial | 272 | 17 | 11 | 224 | 113 | 48 |
| `chem/private/chem_isos_io.f90` | `chem` | partial | 233 | 4 | 0 | 200 | 0 | 33 |
| `chem/private/lodders_mod.f90` | `chem` | partial | 106 | 2 | 0 | 76 | 0 | 30 |
| `chem/private/nuclide_set_mod.f90` | `chem` | partial | 112 | 3 | 0 | 80 | 0 | 32 |
| `chem/public/chem_def.f90` | `chem` | partial | 2,654 | 22 | 0 | 2,087 | 0 | 567 |
| `chem/public/chem_lib.f90` | `chem` | partial | 717 | 24 | 0 | 569 | 0 | 148 |
| `colors/private/bolometric.f90` | `colors` | partial | 191 | 4 | 0 | 149 | 0 | 42 |
| `colors/private/colors_utils.f90` | `colors` | partial | 1,111 | 25 | 0 | 1,019 | 0 | 92 |
| `colors/private/hermite_interp.f90` | `colors` | partial | 443 | 10 | 0 | 384 | 0 | 59 |
| `colors/private/knn_interp.f90` | `colors` | partial | 438 | 7 | 0 | 385 | 0 | 53 |
| `colors/private/linear_interp.f90` | `colors` | partial | 342 | 4 | 0 | 290 | 0 | 52 |
| `colors/private/synthetic.f90` | `colors` | partial | 230 | 5 | 0 | 183 | 0 | 47 |
| `colors/public/colors_def.f90` | `colors` | partial | 319 | 6 | 0 | 176 | 0 | 143 |
| `colors/public/colors_lib.f90` | `colors` | partial | 321 | 18 | 0 | 246 | 0 | 75 |
| `const/public/const_def.f90` | `const` | ported_unverified | 218 | 1 | 0 | 42 | 0 | 176 |
| `const/public/const_lib.f90` | `const` | ported_unverified | 38 | 1 | 0 | 6 | 0 | 32 |
| `eos/private/eos_blend.f90` | `eos` | partial | 208 | 5 | 2 | 140 | 48 | 68 |
| `eos/private/eos_ctrls_io.f90` | `eos` | partial | 697 | 8 | 0 | 414 | 0 | 283 |
| `eos/private/eos_helm_eval.f90` | `eos` | partial | 361 | 4 | 2 | 319 | 214 | 42 |
| `eos/private/eos_initialize.f90` | `eos` | partial | 75 | 1 | 0 | 48 | 0 | 27 |
| `eos/private/eoscms_eval.f90` | `eos` | partial | 703 | 8 | 4 | 629 | 432 | 74 |
| `eos/private/eosde_eval.f90` | `eos` | partial | 113 | 1 | 1 | 82 | 82 | 31 |
| `eos/private/eosdt_eval.f90` | `eos` | partial | 2,537 | 30 | 10 | 2,415 | 1,426 | 122 |
| `eos/private/eosdt_load_tables.f90` | `eos` | partial | 643 | 12 | 0 | 579 | 0 | 64 |
| `eos/private/eosdt_support.f90` | `eos` | partial | 241 | 2 | 1 | 206 | 141 | 35 |
| `eos/private/eospc_eval.f90` | `eos` | partial | 365 | 4 | 3 | 329 | 329 | 36 |
| `eos/private/eospt_eval.f90` | `eos` | partial | 606 | 8 | 4 | 562 | 454 | 44 |
| `eos/private/gauss_fermi.f90` | `eos` | partial | 1,199 | 11 | 1 | 1,141 | 105 | 58 |
| `eos/private/helm.f90` | `eos` | partial | 350 | 3 | 1 | 309 | 33 | 41 |
| `eos/private/helm_alloc.f90` | `eos` | partial | 468 | 8 | 0 | 432 | 0 | 36 |
| `eos/private/helm_coulomb2.dek` | `eos` | partial | 420 | 0 | 0 | 0 | 0 | 420 |
| `eos/private/helm_declare_local_variables.dek` | `eos` | partial | 341 | 0 | 0 | 0 | 0 | 341 |
| `eos/private/helm_electron_positron.dek` | `eos` | partial | 641 | 0 | 0 | 0 | 0 | 641 |
| `eos/private/helm_gammas.dek` | `eos` | partial | 271 | 0 | 0 | 0 | 0 | 271 |
| `eos/private/helm_ideal_ions.dek` | `eos` | partial | 220 | 0 | 0 | 0 | 0 | 220 |
| `eos/private/helm_initialize_local_variables.dek` | `eos` | partial | 521 | 0 | 0 | 0 | 0 | 521 |
| `eos/private/helm_polynomials.f90` | `eos` | partial | 185 | 22 | 22 | 121 | 121 | 64 |
| `eos/private/helm_print_results.dek` | `eos` | partial | 247 | 0 | 0 | 0 | 0 | 247 |
| `eos/private/helm_radiation.dek` | `eos` | partial | 67 | 0 | 0 | 0 | 0 | 67 |
| `eos/private/helm_store_results.dek` | `eos` | partial | 437 | 0 | 0 | 0 | 0 | 437 |
| `eos/private/helm_sum_totals.dek` | `eos` | partial | 169 | 0 | 0 | 0 | 0 | 169 |
| `eos/private/ideal.f90` | `eos` | partial | 206 | 4 | 2 | 168 | 55 | 38 |
| `eos/private/ion_offset.f90` | `eos` | partial | 84 | 1 | 1 | 43 | 43 | 41 |
| `eos/private/pc_eos.f90` | `eos` | partial | 3,028 | 26 | 22 | 2,790 | 2,745 | 238 |
| `eos/private/pc_support.f90` | `eos` | partial | 1,043 | 13 | 1 | 968 | 74 | 75 |
| `eos/private/skye.f90` | `eos` | partial | 402 | 5 | 3 | 337 | 167 | 65 |
| `eos/private/skye_coulomb.f90` | `eos` | partial | 486 | 7 | 1 | 354 | 72 | 132 |
| `eos/private/skye_coulomb_liquid.f90` | `eos` | partial | 181 | 4 | 4 | 107 | 107 | 74 |
| `eos/private/skye_coulomb_solid.f90` | `eos` | partial | 348 | 8 | 8 | 236 | 236 | 112 |
| `eos/private/skye_ideal.f90` | `eos` | partial | 708 | 4 | 3 | 664 | 42 | 44 |
| `eos/private/skye_thermodynamics.f90` | `eos` | partial | 241 | 3 | 3 | 158 | 158 | 83 |
| `eos/public/eos_def.f90` | `eos` | partial | 642 | 8 | 0 | 210 | 0 | 432 |
| `eos/public/eos_lib.f90` | `eos` | partial | 910 | 29 | 9 | 803 | 468 | 107 |
| `eos/public/helm_def.dek` | `eos` | partial | 646 | 0 | 0 | 0 | 0 | 646 |
| `interp_1d/private/interp_1d_misc.f90` | `interp_1d` | partial | 712 | 18 | 0 | 653 | 0 | 59 |
| `interp_1d/private/interp_1d_mp.f90` | `interp_1d` | partial | 388 | 2 | 0 | 353 | 0 | 35 |
| `interp_1d/private/interp_1d_mp_autodiff.f90` | `interp_1d` | partial | 391 | 2 | 0 | 355 | 0 | 36 |
| `interp_1d/private/interp_1d_pm.f90` | `interp_1d` | partial | 458 | 5 | 0 | 410 | 0 | 48 |
| `interp_1d/private/interp_1d_pm_autodiff.f90` | `interp_1d` | partial | 468 | 5 | 0 | 419 | 0 | 49 |
| `interp_1d/public/interp_1d_def.f90` | `interp_1d` | partial | 34 | 0 | 0 | 0 | 0 | 34 |
| `interp_1d/public/interp_1d_lib.f90` | `interp_1d` | partial | 894 | 41 | 0 | 757 | 0 | 137 |
| `interp_2d/private/akima760_db.f` | `interp_2d` | partial | 1,217 | 5 | 0 | 1,209 | 0 | 8 |
| `interp_2d/private/bicub_db.f` | `interp_2d` | partial | 3,487 | 17 | 0 | 3,363 | 0 | 124 |
| `interp_2d/private/bipm_db.f90` | `interp_2d` | ported_unverified | 120 | 2 | 2 | 90 | 90 | 30 |
| `interp_2d/private/renka790_db.f` | `interp_2d` | partial | 1,854 | 9 | 0 | 1,852 | 0 | 2 |
| `interp_2d/public/interp_2d_lib_db.f90` | `interp_2d` | partial | 226 | 10 | 0 | 116 | 0 | 110 |
| `ionization/private/ion_tables_eval.f90` | `ionization` | partial | 576 | 8 | 0 | 530 | 0 | 46 |
| `ionization/private/ion_tables_load.f90` | `ionization` | partial | 556 | 14 | 0 | 510 | 0 | 46 |
| `ionization/private/mod_ionization.f90` | `ionization` | partial | 237 | 8 | 0 | 191 | 0 | 46 |
| `ionization/public/ionization_def.f90` | `ionization` | partial | 172 | 1 | 0 | 51 | 0 | 121 |
| `ionization/public/ionization_lib.f90` | `ionization` | partial | 92 | 5 | 0 | 50 | 0 | 42 |
| `kap/private/condint.f90` | `kap` | partial | 449 | 5 | 0 | 400 | 0 | 49 |
| `kap/private/cubic_interpolator.f90` | `kap` | partial | 185 | 5 | 0 | 130 | 0 | 55 |
| `kap/private/kap_aesopus.f90` | `kap` | partial | 608 | 9 | 0 | 552 | 0 | 56 |
| `kap/private/kap_ctrls_io.f90` | `kap` | partial | 527 | 8 | 0 | 374 | 0 | 153 |
| `kap/private/kap_eval.f90` | `kap` | partial | 874 | 6 | 0 | 826 | 0 | 48 |
| `kap/private/kap_eval_co.f90` | `kap` | partial | 1,241 | 17 | 0 | 1,189 | 0 | 52 |
| `kap/private/kap_eval_fixed.f90` | `kap` | partial | 762 | 9 | 0 | 712 | 0 | 50 |
| `kap/private/kap_eval_support.f90` | `kap` | partial | 311 | 4 | 0 | 270 | 0 | 41 |
| `kap/private/kapcn.f90` | `kap` | partial | 441 | 10 | 0 | 370 | 0 | 71 |
| `kap/private/load_co_kap.f90` | `kap` | partial | 913 | 15 | 0 | 854 | 0 | 59 |
| `kap/private/load_kap.f90` | `kap` | partial | 765 | 12 | 0 | 710 | 0 | 55 |
| `kap/public/kap_def.f90` | `kap` | partial | 834 | 11 | 0 | 320 | 0 | 514 |
| `kap/public/kap_lib.f90` | `kap` | partial | 868 | 26 | 0 | 697 | 0 | 171 |
| `math/private/math_pown.f90` | `math` | partial | 143 | 8 | 0 | 64 | 0 | 79 |
| `math/public/math_def.f90` | `math` | partial | 49 | 0 | 0 | 0 | 0 | 49 |
| `math/public/math_lib_intrinsic.f90` | `math` | partial | 324 | 16 | 0 | 164 | 0 | 160 |
| `math/public/precompute_zs.inc` | `math` | partial | 52 | 1 | 0 | 52 | 0 | 0 |
| `mtx/private/bcyclic.f90` | `mtx` | partial | 603 | 9 | 0 | 539 | 0 | 64 |
| `mtx/private/mtx_support.f90` | `mtx` | partial | 1,099 | 32 | 0 | 998 | 0 | 101 |
| `mtx/private/my_lapack95.f90` | `mtx` | partial | 1,414 | 24 | 0 | 1,338 | 0 | 76 |
| `mtx/public/mtx_bcyclic_dble_decsol.dek` | `mtx` | partial | 45 | 2 | 0 | 41 | 0 | 4 |
| `mtx/public/mtx_dble_decsol.dek` | `mtx` | partial | 264 | 8 | 0 | 244 | 0 | 20 |
| `mtx/public/mtx_debug_decsol.dek` | `mtx` | partial | 68 | 4 | 0 | 60 | 0 | 8 |
| `mtx/public/mtx_decsol.dek` | `mtx` | partial | 21 | 1 | 0 | 21 | 0 | 0 |
| `mtx/public/mtx_decsolblk.dek` | `mtx` | partial | 1 | 0 | 0 | 0 | 0 | 1 |
| `mtx/public/mtx_decsolblk_dble.dek` | `mtx` | partial | 12 | 1 | 0 | 11 | 0 | 1 |
| `mtx/public/mtx_decsolc.dek` | `mtx` | partial | 19 | 1 | 0 | 19 | 0 | 0 |
| `mtx/public/mtx_decsolcs.dek` | `mtx` | partial | 13 | 1 | 0 | 12 | 0 | 1 |
| `mtx/public/mtx_decsols.dek` | `mtx` | partial | 14 | 1 | 0 | 14 | 0 | 0 |
| `mtx/public/mtx_def.f90` | `mtx` | partial | 39 | 0 | 0 | 0 | 0 | 39 |
| `mtx/public/mtx_formats.dek` | `mtx` | partial | 429 | 24 | 0 | 378 | 0 | 51 |
| `mtx/public/mtx_jac.dek` | `mtx` | partial | 137 | 3 | 0 | 133 | 0 | 4 |
| `mtx/public/mtx_lapack95.dek` | `mtx` | partial | 16 | 2 | 0 | 15 | 0 | 1 |
| `mtx/public/mtx_lib.f90` | `mtx` | partial | 340 | 15 | 0 | 207 | 0 | 133 |
| `mtx/public/mtx_null_decsol.dek` | `mtx` | partial | 63 | 5 | 0 | 53 | 0 | 10 |
| `mtx/public/mtx_rcond.dek` | `mtx` | partial | 62 | 2 | 0 | 57 | 0 | 5 |
| `mtx/public/mtx_solve_routines.inc` | `mtx` | partial | 694 | 21 | 0 | 650 | 0 | 44 |
| `mtx/public/mtx_tridiag_decsol.dek` | `mtx` | partial | 19 | 2 | 0 | 12 | 0 | 7 |
| `net/private/burner_derivs.inc` | `net` | partial | 7 | 1 | 0 | 7 | 0 | 0 |
| `net/private/burner_jakob.inc` | `net` | partial | 8 | 1 | 0 | 8 | 0 | 0 |
| `net/private/net_approx21.f90` | `net` | partial | 3,374 | 31 | 0 | 3,108 | 0 | 266 |
| `net/private/net_burn.f90` | `net` | partial | 475 | 5 | 0 | 430 | 0 | 45 |
| `net/private/net_burn_const_density.f90` | `net` | partial | 502 | 5 | 0 | 457 | 0 | 45 |
| `net/private/net_burn_const_p.f90` | `net` | partial | 583 | 6 | 0 | 547 | 0 | 36 |
| `net/private/net_burn_support.f90` | `net` | partial | 641 | 4 | 0 | 596 | 0 | 45 |
| `net/private/net_derivs.f90` | `net` | partial | 1,833 | 13 | 0 | 1,789 | 0 | 44 |
| `net/private/net_derivs_support.f90` | `net` | partial | 454 | 7 | 0 | 385 | 0 | 69 |
| `net/private/net_eval.f90` | `net` | partial | 852 | 13 | 0 | 800 | 0 | 52 |
| `net/private/net_initialize.f90` | `net` | partial | 2,300 | 41 | 0 | 2,206 | 0 | 94 |
| `net/private/net_screen.f90` | `net` | partial | 566 | 10 | 0 | 525 | 0 | 41 |
| `net/public/burner_const_density_get_eos_info.inc` | `net` | partial | 15 | 1 | 0 | 15 | 0 | 0 |
| `net/public/burner_finish_substep.inc` | `net` | partial | 7 | 1 | 0 | 7 | 0 | 0 |
| `net/public/net_def.f90` | `net` | partial | 625 | 11 | 0 | 276 | 0 | 349 |
| `net/public/net_lib.f90` | `net` | partial | 1,210 | 45 | 0 | 1,063 | 0 | 147 |
| `neu/private/mod_neu.f90` | `neu` | ported_unverified | 1,715 | 12 | 10 | 1,628 | 1,561 | 87 |
| `neu/public/neu_def.f90` | `neu` | ported_unverified | 46 | 0 | 0 | 0 | 0 | 46 |
| `neu/public/neu_lib.f90` | `neu` | ported_unverified | 71 | 1 | 1 | 34 | 34 | 37 |
| `num/private/decomc.dek` | `num` | partial | 429 | 2 | 0 | 425 | 0 | 4 |
| `num/private/decomr.dek` | `num` | partial | 349 | 1 | 0 | 349 | 0 | 0 |
| `num/private/decsol_done.dek` | `num` | partial | 138 | 1 | 0 | 138 | 0 | 0 |
| `num/private/mod_bobyqa.f` | `num` | partial | 2,243 | 7 | 0 | 2,224 | 0 | 19 |
| `num/private/mod_brent.f90` | `num` | partial | 707 | 3 | 0 | 688 | 0 | 19 |
| `num/private/mod_dc_decsol.f` | `num` | partial | 14 | 0 | 0 | 0 | 0 | 14 |
| `num/private/mod_dop853.f` | `num` | partial | 836 | 4 | 0 | 802 | 0 | 34 |
| `num/private/mod_dopri5.f` | `num` | partial | 645 | 5 | 0 | 606 | 0 | 39 |
| `num/private/mod_integrate.f90` | `num` | partial | 114 | 2 | 0 | 80 | 0 | 34 |
| `num/private/mod_newton.f90` | `num` | partial | 1,536 | 24 | 0 | 1,500 | 0 | 36 |
| `num/private/mod_newuoa.f` | `num` | partial | 1,583 | 6 | 0 | 1,568 | 0 | 15 |
| `num/private/mod_qsort.f90` | `num` | partial | 160 | 15 | 0 | 138 | 0 | 22 |
| `num/private/mod_random.f90` | `num` | partial | 376 | 4 | 0 | 341 | 0 | 35 |
| `num/private/mod_root.f90` | `num` | partial | 277 | 4 | 0 | 244 | 0 | 33 |
| `num/private/mod_rosenbrock.f` | `num` | partial | 1,862 | 20 | 0 | 1,746 | 0 | 116 |
| `num/private/mod_simplex.f90` | `num` | partial | 386 | 9 | 0 | 355 | 0 | 31 |
| `num/private/mod_solout_root.f90` | `num` | partial | 248 | 4 | 0 | 219 | 0 | 29 |
| `num/private/qsort_inline.inc` | `num` | partial | 194 | 0 | 0 | 0 | 0 | 194 |
| `num/private/rodas_args.dek` | `num` | partial | 55 | 0 | 0 | 0 | 0 | 55 |
| `num/private/slvrod.dek` | `num` | partial | 395 | 1 | 0 | 395 | 0 | 0 |
| `num/public/accurate_sum.f90` | `num` | partial | 285 | 21 | 0 | 183 | 0 | 102 |
| `num/public/accurate_sum_auto_diff_star_order1.f90` | `num` | partial | 325 | 25 | 0 | 215 | 0 | 110 |
| `num/public/newton_procs.dek` | `num` | partial | 260 | 13 | 0 | 138 | 0 | 122 |
| `num/public/num_binary_search.dek` | `num` | partial | 259 | 3 | 0 | 234 | 0 | 25 |
| `num/public/num_bobyqa.dek` | `num` | partial | 106 | 1 | 0 | 13 | 0 | 93 |
| `num/public/num_bobyqa_proc.dek` | `num` | partial | 7 | 1 | 0 | 7 | 0 | 0 |
| `num/public/num_brent.dek` | `num` | partial | 225 | 3 | 0 | 43 | 0 | 182 |
| `num/public/num_def.f90` | `num` | partial | 220 | 0 | 0 | 0 | 0 | 220 |
| `num/public/num_dfridr.dek` | `num` | partial | 55 | 1 | 0 | 55 | 0 | 0 |
| `num/public/num_dfx.dek` | `num` | partial | 10 | 1 | 0 | 10 | 0 | 0 |
| `num/public/num_dop853.dek` | `num` | partial | 183 | 2 | 0 | 133 | 0 | 50 |
| `num/public/num_dopri5.dek` | `num` | partial | 183 | 2 | 0 | 133 | 0 | 50 |
| `num/public/num_fcn.dek` | `num` | partial | 12 | 1 | 0 | 12 | 0 | 0 |
| `num/public/num_fcn_blk_dble.dek` | `num` | partial | 11 | 1 | 0 | 11 | 0 | 0 |
| `num/public/num_interp_y.dek` | `num` | partial | 11 | 1 | 0 | 9 | 0 | 2 |
| `num/public/num_isolve.dek` | `num` | partial | 357 | 3 | 0 | 351 | 0 | 6 |
| `num/public/num_jac.dek` | `num` | partial | 19 | 1 | 0 | 19 | 0 | 0 |
| `num/public/num_jac_blk_dble.dek` | `num` | partial | 12 | 1 | 0 | 12 | 0 | 0 |
| `num/public/num_lib.f90` | `num` | partial | 626 | 21 | 0 | 256 | 0 | 370 |
| `num/public/num_mas.dek` | `num` | partial | 8 | 1 | 0 | 8 | 0 | 0 |
| `num/public/num_newton.dek` | `num` | partial | 322 | 14 | 0 | 283 | 0 | 39 |
| `num/public/num_newuoa.dek` | `num` | partial | 64 | 1 | 0 | 13 | 0 | 51 |
| `num/public/num_newuoa_proc.dek` | `num` | partial | 7 | 1 | 0 | 7 | 0 | 0 |
| `num/public/num_rk2.dek` | `num` | partial | 41 | 1 | 0 | 38 | 0 | 3 |
| `num/public/num_rk4.dek` | `num` | partial | 53 | 1 | 0 | 50 | 0 | 3 |
| `num/public/num_rodas3.dek` | `num` | partial | 347 | 2 | 0 | 293 | 0 | 54 |
| `num/public/num_rodas4.dek` | `num` | partial | 344 | 2 | 0 | 293 | 0 | 51 |
| `num/public/num_rodasp.dek` | `num` | partial | 347 | 2 | 0 | 293 | 0 | 54 |
| `num/public/num_root_fcn.dek` | `num` | partial | 11 | 1 | 0 | 11 | 0 | 0 |
| `num/public/num_ros2.dek` | `num` | partial | 341 | 2 | 0 | 292 | 0 | 49 |
| `num/public/num_ros3p.dek` | `num` | partial | 347 | 2 | 0 | 292 | 0 | 55 |
| `num/public/num_ros3pl.dek` | `num` | partial | 347 | 2 | 0 | 292 | 0 | 55 |
| `num/public/num_rose2.dek` | `num` | partial | 344 | 2 | 0 | 292 | 0 | 52 |
| `num/public/num_safe_root.dek` | `num` | partial | 144 | 6 | 0 | 129 | 0 | 15 |
| `num/public/num_safe_root_x.dek` | `num` | partial | 76 | 3 | 0 | 65 | 0 | 11 |
| `num/public/num_simplex.dek` | `num` | partial | 146 | 3 | 0 | 114 | 0 | 32 |
| `num/public/num_simplex_fcn.dek` | `num` | partial | 11 | 1 | 0 | 10 | 0 | 1 |
| `num/public/num_sjac.dek` | `num` | partial | 14 | 1 | 0 | 14 | 0 | 0 |
| `num/public/num_solout.dek` | `num` | partial | 21 | 1 | 0 | 21 | 0 | 0 |
| `num/public/num_solout_root.dek` | `num` | partial | 85 | 3 | 0 | 76 | 0 | 9 |
| `num/public/num_solout_root_fcn.dek` | `num` | partial | 16 | 1 | 0 | 16 | 0 | 0 |
| `num/public/num_solvers_options.dek` | `num` | partial | 51 | 2 | 0 | 49 | 0 | 2 |
| `rates/private/coulomb.f90` | `rates` | partial | 71 | 1 | 0 | 37 | 0 | 34 |
| `rates/private/eval_coulomb.f90` | `rates` | partial | 350 | 9 | 0 | 293 | 0 | 57 |
| `rates/private/eval_ecapture.f90` | `rates` | partial | 334 | 1 | 0 | 301 | 0 | 33 |
| `rates/private/eval_psi.f90` | `rates` | partial | 163 | 3 | 0 | 127 | 0 | 36 |
| `rates/private/eval_weak.f90` | `rates` | partial | 346 | 3 | 0 | 311 | 0 | 35 |
| `rates/private/load_ecapture.f90` | `rates` | partial | 353 | 5 | 0 | 315 | 0 | 38 |
| `rates/private/load_weak.f90` | `rates` | partial | 710 | 16 | 0 | 668 | 0 | 42 |
| `rates/private/pycno.f90` | `rates` | partial | 308 | 1 | 0 | 276 | 0 | 32 |
| `rates/private/ratelib.f90` | `rates` | partial | 4,619 | 220 | 0 | 3,775 | 0 | 844 |
| `rates/private/rates_initialize.f90` | `rates` | partial | 1,422 | 44 | 0 | 1,354 | 0 | 68 |
| `rates/private/rates_names.f90` | `rates` | partial | 368 | 1 | 0 | 335 | 0 | 33 |
| `rates/private/rates_reverses.f90` | `rates` | partial | 173 | 1 | 0 | 142 | 0 | 31 |
| `rates/private/rates_support.f90` | `rates` | partial | 706 | 10 | 0 | 652 | 0 | 54 |
| `rates/private/raw_rates.f90` | `rates` | partial | 1,140 | 16 | 0 | 1,082 | 0 | 58 |
| `rates/private/reaclib_eval.f90` | `rates` | partial | 292 | 6 | 0 | 245 | 0 | 47 |
| `rates/private/reaclib_input.f90` | `rates` | partial | 670 | 8 | 0 | 628 | 0 | 42 |
| `rates/private/reaclib_support.f90` | `rates` | partial | 766 | 19 | 0 | 724 | 0 | 42 |
| `rates/private/screen.f90` | `rates` | partial | 108 | 1 | 0 | 76 | 0 | 32 |
| `rates/private/screen5.f90` | `rates` | partial | 396 | 2 | 0 | 364 | 0 | 32 |
| `rates/private/screen_chugunov.f90` | `rates` | partial | 309 | 1 | 0 | 239 | 0 | 70 |
| `rates/private/suzuki_tables.f90` | `rates` | partial | 553 | 10 | 0 | 460 | 0 | 93 |
| `rates/private/weak_support.f90` | `rates` | partial | 85 | 2 | 0 | 57 | 0 | 28 |
| `rates/private/weaklib_tables.f90` | `rates` | partial | 514 | 12 | 0 | 456 | 0 | 58 |
| `rates/public/rates_def.f90` | `rates` | partial | 1,295 | 24 | 0 | 377 | 0 | 918 |
| `rates/public/rates_lib.f90` | `rates` | partial | 974 | 55 | 0 | 828 | 0 | 146 |
| `star/job/other_extras.f90` | `star` | partial | 207 | 10 | 0 | 144 | 0 | 63 |
| `star/job/run_star.f90` | `star` | ported_unverified | 48 | 1 | 1 | 22 | 22 | 26 |
| `star/job/run_star_support.f90` | `star` | partial | 3,705 | 49 | 0 | 3,560 | 0 | 145 |
| `star/private/adjust_mass.f90` | `star` | partial | 1,624 | 18 | 3 | 1,555 | 504 | 69 |
| `star/private/adjust_mesh.f90` | `star` | ported_unverified | 700 | 14 | 1 | 663 | 663 | 37 |
| `star/private/adjust_mesh_split_merge.f90` | `star` | ported_unverified | 1,461 | 22 | 1 | 1,387 | 44 | 74 |
| `star/private/adjust_mesh_support.f90` | `star` | ported_unverified | 457 | 12 | 2 | 408 | 305 | 49 |
| `star/private/adjust_xyz.f90` | `star` | partial | 1,468 | 31 | 0 | 1,373 | 0 | 95 |
| `star/private/atm_support.f90` | `star` | partial | 1,289 | 19 | 0 | 1,227 | 0 | 62 |
| `star/private/auto_diff_support.f90` | `star` | partial | 1,460 | 118 | 0 | 1,286 | 0 | 174 |
| `star/private/brunt.f90` | `star` | ported_unverified | 335 | 5 | 2 | 290 | 142 | 45 |
| `star/private/conv_premix.f90` | `star` | ported_unverified | 1,561 | 16 | 1 | 1,448 | 101 | 113 |
| `star/private/ctrls_io.f90` | `star` | partial | 4,272 | 10 | 0 | 3,698 | 0 | 574 |
| `star/private/diffusion.f90` | `star` | kernel_reference_vectors | 2,050 | 30 | 2 | 1,980 | 1,196 | 70 |
| `star/private/diffusion_procs.f90` | `star` | partial | 1,746 | 17 | 0 | 1,677 | 0 | 69 |
| `star/private/diffusion_support.f90` | `star` | kernel_reference_vectors | 1,800 | 16 | 1 | 1,688 | 201 | 112 |
| `star/private/do_one_utils.f90` | `star` | partial | 1,544 | 16 | 1 | 1,477 | 901 | 67 |
| `star/private/element_diffusion.f90` | `star` | ported_unverified | 582 | 6 | 2 | 546 | 546 | 36 |
| `star/private/eos_support.f90` | `star` | partial | 413 | 8 | 0 | 336 | 0 | 77 |
| `star/private/eps_grav.f90` | `star` | ported_unverified | 408 | 6 | 2 | 361 | 40 | 47 |
| `star/private/eps_mdot.f90` | `star` | ported_unverified | 706 | 6 | 6 | 634 | 634 | 72 |
| `star/private/evolve.f90` | `star` | partial | 2,403 | 26 | 0 | 2,331 | 0 | 72 |
| `star/private/evolve_support.f90` | `star` | partial | 305 | 5 | 0 | 267 | 0 | 38 |
| `star/private/history.f90` | `star` | partial | 3,449 | 43 | 0 | 3,389 | 0 | 60 |
| `star/private/history_specs.f90` | `star` | partial | 783 | 15 | 0 | 706 | 0 | 77 |
| `star/private/hydro_alpha_rti_eqns.f90` | `star` | ported_unverified | 195 | 1 | 1 | 161 | 161 | 34 |
| `star/private/hydro_chem_eqns.f90` | `star` | ported_unverified | 224 | 3 | 2 | 188 | 188 | 36 |
| `star/private/hydro_energy.f90` | `star` | ported_unverified | 879 | 17 | 1 | 831 | 19 | 48 |
| `star/private/hydro_eqns.f90` | `star` | partial | 1,164 | 18 | 0 | 1,113 | 0 | 51 |
| `star/private/hydro_momentum.f90` | `star` | ported_unverified | 603 | 15 | 4 | 549 | 127 | 54 |
| `star/private/hydro_riemann.f90` | `star` | partial | 459 | 10 | 0 | 396 | 0 | 63 |
| `star/private/hydro_rotation.f90` | `star` | partial | 922 | 26 | 0 | 831 | 0 | 91 |
| `star/private/hydro_rsp2.f90` | `star` | partial | 1,243 | 34 | 0 | 1,135 | 0 | 108 |
| `star/private/hydro_rsp2_support.f90` | `star` | partial | 445 | 14 | 0 | 407 | 0 | 38 |
| `star/private/hydro_temperature.f90` | `star` | partial | 464 | 6 | 0 | 411 | 0 | 53 |
| `star/private/hydro_vars.f90` | `star` | partial | 1,163 | 19 | 0 | 1,089 | 0 | 74 |
| `star/private/init.f90` | `star` | stub | 1,232 | 27 | 0 | 1,130 | 0 | 102 |
| `star/private/init_model.f90` | `star` | ported_unverified | 468 | 7 | 1 | 427 | 71 | 41 |
| `star/private/ionization_potentials.f90` | `star` | ported_unverified | 596 | 1 | 0 | 558 | 0 | 38 |
| `star/private/kap_support.f90` | `star` | partial | 526 | 5 | 0 | 480 | 0 | 46 |
| `star/private/magnetic_diffusion.f90` | `star` | partial | 137 | 5 | 0 | 62 | 0 | 75 |
| `star/private/mass_utils.f90` | `star` | partial | 594 | 9 | 9 | 441 | 441 | 153 |
| `star/private/mesh_adjust.f90` | `star` | ported_unverified | 2,889 | 39 | 4 | 2,779 | 668 | 110 |
| `star/private/mesh_functions.f90` | `star` | ported_unverified | 327 | 6 | 2 | 283 | 259 | 44 |
| `star/private/mesh_plan.f90` | `star` | ported_unverified | 1,301 | 14 | 1 | 1,255 | 948 | 46 |
| `star/private/micro.f90` | `star` | partial | 766 | 12 | 0 | 710 | 0 | 56 |
| `star/private/mix_info.f90` | `star` | partial | 2,438 | 37 | 7 | 2,338 | 1,301 | 100 |
| `star/private/mod_typical_charge.f90` | `star` | ported_unverified | 111 | 4 | 1 | 51 | 11 | 60 |
| `star/private/net.f90` | `star` | partial | 813 | 10 | 0 | 754 | 0 | 59 |
| `star/private/neu.f90` | `star` | ported_unverified | 129 | 2 | 2 | 93 | 93 | 36 |
| `star/private/overshoot.f90` | `star` | ported_unverified | 329 | 1 | 1 | 292 | 292 | 37 |
| `star/private/overshoot_exp.f90` | `star` | ported_unverified | 176 | 1 | 1 | 143 | 143 | 33 |
| `star/private/overshoot_step.f90` | `star` | ported_unverified | 162 | 1 | 1 | 129 | 129 | 33 |
| `star/private/overshoot_utils.f90` | `star` | ported_unverified | 376 | 4 | 4 | 334 | 334 | 42 |
| `star/private/paquette_coeffs.f90` | `star` | kernel_reference_vectors | 721 | 3 | 3 | 656 | 656 | 65 |
| `star/private/phase_separation.f90` | `star` | kernel_reference_vectors | 400 | 7 | 1 | 354 | 14 | 46 |
| `star/private/photo_in.f90` | `star` | partial | 320 | 3 | 0 | 288 | 0 | 32 |
| `star/private/photo_out.f90` | `star` | partial | 179 | 2 | 0 | 147 | 0 | 32 |
| `star/private/pre_ms_model.f90` | `star` | ported_unverified | 738 | 6 | 1 | 697 | 255 | 41 |
| `star/private/predictive_mix.f90` | `star` | ported_unverified | 965 | 6 | 1 | 919 | 159 | 46 |
| `star/private/profile.f90` | `star` | partial | 1,186 | 23 | 0 | 1,120 | 0 | 66 |
| `star/private/profile_getval.f90` | `star` | partial | 2,466 | 14 | 0 | 2,394 | 0 | 72 |
| `star/private/pulse.f90` | `star` | partial | 185 | 3 | 0 | 138 | 0 | 47 |
| `star/private/pulse_cafein.f90` | `star` | partial | 722 | 6 | 0 | 680 | 0 | 42 |
| `star/private/pulse_fgong.f90` | `star` | partial | 710 | 5 | 0 | 660 | 0 | 50 |
| `star/private/pulse_gr1d.f90` | `star` | partial | 182 | 3 | 0 | 144 | 0 | 38 |
| `star/private/pulse_gyre.f90` | `star` | partial | 476 | 5 | 0 | 435 | 0 | 41 |
| `star/private/pulse_osc.f90` | `star` | partial | 608 | 5 | 0 | 559 | 0 | 49 |
| `star/private/pulse_saio.f90` | `star` | partial | 347 | 4 | 0 | 307 | 0 | 40 |
| `star/private/pulse_utils.f90` | `star` | ported_unverified | 525 | 9 | 9 | 466 | 466 | 59 |
| `star/private/read_model.f90` | `star` | partial | 986 | 13 | 0 | 896 | 0 | 90 |
| `star/private/relax.f90` | `star` | stub | 4,340 | 128 | 0 | 4,096 | 0 | 244 |
| `star/private/report.f90` | `star` | partial | 1,419 | 23 | 0 | 1,359 | 0 | 60 |
| `star/private/rotation_mix_info.f90` | `star` | partial | 1,371 | 13 | 0 | 1,315 | 0 | 56 |
| `star/private/rsp.f90` | `star` | ported_unverified | 1,030 | 19 | 6 | 960 | 406 | 70 |
| `star/private/rsp_build.f90` | `star` | ported_unverified | 986 | 18 | 1 | 933 | 253 | 53 |
| `star/private/rsp_def.f90` | `star` | ported_unverified | 863 | 22 | 17 | 679 | 497 | 184 |
| `star/private/rsp_eval_eos_and_kap.f90` | `star` | ported_unverified | 1,047 | 16 | 14 | 972 | 860 | 75 |
| `star/private/rsp_lina.f90` | `star` | ported_unverified | 1,269 | 3 | 3 | 1,225 | 1,225 | 44 |
| `star/private/rsp_relax_env.f90` | `star` | ported_unverified | 1,335 | 2 | 2 | 1,296 | 1,296 | 39 |
| `star/private/rsp_step.f90` | `star` | ported_unverified | 3,834 | 53 | 27 | 3,547 | 2,393 | 287 |
| `star/private/set_flags.f90` | `star` | stub | 806 | 30 | 0 | 755 | 0 | 51 |
| `star/private/solve_omega_mix.f90` | `star` | kernel_reference_vectors | 669 | 7 | 1 | 636 | 636 | 33 |
| `star/private/solver_support.f90` | `star` | ported_unverified | 1,547 | 21 | 7 | 1,473 | 736 | 74 |
| `star/private/star_bcyclic.f90` | `star` | ported_unverified | 981 | 15 | 3 | 909 | 290 | 72 |
| `star/private/star_job_ctrls_io.f90` | `star` | partial | 1,763 | 9 | 0 | 1,211 | 0 | 552 |
| `star/private/star_solver.f90` | `star` | ported_unverified | 1,940 | 35 | 1 | 1,899 | 52 | 41 |
| `star/private/star_utils.f90` | `star` | partial | 4,149 | 156 | 0 | 3,660 | 0 | 489 |
| `star/private/struct_burn_mix.f90` | `star` | partial | 1,213 | 19 | 0 | 1,150 | 0 | 63 |
| `star/private/timestep.f90` | `star` | ported_unverified | 2,432 | 64 | 3 | 2,271 | 112 | 161 |
| `star/private/turb_info.f90` | `star` | ported_unverified | 608 | 12 | 5 | 552 | 497 | 56 |
| `star/private/turb_support.f90` | `star` | ported_unverified | 512 | 5 | 3 | 458 | 449 | 54 |
| `star/private/winds.f90` | `star` | ported_unverified | 706 | 14 | 4 | 663 | 663 | 43 |
| `star/private/write_model.f90` | `star` | ported_unverified | 281 | 4 | 1 | 246 | 246 | 35 |
| `star/public/star_def.f90` | `star` | ported_unverified | 26 | 0 | 0 | 0 | 0 | 26 |
| `star_data/private/star_controls.inc` | `star_data` | partial | 1,412 | 0 | 0 | 0 | 0 | 1,412 |
| `star_data/private/star_controls_dev.inc` | `star_data` | partial | 49 | 0 | 0 | 0 | 0 | 49 |
| `star_data/private/star_job_controls.inc` | `star_data` | partial | 368 | 0 | 0 | 0 | 0 | 368 |
| `star_data/private/star_job_controls_dev.inc` | `star_data` | partial | 25 | 0 | 0 | 0 | 0 | 25 |
| `star_data/private/star_job_controls_params.inc` | `star_data` | partial | 2 | 0 | 0 | 0 | 0 | 2 |
| `star_data/public/star_data.inc` | `star_data` | partial | 14 | 0 | 0 | 0 | 0 | 14 |
| `star_data/public/star_data_def.f90` | `star_data` | partial | 161 | 4 | 0 | 42 | 0 | 119 |
| `star_data/public/star_data_def.inc` | `star_data` | partial | 1,195 | 0 | 0 | 0 | 0 | 1,195 |
| `star_data/public/star_data_procedures.inc` | `star_data` | partial | 210 | 0 | 0 | 0 | 0 | 210 |
| `star_data/public/star_data_step_input.inc` | `star_data` | partial | 208 | 0 | 0 | 0 | 0 | 208 |
| `star_data/public/star_data_step_work.inc` | `star_data` | partial | 1,192 | 0 | 0 | 0 | 0 | 1,192 |
| `turb/private/mlt.f90` | `turb` | ported_unverified | 183 | 1 | 1 | 129 | 129 | 54 |
| `turb/private/semiconvection.f90` | `turb` | ported_unverified | 131 | 1 | 1 | 75 | 75 | 56 |
| `turb/private/tdc.f90` | `turb` | ported_unverified | 381 | 2 | 1 | 317 | 167 | 64 |
| `turb/private/tdc_support.f90` | `turb` | partial | 681 | 11 | 10 | 448 | 415 | 233 |
| `turb/private/thermohaline.f90` | `turb` | ported_unverified | 363 | 6 | 1 | 307 | 43 | 56 |
| `turb/public/turb.f90` | `turb` | ported_unverified | 294 | 4 | 4 | 151 | 151 | 143 |
| `utils/private/utils_idict.f90` | `utils` | partial | 525 | 20 | 0 | 472 | 0 | 53 |
| `utils/private/utils_nan.f90` | `utils` | partial | 32 | 0 | 0 | 0 | 0 | 32 |
| `utils/private/utils_nan_dp.f90` | `utils` | partial | 270 | 8 | 0 | 191 | 0 | 79 |
| `utils/public/utils_def.f90` | `utils` | partial | 62 | 0 | 0 | 0 | 0 | 62 |
| `utils/public/utils_lib.f90` | `utils` | partial | 1,189 | 75 | 0 | 1,015 | 0 | 174 |

## Unmapped routines in `partial` files (backlog)

3,718 routines in files with status `partial` have no gate-checked `routine_map` entry. Some may already be implemented without a map entry; the table cannot tell. Largest first; the full list is in the JSON file. Ordering by failing cases that reach the routine is not yet derived.

| # | Upstream file | Routine | Kind | Lines | Span |
| ---: | :--- | :--- | :--- | ---: | :--- |
| 1 | `star/private/profile_getval.f90` | `getval_for_profile` | subroutine | 2,202 | 263-2464 |
| 2 | `star/private/history.f90` | `history_getval` | subroutine | 1,843 | 1241-3083 |
| 3 | `star/private/ctrls_io.f90` | `store_controls` | subroutine | 1,720 | 767-2486 |
| 4 | `star/private/ctrls_io.f90` | `set_controls_for_writing` | subroutine | 1,716 | 2489-4204 |
| 5 | `num/private/mod_newton.f90` | `do_newton` | subroutine | 1,250 | 256-1505 |
| 6 | `star/private/history.f90` | `do_history_info` | subroutine | 1,184 | 56-1239 |
| 7 | `star/job/run_star_support.f90` | `do_star_job_controls_after` | subroutine | 1,116 | 1896-3011 |
| 8 | `star/private/evolve.f90` | `do_step_part2` | function | 1,102 | 499-1600 |
| 9 | `star/private/rotation_mix_info.f90` | `set_rotation_mixing_info` | subroutine | 1,077 | 47-1123 |
| 10 | `net/private/net_derivs.f90` | `get_derivs` | subroutine | 1,038 | 33-1070 |
| 11 | `net/private/net_initialize.f90` | `do_read_net_file` | subroutine | 921 | 266-1186 |
| 12 | `binary/private/run_binary_support.f90` | `do_run1_binary` | subroutine | 798 | 37-834 |
| 13 | `num/private/mod_bobyqa.f` | `BOBYQB` | subroutine | 749 | 140-888 |
| 14 | `rates/private/raw_rates.f90` | `set_raw_rate` | subroutine | 718 | 68-785 |
| 15 | `eos/private/skye_ideal.f90` | `compute_ideal_ele` | subroutine | 622 | 85-706 |
| 16 | `num/private/mod_newuoa.f` | `NEWUOB` | subroutine | 621 | 81-701 |
| 17 | `net/private/net_approx21.f90` | `approx21_dfdy` | subroutine | 620 | 2033-2652 |
| 18 | `net/private/net_approx21.f90` | `approx21_dydt` | subroutine | 619 | 926-1544 |
| 19 | `star/private/pulse_cafein.f90` | `get_cafein_data` | subroutine | 616 | 39-654 |
| 20 | `astero/private/extras_support.f90` | `do_astero_extras_check_model` | function | 612 | 58-669 |
| 21 | `num/private/mod_rosenbrock.f` | `roscor` | subroutine | 594 | 606-1199 |
| 22 | `star/private/report.f90` | `do_report` | subroutine | 594 | 139-732 |
| 23 | `star/private/pulse_fgong.f90` | `get_fgong_data` | subroutine | 587 | 47-633 |
| 24 | `star/private/profile.f90` | `do_profile_info` | subroutine | 584 | 358-941 |
| 25 | `interp_2d/private/bicub_db.f` | `fvbicub_db` | subroutine | 572 | 1371-1942 |
| 26 | `net/private/net_approx21.f90` | `approx21_special_reactions` | subroutine | 572 | 352-923 |
| 27 | `net/private/net_burn_const_p.f90` | `burn_1_zone_const_P` | subroutine | 547 | 33-579 |
| 28 | `net/private/net_screen.f90` | `screen_net` | subroutine | 513 | 49-561 |
| 29 | `star/private/star_job_ctrls_io.f90` | `store_star_job_controls` | subroutine | 505 | 613-1117 |
| 30 | `star/private/hydro_eqns.f90` | `eval_equ_for_solver` | subroutine | 504 | 44-547 |
| 31 | `star/private/star_job_ctrls_io.f90` | `set_star_job_controls_for_writing` | subroutine | 504 | 1161-1664 |
| 32 | `star/private/evolve.f90` | `okay_energy_conservation` | function | 498 | 1101-1598 |
| 33 | `star/private/pulse_osc.f90` | `get_osc_data` | subroutine | 498 | 46-543 |
| 34 | `num/private/mod_dop853.f` | `dp86co` | subroutine | 496 | 211-706 |
| 35 | `chem/public/chem_def.f90` | `set_some_isos` | subroutine | 489 | 2088-2576 |
| 36 | `atm/private/table_atm.f90` | `get_table_values` | subroutine | 488 | 195-682 |
| 37 | `interp_2d/private/akima760_db.f` | `RGPD3P_db` | subroutine | 483 | 346-828 |
| 38 | `net/private/net_derivs.f90` | `get1_derivs` | subroutine | 473 | 1073-1545 |
| 39 | `interp_2d/private/renka790_db.f` | `do_CSHEP2_db` | subroutine | 470 | 1-470 |
| 40 | `rates/private/rates_initialize.f90` | `read_reaction_parameters` | subroutine | 466 | 665-1130 |
