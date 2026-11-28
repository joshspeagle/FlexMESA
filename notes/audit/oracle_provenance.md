# Oracle provenance report

<!-- GENERATED VIA scripts/dashboard/build_routine_table.py (DECISIONS #80). Do not edit by hand. -->

Each `[[tests.oracles]]` entry of `benchmarks/dashboard/test_taxonomy.toml` is checked by `flexmesa.testing._oracle_provenance`: the reference is a committed MESA 26.4.1 artifact with a recipe, the test reads it, embedded values appear in it, a `#[test]` function reaches both the routine and the data, and exact entries compare at 0 ULP. A routine counts at the routine table's oracle levels only through a verified claim.

| Measure | Value |
| :--- | ---: |
| Claims (test, reference, routine) | 114 |
| Claims verified | 100 |
| Routines claimed | 67 |
| Routines verified exact (0 ULP) | 45 |
| Routines verified at print precision only | 18 |
| Routines not counted | 4 |
| Not counted only because of `scope` or `precision` | 3 |

## Routines claimed but not counted

| Routine | Test | Reference | Failing checks |
| :--- | :--- | :--- | :--- |
| `neu/private/mod_neu.f90::ifermi12` | `crates/flexmesa-neu/tests/mesa_26_4_1_oracles_test.rs` | `benchmarks/fortran_mesa/neu_oracles/mesa_26_4_1_neu_oracles.tsv` | `scope`: no #[test] function reaches both a call to ['ifermi12'] and the reference data |
| `neu/private/mod_neu.f90::zfermim12` | `crates/flexmesa-neu/tests/mesa_26_4_1_oracles_test.rs` | `benchmarks/fortran_mesa/neu_oracles/mesa_26_4_1_neu_oracles.tsv` | `scope`: no #[test] function reaches both a call to ['zfermim12'] and the reference data |
| `star/private/hydro_chem_eqns.f90::do1_chem_eqns` | `crates/flexmesa-star/tests/equ_he3_parity_test.rs` | `mesa_stepdumps:mesa_stepdumps/star/1.3M_ms_high_Z.stepdump` | `consts`: embedded values need a committed repository reference, not a bundle member |
| `star/private/rsp_lina.f90::do_LINA` | `crates/flexmesa-star/tests/rsp_parity_test.rs` | `tests/fixtures/golden/rsp/lina_26_4_1.tsv` | `scope`: no #[test] function reaches both a call to ['do_lina'] and the reference data |

## Verified routines

| Routine | Level |
| :--- | :--- |
| `eos/private/eos_blend.f90::is_contained` | exact |
| `eos/private/eos_blend.f90::min_distance_to_polygon` | exact |
| `eos/private/eosdt_eval.f90::Get_eosDT_Results` | exact |
| `eos/private/eosdt_eval.f90::get_Rho` | exact |
| `eos/private/eosdt_eval.f90::get_T` | exact |
| `eos/private/eospc_eval.f90::Get_PC_alfa` | exact |
| `eos/private/eospt_eval.f90::Get_PT_Results_using_DT` | exact |
| `eos/private/eospt_eval.f90::get_Pgas` | exact |
| `eos/private/helm.f90::helmeos2` | exact |
| `eos/private/ideal.f90::get_ideal_eos_results` | exact |
| `eos/private/ideal.f90::get_ideal_for_eosdt` | exact |
| `eos/private/ion_offset.f90::compute_ion_offset` | exact |
| `eos/private/skye.f90::Get_Skye_alfa_simple` | exact |
| `eos/public/eos_lib.f90::eosDT_get` | exact |
| `eos/public/eos_lib.f90::eosDT_get_Rho` | exact |
| `eos/public/eos_lib.f90::eosDT_get_T` | exact |
| `eos/public/eos_lib.f90::eosPT_get` | exact |
| `eos/public/eos_lib.f90::helmeos2_eval` | exact |
| `interp_2d/private/bipm_db.f90::do_evbipm_db` | exact |
| `interp_2d/private/bipm_db.f90::do_mkbipm_db` | exact |
| `neu/private/mod_neu.f90::neutrinos` | exact |
| `neu/public/neu_lib.f90::neu_get` | exact |
| `num/private/mod_newuoa.f::BIGDEN` | exact |
| `num/private/mod_newuoa.f::do_newuoa` | exact |
| `num/public/num_newuoa.dek::newuoa` | exact |
| `star/private/diffusion_support.f90::get_matrix_coeffs` | exact |
| `star/private/eps_mdot.f90::calculate_eps_mdot` | exact |
| `star/private/init_model.f90::get_zams_model` | exact |
| `star/private/mix_info.f90::update_rotation_mixing_info` | exact |
| `star/private/neu.f90::do_neu_for_cell` | exact |
| `star/private/overshoot_utils.f90::eval_conv_bdy_Hp` | exact |
| `star/private/overshoot_utils.f90::eval_conv_bdy_r` | exact |
| `star/private/paquette_coeffs.f90::free_collision_integrals` | exact |
| `star/private/paquette_coeffs.f90::initialise_collision_integrals` | exact |
| `star/private/paquette_coeffs.f90::paquette_coefficients` | exact |
| `star/private/solve_omega_mix.f90::do_solve_omega_mix` | exact |
| `star/private/winds.f90::do_set_mdot` | exact |
| `star/private/winds.f90::eval_rlo_wind` | exact |
| `star/private/winds.f90::eval_super_eddington_wind` | exact |
| `turb/private/mlt.f90::calc_MLT` | exact |
| `turb/private/semiconvection.f90::calc_semiconvection` | exact |
| `turb/public/turb.f90::set_MLT` | exact |
| `turb/public/turb.f90::set_TDC` | exact |
| `turb/public/turb.f90::set_semiconvection` | exact |
| `turb/public/turb.f90::set_thermohaline` | exact |
| `const/public/const_def.f90::do_const_init` | print |
| `const/public/const_lib.f90::const_init` | print |
| `eos/private/eos_helm_eval.f90::get_helm_for_eosdt` | print |
| `eos/private/eoscms_eval.f90::get_CMS_for_eosdt` | print |
| `eos/private/eosde_eval.f90::Get_eos_gamma_DE_Results` | print |
| `eos/private/eosdt_eval.f90::get1_for_eosdt` | print |
| `eos/private/eospc_eval.f90::get_PC_for_eosdt` | print |
| `eos/private/eospt_eval.f90::get_T` | print |
| `eos/private/gauss_fermi.f90::dfermi` | print |
| `eos/private/skye.f90::Get_Skye_alfa` | print |
| `eos/private/skye.f90::get_Skye_for_eosdt` | print |
| `eos/public/eos_lib.f90::eosPT_get_T` | print |
| `eos/public/eos_lib.f90::eos_fermi_dirac_integral` | print |
| `eos/public/eos_lib.f90::eos_gamma_DE_get` | print |
| `star/private/diffusion.f90::do_solve_diffusion` | print |
| `star/private/phase_separation.f90::do_phase_separation` | print |
| `star/private/rsp.f90::build_rsp_model` | print |
| `star/private/rsp_build.f90::do_rsp_build` | print |
