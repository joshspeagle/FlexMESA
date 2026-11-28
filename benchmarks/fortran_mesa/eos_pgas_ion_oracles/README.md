# MESA 26.4.1 `eos` Exact Kernel Oracles (`#161`, `DECISIONS #80`)

This directory holds the standalone Fortran generator (`gen_eos_pgas_ion_oracles.f90`), its build recipe (`Makefile`) and the committed IEEE-754 reference table (`mesa_26_4_1_eos_pgas_ion_oracles.tsv`). The table was written by compiled MESA release `26.4.1` (`r26.4.1`, commit `c8c202662922a22b3efefa824473ad615c623afd`) with MESA SDK `x86_64-linux-26.6.1` (`gfortran 15.2.0`, `CRMATH`).

## Regenerating

```bash
make -C benchmarks/fortran_mesa/eos_pgas_ion_oracles MESA_DIR=/path/to/mesa MESASDK_ROOT=/path/to/mesasdk generate clean
```

## Coverage

- `ion_offset::compute_ion_offset` (`crates/flexmesa-eos/tests/wave3a1_golden_tests.rs`):
  - `ion_offset_pure_h`: 1-species pure `h1` (`xa = [1.0]`).
  - `ion_offset_solar_3sp`: 3-species mixture `h1`, `he4`, `c12` (`xa = [0.70, 0.28, 0.02]`).
- `ideal::get_ideal_for_eosdt` / `ideal::get_ideal_eos_results` (`crates/flexmesa-eos/tests/wave3a1_golden_tests.rs`):
  - `ideal_solar_1e5_1em2`: 3-species mixture `h1`, `he4`, `c12` (`xa = [0.70, 0.28, 0.02]`) at `T = 1e5 K`, `Rho = 1e-2 g cm^-3` (`26 x 3` `val`, `d_dlnd`, `d_dlnT`).
- `eos_helm_eval::Get_HELMEOS_Results` (the converted slots only), `helm::helmeos2` / `eos_lib::helmeos2_eval` (`crates/flexmesa-eos/tests/wave3a2_helm_regimes_tests.rs`):
  - `helm_test_components`, `helm_sample_eos_c12`, `helm_pure_he_2e8_100`, `helm_pure_h_2e8_100`: `16 x 5` converted EOS slots (`val`, `d_dlnd`, `d_dlnT`, `d_dabar`, `d_dzbar`) plus `93` raw `helmeos2` `helm_res` slots on `helm_pure_he_2e8_100`.
- `eospc_eval::Get_PC_alfa` (`crates/flexmesa-eos/tests/wave3a3_pc_solid_tests.rs`):
  - `pc_alfa_gamma_c12_2p9_6p5`, `pc_alfa_gamma_c12_3p5_8p35`, `pc_alfa_tlim_c12_3p3_7p65`: `(alfa, d_alfa_dlogT, d_alfa_dlogRho)`.
- `skye::Get_Skye_alfa_simple` (`crates/flexmesa-eos/tests/wave3a3_skye_inverse_tests.rs`):
  - `skye_alfa_simple_2p35_7p3` (default `eos.defaults` bounds) and `skye_alfa_simple_f64_2p35_7p3` (explicit f64 bounds): `(alfa, d_alfa_dlogT, d_alfa_dlogRho)`.
- `eosdt_eval::get1_for_eosdt` (`crates/flexmesa-eos/tests/wave3b1_golden_tests.rs`):
  - `get1_opal_scvh_solar_1em2_1e5` (`eosdt_OPAL_SCVH` at `rho = 1e-2`, `T = 1e5`) and `get1_free_eos_solar_1em2_1e6` (`eosdt_max_FreeEOS` at `rho = 1e-2`, `T = 1e6`) for the 7-species solar mix (`X = 0.70`, `Z = 0.02`), `16 x 3` `val`, `d_dlnd`, `d_dlnT`, plus the `d_dxa` lnPgas and lnE channels for the 7 species (`dxa_s<i>_c<k>`; the by-charge mapping at `eosdt_eval.f90:1618-1627`).
- `eosPT_eval::get_Pgas` (`crates/flexmesa-eos/tests/wave3b1_golden_tests.rs`):
  - `eospt_get_pgas_solar_15_7p2`: 7-species solar mixture (`X = 0.70`, `Z = 0.02`) at `logT = 7.2`, solving for `logPgas` matching `lnE` from `eosPT_get` at `(logPgas=15.0, logT=7.2)` starting from `logPgas_guess = 14.8` with `logPgas_tol = 1e-6`, `other_tol = ln10 * 1e-6`, `max_iter = 100`.

`eosdt_eval::get1_for_eosdt` is not claimed exact either: the generator emits res slots 1-16 (`i_lnPgas` to `i_gamma3`, `eos_def.f90:95-127`) with `d_dlnd` and `d_dlnT`, and the two `d_dxa` channels, and those are compared at 0 ULP in Strict mode; but `num_eos_basic_results = i_frac_ideal` (`eos_def.f90:164`), and the `i_phase`, `i_latent_ddlnT`, `i_latent_ddlnRho` and seven `i_frac_*` slots (17-26) are not emitted or compared. It stays at print precision (`eos_oracles` entry) until every basic slot is emitted and compared.

`eos_helm_eval::get_helm_for_eosdt` is not claimed exact: this generator calls `Get_HELMEOS_Results` (`eos_helm_eval.f90:103-160`), so `composition_info`, `d_dxa` (`:72-85`), the zeroed phase, latent and frac slots (`:86-97`) and `res(i_frac_HELM) = 1` (`:98`) are not compared, and the Rust port has no standalone target for the `d_dxa` slots (`NUM_EOS_D_DXA_RESULTS = 2`, produced only inside the private `eval_helm_component`). It stays at print precision.
