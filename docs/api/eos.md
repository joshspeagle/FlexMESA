# Equations of State (`flexmesa.eos`)

The `flexmesa.eos` subpackage ports MESA's 7-component thermodynamic Equation of State (`eos/`) across Safe Rust (`crates/flexmesa-eos`) and JAX (`float64`). Every evaluation populates the canonical 26-slot thermodynamic state vector `res[0..26]` alongside exact analytical first and second logarithmic density and temperature partial derivatives (`d_dlnd[0..26]`, `d_dln_t[0..26]`) and optional composition partials (`d_dxa`):

1. **`Ideal` (`eval_ideal_eos`)**: Neutral ideal gas plus thermal radiation pressure for reference and outer boundary baselines.
2. **`HELM` (`eval_helm_eos`)**: Timmes & Swesty (2000) Helmholtz free-energy table with biquintic electron-positron interpolation, recursive `@jax.custom_jvp` Horner basis derivatives, and Yakovlev & Shalybkov (1989) Coulomb corrections.
3. **`PC` (`eval_pc_eos`)**: Potekhin & Chabrier (2010) analytical degenerate ionized plasma and Coulomb crystal EOS for white dwarf cores and neutron star envelopes.
4. **`Skye` (`eval_skye_eos`)**: Jermyn et al. (2021) unified third-order forward-mode automatic differentiation (`Order3` / nested `jax.jacfwd`) free-energy minimizer with smooth translational entropy blending across the liquid-to-solid Coulomb crystallization transition ($\Gamma_{\mathrm{crit}} \approx 175$).
5. **`FreeEOS`, `OPAL/SCVH` & `CMS` (`eval_free_eos`, `eval_opal_scvh_eos`, `eval_cms_eos`)**: 4D $(X, Z, \log_{10} Q, \log_{10} T)$ tabulated bicubic spline evaluators covering partially ionized stellar envelopes, solar-like interiors, and giant-planet hydrogen-helium mixtures.
6. **Polygonal $C^2$ Blending & Root-Finding Inversions (`eos_dt_get`, `eos_pt_get`)**: Smooth quintic smootherstep blending ($S_5(u) = 6u^5 - 15u^4 + 10u^3$) across component boundaries in the $(\log_{10}\rho, \log_{10}T)$ plane, and damped Newton-Raphson inversions from $(P_{\mathrm{gas}}, T)$ or $(s, T)$ to $\ln\rho$.

For physical derivations and phase diagrams, see [Equations of State (eos)](../concepts/eos.md).

---

## Quick Usage Example

```python
import numpy as np
import flexmesa.eos as eos
import flexmesa.net as net

net_info = net.get_net_info("basic.net")
xa = np.array([0.70, 2.0e-5, 0.28, 3.0e-3, 1.0e-3, 6.0e-3, 2.0e-3, 9.98e-3])
out = eos.eos_dt_get(
    rho=100.0,
    temp=1.5e7,
    chem_id=net_info.chem_id,
    xa=xa,
)
print(
    f"ln P_gas={out.res[eos.I_LN_PGAS]:.6f}, "
    f"grad_ad={out.res[eos.I_GRAD_AD]:.5f}, "
    f"Gamma_1={out.res[eos.I_GAMMA1]:.5f}"
)
```

---

## Package Overview (`flexmesa.eos`)

::: flexmesa.eos

---

## Core State Vector, Blending & 4D Tables (`flexmesa.eos.core`)

::: flexmesa.eos.core

---

## Aparicio Gauss-Fermi Integrals (`flexmesa.eos.fermi`)

::: flexmesa.eos.fermi

---

## Analytical & Free-Energy Evaluators (`ideal`, `helm`, `pc`, `skye`)

::: flexmesa.eos.ideal

::: flexmesa.eos.helm

::: flexmesa.eos.pc

::: flexmesa.eos.skye

---

## Thermodynamic Root-Finding Inversions (`flexmesa.eos.solver`)

::: flexmesa.eos.solver

---

## Compiled Rust Extension Bridge (`flexmesa._rust`)

Low-level Safe Rust PyO3 bindings exposed via `flexmesa._rust` for equations of state (`_ArrF64` denotes a `float64` NumPy `ndarray`).

### `eos_py`

- **`eos_basic_result_names`** (`() -> list[str]`): Returns the 26 canonical `eos_def` result channel names (`0..26`).
- **`eos_blend_results`** (`(alpha_in: float, d_alpha_dlog_t_in: float, d_alpha_dlog_rho_in: float, linear_blend: bool, res_1: _ArrF64, d_dlnd_1: _ArrF64, d_dln_t_1: _ArrF64, res_2: _ArrF64, d_dlnd_2: _ArrF64, d_dln_t_2: _ArrF64) -> tuple[_ArrF64, _ArrF64, _ArrF64]`): Blends two 26-channel EOS outputs (`res_1`, `res_2`) and their logarithmic derivatives using `blend_eos_results`.
- **`eos_compute_ion_offset`** (`(xa: _ArrF64, chem_id: list[int]) -> float`): Computes the ground-state ionization energy offset ($\mathrm{erg\,g^{-1}}$) from species mass fractions `xa` and 1-based MESA `chem_id` identifiers.
- **`eos_compute_ion_offset_from_za`** (`(xa: _ArrF64, z: list[int], a: _ArrF64) -> float`): Computes the ground-state ionization energy offset ($\mathrm{erg\,g^{-1}}$) directly from `(xa, z, a)`.
- **`eos_dfermi`** (`(dk: float, eta: float, theta: float) -> tuple[float, float, float]`): Evaluates the relativistic Fermi-Dirac integral `dfermi(dk, eta, theta)`, returning `(fd, fd_deta, fd_dtheta)`.
- **`eos_dfermi_zones`** (`(dk: float, eta: _ArrF64, theta: _ArrF64, parallel: bool = True) -> _ArrF64`): Evaluates `dfermi` across `nz` zones, returning an `(nz, 3)` array of `[fd, fd_deta, fd_dtheta]`.
- **`eos_eval_ideal`** (`(temp: float, rho: float, abar: float, zbar: float, xa: _ArrF64, chem_id: list[int]) -> tuple[_ArrF64, _ArrF64, _ArrF64]`): Evaluates the neutral ideal-gas + radiation EOS at a single zone from `(temp, rho, abar, zbar, xa, chem_id)`.
- **`eos_eval_ideal_from_weights`** (`(temp: float, rho: float, abar: float, zbar: float, xa: _ArrF64, weights: _ArrF64, a: _ArrF64) -> tuple[_ArrF64, _ArrF64, _ArrF64]`): Evaluates the neutral ideal-gas + radiation EOS at a single zone directly from `(temp, rho, abar, zbar, xa, weights, a)`.
- **`eos_eval_ideal_zones`** (`(temp: _ArrF64, rho: _ArrF64, abar: float, zbar: float, xa: _ArrF64, weights: _ArrF64, a: _ArrF64, parallel: bool = True) -> _ArrF64`): Evaluates the neutral ideal-gas + radiation EOS across `nz` zones at fixed composition, returning an `(nz, 78)` array packing `[res[0..26], d_dlnd[0..26], d_dln_t[0..26]]`.
- **`eos_is_contained`** (`(coords: _ArrF64, x: float, y: float) -> bool`): Evaluates `is_contained` on an `(N, 2)` polygon coordinate array and a 2D point `(x, y)`.
- **`eos_min_distance_to_polygon`** (`(coords: _ArrF64, x: float, y: float, dx_d1: float = 1.0, dx_d2: float = 0.0, dy_d1: float = 0.0, dy_d2: float = 1.0) -> tuple[float, float, float]`): Evaluates `min_distance_to_polygon` on an `(N, 2)` polygon coordinate array and a 2D dual point `(x, y)` with first-order tangent seeds `(dx_d1, dx_d2)` and `(dy_d1, dy_d2)`.
- **`eos_quintic_blend`** (`(alpha0: float) -> tuple[float, float]`): Evaluates the $C^2$ quintic blend polynomial and its derivative `(s, ds_dalpha0)` at `alpha0`.

### `eos_helm_py`

- **`eos_eval_helm`** (`(temp: float, rho: float, abar: float, zbar: float) -> tuple[_ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64]`): Evaluates the 26-channel `HELM` equation of state at a single zone `(temp, rho, abar, zbar)`, returning `(res, d_dlnd, d_dln_t, d_dabar, d_dzbar)`.
- **`eos_eval_helm_zones`** (`(temp: _ArrF64, rho: _ArrF64, abar: _ArrF64, zbar: _ArrF64, parallel: bool = True) -> _ArrF64`): Evaluates `eval_helm_eos` across `nz` zones, returning an `(nz, 130)` array packing `[res[0..26], d_dlnd[0..26], d_dln_t[0..26], d_dabar[0..26], d_dzbar[0..26]]`.
- **`eos_helm_cell_data`** (`(log_temp: float, din: float) -> tuple[float, float, float, float, _ArrF64, _ArrF64, _ArrF64, _ArrF64]`): Returns the `HelmCell` data `(t_lo, dth, d_lo, dd, fi_f, fi_dpdf, fi_ef, fi_xf)` for a single `(log_temp, din)` lookup.
- **`eos_helm_cells_batch`** (`(log_temp: _ArrF64, din: _ArrF64) -> tuple[_ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64]`): Returns batched `HelmCell` arrays `(t_lo, dth, d_lo, dd, fi_f, fi_dpdf, fi_ef, fi_xf)` across `nz` zones for vectorized `@jax.jit` evaluation.
- **`eos_opal_scvh_cells_batch`** (`(log_rho: _ArrF64, log_temp: _ArrF64, ix_1based: int = 5, iz_1based: int = 2) -> tuple[_ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64]`): Returns batched `OPAL/SCVH` cell bounds and 64-coefficient vertex arrays `(q0, q1, t0, t1, v00, v10, v01, v11)` across `nz` zones for `(ix, iz)`.

### `eos_pc_skye_py`

- **`eos_eval_pc`** (`(rho: float, temp: float, abar: float, zbar: float, chem_id: list[int], xa: _ArrF64, include_radiation: bool = True, pc_gamma_start_crystal: float = 170.0, pc_gamma_full_crystal: float = 180.0, mass_fraction_limit: float = 1.0e-4) -> tuple[_ArrF64, _ArrF64, _ArrF64]`): Evaluates the 26-channel Potekhin-Chabrier (`PC`) equation of state at a single zone `(rho, temp, abar, zbar, chem_id, xa)`.
- **`eos_eval_pc_zones`** (`(rho: _ArrF64, temp: _ArrF64, abar: _ArrF64, zbar: _ArrF64, xa_flat: _ArrF64, azion: _ArrF64, acmi: _ArrF64, include_radiation: bool = True, pc_gamma_start_crystal: float = 170.0, pc_gamma_full_crystal: float = 180.0, mass_fraction_limit: float = 1.0e-4, parallel: bool = True) -> _ArrF64`): Evaluates `PC` across `nz` zones at `(rho, temp, abar, zbar, xa_flat, azion, acmi)`, returning an `(nz, 78)` array packing `[res[0..26], d_dlnd[0..26], d_dln_t[0..26]]`.
- **`eos_eval_skye`** (`(rho: float, temp: float, abar: float, zbar: float, chem_id: list[int], xa: _ArrF64, skye_use_ion_offsets: bool = True, skye_solid_mixing_rule: str = "PC", mass_fraction_limit: float = 1.0e-4) -> tuple[_ArrF64, _ArrF64, _ArrF64]`): Evaluates the 26-channel Jermyn et al. (2021) `Skye` equation of state at a single zone `(rho, temp, abar, zbar, chem_id, xa)`.
- **`eos_eval_skye_zones`** (`(rho: _ArrF64, temp: _ArrF64, abar: _ArrF64, zbar: _ArrF64, xa_flat: _ArrF64, z: list[int], acmi: _ArrF64, a: _ArrF64, skye_use_ion_offsets: bool = True, skye_solid_mixing_rule: str = "PC", mass_fraction_limit: float = 1.0e-4, parallel: bool = True) -> _ArrF64`): Evaluates `Skye` across `nz` zones at `(rho, temp, abar, zbar, xa_flat, z, acmi, a)`, returning an `(nz, 78)` array packing `[res[0..26], d_dlnd[0..26], d_dln_t[0..26]]`.
- **`eos_get_pc_alfa`** (`(log_rho: float, log_t: float, abar: float, zbar: float, use_gamma_limit: bool = True, log_rho1: float = 2.999, log_rho2: float = 2.8, log_t1: float = 7.6, log_t2: float = 7.7, log_gamma_e_all_helm: float = -1.8, log_gamma_e_all_pc: float = -1.7) -> tuple[float, float, float]`): Evaluates `Get_PC_alfa` (`eospc_eval`), returning `(alfa, d_alfa_dlogT, d_alfa_dlogRho)`.
- **`eos_get_skye_alfa`** (`(log_rho: float, log_t: float, abar: float, zbar: float) -> tuple[float, float, float]`): Evaluates `Get_Skye_alfa` (`skye`), returning `(alfa, d_alfa_dlogT, d_alfa_dlogRho)`.
- **`eos_get_skye_alfa_simple`** (`(log_rho: float, log_t: float) -> tuple[float, float, float]`): Evaluates `get_Skye_alfa_simple` (`skye`), returning `(alfa, d_alfa_dlogT, d_alfa_dlogRho)`.

### `eos_solver_py`

- **`eos_cms_slice_coeffs`** (`(ix_1b: int) -> tuple[float, float, int, float, float, int, _ArrF64]`): Returns `(log_t_min, delta_log_t, num_log_ts, log_rho_min, delta_log_rho, num_log_rhos, coeffs_flat)` for a precomputed 4D `CMS` slice (`ix_1b` in `1..=11`).
- **`eos_dt_slice_coeffs`** (`(kind: int, ix_1b: int, iz_1b: int) -> tuple[float, float, int, float, float, int, _ArrF64]`): Returns `(log_q_min, del_log_q, num_log_qs, log_t_min, del_log_t, num_log_ts, coeffs_flat)` for a precomputed 4D `OPAL/SCVH` (`kind=0`) or `FreeEOS` (`kind=1`) slice (`ix_1b`, `iz_1b`).
- **`eos_eval_cms`** (`(rho: float, temp: float, chem_id: list[int], xa: _ArrF64, include_radiation: bool = True) -> tuple[_ArrF64, _ArrF64, _ArrF64]`): Evaluates Chabrier, Mazevet & Soubiran (`CMS`) H/He EOS standalone.
- **`eos_eval_free_eos`** (`(rho: float, temp: float, chem_id: list[int], xa: _ArrF64) -> tuple[_ArrF64, _ArrF64, _ArrF64]`): Evaluates `FreeEOS` 4D tabulated EOS standalone at `(rho, temp, chem_id, xa)`.
- **`eos_eval_opal_scvh`** (`(rho: float, temp: float, chem_id: list[int], xa: _ArrF64) -> tuple[_ArrF64, _ArrF64, _ArrF64]`): Evaluates `OPAL/SCVH` 4D tabulated EOS standalone at `(rho, temp, chem_id, xa)`.
- **`eos_eval_opal_scvh_zones`** (`(rho: _ArrF64, temp: _ArrF64, chem_id: Sequence[int], xa: _ArrF64, parallel: bool = True) -> _ArrF64`): Evaluates `OPAL/SCVH` 4D tabulated EOS across `nz` zones at `(chem_id, xa)`, returning an `(nz, 78)` array `[res[0..26], d_dlnd[0..26], d_dln_t[0..26]]`.
- **`eos_eval_gamma_de_get`** (`(abar: float, zbar: float, rho: float, energy: float, gamma: float = 1.6666666666666667) -> tuple[float, float, _ArrF64, _ArrF64, _ArrF64]`): Evaluates the closed-form ideal $\gamma$-law $(\rho, E)$ inverse solver `eos_gamma_DE_get(abar, zbar, rho, energy, gamma)`.
- **`eos_eval_dt_get`** (`(rho: float, temp: float, chem_id: list[int], xa: _ArrF64, use_skye: bool = True, use_pc: bool = False, pc_use_gamma_limit: bool = True, skye_solid_mixing_rule: str = "PC", use_cms: bool = False, use_free_eos: bool = True, use_opal_scvh: bool = True) -> tuple[_ArrF64, _ArrF64, _ArrF64]`): Evaluates the unified top-level `eosDT_get` 7-level cascade (`CMS -> Skye -> PC -> FreeEOS -> OPAL/SCVH -> HELM -> ideal`) at `(rho, temp, chem_id, xa)`.
- **`eos_eval_dt_get_rho`** (`(temp: float, which_other: int, other_value: float, chem_id: list[int], xa: _ArrF64, log_rho_guess: float = 0.0, use_skye: bool = True, use_pc: bool = False, use_cms: bool = False, use_free_eos: bool = True, use_opal_scvh: bool = True, logrho_tol: float | None = None, other_tol: float | None = None) -> tuple[float, float, _ArrF64, _ArrF64, _ArrF64]`): Evaluates `eosDT_get_Rho` solving for `rho` at fixed `temp` and `res[which_other] == other_value`.
- **`eos_eval_dt_get_t`** (`(rho: float, which_other: int, other_value: float, chem_id: list[int], xa: _ArrF64, log_t_guess: float = 6.0, use_skye: bool = True, use_pc: bool = False, use_cms: bool = False, use_free_eos: bool = True, use_opal_scvh: bool = True, logt_tol: float | None = None, other_tol: float | None = None) -> tuple[float, float, _ArrF64, _ArrF64, _ArrF64]`): Evaluates `eosDT_get_T` solving for `T` at fixed `rho` and `res[which_other] == other_value`.
- **`eos_eval_pt_get`** (`(pgas: float, temp: float, chem_id: list[int], xa: _ArrF64, use_skye: bool = True, use_pc: bool = False, pc_use_gamma_limit: bool = True, skye_solid_mixing_rule: str = "PC", use_cms: bool = False, use_free_eos: bool = True, use_opal_scvh: bool = True, log_pgas: float | None = None, log_t: float | None = None) -> tuple[float, float, float, float, _ArrF64, _ArrF64, _ArrF64]`): Evaluates `eosPT_get` at `(pgas, temp, chem_id, xa)`.
- **`eos_eval_pt_get_pgas`** (`(temp: float, which_other: int, other_value: float, chem_id: list[int], xa: _ArrF64, log_pgas_guess: float = 10.0, use_skye: bool = True, use_pc: bool = False, use_cms: bool = False, use_free_eos: bool = True, use_opal_scvh: bool = True) -> tuple[float, float, _ArrF64, _ArrF64, _ArrF64]`): Evaluates `eosPT_get_Pgas` solving for `pgas` at fixed `temp` and `res[which_other] == other_value`.
- **`eos_eval_pt_get_t`** (`(pgas: float, which_other: int, other_value: float, chem_id: list[int], xa: _ArrF64, log_t_guess: float = 6.0, use_skye: bool = True, use_pc: bool = False, use_cms: bool = False, use_free_eos: bool = True, use_opal_scvh: bool = True) -> tuple[float, float, _ArrF64, _ArrF64, _ArrF64]`): Evaluates `eosPT_get_T` solving for `T` at fixed `pgas` and `res[which_other] == other_value`.
