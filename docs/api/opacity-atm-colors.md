# Opacity, Atmospheres & Photometry (`kap`, `atm`, `colors`)

This reference page covers the radiative and conductive opacity subpackage (`flexmesa.kap`), the surface boundary condition and model atmosphere subpackage (`flexmesa.atm`), and the synthetic spectral energy distribution and filter photometry subpackage (`flexmesa.colors`).

For physical derivations and diagnostic plots, see [Opacity, Atmospheres & Photometry](../concepts/opacity-and-atmospheres.md).

---

## Quick Usage Example

```python
import numpy as np
import flexmesa.atm as atm
import flexmesa.colors as colors
import flexmesa.kap as kap
from flexmesa import constants

# 1. Total Rosseland mean opacity (radiative + electron conduction)
k_res = kap.kap_get_results(
    z=0.02,
    x=0.70,
    zbase=0.02,
    log_rho=-5.0,
    log_t=5.0,
    lnfree_e=20.0,
    zbar=1.10,
)

# 2. Tabulated photospheric surface boundary (P_surf, T_surf)
a_res = atm.eval_table(
    luminosity=constants.LSUN,
    r=constants.RSUN,
    m=constants.MSUN,
    cgrav=constants.STANDARD_CGRAV,
    table_id=atm.ATM_TABLE_PHOTOSPHERE,
    z=0.02,
    skip_partials=False,
    teff=5777.0,
)

# 3. Synthetic Johnson-Cousins colors at Teff = 5777 K, log g = 4.44, [M/H] = 0.0
phot = colors.eval_colors(
    teff=5777.0, log_g=4.44, metallicity=0.0, r=constants.RSUN
)
print(
    f"kappa={k_res.kap:.4f} cm^2/g, T_surf={np.exp(a_res.ln_t):.1f} K, "
    f"M_bol={float(phot[0]):.3f} mag"
)
```

---

## Radiative, Conductive & Compton Opacities (`flexmesa.kap`)

`flexmesa.kap` combines low-temperature molecular (`Ferguson`/`AESOPUS`/`kapCN`) and high-temperature atomic (`OPAL` Type 1/Type 2 and `Opacity Project`) Rosseland mean radiative opacities $\kappa_{\mathrm{rad}}(\log_{10} R, \log_{10} T, X, Z)$ with Poutanen (2017) relativistic Compton scattering and Cassisi et al. (2007) / Blouin et al. (2020) degenerate electron thermal conduction $\kappa_{\mathrm{cond}}$ via harmonic blending $\kappa^{-1} = \kappa_{\mathrm{rad}}^{-1} + \kappa_{\mathrm{cond}}^{-1}$. Matching MESA `26.4.1` `kap.defaults` and `kap_eval.f90` (`#149`, `#161`), `kap_get_results` and `kap_get_results_jax` default to `use_type2_opacities=True`, `use_zbase_for_type1=True`, `use_blouin_conductive_opacities=True`, `kap_co_option="gs98_co"`, and `kap_type2_full_off_x=1.0e-3` / `kap_type2_full_on_x=1.0e-6`, derive Compton and low-$T$/high-$T$ blend boundaries dynamically from the loaded table headers (`7.99` on `OP_a09` vs. `8.69` on `gs98`/`a09`/`OP_gs98`), and require an explicit `zbase >= 0.0` whenever Type 2, `kapCN`, or `AESOPUS` opacities are active.

::: flexmesa.kap.core

::: flexmesa.kap.submodels

::: flexmesa.kap.jax_kernels

::: flexmesa.kap.jax_interp

::: flexmesa.kap.jax_co_lowt

::: flexmesa.kap.tables_jax

::: flexmesa.kap.tables_co_lowt_jax

---

## Model Atmospheres & Outer Boundary Conditions (`flexmesa.atm`)

`flexmesa.atm` evaluates outer photospheric boundary conditions $(\ln P_{\mathrm{surf}}, \ln T_{\mathrm{surf}})$ and their exact $(L, \ln R, \ln M, \ln \kappa)$ partial derivatives using grey $T(\tau)$ relations (`Eddington`, `solar_Hopf`, `Krishna_Swamy`, `Trampedach_solar` via `get_t_tau_id` and `eval_t_tau_atm` across `'fixed'`, `'iterated'`, and `'varying'` opacity modes with `pextra_factor`), Guillot (2010) irradiated planetary atmospheres (`eval_irradiated`), adaptive Dormand-Prince 5(4) (`dopri5`) optical-depth integration (`eval_t_tau_varying`), and pre-validated 2D bicubic spline atmosphere tables (`photosphere`, `tau_100`, `tau_10`, `tau_1`, `tau_1m1`, `WD_tau_25`, `DB_WD_tau_25` via `get_table_id`, `eval_table`, and off-table `'T_tau'` boundary blending in `eval_table_atm` using `get_table_alfa_beta`).

::: flexmesa.atm._core

::: flexmesa.atm._jax

---

## Synthetic SEDs & Filter Photometry (`flexmesa.colors`)

`flexmesa.colors` synthesizes stellar spectral energy distributions (SEDs) across the 1199-wavelength Kurucz/Castelli `(Teff, log_g, [M/H])` grid using a 32-gather 3D cubic Hermite interpolator and integrates bolometric corrections and synthetic absolute magnitudes across Johnson-Cousins $UBVRIJHK$ filter transmission curves via non-uniform quadratic Simpson quadrature.

::: flexmesa.colors.photometry

::: flexmesa.colors.sed

::: flexmesa.colors.synthetic

::: flexmesa.colors.utils

---

## Compiled Rust Extension Bridge (`flexmesa._rust`)

Low-level Safe Rust PyO3 bindings exposed via `flexmesa._rust` for opacities, atmospheres, and synthetic photometry (`_ArrF64` and `_ArrI64` denote `float64` and `int64` NumPy arrays).

### `kap_py`

- **`kap_build_full_tables_bin`** (`(kap_dir: str, aesopus_section_path: str, out_path: str) -> None`): Assembles and checks an `FMKAP001` `kap_tables_full.bin` file from a MESA `kap_data` directory and pre-converted `aesopus_section.bin`.
- **`kap_get_aesopus_rust`** (`(zbase: float, x: float, xc: float, xn: float, xo: float, log_rho: float, log_t: float) -> tuple[float, float, float]`): Evaluates `AESOPUS_AGSS09` low-temperature opacity `(kap, dlnkap_dlnRho, dlnkap_dlnT)`.
- **`kap_get_co_rust`** (`(zbase: float, x: float, dxc: float, dxo: float, log_rho: float, log_t: float, kap_co_option: str = "gn93_co", cubic_x: bool = False, cubic_z: bool = False) -> tuple[float, float, float]`): Evaluates Type 2 C/O-enhanced opacity `(log10_kap, dlnkap_dlnRho, dlnkap_dlnT)`.
- **`kap_get_compton_opacity_rust`** (`(rho: float, t: float, lnfree_e: float, d_lnfree_e_dln_rho: float, d_lnfree_e_dln_t: float, eta: float, d_eta_dln_rho: float, d_eta_dln_t: float) -> tuple[float, float, float]`): Evaluates Poutanen (2017) Compton opacity `(kap, dlnkap_dlnRho, dlnkap_dlnT)`.
- **`kap_get_elect_cond_opacity_rust`** (`(zbar: float, log_rho: float, log_t: float, use_blouin: bool = True) -> tuple[float, float, float]`): Evaluates electron thermal conduction opacity `(kap, dlnkap_dlnRho, dlnkap_dlnT)`.
- **`kap_get_fixed_metal_rust`** (`(z: float, x: float, log_rho: float, log_t: float, prefix: str = "gs98", cubic_x: bool = False, cubic_z: bool = False) -> tuple[float, float, float]`): Evaluates Type 1 fixed-metal opacity `(log10_kap, dlnkap_dlnRho, dlnkap_dlnT)`.
- **`kap_get_kapcn_rust`** (`(zbase: float, x: float, fc: float, fn_val: float, log_rho: float, log_t: float) -> tuple[float, float, float]`): Evaluates `kapCN` Lederer & Aringer (2009) low-temperature opacity `(kap, dlnkap_dlnRho, dlnkap_dlnT)`.
- **`kap_get_results_rust`** (`(z: float, x: float, zbase: float, xc: float, xn: float, xo: float, xne: float, log_rho: float, log_t: float, lnfree_e: float, d_lnfree_e_dln_rho: float, d_lnfree_e_dln_t: float, eta: float, d_eta_dln_rho: float, d_eta_dln_t: float, zbar: float = 1.0, kap_option: str = "gs98", kap_co_option: str = "gs98_co", kap_low_t_option: str = "lowT_fa05_gs98", kap_blend_log_t_upper_bdy: float = 3.88, kap_blend_log_t_lower_bdy: float = 3.80, cubic_interpolation_in_x: bool = False, cubic_interpolation_in_z: bool = False, include_electron_conduction: bool = True, use_blouin_conductive_opacities: bool = True, use_zbase_for_type1: bool = True, use_type2_opacities: bool = True, kap_type2_full_off_x: float = 1.0e-3, kap_type2_full_on_x: float = 1.0e-6, kap_type2_full_off_dz: float = 0.001, kap_type2_full_on_dz: float = 0.01) -> tuple[float, float, float, list[float]]`): Evaluates `get_kap_results`, returning `(kap, dlnkap_dlnRho, dlnkap_dlnT, kap_fracs[4])`.
- **`kap_get_results_zones_rust`** (`(z: float, x: float, zbase: float, xc: float, xn: float, xo: float, xne: float, log_rho_zones: _ArrF64, log_t_zones: _ArrF64, lnfree_e: float, d_lnfree_e_dln_rho: float, d_lnfree_e_dln_t: float, eta: float, d_eta_dln_rho: float, d_eta_dln_t: float, zbar: float = 1.0, kap_option: str = "gs98", kap_co_option: str = "gs98_co", kap_low_t_option: str = "lowT_fa05_gs98", kap_blend_log_t_upper_bdy: float = 3.88, kap_blend_log_t_lower_bdy: float = 3.80, cubic_interpolation_in_x: bool = False, cubic_interpolation_in_z: bool = False, include_electron_conduction: bool = True, use_blouin_conductive_opacities: bool = True, use_zbase_for_type1: bool = True, use_type2_opacities: bool = True, kap_type2_full_off_x: float = 1.0e-3, kap_type2_full_on_x: float = 1.0e-6, kap_type2_full_off_dz: float = 0.001, kap_type2_full_on_dz: float = 0.01, parallel: bool = True) -> _ArrF64`): Evaluates `get_kap_results` across `N_zones` zones at fixed composition, returning an `(N_zones, 7)` array.
- **`kap_get_rust`** (`(chem_id: npt.NDArray[np.uintp], xa: _ArrF64, log_rho: float, log_t: float, lnfree_e: float, d_lnfree_e_dln_rho: float, d_lnfree_e_dln_t: float, eta: float, d_eta_dln_rho: float, d_eta_dln_t: float, zbase: float = -1.0, kap_option: str = "gs98", kap_co_option: str = "gs98_co", kap_low_t_option: str = "lowT_fa05_gs98", kap_blend_log_t_upper_bdy: float = 3.88, kap_blend_log_t_lower_bdy: float = 3.80, cubic_interpolation_in_x: bool = False, cubic_interpolation_in_z: bool = False, include_electron_conduction: bool = True, use_blouin_conductive_opacities: bool = True, use_zbase_for_type1: bool = True, use_type2_opacities: bool = True, kap_type2_full_off_x: float = 1.0e-3, kap_type2_full_on_x: float = 1.0e-6, kap_type2_full_off_dz: float = 0.001, kap_type2_full_on_dz: float = 0.01) -> tuple[float, float, float, list[float]]`): Evaluates `kap_get` from 1-based isotope IDs `chem_id` and mass fractions `xa`.
- **`kap_get_zones_rust`** (`(chem_id: npt.NDArray[np.uintp], xa_zones: _ArrF64, log_rho_zones: _ArrF64, log_t_zones: _ArrF64, lnfree_e_zones: _ArrF64, d_lnfree_e_dln_rho_zones: _ArrF64, d_lnfree_e_dln_t_zones: _ArrF64, eta_zones: _ArrF64, d_eta_dln_rho_zones: _ArrF64, d_eta_dln_t_zones: _ArrF64, zbase: float = -1.0, kap_option: str = "gs98", kap_co_option: str = "gs98_co", kap_low_t_option: str = "lowT_fa05_gs98", kap_blend_log_t_upper_bdy: float = 3.88, kap_blend_log_t_lower_bdy: float = 3.80, cubic_interpolation_in_x: bool = False, cubic_interpolation_in_z: bool = False, include_electron_conduction: bool = True, use_blouin_conductive_opacities: bool = True, use_zbase_for_type1: bool = True, use_type2_opacities: bool = True, kap_type2_full_off_x: float = 1.0e-3, kap_type2_full_on_x: float = 1.0e-6, kap_type2_full_off_dz: float = 0.001, kap_type2_full_on_dz: float = 0.01, parallel: bool = True) -> _ArrF64`): Evaluates `kap_get` across `N_zones` zones in parallel, returning an `(N_zones, 7)` array `[kap, dlnkap_dlnRho, dlnkap_dlnT, frac_lowT, frac_highT, frac_Type2, frac_Compton]`.

### `atm_py`

- **`atm_eval_e2`** (`(x: float) -> tuple[float, float]`): Evaluates the second exponential integral `(E_2(x), dE_2/dx)`.
- **`atm_eval_irradiated`** (`(l: float, r: float, m: float, cgrav: float, t_eq: float, p_surf: float, kap_guess: float, kap_v: float, gamma: float, errtol: float, max_iters: int, skip_partials: bool, chem_id: Sequence[int] | None = None, xa: _ArrF64 | None = None, zbase: float = 0.02, low_t_option: str = "low_t_fa05_gs98") -> tuple[float, ...]`): Evaluates an irradiated grey atmosphere (`eval_irradiated`).
- **`atm_eval_paczynski_gradr`** (`(t: float, p: float, rho: float, tau: float, kap: float, l: float, m: float, r: float, cgrav: float) -> float`): Evaluates the Paczynski (1969) optical-depth-diluted radiative gradient $\nabla_{\mathrm{rad}}$.
- **`atm_eval_t_tau`** (`(t_tau_id: int, tau: float, teff: float) -> float`): Evaluates $\ln T(\tau, T_{\mathrm{eff}})$ for a $T(\tau)$ relation ID.
- **`atm_eval_t_tau_atm`** (`(tau_surf: float, l: float, r: float, m: float, cgrav: float, kap_guess: float, pextra_factor: float, t_tau_relation: str, t_tau_opacity: str, errtol: float, max_iters: int, max_steps: int, skip_partials: bool, teff: float, chem_id: Sequence[int] | None = None, xa: _ArrF64 | None = None, zbase: float = 0.02, low_t_option: str = "low_t_fa05_gs98") -> tuple[float, ...]`): Dispatches $T(\tau)$ atmosphere evaluation across `'fixed'`, `'iterated'`, and `'varying'` opacity modes with `Pextra_factor` (`eval_t_tau_atm`).
- **`atm_eval_t_tau_dq_dtau`** (`(t_tau_id: int, tau: float) -> float`): Evaluates $dq/d\tau$ for a $T(\tau)$ relation ID at optical depth $\tau$.
- **`atm_eval_t_tau_uniform`** (`(tau_surf: float, l: float, r: float, m: float, cgrav: float, kap_guess: float, pextra_factor: float, t_tau_id: int, errtol: float, max_iters: int, skip_partials: bool, teff: float, chem_id: Sequence[int] | None = None, xa: _ArrF64 | None = None, zbase: float = 0.02, low_t_option: str = "low_t_fa05_gs98") -> tuple[float, ...]`): Evaluates a grey $T(\tau)$ atmosphere with uniform opacity (`eval_t_tau_uniform`).
- **`atm_eval_t_tau_varying`** (`(tau_surf: float, l: float, r: float, m: float, cgrav: float, t_tau_id: int, errtol: float, max_steps: int, skip_partials: bool, teff: float, chem_id: Sequence[int], xa: _ArrF64, zbase: float = 0.02, low_t_option: str = "low_t_fa05_gs98") -> tuple[float, ...]`): Evaluates a $T(\tau)$ atmosphere with depth-varying opacity (`eval_t_tau_varying`).
- **`atm_eval_table`** (`(l: float, r: float, m: float, cgrav: float, id: int, z: float, skip_partials: bool, teff: float) -> tuple[float, ...]`): Evaluates a tabulated model atmosphere (`eval_table`).
- **`atm_eval_table_atm`** (`(l: float, r: float, m: float, cgrav: float, atm_table: str, z: float, tau_factor: float, atm_off_table_option: str, t_tau_relation: str, t_tau_opacity: str, kap_guess: float, pextra_factor: float, errtol: float, max_iters: int, max_steps: int, skip_partials: bool, teff: float, chem_id: Sequence[int] | None = None, xa: _ArrF64 | None = None, zbase: float = 0.02, low_t_option: str = "low_t_fa05_gs98") -> tuple[tuple[float, ...], float]`): Evaluates a tabulated model atmosphere with upstream `atm_off_table_option` blending (`eval_table_atm`), returning `(eval_tuple, tau_base)`.
- **`atm_eval_table_zones`** (`(l: _ArrF64, r: _ArrF64, m: _ArrF64, cgrav: float, id: int, z: float, skip_partials: bool, teff: _ArrF64, parallel: bool = True) -> _ArrF64`): Evaluates `eval_table` across `n_zones` surface boundary conditions, returning an `(n_zones, 11)` array.
- **`atm_get_t_tau_base`** (`(t_tau_id: int) -> float`): Returns the base optical depth $\tau_{\mathrm{base}}$ of a $T(\tau)$ relation ID.
- **`atm_get_t_tau_id`** (`(name: str) -> int`): Maps an `atm_T_tau_relation` name (`'Eddington'`, `'solar_Hopf'`, `'Krishna_Swamy'`, `'Trampedach_solar'`) to its integer ID.
- **`atm_get_table_alfa_beta`** (`(l: float, teff: float, r: float, m: float, cgrav: float, id: int) -> tuple[float, float]`): Evaluates blending weights $(\alpha, \beta)$ between grey and table atmospheres.
- **`atm_get_table_base`** (`(id: int) -> float`): Returns the base optical depth $\tau_{\mathrm{base}}$ of an atmosphere table ID.
- **`atm_get_table_id`** (`(name: str) -> int`): Maps an `atm_table` name (`'tau_100'`, `'tau_10'`, `'tau_1'`, `'tau_1m1'`, `'photosphere'`, `'WD_tau_25'`, `'DB_WD_tau_25'`) to its integer ID.
- **`atm_table_photosphere_zsol_spline`** (`() -> tuple[_ArrF64, _ArrF64, bool, bool, _ArrF64]`): Returns `(logg_array, teff_array, iling, ilint, pgas_coeffs)` for the solar metallicity (`log_z == 0.0`) slice of `ATM_TABLE_PHOTOSPHERE`.

### `colors_py`

- **`py_colors_builtin_grid`** (`() -> tuple[_ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrI64, _ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64]`): Returns the built-in `Kurucz2003all` + `Johnson` grid arrays for `@jax.jit`: `(teff_grid, logg_grid, meta_grid, wavelengths, slot_map, sed_table, filters_on_sed_grid, filter_norm_integrals, vega_zp, ab_zp, st_zp)`.
- **`py_colors_calculate_bolometric_phot`** (`(wavelengths: _ArrF64, fluxes: _ArrF64) -> tuple[float, float]`): Computes `(bol_mag, bol_flux)` from `(wavelengths, fluxes)`.
- **`py_colors_calculate_synthetic`** (`(wavelengths: _ArrF64, fluxes: _ArrF64, filter_wavelengths: _ArrF64, filter_trans: _ArrF64, zero_point_flux: float) -> float`): Computes synthetic filter magnitude `calculate_synthetic` from an SED and filter transmission curve.
- **`py_colors_column_names`** (`() -> list[str]`): Returns the ordered 10 `colors` history output column names: `["Mag_bol", "Flux_bol", "Interp_rad", "B", "I", "J", "M", "R", "U", "V"]`.
- **`py_colors_compute_ab_zero_point`** (`(filt_wave: _ArrF64, filt_trans: _ArrF64) -> float`): Computes the AB photon-counting zero-point flux ($f_\nu = 3631\,$Jy).
- **`py_colors_compute_st_zero_point`** (`(filt_wave: _ArrF64, filt_trans: _ArrF64) -> float`): Computes the ST photon-counting zero-point flux ($f_\lambda = 3.63\times 10^{-9}\,\mathrm{erg\,s^{-1}\,cm^{-2}}\,\mathrm{Å}^{-1}$).
- **`py_colors_compute_vega_zero_point`** (`(vega_wave: _ArrF64, vega_flux: _ArrF64, filt_wave: _ArrF64, filt_trans: _ArrF64) -> float`): Computes the Vega photon-counting zero-point flux for a filter curve.
- **`py_colors_dilute_flux`** (`(surface_flux: _ArrF64, r: float, d: float) -> _ArrF64`): Applies geometric flux dilution `surface_flux * (r / d)^2`.
- **`py_colors_eval_colors`** (`(teff: float, log_g: float, metallicity: float, r: float, d: float = 3.0857e19, mag_system: str = "Vega", method: str = "Hermite") -> _ArrF64`): Evaluates the 10-column `colors` output vector `[Mag_bol, Flux_bol, Interp_rad, B, I, J, M, R, U, V]` at `(teff, log_g, metallicity, r, d)` (`teff` in K, `log_g` in `log10(cm s^-2)`, `r` and `d` in cm, `mag_system` in `{"Vega", "AB", "ST"}`, `method` in `{"Hermite", "Linear", "KNN"}`).
- **`py_colors_eval_colors_dual`** (`(teff: float, log_g: float, metallicity: float, r: float, d: float = 3.0857e19, mag_system: str = "Vega") -> tuple[_ArrF64, _ArrF64]`): Evaluates `(values, jacobian)` for `eval_colors` using `DualOrder1<4>` seeded on `(teff, log_g, metallicity, r)` at fixed `d`, where `values` has shape `(10,)` and `jacobian` has shape `(10, 4)`.
- **`py_colors_eval_colors_zones`** (`(teff: _ArrF64, log_g: _ArrF64, metallicity: _ArrF64, r: _ArrF64, d: float = 3.0857e19, mag_system: str = "Vega", method: str = "Hermite", parallel: bool = True) -> _ArrF64`): Evaluates the 10-column `colors` output across `n_zones` stellar models in parallel on the persistent `rayon` worker pool, returning an `(n_zones, 10)` array.
- **`py_colors_eval_sed`** (`(teff: float, log_g: float, metallicity: float, r: float, d: float = 3.0857e19, method: str = "Hermite") -> tuple[_ArrF64, _ArrF64]`): Evaluates the diluted stellar SED `(wavelengths, fluxes)` at `(teff, log_g, metallicity, r, d)` using `method` (`"Hermite"`, `"Linear"`, or `"KNN"`).
- **`py_colors_filter_curve`** (`(idx: int) -> tuple[_ArrF64, _ArrF64, float, float, float]`): Returns `(wavelengths, transmission, vega_zp, ab_zp, st_zp)` for Johnson filter `idx` (`0..7`).
- **`py_colors_filter_names`** (`() -> list[str]`): Returns the ordered 7 Johnson filter names (`["B", "I", "J", "M", "R", "U", "V"]`).
- **`py_colors_interpolate_array`** (`(x_in: _ArrF64, y_in: _ArrF64, x_out: _ArrF64) -> _ArrF64`): Resamples 1D curve `(x_in, y_in)` onto `x_out` via piecewise-linear interpolation with flat clamping outside `[x_in[0], x_in[-1]]`.
- **`py_colors_simpson_integration`** (`(x: _ArrF64, y: _ArrF64) -> float`): Evaluates composite non-uniform Simpson quadrature `simpson_integration(x, y)`.
- **`py_colors_trapezoidal_integration`** (`(x: _ArrF64, y: _ArrF64) -> float`): Evaluates composite trapezoidal quadrature `trapezoidal_integration(x, y)`.
