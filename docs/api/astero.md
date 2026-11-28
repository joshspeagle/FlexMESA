# Asteroseismology & Optimization (`flexmesa.astero`)

!!! warning "Frozen & Unverified JAX Astero Helpers (`#166`)"
    `flexmesa.astero.jax_kernels` is **frozen and unverified**
    (`JAX_ENGINE_STATUS = "frozen-unverified-until-#143"`) until the canonical
    Safe Rust engine passes the upstream MESA `26.4.1` history harness
    (`#143`), and must not be modified before `#143` passes.

The `flexmesa.astero` subpackage implements upstream MESA's `astero/` diagnostic, surface-correction, and seismic model-fitting suite in canonical Safe Rust (`crates/flexmesa-astero`) with frozen (`float64`) JAX helper kernels. In Rust, `crates/flexmesa-astero::adipls_support::AsteroAdipls` bridges `flexmesa_star::pulse_fgong` and `flexmesa-adipls` for `"adipls"` mode evaluation in `eval_model_astero_sample` (supporting `do_redistribute_mesh` startup parsing of `redistrb.c.pruned.in`). For test-side extras, `flexmesa._rust.AsteroAdipls` exposes both `get_one_el_info` (`do_adipls_get_one_el_info`) and `get_astero_one_el_info` (`astero/private/astero_support.f90:55-315`), alongside `astero_interpolate_l0_inertia` (`astero/private/astero_support.f90:702-724`) and `astero_parse_search_controls` (`astero/private/astero_run_support.f90:70-155`). It provides:

- **Acoustic Profiles & Global Scaling Relations**: Cumulative acoustic radius $\tau(r) = \int_0^r dr'/c_s(r')$, acoustic depth $\tau_0 - \tau(r)$, asymptotic and solar-calibrated $\Delta\nu$, $\nu_{\max}$, and acoustic cutoff frequency $\nu_{\mathrm{ac}}$, traditional asymptotic dipole g-mode period spacing $\Delta\Pi_1$ (with optional rotation modification), and acoustic glitch locations ($\tau_{\mathrm{He\,II}}$, $\tau_{\mathrm{BCZ}}$).
- **Empirical Surface-Effect Corrections**: Five empirical frequency-shift schemes accounting for near-surface superadiabatic convection and non-adiabatic wave physics (`"none"`, `"kjeldsen"` / `"power_law"`, `"cubic"` / `"bg14_1"`, `"combined"` / `"bg14_2"`, and `"sonoi"`).
- **Frequency-Separation Ratios & Seismic Likelihoods**: Roxburgh & Vorontsov (2003) 5-point small-to-large frequency separation ratios $(r_{01}, r_{10}, r_{02})$ insensitive to outer surface layers, mode-by-mode $\chi^2_{\mathrm{seism}}$, spectroscopic $\chi^2_{\mathrm{spec}}$, and weighted total $\chi^2_{\mathrm{tot}}$.
- **Derivative-Free & Gradient-Based Optimization**: Standalone Nelder-Mead simplex (`nelder_mead_minimize`), Powell quadratic direction-set (`powell_quadratic_minimize`), Cartesian grid-search drivers (`scan_cartesian_grid`), and multi-stage `astero/test_suite` runner (`run_astero_test_suite_case`), alongside end-to-end differentiable JAX seismic loss functions (`jax.grad`, `jax.hessian`).

For theoretical background and tutorials, see [Asteroseismology & Pulsation Exports](../concepts/asteroseismology.md) and [Tutorial 5: Asteroseismology & Pulsations](../tutorials/05-asteroseismology-and-pulsations.md).

---

## Quick Usage Example

```python
import flexmesa.astero as astero
from flexmesa import constants

dnu = astero.eval_delta_nu_radius_scaling(
    m_g=1.0 * constants.MSUN, r_phot_cm=1.0 * constants.RSUN
)
numax = astero.eval_nu_max(
    m_g=1.0 * constants.MSUN, r_phot_cm=1.0 * constants.RSUN, teff_k=5777.0
)
print(f"Solar scaling: dnu={dnu:.2f} uHz, nu_max={numax:.1f} uHz")
```

---

## Package Overview (`flexmesa.astero`)

::: flexmesa.astero

---

## Scaling Relations & Surface-Correction Wrappers (`flexmesa.astero.scaling_corr`)

::: flexmesa.astero.scaling_corr

---

## Differentiable Acoustic & Surface-Correction Kernels (`flexmesa.astero.jax_kernels`)

::: flexmesa.astero.jax_kernels

---

## Seismic Optimization Drivers (`flexmesa.astero.drivers`)

::: flexmesa.astero.drivers

---

## Compiled Rust Extension Bridge (`flexmesa._rust`)

Low-level Safe Rust PyO3 bindings exposed via `flexmesa._rust` for asteroseismology, GYRE hooks, and optimization (`_ArrF64` and `_ArrI32` denote `float64` and `int32` NumPy arrays).

### `astero_gyre_py` and `gyre_mesa_py`

- **`AsteroGyre`** (`(case_dir: str, mesa_dir: str)`; method `get_modes(model: StarModelHandle, el: int, store_model: bool) -> list[tuple[int, int, int, float, float, float]]`): Wraps `init_gyre` and `astero_gyre_get_modes` for one `astero` run.
- **`StarModelHandle`** (property `model_number: int`; method `gyre_pulse_data(add_center_point: bool, keep_surface_point: bool, add_atmosphere: bool) -> tuple[list[float], list[list[float]]]`): Snapshot copy of the star model handed to extras objects that set `wants_star_model = True`.
- **`GyreMesa`** (`(file: str)`; methods `set_constant`, `set_constant_str`, `set_mesa_constants`, `set_model`, `get_modes`, `get_modes_full`): Wraps `gyre_mesa_m` module state for one run.
- **`GyreMesaMode`** (properties `n_pg`, `n_p`, `n_g`, `freq`, `x`, `xi_r`, `xi_h`, `dw_dx`): Mode summary and eigenfunctions returned by `GyreMesa.get_modes_full` (`#170`), with frequency in $\mathrm{Hz}$ and radial eigenfunction profiles across the GYRE grid (`1..=md%n`).

### `astero_py`

- **`astero_eval_acoustic_profile`** (`(r: _ArrF64, csound: _ArrF64, r_center: float, r_phot: float) -> dict[str, Any]`): Evaluates the acoustic radius profile $\tau(r_k)$, acoustic depth profile $\tau_{\mathrm{depth}}(r_k)$, and photospheric acoustic radius $\tau_0$ ($\mathrm{s}$).
- **`astero_eval_chi2`** (`(obs_freqs: _ArrF64, obs_sigmas: _ArrF64, corr_model_freqs: _ArrF64, r010_obs: _ArrF64, r010_sigmas: _ArrF64, r010_model: _ArrF64, r02_obs: _ArrF64, r02_sigmas: _ArrF64, r02_model: _ArrF64, delta_nu_obs: float, delta_nu_sigma: float, delta_nu_model: float, nu_max_obs: float, nu_max_sigma: float, nu_max_model: float, spectro_obs: _ArrF64, spectro_sigmas: _ArrF64, spectro_model: _ArrF64, chi2_seismo_fraction: float, chi2_seismo_delta_nu_fraction: float, chi2_seismo_nu_max_fraction: float, chi2_seismo_r_010_fraction: float, chi2_seismo_r_02_fraction: float) -> dict[str, float]`): Evaluates composite asteroseismic and spectroscopic $\chi^2$ (`get_chi2`).
- **`astero_eval_delta_nu_asymptotic`** (`(photosphere_acoustic_r: float) -> float`): Evaluates asymptotic $\Delta\nu = 10^6 / (2 \tau_0)$ ($\mu\mathrm{Hz}$).
- **`astero_eval_delta_nu_radius_scaling`** (`(m_g: float, r_phot_cm: float) -> float`): Evaluates solar-scaled $\Delta\nu(M, R_{\mathrm{phot}})$ ($\mu\mathrm{Hz}$).
- **`astero_eval_delta_nu_scaling`** (`(m_g: float, teff_k: float, l_phot_erg_s: float) -> float`): Evaluates solar-scaled $\Delta\nu(M, T_{\mathrm{eff}}, L)$ ($\mu\mathrm{Hz}$).
- **`astero_eval_delta_pg`** (`(r: _ArrF64, r_mid: _ArrF64, brunt_n2: _ArrF64, csound_face: _ArrF64, mode_freq_uhz: float, bildsten2012: bool) -> float`): Evaluates asymptotic dipole ($l=1$) g-mode spacing $\Delta\Pi_1$ ($\mathrm{s}$).
- **`astero_eval_glitch_acoustic_depths`** (`(r: _ArrF64, csound: _ArrF64, gradr: _ArrF64, grada: _ArrF64, gamma1: _ArrF64, t: _ArrF64, r_center: float) -> tuple[float, float, float]`): Evaluates glitch locations $(R_{\mathrm{BCZ}}/R_\odot, \tau_{\mathrm{BCZ}}, \tau_{\mathrm{HeII}})$.
- **`astero_eval_nu_ac`** (`(g_surf_cgs: float, gamma1_surf: float, rho_surf: float, peos_surf: float) -> float`): Evaluates surface acoustic cutoff frequency $\nu_{\mathrm{ac}}$ ($\mu\mathrm{Hz}$).
- **`astero_eval_nu_max`** (`(m_g: float, r_phot_cm: float, teff_k: float) -> float`): Evaluates frequency of maximum oscillation power $\nu_{\max}$ ($\mu\mathrm{Hz}$).
- **`astero_eval_radial_delta_nu_ols`** (`(n_orders: _ArrF64, model_freqs: _ArrF64) -> float`): Evaluates OLS radial large frequency separation $\Delta\nu_{\mathrm{model}}$ ($\mu\mathrm{Hz}$).
- **`astero_eval_ratios_r01_r10`** (`(l0_freqs: _ArrF64, l0_sigs: _ArrF64, l1_freqs: _ArrF64, l1_sigs: _ArrF64) -> dict[str, _ArrF64]`): Evaluates 5-point frequency separation ratios $r_{01}, r_{10}$.
- **`astero_eval_ratios_r02`** (`(l0_freqs: _ArrF64, l0_sigs: _ArrF64, l1_freqs: _ArrF64, l1_sigs: _ArrF64, l2_freqs: _ArrF64, l2_sigs: _ArrF64) -> tuple[_ArrF64, _ArrF64]`): Evaluates quadrupole-radial separation ratios $r_{02}(n)$ and uncertainties.
- **`astero_eval_surface_correction`** (`(scheme: str, obs_freqs: _ArrF64, obs_sigmas: _ArrF64, model_freqs: _ArrF64, model_inertias: _ArrF64, l0_obs_freqs: _ArrF64, l0_model_freqs: _ArrF64, l0_inertias: _ArrF64, l0_orders: _ArrF64, delta_nu_obs: float, nu_max: float, b_power: float, correction_factor: float) -> dict[str, Any]`): Evaluates a surface-effect frequency correction (`"kjeldsen"`, `"cubic"`, `"combined"`, `"power_law"`, or `"sonoi"`).
- **`astero_match_modes_by_degree`** (`(obs_freqs: _ArrF64, cand_freqs: _ArrF64, cand_inertias: _ArrF64, cand_orders: _ArrI32) -> tuple[_ArrF64, _ArrF64, _ArrI32]`): Matches observed mode frequencies to candidate model modes (`find_closest`).
- **`astero_nelder_mead_minimize`** (`(objective: Any, x_first: list[float], x_lower: list[float], x_upper: list[float], max_evals: int, x_atol: float, x_rtol: float, chi2_tol: float, centroid_weight_power: float) -> dict[str, Any]`): Minimizes `objective(x_vec) -> float` via Nelder-Mead.
- **`astero_powell_quadratic_minimize`** (`(objective: Any, x_first: list[float], x_lower: list[float], x_upper: list[float], max_evals: int, rhobeg: float, rhoend: float) -> dict[str, Any]`): Minimizes `objective(x_vec) -> float` via Powell's quadratic trust-region method.
- **`astero_scan_cartesian_grid`** (`(objective: Any, x_lower: list[float], x_upper: list[float], x_delta: list[float]) -> dict[str, Any]`): Scans a Cartesian parameter grid using `objective(x_vec) -> float`.
