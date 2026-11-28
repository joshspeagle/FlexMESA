# Nuclear Networks, Rates & Neutrinos (`chem`, `ionization`, `neu`, `rates`, `net`)

This reference page documents the five nuclear and microphysical reaction subpackages: nuclide catalogs and solar abundance scales (`flexmesa.chem`), partial ionization and effective ionic charges (`flexmesa.ionization`), thermal neutrino cooling (`flexmesa.neu`), thermonuclear REACLIB rates and plasma screening (`flexmesa.rates`), and nuclear reaction networks and stiff 1-zone burners (`flexmesa.net`).

For governing equations and diagnostic plots, see [Nuclear Networks, Rates & Neutrinos](../concepts/nuclear-and-neutrinos.md).

---

## Quick Usage Example

```python
import numpy as np
import flexmesa.chem as chem
import flexmesa.net as net
import flexmesa.neu as neu

# 1. Solar mixture composition moments on basic.net (8 isotopes)
net_info = net.get_net_info("basic.net")
xa = np.array([0.70, 2.0e-5, 0.28, 3.0e-3, 1.0e-3, 6.0e-3, 2.0e-3, 9.98e-3])
comp = chem.get_composition_info(net_info.chem_id, xa)

# 2. Thermal neutrino loss rate (Itoh et al. 1996)
neu_out = neu.neu_get(
    t=1.0e8,
    log10_t=8.0,
    rho=1.0e4,
    log10_rho=4.0,
    abar=float(comp.abar),
    zbar=float(comp.zbar),
)

# 3. Nuclear reaction network RHS and analytical Jacobian
net_out = net.eval_net(
    "basic.net",
    x=xa,
    temp=8.0e7,
    rho=100.0,
    abar=float(comp.abar),
    zbar=float(comp.zbar),
    z2bar=float(comp.z2bar),
    ye=float(comp.ye),
    screening_mode="extended",
)
print(
    f"abar={comp.abar:.4f}, eps_neu={neu_out.loss[0]:.3e} erg/g/s, "
    f"eps_nuc={net_out.eps_nuc:.3e} erg/g/s"
)
```

---

## Nuclide Catalog & Solar Abundance Scales (`flexmesa.chem`)

`flexmesa.chem` provides 347 built-in reference nuclides (plus full `isotopes.data` parsing), 286 solar isotopic abundances, 10 standard solar abundance compilations (`AG89`, `GS98`, `L03`, `AGS05`, `AGSS09`, `L09`, `A09_Prz`, `MB22_phot`, `AAG21_phot`, `AAG21`), and composition moment reductions ($\bar{A}, \bar{Z}, \overline{Z^2}, Y_e, \mu$).

::: flexmesa.chem.catalog

::: flexmesa.chem.composition

---

## Partial Ionization & Effective Ionic Charge (`flexmesa.ionization`)

`flexmesa.ionization` evaluates effective ionic charge states $Z_{\mathrm{eff}}(T, \rho)$ for trace heavy elements (including $\mathrm{Fe}^{56}$ in $\mathrm{He}^4$ via pre-validated 2D bicubic splines and general $Z \in [1, 30]$ depression-of-continuum ionization) used in atomic diffusion and gravitational settling.

::: flexmesa.ionization.typical_charge

::: flexmesa.ionization.fe56_he4

::: flexmesa.ionization.tables

---

## Thermal Neutrino Cooling (`flexmesa.neu`)

`flexmesa.neu` ports the Itoh et al. (1996) fitting formulae across all five thermal neutrino emission channels (`pair`, `plasmon`, `photoneutrino`, `bremsstrahlung`, and `recombination`) with exact $(\rho, T, \bar{A}, \bar{Z})$ partial derivatives.

::: flexmesa.neu.driver

::: flexmesa.neu.inputs

::: flexmesa.neu.pair_plas

::: flexmesa.neu.phot

::: flexmesa.neu.brem

::: flexmesa.neu.reco

---

## Thermonuclear Rates, Weak Reactions & Screening (`flexmesa.rates`)

`flexmesa.rates` evaluates 316 built-in JINA REACLIB 7-parameter thermonuclear rates, weak electron-capture and $\beta$-decay tables, and Salpeter (1954), Dewitt, Graboske & Cooper (1973), Alastuey & Jancovici (1978), Itoh (1979) (`extended`), and Chugunov et al. (2007) plasma screening factors.

::: flexmesa.rates.core

::: flexmesa.rates.screening

::: flexmesa.rates.weak

::: flexmesa.rates.jax_kernels

::: flexmesa.rates.jax_screening

---

## Nuclear Reaction Networks & Stiff Burners (`flexmesa.net`)

`flexmesa.net` constructs arbitrary reaction topologies (`basic.net`, `pp_and_cno_extras.net`, `co_burn.net`, `approx21.net`, `mesa_80.net`), evaluates specific nuclear energy generation $\epsilon_{\mathrm{nuc}}$, reaction neutrino losses $\epsilon_{\nu,\mathrm{nuc}}$, molar abundance derivatives $dY_i/dt$, and exact analytical sparse/dense Jacobians $\partial(dY_i/dt)/\partial Y_j$ backed by deterministic `NetGeneralInfo::identity_id`-keyed thread-local `NetWorkState` buffers, and integrates stiff 1-zone prescribed-trajectory (`burn_1_zone`), constant-density variable-temperature (`burn_1_zone_const_density`), and constant-pressure (`burn_1_zone_const_p`) networks via semi-implicit Bader-Deuflhard and Rosenbrock time stepping.

::: flexmesa.net._core

::: flexmesa.net._jax

---

## Compiled Rust Extension Bridge (`flexmesa._rust`)

Low-level Safe Rust PyO3 bindings exposed via `flexmesa._rust` for nuclide catalogs, ionization, thermal neutrinos, nuclear reaction rates, screening, and reaction networks (`_ArrF64` denotes a `float64` NumPy `ndarray`).

### `chem_py`

- **`py_chem_ag89_namsol`** (`() -> list[str]`): Returns the ordered 286 isotope names (`namsol(1:286)`) in the Anders & Grevesse (1989) table.
- **`py_chem_ag89_yesol`** (`() -> float`): Returns the Anders & Grevesse (1989) solar electron fraction `yesol`.
- **`py_chem_ag89_zsol`** (`() -> float`): Returns the Anders & Grevesse (1989) bulk solar metallicity `zsol`.
- **`py_chem_category_id`** (`(name: str) -> int`): Returns the 1-based reaction category ID (`1..=24`) or `0` if unknown.
- **`py_chem_category_name`** (`(id: int) -> str | None`): Returns the reaction category name for 1-based `id` (`1..=24`).
- **`py_chem_element_long_name`** (`(z: int) -> str | None`): Returns the full element name for $Z \in [0, 112]$.
- **`py_chem_element_main_iso_name`** (`(z: int) -> str | None`): Returns the dominant isotope name for $Z \in [1, 112]$.
- **`py_chem_element_name`** (`(z: int) -> str | None`): Returns the short element symbol for $Z \in [0, 112]$.
- **`py_chem_generate_long_nuclide_names`** (`(z: Sequence[int], a: Sequence[int]) -> list[str]`): Generates long nuclide names from $(Z, A)$ arrays.
- **`py_chem_generate_nuclide_names`** (`(z: Sequence[int], a: Sequence[int]) -> list[str]`): Generates short nuclide names from $(Z, A)$ arrays.
- **`py_chem_get_composition_info`** (`(chem_ids: Sequence[int], x: Sequence[float], skip_partials: bool) -> tuple[tuple[float, ...], list[float], list[float], list[float]]`): Computes single-zone composition moments and optional partial derivatives, returning `((xh, xhe, xz, abar, zbar, z2bar, z53bar, ye, mass_correction, sumx), dabar_dx, dzbar_dx, dmc_dx)`.
- **`py_chem_get_composition_info_zones`** (`(chem_ids: Sequence[int], x_zones: _ArrF64, parallel: bool = False) -> _ArrF64`): Evaluates 10 composition moment columns `(xh, xhe, xz, abar, zbar, z2bar, z53bar, ye, mass_correction, sumx)` across `nz` zones (`x_zones` of shape `(nz, n_species)`).
- **`py_chem_get_element_id`** (`(name: str) -> int | None`): Looks up the 1-based atomic number $Z \in [1, 112]$ for an element symbol.
- **`py_chem_get_nuclide_index`** (`(name: str) -> int | None`): Returns the 1-based `chem_id` for a nuclide name (`"o16"`, `"fe56"`, etc.).
- **`py_chem_get_partition_fcn_indx`** (`(t9: float) -> int`): Returns the 1-based partition function temperature index for $T_9$ ($\mathrm{GK}$).
- **`py_chem_get_solar`** (`(name: str) -> tuple[float, float, float] | None`): Returns `(Z, A, X_sol)` for `name` in the Anders & Grevesse (1989) table.
- **`py_chem_get_stable_mass_frac`** (`(z: Sequence[int], a: Sequence[int], abun_in: Sequence[float]) -> list[float]`): Maps unstable nuclear network abundances onto the 286 stable solar isotopes.
- **`py_chem_lodders03_atom_percent`** (`(name: str) -> float`): Returns the Lodders (2003, Table 6) isotopic atom percentage for `name`.
- **`py_chem_m_div_h`** (`(x: float, z: float, scale: str) -> float`): Computes `[M/H]` (`chem_M_div_h`) for hydrogen fraction `x`, metallicity `z`, and solar scale `scale`.
- **`py_chem_nuclide_info`** (`(chem_id: int) -> tuple[int, str, int, int, float, float, float, float, float, int]`): Returns `(chem_id, name, z, n, spin, mass_excess, binding_energy, w, z53, isomeric_state)` for a 1-based `chem_id`, raising `ValueError` if unknown.
- **`py_chem_reaction_qtotal`** (`(num_in: int, num_out: int, reactants: Sequence[int]) -> float`): Computes the total reaction $Q$-value ($\mathrm{MeV}$) for `num_in` incoming and `num_out` outgoing species.
- **`py_chem_solar_cno_fraction`** (`(scale: str) -> float`): Returns the CNO mass fraction of $Z$ for `scale`.
- **`py_chem_solar_element_zfrac`** (`(scale: str) -> list[float]`): Returns the 112-element solar metal mass-fraction list for `scale`, raising `ValueError` if unsupported.
- **`py_chem_species_props`** (`(chem_ids: Sequence[int]) -> tuple[list[int], list[float], list[float], list[float], list[float]]`): Returns `(z_int, z, a, w, z53)` vectors for 1-based `chem_ids` used by `flexmesa.chem.composition`.
- **`py_chem_xsol`** (`(name: str) -> float`): Returns the Anders & Grevesse (1989) solar mass fraction for `name`.

### `ionization_py`

- **`py_ion_builtin_cell_grid`** (`() -> tuple[float, float, int, float, float, int, _ArrF64]`): Returns `(log_q_min, del_log_q, num_log_qs, log_t_min, del_log_t, num_log_ts, coeffs)` for `IonTableGrid::builtin`, where `coeffs` has shape `(5 * 6 * 28, num_log_qs * num_log_ts * 4)`.
- **`py_ion_charge_of_fe56_in_he4`** (`(log10_ne: float, log10_t: float) -> tuple[float, float, float]`): Evaluates `(z, dz_dlog10_ne, dz_dlog10_t)` for ${}^{56}\mathrm{Fe}$ in ${}^4\mathrm{He}$ at $(\log_{10} n_e, \log_{10} T)$ on the $105 \times 30$ Thomas-Fermi-Dirac-Hubbard grid (`eval_charge_of_Fe56_in_He4`).
- **`py_ion_charge_of_fe56_in_he4_zones`** (`(log10_ne: _ArrF64, log10_t: _ArrF64, parallel: bool = False) -> _ArrF64`): Evaluates `[z, dz_dlog10_ne, dz_dlog10_t]` across `n_zones` zones, dispatching to the 4-thread `rayon` worker pool when `parallel=True` and `n_zones >= 64`.
- **`py_ion_chi_effective`** (`(chi: float, c0: float, c1: float, c2: float, z1: float, z2: float) -> float`): Computes $\chi_{\mathrm{eff}}(z_2) = \chi + c_0 / z_2^3 + \min[c_1 z_2, c_2 ((z_2 / z_1)^{2/3} + 0.6)]$ in $\mathrm{eV}$ (`chi_effective`).
- **`py_ion_chi_info`** (`(a1: float, z1: float, t: float, log10_t: float, rho: float, log10_rho: float) -> tuple[float, float, float, float]`): Computes $(\chi, c_0, c_1, c_2)$ continuum-depression parameters in $\mathrm{eV}$ (`chi_info`).
- **`py_ion_eval_ionization`** (`(z: float, x: float, rho: float | None = None, log10_rho: float | None = None, t: float | None = None, log10_t: float | None = None) -> list[float]`): Evaluates all 28 `eval_ionization` channels for a single zone from either linear or `log10` inputs.
- **`py_ion_eval_ionization_dual`** (`(z: float, x: float, log10_rho: float, log10_t: float) -> tuple[_ArrF64, _ArrF64]`): Evaluates all 28 `eval_ionization` channels and their $28 \times 4$ Jacobian with respect to `(z, x, log10_rho, log10_t)` via `DualOrder1<4>`.
- **`py_ion_eval_ionization_zones`** (`(z: _ArrF64, x: _ArrF64, log10_rho: _ArrF64, log10_t: _ArrF64, parallel: bool = False) -> _ArrF64`): Evaluates all 28 `eval_ionization` channels across `n_zones` zones, dispatching to the 4-thread `rayon` worker pool when `parallel=True` and `n_zones >= 64`.
- **`py_ion_eval_typical_charge`** (`(cid: int, abar: float, free_e: float, t: float, log10_t: float, rho: float, log10_rho: float) -> float`): Computes whole-number typical charge (`eval_typical_charge`) against Allen (1973) ionization potentials (`IONIZATION_POTENTIALS`) for nuclide `cid` ($1 \le Z \le 30$).
- **`py_ion_fe56_he4_grid`** (`() -> tuple[_ArrF64, _ArrF64, _ArrF64]`): Returns `(log_ne, log_t, coeffs)` for the built-in $105 \times 30$ `Fe56He4Table` (`coeffs` shape `(105 * 30, 4)`).
- **`py_ion_potentials_table`** (`() -> _ArrF64`): Returns the `(30, 30)` Allen (1973) ionization potential table in $\mathrm{eV}$.
- **`py_ion_result_names`** (`() -> list[str]`): Returns the ordered list of 28 `eval_ionization` output channel names (`logPgas`, `logpp_H..Fe`, `Z_H..Fe`, `fneut_H..Fe`).

### `neu_py`

- **`neu_get_batch`** (`(t: _ArrF64, log10_t: _ArrF64, rho: _ArrF64, log10_rho: _ArrF64, abar: _ArrF64, zbar: _ArrF64, log10_tlim: float = 7.5, flags: Sequence[bool] = (True, True, True, True, True), parallel: bool = False) -> _ArrF64`): Evaluates `neu_get` across 1D arrays `(t, log10_t, rho, log10_rho, abar, zbar)` of shape `(nz,)`, returning a `(nz, 5)` array of `[sneu, dsneu_dT, dsneu_dRho, dsneu_dAbar, dsneu_dZbar]` ($\mathrm{erg\,g^{-1}\,s^{-1}}$) and partials, dispatching across the persistent 4-thread `rayon` pool when `parallel=True` and `nz >= 64`.
- **`neu_get_full_zones`** (`(log10_t: _ArrF64, log10_rho: _ArrF64, abar: _ArrF64, zbar: _ArrF64, log10_tlim: float = 7.5, flags: Sequence[bool] = (True, True, True, True, True), parallel: bool = False) -> _ArrF64`): Evaluates `neu_get` across 1D arrays `(log10_t, log10_rho, abar, zbar)` of shape `(nz,)`, returning a `(nz, 30)` array packing `loss[0..5]` followed by Fortran column-major `sources` (`d_idx` outer `0..5`, `channel` inner `0..5`).

### `rates_py`

- **`rates_compute_rev_ratio_coeff1`** (`(w_in: Sequence[float], spin_in: Sequence[float], w_out: Sequence[float], spin_out: Sequence[float]) -> float`): Computes detailed-balance first reverse coefficient `inverse_coefficients(1)` from nuclear masses `w` ($\mathrm{AMU}$) and ground-state spins `spin`.
- **`rates_eval_coulomb`** (`(which_mui: str, which_vs: str, temp: float, den: float, zbar: float, abar: float, z1: float, z2: float) -> tuple[float, float, float, float]`): Evaluates Coulomb chemical potential and electron screening shift `(mu1, mu2, dmu_eV, vs)`.
- **`rates_eval_ecapture_pair`** (`(lhs_name: str, rhs_name: str, t9: float, ye_rho: float, eta_k: float, which_mui: str = "pcr2009", which_vs: str = "itoh2002", zbar: float = 10.909090909090908, abar: float = 21.818181818181817) -> list[float]`): Evaluates on-the-fly relativistic Fermi-Dirac electron-capture / beta-decay rate.
- **`rates_eval_fl_epsnuc_3alf`** (`(t: float, rho: float, y: float, ue: float) -> float`): Evaluates Fushiki & Lamb (1987) pycnonuclear 3-alpha energy generation rate.
- **`rates_eval_rate_table`** (`(ir: int, temp: float) -> tuple[float, float]`): Evaluates an `interp_pm` 1D cubic rate table `(rate, drate_dt)` for reaction ID `ir`.
- **`rates_eval_reaclib_handle`** (`(handle: str, t9: float, check_e_flag: bool = True) -> tuple[float, float, float, float]`): Evaluates a REACLIB handle `(forward_rate, dforward_dlnT, reverse_rate, dreverse_dlnT)`.
- **`rates_eval_reaclib_screen_zones`** (`(handle: str, mode: str, temps: _ArrF64, dens: _ArrF64, zbar: float, abar: float, z2bar: float, a1: float, z1: float, a2: float, z2: float, parallel: bool = True) -> _ArrF64`): Evaluates a REACLIB handle and plasma screening across `n_zones` zones, returning an `(n_zones, 3)` array `[safe_log10(fw * scor), fw, scor]`.
- **`rates_eval_weak_pair`** (`(lhs_name: str, rhs_name: str, t9: float, ye_rho: float, use_suzuki: bool = False) -> list[float]`): Evaluates tabulated weak reaction rate (`weaklib` or `Suzuki2016`), returning a 9-element list.
- **`rates_eval_weak_table_index`** (`(idx: int, t9: float, l_ye_rho: float) -> tuple[str, str, list[float]]`): Evaluates the `idx`-th (`1..=463`) weak table at `(t9, l_ye_rho)`.
- **`rates_reaclib_terms`** (`(handle: str) -> tuple[bool, list[tuple[int, int, int, list[float], float, float, list[float]]]]`): Returns `(is_reverse, terms)` for a REACLIB reaction handle, where each term is `(chapter, rflag, inv_exp, coeffs[7], inv_c1, inv_c2, inv_part[24])`.
- **`rates_reaction_id_by_name`** (`(name: str) -> int`): Looks up the 1-based reaction ID (`1..=316`) for a reaction name.
- **`rates_reaction_info`** (`(ir: int) -> tuple[str, int, bool, str, int, int, int, float, float, float]`): Returns metadata `(name, op_code, is_reverse, target_handle, ye_exp, rho_exp, category, std_q, std_qneu, weak_lowt_rate)` for 1-based reaction ID `ir`.
- **`rates_reaction_name_by_id`** (`(ir: int) -> str`): Looks up the reaction name for a 1-based reaction ID (`1..=316`).
- **`rates_screen_pair`** (`(mode: str, temp: float, den: float, zbar: float, abar: float, z2bar: float, a1: float, z1: float, a2: float, z2: float) -> tuple[float, float, float]`): Evaluates plasma electrostatic screening `(scor, dscor_dt, dscor_dd)` for a reacting pair.
- **`rates_screen_pair_zones`** (`(mode: str, temps: _ArrF64, dens: _ArrF64, zbars: _ArrF64, abars: _ArrF64, z2bars: _ArrF64, a1: float, z1: float, a2: float, z2: float) -> _ArrF64`): Evaluates plasma screening across `n_zones`, returning an `(n_zones, 3)` array.
- **`rates_set_raw_rate`** (`(ir: int, temp: float, log_t: float | None = None) -> float`): Evaluates `set_raw_rate` for a single 1-based reaction ID `ir` (`1..=316`).
- **`rates_set_raw_rates`** (`(irs: npt.NDArray[np.uintp], temp: float, log_t: float | None = None) -> _ArrF64`): Evaluates `set_raw_rates` for a 1D array of 1-based reaction IDs `irs`.
- **`rates_set_raw_rates_zones`** (`(irs: npt.NDArray[np.uintp], temps: _ArrF64, log_ts: _ArrF64 | None = None) -> _ArrF64`): Evaluates `set_raw_rates` across `n_zones` temperatures, returning an `(n_zones, n_rxns)` array.
- **`rates_weak_info_by_names`** (`(lhs_name: str, rhs_name: str) -> tuple[float, float]`): Looks up `(halflife_sec, qneu_mev)` from `weak_info.list` by isotope names.

### `net_py`

- **`net_burn_1_zone`** (`(net_name: str, t_start: float, t_end: float, starting_x: _ArrF64, times: _ArrF64, log10_ts: _ArrF64, log10_rhos: _ArrF64, etas: _ArrF64, rate_factors: _ArrF64 | None = None, weak_rate_factor: float = 1.0, screening_mode: str = "extended", stptry_in: float = 0.0, max_steps: int = 10000, eps: float = 1.0e-8, odescal: float = 1.0e-12) -> tuple[_ArrF64, _ArrF64, float, float, int, int, int, int, int]`): Integrates a one-zone burn at constant or prescribed $(T(t), \rho(t))$ (`burn_1_zone`).
- **`net_burn_1_zone_const_density`** (`(net_name: str, t_start: float, t_end: float, starting_x: _ArrF64, starting_log10_t: float, log10_rho: float, rate_factors: _ArrF64 | None = None, weak_rate_factor: float = 1.0, screening_mode: str = "extended", stptry_in: float = 0.0, max_steps: int = 10000, eps: float = 1.0e-5, odescal: float = 1.0e-5) -> tuple[_ArrF64, float, _ArrF64, float, float, int, int, int, int, int]`): Integrates a constant-density variable-temperature one-zone burn (`burn_1_zone_const_density`).
- **`net_burn_1_zone_const_p`** (`(net_name: str, starting_temp: float, starting_x: _ArrF64, t_end: float, pressure: float, solver: str = "rodas4", clip: bool = False, rate_factors: _ArrF64 | None = None, weak_rate_factor: float = 1.0, screening_mode: str = "extended", h0: float = 1.0e-6, max_step_size: float = 1.0, max_steps: int = 1000, rtol: float = 1.0e-7, atol: float = 1.0e-8, x_min: float = -1.0e-10) -> tuple[_ArrF64, float, float, float, float, float, int, int, int, int, int]`): Integrates a constant-pressure one-zone burn (`burn_1_zone_const_p`).
- **`net_eval`** (`(net_name: str, x: _ArrF64, temp: float, log_t: float, rho: float, log_rho: float, abar: float, zbar: float, z2bar: float, ye: float, eta: float = 0.0, d_eta_dlnt: float = 0.0, d_eta_dlnrho: float = 0.0, rate_factors: _ArrF64 | None = None, weak_rate_factor: float = 1.0, screening_mode: str = "extended", skip_jacobian: bool = False) -> tuple[float, float, float, _ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64, float, _ArrF64, _ArrF64]`): Evaluates `net_get` at a single thermodynamic and composition state, returning `(eps_nuc, d_eps_nuc_drho, d_eps_nuc_dt, d_eps_nuc_dx, dxdt, d_dxdt_drho, d_dxdt_dt, d_dxdt_dx, eps_nuc_categories, eps_neu_total, rate_raw, rate_screened)`.
- **`net_eval_zones`** (`(net_name: str, x: _ArrF64, temps: _ArrF64, log_ts: _ArrF64, rhos: _ArrF64, log_rhos: _ArrF64, abar: float, zbar: float, z2bar: float, ye: float, eta: float = 0.0, d_eta_dlnt: float = 0.0, d_eta_dlnrho: float = 0.0, weak_rate_factor: float = 1.0, screening_mode: str = "extended", skip_jacobian: bool = False, parallel: bool = True) -> _ArrF64`): Evaluates `eval_net_with_work` across `n_zones` zones, returning an `(n_zones, 4)` array `[log10_eps_nuc, eps_nuc, d_eps_dt, d_eps_drho]`.
- **`net_list_networks`** (`() -> list[str]`): Lists all built-in `.net` nuclear network definition filenames.
