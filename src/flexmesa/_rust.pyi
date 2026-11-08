# SPDX-License-Identifier: LGPL-3.0-or-later
"""Type annotations for the compiled ``flexmesa._rust`` PyO3 extension.

Declares the public Python-to-Rust boundary across the Three-Tier Hybrid
Architecture:

1. **Tier A (`utils`, `testing`, `schema`, `data`, `auto_diff`, `chem`)**:
   Canonical Safe Rust implementations of IEEE-754 finiteness predicates,
   Fortran lexical scanners, golden-baseline parsers, control schema and
   namelist parsers, SHA-256 digests, `.safetensors` table converters,
   2nd-order dual-number benchmark kernels, periodic table and nuclide
   catalogs, Anders & Grevesse (1989) + Lodders (2003) solar abundances, and
   10 solar abundance scales (`AG89`..`AAG21`).
2. **Tier B (`math`, `interp_1d`, `interp_2d`, `neu`, `turb`, `chem`,
   `ionization`, `colors`)**:
   Domain-safe libm intrinsics, 1D Steffen/Huynh monotonic splines, 2D
   PSPLINE/Steffen bicubic surface interpolants, Itoh et al. (1996) 5-channel
   thermal neutrino losses, Cox (1968) / Kuhfuss (1986) convective transport
   (`MLT` and `TDC`), differentiable composition moments, ionization tables and
   diffusion typical charges, and 4D Kurucz + Johnson stellar photometry, with
   single-zone scalar entry points and persistent `rayon` multi-zone kernels.
3. **Tier C (`mtx`)**:
   Coupled block-tridiagonal linear solvers (`solve_block_tridiagonal`,
   `solve_block_tridiag_bcyclic`, and `block_multiply_xa`).

All array arguments (`_ArrF64`) expect 1D C-contiguous ``float64`` buffers
borrowed zero-copy via ``PyReadonlyArray1::as_slice``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

type _ArrF64 = npt.NDArray[np.float64]
type _ArrI32 = npt.NDArray[np.int32]
type _ArrI64 = npt.NDArray[np.int64]
type _ScalarVal = bool | int | float | str
type _TensorDict = dict[str, _ArrF64 | _ArrI64]

# fmt: off
# Tier A: Infrastructure (`utils`, `testing`, `schema`, `data`, `auto_diff`)
def is_bad(x: float) -> bool: ...
def is_bad_array(xs: _ArrF64) -> bool: ...
def replace_bad_nums(xs: _ArrF64, replacement: float) -> _ArrF64: ...
def remove_underbars(text: str) -> str: ...
def split_line(line: str, num: int) -> list[str]: ...
def tokenize(text: str) -> list[tuple[str, str]]: ...
def is_fortran_float_token(token: str) -> bool: ...
def parse_fortran_float(token: str) -> float: ...
def parse_golden_text(text: str) -> list[tuple[int, str, str, list[int], _ArrF64]]: ...  # noqa: E501
def parse_control_value(raw: str) -> tuple[str, _ScalarVal]: ...
def parse_defaults_text(text: str) -> list[tuple[str, str, _ScalarVal, bool, list[tuple[str, _ScalarVal]], str, str, int]]: ...  # noqa: E501
def parse_columns_text(text: str) -> list[tuple[str, str, str, int]]: ...
def parse_namelist_text(text: str) -> list[tuple[str, list[tuple[str, str | None, str, _ScalarVal, int]]]]: ...  # noqa: E501
def is_valid_sha256_hex(digest: str) -> bool: ...
def compute_sha256_bytes(data: bytes) -> str: ...
def compute_sha256_file(path: str) -> str: ...
def parse_lodders03_text(text: str) -> tuple[_TensorDict, dict[str, str]]: ...
def parse_isotopes_text(text: str) -> tuple[_TensorDict, dict[str, str]]: ...
def parse_helm_table_text(
    text: str, imax: int, jmax: int, logdlo: float, logdhi: float,
    logtlo: float, logthi: float,
) -> tuple[_TensorDict, dict[str, str]]: ...
def eval_autodiff_order2_zone(x: float, y: float) -> list[float]: ...
def eval_autodiff_order2_batch(x: _ArrF64, y: _ArrF64, parallel: bool = False) -> _ArrF64: ...  # noqa: E501

# Tier B: Microphysics & Numerics (`math`, `interp_1d/2d`, `neu`, `turb`)
def safe_sqrt(x: float) -> float: ...
def safe_log(x: float) -> float: ...
def safe_log10(x: float) -> float: ...
def exp10(x: float) -> float: ...
def pow(x: float, y: float) -> float: ...
def sinpi(x: float) -> float: ...
def cospi(x: float) -> float: ...
def tanpi(x: float) -> float: ...
def sgn(x: float) -> float: ...
def heaviside(x: float) -> float: ...
def str_to_vector(text: str) -> _ArrF64: ...
def safe_log_array(xs: _ArrF64) -> _ArrF64: ...
def safe_log10_array(xs: _ArrF64) -> _ArrF64: ...
def safe_sqrt_array(xs: _ArrF64) -> _ArrF64: ...
def exp10_array(xs: _ArrF64) -> _ArrF64: ...
def interp_pm(x: _ArrF64, y: _ArrF64) -> _ArrF64: ...
def interp_m3a(x: _ArrF64, y: _ArrF64) -> _ArrF64: ...
def interp_m3b(x: _ArrF64, y: _ArrF64) -> _ArrF64: ...
def interp_m3q(x: _ArrF64, y: _ArrF64) -> _ArrF64: ...
def interp_values(init_x: _ArrF64, coeffs_flat: _ArrF64, x_query: _ArrF64) -> _ArrF64: ...  # noqa: E501
def interp_values_and_slopes(init_x: _ArrF64, coeffs_flat: _ArrF64, x_query: _ArrF64) -> tuple[_ArrF64, _ArrF64]: ...  # noqa: E501
def integrate_values(init_x: _ArrF64, coeffs_flat: _ArrF64, x_query: _ArrF64) -> _ArrF64: ...  # noqa: E501
def interp_mkbicub_db(
    x: _ArrF64, y: _ArrF64, z_flat: _ArrF64, ibcxmin: int, bcxmin: _ArrF64,
    ibcxmax: int, bcxmax: _ArrF64, ibcymin: int, bcymin: _ArrF64, ibcymax: int,
    bcymax: _ArrF64,
) -> tuple[_ArrF64, bool, bool]: ...
def interp_evbicub_db(
    xget: float, yget: float, x: _ArrF64, y: _ArrF64, ilinx: bool, iliny: bool,
    coeffs_flat: _ArrF64,
) -> list[float]: ...
def interp_evbicub_batch(
    x_q: _ArrF64, y_q: _ArrF64, x: _ArrF64, y: _ArrF64, coeffs_flat: _ArrF64,
    parallel: bool = False,
) -> _ArrF64: ...
def interp_mkbipm_db(x: _ArrF64, y: _ArrF64, z_flat: _ArrF64) -> _ArrF64: ...
def interp_evbipm_db(xget: float, yget: float, x: _ArrF64, y: _ArrF64, coeffs_flat: _ArrF64) -> float: ...  # noqa: E501
def neu_get(
    t: float | None = None, log10_t: float | None = None,
    rho: float | None = None, log10_rho: float | None = None, abar: float = 4.0,
    zbar: float = 2.0, log10_tlim: float = 7.5,
    flags: Sequence[bool] = (True, True, True, True, True),
) -> tuple[list[float], list[list[float]]]: ...
def neu_get_batch(
    t: _ArrF64, log10_t: _ArrF64, rho: _ArrF64, log10_rho: _ArrF64,
    abar: _ArrF64, zbar: _ArrF64, log10_tlim: float = 7.5,
    flags: Sequence[bool] = (True, True, True, True, True),
    parallel: bool = False,
) -> _ArrF64: ...
def set_mlt_f64(
    option: str, mixing_length_alpha: float, henyey_y_param: float,
    henyey_nu_param: float, chi_t: float, chi_rho: float, cp: float,
    grav: float, lambda_mlt: float, rho: float, p: float, t: float,
    opacity: float, gradr: float, grada: float, gradl: float,
) -> tuple[list[float], int]: ...
def set_mlt_batch(s_frac: _ArrF64, parallel: bool = False) -> _ArrF64: ...
def set_tdc_batch(s_frac: _ArrF64, parallel: bool = False) -> _ArrF64: ...

# Tier C: Coupled Linear Solvers (`mtx`)
def solve_block_tridiagonal(nvar: int, nz: int, lblk: _ArrF64, dblk: _ArrF64, ublk: _ArrF64, rhs: _ArrF64) -> _ArrF64: ...  # noqa: E501
def solve_block_tridiag_bcyclic(nvar: int, nz: int, lblk: _ArrF64, dblk: _ArrF64, ublk: _ArrF64, rhs: _ArrF64, parallel: bool = False) -> _ArrF64: ...  # noqa: E501
def block_multiply_xa(nvar: int, nz: int, lblk: _ArrF64, dblk: _ArrF64, ublk: _ArrF64, vec_x: _ArrF64) -> _ArrF64: ...  # noqa: E501

# Wave 2a: Chemical Elements, Nuclides, Solar Scales & Composition (`chem`)
def py_chem_get_element_id(name: str) -> int | None: ...
def py_chem_element_name(z: int) -> str | None: ...
def py_chem_element_long_name(z: int) -> str | None: ...
def py_chem_element_main_iso_name(z: int) -> str | None: ...
def py_chem_category_id(name: str) -> int: ...
def py_chem_category_name(id: int) -> str | None: ...
def py_chem_get_partition_fcn_indx(t9: float) -> int: ...
def py_chem_generate_nuclide_names(z: Sequence[int], a: Sequence[int]) -> list[str]: ...  # noqa: E501
def py_chem_generate_long_nuclide_names(z: Sequence[int], a: Sequence[int]) -> list[str]: ...  # noqa: E501
def py_chem_get_nuclide_index(name: str) -> int | None: ...
def py_chem_nuclide_info(chem_id: int) -> tuple[int, str, int, int, float, float, float, float, float, int]: ...  # noqa: E501
def py_chem_species_props(chem_ids: Sequence[int]) -> tuple[list[int], list[float], list[float], list[float], list[float]]: ...  # noqa: E501
def py_chem_ag89_zsol() -> float: ...
def py_chem_ag89_yesol() -> float: ...
def py_chem_xsol(name: str) -> float: ...
def py_chem_get_solar(name: str) -> tuple[float, float, float] | None: ...
def py_chem_lodders03_atom_percent(name: str) -> float: ...
def py_chem_ag89_namsol() -> list[str]: ...
def py_chem_get_stable_mass_frac(z: Sequence[int], a: Sequence[int], abun_in: Sequence[float]) -> list[float]: ...  # noqa: E501
def py_chem_solar_element_zfrac(scale: str) -> list[float]: ...
def py_chem_solar_cno_fraction(scale: str) -> float: ...
def py_chem_m_div_h(x: float, z: float, scale: str) -> float: ...
def py_chem_reaction_qtotal(num_in: int, num_out: int, reactants: Sequence[int]) -> float: ...  # noqa: E501
def py_chem_get_composition_info(chem_ids: Sequence[int], x: Sequence[float], skip_partials: bool) -> tuple[tuple[float, ...], list[float], list[float], list[float]]: ...  # noqa: E501
def py_chem_get_composition_info_zones(chem_ids: Sequence[int], x_zones: _ArrF64, parallel: bool = False) -> _ArrF64: ...  # noqa: E501

# Wave 2b: Ionization Tables & Diffusion Typical Charge (`ionization`)
def py_ion_result_names() -> list[str]: ...
def py_ion_fe56_he4_grid() -> tuple[_ArrF64, _ArrF64, _ArrF64]: ...
def py_ion_builtin_cell_grid() -> tuple[float, float, int, float, float, int, _ArrF64]: ...  # noqa: E501
def py_ion_potentials_table() -> _ArrF64: ...
def py_ion_charge_of_fe56_in_he4(log10_ne: float, log10_t: float) -> tuple[float, float, float]: ...  # noqa: E501
def py_ion_charge_of_fe56_in_he4_zones(log10_ne: _ArrF64, log10_t: _ArrF64, parallel: bool = False) -> _ArrF64: ...  # noqa: E501
def py_ion_eval_ionization(
    z: float, x: float, rho: float | None = None,
    log10_rho: float | None = None, t: float | None = None,
    log10_t: float | None = None,
) -> list[float]: ...
def py_ion_eval_ionization_dual(z: float, x: float, log10_rho: float, log10_t: float) -> tuple[_ArrF64, _ArrF64]: ...  # noqa: E501
def py_ion_eval_ionization_zones(z: _ArrF64, x: _ArrF64, log10_rho: _ArrF64, log10_t: _ArrF64, parallel: bool = False) -> _ArrF64: ...  # noqa: E501
def py_ion_chi_info(a1: float, z1: float, t: float, log10_t: float, rho: float, log10_rho: float) -> tuple[float, float, float, float]: ...  # noqa: E501
def py_ion_chi_effective(chi: float, c0: float, c1: float, c2: float, z1: float, z2: float) -> float: ...  # noqa: E501
def py_ion_eval_typical_charge(cid: int, abar: float, free_e: float, t: float, log10_t: float, rho: float, log10_rho: float) -> float: ...  # noqa: E501

# Wave 2c: Bolometric & Synthetic Stellar Photometry (`colors`)
def py_colors_column_names() -> list[str]: ...
def py_colors_filter_names() -> list[str]: ...
def py_colors_filter_curve(idx: int) -> tuple[_ArrF64, _ArrF64, float, float, float]: ...  # noqa: E501
def py_colors_builtin_grid() -> tuple[_ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrI64, _ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64]: ...  # noqa: E501
def py_colors_eval_sed(teff: float, log_g: float, metallicity: float, r: float, d: float = 3.0857e19, method: str = "Hermite") -> tuple[_ArrF64, _ArrF64]: ...  # noqa: E501
def py_colors_eval_colors(teff: float, log_g: float, metallicity: float, r: float, d: float = 3.0857e19, mag_system: str = "Vega", method: str = "Hermite") -> _ArrF64: ...  # noqa: E501
def py_colors_eval_colors_dual(teff: float, log_g: float, metallicity: float, r: float, d: float = 3.0857e19, mag_system: str = "Vega") -> tuple[_ArrF64, _ArrF64]: ...  # noqa: E501
def py_colors_eval_colors_zones(teff: _ArrF64, log_g: _ArrF64, metallicity: _ArrF64, r: _ArrF64, d: float = 3.0857e19, mag_system: str = "Vega", method: str = "Hermite", parallel: bool = True) -> _ArrF64: ...  # noqa: E501
def py_colors_simpson_integration(x: _ArrF64, y: _ArrF64) -> float: ...
def py_colors_trapezoidal_integration(x: _ArrF64, y: _ArrF64) -> float: ...
def py_colors_dilute_flux(surface_flux: _ArrF64, r: float, d: float) -> _ArrF64: ...  # noqa: E501
def py_colors_interpolate_array(x_in: _ArrF64, y_in: _ArrF64, x_out: _ArrF64) -> _ArrF64: ...  # noqa: E501
def py_colors_calculate_bolometric_phot(wavelengths: _ArrF64, fluxes: _ArrF64) -> tuple[float, float]: ...  # noqa: E501
def py_colors_calculate_synthetic(wavelengths: _ArrF64, fluxes: _ArrF64, filter_wavelengths: _ArrF64, filter_trans: _ArrF64, zero_point_flux: float) -> float: ...  # noqa: E501
def py_colors_compute_vega_zero_point(vega_wave: _ArrF64, vega_flux: _ArrF64, filt_wave: _ArrF64, filt_trans: _ArrF64) -> float: ...  # noqa: E501
def py_colors_compute_ab_zero_point(filt_wave: _ArrF64, filt_trans: _ArrF64) -> float: ...  # noqa: E501
def py_colors_compute_st_zero_point(filt_wave: _ArrF64, filt_trans: _ArrF64) -> float: ...  # noqa: E501

# Wave 3a.1: EOS Core, Blending & Fermi-Dirac Integrals (`eos`)
def eos_basic_result_names() -> list[str]: ...
def eos_is_contained(coords: _ArrF64, x: float, y: float) -> bool: ...
def eos_min_distance_to_polygon(
    coords: _ArrF64, x: float, y: float, dx_d1: float = 1.0,
    dx_d2: float = 0.0, dy_d1: float = 0.0, dy_d2: float = 1.0,
) -> tuple[float, float, float]: ...
def eos_quintic_blend(alpha0: float) -> tuple[float, float]: ...
def eos_blend_results(
    alpha_in: float, d_alpha_dlog_t_in: float, d_alpha_dlog_rho_in: float,
    linear_blend: bool, res_1: _ArrF64, d_dlnd_1: _ArrF64, d_dln_t_1: _ArrF64,
    res_2: _ArrF64, d_dlnd_2: _ArrF64, d_dln_t_2: _ArrF64,
) -> tuple[_ArrF64, _ArrF64, _ArrF64]: ...
def eos_dfermi(dk: float, eta: float, theta: float) -> tuple[float, float, float]: ...  # noqa: E501
def eos_dfermi_zones(dk: float, eta: _ArrF64, theta: _ArrF64, parallel: bool = True) -> _ArrF64: ...  # noqa: E501
def eos_compute_ion_offset(xa: _ArrF64, chem_id: list[int]) -> float: ...
def eos_compute_ion_offset_from_za(xa: _ArrF64, z: list[int], a: _ArrF64) -> float: ...  # noqa: E501
def eos_eval_ideal(temp: float, rho: float, abar: float, zbar: float, xa: _ArrF64, chem_id: list[int]) -> tuple[_ArrF64, _ArrF64, _ArrF64]: ...  # noqa: E501
def eos_eval_ideal_from_weights(temp: float, rho: float, abar: float, zbar: float, xa: _ArrF64, weights: _ArrF64, a: _ArrF64) -> tuple[_ArrF64, _ArrF64, _ArrF64]: ...  # noqa: E501
def eos_eval_ideal_zones(temp: _ArrF64, rho: _ArrF64, abar: float, zbar: float, xa: _ArrF64, weights: _ArrF64, a: _ArrF64, parallel: bool = True) -> _ArrF64: ...  # noqa: E501
def eos_eval_helm(temp: float, rho: float, abar: float, zbar: float) -> tuple[_ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64]: ...  # noqa: E501
def eos_eval_helm_zones(temp: _ArrF64, rho: _ArrF64, abar: _ArrF64, zbar: _ArrF64, parallel: bool = True) -> _ArrF64: ...  # noqa: E501
def eos_helm_cell_data(log_temp: float, din: float) -> tuple[float, float, float, float, _ArrF64, _ArrF64, _ArrF64, _ArrF64]: ...  # noqa: E501
def eos_helm_cells_batch(log_temp: _ArrF64, din: _ArrF64) -> tuple[_ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64]: ...  # noqa: E501
def eos_get_pc_alfa(
    log_rho: float, log_t: float, abar: float, zbar: float,
    use_gamma_limit: bool = True, log_rho1: float = 2.999,
    log_rho2: float = 2.8, log_t1: float = 7.6, log_t2: float = 7.7,
    log_gamma_e_all_helm: float = -1.8, log_gamma_e_all_pc: float = -1.7,
) -> tuple[float, float, float]: ...
def eos_eval_pc(
    rho: float, temp: float, abar: float, zbar: float, chem_id: list[int],
    xa: _ArrF64, include_radiation: bool = True,
    pc_gamma_start_crystal: float = 170.0, pc_gamma_full_crystal: float = 180.0,
    mass_fraction_limit: float = 1.0e-4,
) -> tuple[_ArrF64, _ArrF64, _ArrF64]: ...
def eos_eval_pc_zones(
    rho: _ArrF64, temp: _ArrF64, abar: _ArrF64, zbar: _ArrF64, xa_flat: _ArrF64,
    azion: _ArrF64, acmi: _ArrF64, include_radiation: bool = True,
    pc_gamma_start_crystal: float = 170.0, pc_gamma_full_crystal: float = 180.0,
    mass_fraction_limit: float = 1.0e-4, parallel: bool = True,
) -> _ArrF64: ...
def eos_get_skye_alfa(log_rho: float, log_t: float, abar: float, zbar: float) -> tuple[float, float, float]: ...  # noqa: E501
def eos_get_skye_alfa_simple(log_rho: float, log_t: float) -> tuple[float, float, float]: ...  # noqa: E501
def eos_eval_skye(
    rho: float, temp: float, abar: float, zbar: float, chem_id: list[int],
    xa: _ArrF64, skye_use_ion_offsets: bool = True,
    skye_solid_mixing_rule: str = "PC", mass_fraction_limit: float = 1.0e-4,
) -> tuple[_ArrF64, _ArrF64, _ArrF64]: ...
def eos_eval_skye_zones(
    rho: _ArrF64, temp: _ArrF64, abar: _ArrF64, zbar: _ArrF64, xa_flat: _ArrF64,
    z: list[int], acmi: _ArrF64, a: _ArrF64, skye_use_ion_offsets: bool = True,
    skye_solid_mixing_rule: str = "PC", mass_fraction_limit: float = 1.0e-4,
    parallel: bool = True,
) -> _ArrF64: ...
def eos_eval_dt_get(
    rho: float, temp: float, chem_id: list[int], xa: _ArrF64,
    use_skye: bool = True, use_pc: bool = False,
    pc_use_gamma_limit: bool = True, skye_solid_mixing_rule: str = "PC",
    use_cms: bool = False, use_free_eos: bool = True,
    use_opal_scvh: bool = True,
) -> tuple[_ArrF64, _ArrF64, _ArrF64]: ...
def eos_eval_pt_get(
    pgas: float, temp: float, chem_id: list[int], xa: _ArrF64,
    use_skye: bool = True, use_pc: bool = False,
    pc_use_gamma_limit: bool = True, skye_solid_mixing_rule: str = "PC",
    use_cms: bool = False, use_free_eos: bool = True,
    use_opal_scvh: bool = True,
) -> tuple[float, float, float, float, _ArrF64, _ArrF64, _ArrF64]: ...
def eos_eval_opal_scvh(
    rho: float, temp: float, chem_id: list[int], xa: _ArrF64,
) -> tuple[_ArrF64, _ArrF64, _ArrF64]: ...
def eos_eval_free_eos(
    rho: float, temp: float, chem_id: list[int], xa: _ArrF64,
) -> tuple[_ArrF64, _ArrF64, _ArrF64]: ...
def eos_eval_cms(
    rho: float, temp: float, chem_id: list[int], xa: _ArrF64,
) -> tuple[_ArrF64, _ArrF64, _ArrF64]: ...
def eos_eval_dt_get_t(
    rho: float, which_other: int, other_value: float, chem_id: list[int],
    xa: _ArrF64, log_t_guess: float = 6.0, use_skye: bool = True,
    use_pc: bool = False, use_cms: bool = False, use_free_eos: bool = True,
    use_opal_scvh: bool = True,
) -> tuple[float, float, _ArrF64, _ArrF64, _ArrF64]: ...
def eos_eval_dt_get_rho(
    temp: float, which_other: int, other_value: float, chem_id: list[int],
    xa: _ArrF64, log_rho_guess: float = 0.0, use_skye: bool = True,
    use_pc: bool = False, use_cms: bool = False, use_free_eos: bool = True,
    use_opal_scvh: bool = True,
) -> tuple[float, float, _ArrF64, _ArrF64, _ArrF64]: ...
def eos_eval_pt_get_t(
    pgas: float, which_other: int, other_value: float, chem_id: list[int],
    xa: _ArrF64, log_t_guess: float = 6.0, use_skye: bool = True,
    use_pc: bool = False, use_cms: bool = False, use_free_eos: bool = True,
    use_opal_scvh: bool = True,
) -> tuple[float, float, _ArrF64, _ArrF64, _ArrF64]: ...
def eos_eval_pt_get_pgas(
    temp: float, which_other: int, other_value: float, chem_id: list[int],
    xa: _ArrF64, log_pgas_guess: float = 10.0, use_skye: bool = True,
    use_pc: bool = False, use_cms: bool = False, use_free_eos: bool = True,
    use_opal_scvh: bool = True,
) -> tuple[float, float, _ArrF64, _ArrF64, _ArrF64]: ...
def eos_eval_gamma_de_get(
    abar: float, zbar: float, rho: float, energy: float,
    gamma: float = 1.6666666666666667,
) -> tuple[float, float, _ArrF64, _ArrF64, _ArrF64]: ...
def kap_get_results_rust(
    z: float, x: float, zbase: float, xc: float, xn: float, xo: float,
    xne: float, log_rho: float, log_t: float, lnfree_e: float,
    d_lnfree_e_dln_rho: float, d_lnfree_e_dln_t: float, eta: float,
    d_eta_dln_rho: float, d_eta_dln_t: float, zbar: float = 1.0,
    kap_option: str = "gs98", kap_co_option: str = "gs98_co",
    kap_low_t_option: str = "lowT_fa05_gs98",
    kap_blend_log_t_upper_bdy: float = 3.88,
    kap_blend_log_t_lower_bdy: float = 3.80,
    cubic_interpolation_in_x: bool = False,
    cubic_interpolation_in_z: bool = False,
    include_electron_conduction: bool = True,
    use_blouin_conductive_opacities: bool = True,
    use_zbase_for_type1: bool = False, use_type2_opacities: bool = False,
    kap_type2_full_off_x: float = 1.0e-3, kap_type2_full_on_x: float = 1.0e-6,
    kap_type2_full_off_dz: float = 0.001, kap_type2_full_on_dz: float = 0.01,
) -> tuple[float, float, float, list[float]]: ...
def kap_get_rust(
    chem_id: npt.NDArray[np.uintp], xa: _ArrF64, log_rho: float, log_t: float,
    lnfree_e: float, d_lnfree_e_dln_rho: float, d_lnfree_e_dln_t: float,
    eta: float, d_eta_dln_rho: float, d_eta_dln_t: float, zbase: float = 0.02,
    kap_option: str = "gs98", kap_co_option: str = "gs98_co",
    kap_low_t_option: str = "lowT_fa05_gs98",
    kap_blend_log_t_upper_bdy: float = 3.88,
    kap_blend_log_t_lower_bdy: float = 3.80,
    cubic_interpolation_in_x: bool = False,
    cubic_interpolation_in_z: bool = False,
    include_electron_conduction: bool = True,
    use_blouin_conductive_opacities: bool = True,
    use_zbase_for_type1: bool = False, use_type2_opacities: bool = False,
    kap_type2_full_off_x: float = 1.0e-3, kap_type2_full_on_x: float = 1.0e-6,
    kap_type2_full_off_dz: float = 0.001, kap_type2_full_on_dz: float = 0.01,
) -> tuple[float, float, float, list[float]]: ...
def kap_get_elect_cond_opacity_rust(
    zbar: float, log_rho: float, log_t: float, use_blouin: bool = True,
) -> tuple[float, float, float]: ...
def kap_get_compton_opacity_rust(
    rho: float, t: float, lnfree_e: float, d_lnfree_e_dln_rho: float,
    d_lnfree_e_dln_t: float, eta: float, d_eta_dln_rho: float,
    d_eta_dln_t: float,
) -> tuple[float, float, float]: ...
def kap_get_fixed_metal_rust(
    z: float, x: float, log_rho: float, log_t: float, prefix: str = "gs98",
    cubic_x: bool = False, cubic_z: bool = False,
) -> tuple[float, float, float]: ...
def kap_get_co_rust(
    zbase: float, x: float, dxc: float, dxo: float, log_rho: float,
    log_t: float, kap_co_option: str = "gn93_co", cubic_x: bool = False,
    cubic_z: bool = False,
) -> tuple[float, float, float]: ...
def kap_get_aesopus_rust(
    zbase: float, x: float, xc: float, xn: float, xo: float, log_rho: float,
    log_t: float,
) -> tuple[float, float, float]: ...
def kap_get_kapcn_rust(
    zbase: float, x: float, fc: float, fn_val: float, log_rho: float,
    log_t: float,
) -> tuple[float, float, float]: ...
def kap_get_zones_rust(
    chem_id: npt.NDArray[np.uintp], xa_zones: _ArrF64, log_rho_zones: _ArrF64,
    log_t_zones: _ArrF64, lnfree_e_zones: _ArrF64,
    d_lnfree_e_dln_rho_zones: _ArrF64, d_lnfree_e_dln_t_zones: _ArrF64,
    eta_zones: _ArrF64, d_eta_dln_rho_zones: _ArrF64,
    d_eta_dln_t_zones: _ArrF64, zbase: float = 0.02, kap_option: str = "gs98",
    kap_co_option: str = "gs98_co", kap_low_t_option: str = "lowT_fa05_gs98",
    kap_blend_log_t_upper_bdy: float = 3.88,
    kap_blend_log_t_lower_bdy: float = 3.80,
    cubic_interpolation_in_x: bool = False,
    cubic_interpolation_in_z: bool = False,
    include_electron_conduction: bool = True,
    use_blouin_conductive_opacities: bool = True,
    use_zbase_for_type1: bool = False, use_type2_opacities: bool = False,
    kap_type2_full_off_x: float = 1.0e-3, kap_type2_full_on_x: float = 1.0e-6,
    kap_type2_full_off_dz: float = 0.001, kap_type2_full_on_dz: float = 0.01,
    parallel: bool = True,
) -> _ArrF64: ...
def rates_reaction_id_by_name(name: str) -> int: ...
def rates_reaction_name_by_id(ir: int) -> str: ...
def rates_reaction_info(
    ir: int,
) -> tuple[str, int, bool, str, int, int, int, float, float, float]: ...
def rates_reaclib_terms(
    handle: str,
) -> tuple[
    bool, list[tuple[int, int, int, list[float], float, float, list[float]]]
]: ...
def rates_eval_reaclib_handle(
    handle: str, t9: float, check_e_flag: bool = True,
) -> tuple[float, float, float, float]: ...
def rates_set_raw_rate(
    ir: int, temp: float, log_t: float | None = None,
) -> float: ...
def rates_set_raw_rates(
    irs: npt.NDArray[np.uintp], temp: float, log_t: float | None = None,
) -> _ArrF64: ...
def rates_set_raw_rates_zones(
    irs: npt.NDArray[np.uintp], temps: _ArrF64, log_ts: _ArrF64 | None = None,
) -> _ArrF64: ...
def rates_eval_fl_epsnuc_3alf(
    t: float, rho: float, y: float, ue: float,
) -> float: ...
def rates_eval_rate_table(ir: int, temp: float) -> tuple[float, float]: ...
def rates_screen_pair(
    mode: str, temp: float, den: float, zbar: float, abar: float, z2bar: float,
    a1: float, z1: float, a2: float, z2: float,
) -> tuple[float, float, float]: ...
def rates_screen_pair_zones(
    mode: str, temps: _ArrF64, dens: _ArrF64, zbars: _ArrF64, abars: _ArrF64,
    z2bars: _ArrF64, a1: float, z1: float, a2: float, z2: float,
) -> _ArrF64: ...
def rates_eval_coulomb(
    which_mui: str, which_vs: str, temp: float, den: float, zbar: float,
    abar: float, z1: float, z2: float,
) -> tuple[float, float, float, float]: ...
def rates_eval_weak_pair(
    lhs_name: str, rhs_name: str, t9: float, ye_rho: float,
    use_suzuki: bool = False,
) -> list[float]: ...
def rates_eval_ecapture_pair(
    lhs_name: str, rhs_name: str, t9: float, ye_rho: float, eta_k: float,
    which_mui: str = "pcr2009", which_vs: str = "itoh2002",
    zbar: float = 10.909090909090908, abar: float = 21.818181818181817,
) -> list[float]: ...
def rates_weak_info_by_names(
    lhs_name: str, rhs_name: str,
) -> tuple[float, float]: ...
def rates_eval_weak_table_index(
    idx: int, t9: float, l_ye_rho: float,
) -> tuple[str, str, list[float]]: ...
def atm_eval_t_tau(t_tau_id: int, tau: float, teff: float) -> float: ...
def atm_eval_t_tau_dq_dtau(t_tau_id: int, tau: float) -> float: ...
def atm_get_t_tau_base(t_tau_id: int) -> float: ...
def atm_eval_e2(x: float) -> tuple[float, float]: ...
def atm_eval_paczynski_gradr(
    t: float, p: float, rho: float, tau: float, kap: float, l: float, m: float,
    r: float, cgrav: float,
) -> float: ...
def atm_eval_table(
    l: float, r: float, m: float, cgrav: float, id: int, z: float,
    skip_partials: bool, teff: float,
) -> tuple[float, ...]: ...
def atm_get_table_alfa_beta(
    l: float, teff: float, r: float, m: float, cgrav: float, id: int,
) -> tuple[float, float]: ...
def atm_get_table_base(id: int) -> float: ...
def atm_eval_t_tau_uniform(
    tau_surf: float, l: float, r: float, m: float, cgrav: float,
    kap_guess: float, pextra_factor: float, t_tau_id: int, errtol: float,
    max_iters: int, skip_partials: bool, teff: float,
    chem_id: Sequence[int] | None = None, xa: _ArrF64 | None = None,
    zbase: float = 0.02, low_t_option: str = "low_t_fa05_gs98",
) -> tuple[float, ...]: ...
def atm_eval_t_tau_varying(
    tau_surf: float, l: float, r: float, m: float, cgrav: float,
    t_tau_id: int, errtol: float, max_steps: int, skip_partials: bool,
    teff: float, chem_id: Sequence[int], xa: _ArrF64, zbase: float = 0.02,
    low_t_option: str = "low_t_fa05_gs98",
) -> tuple[float, ...]: ...
def atm_eval_irradiated(
    l: float, r: float, m: float, cgrav: float, t_eq: float, p_surf: float,
    kap_guess: float, kap_v: float, gamma: float, errtol: float,
    max_iters: int, skip_partials: bool, chem_id: Sequence[int] | None = None,
    xa: _ArrF64 | None = None, zbase: float = 0.02,
    low_t_option: str = "low_t_fa05_gs98",
) -> tuple[float, ...]: ...
def net_list_networks() -> list[str]: ...
def net_info(
    net_name: str,
) -> tuple[int, int, list[int], list[str], list[str]]: ...
def net_eval(
    net_name: str, x: _ArrF64, temp: float, log_t: float, rho: float,
    log_rho: float, abar: float, zbar: float, z2bar: float, ye: float,
    eta: float = 0.0, d_eta_dlnt: float = 0.0, d_eta_dlnrho: float = 0.0,
    rate_factors: _ArrF64 | None = None, weak_rate_factor: float = 1.0,
    screening_mode: str = "extended", skip_jacobian: bool = False,
) -> tuple[
    float, float, float, _ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64,
    float, _ArrF64, _ArrF64,
]: ...
def net_burn_1_zone(
    net_name: str, t_start: float, t_end: float, starting_x: _ArrF64,
    times: _ArrF64, log10_ts: _ArrF64, log10_rhos: _ArrF64, etas: _ArrF64,
    rate_factors: _ArrF64 | None = None, weak_rate_factor: float = 1.0,
    screening_mode: str = "extended", stptry_in: float = 0.0,
    max_steps: int = 10000, eps: float = 1.0e-8, odescal: float = 1.0e-12,
) -> tuple[_ArrF64, _ArrF64, float, float, int, int, int, int, int]: ...
def net_burn_1_zone_const_p(
    net_name: str, starting_temp: float, starting_x: _ArrF64, t_end: float,
    pressure: float, solver: str = "rodas4", clip: bool = False,
    rate_factors: _ArrF64 | None = None, weak_rate_factor: float = 1.0,
    screening_mode: str = "extended", h0: float = 1.0e-6,
    max_step_size: float = 1.0, max_steps: int = 1000, rtol: float = 1.0e-7,
    atol: float = 1.0e-8, x_min: float = -1.0e-10,
) -> tuple[
    _ArrF64, float, float, float, float, float, int, int, int, int, int
]: ...
def eos_eval_opal_scvh_zones(
    rho: _ArrF64, temp: _ArrF64, chem_id: Sequence[int], xa: _ArrF64,
    parallel: bool = True,
) -> _ArrF64: ...
def eos_opal_scvh_cells_batch(
    log_rho: _ArrF64, log_temp: _ArrF64, ix_1based: int = 5,
    iz_1based: int = 2,
) -> tuple[
    _ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64
]: ...
def atm_table_photosphere_zsol_spline() -> tuple[
    _ArrF64, _ArrF64, bool, bool, _ArrF64
]: ...
def kap_get_results_zones_rust(
    z: float, x: float, zbase: float, xc: float, xn: float, xo: float,
    xne: float, log_rho_zones: _ArrF64, log_t_zones: _ArrF64,
    lnfree_e: float, d_lnfree_e_dln_rho: float, d_lnfree_e_dln_t: float,
    eta: float, d_eta_dln_rho: float, d_eta_dln_t: float,
    zbar: float = 1.0, kap_option: str = "gs98",
    kap_co_option: str = "gs98_co", kap_low_t_option: str = "lowT_fa05_gs98",
    kap_blend_log_t_upper_bdy: float = 3.88,
    kap_blend_log_t_lower_bdy: float = 3.80,
    cubic_interpolation_in_x: bool = False,
    cubic_interpolation_in_z: bool = False,
    include_electron_conduction: bool = True,
    use_blouin_conductive_opacities: bool = True,
    use_zbase_for_type1: bool = False, use_type2_opacities: bool = False,
    kap_type2_full_off_x: float = 1.0e-3, kap_type2_full_on_x: float = 1.0e-6,
    kap_type2_full_off_dz: float = 0.001, kap_type2_full_on_dz: float = 0.01,
    parallel: bool = True,
) -> _ArrF64: ...
def rates_eval_reaclib_screen_zones(
    handle: str, mode: str, temps: _ArrF64, dens: _ArrF64, zbar: float,
    abar: float, z2bar: float, a1: float, z1: float, a2: float, z2: float,
    parallel: bool = True,
) -> _ArrF64: ...
def atm_eval_table_zones(
    l: _ArrF64, r: _ArrF64, m: _ArrF64, cgrav: float, id: int, z: float,
    skip_partials: bool, teff: _ArrF64, parallel: bool = True,
) -> _ArrF64: ...
def net_eval_zones(
    net_name: str, x: _ArrF64, temps: _ArrF64, log_ts: _ArrF64,
    rhos: _ArrF64, log_rhos: _ArrF64, abar: float, zbar: float, z2bar: float,
    ye: float, eta: float = 0.0, d_eta_dlnt: float = 0.0,
    d_eta_dlnrho: float = 0.0, weak_rate_factor: float = 1.0,
    screening_mode: str = "extended", skip_jacobian: bool = False,
    parallel: bool = True,
) -> _ArrF64: ...
def star_yrs_for_init_timestep(m_msun: float) -> float: ...
def star_interp_q(dq1: _ArrF64, v1: _ArrF64, dq2: _ArrF64) -> _ArrF64: ...

class RustStarModel:
    nz: int
    species: int
    model_number: int
    star_age: float
    dt: float
    mstar: float
    teff: float
    ln_peos_surf: float
    ln_t_surf: float
    iso_names: list[str]
    chem_id: list[int]
    lnd: _ArrF64
    ln_t: _ArrF64
    ln_r: _ArrF64
    lum: _ArrF64
    dq: _ArrF64
    q: _ArrF64
    m: _ArrF64
    dm: _ArrF64
    dm_bar: _ArrF64
    r: _ArrF64
    rmid: _ArrF64
    grav: _ArrF64
    xa: _ArrF64
    rho: _ArrF64
    t: _ArrF64
    pgas: _ArrF64
    prad: _ArrF64
    peos: _ArrF64
    ln_peos: _ArrF64
    energy: _ArrF64
    entropy: _ArrF64
    grada: _ArrF64
    cp: _ArrF64
    cv: _ArrF64
    chi_rho: _ArrF64
    chi_t: _ArrF64
    gamma1: _ArrF64
    eta: _ArrF64
    mu: _ArrF64
    csound: _ArrF64
    opacity: _ArrF64
    tau: _ArrF64
    non_nuc_neu: _ArrF64
    eps_nuc: _ArrF64
    dxdt_nuc: _ArrF64
    gradt: _ArrF64
    gradr: _ArrF64
    gradl: _ArrF64
    scale_height: _ArrF64
    mlt_vc: _ArrF64
    mlt_d: _ArrF64
    mlt_cdc: _ArrF64
    brunt_b: _ArrF64
    brunt_n2: _ArrF64
    mixing_type: _ArrI32
    @staticmethod
    def from_zams(
        initial_mass: float = 1.0,
        initial_z: float = 0.02,
        initial_y: float = -1.0,
    ) -> RustStarModel: ...
    @staticmethod
    def from_pre_ms(
        initial_mass: float = 1.0,
        initial_z: float = 0.02,
        initial_y: float = -1.0,
        t_c: float = 9.0e5,
        d_log10_p: float = 0.02,
    ) -> RustStarModel: ...
    @staticmethod
    def from_mod_str(text: str) -> RustStarModel: ...
    @staticmethod
    def from_photo_bytes(bytes: bytes) -> RustStarModel: ...
    def to_mod_str(self) -> str: ...
    def to_photo_bytes(self) -> bytes: ...
    def eval_hydro_vars(
        self, skip_net: bool = False, skip_net_jacobian: bool = False
    ) -> None: ...
    def set_primary_state(
        self,
        lnd: _ArrF64,
        ln_t: _ArrF64,
        ln_r: _ArrF64,
        lum: _ArrF64,
        dq: _ArrF64,
        xa: _ArrF64,
    ) -> None: ...

# fmt: on
