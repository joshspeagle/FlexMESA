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
from typing import Any

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
def eos_eval_opal_scvh(rho: float, temp: float, chem_id: list[int], xa: _ArrF64) -> tuple[_ArrF64, _ArrF64, _ArrF64]: ...  # noqa: E501
def eos_eval_free_eos(rho: float, temp: float, chem_id: list[int], xa: _ArrF64) -> tuple[_ArrF64, _ArrF64, _ArrF64]: ...  # noqa: E501
def eos_eval_cms(rho: float, temp: float, chem_id: list[int], xa: _ArrF64) -> tuple[_ArrF64, _ArrF64, _ArrF64]: ...  # noqa: E501
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
def kap_get_results_rust(z: float, x: float, zbase: float, xc: float, xn: float, xo: float, xne: float, log_rho: float, log_t: float, lnfree_e: float, d_lnfree_e_dln_rho: float, d_lnfree_e_dln_t: float, eta: float, d_eta_dln_rho: float, d_eta_dln_t: float, zbar: float = 1.0, kap_option: str = "gs98", kap_co_option: str = "gs98_co", kap_low_t_option: str = "lowT_fa05_gs98", kap_blend_log_t_upper_bdy: float = 3.88, kap_blend_log_t_lower_bdy: float = 3.80, cubic_interpolation_in_x: bool = False, cubic_interpolation_in_z: bool = False, include_electron_conduction: bool = True, use_blouin_conductive_opacities: bool = True, use_zbase_for_type1: bool = False, use_type2_opacities: bool = False, kap_type2_full_off_x: float = 1.0e-3, kap_type2_full_on_x: float = 1.0e-6, kap_type2_full_off_dz: float = 0.001, kap_type2_full_on_dz: float = 0.01) -> tuple[float, float, float, list[float]]: ...  # noqa: E501
def kap_get_rust(chem_id: npt.NDArray[np.uintp], xa: _ArrF64, log_rho: float, log_t: float, lnfree_e: float, d_lnfree_e_dln_rho: float, d_lnfree_e_dln_t: float, eta: float, d_eta_dln_rho: float, d_eta_dln_t: float, zbase: float = 0.02, kap_option: str = "gs98", kap_co_option: str = "gs98_co", kap_low_t_option: str = "lowT_fa05_gs98", kap_blend_log_t_upper_bdy: float = 3.88, kap_blend_log_t_lower_bdy: float = 3.80, cubic_interpolation_in_x: bool = False, cubic_interpolation_in_z: bool = False, include_electron_conduction: bool = True, use_blouin_conductive_opacities: bool = True, use_zbase_for_type1: bool = False, use_type2_opacities: bool = False, kap_type2_full_off_x: float = 1.0e-3, kap_type2_full_on_x: float = 1.0e-6, kap_type2_full_off_dz: float = 0.001, kap_type2_full_on_dz: float = 0.01) -> tuple[float, float, float, list[float]]: ...  # noqa: E501
def kap_get_elect_cond_opacity_rust(zbar: float, log_rho: float, log_t: float, use_blouin: bool = True) -> tuple[float, float, float]: ...  # noqa: E501
def kap_get_compton_opacity_rust(rho: float, t: float, lnfree_e: float, d_lnfree_e_dln_rho: float, d_lnfree_e_dln_t: float, eta: float, d_eta_dln_rho: float, d_eta_dln_t: float) -> tuple[float, float, float]: ...  # noqa: E501
def kap_get_fixed_metal_rust(z: float, x: float, log_rho: float, log_t: float, prefix: str = "gs98", cubic_x: bool = False, cubic_z: bool = False) -> tuple[float, float, float]: ...  # noqa: E501
def kap_get_co_rust(zbase: float, x: float, dxc: float, dxo: float, log_rho: float, log_t: float, kap_co_option: str = "gn93_co", cubic_x: bool = False, cubic_z: bool = False) -> tuple[float, float, float]: ...  # noqa: E501
def kap_get_aesopus_rust(zbase: float, x: float, xc: float, xn: float, xo: float, log_rho: float, log_t: float) -> tuple[float, float, float]: ...  # noqa: E501
def kap_get_kapcn_rust(zbase: float, x: float, fc: float, fn_val: float, log_rho: float, log_t: float) -> tuple[float, float, float]: ...  # noqa: E501
def kap_get_zones_rust(chem_id: npt.NDArray[np.uintp], xa_zones: _ArrF64, log_rho_zones: _ArrF64, log_t_zones: _ArrF64, lnfree_e_zones: _ArrF64, d_lnfree_e_dln_rho_zones: _ArrF64, d_lnfree_e_dln_t_zones: _ArrF64, eta_zones: _ArrF64, d_eta_dln_rho_zones: _ArrF64, d_eta_dln_t_zones: _ArrF64, zbase: float = 0.02, kap_option: str = "gs98", kap_co_option: str = "gs98_co", kap_low_t_option: str = "lowT_fa05_gs98", kap_blend_log_t_upper_bdy: float = 3.88, kap_blend_log_t_lower_bdy: float = 3.80, cubic_interpolation_in_x: bool = False, cubic_interpolation_in_z: bool = False, include_electron_conduction: bool = True, use_blouin_conductive_opacities: bool = True, use_zbase_for_type1: bool = False, use_type2_opacities: bool = False, kap_type2_full_off_x: float = 1.0e-3, kap_type2_full_on_x: float = 1.0e-6, kap_type2_full_off_dz: float = 0.001, kap_type2_full_on_dz: float = 0.01, parallel: bool = True) -> _ArrF64: ...  # noqa: E501
def rates_reaction_id_by_name(name: str) -> int: ...
def rates_reaction_name_by_id(ir: int) -> str: ...
def rates_reaction_info(ir: int) -> tuple[str, int, bool, str, int, int, int, float, float, float]: ...  # noqa: E501
def rates_reaclib_terms(handle: str) -> tuple[bool, list[tuple[int, int, int, list[float], float, float, list[float]]]]: ...  # noqa: E501
def rates_eval_reaclib_handle(handle: str, t9: float, check_e_flag: bool = True) -> tuple[float, float, float, float]: ...  # noqa: E501
def rates_set_raw_rate(ir: int, temp: float, log_t: float | None = None) -> float: ...  # noqa: E501
def rates_set_raw_rates(irs: npt.NDArray[np.uintp], temp: float, log_t: float | None = None) -> _ArrF64: ...  # noqa: E501
def rates_set_raw_rates_zones(irs: npt.NDArray[np.uintp], temps: _ArrF64, log_ts: _ArrF64 | None = None) -> _ArrF64: ...  # noqa: E501
def rates_eval_fl_epsnuc_3alf(t: float, rho: float, y: float, ue: float) -> float: ...  # noqa: E501
def rates_eval_rate_table(ir: int, temp: float) -> tuple[float, float]: ...
def rates_screen_pair(mode: str, temp: float, den: float, zbar: float, abar: float, z2bar: float, a1: float, z1: float, a2: float, z2: float) -> tuple[float, float, float]: ...  # noqa: E501
def rates_screen_pair_zones(mode: str, temps: _ArrF64, dens: _ArrF64, zbars: _ArrF64, abars: _ArrF64, z2bars: _ArrF64, a1: float, z1: float, a2: float, z2: float) -> _ArrF64: ...  # noqa: E501
def rates_eval_coulomb(which_mui: str, which_vs: str, temp: float, den: float, zbar: float, abar: float, z1: float, z2: float) -> tuple[float, float, float, float]: ...  # noqa: E501
def rates_eval_weak_pair(lhs_name: str, rhs_name: str, t9: float, ye_rho: float, use_suzuki: bool = False) -> list[float]: ...  # noqa: E501
def rates_eval_ecapture_pair(lhs_name: str, rhs_name: str, t9: float, ye_rho: float, eta_k: float, which_mui: str = "pcr2009", which_vs: str = "itoh2002", zbar: float = 10.909090909090908, abar: float = 21.818181818181817) -> list[float]: ...  # noqa: E501
def rates_weak_info_by_names(lhs_name: str, rhs_name: str) -> tuple[float, float]: ...  # noqa: E501
def rates_eval_weak_table_index(idx: int, t9: float, l_ye_rho: float) -> tuple[str, str, list[float]]: ...  # noqa: E501
def atm_eval_t_tau(t_tau_id: int, tau: float, teff: float) -> float: ...
def atm_eval_t_tau_dq_dtau(t_tau_id: int, tau: float) -> float: ...
def atm_get_t_tau_base(t_tau_id: int) -> float: ...
def atm_eval_e2(x: float) -> tuple[float, float]: ...
def atm_eval_paczynski_gradr(t: float, p: float, rho: float, tau: float, kap: float, l: float, m: float, r: float, cgrav: float) -> float: ...  # noqa: E501
def atm_eval_table(l: float, r: float, m: float, cgrav: float, id: int, z: float, skip_partials: bool, teff: float) -> tuple[float, ...]: ...  # noqa: E501
def atm_get_table_alfa_beta(l: float, teff: float, r: float, m: float, cgrav: float, id: int) -> tuple[float, float]: ...  # noqa: E501
def atm_get_table_base(id: int) -> float: ...
def atm_eval_t_tau_uniform(tau_surf: float, l: float, r: float, m: float, cgrav: float, kap_guess: float, pextra_factor: float, t_tau_id: int, errtol: float, max_iters: int, skip_partials: bool, teff: float, chem_id: Sequence[int] | None = None, xa: _ArrF64 | None = None, zbase: float = 0.02, low_t_option: str = "low_t_fa05_gs98") -> tuple[float, ...]: ...  # noqa: E501
def atm_eval_t_tau_varying(tau_surf: float, l: float, r: float, m: float, cgrav: float, t_tau_id: int, errtol: float, max_steps: int, skip_partials: bool, teff: float, chem_id: Sequence[int], xa: _ArrF64, zbase: float = 0.02, low_t_option: str = "low_t_fa05_gs98") -> tuple[float, ...]: ...  # noqa: E501
def atm_eval_irradiated(l: float, r: float, m: float, cgrav: float, t_eq: float, p_surf: float, kap_guess: float, kap_v: float, gamma: float, errtol: float, max_iters: int, skip_partials: bool, chem_id: Sequence[int] | None = None, xa: _ArrF64 | None = None, zbase: float = 0.02, low_t_option: str = "low_t_fa05_gs98") -> tuple[float, ...]: ...  # noqa: E501
def net_list_networks() -> list[str]: ...
def net_info(net_name: str) -> tuple[int, int, list[int], list[str], list[str]]: ...  # noqa: E501
def net_eval(net_name: str, x: _ArrF64, temp: float, log_t: float, rho: float, log_rho: float, abar: float, zbar: float, z2bar: float, ye: float, eta: float = 0.0, d_eta_dlnt: float = 0.0, d_eta_dlnrho: float = 0.0, rate_factors: _ArrF64 | None = None, weak_rate_factor: float = 1.0, screening_mode: str = "extended", skip_jacobian: bool = False) -> tuple[float, float, float, _ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64, float, _ArrF64, _ArrF64]: ...  # noqa: E501
def net_burn_1_zone(net_name: str, t_start: float, t_end: float, starting_x: _ArrF64, times: _ArrF64, log10_ts: _ArrF64, log10_rhos: _ArrF64, etas: _ArrF64, rate_factors: _ArrF64 | None = None, weak_rate_factor: float = 1.0, screening_mode: str = "extended", stptry_in: float = 0.0, max_steps: int = 10000, eps: float = 1.0e-8, odescal: float = 1.0e-12) -> tuple[_ArrF64, _ArrF64, float, float, int, int, int, int, int]: ...  # noqa: E501
def net_burn_1_zone_const_p(net_name: str, starting_temp: float, starting_x: _ArrF64, t_end: float, pressure: float, solver: str = "rodas4", clip: bool = False, rate_factors: _ArrF64 | None = None, weak_rate_factor: float = 1.0, screening_mode: str = "extended", h0: float = 1.0e-6, max_step_size: float = 1.0, max_steps: int = 1000, rtol: float = 1.0e-7, atol: float = 1.0e-8, x_min: float = -1.0e-10) -> tuple[_ArrF64, float, float, float, float, float, int, int, int, int, int]: ...  # noqa: E501
def eos_eval_opal_scvh_zones(rho: _ArrF64, temp: _ArrF64, chem_id: Sequence[int], xa: _ArrF64, parallel: bool = True) -> _ArrF64: ...  # noqa: E501
def eos_opal_scvh_cells_batch(log_rho: _ArrF64, log_temp: _ArrF64, ix_1based: int = 5, iz_1based: int = 2) -> tuple[_ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64]: ...  # noqa: E501
def atm_table_photosphere_zsol_spline() -> tuple[_ArrF64, _ArrF64, bool, bool, _ArrF64]: ...  # noqa: E501
def kap_get_results_zones_rust(z: float, x: float, zbase: float, xc: float, xn: float, xo: float, xne: float, log_rho_zones: _ArrF64, log_t_zones: _ArrF64, lnfree_e: float, d_lnfree_e_dln_rho: float, d_lnfree_e_dln_t: float, eta: float, d_eta_dln_rho: float, d_eta_dln_t: float, zbar: float = 1.0, kap_option: str = "gs98", kap_co_option: str = "gs98_co", kap_low_t_option: str = "lowT_fa05_gs98", kap_blend_log_t_upper_bdy: float = 3.88, kap_blend_log_t_lower_bdy: float = 3.80, cubic_interpolation_in_x: bool = False, cubic_interpolation_in_z: bool = False, include_electron_conduction: bool = True, use_blouin_conductive_opacities: bool = True, use_zbase_for_type1: bool = False, use_type2_opacities: bool = False, kap_type2_full_off_x: float = 1.0e-3, kap_type2_full_on_x: float = 1.0e-6, kap_type2_full_off_dz: float = 0.001, kap_type2_full_on_dz: float = 0.01, parallel: bool = True) -> _ArrF64: ...  # noqa: E501
def rates_eval_reaclib_screen_zones(handle: str, mode: str, temps: _ArrF64, dens: _ArrF64, zbar: float, abar: float, z2bar: float, a1: float, z1: float, a2: float, z2: float, parallel: bool = True) -> _ArrF64: ...  # noqa: E501
def atm_eval_table_zones(l: _ArrF64, r: _ArrF64, m: _ArrF64, cgrav: float, id: int, z: float, skip_partials: bool, teff: _ArrF64, parallel: bool = True) -> _ArrF64: ...  # noqa: E501
def net_eval_zones(net_name: str, x: _ArrF64, temps: _ArrF64, log_ts: _ArrF64, rhos: _ArrF64, log_rhos: _ArrF64, abar: float, zbar: float, z2bar: float, ye: float, eta: float = 0.0, d_eta_dlnt: float = 0.0, d_eta_dlnrho: float = 0.0, weak_rate_factor: float = 1.0, screening_mode: str = "extended", skip_jacobian: bool = False, parallel: bool = True) -> _ArrF64: ...  # noqa: E501
def star_yrs_for_init_timestep(m_msun: float) -> float: ...
def star_interp_q(dq1: _ArrF64, v1: _ArrF64, dq2: _ArrF64) -> _ArrF64: ...

class RustStarModel:
    nz: int
    species: int
    model_number: int
    star_age: float
    dt: float
    dt_next: float
    dt_old: float
    mstar: float
    teff: float
    ln_peos_surf: float
    ln_t_surf: float
    iso_names: list[str]
    chem_id: list[int]
    lnd: _ArrF64
    lnd_start: _ArrF64
    ln_t: _ArrF64
    ln_r: _ArrF64
    lum: _ArrF64
    dq: _ArrF64
    q: _ArrF64
    m: _ArrF64
    dm: _ArrF64
    dm_bar: _ArrF64
    xq: _ArrF64
    r: _ArrF64
    rmid: _ArrF64
    grav: _ArrF64
    xa: _ArrF64
    xh_mass: _ArrF64
    yhe_mass: _ArrF64
    rho: _ArrF64
    t: _ArrF64
    pgas: _ArrF64
    prad: _ArrF64
    peos: _ArrF64
    ln_peos: _ArrF64
    energy: _ArrF64
    energy_start: _ArrF64
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
    eps_grav: _ArrF64
    eps_mdot: _ArrF64
    dxdt_nuc: _ArrF64
    gradt: _ArrF64
    gradr: _ArrF64
    gradl: _ArrF64
    scale_height: _ArrF64
    mlt_vc: _ArrF64
    mlt_d: _ArrF64
    mlt_cdc: _ArrF64
    d_mix: _ArrF64
    conv_vel: _ArrF64
    sig: _ArrF64
    omega: _ArrF64
    j_rot: _ArrF64
    brunt_b: _ArrF64
    brunt_n2: _ArrF64
    mixing_type: _ArrI32
    @staticmethod
    def from_zams(initial_mass: float = 1.0, initial_z: float = 0.02, initial_y: float = -1.0) -> RustStarModel: ...  # noqa: E501
    @staticmethod
    def from_pre_ms(initial_mass: float = 1.0, initial_z: float = 0.02, initial_y: float = -1.0, t_c: float = 9.0e5, d_log10_p: float = 0.02) -> RustStarModel: ...  # noqa: E501
    @staticmethod
    def from_mod_str(text: str) -> RustStarModel: ...
    @staticmethod
    def from_photo_bytes(bytes: bytes) -> RustStarModel: ...
    def to_mod_str(self) -> str: ...
    def to_photo_bytes(self) -> bytes: ...
    def eval_hydro_vars(self, skip_net: bool = False, skip_net_jacobian: bool = False) -> None: ...  # noqa: E501
    def set_evolve_controls(self, mesh_delta_coeff: float = 1.0, varcontrol_target: float = 1.0e-4) -> None: ...  # noqa: E501
    def save_start_values(self) -> None: ...
    def save_old_values(self) -> None: ...
    def eval_equations(self, include_chem: bool = False) -> tuple[_ArrF64, _ArrF64, _ArrF64, _ArrF64, _ArrF64]: ...  # noqa: E501
    def solve_henyey(self, include_chem: bool = False, use_bcyclic: bool = True) -> tuple[bool, int, float, float, float, float]: ...  # noqa: E501
    def update_mixing(self, dt: float, overshoot_f: float = 0.0, overshoot_f0: float = 0.004, scheme: str = "exponential") -> int: ...  # noqa: E501
    def eval_dxdt_mix(self) -> _ArrF64: ...
    def solve_diffusion(self, dt: float) -> None: ...
    def run_omega_mix(self, dt: float, omega: _ArrF64, nu_omega: _ArrF64, n_substeps: int = 4) -> None: ...  # noqa: E501
    def step_struct_burn_mix(self, dt: float, op_split_burn: bool = False, use_bcyclic: bool = True, overshoot_f: float = 0.0, overshoot_f0: float = 0.004) -> tuple[bool, int, float, float, float, float]: ...  # noqa: E501
    def apply_wind_and_adjust_mass(self, scheme: str = "Dutch", eta: float = 1.0, dt: float = 0.0) -> float: ...  # noqa: E501
    def adjust_mesh(self) -> tuple[int, int]: ...
    def pick_next_timestep(self, max_timestep_factor: float = 1.2, min_timestep_factor: float = 0.5, dx_nuc_drop_limit: float = 1.0e-2) -> tuple[float, float, float]: ...  # noqa: E501
    def evolve_step(self, do_remesh: bool = False, op_split_burn: bool = False, use_bcyclic: bool = True) -> tuple[int, float, float, float, int, int, int, int, float, bool]: ...  # noqa: E501
    def evolve(self, max_steps: int, do_remesh: bool = False, op_split_burn: bool = False, use_bcyclic: bool = True, max_age: float | None = None) -> list[tuple[int, float, float, float, int, int, int, int, float, bool]]: ...  # noqa: E501
    def history_tuple(self) -> tuple[tuple[int, int, int, int], list[float]]: ...  # noqa: E501
    def to_history_line(self) -> str: ...
    def to_profile_str(self) -> str: ...
    def set_primary_state(self, lnd: _ArrF64, ln_t: _ArrF64, ln_r: _ArrF64, lum: _ArrF64, dq: _ArrF64, xa: _ArrF64) -> None: ...  # noqa: E501

def star_eval_burgers_velocities(a: _ArrF64, z: _ArrF64, x_mass: _ArrF64, rho: float, t: float, dlnt_dr: float, g_grav: float, g_rad: _ArrF64 | None = None, scheme: str = "stanton_murillo") -> tuple[_ArrF64, _ArrF64, float]: ...  # noqa: E501
def star_eval_caplan_ocp_dstar(gamma_i: float, kappa: float) -> float: ...
def star_eval_radiative_levitation_accel(kappa_species: float, lum_face: float, r_face: float) -> float: ...  # noqa: E501
def star_eval_wd_sedimentation_heating(a: _ArrF64, z: _ArrF64, x_mass: _ArrF64, v_gt: _ArrF64, abar: float, zbar: float, g_grav: float) -> float: ...  # noqa: E501
def star_do_element_diffusion(model: RustStarModel, dt: float, scheme: str = "stanton_murillo") -> tuple[_ArrF64, _ArrF64]: ...  # noqa: E501
def star_eval_rotation_fp_ft(w_div_w_crit: float, fp_min: float = 0.75, ft_min: float = 0.95) -> tuple[float, float, float]: ...  # noqa: E501
def star_eval_omega_crit(cgrav: float, m_grav: float, r_polar: float, lum: float, opacity: float, max_gamma_edd: float = 0.99) -> float: ...  # noqa: E501
def star_eval_rotational_mixing_zone(r: float, hp: float, grav: float, rho: float, t: float, cp: float, opacity: float, chi_t: float, chi_rho: float, grada: float, gradt: float, brunt_n2_t: float, brunt_n2_mu: float, omega: float, dln_omega_dln_r: float, abar: float, zbar: float, d_visc: float = 0.0) -> tuple[float, float, float, float, float, float, float, float, float]: ...  # noqa: E501
def star_set_rotation_mixing_info(model: RustStarModel) -> _ArrF64: ...
def star_eval_phase_separation_shift(x_heavy: float, is_one_mixture: bool = False) -> float: ...  # noqa: E501
def star_eval_castellani_penetration_vel(conv_vel: float, r: float, hp: float, gradr: float, grada: float, abar_in: float, abar_out: float) -> float: ...  # noqa: E501
def star_do_conv_premix(model: RustStarModel, max_iters: int = 10, avoid_fuel_increase: bool = True) -> int: ...  # noqa: E501
def star_add_predictive_mixing(model: RustStarModel, ingested_mass_limit_coeff: float = 0.1, superad_thresh: float = 0.005) -> int: ...  # noqa: E501
def star_do_phase_separation(model: RustStarModel, dt: float, gamma_crystallization: float = 175.0, is_one_mixture: bool = False) -> _ArrF64: ...  # noqa: E501
def star_eval_hllc_face(p_l: float, rho_l: float, cs_l: float, u_l: float, dm_l: float, p_r: float, rho_r: float, cs_r: float, u_r: float, dm_r: float, r_face: float, m_grav_face: float, cgrav_face: float) -> tuple[float, float, float, float, float, float, float]: ...  # noqa: E501
def star_eval_hllc_surface_face(p_surf_cell: float, rho_surf: float, cs_surf: float, u_surf: float, dm_surf: float, r_surf: float, m_surf: float, cgrav_surf: float, p_bc: float) -> tuple[float, float, float, float, float, float, float]: ...  # noqa: E501
def star_eval_riemann_momentum_residual(u_k: float, u_old_k: float, dt: float, dm_k: float, r_outer: float, r_inner: float, p_face_outer: float, p_face_inner: float, p_cell_k: float, g_cell_k: float) -> float: ...  # noqa: E501
def star_eval_riemann_profile(model: RustStarModel, u_cell: _ArrF64, u_old_cell: _ArrF64, p_surf_bc: float = 0.0, parallel: bool = True) -> tuple[_ArrF64, _ArrF64]: ...  # noqa: E501
def star_eval_rsp2_zone(w: float, t: float, rho: float, p: float, cp: float, grad_ad: float, delta: float, kappa: float, hp_cell: float, grad_t: float, r_outer: float, r_inner: float, u_outer: float, u_inner: float, dm: float, dlnd_dt: float = 0.0, alfa: float = 1.5, alfap: float = 1.0, alfas: float = 1.0, alfac: float = 1.0, alfad: float = 1.0, alfam: float = 0.25, alfat: float = 0.01, gammar: float = 1.0, min_w2_for_turbulence: float = 1.0e-4) -> tuple[float, float, float, float, float, float, float]: ...  # noqa: E501
def star_eval_rsp2_face(r_face: float, m_grav_face: float, cgrav_face: float, p_face: float, rho_face: float, t_face: float, cp_face: float, grad_ad_face: float, kappa_face: float, w_inner: float, w_outer: float, t4_inner: float, t4_outer: float, lnp_inner: float, lnp_outer: float, chi_face: float, du_over_r_face: float, dm_bar: float, alfa: float = 1.5, alfac: float = 1.0, alfat: float = 0.01) -> tuple[float, float, float, float, float, float]: ...  # noqa: E501
def star_solve_stationary_rsp2_w(t: float, rho: float, p: float, cp: float, grad_ad: float, delta: float, kappa: float, hp_cell: float, grad_t: float, alfa: float = 1.5, alfas: float = 1.0, alfad: float = 1.0, gammar: float = 1.0) -> float: ...  # noqa: E501
def star_eval_center_parabolic(r_inner: float, r_outer: float, v_inner: float, v_outer: float) -> float: ...  # noqa: E501
def star_build_pulse_points(model: RustStarModel, add_center_point: bool = True, add_double_points: bool = False, grad_mu_threshold: float = 1.0e99) -> _ArrF64: ...  # noqa: E501
def star_export_gyre(model: RustStarModel, add_center_point: bool = True) -> str: ...  # noqa: E501
def star_export_fgong(model: RustStarModel, add_center_point: bool = True) -> str: ...  # noqa: E501
def star_export_osc(model: RustStarModel, add_center_point: bool = True) -> str: ...  # noqa: E501

class RustBinarySystem:
    donor: RustStarModel
    accretor: RustStarModel | None
    donor_mass_g: float
    accretor_mass_g: float
    separation: float
    period: float
    eccentricity: float
    angular_momentum_j: float
    rl: tuple[float, float]
    rl_relative_gap: tuple[float, float]
    mtransfer_rate: float
    jdot: float
    binary_age_yr: float
    model_number: int
    max_timestep_sec: float
    @staticmethod
    def new_point_mass(donor: RustStarModel, companion_mass_msun: float, initial_period_days: float, eccentricity: float = 0.0, is_black_hole: bool = False, initial_bh_spin: float = 0.0) -> RustBinarySystem: ...  # noqa: E501
    @staticmethod
    def new_two_stars(donor: RustStarModel, accretor: RustStarModel, initial_period_days: float, eccentricity: float = 0.0) -> RustBinarySystem: ...  # noqa: E501
    def step(self, dt: float) -> tuple[float, int]: ...

def binary_eval_rlobe(m1: float, m2: float, a: float) -> float: ...
def binary_eval_rl_relative_gap(r: float, rl: float, eccentricity: float) -> float: ...  # noqa: E501
def binary_separation_from_period(m1: float, m2: float, period: float) -> float: ...  # noqa: E501
def binary_period_from_separation(m1: float, m2: float, separation: float) -> float: ...  # noqa: E501
def binary_eval_j_orb(m1: float, m2: float, separation: float, eccentricity: float) -> float: ...  # noqa: E501
def binary_separation_from_j_orb(m1: float, m2: float, j_orb: float, eccentricity: float) -> float: ...  # noqa: E501
def binary_eval_l2_overflow_limit(m1: float, m2: float) -> float: ...
def binary_build_anomaly_grid(anomaly_steps: int, eccentricity: float) -> tuple[_ArrF64, _ArrF64]: ...  # noqa: E501
def binary_eval_ritter_mdot_thin(m_donor: float, m_accretor: float, r_donor: float, rl_donor: float, rho_surf: float, p_surf: float, t_surf: float, mu_surf: float, is_kolb_scheme: bool) -> float: ...  # noqa: E501
def binary_eval_kolb_mdot_thick(m_donor: float, m_accretor: float, rl_donor: float, r_profile: _ArrF64, p_profile: _ArrF64, t_profile: _ArrF64, mu_profile: _ArrF64, gamma1_profile: _ArrF64) -> float: ...  # noqa: E501
def binary_eval_arras_mdot_thin(m_donor: float, m_accretor: float, separation: float, r_donor: float, rho_surf: float, t_surf: float, mu_surf: float) -> float: ...  # noqa: E501
def binary_eval_l2_mass_loss_fraction(donor_mass_msun: float, accretor_mass_msun: float, mass_transfer_rate_msun_yr: float, orbital_separation_rsun: float, disk_alpha: float, disk_mu: float) -> float: ...  # noqa: E501
def binary_eval_bh_spin_and_edd(m_bh_init: float, a_spin_init: float, m_bh_current: float, kappa_surf: float) -> tuple[float, float, float, float]: ...  # noqa: E501
def binary_eval_wind_transfer(m_donor: float, m_accretor: float, r_donor: float, rl_donor: float, separation: float, eccentricity: float, b_wind: float, alpha_bh: float, beta_bh: float, max_frac: float) -> tuple[float, float]: ...  # noqa: E501
def binary_eval_accreted_material_j(m_donor: float, m_accretor: float, r_accretor: float, separation: float) -> tuple[float, int]: ...  # noqa: E501
def binary_eval_jdot_gr(m_donor: float, m_accretor: float, separation: float, j_orb: float) -> float: ...  # noqa: E501
def binary_eval_jdot_ml(m_donor: float, m_accretor: float, separation: float, period: float, eccentricity: float, mdot_sys_tr_donor: float, mdot_sys_wind_donor: float, mdot_sys_tr_accretor: float, mdot_sys_wind_accretor: float, mdot_sys_cct: float, gamma_mt: float) -> float: ...  # noqa: E501
def binary_eval_jdot_mb(m_star: float, r_star: float, rl_star: float, period: float, gamma_mb: float, q_conv_env: float, scale_for_low_qconv_env: bool) -> float: ...  # noqa: E501
def binary_eval_jdot_ls(s_donor_old: float, s_donor: float, s_accretor_old: float, s_accretor: float, dt: float) -> float: ...  # noqa: E501
def binary_eval_tides_hut(m_star: float, m_companion: float, r_phot: float, separation: float, eccentricity: float, period: float, omega_star: float, m_env_msun: float, r_env_rsun: float, l_phot_lsun: float, r_gyr_sq: float, f_tid: float, has_convective_envelope: bool) -> tuple[float, float, float, float, float, float, float]: ...  # noqa: E501
def binary_pick_next_timestep_rl(donor_time_step_yr: float, max_timestep_sec: float, rl_gap_old_donor: float, rl_gap_donor: float, fr: float = 0.01, fr_limit: float = 0.01, fr_dt_limit: float = 0.0, dt_softening_factor: float = 0.0, time_delta_coeff: float = 1.0) -> tuple[float, int]: ...  # noqa: E501

def astero_eval_acoustic_profile(r: _ArrF64, csound: _ArrF64, r_center: float, r_phot: float) -> dict[str, Any]: ...  # noqa: E501
def astero_eval_delta_nu_asymptotic(photosphere_acoustic_r: float) -> float: ...
def astero_eval_delta_nu_scaling(m_g: float, teff_k: float, l_phot_erg_s: float) -> float: ...  # noqa: E501
def astero_eval_delta_nu_radius_scaling(m_g: float, r_phot_cm: float) -> float: ...  # noqa: E501
def astero_eval_nu_max(m_g: float, r_phot_cm: float, teff_k: float) -> float: ...  # noqa: E501
def astero_eval_nu_ac(g_surf_cgs: float, gamma1_surf: float, rho_surf: float, peos_surf: float) -> float: ...  # noqa: E501
def astero_eval_delta_pg(r: _ArrF64, r_mid: _ArrF64, brunt_n2: _ArrF64, csound_face: _ArrF64, mode_freq_uhz: float, bildsten2012: bool) -> float: ...  # noqa: E501
def astero_eval_glitch_acoustic_depths(r: _ArrF64, csound: _ArrF64, gradr: _ArrF64, grada: _ArrF64, gamma1: _ArrF64, t: _ArrF64, r_center: float) -> tuple[float, float, float]: ...  # noqa: E501
def astero_eval_radial_delta_nu_ols(n_orders: _ArrF64, model_freqs: _ArrF64) -> float: ...  # noqa: E501
def astero_match_modes_by_degree(obs_freqs: _ArrF64, cand_freqs: _ArrF64, cand_inertias: _ArrF64, cand_orders: _ArrI32) -> tuple[_ArrF64, _ArrF64, _ArrI32]: ...  # noqa: E501
def astero_eval_surface_correction(scheme: str, obs_freqs: _ArrF64, obs_sigmas: _ArrF64, model_freqs: _ArrF64, model_inertias: _ArrF64, l0_obs_freqs: _ArrF64, l0_model_freqs: _ArrF64, l0_inertias: _ArrF64, l0_orders: _ArrF64, delta_nu_obs: float, nu_max: float, b_power: float, correction_factor: float) -> dict[str, Any]: ...  # noqa: E501
def astero_eval_ratios_r01_r10(l0_freqs: _ArrF64, l0_sigs: _ArrF64, l1_freqs: _ArrF64, l1_sigs: _ArrF64) -> dict[str, _ArrF64]: ...  # noqa: E501
def astero_eval_ratios_r02(l0_freqs: _ArrF64, l0_sigs: _ArrF64, l1_freqs: _ArrF64, l1_sigs: _ArrF64, l2_freqs: _ArrF64, l2_sigs: _ArrF64) -> tuple[_ArrF64, _ArrF64]: ...  # noqa: E501
def astero_eval_chi2(obs_freqs: _ArrF64, obs_sigmas: _ArrF64, corr_model_freqs: _ArrF64, r010_obs: _ArrF64, r010_sigmas: _ArrF64, r010_model: _ArrF64, r02_obs: _ArrF64, r02_sigmas: _ArrF64, r02_model: _ArrF64, delta_nu_obs: float, delta_nu_sigma: float, delta_nu_model: float, nu_max_obs: float, nu_max_sigma: float, nu_max_model: float, spectro_obs: _ArrF64, spectro_sigmas: _ArrF64, spectro_model: _ArrF64, chi2_seismo_fraction: float, chi2_seismo_delta_nu_fraction: float, chi2_seismo_nu_max_fraction: float, chi2_seismo_r_010_fraction: float, chi2_seismo_r_02_fraction: float) -> dict[str, float]: ...  # noqa: E501
def astero_nelder_mead_minimize(objective: Any, x_first: list[float], x_lower: list[float], x_upper: list[float], max_evals: int, x_atol: float, x_rtol: float, chi2_tol: float, centroid_weight_power: float) -> dict[str, Any]: ...  # noqa: E501
def astero_powell_quadratic_minimize(objective: Any, x_first: list[float], x_lower: list[float], x_upper: list[float], max_evals: int, rhobeg: float, rhoend: float) -> dict[str, Any]: ...  # noqa: E501
def astero_scan_cartesian_grid(objective: Any, x_lower: list[float], x_upper: list[float], x_delta: list[float]) -> dict[str, Any]: ...  # noqa: E501

def w5_eval_diff_rot_batch(nz: int, parallel: bool = False) -> _ArrF64: ...
def w5_eval_riemann_rsp2_batch(nz: int, parallel: bool = False) -> _ArrF64: ...
def w5_eval_binary_rlof_batch(nz: int, parallel: bool = False) -> _ArrF64: ...
def w5_eval_astero_seism(nz: int) -> float: ...

# fmt: on
