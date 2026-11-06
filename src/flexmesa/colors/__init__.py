# SPDX-License-Identifier: LGPL-3.0-or-later
"""Bolometric and synthetic stellar photometry (`colors`, Wave 2c).

Ports upstream MESA ``colors`` (`colors_def.f90`, `colors_lib.f90`,
`bolometric.f90`, `synthetic.f90`, `hermite_interp.f90`, `linear_interp.f90`,
`knn_interp.f90`, `colors_utils.f90`, and `colors_history.f90`) into the
Three-Tier Hybrid Architecture:

1. **Built-in Atmosphere & Filter Catalogs**:
   - ``Kurucz2003all`` 4D stellar atmosphere SED lookup cube over
     $(T_{\\mathrm{eff}}, \\log g, [\\mathrm{M/H}], \\lambda)$ with 1199
     wavelengths in $[147.2, 1.6 \\times 10^6]\\,\\mathrm{\\AA}$.
   - 7 Johnson photometric passbands (``B, I, J, M, R, U, V``) with Vega, AB
     ($f_\\nu = 3631\\,\\mathrm{Jy}$), and ST
     ($f_\\lambda = 3.63 \\times 10^{-9}$) zero-point fluxes.
2. **SED Interpolation (`sed`)**:
   - 3D cubic Hermite tensor interpolation (`eval_sed`, `method="Hermite"`),
     log-fallback trilinear interpolation (`method="Linear"`), and
     4-nearest-neighbor inverse-distance interpolation (`method="KNN"`).
3. **Bolometric & Synthetic Photometry (`photometry`, `synthetic`, `utils`)**:
   - Composite non-uniform Simpson (`simpson_integration`) and trapezoidal
     (`trapezoidal_integration`) quadrature, geometric dilution (`dilute_flux`),
     bolometric photometry (`calculate_bolometric_phot`), photon-counting
     synthetic filter photometry (`calculate_synthetic`), and 10-column history
     evaluators (`eval_colors`, `eval_colors_dual`, `eval_colors_zones`).
"""

from __future__ import annotations

from flexmesa.colors import photometry, sed, synthetic, utils

COLOR_COLUMN_NAMES = utils.COLOR_COLUMN_NAMES
ColorsBuiltinGrid = utils.ColorsBuiltinGrid
D_10PC_CM = utils.D_10PC_CM
JOHNSON_FILTER_NAMES = utils.JOHNSON_FILTER_NAMES
Z_OVER_X_REF_GS98 = utils.Z_OVER_X_REF_GS98
calculate_bolometric_phot = synthetic.calculate_bolometric_phot
calculate_bolometric_phot_jax = synthetic.calculate_bolometric_phot_jax
calculate_synthetic = synthetic.calculate_synthetic
calculate_synthetic_flux_jax = synthetic.calculate_synthetic_flux_jax
calculate_synthetic_jax = synthetic.calculate_synthetic_jax
compute_ab_zero_point = synthetic.compute_ab_zero_point
compute_interp_radius_jax = sed.compute_interp_radius_jax
compute_st_zero_point = synthetic.compute_st_zero_point
compute_vega_zero_point = synthetic.compute_vega_zero_point
dilute_flux = utils.dilute_flux
dilute_flux_jax = utils.dilute_flux_jax
eval_colors = photometry.eval_colors
eval_colors_dual = photometry.eval_colors_dual
eval_colors_jax = photometry.eval_colors_jax
eval_colors_zones = photometry.eval_colors_zones
eval_sed = sed.eval_sed
eval_sed_hermite_jax = sed.eval_sed_hermite_jax
eval_sed_knn_jax = sed.eval_sed_knn_jax
eval_sed_linear_jax = sed.eval_sed_linear_jax
flux_to_magnitude = synthetic.flux_to_magnitude
get_builtin_colors_grid = utils.get_builtin_colors_grid
h00 = sed.h00
h01 = sed.h01
h10 = sed.h10
h11 = sed.h11
interpolate_array = utils.interpolate_array
interpolate_array_jax = utils.interpolate_array_jax
remove_dat = utils.remove_dat
simpson_integration = utils.simpson_integration
simpson_integration_jax = utils.simpson_integration_jax
trapezoidal_integration = utils.trapezoidal_integration
trapezoidal_integration_jax = utils.trapezoidal_integration_jax

__all__: list[str] = [
    "COLOR_COLUMN_NAMES",
    "D_10PC_CM",
    "JOHNSON_FILTER_NAMES",
    "Z_OVER_X_REF_GS98",
    "ColorsBuiltinGrid",
    "calculate_bolometric_phot",
    "calculate_bolometric_phot_jax",
    "calculate_synthetic",
    "calculate_synthetic_flux_jax",
    "calculate_synthetic_jax",
    "compute_ab_zero_point",
    "compute_interp_radius_jax",
    "compute_st_zero_point",
    "compute_vega_zero_point",
    "dilute_flux",
    "dilute_flux_jax",
    "eval_colors",
    "eval_colors_dual",
    "eval_colors_jax",
    "eval_colors_zones",
    "eval_sed",
    "eval_sed_hermite_jax",
    "eval_sed_knn_jax",
    "eval_sed_linear_jax",
    "flux_to_magnitude",
    "get_builtin_colors_grid",
    "h00",
    "h01",
    "h10",
    "h11",
    "interpolate_array",
    "interpolate_array_jax",
    "photometry",
    "remove_dat",
    "sed",
    "simpson_integration",
    "simpson_integration_jax",
    "synthetic",
    "trapezoidal_integration",
    "trapezoidal_integration_jax",
    "utils",
]
