# SPDX-License-Identifier: LGPL-3.0-or-later
"""Wave 5.4 `astero` package (`flexmesa.astero`).

Provides asteroseismic scaling relations, acoustic transit integrals, g-mode
period spacings, glitch acoustic depths, surface-effect frequency corrections
(`kjeldsen`, `cubic`, `combined`, `power_law`, `sonoi`), frequency separation
ratios ($r_{01}, r_{10}, r_{02}$), mode matching, composite $\\chi^2$
evaluation, and Nelder-Mead / Powell quadratic / Cartesian grid optimization
drivers across Safe Rust and JAX.
"""

from __future__ import annotations

from flexmesa.astero import drivers, jax_kernels, scaling_corr

eval_acoustic_profile = scaling_corr.eval_acoustic_profile
eval_delta_nu_asymptotic = scaling_corr.eval_delta_nu_asymptotic
eval_delta_nu_scaling = scaling_corr.eval_delta_nu_scaling
eval_delta_nu_radius_scaling = scaling_corr.eval_delta_nu_radius_scaling
eval_nu_max = scaling_corr.eval_nu_max
eval_nu_ac = scaling_corr.eval_nu_ac
eval_delta_pg = scaling_corr.eval_delta_pg
eval_glitch_acoustic_depths = scaling_corr.eval_glitch_acoustic_depths
eval_radial_delta_nu_ols = scaling_corr.eval_radial_delta_nu_ols
eval_surface_correction = scaling_corr.eval_surface_correction
eval_ratios_r01_r10 = scaling_corr.eval_ratios_r01_r10
match_modes_by_degree = drivers.match_modes_by_degree
eval_ratios_r02 = drivers.eval_ratios_r02
eval_astero_chi2 = drivers.eval_astero_chi2
nelder_mead_minimize = drivers.nelder_mead_minimize
powell_quadratic_minimize = drivers.powell_quadratic_minimize
scan_cartesian_grid = drivers.scan_cartesian_grid

__all__ = [
    "drivers",
    "eval_acoustic_profile",
    "eval_astero_chi2",
    "eval_delta_nu_asymptotic",
    "eval_delta_nu_radius_scaling",
    "eval_delta_nu_scaling",
    "eval_delta_pg",
    "eval_glitch_acoustic_depths",
    "eval_nu_ac",
    "eval_nu_max",
    "eval_radial_delta_nu_ols",
    "eval_ratios_r01_r10",
    "eval_ratios_r02",
    "eval_surface_correction",
    "jax_kernels",
    "match_modes_by_degree",
    "nelder_mead_minimize",
    "powell_quadratic_minimize",
    "scaling_corr",
    "scan_cartesian_grid",
]
