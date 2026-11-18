# SPDX-License-Identifier: LGPL-3.0-or-later
"""Publication diagnostic plotting package (`flexmesa.plotting`).

Provides publication-styled `matplotlib` replacements for `pgstar`,
including HR diagrams, $\\log T_c$--$\\log \\rho_c$ trajectories, Kippenhahn
diagrams, asteroseismic propagation and echelle diagrams, separation ratio
plots, binary orbital/RLOF evolution plots, radial profile panels, and
MESA `26.4.1` trajectory comparison figures.
"""

from __future__ import annotations

from flexmesa.plotting import (
    astero_binary_plots,
    compare,
    pgstar_coverage,
    stellar_plots,
    style,
)

AASTEX_COLUMN_WIDTH_IN = style.AASTEX_COLUMN_WIDTH_IN
AASTEX_TEXT_WIDTH_IN = style.AASTEX_TEXT_WIDTH_IN
OKABE_ITO = style.OKABE_ITO
ProfileTable = compare.ProfileTable
figsize = style.figsize
load_profile_file = compare.load_profile_file
mesa_rc_params = style.mesa_rc_params
use_mesa_plot_style = style.use_mesa_plot_style
plot_hr_diagram = stellar_plots.plot_hr_diagram
plot_tc_rhoc_diagram = stellar_plots.plot_tc_rhoc_diagram
plot_kippenhahn = stellar_plots.plot_kippenhahn
plot_propagation_diagram = stellar_plots.plot_propagation_diagram
plot_echelle_diagram = astero_binary_plots.plot_echelle_diagram
plot_separation_ratios = astero_binary_plots.plot_separation_ratios
plot_binary_evolution = astero_binary_plots.plot_binary_evolution
plot_case_comparison = compare.plot_case_comparison
plot_profile_kippenhahn = compare.plot_profile_kippenhahn
plot_profile_panels = compare.plot_profile_panels
resolve_case_entry = compare.resolve_case_entry

__all__ = [
    "AASTEX_COLUMN_WIDTH_IN",
    "AASTEX_TEXT_WIDTH_IN",
    "OKABE_ITO",
    "ProfileTable",
    "astero_binary_plots",
    "compare",
    "figsize",
    "load_profile_file",
    "mesa_rc_params",
    "pgstar_coverage",
    "plot_binary_evolution",
    "plot_case_comparison",
    "plot_echelle_diagram",
    "plot_hr_diagram",
    "plot_kippenhahn",
    "plot_profile_kippenhahn",
    "plot_profile_panels",
    "plot_propagation_diagram",
    "plot_separation_ratios",
    "plot_tc_rhoc_diagram",
    "resolve_case_entry",
    "stellar_plots",
    "style",
    "use_mesa_plot_style",
]
