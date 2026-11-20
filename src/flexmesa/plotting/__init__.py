# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Publication diagnostic plotting package (`flexmesa.plotting`).

Provides publication-styled `matplotlib` replacements for `pgstar` and
`pgbinary`, including HR diagrams, $\log T_c$-$\log \rho_c$ trajectories,
Kippenhahn diagrams, radial profile panels (`TRho_Profile`, `Power`,
`Abundance`, `Mixing`, `Summary_Profile`, `Summary_Burn`), binary orbital/RLOF
evolution and Roche geometry plots, multi-panel `Grid` composers, asteroseismic
propagation and echelle diagrams, separation ratio plots, and MESA `26.4.1`
trajectory comparison figures.
"""

from __future__ import annotations

from flexmesa.plotting import (
    astero_binary_plots,
    compare,
    kippenhahn,
    pgbinary_panels,
    pgstar_coverage,
    pgstar_extra,
    pgstar_grid,
    pgstar_nucleosyn,
    pgstar_panels,
    pgstar_summary,
    pgstar_text,
    stellar_plots,
    style,
)

AASTEX_COLUMN_WIDTH_IN = style.AASTEX_COLUMN_WIDTH_IN
AASTEX_TEXT_WIDTH_IN = style.AASTEX_TEXT_WIDTH_IN
ColorMagnitudeSpec = pgstar_extra.ColorMagnitudeSpec
GridPanelSpec = pgstar_grid.GridPanelSpec
OKABE_ITO = style.OKABE_ITO
ProfileTable = compare.ProfileTable
TextSummarySpec = pgstar_text.TextSummarySpec
figsize = style.figsize
format_cmd_label = pgstar_extra.format_cmd_label
load_profile_file = compare.load_profile_file
mesa_rc_params = style.mesa_rc_params
parse_inlist_cmd_spec = pgstar_extra.parse_inlist_cmd_spec
parse_inlist_grid_spec = pgstar_grid.parse_inlist_grid_spec
parse_inlist_text_summary_spec = pgstar_text.parse_inlist_text_summary_spec
plot_abundance_profile = pgstar_panels.plot_abundance_profile
plot_binary_evolution = astero_binary_plots.plot_binary_evolution
plot_case_comparison = compare.plot_case_comparison
plot_color_magnitude = pgstar_extra.plot_color_magnitude
plot_dpg_dnu = pgstar_extra.plot_dpg_dnu
plot_dynamo_profile = pgstar_extra.plot_dynamo_profile
plot_echelle_diagram = astero_binary_plots.plot_echelle_diagram
plot_history_kippenhahn = kippenhahn.plot_history_kippenhahn
plot_history_panels = pgstar_grid.plot_history_panels
plot_history_track = pgstar_grid.plot_history_track
plot_hr_diagram = stellar_plots.plot_hr_diagram
plot_kippenhahn = stellar_plots.plot_kippenhahn
plot_mixing_profile = pgstar_panels.plot_mixing_profile
plot_network_chart = pgstar_nucleosyn.plot_network_chart
plot_pgbinary_orbit = pgbinary_panels.plot_pgbinary_orbit
plot_pgbinary_panels = pgbinary_panels.plot_pgbinary_panels
plot_pgbinary_star = pgbinary_panels.plot_pgbinary_star
plot_pgbinary_summary_history = pgbinary_panels.plot_pgbinary_summary_history
plot_pgbinary_text_summary = pgstar_text.plot_pgbinary_text_summary
plot_pgstar_grid = pgstar_grid.plot_pgstar_grid
plot_power_profile = pgstar_panels.plot_power_profile
plot_production_profile = pgstar_nucleosyn.plot_production_profile
plot_profile_custom_panels = pgstar_summary.plot_profile_custom_panels
plot_profile_kippenhahn = compare.plot_profile_kippenhahn
plot_profile_panels = compare.plot_profile_panels
plot_propagation_diagram = stellar_plots.plot_propagation_diagram
plot_rti_history = pgstar_extra.plot_rti_history
plot_separation_ratios = astero_binary_plots.plot_separation_ratios
plot_summary_burn_profile = pgstar_summary.plot_summary_burn_profile
plot_summary_history = pgstar_grid.plot_summary_history
plot_summary_profile = pgstar_summary.plot_summary_profile
plot_tc_rhoc_diagram = stellar_plots.plot_tc_rhoc_diagram
plot_text_summary = pgstar_text.plot_text_summary
plot_trho_profile = pgstar_panels.plot_trho_profile
resolve_case_entry = compare.resolve_case_entry
use_mesa_plot_style = style.use_mesa_plot_style

__all__ = [
    "AASTEX_COLUMN_WIDTH_IN",
    "AASTEX_TEXT_WIDTH_IN",
    "OKABE_ITO",
    "ColorMagnitudeSpec",
    "GridPanelSpec",
    "ProfileTable",
    "TextSummarySpec",
    "astero_binary_plots",
    "compare",
    "figsize",
    "format_cmd_label",
    "kippenhahn",
    "load_profile_file",
    "mesa_rc_params",
    "parse_inlist_cmd_spec",
    "parse_inlist_grid_spec",
    "parse_inlist_text_summary_spec",
    "pgbinary_panels",
    "pgstar_coverage",
    "pgstar_extra",
    "pgstar_grid",
    "pgstar_nucleosyn",
    "pgstar_panels",
    "pgstar_summary",
    "pgstar_text",
    "plot_abundance_profile",
    "plot_binary_evolution",
    "plot_case_comparison",
    "plot_color_magnitude",
    "plot_dpg_dnu",
    "plot_dynamo_profile",
    "plot_echelle_diagram",
    "plot_history_kippenhahn",
    "plot_history_panels",
    "plot_history_track",
    "plot_hr_diagram",
    "plot_kippenhahn",
    "plot_mixing_profile",
    "plot_network_chart",
    "plot_pgbinary_orbit",
    "plot_pgbinary_panels",
    "plot_pgbinary_star",
    "plot_pgbinary_summary_history",
    "plot_pgbinary_text_summary",
    "plot_pgstar_grid",
    "plot_power_profile",
    "plot_production_profile",
    "plot_profile_custom_panels",
    "plot_profile_kippenhahn",
    "plot_profile_panels",
    "plot_propagation_diagram",
    "plot_rti_history",
    "plot_separation_ratios",
    "plot_summary_burn_profile",
    "plot_summary_history",
    "plot_summary_profile",
    "plot_tc_rhoc_diagram",
    "plot_text_summary",
    "plot_trho_profile",
    "resolve_case_entry",
    "stellar_plots",
    "style",
    "use_mesa_plot_style",
]
