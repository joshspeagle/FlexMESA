# SPDX-License-Identifier: GPL-3.0-or-later
"""Upstream ``pgstar`` / ``pgbinary`` / ``astero_pgstar`` coverage catalog.

Parses MESA ``26.4.1`` plotting ``*.defaults`` schemas and maps each upstream
``*_win_flag`` panel family onto ``flexmesa.plotting`` equivalents (``#173``).
"""

from __future__ import annotations

import collections
import dataclasses
import importlib
import pathlib
import re

from flexmesa.schema import defaults

_NUMBERED_SUFFIX_RE = re.compile(r"[1-9]$")


@dataclasses.dataclass(frozen=True)
class PgstarPanelStatus:
    """Coverage status for one upstream ``PGSTAR``/``PGBINARY`` panel family.

    Attributes:
        namelist: Source namelist (``"pgstar"``, ``"pgbinary"``, or
            ``"astero_pgstar"``).
        family: Canonical panel family identifier (for example ``"HR"`` or
            ``"History_Panels<N>"``).
        window_count: Number of distinct ``*_win_flag`` instances in upstream
            ``.defaults``.
        status: Implementation tier (``"equivalent"``, ``"partial"``, or
            ``"absent"``).
        flexmesa_builders: Comma-separated ``flexmesa.plotting`` entry points
            covering this panel family (or ``"-"`` when absent).
        issue: Tracking GitHub issue reference (``"#173"``, etc.).
        notes: Concise scope note distinguishing implemented panels from
            pending interactive or multi-panel options.
    """

    namelist: str
    family: str
    window_count: int
    status: str
    flexmesa_builders: str
    issue: str
    notes: str


@dataclasses.dataclass(frozen=True)
class PgstarCoverageReport:
    """Aggregated panel and control-key coverage across MESA plotting files.

    Attributes:
        source_counts: Mapping from defaults filename to parsed control count.
        total_windows: Total ``*_win_flag`` window definitions across all
            three plotting ``.defaults`` files.
        panels: Ordered tuple of ``PgstarPanelStatus`` records.
    """

    source_counts: dict[str, int]
    total_windows: int
    panels: tuple[PgstarPanelStatus, ...]


_FAMILY_SPECS: dict[tuple[str, str], tuple[str, str, str, str]] = {
    ("pgstar", "TRho_Profile"): (
        "equivalent",
        "plot_trho_profile, plot_profile_panels",
        "#173",
        (
            "T-rho profile with MESA plot_info degeneracy, P_rad=P_gas, and "
            "burn guides."
        ),
    ),
    ("pgstar", "Abundance"): (
        "equivalent",
        "plot_abundance_profile, plot_profile_panels",
        "#173",
        "Ranked multi-isotope log mass fractions with mixing spine bar.",
    ),
    ("pgstar", "Power"): (
        "equivalent",
        "plot_power_profile, plot_profile_panels",
        "#173",
        "22-category nuclear specific power breakdown from profileN.data.",
    ),
    ("pgstar", "Mixing"): (
        "partial",
        "plot_mixing_profile, plot_profile_panels",
        "#173",
        (
            "Draws the mixing and rotational diffusion curves, but "
            "accepts D_omega as the rotation total where upstream draws "
            "D_mix_rotation only "
            "(star/private/pgstar_mixing_ds.f90:347), and linear D_* "
            "columns and the summed mode omit the D_*_factor weights."
        ),
    ),
    ("pgstar", "Dynamo"): (
        "partial",
        "plot_dynamo_profile, plot_pgstar_grid",
        "#173",
        (
            "Draws log B_phi, log B_r, log Omega and log(j/1e20), but "
            "still draws when some of the four are absent; upstream "
            "always draws all four "
            "(star/private/pgstar_dynamo.f90:190-294)."
        ),
    ),
    ("pgstar", "Mode_Prop"): (
        "equivalent",
        "plot_propagation_diagram",
        "#173",
        "Brunt-Vaisala N and Lamb S_l propagation diagram.",
    ),
    ("pgstar", "Summary_Burn"): (
        "equivalent",
        "plot_summary_burn_profile, plot_case_comparison",
        "#173",
        "Radial logT/logRho/eps_nuc/eps_neu with peak category labels.",
    ),
    ("pgstar", "Summary_Profile"): (
        "equivalent",
        "plot_summary_profile, plot_profile_panels",
        "#173",
        "Normalized [0, 1] 11-quantity interior summary and 4-panel structure.",
    ),
    ("pgstar", "Summary_History"): (
        "equivalent",
        "plot_summary_history, plot_case_comparison",
        "#173",
        "Normalized [0, 1] star history summary and MESA 26.4.1 comparison.",
    ),
    ("pgstar", "Kipp"): (
        "equivalent",
        "plot_history_kippenhahn, plot_profile_kippenhahn, plot_kippenhahn",
        "#173",
        (
            "Mass/q/radius vs model_number/age mixing, burning, and core "
            "boundaries."
        ),
    ),
    ("pgstar", "rti"): (
        "partial",
        "plot_rti_history, plot_pgstar_grid",
        "#173",
        (
            "Draws RTI spans from the rti_type and rti_qtop history "
            "columns as upstream does "
            "(star/private/pgstar_rti.f90:326-399), and also from "
            "mix_type and profile columns, which upstream does not use."
        ),
    ),
    ("pgstar", "TRho"): (
        "equivalent",
        "plot_tc_rhoc_diagram, plot_history_track, plot_case_comparison",
        "#173",
        "Central log_center_Rho vs log_center_T evolutionary track.",
    ),
    ("pgstar", "TmaxRho"): (
        "equivalent",
        "plot_history_track",
        "#173",
        "log_center_Rho vs log_max_T track with psi4 degeneracy guide.",
    ),
    ("pgstar", "HR"): (
        "equivalent",
        "plot_hr_diagram, plot_history_track, plot_case_comparison",
        "#173",
        "Inverted log_Teff vs log_L Hertzsprung-Russell track.",
    ),
    ("pgstar", "logL_Teff"): (
        "equivalent",
        "plot_hr_diagram, plot_history_track, plot_case_comparison",
        "#173",
        "HR coordinate variant matching log_Teff vs log_L.",
    ),
    ("pgstar", "logL_R"): (
        "equivalent",
        "plot_history_track",
        "#173",
        "Radius vs log_L evolutionary track preset.",
    ),
    ("pgstar", "logL_v"): (
        "equivalent",
        "plot_history_track",
        "#173",
        "Surface velocity vs log_L evolutionary track preset.",
    ),
    ("pgstar", "L_Teff"): (
        "equivalent",
        "plot_history_track",
        "#173",
        "Inverted linear Teff vs luminosity track preset.",
    ),
    ("pgstar", "L_v"): (
        "equivalent",
        "plot_history_track",
        "#173",
        "Surface velocity vs linear luminosity track preset.",
    ),
    ("pgstar", "L_R"): (
        "equivalent",
        "plot_history_track",
        "#173",
        "Radius vs linear luminosity track preset.",
    ),
    ("pgstar", "R_Teff"): (
        "equivalent",
        "plot_history_track",
        "#173",
        "Inverted Teff vs radius evolutionary track preset.",
    ),
    ("pgstar", "R_L"): (
        "equivalent",
        "plot_history_track",
        "#173",
        "Luminosity vs radius evolutionary track preset.",
    ),
    ("pgstar", "logg_Teff"): (
        "equivalent",
        "plot_history_track",
        "#173",
        "Spectroscopic Kiel diagram (inverted Teff and log_g).",
    ),
    ("pgstar", "logg_logT"): (
        "equivalent",
        "plot_history_track",
        "#173",
        "Logarithmic Kiel diagram (inverted log_Teff and log_g).",
    ),
    ("pgstar", "dPg_dnu"): (
        "equivalent",
        "plot_dpg_dnu, plot_history_track, plot_pgstar_grid",
        "#173",
        (
            "Gravity-mode period spacing delta_Pg vs delta_nu track from "
            "saved history.data (pulsation physics tracked in #170)."
        ),
    ),
    ("pgstar", "Network"): (
        "partial",
        "plot_network_chart, plot_pgstar_grid",
        "#173",
        (
            "N-Z chart of the isotopes saved in the profile; upstream "
            "draws every species of the net "
            "(star/private/pgstar_network.f90:184-219) and leaves "
            "abundances above the maximum unfilled."
        ),
    ),
    ("pgstar", "Production"): (
        "partial",
        "plot_production_profile, plot_pgstar_grid",
        "#173",
        (
            "Production factors against a scaled AG89 table or a "
            "supplied profile; upstream builds the reference with "
            "get_xa_for_standard_metals from initial_zfracs and maps it "
            "through get_stable_mass_frac "
            "(star/private/pgstar_production.f90:190-211)."
        ),
    ),
    ("pgstar", "Profile_Panels<N>"): (
        "equivalent",
        "plot_profile_custom_panels, plot_profile_panels",
        "#173",
        "Multi-row dual-axis profile column stack and 4-panel default.",
    ),
    ("pgstar", "History_Track<N>"): (
        "equivalent",
        "plot_history_track, plot_case_comparison",
        "#173",
        "Arbitrary two-column history track with MESA 26.4.1 overlay.",
    ),
    ("pgstar", "History_Panels<N>"): (
        "equivalent",
        "plot_history_panels, plot_case_comparison",
        "#173",
        "Multi-row dual-axis history column stack vs model_number or age.",
    ),
    ("pgstar", "Color_magnitude<N>"): (
        "partial",
        "plot_color_magnitude, plot_pgstar_grid",
        "#173",
        (
            "Draws configured colour-magnitude panels, and picks a "
            "default pair when none is configured; upstream's default "
            "has no panels (star/defaults/pgstar.defaults:5284-5323)."
        ),
    ),
    ("pgstar", "Text_Summary<N>"): (
        "partial",
        "plot_text_summary, plot_pgstar_grid, plot_case_comparison",
        "#173",
        (
            "Shows only quantities saved in history.data and leaves "
            "other cells out; upstream evaluates every named quantity "
            "from the model (star/private/pgstar_summary.f90:399-402)."
        ),
    ),
    ("pgstar", "Grid<N>"): (
        "partial",
        "plot_pgstar_grid, plot_case_comparison",
        "#173",
        (
            "Lays out the supported windows; inherits the Text_Summary "
            "and Color_magnitude differences and leaves out default "
            "panel columns absent from the file."
        ),
    ),
    ("pgbinary", "Summary_History"): (
        "equivalent",
        (
            "plot_pgbinary_summary_history, plot_pgbinary_panels, "
            "plot_case_comparison"
        ),
        "#173",
        "Normalized [0, 1] binary summary and 6-panel comparison figure.",
    ),
    ("pgbinary", "History_Track<N>"): (
        "equivalent",
        "plot_history_track, plot_pgbinary_panels, plot_case_comparison",
        "#173",
        "Arbitrary binary two-column track with MESA 26.4.1 overlay.",
    ),
    ("pgbinary", "History_Panels<N>"): (
        "equivalent",
        (
            "plot_pgbinary_panels, plot_history_panels, "
            "plot_binary_evolution, plot_case_comparison"
        ),
        "#173",
        "Multi-panel orbital period, RLOF filling, mdot, and custom stacks.",
    ),
    ("pgbinary", "Text_Summary<N>"): (
        "partial",
        "plot_pgbinary_text_summary, plot_text_summary, plot_pgstar_grid",
        "#173",
        (
            "Shows only quantities saved in binary_history.data and "
            "leaves other cells out; upstream evaluates every named "
            "quantity (binary/private/pgbinary_summary.f90:338-445)."
        ),
    ),
    ("pgbinary", "Grid<N>"): (
        "partial",
        "plot_pgstar_grid, plot_pgbinary_panels, plot_case_comparison",
        "#173",
        (
            "Lays out the supported windows; inherits the Text_Summary "
            "difference and leaves out default panel columns absent "
            "from the file."
        ),
    ),
    ("pgbinary", "Star<N>"): (
        "equivalent",
        "plot_pgbinary_star, plot_pgstar_grid, plot_pgbinary_panels",
        "#173",
        (
            "Embeds any supported pgstar window or nested Grid<N> of Star 1 "
            "or Star 2, with point-mass banner support."
        ),
    ),
    ("pgbinary", "Orbit"): (
        "partial",
        "plot_pgbinary_orbit, plot_pgbinary_panels",
        "#173",
        (
            "Draws the orbit and stellar outlines; derives eccentricity "
            "from J_orb when the column is absent and adds Roche-lobe "
            "contours, where upstream reads b% eccentricity and draws "
            "stellar contours only "
            "(binary/private/pgbinary_orbit.f90:102, 132, 171)."
        ),
    ),
    ("astero_pgstar", "echelle"): (
        "equivalent",
        "plot_echelle_diagram",
        "#173",
        "Echelle diagram for observed and model oscillation modes.",
    ),
    ("astero_pgstar", "ratios"): (
        "equivalent",
        "plot_separation_ratios",
        "#173",
        "Roxburgh & Vorontsov r_010 and r_02 separation ratios.",
    ),
}


def build_pgstar_coverage_report(
    mesa_dir: pathlib.Path,
) -> PgstarCoverageReport:
    """Parse upstream MESA plotting ``.defaults`` and summarize panel coverage.

    Args:
        mesa_dir: Path to extracted MESA ``26.4.1`` source/test-inputs root.

    Returns:
        Populated ``PgstarCoverageReport``.

    Raises:
        FileNotFoundError: If any plotting ``.defaults`` file is missing.
        ValueError: If upstream ``.defaults`` contains an uncatalogued
            ``*_win_flag`` panel family, or a catalogued row names a builder
            that ``flexmesa.plotting`` does not export.
    """
    specs = (
        ("pgstar", mesa_dir / "star" / "defaults" / "pgstar.defaults"),
        ("pgbinary", mesa_dir / "binary" / "defaults" / "pgbinary.defaults"),
        (
            "astero_pgstar",
            mesa_dir / "astero" / "defaults" / "astero_pgstar.defaults",
        ),
    )
    source_counts: dict[str, int] = {}
    panels: list[PgstarPanelStatus] = []
    total_windows = 0

    for namelist, def_path in specs:
        if not def_path.is_file():
            raise FileNotFoundError(
                f"missing upstream plotting defaults file: {def_path}"
            )
        parsed = defaults.parse_defaults_file(def_path)
        source_counts[f"{namelist}.defaults"] = len(parsed)

        family_counts: collections.Counter[str] = collections.Counter()
        for ctrl in parsed.values():
            if not ctrl.name.endswith("_win_flag"):
                continue
            raw_stem = ctrl.name[: -len("_win_flag")]
            family = _NUMBERED_SUFFIX_RE.sub("<N>", raw_stem)
            family_counts[family] += 1
        total_windows += sum(family_counts.values())

        for family, count in family_counts.items():
            key = (namelist, family)
            if key not in _FAMILY_SPECS:
                raise ValueError(
                    f"uncatalogued upstream plotting panel family {key!r}"
                )
            status, builders, issue, notes = _FAMILY_SPECS[key]
            if status not in ("equivalent", "partial", "absent"):
                raise ValueError(f"unknown coverage status {status!r}")
            if builders != "-":
                plotting_pkg = importlib.import_module("flexmesa.plotting")
                for b_name in (b.strip() for b in builders.split(",")):
                    if not callable(getattr(plotting_pkg, b_name, None)):
                        raise ValueError(
                            f"coverage row {key!r} names builder {b_name!r}, "
                            "which flexmesa.plotting does not export"
                        )
            panels.append(
                PgstarPanelStatus(
                    namelist=namelist,
                    family=family,
                    window_count=count,
                    status=status,
                    flexmesa_builders=builders,
                    issue=issue,
                    notes=notes,
                )
            )

    return PgstarCoverageReport(
        source_counts=source_counts,
        total_windows=total_windows,
        panels=tuple(panels),
    )
