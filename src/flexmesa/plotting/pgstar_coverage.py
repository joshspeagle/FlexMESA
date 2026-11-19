# SPDX-License-Identifier: LGPL-3.0-or-later
"""Upstream ``pgstar`` / ``pgbinary`` / ``astero_pgstar`` coverage catalog.

Parses MESA ``26.4.1`` plotting ``*.defaults`` schemas and maps each upstream
``*_win_flag`` panel family onto ``flexmesa.plotting`` equivalents (``#173``).
"""

from __future__ import annotations

import collections
import dataclasses
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
        "equivalent",
        "plot_mixing_profile, plot_profile_panels",
        "#173",
        "log_D_mix segmented by mixing_type and component diffusivities.",
    ),
    ("pgstar", "Dynamo"): (
        "absent",
        "-",
        "#158",
        "Magnetic/rotational interior diagnostics await rotation port.",
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
        "absent",
        "-",
        "#173",
        "Rayleigh-Taylor mixing diagnostic not yet implemented.",
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
        "absent",
        "-",
        "#170",
        "Gravity-mode period spacing diagram awaits GYRE/ADIPLS ports.",
    ),
    ("pgstar", "Network"): (
        "absent",
        "-",
        "#173",
        "Isochrone / nuclide abundance grid plot not yet implemented.",
    ),
    ("pgstar", "Production"): (
        "absent",
        "-",
        "#173",
        "Yield production factor panel not yet implemented.",
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
        "absent",
        "-",
        "#173",
        "Photometric CMD panels not yet wired to plotting.",
    ),
    ("pgstar", "Text_Summary<N>"): (
        "partial",
        "plot_case_comparison",
        "#173",
        "Provenance and tolerance summary banner in figure title.",
    ),
    ("pgstar", "Grid<N>"): (
        "equivalent",
        "plot_pgstar_grid, plot_case_comparison",
        "#173",
        "GridSpec multi-window composer supporting &pgstar Grid<N>_* inlists.",
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
        "plot_pgbinary_panels, plot_case_comparison",
        "#160",
        "Binary provenance and residual tolerance banner in title.",
    ),
    ("pgbinary", "Grid<N>"): (
        "equivalent",
        "plot_pgstar_grid, plot_pgbinary_panels, plot_case_comparison",
        "#173",
        "GridSpec composer supporting &pgbinary Grid<N>_* inlists.",
    ),
    ("pgbinary", "Star<N>"): (
        "equivalent",
        "plot_pgbinary_panels, plot_pgstar_grid",
        "#173",
        "Donor/accretor HR tracks, point-mass status, and Star1/Star2 cells.",
    ),
    ("pgbinary", "Orbit"): (
        "equivalent",
        "plot_pgbinary_orbit, plot_pgbinary_panels",
        "#173",
        "2D Keplerian orbits, CoM, and Roche/stellar equipotential contours.",
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
            ``*_win_flag`` panel family.
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
