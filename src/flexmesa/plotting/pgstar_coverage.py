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
        "partial",
        "plot_profile_panels",
        "#173",
        "T-rho profile curve; EOS regime overlays pending.",
    ),
    ("pgstar", "Abundance"): (
        "partial",
        "plot_profile_panels",
        "#173",
        "Major species H1/He4/Z/C12/O16; full isotope table pending.",
    ),
    ("pgstar", "Power"): (
        "partial",
        "plot_profile_panels",
        "#173",
        "pp/CNO/3alpha category burning panel from profileN.data.",
    ),
    ("pgstar", "Mixing"): (
        "partial",
        "plot_profile_panels",
        "#173",
        "log_D_mix and mixing_type boundaries from profileN.data.",
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
        "partial",
        "plot_case_comparison",
        "#173",
        "log_L, log_LH, log_LHe history traces vs MESA reference.",
    ),
    ("pgstar", "Summary_Profile"): (
        "equivalent",
        "plot_profile_panels",
        "#173",
        "Four-panel interior abundance, T/rho, luminosity, mixing summary.",
    ),
    ("pgstar", "Summary_History"): (
        "equivalent",
        "plot_case_comparison",
        "#173",
        "Multi-panel star history + MESA 26.4.1 comparison & residuals.",
    ),
    ("pgstar", "Kipp"): (
        "equivalent",
        "plot_kippenhahn, plot_profile_kippenhahn",
        "#173",
        "Convective/burning Kippenhahn diagram from array or profiles.",
    ),
    ("pgstar", "rti"): (
        "absent",
        "-",
        "#173",
        "Rayleigh-Taylor mixing diagnostic not yet implemented.",
    ),
    ("pgstar", "TRho"): (
        "equivalent",
        "plot_tc_rhoc_diagram, plot_case_comparison",
        "#173",
        "Central log_center_Rho vs log_center_T evolutionary track.",
    ),
    ("pgstar", "TmaxRho"): (
        "absent",
        "-",
        "#173",
        "Max-temperature vs density panel not yet implemented.",
    ),
    ("pgstar", "HR"): (
        "equivalent",
        "plot_hr_diagram, plot_case_comparison",
        "#173",
        "Inverted log_Teff vs log_L Hertzsprung-Russell track.",
    ),
    ("pgstar", "logL_Teff"): (
        "equivalent",
        "plot_hr_diagram, plot_case_comparison",
        "#173",
        "HR coordinate variant matching log_Teff vs log_L.",
    ),
    ("pgstar", "logL_R"): (
        "absent",
        "-",
        "#173",
        "Luminosity-radius track variant not yet wired.",
    ),
    ("pgstar", "logL_v"): (
        "absent",
        "-",
        "#173",
        "Luminosity-velocity track not yet implemented.",
    ),
    ("pgstar", "L_Teff"): (
        "absent",
        "-",
        "#173",
        "Linear luminosity vs Teff track not yet wired.",
    ),
    ("pgstar", "L_v"): (
        "absent",
        "-",
        "#173",
        "Linear luminosity-velocity track not yet implemented.",
    ),
    ("pgstar", "L_R"): (
        "absent",
        "-",
        "#173",
        "Linear luminosity-radius track not yet wired.",
    ),
    ("pgstar", "R_Teff"): (
        "absent",
        "-",
        "#173",
        "Radius-Teff track not yet implemented.",
    ),
    ("pgstar", "R_L"): (
        "absent",
        "-",
        "#173",
        "Radius-luminosity track not yet implemented.",
    ),
    ("pgstar", "logg_Teff"): (
        "absent",
        "-",
        "#173",
        "Kiel diagram (log_g vs Teff) not yet implemented.",
    ),
    ("pgstar", "logg_logT"): (
        "absent",
        "-",
        "#173",
        "Kiel diagram (log_g vs log_Teff) not yet implemented.",
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
        "partial",
        "plot_profile_panels",
        "#173",
        "Fixed 4-panel layout; user column-name customization pending.",
    ),
    ("pgstar", "History_Track<N>"): (
        "partial",
        "plot_case_comparison",
        "#173",
        "Core tracks implemented; arbitrary column-pair selector pending.",
    ),
    ("pgstar", "History_Panels<N>"): (
        "partial",
        "plot_case_comparison",
        "#173",
        "Core time-series panels; arbitrary column selector pending.",
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
        "partial",
        "plot_case_comparison",
        "#173",
        "3x2 comparison grid; dynamic multi-window layout pending.",
    ),
    ("pgbinary", "Summary_History"): (
        "equivalent",
        "plot_case_comparison, plot_binary_evolution",
        "#173",
        "Binary period, RLOF, masses, overflow, and residual panels.",
    ),
    ("pgbinary", "History_Track<N>"): (
        "partial",
        "plot_case_comparison",
        "#160",
        "Period/separation/mass tracks; arbitrary axis selector pending.",
    ),
    ("pgbinary", "History_Panels<N>"): (
        "equivalent",
        "plot_binary_evolution, plot_case_comparison",
        "#173",
        "Multi-panel orbital period, RLOF rate, and eccentricity plot.",
    ),
    ("pgbinary", "Text_Summary<N>"): (
        "partial",
        "plot_case_comparison",
        "#160",
        "Binary provenance and residual tolerance banner in title.",
    ),
    ("pgbinary", "Grid<N>"): (
        "partial",
        "plot_case_comparison",
        "#173",
        "3x2 binary comparison grid; arbitrary grid spec pending.",
    ),
    ("pgbinary", "Star<N>"): (
        "partial",
        "plot_profile_panels",
        "#160",
        "Individual star profiles supported; dual-star overlay pending.",
    ),
    ("pgbinary", "Orbit"): (
        "absent",
        "-",
        "#160",
        "2D Roche equipotential geometry plot awaits binary geometry port.",
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
