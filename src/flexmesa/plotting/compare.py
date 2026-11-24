# SPDX-License-Identifier: GPL-3.0-or-later
"""MESA 26.4.1 comparison and radial profile plots built solely from outputs.

Every figure in this module reads stored ASCII outputs (``history.data``,
``binary_history.data``, and ``profileN.data``) directly via
``flexmesa.testing`` and ``flexmesa._rust``. No physics quantities are
recomputed or synthesized inside plotting code.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Literal

import matplotlib.axes as mpl_axes
import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt
import numpy as np

from flexmesa import _rust
from flexmesa.plotting import _compare_panels, _profile_panels, style
from flexmesa.testing import (
    case_manifest,
    harness_runner,
    history_compare,
)

_PROFILE_ALIASES: dict[str, str] = {
    "x": "x_mass_fraction_H",
    "y": "y_mass_fraction_He",
    "z": "z_mass_fraction_metals",
    "mlt_mixing_type": "mixing_type",
    "mix_type": "mixing_type",
    "r_cm": "radius_cm",
    "T": "temperature",
    "rho": "density",
}


@dataclasses.dataclass(frozen=True)
class ProfileTable:
    """Columnar ``float64`` representation of a MESA ``profileN.data`` file.

    Attributes:
        header: Scalar header metadata extracted from lines 2 and 3 (for
            example ``model_number``, ``star_age`` [``yr``], ``num_zones``,
            ``version_number``, and ``math_backend``).
        columns: Mapping from canonical column name to a 1D ``float64`` array
            of shape ``(num_zones,)`` in MESA output units.
        num_zones: Number of radial zones stored in the profile table.
    """

    header: dict[str, Any]
    columns: dict[str, np.ndarray]
    num_zones: int


def load_profile_file(path: pathlib.Path | str) -> ProfileTable:
    """Load a 6-line-header MESA ASCII ``profileN.data`` file.

    Args:
        path: Filesystem path to ``profileN.data``.

    Returns:
        Populated ``ProfileTable`` containing ``header`` metadata and 1D
        ``float64`` zone arrays of shape ``(num_zones,)``.

    Raises:
        FileNotFoundError: If ``path`` does not exist on disk.
        ValueError: If the parsed table has zero zones.
    """
    fpath = pathlib.Path(path)
    if not fpath.is_file():
        raise FileNotFoundError(f"missing profile file: {fpath}")
    parsed = _rust.mesa_read_log_file(str(fpath))
    n_zones = int(parsed["num_rows"])
    if n_zones == 0:
        raise ValueError(f"{fpath}: profile file contains zero zones")
    raw_names = [str(name) for name in parsed["column_names"]]
    raw_vecs = [
        np.ascontiguousarray(np.asarray(vec, dtype=np.float64))
        for vec in parsed["column_values"]
    ]
    columns: dict[str, np.ndarray] = {}
    for name, vec in zip(raw_names, raw_vecs, strict=True):
        if name not in columns:
            columns[name] = vec
        canon = _PROFILE_ALIASES.get(name)
        if canon is not None and canon not in columns:
            columns[canon] = vec
    header = {str(k): v for k, v in dict(parsed["header"]).items()}
    return ProfileTable(header=header, columns=columns, num_zones=n_zones)


def resolve_case_entry(
    case: str,
    suite: str | None = None,
    manifest_path: pathlib.Path | str | None = None,
) -> case_manifest.CaseManifestEntry:
    """Look up a ``CaseManifestEntry`` by ``case`` and optional ``suite``.

    Supports ``"<suite>/<case>"`` notation in ``case`` as well as bare case
    names unique across ``star``, ``binary``, and ``astero``.

    Args:
        case: Case directory name or ``"<suite>/<case>"`` specifier.
        suite: Optional suite override (``"star"``, ``"binary"``, or
            ``"astero"``).
        manifest_path: Optional override path to ``case_manifest.toml``.

    Returns:
        Matching ``CaseManifestEntry``.

    Raises:
        ValueError: If ``case`` does not exist in the manifest or matches more
            than one suite without disambiguation.
    """
    manifest = case_manifest.load_case_manifest(manifest_path)
    eff_suite = suite
    eff_case = case.strip()
    if "/" in eff_case and eff_suite is None:
        head, tail = eff_case.split("/", 1)
        if head in case_manifest.VALID_SUITES:
            eff_suite, eff_case = head, tail

    matches = [
        entry
        for entry in manifest.cases
        if entry.case == eff_case
        and (eff_suite is None or entry.suite == eff_suite)
    ]
    if not matches:
        scope = eff_suite if eff_suite is not None else "any suite"
        raise ValueError(f"case {eff_case!r} not found in manifest for {scope}")
    if len(matches) > 1:
        suites = ", ".join(m.suite for m in matches)
        raise ValueError(
            f"ambiguous case {eff_case!r} across suites ({suites}); "
            "pass suite explicitly"
        )
    return matches[0]


def plot_case_comparison(
    case: str,
    *,
    suite: str | None = None,
    flexmesa_history: pathlib.Path | str | None = None,
    flexmesa_dir: pathlib.Path | str | None = None,
    mesa_history: pathlib.Path | str | None = None,
    manifest_path: pathlib.Path | str | None = None,
    save_path: pathlib.Path | str | None = None,
    termination_code: str = "",
    numerics_mode: str = "",
    controls_ignored: bool = False,
    max_steps: int | None = None,
    n_controls_ignored: int = 0,
    case_entry: case_manifest.CaseManifestEntry | None = None,
    comparison: history_compare.HistoryComparison | None = None,
) -> tuple[mpl_figure.Figure, np.ndarray]:
    """Overplot a FlexMESA run against a stored MESA 26.4.1 reference history.

    Reads solely from stored ASCII tables on disk and evaluates residuals with
    :func:`flexmesa.testing.history_compare.compare_histories`. Never computes
    or synthesizes physics values inside plotting code.

    Args:
        case: Manifest case name or ``"<suite>/<case>"`` identifier.
        suite: Optional suite name (``"star"``, ``"binary"``, or ``"astero"``).
        flexmesa_history: Direct path to FlexMESA ``history.data`` or
            ``binary_history.data``.
        flexmesa_dir: Output root directory containing FlexMESA ``LOGS/`` or
            ``binary_history.data`` (ignored if ``flexmesa_history`` is set).
        mesa_history: Optional explicit path overriding the manifest's stored
            MESA reference file.
        manifest_path: Optional override path to ``case_manifest.toml``.
        save_path: Optional output path for saving a ``300 dpi`` figure.
        termination_code: Optional termination label for title provenance.
        numerics_mode: Optional arithmetic mode label (``"strict"``/``"fast"``).
        controls_ignored: Whether the run skipped unsupported controls.
        max_steps: Optional per-stage step cap used during a smoke trace.
        n_controls_ignored: Number of unsupported controls the run ignored,
            shown in the ``diagnostic`` label when positive.
        case_entry: Optional pre-resolved ``CaseManifestEntry`` to avoid
            re-parsing ``case_manifest.toml``.
        comparison: Optional precomputed ``HistoryComparison`` from a harness
            run.

    Returns:
        Tuple ``(fig, axes)`` with shape ``(3, 2)`` of subplots.

    Raises:
        FileNotFoundError: If the FlexMESA history or MESA reference cannot be
            located.
        ValueError: If the manifest entry is unknown, lacks a stored reference,
            or cannot be parsed cleanly.
    """
    spec = (
        case_entry
        if case_entry is not None
        else resolve_case_entry(case, suite=suite, manifest_path=manifest_path)
    )

    if flexmesa_history is not None:
        run_path = pathlib.Path(flexmesa_history)
    elif flexmesa_dir is not None:
        located = harness_runner.locate_written_history(
            spec.suite, pathlib.Path(flexmesa_dir), True
        )
        if located is None:
            raise FileNotFoundError(
                f"no history file found under {flexmesa_dir} for "
                f"{spec.suite}/{spec.case}"
            )
        run_path = located
    else:
        raise FileNotFoundError(
            "provide either flexmesa_history or flexmesa_dir for "
            f"{spec.suite}/{spec.case}"
        )

    if mesa_history is not None:
        ref_path = pathlib.Path(mesa_history)
    else:
        if not spec.reference_path:
            raise ValueError(
                f"{spec.suite}/{spec.case} has reference_status="
                f"{spec.reference_status!r} and no stored reference_path"
            )
        base_dir = (
            pathlib.Path(manifest_path).parent
            if manifest_path is not None
            else None
        )
        ref_path = harness_runner.resolve_reference_history_path(
            spec.reference_path, root_dir=base_dir
        )

    ref_tbl = history_compare.load_history_file(ref_path)
    run_tbl = history_compare.load_history_file(run_path)

    comp = comparison
    if (
        comp is None
        and spec.independent_var in ref_tbl.columns
        and spec.independent_var in run_tbl.columns
    ):
        try:
            comp = history_compare.compare_histories(ref_tbl, run_tbl, spec)
        except (ValueError, KeyError):
            comp = None

    style.use_mesa_plot_style()
    fig, axes = plt.subplots(
        3,
        2,
        figsize=style.figsize("double", 1.15),
        constrained_layout=True,
    )
    if spec.suite == "binary":
        _compare_panels._plot_binary_comparison_panels(
            axes, ref_tbl, run_tbl, spec, comp
        )
    else:
        _compare_panels._plot_star_comparison_panels(
            axes, ref_tbl, run_tbl, spec, comp
        )

    fig.suptitle(
        _compare_panels.format_provenance_title(
            spec,
            ref_tbl,
            run_tbl,
            comp,
            termination_code=termination_code,
            numerics_mode=numerics_mode,
            controls_ignored=controls_ignored,
            n_controls_ignored=n_controls_ignored,
            max_steps=max_steps,
        ),
        fontsize=9.0,
    )
    _compare_panels.save_figure(fig, save_path)
    return fig, axes


def plot_profile_panels(
    profile_path: pathlib.Path | str,
    *,
    ref_profile_path: pathlib.Path | str | None = None,
    save_path: pathlib.Path | str | None = None,
) -> tuple[mpl_figure.Figure, np.ndarray]:
    """Plot radial structure panels from stored ``profileN.data`` output.

    Displays mass-fraction abundances, thermodynamic ``logT`` / ``logRho``,
    nuclear burning or luminosity, and ``log_D_mix`` shaded by integer
    ``mixing_type`` (convective boundaries), optionally overlaying a stored
    MESA ``26.4.1`` reference profile.

    Args:
        profile_path: Path to FlexMESA ``profileN.data``.
        ref_profile_path: Optional path to reference MESA ``profileN.data``.
        save_path: Optional output image path (``300 dpi``).

    Returns:
        Tuple ``(fig, axes)`` with shape ``(2, 2)``.

    Raises:
        FileNotFoundError: If either profile path is missing.
        ValueError: If the profile file contains zero rows.
    """
    prof = load_profile_file(profile_path)
    ref_prof = (
        load_profile_file(ref_profile_path)
        if ref_profile_path is not None
        else None
    )
    return _profile_panels.render_profile_panels(
        prof, ref_prof, save_path=save_path
    )


def plot_profile_kippenhahn(
    profile_paths: list[pathlib.Path | str],
    *,
    mass_grid_points: int = 120,
    xaxis: Literal["model_number", "star_age"] = "star_age",
    yaxis: Literal["mass", "q", "radius"] = "mass",
    history_path: pathlib.Path | str | None = None,
    save_path: pathlib.Path | str | None = None,
) -> tuple[mpl_figure.Figure, mpl_axes.Axes]:
    """Assemble a Kippenhahn diagram strictly from stored ``profileN.data``s.

    Args:
        profile_paths: Ordered sequence of at least two stored
            ``profileN.data`` files spanning time.
        mass_grid_points: Uniform vertical grid size across ``yaxis``
            (default ``120``).
        xaxis: Horizontal coordinate (``"star_age"`` or ``"model_number"``).
        yaxis: Vertical coordinate (``"mass"``, ``"q"``, or ``"radius"``).
        history_path: Optional ``history.data`` providing core mass boundaries.
        save_path: Optional output image path (``300 dpi``).

    Returns:
        Tuple ``(fig, ax)`` produced by
        :func:`flexmesa.plotting.stellar_plots.plot_kippenhahn` or
        :func:`flexmesa.plotting.kippenhahn.plot_history_kippenhahn`.

    Raises:
        ValueError: If fewer than two profiles are given or required columns
            are absent.
    """
    if len(profile_paths) < 2:
        raise ValueError(
            "plot_profile_kippenhahn requires at least 2 profile files"
        )
    if xaxis != "star_age" or yaxis != "mass" or history_path is not None:
        from flexmesa.plotting import kippenhahn

        return kippenhahn.plot_history_kippenhahn(
            history_path=history_path,
            profile_paths=profile_paths,
            xaxis=xaxis,
            yaxis=yaxis,
            mass_grid_points=mass_grid_points,
            save_path=save_path,
        )
    loaded = [load_profile_file(p) for p in profile_paths]
    for pt in loaded:
        if "mass" not in pt.columns or "mixing_type" not in pt.columns:
            raise ValueError(
                "each profile requires 'mass' and 'mixing_type' columns"
            )
    loaded.sort(key=lambda pt: float(pt.header.get("star_age", 0.0)))
    return _profile_panels.render_profile_kippenhahn(
        loaded,
        mass_grid_points=mass_grid_points,
        save_path=save_path,
    )
