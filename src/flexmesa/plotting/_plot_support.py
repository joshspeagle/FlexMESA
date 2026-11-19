# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Shared internal helpers for ``flexmesa.plotting`` panels (`#173`).

Provides canonical MESA ``26.4.1`` mixing-type styles
(`const/public/const_def.f90:159-167`), 6-line header history loading, bundled
``data/star_data/plot_info/*.data`` curve readers
(`star/private/pgstar_support.f90:276-395`), radial coordinate selectors, and
bottom-spine mixing-region bars (`star/private/pgstar_power.f90:220-224`).
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import matplotlib.axes as mpl_axes
import matplotlib.collections as mpl_collections
import matplotlib.colors as mpl_colors
import matplotlib.patches as mpl_patches
import matplotlib.transforms as mpl_transforms
import numpy as np

from flexmesa import _rust
from flexmesa.data import fetch
from flexmesa.plotting import _compare_panels, style
from flexmesa.testing import golden, history_compare

if TYPE_CHECKING:
    from flexmesa.plotting import compare as _compare_mod

# Integer mixing codes from MESA 26.4.1 const/public/const_def.f90:159-167
# mapped to (display_name, okabe_ito_hex, hatch_pattern).
MIXING_STYLE: dict[int, tuple[str, str, str]] = {
    1: ("Convective", style.OKABE_ITO["sky_blue"], "///"),
    2: ("Overshoot", style.OKABE_ITO["orange"], "\\\\\\"),
    3: ("Semiconvective", style.OKABE_ITO["bluish_green"], "..."),
    4: ("Thermohaline", style.OKABE_ITO["reddish_purple"], "xxx"),
    5: ("Rotation", style.OKABE_ITO["yellow"], "---"),
    6: ("Rayleigh-Taylor", style.OKABE_ITO["vermilion"], "+++"),
    7: ("Minimum", style.OKABE_ITO["grey"], ""),
    8: ("Anonymous", style.OKABE_ITO["black"], ".."),
    9: ("Leftover Conv.", style.OKABE_ITO["blue"], "//"),
}


def load_history_with_header(
    path: pathlib.Path | str,
) -> tuple[history_compare.HistoryTable, dict[str, Any]]:
    """Load a MESA history table together with its 6-line scalar header.

    Args:
        path: Path to a 6-line MESA ``history.data``/``binary_history.data``
            log or thinned TSV reference history.

    Returns:
        Tuple ``(table, header)`` where ``table`` is a scrubbed
        ``HistoryTable`` of 1D ``float64`` columns ``(n_models,)`` and
        ``header`` maps header keys to scalar values (empty for TSV inputs).
    """
    fpath = pathlib.Path(path)
    tbl = history_compare.load_history_file(fpath)
    with fpath.open(encoding="utf-8") as fh:
        first_line = fh.readline()
    if first_line.startswith("#") or "\t" in first_line:
        return tbl, {}
    parsed = _rust.mesa_read_log_file(str(fpath))
    header = {str(k): v for k, v in dict(parsed["header"]).items()}
    return tbl, header


def load_plot_info_curve(
    filename: str,
    mesa_dir: pathlib.Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Read a two-column ``(logRho, logT)`` curve from MESA ``plot_info``.

    Mirrors ``read_TRho_data`` in ``star/private/pgstar_support.f90:353-395``.

    Args:
        filename: Basename inside ``data/star_data/plot_info/`` (such as
            ``"psi4.data"`` or ``"hydrogen_burn.data"``).
        mesa_dir: Optional MESA ``26.4.1`` root overriding
            :func:`flexmesa.data.fetch.resolve_mesa_test_inputs_dir`.

    Returns:
        Tuple ``(log_rho, log_t)`` of 1D ``float64`` arrays ``(N,)`` in
        ``dex`` ($\mathrm{g\,cm^{-3}}$ and $\mathrm{K}$).

    Raises:
        FileNotFoundError: If the ``plot_info`` file does not exist.
        ValueError: If no valid two-column coordinate pairs are present.
    """
    base = (
        pathlib.Path(mesa_dir)
        if mesa_dir is not None
        else fetch.resolve_mesa_test_inputs_dir()
    )
    cand = base / "data" / "star_data" / "plot_info" / filename
    if not cand.is_file():
        alt = base / "star_data" / "plot_info" / filename
        if alt.is_file():
            cand = alt
        else:
            raise FileNotFoundError(f"missing MESA plot_info file: {cand}")

    rhos: list[float] = []
    temps: list[float] = []
    for raw_line in cand.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            break
        rhos.append(golden.parse_fortran_float(parts[0]))
        temps.append(golden.parse_fortran_float(parts[1]))
    if not rhos:
        raise ValueError(f"{cand}: plot_info file contains zero points")
    return (
        np.asarray(rhos, dtype=np.float64),
        np.asarray(temps, dtype=np.float64),
    )


def profile_xaxis(
    prof: _compare_mod.ProfileTable,
    xaxis: str = "auto",
) -> tuple[np.ndarray, str, bool]:
    r"""Resolve radial profile x-coordinate array, axis label, and inversion.

    Args:
        prof: Loaded ``ProfileTable`` from ``profileN.data``.
        xaxis: Requested coordinate name (``"auto"``, ``"mass"``, ``"q"``,
            ``"radius"``, ``"logR"``, ``"logxm"``, ``"column_depth"``,
            ``"logP"``, ``"zone"``, or any column in ``prof.columns``).

    Returns:
        Tuple ``(x_vals, x_label, reverse_x)`` with ``x_vals`` of shape
        ``(num_zones,)``.

    Raises:
        KeyError: If the requested coordinate cannot be formed from ``prof``.
    """
    key = xaxis.strip()
    if key == "auto":
        for cand in ("mass", "q", "radius", "zone"):
            if cand in prof.columns:
                key = cand
                break
        else:
            return (
                np.arange(1, prof.num_zones + 1, dtype=np.float64),
                "Zone Index",
                False,
            )

    if key == "mass" and "mass" in prof.columns:
        return prof.columns["mass"], r"Enclosed Mass [$M_r / M_\odot$]", False
    if key == "q" and "q" in prof.columns:
        return prof.columns["q"], r"Mass Coordinate [$q = m / M$]", False
    if key == "radius" and "radius" in prof.columns:
        return prof.columns["radius"], r"Radius [$R_r / R_\odot$]", False
    if key == "logR" and "logR" in prof.columns:
        return (
            prof.columns["logR"],
            r"Log Radius [$\log_{10}(R_r / R_\odot)$]",
            False,
        )
    if key == "logP" and "logP" in prof.columns:
        return (
            prof.columns["logP"],
            r"Log Pressure [$\log_{10}(P / \mathrm{dyn\,cm^{-2}})$]",
            True,
        )
    if key == "logxm":
        if "logxm" in prof.columns:
            return (
                prof.columns["logxm"],
                r"Exterior Mass [$\log_{10}((M - M_r) / M_\odot)$]",
                False,
            )
        if "mass" in prof.columns:
            m_arr = prof.columns["mass"]
            hdr_m = prof.header.get("star_mass")
            m_tot = (
                float(hdr_m)
                if isinstance(hdr_m, (int, float))
                else float(np.nanmax(m_arr))
            )
            dm = np.maximum(m_tot - m_arr, 1.0e-99)
            return (
                np.log10(dm),
                r"Exterior Mass [$\log_{10}((M - M_r) / M_\odot)$]",
                False,
            )
    if key in ("column_depth", "log_column_depth"):
        if "log_column_depth" in prof.columns:
            return (
                prof.columns["log_column_depth"],
                r"Log Column Depth [$\log_{10}(y / \mathrm{g\,cm^{-2}})$]",
                False,
            )
        if "column_depth" in prof.columns:
            cd = np.maximum(prof.columns["column_depth"], 1.0e-99)
            return (
                np.log10(cd),
                r"Log Column Depth [$\log_{10}(y / \mathrm{g\,cm^{-2}})$]",
                False,
            )
    if key == "zone":
        z_arr = prof.columns.get(
            "zone", np.arange(1, prof.num_zones + 1, dtype=np.float64)
        )
        return z_arr, "Zone Index", False
    if key in prof.columns:
        return prof.columns[key], key, False
    raise KeyError(
        f"profile coordinate {xaxis!r} not found in columns "
        f"{sorted(prof.columns)}"
    )


def draw_xaxis_mixing_bar(
    ax: mpl_axes.Axes,
    x_vals: np.ndarray,
    prof: _compare_mod.ProfileTable,
) -> None:
    """Draw thin mixing-regime color bars along the bottom spine of ``ax``.

    Reproduces ``show_mix_regions_on_xaxis`` from
    ``star/private/pgstar_power.f90:220-224`` and
    ``star/private/pgstar_abundance.f90:225-229``.

    Args:
        ax: Target Matplotlib ``Axes``.
        x_vals: 1D radial coordinate vector of shape ``(num_zones,)``.
        prof: Profile table containing optional ``mixing_type`` column.
    """
    if "mixing_type" not in prof.columns or x_vals.size < 2:
        return
    valid = _compare_panels.finite_mask(x_vals) & _compare_panels.finite_mask(
        prof.columns["mixing_type"]
    )
    if int(np.sum(valid)) < 2:
        return
    xv = x_vals[valid]
    mt = np.rint(prof.columns["mixing_type"][valid]).astype(np.int64)
    order = np.argsort(xv, kind="mergesort")
    xs = xv[order]
    ms = mt[order]

    edges = np.empty(xs.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (xs[:-1] + xs[1:])
    edges[0] = xs[0]
    edges[-1] = xs[-1]

    trans = mpl_transforms.blended_transform_factory(ax.transData, ax.transAxes)
    by_code: dict[int, list[np.ndarray]] = {}
    for idx in range(ms.size):
        code = int(ms[idx])
        if code not in MIXING_STYLE:
            continue
        x0, x1 = float(edges[idx]), float(edges[idx + 1])
        if x1 <= x0:
            continue
        quad = np.array(
            [[x0, 0.0], [x1, 0.0], [x1, 0.026], [x0, 0.026]],
            dtype=np.float64,
        )
        by_code.setdefault(code, []).append(quad)

    for code, polys in by_code.items():
        _name, color_hex, _hatch = MIXING_STYLE[code]
        coll = mpl_collections.PolyCollection(
            polys,
            facecolors=color_hex,
            edgecolors="none",
            alpha=0.88,
            transform=trans,
            zorder=5,
            clip_on=True,
        )
        ax.add_collection(coll)


def format_profile_subtitle(
    prof: _compare_mod.ProfileTable,
    ref_prof: _compare_mod.ProfileTable | None = None,
) -> str:
    """Format a provenance subtitle for single or paired profile panels."""
    mod_no = prof.header.get("model_number", "?")
    age_yr = prof.header.get("star_age")
    age_str = (
        f", age={float(age_yr):.3e} yr"
        if isinstance(age_yr, (int, float))
        else ""
    )
    compiler = str(prof.header.get("compiler", "")).strip()
    ver = str(prof.header.get("version_number", "")).strip().strip('"')
    src = "FlexMESA" if compiler == "rustc" or ver == "FlexMESA" else "MESA"
    mode = str(prof.header.get("math_backend", "")).strip().strip('"')
    meta = ", ".join(p for p in (src, ver if ver != src else "", mode) if p)
    base = f"model={mod_no}{age_str}, zones={prof.num_zones} ({meta})"
    if ref_prof is None:
        return base
    ref_ver = str(ref_prof.header.get("version_number", "")).strip().strip('"')
    ref_sdk = (
        str(ref_prof.header.get("MESA_SDK_version", "")).strip().strip('"')
    )
    ref_parts = [p for p in (ref_ver or "MESA ref", ref_sdk) if p]
    return f"{base} | Ref: {', '.join(ref_parts)} ({ref_prof.num_zones} zones)"


def render_mixing_grid_layers(
    ax: mpl_axes.Axes,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    mix_arr: np.ndarray,
) -> list[mpl_patches.Patch]:
    """Render categorical ``mix_type_grid`` contours onto ``ax``."""
    handles: list[mpl_patches.Patch] = []
    present_codes = [
        int(v)
        for v in np.unique(
            np.rint(np.where(np.isfinite(mix_arr), mix_arr, 0.0))
        )
        if int(v) in MIXING_STYLE
    ]
    for code in sorted(present_codes):
        m_name, m_hex, m_hatch = MIXING_STYLE[code]
        mask = (np.rint(mix_arr) == code).astype(np.float64)
        if not np.any(mask > 0.5):
            continue
        face = "none" if code == 1 else mpl_colors.to_rgba(m_hex, 0.22)
        cs = ax.contourf(
            x_arr,
            y_arr,
            mask,
            levels=[0.5, 1.5],
            colors=[face],
            hatches=[m_hatch] if m_hatch else None,
            zorder=2,
        )
        if code != 1:
            cs.set_edgecolor(mpl_colors.to_rgba(m_hex, 0.78))
        handles.append(
            mpl_patches.Patch(
                facecolor=(
                    "none" if code == 1 else mpl_colors.to_rgba(m_hex, 0.28)
                ),
                edgecolor=style.OKABE_ITO["black"] if code == 1 else m_hex,
                hatch=m_hatch or None,
                label=m_name,
            )
        )
    return handles


def plot_profile_pair(
    ax: mpl_axes.Axes,
    prof: _compare_mod.ProfileTable,
    x_run: np.ndarray,
    ref_prof: _compare_mod.ProfileTable | None,
    x_ref: np.ndarray | None,
    col: str,
    *,
    label: str,
    color: str,
    ls: str = "-",
    lw: float = 1.5,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> bool:
    """Overplot optional MESA reference and candidate profile ``col``."""
    drawn = False
    if ref_prof is not None and x_ref is not None and col in ref_prof.columns:
        yr = ref_prof.columns[col]
        if transform is not None:
            yr = transform(yr)
        drawn |= _compare_panels._plot_series(
            ax,
            x_ref,
            yr,
            label=f"{label} MESA",
            color=color,
            ls="--",
            lw=1.05,
            alpha=0.72,
            zorder=2,
        )
    if col in prof.columns:
        yf = prof.columns[col]
        if transform is not None:
            yf = transform(yf)
        drawn |= _compare_panels._plot_series(
            ax,
            x_run,
            yf,
            label=label,
            color=color,
            ls=ls,
            lw=lw,
            zorder=3,
        )
    return drawn
