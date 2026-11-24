# SPDX-License-Identifier: GPL-3.0-or-later
r"""Upstream ``pgstar`` and ``pgbinary`` ``Text_Summary<N>`` table renderer.

Ports ``Summary_plot`` from ``star/private/pgstar_summary.f90:337-447`` and
``binary/private/pgbinary_summary.f90:338-445`` together with upstream default
``Text_Summary1`` grids (`star/defaults/pgstar.defaults:6401-6447`,
`binary/defaults/pgbinary.defaults:1915-1925`) strictly from saved
``history.data`` or ``binary_history.data`` tables (`#173`).
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
from collections.abc import Sequence
from typing import Any

import matplotlib.axes as mpl_axes
import matplotlib.figure as mpl_figure
import matplotlib.patches as mpl_patches

from flexmesa.plotting import _compare_panels, _plot_support, style
from flexmesa.testing import history_compare

# Default star Text_Summary1 layout (8 rows x 4 cols) from
# star/defaults/pgstar.defaults:6401-6447.
DEFAULT_STAR_TEXT_SUMMARY_LAYOUT: tuple[tuple[str, ...], ...] = (
    ("model_number", "star_mass", "log_cntr_T", "log_Lnuc"),
    ("log_star_age", "log_abs_mdot", "log_cntr_Rho", "log_Lneu"),
    ("log_dt", "he_core_mass", "log_center_P", "log_LH"),
    ("log_L", "co_core_mass", "center h1", "log_LHe"),
    ("log_Teff", "cz_bot_mass", "center he4", "log_LZ"),
    ("log_R", "cz_top_mass", "center c12", "num_zones"),
    ("log_g", "cz_bot_radius", "center n14", "num_retries"),
    ("log_surf_cell_P", "cz_top_radius", "center o16", ""),
)

# Default binary Text_Summary1 layout (5 rows x 1 col) from
# binary/defaults/pgbinary.defaults:1915-1925.
DEFAULT_BINARY_TEXT_SUMMARY_LAYOUT: tuple[tuple[str, ...], ...] = (
    ("model_number",),
    ("jdot",),
    ("mass_ratio",),
    ("period_days",),
    ("rl_relative_overflow_1",),
)

_COLUMN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "log_cntr_t": ("log_cntr_T", "log_center_T"),
    "log_center_t": ("log_center_T", "log_cntr_T"),
    "log_cntr_rho": ("log_cntr_Rho", "log_center_Rho"),
    "log_center_rho": ("log_center_Rho", "log_cntr_Rho"),
    "log_cntr_p": ("log_cntr_P", "log_center_P"),
    "log_center_p": ("log_center_P", "log_cntr_P"),
    "jdot": ("Jdot", "jdot"),
    "j_orb": ("J_orb", "j_orb"),
    "age": ("age", "star_age"),
    "star_age": ("star_age", "age"),
}

_INT_COLUMN_NAMES: frozenset[str] = frozenset(
    {
        "model_number",
        "num_zones",
        "num_retries",
        "num_backups",
        "num_iters",
        "donor_index",
        "point_mass_index",
    }
)


@dataclasses.dataclass(frozen=True)
class TextSummarySpec:
    """Parsed ``Text_Summary<N>`` table specification from an inlist.

    Attributes:
        num_rows: Number of rows in the text summary grid.
        num_cols: Number of columns in the text summary grid.
        title: Optional title string from ``Text_Summary<N>_title``.
        layout: Tuple of ``num_rows`` tuples, each of length ``num_cols``,
            giving the requested history column names (``""`` for blank cells).
    """

    num_rows: int
    num_cols: int
    title: str
    layout: tuple[tuple[str, ...], ...]


def _parse_2d_subscript(raw: str | None) -> tuple[int, int] | None:
    """Parse a 1-based Fortran 2D subscript ``'row, col'``."""
    if raw is None or "," not in raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2 or not (parts[0].isdigit() and parts[1].isdigit()):
        return None
    r_i, c_i = int(parts[0]), int(parts[1])
    if r_i < 1 or c_i < 1:
        return None
    return r_i, c_i


def parse_inlist_text_summary_spec(
    inlist_path: pathlib.Path | str,
    *,
    summary_index: int = 1,
    namelist_name: str | None = None,
) -> TextSummarySpec:
    """Parse ``Text_Summary<N>_*`` layout controls from an ``inlist`` file.

    Mirrors ``Text_Summary<N>_num_rows``, ``Text_Summary<N>_num_cols``,
    ``Text_Summary<N>_title``, and ``Text_Summary<N>_name(row, col)`` in
    ``star/defaults/pgstar.defaults:6401-6447`` and
    ``binary/defaults/pgbinary.defaults:1915-1925``.

    Args:
        inlist_path: Path to an ``inlist`` or ``.defaults`` file containing a
            ``&pgstar`` or ``&pgbinary`` namelist block.
        summary_index: 1-based ``Text_Summary<N>`` index (``1..9``).
        namelist_name: Optional filter restricting parsing to ``"pgstar"`` or
            ``"pgbinary"``.

    Returns:
        Populated :class:`TextSummarySpec`.

    Raises:
        ValueError: If ``summary_index`` is outside ``1..9`` or no rows/cols
            are configured.
    """
    if not 1 <= summary_index <= 9:
        raise ValueError(
            f"summary_index must be in 1..9, got {summary_index!r}"
        )
    blocks = _plot_support.load_plotting_namelist_blocks(
        inlist_path, namelist_name=namelist_name
    )
    prefix = f"text_summary{summary_index}_"
    num_rows = 0
    num_cols = 0
    title = ""
    cells: dict[tuple[int, int], str] = {}

    for blk in blocks:
        bname = blk.name.lower()
        if namelist_name is not None and bname != namelist_name.lower():
            continue
        if bname not in ("pgstar", "pgbinary"):
            continue
        for asg in blk.assignments:
            k_low = asg.name.lower()
            if not k_low.startswith(prefix):
                continue
            tail = k_low[len(prefix) :]
            if tail == "num_rows":
                num_rows = int(asg.value)
            elif tail == "num_cols":
                num_cols = int(asg.value)
            elif tail == "title":
                title = str(asg.value).strip()
            elif tail == "name":
                if asg.subscript is not None and ":" in asg.subscript:
                    cells.clear()
                    continue
                rc = _parse_2d_subscript(asg.subscript)
                if rc is not None:
                    cells[rc] = str(asg.value).strip()

    if num_rows <= 0 and cells:
        num_rows = max(r for r, _ in cells)
    if num_cols <= 0 and cells:
        num_cols = max(c for _, c in cells)
    if num_rows <= 0 or num_cols <= 0:
        raise ValueError(
            f"Text_Summary{summary_index} has no rows/cols in {inlist_path}"
        )

    grid = tuple(
        tuple(cells.get((r, c), "") for c in range(1, num_cols + 1))
        for r in range(1, num_rows + 1)
    )
    return TextSummarySpec(
        num_rows=num_rows, num_cols=num_cols, title=title, layout=grid
    )


def _lookup_history_scalar(
    tbl: history_compare.HistoryTable,
    raw_name: str,
    row_idx: int,
    lower_map: dict[str, str],
) -> tuple[str, float] | None:
    """Resolve one history column value at ``row_idx`` without inventing data.

    Spaces are mapped to underscores following
    ``star/private/pgstar_support.f90:552-559`` (e.g. ``'center h1'`` ->
    ``'center_h1'``). Exact upstream MESA column aliases such as
    ``log_cntr_T`` / ``log_center_T`` are resolved when present.
    """
    cleaned = raw_name.strip()
    if not cleaned:
        return None
    norm = cleaned.replace(" ", "_")
    candidates: list[str] = [cleaned, norm]
    norm_low = norm.lower()
    if norm_low in _COLUMN_SYNONYMS:
        candidates.extend(_COLUMN_SYNONYMS[norm_low])
    for cand in candidates:
        if cand in tbl.columns:
            val = float(tbl.columns[cand][row_idx])
            if math.isfinite(val):
                return cleaned, val
            return None
    if norm_low in lower_map:
        actual = lower_map[norm_low]
        val = float(tbl.columns[actual][row_idx])
        if math.isfinite(val):
            return cleaned, val
    return None


def format_summary_value(name: str, value: float) -> str:
    """Format a scalar summary value matching ``pgstar_summary.f90:420-442``.

    Integer columns and exact integer values render as integers; finite floats
    with ``1e-3 < |v| < 1e3`` render in fixed-point; other non-zero magnitudes
    render in scientific notation (``1pe10.3``).
    """
    norm_low = name.strip().replace(" ", "_").lower()
    if value == 0.0:
        return "0"
    nearest = round(value)
    is_exact_int = abs(value - float(nearest)) < 1.0e-10 * max(
        1.0e-10, abs(value)
    )
    if norm_low in _INT_COLUMN_NAMES or (
        is_exact_int and abs(value) < 1.0e12 and abs(value) >= 1.0
    ):
        return f"{nearest:d}"
    if 1.0e-3 < abs(value) < 1.0e3:
        return f"{value:.4f}"
    return f"{value:.3e}"


def _select_row(
    tbl: history_compare.HistoryTable,
    model_number: int | None,
    row_index: int,
) -> int:
    if model_number is not None and "model_number" in tbl.columns:
        diffs = abs(tbl.columns["model_number"] - float(model_number))
        return int(diffs.argmin())
    idx = row_index if row_index >= 0 else tbl.num_rows + row_index
    return max(0, min(idx, tbl.num_rows - 1))


def plot_text_summary(
    history_path: pathlib.Path | str,
    *,
    layout: Sequence[Sequence[str]] | None = None,
    inlist_path: pathlib.Path | str | None = None,
    summary_index: int = 1,
    is_binary: bool = False,
    model_number: int | None = None,
    row_index: int = -1,
    ref_history_path: pathlib.Path | str | None = None,
    title: str = "",
    ax: mpl_axes.Axes | None = None,
    save_path: pathlib.Path | str | None = None,
) -> tuple[mpl_figure.Figure, mpl_axes.Axes]:
    r"""Render a ``Text_Summary<N>`` key-value table from saved history output.

    Implements ``Summary_plot`` from ``star/private/pgstar_summary.f90:337-447``
    and ``binary/private/pgbinary_summary.f90:338-445`` strictly from saved
    ``history.data`` or ``binary_history.data`` files.

    Args:
        history_path: Path to ``history.data`` or ``binary_history.data``.
        layout: Optional 2D sequence ``rows x cols`` of history column names
            overriding upstream ``Text_Summary1`` defaults.
        inlist_path: Optional ``inlist`` or ``.defaults`` file defining
            ``Text_Summary<N>_*`` controls.
        summary_index: 1-based ``Text_Summary<N>`` index (``1..9``) used when
            parsing ``inlist_path``.
        is_binary: When ``True`` and neither ``layout`` nor ``inlist_path`` is
            supplied, use ``pgbinary`` ``Text_Summary1`` defaults
            (`binary/defaults/pgbinary.defaults:1915-1925`).
        model_number: Optional ``model_number`` selecting the displayed row.
        row_index: 0-based (or negative) row index when ``model_number`` is
            ``None`` (default ``-1``, terminal model).
        ref_history_path: Optional MESA ``26.4.1`` reference history recorded
            in the panel title for provenance.
        title: Optional panel title overriding ``Text_Summary<N>_title``.
        ax: Optional target ``matplotlib.axes.Axes``.
        save_path: Optional output image path (`300 dpi`).

    Returns:
        Tuple ``(fig, ax)``.

    Raises:
        KeyError: If none of the requested columns exist in ``history_path``.
        ValueError: If ``layout`` is empty.
    """
    tbl, _ = _plot_support.load_history_with_header(history_path)
    ref_tbl: history_compare.HistoryTable | None = None
    ref_hdr: dict[str, Any] | None = None
    if ref_history_path is not None:
        ref_tbl, ref_hdr = _plot_support.load_history_with_header(
            ref_history_path
        )

    parsed_title = ""
    if layout is not None:
        eff_layout = tuple(tuple(str(c) for c in row) for row in layout)
    elif inlist_path is not None:
        spec = parse_inlist_text_summary_spec(
            inlist_path,
            summary_index=summary_index,
            namelist_name="pgbinary" if is_binary else None,
        )
        eff_layout = spec.layout
        parsed_title = spec.title
    else:
        eff_layout = (
            DEFAULT_BINARY_TEXT_SUMMARY_LAYOUT
            if is_binary
            else DEFAULT_STAR_TEXT_SUMMARY_LAYOUT
        )

    num_rows = len(eff_layout)
    num_cols = max((len(r) for r in eff_layout), default=0)
    if num_rows == 0 or num_cols == 0:
        raise ValueError("Text_Summary layout must be non-empty")

    r_idx = _select_row(tbl, model_number, row_index)
    lower_map = {c.lower(): c for c in tbl.columns}
    resolved_cells: list[tuple[int, int, str, str]] = []
    for r_i, row_names in enumerate(eff_layout):
        for c_i, col_name in enumerate(row_names):
            hit = _lookup_history_scalar(tbl, col_name, r_idx, lower_map)
            if hit is None:
                continue
            disp_name, scalar_val = hit
            resolved_cells.append(
                (
                    r_i,
                    c_i,
                    disp_name,
                    format_summary_value(disp_name, scalar_val),
                )
            )

    if not resolved_cells:
        req = [n for row in eff_layout for n in row if n.strip()]
        raise KeyError(
            f"Text_Summary found none of requested columns {req!r} in "
            f"{history_path}"
        )

    standalone = ax is None
    aspect = max(0.22, min(0.85, 0.08 * num_rows + 0.10))
    fig, ax = style.init_ax(ax, "double" if num_cols >= 3 else "single", aspect)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()

    frame = mpl_patches.FancyBboxPatch(
        (0.005, 0.02),
        0.99,
        0.96,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        facecolor="#FAFAFA",
        edgecolor=style.OKABE_ITO["grey"],
        linewidth=0.7,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(frame)

    # Scale typography so dense 8 x 4 grids stay legible with zero overlap.
    font_pt = max(5.2, min(8.2, 24.0 / max(num_cols * 1.15, num_rows * 0.52)))
    dypos = 0.88 / max(num_rows, 1)
    y_top = 0.92
    dxval = 0.012 if num_cols >= 3 else 0.025

    for r_i, c_i, disp_name, val_str in resolved_cells:
        # Center each key/value pair around (col + 0.55) / num_cols matching
        # pgstar_summary.f90:404-412 with space for longer column labels.
        xpos0 = (float(c_i) + 0.56) / float(num_cols)
        ypos = y_top - (float(r_i) + 0.35) * dypos
        ax.text(
            xpos0 - 0.5 * dxval,
            ypos,
            disp_name,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=font_pt,
            color=style.OKABE_ITO["grey"],
            family="monospace",
            clip_on=True,
        )
        ax.text(
            xpos0 + 0.5 * dxval,
            ypos,
            val_str,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=font_pt,
            color=style.OKABE_ITO["black"],
            family="monospace",
            clip_on=True,
        )

    eff_title = title or parsed_title
    if not eff_title and standalone:
        eff_title = "Binary Text Summary" if is_binary else "Text Summary"
    if eff_title or ref_tbl is not None:
        _plot_support.set_history_panel_title(
            ax,
            eff_title or "Text Summary",
            tbl,
            ref_tbl,
            ref_path=ref_history_path,
            ref_hdr=ref_hdr,
            standalone=standalone,
        )
    _compare_panels.save_figure(fig, save_path)
    return fig, ax


def plot_pgbinary_text_summary(
    binary_history: pathlib.Path | str,
    *,
    layout: Sequence[Sequence[str]] | None = None,
    inlist_path: pathlib.Path | str | None = None,
    summary_index: int = 1,
    model_number: int | None = None,
    row_index: int = -1,
    mesa_binary_history: pathlib.Path | str | None = None,
    title: str = "",
    ax: mpl_axes.Axes | None = None,
    save_path: pathlib.Path | str | None = None,
) -> tuple[mpl_figure.Figure, mpl_axes.Axes]:
    r"""Render a ``pgbinary`` ``Text_Summary<N>`` table from binary history.

    Implements ``Summary_plot`` from
    ``binary/private/pgbinary_summary.f90:338-445`` using upstream default
    rows from ``binary/defaults/pgbinary.defaults:1915-1925``.

    Args:
        binary_history: Path to ``binary_history.data``.
        layout: Optional 2D sequence ``rows x cols`` overriding defaults.
        inlist_path: Optional ``inlist`` defining ``Text_Summary<N>_*``.
        summary_index: 1-based ``Text_Summary<N>`` index (``1..9``).
        model_number: Optional ``model_number`` selecting the displayed row.
        row_index: Row index when ``model_number`` is ``None`` (default ``-1``).
        mesa_binary_history: Optional MESA ``26.4.1`` reference binary history.
        title: Optional panel title.
        ax: Optional target ``matplotlib.axes.Axes``.
        save_path: Optional output image path (`300 dpi`).

    Returns:
        Tuple ``(fig, ax)``.
    """
    return plot_text_summary(
        binary_history,
        layout=layout,
        inlist_path=inlist_path,
        summary_index=summary_index,
        is_binary=True,
        model_number=model_number,
        row_index=row_index,
        ref_history_path=mesa_binary_history,
        title=title,
        ax=ax,
        save_path=save_path,
    )
