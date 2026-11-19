# SPDX-License-Identifier: LGPL-3.0-or-later
"""Internal cell dispatcher and namelist parser for ``pgstar_grid`` (`#173`)."""

from __future__ import annotations

import pathlib
from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.axes as mpl_axes
import numpy as np

from flexmesa import constants
from flexmesa.plotting import (
    compare,
    kippenhahn,
    pgbinary_panels,
    pgstar_panels,
    pgstar_summary,
    stellar_plots,
)
from flexmesa.schema import namelist


def parse_grid_namelist(
    inlist_path: pathlib.Path | str,
    *,
    grid_index: int,
    namelist_name: str | None,
) -> tuple[int, int, str, list[tuple[str, int, int, int, int]]]:
    """Extract ``(num_rows, num_cols, title, cells)`` from ``inlist_path``."""
    text = pathlib.Path(inlist_path).read_text(encoding="utf-8")
    blocks = namelist.parse_namelist_text(text)
    prefix = f"grid{grid_index}_"
    num_rows = num_cols = num_plots = 0
    title = ""
    names: dict[int, str] = {}
    rows: dict[int, int] = {}
    cols: dict[int, int] = {}
    rspan: dict[int, int] = {}
    cspan: dict[int, int] = {}
    star_sub: dict[str, str] = {}

    for blk in blocks:
        bname = blk.name.lower()
        if namelist_name is not None and bname != namelist_name.lower():
            continue
        if bname not in ("pgstar", "pgbinary"):
            continue
        for asg in blk.assignments:
            k_low = asg.name.lower()
            if k_low in ("star1_plot_name", "star2_plot_name"):
                star_sub[k_low[:5]] = str(asg.value).strip()
            if not k_low.startswith(prefix):
                continue
            tail = k_low[len(prefix) :]
            sub_i = (
                int(asg.subscript)
                if asg.subscript is not None and asg.subscript.isdigit()
                else None
            )
            if tail == "num_rows":
                num_rows = int(asg.value)
            elif tail == "num_cols":
                num_cols = int(asg.value)
            elif tail == "num_plots":
                num_plots = int(asg.value)
            elif tail == "title":
                title = str(asg.value)
            elif tail == "plot_name" and sub_i is not None:
                names[sub_i] = str(asg.value).strip()
            elif tail == "plot_row" and sub_i is not None:
                rows[sub_i] = int(asg.value)
            elif tail == "plot_col" and sub_i is not None:
                cols[sub_i] = int(asg.value)
            elif tail == "plot_rowspan" and sub_i is not None:
                rspan[sub_i] = int(asg.value)
            elif tail == "plot_colspan" and sub_i is not None:
                cspan[sub_i] = int(asg.value)

    max_slot = num_plots if num_plots > 0 else max(names, default=0)
    cells: list[tuple[str, int, int, int, int]] = []
    for slot in range(1, max_slot + 1):
        pname = names.get(slot, "").strip()
        if not pname:
            continue
        p_low = pname.lower()
        if star_sub.get(p_low):
            pname = f"{pname}:{star_sub[p_low]}"
        cells.append(
            (
                pname,
                max(rows.get(slot, 1) - 1, 0),
                max(cols.get(slot, 1) - 1, 0),
                max(rspan.get(slot, 1), 1),
                max(cspan.get(slot, 1), 1),
            )
        )
    if not cells or num_rows <= 0 or num_cols <= 0:
        raise ValueError(
            f"Grid{grid_index} specification incomplete in {inlist_path}"
        )
    return num_rows, num_cols, title, cells


def dispatch_grid_cell(
    ax: mpl_axes.Axes,
    panel_name: str,
    *,
    history_path: pathlib.Path | str | None,
    ref_history_path: pathlib.Path | str | None,
    profile_path: pathlib.Path | str | None,
    ref_profile_path: pathlib.Path | str | None,
    profile_paths: Sequence[pathlib.Path | str] | None,
    binary_history_path: pathlib.Path | str | None,
    ref_binary_history_path: pathlib.Path | str | None,
    star1_history_path: pathlib.Path | str | None,
    star2_history_path: pathlib.Path | str | None,
    star1_profile_path: pathlib.Path | str | None,
    star2_profile_path: pathlib.Path | str | None,
    mesa_dir: pathlib.Path | str | None,
) -> None:
    """Render one named ``pgstar``/``pgbinary`` panel into ``ax``."""
    from flexmesa.plotting import pgstar_grid as _pg_grid

    raw = panel_name.strip()
    low = raw.lower()

    if low.startswith(("star1", "star2")):
        star_no = 1 if low.startswith("star1") else 2
        sub_name = raw.split(":", 1)[1].strip() if ":" in raw else "HR"
        s_hist = star1_history_path if star_no == 1 else star2_history_path
        s_prof = star1_profile_path if star_no == 1 else star2_profile_path
        dispatch_grid_cell(
            ax,
            sub_name,
            history_path=s_hist,
            ref_history_path=None,
            profile_path=s_prof,
            ref_profile_path=None,
            profile_paths=None,
            binary_history_path=binary_history_path,
            ref_binary_history_path=ref_binary_history_path,
            star1_history_path=None,
            star2_history_path=None,
            star1_profile_path=None,
            star2_profile_path=None,
            mesa_dir=mesa_dir,
        )
        ax.set_title(f"Star {star_no}: {ax.get_title()}", fontsize=8.8)
        return

    if low in _pg_grid.TRACK_PRESETS:
        h_src = history_path or binary_history_path
        if h_src is None:
            raise ValueError(f"panel {raw!r} requires history_path")
        r_src = (
            ref_history_path
            if history_path is not None
            else (ref_history_path or ref_binary_history_path)
        )
        _pg_grid.plot_history_track(
            h_src,
            preset=raw,
            ref_history_path=r_src,
            mesa_dir=mesa_dir,
            ax=ax,
        )
        return

    if low.startswith("history_track:"):
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"expected 'History_Track:<x_col>:<y_col>', got {raw!r}"
            )
        h_src = history_path or binary_history_path
        if h_src is None:
            raise ValueError(f"panel {raw!r} requires history_path")
        r_src = (
            ref_history_path
            if history_path is not None
            else (ref_history_path or ref_binary_history_path)
        )
        _pg_grid.plot_history_track(
            h_src,
            parts[1],
            parts[2],
            ref_history_path=r_src,
            mesa_dir=mesa_dir,
            ax=ax,
        )
        return

    if low == "kipp":
        kippenhahn.plot_history_kippenhahn(
            history_path=history_path,
            profile_paths=profile_paths,
            ax=ax,
        )
        return
    if low == "trho_profile":
        if profile_path is None:
            raise ValueError("TRho_Profile requires profile_path")
        pgstar_panels.plot_trho_profile(
            profile_path,
            ref_profile_path=ref_profile_path,
            mesa_dir=mesa_dir,
            ax=ax,
        )
        return

    prof_map: dict[str, Callable[..., Any]] = {
        "power": pgstar_panels.plot_power_profile,
        "abundance": pgstar_panels.plot_abundance_profile,
        "mixing": pgstar_panels.plot_mixing_profile,
        "summary_profile": pgstar_summary.plot_summary_profile,
        "summary_burn": pgstar_summary.plot_summary_burn_profile,
    }
    if low in prof_map:
        if profile_path is None:
            raise ValueError(f"{raw} requires profile_path")
        prof_map[low](profile_path, ref_profile_path=ref_profile_path, ax=ax)
        return

    if low == "mode_prop":
        if profile_path is None:
            raise ValueError("Mode_Prop requires profile_path")
        pt = compare.load_profile_file(profile_path)
        r_rsun = pt.columns["radius"]
        r_max = max(float(np.nanmax(r_rsun)), 1.0e-30)
        r_cm = pt.columns.get("radius_cm")
        if r_cm is None:
            r_cm = r_rsun * float(pt.header.get("rsun", constants.RSUN))
        stellar_plots.plot_propagation_diagram(
            r_rsun / r_max,
            pt.columns["brunt_N2"],
            pt.columns["csound"],
            r_cm,
            ax=ax,
        )
        return
    if low == "summary_history":
        if binary_history_path is not None and history_path is None:
            pgbinary_panels.plot_pgbinary_summary_history(
                binary_history_path,
                mesa_binary_history=ref_binary_history_path,
                ax=ax,
            )
            return
        if history_path is None:
            raise ValueError("Summary_History requires history_path")
        _pg_grid.plot_summary_history(
            history_path, ref_history_path=ref_history_path, ax=ax
        )
        return
    if low == "orbit":
        b_src = binary_history_path or history_path
        if b_src is None:
            raise ValueError("Orbit requires binary_history_path")
        pgbinary_panels.plot_pgbinary_orbit(
            b_src, mesa_binary_history=ref_binary_history_path, ax=ax
        )
        return

    raise ValueError(f"unsupported pgstar/pgbinary grid panel {raw!r}")
