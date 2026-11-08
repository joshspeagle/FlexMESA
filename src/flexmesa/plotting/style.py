# SPDX-License-Identifier: LGPL-3.0-or-later
"""Publication `matplotlib` figure styling for `FlexMESA`.

Configures AASTeX column and text widths (`3.352 in` and `7.100 in`), Computer
Modern serif typography at `9/10/12 pt`, inward ticks on all four sides,
frameless legends, and colorblind-safe Okabe-Ito semantic palettes.
"""

from __future__ import annotations

from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt

AASTEX_COLUMN_WIDTH_IN: float = 242.26653 / 72.27
AASTEX_TEXT_WIDTH_IN: float = 513.11743 / 72.27

OKABE_ITO: dict[str, str] = {
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "bluish_green": "#009E73",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "reddish_purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#000000",
    "grey": "#6B7280",
}


def figsize(
    layout: str = "single",
    aspect: float = 0.78,
) -> tuple[float, float]:
    """Returns publication figure dimensions `(width_in, height_in)`.

    Args:
        layout: Either `"single"` (AASTeX column width `3.352 in`) or
            `"double"` / `"wide"` (AASTeX text width `7.100 in`).
        aspect: Height-to-width ratio (`height / width`).

    Returns:
        Tuple `(width_in, height_in)` in inches.
    """
    width = (
        AASTEX_TEXT_WIDTH_IN
        if layout in ("double", "wide", "text")
        else AASTEX_COLUMN_WIDTH_IN
    )
    return width, width * float(aspect)


def mesa_rc_params() -> dict[str, Any]:
    """Returns the canonical `matplotlib` `rcParams` dictionary.

    Returns:
        Dictionary of `matplotlib` style parameters.
    """
    return {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 10.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 9.0,
        "figure.titlesize": 12.0,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }


def use_mesa_plot_style() -> None:
    """Applies the canonical `FlexMESA` publication style to `mpl.rcParams`."""
    mpl.rcParams.update(mesa_rc_params())


def init_ax(
    ax: Any | None,
    layout: str = "single",
    aspect: float = 0.85,
) -> tuple[Any, Any]:
    """Applies `use_mesa_plot_style` and returns `(fig, ax)`."""
    use_mesa_plot_style()
    if ax is None:
        fig, new_ax = plt.subplots(figsize=figsize(layout, aspect))
        return fig, new_ax
    return ax.figure, ax


def finalize_fig(
    fig: Any,
    ax: Any,
    save_path: str | None = None,
) -> tuple[Any, Any]:
    """Optionally saves `fig` at `300 dpi` and returns `(fig, ax)`."""
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax
