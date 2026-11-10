# SPDX-License-Identifier: LGPL-3.0-or-later
"""Canonical MESA I-VI instrument paper figure reproduction suite (`#110`)."""

from __future__ import annotations

import pathlib

from . import (
    reproduce_mesa_i,
    reproduce_mesa_ii,
    reproduce_mesa_iii,
    reproduce_mesa_iv,
    reproduce_mesa_v,
    reproduce_mesa_vi,
)


def generate_all_mesa_paper_figures(
    root: pathlib.Path | None = None,
) -> dict[str, dict[str, float]]:
    """Generates all 18 MESA I-VI figures and 6 CSV ledgers under `root`."""
    return {
        "mesa_i": reproduce_mesa_i.generate_all(root),
        "mesa_ii": reproduce_mesa_ii.generate_all(root),
        "mesa_iii": reproduce_mesa_iii.generate_all(root),
        "mesa_iv": reproduce_mesa_iv.generate_all(root),
        "mesa_v": reproduce_mesa_v.generate_all(root),
        "mesa_vi": reproduce_mesa_vi.generate_all(root),
    }


__all__ = [
    "generate_all_mesa_paper_figures",
    "reproduce_mesa_i",
    "reproduce_mesa_ii",
    "reproduce_mesa_iii",
    "reproduce_mesa_iv",
    "reproduce_mesa_v",
    "reproduce_mesa_vi",
]
