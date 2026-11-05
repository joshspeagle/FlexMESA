# SPDX-License-Identifier: LGPL-3.0-or-later
"""Public interface for `flexmesa.auto_diff` (`auto_diff` module port)."""

from __future__ import annotations

from flexmesa.auto_diff import kernels, order1, order2

STAR_ORDER1_VARS = order1.STAR_ORDER1_VARS
DualOrder1 = order1.DualOrder1
AutoDiffReal1VarOrder1 = order1.AutoDiffReal1VarOrder1
AutoDiffReal2VarOrder1 = order1.AutoDiffReal2VarOrder1
AutoDiffReal4VarOrder1 = order1.AutoDiffReal4VarOrder1
AutoDiffReal4Coeff = order1.AutoDiffReal4Coeff
AutoDiffRealStarOrder1 = order1.AutoDiffRealStarOrder1
AutoDiffReal2VarOrder2 = order2.AutoDiffReal2VarOrder2
AutoDiffReal2ndOrder = order2.AutoDiffReal2ndOrder

__all__: list[str] = [
    "STAR_ORDER1_VARS",
    "AutoDiffReal1VarOrder1",
    "AutoDiffReal2VarOrder1",
    "AutoDiffReal2VarOrder2",
    "AutoDiffReal2ndOrder",
    "AutoDiffReal4Coeff",
    "AutoDiffReal4VarOrder1",
    "AutoDiffRealStarOrder1",
    "DualOrder1",
    "kernels",
]
