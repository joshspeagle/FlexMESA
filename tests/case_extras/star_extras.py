# SPDX-License-Identifier: LGPL-3.0-or-later
"""Upstream non-`RSP` `star/test_suite/*/src/run_star_extras.f90` dispatch
(`#148`).
"""

from __future__ import annotations

import re
from typing import Any

_REDO_STEP_RE = re.compile(r"extras_check_model\s*=\s*redo\b")


class CheckRedoStarExtras:
    """Step-redo hook logic from
    `star/test_suite/check_redo/src/run_star_extras.f90:60-128`.
    """

    def __init__(self) -> None:
        self._done_redo = False
        self._doing_restart = False

    def extras_startup(self, _state: dict[str, Any], restart: bool) -> None:
        """Reset one-shot redo state at startup or restore on restart."""
        self._done_redo = bool(restart)
        self._doing_restart = bool(restart)

    def extras_check_model(self, state: dict[str, Any]) -> str:
        """Trigger a step redo once at `model_number >= 10`."""
        if not self._done_redo and not self._doing_restart:
            if int(state.get("model_number", 0)) >= 10:
                self._done_redo = True
                return "redo"
        return "keep_going"

    def extras_finish_step(self, _state: dict[str, Any]) -> str:
        """Clear one-shot redo guard after the step finishes."""
        self._done_redo = False
        return "keep_going"


def match_star_extras(norm_star_text: str) -> Any:
    """Match non-RSP upstream star extras strictly by Fortran source content."""
    if _REDO_STEP_RE.search(norm_star_text):
        return CheckRedoStarExtras()
    return None
