# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-stage step counts of MESA 26.4.1 and FlexMESA, and their difference.

A faithful port takes the same steps as MESA, stage by stage, so the step
count of each stage is a fidelity signal in its own right (``DECISIONS
#69``). MESA prints, at the end of every stage, the model number reached
(``runtime (minutes), retries, steps``). For a stage that starts at model 0
that is the number of steps; for a stage that continues a loaded model it
includes the earlier models. FlexMESA reports the model number at the end of
each stage the same way, so the two are compared directly.

The MESA counts are read from
``benchmarks/history_harness/mesa_stage_steps.json``, written by
``scripts/mesa_reference/extract_stage_steps.py`` from the stored real-MESA
runs. The harness runs a case with its optional stages skipped, so FlexMESA
may report fewer stages than a full MESA run; the stages are aligned from the
end.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib

from flexmesa.testing import case_manifest

__all__ = [
    "StageDelta",
    "load_mesa_stage_steps",
    "stage_deltas",
    "step_bound",
]

_DATA = pathlib.Path("benchmarks") / "history_harness" / "mesa_stage_steps.json"


@dataclasses.dataclass(frozen=True)
class StageDelta:
    """Step counts of one aligned stage.

    Attributes:
        stage: 1-based stage index in FlexMESA's run.
        run: FlexMESA's model number at the end of the stage.
        mesa: MESA's model number at the end of the stage, or ``None`` when
            MESA reports no such stage.
    """

    stage: int
    run: int
    mesa: int | None

    @property
    def delta(self) -> int | None:
        """Return ``run - mesa``, or ``None`` when MESA has no such stage."""
        return None if self.mesa is None else self.run - self.mesa


def load_mesa_stage_steps(
    path: pathlib.Path | None = None,
) -> dict[tuple[str, str], tuple[int, ...]]:
    """Read the MESA per-stage step counts.

    Args:
        path: Optional path of ``mesa_stage_steps.json`` (default: the
            committed file).

    Returns:
        ``{(suite, case): stage steps}``; empty when the file is absent.
    """
    src = path or case_manifest.repo_root() / _DATA
    if not src.is_file():
        return {}
    raw = json.loads(src.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], tuple[int, ...]] = {}
    for label, row in raw["cases"].items():
        suite, _, case = label.partition("/")
        out[(suite, case)] = tuple(int(n) for n in row["stage_steps"])
    return out


def stage_deltas(
    run: tuple[int, ...], mesa: tuple[int, ...]
) -> tuple[StageDelta, ...]:
    """Align FlexMESA's stages with MESA's from the end and pair them.

    Args:
        run: FlexMESA's model number at the end of each stage.
        mesa: MESA's model number at the end of each stage.

    Returns:
        One pair per FlexMESA stage. A run with more stages than MESA gets
        ``mesa=None`` for the extra leading stages.
    """
    offset = len(mesa) - len(run)
    return tuple(
        StageDelta(
            i + 1,
            n,
            mesa[i + offset] if 0 <= i + offset < len(mesa) else None,
        )
        for i, n in enumerate(run)
    )


def step_bound(mesa_steps: int, margin_abs: int, margin_frac: float) -> int:
    """Return the largest stage step count tolerated before a stop.

    Args:
        mesa_steps: MESA's model number at the end of the stage.
        margin_abs: Smallest margin, in steps.
        margin_frac: Margin as a fraction of ``mesa_steps``.

    Returns:
        ``mesa_steps + max(margin_abs, ceil(margin_frac * mesa_steps))``.
    """
    return mesa_steps + max(margin_abs, math.ceil(margin_frac * mesa_steps))
