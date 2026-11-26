# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-case budgets for harness campaigns, derived from the baseline run.

A uniform time budget makes a campaign wait the full budget for every case
that is slow or has stopped making progress. ``--budget-from-baseline`` gives
each case a wall-time budget of ``max(K x its recorded wall time, floor)``
capped at ``--timeout-sec``, and ``--mesa-step-factor`` bounds its steps at
a multiple of the step count of the MESA reference run. A case stopped by
either bound is inconclusive and is compared as a prefix of its baseline row
(``DECISIONS #59``, ``DECISIONS #68``).

The committed baseline records wall times as zero so that it diffs
deterministically (``DECISIONS #58``), so the wall times come from a
previous harness ``--json-out`` report of the same machine class.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

from flexmesa.testing import _case_verdict, baseline_regression, case_manifest

__all__ = [
    "CaseBudget",
    "derive_budgets",
    "load_wall_times",
    "passing_keys",
]


@dataclasses.dataclass(frozen=True)
class CaseBudget:
    """Wall-time and step budget of one case.

    Attributes:
        timeout_sec: Wall-clock budget in seconds (``s``).
        mesa_steps: Step count of the MESA reference run, or ``None``.
        step_factor: Step bound as a multiple of ``mesa_steps``, or ``None``.
        from_baseline: True when ``timeout_sec`` was derived from a recorded
            wall time rather than being the plain ``--timeout-sec`` cap.
    """

    timeout_sec: float
    mesa_steps: int | None = None
    step_factor: float | None = None
    from_baseline: bool = False


def load_wall_times(path: pathlib.Path | str) -> dict[tuple[str, str], float]:
    """Read the wall time of each executed case from a harness report.

    Args:
        path: A harness ``--json-out`` report (or any JSON with a ``cases``
            list whose rows carry ``suite``, ``case``, ``executed`` and
            ``wall_time_sec``).

    Returns:
        ``{(suite, case): wall_time_sec}`` for rows that ran to completion
        (not cut off by a budget) and recorded a positive wall time.

    Raises:
        ValueError: If the file has no ``cases`` list.
    """
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    rows = payload.get("cases")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: no 'cases' list")
    out: dict[tuple[str, str], float] = {}
    for row in rows:
        wall = float(row.get("wall_time_sec") or 0.0)
        if (
            row.get("executed")
            and wall > 0.0
            and row.get("run_status")
            not in baseline_regression.RESOURCE_LIMITED_STATUSES
        ):
            out[(str(row["suite"]), str(row["case"]))] = wall
    return out


def passing_keys(
    baseline_records: list[_case_verdict.CaseHarnessRecord],
) -> frozenset[tuple[str, str]]:
    """Return the cases whose baseline verdict is ``pass``.

    Args:
        baseline_records: Records rebuilt from the committed baseline run.

    Returns:
        ``(suite, case)`` keys of the cases for the ``passes`` preset.
    """
    return frozenset(
        (b.suite, b.case)
        for b in baseline_records
        if b.executed and b.verdict == "pass"
    )


def derive_budgets(
    entries: list[case_manifest.CaseManifestEntry],
    baseline_records: list[_case_verdict.CaseHarnessRecord],
    wall_times: dict[tuple[str, str], float],
    *,
    timeout_sec: float,
    factor: float,
    floor_sec: float,
    step_factor: float | None,
) -> dict[tuple[str, str], CaseBudget]:
    """Derive the budget of every selected case.

    A case with a recorded wall time ``w`` gets
    ``min(timeout_sec, max(factor * w, floor_sec))``. A case without one
    (not in ``wall_times``, or its baseline row ended on a budget) keeps the
    full ``timeout_sec``. The step bound applies to every case whose baseline
    row records the MESA step count.

    Args:
        entries: Selected manifest entries.
        baseline_records: Records rebuilt from the committed baseline run.
        wall_times: Recorded wall times from :func:`load_wall_times`.
        timeout_sec: The scheduled budget (``--timeout-sec``), the cap.
        factor: Multiple ``K`` of the recorded wall time.
        floor_sec: Smallest derived budget.
        step_factor: Multiple ``M`` of the MESA step count, or ``None``.

    Returns:
        A budget per selected ``(suite, case)``.

    Raises:
        ValueError: If a numeric argument is not positive.
    """
    if factor <= 0.0 or floor_sec <= 0.0 or timeout_sec <= 0.0:
        raise ValueError("budget factor, floor and timeout must be positive")
    if step_factor is not None and step_factor <= 0.0:
        raise ValueError("--mesa-step-factor must be positive")
    base = {(b.suite, b.case): b for b in baseline_records if b.executed}
    budgets: dict[tuple[str, str], CaseBudget] = {}
    for entry in entries:
        rec = base.get(entry.key)
        wall = wall_times.get(entry.key)
        timed_out = (
            rec is not None
            and rec.run_status in baseline_regression.RESOURCE_LIMITED_STATUSES
        )
        if wall is not None and not timed_out:
            limit = min(timeout_sec, max(factor * wall, floor_sec))
            derived = limit < timeout_sec
        else:
            limit = timeout_sec
            derived = False
        steps = rec.mesa_steps if rec is not None else None
        budgets[entry.key] = CaseBudget(
            timeout_sec=limit,
            mesa_steps=steps,
            step_factor=step_factor if steps else None,
            from_baseline=derived,
        )
    return budgets
