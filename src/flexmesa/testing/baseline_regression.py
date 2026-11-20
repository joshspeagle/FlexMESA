"""Regression check of a harness run against the committed baseline run.

While a case is a declared known gap its verdict cannot get worse, so the
known-gaps gate alone cannot see a case that used to evolve and now stops.
This module compares each executed case with the same case in the committed
baseline run (a ``FlexMESA`` regression snapshot, not a MESA reference) and
reports the cases whose measured state got worse (``DECISIONS #27``).
"""

from __future__ import annotations

import dataclasses

from flexmesa.testing import _case_verdict

__all__ = [
    "BaselineRegression",
    "find_baseline_regressions",
    "merge_records_into_baseline",
]

#: Run statuses that say the run was cut off by a resource budget, so the
#: comparison with the baseline is inconclusive and is not counted.
RESOURCE_LIMITED_STATUSES: frozenset[str] = frozenset({"timeout", "oom_killed"})


@dataclasses.dataclass(frozen=True)
class BaselineRegression:
    """One case whose measured state is worse than in the baseline run.

    Attributes:
        suite: Upstream suite (``star``, ``binary`` or ``astero``).
        case: Upstream case name.
        kind: One of ``"stopped taking steps"``,
            ``"lost upstream termination"`` or ``"left tolerance"``.
        detail: Baseline and current values, for the report.
    """

    suite: str
    case: str
    kind: str
    detail: str

    @property
    def label(self) -> str:
        """Return ``"<suite>/<case>: <kind> (<detail>)"``."""
        return f"{self.suite}/{self.case}: {self.kind} ({self.detail})"


def _within_tolerance(rec: _case_verdict.CaseHarnessRecord) -> bool:
    return rec.is_within_tolerances


def find_baseline_regressions(
    records: list[_case_verdict.CaseHarnessRecord],
    baseline_records: list[_case_verdict.CaseHarnessRecord],
) -> list[BaselineRegression]:
    """List executed cases whose state is worse than in the baseline run.

    A case regresses when the baseline run took steps and this run takes
    none, when the baseline run reached upstream's termination condition and
    this run does not, or when the baseline history was within the manifest
    tolerances and this one is not. A run cut off by its time or memory
    budget is inconclusive and is not reported. The check must not be used
    on step-capped runs, whose outcome differs from the baseline by design.

    Args:
        records: Records of the cases executed in this run.
        baseline_records: Records rebuilt from the committed baseline run.

    Returns:
        Regressions in the order of ``records``, at most one per case (the
        earliest of the three kinds that applies).
    """
    by_key = {(b.suite, b.case): b for b in baseline_records if b.executed}
    found: list[BaselineRegression] = []
    for rec in records:
        base = by_key.get((rec.suite, rec.case))
        if base is None or not rec.executed:
            continue
        if rec.run_status in RESOURCE_LIMITED_STATUSES:
            continue
        kind = ""
        detail = ""
        if base.steps > 0 and rec.steps == 0:
            kind = "stopped taking steps"
            detail = f"baseline {base.steps} steps, now 0"
        elif base.termination_matches and not rec.termination_matches:
            kind = "lost upstream termination"
            detail = (
                f"baseline '{base.observed_termination}', "
                f"now '{rec.observed_termination or rec.run_status}'"
            )
        elif _within_tolerance(base) and not _within_tolerance(rec):
            kind = "left tolerance"
            detail = "baseline within manifest tolerances, now outside"
        if kind:
            found.append(BaselineRegression(rec.suite, rec.case, kind, detail))
    return found


def merge_records_into_baseline(
    records: list[_case_verdict.CaseHarnessRecord],
    baseline_records: list[_case_verdict.CaseHarnessRecord],
) -> list[_case_verdict.CaseHarnessRecord]:
    """Replace the executed cases of a run inside the full baseline list.

    Args:
        records: Records of the cases executed in this run.
        baseline_records: Records for every manifest case, rebuilt from the
            committed baseline run.

    Returns:
        A list in the order of ``baseline_records`` in which every case
        executed in this run carries this run's record.
    """
    fresh = {(r.suite, r.case): r for r in records if r.executed}
    return [fresh.get((b.suite, b.case), b) for b in baseline_records]
