"""Regression check of a harness run against the committed baseline run.

While a case is a declared known gap its verdict cannot get worse, so the
known-gaps gate alone cannot see a case that used to evolve and now stops.
This module compares each executed case with the same case in the committed
baseline run (a ``FlexMESA`` regression snapshot, not a MESA reference) and
reports the cases whose measured state got worse (``DECISIONS #27``).

A campaign may run the cases whose baseline row ended on the time budget
with a shorter budget than the rest (the short-budget tier). Strict mode is
bit-identical across machines, so such a run reproduces a prefix of the
baseline trajectory unless the engine changed; the comparison for those
cases is restricted to what a prefix can show.
"""

from __future__ import annotations

import dataclasses
import re

from flexmesa.testing import _case_verdict

__all__ = [
    "EARLY_STOP_STATUS",
    "BaselineRegression",
    "budget_limited_keys",
    "find_baseline_regressions",
    "merge_records_into_baseline",
]

#: Run statuses that say the run was cut off by a resource budget, so the
#: comparison with the baseline is inconclusive and is not counted.
RESOURCE_LIMITED_STATUSES: frozenset[str] = frozenset({"timeout", "oom_killed"})

#: Run status of a case stopped by ``--early-stop-on-divergence``: a tracked
#: column was already out of tolerance, so the case cannot pass
#: (``DECISIONS #68``). Checked as a prefix of the baseline row.
EARLY_STOP_STATUS = "diverged"

_EARLY_STOP_MODEL_RE = re.compile(r"\(model (\d+)\)")


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


def budget_limited_keys(
    baseline_records: list[_case_verdict.CaseHarnessRecord],
) -> frozenset[tuple[str, str]]:
    """Return the cases whose baseline run ended on the time budget.

    Args:
        baseline_records: Records rebuilt from the committed baseline run.

    Returns:
        ``(suite, case)`` keys of executed baseline rows whose run status is
        ``"timeout"``.
    """
    return frozenset(
        (b.suite, b.case)
        for b in baseline_records
        if b.executed and b.run_status == "timeout"
    )


def _short_budget_regression(
    rec: _case_verdict.CaseHarnessRecord,
    base: _case_verdict.CaseHarnessRecord,
) -> tuple[str, str]:
    """Compare a short-budget run with a baseline row cut off by its budget.

    The baseline run was still evolving when its (longer) budget ran out, so
    with an unchanged engine a shorter run reproduces a prefix of it: it
    takes steps, does not stop before the baseline's step count, and first
    leaves a track tolerance at the same model number or not at all. Step
    counts, final values and the span-dependent tolerance verdicts of a
    prefix differ from the baseline by design and are not compared.

    Args:
        rec: Record of the short-budget run.
        base: Baseline record of the same case (``run_status`` ``"timeout"``).

    Returns:
        ``(kind, detail)``, both empty when no regression is found.
    """
    if rec.run_status == "oom_killed":
        return "", ""
    if base.steps > 0 and rec.steps == 0:
        return (
            "stopped taking steps",
            f"baseline {base.steps} steps at its budget, now 0 within the "
            "short budget",
        )
    if (
        rec.run_status != "timeout"
        and not rec.termination_matches
        and rec.steps < base.steps
    ):
        return (
            "ended before the baseline's step count",
            f"baseline still evolving at {base.steps} steps, now "
            f"'{rec.observed_termination or rec.run_status}' at "
            f"{rec.steps} steps",
        )
    b_cmp, r_cmp = base.comparison, rec.comparison
    if b_cmp is None or r_cmp is None or r_cmp.first_diverging_step is None:
        return "", ""
    r_step = r_cmp.first_diverging_step
    if b_cmp.first_diverging_step is not None:
        if r_step < b_cmp.first_diverging_step:
            return (
                "diverges from MESA earlier",
                f"baseline first left a track tolerance at model "
                f"{b_cmp.first_diverging_step}, now at {r_step}",
            )
        return "", ""
    r_x = r_cmp.first_diverging_x
    if (
        b_cmp.overlap_max > b_cmp.overlap_min
        and r_x is not None
        and r_x <= b_cmp.overlap_max
    ):
        return (
            "diverges from MESA earlier",
            f"baseline within track tolerances up to "
            f"{b_cmp.independent_var} = {b_cmp.overlap_max:.6e}, now leaves "
            f"one at model {r_step}",
        )
    return "", ""


def _early_stop_regression(
    rec: _case_verdict.CaseHarnessRecord,
    base: _case_verdict.CaseHarnessRecord,
) -> tuple[str, str]:
    """Compare an early-stopped run with its baseline row.

    The early stop proves that the run cannot pass, so a baseline row that
    completed within the manifest tolerances got worse. Otherwise the run is
    checked as a prefix of the baseline row (``_short_budget_regression``)
    and against the model at which the baseline first left a track
    tolerance.

    Args:
        rec: Record of the run stopped by the divergence check.
        base: Baseline record of the same case.

    Returns:
        ``(kind, detail)``, both empty when no regression is found.
    """
    if (
        base.run_status not in RESOURCE_LIMITED_STATUSES
        and base.executed
        and _within_tolerance(base)
    ):
        return (
            "left tolerance",
            f"baseline within manifest tolerances; {rec.error_message}",
        )
    # The stop was deliberate, so the run is a prefix of the baseline row
    # like a budget cut-off, never "ended before the baseline's step count".
    prefix = dataclasses.replace(rec, run_status="timeout")
    kind, detail = _short_budget_regression(prefix, base)
    if kind:
        return kind, detail
    b_cmp = base.comparison
    match = _EARLY_STOP_MODEL_RE.search(rec.error_message)
    if (
        match is not None
        and b_cmp is not None
        and b_cmp.first_diverging_step is not None
        and int(match.group(1)) < b_cmp.first_diverging_step
    ):
        return (
            "diverges from MESA earlier",
            f"baseline first left a track tolerance at model "
            f"{b_cmp.first_diverging_step}; {rec.error_message}",
        )
    return "", ""


def find_baseline_regressions(
    records: list[_case_verdict.CaseHarnessRecord],
    baseline_records: list[_case_verdict.CaseHarnessRecord],
    *,
    short_budget: frozenset[tuple[str, str]] = frozenset(),
) -> list[BaselineRegression]:
    """List executed cases whose state is worse than in the baseline run.

    A case regresses when the baseline run took steps and this run takes
    none, when the baseline run reached upstream's termination condition and
    this run does not, or when the baseline history was within the manifest
    tolerances and this one is not. A run cut off by its time or memory
    budget is inconclusive on its own; when the baseline run completed, it is
    checked as a prefix of that row (it takes steps and does not leave a
    track tolerance at an earlier model; DECISIONS #59). The check must not
    be used on step-capped runs, whose outcome differs from the baseline by
    design.

    A case in ``short_budget`` whose baseline row ended on the time budget
    was run with a shorter budget and is compared as a prefix of that row
    instead: it regresses when it takes no steps, when it ends (other than
    on upstream's termination) before the baseline's step count, or when it
    first leaves a track tolerance at an earlier model than the baseline.

    Args:
        records: Records of the cases executed in this run.
        baseline_records: Records rebuilt from the committed baseline run.
        short_budget: ``(suite, case)`` keys run with the short budget.

    Returns:
        Regressions in the order of ``records``, at most one per case (the
        earliest kind that applies).
    """
    by_key = {(b.suite, b.case): b for b in baseline_records if b.executed}
    found: list[BaselineRegression] = []
    for rec in records:
        base = by_key.get((rec.suite, rec.case))
        if base is None or not rec.executed:
            continue
        if rec.run_status == EARLY_STOP_STATUS:
            kind, detail = _early_stop_regression(rec, base)
            if kind:
                found.append(
                    BaselineRegression(rec.suite, rec.case, kind, detail)
                )
            continue
        if (rec.suite, rec.case) in short_budget and (
            base.run_status == "timeout"
        ):
            kind, detail = _short_budget_regression(rec, base)
            if kind:
                found.append(
                    BaselineRegression(rec.suite, rec.case, kind, detail)
                )
            continue
        if rec.run_status in RESOURCE_LIMITED_STATUSES:
            # Inconclusive on its own (the budget is machine-dependent). When
            # the baseline run completed, the part that ran must still be a
            # prefix of it (DECISIONS #59).
            kind, detail = "", ""
            if base.run_status not in RESOURCE_LIMITED_STATUSES:
                kind, detail = _short_budget_regression(rec, base)
            if kind:
                found.append(
                    BaselineRegression(rec.suite, rec.case, kind, detail)
                )
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
    *,
    short_budget: frozenset[tuple[str, str]] = frozenset(),
) -> list[_case_verdict.CaseHarnessRecord]:
    """Replace the executed cases of a run inside the full baseline list.

    A case run with the short budget that ended on that budget keeps its
    baseline row, which records a longer run of the same trajectory. A
    short-budget case that ended before the budget (on a termination
    condition or an error) is replaced like any other case.

    Args:
        records: Records of the cases executed in this run.
        baseline_records: Records for every manifest case, rebuilt from the
            committed baseline run.
        short_budget: ``(suite, case)`` keys run with the short budget.

    Returns:
        A list in the order of ``baseline_records`` in which every case
        executed in this run carries this run's record, except short-budget
        cases cut off by that budget.
    """
    fresh = {
        (r.suite, r.case): r
        for r in records
        if r.executed
        and not (
            (r.suite, r.case) in short_budget and r.run_status == "timeout"
        )
    }
    return [fresh.get((b.suite, b.case), b) for b in baseline_records]
