# SPDX-License-Identifier: LGPL-3.0-or-later
"""Declared stops of cleaned upstream cases exposed by strict `#147` checks.

Before `#147`, the engine substituted a default for an option it did not
recognize, a missing network file, or a missing saved model, so these cases
ran to the step cap with a different configuration than the inlist asked for.
Each entry now stops with a specific message. The owning issue is the package
that restores upstream behaviour for that control; the same cases are declared
gaps in ``benchmarks/history_harness/known_gaps.toml``.

A test that finds a case in this table asserts the exact stop. A case that
starts running fails that assertion, so the entry must then be removed.

A stop is either an error whose message matches ``message_pattern``, or, when
``termination_code`` is set, a run that returns with exactly that termination
code instead of reaching the step cap.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping

import pytest


@dataclasses.dataclass(frozen=True)
class DeclaredStop:
    """Expected stop of a cleaned case and the owning issue.

    Attributes:
        message_pattern: Regular expression the raised ``ValueError`` must
            match, or ``None`` when ``termination_code`` is set.
        issue: Open issue of the package that restores upstream behaviour.
        termination_code: Exact termination code the run must return with,
            for a case that stops without raising.
    """

    message_pattern: str | None
    issue: str
    termination_code: str | None = None

    def __post_init__(self) -> None:
        """Checks that exactly one of the two stop forms is given."""
        if (self.message_pattern is None) == (self.termination_code is None):
            raise ValueError(
                "set exactly one of message_pattern and termination_code"
            )


def assert_declared_stop(
    gap: DeclaredStop, run: Callable[[], Mapping[str, object]]
) -> None:
    """Runs a cleaned case and asserts that it stops exactly as declared.

    Args:
        gap: Declared stop of the case.
        run: Runs the cleaned case and returns its report.

    Raises:
        AssertionError: If the case stops differently or not at all.
    """
    if gap.termination_code is not None:
        rep = run()
        assert rep["termination_code"] == gap.termination_code, (
            f"declared stop {gap.termination_code!r} ({gap.issue}) not hit: "
            f"{rep['termination_code']!r}"
        )
        return
    assert gap.message_pattern is not None
    with pytest.raises(ValueError, match=gap.message_pattern):
        run()


DECLARED_STAR_STOPS: dict[str, DeclaredStop] = {
    "extended_convective_penetration": DeclaredStop(
        r"overshoot_scheme", "#231"
    ),
    "make_env": DeclaredStop(r"missing saved model 'start\.mod'", "#154"),
    "tzo": DeclaredStop(
        r"relax ended with termination code 'min_timestep_limit'", "#161"
    ),
}
