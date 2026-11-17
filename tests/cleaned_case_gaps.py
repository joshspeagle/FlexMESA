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
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class DeclaredStop:
    """Expected error text for a cleaned case and the owning issue."""

    message_pattern: str
    issue: str


DECLARED_STAR_STOPS: dict[str, DeclaredStop] = {
    "extended_convective_penetration": DeclaredStop(
        r"overshoot_scheme", "#155"
    ),
    "make_env": DeclaredStop(r"missing saved model 'start\.mod'", "#154"),
    "custom_rates": DeclaredStop(r"unknown net file 'nco\.net'", "#161"),
    "carbon_kh": DeclaredStop(r"unknown net file 'minimal\.net'", "#152"),
    "irradiated_planet": DeclaredStop(
        r"atm_option' option 'irradiated_grey'", "#154"
    ),
    "wd_aic": DeclaredStop(r"unknown net file 'wd_aic\.net'", "#152"),
    "tzo": DeclaredStop(r"dutch_wind_lowt_scheme' option 'other'", "#161"),
}
