# SPDX-License-Identifier: GPL-3.0-or-later
"""`set_param` and `set_constraint_value` of the non-search astero cases.

`astero_adipls`, `astero_gyre` and `surface_effects` share a `set_param`
that only sets `ierr = 0` and a `set_constraint_value` with only the
`case default` history-column branch (`#429`). `astero_gyre` is matched
first by its full port, `AsteroGyreExtras` (`#170`, G3c), so only the
other two cases get this handler.
"""

from __future__ import annotations

import pytest

from flexmesa.testing import harness_runner
from tests.case_extras import classify_case_extras, match_case_extras
from tests.case_extras.astero_extras import (
    STAR_GET_HISTORY_OUTPUT,
    EmptySetParamAsteroExtras,
)

# Routines of each case that hold case logic the handler does not run.
_UNPORTED = {
    "astero_adipls": {
        "extras_after_evolve",
        "extras_check_model",
        "get_adipls_frequency_info",
        "my_other_adipls_mode_info",
    },
    "surface_effects": {
        "extras_after_evolve",
        "extras_check_model",
        "extras_startup",
    },
}


@pytest.mark.parametrize("case", sorted(_UNPORTED))
def test_empty_set_param_cases_get_their_own_handler(case: str) -> None:
    """The case's own routines are matched; its other logic stays unported."""
    case_dir = harness_runner.resolve_case_dir("astero", case)
    handler = match_case_extras("astero", case_dir)
    assert isinstance(handler, EmptySetParamAsteroExtras)
    assert set(handler.unported_routines) == _UNPORTED[case]
    assert classify_case_extras("astero", case_dir) == "unported"


def test_empty_set_param_sets_nothing_and_reads_history_columns() -> None:
    """`set_param` assigns nothing; constraints read the history column."""
    handler = EmptySetParamAsteroExtras()
    for name in ("initial_mass", "alpha", "anything"):
        assert handler.set_param({}, name, 1.0) == {}
    for name in ("log_g", "Teff", "anything"):
        assert handler.set_constraint_value({}, name) == STAR_GET_HISTORY_OUTPUT
