# SPDX-License-Identifier: GPL-3.0-or-later
"""`set_param` and `set_constraint_value` of the non-search astero cases.

`astero_adipls`, `astero_gyre` and `surface_effects` share a `set_param`
that only sets `ierr = 0` and a `set_constraint_value` with only the
`case default` history-column branch (`#429`). All three cases now match their
full star-hook handlers (`AsteroAdiplsExtras` in `#171`, `AsteroGyreExtras` in
`#170`, `SurfaceEffectsExtras` in `#569`), while `EmptySetParamAsteroExtras`
remains the fallback for any synthetic/unported case sharing those empty hooks.
"""

from __future__ import annotations

from flexmesa.testing import harness_runner
from tests.case_extras import (
    classify_case_extras,
    match_case_extras,
    normalize_fortran,
    source_sha256,
)
from tests.case_extras.astero_extras import (
    STAR_GET_HISTORY_OUTPUT,
    EmptySetParamAsteroExtras,
    SurfaceEffectsExtras,
    match_astero_extras,
)


def test_surface_effects_resolves_to_full_handler() -> None:
    """`surface_effects` resolves to `SurfaceEffectsExtras` and `ported`."""
    case_dir = harness_runner.resolve_case_dir("astero", "surface_effects")
    handler = match_case_extras("astero", case_dir)
    assert isinstance(handler, SurfaceEffectsExtras)
    assert classify_case_extras("astero", case_dir) == "ported"


def test_empty_set_param_fallback_on_synthetic_unported_case() -> None:
    """Empty `set_param`/`set_constraint_value` with an unported routine."""
    norm_star = normalize_fortran(
        """
        subroutine set_constraint_value(id, name, val, ierr)
           integer, intent(in) :: id
           character(len=strlen), intent(in) :: name
           real(dp), intent(out) :: val
           integer, intent(out) :: ierr
           type (star_info), pointer :: s
           ierr = 0
           call star_ptr(id, s, ierr)
           if (ierr /= 0) return
           select case (name)
              case default
                 val = star_get_history_output(s, name, ierr)
                 if (ierr /= 0) call mesa_error(__FILE__, __LINE__)
           end select
        end subroutine set_constraint_value

        subroutine set_param(id, name, val, ierr)
           integer, intent(in) :: id
           character(len=strlen), intent(in) :: name
           real(dp), intent(in) :: val
           integer, intent(out) :: ierr
           type (star_info), pointer :: s
           ierr = 0
        end subroutine set_param

        subroutine extras_startup(id, restart, ierr)
           ierr = 0
           call custom_unported_hook()
        end subroutine extras_startup
        """
    )
    handler = match_astero_extras(norm_star, source_sha256)
    assert isinstance(handler, EmptySetParamAsteroExtras)
    assert handler.unported_routines == ("extras_startup",)


def test_empty_set_param_sets_nothing_and_reads_history_columns() -> None:
    """`set_param` assigns nothing; constraints read the history column."""
    handler = EmptySetParamAsteroExtras()
    for name in ("initial_mass", "alpha", "anything"):
        assert handler.set_param({}, name, 1.0) == {}
    for name in ("log_g", "Teff", "anything"):
        assert handler.set_constraint_value({}, name) == STAR_GET_HISTORY_OUTPUT
