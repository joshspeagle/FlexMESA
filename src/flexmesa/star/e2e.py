# SPDX-License-Identifier: LGPL-3.0-or-later
"""End-to-end MESA `inlist` parser, `./rn` stage extractor, and multi-stage
`test_suite` case execution driver (`flexmesa.star.e2e`).
"""

from __future__ import annotations

from flexmesa import _rust


def parse_inlist_bundle(inlist_path: str) -> dict[str, object]:
    """Parse a MESA `inlist` file and recursively resolve `read_extra_*_inlist`.

    Args:
        inlist_path: Path to the entry `inlist` file.

    Returns:
        Dictionary of resolved `&star_job` and `&controls` settings.
    """
    return dict(_rust.star_parse_inlist_bundle(str(inlist_path)))


def parse_rn_stages(
    case_dir: str,
    skip_optional: bool = True,
) -> list[tuple[str, str, str]]:
    """Parse an upstream MESA `./rn` bash script into execution stages.

    Args:
        case_dir: Path to the `test_suite` case directory.
        skip_optional: Whether `MESA_SKIP_OPTIONAL` branches are active.

    Returns:
        List of `(stage_kind, arg1, arg2)` stage tuples.
    """
    return list(_rust.star_parse_rn_stages(str(case_dir), bool(skip_optional)))


def run_test_suite_case(
    case_dir: str,
    skip_optional: bool = True,
    max_steps_per_stage: int | None = None,
    verify_restart: bool = False,
) -> dict[str, object]:
    """Execute a multi-stage upstream MESA `test_suite` case directory.

    Args:
        case_dir: Path to the `test_suite` case directory.
        skip_optional: Whether `MESA_SKIP_OPTIONAL` branches are active.
        max_steps_per_stage: Optional cap on evolutionary steps per stage.
        verify_restart: Whether to verify bit-identical `.photo` restart parity.

    Returns:
        Structured dictionary containing per-stage reports, step counts,
        Newton iterations, and wall-clock runtime.
    """
    return dict(
        _rust.star_run_test_suite_case(
            str(case_dir),
            bool(skip_optional),
            None if max_steps_per_stage is None else int(max_steps_per_stage),
            bool(verify_restart),
        )
    )
