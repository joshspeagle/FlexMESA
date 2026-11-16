# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared pytest configuration for FlexMESA test suites."""

from __future__ import annotations

import gc
import os
import re
import sys
from collections.abc import Generator
from typing import Any

import hypothesis
import jax
import pytest

from flexmesa.data import fetch

# Disable per-example timing deadlines so first-call JAX JIT compilation or
# parallel pytest-xdist worker initialization never triggers FlakyFailure.
hypothesis.settings.register_profile(
    "flexmesa_ci",
    deadline=None,
)
hypothesis.settings.load_profile("flexmesa_ci")

_SKIPPED_NODEIDS: list[str] = []
_ISSUE_LINK_RE = re.compile(r"#[1-9][0-9]*\b")
_LINUX_MAPS_PATH: str = "/proc/self/maps"
_JAX_CACHE_VMA_LIMIT: int = 24_000


def pytest_report_header(config: pytest.Config) -> str:
    """Print the pinned upstream MESA reference in every pytest header."""
    del config
    ref = fetch.load_upstream_reference()
    return f"flexmesa upstream reference: {ref.summary_line()}"


def _vma_count_exceeds(limit: int) -> bool:
    """Return whether the Linux process VMA count reaches `limit`."""
    if sys.platform != "linux":
        return False
    try:
        with open(_LINUX_MAPS_PATH, encoding="ascii") as maps_file:
            for idx, _ in enumerate(maps_file, start=1):
                if idx >= limit:
                    return True
    except OSError:
        return False
    return False


def pytest_runtest_teardown(
    item: pytest.Item, nextitem: pytest.Item | None
) -> None:
    """Reclaim XLA CPU executable `mmap` sections at module boundaries."""
    switching_module = nextitem is None or nextitem.path != item.path
    if switching_module or _vma_count_exceeds(_JAX_CACHE_VMA_LIMIT):
        jax.clear_caches()
        gc.collect()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[object]
) -> Generator[None, Any, None]:
    """Record unauthorized skips or unlinked/non-strict xfails during execution.

    Under `CLAUDE.md`, bare skips and imperative `pytest.xfail(...)` calls are
    forbidden in zero-skip CI mode; only `@pytest.mark.xfail(strict=True,
    reason="... #NNN ...")` linked to an open `#<number>` issue is permitted.
    """
    del call
    outcome = yield
    report: pytest.TestReport = outcome.get_result()
    if not report.skipped:
        return
    xfail_mark = item.get_closest_marker("xfail")
    if (
        hasattr(report, "wasxfail")
        and xfail_mark is not None
        and xfail_mark.kwargs.get("strict", True) is True
        and _ISSUE_LINK_RE.search(str(report.wasxfail)) is not None
    ):
        return
    if report.nodeid not in _SKIPPED_NODEIDS:
        _SKIPPED_NODEIDS.append(report.nodeid)


def pytest_sessionfinish(
    session: pytest.Session, exitstatus: int | pytest.ExitCode
) -> None:
    """Fail the session when zero-skip CI enforcement is active."""
    del exitstatus
    if (
        os.environ.get("FLEXMESA_REQUIRE_ZERO_SKIPS", "").strip() == "1"
        and _SKIPPED_NODEIDS
    ):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
) -> None:
    """Report unexpected skips when zero-skip CI enforcement is active."""
    if (
        os.environ.get("FLEXMESA_REQUIRE_ZERO_SKIPS", "").strip() == "1"
        and _SKIPPED_NODEIDS
    ):
        terminalreporter.write_sep(
            "!",
            f"FLEXMESA_REQUIRE_ZERO_SKIPS=1 violated by "
            f"{len(_SKIPPED_NODEIDS)} skipped test(s)",
        )
        for nodeid in _SKIPPED_NODEIDS:
            terminalreporter.write_line(f"  SKIPPED: {nodeid}")
