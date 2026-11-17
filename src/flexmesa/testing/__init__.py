# SPDX-License-Identifier: LGPL-3.0-or-later
"""Golden baseline parsing and numerical comparison utilities for FlexMESA.

This subpackage provides parsers and numerical assertion helpers for MESA's
standalone `<pkg>/test/test_output` reference files (`1pe24.16` scientific
format with standard `E` or Fortran `D` exponents).
"""

from __future__ import annotations

from flexmesa.testing import (
    case_manifest,
    case_report,
    dashboard,
    golden,
    harness_runner,
    history_compare,
    known_gaps_gate,
    module_hex_gate,
    port_manifest,
    trace_cli,
)

__all__: list[str] = [
    "case_manifest",
    "case_report",
    "dashboard",
    "golden",
    "harness_runner",
    "history_compare",
    "known_gaps_gate",
    "module_hex_gate",
    "port_manifest",
    "trace_cli",
]
