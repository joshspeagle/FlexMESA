# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests of ``flexmesa trace --stage/--occurrence`` option handling (#608)."""

from __future__ import annotations

import pathlib

import pytest

from flexmesa.testing import trace_cli


def _dump(tmp_path: pathlib.Path, header: str) -> pathlib.Path:
    path = tmp_path / "mesa.stepdump"
    path.write_text(header + "RUN\n", encoding="utf-8")
    return path


def test_selectors_default_to_the_dump_header(tmp_path: pathlib.Path) -> None:
    dump = _dump(tmp_path, "# stage: 2\n# occurrence: 3\n")
    assert trace_cli._stepdump_selectors(dump, None, None) == (2, 3)


def test_explicit_selectors_override_the_header(
    tmp_path: pathlib.Path,
) -> None:
    dump = _dump(tmp_path, "# stage: 2\n# occurrence: 3\n")
    assert trace_cli._stepdump_selectors(dump, 4, 1) == (4, 1)


def test_header_without_selectors_means_every_stage_first_occurrence(
    tmp_path: pathlib.Path,
) -> None:
    dump = _dump(tmp_path, "# stage: all\n")
    assert trace_cli._stepdump_selectors(dump, None, None) == (None, 1)


def test_stage_zero_is_rejected_by_the_parser() -> None:
    parser = trace_cli._build_trace_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["star", "ns_h", "--out", "o", "--stage", "0"])


def test_parser_accepts_stage_and_occurrence() -> None:
    ns = trace_cli._build_trace_parser().parse_args(
        ["star", "ns_h", "--out", "o", "--stage", "2", "--occurrence", "2"]
    )
    assert (ns.stage, ns.occurrence) == (2, 2)


def test_stage_without_compare_is_an_error() -> None:
    with pytest.raises(SystemExit):
        trace_cli.trace_main(["star", "ns_h", "--out", "o", "--stage", "2"])
