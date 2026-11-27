# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the first-divergence table renderer and window helpers (#608)."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import ModuleType

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "benchmarks"
    / "fortran_mesa"
    / "stepdumps"
    / "first_divergence_table.py"
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fdt", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fdt"] = mod
    spec.loader.exec_module(mod)
    return mod


fdt = _load()


def _row(case: str, first: str, model: int | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "case": case,
        "window": "start of run",
        "run": "ok",
        "first_divergence": first,
        "detail": "x, 3 ULP" if model else "",
    }
    if model is not None:
        row["divergence"] = {"mesa_model": model}
    return row


def test_header_parse_stops_at_first_record_line() -> None:
    text = "# stage: 2\n# first_model: 90\nRUN\n# late: ignored\n"
    assert fdt.parse_header(text) == {"stage": "2", "first_model": "90"}


def test_window_with_later_stage_or_occurrence_is_skipped() -> None:
    assert "stage 2" in fdt.window_skip_reason({"stage": "2"})
    assert "occurrence 2" in fdt.window_skip_reason({"occurrence": "2"})
    assert fdt.window_skip_reason({"stage": "1", "occurrence": "1"}) == ""
    assert fdt.window_skip_reason({}) == ""


def test_max_steps_is_none_at_run_start_and_past_window_otherwise() -> None:
    assert fdt.window_max_steps({"first_model": "first of the run"}) is None
    got = fdt.window_max_steps({"first_model": "70", "steps_per_star": "12"})
    assert got is not None and got > 70 + 12


def test_ranking_orders_by_mesa_model_and_output_is_deterministic() -> None:
    rows = [
        _row("star/b", "model #9 value DM", 9),
        _row("star/a", "none (bit-identical)"),
        _row("star/c", "model #2 value DM", 2),
    ]
    kw = dict(
        commit="abc123",
        provenance="MESA r26.4.1",
        command="first_divergence_table.py --dumps d",
        verdicts={"star/c": "pass", "star/b": "fail (#1)"},
    )
    page = fdt.render_markdown(rows, **kw)
    assert page == fdt.render_markdown(list(reversed(rows)), **kw)
    ranked = page.split("## Rows that differ")[1].split("## All rows")[0]
    assert ranked.index("star/c") < ranked.index("star/b")
    assert "star/a" not in ranked
    assert "| pass |" in ranked and "fail (#1)" in ranked
    assert "bit-identical over the window 1" in page
    assert "/home/" not in page and "/tmp/" not in page
