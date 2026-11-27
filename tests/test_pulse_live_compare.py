# SPDX-License-Identifier: GPL-3.0-or-later
"""Synthetic-input tests for the live pulsation comparison logic (#612)."""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
from types import ModuleType
from typing import Any

import pytest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "benchmarks/fortran_mesa/gyre_pulse_data/compare_pulse_live_run.py"
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pulse_live", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pulse_live"] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load()


def _mode(
    n: int, f: float, i: float | None = 1.0, g: float | None = 0.5
) -> Any:
    return M.ModeRec(0, 0, n, f, i, g)


def test_identical_calls_are_bit_identical_with_zero_difference() -> None:
    call = [(10, [_mode(1, 2.0), _mode(2, 3.0)])]
    cmp = M.compare_calls(call, call)
    assert cmp.totals["freq"].compared == 2
    assert cmp.totals["freq"].bit_identical == 2
    assert cmp.totals["inertia"].max_rel == 0.0
    assert cmp.calls[0].order_same


def test_one_ulp_change_is_counted_and_gives_relative_difference() -> None:
    f = 2.0
    f_up = math.nextafter(f, 3.0)
    cmp = M.compare_calls([(5, [_mode(1, f_up)])], [(5, [_mode(1, f)])])
    assert cmp.totals["freq"].bit_identical == 0
    assert cmp.totals["growth"].bit_identical == 1
    assert cmp.totals["freq"].max_rel == pytest.approx(2.2e-16, rel=0.01)


def test_calls_and_modes_on_one_side_only_are_reported() -> None:
    mine = [(1, [_mode(1, 1.0), _mode(3, 3.0)]), (7, [_mode(1, 1.0)])]
    ref = [(1, [_mode(1, 1.0), _mode(2, 2.0)]), (9, [_mode(1, 1.0)])]
    cmp = M.compare_calls(mine, ref)
    assert cmp.calls_only_flexmesa == [(7, 0)]
    assert cmp.calls_only_mesa == [(9, 0)]
    assert cmp.modes_only_flexmesa == 1
    assert cmp.modes_only_mesa == 1
    assert cmp.calls[0].n_common == 1
    assert cmp.calls[0].only_flexmesa == [3]
    assert cmp.calls[0].only_mesa == [2]
    _, rows = M.format_rows(cmp)
    assert rows[0].split("\t")[-2:] == ["3", "2"]


def test_repeated_model_number_matches_by_occurrence() -> None:
    mine = [(4, [_mode(1, 1.0)]), (4, [_mode(1, 2.0)])]
    ref = [(4, [_mode(1, 1.0)]), (4, [_mode(1, 2.0)])]
    cmp = M.compare_calls(mine, ref)
    assert [c.occurrence for c in cmp.calls] == [0, 1]
    assert cmp.totals["freq"].bit_identical == 2


def test_unavailable_quantity_is_not_compared() -> None:
    mine = [(1, [_mode(1, 1.0, i=None)])]
    ref = [(1, [_mode(1, 1.0)])]
    cmp = M.compare_calls(mine, ref)
    assert not cmp.totals["inertia"].available
    assert cmp.totals["freq"].available
    assert any("not exposed" in ln for ln in M.format_summary(cmp))


def test_reordered_modes_flag_order_difference() -> None:
    mine = [(1, [_mode(2, 2.0), _mode(1, 1.0)])]
    ref = [(1, [_mode(1, 1.0), _mode(2, 2.0)])]
    assert not M.compare_calls(mine, ref).calls[0].order_same


def test_reference_loader_reads_hex_columns(tmp_path: pathlib.Path) -> None:
    cols = (
        "eval_id model_number l m n_pg freq_Hz_hex inertia_hex "
        "growth_rate_Hz_hex"
    ).split()
    row = ["1", "20", "0", "0", "3", (2.5).hex(), (0.25).hex(), (-1.5).hex()]
    p = tmp_path / "modes.tsv"
    p.write_text("\t".join(cols) + "\n" + "\t".join(row) + "\n")
    calls = M.load_reference(p, "freq_Hz_hex", "growth_rate_Hz_hex")
    assert calls == [(20, [M.ModeRec(0, 0, 3, 2.5, 0.25, -1.5)])]
