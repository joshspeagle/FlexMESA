# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the step-dump first-divergence comparison (#372)."""

from __future__ import annotations

import struct

from flexmesa.testing import stepdump_compare as sc


def _h(x: float) -> str:
    return f"{struct.unpack('<Q', struct.pack('<d', x))[0]:016X}"


def _dump(lnd_zone2: float = 1.0, slope: float = 0.0, extra: str = "") -> str:
    return "\n".join(
        [
            "# case: star/synthetic",
            "RUN",
            "NAMES 1 3 2 1 2 lnd lnT h1",
            "EQNAMES 1 3 dlnd_dt dlnE_dt equ_h1",
            f"CALL 1 5 {_h(10.0)} {_h(1e-3)} {_h(2e-3)} 2 1",
            f"STATE 1 5 1 {_h(0.5)} {_h(2.0)} {_h(0.7)}",
            f"EQU 1 5 1 {_h(1e-4)} {_h(2e-4)} {_h(0.0)}",
            f"STATE 1 5 2 {_h(lnd_zone2)} {_h(2.5)} {_h(0.7)}",
            f"EQU 1 5 2 {_h(1e-4)} {_h(2e-4)} {_h(0.0)}",
            f"CORR 1 5 1 {_h(1e-2)} {_h(-3e-2)} 1 2",
            f"NEWTON 1 5 1 {_h(1.0)} {_h(1.0)} {_h(slope)}",
            f"FOLD 1 5 1 {_h(0.0)}",
            extra,
            f"RESID 1 5 1 {_h(1.0)} {_h(1e-5)} {_h(2e-5)} 2 1",
            "DONE",
        ]
    )


def test_identical_dumps_report_no_divergence() -> None:
    a = sc.parse_dump(_dump())
    b = sc.parse_dump(_dump())
    assert sc.first_divergence(a, b) is None
    assert "bit-identical" in sc.format_report(a, b, None)


def test_zone_state_difference_names_zone_and_mesa_column() -> None:
    a = sc.parse_dump(_dump())
    b = sc.parse_dump(_dump(lnd_zone2=float.fromhex("0x1.0000000000001p+0")))
    div = sc.first_divergence(a, b)
    assert div is not None
    assert (div.kind, div.record, div.zone, div.field) == (
        "zone",
        "STATE",
        2,
        "lnd",
    )
    assert div.ulps == 1


def test_solver_value_difference_reports_record_field_and_position() -> None:
    a = sc.parse_dump(_dump())
    b = sc.parse_dump(_dump(slope=-1e-9))
    div = sc.first_divergence(a, b)
    assert div is not None
    assert (div.kind, div.record, div.field, div.position) == (
        "value",
        "NEWTON",
        "slope",
        2,
    )


def test_extra_record_on_one_side_is_a_control_flow_divergence() -> None:
    try_line = f"TRY 1 5 1 1 {_h(1.0)} {_h(1.0)} {_h(3.0)} {_h(2.0)}"
    a = sc.parse_dump(_dump())
    b = sc.parse_dump(_dump(extra=try_line))
    div = sc.first_divergence(a, b)
    assert div is not None
    assert (div.kind, div.record, div.position) == ("control", "RESID", 4)


def test_models_align_by_order_when_model_numbers_are_offset() -> None:
    a = sc.parse_dump(_dump())
    b = sc.parse_dump(_dump().replace(" 1 5 ", " 1 4 "))
    assert sc.first_divergence(a, b) is None


def test_ulp_distance_counts_across_zero() -> None:
    pos = struct.unpack("<Q", struct.pack("<d", 5e-324))[0]
    neg = struct.unpack("<Q", struct.pack("<d", -5e-324))[0]
    assert sc.ulp_distance(pos, neg) == 2
