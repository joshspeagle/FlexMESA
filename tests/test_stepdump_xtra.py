# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the opt-in XTRA diagnostics patches of the step dump (#608).

These check the generated Fortran text on synthetic sources built from the
patch anchors, so no MESA tree is needed.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys
from types import ModuleType

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
STEPDUMPS = REPO / "benchmarks" / "fortran_mesa" / "stepdumps"


def _load(name: str, path: pathlib.Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, str(STEPDUMPS))
sdp = _load("stepdump_patch", STEPDUMPS / "stepdump_patch.py")
xp = _load("xtra_patch", STEPDUMPS / "xtra_patch.py")
rs = _load("run_stepdumps", STEPDUMPS / "run_stepdumps.py")


def _star_source() -> str:
    """Synthetic ``star_solver.f90`` holding every anchor once."""
    parts = [sdp.EDITS[0][0], sdp.EDITS[1][0]]
    parts.append("            s% solver_iter = iter\n")
    for anchor, _, _ in sdp.EDITS[2:]:
        parts.append(anchor)
    parts.append(xp.STAR_EDITS[1][0])
    parts.append(sdp.END_ANCHOR)
    return "".join(parts)


def _binary_source() -> str:
    """Synthetic ``binary_mdot.f90`` holding every anchor once."""
    seen: list[str] = []
    for anchor, _, _ in xp.BINARY_EDITS:
        if anchor not in seen:
            seen.append(anchor)
    return "".join(seen) + "      end module binary_mdot\n"


def test_parse_zones_expands_inclusive_ranges() -> None:
    assert xp.parse_zones("1, 262,474:476") == [1, 262, 474, 475, 476]
    assert xp.parse_zones("") == []
    for bad in ("0", "5:3", "x"):
        with pytest.raises(ValueError):
            xp.parse_zones(bad)


def test_star_patch_inserts_one_call_per_site_in_solver_order() -> None:
    out = xp.patch_star(_star_source(), [36, 262])
    for call in (
        "call xtra_star(s, nvar, nz, 'RESID0', 0, soln)",
        "call xtra_star(s, nvar, nz, 'CORR', iter, soln)",
        "call xtra_star(s, nvar, nz, 'RESID', iter, soln)",
    ):
        assert out.count(call) == 1
    order = [out.index(f"'{t}'") for t in ("RESID0", "CORR")]
    assert order == sorted(order)


def test_star_patch_zone_lists_default_and_override() -> None:
    both = xp.patch_star(_star_source(), [36, 262])
    assert "zr(2) = [integer :: 36, 262]" in both
    assert "zc(2) = [integer :: 36, 262]" in both
    split = xp.patch_star(_star_source(), [36, 262], [1])
    assert "zc(1) = [integer :: 1]" in split
    none = xp.patch_star(_star_source())
    assert "zr(0) = [integer :: ]" in none
    assert "'RESID_lnd'" in none and "'CORR_SOLNSUM'" in none


def test_star_patch_composes_with_stepdump_patch_and_default_is_unchanged() -> (
    None
):
    src = _star_source()
    plain = sdp.patch_text(src)
    assert "XTRA" not in plain and "xtra_star" not in plain
    both = sdp.patch_text(xp.patch_star(src, [1]))
    # The default records are still present and in the same places.
    for tag in ("'CALL'", "'CORR'", "'RESID'", "stepdump_unit"):
        assert tag in both
    assert both.count("integer function stepdump_unit") == 1


@pytest.mark.parametrize("kind", ["star", "binary-rlo"])
def test_patches_add_output_only(kind: str) -> None:
    if kind == "star":
        src, out = _star_source(), xp.patch_star(_star_source(), [1])
    else:
        src, out = _binary_source(), xp.patch_binary_rlo(_binary_source())
    added = [x for x in out.splitlines() if x not in src.splitlines()]
    assert added
    # No statement assigns to MESA state; only local scratch values change.
    assign = re.compile(r"^\s*(s|b)% \w+(\([^)]*\))?\s*=[^=]")
    assert not [x for x in added if assign.match(x)]
    assert not [
        x
        for x in added
        if re.match(r"^\s*call (?!xtra_|get_environment_variable)", x)
    ]


def test_patches_refuse_a_missing_or_repeated_anchor() -> None:
    with pytest.raises(ValueError, match="found 0 times"):
        xp.patch_star("nothing here\n")
    with pytest.raises(ValueError, match="found 0 times"):
        xp.patch_binary_rlo("nothing here\n")
    once = xp.patch_star(_star_source())
    with pytest.raises(ValueError, match="already carries"):
        xp.patch_star(once)
    once_b = xp.patch_binary_rlo(_binary_source())
    with pytest.raises(ValueError, match="already carries"):
        xp.patch_binary_rlo(once_b)


def test_binary_bracket_records_follow_the_num_tries_reset() -> None:
    out = xp.patch_binary_rlo(_binary_source())
    reset = out.index("b% fixed_delta_mdot = b% mtransfer_rate")
    assert out.index("'RLO_BRACKET', b% model_number") > reset
    assert out.index("'RLO_CALL', b% model_number") < reset
    for tag in (
        "RLO_CALL",
        "RLO_FLAGS",
        "RLO_DO_CUBIC",
        "RLO_NEXT",
        "RLO_REDO",
        "RLO_RESULT_in_tol",
        "RLO_RESULT_redo",
    ):
        assert tag in out, tag


def test_binary_quad_records_bracket_the_find0_fallback() -> None:
    out = xp.patch_binary_rlo(_binary_source())
    fallback = out.index("mdot_next = find0(b% mdot_lo")
    assert out.index("'RLO_QUAD', b%") < fallback
    assert out.index("'RLO_QUAD_FINAL', b%") > fallback
    assert out.count("integer function xtra_bin_unit") == 1
    assert "FLEXMESA_XTRA_STOPMODEL" in out


def test_run_stepdumps_requires_binary_model_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = [
        "run_stepdumps.py",
        "--mesa-dir",
        "m",
        "--mesasdk-root",
        "s",
        "--scratch",
        "x",
        "--out",
        "o",
        "--xtra",
        "binary-rlo",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="--xtra-binary-models"):
        rs.main()
