# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests enforcing MESA 26.4.1 default parity across all namelists (#149)."""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys
from typing import Any

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DIFF_SCRIPT = REPO_ROOT / "scripts" / "audit" / "diff_mesa_defaults.py"


def _load_diff_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "diff_mesa_defaults_under_test", DIFF_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {DIFF_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_mesa_defaults_parity_zero_mismatches_and_empty_allow_list() -> None:
    """Verify 0 value mismatches, 0 None active limits, empty ALLOW_LIST."""
    dmd = _load_diff_module()
    assert frozenset() == dmd.ALLOW_LIST
    mesa_root = dmd.fetch.resolve_mesa_test_inputs_dir(None)
    md_text, val_diff, none_diff, unallowed = dmd.check_defaults_parity(
        mesa_root
    )
    assert val_diff == 0, f"unexpected value mismatches:\n{md_text}"
    assert none_diff == 0, f"unexpected None active limits:\n{md_text}"
    assert unallowed == []


def test_mesa_defaults_parity_covers_dev_aliases_unprefixed_and_slots() -> None:
    """Verify dev defaults, irrad_col_depth alias, RspConfig, and xa slots."""
    dmd = _load_diff_module()
    mesa_root = dmd.fetch.resolve_mesa_test_inputs_dir(None)

    star_controls_up = {}
    for rel in dmd.NAMELIST_SPECS[0][1]:
        star_controls_up.update(dmd.parse_mesa_defaults_file(mesa_root, rel))
    star_controls_rs = dmd.parse_rust_defaults(dmd.NAMELIST_SPECS[0][2])

    for key in (
        "use_tdc_enthalpy_flux_limiter",
        "atm_t_tau_errtol",
        "atm_t_tau_max_iters",
        "xa_function_species",
        "xa_function_weight",
        "xa_function_param",
    ):
        assert key in star_controls_up, f"missing upstream control {key}"
        assert key in star_controls_rs, f"missing Rust control {key}"
        ok, reason = dmd._values_equal(
            key, star_controls_up[key].parsed, star_controls_rs[key]
        )
        assert ok, f"{key} mismatch: {reason}"

    star_job_up = {}
    for rel in dmd.NAMELIST_SPECS[1][1]:
        star_job_up.update(dmd.parse_mesa_defaults_file(mesa_root, rel))
    star_job_rs = dmd.parse_rust_defaults(dmd.NAMELIST_SPECS[1][2])
    assert "save_photo_filename" in star_job_up
    assert "save_photo_filename" in star_job_rs
    assert "irrad_col_depth" in star_job_up
    assert "irrad_col_depth" in star_job_rs
    ok, reason = dmd._values_equal(
        "irrad_col_depth",
        star_job_up["irrad_col_depth"].parsed,
        star_job_rs["irrad_col_depth"],
    )
    assert ok, f"irrad_col_depth mismatch: {reason}"


def test_diff_mesa_defaults_cli_check_enforce_succeeds() -> None:
    """Run scripts/audit/diff_mesa_defaults.py --check --enforce."""
    proc = subprocess.run(
        [sys.executable, str(DIFF_SCRIPT), "--check", "--enforce"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout


def test_diff_mesa_defaults_enforce_rejects_mismatch_and_none(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Injecting a value mismatch or active None fails --enforce cleanly."""
    dmd = _load_diff_module()
    mesa_root = dmd.fetch.resolve_mesa_test_inputs_dir(None)
    fake_root = tmp_path / "repo"
    fake_rs = fake_root / "crates" / "fake_kap_types.rs"
    fake_rs.parent.mkdir(parents=True)
    fake_rs.write_text(
        "impl Default for KapConfig {\n"
        "    fn default() -> Self {\n"
        "        Self {\n"
        "            use_zbase_for_type1: false,\n"
        "            zbase: None,\n"
        "        }\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(dmd, "REPO_ROOT", fake_root)
    monkeypatch.setattr(
        dmd,
        "NAMELIST_SPECS",
        (
            (
                "kap &kap",
                ("kap/defaults/kap.defaults",),
                ("crates/fake_kap_types.rs",),
            ),
        ),
    )
    _, val_diff, none_diff, unallowed = dmd.check_defaults_parity(mesa_root)
    assert val_diff >= 1
    assert none_diff >= 1
    assert len(unallowed) >= 2
    out_md = tmp_path / "out.md"
    out_md.write_text(dmd.generate_markdown(mesa_root), encoding="utf-8")
    rc = dmd.main(["--out", str(out_md), "--check", "--enforce"])
    assert rc == 1
