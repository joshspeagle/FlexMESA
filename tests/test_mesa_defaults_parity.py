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
    """Verify 0 mismatches, empty ALLOW_LIST, and full denominator counts."""
    dmd = _load_diff_module()
    assert frozenset() == dmd.ALLOW_LIST
    mesa_root = dmd.fetch.resolve_mesa_test_inputs_dir(None)
    md_text, val_diff, none_diff, unallowed, summary = (
        dmd.evaluate_defaults_parity(mesa_root)
    )
    assert val_diff == 0, f"unexpected value mismatches:\n{md_text}"
    assert none_diff == 0, f"unexpected None active limits:\n{md_text}"
    assert unallowed == []
    assert summary.upstream_scalars == 2396
    assert summary.matched >= 969
    assert summary.agree == summary.matched
    assert summary.skipped_unparseable_rust <= 12
    assert summary.skipped_no_rust_field >= 0
    assert (
        summary.matched
        + summary.skipped_unparseable_rust
        + summary.skipped_no_rust_field
        == summary.upstream_scalars
    )
    assert summary.upstream_array_only == 11
    assert (
        f"| **Total** | **{summary.upstream_scalars}** | "
        f"**{summary.matched}** |" in md_text
    )
    assert "## Skipped Controls with Rust Fields" in md_text


def test_mesa_defaults_parity_covers_dev_aliases_unprefixed_and_slots() -> None:
    """Verify dev defaults, aliases, RspConfig, xa slots, and repeat arrays."""
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
        "dx_limit",
        "dx_decreases_only",
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

    astero_up = dmd.parse_mesa_defaults_file(
        mesa_root, dmd.NAMELIST_SPECS[3][1][0]
    )
    astero_rs = dmd.parse_rust_defaults(dmd.NAMELIST_SPECS[3][2])
    for key in ("chi2_seismo_fraction", "iscan_factor"):
        assert key in astero_up, f"missing upstream astero control {key}"
        assert key in astero_rs, f"missing Rust astero control {key}"
        ok, reason = dmd._values_equal(
            key, astero_up[key].parsed, astero_rs[key]
        )
        assert ok, f"{key} mismatch: {reason}"


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
        "            kap_file_prefix: compute_prefix(),\n"
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
    md_text, val_diff, none_diff, unallowed, summary = (
        dmd.evaluate_defaults_parity(mesa_root)
    )
    assert val_diff >= 1
    assert none_diff >= 1
    assert len(unallowed) >= 2
    assert summary.matched == 2
    assert summary.skipped_unparseable_rust == 1
    assert summary.skipped_no_rust_field == summary.upstream_scalars - 3
    assert "computed/non-scalar Rust initializer" in md_text
    out_md = tmp_path / "out.md"
    out_md.write_text(dmd.generate_markdown(mesa_root), encoding="utf-8")
    rc = dmd.main(["--out", str(out_md), "--check", "--enforce"])
    assert rc == 1


def test_kap_get_results_jax_rejects_negative_zbase_for_type2_and_jit() -> None:
    """Derived JAX kap kernel rejects zbase < 0 when Type 2 is active (#149)."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from flexmesa.kap import jax_kernels

    with pytest.raises(ValueError, match="zbase >= 0"):
        jax_kernels.kap_get_results_jax(
            0.02, 0.70, -1.0, log_rho=-3.0, log_t=6.0, use_type2_opacities=True
        )
    with pytest.raises(ValueError, match="zbase >= 0"):
        jax_kernels.kap_get_results_jax(
            0.02,
            0.70,
            -1.0,
            log_rho=-6.0,
            log_t=3.6,
            kap_low_t_option="AESOPUS",
        )

    # Type 1 only (use_type2_opacities=False, non-kapCN/AESOPUS low-T) succeeds.
    kap_t1, dr_t1, dt_t1, _ = jax_kernels.kap_get_results_jax(
        0.02, 0.70, -1.0, log_rho=-3.0, log_t=6.0, use_type2_opacities=False
    )
    assert np.isfinite(float(kap_t1))
    assert np.isfinite(float(dr_t1))
    assert np.isfinite(float(dt_t1))

    # Under @jax.jit Tracer, negative zbase returns NaN (no silent Z=0 clamp).
    jit_fn = jax.jit(
        lambda zb: jax_kernels.kap_get_results_jax(
            0.02, 0.70, zb, log_rho=-3.0, log_t=6.0, use_type2_opacities=True
        )
    )
    kap_bad, dr_bad, dt_bad, fr_bad = jit_fn(jnp.asarray(-1.0))
    assert np.isnan(float(kap_bad))
    assert np.isnan(float(dr_bad))
    assert np.isnan(float(dt_bad))
    assert np.all(np.isnan(np.asarray(fr_bad)))
    kap_ok, _, _, _ = jit_fn(jnp.asarray(0.02))
    assert np.isfinite(float(kap_ok)) and float(kap_ok) > 0.0

    # Standalone Python kap.kap_get also defaults zbase=-1.0 and rejects Type 2
    # when zbase is omitted (MESA kap/defaults/kap.defaults Zbase = -1).
    from flexmesa import kap as fm_kap

    with pytest.raises(ValueError, match="zbase"):
        fm_kap.kap_get(
            np.array([2, 7, 40], dtype=np.uintp),
            np.array([0.70, 0.28, 0.02], dtype=np.float64),
            log_rho=-3.0,
            log_t=6.0,
        )

    # StarModel.from_pre_ms defaults match MESA 26.4.1 &star_job
    # (pre_ms_T_c = 3e5, pre_ms_d_log10_P = 0.0).
    import inspect

    from flexmesa import star as fm_star

    sig = inspect.signature(fm_star.StarModel.from_pre_ms)
    assert sig.parameters["t_c"].default == pytest.approx(3.0e5)
    assert sig.parameters["d_log10_p"].default == pytest.approx(0.0)
    pre_def = fm_star.StarModel.from_pre_ms(eval_hydro=False)
    assert pre_def.nz > 50


def test_from_mod_str_leaves_zbase_at_upstream_negative_default() -> None:
    """PyStarModel.from_mod_str leaves kap zbase at upstream -1.0 (#149)."""
    from flexmesa import _rust
    from flexmesa import star as fm_star

    zams = fm_star.StarModel.from_zams(
        initial_mass=1.0, initial_z=0.02, eval_hydro=False
    )
    mod_text = zams.to_mod_str()

    rm = _rust.RustStarModel.from_mod_str(mod_text)
    assert rm.zbase == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="zbase"):
        rm.eval_hydro_vars()

    wrapped = fm_star.StarModel.from_mod_str(mod_text, eval_hydro=False)
    assert wrapped.nz == zams.nz
    with pytest.raises(ValueError, match="zbase"):
        fm_star.StarModel.from_mod_str(mod_text, eval_hydro=True)
