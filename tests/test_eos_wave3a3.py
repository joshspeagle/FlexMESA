# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for Wave 3a.3 (`PC`, `Skye`, `eosDT_get`, and inverse solvers)."""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import flexmesa
from flexmesa import chem, constants, eos
from flexmesa import math as fm_math

_EOS_ORACLES_TSV = (
    pathlib.Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "fortran_mesa"
    / "eos_oracles"
    / "mesa_26_4_1_eos_oracles.tsv"
)


def _assert_unfloored(
    got: np.ndarray,
    ref: np.ndarray,
    *,
    rtol: float,
    zero_atol: float = 1.0e-14,
    near_zero_cutoff: float = 1.0e-13,
) -> None:
    """Assert unfloored `|got - ref| / |ref| <= rtol` (`zero_atol` near 0)."""
    got_a = np.asarray(got, dtype=np.float64)
    ref_a = np.asarray(ref, dtype=np.float64)
    assert got_a.shape == ref_a.shape
    near_zero = np.abs(ref_a) <= near_zero_cutoff
    if np.any(near_zero):
        max_abs = float(np.max(np.abs(got_a[near_zero] - ref_a[near_zero])))
        assert max_abs <= zero_atol, (
            f"Near-zero abs diff {max_abs:.3e} > {zero_atol:.1e}"
        )
    non_zero = ~near_zero
    if np.any(non_zero):
        rel = np.abs(got_a[non_zero] - ref_a[non_zero]) / np.abs(
            ref_a[non_zero]
        )
        max_rel = float(np.max(rel))
        assert max_rel <= rtol, f"Unfloored rel diff {max_rel:.3e} > {rtol:.1e}"


def _load_eos_case(case_id: str) -> np.ndarray:
    """Load a `26 x 3` `EOS` block from `mesa_26_4_1_eos_oracles.tsv`."""
    arr = np.zeros((eos.NUM_EOS_BASIC_RESULTS, 3), dtype=np.float64)
    seen = [False] * eos.NUM_EOS_BASIC_RESULTS
    for line in _EOS_ORACLES_TSV.read_text("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if cols[0] == "EOS" and cols[1] == case_id:
            slot = int(cols[2])
            arr[slot] = [float(cols[3]), float(cols[4]), float(cols[5])]
            seen[slot] = True
    assert all(seen), f"Incomplete EOS oracle case: {case_id}"
    return arr


def _load_scalar_row(prefix: str, case_id: str) -> tuple[float, ...]:
    """Load an `ALFA` or `INV` scalar row from `mesa_26_4_1_eos_oracles.tsv`."""
    for line in _EOS_ORACLES_TSV.read_text("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if cols[0] == prefix and cols[1] == case_id:
            return tuple(float(x) for x in cols[2:])
    raise AssertionError(f"Missing {prefix} row: {case_id}")


def _oracle_7sp_composition(
    y_in: float, xc_in: float, xo_in: float
) -> tuple[list[int], np.ndarray, float, float]:
    """Build 7-species composition matching `gen_eos_oracles.f90` `make_xa`."""
    names = ("h1", "he4", "c12", "n14", "o16", "ne20", "mg24")
    cids = [int(chem.get_nuclide_index(n) or 0) for n in names]
    assert all(c > 0 for c in cids)
    xa = np.array([0.0, y_in, xc_in, 0.0, xo_in, 0.0, 0.0], dtype=np.float64)
    infos = [chem.get_nuclide_info(c) for c in cids]
    a_arr = np.array([float(inf.a) for inf in infos], dtype=np.float64)
    z_arr = np.array([float(inf.z) for inf in infos], dtype=np.float64)
    abar = float(1.0 / np.sum(xa / a_arr))
    zbar = float(abar * np.sum(xa * z_arr / a_arr))
    return cids, xa, abar, zbar


def _c12_o16_composition() -> tuple[list[int], np.ndarray, float, float]:
    """Return 50/50 `c12` + `o16` `(chem_id, xa, abar, zbar)`."""
    ic12 = chem.get_nuclide_index("c12")
    io16 = chem.get_nuclide_index("o16")
    assert ic12 is not None and io16 is not None
    chem_id = [ic12, io16]
    xa = np.array([0.5, 0.5], dtype=np.float64)
    comp = chem.get_composition_info(chem_id, xa)
    return chem_id, xa, float(comp.abar), float(comp.zbar)


def test_get_pc_and_skye_alfa_match_mesa_26_4_1_oracle() -> None:
    """Verify `get_pc_alfa` and `get_skye_alfa*` against MESA `26.4.1`."""
    _, _, abar_c, zbar_c = _oracle_7sp_composition(0.0, 1.0, 0.0)
    _, _, abar_co, zbar_co = _c12_o16_composition()
    with flexmesa.numerics_mode_context("strict"):
        for cid, lr, lt, ab, zb, use_gam in (
            ("pc_alfa_gamma_c12_2p9_6p5", 2.9, 6.5, abar_c, zbar_c, True),
            ("pc_alfa_gamma_c12_3p5_8p35", 3.5, 8.35, abar_c, zbar_c, True),
            ("pc_alfa_tlim_c12_3p3_7p65", 3.3, 7.65, abar_c, zbar_c, False),
            ("pc_alfa_co_1p0_7p0", 1.0, 7.0, abar_co, zbar_co, True),
            ("pc_alfa_co_2p9_7p0", 2.9, 7.0, abar_co, zbar_co, True),
            ("pc_alfa_co_5p0_6p2", 5.0, 6.2, abar_co, zbar_co, True),
            ("pc_alfa_co_6p0_7p65", 6.0, 7.65, abar_co, zbar_co, True),
        ):
            exp = _load_scalar_row("ALFA", cid)
            r_out = eos.get_pc_alfa(
                lr, lt, ab, zb, use_gamma_limit=use_gam, backend="rust"
            )
            j_out = eos.get_pc_alfa(
                jnp.float64(lr),
                jnp.float64(lt),
                jnp.float64(ab),
                jnp.float64(zb),
                use_gamma_limit=use_gam,
                backend="jax",
            )
            for col in range(3):
                np.testing.assert_allclose(
                    float(r_out[col]), exp[col], rtol=1e-13, atol=1e-15
                )
                np.testing.assert_allclose(
                    float(j_out[col]), exp[col], rtol=1e-13, atol=1e-15
                )

        for cid, lr, lt, ab, zb in (
            ("skye_alfa_c12_4p0_7p5", 4.0, 7.5, 12.0, 6.0),
            ("skye_alfa_c12_3p45_7p0", 3.45, 7.0, 12.0, 6.0),
            ("skye_alfa_co_6p0_7p0", 6.0, 7.0, abar_co, zbar_co),
            ("skye_alfa_co_m10_4p0", -10.0, 4.0, abar_co, zbar_co),
        ):
            exp = _load_scalar_row("ALFA", cid)
            sr = eos.get_skye_alfa(lr, lt, ab, zb, backend="rust")
            sj = eos.get_skye_alfa(
                jnp.float64(lr),
                jnp.float64(lt),
                jnp.float64(ab),
                jnp.float64(zb),
                backend="jax",
            )
            for col in range(3):
                np.testing.assert_allclose(
                    float(sr[col]), exp[col], rtol=1e-13, atol=1e-15
                )
                np.testing.assert_allclose(
                    float(sj[col]), exp[col], rtol=1e-12, atol=1e-15
                )

        for cid, lr, lt in (
            ("skye_alfa_simple_2p0_6p0", 2.0, 6.0),
            ("skye_alfa_simple_4p0_8p0", 4.0, 8.0),
            ("skye_alfa_simple_f64_2p35_7p3", 2.35, 7.3),
        ):
            exp = _load_scalar_row("ALFA", cid)
            sm_r = eos.get_skye_alfa_simple(lr, lt, backend="rust")
            sm_j = eos.get_skye_alfa_simple(
                jnp.float64(lr), jnp.float64(lt), backend="jax"
            )
            for col in range(3):
                np.testing.assert_allclose(
                    float(sm_r[col]), exp[col], rtol=1e-13, atol=1e-15
                )
                np.testing.assert_allclose(
                    float(sm_j[col]), exp[col], rtol=1e-13, atol=1e-15
                )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "declared gap (#161): get_skye_alfa_simple uses f64 2.3/2.4/7.2/7.4 "
        "defaults vs MESA eos/defaults/eos.defaults:225-228 real(4) literals"
    ),
)
def test_skye_alfa_simple_default_bounds_match_mesa_26_4_1_real4() -> None:
    """Check `get_skye_alfa_simple` default bounds against MESA `26.4.1`."""
    exp_def = _load_scalar_row("ALFA", "skye_alfa_simple_2p35_7p3")
    with flexmesa.numerics_mode_context("strict"):
        sm_r = eos.get_skye_alfa_simple(2.35, 7.3, backend="rust")
    for col in range(3):
        np.testing.assert_allclose(
            float(sm_r[col]), exp_def[col], rtol=1e-13, atol=1e-15
        )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "declared gap (#161): Rust strict PC and Skye derivatives differ "
        "from MESA 26.4.1 beyond the base tolerances (largest measured: "
        "pc_co_liq d_dlnT 7.1e-6, skye_co_liq d_dlnT 1.3e-6)"
    ),
)
def test_pc_and_skye_rust_strict_match_mesa_26_4_1_at_base_tolerance() -> None:
    """Rust strict `PC` and `Skye` against MESA `26.4.1` with no allowances.

    The test below this one lets known differences through with wider
    derivative tolerances for some cases. This one applies the same base
    tolerances as the Rust tests to every case (`crates/flexmesa-eos/tests/
    wave3a3_pc_tests.rs`: `1.0e-11` on values, `5.0e-9` on derivatives;
    `wave3a3_skye_inverse_tests.rs`: `1.0e-11` and `1.0e-8`), so the
    differences are a declared gap and this test starts passing only when
    they are gone.
    """
    cids, xa_c, ab_c, zb_c = _oracle_7sp_composition(0.0, 1.0, 0.0)
    _, xa_co, ab_co, zb_co = _oracle_7sp_composition(0.0, 0.5, 0.5)
    _, xa_he, ab_he, zb_he = _oracle_7sp_composition(1.0, 0.0, 0.0)
    pc_cases = (
        ("pc_c12_liq", xa_c, ab_c, zb_c, 1.0e6, 1.0e8),
        ("pc_co_liq", xa_co, ab_co, zb_co, 1.0e6, 1.0e7),
        ("pc_c12_trans_eosfi", xa_c, ab_c, zb_c, 1.0e6, 3.1e6),
        ("pc_c12_trans_phase", xa_c, ab_c, zb_c, 1.0e6, 2.05e6),
        ("pc_c12_sol", xa_c, ab_c, zb_c, 1.0e6, 1.0e6),
        ("pc_co_sol", xa_co, ab_co, zb_co, 1.0e6, 1.0e6),
        ("pc_he4_qmelt", xa_he, ab_he, zb_he, 1.0e8, 1.0e5),
    )
    skye_cases = (
        ("skye_c12_liq", xa_c, ab_c, zb_c, 1.0e6, 1.0e8, False, "Ogata"),
        ("skye_c12_liq_ion", xa_c, ab_c, zb_c, 1.0e6, 1.0e8, True, "Ogata"),
        ("skye_co_liq", xa_co, ab_co, zb_co, 1.0e6, 1.0e7, False, "Ogata"),
        (
            "skye_co_sol_ogata",
            xa_co,
            ab_co,
            zb_co,
            1.0e6,
            2.0e6,
            False,
            "Ogata",
        ),
        ("skye_co_sol_pc", xa_co, ab_co, zb_co, 1.0e6, 2.0e6, False, "PC"),
        ("skye_c12_phase", xa_c, ab_c, zb_c, 1.0e6, 2.65e6, False, "Ogata"),
    )
    with flexmesa.numerics_mode_context("strict"):
        for cid, xa, ab, zb, rho, temp in pc_cases:
            exp = _load_eos_case(cid)
            out = eos.eval_pc_eos(rho, temp, ab, zb, cids, xa, backend="rust")
            np.testing.assert_allclose(
                np.asarray(out.res), exp[:, 0], rtol=1.0e-11, atol=1.0e-14
            )
            np.testing.assert_allclose(
                np.asarray(out.d_dlnd), exp[:, 1], rtol=5.0e-9, atol=1.0e-12
            )
            np.testing.assert_allclose(
                np.asarray(out.d_dln_t), exp[:, 2], rtol=5.0e-9, atol=1.0e-12
            )
        for cid, xa, ab, zb, rho, temp, ion, rule in skye_cases:
            exp = _load_eos_case(cid)
            out = eos.eval_skye_eos(
                rho,
                temp,
                ab,
                zb,
                cids,
                xa,
                skye_use_ion_offsets=ion,
                skye_solid_mixing_rule=rule,
                backend="rust",
            )
            np.testing.assert_allclose(
                np.asarray(out.res), exp[:, 0], rtol=1.0e-11, atol=1.0e-14
            )
            np.testing.assert_allclose(
                np.asarray(out.d_dlnd), exp[:, 1], rtol=1.0e-8, atol=1.0e-12
            )
            np.testing.assert_allclose(
                np.asarray(out.d_dln_t), exp[:, 2], rtol=1.0e-8, atol=1.0e-12
            )


def test_pc_and_skye_oracle_cases_match_mesa_26_4_1_across_backends() -> None:
    """Verify all 7 `PC`, 6 `Skye`, and 3 blend cases against MESA `26.4.1`.

    The derivative tolerances here are wide enough to let known differences
    from MESA through (measured: up to `7.1e-6` for `PC`, `1.3e-6` for
    `skye_co_liq`). Those differences are a declared gap: the strict test
    above applies the base tolerances and is expected to fail (`#161`).
    """
    cids, xa_c, ab_c, zb_c = _oracle_7sp_composition(0.0, 1.0, 0.0)
    _, xa_co, ab_co, zb_co = _oracle_7sp_composition(0.0, 0.5, 0.5)
    _, xa_he, ab_he, zb_he = _oracle_7sp_composition(1.0, 0.0, 0.0)

    pc_cases = (
        ("pc_c12_liq", xa_c, ab_c, zb_c, 1.0e6, 1.0e8),
        ("pc_co_liq", xa_co, ab_co, zb_co, 1.0e6, 1.0e7),
        ("pc_c12_trans_eosfi", xa_c, ab_c, zb_c, 1.0e6, 3.1e6),
        ("pc_c12_trans_phase", xa_c, ab_c, zb_c, 1.0e6, 2.05e6),
        ("pc_c12_sol", xa_c, ab_c, zb_c, 1.0e6, 1.0e6),
        ("pc_co_sol", xa_co, ab_co, zb_co, 1.0e6, 1.0e6),
        ("pc_he4_qmelt", xa_he, ab_he, zb_he, 1.0e8, 1.0e5),
    )
    with flexmesa.numerics_mode_context("strict"):
        for cid, xa, ab, zb, rho, temp in pc_cases:
            exp = _load_eos_case(cid)
            out_r = eos.eval_pc_eos(rho, temp, ab, zb, cids, xa, backend="rust")
            out_j = eos.eval_pc_eos(
                jnp.float64(rho),
                jnp.float64(temp),
                jnp.float64(ab),
                jnp.float64(zb),
                cids,
                jnp.asarray(xa),
                backend="jax",
            )
            der_rtol = 1.0e-8 if cid == "pc_he4_qmelt" else 1.0e-5
            for out in (out_r, out_j):
                np.testing.assert_allclose(
                    np.asarray(out.res), exp[:, 0], rtol=5.0e-11, atol=1.0e-14
                )
                np.testing.assert_allclose(
                    np.asarray(out.d_dlnd),
                    exp[:, 1],
                    rtol=der_rtol,
                    atol=1.0e-12,
                )
                np.testing.assert_allclose(
                    np.asarray(out.d_dln_t),
                    exp[:, 2],
                    rtol=der_rtol,
                    atol=1.0e-12,
                )

        skye_cases = (
            ("skye_c12_liq", xa_c, ab_c, zb_c, 1.0e6, 1.0e8, False, "Ogata"),
            ("skye_c12_liq_ion", xa_c, ab_c, zb_c, 1.0e6, 1.0e8, True, "Ogata"),
            ("skye_co_liq", xa_co, ab_co, zb_co, 1.0e6, 1.0e7, False, "Ogata"),
            (
                "skye_co_sol_ogata",
                xa_co,
                ab_co,
                zb_co,
                1.0e6,
                2.0e6,
                False,
                "Ogata",
            ),
            ("skye_co_sol_pc", xa_co, ab_co, zb_co, 1.0e6, 2.0e6, False, "PC"),
            (
                "skye_c12_phase",
                xa_c,
                ab_c,
                zb_c,
                1.0e6,
                2.65e6,
                False,
                "Ogata",
            ),
        )
        for cid, xa, ab, zb, rho, temp, ion, rule in skye_cases:
            exp = _load_eos_case(cid)
            sr = eos.eval_skye_eos(
                rho,
                temp,
                ab,
                zb,
                cids,
                xa,
                skye_use_ion_offsets=ion,
                skye_solid_mixing_rule=rule,
                backend="rust",
            )
            sj = eos.eval_skye_eos(
                jnp.float64(rho),
                jnp.float64(temp),
                jnp.float64(ab),
                jnp.float64(zb),
                cids,
                jnp.asarray(xa),
                skye_use_ion_offsets=ion,
                skye_solid_mixing_rule=rule,
                backend="jax",
            )
            r_der_rtol = 2.0e-6 if cid == "skye_co_liq" else 1.0e-9
            np.testing.assert_allclose(
                np.asarray(sr.res), exp[:, 0], rtol=1.0e-10, atol=1.0e-14
            )
            np.testing.assert_allclose(
                np.asarray(sr.d_dlnd), exp[:, 1], rtol=r_der_rtol, atol=1.0e-12
            )
            np.testing.assert_allclose(
                np.asarray(sr.d_dln_t), exp[:, 2], rtol=r_der_rtol, atol=1.0e-12
            )
            np.testing.assert_allclose(
                np.asarray(sj.res), exp[:, 0], rtol=1.0e-8, atol=1.0e-14
            )
            np.testing.assert_allclose(
                np.asarray(sj.d_dlnd), exp[:, 1], rtol=5.0e-5, atol=1.0e-12
            )
            np.testing.assert_allclose(
                np.asarray(sj.d_dln_t), exp[:, 2], rtol=5.0e-5, atol=1.0e-12
            )

        # Blended eos_dt_get cases against MESA 26.4.1
        rho_pc = float(fm_math.exp10(2.9))
        temp_pc = float(fm_math.exp10(6.5))
        for cid, use_4d in (
            ("eosdt_pc_helm_blend", False),
            ("eosdt_pc_freeeos_blend", True),
        ):
            exp = _load_eos_case(cid)
            br = eos.eos_dt_get(
                rho_pc,
                temp_pc,
                cids,
                xa_c,
                use_skye=False,
                use_pc=True,
                pc_use_gamma_limit=True,
                use_free_eos=use_4d,
                use_opal_scvh=use_4d,
                backend="rust",
            )
            bj = eos.eos_dt_get(
                jnp.float64(rho_pc),
                jnp.float64(temp_pc),
                cids,
                jnp.asarray(xa_c),
                use_skye=False,
                use_pc=True,
                pc_use_gamma_limit=True,
                use_free_eos=use_4d,
                use_opal_scvh=use_4d,
                backend="jax",
            )
            np.testing.assert_allclose(
                np.asarray(br.res), exp[:, 0], rtol=1.0e-9, atol=1.0e-14
            )
            np.testing.assert_allclose(
                np.asarray(br.d_dlnd), exp[:, 1], rtol=2.0e-7, atol=1.0e-10
            )
            np.testing.assert_allclose(
                np.asarray(br.d_dln_t), exp[:, 2], rtol=2.0e-7, atol=1.0e-10
            )
            np.testing.assert_allclose(
                np.asarray(bj.res), exp[:, 0], rtol=1.0e-9, atol=1.0e-14
            )
            np.testing.assert_allclose(
                np.asarray(bj.d_dlnd), exp[:, 1], rtol=1.0e-4, atol=1.0e-10
            )
            np.testing.assert_allclose(
                np.asarray(bj.d_dln_t), exp[:, 2], rtol=1.0e-4, atol=1.0e-10
            )

        exp_sk_bl = _load_eos_case("eosdt_skye_helm_blend_ion")
        rho_sk = float(fm_math.exp10(3.45))
        skr = eos.eos_dt_get(
            rho_sk,
            1.0e7,
            cids,
            xa_c,
            use_skye=True,
            use_pc=False,
            use_free_eos=False,
            use_opal_scvh=False,
            backend="rust",
        )
        skj = eos.eos_dt_get(
            jnp.float64(rho_sk),
            jnp.float64(1.0e7),
            cids,
            jnp.asarray(xa_c),
            use_skye=True,
            use_pc=False,
            use_free_eos=False,
            use_opal_scvh=False,
            backend="jax",
        )
        for out in (skr, skj):
            np.testing.assert_allclose(
                np.asarray(out.res), exp_sk_bl[:, 0], rtol=1.0e-11, atol=1.0e-14
            )
            np.testing.assert_allclose(
                np.asarray(out.d_dlnd),
                exp_sk_bl[:, 1],
                rtol=1.0e-5,
                atol=1.0e-10,
            )
            np.testing.assert_allclose(
                np.asarray(out.d_dln_t),
                exp_sk_bl[:, 2],
                rtol=1.0e-5,
                atol=1.0e-10,
            )


def test_eval_pc_eos_and_zones_match_across_rust_and_jax() -> None:
    """Verify `eval_pc_eos` and `eval_pc_eos_zones` across liquid/solid WDs.

    This compares the derived JAX layer with Rust strict, not with MESA. The
    derivative tolerance is `5.0e-9`. It was `1.0e-9` until commit `850f223`,
    which widened it after the comparison measured `1.432e-9` on the CI
    runner while passing on the author's machine. The cause of the
    machine dependence in the JAX result has not been established.
    """
    chem_id, xa, abar, zbar = _c12_o16_composition()
    azion = np.array(
        [float(chem.get_nuclide_info(c).z) for c in chem_id], dtype=np.float64
    )
    acmi = np.array(
        [float(chem.get_nuclide_info(c).w) for c in chem_id], dtype=np.float64
    )

    for rho, temp, expected_phase, inc_rad in [
        (1.0e5, 1.0e7, 0.0, True),
        (1.0e7, 10.0**6.3, 1.0, False),
    ]:
        out_r = eos.eval_pc_eos(
            rho,
            temp,
            abar,
            zbar,
            chem_id,
            xa,
            include_radiation=inc_rad,
            backend="rust",
        )
        out_j = eos.eval_pc_eos(
            jnp.asarray(rho, dtype=jnp.float64),
            jnp.asarray(temp, dtype=jnp.float64),
            jnp.asarray(abar, dtype=jnp.float64),
            jnp.asarray(zbar, dtype=jnp.float64),
            chem_id,
            jnp.asarray(xa, dtype=jnp.float64),
            include_radiation=inc_rad,
            backend="jax",
        )
        assert abs(float(out_r.res[eos.I_PHASE]) - expected_phase) < 1.0e-12
        assert abs(float(out_r.res[eos.I_FRAC_PC]) - 1.0) < 1.0e-14
        tol_res = 1.0e-13 if rho < 1.0e6 else 5.0e-12
        _assert_unfloored(
            np.asarray(out_j.res), np.asarray(out_r.res), rtol=tol_res
        )
        _assert_unfloored(
            np.asarray(out_j.d_dlnd), np.asarray(out_r.d_dlnd), rtol=5.0e-9
        )
        _assert_unfloored(
            np.asarray(out_j.d_dln_t), np.asarray(out_r.d_dln_t), rtol=5.0e-9
        )

    rhos = np.array([1.0e5, 1.0e6, 1.0e7], dtype=np.float64)
    temps = np.array([1.0e7, 5.0e6, 2.0e6], dtype=np.float64)
    abars = np.full(3, abar, dtype=np.float64)
    zbars = np.full(3, zbar, dtype=np.float64)
    xa_z = np.tile(xa, (3, 1))
    zones_r = eos.eval_pc_eos_zones(
        rhos, temps, abars, zbars, xa_z, azion, acmi, backend="rust"
    )
    zones_j = eos.eval_pc_eos_zones(
        jnp.asarray(rhos),
        jnp.asarray(temps),
        jnp.asarray(abars),
        jnp.asarray(zbars),
        jnp.asarray(xa_z),
        jnp.asarray(azion),
        jnp.asarray(acmi),
        backend="jax",
    )
    assert zones_r.res.shape == (3, eos.NUM_EOS_BASIC_RESULTS)
    assert zones_j.res.shape == (3, eos.NUM_EOS_BASIC_RESULTS)
    _assert_unfloored(
        np.asarray(zones_j.res), np.asarray(zones_r.res), rtol=5.0e-12
    )
    _assert_unfloored(
        np.asarray(zones_j.d_dlnd), np.asarray(zones_r.d_dlnd), rtol=5.0e-9
    )
    _assert_unfloored(
        np.asarray(zones_j.d_dln_t), np.asarray(zones_r.d_dln_t), rtol=5.0e-9
    )


def test_eval_skye_eos_and_eos_dt_get_rust_jax_and_autodiff() -> None:
    """Verify `eval_skye_eos`/`eos_dt_get` consistency and `jax.grad`."""
    chem_id, xa, abar, zbar = _c12_o16_composition()
    rho = 1.0e6
    temp = 1.0e7

    with flexmesa.numerics_mode_context("strict"):
        sk_r = eos.eval_skye_eos(
            rho, temp, abar, zbar, chem_id, xa, backend="rust"
        )
    sk_j = eos.eval_skye_eos(
        jnp.asarray(rho, dtype=jnp.float64),
        jnp.asarray(temp, dtype=jnp.float64),
        jnp.asarray(abar, dtype=jnp.float64),
        jnp.asarray(zbar, dtype=jnp.float64),
        chem_id,
        jnp.asarray(xa, dtype=jnp.float64),
        backend="jax",
    )
    _assert_unfloored(np.asarray(sk_j.res), np.asarray(sk_r.res), rtol=1.0e-11)
    _assert_unfloored(
        np.asarray(sk_j.d_dlnd),
        np.asarray(sk_r.d_dlnd),
        rtol=3.0e-6,
        near_zero_cutoff=1.0e-12,
    )
    # JAX validated against Rust strict at unfloored tolerance 1.0e-5
    # (biquintic F_TTT cancels ~10^16 down to ~10^6 in Cv/Cp/grad_ad d_dln_t).
    _assert_unfloored(
        np.asarray(sk_j.d_dln_t),
        np.asarray(sk_r.d_dln_t),
        rtol=1.0e-5,
        near_zero_cutoff=1.0e-12,
    )
    assert float(sk_r.res[eos.I_FRAC_SKYE]) == 1.0

    infos = [chem.get_nuclide_info(c) for c in chem_id]
    z_arr = np.array([inf.z for inf in infos], dtype=np.int64)
    acmi_arr = np.array([inf.w for inf in infos], dtype=np.float64)
    a_arr = np.array([float(inf.a) for inf in infos], dtype=np.float64)
    rhos_z = np.array([1.0e2, 1.0e6], dtype=np.float64)
    temps_z = np.array([1.0e8, 1.0e7], dtype=np.float64)
    abars_z = np.full(2, abar, dtype=np.float64)
    zbars_z = np.full(2, zbar, dtype=np.float64)
    xa_z = np.tile(xa, (2, 1))
    sk_zr = eos.eval_skye_eos_zones(
        rhos_z,
        temps_z,
        abars_z,
        zbars_z,
        xa_z,
        z_arr,
        acmi_arr,
        a_arr,
        backend="rust",
    )
    sk_zj = eos.eval_skye_eos_zones(
        jnp.asarray(rhos_z),
        jnp.asarray(temps_z),
        jnp.asarray(abars_z),
        jnp.asarray(zbars_z),
        jnp.asarray(xa_z),
        jnp.asarray(z_arr),
        jnp.asarray(acmi_arr),
        jnp.asarray(a_arr),
        backend="jax",
    )
    _assert_unfloored(
        np.asarray(sk_zj.res), np.asarray(sk_zr.res), rtol=1.0e-10
    )

    dt_r = eos.eos_dt_get(rho, temp, chem_id, xa, backend="rust")
    dt_j = eos.eos_dt_get(
        jnp.asarray(rho, dtype=jnp.float64),
        jnp.asarray(temp, dtype=jnp.float64),
        chem_id,
        jnp.asarray(xa, dtype=jnp.float64),
        backend="jax",
    )
    _assert_unfloored(np.asarray(dt_j.res), np.asarray(dt_r.res), rtol=1.0e-11)

    def _ln_pgas_fn(r_in: jax.Array, t_in: jax.Array) -> jax.Array:
        out = eos.eos_dt_get(
            r_in, t_in, chem_id, jnp.asarray(xa), backend="jax"
        )
        return jnp.asarray(out.res)[eos.I_LN_PGAS]

    g_rho, g_t = jax.grad(_ln_pgas_fn, argnums=(0, 1))(
        jnp.asarray(rho, dtype=jnp.float64),
        jnp.asarray(temp, dtype=jnp.float64),
    )
    assert (
        abs(float(g_rho) - float(dt_r.d_dlnd[eos.I_LN_PGAS]) / rho)
        / abs(float(g_rho))
        < 1.0e-11
    )
    assert (
        abs(float(g_t) - float(dt_r.d_dln_t[eos.I_LN_PGAS]) / temp)
        / abs(float(g_t))
        < 1.0e-11
    )


def test_inverse_solvers_match_mesa_26_4_1_oracle_and_jax_ift_grad() -> None:
    """Verify inverse solvers against MESA `26.4.1` `INV` rows and `jax.grad`.

    Upstream Fortran MESA `eos/private/eospt_eval.f90:180-191`
    (`Get_PT_Results_using_DT`) hardcodes `other_tol = 1d-8`,
    `logRho_tol = 1d-8`, and `max_iter = 20`, terminating Newton iteration in
    `num/private/mod_root.f90:102` (`do_safe_root_with_guess`) when
    `|ln P_gas(x_i) - ln P_gas_target| < max(other_tol, |dy/dx| * logRho_tol)`.
    """
    cids7, xa_c, abar_c, zbar_c = _oracle_7sp_composition(0.0, 1.0, 0.0)
    with flexmesa.numerics_mode_context("strict"):
        fwd_c = eos.eos_dt_get(
            1.0e4,
            1.0e8,
            cids7,
            xa_c,
            use_skye=False,
            use_pc=False,
            use_free_eos=False,
            use_opal_scvh=False,
            backend="rust",
        )
        ln_pgas_c = float(fwd_c.res[eos.I_LN_PGAS])
        ln_e_c = float(fwd_c.res[eos.I_LN_E])
        pgas_c = float(fm_math.exp10(ln_pgas_c / constants.LN10))

        exp_pt = _load_scalar_row("INV", "helm_c12_pt_4_8")
        pt_c_r = eos.eos_pt_get(
            pgas_c,
            1.0e8,
            cids7,
            xa_c,
            use_skye=False,
            use_pc=False,
            use_free_eos=False,
            use_opal_scvh=False,
            backend="rust",
        )
        pt_c_j = eos.eos_pt_get(
            pgas_c,
            1.0e8,
            cids7,
            xa_c,
            use_skye=False,
            use_pc=False,
            use_free_eos=False,
            use_opal_scvh=False,
            backend="jax",
        )
        for pt_c in (pt_c_r, pt_c_j):
            np.testing.assert_allclose(float(pt_c.rho), exp_pt[0], rtol=1e-8)
            np.testing.assert_allclose(
                float(pt_c.log10_rho), exp_pt[1], rtol=1e-8
            )
            np.testing.assert_allclose(
                float(pt_c.dln_rho_dln_pgas_const_t), exp_pt[2], rtol=1e-8
            )
            np.testing.assert_allclose(
                float(pt_c.dln_rho_dln_t_const_pgas), exp_pt[3], rtol=1e-8
            )

        exp_get_t = _load_scalar_row("INV", "helm_c12_get_t_4_8")
        inv_tc = eos.eos_dt_get_t(
            1.0e4,
            eos.I_LN_E,
            ln_e_c,
            cids7,
            xa_c,
            log_t_guess=7.5,
            use_skye=False,
            use_pc=False,
            use_free_eos=False,
            use_opal_scvh=False,
        )
        np.testing.assert_allclose(
            float(inv_tc.log10_result), exp_get_t[0], rtol=1e-9
        )
        np.testing.assert_allclose(
            float(inv_tc.eos.res[eos.I_LN_PGAS]), exp_get_t[1], rtol=1e-9
        )

        exp_get_r = _load_scalar_row("INV", "helm_c12_get_rho_4_8")
        inv_rc = eos.eos_dt_get_rho(
            1.0e8,
            eos.I_LN_PGAS,
            ln_pgas_c,
            cids7,
            xa_c,
            log_rho_guess=3.5,
            use_skye=False,
            use_pc=False,
            use_free_eos=False,
            use_opal_scvh=False,
        )
        np.testing.assert_allclose(
            float(inv_rc.log10_result), exp_get_r[0], rtol=1e-10
        )
        np.testing.assert_allclose(
            float(inv_rc.eos.res[eos.I_LN_E]), exp_get_r[1], rtol=1e-9
        )

        exp_gde = _load_scalar_row("INV", "gamma_de_c12_1e4")
        exp_gde_res = _load_scalar_row("INV", "gamma_de_c12_1e4_res")
        gde_c = eos.eos_gamma_de_get(
            abar_c, zbar_c, 1.0e4, float(np.exp(ln_e_c)), gamma=5.0 / 3.0
        )
        np.testing.assert_allclose(float(gde_c.result), exp_gde[0], rtol=1e-13)
        np.testing.assert_allclose(
            float(gde_c.log10_result), exp_gde[1], rtol=1e-13
        )
        np.testing.assert_allclose(
            float(gde_c.eos.res[eos.I_MU]), exp_gde_res[0], rtol=1e-14
        )
        np.testing.assert_allclose(
            float(gde_c.eos.res[eos.I_LN_PGAS]), exp_gde_res[3], rtol=1e-13
        )
        np.testing.assert_allclose(
            float(gde_c.eos.res[eos.I_LN_E]), exp_gde_res[4], rtol=1e-13
        )

    chem_id, xa, _, _ = _c12_o16_composition()
    rho_true = 100.0
    temp_true = 1.0e7
    fwd = eos.eos_dt_get(rho_true, temp_true, chem_id, xa, backend="rust")
    pgas_true = float(np.exp(fwd.res[eos.I_LN_PGAS]))

    # Verify Implicit Function Theorem custom_jvp on eos_pt_get vs finite diff
    def _rho_from_pt(p_in: jax.Array, t_in: jax.Array) -> jax.Array:
        pt_out = eos.eos_pt_get(
            p_in, t_in, chem_id, jnp.asarray(xa), backend="jax"
        )
        return jnp.asarray(pt_out.rho)

    drho_dp, drho_dt = jax.grad(_rho_from_pt, argnums=(0, 1))(
        jnp.asarray(pgas_true, dtype=jnp.float64),
        jnp.asarray(temp_true, dtype=jnp.float64),
    )
    eps = 1.0e-6
    pt_p_hi = eos.eos_pt_get(
        pgas_true * (1.0 + eps), temp_true, chem_id, xa, backend="rust"
    )
    pt_p_lo = eos.eos_pt_get(
        pgas_true * (1.0 - eps), temp_true, chem_id, xa, backend="rust"
    )
    fd_dp = (float(pt_p_hi.rho) - float(pt_p_lo.rho)) / (2.0 * eps * pgas_true)
    assert abs(float(drho_dp) - fd_dp) / abs(fd_dp) < 1.0e-6

    pt_t_hi = eos.eos_pt_get(
        pgas_true, temp_true * (1.0 + eps), chem_id, xa, backend="rust"
    )
    pt_t_lo = eos.eos_pt_get(
        pgas_true, temp_true * (1.0 - eps), chem_id, xa, backend="rust"
    )
    fd_dt = (float(pt_t_hi.rho) - float(pt_t_lo.rho)) / (2.0 * eps * temp_true)
    assert abs(float(drho_dt) - fd_dt) / abs(fd_dt) < 1.0e-6


def test_pc_and_skye_grid_rust_jax_consistency_and_jacfwd() -> None:
    """Verify `eval_pc_eos` and `eval_skye_eos` `jax.jacfwd` consistency."""
    chem_id, xa, abar, zbar = _c12_o16_composition()
    rho0 = jnp.float64(10.0**5.4)
    t0 = jnp.float64(10.0**6.8)
    pc0 = eos.eval_pc_eos(
        rho0,
        t0,
        jnp.float64(abar),
        jnp.float64(zbar),
        chem_id,
        jnp.asarray(xa),
        backend="jax",
    )
    jac_pc_r, jac_pc_t = jax.jacfwd(
        lambda r, t: jnp.asarray(
            eos.eval_pc_eos(
                r,
                t,
                jnp.float64(abar),
                jnp.float64(zbar),
                chem_id,
                jnp.asarray(xa),
                backend="jax",
            ).res
        ),
        argnums=(0, 1),
    )(rho0, t0)
    _assert_unfloored(
        np.asarray(jac_pc_r * rho0), np.asarray(pc0.d_dlnd), rtol=1.0e-11
    )
    _assert_unfloored(
        np.asarray(jac_pc_t * t0), np.asarray(pc0.d_dln_t), rtol=1.0e-11
    )

    sk0 = eos.eval_skye_eos(
        rho0,
        t0,
        jnp.float64(abar),
        jnp.float64(zbar),
        chem_id,
        jnp.asarray(xa),
        backend="jax",
    )
    jac_sk_r, jac_sk_t = jax.jacfwd(
        lambda r, t: jnp.asarray(
            eos.eval_skye_eos(
                r,
                t,
                jnp.float64(abar),
                jnp.float64(zbar),
                chem_id,
                jnp.asarray(xa),
                backend="jax",
            ).res
        ),
        argnums=(0, 1),
    )(rho0, t0)
    _assert_unfloored(
        np.asarray(jac_sk_r * rho0), np.asarray(sk0.d_dlnd), rtol=1.0e-11
    )
    _assert_unfloored(
        np.asarray(jac_sk_t * t0), np.asarray(sk0.d_dln_t), rtol=1.0e-11
    )


def test_neutron_bearing_multispecies_buckets_and_multizone_dispatch() -> None:
    """Verify `Z = 0` neutrons, `32`/`64` buckets, and `N_zones` paths."""
    names_9 = ("neut", "h1", "he4", "c12", "n14", "o16", "ne20", "mg24", "si28")
    cids_9 = [int(chem.get_nuclide_index(n) or 0) for n in names_9]
    assert all(c > 0 for c in cids_9)
    xa_9 = np.array(
        [0.05, 0.02, 0.13, 0.25, 0.03, 0.30, 0.10, 0.07, 0.05], dtype=np.float64
    )
    comp_9 = chem.get_composition_info(cids_9, xa_9)
    ab_9, zb_9 = float(comp_9.abar), float(comp_9.zbar)
    ab_j, zb_j, xa_9j = jnp.float64(ab_9), jnp.float64(zb_9), jnp.asarray(xa_9)

    for rho, temp in [(1.0e5, 1.0e8), (1.0e7, 2.0e6)]:
        r_j, t_j = jnp.float64(rho), jnp.float64(temp)
        sk_r = eos.eval_skye_eos(
            rho,
            temp,
            ab_9,
            zb_9,
            cids_9,
            xa_9,
            mass_fraction_limit=0.0,
            backend="rust",
        )
        sk_j = eos.eval_skye_eos(
            r_j,
            t_j,
            ab_j,
            zb_j,
            cids_9,
            xa_9j,
            mass_fraction_limit=0.0,
            backend="jax",
        )
        _assert_unfloored(np.asarray(sk_j.res), np.asarray(sk_r.res), rtol=1e-6)

        pc_r = eos.eval_pc_eos(
            rho,
            temp,
            ab_9,
            zb_9,
            cids_9,
            xa_9,
            mass_fraction_limit=0.0,
            backend="rust",
        )
        pc_j = eos.eval_pc_eos(
            r_j,
            t_j,
            ab_j,
            zb_j,
            cids_9,
            xa_9j,
            mass_fraction_limit=0.0,
            backend="jax",
        )
        _assert_unfloored(
            np.asarray(pc_j.res), np.asarray(pc_r.res), rtol=1.0e-11
        )
        _assert_unfloored(
            np.asarray(pc_j.d_dlnd), np.asarray(pc_r.d_dlnd), rtol=5.0e-9
        )
        _assert_unfloored(
            np.asarray(pc_j.d_dln_t), np.asarray(pc_r.d_dln_t), rtol=5.0e-9
        )

    dt_r = eos.eos_dt_get(1.0e6, 1.0e7, cids_9, xa_9, backend="rust")
    dt_j = eos.eos_dt_get(
        jnp.float64(1.0e6), jnp.float64(1.0e7), cids_9, xa_9j, backend="jax"
    )
    _assert_unfloored(np.asarray(dt_j.res), np.asarray(dt_r.res), rtol=1.0e-10)

    # 35-species mixture (64-bucket) + N_zones = 20 (> 16 vmap) + N_zones = 0
    cids_35 = list(range(1, 36))
    xa_35 = np.full(35, 1.0 / 35.0, dtype=np.float64)
    comp_35 = chem.get_composition_info(cids_35, xa_35)
    infos_35 = [chem.get_nuclide_info(c) for c in cids_35]
    z_35 = np.array([float(inf.z) for inf in infos_35], dtype=np.float64)
    w_35 = np.array([inf.w for inf in infos_35], dtype=np.float64)
    a_35 = np.array([float(inf.a) for inf in infos_35], dtype=np.float64)
    z_35j, w_35j, a_35j = (
        jnp.asarray(z_35),
        jnp.asarray(w_35),
        jnp.asarray(a_35),
    )

    nz = 20
    r_20, t_20 = np.geomspace(1.0e4, 1.0e7, nz), np.geomspace(2.0e6, 5.0e7, nz)
    ab_20 = np.full(nz, float(comp_35.abar), dtype=np.float64)
    zb_20 = np.full(nz, float(comp_35.zbar), dtype=np.float64)
    xa_20 = np.tile(xa_35, (nz, 1))
    z_args_j = (
        jnp.asarray(r_20),
        jnp.asarray(t_20),
        jnp.asarray(ab_20),
        jnp.asarray(zb_20),
        jnp.asarray(xa_20),
    )

    pc_z_r = eos.eval_pc_eos_zones(
        r_20, t_20, ab_20, zb_20, xa_20, z_35, w_35, backend="rust"
    )
    pc_z_j = eos.eval_pc_eos_zones(*z_args_j, z_35j, w_35j, backend="jax")
    _assert_unfloored(
        np.asarray(pc_z_j.res), np.asarray(pc_z_r.res), rtol=1.0e-11
    )

    sk_z_r = eos.eval_skye_eos_zones(
        r_20,
        t_20,
        ab_20,
        zb_20,
        xa_20,
        z_35.astype(np.int64),
        w_35,
        a_35,
        backend="rust",
    )
    sk_z_j = eos.eval_skye_eos_zones(
        *z_args_j, z_35j, w_35j, a_35j, backend="jax"
    )
    _assert_unfloored(np.asarray(sk_z_j.res), np.asarray(sk_z_r.res), rtol=1e-6)

    ef, ex = jnp.zeros((0,), jnp.float64), jnp.zeros((0, 35), jnp.float64)
    pc_0 = eos.eval_pc_eos_zones(
        ef, ef, ef, ef, ex, z_35j, w_35j, backend="jax"
    )
    sk_0 = eos.eval_skye_eos_zones(
        ef, ef, ef, ef, ex, z_35j, w_35j, a_35j, backend="jax"
    )
    assert pc_0.res.shape == (0, eos.NUM_EOS_BASIC_RESULTS)
    assert sk_0.res.shape == (0, eos.NUM_EOS_BASIC_RESULTS)
