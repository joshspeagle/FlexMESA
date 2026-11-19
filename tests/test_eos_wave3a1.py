# SPDX-License-Identifier: LGPL-3.0-or-later
"""Verification tests for Wave 3a.1 (`flexmesa.eos`: blend, dfermi, ideal)."""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import flexmesa
from flexmesa import chem, constants, eos

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_EOS_TEST_OUTPUT = (
    _REPO_ROOT / "tests" / "fixtures" / "golden" / "eos" / ("test_output.txt")
)
_EOS_ORACLES_TSV = (
    _REPO_ROOT
    / "benchmarks"
    / "fortran_mesa"
    / "eos_oracles"
    / "mesa_26_4_1_eos_oracles.tsv"
)


def _parse_blend_section() -> list[tuple[bool, tuple[float, float, float]]]:
    """Parse the 5 `do_test_eos_blend` blocks at the end of `test_output`."""
    lines = _EOS_TEST_OUTPUT.read_text("utf-8").splitlines()
    out: list[tuple[bool, tuple[float, float, float]]] = []
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        if stripped in {"T", "F"} and idx + 1 < len(lines):
            parts = lines[idx + 1].replace("D", "E").split()
            if len(parts) == 3:
                vals = (float(parts[0]), float(parts[1]), float(parts[2]))
                out.append((stripped == "T", vals))
                idx += 2
                continue
        idx += 1
    return out


def _load_dfermi_rows() -> dict[str, tuple[float, ...]]:
    """Load `DFERMI` rows from `mesa_26_4_1_eos_oracles.tsv`."""
    rows: dict[str, tuple[float, ...]] = {}
    for line in _EOS_ORACLES_TSV.read_text("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if cols[0] == "DFERMI":
            rows[cols[1]] = tuple(float(x) for x in cols[2:])
    return rows


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


def test_eos_basic_result_names_match_upstream_mesa_schema() -> None:
    """Verify the 26 canonical result channel names and indices."""
    assert len(eos.EOS_BASIC_RESULT_NAMES) == eos.NUM_EOS_BASIC_RESULTS == 26
    assert eos.EOS_BASIC_RESULT_NAMES[eos.I_LN_PGAS] == "lnPgas"
    assert eos.EOS_BASIC_RESULT_NAMES[eos.I_LN_E] == "lnE"
    assert eos.EOS_BASIC_RESULT_NAMES[eos.I_LN_S] == "lnS"
    assert eos.EOS_BASIC_RESULT_NAMES[eos.I_FRAC_IDEAL] == "ideal"


def test_eos_blend_polygon_matches_upstream_test_output() -> None:
    """Verify polygonal containment and distance against `test_output`."""
    coords = np.array(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0]],
        dtype=np.float64,
    )
    blend_cases = _parse_blend_section()
    assert len(blend_cases) == 5
    query_points = [
        (0.0, 1.0),
        (-1.0, 1.0),
        (0.0, 0.0),
        (-2.0, 0.0),
        (-1.0, 3.0),
    ]

    with flexmesa.numerics_mode_context("strict"):
        for (x, y), (exp_in, (exp_d, exp_dx, exp_dy)) in zip(
            query_points, blend_cases, strict=True
        ):
            assert eos.is_contained(coords, x, y) is exp_in
            # `do_test_eos_blend` in `eos/test/src/test_eos_blend.f90` assigns
            # constant dual inputs `p(1) = x`, `p(2) = y` (zero input partials).
            assert exp_dx == 0.0 and exp_dy == 0.0
            d_r, _, _ = eos.min_distance_to_polygon(
                coords, x, y, backend="rust"
            )
            d_j, _, _ = eos.min_distance_to_polygon(coords, x, y, backend="jax")
            np.testing.assert_allclose(float(d_r), exp_d, rtol=1e-15, atol=0.0)
            np.testing.assert_allclose(float(d_j), exp_d, rtol=1e-15, atol=0.0)

        # Check seeded gradient at (-2.0, 0.25)
        d_r, dx_r, dy_r = eos.min_distance_to_polygon(
            coords, -2.0, 0.25, backend="rust"
        )
    d_j, dx_j, dy_j = eos.min_distance_to_polygon(
        coords, -2.0, 0.25, backend="jax"
    )
    assert abs(float(d_r) - float(d_j)) < 1.0e-15
    assert abs(float(dx_r) - (-1.0)) < 1.0e-15
    assert abs(float(dx_j) - (-1.0)) < 1.0e-15
    assert abs(float(dy_r)) < 1.0e-15
    assert abs(float(dy_j)) < 1.0e-15


def test_quintic_blend_rust_jax_consistency() -> None:
    """Verify quintic blend polynomial and derivative across Rust and JAX."""
    alphas = np.linspace(-0.2, 1.2, 15, dtype=np.float64)
    with flexmesa.numerics_mode_context("strict"):
        for a in alphas:
            s_r, ds_r = eos.quintic_blend(float(a), backend="rust")
            s_j, ds_j = eos.quintic_blend(jnp.float64(a), backend="jax")
            assert abs(float(s_r) - float(s_j)) < 1.0e-15
            assert abs(float(ds_r) - float(ds_j)) < 1.0e-15


def test_dfermi_matches_mesa_26_4_1_oracle_across_rust_and_jax() -> None:
    """Verify `dfermi` on all 100 MESA `26.4.1` `DFERMI` oracle points."""
    rows = _load_dfermi_rows()
    assert len(rows) == 100

    eta = 1.46722890948893
    temp = 11327678.5183021
    theta = (constants.KERG * temp) / (constants.ME * (constants.CLIGHT**2))

    with flexmesa.numerics_mode_context("strict"):
        for name in (
            "dirac_test_m0p5",
            "dirac_test_p0p5",
            "dirac_test_p1p5",
            "dirac_test_p2p5",
        ):
            dk, eta_ref, theta_ref, exp_fd, exp_de, exp_dt = rows[name]
            np.testing.assert_allclose(eta, eta_ref, rtol=1e-15, atol=0.0)
            np.testing.assert_allclose(theta, theta_ref, rtol=1e-15, atol=0.0)
            fd_r, de_r, dt_r = eos.dfermi(dk, eta, theta, backend="rust")
            fd_j, de_j, dt_j = eos.dfermi(
                dk, jnp.float64(eta), jnp.float64(theta), backend="jax"
            )
            for got, exp in (
                (float(fd_r), exp_fd),
                (float(de_r), exp_de),
                (float(dt_r), exp_dt),
                (float(fd_j), exp_fd),
                (float(de_j), exp_de),
                (float(dt_j), exp_dt),
            ):
                np.testing.assert_allclose(got, exp, rtol=1e-14, atol=0.0)

        for idx in range(96):
            dk, eta_g, theta_g, exp_fd, exp_de, exp_dt = rows[f"grid_{idx}"]
            fd_r, de_r, dt_r = eos.dfermi(dk, eta_g, theta_g, backend="rust")
            fd_j, de_j, dt_j = eos.dfermi(
                dk, jnp.float64(eta_g), jnp.float64(theta_g), backend="jax"
            )
            for got, exp in (
                (float(fd_r), exp_fd),
                (float(de_r), exp_de),
                (float(dt_r), exp_dt),
                (float(fd_j), exp_fd),
                (float(de_j), exp_de),
                (float(dt_j), exp_dt),
            ):
                np.testing.assert_allclose(got, exp, rtol=1e-14, atol=0.0)

    # Batch zone evaluation
    etas = np.array([eta, 0.5, 5.0], dtype=np.float64)
    thetas = np.array([theta, 1.0e-3, 1.0e-2], dtype=np.float64)
    with flexmesa.numerics_mode_context("strict"):
        fd_zr, deta_zr, dtheta_zr = eos.dfermi(
            1.5, etas, thetas, backend="rust"
        )
    fd_zj, deta_zj, dtheta_zj = eos.dfermi(1.5, etas, thetas, backend="jax")
    np.testing.assert_allclose(fd_zr, np.asarray(fd_zj), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(
        deta_zr, np.asarray(deta_zj), rtol=1e-14, atol=1e-14
    )
    np.testing.assert_allclose(
        dtheta_zr, np.asarray(dtheta_zj), rtol=1e-14, atol=1e-14
    )


def test_ion_offset_and_ideal_eos_match_mesa_26_4_1_oracle() -> None:
    """Verify `compute_ion_offset` and `eval_ideal_eos*` vs MESA `26.4.1`."""
    cids = [int(chem.get_nuclide_index(n) or 0) for n in ("h1", "he4", "c12")]
    assert all(c > 0 for c in cids)
    xa = np.array([0.70, 0.28, 0.02], dtype=np.float64)
    infos = [chem.get_nuclide_info(c) for c in cids]
    z = [inf.z for inf in infos]
    a = np.array([float(inf.a) for inf in infos], dtype=np.float64)
    weights = np.array([inf.w for inf in infos], dtype=np.float64)
    comp = chem.get_composition_info(cids, xa)
    abar = float(comp.abar)
    zbar = float(comp.zbar)

    with flexmesa.numerics_mode_context("strict"):
        off_r = eos.compute_ion_offset_from_za(xa, z, a, backend="rust")
    off_j = eos.compute_ion_offset_from_za(xa, z, a, backend="jax")
    assert abs(float(off_r) - float(off_j)) / float(off_r) < 1.0e-15

    temp = 1.0e5
    rho = 1.0e-2
    exp_ideal = _load_eos_case("ideal_solar_1e5_1em2")

    with flexmesa.numerics_mode_context("strict"):
        out_cat = eos.eval_ideal_eos(temp, rho, abar, zbar, xa, cids)
        out_r = eos.eval_ideal_eos_from_weights(
            temp, rho, abar, zbar, xa, weights, a, backend="rust"
        )
    out_j = eos.eval_ideal_eos_from_weights(
        jnp.float64(temp),
        jnp.float64(rho),
        jnp.float64(abar),
        jnp.float64(zbar),
        jnp.asarray(xa),
        jnp.asarray(weights),
        jnp.asarray(a),
        backend="jax",
    )

    non_canceling = [
        idx for idx in range(eos.NUM_EOS_BASIC_RESULTS) if idx != eos.I_DE_DRHO
    ]
    for out in (out_cat, out_r, out_j):
        res_arr = np.asarray(out.res)
        dlnd_arr = np.asarray(out.d_dlnd)
        dlnt_arr = np.asarray(out.d_dln_t)
        np.testing.assert_allclose(
            res_arr[non_canceling],
            exp_ideal[non_canceling, 0],
            rtol=1.0e-13,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            float(res_arr[eos.I_DE_DRHO]),
            float(exp_ideal[eos.I_DE_DRHO, 0]),
            rtol=1.0e-10,
            atol=0.0,
        )
        np.testing.assert_allclose(
            dlnd_arr, exp_ideal[:, 1], rtol=5.0e-10, atol=1.0e-15
        )
        np.testing.assert_allclose(
            dlnt_arr, exp_ideal[:, 2], rtol=5.0e-10, atol=1.0e-15
        )

    # Verify JAX autodiff through eval_ideal_eos_from_weights
    grad_ln_pgas = jax.grad(
        lambda t_in: eos.eval_ideal_eos_from_weights(
            t_in,
            jnp.float64(rho),
            abar,
            zbar,
            jnp.asarray(xa),
            jnp.asarray(weights),
            jnp.asarray(a),
            backend="jax",
        ).res[eos.I_LN_PGAS]
    )(jnp.float64(temp))
    assert abs(float(grad_ln_pgas * temp) - 1.0) < 1.0e-13

    # Verify batch eval_ideal_eos_zones and blend_eos_results consistency
    t_zones = np.array([1.0e5, 2.0e5], dtype=np.float64)
    r_zones = np.array([1.0e-2, 5.0e-2], dtype=np.float64)
    with flexmesa.numerics_mode_context("strict"):
        z_r = eos.eval_ideal_eos_zones(
            t_zones, r_zones, abar, zbar, xa, weights, a, backend="rust"
        )
    z_j = eos.eval_ideal_eos_zones(
        t_zones, r_zones, abar, zbar, xa, weights, a, backend="jax"
    )
    np.testing.assert_allclose(
        z_r.res, np.asarray(z_j.res), rtol=1e-10, atol=1e-10
    )
    with pytest.raises(ValueError, match="strictly positive"):
        eos.eval_ideal_eos_zones(
            np.array([-1.0e5]),
            np.array([1.0e-2]),
            abar,
            zbar,
            xa,
            weights,
            a,
            backend="rust",
        )

    with flexmesa.numerics_mode_context("strict"):
        b_r = eos.blend_eos_results(
            0.35, 0.8, -0.4, False, out_r, out_r, backend="rust"
        )
    b_j = eos.blend_eos_results(
        0.35, 0.8, -0.4, False, out_j, out_j, backend="jax"
    )
    np.testing.assert_allclose(
        b_r.res, np.asarray(b_j.res), rtol=1e-10, atol=1e-10
    )


def test_dfermi_autodiff_and_blend_grid_consistency() -> None:
    """Verify `dfermi` JAX autodiff and `blend_eos_results` across `alfa`."""
    for dk_ad in (0.5, 1.5):
        eta0 = jnp.float64(1.8)
        theta0 = jnp.float64(0.05)
        _, de_an, dt_an = eos.dfermi(dk_ad, eta0, theta0, backend="jax")
        g_eta, g_theta = jax.grad(
            lambda e, th, k=dk_ad: eos.dfermi(k, e, th, backend="jax")[0],
            argnums=(0, 1),
        )(eta0, theta0)
        np.testing.assert_allclose(float(g_eta), float(de_an), rtol=1e-12)
        np.testing.assert_allclose(float(g_theta), float(dt_an), rtol=1e-12)

    xa = np.array([0.70, 0.28, 0.02], dtype=np.float64)
    weights = np.array([1.0078250322, 4.0026032541, 12.0], dtype=np.float64)
    a = np.array([1.0, 4.0, 12.0], dtype=np.float64)
    with flexmesa.numerics_mode_context("strict"):
        out1_r = eos.eval_ideal_eos_from_weights(
            1.0e5, 1.0e-2, 1.3, 1.1, xa, weights, a, backend="rust"
        )
        out2_r = eos.eval_ideal_eos_from_weights(
            2.5e5, 3.0e-2, 1.3, 1.1, xa, weights, a, backend="rust"
        )
    out1_j = eos.EosEvalOutput(
        jnp.asarray(out1_r.res),
        jnp.asarray(out1_r.d_dlnd),
        jnp.asarray(out1_r.d_dln_t),
    )
    out2_j = eos.EosEvalOutput(
        jnp.asarray(out2_r.res),
        jnp.asarray(out2_r.d_dlnd),
        jnp.asarray(out2_r.d_dln_t),
    )
    with flexmesa.numerics_mode_context("strict"):
        for alfa in (0.15, 0.5, 0.85):
            br = eos.blend_eos_results(
                alfa, 0.6, -0.3, False, out1_r, out2_r, backend="rust"
            )
            bj = eos.blend_eos_results(
                jnp.float64(alfa),
                jnp.float64(0.6),
                jnp.float64(-0.3),
                False,
                out1_j,
                out2_j,
                backend="jax",
            )
            np.testing.assert_allclose(
                br.res, np.asarray(bj.res), rtol=1e-12, atol=1e-12
            )
            np.testing.assert_allclose(
                br.d_dlnd, np.asarray(bj.d_dlnd), rtol=1e-12, atol=1e-12
            )
            np.testing.assert_allclose(
                br.d_dln_t, np.asarray(bj.d_dln_t), rtol=1e-12, atol=1e-12
            )
