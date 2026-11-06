# SPDX-License-Identifier: LGPL-3.0-or-later
"""Verification tests for Wave 3a.1 (`flexmesa.eos`: blend, dfermi, ideal)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flexmesa import constants, eos


def test_eos_basic_result_names_match_upstream_mesa_schema() -> None:
    """Verify the 26 canonical result channel names and indices."""
    assert len(eos.EOS_BASIC_RESULT_NAMES) == eos.NUM_EOS_BASIC_RESULTS == 26
    assert eos.EOS_BASIC_RESULT_NAMES[eos.I_LN_PGAS] == "lnPgas"
    assert eos.EOS_BASIC_RESULT_NAMES[eos.I_LN_E] == "lnE"
    assert eos.EOS_BASIC_RESULT_NAMES[eos.I_LN_S] == "lnS"
    assert eos.EOS_BASIC_RESULT_NAMES[eos.I_FRAC_IDEAL] == "ideal"


def test_eos_blend_rust_and_jax_match_golden_outputs() -> None:
    """Verify polygonal containment and distance across Rust and JAX."""
    coords = np.array(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0]],
        dtype=np.float64,
    )
    assert eos.is_contained(coords, 0.0, 1.0) is True
    assert eos.is_contained(coords, -1.0, 1.0) is False
    assert eos.is_contained(coords, 0.0, 0.0) is True
    assert eos.is_contained(coords, -2.0, 0.0) is False
    assert eos.is_contained(coords, -1.0, 3.0) is False

    for x, y, exp_d in [
        (0.0, 1.0, 1.000000013351432e-10),
        (-1.0, 1.0, 1.000000013351432e-10),
        (0.0, 0.0, 1.0),
        (-2.0, 0.0, 1.0),
        (-1.0, 3.0, 2.0),
    ]:
        d_r, _, _ = eos.min_distance_to_polygon(coords, x, y, backend="rust")
        d_j, _, _ = eos.min_distance_to_polygon(coords, x, y, backend="jax")
        assert abs(float(d_r) - exp_d) < 1.0e-15
        assert abs(float(d_j) - exp_d) < 1.0e-15

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


def test_quintic_blend_rust_and_jax_parity() -> None:
    """Verify quintic blend polynomial and derivative across Rust and JAX."""
    alphas = np.linspace(-0.2, 1.2, 15, dtype=np.float64)
    for a in alphas:
        s_r, ds_r = eos.quintic_blend(float(a), backend="rust")
        s_j, ds_j = eos.quintic_blend(jnp.float64(a), backend="jax")
        assert abs(float(s_r) - float(s_j)) < 1.0e-15
        assert abs(float(ds_r) - float(ds_j)) < 1.0e-15


def test_dfermi_rust_and_jax_match_gfortran_oracle() -> None:
    """Verify dfermi on Rust and JAX against the gfortran oracle."""
    eta = 1.46722890948893
    temp = 11327678.5183021
    theta = (constants.KERG * temp) / (constants.ME * (constants.CLIGHT**2))

    fd_r, deta_r, dtheta_r = eos.dfermi(-0.5, eta, theta, backend="rust")
    fd_j, deta_j, dtheta_j = eos.dfermi(-0.5, eta, theta, backend="jax")
    assert abs(float(fd_r) - 2.1896786848098353) < 1.0e-14
    assert abs(float(fd_j) - 2.1896786848098353) < 1.0e-14
    assert abs(float(deta_r) - float(deta_j)) < 1.0e-14
    assert abs(float(dtheta_r) - float(dtheta_j)) < 1.0e-14

    # Batch zone evaluation
    etas = np.array([eta, 0.5, 5.0], dtype=np.float64)
    thetas = np.array([theta, 1.0e-3, 1.0e-2], dtype=np.float64)
    fd_zr, deta_zr, dtheta_zr = eos.dfermi(1.5, etas, thetas, backend="rust")
    fd_zj, deta_zj, dtheta_zj = eos.dfermi(1.5, etas, thetas, backend="jax")
    np.testing.assert_allclose(fd_zr, np.asarray(fd_zj), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(
        deta_zr, np.asarray(deta_zj), rtol=1e-14, atol=1e-14
    )
    np.testing.assert_allclose(
        dtheta_zr, np.asarray(dtheta_zj), rtol=1e-14, atol=1e-14
    )


def test_ion_offset_and_ideal_eos_rust_jax_parity() -> None:
    """Verify ionization energy offset and ideal EOS across Rust and JAX."""
    xa = np.array([0.70, 0.28, 0.02], dtype=np.float64)
    z = [1, 2, 6]
    a = np.array([1.0, 4.0, 12.0], dtype=np.float64)
    weights = np.array([1.0078250322, 4.0026032541, 12.0], dtype=np.float64)

    off_r = eos.compute_ion_offset_from_za(xa, z, a, backend="rust")
    off_j = eos.compute_ion_offset_from_za(xa, z, a, backend="jax")
    assert abs(float(off_r) - float(off_j)) / float(off_r) < 1.0e-15

    abar = float(1.0 / np.sum(xa / weights))
    zbar = float(abar * np.sum(xa * np.array(z, dtype=np.float64) / weights))
    temp = 1.0e5
    rho = 1.0e-2

    out_r = eos.eval_ideal_eos_from_weights(
        temp, rho, abar, zbar, xa, weights, a, backend="rust"
    )
    out_j = eos.eval_ideal_eos_from_weights(
        temp, rho, abar, zbar, xa, weights, a, backend="jax"
    )
    np.testing.assert_allclose(
        out_r.res, np.asarray(out_j.res), rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        out_r.d_dlnd, np.asarray(out_j.d_dlnd), rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        out_r.d_dln_t, np.asarray(out_j.d_dln_t), rtol=1e-10, atol=1e-10
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

    # Verify batch eval_ideal_eos_zones and blend_eos_results parity
    t_zones = np.array([1.0e5, 2.0e5], dtype=np.float64)
    r_zones = np.array([1.0e-2, 5.0e-2], dtype=np.float64)
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

    b_r = eos.blend_eos_results(
        0.35, 0.8, -0.4, False, out_r, out_r, backend="rust"
    )
    b_j = eos.blend_eos_results(
        0.35, 0.8, -0.4, False, out_j, out_j, backend="jax"
    )
    np.testing.assert_allclose(
        b_r.res, np.asarray(b_j.res), rtol=1e-10, atol=1e-10
    )
