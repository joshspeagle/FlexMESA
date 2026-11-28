# SPDX-License-Identifier: GPL-3.0-or-later
"""Wave 5.1 verification tests for multi-species Burgers element diffusion,
convective premixing, predictive mixing, and white-dwarf crystallization
phase separation.
"""

from __future__ import annotations

import pathlib
import struct

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import flexmesa
from flexmesa import _rust, star

_DIFFUSION_ORACLES_TSV = (
    pathlib.Path(__file__).resolve().parent.parent
    / "benchmarks"
    / "fortran_mesa"
    / "diffusion_oracles"
    / "mesa_26_4_1_diffusion_oracles.tsv"
)


def _load_mesa_kernel_oracles() -> dict[tuple[str, str, int], float]:
    """Load Section 5 (`kernels`) IEEE-754 MESA 26.4.1 reference values."""
    out: dict[tuple[str, str, int], float] = {}
    with _DIFFUSION_ORACLES_TSV.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if (
                not stripped
                or stripped.startswith("#")
                or stripped.startswith("section\t")
            ):
                continue
            cols = stripped.split("\t")
            if cols[0] != "kernels":
                continue
            bits = int(cols[5], 16)
            val = struct.unpack(">d", bits.to_bytes(8, "big"))[0]
            out[(cols[1], cols[2], int(cols[3]))] = val
    return out


def test_burgers_and_caplan_ocp_consistency_and_flux() -> None:
    """Verify Rust Caplan OCP matches committed MESA 26.4.1 hex oracles (#156),
    Burgers velocities satisfy flux conservation, and frozen `#166` JAX
    prototypes stay finite and differentiable.

    `src/flexmesa/star/jax_diffusion_rot.py` is frozen under `#166` with its
    pre-`#156` placeholder Caplan/Burgers expressions, so Rust strict is
    checked against the committed MESA 26.4.1 reference table while the frozen
    JAX prototypes are checked for finiteness and reverse-mode autodiff only.
    """
    oracles = _load_mesa_kernel_oracles()
    with flexmesa.numerics_mode_context("strict"):
        for idx in (1, 2, 3):
            gam = oracles[("dstar_ocp", "Gamma", idx)]
            kap = oracles[("dstar_ocp", "kappa", idx)]
            d_mesa = oracles[("dstar_ocp", "Dstar", idx)]
            d_r = _rust.star_eval_caplan_ocp_dstar(gam, kap)
            assert abs(d_r - d_mesa) / d_mesa <= 1.0e-14
            d_j = float(
                star.eval_caplan_ocp_dstar_jax(
                    jnp.asarray(gam), jnp.asarray(kap)
                )
            )
            assert np.isfinite(d_j) and d_j > 0.0

        a = np.array([1.0, 4.0, 16.0], dtype=np.float64)
        z = np.array([1.0, 2.0, 8.0], dtype=np.float64)
        x = np.array([0.70, 0.28, 0.02], dtype=np.float64)
        rho = 10.0
        t = 1.2e7
        dlnt_dr = -1.5e-10
        g_grav = 2.74e4

        v_r, diff_r, ee_r = _rust.star_eval_burgers_velocities(
            a, z, x, rho, t, dlnt_dr, g_grav, None, "stanton_murillo"
        )
    # H1 floats (> 0), He4 and O16 sink (< 0), and sum(X_i * v_i) == 0
    assert v_r[0] > 0.0 and v_r[1] < 0.0 and v_r[2] < 0.0
    assert abs(float(np.dot(x, v_r))) <= 1.0e-12
    assert np.all(np.isfinite(diff_r)) and np.all(diff_r > 0.0)
    assert np.isfinite(ee_r)

    # Frozen #166 JAX prototype finiteness & autodiff check
    v_j, d_j_arr, ee_j = star.eval_burgers_velocities_jax(
        jnp.asarray(a),
        jnp.asarray(z),
        jnp.asarray(x),
        jnp.asarray(rho),
        jnp.asarray(t),
        jnp.asarray(dlnt_dr),
        jnp.asarray(g_grav),
    )
    assert np.all(np.isfinite(np.asarray(v_j)))
    assert np.all(np.isfinite(np.asarray(d_j_arr)))
    assert np.isfinite(float(ee_j))

    dv0_dg = float(
        jax.grad(
            lambda g: star.eval_burgers_velocities_jax(
                jnp.asarray(a),
                jnp.asarray(z),
                jnp.asarray(x),
                jnp.asarray(rho),
                jnp.asarray(t),
                jnp.asarray(dlnt_dr),
                g,
            )[0][0]
        )(jnp.asarray(g_grav))
    )
    assert np.isfinite(dv0_dg) and dv0_dg > 0.0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#295: Rust eval_caplan_ocp_dstar follows MESA 26.4.1 "
        "get_Dstar_OCP (#156); the frozen JAX twin keeps its earlier "
        "expression."
    ),
)
def test_caplan_ocp_dstar_jax_matches_rust() -> None:
    """Compare the JAX Caplan one-component-plasma coefficient with Rust.

    This is the comparison `test_burgers_and_caplan_ocp_consistency_and_flux`
    made before `#156`, with the same inputs and the same `1e-14` bound.
    """
    with flexmesa.numerics_mode_context("strict"):
        for gam, kap in [(5.0, 0.5), (20.0, 1.2), (100.0, 0.8)]:
            d_r = _rust.star_eval_caplan_ocp_dstar(gam, kap)
            d_j = float(
                star.eval_caplan_ocp_dstar_jax(
                    jnp.asarray(gam), jnp.asarray(kap)
                )
            )
            assert abs(d_r - d_j) / d_r <= 1.0e-14


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#295: Rust eval_burgers_velocities follows MESA 26.4.1 element "
        "diffusion (#156); the frozen JAX twin keeps its earlier "
        "expressions."
    ),
)
def test_burgers_velocities_jax_match_rust() -> None:
    """Compare JAX Burgers velocities, coefficients and field with Rust.

    This is the comparison `test_burgers_and_caplan_ocp_consistency_and_flux`
    made before `#156`, with the same inputs and the same `1e-12` bounds.
    """
    a = np.array([1.0, 4.0, 16.0], dtype=np.float64)
    z = np.array([1.0, 2.0, 8.0], dtype=np.float64)
    x = np.array([0.70, 0.28, 0.02], dtype=np.float64)
    rho = 10.0
    t = 1.2e7
    dlnt_dr = -1.5e-10
    g_grav = 2.74e4
    with flexmesa.numerics_mode_context("strict"):
        v_r, diff_r, ee_r = _rust.star_eval_burgers_velocities(
            a, z, x, rho, t, dlnt_dr, g_grav, None, "stanton_murillo"
        )
    v_j, d_j, ee_j = star.eval_burgers_velocities_jax(
        jnp.asarray(a),
        jnp.asarray(z),
        jnp.asarray(x),
        jnp.asarray(rho),
        jnp.asarray(t),
        jnp.asarray(dlnt_dr),
        jnp.asarray(g_grav),
    )
    np.testing.assert_allclose(v_r, np.asarray(v_j), rtol=1.0e-12, atol=1.0e-14)
    np.testing.assert_allclose(
        diff_r, np.asarray(d_j), rtol=1.0e-12, atol=1.0e-14
    )
    assert abs(ee_r - float(ee_j)) / abs(ee_r) <= 1.0e-12


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#295: Rust eval_phase_separation_shift follows the MESA 26.4.1 "
        "Blouin fits (#156); the frozen JAX twin keeps its earlier "
        "expression."
    ),
)
def test_phase_separation_shift_jax_matches_rust() -> None:
    """Compare the JAX phase-separation composition shift with Rust.

    This is the comparison
    `test_phase_separation_and_star_model_wave5_1_methods` made before
    `#156`, with the same inputs and the same `1e-14` bound.
    """
    for x_heavy in [0.1, 0.3, 0.5, 0.7, 0.85]:
        for is_one in [False, True]:
            dx_r = _rust.star_eval_phase_separation_shift(x_heavy, is_one)
            dx_j = float(
                star.eval_phase_separation_shift_jax(
                    jnp.asarray(x_heavy), is_one
                )
            )
            assert abs(dx_r - dx_j) <= 1.0e-14


def test_phase_separation_and_star_model_wave5_1_methods() -> None:
    """Verify Blouin et al. (2021a,b) phase separation shifts match committed
    MESA 26.4.1 hex oracles and StarModel Wave 5.1 methods conserve species
    mass to machine precision.
    """
    oracles = _load_mesa_kernel_oracles()
    for idx in (1, 2, 3, 4, 5):
        for case_id, is_one in [("blouin_xo", False), ("blouin_xne", True)]:
            x_heavy = oracles[(case_id, "X_in", idx)]
            dx_mesa = oracles[(case_id, "dX", idx)]
            dx_r = _rust.star_eval_phase_separation_shift(x_heavy, is_one)
            assert abs(dx_r - dx_mesa) <= 1.0e-14
            # Frozen #166 JAX prototype differs from MESA 26.4.1 Blouin fit;
            # verify finiteness and positivity only until #166 unfreezes.
            dx_j = float(
                star.eval_phase_separation_shift_jax(
                    jnp.asarray(x_heavy), is_one
                )
            )
            assert np.isfinite(dx_j) and dx_j > 0.0

    model = star.StarModel.from_zams(initial_mass=4.0, initial_z=0.02)
    model.eval_hydro_vars()
    dm = model.dm.copy()
    species_mass_before = dm @ model.xa

    # 1. Element diffusion conserves sum_j X_{j,k} = 1 and species mass
    v_gt, eps_wd = model.do_element_diffusion(
        dt=1.0e13, scheme="stanton_murillo"
    )
    assert v_gt.shape == (model.nz, model.species)
    assert eps_wd.shape == (model.nz,)
    np.testing.assert_allclose(np.sum(model.xa, axis=1), 1.0, atol=1.0e-13)
    species_mass_after = dm @ model.xa
    np.testing.assert_allclose(
        species_mass_after, species_mass_before, rtol=1.0e-12, atol=1.0e-12
    )

    # 2. Rotational mixing info
    rot_arr = model.set_rotation_mixing_info()
    assert rot_arr.shape == (model.nz, 9)

    # 3. Predictive mixing and convective premixing on 4 Msun ZAMS model.
    #    Regression snapshot: on this homogeneous model both calls change
    #    nothing and return 0. Neither value is a MESA reference.
    n_pred = model.add_predictive_mixing(
        ingested_mass_limit_coeff=0.1, superad_thresh=0.0
    )
    assert n_pred == 0
    n_premix = model.do_conv_premix(max_iters=2)
    assert n_premix == 0

    # 4. Crystallization phase separation conserves C12 + O16 mass
    eps_ps = model.do_phase_separation(dt=1.0e11, gamma_crystallization=175.0)
    assert eps_ps.shape == (model.nz,)
    np.testing.assert_allclose(
        dm @ model.xa, species_mass_before, rtol=1.0e-12, atol=1.0e-12
    )
