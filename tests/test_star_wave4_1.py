# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for Wave 4.1 (`flexmesa.star` state, ZAMS/pre-MS, and microphysics)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import constants, star


def test_star_zams_4msun_star_test_initialization_and_microphysics() -> None:
    """Verify 4.0 M_sun ZAMS (`star/test`) interpolation and microphysics."""
    assert abs(star.yrs_for_init_timestep(4.0) - 3125.0) <= 1.0e-12

    model = star.StarModel.from_zams(initial_mass=4.0, initial_z=0.02)
    assert model.nz == 2464
    assert model.species == 8
    assert model.iso_names == [
        "h1",
        "he3",
        "he4",
        "c12",
        "n14",
        "o16",
        "ne20",
        "mg24",
    ]
    np.testing.assert_allclose(model.mstar, 4.0 * constants.M_SUN, rtol=1.0e-12)
    np.testing.assert_allclose(model.dq.sum(), 1.0, rtol=1.0e-14)
    np.testing.assert_allclose(model.xa.sum(axis=1), 1.0, rtol=1.0e-13)

    # Central zone thermodynamic and nuclear properties
    assert model.peos[-1] > 1.0e16
    assert model.t[-1] > 2.0e7
    assert model.eps_nuc[-1] > 100.0
    assert 1.0e4 < model.teff < 2.5e4
    assert np.isfinite(model.ln_peos_surf)
    assert np.isfinite(model.ln_t_surf)
    assert int(np.sum(model.mixing_type[-200:] == 1)) > 50

    # Off-grid ZAMS mass (2.5 M_sun) exercises `interp_q` log-mass interpolation
    model_2p5 = star.StarModel.from_zams(
        initial_mass=2.5, initial_z=0.02, eval_hydro=False
    )
    np.testing.assert_allclose(
        model_2p5.mstar, 2.5 * constants.M_SUN, rtol=1.0e-12
    )
    np.testing.assert_allclose(model_2p5.dq.sum(), 1.0, rtol=1.0e-14)
    np.testing.assert_allclose(model_2p5.xa.sum(axis=1), 1.0, rtol=1.0e-13)


def test_star_mod_and_photo_roundtrip_parity() -> None:
    """Verify ASCII `.mod` and binary `.photo` serialization round-trips."""
    model = star.StarModel.from_zams(
        initial_mass=1.0, initial_z=0.02, eval_hydro=False
    )

    mod_text = model.to_mod_str()
    from_mod = star.StarModel.from_mod_str(mod_text, eval_hydro=False)
    assert from_mod.nz == model.nz
    assert from_mod.species == model.species
    np.testing.assert_allclose(from_mod.lnd, model.lnd, rtol=1.0e-14)
    np.testing.assert_allclose(from_mod.ln_t, model.ln_t, rtol=1.0e-14)
    np.testing.assert_allclose(from_mod.ln_r, model.ln_r, rtol=1.0e-14)
    np.testing.assert_allclose(from_mod.lum, model.lum, rtol=1.0e-14)
    np.testing.assert_allclose(
        from_mod.xa, model.xa, rtol=1.0e-14, atol=1.0e-15
    )

    photo_bytes = model.to_photo_bytes()
    from_photo = star.StarModel.from_photo_bytes(photo_bytes, eval_hydro=False)
    np.testing.assert_array_equal(from_photo.lnd, model.lnd)
    np.testing.assert_array_equal(from_photo.ln_t, model.ln_t)
    np.testing.assert_array_equal(from_photo.ln_r, model.ln_r)
    np.testing.assert_array_equal(from_photo.lum, model.lum)
    np.testing.assert_array_equal(from_photo.dq, model.dq)
    np.testing.assert_array_equal(from_photo.xa, model.xa)


def test_star_jax_grid_and_face_mlt_match_rust_and_differentiate() -> None:
    """Verify JAX staggered grid, optical depth, and face MLT match Rust."""
    model = star.StarModel.from_zams(initial_mass=4.0, initial_z=0.02)

    grid_jax = star.compute_staggered_grid_jax(
        dq=jnp.asarray(model.dq),
        ln_r=jnp.asarray(model.ln_r),
        mstar=model.mstar,
    )
    np.testing.assert_allclose(
        grid_jax["q"], model.q, rtol=1.0e-12, atol=1.0e-14
    )
    np.testing.assert_allclose(
        grid_jax["m"], model.m, rtol=1.0e-12, atol=1.0e-14 * model.mstar
    )
    np.testing.assert_allclose(grid_jax["dm"], model.dm, rtol=1.0e-13)
    np.testing.assert_allclose(grid_jax["dm_bar"], model.dm_bar, rtol=1.0e-13)
    np.testing.assert_allclose(grid_jax["rmid"], model.rmid, rtol=1.0e-13)
    np.testing.assert_allclose(
        grid_jax["grav"], model.grav, rtol=1.0e-7, atol=1.0e-10
    )

    tau_jax = star.eval_optical_depth_jax(
        dm=jnp.asarray(model.dm),
        rmid=jnp.asarray(model.rmid),
        opacity=jnp.asarray(model.opacity),
        tau_surf=float(model.tau[0]),
    )
    np.testing.assert_allclose(tau_jax, model.tau, rtol=1.0e-12)

    mlt_jax = model.eval_face_mlt_jax()
    np.testing.assert_allclose(
        mlt_jax["scale_height"], model.scale_height, rtol=1.0e-12
    )
    np.testing.assert_allclose(mlt_jax["gradr"], model.gradr, rtol=1.0e-12)
    np.testing.assert_allclose(mlt_jax["gradl"], model.gradl, rtol=1.0e-12)
    np.testing.assert_allclose(
        mlt_jax["brunt_b"], model.brunt_b, rtol=1.0e-11, atol=1.0e-14
    )
    # Interior zones match closely between JAX and Safe Rust
    np.testing.assert_allclose(
        mlt_jax["gradt"][200:], model.gradt[200:], rtol=1.0e-6
    )

    # Verify JAX forward-mode AD through `eval_face_mlt_and_brunt_jax`
    sl = slice(model.nz - 128, model.nz - 112)
    dq_s = jnp.asarray(model.dq[sl])
    lnr_s = jnp.asarray(model.ln_r[sl])
    lum_s = jnp.asarray(model.lum[sl])
    m_s = jnp.asarray(model.m[sl])
    rho_s = jnp.asarray(model.rho[sl])
    t_s = jnp.asarray(model.t[sl])
    peos_s = jnp.asarray(model.peos[sl])
    lnp_s = jnp.asarray(model.ln_peos[sl])
    grada_s = jnp.asarray(model.grada[sl])
    cp_s = jnp.asarray(model.cp[sl])
    chir_s = jnp.asarray(model.chi_rho[sl])
    chit_s = jnp.asarray(model.chi_t[sl])
    mu_s = jnp.asarray(model.mu[sl])
    kap_s = jnp.asarray(model.opacity[sl])

    def loss_fn(ln_r_in: jax.Array) -> jax.Array:
        out = star.eval_face_mlt_and_brunt_jax(
            dq_s,
            ln_r_in,
            lum_s,
            m_s,
            rho_s,
            t_s,
            peos_s,
            lnp_s,
            grada_s,
            cp_s,
            chir_s,
            chit_s,
            mu_s,
            kap_s,
        )
        return jnp.sum(out["gradt"] + out["scale_height"] / 1.0e10)

    grad_lnr = jax.grad(loss_fn)(lnr_s)
    assert jnp.all(jnp.isfinite(grad_lnr))
    assert float(jnp.linalg.norm(grad_lnr)) > 0.0

    # Central finite-difference verification of `jax.grad(loss_fn)` at zone 0
    eps = 1.0e-6
    e0 = jnp.zeros_like(lnr_s).at[0].set(eps)
    fd_0 = float((loss_fn(lnr_s + e0) - loss_fn(lnr_s - e0)) / (2.0 * eps))
    np.testing.assert_allclose(float(grad_lnr[0]), fd_0, rtol=1.0e-5)


def test_star_pre_ms_polytrope_builder() -> None:
    """Verify 1.0 M_sun n=1.5 pre-MS polytrope construction."""
    model = star.StarModel.from_pre_ms(
        initial_mass=1.0, initial_z=0.02, t_c=9.0e5, d_log10_p=0.04
    )
    assert model.nz > 50
    assert model.t[-1] > 5.0e5
    assert model.peos[-1] > model.peos[0]
    state = model.to_jax_state()
    assert state.lnd.shape == (model.nz,)
