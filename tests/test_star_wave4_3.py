# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for Wave 4.3 convective boundaries, overshooting, structure-burn-mix,
rotational angular momentum diffusion, and stellar winds (`flexmesa.star`).
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.test_util as jtu
import numpy as np

from flexmesa import constants as const
from flexmesa import star


def test_convective_boundaries_overshoot_and_jax_mixing_parity() -> None:
    """Verify boundary detection, overshooting, and `sig`/`dxdt_mix` parity."""
    zams = star.StarModel.from_zams(initial_mass=4.0, initial_z=0.02)
    dt = 1.0e8

    n_bdy = zams.update_mixing(
        dt=dt, overshoot_f=0.014, overshoot_f0=0.004, scheme="exponential"
    )
    assert n_bdy >= 1
    assert np.any(zams.mixing_type == 2)  # OvershootMixing == 2

    sig_rust = zams.sig
    sig_jax = np.asarray(
        star.eval_convection_sigmas_jax(
            jnp.asarray(zams.r),
            jnp.asarray(zams.rho),
            jnp.asarray(zams.dq),
            jnp.asarray(zams.d_mix),
            float(zams.mstar),
            dt,
        )
    )
    np.testing.assert_allclose(sig_jax, sig_rust, rtol=1.0e-12, atol=1.0e-12)

    # Verify save_start_values preserves computed `sig` rather than clobbering.
    zams.save_start_values()
    np.testing.assert_allclose(zams.sig, sig_rust, rtol=1.0e-14, atol=1.0e-14)

    # Create a slight composition gradient to test non-zero diffusive fluxes.
    xa_mod = zams.xa.copy()
    xa_mod[:, 0] += 1.0e-4 * np.linspace(-1.0, 1.0, zams.nz)
    xa_mod[:, 2] -= 1.0e-4 * np.linspace(-1.0, 1.0, zams.nz)
    zams._rust_model.set_primary_state(
        zams.lnd, zams.ln_t, zams.ln_r, zams.lum, zams.dq, xa_mod
    )

    dxdt_rust = zams.eval_dxdt_mix()
    dxdt_jax = np.asarray(
        star.eval_dxdt_mix_jax(
            jnp.asarray(zams.xa),
            jnp.asarray(zams.sig),
            jnp.asarray(zams.dm),
        )
    )
    np.testing.assert_allclose(dxdt_jax, dxdt_rust, rtol=1.0e-12, atol=1.0e-20)

    # Total species mass rate `sum_k dm[k] * dxdt_mix[k, j]` must vanish.
    mass_rates = np.sum(zams.dm[:, None] * dxdt_rust, axis=0)
    scale = np.maximum(
        np.sum(np.abs(zams.dm[:, None] * dxdt_rust), axis=0), 1.0
    )
    np.testing.assert_allclose(mass_rates / scale, 0.0, atol=1.0e-12)

    # Verify autodiff gradients on JAX mixing kernels.
    r_sub = jnp.asarray(zams.r[-12:])
    rho_sub = jnp.asarray(zams.rho[-12:])
    dq_sub = jnp.asarray(zams.dq[-12:])
    dmix_sub = jnp.asarray(np.maximum(zams.d_mix[-12:], 1.0e12))
    jtu.check_grads(  # type: ignore[no-untyped-call]
        lambda rho_v, d_v: jnp.sum(
            star.eval_convection_sigmas_jax(
                r_sub, rho_v, dq_sub, d_v, float(zams.mstar), dt
            )
        ),
        (rho_sub, dmix_sub),
        order=1,
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    xa_sub = jnp.asarray(zams.xa[-12:, :3])
    sig_sub = jnp.asarray(zams.sig[-12:]).at[0].set(0.0)
    dm_sub = jnp.asarray(zams.dm[-12:])
    jtu.check_grads(  # type: ignore[no-untyped-call]
        lambda x_v, s_v: jnp.sum(star.eval_dxdt_mix_jax(x_v, s_v, dm_sub) ** 2),
        (xa_sub, sig_sub),
        order=1,
        rtol=1.0e-5,
        atol=1.0e-5,
    )

    # Step overshooting also populates overshoot zones cleanly.
    n_bdy_step = zams.update_mixing(
        dt=dt, overshoot_f=0.02, overshoot_f0=0.005, scheme="step"
    )
    assert n_bdy_step >= 1


def test_implicit_diffusion_and_omega_mix_conservation() -> None:
    """Verify conservation in `solve_diffusion` and `run_omega_mix`."""
    zams = star.StarModel.from_zams(initial_mass=4.0, initial_z=0.02)
    dt = 1.0e9
    zams.update_mixing(dt=dt, overshoot_f=0.014, overshoot_f0=0.004)

    xa_init = zams.xa.copy()
    xa_init[-300:, 0] -= 0.05
    xa_init[-300:, 2] += 0.05
    zams._rust_model.set_primary_state(
        zams.lnd, zams.ln_t, zams.ln_r, zams.lum, zams.dq, xa_init
    )
    species_mass_before = (
        np.sum(zams.dm[:, None] * zams.xa, axis=0) / zams.mstar
    )

    zams.solve_diffusion(dt)
    species_mass_after = np.sum(zams.dm[:, None] * zams.xa, axis=0) / zams.mstar
    np.testing.assert_allclose(
        species_mass_after, species_mass_before, rtol=1.0e-12, atol=1.0e-14
    )
    np.testing.assert_allclose(np.sum(zams.xa, axis=1), 1.0, atol=1.0e-14)

    # Rotational angular momentum diffusion (`run_omega_mix`)
    omega_init = np.linspace(1.0e-5, 5.0e-5, zams.nz)
    nu_omega = np.full(zams.nz, 1.0e12)
    i_rot = (2.0 / 3.0) * (zams.r**2)
    j_tot_before = float(np.sum(zams.dm_bar * i_rot * omega_init))

    zams.run_omega_mix(
        dt=1.0e7, omega=omega_init, nu_omega=nu_omega, n_substeps=4
    )
    j_tot_after = float(np.sum(zams.dm_bar * zams.j_rot))
    np.testing.assert_allclose(j_tot_after, j_tot_before, rtol=1.0e-12)
    assert float(np.ptp(zams.omega)) < float(np.ptp(omega_init))


def test_step_struct_burn_mix_coupled_and_op_split() -> None:
    """Verify `step_struct_burn_mix` converges and preserves overshooting."""
    zams = star.StarModel.from_zams(initial_mass=4.0, initial_z=0.02)
    rep_zams = zams.step_struct_burn_mix(
        dt=zams.dt, op_split_burn=True, use_bcyclic=True, overshoot_f=0.014
    )
    assert rep_zams.converged
    assert np.any(zams.mixing_type == 2)  # OvershootMixing preserved post-solve
    np.testing.assert_allclose(np.sum(zams.xa, axis=1), 1.0, atol=1.0e-12)

    pre = star.StarModel.from_pre_ms(
        initial_mass=4.0, initial_z=0.02, d_log10_p=0.15
    )
    dt = 10.0 * float(const.SECYER)

    rep_coupled = pre.step_struct_burn_mix(
        dt=dt, op_split_burn=False, use_bcyclic=True, overshoot_f=0.014
    )
    assert rep_coupled.converged
    assert np.any(pre.sig > 0.0)
    np.testing.assert_allclose(np.sum(pre.xa, axis=1), 1.0, atol=1.0e-12)

    rep_split = pre.step_struct_burn_mix(
        dt=dt, op_split_burn=True, use_bcyclic=True, overshoot_f=0.014
    )
    assert rep_split.converged
    assert np.any(pre.sig > 0.0)
    np.testing.assert_allclose(np.sum(pre.xa, axis=1), 1.0, atol=1.0e-12)


def test_stellar_winds_and_3_region_mass_adjustment_parity() -> None:
    """Verify Rust and JAX wind schemes, gradients, and `do_adjust_mass`."""
    zams = star.StarModel.from_zams(initial_mass=4.0, initial_z=0.02)
    winds_jax = zams.eval_winds_jax(eta=1.0)
    secyer = float(const.SECYER)
    m_sun = float(const.M_SUN)

    for scheme, jax_key in [
        ("Reimers", "w_reimers"),
        ("Blocker", "w_blocker"),
        ("Vink", "w_vink"),
        ("Nieuwenhuijzen", "w_nieuwenhuijzen"),
        ("de Jager", "w_de_jager"),
        ("van Loon", "w_van_loon"),
        ("Dutch", "w_dutch"),
    ]:
        mdot_rust = zams.apply_wind_and_adjust_mass(
            scheme=scheme, eta=1.0, dt=0.0
        )
        w_rust = -mdot_rust * secyer / m_sun
        w_jax = float(winds_jax[jax_key])
        np.testing.assert_allclose(w_rust, w_jax, rtol=1.0e-11)

    # Verify autodiff gradients on `eval_winds_jax` via `check_grads`.
    jtu.check_grads(  # type: ignore[no-untyped-call]
        lambda l_s, m_s, t_s: star.eval_winds_jax(l_s, m_s, t_s)["w_dutch"],
        (
            jnp.asarray(float(zams.lum[0])),
            jnp.asarray(float(zams.mstar)),
            jnp.asarray(float(zams.teff)),
        ),
        order=1,
        rtol=1.0e-4,
        atol=1.0e-8,
    )

    # Impose a composition step inside Region II (`0.95 < q < 0.98`) while
    # keeping the lost outer shell (`q > 0.98`) uniform so species mass loss
    # has an exact analytical oracle `delta_m * X_surf`.
    xa_grad = zams.xa.copy()
    x_surf_h1 = float(xa_grad[0, 0])
    for k in range(zams.nz):
        if 0.95 < zams.q[k] <= 0.98:
            xa_grad[k, 0] -= 0.10
            xa_grad[k, 2] += 0.10
    zams._rust_model.set_primary_state(
        zams.lnd, zams.ln_t, zams.ln_r, zams.lum, zams.dq, xa_grad
    )

    m_before = zams.mstar
    m_h1_before = float(np.sum(zams.dm * zams.xa[:, 0]))
    dt_wind = 1.0e6 * secyer
    mdot = zams.apply_wind_and_adjust_mass(scheme="Dutch", eta=10.0, dt=dt_wind)
    assert mdot < 0.0
    delta_m = zams.mstar - m_before
    m_h1_after = float(np.sum(zams.dm * zams.xa[:, 0]))
    np.testing.assert_allclose(delta_m, mdot * dt_wind, rtol=1.0e-11)
    np.testing.assert_allclose(
        m_h1_after - m_h1_before, delta_m * x_surf_h1, rtol=1.0e-11
    )
    np.testing.assert_allclose(float(np.sum(zams.dq)), 1.0, atol=1.0e-14)
    np.testing.assert_allclose(zams.xq, 1.0 - zams.q, atol=1.0e-14)
    np.testing.assert_allclose(np.sum(zams.xa, axis=1), 1.0, atol=1.0e-14)
