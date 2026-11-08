# SPDX-License-Identifier: LGPL-3.0-or-later
"""Wave 4.2 verification tests for `flexmesa.star` Henyey equations and solver.

Tests Safe Rust vs. JAX residual parity (`eval_equations` vs.
`eval_hydro_residuals_jax`), analytical 3-cell stencil block-tridiagonal
Jacobians (`lblk`, `dblk`, `ublk`), differentiable JAX linear Henyey solves
(`solver_mode="pcr"` and `"rust_ffi"`), and damped Newton-Raphson convergence.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import constants, star


def _make_pre_ms_model() -> star.StarModel:
    """Create and initialize a 1 Msun coarse pre-MS model for solver tests."""
    model = star.StarModel.from_pre_ms(
        initial_mass=1.0,
        initial_z=0.02,
        d_log10_p=0.15,
    )
    model.dt = 10.0 * float(constants.SECYER)
    model.eval_hydro_vars()
    model.save_start_values()
    return model


def test_eval_equations_rust_vs_jax_residual_parity() -> None:
    """Verify Rust and JAX hydrostatic Henyey residuals match to <= 1e-13."""
    model = star.StarModel.from_zams(initial_mass=4.0, initial_z=0.02)
    model.dt = 100.0 * float(constants.SECYER)
    model.eval_hydro_vars()
    model.save_start_values()

    # Perturb lnd and ln_t after save_start_values() so de_dt and pdv_dt
    # are non-zero in both Rust and JAX.
    model._rust_model.set_primary_state(
        model.lnd + 1.0e-4,
        model.ln_t + 1.0e-4,
        model.ln_r,
        model.lum,
        model.dq,
        model.xa,
    )

    sys = model.eval_equations(include_chem=False)
    equ_jax = np.asarray(model.eval_hydro_residuals_jax())

    assert sys.equ.shape == (model.nz, 4)
    assert equ_jax.shape == (model.nz, 4)
    np.testing.assert_allclose(sys.equ, equ_jax, rtol=1.0e-13, atol=1.0e-13)


def test_eval_hydro_residuals_jax_is_differentiable_wrt_primary_vars() -> None:
    """Verify `eval_hydro_residuals_jax` supports exact reverse-mode AD."""
    model = _make_pre_ms_model()

    dq = jnp.asarray(model.dq)
    dm = jnp.asarray(model.dm)
    dm_bar = jnp.asarray(model.dm_bar)
    m_grav = jnp.asarray(model.m)
    lnd0 = jnp.asarray(model.lnd)
    lnt0 = jnp.asarray(model.ln_t)
    lnr0 = jnp.asarray(model.ln_r)
    lum0 = jnp.asarray(model.lum)
    ln_peos = jnp.asarray(model.ln_peos)
    ln_e = jnp.log(jnp.asarray(model.energy))
    energy_start = jnp.asarray(model.energy_start)
    gradt = jnp.asarray(model.gradt)
    eps_nuc = jnp.asarray(model.eps_nuc)
    non_nuc_neu = jnp.asarray(model.non_nuc_neu)
    eps_mdot = jnp.zeros_like(lnd0)
    lnp_s = float(model.ln_peos_surf)
    lnt_s = float(model.ln_t_surf)
    dt = float(model.dt)

    def loss_fn(lnd: jax.Array, ln_r: jax.Array) -> jax.Array:
        equ = star.eval_hydro_residuals_jax(
            dq=dq,
            dm=dm,
            dm_bar=dm_bar,
            m_grav=m_grav,
            lnd=lnd,
            ln_t=lnt0,
            ln_r=ln_r,
            lum=lum0,
            lnd_start=lnd0,
            ln_peos=ln_peos,
            ln_e=ln_e,
            energy_start=energy_start,
            gradt=gradt,
            eps_nuc=eps_nuc,
            non_nuc_neu=non_nuc_neu,
            eps_mdot=eps_mdot,
            ln_peos_surf=lnp_s,
            ln_t_surf=lnt_s,
            dt=dt,
        )
        return jnp.sum(equ * equ)

    g_lnd, g_lnr = jax.grad(loss_fn, argnums=(0, 1))(lnd0, lnr0)
    assert jnp.all(jnp.isfinite(g_lnd))
    assert jnp.all(jnp.isfinite(g_lnr))


def test_solve_henyey_linear_jax_pcr_and_rust_ffi_match_and_differentiate() -> (
    None
):
    """Verify JAX `pcr` and `rust_ffi` linear Henyey solves and `custom_vjp`."""
    model = _make_pre_ms_model()

    sys = model.eval_equations(include_chem=False)
    lblk = jnp.asarray(sys.lblk)
    dblk = jnp.asarray(sys.dblk)
    ublk = jnp.asarray(sys.ublk)
    equ = jnp.asarray(sys.equ)

    dx_pcr = np.asarray(
        star.solve_henyey_linear_jax(lblk, dblk, ublk, equ, solver_mode="pcr")
    )
    dx_ffi = np.asarray(
        star.solve_henyey_linear_jax(
            lblk, dblk, ublk, equ, solver_mode="rust_ffi"
        )
    )
    np.testing.assert_allclose(dx_pcr, dx_ffi, rtol=1.0e-7, atol=1.0e-7)

    def scalar_obj(equ_in: jax.Array) -> jax.Array:
        dx = star.solve_henyey_linear_jax(
            lblk, dblk, ublk, equ_in, solver_mode="rust_ffi"
        )
        return jnp.sum(dx * dx)

    grad_ad = np.asarray(jax.grad(scalar_obj)(equ))
    eps = 1.0e-6
    equ_np = np.asarray(equ)
    e_pert = np.zeros_like(equ_np)
    e_pert[model.nz // 2, 2] = eps
    fd = (
        float(scalar_obj(jnp.asarray(equ_np + e_pert)))
        - float(scalar_obj(jnp.asarray(equ_np - e_pert)))
    ) / (2.0 * eps)
    np.testing.assert_allclose(
        grad_ad[model.nz // 2, 2], fd, rtol=1.0e-5, atol=1.0e-6
    )


def test_solve_henyey_damped_newton_converges_for_all_modes() -> None:
    """Verify `solve_henyey` converges for bcyclic, Thomas, and ZAMS chem."""
    model = _make_pre_ms_model()

    model_thomas = star.StarModel.from_photo_bytes(model.to_photo_bytes())
    model_thomas.dt = model.dt
    model_thomas.eval_hydro_vars()
    model_thomas.save_start_values()

    rep_bcyclic = model.solve_henyey(include_chem=False, use_bcyclic=True)
    assert rep_bcyclic.converged
    assert rep_bcyclic.iterations <= 10
    assert rep_bcyclic.residual_norm <= 1.0e-6

    rep_thomas = model_thomas.solve_henyey(
        include_chem=False, use_bcyclic=False
    )
    assert rep_thomas.converged
    np.testing.assert_allclose(
        model.lnd, model_thomas.lnd, rtol=1.0e-6, atol=1.0e-6
    )
    np.testing.assert_allclose(
        model.ln_t, model_thomas.ln_t, rtol=1.0e-6, atol=1.0e-6
    )

    model.save_start_values()
    rep_chem = model.solve_henyey(include_chem=True, use_bcyclic=True)
    assert rep_chem.converged
    assert np.all(np.isfinite(model.eps_grav))

    # Also verify coupled hydro + chem convergence on the 4 Msun ZAMS model
    zams = star.StarModel.from_zams(initial_mass=4.0, initial_z=0.02)
    zams.eval_hydro_vars()
    zams.save_start_values()
    rep_zams = zams.solve_henyey(include_chem=True, use_bcyclic=True)
    assert rep_zams.converged
    assert np.allclose(np.sum(zams.xa, axis=1), 1.0, atol=1.0e-12)
