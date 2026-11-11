# SPDX-License-Identifier: LGPL-3.0-or-later
"""Verification suite for `flexmesa.star` Wave 4.4 (`#71`): adaptive remeshing
(`adjust_mesh`), Söderlind & Wang (2006) H211 timestep control
(`pick_next_timestep`), end-to-end `star/test` evolution (`evolve`), and
`history.data` / `profile.data` output.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import test_util

from flexmesa import constants as const
from flexmesa import star


def test_adaptive_remeshing_and_mesh_gval_jax_parity() -> None:
    """Verify `eval_mesh_gval_jax`, autodiff, and conservative `adjust_mesh`."""
    pre = star.StarModel.from_pre_ms(
        initial_mass=4.0, initial_z=0.02, d_log10_p=0.15
    )
    pre.set_evolve_controls(mesh_delta_coeff=0.5, varcontrol_target=1.0e-3)

    gval = np.asarray(pre.eval_mesh_gval_jax(mesh_delta_coeff=0.5))
    expected_gval = 0.5 * np.maximum(
        30.0 * np.abs(np.diff(pre.ln_peos)) / np.log(10.0),
        110.0 * np.abs(np.diff(pre.ln_t)) / np.log(10.0),
    )
    assert gval.shape == (pre.nz,)
    np.testing.assert_allclose(gval[:-1], expected_gval, rtol=1.0e-14)
    assert gval[-1] == pytest.approx(expected_gval[-1], rel=1.0e-14)
    assert np.any(gval > 1.0)

    test_util.check_grads(
        lambda lnp, lnt: jnp.sum(
            star.eval_mesh_gval_jax(lnp, lnt, mesh_delta_coeff=0.5) ** 2
        ),
        (jnp.asarray(pre.ln_peos), jnp.asarray(pre.ln_t)),
        order=1,
        rtol=1.0e-5,
        atol=1.0e-7,
    )

    nz_before = pre.nz
    species_mass_before = np.sum(pre.dm[:, None] * pre.xa, axis=0) / pre.mstar
    energy_before = float(np.sum(pre.dm * pre.energy))
    n_splits, n_merges = pre.adjust_mesh()
    assert n_splits > 0
    assert pre.nz == nz_before + n_splits - n_merges

    species_mass_after = np.sum(pre.dm[:, None] * pre.xa, axis=0) / pre.mstar
    energy_after = float(np.sum(pre.dm * pre.energy))
    np.testing.assert_allclose(
        species_mass_after, species_mass_before, rtol=1.0e-12, atol=1.0e-14
    )
    assert energy_after == pytest.approx(energy_before, rel=2.0e-3)
    np.testing.assert_allclose(float(np.sum(pre.dq)), 1.0, atol=1.0e-14)
    np.testing.assert_allclose(np.sum(pre.xa, axis=1), 1.0, atol=1.0e-14)


def test_varcontrol_and_h211_timestep_filter_rust_vs_jax() -> None:
    """Verify `varcontrol` and Söderlind & Wang (2006) H211 filter parity."""
    pre = star.StarModel.from_pre_ms(
        initial_mass=4.0, initial_z=0.02, d_log10_p=0.15
    )
    pre.set_evolve_controls(mesh_delta_coeff=1.0, varcontrol_target=1.0e-3)
    pre.dt = 10.0 * float(const.SECYER)
    pre.save_old_values()

    lnd_old = pre.lnd.copy()
    ln_t_old = pre.ln_t.copy()
    ln_r_old = pre.ln_r.copy()
    lum_old = pre.lum.copy()

    rep = pre.step_struct_burn_mix(dt=pre.dt, op_split_burn=False)
    assert rep.converged

    dt_next_rust, vc_rust, ratio_rust = pre.pick_next_timestep(
        max_timestep_factor=1.2, min_timestep_factor=0.5
    )
    vc_jax = float(
        star.eval_varcontrol_jax(
            jnp.asarray(pre.lnd),
            jnp.asarray(pre.ln_t),
            jnp.asarray(pre.ln_r),
            jnp.asarray(pre.lum),
            jnp.asarray(lnd_old),
            jnp.asarray(ln_t_old),
            jnp.asarray(ln_r_old),
            jnp.asarray(lum_old),
        )
    )
    np.testing.assert_allclose(vc_rust, vc_jax, rtol=1.0e-12, atol=1.0e-14)

    dt_next_jax = float(
        star.eval_h211_timestep_jax(
            pre.dt,
            pre.dt_old,
            ratio_rust,
            dt_limit_ratio_old=1.0,
            max_timestep_factor=1.2,
            min_timestep_factor=0.5,
        )
    )
    np.testing.assert_allclose(dt_next_rust, dt_next_jax, rtol=1.0e-12)

    test_util.check_grads(
        lambda w_curr, w_prev: star.eval_h211_timestep_jax(
            1.2e8, 1.0e8, w_curr, w_prev, max_timestep_factor=10.0
        ),
        (jnp.asarray(0.85), jnp.asarray(0.92)),
        order=1,
        rtol=1.0e-5,
        atol=1.0e-7,
    )


def test_star_test_4msun_zams_evolution_and_history_profile() -> None:
    """Verify `star/test` (`inlist_test_r`: 4 Msun ZAMS evolved to step 6)."""
    zams = star.StarModel.from_zams(initial_mass=4.0, initial_z=0.02)
    zams.set_evolve_controls(mesh_delta_coeff=0.5, varcontrol_target=1.0e-3)
    assert zams.model_number == 1

    # Evolve 5 steps from model_number=1 to model_number=6 (`max_model_number`).
    reports = zams.evolve(
        max_steps=5, do_remesh=False, op_split_burn=True, use_bcyclic=True
    )
    assert len(reports) == 5
    assert zams.model_number == 6
    assert all(r.converged for r in reports)
    assert zams.star_age > 0.0
    assert reports[1].dt == pytest.approx(reports[0].dt_next, rel=1.0e-14)
    assert reports[2].dt == pytest.approx(reports[1].dt_next, rel=1.0e-14)

    # Check physical ZAMS locus for 4 Msun (`log_Teff ~ 4.18`, `log_L ~ 2.38`).
    h = zams.history_row()
    assert h.model_number == 6
    assert 4.10 < h.log_teff < 4.25
    assert 2.25 < h.log_l < 2.50
    assert 7.35 < h.log_center_t < 7.50
    assert 0.65 < h.center_h1 <= 0.70

    hist_line = zams.to_history_line()
    assert len(hist_line.split()) == 17
    prof_text = zams.to_profile_str()
    assert "photosphere_L" in prof_text
    assert "x_mass_frac" in prof_text
