# SPDX-License-Identifier: LGPL-3.0-or-later
"""Phase 5a Python/PyO3 & JAX verification tests (`Issue #112`, `PR #135`).

Verifies:
1. `flexmesa.net.burn_1_zone_const_density` (`BurnConstDensityOutput`)
   conserves baryon mass fraction to `<= 1.0e-12` and integrates coupled
   temperature self-heating `(Y_i, ln T)` during exothermic triple-alpha
   helium burning.
2. `flexmesa.star.parse_inlist_bundle` exposes all documented Phase 5a
   `&star_job` and `&controls` keys (`rti_flag`, `load_saved_model_for_rsp2`,
   `okay_to_remesh`, `use_gold_tolerances`, `use_gold2_tolerances`,
   `op_split_burn_min_t`).
3. `flexmesa.star.eval_hydro_residuals_jax` respects trapezoidal thermal
   neutrino time-centering (`non_nuc_neu_start`), `dedt_eqn_r_scale`, and
   `use_momentum_outer_bc`, and its `jax.jacrev` Jacobian matches closed-form
   analytical derivatives to `<= 1.0e-13`.
"""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import constants, net, star


def test_burn_1_zone_const_density_conserves_baryons_and_heats_mixture() -> (
    None
):
    """Constant-density burn heats pure He4 and conserves baryon sum."""
    info = net.get_net_info("approx21.net")
    x0 = np.zeros(info.num_isos, dtype=np.float64)
    he4_idx = info.iso_names.index("he4")
    c12_idx = info.iso_names.index("c12")
    x0[he4_idx] = 1.0

    out = net.burn_1_zone_const_density(
        "approx21.net",
        t_start=0.0,
        t_end=1.0,
        starting_x=x0,
        starting_log10_t=8.35,
        log10_rho=5.0,
        stptry_in=1.0e-4,
        eps=1.0e-9,
        odescal=1.0e-11,
    )
    assert isinstance(out, net.BurnConstDensityOutput)
    assert out.ending_x.shape == (info.num_isos,)
    assert abs(float(np.sum(out.ending_x)) - 1.0) <= 1.0e-12
    assert out.ending_log10_t > 8.35
    assert 10.0**out.ending_log10_t > 10.0**8.35
    assert out.avg_eps_nuc > 0.0
    assert float(out.ending_x[c12_idx]) > 0.0
    assert out.naccpt >= 1


def test_parse_inlist_bundle_exposes_all_phase5a_star_controls(
    tmp_path: pathlib.Path,
) -> None:
    """Inlist bundle parser populates all documented Phase 5a control keys."""
    inlist_file = tmp_path / "inlist_phase5a"
    inlist_file.write_text(
        "&star_job\n"
        "  change_u_flag = .true.\n"
        "  new_u_flag = .true.\n"
        "  change_rti_flag = .true.\n"
        "  new_rti_flag = .true.\n"
        "  load_saved_model_for_rsp2 = .true.\n"
        "  change_net = .true.\n"
        "  new_net_name = 'approx21.net'\n"
        "/\n"
        "&controls\n"
        "  okay_to_remesh = .false.\n"
        "  use_gold_tolerances = .true.\n"
        "  use_gold2_tolerances = .false.\n"
        "  op_split_burn = .true.\n"
        "  op_split_burn_min_t = 1.5d8\n"
        "/\n",
        encoding="utf-8",
    )
    parsed = star.parse_inlist_bundle(inlist_file)
    assert parsed["rti_flag"] is True
    assert parsed["load_saved_model_for_rsp2"] is True
    assert parsed["change_net"] is True
    assert parsed["new_net_name"] == "approx21.net"
    assert parsed["okay_to_remesh"] is False
    assert parsed["u_flag"] is True
    assert parsed["use_gold_tolerances"] is True
    assert parsed["use_gold2_tolerances"] is False
    assert parsed["op_split_burn"] is True
    assert abs(float(parsed["op_split_burn_min_t"]) - 1.5e8) <= 1.0e-6

    inlist_ctrl_rti = tmp_path / "inlist_ctrl_rti"
    inlist_ctrl_rti.write_text(
        "&controls\n  rti_flag = .true.\n/\n",
        encoding="utf-8",
    )
    assert star.parse_inlist_bundle(inlist_ctrl_rti)["rti_flag"] is True


def test_eval_hydro_residuals_jax_jacobian_matches_analytical_partials() -> (
    None
):
    """JAX hydro residual kernel honors Phase 5a flags and exact autodiff."""
    dq = jnp.asarray([0.10, 0.20, 0.30, 0.40], dtype=jnp.float64)
    mstar = 1.988409870698051e33
    ln_r = jnp.log(jnp.asarray([7.0e10, 5.5e10, 3.8e10, 2.0e10]))
    grid = star.compute_staggered_grid_jax(dq, ln_r, mstar)
    lnd = jnp.asarray([-6.0, -2.0, 1.5, 4.2], dtype=jnp.float64)
    lnd_start = lnd - 1.0e-3
    ln_t = jnp.asarray([9.0, 13.5, 15.8, 16.5], dtype=jnp.float64)
    lum = jnp.asarray([3.8e33, 3.7e33, 3.0e33, 1.2e33], dtype=jnp.float64)
    ln_peos = jnp.asarray([12.0, 24.0, 33.0, 38.0], dtype=jnp.float64)
    ln_e = jnp.asarray([29.0, 33.5, 35.8, 36.5], dtype=jnp.float64)
    energy_start = jnp.exp(ln_e - 5.0e-4)
    gradt = jnp.asarray([0.28, 0.32, 0.38, 0.40], dtype=jnp.float64)
    eps_nuc = jnp.asarray([0.0, 0.0, 2.5e7, 1.8e8], dtype=jnp.float64)
    neu = jnp.asarray([0.0, 2.0e6, 4.0e7, 1.2e8], dtype=jnp.float64)
    neu_start = jnp.asarray([0.0, 6.0e6, 8.0e7, 2.0e8], dtype=jnp.float64)
    eps_mdot = jnp.zeros_like(dq)
    ln_p_surf = 11.8
    ln_t_surf = 8.9
    dt = 1.0e4

    equ_tc = star.eval_hydro_residuals_jax(
        dq,
        grid["dm"],
        grid["dm_bar"],
        grid["m"],
        lnd,
        ln_t,
        ln_r,
        lum,
        lnd_start,
        ln_peos,
        ln_e,
        energy_start,
        gradt,
        eps_nuc,
        neu,
        eps_mdot,
        ln_p_surf,
        ln_t_surf,
        dt,
        non_nuc_neu_start=neu_start,
        dedt_eqn_r_scale=10.0,
        use_momentum_outer_bc=True,
    )
    equ_inst = star.eval_hydro_residuals_jax(
        dq,
        grid["dm"],
        grid["dm_bar"],
        grid["m"],
        lnd,
        ln_t,
        ln_r,
        lum,
        lnd_start,
        ln_peos,
        ln_e,
        energy_start,
        gradt,
        eps_nuc,
        neu,
        eps_mdot,
        ln_p_surf,
        ln_t_surf,
        dt,
        non_nuc_neu_start=None,
        dedt_eqn_r_scale=10.0,
        use_momentum_outer_bc=True,
    )
    assert (
        abs(
            float(equ_tc[0, 1])
            - (ln_p_surf - float(ln_peos[0])) / float(ln_peos[0])
        )
        <= 1.0e-15
    )
    assert (
        abs(float(equ_tc[0, 0]) - (ln_t_surf - float(ln_t[0])) / float(ln_t[0]))
        <= 1.0e-15
    )

    total_e = float(jnp.sum(energy_start * grid["dm"]))
    surf_cap = jnp.asarray([1.0e-6, 1.0, 1.0, 1.0], dtype=jnp.float64)
    base_scal = jnp.minimum(
        surf_cap, 10.0 * energy_start * grid["dm"] / total_e
    )
    scal = base_scal * dt / jnp.maximum(energy_start, 1.0)
    expected_shift = 0.5 * (neu - neu_start) * scal
    np.testing.assert_allclose(
        np.asarray(equ_tc[:, 3] - equ_inst[:, 3]),
        np.asarray(expected_shift),
        rtol=1.0e-13,
        atol=1.0e-15,
    )

    jac_lnd = jax.jacrev(
        lambda x_lnd: star.eval_hydro_residuals_jax(
            dq,
            grid["dm"],
            grid["dm_bar"],
            grid["m"],
            x_lnd,
            ln_t,
            ln_r,
            lum,
            lnd_start,
            ln_peos,
            ln_e,
            energy_start,
            gradt,
            eps_nuc,
            neu,
            eps_mdot,
            ln_p_surf,
            ln_t_surf,
            dt,
            non_nuc_neu_start=neu_start,
            dedt_eqn_r_scale=10.0,
            use_momentum_outer_bc=True,
        )[:, 2]
    )(lnd)
    r_arr = jnp.exp(ln_r)
    r_p1_3 = jnp.concatenate(
        [r_arr[1:] ** 3, jnp.zeros((1,), dtype=jnp.float64)]
    )
    dr3 = (3.0 * grid["dm"] / float(constants.FOUR_PI)) / jnp.exp(lnd)
    expected_diag = dr3 / (3.0 * (r_p1_3 + dr3))
    np.testing.assert_allclose(
        np.asarray(jnp.diag(jac_lnd)),
        np.asarray(expected_diag),
        rtol=1.0e-14,
        atol=1.0e-15,
    )
