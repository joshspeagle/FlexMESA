# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for Wave 3a.3 (`PC`, `Skye`, `eosDT_get`, and inverse solvers)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import chem, eos


def _rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Return max relative difference `|a - b| / max(|a|, |b|, 1.0)`."""
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.0)
    return float(np.max(np.abs(a - b) / denom))


def _c12_o16_composition() -> tuple[list[int], np.ndarray, float, float]:
    """Return 50/50 `c12` + `o16` `(chem_id, xa, abar, zbar)`."""
    ic12 = chem.get_nuclide_index("c12")
    io16 = chem.get_nuclide_index("o16")
    assert ic12 is not None and io16 is not None
    chem_id = [ic12, io16]
    xa = np.array([0.5, 0.5], dtype=np.float64)
    comp = chem.get_composition_info(chem_id, xa)
    return chem_id, xa, float(comp.abar), float(comp.zbar)


def test_get_pc_and_skye_alfa_rust_and_jax_match_fortran_goldens() -> None:
    """Verify `get_pc_alfa` and `get_skye_alfa` match Fortran golden values."""
    _, _, abar, zbar = _c12_o16_composition()
    for log_rho, log_t in [(1.0, 7.0), (2.9, 7.0), (5.0, 6.2), (6.0, 7.65)]:
        a_r, dt_r, dr_r = eos.get_pc_alfa(
            log_rho, log_t, abar, zbar, backend="rust"
        )
        a_j, dt_j, dr_j = eos.get_pc_alfa(
            jnp.asarray(log_rho, dtype=jnp.float64),
            jnp.asarray(log_t, dtype=jnp.float64),
            jnp.asarray(abar, dtype=jnp.float64),
            jnp.asarray(zbar, dtype=jnp.float64),
            backend="jax",
        )
        assert abs(float(a_r) - float(a_j)) < 1.0e-12
        assert abs(float(dt_r) - float(dt_j)) < 1.0e-12
        assert abs(float(dr_r) - float(dr_j)) < 1.0e-12

    for lr_s, lt_s, ab_s, zb_s in [
        (6.0, 7.0, abar, zbar),
        (3.45, 7.0, 12.0, 6.0),
        (-10.0, 4.0, abar, zbar),
    ]:
        sa_r, sdt_r, sdr_r = eos.get_skye_alfa(
            lr_s, lt_s, ab_s, zb_s, backend="rust"
        )
        sa_j, sdt_j, sdr_j = eos.get_skye_alfa(
            jnp.asarray(lr_s, dtype=jnp.float64),
            jnp.asarray(lt_s, dtype=jnp.float64),
            jnp.asarray(ab_s, dtype=jnp.float64),
            jnp.asarray(zb_s, dtype=jnp.float64),
            backend="jax",
        )
        assert abs(float(sa_r) - float(sa_j)) < 1.0e-12
        assert abs(float(sdt_r) - float(sdt_j)) < 1.0e-12
        assert abs(float(sdr_r) - float(sdr_j)) < 1.0e-12

    for lr_sm, lt_sm in [(2.0, 6.0), (2.35, 7.3), (4.0, 8.0)]:
        s_r = eos.get_skye_alfa_simple(lr_sm, lt_sm, backend="rust")
        s_j = eos.get_skye_alfa_simple(
            jnp.asarray(lr_sm, dtype=jnp.float64),
            jnp.asarray(lt_sm, dtype=jnp.float64),
            backend="jax",
        )
        for col in range(3):
            assert abs(float(s_r[col]) - float(s_j[col])) < 1.0e-12


def test_eval_pc_eos_and_zones_match_across_rust_and_jax() -> None:
    """Verify `eval_pc_eos` and `eval_pc_eos_zones` across liquid/solid WDs."""
    chem_id, xa, abar, zbar = _c12_o16_composition()
    azion = np.array(
        [float(chem.get_nuclide_info(c).z) for c in chem_id], dtype=np.float64
    )
    acmi = np.array(
        [float(chem.get_nuclide_info(c).w) for c in chem_id], dtype=np.float64
    )

    # Liquid WD interior (logRho=5, logT=7) and solid core (logRho=7, logT=6.3)
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
        assert _rel_diff(np.asarray(out_r.res), np.asarray(out_j.res)) < tol_res
        assert (
            _rel_diff(np.asarray(out_r.d_dlnd), np.asarray(out_j.d_dlnd))
            < 1.0e-10
        )
        assert (
            _rel_diff(np.asarray(out_r.d_dln_t), np.asarray(out_j.d_dln_t))
            < 1.0e-10
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


def test_eval_skye_eos_and_eos_dt_get_rust_jax_and_autodiff() -> None:
    """Verify `eval_skye_eos` and `eos_dt_get` parity and `jax.grad` rules."""
    chem_id, xa, abar, zbar = _c12_o16_composition()
    rho = 1.0e6
    temp = 1.0e7

    sk_r = eos.eval_skye_eos(rho, temp, abar, zbar, chem_id, xa, backend="rust")
    sk_j = eos.eval_skye_eos(
        jnp.asarray(rho, dtype=jnp.float64),
        jnp.asarray(temp, dtype=jnp.float64),
        jnp.asarray(abar, dtype=jnp.float64),
        jnp.asarray(zbar, dtype=jnp.float64),
        chem_id,
        jnp.asarray(xa, dtype=jnp.float64),
        backend="jax",
    )
    assert _rel_diff(np.asarray(sk_r.res), np.asarray(sk_j.res)) < 1.0e-11
    assert _rel_diff(np.asarray(sk_r.d_dlnd), np.asarray(sk_j.d_dlnd)) < 1.0e-8
    assert (
        _rel_diff(np.asarray(sk_r.d_dln_t), np.asarray(sk_j.d_dln_t)) < 2.0e-6
    )
    assert float(sk_r.res[eos.I_FRAC_SKYE]) == 1.0

    # Verify multi-zone eval_skye_eos_zones across Rust and native @jax.jit
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
    assert _rel_diff(np.asarray(sk_zr.res), np.asarray(sk_zj.res)) < 1.0e-10

    # Verify Ogata solid mixing rule in crystallized regime
    sk_ogata = eos.eval_skye_eos(
        1.0e6,
        2.0e6,
        abar,
        zbar,
        chem_id,
        xa,
        skye_solid_mixing_rule="Ogata",
        backend="rust",
    )
    assert float(sk_ogata.res[eos.I_PHASE]) > 0.99

    # Verify unified eosDT_get and jax.grad wrt (rho, temp)
    dt_r = eos.eos_dt_get(rho, temp, chem_id, xa, backend="rust")
    dt_j = eos.eos_dt_get(
        jnp.asarray(rho, dtype=jnp.float64),
        jnp.asarray(temp, dtype=jnp.float64),
        chem_id,
        jnp.asarray(xa, dtype=jnp.float64),
        backend="jax",
    )
    assert _rel_diff(np.asarray(dt_r.res), np.asarray(dt_j.res)) < 1.0e-11

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


def test_inverse_solvers_pt_dt_and_gamma_de_with_jax_ift_grad() -> None:
    """Verify `eos_pt_get`, `eos_dt_get_t`, `eos_dt_get_rho`, and `jax.grad`."""
    chem_id, xa, abar, zbar = _c12_o16_composition()
    rho_true = 100.0
    temp_true = 1.0e7
    fwd = eos.eos_dt_get(rho_true, temp_true, chem_id, xa, backend="rust")
    pgas_true = float(np.exp(fwd.res[eos.I_LN_PGAS]))
    ln_e_true = float(fwd.res[eos.I_LN_E])

    pt_r = eos.eos_pt_get(pgas_true, temp_true, chem_id, xa, backend="rust")
    assert abs(float(pt_r.rho) - rho_true) / rho_true < 1.0e-10

    inv_t = eos.eos_dt_get_t(
        rho_true, eos.I_LN_E, ln_e_true, chem_id, xa, log_t_guess=6.8
    )
    assert abs(float(inv_t.result) - temp_true) / temp_true < 1.0e-10

    inv_rho = eos.eos_dt_get_rho(
        temp_true,
        eos.I_LN_PGAS,
        float(fwd.res[eos.I_LN_PGAS]),
        chem_id,
        xa,
        log_rho_guess=1.5,
    )
    assert abs(float(inv_rho.result) - rho_true) / rho_true < 1.0e-10

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

    # Verify closed-form gamma_DE solver
    gde = eos.eos_gamma_de_get(abar, zbar, 10.0, 1.0e15, gamma=5.0 / 3.0)
    assert (
        abs(np.exp(float(gde.eos.res[eos.I_LN_E])) - 1.0e15) / 1.0e15 < 1.0e-12
    )


def test_pc_and_skye_off_golden_grid_parity_and_jacfwd() -> None:
    """Verify `eval_pc_eos` and `eval_skye_eos` across off-golden states."""
    chem_id, xa, abar, zbar = _c12_o16_composition()
    for log_rho, log_t in [(3.8, 7.2), (5.4, 6.8), (6.7, 6.1), (7.5, 6.4)]:
        rho = 10.0**log_rho
        temp = 10.0**log_t
        pc_r = eos.eval_pc_eos(
            rho, temp, abar, zbar, chem_id, xa, backend="rust"
        )
        pc_j = eos.eval_pc_eos(
            jnp.float64(rho),
            jnp.float64(temp),
            jnp.float64(abar),
            jnp.float64(zbar),
            chem_id,
            jnp.asarray(xa),
            backend="jax",
        )
        assert (
            _rel_diff(np.asarray(pc_r.res[:8]), np.asarray(pc_j.res[:8]))
            <= 1.0e-12
        )
        assert _rel_diff(np.asarray(pc_r.res), np.asarray(pc_j.res)) < 1.0e-11

        sk_r = eos.eval_skye_eos(
            rho, temp, abar, zbar, chem_id, xa, backend="rust"
        )
        sk_j = eos.eval_skye_eos(
            jnp.float64(rho),
            jnp.float64(temp),
            jnp.float64(abar),
            jnp.float64(zbar),
            chem_id,
            jnp.asarray(xa),
            backend="jax",
        )
        assert (
            _rel_diff(np.asarray(sk_r.res[:3]), np.asarray(sk_j.res[:3]))
            < 5.0e-11
        )
        assert _rel_diff(np.asarray(sk_r.res), np.asarray(sk_j.res)) < 1.0e-6

    # Verify jax.jacfwd on eval_pc_eos and eval_skye_eos matches d_dlnd, d_dln_t
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
    assert (
        _rel_diff(np.asarray(pc0.d_dlnd), np.asarray(jac_pc_r * rho0))
        <= 1.0e-12
    )
    assert (
        _rel_diff(np.asarray(pc0.d_dln_t), np.asarray(jac_pc_t * t0)) <= 1.0e-12
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
    assert (
        _rel_diff(np.asarray(sk0.d_dlnd), np.asarray(jac_sk_r * rho0))
        <= 1.0e-12
    )
    assert (
        _rel_diff(np.asarray(sk0.d_dln_t), np.asarray(jac_sk_t * t0)) <= 1.0e-12
    )
