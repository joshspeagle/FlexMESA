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
    assert _rel_diff(np.asarray(zones_r.res), np.asarray(zones_j.res)) < 5.0e-12
    assert (
        _rel_diff(np.asarray(zones_r.d_dlnd), np.asarray(zones_j.d_dlnd))
        < 1.0e-10
    )
    assert (
        _rel_diff(np.asarray(zones_r.d_dln_t), np.asarray(zones_j.d_dln_t))
        < 1.0e-10
    )


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
    """Verify `eos_pt_get`, `eos_dt_get_t`, `eos_dt_get_rho`, and `jax.grad`.

    Upstream Fortran MESA `eos/private/eospt_eval.f90:180-191`
    (`Get_PT_Results_using_DT`) hardcodes `other_tol = 1d-8`,
    `logRho_tol = 1d-8`, and `max_iter = 20`, terminating Newton iteration in
    `num/private/mod_root.f90:102` (`do_safe_root_with_guess`) when
    `|ln P_gas(x_i) - ln P_gas_target| < max(other_tol, |dy/dx| * logRho_tol)`.
    On the `Skye`/`HELM` branch (`use_free_eos=False`), step `i = 3` reaches
    `|rho - rho_true| / rho_true < 1e-10`; on the full `Z = 1` `FreeEOS` table
    (`mesa-FreeEOS_100z00x_CO`), step `i = 2` reaches `|dy| = 3.31e-10 < 2.3e-8`
    with `|log10_rho - 2.0| = 6.25e-11 < 1e-10`.
    """
    chem_id, xa, abar, zbar = _c12_o16_composition()
    rho_true = 100.0
    temp_true = 1.0e7

    fwd_helm = eos.eos_dt_get(
        rho_true, temp_true, chem_id, xa, use_free_eos=False, backend="rust"
    )
    pt_helm = eos.eos_pt_get(
        float(np.exp(fwd_helm.res[eos.I_LN_PGAS])),
        temp_true,
        chem_id,
        xa,
        use_free_eos=False,
        backend="rust",
    )
    assert abs(float(pt_helm.rho) - rho_true) / rho_true < 1.0e-10

    fwd = eos.eos_dt_get(rho_true, temp_true, chem_id, xa, backend="rust")
    pgas_true = float(np.exp(fwd.res[eos.I_LN_PGAS]))
    ln_e_true = float(fwd.res[eos.I_LN_E])

    pt_r = eos.eos_pt_get(pgas_true, temp_true, chem_id, xa, backend="rust")
    pt_j = eos.eos_pt_get(pgas_true, temp_true, chem_id, xa, backend="jax")
    assert abs(float(pt_r.log10_rho) - np.log10(rho_true)) < 1.0e-10
    assert (
        abs(float(pt_r.eos.res[eos.I_LN_PGAS]) - float(fwd.res[eos.I_LN_PGAS]))
        < 1.0e-9
    )
    assert abs(float(pt_j.log10_rho) - float(pt_r.log10_rho)) < 1.0e-14

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
        assert _rel_diff(np.asarray(sk_r.res), np.asarray(sk_j.res)) < 1.0e-6

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
        assert _rel_diff(np.asarray(pc_r.res), np.asarray(pc_j.res)) < 1.0e-11
        assert (
            _rel_diff(np.asarray(pc_r.d_dlnd), np.asarray(pc_j.d_dlnd))
            < 1.0e-10
        )
        assert (
            _rel_diff(np.asarray(pc_r.d_dln_t), np.asarray(pc_j.d_dln_t))
            < 1.0e-10
        )

    dt_r = eos.eos_dt_get(1.0e6, 1.0e7, cids_9, xa_9, backend="rust")
    dt_j = eos.eos_dt_get(
        jnp.float64(1.0e6), jnp.float64(1.0e7), cids_9, xa_9j, backend="jax"
    )
    assert _rel_diff(np.asarray(dt_r.res), np.asarray(dt_j.res)) < 1.0e-10

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
    assert _rel_diff(np.asarray(pc_z_r.res), np.asarray(pc_z_j.res)) < 1.0e-11

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
    assert _rel_diff(np.asarray(sk_z_r.res), np.asarray(sk_z_j.res)) < 1.0e-6

    ef, ex = jnp.zeros((0,), jnp.float64), jnp.zeros((0, 35), jnp.float64)
    pc_0 = eos.eval_pc_eos_zones(
        ef, ef, ef, ef, ex, z_35j, w_35j, backend="jax"
    )
    sk_0 = eos.eval_skye_eos_zones(
        ef, ef, ef, ef, ex, z_35j, w_35j, a_35j, backend="jax"
    )
    assert pc_0.res.shape == (0, eos.NUM_EOS_BASIC_RESULTS)
    assert sk_0.res.shape == (0, eos.NUM_EOS_BASIC_RESULTS)
