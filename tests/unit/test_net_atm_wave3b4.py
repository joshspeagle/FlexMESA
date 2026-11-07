# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit, golden parity, and JAX autodiff tests for `atm` and `net`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import atm, chem, constants, net


def _solar_species_and_xa(
    x: float = 0.70, z: float = 0.02
) -> tuple[list[int], np.ndarray]:
    names = ["h1", "he4", "c12", "n14", "o16", "ne20", "mg24"]
    chem_id = [int(chem.get_nuclide_index(n) or 0) for n in names]
    y = 1.0 - (x + z)
    xc = 3.2724592105263235e-3
    xn = 9.5023842105263292e-4
    xo = 8.8218000000000601e-3
    sum6 = x + y + xc + xn + xo
    xa = np.array([x, y, xc, xn, xo, 0.0, 1.0 - sum6], dtype=np.float64)
    return chem_id, xa


def test_atm_t_tau_relations_and_jax_derivatives_match_analytical_hopf() -> (
    None
):
    """Test `eval_t_tau` and `eval_t_tau_jax` parity and `jax.grad` vs Hopf."""
    tau = 0.42
    teff = 5772.0
    for rel_id in (
        atm.ATM_T_TAU_EDDINGTON,
        atm.ATM_T_TAU_SOLAR_HOPF,
        atm.ATM_T_TAU_KRISHNA_SWAMY,
        atm.ATM_T_TAU_TRAMPEDACH_SOLAR,
    ):
        lnt_rust = atm.eval_t_tau(rel_id, tau, teff)
        lnt_jax = float(
            atm.eval_t_tau_jax(
                rel_id,
                jnp.asarray(tau, dtype=jnp.float64),
                jnp.asarray(teff, dtype=jnp.float64),
            )
        )
        np.testing.assert_allclose(lnt_jax, lnt_rust, rtol=1e-14, atol=1e-14)

        dq_dtau = atm.eval_t_tau_dq_dtau(rel_id, tau)
        dlnt_dtau_jax = float(
            jax.grad(
                lambda t_val, r_id=rel_id: atm.eval_t_tau_jax(r_id, t_val, teff)
            )(jnp.asarray(tau, dtype=jnp.float64))
        )
        q_val = (4.0 / 3.0) * (np.exp(lnt_rust) / teff) ** 4 - tau
        dlnt_dtau_exact = (1.0 + dq_dtau) / (4.0 * (tau + q_val))
        rtol = 1e-4 if rel_id == atm.ATM_T_TAU_TRAMPEDACH_SOLAR else 1e-11
        np.testing.assert_allclose(
            dlnt_dtau_jax, dlnt_dtau_exact, rtol=rtol, atol=1e-12
        )


def test_atm_paczynski_gradr_rust_and_jax_parity() -> None:
    """Test `eval_paczynski_gradr` and `eval_paczynski_gradr_jax` parity."""
    for tau in (0.15, 2.0 / 3.0, 5.0):
        g_rust = atm.eval_paczynski_gradr(
            5800.0,
            1.0e5,
            2.5e-7,
            tau,
            0.35,
            constants.LSUN,
            constants.MSUN,
            constants.RSUN,
            constants.STANDARD_CGRAV,
        )
        g_jax = float(
            atm.eval_paczynski_gradr_jax(
                jnp.asarray(5800.0),
                jnp.asarray(1.0e5),
                jnp.asarray(2.5e-7),
                jnp.asarray(tau),
                jnp.asarray(0.35),
                jnp.asarray(constants.LSUN),
                jnp.asarray(constants.MSUN),
                jnp.asarray(constants.RSUN),
                jnp.asarray(constants.STANDARD_CGRAV),
            )
        )
        np.testing.assert_allclose(g_jax, g_rust, rtol=1e-14, atol=1e-14)


def test_atm_table_and_custom_vjp_match_golden_and_finite_differences() -> None:
    """Test `eval_table` and `eval_table_jax` custom_vjp against FD."""
    l_val = constants.LSUN
    r_val = constants.RSUN
    m_val = constants.MSUN
    cgrav = constants.STANDARD_CGRAV
    teff = (
        l_val / (4.0 * np.pi * r_val * r_val * constants.BOLTZ_SIGMA)
    ) ** 0.25

    out = atm.eval_table(
        l_val,
        r_val,
        m_val,
        cgrav,
        atm.ATM_TABLE_PHOTOSPHERE,
        0.02,
        False,
        teff,
    )
    np.testing.assert_allclose(out.ln_t / constants.LN10, 3.76132658, rtol=1e-7)
    np.testing.assert_allclose(out.ln_p / constants.LN10, 4.91699929, rtol=1e-7)

    l_j = jnp.asarray(l_val, dtype=jnp.float64)
    r_j = jnp.asarray(r_val, dtype=jnp.float64)
    m_j = jnp.asarray(m_val, dtype=jnp.float64)
    lnt_j, lnp_j = atm.eval_table_jax(
        l_j, r_j, m_j, cgrav, atm.ATM_TABLE_PHOTOSPHERE, 0.02
    )
    np.testing.assert_allclose(float(lnt_j), out.ln_t, rtol=1e-14)
    np.testing.assert_allclose(float(lnp_j), out.ln_p, rtol=1e-14)

    grad_lnt = jax.grad(
        lambda lv, rv, mv: atm.eval_table_jax(
            lv, rv, mv, cgrav, atm.ATM_TABLE_PHOTOSPHERE, 0.02
        )[0],
        argnums=(0, 1, 2),
    )(l_j, r_j, m_j)
    grad_lnp = jax.grad(
        lambda lv, rv, mv: atm.eval_table_jax(
            lv, rv, mv, cgrav, atm.ATM_TABLE_PHOTOSPHERE, 0.02
        )[1],
        argnums=(0, 1, 2),
    )(l_j, r_j, m_j)
    np.testing.assert_allclose(float(grad_lnt[0]), out.dln_t_dl, rtol=1e-12)
    np.testing.assert_allclose(
        float(grad_lnt[1]) * r_val, out.dln_t_dln_r, rtol=1e-12
    )
    np.testing.assert_allclose(
        float(grad_lnt[2]) * m_val, out.dln_t_dln_m, rtol=1e-12
    )
    np.testing.assert_allclose(float(grad_lnp[0]), out.dln_p_dl, rtol=1e-12)
    np.testing.assert_allclose(
        float(grad_lnp[1]) * r_val, out.dln_p_dln_r, rtol=1e-12
    )
    np.testing.assert_allclose(
        float(grad_lnp[2]) * m_val, out.dln_p_dln_m, rtol=1e-12
    )

    # Central finite-difference check at Z = 0.0179 (exact GN93 Z-grid node)
    z_node = 0.0179
    grad_lnt_z = jax.grad(
        lambda lv, rv, mv: atm.eval_table_jax(
            lv, rv, mv, cgrav, atm.ATM_TABLE_PHOTOSPHERE, z_node
        )[0],
        argnums=(0, 1, 2),
    )(l_j, r_j, m_j)
    grad_lnp_z = jax.grad(
        lambda lv, rv, mv: atm.eval_table_jax(
            lv, rv, mv, cgrav, atm.ATM_TABLE_PHOTOSPHERE, z_node
        )[1],
        argnums=(0, 1, 2),
    )(l_j, r_j, m_j)
    eps = 1.0e-5
    for idx_arg, (base_v, dlnt_v, dlnp_v) in enumerate(
        [
            (l_val, float(grad_lnt_z[0]), float(grad_lnp_z[0])),
            (r_val, float(grad_lnt_z[1]), float(grad_lnp_z[1])),
            (m_val, float(grad_lnt_z[2]), float(grad_lnp_z[2])),
        ]
    ):
        step = eps * base_v
        args_p = [l_val, r_val, m_val]
        args_m = [l_val, r_val, m_val]
        args_p[idx_arg] += step
        args_m[idx_arg] -= step
        lnt_p, lnp_p = atm.eval_table_jax(
            jnp.asarray(args_p[0]),
            jnp.asarray(args_p[1]),
            jnp.asarray(args_p[2]),
            cgrav,
            atm.ATM_TABLE_PHOTOSPHERE,
            z_node,
        )
        lnt_m, lnp_m = atm.eval_table_jax(
            jnp.asarray(args_m[0]),
            jnp.asarray(args_m[1]),
            jnp.asarray(args_m[2]),
            cgrav,
            atm.ATM_TABLE_PHOTOSPHERE,
            z_node,
        )
        fd_lnt = (float(lnt_p) - float(lnt_m)) / (2.0 * step)
        fd_lnp = (float(lnp_p) - float(lnp_m)) / (2.0 * step)
        if abs(dlnt_v) > 0.0:
            np.testing.assert_allclose(dlnt_v, fd_lnt, rtol=1e-6)
        if abs(dlnp_v) > 0.0:
            np.testing.assert_allclose(dlnp_v, fd_lnp, rtol=1e-6)


def test_atm_uniform_varying_and_irradiated_match_mesa_golden() -> None:
    """Test uniform, varying, and irradiated grey atmosphere evaluators."""
    chem_id, xa = _solar_species_and_xa()
    cgrav = constants.STANDARD_CGRAV
    r_val = 10.0 ** np.log10(constants.RSUN)
    m_val = 10.0 ** np.log10(constants.MSUN)
    teff = 10.0 ** np.log10(5776.0)
    l_val = np.pi * constants.CRAD * constants.CLIGHT * r_val**2 * teff**4

    u_out = atm.eval_t_tau_uniform(
        150.0,
        l_val,
        r_val,
        m_val,
        cgrav,
        0.5,
        1.0,
        atm.ATM_T_TAU_EDDINGTON,
        1.0e-6,
        30,
        True,
        teff,
        chem_id=chem_id,
        xa=xa,
        zbase=0.02,
    )
    np.testing.assert_allclose(
        u_out.ln_p / constants.LN10, 4.7483690740375719, rtol=1e-12
    )
    np.testing.assert_allclose(
        u_out.ln_t / constants.LN10, 4.2748967954324373, rtol=1e-13
    )

    tau_ks = atm.get_t_tau_base(atm.ATM_T_TAU_KRISHNA_SWAMY)
    v_out = atm.eval_t_tau_varying(
        tau_ks,
        l_val,
        r_val,
        m_val,
        cgrav,
        atm.ATM_T_TAU_KRISHNA_SWAMY,
        1.0e-9,
        500,
        True,
        teff,
        chem_id=chem_id,
        xa=xa,
        zbase=0.02,
    )
    np.testing.assert_allclose(
        v_out.ln_t / constants.LN10, 3.76162718, rtol=1e-7
    )
    np.testing.assert_allclose(
        v_out.ln_p / constants.LN10, 4.77402706, rtol=1e-6
    )

    z_irr = 0.01
    cid_irr, xa_irr = _solar_species_and_xa(0.70, z_irr)
    m_irr = 1.5 * constants.M_JUPITER
    r_irr = np.sqrt(cgrav * m_irr * 10.0 / (1.0e6 * 1.5e-2))
    l_irr = np.pi * constants.CRAD * constants.CLIGHT * r_irr**2 * 900.0**4
    irr_out = atm.eval_irradiated(
        l_irr,
        r_irr,
        m_irr,
        cgrav,
        1000.0,
        1.0e6,
        1.5e-2,
        4.0e-3,
        0.0,
        1.0e-6,
        30,
        True,
        chem_id=cid_irr,
        xa=xa_irr,
        zbase=z_irr,
        low_t_option="lowT_Freedman11",
    )
    np.testing.assert_allclose(
        irr_out.ln_t / constants.LN10, 3.1180735947393834, rtol=1e-12
    )
    np.testing.assert_allclose(irr_out.tau, 3.1669763036977079, rtol=1e-11)


def test_net_eval_and_jax_custom_vjp_match_finite_differences() -> None:
    """Test `net.eval_net` and `net.eval_net_jax` custom_vjp against FD."""
    networks = net.list_networks()
    assert "basic.net" in networks
    assert "approx21.net" in networks

    info = net.get_net_info("basic.net")
    assert info.num_isos == 8
    idx = {name: i for i, name in enumerate(info.iso_names)}
    xin = np.zeros(info.num_isos, dtype=np.float64)
    xin[idx["h1"]] = float(np.float32(0.655186))
    xin[idx["he4"]] = 0.31002164
    xin[idx["c12"]] = 0.002725e-1
    xin[idx["n14"]] = 0.203101e-1 + 0.612124e-6 + 0.109305e-2 + 0.356004e-4
    xin[idx["o16"]] = 0.094000e-1
    xin[idx["ne20"]] = 0.162163e-2
    xin[idx["mg24"]] = 0.658226e-25
    xin[idx["he3"]] = 0.201852e-2
    xin[idx["mg24"]] += 1.0 - float(np.sum(xin))

    comp = chem.get_composition_info(list(info.chem_id), xin)
    temp = 10.0**7.833
    rho = 100.0

    res = net.eval_net(
        "basic.net",
        xin,
        temp,
        rho,
        comp.abar,
        comp.zbar,
        comp.z2bar,
        comp.ye,
        eta=-2.0,
    )
    np.testing.assert_allclose(res.eps_nuc, 0.839654151e13, rtol=1e-8)

    eps_j, dxdt_j = net.eval_net_jax(
        "basic.net",
        jnp.asarray(xin),
        jnp.asarray(temp),
        jnp.asarray(rho),
        comp.abar,
        comp.zbar,
        comp.z2bar,
        comp.ye,
        eta=-2.0,
    )
    np.testing.assert_allclose(float(eps_j), res.eps_nuc, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(dxdt_j), res.dxdt, rtol=1e-14)

    grad_eps = jax.grad(
        lambda xv, tv, rv: net.eval_net_jax(
            "basic.net",
            xv,
            tv,
            rv,
            comp.abar,
            comp.zbar,
            comp.z2bar,
            comp.ye,
            eta=-2.0,
        )[0],
        argnums=(0, 1, 2),
    )(jnp.asarray(xin), jnp.asarray(temp), jnp.asarray(rho))
    np.testing.assert_allclose(
        np.asarray(grad_eps[0]), res.d_eps_nuc_dx, rtol=1e-14
    )
    np.testing.assert_allclose(float(grad_eps[1]), res.d_eps_nuc_dt, rtol=1e-14)
    np.testing.assert_allclose(
        float(grad_eps[2]), res.d_eps_nuc_drho, rtol=1e-14
    )

    jac_dxdt_x, jac_dxdt_t, jac_dxdt_rho = jax.jacrev(
        lambda xv, tv, rv: net.eval_net_jax(
            "basic.net",
            xv,
            tv,
            rv,
            comp.abar,
            comp.zbar,
            comp.z2bar,
            comp.ye,
            eta=-2.0,
        )[1],
        argnums=(0, 1, 2),
    )(jnp.asarray(xin), jnp.asarray(temp), jnp.asarray(rho))
    np.testing.assert_allclose(
        np.asarray(jac_dxdt_x), res.d_dxdt_dx, rtol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(jac_dxdt_t), res.d_dxdt_dt, rtol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(jac_dxdt_rho), res.d_dxdt_drho, rtol=1e-14
    )

    # Central finite-difference verification of eps_nuc and dxdt wrt (x, T, rho)
    dt = 1.0e-6 * temp
    eps_tp, dxdt_tp = net.eval_net_jax(
        "basic.net",
        jnp.asarray(xin),
        jnp.asarray(temp + dt),
        jnp.asarray(rho),
        comp.abar,
        comp.zbar,
        comp.z2bar,
        comp.ye,
        eta=-2.0,
    )
    eps_tm, dxdt_tm = net.eval_net_jax(
        "basic.net",
        jnp.asarray(xin),
        jnp.asarray(temp - dt),
        jnp.asarray(rho),
        comp.abar,
        comp.zbar,
        comp.z2bar,
        comp.ye,
        eta=-2.0,
    )
    np.testing.assert_allclose(
        float(grad_eps[1]),
        (float(eps_tp) - float(eps_tm)) / (2.0 * dt),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(jac_dxdt_t),
        (np.asarray(dxdt_tp) - np.asarray(dxdt_tm)) / (2.0 * dt),
        rtol=1e-5,
        atol=1e-25,
    )

    dr = 1.0e-6 * rho
    eps_rp, dxdt_rp = net.eval_net_jax(
        "basic.net",
        jnp.asarray(xin),
        jnp.asarray(temp),
        jnp.asarray(rho + dr),
        comp.abar,
        comp.zbar,
        comp.z2bar,
        comp.ye,
        eta=-2.0,
    )
    eps_rm, dxdt_rm = net.eval_net_jax(
        "basic.net",
        jnp.asarray(xin),
        jnp.asarray(temp),
        jnp.asarray(rho - dr),
        comp.abar,
        comp.zbar,
        comp.z2bar,
        comp.ye,
        eta=-2.0,
    )
    np.testing.assert_allclose(
        float(grad_eps[2]),
        (float(eps_rp) - float(eps_rm)) / (2.0 * dr),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(jac_dxdt_rho),
        (np.asarray(dxdt_rp) - np.asarray(dxdt_rm)) / (2.0 * dr),
        rtol=1e-5,
        atol=1e-25,
    )

    for j in range(info.num_isos):
        hx = 1.0e-7 * max(float(xin[j]), 1.0e-4)
        xp = xin.copy()
        xm = xin.copy()
        xp[j] += hx
        xm[j] -= hx
        eps_xp, dxdt_xp = net.eval_net_jax(
            "basic.net",
            jnp.asarray(xp),
            jnp.asarray(temp),
            jnp.asarray(rho),
            comp.abar,
            comp.zbar,
            comp.z2bar,
            comp.ye,
            eta=-2.0,
        )
        eps_xm, dxdt_xm = net.eval_net_jax(
            "basic.net",
            jnp.asarray(xm),
            jnp.asarray(temp),
            jnp.asarray(rho),
            comp.abar,
            comp.zbar,
            comp.z2bar,
            comp.ye,
            eta=-2.0,
        )
        fd_deps_xj = (float(eps_xp) - float(eps_xm)) / (2.0 * hx)
        if abs(float(grad_eps[0][j])) > 1.0e-6:
            np.testing.assert_allclose(
                float(grad_eps[0][j]), fd_deps_xj, rtol=1e-4
            )
        fd_dxdt_xj = (np.asarray(dxdt_xp) - np.asarray(dxdt_xm)) / (2.0 * hx)
        np.testing.assert_allclose(
            np.asarray(jac_dxdt_x)[:, j], fd_dxdt_xj, rtol=1e-5, atol=1e-13
        )


def test_net_one_zone_burn_and_const_p_burn_match_mesa_golden() -> None:
    """Test `burn_1_zone` and `burn_1_zone_const_p` on `approx21.net`."""
    info21 = net.get_net_info("approx21.net")
    idx21 = {name: i for i, name in enumerate(info21.iso_names)}

    x_burn = np.zeros(info21.num_isos, dtype=np.float64)
    x_burn[idx21["he4"]] = 1.0
    out_burn = net.burn_1_zone(
        "approx21.net",
        t_start=0.0,
        t_end=150.0,
        starting_x=x_burn,
        times=np.array([0.0, 100.0, 150.0], dtype=np.float64),
        log10_ts=np.array([8.3, 8.4, 8.3], dtype=np.float64),
        log10_rhos=np.array([7.0, 7.5, 8.1], dtype=np.float64),
        max_steps=50000,
        eps=1.0e-8,
        odescal=1.0e-12,
    )
    np.testing.assert_allclose(
        out_burn.ending_x[idx21["c12"]], 9.8714342254068388e-1, rtol=1e-6
    )
    np.testing.assert_allclose(
        out_burn.ending_x[idx21["he4"]], 1.2802186836379657e-2, rtol=1e-5
    )

    x21 = np.zeros(info21.num_isos, dtype=np.float64)
    x21[idx21["c12"]] = 0.37
    x21[idx21["o16"]] = 0.63
    out21 = net.burn_1_zone_const_p(
        "approx21.net",
        starting_temp=3.0e9,
        starting_x=x21,
        t_end=1.0e-4,
        pressure=10.0**24.0,
    )
    np.testing.assert_allclose(
        out21.ending_x[idx21["o16"]], 6.9559810130e-1, rtol=1e-9
    )
    np.testing.assert_allclose(
        out21.ending_x[idx21["si28"]], 2.0468825030e-1, rtol=1e-8
    )
    np.testing.assert_allclose(
        out21.ending_x[idx21["s32"]], 4.4368437194e-2, rtol=1e-8
    )
    np.testing.assert_allclose(
        out21.ending_x[idx21["mg24"]], 3.7858761619e-2, rtol=1e-8
    )

    for variant in ("rodasp", "ros3p", "ros3pl"):
        out_var = net.burn_1_zone_const_p(
            "approx21.net",
            starting_temp=3.0e9,
            starting_x=x21,
            t_end=1.0e-4,
            pressure=10.0**24.0,
            solver=variant,
        )
        np.testing.assert_allclose(
            out_var.ending_x[idx21["o16"]], 6.9559810130e-1, rtol=1e-3
        )
