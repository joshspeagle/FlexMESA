# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit, golden parity, and JAX autodiff tests for `atm` and `net`."""

from __future__ import annotations

import pathlib
import struct

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import flexmesa
from flexmesa import _rust, atm, chem, constants, net


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
    """Verify `eval_t_tau` vs `eval_t_tau_jax` consistency and Hopf grad."""
    tau = 0.42
    teff = 5772.0
    with flexmesa.numerics_mode_context("strict"):
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
            np.testing.assert_allclose(
                lnt_jax, lnt_rust, rtol=1e-14, atol=1e-14
            )

            dq_dtau = atm.eval_t_tau_dq_dtau(rel_id, tau)
            dlnt_dtau_jax = float(
                jax.grad(
                    lambda t_val, r_id=rel_id: atm.eval_t_tau_jax(
                        r_id, t_val, teff
                    )
                )(jnp.asarray(tau, dtype=jnp.float64))
            )
            q_val = (4.0 / 3.0) * (np.exp(lnt_rust) / teff) ** 4 - tau
            dlnt_dtau_exact = (1.0 + dq_dtau) / (4.0 * (tau + q_val))
            rtol = 1e-4 if rel_id == atm.ATM_T_TAU_TRAMPEDACH_SOLAR else 1e-11
            np.testing.assert_allclose(
                dlnt_dtau_jax, dlnt_dtau_exact, rtol=rtol, atol=1e-12
            )


def test_atm_paczynski_gradr_rust_and_jax_consistency() -> None:
    """Verify `eval_paczynski_gradr` and `eval_paczynski_gradr_jax` match."""
    with flexmesa.numerics_mode_context("strict"):
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
    # under `fast` mode (`f64` basis terms, exact cubic calculus; `strict` mode
    # mirrors MESA's `f32` `fvbicub_db` basis terms).
    z_node = 0.0179
    eps = 1.0e-5
    with flexmesa.numerics_mode_context("fast"):
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
        irr_out.ln_t / constants.LN10, 3.1115766787614829, rtol=1e-12
    )
    np.testing.assert_allclose(irr_out.tau, 2.9101126947637423, rtol=1e-11)


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


def _hex_to_f64(tok: str) -> float:
    """Decode an IEEE-754 big-endian hex string (`0x...`) to `float`."""
    val: float = struct.unpack(">d", int(tok.strip(), 16).to_bytes(8, "big"))[0]
    return val


def _ulp_diff(a: float, b: float) -> int:
    """Compute IEEE-754 ULP distance between two `float` values."""
    if a == 0.0 and b == 0.0:
        return 0
    ua = int(struct.unpack(">Q", struct.pack(">d", a))[0])
    ub = int(struct.unpack(">Q", struct.pack(">d", b))[0])
    if (ua >> 63) != (ub >> 63):
        return 1 << 62
    return abs(ua - ub)


def _assert_unfloored_rel(
    actual: float, expected: float, rtol: float, ctx: str
) -> None:
    """Assert unfloored relative error `|actual - expected| / |expected|`."""
    if expected == 0.0:
        assert actual == 0.0, f"{ctx}: expected 0.0, got {actual}"
        return
    rel = abs(actual - expected) / abs(expected)
    assert rel <= rtol, f"{ctx}: rel={rel:.3e} > {rtol:.3e}"


def _atm_teff(l_val: float, r_val: float) -> float:
    """Evaluate blackbody `Teff` via `_rust.pow` matching `atm_Teff`."""
    denom = 4.0 * constants.PI * r_val * r_val * constants.BOLTZ_SIGMA
    return float(_rust.pow(l_val / denom, 0.25))


def test_f2_23_eval_t_tau_atm_and_eval_table_atm_modes_and_errors() -> None:
    """Verify `F2-23` `eval_t_tau_atm`, `eval_table_atm`, FD, and errors."""
    assert atm.get_t_tau_id("Eddington") == atm.ATM_T_TAU_EDDINGTON
    assert atm.get_table_id("photosphere") == atm.ATM_TABLE_PHOTOSPHERE

    with pytest.raises(ValueError, match="simple_photosphere"):
        atm.get_t_tau_id("simple_photosphere")
    with pytest.raises(ValueError):
        atm.get_t_tau_id("unknown_relation")
    with pytest.raises(ValueError):
        atm.get_table_id("unknown_table")

    cid, xa = _solar_species_and_xa(0.70, 0.02)
    cgrav = constants.STANDARD_CGRAV
    teff_sun = _atm_teff(constants.LSUN, constants.RSUN)

    with flexmesa.numerics_mode_context("strict"):
        # Verify `pextra_factor` monotonic radiation-pressure shift in `fixed`.
        f_p0 = atm.eval_t_tau_atm(
            2.0 / 3.0,
            constants.LSUN,
            constants.RSUN,
            constants.MSUN,
            cgrav,
            0.40,
            0.0,
            "Eddington",
            "fixed",
            1.0e-6,
            0,
            500,
            False,
            teff_sun,
            chem_id=cid,
            xa=xa,
        )
        f_p15 = atm.eval_t_tau_atm(
            2.0 / 3.0,
            constants.LSUN,
            constants.RSUN,
            constants.MSUN,
            cgrav,
            0.40,
            1.5,
            "Eddington",
            "fixed",
            1.0e-6,
            0,
            500,
            False,
            teff_sun,
            chem_id=cid,
            xa=xa,
        )
        assert f_p15.ln_p > f_p0.ln_p
        assert f_p0.kap == 0.40

        # Verify `'iterated'` implicit opacity partial fixup against FD.
        def _eval_iter(l_in: float, r_in: float, m_in: float) -> float:
            return atm.eval_t_tau_atm(
                2.0 / 3.0,
                l_in,
                r_in,
                m_in,
                cgrav,
                0.50,
                1.5,
                "solar_Hopf",
                "iterated",
                1.0e-11,
                50,
                500,
                False,
                _atm_teff(l_in, r_in),
                chem_id=cid,
                xa=xa,
            ).ln_p

        base_it = atm.eval_t_tau_atm(
            2.0 / 3.0,
            constants.LSUN,
            constants.RSUN,
            constants.MSUN,
            cgrav,
            0.50,
            1.5,
            "solar_Hopf",
            "iterated",
            1.0e-11,
            50,
            500,
            False,
            teff_sun,
            chem_id=cid,
            xa=xa,
        )
        assert base_it.dln_p_dln_kap == 0.0
        h = 1.0e-6
        dl = constants.LSUN * h
        fd_l = (
            _eval_iter(constants.LSUN + dl, constants.RSUN, constants.MSUN)
            - _eval_iter(constants.LSUN - dl, constants.RSUN, constants.MSUN)
        ) / (2.0 * dl)
        _assert_unfloored_rel(base_it.dln_p_dl, fd_l, 5.0e-6, "dlnP_dL")

        # Verify loud errors on invalid options, `tau_factor != 1.0`, off-table.
        with pytest.raises(ValueError, match="max_iters > 0"):
            atm.eval_t_tau_atm(
                2.0 / 3.0,
                constants.LSUN,
                constants.RSUN,
                constants.MSUN,
                cgrav,
                0.40,
                1.0,
                "Eddington",
                "iterated",
                1.0e-6,
                0,
                500,
                False,
                teff_sun,
                chem_id=cid,
                xa=xa,
            )
        with pytest.raises(ValueError, match="tau_factor"):
            atm.eval_table_atm(
                constants.LSUN,
                constants.RSUN,
                constants.MSUN,
                cgrav,
                "photosphere",
                0.02,
                tau_factor=2.0,
            )
        l_hot = constants.LSUN * 1.0e5
        with pytest.raises(ValueError, match="atm_off_table_option"):
            atm.eval_table_atm(
                l_hot,
                constants.RSUN,
                constants.MSUN,
                cgrav,
                "tau_100",
                0.02,
                atm_off_table_option="",
                teff=_atm_teff(l_hot, constants.RSUN),
                chem_id=cid,
                xa=xa,
            )


def test_mesa_26_4_1_atm_oracles_match_in_strict_mode() -> None:
    """Verify all `234` rows of MESA 26.4.1 `atm_oracles` in `strict` mode."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    oracles_dir = repo_root / "benchmarks" / "fortran_mesa" / "atm_oracles"
    forbidden = (
        "/" + "home/",
        "/" + "tmp/",
        "/" + "mnt/",
        "." + "venvs",
        "file" + "://",
        "\u2014",
    )
    for rel_name in (
        "Makefile",
        "README.md",
        "gen_atm_oracles.f90",
        "mesa_26_4_1_atm_oracles.tsv",
    ):
        text = (oracles_dir / rel_name).read_text(encoding="utf-8")
        for tok in forbidden:
            assert tok not in text, f"Forbidden token {tok!r} in {rel_name}"

    tsv_text = (oracles_dir / "mesa_26_4_1_atm_oracles.tsv").read_text(
        encoding="utf-8"
    )
    for header in (
        "# mesa_version: r26.4.1",
        "# mesa_git_commit: c8c202662922a22b3efefa824473ad615c623afd",
        "# mesasdk_version: x86_64-linux-26.6.1",
        "# math_backend: CRMATH",
    ):
        assert header in tsv_text

    groups: dict[
        tuple[str, str, str, str, str, str, str], dict[str, float]
    ] = {}
    row_count = 0
    for line in tsv_text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("row_id\t"):
            continue
        cols = line.split("\t")
        key = (cols[1], cols[2], cols[3], cols[4], cols[5], cols[6], cols[7])
        groups.setdefault(key, {})[cols[8]] = _hex_to_f64(cols[9])
        row_count += 1

    assert row_count == 234

    with flexmesa.numerics_mode_context("strict"):
        for (sec, lbl, x1, x2, x3, x4, x5), outs in groups.items():
            v1, v2, v3, v4, v5 = (
                _hex_to_f64(x1),
                _hex_to_f64(x2),
                _hex_to_f64(x3),
                _hex_to_f64(x4),
                _hex_to_f64(x5),
            )
            if sec == "t_tau_rel":
                rel_id = round(v1)
                if "tau_base" in outs:
                    assert (
                        _ulp_diff(atm.get_t_tau_base(rel_id), outs["tau_base"])
                        == 0
                    )
                else:
                    assert (
                        _ulp_diff(atm.eval_t_tau(rel_id, v2, v3), outs["lnT"])
                        == 0
                    )
                    assert (
                        _ulp_diff(
                            atm.eval_t_tau_dq_dtau(rel_id, v2), outs["dq_dtau"]
                        )
                        == 0
                    )
            elif sec == "e2":
                e2, de2 = _rust.atm_eval_e2(v1)
                assert _ulp_diff(float(e2), outs["e2"]) == 0
                assert _ulp_diff(float(de2), outs["de2_dx"]) == 0
            elif sec == "paczynski":
                gradr = atm.eval_paczynski_gradr(
                    v1,
                    v2,
                    v3,
                    v4,
                    v5,
                    constants.LSUN,
                    constants.MSUN,
                    constants.RSUN,
                    constants.STANDARD_CGRAV,
                )
                assert _ulp_diff(gradr, outs["gradr"]) == 0
            elif sec == "t_tau_atm":
                p0, rel_name, op_mode = lbl.split(":")
                cid, xa = _solar_species_and_xa(0.70, 0.02)
                kap_g = {
                    "fixed": 0.35 if "solar" in p0 else 0.34,
                    "iterated": 0.50,
                    "varying": 0.40,
                }[op_mode]
                errtol = 1.0e-9 if op_mode == "varying" else 1.0e-6
                res = atm.eval_t_tau_atm(
                    v1,
                    v3,
                    v4,
                    v5,
                    constants.STANDARD_CGRAV,
                    kap_g,
                    v2,
                    rel_name,
                    op_mode,
                    errtol,
                    30,
                    500,
                    False,
                    _atm_teff(v3, v4),
                    chem_id=cid,
                    xa=xa,
                    zbase=0.02,
                )
                got_map = {
                    "lnT": res.ln_t,
                    "dlnT_dL": res.dln_t_dl,
                    "dlnT_dlnR": res.dln_t_dln_r,
                    "dlnT_dlnM": res.dln_t_dln_m,
                    "dlnT_dlnkap": res.dln_t_dln_kap,
                    "lnP": res.ln_p,
                    "dlnP_dL": res.dln_p_dl,
                    "dlnP_dlnR": res.dln_p_dln_r,
                    "dlnP_dlnM": res.dln_p_dln_m,
                    "dlnP_dlnkap": res.dln_p_dln_kap,
                    "kap": res.kap,
                }
                for k, exp_v in outs.items():
                    if op_mode in ("fixed", "iterated") or k in (
                        "lnT",
                        "dlnT_dL",
                        "dlnT_dlnR",
                        "dlnT_dlnM",
                        "dlnT_dlnkap",
                        "dlnP_dlnkap",
                        "kap",
                    ):
                        assert _ulp_diff(got_map[k], exp_v) == 0, f"{lbl}.{k}"
                    elif k == "lnP":
                        _assert_unfloored_rel(
                            got_map[k], exp_v, 5.0e-12, f"{lbl}.{k}"
                        )
                    else:
                        _assert_unfloored_rel(
                            got_map[k], exp_v, 1.0e-7, f"{lbl}.{k}"
                        )
            elif sec == "table":
                tid = atm.get_table_id(lbl)
                res = atm.eval_table(
                    v1, v2, v3, constants.STANDARD_CGRAV, tid, v4, False, v5
                )
                got_map = {
                    "lnT": res.ln_t,
                    "dlnT_dL": res.dln_t_dl,
                    "dlnT_dlnR": res.dln_t_dln_r,
                    "dlnT_dlnM": res.dln_t_dln_m,
                    "dlnT_dlnkap": res.dln_t_dln_kap,
                    "lnP": res.ln_p,
                    "dlnP_dL": res.dln_p_dl,
                    "dlnP_dlnR": res.dln_p_dln_r,
                    "dlnP_dlnM": res.dln_p_dln_m,
                    "dlnP_dlnkap": res.dln_p_dln_kap,
                    "kap": res.kap,
                }
                for k, exp_v in outs.items():
                    _assert_unfloored_rel(
                        got_map[k], exp_v, 5.0e-14, f"{lbl}.{k}"
                    )
            elif sec == "table_ab":
                tid = (
                    atm.ATM_TABLE_PHOTOSPHERE
                    if "photosphere" in lbl
                    else atm.ATM_TABLE_TAU_100
                )
                teff = v4 if v4 > 0.0 else _atm_teff(v1, v2)
                a, b = atm.get_table_alfa_beta(
                    v1, teff, v2, v3, constants.STANDARD_CGRAV, tid
                )
                assert _ulp_diff(a, outs["alfa"]) == 0
                assert _ulp_diff(b, outs["beta"]) == 0
                if "tau_base" in outs:
                    assert (
                        _ulp_diff(atm.get_table_base(tid), outs["tau_base"])
                        == 0
                    )
            elif sec == "table_blend":
                off_rel = "Eddington" if "edge" in lbl else "solar_Hopf"
                cid, xa = _solar_species_and_xa(0.70, v4)
                res, tbase = atm.eval_table_atm(
                    v1,
                    v2,
                    v3,
                    constants.STANDARD_CGRAV,
                    "WD_tau_25",
                    v4,
                    atm_off_table_option="T_tau",
                    t_tau_relation=off_rel,
                    t_tau_opacity="fixed",
                    kap_guess=v5,
                    pextra_factor=1.0,
                    errtol=1.0e-6,
                    max_iters=0,
                    max_steps=500,
                    skip_partials=False,
                    teff=_atm_teff(v1, v2),
                    chem_id=cid,
                    xa=xa,
                    zbase=v4,
                )
                assert _ulp_diff(tbase, outs["kap"]) == 0
                got_map = {
                    "lnT": res.ln_t,
                    "dlnT_dL": res.dln_t_dl,
                    "dlnT_dlnR": res.dln_t_dln_r,
                    "dlnT_dlnM": res.dln_t_dln_m,
                    "dlnT_dlnkap": res.dln_t_dln_kap,
                    "lnP": res.ln_p,
                    "dlnP_dL": res.dln_p_dl,
                    "dlnP_dlnR": res.dln_p_dln_r,
                    "dlnP_dlnM": res.dln_p_dln_m,
                    "dlnP_dlnkap": res.dln_p_dln_kap,
                }
                for k, exp_v in outs.items():
                    if k == "kap":
                        continue
                    if "pure_off_table" in lbl:
                        assert _ulp_diff(got_map[k], exp_v) == 0, f"{lbl}.{k}"
                    else:
                        _assert_unfloored_rel(
                            got_map[k], exp_v, 5.0e-14, f"{lbl}.{k}"
                        )
            elif sec == "irradiated":
                cid, xa = _solar_species_and_xa(0.70, 0.01)
                irr = atm.eval_irradiated(
                    v1,
                    v2,
                    v3,
                    constants.STANDARD_CGRAV,
                    v4,
                    1.0e6,
                    1.5e-2,
                    4.0e-3,
                    v5,
                    1.0e-6,
                    30,
                    False,
                    chem_id=cid,
                    xa=xa,
                    zbase=0.01,
                    low_t_option="lowT_Freedman11",
                )
                got_map = {
                    "kap": irr.kap,
                    "tau": irr.tau,
                    "lnT": irr.ln_t,
                    "dlnT_dL": irr.dln_t_dl,
                    "dlnT_dlnR": irr.dln_t_dln_r,
                    "dlnT_dlnM": irr.dln_t_dln_m,
                    "dlnT_dlnkap": irr.dln_t_dln_kap,
                }
                for k, exp_v in outs.items():
                    _assert_unfloored_rel(
                        got_map[k], exp_v, 1.0e-14, f"{lbl}.{k}"
                    )
