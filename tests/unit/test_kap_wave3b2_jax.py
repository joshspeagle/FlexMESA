# SPDX-License-Identifier: LGPL-3.0-or-later
"""JAX autodiff, `kapCN`, and multi-zone batch parity tests for `kap`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flexmesa import kap
from flexmesa.kap import jax_co_lowt, jax_interp, jax_kernels


def test_jax_grad_and_jacfwd_derivatives_match_analytical_and_fd() -> None:
    """Verify `jax.grad` and `jax.jacfwd` on native JAX opacity kernels."""
    fn = kap.make_differentiable_kap(
        z=1.8000000000000016e-2,
        x=0.70,
        zbase=0.0,
        lnfree_e=-1.6965073622377105e-1,
        eta=-10.0,
        zbar=1.1018857434091647,
    )
    lr = jnp.asarray(-6.0, dtype=jnp.float64)
    lt = jnp.asarray(6.0, dtype=jnp.float64)
    lk_val = float(fn(lr, lt))
    grad_lr, grad_lt = jax.grad(fn, argnums=(0, 1))(lr, lt)
    jac_lr, jac_lt = jax.jacfwd(fn, argnums=(0, 1))(lr, lt)

    out = kap.kap_get_results(
        z=1.8000000000000016e-2,
        x=0.70,
        zbase=0.0,
        log_rho=-6.0,
        log_t=6.0,
        lnfree_e=-1.6965073622377105e-1,
        eta=-10.0,
        zbar=1.1018857434091647,
        backend="jax",
    )
    np.testing.assert_allclose(lk_val, out.log10_kap, rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(
        float(grad_lr), out.dlnkap_dln_rho, rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(
        float(grad_lt), out.dlnkap_dln_t, rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(
        float(jac_lr), out.dlnkap_dln_rho, rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(
        float(jac_lt), out.dlnkap_dln_t, rtol=1e-12, atol=0.0
    )

    # Direct autodiff through eval_fixed_metal_jax, eval_co_kap_jax, and
    # electron_conduction_jax with respect to (log_rho, log_t)
    lr_pt = jnp.asarray(-5.15, dtype=jnp.float64)
    lt_pt = jnp.asarray(6.12, dtype=jnp.float64)
    _, dr_f, dt_f = jax_interp.eval_fixed_metal_jax(0.015, 0.65, lr_pt, lt_pt)
    g_lr_f, g_lt_f = jax.grad(
        lambda r, t: jax_interp.eval_fixed_metal_jax(0.015, 0.65, r, t)[0],
        argnums=(0, 1),
    )(lr_pt, lt_pt)
    np.testing.assert_allclose(float(g_lr_f), float(dr_f), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(float(g_lt_f), float(dt_f), rtol=1e-12, atol=0.0)

    _, dr_co, dt_co = jax_co_lowt.eval_co_kap_jax(
        0.018, 0.2, 0.12, 0.08, lr_pt, lt_pt
    )
    g_lr_co, g_lt_co = jax.grad(
        lambda r, t: jax_co_lowt.eval_co_kap_jax(0.018, 0.2, 0.12, 0.08, r, t)[
            0
        ],
        argnums=(0, 1),
    )(lr_pt, lt_pt)
    np.testing.assert_allclose(
        float(g_lr_co), float(dr_co), rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(
        float(g_lt_co), float(dt_co), rtol=1e-12, atol=0.0
    )

    lr_ec = jnp.asarray(5.2, dtype=jnp.float64)
    lt_ec = jnp.asarray(7.9, dtype=jnp.float64)
    _, dr_ec, dt_ec = jax_kernels.electron_conduction_jax(1.5, lr_ec, lt_ec)
    g_lr_ec, g_lt_ec = jax.grad(
        lambda r, t: jnp.log10(
            jax_kernels.electron_conduction_jax(1.5, r, t)[0]
        ),
        argnums=(0, 1),
    )(lr_ec, lt_ec)
    np.testing.assert_allclose(
        float(g_lr_ec), float(dr_ec), rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(
        float(g_lt_ec), float(dt_ec), rtol=1e-12, atol=0.0
    )


def test_kapcn_and_multi_zone_batch_evaluation_parity() -> None:
    """Verify `kap_get_kapcn` knot accuracy and `kap_get_zones` parity."""
    for backend in ("rust", "jax"):
        kap_cn, _, _ = kap.kap_get_kapcn(
            zbase=0.02,
            x=0.70,
            fc=1.0,
            fn_val=1.0,
            log_rho=-10.2,
            log_t=3.6,
            backend=backend,
        )
        np.testing.assert_allclose(
            np.log10(float(kap_cn)), -3.361, rtol=1e-12, atol=0.0
        )

    chem_id = np.array([2, 7, 40, 53, 66], dtype=np.uintp)
    xa_row = np.array([0.70, 0.28, 0.005, 0.002, 0.013], dtype=np.float64)
    nz = 16
    xa_zones = np.tile(xa_row, (nz, 1))
    log_rho_zones = np.linspace(-6.0, -3.0, nz)
    log_t_zones = np.linspace(5.5, 6.5, nz)
    zeros = np.zeros(nz, dtype=np.float64)

    res_par = kap.kap_get_zones(
        chem_id,
        xa_zones,
        log_rho_zones,
        log_t_zones,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zbase=0.02,
        parallel=True,
    )
    res_seq = kap.kap_get_zones(
        chem_id,
        xa_zones,
        log_rho_zones,
        log_t_zones,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zbase=0.02,
        parallel=False,
    )
    np.testing.assert_allclose(res_par, res_seq, rtol=1e-14, atol=1e-15)
    for k in range(nz):
        single = kap.kap_get(
            chem_id,
            xa_zones[k],
            float(log_rho_zones[k]),
            float(log_t_zones[k]),
            zbase=0.02,
        )
        expected = np.array(
            [
                single.kap,
                single.dlnkap_dln_rho,
                single.dlnkap_dln_t,
                *single.kap_fracs,
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(res_par[k], expected, rtol=1e-14, atol=1e-15)

    z6 = (zeros, zeros, zeros, zeros, zeros, zeros)
    with pytest.raises(ValueError, match="mismatched zone"):
        kap.kap_get_zones(
            chem_id, xa_zones[: nz - 1], log_rho_zones, log_t_zones, *z6
        )
    with pytest.raises(ValueError, match="kap_get failed"):
        kap.kap_get_zones(
            np.array([0], dtype=np.uintp),
            np.ones((nz, 1), dtype=np.float64),
            log_rho_zones,
            log_t_zones,
            *z6,
        )

    # Verify make_differentiable_kap with AESOPUS low-T tables under @jax.jit
    fn_aes = kap.make_differentiable_kap(
        0.035, 0.70, 0.035, 0.0, 0.0, 0.0, kap_low_t_option="AESOPUS"
    )
    lk_aes, (dr_aes, dt_aes) = jax.value_and_grad(fn_aes, argnums=(0, 1))(
        jnp.float64(-8.4), jnp.float64(3.55)
    )
    ref_aes = kap.kap_get_results(
        0.035,
        0.70,
        0.035,
        log_rho=-8.4,
        log_t=3.55,
        kap_low_t_option="AESOPUS",
        backend="rust",
    )
    np.testing.assert_allclose(
        [float(10.0**lk_aes), float(dr_aes), float(dt_aes)],
        [ref_aes.kap, ref_aes.dlnkap_dln_rho, ref_aes.dlnkap_dln_t],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
