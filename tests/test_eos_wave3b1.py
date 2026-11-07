# SPDX-License-Identifier: LGPL-3.0-or-later
"""Python and JAX tests for Wave 3b.1 4D tabulated EOS and `(P, T)` solvers."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import chem, constants, eos


def _solar_composition() -> tuple[list[int], np.ndarray]:
    """Build 7-species `(X=0.70, Z=0.02)` composition from `eos_support.f90`."""
    names = ["h1", "he4", "c12", "n14", "o16", "ne20", "mg24"]
    cids = []
    for n in names:
        idx = chem.get_nuclide_index(n)
        assert idx is not None
        cids.append(idx)
    zinit = 0.02
    x = 0.70
    y = 1.0 - (x + zinit)
    xa = np.zeros(7, dtype=np.float64)
    xa[0] = x
    xa[1] = y
    xa[2] = zinit * 0.173312
    xa[3] = zinit * 0.053177
    xa[4] = zinit * 0.482398
    xa[5] = zinit * 0.098675
    xa[6] = 1.0 - float(np.sum(xa[:6]))
    return cids, xa


def test_eos_dt_get_4d_tables_match_fortran_golden() -> None:
    """Verify `eos_dt_get` in Rust and JAX matches `eos/test/test_output`."""
    cids, xa = _solar_composition()
    for rho, temp, p_gold, frac_idx in [
        (1.0e-2, 1.0e6, 1.3329474231512012e12, eos.I_FRAC_FREE_EOS),
        (1.0e-1, 1.0e4, 5.4349014621238190e10, eos.I_FRAC_OPAL_SCVH),
    ]:
        out_r = eos.eos_dt_get(rho, temp, cids, xa, backend="rust")
        out_j = eos.eos_dt_get(
            jnp.asarray(rho, dtype=jnp.float64),
            jnp.asarray(temp, dtype=jnp.float64),
            cids,
            jnp.asarray(xa),
            backend="jax",
        )
        np.testing.assert_allclose(
            out_r.res, np.asarray(out_j.res), rtol=1.0e-14
        )
        np.testing.assert_allclose(
            out_r.d_dlnd, np.asarray(out_j.d_dlnd), rtol=1.0e-14
        )
        np.testing.assert_allclose(
            out_r.d_dln_t, np.asarray(out_j.d_dln_t), rtol=1.0e-14
        )
        prad = constants.CRAD * (temp**4) / 3.0
        p_tot = float(np.exp(out_r.res[eos.I_LN_PGAS])) + prad
        np.testing.assert_allclose(p_tot, p_gold, rtol=1.0e-12)
        np.testing.assert_allclose(
            float(out_r.res[frac_idx]), 1.0, rtol=1.0e-14
        )


def test_standalone_opal_scvh_free_eos_and_cms_jax_jvp() -> None:
    """Verify standalone `OPAL/SCVH`, `FreeEOS`, and `CMS` evaluators/JVPs."""
    cids, xa = _solar_composition()
    for eval_fn, frac_idx in [
        (eos.eval_opal_scvh_eos, eos.I_FRAC_OPAL_SCVH),
        (eos.eval_free_eos, eos.I_FRAC_FREE_EOS),
        (eos.eval_cms_eos, eos.I_FRAC_CMS),
    ]:
        out = eval_fn(1.0e-2, 1.0e5, cids, xa, backend="rust")
        assert float(out.res[frac_idx]) == 1.0

        def _ln_p(r: jax.Array, t: jax.Array, fn: Any = eval_fn) -> jax.Array:
            res: jax.Array = fn(r, t, cids, jnp.asarray(xa), backend="jax").res
            return res[eos.I_LN_PGAS]

        r0 = jnp.asarray(1.0e-2, dtype=jnp.float64)
        t0 = jnp.asarray(1.0e5, dtype=jnp.float64)
        dr = jnp.asarray(1.0e-4, dtype=jnp.float64)
        dt = jnp.asarray(1.0e3, dtype=jnp.float64)
        _, dln_p = jax.jvp(_ln_p, (r0, t0), (dr, dt))
        expected = float(out.d_dlnd[eos.I_LN_PGAS]) * (1.0e-4 / 1.0e-2) + float(
            out.d_dln_t[eos.I_LN_PGAS]
        ) * (1.0e3 / 1.0e5)
        np.testing.assert_allclose(float(dln_p), expected, rtol=1.0e-13)

        # Central finite-difference oracle along the same perturbation direction
        eps = 1.0e-4
        f_plus = eval_fn(
            1.0e-2 + eps * 1.0e-4, 1.0e5 + eps * 1.0e3, cids, xa, backend="rust"
        ).res[eos.I_LN_PGAS]
        f_minus = eval_fn(
            1.0e-2 - eps * 1.0e-4, 1.0e5 - eps * 1.0e3, cids, xa, backend="rust"
        ).res[eos.I_LN_PGAS]
        fd_jvp = float(f_plus - f_minus) / (2.0 * eps)
        np.testing.assert_allclose(float(dln_p), fd_jvp, rtol=2.0e-5)


def test_eos_pt_get_and_inversions_match_fortran_golden() -> None:
    """Verify `eos_pt_get`, `eos_dt_get_*`, and `eos_pt_get_*` solvers."""
    cids, xa = _solar_composition()
    pt = eos.eos_pt_get(10.0**4.2, 10.0**3.0, cids, xa, backend="rust")
    np.testing.assert_allclose(
        float(pt.log10_rho), -6.3454687472634816, rtol=1.0e-12
    )
    assert float(pt.eos.res[eos.I_FRAC_OPAL_SCVH]) == 1.0

    # Verify JAX IFT custom_jvp on eos_pt_get against central finite differences
    def _lrho_fn(pg: jax.Array, t: jax.Array) -> jax.Array:
        return jnp.asarray(
            eos.eos_pt_get(
                pg, t, cids, jnp.asarray(xa), backend="jax"
            ).log10_rho
        )

    p0_f = 10.0**4.2
    t0_f = 10.0**3.0
    p0 = jnp.asarray(p0_f, dtype=jnp.float64)
    t0 = jnp.asarray(t0_f, dtype=jnp.float64)
    dp = p0 * 1.0e-3
    dt = t0 * 2.0e-3
    _, dlrho = jax.jvp(_lrho_fn, (p0, t0), (dp, dt))
    eps = 1.0e-3
    lr_plus = eos.eos_pt_get(
        p0_f * (1.0 + eps * 1.0e-3),
        t0_f * (1.0 + eps * 2.0e-3),
        cids,
        xa,
        backend="rust",
    ).log10_rho
    lr_minus = eos.eos_pt_get(
        p0_f * (1.0 - eps * 1.0e-3),
        t0_f * (1.0 - eps * 2.0e-3),
        cids,
        xa,
        backend="rust",
    ).log10_rho
    fd_dlrho = float(lr_plus - lr_minus) / (2.0 * eps)
    np.testing.assert_allclose(float(dlrho), fd_dlrho, rtol=1.0e-5)

    # Verify eos_pt_get_t and eos_pt_get_pgas round-trip in HELM regime
    pt_helm = eos.eos_pt_get(1.0e15, 10.0**7.2, cids, xa, backend="rust")
    target_lne = float(pt_helm.eos.res[eos.I_LN_E])
    inv_t = eos.eos_pt_get_t(
        1.0e15, eos.I_LN_E, target_lne, cids, xa, log_t_guess=7.0
    )
    np.testing.assert_allclose(float(inv_t.log10_result), 7.2, rtol=1.0e-5)

    inv_pgas = eos.eos_pt_get_pgas(
        10.0**7.2, eos.I_LN_E, target_lne, cids, xa, log_pgas_guess=14.8
    )
    np.testing.assert_allclose(float(inv_pgas.log10_result), 15.0, rtol=1.0e-5)
