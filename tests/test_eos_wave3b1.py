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
            out_r.res, np.asarray(out_j.res), rtol=1.0e-13
        )
        np.testing.assert_allclose(
            out_r.d_dlnd, np.asarray(out_j.d_dlnd), rtol=1.0e-11
        )
        np.testing.assert_allclose(
            out_r.d_dln_t, np.asarray(out_j.d_dln_t), rtol=1.0e-11
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


def _make_xz_composition(x: float, z: float) -> tuple[list[int], np.ndarray]:
    """Build 7-species composition for arbitrary `(X, Z)`."""
    cids, _ = _solar_composition()
    y = max(0.0, 1.0 - (x + z))
    xa = np.array(
        [
            x,
            y,
            z * 0.173312,
            z * 0.053177,
            z * 0.482398,
            z * 0.098675,
            0.0,
        ],
        dtype=np.float64,
    )
    xa[6] = max(0.0, 1.0 - float(np.sum(xa[:6])))
    return cids, xa


def test_4d_tables_and_eos_dt_get_across_stellar_profiles_and_autodiff() -> (
    None
):
    """Verify full 4D tables & `eos_dt_get` across 4 stellar profiles and AD."""
    # 1. Off-golden (X, Z, logRho, logT) slices across OPAL/SCVH, FreeEOS, CMS
    for x_val, z_val, log_rho, log_t in [
        (0.72, 0.015, -3.25, 4.35),
        (0.55, 0.025, -1.15, 5.45),
        (0.35, 0.030, 0.45, 6.15),
    ]:
        cids, xa = _make_xz_composition(x_val, z_val)
        rho = 10.0**log_rho
        temp = 10.0**log_t
        for fn in (
            eos.eval_opal_scvh_eos,
            eos.eval_free_eos,
            eos.eval_cms_eos,
        ):
            out_r = fn(rho, temp, cids, xa, backend="rust")
            out_j = fn(
                jnp.float64(rho),
                jnp.float64(temp),
                cids,
                jnp.asarray(xa),
                backend="jax",
            )
            np.testing.assert_allclose(
                out_r.res, np.asarray(out_j.res), rtol=1.0e-12, atol=1.0e-12
            )
            np.testing.assert_allclose(
                out_r.d_dlnd,
                np.asarray(out_j.d_dlnd),
                rtol=1.0e-10,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                out_r.d_dln_t,
                np.asarray(out_j.d_dln_t),
                rtol=1.0e-10,
                atol=1.0e-12,
            )

    # 2. Four stellar interior profiles: 1 Msun ZAMS, 0.1 Msun BD, 0.6 Msun WD,
    #    and 25 Msun massive star across 100% of sampled zones
    cids_sol, xa_sol = _solar_composition()
    cids_wd, xa_wd = _make_xz_composition(0.0, 1.0)
    profiles = [
        # 1 Msun ZAMS: photosphere to core
        (
            cids_sol,
            xa_sol,
            [
                (-6.8, 3.76),
                (-4.2, 4.45),
                (-1.5, 5.75),
                (0.5, 6.65),
                (2.18, 7.19),
            ],
        ),
        # 0.1 Msun Brown Dwarf: cool molecular envelope to degenerate H/He core
        (
            cids_sol,
            xa_sol,
            [
                (-5.0, 3.35),
                (-2.4, 4.10),
                (-0.2, 4.95),
                (1.6, 5.70),
                (2.85, 6.45),
            ],
        ),
        # 0.6 Msun C/O White Dwarf: non-degenerate envelope to degenerate core
        (
            cids_wd,
            xa_wd,
            [(-3.0, 4.50), (0.0, 5.60), (2.5, 6.40), (4.8, 6.95), (6.5, 7.15)],
        ),
        # 25 Msun Massive Star: radiation-dominated envelope to convective core
        (
            cids_sol,
            xa_sol,
            [
                (-7.5, 4.45),
                (-5.2, 5.30),
                (-3.0, 6.20),
                (-0.8, 7.05),
                (0.65, 7.58),
            ],
        ),
    ]
    for cids_p, xa_p, pts in profiles:
        for lr, lt in pts:
            r_v, t_v = 10.0**lr, 10.0**lt
            dr = eos.eos_dt_get(r_v, t_v, cids_p, xa_p, backend="rust")
            dj = eos.eos_dt_get(
                jnp.float64(r_v),
                jnp.float64(t_v),
                cids_p,
                jnp.asarray(xa_p),
                backend="jax",
            )
            assert not np.any(np.isnan(np.asarray(dr.res)))
            np.testing.assert_allclose(
                dr.res[:3], np.asarray(dj.res[:3]), rtol=1.0e-12, atol=1.0e-12
            )
            np.testing.assert_allclose(
                dr.res, np.asarray(dj.res), rtol=1.0e-8, atol=1.0e-10
            )

    # 3. Verify jax.jacfwd on 4D table and eos_dt_get at off-golden point
    r_ad = jnp.float64(10.0 ** (-1.15))
    t_ad = jnp.float64(10.0**5.45)
    out_ad = eos.eos_dt_get(
        r_ad, t_ad, cids_sol, jnp.asarray(xa_sol), backend="jax"
    )
    jac_r, jac_t = jax.jacfwd(
        lambda r, t: jnp.asarray(
            eos.eos_dt_get(
                r, t, cids_sol, jnp.asarray(xa_sol), backend="jax"
            ).res
        ),
        argnums=(0, 1),
    )(r_ad, t_ad)
    np.testing.assert_allclose(
        np.asarray(out_ad.d_dlnd[:16]),
        np.asarray(jac_r * r_ad)[:16],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(out_ad.d_dln_t[:16]),
        np.asarray(jac_t * t_ad)[:16],
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    # 4. Verify @jax.jit compilation and JAX-first evaluation at unvisited slice
    cids_unvis, xa_unvis = _make_xz_composition(0.15, 0.08)
    cms_j = eos.eval_cms_eos(
        jnp.float64(1.0),
        jnp.float64(1.0e5),
        cids_unvis,
        jnp.asarray(xa_unvis),
        backend="jax",
    )
    cms_r = eos.eval_cms_eos(1.0, 1.0e5, cids_unvis, xa_unvis, backend="rust")
    np.testing.assert_allclose(
        np.asarray(cms_j.res[:4]), cms_r.res[:4], rtol=1.0e-12, atol=1.0e-12
    )
    jit_dt = jax.jit(
        lambda r, t: (
            eos.eos_dt_get(
                r, t, cids_sol, jnp.asarray(xa_sol), backend="jax"
            ).res
        )
    )
    np.testing.assert_allclose(
        np.asarray(jit_dt(r_ad, t_ad)),
        np.asarray(out_ad.res),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
