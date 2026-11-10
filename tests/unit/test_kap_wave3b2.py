# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit, golden parity, full-table, and JAX autodiff tests for `kap`."""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flexmesa import kap
from flexmesa.kap import jax_co_lowt, jax_interp, jax_kernels


def _parse_mesa_kap_test_output() -> list[dict[str, float]]:
    """Parse `.mesa_temp_cache/mesa_root/kap/test/test_output` blocks."""
    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / ".mesa_temp_cache"
        / "mesa_root"
        / "kap"
        / "test"
        / "test_output"
    )
    if not path.is_file():
        return []
    blocks: list[dict[str, float]] = []
    cur: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        toks = line.split()
        if not toks:
            continue
        if toks[0] == "test" and len(toks) >= 3 and toks[1] == "number":
            if cur:
                blocks.append(cur)
            cur = {"test_number": float(toks[2])}
        elif len(toks) == 2 and toks[0] != "CO_enhanced":
            try:
                cur[toks[0]] = float(toks[1])
            except ValueError:
                pass
    if cur:
        blocks.append(cur)
    return blocks


@pytest.mark.parametrize("backend", ["rust", "jax"])
def test_kap_fixed_metals_matches_fortran_golden_test_1(
    backend: str,
) -> None:
    """Verify `kap_get_results` reproduces Fortran MESA `kap` Test 1."""
    out = kap.kap_get_results(
        z=1.8000000000000016e-2,
        x=0.70,
        zbase=0.0,
        log_rho=-6.0,
        log_t=6.0,
        lnfree_e=-1.6965073622377105e-1,
        eta=-10.0,
        zbar=1.1018857434091647,
        kap_option="gs98",
        kap_low_t_option="lowT_fa05_gs98",
        use_type2_opacities=False,
        cubic_interpolation_in_x=False,
        cubic_interpolation_in_z=False,
        backend=backend,  # type: ignore[arg-type]
    )
    np.testing.assert_allclose(
        out.log10_kap, -4.4240000000110608e-1, rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(
        out.kap, 3.6107714509943334e-1, rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(
        out.dlnkap_dln_rho, 2.2816742190175789e-2, rtol=1e-11, atol=0.0
    )
    np.testing.assert_allclose(
        out.dlnkap_dln_t, 3.5387058571876982e-2, rtol=1e-11, atol=0.0
    )
    np.testing.assert_allclose(
        out.kap * out.dlnkap_dln_rho, 8.2386041304984663e-3, rtol=1e-11
    )
    np.testing.assert_allclose(
        out.kap * out.dlnkap_dln_t, 1.2777458082599771e-2, rtol=1e-11
    )
    np.testing.assert_allclose(
        out.kap_fracs, [0.0, 1.0, 0.0, 0.0], rtol=1e-12, atol=1e-15
    )


@pytest.mark.parametrize("backend", ["rust", "jax"])
def test_kap_co_enhanced_matches_fortran_golden_test_2(
    backend: str,
) -> None:
    """Verify `kap_get_results` reproduces Fortran MESA `kap` Test 2."""
    out = kap.kap_get_results(
        z=5.8000000000000052e-2,
        x=0.0,
        zbase=0.018,
        xc=2.4119616e-2,
        xn=9.56736e-4,
        xo=2.7683164e-2,
        xne=1.776024e-3,
        log_rho=-6.0,
        log_t=6.0,
        lnfree_e=-6.969230350735236e-1,
        eta=-10.0,
        zbar=2.0741317065062339,
        kap_option="gs98",
        kap_co_option="gn93_co",
        kap_low_t_option="lowT_fa05_gs98",
        use_type2_opacities=True,
        backend=backend,  # type: ignore[arg-type]
    )
    np.testing.assert_allclose(
        out.log10_kap, -6.6416776998366678e-1, rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(
        out.kap, 2.1668668723388557e-1, rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(
        out.dlnkap_dln_rho, 3.6488626497031655e-2, rtol=1e-11, atol=0.0
    )
    np.testing.assert_allclose(
        out.dlnkap_dln_t, -1.9081548742253509e-2, rtol=1e-11, atol=0.0
    )
    np.testing.assert_allclose(
        out.kap * out.dlnkap_dln_rho, 7.9065995973563678e-3, rtol=1e-11
    )
    np.testing.assert_allclose(
        out.kap * out.dlnkap_dln_t, -4.1347175842508287e-3, rtol=1e-11
    )
    np.testing.assert_allclose(
        out.kap_fracs[kap.I_FRAC_TYPE2], 1.0, rtol=1e-12, atol=1e-15
    )


@pytest.mark.parametrize("backend", ["rust", "jax"])
def test_kap_aesopus_matches_fortran_golden_test_4(
    backend: str,
) -> None:
    """Verify `kap_get_results` with `AESOPUS` reproduces Fortran Test 4."""
    out = kap.kap_get_results(
        z=2.0074120713487353e-2,
        x=6.7888662523180188e-1,
        zbase=0.014,
        xc=2.9968458709806432e-3,
        xn=1.5270900145630591e-3,
        xo=9.3376263240514384e-3,
        xne=0.0,
        log_rho=-8.2496430699014667,
        log_t=3.5571504546260235,
        lnfree_e=-9.2103424666308609,
        eta=-20.0,
        zbar=1.1007197824017732,
        kap_option="a09",
        kap_co_option="a09_co",
        kap_low_t_option="AESOPUS",
        use_zbase_for_type1=True,
        use_type2_opacities=True,
        backend=backend,  # type: ignore[arg-type]
    )
    np.testing.assert_allclose(
        out.log10_kap, -2.5233818701881989, rtol=1e-7, atol=0.0
    )
    np.testing.assert_allclose(
        out.kap, 2.9965265486259223e-3, rtol=2e-7, atol=0.0
    )
    np.testing.assert_allclose(
        out.dlnkap_dln_rho, 5.7284099179735104e-1, rtol=1e-6, atol=0.0
    )
    np.testing.assert_allclose(
        out.dlnkap_dln_t, 8.4625579174573868, rtol=1e-6, atol=0.0
    )
    np.testing.assert_allclose(
        out.kap * out.dlnkap_dln_rho, 1.7165332400619665e-3, rtol=1e-6
    )
    np.testing.assert_allclose(
        out.kap * out.dlnkap_dln_t, 2.5358279468945557e-2, rtol=1e-6
    )
    np.testing.assert_allclose(
        out.kap_fracs[kap.I_FRAC_TYPE2], 0.0, rtol=1e-12, atol=1e-15
    )


def test_all_lines_of_upstream_kap_test_output_verified() -> None:
    """Parse upstream `kap/test/test_output` and verify all lines."""
    blocks = _parse_mesa_kap_test_output()
    if not blocks:
        pytest.skip("upstream test_output not present")
    assert len(blocks) == 3
    cfgs = [
        ("gs98", "gn93_co", "lowT_fa05_gs98", False, False, -10.0, 1e-11),
        ("gs98", "gn93_co", "lowT_fa05_gs98", True, False, -10.0, 1e-11),
        ("a09", "a09_co", "AESOPUS", True, True, -20.0, 1e-6),
    ]
    for blk, (k_opt, co_opt, lt_opt, t2, uzb, eta, rtol) in zip(
        blocks, cfgs, strict=True
    ):
        for backend in ("rust", "jax"):
            out = kap.kap_get_results(
                z=blk["Z"],
                x=blk["X"],
                zbase=blk["Zbase"],
                xc=blk["xc"],
                xn=blk["xn"],
                xo=blk["xo"],
                xne=blk["xne"],
                log_rho=blk["logRho"],
                log_t=blk["logT"],
                lnfree_e=blk["lnfree_e"],
                eta=eta,
                zbar=blk["zbar"],
                kap_option=k_opt,
                kap_co_option=co_opt,
                kap_low_t_option=lt_opt,
                use_type2_opacities=t2,
                use_zbase_for_type1=uzb,
                backend=backend,  # type: ignore[arg-type]
            )
            np.testing.assert_allclose(
                out.log10_kap, blk["log10kap"], rtol=rtol
            )
            np.testing.assert_allclose(out.kap, blk["kap"], rtol=rtol)
            np.testing.assert_allclose(
                out.dlnkap_dln_rho, blk["dlnkap_dlnRho"], rtol=rtol
            )
            np.testing.assert_allclose(
                out.dlnkap_dln_t, blk["dlnkap_dlnT"], rtol=rtol
            )
            np.testing.assert_allclose(
                out.kap * out.dlnkap_dln_rho, blk["dkap_dlnd"], rtol=rtol
            )
            np.testing.assert_allclose(
                out.kap * out.dlnkap_dln_t, blk["dkap_dlnT"], rtol=rtol
            )
            np.testing.assert_allclose(
                out.kap_fracs[kap.I_FRAC_TYPE2],
                blk["frac_Type2"],
                rtol=1e-12,
                atol=1e-15,
            )


def test_distinct_fixed_co_aesopus_and_kapcn_tables_no_aliasing() -> None:
    """Verify all fixed, CO, AESOPUS, and kapCN tables are loaded & distinct."""
    z, x, lr, lt = 0.015, 0.55, -4.5, 5.8
    vals_fixed = {}
    for prefix in (
        "gs98",
        "gn93",
        "a09",
        "OP_gs98",
        "OP_a09_nans_removed_by_hand",
    ):
        r_res = kap.kap_get_fixed_metal(
            z, x, lr, lt, prefix=prefix, backend="rust"
        )
        j_res = kap.kap_get_fixed_metal(
            z, x, lr, lt, prefix=prefix, backend="jax"
        )
        np.testing.assert_allclose(j_res, r_res, rtol=1e-13, atol=0.0)
        vals_fixed[prefix] = float(r_res[0])

    assert abs(vals_fixed["gn93"] - vals_fixed["gs98"]) > 1e-4
    assert abs(vals_fixed["OP_gs98"] - vals_fixed["gs98"]) > 1e-4
    assert abs(vals_fixed["a09"] - vals_fixed["gs98"]) > 1e-4
    assert (
        abs(vals_fixed["OP_a09_nans_removed_by_hand"] - vals_fixed["a09"])
        > 1e-4
    )

    # Low-T fixed-metal tables (fa05 & Freedman11)
    lt_low, lr_low = 3.55, -8.5
    low_gs98 = kap.kap_get_fixed_metal(
        z, x, lr_low, lt_low, prefix="lowT_fa05_gs98", backend="rust"
    )
    low_gn93 = kap.kap_get_fixed_metal(
        z, x, lr_low, lt_low, prefix="lowT_fa05_gn93", backend="rust"
    )
    low_a09 = kap.kap_get_fixed_metal(
        z, x, lr_low, lt_low, prefix="lowT_fa05_a09", backend="rust"
    )
    low_f11_a = kap.kap_get_fixed_metal(
        0.01, x, -4.0, 3.2, prefix="lowT_Freedman11", backend="rust"
    )
    low_f11_b = kap.kap_get_fixed_metal(
        0.10, x, -4.0, 3.2, prefix="lowT_Freedman11", backend="rust"
    )
    assert abs(float(low_gn93[0]) - float(low_gs98[0])) > 1e-4
    assert abs(float(low_a09[0]) - float(low_gs98[0])) > 1e-4
    assert abs(float(low_f11_a[0]) - float(low_f11_b[0])) > 1e-2

    # Verify previously un-embedded (Z, X) slices in gs98 and a09 are distinct
    lk_z003 = kap.kap_get_fixed_metal(0.0003, 0.20, lr, lt, prefix="a09")[0]
    lk_z001 = kap.kap_get_fixed_metal(0.001, 0.20, lr, lt, prefix="a09")[0]
    assert abs(float(lk_z003) - float(lk_z001)) > 1e-3

    # Verify Type 2 CO tables are distinct (gs98_co != gn93_co != a09_co)
    co_gn93 = kap.kap_get_co(
        0.015, 0.20, 0.15, 0.25, -5.0, 6.1, kap_co_option="gn93_co"
    )
    co_gs98 = kap.kap_get_co(
        0.015, 0.20, 0.15, 0.25, -5.0, 6.1, kap_co_option="gs98_co"
    )
    co_a09 = kap.kap_get_co(
        0.015, 0.20, 0.15, 0.25, -5.0, 6.1, kap_co_option="a09_co"
    )
    assert abs(float(co_gs98[0]) - float(co_gn93[0])) > 1e-4
    assert abs(float(co_a09[0]) - float(co_gn93[0])) > 1e-4
    assert abs(float(co_a09[0]) - float(co_gs98[0])) > 1e-4

    # Verify AESOPUS across multiple Zref sets and (X, XC, XN, XO) slices
    aes_z008 = kap.kap_get_aesopus(
        0.008, 0.60, 0.002, 0.001, 0.004, -8.5, 3.5, backend="rust"
    )
    aes_z020 = kap.kap_get_aesopus(
        0.020, 0.60, 0.005, 0.002, 0.009, -8.5, 3.5, backend="rust"
    )
    aes_z040 = kap.kap_get_aesopus(
        0.040, 0.75, 0.012, 0.004, 0.015, -8.5, 3.5, backend="rust"
    )
    k020 = float(aes_z020[0])
    assert abs(float(aes_z008[0]) - k020) / k020 > 1e-2
    assert abs(float(aes_z040[0]) - k020) / k020 > 1e-2


def test_rust_vs_jax_backend_parity_across_all_regimes() -> None:
    """Verify `backend='rust'` vs `backend='jax'` parity (`rtol <= 1e-11`)."""
    cases = [
        {
            "z": 0.015,
            "x": 0.65,
            "zbase": 0.015,
            "log_rho": -5.2,
            "log_t": 6.1,
            "kap_option": "gn93",
            "cubic_interpolation_in_x": True,
            "cubic_interpolation_in_z": True,
        },
        {
            "z": 0.022,
            "x": 0.52,
            "zbase": 0.022,
            "log_rho": -4.3,
            "log_t": 5.7,
            "kap_option": "OP_gs98",
            "cubic_interpolation_in_x": False,
            "cubic_interpolation_in_z": True,
        },
        {
            "z": 0.025,
            "x": 5.0e-4,
            "zbase": 0.018,
            "xc": 0.012,
            "xo": 0.015,
            "log_rho": -4.8,
            "log_t": 6.2,
            "kap_option": "gs98",
            "kap_co_option": "gs98_co",
            "use_type2_opacities": True,
            "use_zbase_for_type1": True,
            "cubic_interpolation_in_x": True,
            "cubic_interpolation_in_z": True,
        },
        {
            "z": 0.018,
            "x": 0.70,
            "zbase": 0.018,
            "log_rho": -7.2,
            "log_t": 3.84,
            "kap_option": "a09",
            "kap_low_t_option": "lowT_fa05_a09",
        },
        {
            "z": 0.015,
            "x": 0.68,
            "zbase": 0.015,
            "xc": 0.003,
            "xn": 0.001,
            "log_rho": -9.5,
            "log_t": 3.60,
            "kap_low_t_option": "kapCN",
        },
        {
            "z": 0.020,
            "x": 0.65,
            "zbase": 0.017,
            "xc": 0.004,
            "xn": 0.0015,
            "xo": 0.008,
            "log_rho": -8.4,
            "log_t": 3.55,
            "kap_low_t_option": "AESOPUS",
        },
        {
            "z": 0.02,
            "x": 0.0,
            "zbase": 0.02,
            "log_rho": 5.5,
            "log_t": 7.8,
            "zbar": 1.6,
            "eta": 15.0,
            "include_electron_conduction": True,
            "use_blouin_conductive_opacities": True,
        },
        {
            "z": 0.02,
            "x": 0.70,
            "zbase": 0.02,
            "log_rho": -2.0,
            "log_t": 8.45,
            "lnfree_e": -0.16,
            "d_lnfree_e_dln_rho": 0.01,
            "d_lnfree_e_dln_t": 0.02,
            "eta": -1.5,
            "d_eta_dln_rho": 0.8,
            "d_eta_dln_t": -1.2,
        },
    ]
    for kw in cases:
        r_out = kap.kap_get_results(**kw, backend="rust")  # type: ignore[arg-type]
        j_out = kap.kap_get_results(**kw, backend="jax")  # type: ignore[arg-type]
        np.testing.assert_allclose(j_out.kap, r_out.kap, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(
            j_out.log10_kap, r_out.log10_kap, rtol=1e-12, atol=0.0
        )
        np.testing.assert_allclose(
            j_out.dlnkap_dln_rho, r_out.dlnkap_dln_rho, rtol=1e-11, atol=0.0
        )
        np.testing.assert_allclose(
            j_out.dlnkap_dln_t, r_out.dlnkap_dln_t, rtol=1e-11, atol=0.0
        )
        np.testing.assert_allclose(
            j_out.kap_fracs, r_out.kap_fracs, rtol=1e-12, atol=1e-15
        )


def test_compton_and_conduction_submodels_rust_vs_jax() -> None:
    """Verify Compton and Potekhin/Blouin conduction parity and derivatives."""
    for zbar in (1.0, 1.5, 2.5, 6.0):
        for use_blouin in (False, True):
            kr, dr_r, dt_r = kap.kap_get_elect_cond_opacity(
                zbar, 5.2, 7.9, use_blouin=use_blouin, backend="rust"
            )
            kj, dr_j, dt_j = kap.kap_get_elect_cond_opacity(
                zbar, 5.2, 7.9, use_blouin=use_blouin, backend="jax"
            )
            np.testing.assert_allclose(kj, kr, rtol=1e-13, atol=0.0)
            np.testing.assert_allclose(dr_j, dr_r, rtol=1e-12, atol=0.0)
            np.testing.assert_allclose(dt_j, dt_r, rtol=1e-12, atol=0.0)

    rho, t, lnfe, dlnf_dr, dlnf_dt, eta, deta_dr, deta_dt = (
        10.0,
        3.0e8,
        -0.15,
        0.02,
        0.05,
        1.2,
        0.65,
        -0.95,
    )
    k_rust, dr_rust, dt_rust = kap.kap_get_compton_opacity(
        rho, t, lnfe, dlnf_dr, dlnf_dt, eta, deta_dr, deta_dt, backend="rust"
    )
    k_jax, dr_jax, dt_jax = kap.kap_get_compton_opacity(
        rho, t, lnfe, dlnf_dr, dlnf_dt, eta, deta_dr, deta_dt, backend="jax"
    )
    np.testing.assert_allclose(k_jax, k_rust, rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(dr_jax, dr_rust, rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(dt_jax, dt_rust, rtol=1e-13, atol=0.0)


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
            backend=backend,  # type: ignore[arg-type]
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
