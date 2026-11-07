# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit, golden parity, and JAX derivative tests for `flexmesa.kap`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flexmesa import kap


def test_kap_fixed_metals_matches_fortran_golden_test_1() -> None:
    """Verify `kap_get_results` reproduces Fortran MESA `kap` Test 1."""
    out = kap.kap_get_results(
        z=1.8000000000000016e-2,
        x=0.70,
        zbase=0.0,
        xc=0.0,
        xn=0.0,
        xo=0.0,
        xne=0.0,
        log_rho=-6.0,
        log_t=6.0,
        lnfree_e=-1.6965073622377105e-1,
        d_lnfree_e_dln_rho=0.0,
        d_lnfree_e_dln_t=0.0,
        eta=-10.0,
        d_eta_dln_rho=0.0,
        d_eta_dln_t=0.0,
        zbar=1.1018857434091647,
        kap_option="gs98",
        kap_low_t_option="lowT_fa05_gs98",
        use_type2_opacities=False,
        cubic_interpolation_in_x=False,
        cubic_interpolation_in_z=False,
    )
    np.testing.assert_allclose(
        out.log10_kap, -4.424000000011061e-1, rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(
        out.kap, 3.6107714509943334e-1, rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(
        out.dlnkap_dln_rho, 2.281674219017579e-2, rtol=1e-11, atol=0.0
    )
    np.testing.assert_allclose(
        out.dlnkap_dln_t, 3.538705857187698e-2, rtol=1e-11, atol=0.0
    )
    np.testing.assert_allclose(
        out.kap_fracs,
        np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
        rtol=1e-12,
        atol=1e-15,
    )


def test_kap_co_enhanced_matches_fortran_golden_test_2() -> None:
    """Verify `kap_get_results` reproduces Fortran MESA `kap` Test 2."""
    out = kap.kap_get_results(
        z=5.800000000000005e-2,
        x=0.0,
        zbase=0.018,
        xc=2.4119616e-2,
        xn=9.56736e-4,
        xo=2.7683164e-2,
        xne=1.776024e-3,
        log_rho=-6.0,
        log_t=6.0,
        lnfree_e=-6.969230350735236e-1,
        d_lnfree_e_dln_rho=0.0,
        d_lnfree_e_dln_t=0.0,
        eta=-10.0,
        d_eta_dln_rho=0.0,
        d_eta_dln_t=0.0,
        zbar=2.074131706506234,
        kap_option="gs98",
        kap_co_option="gn93_co",
        kap_low_t_option="lowT_fa05_gs98",
        use_type2_opacities=True,
    )
    np.testing.assert_allclose(
        out.log10_kap, -6.641677699836668e-1, rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(
        out.kap, 2.1668668723388557e-1, rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(
        out.dlnkap_dln_rho, 3.6488626497031655e-2, rtol=1e-11, atol=0.0
    )
    np.testing.assert_allclose(
        out.dlnkap_dln_t, -1.908154874225351e-2, rtol=1e-11, atol=0.0
    )
    np.testing.assert_allclose(
        out.kap_fracs[kap.I_FRAC_TYPE2], 1.0, rtol=1e-12, atol=1e-15
    )


def test_kap_aesopus_matches_fortran_golden_test_4() -> None:
    """Verify `kap_get_results` with `AESOPUS` reproduces Fortran Test 4."""
    out = kap.kap_get_results(
        z=2.0074120713487353e-2,
        x=6.788866252318019e-1,
        zbase=0.014,
        xc=2.9968458709806432e-3,
        xn=1.5270900145630591e-3,
        xo=9.337626324051438e-3,
        xne=0.0,
        log_rho=-8.249643069901467,
        log_t=3.5571504546260235,
        lnfree_e=-9.210342466630861,
        d_lnfree_e_dln_rho=0.0,
        d_lnfree_e_dln_t=0.0,
        eta=-20.0,
        d_eta_dln_rho=0.0,
        d_eta_dln_t=0.0,
        zbar=1.1007197824017732,
        kap_option="a09",
        kap_co_option="a09_co",
        kap_low_t_option="AESOPUS",
        use_zbase_for_type1=True,
        use_type2_opacities=True,
    )
    np.testing.assert_allclose(
        out.log10_kap, -2.523381870188199, rtol=1e-7, atol=0.0
    )
    np.testing.assert_allclose(
        out.kap, 2.9965265486259223e-3, rtol=2e-7, atol=0.0
    )
    np.testing.assert_allclose(
        out.dlnkap_dln_rho, 5.72840991797351e-1, rtol=1e-6, atol=0.0
    )
    np.testing.assert_allclose(
        out.dlnkap_dln_t, 8.462557917457387, rtol=1e-6, atol=0.0
    )


def test_compton_opacity_rust_matches_jax_and_finite_differences() -> None:
    """Verify Poutanen (2017) Compton opacity parity and finite differences."""
    rho = 10.0
    t = 3.0e8
    lnfree_e = -0.15
    dlnf_dr = 0.02
    dlnf_dt = 0.05
    eta = 1.2
    deta_dr = 0.65
    deta_dt = -0.95

    k_rust, dr_rust, dt_rust = kap.kap_get_compton_opacity(
        rho,
        t,
        lnfree_e,
        dlnf_dr,
        dlnf_dt,
        eta,
        deta_dr,
        deta_dt,
        backend="rust",
    )
    k_jax, dr_jax, dt_jax = kap.kap_get_compton_opacity(
        rho,
        t,
        lnfree_e,
        dlnf_dr,
        dlnf_dt,
        eta,
        deta_dr,
        deta_dt,
        backend="jax",
    )
    np.testing.assert_allclose(k_jax, k_rust, rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(dr_jax, dr_rust, rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(dt_jax, dt_rust, rtol=1e-13, atol=0.0)

    eps = 1.0e-6
    k_rp, _, _ = kap.kap_get_compton_opacity(
        rho * np.exp(eps),
        t,
        lnfree_e + dlnf_dr * eps,
        dlnf_dr,
        dlnf_dt,
        eta + deta_dr * eps,
        deta_dr,
        deta_dt,
        backend="rust",
    )
    k_rm, _, _ = kap.kap_get_compton_opacity(
        rho * np.exp(-eps),
        t,
        lnfree_e - dlnf_dr * eps,
        dlnf_dr,
        dlnf_dt,
        eta - deta_dr * eps,
        deta_dr,
        deta_dt,
        backend="rust",
    )
    fd_dr = (np.log(float(k_rp)) - np.log(float(k_rm))) / (2.0 * eps)
    np.testing.assert_allclose(dr_rust, fd_dr, rtol=1e-8, atol=0.0)

    k_tp, _, _ = kap.kap_get_compton_opacity(
        rho,
        t * np.exp(eps),
        lnfree_e + dlnf_dt * eps,
        dlnf_dr,
        dlnf_dt,
        eta + deta_dt * eps,
        deta_dr,
        deta_dt,
        backend="rust",
    )
    k_tm, _, _ = kap.kap_get_compton_opacity(
        rho,
        t * np.exp(-eps),
        lnfree_e - dlnf_dt * eps,
        dlnf_dr,
        dlnf_dt,
        eta - deta_dt * eps,
        deta_dr,
        deta_dt,
        backend="rust",
    )
    fd_dt = (np.log(float(k_tp)) - np.log(float(k_tm))) / (2.0 * eps)
    np.testing.assert_allclose(dt_rust, fd_dt, rtol=1e-8, atol=0.0)

    k_comb, dr_comb, dt_comb = kap.combine_rad_with_conduction_jax(
        2.0, 0.3, -1.5, 6.0, -0.8, 2.2
    )
    np.testing.assert_allclose(float(k_comb), 1.5, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(
        float(dr_comb), 0.75 * 0.3 + 0.25 * (-0.8), rtol=1e-14, atol=0.0
    )
    np.testing.assert_allclose(
        float(dt_comb), 0.75 * (-1.5) + 0.25 * 2.2, rtol=1e-14, atol=0.0
    )


def test_make_differentiable_kap_custom_jvp_matches_analytical() -> None:
    """Verify `make_differentiable_kap` `jax.grad` matches partials."""
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

    out = kap.kap_get_results(
        z=1.8000000000000016e-2,
        x=0.70,
        zbase=0.0,
        log_rho=-6.0,
        log_t=6.0,
        lnfree_e=-1.6965073622377105e-1,
        eta=-10.0,
        zbar=1.1018857434091647,
    )
    np.testing.assert_allclose(lk_val, out.log10_kap, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(
        float(grad_lr), out.dlnkap_dln_rho, rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(
        float(grad_lt), out.dlnkap_dln_t, rtol=1e-12, atol=0.0
    )


def test_kapcn_and_multi_zone_batch_evaluation_parity() -> None:
    """Verify `kap_get_kapcn` knot accuracy and `kap_get_zones` parity."""
    kap_cn, _, _ = kap.kap_get_kapcn(
        zbase=0.02,
        x=0.70,
        fc=1.0,
        fn_val=1.0,
        log_rho=-10.2,
        log_t=3.6,
    )
    np.testing.assert_allclose(np.log10(kap_cn), -3.361, rtol=1e-12, atol=0.0)

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

    with pytest.raises(ValueError, match="mismatched zone"):
        kap.kap_get_zones(
            chem_id,
            xa_zones[: nz - 1],
            log_rho_zones,
            log_t_zones,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
        )
    with pytest.raises(ValueError, match="kap_get failed"):
        kap.kap_get_zones(
            np.array([0], dtype=np.uintp),
            np.ones((nz, 1), dtype=np.float64),
            log_rho_zones,
            log_t_zones,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
        )
