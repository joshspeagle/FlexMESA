# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit and golden-oracle parity tests for `flexmesa.rates` (Wave 3b.3)."""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from flexmesa import chem, eos, rates
from flexmesa.testing import golden

_GOLDEN_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "fixtures"
    / "golden"
    / "rates"
    / "test_output.txt"
)


def test_reaction_catalog_lookup_roundtrips_id_and_metadata() -> None:
    """Verify reaction ID, handle name, and ReactionInfo metadata lookups."""
    ir_3a = rates.reaction_id_by_name("r_he4_he4_he4_to_c12")
    assert ir_3a > 0
    assert rates.reaction_name_by_id(ir_3a) == "r_he4_he4_he4_to_c12"

    info = rates.reaction_info("r_c12_ag_o16")
    assert info.name == "r_c12_ag_o16"
    assert info.rho_exp == 1
    assert info.std_q > 7.0


def test_section1_screening_modes_and_jax_autodiff_match_golden() -> None:
    """Verify Salpeter, Extended, and Chugunov screening in Rust and JAX."""
    log_t = 7.7110722845770692
    log_rho = 4.5306372623742392
    temp = 10.0**log_t
    den = 10.0**log_rho
    zbar = 2.0156203413143605
    abar = 4.0312406826287210
    z2bar = 4.1624054993206103
    he4 = chem.get_nuclide_info("he4")
    c12 = chem.get_nuclide_info("c12")
    a1, z1 = he4.w, float(he4.z)
    a2, z2 = c12.w, float(c12.z)

    oracles = {
        "salpeter": (
            4.0373162996606471,
            -1.6438665008313957e-7,
            8.3019538122528578e-5,
        ),
        "extended": (
            2.4319226633761173,
            -4.2425820200522379e-8,
            2.1424210372719004e-5,
        ),
        "chugunov": (
            2.3827727933330571,
            -4.6135154522099626e-8,
            2.3297997100016832e-5,
        ),
        "no_screening": (1.0, 0.0, 0.0),
    }

    for mode, (exp_s, exp_dt, exp_dd) in oracles.items():
        res = rates.screen_pair(
            mode, temp, den, zbar, abar, z2bar, a1, z1, a2, z2, backend="rust"
        )
        assert isinstance(res, rates.ScreenEval)
        npt.assert_allclose(res.scor, exp_s, rtol=1e-13, atol=1e-15)
        npt.assert_allclose(res.dscor_dt, exp_dt, rtol=1e-13, atol=1e-15)
        npt.assert_allclose(res.dscor_dd, exp_dd, rtol=1e-13, atol=1e-15)

        s_j, dt_j, dd_j = rates.screen_pair(
            mode,
            jnp.asarray(temp),
            jnp.asarray(den),
            zbar,
            abar,
            z2bar,
            a1,
            z1,
            a2,
            z2,
            backend="jax",
        )
        npt.assert_allclose(float(s_j), exp_s, rtol=1e-13, atol=1e-15)
        npt.assert_allclose(float(dt_j), exp_dt, rtol=1e-13, atol=1e-15)
        npt.assert_allclose(float(dd_j), exp_dd, rtol=1e-13, atol=1e-15)

        if mode != "no_screening":
            grad_t = float(
                jax.grad(
                    lambda t_in, m=mode: rates.screen_pair(
                        m,
                        t_in,
                        jnp.asarray(den),
                        zbar,
                        abar,
                        z2bar,
                        a1,
                        z1,
                        a2,
                        z2,
                        backend="jax",
                    )[0]
                )(jnp.asarray(temp, dtype=jnp.float64))
            )
            grad_d = float(
                jax.grad(
                    lambda d_in, m=mode: rates.screen_pair(
                        m,
                        jnp.asarray(temp),
                        d_in,
                        zbar,
                        abar,
                        z2bar,
                        a1,
                        z1,
                        a2,
                        z2,
                        backend="jax",
                    )[0]
                )(jnp.asarray(den, dtype=jnp.float64))
            )
            npt.assert_allclose(grad_t, exp_dt, rtol=1e-11)
            npt.assert_allclose(grad_d, exp_dd, rtol=1e-11)

        # Verify neutral reactant (z1 = 0) yields finite (1, 0, 0) and 0 grad.
        s0, dt0, dd0 = rates.screen_pair(
            mode,
            jnp.asarray(temp),
            jnp.asarray(den),
            zbar,
            abar,
            z2bar,
            1.0,
            0.0,
            a2,
            z2,
            backend="jax",
        )
        assert (float(s0), float(dt0), float(dd0)) == (1.0, 0.0, 0.0)
        g0 = float(
            jax.grad(
                lambda t_in, m=mode: rates.screen_pair(
                    m,
                    t_in,
                    jnp.asarray(den),
                    zbar,
                    abar,
                    z2bar,
                    1.0,
                    0.0,
                    a2,
                    z2,
                    backend="jax",
                )[0]
            )(jnp.asarray(temp, dtype=jnp.float64))
        )
        assert g0 == 0.0

    zones = rates.screen_pair_zones(
        "extended",
        [temp, temp],
        [den, den],
        [zbar, zbar],
        [abar, abar],
        [z2bar, z2bar],
        a1,
        z1,
        a2,
        z2,
    )
    assert zones.shape == (2, 3)
    npt.assert_allclose(zones[0], oracles["extended"], rtol=1e-13)


def test_section2_and_section3_weak_and_ecapture_match_golden() -> None:
    """Verify weak_info, weaklib, ecapture, and Suzuki2016 golden parity."""
    hl, qneu = rates.weak_info_by_names("o14", "n14")
    npt.assert_allclose(hl, 70.64, rtol=1e-14)
    npt.assert_allclose(qneu, 2.22, rtol=1e-14)

    t9_s2 = (10.0 ** np.log10(9.0e8)) * 1.0e-9
    ye_rho_s2 = 0.5 * (10.0 ** np.log10(4.5e5))
    lhs, rhs, ev1 = rates.eval_weak_table_index(1, t9_s2, np.log10(ye_rho_s2))
    assert (lhs, rhs) == ("h1", "neut")
    assert ev1.qneu > 0.0

    zbar = 10.909090909090908
    abar = 21.818181818181817
    ye = zbar / abar
    t_coul = 10.0**8.5
    rho_coul = 10.0**9.5
    mu1, mu2, dmu_ev, vs = rates.eval_coulomb(
        t_coul,
        rho_coul,
        zbar,
        abar,
        12.0,
        11.0,
        which_mui="pcr2009",
        which_vs="itoh2002",
    )
    npt.assert_allclose(mu1, -43.7453051893886027, rtol=1e-13)
    npt.assert_allclose(mu2, -37.5095737212022584, rtol=1e-13)
    npt.assert_allclose(dmu_ev, 169926.1797009507135954, rtol=1e-13)
    npt.assert_allclose(vs, 0.0079074257877945, rtol=1e-13)

    temp = 10.0**8.3
    rho = 10.0**9.8
    t9 = temp * 1.0e-9
    ye_rho = ye * rho
    helm_res, _, _ = eos.eval_helm_eos(temp, rho, abar, zbar)
    eta_k = float(helm_res.res[eos.I_ETA])

    w_cap = rates.eval_weak_pair("mg24", "na24", t9, ye_rho, use_suzuki=False)
    npt.assert_allclose(w_cap.lambda_rate, 1.4421177069786498e-20, rtol=1e-12)

    ec_cap = rates.eval_ecapture_pair(
        "mg24", "na24", t9, ye_rho, eta_k, zbar=zbar, abar=abar
    )
    npt.assert_allclose(ec_cap.lambda_rate, 3.8507552351991037e-3, rtol=1e-12)

    suz_cap = rates.eval_weak_pair("mg24", "na24", t9, ye_rho, use_suzuki=True)
    npt.assert_allclose(suz_cap.lambda_rate, 4.4797491275210752e-3, rtol=1e-12)


def test_section4_all_307_raw_rates_across_9_temps_match_golden() -> None:
    """Verify set_raw_rates_zones against all 9 x 307 golden reaction rows."""
    lines = _GOLDEN_PATH.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if "do_test_suzuki" in line)
    end = next(i for i, line in enumerate(lines) if "FL_epsnuc_3alf" in line)

    irs = list(range(1, 308))
    log_ts: list[float] = []
    temps: list[float] = []
    expected_blocks: list[list[float]] = []

    i = start
    while i < end:
        toks = lines[i].split()
        if len(toks) == 2 and toks[0] == "logT":
            lt = golden.parse_fortran_float(toks[1])
            log_ts.append(lt)
            temps.append(10.0**lt)
            block = [0.0] * 307
            i += 3
            while i < end:
                row = lines[i].strip()
                if not row or row == "done":
                    break
                rtoks = row.split()
                if len(rtoks) == 2:
                    ir = rates.reaction_id_by_name(rtoks[0])
                    block[ir - 1] = golden.parse_fortran_float(rtoks[1])
                i += 1
            expected_blocks.append(block)
        i += 1

    assert len(temps) == 9
    expected = np.array(expected_blocks, dtype=np.float64)
    got = rates.set_raw_rates_zones(irs, temps, log_ts=log_ts)
    assert got.shape == (9, 307)
    mask = expected > 1.0e-290
    npt.assert_allclose(got[mask], expected[mask], rtol=1e-11)


def test_sections5_6_7_pycno_rate_table_and_reaclib_jax_match_golden() -> None:
    """Verify FL_epsnuc_3alf, eval_rate_table, and eval_reaclib_handle + JAX."""
    eps_rust = rates.eval_fl_epsnuc_3alf(
        1.0e7, 1.0e10, 1.0, 2.0, backend="rust"
    )
    eps_jax = rates.eval_fl_epsnuc_3alf(
        jnp.asarray(1.0e7),
        jnp.asarray(1.0e10),
        jnp.asarray(1.0),
        jnp.asarray(2.0),
        backend="jax",
    )
    npt.assert_allclose(float(eps_rust), 1.1675037697640312e21, rtol=1e-13)
    npt.assert_allclose(float(eps_jax), 1.1675037697640312e21, rtol=1e-13)

    temp_r3 = 10.0 ** np.log10(9.0e8)
    r_tab, _ = rates.eval_rate_table("r3", temp_r3)
    npt.assert_allclose(r_tab, 1.0e-8, rtol=1e-14)

    for handle in ("r_c12_ag_o16", "r_ar36_ga_s32"):
        f_r, df_r, rev_r, drev_r = rates.eval_reaclib_handle(
            handle, 2.0, backend="rust"
        )
        f_j, df_j, rev_j, drev_j = rates.eval_reaclib_handle(
            handle, jnp.asarray(2.0), backend="jax"
        )
        npt.assert_allclose(float(f_j), float(f_r), rtol=1e-13)
        npt.assert_allclose(float(df_j), float(df_r), rtol=1e-13)
        npt.assert_allclose(float(rev_j), float(rev_r), rtol=1e-13)
        npt.assert_allclose(float(drev_j), float(drev_r), rtol=1e-13)

        grad_t9 = float(
            jax.grad(
                lambda t9_in, h=handle: rates.eval_reaclib_handle(
                    h, t9_in, backend="jax"
                )[0]
            )(jnp.asarray(2.0, dtype=jnp.float64))
        )
        npt.assert_allclose(grad_t9 * 2.0, float(df_r), rtol=1e-12)

    f_lo, df_lo, _, _ = rates.eval_reaclib_handle(
        "r_c12_ag_o16", jnp.asarray(0.005), backend="jax"
    )
    assert (float(f_lo), float(df_lo)) == (0.0, 0.0)
