# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit and golden-oracle parity tests for `flexmesa.rates` (Wave 3b.3)."""

from __future__ import annotations

import pathlib
import struct

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

import flexmesa
from flexmesa import _rust, chem, eos, rates
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

    with flexmesa.numerics_mode_context("strict"):
        for mode, (exp_s, exp_dt, exp_dd) in oracles.items():
            res = rates.screen_pair(
                mode,
                temp,
                den,
                zbar,
                abar,
                z2bar,
                a1,
                z1,
                a2,
                z2,
                backend="rust",
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

            # Verify neutral reactant (z1 = 0) yields finite (1, 0, 0) & 0 grad
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
    with flexmesa.numerics_mode_context("strict"):
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


def test_mesa_26_4_1_rates_oracles_and_rev_coeff1_match_strict() -> None:
    """Verify MESA 26.4.1 `rates_oracles` (`185` rows) and `rev_coeff1`."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    oracles_dir = repo_root / "benchmarks" / "fortran_mesa" / "rates_oracles"
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
        "gen_rates_oracles.f90",
        "mesa_26_4_1_rates_oracles.tsv",
    ):
        text = (oracles_dir / rel_name).read_text(encoding="utf-8")
        for tok in forbidden:
            assert tok not in text, f"Forbidden token {tok!r} in {rel_name}"

    tsv_text = (oracles_dir / "mesa_26_4_1_rates_oracles.tsv").read_text(
        encoding="utf-8"
    )
    for header in (
        "# mesa_version: r26.4.1",
        "# mesa_git_commit: c8c202662922a22b3efefa824473ad615c623afd",
        "# mesasdk_version: x86_64-linux-26.6.1",
        "# math_backend: CRMATH",
    ):
        assert header in tsv_text

    rows = [
        line.split("\t")
        for line in tsv_text.splitlines()
        if line.strip()
        and not line.startswith("#")
        and not line.startswith("row_id\t")
    ]
    assert len(rows) == 185

    pair_map = {
        "h1_h1": ("h1", "h1"),
        "he4_c12": ("he4", "c12"),
        "h1_n14": ("h1", "n14"),
        "c12_c12": ("c12", "c12"),
        "o16_o16": ("o16", "o16"),
        "neut_fe": ("neut", "si28"),
        "mg24_mg24": ("mg24", "mg24"),
        "c12_o16": ("c12", "o16"),
    }

    with flexmesa.numerics_mode_context("strict"):
        coulomb_groups: dict[str, list[list[str]]] = {}
        for cols in rows:
            sec, lbl, out_name = cols[1], cols[2], cols[8]
            if sec == "coulomb":
                coulomb_groups.setdefault(lbl, []).append(cols)
                continue
            x1, x2, x3, x4, x5 = (_hex_to_f64(cols[k]) for k in range(3, 8))
            expected = _hex_to_f64(cols[9])

            if sec == "screen":
                parts = lbl.split(":")
                n1, n2 = pair_map[parts[-1]]
                i1 = chem.get_nuclide_info(n1)
                i2 = chem.get_nuclide_info(n2)
                sc = rates.screen_pair(
                    parts[0],
                    x1,
                    x2,
                    x3,
                    x4,
                    x5,
                    i1.w,
                    float(i1.z),
                    i2.w,
                    float(i2.z),
                    backend="rust",
                )
                assert isinstance(sc, rates.ScreenEval)
                got = {
                    "scor": sc.scor,
                    "dscor_dt": sc.dscor_dt,
                    "dscor_dd": sc.dscor_dd,
                }[out_name]
            elif sec == "reaclib":
                lam, dlam, rlam, drlam = rates.eval_reaclib_handle(
                    lbl, x1, backend="rust"
                )
                got = float(
                    {
                        "lambda": lam,
                        "dlambda_dlnT": dlam,
                        "rlambda": rlam,
                        "drlambda_dlnT": drlam,
                    }[out_name]
                )
            elif sec == "raw_rate":
                ir = rates.reaction_id_by_name(lbl)
                got_arr = rates.set_raw_rates_zones([ir], [x2], log_ts=[x1])
                got = float(got_arr[0, 0])
            elif sec == "pycno":
                got = float(
                    rates.eval_fl_epsnuc_3alf(x1, x2, x3, x4, backend="rust")
                )
            elif sec == "weak":
                lhs, rhs = lbl.split(":")[1].split("_")
                w = rates.eval_weak_pair(
                    lhs, rhs, x1, x2, use_suzuki=(x3 != 0.0)
                )
                got = {
                    "lambda": w.lambda_rate,
                    "dlambda_dlnT": w.dlambda_dln_t,
                    "dlambda_dlnRho": w.dlambda_dln_rho,
                    "Q": w.q,
                    "dQ_dlnT": w.dq_dln_t,
                    "dQ_dlnRho": w.dq_dln_rho,
                    "Qneu": w.qneu,
                    "dQneu_dlnT": w.dqneu_dln_t,
                    "dQneu_dlnRho": w.dqneu_dln_rho,
                }[out_name]
            else:
                raise AssertionError(f"Unknown section {sec}")

            assert _ulp_diff(got, expected) == 0, f"{sec}:{lbl}:{out_name}"

        for lbl, crows in coulomb_groups.items():
            mui_opt, vs_opt = lbl.split(":")
            t = _hex_to_f64(crows[0][3])
            rho = _hex_to_f64(crows[0][4])
            zb = _hex_to_f64(crows[0][5])
            ab = _hex_to_f64(crows[0][6])
            z1 = _hex_to_f64(crows[0][7])
            z2 = _hex_to_f64(crows[1][7])
            mu1, mu2, dmu_ev, vs = rates.eval_coulomb(
                t, rho, zb, ab, z1, z2, which_mui=mui_opt, which_vs=vs_opt
            )
            got_map = {"mu1": mu1, "mu2": mu2, "dmu_ev": dmu_ev, "vs": vs}
            for crow in crows:
                assert _ulp_diff(got_map[crow[8]], _hex_to_f64(crow[9])) <= 6

        he4 = chem.get_nuclide_info("he4")
        c12 = chem.get_nuclide_info("c12")
        o16 = chem.get_nuclide_info("o16")
        c1_fwd = float(
            _rust.rates_compute_rev_ratio_coeff1(
                [c12.w, he4.w], [c12.spin, he4.spin], [o16.w], [o16.spin]
            )
        )
        c1_inv = float(
            _rust.rates_compute_rev_ratio_coeff1(
                [o16.w], [o16.spin], [c12.w, he4.w], [c12.spin, he4.spin]
            )
        )
        _, terms = _rust.rates_reaclib_terms("r_c12_ag_o16")
        assert _ulp_diff(c1_fwd, float(terms[0][4])) == 0
        npt.assert_allclose(c1_fwd + c1_inv, 0.0, rtol=0.0, atol=1e-14)
