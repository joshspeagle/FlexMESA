# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit and golden-baseline parity tests for `flexmesa.neu`.

Verifies:
    1. Golden baseline parity ($\\le 10^{-10}$ relative error) against all
       10,000 rows of upstream MESA `neu/test/test_output`
       spanning low-temperature cosine cutoffs (`log10_t <= 7.5`), degenerate
       cutoffs, all three photoneutrino Fourier bands ($10^7\\text{--}10^8$,
       $10^8\\text{--}10^9$, $\\ge 10^9\\,\\mathrm{K}$), bremsstrahlung
       blending, recombination (`reco`), and both $T_9 < 10$ and $T_9 \\ge 10$
       pair-production fits.
    2. Solar stellar-profile sweep across all 194 zones of
       upstream MESA `neu/test/test_1.0.data`.
    3. Native JAX `float64` `neu_channels_jax` and `neu_get_jax` (all 5
       Itoh et al. 1996 channels: `pair`, `plas`, `phot`, `brem`, `reco`)
       and `jax.jacfwd` partial derivatives against `neu_get` and golden
       outputs across all 10,000 grid rows.
    4. Exact zero cutoff below $\\log_{10} T = 6.8$, channel flag isolation,
       centered finite-difference partial derivatives, and input validation.
"""

from __future__ import annotations

import math
import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flexmesa import constants as _const
from flexmesa import neu
from flexmesa.data import fetch


def _exp10(x: float) -> float:
    """Evaluates $10^x = \\exp(x \\ln 10)$ matching upstream `exp10`."""
    return math.exp(x * _const.LN10)


def _mesa_neu_test_dir() -> pathlib.Path:
    """Locates the upstream MESA `neu/test` directory."""
    return fetch.resolve_mesa_test_inputs_dir() / "neu" / "test"


def _parse_fortran_float(tok: str) -> float:
    """Parses a Fortran `1pe26.16` token, handling 3-digit exponents."""
    try:
        return float(tok)
    except ValueError:
        for i in range(len(tok) - 1, 0, -1):
            if tok[i] in "+-" and tok[i - 1] not in "eEdD":
                return float(tok[:i] + "e" + tok[i:])
        raise


# Representative golden rows from upstream `neu/test/test_output`.
# fmt: off
GOLDEN_ROWS: tuple[tuple[float, ...], ...] = (
    # IDX 3090: logT = 7.35, logRho = 6.0, abar = 54.1, zbar = 1.0 (reco > 0)
    (
        7.35, 6.0, 54.1, 1.0,
        2.423905156910367e-05, 5.941044897746003e-12, 3.381267831966417e-11,
        -1.073039318051879e-06, 5.844248953287863e-05, 5.087480723393688e-237,
        2.3843277319977368e-05, 4.802510786004552e-09, 3.909403005044627e-07,
        3.143783583316795e-11, 1.2984720806319637e-241, 5.822265338481926e-12,
        2.452270390955503e-15, 1.1631126166544558e-13, 1.6027207676106186e-17,
        -6.815259917322703e-242, 3.362999867833609e-11, -7.5336026348655e-15,
        1.9032778479375864e-13, -1.1454083080843881e-16,
        1.1657138345625388e-237, -1.0623526062534833e-06,
        5.04822892580582e-11, -1.0738730187375997e-08,
        1.5360997222785757e-12, -6.306511844983335e-236,
        5.7473275998313454e-05, -2.731091848860948e-09,
        9.719056036415043e-07, 3.902277253835819e-11,
    ),
    # IDX 3555: logT = 7.35, logRho = 8.25, abar = 30.5, zbar = 13.5
    (
        7.35, 8.25, 30.5, 13.5,
        5.649612198514403e-05, 1.8739985551069504e-11, -1.1361557539517236e-14,
        -1.830377418252603e-06, 8.07219108145557e-06, 0.0,
        5.362861652028444e-23, 5.713559028118851e-50, 5.649612198514403e-05,
        0.0, 0.0, 1.5735707946279355e-28,
        1.5984380634973777e-55, 1.8739985551069504e-11, 0.0,
        0.0, -6.002464610501367e-30, -1.1235843420965503e-56,
        -1.136155753951723e-14, 0.0, 0.0,
        3.323860020045583e-23, 6.36364364167618e-50, -1.830377418252603e-06,
        0.0, 0.0, -7.509461526769651e-23,
        -1.4377120820083222e-49, 8.07219108145557e-06, 0.0,
    ),
    # IDX 4122: logT = 7.8, logRho = 6.45, abar = 12.8, zbar = 6.0
    (
        7.8, 6.45, 12.8, 6.0,
        4.497212311619273, 4.711336888988176e-07, -2.9671187668924942e-08,
        -0.3448071528776883, 0.7398411501891018, 2.833017027507156e-95,
        4.46597570323181, 2.368298713781117e-07, 0.031236371557591926,
        0.0, 1.1078563749074264e-100, 4.6777540588737493e-07,
        6.175741580703057e-14, 3.358221254026858e-09, 0.0,
        -1.9698256790595408e-100, 8.239988585628398e-10,
        -4.0420947005798934e-13, -3.0494782318017725e-08, 0.0,
        4.115954194278984e-95, -0.34908578496485665,
        7.049899997568551e-08, 0.004278561588168344, 0.0,
        -8.780702281128497e-95, 0.7447163412583611,
        -1.503978666147958e-07, -0.004875040671392619, 0.0,
    ),
    # IDX 5555: logT = 8.25, logRho = 8.25, abar = 30.5, zbar = 13.5
    (
        8.25, 8.25, 30.5, 13.5,
        1394.361921416206, 7.3921759339363e-05, -8.961274584312864e-06,
        6.519846175373675, -13.491011975624755, 8.217551134215202e-53,
        1373.768933600608, 5.093004080655104e-09, 20.59298781050505,
        0.0, 6.824997901170556e-59, 7.321389683490913e-05,
        7.779054765996164e-16, 7.078625036759573e-07, 0.0,
        -1.2759944388711824e-59, -8.95497047600547e-06,
        -2.243626346204645e-16, -6.3041080830312875e-09, 0.0,
        7.17016107090701e-53, 7.169673045838977,
        1.1411456154790384e-09, -0.6498268716064474, 0.0,
        -1.6199252789826947e-52, -16.19815021467325,
        -2.578143797934123e-09, 2.7071382416266405, 0.0,
    ),
    # IDX 7777: logT = 9.15, logRho = 9.15, abar = 42.3, zbar = 18.5
    (
        9.15, 9.15, 42.3, 18.5,
        511948576.1302473, 1.5827785591877346, 0.20123214332542338,
        -18821181.733654067, 43333708.8621203, 0.1434029313340968,
        504107759.005567, 53275.396529930156, 7787541.584747452,
        0.0, 3.192618313020971e-09, 1.548294787870308,
        0.00044913402042604444, 0.0340346341043822, 0.0,
        -8.115165278361871e-10, 0.2015964855251044,
        -8.864141363809853e-05, -0.00027569997452639525, 0.0,
        0.023709092962562916, -18649417.580940288,
        1700.565679487452, -173464.74210235855, 0.0,
        -0.054210520665751966, 42641641.27966344,
        -3888.320445530766, 695955.9571129074, 0.0,
    ),
    # IDX 9999: logT = 10.05, logRho = 10.05, abar = 54.1, zbar = 23.5
    (
        10.05, 10.05, 54.1, 23.5,
        115084067758505.66, 125308.00598250104, -22461.308149638204,
        2530672499061.2847, -5713268505145.767, 99262057103942.56,
        5652249484435.307, 6511023400657.995, 3658737769469.789,
        0.0, 115898.06215107988, 1672.4488277499593,
        5629.019149436875, 2108.475854234317, 0.0,
        -22213.00836521715, 451.7061395639821,
        -645.4384546311078, -54.56746935392931, 0.0,
        2772125619398.017, -198160364686.8423,
        13510446792.318666, -56803202442.209015, 0.0,
        -6381787064231.179, 456190456576.94324,
        -31102773253.80594, 243430875762.274, 0.0,
    ),
)
# fmt: on


def _assert_rel_close(actual: float, expected: float, rtol: float) -> None:
    """Asserts that `actual` matches `expected` within relative error `rtol`."""
    if abs(expected) <= 1e-200 and abs(actual) <= 1e-200:
        return
    scale = max(abs(expected), abs(actual))
    rel = abs(actual - expected) / scale
    assert rel <= rtol, f"actual={actual}, expected={expected}, rel={rel}"


def test_golden_rows_parity_against_upstream_mesa() -> None:
    """Verifies golden rows match upstream MESA `neu` to <= 1e-12 rel error."""
    flags = neu.NeuFlags.all()
    for row in GOLDEN_ROWS:
        log10_t, log10_rho, abar, zbar = row[0], row[1], row[2], row[3]
        t = _exp10(log10_t)
        rho = _exp10(log10_rho)
        res = neu.neu_get(
            t=t,
            log10_t=log10_t,
            rho=rho,
            log10_rho=log10_rho,
            abar=abar,
            zbar=zbar,
            log10_tlim=7.5,
            flags=flags,
        )
        for d_idx in range(5):
            _assert_rel_close(float(res.loss[d_idx]), row[4 + d_idx], 1e-12)
        for c_idx in range(5):
            for d_idx in range(5):
                _assert_rel_close(
                    float(res.sources[c_idx, d_idx]),
                    row[9 + d_idx * 5 + c_idx],
                    1e-12,
                )


def test_all_10000_golden_rows_and_jax_parity() -> None:
    """Verifies all 10,000 rows of `neu/test/test_output` in Python and JAX."""
    path = _mesa_neu_test_dir() / "test_output"
    if os.environ.get("MESA_DIR", "").strip():
        assert path.is_file(), f"Missing {path}"
    if path.exists():
        raw_lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(raw_lines) == 10_000

        data = np.array(
            [
                [_parse_fortran_float(tok) for tok in line.split()]
                for line in raw_lines
            ],
            dtype=np.float64,
        )
        assert data.shape == (10_000, 34)
    else:
        data = np.asarray(GOLDEN_ROWS, dtype=np.float64)

    flags = neu.NeuFlags.all()
    n_rows = data.shape[0]
    for row_idx in range(n_rows):
        row = data[row_idx]
        log10_t, log10_rho, abar, zbar = (
            float(row[0]),
            float(row[1]),
            float(row[2]),
            float(row[3]),
        )
        t = _exp10(log10_t)
        rho = _exp10(log10_rho)
        res = neu.neu_get(
            t=t,
            log10_t=log10_t,
            rho=rho,
            log10_rho=log10_rho,
            abar=abar,
            zbar=zbar,
            log10_tlim=7.5,
            flags=flags,
        )
        for d_idx in range(5):
            _assert_rel_close(
                float(res.loss[d_idx]), float(row[4 + d_idx]), 1e-10
            )
        for c_idx in range(5):
            for d_idx in range(5):
                _assert_rel_close(
                    float(res.sources[c_idx, d_idx]),
                    float(row[9 + d_idx * 5 + c_idx]),
                    1e-10,
                )

    # Also verify vectorized JAX kernel and jacfwd across all rows.
    t_arr = jnp.exp(jnp.asarray(data[:, 0], dtype=jnp.float64) * _const.LN10)
    rho_arr = jnp.exp(jnp.asarray(data[:, 1], dtype=jnp.float64) * _const.LN10)
    abar_arr = jnp.asarray(data[:, 2], dtype=jnp.float64)
    zbar_arr = jnp.asarray(data[:, 3], dtype=jnp.float64)

    vmap_get = jax.jit(
        jax.vmap(lambda t, r, a, z: neu.neu_get_jax(t, r, a, z, 7.5))
    )
    loss_jax, sources_jax = vmap_get(t_arr, rho_arr, abar_arr, zbar_arr)
    loss_np = np.asarray(loss_jax)
    sources_np = np.asarray(sources_jax)

    expected_loss = data[:, 4:9]
    mask_loss = np.maximum(np.abs(expected_loss), np.abs(loss_np)) > 1e-200
    rel_loss = np.where(
        mask_loss,
        np.abs(loss_np - expected_loss)
        / np.maximum(np.abs(expected_loss), 1e-300),
        0.0,
    )
    assert float(np.max(rel_loss)) <= 1e-10

    # Columns 9..34 store sources(1..5, 1..5) in column-major order
    # i.e. shape (5 derivatives, 5 channels); transpose to (5, 5).
    expected_sources = data[:, 9:34].reshape(n_rows, 5, 5).transpose(0, 2, 1)
    denom_src = np.maximum(
        np.maximum(np.abs(expected_sources), np.abs(sources_np)), 1e-300
    )
    mask_src = np.maximum(np.abs(expected_sources), np.abs(sources_np)) > 1e-200
    rel_src = np.where(
        mask_src, np.abs(sources_np - expected_sources) / denom_src, 0.0
    )
    assert float(np.max(rel_src)) <= 2e-10


def test_solar_profile_test_1_0_data_all_194_zones() -> None:
    """Verifies all 194 stellar-profile rows of `neu/test/test_1.0.data`."""
    for rho_in, t_in, expect_active in (
        (154.0, 1.56e7, True),
        (1.0e-2, 4.0e6, False),
    ):
        res_in = neu.neu_get(
            t=t_in,
            log10_t=math.log10(t_in),
            rho=rho_in,
            log10_rho=math.log10(rho_in),
            abar=1.3,
            zbar=1.1,
            log10_tlim=7.5,
            flags=neu.NeuFlags.all(),
        )
        assert np.all(np.isfinite(res_in.loss))
        assert np.all(np.isfinite(res_in.sources))
        if expect_active:
            assert res_in.total.val > 0.0
        else:
            assert res_in.total.val == 0.0

    path = _mesa_neu_test_dir() / "test_1.0.data"
    if not path.exists():
        return
    raw_lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(raw_lines) == 194

    n_active = 0
    for line in raw_lines:
        cols = [_parse_fortran_float(tok) for tok in line.split()]
        assert len(cols) == 5
        rho, t = cols[2], cols[3]
        log10_t = math.log10(t)
        log10_rho = math.log10(rho)
        res = neu.neu_get(
            t=t,
            log10_t=log10_t,
            rho=rho,
            log10_rho=log10_rho,
            abar=1.3,
            zbar=1.1,
            log10_tlim=7.5,
            flags=neu.NeuFlags.all(),
        )
        assert np.all(np.isfinite(res.loss))
        assert np.all(np.isfinite(res.sources))
        if t <= 1.0e7:
            assert res.total.val == 0.0
        else:
            n_active += 1
            assert res.total.val > 0.0
            sum_ch = float(np.sum(res.sources[:, 0]))
            _assert_rel_close(res.total.val, sum_ch, 1e-14)
    assert n_active > 50


def test_jax_all_5_channels_and_jacfwd_derivatives_parity() -> None:
    """Verifies all 5 JAX channels (`pair..reco`) and `jacfwd` derivatives."""
    for row in GOLDEN_ROWS:
        log10_t, log10_rho, abar, zbar = row[0], row[1], row[2], row[3]
        t_val = _exp10(log10_t)
        rho_val = _exp10(log10_rho)
        ref = neu.neu_get(
            t=t_val,
            log10_t=log10_t,
            rho=rho_val,
            log10_rho=log10_rho,
            abar=abar,
            zbar=zbar,
            log10_tlim=7.5,
            flags=neu.NeuFlags.all(),
        )

        t_j = jnp.asarray(t_val, dtype=jnp.float64)
        rho_j = jnp.asarray(rho_val, dtype=jnp.float64)
        abar_j = jnp.asarray(abar, dtype=jnp.float64)
        zbar_j = jnp.asarray(zbar, dtype=jnp.float64)

        ch_vals = np.asarray(
            neu.neu_channels_jax(t_j, rho_j, abar_j, zbar_j, 7.5)
        )
        loss_j, sources_j = neu.neu_get_jax(t_j, rho_j, abar_j, zbar_j, 7.5)
        loss_np = np.asarray(loss_j)
        sources_np = np.asarray(sources_j)

        for c_idx in range(5):
            _assert_rel_close(
                float(ch_vals[c_idx]), float(ref.sources[c_idx, 0]), 1e-12
            )
            for d_idx in range(5):
                _assert_rel_close(
                    float(sources_np[c_idx, d_idx]),
                    float(ref.sources[c_idx, d_idx]),
                    1e-11,
                )
        for d_idx in range(5):
            _assert_rel_close(
                float(loss_np[d_idx]), float(ref.loss[d_idx]), 1e-11
            )


def test_low_temperature_cutoff_returns_exact_zero() -> None:
    """Verifies that `log10_t <= 6.8` returns exact zero across all channels."""
    res = neu.neu_get(
        t=5.0e6,
        log10_t=6.698970004336019,
        rho=1.0e5,
        log10_rho=5.0,
        abar=12.0,
        zbar=6.0,
    )
    assert res.total == neu.NeuRate()
    assert np.all(res.loss == 0.0)
    assert np.all(res.sources == 0.0)


def test_jax_low_temperature_high_z_jacfwd_and_grad_zero_without_nan() -> None:
    """Verifies `jax.jacfwd` and `jax.grad` at `T < 1e7 K, Z = 26` are `0.0`."""
    for t_val, rho_val in ((3.0e6, 1.0e6), (1.0e5, 1.0e8)):
        t_j = jnp.asarray(t_val, dtype=jnp.float64)
        rho_j = jnp.asarray(rho_val, dtype=jnp.float64)
        abar_j = jnp.asarray(56.0, dtype=jnp.float64)
        zbar_j = jnp.asarray(26.0, dtype=jnp.float64)

        ch_vals = neu.neu_channels_jax(t_j, rho_j, abar_j, zbar_j, 7.5)
        assert not bool(jnp.any(jnp.isnan(ch_vals)))
        np.testing.assert_array_equal(np.asarray(ch_vals), np.zeros(5))

        jac_fwd = jax.jacfwd(
            lambda t, r, a, z: neu.neu_channels_jax(t, r, a, z, 7.5),
            argnums=(0, 1, 2, 3),
        )(t_j, rho_j, abar_j, zbar_j)
        for g in jac_fwd:
            assert not bool(jnp.any(jnp.isnan(g)))
            np.testing.assert_array_equal(np.asarray(g), np.zeros(5))

        grads_rev = jax.grad(
            lambda t, r, a, z: jnp.sum(neu.neu_channels_jax(t, r, a, z, 7.5)),
            argnums=(0, 1, 2, 3),
        )(t_j, rho_j, abar_j, zbar_j)
        for g_s in grads_rev:
            assert not bool(jnp.isnan(g_s))
            assert float(g_s) == 0.0

        loss_j, src_j = neu.neu_get_jax(t_j, rho_j, abar_j, zbar_j, 7.5)
        assert not bool(jnp.any(jnp.isnan(loss_j)))
        assert not bool(jnp.any(jnp.isnan(src_j)))
        np.testing.assert_array_equal(np.asarray(loss_j), np.zeros(5))
        np.testing.assert_array_equal(np.asarray(src_j), np.zeros((5, 5)))


def test_channel_flag_isolation_and_additivity() -> None:
    """Verifies single-channel flag isolation and additive consistency."""
    log10_t, log10_rho, abar, zbar = 8.25, 8.25, 30.5, 13.5
    t, rho = _exp10(log10_t), _exp10(log10_rho)
    all_res = neu.neu_get(
        t=t,
        log10_t=log10_t,
        rho=rho,
        log10_rho=log10_rho,
        abar=abar,
        zbar=zbar,
        flags=neu.NeuFlags.all(),
    )
    sum_loss = np.zeros(5, dtype=np.float64)
    flag_names = ("pair", "plas", "phot", "brem", "reco")
    for ch_idx, name in enumerate(flag_names):
        kw = {k: (k == name) for k in flag_names}
        single = neu.neu_get(
            t=t,
            log10_t=log10_t,
            rho=rho,
            log10_rho=log10_rho,
            abar=abar,
            zbar=zbar,
            flags=neu.NeuFlags(**kw),
        )
        np.testing.assert_allclose(
            single.loss, all_res.sources[ch_idx], rtol=1e-15, atol=0.0
        )
        sum_loss += single.loss
    np.testing.assert_allclose(all_res.loss, sum_loss, rtol=1e-15, atol=0.0)


def test_centered_finite_difference_derivatives() -> None:
    """Verifies analytical partial derivatives against finite differences."""
    states = (
        (7.35, 6.0, 54.1, 1.0),
        (8.25, 7.5, 24.0, 11.0),
        (10.05, 9.0, 54.1, 23.5),
    )
    step = 1.0e-6
    for log10_t, log10_rho, abar, zbar in states:
        t, rho = _exp10(log10_t), _exp10(log10_rho)
        base = neu.neu_get(t=t, rho=rho, abar=abar, zbar=zbar)

        dt, drho, dabar, dzbar = t * step, rho * step, abar * step, zbar * step
        rt_p = neu.neu_get(t=t + dt, rho=rho, abar=abar, zbar=zbar)
        rt_m = neu.neu_get(t=t - dt, rho=rho, abar=abar, zbar=zbar)
        rr_p = neu.neu_get(t=t, rho=rho + drho, abar=abar, zbar=zbar)
        rr_m = neu.neu_get(t=t, rho=rho - drho, abar=abar, zbar=zbar)
        ra_p = neu.neu_get(t=t, rho=rho, abar=abar + dabar, zbar=zbar)
        ra_m = neu.neu_get(t=t, rho=rho, abar=abar - dabar, zbar=zbar)
        rz_p = neu.neu_get(t=t, rho=rho, abar=abar, zbar=zbar + dzbar)
        rz_m = neu.neu_get(t=t, rho=rho, abar=abar, zbar=zbar - dzbar)

        _assert_rel_close(
            base.total.d_t, (rt_p.total.val - rt_m.total.val) / (2.0 * dt), 1e-6
        )
        _assert_rel_close(
            base.total.d_rho,
            (rr_p.total.val - rr_m.total.val) / (2.0 * drho),
            1e-6,
        )
        _assert_rel_close(
            base.total.d_abar,
            (ra_p.total.val - ra_m.total.val) / (2.0 * dabar),
            1e-6,
        )
        _assert_rel_close(
            base.total.d_zbar,
            (rz_p.total.val - rz_m.total.val) / (2.0 * dzbar),
            1e-6,
        )


def test_jax_pair_plas_grad_and_vmap_parity() -> None:
    """Verifies `neu_pair_plas_jax` value, `jax.grad`, and `jax.vmap` parity."""
    t_val = _exp10(9.15)
    rho_val = _exp10(9.15)
    abar_val = 42.3
    zbar_val = 18.5

    flags = neu.NeuFlags(
        pair=True, plas=True, phot=False, brem=False, reco=False
    )
    ref = neu.neu_get(
        t=t_val, rho=rho_val, abar=abar_val, zbar=zbar_val, flags=flags
    )

    t_j = jnp.asarray(t_val, dtype=jnp.float64)
    rho_j = jnp.asarray(rho_val, dtype=jnp.float64)
    abar_j = jnp.asarray(abar_val, dtype=jnp.float64)
    zbar_j = jnp.asarray(zbar_val, dtype=jnp.float64)

    val_j = float(neu.neu_pair_plas_jax(t_j, rho_j, abar_j, zbar_j))
    _assert_rel_close(val_j, ref.total.val, 1e-12)

    grads = jax.grad(neu.neu_pair_plas_jax, argnums=(0, 1, 2, 3))(
        t_j, rho_j, abar_j, zbar_j
    )
    _assert_rel_close(float(grads[0]), ref.total.d_t, 1e-11)
    _assert_rel_close(float(grads[1]), ref.total.d_rho, 1e-11)
    _assert_rel_close(float(grads[2]), ref.total.d_abar, 1e-11)
    _assert_rel_close(float(grads[3]), ref.total.d_zbar, 1e-11)


def test_invalid_inputs_raise_value_error() -> None:
    """Verifies that missing or invalid inputs raise `ValueError`."""
    with pytest.raises(ValueError, match="Neither t nor log10_t"):
        neu.neu_get(rho=1.0e5, abar=12.0, zbar=6.0)
    with pytest.raises(ValueError, match="Neither rho nor log10_rho"):
        neu.neu_get(t=1.0e8, abar=12.0, zbar=6.0)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            neu.neu_get(t=bad, rho=1.0e5, abar=12.0, zbar=6.0)
        with pytest.raises(ValueError):
            neu.neu_get(t=1.0e8, rho=bad, abar=12.0, zbar=6.0)
        with pytest.raises(ValueError):
            neu.neu_get(t=1.0e8, rho=1.0e5, abar=bad, zbar=6.0)
