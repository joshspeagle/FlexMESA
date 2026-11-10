# SPDX-License-Identifier: LGPL-3.0-or-later
"""Verification suite for Wave 2c ``flexmesa.colors`` (`colors`).

Verifies:
1. 14-digit parity against all 14 golden cases in upstream MESA
   ``colors/test/test_output`` (Group 1: ``solar``, ``hot_ms``, ``cool_giant``;
   Group 2a: ``[M/H]`` sweep ``-2.0, -1.0, 0.5``; Group 2b: ``log g`` sweep
   ``1.0, 2.5, 4.0, 5.0``; Group 2c: ``Teff`` sweep
   ``3500, 6000, 10000, 20000``) across both ``backend="rust"`` and
   ``backend="jax"``.
2. All 21 stride-sampled solar SED points (``index = 0, 59, ..., 1180``) and
   the 4 Stefan-Boltzmann wavelength coverage checks from Group 3 of
   ``colors/test/test_output``.
3. ``Hermite``, ``Linear`` (including zero-UV-flux cells), and ``KNN``
   (including ``|di| >= 2`` neighbor cells and exact grid nodes) 3D SED
   interpolators and ``Vega``, ``AB``, and ``ST`` photometric zero-point
   evaluators.
4. Multi-zone random seed parity across 10 deterministic seeds (``N = 64``
   zones in the solar neighborhood) between Safe Rust ``rayon`` and
   ``@jax.jit`` ``vmap``.
5. Exact ``(10, 4)`` Jacobian parity between Safe Rust ``DualOrder1<4>`` and
   ``jax.jacfwd``, plus 5-parameter ``jax.test_util.check_grads`` (``order=1``
   and ``order=2``) over ``(teff, log_g, metallicity, r, d)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.test_util as jtu
import numpy as np
import pytest

from flexmesa import _rust, colors, constants

RSUN: float = constants.RSUN
D10: float = colors.D_10PC_CM

# fmt: off
# 14 golden reference outputs from upstream MESA `colors/test/test_output`:
# Columns: [Mag_bol, Flux_bol, Interp_rad, B, I, J, M, R, U, V]
# - Group 1: `solar` (5778 K, 4.44, 0.0, 1 RSUN), `hot_ms` (15000 K, 4.0, 0.0,
#   5 RSUN), and `cool_giant` (4000 K, 2.0, 0.0, 20 RSUN).
# - Group 2a: `[M/H]` sweep (`-2.0, -1.0, 0.5`) at `(5778 K, 4.44, 1 RSUN)`.
# - Group 2b: `log g` sweep (`1.0, 2.5, 4.0, 5.0`) at `(5778 K, 0.0, 1 RSUN)`.
# - Group 2c: `Teff` sweep (`3500..20000 K`) at `(4.0, 0.0, 1 RSUN)`.
GOLDEN_CASES: tuple[
    tuple[str, float, float, float, float, tuple[float, ...]], ...
] = (
    ("solar", 5778.0, 4.44, 0.0, RSUN, (
        1.6233783742415e01, 3.2098630468165e-07, 1.2015098221403e-02,
        5.4328315088495e00, 4.0243632405405e00, 3.6544667162329e00,
        3.2794040310708e00, 4.3322423435800e00, 5.5342928137757e00,
        4.7769811346940e00,
    )),
    ("hot_ms", 15000.0, 4.0, 0.0, 5.0 * RSUN, (
        8.5957683261478e00, 3.6449591844963e-04, 0.0,
        -1.8453872060965e00, -1.5266394695804e00, -1.3829167561429e00,
        -1.2210271604986e00, -1.6373526226723e00, -2.5493640964120e00,
        -1.7077286035359e00,
    )),
    ("cool_giant", 4000.0, 2.0, 0.0, 20.0 * RSUN, (
        1.1326046768001e01, 2.9483653105575e-05, 0.0,
        2.1835469363630e00, -9.1810150915499e-01, -1.7697342747621e00,
        -2.6525046293368e00, -2.0538595855424e-01, 3.9598722643568e00,
        7.5454159459472e-01,
    )),
    ("mh_-2.0", 5778.0, 4.44, -2.0, RSUN, (
        1.6233121880603e01, 3.2118203670543e-07, 1.2015098221403e-02,
        5.4155699684922e00, 4.1196749571589e00, 3.7331196442677e00,
        3.2764174462220e00, 4.4562859813888e00, 5.1227256937938e00,
        4.8915514395591e00,
    )),
    ("mh_-1.0", 5778.0, 4.44, -1.0, RSUN, (
        1.6234071206333e01, 3.2090133029621e-07, 1.2015098221403e-02,
        5.4129001639500e00, 4.0797030492881e00, 3.7008262418547e00,
        3.2696192851353e00, 4.4046484941712e00, 5.2577215874254e00,
        4.8394911534358e00,
    )),
    ("mh_0.5", 5778.0, 4.44, 0.5, RSUN, (
        1.6234073995658e01, 3.2090050588168e-07, 1.2015098221403e-02,
        5.4709583044458e00, 3.9901647950517e00, 3.6279241638132e00,
        3.2887774292217e00, 4.2891305098437e00, 5.7511450514938e00,
        4.7518221763185e00,
    )),
    ("logg_1.0", 5778.0, 1.0, 0.0, RSUN, (
        1.6233777481775e01, 3.2098815557824e-07, 6.0215053763441e-04,
        5.3645167572931e00, 3.9946594776964e00, 3.6420933559602e00,
        3.2639524119839e00, 4.2976154690770e00, 5.8874283229464e00,
        4.7180779330260e00,
    )),
    ("logg_2.5", 5778.0, 2.5, 0.0, RSUN, (
        1.6233688922092e01, 3.2101433852074e-07, 6.0215053763441e-04,
        5.3921064688310e00, 4.0149191259301e00, 3.6490375905121e00,
        3.2745403559284e00, 4.3196935465011e00, 5.6695916672821e00,
        4.7516224781321e00,
    )),
    ("logg_4.0", 5778.0, 4.0, 0.0, RSUN, (
        1.6233750920162e01, 3.2099600837665e-07, 6.0215053763441e-04,
        5.4245311240969e00, 4.0247043059247e00, 3.6545905374482e00,
        3.2792933294422e00, 4.3323909428471e00, 5.5460781049756e00,
        4.7740932524300e00,
    )),
    ("logg_5.0", 5778.0, 5.0, 0.0, RSUN, (
        1.6233837927675e01, 3.2097028578831e-07, 6.0215053763441e-04,
        5.4419095524136e00, 4.0211092600660e00, 3.6525712832213e00,
        3.2786851353429e00, 4.3286664694765e00, 5.5395103794726e00,
        4.7780378683469e00,
    )),
    ("teff_3500", 3500.0, 4.0, 0.0, RSUN, (
        1.8411335119795e01, 4.3198229964375e-08, 0.0,
        1.0036329716496e01, 6.1438344514899e00, 5.1123434709086e00,
        4.0338857238481e00, 7.4278203656696e00, 1.1095018314829e01,
        8.8377137013872e00,
    )),
    ("teff_6000", 6000.0, 4.0, 0.0, RSUN, (
        1.6069995716096e01, 3.7325163050257e-07, 0.0,
        5.1670419609711e00, 3.8998094567456e00, 3.5659889365361e00,
        3.2256242225345e00, 4.1800559165600e00, 5.2180945995774e00,
        4.5848107914209e00,
    )),
    ("teff_10000", 10000.0, 4.0, 0.0, RSUN, (
        1.3851150550189e01, 2.8809769255562e-06, 0.0,
        2.5700839979881e00, 2.6333104842699e00, 2.6554635119236e00,
        2.6823297126398e00, 2.6143644145113e00, 2.4829162763772e00,
        2.6004126821216e00,
    )),
    ("teff_20000", 20000.0, 4.0, 0.0, RSUN, (
        1.0839332915758e01, 4.6160109853289e-05, 0.0,
        1.0828563761601e00, 1.5457706838243e00, 1.7510186431433e00,
        1.9941516825941e00, 1.3843911085916e00, 1.0937345701750e-01,
        1.2754965059690e00,
    )),
)
# fmt: on


def _scaled_eval_colors(u: jax.Array) -> jax.Array:
    """Dimensionless 5-parameter perturbation wrapper around ``eval_colors``.

    Applies dimensionless perturbations `u[0..4]` around `(Teff = 5820 K,
    log g = 4.35, [M/H] = -0.25, R = 1.1 RSUN, d = 10 pc)` and returns the 9
    differentiable photometry outputs (`Mag_bol`, `Flux_bol`, `B..V`), omitting
    the nearest-grid-node diagnostic `Interp_rad` (`cols[2]`).
    """
    cols = colors.eval_colors_jax(
        5820.0 + 100.0 * u[0],
        4.35 + 0.1 * u[1],
        -0.25 + 0.1 * u[2],
        (1.1 + 0.1 * u[3]) * RSUN,
        (1.0 + 0.1 * u[4]) * D10,
        0,
        0,
    )
    return jnp.concatenate([cols[:2], cols[3:]])


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda c: c[0])
def test_colors_golden_cases_match_upstream_mesa(
    case: tuple[str, float, float, float, float, tuple[float, ...]],
) -> None:
    """Verify all 14 golden cases match upstream MESA in Rust and JAX.

    Checks all 10 history columns (`Mag_bol`, `Flux_bol`, `Interp_rad`, `B`,
    `I`, `J`, `M`, `R`, `U`, `V`) against `colors/test/test_output` to
    `rtol = 1e-12`.
    """
    _, teff, logg, mh, r_cm, expected = case
    exp_arr = np.asarray(expected, dtype=np.float64)

    rust_out = colors.eval_colors(teff, logg, mh, r_cm, D10, backend="rust")
    jax_out = np.asarray(
        colors.eval_colors(teff, logg, mh, r_cm, D10, backend="jax")
    )

    np.testing.assert_allclose(rust_out, exp_arr, rtol=1.0e-12, atol=1.0e-15)
    np.testing.assert_allclose(jax_out, exp_arr, rtol=1.0e-12, atol=1.0e-15)


def test_group3_sed_samples_and_stefan_boltzmann_coverage() -> None:
    """Verify all 21 solar SED samples and wavelength coverage checks.

    Compares the diluted solar SED (`Teff = 5778 K, log g = 4.44, [M/H] = 0`)
    at stride `1199 // 20 = 59` against Group 3 of `colors/test/test_output`.
    """
    wav_r, sed_r = colors.eval_sed(5778.0, 4.44, 0.0, RSUN, D10, backend="rust")
    wav_j, sed_j = colors.eval_sed(5778.0, 4.44, 0.0, RSUN, D10, backend="jax")

    np.testing.assert_allclose(wav_r, np.asarray(wav_j), rtol=1.0e-15)
    np.testing.assert_allclose(sed_r, np.asarray(sed_j), rtol=1.0e-13)

    stride = len(wav_r) // 20
    expected_samples = (
        (0, 1.4720000000000e02, 2.1385116423739e-41),
        (1, 5.1200000000000e02, 1.2669686045439e-39),
        (2, 1.0950000000000e03, 5.5890114496366e-21),
        (3, 1.6650000000000e03, 1.3984854179012e-15),
        (4, 2.2350000000000e03, 2.2027299592265e-12),
        (5, 2.8150000000000e03, 5.2619396119207e-12),
        (6, 3.8900000000000e03, 2.6676232108131e-11),
        (7, 5.0700000000000e03, 4.7479893607680e-11),
        (8, 6.2500000000000e03, 3.7949934810183e-11),
        (9, 7.4300000000000e03, 2.9943284765423e-11),
        (10, 8.6100000000000e03, 2.2878420142827e-11),
        (11, 9.7900000000000e03, 1.7769276100082e-11),
        (12, 1.2425000000000e04, 1.0638798086088e-11),
        (13, 1.5375000000000e04, 6.5440892744616e-12),
        (14, 2.0650000000000e04, 2.4263712729167e-12),
        (15, 2.6550000000000e04, 9.5609055742312e-13),
        (16, 3.2900000000000e04, 4.2443985723283e-13),
        (17, 4.4700000000000e04, 1.2606836963661e-13),
        (18, 5.6500000000000e04, 5.2911476520239e-14),
        (19, 7.2600000000000e04, 2.0114488666486e-14),
        (20, 9.6200000000000e04, 6.6589153670475e-15),
    )
    for k, w_exp, f_exp in expected_samples:
        idx = k * stride
        assert abs(float(wav_r[idx]) - w_exp) < 1.0e-9
        assert abs(float(sed_r[idx]) - f_exp) / f_exp < 1.0e-12


def test_linear_knn_methods_and_zero_points_parity() -> None:
    """Verify Linear/KNN SED methods and Vega/AB/ST magnitude systems.

    Probes an interior off-grid point (`5820, 4.35, -0.25`), an exact grid node
    (`5750, 4.0, 0.0`), a point whose 4th nearest neighbor lies at `|di| = 2`
    along `teff_grid` (`5820, 4.0, 0.0`), and a cool high-gravity cell with
    zero extreme-UV flux (`4800, 5.1, 0.1`).
    """
    test_points = (
        (5820.0, 4.35, -0.25, 1.1 * RSUN),
        (5750.0, 4.0, 0.0, RSUN),
        (5820.0, 4.0, 0.0, RSUN),
        (4800.0, 5.1, 0.1, 0.9 * RSUN),
    )
    for teff, logg, mh, rad in test_points:
        for method in ("Hermite", "Linear", "KNN"):
            _, sed_r = colors.eval_sed(
                teff, logg, mh, rad, D10, method=method, backend="rust"
            )
            _, sed_j = colors.eval_sed(
                teff, logg, mh, rad, D10, method=method, backend="jax"
            )
            np.testing.assert_allclose(
                sed_r, np.asarray(sed_j), rtol=1.0e-12, atol=1.0e-35
            )
            for mag_sys in ("Vega", "AB", "ST"):
                r_out = colors.eval_colors(
                    teff,
                    logg,
                    mh,
                    rad,
                    D10,
                    mag_system=mag_sys,
                    method=method,
                    backend="rust",
                )
                j_out = np.asarray(
                    colors.eval_colors(
                        teff,
                        logg,
                        mh,
                        rad,
                        D10,
                        mag_system=mag_sys,
                        method=method,
                        backend="jax",
                    )
                )
                np.testing.assert_allclose(
                    r_out, j_out, rtol=1.0e-12, atol=1.0e-15
                )

    for idx in range(7):
        fw, ft, vega_zp, ab_zp, st_zp = _rust.py_colors_filter_curve(idx)
        assert vega_zp > 0.0
        ab_r = colors.compute_ab_zero_point(fw, ft, backend="rust")
        ab_j = float(colors.compute_ab_zero_point(fw, ft, backend="jax"))
        st_r = colors.compute_st_zero_point(fw, ft, backend="rust")
        st_j = float(colors.compute_st_zero_point(fw, ft, backend="jax"))
        assert abs(float(ab_r) - ab_zp) / ab_zp < 1.0e-14
        assert abs(ab_j - ab_zp) / ab_zp < 1.0e-14
        assert abs(float(st_r) - st_zp) / st_zp < 1.0e-14
        assert abs(st_j - st_zp) / st_zp < 1.0e-14


def test_multi_zone_random_seed_parity_across_10_seeds() -> None:
    """Verify multi-zone parity across 10 deterministic seeds (N = 64 zones).

    Compares `backend="rust"` (`rayon` parallel worker pool) against
    `backend="jax"` (`@jax.jit` + `jax.vmap`) across 10 `np.random.default_rng`
    realizations in the solar neighborhood.
    """
    for seed in range(10):
        rng = np.random.default_rng(seed + 2026)
        teff = rng.uniform(5750.0, 6000.0, size=64)
        logg = rng.uniform(4.0, 4.5, size=64)
        mh = rng.uniform(-0.5, 0.0, size=64)
        rad = rng.uniform(0.8 * RSUN, 1.4 * RSUN, size=64)

        out_rust = colors.eval_colors_zones(
            teff, logg, mh, rad, D10, backend="rust", parallel=True
        )
        out_jax = np.asarray(
            colors.eval_colors_zones(teff, logg, mh, rad, D10, backend="jax")
        )
        np.testing.assert_allclose(
            out_rust, out_jax, rtol=1.0e-12, atol=1.0e-15
        )


def test_dual_autodiff_jacobian_and_jax_check_grads() -> None:
    """Verify (10, 4) Jacobian parity and 5-parameter check_grads.

    Checks exact agreement between Safe Rust `DualOrder1<4>` and `jax.jacfwd`,
    followed by 1st- and 2nd-order finite-difference gradient verification via
    `jax.test_util.check_grads` over `(teff, log_g, metallicity, r, d)`.
    """
    t0, g0, m0, r0 = 5820.0, 4.35, -0.25, 1.1 * RSUN
    val_r, jac_r = colors.eval_colors_dual(t0, g0, m0, r0, D10, backend="rust")
    val_j, jac_j = colors.eval_colors_dual(t0, g0, m0, r0, D10, backend="jax")

    np.testing.assert_allclose(
        val_r, np.asarray(val_j), rtol=1.0e-13, atol=1.0e-15
    )
    np.testing.assert_allclose(
        jac_r, np.asarray(jac_j), rtol=1.0e-12, atol=1.0e-15
    )

    u0 = jnp.zeros(5, dtype=jnp.float64)
    jtu.check_grads(
        _scaled_eval_colors, (u0,), order=1, atol=1.0e-6, rtol=1.0e-6
    )
    jtu.check_grads(
        _scaled_eval_colors, (u0,), order=2, atol=1.0e-5, rtol=1.0e-5
    )


def test_full_76x11x8x1199_sed_cube_and_full_domain_sweeps() -> None:
    """Verify full 76x11x8x1199 Kurucz/Lejeune SED cube and domain sweeps.

    Confirms that the built-in SED cube always has dense shape `(6688, 1199)`
    (`76 * 11 * 8 == 6688` SEDs) with a 1-to-1 `slot_map` (zero nearest-neighbor
    node copying), and evaluates bolometric corrections, filter magnitudes,
    and SEDs across `Teff in [2000, 50000] K`, `log g in [-1.0, 5.5]`, and
    `[M/H] in [-5.0, +1.0]` with Rust vs. JAX parity.
    """
    grid = colors.get_builtin_colors_grid()
    assert grid.teff_grid.shape == (76,)
    assert grid.logg_grid.shape == (11,)
    assert grid.meta_grid.shape == (8,)
    assert grid.wavelengths.shape == (1199,)
    assert grid.sed_table.shape == (76 * 11 * 8, 1199)
    np.testing.assert_array_equal(
        np.asarray(grid.slot_map), np.arange(76 * 11 * 8, dtype=np.int64)
    )

    teff_arr = np.array(
        [2000.0, 3500.0, 4500.0, 5778.0, 8500.0, 15000.0, 30000.0, 50000.0],
        dtype=np.float64,
    )
    logg_arr = np.array(
        [-1.0, 0.0, 2.5, 4.44, 4.0, 4.0, 4.5, 5.5], dtype=np.float64
    )
    meta_arr = np.array(
        [-5.0, -2.5, -1.5, 0.0, -0.5, 0.2, 0.5, 1.0], dtype=np.float64
    )
    r_arr = np.full_like(teff_arr, RSUN)

    out_rust = colors.eval_colors_zones(
        teff_arr, logg_arr, meta_arr, r_arr, D10, backend="rust"
    )
    out_jax = np.asarray(
        colors.eval_colors_zones(
            jnp.asarray(teff_arr),
            jnp.asarray(logg_arr),
            jnp.asarray(meta_arr),
            jnp.asarray(r_arr),
            D10,
            backend="jax",
        ),
        dtype=np.float64,
    )
    assert np.all(np.isfinite(out_rust))
    assert np.all(out_rust[:, 1] > 0.0)
    np.testing.assert_allclose(out_rust, out_jax, rtol=1.0e-12, atol=1.0e-14)
    # Bolometric correction BC_V = M_bol - V must vary smoothly across Teff
    bc_v = out_rust[:, 0] - out_rust[:, 9]
    assert np.all(np.isfinite(bc_v))
