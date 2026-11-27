# SPDX-License-Identifier: GPL-3.0-or-later
"""Standalone PyO3 boundary and analytical tests for `flexmesa._rust`."""

from __future__ import annotations

import math
import pathlib
from typing import Any

import numpy as np
import pytest

from flexmesa import _rust

_D_10PC_CM = 3.0857e19
_AVO = 6.02214076e23
_KERG = 1.380649e-16
_EMPTY_F64 = np.empty(0, dtype=np.float64)


def _arr(*vals: float) -> np.ndarray:
    """Build a C-contiguous float64 1D numpy array."""
    return np.asarray(vals, dtype=np.float64)


def test_interp_evbicub_batch_nonuniform_consistency_and_guards() -> None:
    """Non-uniform grids match `interp_evbicub_db`; bad queries/tables raise."""
    x = _arr(0.0, 0.15, 0.7, 2.0, 5.5)
    y = _arr(-2.0, -0.5, 0.25, 1.5, 4.0)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    z = (xx**2) * (yy + 3.0) - 0.5 * yy**2 + xx
    empty = _EMPTY_F64
    coeffs, ilinx, iliny = _rust.interp_mkbicub_db(
        x, y, z.ravel(), 0, empty, 0, empty, 0, empty, 0, empty
    )
    assert not ilinx and not iliny
    c_flat = np.ascontiguousarray(coeffs, dtype=np.float64).ravel()

    xq = _arr(0.05, 0.4, 1.3, 4.1)
    yq = _arr(-1.2, -0.1, 0.9, 3.2)
    got = _rust.interp_evbicub_batch(xq, yq, x, y, c_flat, False)
    expected = np.vstack(
        [
            _rust.interp_evbicub_db(
                float(qx), float(qy), x, y, ilinx, iliny, c_flat
            )
            for qx, qy in zip(xq, yq, strict=True)
        ]
    )
    np.testing.assert_allclose(got, expected, rtol=1e-14, atol=1e-14)

    empty_batch = _rust.interp_evbicub_batch(
        _EMPTY_F64, _EMPTY_F64, x, y, c_flat, False
    )
    assert empty_batch.shape == (0, 6)

    for qx, qy in [
        (_arr(math.nan), _arr(0.0)),
        (_arr(math.inf), _arr(0.0)),
        (_arr(-0.5), _arr(0.0)),
        (_arr(0.5), _arr(4.5)),
    ]:
        with pytest.raises(ValueError):
            _rust.interp_evbicub_batch(qx, qy, x, y, c_flat, False)

    bad_c = c_flat.copy()
    bad_c[0] = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        _rust.interp_evbicub_batch(xq[:1], yq[:1], x, y, bad_c, False)


def test_chem_boundary_length_and_finiteness_guards() -> None:
    """Chem wrappers reject mismatched Z/A/abundance arrays and bad T9."""
    assert _rust.py_chem_generate_nuclide_names([1, 2], [1, 4]) == ["h1", "he4"]
    assert _rust.py_chem_generate_long_nuclide_names([1], [1]) == ["hydrogen-1"]

    with pytest.raises(ValueError, match="length"):
        _rust.py_chem_generate_nuclide_names([1, 2], [1])
    with pytest.raises(ValueError, match="length"):
        _rust.py_chem_generate_long_nuclide_names([1], [1, 4])
    with pytest.raises(ValueError, match="length"):
        _rust.py_chem_get_stable_mass_frac([1, 2], [1, 4], [0.7])
    with pytest.raises(ValueError, match="finite"):
        _rust.py_chem_get_stable_mass_frac([1, 2], [1, 4], [0.7, math.nan])
    for bad_t9 in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="finite"):
            _rust.py_chem_get_partition_fcn_indx(bad_t9)


def test_colors_domain_and_radius_zero_contract() -> None:
    """Colors wrappers enforce d > 0, r >= 0, finite coords, error checking."""
    zero_r = _rust.py_colors_eval_colors(5778.0, 4.44, 0.0, 0.0, _D_10PC_CM)
    assert float(zero_r[0]) == 99.0
    assert float(zero_r[1]) == 0.0

    for bad_d in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError):
            _rust.py_colors_eval_sed(5778.0, 4.44, 0.0, 7.0e10, bad_d)
        with pytest.raises(ValueError):
            _rust.py_colors_eval_colors(5778.0, 4.44, 0.0, 7.0e10, bad_d)
        with pytest.raises(ValueError):
            _rust.py_colors_eval_colors_dual(5778.0, 4.44, 0.0, 7.0e10, bad_d)
        with pytest.raises(ValueError):
            _rust.py_colors_eval_colors_zones(
                _arr(5778.0),
                _arr(4.44),
                _arr(0.0),
                _arr(7.0e10),
                bad_d,
            )
        with pytest.raises(ValueError):
            _rust.py_colors_dilute_flux(_arr(1.0, 2.0), 7.0e10, bad_d)

    with pytest.raises(ValueError):
        _rust.py_colors_eval_sed(5778.0, 4.44, 0.0, -1.0, _D_10PC_CM)
    with pytest.raises(ValueError):
        _rust.py_colors_dilute_flux(_arr(1.0, math.nan), 7.0e10, _D_10PC_CM)
    with pytest.raises(ValueError):
        _rust.py_colors_eval_colors_zones(
            _arr(math.nan), _arr(4.44), _arr(0.0), _arr(7.0e10), _D_10PC_CM
        )


def test_ionization_domain_and_finiteness_guards() -> None:
    """Ionization dual/zones/chi wrappers reject NaN, Inf, and non-positive."""
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            _rust.py_ion_eval_ionization_dual(0.02, 0.7, bad, 5.0)
        with pytest.raises(ValueError):
            _rust.py_ion_eval_ionization_zones(
                _arr(0.02), _arr(0.7), _arr(-5.0), _arr(bad), False
            )

    for bad_pos in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError):
            _rust.py_ion_chi_info(bad_pos, 1.0, 1e6, 6.0, 1.0, 0.0)
        with pytest.raises(ValueError):
            _rust.py_ion_chi_info(1.0, bad_pos, 1e6, 6.0, 1.0, 0.0)
        with pytest.raises(ValueError):
            _rust.py_ion_chi_effective(1.0, 0.0, 0.0, 0.0, bad_pos, 2.0)
        with pytest.raises(ValueError):
            _rust.py_ion_chi_effective(1.0, 0.0, 0.0, 0.0, 1.0, bad_pos)
        with pytest.raises(ValueError):
            _rust.py_ion_eval_typical_charge(
                1, bad_pos, 0.5, 1e6, 6.0, 1.0, 0.0
            )


def test_eos_ideal_gamma_de_and_inverse_boundary_contracts() -> None:
    """EOS ideal/ion-offset/gamma_de/inverse reject bad inputs."""
    with pytest.raises(ValueError, match="equal length"):
        _rust.eos_compute_ion_offset_from_za(
            _arr(0.7, 0.3), [1], _arr(1.0, 4.0)
        )
    with pytest.raises(ValueError, match="positive"):
        _rust.eos_compute_ion_offset_from_za(_arr(1.0), [1], _arr(0.0))

    # Ideal gas single-species formula: P_gas = (N_A k_B / A) * rho * T.
    rho_val, temp_val, abar = 1.0e-5, 1.0e5, 1.0
    out_zones = _rust.eos_eval_ideal_zones(
        _arr(temp_val),
        _arr(rho_val),
        abar,
        1.0,
        _arr(1.0),
        _arr(1.00794),
        _arr(1.0),
        False,
    )
    assert out_zones.shape == (1, 78)
    expected_pgas = (_AVO * _KERG / abar) * rho_val * temp_val
    np.testing.assert_allclose(
        math.exp(float(out_zones[0, 0])), expected_pgas, rtol=1e-12
    )

    with pytest.raises(ValueError, match="equal length"):
        _rust.eos_eval_ideal_zones(
            _arr(1.0e5),
            _arr(1.0e-5),
            1.0,
            1.0,
            _arr(0.7, 0.3),
            _arr(1.0),
            _arr(1.0, 4.0),
            False,
        )
    for bad_tr in (0.0, -1.0e5, math.nan, math.inf):
        with pytest.raises(ValueError, match="strictly positive"):
            _rust.eos_eval_ideal_zones(
                _arr(bad_tr),
                _arr(1.0e-5),
                1.0,
                1.0,
                _arr(1.0),
                _arr(1.0),
                _arr(1.0),
                False,
            )

    # Closed-form gamma_de relation: T = (gamma - 1) * E / (N_A * k_B / abar).
    gamma, energy = 5.0 / 3.0, 2.5e13
    t_got, logt_got, res, _, _ = _rust.eos_eval_gamma_de_get(
        12.0, 6.0, 1.0e3, energy, gamma
    )
    t_exp = (gamma - 1.0) * energy / (_AVO * _KERG / 12.0)
    np.testing.assert_allclose(t_got, t_exp, rtol=1e-14)
    np.testing.assert_allclose(logt_got, math.log10(t_exp), rtol=1e-14)
    np.testing.assert_allclose(
        math.exp(float(res[0])), (gamma - 1.0) * energy * 1.0e3, rtol=1e-14
    )

    for bad_args in [
        (0.0, 6.0, 1.0, 1e12, 5.0 / 3.0),
        (12.0, math.nan, 1.0, 1e12, 5.0 / 3.0),
        (12.0, 6.0, 0.0, 1e12, 5.0 / 3.0),
        (12.0, 6.0, 1.0, -1.0, 5.0 / 3.0),
        (12.0, 6.0, 1.0, 1e12, 1.0),
        (12.0, 6.0, 1.0, 1e12, math.inf),
    ]:
        with pytest.raises(ValueError):
            _rust.eos_eval_gamma_de_get(*bad_args)

    cids = [1]
    xa = _arr(1.0)
    with pytest.raises(ValueError):
        _rust.eos_eval_pt_get(-1.0, 1e6, cids, xa)
    with pytest.raises(ValueError):
        _rust.eos_eval_dt_get(0.0, 1e6, cids, xa)
    with pytest.raises(ValueError):
        _rust.eos_eval_dt_get_t(-1.0, 0, 10.0, cids, xa)
    with pytest.raises(ValueError):
        _rust.eos_eval_dt_get_rho(math.nan, 0, 10.0, cids, xa)
    with pytest.raises(ValueError):
        _rust.eos_eval_pt_get_t(0.0, 0, 10.0, cids, xa)
    with pytest.raises(ValueError):
        _rust.eos_eval_pt_get_pgas(math.inf, 0, 10.0, cids, xa)


def test_rates_rev_ratio_coeff1_antisymmetry_and_guards() -> None:
    """Detailed balance coefficient obeys antisymmetry and checks lengths."""
    fwd = _rust.rates_compute_rev_ratio_coeff1(
        [12.0, 4.002603], [0.0, 0.0], [15.994915], [0.0]
    )
    rev = _rust.rates_compute_rev_ratio_coeff1(
        [15.994915], [0.0], [12.0, 4.002603], [0.0, 0.0]
    )
    np.testing.assert_allclose(fwd + rev, 0.0, atol=1e-14)

    with pytest.raises(ValueError):
        _rust.rates_compute_rev_ratio_coeff1([], [], [16.0], [0.0])
    with pytest.raises(ValueError):
        _rust.rates_compute_rev_ratio_coeff1([12.0, 4.0], [0.0], [16.0], [0.0])
    with pytest.raises(ValueError):
        _rust.rates_compute_rev_ratio_coeff1([0.0], [0.0], [16.0], [0.0])
    with pytest.raises(ValueError):
        _rust.rates_compute_rev_ratio_coeff1([12.0], [-0.5], [16.0], [0.0])


def test_star_and_binary_boundary_guards(tmp_path: pathlib.Path) -> None:
    """Star/StarLib/Binary PyO3 bindings raise `ValueError` instead of panic."""
    model = _rust.RustStarModel.from_pre_ms(1.0, 0.02)
    nz = model.nz
    sp = len(model.chem_id)

    with pytest.raises(ValueError, match="nz >= 2"):
        model.set_primary_state(
            _arr(0.0),
            _arr(12.0),
            _arr(20.0),
            _arr(1e33),
            _arr(1.0),
            np.zeros((1, sp), dtype=np.float64),
        )
    with pytest.raises(ValueError, match="overshoot"):
        model.update_mixing(1.0e8, 0.02, 0.004, "unknown_scheme")

    bad_u = np.zeros(max(1, nz - 1), dtype=np.float64)
    zeros = np.zeros(nz, dtype=np.float64)
    with pytest.raises(ValueError, match="u_cell"):
        _rust.star_eval_riemann_profile(model, bad_u, zeros, 0.0, False)

    pulse_pts = _rust.star_build_pulse_points(model, True, False, 1.0e99)
    assert pulse_pts.ndim == 2 and pulse_pts.shape[1] == 26

    # Capture a StarModelHandle via minimal synthetic trace in tmp_path.
    case_dir = tmp_path / "starlib_case"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "inlist").write_text(
        "&star_job\n"
        "  create_pre_main_sequence_model = .true.\n"
        "  save_model_when_terminate = .false.\n"
        "/\n"
        "&kap\n  Zbase = 0.02\n/\n"
        "&controls\n"
        "  initial_mass = 1.0\n"
        "  initial_z = 0.02\n"
        "  max_model_number = 1\n"
        "/\n",
        encoding="utf-8",
    )
    (case_dir / "rn").write_text(
        "#!/usr/bin/env bash\ndo_one inlist final.mod\n",
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}

    class _Probe:
        wants_star_model = True

        def extras_after_evolve(self, state: dict[str, Any]) -> None:
            captured["star"] = state["star_model"]

    _rust.star_run_test_suite_case(
        str(case_dir),
        skip_optional=True,
        max_steps_per_stage=1,
        verify_restart=False,
        allow_unsupported_controls=False,
        extras=_Probe(),
    )
    lib = _rust.StarLibModel.new_for_build(captured["star"], "basic.net", 8)
    assert len(lib.xa(0)) == lib.species
    with pytest.raises(ValueError, match="out of range"):
        lib.xa(8)
    with pytest.raises(ValueError, match="out of range"):
        lib.solve_eos_given_pgas_t(8, 5.0, 10.0, -5.0, 1e-8, 1e-8)
    with pytest.raises(ValueError, match="out of range"):
        lib.solve_eos_given_pgas_t_auto(8, 5.0, 10.0, 1e-8, 1e-8)

    # Binary Kolb optically thick RLOF must validate equal profile lengths.
    with pytest.raises(ValueError, match="equal length"):
        _rust.binary_eval_kolb_mdot_thick(
            2.0e33,
            2.0e33,
            1.5e11,
            _arr(2.0e11, 1.4e11, 1.0e11),
            _arr(1e5, 2e5),
            _arr(5e3, 6e3, 7e3),
            _arr(0.6, 0.6, 0.6),
            _arr(5.0 / 3.0, 5.0 / 3.0, 5.0 / 3.0),
        )


def test_astero_boundary_guards_and_optimizer_contract() -> None:
    """Astero wrappers catch length mismatches and avoid optimizer panics."""
    r = _arr(3.0e10, 2.0e10, 1.0e10)
    cs = _arr(1.0e7, 2.0e7, 3.0e7)
    short = _arr(1.0, 2.0)

    with pytest.raises(ValueError, match="equal length"):
        _rust.astero_eval_acoustic_profile(r, short, 0.0, 3.0e10)
    with pytest.raises(ValueError, match="equal length"):
        _rust.astero_eval_delta_pg(r, short, r, cs, 100.0, False)
    with pytest.raises(ValueError, match="equal length"):
        _rust.astero_eval_delta_pg(r, r, r, short, 100.0, True)
    with pytest.raises(ValueError, match="equal length"):
        _rust.astero_eval_glitch_acoustic_depths(r, cs, r, r, r, short, 0.0)
    with pytest.raises(ValueError, match="equal length"):
        _rust.astero_eval_radial_delta_nu_ols(_arr(1.0, 2.0, 3.0), short)
    with pytest.raises(ValueError, match="equal length"):
        _rust.astero_match_modes_by_degree(
            _arr(100.0, 200.0),
            _arr(100.0, 200.0),
            _arr(1.0),
            np.asarray([1, 2], dtype=np.int32),
        )
    with pytest.raises(ValueError, match="equal length"):
        _rust.astero_eval_surface_correction(
            "cubic",
            _arr(100.0, 200.0),
            _arr(0.1),
            _arr(100.0, 200.0),
            _arr(1.0, 1.0),
            _EMPTY_F64,
            _EMPTY_F64,
            _EMPTY_F64,
            _EMPTY_F64,
            135.0,
            3000.0,
            4.9,
            1.0,
        )
    with pytest.raises(ValueError, match="equal length"):
        _rust.astero_eval_ratios_r01_r10(
            _arr(100.0, 235.0, 370.0),
            _arr(0.1, 0.1),
            _arr(165.0, 300.0, 435.0),
            _arr(0.1, 0.1, 0.1),
        )
    with pytest.raises(ValueError, match="equal length"):
        _rust.astero_eval_ratios_r02(
            _arr(100.0, 235.0),
            _arr(0.1, 0.1),
            _arr(165.0, 300.0),
            _arr(0.1, 0.1),
            _arr(90.0),
            _arr(0.1, 0.1),
        )
    with pytest.raises(ValueError, match="equal length"):
        _rust.astero_eval_chi2(
            _arr(100.0, 200.0),
            _arr(0.1),
            _arr(100.0, 200.0),
            _EMPTY_F64,
            _EMPTY_F64,
            _EMPTY_F64,
            _EMPTY_F64,
            _EMPTY_F64,
            _EMPTY_F64,
            135.0,
            0.2,
            135.0,
            3000.0,
            10.0,
            3000.0,
            _EMPTY_F64,
            _EMPTY_F64,
            _EMPTY_F64,
            0.667,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def _quad(x: np.ndarray) -> float:
        return float((x[0] - 0.5) ** 2 + (x[1] + 0.25) ** 2)

    nm = _rust.astero_nelder_mead_minimize(
        _quad, [0.2, 0.1], [0.0, -1.0], [1.0, 1.0], 200, 1e-8, 1e-8, 1e-12, 0.0
    )
    np.testing.assert_allclose(nm["x_best"], [0.5, -0.25], atol=1e-4)

    with pytest.raises(ValueError):
        _rust.astero_nelder_mead_minimize(
            _quad, [0.5], [1.0], [0.0], 20, 1e-6, 1e-6, 1e-8, 0.0
        )
    with pytest.raises(ValueError):
        _rust.astero_powell_quadratic_minimize(
            _quad, [0.5, 0.0], [0.0], [1.0, 1.0], 20, 0.2, 1e-4
        )
    with pytest.raises(ValueError):
        _rust.astero_powell_quadratic_minimize(
            _quad, [0.5, 0.0], [0.0, -1.0], [1.0, 1.0], 20, 0.0, 1e-4
        )
    with pytest.raises(ValueError):
        _rust.astero_scan_cartesian_grid(
            _quad, [0.0, 0.0], [1.0, 1.0], [0.5, 0.0]
        )
