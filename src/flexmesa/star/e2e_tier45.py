# SPDX-License-Identifier: LGPL-3.0-or-later
"""Stage `E2E-4` (`#95`, closing `#4` and `#5`) Python wrappers for linear
oscillation mode solvers (`GYRE`/`ADIPLS` parity), `RSP` linear non-adiabatic
(`LINA`) and nonlinear envelope builders, `astero/test_suite` runners, and
`star/dev_cases_*` pulsation suite runners (`flexmesa.star.e2e_tier45`).
"""

from __future__ import annotations

import pathlib
from typing import Any

from flexmesa import _rust
from flexmesa.star import model as star_model

TIER4_STAR_CASES: tuple[str, ...] = tuple(_rust.star_tier4_star_case_names())
TIER4_ACTIVE_STAR_CASES: tuple[str, ...] = tuple(
    _rust.star_tier4_active_star_case_names()
)
TIER4_ASTERO_CASES: tuple[str, ...] = tuple(
    _rust.star_tier4_astero_case_names()
)
TIER4_DEV_CASES: tuple[tuple[str, str], ...] = tuple(
    (str(s), str(c)) for s, c in _rust.star_tier4_dev_case_names()
)


def eval_linear_oscillation_modes(
    model: star_model.StarModel,
    l_deg: int,
    n_orders: list[int],
) -> list[dict[str, Any]]:
    """Evaluate linearized adiabatic and non-adiabatic oscillation eigenmodes
    (`GYRE`/`ADIPLS` parity, Issue `#4`).

    Args:
        model: Active `StarModel` instance.
        l_deg: Spherical harmonic degree `l >= 0`.
        n_orders: Radial order indices `n_pg` (`> 0` for acoustic p-modes,
            `< 0` for gravity g-modes).

    Returns:
        List of dictionaries with keys `("l", "n_pg", "freq_hz", "freq_uhz",
        "period_days", "growth_rate_s", "growth_timescale_days", "inertia")`.
    """
    raw = _rust.star_eval_linear_oscillation_modes(
        model._rust_model, int(l_deg), [int(n) for n in n_orders]
    )
    return [dict(item) for item in raw]


def eval_target_mode_frequency_and_growth(
    model: star_model.StarModel,
) -> tuple[dict[str, Any], bool]:
    """Evaluate the target oscillation mode specified by `x_integer_ctrl(3..4)`
    and `x_ctrl(1..2)` on `model` (`gyre_in_mesa_*`).

    Args:
        model: Active `StarModel` instance.

    Returns:
        Tuple `(mode_dict, matched_target)`.
    """
    d, matched = _rust.star_eval_target_mode_frequency_and_growth(
        model._rust_model
    )
    return dict(d), bool(matched)


def eval_astero_scaling_dnu_numax_uhz(
    model: star_model.StarModel,
) -> tuple[float, float]:
    """Evaluate global asteroseismic scaling parameters
    $(\\Delta\\nu, \\nu_{\\max})$ in $\\mu\\mathrm{Hz}$ on `model`.

    Args:
        model: Active `StarModel` instance.

    Returns:
        Tuple `(delta_nu_uhz, nu_max_uhz)` in `uHz`.
    """
    dnu, numax = _rust.star_eval_astero_scaling_dnu_numax_uhz(model._rust_model)
    return float(dnu), float(numax)


def eval_rsp_lina_modes(
    m_msun: float,
    l_lsun: float,
    teff_k: float,
    x_h: float = 0.73,
    z_metal: float = 0.007,
    n_modes: int = 3,
    expected_f_period_days: float | None = None,
) -> list[dict[str, Any]]:
    """Evaluate linear non-adiabatic (`LINA`) radial pulsation periods and
    growth rates for an `RSP` envelope (`Issue #5`).

    Args:
        m_msun: Total stellar mass in $M_\\odot$.
        l_lsun: Surface luminosity in $L_\\odot$.
        teff_k: Effective temperature in `K`.
        x_h: Hydrogen mass fraction $X$.
        z_metal: Metal mass fraction $Z$.
        n_modes: Number of radial modes to compute (`1..=5`).
        expected_f_period_days: Optional target fundamental period in `days`.

    Returns:
        List of dictionaries with keys `("mode_index", "period_days",
        "growth_rate")`.
    """
    raw = _rust.star_eval_rsp_lina_modes(
        float(m_msun),
        float(l_lsun),
        float(teff_k),
        float(x_h),
        float(z_metal),
        int(n_modes),
        None
        if expected_f_period_days is None
        else float(expected_f_period_days),
    )
    return [dict(item) for item in raw]


def eval_rsp_2nd_crossing_edges(
    case_dir: str | pathlib.Path,
    delta_teff: float = 100.0,
    history_filename: str = "history_7M.data",
) -> tuple[float, float, float]:
    """Evaluate 2nd-crossing instability strip blue-edge and red-edge
    temperatures (`rsp_check_2nd_crossing`).

    Args:
        case_dir: Path to the `rsp_check_2nd_crossing` directory.
        delta_teff: Temperature grid spacing in `K`.
        history_filename: Input evolutionary track history file name.

    Returns:
        3-tuple `(teff_blue_k, teff_red_k, delta_teff_k)`.
    """
    tb, tr, dt = _rust.star_eval_rsp_2nd_crossing_edges(
        str(case_dir), float(delta_teff), str(history_filename)
    )
    return float(tb), float(tr), float(dt)


def build_rsp_envelope_model(
    m_msun: float = 4.165,
    l_lsun: float = 1438.8,
    teff_k: float = 6050.0,
    x_h: float = 0.73,
    z_metal: float = 0.007,
) -> star_model.StarModel:
    """Build a static radial stellar pulsation (`RSP`) envelope `StarModel`
    (`create_RSP_model = .true.`, Issue `#5`).

    Args:
        m_msun: Envelope total stellar mass in $M_\\odot$.
        l_lsun: Envelope surface luminosity in $L_\\odot$.
        teff_k: Envelope effective temperature in `K`.
        x_h: Hydrogen mass fraction $X$.
        z_metal: Metal mass fraction $Z$.

    Returns:
        Initialized `StarModel` with `RSP_flag = True`.
    """
    rust_mod = _rust.star_build_rsp_envelope_model(
        float(m_msun),
        float(l_lsun),
        float(teff_k),
        float(x_h),
        float(z_metal),
    )
    return star_model.StarModel(rust_mod)


def run_tier4_star_suite(
    mesa_dir: str | pathlib.Path,
    max_steps_per_stage: int | None = 15,
    verify_restart: bool = True,
) -> list[dict[str, Any]]:
    """Execute all 16 Tier 4 `star/test_suite` cases under `mesa_dir`.

    Args:
        mesa_dir: Root path of the MESA installation.
        max_steps_per_stage: Optional cap on evolutionary steps per stage.
        verify_restart: Whether to verify bit-identical `.photo` restart parity.

    Returns:
        List of 16 `CaseRunReport` dictionaries.
    """
    raw = _rust.star_run_tier4_star_suite(
        str(mesa_dir),
        True,
        None if max_steps_per_stage is None else int(max_steps_per_stage),
        bool(verify_restart),
    )
    return [dict(item) for item in raw]


def run_astero_test_suite_case(
    case_dir: str | pathlib.Path,
    skip_optional: bool = True,
    max_steps_per_stage: int | None = 15,
    verify_restart: bool = True,
) -> dict[str, Any]:
    """Execute a single `astero/test_suite` case directory.

    Args:
        case_dir: Path to the `astero/test_suite/<case>` directory.
        skip_optional: Whether `MESA_SKIP_OPTIONAL` branches are active.
        max_steps_per_stage: Optional cap on evolutionary steps per stage.
        verify_restart: Whether to verify bit-identical `.photo` restart parity.

    Returns:
        `CaseRunReport` dictionary.
    """
    return dict(
        _rust.star_run_astero_test_suite_case(
            str(case_dir),
            bool(skip_optional),
            None if max_steps_per_stage is None else int(max_steps_per_stage),
            bool(verify_restart),
        )
    )


def run_astero_suite(
    mesa_dir: str | pathlib.Path,
    max_steps_per_stage: int | None = 15,
    verify_restart: bool = True,
) -> list[dict[str, Any]]:
    """Execute all 8 `astero/test_suite` cases under `mesa_dir`.

    Args:
        mesa_dir: Root path of the MESA installation.
        max_steps_per_stage: Optional cap on evolutionary steps per stage.
        verify_restart: Whether to verify bit-identical `.photo` restart parity.

    Returns:
        List of 8 `CaseRunReport` dictionaries.
    """
    raw = _rust.star_run_astero_suite(
        str(mesa_dir),
        True,
        None if max_steps_per_stage is None else int(max_steps_per_stage),
        bool(verify_restart),
    )
    return [dict(item) for item in raw]


def run_tier4_suite(
    mesa_dir: str | pathlib.Path,
    max_steps_per_stage: int | None = 15,
    verify_restart: bool = True,
) -> list[dict[str, Any]]:
    """Execute all 24 Tier 4 `test_suite` cases (`16` `star` + `8` `astero`)
    under `mesa_dir`.

    Args:
        mesa_dir: Root path of the MESA installation.
        max_steps_per_stage: Optional cap on evolutionary steps per stage.
        verify_restart: Whether to verify bit-identical `.photo` restart parity.

    Returns:
        List of 24 `CaseRunReport` dictionaries.
    """
    raw = _rust.star_run_tier4_suite(
        str(mesa_dir),
        True,
        None if max_steps_per_stage is None else int(max_steps_per_stage),
        bool(verify_restart),
    )
    return [dict(item) for item in raw]


def run_dev_cases_suite(
    mesa_dir: str | pathlib.Path,
    max_steps_per_stage: int | None = 10,
    verify_restart: bool = True,
) -> list[dict[str, Any]]:
    """Execute all 32 `star/dev_cases_*` pulsation cases under `mesa_dir`.

    Args:
        mesa_dir: Root path of the MESA installation.
        max_steps_per_stage: Optional cap on evolutionary steps per stage.
        verify_restart: Whether to verify bit-identical `.photo` restart parity.

    Returns:
        List of 32 `CaseRunReport` dictionaries.
    """
    raw = _rust.star_run_dev_cases_suite(
        str(mesa_dir),
        True,
        None if max_steps_per_stage is None else int(max_steps_per_stage),
        bool(verify_restart),
    )
    return [dict(item) for item in raw]
