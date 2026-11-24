# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end MESA `inlist` parser, `./rn` stage extractor, multi-stage
`test_suite` case execution driver, and linear/envelope pulsation helpers
(`flexmesa.star.e2e`).
"""

from __future__ import annotations

import pathlib
from typing import Any

from flexmesa import _rust
from flexmesa.star import model as star_model


def copy_cleaned_case(
    src_dir: str | pathlib.Path,
    dst_dir: str | pathlib.Path,
    suite: str = "star",
) -> None:
    """Copy an upstream MESA test case directory while stripping unsupported
    and unused namelist keys for strict cleaned-copy verification.
    """
    _rust.star_copy_cleaned_case(str(src_dir), str(dst_dir), str(suite))


def parse_inlist_bundle(
    inlist_path: str | pathlib.Path,
    allow_unsupported_controls: bool = False,
) -> dict[str, Any]:
    """Parse a MESA `inlist` file and recursively resolve `read_extra_*_inlist`.

    Args:
        inlist_path: Path to the entry `inlist` file.
        allow_unsupported_controls: When ``True``, bypass strict failure on
            unread, unused, or unimplemented upstream MESA controls.

    Returns:
        Dictionary of resolved `&star_job` and `&controls` settings.
    """
    return dict(
        _rust.star_parse_inlist_bundle(
            str(inlist_path), bool(allow_unsupported_controls)
        )
    )


def parse_rn_stages(
    case_dir: str | pathlib.Path,
    skip_optional: bool = True,
) -> list[tuple[str, str, str]]:
    """Parse an upstream MESA `./rn` bash script into execution stages.

    Args:
        case_dir: Path to the `test_suite` case directory.
        skip_optional: Whether `MESA_SKIP_OPTIONAL` branches are active.

    Returns:
        List of `(stage_kind, arg1, arg2)` stage tuples.
    """
    return list(_rust.star_parse_rn_stages(str(case_dir), bool(skip_optional)))


def run_test_suite_case(
    case_dir: str | pathlib.Path,
    skip_optional: bool = True,
    max_steps_per_stage: int | None = None,
    verify_restart: bool = False,
    allow_unsupported_controls: bool = False,
    extras: Any = None,
) -> dict[str, Any]:
    """Execute a multi-stage upstream MESA `test_suite` case directory.

    Args:
        case_dir: Path to the `test_suite` case directory.
        skip_optional: Whether `MESA_SKIP_OPTIONAL` branches are active.
        max_steps_per_stage: Optional cap on evolutionary steps per stage.
        verify_restart: Whether to verify internal `.photo` round-trip
            consistency.
        allow_unsupported_controls: When ``True``, bypass strict failure on
            unread, unused, or unimplemented upstream MESA controls.
        extras: Optional Python ``run_star_extras`` object with hook methods.

    Returns:
        Structured dictionary containing per-stage reports, step counts,
        Newton iterations, final physical state, and wall-clock runtime.
    """
    return dict(
        _rust.star_run_test_suite_case(
            str(case_dir),
            bool(skip_optional),
            None if max_steps_per_stage is None else int(max_steps_per_stage),
            bool(verify_restart),
            bool(allow_unsupported_controls),
            extras,
        )
    )


def eval_linear_oscillation_modes(
    model: star_model.StarModel,
    l_deg: int,
    n_orders: list[int],
) -> list[dict[str, Any]]:
    """Evaluate linearized adiabatic and non-adiabatic oscillation eigenmodes.

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
    and `x_ctrl(1..2)` on `model`.

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
) -> list[dict[str, Any]]:
    """Evaluate linear non-adiabatic (`LINA`) radial pulsation periods and
    growth rates for an `RSP` envelope.

    Args:
        m_msun: Total stellar mass in $M_\\odot$.
        l_lsun: Surface luminosity in $L_\\odot$.
        teff_k: Effective temperature in `K`.
        x_h: Hydrogen mass fraction $X$.
        z_metal: Metal mass fraction $Z$.
        n_modes: Number of radial modes to compute (`1..=15`).

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
    )
    return [dict(item) for item in raw]


def build_rsp_envelope_model(
    m_msun: float = 4.165,
    l_lsun: float = 1438.8,
    teff_k: float = 6050.0,
    x_h: float = 0.73,
    z_metal: float = 0.007,
) -> star_model.StarModel:
    """Build a static radial stellar pulsation (`RSP`) envelope `StarModel`
    (`create_RSP_model = .true.`).

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
