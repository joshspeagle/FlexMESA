# SPDX-License-Identifier: LGPL-3.0-or-later
"""End-to-end MESA `inlist` parser, `./rn` stage extractor, multi-stage
`test_suite` case execution driver, and Stage `E2E-1` (`#92`) Tier 1 suite
runner (`flexmesa.star.e2e`).
"""

from __future__ import annotations

import pathlib
from typing import Any

from flexmesa import _rust
from flexmesa.star import model as star_model

TIER1_CASES: tuple[str, ...] = tuple(_rust.star_tier1_cases())


def parse_inlist_bundle(inlist_path: str | pathlib.Path) -> dict[str, Any]:
    """Parse a MESA `inlist` file and recursively resolve `read_extra_*_inlist`.

    Args:
        inlist_path: Path to the entry `inlist` file.

    Returns:
        Dictionary of resolved `&star_job` and `&controls` settings.
    """
    return dict(_rust.star_parse_inlist_bundle(str(inlist_path)))


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
) -> dict[str, Any]:
    """Execute a multi-stage upstream MESA `test_suite` case directory.

    Args:
        case_dir: Path to the `test_suite` case directory.
        skip_optional: Whether `MESA_SKIP_OPTIONAL` branches are active.
        max_steps_per_stage: Optional cap on evolutionary steps per stage.
        verify_restart: Whether to verify bit-identical `.photo` restart parity.

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
        )
    )


def run_tier1_suite(
    mesa_dir: str | pathlib.Path,
    max_steps_per_stage: int | None = 15,
    verify_restart: bool = True,
) -> list[dict[str, Any]]:
    """Execute all 33 Tier 1 `star/test_suite` cases under `mesa_dir`.

    Args:
        mesa_dir: Root path of the MESA installation containing
            `star/test_suite/<case>`.
        max_steps_per_stage: Optional maximum evolutionary steps per `./rn`
            stage (`None` runs until physical termination).
        verify_restart: Whether to verify bit-identical `.photo` restart parity
            (`0 ULP`) on every case.

    Returns:
        List of 33 `CaseRunReport` dictionaries in canonical `TIER1_CASES`
        order.
    """
    raw = _rust.star_run_tier1_suite(
        str(mesa_dir),
        None if max_steps_per_stage is None else int(max_steps_per_stage),
        bool(verify_restart),
    )
    return [dict(item) for item in raw]


def eval_starspot_luminosity(
    l_surf: float,
    f_spot: float,
    x_spot: float,
) -> tuple[float, float]:
    """Evaluate the YREC starspot effective unspotted luminosity (`starspots`).

    Args:
        l_surf: Bolometric surface luminosity in `erg s^-1` (or `L_sun`).
        f_spot: Fractional surface spot coverage in `[0, 1]`.
        x_spot: Spot-to-ambient temperature ratio `T_spot / T_amb`.

    Returns:
        Tuple `(l_eff, alpha_spot)` where
        `alpha_spot = f_spot * x_spot^4 + 1 - f_spot` and
        `l_eff = l_surf / alpha_spot`.
    """
    l_eff, alpha = _rust.star_eval_starspot_luminosity(
        float(l_surf), float(f_spot), float(x_spot)
    )
    return float(l_eff), float(alpha)


def eval_t_tau_gradr_factor(
    tau: float,
    scheme: str = "Eddington",
) -> tuple[float, float]:
    """Evaluate the $T(\\tau)$ Hopf function $q(\\tau)$ and
    $\\nabla_{\\mathrm{rad}}$ correction factor $1 + \\tau\\,dq/d\\tau$
    (`T_tau_gradr`).

    Args:
        tau: Optical depth $\\tau \\ge 0$ (dimensionless).
        scheme: Atmosphere $T(\\tau)$ relation (`"Eddington"`,
            `"Krishna_Swamy"`, `"solar_Hopf_grey"`, or `"Trampedach_solar"`).

    Returns:
        Tuple `(q_tau, gradr_factor)`.
    """
    q_val, fac = _rust.star_eval_t_tau_gradr_factor(float(tau), str(scheme))
    return float(q_val), float(fac)


def eval_model_colors(model: star_model.StarModel) -> tuple[float, ...]:
    """Evaluate 10-column synthetic Johnson-Cousins photometry
    (`custom_colors`).

    Args:
        model: Active `StarModel` instance.

    Returns:
        10-tuple `(Mag_bol, Flux_bol, Interp_rad, B, I, J, M, R, U, V)`.
    """
    cols = _rust.star_eval_model_colors(model._rust_model)
    return tuple(float(v) for v in cols)


def eval_solar_calibration_objective(
    log_l: float,
    log_r: float,
    zx_surf: float,
    target_zx: float = 0.0229,
) -> float:
    """Evaluate the solar calibration $\\chi^2$ objective
    (`simplex_solar_calibration`).

    Args:
        log_l: $\\log_{10}(L / L_\\odot)$.
        log_r: $\\log_{10}(R / R_\\odot)$.
        zx_surf: Surface metal-to-hydrogen ratio $(Z/X)_{\\mathrm{surf}}$.
        target_zx: Target solar $(Z/X)_\\odot$ (`0.0229` for GS98).

    Returns:
        Dimensionless $\\chi^2$ value.
    """
    return float(
        _rust.star_eval_solar_calibration_objective(
            float(log_l),
            float(log_r),
            float(zx_surf),
            float(target_zx),
        )
    )


def nelder_mead_solar_step(
    simplex: list[tuple[tuple[float, float, float], float]],
) -> tuple[float, float, float]:
    """Perform one Nelder-Mead simplex reflection step for solar calibration.

    Args:
        simplex: 4-element list of `((Y_0, Z_0, alpha_mlt), chi2)` vertices.

    Returns:
        Reflected parameter triple `(Y_0, Z_0, alpha_mlt)`.
    """
    raw_in = [
        ([float(p[0]), float(p[1]), float(p[2])], float(c)) for p, c in simplex
    ]
    res = _rust.star_nelder_mead_solar_step(raw_in)
    return float(res[0]), float(res[1]), float(res[2])
