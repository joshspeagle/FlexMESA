# SPDX-License-Identifier: GPL-3.0-or-later
r"""Upstream Spruit-Tayler dynamo, thermohaline rotation, high-rotation profile,
diffusion smoothness, and radiative levitation `StarExtras` ports (`#231`).

Covers:
- `star/test_suite/15M_dynamo/src/run_star_extras.f90:38-212`
- `star/test_suite/1M_thermohaline/src/run_star_extras.f90:37-212`
- `star/test_suite/20M_z2m2_high_rotation/src/run_star_extras.f90:38-156`
- `star/test_suite/diffusion_smoothness/src/run_star_extras.f90:39-175`
- `star/test_suite/radiative_levitation/src/run_star_extras.f90:39-243`
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Any

from flexmesa import _rust, constants
from tests.case_extras import _helpers


def _dq_weighted_slice_avg(
    vals: Sequence[float], dq: Sequence[float], k_surf_1: int, k_cntr_1: int
) -> float:
    """Evaluate `dot_product(v(k_surf:k_cntr), dq(k_surf:k_cntr)) / sum(dq)`
    for 1-based Fortran slice bounds `k_surf_1..k_cntr_1`.
    """
    if k_surf_1 < 1 or k_cntr_1 < 1:
        raise IndexError(
            f"1-based slice bounds ({k_surf_1}, {k_cntr_1}) out of range"
        )
    if k_cntr_1 < k_surf_1:
        return _helpers.fdiv(0.0, 0.0)
    num = 0.0
    den = 0.0
    for idx in range(k_surf_1 - 1, k_cntr_1):
        dq_k = float(dq[idx])
        num += float(vals[idx]) * dq_k
        den += dq_k
    return _helpers.fdiv(num, den)


def _locate_mass_window_1based(
    m_arr: Sequence[float], nz: int, m_cntr_g: float, m_surf_g: float
) -> tuple[int, int]:
    """Sweep `k = nz, 1, -1` to find 1-based `(k_cntr, k_surf)`."""
    k_cntr = 0
    k_surf = 0
    for k_1 in range(nz, 0, -1):
        m_k = float(m_arr[k_1 - 1])
        if m_k > m_cntr_g and k_cntr == 0:
            k_cntr = k_1
        if m_k > m_surf_g and k_surf == 0:
            k_surf = k_1
    return k_cntr, k_surf


class Dynamo15MStarExtras:
    r"""Spruit-Tayler dynamo angular-momentum and $3.0..3.4\,M_\odot$ shell
    checks from `star/test_suite/15M_dynamo/src/run_star_extras.f90:38-212`.
    """

    def __init__(self) -> None:
        self.checks_okay: bool | None = None
        self.failed_checks: list[str] = []

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        """Run all 13 final-stage tolerance checks when `x_logical_ctrl(1)` is
        true (`:69-138`).
        """
        if not bool(state["x_logical_ctrl"][0]):
            return
        nz = int(state["nz"])
        j_rot = state["j_rot"]
        omega = state["omega"]
        r_arr = state["r"]
        m_arr = state["m"]
        dq_arr = state["dq"]

        checks: list[tuple[str, float, float, float]] = [
            ("star_mass", float(state["star_mass"]), 12.5, 14.5),
            (
                "log total_angular_momentum",
                _rust.safe_log10(float(state["total_angular_momentum"])),
                50.0,
                53.0,
            ),
            (
                "log center_omega",
                _rust.safe_log10(float(state["center_omega"])),
                -6.0,
                -2.0,
            ),
            (
                "log he_core_omega",
                _rust.safe_log10(float(state["he_core_omega"])),
                -6.0,
                -3.0,
            ),
            ("he_core_mass", float(state["he_core_mass"]), 2.0, 5.0),
            ("surface j_rot", _rust.safe_log10(float(j_rot[0])), 17.0, 19.0),
            (
                "surface v_rot",
                float(omega[0]) * float(r_arr[0]) * 1.0e-5,
                0.01,
                0.6,
            ),
        ]

        k_cntr, k_surf = _locate_mass_window_1based(
            m_arr, nz, 3.0 * constants.MSUN, 3.4 * constants.MSUN
        )
        ln10 = constants.LN10
        checks.extend(
            [
                (
                    "logT",
                    _dq_weighted_slice_avg(state["lnt"], dq_arr, k_surf, k_cntr)
                    / ln10,
                    7.2,
                    8.1,
                ),
                (
                    "logRho",
                    _dq_weighted_slice_avg(state["lnd"], dq_arr, k_surf, k_cntr)
                    / ln10,
                    1.0,
                    3.0,
                ),
                (
                    "log j_rot",
                    _rust.safe_log10(
                        _dq_weighted_slice_avg(j_rot, dq_arr, k_surf, k_cntr)
                    ),
                    14.5,
                    15.5,
                ),
                (
                    "D_ES",
                    _rust.safe_log10(
                        _dq_weighted_slice_avg(
                            state["d_es"], dq_arr, k_surf, k_cntr
                        )
                    ),
                    1.0,
                    5.0,
                ),
                (
                    "D_ST",
                    _rust.safe_log10(
                        _dq_weighted_slice_avg(
                            state["d_st"], dq_arr, k_surf, k_cntr
                        )
                    ),
                    0.12,
                    12.0,
                ),
                (
                    "nu_ST",
                    _rust.safe_log10(
                        _dq_weighted_slice_avg(
                            state["nu_st"], dq_arr, k_surf, k_cntr
                        )
                    ),
                    8.0,
                    12.0,
                ),
            ]
        )

        failed = [
            label for label, val, lo, hi in checks if not (lo <= val <= hi)
        ]
        self.failed_checks = failed
        self.checks_okay = not failed


class Thermohaline1MStarExtras:
    r"""Thermohaline and rotational mixing post-checks near $0.245\,M_\odot$
    from `star/test_suite/1M_thermohaline/src/run_star_extras.f90:37-212`.
    """

    def __init__(self) -> None:
        self.checks_okay: bool | None = None
        self.failed_checks: list[str] = []

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        """Run 5 global and 5 shell-averaged tolerance checks (`:68-135`)."""
        nz = int(state["nz"])
        j_rot = state["j_rot"]
        omega = state["omega"]
        r_arr = state["r"]
        m_arr = state["m"]
        dq_arr = state["dq"]

        checks: list[tuple[str, float, float, float]] = [
            (
                "log total_angular_momentum",
                _rust.safe_log10(float(state["total_angular_momentum"])),
                45.0,
                55.0,
            ),
            (
                "log center_omega",
                _rust.safe_log10(float(state["center_omega"])),
                -4.0,
                -2.0,
            ),
            (
                "log he_core_omega",
                _rust.safe_log10(float(state["he_core_omega"])),
                -6.0,
                -3.0,
            ),
            ("surface j_rot", _rust.safe_log10(float(j_rot[0])), 5.0, 25.0),
            (
                "surface v_rot",
                float(omega[0]) * float(r_arr[0]) * 1.0e-5,
                0.0,
                1.0,
            ),
        ]

        k_cntr, k_surf = _locate_mass_window_1based(
            m_arr, nz, 0.24 * constants.MSUN, 0.25 * constants.MSUN
        )
        if k_surf >= 1 and k_cntr <= nz:
            ln10 = constants.LN10
            avg_fn = (
                (lambda arr: _helpers.fdiv(0.0, 0.0))
                if k_cntr == 0
                else (
                    lambda arr: _dq_weighted_slice_avg(
                        arr, dq_arr, k_surf, k_cntr
                    )
                )
            )
            checks.extend(
                [
                    ("logT", avg_fn(state["lnt"]) / ln10, 7.0, 7.5),
                    ("logRho", avg_fn(state["lnd"]) / ln10, 0.5, 2.0),
                    ("log j_rot", _rust.safe_log10(avg_fn(j_rot)), 10.0, 20.0),
                    (
                        "D_ST",
                        _rust.safe_log10(avg_fn(state["d_st"])),
                        1.0,
                        8.0,
                    ),
                    (
                        "nu_ST",
                        _rust.safe_log10(avg_fn(state["nu_st"])),
                        4.0,
                        9.0,
                    ),
                ]
            )

        failed = [
            label for label, val, lo, hi in checks if not (lo <= val <= hi)
        ]
        self.failed_checks = failed
        self.checks_okay = not failed


class HighRotation20MZ2m2StarExtras:
    """`zbar_div_abar` profile column handler from
    `star/test_suite/20M_z2m2_high_rotation/src/run_star_extras.f90:38-156`.
    """

    def extra_profile_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, list[float]]]:
        """Return `zbar_div_abar` `(nz,)` profile column (`:116-144`)."""
        return _helpers.zbar_div_abar_profile_column(state)


class DiffusionSmoothnessStarExtras:
    """Inner-150-zone `brunt_N2_composition_term` smoothness check from
    `star/test_suite/diffusion_smoothness/src/run_star_extras.f90:39-175`.
    """

    def __init__(self) -> None:
        self.avg_d2n2: float | None = None
        self.smooth_enough: bool | None = None

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        """Reproduce lines `80-99` verbatim (including `N200 = N2m1;
        N2p1 = N200` at `:87-88`).
        """
        if int(state["x_integer_ctrl"][0]) != 1:
            return
        nz = int(state["nz"])
        n = 150
        if nz < n:
            raise IndexError(f"diffusion_smoothness: nz={nz} < {n} (:84-91)")
        b_comp = state["brunt_n2_composition_term"]
        sum_d2n2 = 0.0
        n2m1 = float(b_comp[nz - 1])
        n200 = float(b_comp[nz - 2])
        for k_1 in range(nz - 2, nz - n - 1, -1):
            n200 = n2m1
            n2p1 = n200
            n2m1 = float(b_comp[k_1 - 1])
            sum_d2n2 = sum_d2n2 + _helpers.fdiv(
                abs(0.5 * (n2m1 + n2p1) - n200), n200
            )
        avg = sum_d2n2 / float(n)
        self.avg_d2n2 = avg
        self.smooth_enough = avg < 5.0e-2


def compute_op_mono_factors(state: dict[str, Any]) -> list[float]:
    """Evaluate `s% op_mono_factors(1:species)`
    (`star/test_suite/radiative_levitation/src/run_star_extras.f90:39-59`).
    """
    species = int(state["species"])
    j_fe56 = int(state["net_iso_fe56"])
    if j_fe56 == 0:
        raise RuntimeError("failed to find fe56 in current net")
    factors = [1.0] * species
    factors[j_fe56 - 1] = float(state["x_ctrl"][0])
    return factors


class RadiativeLevitationStarExtras:
    r"""OP monochromatic $^{56}\mathrm{Fe}$ factor and $^{56}\mathrm{Fe}$ /
    $^{58}\mathrm{Ni}$ accumulation checks from
    `star/test_suite/radiative_levitation/src/run_star_extras.f90:39-243`.
    """

    def __init__(self) -> None:
        self.levitation_ok: bool | None = None
        self.outcome: str | None = None

    def extras_controls(self, state: dict[str, Any]) -> None:
        """Check `x_ctrl(1) > 0` and require `set_op_mono_factors` (`#161`)
        (`:62-101`).
        """
        if float(state["x_ctrl"][0]) <= 0.0:
            return
        op_path = str(state["op_mono_data_path"]).strip()
        has_a06 = bool(op_path) and (pathlib.Path(op_path) / "a06.140").exists()
        detail = (
            "a06.140 present"
            if has_a06
            else f"op_mono_data_path={op_path!r} missing a06.140"
        )
        raise NotImplementedError(
            "radiative_levitation: extras_controls registers "
            f"s% set_op_mono_factors (:89; {detail}), which flexmesa-star "
            "StarExtras does not expose (#161)"
        )

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        r"""Verify convective zone and $^{56}\mathrm{Fe}$/$^{58}\mathrm{Ni}$
        mass fractions (`:116-165`).
        """
        if float(state["x_ctrl"][0]) <= 0.0:
            return
        if float(state["star_age"]) < float(state["max_age"]):
            self.levitation_ok = False
            self.outcome = "failed to reach max age"
            return
        nz = int(state["nz"])
        _ = int(state["species"])
        dq_arr = state["dq"]
        sum_dq = 0.0
        k_1 = nz + 1
        for idx_1 in range(1, nz + 1):
            if sum_dq > 1.0e-10:
                k_1 = idx_1
                break
            sum_dq = sum_dq + float(dq_arr[idx_1 - 1])
        if k_1 > nz:
            raise IndexError("sum_dq never exceeded 1d-10 across 1..nz (:136)")
        k0 = k_1 - 1
        if int(state["mixing_type"][k0]) != int(
            constants.MixingType.CONVECTIVE_MIXING
        ):
            self.levitation_ok = False
            self.outcome = "bad mixing type"
            return
        fe56 = int(state["net_iso_fe56"])
        ni58 = int(state["net_iso_ni58"])
        if fe56 <= 0 or ni58 <= 0:
            self.levitation_ok = False
            self.outcome = "missing fe56 or ni58"
            return
        xa_fe56 = state["xa_fe56"]
        xa_ni58 = state["xa_ni58"]
        if float(xa_fe56[k0]) < 2.0e-2 or float(xa_ni58[k0]) < 7.0e-3:
            self.levitation_ok = False
            self.outcome = "too little fe56 or ni58"
            return
        for i in range(nz):
            if float(xa_fe56[i]) + float(xa_ni58[i]) > 0.8:
                self.levitation_ok = False
                self.outcome = "a region dominated by fe56/ni58 has formed"
                return
        self.levitation_ok = True
        self.outcome = "found expected effects of radiative levitation"
