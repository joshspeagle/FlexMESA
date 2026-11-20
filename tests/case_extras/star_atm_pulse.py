# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Upstream `GYRE`-in-MESA, $T(\tau)$ atmosphere, FGONG pulse-atmosphere, and
`OP_mono` `StarExtras` ports (`#231`).

Covers:
- `star/test_suite/gyre_in_mesa_rsg/src/run_star_extras.f90:43-143` (and
  identical normalized sources in `gyre_in_mesa_bcep`, `gyre_in_mesa_envelope`,
  `gyre_in_mesa_ms`, `gyre_in_mesa_spb`, `gyre_in_mesa_wd`)
- `star/job/gyre_in_mesa_extras_finish_step.inc:14-178`
- `star/test_suite/T_tau_gradr/src/run_star_extras.f90:39-247`
- `star/test_suite/check_pulse_atm/src/run_star_extras.f90:39-324`
- `star/test_suite/1.4M_ms_op_mono/src/run_star_extras.f90:38-202`
"""

from __future__ import annotations

import math
import pathlib
from typing import Any

from flexmesa import _rust, constants
from tests.case_extras import _helpers


def eval_t_tau_q(name: str, tau: float) -> float:
    r"""Evaluate $q(\tau)$ for named $T(\tau)$ atmosphere relations.

    Reproduces `q(name, tau)` in
    `star/test_suite/T_tau_gradr/src/run_star_extras.f90:220-246` and
    `star/test_suite/check_pulse_atm/src/run_star_extras.f90:297-323`.

    Args:
        name: Upstream `atm_T_tau_relation` identifier.
        tau: Optical depth $\tau$ (dimensionless).

    Returns:
        Hopf function $q(\tau)$ (dimensionless).

    Raises:
        ValueError: If `name` is not one of the 4 upstream relation names.
    """
    if name == "Eddington":
        return constants.TWO_THIRDS
    if name == "solar_Hopf":
        return (
            1.0361 - 0.3134 * _rust.crmath_exp(-2.44799995 * tau)
        ) - 0.29589999 * _rust.crmath_exp(-30.0 * tau)
    if name == "Krishna_Swamy":
        return (
            1.39 - 0.815 * _rust.crmath_exp(-2.54 * tau)
        ) - 0.025 * _rust.crmath_exp(-30.0 * tau)
    if name == "Trampedach_solar":
        x = _rust.log10(tau)
        return (
            0.6887302005929656 + 0.0668697860833449 * (x - 0.1148742902769433)
        ) + 0.7657856893402466 * _rust.crmath_exp(
            (x - 0.9262126497691250) / 0.7657856893402466
        )
    raise ValueError(f"Invalid name in q: {name}")


def cycle_atm_t_tau_relation(name: str) -> str:
    """Return the next `atm_T_tau_relation` in the upstream 4-way cycle.

    Reproduces `star/test_suite/T_tau_gradr/src/run_star_extras.f90:134-146`
    and `star/test_suite/check_pulse_atm/src/run_star_extras.f90:195-207`.

    Args:
        name: Current `atm_T_tau_relation` option string.

    Returns:
        Next `atm_T_tau_relation` option string.

    Raises:
        ValueError: If `name` is not a recognized `atm_T_tau_relation`.
    """
    if name == "Eddington":
        return "solar_Hopf"
    if name == "solar_Hopf":
        return "Krishna_Swamy"
    if name == "Krishna_Swamy":
        return "Trampedach_solar"
    if name == "Trampedach_solar":
        return "Eddington"
    raise ValueError(f"Invalid atm_T_tau_relation: {name}")


def cycle_atm_t_tau_opacity(name: str) -> str:
    """Return the next `atm_T_tau_opacity` in the upstream 3-way cycle.

    Reproduces
    `star/test_suite/check_pulse_atm/src/run_star_extras.f90:213-223`.

    Args:
        name: Current `atm_T_tau_opacity` option string.

    Returns:
        Next `atm_T_tau_opacity` option string.

    Raises:
        ValueError: If `name` is not a recognized `atm_T_tau_opacity`.
    """
    if name == "fixed":
        return "iterated"
    if name == "iterated":
        return "varying"
    if name == "varying":
        return "fixed"
    raise ValueError(f"Invalid atm_T_tau_opacity: {name}")


def _t_check_profile_column(
    state: dict[str, Any],
) -> list[tuple[str, list[float]]]:
    r"""Compute `T_check` profile column `(nz,)` in $\mathrm{K}$ (`:187-205` of
    `T_tau_gradr` and `:264-282` of `check_pulse_atm`).
    """
    nz = int(state["nz"])
    teff = float(state["teff"])
    rel = str(state["atm_t_tau_relation"])
    tau = state["tau"]
    vals: list[float] = []
    for k in range(nz):
        tau_k = float(tau[k])
        vals.append(
            teff * _rust.pow(0.75 * (tau_k + eval_t_tau_q(rel, tau_k)), 0.25)
        )
    return [("T_check", vals)]


class GyreInMesaStarExtras:
    """In-evolution `GYRE` driver hooks shared across the 6 `gyre_in_mesa_*`
    cases (`star/test_suite/gyre_in_mesa_rsg/src/run_star_extras.f90:43-143`
    and `star/job/gyre_in_mesa_extras_finish_step.inc:14-178`).
    """

    def __init__(self) -> None:
        self.num_results = 0

    def extras_startup(self, state: dict[str, Any], _restart: bool) -> None:
        """Skip pre-MS creation; otherwise require bundled `GYRE` (`:65-96`)."""
        if bool(state["create_pre_main_sequence_model"]):
            return
        raise NotImplementedError(
            "gyre_in_mesa: extras_startup calls gyre_mesa_m::init('gyre.in') "
            "(:82-95), which requires the bundled GYRE pulsation port (#170)"
        )

    def extras_finish_step(self, state: dict[str, Any]) -> str:
        """Evaluate step cadence and require `GYRE` when active (`:102-121` and
        `star/job/gyre_in_mesa_extras_finish_step.inc:14-79`).
        """
        if bool(state["create_pre_main_sequence_model"]):
            return "keep_going"
        x_int = state["x_integer_ctrl"]
        step_interval = int(x_int[0])
        _ = int(x_int[2])
        _ = float(state["x_ctrl"][1])
        if bool(state["need_to_save_profiles_now"]) and bool(
            state["x_logical_ctrl"][0]
        ):
            call_gyre = True
        elif step_interval > 0:
            call_gyre = (int(state["model_number"]) % step_interval) == 0
        else:
            call_gyre = False
        self.num_results = 0
        if not call_gyre:
            return "keep_going"
        raise NotImplementedError(
            "gyre_in_mesa: gyre_in_mesa_extras_finish_step calls "
            "gyre_mesa_m::set_model and get_modes "
            "(star/job/gyre_in_mesa_extras_finish_step.inc:51-75), which "
            "requires the bundled GYRE pulsation port (#170)"
        )

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        """Skip pre-MS creation; otherwise require `gyre_mesa_m::final`
        (`:125-143`).
        """
        if bool(state["create_pre_main_sequence_model"]):
            return
        raise NotImplementedError(
            "gyre_in_mesa: extras_after_evolve calls gyre_mesa_m::final() "
            "(:141), which requires the bundled GYRE pulsation port (#170)"
        )


class TTauGradrStarExtras:
    r"""Radiative-gradient $T(\tau)$ RMS check and relation cycler from
    `star/test_suite/T_tau_gradr/src/run_star_extras.f90:39-247`.
    """

    def __init__(self) -> None:
        self.failed = False
        self.last_t_rms: float | None = None

    def extras_startup(self, _state: dict[str, Any], _restart: bool) -> None:
        """Reset `failed` flag (`:58-70`)."""
        self.failed = False

    def extras_start_step(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:88-96`)."""
        return "keep_going"

    def compute_t_rms(self, state: dict[str, Any]) -> float:
        r"""Compute outer-zone RMS temperature residual in $\mathrm{K}$
        (`:113-126`).
        """
        nz = int(state["nz"])
        dq = state["dq"]
        lnt = state["lnt"]
        tau = state["tau"]
        teff = float(state["teff"])
        rel = str(state["atm_t_tau_relation"])
        t_rms = 0.0
        n = 0
        for k in range(1, nz):
            dq_km1 = float(dq[k - 1])
            dq_k = float(dq[k])
            t_face = _rust.crmath_exp(
                (dq_km1 * float(lnt[k]) + dq_k * float(lnt[k - 1]))
                / (dq_km1 + dq_k)
            )
            tau_k = float(tau[k])
            t_check = teff * _rust.pow(
                0.75 * (tau_k + eval_t_tau_q(rel, tau_k)), 0.25
            )
            diff = t_face - t_check
            t_rms += diff * diff
            n += 1
            if tau_k > 0.1:
                break
        return math.sqrt(_helpers.fdiv(t_rms, float(n)))

    def extras_check_model(
        self, state: dict[str, Any]
    ) -> str | tuple[str, dict[str, Any]]:
        """Evaluate `T_rms` against `x_ctrl(1)` and cycle `atm_T_tau_relation`
        every 10 models (`:100-149`).
        """
        t_rms = self.compute_t_rms(state)
        self.last_t_rms = t_rms
        if t_rms > float(state["x_ctrl"][0]):
            self.failed = True
        if int(state["model_number"]) % 10 == 0:
            next_rel = cycle_atm_t_tau_relation(
                str(state["atm_t_tau_relation"])
            )
            return ("keep_going", {"atm_t_tau_relation": next_rel})
        return "keep_going"

    def extra_profile_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, list[float]]]:
        """Return `T_check` profile column (`:187-205`)."""
        return _t_check_profile_column(state)

    def extras_finish_step(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:209-217`)."""
        return "keep_going"

    def extras_after_evolve(self, _state: dict[str, Any]) -> None:
        """Complete post-evolution hook (`:73-85`)."""


class CheckPulseAtmStarExtras:
    r"""FGONG exported atmosphere $T(\tau)$ RMS check and control cycler from
    `star/test_suite/check_pulse_atm/src/run_star_extras.f90:39-324`.
    """

    def __init__(self) -> None:
        self.failed = False
        self.last_t_rms: float | None = None

    def extras_startup(self, _state: dict[str, Any], _restart: bool) -> None:
        """Reset `failed` flag (`:58-70`)."""
        self.failed = False

    def extras_start_step(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:88-96`)."""
        return "keep_going"

    def compute_fgong_t_rms(
        self,
        fgong_text: str,
        atm_build_tau_outer: float,
        atm_t_tau_relation: str,
        x_ctrl_1: float,
    ) -> float:
        r"""Parse an FGONG v1300 payload, evaluate $T_{\mathrm{rms}}$ in
        $\mathrm{K}$, and update `failed` against `x_ctrl(1)` (`:140-192`).
        """
        lines = fgong_text.splitlines()
        if len(lines) < 5:
            raise RuntimeError("format of check.fgong not as expected")
        hdr = [int(tok) for tok in lines[4].split()]
        if len(hdr) < 4:
            raise RuntimeError("format of check.fgong not as expected")
        nz, iconst, ivar, ivers = hdr[0], hdr[1], hdr[2], hdr[3]
        if iconst != 15 or ivar != 40 or ivers != 1300:
            raise RuntimeError("format of check.fgong not as expected")
        body = " ".join(lines[5:])
        tokens = _parse_fgong_tokens(body, 15 + nz * ivar)
        teff = tokens[13]
        t_rms = 0.0
        tau = float(atm_build_tau_outer)
        n = 0
        offset = 15
        for k in range(nz - 1):
            base_k = offset + k * ivar
            base_kp1 = base_k + ivar
            t_check = teff * _rust.pow(
                0.75 * (tau + eval_t_tau_q(atm_t_tau_relation, tau)), 0.25
            )
            t_face = tokens[base_k + 2]
            diff = t_face - t_check
            t_rms += diff * diff
            n += 1
            tau = tau + (
                (0.25 * (tokens[base_k + 4] + tokens[base_kp1 + 4]))
                * (tokens[base_k + 7] + tokens[base_kp1 + 7])
            ) * (tokens[base_k] - tokens[base_kp1])
            if tau > 0.1:
                break
        res = math.sqrt(_helpers.fdiv(t_rms, float(n)))
        self.last_t_rms = res
        if res > float(x_ctrl_1):
            self.failed = True
        return res

    def extras_check_model(self, state: dict[str, Any]) -> str:
        """Keep going between 5-step checks; require `#162`/`#161` on export
        and atmosphere-control cycling steps (`:100-226`).
        """
        model_number = int(state["model_number"])
        if model_number % 5 != 0:
            return "keep_going"
        next_rel = cycle_atm_t_tau_relation(str(state["atm_t_tau_relation"]))
        if model_number % 20 == 0:
            _ = cycle_atm_t_tau_opacity(str(state["atm_t_tau_opacity"]))
        raise NotImplementedError(
            f"check_pulse_atm: extras_check_model at model {model_number} "
            "calls star_export_pulse_data('FGONG') (:122-127, #162) and "
            f"cycles s% atm_T_tau_relation to {next_rel!r} (:195-224, #161, "
            "#231), which PyStarExtrasAdapter does not expose"
        )

    def extra_profile_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, list[float]]]:
        """Return `T_check` profile column (`:264-282`)."""
        return _t_check_profile_column(state)

    def extras_finish_step(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:286-294`)."""
        return "keep_going"

    def extras_after_evolve(self, _state: dict[str, Any]) -> None:
        """Complete post-evolution hook (`:73-85`)."""


def _parse_fgong_tokens(body: str, count: int) -> list[float]:
    """Parse fixed-width 16-char or whitespace-delimited FGONG real numbers."""
    raw = body.replace("D", "E").replace("d", "e")
    parts = raw.split()
    if len(parts) >= count:
        return [float(parts[i]) for i in range(count)]
    compact = "".join(ch for ch in raw if ch not in "\r\n")
    vals: list[float] = []
    pos = 0
    while pos < len(compact) and len(vals) < count:
        chunk = compact[pos : pos + 16].strip()
        if chunk:
            vals.append(float(chunk))
        pos += 16
    if len(vals) < count:
        raise RuntimeError("format of check.fgong not as expected")
    return vals


class MsOpMonoStarExtras:
    """OP monochromatic opacity data-path check and partial-derivative profile
    columns
    (`star/test_suite/1.4M_ms_op_mono/src/run_star_extras.f90:38-202`).
    """

    def extras_controls(self, state: dict[str, Any]) -> None:
        """Verify `op_mono_data_path/a06.140` exists on disk (`:38-75`)."""
        op_path = str(state["op_mono_data_path"]).strip()
        if not op_path:
            raise RuntimeError("1.4M_ms_op_mono: op_mono_data_path is empty")
        a06_file = pathlib.Path(op_path) / "a06.140"
        if not a06_file.is_file():
            raise FileNotFoundError(
                f'control op_mono_data_path = "{op_path}": the file '
                f'"{a06_file}" does not exist'
            )

    def extras_start_step(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:106-114`)."""
        return "keep_going"

    def extras_check_model(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:118-126`)."""
        return "keep_going"

    def extra_profile_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, list[float]]]:
        """Return `dlnkap_dlnd` and `dlnkap_dlnT` `(nz,)` (`:164-189`)."""
        nz = int(state["nz"])
        op = state["opacity"]
        dlnd = state["d_opacity_dlnd"]
        dlnt = state["d_opacity_dlnt"]
        col_d = [_helpers.fdiv(float(dlnd[k]), float(op[k])) for k in range(nz)]
        col_t = [_helpers.fdiv(float(dlnt[k]), float(op[k])) for k in range(nz)]
        return [("dlnkap_dlnd", col_d), ("dlnkap_dlnT", col_t)]

    def extras_finish_step(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:193-201`)."""
        return "keep_going"

    def extras_after_evolve(self, _state: dict[str, Any]) -> None:
        """Complete post-evolution hook (`:90-103`)."""
