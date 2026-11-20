# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Upstream relaxation, angular-momentum accretion/conservation, sdB envelope,
and Riemann-shock `StarExtras` ports (`#231`).

Covers:
- `star/test_suite/relax_composition_j_entropy/src/run_star_extras.f90:33-363`
- `star/test_suite/accreted_material_j/src/run_star_extras.f90:37-246`
- `star/test_suite/make_sdb/src/run_star_extras.f90:38-180`
- `star/test_suite/conserve_angular_momentum/src/run_star_extras.f90:32-177`
- `star/test_suite/hse_riemann/src/run_star_extras.f90:37-216`
"""

from __future__ import annotations

import math
import pathlib
import struct
from typing import Any

from flexmesa import _rust, constants
from tests.case_extras import _helpers

# Exact IEEE-754 single-precision literal `1e-5` promoted to `real(dp)` at
# `star/test_suite/accreted_material_j/src/run_star_extras.f90:125`.
_REAL4_1E_MINUS_5: float = struct.unpack("f", struct.pack("f", 1.0e-5))[0]


def _parse_fortran_float(raw: str) -> float:
    """Parse a Fortran real token (`1.0d0`, `3.5D+33`) to `float64`."""
    tok = raw.strip().split()[0].replace("D", "E").replace("d", "e")
    return float(tok)


def _fmt_d26_18(val: float) -> str:
    """Format `val` using Fortran `1pd26.18`-compatible scientific text."""
    return f"{float(val):26.18E}"


class RelaxCompositionJEntropyStarExtras:
    """Relaxation-profile file generator and target `(Teff, L)` checker from
    `star/test_suite/relax_composition_j_entropy/src/run_star_extras.f90:33-363`.
    """

    def __init__(self) -> None:
        self.target_teff = 0.0
        self.target_l = 0.0
        self.loaded_initial_mass: float | None = None
        self.within_tolerance: bool | None = None

    def load_part2_targets(self, case_dir: pathlib.Path | str) -> float:
        """Read `initial_mass.dat` and `target_vals.dat` (`:69-103`).

        Args:
            case_dir: Staged directory containing the relaxation `.dat` files.

        Returns:
            Loaded `initial_mass` in $M_\\odot$.
        """
        root = pathlib.Path(case_dir)
        mass_text = (root / "initial_mass.dat").read_text(encoding="utf-8")
        init_mass = _parse_fortran_float(mass_text)
        t_lines = [
            ln.strip()
            for ln in (root / "target_vals.dat")
            .read_text(encoding="utf-8")
            .splitlines()
            if ln.strip()
        ]
        if len(t_lines) < 2:
            raise ValueError(
                "target_vals.dat requires target_Teff and target_L"
            )
        self.loaded_initial_mass = init_mass
        self.target_teff = _parse_fortran_float(t_lines[0])
        self.target_l = _parse_fortran_float(t_lines[1])
        return init_mass

    def extras_controls(self, state: dict[str, Any]) -> None:
        """Load part-2 targets and require `#154` `initial_mass` update
        (`:41-105`).
        """
        if int(state["x_integer_ctrl"][0]) != 2:
            return
        init_mass = self.load_part2_targets(str(state["case_dir"]))
        raise NotImplementedError(
            "relax_composition_j_entropy: extras_controls sets "
            f"s% initial_mass = {init_mass!r} (:77), which "
            "PyStarExtrasAdapter updates do not expose (#154, #231)"
        )

    def extras_check_model(
        self, state: dict[str, Any]
    ) -> str | tuple[str, str]:
        """Verify relaxed `(Teff, L(1))` within 0.5% in part 2 (`:124-159`)."""
        if int(state["x_integer_ctrl"][0]) == 2:
            teff = float(state["teff"])
            l_surf_erg = float(state["l"][0])
            bad_teff = (
                teff > self.target_teff * 1.005
                or teff < self.target_teff * 0.995
            )
            bad_l = (
                l_surf_erg > self.target_l * 1.005
                or l_surf_erg < self.target_l * 0.995
            )
            if bad_teff or bad_l:
                self.within_tolerance = False
                return ("terminate", "Relaxed model does not match input")
            self.within_tolerance = True
            return ("terminate", "Values for Teff and L are within tolerance")
        return "keep_going"

    def extras_finish_step(
        self, state: dict[str, Any]
    ) -> str | tuple[str, str]:
        """Write the 5 relaxation `.dat` files once `Teff < 8000 K` in part 1
        (`:232-348`).
        """
        if int(state["x_integer_ctrl"][0]) == 1:
            teff = float(state["teff"])
            if teff < 8000.0:
                self._write_relax_files(state)
                return ("terminate", "produced files for relax")
        return "keep_going"

    def _build_zone_relax_tables(
        self, state: dict[str, Any]
    ) -> tuple[list[str], list[str], list[str]]:
        """Format cell-wise entropy, angular-momentum, and composition tables
        (`:260-307`).
        """
        nz = int(state["nz"])
        species = int(state["species"])
        q = state["q"]
        dq = state["dq"]
        entropy = state["entropy"]
        j_rot = state["j_rot"]
        xa = state["xa"]
        ent_lines = [f" {nz}"]
        ang_lines = [f" {nz}"]
        comp_lines = [f" {nz} {species}"]
        for k in range(nz):
            q_k = float(q[k])
            dq_k = float(dq[k])
            xq_mid = 1.0 - (q_k - dq_k / 2.0)
            s_cgs = (float(entropy[k]) * constants.AVO) * constants.KERG
            ent_lines.append(f"{_fmt_d26_18(xq_mid)}{_fmt_d26_18(s_cgs)}")
            ang_lines.append(
                f"{_fmt_d26_18(1.0 - q_k)}{_fmt_d26_18(float(j_rot[k]))}"
            )
            xa_row = xa[k]
            if len(xa_row) != species:
                raise ValueError(
                    f"cell {k} xa length {len(xa_row)} != species {species}"
                )
            xa_fmt = "".join(_fmt_d26_18(float(val)) for val in xa_row)
            comp_lines.append(f"{_fmt_d26_18(xq_mid)}{xa_fmt}")
        return ent_lines, ang_lines, comp_lines

    def _write_relax_files(self, state: dict[str, Any]) -> None:
        """Write `entropy.dat`, `angular_momentum.dat`, `composition.dat`,
        `initial_mass.dat`, and `target_vals.dat` (`:260-339`).
        """
        root = pathlib.Path(str(state["case_dir"]))
        ent_lines, ang_lines, comp_lines = self._build_zone_relax_tables(state)
        (root / "entropy.dat").write_text(
            "\n".join(ent_lines) + "\n", encoding="utf-8"
        )
        (root / "angular_momentum.dat").write_text(
            "\n".join(ang_lines) + "\n", encoding="utf-8"
        )
        (root / "composition.dat").write_text(
            "\n".join(comp_lines) + "\n", encoding="utf-8"
        )
        (root / "initial_mass.dat").write_text(
            f"{_fmt_d26_18(float(state['star_mass']))}\n", encoding="utf-8"
        )
        l1 = float(state["l"][0])
        teff = float(state["teff"])
        (root / "target_vals.dat").write_text(
            f"{_fmt_d26_18(teff)}\n{_fmt_d26_18(l1)}\n", encoding="utf-8"
        )

    def extras_after_evolve(self, _state: dict[str, Any]) -> None:
        """Complete post-evolution hook (`:351-362`)."""


class AccretedMaterialJStarExtras:
    """Keplerian `accreted_material_j` hook and angular-momentum budget check
    from `star/test_suite/accreted_material_j/src/run_star_extras.f90:37-246`.
    """

    def __init__(self) -> None:
        self._lxtra1 = False
        self._lxtra2 = False
        self.wind_switch_ok: bool | None = None

    def extras_controls(self, _state: dict[str, Any]) -> None:
        """Reset `lxtra(1:2)` flags (`:37-61`)."""
        self._lxtra1 = False
        self._lxtra2 = False

    def compute_accretor_adjust_mdot(
        self, state: dict[str, Any], current_mdot: float
    ) -> tuple[float, float]:
        r"""Compute `(accreted_material_j, mstar_dot)` in
        $(\mathrm{cm^2\,s^{-1}}, \mathrm{g\,s^{-1}})$ (`:63-74`).
        """
        x_ctrl = state["x_ctrl"]
        r_phot_cm = float(state["photosphere_r"]) * constants.RSUN
        j_acc = float(x_ctrl[0]) * math.sqrt(
            (float(state["cgrav"][0]) * float(state["mstar"])) * r_phot_cm
        )
        mdot_new = float(current_mdot) + (
            (_rust.pow(10.0, float(x_ctrl[1])) * constants.MSUN)
            / constants.SECYER
        )
        return j_acc, mdot_new

    def other_adjust_mdot(
        self, state: dict[str, Any], current_mdot: float
    ) -> float:
        """Evaluate `accretor_adjust_mdot` and require `#158` `j_acc` mutation
        (`:63-74`).
        """
        j_acc, _ = self.compute_accretor_adjust_mdot(state, current_mdot)
        raise NotImplementedError(
            "accreted_material_j: accretor_adjust_mdot sets "
            f"s% accreted_material_j = {j_acc!r} (:68-69), which "
            "StarExtras::other_adjust_mdot(&StarModel) cannot mutate (#158, "
            "#231)"
        )

    def extras_check_model(
        self, state: dict[str, Any]
    ) -> str | tuple[str, str]:
        """Check step accreted `J` error and implicit-wind sign flip
        (`:101-158`).
        """
        result: str | tuple[str, str] = "keep_going"
        model_number = int(state["model_number"])
        mstar_dot = float(state["mstar_dot"])
        if model_number > 1 and mstar_dot > 0.0:
            expected_dj = (
                float(state["accreted_material_j"])
                * mstar_dot
                * float(state["dt"])
            )
            j_cur = float(state["total_angular_momentum"])
            j_old = float(state["total_angular_momentum_old"])
            am_error = _helpers.fdiv(expected_dj - (j_cur - j_old), expected_dj)
            if (
                abs(am_error) > 1.0e-5
                and expected_dj > _REAL4_1E_MINUS_5 * j_cur
            ):
                result = ("terminate", "Error in accreted J is too high!")

        if model_number > 1 and int(state["generations"]) > 1:
            mstar_dot_old = float(state["mstar_dot_old"])
            if mstar_dot < 0.0 and mstar_dot_old > 0.0:
                self._lxtra1 = True
            if mstar_dot > 0.0 and mstar_dot_old < 0.0:
                self._lxtra2 = True

        if float(state["center_h1"]) < 1.0e-6:
            self.wind_switch_ok = self._lxtra1
            if self._lxtra1:
                return ("terminate", "Test has finished with no problems.")
            return ("terminate", "Apparent problem with implicit wind.")

        return result

    def extras_finish_step(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:225-245`)."""
        return "keep_going"


class MakeSdbStarExtras:
    """Hydrogen-envelope retry and wind-shutoff controller from
    `star/test_suite/make_sdb/src/run_star_extras.f90:38-180`.
    """

    def extras_startup(
        self, state: dict[str, Any], _restart: bool
    ) -> dict[str, float] | None:
        """Turn off winds when initial `h_env_mass < x_ctrl(1)` (`:57-76`)."""
        h_upper_limit = float(state["x_ctrl"][0])
        h_env_mass = float(state["star_mass"]) - float(state["he_core_mass"])
        if h_env_mass < h_upper_limit:
            return {
                "reimers_scaling_factor": 0.0,
                "blocker_scaling_factor": 0.0,
            }
        return None

    def extras_check_model(
        self, state: dict[str, Any]
    ) -> str | tuple[str, dict[str, float]]:
        """Retry below `h_lower_limit` or shut off winds inside target window
        (`:92-117`).
        """
        x_ctrl = state["x_ctrl"]
        h_upper_limit = float(x_ctrl[0])
        h_lower_limit = (1.0 - float(x_ctrl[1])) * h_upper_limit
        h_env_mass = float(state["star_mass"]) - float(state["he_core_mass"])
        if h_env_mass > h_upper_limit:
            return "keep_going"
        if (
            h_env_mass < h_lower_limit
            and float(state["reimers_scaling_factor"]) > 0.0
        ):
            return "retry"
        return (
            "keep_going",
            {
                "reimers_scaling_factor": 0.0,
                "blocker_scaling_factor": 0.0,
            },
        )

    def extras_finish_step(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:171-179`)."""
        return "keep_going"


class ConserveAngularMomentumStarExtras:
    """Total angular-momentum conservation monitor (`1e-9` relative tolerance,
    `star/test_suite/conserve_angular_momentum/src/run_star_extras.f90:32-177`).
    """

    def __init__(self) -> None:
        self._total_angular_momentum = 0.0
        self.rel_err_total_j: float | None = None

    def extras_controls(self, _state: dict[str, Any]) -> None:
        """Zero reference angular momentum (`:40-56`)."""
        self._total_angular_momentum = 0.0

    def extras_check_model(
        self, state: dict[str, Any]
    ) -> str | tuple[str, str]:
        """Record initial `total_angular_momentum` and check `1e-9` drift
        (`:91-114`).
        """
        if bool(state["create_pre_main_sequence_model"]):
            return "keep_going"
        j_cur = float(state["total_angular_momentum"])
        if self._total_angular_momentum == 0.0:
            self._total_angular_momentum = j_cur
            return "keep_going"
        if (
            abs(self._total_angular_momentum - j_cur)
            > 1.0e-9 * self._total_angular_momentum
        ):
            return (
                "terminate",
                "failed to keep constant total angular momentum",
            )
        return "keep_going"

    def extras_finish_step(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:168-176`)."""
        return "keep_going"

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        """Compute final relative angular-momentum drift (`:71-87`)."""
        if not bool(state["create_pre_main_sequence_model"]):
            _ = int(state["model_number"])
            j_cur = float(state["total_angular_momentum"])
            self.rel_err_total_j = _helpers.fdiv(
                self._total_angular_momentum - j_cur,
                self._total_angular_momentum,
            )


class HseRiemannStarExtras:
    """Hydrostatic Riemann shock-reach stop and Mach-number check from
    `star/test_suite/hse_riemann/src/run_star_extras.f90:37-216`.
    """

    def __init__(self) -> None:
        self.max_abs_v_div_cs: float | None = None
        self.within_tolerance: bool | None = None

    def extras_check_model(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:95-103`)."""
        return "keep_going"

    def compute_pmax_history_columns(
        self, state: dict[str, Any], n: int = 0
    ) -> list[tuple[str, float]]:
        """Evaluate `data_for_extra_history_columns` (`:106-163`, where
        `how_many_extra_history_columns = 0` sets `n = 0` by default).
        """
        if n == 0:
            return []
        nz = int(state["nz"])
        peos = state["peos"]
        r = state["r"]
        v = state["v"]
        rho = state["rho"]
        t_arr = state["t"]
        m = state["m"]
        time_sec = float(state["time"])
        k = max(range(nz), key=lambda idx: float(peos[idx]))
        if k == nz - 1:
            k = nz - 2
        p_k = float(peos[k])
        r_mid = 0.5 * (float(r[k]) + float(r[k + 1]))
        v_mid = 0.5 * (float(v[k]) + float(v[k + 1]))
        rho_k = float(rho[k])
        t_k = float(t_arr[k])
        return [
            ("Pmax_P", p_k),
            ("Pmax_r_1m13", r_mid * 1.0e-13),
            ("Pmax_v", v_mid),
            ("Pmax_rho", rho_k),
            ("Pmax_T", t_k),
            ("log_Pmax_P", _rust.log10(p_k)),
            ("log_Pmax_r", _rust.log10(r_mid)),
            ("log_Pmax_v", _rust.log10(v_mid)),
            ("log_Pmax_rho", _rust.log10(rho_k)),
            ("log_Pmax_T", _rust.log10(t_k)),
            ("Pmax_r_div_t", _helpers.fdiv(r_mid, time_sec)),
            ("Pmax_m_div_Msun", float(m[k]) / constants.MSUN),
        ]

    def extras_finish_step(
        self, state: dict[str, Any]
    ) -> str | tuple[str, str]:
        """Stop when a supersonic cell exceeds `x_ctrl(9) * Rsun`
        (`:197-215`).
        """
        x9 = float(state["x_ctrl"][8])
        if x9 <= 0.0:
            return "keep_going"
        nz = int(state["nz"])
        v = state["v"]
        cs = state["csound"]
        r = state["r"]
        r_target = x9 * constants.RSUN
        for k in range(nz):
            if float(v[k]) > float(cs[k]) and float(r[k]) > r_target:
                return ("terminate", "shock has reached target location")
        return "keep_going"

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        """Check `max(|v|/cs) <= 0.1` and `time >= 0.99e5 s` (`:68-91`)."""
        nz = int(state["nz"])
        vel = state["u"] if bool(state["u_flag"]) else state["v"]
        cs = state["csound"]
        max_mach = max(
            (
                abs(_helpers.fdiv(float(vel[k]), float(cs[k])))
                for k in range(nz)
            ),
            default=0.0,
        )
        self.max_abs_v_div_cs = max_mach
        time_sec = float(state["time"])
        self.within_tolerance = (not (max_mach > 0.1)) and (
            not (time_sec < 0.99e5)
        )
