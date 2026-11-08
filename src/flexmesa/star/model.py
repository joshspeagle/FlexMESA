# SPDX-License-Identifier: LGPL-3.0-or-later
"""Python and JAX hybrid stellar structure model (`StarModel`).

Wraps `flexmesa._rust.RustStarModel` (Tier A/B Safe Rust engine) and exposes
both zero-copy NumPy arrays and pure `@jax.jit` staggered-grid, face-gradient,
Henyey residual, diffusive mixing, and stellar wind kernels.
"""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from flexmesa import _rust
from flexmesa.star import _arrays, jax_evolve, jax_kernels, jax_winds_mix

type _ArrF64 = npt.NDArray[np.float64]

JaxStarState = _arrays.JaxStarState
HenyeySystemResult = _arrays.HenyeySystemResult
HenyeySolverReport = _arrays.HenyeySolverReport
EvolveStepReport = _arrays.EvolveStepReport
HistoryRow = _arrays.HistoryRow


class StarModel(_arrays.StarModelArrays):
    """1D Lagrangian stellar structure and evolution model (`Star_Info`).

    Provides a unified Python/JAX interface backed by `RustStarModel`
    (`flexmesa-star`) for ZAMS interpolation, $n=1.5$ pre-MS polytrope
    construction, ASCII `.mod` and binary `.photo` I/O, parallelized zone/face
    microphysics evaluation, convective boundary overshooting, operator-split
    and coupled structure-burn-mix solves, and stellar winds.
    """

    def __init__(self, rust_model: _rust.RustStarModel) -> None:
        """Initialize a `StarModel` from a `RustStarModel` instance."""
        self._rust_model = rust_model

    @classmethod
    def from_zams(
        cls,
        initial_mass: float = 1.0,
        initial_z: float = 0.02,
        initial_y: float = -1.0,
        eval_hydro: bool = True,
    ) -> StarModel:
        """Load a Zero-Age Main Sequence (ZAMS) model (`get_zams_model`).

        Args:
            initial_mass: Initial stellar mass in solar masses (`M_sun`).
            initial_z: Initial metal mass fraction `Z`.
            initial_y: Initial helium mass fraction `Y` (`< 0` uses `0.24+2Z`).
            eval_hydro: Whether to immediately evaluate zone/face microphysics.

        Returns:
            Initialized `StarModel`.
        """
        rm = _rust.RustStarModel.from_zams(
            float(initial_mass), float(initial_z), float(initial_y)
        )
        inst = cls(rm)
        if eval_hydro:
            inst.eval_hydro_vars()
        return inst

    @classmethod
    def from_pre_ms(
        cls,
        initial_mass: float = 1.0,
        initial_z: float = 0.02,
        initial_y: float = -1.0,
        t_c: float = 9.0e5,
        d_log10_p: float = 0.02,
        eval_hydro: bool = True,
    ) -> StarModel:
        """Construct an $n = 1.5$ pre-main-sequence polytrope model.

        Args:
            initial_mass: Initial stellar mass in solar masses (`M_sun`).
            initial_z: Initial metal mass fraction `Z`.
            initial_y: Initial helium mass fraction `Y`.
            t_c: Central temperature in Kelvin (`K`).
            d_log10_p: Outward integration step size in `log10(P_gas)`.
            eval_hydro: Whether to immediately evaluate zone/face microphysics.

        Returns:
            Initialized `StarModel`.
        """
        rm = _rust.RustStarModel.from_pre_ms(
            float(initial_mass),
            float(initial_z),
            float(initial_y),
            float(t_c),
            float(d_log10_p),
        )
        inst = cls(rm)
        if eval_hydro:
            inst.eval_hydro_vars(skip_net=True, skip_net_jacobian=True)
        return inst

    @classmethod
    def from_mod_str(cls, text: str, eval_hydro: bool = False) -> StarModel:
        """Parse an upstream MESA ASCII `.mod` string into a `StarModel`."""
        inst = cls(_rust.RustStarModel.from_mod_str(text))
        if eval_hydro:
            inst.eval_hydro_vars()
        return inst

    @classmethod
    def from_mod_file(
        cls, path: str | pathlib.Path, eval_hydro: bool = False
    ) -> StarModel:
        """Read an ASCII `.mod` file from disk into a `StarModel`."""
        text = pathlib.Path(path).read_text(encoding="utf-8")
        return cls.from_mod_str(text, eval_hydro=eval_hydro)

    @classmethod
    def from_photo_bytes(
        cls, data: bytes, eval_hydro: bool = False
    ) -> StarModel:
        """Restore a `StarModel` from binary `.photo` restart bytes."""
        inst = cls(_rust.RustStarModel.from_photo_bytes(data))
        if eval_hydro:
            inst.eval_hydro_vars()
        return inst

    def to_mod_str(self) -> str:
        """Serialize the model to an ASCII `.mod` string."""
        return self._rust_model.to_mod_str()

    def write_mod_file(self, path: str | pathlib.Path) -> None:
        """Write the model to an ASCII `.mod` file on disk."""
        pathlib.Path(path).write_text(self.to_mod_str(), encoding="utf-8")

    def to_photo_bytes(self) -> bytes:
        """Serialize the model to binary `.photo` checkpoint bytes."""
        return self._rust_model.to_photo_bytes()

    def eval_hydro_vars(
        self, skip_net: bool = False, skip_net_jacobian: bool = False
    ) -> None:
        """Evaluate zone and face microphysics (`set_hydro_vars`)."""
        self._rust_model.eval_hydro_vars(
            bool(skip_net), bool(skip_net_jacobian)
        )

    def set_primary_state(
        self,
        lnd: _ArrF64,
        ln_t: _ArrF64,
        ln_r: _ArrF64,
        lum: _ArrF64,
        dq: _ArrF64,
        xa: _ArrF64,
    ) -> None:
        """Replace primary state arrays and synchronize staggered geometry."""
        self._rust_model.set_primary_state(
            np.ascontiguousarray(lnd, dtype=np.float64),
            np.ascontiguousarray(ln_t, dtype=np.float64),
            np.ascontiguousarray(ln_r, dtype=np.float64),
            np.ascontiguousarray(lum, dtype=np.float64),
            np.ascontiguousarray(dq, dtype=np.float64),
            np.ascontiguousarray(xa, dtype=np.float64),
        )

    def save_start_values(self) -> None:
        """Snapshot current state arrays as start-of-step `*_start` values."""
        self._rust_model.save_start_values()

    def save_old_values(self) -> None:
        """Snapshot current state arrays as previous-step `*_old` values."""
        self._rust_model.save_old_values()

    def set_evolve_controls(
        self,
        mesh_delta_coeff: float = 1.0,
        varcontrol_target: float = 1.0e-4,
    ) -> None:
        """Set `mesh_delta_coeff` and `varcontrol_target` on `controls`."""
        self._rust_model.set_evolve_controls(
            float(mesh_delta_coeff), float(varcontrol_target)
        )

    def eval_equations(self, include_chem: bool = False) -> HenyeySystemResult:
        """Evaluate coupled 1D Henyey residuals and block-tridiagonal Jacobian.

        Args:
            include_chem: Include `species` composition equations in the
                coupled Jacobian (`nvar = nvar_hydro + species`).

        Returns:
            `HenyeySystemResult` containing `(equ, lblk, dblk, ublk, x_scale)`.
        """
        return HenyeySystemResult(
            *self._rust_model.eval_equations(include_chem=bool(include_chem))
        )

    def solve_henyey(
        self, include_chem: bool = False, use_bcyclic: bool = True
    ) -> HenyeySolverReport:
        """Execute a damped Newton-Raphson Henyey step for the current `dt`.

        Args:
            include_chem: Solve coupled hydro + composition system.
            use_bcyclic: Use Hirshman et al. (2010) odd-even cyclic reduction
                (`True`) or sequential block-Thomas (`False`).

        Returns:
            `HenyeySolverReport` summarizing convergence and norms.
        """
        return HenyeySolverReport(
            *self._rust_model.solve_henyey(
                include_chem=bool(include_chem),
                use_bcyclic=bool(use_bcyclic),
            )
        )

    def update_mixing(
        self,
        dt: float,
        overshoot_f: float = 0.0,
        overshoot_f0: float = 0.004,
        scheme: str = "exponential",
    ) -> int:
        """Clean mixing glitches, locate boundaries, and apply overshooting.

        Args:
            dt: Timestep in seconds (`s`).
            overshoot_f: Overshoot decay scale `f` in units of `H_P`.
            overshoot_f0: Anchor step distance `f_0` into CZ in units of `H_P`.
            scheme: `"exponential"` or `"step"`.

        Returns:
            Number of detected convective boundaries (`num_conv_boundaries`).
        """
        return int(
            self._rust_model.update_mixing(
                float(dt), float(overshoot_f), float(overshoot_f0), str(scheme)
            )
        )

    def eval_dxdt_mix(self) -> _ArrF64:
        """Evaluate mass-conserving species diffusion rate `(nz, species)`."""
        return self._rust_model.eval_dxdt_mix()

    def solve_diffusion(self, dt: float) -> None:
        """Solve implicit backward-Euler species diffusion over `dt` (`s`)."""
        self._rust_model.solve_diffusion(float(dt))

    def run_omega_mix(
        self,
        dt: float,
        omega: _ArrF64,
        nu_omega: _ArrF64,
        n_substeps: int = 4,
    ) -> None:
        """Solve substepped rotational angular momentum diffusion."""
        self._rust_model.run_omega_mix(
            float(dt),
            np.ascontiguousarray(omega, dtype=np.float64),
            np.ascontiguousarray(nu_omega, dtype=np.float64),
            int(n_substeps),
        )

    def step_struct_burn_mix(
        self,
        dt: float,
        op_split_burn: bool = False,
        use_bcyclic: bool = True,
        overshoot_f: float = 0.0,
        overshoot_f0: float = 0.004,
    ) -> HenyeySolverReport:
        """Execute a `do_struct_burn_mix` step (coupled or operator-split)."""
        return HenyeySolverReport(
            *self._rust_model.step_struct_burn_mix(
                float(dt),
                bool(op_split_burn),
                bool(use_bcyclic),
                float(overshoot_f),
                float(overshoot_f0),
            )
        )

    def apply_wind_and_adjust_mass(
        self,
        scheme: str = "Dutch",
        eta: float = 1.0,
        dt: float = 0.0,
    ) -> float:
        """Evaluate wind mass-loss rate and optionally adjust 3-region mass."""
        return float(
            self._rust_model.apply_wind_and_adjust_mass(
                str(scheme), float(eta), float(dt)
            )
        )

    def adjust_mesh(self) -> tuple[int, int]:
        """Run adaptive remeshing (`do_adjust_mesh`) -> `(splits, merges)`."""
        s, m = self._rust_model.adjust_mesh()
        return int(s), int(m)

    def pick_next_timestep(
        self,
        max_timestep_factor: float = 1.2,
        min_timestep_factor: float = 0.5,
        dx_nuc_drop_limit: float = 1.0e-2,
    ) -> tuple[float, float, float]:
        """Compute `(dt_next, varcontrol, dt_limit_ratio)` via H211 filter."""
        dt_n, vc, ratio = self._rust_model.pick_next_timestep(
            float(max_timestep_factor),
            float(min_timestep_factor),
            float(dx_nuc_drop_limit),
        )
        return float(dt_n), float(vc), float(ratio)

    def evolve_step(
        self,
        do_remesh: bool = False,
        op_split_burn: bool = False,
        use_bcyclic: bool = True,
    ) -> EvolveStepReport:
        """Execute a single evolutionary step (`star_evolve_step`)."""
        return EvolveStepReport(
            *self._rust_model.evolve_step(
                bool(do_remesh), bool(op_split_burn), bool(use_bcyclic)
            )
        )

    def evolve(
        self,
        max_steps: int,
        do_remesh: bool = False,
        op_split_burn: bool = False,
        use_bcyclic: bool = True,
        max_age: float | None = None,
    ) -> list[EvolveStepReport]:
        """Evolve the model for up to `max_steps` steps (`star_evolve`)."""
        return [
            EvolveStepReport(*t)
            for t in self._rust_model.evolve(
                int(max_steps),
                bool(do_remesh),
                bool(op_split_burn),
                bool(use_bcyclic),
                None if max_age is None else float(max_age),
            )
        ]

    def history_row(self) -> HistoryRow:
        """Extract the current `HistoryRow` summary (`history.data`)."""
        (m_num, n_z, n_ret, n_it), vals = self._rust_model.history_tuple()
        return HistoryRow(
            model_number=int(m_num),
            star_age=float(vals[0]),
            star_mass=float(vals[1]),
            log_dt=float(vals[2]),
            log_lh=float(vals[3]),
            log_l=float(vals[4]),
            log_teff=float(vals[5]),
            log_r=float(vals[6]),
            log_g=float(vals[7]),
            log_center_t=float(vals[8]),
            log_center_rho=float(vals[9]),
            center_h1=float(vals[10]),
            center_he4=float(vals[11]),
            he_core_mass=float(vals[12]),
            num_zones=int(n_z),
            num_retries=int(n_ret),
            total_num_solver_iterations=int(n_it),
        )

    def to_history_line(self) -> str:
        """Format the current state as a fixed-width `history.data` line."""
        return self._rust_model.to_history_line()

    def to_profile_str(self) -> str:
        """Format the current radial structure as a `profile.data` table."""
        return self._rust_model.to_profile_str()

    def eval_mesh_gval_jax(self, mesh_delta_coeff: float = 1.0) -> jax.Array:
        """Evaluate dimensionless mesh spacing variation `g_k` via JAX."""
        return jnp.asarray(
            jax_evolve.eval_mesh_gval_jax(
                jnp.asarray(self.ln_peos),
                jnp.asarray(self.ln_t),
                mesh_delta_coeff=float(mesh_delta_coeff),
            )
        )

    def to_jax_state(self) -> JaxStarState:
        """Export primary structure variables as an immutable `JaxStarState`."""
        return JaxStarState(
            lnd=jnp.asarray(self.lnd),
            ln_t=jnp.asarray(self.ln_t),
            ln_r=jnp.asarray(self.ln_r),
            lum=jnp.asarray(self.lum),
            dq=jnp.asarray(self.dq),
            xa=jnp.asarray(self.xa),
            mstar=float(self.mstar),
        )

    def eval_face_mlt_jax(self) -> dict[str, jax.Array]:
        """Evaluate face temperature gradients, MLT, and Brunt `N^2` via JAX."""
        return dict(
            jax_kernels.eval_face_mlt_and_brunt_jax(
                jnp.asarray(self.dq),
                jnp.asarray(self.ln_r),
                jnp.asarray(self.lum),
                jnp.asarray(self.m),
                jnp.asarray(self.rho),
                jnp.asarray(self.t),
                jnp.asarray(self.peos),
                jnp.asarray(self.ln_peos),
                jnp.asarray(self.grada),
                jnp.asarray(self.cp),
                jnp.asarray(self.chi_rho),
                jnp.asarray(self.chi_t),
                jnp.asarray(self.mu),
                jnp.asarray(self.opacity),
                tau_0=float(self.tau[0]),
            )
        )

    def eval_hydro_residuals_jax(self) -> jax.Array:
        """Evaluate the 4-equation hydrostatic residual array `(nz, 4)`."""
        return jnp.asarray(
            jax_kernels.eval_hydro_residuals_jax(
                jnp.asarray(self.dq),
                jnp.asarray(self.dm),
                jnp.asarray(self.dm_bar),
                jnp.asarray(self.m),
                jnp.asarray(self.lnd),
                jnp.asarray(self.ln_t),
                jnp.asarray(self.ln_r),
                jnp.asarray(self.lum),
                jnp.asarray(self.lnd_start),
                jnp.asarray(self.ln_peos),
                jnp.log(jnp.asarray(self.energy)),
                jnp.asarray(self.energy_start),
                jnp.asarray(self.gradt),
                jnp.asarray(self.eps_nuc),
                jnp.asarray(self.non_nuc_neu),
                jnp.asarray(self.eps_mdot),
                float(self.ln_peos_surf),
                float(self.ln_t_surf),
                float(self.dt),
            )
        )

    def eval_winds_jax(self, eta: float = 1.0) -> dict[str, jax.Array]:
        """Evaluate differentiable wind mass-loss rates via JAX."""
        x_s = float(self.xh_mass[0])
        y_s = float(self.yhe_mass[0])
        z_s = float(max(0.0, 1.0 - x_s - y_s))
        return dict(
            jax_winds_mix.eval_winds_jax(
                float(self.lum[0]),
                float(self.mstar),
                float(self.teff),
                x_s,
                y_s,
                z_s,
                eta=float(eta),
            )
        )

    def do_element_diffusion(
        self, dt: float, scheme: str = "stanton_murillo"
    ) -> tuple[_ArrF64, _ArrF64]:
        """Run multi-species Burgers element diffusion over `dt` (`s`)."""
        return _rust.star_do_element_diffusion(
            self._rust_model, float(dt), str(scheme)
        )

    def set_rotation_mixing_info(self) -> _ArrF64:
        """Update shellular rotation `(f_P, f_T)` and mixing coefficients."""
        return _rust.star_set_rotation_mixing_info(self._rust_model)

    def do_conv_premix(
        self, max_iters: int = 10, avoid_fuel_increase: bool = True
    ) -> int:
        """Run iterative convective premixing (`do_conv_premix`)."""
        return int(
            _rust.star_do_conv_premix(
                self._rust_model, int(max_iters), bool(avoid_fuel_increase)
            )
        )

    def add_predictive_mixing(
        self,
        ingested_mass_limit_coeff: float = 0.1,
        superad_thresh: float = 0.005,
    ) -> int:
        """Apply predictive mixing at convective boundaries."""
        return int(
            _rust.star_add_predictive_mixing(
                self._rust_model,
                float(ingested_mass_limit_coeff),
                float(superad_thresh),
            )
        )

    def do_phase_separation(
        self,
        dt: float,
        gamma_crystallization: float = 175.0,
        is_one_mixture: bool = False,
    ) -> _ArrF64:
        """Execute white-dwarf crystallization phase separation over `dt`."""
        return _rust.star_do_phase_separation(
            self._rust_model,
            float(dt),
            float(gamma_crystallization),
            bool(is_one_mixture),
        )


def yrs_for_init_timestep(m_msun: float) -> float:
    """Compute the initial ZAMS timestep in years (`yrs_for_init_timestep`)."""
    return float(_rust.star_yrs_for_init_timestep(float(m_msun)))


def interp_q(dq1: _ArrF64, v1: _ArrF64, dq2: _ArrF64) -> _ArrF64:
    """Interpolate profile `v1` from Lagrangian grid `dq1` onto `dq2`."""
    return _rust.star_interp_q(
        np.ascontiguousarray(dq1, dtype=np.float64),
        np.ascontiguousarray(v1, dtype=np.float64),
        np.ascontiguousarray(dq2, dtype=np.float64),
    )
