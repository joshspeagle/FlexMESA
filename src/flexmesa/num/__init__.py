# SPDX-License-Identifier: LGPL-3.0-or-later
"""Root-finding, optimization, quadrature, ODE, and Newton solvers (``num``).

Provides ``float64`` NumPy and JAX implementations of MESA's ``num`` module:
- :mod:`flexmesa.num.search`: compensated summation, binary search, sorting,
  and quadratic interpolation/extrema.
- :mod:`flexmesa.num.quadrature`: Ridders differentiation and adaptive
  Simpson 3/8 quadrature.
- :mod:`flexmesa.num.brent`: Brent (1973) derivative-free root-finding and
  univariate local/global minimization.
- :mod:`flexmesa.num.root`: safeguarded Newton-Raphson + bisection root-finding
  and JAX custom-VJP implicit root differentiation (:func:`safe_root_jax`).
- :mod:`flexmesa.num.simplex`: Nelder-Mead downhill simplex optimization and
  Park-Miller ``ran1_zah`` uniform PRNG.
- :mod:`flexmesa.num.rk`: explicit Runge-Kutta integrators (``rk2``, ``rk4``,
  ``dopri5``, ``dop853``) with Shampine 4th-order dense output.
- :mod:`flexmesa.num.rosenbrock`: L-stable Rosenbrock-Wanner stiff ODE/DAE
  integrators across all 7 MESA variants.
- :mod:`flexmesa.num.newton`: damped Newton-Raphson solvers for dense, banded,
  and block-tridiagonal systems, plus :func:`newton_solve_jax`.

All algorithms preserve IEEE-754 ``float64`` precision and match upstream
MESA's ``num`` verification suite (root-finding, minimization, quadrature,
Runge-Kutta/Rosenbrock ODE integration, and 1001-zone implicit diffusion),
with JAX custom-VJP differentiation for implicit roots and Newton solves.
"""

from __future__ import annotations

from flexmesa.num import (
    bobyqa as _bobyqa,
)
from flexmesa.num import (
    brent,
    newton,
    quadrature,
    rk,
    root,
    rosenbrock,
    search,
    simplex,
)
from flexmesa.num import (
    newuoa as _newuoa,
)

BobyqaResult = _bobyqa.BobyqaResult
bobyqa = _bobyqa.bobyqa
NewuoaResult = _newuoa.NewuoaResult
newuoa = _newuoa.newuoa

brent_global_min = brent.brent_global_min
brent_local_min = brent.brent_local_min
brent_safe_zero = brent.brent_safe_zero
glomin = brent.glomin
local_min = brent.local_min

NewtonConfig = newton.NewtonConfig
NewtonResult = newton.NewtonResult
newton_solve_banded = newton.newton_solve_banded
newton_solve_block_tridiagonal = newton.newton_solve_block_tridiagonal
newton_solve_dense = newton.newton_solve_dense
newton_solve_jax = newton.newton_solve_jax

dfridr = quadrature.dfridr
integrate = quadrature.integrate

DopriResult = rk.DopriResult
dop853 = rk.dop853
dopri5 = rk.dopri5
rk2 = rk.rk2
rk4 = rk.rk4

look_for_brackets = root.look_for_brackets
safe_root_jax = root.safe_root_jax
safe_root_with_brackets = root.safe_root_with_brackets
safe_root_with_guess = root.safe_root_with_guess
safe_root_without_brackets = root.safe_root_without_brackets

RosenbrockResult = rosenbrock.RosenbrockResult
RosenbrockVariant = rosenbrock.RosenbrockVariant
rosenbrock_banded = rosenbrock.rosenbrock_banded
rosenbrock_dae_diag = rosenbrock.rosenbrock_dae_diag
rosenbrock_dense = rosenbrock.rosenbrock_dense

binary_search = search.binary_search
binary_search_1based = search.binary_search_1based
find0 = search.find0
find0_quadratic = search.find0_quadratic
find_max_quadratic = search.find_max_quadratic
linear_interp = search.linear_interp
neumaier_sum = search.neumaier_sum
qsort_index = search.qsort_index
two_piece_linear_coeffs = search.two_piece_linear_coeffs
two_sum = search.two_sum

SimplexConfig = simplex.SimplexConfig
SimplexResult = simplex.SimplexResult
i4_uniform = simplex.i4_uniform
nm_simplex = simplex.nm_simplex
perm_uniform = simplex.perm_uniform
r8_uniform_01 = simplex.r8_uniform_01

__all__ = [
    "BobyqaResult",
    "DopriResult",
    "NewtonConfig",
    "NewtonResult",
    "NewuoaResult",
    "RosenbrockResult",
    "RosenbrockVariant",
    "SimplexConfig",
    "SimplexResult",
    "binary_search",
    "binary_search_1based",
    "bobyqa",
    "brent_global_min",
    "brent_local_min",
    "brent_safe_zero",
    "dfridr",
    "dop853",
    "dopri5",
    "find0",
    "find0_quadratic",
    "find_max_quadratic",
    "glomin",
    "i4_uniform",
    "integrate",
    "linear_interp",
    "local_min",
    "look_for_brackets",
    "neumaier_sum",
    "newton_solve_banded",
    "newton_solve_block_tridiagonal",
    "newton_solve_dense",
    "newton_solve_jax",
    "newuoa",
    "nm_simplex",
    "perm_uniform",
    "qsort_index",
    "r8_uniform_01",
    "rk2",
    "rk4",
    "rosenbrock_banded",
    "rosenbrock_dae_diag",
    "rosenbrock_dense",
    "safe_root_jax",
    "safe_root_with_brackets",
    "safe_root_with_guess",
    "safe_root_without_brackets",
    "two_piece_linear_coeffs",
    "two_sum",
]
