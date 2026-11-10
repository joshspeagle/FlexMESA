# Foundational Numerics & Convection (`constants`, `math`, `utils`, `auto_diff`, `interp_1d`, `interp_2d`, `mtx`, `num`, `turb`)

This reference page covers the nine foundational numerical, spline interpolation, linear algebra, nonlinear solver, and convective mixing subpackages in `FlexMESA`.

For architectural details on hybrid dispatch, dual-number automatic differentiation, and block-tridiagonal solvers (`bcyclic` vs. `PCR`), see [Hybrid Architecture, Autodiff & Solvers](../concepts/architecture.md) and [Convection, Diffusion, Rotation & Dynamos](../concepts/convection-mixing-rotation.md).

---

## Quick Usage Example

```python
import jax
import jax.numpy as jnp
import flexmesa.turb as turb
from flexmesa import constants

# Differentiate Mixing Length Theory (MLT) temperature gradient wrt gradr
d_gradt_d_gradr = jax.grad(
    lambda gradr: turb.set_mlt_jax(
        1.0, 1.0, 2.0e8, 2.7e4, 1.5e9, 1.0e-5, 1.0e8, 1.0e5, 1.0, gradr, 0.40, 0.40
    )[1]
)(jnp.float64(0.45))
print(
    f"MSUN={constants.MSUN:.6e} g, d(grad_T)/d(grad_r)={float(d_gradt_d_gradr):.6f}"
)
```

---

## Physical & Astronomical Constants (`flexmesa.constants`)

CODATA physical constants and IAU/MESA solar reference quantities in CGS units (`MSUN`, `RSUN`, `LSUN`, `SECYER`, `SECDAY`, `STANDARD_CGRAV`, `CLIGHT`, `KERG`, `CRAD`, `AVO`, `ME`, `MP`, `PLANCK_H`, `EV2ERG`). In Safe Rust, the corresponding crate is `flexmesa_const`.

::: flexmesa.constants

---

## IEEE-754 Elementary Intrinsics (`flexmesa.math`)

Bit-reproducible elementary and base-10/base-2 transcendental functions (`exp10`, `log10`, `pow`, `safe_log10`, `sinpi`, `cospi`) matching MESA's `math/` module.

::: flexmesa.math.intrinsics

::: flexmesa.math.trig

---

## Array, Timing & Thread-Pool Utilities (`flexmesa.utils`)

Floating-point finiteness guards (`is_bad`), array helpers, and Safe Rust persistent worker-pool utilities.

::: flexmesa.utils

---

## Forward-Mode Dual & Hyperdual Numbers (`flexmesa.auto_diff`)

First-order (`AutoDiffRealOrder1`) and second-order (`AutoDiffRealOrder2`) forward-mode automatic differentiation types mirroring `crates/flexmesa-autodiff` (`DualOrder1<N>`, `DualOrder2<N>`, `StarAdOrder1`).

::: flexmesa.auto_diff.base

::: flexmesa.auto_diff.order1

::: flexmesa.auto_diff.order2

::: flexmesa.auto_diff.kernels

---

## 1D Monotonic & Piecewise-Cubic Interpolation (`flexmesa.interp_1d`)

Steffen (1990) piecewise-monotonic (`interp_pm`) and Huynh (1993) monotonicity-preserving (`interp_mp`) 1D cubic Hermite splines and conservative Lagrangian remeshing integrators.

::: flexmesa.interp_1d.pm

::: flexmesa.interp_1d.mp

::: flexmesa.interp_1d.eval

::: flexmesa.interp_1d.remesh

---

## 2D Bicubic & Akima Surface Interpolation (`flexmesa.interp_2d`)

2D bicubic Hermite splines (`interp_mkbicub_db`, `interp_evbicub_db`) and Akima bivariate surface interpolators used across EOS, opacity, ionization, and atmosphere tables.

::: flexmesa.interp_2d.bicub

::: flexmesa.interp_2d.bicub_eval

::: flexmesa.interp_2d.akima

::: flexmesa.interp_2d.akima_pd

::: flexmesa.interp_2d.bipm

::: flexmesa.interp_2d.renka790

::: flexmesa.interp_2d.spline1d

---

## Dense, Banded & Block-Tridiagonal Linear Solvers (`flexmesa.mtx`)

Dense LU with partial pivoting, banded Gaussian elimination, sequential block-Thomas (`solve_block_tridiagonal`), Hirshman et al. (2010) parallel odd-even block cyclic reduction (`solve_block_tridiag_bcyclic`), native GPU $\lceil\log_2 N\rceil$ Parallel Cyclic Reduction (`solve_block_tridiag_pcr`), and `@jax.custom_vjp` adjoint solvers (`solve_block_tridiagonal_jax`).

::: flexmesa.mtx.dense

::: flexmesa.mtx.band

::: flexmesa.mtx.block_tridiag

::: flexmesa.mtx.pcr

::: flexmesa.mtx.sparse

---

## Nonlinear Root Finders, Quadrature & Stiff ODE Integrators (`flexmesa.num`)

Brent's method (`safe_root_with_brackets`), 1D/multi-D Newton-Raphson, adaptive Simpson quadrature, explicit Runge-Kutta (`rk4`, `dopri5`), and L-stable stiff Rosenbrock (`rodas4`, `ros3p`) integrators.

::: flexmesa.num.brent

::: flexmesa.num.root

::: flexmesa.num.newton

::: flexmesa.num.quadrature

::: flexmesa.num.rk

::: flexmesa.num.rosenbrock

::: flexmesa.num.search

::: flexmesa.num.simplex

::: flexmesa.num.newuoa

::: flexmesa.num.bobyqa

---

## Turbulent Convection & Mixing (`flexmesa.turb`)

Local Mixing Length Theory (`set_mlt`, Cox & Giuli 1968, Henyey 1965, Mihalas 1978), implicit Kuhfuss (1986) Time-Dependent Convection (`set_tdc`, differentiated via the Implicit Function Theorem), Langer (1983) semiconvection (`set_semiconvection`), and Kippenhahn (1980) / Brown (2013) thermohaline mixing (`set_thermohaline`).

::: flexmesa.turb.types

::: flexmesa.turb.mlt

::: flexmesa.turb.tdc

::: flexmesa.turb.tdc_search

::: flexmesa.turb.tdc_support

::: flexmesa.turb.semiconvection

::: flexmesa.turb.thermohaline
