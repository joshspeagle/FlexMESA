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

## Numerics Modes (`strict` vs. `fast`) & `CRMATH` Elementary Intrinsics (`flexmesa.math`)

`flexmesa.math` (backed in Rust by `crates/flexmesa-math` and `crates/flexmesa-crlibm-sys`) provides the single arithmetic layer called by all physics crates (`#165`). Two run-level numerics modes are compiled from the same Rust physics source and selected via `flexmesa.set_numerics_mode`, `flexmesa.numerics_mode_context`, or `--numerics-mode {strict,fast}` on the CLI:

- **`"strict"` (default)**: Evaluates vendored `crlibm 1.0beta4` (`CRMATH 1.2` from MESA SDK `26.6.1`, encapsulated in `flexmesa-crlibm-sys`, the sole workspace crate permitted to use `unsafe` FFI under `#![deny(unsafe_op_in_unsafe_fn)]`). Its contract is `"correctly rounded toward zero as CRMATH does it"` (`*_rz`, with `*_rd` for `exp`, `acos`, and `acospi`, plus `*_rn` on `atan2` and `intrinsic_{log,log10,exp,pow}` for upstream platform-`libm` call sites tracked in `#196` and `#198`), explicit multiplication chains for `pow2` through `pow8` and `pow_i_`/`pow_r_`/`exp10_`, and `a * b + c` with no fused multiply-add contraction. `"strict"` is the **only** mode judged against MESA `26.4.1`, across three fidelity levels:
  1. **Level 1**: `"strict"` mode is bit-identical across machines (`x86_64` and `aarch64`) and thread counts (`1` vs. `4` `rayon` threads), enforced in CI.
  2. **Level 2**: `"strict"` kernels are bit-identical to MESA `26.4.1` against exact hexadecimal reference outputs wherever operations and evaluation order match (`474,062` tested inputs across `47` math sections in `mesa_26_4_1_vectors.rs`).
  3. **Level 3**: Whole evolutionary trajectories are compared against real MESA `26.4.1` histories within the `#143` history-harness tolerances (`notes/audit/case_table.md`).
- **`"fast"`**: Evaluates platform `std` `f64` intrinsics with hardware `mul_add` permitted. `"fast"` results are hardware-dependent, are validated exclusively against `"strict"` (`notes/audit/fast_vs_strict.md`), and are never compared against MESA.

Upstream `real(qp)` (IEEE-754 `binary128` under `gfortran` on `x86_64`) is represented in Rust by `flexmesa_math::Quad`, a software `binary128` type with round-to-nearest-even `+`, `-`, `*`, `/`, integer powers, comparisons, and conversions to and from `f64` (`DECISIONS #46`). It uses integer arithmetic only, so it gives the same bits in both numerics modes and on every machine. It is checked against test vectors computed with exact rational arithmetic, which is a check of the arithmetic and not a comparison with MESA. It has no Python binding. Upstream's compensated-summation type built on it, `accurate_real` (`num/public/accurate_sum.f90`), is `flexmesa_num::AccurateReal`. Both are used for the mass-change bookkeeping (`mass_utils.f90`) and the mass-change heating term (`eps_mdot.f90`), which upstream evaluates entirely in `real(qp)`. Its first use is the cell-specific kinetic and potential energy of the `star` energy equation, remesh energy bookkeeping and total-energy diagnostics (`crates/flexmesa-star/src/cell_energy.rs`, after `cell_specific_KE_qp`, `cell_specific_PE_qp` and `get_dke_dt_dpe_dt` in `star/private/star_utils.f90:2147-2370`).

```python
import flexmesa
from flexmesa import _rust

assert flexmesa.get_numerics_mode() == "strict"
strict_val = _rust.exp10(1.25)

with flexmesa.numerics_mode_context("fast") as active_mode:
    assert active_mode == "fast"
    fast_val = _rust.exp10(1.25)

assert flexmesa.get_numerics_mode() == "strict"
print(f"exp10(1.25): strict={strict_val:.16e}, |fast - strict|={abs(fast_val - strict_val):.2e}")
```

::: flexmesa.math.intrinsics

::: flexmesa.math.trig

---

## Array, Checkpoint Codec, Timing & Thread-Pool Utilities (`flexmesa.utils`)

Floating-point finiteness guards (`is_bad`), array helpers, Safe Rust persistent worker-pool utilities, and the deterministic IEEE-754 little-endian `serde` binary codec (`flexmesa_utils::bin_codec` with shared 13-byte `NumericsMode` wire framing for `StarModel` and `BinarySystem` checkpoints).

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

Local Mixing Length Theory (`set_mlt`, Cox & Giuli 1968, Henyey 1965, Mihalas 1978), implicit Kuhfuss (1986) Time-Dependent Convection (`set_tdc`, carrying partial derivatives through the Newton iteration on `AutoDiffRealTdc` in both Safe Rust and Python/JAX, matching `turb/private/tdc.f90:253-345`), Langer (1983) semiconvection (`set_semiconvection`), and Kippenhahn (1980) / Brown (2013) thermohaline mixing (`set_thermohaline`).

::: flexmesa.turb.types

::: flexmesa.turb.mlt

::: flexmesa.turb.tdc

::: flexmesa.turb.tdc_dual

::: flexmesa.turb.tdc_search

::: flexmesa.turb.tdc_support

::: flexmesa.turb.semiconvection

::: flexmesa.turb.thermohaline

---

## Compiled Rust Extension Bridge (`flexmesa._rust`)

Low-level Safe Rust PyO3 bindings exposed via `flexmesa._rust` for constants, utilities, elementary math, automatic differentiation, splines, linear solvers, and convection (`_ArrF64` denotes a `float64` NumPy `ndarray`).

### `tier_a_data` and `tier_a_parse`

- **`const_probe_values`** (`() -> list[float]`): Returns the 23 CGS constants of `flexmesa-const` printed by upstream `const/test`, in order: `pi`, `ln10`, `boltz_sigma`, `boltz_sigma*4`, `boltz_sigma*4/clight`, `boltz_sigma/clight`, `crad`, `secyer`, `Msun`, `Rsun`, `Lsun`, `ly`, `m_earth`, `au`, `amu`, `mn`, `mp`, `me`, `planck_h`, `qe`, `avo`, `clight`, and `kerg`.
- **`is_fortran_float_token`** (`(token: str) -> bool`): Returns `True` if `token` is a Fortran floating-point literal (standard IEEE literals, `D`/`d` exponents, or exponent-less `1.0-10` notation).
- **`parse_fortran_float`** (`(token: str) -> float`): Parses a Fortran or IEEE-754 floating-point literal into `f64`, raising `ValueError` on invalid input.

### `math_solvers`, `math_hex_probe`, and `math_vec_audit`

- **`crmath_cos`** (`(x: float) -> float`): Evaluates $\cos(x)$ using `flexmesa_math::cos` (`crmath` in `strict` mode).
- **`crmath_exp`** (`(x: float) -> float`): Evaluates $\exp(x)$ using `flexmesa_math::exp` (`crmath` in `strict` mode).
- **`crmath_sin`** (`(x: float) -> float`): Evaluates $\sin(x)$ using `flexmesa_math::sin` (`crmath` in `strict` mode).
- **`exp10_array`** (`(xs: _ArrF64) -> _ArrF64`): Evaluates `exp10` elementwise over a 1D C-contiguous `f64` array.
- **`safe_log_array`** (`(xs: _ArrF64) -> _ArrF64`): Evaluates `safe_log` elementwise over a 1D C-contiguous `f64` array.
- **`safe_log10_array`** (`(xs: _ArrF64) -> _ArrF64`): Evaluates `safe_log10` elementwise over a 1D C-contiguous `f64` array.
- **`safe_sqrt_array`** (`(xs: _ArrF64) -> _ArrF64`): Evaluates `safe_sqrt` elementwise over a 1D C-contiguous `f64` array.
- **`eval_autodiff_order2_zone`** (`(x: float, y: float) -> list[float]`): Evaluates `f(x, y) = exp(0.1*x*y) + x^1.5 / (y + 2)` in `AutoDiffReal2VarOrder2`, returning `[val, df_dx, df_dy, d2f_dx2, d2f_dxdy, d2f_dy2]`.
- **`eval_autodiff_order2_batch`** (`(x: _ArrF64, y: _ArrF64, parallel: bool = False) -> _ArrF64`): Evaluates `eval_autodiff_order2_zone` across 1D arrays `x, y` of shape `(nz,)`, dispatching across the persistent 4-thread `rayon` pool when `parallel=True` and `nz >= 64`.
- **`mtx_solve_dense`** (`(n: int, a_flat: _ArrF64, b: _ArrF64) -> _ArrF64`): Solves a dense $n \times n$ linear system $A x = b$ via pure Rust LU.
- **`math_test_suite_eval`** (`(x: float) -> list[tuple[str, float]]`): Evaluates 33 `flexmesa_math` scalar and integer/real-power functions at `x`, returning `(name, value)` pairs.
- **`auto_diff_test_suite_values`** (`() -> list[float]`): Computes the `auto_diff` 1-variable, 2-variable, and `AutoDiffRealStarOrder1` probe vector (`387` `f64` outputs).
- **`strict_math_vectors_sha256`** (`(vec_path: str) -> tuple[int, str]`): Evaluates all inputs in `vec_path` under `NumericsMode::Strict` and returns `(vector_count, sha256_hex)` over canonicalized `(section, output_u64_be)` records (`#165`).
- **`compare_fast_vs_strict_vectors`** (`(vec_path: str) -> list[tuple[str, int, int, int, int, float, float]]`): Evaluates all inputs in `vec_path` under `NumericsMode::Fast` against `NumericsMode::Strict` (`#165`), returning `(section, total, finite_pairs, nonfinite_mismatch_count, max_ulp, max_abs, max_rel)` in section order.
- **`measure_dispatch_overhead_ns`** (`(n_iters: int) -> tuple[float, float, float, float]`): Measures `(strict_exp_ns, fast_exp_ns, strict_fma_ns, fast_fma_ns)` over `n_iters` black-boxed calls (`#165`).

### `interp_2d`

- **`interp_evbicub_batch`** (`(x_q: _ArrF64, y_q: _ArrF64, x: _ArrF64, y: _ArrF64, coeffs_flat: _ArrF64, parallel: bool = False) -> _ArrF64`): Evaluates a 2D bicubic spline over 1D query arrays `x_q, y_q` of shape `(nz,)`, returning a `(nz, 6)` array and dispatching across the persistent 4-thread `rayon` pool when `parallel=True` and `nz >= 64`.

### `turb_py`

- **`set_mlt_f64`** (`(option: str, mixing_length_alpha: float, henyey_y_param: float, henyey_nu_param: float, chi_t: float, chi_rho: float, cp: float, grav: float, lambda_mlt: float, rho: float, p: float, t: float, opacity: float, gradr: float, grada: float, gradl: float) -> tuple[list[float], int]`): Evaluates `set_mlt` in `f64` at a single zone across all 6 geometric MLT prescriptions (`"Cox"`, `"TDC"`, `"Henyey"`, `"ML1"`, `"ML2"`, `"Mihalas"`) with local CGS inputs, dimensionless $\alpha$, and Henyey $(y, \nu)$ parameters, returning `([gamma, grad_t, y_face, conv_vel, d], mixing_type_code)`.
- **`set_mlt_batch`** (`(s_frac: _ArrF64, parallel: bool = False) -> _ArrF64`): Evaluates Cox (1968) MLT with `DualOrder1<6>` sensitivities across `s_frac`, dispatching across the persistent 4-thread `rayon` worker pool when `parallel=True` and `nz >= 64`.
- **`turb_set_tdc_f64`** (`(params: Sequence[float], include_mlt_corr_to_tdc: bool, use_tdc_enthalpy_flux_limiter: bool, inputs: Sequence[float]) -> tuple[float, float, float, float, float, int]`): Evaluates Time-Dependent Convection (`set_TDC` in `turb/public/turb.f90`) at one point with CGS `params` (`conv_vel_start`, `mixing_length_alpha`, `tdc_alpha_d`, `tdc_alpha_r`, `tdc_alpha_pt`, `tdc_alpha_c`, `tdc_alpha_s`, `dt`, `cgrav`, `m`, `scale`, `max_conv_vel`) and `inputs` (`chi_t`, `chi_rho`, `gradr`, `r`, `p`, `t`, `rho`, `dv`, `cp`, `opacity`, `scale_height`, `gradl`, `grada`, `eq_div_w`, `grav`, `energy`), returning `(gamma, grad_t, y_face, conv_vel, d, num_iters)`.
- **`set_tdc_batch`** (`(s_frac: _ArrF64, parallel: bool = False) -> _ArrF64`): Evaluates Kuhfuss (1986) TDC with `DualOrder1<6>` sensitivities across `s_frac`, dispatching across the persistent 4-thread `rayon` worker pool when `parallel=True` and `nz >= 32`.
