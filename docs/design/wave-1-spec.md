# Wave 1 Specification: Dual-Track Foundation Modules

This document specifies the mathematical formulas, array shape and physical
unit contracts, and numerical verification tolerances for Wave 1 of FlexMESA.
Wave 1 implements the ten foundational numerical and microphysical modules of
upstream MESA in parallel across Safe Rust (`crates/`) and Python/JAX `float64`
(`src/flexmesa/`), followed by a 4-stage performance staircase benchmark.

---

## 1. Module Inventory and Dual-Track Mapping

Each upstream MESA module maps to a dedicated Safe Rust crate and a Python
subpackage:

| Upstream Module | Safe Rust Crate | Python Subpackage | Primary Responsibility |
| :--- | :--- | :--- | :--- |
| `const` | `crates/flexmesa-const` | `src/flexmesa/constants` | CODATA 2018 and IAU 2015 CGS constants and mixing enums |
| `utils` | `crates/flexmesa-utils` | `src/flexmesa/utils` | IEEE-754 finiteness guards, string/integer dicts, `.net` token lexer |
| `math` | `crates/flexmesa-math` | `src/flexmesa/math` | IEEE-754 `float64` mathematical intrinsics and safe powers/exponentials |
| `auto_diff` | `crates/flexmesa-autodiff` | `src/flexmesa/auto_diff` | Forward-mode dual and second-order hyperdual numbers |
| `interp_1d` | `crates/flexmesa-interp1d` | `src/flexmesa/interp_1d` | 1D Steffen (1990) and Huynh (1993) piecewise-monotonic cubic splines |
| `interp_2d` | `crates/flexmesa-interp2d` | `src/flexmesa/interp_2d` | 2D bicubic, Akima, and piecewise-monotonic tensor-product splines |
| `mtx` | `crates/flexmesa-mtx` | `src/flexmesa/mtx` | Dense LU, banded LU, and block-tridiagonal linear solvers |
| `num` | `crates/flexmesa-num` | `src/flexmesa/num` | Compensated summation, Brent root-finding, damped Newton, and ODE solvers |
| `neu` | `crates/flexmesa-neu` | `src/flexmesa/neu` | Itoh et al. (1996) thermal neutrino cooling rates and partials |
| `turb` | `crates/flexmesa-turb` | `src/flexmesa/turb` | Mixing-length theory (MLT), time-dependent convection (TDC), and mixing |

---

## 2. Sub-Wave Specifications

### 2.1 Wave 1A: Physical Constants (`const`) and Core Utilities (`utils`)

#### Physical Constants (`const`)
All physical constants are defined in CGS units in double precision (`f64` /
`float64`) matching upstream MESA `const_def.f90`:

- **Exact SI / CODATA 2018 Constants**:
  - Speed of light: $c = 2.99792458 \times 10^{10}\;\mathrm{cm\,s^{-1}}$
  - Planck constant: $h = 6.62607015 \times 10^{-27}\;\mathrm{erg\,s}$,
    $\hbar = h / (2\pi)$
  - Boltzmann constant: $k_{\mathrm{B}} = 1.380649 \times 10^{-16}\;\mathrm{erg\,K^{-1}}$
  - Avogadro constant: $N_{\mathrm{A}} = 6.02214076 \times 10^{23}\;\mathrm{mol^{-1}}$
  - Electron volt: $1\;\mathrm{eV} = 1.602176634 \times 10^{-12}\;\mathrm{erg}$
  - Elementary charge (ESU): $q_e = (c / 10)\times 1.602176634 \times 10^{-19}\;\mathrm{esu}$
- **Derived Radiation Constants**:
  - Stefan-Boltzmann constant:
    $$\sigma_{\mathrm{SB}} = \frac{\pi^2 k_{\mathrm{B}}^4}{60\,\hbar^3 c^2}\;\mathrm{erg\,cm^{-2}\,s^{-1}\,K^{-4}}$$
  - Radiation density constant:
    $$a_{\mathrm{rad}} = \frac{4\,\sigma_{\mathrm{SB}}}{c}\;\mathrm{erg\,cm^{-3}\,K^{-4}}$$
- **Astronomical Constants (IAU 2012 / 2015 Resolution B3)**:
  - Newtonian gravitational constant: $G = 6.67430 \times 10^{-8}\;\mathrm{cm^3\,g^{-1}\,s^{-2}}$
  - Solar mass parameter: $\mu_{\odot} = 1.3271244 \times 10^{26}\;\mathrm{cm^3\,s^{-2}}$,
    $M_{\odot} = \mu_{\odot} / G$
  - Nominal solar radius, luminosity, and effective temperature:
    $R_{\odot} = 6.957 \times 10^{10}\;\mathrm{cm}$,
    $L_{\odot} = 3.828 \times 10^{33}\;\mathrm{erg\,s^{-1}}$,
    $T_{\mathrm{eff},\odot} = 5772.0\;\mathrm{K}$
  - Astronomical unit and Julian year:
    $\mathrm{au} = 1.49597870700 \times 10^{13}\;\mathrm{cm}$,
    $\mathrm{yr} = 365.25 \times 86400.0\;\mathrm{s}$,
    $\mathrm{ly} = c \cdot \mathrm{yr}$
- **Mixing Type Enumeration (`MixingType`)**:
  Integer codes from `-1` (`Crystallized`) through `10`
  (`PhaseSeparationMixing`).
- **Verification**: Relative error $\le 10^{-15}$ against all 23 derived values
  in upstream `const/test/test_output`.

#### Core Utilities (`utils`)
- **IEEE-754 Finiteness Guards**: `is_bad(value)` returns `true` whenever
  `value` is `NaN` or $\pm\infty$ (`!value.is_finite()`). Vectorized variants
  (`is_bad_slice` / `is_bad_array` and `replace_bad_nums`) operate without
  propagating `0.0 * NaN` in values or gradients.
- **Dictionaries**: `IntegerDict` (string key to `i64`/`int` value) and
  `IntegerIdict` (ordered integer pair `(k1, k2)` to `i64`/`int` value) with
  duplicate reporting upon redefinition.
- **Reaction Network Token Lexer**: Stateful stream tokenizer (`Tokenizer` and
  `tokenize`) emitting `Token` entries (`Name`, `String`, `LeftParen`,
  `RightParen`, `Comma`, and `Eof` in Safe Rust; `TokenKind` variants in Python)
  while stripping `!` line comments and `&` continuations.

---

### 2.2 Wave 1B: Mathematical Intrinsics (`math`) and Autodiff (`auto_diff`)

#### Mathematical Intrinsics (`math`)
Provides `float64` elementary functions, base-10 wrappers (`exp10`, `log10`,
`safe_log10`), `pow`, `expm1`, `log1p`, `hypot`, and hyperbolic functions.
Matches all reference values in `math/test/test_output.INTRINSIC` to
$\le 10^{-15}$ relative error.

#### Forward-Mode Automatic Differentiation (`auto_diff`)
Implements chain-rule propagation for scalar value $v$, gradient vector
$\nabla v \in \mathbb{R}^K$, and optional symmetric Hessian matrix
$H \in \mathbb{R}^{K \times K}$:

- **Unary rule** for $f(u)$:
  $$\nabla f(u) = f'(u)\,\nabla u, \qquad H_f = f'(u)\,H_u + f''(u)\,\nabla u\,\nabla u^{\top}$$
- **Binary product rule** for $u \cdot w$:
  $$\nabla(uw) = w\,\nabla u + u\,\nabla w, \qquad H_{uw} = w\,H_u + u\,H_w + \nabla u\,\nabla w^{\top} + \nabla w\,\nabla u^{\top}$$
- **Specialized Stellar Types**:
  - `AutoDiffReal2Val1`: Value and first derivative $(v, \partial_1 v)$.
  - `AutoDiffReal4Coeff`: 4-variable first-order jet
    $(v, \partial_{\ln T}, \partial_{\ln \rho}, \partial_a, \partial_b)$ used by
    `turb` and `neu`.
  - `AutoDiffReal2ndOrder`: Full second-order hyperdual number supporting all
    704 derivative checks in `auto_diff/test/test_output` to $\le 10^{-14}$
    relative error and verified against `jax.test_util.check_grads`.

---

### 2.3 Wave 1C: 1D (`interp_1d`) and 2D (`interp_2d`) Spline Interpolation

#### 1D Piecewise-Monotonic Cubic Interpolation (`interp_1d`)
Given strictly increasing knots $x_0 < x_1 < \dots < x_{N-1}$ and values $y_i$,
represents each interval $[x_i, x_{i+1}]$ with spacing $h_i = x_{i+1} - x_i$ and
secant slope $s_i = (y_{i+1} - y_i) / h_i$ as the cubic polynomial
$$p_i(\Delta x) = c_{0,i} + \Delta x\,(c_{1,i} + \Delta x\,(c_{2,i} + \Delta x\,c_{3,i})), \qquad \Delta x = x - x_i$$
- **Steffen (1990) Monotonicity Slope Limiter (`interp_pm`)**:
  $$d_i = \left(\mathrm{sign}(s_{i-1}) + \mathrm{sign}(s_i)\right) \min\!\left(|s_{i-1}|, |s_i|, \frac{1}{2}|p_i|\right)$$
  where $p_i = (s_{i-1} h_i + s_i h_{i-1}) / (h_{i-1} + h_i)$, guaranteeing zero
  spurious extrema on every sub-interval.
- **Huynh (1993) MP5 / M3a Slope Limiter (`interp_m3a`, `interp_m3b`, `interp_m3q`)**:
  Higher-order median-limited slopes preserving accuracy at smooth extrema.
- **Integrals**: Exact analytic integration of the piecewise cubic over
  $[x_a, x_b]$.

#### 2D Tensor-Product Spline Interpolation (`interp_2d`)
Constructs $4 \times 4$ bicubic patch coefficients on rectangular grids
$(x_i, y_j)$ of shape `(N_x, N_y)` returning $(f, \partial_x f, \partial_y f,
\partial_{xx} f, \partial_{yy} f, \partial_{xy} f)$ matching
`interp_2d/test/test_output` to $\le 10^{-12}$ relative error.

---

### 2.4 Wave 1D: Linear Algebra (`mtx`) and Numerical Solvers (`num`)

#### Linear Algebra (`mtx`)
- **Dense LU with Partial Pivoting (`dgetrf` / `dgetrs`)**: Solves
  $A x = b$ for $A \in \mathbb{R}^{N \times N}$ in `float64`.
- **Banded LU (`dgbmv` / `dgbtrf` / `dgbtrs`)**: Solves banded systems with
  $k_l$ sub-diagonals and $k_u$ super-diagonals.
- **Block-Tridiagonal Solver (`bcyclic` / Thomas algorithm)**: Solves coupled
  stellar structure and nuclear network Jacobians with block shape
  `(N_zones, N_vars, N_vars)` to $\le 10^{-12}$ residual norm.

#### Numerical Solvers (`num`)
- **Compensated Summation**: Knuth two-sum and Neumaier / Kahan-Babuska
  four-sum primitives preventing catastrophic cancellation in conservation
  checks.
- **1D Root-Finding (`safe_root`)**: Hybrid Brent-Dekker and Newton-Raphson
  solver with bracket maintenance.
- **Damped Newton-Raphson Solver (`newton`)**: Line-search damped Newton step
  with analytical or finite-difference Jacobian.
- **Explicit and Stiff Implicit ODE Integrators**: Dormand-Prince (`dopri5`,
  `dop853`) and Rosenbrock / implicit Runge-Kutta (`ros2`, `rodasp`, `radau5`)
  integrators matching `num/test/test_output` within tolerance.

---

### 2.5 Wave 1E: Thermal Neutrinos (`neu`) and Convection (`turb`)

#### Thermal Neutrino Cooling (`neu`)
Evaluates Itoh et al. (1996, ApJS, 102, 411) fitting formulas for total and
channel-resolved thermal neutrino energy loss rates $\epsilon_{\nu}$
($\mathrm{erg\,g^{-1}\,s^{-1}}$) and their analytical partial derivatives with
respect to $(T, \rho, \bar{A}, \bar{Z})$ across five channels:

1. **Pair production** ($e^+ + e^- \to \nu + \bar{\nu}$)
2. **Photoneutrino** ($e^- + \gamma \to e^- + \nu + \bar{\nu}$)
3. **Plasmon decay** ($\tilde{\gamma} \to \nu + \bar{\nu}$)
4. **Bremsstrahlung** ($e^- + (Z, A) \to e^- + (Z, A) + \nu + \bar{\nu}$)
5. **Recombination** ($e^-_{\mathrm{cont}} \to e^-_{\mathrm{bound}} + \nu + \bar{\nu}$)

Matches all golden output values in `neu/test/test_output` to $\le 10^{-12}$
relative error.

#### Convective and Mixing Transport (`turb`)
Evaluates local and time-dependent stellar mixing coefficients ($D$ in
$\mathrm{cm^2\,s^{-1}}$), convective velocity ($v_c$ in $\mathrm{cm\,s^{-1}}$),
convective flux ($F_{\mathrm{conv}}$), and temperature gradient ($\nabla$) with
first-order autodiff partials via `AutoDiffReal4Coeff`:

1. **Mixing-Length Theory (`set_mlt`)**: Solves the cubic MLT efficiency
   equation across `Cox`, `Henyey`, `ML1`, `ML2`, and `Mihalas` variants.
2. **Time-Dependent Convection (`set_tdc`)**: Solves the Kuhfuss (1986) / MESA
   TDC convective velocity equation over time step $\Delta t$ using Newton
   iteration with bisection fallback.
3. **Semiconvection (`set_semiconvection`)**: Langer, El Eid, & Fricke (1985)
   diffusivity and radiative/convective flux partitioning.
4. **Thermohaline Mixing (`set_thermohaline`)**: Kippenhahn, Ruschenplatt, &
   Thomas (1980), Traxler, Garaud, & Stellmach (2011), and Brown, Garaud, &
   Stellmach (2013) salt-finger diffusivities.

Matches all golden output values in `turb/test/test_output` to $\le 10^{-12}$
relative error.

---

### 2.6 Wave 1F: 4-Stage Performance Staircase and Architecture Synthesis

Evaluates Safe Rust (`crates/`), Python/NumPy (`src/flexmesa/`), and Python/JAX
(`jax.jit` + `jax.vmap` in `float64`) across four optimization stages:

1. **Stage 0 (Idiomatic Baseline)**: Direct scalar / unvectorized reference
   port.
2. **Stage 1 (Contiguous Array Layout)**: Structure-of-Arrays (SoA) batching
   over stellar zones $N_{\mathrm{zones}} \in \{128, 1024, 4096\}$.
3. **Stage 2 (Vectorized / JIT Kernel Fusion)**: `jax.jit` fused XLA kernels and
   Safe Rust slice/SIMD loops with zero inner-loop heap allocations.
4. **Stage 3 (Batched / Multi-Core Parallelism)**: `jax.vmap` across stellar
   grids and multi-threaded Safe Rust execution.

The benchmark suite records median wall-clock latency, memory footprint,
compilation overhead, and autodiff ergonomics in
`docs/design/wave-1-benchmark-report.md` to inform the post-Wave 1 architecture
synthesis.
