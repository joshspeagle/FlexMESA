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
  - `DualOrder1` (`AutoDiffReal1VarOrder1`, `AutoDiffReal2VarOrder1`,
    `AutoDiffRealStarOrder1`): First-order forward-mode dual numbers for
    $N \in \{1, 2, 33\}$ independent variables.
  - `AutoDiffReal4VarOrder1` (`AutoDiffReal4Coeff`): 4-variable first-order jet
    $(v, \partial_{\ln T}, \partial_{\ln \rho}, \partial_a, \partial_b)$ used by
    `turb` and `neu`.
  - `AutoDiffReal2VarOrder2` (`AutoDiffReal2ndOrder`): Full bivariate
    second-order hyperdual number supporting all 704 derivative checks in
    `auto_diff/test/test_output` to $\le 10^{-14}$ relative error and verified
    against `jax.test_util.check_grads`.

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
  $[x_a, x_b]$. Fixes upstream `do_integrate_values` (`interp_1d_misc.f90`),
  which checks `x1 > x0` instead of `x1 /= x0` when subtracting the lower
  sub-interval antiderivative `a1`, so strictly decreasing grids preserve exact
  definite-integral symmetry.

#### 2D Tensor-Product Spline Interpolation (`interp_2d`)
Constructs $4 \times 4$ bicubic patch coefficients on rectangular grids
$(x_i, y_j)$ of shape `(N_x, N_y)` returning $(f, \partial_x f, \partial_y f,
\partial_{xx} f, \partial_{yy} f, \partial_{xy} f)$, alongside 2D
piecewise-monotonic (`interp_mkbipm_db` / `interp_evbipm_db`) and Akima (1996,
ACM Algorithm 760) bivariate surface interpolation (`interp_rgbi3p_db` /
`interp_rgsf3p_db`). Evaluates all cubic Hermite basis terms in `fvbicub_db` in
full IEEE-754 `float64` precision (fixing the missing `implicit none` 32-bit
`REAL` truncation of local basis variables in upstream `bicub_db.f`).

---

### 2.4 Wave 1D: Linear Algebra (`mtx`) and Numerical Solvers (`num`)

#### Index and Memory-Order Contract (`mtx` and `num`)
| Quantity | Upstream Fortran (`mtx` / `num`) | Safe Rust (`crates/flexmesa-{mtx,num}`) | Python/JAX (`src/flexmesa/{mtx,num}`) |
| :--- | :--- | :--- | :--- |
| Array indexing | 1-based (`1..=n`), except `_0_based` sparse routines | 0-based (`0..n`); sparse routines support both 1-based and 0-based CSR/CSC conventions; `binary_search` and `qsort_index` provide both 0-based and upstream-compatible 1-based variants | 0-based (`0..n`), with optional `one_based: bool` for CSR/CSC sparse indices and `binary_search` |
| Dense matrix layout | Column-major `A(lda, n)` | Row-major `n * n` contiguous slice `&[f64]` (`A[i * n + j]`) | Row-major 2D array of shape `(n, n)` (`float64`) |
| Banded matrix layout | Column-major `AB(ldab, n)` with `ldab >= 2*kl + ku + 1`; entry $A_{i,j}$ at row `kl + ku + i - j` (0-based) and col `j` | Row-major `(2*kl + ku + 1, n)` contiguous buffer (`ab[row * n + j]`, `row = kl + ku + i - j`) | 2D array of shape `(2*kl + ku + 1, n)` (`float64`) |
| Block-tridiagonal layout | Column-major `lblk(nvar, nvar, nz)`, `dblk(nvar, nvar, nz)`, `ublk(nvar, nvar, nz)` | Shell-major row-major blocks of shape `(nz, nvar, nvar)` (`blk[(k * nvar + i) * nvar + j]`), with `lblk[0] = 0` and `ublk[nz - 1] = 0` | 3D arrays `lblk`, `dblk`, `ublk` of shape `(nz, nvar, nvar)` and right-hand side `(nz, nvar)` |

#### Linear Algebra (`mtx`)
- **Dense LU with Partial Pivoting (`dgetrf` / `dgetrs` / `solve_dense`)**:
  Factors $P A = L U$ for $A \in \mathbb{R}^{N \times N}$ in `float64` with
  row-interchange pivot vector, solves $A x = b$ and multi-RHS $A X = B$,
  evaluates factored matrix-vector and matrix-matrix products
  (`lu_factored_square_mv`, `lu_factored_square_mm`), and estimates the
  1-norm reciprocal condition number (`rcond_dense`).
- **Implicit Differentiation of Linear Solves (`solve_dense_dual`,
  `solve_dense_jax`)**:
  For parameter-dependent system $A(p)\,x(p) = b(p)$, computes the primal
  solution $x = A^{-1} b$ and propagates exact first-order tangents via the
  implicit function theorem without differentiating through pivoting loops:
  $$A\,\frac{\partial x}{\partial p_m} = \frac{\partial b}{\partial p_m} - \frac{\partial A}{\partial p_m}\,x$$
- **Banded LU (`dgbtrf` / `dgbtrs` / `solve_banded` / `band_multiply_xa`)**:
  Factors and solves banded systems with $k_l$ sub-diagonals and $k_u$
  super-diagonals stored in compact `(2*k_l + k_u + 1, N)` band format,
  alongside `dense_to_band` and `band_to_dense` conversions.
- **Block-Tridiagonal Solver (`block_dble_mv` / `block_thomas_factor` /
  `block_thomas_solve` / `solve_block_tridiagonal` /
  `solve_block_tridiagonal_jax`)**:
  Evaluates block-tridiagonal products
  $$y_k = L_k x_{k-1} + D_k x_k + U_k x_{k+1}, \qquad k \in \{0, \dots, N_z - 1\}$$
  and solves coupled block-tridiagonal systems of block shape
  `(N_zones, N_vars, N_vars)` via block LU Thomas elimination with partial
  pivoting per diagonal block, matching upstream `mtx/test/test_output`
  (`test_square1`, `test_square2`, `test_square_inv`, `test_block_tridiagonal`
  on `block_tri.data` with $N_{\mathrm{var}} = 25, N_z = 20$) to $\le 10^{-12}$
  relative residual norm.
- **Sparse Format Conversions (`sparse`)**:
  Bidirectional conversions between dense, banded, compressed-row sparse (CSR),
  and compressed-column sparse (CSC) representations in both 1-based and
  0-based index conventions (`dense_to_row_sparse`, `row_sparse_to_dense`,
  `dense_to_col_sparse`, `col_sparse_to_dense`, `band_to_column_sparse`,
  `column_sparse_to_band`, `band_to_row_sparse`, `row_sparse_to_band`, and
  `find_loc_in_sparse`).
  - **Upstream Bug Fixes in 0-Based Sparse Routines**: Upstream
    `do_row_sparse_0_based_to_dense` (`mtx_support.f90`) declares `rowptr(0:n)`
    and `colind(0:nz-1)` while indexing `rowptr(i)` for `i = 1..n`, `values(k)`
    with 0-based `k`, and 1-based `a(i, j)` with unshifted 0-based `j =
    colind(k)`; similarly, `find_loc_in_column_0_based` (`mtx_formats.dek`)
    iterates over `k = colptr(j)-1 .. colptr(j+1)` instead of
    `colptr(j) .. colptr(j+1)-1`. FlexMESA uses consistent 0-based half-open
    pointer ranges `ptr[idx]..ptr[idx + 1]` for 0-based CSR/CSC formats and
    `1`-based `(ptr[idx] - 1)..(ptr[idx + 1] - 1)` offsets for 1-based CSR/CSC
    formats.

#### Numerical Solvers (`num`)
- **Compensated Summation, Search, and Algebraic Primitives (`search`)**:
  - Knuth `two_sum` and Neumaier compensated summation (`neumaier_sum`) for
    cancellation-free conservation sums.
  - `binary_search` on ascending and descending grids with optional hint index,
    and 1-based/0-based `qsort_index` permutation sorting matching `test_qsort`.
  - `linear_interp`, `find0`, `find0_quadratic`, `find_max_quadratic`,
    `two_piece_linear_coeffs`, Ridders' Richardson-extrapolated numerical
    derivative `dfridr`, and recursive adaptive Simpson 3/8 quadrature
    `integrate`.
  - **Upstream Bug Fix in `find_max_quadratic`**: Upstream `num_lib.f90` line
    581 checks `if (y2 < max(y1, y2)) ierr = -1` (which is identically false)
    instead of `if (y2 < max(y1, y3) .or. c >= 0) ierr = -1`. FlexMESA enforces
    $y_2 \ge \max(y_1, y_3)$ and negative quadratic curvature $c < 0$ so convex
    parabolas or non-interior points cannot be misclassified as a quadratic
    maximum.
- **1D Root-Finding and Minimization (`root`, `brent`)**:
  - Brent's algorithms (`brent_safe_zero`, `brent_local_min`,
    `brent_global_min`) and hybrid Newton-Brent root-finding
    (`look_for_brackets`, `safe_root`, `safe_root_with_brackets`,
    `safe_root_with_guess`, `safe_root_without_brackets`) matching all golden
    outputs in `test_brent` and `test_support` (`0.74800609950637120`, `0.5`).
  - **Implicit Differentiation of Root-Finding (`safe_root_dual`,
    `safe_root_jax`)**: For $f(x^*(p), p) = 0$, propagates first-order
    derivatives via
    $$\frac{\partial x^*}{\partial p_m} = -\left(\frac{\partial f}{\partial x}(x^*, p)\right)^{-1} \frac{\partial f}{\partial p_m}(x^*, p)$$
- **Derivative-Free Multivariate Optimization (`simplex`)**:
  Nelder-Mead simplex optimizer (`nm_simplex`) with weighted centroids, box
  bounds enforcement, shrink steps, and adaptive random search (`ARS`) using the
  Park-Miller minimal standard generator (`r8_uniform_01`), tested across the
  Wood (`WD`), extended Rosenbrock (`ER`), Freudenstein-Roth (`FR4`), Beale
  (`BLE`), Powell singular (`PS`), and trigonometric (`TR`) test functions.
- **Explicit and Stiff Implicit ODE Integrators (`rk`, `rosenbrock`)**:
  - Explicit Runge-Kutta (`rk2`, `rk4`) and Dormand-Prince order 5(4) (`dopri5`)
    and order 8(5,3) (`dop853`) adaptive integrators with continuous dense
    output (`contd5`), matching the Van der Pol (`vdpol`, $\epsilon = 10^{-3}$)
    reference solution `[1.7632345401889102, -0.83568868191466206]`.
  - Linearly-implicit Rosenbrock and RODAS stiff integrators (`ros2`, `rose2`,
    `ros3p`, `ros3pl`, `rodas3`, `rodas4`, `rodasp`) with Gustafsson predictive
    step-size control, dense and banded Jacobians, and continuous output
    (`contro3`, `contro4`), verified on `vdpol`, `diffusion` ($N_z = 48$), and
    `medakzo` ($N = 400$).
- **Damped Newton-Raphson Solver (`newton`, `newton_solve_jax`)**:
  Line-search damped Newton-Raphson solver supporting analytical and numerical
  Jacobians across dense, banded, and block-tridiagonal matrix structures plus
  custom-JVP implicit differentiation (`newton_solve_jax`), matching the
  1001-zone implicit diffusion benchmark in `test_newton`
  (`2.9347118120566711e-2` to $\le 10^{-12}$ relative error).

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
