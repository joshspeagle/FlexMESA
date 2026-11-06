# Wave 1 Numerics: Constants, Dual Algebra, Splines & Solvers

This guide covers the mathematical formulations and numerical implementations of the Wave 1 foundation and solver modules (`const`, `utils`, `math`, `auto_diff`, `interp_1d`, `interp_2d`, `mtx`, and `num`).

---

## 1. Physical Constants and Mathematical Primitives (`const`, `math`, `utils`)

- **CODATA / IAU Physical Constants (`flexmesa.constants` & `crates/flexmesa-const`)**: Exports bit-exact IEEE-754 double-precision (`f64`) CGS constants matching `const/public/const_def.f90` (`STANDARD_CGRAV`, `PLANCK_H`, `CLIGHT`, `KERG`, `BOLTZ_SIGMA`, `CRAD`, `AMU`, `ME`, `MP`, `QE`, `MSUN`, `RSUN`, `LSUN`, `SECYER`), precomputed roots/logarithms (`LN10`, `ONE_THIRD`, `SQRT_2_DIV_3`), and the `MixingType` enumeration.
- **Utilities, String Tokenization & Parallel Execution (`flexmesa.utils` & `crates/flexmesa-utils`)**: Provides non-finite array sanitizers (`is_bad`, `is_bad_array`, `replace_bad_nums`), MESA string tokenizers (`Token`, `Tokenizer`, `tokenize`, `split_line`, `remove_underbars`), and integer lookup tables (`IntegerDict`, `IntegerIdict`), paired with a persistent 4-thread `rayon` worker pool (`flexmesa_bench::stage3_pool`) used by `flexmesa._rust` batch kernels.
- **IEEE-754 Math Wrappers (`flexmesa.math` & `crates/flexmesa-math`)**: Implements 33 elementary and domain-safe intrinsics (`pow`, `powi`, `powm1`, `pow2`–`pow8`, `exp10`, `expm1`, `log1p`, `log2`, `safe_log`, `safe_log10`, `safe_sqrt`, `sinpi`, `cospi`, `tanpi`, `asinpi`, `acospi`, `atanpi`, `sgn`, `heaviside`, `str_to_double`, `str_to_vector`) polymorphic over scalar `f64`, forward-mode `DualOrder1`, and JAX `jnp.float64` arrays.

---

## 2. Forward-Mode Dual and Hyperdual Numbers (`auto_diff`)

Upstream MESA's `auto_diff` module uses preprocessor templates to generate fixed-size first- and second-order forward-mode automatic differentiation types. `FlexMESA` provides:

1. **Const-Generic First-Order Dual Numbers (`DualOrder1<N>`)**:
   $$u = (v, \nabla v) \in \mathbb{R} \times \mathbb{R}^N, \qquad \phi(u) = \left(\phi(v),\, \phi'(v)\,\nabla v\right),$$
   registered as a stack-allocated `Copy` struct in Safe Rust (`crates/flexmesa-autodiff`) and as a static PyTree in Python/JAX (`flexmesa.auto_diff.DualOrder1`). By parameterizing `N` via Rust const generics (`DualOrder1<1>` for `interp_1d` and `num`, `DualOrder1<6>` and `DualOrder1<12>` for `turb`), `FlexMESA` propagates only the active local independent variables rather than MESA's fixed 33-element stellar tangent vector.
2. **Two-Variable Second-Order Hyperdual Numbers (`AutoDiffReal2VarOrder2` / `AutoDiffReal2ndOrder`)**:
   Propagates exact value, gradient $(\partial_x, \partial_y)$, and Hessian $(\partial_{xx}, \partial_{xy}, \partial_{yy})$ in a single forward pass for 2D thermodynamic tables $(\log_{10} T, \log_{10} \rho)$.

---

## 3. Monotonic 1D and Bicubic 2D Spline Interpolation (`interp_1d`, `interp_2d`)

### 3.1 1D Piecewise Cubic Splines (`interp_1d`)
Given strictly monotonic knots $x_0 < x_1 < \dots < x_{n-1}$ and values $y_i$, `interp_1d` stores 4 cubic coefficients per interval $i$ in C-contiguous shape `(n, 4)` such that for $\Delta x = x - x_i$,

$$P_i(x) = f_{i,0} + \Delta x \left(f_{i,1} + \Delta x \left(f_{i,2} + \Delta x\, f_{i,3}\right)\right),$$

$$P_i'(x) = f_{i,1} + \Delta x \left(2 f_{i,2} + 3 \Delta x\, f_{i,3}\right).$$

Three slope-reconstruction stencils are provided across Safe Rust (`flexmesa._rust`) and `@jax.jit`:

- **Steffen (1990) Monotonic Cubic (`interp_pm`)**: Enforces strict piecewise monotonicity by clamping node slopes $s_i$ to $\operatorname{sign}(d_{i-1/2}) \min(|d_{i-1/2}|, |d_{i+1/2}|, 0.5 |\tilde{s}_i|)$ whenever adjacent secants $d_{i-1/2}, d_{i+1/2}$ agree in sign, and setting $s_i = 0$ at local extrema. Essential for interpolating sharp nuclear composition fronts without unphysical negative mass-fraction overshoots.
- **Quartic / Lagrange 4-Point Slope Stencils (`interp_m3a`, `interp_m3b`, `interp_m3q`)**: Reconstructs fourth-order interior slopes from a 4-point stencil with optional monotonicity limiting.
- **Conservative Cell-Average Reconstruction (`integrate_values`)**: Reconstructs point values from piecewise-integrated shell averages while preserving total mass/energy integrals.

### 3.2 2D Bicubic Hermite Splines (`interp_2d`)
For rectangular thermodynamic tables $(x_i, y_j)$ of size $(N_x, N_y)$ (such as $\log_{10}\rho$ vs. $\log_{10}T$ in `eos` and `kap`), `interp_mkbicub_db` solves tridiagonal spline systems along rows and columns to precompute $f$, $f_{xx}$, $f_{yy}$, and $f_{xxyy}$ at each grid node (shape `(nx, ny, 4, 4)`). `interp_evbicub_db` evaluates the 6-tuple $(f, f_x, f_y, f_{xx}, f_{yy}, f_{xy})$ in $\mathcal{O}(\log N_x + \log N_y)$ time.

![1D monotonic vs. quartic step interpolation and 2D bicubic spline convergence](../assets/figures/interp_stencil_accuracy.png)

!!! important "Upstream Fortran Precision Note (`bicub_db.f`)"
    In upstream Fortran MESA's `interp_2d/private/bicub_db.f`, the subroutine `fvbicub_db` omits `implicit none`, leaving its local Hermite basis variables (`xp`, `yp`, `cx`, `cy`, `hx2`, `hy2`, `z36th`) implicitly typed as 32-bit `REAL` (`f32`) and limiting accuracy to $\sim 7 \times 10^{-9}$. Both Safe Rust and Python/JAX in `FlexMESA` evaluate the entire Hermite polynomial basis in full 64-bit IEEE-754 `f64`, achieving $4.0 \times 10^{-16}$ cross-language agreement.

---

## 4. Coupled Block-Tridiagonal Solvers and Implicit ODE Integrators (`mtx`, `num`)

### 4.1 Block-Tridiagonal Henyey Systems (`mtx`)
A 1D stellar evolution Newton step with $N_{\mathrm{zones}}$ radial shells and $N_{\mathrm{var}}$ structure + composition equations per shell produces a coupled block-tridiagonal linear system:

$$L_k x_{k-1} + D_k x_k + U_k x_{k+1} = b_k, \qquad k \in \{0, \dots, N_{\mathrm{zones}} - 1\}, \quad L_0 = U_{N_{\mathrm{zones}}-1} = 0.$$

`FlexMESA` provides three exact `float64` solvers for this system:

1. **Sequential Block-Thomas (`method="thomas"` in Safe Rust)**: Forward Gaussian elimination with partial pivoting on each $N_{\mathrm{var}} \times N_{\mathrm{var}}$ diagonal pivot block followed by backward substitution ($1.59\,\mathrm{ms}$ at $N_{\mathrm{zones}} = 4096, N_{\mathrm{var}} = 5$).
2. **Hirshman et al. (2010) Odd-Even Block Cyclic Reduction (`method="bcyclic"` in Safe Rust)**: Parallelizes a single block-tridiagonal solve across `rayon` worker threads by normalizing odd multiples $j = (2m + 1)2^s$ in parallel and applying Schur-complement updates to even multiples $k = 2m \cdot 2^s$ across $\lceil \log_2 N_{\mathrm{zones}} \rceil$ levels ($1.02\,\mathrm{ms}$ on 4 threads at $N_{\mathrm{zones}} = 4096$).
3. **Vectorized Parallel Cyclic Reduction (`solver_mode="pcr"` in JAX) & Safe Rust FFI (`solver_mode="rust_ffi"`)**:
   - `solve_block_tridiag_pcr` updates all $N_{\mathrm{zones}}$ rows simultaneously in $\lceil \log_2 N_{\mathrm{zones}} \rceil$ fused batched `jnp.linalg.solve` steps (`[L_k, U_k]` and `[D_k, b_k]`), accelerating single-star GPU solves by $218\times$ over sequential `lax.scan` ($7.43\,\mathrm{ms}$ vs. $1619\,\mathrm{ms}$ at $N_{\mathrm{zones}} = 4096$) and batched `jax.vmap` solves to $5.36\,\mathrm{ms}$ per star.
   - `solve_block_tridiag_rust_ffi` dispatches the forward pass to `flexmesa._rust.solve_block_tridiag_bcyclic` ($2.76\,\mathrm{ms}$) and differentiates reverse-mode `jax.grad` analytically via `@jax.custom_vjp` by solving the transposed block-tridiagonal adjoint system $A^\top \lambda = \bar{x}$.

### 4.2 Root-Finding, Simplex Optimization, and Stiff Rosenbrock / Newton Solvers (`num`)
`flexmesa.num` and `crates/flexmesa-num` port MESA's scalar root-finders (`safe_root_with_brackets`, `safe_root_with_guess`, `safe_root_without_brackets`, `safe_root_jax`, `brent_safe_zero`, `find0`, `find0_quadratic`, `find_max_quadratic`), Nelder-Mead (`nm_simplex`) and Brent (`brent_local_min`, `brent_global_min`) optimizers, adaptive Simpson's 3/8 quadrature (`integrate`) and Ridders differentiation (`dfridr`), explicit Runge-Kutta (`rk2`, `rk4`, `dopri5`, `dop853`), and stiff L-stable Rosenbrock (`rosenbrock_dense`, `rosenbrock_banded` across `RosenbrockVariant` `Ros2`, `Rose2`, `Ros3p`, `Ros3pl`, `Rodas3`, `Rodas4`, `Rodasp`) and damped Newton (`newton_solve_dense`, `newton_solve_banded`, `newton_solve_block_tridiagonal`, `newton_solve_jax`) solvers.
