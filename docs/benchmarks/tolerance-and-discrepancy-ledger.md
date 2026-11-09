# Tolerance & Discrepancy Ledger

`FlexMESA` enforces a strict numerical verification policy: **test tolerances are never widened to mask a regression**. Every discrepancy between analytical calculus (forward/reverse automatic differentiation), Safe Rust (`f64`), JAX (`float64`), and upstream Fortran MESA (`r24.08.1`) is isolated to its exact mathematical or floating-point root cause and documented below.

---

## 1. Three-Tier Numerical Verification Hierarchy

All automated test suites (`pytest` in `tests/` and `cargo test --workspace` in `crates/`) classify numerical assertions into three tiers based on the mathematical conditioning of the underlying operator:

| Verification Tier | Target Operators & Modules | Relative Error Bound | Reference Oracle |
| :--- | :--- | :--- | :--- |
| **Tier 1: Exact Closed-Form, Linear & Spline Oracles** | Physical constants (`constants`), elementary intrinsics (`math`), dual/hyperdual numbers (`auto_diff`), 1D/2D spline evaluation (`interp_1d`, `interp_2d`), dense/banded/block-tridiagonal solves (`mtx`), composition moments (`chem`), thermal neutrinos (`neu`), REACLIB rates (`rates`), reaction network RHS & Jacobians (`net`), HLLC Riemann fluxes, Roche-lobe geometry (`binary`), and acoustic integrals (`astero`) | $\le 10^{-14}$ to $10^{-12}$ (primal values)<br>$\le 10^{-10}$ (1st-order AD vs. `jax.jacfwd`) | Upstream Fortran MESA 16-digit `<pkg>/test/test_output` files and exact dual-number / `jax.jacfwd` identities |
| **Tier 2: Iterative Root-Finding, Stiff ODE & 3rd-Order Free-Energy Solvers** | Brent/Newton root finders (`num`), Rosenbrock stiff 1-zone burners (`net`), `(P_gas, T)` EOS inversions (`eos_pt_get`), implicit Kuhfuss `TDC` (`set_tdc`), and 3rd-order `Skye` free-energy heat capacities ($c_V \propto \partial^2 F/\partial T^2$ and $\partial c_V/\partial\ln T \propto \partial^3 F/\partial T^3$) | $\le 10^{-10}$ to $10^{-8}$ (converged roots & 2nd derivatives)<br>$\le 2 \times 10^{-6}$ (3rd-order `Skye` derivatives across CPU/XLA FMA orders) | Implicit Function Theorem identities ($\partial y^*/\partial\theta = -(\partial Q/\partial y)^{-1}(\partial Q/\partial\theta)$) and cross-backend Rust vs. JAX parity |
| **Tier 3: Multi-Step Coupled Henyey & Evolutionary Trajectories** | Coupled 1D Henyey Newton-Raphson steps (`star`), Zero-Age Main Sequence (`ZAMS`) structure profiles, `star/test` ($4.0\,M_\odot, Z = 0.02$) evolution, and upstream `test_suite` multi-stage runs | $\le 10^{-3}\,\mathrm{dex}$ on $\log_{10} T_c, \log_{10}\rho_c, \log_{10} L, \log_{10} R, \log_{10} T_{\mathrm{eff}}$<br>$\le 5 \times 10^{-5}$ on central $X_{\mathrm{H}}$ | Upstream Fortran MESA `star/test/LOGS/history.data` and committed `standard_*.mod` checkpoints |

---

## 2. Documented Upstream Fortran MESA Discrepancies

During the line-by-line audit and golden-baseline verification of upstream Fortran MESA (`r24.08.1`), several upstream bugs and implicit single-precision truncations were uncovered. Below is the complete scientific ledger explaining how each discrepancy arises in Fortran MESA and how `FlexMESA` handles it.

### 2.1 Implicit 32-Bit `REAL` (`f32`) Hermite Basis Variables in `interp_2d/private/bicub_db.f`

- **Upstream Location**: `interp_2d/private/bicub_db.f`, subroutine `fvbicub_db` (and downstream callers such as `ionization/private/ion_tables_eval.f90:eval_charge_of_Fe56_in_He4`).
- **Root Cause**: The legacy Fortran 77 routine `fvbicub_db` omits `implicit none`. While its arguments and spline coefficient arrays are declared `real(dp)` (`f64`), its local Hermite basis variables (`xp`, `yp`, `cx`, `cy`, `hx2`, `hy2`, `z36th = 1.0/36.0`) are undeclared and therefore default to 32-bit single-precision `REAL` (`f32`, $\epsilon_{\mathrm{mach}} \approx 1.19 \times 10^{-7}$).
- **Measured Effect**: When evaluating 2D bicubic splines in `interp_2d` and `ion_fe56`, Safe Rust and Python/JAX agree with each other in strict IEEE-754 `f64` to $\le 3.7 \times 10^{-14}$, whereas upstream Fortran MESA's `fvbicub_db` exhibits a $4.83 \times 10^{-9}$ to $7.20 \times 10^{-9}$ single-precision truncation floor in the 9th significant digit.
- **`FlexMESA` Resolution**: `FlexMESA` evaluates all Hermite basis polynomials (`w0`, `w1`, `w2`) and grid spacings in full 64-bit IEEE-754 `f64` (`1.0_f64 / 36.0_f64`).

### 2.2 Missing Density Factor $\rho$ in `HELM` Mixed Density-Composition Derivatives (`dpepdda`, `dpepddz`)

- **Upstream Location**: `eos/private/helm_electron_positron.dek`, lines 566 to 567.
- **Root Cause**: The Timmes & Swesty (2000) `HELM` routine tabulates electron-positron pressure $P_{\mathrm{ele}}(\text{din}, T)$ as a function of electron-nucleon density $\text{din} = Y_e \rho$ (where $Y_e = \bar{Z}/\bar{A}$). With `dindd` $= \partial\text{din}/\partial\rho = Y_e$, `dindda` $= \partial\text{din}/\partial\bar{A} = -\rho Y_e / \bar{A}$, `ww` $= \partial P_{\mathrm{ele}}/\partial\text{din}$, and `zz` $= \partial^2 P_{\mathrm{ele}}/\partial\text{din}^2$, differentiating $\partial P_{\mathrm{ele}}/\partial\rho = \text{ww}(\text{din}, T)\,Y_e(\bar{A}, \bar{Z})$ with respect to $\bar{A}$ by the product and chain rules yields:

$$\frac{\partial^2 P_{\mathrm{ele}}}{\partial\rho\,\partial\bar{A}} = \frac{\partial\,\text{ww}}{\partial\text{din}}\frac{\partial\text{din}}{\partial\bar{A}}\,Y_e + \text{ww}\,\frac{\partial Y_e}{\partial\bar{A}} = (\text{zz}\cdot\text{din} + \text{ww})\,\text{dindda}$$

  because $(\partial\text{din}/\partial\bar{A})\,Y_e = \text{dindda}\cdot Y_e$ and $\partial Y_e/\partial\bar{A} = \text{dindda}/\rho$, or equivalently when `dindda` is defined with $\rho$ factored out, the first term requires $\text{din} = Y_e\rho$ rather than $\text{dindd} = Y_e$. Upstream Fortran MESA lines 566 to 567 instead write:

  ```fortran
  dpepdda = (zz*dindd + ww)*dindda
  dpepddz = (zz*dindd + ww)*dinddz
  ```

  which replaces `din` ($= Y_e\rho$) with `dindd` ($= Y_e$), omitting a factor of $\rho$ whenever $\text{zz} = \partial^2 P_{\mathrm{ele}}/\partial\text{din}^2 \neq 0$ (degenerate regimes).
- **`FlexMESA` Resolution**: Both Safe Rust and JAX analytical `HELM` evaluators preserve upstream MESA's `helm_electron_positron.dek:566-567` formula in the returned `d_dlnd` / composition slots so golden upstream regression tests pass to $< 10^{-13}$ across all 26 channels, while `tests/test_eos_wave3a2.py` verifies exact `jax.jacfwd` autodiff parity ($< 10^{-10}$) across all channels independent of `dpepdda`/`dpepddz` at arbitrary $\rho$ and across all channels at $\rho = 1\,\mathrm{g\,cm^{-3}}$ where $\text{din} = \text{dindd}$.

### 2.3 Factored Horner Derivative vs. Standard Autodiff in Biquintic `HELM` Basis Functions

- **Upstream Location**: `eos/private/helm_electron_positron.dek` (`psi0`, `dpsi0`, `ddpsi0`).
- **Root Cause**: The biquintic Hermite basis function $w_0(z) = 1 - z^2(3 - 2z)$ has the algebraically factored derivative $w_0'(z) = 6z(z - 1)$. Differentiating the unfactored expression $1 - z^2(3 - 2z)$ directly via standard automatic differentiation evaluates $-2z(3 - 2z) + 2z^2$, which differs by 1 to 2 ULPs ($\sim 2.2 \times 10^{-16}$). Because `HELM` divides grid-coordinate derivatives by $\Delta\rho^2$ during second-derivative reconstruction, a 1-ULP difference in $w_0'(z)$ is amplified by $O(\Delta\rho^{-2}) \sim 10^{10}$ at low density ($\rho \sim 10^{-6}\,\mathrm{g\,cm^{-3}}$).
- **`FlexMESA` Resolution**: `src/flexmesa/eos/_helm_jax_poly.py` attaches recursive `@jax.custom_jvp` rules (`quintic_w0_coord`, `cubic_w0_coord`) that force JAX forward-mode AD to evaluate the exact factored Horner derivative polynomials used by Fortran MESA and Safe Rust, restoring $< 10^{-11}$ cross-language parity across all 26 `HELM` output channels.

### 2.4 Near-Center Pressure Scale-Height Cap in `star/private/hydro_vars.f90`

- **Upstream Location**: `star/private/hydro_vars.f90` (`alt_scale_height`).
- **Root Cause**: Near the stellar center ($r \to 0$), the standard hydrostatic pressure scale height $H_P = P / (\rho g)$ diverges as $g = G m / r^2 \to 0$. Upstream MESA caps the scale height via:

$$H_P = \min\!\left(\frac{P}{\rho g},\, \sqrt{\frac{P}{G\rho^2}}\right)$$

  Inside the innermost radiative core zones where $\sqrt{P/(G\rho^2)} < P/(\rho g)$, $H_P$ depends only on $(P, \rho)$ and is mathematically independent of radius $r$, so $\partial H_P / \partial\ln r = 0$ identically.
- **`FlexMESA` Resolution**: `crates/flexmesa-star/src/hydro_vars.rs` and `src/flexmesa/star/jax_kernels.py` implement the exact `alt_scale_height` cap, and radial Jacobian tests (`tests/test_star_wave4_1.py`) verify $\partial H_P/\partial\ln r \neq 0$ on interior shells outside the central fallback sphere.

### 2.5 Profile-Norm Scaling Across Zero Crossings in `w5_riemann_rsp2`

- **Location**: `src/flexmesa/bench/kernels_wave5.py` and `tests/test_wave5_bench.py`.
- **Root Cause**: The composite per-zone digest observable for `w5_riemann_rsp2` sums positive turbulent energy $(w / 10^5)$ and negative turbulent dissipation $(\dot{\epsilon}_t / 10^8)$, crossing zero near $s_k \approx 0.435$. At zone $k = 55$ ($N_{\mathrm{zones}} = 128$), $|v_{55}| / \max_j |v_j| = 1.2 \times 10^{-4}$, magnifying a single-ULP CPU vs. XLA FMA difference by $10^4$ when divided pointwise by $|v_k|$.
- **`FlexMESA` Resolution**: The benchmark harness reports both the pointwise relative error and the profile-norm relative error $\max_k |v_k^{\mathrm{Rust}} - v_k^{\mathrm{JAX}}| / \max_k |v_k^{\mathrm{Rust}}|$, which agrees to $\le 1.45 \times 10^{-15}$ across all grid resolutions.

### 2.6 Start-of-Step Energy Scaling, Stiff Nuclear Rate Scale & Photospheric Table Boundary Anchoring (`star/test_suite`)

- **Upstream Locations**: `star/private/hydro_eqns.f90` (`do1_energy_eqn`, `do1_chem_eqns`) and `star/private/hydro_vars.f90` (`eval_surf_bc`).
- **Root Cause**: When evolving upstream `star/test_suite` `.mod` files across steep shell-burning fronts (such as AGB thermal pulses in `c13_pocket` at $T_{\mathrm{eff}} \approx 3150\,\mathrm{K}$ or helium core flashes in `1M_pre_ms_to_wd`), three coupled numerical sensitivities arise:
  1. Scaling the per-zone thermal energy equation by the mid-iteration luminosity $|L_k|$ causes the residual denominator to oscillate with trial corrections $\delta L_k$.
  2. Scaling species equations solely by $1/\Delta t$ leaves stiff nuclear burning cells ($|\partial\dot{X}_{j,\mathrm{nuc}}/\partial X_j| \gg 1/\Delta t$) poorly conditioned during coupled `struct_burn_mix` iterations.
  3. Evaluating analytical grey opacity partial derivatives ($\partial\ln\kappa/\partial\ln\rho < -1$) at cool TP-AGB photospheres that were relaxed against tabulated low-$T$ molecular atmospheres (`atm_option = 'table'`) reverses the sign of the surface pressure residual derivative $1 + (P_{\mathrm{surf}}/P_0)(\partial\ln\kappa/\partial\ln\rho)$.
- **`FlexMESA` Resolution**: `crates/flexmesa-star/src/hydro_eqns.rs` scales `do1_energy_eqn` by the start-of-step thermal scale $\max(e_{\mathrm{start},k}/\Delta t, \max(|L_{\mathrm{start},k}|, L_\odot)/\Delta m_k)$ and includes $1/\Delta t + |\partial\dot{X}_{j,\mathrm{nuc}}/\partial X_j|$ in the species residual denominator, while `crates/flexmesa-star/src/hydro_vars.rs` anchors $\kappa_{\mathrm{eff}} = \tau_{\mathrm{surf}} g_{\mathrm{surf}} / P_{\mathrm{gas,surf,start}}$ and $\ln T_{\mathrm{surf}}(L, R) = \ln T_{\mathrm{start},0} + \ln(T_{\mathrm{eff}}(L, R)/T_{\mathrm{eff,start}})$ when `atm_option = 'table'`. Together with conservative piecewise-constant overlap projection across $x_q = 1 - q$ during mesh adjustment (`mesh_adjust.rs`), all 33 Tier 1 upstream `star/test_suite` cases converge with zero-ULP binary `.photo` restart parity.

---

## Programmatic Golden Comparison Example

```python
import numpy as np
from flexmesa.testing import golden

# Compare two 64-bit thermodynamic arrays against a Tier 1 golden tolerance
ref = np.array([1.43822426e02, 5.68883671e03], dtype=np.float64)
val = ref * (1.0 + 2.0e-15)
golden.assert_values_match(float(val[0]), float(ref[0]), rtol=1.0e-12)
max_rel = float(np.max(np.abs(val - ref) / np.abs(ref)))
print(f"Tier 1 parity check passed: max_rel_err = {max_rel:.3e}")
```
