# Tolerance & Discrepancy Ledger

`FlexMESA` enforces a strict numerical governance policy (`notes/SCOPE.md`, `DECISIONS #9`, `#17`, `#20`, `#22`, `#23`): **test tolerances are governed prospectively against unauthorized widening** (with pre-audit tolerance broadenings documented in the repository audit under [#142](https://github.com/joshspeagle/FlexMESA/issues/142)), and **only Safe Rust `"strict"` mode is judged against MESA release `26.4.1`** (Zenodo record `19722306`; `#141`, `#165`). Safe Rust `"fast"` mode (`std` `f64` intrinsics with `mul_add`) and the derived JAX layer (`#166`) are validated against Rust `"strict"` mode and are never treated as upstream MESA oracles (wider claim and benchmark updates on this page are tracked in [#142](https://github.com/joshspeagle/FlexMESA/issues/142) and [#178](https://github.com/joshspeagle/FlexMESA/issues/178)).

---

## 1. Three-Level Numerics Policy (`#165`) & Test-Suite Tolerance Classes

### 1.1 Three Fidelity Levels for Canonical Rust `"strict"` Mode

| Level | Requirement | Enforcement Gate |
| :--- | :--- | :--- |
| **Level 1** | `"strict"` mode is bit-identical across machines (`x86_64` and `aarch64`) and thread counts (`1` vs. `4` threads) | Enforced per PR in CI (`strict-determinism` and `strict-cross-machine-gate` jobs running `scripts/check_strict_determinism.py`) |
| **Level 2** | `"strict"` kernels are bit-identical to MESA `26.4.1` wherever operations and evaluation order match | Tested against exact 16-digit hexadecimal MESA `26.4.1` outputs (`474,062` math vectors across `47` sections in `mesa_26_4_1_vectors.rs`, extended module by module under `#141` and Phase C) |
| **Level 3** | Full evolutionary trajectories match stored real MESA `26.4.1` histories | Enforced via `benchmarks/history_harness/case_manifest.toml` tolerances (`DECISIONS #20`, `#143`), tracking `first_diverging_step` toward zero |

### 1.2 Automated Test-Suite Tolerance Classes (`pytest` & `cargo test`)

| Check Class | Target Operators & Modules | Relative Bound | Comparison Target |
| :--- | :--- | :--- | :--- |
| **Module Reference & Autodiff Checks** | Physical constants (`constants`), elementary intrinsics (`math`), dual/hyperdual numbers (`auto_diff`), 1D/2D spline evaluation (`interp_1d`, `interp_2d`), dense/banded/block-tridiagonal solves (`mtx`), composition moments (`chem`), thermal neutrinos (`neu`), REACLIB/weak/screening rates (`rates`), grey `'fixed'`/`'iterated'`, irradiated, and knot-aligned tabulated model atmospheres (`atm`), reaction network RHS & Jacobians (`net`), Roche-lobe geometry (`binary`), and acoustic integrals (`astero`) | $\le 10^{-14}$ to $10^{-10}$ (module `test_output` & `26.4.1` oracle rows; `0 ULP` on `neu` `300,000/300,000` and `atm` `'fixed'`/`'iterated'`/`t_tau_rel`)<br>$\le 10^{-10}$ (1st-order AD vs. `jax.jacfwd`) | Upstream Fortran MESA `<pkg>/test/test_output` files (`#141`), `benchmarks/fortran_mesa/*_oracles/`, and forward/reverse AD identities |
| **Iterative Solvers & Derived-JAX Validation (`#166`)** | Brent/Newton root finders (`num`), Dormand-Prince `dopri5` depth-varying $T(\tau)$ atmospheres (`atm` `'varying'`, `atm/private/atm_t_tau_varying.f90:182-279` & `num/private/mod_dopri5.f`), Rosenbrock stiff 1-zone burners (`net`), `(P_gas, T)` EOS inversions (`eos_pt_get`), implicit Kuhfuss `TDC` (`set_tdc`), and 3rd-order `Skye` free-energy heat capacities ($c_V \propto \partial^2 F/\partial T^2$ and $\partial c_V/\partial\ln T \propto \partial^3 F/\partial T^3$) | $\le 10^{-10}$ to $10^{-8}$ (converged roots, `dopri5` $\ln P_{\mathrm{surf}} \le 5 \times 10^{-12}$, & 2nd derivatives)<br>$\le 10^{-7}$ (`atm` `'varying'` finite-difference $\partial\ln P_{\mathrm{surf}}/\partial(L, \ln R, \ln M)$)<br>$\le 3 \times 10^{-6}$ (3rd-order `Skye` JAX vs. Rust `strict` across `crlibm` `RZ` vs. XLA `RN`/FMA) | Implicit Function Theorem identities ($\partial y^*/\partial\theta = -(\partial Q/\partial y)^{-1}(\partial Q/\partial\theta)$), MESA `26.4.1` `dopri5` oracles, and JAX validated against Rust `strict` |
| **End-to-End Trajectory Harness (`#143`, `DECISIONS #20`)** | Uncapped `star` (`90`), `binary` (`8`), and `astero` (`8`) `test_suite` evolutions in `"strict"` mode | Final `0.0005`–`0.002`, track `0.002`–`0.005` (`case_manifest.toml`) | Stored real Fortran MESA `26.4.1` `LOGS/history.data` and `binary_history.data` (`#141`, `#143`) |

---

## 2. Documented Upstream Fortran MESA Discrepancies

During line-by-line inspection and baseline testing against upstream Fortran MESA (release `26.4.1`), several upstream bugs and implicit single-precision truncations were identified. Below is the ledger explaining how each arises in Fortran MESA and how `FlexMESA` handles it (note that pre-audit `RSP2` departures in §2.9 are scheduled to revert to upstream `26.4.1` behaviour under [#178](https://github.com/joshspeagle/FlexMESA/issues/178)).

### 2.1 Implicit 32-Bit `REAL` (`f32`) Hermite Basis Variables in `interp_2d/private/bicub_db.f`

- **Upstream Location**: `interp_2d/private/bicub_db.f`, subroutine `fvbicub_db` (and downstream callers such as `ionization/private/ion_tables_eval.f90:eval_charge_of_Fe56_in_He4` and `atm/private/table_atm.f90:405` in `get_table_values`).
- **Root Cause**: The legacy Fortran 77 routine `fvbicub_db` omits `implicit none`. While its arguments and spline coefficient arrays are declared `real(dp)` (`f64`), its local Hermite basis variables (`xp`, `yp`, `cx`, `cy`, `hx2`, `hy2`, `z36th = 1.0/36.0`) are undeclared and therefore default to 32-bit single-precision `REAL` (`f32`, $\epsilon_{\mathrm{mach}} \approx 1.19 \times 10^{-7}$).
- **Measured Effect**: When evaluating 2D bicubic splines in `interp_2d`, `ion_fe56`, and `atm` tables, Safe Rust and Python/JAX agree with each other in strict IEEE-754 `f64` to $\le 3.7 \times 10^{-14}$ (and `atm` table queries on exact $(\log g, T_{\mathrm{eff}})$ grid knots such as `WD_tau_25` and `DB_WD_tau_25` in `mesa_26_4_1_atm_oracles.tsv` agree with MESA `26.4.1` to $\le 5 \times 10^{-14}$), whereas off-knot queries (`photosphere` and `tau_*` in `atm/test/test_output` / `module_hex`, up to `7.86 \times 10^6` ULP) exhibit a $4.83 \times 10^{-9}$ to $7.20 \times 10^{-9}$ single-precision truncation floor in the 9th significant digit.
- **`FlexMESA` Behaviour (Subject to `#161`/`#169` Upstream Review)**: `FlexMESA` evaluates all Hermite basis polynomials (`w0`, `w1`, `w2`) and grid spacings in full 64-bit IEEE-754 `f64` (`1.0_f64 / 36.0_f64`). Whether `"strict"` mode should match MESA `26.4.1` `fvbicub_db` vs. retaining this difference as a post-parity deferred issue is tracked under [#169](https://github.com/joshspeagle/FlexMESA/issues/169) and [#161](https://github.com/joshspeagle/FlexMESA/issues/161).

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
- **`FlexMESA` Resolution**: Both Safe Rust and JAX analytical `HELM` evaluators preserve upstream MESA's `helm_electron_positron.dek:566-567` formula in the returned `d_dlnd` / composition slots. Safe Rust `"strict"` mode matches compiled MESA `26.4.1` (`benchmarks/fortran_mesa/eos_oracles/mesa_26_4_1_eos_oracles.tsv`) across all `17` `HELM` test states (`1,360` values and partial derivatives) to unfloored relative error $< 10^{-12}$ (`zero_atol = 1e-14` on near-zero cancelled stencil slots; `crates/flexmesa-eos/tests/` and `tests/test_eos_wave3a2.py`), while `tests/test_eos_wave3a2.py` checks unfloored `jax.jacfwd` consistency across unaffected channels and at $\rho = 1\,\mathrm{g\,cm^{-3}}$ where $\text{din} = \text{dindd}$ (with derived JAX `HELM` 3rd-order Hermite derivatives agreeing with MESA `26.4.1` within `3e-6` to `5e-5` across `crlibm` vs. XLA evaluation).

### 2.3 Factored Horner Derivative vs. Standard Autodiff in Biquintic `HELM` Basis Functions

- **Upstream Location**: `eos/private/helm_electron_positron.dek` (`psi0`, `dpsi0`, `ddpsi0`).
- **Root Cause**: The biquintic Hermite basis function $w_0(z) = 1 - z^2(3 - 2z)$ has the algebraically factored derivative $w_0'(z) = 6z(z - 1)$. Differentiating the unfactored expression $1 - z^2(3 - 2z)$ directly via standard automatic differentiation evaluates $-2z(3 - 2z) + 2z^2$, which differs by 1 to 2 ULPs ($\sim 2.2 \times 10^{-16}$). Because `HELM` divides grid-coordinate derivatives by $\Delta\rho^2$ during second-derivative reconstruction, a 1-ULP difference in $w_0'(z)$ is amplified by $O(\Delta\rho^{-2}) \sim 10^{10}$ at low density ($\rho \sim 10^{-6}\,\mathrm{g\,cm^{-3}}$).
- **`FlexMESA` Resolution**: `src/flexmesa/eos/_helm_jax_poly.py` attaches recursive `@jax.custom_jvp` rules (`quintic_w0_coord`, `cubic_w0_coord`) that force JAX forward-mode AD to evaluate the factored Horner derivative polynomials matching Safe Rust `"strict"` mode (`< 10^{-11}` agreement between JAX and Safe Rust across the 26 `HELM` channels).

### 2.4 Near-Center Pressure Scale-Height Cap in `star/private/hydro_vars.f90`

- **Upstream Location**: `star/private/hydro_vars.f90` (`alt_scale_height`).
- **Root Cause**: Near the stellar center ($r \to 0$), the standard hydrostatic pressure scale height $H_P = P / (\rho g)$ diverges as $g = G m / r^2 \to 0$. Upstream MESA caps the scale height via:

$$H_P = \min\!\left(\frac{P}{\rho g},\, \sqrt{\frac{P}{G\rho^2}}\right)$$

  Inside the innermost radiative core zones where $\sqrt{P/(G\rho^2)} < P/(\rho g)$, $H_P$ depends only on $(P, \rho)$ and is mathematically independent of radius $r$, so $\partial H_P / \partial\ln r = 0$ identically.
- **`FlexMESA` Resolution**: `crates/flexmesa-star/src/hydro_vars.rs` and `src/flexmesa/star/jax_kernels.py` implement the `alt_scale_height` cap, and radial Jacobian tests (`tests/test_star_wave4_1.py`) check $\partial H_P/\partial\ln r \neq 0$ on interior shells outside the central fallback sphere.

### 2.5 Profile-Norm Scaling Across Zero Crossings in `w5_riemann_rsp2`

- **Location**: `src/flexmesa/bench/kernels_wave5.py` and `tests/test_wave5_bench.py`.
- **Root Cause**: The composite per-zone digest observable for `w5_riemann_rsp2` sums positive turbulent energy $(w / 10^5)$ and negative turbulent dissipation $(\dot{\epsilon}_t / 10^8)$, crossing zero near $s_k \approx 0.435$. At zone $k = 55$ ($N_{\mathrm{zones}} = 128$), $|v_{55}| / \max_j |v_j| = 1.2 \times 10^{-4}$, magnifying a single-ULP CPU vs. XLA FMA difference by $10^4$ when divided pointwise by $|v_k|$.
- **`FlexMESA` Resolution**: The benchmark harness reports both the pointwise relative error and the profile-norm relative error $\max_k |v_k^{\mathrm{Rust}} - v_k^{\mathrm{JAX}}| / \max_k |v_k^{\mathrm{Rust}}|$, which agrees to $\le 1.45 \times 10^{-15}$ across all grid resolutions.

### 2.6 Thin-Zone Energy Scaling (`set_energy_eqn_scal`), Chemical Rate Floor (`min_chem_eqn_scale`), & Photospheric Table Boundary Anchoring (`star/test_suite`)

- **Upstream Locations**: `star/private/star_utils.f90:3678-3697` (subroutine `set_energy_eqn_scal`), `star/private/hydro_chem_eqns.f90:130` (subroutine `do1_chem_eqns`), and `star/private/hydro_vars.f90:779-1030` (subroutine `get_surf_PT`).
- **Root Cause**: Three scaling sensitivities govern Henyey residual conditioning on realistic multi-thousand-zone stellar grids:
  1. In thin outer envelope zones ($\Delta q_k \sim 10^{-13}$, $\Delta m_k \sim 10^{21}\,\mathrm{g}$ on a $4\,M_\odot$ ZAMS grid), a single IEEE-754 `f64` ULP in $L_k - L_{k+1}$ ($L \cdot 2^{-53} \sim 10^{20}\,\mathrm{erg\,s^{-1}}$) divided by $\Delta m_k$ and scaled naively by $\Delta t / e_{\mathrm{start},k}$ produces a residual quantization floor of $\sim 3.5 \times 10^{-4}$, exceeding `gold_tol_max_residual3 = 1e-4` even when Newton corrections reach $10^{-12}$. Upstream `set_energy_eqn_scal` caps the dimensionless prefactor at $(e_{\mathrm{start},k}\Delta m_k / E_{\mathrm{int,start}})\,\mathrm{dedt\_eqn\_r\_scale}$ (`dedt_eqn_r_scale = 1d8`), cancelling $\Delta m_k$ in zones holding less than $10^{-8}$ of total internal energy while leaving thicker interior zones ($\Delta q_k \ge 10^{-8}$) unchanged; on `make_wave4_zams_model(8)`, surface zone $k = 0$ has $\Delta q_0 \approx 10^{-10} < 10^{-8}$ and is likewise capped by `set_energy_eqn_scal`.
  2. Upstream `do1_chem_eqns` (`hydro_chem_eqns.f90:130`) scales abundance rate residuals by $\max(\mathrm{min\_chem\_eqn\_scale}, \mathrm{x\_scale}_{j,k}/\Delta t)$ with `min_chem_eqn_scale = 1.0` $\mathrm{s^{-1}}$ so slow hydrostatic nuclear drift does not inflate dimensionless residual norms when $\Delta t \gg 1\,\mathrm{s}$, while `sizeB` still enforces scaled abundance corrections against `xa_scale = 1e-5`.
  3. Evaluating analytical grey opacity partial derivatives ($\partial\ln\kappa/\partial\ln\rho < -1$) at cool TP-AGB photospheres relaxed against tabulated low-$T$ molecular atmospheres (`atm_option = 'table'`) reverses the sign of the surface pressure residual derivative $1 + (P_{\mathrm{surf}}/P_0)(\partial\ln\kappa/\partial\ln\rho)$.
- **`FlexMESA` Resolution**: `crates/flexmesa-star/src/hydro_energy.rs` and `src/flexmesa/star/jax_kernels.py` implement upstream `set_energy_eqn_scal` (`dedt_eqn_r_scale = 1e8`), `crates/flexmesa-star/src/hydro_eqns.rs` implements `min_chem_eqn_scale = 1.0` $\mathrm{s^{-1}}$ in `do1_chem_eqns`, and `crates/flexmesa-star/src/hydro_vars.rs` anchors photospheric table boundaries consistently across `star/test_suite`.

### 2.7 Hardcoded `1d-8` Newton Stopping Tolerance in `eos/private/eospt_eval.f90`

- **Upstream Locations**: `eos/private/eospt_eval.f90` (lines 180 to 191, subroutine `Get_PT_Results_using_DT`) and `num/private/mod_root.f90` (line 102, subroutine `do_safe_root_with_guess`).
- **Root Cause**: Upstream Fortran MESA's $(P_{\mathrm{gas}}, T) \to \rho$ wrapper `Get_PT_Results_using_DT` hardcodes `other_tol = 1d-8`, `logRho_tol = 1d-8`, and `max_iter = 20` when calling `do_eosDT_get_Rho`, where `do_safe_root_with_guess` terminates Newton-Brent iteration as soon as $|y(x_i)| < \max(\epsilon_y, |\mathrm{d}y/\mathrm{d}x|\,\epsilon_x)$ for $x = \log_{10}\rho$ and $y(x) = \ln P_{\mathrm{gas}}(x) - \ln P_{\mathrm{gas,target}}$. Tightening `logRho_tol` beyond `1d-8` shifts the converged density by $\sim 5 \times 10^{-11}\,\mathrm{dex}$ and shifts the `atm/test/test_output` reference value (`4.3649748642304882e-3` on irradiated $T(\tau)$ opacity at MESA release `26.4.1`). On the full `Z = 1.0` `FreeEOS` table (`mesa-FreeEOS_100z00x_CO` at $\rho = 100\,\mathrm{g\,cm^{-3}}, T = 10^7\,\mathrm{K}$), Newton step $i = 2$ reaches $|y(x_2)| = 3.31 \times 10^{-10} < |\mathrm{d}\ln P_{\mathrm{gas}}/\mathrm{d}\log_{10}\rho|\times 10^{-8} \approx 2.30 \times 10^{-8}$ and returns $|\log_{10}\rho - 2.0| = 6.25 \times 10^{-11} < 10^{-10}$, corresponding to $|\rho - \rho_{\mathrm{true}}|/\rho_{\mathrm{true}} = \ln(10)\,|\Delta\log_{10}\rho| = 1.44 \times 10^{-10}$.
- **`FlexMESA` Resolution**: Both `crates/flexmesa-eos/src/eos_pt.rs` and `src/flexmesa/eos/_inverse_solver.py` preserve upstream Fortran MESA's `1d-8` Newton stopping rule, matching all `40` `test1_eosPT` scalars in `eos/test/test_output` to unfloored relative error $< 10^{-12}$ in Safe Rust `"strict"` mode ($< 10^{-11}$ in JAX; `tests/test_eos_wave3b1.py`) and matching compiled MESA `26.4.1` `eosPT_get`, `eosPT_get_T`, `eosDT_get_T`, and `eosDT_get_Rho` oracle rows in `mesa_26_4_1_eos_oracles.tsv` ($< 10^{-12}$ to $< 10^{-8}$; `wave3a3_skye_inverse_tests.rs`, `wave3b1_golden_tests.rs`, and `tests/test_eos_wave3a3.py`). Three remaining pre-existing EOS component and control-wiring differences against MESA `26.4.1` (`PC` `eos/private/pc_eos.f90:1895` `use_EXCOR7_table = .true.` bicubic spline vs. analytical `EXCOR7`, `Skye`/`Skye+HELM` Hermite 3rd-derivative operation order, and Python `eos.eos_dt_get_t` `logT_tol`/`other_tol` kwargs) are registered bijectively in `benchmarks/history_harness/known_gaps.toml` under [#161](https://github.com/joshspeagle/FlexMESA/issues/161) (`eos/defaults/eos.defaults:163` `include_radiation = .true.` in `CMS` and `eos/defaults/eos.defaults:225-228` `real(4)` bounds in `Get_Skye_alfa_simple` are aligned in [#149](https://github.com/joshspeagle/FlexMESA/issues/149)).

### 2.8 Zeroed Secant Step `D = F2*(T4_0 - T4_0)/(F2 - F1)` in `RSP` `STAH` Anchor-Temperature Root Finder

- **Upstream Location**: `star/private/rsp_build.f90` (lines 675 to 676 and line 772, inside `stah` and `get_T_estimate`).
- **Root Cause**: During static envelope integration (`create_RSP_model`), upstream Fortran MESA's `stah` subroutine solves for the anchor shell mass $\Delta m_N$ where the zone-$N$ temperature equals $T_{\mathrm{anchor}} = 1.1 \times 10^4\,\mathrm{K}$ to relative tolerance `TH0_tol = 1d-8`. Two upstream properties interact in this loop:
  1. Line 772 writes `D = F2 * (T4_0 - T4_0) / (F2 - F1)` (subtracting `T4_0` from itself rather than `T4_0 - T4_prev`), which evaluates identically to `0.0d0` so the outer `stah` iteration reduces to pure bisection on `[dmN_small, dmN_large]` after initial sign bracketing.
  2. Each shell temperature step invokes `get_T_estimate` with loose root-finding tolerances `epsx = 1d-5, epsy = 1d-5` (`rsp_build.f90:675-676`). On low-metallicity `Gs98` opacity tables (`Z = 0.0014`), discrete switching of the internal `brent_safe_zero` iteration count introduces $\sim 2 \times 10^{-5}$ quantization jumps in $T_N(\Delta m_N)$ that exceed `TH0_tol = 1d-8` even when the bisection interval width has converged to machine precision.
- **`FlexMESA` Note**: `crates/flexmesa-star/src/rsp_stah.rs` preserves upstream Fortran MESA's `epsx = 1e-5, epsy = 1e-5` inner step and bisection update (tested against all 7 MESA release `26.4.1` `rsp_*_26_4_1.mod` models in `rsp_build_lina_test.rs`; removal of non-upstream departures is tracked under [#178](https://github.com/joshspeagle/FlexMESA/issues/178)).

### 2.9 Dimensional $w^4$ Turbulent Pressure, `Cp` vs. `T` Flux-Limiter Typo, and Dropped $\partial D_r / \partial w$ in `RSP2`

- **Upstream Locations**: `star/private/star_utils.f90` (`star_utils.f90:3334`, `star_utils.f90:3341-3342`, and `star_utils.f90:3346`, inside `calc_Ptrb_ad_tw`) and `star/private/hydro_rsp2.f90` (`hydro_rsp2.f90:311` inside `do1_turbulent_energy_eqn`, `hydro_rsp2.f90:567` inside `compute_PII_face`, and `hydro_rsp2.f90:912` and `hydro_rsp2.f90:922` inside `compute_Dr`).
- **Root Cause**: Four distinct coding defects appear in the experimental `RSP2` turbulent-convection solver in upstream Fortran MESA release `26.4.1` (tracked for restoration to upstream behaviour under [#178](https://github.com/joshspeagle/FlexMESA/issues/178)):
  1. `calc_Ptrb_ad_tw` (`star/private/star_utils.f90:3341-3342`) defines `Ptrb_div_etrb = s% RSP2_alfap * x_ALFAP * etrb * rho` (which already equals $P_{\mathrm{trb}} = \frac{2}{3}\alpha_p \rho w^2$ because $e_{\mathrm{trb}} = w^2$) and then multiplies by `etrb` a second time (`Ptrb = Ptrb_div_etrb * etrb = (2/3) * RSP2_alfap * rho * w^4`), yielding a dimensionally invalid $\mathrm{g\,cm^{-3}\,(cm\,s^{-1})^4}$ pressure while `Ptrb_start` on line 3346 omits `x_ALFAP = 2/3` and line 3334 checks `RSP_nz_div_IBOTOM` rather than `RSP2_nz_div_IBOTOM`.
  2. `compute_PII_face` (`star/private/hydro_rsp2.f90:567`) writes `T_face = alfa*Cp_00 + beta*Cp_m1` instead of `alfa*T_00 + beta*T_m1` inside the `use_TDC_enthalpy_flux_limiter` branch, substituting specific heat $C_P$ for temperature $T$ in the dimensionless flux-limiter argument $X = (C_{P,\mathrm{face}} T_{\mathrm{face}} / h_{\mathrm{face}})(x_{\alpha_s}\alpha Y_{\mathrm{face}} / \sqrt{2/3})$.
  3. `compute_Dr` (`star/private/hydro_rsp2.f90:912` and `star/private/hydro_rsp2.f90:922`) constructs the automatic-differentiation variable `w_00 = wrap_w_00(s, k)` on line 912 yet multiplies by the scalar `get_etrb(s, k)` (`pow2(s% w(k))`) on line 922, dropping $\partial D_r / \partial w_k$ from the analytical Henyey Jacobian of `do1_turbulent_energy_eqn`.
  4. `do1_turbulent_energy_eqn` (`star/private/hydro_rsp2.f90:311`) scales the non-turbulent cell residual `w_00 / s% csound(k)` by the un-wrapped trial sound speed rather than start-of-step `csound_start(k)`, omitting $(\ln\rho_k, \ln T_k)$ partial derivatives whenever trial $w_k \neq 0$.
- **Pre-Audit Departure (Scheduled for Upstream Restoration under `#178`)**: `crates/flexmesa-star/src/rsp2_ad.rs`, `crates/flexmesa-star/src/mesh.rs`, and `crates/flexmesa-star/src/hydro_state.rs` currently evaluate $P_{\mathrm{trb}} = \frac{2}{3}\alpha_p \rho w^2$ and $P_{\mathrm{trb,start}} = \frac{2}{3}\alpha_p \rho_{\mathrm{start}} w_{\mathrm{start}}^2$ gated by `RSP2_nz_div_IBOTOM`, interpolate face temperature `t_face = t_00 * alfa + t_m1 * beta` in `use_tdc_enthalpy_flux_limiter`, retain `w_00 * w_00` in `compute_Dr`, and scale non-turbulent damping by `csound_start` (`crates/flexmesa-star/tests/rsp2_test.rs`). Per `DECISIONS #17`, these departures are scheduled to revert to exact MESA `26.4.1` behaviour under [#178](https://github.com/joshspeagle/FlexMESA/issues/178).

### 2.10 MLT Cubic Cancellation Near Marginal Convection (`mlt.f90`)

- **Upstream Location**: `turb/private/mlt.f90` (Cardano/Cox convective efficiency polynomial).
- **Root Cause**: When a stellar face approaches marginal convective neutrality ($0 < \delta = a_0 B^3 = a_0 A^2 (\nabla_{\mathrm{rad}} - \nabla_L) \ll 1$), evaluating the Cardano cubic root $\Gamma(B)$ directly in IEEE-754 `f64` suffers subtractive cancellation between nearly equal cube roots, producing positive-ULP spikes ($\Gamma \sim +10^{-16} \gg B \implies \zeta \to 1$) and discontinuous jumps in $\partial\nabla_T / \partial\ln T$ when $0 < \delta \lesssim 10^{-12}$, whereas a dimensional floor on $\nabla_{\mathrm{rad}} - \nabla_L$ would misclassify deep high-$A$ convective cores ($A \sim 10^8$, $\delta \gg 1$ even at $\nabla_{\mathrm{rad}} - \nabla_L \sim 10^{-13}$).
- **`FlexMESA` Note (Subject to `#157` Upstream Review)**: `crates/flexmesa-turb/src/mlt.rs` evaluates the 3-term small-$\delta$ cubic series $\Gamma(\delta) = \delta\,[1 - \delta + (2 - a_0)\delta^2]$ and $\zeta(\delta) = a_0 \delta^2 [1 - \delta + (2 - a_0)\delta^2]^3$ whenever $0 < \delta \le 10^{-6}$ and returns the radiative limit ($\Gamma = 0$, $\zeta = 0$, $\nabla_T = \nabla_{\mathrm{rad}}$) for $\nabla_{\mathrm{rad}} \le \nabla_L$. Upstream MESA `26.4.1` `mlt.f90` alignment is under review in [#157](https://github.com/joshspeagle/FlexMESA/issues/157).

### 2.11 Mass-Change Heating, Mass-Adjustment Clamping, and Stop-Condition Deferred Upstream Bugs (`eps_mdot.f90` / `adjust_mass.f90` / `do_one_utils.f90` / `history.f90`)

- **Upstream `MESA` Behavior**: Upstream `adjust_mass` evaluates $\delta e_{\mathrm{tot}}$ across small wind and accretion steps using 128-bit quadruple-precision (`real(qp)`) total-energy arrays so changes in gravitational potential $\Delta(-G m_{\mathrm{grav}}/r)$ do not suffer catastrophic cancellation against $-G M / r$. Additionally, MESA `26.4.1` contains three audited bugs tracked under [#169](https://github.com/joshspeagle/FlexMESA/issues/169):
  1. `star/private/adjust_mass.f90:239-265` ([#263](https://github.com/joshspeagle/FlexMESA/issues/263)): when clamping a crossing step against `min_star_mass_for_loss` or `max_star_mass_for_gain`, upstream updates `new_mstar`, `delta_m`, and `mstar_dot` without updating `new_xmstar`, leaving `s% xmstar` and `s% mstar = s% xmstar + s% M_center` at the unclamped mass on the crossing step.
  2. `star/private/do_one_utils.f90:738-745` and `:790-796` ([#264](https://github.com/joshspeagle/FlexMESA/issues/264)): upstream aborts via `mesa_error` when `peak_burn_vconv_div_cs > 0.75 * peak_burn_vconv_div_cs_limit` before the clean `>= limit` stop check can be reached.
  3. `star/private/do_one_utils.f90:635` and `star/private/history.f90:1956-1957` ([#265](https://github.com/joshspeagle/FlexMESA/issues/265)): upstream computes `v_surf_div_v_kh` as `v_surf / (R_phot_Rsun / kh_timescale_yr)` in mixed units ($\mathrm{cm\,s^{-1}\cdot yr / R_\odot}$).
- **`FlexMESA` Note (`#159`)**: `crates/flexmesa-star/src/eps_mdot.rs` (`calculate_eps_mdot_with_change`, given pre-step `PrevMassState` arrays) computes $\Delta r_k = \frac{3}{4\pi} \Delta V_k / (r_{\mathrm{new},k}^2 + r_{\mathrm{new},k} r_{\mathrm{old},k} + r_{\mathrm{old},k}^2)$ and $\Delta\phi_k = -G (r_{\mathrm{old},k} \Delta m_{\mathrm{grav},k} - m_{\mathrm{grav},\mathrm{old},k} \Delta r_k) / (r_{\mathrm{old},k} r_{\mathrm{new},k})$ in `f64` (including `v_flag` kinetic energy and `mass_correction` in `eval_cell_specific_total_energy` and `compute_leak_frac` at `k = nz - 1`), and `crates/flexmesa-star/src/adjust_mass.rs`, `termination.rs`, and `mesa_history.rs` reproduce `#263`, `#264`, and `#265` verbatim pending the post-`v0.1.0` `#169` upstream-bug review.

### 2.12 Swapped `(d_alfa_dlogT, d_alfa_dlogRho)` Arguments in `HELM`-to-Ideal Blend (`eosdt_eval.f90:728,749-750`)

- **Upstream Location**: `eos/private/eosdt_eval.f90`, line 728 (`call get_helm_alfa`) and lines 749 to 750 (`subroutine get_helm_alfa`).
- **Root Cause**: In MESA `26.4.1`, `subroutine get_helm_alfa(rq, logRho, logT, Z, alfa, d_alfa_dlogT, d_alfa_dlogRho)` declares `d_alfa_dlogT` before `d_alfa_dlogRho` (`eosdt_eval.f90:749-750`), whereas the call site in `eval_eosDT_helm_and_ideal` invokes `call get_helm_alfa(rq, logRho, logT, Z, alfa, d_alfa_dlogRho, d_alfa_dlogT, ierr)` (`eosdt_eval.f90:728`), swapping the density and temperature blend-weight partial derivatives across the narrow outer `HELM`-to-ideal-gas edge ramp ($\log_{10}\rho \in [10.8, 10.9]$ and $\log_{10} T \in [2.9, 3.1]$ when $e^-e^+$ pairs are excluded).
- **`FlexMESA` Behaviour (`#176`, Deferred Post-Baseline Review in `#169`)**: Per `DECISIONS #17` and `#36`, both `crates/flexmesa-eos/src/blend_dt.rs` (`eval_lvl6`) and `src/flexmesa/eos/_blend_dt_jax.py` (`_lvl6`) reproduce MESA `26.4.1`'s swapped `(d_alfa_dlogRho, d_alfa_dlogT)` call-site argument order without option flags, matching compiled MESA `26.4.1` `eosdt_helm_ideal_hi_rho_blend` and `eosdt_helm_ideal_lo_t_noep_blend` oracle rows in `mesa_26_4_1_eos_oracles.tsv` (`< 10^{-13}` in Safe Rust `"strict"` mode). Whether to fix the swapped derivatives after the `26.4.1` baseline is tracked in [#169](https://github.com/joshspeagle/FlexMESA/issues/169) and [#176](https://github.com/joshspeagle/FlexMESA/issues/176).

---

## 3. Script-Generated Trajectory & Numerics Reports (`#142`, `#143`, `#165`, `#174`)

Earlier hand-typed `E2E-1` through `E2E-5` tables on this page claimed `107 / 107` cases verified at `0 ULP` from step-capped runs and have been removed following the repository audit (`notes/audit/README.md`, [#142](https://github.com/joshspeagle/FlexMESA/issues/142), `DECISIONS #17`). Under current governance:

- **End-to-End `test_suite` Status (`106` MESA `26.4.1` Cases)**: Reported exclusively by the script-generated table in [`notes/audit/case_table.md`](https://github.com/joshspeagle/FlexMESA/blob/main/notes/audit/case_table.md) (`flexmesa harness table`, [#143](https://github.com/joshspeagle/FlexMESA/issues/143)) and the generated port dashboard ([#174](https://github.com/joshspeagle/FlexMESA/issues/174)).
- **`Fast` vs. `Strict` Numerics Differences (`#165`)**: Reported by the script-generated table in [`notes/audit/fast_vs_strict.md`](https://github.com/joshspeagle/FlexMESA/blob/main/notes/audit/fast_vs_strict.md) (`scripts/generate_fast_vs_strict_report.py`), covering all `47` math vector sections (`474,062` inputs) and a short `star/1.3M_ms_high_Z` evolution.

---

## Programmatic Tolerance Comparison Example

```python
import numpy as np
from flexmesa.testing import golden

# Compare two 64-bit thermodynamic arrays against a relative tolerance bound
ref = np.array([1.43822426e02, 5.68883671e03], dtype=np.float64)
val = ref * (1.0 + 2.0e-15)
golden.assert_values_match(float(val[0]), float(ref[0]), rtol=1.0e-12)
max_rel = float(np.max(np.abs(val - ref) / np.abs(ref)))
print(f"Relative tolerance check passed: max_rel_err = {max_rel:.3e}")
```
