# Canonical MESA I–VI Instrument Paper Figure Atlas

Beyond unit, property-based, and end-to-end `test_suite` verification, `FlexMESA` reproduces canonical diagnostic figures across all six MESA instrument papers:

- **MESA I** ([Paxton et al. 2011, ApJS, 192, 3](https://doi.org/10.1088/0067-0049/192/1/3)): Microphysics coverage, Zero-Age Main Sequence (ZAMS) and evolutionary tracks across mass, and Kippenhahn interior structure diagrams.
- **MESA II** ([Paxton et al. 2013, ApJS, 208, 4](https://doi.org/10.1088/0067-0049/208/1/4)): Irradiated giant planets, solar-like and red-giant asteroseismology, Spruit-Tayler magnetic dynamos, and core-collapse supernova (`CCSN`) progenitors.
- **MESA III** ([Paxton et al. 2015, ApJS, 220, 15](https://doi.org/10.1088/0067-0049/220/1/15)): Interacting binary Roche-lobe overflow (`RLOF`), weak nuclear Urca cooling shells, HLLC Riemann shock hydrodynamics, and Type II-P supernova Rayleigh-Taylor mixing.
- **MESA IV** ([Paxton et al. 2018, ApJS, 234, 34](https://doi.org/10.3847/1538-4365/aaa5a8)): Convective premixing and predictive mixing, Burgers multi-component atomic diffusion with radiative levitation, and radial stellar pulsations (`RSP`).
- **MESA V** ([Paxton et al. 2019, ApJS, 243, 10](https://doi.org/10.3847/1538-4365/ab2241)): Radial pulsation work integrals and multi-band light curves, time-centered `dedt` energy conservation, and white dwarf crystallization with ${}^{22}\mathrm{Ne}$ phase-separation heating.
- **MESA VI** ([Jermyn et al. 2023, ApJS, 265, 15](https://doi.org/10.3847/1538-4365/acae8d)): Time-dependent convection (`TDC`) vs. steady `Cox` mixing-length theory (`MLT`), operator-overloaded forward automatic differentiation (`AutoDiff`), and `Skye` solid/liquid Coulomb EOS phase separation.

Every figure in this atlas is generated deterministically by `scripts/mesa_paper_figures/reproduce_mesa_{i..vi}.py` and paired with a machine-readable quantitative landmark ledger (`docs/assets/figures/mesa_papers/mesa_<i..vi>/ledger.csv`) verified in CI by `tests/test_mesa_paper_figures.py`.

---

## 1. MESA I (Paxton et al. 2011): Microphysics, HR Tracks & Interior Structure

### 1.1 Equation of State (`eos`) & Opacity (`kap`) Regime Coverage

The left panel maps the $(\log_{10}\rho, \log_{10}T)$ thermodynamic domain across the Coulomb/PC vs. `HELM` blend weight $1 - \alpha_{\mathrm{HELM}}$ (`eos.get_pc_alfa`), alongside the electron degeneracy ($\eta \approx 4$), radiation-pressure ($P_{\mathrm{rad}} = P_{\mathrm{gas}}$), and $e^\pm$ pair-instability ($\Gamma_1 < 4/3$) boundaries (reproducing MESA I Figs. 1–2). The right panel evaluates Rosseland mean opacity $\log_{10}\kappa(T)$ across $\log_{10} R \in [-5, +1]$ (`kap.kap_get_results`), showing the H$^{-}$, bound-free metal, and iron-bump peaks together with electron conduction turnover at high density (MESA I Figs. 3–5).

![MESA I EOS regime map and Rosseland opacity profiles](../assets/figures/mesa_papers/mesa_i/mesa_i_eos_kap_regimes.png)

### 1.2 Hertzsprung-Russell Diagram & Central $(\log_{10}\rho_c, \log_{10}T_c)$ Trajectories

The left panel traces the ZAMS locus from $0.1\,M_\odot$ to $100\,M_\odot$, alongside the complete $1\,M_\odot$ trajectory from pre-main-sequence contraction through the red-giant branch (`RGB`) tip, core-helium flash, horizontal branch, asymptotic giant branch (`AGB`), and white dwarf cooling sequence (MESA I Figs. 10–15). The right panel tracks the corresponding $1\,M_\odot$ and $25\,M_\odot$ central $(\log_{10}\rho_c, \log_{10}T_c)$ trajectories across H, He, and C burning ignition thresholds (MESA I Figs. 11–16).

![MESA I Hertzsprung-Russell diagram and central density-temperature trajectories](../assets/figures/mesa_papers/mesa_i/mesa_i_hr_and_central_tracks.png)

### 1.3 Kippenhahn Convection/Burning History & Interior Abundance Profiles

The left panel shows the Kippenhahn mass-coordinate diagram $m/M_\odot$ vs. evolutionary stage coordinate for a $1\,M_\odot$ star, delineating the convective envelope, He core boundary, and C/O core boundary (MESA I Figs. 13–14). The right panel displays the interior mass-fraction profiles $X_i(m/M_\star)$ for ${}^1\mathrm{H}$, ${}^4\mathrm{He}$, ${}^{12}\mathrm{C}$, and ${}^{16}\mathrm{O}$ across enclosed mass fraction $q = m/M_\star$ (MESA I Fig. 22).

![MESA I Kippenhahn diagram and interior chemical abundance profiles](../assets/figures/mesa_papers/mesa_i/mesa_i_kippenhahn_and_abundances.png)

---

## 2. MESA II (Paxton et al. 2013): Giant Planets, Asteroseismology & Massive Stars

### 2.1 Irradiated Gas-Giant Radius Contraction & Interior $T(P)$ Profiles

The left panel compares radius contraction histories $R_p(t)/R_{\mathrm{Jup}}$ for $1\,M_{\mathrm{Jup}}$ gas giants across equilibrium irradiation temperatures $T_{\mathrm{eq}} \in \{1500\,\mathrm{K}, 1000\,\mathrm{K}\}$, isolated cooling, and a $10\,M_\oplus$ heavy-element core, evaluated via `star.e2e.eval_planetary_irradiation_heating` with column depth $\Sigma_\star = 300\,\mathrm{g\,cm^{-2}}$ (MESA II Figs. 1–4). The right panel shows the interior temperature-pressure profiles $\log_{10} T(\log_{10} P)$, illustrating the outer radiative isothermal plateau and deepening of the radiative-convective boundary under strong stellar irradiation.

![MESA II irradiated gas-giant radius contraction and temperature-pressure profiles](../assets/figures/mesa_papers/mesa_ii/mesa_ii_giant_planets.png)

### 2.2 Solar-Like Echelle Diagram & Red-Giant Mixed-Mode Period Spacings

The left panel constructs the solar-like $(\ell = 0, 1, 2)$ echelle diagram ($\nu \bmod \Delta\nu$ vs. $\nu$) evaluated via `star.eval_linear_oscillation_modes`, `star.eval_astero_scaling_dnu_numax_uhz`, and `astero.eval_surface_correction`, showing the high-frequency acoustic surface-term shift toward lower modulo frequency (MESA II Figs. 6–9). The right panel plots the red-giant dipole ($\ell = 1$) mixed-mode period spacing $\Delta P_1(\nu)$ across acoustic-gravity coupling dips against the asymptotic buoyancy spacing $\Delta\Pi_1$ (MESA II Figs. 10–11).

![MESA II solar-like echelle diagram and red-giant mixed-mode period spacings](../assets/figures/mesa_papers/mesa_ii/mesa_ii_asteroseismology_echelle.png)

### 2.3 Spruit-Tayler Dynamo Differential Rotation & Core-Collapse Progenitor Structure

The left panel compares internal angular velocity $\log_{10}\Omega(m)$ and toroidal magnetic field $\log_{10}(B_\phi/10^6\,\mathrm{G})$ profiles in a $15\,M_\odot$ massive star with and without Spruit-Tayler (`ST`) magnetic dynamo torques (`star.e2e.eval_spruit_tayler_dynamo_diagnostics`, MESA II Figs. 26–31). The right panel shows the pre-collapse onion-skin composition stratification (${}^{54,56}\mathrm{Fe}, {}^{28}\mathrm{Si}, {}^{16}\mathrm{O}/{}^{20}\mathrm{Ne}, {}^{4}\mathrm{He}, {}^{1}\mathrm{H}$) and specific entropy $s(m)/(N_{\mathrm{A}} k_{\mathrm{B}})$ at iron-core infall (`star.e2e.eval_fe_core_infall_state`, MESA II Figs. 33–38).

![MESA II Spruit-Tayler dynamo differential rotation and core-collapse progenitor structure](../assets/figures/mesa_papers/mesa_ii/mesa_ii_rotation_and_ccsn_progenitor.png)

---

## 3. MESA III (Paxton et al. 2015): Interacting Binaries, Weak Urca Rates & Shock Hydrodynamics

### 3.1 Binary Roche-Lobe Overflow (`RLOF`) & Orbital Angular Momentum Evolution

The left panel compares optically thin exponential (`binary.eval_ritter_mdot_thin`, Ritter 1988) and optically thick (`binary.eval_kolb_mdot_thick`, Kolb & Ritter 1990) Roche-lobe overflow mass-transfer rates $\log_{10}|\dot{M}_{\mathrm{tr}}|$ across relative donor overflow $(R_1 - R_{\mathrm{RL},1})/R_{\mathrm{RL},1}$ (MESA III Figs. 1–5). The right panel shows gravitational-wave orbital inspiral (`binary.eval_gr_expected_separation_rsun`, Peters 1964) alongside non-conservative mass-transfer orbital widening (`binary.eval_tauris_vdh_expected_separation_rsun`, MESA III Figs. 6–9).

![MESA III binary Roche-lobe overflow regimes and orbital separation evolution](../assets/figures/mesa_papers/mesa_iii/mesa_iii_binary_rlof_and_jdot.png)

### 3.2 Degenerate Weak Urca Rates & Neutrino Cooling Shells

The left panel plots electron-capture and $\beta^-$-decay weak nuclear reaction rates $\log_{10}(\lambda/\mathrm{s^{-1}})$ as a function of electron density $\log_{10}(\rho Y_e)$ for the $A=23$ (${}^{23}\mathrm{Na} \leftrightarrow {}^{23}\mathrm{Ne}$) and $A=25$ (${}^{25}\mathrm{Mg} \leftrightarrow {}^{25}\mathrm{Na}$) Urca pairs, marking the Fermi-energy threshold crossings where $\lambda_{\mathrm{ec}} = \lambda_{\beta^-}$ (MESA III Figs. 15–18). The right panel shows the localized Urca thermal neutrino cooling spike $\epsilon_{\mathrm{Urca}} < 0$ at each shell threshold (MESA III Figs. 19–20).

![MESA III weak Urca electron-capture/beta-decay rates and neutrino cooling shells](../assets/figures/mesa_papers/mesa_iii/mesa_iii_weak_rates_urca.png)

### 3.3 HLLC Riemann Shock-Tube & Type II-P Supernova Rayleigh-Taylor Mixing

The left panel shows normalized Sod shock-tube density $\rho(x)$, pressure $P(x)$, and velocity $u(x)$ profiles across rarefaction, contact discontinuity, and shock waves evaluated with the HLLC Riemann solver (`StarModel.eval_riemann_profile` and `star.eval_hllc_face_jax`, MESA III Figs. 24–27). The right panel shows Type II-P core-collapse supernova shock propagation, Duffell (2016) Rayleigh-Taylor turbulent mixing, and ${}^{56}\mathrm{Ni}$ / ${}^1\mathrm{H}$ mixing through composition interfaces (`star.e2e.eval_duffell_rti_shock_state`, MESA III Figs. 28–35).

![MESA III HLLC shock-tube profiles and Type II-P supernova Rayleigh-Taylor mixing](../assets/figures/mesa_papers/mesa_iii/mesa_iii_hllc_shock_and_ccsn.png)

---

## 4. MESA IV (Paxton et al. 2018): Convective Premixing, Radiative Levitation & Radial Pulsations

### 4.1 Convective Premixing Boundary Alignment & Core Mass Growth

The left panel demonstrates how convective premixing (`StarModel.do_conv_premix`) advances the convective core boundary outward until the convective side of the boundary satisfies neutral buoyancy $\nabla_{\mathrm{rad}} / \nabla_{\mathrm{ad}} = 1$, removing the discontinuous jump left by standard MLT (MESA IV Figs. 1–4). The right panel compares the resulting core-helium burning (`CHeB`) convective He core mass evolution $m_{\mathrm{core}}(t)/M_\odot$ across unextended MLT, predictive mixing (`StarModel.add_predictive_mixing`), and convective premixing (MESA IV Figs. 5–8).

![MESA IV convective premixing gradient ratio and core mass evolution](../assets/figures/mesa_papers/mesa_iv/mesa_iv_conv_premix_and_predictive.png)

### 4.2 Burgers Atomic Diffusion, Radiative Levitation & Surface Abundance Evolution

The left panel compares outward radiative levitation acceleration $\log_{10} g_{\mathrm{rad},i}$ for ${}^{56}\mathrm{Fe}$ and ${}^{58}\mathrm{Ni}$ against inward gravitational acceleration $\log_{10} g(r)$ in outer stellar envelopes (`StarModel.do_element_diffusion`, MESA IV Figs. 10–12). The right panel tracks surface abundance evolution $\log_{10}(X_{\mathrm{surf}}/X_0)$ across main-sequence age, showing gravitational settling of ${}^4\mathrm{He}$ and $\mathrm{C, N, O}$ contrasted with radiative levitation enrichment of ${}^{56}\mathrm{Fe}$ (MESA IV Figs. 13–15).

![MESA IV radiative levitation acceleration and surface abundance evolution](../assets/figures/mesa_papers/mesa_iv/mesa_iv_diffusion_and_levitation.png)

### 4.3 `RSP` Petersen Period-Ratio Diagram & Nonlinear Cepheid Pulsation Curves

The left panel plots the double-mode Petersen diagram ($P_1/P_0$ vs. $\log_{10} P_0$) evaluated with the linear non-adiabatic (`LINA`) pulsation eigenmode solver (`star.eval_rsp_lina_modes`) across RR Lyrae and classical Cepheid sequences (MESA IV Figs. 25–28). The right panel shows the nonlinear radial pulsation light curve $\Delta\log_{10}(L/L_0)$ and radial velocity curve $v_{\mathrm{surf}}(t/P_0)$ over two pulsation cycles, reproducing the asymmetric sawtooth rise and bump progression (MESA IV Figs. 29–32).

![MESA IV RSP Petersen period-ratio diagram and nonlinear Cepheid pulsation curves](../assets/figures/mesa_papers/mesa_iv/mesa_iv_rsp_periods_and_lightcurves.png)

---

## 5. MESA V (Paxton et al. 2019): Pulsation Work Integrals, `dedt` Energy Conservation & WD Cooling Delays

### 5.1 Differential Pulsation Work Integral & Multi-Band Cepheid Light Curves

The left panel plots the differential radial work integral $\mathrm{d}W/\mathrm{d}\log_{10} T$ and cumulative work $W(<T)$ across the $\mathrm{H/He\,I}$ ($\log_{10} T \approx 4.06$) and $\mathrm{He\,II}$ ($\log_{10} T \approx 4.62$) $\kappa$-mechanism ionization driving zones (`star.eval_target_mode_frequency_and_growth`, MESA V Figs. 2–6). The right panel displays synthetic Johnson-Cousins $B$, $V$, and $I$ light curves over two pulsation phases, illustrating decreasing amplitude and phase lag toward longer wavelengths (MESA V Figs. 7–12).

![MESA V radial pulsation work integral and multi-band Cepheid light curves](../assets/figures/mesa_papers/mesa_v/mesa_v_rsp_work_and_multiband.png)

### 5.2 Time-Centered `dedt` vs. `eps_grav` Energy Conservation Budget

The left panel compares per-step relative energy conservation residuals $\log_{10}|\Delta E / E|$ between the time-centered total internal + kinetic energy equation (`dedt_form`, achieving $\sim 10^{-10}$ step precision) and the legacy $-\,T\,\mathrm{d}s/\mathrm{d}t$ (`eps_grav`) formulation (MESA V Figs. 15–18). The right panel shows cumulative relative energy error $\log_{10}|\sum \Delta E / E_{\mathrm{tot}}|$ scaling linearly with Henyey solver residual tolerance under `dedt_form` versus saturating at an $\mathcal{O}(\Delta t)$ truncation floor under `eps_grav` (MESA V Figs. 19–21).

![MESA V time-centered dedt vs eps_grav energy conservation residuals](../assets/figures/mesa_papers/mesa_v/mesa_v_energy_conservation_budget.png)

### 5.3 White Dwarf Crystallization, ${}^{22}\mathrm{Ne}$ Sedimentation & Cooling Age Delay

The left panel compares $0.6\,M_\odot$ C/O white dwarf cooling tracks $\log_{10}(L/L_\odot)$ vs. cooling age $t_{\mathrm{cool}}$ across baseline Mestel cooling, Coulomb crystallization latent heat + C/O phase separation (`star.e2e.eval_wd_phase_separation` and `StarModel.do_phase_separation`), and ${}^{22}\mathrm{Ne}$ gravitational sedimentation heating (MESA V Figs. 22–25). The right panel quantifies the resulting multi-Gyr cooling age delay $\Delta t_{\mathrm{cool}}(L)$ at faint luminosities $\log_{10}(L/L_\odot) \lesssim -3.8$ (MESA V Figs. 26–27).

![MESA V white dwarf crystallization cooling tracks and Ne22 sedimentation age delay](../assets/figures/mesa_papers/mesa_v/mesa_v_wd_crystallization_and_22ne.png)

---

## 6. MESA VI (Jermyn et al. 2023): Time-Dependent Convection, AutoDiff & `Skye` Phase Separation

### 6.1 Time-Dependent Convection (`TDC`) vs. Steady `Cox` MLT

The left panel compares convective velocity $v_{\mathrm{conv}}(\nabla_r)$ between steady `Cox` MLT (`turb.set_mlt`), transient short-timestep `TDC` ($\Delta t = 15\,\mathrm{s}$), and asymptotic long-timestep `TDC` ($\Delta t = 10^6\,\mathrm{s}$, `turb.set_tdc`, MESA VI Figs. 2–4). The right panel demonstrates exact asymptotic relaxation of $v_{\mathrm{TDC}}/v_{\mathrm{Cox}} \to 1$ and $\nabla_{T,\mathrm{TDC}}/\nabla_{T,\mathrm{Cox}} \to 1$ as $\Delta t / \tau_{\mathrm{conv}}$ increases across seven orders of magnitude (MESA VI Figs. 5–7).

![MESA VI time-dependent convection vs steady Cox MLT convergence](../assets/figures/mesa_papers/mesa_vi/mesa_vi_tdc_vs_cox_mlt.png)

### 6.2 Analytical Forward `AutoDiff` V-Curve & Multi-Zone Throughput Scaling

The left panel plots the classic numerical differentiation V-curve for $f(x) = e^x \sin x$ across finite-difference step size $\log_{10} h \in [-15, -1]$ (where truncation error $\propto h^2$ for large $h$ and floating-point roundoff cancellation $\propto \epsilon_{\mathrm{mach}}/h$ for small $h$ limit centered finite differences to $\sim 10^{-12}$), contrasted against `auto_diff.DualOrder1` forward automatic differentiation at exact IEEE-754 machine epsilon ($\le 10^{-15}$, MESA VI Figs. 10–12). The right panel compares per-zone stencil Jacobian throughput across $N_{\mathrm{zones}} \in [10^2, 4\times 10^3]$ (MESA VI Figs. 13–14).

![MESA VI automatic differentiation V-curve and multi-zone Jacobian scaling](../assets/figures/mesa_papers/mesa_vi/mesa_vi_autodiff_vcurve_and_scaling.png)

### 6.3 `Skye` Solid/Liquid Coulomb EOS Phase Transition & White Dwarf C/O Separation

The left panel evaluates the `Skye` solid phase parameter $\phi_{\mathrm{cryst}}$ and adiabatic exponent shift $\Gamma_1 - 1$ across $\log_{10}(\rho/\mathrm{g\,cm^{-3}}) \in [5.8, 8.6]$ (`eos.eval_skye_eos_zones`, MESA VI Figs. 16–19). The right panel plots the resulting phase-separated ${}^{16}\mathrm{O}$ and ${}^{12}\mathrm{C}$ abundance profiles (`star.eval_phase_separation_shift_jax`) and the Ledoux buoyancy term $B$ peak at the crystallization boundary $q_{\mathrm{cryst}}$ across the white dwarf interior mass coordinate $q = m/M_{\mathrm{WD}}$ (MESA VI Figs. 20–22).

![MESA VI Skye Coulomb phase transition and white dwarf oxygen enrichment](../assets/figures/mesa_papers/mesa_vi/mesa_vi_skye_wd_phase_separation.png)

---

## Summary of Quantitative Landmarks Across MESA I–VI

| Instrument Paper | Canonical Diagnostic | Ledger Key | Verified Value | Physical Invariant / Reference Bound |
| :--- | :--- | :--- | ---: | :--- |
| **MESA I** | $1\,M_\odot$ ZAMS Effective Temperature | `solar_zams_log_teff` | `3.7495` | Solar-metallicity ZAMS $\log_{10}(T_{\mathrm{eff}}/\mathrm{K}) \in [3.74, 3.79]$ |
| **MESA I** | Iron-Bump Rosseland Opacity | `kap_fe_bump_log_kap` | `1.5523` | Z-bump enhancement $\log_{10}(\kappa/\mathrm{cm^2\,g^{-1}}) > 0.5$ |
| **MESA II** | Irradiated $1\,M_{\mathrm{Jup}}$ Radius ($4.5\,\mathrm{Gyr}$) | `jup_hot_1500k_4p5gyr_rjup` | `1.4724 R_Jup` | Inflated radius $R_p(T_{\mathrm{eq}}=1500\,\mathrm{K}) > R_p(\mathrm{cold})$ |
| **MESA II** | CCSN Iron-Core Mass at Infall | `ccsn_fe_core_mass_msun` | `1.7700 Msun` | Chandrasekhar-scale Fe core $M_{\mathrm{Fe}} \in [1.20, 1.85]\,M_\odot$ |
| **MESA III** | Peak Kolb Optically Thick RLOF Rate | `rlof_peak_log_mdot_msun_yr` | `-4.082` | $\log_{10}(|\dot{M}_{\mathrm{tr}}|/M_\odot\,\mathrm{yr}^{-1}) \in [-6.0, -1.0]$ |
| **MESA III** | HLLC Rankine-Hugoniot Shock Residual | `hllc_rankine_hugoniot_rel_err` | `1.20e-4` | Well-balanced Lagrangian HLLC shock jump error $< 10^{-2}$ |
| **MESA IV** | Radiative Levitation ${}^{56}\mathrm{Fe}$ Peak Ratio | `levitation_fe_peak_over_g` | `1.382` | Selective Z-bump metal support $g_{\mathrm{rad}}(\mathrm{Fe}) / g > 1.0$ |
| **MESA IV** | Fundamental Cepheid `LINA` Period | `cepheid_p0_days` | `4.673 d` | Classical Cepheid fundamental period $P_0 \in [3.0, 7.0]\,\mathrm{d}$ |
| **MESA V** | `dedt_form` Median Step Energy Residual | `dedt_rel_energy_err` | `1.52e-10` | Time-centered `dedt` step residual $|\Delta E/E| < 10^{-8}$ |
| **MESA V** | ${}^{22}\mathrm{Ne}$ Sedimentation Cooling Delay | `wd_22ne_delay_gyr` | `1.126 Gyr` | Faint WD gravitational settling delay $\Delta t_{\mathrm{cool}} > 0.8\,\mathrm{Gyr}$ |
| **MESA VI** | Asymptotic `TDC` to `Cox` MLT Ratio | `tdc_long_dt_vc_rel_err` | `1.22e-16` | Exact steady-state convergence $v_{\mathrm{TDC}}(\Delta t \gg \tau) \to v_{\mathrm{Cox}}$ |
| **MESA VI** | `DualOrder1` Forward AutoDiff Error | `autodiff_exact_rel_err` | `1.00e-16` | Machine-precision analytic tangent vs. $\sim 10^{-12}$ finite-diff floor |

---

## Reproducing the MESA I–VI Figure Atlas Locally

You can regenerate all 18 figures and 6 CSV ledgers directly via `scripts/mesa_paper_figures` or inspect any paper's quantitative landmarks in Python:

```python
import csv
import pathlib

root = pathlib.Path.cwd()
ledger_path = (
    root / "docs" / "assets" / "figures" / "mesa_papers" / "mesa_vi" / "ledger.csv"
)
if ledger_path.exists():
    with ledger_path.open(encoding="utf-8") as f:
        rows = {row["key"]: float(row["value"]) for row in csv.DictReader(f)}
    assert rows["autodiff_exact_rel_err"] <= 1.0e-14
    assert rows["tdc_long_dt_vc_rel_err"] < 1.0e-4
```
