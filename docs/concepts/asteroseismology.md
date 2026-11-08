# Asteroseismology, Surface Corrections & Pulsation Exports (`flexmesa.astero`, `flexmesa.star`)

Asteroseismology probes the interior sound speed $c_s(r)$ and buoyancy frequency $N(r)$ of solar-like oscillators, red giants, and compact pulsators to sub-percent precision. `flexmesa.astero` (`crates/flexmesa-astero` and `src/flexmesa/astero/`) and the pulsation exporters in `flexmesa.star` provide cumulative acoustic transit integrals, global scaling relations, empirical surface-effect frequency corrections, Roxburgh & Vorontsov (2003) separation ratios, standard pulsation structure file exporters (`GYRE`, `FGONG`, `OSC`), and seismic $\chi^2$ optimization drivers.

![Asteroseismic echelle diagram with surface-effect corrections and 5-point separation ratios](../assets/figures/wave5_astero_echelle_surface_corr.png)

*Figure 1: The left panel shows an asteroseismic echelle diagram ($\nu \bmod \Delta\nu$ vs. $\nu$) for radial ($l = 0$) and dipole ($l = 1$) acoustic modes of a solar-like oscillator ($\Delta\nu = 135.0\,\mu\mathrm{Hz}$, $\nu_{\max} = 3090.0\,\mu\mathrm{Hz}$), comparing uncorrected adiabatic model frequencies (open squares) against Ball & Gizon (2014) 2-term surface-corrected frequencies (filled triangles) and observed frequencies (circles with error bars). The right panel displays the Roxburgh & Vorontsov (2003) 5-point frequency separation ratios $r_{010}(\nu)$ and quadrupole-radial ratios $r_{02}(\nu)$, which cancel near-surface structure errors.*

---

## 1. Cumulative Acoustic Transit Integrals & Global Scaling Relations

### 1.1 Acoustic Radius, Acoustic Depth & Asymptotic $\Delta\nu$
Given the adiabatic sound speed $c_s(r) = \sqrt{\Gamma_1 P / \rho}$ across a stellar profile, `eval_acoustic_profile` and `eval_delta_nu_asymptotic` integrate the cumulative acoustic radius $\tau(r)$ from the center and the acoustic depth $\tilde{\tau}(r)$ from the surface photosphere $R_\star$:

$$\tau(r) = \int_0^r \frac{dr'}{c_s(r')}, \qquad \tilde{\tau}(r) = \int_r^{R_\star} \frac{dr'}{c_s(r')}, \qquad \Delta\nu_{\mathrm{asymp}} = \left(2 \int_0^{R_\star} \frac{dr}{c_s(r)}\right)^{-1}.$$

To locate sharp acoustic glitches caused by the base of the convective envelope (`BCZ`) and the second helium ionization zone ($\mathrm{He\,II}$ near $T \approx 4.7 \times 10^4\,\mathrm{K}$), `eval_glitch_acoustic_depths` computes $(\tau_{\mathrm{BCZ}}, \tau_{\mathrm{He\,II}})$, which modulate oscillation frequencies with period $\sim (2\tilde{\tau}_{\mathrm{glitch}})^{-1}$.

### 1.2 Solar-Scaled $\Delta\nu$, $\nu_{\max}$, $\nu_{\mathrm{ac}}$ & Dipole Gravity-Mode Period Spacing $\Delta\Pi_1$
- **Solar Scaling Relations (`eval_delta_nu_scaling`, `eval_nu_max`, `eval_nu_ac`)**:

  $$\frac{\Delta\nu}{\Delta\nu_\odot} = \left(\frac{M}{M_\odot}\right)^{1/2}\left(\frac{R}{R_\odot}\right)^{-3/2}, \qquad \frac{\nu_{\max}}{\nu_{\max,\odot}} = \left(\frac{M}{M_\odot}\right)\left(\frac{R}{R_\odot}\right)^{-2}\left(\frac{T_{\mathrm{eff}}}{T_{\mathrm{eff},\odot}}\right)^{-1/2}.$$

- **Asymptotic Dipole Gravity-Mode Period Spacing (`eval_delta_pg`)**:
  In subgiants, red giants, and white dwarfs where buoyancy-driven gravity waves propagate across a radiative core ($N^2 > 0$), `eval_delta_pg` integrates the asymptotic $l = 1$ g-mode period spacing $\Delta\Pi_1$ (both the high-frequency limit and the Bildsten et al. 2012 finite-frequency formulation):

  $$\Delta\Pi_1 = \frac{2\pi^2}{\sqrt{l(l+1)}} \left(\int_{r_1}^{r_2} \frac{N(r)}{r}\,dr\right)^{-1} = \frac{\sqrt{2}\,\pi^2}{\int_{r_1}^{r_2} (N/r)\,dr}.$$

---

## 2. Five Surface-Effect Frequency Correction Laws (`eval_surface_correction`)

One-dimensional mixing-length atmospheres neglect 3D turbulent pressure support and non-adiabatic radiative damping in the near-surface superadiabatic layers, causing adiabatic model frequencies $\nu_{\mathrm{mod}}$ to systematically overestimate observed p-mode frequencies $\nu_{\mathrm{obs}}$ at high frequency ($\nu \gtrsim \nu_{\max}$). `eval_surface_correction` implements all five MESA empirical surface-effect correction laws weighted by the normalized mode inertia $\mathcal{I}_i$ and inverse variance $\sigma_i^{-2}$:

| Scheme Name | Reference | Frequency Shift Formula $\delta\nu_i = \nu_{\mathrm{corr},i} - \nu_{\mathrm{mod},i}$ |
| :--- | :--- | :--- |
| **`"kjeldsen"`** | Kjeldsen, Bedding & Christensen-Dalsgaard (2008) | $\delta\nu_i = a\,\left(\nu_{\mathrm{obs},i} / \nu_0\right)^b / \mathcal{Q}_i$ with fixed solar exponent $b$ (typically $b \approx 4.90$) |
| **`"cubic"`** | Ball & Gizon (2014) 1-Term Cubic | $\delta\nu_i = a_3\,\left(\nu_{\mathrm{mod},i} / \nu_{\max}\right)^3 / \mathcal{I}_i$ (linear least-squares solve for $a_3$) |
| **`"combined"`** | Ball & Gizon (2014) 2-Term Inverse + Cubic | $\delta\nu_i = \left[a_{-1}\left(\nu_{\mathrm{mod},i} / \nu_{\max}\right)^{-1} + a_3\left(\nu_{\mathrm{mod},i} / \nu_{\max}\right)^3\right] / \mathcal{I}_i$ ($2\times 2$ linear solve) |
| **`"power_law"`** | Free 2-Parameter Power Law | $\delta\nu_i = a\,\left(\nu_{\mathrm{mod},i} / \nu_{\max}\right)^b / \mathcal{I}_i$ (nonlinear fit for both amplitude $a$ and exponent $b$) |
| **`"sonoi"`** | Sonoi et al. (2015) Modified Lorentzian | $\delta\nu_i = a\,\left[1 - \left(1 + (\nu_{\mathrm{mod},i}/\nu_{\max})^b\right)^{-1}\right] / \mathcal{I}_i$ |

---

## 3. Roxburgh & Vorontsov (2003) Separation Ratios $(r_{01}, r_{10}, r_{02})$

As an alternative to parametric surface corrections, Roxburgh & Vorontsov (2003, A&A, 411, 215) showed that ratios of small-to-large frequency separations constructed from 5-point smoothed radial ($l = 0$), dipole ($l = 1$), and quadrupole ($l = 2$) frequencies are insensitive to the outer convective boundary layers and depend almost exclusively on the interior sound-speed gradient (`eval_ratios_r01_r10`, `eval_ratios_r02`):

$$d_{01}(n) = \frac{1}{8}\left(\nu_{n-1,0} - 4\nu_{n-1,1} + 6\nu_{n,0} - 4\nu_{n,1} + \nu_{n+1,0}\right), \qquad r_{01}(n) = \frac{d_{01}(n)}{\Delta\nu_1(n)},$$

$$d_{10}(n) = -\frac{1}{8}\left(\nu_{n-1,1} - 4\nu_{n,0} + 6\nu_{n,1} - 4\nu_{n+1,0} + \nu_{n+1,1}\right), \qquad r_{10}(n) = \frac{d_{10}(n)}{\Delta\nu_0(n+1)},$$

$$r_{02}(n) = \frac{\nu_{n,0} - \nu_{n-1,2}}{\nu_{n,1} - \nu_{n-1,1}}.$$

---

## 4. Dziembowski Variables & Pulsation Exporters (`GYRE`, `FGONG`, `OSC`)

To compute full non-radial adiabatic or non-adiabatic oscillation eigenfrequencies with external pulsation solvers such as [`GYRE`](https://gyre.readthedocs.io) or `ADIPLS`, `flexmesa.star` evaluates the five dimensionless Dziembowski (1971) pulsation structure variables (`eval_dziembowski_variables_jax`, `build_pulse_points`):

$$V = \frac{G m \rho}{P r}, \qquad V_g = \frac{V}{\Gamma_1}, \qquad A^* = \frac{r N^2}{g}, \qquad U = \frac{4\pi r^3 \rho}{m}, \qquad c_1 = \frac{(r/R_\star)^3}{m/M_\star},$$

and exports complete stellar structure profiles (with optional central boundary point insertion and double-point splitting across sharp composition discontinuities) via:

- `model.to_gyre_str(add_center_point=True)`: Standard `GYRE` ASCII format (`version 101` / `120`).
- `model.to_fgong_str(add_center_point=True)`: Standard `FGONG` format (`ivers = 1300`).
- `model.to_osc_str(add_center_point=True)`: `OSC` / `CESAM` pulsation structure format.

---

## 5. Seismic $\chi^2$ Evaluation & Optimization Drivers

`eval_astero_chi2` combines individual mode frequency residuals (or separation ratio residuals) with global spectroscopic and seismic constraints ($T_{\mathrm{eff}}, [\mathrm{Fe/H}], \log g, L, \Delta\nu, \nu_{\max}$) into a weighted composite objective:

$$\chi^2_{\mathrm{total}} = f_{\mathrm{seismo}}\,\chi^2_{\mathrm{seismo}} + \left(1 - f_{\mathrm{seismo}}\right)\,\chi^2_{\mathrm{spectro}}.$$

For model calibration and parameter estimation over $(M_\star, Y_0, Z_0, \alpha_{\mathrm{MLT}}, f_{\mathrm{ov}})$, `flexmesa.astero` provides three built-in derivative-free drivers (`nelder_mead_minimize`, `powell_quadratic_minimize`, and `scan_cartesian_grid`) alongside full reverse-mode `jax.grad` differentiation of the JAX seismic kernels (`flexmesa.astero.jax_kernels`).

=== "Python & JAX (`flexmesa.astero`, `flexmesa.star`)"

    ```python
    import numpy as np
    from flexmesa import astero, star

    # 1. Export a 1.0 Msun ZAMS model to GYRE and FGONG pulsation formats
    model = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
    gyre_txt = model.to_gyre_str(add_center_point=True)
    fgong_txt = model.to_fgong_str(add_center_point=True)

    # 2. Ball & Gizon (2014) 2-term surface correction on radial p-modes
    n_ord = np.arange(14, 26, dtype=np.float64)
    nu_mod = 135.0 * (n_ord + 1.45)
    inertia = 1.0e-7 * np.exp((3100.0 - nu_mod) / 1000.0)
    sigma = np.full_like(nu_mod, 0.15)
    nu_obs = nu_mod - 1.2 * ((nu_mod / 3090.0) ** 3) / inertia
    corr = astero.eval_surface_correction(
        "combined", nu_obs, sigma, nu_mod, inertia, nu_max=3090.0
    )
    chi2_corr = astero.eval_astero_chi2(
        nu_obs, sigma, corr["nu_corrected"], [], [], []
    )["chi2_frequencies"]
    print(
        f"a_3 = {corr['p0']:.4e}, a_-1 = {corr['p1']:.4e}, "
        f"chi^2 = {chi2_corr:.4e}"
    )
    ```

=== "Safe Rust (`crates/flexmesa-astero`)"

    ```rust
    use flexmesa_astero::{eval_combined_corr, eval_seismic_chi2};

    fn main() {
        let nu_mod: Vec<f64> = (14..26).map(|n| 135.0 * (n as f64 + 1.45)).collect();
        let inertia: Vec<f64> = nu_mod.iter().map(|&f| 1.0e-7 * ((3100.0 - f) / 1000.0).exp()).collect();
        let sigma = vec![0.15; nu_mod.len()];
        let nu_obs: Vec<f64> = nu_mod.iter().zip(&inertia)
            .map(|(&f, &e)| f - 1.2 * (f / 3090.0).powi(3) / e)
            .collect();
        let (nu_corr, a3, a_m1) = eval_combined_corr(
            &nu_obs, &sigma, &nu_mod, &inertia, 1.0,
        );
        let chi2 = eval_seismic_chi2(&nu_obs, &sigma, &nu_corr);
        println!("a3 = {:.4e}, a_-1 = {:.4e}, Corrected chi^2 = {:.4e}", a3, a_m1, chi2);
    }
    ```
