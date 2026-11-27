# Opacity, Model Atmospheres & Synthetic Photometry (`kap`, `atm`, `colors`)

Radiation transport through a stellar interior and emergent flux at the stellar surface depend on three coupled modules: the Rosseland mean opacity (`flexmesa.kap`), the outer photospheric boundary condition (`flexmesa.atm`), and the synthetic spectral energy distribution and filter photometry (`flexmesa.colors`).

---

## 1. Radiative, Conductive & Compton Opacities (`flexmesa.kap`)

In the diffusion approximation for radiative energy transport, the radiative temperature gradient $\nabla_{\mathrm{rad}}$ across a shell of enclosed luminosity $L$ and pressure $P$ is proportional to the total Rosseland mean opacity $\kappa(\rho, T, \mathbf{X})$ ($\mathrm{cm^2\,g^{-1}}$):

$$\nabla_{\mathrm{rad}} = \frac{3}{16\pi a c G}\,\frac{\kappa\,L\,P}{m\,T^4}, \qquad \frac{1}{\kappa} = \frac{1}{\kappa_{\mathrm{rad}}} + \frac{1}{\kappa_{\mathrm{cond}}}.$$

`flexmesa.kap` (`crates/flexmesa-kap`) evaluates both radiative opacity $\kappa_{\mathrm{rad}}$ and degenerate electron thermal conduction opacity $\kappa_{\mathrm{cond}}$ alongside their exact logarithmic partial derivatives $(\partial \ln\kappa/\partial \ln\rho)_T$ and $(\partial \ln\kappa/\partial \ln T)_\rho$:

1. **High-Temperature Radiative Tables (Type 1 Fixed-Metal & Type 2 C/O-Enhanced)**:
   - `kap_get_fixed_metal` (Type 1) and `kap_get_co` (Type 2) interpolate 2D pre-validated bicubic Hermite splines (`interp_evbicub_prevalidated_db`) in $(\log_{10} R, \log_{10} T)$ where the rescaled density parameter is

     $$\log_{10} R = \log_{10}\rho - 3\log_{10} T + 18.$$

   - Across hydrogen mass fraction $X$ and metallicity $Z$, `FlexMESA` supports both linear and Steffen (1990) monotonic cubic (`interp_pm`) interpolation, paired with 2D biquadratic $(f_{\mathrm{C}}, f_{\mathrm{O}})$ carbon-oxygen enhancement subtables for helium-burning cores and white-dwarf interiors.
2. **Low-Temperature Molecular & Dust Opacities**:
   - In cool stellar atmospheres and red-giant envelopes ($\log_{10} T \lesssim 4.0$), radiative tables blend smoothly into Ferguson et al. (2005) (`lowT_fa05_gs98`, `gn93`, `a09p`), Freedman et al. (2008, 2014) (`lowT_Freedman11`), Lederer & Aringer (2009) C/N-varying `kapCN`, and Marigo & Aringer (2009) C/N/O-varying `AESOPUS` tables.
3. **Relativistic Compton Scattering & Degenerate Electron Conduction**:
   - Near the upper temperature or lower $\log_{10} R$ edges of the loaded Type 1 and Type 2 radiative tables ($\log_{10} T_{\mathrm{hi}} = \min(\log_{10} T_{\max,\mathrm{Type1}}, \log_{10} T_{\max,\mathrm{CO}}) - 0.01$, i.e. $8.69$ for `gs98`/`a09`/`OP_gs98` and $7.99$ for `OP_a09`), bound-free and free-free absorption give way to Compton scattering off partially degenerate, relativistic electrons (`kap/private/kap_eval.f90:287-318`). `compton_opacity_jax` evaluates the Poutanen (2017) relativistic Compton opacity and blends it with the radiative table via a $C^2$ quintic smootherstep over a $0.5\,\mathrm{dex}$ band (`get_kap_blend_bounds_jax`).
   - In degenerate cores, `combine_rad_with_conduction_jax` combines radiative opacity harmonically with Potekhin et al. (1999, `condint.f`) electron conduction tables and Blouin et al. (2020) H/He strong-coupling conductive corrections.
   - `make_differentiable_kap` wraps the full composite opacity evaluator in a `@jax.custom_jvp` rule so that `jax.grad` and `jax.jacfwd` propagate exact analytical $(\log_{10}\rho, \log_{10} T)$ spline derivatives without finite differences.

![Rosseland mean opacity regimes and basic.net nuclear energy generation across Safe Rust and JAX](../assets/figures/wave3b_microphysics_diagnostics.png)

*Figure 1: The left panel displays total Rosseland mean opacity $\log_{10}(\kappa / \mathrm{cm^2\,g^{-1}})$ as a function of $\log_{10}(T/\mathrm{K})$ at fixed $\log_{10} R = -3.0$ for solar composition ($X = 0.70, Z = 0.02$), comparing Safe Rust and JAX `custom_jvp` across the shaded low/high-$T$ transition band ($\log_{10} T \in [3.80, 3.88]$). The right panel shows `basic.net` nuclear energy generation $\log_{10}\epsilon_{\mathrm{nuc}}$ at $\rho = 100\,\mathrm{g\,cm^{-3}}$ (Safe Rust curve and JAX samples) alongside the shifted forward ${}^{12}\mathrm{C}(\alpha,\gamma){}^{16}\mathrm{O}$ REACLIB rate $\log_{10}\lambda + 22$.*

---

## 2. Stellar Model Atmospheres (`flexmesa.atm`)

Integrating the 1D stellar structure equations requires outer boundary values for the photospheric gas pressure $\ln P_{\mathrm{surf}}$ and temperature $\ln T_{\mathrm{surf}}$ at the surface optical depth $\tau_{\mathrm{surf}}$, together with their partial derivatives with respect to $(L, \ln R, \ln M, \ln \kappa)$. `flexmesa.atm` (`crates/flexmesa-atm`) provides three atmosphere boundary families:

1. **Analytic & Integrated Grey $T(\tau)$ Relations (`eval_t_tau_atm`, `eval_t_tau_uniform`, `eval_t_tau_varying`, `get_t_tau_id`)**:
   In a plane-parallel grey atmosphere in radiative equilibrium, temperature varies with optical depth $\tau$ according to a dimensionless Hopf function $q(\tau)$:

   $$T^4(\tau) = \frac{3}{4}\,T_{\mathrm{eff}}^4\,\left[\tau + q(\tau)\right], \qquad \frac{dP}{d\tau} = \frac{g}{\kappa(\rho, T)}.$$

   `flexmesa.atm` supports the classic Eddington ($q(\tau) = 2/3$), solar Hopf, Krishna-Swamy (1966), and Trampedach et al. (2014) / Ball (2021) radiation-hydrodynamics $T(\tau)$ relations (`get_t_tau_id`), solving for $P_{\mathrm{surf}}$ via `eval_t_tau_atm` (`star/private/atm_support.f90:218-343`) across `'fixed'` uniform opacity (`max_iters = 0`), `'iterated'` 0.5-underrelaxed fixed-point opacity iteration with radiation-pressure enhancement `pextra_factor` and implicit $(\chi_\rho, \chi_T, \kappa_\rho, \kappa_T)$ partial-derivative corrections, or `'varying'` adaptive Dormand-Prince 5(4) (`dopri5`, `num/private/mod_dopri5.f` and `atm/private/atm_t_tau_varying.f90:182-279`) integration of $dP/d\tau = g/\kappa$.
2. **Irradiated Planetary & Binary Atmospheres (`eval_irradiated`)**:
   Implements the Guillot (2010) semi-grey two-stream irradiated atmosphere model for close-in giant planets and strongly irradiated binary companions, using a precomputed Steffen (1990) monotonic spline (`eval_e2`) for the second exponential integral $E_2(x)$.
3. **Tabulated Model Atmospheres & Off-Table Blending (`eval_table_atm`, `eval_table`, `get_table_alfa_beta`, `get_table_id`)**:
   Performs 2D bicubic Hermite interpolation in $(\log g, T_{\mathrm{eff}})$ and 4-point 1D Steffen interpolation in metallicity $[\mathrm{M/H}]$ across all 7 MESA atmosphere table sets (`ATM_TABLE_PHOTOSPHERE`, `TAU_100`, `TAU_10`, `TAU_1`, `TAU_1M1`, `WD_TAU_25`, and `DB_WD_TAU_25`; resolved by name via `get_table_id`). When `get_table_alfa_beta` returns $\alpha \neq 0$, `&controls::atm_off_table_option` (`star/defaults/controls.defaults:4328-4335`, `star/private/atm_support.f90:604-650`) selects either `'T_tau'` (default, blending smoothly into the configured $T(\tau)$ relation via `eval_table_atm` / `get_table` and switching $\tau_{\mathrm{base}}$ when $\beta = 0$) or `''` (halting when stepping outside the table or into the blend margin; case-sensitive, with other values rejected loudly); when $\alpha = 0$, `atm_off_table_option` is not checked.

---

## 3. Synthetic Spectra & Filter Photometry (`flexmesa.colors`)

To compare theoretical evolutionary tracks directly against observed color-magnitude diagrams without intermediate post-processing tools, `flexmesa.colors` (`crates/flexmesa-colors`) synthesizes emergent stellar spectral energy distributions (SEDs), bolometric magnitudes $M_{\mathrm{bol}}$, bolometric fluxes $F_{\mathrm{bol}}$, normalized grid interpolation distances `Interp_rad`, and synthetic magnitudes across the 7 Johnson photometric passbands (`U, B, V, R, I, J, M`, 10 output columns total) in Vega, AB, and ST zero-point systems from the Kurucz (2003, `lcb98cor.dat`) stellar atmosphere atlas ($76 \times 11 \times 8$ grid over $T_{\mathrm{eff}} \in [3500, 50000]\,\mathrm{K}$, $\log g \in [0.0, 5.0]$, $[\mathrm{M/H}] \in [-5.0, 1.0]$ at $N_\lambda = 1199$ wavelengths).

### 3.1 32-Gather Deduplicated 3D Cubic Hermite SED Synthesis
Upstream Fortran MESA evaluates each stellar SED by performing a 64-point (`4 x 4 x 4`) 3D cubic Hermite interpolation at every wavelength $\lambda_m$. Because the three 1D Catmull-Rom finite-difference tangent operators share the 8 interior cell corners $(i_x + k_x, i_y + k_y, i_z + k_z)$ for $k_x, k_y, k_z \in \{0, 1\}$, `FlexMESA` factors the interior corner value weight $w_{\mathrm{val}}$ and the three tangent weights $(w_{dx}, w_{dy}, w_{dz})$ into a single combined weight:

$$w_{\mathrm{corner}}(k_x, k_y, k_z) = h_{00}(t_x)\text{ or }h_{01}(t_x)\;\cdots\; + w_{dx} + w_{dy} + w_{dz}.$$

Factoring the shared interior stencil nodes reduces memory gathers per stellar model from 56 to **32 distinct grid spectra** (a 43% reduction in memory bandwidth) while remaining internally consistent with 64-point cubic evaluation to $3.57 \times 10^{-16}$ (MESA `26.4.1` hex vector promotion pending `#161`).

### 3.2 Precomputed Non-Uniform Simpson Panel Weights
For each filter transmission curve $S_k(\lambda)$ resampled onto the 1199-point Kurucz wavelength grid, the photon-counting filter flux integral and energy-counting bolometric integral are integrated via non-uniform quadratic Simpson panels over $(x_{2p}, x_{2p+1}, x_{2p+2})$ with step widths $h_0 = x_{2p+1} - x_{2p}$ and $h_1 = x_{2p+2} - x_{2p+1}$:

$$I = \sum_{p=0}^{598} \left(c_{0,p} y_{2p} + c_{1,p} y_{2p+1} + c_{2,p} y_{2p+2}\right), \qquad \begin{cases} c_{0,p} = \dfrac{h_0 + h_1}{6 h_0}(2 h_0 - h_1), \\[6pt] c_{1,p} = \dfrac{(h_0 + h_1)^3}{6 h_0 h_1}, \\[6pt] c_{2,p} = \dfrac{h_0 + h_1}{6 h_1}(2 h_1 - h_0). \end{cases}$$

Precomputing $(c_{0,p}, c_{1,p}, c_{2,p})$ and the wavelength-weighted filter vectors $\lambda_m S_k(\lambda_m)$ once at table initialization avoids per-zone filter weight reconstruction over the `[T; 1199]` buffer (see [Verification & Benchmarks Overview](../benchmarks/overview.md) and `#146` for pre-audit timing notes and upcoming MESA `26.4.1` benchmarks).

![Kurucz emergent SEDs with Johnson B, V, R passbands and synthetic B-V colors](../assets/figures/colors_kurucz_sed_johnson.png)

*Figure 2: The left panel shows emergent Kurucz (2003) stellar spectral energy distributions at $T_{\mathrm{eff}} = 5920\,\mathrm{K}$ ($\log g = 4.30$, $[\mathrm{M/H}] = 0.0$) and $T_{\mathrm{eff}} = 5770\,\mathrm{K}$ ($\log g = 4.10$, $[\mathrm{M/H}] = -0.4$) overlaid with scaled Johnson $B, V, R$ filter transmission curves. The right panel displays the resulting synthetic $(B - V)$ color index across $T_{\mathrm{eff}} \in [5760, 5990]\,\mathrm{K}$ at fixed $\log g = 4.25$ and $[\mathrm{M/H}] \in \{0.0, -0.4\}$ comparing Safe Rust and JAX `@jit`.*

=== "Python & JAX (`flexmesa.kap`, `flexmesa.atm`, `flexmesa.colors`)"

    ```python
    import jax
    import jax.numpy as jnp
    from flexmesa import atm, colors, constants, kap

    # 1. Differentiable opacity evaluation with @jax.custom_jvp
    kap_fn = kap.make_differentiable_kap(z=0.02, x=0.70, zbase=0.02, zbar=1.28)
    dlogkap_dlogrho, dlogkap_dlogt = jax.grad(kap_fn, argnums=(0, 1))(
        jnp.float64(-5.0), jnp.float64(6.0)
    )

    # 2. Tabulated photosphere boundary condition (tau = 2/3)
    surf = atm.eval_table(
        luminosity=constants.LSUN,
        r=constants.RSUN,
        m=constants.MSUN,
        cgrav=constants.STANDARD_CGRAV,
        table_id=atm.ATM_TABLE_PHOTOSPHERE,
        z=0.0179,
        skip_partials=False,
        teff=5777.0,
    )

    # 3. Synthetic Johnson UBVRIJM photometry for a solar-type star
    phot = colors.eval_colors(
        teff=5777.0, log_g=4.44, metallicity=0.0, r=constants.RSUN
    )
    ```

=== "Safe Rust (`crates/flexmesa-kap`, `crates/flexmesa-atm`)"

    ```rust
    use flexmesa_atm::{ATM_TABLE_PHOTOSPHERE, eval_table};
    use flexmesa_const::{CLIGHT, CRAD, MSUN, PI, RSUN, STANDARD_CGRAV};
    use flexmesa_kap::{KapConfig, get_kap_results};

    // 1. Evaluate Type 1 + conductive + Compton opacity
    let kap_out = get_kap_results(
        &KapConfig::default(), 1.28, 0.70, 0.02, 0.004, 0.001, 0.009, 0.003,
        -5.0, 6.0, -0.16, 0.0, 0.0, -10.0, 1.0, -1.5,
    ).unwrap();

    // 2. Evaluate photosphere table boundary condition
    let l_sun = PI * CRAD * CLIGHT * RSUN * RSUN * 5777.0_f64.powi(4);
    let atm_out = eval_table(
        l_sun, RSUN, MSUN, STANDARD_CGRAV, ATM_TABLE_PHOTOSPHERE,
        0.0179, false, 5777.0,
    ).unwrap();
    ```
