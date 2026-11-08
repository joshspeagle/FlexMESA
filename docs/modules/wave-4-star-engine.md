# Wave 4 1D Stellar Structure & Evolution Engine (`star_data` & `star`)

`flexmesa.star` (`crates/flexmesa-star` + `src/flexmesa/star/`) implements the 1D Lagrangian hydrostatic and hydrodynamic stellar evolution engine, integrating every microphysics and numerical solver package from Waves 1 through 3b (`const`, `auto_diff`, `interp_1d`, `interp_2d`, `mtx`, `num`, `neu`, `turb`, `chem`, `ionization`, `colors`, `eos`, `kap`, `rates`, `net`, and `atm`).

![Wave 4 internal convective structure of a 4 Msun ZAMS star and 8-mass ZAMS HR diagram](../assets/figures/wave4_star_structure_evolution.png)

---

## 1. Zero-Age Main Sequence (`ZAMS`) & `.mod` Model Initialization

`StarModel` represents a 1D spherical star discretized into $\mathrm{nz}$ Lagrangian mass shells from the outer photosphere ($k = 0$) to the stellar center ($k = \mathrm{nz} - 1$). Models can be initialized in four ways:

1. **Built-In ZAMS Catalog (`StarModel.from_zams`)**: Loads or interpolates a $Z = 0.02$ Zero-Age Main Sequence model for any stellar mass $M \in [0.51, 15.0]\,M_\odot$ (`0.51, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0 Msun` anchor models).
2. **Upstream Fortran MESA `.mod` Text Files (`StarModel.from_mod_str`)**: Parses and validates MESA ASCII `.mod` files (`lnd`, `lnT`, `lnR`, `L`, `dq`, and species mass fractions `xa`).
3. **Analytic Polytropes (`StarModel.from_pre_ms`)**: Integrates the $n = 1.5$ convective pre-main-sequence Lane-Emden polytrope and maps it onto the requested $(M_\star, T_c, Z)$ configuration.
4. **Custom Primary Arrays (`StarModel.set_primary_state`)**: Directly populates $(\ln\rho_k, \ln T_k, \ln r_k, L_k, \Delta q_k, X_{j,k})$ from NumPy arrays.

=== "Python & JAX (`flexmesa.star`)"

    ```python
    import jax
    import jax.numpy as jnp
    from flexmesa import star

    # Load the canonical 4.0 Msun, Z = 0.02 ZAMS model (star/test)
    model = star.StarModel.from_zams(initial_mass=4.0, initial_z=0.02)
    print(f"Zones: {model.nz}, Species: {model.iso_names}")

    # Evolve to 1.0e5 yr using coupled Henyey + Hirshman bcyclic solver
    reports = model.evolve(
        max_steps=20,
        do_remesh=False,
        op_split_burn=False,
        use_bcyclic=True,
        max_age=1.0e5,
    )
    row = model.history_row()
    print(
        f"Age = {row.star_age:.2e} yr, log10(Tc) = {row.log_center_t:.4f}, "
        f"log10(L/Lsun) = {row.log_l:.4f}, H_cntr = {row.center_h1:.6f}"
    )
    ```

=== "Safe Rust (`crates/flexmesa-star`)"

    ```rust
    use flexmesa_star::{
        StarControls, StarJobControls, extract_history_row, load_zams_model,
        star_evolve,
    };

    fn main() {
        let controls = StarControls {
            initial_mass: 4.0,
            initial_z: 0.02,
            max_age: Some(1.0e5),
            ..StarControls::default()
        };
        let mut model =
            load_zams_model(controls, StarJobControls::default()).unwrap();
        let reports =
            star_evolve(&mut model, 20, false, false, true).unwrap();
        let row = extract_history_row(&model);
        println!(
            "Steps = {}, log10(Tc) = {:.4}, log10(L/Lsun) = {:.4}",
            reports.len(),
            row.log_center_t,
            row.log_l
        );
    }
    ```

---

## 2. Differentiable Face Microphysics, Coupled Henyey Residuals & Linear Solvers

Every zone and face evaluation in `flexmesa.star` is paired with a pure `@jax.jit` `float64` counterpart:

- **`eval_face_mlt_and_brunt_jax`**: Evaluates face radiative gradients $\nabla_{\mathrm{rad},k}$, Ledoux composition gradients $B_k$, Cox (1968) mixing-length cubic roots $\nabla_{T,k}$, convective velocities $v_{\mathrm{conv},k}$, Eulerian diffusion coefficients $D_{\mathrm{mix},k}$, and Brunt-Väisälä frequencies $N^2_k = N^2_{\mathrm{struc},k} + N^2_{\mathrm{comp},k}$.
- **`eval_hydro_residuals_jax` & `eval_hydro_system_jax`**: Evaluates the coupled 4-equation hydrostatic/thermal residual vector `(nz, 4)` and extracts the exact block-tridiagonal Jacobian `(lblk, dblk, ublk)` via `jax.jacfwd`.
- **`solve_henyey_linear_jax`**: Solves the block-tridiagonal Newton step $\mathbf{J}\,\delta\tilde{\mathbf{x}} = -\mathbf{F}$ using either native $\lceil \log_2 N_{\mathrm{zones}} \rceil$ Parallel Cyclic Reduction (`solver_mode="pcr"`) or Safe Rust `bcyclic` FFI equipped with an analytical Implicit Function Theorem `jax.custom_vjp` rule (`solver_mode="rust_ffi"`).

```python
import jax
import jax.numpy as jnp
from flexmesa import star

model = star.StarModel.from_zams(4.0, 0.02)
sys = model.eval_equations(include_chem=False)

# Differentiate a scalar objective of the Newton correction w.r.t. the residual vector
def loss_fn(equ_j: jax.Array) -> jax.Array:
    dx = star.solve_henyey_linear_jax(
        jnp.asarray(sys.lblk),
        jnp.asarray(sys.dblk),
        jnp.asarray(sys.ublk),
        equ_j,
        solver_mode="rust_ffi",
    )
    return jnp.sum(dx**2)

grad_equ = jax.grad(loss_fn)(jnp.asarray(sys.equ))
```

---

## 3. Convective Overshooting, Stellar Winds, Adaptive Remeshing & Timestep Control

- **Convective Boundaries & Overshooting (`OvershootConfig`)**: `model.locate_convection_boundaries()` detects exact sub-cell Schwarzschild/Ledoux sign crossings and supports step or exponential diffusive overshooting above and below hydrogen-, helium-, and metal-burning zones.
- **Mass-Loss Winds (`WindConfig`, `eval_wind_mdot`)**: Evaluates Reimers (1975), Blöcker (1995), Vink et al. (2001), de Jager et al. (1988), Nieuwenhuijzen & de Jager (1990), and composite `Dutch` wind mass-loss rates $\dot{M}_{\mathrm{wind}} \le 0$ ($\mathrm{g\,s^{-1}}$) with proportional super-Eddington enhancement.
- **Adaptive Mesh Adjustment (`plan_mesh_adjust`, `remesh`)**: Computes the monotonically increasing mesh spacing function $g_k$ (`eval_mesh_gval` / `eval_mesh_gval_jax`), plans cell splits and merges, and conservatively remaps mass fractions and thermodynamic state across the updated grid (`interp_q`).
- **H211 Digital Timestep Controller (`eval_varcontrol`, `eval_h211_timestep`, `propose_next_timestep`)**: Combines the 5-point smoothed relative variation norm `varcontrol`, nuclear fuel depletion limits, and the Söderlind & Wang (2006) second-order digital filter (`eval_h211_timestep_jax`).
