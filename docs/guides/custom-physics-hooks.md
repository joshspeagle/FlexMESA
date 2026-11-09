# User Guide: Custom Physics Hooks & Replacing `run_star_extras.f90`

In Fortran MESA, customizing a simulation (for example, stopping at a user-defined core helium fraction, rescaling an individual nuclear reaction rate, modifying convective mixing diffusivities, or coupling an external orbital/seismic optimizer) requires editing `src/run_star_extras.f90` or `src/run_binary_extras.f90`, assigning procedure pointers (`s% other_wind => ...`, `s% extras_finish_step => ...`), and recompiling the Fortran executable.

In `FlexMESA`, the evolution loop and microphysics evaluators are ordinary Python and Rust functions operating on inspectable `StarModel` and `BinarySystem` objects. You never recompile the library to inject custom physics or stopping criteria.

---

## 1. Replacing `extras_check_model` and `extras_finish_step`

In Fortran MESA's `run_star_extras.f90`, users implement `extras_check_model`, `extras_finish_step`, and `how_many_extra_history_columns` to monitor custom diagnostics and trigger termination. In `FlexMESA`, you write a standard Python evolution loop around `model.evolve_step`, inspect any 1D profile array (`model.t`, `model.rho`, `model.xa`, `model.brunt_n2`) or scalar `model.history_row()`, and break or adjust controls dynamically:

```python
import numpy as np
from flexmesa import constants, star

model = star.StarModel.from_zams(initial_mass=2.0, initial_z=0.02)
model.set_evolve_controls(mesh_delta_coeff=1.0, varcontrol_target=1.0e-4)

extra_history = []
for step in range(1, 10):
    report = model.evolve_step(do_remesh=False, use_bcyclic=True)
    row = model.history_row()

    # Custom diagnostic: mass-weighted mean sound speed and convective core mass
    dm = np.asarray(model.dm)
    cs_mean = float(np.sum(dm * np.asarray(model.csound)) / model.mstar)
    conv_mask = np.asarray(model.gradr) > np.asarray(model.grada)
    m_conv_core_msun = (
        float(np.asarray(model.m)[conv_mask].max() / constants.MSUN)
        if np.any(conv_mask)
        else 0.0
    )
    extra_history.append(
        {
            "step": step,
            "age_yr": row.star_age,
            "cs_mean_cm_s": cs_mean,
            "m_conv_core_msun": m_conv_core_msun,
        }
    )

    # Custom stopping condition: stop once central hydrogen drops by 5e-5
    if row.center_h1 < 0.70 - 5.0e-5:
        print(
            f"Custom termination triggered at step {step}: "
            f"age={row.star_age:.3e} yr, X_c={row.center_h1:.6f}, "
            f"M_cc={m_conv_core_msun:.3f} Msun"
        )
        break
```

---

## 2. Custom Nuclear Reaction Rate Multipliers (`rate_factors` & `weak_rate_factor`)

Nuclear astrophysics sensitivity studies frequently vary an individual bottleneck reaction rate (such as ${}^{12}\mathrm{C}(\alpha,\gamma){}^{16}\mathrm{O}$ or ${}^{14}\mathrm{N}(p,\gamma){}^{15}\mathrm{O}$) or scale all weak rates across a grid of factors. Both `net.eval_net` and the stiff 1-zone burners (`net.burn_1_zone`, `net.burn_1_zone_const_p`) accept a per-reaction multiplier array `rate_factors` of length `net_info.num_reactions` as well as a scalar `weak_rate_factor`:

```python
import numpy as np
from flexmesa import chem, net

net_info = net.get_net_info("basic.net")
print("Reactions in basic.net:", net_info.reaction_names)

# Locate the index of the triple-alpha (r_he4_he4_he4_to_c12) and 12C(alpha,gamma)16O reactions
idx_cag = net_info.reaction_names.index("r_c12_ag_o16")

xa = np.array([0.0, 0.0, 0.50, 0.25, 0.0, 0.25, 0.0, 0.0], dtype=np.float64)
comp = chem.get_composition_info(net_info.chem_id, xa)

for scale_cag in (0.5, 1.0, 2.0):
    factors = np.ones(net_info.num_reactions, dtype=np.float64)
    factors[idx_cag] = scale_cag

    out = net.eval_net(
        "basic.net",
        x=xa,
        temp=2.2e8,
        rho=1.0e4,
        abar=float(comp.abar),
        zbar=float(comp.zbar),
        z2bar=float(comp.z2bar),
        ye=float(comp.ye),
        rate_factors=factors,
    )
    print(
        f"12C(a,g)16O factor = {scale_cag:.1f}x: "
        f"eps_nuc = {out.eps_nuc:.4e} erg/g/s, dX(16O)/dt = {out.dxdt[5]:.4e} s^-1"
    )
```

---

## 3. Customizing State Profiles, Mixing & Step Sub-Stages

When you need finer control than `model.evolve_step` (equivalent to Fortran MESA's `other_mlt`, `other_D_mix`, or `other_energy` hooks), `StarModel` exposes each sub-stage of the evolutionary step directly:

1. `model.save_old_values()` and `model.save_start_values()`: Snapshot the current state for timestep control and backward-Euler time derivatives.
2. `model.update_mixing(dt, overshoot_f=..., overshoot_f0=..., scheme="exponential")`: Recompute convective boundaries and overshooting diffusivities.
3. `model.do_conv_premix(max_iters=10)` and `model.add_predictive_mixing(...)`: Apply convective premixing (`CPM`) or predictive mixing at core boundaries.
4. `model.step_struct_burn_mix(dt, op_split_burn=False, use_bcyclic=True, overshoot_f=0.014, overshoot_f0=0.004)`: Execute a single coupled Henyey structure + burning + mixing solve for a prescribed timestep $\Delta t$ (`s`) and return the `HenyeySolverReport`.
5. `model.set_primary_state(lnd, ln_t, ln_r, lum, dq, xa)` and `model.eval_hydro_vars()`: Inject a custom modified state profile and refresh all derived thermodynamic, opacity, and nuclear arrays.

```python
from flexmesa import star

model = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)

# Save start-of-step state and apply custom exponential core overshooting (f = 0.016)
model.save_old_values()
model.save_start_values()
n_conv_bdy = model.update_mixing(
    dt=1.0e5 * 3.15576e7,
    overshoot_f=0.016,
    overshoot_f0=0.004,
    scheme="exponential",
)
henyey_rep = model.step_struct_burn_mix(
    dt=1.0e5 * 3.15576e7,
    op_split_burn=False,
    use_bcyclic=True,
    overshoot_f=0.016,
    overshoot_f0=0.004,
)
print(
    f"Custom step converged={henyey_rep.converged} in {henyey_rep.iterations} Newton iterations "
    f"(max_res={henyey_rep.max_residual:.2e}, conv boundaries={n_conv_bdy})"
)
```
