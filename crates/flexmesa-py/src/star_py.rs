// SPDX-License-Identifier: LGPL-3.0-or-later
//! `PyO3` bindings for `flexmesa-star` (`RustStarModel`, ZAMS/pre-MS builders,
//! `.mod`/`.photo` I/O, parallel zone/face microphysics evaluation, coupled 1D
//! Henyey solver, convective boundary overshooting, stellar winds, adaptive
//! remeshing, H211 timestep control, and the evolution driver).

use flexmesa_star::{
    apply_overshooting, build_pre_ms_model, clean_mixing_glitches,
    compute_convection_sigmas, do_adjust_mass, do_adjust_mesh,
    do_struct_burn_mix, eval_dxdt_mix, eval_henyey_equations,
    extract_history_row, format_history_row, format_profile_string, interp_q,
    load_zams_model, locate_convection_boundaries, pick_next_timestep,
    read_mod_str, read_photo_bytes, set_hydro_vars, set_mdot,
    solve_composition_diffusion, solve_henyey_step, solve_omega_mix,
    star_evolve, star_evolve_step, write_mod_string, write_photo_bytes,
    yrs_for_init_timestep, DutchLowTScheme, EvolveStepReport,
    HenyeySolverReport, MeshFunctionConfig, OvershootConfig, OvershootScheme,
    StarControls, StarError, StarJobControls, StarModel, WindScheme,
};
use numpy::{
    PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArray2,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

type HenyeyEquationsPyTuple<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray2<f64>>,
);
type SolverReportTuple = (bool, usize, f64, f64, f64, f64);
type EvolveStepTuple =
    (usize, f64, f64, f64, usize, usize, usize, usize, f64, bool);
type HistoryTuple = ((usize, usize, usize, usize), [f64; 13]);

fn to_py_err(e: StarError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

const fn report_to_tuple(r: HenyeySolverReport) -> SolverReportTuple {
    (
        r.converged,
        r.iterations,
        r.residual_norm,
        r.max_residual,
        r.correction_norm,
        r.max_correction,
    )
}

const fn evolve_to_tuple(r: EvolveStepReport) -> EvolveStepTuple {
    (
        r.model_number,
        r.star_age,
        r.dt,
        r.dt_next,
        r.iterations,
        r.retries,
        r.num_splits,
        r.num_merges,
        r.varcontrol,
        r.converged,
    )
}

fn build_overshoot_configs(
    f: f64,
    f0: f64,
    scheme: &str,
) -> Vec<OvershootConfig> {
    if f <= 0.0 {
        return Vec::new();
    }
    let s = if scheme.eq_ignore_ascii_case("step") {
        OvershootScheme::Step
    } else {
        OvershootScheme::Exponential
    };
    let d = OvershootConfig::default();
    vec![
        OvershootConfig {
            scheme: s,
            top_conv_bdy: true,
            f,
            f0,
            ..d
        },
        OvershootConfig {
            scheme: s,
            top_conv_bdy: false,
            f,
            f0,
            ..d
        },
    ]
}

fn parse_wind_scheme(scheme: &str) -> PyResult<WindScheme> {
    match scheme.to_ascii_lowercase().as_str() {
        "" | "none" => Ok(WindScheme::None),
        "reimers" => Ok(WindScheme::Reimers),
        "blocker" => Ok(WindScheme::Blocker),
        "vink" => Ok(WindScheme::Vink),
        "nieuwenhuijzen" => Ok(WindScheme::Nieuwenhuijzen),
        "de jager" | "de_jager" | "dejager" => Ok(WindScheme::DeJager),
        "van loon" | "van_loon" | "vanloon" => Ok(WindScheme::VanLoon),
        "dutch" => Ok(WindScheme::Dutch),
        s => Err(PyValueError::new_err(format!("unknown wind scheme: {s}"))),
    }
}

/// Python-exposed wrapper around [`flexmesa_star::StarModel`].
#[pyclass(name = "RustStarModel")]
#[derive(Clone)]
pub struct PyStarModel {
    /// Inner canonical Rust stellar model.
    pub inner: StarModel,
}

macro_rules! impl_py_star_model {
    (
        scalars: [ $( $sget:ident : $sty:ty ),+ $(,)? ],
        arrays: [ $( $aget:ident ),+ $(,)? ]
    ) => {
        #[pymethods]
        impl PyStarModel {
            /// Loads a ZAMS stellar model from the built-in catalog.
            #[staticmethod]
            #[pyo3(signature = (
                initial_mass=1.0, initial_z=0.02, initial_y=-1.0
            ))]
            pub fn from_zams(
                initial_mass: f64,
                initial_z: f64,
                initial_y: f64,
            ) -> PyResult<Self> {
                let c = StarControls {
                    initial_mass,
                    initial_z,
                    initial_y,
                    zbase: initial_z,
                    ..StarControls::default()
                };
                let inner = load_zams_model(c, StarJobControls::default())
                    .map_err(to_py_err)?;
                Ok(Self { inner })
            }

            /// Constructs an $n = 1.5$ pre-main-sequence polytrope model.
            #[staticmethod]
            #[pyo3(signature = (
                initial_mass=1.0, initial_z=0.02, initial_y=-1.0,
                t_c=9.0e5, d_log10_p=0.02
            ))]
            pub fn from_pre_ms(
                initial_mass: f64,
                initial_z: f64,
                initial_y: f64,
                t_c: f64,
                d_log10_p: f64,
            ) -> PyResult<Self> {
                let c = StarControls {
                    initial_mass,
                    initial_z,
                    initial_y,
                    zbase: initial_z,
                    ..StarControls::default()
                };
                let j = StarJobControls {
                    create_pre_ms_model: true,
                    pre_ms_t_c: t_c,
                    pre_ms_d_log10_p: d_log10_p,
                    ..StarJobControls::default()
                };
                Ok(Self {
                    inner: build_pre_ms_model(c, j).map_err(to_py_err)?,
                })
            }

            /// Parses an ASCII `.mod` model string.
            #[staticmethod]
            pub fn from_mod_str(text: &str) -> PyResult<Self> {
                let (c, j) =
                    (StarControls::default(), StarJobControls::default());
                Ok(Self {
                    inner: read_mod_str(text, c, j).map_err(to_py_err)?,
                })
            }

            /// Restores a [`PyStarModel`] from binary `.photo` bytes.
            #[staticmethod]
            pub fn from_photo_bytes(bytes: &[u8]) -> PyResult<Self> {
                let (c, j) =
                    (StarControls::default(), StarJobControls::default());
                Ok(Self {
                    inner: read_photo_bytes(bytes, c, j).map_err(to_py_err)?,
                })
            }

            /// Serializes the model into an ASCII `.mod` string.
            pub fn to_mod_str(&self) -> PyResult<String> {
                write_mod_string(&self.inner).map_err(to_py_err)
            }

            /// Serializes the model into binary `.photo` checkpoint bytes.
            pub fn to_photo_bytes<'py>(
                &self,
                py: Python<'py>,
            ) -> Bound<'py, PyBytes> {
                PyBytes::new(py, &write_photo_bytes(&self.inner))
            }

            /// Evaluates zone and face microphysics (`set_hydro_vars`).
            #[pyo3(signature = (skip_net=false, skip_net_jacobian=false))]
            pub fn eval_hydro_vars(
                &mut self,
                py: Python<'_>,
                skip_net: bool,
                skip_net_jacobian: bool,
            ) -> PyResult<()> {
                let inner = &mut self.inner;
                py.detach(|| {
                    set_hydro_vars(inner, skip_net, skip_net_jacobian)
                })
                .map_err(to_py_err)
            }

            /// Replaces primary state arrays and updates staggered geometry.
            pub fn set_primary_state(
                &mut self,
                lnd: PyReadonlyArray1<'_, f64>,
                ln_t: PyReadonlyArray1<'_, f64>,
                ln_r: PyReadonlyArray1<'_, f64>,
                lum: PyReadonlyArray1<'_, f64>,
                dq: PyReadonlyArray1<'_, f64>,
                xa: PyReadonlyArray2<'_, f64>,
            ) -> PyResult<()> {
                let (lnd_s, lnt_s, lnr_s, lum_s, dq_s, xa_s) = (
                    lnd.as_slice()?,
                    ln_t.as_slice()?,
                    ln_r.as_slice()?,
                    lum.as_slice()?,
                    dq.as_slice()?,
                    xa.as_slice()?,
                );
                let (nz, sp) = (lnd_s.len(), self.inner.species);
                if lnt_s.len() != nz
                    || lnr_s.len() != nz
                    || lum_s.len() != nz
                    || dq_s.len() != nz
                    || xa_s.len() != nz * sp
                {
                    return Err(PyValueError::new_err(
                        "mismatched array lengths in set_primary_state",
                    ));
                }
                self.inner.lnd.copy_from_slice(lnd_s);
                self.inner.ln_t.copy_from_slice(lnt_s);
                self.inner.ln_r.copy_from_slice(lnr_s);
                self.inner.lum.copy_from_slice(lum_s);
                self.inner.dq.copy_from_slice(dq_s);
                self.inner.xa.copy_from_slice(xa_s);
                self.inner.sync_mesh_and_pack().map_err(to_py_err)
            }

            /// Sets the current and proposed next timestep $\Delta t$ in seconds.
            #[setter]
            pub fn set_dt(&mut self, dt: f64) {
                self.inner.dt = dt;
                self.inner.dt_next = dt;
            }

            /// Sets `mesh_delta_coeff` and `varcontrol_target` on `controls`.
            #[pyo3(signature = (
                mesh_delta_coeff=1.0, varcontrol_target=1.0e-4
            ))]
            pub fn set_evolve_controls(
                &mut self,
                mesh_delta_coeff: f64,
                varcontrol_target: f64,
            ) {
                self.inner.controls.mesh_delta_coeff = mesh_delta_coeff;
                self.inner.controls.varcontrol_target = varcontrol_target;
            }

            /// Snapshots current state arrays as start-of-step `*_start` values.
            pub fn save_start_values(&mut self) {
                self.inner.save_start_values();
            }

            /// Snapshots current state arrays as previous-step `*_old` values.
            pub fn save_old_values(&mut self) {
                self.inner.save_old_values();
            }

            /// Evaluates the coupled 1D Henyey residual vector and scaled
            /// block-tridiagonal Jacobian `(equ, lblk, dblk, ublk, x_scale)`.
            #[pyo3(signature = (include_chem=false))]
            pub fn eval_equations<'py>(
                &mut self,
                py: Python<'py>,
                include_chem: bool,
            ) -> PyResult<HenyeyEquationsPyTuple<'py>> {
                let inner = &mut self.inner;
                let sys = py
                    .detach(|| eval_henyey_equations(inner, include_chem))
                    .map_err(to_py_err)?;
                let (nz, nv) = (sys.nz, sys.nvar);
                Ok((
                    PyArray1::from_vec(py, sys.equ).reshape([nz, nv])?,
                    PyArray1::from_vec(py, sys.lblk).reshape([nz, nv, nv])?,
                    PyArray1::from_vec(py, sys.dblk).reshape([nz, nv, nv])?,
                    PyArray1::from_vec(py, sys.ublk).reshape([nz, nv, nv])?,
                    PyArray1::from_vec(py, sys.x_scale).reshape([nz, nv])?,
                ))
            }

            /// Executes a damped Newton-Raphson Henyey step.
            #[pyo3(signature = (include_chem=false, use_bcyclic=true))]
            pub fn solve_henyey(
                &mut self,
                py: Python<'_>,
                include_chem: bool,
                use_bcyclic: bool,
            ) -> PyResult<SolverReportTuple> {
                let inner = &mut self.inner;
                let rep = py
                    .detach(|| {
                        solve_henyey_step(inner, include_chem, use_bcyclic)
                    })
                    .map_err(to_py_err)?;
                Ok(report_to_tuple(rep))
            }

            /// Surface $\ln P_{\mathrm{surf}}$ from atmosphere BC.
            #[getter]
            pub const fn ln_peos_surf(&self) -> f64 {
                self.inner.ln_peos_surf_ad.val
            }

            /// Surface $\ln T_{\mathrm{surf}}$ from atmosphere BC.
            #[getter]
            pub const fn ln_t_surf(&self) -> f64 {
                self.inner.ln_t_surf_ad.val
            }

            /// Active isotope names.
            #[getter]
            pub fn iso_names(&self) -> Vec<String> {
                self.inner.iso_names.clone()
            }

            /// Active 1-based `chem_id` indices.
            #[getter]
            pub fn chem_id(&self) -> Vec<usize> {
                self.inner.chem_id.clone()
            }

            /// Cell species mass fractions `(nz, species)`.
            #[getter]
            pub fn xa<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                PyArray1::from_slice(py, &self.inner.xa)
                    .reshape([self.inner.nz, self.inner.species])
            }

            /// Cell gravitational energy rate $\epsilon_{\mathrm{grav}}$.
            #[getter]
            pub fn eps_grav<'py>(
                &self,
                py: Python<'py>,
            ) -> Bound<'py, PyArray1<f64>> {
                let vals: Vec<f64> =
                    self.inner.eps_grav_ad.iter().map(|a| a.val).collect();
                PyArray1::from_vec(py, vals)
            }

            /// Cell nuclear abundance rate $\dot{X}_{j,k}$ `(nz, species)`.
            #[getter]
            pub fn dxdt_nuc<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                PyArray1::from_slice(py, &self.inner.dxdt_nuc)
                    .reshape([self.inner.nz, self.inner.species])
            }

            /// Face integer mixing classification codes (`0 = NoMixing`, etc.).
            #[getter]
            pub fn mixing_type<'py>(
                &self,
                py: Python<'py>,
            ) -> Bound<'py, PyArray1<i32>> {
                let codes: Vec<i32> =
                    self.inner.mixing_type.iter().map(|m| m.code()).collect();
                PyArray1::from_vec(py, codes)
            }

            /// Cleans mixing glitches, locates convective boundaries, applies
            /// overshooting, and updates `sig`, returning `num_conv_boundaries`.
            #[pyo3(signature = (
                dt, overshoot_f=0.0, overshoot_f0=0.004, scheme="exponential"
            ))]
            pub fn update_mixing(
                &mut self,
                dt: f64,
                overshoot_f: f64,
                overshoot_f0: f64,
                scheme: &str,
            ) -> usize {
                clean_mixing_glitches(&mut self.inner, 1.0, 0.1);
                let (_cz_dq, bdy) = locate_convection_boundaries(&self.inner);
                let cfgs =
                    build_overshoot_configs(overshoot_f, overshoot_f0, scheme);
                if !cfgs.is_empty() {
                    apply_overshooting(&mut self.inner, &bdy, &cfgs);
                }
                compute_convection_sigmas(&mut self.inner, dt, 1.0, 1.0e10);
                bdy.len()
            }

            /// Explicit mass-conserving diffusive species mixing rate.
            pub fn eval_dxdt_mix<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                PyArray1::from_vec(py, eval_dxdt_mix(&self.inner))
                    .reshape([self.inner.nz, self.inner.species])
            }

            /// Implicit backward-Euler species diffusion over `dt` (s).
            pub fn solve_diffusion(&mut self, dt: f64) {
                solve_composition_diffusion(&mut self.inner, dt);
            }

            /// Substepped tridiagonal rotational angular momentum diffusion.
            #[pyo3(signature = (dt, omega, nu_omega, n_substeps=4))]
            pub fn run_omega_mix(
                &mut self,
                dt: f64,
                omega: PyReadonlyArray1<'_, f64>,
                nu_omega: PyReadonlyArray1<'_, f64>,
                n_substeps: usize,
            ) -> PyResult<()> {
                let (w_s, nu_s) = (omega.as_slice()?, nu_omega.as_slice()?);
                if w_s.len() != self.inner.nz || nu_s.len() != self.inner.nz {
                    return Err(PyValueError::new_err(
                        "omega and nu_omega must have length nz",
                    ));
                }
                self.inner.omega.copy_from_slice(w_s);
                solve_omega_mix(&mut self.inner, dt, nu_s, n_substeps);
                Ok(())
            }

            /// Executes `do_struct_burn_mix` (coupled or operator-split).
            #[pyo3(signature = (
                dt, op_split_burn=false, use_bcyclic=true,
                overshoot_f=0.0, overshoot_f0=0.004
            ))]
            pub fn step_struct_burn_mix(
                &mut self,
                py: Python<'_>,
                dt: f64,
                op_split_burn: bool,
                use_bcyclic: bool,
                overshoot_f: f64,
                overshoot_f0: f64,
            ) -> PyResult<SolverReportTuple> {
                let cfgs = build_overshoot_configs(
                    overshoot_f,
                    overshoot_f0,
                    "exponential",
                );
                let inner = &mut self.inner;
                let rep = py
                    .detach(|| {
                        do_struct_burn_mix(
                            inner,
                            dt,
                            op_split_burn,
                            use_bcyclic,
                            &cfgs,
                        )
                    })
                    .map_err(to_py_err)?;
                Ok(report_to_tuple(rep))
            }

            /// Evaluates stellar wind mass loss and applies 3-region mass
            /// adjustment (`do_adjust_mass`), returning `mstar_dot`.
            #[pyo3(signature = (scheme="Dutch", eta=1.0, dt=0.0))]
            pub fn apply_wind_and_adjust_mass(
                &mut self,
                scheme: &str,
                eta: f64,
                dt: f64,
            ) -> PyResult<f64> {
                let ws = parse_wind_scheme(scheme)?;
                let mdot = set_mdot(
                    &mut self.inner,
                    ws,
                    DutchLowTScheme::DeJager,
                    eta,
                );
                if dt > 0.0 && mdot != 0.0 {
                    do_adjust_mass(&mut self.inner, mdot * dt, None, 0.99, 0.95)
                        .map_err(to_py_err)?;
                }
                Ok(mdot)
            }

            /// Executes adaptive remeshing (`do_adjust_mesh`), returning
            /// `(num_splits, num_merges)`.
            pub fn adjust_mesh(&mut self) -> PyResult<(usize, usize)> {
                do_adjust_mesh(&mut self.inner, &MeshFunctionConfig::default())
                    .map_err(to_py_err)
            }

            /// Computes `(dt_next, varcontrol, dt_limit_ratio)` via H211 filter.
            #[pyo3(signature = (
                max_timestep_factor=1.2, min_timestep_factor=0.5,
                dx_nuc_drop_limit=1.0e-2
            ))]
            pub fn pick_next_timestep(
                &mut self,
                max_timestep_factor: f64,
                min_timestep_factor: f64,
                dx_nuc_drop_limit: f64,
            ) -> (f64, f64, f64) {
                let r = pick_next_timestep(
                    &mut self.inner,
                    max_timestep_factor,
                    min_timestep_factor,
                    dx_nuc_drop_limit,
                );
                (r.dt_next, r.varcontrol, r.dt_limit_ratio)
            }

            /// Executes a single evolutionary step (`star_evolve_step`).
            #[pyo3(signature = (
                do_remesh=false, op_split_burn=false, use_bcyclic=true
            ))]
            pub fn evolve_step(
                &mut self,
                py: Python<'_>,
                do_remesh: bool,
                op_split_burn: bool,
                use_bcyclic: bool,
            ) -> PyResult<EvolveStepTuple> {
                let inner = &mut self.inner;
                let r = py
                    .detach(|| {
                        star_evolve_step(
                            inner,
                            do_remesh,
                            op_split_burn,
                            use_bcyclic,
                        )
                    })
                    .map_err(to_py_err)?;
                Ok(evolve_to_tuple(r))
            }

            /// Evolves the model for up to `max_steps` steps (`star_evolve`).
            #[pyo3(signature = (
                max_steps, do_remesh=false, op_split_burn=false,
                use_bcyclic=true
            ))]
            pub fn evolve(
                &mut self,
                py: Python<'_>,
                max_steps: usize,
                do_remesh: bool,
                op_split_burn: bool,
                use_bcyclic: bool,
            ) -> PyResult<Vec<EvolveStepTuple>> {
                let inner = &mut self.inner;
                let reps = py
                    .detach(|| {
                        star_evolve(
                            inner,
                            max_steps,
                            do_remesh,
                            op_split_burn,
                            use_bcyclic,
                        )
                    })
                    .map_err(to_py_err)?;
                Ok(reps.into_iter().map(evolve_to_tuple).collect())
            }

            /// Extracts `(int_counts, float_metrics)` for `history.data`.
            pub fn history_tuple(&self) -> HistoryTuple {
                let h = extract_history_row(&self.inner);
                (
                    (
                        h.model_number,
                        h.num_zones,
                        h.num_retries,
                        h.total_num_solver_iterations,
                    ),
                    [
                        h.star_age,
                        h.star_mass,
                        h.log_dt,
                        h.log_lh,
                        h.log_l,
                        h.log_teff,
                        h.log_r,
                        h.log_g,
                        h.log_center_t,
                        h.log_center_rho,
                        h.center_h1,
                        h.center_he4,
                        h.he_core_mass,
                    ],
                )
            }

            /// Formats a fixed-width MESA `history.data` row string.
            pub fn to_history_line(&self) -> String {
                format_history_row(&extract_history_row(&self.inner))
            }

            /// Formats a radial MESA `profile.data` table string.
            pub fn to_profile_str(&self) -> String {
                format_profile_string(&self.inner)
            }

            $(
                #[doc = concat!("Returns scalar `", stringify!($sget), "`.")]
                #[getter]
                pub const fn $sget(&self) -> $sty {
                    self.inner.$sget
                }
            )+

            $(
                #[doc = concat!("Returns 1D array `", stringify!($aget), "`.")]
                #[getter]
                pub fn $aget<'py>(
                    &self,
                    py: Python<'py>,
                ) -> Bound<'py, PyArray1<f64>> {
                    PyArray1::from_slice(py, &self.inner.$aget)
                }
            )+
        }
    };
}

impl_py_star_model!(
    scalars: [
        nz: usize, species: usize, model_number: usize, star_age: f64,
        dt: f64, dt_next: f64, dt_old: f64, mstar: f64, teff: f64
    ],
    arrays: [
        lnd, lnd_start, ln_t, ln_r, lum, dq, q, xq, m, dm, dm_bar, r, rmid,
        grav, rho, t, xh_mass, yhe_mass, pgas, prad, peos, ln_peos, energy,
        energy_start, entropy, grada, cp, cv, chi_rho, chi_t, gamma1, eta, mu,
        csound, opacity, tau, non_nuc_neu, eps_nuc, eps_mdot, gradt, gradr,
        gradl, scale_height, mlt_vc, mlt_d, mlt_cdc, d_mix, conv_vel, sig,
        omega, j_rot, brunt_b, brunt_n2
    ]
);

/// Computes the initial ZAMS timestep in years (`yrs_for_init_timestep`).
#[pyfunction]
pub fn star_yrs_for_init_timestep(m_msun: f64) -> f64 {
    yrs_for_init_timestep(m_msun)
}

/// Interpolates profile `v1` from Lagrangian grid `dq1` onto `dq2` (`interp_q`).
#[pyfunction]
pub fn star_interp_q<'py>(
    py: Python<'py>,
    dq1: PyReadonlyArray1<'py, f64>,
    v1: PyReadonlyArray1<'py, f64>,
    dq2: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let out = interp_q(dq1.as_slice()?, v1.as_slice()?, dq2.as_slice()?)
        .map_err(to_py_err)?;
    Ok(PyArray1::from_vec(py, out))
}

/// Registers `flexmesa-star` classes and functions on `flexmesa._rust`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStarModel>()?;
    m.add_function(wrap_pyfunction!(star_yrs_for_init_timestep, m)?)?;
    m.add_function(wrap_pyfunction!(star_interp_q, m)?)?;
    Ok(())
}
