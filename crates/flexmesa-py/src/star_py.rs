// SPDX-License-Identifier: LGPL-3.0-or-later
//! `PyO3` bindings for `flexmesa-star` (`RustStarModel`, ZAMS/pre-MS builders,
//! `.mod`/`.photo` I/O, parallel zone/face microphysics evaluation, and the
//! coupled 1D Henyey block-tridiagonal solver).
//!
//! Exposes [`PyStarModel`] (`flexmesa._rust.RustStarModel`) with zero-copy
//! slice-to-NumPy array getters across primary structure variables, EOS and
//! opacity diagnostics, nuclear energy generation rates, MLT convective
//! transport profiles, and Brunt-Väisälä buoyancy frequencies.

use flexmesa_star::{
    build_pre_ms_model, eval_henyey_equations, interp_q, load_zams_model,
    read_mod_str, read_photo_bytes, set_hydro_vars, solve_henyey_step,
    write_mod_string, write_photo_bytes, yrs_for_init_timestep, StarControls,
    StarError, StarJobControls, StarModel,
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

fn to_py_err(e: StarError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Python-exposed wrapper around [`flexmesa_star::StarModel`].
#[pyclass(name = "RustStarModel")]
#[derive(Clone)]
pub struct PyStarModel {
    /// Inner canonical Rust stellar model.
    pub inner: StarModel,
}

macro_rules! impl_py_star_model {
    ($( $(#[$doc:meta])+ $getter:ident => $field:ident ),+ $(,)?) => {
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
                let controls = StarControls {
                    initial_mass,
                    initial_z,
                    initial_y,
                    zbase: initial_z,
                    ..StarControls::default()
                };
                let job = StarJobControls::default();
                let inner = load_zams_model(controls, job).map_err(to_py_err)?;
                Ok(Self { inner })
            }

            /// Constructs an $n = 1.5$ pre-main-sequence polytrope model.
            #[staticmethod]
            #[pyo3(signature = (
                initial_mass=1.0,
                initial_z=0.02,
                initial_y=-1.0,
                t_c=9.0e5,
                d_log10_p=0.02
            ))]
            pub fn from_pre_ms(
                initial_mass: f64,
                initial_z: f64,
                initial_y: f64,
                t_c: f64,
                d_log10_p: f64,
            ) -> PyResult<Self> {
                let controls = StarControls {
                    initial_mass,
                    initial_z,
                    initial_y,
                    zbase: initial_z,
                    ..StarControls::default()
                };
                let job = StarJobControls {
                    create_pre_ms_model: true,
                    pre_ms_t_c: t_c,
                    pre_ms_d_log10_p: d_log10_p,
                    ..StarJobControls::default()
                };
                let inner =
                    build_pre_ms_model(controls, job).map_err(to_py_err)?;
                Ok(Self { inner })
            }

            /// Parses an ASCII `.mod` model string.
            #[staticmethod]
            pub fn from_mod_str(text: &str) -> PyResult<Self> {
                let inner = read_mod_str(
                    text,
                    StarControls::default(),
                    StarJobControls::default(),
                )
                .map_err(to_py_err)?;
                Ok(Self { inner })
            }

            /// Restores a [`PyStarModel`] from binary `.photo` bytes.
            #[staticmethod]
            pub fn from_photo_bytes(bytes: &[u8]) -> PyResult<Self> {
                let inner = read_photo_bytes(
                    bytes,
                    StarControls::default(),
                    StarJobControls::default(),
                )
                .map_err(to_py_err)?;
                Ok(Self { inner })
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
                let buf = write_photo_bytes(&self.inner);
                PyBytes::new(py, &buf)
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
                let lnd_s = lnd.as_slice()?;
                let lnt_s = ln_t.as_slice()?;
                let lnr_s = ln_r.as_slice()?;
                let lum_s = lum.as_slice()?;
                let dq_s = dq.as_slice()?;
                let xa_s = xa.as_slice()?;
                let nz = lnd_s.len();
                let species = self.inner.species;
                if lnt_s.len() != nz
                    || lnr_s.len() != nz
                    || lum_s.len() != nz
                    || dq_s.len() != nz
                    || xa_s.len() != nz * species
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
                self.inner.sync_mesh_and_pack().map_err(to_py_err)?;
                Ok(())
            }

            /// Number of Lagrangian zones (`nz`).
            #[getter]
            pub const fn nz(&self) -> usize {
                self.inner.nz
            }

            /// Number of nuclear species (`species`).
            #[getter]
            pub const fn species(&self) -> usize {
                self.inner.species
            }

            /// Current evolutionary model number.
            #[getter]
            pub const fn model_number(&self) -> usize {
                self.inner.model_number
            }

            /// Stellar age in years.
            #[getter]
            pub const fn star_age(&self) -> f64 {
                self.inner.star_age
            }

            /// Current timestep $\Delta t$ in seconds.
            #[getter]
            pub const fn dt(&self) -> f64 {
                self.inner.dt
            }

            /// Sets the current timestep $\Delta t$ in seconds.
            #[setter]
            pub fn set_dt(&mut self, dt: f64) {
                self.inner.dt = dt;
            }

            /// Snapshots current state arrays as start-of-step `*_start` values.
            pub fn save_start_values(&mut self) {
                self.inner.save_start_values();
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
                let nz = sys.nz;
                let nvar = sys.nvar;
                let equ =
                    PyArray1::from_vec(py, sys.equ).reshape([nz, nvar])?;
                let lblk = PyArray1::from_vec(py, sys.lblk)
                    .reshape([nz, nvar, nvar])?;
                let dblk = PyArray1::from_vec(py, sys.dblk)
                    .reshape([nz, nvar, nvar])?;
                let ublk = PyArray1::from_vec(py, sys.ublk)
                    .reshape([nz, nvar, nvar])?;
                let x_scale =
                    PyArray1::from_vec(py, sys.x_scale).reshape([nz, nvar])?;
                Ok((equ, lblk, dblk, ublk, x_scale))
            }

            /// Executes a damped Newton-Raphson Henyey step, returning
            /// `(converged, iterations, residual_norm, max_residual,
            /// correction_norm, max_correction)`.
            #[pyo3(signature = (include_chem=false, use_bcyclic=true))]
            pub fn solve_henyey(
                &mut self,
                py: Python<'_>,
                include_chem: bool,
                use_bcyclic: bool,
            ) -> PyResult<(bool, usize, f64, f64, f64, f64)> {
                let inner = &mut self.inner;
                let rep = py
                    .detach(|| {
                        solve_henyey_step(inner, include_chem, use_bcyclic)
                    })
                    .map_err(to_py_err)?;
                Ok((
                    rep.converged,
                    rep.iterations,
                    rep.residual_norm,
                    rep.max_residual,
                    rep.correction_norm,
                    rep.max_correction,
                ))
            }

            /// Total stellar mass $M_*$ in grams.
            #[getter]
            pub const fn mstar(&self) -> f64 {
                self.inner.mstar
            }

            /// Surface effective temperature $T_{\mathrm{eff}}$ in K.
            #[getter]
            pub const fn teff(&self) -> f64 {
                self.inner.teff
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

            $(
                $(#[$doc])+
                #[getter]
                pub fn $getter<'py>(
                    &self,
                    py: Python<'py>,
                ) -> Bound<'py, PyArray1<f64>> {
                    PyArray1::from_slice(py, &self.inner.$field)
                }
            )+
        }
    };
}

impl_py_star_model!(
    /// Cell-centered $\ln(\rho/\mathrm{g\,cm^{-3}})$.
    lnd => lnd,
    /// Start-of-step cell-centered $\ln(\rho/\mathrm{g\,cm^{-3}})$.
    lnd_start => lnd_start,
    /// Cell-centered $\ln(T/\mathrm{K})$.
    ln_t => ln_t,
    /// Face-centered $\ln(R/\mathrm{cm})$.
    ln_r => ln_r,
    /// Face-centered luminosity $L$ ($\mathrm{erg\,s^{-1}}$).
    lum => lum,
    /// Cell relative mass fraction $\Delta q$.
    dq => dq,
    /// Face enclosed mass fraction $q$.
    q => q,
    /// Face enclosed mass $m$ ($\mathrm{g}$).
    m => m,
    /// Cell mass $\Delta m$ ($\mathrm{g}$).
    dm => dm,
    /// Face control-volume mass $\overline{\Delta m}$ ($\mathrm{g}$).
    dm_bar => dm_bar,
    /// Face radius $r$ ($\mathrm{cm}$).
    r => r,
    /// Cell volumetric mid-radius $r_{\mathrm{mid}}$ ($\mathrm{cm}$).
    rmid => rmid,
    /// Face gravitational acceleration $g$ ($\mathrm{cm\,s^{-2}}$).
    grav => grav,
    /// Cell mass density $\rho$ ($\mathrm{g\,cm^{-3}}$).
    rho => rho,
    /// Cell temperature $T$ ($\mathrm{K}$).
    t => t,
    /// Cell gas pressure $P_{\mathrm{gas}}$ ($\mathrm{dyn\,cm^{-2}}$).
    pgas => pgas,
    /// Cell radiation pressure $P_{\mathrm{rad}}$ ($\mathrm{dyn\,cm^{-2}}$).
    prad => prad,
    /// Cell total pressure $P_{\mathrm{eos}}$ ($\mathrm{dyn\,cm^{-2}}$).
    peos => peos,
    /// Cell $\ln P_{\mathrm{eos}}$.
    ln_peos => ln_peos,
    /// Cell specific internal energy $e$ ($\mathrm{erg\,g^{-1}}$).
    energy => energy,
    /// Start-of-step cell specific internal energy $e_{\mathrm{start}}$.
    energy_start => energy_start,
    /// Cell specific entropy $s$ ($\mathrm{erg\,g^{-1}\,K^{-1}}$).
    entropy => entropy,
    /// Cell adiabatic temperature gradient $\nabla_{\mathrm{ad}}$.
    grada => grada,
    /// Cell specific heat $C_P$ ($\mathrm{erg\,g^{-1}\,K^{-1}}$).
    cp => cp,
    /// Cell specific heat $C_V$ ($\mathrm{erg\,g^{-1}\,K^{-1}}$).
    cv => cv,
    /// Cell $\chi_\rho = (\partial\ln P/\partial\ln\rho)_T$.
    chi_rho => chi_rho,
    /// Cell $\chi_T = (\partial\ln P/\partial\ln T)_\rho$.
    chi_t => chi_t,
    /// Cell adiabatic exponent $\Gamma_1$.
    gamma1 => gamma1,
    /// Cell electron degeneracy parameter $\eta$.
    eta => eta,
    /// Cell mean molecular weight $\mu$.
    mu => mu,
    /// Cell adiabatic sound speed $c_s$ ($\mathrm{cm\,s^{-1}}$).
    csound => csound,
    /// Cell Rosseland mean opacity $\kappa$ ($\mathrm{cm^2\,g^{-1}}$).
    opacity => opacity,
    /// Face optical depth $\tau$.
    tau => tau,
    /// Cell thermal neutrino energy loss rate $\epsilon_{\nu,\mathrm{thermal}}$.
    non_nuc_neu => non_nuc_neu,
    /// Cell nuclear energy generation rate $\epsilon_{\mathrm{nuc}}$.
    eps_nuc => eps_nuc,
    /// Face actual temperature gradient $\nabla_T$.
    gradt => gradt,
    /// Face radiative temperature gradient $\nabla_{\mathrm{rad}}$.
    gradr => gradr,
    /// Face Ledoux temperature gradient $\nabla_L$.
    gradl => gradl,
    /// Face pressure scale height $H_P$ ($\mathrm{cm}$).
    scale_height => scale_height,
    /// Face convective velocity $v_{\mathrm{conv}}$ ($\mathrm{cm\,s^{-1}}$).
    mlt_vc => mlt_vc,
    /// Face MLT diffusion coefficient $D_{\mathrm{MLT}}$ ($\mathrm{cm^2\,s^{-1}}$).
    mlt_d => mlt_d,
    /// Face convective conductivity $c_{\mathrm{dc}}$.
    mlt_cdc => mlt_cdc,
    /// Face composition gradient term $B$.
    brunt_b => brunt_b,
    /// Face Brunt-Väisälä frequency squared $N^2$ ($\mathrm{s^{-2}}$).
    brunt_n2 => brunt_n2,
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
