// SPDX-License-Identifier: LGPL-3.0-or-later
//! `PyO3` bindings for `flexmesa-star` (`RustStarModel`, ZAMS/pre-MS builders,
//! `.mod`/`.photo` I/O, and parallel zone/face microphysics evaluation).

use flexmesa_star::{
    build_pre_ms_model, interp_q, load_zams_model, read_mod_str,
    read_photo_bytes, set_hydro_vars, write_mod_string, write_photo_bytes,
    yrs_for_init_timestep, StarControls, StarError, StarJobControls, StarModel,
};
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

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

#[pymethods]
impl PyStarModel {
    /// Loads a ZAMS stellar model from the built-in `zams_z2m2_y28` catalog.
    #[staticmethod]
    #[pyo3(signature = (initial_mass=1.0, initial_z=0.02, initial_y=-1.0))]
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
    #[pyo3(signature = (initial_mass=1.0, initial_z=0.02, initial_y=-1.0, t_c=9.0e5, d_log10_p=0.02))]
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
        let inner = build_pre_ms_model(controls, job).map_err(to_py_err)?;
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

    /// Restores a [`PyStarModel`] from binary `.photo` checkpoint bytes.
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
    pub fn to_photo_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let buf = write_photo_bytes(&self.inner);
        PyBytes::new(py, &buf)
    }

    /// Evaluates zone and face microphysics and hydrodynamic variables
    /// (`set_hydro_vars`).
    #[pyo3(signature = (skip_net=false, skip_net_jacobian=false))]
    pub fn eval_hydro_vars(
        &mut self,
        py: Python<'_>,
        skip_net: bool,
        skip_net_jacobian: bool,
    ) -> PyResult<()> {
        let inner = &mut self.inner;
        py.detach(|| set_hydro_vars(inner, skip_net, skip_net_jacobian))
            .map_err(to_py_err)
    }

    /// Replaces primary state arrays (`lnd`, `ln_t`, `ln_r`, `lum`, `dq`, `xa`)
    /// and updates staggered mass/radius geometry.
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

    /// Cell-centered $\ln(\rho/\mathrm{g\,cm^{-3}})$.
    #[getter]
    pub fn lnd<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.lnd)
    }

    /// Cell-centered $\ln(T/\mathrm{K})$.
    #[getter]
    pub fn ln_t<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.ln_t)
    }

    /// Face-centered $\ln(R/\mathrm{cm})$.
    #[getter]
    pub fn ln_r<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.ln_r)
    }

    /// Face-centered luminosity $L$ ($\mathrm{erg\,s^{-1}}$).
    #[getter]
    pub fn lum<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.lum)
    }

    /// Cell relative mass fraction $\Delta q$.
    #[getter]
    pub fn dq<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.dq)
    }

    /// Face enclosed mass fraction $q$.
    #[getter]
    pub fn q<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.q)
    }

    /// Face enclosed mass $m$ ($\mathrm{g}$).
    #[getter]
    pub fn m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.m)
    }

    /// Cell mass $\Delta m$ ($\mathrm{g}$).
    #[getter]
    pub fn dm<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.dm)
    }

    /// Face control-volume mass $\overline{\Delta m}$ ($\mathrm{g}$).
    #[getter]
    pub fn dm_bar<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.dm_bar)
    }

    /// Face radius $r$ ($\mathrm{cm}$).
    #[getter]
    pub fn r<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.r)
    }

    /// Cell volumetric mid-radius $r_{\mathrm{mid}}$ ($\mathrm{cm}$).
    #[getter]
    pub fn rmid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.rmid)
    }

    /// Face gravitational acceleration $g$ ($\mathrm{cm\,s^{-2}}$).
    #[getter]
    pub fn grav<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.grav)
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

    /// Cell mass density $\rho$ ($\mathrm{g\,cm^{-3}}$).
    #[getter]
    pub fn rho<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.rho)
    }

    /// Cell temperature $T$ ($\mathrm{K}$).
    #[getter]
    pub fn t<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.t)
    }

    /// Cell gas pressure $P_{\mathrm{gas}}$ ($\mathrm{dyn\,cm^{-2}}$).
    #[getter]
    pub fn pgas<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.pgas)
    }

    /// Cell radiation pressure $P_{\mathrm{rad}}$ ($\mathrm{dyn\,cm^{-2}}$).
    #[getter]
    pub fn prad<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.prad)
    }

    /// Cell total pressure $P_{\mathrm{eos}}$ ($\mathrm{dyn\,cm^{-2}}$).
    #[getter]
    pub fn peos<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.peos)
    }

    /// Cell $\ln P_{\mathrm{eos}}$.
    #[getter]
    pub fn ln_peos<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.ln_peos)
    }

    /// Cell specific internal energy $e$ ($\mathrm{erg\,g^{-1}}$).
    #[getter]
    pub fn energy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.energy)
    }

    /// Cell specific entropy $s$ ($\mathrm{erg\,g^{-1}\,K^{-1}}$).
    #[getter]
    pub fn entropy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.entropy)
    }

    /// Cell adiabatic temperature gradient $\nabla_{\mathrm{ad}}$.
    #[getter]
    pub fn grada<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.grada)
    }

    /// Cell specific heat $C_P$ ($\mathrm{erg\,g^{-1}\,K^{-1}}$).
    #[getter]
    pub fn cp<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.cp)
    }

    /// Cell specific heat $C_V$ ($\mathrm{erg\,g^{-1}\,K^{-1}}$).
    #[getter]
    pub fn cv<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.cv)
    }

    /// Cell $\chi_\rho = (\partial\ln P/\partial\ln\rho)_T$.
    #[getter]
    pub fn chi_rho<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.chi_rho)
    }

    /// Cell $\chi_T = (\partial\ln P/\partial\ln T)_\rho$.
    #[getter]
    pub fn chi_t<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.chi_t)
    }

    /// Cell adiabatic exponent $\Gamma_1$.
    #[getter]
    pub fn gamma1<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.gamma1)
    }

    /// Cell electron degeneracy parameter $\eta$.
    #[getter]
    pub fn eta<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.eta)
    }

    /// Cell mean molecular weight $\mu$.
    #[getter]
    pub fn mu<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.mu)
    }

    /// Cell adiabatic sound speed $c_s$ ($\mathrm{cm\,s^{-1}}$).
    #[getter]
    pub fn csound<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.csound)
    }

    /// Cell Rosseland mean opacity $\kappa$ ($\mathrm{cm^2\,g^{-1}}$).
    #[getter]
    pub fn opacity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.opacity)
    }

    /// Face optical depth $\tau$.
    #[getter]
    pub fn tau<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.tau)
    }

    /// Cell thermal neutrino energy loss rate $\epsilon_{\nu,\mathrm{thermal}}$.
    #[getter]
    pub fn non_nuc_neu<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.non_nuc_neu)
    }

    /// Cell nuclear energy generation rate $\epsilon_{\mathrm{nuc}}$ ($\mathrm{erg\,g^{-1}\,s^{-1}}$).
    #[getter]
    pub fn eps_nuc<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.eps_nuc)
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

    /// Face actual temperature gradient $\nabla_T$.
    #[getter]
    pub fn gradt<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.gradt)
    }

    /// Face radiative temperature gradient $\nabla_{\mathrm{rad}}$.
    #[getter]
    pub fn gradr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.gradr)
    }

    /// Face Ledoux temperature gradient $\nabla_L$.
    #[getter]
    pub fn gradl<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.gradl)
    }

    /// Face pressure scale height $H_P$ ($\mathrm{cm}$).
    #[getter]
    pub fn scale_height<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.scale_height)
    }

    /// Face convective velocity $v_{\mathrm{conv}}$ ($\mathrm{cm\,s^{-1}}$).
    #[getter]
    pub fn mlt_vc<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.mlt_vc)
    }

    /// Face MLT diffusion coefficient $D_{\mathrm{MLT}}$ ($\mathrm{cm^2\,s^{-1}}$).
    #[getter]
    pub fn mlt_d<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.mlt_d)
    }

    /// Face convective conductivity $c_{\mathrm{dc}}$.
    #[getter]
    pub fn mlt_cdc<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.mlt_cdc)
    }

    /// Face composition gradient term $B$.
    #[getter]
    pub fn brunt_b<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.brunt_b)
    }

    /// Face Brunt-Väisälä frequency squared $N^2$ ($\mathrm{s^{-2}}$).
    #[getter]
    pub fn brunt_n2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.brunt_n2)
    }

    /// Face integer mixing classification codes (`0 = NoMixing`, `1 = ConvectiveMixing`, etc.).
    #[getter]
    pub fn mixing_type<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray1<i32>> {
        let codes: Vec<i32> =
            self.inner.mixing_type.iter().map(|m| m.code()).collect();
        PyArray1::from_vec(py, codes)
    }
}

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
