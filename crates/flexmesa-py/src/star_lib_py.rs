// SPDX-License-Identifier: GPL-3.0-or-later
//! Python access to the `star_lib` subset in [`flexmesa_star::star_lib`], for
//! test-side extras that build a starting model cell by cell (`make_env`'s
//! `create_env`, `star/test_suite/make_env/src/run_star_extras.f90:158-568`).
//!
//! `StarLibModel` owns the model being built. Cell indices are 0-based
//! (upstream's `k - 1`). Arrays and scalars are read and written by their
//! upstream `star_info` names.

use std::path::Path;

use flexmesa_star::controls::{
    AtmOption, AtmTTauOpacity, AtmTTauRelation, StarError,
};
use flexmesa_star::model::StarModel;
use flexmesa_star::star_ad::I_LNT_00;
use flexmesa_star::star_lib;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::astero_gyre_py::PyStarModelHandle;
use crate::star_helpers_py::to_py_err;

/// A star model under construction through the `star_lib` subset.
#[pyclass(name = "StarLibModel", module = "flexmesa._rust")]
pub struct PyStarLibModel {
    model: StarModel,
}

fn unknown(kind: &str, name: &str) -> PyErr {
    PyValueError::new_err(format!("StarLibModel: unknown {kind} '{name}'"))
}

fn cell_array<'a>(
    m: &'a mut StarModel,
    name: &str,
) -> PyResult<&'a mut Vec<f64>> {
    Ok(match name {
        "q" => &mut m.q,
        "dq" => &mut m.dq,
        "m" => &mut m.m,
        "dm" => &mut m.dm,
        "dm_bar" => &mut m.dm_bar,
        "m_grav" => &mut m.m_grav,
        "cgrav" => &mut m.cgrav,
        "r" => &mut m.r,
        "lnR" => &mut m.ln_r,
        "L" => &mut m.lum,
        "lnd" => &mut m.lnd,
        "lnT" => &mut m.ln_t,
        "rho" => &mut m.rho,
        "T" => &mut m.t,
        "lnT_start" => &mut m.ln_t_start,
        "lnd_start" => &mut m.lnd_start,
        "opacity" => &mut m.opacity,
        "csound" => &mut m.csound,
        "gradT" => &mut m.gradt,
        "gradr" => &mut m.gradr,
        "grada" => &mut m.grada,
        "csound_start" => &mut m.aux.csound_start,
        "alpha_mlt" => &mut m.aux.mixing.alpha_mlt,
        "gradr_factor" => &mut m.aux.mixing.gradr_factor,
        "gradL_composition_term" => &mut m.aux.mixing.gradl_composition_term,
        "adjust_mlt_gradT_fraction" => {
            &mut m.aux.mixing.adjust_mlt_gradt_fraction
        }
        other => return Err(unknown("cell array", other)),
    })
}

fn scalar<'a>(m: &'a mut StarModel, name: &str) -> PyResult<&'a mut f64> {
    Ok(match name {
        "mstar" => &mut m.mstar,
        "xmstar" => &mut m.xmstar,
        "M_center" => &mut m.m_center,
        "R_center" => &mut m.r_center,
        "L_center" => &mut m.l_center,
        "Teff" => &mut m.teff,
        "tau_base" => &mut m.tau_base,
        "star_age" => &mut m.star_age,
        "crystal_core_boundary_mass" => &mut m.aux.crystal_core_boundary_mass,
        "tau_factor" => &mut m.controls.tau_factor,
        "Pextra_factor" => &mut m.controls.pextra_factor,
        "mixing_length_alpha" => &mut m.controls.mixing_length_alpha,
        "mlt_gradT_fraction" => &mut m.controls.mix_wind.mlt_gradt_fraction,
        other => return Err(unknown("scalar", other)),
    })
}

fn sp(e: StarError) -> PyErr {
    to_py_err(e)
}

#[pymethods]
impl PyStarLibModel {
    /// `star_set_net`, `star_set_var_info`, `star_set_chem_names` and
    /// `star_allocate_arrays` (`star/public/star_lib.f90:301-345`) for a
    /// star with the controls and `&star_job` values of `star`.
    #[staticmethod]
    #[pyo3(signature = (star, net_name, nz, search_dir=None))]
    fn new_for_build(
        star: &PyStarModelHandle,
        net_name: &str,
        nz: usize,
        search_dir: Option<&str>,
    ) -> PyResult<Self> {
        let src = &star.model;
        let mut model = star_lib::star_new_model_for_build(
            net_name,
            nz,
            src.controls.clone(),
            src.job.clone(),
            search_dir.map(Path::new),
        )
        .map_err(sp)?;
        model.eos_config = src.eos_config;
        model.kap_config = src.kap_config.clone();
        Ok(Self { model })
    }

    /// Number of cells.
    #[getter]
    fn nz(&self) -> usize {
        self.model.nz
    }

    /// Number of species.
    #[getter]
    fn species(&self) -> usize {
        self.model.species
    }

    /// `s% model_number`.
    #[getter]
    fn get_model_number(&self) -> usize {
        self.model.model_number
    }

    /// Sets `s% model_number`.
    #[setter]
    fn set_model_number(&mut self, v: usize) {
        self.model.model_number = v;
    }

    /// `s% <name>(k)`.
    fn get(&mut self, name: &str, k: usize) -> PyResult<f64> {
        let a = cell_array(&mut self.model, name)?;
        a.get(k).copied().ok_or_else(|| {
            PyValueError::new_err(format!("{name}: cell {k} out of range"))
        })
    }

    /// Sets `s% <name>(k) = val`.
    fn set(&mut self, name: &str, k: usize, val: f64) -> PyResult<()> {
        let a = cell_array(&mut self.model, name)?;
        let slot = a.get_mut(k).ok_or_else(|| {
            PyValueError::new_err(format!("{name}: cell {k} out of range"))
        })?;
        *slot = val;
        Ok(())
    }

    /// `s% <name>` for a scalar.
    fn get_scalar(&mut self, name: &str) -> PyResult<f64> {
        Ok(*scalar(&mut self.model, name)?)
    }

    /// Sets the scalar `s% <name> = val`.
    fn set_scalar(&mut self, name: &str, val: f64) -> PyResult<()> {
        *scalar(&mut self.model, name)? = val;
        Ok(())
    }

    /// `s% xa(:, k)`.
    fn xa(&self, k: usize) -> Vec<f64> {
        self.model.cell_xa(k).to_vec()
    }

    /// `(s% atm_option, s% atm_T_tau_relation, s% atm_T_tau_opacity)`.
    fn get_atm_options(&self) -> (String, String, String) {
        let c = &self.model.controls;
        let option = match c.atm_option {
            AtmOption::TTau => "T_tau",
            AtmOption::Table => "table",
            AtmOption::IrradiatedGrey => "irradiated_grey",
        };
        let relation = match c.atm_t_tau_relation {
            AtmTTauRelation::Eddington => "Eddington",
            AtmTTauRelation::SolarHopf => "solar_Hopf",
            AtmTTauRelation::KrishnaSwamy => "Krishna_Swamy",
            AtmTTauRelation::TrampedachSolar => "Trampedach_solar",
        };
        let opacity = match c.atm_t_tau_opacity {
            AtmTTauOpacity::Fixed => "fixed",
            AtmTTauOpacity::Iterated => "iterated",
            AtmTTauOpacity::Varying => "varying",
        };
        (option.into(), relation.into(), opacity.into())
    }

    /// Sets `s% atm_option`, `s% atm_T_tau_relation` and
    /// `s% atm_T_tau_opacity`.
    fn set_atm_options(
        &mut self,
        option: &str,
        relation: &str,
        opacity: &str,
    ) -> PyResult<()> {
        let c = &mut self.model.controls;
        c.atm_option = match option {
            "T_tau" => AtmOption::TTau,
            "table" => AtmOption::Table,
            "irradiated_grey" => AtmOption::IrradiatedGrey,
            other => return Err(unknown("atm_option", other)),
        };
        c.atm_t_tau_relation = match relation {
            "Eddington" => AtmTTauRelation::Eddington,
            "solar_Hopf" => AtmTTauRelation::SolarHopf,
            "Krishna_Swamy" => AtmTTauRelation::KrishnaSwamy,
            "Trampedach_solar" => AtmTTauRelation::TrampedachSolar,
            other => return Err(unknown("atm_T_tau_relation", other)),
        };
        c.atm_t_tau_opacity = match opacity {
            "fixed" => AtmTTauOpacity::Fixed,
            "iterated" => AtmTTauOpacity::Iterated,
            "varying" => AtmTTauOpacity::Varying,
            other => return Err(unknown("atm_T_tau_opacity", other)),
        };
        Ok(())
    }

    /// `star_normalize_dqs` (`star/public/star_lib.f90:2355-2367`).
    fn normalize_dqs(&mut self) -> PyResult<()> {
        star_lib::star_normalize_dqs(&mut self.model).map_err(sp)
    }

    /// `star_set_qs` (`star/public/star_lib.f90:2370-2381`).
    fn set_qs(&mut self) {
        star_lib::star_set_qs(&mut self.model);
    }

    /// `star_set_m_and_dm` (`star/public/star_lib.f90:2384-2393`).
    fn set_m_and_dm(&mut self) {
        star_lib::star_set_m_and_dm(&mut self.model);
    }

    /// `star_set_dm_bar` (`star/public/star_lib.f90:2396-2405`).
    fn set_dm_bar(&mut self) {
        star_lib::star_set_dm_bar(&mut self.model);
    }

    /// `change_to_xa_for_accretion` (`star/public/star_lib.f90:1038-1044`)
    /// for cells `k_lo..=k_hi`.
    fn change_to_xa_for_accretion(
        &mut self,
        k_lo: usize,
        k_hi: usize,
    ) -> PyResult<()> {
        star_lib::change_to_xa_for_accretion(&mut self.model, k_lo, k_hi)
            .map_err(sp)
    }

    /// `basic_composition_info` for cell `k` (`chem/public/chem_lib.f90`).
    fn basic_composition_info(&mut self, k: usize) -> PyResult<()> {
        star_lib::star_basic_composition_info(&mut self.model, k).map_err(sp)
    }

    /// `star_get_atm_PT` (`star/public/star_lib.f90:2223-2247`): returns
    /// `(lnT_surf, lnP_surf)`.
    #[allow(clippy::too_many_arguments)]
    fn get_atm_pt(
        &self,
        tau_surf: f64,
        l: f64,
        r: f64,
        m: f64,
        cgrav: f64,
        teff: f64,
    ) -> PyResult<(f64, f64)> {
        star_lib::star_get_atm_pt(&self.model, tau_surf, l, r, m, cgrav, teff)
            .map_err(sp)
    }

    /// `star_get_surf_PT` with both `need_atm_*` flags set
    /// (`star/public/star_lib.f90:2250-2270`): returns `(lnT_surf,
    /// lnP_surf)`.
    fn get_surf_pt(&mut self) -> PyResult<(f64, f64)> {
        star_lib::star_get_surf_pt(&mut self.model).map_err(sp)
    }

    /// `star_solve_eos_given_PgasT` (`star/public/star_lib.f90:2114-2139`)
    /// with the composition of cell `k`: returns `logRho`.
    #[allow(clippy::too_many_arguments)]
    fn solve_eos_given_pgas_t(
        &self,
        k: usize,
        log_t: f64,
        log_pgas: f64,
        log_rho_guess: f64,
        log_rho_tol: f64,
        log_pgas_tol: f64,
    ) -> PyResult<f64> {
        let xa = self.model.cell_xa(k).to_vec();
        star_lib::star_solve_eos_given_pgas_t(
            &self.model,
            &xa,
            log_t,
            log_pgas,
            log_rho_guess,
            log_rho_tol,
            log_pgas_tol,
        )
        .map(|(lr, _)| lr)
        .map_err(sp)
    }

    /// `star_solve_eos_given_PgasT_auto` (`star/public/star_lib.f90:
    /// 2142-2165`) with the composition of cell `k`: returns `logRho`.
    fn solve_eos_given_pgas_t_auto(
        &self,
        k: usize,
        log_t: f64,
        log_pgas: f64,
        log_rho_tol: f64,
        log_pgas_tol: f64,
    ) -> PyResult<f64> {
        let xa = self.model.cell_xa(k).to_vec();
        star_lib::star_solve_eos_given_pgas_t_auto(
            &self.model,
            &xa,
            log_t,
            log_pgas,
            log_rho_tol,
            log_pgas_tol,
        )
        .map(|(lr, _)| lr)
        .map_err(sp)
    }

    /// `star_do_eos_for_cell` (`star/public/star_lib.f90:2197-2207`).
    fn do_eos_for_cell(&mut self, k: usize) -> PyResult<()> {
        star_lib::star_do_eos_for_cell(&mut self.model, k).map_err(sp)
    }

    /// `star_do_kap_for_cell` (`star/public/star_lib.f90:2210-2220`).
    fn do_kap_for_cell(&mut self, k: usize) -> PyResult<()> {
        star_lib::star_do_kap_for_cell(&mut self.model, k).map_err(sp)
    }

    /// `star_set_mlt_vars` (`star/public/star_lib.f90:3064-3074`) for faces
    /// `k_lo..=k_hi`.
    fn set_mlt_vars(&mut self, k_lo: usize, k_hi: usize) -> PyResult<()> {
        star_lib::star_set_mlt_vars(&mut self.model, k_lo, k_hi).map_err(sp)
    }

    /// `s% gradT_ad(k)%d1Array(i_lnT_00)`.
    fn d_gradt_dlnt00(&self, k: usize) -> PyResult<f64> {
        self.model
            .gradt_ad
            .get(k)
            .map(|g| g.d1[I_LNT_00])
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "gradT_ad: cell {k} out of range"
                ))
            })
    }

    /// Packs `lnd`, `lnT`, `lnR` and `L` into `xh` (`s% xh(i_lnd, k)`, ...).
    fn pack_xh(&mut self) {
        self.model.pack_xh();
    }

    /// `star_write_model` (`star/public/star_lib.f90:483-489`).
    fn write_model(&self, path: &str) -> PyResult<()> {
        star_lib::star_write_model(&self.model, Path::new(path)).map_err(sp)
    }
}

/// `atm_Teff(L, R)` (`atm/public/atm_lib.f90:74-78`): the black-body
/// effective temperature in K for `L` in erg s$^{-1}$ and `R` in cm.
#[pyfunction]
fn atm_teff(l: f64, r: f64) -> f64 {
    flexmesa_atm::utils::atm_teff(l, r)
}

/// Correctly rounded natural logarithm (`log` in `math_lib`).
#[pyfunction]
fn crmath_log(x: f64) -> f64 {
    flexmesa_math::log(x)
}

/// Registers `StarLibModel`, `atm_teff` and `crmath_log`.
///
/// # Errors
/// Returns [`PyErr`] if registration fails.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStarLibModel>()?;
    m.add_function(wrap_pyfunction!(atm_teff, m)?)?;
    m.add_function(wrap_pyfunction!(crmath_log, m)?)?;
    Ok(())
}
