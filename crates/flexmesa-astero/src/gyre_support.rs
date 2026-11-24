// SPDX-License-Identifier: GPL-3.0-or-later
//! Astero's GYRE interface (`astero/private/gyre_support.f90`): `init_gyre`,
//! `do_gyre_get_modes` with `store_model_for_gyre` and `gyre_call_back`,
//! and the oscillation-result arrays `store_new_oscillation_results` fills
//! (`astero/public/astero_def.f90:673-716`), over the GYRE port in
//! `flexmesa-gyre` (#170, package G3c).

use std::path::Path;

use flexmesa_const::{CLIGHT, CRAD, LSUN, MSUN, RSUN, STANDARD_CGRAV};
use flexmesa_gyre::mesa::GyreMesa;
use flexmesa_gyre::mode::Mode;
use flexmesa_gyre::GyreError;
use flexmesa_star::controls::StarError;
use flexmesa_star::model::StarModel;
use flexmesa_star::pulse_gyre_data::star_get_pulse_data_gyre;

use crate::inlist_astero::AsteroSearchControls;

/// One stored oscillation result (`astero_def.f90:673-716`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OscResult {
    /// `el(n)`.
    pub el: i32,
    /// `order(n)` (GYRE `n_pg`).
    pub order: i32,
    /// `em(n)`.
    pub em: i32,
    /// `inertia(n)` (GYRE `E_norm`).
    pub inertia: f64,
    /// `cyclic_freq(n)` ($\mu$Hz).
    pub cyclic_freq: f64,
    /// `growth_rate(n)` (rad/s).
    pub growth_rate: f64,
}

fn gyre_err(e: &GyreError) -> StarError {
    StarError::InvalidInput(format!("GYRE: {e}"))
}

/// GYRE state for an astero run (`gyre_mesa_m`'s module variables, owned
/// here) and the astero controls `do_gyre_get_modes` reads.
#[derive(Debug, Clone)]
pub struct AsteroGyre {
    /// The GYRE library state.
    pub gyre: GyreMesa,
    add_center_point: bool,
    keep_surface_point: bool,
    add_atmosphere: bool,
}

impl AsteroGyre {
    /// `init_gyre(gyre_file, ierr)` (`gyre_support.f90:45-72`): read the
    /// GYRE input file and set MESA's constants. `mesa_dir` gives
    /// `GYRE_DIR = mesa_dir//'/gyre/gyre'` (only checked to be non-empty).
    ///
    /// # Errors
    ///
    /// [`StarError::IoError`] for an unreadable input file;
    /// [`StarError::InvalidInput`] for GYRE parameter errors and for
    /// `oscillation_code /= 'gyre'`.
    pub fn init(
        case_dir: &Path,
        ctrl: &AsteroSearchControls,
        mesa_dir: &str,
    ) -> Result<Self, StarError> {
        if !ctrl.oscillation_code.trim().eq_ignore_ascii_case("gyre") {
            return Err(StarError::InvalidInput(format!(
                "AsteroGyre requires oscillation_code = 'gyre', got '{}'",
                ctrl.oscillation_code
            )));
        }
        let path = case_dir.join(ctrl.gyre_input_file.trim());
        let text = std::fs::read_to_string(&path).map_err(|e| {
            StarError::IoError(format!(
                "failed to read GYRE input file '{}': {e}",
                path.display()
            ))
        })?;
        let mut gyre = GyreMesa::init(&text).map_err(|e| gyre_err(&e))?;
        for (name, value) in [
            ("G_GRAVITY", STANDARD_CGRAV),
            ("C_LIGHT", CLIGHT),
            ("A_RADIATION", CRAD),
            ("M_SUN", MSUN),
            ("R_SUN", RSUN),
            ("L_SUN", LSUN),
        ] {
            gyre.set_constant(name, value).map_err(|e| gyre_err(&e))?;
        }
        gyre.set_constant_str(
            "GYRE_DIR",
            &format!("{}/gyre/gyre", mesa_dir.trim_end()),
        )
        .map_err(|e| gyre_err(&e))?;
        Ok(Self {
            gyre,
            add_center_point: ctrl.add_center_point,
            keep_surface_point: ctrl.keep_surface_point,
            add_atmosphere: ctrl.add_atmosphere,
        })
    }

    /// `store_model_for_gyre(s, add_center_point, keep_surface_point,
    /// add_atmosphere, ierr)` (`gyre_support.f90:131-229`) without the
    /// optional FGONG and GYRE file writes (`write_fgong_for_each_model`
    /// and `write_gyre_for_each_model` are `.false.` by default and are
    /// not accepted by FlexMESA's astero parser).
    ///
    /// # Errors
    ///
    /// Pulsation-data and GYRE model errors.
    pub fn store_model(&mut self, model: &StarModel) -> Result<(), StarError> {
        let (global_data, point_data) = star_get_pulse_data_gyre(
            model,
            self.add_center_point,
            self.keep_surface_point,
            self.add_atmosphere,
        )
        .map_err(|e| {
            StarError::InvalidInput(format!("star_get_pulse_data('GYRE'): {e}"))
        })?;
        self.gyre
            .set_model(
                &global_data,
                &point_data,
                model.controls.gyre_data_schema,
            )
            .map_err(|e| gyre_err(&e))
    }

    /// `do_gyre_get_modes(s, el, store_model, ierr)`
    /// (`gyre_support.f90:75-119`) with `gyre_call_back`
    /// (`gyre_support.f90:231-263`). Returns the results
    /// `store_new_oscillation_results` appends, in call order.
    /// `save_gyre_mode_info` writes nothing unless `save_mode_model_number`
    /// matches the model number; that control keeps its default (0).
    ///
    /// # Errors
    ///
    /// As [`Self::store_model`]; GYRE search errors.
    pub fn get_modes(
        &mut self,
        model: &StarModel,
        el: i32,
        store_model: bool,
    ) -> Result<Vec<OscResult>, StarError> {
        if store_model {
            self.store_model(model)?;
        }
        let mut out = Vec::new();
        let mut cb = |md: &Mode<'_>| -> Result<i32, GyreError> {
            out.push(OscResult {
                el: md.wave.md_p.l,
                order: md.n_pg,
                em: 0,
                inertia: md.e_norm()?,
                cyclic_freq: md.freq("UHZ")?.re,
                growth_rate: md.freq("RAD_PER_SEC")?.im,
            });
            Ok(0)
        };
        self.gyre.get_modes(el, &mut cb).map_err(|e| gyre_err(&e))?;
        Ok(out)
    }
}
