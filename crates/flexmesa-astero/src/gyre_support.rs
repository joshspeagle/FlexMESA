// SPDX-License-Identifier: GPL-3.0-or-later
//! Astero's GYRE interface (`astero/private/gyre_support.f90`): `init_gyre`,
//! `do_gyre_get_modes` with `store_model_for_gyre` and `gyre_call_back`,
//! and the oscillation-result arrays `store_new_oscillation_results` fills
//! (`astero/public/astero_def.f90:673-716`), over the GYRE port in
//! `flexmesa-gyre` (#170, package G3c).

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use flexmesa_const::{CLIGHT, CRAD, LSUN, MSUN, RSUN, STANDARD_CGRAV};
use flexmesa_gyre::mesa::GyreMesa;
use flexmesa_gyre::mode::Mode;
use flexmesa_gyre::GyreError;
use flexmesa_star::controls::StarError;
use flexmesa_star::model::StarModel;
use flexmesa_star::pulse::default_fgong_header;
use flexmesa_star::pulse_fgong::{get_fgong_data, PulseDataInput};
use flexmesa_star::pulse_gyre_data::star_get_pulse_data_gyre;
use flexmesa_star::pulse_write::{
    fortran_1pe, write_fgong_data, write_gyre_data,
};

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
#[allow(clippy::struct_excessive_bools)]
pub struct AsteroGyre {
    /// The GYRE library state.
    pub gyre: GyreMesa,
    add_center_point: bool,
    keep_surface_point: bool,
    add_atmosphere: bool,
    gyre_non_ad: bool,
    write_fgong_for_each_model: bool,
    fgong_prefix: String,
    fgong_postfix: String,
    write_gyre_for_each_model: bool,
    gyre_prefix: String,
    gyre_postfix: String,
    max_num_gyre_points: isize,
    model_num_digits: usize,
    astero_results_dir: PathBuf,
    mode_output_dir: PathBuf,
    save_mode_model_number: isize,
    save_mode_filename: String,
    el_to_save: i32,
    order_to_save: i32,
    trace_time_in_oscillation_code: bool,
    last_oscillation_elapsed_sec: Option<f64>,
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
            gyre_non_ad: ctrl.gyre_non_ad,
            write_fgong_for_each_model: ctrl.write_fgong_for_each_model,
            fgong_prefix: ctrl.fgong_prefix.clone(),
            fgong_postfix: ctrl.fgong_postfix.clone(),
            write_gyre_for_each_model: ctrl.write_gyre_for_each_model,
            gyre_prefix: ctrl.gyre_prefix.clone(),
            gyre_postfix: ctrl.gyre_postfix.clone(),
            max_num_gyre_points: ctrl.max_num_gyre_points,
            model_num_digits: ctrl.model_num_digits,
            astero_results_dir: case_dir.join(&ctrl.astero_results_directory),
            mode_output_dir: case_dir.to_path_buf(),
            save_mode_model_number: ctrl.save_mode_model_number,
            save_mode_filename: ctrl.save_mode_filename.clone(),
            el_to_save: ctrl.el_to_save,
            order_to_save: ctrl.order_to_save,
            trace_time_in_oscillation_code: ctrl.trace_time_in_oscillation_code,
            last_oscillation_elapsed_sec: None,
        })
    }

    /// Redirects per-model FGONG/GYRE and eigenfunction output roots.
    pub fn set_output_root(
        &mut self,
        output_root: &Path,
        results_subdir: &str,
    ) {
        self.mode_output_dir = output_root.to_path_buf();
        self.astero_results_dir = output_root.join(results_subdir);
    }

    /// Configured `gyre_non_ad` control value.
    #[must_use]
    pub const fn gyre_non_ad(&self) -> bool {
        self.gyre_non_ad
    }

    /// Configured `max_num_gyre_points` control value.
    #[must_use]
    pub const fn max_num_gyre_points(&self) -> isize {
        self.max_num_gyre_points
    }

    /// Wall-clock duration in seconds of the most recent GYRE call.
    #[must_use]
    pub const fn last_oscillation_elapsed_sec(&self) -> Option<f64> {
        self.last_oscillation_elapsed_sec
    }

    /// `store_model_for_gyre(s, add_center_point, keep_surface_point,
    /// add_atmosphere, ierr)` (`gyre_support.f90:131-229`) including
    /// `write_fgong_for_each_model` and `write_gyre_for_each_model`.
    ///
    /// # Errors
    ///
    /// Pulsation-data, file-export, and GYRE model errors.
    pub fn store_model(&mut self, model: &StarModel) -> Result<(), StarError> {
        let num_str = format!(
            "{:0width$}",
            model.model_number,
            width = self.model_num_digits
        );
        if self.write_fgong_for_each_model {
            let pulse_in = PulseDataInput::from_model(model).map_err(|e| {
                StarError::InvalidInput(format!(
                    "star_get_pulse_data('FGONG'): {e}"
                ))
            })?;
            let (fg_global, fg_point) = get_fgong_data(
                &pulse_in,
                self.add_center_point,
                self.keep_surface_point,
                self.add_atmosphere,
            )
            .map_err(|e| {
                StarError::InvalidInput(format!("get_fgong_data: {e}"))
            })?;
            fs::create_dir_all(&self.astero_results_dir).map_err(|e| {
                StarError::IoError(format!(
                    "failed creating astero_results_directory '{}': {e}",
                    self.astero_results_dir.display()
                ))
            })?;
            let path = self.astero_results_dir.join(format!(
                "{}{}{}",
                self.fgong_prefix.trim(),
                num_str,
                self.fgong_postfix.trim()
            ));
            let text = write_fgong_data(
                &fg_global,
                &fg_point,
                &default_fgong_header(),
                model.controls.fgong_ivers,
            )
            .map_err(|e| StarError::InvalidInput(e.to_string()))?;
            fs::write(&path, text).map_err(|e| {
                StarError::IoError(format!(
                    "failed writing FGONG file '{}': {e}",
                    path.display()
                ))
            })?;
        }
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
            .map_err(|e| gyre_err(&e))?;
        if self.write_gyre_for_each_model {
            fs::create_dir_all(&self.astero_results_dir).map_err(|e| {
                StarError::IoError(format!(
                    "failed creating astero_results_directory '{}': {e}",
                    self.astero_results_dir.display()
                ))
            })?;
            let path = self.astero_results_dir.join(format!(
                "{}{}{}",
                self.gyre_prefix.trim(),
                num_str,
                self.gyre_postfix.trim()
            ));
            let text = write_gyre_data(
                &global_data,
                &point_data,
                model.controls.gyre_data_schema,
            )
            .map_err(|e| StarError::InvalidInput(e.to_string()))?;
            fs::write(&path, text).map_err(|e| {
                StarError::IoError(format!(
                    "failed writing GYRE file '{}': {e}",
                    path.display()
                ))
            })?;
        }
        Ok(())
    }

    /// `do_gyre_get_modes(s, el, store_model, ierr)`
    /// (`gyre_support.f90:75-119`) with `gyre_call_back`
    /// (`gyre_support.f90:231-263`). Returns the results
    /// `store_new_oscillation_results` appends, in call order, and writes
    /// `save_gyre_mode_info` when `save_mode_model_number`, `el_to_save`, and
    /// `order_to_save` match.
    ///
    /// # Errors
    ///
    /// As [`Self::store_model`]; GYRE search and mode-file errors.
    pub fn get_modes(
        &mut self,
        model: &StarModel,
        el: i32,
        store_model: bool,
    ) -> Result<Vec<OscResult>, StarError> {
        if store_model {
            self.store_model(model)?;
        }
        let save_active = self.save_mode_model_number > 0
            && model.model_number.cast_signed() == self.save_mode_model_number
            && el == self.el_to_save;
        let save_rel = self.save_mode_filename.trim();
        let save_rel = if save_rel.is_empty() {
            "save_mode.data"
        } else {
            save_rel
        };
        let save_path = if Path::new(save_rel).is_absolute() {
            PathBuf::from(save_rel)
        } else {
            self.mode_output_dir.join(save_rel)
        };
        let target_el = self.el_to_save;
        let target_order = self.order_to_save;
        let mut save_err: Option<StarError> = None;
        let mut out = Vec::new();
        let t0 = Instant::now();
        {
            let mut cb = |md: &Mode<'_>| -> Result<i32, GyreError> {
                let rec = OscResult {
                    el: md.wave.md_p.l,
                    order: md.n_pg,
                    em: 0,
                    inertia: md.e_norm()?,
                    cyclic_freq: md.freq("UHZ")?.re,
                    growth_rate: md.freq("RAD_PER_SEC")?.im,
                };
                if save_active
                    && save_err.is_none()
                    && rec.el == target_el
                    && rec.order == target_order
                {
                    let mut buf = String::new();
                    let _ = writeln!(
                        buf,
                        "{:>8}{:>8}{:>8}{:>20}{:>20}{:>20}",
                        "el",
                        "order",
                        "em",
                        "freq (microHz)",
                        "inertia",
                        "growth rate (s)"
                    );
                    let _ = writeln!(
                        buf,
                        "{:8}{:8}{:8}{:20.10}{}{}",
                        rec.el,
                        rec.order,
                        rec.em,
                        rec.cyclic_freq,
                        fortran_1pe(rec.inertia, 20, 10, None),
                        fortran_1pe(rec.growth_rate, 20, 10, None),
                    );
                    if let Some(parent) = save_path.parent() {
                        if let Err(e) = fs::create_dir_all(parent) {
                            save_err = Some(StarError::IoError(format!(
                                "failed creating save_mode_filename dir '{}': {e}",
                                parent.display()
                            )));
                            return Ok(0);
                        }
                    }
                    if let Err(e) = fs::write(&save_path, buf) {
                        save_err = Some(StarError::IoError(format!(
                            "failed writing save_mode_filename '{}': {e}",
                            save_path.display()
                        )));
                    }
                }
                out.push(rec);
                Ok(0)
            };
            self.gyre.get_modes(el, &mut cb).map_err(|e| gyre_err(&e))?;
        }
        let elapsed = t0.elapsed().as_secs_f64();
        self.last_oscillation_elapsed_sec = Some(elapsed);
        if self.trace_time_in_oscillation_code {
            eprintln!("time in gyre: {elapsed:.6} s");
        }
        if let Some(err) = save_err {
            return Err(err);
        }
        Ok(out)
    }
}
