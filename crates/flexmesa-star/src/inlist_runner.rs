// SPDX-License-Identifier: GPL-3.0-or-later
//! Multi-stage `inlist` execution driver (`run_inlist_stage`,
//! `run_test_suite_case`), `run_star_extras` hook trait ([`StarExtras`]),
//! and bit-identical photo restart verification (`verify_photo_restart_parity`).

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::case_staging::TempCaseStaging;
use crate::controls::StarError;
use crate::evolve::{star_evolve_step, star_evolve_step_outcome_with_extras};
use crate::evolve_support::snapshot_step_start_extras;
use crate::history::{extract_history_row, HistoryRow};
use crate::hydro_vars::set_hydro_vars;
use crate::inlist::{InlistBundle, InlistLoadOptions};
use crate::mesa_profile::{StarLogsWriter, StarStepLogExtras};
use crate::model::StarModel;
use crate::model_io::{
    compute_photo_checksum, load_photo_bytes, write_mod_string,
    write_photo_bytes,
};
use crate::photo_io::try_write_photo_bytes;
use crate::rn_parser::{parse_rn_stages, RnStage};
use crate::rsp_eos_kap::RspChemCache;
use crate::star_job_init::{
    init_or_update_model_from_bundle_outcome, InitModelOutcome,
};
use crate::termination::{do_check_limits, LimitCheckOutcome};

/// Control flow directive returned by [`StarExtras`] step hooks (`keep_going`,
/// `redo`, `retry`, `terminate` in `star/public/star_def.inc`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtrasDirective {
    /// Continue normal evolution (`keep_going`).
    KeepGoing,
    /// Re-run the current step at identical timestep (`redo`).
    Redo,
    /// Force a step retry with reduced timestep (`retry`).
    Retry,
    /// Terminate the evolutionary stage with the given reason string (`terminate`).
    Terminate(String),
    /// Abort the stage immediately with a fatal error message (`#147`).
    Error(String),
}

/// Outcome returned by [`StarExtras::other_rsp_build_model`] (`RSP` envelope
/// construction or clean pre-evolution termination as in `rsp_check_2nd_crossing`).
#[derive(Debug, Clone)]
pub enum RspBuildOutcome {
    /// Constructed static `RSP` [`StarModel`].
    Model(Box<StarModel>),
    /// Terminated cleanly during model construction with termination code (`ierr = -1`).
    Terminate(String),
}

/// Rust trait equivalent of upstream MESA's `run_star_extras.f90` hooks and
/// `other_*` physics callbacks.
pub trait StarExtras {
    /// Invoked after `inlist` controls are loaded before `&star_job` relaxations.
    ///
    /// # Errors
    /// Returns [`StarError`] if custom control setup fails.
    fn extras_controls(
        &mut self,
        model: &mut StarModel,
    ) -> Result<(), StarError> {
        install_star_aux_hooks(self, model);
        Ok(())
    }

    /// Invoked once at the start of an evolutionary stage after model loading.
    ///
    /// # Errors
    /// Returns [`StarError`] if custom startup initialization fails.
    fn extras_startup(
        &mut self,
        _m: &mut StarModel,
        _r: bool,
    ) -> Result<(), StarError> {
        Ok(())
    }
    /// Invoked at the start of each evolutionary step before remeshing.
    fn extras_start_step(&mut self, _m: &mut StarModel) -> ExtrasDirective {
        ExtrasDirective::KeepGoing
    }
    /// Invoked immediately after a step converges to check custom limits.
    fn extras_check_model(&mut self, _m: &mut StarModel) -> ExtrasDirective {
        ExtrasDirective::KeepGoing
    }
    /// Optional custom timestep limiter (`use_other_timestep_limit` in `timestep.f90:134-138`),
    /// returning `(ExtrasDirective, dt_limit_ratio)`.
    fn other_timestep_limit(
        &mut self,
        _m: &mut StarModel,
        _skip_hard_limit: bool,
        _dt: f64,
    ) -> (ExtrasDirective, f64) {
        (ExtrasDirective::KeepGoing, 0.0)
    }
    /// Invoked at the end of a step after history extraction.
    fn extras_finish_step(&mut self, _m: &mut StarModel) -> ExtrasDirective {
        ExtrasDirective::KeepGoing
    }
    /// Invoked once after the evolutionary loop terminates.
    ///
    /// # Errors
    /// Returns [`StarError`] if post-run export or verification fails.
    fn extras_after_evolve(
        &mut self,
        _m: &mut StarModel,
    ) -> Result<(), StarError> {
        Ok(())
    }
    /// Custom effective gravitational constant profile $G_{\mathrm{eff}}(k)$.
    ///
    /// # Errors
    /// Returns [`StarError`] if evaluating or validating `model.cgrav` fails.
    fn other_cgrav(&self, _m: &mut StarModel) -> Result<(), StarError> {
        Ok(())
    }
    /// Custom mixing-length parameter profile $\alpha_{\mathrm{MLT}}(k)$.
    fn other_alpha_mlt(&self, _m: &mut StarModel) {}
    /// Custom wind mass-loss rate $\dot{M}$ ($M_\odot\,\mathrm{yr^{-1}}$).
    ///
    /// # Errors
    /// Returns [`StarError`] if evaluating custom wind fails.
    fn other_wind(
        &self,
        _m: &StarModel,
        _l: f64,
        _ms: f64,
        _r: f64,
        _t: f64,
    ) -> Result<Option<f64>, StarError> {
        Ok(None)
    }
    /// Custom adjusted $\dot{M}$ ($\mathrm{g\,s^{-1}}$) or `None` to keep `current_mdot`.
    /// Hooks may also mutate `m` (such as `m.controls.mix_wind.accreted_material_j`,
    /// mirroring `star/other/other_adjust_mdot.f90` and `star/private/winds.f90:44-50`).
    ///
    /// # Errors
    /// Returns [`StarError`] if evaluating custom $\dot{M}$ fails.
    fn other_adjust_mdot(
        &mut self,
        _m: &mut StarModel,
        _current_mdot: f64,
    ) -> Result<Option<f64>, StarError> {
        Ok(None)
    }
    /// Custom specific angular momentum torque $\dot{j}_{\mathrm{extra}}(k)$ ($\mathrm{cm^2\,s^{-2}}$).
    fn other_torque(&self, _m: &StarModel) -> Option<Vec<f64>> {
        None
    }
    /// Populates `model.aux.extra_jdot` and/or `model.aux.extra_omegadot`
    /// inside `do_solve_omega_mix` when `use_other_torque` is enabled
    /// (`other_torque` in `star/private/solve_omega_mix.f90:81-88`).
    ///
    /// By default delegates to [`Self::other_torque`] and copies any returned
    /// profile into `model.aux.extra_jdot`.
    ///
    /// # Errors
    /// Returns [`StarError`] if evaluating or validating the torque fails.
    fn apply_other_torque(
        &mut self,
        m: &mut StarModel,
    ) -> Result<(), StarError> {
        if let Some(jdot) = self.other_torque(m) {
            if jdot.len() != m.nz {
                return Err(StarError::InvalidInput(format!(
                    "other_torque returned {} values for nz = {}",
                    jdot.len(),
                    m.nz
                )));
            }
            m.aux.extra_jdot.resize(m.nz, 0.0);
            m.aux.extra_jdot[..m.nz].copy_from_slice(&jdot);
        }
        Ok(())
    }
    /// Populates implicit torque sources (`extra_jdot`, `extra_omegadot`) and
    /// their $\Omega$-partials (`d_extra_{jdot,omegadot}_domega_{m1,00,p1}`)
    /// inside each Newton iteration of `do_solve_omega_mix` when
    /// `use_other_torque_implicit` is enabled
    /// (`other_torque_implicit` in `star/private/solve_omega_mix.f90:201-208`).
    ///
    /// # Errors
    /// Returns [`StarError`] to trigger an outer-step retry (`solve_omega_mix.f90:204-206`).
    fn other_torque_implicit(
        &mut self,
        _m: &mut StarModel,
    ) -> Result<(), StarError> {
        Ok(())
    }
    /// Custom extra specific energy generation $\epsilon_{\mathrm{extra}}(k)$ ($\mathrm{erg\,g^{-1}\,s^{-1}}$).
    fn other_energy(&self, _m: &StarModel) -> Option<Vec<f64>> {
        None
    }
    /// Optional custom initial model builder (`use_other_build_initial_model`).
    ///
    /// # Errors
    /// Returns [`StarError`] if custom model construction fails.
    fn other_build_initial_model(
        &mut self,
        _bundle: &InlistBundle,
    ) -> Result<Option<StarModel>, StarError> {
        Ok(None)
    }
    /// Optional custom `RSP` model builder (`use_other_rsp_build_model`).
    ///
    /// # Errors
    /// Returns [`StarError`] if custom `RSP` envelope construction fails.
    fn other_rsp_build_model(
        &mut self,
        _bundle: &InlistBundle,
    ) -> Result<Option<RspBuildOutcome>, StarError> {
        Ok(None)
    }
    /// Optional custom `RSP` linear analysis (`use_other_rsp_linear_analysis`).
    ///
    /// Returns `Ok(true)` when the hook populated `lina_*`, `period`, and
    /// initial velocities, or `Ok(false)` when unimplemented.
    ///
    /// # Errors
    /// Returns [`StarError`] if custom linear pulsation analysis fails.
    fn other_rsp_linear_analysis(
        &mut self,
        _model: &mut StarModel,
        _chem: &RspChemCache,
        _log_dir: Option<&Path>,
    ) -> Result<bool, StarError> {
        Ok(false)
    }
    /// Extra per-step history columns appended to `history.data` rows
    /// (`data_for_extra_history_columns`).
    fn extra_history_columns(&self, _m: &StarModel) -> Vec<(String, f64)> {
        Vec::new()
    }
    /// Extra per-zone profile columns appended to `profileN.data` rows
    /// (`how_many_extra_profile_columns` + `data_for_extra_profile_columns`).
    ///
    /// # Errors
    /// Returns [`StarError`] if extra profile column generation fails.
    fn extra_profile_columns(
        &self,
        _m: &StarModel,
    ) -> Result<Vec<(String, Vec<f64>)>, StarError> {
        Ok(Vec::new())
    }
    /// Extra header items emitted when starting a `history.data` log
    /// (`how_many_extra_history_header_items` +
    /// `data_for_extra_history_header_items`).
    ///
    /// # Errors
    /// Returns [`StarError`] if extra history header generation fails.
    fn extra_history_header_items(
        &self,
        _m: &StarModel,
    ) -> Result<Vec<(String, f64)>, StarError> {
        Ok(Vec::new())
    }
    /// Extra header items emitted on each `profileN.data` snapshot
    /// (`how_many_extra_profile_header_items` +
    /// `data_for_extra_profile_header_items`).
    ///
    /// # Errors
    /// Returns [`StarError`] if extra profile header generation fails.
    fn extra_profile_header_items(
        &self,
        _m: &StarModel,
    ) -> Result<Vec<(String, f64)>, StarError> {
        Ok(Vec::new())
    }
    /// Receives the staged temporary case directory path before stage execution
    /// so caller-supplied extras write auxiliary outputs inside the staged copy.
    fn set_staged_case_dir(&mut self, _case_dir: &Path) {}

    /// Optional custom convective overshooting scheme handle installed onto
    /// `model.aux.other_overshooting_scheme` (`other_overshooting_scheme` in
    /// `star/other/other_overshooting_scheme.f90`).
    fn other_overshooting_scheme_handle(
        &self,
    ) -> Option<crate::hydro_state::OtherOvershootingSchemeHandle> {
        None
    }

    /// Optional custom `adjust_mlt_gradT_fraction` handle installed onto
    /// `model.aux.other_adjust_mlt_gradt_fraction`
    /// (`other_adjust_mlt_gradT_fraction` in
    /// `star/other/other_adjust_mlt_gradt_fraction.f90`).
    fn other_adjust_mlt_gradt_fraction_handle(
        &self,
    ) -> Option<crate::hydro_state::OtherAdjustMltGradtFractionHandle> {
        None
    }

    /// Optional custom mesh spacing functions handle installed onto
    /// `model.aux.other_mesh_functions` (`other_mesh_functions` in
    /// `star/other/other_mesh_functions.f90`).
    fn other_mesh_functions_handle(
        &self,
    ) -> Option<crate::mesh_functions::OtherMeshFunctionsHandle> {
        None
    }

    /// Optional custom local mesh multiplier handle installed onto
    /// `model.aux.other_mesh_delta_coeff_factor`
    /// (`other_mesh_delta_coeff_factor` in
    /// `star/other/other_mesh_delta_coeff_factor.f90`).
    fn other_mesh_delta_coeff_factor_handle(
        &self,
    ) -> Option<crate::adjust_mesh_support::OtherMeshDeltaCoeffFactorHandle>
    {
        None
    }
}

/// Installs any `other_overshooting_scheme`, `other_adjust_mlt_gradT_fraction`,
/// `other_mesh_functions`, and `other_mesh_delta_coeff_factor` handles exposed
/// by `extras` onto `model.aux` without clearing handles already installed
/// directly by `extras_controls`.
pub fn install_star_aux_hooks<E: StarExtras + ?Sized>(
    extras: &E,
    model: &mut StarModel,
) {
    if let Some(handle) = extras.other_overshooting_scheme_handle() {
        model.aux.other_overshooting_scheme = Some(handle);
    }
    if let Some(handle) = extras.other_adjust_mlt_gradt_fraction_handle() {
        model.aux.other_adjust_mlt_gradt_fraction = Some(handle);
    }
    if let Some(handle) = extras.other_mesh_functions_handle() {
        model.aux.other_mesh_functions = Some(handle);
    }
    if let Some(handle) = extras.other_mesh_delta_coeff_factor_handle() {
        model.aux.other_mesh_delta_coeff_factor = Some(handle);
    }
}

/// Default no-op [`StarExtras`] implementation (`standard_run_star_extras.inc`).
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultStarExtras;
impl StarExtras for DefaultStarExtras {}

/// Formats the restart `.photo` leaf filename for `model_number` and
/// `photo_digits`, matching `get_name_for_restart_file` and
/// `get_string_for_model_number('x', n, num_digits, num)`
/// (`star/private/star_utils.f90:329-351`).
///
/// Wraps `val = model_number % 10^photo_digits`. When `val == 0` (such as
/// `model_number == 1000` with `photo_digits == 3`), returns the full
/// `model_number` without the `'x'` prefix; otherwise returns `'x'` followed
/// by `val` zero-padded to `photo_digits` digits.
#[must_use]
pub fn photo_restart_filename(
    model_number: usize,
    photo_digits: usize,
) -> String {
    let digits = photo_digits.max(1);
    let modulus = u32::try_from(digits)
        .ok()
        .and_then(|d| 10_u128.checked_pow(d));
    let val = match modulus {
        Some(m) => (model_number as u128) % m,
        None => model_number as u128,
    };
    if val == 0 {
        model_number.to_string()
    } else {
        format!("x{val:0digits$}")
    }
}

/// Checks whether `termination_code` satisfies
/// `&star_job::required_termination_code_string(1:9)`, matching
/// `termination_code_string_okay` (`star/job/run_star_support.f90:828-843`).
///
/// Returns `true` when `termination_code` is empty (`s% termination_code == 0`
/// at `:831`), when all entries of `required` are empty (`:833-834`), or when
/// any non-empty trimmed entry matches `termination_code` (including
/// species-parameterized limits such as `xa_central_lower_limit(h1)` matching
/// `xa_central_lower_limit`).
#[must_use]
pub fn termination_code_string_okay(
    required: &[String],
    termination_code: &str,
) -> bool {
    let term = termination_code.trim();
    if term.is_empty() {
        return true;
    }
    let mut any_non_empty = false;
    for raw in required {
        let req = raw.trim();
        if req.is_empty() {
            continue;
        }
        any_non_empty = true;
        if term == req
            || term
                .strip_prefix(req)
                .is_some_and(|rest| rest.starts_with('('))
        {
            return true;
        }
    }
    !any_non_empty
}

/// Checks whether `abs(cumulative_energy_error / total_energy_end)` exceeds
/// `warn_when_large_rel_run_E_err`, matching
/// `star/job/run_star_support.f90:699-705`.
#[must_use]
pub fn rel_run_e_err_exceeds_warn_limit(model: &StarModel) -> Option<f64> {
    let tot = model.aux.step_hist.total_energy_end;
    if tot == 0.0 {
        return None;
    }
    let rel_err = (model.aux.cumulative_energy_error / tot).abs();
    if rel_err > model.controls.output.warn_when_large_rel_run_e_err {
        Some(rel_err)
    } else {
        None
    }
}

/// Checks whether `abs(1 + total_gravitational_energy_end / virial_thm_P_avg)`
/// exceeds `warn_when_large_virial_thm_rel_err` for static non-rotating models
/// without `u`/`v`, mass corrections, nonzero inner core mass, or mass
/// loss/accretion, matching `star/job/run_star_support.f90:707-715`.
#[must_use]
pub fn virial_thm_rel_err_exceeds_warn_limit(model: &StarModel) -> Option<f64> {
    let c = &model.controls;
    if c.rotation_flag
        || c.u_flag
        || c.use_mass_corrections
        || c.v_flag
        || model.m_center > 0.0
        || model.mstar_dot != 0.0
    {
        return None;
    }
    let virial = model.aux.step_hist.virial_thm_p_avg;
    if virial == 0.0 || !virial.is_finite() {
        return None;
    }
    let rel_err = (1.0
        + model.aux.step_hist.total_gravitational_energy_end / virial)
        .abs();
    if rel_err > c.output.warn_when_large_virial_thm_rel_err {
        Some(rel_err)
    } else {
        None
    }
}

/// Writes a binary `.photo` checkpoint inside `case_dir` using
/// [`try_write_photo_bytes`], matching `star_write_photo`
/// (`star/public/star_lib.f90:492-502`), `output`
/// (`star/private/evolve_support.f90:243-272`), and `output_to_file`
/// (`star/private/evolve_support.f90:275-303`).
///
/// When `explicit_filename` is empty, writes
/// `<case_dir>/<photo_directory>/<photo_restart_filename(model_number, photo_digits)>`.
/// When `explicit_filename` is non-empty, writes `<case_dir>/<explicit_filename>`.
/// Returns `Ok(None)` if `case_dir` no longer exists (for example, when an
/// in-memory benchmark caller already dropped its temporary staging directory).
fn write_photo_to_case_dir(
    model: &StarModel,
    case_dir: &Path,
    explicit_filename: &str,
) -> Result<Option<PathBuf>, StarError> {
    if !case_dir.exists() {
        return Ok(None);
    }
    if !case_dir.is_dir() {
        return Err(StarError::IoError(format!(
            "photo output case_dir '{}' is not a directory",
            case_dir.display()
        )));
    }
    let trimmed_explicit = explicit_filename.trim();
    let out_path = if trimmed_explicit.is_empty() {
        let photo_dir = model.controls.output.photo_directory.trim();
        if photo_dir.is_empty() {
            return Err(StarError::InvalidInput(String::from(
                "'&controls::photo_directory' must be non-empty when writing photos",
            )));
        }
        let dir_path = case_dir.join(photo_dir);
        fs::create_dir_all(&dir_path).map_err(|e| {
            StarError::IoError(format!(
                "failed to create photo_directory {}: {e}",
                dir_path.display()
            ))
        })?;
        let leaf = photo_restart_filename(
            model.model_number,
            model.controls.output.photo_digits,
        );
        dir_path.join(leaf)
    } else {
        let target = case_dir.join(trimmed_explicit);
        if let Some(parent) = target.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                fs::create_dir_all(parent).map_err(|e| {
                    StarError::IoError(format!(
                        "failed to create parent directory for {}: {e}",
                        target.display()
                    ))
                })?;
            }
        }
        target
    };
    let bytes = try_write_photo_bytes(model)?;
    fs::write(&out_path, bytes).map_err(|e| {
        StarError::IoError(format!(
            "failed to write photo {}: {e}",
            out_path.display()
        ))
    })?;
    Ok(Some(out_path))
}

fn write_mod_if_missing(
    model: &StarModel,
    out_path: &Path,
) -> Result<(), StarError> {
    if !out_path.exists() || model.controls.rsp_flag {
        let mod_text = write_mod_string(model)?;
        fs::write(out_path, mod_text).map_err(|e| {
            StarError::IoError(format!(
                "failed to write {}: {e}",
                out_path.display()
            ))
        })?;
    }
    Ok(())
}

/// Telemetry and final state returned by a single `inlist` stage execution.
#[derive(Debug, Clone)]
pub struct StageRunReport {
    /// Entry `inlist` name executed in this stage.
    pub inlist_name: String,
    /// Number of evolutionary steps completed in this stage.
    pub steps_taken: usize,
    /// Total Henyey Newton-Raphson iterations across all steps in this stage.
    pub total_newton_iterations: usize,
    /// Total step retries across all steps in this stage.
    pub total_retries: usize,
    /// Termination code that ended the stage (e.g. `"max_age"`,
    /// `"xa_central_lower_limit"`, `"max_model_number"`).
    pub termination_code: String,
    /// Wall-clock runtime of the stage in seconds (`s`).
    pub runtime_sec: f64,
    /// Final [`HistoryRow`] summary at the end of the stage.
    pub final_history: HistoryRow,
    /// Complete sequence of [`HistoryRow`] records across the stage.
    pub history_rows: Vec<HistoryRow>,
}

/// Telemetry and stage reports returned by a full multi-stage `test_suite` case.
#[derive(Debug, Clone)]
pub struct CaseRunReport {
    /// Case directory name (e.g. `"1M_pre_ms_to_wd"`).
    pub case_name: String,
    /// Suite tier number (`1` for Tier 1 cases, `0` otherwise).
    pub tier: usize,
    /// Whether the case passed execution, physical assertions, and restart parity.
    pub passed: bool,
    /// Final stage termination code string.
    pub termination_code: String,
    /// Per-stage execution reports in chronological order.
    pub stages: Vec<StageRunReport>,
    /// Total evolutionary steps across all stages.
    pub total_steps: usize,
    /// Total Newton-Raphson iterations across all stages.
    pub total_newton_iterations: usize,
    /// Total retries across all stages.
    pub total_retries: usize,
    /// Total wall-clock runtime in seconds (`s`).
    pub total_runtime_sec: f64,
    /// Total wall-clock runtime in milliseconds (`ms`).
    pub wall_time_ms: f64,
    /// Final effective temperature $T_{\mathrm{eff}}$ (`K`).
    pub final_teff: f64,
    /// Final $\log_{10}(L / L_\odot)$.
    pub final_log_l: f64,
    /// Final $\log_{10}(R / R_\odot)$.
    pub final_log_r: f64,
    /// Final central $\log_{10} T_c$ (`K`).
    pub final_log_tc: f64,
    /// Final central $\log_{10} \rho_c$ ($\mathrm{g\,cm^{-3}}$).
    pub final_log_rhoc: f64,
    /// Final central ${}^1\mathrm{H}$ mass fraction $X_{\mathrm{c,H1}}$.
    pub final_center_h1: f64,
    /// Final central ${}^4\mathrm{He}$ mass fraction $X_{\mathrm{c,He4}}$.
    pub final_center_he4: f64,
    /// Whether bit-identical `.photo` restart parity passed.
    pub photo_restart_verified: bool,
    /// Whether post-run physical assertions passed.
    pub assertions_passed: bool,
}

/// Configuration for incremental MESA `LOGS/` file emission (`history.data`,
/// `profileN.data`, `profiles.index`) during stage and `test_suite` execution (`#163`).
#[derive(Debug, Clone, Copy, Default)]
pub struct StarLogsOptions<'a> {
    /// Optional root directory under which `<log_directory>/` files are written.
    pub output_root: Option<&'a Path>,
    /// Optional explicit output file path for `history.data` (used by `star_case_trace`).
    pub explicit_history_file: Option<&'a Path>,
    /// Optional override for `&controls` `history_interval` (`Some(1)` writes every step).
    pub history_interval_override: Option<usize>,
    /// Optional override for `&controls` `write_profiles_flag`.
    pub write_profiles_override: Option<bool>,
}

impl<'a> StarLogsOptions<'a> {
    /// Constructs [`StarLogsOptions`] targeting either a `LOGS/` root directory
    /// (`is_logs_dir = true`) or an explicit `history.data` path (`false`).
    #[must_use]
    pub fn from_target(
        out_target: &'a Path,
        is_logs_dir: bool,
        respect_intervals: bool,
        write_profiles: bool,
    ) -> Self {
        let history_interval_override = (!respect_intervals).then_some(1);
        if is_logs_dir {
            Self {
                output_root: Some(out_target),
                explicit_history_file: None,
                history_interval_override,
                write_profiles_override: Some(write_profiles),
            }
        } else {
            Self {
                output_root: None,
                explicit_history_file: Some(out_target),
                history_interval_override,
                write_profiles_override: Some(false),
            }
        }
    }

    /// Builds the per-stage [`StarLogsWriter`] configured by `self` (if any).
    ///
    /// # Errors
    /// Returns [`StarError`] if column resolution or output directory setup fails.
    pub fn build_writer(
        &self,
        model: &StarModel,
        case_dir: &Path,
        stage_logs_dir: Option<&str>,
    ) -> Result<Option<StarLogsWriter>, StarError> {
        self.build_writer_with_extras(
            model,
            case_dir,
            stage_logs_dir,
            &DefaultStarExtras,
        )
    }

    /// Builds the per-stage [`StarLogsWriter`] configured by `self`, forwarding
    /// `extras` into `profile_starting_model` if enabled.
    ///
    /// # Errors
    /// Returns [`StarError`] if column resolution, extra profile hook calls, or
    /// output directory setup fails.
    pub fn build_writer_with_extras(
        &self,
        model: &StarModel,
        case_dir: &Path,
        stage_logs_dir: Option<&str>,
        extras: &dyn StarExtras,
    ) -> Result<Option<StarLogsWriter>, StarError> {
        if let Some(exp_hist) = self.explicit_history_file {
            Ok(Some(StarLogsWriter::for_explicit_history_file(
                model,
                case_dir,
                exp_hist.to_path_buf(),
                self.history_interval_override,
            )?))
        } else if let Some(out_root) = self.output_root {
            Ok(Some(
                StarLogsWriter::for_stage_with_logs_dir_and_profile_extras(
                    model,
                    case_dir,
                    out_root,
                    stage_logs_dir,
                    self.history_interval_override,
                    self.write_profiles_override,
                    |m| {
                        Ok((
                            extras.extra_profile_header_items(m)?,
                            extras.extra_profile_columns(m)?,
                        ))
                    },
                )?,
            ))
        } else {
            Ok(None)
        }
    }
}

fn record_logs_step_with_extras(
    lw: &mut StarLogsWriter,
    model: &mut StarModel,
    extras: &dyn StarExtras,
    is_first_step: bool,
    is_terminating: bool,
) -> Result<(), StarError> {
    let hist_due = lw.is_history_due(model, is_first_step, is_terminating);
    let prof_due = lw.is_profile_due(model, is_first_step, is_terminating);
    if !hist_due && !prof_due {
        return Ok(());
    }
    if hist_due {
        lw.sync_history_target(model);
    }
    let hist_hdr = if hist_due && lw.needs_history_header() {
        extras.extra_history_header_items(model)?
    } else {
        Vec::new()
    };
    let hist_cols = if hist_due {
        extras.extra_history_columns(model)
    } else {
        Vec::new()
    };
    let prof_hdr = if prof_due {
        extras.extra_profile_header_items(model)?
    } else {
        Vec::new()
    };
    let prof_cols = if prof_due {
        extras.extra_profile_columns(model)?
    } else {
        Vec::new()
    };
    lw.record_step_with_extras(
        model,
        is_first_step,
        is_terminating,
        StarStepLogExtras {
            history_header_items: &hist_hdr,
            history_columns: &hist_cols,
            profile_header_items: &prof_hdr,
            profile_columns: &prof_cols,
        },
    )
}

/// Executes a single `inlist` stage on `model`, stepping until either a
/// built-in `&controls` termination limit, an [`ExtrasDirective::Terminate`],
/// or `max_steps_cap` is reached.
///
/// # Errors
/// Returns [`StarError`] if a step fails to converge after all retries or if
/// saving the final `.mod` file fails.
pub fn run_inlist_stage(
    model: &mut StarModel,
    bundle: &InlistBundle,
    extras: &mut dyn StarExtras,
    max_steps_cap: Option<usize>,
) -> Result<StageRunReport, StarError> {
    run_inlist_stage_with_logs(model, bundle, extras, max_steps_cap, None)
}

/// Executes a single `inlist` stage on `model`, optionally streaming incremental
/// MESA-format `LOGS/history.data`, `LOGS/profileN.data`, and `LOGS/profiles.index`
/// through `logs_writer` (`#163`).
///
/// # Errors
/// Returns [`StarError`] if step evolution, log emission, or final `.mod` saving fails.
pub fn run_inlist_stage_with_logs(
    model: &mut StarModel,
    bundle: &InlistBundle,
    extras: &mut dyn StarExtras,
    max_steps_cap: Option<usize>,
    logs_writer: Option<&mut StarLogsWriter>,
) -> Result<StageRunReport, StarError> {
    let (rep, _) = run_inlist_stage_with_logs_inner(
        model,
        bundle,
        extras,
        max_steps_cap,
        logs_writer,
    )?;
    Ok(rep)
}

fn run_inlist_stage_with_logs_inner(
    model: &mut StarModel,
    bundle: &InlistBundle,
    extras: &mut dyn StarExtras,
    max_steps_cap: Option<usize>,
    mut logs_writer: Option<&mut StarLogsWriter>,
) -> Result<(StageRunReport, bool), StarError> {
    let _net_dir_guard = crate::micro::NetSearchDirGuard::new(&bundle.case_dir);
    if logs_writer.is_none() && model.controls.colors.use_colors {
        let _ = crate::mesa_history::load_star_colors_context(
            model,
            Some(&bundle.case_dir),
        )?;
    }
    install_star_aux_hooks(extras, model);
    let t0 = Instant::now();
    let (mut steps_taken, mut total_newton_iterations, mut total_retries) =
        (0, 0, 0);
    let mut history_rows = Vec::new();
    let mut termination_code = String::from("max_steps_cap");
    let mut terminated_by_extras = false;
    let mut last_default_photo_model: Option<usize> = None;

    let job_step_cap = (model.job.steps_to_take_before_terminate >= 0)
        .then_some(model.job.steps_to_take_before_terminate as usize);
    let cap = max_steps_cap
        .into_iter()
        .chain(job_step_cap)
        .min()
        .unwrap_or(50_000);
    if job_step_cap.is_some_and(|b| b <= cap) {
        termination_code = String::from("steps_to_take_before_terminate");
    }
    let do_remesh =
        model.controls.okay_to_remesh && model.controls.mesh_delta_coeff > 0.0;
    let op_split_burn = model.controls.op_split_burn;

    let init_hr = extract_history_row(model);
    if cap == 0 {
        if job_step_cap != Some(0) {
            if let LimitCheckOutcome::Terminate { code, .. } =
                do_check_limits(model, None, true)?
            {
                termination_code = code;
            }
        }
        if let Some(lw) = logs_writer.as_deref_mut() {
            record_logs_step_with_extras(lw, model, extras, true, true)?;
        }
    } else {
        for _ in 0..cap {
            // `before_step_loop` (`star/job/run_star_support.f90:567-577`)
            // runs before `extras_start_step` (`:191-194`).
            if model.job.enable_adaptive_network {
                crate::adjust_net::check_adjust_net(model, &bundle.case_dir)?;
            }
            // `run_star_support.f90:579-582`.
            if model.job.auto_extend_net {
                crate::extend_net::extend_net(model, &bundle.case_dir)?;
            }
            // `before_step_loop` (`star/job/run_star_support.f90:636-640`):
            // reset cumulative energy error before each step while `star_age`
            // is below `set_cumulative_energy_error_each_step_if_age_less_than`.
            if model.star_age
                < model
                    .job
                    .set_cumulative_energy_error_each_step_if_age_less_than
            {
                model.aux.cumulative_energy_error =
                    model.job.new_cumulative_energy_error;
            }
            match extras.extras_start_step(model) {
                ExtrasDirective::Terminate(reason) => {
                    termination_code = reason;
                    terminated_by_extras = true;
                    break;
                }
                ExtrasDirective::Error(msg) => {
                    return Err(StarError::ConvergenceFailure(msg));
                }
                ExtrasDirective::Retry | ExtrasDirective::Redo => {
                    return Err(StarError::ConvergenceFailure(String::from(
                        "extras step retry/redo is not yet implemented (#152)",
                    )));
                }
                ExtrasDirective::KeepGoing => {}
            }
            if model.controls.use_other_cgrav {
                extras.other_cgrav(model)?;
            }
            if model.controls.use_other_alpha_mlt {
                extras.other_alpha_mlt(model);
            }

            let (step_rep, check_term_opt) =
                match star_evolve_step_outcome_with_extras(
                    model,
                    do_remesh,
                    op_split_burn,
                    true,
                    extras,
                ) {
                    Ok(pair) => pair,
                    Err(StarError::Terminated { reason, .. }) => {
                        termination_code = reason;
                        break;
                    }
                    Err(e) => return Err(e),
                };
            steps_taken += 1;
            total_newton_iterations += step_rep.iterations;
            total_retries += step_rep.retries;

            // `after_step_loop` (`star/job/run_star_support.f90:689-692`):
            // reset `cumulative_energy_error` before `rel_run_E_err` checks,
            // `extras_finish_step`, and history/profile output.
            if i64::try_from(model.model_number).is_ok_and(|m| {
                m == model.job.set_cumulative_energy_error_at_step
            }) {
                model.aux.cumulative_energy_error =
                    model.job.new_cumulative_energy_error;
            }

            let in_step_termination =
                model.aux.step_hist.termination_code.clone();
            let finish_dir = extras.extras_finish_step(model);
            model.mstar_start = model.mstar;
            if !model.controls.rsp_flag {
                model.sync_mesh_and_pack()?;
                set_hydro_vars(model, false, false)?;
                crate::struct_burn_mix::refresh_mixing_state(
                    model,
                    model.dt,
                    &[],
                )?;
            }
            let hr = extract_history_row(model);
            history_rows.push(hr);

            let mut will_terminate = false;
            if let Some(code) = in_step_termination {
                termination_code = code;
                will_terminate = true;
            } else if let Some(code) = check_term_opt {
                termination_code = code;
                will_terminate = true;
                terminated_by_extras = true;
            }
            match finish_dir {
                ExtrasDirective::Error(msg) => {
                    return Err(StarError::ConvergenceFailure(msg));
                }
                ExtrasDirective::Retry | ExtrasDirective::Redo => {
                    return Err(StarError::ConvergenceFailure(String::from(
                        "extras_finish_step must return keep_going or terminate (run_star_support.f90:249)",
                    )));
                }
                ExtrasDirective::Terminate(reason) => {
                    if !will_terminate {
                        termination_code = reason;
                        will_terminate = true;
                        terminated_by_extras = true;
                    }
                }
                ExtrasDirective::KeepGoing => {}
            }
            if !will_terminate {
                if let LimitCheckOutcome::Terminate { code, .. } =
                    do_check_limits(model, None, false)?
                {
                    termination_code = code;
                    will_terminate = true;
                }
            }
            let is_last_cap_step = steps_taken == cap;
            let is_terminating = will_terminate || is_last_cap_step;
            if model.controls.rsp_flag && model.controls.rsp.write_map {
                let map_log_dir = logs_writer
                    .as_deref()
                    .filter(|lw| !lw.is_explicit_history_file())
                    .map(|lw| lw.log_dir().to_path_buf());
                crate::rsp_driver::rsp_after_step_map(
                    model,
                    &bundle.case_dir,
                    map_log_dir.as_deref(),
                    extras,
                )?;
            }
            if let Some(lw) = logs_writer.as_deref_mut() {
                record_logs_step_with_extras(
                    lw,
                    model,
                    extras,
                    steps_taken == 1,
                    is_terminating,
                )?;
            }
            if !model.controls.rsp_flag {
                model.save_start_values()?;
                snapshot_step_start_extras(model);
            }
            // Periodic restart photo output (`star/private/evolve.f90:2340-2361`).
            let photo_interval = model.controls.output.photo_interval;
            if photo_interval > 0
                && model.model_number.is_multiple_of(photo_interval as usize)
            {
                write_photo_to_case_dir(model, &bundle.case_dir, "")?;
                last_default_photo_model = Some(model.model_number);
            }
            // Mid-run `save_photo_number` save (`star/job/run_star_support.f90:263,1276-1281`).
            if !is_terminating
                && model.job.save_photo_number >= 0
                && model.model_number == model.job.save_photo_number as usize
            {
                let fname = model.job.save_photo_filename.trim();
                if !fname.is_empty()
                    || last_default_photo_model != Some(model.model_number)
                {
                    write_photo_to_case_dir(model, &bundle.case_dir, fname)?;
                    if fname.is_empty() {
                        last_default_photo_model = Some(model.model_number);
                    }
                }
            }
            if let Some(rel_err) = rel_run_e_err_exceeds_warn_limit(model) {
                eprintln!(
                    "WARNING: rel_run_E_err  {}  {:.16e}",
                    model.model_number, rel_err
                );
            }
            if let Some(rel_err) = virial_thm_rel_err_exceeds_warn_limit(model)
            {
                eprintln!(
                    "WARNING: virial_thm_rel_err  {}  {:.16e}",
                    model.model_number, rel_err
                );
            }
            if will_terminate {
                break;
            }
            // `adjust_tau_factor`, `do_rotation_near_zams` and `do_rotation`
            // after `star_finish_step` returned `keep_going`
            // (`star/job/run_star_support.f90:752-763`).
            crate::job_rotation::after_step_rotation_and_tau(model, extras)?;
        }
    }

    extras.extras_after_evolve(model)?;

    let abnormal_termination = matches!(
        termination_code.as_str(),
        "min_timestep_limit"
            | "max_number_retries"
            | "redo_limit"
            | "dt_is_zero"
            | "varcontrol_target < min_allowed_varcontrol_target"
            | "invalid dt_next in timestep_controller"
    );
    let term_str_ok = termination_code_string_okay(
        &model.job.required_termination_code_string,
        &termination_code,
    );

    // Termination photo save (`star/job/run_star_support.f90:789-790,823,1276-1281`).
    let save_photo_at_end = (model.job.save_photo_when_terminate
        && term_str_ok)
        || (model.job.save_photo_number >= 0
            && model.model_number == model.job.save_photo_number as usize);
    if !abnormal_termination && save_photo_at_end {
        let fname = model.job.save_photo_filename.trim();
        if !fname.is_empty()
            || last_default_photo_model != Some(model.model_number)
        {
            write_photo_to_case_dir(model, &bundle.case_dir, fname)?;
        }
    }

    // Termination model save (`star/job/run_star_support.f90:791-792,823,1270-1274`).
    if !abnormal_termination
        && term_str_ok
        && model.job.save_model_when_terminate
        && !model.job.save_model_filename.is_empty()
        && !model
            .job
            .save_model_filename
            .eq_ignore_ascii_case("undefined")
    {
        let out_path = bundle.case_dir.join(&model.job.save_model_filename);
        write_mod_if_missing(model, &out_path)?;
    }

    let final_history = history_rows.last().copied().unwrap_or(init_hr);

    Ok((
        StageRunReport {
            inlist_name: bundle.entry_inlist.clone(),
            steps_taken,
            total_newton_iterations,
            total_retries,
            termination_code,
            runtime_sec: t0.elapsed().as_secs_f64(),
            final_history,
            history_rows,
        },
        terminated_by_extras,
    ))
}

/// Verifies bit-identical `.photo` restart parity on `model`:
/// 1. Clones `model`, advances 1 warm-up step if `lnd_old` is unpopulated so
///    H211 filter memory (`dt_limit_ratio_old`) and `*_old` arrays are active,
///    then serializes `.photo` bytes and deserializes a fresh `restarted` model.
/// 2. Evolves both `continued` and `restarted` for `n_verify_steps` steps.
/// 3. Asserts `compute_photo_checksum` of both serialized states is equal.
///
/// # Errors
/// Returns [`StarError`] if `.photo` serialization, step evolution, or
/// bit-identical checksum comparison fails.
pub fn verify_photo_restart_parity(
    model: &StarModel,
    n_verify_steps: usize,
) -> Result<(u64, u64), StarError> {
    let mut continued = model.clone();
    // Clear age-stop limits on the cloned parity models so a model that ended
    // at `max_age`, `max_age_in_days`, or `max_age_in_seconds` can still take
    // `n_verify_steps` continuation steps instead of immediately returning
    // `StarError::Terminated`.
    continued.controls.max_age = -1.0;
    continued.controls.ts_stop.max_age_in_days = -1.0;
    continued.controls.max_age_in_seconds = None;
    continued
        .controls
        .ts_stop
        .num_adjusted_dt_steps_before_max_age = 0;
    continued.aux.step_hist.using_revised_max_yr_dt = false;
    if continued.dt_next <= 0.0 {
        let fallback_dt = continued.dt.max(continued.dt_old);
        if fallback_dt > 0.0 {
            continued.dt_next = fallback_dt;
        }
    }
    let do_remesh = continued.controls.okay_to_remesh
        && continued.controls.mesh_delta_coeff > 0.0;
    let op_split = continued.controls.op_split_burn;
    if continued.lnd_old.is_empty() || continued.doing_first_model_of_run {
        star_evolve_step(&mut continued, do_remesh, op_split, true)?;
    }

    let photo_bytes = write_photo_bytes(&continued);
    let mut restarted = load_photo_bytes(&photo_bytes)?;
    // Re-attach non-serialized `other_*` callback handles onto the reloaded
    // photo model (matching MESA's `extras_controls` hook re-registration on
    // restart; `.photo` files serialize numerical state only).
    restarted
        .aux
        .tidal_deformation
        .clone_from(&continued.aux.tidal_deformation);
    restarted
        .aux
        .binary_other_torque
        .clone_from(&continued.aux.binary_other_torque);
    restarted
        .aux
        .binary_other_torque_implicit
        .clone_from(&continued.aux.binary_other_torque_implicit);
    restarted
        .aux
        .other_overshooting_scheme
        .clone_from(&continued.aux.other_overshooting_scheme);
    restarted
        .aux
        .other_adjust_mlt_gradt_fraction
        .clone_from(&continued.aux.other_adjust_mlt_gradt_fraction);
    restarted
        .aux
        .other_mesh_functions
        .clone_from(&continued.aux.other_mesh_functions);
    restarted
        .aux
        .other_mesh_delta_coeff_factor
        .clone_from(&continued.aux.other_mesh_delta_coeff_factor);

    for _ in 0..n_verify_steps {
        star_evolve_step(&mut continued, do_remesh, op_split, true)?;
        star_evolve_step(&mut restarted, do_remesh, op_split, true)?;
    }

    let (b_cont, b_rest) =
        (write_photo_bytes(&continued), write_photo_bytes(&restarted));
    let (c_cont, c_rest) = (
        compute_photo_checksum(&b_cont),
        compute_photo_checksum(&b_rest),
    );
    if c_cont != c_rest {
        let diff_at = b_cont
            .iter()
            .zip(&b_rest)
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| b_cont.len().min(b_rest.len()));
        return Err(StarError::IoError(format!(
            "photo restart parity mismatch at byte {diff_at}: {c_cont:#x} != {c_rest:#x}"
        )));
    }
    Ok((c_cont, c_rest))
}

/// Executes all stages of an upstream MESA `test_suite` case directory
/// according to its `./rn` script (`skip_optional` controls whether
/// `MESA_SKIP_OPTIONAL` branches are skipped).
///
/// # Errors
/// Returns [`StarError`] if `inlist` parsing or stage execution fails.
pub fn run_test_suite_case(
    orig_case_dir: &Path,
    skip_optional: bool,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
) -> Result<CaseRunReport, StarError> {
    run_test_suite_case_with_logs(
        orig_case_dir,
        skip_optional,
        max_steps_per_stage,
        verify_restart,
        StarLogsOptions::default(),
    )
}

/// Executes all stages of an upstream MESA `test_suite` case directory while
/// optionally streaming incremental MESA-format `LOGS/history.data`,
/// `LOGS/profileN.data`, and `LOGS/profiles.index` files according to `logs_opts` (`#163`).
///
/// Uses default strict inlist validation (`allow_unsupported_controls = false`).
///
/// # Errors
/// Returns [`StarError`] if `inlist` parsing, stage execution, or log output fails.
pub fn run_test_suite_case_with_logs(
    orig_case_dir: &Path,
    skip_optional: bool,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
    logs_opts: StarLogsOptions<'_>,
) -> Result<CaseRunReport, StarError> {
    run_test_suite_case_with_options(
        orig_case_dir,
        skip_optional,
        max_steps_per_stage,
        verify_restart,
        logs_opts,
        InlistLoadOptions::default(),
    )
}

/// Executes all stages of an upstream MESA `test_suite` case directory with
/// explicit `logs_opts` and `load_opts` (`#147`).
///
/// # Errors
/// Returns [`StarError`] if `inlist` parsing, stage execution, or log output fails.
pub fn run_test_suite_case_with_options(
    orig_case_dir: &Path,
    skip_optional: bool,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
    logs_opts: StarLogsOptions<'_>,
    load_opts: InlistLoadOptions,
) -> Result<CaseRunReport, StarError> {
    let mut extras = DefaultStarExtras;
    run_test_suite_case_with_extras(
        orig_case_dir,
        skip_optional,
        max_steps_per_stage,
        verify_restart,
        logs_opts,
        load_opts,
        &mut extras,
    )
}

/// Executes all stages of an upstream MESA `test_suite` case directory with
/// explicit `logs_opts`, `load_opts`, and caller-supplied [`StarExtras`] hooks.
///
/// # Errors
/// Returns [`StarError`] if `inlist` parsing, stage execution, or log output fails.
pub fn run_test_suite_case_with_extras(
    orig_case_dir: &Path,
    skip_optional: bool,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
    logs_opts: StarLogsOptions<'_>,
    load_opts: InlistLoadOptions,
    extras: &mut dyn StarExtras,
) -> Result<CaseRunReport, StarError> {
    let t0 = Instant::now();
    let staging = TempCaseStaging::new(orig_case_dir)?;
    let case_dir = staging.staged_dir.as_path();
    let _net_dir_guard = crate::micro::NetSearchDirGuard::new(case_dir);
    extras.set_staged_case_dir(case_dir);
    let case_name = case_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    let rn_stages = parse_rn_stages(case_dir, skip_optional)?;

    let mut current_model: Option<StarModel> = None;
    let mut stage_reports = Vec::with_capacity(rn_stages.len());
    let (mut total_steps, mut total_newton, mut total_retries) = (0, 0, 0);
    let mut photo_restart_verified = false;
    let mut terminated_in_init = false;
    let mut last_stage_terminated_by_extras = false;
    let mut all_required_terminations_ok = true;

    for stage in &rn_stages {
        match stage {
            RnStage::CopyCheckpoint { src_mod, dst_mod } => {
                current_model = None;
                last_stage_terminated_by_extras = false;
                let (src, dst) =
                    (case_dir.join(src_mod), case_dir.join(dst_mod));
                if !src.exists() {
                    return Err(StarError::IoError(format!(
                        "missing checkpoint '{}' for copy to '{}'",
                        src.display(),
                        dst.display()
                    )));
                }
                fs::copy(&src, &dst).map_err(|e| {
                    StarError::IoError(format!(
                        "failed to copy checkpoint {} -> {}: {e}",
                        src.display(),
                        dst.display()
                    ))
                })?;
            }
            RnStage::DoOne {
                inlist,
                output_mod,
                logs_dir,
            } => {
                let bundle = InlistBundle::from_file_in_case_dir_with_options(
                    case_dir, inlist, load_opts,
                )?;
                // Remove stale stage output `.mod` before execution so a
                // `required_termination_code_string` mismatch cannot mask a
                // withheld save (`star/test_suite/test_suite_helpers:22`).
                if bundle.job.save_model_when_terminate {
                    if let Some(out_name) = output_mod {
                        let out_path = case_dir.join(out_name);
                        if out_path.is_file() {
                            fs::remove_file(&out_path).map_err(|e| {
                                StarError::IoError(format!(
                                    "failed to remove stale {}: {e}",
                                    out_path.display()
                                ))
                            })?;
                        }
                    }
                }

                let mut model = match init_or_update_model_from_bundle_outcome(
                    &bundle,
                    current_model.take(),
                    extras,
                )? {
                    InitModelOutcome::Model(m) => *m,
                    InitModelOutcome::Terminate(reason) => {
                        terminated_in_init = true;
                        if !termination_code_string_okay(
                            &bundle.job.required_termination_code_string,
                            &reason,
                        ) {
                            all_required_terminations_ok = false;
                        }
                        stage_reports.push(StageRunReport {
                            inlist_name: bundle.entry_inlist.clone(),
                            steps_taken: 0,
                            total_newton_iterations: 0,
                            total_retries: 0,
                            termination_code: reason,
                            runtime_sec: t0.elapsed().as_secs_f64(),
                            final_history: HistoryRow::default(),
                            history_rows: Vec::new(),
                        });
                        break;
                    }
                };

                let mut stage_logs = logs_opts.build_writer_with_extras(
                    &model,
                    case_dir,
                    logs_dir.as_deref(),
                    extras,
                )?;

                let (rep, stage_terminated_by_extras) =
                    run_inlist_stage_with_logs_inner(
                        &mut model,
                        &bundle,
                        extras,
                        max_steps_per_stage,
                        stage_logs.as_mut(),
                    )?;
                last_stage_terminated_by_extras = stage_terminated_by_extras;

                let stage_abnormal = matches!(
                    rep.termination_code.as_str(),
                    "min_timestep_limit"
                        | "max_number_retries"
                        | "redo_limit"
                        | "dt_is_zero"
                        | "varcontrol_target < min_allowed_varcontrol_target"
                        | "invalid dt_next in timestep_controller"
                );
                let stage_term_ok = termination_code_string_okay(
                    &model.job.required_termination_code_string,
                    &rep.termination_code,
                );
                if !stage_term_ok {
                    all_required_terminations_ok = false;
                }
                let is_capped_smoke = max_steps_per_stage.is_some()
                    && rep.termination_code == "max_steps_cap";
                let stage_term_mismatch = !stage_term_ok && !is_capped_smoke;
                if verify_restart && !photo_restart_verified && !stage_abnormal
                {
                    let (c1, c2) = verify_photo_restart_parity(&model, 1)?;
                    photo_restart_verified = c1 == c2;
                }
                if !stage_abnormal && !stage_term_mismatch {
                    if let Some(out_name) = output_mod {
                        let out_path = case_dir.join(out_name);
                        write_mod_if_missing(&model, &out_path)?;
                    }
                }
                total_steps += rep.steps_taken;
                total_newton += rep.total_newton_iterations;
                total_retries += rep.total_retries;
                stage_reports.push(rep);
                current_model = Some(model);
                if stage_abnormal || stage_term_mismatch {
                    break;
                }
            }
        }
    }

    let runtime_sec = t0.elapsed().as_secs_f64();
    let assertions_passed = terminated_in_init
        || current_model
            .as_ref()
            .is_some_and(|m| m.nz >= 2 && m.teff.is_finite() && m.teff > 0.0);
    let final_hr = stage_reports
        .last()
        .map(|s| s.final_history)
        .or_else(|| current_model.as_ref().map(extract_history_row));
    let termination_code = stage_reports.last().map_or_else(
        || String::from("no_stages_executed"),
        |s| s.termination_code.clone(),
    );
    let abnormal_termination = matches!(
        termination_code.as_str(),
        "min_timestep_limit"
            | "max_number_retries"
            | "redo_limit"
            | "dt_is_zero"
            | "varcontrol_target < min_allowed_varcontrol_target"
            | "invalid dt_next in timestep_controller"
    );
    let reached_terminal_stop = (last_stage_terminated_by_extras
        || !termination_code.is_empty())
        && termination_code != "max_steps_cap"
        && termination_code != "no_stages_executed"
        && !abnormal_termination;
    let passed = assertions_passed
        && reached_terminal_stop
        && all_required_terminations_ok
        && (terminated_in_init || !verify_restart || photo_restart_verified);

    Ok(CaseRunReport {
        case_name,
        tier: 0,
        passed,
        termination_code,
        stages: stage_reports,
        total_steps,
        total_newton_iterations: total_newton,
        total_retries,
        total_runtime_sec: runtime_sec,
        wall_time_ms: runtime_sec * 1000.0,
        final_teff: final_hr.map_or(0.0, |h| flexmesa_math::exp10(h.log_teff)),
        final_log_l: final_hr.map_or(0.0, |h| h.log_l),
        final_log_r: final_hr.map_or(0.0, |h| h.log_r),
        final_log_tc: final_hr.map_or(0.0, |h| h.log_center_t),
        final_log_rhoc: final_hr.map_or(0.0, |h| h.log_center_rho),
        final_center_h1: final_hr.map_or(0.0, |h| h.center_h1),
        final_center_he4: final_hr.map_or(0.0, |h| h.center_he4),
        photo_restart_verified,
        assertions_passed,
    })
}
