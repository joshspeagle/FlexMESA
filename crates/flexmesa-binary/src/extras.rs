// SPDX-License-Identifier: LGPL-3.0-or-later
//! User-extensible hooks for coupled [`crate::system::BinarySystem`] evolution
//! (`run_binary_extras.f90` and coupled `other_wind` callbacks).

use flexmesa_star::{StarError, StarModel};

use crate::system::BinarySystem;

/// Post-step control flow action returned by [`BinaryExtras::check_model`] and
/// [`BinaryExtras::finish_step`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinaryExtrasAction {
    /// Continue coupled binary evolution (`keep_going`).
    KeepGoing,
    /// Terminate coupled binary evolution with the supplied reason code.
    Terminate(String),
    /// Abort coupled binary evolution with a fatal error message.
    Error(String),
}

/// Trait equivalent of upstream MESA's `run_binary_extras.f90` step hooks and
/// binary-coupled `other_wind` prescription (`star/private/winds.f90`).
pub trait BinaryExtras {
    /// Evaluates a custom intrinsic wind mass-loss rate $\dot{M}_{\mathrm{wind}} \le 0$
    /// in $\mathrm{g\,s^{-1}}$ for component `star_idx` (`1` or `2`) when
    /// `star.controls.use_other_wind` is `true`.
    ///
    /// Returning `Ok(Some(wind_cgs))` supplies the raw signed wind rate before
    /// `max_wind`, rotational $\Omega/\Omega_{\mathrm{crit}}$ enhancement, and
    /// Tout & Eggleton (1988) tidal wind enhancement are applied. Returning
    /// `Ok(None)` when `use_other_wind = true` causes wind evaluation to fail
    /// loudly with [`StarError::InvalidInput`].
    ///
    /// # Errors
    /// Returns [`StarError`] if custom wind evaluation fails.
    fn other_wind_cgs(
        &mut self,
        _star: &StarModel,
        _star_idx: usize,
        _ignore_rlof_flag: bool,
    ) -> Result<Option<f64>, StarError> {
        Ok(None)
    }

    /// Invoked after an accepted coupled binary step (`extras_binary_check_model`).
    fn check_model(&mut self, _bin: &mut BinarySystem) -> BinaryExtrasAction {
        BinaryExtrasAction::KeepGoing
    }

    /// Invoked at the end of an accepted coupled binary step (`extras_binary_finish_step`).
    fn finish_step(&mut self, _bin: &mut BinarySystem) -> BinaryExtrasAction {
        BinaryExtrasAction::KeepGoing
    }

    /// Additional `(name, value)` columns appended to `binary_history.data`
    /// (`how_many_extra_binary_history_columns` / `data_for_extra_binary_history_columns`).
    fn extra_binary_history_columns(
        &self,
        _bin: &BinarySystem,
    ) -> Vec<(String, f64)> {
        Vec::new()
    }

    /// Additional `(name, value)` columns appended to component star `LOGS1` /
    /// `LOGS2` `history.data` (`star_idx` in `1..=2`).
    fn extra_star_history_columns(
        &self,
        _bin: &BinarySystem,
        _star_idx: usize,
    ) -> Vec<(String, f64)> {
        Vec::new()
    }

    /// Receives the active case directory path before binary initialization.
    fn set_staged_case_dir(&mut self, _case_dir: &std::path::Path) {}
}

/// Default no-op [`BinaryExtras`] implementation (`standard_run_binary_extras.inc`).
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultBinaryExtras;

impl BinaryExtras for DefaultBinaryExtras {}
