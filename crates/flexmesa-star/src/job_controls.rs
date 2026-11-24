// SPDX-License-Identifier: LGPL-3.0-or-later
//! Job-level initialization, relaxation, and output controls
//! (`StarJobControls`, corresponding to `star_data/private/star_job_controls.inc`),
//! atmosphere/energy enums, `VarLayout`, and `StarError`.

use std::fmt;

use flexmesa_atm::{
    ATM_T_TAU_EDDINGTON, ATM_T_TAU_KRISHNA_SWAMY, ATM_T_TAU_SOLAR_HOPF,
    ATM_T_TAU_TRAMPEDACH_SOLAR,
};

/// Atmosphere boundary condition option (`atm_option` in `star_controls.inc`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub enum AtmOption {
    /// Grey $T(\tau)$ relation (`'T_tau'`).
    TTau,
    /// Tabulated atmosphere (`'table'`).
    Table,
    /// Irradiated grey atmosphere (`'irradiated_grey'`, Guillot & Havel 2011;
    /// `star/defaults/controls.defaults:4292-4297`, `star/private/atm_support.f90:93-100`).
    IrradiatedGrey,
    /// Simple photosphere $P_0 = \tau g / \kappa$, $T = T_{\mathrm{eff}}$.
    SimplePhotosphere,
}

/// Grey $T(\tau)$ relation identifier (`atm_T_tau_relation` in `star_controls.inc`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub enum AtmTTauRelation {
    /// Eddington grey $T^4 = \tfrac{3}{4} T_{\mathrm{eff}}^4 (\tau + 2/3)$ (`'Eddington'`).
    Eddington,
    /// Solar Hopf relation (`'solar_Hopf'`).
    SolarHopf,
    /// Krishna Swamy (1966) empirical relation (`'Krishna_Swamy'`).
    KrishnaSwamy,
    /// Trampedach et al. (2014) 3D radiation-hydrodynamics relation (`'Trampedach_solar'`).
    TrampedachSolar,
}

impl AtmTTauRelation {
    /// Returns the `flexmesa-atm` integer relation ID (`ATM_T_TAU_*`).
    #[must_use]
    pub const fn id(self) -> i32 {
        match self {
            Self::Eddington => ATM_T_TAU_EDDINGTON,
            Self::SolarHopf => ATM_T_TAU_SOLAR_HOPF,
            Self::KrishnaSwamy => ATM_T_TAU_KRISHNA_SWAMY,
            Self::TrampedachSolar => ATM_T_TAU_TRAMPEDACH_SOLAR,
        }
    }
}

/// Grey $T(\tau)$ atmosphere opacity prescription (`atm_T_tau_opacity` in
/// `star/defaults/controls.defaults:4440` and `star/private/atm_support.f90:278-341`).
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum AtmTTauOpacity {
    /// Constant opacity equal to outer-cell `opacity[0]` (`'fixed'`).
    #[default]
    Fixed,
    /// Uniform atmosphere opacity iterated to consistency with EOS and kap (`'iterated'`).
    Iterated,
    /// Depth-varying opacity integrated inward via `dopri5` (`'varying'`).
    Varying,
}

/// Thermal opacity treatment for `atm_option = 'irradiated_grey'`
/// (`atm_irradiated_opacity` in `star/defaults/controls.defaults:4524-4540`
/// and `star/private/atm_support.f90:790-826`).
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum AtmIrradiatedOpacity {
    /// Uniform opacity fixed to `opacity_start[0]` / `opacity[0]` (`'fixed'`).
    #[default]
    Fixed,
    /// Uniform thermal opacity iterated to consistency with EOS and kap (`'iterated'`).
    Iterated,
}

/// Structured diagnostic payload returned inside [`StarError::StepFailure`].
#[derive(Debug, Clone, PartialEq)]
pub struct StepFailureInfo {
    /// Stellar `model_number` being attempted or evaluated.
    pub model_number: usize,
    /// Zero-based zone index holding the maximum residual or failing condition.
    pub zone: Option<usize>,
    /// Name of the equation holding the maximum residual.
    pub equation: Option<String>,
    /// Name of the paired solver variable at `zone`.
    pub variable: Option<String>,
    /// Signed offending residual or diagnostic value.
    pub value: Option<f64>,
    /// L2 residual norm across all solver equations.
    pub residual_norm: f64,
    /// L-infinity maximum absolute residual across solver equations.
    pub max_residual: f64,
    /// L2 scaled correction norm across solver variables.
    pub correction_norm: f64,
    /// L-infinity maximum scaled correction.
    pub max_correction: f64,
    /// Attempted timestep $\Delta t$ in seconds.
    pub dt: f64,
    /// Number of Newton iterations completed before failure.
    pub iterations: usize,
    /// Retry attempt count recorded when step retries are exhausted.
    pub retries: usize,
    /// Reason recorded for the retry or step rejection.
    pub retry_reason: String,
    /// Concise summary of the failure site and cause.
    pub detail: String,
}

impl fmt::Display for StepFailureInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "model={} dt={:.3e} s iters={} retries={}",
            self.model_number, self.dt, self.iterations, self.retries
        )?;
        if let Some(zone) = self.zone {
            write!(f, " zone={zone}")?;
        }
        if let Some(ref eq) = self.equation {
            write!(f, " eq={eq}")?;
        }
        if let Some(ref var) = self.variable {
            write!(f, " var={var}")?;
        }
        if let Some(val) = self.value {
            write!(f, " val={val:.3e}")?;
        }
        write!(
            f,
            " rnorm={:.3e} rmax={:.3e} cnorm={:.3e} cmax={:.3e}",
            self.residual_norm,
            self.max_residual,
            self.correction_norm,
            self.max_correction
        )?;
        if !self.retry_reason.is_empty() {
            write!(f, " retry_reason={}", self.retry_reason)?;
        }
        if !self.detail.is_empty() {
            write!(f, ": {}", self.detail)?;
        }
        Ok(())
    }
}

/// Error conditions returned by `flexmesa-star` routines.
#[derive(Debug, Clone, PartialEq)]
pub enum StarError {
    /// Invalid configuration or input dimension.
    InvalidInput(String),
    /// Failure during ZAMS or `.mod`/`.photo` file parsing or serialization.
    IoError(String),
    /// Microphysics (EOS, opacity, reaction network, or atmosphere) failure.
    MicrophysicsError(String),
    /// Non-convergence in root-finding or Henyey Newton-Raphson solver.
    ConvergenceFailure(String),
    /// Structured Newton-step or retry-cascade failure with diagnostics.
    StepFailure(Box<StepFailureInfo>),
    /// Controlled runtime termination (e.g. timestep controller limit).
    Terminated {
        /// Active stellar model number when termination triggered.
        model_number: usize,
        /// Termination code or reason string.
        reason: String,
    },
}

impl StarError {
    /// Returns `true` for errors that may recover under a smaller timestep or
    /// outer retry, and `false` for non-retriable input, I/O, or termination
    /// errors (`DECISIONS #30`).
    #[must_use]
    pub const fn is_retriable(&self) -> bool {
        !matches!(
            self,
            Self::InvalidInput(_) | Self::IoError(_) | Self::Terminated { .. }
        )
    }
}

impl fmt::Display for StarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid star input: {msg}"),
            Self::IoError(msg) => write!(f, "star model I/O error: {msg}"),
            Self::MicrophysicsError(msg) => {
                write!(f, "star microphysics error: {msg}")
            }
            Self::ConvergenceFailure(msg) => {
                write!(f, "star solver convergence failure: {msg}")
            }
            Self::StepFailure(info) => write!(f, "star step failed: {info}"),
            Self::Terminated {
                model_number,
                reason,
            } => write!(f, "star terminated at model {model_number}: {reason}"),
        }
    }
}

impl std::error::Error for StarError {}

/// Energy conservation equation formulation (`energy_eqn_option` in MESA).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub enum EnergyEqnOption {
    /// Total energy conservation $de/dt + P\,d(1/\rho)/dt$ (`'dedt'`).
    Dedt,
    /// Gravitational energy form $\epsilon_{\mathrm{grav}} = -T\,ds/dt$ (`'eps_grav'`).
    EpsGrav,
}

/// Dynamic hydrodynamic variable and equation slot layout (`set_var_info` in
/// `star/private/alloc.f90:2002-2110`).
///
/// All indices are 0-based into per-zone slices of length `nvar_hydro` or
/// `nvar_total = nvar_hydro + species`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct VarLayout {
    /// Cell-centered $\ln\rho$ variable slot (`0`, Fortran `1`).
    pub i_lnd: usize,
    /// Cell-centered $\ln T$ variable slot (`1`, Fortran `2`).
    pub i_lnt: usize,
    /// Face-centered $\ln R$ variable slot (`2`, Fortran `3`).
    pub i_lnr: usize,
    /// Face-centered luminosity $L$ variable slot (`3`, Fortran `4`).
    pub i_lum: usize,
    /// Optional face-centered velocity $v$ variable slot (`v_flag`).
    pub i_v: Option<usize>,
    /// Optional cell-centered Riemann velocity $u$ variable slot (`u_flag`).
    pub i_u: Option<usize>,
    /// Optional cell-centered turbulent velocity $w = \sqrt{e_t}$ variable slot (`RSP2_flag`).
    pub i_w: Option<usize>,
    /// Optional face-centered pressure scale height $H_P$ variable slot (`RSP2_flag`).
    pub i_hp: Option<usize>,
    /// Optional cell-centered Rayleigh-Taylor instability amplitude $\alpha_{\mathrm{RTI}}$ slot (`RTI_flag`).
    pub i_alpha_rti: Option<usize>,
    /// Number of hydrodynamic variables per zone (`4` in hydrostatic mode, `5`
    /// with velocity, `6` with velocity + `RTI_flag`, `7` with velocity + `RSP2_flag`).
    pub nvar_hydro: usize,
    /// Number of nuclear network species (`species`).
    pub species: usize,
    /// Total coupled solver variables per zone (`nvar_hydro + species`).
    pub nvar_total: usize,
    /// Temperature gradient / surface $T$ equation slot (`i_equL`).
    pub i_equ_l: usize,
    /// Momentum / hydrostatic / surface $P$ equation slot (`i_dv_dt`).
    pub i_dv_dt: usize,
    /// Continuity / volume-radius equation slot (`i_dlnd_dt`).
    pub i_dlnd_dt: usize,
    /// Energy conservation equation slot (`i_dlnE_dt`).
    pub i_dlne_dt: usize,
    /// Optional radius-velocity kinematic equation slot (`i_dlnR_dt`).
    pub i_dlnr_dt: Option<usize>,
    /// Optional `RSP2` turbulent energy equation slot (`i_turbulent_energy`).
    pub i_turbulent_energy: Option<usize>,
    /// Optional `RSP2` pressure scale height algebraic equation slot (`i_equ_Hp`).
    pub i_equ_hp: Option<usize>,
    /// Optional Rayleigh-Taylor amplitude transport equation slot (`i_dalpha_RTI_dt`).
    pub i_dalpha_rti_dt: Option<usize>,
}

impl VarLayout {
    /// Constructs the variable and equation layout for `(species, v_flag, u_flag)`
    /// matching `set_var_info` in `star/private/alloc.f90:2301-2437`.
    #[must_use]
    pub const fn new(species: usize, v_flag: bool, u_flag: bool) -> Self {
        Self::with_flags(species, v_flag, u_flag, false, false)
    }

    /// Constructs the variable and equation layout including optional `RSP2`
    /// coupled `(w, Hp)` variables (`star/private/alloc.f90:2347-2405`).
    #[must_use]
    pub const fn with_rsp2(
        species: usize,
        v_flag: bool,
        u_flag: bool,
        rsp2_flag: bool,
    ) -> Self {
        Self::with_flags(species, v_flag, u_flag, rsp2_flag, false)
    }

    /// Constructs the full variable and equation layout including optional
    /// `RSP2` `(w, Hp)` and `RTI` `alpha_RTI` variables
    /// (`star/private/alloc.f90:2342-2426`).
    #[must_use]
    pub const fn with_flags(
        species: usize,
        v_flag: bool,
        u_flag: bool,
        rsp2_flag: bool,
        rti_flag: bool,
    ) -> Self {
        let i_lnd = 0;
        let i_lnt = 1;
        let i_lnr = 2;
        let i_lum = 3;
        let mut nvar_hydro = 4;
        let i_v = if v_flag {
            let idx = nvar_hydro;
            nvar_hydro += 1;
            Some(idx)
        } else {
            None
        };
        let i_u = if !v_flag && u_flag {
            let idx = nvar_hydro;
            nvar_hydro += 1;
            Some(idx)
        } else {
            None
        };
        let (i_w, i_hp) = if rsp2_flag {
            let iw = nvar_hydro;
            let ihp = nvar_hydro + 1;
            nvar_hydro += 2;
            (Some(iw), Some(ihp))
        } else {
            (None, None)
        };
        let i_alpha_rti = if rti_flag {
            let ia = nvar_hydro;
            nvar_hydro += 1;
            Some(ia)
        } else {
            None
        };
        let has_vel = v_flag || u_flag;
        let (i_equ_l, i_dv_dt, i_dlnd_dt, i_dlne_dt, i_dlnr_dt) = if has_vel {
            (i_lum, 4, i_lnd, i_lnt, Some(i_lnr))
        } else {
            (i_lnd, i_lnt, i_lnr, i_lum, None)
        };
        Self {
            i_lnd,
            i_lnt,
            i_lnr,
            i_lum,
            i_v,
            i_u,
            i_w,
            i_hp,
            i_alpha_rti,
            nvar_hydro,
            species,
            nvar_total: nvar_hydro + species,
            i_equ_l,
            i_dv_dt,
            i_dlnd_dt,
            i_dlne_dt,
            i_dlnr_dt,
            i_turbulent_energy: i_w,
            i_equ_hp: i_hp,
            i_dalpha_rti_dt: i_alpha_rti,
        }
    }

    /// Returns the solver variable index (`0..nvar_total`) for species `j`
    /// (`0..species`).
    #[must_use]
    #[inline]
    pub const fn chem_var_index(&self, j: usize) -> usize {
        self.nvar_hydro + j
    }

    /// Returns the number of coupled variables per zone (`nvar_total` when
    /// `include_chem` is true, or `nvar_hydro` otherwise).
    #[must_use]
    #[inline]
    pub const fn nvar(&self, include_chem: bool) -> usize {
        if include_chem {
            self.nvar_total
        } else {
            self.nvar_hydro
        }
    }

    /// Returns the MESA equation slot name for 0-based equation `equ_idx`.
    #[must_use]
    pub fn equation_name(
        &self,
        equ_idx: usize,
        iso_names: &[String],
    ) -> String {
        if equ_idx == self.i_equ_l {
            return String::from("equ_L");
        }
        if equ_idx == self.i_dv_dt {
            return String::from("dv_dt");
        }
        if equ_idx == self.i_dlnd_dt {
            return String::from("dlnd_dt");
        }
        if equ_idx == self.i_dlne_dt {
            return String::from("dlnE_dt");
        }
        if self.i_dlnr_dt == Some(equ_idx) {
            return String::from("dlnR_dt");
        }
        if self.i_turbulent_energy == Some(equ_idx) {
            return String::from("d_etrb_dt");
        }
        if self.i_equ_hp == Some(equ_idx) {
            return String::from("equ_Hp");
        }
        if self.i_dalpha_rti_dt == Some(equ_idx) {
            return String::from("dalpha_RTI_dt");
        }
        if equ_idx >= self.nvar_hydro {
            let j = equ_idx - self.nvar_hydro;
            return iso_names.get(j).map_or_else(
                || format!("equ_chem_{j}"),
                |name| format!("d_dt_{name}"),
            );
        }
        format!("equ_{equ_idx}")
    }

    /// Returns the MESA solver variable name for 0-based slot `var_idx`.
    #[must_use]
    pub fn variable_name(
        &self,
        var_idx: usize,
        iso_names: &[String],
    ) -> String {
        if var_idx == self.i_lnd {
            return String::from("lnd");
        }
        if var_idx == self.i_lnt {
            return String::from("lnT");
        }
        if var_idx == self.i_lnr {
            return String::from("lnR");
        }
        if var_idx == self.i_lum {
            return String::from("L");
        }
        if self.i_v == Some(var_idx) {
            return String::from("v");
        }
        if self.i_u == Some(var_idx) {
            return String::from("u");
        }
        if self.i_w == Some(var_idx) {
            return String::from("w");
        }
        if self.i_hp == Some(var_idx) {
            return String::from("Hp");
        }
        if self.i_alpha_rti == Some(var_idx) {
            return String::from("alpha_RTI");
        }
        if var_idx >= self.nvar_hydro {
            let j = var_idx - self.nvar_hydro;
            return iso_names
                .get(j)
                .cloned()
                .unwrap_or_else(|| format!("chem_{j}"));
        }
        format!("var_{var_idx}")
    }

    /// Returns the physical solver variable name paired with 0-based equation
    /// slot `equ_idx` (accounting for hydrostatic bandwidth reordering in
    /// `alloc.f90:2372-2386`).
    #[must_use]
    pub fn paired_variable_name(
        &self,
        equ_idx: usize,
        iso_names: &[String],
    ) -> String {
        let has_vel = self.i_v.is_some() || self.i_u.is_some();
        if has_vel {
            if equ_idx == self.i_dlnd_dt {
                return String::from("lnd");
            }
            if equ_idx == self.i_dlne_dt {
                return String::from("lnT");
            }
            if self.i_dlnr_dt == Some(equ_idx) {
                return String::from("lnR");
            }
            if equ_idx == self.i_equ_l {
                return String::from("L");
            }
            if equ_idx == self.i_dv_dt {
                let vel = if self.i_v.is_some() { "v" } else { "u" };
                return String::from(vel);
            }
        } else {
            if equ_idx == self.i_dv_dt {
                return String::from("lnd");
            }
            if equ_idx == self.i_equ_l {
                return String::from("lnT");
            }
            if equ_idx == self.i_dlnd_dt {
                return String::from("lnR");
            }
            if equ_idx == self.i_dlne_dt {
                return String::from("L");
            }
        }
        if self.i_turbulent_energy == Some(equ_idx) {
            return String::from("w");
        }
        if self.i_equ_hp == Some(equ_idx) {
            return String::from("Hp");
        }
        if self.i_dalpha_rti_dt == Some(equ_idx) {
            return String::from("alpha_RTI");
        }
        if equ_idx >= self.nvar_hydro {
            let j = equ_idx - self.nvar_hydro;
            return iso_names
                .get(j)
                .cloned()
                .unwrap_or_else(|| format!("chem_{j}"));
        }
        format!("var_{equ_idx}")
    }
}

/// Job-level initialization and model-building controls (`&star_job` namelist
/// in `star_data/private/star_job_controls.inc`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct StarJobControls {
    /// Whether to construct an initial $n = 1.5$ pre-MS polytrope model.
    pub create_pre_ms_model: bool,
    /// Central temperature $T_c$ ($\mathrm{K}$) for `create_pre_ms_model`.
    pub pre_ms_t_c: f64,
    /// Optional initial central density guess $\rho_c$ ($\mathrm{g\,cm^{-3}}$).
    pub pre_ms_guess_rho_c: f64,
    /// Step size $\Delta \log_{10} P_{\mathrm{gas}}$ for outward pre-MS integration.
    pub pre_ms_d_log10_p: f64,
    /// Surface $\log_{10} P$ cutoff for pre-MS integration (`3.5` default).
    pub pre_ms_log_p_surf_limit: f64,
    /// Surface $\log_{10} T$ cutoff for pre-MS integration (`3.7` default).
    pub pre_ms_log_t_surf_limit: f64,
    /// Logarithmic luminosity boost factor for pre-MS contraction (`0.0` default).
    pub pre_ms_log_lum_factor: f64,
    /// Steps of `do_relax_num_steps` after building a pre-MS model
    /// (`pre_ms_relax_num_steps`, `300` default,
    /// `star/defaults/star_job.defaults:441`; `<= 0` skips the relax).
    pub pre_ms_relax_num_steps: isize,
    /// Whether to relax a new pre-MS model until a radiative core starts
    /// (`pre_ms_relax_to_start_radiative_core`, `.true.` default,
    /// `star/defaults/star_job.defaults:453`).
    pub pre_ms_relax_to_start_radiative_core: bool,
    /// Declared but never read upstream
    /// (`pre_ms_min_steps_before_check_radiative_core`, `50` default,
    /// `star/defaults/star_job.defaults:475`).
    pub pre_ms_min_steps_before_check_radiative_core: isize,
    /// Start testing for the core once the minimum `conv_mx1_bot` falls
    /// below this fractional mass (`pre_ms_check_radiative_core_start`,
    /// `1d-6` default, `star/defaults/star_job.defaults:476`).
    pub pre_ms_check_radiative_core_start: f64,
    /// Stop when `conv_mx1_bot` exceeds this fractional mass
    /// (`pre_ms_check_radiative_core_stop`, `1d-3` default,
    /// `star/defaults/star_job.defaults:477`).
    pub pre_ms_check_radiative_core_stop: f64,
    /// Stop when $L_{\mathrm{nuc}}/L_{\mathrm{phot}}$ exceeds this
    /// (`pre_ms_check_radiative_core_Lnuc_div_L_limit`, `0.1` default,
    /// `star/defaults/star_job.defaults:478`).
    pub pre_ms_check_radiative_core_lnuc_div_l_limit: f64,
    /// Skip the radiative-core relax below this mass in $M_\odot$
    /// (`pre_ms_check_radiative_core_min_mass`, `0.3` default,
    /// `star/defaults/star_job.defaults:479`).
    pub pre_ms_check_radiative_core_min_mass: f64,
    /// Whether to construct an isentropic initial planet/WD model (`create_initial_model`).
    pub create_initial_model: bool,
    /// Radius in cm for `create_initial_model`.
    pub radius_in_cm_for_create_initial_model: f64,
    /// Mass in grams for `create_initial_model`.
    pub mass_in_gm_for_create_initial_model: f64,
    /// First-guess $\log_{10} P_c$ for `create_initial_model`
    /// (`center_logP_1st_try_for_create_initial_model`, `10.9` default).
    pub center_log_p_1st_try_for_create_initial_model: f64,
    /// First-guess entropy in units of $k_B / m_p$ for `create_initial_model`
    /// (`entropy_1st_try_for_create_initial_model`, `11.5` default).
    pub entropy_1st_try_for_create_initial_model: f64,
    /// Maximum Newton tries for `create_initial_model`
    /// (`max_tries_for_create_initial_model`, `100` default).
    pub max_tries_for_create_initial_model: isize,
    /// Tolerance on the relative mass error of `create_initial_model`
    /// (`abs_e01_tolerance_for_create_initial_model`, `1d-4` default).
    pub abs_e01_tolerance_for_create_initial_model: f64,
    /// Tolerance on the relative radius error of `create_initial_model`
    /// (`abs_e02_tolerance_for_create_initial_model`, `1d-4` default).
    pub abs_e02_tolerance_for_create_initial_model: f64,
    /// Relaxation steps after `create_initial_model`
    /// (`initial_model_relax_num_steps`, `10` default).
    pub initial_model_relax_num_steps: isize,
    /// Runge-Kutta step-size accuracy for `create_initial_model`
    /// (`initial_model_eps`, `0.05` default).
    pub initial_model_eps: f64,
    /// Put metals missing from the net into its heaviest isotope when building
    /// a model (`dump_missing_metals_into_heaviest`, `.true.` default).
    pub dump_missing_metals_into_heaviest: bool,
    /// Whether `use_other_build_initial_model` hook is enabled.
    pub use_other_build_initial_model: bool,
    /// Whether to construct a static radial stellar pulsation envelope (`create_RSP_model`).
    pub create_rsp_model: bool,
    /// Whether to load a saved `.mod` file on startup.
    pub load_saved_model: bool,
    /// Path to saved `.mod` file when `load_saved_model` is true.
    pub saved_model_name: String,
    /// Whether to save a `.mod` file when evolution terminates (`save_model_when_terminate`).
    pub save_model_when_terminate: bool,
    /// Whether `astero` runs the star once and leaves every decision to the
    /// user's `extras_check_model`, skipping its own search and chi^2 logic
    /// (`astero_just_call_my_extras_check_model`;
    /// `astero/private/astero_run_support.f90:171-174,239`).
    pub astero_just_call_my_extras_check_model: bool,
    /// Output `.mod` filename on termination (`save_model_filename`).
    pub save_model_filename: String,
    /// Model number at which to save a `.mod` file (`save_model_number`, `< 0` disables).
    pub save_model_number: isize,
    /// Whether to save a `.photo` restart file on normal termination when
    /// `required_termination_code_string` is satisfied
    /// (`save_photo_when_terminate`, `.true.` default,
    /// `star/defaults/star_job.defaults:270`,
    /// `star/job/run_star_support.f90:789-790`).
    pub save_photo_when_terminate: bool,
    /// Optional explicit output `.photo` filename; when empty, writes
    /// `<photo_directory>/<get_name_for_restart_file(model_number, photo_digits)>`
    /// (`save_photo_filename`, `''` default,
    /// `star/defaults/star_job_dev.defaults:118`,
    /// `star/public/star_lib.f90:492-502`).
    pub save_photo_filename: String,
    /// Specific model number at which to save a `.photo` checkpoint
    /// (`save_photo_number`, `-1` default = disabled,
    /// `star/defaults/star_job_dev.defaults:117`,
    /// `star/job/run_star_support.f90:1276-1281`).
    pub save_photo_number: isize,
    /// Whether to export pulsation data on termination (`save_pulse_data_when_terminate`).
    pub save_pulse_data_when_terminate: bool,
    /// Output pulsation filename on termination (`save_pulse_data_filename`).
    pub save_pulse_data_filename: String,
    /// Pulsation export format (`"GYRE"`, `"FGONG"`, `"OSC"`).
    pub pulse_data_format: String,
    /// Whether to export `STELLA` hydro/abundance data on termination (`save_stella_data_when_terminate`).
    pub save_stella_data_when_terminate: bool,
    /// Output `STELLA` filename prefix (`save_stella_data_filename`).
    pub save_stella_data_filename: String,
    /// Nuclear reaction network name (`"basic.net"` default).
    pub net_name: String,
    /// Whether to switch nuclear reaction networks (`change_net` or `change_initial_net`).
    pub change_net: bool,
    /// Target nuclear reaction network name (`new_net_name`).
    pub new_net_name: String,
    /// Whether to automatically extend the reaction network with evolutionary stage (`auto_extend_net`).
    pub auto_extend_net: bool,
    /// H/He-burning network for `auto_extend_net` (`"basic.net"` default).
    pub h_he_net: String,
    /// C/O-burning network for `auto_extend_net` (`"co_burn.net"` default).
    pub co_net: String,
    /// Advanced burning network for `auto_extend_net` (`"approx21.net"` default).
    pub adv_net: String,
    /// Number of per-reaction special rate multipliers (`num_special_rate_factors`, `0` default).
    #[serde(default)]
    pub num_special_rate_factors: usize,
    /// Reaction names for special rate factors (`reaction_for_special_factor(1..)`).
    #[serde(default)]
    pub reaction_for_special_factor: Vec<String>,
    /// Multipliers for special rate factors (`special_rate_factor(1..)`, `1.0` default).
    #[serde(default)]
    pub special_rate_factor: Vec<f64>,
    /// Optional 1D rate-table filenames overriding reactions named in
    /// `reaction_for_special_factor(1..)` (`filename_of_special_rate(1..)`,
    /// `star/defaults/star_job.defaults:2465`, `star/private/net.f90:735`).
    #[serde(default)]
    pub filename_of_special_rate: Vec<String>,
    /// Use Fushiki & Lamb (1987) pycnonuclear triple-alpha rate (`use_3a_fl87`, `false` default).
    #[serde(default)]
    pub use_3a_fl87: bool,
    /// Use Suzuki et al. (2016) $A=17\text{--}28$ weak rates (`use_suzuki_weak_rates`, `false` default).
    #[serde(default)]
    pub use_suzuki_weak_rates: bool,
    /// Whether to change `v_flag` on startup (`change_v_flag` / `change_initial_v_flag`).
    pub change_v_flag: bool,
    /// Target `v_flag` when `change_v_flag` is true (`new_v_flag`).
    pub new_v_flag: bool,
    /// Whether to change `u_flag` on startup (`change_u_flag` / `change_initial_u_flag`).
    pub change_u_flag: bool,
    /// Target `u_flag` when `change_u_flag` is true (`new_u_flag`).
    pub new_u_flag: bool,
    /// Whether to change `rotation_flag` on startup (`change_rotation_flag`).
    pub change_rotation_flag: bool,
    /// Target `rotation_flag` when `change_rotation_flag` is true (`new_rotation_flag`).
    pub new_rotation_flag: bool,
    /// Apply `new_omega` at startup (`set_omega`, `false` default).
    pub set_omega: bool,
    /// Apply `new_omega` only on initial non-restart run (`set_initial_omega`, `false` default).
    pub set_initial_omega: bool,
    /// Uniform angular velocity $\Omega$ ($\mathrm{rad\,s^{-1}}$) on startup (`new_omega`, `0.0` default).
    pub new_omega: f64,
    /// Apply `new_omega_div_omega_crit` at startup (`set_omega_div_omega_crit`, `false` default).
    pub set_omega_div_omega_crit: bool,
    /// Apply `new_omega_div_omega_crit` only on initial non-restart run (`set_initial_omega_div_omega_crit`, `false` default).
    pub set_initial_omega_div_omega_crit: bool,
    /// Uniform $\Omega / \Omega_{\mathrm{crit}}$ on startup (`new_omega_div_omega_crit`, `0.0` default).
    pub new_omega_div_omega_crit: f64,
    /// Apply `new_surface_rotation_v` at startup (`set_surface_rotation_v`, `false` default).
    pub set_surface_rotation_v: bool,
    /// Apply `new_surface_rotation_v` only on initial non-restart run (`set_initial_surface_rotation_v`, `false` default).
    pub set_initial_surface_rotation_v: bool,
    /// Surface equatorial velocity ($\mathrm{km\,s^{-1}}$) on startup (`new_surface_rotation_v`, `0.0` default).
    pub new_surface_rotation_v: f64,
    /// Relax to a uniform `new_omega` (`relax_omega`, `false` default).
    pub relax_omega: bool,
    /// Relax to a uniform `new_omega` on a non-restart start only
    /// (`relax_initial_omega`, `false` default).
    pub relax_initial_omega: bool,
    /// Relax to a uniform $\Omega = $ `new_omega_div_omega_crit`
    /// $\times\,\Omega_{\mathrm{crit}}$ (`relax_omega_div_omega_crit`, `false` default).
    pub relax_omega_div_omega_crit: bool,
    /// As `relax_omega_div_omega_crit`, on a non-restart start only
    /// (`relax_initial_omega_div_omega_crit`, `false` default).
    pub relax_initial_omega_div_omega_crit: bool,
    /// Relax to a uniform $\Omega$ with surface velocity `new_surface_rotation_v`
    /// (`relax_surface_rotation_v`, `false` default).
    pub relax_surface_rotation_v: bool,
    /// As `relax_surface_rotation_v`, on a non-restart start only
    /// (`relax_initial_surface_rotation_v`, `false` default).
    pub relax_initial_surface_rotation_v: bool,
    /// Steps over which the rotation relaxes (`num_steps_to_relax_rotation`,
    /// `100` default).
    pub num_steps_to_relax_rotation: isize,
    /// Maximum timestep while the rotation relaxes, in years
    /// (`relax_omega_max_yrs_dt`, `1d9` default).
    pub relax_omega_max_yrs_dt: f64,
    /// Rotation is switched on near the ZAMS and set from `new_omega` for this
    /// many steps (`set_near_zams_omega_steps`, `-1` default).
    pub set_near_zams_omega_steps: isize,
    /// As `set_near_zams_omega_steps`, from `new_omega_div_omega_crit`
    /// (`set_near_zams_omega_div_omega_crit_steps`, `-1` default).
    pub set_near_zams_omega_div_omega_crit_steps: isize,
    /// As `set_near_zams_omega_steps`, from `new_surface_rotation_v`
    /// (`set_near_zams_surface_rotation_v_steps`, `-1` default).
    pub set_near_zams_surface_rotation_v_steps: isize,
    /// Last model number at which `do_rotation` sets a uniform `new_omega`
    /// (`set_omega_step_limit`, `-1` default).
    pub set_omega_step_limit: isize,
    /// As `set_omega_step_limit`, for `new_omega_div_omega_crit`
    /// (`set_omega_div_omega_crit_step_limit`, `-1` default).
    pub set_omega_div_omega_crit_step_limit: isize,
    /// As `set_omega_step_limit`, for `new_surface_rotation_v`
    /// (`set_surf_rotation_v_step_limit`, `-1` default).
    pub set_surf_rotation_v_step_limit: isize,
    /// Switch rotation on near the ZAMS and relax to `new_omega`
    /// (`near_zams_relax_omega`, `false` default).
    pub near_zams_relax_omega: bool,
    /// Switch rotation on near the ZAMS and relax to
    /// `new_omega_div_omega_crit` (`near_zams_relax_omega_div_omega_crit`,
    /// `false` default).
    pub near_zams_relax_omega_div_omega_crit: bool,
    /// Switch rotation on near the ZAMS and relax to `new_surface_rotation_v`
    /// (`near_zams_relax_initial_surface_rotation_v`, `false` default).
    pub near_zams_relax_initial_surface_rotation_v: bool,
    /// Set `tau_factor = rho(1)/base_for_adjust_tau_factor_to_surf_density`
    /// after every step (`adjust_tau_factor_to_surf_density`, `false` default).
    pub adjust_tau_factor_to_surf_density: bool,
    /// Density for `adjust_tau_factor_to_surf_density`, in g cm^-3
    /// (`base_for_adjust_tau_factor_to_surf_density`, `0` default).
    pub base_for_adjust_tau_factor_to_surf_density: f64,
    /// Central He4 below which `tau_factor` is set to `set_to_this_tau_factor`
    /// (`set_tau_factor_after_core_He_burn`, `-1` default).
    pub set_tau_factor_after_core_he_burn: f64,
    /// Central C12 below which `tau_factor` is set to `set_to_this_tau_factor`
    /// (`set_tau_factor_after_core_C_burn`, `-1` default).
    pub set_tau_factor_after_core_c_burn: f64,
    /// Central He4 below which `tau_factor` steps toward
    /// `relax_to_this_tau_factor` (`relax_tau_factor_after_core_He_burn`, `-1` default).
    pub relax_tau_factor_after_core_he_burn: f64,
    /// Central C12 below which `tau_factor` steps toward
    /// `relax_to_this_tau_factor` (`relax_tau_factor_after_core_C_burn`, `-1` default).
    pub relax_tau_factor_after_core_c_burn: f64,
    /// Whether to set initial stellar age (`set_initial_age`).
    pub set_initial_age: bool,
    /// Initial stellar age in years (`initial_age`).
    pub initial_age: f64,
    /// Whether to set initial model number (`set_initial_model_number`).
    pub set_initial_model_number: bool,
    /// Initial model number (`initial_model_number`).
    pub initial_model_number: usize,
    /// Whether to reset cumulative retries on initial model load
    /// (`set_initial_number_retries`, `true` default in `star/defaults/star_job.defaults:633`).
    #[serde(default = "default_true")]
    pub set_initial_number_retries: bool,
    /// Initial cumulative retry count when `set_initial_number_retries = true`
    /// (`initial_number_retries`, `0` default in `star/defaults/star_job.defaults:634`).
    #[serde(default)]
    pub initial_number_retries: usize,
    /// Whether to set initial timestep (`set_initial_dt`).
    pub set_initial_dt: bool,
    /// Initial timestep in years (`years_for_initial_dt`).
    pub years_for_initial_dt: f64,
    /// Initial timestep in seconds (`seconds_for_initial_dt`).
    pub seconds_for_initial_dt: f64,
    /// Whether to reset cumulative energy error on startup (`set_initial_cumulative_energy_error`).
    pub set_initial_cumulative_energy_error: bool,
    /// Target cumulative energy error in erg (`new_cumulative_energy_error`).
    pub new_cumulative_energy_error: f64,
    /// Set `cumulative_energy_error` to `new_cumulative_energy_error` at the
    /// end of every relax (`set_cumulative_energy_error_each_relax`, `.true.`
    /// default, `star/defaults/star_job.defaults:2374`;
    /// `star/private/relax.f90:4153-4154`).
    pub set_cumulative_energy_error_each_relax: bool,
    /// Whether to impose uniform initial composition (`set_uniform_initial_composition`).
    pub set_uniform_initial_composition: bool,
    /// Uniform initial $^1\mathrm{H}$ mass fraction (`initial_h1`).
    pub initial_h1: f64,
    /// Uniform initial $^2\mathrm{H}$ mass fraction (`initial_h2`).
    pub initial_h2: f64,
    /// Uniform initial $^3\mathrm{He}$ mass fraction (`initial_he3`).
    pub initial_he3: f64,
    /// Uniform initial $^4\mathrm{He}$ mass fraction (`initial_he4`).
    pub initial_he4: f64,
    /// Solar heavy-element mixture selector (`initial_zfracs`, `3` = GS98 default).
    pub initial_zfracs: i32,
    /// `relax_mass`: relax the total mass to `new_mass` with
    /// `star_relax_mass` (`star/job/run_star_support.f90:2422-2426`).
    pub relax_mass: bool,
    /// `relax_initial_mass`: as `relax_mass`, later in the sequence and only
    /// for a new run (`star/job/run_star_support.f90:2467-2471`).
    pub relax_initial_mass: bool,
    /// `relax_mass_scale`: relax the mass by rescaling `dm`
    /// (`star_relax_mass_scale`, `star/job/run_star_support.f90:2480-2487`).
    pub relax_mass_scale: bool,
    /// `relax_initial_mass_scale`: as `relax_mass_scale`, for a new run.
    pub relax_initial_mass_scale: bool,
    /// Target stellar mass in $M_\odot$ (`new_mass`).
    pub new_mass: f64,
    /// $\log_{10}$ of the largest $|\dot M|$ in $M_\odot\,\mathrm{yr}^{-1}$
    /// for `relax_mass`; `<= -100` picks a default by mass
    /// (`lg_max_abs_mdot`, default `-100`).
    pub lg_max_abs_mdot: f64,
    /// $\Delta\log_{10} M$ per step for `relax_mass_scale`
    /// (`dlgm_per_step`, default `1d-3`).
    pub dlgm_per_step: f64,
    /// Timestep in years for `relax_mass_scale`
    /// (`change_mass_years_for_dt`, default `1`).
    pub change_mass_years_for_dt: f64,
    /// `relax_Z`: relax $Z$ to `new_Z` over several steps
    /// (`star_relax_Z`, `star/job/run_star_support.f90:2395-2402`).
    pub relax_z: bool,
    /// `relax_initial_Z`: as `relax_Z`, for a new run.
    pub relax_initial_z: bool,
    /// `change_Z`: set $Z$ to `new_Z` at once (`star_set_z`,
    /// `star/job/run_star_support.f90:2289-2298`).
    pub change_z: bool,
    /// `change_initial_Z`: as `change_Z`, for a new run.
    pub change_initial_z: bool,
    /// Lowest $q$ changed by `relax_Z` (`relax_Z_minq`, default `0`).
    pub relax_z_minq: f64,
    /// Highest $q$ changed by `relax_Z` (`relax_Z_maxq`, default `1`).
    pub relax_z_maxq: f64,
    /// Target metal mass fraction $Z$ (`new_Z`).
    pub new_z: f64,
    /// `relax_Y`: relax $Y$ to `new_Y` over several steps
    /// (`star_relax_Y`, `star/job/run_star_support.f90:2413-2420`).
    pub relax_y: bool,
    /// `relax_initial_Y`: as `relax_Y`, for a new run.
    pub relax_initial_y: bool,
    /// `change_Y`: set $Y$ to `new_Y` at once (`star_set_y`,
    /// `star/job/run_star_support.f90:2300-2308`).
    pub change_y: bool,
    /// `change_initial_Y`: as `change_Y`, for a new run.
    pub change_initial_y: bool,
    /// Lowest $q$ changed by `relax_Y` (`relax_Y_minq`, default `0`).
    pub relax_y_minq: f64,
    /// Highest $q$ changed by `relax_Y` (`relax_Y_maxq`, default `1`).
    pub relax_y_maxq: f64,
    /// Target helium mass fraction $Y$ (`new_Y`).
    pub new_y: f64,
    /// `relax_initial_composition`: relax to the composition profile in
    /// `relax_composition_filename` (`star/job/run_star_support.f90:2236-2239,
    /// 2793-2845`).
    pub relax_initial_composition: bool,
    /// Steps for `relax_initial_composition`
    /// (`num_steps_to_relax_composition`, default `100`).
    pub num_steps_to_relax_composition: usize,
    /// Composition profile file (`relax_composition_filename`).
    pub relax_composition_filename: String,
    /// Timescale in seconds for `relax_initial_composition`; `< 0` uses
    /// `num_steps_to_relax_composition` (`timescale_for_relax_composition`,
    /// default `-1`).
    pub timescale_for_relax_composition: f64,
    /// `set_tau_factor`: set `tau_factor` to `set_to_this_tau_factor`
    /// (`star/job/run_star_support.f90:1949-1953`).
    pub set_tau_factor: bool,
    /// `set_initial_tau_factor`: as `set_tau_factor`, for a new run.
    pub set_initial_tau_factor: bool,
    /// `set_to_this_tau_factor` (default `-1`).
    pub set_to_this_tau_factor: f64,
    /// `relax_tau_factor`: relax `tau_factor` to `relax_to_this_tau_factor`
    /// (`star_relax_tau_factor`, `star/job/run_star_support.f90:2542-2548`).
    pub relax_tau_factor: bool,
    /// `relax_initial_tau_factor`: as `relax_tau_factor`, for a new run.
    pub relax_initial_tau_factor: bool,
    /// `relax_to_this_tau_factor` (default `-1`).
    pub relax_to_this_tau_factor: f64,
    /// $\Delta\log_{10}$ `tau_factor` per step (`dlogtau_factor`, default
    /// `0.1`).
    pub dlogtau_factor: f64,
    /// `relax_core`: relax the inner boundary mass to `new_core_mass`
    /// (`star_relax_core`, `star/job/run_star_support.f90:2489-2497`).
    pub relax_core: bool,
    /// `relax_initial_core`: as `relax_core`, for a new run.
    pub relax_initial_core: bool,
    /// `relax_M_center`: relax the total mass to `new_mass` at fixed envelope
    /// mass (`star_relax_M_center`, `star/job/run_star_support.f90:2502-2508`).
    pub relax_m_center: bool,
    /// `relax_initial_M_center`: as `relax_M_center`, for a new run.
    pub relax_initial_m_center: bool,
    /// Target inner core mass in $M_\odot$ (`new_core_mass`, default `0`).
    pub new_core_mass: f64,
    /// $\Delta\log_{10} M_{\rm center}$ per step (`dlg_core_mass_per_step`,
    /// default `1d-3`).
    pub dlg_core_mass_per_step: f64,
    /// Timestep in years for `relax_core` (`relax_core_years_for_dt`,
    /// default `1`).
    pub relax_core_years_for_dt: f64,
    /// Timestep in seconds for `relax_M_center` (`relax_M_center_dt`,
    /// default `3.1558149984d1`).
    pub relax_m_center_dt: f64,
    /// Mean density of inert planetary/stellar core in $\mathrm{g\,cm^{-3}}$ (`core_avg_rho`).
    pub core_avg_rho: f64,
    /// Specific energy generation of inert core in $\mathrm{erg\,g^{-1}\,s^{-1}}$ (`core_avg_eps`).
    pub core_avg_eps: f64,
    /// Whether to relax inner boundary radius (`relax_R_center`).
    pub relax_r_center: bool,
    /// Target inner boundary radius in cm (`new_R_center`).
    pub new_r_center: f64,
    /// Whether to relax inner boundary luminosity (`relax_L_center`).
    pub relax_l_center: bool,
    /// Target inner boundary luminosity in $\mathrm{erg\,s^{-1}}$ (`new_L_center`).
    pub new_l_center: f64,
    /// `relax_irradiation` (`star_relax_irradiation`,
    /// `star/job/run_star_support.f90:2558-2567`).
    pub relax_irradiation: bool,
    /// `relax_initial_irradiation`: as `relax_irradiation`, for a new run.
    pub relax_initial_irradiation: bool,
    /// `relax_irradiation_min_steps` (default `0`).
    pub relax_irradiation_min_steps: i64,
    /// `relax_irradiation_max_yrs_dt` (default `-1`).
    pub relax_irradiation_max_yrs_dt: f64,
    /// Target irradiation flux in $\mathrm{erg\,cm^{-2}\,s^{-1}}$ (`new_irradiation_flux` / `relax_to_this_irrad_flux`).
    pub new_irradiation_flux: f64,
    /// Irradiation attenuation column depth in $\mathrm{g\,cm^{-2}}$ (`column_depth_for_irradiation` / `irrad_col_depth`).
    pub column_depth_for_irradiation: f64,
    /// Whether to relax composition to accretion abundances (`relax_initial_to_xaccrete`).
    pub relax_initial_to_xaccrete: bool,
    /// `relax_initial_mass_to_remove_H_env`: relax the mass to
    /// `he_core_mass + extra_mass_retained_by_remove_H_env`
    /// (`star_relax_mass_to_remove_H_env`,
    /// `star/job/run_star_support.f90:2473-2478`).
    pub relax_initial_mass_to_remove_h_env: bool,
    /// `relax_mass_to_remove_H_env` (`star/job/run_star_support.f90:2428-2433`).
    pub relax_mass_to_remove_h_env: bool,
    /// Mass kept above the He core in $M_\odot$
    /// (`extra_mass_retained_by_remove_H_env`, default `5d-3`).
    pub extra_mass_retained_by_remove_h_env: f64,
    /// `remove_surface_by_relax_to_star_cut`. Not ported: a run that sets it
    /// stops with an error (#436).
    pub remove_surface_by_relax_to_star_cut: bool,
    /// `set_v_center`: set `v_center` to `new_v_center`
    /// (`star/job/run_star_support.f90:2370-2374`).
    pub set_v_center: bool,
    /// `set_initial_v_center`: as `set_v_center`, for a new run.
    pub set_initial_v_center: bool,
    /// `new_v_center` in cm s$^{-1}$ (default `0`).
    pub new_v_center: f64,
    /// `set_L_center`: set `L_center` to `new_L_center` $L_\odot$
    /// (`star/job/run_star_support.f90:2376-2380`).
    pub set_l_center: bool,
    /// `set_initial_L_center`: as `set_L_center`, for a new run.
    pub set_initial_l_center: bool,
    /// `set_irradiation`: set `irradiation_flux` and
    /// `column_depth_for_irradiation` (`star/job/run_star_support.f90:
    /// 2355-2360`).
    pub set_irradiation: bool,
    /// `set_initial_irradiation`: as `set_irradiation`, for a new run.
    pub set_initial_irradiation: bool,
    /// `set_to_this_irrad_flux` in erg cm$^{-2}$ s$^{-1}$ (default `0`).
    pub set_to_this_irrad_flux: f64,
    /// `relax_initial_angular_momentum` (not ported; an error, #436).
    pub relax_initial_angular_momentum: bool,
    /// `relax_initial_entropy` (not ported; an error, #436).
    pub relax_initial_entropy: bool,
    /// Whether to change Rayleigh-Taylor instability flag (`change_initial_RTI_flag`).
    pub change_initial_rti_flag: bool,
    /// Target RTI flag (`new_RTI_flag`).
    pub new_rti_flag: bool,
    /// Termination codes accepted by `test_suite` verification, in subscript
    /// order (`required_termination_code_string(1:9)`,
    /// `star_data/public/star_data_def.inc:369`). An unsubscripted
    /// assignment sets element 1, as a Fortran namelist does.
    pub required_termination_code_string: Vec<String>,
    /// Number of evolutionary steps to take before terminating (`steps_to_take_before_terminate`, `< 0` disables).
    pub steps_to_take_before_terminate: isize,
    /// Inner temperature cut in K (`remove_initial_center_by_temperature`, `<= 0` disables).
    pub remove_initial_center_by_temperature: f64,
    /// Whether `pgstar` live plotting was requested in the inlist (accepted as no-op).
    pub pgstar_flag: bool,
    /// Custom `history_columns.list` path (`history_columns_file` in `star/defaults/star_job.defaults:214`).
    pub history_columns_file: String,
    /// Custom `profile_columns.list` path (`profile_columns_file` in `star/defaults/star_job.defaults:224`).
    pub profile_columns_file: String,
    /// Save an explicit radial profile upon stage termination (`write_profile_when_terminate` in `star_job.defaults:306`).
    pub write_profile_when_terminate: bool,
    /// Output filename when `write_profile_when_terminate` is enabled (`filename_for_profile_when_terminate` in `star_job.defaults:307`).
    pub filename_for_profile_when_terminate: String,
    /// Save a priority-3 profile of the starting model before step 1 (`profile_starting_model` in `star_job.defaults:282`).
    pub profile_starting_model: bool,
}

impl StarJobControls {
    /// Whether a `&star_job` control changes or relaxes $Z$ after the model
    /// is loaded (`change_Z`, `change_initial_Z`, `relax_Z`,
    /// `relax_initial_Z`).
    #[must_use]
    pub fn changes_z(&self) -> bool {
        self.change_z
            || self.change_initial_z
            || self.relax_z
            || self.relax_initial_z
    }

    /// Whether a `&star_job` control changes or relaxes $Y$ after the model
    /// is loaded (`change_Y`, `change_initial_Y`, `relax_Y`,
    /// `relax_initial_Y`).
    #[must_use]
    pub fn changes_y(&self) -> bool {
        self.change_y
            || self.change_initial_y
            || self.relax_y
            || self.relax_initial_y
    }
}

impl Default for StarJobControls {
    fn default() -> Self {
        Self {
            create_pre_ms_model: false,
            pre_ms_t_c: 3.0e5,
            pre_ms_guess_rho_c: 0.0,
            pre_ms_d_log10_p: 0.0,
            pre_ms_log_p_surf_limit: 3.5,
            pre_ms_log_t_surf_limit: 3.7,
            pre_ms_log_lum_factor: 0.0,
            pre_ms_relax_num_steps: 300,
            // star/defaults/star_job.defaults:453, 475-479.
            pre_ms_relax_to_start_radiative_core: true,
            pre_ms_min_steps_before_check_radiative_core: 50,
            pre_ms_check_radiative_core_start: 1.0e-6,
            pre_ms_check_radiative_core_stop: 1.0e-3,
            pre_ms_check_radiative_core_lnuc_div_l_limit: 0.1,
            pre_ms_check_radiative_core_min_mass: 0.3,
            create_initial_model: false,
            radius_in_cm_for_create_initial_model: 0.0,
            mass_in_gm_for_create_initial_model: 0.0,
            // star/defaults/star_job.defaults:535-539, 549, 559, 739.
            center_log_p_1st_try_for_create_initial_model: 10.9,
            entropy_1st_try_for_create_initial_model: 11.5,
            max_tries_for_create_initial_model: 100,
            abs_e01_tolerance_for_create_initial_model: 1.0e-4,
            abs_e02_tolerance_for_create_initial_model: 1.0e-4,
            initial_model_relax_num_steps: 10,
            initial_model_eps: 0.05,
            dump_missing_metals_into_heaviest: true,
            use_other_build_initial_model: false,
            create_rsp_model: false,
            load_saved_model: false,
            saved_model_name: String::new(),
            save_model_when_terminate: false,
            astero_just_call_my_extras_check_model: false,
            save_model_filename: String::from("undefined"),
            save_model_number: -111,
            save_photo_when_terminate: true,
            save_photo_filename: String::new(),
            save_photo_number: -1,
            save_pulse_data_when_terminate: false,
            save_pulse_data_filename: String::from("undefined"),
            pulse_data_format: String::from("FGONG"),
            save_stella_data_when_terminate: false,
            save_stella_data_filename: String::new(),
            net_name: String::from("basic.net"),
            change_net: false,
            new_net_name: String::new(),
            auto_extend_net: true,
            h_he_net: String::from("basic.net"),
            co_net: String::from("co_burn.net"),
            adv_net: String::from("approx21.net"),
            num_special_rate_factors: 0,
            reaction_for_special_factor: vec![String::new(); 50],
            special_rate_factor: vec![1.0; 50],
            filename_of_special_rate: vec![String::new(); 50],
            use_3a_fl87: false,
            use_suzuki_weak_rates: false,
            change_v_flag: false,
            new_v_flag: false,
            change_u_flag: false,
            new_u_flag: false,
            change_rotation_flag: false,
            new_rotation_flag: false,
            set_omega: false,
            set_initial_omega: false,
            new_omega: 0.0,
            set_omega_div_omega_crit: false,
            set_initial_omega_div_omega_crit: false,
            new_omega_div_omega_crit: 0.0,
            set_surface_rotation_v: false,
            set_initial_surface_rotation_v: false,
            new_surface_rotation_v: 0.0,
            relax_omega: false,
            relax_initial_omega: false,
            relax_omega_div_omega_crit: false,
            relax_initial_omega_div_omega_crit: false,
            relax_surface_rotation_v: false,
            relax_initial_surface_rotation_v: false,
            num_steps_to_relax_rotation: 100,
            relax_omega_max_yrs_dt: 1.0e9,
            set_near_zams_omega_steps: -1,
            set_near_zams_omega_div_omega_crit_steps: -1,
            set_near_zams_surface_rotation_v_steps: -1,
            set_omega_step_limit: -1,
            set_omega_div_omega_crit_step_limit: -1,
            set_surf_rotation_v_step_limit: -1,
            near_zams_relax_omega: false,
            near_zams_relax_omega_div_omega_crit: false,
            near_zams_relax_initial_surface_rotation_v: false,
            adjust_tau_factor_to_surf_density: false,
            base_for_adjust_tau_factor_to_surf_density: 0.0,
            set_tau_factor_after_core_he_burn: -1.0,
            set_tau_factor_after_core_c_burn: -1.0,
            relax_tau_factor_after_core_he_burn: -1.0,
            relax_tau_factor_after_core_c_burn: -1.0,
            set_initial_age: false,
            initial_age: 0.0,
            set_initial_model_number: false,
            initial_model_number: 0,
            set_initial_number_retries: true,
            initial_number_retries: 0,
            set_initial_dt: false,
            years_for_initial_dt: -1.0,
            seconds_for_initial_dt: -1.0,
            set_initial_cumulative_energy_error: false,
            new_cumulative_energy_error: 0.0,
            set_cumulative_energy_error_each_relax: true,
            set_uniform_initial_composition: false,
            initial_h1: -1.0,
            initial_h2: -1.0,
            initial_he3: -1.0,
            initial_he4: -1.0,
            initial_zfracs: 3,
            relax_mass: false,
            relax_initial_mass: false,
            relax_mass_scale: false,
            relax_initial_mass_scale: false,
            new_mass: -1.0,
            lg_max_abs_mdot: -100.0,
            dlgm_per_step: 1.0e-3,
            change_mass_years_for_dt: 1.0,
            relax_z: false,
            relax_initial_z: false,
            change_z: false,
            change_initial_z: false,
            relax_z_minq: 0.0,
            relax_z_maxq: 1.0,
            new_z: -1.0,
            relax_y: false,
            relax_initial_y: false,
            change_y: false,
            change_initial_y: false,
            relax_y_minq: 0.0,
            relax_y_maxq: 1.0,
            new_y: -1.0,
            relax_initial_composition: false,
            num_steps_to_relax_composition: 100,
            relax_composition_filename: String::new(),
            timescale_for_relax_composition: -1.0,
            set_tau_factor: false,
            set_initial_tau_factor: false,
            set_to_this_tau_factor: -1.0,
            relax_tau_factor: false,
            relax_initial_tau_factor: false,
            relax_to_this_tau_factor: -1.0,
            dlogtau_factor: 0.1,
            relax_core: false,
            relax_initial_core: false,
            relax_m_center: false,
            relax_initial_m_center: false,
            new_core_mass: 0.0,
            dlg_core_mass_per_step: 1.0e-3,
            relax_core_years_for_dt: 1.0,
            relax_m_center_dt: 3.155_814_998_4e1,
            core_avg_rho: 10.0,
            core_avg_eps: 1.0e-6,
            relax_r_center: false,
            new_r_center: 0.0,
            relax_l_center: false,
            new_l_center: 0.0,
            relax_irradiation: false,
            relax_initial_irradiation: false,
            relax_irradiation_min_steps: 0,
            relax_irradiation_max_yrs_dt: -1.0,
            new_irradiation_flux: 0.0,
            column_depth_for_irradiation: -1.0,
            relax_initial_to_xaccrete: false,
            relax_initial_mass_to_remove_h_env: false,
            relax_mass_to_remove_h_env: false,
            extra_mass_retained_by_remove_h_env: 5.0e-3,
            remove_surface_by_relax_to_star_cut: false,
            set_v_center: false,
            set_initial_v_center: false,
            new_v_center: 0.0,
            set_l_center: false,
            set_initial_l_center: false,
            set_irradiation: false,
            set_initial_irradiation: false,
            set_to_this_irrad_flux: 0.0,
            relax_initial_angular_momentum: false,
            relax_initial_entropy: false,
            change_initial_rti_flag: false,
            new_rti_flag: false,
            required_termination_code_string: Vec::new(),
            steps_to_take_before_terminate: -1,
            remove_initial_center_by_temperature: 0.0,
            pgstar_flag: false,
            history_columns_file: String::new(),
            profile_columns_file: String::new(),
            write_profile_when_terminate: false,
            filename_for_profile_when_terminate: String::new(),
            profile_starting_model: false,
        }
    }
}

const fn default_true() -> bool {
    true
}
