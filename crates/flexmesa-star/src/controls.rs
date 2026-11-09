// SPDX-License-Identifier: LGPL-3.0-or-later
//! Stellar evolution control structs (`StarControls`, `StarJobControls`),
//! variable/equation index layouts (`VarLayout`), and error types (`StarError`)
//! corresponding to `star_data/private/star_controls.inc`,
//! `star_job_controls.inc`, and `star/private/alloc.f90`.

use std::fmt;

use flexmesa_atm::{
    ATM_T_TAU_EDDINGTON, ATM_T_TAU_KRISHNA_SWAMY, ATM_T_TAU_SOLAR_HOPF,
    ATM_T_TAU_TRAMPEDACH_SOLAR,
};
use flexmesa_rates::ScreeningMode;
use flexmesa_turb::MltOption;

pub use crate::job_controls::StarJobControls;

/// Atmosphere boundary condition option (`atm_option` in `star_controls.inc`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtmOption {
    /// Grey $T(\tau)$ relation (`'T_tau'`).
    TTau,
    /// Tabulated atmosphere (`'table'`).
    Table,
    /// Simple photosphere $P_0 = \tau g / \kappa$, $T = T_{\mathrm{eff}}$.
    SimplePhotosphere,
}

/// Grey $T(\tau)$ relation identifier (`atm_T_tau_relation` in `star_controls.inc`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        }
    }
}

impl std::error::Error for StarError {}

/// Energy conservation equation formulation (`energy_eqn_option` in MESA).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
    /// Number of hydrodynamic variables per zone (`4` in hydrostatic mode, `5`
    /// when `v_flag` or `u_flag` is enabled).
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
}

impl VarLayout {
    /// Constructs the variable and equation layout for `(species, v_flag, u_flag)`
    /// matching `set_var_info` in `star/private/alloc.f90:2002-2110`.
    #[must_use]
    pub const fn new(species: usize, v_flag: bool, u_flag: bool) -> Self {
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
        let has_vel = v_flag || u_flag;
        let (i_equ_l, i_dv_dt, i_dlnd_dt, i_dlne_dt, i_dlnr_dt) = if has_vel {
            let iv = if let Some(idx) = i_v {
                idx
            } else if let Some(idx) = i_u {
                idx
            } else {
                4
            };
            (i_lum, iv, i_lnd, i_lnt, Some(i_lnr))
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
            nvar_hydro,
            species,
            nvar_total: nvar_hydro + species,
            i_equ_l,
            i_dv_dt,
            i_dlnd_dt,
            i_dlne_dt,
            i_dlnr_dt,
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
}

/// Evolutionary and physics controls (`&controls` namelist in
/// `star_data/private/star_controls.inc`).
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct StarControls {
    /// Initial stellar mass in solar masses ($M_\odot$).
    pub initial_mass: f64,
    /// Initial metal mass fraction $Z$.
    pub initial_z: f64,
    /// Initial helium mass fraction $Y$ ($< 0$ uses $0.24 + 2 Z$).
    pub initial_y: f64,
    /// Base metallicity $Z_{\mathrm{base}}$ for Type 2 opacities.
    pub zbase: f64,
    /// Maximum model number before stopping (`max_model_number`).
    pub max_model_number: Option<usize>,
    /// Maximum stellar age in years (`max_age`).
    pub max_age: Option<f64>,
    /// Maximum stellar age in seconds (`max_age_in_seconds`).
    pub max_age_in_seconds: Option<f64>,
    /// Maximum timestep in years (`max_years_for_timestep`).
    pub max_years_for_timestep: Option<f64>,
    /// Minimum timestep limit in seconds (`min_timestep_limit`).
    pub min_timestep_limit: f64,
    /// Mixing-length parameter $\alpha_{\mathrm{MLT}}$ (`2.0` default).
    pub mixing_length_alpha: f64,
    /// Convection theory option (`MltOption::Cox` default).
    pub mlt_option: MltOption,
    /// Whether to use the Ledoux criterion (`true`) or Schwarzschild (`false`).
    pub use_ledoux_criterion: bool,
    /// Semiconvection efficiency parameter $\alpha_{\mathrm{sc}}$.
    pub alpha_semiconvection: f64,
    /// Thermohaline mixing coefficient.
    pub thermohaline_coeff: f64,
    /// Minimum diffusive mixing floor $D_{\min}$ ($\mathrm{cm^2\,s^{-1}}$).
    pub min_d_mix: f64,
    /// Whether to remove single-zone mixing glitches (`true` default).
    pub remove_mixing_glitches: bool,
    /// Whether to redo MLT in thin convection zones with $\Delta r < \Lambda$.
    pub redo_conv_for_dr_lt_mixing_length: bool,
    /// Surface boundary condition option (`AtmOption`).
    pub atm_option: AtmOption,
    /// Grey $T(\tau)$ relation (`AtmTTauRelation::Eddington` default).
    pub atm_t_tau_relation: AtmTTauRelation,
    /// Optical depth multiplicative factor $\tau_{\mathrm{factor}}$ (`1.0`).
    pub tau_factor: f64,
    /// Opacity multiplicative factor (`1.0`).
    pub opacity_factor: f64,
    /// Minimum temperature for thermal neutrino evaluation ($\mathrm{K}$).
    pub min_t_for_neu: f64,
    /// Thermal neutrino multiplicative factor (`1.0`).
    pub non_nuc_neu_factor: f64,
    /// Minimum $\log_{10}(T/\mathrm{K})$ for nuclear network evaluation.
    pub min_logt_for_net: f64,
    /// Whether operator-split 1-zone burning is active (`op_split_burn`).
    pub op_split_burn: bool,
    /// Nuclear energy generation multiplicative factor (`1.0`).
    pub eps_nuc_factor: f64,
    /// Nuclear abundance rate multiplicative factor (`1.0`).
    pub dxdt_nuc_factor: f64,
    /// Plasma screening prescription (`ScreeningMode::Extended` default).
    pub screening_mode: ScreeningMode,
    /// Whether to compute the Brunt-Väisälä composition term $B$ and $N^2$.
    pub calculate_brunt_n2: bool,
    /// Whether to use the $\nabla \ln \mu_X$ form for Brunt $B$.
    pub use_brunt_gradmux_form: bool,
    /// Energy equation form (`EnergyEqnOption::Dedt` default).
    pub energy_eqn_option: EnergyEqnOption,
    /// Mesh spacing multiplier `mesh_delta_coeff` (`1.0` default, `0.5` in `star/test`).
    pub mesh_delta_coeff: f64,
    /// Structural variation target `varcontrol_target` (`1e-4` default, `1e-3` in `star/test`).
    pub varcontrol_target: f64,
    /// Maximum Newton iterations per step (`max_tries1`).
    pub max_newton_iters: usize,
    /// Convergence tolerance on mean scaled correction (`tol_correction_norm`).
    pub tol_correction_norm: f64,
    /// Convergence tolerance on maximum scaled correction (`tol_max_correction`).
    pub tol_max_correction: f64,
    /// Convergence tolerance on mean residual (`tol_residual_norm1`).
    pub tol_residual_norm: f64,
    /// Convergence tolerance on maximum residual (`tol_max_residual1`).
    pub tol_max_residual: f64,
    /// Whether hydrodynamic face velocity $v_k$ is an active variable (`v_flag`).
    pub v_flag: bool,
    /// Whether Riemann cell velocity $u_k$ is an active variable (`u_flag`).
    pub u_flag: bool,
    /// Whether rotation $\Omega_k$ is active (`rotation_flag`).
    pub rotation_flag: bool,
    /// Whether nonlinear pulsation (`RSP_flag`) is active.
    pub rsp_flag: bool,
    /// Whether turbulent pulsation energy (`RSP2_flag`) is active.
    pub rsp2_flag: bool,
    /// Whether element diffusion (`do_element_diffusion`) is active.
    pub do_element_diffusion: bool,
    /// Whether convective premixing (`do_conv_premix`) is active.
    pub do_conv_premix: bool,
    /// Whether crystallization phase separation (`do_phase_separation`) is active.
    pub do_phase_separation: bool,
    /// Whether to use momentum equation at outer boundary $k = 0$.
    pub use_momentum_outer_bc: bool,
    /// Constant mass-change rate $\dot{M}$ in $M_\odot\,\mathrm{yr}^{-1}$ (`mass_change`).
    pub mass_change: f64,
    /// Reimers wind scaling factor $\eta_{\mathrm{Reimers}}$.
    pub reimers_scaling_factor: f64,
    /// Blöcker wind scaling factor $\eta_{\mathrm{Blocker}}$.
    pub blocker_scaling_factor: f64,
    /// Vink wind scaling factor $\eta_{\mathrm{Vink}}$.
    pub vink_scaling_factor: f64,
    /// Dutch wind scaling factor $\eta_{\mathrm{Dutch}}$.
    pub dutch_scaling_factor: f64,
    /// Cool RGB wind scheme name (`""`, `"Reimers"`, `"Dutch"`, etc.).
    pub cool_wind_rgb_scheme: String,
    /// Cool AGB wind scheme name (`""`, `"Blocker"`, `"Dutch"`, etc.).
    pub cool_wind_agb_scheme: String,

    // --- Evolutionary Stopping Conditions (`check_model` in `do_one_utils.f90`) ---
    /// Stop when `L_nuc / L >= lnuc_div_l_zams_limit` (`stop_near_zams`).
    pub stop_near_zams: bool,
    /// Threshold $L_{\mathrm{nuc}} / L$ for ZAMS arrival (`Lnuc_div_L_zams_limit`, `0.99` default).
    pub lnuc_div_l_zams_limit: f64,
    /// Central abundance lower-limit species names (`xa_central_lower_limit_species(1..)`).
    pub xa_central_lower_limit_species: Vec<String>,
    /// Central abundance lower-limit thresholds (`xa_central_lower_limit(1..)`).
    pub xa_central_lower_limit: Vec<f64>,
    /// Central abundance upper-limit species names (`xa_central_upper_limit_species(1..)`).
    pub xa_central_upper_limit_species: Vec<String>,
    /// Central abundance upper-limit thresholds (`xa_central_upper_limit(1..)`).
    pub xa_central_upper_limit: Vec<f64>,
    /// Upper limit on $\log_{10}(L / L_\odot)$ (`log_L_upper_limit`).
    pub log_l_upper_limit: Option<f64>,
    /// Lower limit on $\log_{10}(L / L_\odot)$ (`log_L_lower_limit`).
    pub log_l_lower_limit: Option<f64>,
    /// Upper limit on $T_{\mathrm{eff}}$ in K (`Teff_upper_limit`).
    pub teff_upper_limit: Option<f64>,
    /// Lower limit on $T_{\mathrm{eff}}$ in K (`Teff_lower_limit`).
    pub teff_lower_limit: Option<f64>,
    /// Upper limit on $\log_{10}(T_{\mathrm{eff}}/\mathrm{K})$ (`log_Teff_upper_limit`).
    pub log_teff_upper_limit: Option<f64>,
    /// Lower limit on $\log_{10}(T_{\mathrm{eff}}/\mathrm{K})$ (`log_Teff_lower_limit`).
    pub log_teff_lower_limit: Option<f64>,
    /// Upper limit on $\log_{10}(T_c/\mathrm{K})$ (`log_center_temp_upper_limit`).
    pub log_center_temp_upper_limit: Option<f64>,
    /// Lower limit on $\log_{10}(T_c/\mathrm{K})$ (`log_center_temp_lower_limit`).
    pub log_center_temp_lower_limit: Option<f64>,
    /// Upper limit on $\log_{10}(\rho_c/\mathrm{g\,cm^{-3}})$ (`log_center_density_upper_limit`).
    pub log_center_density_upper_limit: Option<f64>,
    /// Lower limit on $\log_{10}(\rho_c/\mathrm{g\,cm^{-3}})$ (`log_center_density_lower_limit`).
    pub log_center_density_lower_limit: Option<f64>,
    /// Upper limit on $\max_k \log_{10}(T_k/\mathrm{K})$ (`log_max_temp_upper_limit`).
    pub log_max_temp_upper_limit: Option<f64>,
    /// Upper limit on $\log_{10} g_{\mathrm{surf}}$ (`log_g_upper_limit`).
    pub log_g_upper_limit: Option<f64>,
    /// Lower limit on $\log_{10} g_{\mathrm{surf}}$ (`log_g_lower_limit`).
    pub log_g_lower_limit: Option<f64>,
    /// Upper limit on $\log_{10}(P_{\mathrm{surf}}/\mathrm{dyn\,cm^{-2}})$ (`log_Psurf_upper_limit`).
    pub log_psurf_upper_limit: Option<f64>,
    /// Upper limit on $\log_{10}(R_{\mathrm{surf}}/R_\odot)$ (`log_Rsurf_upper_limit`).
    pub log_rsurf_upper_limit: Option<f64>,
    /// Lower limit on inner boundary radius $R_{\mathrm{center}} / R_\odot$ (`center_R_lower_limit`).
    pub center_r_lower_limit: Option<f64>,
    /// Upper limit on He-core mass in $M_\odot$ (`he_core_mass_limit`).
    pub he_core_mass_limit: Option<f64>,
    /// Upper limit on CO-core mass in $M_\odot$ (`co_core_mass_limit`).
    pub co_core_mass_limit: Option<f64>,
    /// Upper limit on ONe-core mass in $M_\odot$ (`one_core_mass_limit`).
    pub one_core_mass_limit: Option<f64>,
    /// Lower limit on H-rich envelope mass $M_* - M_{\mathrm{He,core}}$ in $M_\odot$ (`envelope_mass_limit`).
    pub envelope_mass_limit: Option<f64>,
    /// Lower limit on He-shell mass $M_{\mathrm{He,core}} - M_{\mathrm{CO,core}}$ in $M_\odot$ (`he_layer_mass_lower_limit`).
    pub he_layer_mass_lower_limit: Option<f64>,
    /// Upper limit on total nuclear luminosity in $L_\odot$ (`power_nuc_burn_upper_limit`).
    pub power_nuc_burn_upper_limit: Option<f64>,
    /// Lower limit on total nuclear luminosity in $L_\odot$ (`power_nuc_burn_lower_limit`).
    pub power_nuc_burn_lower_limit: Option<f64>,
    /// Upper limit on He-burning luminosity in $L_\odot$ (`power_he_burn_upper_limit`).
    pub power_he_burn_upper_limit: Option<f64>,
    /// Maximum total stellar mass in $M_\odot$ (`star_mass_max_limit`).
    pub star_mass_max_limit: Option<f64>,
    /// Minimum total stellar mass in $M_\odot$ (`star_mass_min_limit`).
    pub star_mass_min_limit: Option<f64>,
    /// Minimum remnant/baryonic mass in $M_\odot$ (`remnant_mass_min_limit`).
    pub remnant_mass_min_limit: Option<f64>,
    /// Minimum total mass of species `star_species_mass_min_limit_iso` in $M_\odot$ (`star_species_mass_min_limit`).
    pub star_species_mass_min_limit: Option<f64>,
    /// Isotope name for `star_species_mass_min_limit` (`star_species_mass_min_limit_iso`).
    pub star_species_mass_min_limit_iso: String,
    /// Maximum total mass of species `star_species_mass_max_limit_iso` in $M_\odot$ (`star_species_mass_max_limit`).
    pub star_species_mass_max_limit: Option<f64>,
    /// Isotope name for `star_species_mass_max_limit` (`star_species_mass_max_limit_iso`).
    pub star_species_mass_max_limit_iso: String,
    /// Iron-core infall velocity threshold in $\mathrm{cm\,s^{-1}}$ (`fe_core_infall_limit`).
    pub fe_core_infall_limit: Option<f64>,
    /// Upper limit on $|v_{\mathrm{surf}}| / v_{\mathrm{KH}}$ (`v_surf_div_v_kh_upper_limit`).
    pub v_surf_div_v_kh_upper_limit: Option<f64>,
    /// Stop when evolutionary phase reaches TP-AGB (`stop_at_phase_TP_AGB`).
    pub stop_at_phase_tp_agb: bool,
    /// Maximum pulsation periods for RSP (`RSP_max_num_periods`).
    pub rsp_max_num_periods: Option<isize>,
    /// Target steps per pulsation cycle (`RSP_target_steps_per_cycle`).
    pub rsp_target_steps_per_cycle: usize,
    /// RSP envelope mass in $M_\odot$ (`RSP_mass`).
    pub rsp_mass: f64,
    /// RSP effective temperature in K (`RSP_Teff`).
    pub rsp_teff: f64,
    /// RSP luminosity in $L_\odot$ (`RSP_L`).
    pub rsp_l: f64,
    /// RSP hydrogen mass fraction $X$ (`RSP_X`).
    pub rsp_x: f64,
    /// RSP metal mass fraction $Z$ (`RSP_Z`).
    pub rsp_z: f64,

    // --- `other_*` Physics Hook Enables & User `x_ctrl` Arrays ---
    /// Enable custom `other_cgrav` hook (`use_other_cgrav`).
    pub use_other_cgrav: bool,
    /// Enable custom `other_alpha_mlt` hook (`use_other_alpha_mlt`).
    pub use_other_alpha_mlt: bool,
    /// Enable custom `other_wind` hook (`use_other_wind`).
    pub use_other_wind: bool,
    /// Enable custom `other_adjust_mdot` hook (`use_other_adjust_mdot`).
    pub use_other_adjust_mdot: bool,
    /// Enable custom `other_torque` hook (`use_other_torque`).
    pub use_other_torque: bool,
    /// Enable custom `other_energy` hook (`use_other_energy`).
    pub use_other_energy: bool,
    /// Enable custom `other_eps_grav` hook (`use_other_eps_grav`).
    pub use_other_eps_grav: bool,
    /// Enable custom `other_neu` hook (`use_other_neu`).
    pub use_other_neu: bool,
    /// Enable custom `other_kap` hook (`use_other_kap`).
    pub use_other_kap: bool,
    /// Enable custom `other_surface_PT` hook (`use_other_surface_PT`).
    pub use_other_surface_pt: bool,
    /// Enable custom `other_overshooting_scheme` hook (`use_other_overshooting_scheme`).
    pub use_other_overshooting_scheme: bool,
    /// Enable custom `other_momentum_implicit` hook (`use_other_momentum_implicit`).
    pub use_other_momentum_implicit: bool,
    /// User floating-point control vector `x_ctrl(1..99)` (0-based `0..100`).
    pub x_ctrl: Vec<f64>,
    /// User integer control vector `x_integer_ctrl(1..99)` (0-based `0..100`).
    pub x_integer_ctrl: Vec<i64>,
    /// User logical control vector `x_logical_ctrl(1..99)` (0-based `0..100`).
    pub x_logical_ctrl: Vec<bool>,
    /// User string control vector `x_character_ctrl(1..99)` (0-based `0..100`).
    pub x_character_ctrl: Vec<String>,
}

impl Default for StarControls {
    fn default() -> Self {
        Self {
            initial_mass: 1.0,
            initial_z: 0.02,
            initial_y: -1.0,
            zbase: 0.02,
            max_model_number: None,
            max_age: None,
            max_age_in_seconds: None,
            max_years_for_timestep: None,
            min_timestep_limit: 1.0e-6,
            mixing_length_alpha: 2.0,
            mlt_option: MltOption::Cox,
            use_ledoux_criterion: false,
            alpha_semiconvection: 0.0,
            thermohaline_coeff: 0.0,
            min_d_mix: 0.0,
            remove_mixing_glitches: true,
            redo_conv_for_dr_lt_mixing_length: false,
            atm_option: AtmOption::TTau,
            atm_t_tau_relation: AtmTTauRelation::Eddington,
            tau_factor: 1.0,
            opacity_factor: 1.0,
            min_t_for_neu: 1.0e7,
            non_nuc_neu_factor: 1.0,
            min_logt_for_net: 6.3,
            op_split_burn: false,
            eps_nuc_factor: 1.0,
            dxdt_nuc_factor: 1.0,
            screening_mode: ScreeningMode::Extended,
            calculate_brunt_n2: true,
            use_brunt_gradmux_form: false,
            energy_eqn_option: EnergyEqnOption::Dedt,
            mesh_delta_coeff: 1.0,
            varcontrol_target: 1.0e-4,
            max_newton_iters: 25,
            tol_correction_norm: 3.0e-5,
            tol_max_correction: 3.0e-3,
            tol_residual_norm: 1.0e-4,
            tol_max_residual: 1.0e-2,
            v_flag: false,
            u_flag: false,
            rotation_flag: false,
            rsp_flag: false,
            rsp2_flag: false,
            do_element_diffusion: false,
            do_conv_premix: false,
            do_phase_separation: false,
            use_momentum_outer_bc: false,
            mass_change: 0.0,
            reimers_scaling_factor: 0.0,
            blocker_scaling_factor: 0.0,
            vink_scaling_factor: 0.0,
            dutch_scaling_factor: 0.0,
            cool_wind_rgb_scheme: String::new(),
            cool_wind_agb_scheme: String::new(),
            stop_near_zams: false,
            lnuc_div_l_zams_limit: 0.99,
            xa_central_lower_limit_species: Vec::new(),
            xa_central_lower_limit: Vec::new(),
            xa_central_upper_limit_species: Vec::new(),
            xa_central_upper_limit: Vec::new(),
            log_l_upper_limit: None,
            log_l_lower_limit: None,
            teff_upper_limit: None,
            teff_lower_limit: None,
            log_teff_upper_limit: None,
            log_teff_lower_limit: None,
            log_center_temp_upper_limit: None,
            log_center_temp_lower_limit: None,
            log_center_density_upper_limit: None,
            log_center_density_lower_limit: None,
            log_max_temp_upper_limit: None,
            log_g_upper_limit: None,
            log_g_lower_limit: None,
            log_psurf_upper_limit: None,
            log_rsurf_upper_limit: None,
            center_r_lower_limit: None,
            he_core_mass_limit: None,
            co_core_mass_limit: None,
            one_core_mass_limit: None,
            envelope_mass_limit: None,
            he_layer_mass_lower_limit: None,
            power_nuc_burn_upper_limit: None,
            power_nuc_burn_lower_limit: None,
            power_he_burn_upper_limit: None,
            star_mass_max_limit: None,
            star_mass_min_limit: None,
            remnant_mass_min_limit: None,
            star_species_mass_min_limit: None,
            star_species_mass_min_limit_iso: String::new(),
            star_species_mass_max_limit: None,
            star_species_mass_max_limit_iso: String::new(),
            fe_core_infall_limit: None,
            v_surf_div_v_kh_upper_limit: None,
            stop_at_phase_tp_agb: false,
            rsp_max_num_periods: None,
            rsp_target_steps_per_cycle: 150,
            rsp_mass: 0.65,
            rsp_teff: 6500.0,
            rsp_l: 45.0,
            rsp_x: 0.75,
            rsp_z: 0.0004,
            use_other_cgrav: false,
            use_other_alpha_mlt: false,
            use_other_wind: false,
            use_other_adjust_mdot: false,
            use_other_torque: false,
            use_other_energy: false,
            use_other_eps_grav: false,
            use_other_neu: false,
            use_other_kap: false,
            use_other_surface_pt: false,
            use_other_overshooting_scheme: false,
            use_other_momentum_implicit: false,
            x_ctrl: vec![0.0; 100],
            x_integer_ctrl: vec![0; 100],
            x_logical_ctrl: vec![false; 100],
            x_character_ctrl: vec![String::new(); 100],
        }
    }
}
