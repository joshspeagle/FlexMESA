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

/// Job-level initialization and model-building controls (`&star_job` namelist
/// in `star_data/private/star_job_controls.inc`).
#[derive(Debug, Clone)]
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
    /// Number of relaxation steps during pre-MS startup.
    pub pre_ms_relax_num_steps: usize,
    /// Whether to construct an isentropic initial planet/WD model (`create_initial_model`).
    pub create_initial_model: bool,
    /// Radius in cm for `create_initial_model`.
    pub radius_in_cm_for_create_initial_model: f64,
    /// Mass in grams for `create_initial_model`.
    pub mass_in_gm_for_create_initial_model: f64,
    /// Whether `use_other_build_initial_model` hook is enabled (`conductive_flame`).
    pub use_other_build_initial_model: bool,
    /// Whether to construct a static radial stellar pulsation envelope (`create_RSP_model`).
    pub create_rsp_model: bool,
    /// Whether to load a saved `.mod` file on startup.
    pub load_saved_model: bool,
    /// Path to saved `.mod` file when `load_saved_model` is true.
    pub saved_model_name: String,
    /// Whether to save a `.mod` file when evolution terminates (`save_model_when_terminate`).
    pub save_model_when_terminate: bool,
    /// Output `.mod` filename on termination (`save_model_filename`).
    pub save_model_filename: String,
    /// Optional model number at which to save a `.mod` file (`save_model_number`).
    pub save_model_number: Option<usize>,
    /// Whether to save a `.photo` restart file on termination (`save_photo_when_terminate`).
    pub save_photo_when_terminate: bool,
    /// Output `.photo` filename on termination (`save_photo_filename`).
    pub save_photo_filename: String,
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
    /// Optional uniform angular velocity $\Omega$ ($\mathrm{rad\,s^{-1}}$) on startup (`new_omega`).
    pub new_omega: Option<f64>,
    /// Optional uniform $\Omega / \Omega_{\mathrm{crit}}$ on startup (`new_omega_div_omega_crit`).
    pub new_omega_div_omega_crit: Option<f64>,
    /// Optional surface equatorial velocity ($\mathrm{km\,s^{-1}}$) on startup (`new_surface_rotation_v`).
    pub new_surface_rotation_v: Option<f64>,
    /// Whether to set initial stellar age (`set_initial_age`).
    pub set_initial_age: bool,
    /// Initial stellar age in years (`initial_age`).
    pub initial_age: f64,
    /// Whether to set initial model number (`set_initial_model_number`).
    pub set_initial_model_number: bool,
    /// Initial model number (`initial_model_number`).
    pub initial_model_number: usize,
    /// Whether to set initial timestep (`set_initial_dt`).
    pub set_initial_dt: bool,
    /// Initial timestep in years (`years_for_initial_dt`).
    pub years_for_initial_dt: f64,
    /// Initial timestep in seconds (`seconds_for_initial_dt`).
    pub seconds_for_initial_dt: f64,
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
    /// Whether to relax stellar mass on startup (`relax_mass` / `relax_initial_mass`).
    pub relax_mass: bool,
    /// Target stellar mass in $M_\odot$ for `relax_mass` (`new_mass`).
    pub new_mass: f64,
    /// Whether to relax metal mass fraction $Z$ (`relax_z` / `change_z`).
    pub relax_z: bool,
    /// Target metal mass fraction $Z$ for `relax_z` (`new_z`).
    pub new_z: f64,
    /// Whether to relax helium mass fraction $Y$ (`relax_y` / `change_y`).
    pub relax_y: bool,
    /// Target helium mass fraction $Y$ for `relax_y` (`new_y`).
    pub new_y: f64,
    /// Whether to set or relax `tau_factor` (`set_tau_factor` / `relax_tau_factor`).
    pub set_tau_factor: bool,
    /// Target `tau_factor` (`set_to_this_tau_factor` / `relax_to_this_tau_factor`).
    pub set_to_this_tau_factor: f64,
    /// Whether to relax inner core boundary (`relax_core` / `relax_M_center`).
    pub relax_m_center: bool,
    /// Target inner core mass in $M_\odot$ (`new_mass` / `new_core_mass`).
    pub new_core_mass: f64,
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
    /// Whether to relax surface irradiation (`relax_irradiation` / `relax_initial_irradiation`).
    pub relax_irradiation: bool,
    /// Target irradiation flux in $\mathrm{erg\,cm^{-2}\,s^{-1}}$ (`new_irradiation_flux` / `relax_to_this_irrad_flux`).
    pub new_irradiation_flux: f64,
    /// Irradiation attenuation column depth in $\mathrm{g\,cm^{-2}}$ (`column_depth_for_irradiation` / `irrad_col_depth`).
    pub column_depth_for_irradiation: f64,
    /// Whether to relax composition to accretion abundances (`relax_initial_to_xaccrete`).
    pub relax_initial_to_xaccrete: bool,
    /// Whether to strip the H-rich envelope on startup (`relax_initial_mass_to_remove_H_env`).
    pub relax_initial_mass_to_remove_h_env: bool,
    /// Whether to set initial central velocity (`set_initial_v_center`).
    pub set_initial_v_center: bool,
    /// Whether to set initial central luminosity (`set_initial_L_center`).
    pub set_initial_l_center: bool,
    /// Whether to change Rayleigh-Taylor instability flag (`change_initial_RTI_flag`).
    pub change_initial_rti_flag: bool,
    /// Target RTI flag (`new_RTI_flag`).
    pub new_rti_flag: bool,
    /// Expected termination code string for `test_suite` verification (`required_termination_code_string`).
    pub required_termination_code_string: String,
    /// Whether `pgstar` live plotting was requested in the inlist (accepted as no-op).
    pub pgstar_flag: bool,
}

impl Default for StarJobControls {
    fn default() -> Self {
        Self {
            create_pre_ms_model: false,
            pre_ms_t_c: 9.0e5,
            pre_ms_guess_rho_c: 0.0,
            pre_ms_d_log10_p: 0.004,
            pre_ms_log_p_surf_limit: 3.5,
            pre_ms_log_t_surf_limit: 3.7,
            pre_ms_log_lum_factor: 0.0,
            pre_ms_relax_num_steps: 20,
            create_initial_model: false,
            radius_in_cm_for_create_initial_model: 0.0,
            mass_in_gm_for_create_initial_model: 0.0,
            use_other_build_initial_model: false,
            create_rsp_model: false,
            load_saved_model: false,
            saved_model_name: String::new(),
            save_model_when_terminate: false,
            save_model_filename: String::new(),
            save_model_number: None,
            save_photo_when_terminate: false,
            save_photo_filename: String::new(),
            save_pulse_data_when_terminate: false,
            save_pulse_data_filename: String::new(),
            pulse_data_format: String::from("GYRE"),
            save_stella_data_when_terminate: false,
            save_stella_data_filename: String::new(),
            net_name: String::from("basic.net"),
            change_net: false,
            new_net_name: String::new(),
            auto_extend_net: false,
            h_he_net: String::from("basic.net"),
            co_net: String::from("co_burn.net"),
            adv_net: String::from("approx21.net"),
            change_v_flag: false,
            new_v_flag: false,
            change_u_flag: false,
            new_u_flag: false,
            change_rotation_flag: false,
            new_rotation_flag: false,
            new_omega: None,
            new_omega_div_omega_crit: None,
            new_surface_rotation_v: None,
            set_initial_age: false,
            initial_age: 0.0,
            set_initial_model_number: false,
            initial_model_number: 0,
            set_initial_dt: false,
            years_for_initial_dt: 0.0,
            seconds_for_initial_dt: 0.0,
            set_uniform_initial_composition: false,
            initial_h1: -1.0,
            initial_h2: -1.0,
            initial_he3: -1.0,
            initial_he4: -1.0,
            initial_zfracs: 3,
            relax_mass: false,
            new_mass: -1.0,
            relax_z: false,
            new_z: -1.0,
            relax_y: false,
            new_y: -1.0,
            set_tau_factor: false,
            set_to_this_tau_factor: 1.0,
            relax_m_center: false,
            new_core_mass: 0.0,
            core_avg_rho: 10.0,
            core_avg_eps: 0.0,
            relax_r_center: false,
            new_r_center: 0.0,
            relax_l_center: false,
            new_l_center: 0.0,
            relax_irradiation: false,
            new_irradiation_flux: 0.0,
            column_depth_for_irradiation: 0.0,
            relax_initial_to_xaccrete: false,
            relax_initial_mass_to_remove_h_env: false,
            set_initial_v_center: false,
            set_initial_l_center: false,
            change_initial_rti_flag: false,
            new_rti_flag: false,
            required_termination_code_string: String::new(),
            pgstar_flag: false,
        }
    }
}
