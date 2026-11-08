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
pub struct StarJobControls {
    /// Whether to construct an initial $n = 1.5$ pre-MS polytrope model.
    pub create_pre_ms_model: bool,
    /// Central temperature $T_c$ ($\mathrm{K}$) for `create_pre_ms_model`.
    pub pre_ms_t_c: f64,
    /// Optional initial central density guess $\rho_c$ ($\mathrm{g\,cm^{-3}}$;
    /// $\le 0$ uses the Ushomirsky et al. 1998 analytic $n=1.5$ guess).
    pub pre_ms_guess_rho_c: f64,
    /// Step size $\Delta \log_{10} P_{\mathrm{gas}}$ for outward pre-MS
    /// integration (`0.004` default, `0.02` for coarse grids).
    pub pre_ms_d_log10_p: f64,
    /// Surface $\log_{10} P$ cutoff for pre-MS integration (`3.5` default).
    pub pre_ms_log_p_surf_limit: f64,
    /// Surface $\log_{10} T$ cutoff for pre-MS integration (`3.7` default).
    pub pre_ms_log_t_surf_limit: f64,
    /// Logarithmic luminosity boost factor for pre-MS contraction (`0.0` default).
    pub pre_ms_log_lum_factor: f64,
    /// Number of relaxation steps during pre-MS startup (`100` in MESA).
    pub pre_ms_relax_num_steps: usize,
    /// Whether to load a saved `.mod` file on startup.
    pub load_saved_model: bool,
    /// Path to saved `.mod` file when `load_saved_model` is true.
    pub saved_model_name: String,
    /// Nuclear reaction network name (`"basic.net"` default).
    pub net_name: String,
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
            load_saved_model: false,
            saved_model_name: String::new(),
            net_name: String::from("basic.net"),
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
    /// Whether to use momentum equation at outer boundary $k = 0$.
    pub use_momentum_outer_bc: bool,
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
            use_momentum_outer_bc: false,
            reimers_scaling_factor: 0.0,
            blocker_scaling_factor: 0.0,
            vink_scaling_factor: 0.0,
            dutch_scaling_factor: 0.0,
            cool_wind_rgb_scheme: String::new(),
            cool_wind_agb_scheme: String::new(),
        }
    }
}
