// SPDX-License-Identifier: LGPL-3.0-or-later
//! Stellar evolution control structs (`StarControls`, `StarJobControls`),
//! variable/equation index layouts (`VarLayout`), and error types (`StarError`)
//! corresponding to `star_data/private/star_controls.inc`,
//! `star_job_controls.inc`, and `star/private/alloc.f90`.

use flexmesa_rates::ScreeningMode;
use flexmesa_turb::{MltOption, SemiconvectionOption, ThermohalineOption};

pub use crate::job_controls::{
    AtmOption, AtmTTauRelation, EnergyEqnOption, StarError, StarJobControls,
    VarLayout,
};

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
    /// Optional custom ZAMS filename (`zams_filename`).
    pub zams_filename: String,
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
    /// Semiconvective mixing prescription (`semiconvection_option`).
    pub semiconvection_option: SemiconvectionOption,
    /// Thermohaline mixing coefficient.
    pub thermohaline_coeff: f64,
    /// Thermohaline mixing prescription (`thermohaline_option`).
    pub thermohaline_option: ThermohalineOption,
    /// Radiation-pressure surface correction factor for `SimplePhotosphere` (`Pextra_factor`).
    pub pextra_factor: f64,
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
    /// Whether latent/gravitational phase-separation heating is active (`do_phase_separation_heating`).
    pub do_phase_separation_heating: bool,
    /// Phase-separation mixture option (`"CO"` or `"ONe"`, `phase_separation_option`).
    pub phase_separation_option: String,
    /// Incident planetary/stellar irradiation flux in $\mathrm{erg\,cm^{-2}\,s^{-1}}$ (`irradiation_flux`).
    pub irradiation_flux: f64,
    /// Attenuation column depth for irradiation in $\mathrm{g\,cm^{-2}}$ (`column_depth_for_irradiation`).
    pub column_depth_for_irradiation: f64,
    /// Whether to use momentum equation at outer boundary $k = 0$.
    pub use_momentum_outer_bc: bool,
    /// Constant mass-change rate $\dot{M}$ in $M_\odot\,\mathrm{yr}^{-1}$ (`mass_change`).
    pub mass_change: f64,
    /// Whether accreted material matches surface composition (`accrete_same_as_surface`).
    pub accrete_same_as_surface: bool,
    /// Whether accreted material uses explicit mass fractions (`accrete_given_mass_fractions`).
    pub accrete_given_mass_fractions: bool,
    /// Accreted $^1\mathrm{H}$ mass fraction (`accretion_h1`).
    pub accretion_h1: f64,
    /// Accreted $^4\mathrm{He}$ mass fraction (`accretion_he4`).
    pub accretion_he4: f64,
    /// Custom accreted isotope identifiers (`accretion_species_id(1..N)`).
    pub accretion_species_id: Vec<String>,
    /// Custom accreted isotope mass fractions (`accretion_species_xa(1..N)`).
    pub accretion_species_xa: Vec<f64>,
    /// Super-Eddington wind scaling factor (`super_eddington_scaling_factor`).
    pub super_eddington_scaling_factor: f64,
    /// Temperature threshold where hot wind turns fully on in K (`hot_wind_full_on_T`).
    pub hot_wind_full_on_t: f64,
    /// Temperature threshold where cool wind turns fully on in K (`cool_wind_full_on_T`).
    pub cool_wind_full_on_t: f64,
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
    /// Upper limit on central Coulomb coupling parameter $\Gamma_c$ (`gamma_center_limit`).
    pub gamma_center_limit: Option<f64>,
    /// Upper limit on $v_{\mathrm{surf}} / v_{\mathrm{esc}}$ (`v_surf_div_v_esc_limit`).
    pub v_surf_div_v_esc_limit: Option<f64>,
    /// Upper limit on $\Omega_{\mathrm{surf}} / \Omega_{\mathrm{crit}}$ (`surf_omega_div_omega_crit_limit`).
    pub surf_omega_div_omega_crit_limit: Option<f64>,
    /// Minimum convective core gap $\Delta q$ (`conv_core_gap_dq_limit`).
    pub conv_core_gap_dq_limit: Option<f64>,
    /// Stop when evolutionary phase reaches TP-AGB (`stop_at_phase_TP_AGB`).
    pub stop_at_phase_tp_agb: bool,
    /// Stop when carbon burning ignites (`stop_at_phase_C_burn`).
    pub stop_at_phase_c_burn: bool,
    /// Stop when neon burning ignites (`stop_at_phase_Ne_burn`).
    pub stop_at_phase_ne_burn: bool,
    /// Stop when oxygen burning ignites (`stop_at_phase_O_burn`).
    pub stop_at_phase_o_burn: bool,
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
    /// Full nonlinear `RSP1` namelist controls (`controls.defaults:3063-3333`).
    pub rsp: crate::rsp_config::RspConfig,
    /// Full coupled Henyey `RSP2` namelist controls (`controls_dev.defaults:59-131`).
    pub rsp2: crate::rsp_config::Rsp2Controls,

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

    // --- Fidelity-1 Hydrodynamic, Energy & `eps_grav` Controls (`hydro_*.f90`, `eps_grav.f90`) ---
    /// Whether velocity time-centering is active (`using_velocity_time_centering`).
    pub using_velocity_time_centering: bool,
    /// Whether luminosity $L_k$ is time-centered during velocity time-centering (`include_L_in_velocity_time_centering`).
    pub include_l_in_velocity_time_centering: bool,
    /// Whether total pressure $P_{\mathrm{eos},k}$ is time-centered (`include_P_in_velocity_time_centering`).
    pub include_p_in_velocity_time_centering: bool,
    /// Time-centering weight $\theta_L \in [0, 1]$ for luminosity (`L_theta_for_velocity_time_centering`).
    pub l_theta_for_velocity_time_centering: f64,
    /// Time-centering weight $\theta_P \in [0, 1]$ for pressure (`P_theta_for_velocity_time_centering`).
    pub p_theta_for_velocity_time_centering: f64,
    /// Maximum $\log_{10}(T/\mathrm{K})$ for $P$ and $L$ time-centering (`max_logT_for_include_P_and_L_in_velocity_time_centering`).
    pub max_logt_for_include_p_and_l_in_velocity_time_centering: f64,
    /// Use $P\,d(1/\rho)/dt$ form of work when time-centering velocity (`use_P_d_1_div_rho_form_of_work_when_time_centering_velocity`).
    pub use_p_d_1_div_rho_form_of_work_when_time_centering_velocity: bool,
    /// Minimum temperature (K) for operator-split burning (`op_split_burn_min_T`).
    pub op_split_burn_min_t: f64,
    /// Whether non-local $^{56}\mathrm{Ni}/^{56}\mathrm{Co}$ decay heating replaces local `eps_nuc` (`nonlocal_NiCo_decay_heat`).
    pub nonlocal_nico_decay_heat: bool,
    /// Whether convective premixing energy release is included (`do_premix_heating`).
    pub do_premix_heating: bool,
    /// TDC turbulent pressure/work coefficient $\alpha_m$ (`TDC_alpha_M`).
    pub tdc_alpha_m: f64,
    /// Include TDC turbulent energy in the energy equation (`TDC_include_eturb_in_energy_equation`).
    pub tdc_include_eturb_in_energy_equation: bool,
    /// Whether Rayleigh-Taylor instability (`RTI_flag`) transport is active.
    pub rti_flag: bool,
    /// Multiplicative factor for RTI internal-energy diffusion (`dedt_RTI_diffusion_factor`).
    pub dedt_rti_diffusion_factor: f64,
    /// Multiplicative factor for RTI momentum diffusion (`dudt_RTI_diffusion_factor`).
    pub dudt_rti_diffusion_factor: f64,
    /// Include velocity drag heating in the energy equation (`use_drag_energy`).
    pub use_drag_energy: bool,
    /// Velocity drag coefficient (`drag_coefficient`).
    pub drag_coefficient: f64,
    /// Minimum enclosed mass fraction $q$ for velocity drag (`min_q_for_drag`).
    pub min_q_for_drag: f64,
    /// MLT turbulent pressure factor $\alpha_{P,\mathrm{turb}}$ (`mlt_Pturb_factor`).
    pub mlt_pturb_factor: f64,
    /// Enable artificial viscosity $P_{\mathrm{vsc}}$ (`use_Pvsc_art_visc`).
    pub use_pvsc_art_visc: bool,
    /// Quadratic artificial viscosity coefficient $C_q$ (`Pvsc_Cq`).
    pub pvsc_cq: f64,
    /// Shock Mach threshold $z_{\mathrm{shock}}$ for artificial viscosity (`Pvsc_zshock`).
    pub pvsc_zshock: f64,
    /// Time-center $\epsilon_{\mathrm{grav}}$ across the step (`use_time_centered_eps_grav`).
    pub use_time_centered_eps_grav: bool,
    /// Include composition time derivatives in $\epsilon_{\mathrm{grav}}$ (`include_composition_in_eps_grav`).
    pub include_composition_in_eps_grav: bool,
    /// Multiplicative factor on $\epsilon_{\mathrm{grav}}$ (`eps_grav_factor`).
    pub eps_grav_factor: f64,
    /// Coulomb $\Gamma$ above which PC $\ln S$ $\epsilon_{\mathrm{grav}}$ is fully on (`gamma_lnS_eps_grav_full_on`).
    pub gamma_ln_s_eps_grav_full_on: f64,
    /// Coulomb $\Gamma$ below which PC $\ln S$ $\epsilon_{\mathrm{grav}}$ is fully off (`gamma_lnS_eps_grav_full_off`).
    pub gamma_ln_s_eps_grav_full_off: f64,
    /// Apply rotational distortion factor $f_P$ to gravity (`use_gravity_rotation_correction`).
    pub use_gravity_rotation_correction: bool,
    /// Apply baryonic mass correction factors (`use_mass_corrections`).
    pub use_mass_corrections: bool,
    /// Use $\nabla_T \Delta\ln P - \Delta\ln T$ form of temperature equation (`use_gradT_actual_vs_gradT_MLT_for_T_gradient_eqn`).
    pub use_gradt_actual_vs_gradt_mlt_for_t_gradient_eqn: bool,
    /// Use $dP_{\mathrm{rad}}/dm$ form of temperature equation (`use_dPrad_dm_form_of_T_gradient_eqn`).
    pub use_dprad_dm_form_of_t_gradient_eqn: bool,
    /// Enable Levermore-Pomraning (1981) flux limiting in `do1_alt_dlnT_dm_eqn` (`use_flux_limiting_with_dPrad_dm_form`).
    pub use_flux_limiting_with_dprad_dm_form: bool,
    /// Minimum face opacity floor in $\mathrm{cm^2\,g^{-1}}$ for `do1_alt_dlnT_dm_eqn` (`min_kap_for_dPrad_dm_eqn`).
    pub min_kap_for_dprad_dm_eqn: f64,
    /// Use RSP Stefan-Boltzmann outer luminosity boundary condition (`use_RSP_L_eqn_outer_BC`).
    pub use_rsp_l_eqn_outer_bc: bool,
    /// Upper $q$ bound above which velocity is zeroed (`velocity_q_upper_bound`).
    pub velocity_q_upper_bound: f64,
    /// Lower $\tau$ bound below which velocity is zeroed (`velocity_tau_lower_bound`).
    pub velocity_tau_lower_bound: f64,
    /// Lower $\log_{10} T$ bound below which velocity is zeroed (`velocity_logT_lower_bound`).
    pub velocity_logt_lower_bound: f64,
    /// Enable nuclear reaction rate terms in `do1_chem_eqns` (`do_burn`).
    pub do_burn: bool,
    /// Enable diffusive mixing terms in `do1_chem_eqns` (`do_mix`).
    pub do_mix: bool,
    /// Multiplicative factor on diffusive mixing $\sigma_k$ (`mix_factor`).
    pub mix_factor: f64,
}

impl Default for StarControls {
    fn default() -> Self {
        Self {
            initial_mass: 1.0,
            initial_z: 0.02,
            initial_y: -1.0,
            zbase: 0.02,
            zams_filename: String::new(),
            max_model_number: None,
            max_age: None,
            max_age_in_seconds: None,
            max_years_for_timestep: None,
            min_timestep_limit: 1.0e-6,
            mixing_length_alpha: 2.0,
            mlt_option: MltOption::Cox,
            use_ledoux_criterion: false,
            alpha_semiconvection: 0.0,
            semiconvection_option: SemiconvectionOption::Langer85GradTIsGradr,
            thermohaline_coeff: 0.0,
            thermohaline_option: ThermohalineOption::Kippenhahn,
            pextra_factor: 1.0,
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
            do_phase_separation_heating: false,
            phase_separation_option: String::from("CO"),
            irradiation_flux: 0.0,
            column_depth_for_irradiation: 0.0,
            use_momentum_outer_bc: false,
            mass_change: 0.0,
            accrete_same_as_surface: true,
            accrete_given_mass_fractions: false,
            accretion_h1: -1.0,
            accretion_he4: -1.0,
            accretion_species_id: Vec::new(),
            accretion_species_xa: Vec::new(),
            super_eddington_scaling_factor: 0.0,
            hot_wind_full_on_t: 1.0e4,
            cool_wind_full_on_t: 1.0e4,
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
            gamma_center_limit: None,
            v_surf_div_v_esc_limit: None,
            surf_omega_div_omega_crit_limit: None,
            conv_core_gap_dq_limit: None,
            stop_at_phase_tp_agb: false,
            stop_at_phase_c_burn: false,
            stop_at_phase_ne_burn: false,
            stop_at_phase_o_burn: false,
            rsp_max_num_periods: None,
            rsp_target_steps_per_cycle: 600,
            rsp_mass: 0.65,
            rsp_teff: 6500.0,
            rsp_l: 45.0,
            rsp_x: 0.75,
            rsp_z: 0.0004,
            rsp: crate::rsp_config::RspConfig::default(),
            rsp2: crate::rsp_config::Rsp2Controls::default(),
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
            using_velocity_time_centering: false,
            include_l_in_velocity_time_centering: false,
            include_p_in_velocity_time_centering: false,
            l_theta_for_velocity_time_centering: 0.5,
            p_theta_for_velocity_time_centering: 0.5,
            max_logt_for_include_p_and_l_in_velocity_time_centering: 99.0,
            use_p_d_1_div_rho_form_of_work_when_time_centering_velocity: true,
            op_split_burn_min_t: 0.0,
            nonlocal_nico_decay_heat: false,
            do_premix_heating: false,
            tdc_alpha_m: 0.0,
            tdc_include_eturb_in_energy_equation: false,
            rti_flag: false,
            dedt_rti_diffusion_factor: 1.0,
            dudt_rti_diffusion_factor: 1.0,
            use_drag_energy: true,
            drag_coefficient: 0.0,
            min_q_for_drag: 0.0,
            mlt_pturb_factor: 0.0,
            use_pvsc_art_visc: false,
            pvsc_cq: 4.0,
            pvsc_zshock: 0.0,
            use_time_centered_eps_grav: false,
            include_composition_in_eps_grav: false,
            eps_grav_factor: 1.0,
            gamma_ln_s_eps_grav_full_on: 160.0,
            gamma_ln_s_eps_grav_full_off: 120.0,
            use_gravity_rotation_correction: true,
            use_mass_corrections: false,
            use_gradt_actual_vs_gradt_mlt_for_t_gradient_eqn: false,
            use_dprad_dm_form_of_t_gradient_eqn: false,
            use_flux_limiting_with_dprad_dm_form: false,
            min_kap_for_dprad_dm_eqn: 1.0e-4,
            use_rsp_l_eqn_outer_bc: false,
            velocity_q_upper_bound: 1.0,
            velocity_tau_lower_bound: 0.0,
            velocity_logt_lower_bound: 0.0,
            do_burn: true,
            do_mix: true,
            mix_factor: 1.0,
        }
    }
}
