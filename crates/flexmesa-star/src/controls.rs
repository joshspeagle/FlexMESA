// SPDX-License-Identifier: GPL-3.0-or-later
//! Stellar evolution control structs (`StarControls`, `StarJobControls`),
//! variable/equation index layouts (`VarLayout`), and error types (`StarError`)
//! corresponding to `star_data/private/star_controls.inc`,
//! `star_job_controls.inc`, and `star/private/alloc.f90`.

use flexmesa_rates::ScreeningMode;
use flexmesa_turb::{MltOption, SemiconvectionOption, ThermohalineOption};

pub use crate::job_controls::{
    AtmIrradiatedOpacity, AtmOption, AtmTTauOpacity, AtmTTauRelation,
    EnergyEqnOption, StarError, StarJobControls, StepFailureInfo, VarLayout,
};

/// Evolutionary and physics controls (`&controls` namelist in
/// `star_data/private/star_controls.inc`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
    /// Maximum stellar age in years (`max_age`, `<= 0` disables).
    pub max_age: f64,
    /// Maximum stellar age in seconds (`max_age_in_seconds`).
    pub max_age_in_seconds: Option<f64>,
    /// Maximum cumulative retries across a run (`max_number_retries`, `< 0` disables).
    pub max_number_retries: isize,
    /// Maximum cumulative retries during relaxation (`relax_max_number_retries`).
    pub relax_max_number_retries: isize,
    /// $\Delta Y$ per step of `relax_Y` (`relax_dY`, default `0.005`,
    /// `star/defaults/controls.defaults:11670`).
    pub relax_dy: f64,
    /// $\Delta\ln Z$ per step of `relax_Z` (`relax_dlnZ`, default
    /// `2.3025850929940459d-1`, `star/defaults/controls.defaults:11681`).
    pub relax_dlnz: f64,
    /// Maximum timestep in years (`max_years_for_timestep`, `<= 0` disables).
    pub max_years_for_timestep: f64,
    /// Minimum timestep limit in seconds (`min_timestep_limit`).
    pub min_timestep_limit: f64,
    /// Mixing-length parameter $\alpha_{\mathrm{MLT}}$ (`2.0` default).
    pub mixing_length_alpha: f64,
    /// Convection theory option (`MltOption::Cox` default).
    pub mlt_option: MltOption,
    /// Whether to use the Ledoux criterion (`true`) or Schwarzschild (`false`).
    pub use_ledoux_criterion: bool,
    /// Semiconvective efficiency parameter $\alpha_{\mathrm{sc}}$.
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
    /// Named atmosphere table when `atm_option == AtmOption::Table` (e.g., `"photosphere"`).
    pub atm_table: String,
    /// Grey $T(\tau)$ relation (`AtmTTauRelation::Eddington` default).
    pub atm_t_tau_relation: AtmTTauRelation,
    /// Grey $T(\tau)$ opacity prescription (`atm_T_tau_opacity`, `AtmTTauOpacity::Fixed` default).
    #[serde(default)]
    pub atm_t_tau_opacity: AtmTTauOpacity,
    /// Error tolerance for `'iterated'` and `'varying'` $T(\tau)$ atmospheres (`atm_T_tau_errtol`, `1e-7` default).
    #[serde(default = "default_atm_t_tau_errtol")]
    pub atm_t_tau_errtol: f64,
    /// Maximum opacity iterations for `'iterated'` $T(\tau)$ atmospheres (`atm_T_tau_max_iters`, `50` default).
    #[serde(default = "default_atm_t_tau_max_iters")]
    pub atm_t_tau_max_iters: usize,
    /// Maximum ODE steps for `'varying'` $T(\tau)$ atmospheres (`atm_T_tau_max_steps`, `500` default).
    #[serde(default = "default_atm_t_tau_max_steps")]
    pub atm_t_tau_max_steps: usize,
    /// Thermal opacity treatment for `'irradiated_grey'` atmospheres (`atm_irradiated_opacity`, `'fixed'` default; `star/defaults/controls.defaults:4540`).
    #[serde(default)]
    pub atm_irradiated_opacity: AtmIrradiatedOpacity,
    /// Error tolerance for `'iterated'` irradiated atmospheres (`atm_irradiated_errtol`, `1e-7` default; `star/defaults/controls.defaults:4551`).
    #[serde(default = "default_atm_irradiated_errtol")]
    pub atm_irradiated_errtol: f64,
    /// Maximum opacity iterations for `'iterated'` irradiated atmospheres (`atm_irradiated_max_iters`, `50` default; `star/defaults/controls.defaults:4562`).
    #[serde(default = "default_atm_irradiated_max_iters")]
    pub atm_irradiated_max_iters: usize,
    /// Equilibrium temperature $T_{\mathrm{eq}}$ ($\mathrm{K}$) from stellar irradiation (`atm_irradiated_T_eq`, `100.0` default; `star/defaults/controls.defaults:4580`).
    #[serde(default = "default_atm_irradiated_t_eq")]
    pub atm_irradiated_t_eq: f64,
    /// Visible-to-thermal opacity ratio $\kappa_v / \kappa_{\mathrm{th}}$ when $> 0$ (`atm_irradiated_kap_v_div_kap_th`, `0.0` default; `star/defaults/controls.defaults:4600`).
    #[serde(default)]
    pub atm_irradiated_kap_v_div_kap_th: f64,
    /// Visible stellar irradiation opacity $\kappa_v$ ($\mathrm{cm^2\,g^{-1}}$) used when `atm_irradiated_kap_v_div_kap_th == 0` (`atm_irradiated_kap_v`, `4e-3` default; `star/defaults/controls.defaults:4601`).
    #[serde(default = "default_atm_irradiated_kap_v")]
    pub atm_irradiated_kap_v: f64,
    /// Surface pressure $P_{\mathrm{surf}}$ ($\mathrm{dyn\,cm^{-2}}$) for `'irradiated_grey'` (`atm_irradiated_P_surf`, `1e6` default; `star/defaults/controls.defaults:4612`).
    #[serde(default = "default_atm_irradiated_p_surf")]
    pub atm_irradiated_p_surf: f64,
    /// Optical depth multiplicative factor $\tau_{\mathrm{factor}}$ (`1.0`).
    pub tau_factor: f64,
    /// Opacity multiplicative factor (`1.0`).
    pub opacity_factor: f64,
    /// Use analytic electron-scattering opacity $\kappa = 0.2(1 + X)$ (`use_simple_es_for_kap`).
    #[serde(default)]
    pub use_simple_es_for_kap: bool,
    /// Evaluate opacity using start-of-step `xa_start` composition (`use_starting_composition_for_kap`).
    #[serde(default)]
    pub use_starting_composition_for_kap: bool,
    /// Upper clamp on Rosseland opacity (`opacity_max`, `<= 0` disables; `-1.0` default).
    #[serde(default = "default_neg_one")]
    pub opacity_max: f64,
    /// Lower clamp on Rosseland opacity (`opacity_min`, `<= 0` disables; `-1.0` default).
    #[serde(default = "default_neg_one")]
    pub opacity_min: f64,
    /// Minimum temperature for thermal neutrino evaluation ($\mathrm{K}$).
    pub min_t_for_neu: f64,
    /// Thermal neutrino multiplicative factor (`1.0`).
    pub non_nuc_neu_factor: f64,
    /// Maximum $\log_{10}(T/\mathrm{K})$ above which network $T$ is clipped (`max_logT_for_net`).
    pub max_logt_for_net: f64,
    /// Lower $\log_{10}(T/\mathrm{K})$ strong-rate zero threshold (`net_logTcut_lo`, `-1.0` default).
    #[serde(default = "default_neg_one")]
    pub net_logtcut_lo: f64,
    /// Upper $\log_{10}(T/\mathrm{K})$ strong-rate smooth cutoff threshold (`net_logTcut_lim`, `-1.0` default).
    #[serde(default = "default_neg_one")]
    pub net_logtcut_lim: f64,
    /// Maximum mean atomic mass $\bar{A}$ for nuclear burning (`max_abar_for_burning`).
    pub max_abar_for_burning: f64,
    /// Whether operator-split 1-zone burning is active (`op_split_burn`).
    pub op_split_burn: bool,
    /// Nuclear energy generation multiplicative factor (`1.0`).
    pub eps_nuc_factor: f64,
    /// Nuclear abundance rate multiplicative factor (`1.0`).
    pub dxdt_nuc_factor: f64,
    /// Global weak-reaction rate multiplier (`weak_rate_factor`, `1.0` default).
    #[serde(default = "default_one")]
    pub weak_rate_factor: f64,
    /// Global reaction neutrino $Q_\nu$ multiplier (`reaction_neuQs_factor`, `1.0` default).
    #[serde(default = "default_one")]
    pub reaction_neuqs_factor: f64,
    /// Magnitude cap on $|\epsilon_{\mathrm{nuc}}|$ before `eps_nuc_factor` (`max_abs_eps_nuc`, `1e99` default).
    #[serde(default = "default_max_abs_eps_nuc")]
    pub max_abs_eps_nuc: f64,
    /// Plasma screening prescription (`ScreeningMode::Chugunov` default).
    pub screening_mode: ScreeningMode,
    /// Whether to compute the Brunt-Väisälä composition term $B$ (`calculate_Brunt_B`, `true` default, `controls.defaults:7915`).
    pub calculate_brunt_b: bool,
    /// Whether to compute the Brunt-Väisälä frequency $N^2$ (`calculate_Brunt_N2`, `true` default).
    pub calculate_brunt_n2: bool,
    /// Whether to use the $\nabla \ln \mu_X$ form for Brunt $B$.
    pub use_brunt_gradmux_form: bool,
    /// Energy equation form (`EnergyEqnOption::Dedt` default).
    pub energy_eqn_option: EnergyEqnOption,
    /// Use the `eps_grav` form of the energy equation while relaxing
    /// (`no_dedt_form_during_relax`, `.true.` default,
    /// `star/defaults/controls.defaults:8005`; `star/private/hydro_energy.f90:402`).
    pub no_dedt_form_during_relax: bool,
    /// Whether adaptive mesh adjustment is permitted (`okay_to_remesh`, `true` default).
    pub okay_to_remesh: bool,
    /// Mesh spacing multiplier `mesh_delta_coeff` (`1.0` default, `0.5` in `star/test`).
    pub mesh_delta_coeff: f64,
    /// Structural variation target `varcontrol_target` (`1e-3` default).
    pub varcontrol_target: f64,
    /// Maximum Newton iterations before rejecting a normal step (`solver_max_tries_before_reject`, `25` default).
    pub solver_max_tries_before_reject: usize,
    /// Maximum Newton iterations on the first model of a run (`max_tries1`, `250` default).
    pub max_tries1: usize,
    /// Maximum Newton iterations on step retries 1..=5 (`max_tries_for_retry`, `25` default).
    pub max_tries_for_retry: usize,
    /// Maximum Newton iterations after 5 retries (`max_tries_after_5_retries`, `35` default).
    pub max_tries_after_5_retries: usize,
    /// Maximum Newton iterations after 10 retries (`max_tries_after_10_retries`, `50` default).
    pub max_tries_after_10_retries: usize,
    /// Maximum Newton iterations after 20 retries (`max_tries_after_20_retries`, `75` default).
    pub max_tries_after_20_retries: usize,
    /// Convergence tolerance on mean scaled correction (`tol_correction_norm`).
    pub tol_correction_norm: f64,
    /// Convergence tolerance on maximum scaled correction (`tol_max_correction`).
    pub tol_max_correction: f64,
    /// Whether hydrodynamic face velocity $v_k$ is an active variable (`v_flag`).
    pub v_flag: bool,
    /// Whether Riemann cell velocity $u_k$ is an active variable (`u_flag`).
    pub u_flag: bool,
    /// Whether rotation $\Omega_k$ is active (`rotation_flag`).
    pub rotation_flag: bool,
    /// Whether nonlinear pulsation (`RSP_flag`) is active.
    pub rsp_flag: bool,
    /// Reset `ln rho` to `dm / dV` before each step when neither `u_flag`
    /// nor `RSP_flag` is set (`set_rho_to_dm_div_dV`, `.true.` default).
    pub set_rho_to_dm_div_dv: bool,
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
    /// Whether outer boundary sets $P_{\mathrm{gas}} = 0$ so $P_{\mathrm{surf}} = P_{\mathrm{rad}}(T_{\mathrm{surf}})$ (`use_zero_Pgas_outer_BC`).
    pub use_zero_pgas_outer_bc: bool,
    /// Whether outer boundary enforces zero compression gradient (`use_compression_outer_BC`).
    pub use_compression_outer_bc: bool,
    /// Whether outer boundary enforces fixed surface pressure `fixed_psurf` (`use_fixed_Psurf_outer_BC`).
    pub use_fixed_psurf_outer_bc: bool,
    /// Fixed surface pressure in $\mathrm{dyn\,cm^{-2}}$ (`fixed_Psurf`).
    pub fixed_psurf: f64,
    /// Whether outer boundary enforces fixed surface velocity `fixed_vsurf` (`use_fixed_vsurf_outer_BC`).
    pub use_fixed_vsurf_outer_bc: bool,
    /// Fixed surface velocity in $\mathrm{cm\,s^{-1}}$ (`fixed_vsurf`).
    pub fixed_vsurf: f64,
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
    /// Upper limit on $\log_{10}(T_c/\mathrm{K})$ (`log_center_temp_upper_limit`, `>= 1e99` disables).
    pub log_center_temp_upper_limit: f64,
    /// Lower limit on $\log_{10}(T_c/\mathrm{K})$ (`log_center_temp_lower_limit`).
    pub log_center_temp_lower_limit: Option<f64>,
    /// Upper limit on $\log_{10}(\rho_c/\mathrm{g\,cm^{-3}})$ (`log_center_density_upper_limit`, `>= 1e99` disables).
    pub log_center_density_upper_limit: f64,
    /// Lower limit on $\log_{10}(\rho_c/\mathrm{g\,cm^{-3}})$ (`log_center_density_lower_limit`).
    pub log_center_density_lower_limit: Option<f64>,
    /// Upper limit on $\max_k \log_{10}(T_k/\mathrm{K})$ (`log_max_temp_upper_limit`, `>= 1e99` disables).
    pub log_max_temp_upper_limit: f64,
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
    /// Lower limit on H-rich envelope mass $M_* - M_{\mathrm{He,core}}$ in $M_\odot$ (`envelope_mass_limit`, `<= 0` disables).
    pub envelope_mass_limit: f64,
    /// Lower limit on He-shell mass $M_{\mathrm{He,core}} - M_{\mathrm{CO,core}}$ in $M_\odot$ (`he_layer_mass_lower_limit`, `<= 0` disables).
    pub he_layer_mass_lower_limit: f64,
    /// Upper limit on total nuclear luminosity in $L_\odot$ (`power_nuc_burn_upper_limit`).
    pub power_nuc_burn_upper_limit: Option<f64>,
    /// Lower limit on total nuclear luminosity in $L_\odot$ (`power_nuc_burn_lower_limit`).
    pub power_nuc_burn_lower_limit: Option<f64>,
    /// Upper limit on He-burning luminosity in $L_\odot$ (`power_he_burn_upper_limit`).
    pub power_he_burn_upper_limit: Option<f64>,
    /// Maximum total stellar mass in $M_\odot$ (`star_mass_max_limit`, `<= 0` disables).
    pub star_mass_max_limit: f64,
    /// Minimum total stellar mass in $M_\odot$ (`star_mass_min_limit`, `<= 0` disables).
    pub star_mass_min_limit: f64,
    /// Minimum remnant/baryonic mass in $M_\odot$ (`remnant_mass_min_limit`, `<= 0` disables).
    pub remnant_mass_min_limit: f64,
    /// Minimum total mass of species `star_species_mass_min_limit_iso` in $M_\odot$ (`star_species_mass_min_limit`, `<= 0` disables).
    pub star_species_mass_min_limit: f64,
    /// Isotope name for `star_species_mass_min_limit` (`star_species_mass_min_limit_iso`).
    pub star_species_mass_min_limit_iso: String,
    /// Maximum total mass of species `star_species_mass_max_limit_iso` in $M_\odot$ (`star_species_mass_max_limit`, `<= 0` disables).
    pub star_species_mass_max_limit: f64,
    /// Isotope name for `star_species_mass_max_limit` (`star_species_mass_max_limit_iso`).
    pub star_species_mass_max_limit_iso: String,
    /// Iron-core infall velocity threshold in $\mathrm{cm\,s^{-1}}$ (`fe_core_infall_limit`, `<= 0` or hydrostatic disables).
    pub fe_core_infall_limit: f64,
    /// Upper limit on $|v_{\mathrm{surf}}| / v_{\mathrm{KH}}$ (`v_surf_div_v_kh_upper_limit`).
    pub v_surf_div_v_kh_upper_limit: Option<f64>,
    /// Upper limit on central Coulomb coupling parameter $\Gamma_c$ (`gamma_center_limit`).
    pub gamma_center_limit: Option<f64>,
    /// Upper limit on $v_{\mathrm{surf}} / v_{\mathrm{esc}}$ (`v_surf_div_v_esc_limit`).
    pub v_surf_div_v_esc_limit: Option<f64>,
    /// Upper limit on $\Omega_{\mathrm{surf}} / \Omega_{\mathrm{crit}}$ (`surf_omega_div_omega_crit_limit`).
    pub surf_omega_div_omega_crit_limit: f64,
    /// Convective-core boundary gap tolerance in fractional mass (`conv_core_gap_dq_limit`).
    pub conv_core_gap_dq_limit: f64,
    /// Stop when evolutionary phase reaches TP-AGB (`stop_at_phase_TP_AGB`).
    pub stop_at_phase_tp_agb: bool,
    /// Stop when carbon burning ignites (`stop_at_phase_C_burn`).
    pub stop_at_phase_c_burn: bool,
    /// Stop when neon burning ignites (`stop_at_phase_Ne_burn`).
    pub stop_at_phase_ne_burn: bool,
    /// Stop when oxygen burning ignites (`stop_at_phase_O_burn`).
    pub stop_at_phase_o_burn: bool,
    /// Maximum pulsation periods for RSP (`RSP_max_num_periods`, `< 0` disables).
    pub rsp_max_num_periods: isize,
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
    /// Enable custom `other_torque_implicit` hook (`use_other_torque_implicit`).
    pub use_other_torque_implicit: bool,
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
    /// Enable custom `other_overshooting` hook (`use_other_overshooting`).
    pub use_other_overshooting_scheme: bool,
    /// Enable custom `other_momentum_implicit` hook (`use_other_momentum_implicit`).
    pub use_other_momentum_implicit: bool,
    /// User floating-point control vector `x_ctrl(1..=100)` (0-based `0..100`).
    pub x_ctrl: Vec<f64>,
    /// User integer control vector `x_integer_ctrl(1..=100)` (0-based `0..100`).
    pub x_integer_ctrl: Vec<i64>,
    /// User logical control vector `x_logical_ctrl(1..=100)` (0-based `0..100`).
    pub x_logical_ctrl: Vec<bool>,
    /// User string control vector `x_character_ctrl(1..=100)` (0-based `0..100`).
    pub x_character_ctrl: Vec<String>,

    // --- Fidelity-1 Hydrodynamic, Energy & `eps_grav` Controls (`hydro_*.f90`, `eps_grav.f90`) ---
    /// Whether velocity time-centering is active (`using_velocity_time_centering`).
    pub using_velocity_time_centering: bool,
    /// Step threshold at or after which velocity time-centering activates (`steps_before_use_velocity_time_centering`, `-1` disables).
    pub steps_before_use_velocity_time_centering: isize,
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
    /// Minimum temperature (K) for operator-split burning (`op_split_burn_min_T`, `2e9` default).
    pub op_split_burn_min_t: f64,
    /// Relative tolerance for 1-zone operator-split burning (`op_split_burn_eps`, `1e-5` default).
    pub op_split_burn_eps: f64,
    /// Abundance scaling floor for 1-zone operator-split burning (`op_split_burn_odescal`, `1e-5` default).
    pub op_split_burn_odescal: f64,
    /// Infall velocity cutoff in $\mathrm{cm\,s^{-1}}$ for operator-split burning (`op_split_burn_eps_nuc_infall_limit`).
    pub op_split_burn_eps_nuc_infall_limit: f64,
    /// Temperature threshold in K for variable-$T$ constant-density 1-zone burning (`op_split_burn_min_T_for_variable_T_solver`).
    pub op_split_burn_min_t_for_variable_t_solver: f64,
    /// Hard substep limit per zone in 1-zone burning (`burn_steps_hard_limit`, `20000` default).
    pub burn_steps_hard_limit: usize,
    /// Whether non-local $^{56}\mathrm{Ni}/^{56}\mathrm{Co}$ decay heating replaces local `eps_nuc` (`nonlocal_NiCo_decay_heat`).
    pub nonlocal_nico_decay_heat: bool,
    /// Whether convective premixing energy release is included (`do_premix_heating`).
    pub do_premix_heating: bool,
    /// Whether to cap diffusive `sig` at `sig_term_limit` (`sig_term_limit`, `1e13` default).
    pub sig_term_limit: f64,
    /// Fake electron-capture scaling factor for $^{56}\mathrm{Fe}$ in `approx21` (`fe56ec_fake_factor`, `1e-7` default).
    pub fe56ec_fake_factor: f64,
    /// Minimum temperature (K) above which `fe56ec_fake_factor` is active (`min_T_for_fe56ec_fake_factor`, `3e9` default).
    pub min_t_for_fe56ec_fake_factor: f64,
    /// Cap convective velocity at `csound * max_conv_vel_div_csound` (`max_conv_vel_div_csound`, `1e99` default).
    pub max_conv_vel_div_csound: f64,
    /// Henyey MLT optical-depth parameter $y$ (`Henyey_MLT_y_param`, `0.33333333` default).
    pub henyey_mlt_y_param: f64,
    /// Henyey MLT velocity parameter $\nu$ (`Henyey_MLT_nu_param`, `8.0` default).
    pub henyey_mlt_nu_param: f64,
    /// Step threshold at or after which TDC replaces Cox MLT (`steps_before_use_tdc`, `0` disables override).
    pub steps_before_use_tdc: isize,
    /// TDC turbulent dissipation coefficient $\alpha_d$ (`TDC_alpha_D`, `1.0` default).
    pub tdc_alpha_d: f64,
    /// TDC radiative damping coefficient $\alpha_r$ (`TDC_alpha_R`, `0.0` default).
    pub tdc_alpha_r: f64,
    /// TDC turbulent pressure $P_t\,dV/dt$ work multiplier $\alpha_{P_t}$ (`TDC_alpha_Pt`, `0.0` default).
    pub tdc_alpha_pt: f64,
    /// TDC convective flux multiplier $\alpha_c$ (`TDC_alpha_C`, `1.0` default).
    pub tdc_alpha_c: f64,
    /// TDC buoyant source multiplier $\alpha_s$ (`TDC_alpha_S`, `1.0` default).
    pub tdc_alpha_s: f64,
    /// TDC eddy-viscous dissipation multiplier $\alpha_m$ (`TDC_alpha_M`, `0.0` default).
    pub tdc_alpha_m: f64,
    /// Include $\Gamma/(1+\Gamma)$ MLT geometric correction in TDC (`include_mlt_corr_to_TDC`, `true` default).
    pub include_mlt_corr_to_tdc: bool,
    /// Include TDC turbulent energy in the energy equation (`TDC_include_eturb_in_energy_equation`).
    pub tdc_include_eturb_in_energy_equation: bool,
    /// Whether Rayleigh-Taylor instability (`RTI_flag`) transport is active.
    pub rti_flag: bool,
    /// Multiplicative factor for $\alpha_{\mathrm{RTI}}$ diffusion (`alpha_RTI_diffusion_factor`).
    pub alpha_rti_diffusion_factor: f64,
    /// Multiplicative factor for RTI internal-energy diffusion (`dedt_RTI_diffusion_factor`).
    pub dedt_rti_diffusion_factor: f64,
    /// Multiplicative factor for RTI momentum diffusion (`dudt_RTI_diffusion_factor`).
    pub dudt_rti_diffusion_factor: f64,
    /// Multiplicative factor for RTI density diffusion (`dlnddt_RTI_diffusion_factor`).
    pub dlnddt_rti_diffusion_factor: f64,
    /// Multiplicative factor for RTI composition diffusion (`composition_RTI_diffusion_factor`).
    pub composition_rti_diffusion_factor: f64,
    /// Minimum $|v|/c_s$ for $\alpha_{\mathrm{RTI}}$ source growth (`alpha_RTI_src_min_v_div_cs`).
    pub alpha_rti_src_min_v_div_cs: f64,
    /// Minimum enclosed mass fraction $q$ for $\alpha_{\mathrm{RTI}}$ source (`alpha_RTI_src_min_q`).
    pub alpha_rti_src_min_q: f64,
    /// Maximum enclosed mass fraction $q$ for $\alpha_{\mathrm{RTI}}$ source (`alpha_RTI_src_max_q`).
    pub alpha_rti_src_max_q: f64,
    /// Duffell (2016) RTI growth coefficient $A$ (`RTI_A`).
    pub rti_a: f64,
    /// Duffell (2016) RTI growth coefficient $B$ (`RTI_B`).
    pub rti_b: f64,
    /// Duffell (2016) RTI diffusivity coefficient $C$ (`RTI_C`).
    pub rti_c: f64,
    /// Duffell (2016) RTI damping coefficient $D$ (`RTI_D`).
    pub rti_d: f64,
    /// Saturation cap on $\alpha_{\mathrm{RTI}}$ (`RTI_max_alpha`).
    pub rti_max_alpha: f64,
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

    // --- Upstream `star_solver.f90` & `sizeB` Convergence Controls ---
    /// Include $L_k$ in `sizeB` correction norms (`include_L_in_correction_limits`).
    pub include_l_in_correction_limits: bool,
    /// Include $v_k$ in `sizeB` correction norms (`include_v_in_correction_limits`).
    pub include_v_in_correction_limits: bool,
    /// Include $u_k$ in `sizeB` correction norms (`include_u_in_correction_limits`).
    pub include_u_in_correction_limits: bool,
    /// Include $w_k$ in `sizeB` correction norms (`include_w_in_correction_limits`).
    pub include_w_in_correction_limits: bool,
    /// Abundance threshold below which `sizeB` excludes species corrections (`correction_xa_limit`).
    pub correction_xa_limit: f64,
    /// Exclude species from `max_correction` (`ignore_species_in_max_correction`).
    pub ignore_species_in_max_correction: bool,
    /// Minimum species variable scale floor (`xa_scale`, `1e-5` default).
    pub xa_scale: f64,
    /// Minimum chemical equation rate scale in $\mathrm{s^{-1}}$ (`min_chem_eqn_scale`, `1.0` default).
    pub min_chem_eqn_scale: f64,
    /// Relative cell internal-energy scale factor in `set_energy_eqn_scal` (`dedt_eqn_r_scale`, `1e8` default).
    pub dedt_eqn_r_scale: f64,
    /// Ignore `i_equ_L` residuals in convergence norm (`convergence_ignore_equL_residuals`).
    pub convergence_ignore_equl_residuals: bool,
    /// Ignore `i_dalpha_RTI_dt` residuals in convergence norm (`convergence_ignore_alpha_RTI_residuals`).
    pub convergence_ignore_alpha_rti_residuals: bool,
    /// Stage-1 non-gold residual norm tolerance (`tol_residual_norm1`).
    pub tol_residual_norm1: f64,
    /// Stage-1 non-gold max residual tolerance (`tol_max_residual1`).
    pub tol_max_residual1: f64,
    /// Iteration threshold for stage-2 non-gold residual tolerances (`iter_for_resid_tol2`).
    pub iter_for_resid_tol2: usize,
    /// Stage-2 non-gold residual norm tolerance (`tol_residual_norm2`).
    pub tol_residual_norm2: f64,
    /// Stage-2 non-gold max residual tolerance (`tol_max_residual2`).
    pub tol_max_residual2: f64,
    /// Iteration threshold for stage-3 non-gold residual tolerances (`iter_for_resid_tol3`).
    pub iter_for_resid_tol3: usize,
    /// Stage-3 non-gold residual norm tolerance (`tol_residual_norm3`).
    pub tol_residual_norm3: f64,
    /// Stage-3 non-gold max residual tolerance (`tol_max_residual3`).
    pub tol_max_residual3: f64,
    /// Enable level-1 gold solver tolerances (`use_gold_tolerances`).
    pub use_gold_tolerances: bool,
    /// Step delay before enabling gold tolerances (`steps_before_use_gold_tolerances`, `-1` disabled).
    pub steps_before_use_gold_tolerances: isize,
    /// Maximum temperature (K) for gold tolerances (`maxT_for_gold_tolerances`).
    pub max_t_for_gold_tolerances: f64,
    /// Stage-1 gold residual norm tolerance (`gold_tol_residual_norm1`).
    pub gold_tol_residual_norm1: f64,
    /// Stage-1 gold max residual tolerance (`gold_tol_max_residual1`).
    pub gold_tol_max_residual1: f64,
    /// Iteration threshold for stage-2 gold tolerances (`gold_iter_for_resid_tol2`).
    pub gold_iter_for_resid_tol2: usize,
    /// Stage-2 gold residual norm tolerance (`gold_tol_residual_norm2`).
    pub gold_tol_residual_norm2: f64,
    /// Stage-2 gold max residual tolerance (`gold_tol_max_residual2`).
    pub gold_tol_max_residual2: f64,
    /// Iteration threshold for stage-3 gold tolerances (`gold_iter_for_resid_tol3`).
    pub gold_iter_for_resid_tol3: usize,
    /// Stage-3 gold residual norm tolerance (`gold_tol_residual_norm3`).
    pub gold_tol_residual_norm3: f64,
    /// Stage-3 gold max residual tolerance (`gold_tol_max_residual3`).
    pub gold_tol_max_residual3: f64,
    /// Enable level-2 gold solver tolerances (`use_gold2_tolerances`).
    pub use_gold2_tolerances: bool,
    /// Step delay before enabling gold2 tolerances (`steps_before_use_gold2_tolerances`).
    pub steps_before_use_gold2_tolerances: isize,
    /// Stage-1 gold2 residual norm tolerance (`gold2_tol_residual_norm1`).
    pub gold2_tol_residual_norm1: f64,
    /// Stage-1 gold2 max residual tolerance (`gold2_tol_max_residual1`).
    pub gold2_tol_max_residual1: f64,
    /// Iteration threshold for stage-2 gold2 tolerances (`gold2_iter_for_resid_tol2`).
    pub gold2_iter_for_resid_tol2: usize,
    /// Stage-2 gold2 residual norm tolerance (`gold2_tol_residual_norm2`).
    pub gold2_tol_residual_norm2: f64,
    /// Stage-2 gold2 max residual tolerance (`gold2_tol_max_residual2`).
    pub gold2_tol_max_residual2: f64,
    /// Iteration threshold for stage-3 gold2 tolerances (`gold2_iter_for_resid_tol3`).
    pub gold2_iter_for_resid_tol3: usize,
    /// Stage-3 gold2 residual norm tolerance (`gold2_tol_residual_norm3`).
    pub gold2_tol_residual_norm3: f64,
    /// Stage-3 gold2 max residual tolerance (`gold2_tol_max_residual3`).
    pub gold2_tol_max_residual3: f64,
    /// Minimum Newton iterations per hydro solve (`solver_itermin`, `2` default).
    pub solver_itermin: usize,
    /// Iteration count before reducing `min_corr_coeff` (`solver_itermin_until_reduce_min_corr_coeff`, `8` default).
    pub solver_itermin_until_reduce_min_corr_coeff: usize,
    /// Reduced `min_corr_coeff` after `solver_itermin_until_reduce_min_corr_coeff` (`solver_reduced_min_corr_coeff`, `0.1` default).
    pub solver_reduced_min_corr_coeff: f64,
    /// Maximum consecutive iterations with `coeff < 1` before rejecting (`tiny_corr_coeff_limit`, `100` default).
    pub tiny_corr_coeff_limit: usize,
    /// Target norm ceiling for step scaling (`scale_correction_norm`, `0.1` default).
    pub scale_correction_norm: f64,
    /// Multiplier on `scale_correction_norm` rejecting excessive average corrections (`corr_param_factor`, `10.0` default).
    pub corr_param_factor: f64,
    /// Maximum correction ceiling before scaling (`scale_max_correction`, `1e99` default).
    pub scale_max_correction: f64,
    /// Bypass `corr_param_factor * scale_correction_norm` rejection (`ignore_too_large_correction`).
    pub ignore_too_large_correction: bool,
    /// Minimum line-search damping coefficient on retries (`corr_coeff_limit`, `1e-2` default).
    pub corr_coeff_limit: f64,
    /// Multiplier defining a tiny line-search damping factor (`tiny_corr_factor`, `2.0` default).
    pub tiny_corr_factor: f64,
    /// Allow `scale_max_correction` to damp below `min_corr_coeff` (`ignore_min_corr_coeff_for_scale_max_correction`).
    pub ignore_min_corr_coeff_for_scale_max_correction: bool,
    /// Maximum iteration-to-iteration increase factor in `correction_norm` (`corr_norm_jump_limit`, `1e99` default).
    pub corr_norm_jump_limit: f64,
    /// Maximum iteration-to-iteration increase factor in `max_correction` (`max_corr_jump_limit`, `1e12` default).
    pub max_corr_jump_limit: f64,
    /// Maximum iteration-to-iteration increase factor in `residual_norm` (`resid_norm_jump_limit`, `1e99` default).
    pub resid_norm_jump_limit: f64,
    /// Maximum iteration-to-iteration increase factor in `max_residual` (`max_resid_jump_limit`, `1e12` default).
    pub max_resid_jump_limit: f64,
    /// Peak temperature threshold (K) relaxing correction tolerances (`tol_correction_high_T_limit`, `1e9` default).
    pub tol_correction_high_t_limit: f64,
    /// High-$T$ `tol_correction_norm` (`tol_correction_norm_high_T`, `3e-3` default).
    pub tol_correction_norm_high_t: f64,
    /// High-$T$ `tol_max_correction` (`tol_max_correction_high_T`, `3e-1` default).
    pub tol_max_correction_high_t: f64,
    /// Extreme peak temperature threshold (K) relaxing correction tolerances (`tol_correction_extreme_T_limit`, `6e9` default).
    pub tol_correction_extreme_t_limit: f64,
    /// Extreme-$T$ `tol_correction_norm` (`tol_correction_norm_extreme_T`, `8e-3` default).
    pub tol_correction_norm_extreme_t: f64,
    /// Extreme-$T$ `tol_max_correction` (`tol_max_correction_extreme_T`, `8e-1` default).
    pub tol_max_correction_extreme_t: f64,
    /// Fallback ceiling on `max_correction` when out of tries (`tol_bad_max_correction`, `0.0` default).
    pub tol_bad_max_correction: f64,
    /// Maximum consecutive steps accepted via `tol_bad_max_correction` (`bad_max_correction_series_limit`, `2` default).
    pub bad_max_correction_series_limit: usize,
    /// Enable gold tolerances during relaxation (`relax_use_gold_tolerances`).
    pub relax_use_gold_tolerances: bool,
    /// Solver iteration timestep limit during relaxation (`relax_solver_iters_timestep_limit`, `0` default).
    pub relax_solver_iters_timestep_limit: isize,
    /// Relaxation override for `tol_correction_norm` (`relax_tol_correction_norm`, `0.0` default).
    pub relax_tol_correction_norm: f64,
    /// Relaxation override for `tol_max_correction` (`relax_tol_max_correction`, `0.0` default).
    pub relax_tol_max_correction: f64,
    /// Relaxation override for `tol_residual_norm1` (`relax_tol_residual_norm1`, `0.0` default).
    pub relax_tol_residual_norm1: f64,
    /// Relaxation override for `tol_max_residual1` (`relax_tol_max_residual1`, `0.0` default).
    pub relax_tol_max_residual1: f64,
    /// Relaxation override for `iter_for_resid_tol2` (`relax_iter_for_resid_tol2`, `3` default).
    pub relax_iter_for_resid_tol2: usize,
    /// Relaxation override for `tol_residual_norm2` (`relax_tol_residual_norm2`, `0.0` default).
    pub relax_tol_residual_norm2: f64,
    /// Relaxation override for `tol_max_residual2` (`relax_tol_max_residual2`, `0.0` default).
    pub relax_tol_max_residual2: f64,
    /// Relaxation override for `iter_for_resid_tol3` (`relax_iter_for_resid_tol3`, `0` default).
    pub relax_iter_for_resid_tol3: usize,
    /// Relaxation override for `tol_residual_norm3` (`relax_tol_residual_norm3`, `0.0` default).
    pub relax_tol_residual_norm3: f64,
    /// Relaxation override for `tol_max_residual3` (`relax_tol_max_residual3`, `0.0` default).
    pub relax_tol_max_residual3: f64,
    /// Relaxation override for `maxT_for_gold_tolerances` (`relax_maxT_for_gold_tolerances`, `-1.0` default).
    pub relax_maxt_for_gold_tolerances: f64,
    /// Hard negative abundance rejection floor (`min_xa_hard_limit`, `-1e-5` default).
    pub min_xa_hard_limit: f64,
    /// High-$T_c$ negative abundance rejection floor (`min_xa_hard_limit_for_highT`, `-3e-5` default).
    pub min_xa_hard_limit_for_hight: f64,
    /// Central $\log_{10} T$ up to which `min_xa_hard_limit` applies (`logT_max_for_min_xa_hard_limit`, `9.49` default).
    pub logt_max_for_min_xa_hard_limit: f64,
    /// Central $\log_{10} T$ above which `min_xa_hard_limit_for_highT` applies (`logT_min_for_min_xa_hard_limit_for_highT`, `9.51` default).
    pub logt_min_for_min_xa_hard_limit_for_hight: f64,
    /// Hard abundance sum error limit $|\sum X_i - 1|$ (`sum_xa_hard_limit`, `5e-4` default).
    pub sum_xa_hard_limit: f64,
    /// High-$T_c$ hard abundance sum error limit (`sum_xa_hard_limit_for_highT`, `1e-3` default).
    pub sum_xa_hard_limit_for_hight: f64,
    /// Central $\log_{10} T$ up to which `sum_xa_hard_limit` applies (`logT_max_for_sum_xa_hard_limit`, `9.40` default).
    pub logt_max_for_sum_xa_hard_limit: f64,
    /// Central $\log_{10} T$ above which `sum_xa_hard_limit_for_highT` applies (`logT_min_for_sum_xa_hard_limit_for_highT`, `9.44` default).
    pub logt_min_for_sum_xa_hard_limit_for_hight: f64,
    /// Uniformly damp Newton corrections to prevent negative mass fractions (`do_solver_damping_for_neg_xa`, `true` default).
    pub do_solver_damping_for_neg_xa: bool,
    /// Scale Newton correction to prevent negative surface luminosity (`scale_max_correction_for_negative_surf_lum`).
    pub scale_max_correction_for_negative_surf_lum: bool,
    /// Maximum fractional drop in surface luminosity when `scale_max_correction_for_negative_surf_lum` is active (`max_frac_for_negative_surf_lum`, `0.8_f32` default).
    pub max_frac_for_negative_surf_lum: f64,
    /// Maximum $|\Delta\log_{10} T|$ allowed in `set_vars_for_solver` (`hydro_mtx_max_allowed_abs_dlogT`, `99.0` default).
    pub hydro_mtx_max_allowed_abs_dlogt: f64,
    /// Maximum $|\Delta\log_{10}\rho|$ allowed in `set_vars_for_solver` (`hydro_mtx_max_allowed_abs_dlogRho`, `99.0` default).
    pub hydro_mtx_max_allowed_abs_dlogrho: f64,
    /// Minimum $\log_{10} T$ gating `hydro_mtx_max_allowed_*` (`min_logT_for_hydro_mtx_max_allowed`, `-1e99` default).
    pub min_logt_for_hydro_mtx_max_allowed: f64,
    /// Maximum $\log_{10} T$ allowed in `set_vars_for_solver` (`hydro_mtx_max_allowed_logT`, `12.0` default).
    pub hydro_mtx_max_allowed_logt: f64,
    /// Maximum $\log_{10}\rho$ allowed in `set_vars_for_solver` (`hydro_mtx_max_allowed_logRho`, `12.0` default).
    pub hydro_mtx_max_allowed_logrho: f64,
    /// Minimum $\log_{10} T$ allowed in `set_vars_for_solver` (`hydro_mtx_min_allowed_logT`, `1.0` default).
    pub hydro_mtx_min_allowed_logt: f64,
    /// Minimum $\log_{10}\rho$ allowed in `set_vars_for_solver` (`hydro_mtx_min_allowed_logRho`, `-1e2` default).
    pub hydro_mtx_min_allowed_logrho: f64,
    /// Abundance floor below which `check1_chem` clips species to zero (`xa_clip_limit`, `1e-99` default).
    pub xa_clip_limit: f64,
    /// Specific energy floor in `RTI_check_after_converge` (`RTI_energy_floor`, `0.0` default).
    pub rti_energy_floor: f64,
    /// Maximum forced solver iterations for accreted surface entropy changes (`max_num_surf_revisions`, `1` default).
    pub max_num_surf_revisions: usize,
    /// Relative surface $\ln S$ change threshold triggering extra accretion iteration (`max_abs_rel_change_surf_lnS`, `5e-4` default).
    pub max_abs_rel_change_surf_lns: f64,
    /// Maximum iterations for implicit stellar wind convergence (`max_tries_for_implicit_wind`, `0` default).
    pub max_tries_for_implicit_wind: usize,
    /// Relative convergence tolerance for implicit stellar wind (`iwind_tolerance`, `1e-3` default).
    pub iwind_tolerance: f64,
    /// Relaxation weight for implicit stellar wind fixed-point/secant loop (`iwind_lambda`, `1.0` default).
    pub iwind_lambda: f64,
    /// Include rotational kinetic energy in `total_energy` and energy accounting
    /// (`include_rotation_in_total_energy`, `star/defaults/controls.defaults:9399`, `false` default).
    #[serde(default)]
    pub include_rotation_in_total_energy: bool,
    /// Accumulate `abs(error_in_energy_conservation)` in `cumulative_energy_error`
    /// (`absolute_cumulative_energy_err`, `star/defaults/controls.defaults:11629`, `true` default).
    #[serde(default = "default_true")]
    pub absolute_cumulative_energy_err: bool,
    /// Overshooting, predictive mixing, premixing, wind, `eps_mdot`, and mesh controls.
    pub mix_wind: crate::controls_mix_wind::MixWindControls,
    /// Core-boundary thresholds, extended stopping criteria, and timestep limits.
    pub ts_stop: crate::controls_timestep_stop::TimestepStopControls,
    /// History, radial profile, and report-averaging output controls (`controls.defaults:216-857`).
    pub output: crate::controls_output::OutputControls,
    /// Element diffusion and white-dwarf phase separation controls (`controls.defaults:6880-7650`).
    pub diffusion: crate::controls_diffusion::DiffusionControls,
    /// Synthetic photometry and SED history controls (`colors/defaults/colors.defaults`).
    #[serde(default)]
    pub colors: ColorsControls,
}

/// Synthetic photometry and spectral-energy-distribution history controls
/// (`&colors` namelist in `colors/defaults/colors.defaults`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ColorsControls {
    /// Enable `&colors` synthetic photometry history columns (`use_colors`, `false` default).
    pub use_colors: bool,
    /// Filter directory / instrument index (`instrument`,
    /// `'data/colors_data/filters/Generic/Johnson'` default).
    pub instrument: String,
    /// Stellar atmosphere SED grid directory (`stellar_atm`,
    /// `'data/colors_data/stellar_models/Kurucz2003all/'` default).
    pub stellar_atm: String,
    /// Vega reference spectrum path (`vega_sed`,
    /// `'data/colors_data/stellar_models/vega_flam.csv'` default).
    pub vega_sed: String,
    /// Observer distance in $\mathrm{cm}$ (`distance`, `3.0857e19` = $10\,\mathrm{pc}$ default).
    pub distance: f64,
    /// Write per-filter SED CSV tables on history output (`make_csv`, `false` default).
    pub make_csv: bool,
    /// Output directory for `make_csv` CSV files (`colors_results_directory`, `'SED'` default).
    pub colors_results_directory: String,
    /// Magnitude zero-point system (`mag_system`: `'Vega'`, `'AB'`, or `'ST'`, `'Vega'` default).
    pub mag_system: String,
    /// Include `_{model_number:08}` suffix in `make_csv` filenames (`sed_per_model`, `false` default).
    pub sed_per_model: bool,
    /// Solar metal-to-hydrogen reference ratio $(Z/X)_\odot$ (`z_over_x_ref`,
    /// `2.30057173972e-2` GS98 default).
    pub z_over_x_ref: f64,
}

impl Default for ColorsControls {
    fn default() -> Self {
        Self {
            use_colors: false,
            instrument: String::from(
                "data/colors_data/filters/Generic/Johnson",
            ),
            stellar_atm: String::from(
                "data/colors_data/stellar_models/Kurucz2003all/",
            ),
            vega_sed: String::from(
                "data/colors_data/stellar_models/vega_flam.csv",
            ),
            distance: flexmesa_colors::D_10PC_CM,
            make_csv: false,
            colors_results_directory: String::from("SED"),
            mag_system: String::from("Vega"),
            sed_per_model: false,
            z_over_x_ref: flexmesa_colors::Z_OVER_X_REF_GS98,
        }
    }
}

impl StarControls {
    /// Returns `true` when velocity time-centering is active at `model_number`
    /// (`star/private/evolve.f90:386-388`).
    ///
    /// Upstream evaluates the threshold on the first try of a step, before
    /// `prepare_for_new_try` advances `model_number` (`evolve.f90:401-402,
    /// 2012`), so callers pass the previous generation's number,
    /// `StarModel::model_number_old`.
    #[must_use]
    pub fn is_velocity_time_centered(&self, model_number: usize) -> bool {
        self.using_velocity_time_centering
            || (self.steps_before_use_velocity_time_centering >= 0
                && (model_number as isize)
                    >= self.steps_before_use_velocity_time_centering)
    }
}

const fn default_true() -> bool {
    true
}

const fn default_neg_one() -> f64 {
    -1.0
}

const fn default_one() -> f64 {
    1.0
}

const fn default_max_abs_eps_nuc() -> f64 {
    1.0e99
}

const fn default_atm_t_tau_errtol() -> f64 {
    1.0e-7
}

const fn default_atm_t_tau_max_iters() -> usize {
    50
}

const fn default_atm_t_tau_max_steps() -> usize {
    500
}

const fn default_atm_irradiated_errtol() -> f64 {
    1.0e-7
}

const fn default_atm_irradiated_max_iters() -> usize {
    50
}

const fn default_atm_irradiated_t_eq() -> f64 {
    100.0
}

const fn default_atm_irradiated_kap_v() -> f64 {
    4.0e-3
}

const fn default_atm_irradiated_p_surf() -> f64 {
    1.0e6
}
