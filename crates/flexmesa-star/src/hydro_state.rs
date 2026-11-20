// SPDX-License-Identifier: LGPL-3.0-or-later
//! Auxiliary start-of-step thermodynamic arrays, heating channels, turbulent
//! work/viscosity variables, rotational distortion factors, Rayleigh-Taylor
//! instability (RTI) profiles, and EOS composition partial derivatives on
//! [`crate::StarModel`] (`star_data_step_work.inc` and `star/private/alloc.f90`).

use crate::star_ad::{StarAdExt, StarAdOrder1};

/// Per-step scalar history, solver/substep counters, and retry-hold state
/// consumed by [`crate::timestep_limits::timestep_controller`] and
/// [`crate::evolve`] (`star_data_step_work.inc`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StepHistoryScalars {
    /// Previous step effective temperature $T_{\mathrm{eff,old}}$ in K (`Teff_old`).
    pub teff_old: f64,
    /// Previous step surface luminosity $L_{\mathrm{surf,old}} / L_\odot$ (`L_surf_old`).
    pub l_surf_old: f64,
    /// Previous step mass-change rate $\dot{M}_{\mathrm{old}}$ in $\mathrm{g\,s^{-1}}$ (`mstar_dot_old`).
    pub mstar_dot_old: f64,
    /// Previous step H-burning luminosity in $L_\odot$ (`power_h_burn_old`).
    pub power_h_burn_old: f64,
    /// Previous step He-burning luminosity in $L_\odot$ (`power_he_burn_old`).
    pub power_he_burn_old: f64,
    /// Previous step Z-burning luminosity in $L_\odot$ (`power_z_burn_old`).
    pub power_z_burn_old: f64,
    /// Previous step photodisintegration luminosity in $L_\odot$ (`power_photo_old`).
    pub power_photo_old: f64,
    /// Previous step total nuclear luminosity in $L_\odot$ (`power_nuc_burn_old`).
    pub power_nuc_burn_old: f64,
    /// Previous step rotational mass-loss boost factor (`rotational_mdot_boost`).
    pub rotational_mdot_boost_old: f64,
    /// Whether the previous step invoked the critical-rotation implicit wind limit (`was_in_implicit_wind_limit`).
    pub was_in_implicit_wind_limit: bool,
    /// Newton iterations used on the just-completed step (`num_solver_iterations`).
    pub last_solver_iterations: usize,
    /// Peak 1-zone burn substeps across zones on the current step (`burn_num_iters`).
    pub burn_max_num_substeps: usize,
    /// Element diffusion substeps taken on the current step (`num_diffusion_solver_steps`).
    pub num_diffusion_solver_steps: usize,
    /// Element diffusion solver iterations taken on the current step (`num_diffusion_solver_iters`).
    pub num_diffusion_solver_iters: usize,
    /// Model-number ceiling below which $\Delta t$ is held non-increasing (`timestep_hold`).
    pub timestep_hold: usize,
    /// Model number at which the most recent step retry occurred (`model_number_for_last_retry`).
    pub model_number_for_last_retry: Option<usize>,
    /// Per-step global energy conservation residual in $\mathrm{erg}$ (`error_in_energy_conservation`).
    pub error_in_energy_conservation: f64,
    /// End-of-step total energy in $\mathrm{erg}$ (`total_energy_end`).
    pub total_energy_end: f64,
    /// Previous accepted step total energy in $\mathrm{erg}$ for rollback on retry.
    pub total_energy_old: f64,
    /// Previous accepted step cumulative energy error in $\mathrm{erg}$ for rollback on retry.
    pub cumulative_energy_error_old: f64,
    /// Start-of-step optical-depth-averaged Eddington luminosity in $\mathrm{erg\,s^{-1}}$ (`prev_Ledd`, `evolve.f90:1787`).
    pub prev_ledd: Option<f64>,
    /// Start-of-step central temperature in K (`starting_T_center`, `evolve.f90:1784`).
    pub starting_t_center: Option<f64>,
    /// Super-Eddington wind rate $\dot{M}_{\mathrm{sEdd}} \ge 0$ in $\mathrm{g\,s^{-1}}$ (`super_eddington_wind_mdot`, `winds.f90:653`).
    pub super_eddington_wind_mdot: f64,
    /// Whether single-star RLO wind is active on the current step (`doing_rlo_wind`, `winds.f90:110`).
    pub doing_rlo_wind: bool,
    /// Explicit mass-change rate $\dot{M}_{\mathrm{explicit}}$ in $\mathrm{g\,s^{-1}}$ (`explicit_mstar_dot`, `winds.f90:304`).
    pub explicit_mstar_dot: f64,
    /// Evolutionary phase integer code (`phase_of_evolution`, `star_utils.f90:2910-2969`).
    pub phase_of_evolution: i32,
    /// Exact stellar age in seconds (`time`, `evolve.f90:2079-2143`).
    pub time_sec: f64,
    /// Previous-step stellar age in seconds (`time_old`).
    pub time_old_sec: f64,
    /// Whether graded `max_age` timestep reduction is currently active (`using_revised_max_yr_dt`).
    pub using_revised_max_yr_dt: bool,
    /// Revised maximum timestep in years for graded `max_age` approach (`revised_max_yr_dt`).
    pub revised_max_yr_dt: f64,
    /// Cumulative integrated extra heating in $\mathrm{erg}$ (`cumulative_extra_heating`).
    pub cumulative_extra_heating: f64,
    /// Canonical MESA `termination_code_str` set when `do_check_limits`
    /// terminates an accepted step inside `star_evolve_step_with_extras`
    /// (`star/job/run_star_support.f90:211-233`).
    #[serde(skip)]
    pub termination_code: Option<String>,
}

impl Default for StepHistoryScalars {
    fn default() -> Self {
        Self {
            teff_old: 0.0,
            l_surf_old: -1.0,
            mstar_dot_old: 0.0,
            power_h_burn_old: -1.0,
            power_he_burn_old: -1.0,
            power_z_burn_old: -1.0,
            power_photo_old: -1.0,
            power_nuc_burn_old: -1.0,
            rotational_mdot_boost_old: 1.0,
            was_in_implicit_wind_limit: false,
            last_solver_iterations: 0,
            burn_max_num_substeps: 0,
            num_diffusion_solver_steps: 0,
            num_diffusion_solver_iters: 0,
            timestep_hold: 0,
            model_number_for_last_retry: None,
            error_in_energy_conservation: 0.0,
            total_energy_end: 0.0,
            total_energy_old: 0.0,
            cumulative_energy_error_old: 0.0,
            prev_ledd: None,
            starting_t_center: None,
            super_eddington_wind_mdot: 0.0,
            doing_rlo_wind: false,
            explicit_mstar_dot: 0.0,
            phase_of_evolution: 1,
            time_sec: 0.0,
            time_old_sec: 0.0,
            using_revised_max_yr_dt: false,
            revised_max_yr_dt: 0.0,
            cumulative_extra_heating: 0.0,
            termination_code: None,
        }
    }
}

/// Shell-level binary Roche tidal distortion output supplied by an external
/// binary driver (`binary/private/binary_roche_deformation.f90`).
#[derive(Debug, Clone, Copy, Default)]
pub struct TidalShellDeformation {
    /// Tidal pressure distortion factor $f_P$.
    pub fp: f64,
    /// Logarithmic radius derivative $\partial f_P / \partial \ln r$.
    pub d_fp_dlnr: f64,
    /// Tidal temperature distortion factor $f_T$.
    pub ft: f64,
    /// Logarithmic radius derivative $\partial f_T / \partial \ln r$.
    pub d_ft_dlnr: f64,
    /// Tidal specific moment of inertia $i_{\mathrm{rot}}$ in $\mathrm{cm^2}$.
    pub i_rot: f64,
    /// Logarithmic radius derivative $\partial i_{\mathrm{rot}} / \partial \ln r$ in $\mathrm{cm^2}$.
    pub d_irot_dlnr: f64,
    /// Synchronicity blending weight $f_{\mathrm{switch}} \in [0, 1]$ (`1` = synchronous Roche, `0` = single-star).
    pub f_switch: f64,
}

/// Acyclic callback trait allowing `flexmesa-binary` to supply full Roche-lobe
/// tidal distortion factors $(f_P, f_T, i_{\mathrm{rot}})$ to `flexmesa-star`
/// (`use_tidal_deformation`).
pub trait TidalDeformationHook: Send + Sync {
    /// Evaluates the Roche tidal distortion and synchronicity blending factor
    /// at shell index `k` with volume-equivalent radius `r_cm` ($\mathrm{cm}$)
    /// and angular velocity `omega` ($\mathrm{rad\,s^{-1}}$).
    fn eval_shell(
        &self,
        k: usize,
        r_cm: f64,
        omega: f64,
    ) -> TidalShellDeformation;
}

/// Cloneable reference-counted handle wrapping a [`TidalDeformationHook`].
#[derive(Clone)]
pub struct TidalDeformationHandle(pub std::sync::Arc<dyn TidalDeformationHook>);

impl std::fmt::Debug for TidalDeformationHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("TidalDeformationHandle(..)")
    }
}

/// Auxiliary hydrodynamic, thermodynamic, and heating state vectors for
/// `flexmesa-star` (`star_data_step_work.inc`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HydroAuxState {
    /// Whether the model is currently in an initialization relaxation phase
    /// (`doing_relax` in `star_data_def.f90`).
    pub doing_relax: bool,
    /// Whether the model is currently inside `do_struct_burn_mix`
    /// (`doing_struct_burn_mix` in `star_data_step_work.inc`).
    pub doing_struct_burn_mix: bool,
    /// Total integrated model energy $E_{\mathrm{tot}}$ in $\mathrm{erg}$
    /// (`total_energy` in `star_data_step_work.inc`).
    pub total_energy: f64,
    /// Start-of-step total integrated internal energy $\sum_k \Delta m_k e_{\mathrm{start},k}$
    /// in $\mathrm{erg}$ (`total_internal_energy_old` in `star_utils.f90:3706`).
    pub total_internal_energy_start: f64,
    /// Cumulative global energy conservation error in $\mathrm{erg}$
    /// (`cumulative_energy_error` in `star_data_step_work.inc`).
    pub cumulative_energy_error: f64,
    /// Internal energy change from operator-split composition change at fixed
    /// $(\rho, T)$ (`non_epsnuc_energy_change_from_split_burn`).
    pub non_epsnuc_energy_change_from_split_burn: f64,
    /// Start-of-step specific heat at constant pressure $C_{P,\mathrm{start},k}$ (`Cp_start`).
    pub cp_start: Vec<f64>,
    /// Start-of-step adiabatic temperature gradient $\nabla_{\mathrm{ad,start},k}$ (`grada_start`).
    pub grada_start: Vec<f64>,
    /// Start-of-step temperature exponent $\chi_{T,\mathrm{start},k}$ (`chiT_start`).
    pub chi_t_start: Vec<f64>,
    /// Start-of-step density exponent $\chi_{\rho,\mathrm{start},k}$ (`chiRho_start`).
    pub chi_rho_start: Vec<f64>,
    /// Fraction of Potekhin-Chabrier (`PC`) EOS in cell `k` (`eos_frac_PC`).
    pub eos_frac_pc: Vec<f64>,
    /// Opacity regime blend fractions `[lowT, highT, Type2, Compton]` per cell (`kap_frac_*`).
    #[serde(default)]
    pub kap_frac: Vec<[f64; 4]>,
    /// Unscaled thermal neutrino channel losses `[plas, brem, phot, pair, reco]` per cell (`nonnucneu_*`).
    #[serde(default)]
    pub nonnucneu_channels: Vec<[f64; 5]>,
    /// Phase-transition latent heat density derivative (`latent_ddlnRho`).
    pub latent_ddlnrho: Vec<f64>,
    /// Phase-transition latent heat temperature derivative (`latent_ddlnT`).
    pub latent_ddlnt: Vec<f64>,
    /// Composition partial derivatives $\partial\epsilon_{\mathrm{grav},k}/\partial X_{j,k}$
    /// (`d_eps_grav_dx`, `len = nz * species`).
    pub d_eps_grav_dx: Vec<f64>,
    /// Operator-split time-averaged nuclear energy rate $\bar{\epsilon}_{\mathrm{nuc},k}$
    /// (`burn_avg_epsnuc`).
    pub burn_avg_epsnuc: Vec<f64>,
    /// Category-resolved nuclear energy generation rates (`eps_nuc_categories`,
    /// `len = nz`).
    pub eps_nuc_categories: Vec<[f64; flexmesa_chem::NUM_CATEGORIES]>,
    /// Extra specific heating rate $\epsilon_{\mathrm{extra},k}$ in $\mathrm{erg\,g^{-1}\,s^{-1}}$ (`extra_heat`).
    pub extra_heat: Vec<f64>,
    /// Partial derivative $\partial\epsilon_{\mathrm{extra},k}/\partial\ln\rho_k$ (`d_extra_heat_dlnd`).
    pub d_extra_heat_dlnd: Vec<f64>,
    /// Partial derivative $\partial\epsilon_{\mathrm{extra},k}/\partial\ln T_k$ (`d_extra_heat_dlnT`).
    pub d_extra_heat_dlnt: Vec<f64>,
    /// White-dwarf gravitational sedimentation heating $\epsilon_{\mathrm{WD,sed},k}$ (`eps_WD_sedimentation`).
    pub eps_wd_sedimentation: Vec<f64>,
    /// Element diffusion heating rate $\epsilon_{\mathrm{diff},k}$ (`eps_diffusion`).
    pub eps_diffusion: Vec<f64>,
    /// Convective premixing heating rate $\epsilon_{\mathrm{premix},k}$ (`eps_pre_mix`).
    pub eps_pre_mix: Vec<f64>,
    /// Crystallization phase-separation heating rate $\epsilon_{\mathrm{phase},k}$ (`eps_phase_separation`).
    pub eps_phase_separation: Vec<f64>,
    /// External irradiation specific heating rate $\epsilon_{\mathrm{irrad},k}$ (`irradiation_heat`).
    pub irradiation_heat: Vec<f64>,
    /// Turbulent energy work cell heating $E_{q,k}$ (`Eq_ad`).
    pub eq_ad: Vec<StarAdOrder1>,
    /// Turbulent specific kinetic energy $e_{\mathrm{trb},k}$ in $\mathrm{erg\,g^{-1}}$ (`etrb`).
    pub etrb: Vec<f64>,
    /// Start-of-step turbulent specific kinetic energy $e_{\mathrm{trb,start},k}$ (`etrb_start`).
    pub etrb_start: Vec<f64>,
    /// Previous-step `RSP2` turbulent energy $e_{t,\mathrm{old},k}$ for step retry (`et_rsp_old`).
    pub et_rsp_old: Vec<f64>,
    /// Previous-step `RSP2` pressure scale height $H_{P,\mathrm{old},k}$ for step retry (`hp_face_old`).
    pub hp_face_old: Vec<f64>,
    /// Start-of-step adiabatic sound speed $c_{s,\mathrm{start},k}$ (`csound_start`).
    pub csound_start: Vec<f64>,
    /// Start-of-step face convective velocity $v_{c,\mathrm{old},k}$ in $\mathrm{cm\,s^{-1}}$ (`mlt_vc_old`).
    pub mlt_vc_old: Vec<f64>,
    /// Whether `mlt_vc` holds valid convective velocities from the prior step (`have_mlt_vc`).
    pub have_mlt_vc: bool,
    /// Whether MLT/TDC evaluations may overwrite `mlt_vc` (`okay_to_set_mlt_vc` in `evolve.f90:321`).
    pub okay_to_set_mlt_vc: bool,
    /// Constant-mass inner boundary zone index during mass adjustment (`k_const_mass`).
    pub k_const_mass: usize,
    /// Rayleigh-Taylor instability face kinematic diffusivity $\eta_{\mathrm{RTI},k}$ (`eta_RTI`).
    pub eta_rti: Vec<f64>,
    /// Rayleigh-Taylor instability cell kinematic diffusivity $\eta_{\mathrm{mid,RTI},k}$ (`etamid_RTI`).
    pub etamid_rti: Vec<f64>,
    /// Rayleigh-Taylor instability Lagrangian diffusion coefficient $\sigma_{\mathrm{RTI},k}$ (`sig_RTI`).
    pub sig_rti: Vec<f64>,
    /// Rayleigh-Taylor instability cell-mid Lagrangian diffusion coefficient (`sigmid_RTI`).
    pub sigmid_rti: Vec<f64>,
    /// Rayleigh-Taylor baroclinic instability product $(\partial P/\partial r)(\partial\rho/\partial r)$ (`dPdr_dRhodr_info`).
    pub dpdr_drhodr_info: Vec<f64>,
    /// Rayleigh-Taylor instability growth source term (`source_plus_alpha_RTI`).
    pub source_plus_alpha_rti: Vec<f64>,
    /// Rayleigh-Taylor instability damping sink term (`source_minus_alpha_RTI`).
    pub source_minus_alpha_rti: Vec<f64>,
    /// Rayleigh-Taylor instability amplitude $\alpha_{\mathrm{RTI},k}$ (`alpha_RTI`).
    pub alpha_rti: Vec<f64>,
    /// Start-of-step Rayleigh-Taylor amplitude $\alpha_{\mathrm{RTI,start},k}$ (`alpha_RTI_start`).
    pub alpha_rti_start: Vec<f64>,
    /// Previous-step Rayleigh-Taylor amplitude $\alpha_{\mathrm{RTI,old},k}$ for step retry.
    pub alpha_rti_old: Vec<f64>,
    /// HLLC Riemann contact velocity $u_{\mathrm{face},k}$ with 3-cell AD partials (`u_face_ad`).
    pub u_face_ad: Vec<StarAdOrder1>,
    /// HLLC Riemann contact pressure $P_{\mathrm{face},k}$ with 3-cell AD partials (`P_face_ad`).
    pub p_face_ad: Vec<StarAdOrder1>,
    /// Start-of-step HLLC contact velocity $u_{\mathrm{face,start},k}$ (`u_face_start`).
    pub u_face_start: Vec<f64>,
    /// Start-of-step HLLC contact pressure $P_{\mathrm{face,start},k}$ (`P_face_start`, `-1` sentinel).
    pub p_face_start: Vec<f64>,
    /// Rotational velocity to critical velocity ratio $\omega_k / \omega_{\mathrm{crit},k}$ (`w_div_wc`).
    pub w_div_wc: Vec<f64>,
    /// Rotational pressure distortion correction factor $f_{P,\mathrm{rot},k}$ (`fp_rot`).
    pub fp_rot: Vec<f64>,
    /// Logarithmic radius derivative $\partial f_{P,\mathrm{rot},k} / \partial \ln r_k$ (`d_fp_rot_dlnR`).
    pub d_fp_rot_dlnr: Vec<f64>,
    /// Rotational temperature distortion correction factor $f_{T,\mathrm{rot},k}$ (`ft_rot`).
    pub ft_rot: Vec<f64>,
    /// Logarithmic radius derivative $\partial f_{T,\mathrm{rot},k} / \partial \ln r_k$ (`d_ft_rot_dlnR`).
    pub d_ft_rot_dlnr: Vec<f64>,
    /// Face specific moment of inertia $i_{\mathrm{rot},k}$ in $\mathrm{cm^2}$ (`i_rot`).
    pub i_rot: Vec<f64>,
    /// Logarithmic radius derivative $\partial i_{\mathrm{rot},k} / \partial \ln r_k$ (`d_i_rot_dlnR`).
    pub d_i_rot_dlnr: Vec<f64>,
    /// Optional binary Roche tidal distortion hook (`use_tidal_deformation`).
    #[serde(skip)]
    pub tidal_deformation: Option<TidalDeformationHandle>,
    /// Brunt-Väisälä composition buoyancy term $N_{\mu,k}^2$ in $\mathrm{s^{-2}}$ (`brunt_N2_composition_term`).
    pub brunt_n2_composition_term: Vec<f64>,
    /// Binomial-smoothed Ledoux composition buoyancy term $B_{\mathrm{smooth},k}$ (`smoothed_brunt_B`).
    pub smoothed_brunt_b: Vec<f64>,
    /// Baryonic mass correction factor per cell (`mass_correction`).
    pub mass_correction: Vec<f64>,
    /// Extra face gravitational acceleration $g_{\mathrm{extra},k}$ (`extra_grav`).
    pub extra_grav: Vec<StarAdOrder1>,
    /// Turbulent viscosity face acceleration $U_{q,k}$ (`Uq_ad`).
    pub uq_ad: Vec<StarAdOrder1>,
    /// EOS specific internal energy composition partials $\partial\ln e_k/\partial X_{j,k}$
    /// (`dlnE_dxa_for_partials`, `len = nz * species`).
    pub dln_e_dxa_for_partials: Vec<f64>,
    /// EOS total pressure composition partials $\partial\ln P_{\mathrm{eos},k}/\partial X_{j,k}$
    /// (`dlnPeos_dxa_for_partials`, `len = nz * species`).
    pub dln_peos_dxa_for_partials: Vec<f64>,
    /// Radial Stellar Pulsation (`RSP` / `RSP2`) zone arrays and cycle state.
    pub rsp: crate::rsp_def::RspState,
    /// Previous-step rotational angular velocity $\Omega_{\mathrm{old},k}$ (`omega_old`).
    pub omega_old: Vec<f64>,
    /// Previous-step specific angular momentum $j_{\mathrm{rot,old},k}$ (`j_rot_old`).
    pub j_rot_old: Vec<f64>,
    /// Previous-step mass-change heating rate $\epsilon_{\dot{M},\mathrm{old},k}$ for step retry.
    pub eps_mdot_old: Vec<f64>,
    /// Solver `set_mlt_vars` evaluation counter within the active `do_struct_burn_mix` solve.
    pub solver_mlt_eval_count: usize,
    /// Per-face sticky radiative latch (`fixed_gradr_for_rest_of_solver_iters` in `turb_info.f90:56-61`).
    pub fixed_gradr_sticky: Vec<bool>,
    /// Auxiliary mixing, MLT, convective-boundary, and Brunt-smoothing state.
    pub mixing: crate::mixing_state::MixingAuxState,
    /// Per-step scalar history, solver/substep counters, and retry-hold state.
    pub step_hist: StepHistoryScalars,
    /// White-dwarf solid-core boundary mass in grams (`crystal_core_boundary_mass`
    /// in `star/private/phase_separation.f90:70, 121` and `mesh_plan.f90:965`).
    pub crystal_core_boundary_mass: f64,
    /// Cumulative mesh adjustment invocations (`mesh_call_number`).
    pub mesh_call_number: usize,
    /// Cumulative `remesh_split_merge` split count (`num_hydro_splits`).
    pub num_hydro_splits: usize,
    /// Cumulative `remesh_split_merge` merge count (`num_hydro_merges`).
    pub num_hydro_merges: usize,
    /// Last `do_mesh_adjust` relative internal-energy conservation residual (`mesh_adjust_IE_conservation`).
    pub mesh_adjust_ie_conservation: f64,
    /// Last `do_mesh_adjust` relative potential-energy conservation residual (`mesh_adjust_PE_conservation`).
    pub mesh_adjust_pe_conservation: f64,
    /// Last `do_mesh_adjust` relative kinetic-energy conservation residual (`mesh_adjust_KE_conservation`).
    pub mesh_adjust_ke_conservation: f64,
    /// Last `do_mesh_adjust` relative turbulent-energy conservation residual (`mesh_adjust_Eturb_conservation`).
    pub mesh_adjust_eturb_conservation: f64,
    /// Optional custom mesh spacing functions callback (`use_other_mesh_functions`).
    #[serde(skip)]
    pub other_mesh_functions:
        Option<crate::mesh_functions::OtherMeshFunctionsHandle>,
    /// Optional custom local mesh multiplier callback (`use_other_mesh_delta_coeff_factor`).
    #[serde(skip)]
    pub other_mesh_delta_coeff_factor:
        Option<crate::adjust_mesh_support::OtherMeshDeltaCoeffFactorHandle>,
    /// Ephemeral planned mesh cache bridging `plan_mesh` and `remap_onto_dq`.
    #[serde(skip)]
    pub(crate) planned_mesh: crate::mesh_adjust::PlannedMeshSlot,
    /// Per-model 2-slot microphysics cache (never duplicated on clone).
    #[serde(skip)]
    pub micro_cache: crate::micro_cache::MicrophysicsCache,
}

impl HydroAuxState {
    /// Allocates a default-initialized [`HydroAuxState`] for `nz` zones and
    /// `species` nuclear isotopes.
    #[must_use]
    pub fn new(nz: usize, species: usize) -> Self {
        Self {
            doing_relax: false,
            doing_struct_burn_mix: false,
            total_energy: 0.0,
            total_internal_energy_start: 0.0,
            cumulative_energy_error: 0.0,
            non_epsnuc_energy_change_from_split_burn: 0.0,
            cp_start: vec![0.0; nz],
            grada_start: vec![0.0; nz],
            chi_t_start: vec![0.0; nz],
            chi_rho_start: vec![0.0; nz],
            eos_frac_pc: vec![0.0; nz],
            kap_frac: vec![[0.0; 4]; nz],
            nonnucneu_channels: vec![[0.0; 5]; nz],
            latent_ddlnrho: vec![0.0; nz],
            latent_ddlnt: vec![0.0; nz],
            d_eps_grav_dx: vec![0.0; nz * species],
            burn_avg_epsnuc: vec![0.0; nz],
            eps_nuc_categories: vec![[0.0; flexmesa_chem::NUM_CATEGORIES]; nz],
            extra_heat: vec![0.0; nz],
            d_extra_heat_dlnd: vec![0.0; nz],
            d_extra_heat_dlnt: vec![0.0; nz],
            eps_wd_sedimentation: vec![0.0; nz],
            eps_diffusion: vec![0.0; nz],
            eps_pre_mix: vec![0.0; nz],
            eps_phase_separation: vec![0.0; nz],
            irradiation_heat: vec![0.0; nz],
            eq_ad: vec![StarAdOrder1::ZERO; nz],
            etrb: vec![0.0; nz],
            etrb_start: vec![0.0; nz],
            et_rsp_old: vec![0.0; nz],
            hp_face_old: vec![0.0; nz],
            csound_start: vec![0.0; nz],
            mlt_vc_old: vec![0.0; nz],
            have_mlt_vc: false,
            okay_to_set_mlt_vc: true,
            k_const_mass: 0,
            eta_rti: vec![0.0; nz],
            etamid_rti: vec![0.0; nz],
            sig_rti: vec![0.0; nz],
            sigmid_rti: vec![0.0; nz],
            dpdr_drhodr_info: vec![0.0; nz],
            source_plus_alpha_rti: vec![0.0; nz],
            source_minus_alpha_rti: vec![0.0; nz],
            alpha_rti: vec![0.0; nz],
            alpha_rti_start: vec![0.0; nz],
            alpha_rti_old: vec![0.0; nz],
            u_face_ad: vec![StarAdOrder1::ZERO; nz],
            p_face_ad: vec![StarAdOrder1::ZERO; nz],
            u_face_start: vec![0.0; nz],
            p_face_start: vec![-1.0; nz],
            w_div_wc: vec![0.0; nz],
            fp_rot: vec![1.0; nz],
            d_fp_rot_dlnr: vec![0.0; nz],
            ft_rot: vec![1.0; nz],
            d_ft_rot_dlnr: vec![0.0; nz],
            i_rot: vec![0.0; nz],
            d_i_rot_dlnr: vec![0.0; nz],
            tidal_deformation: None,
            brunt_n2_composition_term: vec![0.0; nz],
            smoothed_brunt_b: vec![0.0; nz],
            mass_correction: vec![1.0; nz],
            extra_grav: vec![StarAdOrder1::ZERO; nz],
            uq_ad: vec![StarAdOrder1::ZERO; nz],
            dln_e_dxa_for_partials: vec![0.0; nz * species],
            dln_peos_dxa_for_partials: vec![0.0; nz * species],
            rsp: crate::rsp_def::RspState::new(nz),
            omega_old: vec![0.0; nz],
            j_rot_old: vec![0.0; nz],
            eps_mdot_old: vec![0.0; nz],
            solver_mlt_eval_count: 0,
            fixed_gradr_sticky: vec![false; nz],
            mixing: crate::mixing_state::MixingAuxState::new(nz),
            step_hist: StepHistoryScalars::default(),
            crystal_core_boundary_mass: 0.0,
            mesh_call_number: 0,
            num_hydro_splits: 0,
            num_hydro_merges: 0,
            mesh_adjust_ie_conservation: 0.0,
            mesh_adjust_pe_conservation: 0.0,
            mesh_adjust_ke_conservation: 0.0,
            mesh_adjust_eturb_conservation: 0.0,
            other_mesh_functions: None,
            other_mesh_delta_coeff_factor: None,
            planned_mesh: crate::mesh_adjust::PlannedMeshSlot::default(),
            micro_cache: crate::micro_cache::MicrophysicsCache::default(),
        }
    }

    /// Resizes all zone- and species-indexed arrays to `(nz, species)`.
    pub fn resize(&mut self, nz: usize, species: usize) {
        for v in [
            &mut self.cp_start,
            &mut self.grada_start,
            &mut self.chi_t_start,
            &mut self.chi_rho_start,
            &mut self.eos_frac_pc,
            &mut self.latent_ddlnrho,
            &mut self.latent_ddlnt,
            &mut self.burn_avg_epsnuc,
            &mut self.extra_heat,
            &mut self.d_extra_heat_dlnd,
            &mut self.d_extra_heat_dlnt,
            &mut self.eps_wd_sedimentation,
            &mut self.eps_diffusion,
            &mut self.eps_pre_mix,
            &mut self.eps_phase_separation,
            &mut self.irradiation_heat,
            &mut self.etrb,
            &mut self.etrb_start,
            &mut self.et_rsp_old,
            &mut self.hp_face_old,
            &mut self.csound_start,
            &mut self.mlt_vc_old,
            &mut self.eta_rti,
            &mut self.etamid_rti,
            &mut self.sig_rti,
            &mut self.sigmid_rti,
            &mut self.dpdr_drhodr_info,
            &mut self.source_plus_alpha_rti,
            &mut self.source_minus_alpha_rti,
            &mut self.alpha_rti,
            &mut self.alpha_rti_start,
            &mut self.alpha_rti_old,
            &mut self.u_face_start,
            &mut self.w_div_wc,
            &mut self.d_fp_rot_dlnr,
            &mut self.d_ft_rot_dlnr,
            &mut self.i_rot,
            &mut self.d_i_rot_dlnr,
            &mut self.brunt_n2_composition_term,
            &mut self.smoothed_brunt_b,
            &mut self.omega_old,
            &mut self.j_rot_old,
            &mut self.eps_mdot_old,
        ] {
            v.resize(nz, 0.0);
        }
        self.p_face_start.resize(nz, -1.0);
        for v in [
            &mut self.fp_rot,
            &mut self.ft_rot,
            &mut self.mass_correction,
        ] {
            v.resize(nz, 1.0);
        }
        for ad in [
            &mut self.eq_ad,
            &mut self.extra_grav,
            &mut self.uq_ad,
            &mut self.u_face_ad,
            &mut self.p_face_ad,
        ] {
            ad.resize(nz, StarAdOrder1::ZERO);
        }
        let n_sp = nz * species;
        self.kap_frac.resize(nz, [0.0; 4]);
        self.nonnucneu_channels.resize(nz, [0.0; 5]);
        self.eps_nuc_categories
            .resize(nz, [0.0; flexmesa_chem::NUM_CATEGORIES]);
        self.d_eps_grav_dx.resize(n_sp, 0.0);
        self.dln_e_dxa_for_partials.resize(n_sp, 0.0);
        self.dln_peos_dxa_for_partials.resize(n_sp, 0.0);
        self.fixed_gradr_sticky.resize(nz, false);
        self.mixing.resize(nz);
        self.rsp.resize(nz);
    }

    /// Snapshots start-of-step thermodynamic and turbulent velocity arrays.
    pub fn save_start(
        &mut self,
        cp: &[f64],
        grada: &[f64],
        chi_t: &[f64],
        chi_rho: &[f64],
        csound: &[f64],
        _mlt_vc: &[f64],
    ) {
        self.cp_start.clone_from_slice(cp);
        self.grada_start.clone_from_slice(grada);
        self.chi_t_start.clone_from_slice(chi_t);
        self.chi_rho_start.clone_from_slice(chi_rho);
        self.csound_start.clone_from_slice(csound);
        self.etrb_start.clone_from(&self.etrb);
        self.alpha_rti_start.clone_from(&self.alpha_rti);
    }
}

// Retains `Result` signature for `evolve_support.rs` caller compatibility until `#152` lands.
#[allow(clippy::unnecessary_wraps)]
pub(crate) fn smooth_post_remesh_hydro_and_rotation(
    _model: &mut crate::model::StarModel,
    _pre: &crate::primary_state::PrimaryStateBuffer,
) -> Result<(), crate::controls::StarError> {
    Ok(())
}
