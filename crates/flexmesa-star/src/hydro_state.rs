// SPDX-License-Identifier: LGPL-3.0-or-later
//! Auxiliary start-of-step thermodynamic arrays, heating channels, turbulent
//! work/viscosity variables, rotational distortion factors, Rayleigh-Taylor
//! instability (RTI) profiles, and EOS composition partial derivatives on
//! [`crate::StarModel`] (`star_data_step_work.inc` and `star/private/alloc.f90`).

use crate::star_ad::{StarAdExt, StarAdOrder1};

/// Per-step scalar history, solver/substep counters, and retry-hold state
/// consumed by [`crate::timestep_limits::timestep_controller`] and
/// [`crate::evolve`] (`star_data_step_work.inc`).
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
    pub tidal_deformation: Option<TidalDeformationHandle>,
    /// Brunt-Väisälä composition buoyancy term $N_{\mu,k}^2$ in $\mathrm{s^{-2}}$ (`brunt_N2_composition_term`).
    pub brunt_n2_composition_term: Vec<f64>,
    /// Triangular-smoothed Ledoux composition buoyancy term $B_{\mathrm{smooth},k}$ (`smoothed_brunt_B`).
    pub smoothed_brunt_b: Vec<f64>,
    /// Bit-level hash of `model.brunt_b` when `smoothed_brunt_b` was last refreshed.
    pub smoothed_brunt_b_hash: u64,
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
    /// Whether any face has completed a convective/radiative round-trip flip during the active Henyey solve.
    pub solver_had_mlt_flip: bool,
    /// Per-face sticky radiative/sub-Ledoux latch (`fixed_gradr_for_rest_of_solver_iters` in `turb_info.f90:57-62`).
    pub fixed_gradr_sticky: Vec<bool>,
    /// Per-face previous convective indicator (`1` if `ConvectiveMixing`, `0` otherwise) during a Henyey solve.
    pub mlt_prev_conv: Vec<u8>,
    /// Per-face convective/non-convective transition count within the active Henyey solve.
    pub mlt_flip_count: Vec<u8>,
    /// Per-step scalar history, solver/substep counters, and retry-hold state.
    pub step_hist: StepHistoryScalars,
    /// Per-model 2-slot microphysics cache (never duplicated on clone).
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
            smoothed_brunt_b_hash: 0,
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
            solver_had_mlt_flip: false,
            fixed_gradr_sticky: vec![false; nz],
            mlt_prev_conv: vec![0; nz],
            mlt_flip_count: vec![0; nz],
            step_hist: StepHistoryScalars::default(),
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
        self.eps_nuc_categories
            .resize(nz, [0.0; flexmesa_chem::NUM_CATEGORIES]);
        self.d_eps_grav_dx.resize(n_sp, 0.0);
        self.dln_e_dxa_for_partials.resize(n_sp, 0.0);
        self.dln_peos_dxa_for_partials.resize(n_sp, 0.0);
        self.fixed_gradr_sticky.resize(nz, false);
        self.mlt_prev_conv.resize(nz, 0);
        self.mlt_flip_count.resize(nz, 0);
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
        mlt_vc: &[f64],
    ) {
        self.cp_start.clone_from_slice(cp);
        self.grada_start.clone_from_slice(grada);
        self.chi_t_start.clone_from_slice(chi_t);
        self.chi_rho_start.clone_from_slice(chi_rho);
        self.csound_start.clone_from_slice(csound);
        self.etrb_start.clone_from(&self.etrb);
        self.alpha_rti_start.clone_from(&self.alpha_rti);
        self.mlt_vc_old.clone_from_slice(mlt_vc);
    }
}

#[inline]
pub(crate) fn hash_brunt_b(b: &[f64]) -> u64 {
    let mut h = 0xcbf29ce484222325_u64;
    for &x in b {
        h ^= x.to_bits();
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub(crate) fn smooth_triangular_signed(
    src: &[f64],
    half_width: usize,
) -> Vec<f64> {
    let nz = src.len();
    if half_width == 0 || nz < 3 {
        return src.to_vec();
    }
    let mut out = vec![0.0; nz];
    out[0] = src[0];
    let w_i = half_width as isize;
    for k in 1..nz {
        let (mut sum, mut wsum) = (0.0, 0.0);
        for dj in -w_i..=w_i {
            let j = k as isize + dj;
            if j >= 1 && (j as usize) < nz {
                let wt = (half_width + 1 - dj.unsigned_abs()) as f64;
                sum += wt * src[j as usize];
                wsum += wt;
            }
        }
        out[k] = if wsum > 0.0 { sum / wsum } else { src[k] };
    }
    out
}

pub(crate) fn refresh_smoothed_brunt_b(model: &mut crate::model::StarModel) {
    let mw = &model.controls.mix_wind;
    let raw_b = model.brunt_b.clone();
    model.aux.smoothed_brunt_b = smooth_triangular_signed(
        &raw_b,
        mw.num_cells_for_smooth_gradl_composition_term,
    );
    if mw.num_cells_for_smooth_brunt_b > 0 {
        model.brunt_b =
            smooth_triangular_signed(&raw_b, mw.num_cells_for_smooth_brunt_b);
    }
    model.aux.smoothed_brunt_b_hash = hash_brunt_b(&model.brunt_b);
}

#[inline]
pub(crate) fn use_smoothed_brunt_b(model: &crate::model::StarModel) -> bool {
    model.aux.smoothed_brunt_b.len() == model.nz
        && model.aux.smoothed_brunt_b_hash != 0
        && hash_brunt_b(&model.brunt_b) == model.aux.smoothed_brunt_b_hash
}

fn interp_hermite_p1(
    x_p1: &[f64],
    y_p1: &[f64],
    dy_p1: &[f64],
    kk: usize,
    x: f64,
) -> f64 {
    let (x0, x1) = (x_p1[kk], x_p1[kk + 1]);
    if (x - x0).abs() <= 1.0e-15 {
        return y_p1[kk];
    }
    if (x - x1).abs() <= 1.0e-15 {
        return y_p1[kk + 1];
    }
    let h = (x1 - x0).max(1.0e-30);
    let t = ((x - x0) / h).clamp(0.0, 1.0);
    let (t2, t3) = (t * t, t * t * t);
    (2.0 * t3 - 3.0 * t2 + 1.0) * y_p1[kk]
        + (t3 - 2.0 * t2 + t) * h * dy_p1[kk]
        + (-2.0 * t3 + 3.0 * t2) * y_p1[kk + 1]
        + (t3 - t2) * h * dy_p1[kk + 1]
}

pub(crate) fn smooth_post_remesh_hydro_and_rotation(
    model: &mut crate::model::StarModel,
    pre: &crate::primary_state::PrimaryStateBuffer,
) -> Result<(), crate::controls::StarError> {
    let (nz_new, nz_old) = (model.nz, pre.nz);
    if nz_new < 2 || nz_old < 2 {
        return Ok(());
    }
    let mut xq_p1 = Vec::with_capacity(nz_old + 1);
    xq_p1.extend_from_slice(&pre.xq[..nz_old]);
    xq_p1.push(1.0);
    let mut v_p1 = vec![0.0; nz_old + 1];
    for kk in (0..nz_old).rev() {
        v_p1[kk] = v_p1[kk + 1] + pre.dq[kk] * flexmesa_math::exp(-pre.lnd[kk]);
    }
    let mut dv_p1 = crate::pulse_bvp::steffen_slopes(&xq_p1, &v_p1);
    dv_p1[0] = (v_p1[1] - v_p1[0]) / (xq_p1[1] - xq_p1[0]).max(1.0e-30);

    let (mut e_c1, mut lnt_c1) = (vec![0.0; nz_old], vec![0.0; nz_old]);
    for kk in 1..(nz_old - 1) {
        let dm_lo = 0.5 * (pre.dq[kk - 1] + pre.dq[kk]).max(1.0e-99);
        let dm_hi = 0.5 * (pre.dq[kk] + pre.dq[kk + 1]).max(1.0e-99);
        let (se_lo, se_hi) = (
            (pre.energy[kk] - pre.energy[kk - 1]) / dm_lo,
            (pre.energy[kk + 1] - pre.energy[kk]) / dm_hi,
        );
        if se_lo * se_hi > 0.0 {
            let cap = 1.6 * pre.energy[kk].abs() / pre.dq[kk].max(1.0e-99);
            e_c1[kk] = (2.0 * se_lo * se_hi / (se_lo + se_hi)).clamp(-cap, cap);
        }
        let (st_lo, st_hi) = (
            (pre.ln_t[kk] - pre.ln_t[kk - 1]) / dm_lo,
            (pre.ln_t[kk + 1] - pre.ln_t[kk]) / dm_hi,
        );
        if st_lo * st_hi > 0.0 {
            lnt_c1[kk] = 2.0 * st_lo * st_hi / (st_lo + st_hi);
        }
    }

    let (mut exact_flags, mut e_target, mut kk_hint) =
        (vec![false; nz_new], vec![0.0; nz_new], 0_usize);
    for k in 0..nz_new {
        let (a0, dx) = (model.xq[k], model.dq[k]);
        let a1 = if k + 1 < nz_new { model.xq[k + 1] } else { 1.0 };
        while kk_hint + 1 < nz_old && xq_p1[kk_hint + 1] <= a0 {
            kk_hint += 1;
        }
        let exact_kk = (kk_hint.saturating_sub(1)..(kk_hint + 3).min(nz_old))
            .find(|&kk| {
                (dx - pre.dq[kk]).abs() <= 1.0e-11 * dx
                    && (a0 - pre.xq[kk]).abs()
                        <= 1.0e-11 * a0.max(dx).max(1.0e-15)
            });
        if let Some(kk) = exact_kk {
            exact_flags[k] = true;
            model.lnd[k] = pre.lnd[kk];
            model.ln_t[k] = pre.ln_t[kk];
            model.energy[k] = pre.energy[kk];
            model.lum[k] = pre.lum[kk];
            if model.controls.rotation_flag {
                model.omega[k] = pre.omega[kk];
            }
            e_target[k] = pre.energy[kk];
            continue;
        }
        let (xq_lo, xq_hi) = (xq_p1[kk_hint], xq_p1[kk_hint + 1]);
        let tf = if (a0 - xq_lo).abs() <= 1.0e-14 * dx {
            0.0
        } else {
            ((a0 - xq_lo) / (xq_hi - xq_lo).max(1.0e-30)).clamp(0.0, 1.0)
        };
        let l_hi = if kk_hint + 1 < nz_old {
            pre.lum[kk_hint + 1]
        } else {
            pre.l_center
        };
        model.lum[k] = pre.lum[kk_hint] + tf * (l_hi - pre.lum[kk_hint]);
        if model.controls.rotation_flag {
            model.omega[k] = pre.omega[kk_hint]
                + tf * (pre.omega[(kk_hint + 1).min(nz_old - 1)]
                    - pre.omega[kk_hint]);
        }
        let mut kk1 = kk_hint;
        while kk1 + 1 < nz_old && xq_p1[kk1 + 1] < a1 {
            kk1 += 1;
        }
        let v0 = interp_hermite_p1(&xq_p1, &v_p1, &dv_p1, kk_hint, a0);
        let v1 = interp_hermite_p1(&xq_p1, &v_p1, &dv_p1, kk1, a1);
        model.lnd[k] =
            flexmesa_math::log((dx / (v0 - v1).max(1.0e-99)).max(1.0e-30));
        let (mut e_acc, mut lnt_acc, mut w_sum) = (0.0, 0.0, 0.0);
        for kk in kk_hint..=kk1 {
            let (lo, hi) = (a0.max(xq_p1[kk]), a1.min(xq_p1[kk + 1]));
            if hi > lo {
                let (ov, d_xc) =
                    (hi - lo, 0.5 * (lo + hi) - (xq_p1[kk] + 0.5 * pre.dq[kk]));
                e_acc += ov * (pre.energy[kk] + e_c1[kk] * d_xc);
                lnt_acc += ov * (pre.ln_t[kk] + lnt_c1[kk] * d_xc);
                w_sum += ov;
            }
        }
        e_target[k] = if w_sum > 0.0 {
            model.ln_t[k] = (lnt_acc / w_sum).clamp(3.0, 26.0);
            (e_acc / w_sum).max(1.0)
        } else {
            model.energy[k]
        };
    }
    model.reconstruct_ln_r_from_lnd_and_dm();
    model.set_m_grav_and_grav();
    model.pack_xh();
    crate::mesh_remap::set_lnt_for_energy(model, &e_target, &exact_flags)?;
    model.pack_xh();
    crate::rotation::update_rotation_distortion_factors(model);
    if model.controls.rotation_flag && pre.j_rot.len() == nz_old {
        let j_old: f64 =
            pre.dm_bar.iter().zip(&pre.j_rot).map(|(m, j)| m * j).sum();
        for _ in 0..2 {
            let j_new: f64 = model
                .dm_bar
                .iter()
                .zip(&model.j_rot)
                .map(|(m, j)| m * j)
                .sum();
            if j_old.abs() > 0.0 && j_new.abs() > 1.0e-30 && j_old * j_new > 0.0
            {
                let sc = j_old / j_new;
                model.omega[..nz_new].iter_mut().for_each(|w| *w *= sc);
                crate::rotation::update_rotation_distortion_factors(model);
            }
        }
    }
    Ok(())
}
