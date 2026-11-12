// SPDX-License-Identifier: LGPL-3.0-or-later
//! Configuration schemas and namelist population for nonlinear Radial Stellar
//! Pulsation (`RSP1`, `controls.defaults:3063-3333`) and Henyey turbulent-energy
//! pulsation (`RSP2`, `controls_dev.defaults:59-131`).

use crate::controls::{StarControls, StarJobControls};
use crate::inlist::SectionControls;

/// Nonlinear Radial Stellar Pulsation (`RSP1`) controls (`&controls` namelist in
/// `star/defaults/controls.defaults:3063-3333`).
#[derive(Debug, Clone)]
pub struct RspConfig {
    /// Kuhfuss / Smolec & Moskalik (2008) convection multipliers:
    /// $(\alpha, \alpha_c, \alpha_s, \alpha_d, \alpha_p, \alpha_t, \alpha_m, \gamma_r)$.
    pub alfa: f64,
    /// Convective flux multiplier $\alpha_c$ (`1.0`).
    pub alfac: f64,
    /// Turbulent source multiplier $\alpha_s$ (`1.0`).
    pub alfas: f64,
    /// Turbulent dissipation multiplier $\alpha_d$ (`1.0`).
    pub alfad: f64,
    /// Turbulent pressure multiplier $\alpha_p$ (`0.0`).
    pub alfap: f64,
    /// Turbulent kinetic-flux multiplier $\alpha_t$ (`0.0`).
    pub alfat: f64,
    /// Eddy-viscosity multiplier $\alpha_m$ (`0.25`).
    pub alfam: f64,
    /// Radiative turbulent damping multiplier $\gamma_r$ (`0.0`).
    pub gammar: f64,
    /// Richtmyer-Morton artificial viscosity $C_q$ (`4.0`), Mach threshold $z_{\mathrm{sh}}$ (`0.1`), and flux limiter (`1e2`).
    pub cq: f64,
    /// Artificial viscosity threshold $z_{\mathrm{sh}}$ (`0.1`).
    pub zsh: f64,
    /// Flux-limiter scale (`1e2`).
    pub efl0: f64,
    /// Time-centering weights $(\theta, \theta_t, \theta_e, \theta_q, \theta_u, w_r, w_c, w_t, \gamma)$.
    pub theta: f64,
    /// Turbulent pressure time weight $\theta_t$ (`0.5`).
    pub thetat: f64,
    /// Radiation energy time weight $\theta_e$ (`0.5`).
    pub thetae: f64,
    /// Artificial viscosity time weight $\theta_q$ (`1.0`).
    pub thetaq: f64,
    /// Eddy-viscosity time weight $\theta_u$ (`1.0`).
    pub thetau: f64,
    /// Radiative luminosity time weight $w_r$ (`0.6667`).
    pub wtr: f64,
    /// Convective luminosity time weight $w_c$ (`0.6667`).
    pub wtc: f64,
    /// Turbulent luminosity time weight $w_t$ (`0.6667`).
    pub wtt: f64,
    /// Turbulent source/sink time weight $\gamma$ (`1.0`).
    pub gam: f64,
    /// Static envelope zone count `RSP_nz` (`150`) and outer zone count `RSP_nz_outer` (`40`).
    pub nz: usize,
    /// Outer uniform-mass zone count (`40`).
    pub nz_outer: usize,
    /// Anchor temperature $T_{\mathrm{anchor}}$ (`11e3` K) and inner temperature $T_{\mathrm{inner}}$ (`2e6` K).
    pub t_anchor: f64,
    /// Inner boundary temperature $T_{\mathrm{inner}}$ (`2e6` K).
    pub t_inner: f64,
    /// Surface cell mass factor $\Delta m_N / \Delta m_{\mathrm{outer}}$ (`2.0`).
    pub dq_1_factor: f64,
    /// Maximum outer/inner shooting tries (`100`) and anchor/inner tolerances (`1e-8`).
    pub max_outer_dm_tries: usize,
    /// Maximum inner geometric mass-scale tries (`100`).
    pub max_inner_scale_tries: usize,
    /// Relative tolerance on $T_{\mathrm{anchor}}$ (`1e-8`).
    pub t_anchor_tolerance: f64,
    /// Relative tolerance on $T_{\mathrm{inner}}$ (`1e-8`).
    pub t_inner_tolerance: f64,
    /// `RELAX_ENV` controls (`relax_initial_model`, homotopy order, max tries `1000`, `dm` tol `1e-6`).
    pub relax_initial_model: bool,
    /// Ramp $\alpha_p$ before $\alpha_t$ (`true`).
    pub relax_alfap_before_alfat: bool,
    /// Maximum `RELAX_ENV` tries (`1000`).
    pub relax_max_tries: usize,
    /// Outer secant tolerance on $\Delta m_N$ (`1e-6`).
    pub relax_dm_tolerance: f64,
    /// Update inner geometric mass distribution during `RELAX_ENV` (`true`).
    pub relax_adjust_inner_mass_distribution: bool,
    /// Initial `LINA` kick amplitude ($\mathrm{km\,s^{-1}}$) and 1O/2O mode fractions.
    pub kick_vsurf_km_per_sec: f64,
    /// 1O mode fraction (`0.0`).
    pub fraction_1st_overtone: f64,
    /// 2O mode fraction (`0.0`).
    pub fraction_2nd_overtone: f64,
    /// Linear and random velocity perturbations (`RSP_Avel`, `RSP_Arnd` in $\mathrm{km\,s^{-1}}$).
    pub avel: f64,
    /// Random velocity amplitude (`0.0`).
    pub arnd: f64,
    /// Number of `LINA` modes (`3`), mode index for `PERIODLIN` (`1`), and fallback `PERIODLIN` (`34560` s).
    pub nmodes: usize,
    /// Mode index (`0`=F, `1`=1O, `2`=2O) setting `PERIODLIN` (`1`).
    pub mode_for_setting_periodlin: usize,
    /// Fallback linear period in $\mathrm{s}$ (`34560.0`).
    pub default_periodlin: f64,
    /// Cycle acceptance thresholds (`min_max_R`, `min_deltaR` in $R_\odot$, `min_PERIOD_div_PERIODLIN`).
    pub min_max_r_for_periods: f64,
    /// Minimum $\Delta R / R_\odot$ to count a period (`-1.0`).
    pub min_deltar_for_periods: f64,
    /// Minimum $P / P_{\mathrm{lin}}$ to count a period (`0.5`).
    pub min_period_div_periodlin: f64,
    /// Exponential smoothing fraction (`0.1`) and limit (`-1.0`) for $|\mathrm{GREKM}|$.
    pub grekm_avg_abs_frac_new: f64,
    /// Stopping limit on $\langle|\mathrm{GREKM}|\rangle$ (`-1.0`).
    pub grekm_avg_abs_limit: f64,
    /// Initial and Courant/diffusion timestep controls.
    pub initial_dt_factor: f64,
    /// Mach threshold $|v|/c_s$ triggering Courant limit (`0.8`).
    pub v_div_cs_threshold_for_dt_limit: f64,
    /// Courant factor $\Delta t_{\max} / \min(\Delta r / c_s)$ (`2.0`).
    pub max_dt_times_min_dr_div_cs: f64,
    /// Radiative diffusion timestep factor (`-1.0`).
    pub max_dt_times_min_rad_diff_time: f64,
    /// Hard cap on $\Delta t$ in $\mathrm{s}$ (`-1.0`).
    pub max_dt: f64,
    /// Surface pressure boundary-condition options and optical depths.
    pub use_prad_for_psurf: bool,
    /// Solve Eddington grey atmosphere with iterated $\kappa$ for $P_{\mathrm{surf}}$ (`false`).
    pub use_atm_grey_with_kap_for_psurf: bool,
    /// Optical depth for `use_atm_grey_with_kap_for_psurf` (`3e-3`).
    pub tau_surf_for_atm_grey_with_kap: f64,
    /// Keep $P_{\mathrm{surf}}$ fixed at initial value (`true`).
    pub fixed_psurf: bool,
    /// Explicit $P_{\mathrm{surf}}$ in $\mathrm{dyn\,cm^{-2}}$ (`0.0`).
    pub psurf: f64,
    /// Multiplier setting $P_{\mathrm{surf}} = f \cdot P_1(0)$ (`-1.0`).
    pub set_psurf_to_multiple_of_initial_p1: f64,
    /// Surface optical depth $\tau_{\mathrm{surf}}$ (`0.001`).
    pub surface_tau: f64,
    /// `HYD` Newton solver tolerances, iteration limits, and turbulent zone cutoffs.
    pub tol_max_corr: f64,
    /// Maximum relative residual tolerance (`1e-6`).
    pub tol_max_resid: f64,
    /// Maximum Newton iterations per try (`100`).
    pub max_iters_per_try: usize,
    /// Maximum retries per step (`50`).
    pub max_retries_per_step: usize,
    /// Divisor setting `IBOTOM = nz / nz_div_ibotom` (`30.0`).
    pub nz_div_ibotom: f64,
    /// Optical depth threshold for damping $L_t$ near surface (`2e2`).
    pub min_tau_for_turbulent_flux: f64,
    /// Analytic opacity test factor (`-1.0` = disabled).
    pub kap_density_factor: f64,
    /// Work-integral and 3D map output controls.
    pub work_period: isize,
    /// Work-integral output filename (`"work.data"`).
    pub work_filename: String,
    /// Enable 3D map output (`false`).
    pub write_map: bool,
    /// Map column specification filename (`"map_columns.list"`).
    pub map_columns_filename: String,
    /// Map output filename (`"map.data"`).
    pub map_filename: String,
    /// First period written to map (`-1`).
    pub map_first_period: isize,
    /// Last period written to map (`-1`).
    pub map_last_period: isize,
    /// Zone stride for map output (`2`).
    pub map_zone_interval: usize,
    /// Map history filename (`"map_history.data"`).
    pub map_history_filename: String,
    /// Hook and solver mode flags (`use_other_RSP_*`, `do_check_omega`, `hydro_only`).
    pub use_other_rsp_linear_analysis: bool,
    /// Custom `RSP` model builder hook (`false`).
    pub use_other_rsp_build_model: bool,
    /// Alternative startup scheme (`false`).
    pub use_rsp_new_start_scheme: bool,
    /// Check/refine `LINA` frequencies (`true`).
    pub do_check_omega: bool,
    /// Pure-hydrodynamics test flag (`false`).
    pub hydro_only: bool,
    /// Dump missing trace solar metals into heaviest network species (`false`).
    ///
    /// Defaults to `false` to match upstream MESA (`star_data_def.f90:93` and
    /// `rsp_eval_eos_and_kap.f90:91`), where `star_create_RSP_model` leaves
    /// `s% initial_dump_missing_heaviest` at `.false.` so `adjust_z_fractions`
    /// renormalizes present network metals proportionally (`jskip = 0`).
    pub initial_dump_missing_heaviest: bool,
    /// Temperature-ramped `opacity_factor` bounds and grey $T(\tau)$ iteration controls.
    pub min_logt_for_opacity_factor_off: f64,
    /// Lower $\log_{10} T$ bound for full `opacity_factor` (`-1.0`).
    pub min_logt_for_opacity_factor_on: f64,
    /// Upper $\log_{10} T$ bound for full `opacity_factor` (`-1.0`).
    pub max_logt_for_opacity_factor_on: f64,
    /// Upper $\log_{10} T$ bound where `opacity_factor` turns off (`-1.0`).
    pub max_logt_for_opacity_factor_off: f64,
    /// Relative tolerance for grey $T(\tau)$ iteration (`1e-4`).
    pub atm_t_tau_errtol: f64,
    /// Maximum iterations for grey $T(\tau)$ opacity solve (`50`).
    pub atm_t_tau_max_iters: usize,
    /// Maximum per-step timestep growth factor (`1.2` in `controls.defaults`).
    pub max_timestep_factor: f64,
    /// Hard upper bound on timestep in $\mathrm{s}$ (`0.0` disables).
    pub max_timestep: f64,
    /// Forced fixed timestep in $\mathrm{s}$ (`0.0` disables).
    pub force_timestep: f64,
    /// Forced minimum timestep floor in $\mathrm{s}$ (`0.0` disables).
    pub force_timestep_min: f64,
}

impl Default for RspConfig {
    fn default() -> Self {
        Self {
            alfa: 1.5,
            alfac: 1.0,
            alfas: 1.0,
            alfad: 1.0,
            alfap: 0.0,
            alfat: 0.0,
            alfam: 0.25,
            gammar: 0.0,
            cq: 4.0,
            zsh: 0.1,
            efl0: 1.0e2,
            theta: 0.5,
            thetat: 0.5,
            thetae: 0.5,
            thetaq: 1.0,
            thetau: 1.0,
            wtr: 0.6667,
            wtc: 0.6667,
            wtt: 0.6667,
            gam: 1.0,
            nz: 150,
            nz_outer: 40,
            t_anchor: 11.0e3,
            t_inner: 2.0e6,
            dq_1_factor: 2.0,
            max_outer_dm_tries: 100,
            max_inner_scale_tries: 100,
            t_anchor_tolerance: 1.0e-8,
            t_inner_tolerance: 1.0e-8,
            relax_initial_model: true,
            relax_alfap_before_alfat: true,
            relax_max_tries: 1000,
            relax_dm_tolerance: 1.0e-6,
            relax_adjust_inner_mass_distribution: true,
            kick_vsurf_km_per_sec: 0.1,
            fraction_1st_overtone: 0.0,
            fraction_2nd_overtone: 0.0,
            avel: 0.0,
            arnd: 0.0,
            nmodes: 3,
            mode_for_setting_periodlin: 1,
            default_periodlin: 34_560.0,
            min_max_r_for_periods: -1.0,
            min_deltar_for_periods: -1.0,
            min_period_div_periodlin: 0.5,
            grekm_avg_abs_frac_new: 0.1,
            grekm_avg_abs_limit: -1.0,
            initial_dt_factor: 1.0e-2,
            v_div_cs_threshold_for_dt_limit: 0.8,
            max_dt_times_min_dr_div_cs: 2.0,
            max_dt_times_min_rad_diff_time: -1.0,
            max_dt: -1.0,
            use_prad_for_psurf: false,
            use_atm_grey_with_kap_for_psurf: false,
            tau_surf_for_atm_grey_with_kap: 3.0e-3,
            fixed_psurf: true,
            psurf: 0.0,
            set_psurf_to_multiple_of_initial_p1: -1.0,
            surface_tau: 0.001,
            tol_max_corr: 1.0e-8,
            tol_max_resid: 1.0e-6,
            max_iters_per_try: 100,
            max_retries_per_step: 50,
            nz_div_ibotom: 30.0,
            min_tau_for_turbulent_flux: 2.0e2,
            kap_density_factor: -1.0,
            work_period: -1,
            work_filename: String::from("work.data"),
            write_map: false,
            map_columns_filename: String::from("map_columns.list"),
            map_filename: String::from("map.data"),
            map_first_period: -1,
            map_last_period: -1,
            map_zone_interval: 2,
            map_history_filename: String::from("map_history.data"),
            use_other_rsp_linear_analysis: false,
            use_other_rsp_build_model: false,
            use_rsp_new_start_scheme: false,
            do_check_omega: true,
            hydro_only: false,
            initial_dump_missing_heaviest: false,
            min_logt_for_opacity_factor_off: -1.0,
            min_logt_for_opacity_factor_on: -1.0,
            max_logt_for_opacity_factor_on: -1.0,
            max_logt_for_opacity_factor_off: -1.0,
            atm_t_tau_errtol: 1.0e-4,
            atm_t_tau_max_iters: 50,
            max_timestep_factor: 1.2,
            max_timestep: 0.0,
            force_timestep: 0.0,
            force_timestep_min: 0.0,
        }
    }
}

/// Coupled Henyey turbulent-energy pulsation (`RSP2`) controls
/// (`star/defaults/controls_dev.defaults:59-131` and `star_job_dev.defaults:26-30`).
#[derive(Debug, Clone)]
pub struct Rsp2Controls {
    /// Dissipation, pressure, flux, eddy-viscosity, and radiative cooling multipliers.
    pub alfad: f64,
    /// Turbulent pressure multiplier (`0.0`).
    pub alfap: f64,
    /// Turbulent flux multiplier (`0.0`).
    pub alfat: f64,
    /// Eddy-viscosity multiplier (`0.25`).
    pub alfam: f64,
    /// Radiative cooling multiplier (`0.0`).
    pub alfar: f64,
    /// Surface radiative luminosity factor (`0.5`).
    pub lsurf_factor: f64,
    /// Surface luminosity BC, hydrostatic $H_p$, $Y_{\mathrm{face}}$, face interpolation, and Stellingwerf $L_r$ flags.
    pub use_l_eqn_at_surface: bool,
    /// Assume HSE in $H_p$ equation (`true`).
    pub assume_hse: bool,
    /// Use `RSP` stencil for $Y_{\mathrm{face}}$ (`true`).
    pub use_rsp_eqn_for_y_face: bool,
    /// Mass-weighted face interpolation (`false`).
    pub use_mass_interp_face_values: bool,
    /// Stellingwerf radiative luminosity form (`true`).
    pub use_stellingwerf_lr: bool,
    /// Outermost and innermost nonturbulent cell counts (`0`).
    pub num_outermost_cells_forced_nonturbulent: usize,
    /// Innermost nonturbulent cell count (`0`).
    pub num_innermost_cells_forced_nonturbulent: usize,
    /// Floor for negative $w$, damping threshold $w_{\min}$, and buoyant source seed ($\mathrm{cm\,s^{-1}}$).
    pub w_fix_if_neg: f64,
    /// Minimum $w$ for damping (`1e2`).
    pub w_min_for_damping: f64,
    /// Source seed velocity (`0.0`).
    pub source_seed: f64,
    /// Remesh-on-load controls (`RSP2_remesh_when_load`, `nz`, `nz_outer`, `T_anchor`, `dq_1_factor`, `nz_div_IBOTOM`).
    pub remesh_when_load: bool,
    /// Target `RSP2` zone count (`150`).
    pub nz: usize,
    /// Target outer zone count (`40`).
    pub nz_outer: usize,
    /// Anchor temperature in $\mathrm{K}$ (`11e3`).
    pub t_anchor: f64,
    /// Surface cell mass factor (`2.0`).
    pub dq_1_factor: f64,
    /// Divisor for `IBOTOM` (`30.0`).
    pub nz_div_ibotom: f64,
    /// Target steps per cycle (`600`), max periods (`-1`), and work dump controls.
    pub target_steps_per_cycle: usize,
    /// Maximum pulsation periods (`-1`).
    pub max_num_periods: isize,
    /// Work-integral period (`-1`).
    pub work_period: isize,
    /// Work-integral filename (`"work.data"`).
    pub work_filename: String,
    /// `&star_job` flag overrides (`change_RSP_flag`, `change_RSP2_flag`, `create_RSP2_model`).
    pub change_rsp_flag: bool,
    /// Target `RSP_flag` (`false`).
    pub new_rsp_flag: bool,
    /// Change `RSP2_flag` on startup (`false`).
    pub change_rsp2_flag: bool,
    /// Model number at which `RSP2_flag` toggles (`-111`).
    pub change_rsp2_flag_at_model_number: isize,
    /// Target `RSP2_flag` (`false`).
    pub new_rsp2_flag: bool,
    /// Build `RSP2` model on startup (`false`).
    pub create_rsp2_model: bool,
    /// Apply quintic enthalpy flux limiter to `PII` (`use_TDC_enthalpy_flux_limiter`, `false`).
    pub use_tdc_enthalpy_flux_limiter: bool,
}

impl Default for Rsp2Controls {
    fn default() -> Self {
        Self {
            alfad: 1.0,
            alfap: 0.0,
            alfat: 0.0,
            alfam: 0.25,
            alfar: 0.0,
            lsurf_factor: 0.5,
            use_l_eqn_at_surface: true,
            assume_hse: true,
            use_rsp_eqn_for_y_face: true,
            use_mass_interp_face_values: false,
            use_stellingwerf_lr: true,
            num_outermost_cells_forced_nonturbulent: 0,
            num_innermost_cells_forced_nonturbulent: 0,
            w_fix_if_neg: 5.0e-5,
            w_min_for_damping: 1.0e2,
            source_seed: 0.0,
            remesh_when_load: true,
            nz: 150,
            nz_outer: 40,
            t_anchor: 11.0e3,
            dq_1_factor: 2.0,
            nz_div_ibotom: 30.0,
            target_steps_per_cycle: 600,
            max_num_periods: -1,
            work_period: -1,
            work_filename: String::from("work.data"),
            change_rsp_flag: false,
            new_rsp_flag: false,
            change_rsp2_flag: false,
            change_rsp2_flag_at_model_number: -111,
            new_rsp2_flag: false,
            create_rsp2_model: false,
            use_tdc_enthalpy_flux_limiter: false,
        }
    }
}

/// Populates [`RspConfig`] and [`Rsp2Controls`] from `&star_job` and `&controls`.
#[rustfmt::skip]
pub(crate) fn populate_rsp_inlist(
    sj: Option<&SectionControls>,
    ctrl: Option<&SectionControls>,
    job: &mut StarJobControls,
    c: &mut StarControls,
) {
    if let Some(sj) = sj {
        if sj.get_bool("load_saved_model_for_rsp2") == Some(true) {
            job.load_saved_model = true;
            c.rsp2.change_rsp2_flag = true;
            c.rsp2.new_rsp2_flag = true;
            c.rsp2_flag = true;
        }
        if let Some(b) = sj.get_bool("change_rsp_flag").or_else(|| sj.get_bool("change_initial_rsp_flag")) {
            c.rsp2.change_rsp_flag = b;
        }
        if let Some(b) = sj.get_bool("new_rsp_flag") {
            c.rsp2.new_rsp_flag = b;
            if c.rsp2.change_rsp_flag { c.rsp_flag = b; }
        }
        if let Some(b) = sj.get_bool("change_rsp2_flag").or_else(|| sj.get_bool("change_initial_rsp2_flag")) {
            c.rsp2.change_rsp2_flag = b;
        }
        if let Some(v) = sj.get_i64("change_rsp2_flag_at_model_number") {
            c.rsp2.change_rsp2_flag_at_model_number = v as isize;
        }
        if let Some(b) = sj.get_bool("new_rsp2_flag") {
            c.rsp2.new_rsp2_flag = b;
            if c.rsp2.change_rsp2_flag { c.rsp2_flag = b; }
        }
        if let Some(b) = sj.get_bool("create_rsp2_model") {
            c.rsp2.create_rsp2_model = b;
        }
    }
    let Some(ctrl) = ctrl else { return; };
    let r = &mut c.rsp;
    for (k, dst) in [
        ("rsp_alfa", &mut r.alfa), ("rsp_alfac", &mut r.alfac), ("rsp_alfas", &mut r.alfas),
        ("rsp_alfad", &mut r.alfad), ("rsp_alfap", &mut r.alfap), ("rsp_alfat", &mut r.alfat),
        ("rsp_alfam", &mut r.alfam), ("rsp_gammar", &mut r.gammar), ("rsp_cq", &mut r.cq),
        ("rsp_zsh", &mut r.zsh), ("rsp_efl0", &mut r.efl0), ("rsp_theta", &mut r.theta),
        ("rsp_thetat", &mut r.thetat), ("rsp_thetae", &mut r.thetae), ("rsp_thetaq", &mut r.thetaq),
        ("rsp_thetau", &mut r.thetau), ("rsp_wtr", &mut r.wtr), ("rsp_wtc", &mut r.wtc),
        ("rsp_wtt", &mut r.wtt), ("rsp_gam", &mut r.gam), ("rsp_t_anchor", &mut r.t_anchor),
        ("rsp_t_inner", &mut r.t_inner), ("rsp_dq_1_factor", &mut r.dq_1_factor),
        ("rsp_t_anchor_tolerance", &mut r.t_anchor_tolerance),
        ("rsp_t_inner_tolerance", &mut r.t_inner_tolerance),
        ("rsp_relax_dm_tolerance", &mut r.relax_dm_tolerance),
        ("rsp_kick_vsurf_km_per_sec", &mut r.kick_vsurf_km_per_sec),
        ("rsp_fraction_1st_overtone", &mut r.fraction_1st_overtone),
        ("rsp_fraction_2nd_overtone", &mut r.fraction_2nd_overtone),
        ("rsp_avel", &mut r.avel), ("rsp_arnd", &mut r.arnd),
        ("rsp_default_periodlin", &mut r.default_periodlin),
        ("rsp_min_max_r_for_periods", &mut r.min_max_r_for_periods),
        ("rsp_min_deltar_for_periods", &mut r.min_deltar_for_periods),
        ("rsp_min_period_div_periodlin", &mut r.min_period_div_periodlin),
        ("rsp_grekm_avg_abs_frac_new", &mut r.grekm_avg_abs_frac_new),
        ("rsp_grekm_avg_abs_limit", &mut r.grekm_avg_abs_limit),
        ("rsp_initial_dt_factor", &mut r.initial_dt_factor),
        ("rsp_v_div_cs_threshold_for_dt_limit", &mut r.v_div_cs_threshold_for_dt_limit),
        ("rsp_max_dt_times_min_dr_div_cs", &mut r.max_dt_times_min_dr_div_cs),
        ("rsp_max_dt_times_min_rad_diff_time", &mut r.max_dt_times_min_rad_diff_time),
        ("rsp_max_dt", &mut r.max_dt),
        ("rsp_tau_surf_for_atm_grey_with_kap", &mut r.tau_surf_for_atm_grey_with_kap),
        ("rsp_psurf", &mut r.psurf),
        ("set_rsp_psurf_to_multiple_of_initial_p1", &mut r.set_psurf_to_multiple_of_initial_p1),
        ("rsp_surface_tau", &mut r.surface_tau), ("rsp_tol_max_corr", &mut r.tol_max_corr),
        ("rsp_tol_max_resid", &mut r.tol_max_resid), ("rsp_nz_div_ibotom", &mut r.nz_div_ibotom),
        ("rsp_min_tau_for_turbulent_flux", &mut r.min_tau_for_turbulent_flux),
        ("rsp_kap_density_factor", &mut r.kap_density_factor),
        ("min_logt_for_opacity_factor_off", &mut r.min_logt_for_opacity_factor_off),
        ("min_logt_for_opacity_factor_on", &mut r.min_logt_for_opacity_factor_on),
        ("max_logt_for_opacity_factor_on", &mut r.max_logt_for_opacity_factor_on),
        ("max_logt_for_opacity_factor_off", &mut r.max_logt_for_opacity_factor_off),
        ("atm_t_tau_errtol", &mut r.atm_t_tau_errtol),
        ("max_timestep_factor", &mut r.max_timestep_factor),
        ("max_timestep", &mut r.max_timestep),
        ("force_timestep", &mut r.force_timestep),
        ("force_timestep_min", &mut r.force_timestep_min),
    ] {
        if let Some(v) = ctrl.get_f64(k) { *dst = v; }
    }
    for (k, dst) in [
        ("rsp_relax_initial_model", &mut r.relax_initial_model),
        ("rsp_relax_alfap_before_alfat", &mut r.relax_alfap_before_alfat),
        ("rsp_relax_adjust_inner_mass_distribution", &mut r.relax_adjust_inner_mass_distribution),
        ("rsp_use_prad_for_psurf", &mut r.use_prad_for_psurf),
        ("rsp_use_atm_grey_with_kap_for_psurf", &mut r.use_atm_grey_with_kap_for_psurf),
        ("rsp_fixed_psurf", &mut r.fixed_psurf), ("rsp_write_map", &mut r.write_map),
        ("use_other_rsp_linear_analysis", &mut r.use_other_rsp_linear_analysis),
        ("use_other_rsp_build_model", &mut r.use_other_rsp_build_model),
        ("use_rsp_new_start_scheme", &mut r.use_rsp_new_start_scheme),
        ("rsp_do_check_omega", &mut r.do_check_omega), ("rsp_hydro_only", &mut r.hydro_only),
        ("initial_dump_missing_heaviest", &mut r.initial_dump_missing_heaviest),
    ] {
        if let Some(v) = ctrl.get_bool(k) { *dst = v; }
    }
    for (k, dst) in [
        ("rsp_nz", &mut r.nz), ("rsp_nz_outer", &mut r.nz_outer),
        ("rsp_max_outer_dm_tries", &mut r.max_outer_dm_tries),
        ("rsp_max_inner_scale_tries", &mut r.max_inner_scale_tries),
        ("rsp_relax_max_tries", &mut r.relax_max_tries), ("rsp_nmodes", &mut r.nmodes),
        ("rsp_mode_for_setting_periodlin", &mut r.mode_for_setting_periodlin),
        ("rsp_max_iters_per_try", &mut r.max_iters_per_try),
        ("rsp_max_retries_per_step", &mut r.max_retries_per_step),
        ("rsp_map_zone_interval", &mut r.map_zone_interval),
        ("atm_t_tau_max_iters", &mut r.atm_t_tau_max_iters),
    ] {
        if let Some(v) = ctrl.get_i64(k).filter(|&v| v >= 0) { *dst = v as usize; }
    }
    for (k, dst) in [
        ("rsp_work_period", &mut r.work_period),
        ("rsp_map_first_period", &mut r.map_first_period),
        ("rsp_map_last_period", &mut r.map_last_period),
    ] {
        if let Some(v) = ctrl.get_i64(k) { *dst = v as isize; }
    }
    for (k, dst) in [
        ("rsp_work_filename", &mut r.work_filename),
        ("rsp_map_columns_filename", &mut r.map_columns_filename),
        ("rsp_map_filename", &mut r.map_filename),
        ("rsp_map_history_filename", &mut r.map_history_filename),
    ] {
        if let Some(s) = ctrl.get_str(k) { *dst = s.to_string(); }
    }

    let r2 = &mut c.rsp2;
    for (k, dst) in [
        ("rsp2_alfad", &mut r2.alfad), ("rsp2_alfap", &mut r2.alfap),
        ("rsp2_alfat", &mut r2.alfat), ("rsp2_alfam", &mut r2.alfam),
        ("rsp2_alfar", &mut r2.alfar), ("rsp2_lsurf_factor", &mut r2.lsurf_factor),
        ("rsp2_w_fix_if_neg", &mut r2.w_fix_if_neg),
        ("rsp2_w_min_for_damping", &mut r2.w_min_for_damping),
        ("rsp2_source_seed", &mut r2.source_seed), ("rsp2_t_anchor", &mut r2.t_anchor),
        ("rsp2_dq_1_factor", &mut r2.dq_1_factor), ("rsp2_nz_div_ibotom", &mut r2.nz_div_ibotom),
    ] {
        if let Some(v) = ctrl.get_f64(k) { *dst = v; }
    }
    for (k, dst) in [
        ("rsp2_use_l_eqn_at_surface", &mut r2.use_l_eqn_at_surface),
        ("rsp2_assume_hse", &mut r2.assume_hse),
        ("rsp2_use_rsp_eqn_for_y_face", &mut r2.use_rsp_eqn_for_y_face),
        ("rsp2_use_mass_interp_face_values", &mut r2.use_mass_interp_face_values),
        ("rsp2_use_stellingwerf_lr", &mut r2.use_stellingwerf_lr),
        ("rsp2_remesh_when_load", &mut r2.remesh_when_load),
        ("use_tdc_enthalpy_flux_limiter", &mut r2.use_tdc_enthalpy_flux_limiter),
    ] {
        if let Some(v) = ctrl.get_bool(k) { *dst = v; }
    }
    for (k, dst) in [
        ("rsp2_num_outermost_cells_forced_nonturbulent", &mut r2.num_outermost_cells_forced_nonturbulent),
        ("rsp2_num_innermost_cells_forced_nonturbulent", &mut r2.num_innermost_cells_forced_nonturbulent),
        ("rsp2_nz", &mut r2.nz), ("rsp2_nz_outer", &mut r2.nz_outer),
        ("rsp2_target_steps_per_cycle", &mut r2.target_steps_per_cycle),
    ] {
        if let Some(v) = ctrl.get_i64(k).filter(|&v| v >= 0) { *dst = v as usize; }
    }
    for (k, dst) in [
        ("rsp2_max_num_periods", &mut r2.max_num_periods),
        ("rsp2_work_period", &mut r2.work_period),
    ] {
        if let Some(v) = ctrl.get_i64(k) { *dst = v as isize; }
    }
    if let Some(s) = ctrl.get_str("rsp2_work_filename") {
        r2.work_filename = s.to_string();
    }
}
