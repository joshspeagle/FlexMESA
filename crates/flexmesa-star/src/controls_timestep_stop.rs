// SPDX-License-Identifier: LGPL-3.0-or-later
//! Core-boundary thresholds, stopping criteria extensions, and soft/hard
//! timestep controller parameters on [`crate::StarControls::ts_stop`]
//! (`star/defaults/controls.defaults:762-950` and `9425-11300`).

/// Number of indexed abundance-change timestep limit slots (`max_dX_limit_ctrls`
/// in `star_data/public/star_data_def.inc`).
pub const MAX_DX_LIMIT_CTRLS: usize = 8;

/// Core-boundary definitions, additional evolutionary stopping limits, and
/// soft/hard timestep controller settings (`timestep.f90`, `report.f90`,
/// `do_one_utils.f90`).
#[derive(Debug, Clone)]
pub struct TimestepStopControls {
    // --- Core Boundary Composition Thresholds (`controls.defaults:762-829`) ---
    /// Outermost $^1\mathrm{H}$ threshold for `he_core_mass` (`he_core_boundary_h1_fraction`, `0.1` default).
    pub he_core_boundary_h1_fraction: f64,
    /// Outermost $^4\mathrm{He}$ threshold for `co_core_mass` (`co_core_boundary_he4_fraction`, `0.1` default).
    pub co_core_boundary_he4_fraction: f64,
    /// Outermost $^4\mathrm{He}+^{12}\mathrm{C}$ threshold for `one_core_mass` (`one_core_boundary_he4_c12_fraction`, `0.1` default).
    pub one_core_boundary_he4_c12_fraction: f64,
    /// Outermost $^{28}\mathrm{Si}$ threshold for `fe_core_mass` (`fe_core_boundary_si28_fraction`, `0.1` default).
    pub fe_core_boundary_si28_fraction: f64,
    /// Outermost $Y_e$ threshold for `neutron_rich_core_mass` (`neutron_rich_core_boundary_Ye_max`, `0.48` default).
    pub neutron_rich_core_boundary_ye_max: f64,
    /// Minimum core species abundance fraction (`min_boundary_fraction`, `0.1` default).
    pub min_boundary_fraction: f64,

    // --- Additional Stopping Criteria (`controls.defaults:937-1720`) ---
    /// Upper limit on central electron degeneracy $\eta_c$ (`eta_center_limit`).
    pub eta_center_limit: Option<f64>,
    /// Collapsing mass threshold ($M_\odot$) inside `fe_core_mass` (`fe_core_infall_mass`, `0.1` default).
    pub fe_core_infall_mass: f64,
    /// Integrate collapsing mass inside `fe_core_mass` when checking `fe_core_infall_limit` (`check_mass_sum_for_infall`).
    pub check_mass_sum_for_infall: bool,
    /// Non-Fe core infall velocity limit in $\mathrm{cm\,s^{-1}}$ (`non_fe_core_infall_limit`).
    pub non_fe_core_infall_limit: Option<f64>,
    /// Collapsing mass threshold ($M_\odot$) between `fe_core_mass` and `he_core_mass` (`non_fe_core_infall_mass`).
    pub non_fe_core_infall_mass: f64,
    /// Upper limit on H-burning luminosity in $L_\odot$ (`power_h_burn_upper_limit`).
    pub power_h_burn_upper_limit: Option<f64>,
    /// Lower limit on H-burning luminosity in $L_\odot$ (`power_h_burn_lower_limit`).
    pub power_h_burn_lower_limit: Option<f64>,
    /// Lower limit on He-burning luminosity in $L_\odot$ (`power_he_burn_lower_limit`).
    pub power_he_burn_lower_limit: Option<f64>,
    /// Upper limit on Z-burning luminosity in $L_\odot$ (`power_z_burn_upper_limit`).
    pub power_z_burn_upper_limit: Option<f64>,
    /// Lower limit on Z-burning luminosity in $L_\odot$ (`power_z_burn_lower_limit`).
    pub power_z_burn_lower_limit: Option<f64>,

    // --- Global Timestep Multipliers, Filters & Retry Holds (`controls.defaults:9425-9754`) ---
    /// Global multiplier on real-valued timestep soft/hard limits (`time_delta_coeff`, `1.0` default).
    pub time_delta_coeff: f64,
    /// Maximum timestep in seconds (`max_timestep`, `<= 0` means unlimited).
    pub max_timestep: f64,
    /// Maximum temperature (K) triggering `hi_t_max_years_for_timestep` (`max_timestep_hi_T_limit`).
    pub max_timestep_hi_t_limit: f64,
    /// Maximum timestep in years when $\max_k T_k \ge T_{\mathrm{hi}}$ (`hi_T_max_years_for_timestep`).
    pub hi_t_max_years_for_timestep: f64,
    /// Minimum allowed $\Delta t_{n+1} / \Delta t_n$ (`min_timestep_factor`, `0.8` default).
    pub min_timestep_factor: f64,
    /// Maximum allowed $\Delta t_{n+1} / \Delta t_n$ (`max_timestep_factor`, `1.2` default).
    pub max_timestep_factor: f64,
    /// High-$T$ maximum $\Delta t_{n+1} / \Delta t_n$ (`max_timestep_factor_at_high_T`, `1.1` default).
    pub max_timestep_factor_at_high_t: f64,
    /// $\log_{10}(T/\mathrm{K})$ threshold for `max_timestep_factor_at_high_t` (`min_logT_for_max_timestep_factor_at_high_T`).
    pub min_logt_for_max_timestep_factor_at_high_t: f64,
    /// Forced timestep in seconds (`force_timestep`, `<= 0` disabled).
    pub force_timestep: f64,
    /// Forced timestep in years (`force_timestep_years`, `<= 0` disabled).
    pub force_timestep_years: f64,
    /// Forced minimum timestep in seconds (`force_timestep_min`, `<= 0` disabled).
    pub force_timestep_min: f64,
    /// Forced minimum timestep in years (`force_timestep_min_years`, `<= 0` disabled).
    pub force_timestep_min_years: f64,
    /// Growth factor toward `force_timestep_min` (`force_timestep_min_factor`, `2.0` default).
    pub force_timestep_min_factor: f64,
    /// Multiplicative $\Delta t$ reduction on step retry (`timestep_factor_for_retries`, `0.5` default).
    pub timestep_factor_for_retries: f64,
    /// Number of steps after a retry during which $\Delta t$ cannot grow (`retry_hold`, `1` default).
    pub retry_hold: usize,
    /// Hold steps after a negative-abundance retry (`neg_mass_fraction_hold`, `2` default).
    pub neg_mass_fraction_hold: usize,
    /// Maximum retries allowed per step (`retry_limit`, `100` default).
    pub retry_limit: usize,
    /// Maximum redos allowed per step (`redo_limit`, `100` default).
    pub redo_limit: usize,
    /// Soft-limit reduction factor (`timestep_dt_factor`, `0.9` default).
    pub timestep_dt_factor: f64,
    /// Enable Söderlind & Wang (2006) H211 low-pass controller (`use_dt_low_pass_controller`, `true` default).
    pub use_dt_low_pass_controller: bool,
    /// Minimum allowed `varcontrol_target` (`min_allowed_varcontrol_target`, `1e-4` default).
    pub min_allowed_varcontrol_target: f64,
    /// Hard retry limit on `varcontrol / varcontrol_target` (`varcontrol_dt_limit_ratio_hard_max`, `1e99` default).
    pub varcontrol_dt_limit_ratio_hard_max: f64,
    /// Never skip hard limits after a retry (`never_skip_hard_limits`, `true` default).
    pub never_skip_hard_limits: bool,
    /// Relax hard limits immediately after a retry (`relax_hard_limits_after_retry`, `false` default).
    pub relax_hard_limits_after_retry: bool,
    /// Soft limit on non-gold Newton solver iterations (`solver_iters_timestep_limit`, `7` default).
    pub solver_iters_timestep_limit: isize,
    /// Soft limit on gold Newton solver iterations (`gold_solver_iters_timestep_limit`, `14` default).
    pub gold_solver_iters_timestep_limit: isize,
    /// Soft limit on gold2 Newton solver iterations (`gold2_solver_iters_timestep_limit`, `20` default).
    pub gold2_solver_iters_timestep_limit: isize,
    /// Soft limit on 1-zone burn substeps (`burn_steps_limit`, `10000` default).
    pub burn_steps_limit: isize,
    /// Soft limit on element diffusion substeps (`diffusion_steps_limit`, `500` default).
    pub diffusion_steps_limit: isize,
    /// Hard retry limit on element diffusion substeps (`diffusion_steps_hard_limit`, `700` default).
    pub diffusion_steps_hard_limit: isize,
    /// Soft limit on element diffusion iterations (`diffusion_iters_limit`, `600` default).
    pub diffusion_iters_limit: isize,
    /// Hard retry limit on element diffusion iterations (`diffusion_iters_hard_limit`, `800` default).
    pub diffusion_iters_hard_limit: isize,

    // --- Abundance & Nuclear Depletion Timestep Limits (`controls.defaults:9758-9950`) ---
    /// Distance ($M_\odot$) from convective boundaries excluded from `dX` checks (`dX_mix_dist_limit`).
    pub dx_mix_dist_limit: f64,
    /// Species names (`"h1"`, `"he4"`, `"X"`, `"Y"`, `"Z"`, ...) for `dX_limit(1..8)`.
    pub dx_limit_species: [String; MAX_DX_LIMIT_CTRLS],
    /// Minimum cell abundance for `dX_limit(1..8)` (`dX_limit_min_X`).
    pub dx_limit_min_x: [f64; MAX_DX_LIMIT_CTRLS],
    /// Soft limit on $|\Delta X|$ (`dX_limit(1..8)`).
    pub dx_limit: [f64; MAX_DX_LIMIT_CTRLS],
    /// Hard retry limit on $|\Delta X|$ (`dX_hard_limit(1..8)`).
    pub dx_hard_limit: [f64; MAX_DX_LIMIT_CTRLS],
    /// Only check decreases in abundance (`dX_decreases_only(1..8)`).
    pub dx_decreases_only: [bool; MAX_DX_LIMIT_CTRLS],
    /// Minimum cell abundance for `dX_div_X_limit(1..8)` (`dX_div_X_limit_min_X`).
    pub dx_div_x_limit_min_x: [f64; MAX_DX_LIMIT_CTRLS],
    /// Soft limit on $|\Delta X| / X$ (`dX_div_X_limit(1..8)`).
    pub dx_div_x_limit: [f64; MAX_DX_LIMIT_CTRLS],
    /// Hard retry limit on $|\Delta X| / X$ (`dX_div_X_hard_limit(1..8)`).
    pub dx_div_x_hard_limit: [f64; MAX_DX_LIMIT_CTRLS],
    /// Minimum abundance for `dX_nuc_drop` (`dX_nuc_drop_min_X_limit`, `1e-4` default).
    pub dx_nuc_drop_min_x_limit: f64,
    /// Maximum mass number $A$ for `dX_nuc_drop` (`dX_nuc_drop_max_A_limit`, `52.0` default).
    pub dx_nuc_drop_max_a_limit: f64,
    /// High-$T$ override for `dX_nuc_drop_limit` when $\max \log_{10} T \ge 9.45$ (`dX_nuc_drop_limit_at_high_T`).
    pub dx_nuc_drop_limit_at_high_t: f64,
    /// Soft limit on nuclear burning + inflow drop (`dX_nuc_drop_limit`, `5e-2` default).
    pub dx_nuc_drop_limit: f64,
    /// Hard retry limit on nuclear burning + inflow drop (`dX_nuc_drop_hard_limit`, `1e99` default).
    pub dx_nuc_drop_hard_limit: f64,
    /// Minimum $\Delta t$ in years below which `dX_nuc_drop` is inactive (`dX_nuc_drop_min_yrs_for_dt`, `1e-9` default).
    pub dx_nuc_drop_min_yrs_for_dt: f64,

    // --- Structural, Luminosity, HR, Center & Courant Limits (`controls.defaults:9951-11300`) ---
    /// Soft limit on $|L_k - L_{\mathrm{start},k}| / L_k$ (`dL_div_L_limit`).
    pub dl_div_l_limit: f64,
    /// Hard limit on $|L_k - L_{\mathrm{start},k}| / L_k$ (`dL_div_L_hard_limit`).
    pub dl_div_l_hard_limit: f64,
    /// Minimum $L_k / L_\odot$ for `dL_div_L` (`dL_div_L_limit_min_L`).
    pub dl_div_l_limit_min_l: f64,
    /// Soft/hard/min limits on $|\Delta\log_{10} P_{\mathrm{eos}}|$ (`delta_lgP_*`).
    pub delta_lgp_limit: f64,
    /// Hard retry limit on $|\Delta\log_{10} P_{\mathrm{eos}}|$.
    pub delta_lgp_hard_limit: f64,
    /// Minimum $\log_{10} P_{\mathrm{eos}}$ for `delta_lgP` checks.
    pub delta_lgp_limit_min_lgp: f64,
    /// Soft/hard/min limits on $|\Delta\log_{10}\rho|$ (`delta_lgRho_*`).
    pub delta_lgrho_limit: f64,
    /// Hard retry limit on $|\Delta\log_{10}\rho|$.
    pub delta_lgrho_hard_limit: f64,
    /// Minimum $\log_{10}\rho$ for `delta_lgRho` checks.
    pub delta_lgrho_limit_min_lgrho: f64,
    /// Soft/hard/min limits on $|\Delta\log_{10} T|$ (`delta_lgT_*`).
    pub delta_lgt_limit: f64,
    /// Hard retry limit on $|\Delta\log_{10} T|$.
    pub delta_lgt_hard_limit: f64,
    /// Minimum $\log_{10} T$ for `delta_lgT` checks.
    pub delta_lgt_limit_min_lgt: f64,
    /// Soft/hard/min limits on $|\Delta\log_{10} e|$ (`delta_lgE_*`).
    pub delta_lge_limit: f64,
    /// Hard retry limit on $|\Delta\log_{10} e|$.
    pub delta_lge_hard_limit: f64,
    /// Minimum $\log_{10} e$ for `delta_lgE` checks.
    pub delta_lge_limit_min_lge: f64,
    /// Soft/hard/min limits on $|\Delta\log_{10} r|$ (`delta_lgR_*`).
    pub delta_lgr_limit: f64,
    /// Hard retry limit on $|\Delta\log_{10} r|$.
    pub delta_lgr_hard_limit: f64,
    /// Minimum $\log_{10} r$ for `delta_lgR` checks.
    pub delta_lgr_limit_min_lgr: f64,
    /// Soft/hard/min limits on $|\Delta\log_{10} L_{\mathrm{H}}|$ (`delta_lgL_H_*`).
    pub delta_lgl_h_limit: f64,
    /// Hard retry limit on $|\Delta\log_{10} L_{\mathrm{H}}|$.
    pub delta_lgl_h_hard_limit: f64,
    /// Minimum $\log_{10}(L_{\mathrm{H}}/L_\odot)$ for `delta_lgL_H` (`lgL_H_burn_min`).
    pub lgl_h_burn_min: f64,
    /// Multiplier when $L_{\mathrm{H}}$ is dropping (`lgL_H_drop_factor`).
    pub lgl_h_drop_factor: f64,
    /// Soft/hard/min limits on $|\Delta\log_{10} L_{\mathrm{He}}|$ (`delta_lgL_He_*`).
    pub delta_lgl_he_limit: f64,
    /// Hard retry limit on $|\Delta\log_{10} L_{\mathrm{He}}|$.
    pub delta_lgl_he_hard_limit: f64,
    /// Minimum $\log_{10}(L_{\mathrm{He}}/L_\odot)$ for `delta_lgL_He` (`lgL_He_burn_min`).
    pub lgl_he_burn_min: f64,
    /// Multiplier when $L_{\mathrm{He}}$ is dropping (`lgL_He_drop_factor`).
    pub lgl_he_drop_factor: f64,
    /// Soft/hard/min limits on $|\Delta\log_{10} L_Z|$ (`delta_lgL_z_*`).
    pub delta_lgl_z_limit: f64,
    /// Hard retry limit on $|\Delta\log_{10} L_Z|$.
    pub delta_lgl_z_hard_limit: f64,
    /// Minimum $\log_{10}(L_Z/L_\odot)$ for `delta_lgL_z` (`lgL_z_burn_min`).
    pub lgl_z_burn_min: f64,
    /// Multiplier when $L_Z$ is dropping (`lgL_z_drop_factor`).
    pub lgl_z_drop_factor: f64,
    /// Soft/hard/min limits on $|\Delta\log_{10} L_{\mathrm{nuc}}|$ (`delta_lgL_nuc_*`).
    pub delta_lgl_nuc_limit: f64,
    /// Hard retry limit on $|\Delta\log_{10} L_{\mathrm{nuc}}|$.
    pub delta_lgl_nuc_hard_limit: f64,
    /// Minimum $\log_{10}(L_{\mathrm{nuc}}/L_\odot)$ for `delta_lgL_nuc` (`lgL_nuc_burn_min`).
    pub lgl_nuc_burn_min: f64,
    /// Multiplier when $L_{\mathrm{nuc}}$ is dropping (`lgL_nuc_drop_factor`).
    pub lgl_nuc_drop_factor: f64,
    /// Soft/hard/min limits on $|\Delta\log_{10} |L_{\mathrm{photo}}||$ (`delta_lgL_power_photo_*`).
    pub delta_lgl_power_photo_limit: f64,
    /// Hard retry limit on $|\Delta\log_{10} |L_{\mathrm{photo}}||$.
    pub delta_lgl_power_photo_hard_limit: f64,
    /// Minimum $\max_k \log_{10} T_k$ for `delta_lgL_power_photo` (`min_lgT_for_lgL_power_photo_limit`).
    pub min_lgt_for_lgl_power_photo_limit: f64,
    /// Minimum $\log_{10}(|L_{\mathrm{photo}}|/L_\odot)$ (`lgL_power_photo_burn_min`).
    pub lgl_power_photo_burn_min: f64,
    /// Multiplier when $|L_{\mathrm{photo}}|$ is dropping (`lgL_power_photo_drop_factor`).
    pub lgl_power_photo_drop_factor: f64,
    /// Soft/hard limits on $|\Delta\log_{10} T_{\mathrm{eff}}|$ (`delta_lgTeff_*`).
    pub delta_lgteff_limit: f64,
    /// Hard retry limit on $|\Delta\log_{10} T_{\mathrm{eff}}|$.
    pub delta_lgteff_hard_limit: f64,
    /// Soft/hard/min limits on $|\Delta\log_{10}(L_{\mathrm{surf}}/L_\odot)|$ (`delta_lgL_*`).
    pub delta_lgl_limit_l_min: f64,
    /// Soft limit on $|\Delta\log_{10}(L_{\mathrm{surf}}/L_\odot)|$.
    pub delta_lgl_limit: f64,
    /// Hard retry limit on $|\Delta\log_{10}(L_{\mathrm{surf}}/L_\odot)|$.
    pub delta_lgl_hard_limit: f64,
    /// Soft/hard limits on central $|\Delta\log_{10} T_c|$ (`delta_lgT_cntr_*`).
    pub delta_lgt_cntr_limit: f64,
    /// Hard retry limit on central $|\Delta\log_{10} T_c|$.
    pub delta_lgt_cntr_hard_limit: f64,
    /// Soft/hard limits on central $|\Delta\log_{10}\rho_c|$ (`delta_lgRho_cntr_*`).
    pub delta_lgrho_cntr_limit: f64,
    /// Hard retry limit on central $|\Delta\log_{10}\rho_c|$.
    pub delta_lgrho_cntr_hard_limit: f64,
    /// Soft/hard limits on central $|\Delta\log_{10} P_c|$ (`delta_lgP_cntr_*`).
    pub delta_lgp_cntr_limit: f64,
    /// Hard retry limit on central $|\Delta\log_{10} P_c|$.
    pub delta_lgp_cntr_hard_limit: f64,
    /// Soft/hard limits on $|\Delta\max_k\log_{10} T_k|$ (`delta_lgT_max_*`).
    pub delta_lgt_max_limit: f64,
    /// Hard retry limit on $|\Delta\max_k\log_{10} T_k|$.
    pub delta_lgt_max_hard_limit: f64,
    /// HR-diagram step weights & limits (`delta_HR_*`).
    pub delta_hr_ds_l: f64,
    /// HR-diagram $\log_{10} T_{\mathrm{eff}}$ weight (`delta_HR_ds_Teff`).
    pub delta_hr_ds_teff: f64,
    /// Soft limit on $\Delta_{\mathrm{HR}}$ (`delta_HR_limit`).
    pub delta_hr_limit: f64,
    /// Hard retry limit on $\Delta_{\mathrm{HR}}$ (`delta_HR_hard_limit`).
    pub delta_hr_hard_limit: f64,
    /// Soft/hard limits on $|\Delta\log_{10}(M_*/M_\odot)|$ (`delta_lg_star_mass_*`).
    pub delta_lg_star_mass_limit: f64,
    /// Hard retry limit on $|\Delta\log_{10}(M_*/M_\odot)|$.
    pub delta_lg_star_mass_hard_limit: f64,
    /// Soft/hard limits on per-step relative energy conservation error (`*limit_for_rel_error_in_energy_conservation`).
    pub limit_for_rel_error_in_energy_conservation: f64,
    /// Hard retry limit on per-step relative energy conservation error.
    pub hard_limit_for_rel_error_in_energy_conservation: f64,
    /// Soft/hard Courant-number limits $\Delta t / \min_k(\Delta r_k / c_{s,k})$ (`dt_div_min_dr_div_cs_*`).
    pub dt_div_min_dr_div_cs_limit: f64,
    /// Hard retry limit on Courant number $\Delta t / \min_k(\Delta r_k / c_{s,k})$.
    pub dt_div_min_dr_div_cs_hard_limit: f64,
    /// Soft/hard cell-collapse limits $\Delta t / \Delta t_{\mathrm{collapse}}$ (`dt_div_dt_cell_collapse_*`).
    pub dt_div_dt_cell_collapse_limit: f64,
    /// Hard retry limit on $\Delta t / \Delta t_{\mathrm{collapse}}$.
    pub dt_div_dt_cell_collapse_hard_limit: f64,
}

impl Default for TimestepStopControls {
    fn default() -> Self {
        let mut dx_limit_species: [String; MAX_DX_LIMIT_CTRLS] =
            std::array::from_fn(|_| String::new());
        dx_limit_species[0] = String::from("h1");
        dx_limit_species[1] = String::from("he4");
        let mut dx_div_x_limit_min_x = [1.0e99; MAX_DX_LIMIT_CTRLS];
        dx_div_x_limit_min_x[0] = 1.0e-3;
        dx_div_x_limit_min_x[1] = 1.0e-3;
        let mut dx_div_x_limit = [1.0e99; MAX_DX_LIMIT_CTRLS];
        dx_div_x_limit[0] = 0.9;
        dx_div_x_limit[1] = 0.9;

        Self {
            he_core_boundary_h1_fraction: 0.1,
            co_core_boundary_he4_fraction: 0.1,
            one_core_boundary_he4_c12_fraction: 0.1,
            fe_core_boundary_si28_fraction: 0.1,
            neutron_rich_core_boundary_ye_max: 0.48,
            min_boundary_fraction: 0.1,
            eta_center_limit: None,
            fe_core_infall_mass: 0.1,
            check_mass_sum_for_infall: false,
            non_fe_core_infall_limit: None,
            non_fe_core_infall_mass: 0.1,
            power_h_burn_upper_limit: None,
            power_h_burn_lower_limit: None,
            power_he_burn_lower_limit: None,
            power_z_burn_upper_limit: None,
            power_z_burn_lower_limit: None,
            time_delta_coeff: 1.0,
            max_timestep: 0.0,
            max_timestep_hi_t_limit: -1.0,
            hi_t_max_years_for_timestep: 0.0,
            min_timestep_factor: 0.8,
            max_timestep_factor: 1.2,
            max_timestep_factor_at_high_t: 1.1,
            min_logt_for_max_timestep_factor_at_high_t: 1.0e99,
            force_timestep: 0.0,
            force_timestep_years: 0.0,
            force_timestep_min: 0.0,
            force_timestep_min_years: 0.0,
            force_timestep_min_factor: 2.0,
            timestep_factor_for_retries: 0.5,
            retry_hold: 1,
            neg_mass_fraction_hold: 2,
            retry_limit: 100,
            redo_limit: 100,
            timestep_dt_factor: 0.9,
            use_dt_low_pass_controller: true,
            min_allowed_varcontrol_target: 1.0e-4,
            varcontrol_dt_limit_ratio_hard_max: 1.0e99,
            never_skip_hard_limits: true,
            relax_hard_limits_after_retry: false,
            solver_iters_timestep_limit: 7,
            gold_solver_iters_timestep_limit: 14,
            gold2_solver_iters_timestep_limit: 20,
            burn_steps_limit: 10_000,
            diffusion_steps_limit: 500,
            diffusion_steps_hard_limit: 700,
            diffusion_iters_limit: 600,
            diffusion_iters_hard_limit: 800,
            dx_mix_dist_limit: 1.0e-4,
            dx_limit_species,
            dx_limit_min_x: [1.0e99; MAX_DX_LIMIT_CTRLS],
            dx_limit: [1.0e99; MAX_DX_LIMIT_CTRLS],
            dx_hard_limit: [1.0e99; MAX_DX_LIMIT_CTRLS],
            dx_decreases_only: [true; MAX_DX_LIMIT_CTRLS],
            dx_div_x_limit_min_x,
            dx_div_x_limit,
            dx_div_x_hard_limit: [1.0e99; MAX_DX_LIMIT_CTRLS],
            dx_nuc_drop_min_x_limit: 1.0e-4,
            dx_nuc_drop_max_a_limit: 52.0,
            dx_nuc_drop_limit_at_high_t: -1.0,
            dx_nuc_drop_limit: 5.0e-2,
            dx_nuc_drop_hard_limit: 1.0e99,
            dx_nuc_drop_min_yrs_for_dt: 1.0e-9,
            dl_div_l_limit: -1.0,
            dl_div_l_hard_limit: -1.0,
            dl_div_l_limit_min_l: 1.0e99,
            delta_lgp_limit: 1.0,
            delta_lgp_hard_limit: -1.0,
            delta_lgp_limit_min_lgp: 1.0e99,
            delta_lgrho_limit: 1.0,
            delta_lgrho_hard_limit: -1.0,
            delta_lgrho_limit_min_lgrho: 1.0e99,
            delta_lgt_limit: 0.5,
            delta_lgt_hard_limit: -1.0,
            delta_lgt_limit_min_lgt: 1.0e99,
            delta_lge_limit: 0.1,
            delta_lge_hard_limit: -1.0,
            delta_lge_limit_min_lge: 1.0e99,
            delta_lgr_limit: 0.5,
            delta_lgr_hard_limit: -1.0,
            delta_lgr_limit_min_lgr: 1.0e99,
            delta_lgl_h_limit: -1.0,
            delta_lgl_h_hard_limit: -1.0,
            lgl_h_burn_min: 1.5,
            lgl_h_drop_factor: 1.0,
            delta_lgl_he_limit: 0.025,
            delta_lgl_he_hard_limit: -1.0,
            lgl_he_burn_min: 2.5,
            lgl_he_drop_factor: 1.0,
            delta_lgl_z_limit: -1.0,
            delta_lgl_z_hard_limit: -1.0,
            lgl_z_burn_min: 2.5,
            lgl_z_drop_factor: 1.0,
            delta_lgl_nuc_limit: -1.0,
            delta_lgl_nuc_hard_limit: -1.0,
            lgl_nuc_burn_min: 0.5,
            lgl_nuc_drop_factor: 10.0,
            delta_lgl_power_photo_limit: -1.0,
            delta_lgl_power_photo_hard_limit: -1.0,
            min_lgt_for_lgl_power_photo_limit: 9.0,
            lgl_power_photo_burn_min: 10.0,
            lgl_power_photo_drop_factor: 10.0,
            delta_lgteff_limit: 0.01,
            delta_lgteff_hard_limit: -1.0,
            delta_lgl_limit_l_min: -100.0,
            delta_lgl_limit: 0.1,
            delta_lgl_hard_limit: -1.0,
            delta_lgt_cntr_limit: 0.01,
            delta_lgt_cntr_hard_limit: -1.0,
            delta_lgrho_cntr_limit: 0.05,
            delta_lgrho_cntr_hard_limit: -1.0,
            delta_lgp_cntr_limit: 0.05,
            delta_lgp_cntr_hard_limit: -1.0,
            delta_lgt_max_limit: 0.01,
            delta_lgt_max_hard_limit: -1.0,
            delta_hr_ds_l: 1.0,
            delta_hr_ds_teff: 1.0,
            delta_hr_limit: -1.0,
            delta_hr_hard_limit: -1.0,
            delta_lg_star_mass_limit: 5.0e-3,
            delta_lg_star_mass_hard_limit: -1.0,
            limit_for_rel_error_in_energy_conservation: 1.0e-4,
            hard_limit_for_rel_error_in_energy_conservation: 1.0e-3,
            dt_div_min_dr_div_cs_limit: -1.0,
            dt_div_min_dr_div_cs_hard_limit: -1.0,
            dt_div_dt_cell_collapse_limit: -1.0,
            dt_div_dt_cell_collapse_hard_limit: -1.0,
        }
    }
}
