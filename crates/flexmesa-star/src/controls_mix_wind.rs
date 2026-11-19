// SPDX-License-Identifier: LGPL-3.0-or-later
//! Overshooting, predictive mixing, convective premixing, stellar wind,
//! rotational mass-loss enhancement, `eps_mdot`, extra heating, and adaptive
//! mesh spacing control sub-structs on [`crate::StarControls`]
//! (`star/defaults/controls.defaults`).

/// Number of indexed overshooting parameter sets (`NUM_OVERSHOOT_PARAM_SETS` in
/// `star_data/public/star_data_def.inc`).
pub const NUM_OVERSHOOT_PARAM_SETS: usize = 16;

/// Number of indexed predictive mixing parameter sets
/// (`NUM_PREDICTIVE_PARAM_SETS` in `star_data/public/star_data_def.inc`).
pub const NUM_PREDICTIVE_PARAM_SETS: usize = 16;

/// Number of indexed species abundance mesh spacing functions
/// (`num_xa_function` in `star_data/public/star_data_def.inc`).
pub const NUM_XA_FUNCTION: usize = 9;

/// Number of indexed $\mathrm{d}\log X / \mathrm{d}\log P$ mesh refinement rules
/// (`num_mesh_logX` in `star_data/public/star_data_def.inc`).
pub const NUM_MESH_LOGX: usize = 9;

/// One indexed convective overshooting parameter set (`1..=16` in
/// `controls.defaults:2580-2660`).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct OvershootParamSet {
    /// Scheme name (`""`, `"exponential"`, `"step"`, `"step+exponential"`, `"other"`).
    pub scheme: String,
    /// Convection-zone nuclear burning criterion (`""`, `"burn_H"`, `"burn_He"`, `"burn_Z"`, `"nonburn"`, `"any"`).
    pub zone_type: String,
    /// Convection-zone radial location criterion (`""`, `"core"`, `"shell"`, `"any"`).
    pub zone_loc: String,
    /// Boundary orientation criterion (`""`, `"bottom"`, `"top"`, `"any"`).
    pub bdy_loc: String,
    /// Primary overshooting length parameter $f$ in units of $H_{P,0}$ (`overshoot_f`).
    pub f: f64,
    /// Step-back fraction $f_0$ inside the convective boundary (`overshoot_f0`).
    pub f0: f64,
    /// Secondary exponential length parameter $f_2$ for `"step+exponential"` (`overshoot_f2`).
    pub f2: f64,
    /// Additive diffusivity floor $D_0$ ($\mathrm{cm^2\,s^{-1}}$, `overshoot_D0`).
    pub d0: f64,
    /// Multiplicative boundary diffusivity scale $\Delta_0$ (`overshoot_Delta0`, `1.0` default).
    pub delta0: f64,
    /// Stellar mass ($M_\odot$) at which H-burning overshooting is fully enabled (`overshoot_mass_full_on`).
    pub mass_full_on: f64,
    /// Stellar mass ($M_\odot$) below which H-burning overshooting vanishes (`overshoot_mass_full_off`).
    pub mass_full_off: f64,
}

impl OvershootParamSet {
    /// Upstream default parameter set (`overshoot_Delta0 = 1.0`, others zero/empty).
    #[must_use]
    pub fn mesa_default() -> Self {
        Self {
            delta0: 1.0,
            ..Self::default()
        }
    }
}

/// One indexed predictive mixing parameter set (`1..=16` in
/// `controls.defaults:2782-2885`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PredictiveParamSet {
    /// Enable this predictive mixing parameter set (`predictive_mix(i)`).
    pub predictive_mix: bool,
    /// Convection-zone burning criterion (`predictive_zone_type(i)`).
    pub zone_type: String,
    /// Convection-zone location criterion (`predictive_zone_loc(i)`).
    pub zone_loc: String,
    /// Boundary orientation criterion (`predictive_bdy_loc(i)`).
    pub bdy_loc: String,
    /// Minimum boundary mass fraction $q_{\min}$ (`predictive_bdy_q_min(i)`).
    pub bdy_q_min: f64,
    /// Maximum boundary mass fraction $q_{\max}$ (`predictive_bdy_q_max(i)`).
    pub bdy_q_max: f64,
    /// Minimum superadiabaticity $\nabla_r / \nabla_{\mathrm{ad}} - 1$ (`predictive_superad_thresh(i)`).
    pub superad_thresh: f64,
    /// Isotope monitored to prevent abundance evolution reversals (`predictive_avoid_reversal(i)`).
    pub avoid_reversal: String,
    /// Isotope subject to Constantino et al. (2017) ingestion rate cap (`predictive_limit_ingestion(i)`).
    pub limit_ingestion: String,
    /// Ingestion rate multiplier (`predictive_ingestion_factor(i)`).
    pub ingestion_factor: f64,
}

impl Default for PredictiveParamSet {
    fn default() -> Self {
        Self {
            predictive_mix: false,
            zone_type: String::new(),
            zone_loc: String::new(),
            bdy_loc: String::new(),
            bdy_q_min: 0.0,
            bdy_q_max: 1.0,
            superad_thresh: 0.0,
            avoid_reversal: String::new(),
            limit_ingestion: String::new(),
            ingestion_factor: 0.0,
        }
    }
}

/// Mixing, wind, accretion, `eps_mdot`, extra-heat, and adaptive mesh controls
/// grouped onto [`crate::StarControls::mix_wind`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MixWindControls {
    /// Indexed overshooting parameter sets `1..=16`.
    pub overshoot_sets: [OvershootParamSet; NUM_OVERSHOOT_PARAM_SETS],
    /// Minimum diffusivity cutoff $D_{\min}$ ($\mathrm{cm^2\,s^{-1}}$, `overshoot_D_min`).
    pub overshoot_d_min: f64,
    /// Stabilizing composition buoyancy cutoff $B_{\max}$ (`overshoot_brunt_B_max`, `<= 0` disabled).
    pub overshoot_brunt_b_max: f64,
    /// Minimum enclosed mass fraction $q$ permitted to overshoot (`min_overshoot_q`).
    pub min_overshoot_q: f64,
    /// Override mixing-length $\alpha$ when capping $H_P \le \Delta r_{\mathrm{cz}} / \alpha$ (`overshoot_alpha`).
    pub overshoot_alpha: f64,
    /// Cap overshooting scale height by convection-zone radial thickness (`limit_overshoot_Hp_using_size_of_convection_zone`).
    pub limit_overshoot_hp_using_size_of_convection_zone: bool,
    /// Minimum $\log_{10}(T/\mathrm{K})$ classifying a convective zone as `burn_Z` (`burn_z_mix_region_logT`).
    pub burn_z_mix_region_logt: f64,
    /// Minimum $\log_{10}(T/\mathrm{K})$ classifying a convective zone as `burn_He` (`burn_he_mix_region_logT`).
    pub burn_he_mix_region_logt: f64,
    /// Minimum $\log_{10}(T/\mathrm{K})$ classifying a convective zone as `burn_H` (`burn_h_mix_region_logT`).
    pub burn_h_mix_region_logt: f64,
    /// Maximum $Y$ across zone for `burn_Z` classification (`max_Y_for_burn_z_mix_region`).
    pub max_y_for_burn_z_mix_region: f64,
    /// Maximum $X$ across zone for `burn_He` classification (`max_X_for_burn_he_mix_region`).
    pub max_x_for_burn_he_mix_region: f64,
    /// Péclet number above which full convective diffusivity is applied in overshooting (`peclet_function_full_on`).
    pub peclet_function_full_on: f64,
    /// Péclet number below which thermal diffusivity replaces convective diffusivity (`peclet_function_full_off`).
    pub peclet_function_full_off: f64,
    /// Indexed predictive mixing parameter sets `1..=16`.
    pub predictive_sets: [PredictiveParamSet; NUM_PREDICTIVE_PARAM_SETS],
    /// Prevent fuel mass-fraction increases during `do_conv_premix` (`conv_premix_avoid_increase`).
    pub conv_premix_avoid_increase: bool,
    /// Convective turnover time limit factor during `do_conv_premix` (`conv_premix_time_factor`).
    pub conv_premix_time_factor: f64,
    /// Hold $P_{\mathrm{gas}}$ fixed (`true`) or $P_{\mathrm{tot}}$ fixed (`false`) during `do_conv_premix` (`conv_premix_fix_pgas`).
    pub conv_premix_fix_pgas: bool,
    /// Latch radiative faces to `NoMixing` after Newton iteration 3 (`make_gradr_sticky_in_solver_iters`).
    pub make_gradr_sticky_in_solver_iters: bool,
    /// Peak $\log_{10}(T/\mathrm{K})$ threshold enabling `make_gradr_sticky_in_solver_iters` (`min_logT_for_make_gradr_sticky_in_solver_iters`).
    pub min_logt_for_make_gradr_sticky_in_solver_iters: f64,
    /// Enable diffusive mixing floor `min_D_mix` (`set_min_D_mix`, `false` default).
    pub set_min_d_mix: bool,
    /// Minimum mass coordinate ($M_\odot$) for applying `min_D_mix` (`mass_lower_limit_for_min_D_mix`, `0.0` default).
    pub mass_lower_limit_for_min_d_mix: f64,
    /// Maximum mass coordinate ($M_\odot$) for applying `min_D_mix` (`mass_upper_limit_for_min_D_mix`, `1e99` default).
    pub mass_upper_limit_for_min_d_mix: f64,
    /// Minimum central electron fraction $Y_{e,c}$ required to apply `min_D_mix` (`min_center_Ye_for_min_D_mix`, `0.47` default).
    pub min_center_ye_for_min_d_mix: f64,
    /// Remove isolated 1-zone mixing spikes when `remove_mixing_glitches` is active (`okay_to_remove_mixing_singleton`, `true` default).
    pub okay_to_remove_mixing_singleton: bool,
    /// Minimum diffusivity threshold ($\mathrm{cm^2\,s^{-1}}$) below which mixing is zeroed (`clip_D_limit`, `0.0` default).
    pub clip_d_limit: f64,
    /// Maximum radiative gap ($H_P$) closed between convective zones (`min_convective_gap`, `-1.0` default = off).
    pub min_convective_gap: f64,
    /// Maximum gap ($H_P$) closed between thermohaline zones (`min_thermohaline_gap`, `-1.0` default = off).
    pub min_thermohaline_gap: f64,
    /// Maximum radiative dropout ($H_P$) bridged inside thermohaline regions (`min_thermohaline_dropout`, `-1.0` default = off).
    pub min_thermohaline_dropout: f64,
    /// Maximum $\nabla_L - \nabla_{\mathrm{ad}}$ permitted inside a bridged thermohaline dropout (`max_dropout_gradL_sub_grada`, `1e-3` default).
    pub max_dropout_gradl_sub_grada: f64,
    /// Maximum gap ($H_P$) closed between semiconvective zones (`min_semiconvection_gap`, `-1.0` default = off).
    pub min_semiconvection_gap: f64,
    /// Convert semiconvective islands bounded by convection into convective zones (`remove_embedded_semiconvection`, `false` default).
    pub remove_embedded_semiconvection: bool,

    // --- Winds, Rotational Enhancement, Accretion & `eps_mdot` ---
    /// Hot-star wind prescription name (`hot_wind_scheme`, `""` default).
    pub hot_wind_scheme: String,
    /// Central $^4\mathrm{He}$ threshold switching cool wind from RGB to AGB (`RGB_to_AGB_wind_switch`).
    pub rgb_to_agb_wind_switch: f64,
    /// Low-$T_{\mathrm{eff}}$ sub-scheme for `"Dutch"` wind (`Dutch_wind_lowT_scheme`, `"de Jager"` default).
    pub dutch_wind_lowt_scheme: String,
    /// Björklund et al. (2021/2023) O-star wind scaling factor (`Bjorklund_scaling_factor`).
    pub bjorklund_scaling_factor: f64,
    /// de Jager et al. (1988) cool-wind scaling factor (`de_Jager_scaling_factor`).
    pub de_jager_scaling_factor: f64,
    /// van Loon et al. (2005) dust-driven wind scaling factor (`van_Loon_scaling_factor`).
    pub van_loon_scaling_factor: f64,
    /// Nieuwenhuijzen & de Jager (1990) wind scaling factor (`Nieuwenhuijzen_scaling_factor`).
    pub nieuwenhuijzen_scaling_factor: f64,
    /// Floor on $|\dot{M}_{\mathrm{wind}}|$ in $M_\odot\,\mathrm{yr}^{-1}$ (`min_wind`, `<= 0` disabled).
    pub min_wind: f64,
    /// Ceiling on $|\dot{M}_{\mathrm{wind}}|$ in $M_\odot\,\mathrm{yr}^{-1}$ (`max_wind`, `<= 0` disabled).
    pub max_wind: f64,
    /// Heger, Langer & Woosley (2000) rotational wind exponent (`mdot_omega_power`, `0.43` default).
    pub mdot_omega_power: f64,
    /// Cap on rotational wind enhancement factor (`max_rotational_mdot_boost`, `1e4` default).
    pub max_rotational_mdot_boost: f64,
    /// Maximum step-to-step rotational boost jump (`max_mdot_jump_for_rotation`, `2.0` default).
    pub max_mdot_jump_for_rotation: f64,
    /// Near-critical rotational boost multiplier (`rotational_mdot_boost_fac`, `1e5` default).
    pub rotational_mdot_boost_fac: f64,
    /// Kelvin-Helmholtz mass-loss cap multiplier for rotational boost (`rotational_mdot_kh_fac`, `0.3` default).
    pub rotational_mdot_kh_fac: f64,
    /// Maximum implicit mass-loss redo iterations per step (`max_mdot_redo_cnt`, `0` default).
    pub max_mdot_redo_cnt: usize,
    /// Minimum timestep in years required to trigger rotational `mdot` redo (`min_years_dt_for_redo_mdot`, `0.0` default).
    pub min_years_dt_for_redo_mdot: f64,
    /// Relative convergence tolerance on critical-rotation mass loss (`surf_omega_div_omega_crit_tol`, `0.05` default).
    pub surf_omega_div_omega_crit_tol: f64,
    /// Multiplicative revision factor per `mdot` redo iteration (`mdot_revise_factor`, `1.1` default).
    pub mdot_revise_factor: f64,
    /// Additive fraction of initial $|\dot{M}|$ on odd `mdot` redo tries (`implicit_mdot_boost`, `0.1` default).
    pub implicit_mdot_boost: f64,
    /// Minimum optical depth for surface rotation averaging (`surf_avg_tau_min`, `1.0` default).
    pub surf_avg_tau_min: f64,
    /// Maximum optical depth for surface rotation averaging (`surf_avg_tau`, `100.0` default).
    pub surf_avg_tau: f64,
    /// Rotational mixing diffusivity scale factor (`am_D_mix_factor`, `0.0` default).
    pub am_d_mix_factor: f64,
    /// Suppress stellar wind while `rotation_flag` is false (`no_wind_if_no_rotation`).
    pub no_wind_if_no_rotation: bool,
    /// Divisor on $L$ for super-Eddington wind onset (`super_eddington_wind_Ledd_factor`, `1.0` default).
    pub super_eddington_wind_ledd_factor: f64,
    /// $L / L_{\mathrm{Edd}}$ threshold below which super-Eddington boost is off (`wind_boost_full_off_L_div_Ledd`).
    pub wind_boost_full_off_l_div_ledd: f64,
    /// $L / L_{\mathrm{Edd}}$ threshold above which super-Eddington boost is maximal (`wind_boost_full_on_L_div_Ledd`).
    pub wind_boost_full_on_l_div_ledd: f64,
    /// Maximum multiplicative super-Eddington wind boost (`super_eddington_wind_max_boost`, `1.0` default).
    pub super_eddington_wind_max_boost: f64,
    /// Upper stellar mass ($M_\odot$) halting accretion (`max_star_mass_for_gain`, `< 0` disabled).
    pub max_star_mass_for_gain: f64,
    /// Lower stellar mass ($M_\odot$) halting mass loss (`min_star_mass_for_loss`, `< 0` disabled).
    pub min_star_mass_for_loss: f64,
    /// Assign explicit specific angular momentum `accreted_material_j` to accreted mass (`use_accreted_material_j`).
    pub use_accreted_material_j: bool,
    /// Specific angular momentum ($\mathrm{cm^2\,s^{-1}}$) of accreted material (`accreted_material_j`).
    pub accreted_material_j: f64,
    /// Lower $q$ bound for Region I/II transition `k_below_const_q` (`min_q_for_k_below_const_q`).
    pub min_q_for_k_below_const_q: f64,
    /// Upper $q$ bound for Region I/II transition `k_below_const_q` (`max_q_for_k_below_const_q`).
    pub max_q_for_k_below_const_q: f64,
    /// Maximum $\log_{10} T$ allowed for Region I/II transition `k_below_const_q` (`max_logT_for_k_below_const_q`).
    pub max_logt_for_k_below_const_q: f64,
    /// Lower $q$ bound for Region II/III transition `k_const_mass` (`min_q_for_k_const_mass`).
    pub min_q_for_k_const_mass: f64,
    /// Upper $q$ bound for Region II/III transition `k_const_mass` (`max_q_for_k_const_mass`).
    pub max_q_for_k_const_mass: f64,
    /// Maximum $\log_{10} T$ allowed for Region II/III transition `k_const_mass` (`max_logT_for_k_const_mass`).
    pub max_logt_for_k_const_mass: f64,
    /// Multiplicative factor on `eps_mdot` (`eps_mdot_factor`, `1.0` default).
    pub eps_mdot_factor: f64,
    /// Multiplicative factor on radiative leakage in `eps_mdot` (`eps_mdot_leak_frac_factor`, `1.0` default).
    pub eps_mdot_leak_frac_factor: f64,
    /// Multiplicative factor on WD $^{22}\mathrm{Ne}$ sedimentation heating (`eps_WD_sedimentation_factor`).
    pub eps_wd_sedimentation_factor: f64,
    /// Uniform extra specific power across the entire model ($\mathrm{erg\,g^{-1}\,s^{-1}}$, `extra_power_source`).
    pub extra_power_source: f64,
    /// Uniform extra specific heat in $[q_{\min}, q_{\max}]$ ($\mathrm{erg\,g^{-1}\,s^{-1}}$, `inject_uniform_extra_heat`).
    pub inject_uniform_extra_heat: f64,
    /// Inner mass-fraction boundary for `inject_uniform_extra_heat` (`min_q_for_uniform_extra_heat`).
    pub min_q_for_uniform_extra_heat: f64,
    /// Outer mass-fraction boundary for `inject_uniform_extra_heat` (`max_q_for_uniform_extra_heat`).
    pub max_q_for_uniform_extra_heat: f64,

    // --- Adaptive Mesh Spacing & Split-Merge AMR (`controls.defaults:5756-6744`) ---
    /// Maximum allowed number of grid zones (`max_allowed_nz`, `8000` default; `<= 0` unlimited).
    pub max_allowed_nz: isize,
    /// Disable remeshing if any cell has $\log_{10} T$ above this ceiling (`remesh_max_allowed_logT`, `1e99` default).
    pub remesh_max_allowed_logt: f64,
    /// Exponent on $P_{\mathrm{gas}} / P_{\mathrm{eos}}$ applied to `delta_gval_max` (`mesh_Pgas_div_P_exponent`, `0.0` default).
    pub mesh_pgas_div_p_exponent: f64,
    /// High-temperature `mesh_delta_coeff` value (`mesh_delta_coeff_for_highT`, `3.0` default).
    pub mesh_delta_coeff_for_hight: f64,
    /// Central $\log_{10} T$ ceiling for standard `mesh_delta_coeff` (`logT_max_for_standard_mesh_delta_coeff`, `9.0` default).
    pub logt_max_for_standard_mesh_delta_coeff: f64,
    /// Central $\log_{10} T$ floor for `mesh_delta_coeff_for_highT` (`logT_min_for_highT_mesh_delta_coeff`, `9.5` default).
    pub logt_min_for_hight_mesh_delta_coeff: f64,
    /// Maximum relative cell mass fraction $\Delta q_{\max}$ (`max_dq`, `1e-2` default).
    pub max_dq: f64,
    /// Minimum relative cell mass fraction $\Delta q_{\min}$ (`min_dq`, `1e-14` default).
    pub min_dq: f64,
    /// Minimum cell $\Delta q$ eligible for splitting (`min_dq_for_split`, `1e-14` default).
    pub min_dq_for_split: f64,
    /// Minimum cell $\Delta q$ for composition-driven splitting (`min_dq_for_xa`, `1e-14` default).
    pub min_dq_for_xa: f64,
    /// Minimum cell $\Delta q$ for composition-driven splitting in convective zones (`min_dq_for_xa_convective`, `1e-6` default).
    pub min_dq_for_xa_convective: f64,
    /// Minimum cell $\Delta q$ for $\log_{10} T$-driven splitting (`min_dq_for_logT`, `1e-14` default).
    pub min_dq_for_logt: f64,
    /// Minimum $\Delta\ln R$ across a cell during mesh refinement (`mesh_min_dlnR`, `1e-9` default).
    pub mesh_min_dlnr: f64,
    /// Force cell merge when $\Delta\ln R < \text{mesh\_min\_dlnR}$ (`merge_if_dlnR_too_small`, `false` default).
    pub merge_if_dlnr_too_small: bool,
    /// Minimum $\Delta r / (R_* - R_{\mathrm{center}})$ for splitting (`mesh_min_dr_div_dRstar`, `-1.0` default).
    pub mesh_min_dr_div_drstar: f64,
    /// Force cell merge when $\Delta r / \Delta R_* < \text{mesh\_min\_dr\_div\_dRstar}$ (`merge_if_dr_div_dRstar_too_small`, `true` default).
    pub merge_if_dr_div_drstar_too_small: bool,
    /// Minimum sound-crossing time $\Delta r / c_s$ in seconds for splitting (`mesh_min_dr_div_cs`, `-1.0` default).
    pub mesh_min_dr_div_cs: f64,
    /// Force cell merge when $\Delta r / c_s < \text{mesh\_min\_dr\_div\_cs}$ (`merge_if_dr_div_cs_too_small`, `true` default).
    pub merge_if_dr_div_cs_too_small: bool,
    /// Maximum allowed adjacent cell $\Delta q$ ratio (`mesh_max_allowed_ratio`, `2.5` default).
    pub mesh_max_allowed_ratio: f64,
    /// Maximum composition jump allowed when merging adjacent cells (`max_delta_x_for_merge`, `0.1` default).
    pub max_delta_x_for_merge: f64,
    /// Maximum surface cell mass fraction (`max_surface_cell_dq`, `1e-12` default).
    pub max_surface_cell_dq: f64,
    /// Minimum surface cell mass fraction (`min_surface_cell_dq`, `1e-99` default).
    pub min_surface_cell_dq: f64,
    /// Maximum center cell mass fraction (`max_center_cell_dq`, `1e-7` default).
    pub max_center_cell_dq: f64,
    /// Candidate subcells per old zone in `round_off_xq` (`max_num_subcells`, `2` default).
    pub max_num_subcells: usize,
    /// Maximum number of old cells merged into one interior zone (`max_num_merge_cells`, `2` default).
    pub max_num_merge_cells: usize,
    /// Maximum number of old cells merged into the surface zone (`max_num_merge_surface_cells`, `2` default).
    pub max_num_merge_surface_cells: usize,
    /// Invert EOS $\ln T(\rho, e)$ during remapping (`mesh_adjust_get_T_from_E`, `true` default).
    pub mesh_adjust_get_t_from_e: bool,
    /// Permit cell merges during `do_mesh_plan` (`mesh_ok_to_merge`, `true` default).
    pub mesh_ok_to_merge: bool,
    /// Maximum 1-based old cell index eligible for split/merge (`mesh_max_k_old_for_split`, `999_999_999` default).
    pub mesh_max_k_old_for_split: isize,
    /// Minimum 1-based old cell index eligible for split/merge (`mesh_min_k_old_for_split`, `0` default).
    pub mesh_min_k_old_for_split: isize,
    /// Internal energy mesh function weight (`E_function_weight`, `0.0` default).
    pub e_function_weight: f64,
    /// Floor on $\log_{10} e$ in `E_function` (`E_function_param`, `16.0` default).
    pub e_function_param: f64,
    /// Pressure gradient weight in mesh spacing function (`P_function_weight`, `40.0` default).
    pub p_function_weight: f64,
    /// Primary temperature gradient weight in mesh spacing function (`T_function1_weight`, `110.0` default).
    pub t_function1_weight: f64,
    /// Envelope ionization temperature mesh function weight (`T_function2_weight`, `0.0` default).
    pub t_function2_weight: f64,
    /// Transition temperature in K for `T_function2` (`T_function2_param`, `2e4` default).
    pub t_function2_param: f64,
    /// Radius gradient weight in mesh spacing function (`R_function_weight`, `0.0` default).
    pub r_function_weight: f64,
    /// Scale parameter in $R_\odot$ for `R_function` (`R_function_param`, `1e-4` default).
    pub r_function_param: f64,
    /// Clamped fractional radius mesh function weight (`R_function2_weight`, `0.0` default).
    pub r_function2_weight: f64,
    /// Upper $r / R_*$ clamp for `R_function2` (`R_function2_param1`, `0.4` default).
    pub r_function2_param1: f64,
    /// Lower $r / R_*$ clamp for `R_function2` (`R_function2_param2`, `0.0` default).
    pub r_function2_param2: f64,
    /// Linear $r / R_*$ mesh function weight (`R_function3_weight`, `0.0` default).
    pub r_function3_weight: f64,
    /// Mass gradient weight in mesh spacing function (`M_function_weight`, `0.0` default).
    pub m_function_weight: f64,
    /// Mass transition scale parameter in mesh spacing function (`M_function_param`, `1e-6` default).
    pub m_function_param: f64,
    /// Temperature-gradient weight in mesh spacing function (`gradT_function_weight`, `0.0` default).
    pub gradt_function_weight: f64,
    /// Optical-depth $\log_{10}\tau$ mesh function weight (`log_tau_function_weight`, `0.0` default).
    pub log_tau_function_weight: f64,
    /// Opacity $\log_{10}\kappa$ mesh function weight (`log_kap_function_weight`, `0.0` default).
    pub log_kap_function_weight: f64,
    /// Angular-velocity gradient weight in mesh spacing function (`omega_function_weight`, `0.0` default).
    pub omega_function_weight: f64,
    /// Coulomb $\Gamma$ gradient weight in mesh spacing function (`gam_function_weight`, `0.0` default).
    pub gam_function_weight: f64,
    /// Center Coulomb $\Gamma_1$ for `gam_function` (`gam_function_param1`, `170.0` default).
    pub gam_function_param1: f64,
    /// Width Coulomb $\Gamma_2$ for `gam_function` (`gam_function_param2`, `20.0` default).
    pub gam_function_param2: f64,
    /// Species names for abundance mesh spacing functions (`xa_function_species(1..)`).
    pub xa_function_species: Vec<String>,
    /// Weights for abundance mesh spacing functions (`xa_function_weight(1..)`).
    pub xa_function_weight: Vec<f64>,
    /// Transition scales for abundance mesh spacing functions (`xa_function_param(1..)`).
    pub xa_function_param: Vec<f64>,
    /// Multiplier on `delta_coeff` when `xa_function_species(j)` is the heaviest net isotope (`xa_mesh_delta_coeff(1..)`).
    pub xa_mesh_delta_coeff: Vec<f64>,
    /// Smoothing iterations applied to `delta_gval_max` (`mesh_delta_coeff_factor_smooth_iters`, `3` default).
    pub mesh_delta_coeff_factor_smooth_iters: isize,
    /// Species names for $|\mathrm{d}\log X / \mathrm{d}\log P|$ refinement (`mesh_logX_species(1..)`).
    pub mesh_logx_species: Vec<String>,
    /// Minimum $\log_{10} X$ threshold for `dlogX/dlogP` refinement (`mesh_logX_min_for_extra(1..)`).
    pub mesh_logx_min_for_extra: Vec<f64>,
    /// Local `delta_gval_max` factor for steep `dlogX/dlogP` (`mesh_dlogX_dlogP_extra(1..)`).
    pub mesh_dlogx_dlogp_extra: Vec<f64>,
    /// $|\mathrm{d}\log X / \mathrm{d}\log P|$ above which full `extra` applies (`mesh_dlogX_dlogP_full_on(1..)`).
    pub mesh_dlogx_dlogp_full_on: Vec<f64>,
    /// $|\mathrm{d}\log X / \mathrm{d}\log P|$ below which no `extra` applies (`mesh_dlogX_dlogP_full_off(1..)`).
    pub mesh_dlogx_dlogp_full_off: Vec<f64>,
    /// Minimum $\log_{10}\epsilon_{\mathrm{cat}}$ for burning-gradient refinement (`mesh_dlog_eps_min_for_extra`, `-2.0` default).
    pub mesh_dlog_eps_min_for_extra: f64,
    /// $|\mathrm{d}\log\epsilon / \mathrm{d}\log P|$ above which full burning `extra` applies (`mesh_dlog_eps_dlogP_full_on`, `4.0` default).
    pub mesh_dlog_eps_dlogp_full_on: f64,
    /// $|\mathrm{d}\log\epsilon / \mathrm{d}\log P|$ below which no burning `extra` applies (`mesh_dlog_eps_dlogP_full_off`, `1.0` default).
    pub mesh_dlog_eps_dlogp_full_off: f64,
    /// Burning-gradient `delta_gval_max` factor for `pp` (`mesh_dlog_pp_dlogP_extra`, `0.25` default).
    pub mesh_dlog_pp_dlogp_extra: f64,
    /// Burning-gradient `delta_gval_max` factor for `cno` (`mesh_dlog_cno_dlogP_extra`, `0.25` default).
    pub mesh_dlog_cno_dlogp_extra: f64,
    /// Burning-gradient `delta_gval_max` factor for `tri_alpha` (`mesh_dlog_3alf_dlogP_extra`, `0.25` default).
    pub mesh_dlog_3alf_dlogp_extra: f64,
    /// Burning-gradient `delta_gval_max` factor for `c_alpha` (`mesh_dlog_burn_c_dlogP_extra`, `0.25` default).
    pub mesh_dlog_burn_c_dlogp_extra: f64,
    /// Burning-gradient `delta_gval_max` factor for `n_alpha` (`mesh_dlog_burn_n_dlogP_extra`, `0.25` default).
    pub mesh_dlog_burn_n_dlogp_extra: f64,
    /// Burning-gradient `delta_gval_max` factor for `o_alpha` (`mesh_dlog_burn_o_dlogP_extra`, `0.25` default).
    pub mesh_dlog_burn_o_dlogp_extra: f64,
    /// Burning-gradient `delta_gval_max` factor for `ne_alpha` (`mesh_dlog_burn_ne_dlogP_extra`, `0.25` default).
    pub mesh_dlog_burn_ne_dlogp_extra: f64,
    /// Burning-gradient `delta_gval_max` factor for `na_alpha` (`mesh_dlog_burn_na_dlogP_extra`, `0.25` default).
    pub mesh_dlog_burn_na_dlogp_extra: f64,
    /// Burning-gradient `delta_gval_max` factor for `mg_alpha` (`mesh_dlog_burn_mg_dlogP_extra`, `0.25` default).
    pub mesh_dlog_burn_mg_dlogp_extra: f64,
    /// Burning-gradient `delta_gval_max` factor for `c12_c12` (`mesh_dlog_cc_dlogP_extra`, `0.25` default).
    pub mesh_dlog_cc_dlogp_extra: f64,
    /// Burning-gradient `delta_gval_max` factor for `c12_o16` (`mesh_dlog_co_dlogP_extra`, `0.25` default).
    pub mesh_dlog_co_dlogp_extra: f64,
    /// Burning-gradient `delta_gval_max` factor for `o16_o16` (`mesh_dlog_oo_dlogP_extra`, `0.25` default).
    pub mesh_dlog_oo_dlogp_extra: f64,
    /// Burning-gradient `delta_gval_max` factor for `si_alpha` (`mesh_dlog_burn_si_dlogP_extra`, `0.25` default).
    pub mesh_dlog_burn_si_dlogp_extra: f64,
    /// Burning-gradient `delta_gval_max` factor for `s_alpha` (`mesh_dlog_burn_s_dlogP_extra`, `0.25` default).
    pub mesh_dlog_burn_s_dlogp_extra: f64,
    /// Burning-gradient `delta_gval_max` factor for `ar_alpha` (`mesh_dlog_burn_ar_dlogP_extra`, `0.25` default).
    pub mesh_dlog_burn_ar_dlogp_extra: f64,
    /// Burning-gradient `delta_gval_max` factor for `ca_alpha` (`mesh_dlog_burn_ca_dlogP_extra`, `0.25` default).
    pub mesh_dlog_burn_ca_dlogp_extra: f64,
    /// Burning-gradient `delta_gval_max` factor for `ti_alpha` (`mesh_dlog_burn_ti_dlogP_extra`, `0.25` default).
    pub mesh_dlog_burn_ti_dlogp_extra: f64,
    /// Burning-gradient `delta_gval_max` factor for `cr_alpha` (`mesh_dlog_burn_cr_dlogP_extra`, `0.25` default).
    pub mesh_dlog_burn_cr_dlogp_extra: f64,
    /// Burning-gradient `delta_gval_max` factor for `fe_co_ni` (`mesh_dlog_burn_fe_dlogP_extra`, `0.25` default).
    pub mesh_dlog_burn_fe_dlogp_extra: f64,
    /// Burning-gradient `delta_gval_max` factor for `pnhe4` (`mesh_dlog_pnhe4_dlogP_extra`, `0.25` default).
    pub mesh_dlog_pnhe4_dlogp_extra: f64,
    /// Burning-gradient `delta_gval_max` factor for `other` (`mesh_dlog_other_dlogP_extra`, `0.25` default).
    pub mesh_dlog_other_dlogp_extra: f64,
    /// Burning-gradient `delta_gval_max` factor for `photo` (`mesh_dlog_photo_dlogP_extra`, `1.0` default).
    pub mesh_dlog_photo_dlogp_extra: f64,
    /// Convective-boundary step mesh function weight (`convective_bdy_weight`, `0.0` default).
    pub convective_bdy_weight: f64,
    /// Minimum cell $\Delta q$ at convective boundary for `convective_bdy` (`convective_bdy_dq_limit`, `3e-5` default).
    pub convective_bdy_dq_limit: f64,
    /// Minimum timestep in years enabling `convective_bdy` (`convective_bdy_min_dt_yrs`, `1e-3` default).
    pub convective_bdy_min_dt_yrs: f64,
    /// Maximum relative cell internal-energy adjustment balancing $\Delta(\mathrm{PE} + \mathrm{KE})$ (`max_rel_delta_IE_for_mesh_total_energy_balance`, `0.05` default).
    pub max_rel_delta_ie_for_mesh_total_energy_balance: f64,
    /// Enforce strict IE/KE/Eturb conservation bounds during `do_mesh_adjust` (`trace_mesh_adjust_error_in_conservation`, `false` default).
    pub trace_mesh_adjust_error_in_conservation: bool,
    /// Undo remeshing on step retry (`restore_mesh_on_retry`, `false` default).
    pub restore_mesh_on_retry: bool,
    /// Steps to suppress remeshing after a retry when `restore_mesh_on_retry` is active (`num_steps_to_hold_mesh_after_retry`, `0` default).
    pub num_steps_to_hold_mesh_after_retry: isize,
    /// Minimum timestep in seconds required to execute adaptive remeshing (`remesh_dt_limit`, `-1.0` default).
    pub remesh_dt_limit: f64,
    /// Enable custom `other_mesh_functions` hook (`use_other_mesh_functions`, `false` default).
    pub use_other_mesh_functions: bool,
    /// Enable custom `other_mesh_delta_coeff_factor` hook (`use_other_mesh_delta_coeff_factor`, `false` default).
    pub use_other_mesh_delta_coeff_factor: bool,
    /// Enforce hydro velocity-jump merge limits inside `do_mesh_plan` (`use_hydro_merge_limits_in_mesh_plan`, `false` default).
    pub use_hydro_merge_limits_in_mesh_plan: bool,
    /// Use hydrodynamic `remesh_split_merge` instead of `remesh` (`use_split_merge_amr`, `false` default).
    pub use_split_merge_amr: bool,
    /// Target uniform $\ln\tau$ spacing in `remesh_split_merge` (`split_merge_amr_logtau_zoning`, `false` default).
    pub split_merge_amr_logtau_zoning: bool,
    /// Target uniform $\ln r$ spacing in `remesh_split_merge` (`split_merge_amr_log_zoning`, `true` default).
    pub split_merge_amr_log_zoning: bool,
    /// Target linear-$r$ core + $\ln r$ envelope spacing in `remesh_split_merge` (`split_merge_amr_hybrid_zoning`, `false` default).
    pub split_merge_amr_hybrid_zoning: bool,
    /// Target $\ln r$ core + linear-$r$ envelope spacing in `remesh_split_merge` (`split_merge_amr_flipped_hybrid_zoning`, `false` default; `#242`).
    pub split_merge_amr_flipped_hybrid_zoning: bool,
    /// Baseline target zone count for `remesh_split_merge` (`split_merge_amr_nz_baseline`, `1000` default).
    pub split_merge_amr_nz_baseline: isize,
    /// Target core zone count inside `r_core_cm` for hybrid AMR (`split_merge_amr_nz_r_core`, `0` default).
    pub split_merge_amr_nz_r_core: isize,
    /// Core radius fraction of $r_1 - R_{\mathrm{center}}$ when `split_merge_amr_r_core_cm <= 0` (`split_merge_amr_nz_r_core_fraction`, `0.0` default).
    pub split_merge_amr_nz_r_core_fraction: f64,
    /// Resolution divisor on `nz_baseline` and `nz_r_core` (`split_merge_amr_mesh_delta_coeff`, `1.0` default).
    pub split_merge_amr_mesh_delta_coeff: f64,
    /// Maximum actual/baseline cell-width ratio before splitting (`split_merge_amr_MaxLong`, `1.5` default).
    pub split_merge_amr_maxlong: f64,
    /// Maximum baseline/actual cell-width ratio before merging (`split_merge_amr_MaxShort`, `1.5` default).
    pub split_merge_amr_maxshort: f64,
    /// Maximum $|\Delta u| / c_s$ permitted for AMR merging (`merge_amr_max_abs_du_div_cs`, `0.1` default).
    pub merge_amr_max_abs_du_div_cs: f64,
    /// Apply `merge_amr_max_abs_du_div_cs` only on compressive interfaces (`merge_amr_du_div_cs_limit_only_for_compression`, `false` default).
    pub merge_amr_du_div_cs_limit_only_for_compression: bool,
    /// Soften rather than block merges across velocity jumps (`merge_amr_inhibit_at_jumps`, `false` default).
    pub merge_amr_inhibit_at_jumps: bool,
    /// Skip outermost `merge_amr_k_for_ignore_surface_cells` zones for AMR merging (`merge_amr_ignore_surface_cells`, `true` default).
    pub merge_amr_ignore_surface_cells: bool,
    /// Count of outermost zones skipped when `merge_amr_ignore_surface_cells` is active (`merge_amr_k_for_ignore_surface_cells`, `2` default).
    pub merge_amr_k_for_ignore_surface_cells: isize,
    /// Skip high-temperature core zones for AMR merging (`merge_amr_ignore_core_cells`, `false` default).
    pub merge_amr_ignore_core_cells: bool,
    /// $\log_{10} T$ threshold for `merge_amr_ignore_core_cells` (`merge_amr_logT_for_ignore_core_cells`, `1e99` default).
    pub merge_amr_logt_for_ignore_core_cells: f64,
    /// Skip high-temperature core zones for AMR splitting (`split_amr_ignore_core_cells`, `false` default).
    pub split_amr_ignore_core_cells: bool,
    /// $\log_{10} T$ threshold for `split_amr_ignore_core_cells` (`split_amr_logT_for_ignore_core_cells`, `1e99` default).
    pub split_amr_logt_for_ignore_core_cells: f64,
    /// Prevent touching a zone more than once per `remesh_split_merge` pass (`split_merge_amr_avoid_repeated_remesh`, `false` default).
    pub split_merge_amr_avoid_repeated_remesh: bool,
    /// Minimum cell $\Delta q$ in `remesh_split_merge` (`split_merge_amr_dq_min`, `1e-14` default).
    pub split_merge_amr_dq_min: f64,
    /// Maximum cell $\Delta q$ in `remesh_split_merge` (`split_merge_amr_dq_max`, `1.0` default).
    pub split_merge_amr_dq_max: f64,
    /// Hybrid AMR core transition radius in cm (`split_merge_amr_r_core_cm`, `1e8` default).
    pub split_merge_amr_r_core_cm: f64,
    /// Maximum split/merge iterations per `remesh_split_merge` call (`split_merge_amr_max_iters`, `100` default).
    pub split_merge_amr_max_iters: usize,
    /// Permit splitting surface zone `k = 1` in `remesh_split_merge` (`split_merge_amr_okay_to_split_1`, `true` default).
    pub split_merge_amr_okay_to_split_1: bool,
    /// Permit splitting center zone `k = nz` in `remesh_split_merge` (`split_merge_amr_okay_to_split_nz`, `true` default).
    pub split_merge_amr_okay_to_split_nz: bool,
    /// Use uniform density across subcells when splitting in `remesh_split_merge` (`equal_split_density_amr`, `false` default).
    pub equal_split_density_amr: bool,
    /// Absolute mass-loss tolerance in $M_\odot\,\mathrm{yr}^{-1}$ for `check_delta_mdot` (`delta_mdot_atol`, `1e-3` default).
    pub delta_mdot_atol: f64,
    /// Relative mass-loss tolerance for `check_delta_mdot` (`delta_mdot_rtol`, `0.5` default).
    pub delta_mdot_rtol: f64,
    /// Soft timestep limit on normalized $|\Delta\dot{M}|$ (`delta_mdot_limit`, `-1.0` default).
    pub delta_mdot_limit: f64,
    /// Hard retry limit on normalized $|\Delta\dot{M}|$ (`delta_mdot_hard_limit`, `-1.0` default).
    pub delta_mdot_hard_limit: f64,
    /// Upper cap on shellular $\Omega / \Omega_{\mathrm{crit,Roche}}$ in distortion factors (`w_div_wcrit_max`, `0.89` default).
    pub w_div_wcrit_max: f64,
    /// Mean-molecular-weight barrier scale factor in rotational instabilities (`am_gradmu_factor`, `0.05` default).
    pub am_gradmu_factor: f64,
    /// Dynamical shear instability scale factor (`D_DSI_factor`, `0.0` default).
    pub d_dsi_factor: f64,
    /// Solberg-Hoiland instability scale factor (`D_SH_factor`, `0.0` default).
    pub d_sh_factor: f64,
    /// Secular shear instability scale factor (`D_SSI_factor`, `0.0` default).
    pub d_ssi_factor: f64,
    /// Eddington-Sweet circulation scale factor (`D_ES_factor`, `0.0` default).
    pub d_es_factor: f64,
    /// Goldreich-Schubert-Fricke instability scale factor (`D_GSF_factor`, `0.0` default).
    pub d_gsf_factor: f64,
    /// Spruit-Tayler magnetic compositional diffusivity scale factor (`D_ST_factor`, `0.0` default).
    pub d_st_factor: f64,
    /// Kinematic shear viscosity compositional diffusivity scale factor (`D_visc_factor`, `0.0` default).
    pub d_visc_factor: f64,
    /// Overall rotational angular-momentum viscosity multiplier (`am_nu_factor`, `1.0` default).
    pub am_nu_factor: f64,
    /// Non-rotational (`MLT`/overshoot) angular-momentum viscosity factor (`am_nu_non_rotation_factor`, `1.0` default).
    pub am_nu_non_rotation_factor: f64,
    /// Dynamical shear angular-momentum viscosity factor (`am_nu_DSI_factor`, `-1.0` default).
    pub am_nu_dsi_factor: f64,
    /// Solberg-Hoiland angular-momentum viscosity factor (`am_nu_SH_factor`, `-1.0` default).
    pub am_nu_sh_factor: f64,
    /// Secular shear angular-momentum viscosity factor (`am_nu_SSI_factor`, `-1.0` default).
    pub am_nu_ssi_factor: f64,
    /// Eddington-Sweet angular-momentum viscosity factor (`am_nu_ES_factor`, `-1.0` default).
    pub am_nu_es_factor: f64,
    /// Goldreich-Schubert-Fricke angular-momentum viscosity factor (`am_nu_GSF_factor`, `-1.0` default).
    pub am_nu_gsf_factor: f64,
    /// Spruit-Tayler magnetic angular-momentum viscosity factor (`am_nu_ST_factor`, `-1.0` default).
    pub am_nu_st_factor: f64,
    /// Kinematic shear angular-momentum viscosity factor (`am_nu_visc_factor`, `1.0` default).
    pub am_nu_visc_factor: f64,
    /// Dimensionless tridiagonal stiffness cap for `solve_omega_mix` (`am_sig_term_limit`, `1e13` default).
    pub am_sig_term_limit: f64,
    /// Strang-split half-step angular-momentum transport before and after Henyey (`premix_omega`, `true` default).
    pub premix_omega: bool,
    /// Remove surface angular momentum deficit after mass loss (`do_adjust_J_lost`, `true` default).
    pub do_adjust_j_lost: bool,
    /// Fraction of surface angular momentum removed by mass loss (`adjust_J_fraction`, `1.0` default).
    pub adjust_j_fraction: f64,
    /// Upper $q$ threshold below which `adjust_J_lost` stops (`min_q_for_adjust_J_lost`, `0.995` default).
    pub min_q_for_adjust_j_lost: f64,
    /// Minimum ratio $J_{\mathrm{env}} / |\Delta J|$ for `adjust_J_lost` (`min_J_div_delta_J`, `3.0` default).
    pub min_j_div_delta_j: f64,
    /// Zero rotational instability diffusivities inside convective zones (`skip_rotation_in_convection_zones`, `false` default).
    pub skip_rotation_in_convection_zones: bool,
    /// Half-width cell count for triangular smoothing of $\nabla_L$ composition term (`num_cells_for_smooth_gradL_composition_term`, `3` default).
    pub num_cells_for_smooth_gradl_composition_term: usize,
    /// Half-width cell count for triangular smoothing of Brunt $B$ (`num_cells_for_smooth_brunt_B`, `2` default).
    pub num_cells_for_smooth_brunt_b: usize,

    // --- Upstream Wind Gates & RLO Wind (`winds.f90:104-285, 661-703`) ---
    /// Timestep in seconds above which wind scheme mass loss is at full strength (`mass_change_full_on_dt`, `1e-99` default).
    pub mass_change_full_on_dt: f64,
    /// Timestep in seconds below which wind scheme mass loss is zero (`mass_change_full_off_dt`, `1e-99` default).
    pub mass_change_full_off_dt: f64,
    /// Central temperature in K above which wind mass loss is zero (`max_T_center_for_any_mass_loss`, `2e9` default).
    pub max_t_center_for_any_mass_loss: f64,
    /// Central temperature in K below which wind mass loss is unattenuated (`max_T_center_for_full_mass_loss`, `1e9` default).
    pub max_t_center_for_full_mass_loss: f64,
    /// Minimum H-rich envelope mass in $M_\odot$ for wind loss (`wind_H_envelope_limit`, `-1` disabled).
    pub wind_h_envelope_limit: f64,
    /// Minimum H+He envelope mass in $M_\odot$ for wind loss (`wind_H_He_envelope_limit`, `-1` disabled).
    pub wind_h_he_envelope_limit: f64,
    /// Minimum He-layer mass in $M_\odot$ for wind loss (`wind_He_layer_limit`, `-1` disabled).
    pub wind_he_layer_limit: f64,
    /// Scaling factor for single-star RLO wind (`rlo_scaling_factor`, `0` default).
    pub rlo_scaling_factor: f64,
    /// Minimum luminosity in $L_\odot$ for RLO wind (`rlo_wind_min_L`, `1e-6` default).
    pub rlo_wind_min_l: f64,
    /// Maximum $T_{\mathrm{eff}}$ in K for RLO wind (`rlo_wind_max_Teff`, `1e99` default).
    pub rlo_wind_max_teff: f64,
    /// Roche-lobe radius in $R_\odot$ for RLO wind (`rlo_wind_roche_lobe_radius`, `0.40` default).
    pub rlo_wind_roche_lobe_radius: f64,
    /// Base RLO wind mass-loss rate in $M_\odot\,\mathrm{yr}^{-1}$ (`rlo_wind_base_mdot`, `1e-3` default).
    pub rlo_wind_base_mdot: f64,
    /// Scale height in $R_\odot$ for exponential RLO wind (`rlo_wind_scale_height`, `0.1` default; `<= 0` uses surface pressure scale height).
    pub rlo_wind_scale_height: f64,
    /// Radius-to-Roche-lobe ratio below which accretion transfer is unattenuated (`roche_lobe_xfer_full_on`, `0.5` default).
    pub roche_lobe_xfer_full_on: f64,
    /// Radius-to-Roche-lobe ratio above which accretion transfer is shut off (`roche_lobe_xfer_full_off`, `1.0` default).
    pub roche_lobe_xfer_full_off: f64,

    // --- Accretion Composition & Outer Composition Smoothing (`adjust_xyz.f90`, `adjust_mass.f90`) ---
    /// Number of explicitly specified accretion species (`num_accretion_species`, `0` default).
    pub num_accretion_species: usize,
    /// Accreted $^2\mathrm{H}$ mass fraction when `accretion_h1 > 0` (`accretion_h2`, `0` default).
    pub accretion_h2: f64,
    /// Accreted $^3\mathrm{He}$ mass fraction when `accretion_he4 > 0` (`accretion_he3`, `0` default).
    pub accretion_he3: f64,
    /// Standard metal-distribution selector for accretion (`accretion_zfracs`, `-1` errors loudly on nonzero metals per `adjust_xyz.f90:1069-1079`, `0` uses `z_fraction_*`, `1..=10` uses solar scale).
    pub accretion_zfracs: i32,
    /// Dump unrepresented metal mass fractions into the heaviest net species (`accretion_dump_missing_metals_into_heaviest`, `true` default).
    pub accretion_dump_missing_metals_into_heaviest: bool,
    /// Relative Li mass fraction in metals for `accretion_zfracs == 0` (`z_fraction_li`).
    pub z_fraction_li: f64,
    /// Relative Be mass fraction in metals (`z_fraction_be`).
    pub z_fraction_be: f64,
    /// Relative B mass fraction in metals (`z_fraction_b`).
    pub z_fraction_b: f64,
    /// Relative C mass fraction in metals (`z_fraction_c`).
    pub z_fraction_c: f64,
    /// Relative N mass fraction in metals (`z_fraction_n`).
    pub z_fraction_n: f64,
    /// Relative O mass fraction in metals (`z_fraction_o`).
    pub z_fraction_o: f64,
    /// Relative F mass fraction in metals (`z_fraction_f`).
    pub z_fraction_f: f64,
    /// Relative Ne mass fraction in metals (`z_fraction_ne`).
    pub z_fraction_ne: f64,
    /// Relative Na mass fraction in metals (`z_fraction_na`).
    pub z_fraction_na: f64,
    /// Relative Mg mass fraction in metals (`z_fraction_mg`).
    pub z_fraction_mg: f64,
    /// Relative Al mass fraction in metals (`z_fraction_al`).
    pub z_fraction_al: f64,
    /// Relative Si mass fraction in metals (`z_fraction_si`).
    pub z_fraction_si: f64,
    /// Relative P mass fraction in metals (`z_fraction_p`).
    pub z_fraction_p: f64,
    /// Relative S mass fraction in metals (`z_fraction_s`).
    pub z_fraction_s: f64,
    /// Relative Cl mass fraction in metals (`z_fraction_cl`).
    pub z_fraction_cl: f64,
    /// Relative Ar mass fraction in metals (`z_fraction_ar`).
    pub z_fraction_ar: f64,
    /// Relative K mass fraction in metals (`z_fraction_k`).
    pub z_fraction_k: f64,
    /// Relative Ca mass fraction in metals (`z_fraction_ca`).
    pub z_fraction_ca: f64,
    /// Relative Sc mass fraction in metals (`z_fraction_sc`).
    pub z_fraction_sc: f64,
    /// Relative Ti mass fraction in metals (`z_fraction_ti`).
    pub z_fraction_ti: f64,
    /// Relative V mass fraction in metals (`z_fraction_v`).
    pub z_fraction_v: f64,
    /// Relative Cr mass fraction in metals (`z_fraction_cr`).
    pub z_fraction_cr: f64,
    /// Relative Mn mass fraction in metals (`z_fraction_mn`).
    pub z_fraction_mn: f64,
    /// Relative Fe mass fraction in metals (`z_fraction_fe`).
    pub z_fraction_fe: f64,
    /// Relative Co mass fraction in metals (`z_fraction_co`).
    pub z_fraction_co: f64,
    /// Relative Ni mass fraction in metals (`z_fraction_ni`).
    pub z_fraction_ni: f64,
    /// Relative Cu mass fraction in metals (`z_fraction_cu`).
    pub z_fraction_cu: f64,
    /// Relative Zn mass fraction in metals (`z_fraction_zn`).
    pub z_fraction_zn: f64,
    /// Outer mass fraction smoothed during mass change (`smooth_outer_xa_big`, `-1` disabled).
    pub smooth_outer_xa_big: f64,
    /// Inner smoothing boundary mass fraction during mass change (`smooth_outer_xa_small`, `-1` disabled).
    pub smooth_outer_xa_small: f64,
}

impl Default for MixWindControls {
    fn default() -> Self {
        Self {
            overshoot_sets: std::array::from_fn(|_| {
                OvershootParamSet::mesa_default()
            }),
            overshoot_d_min: 1.0e-2,
            overshoot_brunt_b_max: 0.0,
            min_overshoot_q: 0.0,
            overshoot_alpha: -1.0,
            limit_overshoot_hp_using_size_of_convection_zone: true,
            burn_z_mix_region_logt: 8.7,
            burn_he_mix_region_logt: 7.7,
            burn_h_mix_region_logt: 6.7,
            max_y_for_burn_z_mix_region: 1.0e-4,
            max_x_for_burn_he_mix_region: 1.0e-4,
            peclet_function_full_on: 0.0,
            peclet_function_full_off: 0.0,
            predictive_sets: std::array::from_fn(|_| {
                PredictiveParamSet::default()
            }),
            conv_premix_avoid_increase: false,
            conv_premix_time_factor: 0.0,
            conv_premix_fix_pgas: true,
            make_gradr_sticky_in_solver_iters: false,
            min_logt_for_make_gradr_sticky_in_solver_iters: 1.0e99,
            set_min_d_mix: false,
            mass_lower_limit_for_min_d_mix: 0.0,
            mass_upper_limit_for_min_d_mix: 1.0e99,
            min_center_ye_for_min_d_mix: 0.47,
            okay_to_remove_mixing_singleton: true,
            clip_d_limit: 0.0,
            min_convective_gap: -1.0,
            min_thermohaline_gap: -1.0,
            min_thermohaline_dropout: -1.0,
            max_dropout_gradl_sub_grada: 1.0e-3,
            min_semiconvection_gap: -1.0,
            remove_embedded_semiconvection: false,
            hot_wind_scheme: String::new(),
            rgb_to_agb_wind_switch: 1.0e-4,
            dutch_wind_lowt_scheme: String::from("de Jager"),
            bjorklund_scaling_factor: 0.0,
            de_jager_scaling_factor: 0.0,
            van_loon_scaling_factor: 0.0,
            nieuwenhuijzen_scaling_factor: 0.0,
            min_wind: 0.0,
            max_wind: 0.0,
            mdot_omega_power: 0.43,
            max_rotational_mdot_boost: 1.0e4,
            max_mdot_jump_for_rotation: 2.0,
            rotational_mdot_boost_fac: 1.0e5,
            rotational_mdot_kh_fac: 0.3,
            max_mdot_redo_cnt: 0,
            min_years_dt_for_redo_mdot: 0.0,
            surf_omega_div_omega_crit_tol: 0.05,
            mdot_revise_factor: 1.1,
            implicit_mdot_boost: 0.1,
            surf_avg_tau_min: 1.0,
            surf_avg_tau: 100.0,
            am_d_mix_factor: 0.0,
            no_wind_if_no_rotation: false,
            super_eddington_wind_ledd_factor: 1.0,
            wind_boost_full_off_l_div_ledd: 1.5,
            wind_boost_full_on_l_div_ledd: 5.0,
            super_eddington_wind_max_boost: 1.0,
            max_star_mass_for_gain: -1.0,
            min_star_mass_for_loss: -1.0,
            use_accreted_material_j: false,
            accreted_material_j: 0.0,
            min_q_for_k_below_const_q: 0.999,
            max_q_for_k_below_const_q: 1.0,
            max_logt_for_k_below_const_q: 5.0,
            min_q_for_k_const_mass: 0.995,
            max_q_for_k_const_mass: 1.0,
            max_logt_for_k_const_mass: 6.0,
            eps_mdot_factor: 1.0,
            eps_mdot_leak_frac_factor: 1.0,
            eps_wd_sedimentation_factor: 1.0,
            extra_power_source: 0.0,
            inject_uniform_extra_heat: 0.0,
            min_q_for_uniform_extra_heat: 0.0,
            max_q_for_uniform_extra_heat: 1.0,
            max_allowed_nz: 8000,
            remesh_max_allowed_logt: 1.0e99,
            mesh_pgas_div_p_exponent: 0.0,
            mesh_delta_coeff_for_hight: 3.0,
            logt_max_for_standard_mesh_delta_coeff: 9.0,
            logt_min_for_hight_mesh_delta_coeff: 9.5,
            max_dq: 1.0e-2,
            min_dq: 1.0e-14,
            min_dq_for_split: 1.0e-14,
            min_dq_for_xa: 1.0e-14,
            min_dq_for_xa_convective: 1.0e-6,
            min_dq_for_logt: 1.0e-14,
            mesh_min_dlnr: 1.0e-9,
            merge_if_dlnr_too_small: false,
            mesh_min_dr_div_drstar: -1.0,
            merge_if_dr_div_drstar_too_small: true,
            mesh_min_dr_div_cs: -1.0,
            merge_if_dr_div_cs_too_small: true,
            mesh_max_allowed_ratio: 2.5,
            max_delta_x_for_merge: 0.1,
            max_surface_cell_dq: 1.0e-12,
            min_surface_cell_dq: 1.0e-99,
            max_center_cell_dq: 1.0e-7,
            max_num_subcells: 2,
            max_num_merge_cells: 2,
            max_num_merge_surface_cells: 2,
            mesh_adjust_get_t_from_e: true,
            mesh_ok_to_merge: true,
            mesh_max_k_old_for_split: 999_999_999,
            mesh_min_k_old_for_split: 0,
            e_function_weight: 0.0,
            e_function_param: 16.0,
            p_function_weight: 40.0,
            t_function1_weight: 110.0,
            t_function2_weight: 0.0,
            t_function2_param: 2.0e4,
            r_function_weight: 0.0,
            r_function_param: 1.0e-4,
            r_function2_weight: 0.0,
            r_function2_param1: 0.4,
            r_function2_param2: 0.0,
            r_function3_weight: 0.0,
            m_function_weight: 0.0,
            m_function_param: 1.0e-6,
            gradt_function_weight: 0.0,
            log_tau_function_weight: 0.0,
            log_kap_function_weight: 0.0,
            omega_function_weight: 0.0,
            gam_function_weight: 0.0,
            gam_function_param1: 170.0,
            gam_function_param2: 20.0,
            xa_function_species: vec![String::from("he4")],
            xa_function_weight: vec![30.0],
            xa_function_param: vec![1.0e-2],
            xa_mesh_delta_coeff: vec![1.0],
            mesh_delta_coeff_factor_smooth_iters: 3,
            mesh_logx_species: vec![String::new()],
            mesh_logx_min_for_extra: vec![-6.0],
            mesh_dlogx_dlogp_extra: vec![1.0],
            mesh_dlogx_dlogp_full_on: vec![2.0],
            mesh_dlogx_dlogp_full_off: vec![1.0],
            mesh_dlog_eps_min_for_extra: -2.0,
            mesh_dlog_eps_dlogp_full_on: 4.0,
            mesh_dlog_eps_dlogp_full_off: 1.0,
            mesh_dlog_pp_dlogp_extra: 0.25,
            mesh_dlog_cno_dlogp_extra: 0.25,
            mesh_dlog_3alf_dlogp_extra: 0.25,
            mesh_dlog_burn_c_dlogp_extra: 0.25,
            mesh_dlog_burn_n_dlogp_extra: 0.25,
            mesh_dlog_burn_o_dlogp_extra: 0.25,
            mesh_dlog_burn_ne_dlogp_extra: 0.25,
            mesh_dlog_burn_na_dlogp_extra: 0.25,
            mesh_dlog_burn_mg_dlogp_extra: 0.25,
            mesh_dlog_cc_dlogp_extra: 0.25,
            mesh_dlog_co_dlogp_extra: 0.25,
            mesh_dlog_oo_dlogp_extra: 0.25,
            mesh_dlog_burn_si_dlogp_extra: 0.25,
            mesh_dlog_burn_s_dlogp_extra: 0.25,
            mesh_dlog_burn_ar_dlogp_extra: 0.25,
            mesh_dlog_burn_ca_dlogp_extra: 0.25,
            mesh_dlog_burn_ti_dlogp_extra: 0.25,
            mesh_dlog_burn_cr_dlogp_extra: 0.25,
            mesh_dlog_burn_fe_dlogp_extra: 0.25,
            mesh_dlog_pnhe4_dlogp_extra: 0.25,
            mesh_dlog_other_dlogp_extra: 0.25,
            mesh_dlog_photo_dlogp_extra: 1.0,
            convective_bdy_weight: 0.0,
            convective_bdy_dq_limit: 3.0e-5,
            convective_bdy_min_dt_yrs: 1.0e-3,
            max_rel_delta_ie_for_mesh_total_energy_balance: 0.05,
            trace_mesh_adjust_error_in_conservation: false,
            restore_mesh_on_retry: false,
            num_steps_to_hold_mesh_after_retry: 0,
            remesh_dt_limit: -1.0,
            use_other_mesh_functions: false,
            use_other_mesh_delta_coeff_factor: false,
            use_hydro_merge_limits_in_mesh_plan: false,
            use_split_merge_amr: false,
            split_merge_amr_logtau_zoning: false,
            split_merge_amr_log_zoning: true,
            split_merge_amr_hybrid_zoning: false,
            split_merge_amr_flipped_hybrid_zoning: false,
            split_merge_amr_nz_baseline: 1000,
            split_merge_amr_nz_r_core: 0,
            split_merge_amr_nz_r_core_fraction: 0.0,
            split_merge_amr_mesh_delta_coeff: 1.0,
            split_merge_amr_maxlong: 1.5,
            split_merge_amr_maxshort: 1.5,
            merge_amr_max_abs_du_div_cs: 0.1,
            merge_amr_du_div_cs_limit_only_for_compression: false,
            merge_amr_inhibit_at_jumps: false,
            merge_amr_ignore_surface_cells: true,
            merge_amr_k_for_ignore_surface_cells: 2,
            merge_amr_ignore_core_cells: false,
            merge_amr_logt_for_ignore_core_cells: 1.0e99,
            split_amr_ignore_core_cells: false,
            split_amr_logt_for_ignore_core_cells: 1.0e99,
            split_merge_amr_avoid_repeated_remesh: false,
            split_merge_amr_dq_min: 1.0e-14,
            split_merge_amr_dq_max: 1.0,
            split_merge_amr_r_core_cm: 1.0e8,
            split_merge_amr_max_iters: 100,
            split_merge_amr_okay_to_split_1: true,
            split_merge_amr_okay_to_split_nz: true,
            equal_split_density_amr: false,
            delta_mdot_atol: 1.0e-3,
            delta_mdot_rtol: 0.5,
            delta_mdot_limit: -1.0,
            delta_mdot_hard_limit: -1.0,
            w_div_wcrit_max: 0.89,
            am_gradmu_factor: 0.05,
            d_dsi_factor: 0.0,
            d_sh_factor: 0.0,
            d_ssi_factor: 0.0,
            d_es_factor: 0.0,
            d_gsf_factor: 0.0,
            d_st_factor: 0.0,
            d_visc_factor: 0.0,
            am_nu_factor: 1.0,
            am_nu_non_rotation_factor: 1.0,
            am_nu_dsi_factor: -1.0,
            am_nu_sh_factor: -1.0,
            am_nu_ssi_factor: -1.0,
            am_nu_es_factor: -1.0,
            am_nu_gsf_factor: -1.0,
            am_nu_st_factor: -1.0,
            am_nu_visc_factor: 1.0,
            am_sig_term_limit: 1.0e13,
            premix_omega: true,
            do_adjust_j_lost: true,
            adjust_j_fraction: 1.0,
            min_q_for_adjust_j_lost: 0.995,
            min_j_div_delta_j: 3.0,
            skip_rotation_in_convection_zones: false,
            num_cells_for_smooth_gradl_composition_term: 3,
            num_cells_for_smooth_brunt_b: 2,
            mass_change_full_on_dt: 1.0e-99,
            mass_change_full_off_dt: 1.0e-99,
            max_t_center_for_any_mass_loss: 2.0e9,
            max_t_center_for_full_mass_loss: 1.0e9,
            wind_h_envelope_limit: -1.0,
            wind_h_he_envelope_limit: -1.0,
            wind_he_layer_limit: -1.0,
            rlo_scaling_factor: 0.0,
            rlo_wind_min_l: 1.0e-6,
            rlo_wind_max_teff: 1.0e99,
            rlo_wind_roche_lobe_radius: 0.40,
            rlo_wind_base_mdot: 1.0e-3,
            rlo_wind_scale_height: 1.0e-1,
            roche_lobe_xfer_full_on: 0.5,
            roche_lobe_xfer_full_off: 1.0,
            num_accretion_species: 0,
            accretion_h2: 0.0,
            accretion_he3: 0.0,
            accretion_zfracs: -1,
            accretion_dump_missing_metals_into_heaviest: true,
            z_fraction_li: 0.0,
            z_fraction_be: 0.0,
            z_fraction_b: 0.0,
            z_fraction_c: 0.0,
            z_fraction_n: 0.0,
            z_fraction_o: 0.0,
            z_fraction_f: 0.0,
            z_fraction_ne: 0.0,
            z_fraction_na: 0.0,
            z_fraction_mg: 0.0,
            z_fraction_al: 0.0,
            z_fraction_si: 0.0,
            z_fraction_p: 0.0,
            z_fraction_s: 0.0,
            z_fraction_cl: 0.0,
            z_fraction_ar: 0.0,
            z_fraction_k: 0.0,
            z_fraction_ca: 0.0,
            z_fraction_sc: 0.0,
            z_fraction_ti: 0.0,
            z_fraction_v: 0.0,
            z_fraction_cr: 0.0,
            z_fraction_mn: 0.0,
            z_fraction_fe: 0.0,
            z_fraction_co: 0.0,
            z_fraction_ni: 0.0,
            z_fraction_cu: 0.0,
            z_fraction_zn: 0.0,
            smooth_outer_xa_big: -1.0,
            smooth_outer_xa_small: -1.0,
        }
    }
}
