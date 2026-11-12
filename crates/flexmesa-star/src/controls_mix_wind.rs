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

/// One indexed convective overshooting parameter set (`1..=16` in
/// `controls.defaults:2580-2660`).
#[derive(Debug, Clone, Default)]
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
    /// Lower $q$ bound for Region II/III transition `k_const_mass` (`min_q_for_k_const_mass`).
    pub min_q_for_k_const_mass: f64,
    /// Upper $q$ bound for Region II/III transition `k_const_mass` (`max_q_for_k_const_mass`).
    pub max_q_for_k_const_mass: f64,
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

    // --- Adaptive Mesh Spacing Weights (`mesh_function.f90`) ---
    /// Maximum relative cell mass fraction $\Delta q_{\max}$ (`max_dq`, `1e-2` default).
    pub max_dq: f64,
    /// Minimum relative cell mass fraction $\Delta q_{\min}$ (`min_dq`, `1e-14` default).
    pub min_dq: f64,
    /// Minimum cell $\Delta q$ eligible for splitting (`min_dq_for_split`, `1e-14` default).
    pub min_dq_for_split: f64,
    /// Maximum surface cell mass fraction (`max_surface_cell_dq`, `1e-8` default).
    pub max_surface_cell_dq: f64,
    /// Maximum center cell mass fraction (`max_center_cell_dq`, `1e-6` default).
    pub max_center_cell_dq: f64,
    /// Pressure gradient weight in mesh spacing function (`P_function_weight`, `30.0` default).
    pub p_function_weight: f64,
    /// Primary temperature gradient weight in mesh spacing function (`T_function1_weight`, `110.0` default).
    pub t_function1_weight: f64,
    /// Radius gradient weight in mesh spacing function (`R_function_weight`, `0.0` default).
    pub r_function_weight: f64,
    /// Mass gradient weight in mesh spacing function (`M_function_weight`, `0.0` default).
    pub m_function_weight: f64,
    /// Angular-velocity gradient weight in mesh spacing function (`omega_function_weight`, `0.0` default).
    pub omega_function_weight: f64,
    /// Coulomb $\Gamma$ gradient weight in mesh spacing function (`gam_function_weight`, `0.0` default).
    pub gam_function_weight: f64,
    /// Species names for abundance mesh spacing functions (`xa_function_species(1..)`).
    pub xa_function_species: Vec<String>,
    /// Weights for abundance mesh spacing functions (`xa_function_weight(1..)`).
    pub xa_function_weight: Vec<f64>,
    /// Transition scales for abundance mesh spacing functions (`xa_function_param(1..)`).
    pub xa_function_param: Vec<f64>,
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
            conv_premix_avoid_increase: true,
            conv_premix_time_factor: 0.0,
            conv_premix_fix_pgas: true,
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
            no_wind_if_no_rotation: false,
            super_eddington_wind_ledd_factor: 1.0,
            wind_boost_full_off_l_div_ledd: 1.5,
            wind_boost_full_on_l_div_ledd: 5.0,
            super_eddington_wind_max_boost: 1.0,
            max_star_mass_for_gain: -1.0,
            min_star_mass_for_loss: -1.0,
            use_accreted_material_j: false,
            accreted_material_j: 0.0,
            min_q_for_k_below_const_q: 0.995,
            max_q_for_k_below_const_q: 0.999,
            min_q_for_k_const_mass: 0.99,
            max_q_for_k_const_mass: 0.995,
            eps_mdot_factor: 1.0,
            eps_mdot_leak_frac_factor: 1.0,
            eps_wd_sedimentation_factor: 1.0,
            extra_power_source: 0.0,
            inject_uniform_extra_heat: 0.0,
            min_q_for_uniform_extra_heat: 0.0,
            max_q_for_uniform_extra_heat: 1.0,
            max_dq: 1.0e-2,
            min_dq: 1.0e-14,
            min_dq_for_split: 1.0e-14,
            max_surface_cell_dq: 1.0e-8,
            max_center_cell_dq: 1.0e-6,
            p_function_weight: 30.0,
            t_function1_weight: 110.0,
            r_function_weight: 0.0,
            m_function_weight: 0.0,
            omega_function_weight: 0.0,
            gam_function_weight: 0.0,
            xa_function_species: Vec::new(),
            xa_function_weight: Vec::new(),
            xa_function_param: Vec::new(),
        }
    }
}
