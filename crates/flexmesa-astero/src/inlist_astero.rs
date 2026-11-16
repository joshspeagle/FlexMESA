// SPDX-License-Identifier: LGPL-3.0-or-later
//! Configuration structs and defaults for `&astero_search_controls` namelists
//! and `from_file` parameter tables (`astero/defaults/astero_search.defaults`,
//! `astero/private/astero_run_support.f90`).

use std::fs;
use std::path::Path;

pub use crate::inlist_parser::parse_astero_search_controls;
use crate::ratios_chi2::AsteroChi2Weights;

/// Searchable stellar model parameter specification (`first_*`, `min_*`, `max_*`, `delta_*`).
#[derive(Debug, Clone, Copy)]
pub struct AsteroSearchParam {
    /// Whether this parameter is varied by the optimizer (`vary_*`).
    pub vary: bool,
    /// Initial / fixed parameter value (`first_*`).
    pub first: f64,
    /// Lower search bound (`min_*`).
    pub min: f64,
    /// Upper search bound (`max_*`).
    pub max: f64,
    /// Grid step size (`delta_*`).
    pub delta: f64,
}

impl AsteroSearchParam {
    /// Creates a fixed parameter with `first = min = max = val`.
    #[must_use]
    pub const fn fixed(val: f64) -> Self {
        Self { vary: false, first: val, min: val, max: val, delta: 0.0 }
    }
}

/// Spectroscopic observable constraint (`*_target`, `*_sigma`, `include_*_in_chi2_spectro`).
#[derive(Debug, Clone, Copy, Default)]
pub struct SpectroConstraint {
    /// Whether this observable contributes to $\chi^2_{\mathrm{spectro}}$.
    pub include: bool,
    /// Target observed value.
    pub target: f64,
    /// $1\sigma$ observational uncertainty.
    pub sigma: f64,
    /// Multiplier for hard rejection limit (`sigmas_coeff_for_*_limit`).
    pub sigmas_coeff: f64,
}

/// Evaluated parameter vector for one stellar track (`astero_support.f90:54-63`).
#[derive(Debug, Clone, Copy)]
pub struct AsteroParamSet {
    /// Initial stellar mass $M / M_\odot$.
    pub initial_mass: f64,
    /// Initial helium mass fraction $Y_0$.
    pub initial_y: f64,
    /// Initial metallicity $[\mathrm{Fe/H}]_0$.
    pub initial_feh: f64,
    /// Mixing-length parameter $\alpha_{\mathrm{MLT}}$.
    pub alpha: f64,
    /// Exponential convective core overshoot parameter $f_{\mathrm{ov}}$.
    pub f_ov: f64,
    /// User parameter 1 (`my_param1`).
    pub my_param1: f64,
    /// User parameter 2 (`my_param2`).
    pub my_param2: f64,
    /// User parameter 3 (`my_param3`).
    pub my_param3: f64,
}

/// Parsed `&astero_search_controls` configuration (`astero/defaults/astero_search.defaults`).
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct AsteroSearchControls {
    /// Composite $\chi^2$ weights and normalization flags.
    pub weights: AsteroChi2Weights,
    /// Whether to log limit-rejection diagnostics (`trace_limits`).
    pub trace_limits: bool,
    /// Effective temperature $T_{\mathrm{eff}}$ (K) constraint.
    pub teff: SpectroConstraint,
    /// Surface gravity $\log g$ (cgs) constraint.
    pub logg: SpectroConstraint,
    /// Luminosity $\log(L/L_\odot)$ constraint.
    pub log_l: SpectroConstraint,
    /// Surface metallicity $[\mathrm{Fe/H}]$ constraint.
    pub feh: SpectroConstraint,
    /// Log photospheric radius $\log(R/R_\odot)$ constraint.
    pub log_r: SpectroConstraint,
    /// Linear photospheric radius $R/R_\odot$ constraint.
    pub radius: SpectroConstraint,
    /// Surface $[\mathrm{M/H}]$ constraint.
    pub m_h: SpectroConstraint,
    /// Surface $(Z/X)$ constraint.
    pub surface_z_div_x: SpectroConstraint,
    /// Surface helium mass fraction $Y_{\mathrm{surf}}$ constraint.
    pub surface_he: SpectroConstraint,
    /// Convective envelope base radius $R_{\mathrm{cz}}/R_\odot$ constraint.
    pub rcz: SpectroConstraint,
    /// Solar reference $(Z/X)_\odot$.
    pub z_div_x_solar: f64,
    /// Number of observed target modes per spherical degree $l \in \{0, 1, 2, 3\}$.
    pub nl: [usize; 4],
    /// Target mode frequencies ($\mu\mathrm{Hz}$) per degree $l \in \{0, 1, 2, 3\}$.
    pub freq_target: [Vec<f64>; 4],
    /// Target mode $1\sigma$ uncertainties ($\mu\mathrm{Hz}$) per degree $l \in \{0, 1, 2, 3\}$.
    pub freq_sigma: [Vec<f64>; 4],
    /// Observed radial orders $n$ for $l = 0$ modes (`l0_n_obs`).
    pub l0_n_obs: Vec<f64>,
    /// Observed large frequency separation $\Delta\nu$ ($\mu\mathrm{Hz}$).
    pub delta_nu: f64,
    /// Observed $\Delta\nu$ uncertainty $\sigma_{\Delta\nu}$ ($\mu\mathrm{Hz}$).
    pub delta_nu_sigma: f64,
    /// Multiplier for hard $\Delta\nu$ limit (`sigmas_coeff_for_delta_nu_limit`).
    pub sigmas_coeff_for_delta_nu_limit: f64,
    /// Observed frequency of maximum oscillation power $\nu_{\max}$ ($\mu\mathrm{Hz}$).
    pub nu_max: f64,
    /// Observed $\nu_{\max}$ uncertainty $\sigma_{\nu_{\max}}$ ($\mu\mathrm{Hz}$).
    pub nu_max_sigma: f64,
    /// Oscillation solver backend (`"adipls"` or `"gyre"`).
    pub oscillation_code: String,
    /// Lower frequency scan bracket factor (`nu_lower_factor`).
    pub nu_lower_factor: f64,
    /// Upper frequency scan bracket factor (`nu_upper_factor`).
    pub nu_upper_factor: f64,
    /// Per-degree scan grid density multiplier (`iscan_factor(0..3)`).
    pub iscan_factor: [f64; 4],
    /// Path to the `GYRE` namelist input file (`gyre_input_file`).
    pub gyre_input_file: String,
    /// Whether non-adiabatic `GYRE` mode calculations are enabled (`gyre_non_ad`).
    pub gyre_non_ad: bool,
    /// Whether to attach a $T(\tau)$ atmosphere before pulsation export (`add_atmosphere`).
    pub add_atmosphere: bool,
    /// Whether to retain the outermost $x = 1$ surface zone (`keep_surface_point`).
    pub keep_surface_point: bool,
    /// Whether to add an extrapolated $r = 0$ center point (`add_center_point`).
    pub add_center_point: bool,
    /// Surface-effect frequency correction scheme (`correction_scheme`).
    pub correction_scheme: String,
    /// Kjeldsen et al. (2008) power-law exponent $b$ (`correction_b`).
    pub correction_b: f64,
    /// Fraction of surface correction applied (`correction_factor`).
    pub correction_factor: f64,
    /// Prescribed Kjeldsen $a/r$ parameter (`a_div_r`).
    pub a_div_r: f64,
    /// Solar reference $\nu_{\max,\odot}$ ($\mu\mathrm{Hz}$).
    pub solar_nu_max: f64,
    /// Solar reference $\Delta\nu_\odot$ ($\mu\mathrm{Hz}$).
    pub solar_delta_nu: f64,
    /// Optimization driver selection (`search_type`).
    pub search_type: String,
    /// Whether stellar age is included in $\chi^2_{\mathrm{spectro}}$.
    pub include_age_in_chi2_spectro: bool,
    /// Whether $\chi^2$ is evaluated only at `age_target` (`eval_chi2_at_target_age_only`).
    pub eval_chi2_at_target_age_only: bool,
    /// Target stellar age (yr).
    pub age_target: f64,
    /// Uncertainty on target stellar age (yr).
    pub age_sigma: f64,
    /// Number of reduced-timestep steps before `age_target`.
    pub num_smaller_steps_before_age_target: usize,
    /// Timestep size (yr) for steps approaching `age_target`.
    pub dt_for_smaller_steps_before_age_target: f64,
    /// Minimum stellar age (yr) for $\chi^2$ evaluation (`min_age_for_chi2`).
    pub min_age_for_chi2: Option<f64>,
    /// Maximum stellar age (yr) for $\chi^2$ evaluation (`max_age_for_chi2`).
    pub max_age_for_chi2: Option<f64>,
    /// Minimum stellar age (yr) before seismic checks start (`min_age_limit`).
    pub min_age_limit: f64,
    /// Minimum $L_{\mathrm{nuc}}/L$ ratio before seismic checks start (`lnuc_div_l_limit`).
    pub lnuc_div_l_limit: f64,
    /// Maximum $\chi^2_{\mathrm{spectro}}$ threshold before mode evaluation (`chi2_spectroscopic_limit`).
    pub chi2_spectroscopic_limit: f64,
    /// Maximum $\chi^2_{\Delta\nu}$ threshold before mode evaluation (`chi2_delta_nu_limit`).
    pub chi2_delta_nu_limit: f64,
    /// Maximum radial-mode $\chi^2_{\mathrm{radial}}$ threshold before nonradial modes (`chi2_radial_limit`).
    pub chi2_radial_limit: f64,
    /// Search limit 1 (`chi2_search_limit1`).
    pub chi2_search_limit1: f64,
    /// Search limit 2 (`chi2_search_limit2`).
    pub chi2_search_limit2: f64,
    /// Relative $\chi^2$ increase threshold (`chi2_relative_increase_limit`).
    pub chi2_relative_increase_limit: f64,
    /// Consecutive steps with increasing $\chi^2$ before track termination (`limit_num_chi2_too_big`).
    pub limit_num_chi2_too_big: usize,
    /// Whether initial helium follows $Y = Y_0 + (dY/dZ) Z$ (`y_depends_on_z`).
    pub y_depends_on_z: bool,
    /// Primordial helium mass fraction $Y_0$ (`y0`).
    pub y0: f64,
    /// Helium-to-metal enrichment slope $dY/dZ$ (`dydz`).
    pub dydz: f64,
    /// Initial mass search parameter (`mass`).
    pub mass: AsteroSearchParam,
    /// Initial helium search parameter (`y`).
    pub y: AsteroSearchParam,
    /// Initial $[\mathrm{Fe/H}]$ search parameter (`feh`).
    pub feh_param: AsteroSearchParam,
    /// Mixing-length $\alpha_{\mathrm{MLT}}$ search parameter (`alpha`).
    pub alpha: AsteroSearchParam,
    /// Core overshoot $f_{\mathrm{ov}}$ search parameter (`f_ov`).
    pub f_ov: AsteroSearchParam,
    /// User-defined search parameters (`my_param1..3`).
    pub my_param: [AsteroSearchParam; 3],
    /// Ratio $f_{0,\mathrm{ov}} / f_{\mathrm{ov}}$ (`f0_ov_div_f_ov`).
    pub f0_ov_div_f_ov: f64,
    /// Primordial ${}^3\mathrm{He}/Y$ fraction (`y_frac_he3`).
    pub y_frac_he3: f64,
    /// Maximum Nelder-Mead iterations (`simplex_itermax`).
    pub simplex_itermax: usize,
    /// Maximum Nelder-Mead function evaluations (`simplex_fcn_calls_max`).
    pub simplex_fcn_calls_max: usize,
    /// Nelder-Mead absolute parameter tolerance (`simplex_x_atol`).
    pub simplex_x_atol: f64,
    /// Nelder-Mead relative parameter tolerance (`simplex_x_rtol`).
    pub simplex_x_rtol: f64,
    /// Nelder-Mead $\chi^2$ termination threshold (`simplex_chi2_tol`).
    pub simplex_chi2_tol: f64,
    /// Nelder-Mead centroid weight power (`simplex_centroid_weight_power`).
    pub simplex_centroid_weight_power: f64,
    /// Whether Nelder-Mead clamps trial points to bounds (`simplex_enforce_bounds`).
    pub simplex_enforce_bounds: bool,
    /// Whether Nelder-Mead tests adaptive random points (`simplex_adaptive_random_search`).
    pub simplex_adaptive_random_search: bool,
    /// RNG seed for Nelder-Mead random search (`simplex_seed`).
    pub simplex_seed: i32,
    /// Output filename for `simplex` search (`simplex_output_filename`).
    pub simplex_output_filename: String,
    /// Initial normalized trust-region radius for `bobyqa` (`bobyqa_rho_beg`).
    pub bobyqa_rho_beg: f64,
    /// Final normalized trust-region radius for `bobyqa` (`bobyqa_rho_end`).
    pub bobyqa_rho_end: f64,
    /// Maximum function evaluations for `bobyqa` (`bobyqa_max_fcn_calls`).
    pub bobyqa_max_fcn_calls: usize,
    /// Output filename for `bobyqa` search (`bobyqa_output_filename`).
    pub bobyqa_output_filename: String,
    /// Initial normalized trust-region radius for `newuoa` (`newuoa_rho_beg`).
    pub newuoa_rho_beg: f64,
    /// Final normalized trust-region radius for `newuoa` (`newuoa_rho_end`).
    pub newuoa_rho_end: f64,
    /// Maximum function evaluations for `newuoa` (`newuoa_max_fcn_calls`).
    pub newuoa_max_fcn_calls: usize,
    /// Output filename for `newuoa` search (`newuoa_output_filename`).
    pub newuoa_output_filename: String,
    /// Output directory for sample and summary files (`astero_results_directory`).
    pub astero_results_directory: String,
    /// Output filename for `scan_grid` search (`scan_grid_output_filename`).
    pub scan_grid_output_filename: String,
    /// Input parameter table filename for `from_file` (`filename_for_parameters`).
    pub filename_for_parameters: String,
    /// Maximum rows to evaluate from `filename_for_parameters` (`max_num_from_file`).
    pub max_num_from_file: isize,
    /// 1-based column for `initial_mass` in `filename_for_parameters`.
    pub file_column_for_mass: isize,
    /// 1-based column for `initial_Y` in `filename_for_parameters`.
    pub file_column_for_y: isize,
    /// 1-based column for `initial_FeH` in `filename_for_parameters`.
    pub file_column_for_feh: isize,
    /// 1-based column for `alpha` in `filename_for_parameters`.
    pub file_column_for_alpha: isize,
    /// 1-based column for `f_ov` in `filename_for_parameters`.
    pub file_column_for_f_ov: isize,
    /// 1-based columns for `my_param1..3` in `filename_for_parameters`.
    pub file_column_for_my_param: [isize; 3],
    /// Output filename for `from_file` search (`from_file_output_filename`).
    pub from_file_output_filename: String,
    /// Maximum timestep (yr) when far from target box (`max_yrs_dt_when_cold`).
    pub max_yrs_dt_when_cold: f64,
    /// Maximum timestep (yr) when approaching target box (`max_yrs_dt_when_warm`).
    pub max_yrs_dt_when_warm: f64,
    /// Maximum timestep (yr) inside spectroscopic/$\Delta\nu$ box (`max_yrs_dt_when_hot`).
    pub max_yrs_dt_when_hot: f64,
    /// Threshold on $\chi^2$ for reduced timesteps (`chi2_limit_for_small_timesteps`).
    pub chi2_limit_for_small_timesteps: f64,
    /// Maximum timestep (yr) when $\chi^2 \le \chi^2_{\mathrm{limit}}$ (`max_yrs_dt_chi2_small_limit`).
    pub max_yrs_dt_chi2_small_limit: f64,
    /// Sigma threshold for entering `hot` timesteps (`sigmas_coeff_for_hot_timesteps`).
    pub sigmas_coeff_for_hot_timesteps: f64,
    /// Sigma threshold for entering `warm` timesteps (`sigmas_coeff_for_warm_timesteps`).
    pub sigmas_coeff_for_warm_timesteps: f64,
    pub(crate) constraint_names: [String; 10],
    pub(crate) indexed_constraints: [SpectroConstraint; 10],
    pub(crate) param_names: [String; 10],
    pub(crate) indexed_params: [AsteroSearchParam; 10],
    pub(crate) indexed_file_cols: [isize; 10],
    pub(crate) has_custom_param: [bool; 3],
}

impl Default for AsteroSearchControls {
    fn default() -> Self {
        Self {
            weights: AsteroChi2Weights::default(),
            trace_limits: false,
            teff: SpectroConstraint::default(), logg: SpectroConstraint::default(),
            log_l: SpectroConstraint::default(), feh: SpectroConstraint::default(),
            log_r: SpectroConstraint::default(), radius: SpectroConstraint::default(),
            m_h: SpectroConstraint::default(), surface_z_div_x: SpectroConstraint::default(),
            surface_he: SpectroConstraint::default(), rcz: SpectroConstraint::default(),
            z_div_x_solar: 0.02293,
            nl: [0; 4],
            freq_target: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            freq_sigma: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            l0_n_obs: Vec::new(),
            delta_nu: -1.0, delta_nu_sigma: -1.0, sigmas_coeff_for_delta_nu_limit: 0.0,
            nu_max: -1.0, nu_max_sigma: -1.0,
            oscillation_code: String::from("adipls"),
            nu_lower_factor: 0.8, nu_upper_factor: 1.2, iscan_factor: [1.0; 4],
            gyre_input_file: String::from("gyre.in"), gyre_non_ad: false,
            add_atmosphere: false, keep_surface_point: false, add_center_point: true,
            correction_scheme: String::from("kjeldsen"), correction_b: 4.90,
            correction_factor: 0.0, a_div_r: 0.0,
            solar_nu_max: 3090.0, solar_delta_nu: 135.1,
            search_type: String::from("use_first_values"),
            include_age_in_chi2_spectro: false, eval_chi2_at_target_age_only: false,
            age_target: 4.5695e9, age_sigma: 0.0065e9,
            num_smaller_steps_before_age_target: 50,
            dt_for_smaller_steps_before_age_target: 6.5e5,
            min_age_for_chi2: None, max_age_for_chi2: None,
            min_age_limit: 1.0e6, lnuc_div_l_limit: 0.95,
            chi2_spectroscopic_limit: 1000.0, chi2_delta_nu_limit: 1000.0,
            chi2_radial_limit: 100.0, chi2_search_limit1: 3.0, chi2_search_limit2: 4.0,
            chi2_relative_increase_limit: 2.0, limit_num_chi2_too_big: 20,
            y_depends_on_z: false, y0: 0.248, dydz: 1.4,
            mass: AsteroSearchParam::fixed(1.0), y: AsteroSearchParam::fixed(0.274),
            feh_param: AsteroSearchParam::fixed(0.0), alpha: AsteroSearchParam::fixed(1.90),
            f_ov: AsteroSearchParam::fixed(0.0), my_param: [AsteroSearchParam::fixed(0.0); 3],
            f0_ov_div_f_ov: 1.0, y_frac_he3: 1.0e-4,
            simplex_itermax: 1000, simplex_fcn_calls_max: 1000,
            simplex_x_atol: 1.0e-10, simplex_x_rtol: 1.0e-10, simplex_chi2_tol: 1.0e-10,
            simplex_centroid_weight_power: 0.0,
            simplex_enforce_bounds: false, simplex_adaptive_random_search: false,
            simplex_seed: 1_074_698_122,
            simplex_output_filename: String::from("simplex_results.data"),
            bobyqa_rho_beg: 0.45, bobyqa_rho_end: 1.0e-6, bobyqa_max_fcn_calls: 1000,
            bobyqa_output_filename: String::from("bobyqa_results.data"),
            newuoa_rho_beg: 0.45, newuoa_rho_end: 1.0e-6, newuoa_max_fcn_calls: 1000,
            newuoa_output_filename: String::from("newuoa_results.data"),
            astero_results_directory: String::from("outputs"),
            scan_grid_output_filename: String::from("scan_grid_results.data"),
            filename_for_parameters: String::from("parameters.txt"),
            max_num_from_file: -1,
            file_column_for_mass: 3, file_column_for_y: 4, file_column_for_feh: 5,
            file_column_for_alpha: 6, file_column_for_f_ov: 7,
            file_column_for_my_param: [8, 9, 10],
            from_file_output_filename: String::from("from_file_results.data"),
            max_yrs_dt_when_cold: 1.0e8, max_yrs_dt_when_warm: 1.0e7,
            max_yrs_dt_when_hot: 1.0e6,
            chi2_limit_for_small_timesteps: -1.0, max_yrs_dt_chi2_small_limit: 3.0e5,
            sigmas_coeff_for_hot_timesteps: 2.0, sigmas_coeff_for_warm_timesteps: 5.0,
            constraint_names: std::array::from_fn(|_| String::new()),
            indexed_constraints: [SpectroConstraint::default(); 10],
            param_names: std::array::from_fn(|_| String::new()),
            indexed_params: [AsteroSearchParam::fixed(0.0); 10],
            indexed_file_cols: [3, 4, 5, 6, 7, 8, 9, 10, -1, -1],
            has_custom_param: [false; 3],
        }
    }
}

impl AsteroSearchControls {
    /// Returns the initial parameter set (`first_*`).
    #[must_use]
    pub fn first_params(&self) -> AsteroParamSet {
        AsteroParamSet {
            initial_mass: self.mass.first,
            initial_y: self.y.first,
            initial_feh: self.feh_param.first,
            alpha: self.alpha.first,
            f_ov: self.f_ov.first,
            my_param1: self.my_param[0].first,
            my_param2: self.my_param[1].first,
            my_param3: self.my_param[2].first,
        }
    }

    /// Reads candidate parameter rows from `filename_for_parameters` (`from_file`).
    #[must_use]
    pub fn read_parameters_from_file(&self, case_dir: &Path) -> Vec<AsteroParamSet> {
        let Ok(text) = fs::read_to_string(case_dir.join(&self.filename_for_parameters)) else {
            return vec![self.first_params()];
        };
        let mut out = Vec::new();
        for line in text.lines() {
            let trimmed = line.split('!').next().unwrap_or("").trim();
            if trimmed.is_empty() {
                continue;
            }
            let tokens: Vec<&str> = trimmed.split_whitespace().collect();
            let cols: Vec<f64> = tokens
                .iter()
                .filter_map(|t| t.replace(['d', 'D'], "e").parse::<f64>().ok())
                .collect();
            if cols.len() != tokens.len() || cols.is_empty() {
                continue;
            }
            if cols.len() >= 3
                && (cols[0] - 1.0).abs() < 1e-12
                && (cols[1] - 2.0).abs() < 1e-12
                && (cols[2] - 3.0).abs() < 1e-12
            {
                continue;
            }
            let mut p = self.first_params();
            let get = |c: isize, fb: f64| {
                if c > 0 && (c as usize) <= cols.len() {
                    cols[(c as usize) - 1]
                } else {
                    fb
                }
            };
            p.initial_mass = get(self.file_column_for_mass, p.initial_mass);
            p.initial_y = get(self.file_column_for_y, p.initial_y);
            p.initial_feh = get(self.file_column_for_feh, p.initial_feh);
            p.alpha = get(self.file_column_for_alpha, p.alpha);
            p.f_ov = get(self.file_column_for_f_ov, p.f_ov);
            p.my_param1 = get(self.file_column_for_my_param[0], p.my_param1);
            p.my_param2 = get(self.file_column_for_my_param[1], p.my_param2);
            p.my_param3 = get(self.file_column_for_my_param[2], p.my_param3);
            out.push(p);
            if self.max_num_from_file > 0
                && out.len() >= (self.max_num_from_file as usize)
            {
                break;
            }
        }
        if out.is_empty() {
            out.push(self.first_params());
        }
        out
    }
}
