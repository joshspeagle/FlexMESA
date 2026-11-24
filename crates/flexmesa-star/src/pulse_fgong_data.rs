// SPDX-License-Identifier: GPL-3.0-or-later
//! Faithful in-memory `star_get_pulse_data_fgong` exporter (`FGONG` branch of
//! `star/private/pulse.f90:89-147` and `star/private/pulse_fgong.f90:47-633`),
//! paired with upstream `build_atm` (`star/private/atm_support.f90:1001-1287`)
//! and `pulse_utils` interpolation (`star/private/pulse_utils.f90:42-523`).
//!
//! Produces MESA's column-major `(global_data, point_data)` (`ICONST = 15`,
//! `IVAR = 40`) in outward-to-inward order (`atmosphere -> envelope -> center`)
//! for direct consumption by `flexmesa-adipls::FgongModel`.

use std::borrow::Cow;
use std::error::Error;
use std::fmt::{self, Display};

use flexmesa_atm::{
    build_t_tau_uniform, build_t_tau_varying, eval_t_tau_uniform,
    eval_t_tau_varying, radiation_pressure, AtmEosCallbackResult, AtmError,
    AtmKapCallbackResult, AtmStructure, ATM_CHIRHO, ATM_CHIT, ATM_CP,
    ATM_DELTA_R, ATM_GAMMA1, ATM_GRADA, ATM_GRADT, ATM_KAP, ATM_LND,
    ATM_LNFREE_E, ATM_LNP, ATM_LNT,
};
use flexmesa_chem::{
    get_composition_info_from_props, NuclideCatalog, SpeciesProps,
};
use flexmesa_const::{
    MixingType, AVO, FOUR_THIRDS, KERG, LN10, PI, PI4, R_SUN, STANDARD_CGRAV,
};
use flexmesa_eos::{
    eos_dt_get_rho, EosConfig, I_CHI_RHO, I_CHI_T, I_ETA, I_GAMMA1,
    I_LN_FREE_E, I_LN_PGAS,
};
use flexmesa_interp1d::interp_4_to_1;
use flexmesa_kap::{get_kap_results, KapConfig};
use flexmesa_math::{exp, log, log10, pow2};

use crate::controls::{StarControls, StarError};
use crate::job_controls::{AtmOption, AtmTTauOpacity, AtmTTauRelation};
use crate::micro::EosZoneContext;
use crate::model::StarModel;
use crate::report_utils::eval_photosphere_info;

/// Fixed FGONG global-vector length (`ICONST = 15` in
/// `star/private/pulse_fgong.f90:42`).
pub const FGONG_ICONST: usize = 15;

/// Fixed FGONG point-variable count per mesh point (`IVAR = 40` in
/// `star/private/pulse_fgong.f90:43`).
pub const FGONG_IVAR: usize = 40;

/// Abundance perturbation $\Delta X$ used to evaluate $\partial\ln\Gamma_1 /
/// \partial Y$ by central finite difference at constant $(\rho, T)$
/// (`dxa = 1d-3` in `star/private/pulse_fgong.f90:98`).
const FGONG_DXA_STEP: f64 = 1.0e-3;

/// `LOGRHO_TOL` and `LOGPGAS_TOL` used by `atm_support::eos_proc`
/// (`star/private/atm_support.f90:1194-1195`).
const ATM_LOG_TOL: f64 = 1.0e-11;

/// Maximum Newton iterations in `solve_eos_given_PgasT`
/// (`star/private/eos_support.f90:345`).
const ATM_EOS_MAX_ITERS: usize = 100;

/// FGONG output slot indices (`point_data[20..=24]`, `point_data[28..=35]`) and
/// corresponding nuclide names (`star/private/pulse_fgong.f90:400-599`).
const FGONG_ISO_SPEC: [(usize, &str); 13] = [
    (20, "he3"),
    (21, "c12"),
    (22, "c13"),
    (23, "n14"),
    (24, "o16"),
    (28, "h2"),
    (29, "he4"),
    (30, "li7"),
    (31, "be7"),
    (32, "n15"),
    (33, "o17"),
    (34, "o18"),
    (35, "ne20"),
];

/// Errors returned by [`get_fgong_data`] and [`star_get_pulse_data_fgong`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FgongDataError {
    /// Invalid or incomplete stellar structure / control input.
    InvalidInput(String),
    /// Upstream option intentionally unimplemented for this path (e.g.
    /// `add_atmosphere = true` with `atm_option != AtmOption::TTau`).
    UnsupportedOption(String),
    /// Failure during `interp_4_to_1` interior face interpolation.
    Interpolation(String),
    /// Failure during EOS or opacity evaluation.
    Microphysics(String),
    /// Failure during atmospheric $T(\tau)$ solve or profile construction.
    Atmosphere(String),
}

impl Display for FgongDataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "FGONG invalid input: {msg}"),
            Self::UnsupportedOption(msg) => {
                write!(f, "FGONG unsupported option: {msg}")
            }
            Self::Interpolation(msg) => {
                write!(f, "FGONG interpolation error: {msg}")
            }
            Self::Microphysics(msg) => {
                write!(f, "FGONG microphysics error: {msg}")
            }
            Self::Atmosphere(msg) => write!(f, "FGONG atmosphere error: {msg}"),
        }
    }
}

impl Error for FgongDataError {}

impl From<FgongDataError> for StarError {
    fn from(err: FgongDataError) -> Self {
        match err {
            FgongDataError::InvalidInput(msg)
            | FgongDataError::UnsupportedOption(msg) => {
                StarError::InvalidInput(msg)
            }
            FgongDataError::Interpolation(msg)
            | FgongDataError::Microphysics(msg)
            | FgongDataError::Atmosphere(msg) => {
                StarError::MicrophysicsError(msg)
            }
        }
    }
}

/// In-memory FGONG pulsation structure returned by [`get_fgong_data`] and
/// [`star_get_pulse_data_fgong`] (`star/private/pulse_fgong.f90:47-633`).
#[derive(Debug, Clone, PartialEq)]
pub struct FgongPulseData {
    /// Global stellar header quantities `global_data(1:15)` (`ICONST = 15`).
    pub global_data: [f64; FGONG_ICONST],
    /// Number of radial mesh points `nn` stored in `point_data`.
    pub nn: usize,
    /// Outward-to-inward mesh points `point_data(1:40, 1:nn)` (`IVAR = 40`).
    pub point_data: Vec<[f64; FGONG_IVAR]>,
}

impl FgongPulseData {
    /// Return `global_data` as an owned `Vec<f64>` of length [`FGONG_ICONST`].
    #[must_use]
    pub fn global_data_vec(&self) -> Vec<f64> {
        self.global_data.to_vec()
    }

    /// Flatten `point_data` into Fortran column-major layout `var(1:40, 1:nn)`
    /// (`len == FGONG_IVAR * nn`) suitable for `flexmesa_adipls::FgongModel`.
    #[must_use]
    pub fn point_var_flat(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.nn * FGONG_IVAR);
        for col in &self.point_data {
            out.extend_from_slice(col);
        }
        out
    }
}

/// Borrowed stellar structure and control inputs for [`get_fgong_data`]
/// (`star/private/pulse_fgong.f90` and `star/private/atm_support.f90`).
///
/// Construct directly for tests or custom models, or build from a [`StarModel`]
/// via [`FgongDataInput::from_model`].
#[derive(Debug, Clone)]
pub struct FgongDataInput<'a> {
    /// Number of interior zones (`s%nz`).
    pub nz: usize,
    /// Number of nuclear species in the reaction network (`s%species`).
    pub species: usize,
    /// 1-based nuclide identifiers `s%chem_id(1:species)`.
    pub chem_id: &'a [usize],
    /// Map `net_iso[chem_id] -> Some(0-based_species_idx)` (`s%net_iso`).
    pub net_iso: &'a [Option<usize>],
    /// Photospheric radius $R_{\mathrm{phot}} / R_\odot$ (`s%photosphere_r`).
    pub photosphere_r: f64,
    /// Effective temperature $T_{\mathrm{eff}}$ in K (`s%Teff`).
    pub teff: f64,
    /// Stellar age in yr (`s%star_age`).
    pub star_age: f64,
    /// Initial bulk metal mass fraction (`s%initial_z`).
    pub initial_z: f64,
    /// Initial bulk helium mass fraction (`s%initial_y`).
    pub initial_y: f64,
    /// Mixing-length parameter $\alpha_{\mathrm{MLT}}$ (`s%mixing_length_alpha`).
    pub mixing_length_alpha: f64,
    /// Whether Brunt-Väisälä $N^2$ has been evaluated (`s%calculate_Brunt_N2`).
    pub calculate_brunt_n2: bool,
    /// Linearly/4-point interpolate $\rho$ instead of conserving shell mass (`s%interpolate_rho_for_pulse_data`).
    pub interpolate_rho_for_pulse_data: bool,
    /// Insert composition-discontinuity double points (`s%add_double_points_to_pulse_data`).
    pub add_double_points_to_pulse_data: bool,
    /// $|\mathrm{d}\ln\mu / \mathrm{d}\ln P|$ threshold for double points (`s%threshold_grad_mu_for_double_point`).
    pub threshold_grad_mu_for_double_point: f64,
    /// Maximum number of double points (`0` = unlimited; `s%max_number_of_double_points`).
    pub max_number_of_double_points: usize,
    /// Zero $A^*$ inside convective regions with $r \le \mathrm{fgong\_zero\_A\_inside\_r} R_\odot$.
    pub fgong_zero_a_inside_r: f64,
    /// Surface atmosphere option (`s%atm_option`).
    pub atm_option: AtmOption,
    /// $T(\tau)$ relation (`s%atm_T_tau_relation`).
    pub atm_t_tau_relation: AtmTTauRelation,
    /// $T(\tau)$ opacity prescription (`s%atm_T_tau_opacity`).
    pub atm_t_tau_opacity: AtmTTauOpacity,
    /// $T(\tau)$ root tolerance (`s%atm_T_tau_errtol`).
    pub atm_t_tau_errtol: f64,
    /// $T(\tau)$ uniform opacity iteration limit (`s%atm_T_tau_max_iters`).
    pub atm_t_tau_max_iters: usize,
    /// $T(\tau)$ varying-opacity ODE step limit (`s%atm_T_tau_max_steps`).
    pub atm_t_tau_max_steps: usize,
    /// Outermost optical depth for `build_atm` (`s%atm_build_tau_outer`).
    pub atm_build_tau_outer: f64,
    /// Integration step $\Delta\log_{10}\tau$ for `build_atm` (`s%atm_build_dlogtau`).
    pub atm_build_dlogtau: f64,
    /// Error tolerance for `build_atm` (`s%atm_build_errtol`).
    pub atm_build_errtol: f64,
    /// Extra non-radiative surface pressure factor (`s%Pextra_factor`).
    pub pextra_factor: f64,
    /// Surface boundary optical depth multiplier (`s%tau_factor`).
    pub tau_factor: f64,
    /// Photospheric optical depth base $\tau_{\mathrm{base}}$ (`s%tau_base`).
    pub tau_base: f64,
    /// EOS configuration (`eos_config`).
    pub eos_config: &'a EosConfig,
    /// Opacity configuration (`kap_config`).
    pub kap_config: &'a KapConfig,
    /// Controls providing `opacity_factor` and kap limits (`controls`).
    pub kap_controls: &'a StarControls,
    /// Whether `kap` uses `use_other_kap` (`controls.use_other_kap`).
    pub use_other_kap: bool,
    /// Whether `kap` uses electron scattering (`controls.use_simple_es_for_kap`).
    pub use_simple_es_for_kap: bool,
    /// Whether `kap` uses starting composition (`controls.use_starting_composition_for_kap`).
    pub use_starting_composition_for_kap: bool,
    /// Starting composition if `use_starting_composition_for_kap` is true (`xa_start`).
    pub xa_start: &'a [f64],
    /// Cell mass fractions `dq(1:nz)`.
    pub dq: &'a [f64],
    /// Cell masses `dm(1:nz)` in g.
    pub dm: &'a [f64],
    /// Enclosed gravitational masses at outer faces `m_grav(1:nz)` in g.
    pub m_grav: &'a [f64],
    /// Gravitational constant `cgrav(1:nz)` in $\mathrm{cm^3\,g^{-1}\,s^{-2}}$.
    pub cgrav: &'a [f64],
    /// Face gravitational accelerations `grav(1:nz)` in $\mathrm{cm\,s^{-2}}$.
    pub grav: &'a [f64],
    /// Outer face radii `r(1:nz)` in cm.
    pub r: &'a [f64],
    /// Cell midpoint radii `rmid(1:nz)` in cm.
    pub rmid: &'a [f64],
    /// Outer face luminosities `L(1:nz)` in $\mathrm{erg\,s^{-1}}$.
    pub lum: &'a [f64],
    /// Cell temperatures `T(1:nz)` in K.
    pub t: &'a [f64],
    /// Cell total EOS pressures `Peos(1:nz)` in $\mathrm{dyn\,cm^{-2}}$.
    pub peos: &'a [f64],
    /// Cell mass densities `rho(1:nz)` in $\mathrm{g\,cm^{-3}}$.
    pub rho: &'a [f64],
    /// Cell mean molecular weights `mu(1:nz)`.
    pub mu: &'a [f64],
    /// Cell hydrogen mass fractions `X(1:nz)`.
    pub xh_mass: &'a [f64],
    /// Cell helium mass fractions `Y(1:nz)`.
    pub yhe_mass: &'a [f64],
    /// Cell Rosseland mean opacities `opacity(1:nz)` in $\mathrm{cm^2\,g^{-1}}$.
    pub opacity: &'a [f64],
    /// Cell nuclear energy generation rates `eps_nuc(1:nz)` in $\mathrm{erg\,g^{-1}\,s^{-1}}$.
    pub eps_nuc: &'a [f64],
    /// Cell gravothermal energy rates `eps_grav_ad(:)%val` in $\mathrm{erg\,g^{-1}\,s^{-1}}$.
    pub eps_grav: Cow<'a, [f64]>,
    /// Cell first adiabatic exponents `gamma1(1:nz)`.
    pub gamma1: &'a [f64],
    /// Cell adiabatic temperature gradients `grada(1:nz)`.
    pub grada: &'a [f64],
    /// Cell density susceptibilities `chiRho(1:nz)`.
    pub chi_rho: &'a [f64],
    /// Cell temperature susceptibilities `chiT(1:nz)`.
    pub chi_t: &'a [f64],
    /// Cell isobaric specific heats `cp(1:nz)` in $\mathrm{erg\,g^{-1}\,K^{-1}}$.
    pub cp: &'a [f64],
    /// Cell log free-electron number densities `lnfree_e(1:nz)`.
    pub lnfree_e: &'a [f64],
    /// Face Brunt-Väisälä squared $N^2$ (`brunt_N2(1:nz)`) in $\mathrm{s^{-2}}$.
    pub brunt_n2: &'a [f64],
    /// Cell mean ion charges `zbar(1:nz)`.
    pub zbar: &'a [f64],
    /// Cell mean atomic masses `abar(1:nz)`.
    pub abar: &'a [f64],
    /// Face mixing types `mixing_type(1:nz)`.
    pub mixing_type: &'a [MixingType],
    /// Zone-major nuclear composition rates `dxdt_nuc[k * species + j]` in $\mathrm{s^{-1}}$.
    pub dxdt_nuc: &'a [f64],
    /// Zone-major mass fractions `xa[k * species + j]`.
    pub xa: &'a [f64],
}

impl<'a> FgongDataInput<'a> {
    /// Construct [`FgongDataInput`] from a [`StarModel`], reading MESA 26.4.1
    /// `controls.defaults` defaults (`atm_build_*`,
    /// `interpolate_rho_for_pulse_data`, double-point flags,
    /// `fgong_zero_a_inside_r`) for controls not yet present on [`StarControls`].
    #[must_use]
    pub fn from_model(model: &'a StarModel) -> Self {
        let phot = eval_photosphere_info(model);
        let initial_y = if model.controls.initial_y < 0.0 {
            (0.24 + 2.0 * model.controls.initial_z).clamp(0.0, 1.0)
        } else {
            model.controls.initial_y
        };
        let eps_grav: Vec<f64> =
            model.eps_grav_ad.iter().map(|v| v.val).collect();
        Self {
            nz: model.nz,
            species: model.species,
            chem_id: &model.chem_id,
            net_iso: &model.net_iso,
            photosphere_r: phot.r_rsun,
            teff: model.teff,
            star_age: model.star_age,
            initial_z: model.controls.initial_z,
            initial_y,
            mixing_length_alpha: model.controls.mixing_length_alpha,
            calculate_brunt_n2: model.controls.calculate_brunt_n2,
            // MESA 26.4.1 `star/defaults/controls.defaults:612-644, 712, 4741-4743`.
            interpolate_rho_for_pulse_data: true,
            add_double_points_to_pulse_data: false,
            threshold_grad_mu_for_double_point: 10.0,
            max_number_of_double_points: 0,
            fgong_zero_a_inside_r: 0.0,
            atm_option: model.controls.atm_option,
            atm_t_tau_relation: model.controls.atm_t_tau_relation,
            atm_t_tau_opacity: model.controls.atm_t_tau_opacity,
            atm_t_tau_errtol: model.controls.atm_t_tau_errtol,
            atm_t_tau_max_iters: model.controls.atm_t_tau_max_iters,
            atm_t_tau_max_steps: model.controls.atm_t_tau_max_steps,
            atm_build_tau_outer: 1.0e-3,
            atm_build_dlogtau: 0.01,
            atm_build_errtol: 1.0e-8,
            pextra_factor: model.controls.pextra_factor,
            tau_factor: model.controls.tau_factor,
            tau_base: model.tau_base,
            eos_config: &model.eos_config,
            kap_config: &model.kap_config,
            kap_controls: &model.controls,
            use_other_kap: model.controls.use_other_kap,
            use_simple_es_for_kap: model.controls.use_simple_es_for_kap,
            use_starting_composition_for_kap: model
                .controls
                .use_starting_composition_for_kap,
            xa_start: &model.xa_start,
            dq: &model.dq,
            dm: &model.dm,
            m_grav: &model.m_grav,
            cgrav: &model.cgrav,
            grav: &model.grav,
            r: &model.r,
            rmid: &model.rmid,
            lum: &model.lum,
            t: &model.t,
            peos: &model.peos,
            rho: &model.rho,
            mu: &model.mu,
            xh_mass: &model.xh_mass,
            yhe_mass: &model.yhe_mass,
            opacity: &model.opacity,
            eps_nuc: &model.eps_nuc,
            eps_grav: Cow::Owned(eps_grav),
            gamma1: &model.gamma1,
            grada: &model.grada,
            chi_rho: &model.chi_rho,
            chi_t: &model.chi_t,
            cp: &model.cp,
            lnfree_e: &model.lnfree_e,
            brunt_n2: &model.brunt_n2,
            zbar: &model.zbar,
            abar: &model.abar,
            mixing_type: &model.mixing_type,
            dxdt_nuc: &model.dxdt_nuc,
            xa: &model.xa,
        }
    }
}

/// Convenience wrapper matching `star_get_pulse_data` (`'FGONG'` branch in
/// `star/private/pulse.f90:89-147`).
///
/// # Errors
/// Returns [`FgongDataError`] if model validation, atmosphere construction,
/// face interpolation, or EOS partial evaluation fails.
pub fn star_get_pulse_data_fgong(
    model: &StarModel,
    add_center_point: bool,
    keep_surface_point: bool,
    add_atmosphere: bool,
) -> Result<FgongPulseData, FgongDataError> {
    let input = FgongDataInput::from_model(model);
    get_fgong_data(&input, add_center_point, keep_surface_point, add_atmosphere)
}

/// Evaluates `get_fgong_data` (`star/private/pulse_fgong.f90:47-633`) from
/// explicit [`FgongDataInput`].
///
/// # Errors
/// Returns [`FgongDataError`] if inputs are malformed, `calculate_brunt_n2` is
/// disabled, `add_atmosphere` encounters an unsupported `atm_option`, or any
/// EOS/kap/interpolation step fails.
#[allow(clippy::too_many_lines)]
pub fn get_fgong_data(
    input: &FgongDataInput<'_>,
    add_center_point: bool,
    keep_surface_point: bool,
    add_atmosphere: bool,
) -> Result<FgongPulseData, FgongDataError> {
    validate_inputs(input)?;

    let nz = input.nz;
    let species = input.species;
    let (k_a, k_b) = set_segment_indices(input, add_center_point)?;
    let n_sg = k_a.len();

    let catalog = NuclideCatalog::builtin_ref();
    let net_idx = |name: &str| -> Option<usize> {
        let cid = catalog.get_nuclide_index(name)?;
        input.net_iso.get(cid).copied().flatten()
    };
    let h1 = net_idx("h1");
    let he4 = net_idx("he4");
    let iso_slots: [(usize, Option<usize>); 13] =
        FGONG_ISO_SPEC.map(|(slot, name)| (slot, net_idx(name)));

    let eos_ctx = EosZoneContext::new(input.chem_id)
        .map_err(|e| FgongDataError::Microphysics(e.to_string()))?;

    let atm_opt: Option<AtmStructure> = if add_atmosphere {
        Some(build_atm_structure(input, catalog)?)
    } else {
        None
    };

    let n_atm = atm_opt.as_ref().map_or(0, |a| a.points.len());
    if add_atmosphere && n_atm < 2 {
        return Err(FgongDataError::Atmosphere(format!(
            "build_atm returned only {n_atm} points (expected >= 2)"
        )));
    }
    let nn_atm = n_atm.saturating_sub(1);
    let n_env = nz;
    let nn_env = if keep_surface_point {
        n_env + n_sg - 1
    } else {
        n_env - 1 + n_sg - 1
    };
    let nn = if add_center_point {
        nn_env + nn_atm + 1
    } else {
        nn_env + nn_atm
    };

    let mut global_data = [0.0_f64; FGONG_ICONST];
    let r_outer = R_SUN * input.photosphere_r;
    let m_outer = input.m_grav[0];

    global_data[0] = m_outer;
    global_data[1] = r_outer;
    global_data[2] = input.lum[0];
    global_data[3] = input.initial_z;
    global_data[4] = 1.0 - (input.initial_y + input.initial_z);
    global_data[5] = input.mixing_length_alpha;
    // global_data[6..=9] (`global_data(7)-global_data(10)`) are unused (`0.0`).

    let last_sg = n_sg - 1;
    let rho_c = if input.interpolate_rho_for_pulse_data {
        eval_center_rho(input, k_b[last_sg])
    } else {
        eval_center(
            input.rmid,
            input.rho,
            k_a[last_sg],
            k_b[last_sg],
            None,
            None,
        )?
    };
    let d2p_dr2_c = -FOUR_THIRDS * PI * input.cgrav[nz - 1] * pow2(rho_c);
    let p_c = input.peos[nz - 1] - 0.5 * d2p_dr2_c * pow2(input.rmid[nz - 1]);
    global_data[10] = pow2(r_outer) * d2p_dr2_c / p_c;
    global_data[11] = pow2(r_outer)
        * eval_center_d2(input.rmid, input.rho, k_a[last_sg], k_b[last_sg])?
        / rho_c;
    global_data[12] = input.star_age;
    global_data[13] = input.teff;
    global_data[14] = STANDARD_CGRAV;

    let mut point_data = Vec::with_capacity(nn);
    let mut xa_buf = vec![0.0_f64; species];
    let mut ctx = PointHelperCtx {
        input,
        eos_ctx: &eos_ctx,
        xa_buf: &mut xa_buf,
        r_outer,
        m_outer,
        h1,
        he4,
        iso_slots,
    };

    if let Some(ref atm) = atm_opt {
        let surf_fg = SurfaceAtmFgong::new(&ctx, k_a[0], k_b[0])?;
        let xa_surf = &input.xa[0..species];
        for k in 0..nn_atm {
            point_data.push(store_point_data_atm(
                &mut ctx,
                &atm.points[k],
                &surf_fg,
                xa_surf,
                k,
            )?);
        }
    }

    let mut sg = 0_usize;
    for k in 0..n_env {
        if k == 0 && !keep_surface_point {
            continue;
        }
        point_data.push(store_point_data_env(&mut ctx, k, k_a[sg], k_b[sg])?);
        if sg + 1 < n_sg && k == k_b[sg] + 1 {
            sg += 1;
            point_data
                .push(store_point_data_env(&mut ctx, k, k_a[sg], k_b[sg])?);
        }
    }

    if add_center_point {
        point_data.push(store_point_data_ctr(
            &mut ctx,
            k_a[last_sg],
            k_b[last_sg],
            p_c,
            rho_c,
        )?);
    }

    debug_assert_eq!(point_data.len(), nn);
    Ok(FgongPulseData {
        global_data,
        nn,
        point_data,
    })
}

struct PointHelperCtx<'a, 'b> {
    input: &'b FgongDataInput<'a>,
    eos_ctx: &'b EosZoneContext,
    xa_buf: &'b mut [f64],
    r_outer: f64,
    m_outer: f64,
    h1: Option<usize>,
    he4: Option<usize>,
    iso_slots: [(usize, Option<usize>); 13],
}

struct SurfaceAtmFgong {
    lnq: f64,
    x: f64,
    l: f64,
    z: f64,
    isos: [(usize, f64); 13],
}

impl SurfaceAtmFgong {
    fn new(
        ctx: &PointHelperCtx<'_, '_>,
        k_a: usize,
        k_b: usize,
    ) -> Result<Self, FgongDataError> {
        let s = ctx.input;
        let lnq = log(s.m_grav[0] / ctx.m_outer);
        let x_unclamped = eval_face(s.dq, s.xh_mass, 0, k_a, k_b, None, None)?;
        let x = x_unclamped.clamp(0.0, 1.0);
        let y_unclamped = eval_face(s.dq, s.yhe_mass, 0, k_a, k_b, None, None)?;
        let z = (1.0 - x_unclamped - y_unclamped).clamp(0.0, 1.0);
        let mut isos = [(0_usize, 0.0_f64); 13];
        for (dst, &(slot, iso)) in isos.iter_mut().zip(&ctx.iso_slots) {
            *dst = (slot, eval_face_x(s, iso, 0, k_a, k_b)?);
        }
        Ok(Self {
            lnq,
            x,
            l: s.lum[0],
            z,
            isos,
        })
    }
}

fn validate_inputs(input: &FgongDataInput<'_>) -> Result<(), FgongDataError> {
    let nz = input.nz;
    let sp = input.species;
    if nz < 2 {
        return Err(FgongDataError::InvalidInput(format!(
            "get_fgong_data requires nz >= 2, got {nz}"
        )));
    }
    if sp == 0 {
        return Err(FgongDataError::InvalidInput(String::from(
            "get_fgong_data requires species >= 1",
        )));
    }
    if !input.calculate_brunt_n2 {
        return Err(FgongDataError::InvalidInput(String::from(
            "eval_face_A_ast: must have calculate_Brunt_N2 = .true.",
        )));
    }
    if [
        input.m_grav.len(),
        input.dq.len(),
        input.dm.len(),
        input.cgrav.len(),
        input.grav.len(),
        input.r.len(),
        input.rmid.len(),
        input.lum.len(),
        input.t.len(),
        input.peos.len(),
        input.rho.len(),
        input.mu.len(),
        input.xh_mass.len(),
        input.yhe_mass.len(),
        input.opacity.len(),
        input.eps_nuc.len(),
        input.eps_grav.len(),
        input.gamma1.len(),
        input.grada.len(),
        input.chi_rho.len(),
        input.chi_t.len(),
        input.cp.len(),
        input.lnfree_e.len(),
        input.brunt_n2.len(),
        input.zbar.len(),
        input.abar.len(),
        input.mixing_type.len(),
    ]
    .into_iter()
    .any(|len| len != nz)
    {
        return Err(FgongDataError::InvalidInput(format!(
            "one or more zone arrays do not match nz={nz}"
        )));
    }
    if input.chem_id.len() != sp {
        return Err(FgongDataError::InvalidInput(format!(
            "chem_id len ({}) != species ({sp})",
            input.chem_id.len()
        )));
    }
    if input.xa.len() != nz * sp || input.dxdt_nuc.len() != nz * sp {
        return Err(FgongDataError::InvalidInput(format!(
            "xa/dxdt_nuc length mismatch for nz={nz}, species={sp}"
        )));
    }
    if input.m_grav[0] <= 0.0 || !input.m_grav[0].is_finite() {
        return Err(FgongDataError::InvalidInput(format!(
            "invalid outer m_grav[0] = {}",
            input.m_grav[0]
        )));
    }
    if input.r[0] <= 0.0 || !input.r[0].is_finite() {
        return Err(FgongDataError::InvalidInput(format!(
            "invalid outer r[0] = {}",
            input.r[0]
        )));
    }
    if input.photosphere_r <= 0.0 || !input.photosphere_r.is_finite() {
        return Err(FgongDataError::InvalidInput(format!(
            "invalid photosphere_r = {}",
            input.photosphere_r
        )));
    }
    Ok(())
}

/// Builds the atmosphere profile via upstream `build_atm`
/// (`star/private/atm_support.f90:1001-1287`).
#[allow(clippy::too_many_lines)]
fn build_atm_structure(
    input: &FgongDataInput<'_>,
    catalog: &NuclideCatalog,
) -> Result<AtmStructure, FgongDataError> {
    if input.atm_option != AtmOption::TTau {
        return Err(FgongDataError::UnsupportedOption(format!(
            "Cannot create atm structure for atm_option '{:?}'; upstream build_atm only supports 'T_tau'",
            input.atm_option
        )));
    }
    if input.use_other_kap {
        return Err(FgongDataError::UnsupportedOption(String::from(
            "build_atm cannot evaluate kap with use_other_kap = .true.",
        )));
    }

    let tau_surf = input.tau_factor * input.tau_base;
    let l = input.lum[0];
    let r = input.r[0];
    let teff = input.teff;
    let m = input.m_grav[0];
    let cgrav = input.cgrav[0];
    let t_tau_id = input.atm_t_tau_relation.id();
    let ht = crate::micro::builtin_helm_table()
        .map_err(|e| FgongDataError::Microphysics(e.to_string()))?;

    let sp = input.species;
    let xa_surf = &input.xa[0..sp];
    let abar_surf = input.abar[0];

    let (mut xc_kap, mut xn_kap, mut xo_kap, mut xne_kap) =
        (0.0, 0.0, 0.0, 0.0);
    let (zbar_kap, xh_kap, z_kap) = if input.use_starting_composition_for_kap {
        if input.xa_start.len() != input.nz * sp {
            return Err(FgongDataError::InvalidInput(format!(
                    "use_starting_composition_for_kap = .true. but xa_start len {} != nz*species ({})",
                    input.xa_start.len(),
                    input.nz * sp
                )));
        }
        let xs = &input.xa_start[0..sp];
        let props = SpeciesProps::from_catalog(catalog, input.chem_id)
            .map_err(|e| FgongDataError::Microphysics(e.to_string()))?;
        let comp = get_composition_info_from_props(&props, xs, true);
        let xh_c = comp.xh.clamp(0.0, 1.0);
        let xhe_c = comp.xhe.clamp(0.0, 1.0);
        let z_c = (1.0 - (xh_c + xhe_c)).clamp(0.0, 1.0);
        for (j, &cid) in input.chem_id.iter().enumerate() {
            if let Some(n) = catalog.get_by_id(cid) {
                match n.z {
                    6 => xc_kap += xs[j],
                    7 => xn_kap += xs[j],
                    8 => xo_kap += xs[j],
                    10 => xne_kap += xs[j],
                    _ => {}
                }
            }
        }
        (comp.zbar, xh_c, z_c)
    } else {
        let mut xh_s = 0.0;
        let mut xhe_s = 0.0;
        for (j, &cid) in input.chem_id.iter().enumerate() {
            if let Some(n) = catalog.get_by_id(cid) {
                match (n.z, n.z_plus_n) {
                    (1, 1 | 2) => xh_s += xa_surf[j],
                    (2, 3 | 4) => xhe_s += xa_surf[j],
                    (6, _) => xc_kap += xa_surf[j],
                    (7, _) => xn_kap += xa_surf[j],
                    (8, _) => xo_kap += xa_surf[j],
                    (10, _) => xne_kap += xa_surf[j],
                    _ => {}
                }
            }
        }
        let xh_c = xh_s.clamp(0.0, 1.0);
        let xhe_c = xhe_s.clamp(0.0, 1.0);
        let z_c = (1.0 - (xh_c + xhe_c)).clamp(0.0, 1.0);
        (input.zbar[0], xh_c, z_c)
    };

    let eos_cfg = input.eos_config;
    let chem_id = input.chem_id;
    let kap_cfg = input.kap_config;
    let ctrl = input.kap_controls;
    let use_simple_es = input.use_simple_es_for_kap;

    let eos_proc =
        |ln_p: f64, ln_t: f64| -> Result<AtmEosCallbackResult, AtmError> {
            let t = exp(ln_t);
            let p = exp(ln_p);
            let prad = radiation_pressure(t);
            let pgas = (p - prad).max(1.0e-99);
            let rho_guess = ((p / t) * abar_surf / (AVO * KERG)).max(1.0e-40);
            let inv = eos_dt_get_rho(
                ht,
                chem_id,
                xa_surf,
                ln_t / LN10,
                I_LN_PGAS as isize,
                log(pgas),
                log10(rho_guess),
                ATM_LOG_TOL,
                ATM_LOG_TOL * LN10,
                ATM_EOS_MAX_ITERS,
                eos_cfg,
            )
            .map_err(|e| AtmError::CallbackError {
                context: "build_atm::eos_proc",
                detail: e.to_string(),
            })?;
            Ok(AtmEosCallbackResult {
                ln_rho: inv.log10_result * LN10,
                res: inv.eos.res,
                dres_dln_rho: inv.eos.d_dlnd,
                dres_dln_t: inv.eos.d_dln_t,
            })
        };

    let kap_proc = |ln_rho: f64,
                    ln_t: f64,
                    eos: &AtmEosCallbackResult|
     -> Result<AtmKapCallbackResult, AtmError> {
        let log_rho = ln_rho / LN10;
        let log_t = ln_t / LN10;
        let (kap_raw, dlnkap_dlnd_raw, dlnkap_dlnt_raw) = if use_simple_es {
            (0.2 * (1.0 + xh_kap), 0.0, 0.0)
        } else {
            let out = get_kap_results(
                kap_cfg,
                zbar_kap,
                xh_kap,
                z_kap,
                xc_kap,
                xn_kap,
                xo_kap,
                xne_kap,
                log_rho,
                log_t,
                eos.res[I_LN_FREE_E],
                eos.dres_dln_rho[I_LN_FREE_E],
                eos.dres_dln_t[I_LN_FREE_E],
                eos.res[I_ETA],
                eos.dres_dln_rho[I_ETA],
                eos.dres_dln_t[I_ETA],
            )
            .map_err(|e| AtmError::CallbackError {
                context: "build_atm::kap_proc",
                detail: e.to_string(),
            })?;
            (out.kap, out.dlnkap_dln_rho, out.dlnkap_dln_t)
        };

        let opacity_factor =
            crate::rsp_eos_kap::effective_opacity_factor(ctrl, log_t);
        let mut kap = kap_raw * opacity_factor;
        let mut dlnkap_dlnd = dlnkap_dlnd_raw;
        let mut dlnkap_dlnt = dlnkap_dlnt_raw;
        if ctrl.opacity_max > 0.0 && kap > ctrl.opacity_max {
            kap = ctrl.opacity_max;
            dlnkap_dlnd = 0.0;
            dlnkap_dlnt = 0.0;
        }
        if ctrl.opacity_min > 0.0 && kap < ctrl.opacity_min {
            kap = ctrl.opacity_min;
            dlnkap_dlnd = 0.0;
            dlnkap_dlnt = 0.0;
        }
        if kap <= 0.0 || !kap.is_finite() {
            return Err(AtmError::CallbackError {
                context: "build_atm::kap_proc",
                detail: format!("invalid atmosphere opacity kap={kap}"),
            });
        }
        Ok(AtmKapCallbackResult {
            kap,
            dlnkap_dln_rho: dlnkap_dlnd,
            dlnkap_dln_t: dlnkap_dlnt,
        })
    };

    let run_uniform =
        |max_iters: usize| -> Result<AtmStructure, FgongDataError> {
            let surf = eval_t_tau_uniform(
                tau_surf,
                l,
                r,
                m,
                cgrav,
                input.opacity[0],
                input.pextra_factor,
                t_tau_id,
                eos_proc,
                kap_proc,
                input.atm_t_tau_errtol,
                max_iters,
                true,
                teff,
            )
            .map_err(|e| FgongDataError::Atmosphere(e.to_string()))?;

            build_t_tau_uniform(
                tau_surf,
                l,
                r,
                teff,
                m,
                cgrav,
                surf.kap,
                input.pextra_factor,
                input.atm_build_tau_outer,
                t_tau_id,
                eos_proc,
                input.atm_build_errtol,
                input.atm_build_dlogtau,
            )
            .map_err(|e| FgongDataError::Atmosphere(e.to_string()))
        };

    match input.atm_t_tau_opacity {
        AtmTTauOpacity::Fixed => run_uniform(0),
        AtmTTauOpacity::Iterated => run_uniform(input.atm_t_tau_max_iters),
        AtmTTauOpacity::Varying => {
            let surf = eval_t_tau_varying(
                tau_surf,
                l,
                r,
                m,
                cgrav,
                t_tau_id,
                eos_proc,
                kap_proc,
                input.atm_t_tau_errtol,
                input.atm_t_tau_max_steps,
                true,
                teff,
            )
            .map_err(|e| FgongDataError::Atmosphere(e.to_string()))?;

            build_t_tau_varying(
                tau_surf,
                l,
                r,
                teff,
                m,
                cgrav,
                surf.ln_p,
                input.atm_build_tau_outer,
                t_tau_id,
                eos_proc,
                kap_proc,
                input.atm_build_errtol,
                input.atm_build_dlogtau,
            )
            .map_err(|e| FgongDataError::Atmosphere(e.to_string()))
        }
    }
}

fn store_point_data_atm(
    ctx: &mut PointHelperCtx<'_, '_>,
    atm_pt: &[f64; 19],
    surf_fg: &SurfaceAtmFgong,
    xa_surf: &[f64],
    k_atm: usize,
) -> Result<[f64; FGONG_IVAR], FgongDataError> {
    let mut pt = [0.0_f64; FGONG_IVAR];
    let s = ctx.input;

    let r = s.r[0] + atm_pt[ATM_DELTA_R];
    let t = exp(atm_pt[ATM_LNT]);
    let p = exp(atm_pt[ATM_LNP]);
    let rho = exp(atm_pt[ATM_LND]);
    let kap = atm_pt[ATM_KAP];
    let gamma1 = atm_pt[ATM_GAMMA1];
    let nabla_ad = atm_pt[ATM_GRADA];
    let delta = atm_pt[ATM_CHIT] / atm_pt[ATM_CHIRHO];
    let c_p = atm_pt[ATM_CP];
    let rec_mu_e = exp(atm_pt[ATM_LNFREE_E]);

    let grav = s.cgrav[0] * s.m_grav[0] / (r * r);
    let n2 = grav * grav * (rho / p) * delta * (nabla_ad - atm_pt[ATM_GRADT]);
    let a_ast = n2 * r / grav;
    let r_r = ctx.r_outer - r;

    pt[0] = r;
    pt[1] = surf_fg.lnq;
    pt[2] = t;
    pt[3] = p;
    pt[4] = rho;
    pt[5] = surf_fg.x;
    pt[6] = surf_fg.l;
    pt[7] = kap;
    pt[8] = 0.0;
    pt[9] = gamma1;
    pt[10] = nabla_ad;
    pt[11] = delta;
    pt[12] = c_p;
    pt[13] = rec_mu_e;
    pt[14] = a_ast;
    pt[15] = 0.0;
    pt[16] = surf_fg.z;
    pt[17] = r_r;
    pt[18] = 0.0;

    let (d1_rho, d1_p, d1_y) = eval_gamma1_derivs(
        ctx.eos_ctx,
        s.eos_config,
        ctx.xa_buf,
        k_atm,
        xa_surf,
        rho,
        t,
        ctx.h1,
        ctx.he4,
    )?;
    pt[25] = d1_rho;
    pt[26] = d1_p;
    pt[27] = d1_y;

    for &(slot, val) in &surf_fg.isos {
        pt[slot] = val;
    }

    Ok(pt)
}

fn store_point_data_env(
    ctx: &mut PointHelperCtx<'_, '_>,
    k: usize,
    k_a: usize,
    k_b: usize,
) -> Result<[f64; FGONG_IVAR], FgongDataError> {
    let mut pt = [0.0_f64; FGONG_IVAR];
    let s = ctx.input;
    let nz = s.nz;
    let sp = s.species;

    let r = s.r[k];
    let lnq = log(s.m_grav[k] / ctx.m_outer);
    // `star/private/pulse_fgong.f90:431-432`: T and P interpolate over `(1, nz)`.
    let t = eval_face(s.dq, s.t, k, 0, nz - 1, None, None)?;
    let p = eval_face(s.dq, s.peos, k, 0, nz - 1, None, None)?;
    let rho = if s.interpolate_rho_for_pulse_data {
        eval_face(s.dq, s.rho, k, k_a, k_b, None, None)?
    } else {
        eval_face_rho(s, k, k_a, k_b)?
    };
    let x_unclamped = eval_face(s.dq, s.xh_mass, k, k_a, k_b, None, None)?;
    let x = x_unclamped.clamp(0.0, 1.0);
    let l = s.lum[k];
    let kap = eval_face(s.dq, s.opacity, k, k_a, k_b, None, None)?;
    let eps_grav = eval_face(s.dq, &s.eps_grav, k, k_a, k_b, None, None)?;
    let eps = eval_face(s.dq, s.eps_nuc, k, k_a, k_b, None, None)? + eps_grav;
    let gamma1 = eval_face(s.dq, s.gamma1, k, k_a, k_b, None, None)?;
    let nabla_ad = eval_face(s.dq, s.grada, k, k_a, k_b, None, None)?;
    let delta = eval_face(s.dq, s.chi_t, k, k_a, k_b, None, None)?
        / eval_face(s.dq, s.chi_rho, k, k_a, k_b, None, None)?;
    let c_p = eval_face(s.dq, s.cp, k, k_a, k_b, None, None)?;
    let rec_mu_e = exp(eval_face(s.dq, s.lnfree_e, k, k_a, k_b, None, None)?);
    let a_ast = if r <= s.fgong_zero_a_inside_r * R_SUN
        && s.mixing_type[k] != MixingType::NoMixing
    {
        0.0
    } else {
        eval_face_a_ast(s, k, k_a, k_b)?
    };
    let r_x = if let Some(ih1) = ctx.h1 {
        eval_face_raw_by(s.dq, k, k_a, k_b, "eval_face", |z| {
            s.dxdt_nuc[z * sp + ih1]
        })?
    } else {
        0.0
    };
    let y_unclamped = eval_face(s.dq, s.yhe_mass, k, k_a, k_b, None, None)?;
    let z = (1.0 - x_unclamped - y_unclamped).clamp(0.0, 1.0);
    let r_r = ctx.r_outer - s.r[k];

    pt[0] = r;
    pt[1] = lnq;
    pt[2] = t;
    pt[3] = p;
    pt[4] = rho;
    pt[5] = x;
    pt[6] = l;
    pt[7] = kap;
    pt[8] = eps;
    pt[9] = gamma1;
    pt[10] = nabla_ad;
    pt[11] = delta;
    pt[12] = c_p;
    pt[13] = rec_mu_e;
    pt[14] = a_ast;
    pt[15] = r_x;
    pt[16] = z;
    pt[17] = r_r;
    pt[18] = eps_grav;

    let xa_k = &s.xa[k * sp..(k + 1) * sp];
    let (d1_rho, d1_p, d1_y) = eval_gamma1_derivs(
        ctx.eos_ctx,
        s.eos_config,
        ctx.xa_buf,
        k,
        xa_k,
        rho,
        t,
        ctx.h1,
        ctx.he4,
    )?;
    pt[25] = d1_rho;
    pt[26] = d1_p;
    pt[27] = d1_y;

    fill_fgong_isotopes(&mut pt, &ctx.iso_slots, |iso| {
        eval_face_x(s, iso, k, k_a, k_b)
    })?;

    Ok(pt)
}

fn store_point_data_ctr(
    ctx: &mut PointHelperCtx<'_, '_>,
    k_a: usize,
    k_b: usize,
    p_c: f64,
    rho_c: f64,
) -> Result<[f64; FGONG_IVAR], FgongDataError> {
    let mut pt = [0.0_f64; FGONG_IVAR];
    let s = ctx.input;
    let nz = s.nz;
    let sp = s.species;

    let lnq = log(f64::MIN_POSITIVE);
    // `star/private/pulse_fgong.f90:559`: T extrapolates over `(1, nz)`.
    let t = eval_center(s.rmid, s.t, 0, nz - 1, None, None)?;
    let p = p_c;
    let rho = rho_c;
    let x_unclamped = eval_center(s.rmid, s.xh_mass, k_a, k_b, None, None)?;
    let x = x_unclamped.clamp(0.0, 1.0);
    let kap = eval_center(s.rmid, s.opacity, k_a, k_b, None, None)?;
    let eps_grav = eval_center(s.rmid, &s.eps_grav, k_a, k_b, None, None)?;
    let eps = eval_center(s.rmid, s.eps_nuc, k_a, k_b, None, None)? + eps_grav;
    let gamma1 = eval_center(s.rmid, s.gamma1, k_a, k_b, None, None)?;
    let nabla_ad = eval_center(s.rmid, s.grada, k_a, k_b, None, None)?;
    let delta = eval_center(s.rmid, s.chi_t, k_a, k_b, None, None)?
        / eval_center(s.rmid, s.chi_rho, k_a, k_b, None, None)?;
    let c_p = eval_center(s.rmid, s.cp, k_a, k_b, None, None)?;
    let rec_mu_e = exp(eval_center(s.rmid, s.lnfree_e, k_a, k_b, None, None)?);
    let r_x = if let Some(ih1) = ctx.h1 {
        eval_center_raw_by(s.rmid, k_a, k_b, "eval_center", |z| {
            s.dxdt_nuc[z * sp + ih1]
        })?
    } else {
        0.0
    };
    let y_unclamped = eval_center(s.rmid, s.yhe_mass, k_a, k_b, None, None)?;
    let z = (1.0 - x_unclamped - y_unclamped).clamp(0.0, 1.0);

    pt[0] = 0.0;
    pt[1] = lnq;
    pt[2] = t;
    pt[3] = p;
    pt[4] = rho;
    pt[5] = x;
    pt[6] = 0.0;
    pt[7] = kap;
    pt[8] = eps;
    pt[9] = gamma1;
    pt[10] = nabla_ad;
    pt[11] = delta;
    pt[12] = c_p;
    pt[13] = rec_mu_e;
    pt[14] = 0.0;
    pt[15] = r_x;
    pt[16] = z;
    pt[17] = ctx.r_outer;
    pt[18] = eps_grav;

    let xa_ctr = &s.xa[(nz - 1) * sp..nz * sp];
    let (d1_rho, d1_p, d1_y) = eval_gamma1_derivs(
        ctx.eos_ctx,
        s.eos_config,
        ctx.xa_buf,
        nz - 1,
        xa_ctr,
        rho,
        t,
        ctx.h1,
        ctx.he4,
    )?;
    pt[25] = d1_rho;
    pt[26] = d1_p;
    pt[27] = d1_y;

    fill_fgong_isotopes(&mut pt, &ctx.iso_slots, |iso| {
        eval_center_x(s, iso, k_a, k_b)
    })?;

    Ok(pt)
}

fn fill_fgong_isotopes(
    pt: &mut [f64; FGONG_IVAR],
    iso_slots: &[(usize, Option<usize>); 13],
    mut eval_iso: impl FnMut(Option<usize>) -> Result<f64, FgongDataError>,
) -> Result<(), FgongDataError> {
    for &(slot, iso) in iso_slots {
        pt[slot] = eval_iso(iso)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn eval_gamma1_derivs(
    eos_ctx: &EosZoneContext,
    eos_cfg: &EosConfig,
    xa_buf: &mut [f64],
    k: usize,
    xa_actual: &[f64],
    rho: f64,
    t: f64,
    h1: Option<usize>,
    he4: Option<usize>,
) -> Result<(f64, f64, f64), FgongDataError> {
    let log_rho = log10(rho);
    let log_t = log10(t);

    let (d_gamma1_dxa, d_lnpgas_dxa) = if let (Some(ih1), Some(ihe4)) =
        (h1, he4)
    {
        xa_buf.copy_from_slice(xa_actual);
        xa_buf[ih1] = xa_actual[ih1] - FGONG_DXA_STEP;
        xa_buf[ihe4] = xa_actual[ihe4] + FGONG_DXA_STEP;
        let lo = eos_ctx
            .eval_cell(k, xa_buf, rho, log_rho, t, log_t, eos_cfg)
            .map_err(|e| FgongDataError::Microphysics(e.to_string()))?;

        xa_buf[ih1] = xa_actual[ih1] + FGONG_DXA_STEP;
        xa_buf[ihe4] = xa_actual[ihe4] - FGONG_DXA_STEP;
        let hi = eos_ctx
            .eval_cell(k, xa_buf, rho, log_rho, t, log_t, eos_cfg)
            .map_err(|e| FgongDataError::Microphysics(e.to_string()))?;

        let dg1 = (hi.res[I_GAMMA1] - lo.res[I_GAMMA1]) / 2.0 / FGONG_DXA_STEP;
        let dlp =
            (hi.res[I_LN_PGAS] - lo.res[I_LN_PGAS]) / 2.0 / FGONG_DXA_STEP;
        (dg1, dlp)
    } else {
        (0.0, 0.0)
    };

    let act = eos_ctx
        .eval_cell(k, xa_actual, rho, log_rho, t, log_t, eos_cfg)
        .map_err(|e| FgongDataError::Microphysics(e.to_string()))?;

    let d_dlnd = act.d_dlnd[I_GAMMA1];
    let d_dlnt = act.d_dln_t[I_GAMMA1];
    let chi_rho = act.res[I_CHI_RHO];
    let chi_t = act.res[I_CHI_T];
    let g1 = act.res[I_GAMMA1];

    let dln_gamma1_dln_rho = (d_dlnd - d_dlnt * chi_rho / chi_t) / g1;
    let dln_gamma1_dln_p = d_dlnt / chi_t / g1;
    let dln_gamma1_dy = if h1.is_some() && he4.is_some() {
        -(d_gamma1_dxa - d_dlnt * d_lnpgas_dxa / chi_t) / g1
    } else {
        0.0
    };

    Ok((dln_gamma1_dln_rho, dln_gamma1_dln_p, dln_gamma1_dy))
}

fn set_segment_indices(
    input: &FgongDataInput<'_>,
    include_last_face: bool,
) -> Result<(Vec<usize>, Vec<usize>), FgongDataError> {
    let nz = input.nz;
    if !input.add_double_points_to_pulse_data {
        return Ok((vec![0], vec![nz - 1]));
    }
    let mut mask = vec![false; nz];

    let mut grad_mu = vec![0.0_f64; nz];
    for k in 1..(nz - 1) {
        grad_mu[k] = log(input.mu[k] / input.mu[k - 1])
            / log(input.peos[k] / input.peos[k - 1]);
    }
    if include_last_face {
        let k = nz - 1;
        grad_mu[k] = log(input.mu[k] / input.mu[k - 1])
            / log(input.peos[k] / input.peos[k - 1]);
    }
    for k in 0..nz {
        mask[k] = grad_mu[k].abs() > input.threshold_grad_mu_for_double_point;
    }
    if input.max_number_of_double_points > 0 {
        let n_mk = input.max_number_of_double_points.min(nz);
        let mut order: Vec<usize> = (0..nz).collect();
        order.sort_by(|&a, &b| {
            grad_mu[b]
                .abs()
                .total_cmp(&grad_mu[a].abs())
                .then_with(|| a.cmp(&b))
        });
        for &idx in &order[n_mk..] {
            mask[idx] = false;
        }
    }

    let n_sg = mask.iter().filter(|&&m| m).count() + 1;
    let mut k_a = Vec::with_capacity(n_sg);
    let mut k_b = Vec::with_capacity(n_sg);
    k_a.push(0);
    for (k, &is_split) in mask.iter().enumerate() {
        if is_split {
            if k == 0 {
                return Err(FgongDataError::InvalidInput(String::from(
                    "double point cannot occur at k=0",
                )));
            }
            k_b.push(k - 1);
            k_a.push(k);
        }
    }
    k_b.push(nz - 1);
    Ok((k_a, k_b))
}

fn eval_face_raw_by(
    dq: &[f64],
    k: usize,
    k_a: usize,
    k_b: usize,
    err_ctx: &str,
    v_at: impl Fn(usize) -> f64,
) -> Result<f64, FgongDataError> {
    if k_b < k_a || k < k_a || k > k_b + 1 {
        return Err(FgongDataError::InvalidInput(format!(
            "{err_ctx}: invalid indices k={k}, segment=({k_a}, {k_b})"
        )));
    }
    if k_b == k_a {
        Ok(v_at(k_a))
    } else if k == k_a {
        Ok(v_at(k_a)
            - dq[k_a] * (v_at(k_a + 1) - v_at(k_a)) / (dq[k_a + 1] + dq[k_a]))
    } else if k == k_a + 1 {
        Ok(v_at(k_a)
            + dq[k_a] * (v_at(k_a + 1) - v_at(k_a)) / (dq[k_a + 1] + dq[k_a]))
    } else if k == k_b {
        Ok(v_at(k_b)
            - dq[k_b] * (v_at(k_b) - v_at(k_b - 1)) / (dq[k_b] + dq[k_b - 1]))
    } else if k == k_b + 1 {
        Ok(v_at(k_b)
            + dq[k_b] * (v_at(k_b) - v_at(k_b - 1)) / (dq[k_b] + dq[k_b - 1]))
    } else {
        interp_4_to_1(
            0.5 * (dq[k - 2] + dq[k - 1]),
            0.5 * (dq[k - 1] + dq[k]),
            0.5 * (dq[k] + dq[k + 1]),
            0.5 * dq[k - 2] + dq[k - 1],
            v_at(k - 2),
            v_at(k - 1),
            v_at(k),
            v_at(k + 1),
        )
        .map_err(|e| FgongDataError::Interpolation(e.to_string()))
    }
}

fn eval_face(
    dq: &[f64],
    v: &[f64],
    k: usize,
    k_a: usize,
    k_b: usize,
    v_lo: Option<f64>,
    v_hi: Option<f64>,
) -> Result<f64, FgongDataError> {
    let mut val = eval_face_raw_by(dq, k, k_a, k_b, "eval_face", |z| v[z])?;
    if let Some(lo) = v_lo {
        val = val.max(lo);
    }
    if let Some(hi) = v_hi {
        val = val.min(hi);
    }
    Ok(val)
}

fn eval_face_x(
    input: &FgongDataInput<'_>,
    species_idx: Option<usize>,
    k: usize,
    k_a: usize,
    k_b: usize,
) -> Result<f64, FgongDataError> {
    let Some(i) = species_idx else {
        return Ok(0.0);
    };
    let sp = input.species;
    let raw = eval_face_raw_by(input.dq, k, k_a, k_b, "eval_face_X", |z| {
        input.xa[z * sp + i]
    })?;
    Ok(raw.clamp(0.0, 1.0))
}

fn eval_face_a_ast(
    input: &FgongDataInput<'_>,
    k: usize,
    k_a: usize,
    k_b: usize,
) -> Result<f64, FgongDataError> {
    if k_b < k_a || k < k_a || k > k_b + 1 || k >= input.nz {
        return Err(FgongDataError::InvalidInput(format!(
            "eval_face_A_ast: invalid indices k={k}, segment=({k_a}, {k_b}), nz={}",
            input.nz
        )));
    }
    if k_b == k_a {
        return Ok(input.brunt_n2[k] * input.r[k] / input.grav[k]);
    }
    if k == k_a {
        if k_a + 2 >= input.nz {
            return Err(FgongDataError::InvalidInput(format!(
                "eval_face_A_ast: undefined upstream read at k_a+2={} >= nz={} for segment ({k_a}, {k_b}) (pulse_utils.f90:275-276)",
                k_a + 2,
                input.nz
            )));
        }
        let a1 =
            input.brunt_n2[k_a + 1] * input.r[k_a + 1] / input.grav[k_a + 1];
        let a2 =
            input.brunt_n2[k_a + 2] * input.r[k_a + 2] / input.grav[k_a + 2];
        Ok(a1 - input.dq[k_a] * (a2 - a1) / input.dq[k_a + 1])
    } else if k == k_b + 1 {
        // Matches `pulse_utils.f90:282-285` (both terms divide by `grav(k_b)`).
        let a1 = input.brunt_n2[k_b - 1] * input.r[k_b - 1] / input.grav[k_b];
        let a2 = input.brunt_n2[k_b] * input.r[k_b] / input.grav[k_b];
        Ok(a2 + input.dq[k_b] * (a2 - a1) / input.dq[k_b - 1])
    } else {
        Ok(input.brunt_n2[k] * input.r[k] / input.grav[k])
    }
}

fn eval_face_rho(
    input: &FgongDataInput<'_>,
    k: usize,
    k_a: usize,
    k_b: usize,
) -> Result<f64, FgongDataError> {
    if k_b < k_a || k < k_a || k > k_b + 1 || k >= input.nz {
        return Err(FgongDataError::InvalidInput(format!(
            "eval_face_rho: invalid indices k={k}, segment=({k_a}, {k_b}), nz={}",
            input.nz
        )));
    }
    if k_b == k_a {
        return Ok(input.rho[k]);
    }
    let (r, dm, dlnr) = if k == k_a {
        let r = input.r[k_a];
        (r, 0.5 * input.dm[k_a], 1.0 - input.rmid[k_a] / r)
    } else if k == k_b + 1 {
        let r = input.r[k_b + 1];
        (r, 0.5 * input.dm[k_b], input.rmid[k_b] / r - 1.0)
    } else {
        let r = input.r[k];
        (
            r,
            0.5 * (input.dm[k] + input.dm[k - 1]),
            (input.rmid[k - 1] - input.rmid[k]) / r,
        )
    };
    Ok(dm / (PI4 * r * r * r * dlnr))
}

fn eval_center_raw_by(
    r: &[f64],
    k_a: usize,
    k_b: usize,
    err_ctx: &str,
    v_at: impl Fn(usize) -> f64,
) -> Result<f64, FgongDataError> {
    if k_b < k_a {
        return Err(FgongDataError::InvalidInput(format!(
            "{err_ctx}: invalid segment indices ({k_a}, {k_b})"
        )));
    }
    if k_a == k_b {
        Ok(v_at(k_a))
    } else {
        let r1 = r[k_b];
        let r2 = r[k_b - 1];
        let v1 = v_at(k_b);
        let v2 = v_at(k_b - 1);
        Ok((v1 * r2 * r2 - v2 * r1 * r1) / (r2 * r2 - r1 * r1))
    }
}

fn eval_center(
    r: &[f64],
    v: &[f64],
    k_a: usize,
    k_b: usize,
    v_lo: Option<f64>,
    v_hi: Option<f64>,
) -> Result<f64, FgongDataError> {
    let mut val = eval_center_raw_by(r, k_a, k_b, "eval_center", |z| v[z])?;
    if let Some(lo) = v_lo {
        val = val.max(lo);
    }
    if let Some(hi) = v_hi {
        val = val.min(hi);
    }
    Ok(val)
}

fn eval_center_x(
    input: &FgongDataInput<'_>,
    species_idx: Option<usize>,
    k_a: usize,
    k_b: usize,
) -> Result<f64, FgongDataError> {
    let Some(i) = species_idx else {
        return Ok(0.0);
    };
    let sp = input.species;
    let raw = eval_center_raw_by(input.rmid, k_a, k_b, "eval_center_X", |z| {
        input.xa[z * sp + i]
    })?;
    Ok(raw.clamp(0.0, 1.0))
}

fn eval_center_rho(input: &FgongDataInput<'_>, k_b: usize) -> f64 {
    let r1 = input.rmid[k_b];
    let rho1 = input.rho[k_b];
    let m1 = input.m_grav[k_b] - 0.5 * input.dm[k_b];
    3.0 * (5.0 * m1 / (PI * r1 * r1 * r1) - 4.0 * rho1) / 8.0
}

fn eval_center_d2(
    r: &[f64],
    v: &[f64],
    k_a: usize,
    k_b: usize,
) -> Result<f64, FgongDataError> {
    if k_b < k_a {
        return Err(FgongDataError::InvalidInput(format!(
            "eval_center_d2: invalid segment indices ({k_a}, {k_b})"
        )));
    }
    if k_a == k_b {
        Ok(0.0)
    } else {
        let r1 = r[k_b];
        let r2 = r[k_b - 1];
        let v1 = v[k_b];
        let v2 = v[k_b - 1];
        Ok(2.0 * (v2 - v1) / (r2 * r2 - r1 * r1))
    }
}
