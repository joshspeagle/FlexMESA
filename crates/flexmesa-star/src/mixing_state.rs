// SPDX-License-Identifier: LGPL-3.0-or-later
//! Auxiliary mixing, MLT, Brunt composition-gradient smoothing, convective
//! boundary tracking, and thermohaline driving-species state
//! (`star_data_step_work.inc`, `star/private/star_utils.f90:2777-2887`, and
//! `star/private/hydro_vars.f90:1078-1133`).

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::MixingType;
use flexmesa_turb::DrivingSpecies;

use crate::model::StarModel;
use crate::star_ad::{StarAdExt, StarAdOrder1};

/// Auxiliary mixing, MLT, convective-boundary, and composition-gradient arrays
/// stored on [`crate::hydro_state::HydroAuxState::mixing`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MixingAuxState {
    /// Raw MLT/TDC/semiconvection/thermohaline mixing type at face `k`, kept
    /// separate from `model.mixing_type` so `set_mlt_vars` never clobbers
    /// predictive mixing, overshooting, or post-glitch `mixing_type`
    /// (`mlt_mixing_type` in `turb_info.f90:253`).
    pub mlt_mixing_type: Vec<MixingType>,
    /// MLT mixing length $\Lambda_k = \alpha_{\mathrm{MLT}} H_{P,k}$ in cm
    /// (`mlt_mixing_length` in `turb_info.f90:287`).
    pub mlt_mixing_length: Vec<f64>,
    /// Face-interpolated adiabatic gradient $\nabla_{\mathrm{ad,face},k}$
    /// (`grada_face` in `turb_info.f90:256`).
    pub grada_face: Vec<f64>,
    /// Face-interpolated $\nabla_{\mathrm{ad,face},k}$ with 3-cell AD partials
    /// (`grada_face_ad` in `turb_info.f90:255`).
    pub grada_face_ad: Vec<StarAdOrder1>,
    /// Superadiabaticity $\nabla_{T,k} - \nabla_{\mathrm{ad,face},k}$
    /// (`gradT_sub_grada` in `turb_info.f90:354`).
    pub gradt_sub_grada: Vec<f64>,
    /// Unadjusted MLT $\nabla_{T,k}$ prior to `adjust_gradT_fraction` /
    /// `adjust_gradT_excess` (`mlt_gradT` in `turb_info.f90:260`).
    pub mlt_gradt: Vec<f64>,
    /// Superadiabatic excess $Y_{\mathrm{face},k}$ (`Y_face` in `turb_info.f90:263`).
    pub y_face: Vec<f64>,
    /// Convective luminosity $L_{\mathrm{conv},k}$ in $\mathrm{erg\,s^{-1}}$
    /// (`L_conv` in `turb_info.f90:245-248`).
    pub l_conv: Vec<f64>,
    /// Unsmoothed Brassard/MHM Brunt composition term $B_k$ saved before
    /// `threshold_smoothing` (`unsmoothed_brunt_B` in `brunt.f90:76`).
    pub unsmoothed_brunt_b: Vec<f64>,
    /// Smoothed Ledoux composition term copied into $\nabla_L$
    /// (`gradL_composition_term` in `hydro_vars.f90:1089-1093`).
    pub gradl_composition_term: Vec<f64>,
    /// 1-based catalog `chem_id` of the dominant thermohaline driving isotope
    /// at face `k` (`dominant_iso_for_thermohaline` in `hydro_vars.f90:1099-1117`).
    pub dominant_iso_for_thermohaline: Vec<usize>,
    /// Multiplicative factor applied to $\nabla_{\mathrm{rad},k}$
    /// (`gradr_factor` in `hydro_vars.f90:582-600`).
    pub gradr_factor: Vec<f64>,
    /// Per-face mixing-length parameter $\alpha_{\mathrm{MLT},k}$
    /// (`alpha_mlt` in `hydro_vars.f90:571`).
    pub alpha_mlt: Vec<f64>,
    /// Face mixing diffusivity factor $(4\pi r_k^2 \rho_{\mathrm{face},k})^2 D_{\mathrm{mix},k}$
    /// kept separate from `model.mlt_cdc` (`cdc` in `mix_info.f90:115`).
    pub cdc: Vec<f64>,
    /// Uncapped raw Lagrangian diffusion coupling $\sigma_{\mathrm{raw},k}$
    /// (`sig_raw` in `mix_info.f90:1519`).
    pub sig_raw: Vec<f64>,
    /// Non-rotational mixing diffusivity $D_{\mathrm{mix,non\_rot},k}$ recorded
    /// prior to adding rotational diffusivities (`D_mix_non_rotation` in
    /// `mix_info.f90:255`).
    pub d_mix_non_rotation: Vec<f64>,
    /// Sub-cell convective/semiconvective boundary mass-fraction offset
    /// $\Delta q_{\mathrm{bdy},k}$ (`cz_bdy_dq` in `mix_info.f90:428-469`).
    pub cz_bdy_dq: Vec<f64>,
    /// Convective boundary 0-based cell indices ordered center to surface
    /// (`conv_bdy_loc` in `mix_info.f90:579`, converted to 0-based).
    pub conv_bdy_loc: Vec<usize>,
    /// Convective boundary mass-fraction coordinates $q_{\mathrm{cb}}$
    /// (`conv_bdy_q` in `mix_info.f90:580`).
    pub conv_bdy_q: Vec<f64>,
    /// True if convective boundary `i` is the top (outer) edge of a CZ
    /// (`top_conv_bdy` in `mix_info.f90:578`).
    pub top_conv_bdy: Vec<bool>,
    /// H-burning classification flags for convective boundaries (`burn_h_conv_region`).
    pub burn_h_conv_region: Vec<bool>,
    /// He-burning classification flags for convective boundaries (`burn_he_conv_region`).
    pub burn_he_conv_region: Vec<bool>,
    /// Metal-burning classification flags for convective boundaries (`burn_z_conv_region`).
    pub burn_z_conv_region: Vec<bool>,
    /// Convective zone top mass coordinates in g (`cz_top_mass` in `mix_info.f90:472-517`).
    pub cz_top_mass: Vec<f64>,
    /// Convective zone bottom mass coordinates in g (`cz_bot_mass` in `mix_info.f90:472-517`).
    pub cz_bot_mass: Vec<f64>,
    /// All-mixing-region boundary 0-based cell indices ordered center to surface
    /// (`mix_bdy_loc` in `mix_info.f90:788-880`, converted to 0-based).
    pub mix_bdy_loc: Vec<usize>,
    /// All-mixing-region boundary mass fractions (`mix_bdy_q`).
    pub mix_bdy_q: Vec<f64>,
    /// True if mixing boundary `i` is the top (outer) edge (`top_mix_bdy`).
    pub top_mix_bdy: Vec<bool>,
    /// H-burning flags for mixing boundaries (`burn_h_mix_region`).
    pub burn_h_mix_region: Vec<bool>,
    /// He-burning flags for mixing boundaries (`burn_he_mix_region`).
    pub burn_he_mix_region: Vec<bool>,
    /// Metal-burning flags for mixing boundaries (`burn_z_mix_region`).
    pub burn_z_mix_region: Vec<bool>,
    /// Whether `set_hydro_vars` is permitted to run `set_mixing_info`
    /// (`okay_to_set_mixing_info` in `evolve.f90:320,569,632`).
    pub okay_to_set_mixing_info: bool,
    /// Whether `set_mixing_info` has populated `mixing_type`/`d_mix`/`cdc`
    /// (`have_mixing_info` in `mix_info.f90:290`).
    pub have_mixing_info: bool,
    /// Whether `set_cz_bdy_mass` has been updated since the last mixing update
    /// (`have_new_cz_bdy_info` in `mix_info.f90:472`).
    pub have_new_cz_bdy_info: bool,
    /// Implicit superadiabaticity reduction factor (`superad_reduction_factor`
    /// in `turb_support.f90:289,504`).
    pub superad_reduction_factor: Vec<f64>,
    /// Active MLT++ efficiency boost scalar (`gradT_excess_alpha` in
    /// `turb_info.f90:411-519`).
    pub gradt_excess_alpha: f64,
    /// Previous-step `gradT_excess_alpha` for time smoothing and retry rollback
    /// (`gradT_excess_alpha_old`).
    pub gradt_excess_alpha_old: f64,
    /// Minimum gas-pressure fraction $\beta$ recorded by `set_gradT_excess_alpha`
    /// (`gradT_excess_min_beta`).
    pub gradt_excess_min_beta: f64,
    /// Maximum $L_{\mathrm{rad}}/L_{\mathrm{Edd}}$ recorded by `set_gradT_excess_alpha`
    /// (`gradT_excess_max_lambda`).
    pub gradt_excess_max_lambda: f64,
    /// Per-face MLT++ reduction effect $1 - \alpha$ (`gradT_excess_effect` in
    /// `turb_info.f90:367,385`).
    pub gradt_excess_effect: Vec<f64>,
    /// Per-face custom $\nabla_T$ blend fraction (`adjust_mlt_gradT_fraction` in
    /// `turb_info.f90:240`; negative disables).
    pub adjust_mlt_gradt_fraction: Vec<f64>,
    /// Crystallized core boundary mass in g (`crystal_core_boundary_mass` in
    /// `turb_info.f90:145`; negative when absent).
    pub crystal_core_boundary_mass: f64,
    /// Previous-step `crystal_core_boundary_mass` for retry rollback.
    pub crystal_core_boundary_mass_old: f64,
    /// Outer mass coordinate mixed by phase separation in g
    /// (`phase_sep_mixing_mass` in `turb_info.f90:154`; negative at step start).
    pub phase_sep_mixing_mass: f64,
}

impl MixingAuxState {
    /// Allocates a default-initialized [`MixingAuxState`] for `nz` zones.
    #[must_use]
    pub fn new(nz: usize) -> Self {
        Self {
            mlt_mixing_type: vec![MixingType::NoMixing; nz],
            mlt_mixing_length: vec![0.0; nz],
            grada_face: vec![0.0; nz],
            grada_face_ad: vec![StarAdOrder1::ZERO; nz],
            gradt_sub_grada: vec![0.0; nz],
            mlt_gradt: vec![0.0; nz],
            y_face: vec![0.0; nz],
            l_conv: vec![0.0; nz],
            unsmoothed_brunt_b: vec![0.0; nz],
            gradl_composition_term: vec![0.0; nz],
            dominant_iso_for_thermohaline: vec![0; nz],
            gradr_factor: vec![1.0; nz],
            alpha_mlt: vec![0.0; nz],
            cdc: vec![0.0; nz],
            sig_raw: vec![0.0; nz],
            d_mix_non_rotation: vec![0.0; nz],
            cz_bdy_dq: vec![0.0; nz],
            conv_bdy_loc: Vec::new(),
            conv_bdy_q: Vec::new(),
            top_conv_bdy: Vec::new(),
            burn_h_conv_region: Vec::new(),
            burn_he_conv_region: Vec::new(),
            burn_z_conv_region: Vec::new(),
            cz_top_mass: Vec::new(),
            cz_bot_mass: Vec::new(),
            mix_bdy_loc: Vec::new(),
            mix_bdy_q: Vec::new(),
            top_mix_bdy: Vec::new(),
            burn_h_mix_region: Vec::new(),
            burn_he_mix_region: Vec::new(),
            burn_z_mix_region: Vec::new(),
            okay_to_set_mixing_info: true,
            have_mixing_info: false,
            have_new_cz_bdy_info: false,
            superad_reduction_factor: vec![1.0; nz],
            gradt_excess_alpha: 0.0,
            gradt_excess_alpha_old: 0.0,
            gradt_excess_min_beta: 0.0,
            gradt_excess_max_lambda: 0.0,
            gradt_excess_effect: vec![0.0; nz],
            adjust_mlt_gradt_fraction: vec![-1.0; nz],
            crystal_core_boundary_mass: -1.0,
            crystal_core_boundary_mass_old: -1.0,
            phase_sep_mixing_mass: -1.0,
        }
    }

    /// Resizes all zone-indexed vectors in [`MixingAuxState`] to `nz`,
    /// clearing zone-index boundary tables whenever `nz` changes so stale
    /// indices cannot survive across a shrinking remesh or rollback.
    pub fn resize(&mut self, nz: usize) {
        let len_changed = self.mlt_mixing_type.len() != nz;
        self.mlt_mixing_type.resize(nz, MixingType::NoMixing);
        for v in [
            &mut self.mlt_mixing_length,
            &mut self.grada_face,
            &mut self.gradt_sub_grada,
            &mut self.mlt_gradt,
            &mut self.y_face,
            &mut self.l_conv,
            &mut self.unsmoothed_brunt_b,
            &mut self.gradl_composition_term,
            &mut self.alpha_mlt,
            &mut self.cdc,
            &mut self.sig_raw,
            &mut self.d_mix_non_rotation,
            &mut self.cz_bdy_dq,
            &mut self.gradt_excess_effect,
        ] {
            v.resize(nz, 0.0);
        }
        self.grada_face_ad.resize(nz, StarAdOrder1::ZERO);
        self.dominant_iso_for_thermohaline.resize(nz, 0);
        self.gradr_factor.resize(nz, 1.0);
        self.superad_reduction_factor.resize(nz, 1.0);
        self.adjust_mlt_gradt_fraction.resize(nz, -1.0);
        if len_changed {
            self.conv_bdy_loc.clear();
            self.conv_bdy_q.clear();
            self.top_conv_bdy.clear();
            self.burn_h_conv_region.clear();
            self.burn_he_conv_region.clear();
            self.burn_z_conv_region.clear();
            self.cz_top_mass.clear();
            self.cz_bot_mass.clear();
            self.mix_bdy_loc.clear();
            self.mix_bdy_q.clear();
            self.top_mix_bdy.clear();
            self.burn_h_mix_region.clear();
            self.burn_he_mix_region.clear();
            self.burn_z_mix_region.clear();
        }
    }

    /// Returns total live heap bytes held by [`MixingAuxState`] vectors.
    #[must_use]
    pub(crate) fn heap_bytes(&self) -> usize {
        use std::mem::size_of_val;
        size_of_val(self.mlt_mixing_type.as_slice())
            + size_of_val(self.mlt_mixing_length.as_slice())
            + size_of_val(self.grada_face.as_slice())
            + size_of_val(self.grada_face_ad.as_slice())
            + size_of_val(self.gradt_sub_grada.as_slice())
            + size_of_val(self.mlt_gradt.as_slice())
            + size_of_val(self.y_face.as_slice())
            + size_of_val(self.l_conv.as_slice())
            + size_of_val(self.unsmoothed_brunt_b.as_slice())
            + size_of_val(self.gradl_composition_term.as_slice())
            + size_of_val(self.dominant_iso_for_thermohaline.as_slice())
            + size_of_val(self.gradr_factor.as_slice())
            + size_of_val(self.alpha_mlt.as_slice())
            + size_of_val(self.cdc.as_slice())
            + size_of_val(self.sig_raw.as_slice())
            + size_of_val(self.d_mix_non_rotation.as_slice())
            + size_of_val(self.cz_bdy_dq.as_slice())
            + size_of_val(self.conv_bdy_loc.as_slice())
            + size_of_val(self.conv_bdy_q.as_slice())
            + size_of_val(self.top_conv_bdy.as_slice())
            + size_of_val(self.burn_h_conv_region.as_slice())
            + size_of_val(self.burn_he_conv_region.as_slice())
            + size_of_val(self.burn_z_conv_region.as_slice())
            + size_of_val(self.cz_top_mass.as_slice())
            + size_of_val(self.cz_bot_mass.as_slice())
            + size_of_val(self.mix_bdy_loc.as_slice())
            + size_of_val(self.mix_bdy_q.as_slice())
            + size_of_val(self.top_mix_bdy.as_slice())
            + size_of_val(self.burn_h_mix_region.as_slice())
            + size_of_val(self.burn_he_mix_region.as_slice())
            + size_of_val(self.burn_z_mix_region.as_slice())
            + size_of_val(self.superad_reduction_factor.as_slice())
            + size_of_val(self.gradt_excess_effect.as_slice())
            + size_of_val(self.adjust_mlt_gradt_fraction.as_slice())
    }
}

/// Compact rollback snapshot of persistent mixing state for
/// [`crate::primary_state::PrimaryStateBuffer`].
#[derive(Debug, Clone)]
pub(crate) struct MixingPrimarySnapshot {
    pub(crate) mlt_mixing_type: Vec<MixingType>,
    pub(crate) alpha_mlt: Vec<f64>,
    pub(crate) cdc: Vec<f64>,
    pub(crate) sig_raw: Vec<f64>,
    pub(crate) cz_bdy_dq: Vec<f64>,
    pub(crate) d_mix_non_rotation: Vec<f64>,
    pub(crate) conv_bdy_loc: Vec<usize>,
    pub(crate) conv_bdy_q: Vec<f64>,
    pub(crate) top_conv_bdy: Vec<bool>,
    pub(crate) burn_h_conv_region: Vec<bool>,
    pub(crate) burn_he_conv_region: Vec<bool>,
    pub(crate) burn_z_conv_region: Vec<bool>,
    pub(crate) cz_top_mass: Vec<f64>,
    pub(crate) cz_bot_mass: Vec<f64>,
    pub(crate) mix_bdy_loc: Vec<usize>,
    pub(crate) mix_bdy_q: Vec<f64>,
    pub(crate) top_mix_bdy: Vec<bool>,
    pub(crate) burn_h_mix_region: Vec<bool>,
    pub(crate) burn_he_mix_region: Vec<bool>,
    pub(crate) burn_z_mix_region: Vec<bool>,
    pub(crate) okay_to_set_mixing_info: bool,
    pub(crate) have_mixing_info: bool,
    pub(crate) have_new_cz_bdy_info: bool,
    pub(crate) gradt_excess_alpha: f64,
    pub(crate) gradt_excess_alpha_old: f64,
    pub(crate) crystal_core_boundary_mass: f64,
    pub(crate) crystal_core_boundary_mass_old: f64,
    pub(crate) phase_sep_mixing_mass: f64,
}

impl Default for MixingPrimarySnapshot {
    fn default() -> Self {
        Self {
            mlt_mixing_type: Vec::new(),
            alpha_mlt: Vec::new(),
            cdc: Vec::new(),
            sig_raw: Vec::new(),
            cz_bdy_dq: Vec::new(),
            d_mix_non_rotation: Vec::new(),
            conv_bdy_loc: Vec::new(),
            conv_bdy_q: Vec::new(),
            top_conv_bdy: Vec::new(),
            burn_h_conv_region: Vec::new(),
            burn_he_conv_region: Vec::new(),
            burn_z_conv_region: Vec::new(),
            cz_top_mass: Vec::new(),
            cz_bot_mass: Vec::new(),
            mix_bdy_loc: Vec::new(),
            mix_bdy_q: Vec::new(),
            top_mix_bdy: Vec::new(),
            burn_h_mix_region: Vec::new(),
            burn_he_mix_region: Vec::new(),
            burn_z_mix_region: Vec::new(),
            okay_to_set_mixing_info: true,
            have_mixing_info: false,
            have_new_cz_bdy_info: false,
            gradt_excess_alpha: 0.0,
            gradt_excess_alpha_old: 0.0,
            crystal_core_boundary_mass: -1.0,
            crystal_core_boundary_mass_old: -1.0,
            phase_sep_mixing_mass: -1.0,
        }
    }
}

impl MixingPrimarySnapshot {
    pub(crate) fn capture_from(&mut self, m: &MixingAuxState) {
        self.mlt_mixing_type.clone_from(&m.mlt_mixing_type);
        self.alpha_mlt.clone_from(&m.alpha_mlt);
        self.cdc.clone_from(&m.cdc);
        self.sig_raw.clone_from(&m.sig_raw);
        self.cz_bdy_dq.clone_from(&m.cz_bdy_dq);
        self.d_mix_non_rotation.clone_from(&m.d_mix_non_rotation);
        self.conv_bdy_loc.clone_from(&m.conv_bdy_loc);
        self.conv_bdy_q.clone_from(&m.conv_bdy_q);
        self.top_conv_bdy.clone_from(&m.top_conv_bdy);
        self.burn_h_conv_region.clone_from(&m.burn_h_conv_region);
        self.burn_he_conv_region.clone_from(&m.burn_he_conv_region);
        self.burn_z_conv_region.clone_from(&m.burn_z_conv_region);
        self.cz_top_mass.clone_from(&m.cz_top_mass);
        self.cz_bot_mass.clone_from(&m.cz_bot_mass);
        self.mix_bdy_loc.clone_from(&m.mix_bdy_loc);
        self.mix_bdy_q.clone_from(&m.mix_bdy_q);
        self.top_mix_bdy.clone_from(&m.top_mix_bdy);
        self.burn_h_mix_region.clone_from(&m.burn_h_mix_region);
        self.burn_he_mix_region.clone_from(&m.burn_he_mix_region);
        self.burn_z_mix_region.clone_from(&m.burn_z_mix_region);
        self.okay_to_set_mixing_info = m.okay_to_set_mixing_info;
        self.have_mixing_info = m.have_mixing_info;
        self.have_new_cz_bdy_info = m.have_new_cz_bdy_info;
        self.gradt_excess_alpha = m.gradt_excess_alpha;
        self.gradt_excess_alpha_old = m.gradt_excess_alpha_old;
        self.crystal_core_boundary_mass = m.crystal_core_boundary_mass;
        self.crystal_core_boundary_mass_old = m.crystal_core_boundary_mass_old;
        self.phase_sep_mixing_mass = m.phase_sep_mixing_mass;
    }

    pub(crate) fn restore_pre_hydro(&self, m: &mut MixingAuxState) {
        m.alpha_mlt.clone_from(&self.alpha_mlt);
        m.gradt_excess_alpha = self.gradt_excess_alpha;
        m.gradt_excess_alpha_old = self.gradt_excess_alpha_old;
        m.crystal_core_boundary_mass = self.crystal_core_boundary_mass;
        m.crystal_core_boundary_mass_old = self.crystal_core_boundary_mass_old;
        m.phase_sep_mixing_mass = self.phase_sep_mixing_mass;
        m.okay_to_set_mixing_info = false;
    }

    pub(crate) fn restore_post_hydro(&self, m: &mut MixingAuxState) {
        m.mlt_mixing_type.clone_from(&self.mlt_mixing_type);
        m.cdc.clone_from(&self.cdc);
        m.sig_raw.clone_from(&self.sig_raw);
        m.cz_bdy_dq.clone_from(&self.cz_bdy_dq);
        m.d_mix_non_rotation.clone_from(&self.d_mix_non_rotation);
        m.conv_bdy_loc.clone_from(&self.conv_bdy_loc);
        m.conv_bdy_q.clone_from(&self.conv_bdy_q);
        m.top_conv_bdy.clone_from(&self.top_conv_bdy);
        m.burn_h_conv_region.clone_from(&self.burn_h_conv_region);
        m.burn_he_conv_region.clone_from(&self.burn_he_conv_region);
        m.burn_z_conv_region.clone_from(&self.burn_z_conv_region);
        m.cz_top_mass.clone_from(&self.cz_top_mass);
        m.cz_bot_mass.clone_from(&self.cz_bot_mass);
        m.mix_bdy_loc.clone_from(&self.mix_bdy_loc);
        m.mix_bdy_q.clone_from(&self.mix_bdy_q);
        m.top_mix_bdy.clone_from(&self.top_mix_bdy);
        m.burn_h_mix_region.clone_from(&self.burn_h_mix_region);
        m.burn_he_mix_region.clone_from(&self.burn_he_mix_region);
        m.burn_z_mix_region.clone_from(&self.burn_z_mix_region);
        m.okay_to_set_mixing_info = self.okay_to_set_mixing_info;
        m.have_mixing_info = self.have_mixing_info;
        m.have_new_cz_bdy_info = self.have_new_cz_bdy_info;
    }

    #[must_use]
    pub(crate) fn heap_bytes(&self) -> usize {
        use std::mem::size_of_val;
        size_of_val(self.mlt_mixing_type.as_slice())
            + size_of_val(self.alpha_mlt.as_slice())
            + size_of_val(self.cdc.as_slice())
            + size_of_val(self.sig_raw.as_slice())
            + size_of_val(self.cz_bdy_dq.as_slice())
            + size_of_val(self.d_mix_non_rotation.as_slice())
            + size_of_val(self.conv_bdy_loc.as_slice())
            + size_of_val(self.conv_bdy_q.as_slice())
            + size_of_val(self.top_conv_bdy.as_slice())
            + size_of_val(self.burn_h_conv_region.as_slice())
            + size_of_val(self.burn_he_conv_region.as_slice())
            + size_of_val(self.burn_z_conv_region.as_slice())
            + size_of_val(self.cz_top_mass.as_slice())
            + size_of_val(self.cz_bot_mass.as_slice())
            + size_of_val(self.mix_bdy_loc.as_slice())
            + size_of_val(self.mix_bdy_q.as_slice())
            + size_of_val(self.top_mix_bdy.as_slice())
            + size_of_val(self.burn_h_mix_region.as_slice())
            + size_of_val(self.burn_he_mix_region.as_slice())
            + size_of_val(self.burn_z_mix_region.as_slice())
    }
}

/// Binomial-weighted smoothing over `2 * ns + 1` cells with untouched endpoints
/// (`weighted_smoothing` in `star/private/star_utils.f90:2777-2830`).
pub fn weighted_smoothing(
    dd: &mut [f64],
    ns: usize,
    preserve_sign: bool,
    ddold: &mut [f64],
) {
    let n = dd.len();
    if ns == 0 || n < 3 {
        return;
    }
    ddold[..n].copy_from_slice(dd);

    let nweight = ns;
    let mweight = 2 * nweight + 1;
    let mut weight = vec![0.0_f64; mweight];
    weight[0] = 1.0;
    for i in 0..(mweight - 1) {
        for j in (1..=(i + 1)).rev() {
            weight[j] += weight[j - 1];
        }
    }

    for i in 1..(n - 1) {
        let mut sweight = 0.0_f64;
        dd[i] = 0.0;
        let v0 = ddold[i];
        let j_lo = i.saturating_sub(nweight);
        for j in (j_lo..=i).rev() {
            let k = nweight + j - i;
            if preserve_sign && v0 * ddold[j] <= 0.0 {
                break;
            }
            sweight += weight[k];
            dd[i] += ddold[j] * weight[k];
        }
        let j_hi = (i + nweight).min(n - 1);
        for j in (i + 1)..=j_hi {
            let k = nweight + j - i;
            if preserve_sign && v0 * ddold[j] <= 0.0 {
                break;
            }
            sweight += weight[k];
            dd[i] += ddold[j] * weight[k];
        }
        if sweight > 0.0 {
            let inv_sweight = 1.0 / sweight;
            dd[i] *= inv_sweight;
        }
    }
}

/// Applies [`weighted_smoothing`] independently to each contiguous region where
/// `|dd[i]| >= dd_thresh`, leaving sub-threshold entries and region endpoints
/// unchanged (`threshold_smoothing` in `star/private/star_utils.f90:2833-2887`).
pub fn threshold_smoothing(
    dd: &mut [f64],
    dd_thresh: f64,
    ns: usize,
    preserve_sign: bool,
    ddold: &mut [f64],
) {
    let n = dd.len();
    if ns == 0 || n < 2 {
        return;
    }
    let mut in_region = false;
    let mut i_a = 0_usize;

    for i in 0..n {
        if in_region {
            if dd[i].abs() < dd_thresh {
                let i_b = i - 1;
                if i_b > i_a {
                    weighted_smoothing(
                        &mut dd[i_a..=i_b],
                        ns,
                        preserve_sign,
                        &mut ddold[i_a..=i_b],
                    );
                }
                in_region = false;
            }
        } else if dd[i].abs() >= dd_thresh {
            i_a = i;
            in_region = true;
        }
    }

    if in_region {
        let i_b = n - 1;
        if i_b > i_a {
            weighted_smoothing(
                &mut dd[i_a..=i_b],
                ns,
                preserve_sign,
                &mut ddold[i_a..=i_b],
            );
        }
    }
}

/// Computes `smoothed_brunt_B`, `gradL_composition_term`, and
/// `dominant_iso_for_thermohaline` from `unsmoothed_brunt_B` and zone
/// abundances (`set_grads` in `star/private/hydro_vars.f90:1078-1133`).
pub fn set_grads(model: &mut StarModel) {
    let nz = model.nz;
    if nz == 0 {
        return;
    }
    if model.aux.smoothed_brunt_b.len() != nz {
        model.aux.smoothed_brunt_b.resize(nz, 0.0);
    }
    if model.aux.mixing.unsmoothed_brunt_b.len() != nz {
        model.aux.mixing.resize(nz);
    }

    if model.controls.calculate_brunt_b {
        model
            .aux
            .smoothed_brunt_b
            .copy_from_slice(&model.aux.mixing.unsmoothed_brunt_b[..nz]);
        let ns = model
            .controls
            .mix_wind
            .num_cells_for_smooth_gradl_composition_term;
        if ns > 0 {
            let thresh = model
                .controls
                .mix_wind
                .threshold_for_smooth_gradl_composition_term;
            let mut work = vec![0.0_f64; nz];
            threshold_smoothing(
                &mut model.aux.smoothed_brunt_b[..nz],
                thresh,
                ns,
                false,
                &mut work,
            );
        }
    } else {
        model.aux.smoothed_brunt_b[..nz].fill(0.0);
    }

    if model.controls.use_ledoux_criterion && model.controls.calculate_brunt_b {
        model
            .aux
            .mixing
            .gradl_composition_term
            .copy_from_slice(&model.aux.smoothed_brunt_b[..nz]);
    } else {
        model.aux.mixing.gradl_composition_term[..nz].fill(0.0);
    }

    let species = model.species;
    let dom = &mut model.aux.mixing.dominant_iso_for_thermohaline;
    dom[..nz].fill(0);
    if nz >= 5 && species > 0 {
        let cat = NuclideCatalog::builtin_ref();
        let zn_pairs: Vec<(usize, f64, f64)> = model
            .chem_id
            .iter()
            .take(species)
            .map(|&cid| {
                if let Some(entry) = cat.get_by_id(cid) {
                    (cid, entry.z as f64, entry.z_plus_n as f64)
                } else {
                    (cid, 0.0, 1.0)
                }
            })
            .collect();
        for k in 2..(nz - 2) {
            let mut max_cid = 0_usize;
            let mut max_val = -1.0e99_f64;
            let off_m2 = (k - 2) * species;
            let off_m1 = (k - 1) * species;
            let off_00 = k * species;
            let off_p1 = (k + 1) * species;
            for (j, &(cid, z, a)) in zn_pairs.iter().enumerate() {
                if a <= 0.0 {
                    continue;
                }
                let val = (model.xa[off_m2 + j] + model.xa[off_m1 + j]
                    - model.xa[off_00 + j]
                    - model.xa[off_p1 + j])
                    * (1.0 + z)
                    / a;
                if val > max_val {
                    max_val = val;
                    max_cid = cid;
                }
            }
            dom[k] = max_cid;
        }
        dom[0] = dom[2];
        dom[1] = dom[2];
        dom[nz - 2] = dom[nz - 3];
        dom[nz - 1] = dom[nz - 3];
    }
}

/// Converts a 1-based catalog `chem_id` from `dominant_iso_for_thermohaline`
/// into a [`DrivingSpecies`] for `flexmesa_turb::set_thermohaline`.
#[must_use]
pub fn driving_species_for_cid(cid: usize) -> DrivingSpecies {
    if cid > 0 {
        if let Some(entry) = NuclideCatalog::builtin_ref().get_by_id(cid) {
            if entry.z_plus_n > 0 {
                return DrivingSpecies::new(
                    entry.z as f64,
                    entry.z_plus_n as f64,
                );
            }
        }
    }
    DrivingSpecies::he4()
}
