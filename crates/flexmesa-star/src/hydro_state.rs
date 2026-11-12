// SPDX-License-Identifier: LGPL-3.0-or-later
//! Auxiliary start-of-step thermodynamic arrays, heating channels, turbulent
//! work/viscosity variables, rotational distortion factors, Rayleigh-Taylor
//! instability (RTI) profiles, and EOS composition partial derivatives on
//! [`crate::StarModel`] (`star_data_step_work.inc` and `star/private/alloc.f90`).

use crate::star_ad::{StarAdExt, StarAdOrder1};

/// Auxiliary hydrodynamic, thermodynamic, and heating state vectors for
/// `flexmesa-star` (`star_data_step_work.inc`).
#[derive(Debug, Clone)]
pub struct HydroAuxState {
    /// Whether the model is currently in an initialization relaxation phase
    /// (`doing_relax` in `star_data_def.f90`).
    pub doing_relax: bool,
    /// Total integrated model energy $E_{\mathrm{tot}}$ in $\mathrm{erg}$
    /// (`total_energy` in `star_data_step_work.inc`).
    pub total_energy: f64,
    /// Cumulative global energy conservation error in $\mathrm{erg}$
    /// (`cumulative_energy_error` in `star_data_step_work.inc`).
    pub cumulative_energy_error: f64,
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
    /// Start-of-step face convective velocity $v_{c,\mathrm{old},k}$ in $\mathrm{cm\,s^{-1}}$ (`mlt_vc_old`).
    pub mlt_vc_old: Vec<f64>,
    /// Rayleigh-Taylor instability Lagrangian diffusion coefficient $\sigma_{\mathrm{RTI},k}$ (`sig_RTI`).
    pub sig_rti: Vec<f64>,
    /// Rayleigh-Taylor instability momentum source term (`source_plus_alpha_RTI`).
    pub source_plus_alpha_rti: Vec<f64>,
    /// Rayleigh-Taylor instability amplitude $\alpha_{\mathrm{RTI},k}$ (`alpha_RTI`).
    pub alpha_rti: Vec<f64>,
    /// Rotational velocity to critical velocity ratio $\omega_k / \omega_{\mathrm{crit},k}$ (`w_div_wc`).
    pub w_div_wc: Vec<f64>,
    /// Rotational pressure distortion correction factor $f_{P,\mathrm{rot},k}$ (`fp_rot`).
    pub fp_rot: Vec<f64>,
    /// Rotational temperature distortion correction factor $f_{T,\mathrm{rot},k}$ (`ft_rot`).
    pub ft_rot: Vec<f64>,
    /// Face specific moment of inertia $i_{\mathrm{rot},k}$ in $\mathrm{cm^2}$ (`i_rot`).
    pub i_rot: Vec<f64>,
    /// Brunt-Väisälä composition buoyancy term $N_{\mu,k}^2$ in $\mathrm{s^{-2}}$ (`brunt_N2_composition_term`).
    pub brunt_n2_composition_term: Vec<f64>,
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
}

impl HydroAuxState {
    /// Allocates a default-initialized [`HydroAuxState`] for `nz` zones and
    /// `species` nuclear isotopes.
    #[must_use]
    pub fn new(nz: usize, species: usize) -> Self {
        Self {
            doing_relax: false,
            total_energy: 0.0,
            cumulative_energy_error: 0.0,
            cp_start: vec![0.0; nz],
            grada_start: vec![0.0; nz],
            chi_t_start: vec![0.0; nz],
            chi_rho_start: vec![0.0; nz],
            eos_frac_pc: vec![0.0; nz],
            latent_ddlnrho: vec![0.0; nz],
            latent_ddlnt: vec![0.0; nz],
            d_eps_grav_dx: vec![0.0; nz * species],
            burn_avg_epsnuc: vec![0.0; nz],
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
            mlt_vc_old: vec![0.0; nz],
            sig_rti: vec![0.0; nz],
            source_plus_alpha_rti: vec![0.0; nz],
            alpha_rti: vec![0.0; nz],
            w_div_wc: vec![0.0; nz],
            fp_rot: vec![1.0; nz],
            ft_rot: vec![1.0; nz],
            i_rot: vec![0.0; nz],
            brunt_n2_composition_term: vec![0.0; nz],
            mass_correction: vec![1.0; nz],
            extra_grav: vec![StarAdOrder1::ZERO; nz],
            uq_ad: vec![StarAdOrder1::ZERO; nz],
            dln_e_dxa_for_partials: vec![0.0; nz * species],
            dln_peos_dxa_for_partials: vec![0.0; nz * species],
            rsp: crate::rsp_def::RspState::new(nz),
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
            &mut self.mlt_vc_old,
            &mut self.sig_rti,
            &mut self.source_plus_alpha_rti,
            &mut self.alpha_rti,
            &mut self.w_div_wc,
            &mut self.i_rot,
            &mut self.brunt_n2_composition_term,
        ] {
            v.resize(nz, 0.0);
        }
        for v in [
            &mut self.fp_rot,
            &mut self.ft_rot,
            &mut self.mass_correction,
        ] {
            v.resize(nz, 1.0);
        }
        for ad in [&mut self.eq_ad, &mut self.extra_grav, &mut self.uq_ad] {
            ad.resize(nz, StarAdOrder1::ZERO);
        }
        let n_sp = nz * species;
        self.d_eps_grav_dx.resize(n_sp, 0.0);
        self.dln_e_dxa_for_partials.resize(n_sp, 0.0);
        self.dln_peos_dxa_for_partials.resize(n_sp, 0.0);
        self.rsp.resize(nz);
    }

    /// Snapshots start-of-step thermodynamic and turbulent velocity arrays.
    pub fn save_start(
        &mut self,
        cp: &[f64],
        grada: &[f64],
        chi_t: &[f64],
        chi_rho: &[f64],
        mlt_vc: &[f64],
    ) {
        self.cp_start.clone_from_slice(cp);
        self.grada_start.clone_from_slice(grada);
        self.chi_t_start.clone_from_slice(chi_t);
        self.chi_rho_start.clone_from_slice(chi_rho);
        self.etrb_start.clone_from(&self.etrb);
        self.mlt_vc_old.clone_from_slice(mlt_vc);
    }
}
