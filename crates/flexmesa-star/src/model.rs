// SPDX-License-Identifier: LGPL-3.0-or-later
//! Canonical 1D stellar structure state (`StarModel`) corresponding to
//! `star_data_def.f90` and `alloc.f90`.
//!
//! # Staggered Grid Convention
//! Zones run from **surface** (`k = 0`, Fortran `k = 1`) to **center**
//! (`k = nz - 1`, Fortran `k = nz`).
//! - **Cell-centered quantities**: `lnd[k]`, `ln_t[k]`, `u[k]`,
//!   `xa[k * species + j]`, `dq[k]`, `dm[k]`, `rmid[k]`, and all EOS/opacity/net
//!   thermodynamic quantities (`rho`, `t`, `pgas`, `prad`, `peos`, `energy`,
//!   `entropy`, `grada`, `cv`, `cp`, `chi_rho`, `chi_t`, `gamma1`, `gamma3`,
//!   `eta`, `gam`, `mu`, `lnfree_e`, `csound`, `opacity`, `non_nuc_neu`,
//!   `eps_nuc`, `eps_grav`).
//! - **Face-centered quantities (outer boundary of cell `k`)**: `ln_r[k]`,
//!   `lum[k]`, `v[k]`, `omega[k]`, `q[k]`, `xq[k]`, `m[k]`, `dm_bar[k]`,
//!   `grav[k]`, `tau[k]`, `gradt[k]`, `gradr[k]`, `gradl[k]`,
//!   `scale_height[k]`, `mlt_vc[k]`, `mlt_d[k]`, `mlt_cdc[k]`, `brunt_b[k]`,
//!   and `brunt_n2[k]`.

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{MixingType, L_SUN, M_SUN, R_SUN, SECYER, STANDARD_CGRAV};
use flexmesa_eos::{EosConfig, NUM_EOS_BASIC_RESULTS};
use flexmesa_kap::KapConfig;

use crate::controls::{StarControls, StarJobControls, VarLayout};
use crate::hydro_state::HydroAuxState;
use crate::star_ad::{StarAdExt, StarAdOrder1};

/// Canonical 1D stellar structure and evolution state (`Star_Info` in
/// `star_data/public/star_data_def.f90` and `star_data_step_work.inc`).
#[derive(Debug, Clone)]
pub struct StarModel {
    /// Active number of Lagrangian zones (`nz`).
    pub nz: usize,
    /// Number of active nuclear species (`species`).
    pub species: usize,
    /// 1-based `chem_id` indices of active species (`len = species`).
    pub chem_id: Vec<usize>,
    /// Isotope names (`"h1"`, `"he4"`, etc., `len = species`).
    pub iso_names: Vec<String>,
    /// Lookup from 1-based `chem_id` to 0-based species index (`None` if absent).
    pub net_iso: Vec<Option<usize>>,
    /// Dynamic solver variable and equation layout (`VarLayout`).
    pub layout: VarLayout,
    /// Job-level controls (`&star_job`).
    pub job: StarJobControls,
    /// Evolutionary and physics controls (`&controls`).
    pub controls: StarControls,
    /// EOS configuration (`EosConfig`).
    pub eos_config: EosConfig,
    /// Opacity configuration (`KapConfig`).
    pub kap_config: KapConfig,

    // --- Scalar Evolutionary State ---
    /// Current evolutionary step counter (`model_number`).
    pub model_number: usize,
    /// Stellar age in years (`time` / `star_age`).
    pub star_age: f64,
    /// Previous step stellar age in years (`time_old`).
    pub star_age_old: f64,
    /// Current timestep $\Delta t$ in seconds (`dt`).
    pub dt: f64,
    /// Next proposed timestep $\Delta t_{\mathrm{next}}$ in seconds (`dt_next`).
    pub dt_next: f64,
    /// Previous timestep $\Delta t_{\mathrm{old}}$ in seconds (`dt_old`).
    pub dt_old: f64,
    /// Previous step timestep-limiter ratio (`dt_limit_ratio_old` for H211 filter).
    pub dt_limit_ratio_old: f64,
    /// Total stellar mass $M_*$ in grams (`mstar`).
    pub mstar: f64,
    /// Start-of-step stellar mass in grams (`mstar_start`).
    pub mstar_start: f64,
    /// Previous step stellar mass in grams (`mstar_old`).
    pub mstar_old: f64,
    /// Enveloped mass exterior to the inner boundary $M_* - M_{\mathrm{center}}$ (`xmstar`).
    pub xmstar: f64,
    /// Mass-loss/accretion rate $\dot{M}$ in $\mathrm{g\,s^{-1}}$ (`mstar_dot`).
    pub mstar_dot: f64,
    /// Inner boundary core mass $M_{\mathrm{center}}$ in grams (`M_center`).
    pub m_center: f64,
    /// Inner boundary core radius $R_{\mathrm{center}}$ in cm (`R_center`).
    pub r_center: f64,
    /// Inner boundary core luminosity $L_{\mathrm{center}}$ in $\mathrm{erg\,s^{-1}}$ (`L_center`).
    pub l_center: f64,
    /// Inner boundary core velocity $v_{\mathrm{center}}$ in $\mathrm{cm\,s^{-1}}$ (`v_center`).
    pub v_center: f64,
    /// Cumulative Newton solver iterations across run (`total_num_solver_iterations`).
    pub total_num_solver_iterations: usize,
    /// Cumulative step retries (`num_retries`).
    pub num_retries: usize,

    // --- Primary State Vectors (len = nz, `xa` len = nz * species) ---
    /// Cell-centered $\ln(\rho/\mathrm{g\,cm^{-3}})$ (`lnd`).
    pub lnd: Vec<f64>,
    /// Cell-centered $\ln(T/\mathrm{K})$ (`lnT`).
    pub ln_t: Vec<f64>,
    /// Face-centered $\ln(R/\mathrm{cm})$ (`lnR`).
    pub ln_r: Vec<f64>,
    /// Face-centered luminosity $L_k$ in $\mathrm{erg\,s^{-1}}$ (`L`).
    pub lum: Vec<f64>,
    /// Face-centered velocity $v_k$ in $\mathrm{cm\,s^{-1}}$ (`v`).
    pub v: Vec<f64>,
    /// Cell-centered Riemann velocity $u_k$ in $\mathrm{cm\,s^{-1}}$ (`u`).
    pub u: Vec<f64>,
    /// Face-centered angular velocity $\Omega_k$ in $\mathrm{rad\,s^{-1}}$ (`omega`).
    pub omega: Vec<f64>,
    /// Face-centered specific angular momentum $j_{\mathrm{rot},k}$ in $\mathrm{cm^2\,s^{-1}}$.
    pub j_rot: Vec<f64>,
    /// Cell-centered species mass fractions $X_{j,k}$ in zone-major layout `xa[k * species + j]`.
    pub xa: Vec<f64>,
    /// Packed hydro state vector `xh[k * nvar_hydro + i]` (`xh`).
    pub xh: Vec<f64>,

    // --- Start-of-Step and Old State Vectors ---
    /// Start-of-step $\ln\rho$ (`lnd_start`).
    pub lnd_start: Vec<f64>,
    /// Start-of-step $\ln T$ (`lnT_start`).
    pub ln_t_start: Vec<f64>,
    /// Start-of-step $\ln R$ (`lnR_start`).
    pub ln_r_start: Vec<f64>,
    /// Start-of-step luminosity $L$ (`L_start`).
    pub lum_start: Vec<f64>,
    /// Start-of-step face velocity $v$ (`v_start`).
    pub v_start: Vec<f64>,
    /// Start-of-step cell velocity $u$ (`u_start`).
    pub u_start: Vec<f64>,
    /// Start-of-step $\ln P_{\mathrm{gas}}$ (`lnPgas_start`).
    pub ln_pgas_start: Vec<f64>,
    /// Start-of-step $\ln P_{\mathrm{eos}}$ (`lnPeos_start`).
    pub ln_peos_start: Vec<f64>,
    /// Start-of-step $\ln E$ (`lnE_start`).
    pub ln_e_start: Vec<f64>,
    /// Start-of-step $\ln S$ (`lnS_start`).
    pub ln_s_start: Vec<f64>,
    /// Start-of-step specific internal energy $e$ (`energy_start`).
    pub energy_start: Vec<f64>,
    /// Start-of-step total pressure $P_{\mathrm{eos}}$ (`Peos_start`).
    pub peos_start: Vec<f64>,
    /// Start-of-step thermal neutrino energy loss rate (`non_nuc_neu_start`).
    pub non_nuc_neu_start: Vec<f64>,
    /// Start-of-step species mass fractions (`xa_start`).
    pub xa_start: Vec<f64>,
    /// Start-of-step packed hydro vector (`xh_start`).
    pub xh_start: Vec<f64>,
    /// Previous-step $\ln\rho$ (`lnd_old`).
    pub lnd_old: Vec<f64>,
    /// Previous-step $\ln T$ (`lnT_old`).
    pub ln_t_old: Vec<f64>,
    /// Previous-step $\ln R$ (`lnR_old`).
    pub ln_r_old: Vec<f64>,
    /// Previous-step luminosity $L$ (`L_old`).
    pub lum_old: Vec<f64>,
    /// Previous-step radial velocity $v$ (`v_old`).
    pub v_old: Vec<f64>,
    /// Previous-step cell velocity $u$ (`u_old`).
    pub u_old: Vec<f64>,
    /// Previous-step species mass fractions (`xa_old`).
    pub xa_old: Vec<f64>,
    /// Previous-step zone mass fractions (`dq_old`).
    pub dq_old: Vec<f64>,

    // --- Lagrangian Mass Grid and Geometry ---
    /// Cell relative mass fraction $\Delta q_k = \Delta m_k / M_{\mathrm{env}}$ (`dq`).
    pub dq: Vec<f64>,
    /// Face interior mass fraction $q_k \in (0, 1]$ (`q[0] = 1.0`).
    pub q: Vec<f64>,
    /// Face exterior mass fraction $1 - q_k$ (`xq`).
    pub xq: Vec<f64>,
    /// Cell mass $\Delta m_k = \Delta q_k M_{\mathrm{env}}$ in grams (`dm`).
    pub dm: Vec<f64>,
    /// Face control-volume mass $\overline{\Delta m}_k$ in grams (`dm_bar`).
    pub dm_bar: Vec<f64>,
    /// Face enclosed mass $m_k = M_{\mathrm{center}} + q_k M_{\mathrm{env}}$ in grams (`m`).
    pub m: Vec<f64>,
    /// Face gravitational mass $m_{\mathrm{grav},k}$ in grams (`m_grav`).
    pub m_grav: Vec<f64>,
    /// Face local gravitational constant $G_k$ in $\mathrm{cm^3\,g^{-1}\,s^{-2}}$ (`cgrav`).
    pub cgrav: Vec<f64>,
    /// Face gravitational acceleration $g_k = G_k m_{\mathrm{grav},k} / r_k^2$ (`grav`).
    pub grav: Vec<f64>,
    /// Face radius $r_k = \exp(\ln R_k)$ in cm (`r`).
    pub r: Vec<f64>,
    /// Cell volumetric mid-radius $r_{\mathrm{mid},k} = [\tfrac{1}{2}(r_k^3 + r_{k+1}^3)]^{1/3}$ (`rmid`).
    pub rmid: Vec<f64>,

    // --- Composition Moments (len = nz) ---
    /// Mean atomic mass number $\bar{A}_k$ (`abar`).
    pub abar: Vec<f64>,
    /// Mean atomic charge $\bar{Z}_k$ (`zbar`).
    pub zbar: Vec<f64>,
    /// Mean squared atomic charge $\overline{Z^2}_k$ (`z2bar`).
    pub z2bar: Vec<f64>,
    /// Mean $Z^{5/3}$ (`z53bar`).
    pub z53bar: Vec<f64>,
    /// Electron mole fraction $Y_{e,k} = \bar{Z}_k / \bar{A}_k$ (`ye`).
    pub ye: Vec<f64>,
    /// Hydrogen mass fraction $X_k$ (`X`).
    pub xh_mass: Vec<f64>,
    /// Helium mass fraction $Y_k$ (`Y`).
    pub yhe_mass: Vec<f64>,
    /// Metal mass fraction $Z_k$ (`Z`).
    pub z_mass: Vec<f64>,

    // --- EOS Thermodynamic Quantities (len = nz) ---
    /// Cell mass density $\rho_k = \exp(\ln\rho_k)$ in $\mathrm{g\,cm^{-3}}$ (`rho`).
    pub rho: Vec<f64>,
    /// Cell temperature $T_k = \exp(\ln T_k)$ in K (`T`).
    pub t: Vec<f64>,
    /// Cell gas pressure $P_{\mathrm{gas},k}$ in $\mathrm{dyn\,cm^{-2}}$ (`Pgas`).
    pub pgas: Vec<f64>,
    /// Cell $\ln P_{\mathrm{gas},k}$ (`lnPgas`).
    pub ln_pgas: Vec<f64>,
    /// Cell radiation pressure $P_{\mathrm{rad},k} = \tfrac{1}{3} a T_k^4$ (`Prad`).
    pub prad: Vec<f64>,
    /// Cell total pressure $P_{\mathrm{eos},k} = P_{\mathrm{gas},k} + P_{\mathrm{rad},k}$ (`Peos`).
    pub peos: Vec<f64>,
    /// Cell $\ln P_{\mathrm{eos},k}$ (`lnPeos`).
    pub ln_peos: Vec<f64>,
    /// Cell specific internal energy $e_k$ in $\mathrm{erg\,g^{-1}}$ (`energy`).
    pub energy: Vec<f64>,
    /// Cell $\ln e_k$ (`lnE`).
    pub ln_e: Vec<f64>,
    /// Cell specific entropy $s_k$ in $\mathrm{erg\,g^{-1}\,K^{-1}}$ (`entropy`).
    pub entropy: Vec<f64>,
    /// Cell $\ln s_k$ (`lnS`).
    pub ln_s: Vec<f64>,
    /// Adiabatic temperature gradient $\nabla_{\mathrm{ad},k} = (\partial\ln T/\partial\ln P)_s$ (`grada`).
    pub grada: Vec<f64>,
    /// Specific heat at constant volume $C_{V,k}$ (`Cv`).
    pub cv: Vec<f64>,
    /// Specific heat at constant pressure $C_{P,k}$ (`Cp`).
    pub cp: Vec<f64>,
    /// Density exponent $\chi_{\rho,k} = (\partial\ln P/\partial\ln\rho)_T$ (`chiRho`).
    pub chi_rho: Vec<f64>,
    /// Temperature exponent $\chi_{T,k} = (\partial\ln P/\partial\ln T)_\rho$ (`chiT`).
    pub chi_t: Vec<f64>,
    /// First adiabatic exponent $\Gamma_{1,k} = (\partial\ln P/\partial\ln\rho)_s$ (`gamma1`).
    pub gamma1: Vec<f64>,
    /// Third adiabatic exponent $\Gamma_{3,k}$ (`gamma3`).
    pub gamma3: Vec<f64>,
    /// Electron degeneracy parameter $\eta_k = \mu_e / (k_{\mathrm{B}} T)$ (`eta`).
    pub eta: Vec<f64>,
    /// Coulomb coupling parameter $\Gamma_{\mathrm{Coul},k}$ (`gam`).
    pub gam: Vec<f64>,
    /// Mean molecular weight $\mu_k$ (`mu`).
    pub mu: Vec<f64>,
    /// Logarithm of free electrons per nucleon $\ln n_{e,\mathrm{free}}$ (`lnfree_e`).
    pub lnfree_e: Vec<f64>,
    /// Adiabatic sound speed $c_{s,k} = \sqrt{\Gamma_{1,k} P_k / \rho_k}$ in $\mathrm{cm\,s^{-1}}$ (`csound`).
    pub csound: Vec<f64>,
    /// Raw 26-slot EOS basic results per zone `eos_res[k]` (`res`).
    pub eos_res: Vec<[f64; NUM_EOS_BASIC_RESULTS]>,
    /// Raw 26-slot $\partial/\partial\ln\rho$ EOS partials per zone (`d_dlnd`).
    pub eos_d_dlnd: Vec<[f64; NUM_EOS_BASIC_RESULTS]>,
    /// Raw 26-slot $\partial/\partial\ln T$ EOS partials per zone (`d_dlnT`).
    pub eos_d_dlnt: Vec<[f64; NUM_EOS_BASIC_RESULTS]>,
    /// Cell $\ln P_{\mathrm{eos},k}$ as [`StarAdOrder1`] (`lnPeos_ad`).
    pub ln_peos_ad: Vec<StarAdOrder1>,
    /// Cell $\ln e_k$ as [`StarAdOrder1`] (`lnE_ad`).
    pub ln_e_ad: Vec<StarAdOrder1>,
    /// Cell $\ln s_k$ as [`StarAdOrder1`] (`lnS_ad`).
    pub ln_s_ad: Vec<StarAdOrder1>,

    // --- Opacity and Optical Depth (len = nz) ---
    /// Cell Rosseland mean opacity $\kappa_k$ in $\mathrm{cm^2\,g^{-1}}$ (`opacity`).
    pub opacity: Vec<f64>,
    /// Cell $\partial\kappa_k / \partial\ln\rho_k$ (`d_opacity_dlnd`).
    pub d_opacity_dlnd: Vec<f64>,
    /// Cell $\partial\kappa_k / \partial\ln T_k$ (`d_opacity_dlnT`).
    pub d_opacity_dlnt: Vec<f64>,
    /// Face optical depth $\tau_k$ (`tau`).
    pub tau: Vec<f64>,
    /// Photospheric optical depth base $\tau_{\mathrm{base}}$ (`tau_base`).
    pub tau_base: f64,

    // --- Thermal Neutrinos and Nuclear Reaction Network ---
    /// Cell thermal neutrino energy loss rate $\epsilon_{\nu,\mathrm{thermal},k}$ (`non_nuc_neu`).
    pub non_nuc_neu: Vec<f64>,
    /// Cell $\partial \epsilon_{\nu,\mathrm{thermal},k} / \partial\ln\rho_k$.
    pub d_non_nuc_neu_dlnd: Vec<f64>,
    /// Cell $\partial \epsilon_{\nu,\mathrm{thermal},k} / \partial\ln T_k$.
    pub d_non_nuc_neu_dlnt: Vec<f64>,
    /// Cell nuclear energy generation rate $\epsilon_{\mathrm{nuc},k}$ in $\mathrm{erg\,g^{-1}\,s^{-1}}$ (`eps_nuc`).
    pub eps_nuc: Vec<f64>,
    /// Cell $\partial \epsilon_{\mathrm{nuc},k} / \partial\ln\rho_k$ (`d_epsnuc_dlnd`).
    pub d_epsnuc_dlnd: Vec<f64>,
    /// Cell $\partial \epsilon_{\mathrm{nuc},k} / \partial\ln T_k$ (`d_epsnuc_dlnT`).
    pub d_epsnuc_dlnt: Vec<f64>,
    /// Cell $\partial \epsilon_{\mathrm{nuc},k} / \partial X_{j,k}$ (`len = nz * species`).
    pub d_epsnuc_dx: Vec<f64>,
    /// Cell nuclear neutrino energy loss rate $\epsilon_{\nu,\mathrm{nuc},k}$.
    pub eps_nuc_neu_total: Vec<f64>,
    /// Cell nuclear abundance rate $\dot{X}_{j,k,\mathrm{nuc}}$ (`len = nz * species`).
    pub dxdt_nuc: Vec<f64>,
    /// Cell $\partial \dot{X}_{j,k,\mathrm{nuc}} / \partial\ln\rho_k$ (`len = nz * species`).
    pub d_dxdt_nuc_dlnd: Vec<f64>,
    /// Cell $\partial \dot{X}_{j,k,\mathrm{nuc}} / \partial\ln T_k$ (`len = nz * species`).
    pub d_dxdt_nuc_dlnt: Vec<f64>,
    /// Cell dense nuclear Jacobian $\partial \dot{X}_{i,k} / \partial X_{j,k}$ (`len = nz * species * species`).
    pub d_dxdt_nuc_dx: Vec<f64>,
    /// Cell gravitational energy rate $\epsilon_{\mathrm{grav},k}$ as [`StarAdOrder1`] (`eps_grav_ad`).
    pub eps_grav_ad: Vec<StarAdOrder1>,
    /// Cell mass-loss advection energy rate $\epsilon_{\dot{M},k}$ (`eps_mdot`).
    pub eps_mdot: Vec<f64>,

    // --- Face Gradients, MLT, Mixing, and Brunt-Väisälä (len = nz) ---
    /// Face actual temperature gradient $\nabla_{T,k}$ (`gradT`).
    pub gradt: Vec<f64>,
    /// Face radiative temperature gradient $\nabla_{\mathrm{rad},k}$ (`gradr`).
    pub gradr: Vec<f64>,
    /// Face Ledoux temperature gradient $\nabla_{L,k} = \nabla_{\mathrm{ad},k} + B_k$ (`gradL`).
    pub gradl: Vec<f64>,
    /// Face pressure scale height $H_{P,k}$ in cm (`scale_height`).
    pub scale_height: Vec<f64>,
    /// Face convective velocity $v_{\mathrm{conv},k}$ in $\mathrm{cm\,s^{-1}}$ (`mlt_vc`).
    pub mlt_vc: Vec<f64>,
    /// Face MLT diffusion coefficient $D_{\mathrm{MLT},k}$ in $\mathrm{cm^2\,s^{-1}}$ (`mlt_D`).
    pub mlt_d: Vec<f64>,
    /// Face convective conductivity $c_{\mathrm{dc},k}$ (`mlt_cdc`).
    pub mlt_cdc: Vec<f64>,
    /// Face convective efficiency $\Gamma_{\mathrm{MLT},k}$ (`mlt_Gamma`).
    pub mlt_gamma: Vec<f64>,
    /// Face final Eulerian diffusion coefficient $D_{\mathrm{mix},k}$ in $\mathrm{cm^2\,s^{-1}}$ (`D_mix`).
    pub d_mix: Vec<f64>,
    /// Face convective velocity after boundary/overshoot resolution (`conv_vel`).
    pub conv_vel: Vec<f64>,
    /// Face Lagrangian diffusion coefficient $\sigma_k = (4\pi r_k^2 \rho_{\mathrm{face},k})^2 D_{\mathrm{mix},k} / \overline{\Delta m}_k$ (`sig`).
    pub sig: Vec<f64>,
    /// Face mixing classification (`mixing_type`).
    pub mixing_type: Vec<MixingType>,
    /// Face composition gradient term $B_k$ (`brunt_B`).
    pub brunt_b: Vec<f64>,
    /// Face Brunt-Väisälä frequency squared $N_k^2$ in $\mathrm{s^{-2}}$ (`brunt_N2`).
    pub brunt_n2: Vec<f64>,
    /// Face $\nabla_{T,k}$ with 3-cell stencil partials (`gradT_ad`).
    pub gradt_ad: Vec<StarAdOrder1>,
    /// Face $\nabla_{\mathrm{rad},k}$ with 3-cell stencil partials (`gradr_ad`).
    pub gradr_ad: Vec<StarAdOrder1>,
    /// Face $\nabla_{L,k}$ with 3-cell stencil partials (`gradL_ad`).
    pub gradl_ad: Vec<StarAdOrder1>,
    /// Face $H_{P,k}$ with 3-cell stencil partials (`scale_height_ad`).
    pub scale_height_ad: Vec<StarAdOrder1>,
    /// Face $v_{\mathrm{conv},k}$ with 3-cell stencil partials (`mlt_vc_ad`).
    pub mlt_vc_ad: Vec<StarAdOrder1>,
    /// Face $D_{\mathrm{MLT},k}$ with 3-cell stencil partials (`mlt_D_ad`).
    pub mlt_d_ad: Vec<StarAdOrder1>,
    /// Face $c_{\mathrm{dc},k}$ with 3-cell stencil partials (`mlt_cdc_ad`).
    pub mlt_cdc_ad: Vec<StarAdOrder1>,

    // --- Surface Boundary Conditions ---
    /// Effective temperature $T_{\mathrm{eff}}$ in K (`Teff`).
    pub teff: f64,
    /// Atmosphere surface $\ln P_{\mathrm{surf}}$ with stencil partials (`lnPeos_surf_ad`).
    pub ln_peos_surf_ad: StarAdOrder1,
    /// Atmosphere surface $\ln T_{\mathrm{surf}}$ with stencil partials (`lnT_surf_ad`).
    pub ln_t_surf_ad: StarAdOrder1,
    /// Auxiliary start-of-step thermodynamic arrays, heating channels,
    /// rotational factors, RTI profiles, and EOS composition partials (`aux`).
    pub aux: HydroAuxState,
}

impl StarModel {
    /// Allocates a zero-initialized [`StarModel`] with `nz` zones and the
    /// specified nuclear species `chem_id` list.
    #[must_use]
    pub fn new(
        nz: usize,
        chem_id: Vec<usize>,
        job: StarJobControls,
        controls: StarControls,
    ) -> Self {
        let species = chem_id.len();
        let catalog = NuclideCatalog::builtin_ref();
        let mut iso_names = Vec::with_capacity(species);
        let mut net_iso = vec![None; 7857];
        for (j, &cid) in chem_id.iter().enumerate() {
            let name = catalog
                .get_by_id(cid)
                .map_or_else(|| format!("iso_{cid}"), |n| n.name.clone());
            iso_names.push(name);
            if cid < net_iso.len() {
                net_iso[cid] = Some(j);
            }
        }
        let layout = VarLayout::with_flags(
            species,
            controls.v_flag,
            controls.u_flag,
            controls.rsp2_flag,
            controls.rti_flag,
        );
        let mstar = controls.initial_mass * M_SUN;
        let kap_config = KapConfig {
            zbase: controls.zbase,
            ..KapConfig::default()
        };

        Self {
            nz,
            species,
            chem_id,
            iso_names,
            net_iso,
            layout: layout.clone(),
            job,
            controls,
            eos_config: EosConfig::default(),
            kap_config,
            model_number: 0,
            star_age: 0.0,
            star_age_old: 0.0,
            dt: 1.0e3 * SECYER,
            dt_next: 1.0e3 * SECYER,
            dt_old: 1.0e3 * SECYER,
            dt_limit_ratio_old: 1.0,
            mstar,
            mstar_start: mstar,
            mstar_old: mstar,
            xmstar: mstar,
            mstar_dot: 0.0,
            m_center: 0.0,
            r_center: 0.0,
            l_center: 0.0,
            v_center: 0.0,
            total_num_solver_iterations: 0,
            num_retries: 0,
            lnd: vec![0.0; nz],
            ln_t: vec![0.0; nz],
            ln_r: vec![0.0; nz],
            lum: vec![0.0; nz],
            v: vec![0.0; nz],
            u: vec![0.0; nz],
            omega: vec![0.0; nz],
            j_rot: vec![0.0; nz],
            xa: vec![0.0; nz * species],
            xh: vec![0.0; nz * layout.nvar_hydro],
            lnd_start: vec![0.0; nz],
            ln_t_start: vec![0.0; nz],
            ln_r_start: vec![0.0; nz],
            lum_start: vec![0.0; nz],
            v_start: vec![0.0; nz],
            u_start: vec![0.0; nz],
            ln_pgas_start: vec![0.0; nz],
            ln_peos_start: vec![0.0; nz],
            ln_e_start: vec![0.0; nz],
            ln_s_start: vec![0.0; nz],
            energy_start: vec![0.0; nz],
            peos_start: vec![0.0; nz],
            non_nuc_neu_start: vec![0.0; nz],
            xa_start: vec![0.0; nz * species],
            xh_start: vec![0.0; nz * layout.nvar_hydro],
            lnd_old: vec![0.0; nz],
            ln_t_old: vec![0.0; nz],
            ln_r_old: vec![0.0; nz],
            lum_old: vec![0.0; nz],
            v_old: vec![0.0; nz],
            u_old: vec![0.0; nz],
            xa_old: vec![0.0; nz * species],
            dq_old: vec![0.0; nz],
            dq: vec![1.0 / (nz.max(1) as f64); nz],
            q: vec![0.0; nz],
            xq: vec![0.0; nz],
            dm: vec![0.0; nz],
            dm_bar: vec![0.0; nz],
            m: vec![0.0; nz],
            m_grav: vec![0.0; nz],
            cgrav: vec![STANDARD_CGRAV; nz],
            grav: vec![0.0; nz],
            r: vec![0.0; nz],
            rmid: vec![0.0; nz],
            abar: vec![0.0; nz],
            zbar: vec![0.0; nz],
            z2bar: vec![0.0; nz],
            z53bar: vec![0.0; nz],
            ye: vec![0.0; nz],
            xh_mass: vec![0.0; nz],
            yhe_mass: vec![0.0; nz],
            z_mass: vec![0.0; nz],
            rho: vec![0.0; nz],
            t: vec![0.0; nz],
            pgas: vec![0.0; nz],
            ln_pgas: vec![0.0; nz],
            prad: vec![0.0; nz],
            peos: vec![0.0; nz],
            ln_peos: vec![0.0; nz],
            energy: vec![0.0; nz],
            ln_e: vec![0.0; nz],
            entropy: vec![0.0; nz],
            ln_s: vec![0.0; nz],
            grada: vec![0.0; nz],
            cv: vec![0.0; nz],
            cp: vec![0.0; nz],
            chi_rho: vec![0.0; nz],
            chi_t: vec![0.0; nz],
            gamma1: vec![0.0; nz],
            gamma3: vec![0.0; nz],
            eta: vec![0.0; nz],
            gam: vec![0.0; nz],
            mu: vec![0.0; nz],
            lnfree_e: vec![0.0; nz],
            csound: vec![0.0; nz],
            eos_res: vec![[0.0; NUM_EOS_BASIC_RESULTS]; nz],
            eos_d_dlnd: vec![[0.0; NUM_EOS_BASIC_RESULTS]; nz],
            eos_d_dlnt: vec![[0.0; NUM_EOS_BASIC_RESULTS]; nz],
            ln_peos_ad: vec![StarAdOrder1::ZERO; nz],
            ln_e_ad: vec![StarAdOrder1::ZERO; nz],
            ln_s_ad: vec![StarAdOrder1::ZERO; nz],
            opacity: vec![0.0; nz],
            d_opacity_dlnd: vec![0.0; nz],
            d_opacity_dlnt: vec![0.0; nz],
            tau: vec![0.0; nz],
            tau_base: 2.0 / 3.0,
            non_nuc_neu: vec![0.0; nz],
            d_non_nuc_neu_dlnd: vec![0.0; nz],
            d_non_nuc_neu_dlnt: vec![0.0; nz],
            eps_nuc: vec![0.0; nz],
            d_epsnuc_dlnd: vec![0.0; nz],
            d_epsnuc_dlnt: vec![0.0; nz],
            d_epsnuc_dx: vec![0.0; nz * species],
            eps_nuc_neu_total: vec![0.0; nz],
            dxdt_nuc: vec![0.0; nz * species],
            d_dxdt_nuc_dlnd: vec![0.0; nz * species],
            d_dxdt_nuc_dlnt: vec![0.0; nz * species],
            d_dxdt_nuc_dx: vec![0.0; nz * species * species],
            eps_grav_ad: vec![StarAdOrder1::ZERO; nz],
            eps_mdot: vec![0.0; nz],
            gradt: vec![0.0; nz],
            gradr: vec![0.0; nz],
            gradl: vec![0.0; nz],
            scale_height: vec![0.0; nz],
            mlt_vc: vec![0.0; nz],
            mlt_d: vec![0.0; nz],
            mlt_cdc: vec![0.0; nz],
            mlt_gamma: vec![0.0; nz],
            d_mix: vec![0.0; nz],
            conv_vel: vec![0.0; nz],
            sig: vec![0.0; nz],
            mixing_type: vec![MixingType::NoMixing; nz],
            brunt_b: vec![0.0; nz],
            brunt_n2: vec![0.0; nz],
            gradt_ad: vec![StarAdOrder1::ZERO; nz],
            gradr_ad: vec![StarAdOrder1::ZERO; nz],
            gradl_ad: vec![StarAdOrder1::ZERO; nz],
            scale_height_ad: vec![StarAdOrder1::ZERO; nz],
            mlt_vc_ad: vec![StarAdOrder1::ZERO; nz],
            mlt_d_ad: vec![StarAdOrder1::ZERO; nz],
            mlt_cdc_ad: vec![StarAdOrder1::ZERO; nz],
            teff: 5777.0,
            ln_peos_surf_ad: StarAdOrder1::ZERO,
            ln_t_surf_ad: StarAdOrder1::ZERO,
            aux: HydroAuxState::new(nz, species),
        }
    }

    /// Returns the species mass-fraction slice `xa[k, 0..species]` for cell `k`.
    #[must_use]
    #[inline]
    pub fn cell_xa(&self, k: usize) -> &[f64] {
        &self.xa[k * self.species..(k + 1) * self.species]
    }

    /// Returns the mutable species mass-fraction slice `xa[k, 0..species]` for cell `k`.
    #[inline]
    pub fn cell_xa_mut(&mut self, k: usize) -> &mut [f64] {
        &mut self.xa[k * self.species..(k + 1) * self.species]
    }

    /// Surface luminosity in solar luminosities ($L_0 / L_\odot$).
    #[must_use]
    #[inline]
    pub fn surface_l_lsun(&self) -> f64 {
        self.lum.first().copied().unwrap_or(0.0) / L_SUN
    }

    /// Surface radius in solar radii ($r_0 / R_\odot$).
    #[must_use]
    #[inline]
    pub fn surface_r_rsun(&self) -> f64 {
        self.ln_r.first().copied().unwrap_or(0.0).exp() / R_SUN
    }
}
