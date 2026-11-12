// SPDX-License-Identifier: LGPL-3.0-or-later
//! Derived Smolec & Moskalik (2008) / Kuhfuss (1986) physical constants and
//! per-zone / cycle state accumulators for Radial Stellar Pulsation (`RSP` and
//! `RSP2`, `star/private/rsp_def.f90`).

use flexmesa_const::{BOLTZ_SIGMA, L_SUN, M_SUN, PI, R_SUN, STANDARD_CGRAV};

pub use crate::rsp_config::{Rsp2Controls, RspConfig};
use crate::rsp_grid::MesaRanmar;

/// Isotropic Eddington factor $f_{\mathrm{Edd}} = 1/3$ (`rsp_def.f90:33`).
pub const F_EDD_ISOTROPIC: f64 = 1.0 / 3.0;

/// Free-streaming Eddington factor $f_{\mathrm{Edd}} = 1$ (`rsp_def.f90:33`).
pub const F_EDD_FREE_STREAM: f64 = 1.0;

/// Maximum static/nonlinear `RSP` zone capacity (`MAX_NZN = 1501` in `rsp_def.f90:31`).
pub const MAX_RSP_NZN: usize = 1501;

/// Derived physical and numerical constants matching `init_def`,
/// `turn_on_time_weighting`, and `turn_off_time_weighting` in
/// `star/private/rsp_def.f90:141-246`.
#[derive(Debug, Clone, Copy)]
pub struct RspDerivedConsts {
    /// $4\pi$ and $4\pi / 3$.
    pub p4: f64,
    /// $4\pi / 3$.
    pub p43: f64,
    /// CGS gravitational constant $G$.
    pub g: f64,
    /// Stefan-Boltzmann constant $\sigma_{\mathrm{SB}}$.
    pub sig: f64,
    /// Solar luminosity $L_\odot$ ($\mathrm{erg\,s^{-1}}$).
    pub sunl: f64,
    /// Solar mass $M_\odot$ ($\mathrm{g}$).
    pub sunm: f64,
    /// Solar radius $R_\odot$ ($\mathrm{cm}$).
    pub sunr: f64,
    /// Radiative diffusion prefactor $C_L = \frac{4}{3}(4\pi)^2 \sigma_{\mathrm{SB}}$.
    pub cl: f64,
    /// Mixing-length parameter $\alpha$.
    pub alfa: f64,
    /// Scaled turbulent pressure coefficient $\frac{2}{3}\alpha_p$.
    pub alfap: f64,
    /// Eddy-viscosity coefficient $\alpha_m$.
    pub alfam: f64,
    /// Turbulent kinetic-flux coefficient $\alpha_t$.
    pub alfat: f64,
    /// Scaled source coefficient $\frac{1}{2}\sqrt{2/3}\,\alpha_s$.
    pub alfas: f64,
    /// Scaled convective-flux coefficient $\frac{1}{2}\sqrt{2/3}\,\alpha_c$.
    pub alfac: f64,
    /// Scaled Kuhfuss dissipation coefficient $\frac{8}{3}\sqrt{2/3}\,\alpha_d$.
    pub cede: f64,
    /// Scaled radiative loss coefficient $2\sqrt{3}\,\gamma_r$.
    pub gammar: f64,
    /// Artificial viscosity $C_q$.
    pub cq: f64,
    /// Mach threshold $z_{\mathrm{sh}}$.
    pub zsh: f64,
    /// Flux-limiter scale (`0` when `alfa == 0`).
    pub efl0: f64,
    /// Planck/energy opacity factor (`1.0`).
    pub kape_factor: f64,
    /// Flux opacity factor (`1.0`).
    pub kapp_factor: f64,
    /// End-of-step pressure time-centering weight $\theta$.
    pub theta: f64,
    /// $1 - \theta$.
    pub theta1: f64,
    /// Turbulent pressure time weight $\theta_t$.
    pub thetat: f64,
    /// $1 - \theta_t$.
    pub thetat1: f64,
    /// Artificial viscosity time weight $\theta_q$.
    pub thetaq: f64,
    /// $1 - \theta_q$.
    pub thetaq1: f64,
    /// Radiation energy time weight $\theta_e$.
    pub thetae: f64,
    /// $1 - \theta_e$.
    pub thetae1: f64,
    /// Eddy-viscosity time weight $\theta_u$.
    pub thetau: f64,
    /// $1 - \theta_u$.
    pub thetau1: f64,
    /// Radiative luminosity time weight $w_r$.
    pub wtr: f64,
    /// $1 - w_r$.
    pub wtr1: f64,
    /// Convective luminosity time weight $w_c$.
    pub wtc: f64,
    /// $1 - w_c$.
    pub wtc1: f64,
    /// Turbulent luminosity time weight $w_t$.
    pub wtt: f64,
    /// $1 - w_t$.
    pub wtt1: f64,
    /// Turbulent source/sink time weight $\gamma$.
    pub gam: f64,
    /// $1 - \gamma$.
    pub gam1: f64,
    /// 1-based inner nonturbulent zone boundary `IBOTOM = nz / nz_div_ibotom`.
    pub ibotom: usize,
}

impl RspDerivedConsts {
    /// Computes derived Smolec & Moskalik (2008) constants from [`RspConfig`]
    /// and envelope zone count `nz` (`init_def` in `rsp_def.f90:141-186`).
    #[must_use]
    pub fn from_config(cfg: &RspConfig, nz: usize) -> Self {
        let p4 = 4.0 * PI;
        let sqrt_2_div_3 = (2.0_f64 / 3.0).sqrt();
        let alfa = cfg.alfa;
        let ibotom = if cfg.nz_div_ibotom > 0.0 {
            ((nz as f64) / cfg.nz_div_ibotom) as usize
        } else {
            0
        };
        let mut out = Self {
            p4,
            p43: p4 / 3.0,
            g: STANDARD_CGRAV,
            sig: BOLTZ_SIGMA,
            sunl: L_SUN,
            sunm: M_SUN,
            sunr: R_SUN,
            cl: 4.0 * p4 * p4 * BOLTZ_SIGMA / 3.0,
            alfa,
            alfap: cfg.alfap * (2.0 / 3.0),
            alfam: cfg.alfam,
            alfat: cfg.alfat,
            alfas: cfg.alfas * 0.5 * sqrt_2_div_3,
            alfac: cfg.alfac * 0.5 * sqrt_2_div_3,
            cede: cfg.alfad * (8.0 / 3.0) * sqrt_2_div_3,
            gammar: cfg.gammar * 2.0 * 3.0_f64.sqrt(),
            cq: cfg.cq,
            zsh: cfg.zsh,
            efl0: if alfa == 0.0 { 0.0 } else { cfg.efl0 },
            kape_factor: 1.0,
            kapp_factor: 1.0,
            theta: 1.0,
            theta1: 0.0,
            thetat: 1.0,
            thetat1: 0.0,
            thetaq: 1.0,
            thetaq1: 0.0,
            thetae: 1.0,
            thetae1: 0.0,
            thetau: 1.0,
            thetau1: 0.0,
            wtr: 1.0,
            wtr1: 0.0,
            wtc: 1.0,
            wtc1: 0.0,
            wtt: 1.0,
            wtt1: 0.0,
            gam: 1.0,
            gam1: 0.0,
            ibotom,
        };
        out.turn_on_time_weighting(cfg);
        out
    }

    /// Restores user time-centering weights (`turn_on_time_weighting` in `rsp_def.f90:212-232`).
    pub fn turn_on_time_weighting(&mut self, cfg: &RspConfig) {
        (
            self.theta,
            self.thetat,
            self.thetaq,
            self.thetae,
            self.thetau,
            self.wtr,
            self.wtc,
            self.wtt,
            self.gam,
        ) = (
            cfg.theta, cfg.thetat, cfg.thetaq, cfg.thetae, cfg.thetau, cfg.wtr,
            cfg.wtc, cfg.wtt, cfg.gam,
        );
        self.theta1 = 1.0 - self.theta;
        self.thetat1 = 1.0 - self.thetat;
        self.thetaq1 = 1.0 - self.thetaq;
        self.thetae1 = 1.0 - self.thetae;
        self.thetau1 = 1.0 - self.thetau;
        self.wtr1 = 1.0 - self.wtr;
        self.wtc1 = 1.0 - self.wtc;
        self.wtt1 = 1.0 - self.wtt;
        self.gam1 = 1.0 - self.gam;
    }

    /// Forces backward-Euler fully implicit weights (`1.0` / `0.0`, `rsp_def.f90:189-209`).
    pub fn turn_off_time_weighting(&mut self) {
        (
            self.theta,
            self.thetat,
            self.thetaq,
            self.thetae,
            self.thetau,
            self.wtr,
            self.wtc,
            self.wtt,
            self.gam,
        ) = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        (
            self.theta1,
            self.thetat1,
            self.thetaq1,
            self.thetae1,
            self.thetau1,
            self.wtr1,
            self.wtc1,
            self.wtt1,
            self.gam1,
        ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
}

/// Dynamic zone arrays (`k = 0..nz-1`, surface-to-center) and cycle-level
/// energy/work integrators for `RSP` and `RSP2` (`rsp_def.f90:85-116, 365-489`),
/// persisted across `.mod` (`32812`) and `.photo` checkpoints.
#[derive(Debug, Clone)]
pub struct RspState {
    /// Specific turbulent energy $e_t = w^2$ (`et_RSP`), radiation energy `erad_RSP`, and flux `Fr_RSP`.
    pub et_rsp: Vec<f64>,
    /// Specific radiation energy $e_{\mathrm{rad}}$ ($\mathrm{erg\,g^{-1}}$).
    pub erad_rsp: Vec<f64>,
    /// Face radiative flux $F_r = L_r / (4\pi r^2)$ ($\mathrm{erg\,cm^{-2}\,s^{-1}}$).
    pub fr_rsp: Vec<f64>,
    /// Face convective and turbulent luminosities $(L_c, L_t)$ ($\mathrm{erg\,s^{-1}}$).
    pub lc_rsp: Vec<f64>,
    /// Face turbulent luminosity $L_t$ ($\mathrm{erg\,s^{-1}}$).
    pub lt_rsp: Vec<f64>,
    /// Turbulent and artificial-viscosity pressures $(P_{\mathrm{trb}}, P_{\mathrm{vsc}})$ ($\mathrm{dyn\,cm^{-2}}$).
    pub ptrb_rsp: Vec<f64>,
    /// Artificial viscosity pressure $P_{\mathrm{vsc}}$ ($\mathrm{dyn\,cm^{-2}}$).
    pub pvsc_rsp: Vec<f64>,
    /// Specific gas energy $e_{\mathrm{gas}}$ ($\mathrm{erg\,g^{-1}}$) and face scale height $H_p$ ($\mathrm{cm}$).
    pub egas_rsp: Vec<f64>,
    /// Face pressure scale height $H_{p,\mathrm{face}}$ ($\mathrm{cm}$).
    pub hp_face: Vec<f64>,
    /// Active zone specific volume $V_k = 1/\rho_k$ ($\mathrm{cm^3\,g^{-1}}$).
    pub vol_rsp: Vec<f64>,
    /// Zone Eddington factor $f_{\mathrm{Edd},k}$.
    pub f_edd_rsp: Vec<f64>,
    /// Face superadiabatic gradient $Y_k$ and source coefficient $\Pi_k$.
    pub y_face_rsp: Vec<f64>,
    /// Face turbulent source coefficient $\Pi_k$.
    pub pii_rsp: Vec<f64>,
    /// Face eddy-viscosity stress $\chi_k$, radiative coupling `COUPL`, and zone dissipation $E_q$.
    pub chi_rsp: Vec<f64>,
    /// Turbulent radiative cooling `COUPL` ($\mathrm{erg\,g^{-1}\,s^{-1}}$).
    pub coupl_rsp: Vec<f64>,
    /// Zone eddy-viscosity dissipation $E_q$ ($\mathrm{erg\,g^{-1}\,s^{-1}}$).
    pub eq_rsp: Vec<f64>,
    /// Zone thermal expansion $Q_k$, turbulent source $S_k$, and dissipations $(D_k, D_{r,k})$.
    pub qq_rsp: Vec<f64>,
    /// Turbulent buoyancy generation `SOURCE` ($\mathrm{erg\,g^{-1}\,s^{-1}}$).
    pub source_rsp: Vec<f64>,
    /// Kuhfuss turbulent dissipation `DAMP` ($\mathrm{erg\,g^{-1}\,s^{-1}}$).
    pub damp_rsp: Vec<f64>,
    /// Radiative turbulent dissipation `DAMPR` ($\mathrm{erg\,g^{-1}\,s^{-1}}$).
    pub dampr_rsp: Vec<f64>,
    /// Start-of-step `RSP` dynamic, thermodynamic, and flux profiles (`len = nz`).
    pub vol_start: Vec<f64>,
    /// Start-of-step turbulent speed $w_0$ ($\mathrm{cm\,s^{-1}}$) or $e_{t,0}$ ($\mathrm{erg\,g^{-1}}$).
    pub w_start: Vec<f64>,
    /// Start-of-step radiative flux $F_{r,0}$.
    pub fr_start: Vec<f64>,
    /// Start-of-step specific radiation energy $e_{\mathrm{rad},0}$.
    pub erad_start: Vec<f64>,
    /// Start-of-step specific gas energy $e_{\mathrm{gas},0}$.
    pub egas_start: Vec<f64>,
    /// Start-of-step gas pressure $P_{\mathrm{gas},0}$.
    pub pgas_start: Vec<f64>,
    /// Start-of-step radiation pressure $P_{\mathrm{rad},0}$.
    pub prad_start: Vec<f64>,
    /// Start-of-step Rosseland opacity $\kappa_0$.
    pub opacity_start: Vec<f64>,
    /// Start-of-step thermal expansion $Q_0$.
    pub qq_start: Vec<f64>,
    /// Start-of-step convective luminosity $L_{c,0}$.
    pub lc_start: Vec<f64>,
    /// Start-of-step turbulent luminosity $L_{t,0}$.
    pub lt_start: Vec<f64>,
    /// Start-of-step temperature $T_0$, radius $R_0$, velocity $v_0$, sound speed $c_{s,0}$.
    pub t_start_rsp: Vec<f64>,
    /// Start-of-step outer face radius $R_0$ ($\mathrm{cm}$).
    pub r_start_rsp: Vec<f64>,
    /// Start-of-step face velocity $v_0$ ($\mathrm{cm\,s^{-1}}$).
    pub v_start_rsp: Vec<f64>,
    /// Start-of-step adiabatic sound speed $c_{s,0}$ ($\mathrm{cm\,s^{-1}}$).
    pub csound_start_rsp: Vec<f64>,
    /// Start-of-step artificial viscosity $P_{\mathrm{vsc},0}$, turbulent pressure $P_{\mathrm{trb},0}$, $\chi_0$, `COUPL_0`.
    pub pvsc_start: Vec<f64>,
    /// Start-of-step turbulent pressure $P_{\mathrm{trb},0}$.
    pub ptrb_start: Vec<f64>,
    /// Start-of-step eddy-viscosity stress $\chi_0$.
    pub chi_start: Vec<f64>,
    /// Start-of-step radiative coupling `COUPL_0`.
    pub coupl_start: Vec<f64>,
    /// Previous-step pressures `PPP0..PPC0`, `VV0`, and cumulative cycle work integrals `WORK..WORKC`.
    pub ppp0: Vec<f64>,
    /// Previous-step eddy-viscosity work integrand `PPQ0`.
    pub ppq0: Vec<f64>,
    /// Previous-step turbulent-pressure work integrand `PPT0`.
    pub ppt0: Vec<f64>,
    /// Previous-step artificial-viscosity work integrand `PPC0`.
    pub ppc0: Vec<f64>,
    /// Previous-step specific volume `VV0`.
    pub vv0: Vec<f64>,
    /// Cumulative cycle $P\,\mathrm{d}V$ work profile `WORK` ($\mathrm{erg}$).
    pub work: Vec<f64>,
    /// Cumulative cycle eddy-viscosity work profile `WORKQ` ($\mathrm{erg}$).
    pub workq: Vec<f64>,
    /// Cumulative cycle turbulent-pressure work profile `WORKT` ($\mathrm{erg}$).
    pub workt: Vec<f64>,
    /// Cumulative cycle artificial-viscosity work profile `WORKC` ($\mathrm{erg}$).
    pub workc: Vec<f64>,
    /// Deterministic Marsaglia-Zaman `RANMAR` generator (`rsp_def.f90:78`).
    pub rng: MesaRanmar,
    /// Linear radial periods `[P_0, P_1, P_2]` ($\mathrm{s}$) and growth rates `[\eta_0, \eta_1, \eta_2]`.
    pub lina_periods: [f64; 3],
    /// Linear fractional growth rates `[\eta_0, \eta_1, \eta_2]`.
    pub lina_growth_rates: [f64; 3],
    /// Completed nonlinear pulsation cycles (`rsp_num_periods`).
    pub num_periods: usize,
    /// Current nonlinear period, linear reference period, and nominal `rsp_dt` ($\mathrm{s}$).
    pub period: f64,
    /// Reference linear period `PERIODLIN` ($\mathrm{s}$).
    pub period_lin: f64,
    /// Nominal timestep $P / N_{\mathrm{steps/cycle}}$ ($\mathrm{s}$).
    pub rsp_dt: f64,
    /// Cycle excursions $(\Delta R, \Delta M_{\mathrm{bol}})$ and growth rates $(\mathrm{GRPDV}, \mathrm{GREKM}, \langle|\mathrm{GREKM}|\rangle)$.
    pub delta_r: f64,
    /// Bolometric magnitude excursion $\Delta M_{\mathrm{bol}}$.
    pub delta_mag: f64,
    /// Cycle $P\,\mathrm{d}V$ growth rate (`rsp_GRPDV`).
    pub grpdv: f64,
    /// Cycle peak-kinetic-energy growth rate (`rsp_GREKM`).
    pub grekm: f64,
    /// Exponentially smoothed $|\mathrm{GREKM}|$ (`rsp_GREKM_avg_abs`).
    pub grekm_avg_abs: f64,
    /// Surface boundary pressure $P_{\mathrm{surf,atm}}$ ($\mathrm{dyn\,cm^{-2}}$).
    pub psurf_from_atm: f64,
    /// Global energy registers $(E_{\mathrm{tot}}, E_0, E_{\mathrm{start}}, E_{\mathrm{con}}, E_{\mathrm{grv}}, E_{\mathrm{the}}, E_{\mathrm{kin}}, E_{\mathrm{tot,old}})$.
    pub etot: f64,
    /// Initial benchmark energy $E_0$ ($\mathrm{erg}$).
    pub e0: f64,
    /// Start-of-step total energy $E_{\mathrm{start}}$ ($\mathrm{erg}$).
    pub ede_start: f64,
    /// Cumulative radiated boundary energy $E_{\mathrm{con}}$ ($\mathrm{erg}$).
    pub econ: f64,
    /// Gravitational potential energy $E_{\mathrm{grv}}$ ($\mathrm{erg}$).
    pub egrv: f64,
    /// Thermal + radiation + turbulent internal energy $E_{\mathrm{the}}$ ($\mathrm{erg}$).
    pub ethe: f64,
    /// Bulk radial kinetic energy $E_{\mathrm{kin}}$ ($\mathrm{erg}$).
    pub ekin: f64,
    /// Previous-step `StarModel` total energy $E_{\mathrm{tot,old}}$ ($\mathrm{erg}$).
    pub total_energy_old: f64,
    /// Cycle timing and extrema registers $(t_{\mathrm{start}}, t_{\mathrm{cross}}, u_N, u_{N,\mathrm{prev}}, R_{\max/\min}, L_{\max/\min}, E_{K,\max/\min})$.
    pub te_start: f64,
    /// Timestamp of most recent downward $v_1 = 0$ crossing ($\mathrm{s}$).
    pub tt1: f64,
    /// Previous-step surface velocity $v_1$ (`UN`, $\mathrm{cm\,s^{-1}}$).
    pub un: f64,
    /// Two-steps-ago surface velocity (`ULL`, $\mathrm{cm\,s^{-1}}$).
    pub ull: f64,
    /// Cycle kinetic-energy excursion `EKDEL` ($\mathrm{erg}$) and peak speed `VMAX` ($\mathrm{km\,s^{-1}}$).
    pub ekdel: f64,
    /// Cycle peak surface speed `VMAX` ($\mathrm{km\,s^{-1}}$).
    pub vmax: f64,
    /// Cycle maximum radius ($\mathrm{cm}$).
    pub rmax: f64,
    /// Cycle minimum radius ($\mathrm{cm}$).
    pub rmin: f64,
    /// Cycle maximum luminosity ($\mathrm{erg\,s^{-1}}$).
    pub lmax: f64,
    /// Cycle minimum luminosity ($\mathrm{erg\,s^{-1}}$).
    pub lmin: f64,
    /// Cycle maximum kinetic energy ($\mathrm{erg}$).
    pub ekmax: f64,
    /// Cycle minimum kinetic energy ($\mathrm{erg}$).
    pub ekmin: f64,
    /// Previous-cycle maximum kinetic energy ($\mathrm{erg}$).
    pub ekmaxl: f64,
    /// Cycle $P\,\mathrm{d}V$ work sum ($\mathrm{erg}$).
    pub pdvwork: f64,
    /// Courant / diffusion diagnostics, cycle state flags, and step time registers.
    pub min_dr_div_cs: f64,
    /// Minimum radiative diffusion timescale across zones ($\mathrm{s}$).
    pub min_rad_diff_time: f64,
    /// Surface Eddington factor $g_{\mathrm{Edd}}$ (`0.5`) and saved optical depth factor.
    pub g_edd: f64,
    /// Saved `RSP` `tau_factor`.
    pub rsp_tau_factor: f64,
    /// Exact elapsed simulation time `TIME` and `TIME_OLD` ($\mathrm{s}$).
    pub time_sec: f64,
    /// Previous step simulation time `TIME_OLD` ($\mathrm{s}$).
    pub time_old_sec: f64,
    /// Cycle-detection flags (`FIRST`, `INSIDE`), step counter `ID`, turbulent-flux floor `min_k`, and velocity flags.
    pub first: i32,
    /// Cycle phase state `INSIDE` (`1` before first downward crossing, `0` inside cycle).
    pub inside: i32,
    /// Cumulative `HYD` step counter `ID`.
    pub id_step: usize,
    /// Outermost 1-based Smolec index (`k_smolec`) allowed to carry nonzero $L_t$.
    pub min_k_for_turbulent_flux: usize,
    /// Whether initial velocities have been applied (`have_set_velocities`).
    pub have_set_velocities: bool,
    /// Whether velocities were freshly injected on the current step (`just_set_velocities`).
    pub just_set_velocities: bool,
    /// Step-within-cycle counter (`IWORK`).
    pub iwork: usize,
}

impl RspState {
    /// Allocates a zero-initialized [`RspState`] for `nz` zones.
    #[must_use]
    pub fn new(nz: usize) -> Self {
        Self {
            et_rsp: vec![0.0; nz],
            erad_rsp: vec![0.0; nz],
            fr_rsp: vec![0.0; nz],
            lc_rsp: vec![0.0; nz],
            lt_rsp: vec![0.0; nz],
            ptrb_rsp: vec![0.0; nz],
            pvsc_rsp: vec![0.0; nz],
            egas_rsp: vec![0.0; nz],
            hp_face: vec![0.0; nz],
            vol_rsp: vec![0.0; nz],
            f_edd_rsp: vec![F_EDD_ISOTROPIC; nz],
            y_face_rsp: vec![0.0; nz],
            pii_rsp: vec![0.0; nz],
            chi_rsp: vec![0.0; nz],
            coupl_rsp: vec![0.0; nz],
            eq_rsp: vec![0.0; nz],
            qq_rsp: vec![0.0; nz],
            source_rsp: vec![0.0; nz],
            damp_rsp: vec![0.0; nz],
            dampr_rsp: vec![0.0; nz],
            vol_start: vec![0.0; nz],
            w_start: vec![0.0; nz],
            fr_start: vec![0.0; nz],
            erad_start: vec![0.0; nz],
            egas_start: vec![0.0; nz],
            pgas_start: vec![0.0; nz],
            prad_start: vec![0.0; nz],
            opacity_start: vec![0.0; nz],
            qq_start: vec![0.0; nz],
            lc_start: vec![0.0; nz],
            lt_start: vec![0.0; nz],
            t_start_rsp: vec![0.0; nz],
            r_start_rsp: vec![0.0; nz],
            v_start_rsp: vec![0.0; nz],
            csound_start_rsp: vec![0.0; nz],
            pvsc_start: vec![0.0; nz],
            ptrb_start: vec![0.0; nz],
            chi_start: vec![0.0; nz],
            coupl_start: vec![0.0; nz],
            ppp0: vec![0.0; nz],
            ppq0: vec![0.0; nz],
            ppt0: vec![0.0; nz],
            ppc0: vec![0.0; nz],
            vv0: vec![0.0; nz],
            work: vec![0.0; nz],
            workq: vec![0.0; nz],
            workt: vec![0.0; nz],
            workc: vec![0.0; nz],
            rng: MesaRanmar::new(),
            lina_periods: [0.0; 3],
            lina_growth_rates: [0.0; 3],
            num_periods: 0,
            period: 0.0,
            period_lin: 0.0,
            rsp_dt: 0.0,
            delta_r: 0.0,
            delta_mag: 0.0,
            grpdv: 0.0,
            grekm: 0.0,
            grekm_avg_abs: 0.0,
            psurf_from_atm: 0.0,
            etot: 0.0,
            e0: 0.0,
            ede_start: 0.0,
            econ: 0.0,
            egrv: 0.0,
            ethe: 0.0,
            ekin: 0.0,
            total_energy_old: 0.0,
            te_start: 0.0,
            tt1: 0.0,
            un: 0.0,
            ull: 0.0,
            ekdel: 0.0,
            vmax: -1.0e99,
            rmax: -1.0e99,
            rmin: 1.0e99,
            lmax: -1.0e99,
            lmin: 1.0e99,
            ekmax: -1.0e99,
            ekmin: 1.0e99,
            ekmaxl: 0.0,
            pdvwork: 0.0,
            min_dr_div_cs: 1.0e99,
            min_rad_diff_time: 1.0e99,
            g_edd: 0.5,
            rsp_tau_factor: 0.0,
            time_sec: 0.0,
            time_old_sec: 0.0,
            first: 0,
            inside: 1,
            id_step: 0,
            min_k_for_turbulent_flux: 0,
            have_set_velocities: false,
            just_set_velocities: false,
            iwork: 0,
        }
    }
}
