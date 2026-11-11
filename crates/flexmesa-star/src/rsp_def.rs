// SPDX-License-Identifier: LGPL-3.0-or-later
//! Derived Smolec & Moskalik (2008) / Kuhfuss (1986) physical constants and
//! per-zone / cycle state accumulators for Radial Stellar Pulsation (`RSP` and
//! `RSP2`, `star/private/rsp_def.f90`).

use flexmesa_const::{BOLTZ_SIGMA, L_SUN, M_SUN, PI, R_SUN, STANDARD_CGRAV};

pub use crate::rsp_config::{Rsp2Controls, RspConfig};

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
    /// CGS constants $(G, \sigma_{\mathrm{SB}}, L_\odot, M_\odot, R_\odot)$ and $C_L = \frac{4}{3}(4\pi)^2\sigma_{\mathrm{SB}}$.
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
    /// Scaled Smolec & Moskalik (2008) parameters:
    /// $(\alpha, \frac{2}{3}\alpha_p, \alpha_m, \alpha_t, \frac{1}{2}\sqrt{2/3}\alpha_s, \frac{1}{2}\sqrt{2/3}\alpha_c, \frac{8}{3}\sqrt{2/3}\alpha_d, 2\sqrt{3}\gamma_r)$.
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
    /// Artificial viscosity parameters $(C_q, z_{\mathrm{sh}})$, flux limiter `efl0`, and opacity factors.
    pub cq: f64,
    /// Mach threshold $z_{\mathrm{sh}}$.
    pub zsh: f64,
    /// Flux-limiter scale (`0` when `alfa == 0`).
    pub efl0: f64,
    /// Planck/energy opacity factor (`1.0`).
    pub kape_factor: f64,
    /// Flux opacity factor (`1.0`).
    pub kapp_factor: f64,
    /// End-of-step and start-of-step time-centering weights.
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
        self.theta = cfg.theta;
        self.thetat = cfg.thetat;
        self.thetaq = cfg.thetaq;
        self.thetae = cfg.thetae;
        self.thetau = cfg.thetau;
        self.wtr = cfg.wtr;
        self.wtc = cfg.wtc;
        self.wtt = cfg.wtt;
        self.gam = cfg.gam;
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
/// energy/work integrators for `RSP` and `RSP2` (`rsp_def.f90:85-116, 365-444`),
/// persisted across `.mod` (`32812`) and `.photo` checkpoints.
#[derive(Debug, Clone)]
pub struct RspState {
    /// Specific turbulent kinetic energy $e_t = w^2$ in $\mathrm{erg\,g^{-1}}$ (`et_RSP`).
    pub et_rsp: Vec<f64>,
    /// Specific radiation energy $e_{\mathrm{rad}}$ in $\mathrm{erg\,g^{-1}}$ (`erad_RSP`).
    pub erad_rsp: Vec<f64>,
    /// Face radiative flux $F_r = L_r / (4\pi r^2)$ in $\mathrm{erg\,cm^{-2}\,s^{-1}}$ (`Fr_RSP`).
    pub fr_rsp: Vec<f64>,
    /// Face convective and turbulent luminosities $(L_c, L_t)$ in $\mathrm{erg\,s^{-1}}$.
    pub lc_rsp: Vec<f64>,
    /// Face turbulent luminosity $L_t$ ($\mathrm{erg\,s^{-1}}$).
    pub lt_rsp: Vec<f64>,
    /// Turbulent and artificial-viscosity pressures $(P_{\mathrm{trb}}, P_{\mathrm{vsc}})$ in $\mathrm{dyn\,cm^{-2}}$.
    pub ptrb_rsp: Vec<f64>,
    /// Artificial viscosity pressure $P_{\mathrm{vsc}}$ ($\mathrm{dyn\,cm^{-2}}$).
    pub pvsc_rsp: Vec<f64>,
    /// Specific gas energy $e_{\mathrm{gas}}$ ($\mathrm{erg\,g^{-1}}$) and `RSP2` face scale height $H_p$ ($\mathrm{cm}$).
    pub egas_rsp: Vec<f64>,
    /// Face pressure scale height $H_{p,\mathrm{face}}$ ($\mathrm{cm}$).
    pub hp_face: Vec<f64>,
    /// Start-of-step `RSP` thermodynamic and flux profiles (`len = nz`).
    pub vol_start: Vec<f64>,
    /// Start-of-step turbulent speed $w_0$ ($\mathrm{cm\,s^{-1}}$).
    pub w_start: Vec<f64>,
    /// Start-of-step radiative flux $F_{r,0}$ ($\mathrm{erg\,cm^{-2}\,s^{-1}}$).
    pub fr_start: Vec<f64>,
    /// Start-of-step specific radiation energy $e_{\mathrm{rad},0}$ ($\mathrm{erg\,g^{-1}}$).
    pub erad_start: Vec<f64>,
    /// Start-of-step specific gas energy $e_{\mathrm{gas},0}$ ($\mathrm{erg\,g^{-1}}$).
    pub egas_start: Vec<f64>,
    /// Start-of-step gas pressure $P_{\mathrm{gas},0}$ ($\mathrm{dyn\,cm^{-2}}$).
    pub pgas_start: Vec<f64>,
    /// Start-of-step radiation pressure $P_{\mathrm{rad},0}$ ($\mathrm{dyn\,cm^{-2}}$).
    pub prad_start: Vec<f64>,
    /// Start-of-step Rosseland opacity $\kappa_0$ ($\mathrm{cm^2\,g^{-1}}$).
    pub opacity_start: Vec<f64>,
    /// Start-of-step thermal expansion coefficient $Q_0$ ($\mathrm{K^{-1}}$).
    pub qq_start: Vec<f64>,
    /// Start-of-step convective luminosity $L_{c,0}$ ($\mathrm{erg\,s^{-1}}$).
    pub lc_start: Vec<f64>,
    /// Start-of-step turbulent luminosity $L_{t,0}$ ($\mathrm{erg\,s^{-1}}$).
    pub lt_start: Vec<f64>,
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
    /// Linear radial periods `[P_0, P_1, P_2]` in $\mathrm{s}$ and growth rates `[\eta_0, \eta_1, \eta_2]`.
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
    /// Global energy conservation registers $(E_{\mathrm{tot}}, E_0, E_{\mathrm{start}}, E_{\mathrm{con}})$ in $\mathrm{erg}$.
    pub etot: f64,
    /// Initial benchmark energy $E_0$ ($\mathrm{erg}$).
    pub e0: f64,
    /// Start-of-step total energy $E_{\mathrm{start}}$ ($\mathrm{erg}$).
    pub ede_start: f64,
    /// Cumulative radiated boundary energy $E_{\mathrm{con}}$ ($\mathrm{erg}$).
    pub econ: f64,
    /// Cycle timing and extrema registers $(t_{\mathrm{start}}, t_{\mathrm{cross}}, u_N, R_{\max/\min}, L_{\max/\min}, E_{K,\max/\min})$.
    pub te_start: f64,
    /// Timestamp of most recent downward $v_1 = 0$ crossing ($\mathrm{s}$).
    pub tt1: f64,
    /// Previous-step surface velocity $v_1$ ($\mathrm{cm\,s^{-1}}$).
    pub un: f64,
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
    /// Velocity initialization flag and step-within-cycle counter.
    pub have_set_velocities: bool,
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
            ppp0: vec![0.0; nz],
            ppq0: vec![0.0; nz],
            ppt0: vec![0.0; nz],
            ppc0: vec![0.0; nz],
            vv0: vec![0.0; nz],
            work: vec![0.0; nz],
            workq: vec![0.0; nz],
            workt: vec![0.0; nz],
            workc: vec![0.0; nz],
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
            te_start: 0.0,
            tt1: 0.0,
            un: 0.0,
            rmax: -1.0e99,
            rmin: 1.0e99,
            lmax: -1.0e99,
            lmin: 1.0e99,
            ekmax: -1.0e99,
            ekmin: 1.0e99,
            ekmaxl: 0.0,
            pdvwork: 0.0,
            have_set_velocities: false,
            iwork: 0,
        }
    }

    /// Resizes all zone-indexed `RSP` vectors to `nz`.
    pub fn resize(&mut self, nz: usize) {
        for v in [
            &mut self.et_rsp,
            &mut self.erad_rsp,
            &mut self.fr_rsp,
            &mut self.lc_rsp,
            &mut self.lt_rsp,
            &mut self.ptrb_rsp,
            &mut self.pvsc_rsp,
            &mut self.egas_rsp,
            &mut self.hp_face,
            &mut self.vol_start,
            &mut self.w_start,
            &mut self.fr_start,
            &mut self.erad_start,
            &mut self.egas_start,
            &mut self.pgas_start,
            &mut self.prad_start,
            &mut self.opacity_start,
            &mut self.qq_start,
            &mut self.lc_start,
            &mut self.lt_start,
            &mut self.ppp0,
            &mut self.ppq0,
            &mut self.ppt0,
            &mut self.ppc0,
            &mut self.vv0,
            &mut self.work,
            &mut self.workq,
            &mut self.workt,
            &mut self.workc,
        ] {
            v.resize(nz, 0.0);
        }
    }

    /// Packs scalar `RSP` cycle state into a fixed-length `[f64; 34]` array for `.photo` I/O.
    #[must_use]
    pub fn pack_scalars(&self) -> [f64; 34] {
        [
            self.lina_periods[0],
            self.lina_periods[1],
            self.lina_periods[2],
            self.lina_growth_rates[0],
            self.lina_growth_rates[1],
            self.lina_growth_rates[2],
            self.num_periods as f64,
            self.period,
            self.period_lin,
            self.rsp_dt,
            self.delta_r,
            self.delta_mag,
            self.grpdv,
            self.grekm,
            self.grekm_avg_abs,
            self.psurf_from_atm,
            self.etot,
            self.e0,
            self.ede_start,
            self.econ,
            self.te_start,
            self.tt1,
            self.un,
            self.rmax,
            self.rmin,
            self.lmax,
            self.lmin,
            self.ekmax,
            self.ekmin,
            self.ekmaxl,
            self.pdvwork,
            f64::from(u8::from(self.have_set_velocities)),
            self.iwork as f64,
            0.0,
        ]
    }

    /// Unpacks scalar `RSP` cycle state from a `[f64]` slice read during `.photo` I/O.
    pub fn unpack_scalars(&mut self, s: &[f64]) {
        if s.len() < 33 {
            return;
        }
        self.lina_periods = [s[0], s[1], s[2]];
        self.lina_growth_rates = [s[3], s[4], s[5]];
        self.num_periods = s[6].max(0.0) as usize;
        self.period = s[7];
        self.period_lin = s[8];
        self.rsp_dt = s[9];
        self.delta_r = s[10];
        self.delta_mag = s[11];
        self.grpdv = s[12];
        self.grekm = s[13];
        self.grekm_avg_abs = s[14];
        self.psurf_from_atm = s[15];
        self.etot = s[16];
        self.e0 = s[17];
        self.ede_start = s[18];
        self.econ = s[19];
        self.te_start = s[20];
        self.tt1 = s[21];
        self.un = s[22];
        self.rmax = s[23];
        self.rmin = s[24];
        self.lmax = s[25];
        self.lmin = s[26];
        self.ekmax = s[27];
        self.ekmin = s[28];
        self.ekmaxl = s[29];
        self.pdvwork = s[30];
        self.have_set_velocities = s[31] != 0.0;
        self.iwork = s[32].max(0.0) as usize;
    }
}
