//! Helmholtz (`HELM`) biquintic/bicubic electron-positron equation of state
//! (`eos/private/helm.f90`, `helm_polynomials.f90`, `helm_alloc.f90`, and
//! `eos_helm_eval.f90`).
//!
//! Exports:
//! - [`polynomials`]: 1D and 2D quintic (`psi0..dddpsi2`, `h5`) and cubic
//!   (`xpsi0..xdddpsi1`, `h3`) Hermite basis polynomials.
//! - [`table`]: [`HelmTable`], [`HelmCell`], and [`HelmVertex`] for the
//!   $(\rho Y_e, T)$ Helmholtz free-energy grid.
//! - [`components`]: [`HelmThermoVar`] and radiation, ideal-ion, and
//!   electron-positron thermodynamic component evaluators.
//! - [`coulomb`]: Yakovlev & Shalybkov (1989) uniform-background Coulomb
//!   corrections (`eval_helm_coulomb`).
//! - [`eval`]: Full `helmeos2`, `do_convert_helm_results`, `eval_helm_eos`,
//!   and `eval_helm_eos_zones` evaluators.

pub mod components;
pub mod convert;
pub mod coulomb;
pub mod elec_pos;
pub mod eval;
pub mod polynomials;
pub mod table;

pub use components::{
    HELM_SIFAC, HelmThermoVar, eval_ideal_ions, eval_radiation,
};
pub use convert::{
    do_convert_helm_results, eval_helm_eos, eval_helm_eos_zones,
};
pub use coulomb::eval_helm_coulomb;
pub use elec_pos::eval_electron_positron;
pub use eval::{HelmConfig, helmeos2};
pub use polynomials::{
    CubicBasis1D, QuinticBasis1D, dddpsi0, dddpsi1, dddpsi2, ddpsi0, ddpsi1,
    ddpsi2, dpsi0, dpsi1, dpsi2, h3, h5, psi0, psi1, psi2, xdddpsi0, xdddpsi1,
    xddpsi0, xddpsi1, xdpsi0, xdpsi1, xpsi0, xpsi1,
};
pub use table::{
    HELM_IMAX, HELM_JMAX, HELM_LOGDHI, HELM_LOGDLO, HELM_LOGDSTP,
    HELM_LOGDSTPI, HELM_LOGTHI, HELM_LOGTLO, HELM_LOGTSTP, HELM_LOGTSTPI,
    HelmCell, HelmTable, HelmVertex,
};

// 1-based result indices from `eos/public/helm_def.dek` (`409` slots).
/// Total number of slots in a raw `helmeos2` output array.
pub const NUM_HELM_RESULTS: usize = 409;

/// Total pressure $P_{\mathrm{tot}}$ (`h_ptot = 1`).
pub const H_PTOT: usize = 1;
/// $\partial P_{\mathrm{tot}} / \partial T$ (`h_dpt = 2`).
pub const H_DPT: usize = 2;
/// $\partial P_{\mathrm{tot}} / \partial \rho$ (`h_dpd = 3`).
pub const H_DPD: usize = 3;
/// $\partial P_{\mathrm{tot}} / \partial \bar{A}$ (`h_dpa = 4`).
pub const H_DPA: usize = 4;
/// $\partial P_{\mathrm{tot}} / \partial \bar{Z}$ (`h_dpz = 5`).
pub const H_DPZ: usize = 5;

/// Total specific energy $E_{\mathrm{tot}}$ (`h_etot = 16`).
pub const H_ETOT: usize = 16;
/// $\partial E_{\mathrm{tot}} / \partial T$ (`h_det = 17`).
pub const H_DET: usize = 17;
/// $\partial E_{\mathrm{tot}} / \partial \rho$ (`h_ded = 18`).
pub const H_DED: usize = 18;
/// $\partial E_{\mathrm{tot}} / \partial \bar{A}$ (`h_dea = 19`).
pub const H_DEA: usize = 19;
/// $\partial E_{\mathrm{tot}} / \partial \bar{Z}$ (`h_dez = 20`).
pub const H_DEZ: usize = 20;
/// $\partial^2 E_{\mathrm{tot}} / \partial \rho^2$ (`h_dedd = 21`).
pub const H_DEDD: usize = 21;
/// $\partial^2 E_{\mathrm{tot}} / \partial \rho \partial T$ (`h_dedt = 22`).
pub const H_DEDT: usize = 22;
/// $\partial^2 E_{\mathrm{tot}} / \partial \rho \partial \bar{A}$ (`23`).
pub const H_DEDA: usize = 23;
/// $\partial^2 E_{\mathrm{tot}} / \partial \rho \partial \bar{Z}$ (`24`).
pub const H_DEDZ: usize = 24;

/// Total specific entropy $S_{\mathrm{tot}}$ (`h_stot = 31`).
pub const H_STOT: usize = 31;
/// $\partial S_{\mathrm{tot}} / \partial T$ (`h_dst = 32`).
pub const H_DST: usize = 32;
/// $\partial S_{\mathrm{tot}} / \partial \rho$ (`h_dsd = 33`).
pub const H_DSD: usize = 33;
/// $\partial S_{\mathrm{tot}} / \partial \bar{A}$ (`h_dsa = 34`).
pub const H_DSA: usize = 34;
/// $\partial S_{\mathrm{tot}} / \partial \bar{Z}$ (`h_dsz = 35`).
pub const H_DSZ: usize = 35;
/// $\partial^2 S_{\mathrm{tot}} / \partial \rho^2$ (`h_dsdd = 36`).
pub const H_DSDD: usize = 36;
/// $\partial^2 S_{\mathrm{tot}} / \partial \rho \partial T$ (`h_dsdt = 37`).
pub const H_DSDT: usize = 37;
/// $\partial^2 S_{\mathrm{tot}} / \partial \rho \partial \bar{A}$ (`38`).
pub const H_DSDA: usize = 38;
/// $\partial^2 S_{\mathrm{tot}} / \partial \rho \partial \bar{Z}$ (`39`).
pub const H_DSDZ: usize = 39;
/// $\partial^2 S_{\mathrm{tot}} / \partial T^2$ (`h_dstt = 40`).
pub const H_DSTT: usize = 40;
/// $\partial^2 S_{\mathrm{tot}} / \partial T \partial \bar{A}$ (`h_dsta = 41`).
pub const H_DSTA: usize = 41;
/// $\partial^2 S_{\mathrm{tot}} / \partial T \partial \bar{Z}$ (`h_dstz = 42`).
pub const H_DSTZ: usize = 42;

/// Radiation pressure $P_{\mathrm{rad}}$ (`h_prad = 46`).
pub const H_PRAD: usize = 46;
/// $\partial P_{\mathrm{rad}} / \partial T$ (`h_dpradt = 47`).
pub const H_DPRADT: usize = 47;
/// $\partial P_{\mathrm{rad}} / \partial \rho$ (`h_dpradd = 48`).
pub const H_DPRADD: usize = 48;
/// Radiation energy $E_{\mathrm{rad}}$ (`h_erad = 61`).
pub const H_ERAD: usize = 61;
/// Radiation entropy $S_{\mathrm{rad}}$ (`h_srad = 76`).
pub const H_SRAD: usize = 76;

/// Gas pressure $P_{\mathrm{gas}}$ (`h_pgas = 91`).
pub const H_PGAS: usize = 91;
/// $\partial P_{\mathrm{gas}} / \partial T$ (`h_dpgast = 92`).
pub const H_DPGAST: usize = 92;
/// $\partial P_{\mathrm{gas}} / \partial \rho$ (`h_dpgasd = 93`).
pub const H_DPGASD: usize = 93;
/// $\partial P_{\mathrm{gas}} / \partial \bar{A}$ (`h_dpgasa = 94`).
pub const H_DPGASA: usize = 94;
/// $\partial P_{\mathrm{gas}} / \partial \bar{Z}$ (`h_dpgasz = 95`).
pub const H_DPGASZ: usize = 95;
/// $\partial^2 P_{\mathrm{gas}} / \partial \rho^2$ (`h_dpgasdd = 96`).
pub const H_DPGASDD: usize = 96;
/// $\partial^2 P_{\mathrm{gas}} / \partial \rho \partial T$ (`h_dpgasdt = 97`).
pub const H_DPGASDT: usize = 97;
/// Gas specific energy $E_{\mathrm{gas}}$ (`h_egas = 106`).
pub const H_EGAS: usize = 106;
/// Gas specific entropy $S_{\mathrm{gas}}$ (`h_sgas = 121`).
pub const H_SGAS: usize = 121;

/// Ion number density $n_{\mathrm{ion}}$ (`h_xni = 196`).
pub const H_XNI: usize = 196;
/// Electron degeneracy parameter $\eta_e$ (`h_etaele = 211`).
pub const H_ETAELE: usize = 211;
/// $\partial \eta_e / \partial T$ (`h_detat = 213`).
pub const H_DETAT: usize = 213;
/// $\partial \eta_e / \partial \rho$ (`h_detad = 214`).
pub const H_DETAD: usize = 214;
/// $\partial \eta_e / \partial \bar{A}$ (`h_detaa = 215`).
pub const H_DETAA: usize = 215;
/// $\partial \eta_e / \partial \bar{Z}$ (`h_detaz = 216`).
pub const H_DETAZ: usize = 216;

/// Free electron-positron number density $n_e$ (`h_xne = 275`).
pub const H_XNE: usize = 275;
/// $\partial n_e / \partial T$ (`h_dxnet = 278`).
pub const H_DXNET: usize = 278;
/// $\partial n_e / \partial \rho$ (`h_dxned = 279`).
pub const H_DXNED: usize = 279;
/// $\partial n_e / \partial \bar{A}$ (`h_dxnea = 280`).
pub const H_DXNEA: usize = 280;
/// $\partial n_e / \partial \bar{Z}$ (`h_dxnez = 281`).
pub const H_DXNEZ: usize = 281;

/// Ion coupling parameter $\Gamma$ (`h_plasg = 310`).
pub const H_PLASG: usize = 310;

const X_DGAS: usize = 348;
/// Specific heat at constant pressure $C_P$ (`h_cp = 349`).
pub const H_CP: usize = X_DGAS + 1;
/// $\partial C_P / \partial \rho$ (`h_dcpdd = 350`).
pub const H_DCPDD: usize = X_DGAS + 2;
/// $\partial C_P / \partial T$ (`h_dcpdt = 351`).
pub const H_DCPDT: usize = X_DGAS + 3;
/// $\partial C_P / \partial \bar{A}$ (`h_dcpda = 352`).
pub const H_DCPDA: usize = X_DGAS + 4;
/// $\partial C_P / \partial \bar{Z}$ (`h_dcpdz = 353`).
pub const H_DCPDZ: usize = X_DGAS + 5;

/// Specific heat at constant volume $C_V$ (`h_cv = 354`).
pub const H_CV: usize = X_DGAS + 6;
/// $\partial C_V / \partial \rho$ (`h_dcvdd = 355`).
pub const H_DCVDD: usize = X_DGAS + 7;
/// $\partial C_V / \partial T$ (`h_dcvdt = 356`).
pub const H_DCVDT: usize = X_DGAS + 8;
/// $\partial C_V / \partial \bar{A}$ (`h_dcvda = 357`).
pub const H_DCVDA: usize = X_DGAS + 9;
/// $\partial C_V / \partial \bar{Z}$ (`h_dcvdz = 358`).
pub const H_DCVDZ: usize = X_DGAS + 10;

/// First adiabatic exponent $\Gamma_1$ (`h_gam1 = 359`).
pub const H_GAM1: usize = X_DGAS + 11;
/// $\partial \Gamma_1 / \partial \rho$ (`h_dgam1dd = 360`).
pub const H_DGAM1DD: usize = X_DGAS + 12;
/// $\partial \Gamma_1 / \partial T$ (`h_dgam1dt = 361`).
pub const H_DGAM1DT: usize = X_DGAS + 13;
/// $\partial \Gamma_1 / \partial \bar{A}$ (`h_dgam1da = 362`).
pub const H_DGAM1DA: usize = X_DGAS + 14;
/// $\partial \Gamma_1 / \partial \bar{Z}$ (`h_dgam1dz = 363`).
pub const H_DGAM1DZ: usize = X_DGAS + 15;

/// Second adiabatic exponent $\Gamma_2$ (`h_gam2 = 364`).
pub const H_GAM2: usize = X_DGAS + 16;
/// $\partial \Gamma_2 / \partial \rho$ (`h_dgam2dd = 365`).
pub const H_DGAM2DD: usize = X_DGAS + 17;
/// $\partial \Gamma_2 / \partial T$ (`h_dgam2dt = 366`).
pub const H_DGAM2DT: usize = X_DGAS + 18;
/// $\partial \Gamma_2 / \partial \bar{A}$ (`h_dgam2da = 367`).
pub const H_DGAM2DA: usize = X_DGAS + 19;
/// $\partial \Gamma_2 / \partial \bar{Z}$ (`h_dgam2dz = 368`).
pub const H_DGAM2DZ: usize = X_DGAS + 20;

/// Third adiabatic exponent $\Gamma_3$ (`h_gam3 = 369`).
pub const H_GAM3: usize = X_DGAS + 21;
/// $\partial \Gamma_3 / \partial \rho$ (`h_dgam3dd = 370`).
pub const H_DGAM3DD: usize = X_DGAS + 22;
/// $\partial \Gamma_3 / \partial T$ (`h_dgam3dt = 371`).
pub const H_DGAM3DT: usize = X_DGAS + 23;
/// $\partial \Gamma_3 / \partial \bar{A}$ (`h_dgam3da = 372`).
pub const H_DGAM3DA: usize = X_DGAS + 24;
/// $\partial \Gamma_3 / \partial \bar{Z}$ (`h_dgam3dz = 373`).
pub const H_DGAM3DZ: usize = X_DGAS + 25;

/// Adiabatic temperature gradient $\nabla_{\mathrm{ad}}$ (`h_nabad = 374`).
pub const H_NABAD: usize = X_DGAS + 26;
/// $\partial \nabla_{\mathrm{ad}} / \partial \rho$ (`h_dnabdd = 375`).
pub const H_DNABDD: usize = X_DGAS + 27;
/// $\partial \nabla_{\mathrm{ad}} / \partial T$ (`h_dnabdt = 376`).
pub const H_DNABDT: usize = X_DGAS + 28;
/// $\partial \nabla_{\mathrm{ad}} / \partial \bar{A}$ (`h_dnabda = 377`).
pub const H_DNABDA: usize = X_DGAS + 29;
/// $\partial \nabla_{\mathrm{ad}} / \partial \bar{Z}$ (`h_dnabdz = 378`).
pub const H_DNABDZ: usize = X_DGAS + 30;

/// Relativistic sound speed $c_s$ (`h_cs = 379`).
pub const H_CS: usize = X_DGAS + 31;

/// Temperature susceptibility $\chi_T$ (`h_chit = 384`).
pub const H_CHIT: usize = X_DGAS + 36;
/// $\partial \chi_T / \partial \rho$ (`h_dchitdd = 385`).
pub const H_DCHITDD: usize = X_DGAS + 37;
/// $\partial \chi_T / \partial T$ (`h_dchitdt = 386`).
pub const H_DCHITDT: usize = X_DGAS + 38;
/// $\partial \chi_T / \partial \bar{A}$ (`h_dchitda = 387`).
pub const H_DCHITDA: usize = X_DGAS + 39;
/// $\partial \chi_T / \partial \bar{Z}$ (`h_dchitdz = 388`).
pub const H_DCHITDZ: usize = X_DGAS + 40;

/// Density susceptibility $\chi_\rho$ (`h_chid = 389`).
pub const H_CHID: usize = X_DGAS + 41;
/// $\partial \chi_\rho / \partial \rho$ (`h_dchiddd = 390`).
pub const H_DCHIDDD: usize = X_DGAS + 42;
/// $\partial \chi_\rho / \partial T$ (`h_dchiddt = 391`).
pub const H_DCHIDDT: usize = X_DGAS + 43;
/// $\partial \chi_\rho / \partial \bar{A}$ (`h_dchidda = 392`).
pub const H_DCHIDDA: usize = X_DGAS + 44;
/// $\partial \chi_\rho / \partial \bar{Z}$ (`h_dchiddz = 393`).
pub const H_DCHIDDZ: usize = X_DGAS + 45;

/// Validity flag (`h_valid = 409`).
pub const H_VALID: usize = 409;
