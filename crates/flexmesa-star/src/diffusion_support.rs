// SPDX-License-Identifier: GPL-3.0-or-later
//! Burgers equation assembly, collision coefficients, and diffusion velocity
//! evaluation ported from MESA `26.4.1` `star/private/diffusion_support.f90`.
//!
//! Covers Paquette et al. (1986), Stanton & Murillo (2016), and Caplan, Bauer,
//! & Freeman (2022) resistance coefficients alongside both the Thoul-Hu and
//! CGS (Iben & MacDonald / Bauer) Burgers linear solvers.

use flexmesa_const::{
    AMU, BOLTZM, FINE, FOUR_THIRDS_PI, HBAR, ME, MP, MSUN, ONE_THIRD, PI, PI4,
    QE, RSUN, SECYER,
};
use flexmesa_mtx::{dgetrf, dgetrs};

use crate::job_controls::StarError;
use crate::paquette_coeffs::paquette_coefficients;

/// Mass-fraction limit constant matching `diffusion_support.f90:34`.
pub const X_LIM: f64 = 1.0e-14;

/// Minimum cell mass threshold (`1e3 g = 1 kg`) matching
/// `diffusion_support.f90:35`.
pub const TINY_MASS: f64 = 1.0e3;

/// Tiny mass-fraction floor (`1e-50`) matching `diffusion_support.f90:36`.
pub const TINY_X: f64 = 1.0e-50;

/// Small mass-fraction floor (`1e-20`) matching `diffusion_support.f90:37`.
pub const SMALL_X: f64 = 1.0e-20;

/// Scalar and profile inputs for [`get_matrix_coeffs`], mirroring
/// `diffusion_support.f90:41-76`.
pub struct MatrixCoeffsInput<'a> {
    /// Total number of zones in the stellar model (`nz`).
    pub nz: usize,
    /// Number of ion classes (`nc`).
    pub nc: usize,
    /// Total number of fluids including electrons (`m = nc + 1`).
    pub m: usize,
    /// Zero-based outermost active diffusion zone (`nzlo`).
    pub nzlo: usize,
    /// Zero-based innermost active diffusion zone (`nzhi`).
    pub nzhi: usize,
    /// Whether to use pure-Coulomb Debye-Huckel resistance coefficients.
    pub pure_coulomb: bool,
    /// Whether to use Paquette et al. (1986) ion-ion coefficients instead of
    /// Stanton & Murillo (2016) / Caplan, Bauer, & Freeman (2022).
    pub diffusion_use_paquette: bool,
    /// Whether to blend to Caplan, Bauer, & Freeman (2022) at strong coupling.
    pub diffusion_use_caplan: bool,
    /// Whether to solve the CGS Burgers equations (rather than Thoul-Hu).
    pub diffusion_use_cgs_solver: bool,
    /// Electron degeneracy parameter where CGS thermal diffusion begins to fade.
    pub cgs_thermal_diffusion_eta_full_on: f64,
    /// Electron degeneracy parameter where CGS thermal diffusion turns off.
    pub cgs_thermal_diffusion_eta_full_off: f64,
    /// Whether `extra_diffusion_factor` multiplies `diffusion_factor`.
    pub use_other_diffusion_factor: bool,
    /// Full stellar timestep `s% dt` in seconds (used for `Vlimit` per
    /// `diffusion_support.f90:306`).
    pub star_dt: f64,
    /// Velocity cap (`v_advection_max`, cm/s).
    pub v_advection_max: f64,
    /// Velocity limiter factor (`diffusion_Vlimit`).
    pub diffusion_vlimit: f64,
    /// Outer mass threshold (`Msun`) where velocity limiting is full on.
    pub diffusion_vlimit_dm_full_on: f64,
    /// Outer mass threshold (`Msun`) where velocity limiting is full off.
    pub diffusion_vlimit_dm_full_off: f64,
    /// Outer mass threshold (`Msun`) where artificial diffusion is full on.
    pub diffusion_ad_dm_full_on: f64,
    /// Outer mass threshold (`Msun`) where artificial diffusion is full off.
    pub diffusion_ad_dm_full_off: f64,
    /// Artificial diffusion boost multiplier (`diffusion_AD_boost_factor`).
    pub diffusion_ad_boost_factor: f64,
    /// Multiplier for concentration diffusion terms (`diffusion_SIG_factor`).
    pub diffusion_sig_factor: f64,
    /// Multiplier for advection terms (`diffusion_GT_factor`).
    pub diffusion_gt_factor: f64,
    /// Minimum temperature (K) for radiative acceleration.
    pub min_t_for_radaccel: f64,
    /// Maximum temperature (K) for radiative acceleration.
    pub max_t_for_radaccel: f64,
    /// Per-class constant diffusion multipliers, length `nc`.
    pub diffusion_factor: &'a [f64],
    /// Optional per-zone per-class multipliers, flat `(nz, nc)`.
    pub extra_diffusion_factor: Option<&'a [f64]>,
    /// Atomic mass numbers `A`, length `m` (with `A[nc] = m_e / m_u`).
    pub a: &'a [f64],
    /// Ion charges `Z`, flat `(nz, m)` (with `Z[k * m + nc] = -1.0`).
    pub z: &'a [f64],
    /// Face mass densities (`g cm^-3`), length `nz`.
    pub rho_face: &'a [f64],
    /// Face temperatures (`K`), length `nz`.
    pub t_face: &'a [f64],
    /// `4 * pi * r^2 * rho` at cell faces, length `nz`.
    pub four_pi_r2_rho_face: &'a [f64],
    /// Exterior mass `xm` at cell faces (`g`), length `nz`.
    pub xm_face: &'a [f64],
    /// Zone masses `dm` (`g`), length `nz`.
    pub cell_dm: &'a [f64],
    /// Face-averaged zone masses `dm_bar` (`g`), length `nz`.
    pub dm_bar: &'a [f64],
    /// Pressure log-gradient `d ln P / dr` at cell faces (`cm^-1`), length `nz`.
    pub dln_p_dr_face: &'a [f64],
    /// Temperature log-gradient `d ln T / dr` at cell faces (`cm^-1`), length `nz`.
    pub dln_t_dr_face: &'a [f64],
    /// Density log-gradient `d ln rho / dr` at cell faces (`cm^-1`), length `nz`.
    pub dln_rho_dr_face: &'a [f64],
    /// Cell-center radii `r_mid` (`cm`), length `nz`.
    pub r_mid: &'a [f64],
    /// Gravitational acceleration `s% grav` (`cm s^-2`), length `nz`.
    pub grav: &'a [f64],
    /// Electron degeneracy parameter `s% eta`, length `nz`.
    pub eta: &'a [f64],
    /// Plasma coupling parameter `s% gam`, length `nz`.
    pub gam: &'a [f64],
    /// Gamma/T diffusion ramp coefficients at cell faces, length `nz`.
    pub gamma_t_limit_coeffs_face: &'a [f64],
    /// Face interpolation weights `alfa_face`, length `nz`.
    pub alfa_face: &'a [f64],
    /// Radiative acceleration at faces (`cm s^-2`), flat `(nz, m)`.
    pub rad_accel_face: &'a [f64],
}

/// Output arrays populated by [`get_matrix_coeffs`], mirroring
/// `diffusion_support.f90:67-76`.
pub struct MatrixCoeffsOutput<'a> {
    /// Class mass fractions (clamped in-place to `TINY_X` and electron column
    /// updated), flat `(nz, m)`.
    pub x: &'a mut [f64],
    /// Face-interpolated class mass fractions, flat `(nz, m)`.
    pub x_face: &'a mut [f64],
    /// Class concentrations `C = n_s / n_e`, flat `(nz, m)`.
    pub c: &'a mut [f64],
    /// Concentration-to-mass-fraction ratios `C / X`, flat `(nz, m)`.
    pub c_div_x: &'a mut [f64],
    /// Face ratios `C_face / X_face`, flat `(nz, m)`.
    pub c_div_x_face: &'a mut [f64],
    /// Electric-field pressure coefficient at faces, length `nz`.
    pub e_ap: &'a mut [f64],
    /// Electric-field temperature coefficient at faces, length `nz`.
    pub e_at: &'a mut [f64],
    /// Electric-field radiative coefficient at faces, length `nz`.
    pub e_ar: &'a mut [f64],
    /// Electric-field concentration coefficients at faces, flat `(nz, m)`.
    pub e_ax: &'a mut [f64],
    /// Electric field at faces (`statV cm^-1`), length `nz`.
    pub e_field_face: &'a mut [f64],
    /// Gravity-field pressure coefficient at faces, length `nz`.
    pub g_ap: &'a mut [f64],
    /// Gravity-field temperature coefficient at faces, length `nz`.
    pub g_at: &'a mut [f64],
    /// Gravity-field radiative coefficient at faces, length `nz`.
    pub g_ar: &'a mut [f64],
    /// Gravity-field concentration coefficients at faces, flat `(nz, m)`.
    pub g_ax: &'a mut [f64],
    /// Gravitational field at faces (`cm s^-2`), length `nz`.
    pub g_field_face: &'a mut [f64],
    /// Advective diffusion velocities at faces (`cm s^-1`), flat `(nz, nc)`.
    pub v_advection_face: &'a mut [f64],
    /// Total diffusion velocities at faces (`cm s^-1`), flat `(nz, nc)`.
    pub v_total_face: &'a mut [f64],
    /// Pressure/gravity velocity component at faces (`cm s^-1`), flat `(nz, nc)`.
    pub vln_p_face: &'a mut [f64],
    /// Temperature-gradient velocity component at faces (`cm s^-1`), flat `(nz, nc)`.
    pub vln_t_face: &'a mut [f64],
    /// Radiative-levitation velocity component at faces (`cm s^-1`), flat `(nz, nc)`.
    pub v_rad_face: &'a mut [f64],
    /// Advective flux coefficients `GT_face`, flat `(nz, nc)`.
    pub gt_face: &'a mut [f64],
    /// Self-diffusion coefficients `D_self_face` (`cm^2 s^-1`), flat `(nz, nc)`.
    pub d_self_face: &'a mut [f64],
    /// Artificial diffusion face coefficients `AD_face`, length `nz`.
    pub ad_face: &'a mut [f64],
    /// Concentration flux matrix `SIG_face`, flat `(nz, nc, nc)`.
    pub sig_face: &'a mut [f64],
    /// Concentration diffusion matrix `sigma_lnC`, flat `(nz, nc, nc)`.
    pub sigma_lnc: &'a mut [f64],
}

/// Computes Burgers equation coefficients, diffusion velocities, electric and
/// gravitational fields, and finite-volume flow matrices across active cell
/// faces `nzlo + 1 ..= nzhi` (`diffusion_support.f90:41-241`).
///
/// # Errors
///
/// Returns [`StarError::ConvergenceFailure`] if any face Burgers linear system
/// is singular or non-finite.
#[allow(clippy::similar_names)]
pub fn get_matrix_coeffs(
    inp: &MatrixCoeffsInput<'_>,
    out: &mut MatrixCoeffsOutput<'_>,
) -> Result<(), StarError> {
    let nc = inp.nc;
    let m = inp.m;
    let nzlo = inp.nzlo;
    let nzhi = inp.nzhi;

    for k in nzlo..=nzhi {
        let row = k * m;
        for j in 0..nc {
            if out.x[row + j] < TINY_X {
                out.x[row + j] = TINY_X;
            }
        }
        let mut tmp = 0.0;
        for j in 0..nc {
            tmp += inp.z[row + j] * out.x[row + j] / inp.a[j];
        }
        for j in 0..nc {
            out.c_div_x[row + j] = 1.0 / (inp.a[j] * tmp);
            out.c[row + j] = out.x[row + j] * out.c_div_x[row + j];
        }
        out.c[row + nc] = 1.0;
        let mut ac = 0.0;
        for j in 0..nc {
            ac += inp.a[j] * out.c[row + j];
        }
        out.x[row + nc] = inp.a[nc] / ac;
    }

    let vlimit_dm_full_on = inp.diffusion_vlimit_dm_full_on * MSUN;
    let vlimit_dm_full_off = inp.diffusion_vlimit_dm_full_off * MSUN;
    let vlimit = inp.diffusion_vlimit;

    let mut total_diffusion_factor = vec![0.0_f64; nc];
    let mut c_face = vec![0.0_f64; m];
    let mut z_face = vec![0.0_f64; m];
    let mut dc_dr_face = vec![0.0_f64; m];

    for k in (nzlo + 1)..=nzhi {
        if inp.use_other_diffusion_factor {
            if let Some(extra) = inp.extra_diffusion_factor {
                for j in 0..nc {
                    total_diffusion_factor[j] =
                        inp.diffusion_factor[j] * extra[k * nc + j];
                }
            } else {
                total_diffusion_factor[..nc]
                    .copy_from_slice(&inp.diffusion_factor[..nc]);
            }
        } else {
            total_diffusion_factor[..nc]
                .copy_from_slice(&inp.diffusion_factor[..nc]);
        }

        let mut dlnne_dr_face = 0.0;
        get1_cxzn_face(
            k,
            nc,
            m,
            out.c,
            out.x,
            inp.z,
            inp.a,
            inp.alfa_face[k],
            inp.four_pi_r2_rho_face[k] / inp.dm_bar[k],
            inp.dln_rho_dr_face[k],
            &mut c_face,
            &mut out.x_face[k * m..(k + 1) * m],
            &mut z_face,
            &mut out.c_div_x_face[k * m..(k + 1) * m],
            &mut dc_dr_face,
            &mut dlnne_dr_face,
        );

        let rad = inp.t_face[k] <= inp.max_t_for_radaccel
            && inp.t_face[k] >= inp.min_t_for_radaccel;

        let face_nc_lo = k * nc;
        let face_nc_hi = (k + 1) * nc;
        let face_m_lo = k * m;
        let face_m_hi = (k + 1) * m;
        let face_mat_lo = k * nc * nc;
        let face_mat_hi = (k + 1) * nc * nc;

        get1_coeffs_face(
            k,
            nc,
            m,
            inp.pure_coulomb,
            inp.diffusion_use_paquette,
            inp.diffusion_use_caplan,
            inp.gam[k],
            inp.rho_face[k],
            inp.t_face[k],
            inp.gamma_t_limit_coeffs_face[k],
            inp.v_advection_max,
            inp.dln_p_dr_face[k],
            inp.dln_t_dr_face[k],
            inp.grav[k],
            dlnne_dr_face,
            vlimit_dm_full_on,
            vlimit_dm_full_off,
            vlimit,
            inp.xm_face[k],
            inp.r_mid,
            inp.star_dt,
            inp.a,
            &out.x_face[face_m_lo..face_m_hi],
            &z_face,
            &c_face,
            &total_diffusion_factor,
            &inp.rad_accel_face[face_m_lo..face_m_hi],
            inp.diffusion_use_cgs_solver,
            inp.eta[k],
            inp.cgs_thermal_diffusion_eta_full_on,
            inp.cgs_thermal_diffusion_eta_full_off,
            rad,
            &mut out.v_advection_face[face_nc_lo..face_nc_hi],
            &mut out.vln_p_face[face_nc_lo..face_nc_hi],
            &mut out.vln_t_face[face_nc_lo..face_nc_hi],
            &mut out.v_rad_face[face_nc_lo..face_nc_hi],
            &mut out.e_ap[k],
            &mut out.e_at[k],
            &mut out.e_ar[k],
            &mut out.e_ax[face_m_lo..face_m_hi],
            &mut out.g_ap[k],
            &mut out.g_at[k],
            &mut out.g_ar[k],
            &mut out.g_ax[face_m_lo..face_m_hi],
            &mut out.sigma_lnc[face_mat_lo..face_mat_hi],
        )?;

        if inp.diffusion_use_cgs_solver {
            let mut ef =
                BOLTZM * inp.t_face[k] * inp.dln_t_dr_face[k] * out.e_at[k]
                    + AMU * inp.grav[k] * out.e_ap[k];
            for j in 0..nc {
                ef += BOLTZM
                    * inp.t_face[k]
                    * out.e_ax[face_m_lo + j]
                    * (dc_dr_face[j] / c_face[j] + dlnne_dr_face);
            }
            out.e_field_face[k] = ef / QE;
            out.g_field_face[k] = inp.grav[k];
        } else {
            let mut ef = out.e_ap[k] * inp.dln_p_dr_face[k]
                + out.e_at[k] * inp.dln_t_dr_face[k];
            let mut gf = out.g_ap[k] * inp.dln_p_dr_face[k]
                + out.g_at[k] * inp.dln_t_dr_face[k];
            for j in 0..nc {
                if c_face[j] < 1.0e-20 {
                    continue;
                }
                ef += out.e_ax[face_m_lo + j] * dc_dr_face[j] / c_face[j];
                gf += out.g_ax[face_m_lo + j] * dc_dr_face[j] / c_face[j];
            }
            let mut tmp_f = 0.0;
            for j in 0..nc {
                tmp_f += z_face[j] * out.x_face[face_m_lo + j] / inp.a[j];
            }
            out.g_field_face[k] =
                gf * 1.144e-40 * inp.t_face[k] * tmp_f / (AMU * AMU);
            out.e_field_face[k] =
                ef * 1.144e-40 * inp.t_face[k] * tmp_f / (QE * AMU);
        }

        for i in 0..nc {
            let mut vtot = out.v_advection_face[face_nc_lo + i];
            for j in 0..nc {
                vtot -= out.sigma_lnc[face_mat_lo + i * nc + j] * dc_dr_face[j]
                    / c_face[j];
            }
            out.v_total_face[face_nc_lo + i] = vtot;
        }
    }

    let mut sum_dm = inp.cell_dm[nzlo];
    let ad_dm_full_on = inp.diffusion_ad_dm_full_on * MSUN;
    let ad_dm_full_off = inp.diffusion_ad_dm_full_off * MSUN;
    let ad_boost_factor = inp.diffusion_ad_boost_factor;
    let sig_factor = inp.diffusion_sig_factor;
    let gt_factor = inp.diffusion_gt_factor;

    for k in (nzlo + 1)..=nzhi {
        let face_nc_lo = k * nc;
        let face_nc_hi = (k + 1) * nc;
        let face_m_lo = k * m;
        let face_m_hi = (k + 1) * m;
        let face_mat_lo = k * nc * nc;
        let face_mat_hi = (k + 1) * nc * nc;

        get1_flow_coeffs(
            nc,
            &out.v_advection_face[face_nc_lo..face_nc_hi],
            sig_factor,
            gt_factor,
            &out.sigma_lnc[face_mat_lo..face_mat_hi],
            inp.four_pi_r2_rho_face[k],
            inp.dm_bar[k],
            &out.c_div_x_face[face_m_lo..face_m_hi],
            &mut out.gt_face[face_nc_lo..face_nc_hi],
            &mut out.d_self_face[face_nc_lo..face_nc_hi],
            &mut out.sig_face[face_mat_lo..face_mat_hi],
        );

        if sum_dm >= ad_dm_full_off {
            out.ad_face[k] = 0.0;
        } else {
            let mut sigmax = 0.0;
            for j in 0..nc {
                let diag = out.sig_face[face_mat_lo + j * nc + j];
                if diag > sigmax {
                    sigmax = diag;
                }
            }
            out.ad_face[k] = ad_boost_factor * sigmax;
            if sum_dm > ad_dm_full_on {
                out.ad_face[k] *= (sum_dm - ad_dm_full_off)
                    / (ad_dm_full_on - ad_dm_full_off);
            }
        }
        sum_dm += inp.cell_dm[k];
    }

    // Copy face nzlo+1 values to nzlo for plotting, matching
    // `diffusion_support.f90:228-239`.
    let src_nc = (nzlo + 1) * nc;
    let dst_nc = nzlo * nc;
    let src_mat = (nzlo + 1) * nc * nc;
    let dst_mat = nzlo * nc * nc;
    for j in 0..nc {
        out.d_self_face[dst_nc + j] = out.d_self_face[src_nc + j];
        out.v_advection_face[dst_nc + j] = out.v_advection_face[src_nc + j];
        out.v_total_face[dst_nc + j] = out.v_total_face[src_nc + j];
        out.vln_p_face[dst_nc + j] = out.vln_p_face[src_nc + j];
        out.vln_t_face[dst_nc + j] = out.vln_t_face[src_nc + j];
        out.v_rad_face[dst_nc + j] = out.v_rad_face[src_nc + j];
        out.gt_face[dst_nc + j] = out.gt_face[src_nc + j];
        for i in 0..nc {
            out.sig_face[dst_mat + i * nc + j] =
                out.sig_face[src_mat + i * nc + j];
        }
    }

    Ok(())
}

/// Evaluates face-interpolated `X`, `Z`, `C`, `C/X`, `dC/dr`, and `d ln n_e / dr`
/// at cell face `k` (`diffusion_support.f90:314-362`).
#[allow(clippy::too_many_arguments)]
pub fn get1_cxzn_face(
    k: usize,
    nc: usize,
    m: usize,
    c: &[f64],
    x: &[f64],
    z: &[f64],
    a: &[f64],
    alfa: f64,
    d_dr_factor: f64,
    dln_rho_dr_face: f64,
    c_face: &mut [f64],
    x_face_k: &mut [f64],
    z_face: &mut [f64],
    c_div_x_face_k: &mut [f64],
    dc_dr_face: &mut [f64],
    dlnne_dr_face: &mut f64,
) {
    let beta = 1.0 - alfa;
    let row_k = k * m;
    let row_km1 = (k - 1) * m;

    for j in 0..m {
        x_face_k[j] = alfa * x[row_k + j] + beta * x[row_km1 + j];
        z_face[j] = alfa * z[row_k + j] + beta * z[row_km1 + j];
    }

    let mut tmp = 0.0;
    for j in 0..nc {
        tmp += z_face[j] * x_face_k[j] / a[j];
    }
    for j in 0..m {
        c_div_x_face_k[j] = 1.0 / (a[j] * tmp);
        c_face[j] = x_face_k[j] * c_div_x_face_k[j];
        dc_dr_face[j] = (c[row_km1 + j] - c[row_k + j]) * d_dr_factor;
    }

    let mut tmp1 = 0.0;
    let mut tmp2 = 0.0;
    for j in 0..nc {
        tmp1 += z[row_k + j] * x[row_k + j] / a[j];
        tmp2 += z[row_km1 + j] * x[row_km1 + j] / a[j];
    }
    let dlntmp_dr_face = ((tmp2 - tmp1) / tmp) * d_dr_factor;
    *dlnne_dr_face = dln_rho_dr_face + dlntmp_dr_face;
}

/// Evaluates Burgers resistance coefficients, gradient coefficients, and
/// diffusion velocities at a single cell face `k`
/// (`diffusion_support.f90:244-311`).
///
/// # Errors
///
/// Returns [`StarError::ConvergenceFailure`] if the Burgers linear system fails
/// to factor or solve.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn get1_coeffs_face(
    k: usize,
    nc: usize,
    m: usize,
    pure_coulomb: bool,
    diffusion_use_paquette: bool,
    diffusion_use_caplan: bool,
    gamma: f64,
    rho_face: f64,
    t_face: f64,
    gamma_t_limit_coeff_face: f64,
    v_advection_max: f64,
    dln_p_dr_face: f64,
    dln_t_dr_face: f64,
    grav: f64,
    dlnne_dr_face: f64,
    vlimit_dm_full_on: f64,
    vlimit_dm_full_off: f64,
    vlimit: f64,
    xm_face: f64,
    r_mid: &[f64],
    star_dt: f64,
    a: &[f64],
    x_face: &[f64],
    z_face: &[f64],
    c_face: &[f64],
    diffusion_factor: &[f64],
    rad_accel_face: &[f64],
    use_cgs_solver: bool,
    eta: f64,
    eta_on: f64,
    eta_off: f64,
    rad: bool,
    v_advection_face: &mut [f64],
    vln_p_face: &mut [f64],
    vln_t_face: &mut [f64],
    v_rad_face: &mut [f64],
    e_ap: &mut f64,
    e_at: &mut f64,
    e_ar: &mut f64,
    e_ax: &mut [f64],
    g_ap: &mut f64,
    g_at: &mut f64,
    g_ar: &mut f64,
    g_ax: &mut [f64],
    sigma_lnc: &mut [f64],
) -> Result<(), StarError> {
    let mut ap = vec![0.0_f64; m];
    let mut at = vec![0.0_f64; m];
    let mut ar = vec![0.0_f64; m];
    let mut kappa_st = vec![0.0_f64; m * m];
    let mut zdiff = vec![0.0_f64; m * m];
    let mut zdiff1 = vec![0.0_f64; m * m];
    let mut zdiff2 = vec![0.0_f64; m * m];
    let mut ax = vec![0.0_f64; m * m];

    get1_burgers_coeffs(
        nc,
        m,
        a,
        z_face,
        x_face,
        c_face,
        rho_face,
        t_face,
        pure_coulomb,
        diffusion_use_paquette,
        diffusion_use_caplan,
        gamma,
        &mut kappa_st,
        &mut zdiff,
        &mut zdiff1,
        &mut zdiff2,
    );

    get1_gradient_coeffs(
        k,
        m,
        a,
        z_face,
        x_face,
        c_face,
        rho_face,
        t_face,
        use_cgs_solver,
        eta,
        eta_on,
        eta_off,
        rad,
        rad_accel_face,
        &kappa_st,
        &zdiff,
        &zdiff1,
        &zdiff2,
        &mut ap,
        &mut at,
        &mut ar,
        &mut ax,
        e_ap,
        e_at,
        e_ar,
        e_ax,
        g_ap,
        g_at,
        g_ar,
        g_ax,
    )?;

    get1_diffusion_velocities(
        k,
        nc,
        m,
        &ap,
        &at,
        &ar,
        &ax,
        rho_face,
        t_face,
        dln_p_dr_face,
        dln_t_dr_face,
        grav,
        dlnne_dr_face,
        x_face,
        vlimit_dm_full_on,
        vlimit_dm_full_off,
        vlimit,
        xm_face,
        r_mid,
        star_dt,
        gamma_t_limit_coeff_face,
        v_advection_max,
        diffusion_factor,
        use_cgs_solver,
        v_advection_face,
        vln_p_face,
        vln_t_face,
        v_rad_face,
        sigma_lnc,
    );

    Ok(())
}

/// Computes normalized Burgers resistance matrices `(kappa_st, Zdiff, Zdiff1,
/// Zdiff2)` in row-major `(m, m)` layout (`diffusion_support.f90:365-479`).
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn get1_burgers_coeffs(
    nc: usize,
    m: usize,
    a: &[f64],
    z: &[f64],
    x: &[f64],
    c: &[f64],
    rho: f64,
    t: f64,
    pure_coulomb: bool,
    diffusion_use_paquette: bool,
    diffusion_use_caplan: bool,
    gamma: f64,
    kappa_st: &mut [f64],
    zdiff: &mut [f64],
    zdiff1: &mut [f64],
    zdiff2: &mut [f64],
) {
    let mut charge = vec![0.0_f64; m];
    for i in 0..nc {
        charge[i] = 1.0_f64.max(z[i]);
    }
    charge[nc] = z[nc];

    if !pure_coulomb {
        let mut na = vec![0.0_f64; m];
        for i in 0..nc {
            na[i] = rho * x[i] / (a[i] * AMU);
        }
        let mut ne = 0.0;
        for i in 0..nc {
            ne += charge[i] * na[i];
        }
        na[nc] = ne;

        let mut ddiff = vec![0.0_f64; m * m];
        let mut kdiff = vec![0.0_f64; m * m];
        let mut ath = vec![0.0_f64; m * m];

        paquette_coefficients(
            rho, t, m, a, &charge, &na, &mut ddiff, &mut kdiff, zdiff, zdiff1,
            zdiff2, &mut ath,
        );

        if !diffusion_use_paquette {
            let gamlo = 3.0;
            let gamhi = 10.0;
            let mut ddiff_caplan = vec![0.0_f64; nc];
            if gamma < gamlo || !diffusion_use_caplan {
                let _kappa_sm = get_sm_coeffs(
                    nc, m, rho, t, a, &charge, &na, &mut kdiff, zdiff, zdiff1,
                    zdiff2,
                );
            } else if gamma >= gamlo && gamma <= gamhi {
                let alfa = (gamhi - gamma) / (gamhi - gamlo);
                let mut kdiff2 = kdiff.clone();
                let _kappa_sm = get_sm_coeffs(
                    nc, m, rho, t, a, &charge, &na, &mut kdiff, zdiff, zdiff1,
                    zdiff2,
                );
                get_cbf_coeffs(
                    nc,
                    m,
                    rho,
                    t,
                    a,
                    &charge,
                    &na,
                    gamma,
                    &mut ddiff_caplan,
                    &mut kdiff2,
                );
                for idx in 0..(m * m) {
                    kdiff[idx] = alfa * kdiff[idx] + (1.0 - alfa) * kdiff2[idx];
                }
            } else {
                get_cbf_coeffs(
                    nc,
                    m,
                    rho,
                    t,
                    a,
                    &charge,
                    &na,
                    gamma,
                    &mut ddiff_caplan,
                    &mut kdiff,
                );
            }
        }

        let norm = 1.41e-25 * flexmesa_math::pow(t, -1.5) * na[nc] * na[nc];
        for idx in 0..(m * m) {
            kappa_st[idx] = kdiff[idx] / norm;
        }
        return;
    }

    let mut ac = 0.0;
    for i in 0..m {
        ac += a[i] * c[i];
    }
    let ne = rho / (MP * ac);
    let mut ni = 0.0;
    for i in 0..nc {
        ni += c[i] * ne;
    }
    let ao = flexmesa_math::pow(0.23873 / ni, ONE_THIRD);
    let mut cz = 0.0;
    for i in 0..m {
        cz += c[i] * charge[i] * charge[i];
    }
    let lambdad = 6.9010 * (t / (ne * cz)).sqrt();
    let lambda = lambdad.max(ao);

    let mut cl = vec![0.0_f64; m * m];
    for i in 0..m {
        for j in 0..m {
            let xij = 2.3939e3 * t * lambda / (charge[i] * charge[j]).abs();
            cl[i * m + j] = 0.81245
                * flexmesa_math::log1p(0.18769 * flexmesa_math::pow(xij, 1.2));
        }
    }

    for i in 0..m {
        for j in 0..m {
            let idx = i * m + j;
            zdiff[idx] = 0.6;
            zdiff1[idx] = 1.3;
            zdiff2[idx] = 2.0;
            kappa_st[idx] = cl[idx]
                * (a[i] * a[j] / (a[i] + a[j])).sqrt()
                * c[i]
                * c[j]
                * charge[i]
                * charge[i]
                * charge[j]
                * charge[j];
        }
    }
}

/// Solves for the Burgers gradient coefficients `(AP, AT, AR, AX)` and field
/// coefficients using either the CGS solver (with degeneracy blending across
/// `[eta_on, eta_off]`) or the Thoul-Hu solver (`diffusion_support.f90:482-618`).
///
/// # Errors
///
/// Returns [`StarError::ConvergenceFailure`] if any linear solve fails.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn get1_gradient_coeffs(
    k: usize,
    m: usize,
    a: &[f64],
    z: &[f64],
    x: &[f64],
    c: &[f64],
    rho: f64,
    t: f64,
    use_cgs_solver: bool,
    eta: f64,
    eta_on: f64,
    eta_off: f64,
    rad: bool,
    rad_accel: &[f64],
    kappa_st: &[f64],
    zdiff: &[f64],
    zdiff1: &[f64],
    zdiff2: &[f64],
    ap: &mut [f64],
    at: &mut [f64],
    ar: &mut [f64],
    ax: &mut [f64],
    e_ap: &mut f64,
    e_at: &mut f64,
    e_ar: &mut f64,
    e_ax: &mut [f64],
    g_ap: &mut f64,
    g_at: &mut f64,
    g_ar: &mut f64,
    g_ax: &mut [f64],
) -> Result<(), StarError> {
    let nc = m - 1;
    let mut charge = vec![0.0_f64; m];
    for i in 0..nc {
        charge[i] = 1.0_f64.max(z[i]);
    }
    charge[nc] = z[nc];

    if use_cgs_solver {
        let mut nd = vec![0.0_f64; m];
        for i in 0..nc {
            nd[i] = rho * x[i] / (a[i] * AMU);
            nd[nc] += nd[i] * charge[i];
        }
        let scale = 1.41e-25 * flexmesa_math::pow(t, -1.5) * nd[nc] * nd[nc];
        let mut kdiff = vec![0.0_f64; m * m];
        for idx in 0..(m * m) {
            kdiff[idx] = kappa_st[idx] * scale;
        }

        if eta < eta_on {
            solve_burgers_cgs_with_thermal(
                2 * m + 1,
                m,
                a,
                &charge,
                &nd,
                rad_accel,
                rad,
                &kdiff,
                zdiff,
                zdiff1,
                zdiff2,
                ap,
                at,
                ar,
                ax,
                e_ap,
                e_at,
                e_ar,
                e_ax,
            )
            .map_err(|err| {
                StarError::ConvergenceFailure(format!(
                    "solve_burgers_cgs_with_thermal failed at zone {k}: {err}"
                ))
            })?;
        } else if eta > eta_off {
            solve_burgers_cgs_no_thermal(
                m + 1,
                m,
                a,
                &charge,
                &nd,
                rad_accel,
                rad,
                &kdiff,
                ap,
                at,
                ar,
                ax,
                e_ap,
                e_at,
                e_ar,
                e_ax,
            )
            .map_err(|err| {
                StarError::ConvergenceFailure(format!(
                    "solve_burgers_cgs_no_thermal failed at zone {k}: {err}"
                ))
            })?;
        } else {
            let alfa = (eta - eta_on) / (eta_off - eta_on);
            let beta = 1.0 - alfa;

            let mut ap1 = vec![0.0_f64; m];
            let mut at1 = vec![0.0_f64; m];
            let mut ar1 = vec![0.0_f64; m];
            let mut ax1 = vec![0.0_f64; m * m];
            let mut e_ap1 = 0.0;
            let mut e_at1 = 0.0;
            let mut e_ar1 = 0.0;
            let mut e_ax1 = vec![0.0_f64; m];

            solve_burgers_cgs_no_thermal(
                m + 1,
                m,
                a,
                &charge,
                &nd,
                rad_accel,
                rad,
                &kdiff,
                &mut ap1,
                &mut at1,
                &mut ar1,
                &mut ax1,
                &mut e_ap1,
                &mut e_at1,
                &mut e_ar1,
                &mut e_ax1,
            )
            .map_err(|err| {
                StarError::ConvergenceFailure(format!(
                    "solve_burgers_cgs_no_thermal failed in eta blend at zone {k}: {err}"
                ))
            })?;

            let mut ap2 = vec![0.0_f64; m];
            let mut at2 = vec![0.0_f64; m];
            let mut ar2 = vec![0.0_f64; m];
            let mut ax2 = vec![0.0_f64; m * m];
            let mut e_ap2 = 0.0;
            let mut e_at2 = 0.0;
            let mut e_ar2 = 0.0;
            let mut e_ax2 = vec![0.0_f64; m];

            solve_burgers_cgs_with_thermal(
                2 * m + 1,
                m,
                a,
                &charge,
                &nd,
                rad_accel,
                rad,
                &kdiff,
                zdiff,
                zdiff1,
                zdiff2,
                &mut ap2,
                &mut at2,
                &mut ar2,
                &mut ax2,
                &mut e_ap2,
                &mut e_at2,
                &mut e_ar2,
                &mut e_ax2,
            )
            .map_err(|err| {
                StarError::ConvergenceFailure(format!(
                    "solve_burgers_cgs_with_thermal failed in eta blend at zone {k}: {err}"
                ))
            })?;

            for i in 0..m {
                ap[i] = alfa * ap1[i] + beta * ap2[i];
                at[i] = alfa * at1[i] + beta * at2[i];
                ar[i] = alfa * ar1[i] + beta * ar2[i];
                for j in 0..m {
                    ax[i * m + j] =
                        alfa * ax1[i * m + j] + beta * ax2[i * m + j];
                }
                e_ax[i] = alfa * e_ax1[i] + beta * e_ax2[i];
            }
            *e_ap = alfa * e_ap1 + beta * e_ap2;
            *e_at = alfa * e_at1 + beta * e_at2;
            *e_ar = alfa * e_ar1 + beta * e_ar2;
        }

        *g_ap = 0.0;
        *g_at = 0.0;
        *g_ar = 0.0;
        g_ax[..m].fill(0.0);
        Ok(())
    } else {
        do1_solve_thoul_hu(
            2 * m + 2,
            m,
            a,
            &charge,
            x,
            c,
            rad_accel,
            rad,
            kappa_st,
            zdiff,
            zdiff1,
            zdiff2,
            ap,
            at,
            ar,
            ax,
            e_ap,
            e_at,
            e_ar,
            e_ax,
            g_ap,
            g_at,
            g_ar,
            g_ax,
        )
        .map_err(|err| {
            StarError::ConvergenceFailure(format!(
                "do1_solve_thoul_hu failed at zone {k}: {err}"
            ))
        })
    }
}

/// Converts Burgers gradient coefficients into advective velocities and the
/// concentration-diffusion matrix `sigma_lnC` at face `k`, enforcing baryon
/// conservation and velocity caps (`diffusion_support.f90:621-747`).
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn get1_diffusion_velocities(
    k: usize,
    nc: usize,
    m: usize,
    ap: &[f64],
    at: &[f64],
    ar: &[f64],
    ax: &[f64],
    rho: f64,
    t: f64,
    dln_p_dr: f64,
    dln_t_dr: f64,
    grav: f64,
    dlnne_dr: f64,
    x_face: &[f64],
    vlimit_dm_full_on: f64,
    vlimit_dm_full_off: f64,
    vlimit: f64,
    xm_face: f64,
    r_mid: &[f64],
    dt: f64,
    limit_coeff: f64,
    v_advection_max: f64,
    diffusion_factor: &[f64],
    use_cgs_solver: bool,
    vgt: &mut [f64],
    vln_p: &mut [f64],
    vln_t: &mut [f64],
    vrad: &mut [f64],
    sigma_lnc: &mut [f64],
) {
    const RHO_UNIT: f64 = 1.0e2;
    const T_UNIT: f64 = 1.0e7;

    if limit_coeff <= 0.0 {
        vgt[..nc].fill(0.0);
        sigma_lnc[..(nc * nc)].fill(0.0);
        return;
    }

    let dr = r_mid[k - 1] - r_mid[k];
    let vcross = dr / dt;
    let (vmax_limit, alfa) = if xm_face >= vlimit_dm_full_off || vlimit <= 0.0 {
        (1.0e99, 0.0)
    } else if xm_face <= vlimit_dm_full_on {
        (vcross * vlimit, 1.0)
    } else {
        let a_lin = (xm_face - vlimit_dm_full_off)
            / (vlimit_dm_full_on - vlimit_dm_full_off);
        (vcross * vlimit / a_lin, a_lin)
    };

    if use_cgs_solver {
        for i in 0..nc {
            for j in 0..nc {
                sigma_lnc[j * nc + i] = -limit_coeff
                    * diffusion_factor[i]
                    * BOLTZM
                    * t
                    * ax[j * m + i];
            }
            vln_p[i] = ap[i] * AMU * grav * diffusion_factor[i] * limit_coeff;
            vln_t[i] = at[i]
                * BOLTZM
                * t
                * dln_t_dr
                * diffusion_factor[i]
                * limit_coeff;
            vrad[i] = ar[i] * diffusion_factor[i] * limit_coeff;
            vgt[i] = vln_p[i] + vln_t[i] + vrad[i];
        }
        for i in 0..nc {
            let mut sum_sig = 0.0;
            for j in 0..nc {
                sum_sig += sigma_lnc[i * nc + j];
            }
            vgt[i] -= dlnne_dr * sum_sig;
            if x_face[i] < 1.0e-15 {
                vgt[i] = 0.0;
            }
        }
    } else {
        let tau0 = 6.0e13 * SECYER;
        let t2pt5 = flexmesa_math::pow(t / T_UNIT, 2.5);
        let coef =
            limit_coeff * RSUN * t2pt5 / (rho / RHO_UNIT) * (RSUN / tau0);
        let coef_vrad =
            (limit_coeff / t) * RSUN * RSUN * t2pt5 / (rho / RHO_UNIT) / tau0;

        for i in 0..nc {
            for j in 0..nc {
                sigma_lnc[j * nc + i] =
                    -diffusion_factor[i] * coef * ax[j * m + i];
            }
            vln_p[i] = ap[i] * dln_p_dr * diffusion_factor[i] * coef;
            vln_t[i] = at[i] * dln_t_dr * diffusion_factor[i] * coef;
            vrad[i] = ar[i] * diffusion_factor[i] * coef_vrad;
            vgt[i] = vln_p[i] + vln_t[i] + vrad[i];
            if x_face[i] < 1.0e-15 {
                vgt[i] = 0.0;
            }
        }
    }

    // Baryon conservation adjustment on the most abundant class (`maxloc` returns
    // first maximum).
    let mut im = 0_usize;
    let mut x_max = x_face[0];
    for (i, &xv) in x_face.iter().enumerate().take(nc).skip(1) {
        if xv > x_max {
            x_max = xv;
            im = i;
        }
    }
    let mut xv_dot = 0.0;
    for i in 0..nc {
        xv_dot += x_face[i] * vgt[i];
    }
    let dv_im = -xv_dot / x_face[im];
    vgt[im] += dv_im;

    let mut vmax = 0.0_f64;
    for &v in &vgt[..nc] {
        let av = v.abs();
        if av > vmax {
            vmax = av;
        }
    }
    if vmax > v_advection_max {
        let frac = v_advection_max / vmax;
        for i in 0..nc {
            vgt[i] *= frac;
            for j in 0..nc {
                sigma_lnc[j * nc + i] *= frac;
            }
        }
        vmax = v_advection_max;
    }

    if alfa > 0.0 && vmax > vmax_limit {
        let frac = vmax_limit / vmax;
        for i in 0..nc {
            vgt[i] *= frac;
            for j in 0..nc {
                sigma_lnc[j * nc + i] *= frac;
            }
        }
    }
}

/// Assembles finite-volume advection `GT_face`, self-diffusion `D_self_face`,
/// and concentration-diffusion matrix `SIG_face` at one face
/// (`diffusion_support.f90:750-779`).
#[allow(clippy::too_many_arguments)]
pub fn get1_flow_coeffs(
    nc: usize,
    v_advection_face: &[f64],
    sig_factor: f64,
    gt_factor: f64,
    sigma_lnc_face: &[f64],
    four_pi_r2_rho_face: f64,
    dm_bar: f64,
    c_div_x_face: &[f64],
    gt_face: &mut [f64],
    d_self_face: &mut [f64],
    sig_face: &mut [f64],
) {
    let c = sig_factor * four_pi_r2_rho_face * four_pi_r2_rho_face / dm_bar;
    for i in 0..nc {
        gt_face[i] = gt_factor * four_pi_r2_rho_face * v_advection_face[i];
        d_self_face[i] = sigma_lnc_face[i * nc + i];
        for j in 0..nc {
            sig_face[i * nc + j] =
                c * sigma_lnc_face[i * nc + j] / c_div_x_face[j];
        }
    }
}

/// Solves the `(2m + 2) x (2m + 2)` Thoul et al. (1994) / Hu et al. (2011)
/// Burgers linear system for `(ap, at, ar, ax)` and electric/gravitational
/// field coefficients (`diffusion_support.f90:819-1052`).
///
/// # Errors
///
/// Returns [`StarError::ConvergenceFailure`] if LU factorization or solve fails.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn do1_solve_thoul_hu(
    n: usize,
    m: usize,
    a: &[f64],
    z: &[f64],
    _x: &[f64],
    c: &[f64],
    rad_accel: &[f64],
    rad: bool,
    kappa_st: &[f64],
    zdiff: &[f64],
    zdiff1: &[f64],
    zdiff2: &[f64],
    ap: &mut [f64],
    at: &mut [f64],
    ar: &mut [f64],
    ax: &mut [f64],
    e_ap: &mut f64,
    e_at: &mut f64,
    e_ar: &mut f64,
    e_ax: &mut [f64],
    g_ap: &mut f64,
    g_at: &mut f64,
    g_ar: &mut f64,
    g_ax: &mut [f64],
) -> Result<(), StarError> {
    let ko = 2.0_f64;
    let mut cc = 0.0;
    let mut ac = 0.0;
    for i in 0..m {
        cc += c[i];
        ac += a[i] * c[i];
    }

    let mut xx = vec![0.0_f64; m * m];
    let mut y = vec![0.0_f64; m * m];
    let mut yy = vec![0.0_f64; m * m];
    let mut kmat = vec![0.0_f64; m * m];
    for i in 0..m {
        for j in 0..m {
            let idx = i * m + j;
            xx[idx] = a[j] / (a[i] + a[j]);
            y[idx] = a[i] / (a[i] + a[j]);
            yy[idx] = 3.0 * y[idx] + zdiff1[idx] * xx[idx] * a[j] / a[i];
            kmat[idx] = kappa_st[idx];
        }
    }

    let mut alpha = vec![0.0_f64; n];
    let mut nu = vec![0.0_f64; n];
    let mut beta = vec![0.0_f64; n];
    let mut gamma = vec![0.0_f64; n * n];
    let mut delta = vec![0.0_f64; n * n];

    for i in 0..m {
        alpha[i] = c[i] / cc;
        nu[i] = 0.0;
        if rad {
            beta[i] = -(AMU / BOLTZM) * alpha[i] * a[i] * rad_accel[i];
        } else {
            beta[i] = 0.0;
        }
        for j in 0..m {
            if j != m - 1 {
                let mut gij = -c[j] / cc;
                if j == i {
                    gij += 1.0;
                }
                gamma[i * n + j] = gij * c[i] / cc;
            }
        }
    }

    for i in m..(n - 2) {
        alpha[i] = 0.0;
        nu[i] = 2.5 * c[i - m] / cc;
        beta[i] = 0.0;
    }

    // Momentum rows 0..m
    for i in 0..m {
        for j in 0..m {
            if j == i {
                let mut s = 0.0;
                for l in 0..m {
                    if l != i {
                        s -= kmat[i * m + l];
                    }
                }
                delta[i * n + j] = s;
            } else {
                delta[i * n + j] = kmat[i * m + j];
            }
        }
        for j in m..(n - 2) {
            let jm = j - m;
            if jm == i {
                let mut s = 0.0;
                for l in 0..m {
                    if l != i {
                        let idx = i * m + l;
                        s += zdiff[idx] * xx[idx] * kmat[idx];
                    }
                }
                delta[i * n + j] = s;
            } else {
                let idx = i * m + jm;
                delta[i * n + j] = -zdiff[idx] * y[idx] * kmat[idx];
            }
        }
        delta[i * n + (n - 2)] = c[i] * z[i];
        delta[i * n + (n - 1)] = -c[i] * a[i];
    }

    // Energy rows m..n-2
    for i in m..(n - 2) {
        let im = i - m;
        for j in 0..m {
            if j == im {
                let mut s = 0.0;
                for l in 0..m {
                    if l != im {
                        let idx = im * m + l;
                        s += 2.5 * zdiff[idx] * xx[idx] * kmat[idx];
                    }
                }
                delta[i * n + j] = s;
            } else {
                let idx = im * m + j;
                delta[i * n + j] = -(2.5 * zdiff[idx]) * xx[idx] * kmat[idx];
            }
        }
        for j in m..(n - 2) {
            let jm = j - m;
            if jm == im {
                let mut s = 0.0;
                for l in 0..m {
                    if l != im {
                        let idx = im * m + l;
                        s -= y[idx]
                            * kmat[idx]
                            * (0.8 * zdiff2[idx] * xx[idx] + yy[idx]);
                    }
                }
                let diag_idx = im * m + im;
                s -= 0.4 * zdiff2[diag_idx] * kmat[diag_idx];
                delta[i * n + j] = s;
            } else {
                let idx = im * m + jm;
                delta[i * n + j] = kmat[idx]
                    * xx[idx]
                    * y[idx]
                    * (3.0 + zdiff1[idx] - 0.8 * zdiff2[idx]);
            }
        }
    }

    // Current-neutrality and zero-fluid-velocity rows n-2, n-1
    for j in 0..m {
        delta[(n - 2) * n + j] = c[j] * z[j];
        delta[(n - 1) * n + j] = c[j] * a[j];
    }

    let mut indx = vec![0_usize; n];
    dgetrf(n, &mut delta, &mut indx).map_err(|err| {
        StarError::ConvergenceFailure(format!("dgetrf failed: {err}"))
    })?;
    dgetrs(n, &delta, &indx, &mut alpha).map_err(|err| {
        StarError::ConvergenceFailure(format!("dgetrs alpha failed: {err}"))
    })?;
    dgetrs(n, &delta, &indx, &mut nu).map_err(|err| {
        StarError::ConvergenceFailure(format!("dgetrs nu failed: {err}"))
    })?;
    if rad {
        dgetrs(n, &delta, &indx, &mut beta).map_err(|err| {
            StarError::ConvergenceFailure(format!("dgetrs beta failed: {err}"))
        })?;
    }

    let mut ga = vec![0.0_f64; n];
    for j in 0..n {
        for i in 0..n {
            ga[i] = gamma[i * n + j];
        }
        dgetrs(n, &delta, &indx, &mut ga).map_err(|err| {
            StarError::ConvergenceFailure(format!("dgetrs gamma failed: {err}"))
        })?;
        for i in 0..n {
            gamma[i * n + j] = ga[i];
        }
    }

    let f = ko * ac * cc;
    for j in 0..m {
        ap[j] = alpha[j] * f;
        at[j] = nu[j] * f;
        ar[j] = beta[j] * f;
        for i in 0..m {
            ax[i * m + j] = gamma[i * n + j] * f;
        }
    }
    *e_ap = alpha[n - 2] * f;
    *g_ap = alpha[n - 1] * f;
    *e_at = nu[n - 2] * f;
    *g_at = nu[n - 1] * f;
    *e_ar = beta[n - 2] * f;
    *g_ar = beta[n - 1] * f;
    for i in 0..m {
        e_ax[i] = gamma[(n - 2) * n + i] * f;
        g_ax[i] = gamma[(n - 1) * n + i] * f;
    }
    Ok(())
}

/// Solves the `(m + 1) x (m + 1)` CGS Burgers system without thermal diffusion
/// (`diffusion_support.f90:1055-1211`).
///
/// # Errors
///
/// Returns [`StarError::ConvergenceFailure`] if LU factorization or solve fails.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn solve_burgers_cgs_no_thermal(
    n: usize,
    m: usize,
    a: &[f64],
    z: &[f64],
    nd: &[f64],
    rad_accel: &[f64],
    rad: bool,
    kdiff: &[f64],
    ap: &mut [f64],
    at: &mut [f64],
    ar: &mut [f64],
    ax: &mut [f64],
    e_ap: &mut f64,
    e_at: &mut f64,
    e_ar: &mut f64,
    e_ax: &mut [f64],
) -> Result<(), StarError> {
    let mut delta = vec![0.0_f64; n * n];
    let mut alpha = vec![0.0_f64; n];
    let mut beta = vec![0.0_f64; n];
    let mut nu = vec![0.0_f64; n];
    let mut gamm = vec![0.0_f64; n * n];

    *e_ap = 0.0;
    *e_at = 0.0;
    *e_ar = 0.0;
    e_ax[..m].fill(0.0);

    for i in 0..(m - 1) {
        for j in 0..m {
            if j == i {
                let mut s = 0.0;
                for l in 0..m {
                    if l != j {
                        s -= kdiff[i * m + l];
                    }
                }
                delta[i * n + j] = s;
            } else {
                delta[i * n + j] = kdiff[i * m + j];
            }
        }
        delta[i * n + (n - 1)] = nd[i] * z[i];
    }

    for j in 0..m {
        delta[(m - 1) * n + j] = a[j] * nd[j];
        delta[(n - 1) * n + j] = z[j] * nd[j];
    }

    for i in 0..(m - 1) {
        alpha[i] = nd[i] * a[i];
        nu[i] = nd[i];
        gamm[i * n + i] = nd[i];
        if rad {
            beta[i] = -alpha[i] * AMU * rad_accel[i];
        }
    }

    let mut indx = vec![0_usize; n];
    dgetrf(n, &mut delta, &mut indx).map_err(|err| {
        StarError::ConvergenceFailure(format!("dgetrf failed: {err}"))
    })?;
    dgetrs(n, &delta, &indx, &mut alpha).map_err(|err| {
        StarError::ConvergenceFailure(format!("dgetrs alpha failed: {err}"))
    })?;
    dgetrs(n, &delta, &indx, &mut nu).map_err(|err| {
        StarError::ConvergenceFailure(format!("dgetrs nu failed: {err}"))
    })?;
    if rad {
        dgetrs(n, &delta, &indx, &mut beta).map_err(|err| {
            StarError::ConvergenceFailure(format!("dgetrs beta failed: {err}"))
        })?;
    } else {
        beta.fill(0.0);
    }

    let mut ga = vec![0.0_f64; n];
    for j in 0..n {
        for i in 0..n {
            ga[i] = gamm[i * n + j];
        }
        dgetrs(n, &delta, &indx, &mut ga).map_err(|err| {
            StarError::ConvergenceFailure(format!("dgetrs gamma failed: {err}"))
        })?;
        for i in 0..n {
            gamm[i * n + j] = ga[i];
        }
    }

    for j in 0..m {
        ap[j] = alpha[j];
        at[j] = nu[j];
        ar[j] = beta[j];
        for i in 0..m {
            ax[i * m + j] = gamm[i * n + j];
        }
    }
    *e_ap = alpha[n - 1];
    *e_at = nu[n - 1];
    *e_ar = beta[n - 1];
    for i in 0..m {
        e_ax[i] = gamm[(n - 1) * n + i];
    }
    Ok(())
}

/// Solves the `(2m + 1) x (2m + 1)` CGS Burgers system including thermal
/// diffusion heat-flow equations (`diffusion_support.f90:1214-1453`).
///
/// # Errors
///
/// Returns [`StarError::ConvergenceFailure`] if LU factorization or solve fails.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn solve_burgers_cgs_with_thermal(
    n: usize,
    m: usize,
    a: &[f64],
    z: &[f64],
    nd: &[f64],
    rad_accel: &[f64],
    rad: bool,
    kdiff: &[f64],
    zdiff: &[f64],
    zdiff1: &[f64],
    zdiff2: &[f64],
    ap: &mut [f64],
    at: &mut [f64],
    ar: &mut [f64],
    ax: &mut [f64],
    e_ap: &mut f64,
    e_at: &mut f64,
    e_ar: &mut f64,
    e_ax: &mut [f64],
) -> Result<(), StarError> {
    let mut delta = vec![0.0_f64; n * n];
    let mut alpha = vec![0.0_f64; n];
    let mut beta = vec![0.0_f64; n];
    let mut nu = vec![0.0_f64; n];
    let mut gamm = vec![0.0_f64; n * n];

    *e_ap = 0.0;
    *e_at = 0.0;
    *e_ar = 0.0;
    e_ax[..m].fill(0.0);

    let rightshift = m;
    let downshift = m - 1;

    // Momentum equations for ions 0..m-1
    for i in 0..(m - 1) {
        for j in 0..m {
            if j == i {
                let mut s = 0.0;
                for l in 0..m {
                    if l != j {
                        s -= kdiff[i * m + l];
                    }
                }
                delta[i * n + j] = s;
            } else {
                delta[i * n + j] = kdiff[i * m + j];
            }
        }

        for j in 0..m {
            if j == i {
                let mut s = 0.0;
                for l in 0..m {
                    if l != j {
                        let idx = i * m + l;
                        s += kdiff[idx] * zdiff[idx] * a[l] / (a[i] + a[l]);
                    }
                }
                delta[i * n + (j + rightshift)] = s;
            } else {
                let idx = i * m + j;
                delta[i * n + (j + rightshift)] =
                    -kdiff[idx] * zdiff[idx] * a[i] / (a[i] + a[j]);
            }
        }

        delta[i * n + (n - 1)] = nd[i] * z[i];
    }

    // Energy equations for all m species (ions + electrons)
    for i in 0..m {
        let row = i + downshift;
        for j in 0..m {
            if j == i {
                let mut s = 0.0;
                for l in 0..m {
                    if l != j {
                        let idx = i * m + l;
                        s += 2.5 * kdiff[idx] * zdiff[idx] * a[l]
                            / (a[i] + a[l]);
                    }
                }
                delta[row * n + j] = s;
            } else {
                let idx = i * m + j;
                delta[row * n + j] =
                    -2.5 * kdiff[idx] * zdiff[idx] * a[j] / (a[i] + a[j]);
            }
        }

        for j in 0..m {
            if j == i {
                let diag_idx = i * m + j;
                let mut s = -0.4 * kdiff[diag_idx] * zdiff2[diag_idx];
                for l in 0..m {
                    if l != j {
                        let idx = i * m + l;
                        let denom = flexmesa_math::pow2(a[i] + a[l]);
                        s -= kdiff[idx]
                            * ((3.0 * flexmesa_math::pow2(a[i])
                                + flexmesa_math::pow2(a[l]) * zdiff1[idx])
                                / denom
                                + 0.8 * zdiff2[idx] * a[i] * a[l] / denom);
                    }
                }
                delta[row * n + (j + rightshift)] = s;
            } else {
                let idx = i * m + j;
                delta[row * n + (j + rightshift)] = kdiff[idx]
                    * (3.0 + zdiff1[idx] - 0.8 * zdiff2[idx])
                    * a[i]
                    * a[j]
                    / flexmesa_math::pow2(a[i] + a[j]);
            }
        }

        delta[row * n + (n - 1)] = 0.0;
    }

    // Mass and charge conservation rows n-2 and n-1
    for j in 0..m {
        delta[(n - 2) * n + j] = a[j] * nd[j];
        delta[(n - 1) * n + j] = z[j] * nd[j];
    }

    for i in 0..(m - 1) {
        alpha[i] = nd[i] * a[i];
        nu[i] = nd[i];
        gamm[i * n + i] = nd[i];
        if rad {
            beta[i] = -alpha[i] * AMU * rad_accel[i];
        }
    }
    for i in 0..m {
        nu[i + downshift] = 2.5 * nd[i];
    }

    let mut indx = vec![0_usize; n];
    dgetrf(n, &mut delta, &mut indx).map_err(|err| {
        StarError::ConvergenceFailure(format!("dgetrf failed: {err}"))
    })?;
    dgetrs(n, &delta, &indx, &mut alpha).map_err(|err| {
        StarError::ConvergenceFailure(format!("dgetrs alpha failed: {err}"))
    })?;
    dgetrs(n, &delta, &indx, &mut nu).map_err(|err| {
        StarError::ConvergenceFailure(format!("dgetrs nu failed: {err}"))
    })?;
    if rad {
        dgetrs(n, &delta, &indx, &mut beta).map_err(|err| {
            StarError::ConvergenceFailure(format!("dgetrs beta failed: {err}"))
        })?;
    } else {
        beta.fill(0.0);
    }

    let mut ga = vec![0.0_f64; n];
    for j in 0..n {
        for i in 0..n {
            ga[i] = gamm[i * n + j];
        }
        dgetrs(n, &delta, &indx, &mut ga).map_err(|err| {
            StarError::ConvergenceFailure(format!("dgetrs gamma failed: {err}"))
        })?;
        for i in 0..n {
            gamm[i * n + j] = ga[i];
        }
    }

    for j in 0..m {
        ap[j] = alpha[j];
        at[j] = nu[j];
        ar[j] = beta[j];
        for i in 0..m {
            ax[i * m + j] = gamm[i * n + j];
        }
    }
    *e_ap = alpha[n - 1];
    *e_at = nu[n - 1];
    *e_ar = beta[n - 1];
    for i in 0..m {
        e_ax[i] = gamm[(n - 1) * n + i];
    }
    Ok(())
}

/// Evaluates ion-ion resistance coefficients following Stanton & Murillo (2016,
/// Phys. Rev. E 93, 043203, Appendix C.3), modifying only `(0..nc, 0..nc)`
/// entries of `kdiff`, `zdiff`, `zdiff1`, and `zdiff2`
/// (`diffusion_support.f90:1457-1618`).
///
/// Returns `kappa` (ion separation relative to electron screening length).
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn get_sm_coeffs(
    nc: usize,
    m: usize,
    rho: f64,
    t: f64,
    a: &[f64],
    z: &[f64],
    nd: &[f64],
    kdiff: &mut [f64],
    zdiff: &mut [f64],
    zdiff1: &mut [f64],
    zdiff2: &mut [f64],
) -> f64 {
    // Table IV coefficients indexed by [no - 1][mo - 1]
    const A1: [[f64; 3]; 2] = [[1.4660, 0.52094, 0.30346], [0.0, 0.85401, 0.0]];
    const A2: [[f64; 3]; 2] =
        [[-1.7836, 0.25153, 0.23739], [0.0, -0.22898, 0.0]];
    const A3: [[f64; 3]; 2] =
        [[1.4313, -1.1337, -0.62167], [0.0, -0.60059, 0.0]];
    const A4: [[f64; 3]; 2] =
        [[-0.55833, 1.2155, 0.56110], [0.0, 0.80591, 0.0]];
    const A5: [[f64; 3]; 2] =
        [[0.061162, -0.43784, -0.18046], [0.0, -0.30555, 0.0]];

    const B0: [[f64; 3]; 2] =
        [[0.081033, 0.20572, 0.68375], [0.0, 0.43475, 0.0]];
    const B1: [[f64; 3]; 2] =
        [[-0.091336, -0.16536, -0.38459], [0.0, -0.21147, 0.0]];
    const B2: [[f64; 3]; 2] =
        [[0.051760, 0.061572, 0.10711], [0.0, 0.11116, 0.0]];
    const B3: [[f64; 3]; 2] =
        [[-0.50026, -0.12770, 0.10649], [0.0, 0.19665, 0.0]];
    const B4: [[f64; 3]; 2] =
        [[0.17044, 0.066993, 0.028760], [0.0, 0.15195, 0.0]];

    let (lambda, kappa) = lam_sm(nc, m, rho, t, z, nd);

    let kbt32 = flexmesa_math::pow(BOLTZM * t, 1.5);
    let mut keff = [[0.0_f64; 3]; 2];
    let mut omega = [[0.0_f64; 3]; 2];

    for i in 0..nc {
        for j in 0..nc {
            let gp = z[i] * z[j] * QE * QE / (lambda * BOLTZM * t);
            let lgp = flexmesa_math::safe_log(gp);
            let gp1 = gp;
            let gp2 = flexmesa_math::pow2(gp1);
            let gp3 = flexmesa_math::pow3(gp1);
            let gp4 = flexmesa_math::pow4(gp1);
            let gp5 = flexmesa_math::pow5(gp1);

            if gp < 0.0 {
                keff = [[0.0; 3]; 2];
            } else if gp < 1.0 {
                for no_idx in 0..2 {
                    let no_f = (no_idx + 1) as f64;
                    for mo_idx in 0..3 {
                        let facmo = if mo_idx == 2 { 2.0 } else { 1.0 };
                        keff[no_idx][mo_idx] = (-no_f / 4.0)
                            * facmo
                            * flexmesa_math::safe_log(
                                A1[no_idx][mo_idx] * gp1
                                    + A2[no_idx][mo_idx] * gp2
                                    + A3[no_idx][mo_idx] * gp3
                                    + A4[no_idx][mo_idx] * gp4
                                    + A5[no_idx][mo_idx] * gp5,
                            );
                    }
                }
            } else {
                let lgp2 = flexmesa_math::pow2(lgp);
                for no_idx in 0..2 {
                    for mo_idx in 0..3 {
                        keff[no_idx][mo_idx] = (B0[no_idx][mo_idx]
                            + B1[no_idx][mo_idx] * lgp
                            + B2[no_idx][mo_idx] * lgp2)
                            / (1.0
                                + B3[no_idx][mo_idx] * gp1
                                + B4[no_idx][mo_idx] * gp2);
                    }
                }
            }

            let mu_ij = AMU * a[i] * a[j] / (a[i] + a[j]);
            let tmp = (2.0 * PI / mu_ij).sqrt()
                * (flexmesa_math::pow2(z[i] * z[j] * QE * QE) / kbt32);
            for no_idx in 0..2 {
                for mo_idx in 0..3 {
                    omega[no_idx][mo_idx] = tmp * keff[no_idx][mo_idx];
                }
            }

            let idx = i * m + j;
            kdiff[idx] = (16.0 / 3.0) * nd[i] * nd[j] * mu_ij * omega[0][0];
            zdiff[idx] = 1.0 - 2.0 * omega[0][1] / (5.0 * omega[0][0]);
            zdiff1[idx] = 2.5
                + (2.0 * omega[0][2] - 10.0 * omega[0][1])
                    / (5.0 * omega[0][0]);
            zdiff2[idx] = omega[1][1] / omega[0][0];
        }
    }

    kappa
}

/// Evaluates the effective screening length `lam_eff` (`cm`) and screening
/// parameter `kappa` following Stanton & Murillo (2016)
/// (`diffusion_support.f90:1621-1671`).
#[must_use]
pub fn lam_sm(
    nc: usize,
    m: usize,
    _rho: f64,
    t: f64,
    z: &[f64],
    nd: &[f64],
) -> (f64, f64) {
    let tiny_n = 1.0e-20;
    let ne = nd[m - 1];
    let ef = (HBAR * HBAR * flexmesa_math::pow(3.0 * PI * PI * ne, 2.0 / 3.0))
        / (2.0 * ME);
    let lam_e = flexmesa_math::pow(
        PI4 * QE * QE * ne
            / (flexmesa_math::pow2(BOLTZM * t)
                + flexmesa_math::pow2((2.0 / 3.0) * ef))
            .sqrt(),
        -0.5,
    );

    let mut ni_tot = 0.0;
    for &n_i in &nd[..nc] {
        ni_tot += n_i;
    }
    let kappa = flexmesa_math::pow(3.0 / (PI4 * ni_tot), ONE_THIRD) / lam_e;

    let mut rhotot = 0.0;
    for i in 0..nc {
        rhotot += z[i] * QE * nd[i];
    }

    let mut lam_sum = 1.0 / flexmesa_math::pow2(lam_e);
    for i in 0..nc {
        let ai = flexmesa_math::pow(
            z[i] * QE / (FOUR_THIRDS_PI * rhotot),
            ONE_THIRD,
        );
        let gam_is = z[i] * z[i] * QE * QE / (ai * BOLTZM * t);
        let lam_i = if nd[i] < tiny_n {
            1.0e99
        } else {
            (BOLTZM * t / (PI4 * flexmesa_math::pow2(z[i] * QE) * nd[i])).sqrt()
        };
        lam_sum += flexmesa_math::pow(
            flexmesa_math::pow2(lam_i) * (1.0 + 3.0 * gam_is),
            -1.0,
        );
    }
    let lam_eff = flexmesa_math::pow(lam_sum, -0.5);
    (lam_eff, kappa)
}

/// Evaluates ion-ion diffusion and Burgers resistance coefficients following
/// Caplan, Bauer, & Freeman (2022, Section 4), modifying only `(0..nc, 0..nc)`
/// entries of `kdiff` (`diffusion_support.f90:1674-1714`).
#[allow(clippy::too_many_arguments)]
pub fn get_cbf_coeffs(
    nc: usize,
    m: usize,
    rho: f64,
    t: f64,
    a: &[f64],
    z: &[f64],
    nd: &[f64],
    gamma: f64,
    ddiff: &mut [f64],
    kdiff: &mut [f64],
) {
    let mut ni_sum = 0.0;
    let mut a_num = 0.0;
    let mut z_num = 0.0;
    for i in 0..nc {
        ni_sum += nd[i];
        a_num += a[i] * nd[i];
        z_num += z[i] * nd[i];
    }
    let abar = a_num / ni_sum;
    let zbar = z_num / ni_sum;
    let mut zp6bar = 0.0;
    for j in 0..nc {
        zp6bar += flexmesa_math::pow(z[j], 0.6) * nd[j] / ni_sum;
    }

    let (omegap, ai, _lam_e, kappa) = kappa_cbf(nc, m, rho, t, abar, zbar, nd);
    let dstar_ocp = get_dstar_ocp(gamma, kappa);
    let docp = dstar_ocp * ai * ai * omegap;

    for j in 0..nc {
        ddiff[j] = docp * flexmesa_math::pow(z[j] / zbar, -0.6);
    }

    let kfac =
        BOLTZM * t / (ni_sum * zp6bar * flexmesa_math::pow(zbar, 0.6) * docp);
    for i in 0..nc {
        for j in 0..nc {
            kdiff[i * m + j] =
                nd[i] * nd[j] * flexmesa_math::pow(z[i] * z[j], 0.6) * kfac;
        }
    }
}

/// Evaluates the dimensionless one-component-plasma diffusion fit `D*` from
/// Caplan, Bauer, & Freeman (2022, Eq. 5) (`diffusion_support.f90:1716-1743`).
#[must_use]
pub fn get_dstar_ocp(gamma: f64, kappa: f64) -> f64 {
    let a0 = 1.55973;
    let a1 = 1.10941;
    let a2 = 1.36909;

    let b0 = 0.0070782;
    let b1 = 0.80499;
    let b2 = 4.53523;

    let c0 = 2.20689;
    let c1 = 1.351594;
    let c2 = 1.57138;
    let c3 = 3.34187;

    let ak = (PI / 3.0).sqrt() * (a0 + a1 * flexmesa_math::pow(kappa, a2));
    let bk = b0 * flexmesa_math::exp(-b1 * flexmesa_math::pow(kappa, b2));
    let ck = c0 + c1 * bitsafe_erf_fit(c2 * flexmesa_math::pow(kappa, c3));

    (PI / 3.0).sqrt()
        * ak
        * flexmesa_math::pow(gamma, -2.5)
        * flexmesa_math::exp(-bk * gamma)
        / flexmesa_math::log(
            1.0 + ck * flexmesa_math::pow(gamma, -1.5) / 3.0_f64.sqrt(),
        )
}

/// Evaluates plasma frequency `omegap` (`s^-1`), ion-sphere radius `ai` (`cm`),
/// electron screening length `lam_e` (`cm`), and screening parameter `kappa`
/// for Caplan, Bauer, & Freeman (2022) (`diffusion_support.f90:1747-1781`).
#[must_use]
pub fn kappa_cbf(
    nc: usize,
    m: usize,
    rho: f64,
    t: f64,
    abar: f64,
    zbar: f64,
    nd: &[f64],
) -> (f64, f64, f64, f64) {
    let ne = nd[m - 1];
    let ef = (HBAR * HBAR * flexmesa_math::pow(3.0 * PI * PI * ne, 2.0 / 3.0))
        / (2.0 * ME);
    let lam_e_sm = flexmesa_math::pow(
        PI4 * QE * QE * ne
            / (flexmesa_math::pow2(BOLTZM * t)
                + flexmesa_math::pow2((2.0 / 3.0) * ef))
            .sqrt(),
        -0.5,
    );

    let lam_e = if rho < 1.0e6 {
        lam_e_sm
    } else {
        let kf = flexmesa_math::pow(3.0 * PI * PI * ne, ONE_THIRD);
        let lam_e_rel = 1.0 / (2.0 * kf * (FINE / PI).sqrt());
        lam_e_sm.min(lam_e_rel)
    };

    let mut ni_tot = 0.0;
    for &n_i in &nd[..nc] {
        ni_tot += n_i;
    }
    let omegap = (PI4 * ni_tot * zbar * zbar * QE * QE / (abar * AMU)).sqrt();
    let ai = flexmesa_math::pow(3.0 / (PI4 * ni_tot), ONE_THIRD);
    let kappa = ai / lam_e;
    (omegap, ai, lam_e, kappa)
}

/// Abramowitz & Stegun (1964) error-function rational fit using bit-safe
/// `exp` (`diffusion_support.f90:1786-1798`).
#[must_use]
pub fn bitsafe_erf_fit(x: f64) -> f64 {
    let p = 0.47047;
    let a1 = 0.3480242;
    let a2 = -0.0958798;
    let a3 = 0.7478556;

    let t = 1.0 / (1.0 + p * x);
    1.0 - (a1 * t + a2 * t * t + a3 * t * t * t) * flexmesa_math::exp(-x * x)
}
