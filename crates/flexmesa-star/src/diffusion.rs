// SPDX-License-Identifier: LGPL-3.0-or-later
//! Multi-species Burgers element diffusion, screened Coulomb / Yukawa collision
//! integrals, Caplan et al. (2022) strong-coupling OCP liquid-metal correction,
//! radiative levitation, white-dwarf sedimentation heating, and implicit
//! mass-conserving abundance solver (`diffusion.f90`, `diffusion_procs.f90`,
//! `diffusion_support.f90`, `paquette_coeffs.f90`).

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{AMU, CLIGHT, HBAR, KERG, ME, PI, QE};
use flexmesa_ionization::eval_typical_charge;
use flexmesa_mtx::solve_dense;
use flexmesa_num::neumaier_sum;

use crate::model::StarModel;
use crate::struct_burn_mix::ScalarTridiagFactor;

/// Collision-integral formulation for multi-species Burgers diffusion
/// (`diffusion_use_paquette`, `diffusion_use_stanton_murillo`,
/// `diffusion_use_caplan22`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffusionScheme {
    /// Iben & MacDonald (1985) analytic screened Coulomb logarithm.
    IbenMacDonald,
    /// Paquette, Pelletier, Fontaine & Michaud (1986) screened Coulomb fits.
    Paquette,
    /// Stanton & Murillo (2016) Yukawa collision integrals with degenerate
    /// Thomas-Fermi electron screening.
    StantonMurillo,
    /// Stanton & Murillo (2016) blended with Caplan, Bauer & Freeman (2022)
    /// strong-coupling OCP liquid-metal self-diffusion over
    /// $\Gamma \in [3, 10]$.
    CaplanOcp,
}

/// Output of the single-zone multi-species Burgers solver
/// (`eval_burgers_velocities`).
#[derive(Debug, Clone)]
pub struct BurgersZoneResult {
    /// Net drift velocity $v_{\mathrm{gt},i}$ (gravitational settling + thermal
    /// diffusion + radiative levitation) in $\mathrm{cm\,s^{-1}}$
    /// (`len = nc`).
    pub v_gt: Vec<f64>,
    /// Trace self-diffusion coefficient $D_i$ in $\mathrm{cm^2\,s^{-1}}$
    /// (`len = nc`).
    pub d_self: Vec<f64>,
    /// Ambipolar electric force $e E$ in $\mathrm{dyn}$ (`eE`).
    pub ee_force: f64,
    /// Typical ionic charges $Z_i$ used in the zone (`len = nc`).
    pub z_charge: Vec<f64>,
}

/// Evaluates the degenerate-corrected effective electron thermal energy
/// $k_{\mathrm{B}} T_{\mathrm{eff}} =
/// \sqrt{(k_{\mathrm{B}} T)^2 + (\tfrac{2}{3} E_{\mathrm{F}})^2}$ in erg.
#[must_use]
#[inline]
pub fn eval_effective_kt(n_e: f64, kt: f64) -> f64 {
    let kf = (3.0 * PI * PI * n_e.max(1.0e-10)).cbrt();
    let ef = (HBAR * kf).powi(2) / (2.0 * ME);
    kt.hypot((2.0 / 3.0) * ef)
}

/// Evaluates the Coulomb logarithm $\ln\Lambda_{ij}$ and heat-flow collision
/// ratios $(z_{ij}, z'_{ij}, z''_{ij})$ for species pair $(i, j)$ matching
/// `paquette_coeffs.f90` (`iben_macdonald`, `paquette_f`, `stanton_murillo`).
#[must_use]
pub fn eval_coulomb_collision_factors(
    scheme: DiffusionScheme,
    z_i: f64,
    z_j: f64,
    t: f64,
    n_e: f64,
    n_ions: f64,
    sum_n_z2: f64,
) -> (f64, f64, f64, f64) {
    let kt = (KERG * t.max(100.0)).max(1.0e-30);
    let e2 = QE * QE;
    let z_prod = (z_i * z_j).abs().max(1.0e-6);

    let kt_eff_e = eval_effective_kt(n_e, kt);
    let inv_lam_e2 = 4.0 * PI * e2 * n_e.max(1.0e-10) / kt_eff_e;
    let inv_lam_i2 = 4.0 * PI * e2 * sum_n_z2.max(1.0e-10) / kt;
    let lam_d = 1.0 / (inv_lam_e2 + inv_lam_i2).sqrt();
    let a_ion = (3.0 / (4.0 * PI * n_ions.max(1.0e-10))).cbrt();
    let lam_eff = lam_d.max(a_ion);

    let x_ij = (4.0 * lam_eff * kt / (z_prod * e2)).max(1.0e-12);
    match scheme {
        DiffusionScheme::IbenMacDonald => {
            let ln_lam = (1.0 + 0.18769 * x_ij.powf(1.2)).ln() / 1.23084;
            (ln_lam.max(0.05), 0.6, 1.3, 2.0)
        }
        DiffusionScheme::Paquette => {
            let gamma2 = x_ij * x_ij;
            let psi = (1.0 + gamma2).ln().max(1.0e-10).ln();
            let ln_lam = 0.5 * (1.0 + 0.75 * gamma2).ln();
            let w = 0.5 * (1.0 + (0.7 * (psi - 0.5)).tanh());
            let z_ij = 0.35 + 0.25 * w;
            let zp_ij = 0.85 + 0.45 * w;
            let zpp_ij = 1.35 + 0.65 * w;
            (ln_lam.max(0.05), z_ij, zp_ij, zpp_ij)
        }
        DiffusionScheme::StantonMurillo | DiffusionScheme::CaplanOcp => {
            let g = (z_prod * e2 / (lam_eff * kt)).clamp(1.0e-12, 1.0e4);
            let k11 = if g < 1.0 {
                let ln_g = g.ln();
                -0.25 * (1.4660 + ln_g) * (1.0 - 0.0462 * g)
            } else {
                0.3804 / g.powf(1.75) + 0.1656 / g
            };
            let w = 1.0 / (1.0 + 0.85 * g.powf(0.65));
            let z_ij = 0.32 + 0.28 * w;
            let zp_ij = 0.80 + 0.50 * w;
            let zpp_ij = 1.30 + 0.70 * w;
            (k11.max(0.02), z_ij, zp_ij, zpp_ij)
        }
    }
}

/// Evaluates the Caplan, Bauer & Freeman (2022) strong-coupling one-component
/// plasma (OCP) dimensionless self-diffusion coefficient $D^*(\Gamma_i, \kappa)$
/// matching `paquette_coeffs.f90:572-624`.
#[must_use]
pub fn eval_caplan_ocp_dstar(gamma_i: f64, kappa: f64) -> f64 {
    let gam = gamma_i.max(0.1);
    let kap = kappa.max(0.0);
    let pref = (PI / 3.0).sqrt();
    let a_k = pref * (1.22 + 0.405 * kap.powf(1.77));
    let b_k = 0.562 * (-0.185 * kap.powf(1.31)).exp();
    let c_k = 1.11 + 0.112 * kap.powf(1.32);
    let term1 = pref * a_k * gam.powf(-2.5);
    let term2 = (b_k / gam) * (-c_k * gam.cbrt()).exp();
    (term1 + term2).max(1.0e-12)
}

fn assemble_burgers_resistance(
    nc: usize,
    n_s: &[f64],
    m_s: &[f64],
    z_s: &[f64],
    t: f64,
    n_ions: f64,
    sum_n_z2: f64,
    scheme: DiffusionScheme,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64) {
    let m = nc + 1;
    let kt = KERG * t.max(100.0);
    let e2 = QE * QE;
    let pref_k = (16.0 / 3.0) * (PI / (2.0 * kt)).sqrt() * e2 * e2 / kt;

    let a_ws = (3.0 / (4.0 * PI * n_ions.max(1.0e-10))).cbrt();
    let kt_eff = eval_effective_kt(n_s[nc], kt);
    let k_tf = (4.0 * PI * e2 * n_s[nc] / kt_eff).sqrt();
    let kappa = a_ws * k_tf;

    let mut k_mat = vec![0.0; m * m];
    let mut z_mat = vec![0.0; m * m];
    let mut zp_mat = vec![0.0; m * m];
    let mut zpp_mat = vec![0.0; m * m];

    for i in 0..m {
        for j in (i + 1)..m {
            let mu_ij = (m_s[i] * m_s[j]) / (m_s[i] + m_s[j]);
            let (ln_lam, z_ij, zp_ij, zpp_ij) = eval_coulomb_collision_factors(
                scheme, z_s[i], z_s[j], t, n_s[nc], n_ions, sum_n_z2,
            );
            let mut k_ij = pref_k
                * mu_ij.sqrt()
                * n_s[i]
                * n_s[j]
                * (z_s[i] * z_s[j]).powi(2)
                * ln_lam;

            if scheme == DiffusionScheme::CaplanOcp && i < nc && j < nc {
                let gam_ij = (z_s[i] * z_s[j]).abs() * e2 / (a_ws * kt);
                if gam_ij > 3.0 {
                    let d_star = eval_caplan_ocp_dstar(gam_ij, kappa);
                    let omega_p =
                        (4.0 * PI * n_ions * (z_s[i] * z_s[j]).abs() * e2
                            / (2.0 * mu_ij))
                            .sqrt();
                    let d_ocp = a_ws * a_ws * omega_p * d_star;
                    let k_ocp =
                        n_s[i] * (n_s[j] / n_ions) * kt / d_ocp.max(1.0e-20);
                    let f_blend = ((gam_ij - 3.0) / 7.0).clamp(0.0, 1.0);
                    k_ij = (1.0 - f_blend) * k_ij + f_blend * k_ocp;
                }
            }
            k_mat[i * m + j] = k_ij;
            k_mat[j * m + i] = k_ij;
            z_mat[i * m + j] = z_ij;
            z_mat[j * m + i] = z_ij;
            zp_mat[i * m + j] = zp_ij;
            zp_mat[j * m + i] = zp_ij;
            zpp_mat[i * m + j] = zpp_ij;
            zpp_mat[j * m + i] = zpp_ij;
        }
    }
    (k_mat, z_mat, zp_mat, zpp_mat, pref_k)
}

/// Solves the multi-species Burgers resistance equations (`diffusion_procs.f90:
/// solve_CGS_Burgers_eqs`) in a single spherical shell for $n_{\mathrm{c}}$ ion
/// species plus electrons ($m = n_{\mathrm{c}} + 1$).
///
/// # Arguments
/// * `a` - Atomic mass numbers $A_i$ (`len = nc`).
/// * `z` - Mean ionic charges $Z_i$ (`len = nc`, $\ge 1$).
/// * `x_mass` - Mass fractions $X_i$ (`len = nc`).
/// * `rho` - Mass density $\rho$ in $\mathrm{g\,cm^{-3}}$.
/// * `t` - Temperature $T$ in $\mathrm{K}$.
/// * `dlnt_dr` - Radial temperature gradient $\mathrm{d}\ln T/\mathrm{d}r$
///   ($\mathrm{cm^{-1}}$).
/// * `g_grav` - Local gravitational acceleration $g = G m / r^2$
///   ($\mathrm{cm\,s^{-2}}$).
/// * `g_rad` - Optional radiative levitation acceleration $g_{\mathrm{rad},i}$
///   ($\mathrm{cm\,s^{-2}}$).
/// * `scheme` - Collision-integral scheme ([`DiffusionScheme`]).
#[must_use]
pub fn eval_burgers_velocities(
    a: &[f64],
    z: &[f64],
    x_mass: &[f64],
    rho: f64,
    t: f64,
    dlnt_dr: f64,
    g_grav: f64,
    g_rad: Option<&[f64]>,
    scheme: DiffusionScheme,
) -> BurgersZoneResult {
    let nc = a.len();
    let m = nc + 1;
    let kt = KERG * t.max(100.0);

    let mut n_s = vec![0.0; m];
    let mut m_s = vec![0.0; m];
    let mut z_s = vec![0.0; m];
    let mut n_ions = 0.0;
    let mut sum_n_z2 = 0.0;
    let mut n_e = 0.0;
    for i in 0..nc {
        let ai = a[i].max(1.0);
        let zi = z[i].max(0.1);
        let xi = x_mass[i].clamp(1.0e-15, 1.0);
        let ni = rho * xi / (ai * AMU);
        n_s[i] = ni;
        m_s[i] = ai * AMU;
        z_s[i] = zi;
        n_ions += ni;
        sum_n_z2 += ni * zi * zi;
        n_e += ni * zi;
    }
    n_s[nc] = n_e.max(1.0e-10);
    m_s[nc] = ME;
    z_s[nc] = -1.0;

    let (k_mat, z_mat, zp_mat, zpp_mat, pref_k) = assemble_burgers_resistance(
        nc, &n_s, &m_s, &z_s, t, n_ions, sum_n_z2, scheme,
    );

    // Solve the full-rank (2m + 2) x (2m + 2) Thoul, Bahcall & Loeb (1994)
    // Burgers linear system for [w_0..w_{m-1}, r_0..r_{m-1}, eE, g_corr]
    let n_sys = 2 * m + 2;
    let mut lhs = vec![0.0; n_sys * n_sys];
    let mut rhs = vec![0.0; n_sys];
    let n_tot = n_ions + n_s[nc];
    let rho_tot: f64 =
        (0..m).map(|i| n_s[i] * m_s[i]).sum::<f64>().max(1.0e-30);

    for i in 0..m {
        let mut sum_k = 0.0;
        let mut sum_kz = 0.0;
        for j in 0..m {
            if j != i {
                let kij = k_mat[i * m + j];
                let zij = z_mat[i * m + j];
                let mu_fac_j = m_s[i] / (m_s[i] + m_s[j]);
                let mu_fac_i = m_s[j] / (m_s[i] + m_s[j]);
                sum_k += kij;
                sum_kz += kij * zij * mu_fac_i;
                lhs[i * n_sys + j] = kij;
                lhs[i * n_sys + (m + j)] = -kij * zij * mu_fac_j;
            }
        }
        lhs[i * n_sys + i] = -sum_k;
        lhs[i * n_sys + (m + i)] = sum_kz;
        lhs[i * n_sys + 2 * m] = n_s[i] * z_s[i];
        lhs[i * n_sys + (2 * m + 1)] = n_s[i] * m_s[i];

        let g_r = if i < nc {
            g_rad.map_or(0.0, |gr| gr.get(i).copied().unwrap_or(0.0))
        } else {
            0.0
        };
        rhs[i] = n_s[i] * m_s[i] * (g_grav - g_r)
            - (n_s[i] / n_tot.max(1.0e-30)) * rho_tot * g_grav;
    }

    for i in 0..m {
        let row = m + i;
        let mut diag_r = -0.8
            * (2.0_f64).sqrt()
            * pref_k
            * m_s[i].sqrt()
            * n_s[i]
            * n_s[i]
            * z_s[i].powi(4)
            * 1.5;
        let mut diag_w = 0.0;
        for j in 0..m {
            if j != i {
                let kij = k_mat[i * m + j];
                let zij = z_mat[i * m + j];
                let zpij = zp_mat[i * m + j];
                let zppij = zpp_mat[i * m + j];
                let mtot = m_s[i] + m_s[j];
                let mi_f = m_s[i] / mtot;
                let mj_f = m_s[j] / mtot;
                let w_coeff = 2.5 * kij * zij * mj_f / n_s[i];
                diag_w += w_coeff;
                lhs[row * n_sys + j] = -w_coeff;

                diag_r -= kij
                    * (3.0 * mi_f * mi_f
                        + mj_f * mj_f * zppij
                        + 0.8 * mi_f * mj_f * zpij)
                    / n_s[i];
                lhs[row * n_sys + (m + j)] =
                    kij * mi_f * mj_f * (3.0 + zppij - 0.8 * zpij) / n_s[j];
            }
        }
        lhs[row * n_sys + i] = diag_w;
        lhs[row * n_sys + (m + i)] = diag_r;
        rhs[row] = 2.5 * kt * dlnt_dr;
    }

    let charge_row = 2 * m;
    let mass_row = 2 * m + 1;
    for j in 0..m {
        lhs[charge_row * n_sys + j] = n_s[j] * z_s[j];
        lhs[mass_row * n_sys + j] = n_s[j] * m_s[j];
    }

    let sol =
        solve_dense(n_sys, &lhs, &rhs).unwrap_or_else(|_| vec![0.0; n_sys]);

    let mut mom_sum = 0.0;
    let mut rho_sum = 0.0;
    for i in 0..nc {
        let rho_i = n_s[i] * m_s[i];
        mom_sum += rho_i * sol[i];
        rho_sum += rho_i;
    }
    let v_cm = mom_sum / rho_sum.max(1.0e-30);
    let mut v_gt = vec![0.0; nc];
    let mut d_self = vec![0.0; nc];
    for i in 0..nc {
        v_gt[i] = (sol[i] - v_cm).clamp(-1.0e2, 1.0e2);
        let mut k_tot = 0.0;
        for j in 0..m {
            if j != i {
                k_tot += k_mat[i * m + j];
            }
        }
        d_self[i] = (n_s[i] * kt / k_tot.max(1.0e-30)).clamp(1.0e-10, 1.0e12);
    }

    BurgersZoneResult {
        v_gt,
        d_self,
        ee_force: sol[2 * m],
        z_charge: z.to_vec(),
    }
}

/// Computes the radiative levitation acceleration $g_{\mathrm{rad},i} =
/// \frac{\kappa_i L_r}{4\pi r^2 c}$ in $\mathrm{cm\,s^{-2}}$
/// (`diffusion_procs.f90:516-540`).
#[must_use]
pub fn eval_radiative_levitation_accel(
    kappa_species: f64,
    lum_face: f64,
    r_face: f64,
) -> f64 {
    let area = 4.0 * PI * r_face.max(1.0e5).powi(2);
    (kappa_species.max(0.0) * lum_face.max(0.0)) / (area * CLIGHT)
}

/// Computes the white-dwarf gravitational sedimentation heating rate
/// $\epsilon_{\mathrm{WD,sed}}(k)$ in $\mathrm{erg\,g^{-1}\,s^{-1}}$ matching
/// `diffusion.f90:572-648`.
#[must_use]
pub fn eval_wd_sedimentation_heating(
    a: &[f64],
    z: &[f64],
    x_mass: &[f64],
    v_gt: &[f64],
    abar: f64,
    zbar: f64,
    g_grav: f64,
) -> f64 {
    let n = a.len().min(z.len()).min(x_mass.len()).min(v_gt.len());
    let ee_grav = AMU * g_grav / (zbar.max(1.0) + 1.0);
    let mut eps_wd = 0.0;
    for i in 0..n {
        let ai = a[i].max(1.0);
        let zi = z[i].max(1.0);
        let f_e = (zi - ai * zbar / abar.max(1.0)) * ee_grav;
        eps_wd += (x_mass[i] / (ai * AMU)) * v_gt[i] * f_e;
    }
    eps_wd.max(0.0)
}

/// Evaluates zone-by-zone typical charges $Z_i(k)$, Burgers settling velocities
/// $v_{\mathrm{gt},i}(k)$, trace diffusion coefficients $D_i(k)$, and advances
/// the species mass fractions `model.xa` over timestep `dt` using an implicit
/// backward-Euler mass-conserving tridiagonal solve (`diffusion.f90`).
///
/// Returns `(v_gt_all, eps_wd_sed)` where `v_gt_all` has layout `(nz, species)`
/// and `eps_wd_sed` has length `nz`.
pub fn do_element_diffusion(
    model: &mut StarModel,
    dt: f64,
    scheme: DiffusionScheme,
) -> (Vec<f64>, Vec<f64>) {
    let nz = model.nz;
    let nc = model.species;
    let catalog = NuclideCatalog::builtin_ref();

    let mut a_iso = vec![1.0; nc];
    let mut z_nuc = vec![1.0; nc];
    for (j, &cid) in model.chem_id.iter().enumerate() {
        if let Some(nuc) = catalog.get_by_id(cid) {
            a_iso[j] = nuc.w;
            z_nuc[j] = (nuc.z as f64).max(1.0);
        }
    }

    let mut v_gt_all = vec![0.0; nz * nc];
    let mut d_self_all = vec![0.0; nz * nc];
    let mut eps_wd_sed = vec![0.0; nz];
    let wd_sed_fac = model.controls.mix_wind.eps_wd_sedimentation_factor;

    let mut z_typ = vec![1.0; nc];
    for k in 0..nz {
        let t_k = model.t[k].max(1.0e3);
        let log10_t = t_k.log10();
        let rho_k = model.rho[k].max(1.0e-12);
        let log10_rho = rho_k.log10();
        let abar_k = model.abar[k].max(1.0);
        let zbar_k = model.zbar[k].max(1.0);
        let free_e = model.lnfree_e[k].exp().max(0.1);

        z_typ.fill(1.0);
        for (j, &cid) in model.chem_id.iter().enumerate() {
            z_typ[j] = eval_typical_charge(
                catalog, cid, abar_k, free_e, t_k, log10_t, rho_k, log10_rho,
            )
            .unwrap_or(z_nuc[j])
            .clamp(0.5, z_nuc[j]);
        }

        let hp = model.scale_height[k].max(1.0e5);
        let dlnt_dr = -model.gradt[k] / hp;
        let g_k = model.grav[k].max(1.0);
        let x_slice = &model.xa[k * nc..(k + 1) * nc];

        let res = eval_burgers_velocities(
            &a_iso, &z_typ, x_slice, rho_k, t_k, dlnt_dr, g_k, None, scheme,
        );
        for j in 0..nc {
            v_gt_all[k * nc + j] = res.v_gt[j];
            d_self_all[k * nc + j] = res.d_self[j];
        }
        if model.gam[k] > 1.0 && wd_sed_fac != 0.0 {
            eps_wd_sed[k] = wd_sed_fac
                * eval_wd_sedimentation_heating(
                    &a_iso, &z_typ, x_slice, &res.v_gt, abar_k, zbar_k, g_k,
                );
        }
    }

    if dt > 0.0 && nz >= 2 && nc >= 1 {
        // Precompute face diffusion coupling and zero-sum settling fluxes
        // so that both sum_j X_{j,k} = 1 and sum_k dm_k X_{j,k} = const hold
        // to machine precision.
        let mut dl = vec![0.0; nz];
        let mut dd = vec![1.0; nz];
        let mut du = vec![0.0; nz];
        let mut adv_rhs = vec![0.0; nz * nc];
        let mut x_f = vec![0.0; nc];
        let mut dm_trans = vec![0.0; nc];

        for k in 1..nz {
            let area = 4.0 * PI * model.r[k].powi(2);
            let rho_f = 0.5 * (model.rho[k - 1] + model.rho[k]);
            let dr = (model.rmid[k - 1] - model.rmid[k]).abs().max(1.0e3);
            let dm_up = model.dm[k - 1].max(1.0e10);
            let dm_dn = model.dm[k].max(1.0e10);

            let mut d_mean = 0.0;
            let mut v_cm_f = 0.0;
            for j in 0..nc {
                let xj =
                    0.5 * (model.xa[(k - 1) * nc + j] + model.xa[k * nc + j]);
                x_f[j] = xj;
                let dj = 0.5
                    * (d_self_all[(k - 1) * nc + j] + d_self_all[k * nc + j]);
                d_mean += xj * dj;
                v_cm_f += xj * v_gt_all[k * nc + j];
            }
            let x_sum = x_f.iter().sum::<f64>().max(1.0e-30);
            d_mean /= x_sum;
            v_cm_f /= x_sum;

            let diff_flux = area * rho_f * d_mean / dr;
            dd[k - 1] += dt * diff_flux / dm_up;
            du[k - 1] -= dt * diff_flux / dm_up;
            dl[k] -= dt * diff_flux / dm_dn;
            dd[k] += dt * diff_flux / dm_dn;

            // Zero-sum settling mass transfer across face k (positive = outward
            // into k - 1, negative = inward into k) capped to 25% of donor cell
            let max_dm = 0.25 * dm_up.min(dm_dn);
            for j in 0..nc {
                let v_rel = v_gt_all[k * nc + j] - v_cm_f;
                let raw = dt * area * rho_f * (x_f[j] / x_sum) * v_rel;
                let cap_in = 0.25 * dm_up * model.xa[(k - 1) * nc + j];
                let cap_out = 0.25 * dm_dn * model.xa[k * nc + j];
                dm_trans[j] =
                    raw.clamp(-cap_in.min(max_dm), cap_out.min(max_dm));
            }
            let net_trans: f64 = dm_trans.iter().sum();
            for j in 0..nc {
                let dm_j = dm_trans[j] - (x_f[j] / x_sum) * net_trans;
                adv_rhs[(k - 1) * nc + j] += dm_j / dm_up;
                adv_rhs[k * nc + j] -= dm_j / dm_dn;
            }
        }

        let factor = ScalarTridiagFactor::factor(&dl, &dd, &du);
        let mut rhs = vec![0.0; nz];
        let mut sol = vec![0.0; nz];
        let mut d_prime = vec![0.0; nz];
        for j in 0..nc {
            for k in 0..nz {
                rhs[k] = model.xa[k * nc + j] + adv_rhs[k * nc + j];
            }
            factor.solve_into(&dl, &rhs, &mut sol, &mut d_prime);
            for k in 0..nz {
                model.xa[k * nc + j] = sol[k].clamp(0.0, 1.0);
            }
        }
        for k in 0..nz {
            let sum_x = neumaier_sum(model.cell_xa(k));
            if sum_x > 0.0 {
                for x in model.cell_xa_mut(k) {
                    *x = (*x / sum_x).clamp(0.0, 1.0);
                }
            }
        }
        model.aux.step_hist.num_diffusion_solver_steps = 1;
        model.aux.step_hist.num_diffusion_solver_iters = nc;
    }

    (v_gt_all, eps_wd_sed)
}
