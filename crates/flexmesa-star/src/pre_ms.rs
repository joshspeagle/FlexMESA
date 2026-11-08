// SPDX-License-Identifier: LGPL-3.0-or-later
//! Pre-main-sequence $n = 1.5$ polytropic model builder (`build_pre_ms_model`
//! and `build1_pre_ms_model` in `star/private/pre_ms_model.f90`).

use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_const::{
    CLIGHT, CRAD, FOUR_PI, KERG, LN10, L_SUN, MP, M_SUN, R_SUN, SECYER,
    STANDARD_CGRAV,
};
use flexmesa_eos::{
    eos_dt_get, eos_pt_get, EosConfig, HelmTable, I_CHI_RHO, I_CHI_T, I_CP,
    I_ETA, I_GRAD_AD, I_LN_FREE_E, I_LN_PGAS, I_MU,
};
use flexmesa_kap::{get_kap_results, KapConfig};
use flexmesa_math::exp10;
use flexmesa_turb::{set_mlt, MltInputs, MltParams};

use crate::controls::{StarControls, StarError, StarJobControls};
use crate::model::StarModel;

/// Single shell produced during outward pre-MS polytrope integration.
#[derive(Debug, Clone, Copy)]
pub struct PreMsShell {
    /// Enclosed mass $m$ at outer face of shell ($\mathrm{g}$).
    pub m: f64,
    /// Outer radius $r$ of shell ($\mathrm{cm}$).
    pub r: f64,
    /// Luminosity $L$ at outer face ($\mathrm{erg\,s^{-1}}$).
    pub lum: f64,
    /// Cell temperature $T$ ($\mathrm{K}$).
    pub t: f64,
    /// Cell density $\rho$ ($\mathrm{g\,cm^{-3}}$).
    pub rho: f64,
}

/// Returns standard 8-isotope `basic.net` `chem_id` indices and homogeneous
/// initial mass fractions `xa` for `(initial_y, initial_z)`.
#[must_use]
pub fn default_basic_net_composition(
    initial_y: f64,
    initial_z: f64,
) -> (Vec<usize>, Vec<f64>) {
    let catalog = NuclideCatalog::builtin_ref();
    let names = ["h1", "he3", "he4", "c12", "n14", "o16", "ne20", "mg24"];
    let chem_id: Vec<usize> = names
        .iter()
        .map(|n| catalog.get_nuclide_index(n).unwrap_or(1))
        .collect();

    let z = initial_z.clamp(0.0, 1.0);
    let y = if initial_y < 0.0 {
        (0.24 + 2.0 * z).min(1.0 - z)
    } else {
        initial_y.clamp(0.0, 1.0 - z)
    };
    let x = (1.0 - y - z).max(0.0);
    let x_he3 = 2.98e-5 * y;
    let x_he4 = (y - x_he3).max(0.0);

    // GS98 relative metal weights for (c12, n14, o16, ne20, mg24)
    let z_weights = [0.1721, 0.0504, 0.4682, 0.1050, 0.2043];
    let mut xa = vec![
        x,
        x_he3,
        x_he4,
        z * z_weights[0],
        z * z_weights[1],
        z * z_weights[2],
        z * z_weights[3],
        z * z_weights[4],
    ];
    let sum: f64 = xa.iter().sum();
    if sum > 0.0 {
        for v in &mut xa {
            *v /= sum;
        }
    }
    (chem_id, xa)
}

#[allow(clippy::too_many_arguments)]
fn eval_pre_ms_gradt(
    ht: &HelmTable,
    eos_cfg: &EosConfig,
    kap_cfg: &KapConfig,
    controls: &StarControls,
    chem_id: &[usize],
    xa: &[f64],
    xh: f64,
    xz: f64,
    zbar: f64,
    xc: f64,
    xn: f64,
    xo: f64,
    rho: f64,
    t: f64,
    r: f64,
    m: f64,
    lum: f64,
) -> Result<(f64, f64, f64), StarError> {
    let eos = eos_dt_get(ht, rho, t, chem_id, xa, eos_cfg)
        .map_err(|e| StarError::MicrophysicsError(format!("eos error: {e}")))?;
    let log_rho = rho.log10();
    let log_t = t.log10();
    let lnfree_e = eos.res[I_LN_FREE_E];
    let d_lnfree_e_dlnd = eos.d_dlnd[I_LN_FREE_E];
    let d_lnfree_e_dlnt = eos.d_dln_t[I_LN_FREE_E];
    let eta = eos.res[I_ETA];
    let d_eta_dlnd = eos.d_dlnd[I_ETA];
    let d_eta_dlnt = eos.d_dln_t[I_ETA];
    let xne = rho * flexmesa_const::AVO * lnfree_e.exp();

    let kap_out = get_kap_results(
        kap_cfg,
        zbar,
        xh,
        xz,
        xc,
        xn,
        xo,
        xne,
        log_rho,
        log_t,
        lnfree_e,
        d_lnfree_e_dlnd,
        d_lnfree_e_dlnt,
        eta,
        d_eta_dlnd,
        d_eta_dlnt,
    )
    .map_err(|e| StarError::MicrophysicsError(format!("kap error: {e}")))?;

    let pgas = eos.res[I_LN_PGAS].exp();
    let prad = CRAD * t.powi(4) / 3.0;
    let p = pgas + prad;
    let grav = STANDARD_CGRAV * m / (r * r);
    let kap = kap_out.kap * controls.opacity_factor;
    let gradr = p * kap * lum / (FOUR_PI * CLIGHT * STANDARD_CGRAV * m * prad);
    let grada = eos.res[I_GRAD_AD];

    let mlt_params = MltParams {
        option: controls.mlt_option,
        mixing_length_alpha: controls.mixing_length_alpha,
        henyey_nu: 8.0,
        henyey_y: 1.0 / 3.0,
        max_conv_vel: 1.0e99,
    };
    let mlt_in = MltInputs {
        chi_t: eos.res[I_CHI_T],
        chi_rho: eos.res[I_CHI_RHO],
        cp: eos.res[I_CP],
        grav,
        lambda: controls.mixing_length_alpha * p / (grav * rho),
        rho,
        p,
        t,
        opacity: kap,
        gradr,
        grada,
        gradl: grada,
    };
    let mlt_out = set_mlt(&mlt_params, &mlt_in)
        .map_err(|e| StarError::MicrophysicsError(format!("mlt error: {e}")))?;
    Ok((p, eos.res[I_MU], mlt_out.grad_t))
}

/// Outward shooting integration of an $n = 1.5$ pre-MS model at trial central
/// density $\rho_c = \exp(\ln\rho_c)$ (`build1_pre_ms_model` in
/// `star/private/pre_ms_model.f90:363-666`).
///
/// Returns `(m_star_1, shells)` ordered from surface (`k = 0`) to center
/// (`k = nz - 1`).
///
/// # Errors
/// Returns [`StarError`] if EOS/opacity evaluation fails during integration.
#[allow(clippy::too_many_arguments)]
pub fn build1_pre_ms_model(
    ht: &HelmTable,
    eos_cfg: &EosConfig,
    kap_cfg: &KapConfig,
    controls: &StarControls,
    job: &StarJobControls,
    chem_id: &[usize],
    xa: &[f64],
    ln_rho_c: f64,
    t_c: f64,
    mstar: f64,
) -> Result<(f64, Vec<PreMsShell>), StarError> {
    let catalog = NuclideCatalog::builtin_ref();
    let comp = get_composition_info(catalog, chem_id, xa, false)
        .map_err(|e| StarError::InvalidInput(e.to_string()))?;
    let xc = xa.get(3).copied().unwrap_or(0.0);
    let xn = xa.get(4).copied().unwrap_or(0.0);
    let xo = xa.get(5).copied().unwrap_or(0.0);

    let rho_c = ln_rho_c.exp();
    let eos_c = eos_dt_get(ht, rho_c, t_c, chem_id, xa, eos_cfg)
        .map_err(|e| StarError::MicrophysicsError(format!("eos_c: {e}")))?;
    let pgas_c = eos_c.res[I_LN_PGAS].exp();
    let prad_c = CRAD * t_c.powi(4) / 3.0;
    let p_c = pgas_c + prad_c;

    // Estimated total luminosity `l_star` (`pre_ms_model.f90:407`)
    let r_est = (1.430 * mstar / (FOUR_PI * rho_c / 3.0)).cbrt();
    let l_star = L_SUN
        * (mstar / M_SUN).powf(1.5)
        * (r_est / R_SUN).powi(2)
        * exp10(job.pre_ms_log_lum_factor);

    // Inner-most shell (`q = 1e-5`, `pre_ms_model.f90:423-444`)
    let q_c = 1.0e-5;
    let m_c = q_c * mstar;
    let r_c = (m_c / (FOUR_PI * rho_c / 3.0)).cbrt();
    let lum_c = q_c * l_star;
    let (_, _, gradt_c) = eval_pre_ms_gradt(
        ht, eos_cfg, kap_cfg, controls, chem_id, xa, comp.xh, comp.xz,
        comp.zbar, xc, xn, xo, rho_c, t_c, r_c, m_c, lum_c,
    )?;

    let dp_c = -STANDARD_CGRAV * m_c * rho_c / (2.0 * r_c);
    let p_outer_c = (p_c + dp_c).max(0.5 * p_c);
    let dlnp_c = p_outer_c.ln() - p_c.ln();
    let t_outer_c = t_c * (1.0 + dlnp_c * gradt_c).max(0.5);

    let mut shells_center_to_surf = Vec::with_capacity(1200);
    shells_center_to_surf.push(PreMsShell {
        m: m_c,
        r: r_c,
        lum: lum_c,
        t: t_c,
        rho: rho_c,
    });

    let d_log10_p = job.pre_ms_d_log10_p.max(1.0e-4);
    let mut m_prev = m_c;
    let mut r_prev = r_c;
    let mut t_prev = t_outer_c;
    let mut p_prev = p_outer_c;
    let mut pgas_prev =
        (p_prev - CRAD * t_prev.powi(4) / 3.0).max(0.1 * p_prev);
    let mut rho_prev = rho_c;
    let mut gradt_prev = gradt_c;

    let max_shells = 4000_usize;
    for _step in 0..max_shells {
        let log_pgas_mid = pgas_prev.log10() - 0.5 * d_log10_p;
        let pgas_mid = exp10(log_pgas_mid);

        let mut m_curr = m_prev;
        let mut r_curr = r_prev;
        let mut t_mid = t_prev;
        let mut rho_mid = rho_prev;
        let mut p_mid = p_prev;
        let mut gradt_mid = gradt_prev;

        // 2 predictor-corrector passes (`i = 1..=2`, `pre_ms_model.f90:512`)
        for i_pass in 1..=2 {
            let dlnp_half = if i_pass == 1 {
                -0.5 * d_log10_p * LN10 * (pgas_prev / p_prev)
            } else {
                p_mid.ln() - p_prev.ln()
            };

            for _ii in 1..=2 {
                t_mid = (t_prev * (1.0 + dlnp_half * gradt_mid)).max(100.0);
                let log_t_mid = t_mid.log10();
                if let Ok(pt_out) = eos_pt_get(
                    ht,
                    chem_id,
                    xa,
                    log_pgas_mid,
                    log_t_mid,
                    eos_cfg,
                ) {
                    rho_mid = pt_out.rho;
                } else {
                    rho_mid = pgas_mid * comp.abar * MP
                        / (KERG * t_mid * (1.0 + comp.zbar));
                }
                p_mid = pgas_mid + CRAD * t_mid.powi(4) / 3.0;
            }

            let dlnp_full = 2.0 * (p_mid.ln() - p_prev.ln());
            let dp_step = p_prev * dlnp_full.exp_m1();

            // 10 inner iterations for (m_curr, r_curr) (`pre_ms_model.f90:547`)
            for _j in 1..=10 {
                let r_bar = 0.5 * (r_prev + r_curr);
                let m_bar = 0.5 * (m_prev + m_curr);
                let dm_step = (-FOUR_PI * r_bar.powi(4) * dp_step
                    / (STANDARD_CGRAV * m_bar))
                    .max(1.0e-12 * mstar);
                let dr_step =
                    (dm_step / (FOUR_PI * r_bar * r_bar * rho_mid)).max(0.0);
                m_curr = m_prev + dm_step;
                r_curr = r_prev + dr_step;
            }

            let r_mid = 0.5 * (r_prev + r_curr);
            let m_mid = 0.5 * (m_prev + m_curr);
            let lum_mid = (m_mid / mstar) * l_star;
            if let Ok((_, _, gt)) = eval_pre_ms_gradt(
                ht, eos_cfg, kap_cfg, controls, chem_id, xa, comp.xh, comp.xz,
                comp.zbar, xc, xn, xo, rho_mid, t_mid, r_mid, m_mid, lum_mid,
            ) {
                gradt_mid = gt;
            }
        }

        let lum_curr = (m_curr / mstar) * l_star;
        shells_center_to_surf.push(PreMsShell {
            m: m_curr,
            r: r_curr,
            lum: lum_curr,
            t: t_mid,
            rho: rho_mid,
        });

        let log_pgas_next = pgas_prev.log10() - d_log10_p;
        pgas_prev = exp10(log_pgas_next);
        let dlnp_half = p_mid.ln() - p_prev.ln();
        t_prev = (t_mid * (1.0 + dlnp_half * gradt_mid)).max(100.0);
        p_prev = pgas_prev + CRAD * t_prev.powi(4) / 3.0;
        rho_prev = rho_mid;
        m_prev = m_curr;
        r_prev = r_curr;
        gradt_prev = gradt_mid;

        if p_mid.log10() <= job.pre_ms_log_p_surf_limit
            || t_mid.log10() <= job.pre_ms_log_t_surf_limit
            || m_curr >= 5.0 * mstar
        {
            break;
        }
    }

    let m_star_1 = m_prev;
    shells_center_to_surf.reverse();
    Ok((m_star_1, shells_center_to_surf))
}

/// Constructs an initial $n = 1.5$ pre-main-sequence [`StarModel`] by shooting
/// for $\ln\rho_c$ such that $M_{*,1}(\ln\rho_c) = M_*$ (`build_pre_ms_model`
/// in `star/private/pre_ms_model.f90:96-360`).
///
/// # Errors
/// Returns [`StarError`] if root-finding or model construction fails.
pub fn build_pre_ms_model(
    controls: StarControls,
    job: StarJobControls,
) -> Result<StarModel, StarError> {
    let ht = HelmTable::builtin();
    let eos_cfg = EosConfig::default();
    let kap_cfg = KapConfig {
        zbase: controls.zbase,
        ..KapConfig::default()
    };

    let (chem_id, xa_single) =
        default_basic_net_composition(controls.initial_y, controls.initial_z);
    let mstar = controls.initial_mass * M_SUN;
    let t_c = job.pre_ms_t_c;

    // Initial central density guess (`pre_ms_model.f90:155-204`)
    let rho_c_guess = if job.pre_ms_guess_rho_c > 0.0 {
        job.pre_ms_guess_rho_c
    } else {
        let mu_est = 0.61;
        let r_rsun = 0.6929 * mu_est * controls.initial_mass * (1.0e6 / t_c);
        let r_cgs = (r_rsun * R_SUN).max(0.1 * R_SUN);
        (1.430 * mstar / (FOUR_PI * r_cgs.powi(3) / 3.0)).clamp(1.0e-5, 1.0e2)
    };

    let mut x1 = rho_c_guess.ln();
    let eval_f = |ln_rc: f64| -> Result<(f64, Vec<PreMsShell>), StarError> {
        let (m1, shells) = build1_pre_ms_model(
            ht, &eos_cfg, &kap_cfg, &controls, &job, &chem_id, &xa_single,
            ln_rc, t_c, mstar,
        )?;
        Ok(((mstar - m1) / mstar, shells))
    };

    let (mut f1, mut best_shells) = eval_f(x1)?;
    if f1.abs() > 1.0e-3 {
        // Bracket root by stepping ln_rho_c
        let dx = if f1 > 0.0 { 0.15 * LN10 } else { -0.15 * LN10 };
        let mut x2 = x1;
        let mut f2 = f1;
        for _ in 0..30 {
            x2 += dx;
            let (fx, shells) = eval_f(x2)?;
            best_shells = shells;
            f2 = fx;
            if (f1 > 0.0 && f2 <= 0.0) || (f1 < 0.0 && f2 >= 0.0) {
                break;
            }
            x1 = x2;
            f1 = f2;
        }

        // Bisection + secant refinement on [x1, x2]
        let mut a = x1;
        let mut fa = f1;
        let mut b = x2;
        let mut fb = f2;
        if fa * fb <= 0.0 {
            for _ in 0..16 {
                let mid = if (fb - fa).abs() > 1.0e-12 {
                    let sec = a - fa * (b - a) / (fb - fa);
                    if sec > a.min(b) && sec < a.max(b) {
                        sec
                    } else {
                        0.5 * (a + b)
                    }
                } else {
                    0.5 * (a + b)
                };
                let (fm, shells) = eval_f(mid)?;
                best_shells = shells;
                if fm.abs() <= 1.0e-3 || (b - a).abs() <= 1.0e-4 {
                    break;
                }
                if fa * fm <= 0.0 {
                    b = mid;
                    fb = fm;
                } else {
                    a = mid;
                    fa = fm;
                }
            }
        }
    }

    let nz = best_shells.len();
    if nz < 2 {
        return Err(StarError::ConvergenceFailure(String::from(
            "pre-MS builder produced fewer than 2 zones",
        )));
    }
    let m_total = best_shells[0].m;
    let mut model = StarModel::new(nz, chem_id, job, controls);
    model.mstar = mstar;
    model.mstar_start = mstar;
    model.mstar_old = mstar;
    model.model_number = 1;
    model.star_age = 0.0;
    model.dt = 1.0e2 * SECYER;
    model.dt_next = model.dt;
    model.dt_old = model.dt;

    for k in 0..nz {
        let sh = &best_shells[k];
        model.lnd[k] = sh.rho.ln();
        model.ln_t[k] = sh.t.ln();
        model.ln_r[k] = sh.r.ln();
        model.lum[k] = sh.lum;
        let m_inner = if k + 1 < nz {
            best_shells[k + 1].m
        } else {
            0.0
        };
        model.dq[k] = ((sh.m - m_inner) / m_total).max(1.0e-12);
        model.cell_xa_mut(k).copy_from_slice(&xa_single);
    }

    model.finish_init_model()?;

    Ok(model)
}
