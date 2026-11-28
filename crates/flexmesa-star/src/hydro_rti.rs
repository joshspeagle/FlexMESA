// SPDX-License-Identifier: GPL-3.0-or-later
//! Duffell (2016) Rayleigh-Taylor instability (`RTI`) turbulent diffusivity
//! profiles (`set_dpdr_drhodr_info`, `set_rti_mixing_info` in
//! `star/private/mix_info.f90:1960-2240`) and coupled $\alpha_{\mathrm{RTI}}$
//! transport-source-sink equation (`do1_dalpha_rti_dt_eqn` in
//! `star/private/hydro_alpha_rti_eqns.f90:33-193`).
//!
//! In upstream MESA, `dPdr_dRhodr_info`, `eta_RTI`, `sig_RTI`, and
//! `sigmid_RTI` are evaluated once at the start of each step (outside Newton
//! iterations), while `alpha_RTI[k]` is solved implicitly inside the coupled
//! block-tridiagonal Henyey system with exact 3-point `(k - 1, k, k + 1)`
//! Jacobian entries.

use flexmesa_const::{FOUR_PI, M_SUN};

use crate::hydro_eqns::HenyeySystem;
use crate::model::StarModel;
use crate::report_utils::{csound_face_at, eval_shock_info};

/// Upstream MESA `26.4.1` default `RTI_log_max_boost` (`controls.defaults:8601`).
const RTI_LOG_MAX_BOOST: f64 = 3.0;
/// Upstream MESA `26.4.1` default `RTI_m_full_boost` in solar masses (`controls.defaults:8602`).
const RTI_M_FULL_BOOST_MSUN: f64 = 4.0;
/// Upstream MESA `26.4.1` default `RTI_m_no_boost` in solar masses (`controls.defaults:8603`).
const RTI_M_NO_BOOST_MSUN: f64 = 5.0;
/// Upstream MESA `26.4.1` default `RTI_dm_for_center_eta_nondecreasing` in solar masses
/// (`controls.defaults:8594`).
const RTI_DM_FOR_CENTER_ETA_NONDECREASING_MSUN: f64 = 0.02;

/// Evaluates the start-of-step Rayleigh-Taylor baroclinic instability indicator
/// $(\partial P / \partial r)(\partial \rho / \partial r) \le 0$ across all
/// cells `k = 0..nz-1` (`set_dPdr_dRhodr_info` in
/// `star/private/mix_info.f90:2129-2240`).
pub fn set_dpdr_drhodr_info(model: &mut StarModel) {
    let nz = model.nz;
    if nz == 0 {
        return;
    }
    if model.aux.dpdr_drhodr_info.len() != nz {
        model.aux.dpdr_drhodr_info.resize(nz, 0.0);
    }
    if !model.controls.rti_flag || nz < 3 {
        model.aux.dpdr_drhodr_info.fill(0.0);
        return;
    }

    let p_face: Vec<f64> = (0..nz)
        .map(|k| model.face_value_alfa_beta(&model.peos, k))
        .collect();
    let rho_face: Vec<f64> = (0..nz)
        .map(|k| model.face_value_alfa_beta(&model.rho, k))
        .collect();

    let min_mach = model.controls.alpha_rti_src_min_v_div_cs;
    let mut dpdr = vec![0.0_f64; nz];
    let mut drhodr = vec![0.0_f64; nz];
    for k in 1..(nz - 1) {
        let a_start = model
            .aux
            .alpha_rti_start
            .get(k)
            .copied()
            .unwrap_or(model.aux.alpha_rti[k]);
        let vel_k = if model.controls.u_flag {
            model.u[k]
        } else if model.controls.v_flag {
            model.v[k]
        } else {
            0.0
        };
        let cs_k = model.csound[k];
        // Guard `cs_k <= 0.0` before computing Mach ratio (`mix_info.f90:2166-2168`).
        if a_start == 0.0 && (cs_k <= 0.0 || vel_k / cs_k < min_mach) {
            continue;
        }
        let r_in = model.r[k + 1];
        let rmid = 0.5 * (model.r[k] + r_in);
        let dr_k = (model.dm[k]
            / (FOUR_PI * rmid * rmid * model.rho[k].max(1.0e-30)))
        .max(1.0e-30);
        dpdr[k] = (p_face[k] - p_face[k + 1]) / dr_k;
        drhodr[k] = (rho_face[k] - rho_face[k + 1]) / dr_k;
    }

    // MESA `26.4.1` defaults leave `RTI_smooth_mass = 0`,
    // `RTI_smooth_iterations = 0`, and `RTI_smooth_fraction = 1`
    // (`star/defaults/controls.defaults:3100-3102`,
    // `star/private/mix_info.f90:2185-2216`), so `dpdr` and `drhodr`
    // remain unsmoothed.
    for k in 0..nz {
        model.aux.dpdr_drhodr_info[k] = (dpdr[k] * drhodr[k]).min(0.0);
    }

    refresh_alpha_rti_diagnostic_sources(model);
}

fn refresh_alpha_rti_diagnostic_sources(model: &mut StarModel) {
    let nz = model.nz;
    if model.aux.source_plus_alpha_rti.len() != nz {
        model.aux.source_plus_alpha_rti.resize(nz, 0.0);
    }
    if model.aux.source_minus_alpha_rti.len() != nz {
        model.aux.source_minus_alpha_rti.resize(nz, 0.0);
    }
    for k in 0..nz {
        let (sp, _, sm, _) = eval_alpha_rti_sources(model, k);
        model.aux.source_plus_alpha_rti[k] = sp;
        model.aux.source_minus_alpha_rti[k] = sm;
    }
}

/// Computes Lagrangian RTI mass-diffusion coefficients `sig` from kinematic
/// diffusivity `eta`, area-density factor `4*pi*r^2*rho`, and zone mass
/// spacing `dm_bar`, gating out cells ahead of the shock and capping at
/// `sig_term_limit * dm_bar / dt` when `max_sig >= 1` (`get_RTI_sigmas` in
/// `star/private/mix_info.f90:2079-2126`).
#[allow(clippy::too_many_arguments)]
fn get_rti_sigmas(
    sig: &mut [f64],
    eta: &[f64],
    rho: &[f64],
    r: &[f64],
    dm_bar: &[f64],
    q: &[f64],
    shock_q: f64,
    mix_factor: f64,
    sig_term_limit: f64,
    dt: f64,
) {
    let nz = sig.len();
    if nz == 0 {
        return;
    }
    sig[0] = 0.0;
    let mut max_sig = 0.0_f64;
    for k in 1..nz {
        let d = eta[k];
        if d <= 0.0 || (q[k] > shock_q && shock_q > 0.0) {
            sig[k] = 0.0;
            continue;
        }
        let f1 = FOUR_PI * r[k] * r[k] * rho[k];
        let f = f1 * f1;
        let cdcterm = mix_factor * d * f;
        let mut sig_k = cdcterm / dm_bar[k];
        if !sig_k.is_finite() {
            sig_k = 1.0e99;
        }
        sig[k] = sig_k;
        if sig_k > max_sig {
            max_sig = sig_k;
        }
    }

    if max_sig < 1.0 || dt <= 0.0 {
        return;
    }
    for k in 0..nz {
        if sig[k] == 0.0 {
            continue;
        }
        let siglim = sig_term_limit * dm_bar[k] / dt;
        if sig[k] > siglim {
            sig[k] = siglim;
        }
    }
}

/// Evaluates start-of-step Duffell (2016) RTI kinematic diffusivity
/// $\eta_{\mathrm{RTI},k} = C \alpha_{\mathrm{RTI,face},k} c_{s,k} r_k$ and
/// Lagrangian mass-diffusion coefficients `sig_rti` and `sigmid_rti` without
/// mutating convective `d_mix` or `sig` (`set_RTI_mixing_info` in
/// `star/private/mix_info.f90:1960-2076`, `star/private/micro.f90:96-100`,
/// `star/private/star_utils.f90:1328-1403`).
pub fn set_rti_mixing_info(model: &mut StarModel) {
    let nz = model.nz;
    if nz == 0 {
        return;
    }
    for v in [
        &mut model.aux.eta_rti,
        &mut model.aux.etamid_rti,
        &mut model.aux.sig_rti,
        &mut model.aux.sigmid_rti,
    ] {
        if v.len() != nz {
            v.resize(nz, 0.0);
        }
        v.fill(0.0);
    }
    if !model.controls.rti_flag || model.controls.rti_c <= 0.0 {
        return;
    }

    let c_base = model.controls.rti_c;
    let max_boost = flexmesa_math::exp10(RTI_LOG_MAX_BOOST);
    let m_full_boost = RTI_M_FULL_BOOST_MSUN * M_SUN;
    let m_no_boost = RTI_M_NO_BOOST_MSUN * M_SUN;
    let dm_for_center_eta_nondecreasing =
        RTI_DM_FOR_CENTER_ETA_NONDECREASING_MSUN * M_SUN;
    let mut min_eta = -1.0_f64;

    // Outside-in pass sets `eta_rti` and `etamid_rti` using current `alpha_rti`
    // and `csound_face_at` (`mix_info.f90:2008-2060`, `micro.f90:96-100`),
    // applying the `4-5 M_sun` core mass boost and inner `0.02 M_sun`
    // nondecreasing floor (`controls.defaults:8594, 8601-8603`).
    for k in 0..nz {
        let mut c = c_base;
        let m_k = model.m[k];
        if m_k < m_no_boost {
            if m_k <= m_full_boost {
                c *= max_boost;
            } else {
                let alfa = (m_no_boost - m_k) / (m_no_boost - m_full_boost);
                c *= flexmesa_math::exp10(alfa * RTI_LOG_MAX_BOOST);
            }
        }

        let a_k = model.aux.alpha_rti.get(k).copied().unwrap_or(0.0);
        if k > 0 {
            let (alfa, beta) = model.get_face_weights(k);
            let a_m1 = model.aux.alpha_rti.get(k - 1).copied().unwrap_or(0.0);
            let alpha_face = alfa * a_k + beta * a_m1;
            let cs = csound_face_at(model, k);
            let mut eta_k = c * alpha_face * cs * model.r[k];
            if m_k - model.m_center <= dm_for_center_eta_nondecreasing {
                if min_eta < 0.0 {
                    min_eta = eta_k;
                } else if min_eta > eta_k {
                    eta_k = min_eta;
                }
            }
            model.aux.eta_rti[k] = eta_k;
        }

        model.aux.etamid_rti[k] =
            min_eta.max(c * a_k * model.csound[k] * model.rmid[k]);
    }

    let shock_q = eval_shock_info(model).q;
    let rho_face: Vec<f64> = (0..nz)
        .map(|k| model.face_value_alfa_beta(&model.rho, k))
        .collect();

    get_rti_sigmas(
        &mut model.aux.sig_rti,
        &model.aux.eta_rti,
        &rho_face,
        &model.r,
        &model.dm_bar,
        &model.q,
        shock_q,
        model.controls.mix_factor,
        model.controls.sig_term_limit,
        model.dt,
    );

    if model.controls.v_flag {
        get_rti_sigmas(
            &mut model.aux.sigmid_rti,
            &model.aux.etamid_rti,
            &model.rho,
            &model.rmid,
            &model.dm,
            &model.q,
            shock_q,
            model.controls.mix_factor,
            model.controls.sig_term_limit,
            model.dt,
        );
    }

    refresh_alpha_rti_diagnostic_sources(model);
}

/// Augments `d_mix` and `sig` with `composition_rti_diffusion_factor * (eta_rti, sig_rti)`
/// once per fresh convective mixing pass (`mix_info.f90:2038-2046`).
pub fn augment_mixing_with_rti(model: &mut StarModel) {
    let f_comp = model.controls.composition_rti_diffusion_factor;
    if !model.controls.rti_flag
        || !model.controls.do_mix
        || model.controls.mix_factor <= 0.0
        || f_comp <= 0.0
    {
        return;
    }
    let nz = model.nz;
    for k in 1..nz {
        let eta_k = model.aux.eta_rti.get(k).copied().unwrap_or(0.0);
        let sig_k = model.aux.sig_rti.get(k).copied().unwrap_or(0.0);
        if eta_k > 0.0 || sig_k > 0.0 {
            model.d_mix[k] += f_comp * eta_k;
            model.sig[k] += f_comp * sig_k;
        }
    }
}

/// Evaluates `(source_plus, d_source_plus_da00, source_minus, d_source_minus_da00)`
/// for cell `k` (`hydro_alpha_rti_eqns.f90:110-142`).
#[must_use]
pub fn eval_alpha_rti_sources(
    model: &StarModel,
    k: usize,
) -> (f64, f64, f64, f64) {
    let a_00 = model.aux.alpha_rti.get(k).copied().unwrap_or(0.0);
    let a_start = model.aux.alpha_rti_start.get(k).copied().unwrap_or(a_00);

    // Sink term (`hydro_alpha_rti_eqns.f90:112-123`):
    // `RTI_D = s% RTI_D * max(1, a_00 / RTI_max_alpha)`
    // `source_minus = RTI_D * a_00 * cs / rmid`
    let r_s_00 = flexmesa_math::exp(model.ln_r_start[k]);
    let r_s_in = if k + 1 < model.nz {
        flexmesa_math::exp(model.ln_r_start[k + 1])
    } else {
        model.r_center
    };
    let rmid_s = (0.5 * (r_s_00 + r_s_in)).max(1.0);
    let cs_s = model
        .aux
        .csound_start
        .get(k)
        .copied()
        .filter(|&c| c > 0.0)
        .unwrap_or(model.csound[k]);

    let (source_minus, d_sm_da) =
        if a_00 > 0.0 && model.controls.rti_d > 0.0 && cs_s > 0.0 {
            let max_a = model.controls.rti_max_alpha.max(1.0e-30);
            if a_00 > max_a {
                let rti_d_eff = model.controls.rti_d * (a_00 / max_a);
                let source_minus = rti_d_eff * a_00 * cs_s / rmid_s;
                let d_sm_da = 2.0 * rti_d_eff * cs_s / rmid_s;
                (source_minus, d_sm_da)
            } else {
                let rti_d_eff = model.controls.rti_d;
                let source_minus = rti_d_eff * a_00 * cs_s / rmid_s;
                let d_sm_da = rti_d_eff * cs_s / rmid_s;
                (source_minus, d_sm_da)
            }
        } else {
            (0.0, 0.0)
        };

    // Baroclinic Rayleigh-Taylor growth source (`hydro_alpha_rti_eqns.f90:125-142`):
    // `A_plus_B_div_rho = (RTI_A + RTI_B * a_00) / rho_start`
    // `source_plus = A_plus_B_div_rho * sqrt(-dPdr_dRhodr_info)`
    let instab2 = -model.aux.dpdr_drhodr_info.get(k).copied().unwrap_or(0.0);
    let (source_plus, d_sp_da) = if instab2 > 0.0
        && model.q[k] >= model.controls.alpha_rti_src_min_q
        && model.q[k] <= model.controls.alpha_rti_src_max_q
        && a_start < model.controls.rti_max_alpha
    {
        let rho_s = flexmesa_math::exp(
            model.lnd_start.get(k).copied().unwrap_or(model.lnd[k]),
        )
        .max(1.0e-30);
        let instab = instab2.sqrt();
        let a_plus_b_div_rho =
            (model.controls.rti_a + model.controls.rti_b * a_00) / rho_s;
        (
            a_plus_b_div_rho * instab,
            (model.controls.rti_b / rho_s) * instab,
        )
    } else {
        (0.0, 0.0)
    };

    (source_plus, d_sp_da, source_minus, d_sm_da)
}

/// Assembles the coupled $\alpha_{\mathrm{RTI}}$ conservation residual and
/// 3-cell `(k - 1, k, k + 1)` analytical Jacobian entries for zone `k`
/// (`do1_dalpha_RTI_dt_eqn` in `star/private/hydro_alpha_rti_eqns.f90:33-193`).
pub fn do1_dalpha_rti_dt_eqn(
    model: &StarModel,
    k: usize,
    sys: &mut HenyeySystem,
) {
    let (Some(i_var), Some(i_equ)) =
        (model.layout.i_alpha_rti, model.layout.i_dalpha_rti_dt)
    else {
        return;
    };
    if i_var >= sys.nvar || i_equ >= sys.nvar {
        return;
    }

    let nz = model.nz;
    let dt = model.dt.max(1.0e-30);
    let dm_k = model.dm[k].max(1.0e-30);
    // Upstream `hydro_alpha_rti_eqns.f90:71` reads `alpha_RTI_diffusion_factor` directly.
    let diff_fac = model.controls.alpha_rti_diffusion_factor;

    let a_00 = model.aux.alpha_rti[k];
    let a_start = model.aux.alpha_rti_start.get(k).copied().unwrap_or(a_00);

    let sig00 = if k > 0 && diff_fac > 0.0 {
        diff_fac * model.aux.sig_rti.get(k).copied().unwrap_or(0.0)
    } else {
        0.0
    };
    let sigp1 = if k + 1 < nz && diff_fac > 0.0 {
        diff_fac * model.aux.sig_rti.get(k + 1).copied().unwrap_or(0.0)
    } else {
        0.0
    };

    let flux00 = if k > 0 {
        -sig00 * (model.aux.alpha_rti[k - 1] - a_00)
    } else {
        0.0
    };
    let fluxp1 = if k + 1 < nz {
        -sigp1 * (a_00 - model.aux.alpha_rti[k + 1])
    } else {
        0.0
    };
    // Note: in `hydro_alpha_rti_eqns.f90:74-86`, `flux00 = -sig00 * (a_m1 - a_00)`,
    // `fluxp1 = -sigp1 * (a_00 - a_p1)`, `dadt_mix = (fluxp1 - flux00) / dm =
    // (sig00 * (a_m1 - a_00) - sigp1 * (a_00 - a_p1)) / dm`.
    let dadt_mix = (fluxp1 - flux00) / dm_k;

    let (source_plus, d_sp_da, source_minus, d_sm_da) =
        eval_alpha_rti_sources(model, k);
    let dadt_actual = (a_00 - a_start) / dt;

    // Upstream `hydro_alpha_rti_eqns.f90:157` scales by `x_scale(i_dalpha_RTI_dt, k) / dt`.
    let x_scale_k = sys.x_scale[k * sys.nvar + i_equ];
    let eqn_scale = (x_scale_k / dt).max(1.0);

    let dadt_source = source_plus - source_minus;
    let dadt_expected = dadt_mix + dadt_source;
    sys.equ[k * sys.nvar + i_equ] = (dadt_expected - dadt_actual) / eqn_scale;

    let d_diag =
        (-(sig00 + sigp1) / dm_k + d_sp_da - d_sm_da - 1.0 / dt) / eqn_scale;
    sys.add_partial(k, i_equ, k, i_var, d_diag);
    if k > 0 && sig00 != 0.0 {
        sys.add_partial(k, i_equ, k - 1, i_var, (sig00 / dm_k) / eqn_scale);
    }
    if k + 1 < nz && sigp1 != 0.0 {
        sys.add_partial(k, i_equ, k + 1, i_var, (sigp1 / dm_k) / eqn_scale);
    }
}
