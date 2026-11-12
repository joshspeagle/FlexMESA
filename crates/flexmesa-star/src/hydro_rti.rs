// SPDX-License-Identifier: LGPL-3.0-or-later
//! Duffell (2016) Rayleigh-Taylor instability (`RTI`) turbulent diffusivity
//! profiles (`set_dpdr_drhodr_info`, `set_rti_mixing_info` in
//! `star/private/mix_info.f90:1966-2237`) and coupled $\alpha_{\mathrm{RTI}}$
//! transport-source-sink equation (`do1_dalpha_rti_dt_eqn` in
//! `star/private/hydro_alpha_rti_eqns.f90:33-193`).
//!
//! In upstream MESA, `dPdr_dRhodr_info`, `eta_RTI`, `sig_RTI`, and
//! `sigmid_RTI` are evaluated once at the start of each step (outside Newton
//! iterations), while `alpha_RTI[k]` is solved implicitly inside the coupled
//! block-tridiagonal Henyey system with exact 3-point `(k - 1, k, k + 1)`
//! Jacobian entries.

use flexmesa_const::{FOUR_PI, LN10};

use crate::hydro_eqns::HenyeySystem;
use crate::model::StarModel;

/// Evaluates the start-of-step Rayleigh-Taylor baroclinic instability indicator
/// $(\partial P / \partial r)(\partial \rho / \partial r) \le 0$ across all
/// cells `k = 0..nz-1` (`set_dPdr_dRhodr_info` in
/// `star/private/mix_info.f90:2124-2237`).
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

    let mut p_face = vec![0.0_f64; nz];
    let mut rho_face = vec![0.0_f64; nz];
    p_face[0] = model.peos[0];
    rho_face[0] = model.rho[0];
    for k in 1..nz {
        let (alfa, beta) = model.get_face_weights(k);
        p_face[k] = if model.controls.u_flag
            && model.aux.p_face_start.get(k).copied().unwrap_or(-1.0) > 0.0
        {
            model.aux.p_face_start[k]
        } else {
            alfa * model.peos[k] + beta * model.peos[k - 1]
        };
        rho_face[k] = alfa * model.rho[k] + beta * model.rho[k - 1];
    }

    let min_mach = model.controls.alpha_rti_src_min_v_div_cs;
    let mut dpdr = vec![0.0_f64; nz];
    let mut drhodr = vec![0.0_f64; nz];
    for k in 0..(nz - 1) {
        let a_start = model
            .aux
            .alpha_rti_start
            .get(k)
            .copied()
            .unwrap_or(model.aux.alpha_rti[k]);
        let vel_k = if model.controls.u_flag {
            model.u[k].abs()
        } else if model.controls.v_flag {
            model.v[k].abs()
        } else {
            0.0
        };
        let cs_k = model.csound[k].max(1.0);
        if a_start <= 0.0 && min_mach > 0.0 && vel_k / cs_k < min_mach {
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

    // Optional 3-point smoothing (`rti_smooth_fraction = 0.5` default in MESA)
    let f = 0.5_f64;
    if nz >= 4 {
        let dp_orig = dpdr.clone();
        let dr_orig = drhodr.clone();
        for k in 1..(nz - 2) {
            dpdr[k] = (1.0 - f) * dp_orig[k]
                + 0.5 * f * (dp_orig[k - 1] + dp_orig[k + 1]);
            drhodr[k] = (1.0 - f) * dr_orig[k]
                + 0.5 * f * (dr_orig[k - 1] + dr_orig[k + 1]);
        }
    }

    for k in 0..nz {
        model.aux.dpdr_drhodr_info[k] = (dpdr[k] * drhodr[k]).min(0.0);
    }
}

/// Evaluates start-of-step Duffell (2016) RTI kinematic diffusivity
/// $\eta_{\mathrm{RTI},k} = C \alpha_{\mathrm{RTI,face},k} c_{s,k} r_k$ and
/// Lagrangian mass-diffusion coefficients `sig_rti` and `sigmid_rti`,
/// optionally augmenting `d_mix` and `sig` when
/// `composition_rti_diffusion_factor > 0` (`set_RTI_mixing_info` in
/// `star/private/mix_info.f90:1966-2121`).
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
    let dt = model.dt.max(1.0e-30);
    let mix_fac = if model.controls.do_mix {
        model.controls.mix_factor
    } else {
        1.0
    };

    for k in 1..nz {
        let (alfa, beta) = model.get_face_weights(k);
        let a_00 = model
            .aux
            .alpha_rti_start
            .get(k)
            .copied()
            .unwrap_or(model.aux.alpha_rti[k])
            .max(0.0);
        let a_m1 = model
            .aux
            .alpha_rti_start
            .get(k - 1)
            .copied()
            .unwrap_or(model.aux.alpha_rti[k - 1])
            .max(0.0);
        let alpha_face = alfa * a_00 + beta * a_m1;
        if alpha_face <= 0.0 {
            continue;
        }
        let cs_00 = model
            .aux
            .csound_start
            .get(k)
            .copied()
            .filter(|&c| c > 0.0)
            .unwrap_or(model.csound[k]);
        let cs_m1 = model
            .aux
            .csound_start
            .get(k - 1)
            .copied()
            .filter(|&c| c > 0.0)
            .unwrap_or(model.csound[k - 1]);
        let cs_face = alfa * cs_00 + beta * cs_m1;
        let eta_k = c_base * alpha_face * cs_face * model.r[k];
        model.aux.eta_rti[k] = eta_k;

        let rho_f = alfa * model.rho[k] + beta * model.rho[k - 1];
        let area_rho = FOUR_PI * model.r[k] * model.r[k] * rho_f;
        let dm_bar_k = model.dm_bar[k].max(1.0e-30);
        let mut sig_k = mix_fac * eta_k * (area_rho * area_rho) / dm_bar_k;
        if model.dt > 0.0 {
            sig_k = sig_k.min(1.0e6 * dm_bar_k / dt);
        }
        model.aux.sig_rti[k] = sig_k;

        if model.controls.composition_rti_diffusion_factor > 0.0 {
            let d_add = model.controls.composition_rti_diffusion_factor * eta_k;
            model.d_mix[k] += d_add;
            model.sig[k] +=
                model.controls.composition_rti_diffusion_factor * sig_k;
        }
    }

    for k in 0..nz {
        let a_k = model
            .aux
            .alpha_rti_start
            .get(k)
            .copied()
            .unwrap_or(model.aux.alpha_rti[k])
            .max(0.0);
        if a_k <= 0.0 {
            continue;
        }
        let cs_k = model
            .aux
            .csound_start
            .get(k)
            .copied()
            .filter(|&c| c > 0.0)
            .unwrap_or(model.csound[k]);
        let rmid = model.rmid[k];
        let etamid = c_base * a_k * cs_k * rmid;
        model.aux.etamid_rti[k] = etamid;
        let area_rho = FOUR_PI * rmid * rmid * model.rho[k];
        let dm_k = model.dm[k].max(1.0e-30);
        let mut sigmid = mix_fac * etamid * (area_rho * area_rho) / dm_k;
        if model.dt > 0.0 {
            sigmid = sigmid.min(1.0e6 * dm_k / dt);
        }
        model.aux.sigmid_rti[k] = sigmid;
    }

    // Populate diagnostic `source_plus_alpha_rti` and `source_minus_alpha_rti`
    let _ = LN10;
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

/// Evaluates `(source_plus, d_source_plus_da00, source_minus, d_source_minus_da00)`
/// for cell `k` (`hydro_alpha_rti_eqns.f90:91-147`).
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
    let r_s_00 = model
        .ln_r_start
        .get(k)
        .copied()
        .unwrap_or(model.ln_r[k])
        .exp();
    let r_s_in = if k + 1 < model.nz {
        model
            .ln_r_start
            .get(k + 1)
            .copied()
            .unwrap_or(model.ln_r[k + 1])
            .exp()
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
            let k_fac = model.controls.rti_d * cs_s / rmid_s;
            if a_00 > max_a {
                (k_fac * (a_00 * a_00 / max_a), 2.0 * k_fac * (a_00 / max_a))
            } else {
                (k_fac * a_00, k_fac)
            }
        } else {
            (0.0, 0.0)
        };

    // Baroclinic Rayleigh-Taylor growth source: `((A + B * a_00) / rho_start) * sqrt(-dPdr * drhodr)`
    let instab2 = -model.aux.dpdr_drhodr_info.get(k).copied().unwrap_or(0.0);
    let (source_plus, d_sp_da) = if instab2 > 0.0
        && model.q[k] >= model.controls.alpha_rti_src_min_q
        && model.q[k] <= model.controls.alpha_rti_src_max_q
        && a_start < model.controls.rti_max_alpha
    {
        let rho_s = model
            .lnd_start
            .get(k)
            .copied()
            .unwrap_or(model.lnd[k])
            .exp()
            .max(1.0e-30);
        let root = instab2.sqrt() / rho_s;
        (
            (model.controls.rti_a + model.controls.rti_b * a_00) * root,
            model.controls.rti_b * root,
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
    let diff_fac = if model.controls.do_mix {
        model.controls.alpha_rti_diffusion_factor
    } else {
        0.0
    };

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

    let x_scale_k = sys.x_scale[k * sys.nvar + i_var];
    let eqn_scale = (x_scale_k / dt).max(1.0);

    sys.equ[k * sys.nvar + i_equ] =
        (dadt_mix + source_plus - source_minus - dadt_actual) / eqn_scale;

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
