// SPDX-License-Identifier: LGPL-3.0-or-later
//! Pre-main-sequence model builder (`build_pre_ms_model`, `pre_ms_f`,
//! `build1_pre_ms_model` and `eval_gradT` in
//! `star/private/pre_ms_model.f90`).
//!
//! The builder integrates hydrostatic equilibrium outward from a trial
//! central density $\rho_c$ at fixed central temperature $T_c$, with a uniform
//! gravothermal energy release $\epsilon_{\rm grav}$ that sets $L(m)$, until
//! the surface pressure or temperature limit is reached. The mass of that
//! model is a function of $\ln\rho_c$, and upstream's `look_for_brackets` and
//! `safe_root` find the $\ln\rho_c$ whose model has the requested mass. Every
//! evaluation rebuilds the model, so the model kept is the one built by the
//! last evaluation the root finder makes, exactly as upstream keeps the
//! `s% xh`, `s% q` and `s% dq` of its last `pre_ms_f` call.

use std::path::Path;

use flexmesa_chem::{chem_xsol, get_composition_info, NuclideCatalog};
use flexmesa_const::{
    MixingType, AVO, CLIGHT, CRAD, FOUR_THIRDS_PI, KERG, LN10, L_SUN, M_SUN,
    PI, PI4, R_SUN, SECYER, STANDARD_CGRAV,
};
use flexmesa_eos::{
    eos_dt_get, eos_dt_get_rho, EosConfig, EosEvalOutput, HelmTable,
    I_CHI_RHO, I_CHI_T, I_CP, I_ETA, I_GRAD_AD, I_LN_FREE_E, I_LN_PGAS,
};
use flexmesa_kap::{get_kap_results, KapConfig};
use flexmesa_math::{exp, exp10, log, log10, pow, pow2, pow3, pow4};
use flexmesa_num::{look_for_brackets, safe_root_with_brackets, NumError};
use flexmesa_turb::{
    set_mlt, set_semiconvection, MltInputs, MltParams, SemiconvectionInputs,
    SemiconvectionParams,
};
use flexmesa_utils::is_bad;

use crate::accretion_xa::standard_metals_xa;
use crate::controls::{StarControls, StarError, StarJobControls};
use crate::hydro_vars::eval_superad_reduction;
use crate::mesh::{normalize_dqs_slice, set_qs_slice};
use crate::model::StarModel;
use crate::star_ad::StarAdOrder1;

/// `one_third` and `two_thirds` (`const/public/const_def.f90:58-59`).
const ONE_THIRD: f64 = 1.0 / 3.0;
const TWO_THIRDS: f64 = 2.0 / 3.0;
/// `arg_not_provided` (`const/public/const_def.f90:145`).
const ARG_NOT_PROVIDED: f64 = -9.0e99;
/// `MAX_ITER_FOR_SOLVE` (`star/private/eos_support.f90:38`).
const MAX_ITER_FOR_SOLVE: usize = 100;
/// `imax` for `look_for_brackets` and `safe_root`
/// (`star/private/pre_ms_model.f90:48`).
const IMAX: usize = 100;
/// `min_mass_for_create_pre_ms` (`star/private/init.f90:877`).
const MIN_MASS_FOR_CREATE_PRE_MS: f64 = 0.03;
/// `&controls initial_he3` default (`star/defaults/controls.defaults:87`).
/// The key is not parsed, and an inlist that sets it is rejected as an
/// unsupported control, so the default always applies.
const CONTROLS_INITIAL_HE3: f64 = -1.0;
/// `&star_job dump_missing_metals_into_heaviest` default
/// (`star/defaults/star_job.defaults:739`). Not parsed; an inlist that sets
/// it is rejected as an unsupported control.
const DUMP_MISSING_METALS_INTO_HEAVIEST: bool = true;

/// `Radiation_Pressure` (`eos/public/eos_lib.f90:88-92`).
fn radiation_pressure(t: f64) -> f64 {
    CRAD * t * t * t * t / 3.0
}

/// Composition-dependent inputs shared by every EOS and opacity call of one
/// pre-MS build.
struct PreMsCtx<'a> {
    ht: &'a HelmTable,
    eos_cfg: &'a EosConfig,
    kap_cfg: &'a KapConfig,
    controls: &'a StarControls,
    job: &'a StarJobControls,
    chem_id: &'a [usize],
    xa: &'a [f64],
    /// Hydrogen mass fraction `x` from `basic_composition_info`.
    x: f64,
    /// Metal mass fraction `z` from `basic_composition_info`, as `kap_get`
    /// computes it (`kap/public/kap_lib.f90`).
    z: f64,
    abar: f64,
    zbar: f64,
    xc: f64,
    xn: f64,
    xo: f64,
    xne: f64,
}

/// EOS results carried between points (`unpack_eos_results`,
/// `star/private/pre_ms_model.f90:652-663`).
#[derive(Debug, Clone, Copy)]
struct EosUnpacked {
    chi_rho: f64,
    chi_t: f64,
    cp: f64,
    grada: f64,
    lnfree_e: f64,
    d_lnfree_e_dlnrho: f64,
    d_lnfree_e_dlnt: f64,
    eta: f64,
    d_eta_dlnrho: f64,
    d_eta_dlnt: f64,
}

impl EosUnpacked {
    fn from_eos(out: &EosEvalOutput) -> Self {
        Self {
            chi_rho: out.res[I_CHI_RHO],
            chi_t: out.res[I_CHI_T],
            cp: out.res[I_CP],
            grada: out.res[I_GRAD_AD],
            lnfree_e: out.res[I_LN_FREE_E],
            d_lnfree_e_dlnrho: out.d_dlnd[I_LN_FREE_E],
            d_lnfree_e_dlnt: out.d_dln_t[I_LN_FREE_E],
            eta: out.res[I_ETA],
            d_eta_dlnrho: out.d_dlnd[I_ETA],
            d_eta_dlnt: out.d_dln_t[I_ETA],
        }
    }
}

/// `get_eos` with `k = 0` (`star/private/eos_support.f90:44-101`).
fn get_eos(
    ctx: &PreMsCtx<'_>,
    rho: f64,
    t: f64,
) -> Result<EosEvalOutput, StarError> {
    let log_rho = log10(rho);
    if log_rho < -25.0 {
        return Err(StarError::MicrophysicsError(String::from(
            "pre-MS get_eos: eos evaluated at too low a density \
             (`star/private/eos_support.f90:68-74`)",
        )));
    }
    eos_dt_get(ctx.ht, rho, t, ctx.chem_id, ctx.xa, ctx.eos_cfg).map_err(|e| {
        StarError::MicrophysicsError(format!("pre-MS get_eos failed: {e}"))
    })
}

/// `solve_eos_given_PgasT_auto` with `k = 0`
/// (`star/private/eos_support.f90:365-411`): a $\gamma = 5/3$ ideal-gas
/// density guess (`eos_gamma_PT_get`, `eos/public/eos_lib.f90:380-463`)
/// followed by `eosDT_get_Rho` on $\ln P_{\rm gas}$.
///
/// Returns $(\log_{10}\rho, \text{EOS results})$, where the results are
/// those of the last EOS evaluation the solver made, as upstream's `res`.
fn solve_eos_given_pgas_t_auto(
    ctx: &PreMsCtx<'_>,
    log_t: f64,
    log_pgas: f64,
    log_rho_tol: f64,
    log_pgas_tol: f64,
) -> Result<(f64, EosEvalOutput), StarError> {
    let p = exp10(log_pgas);
    let t = exp10(log_t);
    // `eos_gamma_PT_get` fails only for `P <= 0` or `T <= 0`, in which case
    // the guess is `arg_not_provided` (`star/private/eos_support.f90:399-402`).
    let log_rho_guess = if p <= 0.0 || t <= 0.0 {
        ARG_NOT_PROVIDED
    } else {
        // `eos_gamma_PT_get_rho_energy` (`eos/public/eos_lib.f90:387`).
        let rho = (p / t) * ctx.abar / (AVO * KERG);
        log10(rho)
    };
    // `solve_eos_given_PgasT` (`star/private/eos_support.f90:315-358`).
    let inv = eos_dt_get_rho(
        ctx.ht,
        ctx.chem_id,
        ctx.xa,
        log_t,
        I_LN_PGAS as isize,
        log_pgas * LN10,
        log_rho_guess,
        log_rho_tol,
        log_pgas_tol * LN10,
        MAX_ITER_FOR_SOLVE,
        ctx.eos_cfg,
    )
    .map_err(|e| {
        StarError::MicrophysicsError(format!(
            "pre-MS solve_eos_given_PgasT failed (logT = {log_t}, \
             logPgas = {log_pgas}): {e}"
        ))
    })?;
    Ok((inv.log10_result, inv.eos))
}

/// Thermodynamic state of one trial point for [`eval_grad_t`].
#[derive(Debug, Clone, Copy)]
struct GradTPoint {
    rho: f64,
    m: f64,
    r: f64,
    t: f64,
    ln_t: f64,
    l: f64,
    p: f64,
    chi_rho: f64,
    chi_t: f64,
    cp: f64,
    grada: f64,
}

/// `eval_gradT` (`star/private/pre_ms_model.f90:673-736`): opacity, scale
/// height and $\nabla_{\rm rad}$ for a trial point, then `get_gradT`.
/// `e` supplies the current `lnfree_e` and `eta` for the opacity.
fn eval_grad_t(
    ctx: &PreMsCtx<'_>,
    pt: &GradTPoint,
    e: &EosUnpacked,
    s_nz: usize,
) -> Result<f64, StarError> {
    let opacity = if ctx.controls.use_simple_es_for_kap {
        0.2 * (1.0 + ctx.x)
    } else {
        // `kap_get` (`kap/public/kap_lib.f90`): `Get_kap_Results` with the
        // composition moments of `xa`; no `opacity_factor` here.
        get_kap_results(
            ctx.kap_cfg,
            ctx.zbar,
            ctx.x,
            ctx.z,
            ctx.xc,
            ctx.xn,
            ctx.xo,
            ctx.xne,
            log10(pt.rho),
            pt.ln_t / LN10,
            e.lnfree_e,
            e.d_lnfree_e_dlnrho,
            e.d_lnfree_e_dlnt,
            e.eta,
            e.d_eta_dlnrho,
            e.d_eta_dlnt,
        )
        .map_err(|err| {
            StarError::MicrophysicsError(format!(
                "failed in kap_get in eval_gradT: {err}"
            ))
        })?
        .kap
    };

    let cgrav = STANDARD_CGRAV;
    let grav = cgrav * pt.m / pow2(pt.r);
    let mut scale_height = pt.p / (grav * pt.rho);
    if ctx.controls.mix_wind.alt_scale_height_flag {
        let scale_height2 = (pt.p / cgrav).sqrt() / pt.rho;
        if scale_height2 < scale_height {
            scale_height = scale_height2;
        }
    }
    let gradr = pt.p * opacity * pt.l
        / (16.0 * PI * CLIGHT * pt.m * cgrav * CRAD * pow4(pt.t) / 3.0);

    get_grad_t(ctx, pt, opacity, gradr, scale_height, cgrav, s_nz)
}

/// `get_gradT` and `Get_results` at `k = 0`
/// (`star/private/turb_support.f90:58-108, 198-445`), with
/// `gradL_composition_term = 0`. At `k = 0` TDC is never used, there is no
/// turbulent pressure, and the maximum convective velocity is `1d99`.
fn get_grad_t(
    ctx: &PreMsCtx<'_>,
    pt: &GradTPoint,
    opacity: f64,
    gradr: f64,
    scale_height: f64,
    cgrav: f64,
    s_nz: usize,
) -> Result<f64, StarError> {
    let c = ctx.controls;
    let gradl_composition_term = 0.0;
    let mixing_length_alpha = c.mixing_length_alpha;
    let (r, l, t, p, rho, m) = (pt.r, pt.l, pt.t, pt.p, pt.rho, pt.m);
    let grada = pt.grada;
    let ptot = p;
    let pr = CRAD * pow4(t) / 3.0;
    let pg = ptot - pr;
    let beta = pg / ptot;
    let lambda = mixing_length_alpha * scale_height;
    let grav = cgrav * m / pow2(r);
    let gradl = if c.use_ledoux_criterion {
        grada + gradl_composition_term
    } else {
        grada
    };
    let max_conv_vel = 1.0e99;

    let mut convecting = false;
    let mut grad_t = gradr;
    let mut d = 0.0_f64;

    // `turb_support.f90:299-301`.
    if beta < 1.0e-10
        || mixing_length_alpha <= 0.0
        || opacity < 1.0e-10
        || p < 1.0e-20
        || t < 1.0e-10
        || rho < 1.0e-20
        || m < 1.0e-10
        || r < 1.0e-10
        || cgrav < 1.0e-10
    {
        return Ok(grad_t);
    }

    if gradr > gradl {
        // `turb_support.f90:361-394`.
        let mlt_params = MltParams {
            option: c.mlt_option,
            mixing_length_alpha,
            henyey_nu: c.henyey_mlt_nu_param,
            henyey_y: c.henyey_mlt_y_param,
            max_conv_vel,
        };
        let mut mlt_in = MltInputs {
            chi_t: pt.chi_t,
            chi_rho: pt.chi_rho,
            cp: pt.cp,
            grav,
            lambda,
            rho,
            p: ptot,
            t,
            opacity,
            gradr,
            grada,
            gradl,
        };
        let out = set_mlt(&mlt_params, &mlt_in).map_err(|e| {
            StarError::MicrophysicsError(format!(
                "ierr from set_MLT in pre-MS eval_gradT: {e}"
            ))
        })?;
        convecting = out.mixing_type != MixingType::NoMixing;
        grad_t = out.grad_t;
        d = out.d;
        if c.mix_wind.use_superad_reduction {
            let (gamma_factor, gradr_scaled) = eval_superad_reduction(
                &c.mix_wind,
                StarAdOrder1::constant(t),
                StarAdOrder1::constant(p),
                StarAdOrder1::constant(beta),
                StarAdOrder1::constant(grad_t),
                StarAdOrder1::constant(gradr),
                StarAdOrder1::constant(gradl),
            );
            if gamma_factor.val > 1.0 {
                mlt_in.gradr = gradr_scaled.val;
                let out2 = set_mlt(&mlt_params, &mlt_in).map_err(|e| {
                    StarError::MicrophysicsError(format!(
                        "ierr from set_MLT when using superad_reduction in \
                         pre-MS eval_gradT: {e}"
                    ))
                })?;
                convecting = out2.mixing_type != MixingType::NoMixing;
                grad_t = out2.grad_t;
                d = out2.d;
            }
        }
    }

    // `turb_support.f90:398-419`. `gradL_composition_term = 0` here, so the
    // thermohaline branch is never taken.
    if !convecting && gradr > grada {
        let sc_params = SemiconvectionParams {
            option: c.semiconvection_option,
            alpha_semiconvection: c.alpha_semiconvection,
            cgrav,
            gradl_composition_term,
            m,
        };
        let sc_in = SemiconvectionInputs {
            l,
            lambda,
            t,
            p: ptot,
            pr,
            beta,
            opacity,
            rho,
            cp: pt.cp,
            gradr,
            grada,
            gradl,
        };
        let sc = set_semiconvection(&sc_params, &sc_in).map_err(|e| {
            StarError::MicrophysicsError(format!(
                "ierr from set_semiconvection in pre-MS eval_gradT: {e}"
            ))
        })?;
        d = sc.d;
        if sc.mixing_type != MixingType::NoMixing {
            grad_t = sc.grad_t;
        }
    }

    // `turb_support.f90:422-430`.
    if d < c.mix_wind.remove_small_d_limit || is_bad(d) {
        grad_t = gradr;
    }

    // `turb_support.f90:434-443` at `k = 0`: `0 > s% nz - n_in` or
    // `0 < n_out`, with `s% nz` the zone count of the previous `pre_ms_f`
    // evaluation (`0` before the first, `star/private/init.f90:520`).
    let n_in = c.mix_wind.tdc_num_innermost_cells_forced_nonturbulent;
    let n_out = c.mix_wind.tdc_num_outermost_cells_forced_nonturbulent;
    if n_in > s_nz || n_out > 0 {
        grad_t = gradr;
    }

    Ok(grad_t)
}

/// One trial pre-MS model (`build1_pre_ms_model` output): cell arrays
/// ordered from the surface (`k = 0`) to the center (`k = nz - 1`), with
/// `dq` normalized and `q` set by `set_qs`.
#[derive(Debug, Clone)]
struct PreMsBuild {
    nz: usize,
    lnd: Vec<f64>,
    ln_t: Vec<f64>,
    ln_r: Vec<f64>,
    lum: Vec<f64>,
    q: Vec<f64>,
    dq: Vec<f64>,
    /// Mass of the constructed model ($\mathrm{g}$).
    mstar1: f64,
}

/// `build1_pre_ms_model` (`star/private/pre_ms_model.f90:373-670`).
#[allow(clippy::too_many_lines)]
fn build1_pre_ms_model(
    ctx: &PreMsCtx<'_>,
    t_c: f64,
    rho_c: f64,
    d_log10_p_in: f64,
    eps_grav_in: f64,
    mstar_desired: f64,
    s_nz: usize,
) -> Result<PreMsBuild, StarError> {
    const LOGRHO_TOL: f64 = 1.0e-6;
    const LOGPGAS_TOL: f64 = 1.0e-6;
    const DELTA_LOGPGAS: f64 = 0.004;
    const Q_AT_NZ: f64 = 1.0e-5;

    let job = ctx.job;
    let mut log_p_surf_limit = job.pre_ms_log_p_surf_limit;
    if log_p_surf_limit <= 0.0 {
        log_p_surf_limit = 3.5;
    }
    let p_surf_limit = exp10(log_p_surf_limit);

    let mut log_t_surf_limit = job.pre_ms_log_t_surf_limit;
    if log_t_surf_limit <= 0.0 {
        log_t_surf_limit = 3.7;
    }

    let cgrav = STANDARD_CGRAV;
    let mut eps_grav = eps_grav_in;

    let dlog_pgas = if d_log10_p_in == 0.0 {
        DELTA_LOGPGAS
    } else {
        d_log10_p_in.abs()
    };

    // Center (`pre_ms_model.f90:438-447`).
    let res_c = get_eos(ctx, rho_c, t_c)?;
    let mut e = EosUnpacked::from_eos(&res_c);
    let mut log_pgas = res_c.res[I_LN_PGAS] / LN10;
    let mut pgas = exp10(log_pgas);
    let p_c = pgas + radiation_pressure(t_c);

    let mstar = mstar_desired;
    let mut m = Q_AT_NZ * mstar;
    // Pressure at the innermost point, Kippenhahn & Weigert 10.6
    // (`pre_ms_model.f90:452`).
    let mut p = p_c
        - 3.0 * cgrav / (8.0 * PI)
            * pow(PI4 * rho_c / 3.0, 4.0 / 3.0)
            * pow(m, TWO_THIRDS);
    let log_p = log10(p);

    // `nz = 1 + (logP - s% pre_ms_logP_surf_limit)/dlogPgas`: the real
    // result is truncated toward zero on assignment to the integer
    // (`pre_ms_model.f90:456`).
    #[allow(clippy::cast_possible_truncation)]
    let nz_est =
        (1.0 + (log_p - job.pre_ms_log_p_surf_limit) / dlog_pgas) as i64;
    let nz0 = usize::try_from(nz_est).ok().filter(|&n| n >= 1).ok_or_else(
        || {
            // Upstream then writes `q(nz)` out of bounds; there is no
            // defined behaviour to reproduce (DECISIONS #43).
            StarError::InvalidInput(format!(
                "build1_pre_ms_model: estimated nz = {nz_est} < 1 \
                 (`star/private/pre_ms_model.f90:456,499-500`)"
            ))
        },
    )?;

    // Temperature at nz, K&W 10.9 assuming a convective core
    // (`pre_ms_model.f90:459-461`).
    let mut ln_t = log(t_c)
        - pow(PI / 6.0, ONE_THIRD)
            * cgrav
            * e.grada
            * pow(rho_c * rho_c * m, TWO_THIRDS)
            / p_c;
    let mut t = exp(ln_t);

    // Density at nz, at the central gas pressure
    // (`pre_ms_model.f90:464-471`).
    let (mut log_rho, res) = solve_eos_given_pgas_t_auto(
        ctx,
        ln_t / LN10,
        log10(pgas),
        LOGRHO_TOL,
        LOGPGAS_TOL,
    )?;
    let mut rho = exp10(log_rho);
    e = EosUnpacked::from_eos(&res);

    let mut r = pow(m / (PI4 * rho / 3.0), ONE_THIRD);

    // Raise eps_grav until the innermost point is convective
    // (`pre_ms_model.f90:477-493`).
    let mut l;
    loop {
        l = eps_grav * m;
        let pt = GradTPoint {
            rho,
            m,
            r,
            t,
            ln_t,
            l,
            p,
            chi_rho: e.chi_rho,
            chi_t: e.chi_t,
            cp: e.cp,
            grada: e.grada,
        };
        let grad_t = eval_grad_t(ctx, &pt, &e, s_nz)?;
        if grad_t >= e.grada {
            break;
        }
        eps_grav *= 1.1;
    }

    let mut lnd_a = vec![0.0_f64; nz0];
    let mut ln_t_a = vec![0.0_f64; nz0];
    let mut ln_r_a = vec![0.0_f64; nz0];
    let mut lum_a = vec![0.0_f64; nz0];
    let mut q = vec![0.0_f64; nz0];
    let mut dq = vec![0.0_f64; nz0];

    // `store_*_in_xh` (`star/private/star_utils.f90:456-557`).
    let kz = nz0 - 1;
    lnd_a[kz] = log_rho * LN10;
    ln_t_a[kz] = ln_t;
    ln_r_a[kz] = log(r);
    lum_a[kz] = l;
    q[kz] = Q_AT_NZ;
    dq[kz] = Q_AT_NZ;

    // `step_loop` (`pre_ms_model.f90:513-630`); upstream index `k` is
    // `k - 1` here. Every path through `retry_loop` leaves it at `j = 1`.
    let mut prune = 0_usize;
    for k in (1..nz0).rev() {
        let try_dlog_pgas = dlog_pgas;
        let log_pgas0 = log_pgas;
        let p0 = p;
        let m0 = m;
        let l0 = l;
        let r0 = r;
        let t0 = t;
        let rho0 = rho;
        let chi_rho0 = e.chi_rho;
        let chi_t0 = e.chi_t;
        let cp0 = e.cp;
        let grada0 = e.grada;
        let mut dm = 0.0_f64;

        log_pgas = log_pgas0 - try_dlog_pgas;
        pgas = exp10(log_pgas);

        for i in 1..=2 {
            let prad = radiation_pressure(t);
            p = pgas + prad;

            let rho_mid = (rho + rho0) / 2.0;

            // Repeat to get hydrostatic balance (`pre_ms_model.f90:546-554`).
            let mut rmid = 0.0_f64;
            let mut mmid = 0.0_f64;
            for ii in 1..=10 {
                rmid = pow((r * r * r + r0 * r0 * r0) / 2.0, ONE_THIRD);
                mmid = (m + m0) / 2.0;
                if ii == 10 {
                    break;
                }
                dm = -PI4 * pow4(rmid) * (p - p0) / (cgrav * mmid);
                m = m0 + dm;
                r = pow(
                    r0 * r0 * r0 + dm / (FOUR_THIRDS_PI * rho_mid),
                    ONE_THIRD,
                );
            }

            l = l0 + dm * eps_grav;
            let lmid = (l0 + l) / 2.0;

            let pmid = (p + p0) / 2.0;

            let chi_rho_mid = (chi_rho0 + e.chi_rho) / 2.0;
            let chi_t_mid = (chi_t0 + e.chi_t) / 2.0;
            let cp_mid = (cp0 + e.cp) / 2.0;
            let grada_mid = (grada0 + e.grada) / 2.0;

            for _ in 1..=2 {
                let tmid = (t + t0) / 2.0;
                let pt = GradTPoint {
                    rho: rho_mid,
                    m: mmid,
                    r: rmid,
                    t: tmid,
                    ln_t: log(tmid),
                    l: lmid,
                    p: pmid,
                    chi_rho: chi_rho_mid,
                    chi_t: chi_t_mid,
                    cp: cp_mid,
                    grada: grada_mid,
                };
                let grad_t = eval_grad_t(ctx, &pt, &e, s_nz)?;
                t = t0 + tmid * grad_t * (p - p0) / pmid;
                ln_t = log(t);
            }

            if i == 2 {
                break;
            }

            let (lr, res_k) = solve_eos_given_pgas_t_auto(
                ctx,
                ln_t / LN10,
                log_pgas,
                LOGRHO_TOL,
                LOGPGAS_TOL,
            )?;
            log_rho = lr;
            rho = exp10(log_rho);
            e = EosUnpacked::from_eos(&res_k);
        }

        if ln_t <= log_t_surf_limit * LN10 {
            prune = k;
            break;
        }
        if p <= p_surf_limit {
            prune = k;
            break;
        }

        let kk = k - 1;
        lnd_a[kk] = log_rho * LN10;
        ln_t_a[kk] = ln_t;
        ln_r_a[kk] = log(r);
        lum_a[kk] = l;
        q[kk] = m / mstar;
        dq[kk] = dm / mstar;
    }

    let mut nz = nz0;
    if prune > 0 {
        // Move stuff and reduce nz (`pre_ms_model.f90:632-641`).
        for k in 0..nz - prune {
            lnd_a[k] = lnd_a[k + prune];
            ln_t_a[k] = ln_t_a[k + prune];
            ln_r_a[k] = ln_r_a[k + prune];
            lum_a[k] = lum_a[k + prune];
            q[k] = q[k + prune];
            dq[k] = dq[k + prune];
        }
        m = mstar * q[0];
        nz -= prune;
    }
    for v in [
        &mut lnd_a,
        &mut ln_t_a,
        &mut ln_r_a,
        &mut lum_a,
        &mut q,
        &mut dq,
    ] {
        v.truncate(nz);
    }

    let mstar1 = m;

    // `do_normalize_dqs_as_part_of_set_qs = .false.` (not parsed; the
    // `controls.defaults:8993` default), so `normalize_dqs` then `set_qs`
    // (`pre_ms_model.f90:645-656`).
    normalize_dqs_slice(&mut dq).map_err(|err| {
        StarError::InvalidInput(format!(
            "normalize_dqs failed in pre ms model: {err}"
        ))
    })?;
    set_qs_slice(&mut q, &dq);

    Ok(PreMsBuild {
        nz,
        lnd: lnd_a,
        ln_t: ln_t_a,
        ln_r: ln_r_a,
        lum: lum_a,
        q,
        dq,
        mstar1,
    })
}

/// Constructs the pre-main-sequence [`StarModel`] for `controls` and `job`
/// (`build_pre_ms_model`, `star/private/pre_ms_model.f90:34-288`, called
/// from `model_builder`, `star/private/init.f90:966-1007`), with default
/// EOS and opacity settings and the network named by `job.net_name`.
///
/// # Errors
/// Returns [`StarError`] if the inputs are out of range, the network cannot
/// be loaded, or the EOS, opacity or root finding fails.
pub fn build_pre_ms_model(
    controls: StarControls,
    job: StarJobControls,
) -> Result<StarModel, StarError> {
    let kap_cfg = KapConfig {
        zbase: controls.zbase,
        ..KapConfig::default()
    };
    build_pre_ms_model_with_configs(
        controls,
        job,
        EosConfig::default(),
        kap_cfg,
        None,
    )
}

/// Constructs the pre-main-sequence [`StarModel`] with explicit
/// [`EosConfig`] and [`KapConfig`] (`build_pre_ms_model`,
/// `star/private/pre_ms_model.f90:34-288`, and its caller `model_builder`,
/// `star/private/init.f90:894, 944, 966-1007`).
///
/// The network is `job.net_name`, which already holds `new_net_name` when
/// `change_net` or `change_initial_net` is set (`pre_ms_change_net`,
/// `star/job/run_star_support.f90:1661-1667`); `search_dir` is passed to the
/// network loader.
///
/// # Errors
/// Returns [`StarError`] if `initial_mass < 0.03`, `pre_ms_T_c >= 1d6`, the
/// network cannot be loaded, the composition is invalid, or the EOS,
/// opacity, MLT or root finding fails.
#[allow(clippy::too_many_lines)]
pub fn build_pre_ms_model_with_configs(
    controls: StarControls,
    job: StarJobControls,
    eos_cfg: EosConfig,
    kap_cfg: KapConfig,
    search_dir: Option<&Path>,
) -> Result<StarModel, StarError> {
    // `init.f90:968-990`.
    if controls.initial_mass < MIN_MASS_FOR_CREATE_PRE_MS {
        return Err(StarError::InvalidInput(format!(
            "sorry: cannot create pre-ms smaller than \
             {MIN_MASS_FOR_CREATE_PRE_MS} (initial_mass = {}); create it and \
             relax to lower mass (`star/private/init.f90:968-990`)",
            controls.initial_mass
        )));
    }
    let ht = crate::micro::builtin_helm_table()?;

    // `set_net(s, s% net_name)` or `do_micro_change_net`
    // (`init.f90:991-995`).
    let chem_id: Vec<usize> =
        crate::micro::get_or_load_net(&job.net_name, search_dir)?
            .chem_id
            .clone();

    // `init.f90:894`.
    let initial_z = controls.initial_z;
    let mut initial_y = controls.initial_y;
    if initial_y < 0.0 {
        initial_y = (0.24 + 2.0 * initial_z).clamp(0.0, 1.0);
    }

    // `s% mstar = s% initial_mass*Msun` (`init.f90:944`), then clamped to
    // `[min_mass_to_create, max_mass_to_create]`
    // (`pre_ms_model.f90:66-69`).
    let mstar_in = controls.initial_mass * M_SUN;
    let mstar = (90.0 * M_SUN).min((0.03 * M_SUN).max(mstar_in));

    // `pre_ms_model.f90:71-79`.
    let mut t_c = job.pre_ms_t_c;
    if t_c <= 0.0 {
        t_c = 9.0e5;
    }
    if t_c >= 1.0e6 {
        return Err(StarError::InvalidInput(format!(
            "center temperature is too high for a pre main sequence model \
             (log T_c = {}); pick T_c below 10^6 \
             (`star/private/pre_ms_model.f90:73-79`)",
            log10(t_c)
        )));
    }

    // Composition (`pre_ms_model.f90:89-130`).
    let initial_h1 = (1.0 - (initial_z + initial_y)).clamp(0.0, 1.0);
    let initial_h2 = 0.0;
    let (initial_he3, initial_he4) = if CONTROLS_INITIAL_HE3 < 0.0 {
        let xsol_he3 = chem_xsol("he3");
        let xsol_he4 = chem_xsol("he4");
        (
            initial_y * xsol_he3 / (xsol_he3 + xsol_he4),
            initial_y * xsol_he4 / (xsol_he3 + xsol_he4),
        )
    } else if CONTROLS_INITIAL_HE3 < initial_y {
        (CONTROLS_INITIAL_HE3, initial_y - CONTROLS_INITIAL_HE3)
    } else {
        // Upstream sets `ierr = -1` but continues, and
        // `get_xa_for_standard_metals` resets `ierr` and reads the unset
        // `initial_he3`/`initial_he4`: no defined behaviour (DECISIONS #43).
        return Err(StarError::InvalidInput(format!(
            "ERROR: initial_he3 ({CONTROLS_INITIAL_HE3}) is larger than \
             initial_y ({initial_y}) (`star/private/pre_ms_model.f90:101-104`)"
        )));
    };
    let xa = standard_metals_xa(
        &chem_id,
        &controls.mix_wind,
        initial_h1,
        initial_h2,
        initial_he3,
        initial_he4,
        job.initial_zfracs,
        DUMP_MISSING_METALS_INTO_HEAVIEST,
    )?;
    let sum_xa: f64 = xa.iter().sum();
    if (1.0 - sum_xa).abs() > 1.0e-8 {
        return Err(StarError::InvalidInput(format!(
            "build_pre_ms_model: sum(xa) = {sum_xa} \
             (`star/private/pre_ms_model.f90:122-136`)"
        )));
    }

    let catalog = NuclideCatalog::builtin_ref();
    let comp = get_composition_info(catalog, &chem_id, &xa, true)
        .map_err(|e| StarError::InvalidInput(e.to_string()))?;
    let x = comp.xh;
    // `kap_get` sums of C, N, O and Ne isotopes (`kap/public/kap_lib.f90`).
    let (mut xc, mut xn, mut xo, mut xne) = (0.0, 0.0, 0.0, 0.0);
    for (&cid, &xj) in chem_id.iter().zip(&xa) {
        let iz = catalog
            .get_by_id(cid)
            .ok_or_else(|| {
                StarError::InvalidInput(format!("unknown chem_id {cid}"))
            })?
            .z;
        match iz {
            6 => xc += xj,
            7 => xn += xj,
            8 => xo += xj,
            10 => xne += xj,
            _ => {}
        }
    }

    // Estimate mu_eff assuming complete ionization and Z << 1
    // (`pre_ms_model.f90:142`).
    let mu_eff = 4.0 / (3.0 + 5.0 * x);

    // `pre_ms_model.f90:145-152`.
    let rho_c = if job.pre_ms_guess_rho_c <= 0.0 {
        // n = 3/2 polytrope, Ushomirsky et al. (1998) Eqs. 5-6; the
        // polytrope value is usually too small.
        let rstar = R_SUN * 7.41e6 * (mu_eff / 0.6) * (mstar / M_SUN) / t_c;
        let rho_c = 8.44 * (mstar / M_SUN) * pow3(R_SUN / rstar);
        1.2 * rho_c
    } else {
        job.pre_ms_guess_rho_c
    };

    // Pick a luminosity above the ZAMS level (`pre_ms_model.f90:155-164`).
    let lg_m = log10(mstar / M_SUN);
    let lg_l = if lg_m > 1.0 {
        4.5 + 2.5 * (lg_m - 1.0)
    } else if lg_m > 0.5 {
        3.0 + 3.0 * (lg_m - 0.5)
    } else if lg_m > 0.0 {
        4.0 * lg_m + 0.5
    } else {
        0.5
    };
    // Uniform eps_grav giving that luminosity (`pre_ms_model.f90:167`).
    let eps_grav = exp10(lg_l) * L_SUN / mstar;

    let d_log10_p = job.pre_ms_d_log10_p;

    let ctx = PreMsCtx {
        ht,
        eos_cfg: &eos_cfg,
        kap_cfg: &kap_cfg,
        controls: &controls,
        job: &job,
        chem_id: &chem_id,
        xa: &xa,
        x,
        z: comp.xz,
        abar: comp.abar,
        zbar: comp.zbar,
        xc,
        xn,
        xo,
        xne,
    };

    // `pre_ms_f` (`pre_ms_model.f90:291-370`). The model of each call
    // replaces the previous one, as `s% xh`, `s% q` and `s% dq` do upstream.
    let mut last: Option<PreMsBuild> = None;
    let mut failure: Option<StarError> = None;
    let mut s_nz = 0_usize;
    let mut pre_ms_f = |lnd: f64| -> Result<(f64, f64), NumError> {
        let rho_c = exp(lnd);
        match build1_pre_ms_model(
            &ctx, t_c, rho_c, d_log10_p, eps_grav, mstar, s_nz,
        ) {
            Ok(b) => {
                s_nz = b.nz;
                let f = (mstar - b.mstar1) / mstar;
                last = Some(b);
                Ok((f, 0.0))
            }
            Err(e) => {
                failure = Some(e);
                Err(NumError::ConvergenceFailed(
                    "failed in build1_pre_ms_model",
                ))
            }
        }
    };

    // `pre_ms_model.f90:211-238`.
    let lnd = log(rho_c);
    let dlnd = 0.01;
    let epsx = 1.0e-3; // limit for variation in lnd
    let epsy = 1.0e-3; // limit for matching desired mass as fraction of total
    let root =
        look_for_brackets(lnd, dlnd, -f64::MAX, f64::MAX, &mut pre_ms_f, IMAX)
            .and_then(|((lnd1, lnd3), (y1, y3))| {
                safe_root_with_brackets(
                    &mut pre_ms_f,
                    lnd1,
                    lnd3,
                    y1,
                    y3,
                    IMAX,
                    epsx,
                    epsy,
                )
            });
    if let Err(e) = root {
        return Err(failure.take().unwrap_or_else(|| {
            StarError::ConvergenceFailure(format!(
                "pre-MS root finding failed: {e}"
            ))
        }));
    }
    let build = last.take().ok_or_else(|| {
        StarError::ConvergenceFailure(String::from(
            "pre-MS root finding built no model",
        ))
    })?;

    let nz = build.nz;
    let mut model = StarModel::new(nz, chem_id, job, controls);
    model.eos_config = eos_cfg;
    model.kap_config = kap_cfg;
    model.controls.initial_y = initial_y;
    model.mstar = mstar;
    model.xmstar = mstar;
    model.mstar_start = mstar;
    model.mstar_old = mstar;
    // `pre_ms_model.f90:85-88`.
    model.m_center = 0.0;
    model.l_center = 0.0;
    model.r_center = 0.0;
    model.v_center = 0.0;
    // A newly created model is model 0 (`set_starting_star_data`,
    // `star/private/init.f90:676`); `build_pre_ms_model` does not change it
    // (`star/private/pre_ms_model.f90`), so the first step computes model 1.
    model.model_number = 0;
    model.star_age = 0.0;
    // `create_star` sets `dt_next = 1d-5*secyer` after `build_pre_ms_model`
    // (`star/private/init.f90:1007`); the first step takes it as `dt`.
    model.dt = 1.0e-5 * SECYER;
    model.dt_next = model.dt;
    model.dt_old = model.dt;
    // `model_builder` (`star/private/init.f90:1042-1043`).
    if model.controls.max_years_for_timestep > 0.0 {
        model.dt_next =
            model.dt_next.min(SECYER * model.controls.max_years_for_timestep);
    }

    // Copy the last model (`pre_ms_model.f90:272-281`).
    for k in 0..nz {
        model.lnd[k] = build.lnd[k];
        model.ln_t[k] = build.ln_t[k];
        model.ln_r[k] = build.ln_r[k];
        model.lum[k] = build.lum[k];
        model.q[k] = build.q[k];
        model.dq[k] = build.dq[k];
        model.cell_xa_mut(k).copy_from_slice(&xa);
    }

    model.finish_init_model()?;

    Ok(model)
}
