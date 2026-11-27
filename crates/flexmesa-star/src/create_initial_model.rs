// SPDX-License-Identifier: GPL-3.0-or-later
//! Isentropic initial model for a given mass and radius
//! (`star/private/create_initial_model.f90` at MESA `26.4.1`).
//!
//! Given a central pressure $P_c$ and specific entropy $S$, `PSerrfunc`
//! integrates the adiabat outward in pressure with a fixed-step fourth-order
//! Runge-Kutta scheme until the optical depth estimate
//! $\tau = \kappa P / g$ falls below $2/3$, then interpolates the last point to
//! $\tau = 2/3$. `build_initial_model` adjusts $(P_c, S)$ with a two-variable
//! secant Newton iteration until the integrated mass and radius match the
//! requested values. Radiation pressure is ignored ($P = P_{\mathrm{gas}}$), and
//! the luminosity is set so that $L(m) \propto \int_0^m T\,dm$ with the
//! surface value $4\pi R^2 \sigma T_{\mathrm{eff}}^4$.
//!
//! Arithmetic follows upstream's order of operations term by term; every
//! transcendental call goes through `flexmesa_math`.

use flexmesa_const::{
    BOLTZM, BOLTZ_SIGMA, FOUR_THIRDS_PI, MP, PI, PI4, SECYER, STANDARD_CGRAV,
    TWO_THIRDS,
};
use flexmesa_eos::{eos_pt_get_t, EosConfig, HelmTable, I_LN_S};
use flexmesa_kap::{kap_get, KapConfig};
use flexmesa_math::{exp10, log, log10, pow2, pow4};

use crate::accretion_xa::standard_metals_xa;
use crate::controls::{StarControls, StarError, StarJobControls, VarLayout};
use crate::micro::get_or_load_net;
use crate::model::StarModel;

/// Upstream `nzmax` (`create_initial_model.f90:41`): the fixed length of the
/// integration grids.
const NZMAX: usize = 100_000;

/// Upstream `default_net_name` control default
/// (`star/defaults/controls.defaults:6759`), which `model_builder` installs
/// as `s% net_name` before building (`star/private/init.f90:933`). The control
/// itself is not parsed, so an inlist that sets it is rejected as unsupported.
const DEFAULT_NET_NAME: &str = "basic.net";

/// Composition and microphysics shared by every integration
/// (the module variables of `create_initial_model.f90:32-39`).
struct BuildCtx<'a> {
    ht: &'a HelmTable,
    eos_cfg: &'a EosConfig,
    kap_cfg: &'a KapConfig,
    chem_id: &'a [usize],
    xa: &'a [f64],
    /// Runge-Kutta accuracy `eps` (`initial_model_eps`).
    eps: f64,
    /// Radius that sets the first off-center point (`R_try`).
    r_try: f64,
}

/// Integration grids of one trial model (`create_star_info`,
/// `create_initial_model.f90:43-47`). Index `0` is the center.
#[derive(Default)]
struct CreateStarInfo {
    rg: Vec<f64>,
    mg: Vec<f64>,
    pg: Vec<f64>,
    tg: Vec<f64>,
    taug: Vec<f64>,
    lg: Vec<f64>,
    rhog: Vec<f64>,
    intdm_tg: Vec<f64>,
    mass: f64,
    radius: f64,
}

/// Returns `(T, rho)` on the adiabat `S` at gas pressure `P`
/// (`get_TRho_from_PS`, `create_initial_model.f90:452-497`).
fn get_t_rho_from_ps(
    ctx: &BuildCtx<'_>,
    p: f64,
    s: f64,
) -> Result<(f64, f64), StarError> {
    // `logT_tol = other_tol = 1d-6`, `max_iter = 100`, `logT_guess = 4`,
    // no bounds (`create_initial_model.f90:466-468`).
    let inv = eos_pt_get_t(
        ctx.ht,
        ctx.chem_id,
        ctx.xa,
        log10(p),
        I_LN_S,
        log(s),
        4.0,
        1.0e-6,
        1.0e-6,
        100,
        ctx.eos_cfg,
    )
    .map_err(|e| {
        // Upstream prints "failure in eosPT_get_T" and stops
        // (`create_initial_model.f90:484-487`).
        StarError::ConvergenceFailure(format!(
            "create_initial_model: failure in eosPT_get_T (P = {p:e}, S = {s:e}): {e}"
        ))
    })?;
    let mut t = exp10(inv.log10_result);
    // `Tmin = 0` sets the surface isotherm (`create_initial_model.f90:493`).
    const T_MIN: f64 = 0.0;
    if t < T_MIN {
        t = T_MIN;
    }
    Ok((t, inv.pt.rho))
}

/// Opacity with `lnfree_e` and `eta` set to zero
/// (`get_kap_from_rhoT`, `create_initial_model.f90:406-447`).
fn get_kap_from_rho_t(
    ctx: &BuildCtx<'_>,
    log_rho: f64,
    log_t: f64,
) -> Result<f64, StarError> {
    kap_get(
        ctx.kap_cfg,
        ctx.chem_id,
        ctx.xa,
        log_rho,
        log_t,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    .map(|out| out.kap)
    .map_err(|e| {
        // Upstream prints "failed in kap_get get_kap_from_rhoT" and returns
        // with `kap` unset, which the caller then reads
        // (`create_initial_model.f90:435-438`). There is no defined value to
        // reproduce, so this path is an error (DECISIONS #43, deferred #399).
        StarError::MicrophysicsError(format!(
            "create_initial_model: failed in kap_get get_kap_from_rhoT \
             (logRho = {log_rho}, logT = {log_t}): {e}"
        ))
    })
}

/// Right-hand side $dy/dP$ for $y = (r, m, \int_0^m T\,dm)$
/// (`derivs`, `create_initial_model.f90:384-400`).
fn derivs(
    ctx: &BuildCtx<'_>,
    p: f64,
    s: f64,
    y: &[f64; 3],
) -> Result<[f64; 3], StarError> {
    let g = STANDARD_CGRAV;
    let r = y[0];
    let m = y[1];
    let (t, rho) = get_t_rho_from_ps(ctx, p, s)?;
    let d0 = -(r * r / (g * m * rho));
    let d1 = -(PI4 * r * r * r * r / (g * m));
    Ok([d0, d1, d1 * t])
}

fn axpy_scaled(y: &[f64; 3], dy: &[f64; 3], div: f64) -> [f64; 3] {
    [y[0] + dy[0] / div, y[1] + dy[1] / div, y[2] + dy[2] / div]
}

fn scale3(v: &[f64; 3], f: f64) -> [f64; 3] {
    [v[0] * f, v[1] * f, v[2] * f]
}

/// Integrates the adiabat for `(Pc, S)` into `cs` and returns the relative
/// mass and radius errors (`PSerrfunc`, `create_initial_model.f90:252-376`).
fn ps_errfunc(
    ctx: &BuildCtx<'_>,
    cs: &mut CreateStarInfo,
    pc: f64,
    s: f64,
    m_target: f64,
    r_target: f64,
) -> Result<[f64; 2], StarError> {
    let g = STANDARD_CGRAV;
    let (tc, rhoc) = get_t_rho_from_ps(ctx, pc, s)?;

    *cs = CreateStarInfo::default();

    // Central point (`create_initial_model.f90:291-299`).
    cs.rg.push(0.0);
    cs.mg.push(0.0);
    cs.pg.push(pc);
    cs.tg.push(tc);
    cs.rhog.push(rhoc);
    cs.intdm_tg.push(0.0);
    cs.taug.push(1.0e20);
    cs.lg.push(0.0);

    // First point off center (`create_initial_model.f90:302-319`).
    let r1 = 1.0e-2 * ctx.r_try * 0.1_f64.min(ctx.eps);
    let m1 = FOUR_THIRDS_PI * rhoc * r1 * r1 * r1;
    let mut p = pc - TWO_THIRDS * PI * g * rhoc * rhoc * r1 * r1;
    let intdm_t1 = m1 * tc;
    let mut y = [r1, m1, intdm_t1];
    let (t, rho) = get_t_rho_from_ps(ctx, p, s)?;
    cs.rg.push(r1);
    cs.mg.push(m1);
    cs.pg.push(p);
    cs.tg.push(t);
    cs.rhog.push(rho);
    cs.intdm_tg.push(intdm_t1);
    cs.taug.push(1.0e20);
    cs.lg.push(0.0);

    // Fourth-order Runge-Kutta steps in P (`create_initial_model.f90:321-363`).
    loop {
        let dydp = derivs(ctx, p, s, &y)?;
        let d_p = -(ctx.eps
            * p.min((y[0] / dydp[0]).abs()).min((y[1] / dydp[1]).abs()));

        let dy1 = scale3(&dydp, d_p);
        let dydp = derivs(ctx, p + d_p / 2.0, s, &axpy_scaled(&y, &dy1, 2.0))?;
        let dy2 = scale3(&dydp, d_p);
        let dydp = derivs(ctx, p + d_p / 2.0, s, &axpy_scaled(&y, &dy2, 2.0))?;
        let dy3 = scale3(&dydp, d_p);
        let y3 = [y[0] + dy3[0], y[1] + dy3[1], y[2] + dy3[2]];
        let dydp = derivs(ctx, p + d_p, s, &y3)?;
        let dy4 = scale3(&dydp, d_p);
        for j in 0..3 {
            y[j] = y[j]
                + dy1[j] / 6.0
                + dy2[j] / 3.0
                + dy3[j] / 3.0
                + dy4[j] / 6.0;
        }
        p += d_p;

        let (t, rho) = get_t_rho_from_ps(ctx, p, s)?;
        let kap = get_kap_from_rho_t(ctx, log10(rho), log10(t))?;
        let grav = g * y[1] / (y[0] * y[0]);
        let tau = kap * p / grav;

        if cs.rg.len() >= NZMAX {
            // Upstream writes past the end of its `nzmax` grids here; there
            // is no defined behaviour to reproduce (DECISIONS #43, deferred #400).
            return Err(StarError::ConvergenceFailure(format!(
                "create_initial_model: PSerrfunc integration exceeded nzmax = {NZMAX} points"
            )));
        }
        cs.rg.push(y[0]);
        cs.mg.push(y[1]);
        cs.pg.push(p);
        cs.tg.push(t);
        cs.rhog.push(rho);
        cs.taug.push(tau);
        cs.intdm_tg.push(y[2]);
        cs.lg.push(0.0);

        if t < 150.0 {
            // Upstream prints "temp too low in integration" and stops
            // (`create_initial_model.f90:352-355`).
            return Err(StarError::ConvergenceFailure(String::from(
                "create_initial_model: temp too low in integration (T < 150 K)",
            )));
        }
        if tau < 2.0 / 3.0 {
            break;
        }
    }

    // Interpolate the last point to tau = 2/3 (`create_initial_model.f90:365-375`).
    let nz = cs.rg.len();
    let i = nz - 1;
    let im = i - 1;
    let p_i = cs.pg[im]
        + (cs.pg[i] - cs.pg[im]) * (2.0 / 3.0 - cs.taug[im])
            / (cs.taug[i] - cs.taug[im]);
    let lerp = |v: &[f64]| {
        v[im] + (v[i] - v[im]) * (p_i - cs.pg[im]) / (cs.pg[i] - cs.pg[im])
    };
    let rg_i = lerp(&cs.rg);
    let mg_i = lerp(&cs.mg);
    let tg_i = lerp(&cs.tg);
    let rhog_i = lerp(&cs.rhog);
    let intdm_tg_i = lerp(&cs.intdm_tg);
    cs.rg[i] = rg_i;
    cs.mg[i] = mg_i;
    cs.tg[i] = tg_i;
    cs.rhog[i] = rhog_i;
    cs.intdm_tg[i] = intdm_tg_i;
    cs.pg[i] = p_i;
    cs.taug[i] = 2.0 / 3.0;

    // Surface luminosity and L(m) (`create_initial_model.f90:377-383`).
    cs.mass = cs.mg[i];
    cs.radius = cs.rg[i];
    let teff = cs.tg[i];
    let luminosity = PI4 * pow2(cs.radius) * BOLTZ_SIGMA * pow4(teff);
    let intdm_t_nz = cs.intdm_tg[i];
    for k in 0..nz {
        cs.lg[k] = luminosity * cs.intdm_tg[k] / intdm_t_nz;
    }

    Ok([
        (cs.mass - m_target) / m_target,
        (cs.radius - r_target) / r_target,
    ])
}

/// Builds the isentropic initial model for `mass_in_gm_for_create_initial_model`
/// and `radius_in_cm_for_create_initial_model` (`build_initial_model`,
/// `star/private/create_initial_model.f90:70-247`), together with the
/// `do_create_initial_model` branch of `model_builder`
/// (`star/private/init.f90:929-966`) and the surrounding `create_star`
/// settings (`init.f90:893-895`, `1026-1027`).
///
/// The returned model has `nz = cs% nz - 1` cells (the center point is
/// skipped), uniform composition from `get_xa_for_standard_metals`, and its
/// mesh set by [`StarModel::finish_init_model`] (upstream `set_qs` in
/// `finish_load_model`, without `normalize_dqs`).
///
/// # Errors
/// Returns [`StarError`] where upstream sets `ierr /= 0` or stops: invalid
/// mass or radius, more than four hydro variables, composition failure, EOS
/// inversion failure, `T < 150 K` during integration, opacity failure, or no
/// convergence within `max_tries_for_create_initial_model`.
pub fn build_initial_model(
    controls: StarControls,
    job: StarJobControls,
    eos_cfg: EosConfig,
    kap_cfg: KapConfig,
) -> Result<StarModel, StarError> {
    build_initial_model_with_configs(controls, job, eos_cfg, kap_cfg, None)
}

/// Builds the isentropic initial model with an optional `search_dir` for
/// custom nuclear reaction network files (`build_initial_model`,
/// `star/private/create_initial_model.f90:70-247`, and `model_builder`,
/// `star/private/init.f90:929-966`).
///
/// When `job.change_net` is enabled and `job.new_net_name` is non-empty, loads
/// that network (`do_micro_change_net` in `init.f90:950-955` and
/// `initial_model_change_net` in `run_star_support.f90:1695`); otherwise loads
/// `job.net_name` (or `basic.net` if empty). Sets `model.job.net_name` to the
/// loaded network name.
///
/// # Errors
/// Returns [`StarError`] where upstream sets `ierr /= 0` or stops: invalid
/// mass or radius, more than four hydro variables, network or composition
/// failure, EOS inversion failure, `T < 150 K` during integration, opacity
/// failure, or no convergence within `max_tries_for_create_initial_model`.
pub fn build_initial_model_with_configs(
    mut controls: StarControls,
    mut job: StarJobControls,
    eos_cfg: EosConfig,
    kap_cfg: KapConfig,
    search_dir: Option<&std::path::Path>,
) -> Result<StarModel, StarError> {
    // `create_star` (`init.f90:893`).
    if controls.initial_y < 0.0 {
        controls.initial_y = (0.24 + 2.0 * controls.initial_z).clamp(0.0, 1.0);
    }
    // `model_builder` resets these before building (`init.f90:935-939`).
    controls.v_flag = false;
    controls.u_flag = false;
    controls.rotation_flag = false;

    // `model_builder` calls `do_micro_change_net` instead of `set_net` when
    // `initial_model_change_net` is set (`init.f90:950-955`,
    // `run_star_support.f90:1695`).
    let net_name = if job.change_net && !job.new_net_name.trim().is_empty() {
        job.new_net_name.trim()
    } else if !job.net_name.trim().is_empty() {
        job.net_name.trim()
    } else {
        DEFAULT_NET_NAME
    };
    let net = get_or_load_net(net_name, search_dir)?;
    job.net_name = net_name.to_string();
    let chem_id = net.chem_id.clone();

    let layout = VarLayout::with_flags(
        chem_id.len(),
        controls.v_flag,
        controls.u_flag,
        controls.rsp2_flag,
        controls.rti_flag,
    );
    if layout.nvar_hydro > 4 {
        return Err(StarError::InvalidInput(String::from(
            "sorry, current build_initial_model only supports the basic 4 vars.",
        )));
    }

    let m_target = job.mass_in_gm_for_create_initial_model;
    let r_target = job.radius_in_cm_for_create_initial_model;
    if m_target <= 0.0 {
        return Err(StarError::InvalidInput(format!(
            "must set mass_in_gm_for_create_initial_model ({m_target})"
        )));
    }
    if r_target <= 0.0 {
        return Err(StarError::InvalidInput(format!(
            "must set radius_in_cm_for_create_initial_model ({r_target})"
        )));
    }

    // Composition (`create_initial_model.f90:120-146`).
    let initial_z = controls.initial_z;
    let initial_y = controls.initial_y;
    let initial_h1 = 0.0_f64.max(1.0_f64.min(1.0 - (initial_z + initial_y)));
    let initial_h2 = 0.0;
    let xsol_he3 = flexmesa_chem::chem_xsol("he3");
    let xsol_he4 = flexmesa_chem::chem_xsol("he4");
    let initial_he3 = initial_y * xsol_he3 / (xsol_he3 + xsol_he4);
    let initial_he4 = initial_y * xsol_he4 / (xsol_he3 + xsol_he4);

    let xa = standard_metals_xa(
        &chem_id,
        &controls.mix_wind,
        initial_h1,
        initial_h2,
        initial_he3,
        initial_he4,
        job.initial_zfracs,
        job.dump_missing_metals_into_heaviest,
    )
    .map_err(|e| {
        StarError::InvalidInput(format!(
            "create initial model failed in get_xa_for_standard_metals: {e}"
        ))
    })?;

    let ht = crate::micro::builtin_helm_table()?;
    let ctx = BuildCtx {
        ht,
        eos_cfg: &eos_cfg,
        kap_cfg: &kap_cfg,
        chem_id: &chem_id,
        xa: &xa,
        eps: job.initial_model_eps,
        r_try: r_target,
    };

    // Secant Newton iteration on (Pc, S) (`create_initial_model.f90:155-205`).
    let mut pc0 = exp10(job.center_log_p_1st_try_for_create_initial_model);
    let mut s0 = BOLTZM / MP * job.entropy_1st_try_for_create_initial_model;
    let max_try = job.max_tries_for_create_initial_model;
    let mut cs0 = CreateStarInfo::default();
    let mut cs1 = CreateStarInfo::default();
    let mut cs2 = CreateStarInfo::default();
    let mut itry: isize = 1;
    loop {
        if itry > max_try {
            return Err(StarError::ConvergenceFailure(format!(
                "create initial model failed to converge in allowed number of tries {max_try}"
            )));
        }
        let pc1 = pc0 * 1.01;
        let s1 = s0;
        let pc2 = pc0;
        let s2 = s0 * 1.01;

        // Upstream evaluates the three trials in an OpenMP parallel loop on
        // separate handles (`create_initial_model.f90:177-188`); each trial
        // is independent, so running them concurrently changes no result.
        let (r0, (r1, r2)) = rayon::join(
            || ps_errfunc(&ctx, &mut cs0, pc0, s0, m_target, r_target),
            || {
                rayon::join(
                    || ps_errfunc(&ctx, &mut cs1, pc1, s1, m_target, r_target),
                    || ps_errfunc(&ctx, &mut cs2, pc2, s2, m_target, r_target),
                )
            },
        );
        let (e0, e1, e2) = (r0?, r1?, r2?);

        if e0[0].abs() < job.abs_e01_tolerance_for_create_initial_model
            && e0[1].abs() < job.abs_e02_tolerance_for_create_initial_model
        {
            break;
        }

        let mat11 = (e1[0] - e0[0]) / (0.01 * pc0);
        let mat12 = (e2[0] - e0[0]) / (0.01 * s0);
        let mat21 = (e1[1] - e0[1]) / (0.01 * pc0);
        let mat22 = (e2[1] - e0[1]) / (0.01 * s0);
        let det = mat11 * mat22 - mat12 * mat21;
        let minv11 = mat22 / det;
        let minv22 = mat11 / det;
        let minv12 = -mat12 / det;
        let minv21 = -mat21 / det;
        let d_pc = -(minv11 * e0[0] + minv12 * e0[1]);
        let d_s = -(minv21 * e0[0] + minv22 * e0[1]);

        let safefac = 1.0 + 10.0 * (d_pc / pc0).abs().max((d_s / s0).abs());
        pc0 += d_pc / safefac;
        s0 += d_s / safefac;
        itry += 1;
    }

    // Copy the converged trial into the star (`create_initial_model.f90:207-245`).
    let cs = cs0;
    let nz = cs.rg.len() - 1; // skip center point
    let mut model = StarModel::new(nz, chem_id, job, controls);
    model.eos_config = eos_cfg;
    model.kap_config = kap_cfg;
    model.m_center = 0.0;
    model.l_center = 0.0;
    model.r_center = 0.0;
    model.v_center = 0.0;
    model.mstar = cs.mass;
    model.xmstar = cs.mass;
    model.mstar_start = cs.mass;
    model.mstar_old = cs.mass;

    let mut q = vec![0.0_f64; nz];
    for k in 0..nz {
        // Upstream `i = nz - k + 2` with 1-based `k` and `i`.
        let i = nz - k;
        model.lnd[k] = log(cs.rhog[i]);
        model.ln_t[k] = log(cs.tg[i]);
        model.ln_r[k] = log(cs.rg[i]);
        model.lum[k] = cs.lg[i];
        model.cell_xa_mut(k).copy_from_slice(&xa);
        q[k] = cs.mg[i] / model.xmstar;
    }
    model.dq[nz - 1] = q[nz - 1];
    for k in 0..nz - 1 {
        model.dq[k] = q[k] - q[k + 1];
    }

    // `model_builder` after `build_initial_model` (`init.f90:963-964`) and
    // `create_star` (`init.f90:895`, `1026-1027`). The model enters the run
    // at `model_number = 0` (`alloc_star_data`, `init.f90:367`).
    model.model_number = 0;
    model.star_age = 0.0;
    let mut dt_next = 1.0e-5 * SECYER;
    if model.controls.max_years_for_timestep > 0.0 {
        dt_next = dt_next.min(SECYER * model.controls.max_years_for_timestep);
    }
    (model.dt, model.dt_next, model.dt_old) = (dt_next, dt_next, 0.0);

    // `finish_load_model`: `set_qs` (no `normalize_dqs`), `set_m_and_dm`,
    // `set_m_grav_and_grav`, `set_dm_bar`, then `set_vars`
    // (`star/private/read_model.f90:88-145`).
    model.finish_init_model()?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_with(job: StarJobControls) -> Result<StarModel, StarError> {
        build_initial_model(
            StarControls::default(),
            job,
            EosConfig::default(),
            KapConfig::default(),
        )
    }

    #[test]
    fn test_nonpositive_mass_fails_like_upstream_ierr() {
        let job = StarJobControls {
            create_initial_model: true,
            radius_in_cm_for_create_initial_model: 1.0e10,
            ..StarJobControls::default()
        };
        let err = build_with(job).unwrap_err().to_string();
        assert!(err.contains("must set mass_in_gm_for_create_initial_model"));
    }

    #[test]
    fn test_nonpositive_radius_fails_like_upstream_ierr() {
        let job = StarJobControls {
            create_initial_model: true,
            mass_in_gm_for_create_initial_model: 1.0e30,
            ..StarJobControls::default()
        };
        let err = build_with(job).unwrap_err().to_string();
        assert!(err.contains("must set radius_in_cm_for_create_initial_model"));
    }

    #[test]
    fn test_zero_max_tries_fails_before_any_integration() {
        let job = StarJobControls {
            create_initial_model: true,
            mass_in_gm_for_create_initial_model: 1.0e30,
            radius_in_cm_for_create_initial_model: 1.0e10,
            max_tries_for_create_initial_model: 0,
            ..StarJobControls::default()
        };
        let err = build_with(job).unwrap_err().to_string();
        assert!(err.contains("failed to converge in allowed number of tries 0"));
    }
}
