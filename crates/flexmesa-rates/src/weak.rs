// SPDX-License-Identifier: GPL-3.0-or-later
//! Tabulated weak reaction evaluation (`weakreactions.tables`,
//! `S13_r_be7_wk_li7.h5`, and `Suzuki2016.h5`) ported from
//! `rates/private/eval_weak.f90`, `weaklib_tables.f90`, and `suzuki_tables.f90`.

use flexmesa_const::LN10;

use crate::loader::{RatesCatalog, SuzukiTable, WeakTable};
use crate::types::{RatesError, WeakRateEval};

#[inline]
fn find_location(xs: &[f64], x: f64) -> usize {
    // Matches `find_location` in `rates/private/weaklib_tables.f90:272-286`
    // and `rates/private/suzuki_tables.f90:256-270`: exact interior grid points
    // `x == xs[i]` select cell `[xs[i - 1], xs[i]]` (`ix = i - 1`).
    let nx = xs.len();
    for (i, &xi) in xs.iter().enumerate().take(nx - 1).skip(1) {
        if x <= xi {
            return i - 1;
        }
    }
    nx - 2
}

/// Performs 2D bilinear interpolation on a row-major `(nx, ny)` matrix `z_flat`
/// with axis clamping and exact partial derivatives `(f, df_dx, df_dy)` matching
/// `weaklib_tables.f90` and `suzuki_tables.f90`.
#[must_use]
pub fn bilinear_interp_2d(
    xs: &[f64],
    ys: &[f64],
    z_flat: &[f64],
    x_in: f64,
    y_in: f64,
) -> (f64, f64, f64) {
    let nx = xs.len();
    let ny = ys.len();
    if nx < 2 || ny < 2 {
        return (0.0, 0.0, 0.0);
    }
    let x = x_in.clamp(xs[0], xs[nx - 1]);
    let y = y_in.clamp(ys[0], ys[ny - 1]);

    let ix = find_location(xs, x);
    let iy = find_location(ys, y);

    let (x0, x1) = (xs[ix], xs[ix + 1]);
    let (y0, y1) = (ys[iy], ys[iy + 1]);
    let xb = (x - x0) / (x1 - x0);
    let xa = 1.0 - xb;
    let yb = (y - y0) / (y1 - y0);
    let ya = 1.0 - yb;

    let f00 = z_flat[ix * ny + iy];
    let f01 = z_flat[ix * ny + iy + 1];
    let f10 = z_flat[(ix + 1) * ny + iy];
    let f11 = z_flat[(ix + 1) * ny + iy + 1];

    let fx0 = ya * f00 + yb * f01;
    let fx1 = ya * f10 + yb * f11;
    let fy0 = xa * f00 + xb * f10;
    let fy1 = xa * f01 + xb * f11;

    let val = xa * fx0 + xb * fx1;
    let df_dx = (fx1 - fx0) / (x1 - x0);
    let df_dy = (fy1 - fy0) / (y1 - y0);
    (val, df_dx, df_dy)
}

/// Evaluates a single [`WeakTable`] at `(t9, l_ye_rho)`.
#[must_use]
pub fn eval_weaklib_one(
    tab: &WeakTable,
    t9_in: f64,
    l_ye_rho_in: f64,
) -> WeakRateEval {
    // `eval_weak.f90:226-235` + `weaklib_tables.f90:249-265`
    let t9 = if tab.t9s.is_empty() {
        t9_in
    } else {
        t9_in.clamp(tab.t9s[0], tab.t9s[tab.t9s.len() - 1])
    };
    let l_ye_rho = if tab.lyerhos.is_empty() {
        l_ye_rho_in
    } else {
        l_ye_rho_in.clamp(tab.lyerhos[0], tab.lyerhos[tab.lyerhos.len() - 1])
    };

    let (ldec, dldec_dt9, dldec_dly) =
        bilinear_interp_2d(&tab.t9s, &tab.lyerhos, &tab.ldecay, t9, l_ye_rho);
    let (lcap, dlcap_dt9, dlcap_dly) =
        bilinear_interp_2d(&tab.t9s, &tab.lyerhos, &tab.lcapture, t9, l_ye_rho);
    let (lneu, dlneu_dt9, dlneu_dly) = bilinear_interp_2d(
        &tab.t9s,
        &tab.lyerhos,
        &tab.lneutrino,
        t9,
        l_ye_rho,
    );

    let decay = flexmesa_math::exp10(ldec);
    let capture = flexmesa_math::exp10(lcap);
    let lambda = decay + capture;
    let dlambda_dln_t = LN10 * t9 * (decay * dldec_dt9 + capture * dlcap_dt9);
    let dlambda_dln_rho = decay * dldec_dly + capture * dlcap_dly;

    let (qneu, dqneu_dln_t, dqneu_dln_rho) = if lambda < 1.0e-30 {
        (0.0, 0.0, 0.0)
    } else {
        let qn = flexmesa_math::exp10(lneu) / lambda;
        let dqn_dt = LN10 * t9 * qn * dlneu_dt9 - dlambda_dln_t * qn / lambda;
        let dqn_dr = qn * dlneu_dly - dlambda_dln_rho * qn / lambda;
        (qn, dqn_dt, dqn_dr)
    };

    WeakRateEval {
        lambda,
        dlambda_dln_t,
        dlambda_dln_rho,
        q: tab.q_tot,
        dq_dln_t: 0.0,
        dq_dln_rho: 0.0,
        qneu,
        dqneu_dln_t,
        dqneu_dln_rho,
    }
}

fn eval_suzuki_channel(
    enabled: bool,
    logts: &[f64],
    lyerhos: &[f64],
    rate_grid: &[f64],
    nu_grid: &[f64],
    log_t: f64,
    l_ye_rho: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    if !enabled {
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let (lr, dlr_dt, dlr_dr) =
        bilinear_interp_2d(logts, lyerhos, rate_grid, log_t, l_ye_rho);
    let (ln, dln_dt, dln_dr) =
        bilinear_interp_2d(logts, lyerhos, nu_grid, log_t, l_ye_rho);
    (
        flexmesa_math::exp10(lr),
        dlr_dt,
        dlr_dr,
        flexmesa_math::exp10(ln),
        dln_dt,
        dln_dr,
    )
}

/// Evaluates a single [`SuzukiTable`] at `(t9, l_ye_rho)` matching
/// `rates/private/suzuki_tables.f90:136-254`.
#[must_use]
pub fn eval_suzuki_one(
    tab: &SuzukiTable,
    t9: f64,
    l_ye_rho: f64,
) -> WeakRateEval {
    let log_t = flexmesa_math::log10(t9) + 9.0;
    let (decay, dldec_dlogt, dldec_dly, decay_nu, dldec_nu_dlogt, dldec_nu_dly) =
        eval_suzuki_channel(
            tab.has_decay,
            &tab.logts,
            &tab.lyerhos,
            &tab.decay_rate,
            &tab.decay_nu,
            log_t,
            l_ye_rho,
        );
    let (
        capture,
        dlcap_dlogt,
        dlcap_dly,
        capture_nu,
        dlcap_nu_dlogt,
        dlcap_nu_dly,
    ) = eval_suzuki_channel(
        tab.has_capture,
        &tab.logts,
        &tab.lyerhos,
        &tab.capture_rate,
        &tab.capture_nu,
        log_t,
        l_ye_rho,
    );

    let lambda = decay + capture;
    let dlambda_dln_t = decay * dldec_dlogt + capture * dlcap_dlogt;
    let dlambda_dln_rho = decay * dldec_dly + capture * dlcap_dly;

    let (qneu, dqneu_dln_t, dqneu_dln_rho) = if lambda > 1.0e-30 {
        let nu = capture_nu + decay_nu;
        let qn = nu / lambda;
        let mut dqn_dt = 0.0;
        let mut dqn_dr = 0.0;
        if nu > 1.0e-30 {
            if tab.has_decay {
                dqn_dt += qn
                    * ((decay_nu / nu) * dldec_nu_dlogt
                        - (decay / lambda) * dldec_dlogt);
                dqn_dr += qn
                    * ((decay_nu / nu) * dldec_nu_dly
                        - (decay / lambda) * dldec_dly);
            }
            if tab.has_capture {
                dqn_dt += qn
                    * ((capture_nu / nu) * dlcap_nu_dlogt
                        - (capture / lambda) * dlcap_dlogt);
                dqn_dr += qn
                    * ((capture_nu / nu) * dlcap_nu_dly
                        - (capture / lambda) * dlcap_dly);
            }
        }
        (qn, dqn_dt, dqn_dr)
    } else {
        (0.0, 0.0, 0.0)
    };

    WeakRateEval {
        lambda,
        dlambda_dln_t,
        dlambda_dln_rho,
        q: tab.q_tot,
        dq_dln_t: 0.0,
        dq_dln_rho: 0.0,
        qneu,
        dqneu_dln_t,
        dqneu_dln_rho,
    }
}

/// Evaluates weak reaction rates for `(lhs_name, rhs_name)` at `(t9, ye_rho)`
/// using either `weaklib` (`use_suzuki = false`) or `Suzuki2016` (`use_suzuki = true`),
/// falling back to `weaklib` when `(log_t, l_ye_rho)` lies outside the `Suzuki2016`
/// grid bounds (`eval_weak.f90:225-268`, `suzuki_tables.f90:144-155`).
///
/// # Errors
/// Returns [`RatesError::UnknownWeakPair`] if `(lhs_name, rhs_name)` is not found.
pub fn eval_weak_reaction_pair(
    lhs_name: &str,
    rhs_name: &str,
    t9_in: f64,
    ye_rho: f64,
    use_suzuki: bool,
) -> Result<WeakRateEval, RatesError> {
    let cat = RatesCatalog::builtin();
    let wtab = cat.weak_by_pair(lhs_name, rhs_name)?;
    let t9 = t9_in.clamp(wtab.t9s[0], wtab.t9s[wtab.t9s.len() - 1]);
    let l_ye_rho = flexmesa_math::log10(ye_rho.max(1.0e-99))
        .clamp(wtab.lyerhos[0], wtab.lyerhos[wtab.lyerhos.len() - 1]);
    if use_suzuki {
        if let Some(stab) = cat.suzuki_by_pair(lhs_name, rhs_name) {
            let log_t = flexmesa_math::log10(t9) + 9.0;
            let in_bounds = !stab.logts.is_empty()
                && !stab.lyerhos.is_empty()
                && log_t >= stab.logts[0]
                && log_t <= stab.logts[stab.logts.len() - 1]
                && l_ye_rho >= stab.lyerhos[0]
                && l_ye_rho <= stab.lyerhos[stab.lyerhos.len() - 1];
            if in_bounds {
                return Ok(eval_suzuki_one(stab, t9, l_ye_rho));
            }
        }
    }
    Ok(eval_weaklib_one(wtab, t9, l_ye_rho))
}
