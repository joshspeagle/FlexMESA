// SPDX-License-Identifier: LGPL-3.0-or-later
//! Substepped implicit rotational angular momentum transport (`solve_omega_mix`)
//! corresponding to `star/private/solve_omega_mix.f90:32-667`.

use flexmesa_const::FOUR_PI;

use crate::model::StarModel;
use crate::struct_burn_mix::ScalarTridiagFactor;

/// Solves rotational angular momentum transport (`do_solve_omega_mix` in
/// `star/private/solve_omega_mix.f90:32-667`).
///
/// Evolves face angular velocity $\Omega_k$ and specific angular momentum
/// $j_{\mathrm{rot},k} = i_{\mathrm{rot},k} \Omega_k$ ($i_{\mathrm{rot},k} = \tfrac{2}{3} r_k^2$)
/// over `dt` using `n_substeps` implicit backward-Euler tridiagonal steps,
/// conserving total angular momentum $J_{\mathrm{tot}} = \sum_{k=0}^{\mathrm{nz}-1} \overline{\Delta m}_k j_{\mathrm{rot},k}$
/// to machine precision.
pub fn solve_omega_mix(
    model: &mut StarModel,
    dt: f64,
    nu_omega: &[f64],
    n_substeps: usize,
) {
    let nz = model.nz;
    if nz < 2 || dt <= 0.0 || nu_omega.len() != nz {
        return;
    }
    let steps = n_substeps.max(1);
    let dt_sub = dt / (steps as f64);

    let i_rot: Vec<f64> = (0..nz)
        .map(|k| {
            let ir = model.aux.i_rot.get(k).copied().unwrap_or(0.0);
            if ir > 0.0 {
                ir
            } else {
                (2.0 / 3.0) * model.r[k] * model.r[k]
            }
        })
        .collect();

    let mut c_omega = vec![0.0; nz - 1];
    for (k, c_om_k) in c_omega.iter_mut().enumerate() {
        let nu0 = nu_omega[k].max(0.0);
        let nu1 = nu_omega[k + 1].max(0.0);
        let d_harm = if nu0 + nu1 > 1.0e-99 {
            2.0 * nu0 * nu1 / (nu0 + nu1)
        } else {
            0.0
        };
        let r_mid = 0.5 * (model.r[k] + model.r[k + 1]);
        let area_rho = FOUR_PI * r_mid * r_mid * model.rho[k];
        let am_sig_raw =
            (area_rho * area_rho * d_harm) / model.dm[k].max(1.0e-99);
        let i_mid = 0.5 * (i_rot[k] + i_rot[k + 1]);
        *c_om_k = am_sig_raw * i_mid;
    }

    let mut dl = vec![0.0; nz];
    let mut d = vec![1.0; nz];
    let mut du = vec![0.0; nz];
    for k in 0..nz {
        let denom = (model.dm_bar[k] * i_rot[k]).max(1.0e-99);
        let f_k = dt_sub / denom;
        let c_m1 = if k > 0 { c_omega[k - 1] } else { 0.0 };
        let c_00 = if k + 1 < nz { c_omega[k] } else { 0.0 };
        dl[k] = -f_k * c_m1;
        d[k] = 1.0 + f_k * (c_m1 + c_00);
        du[k] = -f_k * c_00;
    }

    let factor = ScalarTridiagFactor::factor(&dl, &d, &du);
    let mut rhs = vec![0.0; nz];
    let mut d_omega = vec![0.0; nz];
    let mut d_prime = vec![0.0; nz];
    for _ in 0..steps {
        for (k, r_k) in rhs.iter_mut().enumerate() {
            let denom = (model.dm_bar[k] * i_rot[k]).max(1.0e-99);
            let f_k = dt_sub / denom;
            let torque_m1 = if k > 0 {
                c_omega[k - 1] * (model.omega[k - 1] - model.omega[k])
            } else {
                0.0
            };
            let torque_00 = if k + 1 < nz {
                c_omega[k] * (model.omega[k] - model.omega[k + 1])
            } else {
                0.0
            };
            *r_k = f_k * (torque_m1 - torque_00);
        }
        factor.solve_into(&dl, &rhs, &mut d_omega, &mut d_prime);
        for (k, &dw) in d_omega.iter().enumerate() {
            model.omega[k] += dw;
        }
        if model.dm_bar[0] <= 0.0 && nz >= 2 {
            model.omega[0] = model.omega[1];
        }
    }
    for (k, &ir) in i_rot.iter().enumerate() {
        model.j_rot[k] = ir * model.omega[k];
    }
}
