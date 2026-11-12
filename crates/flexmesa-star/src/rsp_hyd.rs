// SPDX-License-Identifier: LGPL-3.0-or-later
//! Nonlinear implicit `RSP` `HYD` timestepper (`HYD`, `init_HYD`,
//! `apply_corrections`, `calculate_energies`, and `rsp_set_Teff` in
//! `star/private/rsp_step.f90:202-696, 945-1083`).

use flexmesa_atm::atm_teff;
use flexmesa_const::{CRAD, PI};

use crate::controls::StarError;
use crate::model::StarModel;
use crate::rsp_def::{RspDerivedConsts, F_EDD_ISOTROPIC};
use crate::rsp_eos_kap::{get_surf_p_t_kap, RspChemCache};
use crate::rsp_eqns::{eval_eqns, solve_for_corrections, RspHydBand, NV};
use crate::rsp_hyd_vars::eval_vars;
use crate::rsp_hyd_work::RspHydWork;

/// Initializes `HYD` start-of-step quantities and residuals on a freshly built
/// or loaded `RSP` model (`init_HYD` in `rsp_step.f90:580-595`).
pub fn init_hyd(
    model: &mut StarModel,
    chem: &RspChemCache,
    dc: &RspDerivedConsts,
    wk: &mut RspHydWork,
    band: &mut RspHydBand,
) -> Result<(), StarError> {
    let nz = model.nz;
    model.aux.rsp.f_edd_rsp[..nz].fill(F_EDD_ISOTROPIC);
    eval_vars(model, chem, dc, wk, 0, 1, nz)?;
    set_f_edd(model);
    save_start_vars(model, wk);
    eval_eqns(model, dc, wk, band, 0.0);
    Ok(())
}

/// Advances the coupled 5-equation nonlinear `RSP` envelope by one implicit
/// `HYD` step with step-halving retries (`HYD` in `rsp_step.f90:202-504`).
///
/// Returns `(solver_iterations, retries_taken)`.
pub fn hyd(
    model: &mut StarModel,
    chem: &RspChemCache,
    dc: &RspDerivedConsts,
    wk: &mut RspHydWork,
    band: &mut RspHydBand,
) -> Result<(usize, usize), StarError> {
    let nz = model.nz;
    let prec2 = model.controls.rsp.tol_max_corr;
    let dxh = 0.3_f64;
    let max_retries = model.controls.rsp.max_retries_per_step;
    let max_iters = model.controls.rsp.max_iters_per_try.max(2);
    let r_center_start = model.r_center;

    save_start_vars(model, wk);
    for k0 in 0..nz {
        let l_k = model.aux.rsp.fr_rsp[k0] * 4.0 * PI * model.r[k0].powi(2)
            + model.aux.rsp.lc_rsp[k0]
            + model.aux.rsp.lt_rsp[k0];
        model.lum[k0] = l_k;
        model.lum_start[k0] = l_k;
    }
    set_f_edd(model);
    let p_surf = get_psurf(model, chem)?;
    set_min_k_for_turbulent_flux(model, dc);

    if model.v_center > model.v[nz - 1] && model.v_center > 0.0 {
        let dt_max = 1.0e-2 * (model.r[nz - 1] - model.r_center)
            / (model.v_center - model.v[nz - 1]);
        if model.dt > dt_max {
            model.dt = dt_max;
        }
    }
    let f_min = model.controls.rsp.force_timestep_min;
    if f_min > 0.0 && model.dt < f_min {
        model.dt = f_min;
    }
    if model.controls.rsp.force_timestep > 0.0 {
        model.dt = model.controls.rsp.force_timestep;
    }

    let mut retries = 0usize;
    for num_tries in 1..=(max_retries + 1) {
        let mut converged = false;
        set_1st_iter_r_using_v_start(model);
        model.r_center += model.dt * model.v_center;

        for iter in 1..=max_iters {
            eval_vars(model, chem, dc, wk, iter, 1, nz)?;
            eval_eqns(model, dc, wk, band, p_surf);
            if converged {
                return Ok((iter - 1, retries));
            }
            if let Err(err) = solve_for_corrections(band) {
                if num_tries == max_retries + 1 {
                    return Err(err);
                }
                break;
            }
            let (dxxt, dxxc) = apply_corrections(model, dc, wk, band, dxh);
            if iter == 1 {
                continue;
            }
            converged = dxxt < prec2 && dxxc < prec2;
        }

        if num_tries == max_retries + 1 {
            return Err(StarError::ConvergenceFailure(format!(
                "RSP HYD failed to converge at model {} after {} tries",
                model.model_number, num_tries
            )));
        }
        restore_start_vars(model, wk);
        model.r_center = r_center_start;
        model.dt *= 0.5;
        retries += 1;
    }

    Err(StarError::ConvergenceFailure(format!(
        "RSP HYD exhausted retries at model {}",
        model.model_number
    )))
}

/// Evaluates surface pressure $P_{\mathrm{surf}}$ (`get_Psurf` in `rsp_step.f90:507-534`).
pub fn get_psurf(
    model: &StarModel,
    chem: &RspChemCache,
) -> Result<f64, StarError> {
    if model.controls.rsp.fixed_psurf {
        Ok(model.aux.rsp.psurf_from_atm)
    } else if model.controls.rsp.use_atm_grey_with_kap_for_psurf {
        let (_t_s, p_s, _kap_s, _teff_a) = get_surf_p_t_kap(
            model,
            chem,
            model.m[0],
            model.r[0],
            model.lum[0],
            (2.0 / 3.0) * model.controls.tau_factor,
            1.0e-2,
        )?;
        Ok(p_s)
    } else if model.controls.rsp.use_prad_for_psurf {
        Ok(CRAD * model.t[0].powi(4) / 3.0)
    } else if model.controls.rsp.psurf >= 0.0 {
        Ok(model.controls.rsp.psurf)
    } else {
        Ok(0.0)
    }
}

/// Evaluates thermal, turbulent, kinetic, gravitational, and radiated energies
/// (`calculate_energies` in `rsp_step.f90:537-577`, `star_utils.f90:2215-2290`).
pub fn calculate_energies(
    model: &mut StarModel,
    dc: &RspDerivedConsts,
    wk: &RspHydWork,
) -> f64 {
    let nzn = wk.nzn;
    let mut egrv = 0.0_f64;
    let mut ethe = 0.0_f64;
    let mut ekin = 0.0_f64;
    let mut econ = 0.0_f64;

    for i in 1..=nzn {
        let k0 = nzn - i;
        let dm = model.dm[k0];
        let e_therm = if model.controls.rsp2_flag {
            model.energy.get(k0).copied().unwrap_or(0.0)
        } else {
            model.aux.rsp.egas_rsp[k0] + model.aux.rsp.erad_rsp[k0]
        };
        ethe += e_therm * dm;
        econ += wk.w[k0].powi(2) * dm;

        let v1 = if k0 + 1 < nzn {
            model.v[k0 + 1]
        } else {
            model.v_center
        };
        let ke_spec = 0.25 * (model.v[k0].powi(2) + v1.powi(2));
        ekin += ke_spec * dm;

        let (rp1, mp1, gp1) = if k0 + 1 == nzn {
            (model.r_center, model.m_center, model.cgrav[nzn - 1])
        } else {
            (model.r[k0 + 1], model.m_grav[k0 + 1], model.cgrav[k0 + 1])
        };
        let gravp1 = if rp1 <= 0.0 { 0.0 } else { -gp1 * mp1 / rp1 };
        let grav00 = -model.cgrav[k0] * model.m_grav[k0] / model.r[k0];
        let pe_spec = 0.5 * (gravp1 + grav00);
        egrv += pe_spec * dm;
    }

    let total_radiation = if model.controls.rsp.hydro_only {
        0.0
    } else {
        let (wtr, wtr1) = if model.controls.rsp2_flag {
            let l_th = if model.controls.using_velocity_time_centering
                && model.controls.include_l_in_velocity_time_centering
            {
                model.controls.l_theta_for_velocity_time_centering
            } else {
                1.0
            };
            (l_th, 1.0 - l_th)
        } else {
            (dc.wtr, dc.wtr1)
        };
        model.dt
            * (model.l_center
                - (wtr * model.lum[0] + wtr1 * model.lum_start[0]))
    };

    let rsp = &mut model.aux.rsp;
    rsp.ethe = ethe;
    rsp.econ = econ;
    rsp.ekin = ekin;
    rsp.egrv = egrv;
    rsp.etot = ethe + ekin + econ + egrv;
    rsp.ede_start -= rsp.e0;
    if ekin > rsp.ekmax {
        rsp.ekmax = ekin;
    }
    if ekin < rsp.ekmin {
        rsp.ekmin = ekin;
    }
    total_radiation
}

/// Evaluates photospheric radius and updates `model.teff` via `get_phot_info` +
/// `atm_Teff` (`rsp_set_Teff` in `rsp_step.f90:621-630`, `star_utils.f90:1129-1208`).
pub fn rsp_set_teff(model: &mut StarModel) {
    let (l_phot, r_phot) = get_phot_l_and_r(model);
    model.teff = atm_teff(l_phot, r_phot);
}

fn get_phot_l_and_r(model: &StarModel) -> (f64, f64) {
    let nz = model.nz;
    if nz == 0 {
        return (1.0, 1.0);
    }
    let l_surf = model.lum[0].max(1.0);
    if model.controls.tau_factor >= 1.0 {
        return (l_surf, model.r[0]);
    }
    let tau_phot = model.tau_base;
    let mut tau00 = model.controls.tau_factor * model.tau_base;
    let pi4 = 4.0 * PI;
    for k0 in 0..(nz.saturating_sub(1)) {
        let rmid = 0.5 * (model.r[k0] + model.r[k0 + 1]);
        let dtau = model.dm[k0] * model.opacity[k0] / (pi4 * rmid * rmid);
        let taup1 = tau00 + dtau;
        if taup1 >= tau_phot && dtau > 0.0 {
            let r003 = model.r[k0].powi(3);
            let rp13 = model.r[k0 + 1].powi(3);
            let r3 = r003 + (rp13 - r003) * (tau_phot - tau00) / dtau;
            return (l_surf, r3.cbrt());
        }
        tau00 = taup1;
    }
    (model.l_center.max(1.0), model.r_center)
}

fn set_f_edd(model: &mut StarModel) {
    let nz = model.nz;
    model.aux.rsp.g_edd = 0.5;
    model.aux.rsp.f_edd_rsp[..nz].fill(F_EDD_ISOTROPIC);
}

fn set_min_k_for_turbulent_flux(model: &mut StarModel, dc: &RspDerivedConsts) {
    if dc.alfat <= 0.0 {
        model.aux.rsp.min_k_for_turbulent_flux = 0;
        return;
    }
    let nz = model.nz;
    let mut tau = 0.0_f64;
    let mut min_k = nz;
    for k0 in 0..(nz.saturating_sub(1)) {
        let rmid = 0.5 * (model.r[k0] + model.r[k0 + 1]);
        tau += model.dm[k0] * model.opacity[k0] / (4.0 * PI * rmid * rmid);
        if tau >= model.controls.rsp.min_tau_for_turbulent_flux {
            min_k = k0 + 1;
            break;
        }
    }
    model.aux.rsp.min_k_for_turbulent_flux = min_k;
}

fn set_1st_iter_r_using_v_start(model: &mut StarModel) {
    let nzn = model.nz;
    let mut ehjt = 1.0_f64;
    for k0 in 0..nzn {
        let eh1 = if k0 + 1 == nzn {
            -(model.aux.rsp.v_start_rsp[k0] - model.v_center) * model.dt
                / (model.aux.rsp.r_start_rsp[k0] - model.r_center)
                / 0.8
        } else {
            -(model.aux.rsp.v_start_rsp[k0] - model.aux.rsp.v_start_rsp[k0 + 1])
                * model.dt
                / (model.aux.rsp.r_start_rsp[k0]
                    - model.aux.rsp.r_start_rsp[k0 + 1])
                / 0.8
        };
        ehjt = 1.0 / (1.0 / ehjt).max(eh1).max(1.0);
    }
    for k0 in 0..nzn {
        model.r[k0] = model.aux.rsp.r_start_rsp[k0]
            + ehjt * model.dt * model.aux.rsp.v_start_rsp[k0];
        model.v[k0] = (2.0 * ehjt - 1.0) * model.aux.rsp.v_start_rsp[k0];
    }
}

fn save_start_vars(model: &mut StarModel, wk: &RspHydWork) {
    for i in 1..=wk.nzn {
        let k0 = wk.nzn - i;
        let rsp = &mut model.aux.rsp;
        rsp.t_start_rsp[k0] = model.t[k0];
        rsp.r_start_rsp[k0] = model.r[k0];
        rsp.pgas_start[k0] = model.pgas[k0];
        rsp.prad_start[k0] = model.prad[k0];
        rsp.pvsc_start[k0] = rsp.pvsc_rsp[k0];
        rsp.vol_start[k0] = rsp.vol_rsp[k0];
        rsp.csound_start_rsp[k0] = model.csound[k0];
        rsp.opacity_start[k0] = model.opacity[k0];
        rsp.egas_start[k0] = rsp.egas_rsp[k0];
        rsp.erad_start[k0] = rsp.erad_rsp[k0];
        rsp.w_start[k0] = wk.w[k0];
        rsp.ptrb_start[k0] = rsp.ptrb_rsp[k0];
        rsp.chi_start[k0] = rsp.chi_rsp[k0];
        rsp.v_start_rsp[k0] = model.v[k0];
        rsp.fr_start[k0] = rsp.fr_rsp[k0];
        rsp.lc_start[k0] = rsp.lc_rsp[k0];
        rsp.lt_start[k0] = rsp.lt_rsp[k0];
        rsp.coupl_start[k0] = rsp.coupl_rsp[k0];
    }
}

fn restore_start_vars(model: &mut StarModel, wk: &mut RspHydWork) {
    for i in 1..=wk.nzn {
        let k0 = wk.nzn - i;
        let rsp = &mut model.aux.rsp;
        model.t[k0] = rsp.t_start_rsp[k0];
        model.ln_t[k0] = model.t[k0].ln();
        model.r[k0] = rsp.r_start_rsp[k0];
        model.pgas[k0] = rsp.pgas_start[k0];
        model.prad[k0] = rsp.prad_start[k0];
        rsp.pvsc_rsp[k0] = rsp.pvsc_start[k0];
        rsp.vol_rsp[k0] = rsp.vol_start[k0];
        model.csound[k0] = rsp.csound_start_rsp[k0];
        model.opacity[k0] = rsp.opacity_start[k0];
        rsp.egas_rsp[k0] = rsp.egas_start[k0];
        rsp.erad_rsp[k0] = rsp.erad_start[k0];
        wk.w[k0] = rsp.w_start[k0];
        rsp.et_rsp[k0] = wk.w[k0] * wk.w[k0];
        rsp.ptrb_rsp[k0] = rsp.ptrb_start[k0];
        rsp.chi_rsp[k0] = rsp.chi_start[k0];
        model.v[k0] = rsp.v_start_rsp[k0];
        model.lum[k0] = model.lum_start[k0];
        rsp.fr_rsp[k0] = rsp.fr_start[k0];
        rsp.lc_rsp[k0] = rsp.lc_start[k0];
        rsp.lt_rsp[k0] = rsp.lt_start[k0];
        rsp.coupl_rsp[k0] = rsp.coupl_start[k0];
    }
}

fn apply_corrections(
    model: &mut StarModel,
    dc: &RspDerivedConsts,
    wk: &mut RspHydWork,
    band: &RspHydBand,
    dxh: f64,
) -> (f64, f64) {
    let dx = &band.hr;
    let mut ezh = 1.0_f64;
    for i in 1..=wk.nzn {
        let k0 = wk.nzn - i;
        let (it, iw, ie, _il, ir) = RspHydBand::row_indices(i);
        let xxc = if wk.w[k0] > 1.0e2 * dc.efl0 {
            (dx[iw] / wk.w[k0]).abs() / dxh
        } else {
            0.0
        };
        let xxr = if i == 1 {
            (-dx[ir] / (model.r[k0] - model.r_center)) / dxh
        } else {
            ((dx[ir - NV] - dx[ir]) / (model.r[k0] - model.r[k0 + 1])) / dxh
        };
        let xxe = (dx[ie] / model.aux.rsp.erad_rsp[k0]).abs() / dxh;
        let xxt = (dx[it] / model.t[k0]).abs() / dxh;
        ezh = 1.0 / (1.0 / ezh).max(xxr).max(xxt).max(xxc).max(xxe);
    }

    let mut dxxt = 0.0_f64;
    let mut dxxc = 0.0_f64;
    for i in 1..=wk.nzn {
        let k0 = wk.nzn - i;
        let (it, iw, ie, il, ir) = RspHydBand::row_indices(i);
        model.t[k0] += ezh * dx[it];
        model.ln_t[k0] = model.t[k0].ln();
        model.aux.rsp.erad_rsp[k0] += ezh * dx[ie];
        if i > dc.ibotom && i < wk.nzn {
            if wk.w[k0] + ezh * dx[iw] <= 0.0 {
                wk.w[k0] = dc.efl0 * model.aux.rsp.rng.next_f64() * 1.0e-6;
            } else {
                wk.w[k0] += ezh * dx[iw];
            }
            model.aux.rsp.et_rsp[k0] = wk.w[k0] * wk.w[k0];
        }
        model.v[k0] = model.v[k0]
            - (model.v[k0] + model.aux.rsp.v_start_rsp[k0])
            + 2.0 / model.dt
                * (ezh * dx[ir]
                    + (model.r[k0] - model.aux.rsp.r_start_rsp[k0]));
        model.r[k0] += ezh * dx[ir];
        model.aux.rsp.fr_rsp[k0] += ezh * dx[il];

        dxxt = dxxt.max((dx[it] / model.t[k0]).abs());
        if wk.w[k0] > 1.0e-2 * dc.efl0 {
            dxxc = dxxc.max((dx[iw] / wk.w[k0]).abs());
        }
    }
    (dxxt, dxxc)
}
