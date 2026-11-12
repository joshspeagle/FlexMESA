// SPDX-License-Identifier: LGPL-3.0-or-later
//! `GYRE` `l = 0` linear velocity kick, cycle period/amplitude detection
//! (`gather_pulse_statistics`, `check_cycle_completed`), and $P\,\mathrm{d}V$
//! work integrals (`star/private/rsp.f90:574-1028`,
//! `star/test_suite/rsp_gyre/src/run_star_extras.f90`).

use std::fmt::Write as _;
use std::path::Path;

use crate::controls::StarError;
use crate::hydro_brunt_surf::{do_brunt_b, do_brunt_n2};
use crate::model::StarModel;
use crate::pulse::export_gyre_with_schema;
use crate::pulse_modes::solve_gyre_mesa_modes;
use crate::rsp_def::RspDerivedConsts;
use crate::rsp_eos_kap::RspChemCache;

pub(crate) fn rsp_set_gyre_linear_analysis(
    model: &mut StarModel,
    chem: &RspChemCache,
    log_dir: Option<&Path>,
) -> Result<(), StarError> {
    let nzn = model.nz;
    for k0 in 0..nzn {
        if k0 > 0 {
            model.gradt[k0] = model.aux.rsp.y_face_rsp[k0]
                + 0.5 * (model.grada[k0 - 1] + model.grada[k0]);
        }
        model.abar[k0] = chem.abar;
        model.zbar[k0] = chem.zbar;
        for j in 0..model.species.min(chem.xa.len()) {
            model.xa[k0 * model.species + j] = chem.xa[j];
        }
    }
    if nzn >= 2 {
        model.gradt[0] = model.gradt[1];
    }
    model.controls.calculate_brunt_n2 = true;
    do_brunt_b(model);
    do_brunt_n2(model);

    let gyre_txt = export_gyre_with_schema(model, false, 120);
    if let Some(dir) = log_dir {
        let _ = std::fs::write(dir.join("gyre.data"), &gyre_txt);
    }
    let modes = solve_gyre_mesa_modes(&gyre_txt, 0, 0.5, 25.0, 400);
    if modes.is_empty() {
        return Err(StarError::ConvergenceFailure(String::from(
            "rsp_gyre BVP scan returned zero radial eigenmodes",
        )));
    }
    for (m_idx, mode) in modes.iter().take(3).enumerate() {
        let p_sec = 1.0 / (mode.freq_uhz * 1.0e-6).max(1.0e-30);
        model.aux.rsp.lina_periods[m_idx] = p_sec;
        model.aux.rsp.lina_growth_rates[m_idx] = mode.stellingwerf_eta;
        if let Some(dir) = log_dir {
            let mut ef = String::new();
            for (j, yj) in mode.y.iter().enumerate() {
                let _ = writeln!(ef, "{:6} {:24.16e}", j + 1, yj[0]);
            }
            let _ = std::fs::write(
                dir.join(format!("eigfunc.{}.dat", m_idx + 1)),
                ef,
            );
        }
    }

    let mode_idx = model
        .controls
        .rsp
        .mode_for_setting_periodlin
        .min(modes.len() - 1);
    model.aux.rsp.period =
        1.0 / (modes[mode_idx].freq_uhz * 1.0e-6).max(1.0e-30);

    let amix1 = model.controls.x_ctrl.get(4).copied().unwrap_or(0.0);
    let amix2 = model.controls.x_ctrl.get(5).copied().unwrap_or(0.0);
    let velkm = model
        .controls
        .x_ctrl
        .get(6)
        .copied()
        .filter(|&v| v != 0.0)
        .unwrap_or(model.controls.rsp.kick_vsurf_km_per_sec);
    let amix_f = 1.0 - (amix1 + amix2);

    let n_pts = modes[0].y.len();
    let xi_at = |m: usize, j: usize| -> f64 {
        modes.get(m).map_or(0.0, |md| {
            let k0 = nzn.saturating_sub(1 + j);
            let x_j = model.r[k0] / model.r[0].max(1.0e-30);
            x_j * md.y[j][0]
        })
    };
    let j_surf = n_pts - 1;
    let v_surf = amix_f * xi_at(0, j_surf)
        + amix1 * xi_at(1, j_surf)
        + amix2 * xi_at(2, j_surf);
    if v_surf.abs() > 0.0 && velkm != 0.0 {
        let scale = 1.0e5 * velkm / v_surf;
        for j in 0..n_pts.min(nzn) {
            let k0 = nzn - 1 - j;
            model.v[k0] = scale
                * (amix_f * xi_at(0, j)
                    + amix1 * xi_at(1, j)
                    + amix2 * xi_at(2, j));
        }
        if n_pts < nzn && nzn >= 2 {
            model.v[0] = model.v[1];
        }
    }
    model.v_center = 0.0;
    model.aux.rsp.have_set_velocities = true;
    Ok(())
}

pub(crate) fn gather_pulse_statistics(
    model: &mut StarModel,
    dc: &RspDerivedConsts,
) {
    let l_sun_ratio = model.lum[0] / dc.sunl;
    if l_sun_ratio > model.aux.rsp.lmax {
        model.aux.rsp.lmax = l_sun_ratio;
    }
    if l_sun_ratio < model.aux.rsp.lmin {
        model.aux.rsp.lmin = l_sun_ratio;
    }
    model.aux.rsp.inside = 0;
    let cycle_complete = check_cycle_completed(model, dc);
    model.aux.rsp.ull = model.aux.rsp.un;
    model.aux.rsp.te_start = model.aux.rsp.time_sec;
    if cycle_complete {
        let rsp = &mut model.aux.rsp;
        if rsp.num_periods > 1 {
            rsp.grekm =
                (rsp.ekmax - rsp.ekmaxl) / (rsp.ekmax + rsp.ekmaxl) * 2.0;
            let f_new = model.controls.rsp.grekm_avg_abs_frac_new;
            rsp.grekm_avg_abs =
                f_new * rsp.grekm.abs() + (1.0 - f_new) * rsp.grekm_avg_abs;
        }
        rsp.ekdel = rsp.ekmax - rsp.ekmin;
        rsp.ekmaxl = rsp.ekmax;
        rsp.ekmax = -10.0e50;
        rsp.ekmin = 10.0e50;
        rsp.delta_r = rsp.rmax - rsp.rmin;
        rsp.rmax = -dc.sunr;
        rsp.rmin = dc.sunr;
        if rsp.lmax > 0.0 && rsp.lmin > 0.0 {
            rsp.delta_mag = 2.5 * (rsp.lmax / rsp.lmin).log10();
        }
        rsp.lmin = 10.0e10;
        rsp.lmax = -10.0e10;
        rsp.vmax = 0.0;
    }
}

fn check_cycle_completed(model: &mut StarModel, dc: &RspDerivedConsts) -> bool {
    let tet = model.aux.rsp.time_sec;
    let un = model.v[0];
    model.aux.rsp.un = un;
    if model.model_number == 1 {
        return false;
    }
    let ull = model.aux.rsp.ull;
    if un > 0.0 && ull <= 0.0 {
        model.aux.rsp.rmin = model.r[0] / dc.sunr;
    }
    if !model.aux.rsp.have_set_velocities {
        return false;
    }
    let r_surf_sun = model.r[0] / dc.sunr;
    if r_surf_sun < model.controls.rsp.min_max_r_for_periods {
        return false;
    }
    let mach_surf = un / model.csound[0];
    if mach_surf > model.aux.rsp.vmax {
        model.aux.rsp.vmax = mach_surf;
    }
    if un * ull > 0.0 || un > 0.0 {
        return false;
    }
    let mut t0 = tet;
    let min_period =
        model.aux.rsp.period_lin * model.controls.rsp.min_period_div_periodlin;
    if (un - ull).abs() > 1.0e-10 {
        let te_st = model.aux.rsp.te_start;
        t0 = te_st - (te_st - tet) * ull / (ull - un);
    }
    if min_period > 0.0 && t0 - model.aux.rsp.tt1 < min_period {
        return false;
    }
    if r_surf_sun - model.aux.rsp.rmin
        < model.controls.rsp.min_deltar_for_periods
    {
        return false;
    }
    if model.aux.rsp.first == 1 {
        model.aux.rsp.num_periods += 1;
        model.aux.rsp.period = t0 - model.aux.rsp.tt1;
        model.aux.rsp.rmax = r_surf_sun;
        model.aux.rsp.tt1 = t0;
        model.aux.rsp.inside = 1;
        model.aux.rsp.vmax = 0.0;
        true
    } else {
        model.aux.rsp.tt1 = t0;
        model.aux.rsp.first = 1;
        model.aux.rsp.id_step = 0;
        false
    }
}

pub(crate) fn get_grpdv(model: &mut StarModel, dc: &RspDerivedConsts) {
    let nzn = model.nz;
    let dt = model.dt;
    let mut pdvwork = 0.0_f64;
    for i in 1..=nzn {
        let k0 = nzn - i;
        let idx = i - 1;
        let dm = model.dm[k0];
        let rsp = &mut model.aux.rsp;
        let dv = rsp.vv0[idx] - rsp.vol_rsp[k0];
        rsp.work[idx] += dv
            * dm
            * (dc.theta * (rsp.ppp0[idx] + rsp.ppq0[idx])
                + dc.theta1
                    * (model.pgas[k0] + model.prad[k0] + rsp.pvsc_rsp[k0]))
            - dt * dm * rsp.eq_rsp[k0];
        rsp.workq[idx] +=
            dv * dm * (dc.theta * rsp.ppq0[idx] + dc.theta1 * rsp.pvsc_rsp[k0]);
        pdvwork += rsp.work[idx];
    }
    model.aux.rsp.pdvwork = pdvwork;
    let grpdv = pdvwork / model.aux.rsp.ekdel;
    model.aux.rsp.grpdv = if grpdv.is_finite() { grpdv } else { 0.0 };
}

pub(crate) fn calculate_work_integrals(
    model: &mut StarModel,
    dc: &RspDerivedConsts,
    log_dir: Option<&Path>,
) {
    let nzn = model.nz;
    let dt = model.dt;
    if model.aux.rsp.inside == 1 && model.aux.rsp.iwork == 1 {
        model.aux.rsp.iwork = 0;
        for i in 1..=nzn {
            let k0 = nzn - i;
            let idx = i - 1;
            let dm = model.dm[k0];
            let rsp = &mut model.aux.rsp;
            let dvol = rsp.vv0[idx] - rsp.vol_start[k0];
            let p_tw = dc.theta * rsp.ppp0[idx]
                + dc.theta1 * (rsp.pgas_start[k0] + rsp.prad_start[k0]);
            let pvsc_tw =
                dc.theta * rsp.ppq0[idx] + dc.theta1 * rsp.pvsc_start[k0];
            let ptrb_tw =
                dc.thetat * rsp.ppt0[idx] + dc.thetat1 * rsp.ptrb_start[k0];
            rsp.work[idx] += dvol * dm * (p_tw + pvsc_tw + ptrb_tw)
                - dt * dm * rsp.eq_rsp[k0];
            rsp.workq[idx] += dvol * dm * pvsc_tw;
            rsp.workt[idx] += dvol * dm * ptrb_tw;
            rsp.workc[idx] -= dt * dm * rsp.eq_rsp[k0];
        }
        if model.controls.rsp.work_period > 0
            && (model.aux.rsp.num_periods as isize)
                == model.controls.rsp.work_period
            && !model.controls.rsp.work_filename.is_empty()
        {
            if let Some(dir) = log_dir {
                let mut buf = String::new();
                let ekd = model.aux.rsp.ekdel;
                for i in 1..=nzn {
                    let k0 = nzn - i;
                    let idx = i - 1;
                    let dm_ext: f64 = model.dm[..=k0].iter().sum();
                    let rsp = &model.aux.rsp;
                    let _ = writeln!(
                        buf,
                        "{:3} {:7.4} {:16.10e} {:16.10e} {:16.10e} {:16.10e}",
                        i,
                        dm_ext.max(1.0e-99).log10(),
                        rsp.work[idx] / ekd,
                        rsp.workq[idx] / ekd,
                        rsp.workt[idx] / ekd,
                        rsp.workc[idx] / ekd
                    );
                }
                let _ = std::fs::write(
                    dir.join(&model.controls.rsp.work_filename),
                    buf,
                );
            }
        }
        let mut pdvwork = 0.0_f64;
        for idx in 0..nzn {
            pdvwork += model.aux.rsp.work[idx];
            model.aux.rsp.work[idx] = 0.0;
            model.aux.rsp.workq[idx] = 0.0;
            model.aux.rsp.workt[idx] = 0.0;
            model.aux.rsp.workc[idx] = 0.0;
        }
        model.aux.rsp.pdvwork = pdvwork;
        let gr = pdvwork / model.aux.rsp.ekdel;
        model.aux.rsp.grpdv = if gr.is_finite() { gr } else { 0.0 };
    }

    if (model.aux.rsp.num_periods == 0 || model.aux.rsp.inside == 1)
        && model.aux.rsp.iwork == 0
    {
        model.aux.rsp.iwork = 1;
        for i in 1..=nzn {
            let k0 = nzn - i;
            let idx = i - 1;
            let rsp = &mut model.aux.rsp;
            rsp.vv0[idx] = rsp.vol_start[k0];
            rsp.ppp0[idx] = rsp.pgas_start[k0] + rsp.prad_start[k0];
            rsp.ppq0[idx] = rsp.pvsc_start[k0];
            rsp.ppt0[idx] = rsp.ptrb_start[k0];
            rsp.ppc0[idx] = rsp.chi_start[k0];
        }
    }

    if model.aux.rsp.iwork == 1 {
        for i in 1..=nzn {
            let k0 = nzn - i;
            let idx = i - 1;
            let dm = model.dm[k0];
            let rsp = &mut model.aux.rsp;
            let dvol = rsp.vol_rsp[k0] - rsp.vol_start[k0];
            let p_tw = dc.theta * (model.pgas[k0] + model.prad[k0])
                + dc.theta1 * (rsp.pgas_start[k0] + rsp.prad_start[k0]);
            let pvsc_tw =
                dc.theta * rsp.pvsc_rsp[k0] + dc.theta1 * rsp.pvsc_start[k0];
            let ptrb_tw =
                dc.thetat * rsp.ptrb_rsp[k0] + dc.thetat1 * rsp.ptrb_start[k0];
            rsp.work[idx] +=
                dm * (dvol * (p_tw + pvsc_tw + ptrb_tw) - dt * rsp.eq_rsp[k0]);
            rsp.workq[idx] += dm * dvol * pvsc_tw;
            rsp.workt[idx] += dm * dvol * ptrb_tw;
            rsp.workc[idx] -= dt * dm * rsp.eq_rsp[k0];
        }
    }
}
