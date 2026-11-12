// SPDX-License-Identifier: LGPL-3.0-or-later
//! Linear Non-Adiabatic (`LINA`) $4N \times 4N$ radial pulsation stability
//! eigensolver (`do_LINA` in `star/private/rsp_lina.f90:33-1167`).

use std::path::Path;

use flexmesa_const::PI;
use flexmesa_mtx::dgeev_right;

use crate::controls::StarError;
use crate::model::StarModel;
use crate::rsp_def::RspDerivedConsts;
use crate::rsp_eos_kap::RspChemCache;
use crate::rsp_grid::{mesa_eos_kap, RspEnvGrid};
use crate::rsp_lina_io::{write_lina_mode_logs, write_lina_summary, Cpx};
use crate::rsp_turb_partials::RspTurbPartials;
use crate::rsp_zone_micro::{LcFilterMode, RspZoneMicro};

/// Output of [`do_lina`] across the lowest `nmodes` acoustic radial overtones.
#[derive(Debug, Clone)]
pub struct RspLinaOutput {
    /// Linear radial periods $P_j = 2\pi / \omega_{\mathrm{imag},j}$ in seconds (`len = nmodes`).
    pub periods_sec: Vec<f64>,
    /// Linear fractional amplitude growth rates $\eta_j = 4\pi \omega_{\mathrm{real},j} / \omega_{\mathrm{imag},j}$ (`len = nmodes`).
    pub growth_rates: Vec<f64>,
    /// Pulsation constants $Q_j$ in days (`len = nmodes`).
    pub q_days: Vec<f64>,
    /// Mode kinetic energies $E_{K,j}$ in $\mathrm{erg}$ (`len = nmodes`).
    pub kinetic_energies: Vec<f64>,
    /// Signed surface-normalized radial velocity eigenfunctions `vel[j][i]` (`1..=nzn`, Smolec center-to-surface).
    pub vel_modes: Vec<Vec<f64>>,
}

/// Solves the $4N \times 4N$ linearized non-adiabatic radial pulsation stability
/// eigenvalue problem $\mathbf{L}\,\mathbf{x} = \sigma\,\mathbf{x}$ (`do_LINA`
/// in `star/private/rsp_lina.f90:33-1167`).
#[allow(clippy::too_many_lines)]
pub fn do_lina(
    model: &mut StarModel,
    chem: &RspChemCache,
    dc: &RspDerivedConsts,
    grid_in: &RspEnvGrid,
    l0: f64,
    nmodes: usize,
    log_dir: Option<&Path>,
) -> Result<RspLinaOutput, StarError> {
    let nzn = grid_in.nzn;
    let mut grid = grid_in.clone();
    let mut efl02 = dc.efl0 * dc.efl0;
    if dc.alfa == 0.0 {
        efl02 = 0.0;
        grid.et.fill(0.0);
    }

    let mut mc = RspZoneMicro::new(nzn);
    for i in 1..=nzn {
        if grid.et[i] <= efl02 {
            grid.et[i] = efl02;
        }
        let out = mesa_eos_kap(model, chem, 0, nzn, grid.t[i], grid.vol[i])?;
        mc.p[i] = out.p;
        mc.dpv[i] = out.dpv;
        mc.dp_dt_00[i] = out.dpt;
        mc.e[i] = out.e;
        mc.dev[i] = out.dev;
        mc.de_dt_00[i] = out.det;
        mc.cps[i] = out.cp;
        mc.cpv[i] = out.cpv;
        mc.dcp_dt_00[i] = out.cpt;
        mc.qqs[i] = out.qq;
        mc.qqv[i] = out.qqv;
        mc.dqq_dt_00[i] = out.qqt;
        mc.k[i] = out.op;
        mc.dk_dv_00[i] = out.opv;
        mc.dk_dt_00[i] = out.opt;

        let t1 = dc.p43 / grid.dm[i];
        mc.dvr[i] = 3.0 * t1 * grid.r[i].powi(2);
        mc.dvrm[i] = if i == 1 {
            -3.0 * t1 * model.r_center.powi(2)
        } else {
            -3.0 * t1 * grid.r[i - 1].powi(2)
        };
        mc.update_radial_chain(i);
    }

    let mut tb = RspTurbPartials::new(nzn);
    tb.evaluate(
        dc,
        dc.alfa,
        dc.alfap,
        dc.alfat,
        efl02,
        &mut grid,
        &mc,
        LcFilterMode::Lina { efl02 },
    );

    let nzn4 = 4 * nzn;
    let mut lll = vec![0.0_f64; nzn4 * nzn4];
    let set_l = |mat: &mut [f64], r1: isize, c1: isize, val: f64| {
        if r1 >= 1 && r1 <= nzn4 as isize && c1 >= 1 && c1 <= nzn4 as isize {
            mat[(r1 as usize - 1) * nzn4 + (c1 as usize - 1)] = val;
        }
    };

    let len = nzn + 2;
    let mut elrp = vec![0.0_f64; len];
    let mut elrm = vec![0.0_f64; len];
    let mut elr = vec![0.0_f64; len];
    let mut eltp = vec![0.0_f64; len];
    let mut elt = vec![0.0_f64; len];
    let mut elz0 = vec![0.0_f64; len];
    let mut elzp = vec![0.0_f64; len];

    let (mut dlr, mut dlrp, mut dlrm, mut dlt, mut dltp) =
        (0.0, 0.0, 0.0, 0.0, 0.0);
    let (
        mut dfcx0,
        mut dfcxm,
        mut dfcxp,
        mut dfcy0,
        mut dfcyp,
        mut dfcz0,
        mut dfczp,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    for i in 1..=nzn {
        let (dlmr, dlmrp, dlmrm, dlmt, dlmtp) = (dlr, dlrp, dlrm, dlt, dltp);
        let (dfcmx0, dfcmxm, dfcmxp, dfcmy0, dfcmyp, dfcmz0, dfcmzp) =
            (dfcx0, dfcxm, dfcxp, dfcy0, dfcyp, dfcz0, dfczp);

        if i < nzn {
            let w_00 = grid.t[i].powi(4);
            let w_out = grid.t[i + 1].powi(4);
            let bw = (w_out / w_00).ln();
            let bk = (mc.k[i + 1] / mc.k[i]).ln();
            let t1 = -dc.cl * grid.r[i].powi(4) / grid.dm_bar[i];
            let t2 = (w_out / mc.k[i + 1] - w_00 / mc.k[i]) / (1.0 - bk / bw);
            let t3 = t1 / (bw - bk);
            let dlk = (t3 / mc.k[i]) * (w_00 * bw / mc.k[i] - t2);
            let dlkp = -(t3 / mc.k[i + 1]) * (w_out * bw / mc.k[i + 1] - t2);
            dlrp = dlkp * mc.dk_dv_00[i + 1] * mc.dvr[i + 1];
            dlrm = dlk * mc.dk_dv_00[i] * mc.dvrm[i];
            dlr = 4.0 * t1 * t2 / grid.r[i]
                + dlk * mc.dk_dv_00[i] * mc.dvr[i]
                + dlkp * mc.dk_dv_00[i + 1] * mc.dvrm[i + 1];
            dltp = 4.0
                * (t3 / grid.t[i + 1])
                * (w_out * bw / mc.k[i + 1] - t2 * bk / bw)
                + dlkp * mc.dk_dt_00[i + 1];
            dlt =
                -4.0 * (t3 / grid.t[i]) * (w_00 * bw / mc.k[i] - t2 * bk / bw)
                    + dlk * mc.dk_dt_00[i];
            dfcx0 = tb.dlcx0[i];
            dfcxm = tb.dlcxm[i];
            dfcxp = tb.dlcxp[i];
            dfcy0 = tb.dlcy0[i];
            dfcyp = tb.dlcyp[i];
            dfcz0 = tb.dlcz0[i];
            dfczp = tb.dlczp[i];
        } else {
            dlt = 4.0 * l0 / grid.t[i];
            dlr = 2.0 * l0 / grid.r[i];
            (dlrm, dlrp, dltp) = (0.0, 0.0, 0.0);
            (dfcx0, dfcxm, dfcxp, dfcy0, dfcyp, dfcz0, dfczp) =
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }

        elrp[i] = dlrp + dfcxp + tb.dltxp[i];
        elrm[i] = dlrm + dfcxm + tb.dltxm[i];
        elr[i] = dlr + dfcx0 + tb.dltx0[i];
        eltp[i] = dltp + dfcyp + tb.dltyp[i];
        elt[i] = dlt + dfcy0 + tb.dlty0[i];
        elz0[i] = dfcz0 + tb.dltz0[i];
        elzp[i] = dfczp + tb.dltzp[i];

        let t3_e = (dc.p4 / grid.dm[i]) * (mc.p[i] + mc.dev[i]);
        let t4_e = 1.0 / grid.dm[i];
        let ex20 = -t4_e * (-dlmrm) - tb.dc_dr_in2[i] - t4_e * (-dfcmxm);
        let ex10 =
            -t4_e * (dlrm - dlmr) - tb.dc_dr_in[i] - t4_e * (dfcxm - dfcmx0);
        let ex00 =
            -t4_e * (dlr - dlmrp) - tb.dc_dr_00[i] - t4_e * (dfcx0 - dfcmxp);
        let ex01 = -t4_e * dlrp - tb.dc_dr_out[i] - t4_e * dfcxp;
        let ey10 = -t4_e * (-dlmt) - tb.dc_dt_in[i] - t4_e * (-dfcmy0);
        let ey00 =
            -t4_e * (dlt - dlmtp) - tb.dc_dt_00[i] - t4_e * (dfcy0 - dfcmyp);
        let ey01 = -t4_e * dltp - tb.dc_dt_out[i] - t4_e * dfcyp;
        let ez10 = -t4_e * (-dfcmz0);
        let ez00 = -tb.dc_dw_00[i] - t4_e * (dfcz0 - dfcmzp);
        let ez01 = -t4_e * dfczp;
        let r_in2 = if i == 1 {
            model.r_center.powi(2)
        } else {
            grid.r[i - 1].powi(2)
        };
        let eu10 = t3_e * r_in2;
        let eu00 = -t3_e * grid.r[i].powi(2);

        let t3_c = (dc.p4 / grid.dm[i]) * tb.pturb[i];
        let (cx20, cx10, cx00, cy10, cy00, cz10, cz00) = if i == 1 {
            (
                tb.dc_dr_in2[i],
                tb.dc_dr_in[i] - t4_e * tb.dltxm[i],
                tb.dc_dr_00[i] - t4_e * tb.dltx0[i],
                tb.dc_dt_in[i],
                tb.dc_dt_00[i] - t4_e * tb.dlty0[i],
                0.0,
                tb.dc_dw_00[i] - t4_e * tb.dltz0[i],
            )
        } else {
            (
                tb.dc_dr_in2[i] - t4_e * (-tb.dltxm[i - 1]),
                tb.dc_dr_in[i] - t4_e * (tb.dltxm[i] - tb.dltx0[i - 1]),
                tb.dc_dr_00[i] - t4_e * (tb.dltx0[i] - tb.dltxp[i - 1]),
                tb.dc_dt_in[i] - t4_e * (-tb.dlty0[i - 1]),
                tb.dc_dt_00[i] - t4_e * (tb.dlty0[i] - tb.dltyp[i - 1]),
                -t4_e * (-tb.dltz0[i - 1]),
                tb.dc_dw_00[i] - t4_e * (tb.dltz0[i] - tb.dltzp[i - 1]),
            )
        };
        let cy01 = tb.dc_dt_out[i] - t4_e * tb.dltyp[i];
        let cx01 = tb.dc_dr_out[i] - t4_e * tb.dltxp[i];
        let cz01 = -t4_e * tb.dltzp[i];
        let cu10 = t3_c * r_in2;
        let cu00 = -t3_c * grid.r[i].powi(2);

        let t1_m = dc.p4 * grid.r[i].powi(2) / grid.dm_bar[i];
        let t4_m = if dc.alfam >= 0.0 {
            dc.p4 / (grid.dm_bar[i] * grid.r[i])
        } else {
            -t1_m
        };
        let mu10 = t4_m * (-tb.evuum[i]);
        let mz00 = -t1_m * (-tb.dptrb_dw_00[i]);
        let mx10 = -t1_m * (-mc.dp_dr_in[i] - tb.dptrb_dr_in[i]);
        let my00 = -t1_m * (-mc.dp_dt_00[i]);
        let (mx00, mx01, my01, mu00, mu01, mz01) = if i < nzn {
            (
                4.0 * dc.g * grid.m[i] / grid.r[i].powi(3)
                    - t1_m
                        * (mc.dp_dr_in[i + 1] - mc.dp_dr_00[i]
                            + tb.dptrb_dr_in[i + 1]
                            - tb.dptrb_dr_00[i]),
                -t1_m * (mc.dp_dr_00[i + 1] + tb.dptrb_dr_00[i + 1]),
                -t1_m * mc.dp_dt_00[i + 1],
                t4_m * (tb.evuum[i + 1] - tb.evuu0[i]),
                t4_m * tb.evuu0[i + 1],
                -t1_m * tb.dptrb_dw_00[i + 1],
            )
        } else {
            (
                4.0 * dc.g * grid.m[i] / grid.r[i].powi(3)
                    - t1_m * (-mc.dp_dr_00[i]),
                0.0,
                0.0,
                t4_m * (-tb.evuu0[i]),
                0.0,
                0.0,
            )
        };

        let ig = (4 * i - 3) as isize;
        let ir = (4 * i - 2) as isize;
        let ie = (4 * i - 1) as isize;
        let ic = (4 * i) as isize;
        let inv_cv = 1.0 / mc.de_dt_00[i];

        set_l(&mut lll, ig, ig + 1, 1.0);
        set_l(&mut lll, ir, ir - 4, mu10);
        set_l(&mut lll, ir, ir, mu00);
        set_l(&mut lll, ir, ir + 4, mu01);
        set_l(&mut lll, ir, ir - 5, mx10);
        set_l(&mut lll, ir, ir - 1, mx00);
        set_l(&mut lll, ir, ir + 3, mx01);
        set_l(&mut lll, ir, ir + 1, my00);
        set_l(&mut lll, ir, ir + 5, my01);
        set_l(&mut lll, ir, ir + 2, mz00);
        set_l(&mut lll, ir, ir + 6, mz01);

        set_l(&mut lll, ie, ie - 5, eu10 * inv_cv);
        set_l(&mut lll, ie, ie - 1, eu00 * inv_cv);
        set_l(&mut lll, ie, ie - 10, ex20 * inv_cv);
        set_l(&mut lll, ie, ie - 6, ex10 * inv_cv);
        set_l(&mut lll, ie, ie - 2, ex00 * inv_cv);
        set_l(&mut lll, ie, ie + 2, ex01 * inv_cv);
        set_l(&mut lll, ie, ie - 4, ey10 * inv_cv);
        set_l(&mut lll, ie, ie, ey00 * inv_cv);
        set_l(&mut lll, ie, ie + 4, ey01 * inv_cv);
        set_l(&mut lll, ie, ie - 3, ez10 * inv_cv);
        set_l(&mut lll, ie, ie + 1, ez00 * inv_cv);
        set_l(&mut lll, ie, ie + 5, ez01 * inv_cv);

        set_l(&mut lll, ic, ic - 11, cx20);
        set_l(&mut lll, ic, ic - 7, cx10);
        set_l(&mut lll, ic, ic - 3, cx00);
        set_l(&mut lll, ic, ic + 1, cx01);
        set_l(&mut lll, ic, ic - 6, cu10);
        set_l(&mut lll, ic, ic - 2, cu00);
        set_l(&mut lll, ic, ic - 5, cy10);
        set_l(&mut lll, ic, ic - 1, cy00);
        set_l(&mut lll, ic, ic + 3, cy01);
        set_l(&mut lll, ic, ic - 4, cz10);
        set_l(&mut lll, ic, ic, cz00);
        set_l(&mut lll, ic, ic + 4, cz01);
    }

    let eig = dgeev_right(nzn4, &lll).map_err(|e| {
        StarError::ConvergenceFailure(format!("LINA dgeev_right failed: {e}"))
    })?;

    let mut order: Vec<usize> = (0..nzn4).collect();
    order.sort_by(|&a, &b| eig.wi[a].total_cmp(&eig.wi[b]));

    let mut mini = 5.0_f64;
    let mut imi_opt = None;
    for (idx, &orig) in order.iter().enumerate() {
        let wi_v = eig.wi[orig];
        let wr_v = eig.wr[orig];
        if wi_v > 1.0e-9 && wi_v < mini && dc.p4 * wr_v / wi_v > -3.0 {
            mini = wi_v;
            imi_opt = Some(idx);
        }
    }
    let mut imi = imi_opt.ok_or_else(|| {
        StarError::ConvergenceFailure(String::from(
            "LINA failed to locate positive acoustic eigenmode",
        ))
    })?;

    let get_cpx = |row_1b: usize, col_0b: usize| -> Cpx {
        let r0 = row_1b - 1;
        Cpx::new(eig.vr[r0 * nzn4 + col_0b], eig.vr[r0 * nzn4 + col_0b + 1])
    };

    let mut periods_sec = Vec::with_capacity(nmodes);
    let mut growth_rates = Vec::with_capacity(nmodes);
    let mut q_days = Vec::with_capacity(nmodes);
    let mut kinetic_energies = Vec::with_capacity(nmodes);
    let mut vel_modes = Vec::with_capacity(nmodes);

    for j in 0..nmodes {
        while imi + j < nzn4 {
            let orig = order[imi + j];
            if dc.p4 * eig.wr[orig] / eig.wi[orig] < -5.0 {
                imi += 1;
            } else {
                break;
            }
        }
        if imi + j >= nzn4 || order[imi + j] + 1 >= nzn4 {
            return Err(StarError::ConvergenceFailure(format!(
                "LINA exhausted eigenvalues while selecting mode {j}"
            )));
        }
        let col = order[imi + j];
        let omeg = eig.wi[col];
        let pers = 2.0 * PI / omeg;
        let eto = dc.p4 * eig.wr[col] / omeg;
        let q_val = pers
            * ((grid.m[nzn] / dc.sunm) * (dc.sunr / grid.r[nzn]).powi(3))
                .sqrt()
            / 86_400.0;

        let vtts = get_cpx(4 * nzn - 3, col);
        let vrrs = vtts.abs();
        let scale = Cpx::new(grid.r[nzn], 0.0).div(vtts);

        let mut ek = 0.0_f64;
        let mut vel = vec![0.0_f64; len];
        let mut vrr = vec![Cpx::default(); len];
        let mut vrt = vec![Cpx::default(); len];
        let mut vrc = vec![Cpx::default(); len];
        let mut vru = vec![Cpx::default(); len];
        let mut qwk = vec![0.0_f64; len];
        let mut qwkev = vec![0.0_f64; len];
        let mut qwkpt = vec![0.0_f64; len];
        let mut normc = 1.0e-99_f64;

        for i in 1..=nzn {
            vrr[i] = get_cpx(4 * i - 3, col);
            vru[i] = get_cpx(4 * i - 2, col);
            vrt[i] = get_cpx(4 * i - 1, col);
            vrc[i] = get_cpx(4 * i, col);
            normc = normc.max(vrc[i].abs());

            let dvrr = vrr[i].mul(scale).scale(1.0 / grid.r[i]);
            let phr = dvrr.arg();
            ek +=
                0.5 * grid.dm_bar[i] * (omeg * vrr[i].mul(scale).abs()).powi(2);

            let dv_0 = if i == 1 {
                vrr[i].scale(mc.dvr[i]).mul(scale)
            } else {
                vrr[i]
                    .scale(mc.dvr[i])
                    .add(vrr[i - 1].scale(mc.dvrm[i]))
                    .mul(scale)
            };
            let dp_0 = dv_0
                .scale(mc.dpv[i])
                .add(vrt[i].mul(scale).scale(mc.dp_dt_00[i]));
            let mut dpev = vru[i].scale(tb.evuu0[i]);
            if i >= 2 {
                dpev = dpev.add(vru[i - 1].scale(tb.evuum[i]));
            }
            dpev = dpev.mul(scale);

            let mut dptrb = vrr[i]
                .scale(tb.dptrb_dr_00[i])
                .add(vrc[i].scale(tb.dptrb_dw_00[i]));
            if i >= 2 {
                dptrb = dptrb.add(vrr[i - 1].scale(tb.dptrb_dr_in[i]));
            }
            dptrb = dptrb.mul(scale);

            qwk[i] = -PI * grid.dm[i] * dp_0.conj_mul_imag(dv_0);
            qwkpt[i] = -PI * grid.dm[i] * dptrb.conj_mul_imag(dv_0);
            if dc.alfam < 0.0 {
                qwkev[i] = -PI * grid.dm[i] * dpev.conj_mul_imag(dv_0);
            } else if dc.alfam > 0.0 {
                let strain = dv_0.scale(1.0 / grid.r[i].powi(3)).add(
                    vrr[i]
                        .mul(scale)
                        .scale(-3.0 * grid.vol[i] / grid.r[i].powi(4)),
                );
                qwkev[i] = PI * grid.dm[i] * dpev.conj_mul_imag(strain);
            }

            let mut v_i = vrr[i].abs() / vrrs;
            if phr.abs() > 1.57 {
                v_i = -v_i;
            }
            vel[i] = v_i;
        }

        if let Some(dir) = log_dir {
            write_lina_mode_logs(
                dir,
                j + 1,
                nzn,
                l0,
                ek,
                normc,
                scale,
                &grid,
                &elr,
                &elrm,
                &elrp,
                &elt,
                &eltp,
                &elz0,
                &elzp,
                &vrr,
                &vrt,
                &vrc,
                &qwk,
                &qwkev,
                &qwkpt,
            )?;
        }

        periods_sec.push(pers);
        growth_rates.push(eto);
        q_days.push(q_val);
        kinetic_energies.push(ek);
        vel_modes.push(vel);
    }

    if let Some(dir) = log_dir {
        write_lina_summary(dir, &periods_sec, &growth_rates)?;
    }

    Ok(RspLinaOutput {
        periods_sec,
        growth_rates,
        q_days,
        kinetic_energies,
        vel_modes,
    })
}
