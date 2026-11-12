// SPDX-License-Identifier: LGPL-3.0-or-later
//! Static Kuhfuss/Smolec envelope relaxation (`RELAX_ENV` in
//! `star/private/rsp_relax_env.f90:127-1333`).

use flexmesa_const::{CRAD, PI};
use flexmesa_mtx::{dgbtrf, dgbtrs};

use crate::controls::StarError;
use crate::model::StarModel;
use crate::rsp_def::RspDerivedConsts;
use crate::rsp_eos_kap::RspChemCache;
use crate::rsp_grid::{
    eop, mesa_eos_kap, MesaRanmar, RspEnvGrid, RspTotalEosKap,
};
use crate::rsp_turb_partials::RspTurbPartials;
use crate::rsp_zone_micro::{LcFilterMode, RspZoneMicro};

/// Relaxes the static `RSP` envelope via damped $3N \times 3N$ banded Newton
/// iteration, `ALFAP`/`ALFAT` 20-step continuation, outer-mass `dmN` secant
/// root-finding, and `NEGFLU = 0 -> 1 -> 2` negative-flux continuation
/// (`RELAX_ENV` in `star/private/rsp_relax_env.f90:127-1333`).
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn relax_env(
    model: &mut StarModel,
    chem: &RspChemCache,
    dc: &RspDerivedConsts,
    rng: &mut MesaRanmar,
    grid: &mut RspEnvGrid,
    l0: f64,
    th0: f64,
    te: f64,
    nzt: usize,
) -> Result<(), StarError> {
    let nzn = grid.nzn;
    let efl02 = dc.efl0 * dc.efl0;
    let fsub = model.controls.rsp.dq_1_factor;
    let mstar = grid.m[nzn];
    let psurf = if model.controls.rsp.use_prad_for_psurf {
        let t_0 = 0.5_f64.powf(0.25) * te;
        CRAD * t_0.powi(4) / 3.0
    } else {
        0.0
    };

    let n_anchor = nzn - nzt + 1;
    let summ: f64 = (1..=n_anchor).map(|i| grid.dm[i]).sum();

    let aalfa = dc.alfa;
    let alfa = dc.alfa;
    let aalfap = dc.alfap;
    let aalfat = dc.alfat;
    let mut alfap = 0.0_f64;
    let mut alfat = 0.0_f64;
    let (mut icap, mut icat) = (1_usize, 1_usize);
    let (mut ndivap, mut ndivat) = (20_usize, 20_usize);
    let mut negflu = 0_i32;
    let mut ending = false;

    let precr = 1.0e-10_f64;
    let dxh = 0.01_f64;
    let prec1 = model.controls.rsp.relax_dm_tolerance;
    let max_dmn_cnt = model.controls.rsp.relax_max_tries;
    let mut dmn_cnt = 0_usize;
    let mut ddt = -grid.dm[nzn] / 1000.0;
    let mut dmnl = 0.0_f64;
    let mut tnl = 0.0_f64;
    let mut iop = 0_usize;

    let mut mc = RspZoneMicro::new(nzn);
    let mut tb = RspTurbPartials::new(nzn);
    let n_eq = 3 * nzn;
    let ldab = 16_usize;
    let mut hd = vec![[0.0_f64; 12]; n_eq + 2];
    let mut hr = vec![0.0_f64; n_eq + 1];
    let mut ab = vec![0.0_f64; ldab * n_eq];
    let mut ipiv = vec![0_usize; n_eq];

    loop {
        if iop > 0 {
            let mut dmn = grid.dm[nzn - 1];
            let tt = grid.t[n_anchor] - th0;
            if iop > 1 {
                ddt = tt * (dmn - dmnl) / (tt - tnl);
                if (ddt / dmn).abs() > 0.1 {
                    ddt = 0.1 * dmn * ddt.signum();
                }

                let mut advanced_stage = false;
                if model.controls.rsp.relax_alfap_before_alfat {
                    if (ddt / dmn).abs() < prec1 && alfap != aalfap {
                        alfap = aalfap * (icap as f64) / 20.0;
                        icap += 1;
                        ddt = -grid.dm[nzn] / 100_000.0;
                        advanced_stage = true;
                    } else {
                        if (ddt / dmn).abs() < prec1
                            && icap == 21
                            && ndivap == 20
                        {
                            dmn_cnt = 0;
                            ndivap = 0;
                        }
                        if (ddt / dmn).abs() < prec1 && alfat != aalfat {
                            alfat = aalfat * (icat as f64) / 20.0;
                            icat += 1;
                            ddt = -grid.dm[nzn] / 100_000.0;
                            advanced_stage = true;
                        } else if (ddt / dmn).abs() < prec1
                            && icat == 21
                            && ndivat == 20
                        {
                            dmn_cnt = 0;
                            ndivat = 0;
                        }
                    }
                } else if (ddt / dmn).abs() < prec1 && alfat != aalfat {
                    alfat = aalfat * (icat as f64) / 20.0;
                    icat += 1;
                    ddt = -grid.dm[nzn] / 100_000.0;
                    advanced_stage = true;
                } else {
                    if (ddt / dmn).abs() < prec1 && icat == 21 && ndivat == 20 {
                        dmn_cnt = 0;
                        ndivat = 0;
                    }
                    if (ddt / dmn).abs() < prec1 && alfap != aalfap {
                        alfap = aalfap * (icap as f64) / 20.0;
                        icap += 1;
                        ddt = -grid.dm[nzn] / 100_000.0;
                        advanced_stage = true;
                    } else if (ddt / dmn).abs() < prec1
                        && icap == 21
                        && ndivap == 20
                    {
                        dmn_cnt = 0;
                        ndivap = 0;
                    }
                }

                if !advanced_stage && (ddt / dmn).abs() < prec1 {
                    if negflu >= 1 {
                        if negflu == 2 {
                            ending = true;
                        } else {
                            ddt = -grid.dm[nzn] / 100_000.0;
                            negflu = 2;
                            advanced_stage = true;
                        }
                    } else {
                        ddt = -grid.dm[nzn] / 100_000.0;
                        negflu = 1;
                        advanced_stage = true;
                    }
                }

                if !ending
                    && !advanced_stage
                    && dmn_cnt >= max_dmn_cnt
                    && aalfa == alfa
                {
                    return Err(StarError::ConvergenceFailure(format!(
                        "RELAX_ENV exceeded max tries ({max_dmn_cnt}) for outer dm"
                    )));
                }
            }

            if !ending {
                tnl = tt;
                dmnl = dmn;
                dmn_cnt += 1;
                if model.controls.rsp.relax_adjust_inner_mass_distribution {
                    dmn -= ddt;
                    let aone = dmn;
                    let mut haha = 1.01_f64;
                    let mut converged_h = false;
                    for _ in 1..=100 {
                        let arg = 1.0 - (summ / aone) * (1.0 - haha);
                        let next_h =
                            10.0_f64.powf(arg.log10() / (n_anchor as f64));
                        if (haha - next_h).abs() < 1.0e-10 {
                            haha = next_h;
                            converged_h = true;
                            break;
                        }
                        haha = next_h;
                    }
                    if !converged_h {
                        return Err(StarError::ConvergenceFailure(
                            String::from("RELAX_ENV failed to converge HAHA"),
                        ));
                    }
                    for i in (1..=nzn).rev() {
                        if i == nzn {
                            grid.m[i] = mstar;
                            grid.dm[i] = dmn * fsub;
                            grid.dm_bar[i] = 0.5 * grid.dm[i];
                        } else {
                            if i >= n_anchor {
                                grid.dm[i] = dmn;
                            } else {
                                grid.dm[i] =
                                    aone * haha.powi((n_anchor - i) as i32);
                            }
                            grid.dm_bar[i] =
                                0.5 * (grid.dm[i] + grid.dm[i + 1]);
                            grid.m[i] = grid.m[i + 1] - grid.dm[i + 1];
                        }
                    }
                }
            }
        }

        let mut inner_converged = false;
        for iter_ii in 1..=8000 {
            for i in 2..=nzn {
                let t1 = dc.p43 / grid.dm[i];
                grid.vol[i] = t1 * (grid.r[i].powi(3) - grid.r[i - 1].powi(3));
                mc.dvrm[i] = -3.0 * t1 * grid.r[i - 1].powi(2);
                mc.dvr[i] = 3.0 * t1 * grid.r[i].powi(2);
                if i == nzn {
                    grid.t[nzn] = (0.5 * l0
                        / (4.0 * PI * dc.sig * grid.r[nzn].powi(2)))
                    .powf(0.25);
                }
                let out =
                    mesa_eos_kap(model, chem, 0, nzn, grid.t[i], grid.vol[i])?;
                store_micro_cell(&mut mc, i, &out);
                mc.update_radial_chain(i);
            }

            mc.p[1] = mc.p[2]
                + dc.g * grid.m[1] * grid.dm_bar[1]
                    / (dc.p4 * grid.r[1].powi(4));
            let (vol1, _) = eop(model, chem, nzn, grid.t[1], mc.p[1])?;
            grid.vol[1] = vol1;
            model.r_center = (grid.r[1].powi(3)
                - 3.0 * grid.vol[1] * grid.dm[1] / dc.p4)
                .cbrt();
            let t1_1 = dc.p43 / grid.dm[1];
            mc.dvrm[1] = -3.0 * t1_1 * model.r_center.powi(2);
            mc.dvr[1] = 3.0 * t1_1 * grid.r[1].powi(2);
            let out1 =
                mesa_eos_kap(model, chem, 0, nzn, grid.t[1], grid.vol[1])?;
            store_micro_cell(&mut mc, 1, &out1);
            mc.update_radial_chain(1);

            tb.evaluate(
                dc,
                alfa,
                alfap,
                alfat,
                efl02,
                grid,
                &mc,
                LcFilterMode::Relax { negflu, iter_ii },
            );

            for row in hd.iter_mut().take(n_eq + 1) {
                row.fill(0.0);
            }
            hr.fill(0.0);

            for i in 1..=nzn {
                let iw = 1 + 3 * (i - 1);
                let ic = 2 + 3 * (i - 1);
                let ir = 3 + 3 * (i - 1);
                let (lr_i, dlt, dltp, dlrm, dlr, dlrp) = if i < nzn {
                    let w_00 = grid.t[i].powi(4);
                    let w_out = grid.t[i + 1].powi(4);
                    let bw = (w_out / w_00).ln();
                    let bk = (mc.k[i + 1] / mc.k[i]).ln();
                    let t1 = -dc.cl * grid.r[i].powi(4) / grid.dm_bar[i];
                    let t2 = (w_out / mc.k[i + 1] - w_00 / mc.k[i])
                        / (1.0 - bk / bw);
                    let lr_val = t1 * t2;
                    let t3 = t1 / (bw - bk);
                    let dlk = (t3 / mc.k[i]) * (w_00 * bw / mc.k[i] - t2);
                    let dlkp =
                        -(t3 / mc.k[i + 1]) * (w_out * bw / mc.k[i + 1] - t2);
                    (
                        lr_val,
                        -4.0 * (t3 / grid.t[i])
                            * (w_00 * bw / mc.k[i] - t2 * bk / bw)
                            + dlk * mc.dk_dt_00[i],
                        4.0 * (t3 / grid.t[i + 1])
                            * (w_out * bw / mc.k[i + 1] - t2 * bk / bw)
                            + dlkp * mc.dk_dt_00[i + 1],
                        dlk * mc.dk_dv_00[i] * mc.dvrm[i],
                        4.0 * lr_val / grid.r[i]
                            + dlk * mc.dk_dv_00[i] * mc.dvr[i]
                            + dlkp * mc.dk_dv_00[i + 1] * mc.dvrm[i + 1],
                        dlkp * mc.dk_dv_00[i + 1] * mc.dvr[i + 1],
                    )
                } else {
                    (
                        l0,
                        4.0 * l0 / grid.t[i],
                        0.0,
                        0.0,
                        2.0 * l0 / grid.r[i],
                        0.0,
                    )
                };
                hr[iw] = -(lr_i / l0 + tb.lc[i] / l0 + tb.lt[i] / l0 - 1.0);
                hd[iw][6] = (dlt + tb.dlcy0[i] + tb.dlty0[i]) / l0;
                hd[iw][9] = (dltp + tb.dlcyp[i] + tb.dltyp[i]) / l0;
                hd[iw][5] = (dlrm + tb.dlcxm[i] + tb.dltxm[i]) / l0;
                hd[iw][8] = (dlr + tb.dlcx0[i] + tb.dltx0[i]) / l0;
                hd[iw][11] = (dlrp + tb.dlcxp[i] + tb.dltxp[i]) / l0;
                hd[iw][7] = (tb.dlcz0[i] + tb.dltz0[i]) / l0;
                hd[iw][10] = (tb.dlczp[i] + tb.dltzp[i]) / l0;

                let t1_m = -dc.p4 * grid.r[i].powi(2) / grid.dm_bar[i];
                hr[ir] = if i == nzn {
                    -t1_m * (psurf - mc.p[i] - tb.pturb[i])
                        + dc.g * grid.m[i] / grid.r[i].powi(2)
                } else {
                    -t1_m
                        * (mc.p[i + 1] - mc.p[i] + tb.pturb[i + 1]
                            - tb.pturb[i])
                        + dc.g * grid.m[i] / grid.r[i].powi(2)
                };
                hd[ir][3] = t1_m * (-mc.dp_dr_in[i] - tb.dptrb_dr_in[i]);
                hd[ir][4] = t1_m * (-mc.dp_dt_00[i]);
                hd[ir][5] = t1_m * (-tb.dptrb_dw_00[i]);
                hd[ir][8] = t1_m * tb.dptrb_dw_00[i + 1];
                if i < nzn {
                    hd[ir][6] = t1_m
                        * (mc.dp_dr_in[i + 1] - mc.dp_dr_00[i]
                            + tb.dptrb_dr_in[i + 1]
                            - tb.dptrb_dr_00[i])
                        + 4.0 * dc.g * grid.m[i] / grid.r[i].powi(3);
                    hd[ir][7] = t1_m * mc.dp_dt_00[i + 1];
                    hd[ir][9] =
                        t1_m * (mc.dp_dr_00[i + 1] + tb.dptrb_dr_00[i + 1]);
                } else {
                    hd[ir][6] = t1_m * (-mc.dp_dr_00[i])
                        + 4.0 * dc.g * grid.m[i] / grid.r[i].powi(3);
                }

                if i <= dc.ibotom || i == nzn || alfa == 0.0 {
                    hd[ic][6] = 1.0;
                    hr[ic] = 0.0;
                } else {
                    let t1_c = -1.0 / grid.dm[i];
                    hr[ic] = -(t1_c * (tb.lt[i] - tb.lt[i - 1]) + tb.source[i]
                        - tb.damp[i]
                        - tb.dampr[i]);
                    hd[ic][3] = t1_c * (-tb.dltz0[i - 1]);
                    hd[ic][6] =
                        tb.dc_dw_00[i] + t1_c * (tb.dltz0[i] - tb.dltzp[i - 1]);
                    hd[ic][9] = t1_c * tb.dltzp[i];
                    hd[ic][1] = tb.dc_dr_in2[i] + t1_c * (-tb.dltxm[i - 1]);
                    hd[ic][4] =
                        tb.dc_dr_in[i] + t1_c * (tb.dltxm[i] - tb.dltx0[i - 1]);
                    hd[ic][7] =
                        tb.dc_dr_00[i] + t1_c * (tb.dltx0[i] - tb.dltxp[i - 1]);
                    hd[ic][10] = tb.dc_dr_out[i] + t1_c * tb.dltxp[i];
                    hd[ic][2] = tb.dc_dt_in[i] + t1_c * (-tb.dlty0[i - 1]);
                    hd[ic][5] =
                        tb.dc_dt_00[i] + t1_c * (tb.dlty0[i] - tb.dltyp[i - 1]);
                    hd[ic][8] = tb.dc_dt_out[i] + t1_c * tb.dltyp[i];
                }

                if i == nzn {
                    hd[iw].fill(0.0);
                    hd[iw][6] = 1.0;
                    hr[iw] = 0.0;
                    hd[ir][4] = 0.0;
                    hd[ic][5] = 0.0;
                }
                if i == nzn - 1 {
                    hd[iw][9] = 0.0;
                    hd[ir][7] = 0.0;
                    hd[ic][8] = 0.0;
                }
            }

            if ending {
                return Ok(());
            }

            ab.fill(0.0);
            for row in 1..=n_eq {
                let col_lo = row.saturating_sub(5).max(1);
                let col_hi = (row + 5).min(n_eq);
                for col in col_lo..=col_hi {
                    let band_row = (10 + row) - col;
                    let hd_idx = (6 + col) - row;
                    ab[band_row * n_eq + (col - 1)] = hd[row][hd_idx];
                }
            }
            let rhs = &mut hr[1..=n_eq];
            dgbtrf(n_eq, 5, 5, &mut ab, ldab, &mut ipiv).map_err(|e| {
                StarError::ConvergenceFailure(format!(
                    "RELAX_ENV dgbtrf failed at iter {iter_ii}: {e}"
                ))
            })?;
            dgbtrs(n_eq, 5, 5, &ab, ldab, &ipiv, rhs).map_err(|e| {
                StarError::ConvergenceFailure(format!(
                    "RELAX_ENV dgbtrs failed at iter {iter_ii}: {e}"
                ))
            })?;

            let mut ezh = 1.0_f64;
            let mut xxc = 0.0_f64;
            for i in 2..=nzn {
                let ir = 3 + 3 * (i - 1) - 1;
                let ic = ir - 1;
                let iw = ir - 2;
                if grid.et[i] > 1.0e-20 * efl02 {
                    xxc = (rhs[ic] / grid.et[i]).abs();
                }
                let xxr = ((rhs[ir - 3] - rhs[ir])
                    / (grid.r[i] - grid.r[i - 1]))
                    / dxh;
                let xxt = (rhs[iw] / grid.t[i]).abs() / dxh;
                ezh = 1.0 / (1.0 / ezh).max(xxr).max(xxt).max(xxc);
            }

            let (mut dxxr, mut dxxt, mut dxxc) = (0.0_f64, 0.0_f64, 0.0_f64);
            for i in 1..=nzn {
                let iw = 3 * (i - 1);
                let ic = iw + 1;
                let ir = iw + 2;
                grid.t[i] += 0.5 * ezh * rhs[iw];
                grid.r[i] += 0.5 * ezh * rhs[ir];
                grid.et[i] += 0.5 * ezh * rhs[ic];
                if grid.et[i] <= 0.0 {
                    grid.et[i] = efl02 * 1.0e-4 * rng.next_f64();
                }
                dxxr = dxxr.max((rhs[ir] / grid.r[i]).abs());
                dxxt = dxxt.max((rhs[iw] / grid.t[i]).abs());
                if grid.et[i] > 1.0e-20 * efl02 {
                    dxxc = dxxc.max((rhs[ic] / grid.et[i]).abs());
                }
            }

            if dxxc.abs() < precr && dxxr.abs() < precr && dxxt.abs() < precr {
                iop += 1;
                inner_converged = true;
                break;
            }
        }

        if !inner_converged {
            return Err(StarError::ConvergenceFailure(String::from(
                "RELAX_ENV failed to converge inner Newton loop in 8000 iterations",
            )));
        }
    }
}

fn store_micro_cell(mc: &mut RspZoneMicro, i: usize, out: &RspTotalEosKap) {
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
}
