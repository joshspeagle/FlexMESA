// SPDX-License-Identifier: LGPL-3.0-or-later
//! Static `RSP` envelope integrator (`STAH`, `ZNVAR`, and `CFLUX` matching
//! `star/private/rsp_build.f90:301-986`).

use flexmesa_const::{CRAD, PI};
use flexmesa_num::{brent_safe_zero, NumError};

use crate::controls::StarError;
use crate::model::StarModel;
use crate::rsp_def::RspDerivedConsts;
use crate::rsp_eos_kap::{get_surf_p_t_kap, RspChemCache};
use crate::rsp_grid::{eop, mesa_eos_kap, RspEnvGrid};

/// Initial outer zone mass $\Delta m_N$ and growth ratio $H$ (`ZNVAR` in `rsp_build.f90:839-919`).
pub fn znvar(
    model: &mut StarModel,
    chem: &RspChemCache,
    dc: &RspDerivedConsts,
    nzn: usize,
    l: f64,
    te: f64,
    mx: f64,
) -> Result<(f64, f64), StarError> {
    let h = 1.02_f64;
    let t0 = 0.5_f64.powf(0.25) * te;
    let r = (l / (4.0 * PI * dc.sig)).sqrt() / (te * te);
    let dtau = 0.003_f64;
    let kap0 = if t0 < 4700.0 {
        1.0e-3
    } else if t0 > 5100.0 {
        1.0e-2 * (1.0 + chem.x)
    } else {
        let a = (t0 - 4700.0) / 400.0;
        a * 1.0e-2 * (1.0 + chem.x) + (1.0 - a) * 1.0e-3
    };
    let g_m_dtau_div_r2 = dc.g * mx * dtau / (r * r);
    let prad = CRAD * t0.powi(4) / 3.0;
    let p_init = g_m_dtau_div_r2 / kap0 + prad;
    let (mut v, _) = eop(model, chem, nzn, t0, p_init)?;
    let mut kap = kap0;
    for _ in 1..=25 {
        let out = mesa_eos_kap(model, chem, 0, nzn, t0, v)?;
        kap = out.op;
        let residual = out.p - (prad + g_m_dtau_div_r2 / kap);
        if residual.abs() < 1.0e-6 * out.p {
            break;
        }
        let d_res_dlnv =
            out.dpv * v + g_m_dtau_div_r2 * (out.opv * v) / (kap * kap);
        v = (v.ln() - residual / d_res_dlnv).exp();
    }
    Ok((h, 4.0 * PI * r * r * dtau / kap))
}

#[derive(Debug, Clone, Copy, Default)]
struct StahStepVars {
    e_0: f64,
    e_1: f64,
    t_0: f64,
    t_1: f64,
    t4_0: f64,
    t4_1: f64,
    v_0: f64,
    v_1: f64,
    p_0: f64,
    p_1: f64,
    qq_0: f64,
    qq_1: f64,
    cp_0: f64,
    cp_1: f64,
    op_0: f64,
    op_1: f64,
    r_1: f64,
    m_0: f64,
    dm_0: f64,
    dm_bar_0: f64,
    hp_0: f64,
    igr_0: f64,
    lc_0: f64,
    lr_0: f64,
    w_0: f64,
}

impl StahStepVars {
    fn cflux(&mut self, dc: &RspDerivedConsts, n: usize) {
        if dc.alfa == 0.0 {
            self.w_0 = 0.0;
            self.lc_0 = 0.0;
            return;
        }
        self.hp_0 = self.r_1 * self.r_1 / (dc.g * self.m_0)
            * 0.5
            * (self.p_0 * self.v_0 + self.p_1 * self.v_1);
        let pom = dc.p4 * self.r_1 * self.r_1 * self.hp_0 * 2.0
            / (self.dm_bar_0 * (self.v_0 + self.v_1));
        self.igr_0 = pom
            * (0.5
                * (self.qq_0 / self.cp_0 + self.qq_1 / self.cp_1)
                * (self.p_1 - self.p_0)
                - (self.t_1.ln() - self.t_0.ln()));
        let ff = (2.0_f64 / 3.0).sqrt()
            * 0.5
            * ((self.e_0 + self.p_0 * self.v_0) / self.t_0
                + (self.e_1 + self.p_1 * self.v_1) / self.t_1);
        let gg =
            dc.alfas * dc.alfa * 0.5 * (self.cp_0 + self.cp_1) * self.igr_0;
        let gpf = gg / ff;
        if n <= dc.ibotom || self.igr_0 <= 0.0 {
            self.lc_0 = 0.0;
            self.w_0 = 0.0;
            return;
        }
        self.w_0 = (dc.alfa / dc.cede
            * ff
            * gpf
            * 0.5
            * (self.t_0 * self.p_0 * self.qq_0 / self.cp_0
                + self.t_1 * self.p_1 * self.qq_1 / self.cp_1))
            .max(0.0)
            .sqrt();
        let pii = ff * self.w_0 * gpf;
        self.lc_0 = dc.p4
            * self.r_1
            * self.r_1
            * 0.5
            * (self.t_0 / self.v_0 + self.t_1 / self.v_1)
            * pii
            * (dc.alfac / dc.alfas);
    }

    fn eval_t_residual(
        &mut self,
        model: &mut StarModel,
        chem: &RspChemCache,
        dc: &RspDerivedConsts,
        nzn: usize,
        n: usize,
        l: f64,
    ) -> Result<f64, StarError> {
        let (v, out) = eop(model, chem, nzn, self.t_0, self.p_0)?;
        self.v_0 = v;
        (self.e_0, self.cp_0, self.qq_0, self.op_0) =
            (out.e, out.cp, out.qq, out.op);
        self.cflux(dc, n);
        let tt = 4.0 * dc.sig * dc.p4 * dc.p4 * self.r_1.powi(4)
            / (3.0 * self.dm_bar_0 * l);
        self.t4_0 = self.t_0.powi(4);
        let dlnt4 = (self.t4_0 / self.t4_1).ln();
        self.lr_0 = if dlnt4 == 0.0 {
            0.0
        } else {
            tt * (self.t4_0 / self.op_0 - self.t4_1 / self.op_1)
                / (1.0 - (self.op_0 / self.op_1).ln() / dlnt4)
                * l
        };
        Ok((self.lr_0 + self.lc_0) / l - 1.0)
    }
}

/// Integrates the static `RSP` envelope (`STAH` in `rsp_build.f90:301-836`)
/// and initializes zone $e_t = w^2$ (`rsp_build.f90:151-167`).
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn stah(
    model: &mut StarModel,
    chem: &RspChemCache,
    dc: &RspDerivedConsts,
    mx: f64,
    l: f64,
    te: f64,
    th0: f64,
    tin: f64,
    nzt: usize,
    mut nzn: usize,
) -> Result<RspEnvGrid, StarError> {
    let fsub = model.controls.rsp.dq_1_factor;
    let th0_tol = model.controls.rsp.t_anchor_tolerance;
    let tin_tol = model.controls.rsp.t_inner_tolerance;
    let max_outer_tries = model.controls.rsp.max_outer_dm_tries;
    let max_inner_tries = model.controls.rsp.max_inner_scale_tries;

    let (mut h, mut dmn) = znvar(model, chem, dc, nzn, l, te, mx)?;
    let mut dmn_cnt = 1_usize;
    let mut h_cnt = 1_usize;
    let mut adjusting_dmn = true;
    let (mut have_dmn_large, mut have_dmn_small) = (false, false);
    let (mut dmn_large, mut dmn_small) = (0.0_f64, 0.0_f64);
    let (mut have_h_large, mut have_h_small) = (false, false);
    let (mut h_large, mut h_small) = (0.0_f64, 0.0_f64);

    let mut grid = RspEnvGrid::new(nzn);
    let mut w_face = vec![0.0_f64; nzn + 2];
    let mut s = StahStepVars::default();
    let mut n: usize;

    'top: loop {
        s.dm_0 = dmn * fsub;
        s.m_0 = mx;
        s.dm_bar_0 = 0.5 * s.dm_0;
        let we = te.powi(4);
        s.t4_0 = 0.5 * we;
        s.t_0 = 0.5_f64.powf(0.25) * te;
        s.r_1 = (l / (dc.p4 * dc.sig * we)).sqrt();
        let psurf = if model.controls.rsp.use_atm_grey_with_kap_for_psurf {
            let (_, ps, _, _) = get_surf_p_t_kap(
                model,
                chem,
                mx,
                s.r_1,
                l,
                model.controls.rsp.tau_surf_for_atm_grey_with_kap,
                1.0e-2,
            )?;
            ps
        } else if model.controls.rsp.use_prad_for_psurf {
            CRAD * s.t_0.powi(4) / 3.0
        } else {
            0.0
        };
        model.aux.rsp.psurf_from_atm = psurf;
        s.p_0 = psurf + dc.g * s.m_0 * s.dm_bar_0 / (dc.p4 * s.r_1.powi(4));
        let (v0, out0) = eop(model, chem, nzn, s.t_0, s.p_0)?;
        s.v_0 = v0;
        (s.e_0, s.cp_0, s.qq_0, s.op_0) = (out0.e, out0.cp, out0.qq, out0.op);
        (s.w_0, s.lc_0, s.lr_0) = (0.0, 0.0, l);
        n = nzn;

        store_stah_zone(&mut grid, &mut w_face, &s, n);

        'zone: loop {
            s.r_1 = (s.r_1.powi(3) - 3.0 * s.v_0 * s.dm_0 / dc.p4)
                .max(0.0)
                .cbrt();
            n -= 1;
            if n == 0 || s.t_0 >= tin {
                h_cnt += 1;
                let hh = if s.t_0 < tin {
                    h_small = h;
                    have_h_small = true;
                    if have_h_large {
                        0.5 * (h_large + h_small)
                    } else {
                        h * 1.05
                    }
                } else {
                    h_large = h;
                    have_h_large = true;
                    if have_h_small {
                        0.5 * (h_large + h_small)
                    } else {
                        h / 1.05
                    }
                };
                let h_bracket_converged = have_h_large
                    && have_h_small
                    && (h_large - h_small).abs() <= 1.0e-13 * h.abs();
                if n == 0
                    && ((grid.t[1] - tin).abs() < tin * tin_tol
                        || h_bracket_converged)
                {
                    let m_center_raw = s.m_0 - s.dm_0;
                    model.mstar = model.controls.rsp_mass * dc.sunm;
                    model.xmstar = model.mstar - m_center_raw;
                    model.m_center = model.mstar - model.xmstar;
                    model.l_center = model.controls.rsp_l * dc.sunl;
                    model.r_center = s.r_1;
                    model.v_center = 0.0;
                    break 'top;
                }
                if h_cnt >= max_inner_tries {
                    return Err(StarError::ConvergenceFailure(format!(
                        "STAH failed to satisfy T_inner ({tin}) in {max_inner_tries} tries"
                    )));
                }
                let n1 = nzn - nzt + 1;
                s.r_1 = grid.r[n1];
                s.v_0 = grid.vol[n1];
                s.p_0 = grid.p[n1];
                s.t_0 = grid.t[n1];
                s.t4_0 = s.t_0.powi(4);
                s.m_0 = grid.m[n1];
                s.dm_0 = grid.dm[n1];
                s.op_0 = grid.k[n1];
                s.hp_0 = grid.hp_face[n1];
                s.igr_0 = grid.y_face[n1];
                s.qq_0 = grid.qqs[n1];
                s.cp_0 = grid.cps[n1];
                s.lc_0 = grid.lc[n1];
                s.e_0 = grid.e[n1];
                n = n1;
                h = hh;
                continue 'zone;
            }

            (s.p_1, s.v_1, s.op_1, s.t4_1) = (s.p_0, s.v_0, s.op_0, s.t4_0);
            s.m_0 -= s.dm_0;
            let dml = s.dm_0;
            (s.cp_1, s.qq_1, s.t_1, s.e_1) =
                (s.cp_0, s.qq_0, s.t4_1.sqrt().sqrt(), s.e_0);
            if n == nzn - 1 {
                s.dm_0 = dmn;
            } else if s.t_1 > th0 {
                s.dm_0 *= h;
            }
            s.dm_bar_0 = 0.5 * (s.dm_0 + dml);
            s.p_0 = s.p_1 + dc.g * s.m_0 * s.dm_bar_0 / (dc.p4 * s.r_1.powi(4));

            if !solve_stah_zone_t(model, chem, dc, &mut s, nzn, n, l)? {
                dmn *= 0.25;
                dmn -= dmn / 1.0e3;
                continue 'top;
            }

            s.t_0 = s.t4_0.sqrt().sqrt();
            let (vv, outv) = eop(model, chem, nzn, s.t_0, s.p_0)?;
            s.v_0 = vv;
            (s.e_0, s.cp_0, s.qq_0, s.op_0) =
                (outv.e, outv.cp, outv.qq, outv.op);
            if n != nzn {
                s.cflux(dc, n);
            }

            if (nzn - n + 1 == nzt || s.t_0 >= th0) && adjusting_dmn {
                if dmn_cnt >= max_outer_tries {
                    return Err(StarError::ConvergenceFailure(format!(
                        "STAH failed to satisfy T_anchor ({th0}) in {max_outer_tries} tries"
                    )));
                }
                dmn_cnt += 1;
                if s.t_0 < th0 {
                    dmn_small = dmn;
                    have_dmn_small = true;
                    dmn = if have_dmn_large {
                        0.5 * (dmn_large + dmn_small)
                    } else {
                        dmn * 2.0
                    };
                } else {
                    dmn_large = dmn;
                    have_dmn_large = true;
                    dmn = if have_dmn_small {
                        0.5 * (dmn_large + dmn_small)
                    } else {
                        dmn * 0.5
                    };
                }
                let dmn_bracket_converged = have_dmn_large
                    && have_dmn_small
                    && (dmn_large - dmn_small).abs() <= 1.0e-13 * dmn.abs();
                if nzn - n + 1 == nzt
                    && ((s.t_0 - th0).abs() < th0_tol * th0
                        || dmn_bracket_converged)
                {
                    adjusting_dmn = false;
                }
                continue 'top;
            }

            store_stah_zone(&mut grid, &mut w_face, &s, n);
        }
    }

    if n != 0 {
        nzn -= n;
        grid.nzn = nzn;
        for i in 1..=nzn {
            grid.r[i] = grid.r[i + n];
            grid.p[i] = grid.p[i + n];
            grid.vol[i] = grid.vol[i + n];
            grid.e[i] = grid.e[i + n];
            grid.t[i] = grid.t[i + n];
            grid.k[i] = grid.k[i + n];
            grid.m[i] = grid.m[i + n];
            grid.dm[i] = grid.dm[i + n];
            grid.dm_bar[i] = grid.dm_bar[i + n];
            grid.hp_face[i] = grid.hp_face[i + n];
            grid.y_face[i] = grid.y_face[i + n];
            grid.cps[i] = grid.cps[i + n];
            grid.qqs[i] = grid.qqs[i + n];
            grid.lc[i] = grid.lc[i + n];
            grid.lr[i] = grid.lr[i + n];
            w_face[i] = w_face[i + n];
        }
    }

    let efl02 = dc.efl0 * dc.efl0;
    grid.et[1] = efl02;
    for i in 2..=nzn {
        let et_i = (w_face[i].powi(2) * grid.dm[i]
            + w_face[i - 1].powi(2) * grid.dm[i - 1])
            / (grid.dm[i] + grid.dm[i - 1]);
        grid.et[i] = et_i.max(efl02);
    }
    Ok(grid)
}

fn store_stah_zone(
    grid: &mut RspEnvGrid,
    w_face: &mut [f64],
    s: &StahStepVars,
    n: usize,
) {
    grid.r[n] = s.r_1;
    grid.p[n] = s.p_0;
    grid.vol[n] = s.v_0;
    grid.e[n] = s.e_0;
    grid.t[n] = s.t_0;
    grid.k[n] = s.op_0;
    grid.m[n] = s.m_0;
    grid.dm[n] = s.dm_0;
    grid.dm_bar[n] = s.dm_bar_0;
    grid.hp_face[n] = s.hp_0;
    grid.y_face[n] = s.igr_0;
    grid.cps[n] = s.cp_0;
    grid.qqs[n] = s.qq_0;
    grid.lc[n] = s.lc_0;
    grid.lr[n] = s.lr_0;
    w_face[n] = s.w_0;
}

fn solve_stah_zone_t(
    model: &mut StarModel,
    chem: &RspChemCache,
    dc: &RspDerivedConsts,
    s: &mut StahStepVars,
    nzn: usize,
    n: usize,
    l: f64,
) -> Result<bool, StarError> {
    let tmax = 0.99 * (3.0 * s.p_0 / CRAD).powf(0.25);
    let lnt_min = s.t_1.ln();
    let lnt_max = tmax.ln();
    let mut got_est = false;
    if lnt_max > lnt_min {
        s.t_0 = lnt_min.exp();
        if let Ok(r_min) = s.eval_t_residual(model, chem, dc, nzn, n, l) {
            s.t_0 = lnt_max.exp();
            if let Ok(r_max) = s.eval_t_residual(model, chem, dc, nzn, n, l) {
                let root = brent_safe_zero(
                    lnt_min,
                    lnt_max,
                    1.0e-14,
                    1.0e-5,
                    1.0e-5,
                    |lnt| {
                        s.t_0 = lnt.exp();
                        s.eval_t_residual(model, chem, dc, nzn, n, l)
                            .map_err(|_| NumError::NonFiniteValue("STAH get_T"))
                    },
                    Some(r_min),
                    Some(r_max),
                    200,
                );
                if let Ok((lnt, _)) = root {
                    s.t_0 = lnt.exp();
                    let _ = s.eval_t_residual(model, chem, dc, nzn, n, l)?;
                    s.t4_0 = s.t_0.powi(4);
                    got_est = true;
                }
            }
        }
    }
    if !got_est {
        s.t4_0 = s.t4_1 * 1.2;
        s.t_0 = s.t4_0.sqrt().sqrt();
        while CRAD * s.t_0.powi(4) / 3.0 >= s.p_0 {
            s.t_0 *= 0.99;
        }
        s.t4_0 = s.t_0.powi(4);
    }

    loop {
        let (v, out) = eop(model, chem, nzn, s.t_0, s.p_0)?;
        s.v_0 = v;
        (s.e_0, s.cp_0, s.qq_0, s.op_0) = (out.e, out.cp, out.qq, out.op);
        s.cflux(dc, n);
        if s.lc_0 < l {
            break;
        }
        s.t_0 -= 10.0 * (s.lc_0 / l);
        if s.t_0 <= 0.0 {
            return Err(StarError::ConvergenceFailure(String::from(
                "STAH T_0 <= 0 in Lc_loop1",
            )));
        }
        s.t4_0 = s.t_0.powi(4);
    }

    let tt =
        4.0 * dc.sig * dc.p4 * dc.p4 * s.r_1.powi(4) / (3.0 * s.dm_bar_0 * l);
    s.lr_0 = tt * (s.t4_0 / s.op_0 - s.t4_1 / s.op_1)
        / (1.0 - (s.op_0 / s.op_1).ln() / (s.t4_0 / s.t4_1).ln())
        * l;
    let mut f1 = (s.lr_0 + s.lc_0) / l - 1.0;
    let mut d = s.t4_0 / 1.0e3;
    for _ in 1..=10_000 {
        while (d / s.t4_0).abs() > 0.5 {
            d = 0.5 * s.t4_0 * d.signum();
        }
        s.t4_0 -= d;
        s.t_0 = s.t4_0.sqrt().sqrt();
        loop {
            let (v, out) = eop(model, chem, nzn, s.t_0, s.p_0)?;
            s.v_0 = v;
            (s.e_0, s.cp_0, s.qq_0, s.op_0) = (out.e, out.cp, out.qq, out.op);
            s.cflux(dc, n);
            if s.lc_0 < l {
                break;
            }
            s.t_0 -= 10.0 * (s.lc_0 / l);
            if s.t_0 <= 0.0 {
                return Err(StarError::ConvergenceFailure(String::from(
                    "STAH T_0 <= 0 in Lc_loop",
                )));
            }
            s.t4_0 = s.t_0.powi(4);
        }
        s.lr_0 = tt * (s.t4_0 / s.op_0 - s.t4_1 / s.op_1)
            / (1.0 - (s.op_0 / s.op_1).ln() / (s.t4_0 / s.t4_1).ln())
            * l;
        let f2 = (s.lr_0 + s.lc_0) / l - 1.0;
        if f2.abs() < 1.0e-10 || f2 == f1 {
            return Ok(true);
        }
        // Matches upstream `rsp_build.f90:772` (`D = F2*(T4_0-T4_0)/(F2-F1)`),
        // preserving exact outer-anchor and inner-scale bisection parity.
        d = 0.0;
        f1 = f2;
    }
    Ok(false)
}
