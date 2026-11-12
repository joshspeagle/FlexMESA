// SPDX-License-Identifier: LGPL-3.0-or-later
//! Bader-Deuflhard semi-implicit midpoint extrapolation integrator
//! (`netint`, `stifbs`, `simpr`, `net_pzextr`) ported from
//! `net/private/net_burn.f90` and `net/private/net_burn_support.f90`.

use flexmesa_mtx::{dgetrf, dgetrs};

pub(crate) use crate::rosenbrock_tableau::tableau_for;
use crate::types::NetError;

const QCOL_IMAX: usize = 13;
const KMAXX: usize = 7;
const STIFBS_IMAX: usize = KMAXX + 1;
const NSEQ: [usize; STIFBS_IMAX + 1] = [0, 2, 6, 10, 14, 22, 34, 50, 70];

#[derive(Debug, Clone)]
struct StifbsState {
    a: [f64; STIFBS_IMAX + 1],
    alf: [[f64; KMAXX + 1]; KMAXX + 1],
    epsold: f64,
    xnew: f64,
    scale: f64,
    red: f64,
    first: bool,
    kmax: usize,
    kopt: usize,
    nvold: usize,
    ysav: Vec<f64>,
    yseq: Vec<f64>,
    yerr: Vec<f64>,
    qcol: Vec<[f64; QCOL_IMAX + 1]>,
    ipiv: Vec<usize>,
    del: Vec<f64>,
    ytemp: Vec<f64>,
    d_pz: Vec<f64>,
}

impl Default for StifbsState {
    fn default() -> Self {
        Self {
            a: [0.0; STIFBS_IMAX + 1],
            alf: [[0.0; KMAXX + 1]; KMAXX + 1],
            epsold: -1.0,
            xnew: -1.0e29,
            scale: 1.0,
            red: 1.0,
            first: true,
            kmax: KMAXX,
            kopt: KMAXX,
            nvold: 0,
            ysav: Vec::new(),
            yseq: Vec::new(),
            yerr: Vec::new(),
            qcol: Vec::new(),
            ipiv: Vec::new(),
            del: Vec::new(),
            ytemp: Vec::new(),
            d_pz: Vec::new(),
        }
    }
}

impl StifbsState {
    fn ensure_scratch(&mut self, nvar: usize) {
        if self.ysav.len() != nvar {
            self.ysav.resize(nvar, 0.0);
            self.yseq.resize(nvar, 0.0);
            self.yerr.resize(nvar, 0.0);
            self.qcol.resize(nvar, [0.0; QCOL_IMAX + 1]);
            self.ipiv.resize(nvar, 0);
            self.del.resize(nvar, 0.0);
            self.ytemp.resize(nvar, 0.0);
            self.d_pz.resize(nvar, 0.0);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn netint<W, D, J>(
    start: f64,
    stptry: f64,
    stpmin: f64,
    max_steps: usize,
    stopp: f64,
    y: &mut [f64],
    eps: f64,
    species: usize,
    nvar: usize,
    odescal: f64,
    dens_dfdy: &mut [f64],
    dmat: &mut [f64],
    work: &mut W,
    derivs: &mut D,
    jakob: &mut J,
) -> Result<(usize, usize, usize), NetError>
where
    D: FnMut(f64, &mut [f64], &mut [f64], &mut W) -> Result<(), NetError>,
    J: FnMut(f64, &mut [f64], &mut [f64], &mut W) -> Result<(), NetError>,
{
    let mut state = StifbsState::default();
    let mut x = start;
    let mut h = stptry.copysign(stopp - start);
    let mut nok = 0_usize;
    let mut nbad = 0_usize;
    let mut dydx = vec![0.0_f64; nvar];
    let mut yscal = vec![0.0_f64; nvar];

    for nstp in 1..=max_steps {
        for val in y.iter_mut().take(species) {
            *val = val.clamp(1.0e-30, 1.0);
        }
        derivs(x, y, &mut dydx, work)?;
        for i in 0..nvar {
            yscal[i] = odescal.max(y[i].abs());
        }
        if (x + h - stopp) * (x + h - start) > 0.0 {
            h = stopp - x;
        }
        let mut hdid = 0.0_f64;
        let mut hnext = 0.0_f64;
        stifbs(
            y, &dydx, nvar, &mut x, h, eps, &yscal, &mut hdid, &mut hnext,
            &mut state, dens_dfdy, dmat, work, derivs, jakob,
        )?;
        if hdid == h {
            nok += 1;
        } else {
            nbad += 1;
        }
        if nstp == max_steps || (x - stopp) * (stopp - start) >= 0.0 {
            return Ok((nok, nbad, nstp));
        }
        h = hnext;
        if h.abs() < stpmin {
            return Err(NetError::IntegrationError(format!(
                "netint failed: |h|={} < stpmin={stpmin}",
                h.abs()
            )));
        }
    }
    Err(NetError::IntegrationError(format!(
        "netint exceeded max_steps={max_steps}"
    )))
}

#[allow(clippy::too_many_arguments)]
fn stifbs<W, D, J>(
    y: &mut [f64],
    dydx: &[f64],
    nvar: usize,
    x: &mut f64,
    htry: f64,
    eps: f64,
    yscal: &[f64],
    hdid: &mut f64,
    hnext: &mut f64,
    state: &mut StifbsState,
    dens_dfdy: &mut [f64],
    dmat: &mut [f64],
    work: &mut W,
    derivs: &mut D,
    jakob: &mut J,
) -> Result<(), NetError>
where
    D: FnMut(f64, &mut [f64], &mut [f64], &mut W) -> Result<(), NetError>,
    J: FnMut(f64, &mut [f64], &mut [f64], &mut W) -> Result<(), NetError>,
{
    const SAFE1: f64 = 0.25;
    const SAFE2: f64 = 0.7;
    const REDMAX: f64 = 1.0e-5;
    const REDMIN: f64 = 0.7;
    const TINY: f64 = 1.0e-30;
    const SCALMX: f64 = 0.1;

    if eps != state.epsold || nvar != state.nvold {
        *hnext = -1.0e29;
        state.xnew = -1.0e29;
        let eps1 = SAFE1 * eps;
        state.a[1] = (NSEQ[1] + 1) as f64;
        for k in 1..=KMAXX {
            state.a[k + 1] = state.a[k] + (NSEQ[k + 1] as f64);
        }
        for iq in 2..=KMAXX {
            for k in 1..iq {
                state.alf[k][iq] = eps1.powf(
                    (state.a[k + 1] - state.a[iq + 1])
                        / ((state.a[iq + 1] - state.a[1] + 1.0)
                            * ((2 * k + 1) as f64)),
                );
            }
        }
        state.epsold = eps;
        state.nvold = nvar;
        state.a[1] += nvar as f64;
        for k in 1..=KMAXX {
            state.a[k + 1] = state.a[k] + (NSEQ[k + 1] as f64);
        }
        let mut kopt_found = KMAXX;
        for ko in 2..KMAXX {
            if state.a[ko + 1] > state.a[ko] * state.alf[ko - 1][ko] {
                kopt_found = ko;
                break;
            }
        }
        state.kopt = kopt_found;
        state.kmax = state.kopt;
    }

    state.ensure_scratch(nvar);
    let mut h = htry;
    state.ysav.copy_from_slice(&y[..nvar]);
    jakob(*x, y, dens_dfdy, work)?;

    if h != *hnext || *x != state.xnew {
        state.first = true;
        state.kopt = state.kmax;
    }
    let mut reduct = false;
    state.yseq.fill(0.0);
    state.yerr.fill(0.0);
    let mut err = [0.0_f64; KMAXX + 1];
    state.qcol.fill([0.0_f64; QCOL_IMAX + 1]);
    let mut x_pzextr = [0.0_f64; QCOL_IMAX + 1];
    let mut km = 1_usize;

    'outer: loop {
        let mut k_final = 1_usize;
        let mut converged = false;
        for k in 1..=state.kmax {
            k_final = k;
            state.xnew = *x + h;
            if state.xnew == *x {
                return Err(NetError::IntegrationError(
                    "stepsize too small in stifbs".to_string(),
                ));
            }
            if simpr(
                &state.ysav,
                dydx,
                nvar,
                *x,
                h,
                NSEQ[k],
                &mut state.yseq,
                dens_dfdy,
                dmat,
                &mut state.ipiv,
                &mut state.del,
                &mut state.ytemp,
                work,
                derivs,
            )
            .is_err()
            {
                h *= 0.1;
                reduct = true;
                continue 'outer;
            }
            let h_sub = h / (NSEQ[k] as f64);
            net_pzextr(
                k,
                h_sub * h_sub,
                &state.yseq,
                y,
                &mut state.yerr,
                nvar,
                &mut state.qcol,
                &mut x_pzextr,
                &mut state.d_pz,
            );
            let mut errmax = TINY;
            if k != 1 {
                for i in 0..nvar {
                    errmax = errmax.max((state.yerr[i] / yscal[i]).abs());
                }
                errmax /= eps;
                km = k - 1;
                err[km] = (errmax / SAFE1).powf(1.0 / ((2 * km + 1) as f64));
            }
            if k != 1 && (k >= state.kopt - 1 || state.first) {
                if errmax < 1.0 {
                    converged = true;
                    break;
                }
                if k == state.kmax || k == state.kopt + 1 {
                    state.red = SAFE2 / err[km];
                    break;
                } else if k == state.kopt {
                    if state.alf[state.kopt - 1][state.kopt] < err[km] {
                        state.red = 1.0 / err[km];
                        break;
                    }
                } else if state.kopt == state.kmax {
                    if state.alf[km][state.kmax - 1] < err[km] {
                        state.red =
                            state.alf[km][state.kmax - 1] * SAFE2 / err[km];
                        break;
                    }
                } else if state.alf[km][state.kopt] < err[km] {
                    state.red = state.alf[km][state.kopt - 1] / err[km];
                    break;
                }
            }
        }

        if converged {
            *x = state.xnew;
            *hdid = h;
            state.first = false;
            let mut wrkmin = 1.0e35_f64;
            for kk in 1..=km {
                let fact = err[kk].max(SCALMX);
                let work_k = fact * state.a[kk + 1];
                if work_k < wrkmin {
                    state.scale = fact;
                    wrkmin = work_k;
                    state.kopt = kk + 1;
                }
            }
            *hnext = h / state.scale;
            if state.kopt >= k_final && state.kopt != state.kmax && !reduct {
                let fact = (state.scale
                    / state.alf[state.kopt - 1][state.kopt])
                    .max(SCALMX);
                if state.a[state.kopt + 1] * fact <= wrkmin {
                    *hnext = h / fact;
                    state.kopt += 1;
                }
            }
            return Ok(());
        }

        state.red = state.red.clamp(REDMAX, REDMIN);
        h *= state.red;
        reduct = true;
    }
}

#[allow(clippy::too_many_arguments)]
fn simpr<W, D>(
    y: &[f64],
    dydx: &[f64],
    nvar: usize,
    xs: f64,
    htot: f64,
    nstep: usize,
    yout: &mut [f64],
    dens_dfdy: &[f64],
    dmat: &mut [f64],
    ipiv: &mut [usize],
    del: &mut [f64],
    ytemp: &mut [f64],
    work: &mut W,
    derivs: &mut D,
) -> Result<(), NetError>
where
    D: FnMut(f64, &mut [f64], &mut [f64], &mut W) -> Result<(), NetError>,
{
    let h = htot / (nstep as f64);
    for i in 0..nvar {
        for j in 0..nvar {
            dmat[i * nvar + j] = -h * dens_dfdy[i * nvar + j];
        }
        dmat[i * nvar + i] += 1.0;
    }
    ipiv[..nvar].fill(0);
    dgetrf(nvar, dmat, ipiv)
        .map_err(|e| NetError::IntegrationError(e.to_string()))?;

    for i in 0..nvar {
        yout[i] = h * dydx[i];
    }
    dgetrs(nvar, dmat, ipiv, yout)
        .map_err(|e| NetError::IntegrationError(e.to_string()))?;

    del[..nvar].copy_from_slice(&yout[..nvar]);
    for i in 0..nvar {
        ytemp[i] = y[i] + del[i];
    }
    let mut x = xs + h;
    derivs(x, ytemp, yout, work)?;

    for _nn in 2..=nstep {
        for i in 0..nvar {
            yout[i] = h.mul_add(yout[i], -del[i]);
        }
        dgetrs(nvar, dmat, ipiv, yout)
            .map_err(|e| NetError::IntegrationError(e.to_string()))?;
        for i in 0..nvar {
            del[i] = 2.0_f64.mul_add(yout[i], del[i]);
            ytemp[i] += del[i];
        }
        x += h;
        derivs(x, ytemp, yout, work)?;
    }

    for i in 0..nvar {
        yout[i] = h.mul_add(yout[i], -del[i]);
    }
    dgetrs(nvar, dmat, ipiv, yout)
        .map_err(|e| NetError::IntegrationError(e.to_string()))?;
    for i in 0..nvar {
        yout[i] += ytemp[i];
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn net_pzextr(
    iest: usize,
    xest: f64,
    yest: &[f64],
    yz: &mut [f64],
    dy: &mut [f64],
    nvar: usize,
    qcol: &mut [[f64; QCOL_IMAX + 1]],
    x: &mut [f64; QCOL_IMAX + 1],
    d: &mut [f64],
) {
    x[iest] = xest;
    dy[..nvar].copy_from_slice(&yest[..nvar]);
    yz[..nvar].copy_from_slice(&yest[..nvar]);
    if iest == 1 {
        for j in 0..nvar {
            qcol[j][1] = yest[j];
        }
    } else {
        d[..nvar].copy_from_slice(&yest[..nvar]);
        for k1 in 1..iest {
            let delta = 1.0 / (x[iest - k1] - xest);
            let f1 = xest * delta;
            let f2 = x[iest - k1] * delta;
            for j in 0..nvar {
                let q = qcol[j][k1];
                qcol[j][k1] = dy[j];
                let d_sub_q = d[j] - q;
                dy[j] = f1 * d_sub_q;
                d[j] = f2 * d_sub_q;
                yz[j] += dy[j];
            }
        }
        for j in 0..nvar {
            qcol[j][iest] = dy[j];
        }
    }
}
