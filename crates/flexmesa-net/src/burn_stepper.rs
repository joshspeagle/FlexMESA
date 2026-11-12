// SPDX-License-Identifier: LGPL-3.0-or-later
//! Bader-Deuflhard semi-implicit midpoint extrapolation integrator
//! (`netint`, `stifbs`, `simpr`, `net_pzextr`) ported from
//! `net/private/net_burn.f90` and `net/private/net_burn_support.f90`.

use flexmesa_mtx::{dgetrf, dgetrs};
use flexmesa_num::RosenbrockVariant;

use crate::types::NetError;

pub(crate) struct RodasTableau {
    pub(crate) ns: usize,
    pub(crate) gamma: f64,
    pub(crate) elo: f64,
    pub(crate) no_aux_in_error: bool,
    pub(crate) newf: [bool; 6],
    pub(crate) alpha: [f64; 6],
    pub(crate) ra: [[f64; 6]; 6],
    pub(crate) rc: [[f64; 6]; 6],
    pub(crate) ros_m: [f64; 6],
    pub(crate) ros_e: [f64; 6],
}

pub(crate) fn tableau_for(variant: RosenbrockVariant) -> RodasTableau {
    let mut tab = RodasTableau {
        ns: 6,
        gamma: 0.25,
        elo: 4.0,
        no_aux_in_error: false,
        newf: [true; 6],
        alpha: [0.0, 0.386, 0.210, 0.630, 1.0, 1.0],
        ra: [[0.0; 6]; 6],
        rc: [[0.0; 6]; 6],
        ros_m: [0.0; 6],
        ros_e: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    };
    match variant {
        RosenbrockVariant::Ros3p => {
            tab.ns = 3;
            tab.gamma = 0.78867513459481286;
            tab.elo = 3.0;
            tab.no_aux_in_error = true;
            tab.newf = [true, true, false, false, false, false];
            tab.alpha = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
            tab.ra[1][0] = 1.26794919243112273;
            tab.ra[2][0] = 1.26794919243112273;
            tab.rc[1][0] = -1.60769515458673630;
            tab.rc[2][0] = -3.46410161513775469;
            tab.rc[2][1] = -1.73205080756887734;
            tab.ros_m[0..3].copy_from_slice(&[
                2.0,
                0.57735026918962578,
                0.42264973081037424,
            ]);
            tab.ros_e = [
                2.0 - 2.11324865405187124,
                0.57735026918962578 - 1.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ];
        }
        RosenbrockVariant::Ros3pl => {
            tab.ns = 4;
            tab.gamma = 0.435866521508459;
            tab.elo = 3.0;
            tab.no_aux_in_error = true;
            tab.newf = [true, true, true, false, false, false];
            tab.alpha = [0.0, 0.5, 1.0, 1.0, 0.0, 0.0];
            tab.ra[1][0] = 1.147140180139521;
            tab.ra[2][0] = 2.463070773030053;
            tab.ra[2][1] = 1.147140180139521;
            tab.ra[3][0] = 2.463070773030053;
            tab.ra[3][1] = 1.147140180139521;
            tab.rc[1][0] = -2.631861185781065;
            tab.rc[2][0] = -1.302364158113095;
            tab.rc[2][1] = 2.769432022251304;
            tab.rc[3][0] = -1.552568958732400;
            tab.rc[3][1] = 2.587743501215153;
            tab.rc[3][2] = -1.416993298352020;
            tab.ros_m[0..4].copy_from_slice(&[
                2.463070773030053,
                1.147140180139521,
                0.0,
                1.0,
            ]);
            tab.ros_e = [
                2.463070773030053 - 2.346947683513665,
                1.147140180139521 - 0.456530569451895,
                -0.056949243945495,
                1.0 - 0.738684936166224,
                0.0,
                0.0,
            ];
        }
        RosenbrockVariant::Rodasp => {
            tab.alpha = [0.0, 0.750, 0.210, 0.630, 1.0, 1.0];
            tab.ra[1][0] = 3.0;
            tab.ra[2][0] = 0.1831036793486759e1;
            tab.ra[2][1] = 0.4955183967433795;
            tab.ra[3][0] = 0.2304376582692669e1;
            tab.ra[3][1] = -0.5249275245743001e-1;
            tab.ra[3][2] = -0.1176798761832782e1;
            tab.ra[4][0] = -0.7170454962423024e1;
            tab.ra[4][1] = -0.4741636671481785e1;
            tab.ra[4][2] = -0.1631002631330971e2;
            tab.ra[4][3] = -0.1062004044111401e1;
            let ra4 = [tab.ra[4][0], tab.ra[4][1], tab.ra[4][2], tab.ra[4][3]];
            tab.ra[5][0..4].copy_from_slice(&ra4);
            tab.ra[5][4] = 1.0;

            tab.rc[1][0] = -12.0;
            tab.rc[2][0] = -0.8791795173947035e1;
            tab.rc[2][1] = -0.2207865586973518e1;
            tab.rc[3][0] = 0.1081793056857153e2;
            tab.rc[3][1] = 0.6780270611428266e1;
            tab.rc[3][2] = 0.1953485944642410e2;
            tab.rc[4][0] = 0.3419095006749676e2;
            tab.rc[4][1] = 0.1549671153725963e2;
            tab.rc[4][2] = 0.5474760875964130e2;
            tab.rc[4][3] = 0.1416005392148534e2;
            tab.rc[5][0] = 0.3462605830930532e2;
            tab.rc[5][1] = 0.1530084976114473e2;
            tab.rc[5][2] = 0.5699955578662667e2;
            tab.rc[5][3] = 0.1840807009793095e2;
            tab.rc[5][4] = -0.5714285714285717e1;

            tab.ros_m[0..4].copy_from_slice(&tab.ra[4][0..4]);
            tab.ros_m[4] = 1.0;
            tab.ros_m[5] = 1.0;
        }
        RosenbrockVariant::Rodas4
        | RosenbrockVariant::Rodas3
        | RosenbrockVariant::Ros2
        | RosenbrockVariant::Rose2 => {
            tab.ra[1][0] = 1.544;
            tab.ra[2][0] = 0.9466785280815826;
            tab.ra[2][1] = 0.2557011698983284;
            tab.ra[3][0] = 0.3314825187068521e1;
            tab.ra[3][1] = 0.2896124015972201e1;
            tab.ra[3][2] = 0.9986419139977817;
            tab.ra[4][0] = 0.1221224509226641e1;
            tab.ra[4][1] = 0.6019134481288629e1;
            tab.ra[4][2] = 0.1253708332932087e2;
            tab.ra[4][3] = -0.6878860361058950;
            let ra4 = [tab.ra[4][0], tab.ra[4][1], tab.ra[4][2], tab.ra[4][3]];
            tab.ra[5][0..4].copy_from_slice(&ra4);
            tab.ra[5][4] = 1.0;

            tab.rc[1][0] = -5.6688;
            tab.rc[2][0] = -0.2430093356833875e1;
            tab.rc[2][1] = -0.2063599157091915;
            tab.rc[3][0] = -0.1073529058151375;
            tab.rc[3][1] = -0.9594562251023355e1;
            tab.rc[3][2] = -0.2047028614809616e2;
            tab.rc[4][0] = 0.7496443313967647e1;
            tab.rc[4][1] = -0.1024680431464352e2;
            tab.rc[4][2] = -0.3399990352819905e2;
            tab.rc[4][3] = 0.1170890893206160e2;
            tab.rc[5][0] = 0.8083246795921522e1;
            tab.rc[5][1] = -0.7981132988064893e1;
            tab.rc[5][2] = -0.3152159432874371e2;
            tab.rc[5][3] = 0.1631930543123136e2;
            tab.rc[5][4] = -0.6058818238834054e1;

            tab.ros_m[0..4].copy_from_slice(&tab.ra[4][0..4]);
            tab.ros_m[4] = 1.0;
            tab.ros_m[5] = 1.0;
        }
    }
    tab
}

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

    let mut h = htry;
    let ysav = y.to_vec();
    jakob(*x, y, dens_dfdy, work)?;

    if h != *hnext || *x != state.xnew {
        state.first = true;
        state.kopt = state.kmax;
    }
    let mut reduct = false;
    let mut yseq = vec![0.0_f64; nvar];
    let mut yerr = vec![0.0_f64; nvar];
    let mut err = [0.0_f64; KMAXX + 1];
    let mut qcol = vec![[0.0_f64; QCOL_IMAX + 1]; nvar];
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
                &ysav, dydx, nvar, *x, h, NSEQ[k], &mut yseq, dens_dfdy, dmat,
                work, derivs,
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
                &yseq,
                y,
                &mut yerr,
                nvar,
                &mut qcol,
                &mut x_pzextr,
            );
            let mut errmax = TINY;
            if k != 1 {
                for i in 0..nvar {
                    errmax = errmax.max((yerr[i] / yscal[i]).abs());
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
    let mut ipiv = vec![0_usize; nvar];
    dgetrf(nvar, dmat, &mut ipiv)
        .map_err(|e| NetError::IntegrationError(e.to_string()))?;

    for i in 0..nvar {
        yout[i] = h * dydx[i];
    }
    dgetrs(nvar, dmat, &ipiv, yout)
        .map_err(|e| NetError::IntegrationError(e.to_string()))?;

    let mut del = yout.to_vec();
    let mut ytemp: Vec<f64> =
        y.iter().zip(&del).map(|(yi, di)| yi + di).collect();
    let mut x = xs + h;
    derivs(x, &mut ytemp, yout, work)?;

    for _nn in 2..=nstep {
        for i in 0..nvar {
            yout[i] = h.mul_add(yout[i], -del[i]);
        }
        dgetrs(nvar, dmat, &ipiv, yout)
            .map_err(|e| NetError::IntegrationError(e.to_string()))?;
        for i in 0..nvar {
            del[i] = 2.0_f64.mul_add(yout[i], del[i]);
            ytemp[i] += del[i];
        }
        x += h;
        derivs(x, &mut ytemp, yout, work)?;
    }

    for i in 0..nvar {
        yout[i] = h.mul_add(yout[i], -del[i]);
    }
    dgetrs(nvar, dmat, &ipiv, yout)
        .map_err(|e| NetError::IntegrationError(e.to_string()))?;
    for i in 0..nvar {
        yout[i] += ytemp[i];
    }
    Ok(())
}

fn net_pzextr(
    iest: usize,
    xest: f64,
    yest: &[f64],
    yz: &mut [f64],
    dy: &mut [f64],
    nvar: usize,
    qcol: &mut [[f64; QCOL_IMAX + 1]],
    x: &mut [f64; QCOL_IMAX + 1],
) {
    x[iest] = xest;
    dy[..nvar].copy_from_slice(&yest[..nvar]);
    yz[..nvar].copy_from_slice(&yest[..nvar]);
    if iest == 1 {
        for j in 0..nvar {
            qcol[j][1] = yest[j];
        }
    } else {
        let mut d = yest.to_vec();
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
