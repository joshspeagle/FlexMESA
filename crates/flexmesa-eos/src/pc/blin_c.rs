// SPDX-License-Identifier: GPL-3.0-or-later
//! High-degeneracy regime $\chi \ge 14$ (`BLIN9c`) for Blinnikov et al. (1996)
//! relativistic Fermi-Dirac integrals (`eos/private/pc_eos.f90` lines 2655-2887).

use flexmesa_autodiff::AutoDiffReal2VarOrder1;
use flexmesa_const::PI;

use crate::pc::blin::{Blin9Output, BlinSeries};

/// Strongly degenerate regime $\chi \ge 14$ (`BLIN9c` in `eos/private/pc_eos.f90`
/// lines 2655-2887).
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn blin9c(
    temp: AutoDiffReal2VarOrder1,
    chi: AutoDiffReal2VarOrder1,
) -> Blin9Output {
    const PI26: f64 = PI * PI / 6.0;
    let mut out = Blin9Output::default();

    if (chi * temp).val < 0.1 {
        let sqchi = chi.sqrt();
        let chi2 = chi * chi;
        let chi3 = chi2 * chi;
        let chi4 = chi2 * chi2;
        let temp2 = temp * temp;
        let pi26_2_07 = 0.7 * PI26 * PI26;

        for k in 0..3 {
            let mut w = AutoDiffReal2VarOrder1::zero();
            let mut wdx = AutoDiffReal2VarOrder1::zero();
            let mut wdt = AutoDiffReal2VarOrder1::zero();
            let mut wdxx = AutoDiffReal2VarOrder1::zero();
            let mut wdtt = AutoDiffReal2VarOrder1::zero();
            let mut wdxt = AutoDiffReal2VarOrder1::zero();
            let mut wdxxx = AutoDiffReal2VarOrder1::zero();
            let mut wdxtt = AutoDiffReal2VarOrder1::zero();
            let mut wdxxt = AutoDiffReal2VarOrder1::zero();
            let mut c = AutoDiffReal2VarOrder1::zero();

            for j in 0..=4 {
                let cnu = (k + j) as f64 + 0.5;
                let chinu = chi.powi((k + j) as i32) * sqchi;
                let f = chinu
                    * (chi / (cnu + 1.0)
                        + chi.recip() * (PI26 * cnu)
                        + chi3.recip()
                            * (pi26_2_07 * cnu * (cnu - 1.0) * (cnu - 2.0)));
                let fdx = chinu
                    * (chi2.recip() * (PI26 * cnu * (cnu - 1.0))
                        + chi4.recip()
                            * (pi26_2_07
                                * cnu
                                * (cnu - 1.0)
                                * (cnu - 2.0)
                                * (cnu - 3.0))
                        + 1.0);
                let fdxx = (chinu / chi * cnu)
                    * (chi2.recip() * (PI26 * (cnu - 1.0) * (cnu - 2.0))
                        + chi4.recip()
                            * (pi26_2_07
                                * (cnu - 1.0)
                                * (cnu - 2.0)
                                * (cnu - 3.0)
                                * (cnu - 4.0))
                        + 1.0);
                let fdxxx = (chinu / chi2 * (cnu * (cnu - 1.0)))
                    * (chi2.recip() * (PI26 * (cnu - 2.0) * (cnu - 3.0))
                        + chi4.recip()
                            * (pi26_2_07
                                * (cnu - 2.0)
                                * (cnu - 3.0)
                                * (cnu - 4.0)
                                * (cnu - 5.0))
                        + 1.0);
                if j == 0 {
                    w = f;
                    wdx = fdx;
                    wdxx = fdxx;
                    wdxxx = fdxxx;
                } else if j == 1 {
                    c = temp * 0.25;
                    w = w + c * f;
                    wdx = wdx + c * fdx;
                    wdxx = wdxx + c * fdxx;
                    wdt = f * 0.25;
                    wdxt = fdx * 0.25;
                    wdtt = AutoDiffReal2VarOrder1::zero();
                    wdxxx = wdxxx + c * fdxxx;
                    wdxxt = fdxx * 0.25;
                    wdxtt = AutoDiffReal2VarOrder1::zero();
                } else {
                    let jf = j as f64;
                    c = -c * temp * ((2.0 * jf - 3.0) / (4.0 * jf));
                    let cj_t = c * jf / temp;
                    let cjj1_t2 = c * (jf * (jf - 1.0)) / temp2;
                    w = w + c * f;
                    wdx = wdx + c * fdx;
                    wdt = wdt + cj_t * f;
                    wdxx = wdxx + c * fdxx;
                    wdtt = wdtt + cjj1_t2 * f;
                    wdxt = wdxt + cj_t * fdx;
                    wdxxx = wdxxx + c * fdxxx;
                    wdxtt = wdxtt + cjj1_t2 * fdx;
                    wdxxt = wdxxt + cj_t * fdxx;
                }
            }

            let s = BlinSeries {
                w,
                wdx,
                wdt,
                wdxx,
                wdtt,
                wdxt,
            };
            match k {
                0 => {
                    out.w0 = s;
                    out.w0xxx = wdxxx;
                    out.w0xtt = wdxtt;
                    out.w0xxt = wdxxt;
                }
                1 => out.w1 = s,
                _ => out.w2 = s,
            }
        }
    } else {
        let d = 1.0 + chi * temp / 2.0;
        let r = (chi * d).sqrt();
        let rx = 0.5 / chi + 0.25 * temp / d;
        let rdx = r * rx;
        let rdt = 0.25 * chi.pow2() / r;
        let rxx = -0.5 / chi.pow2() - 0.125 * (temp / d).pow2();
        let rdxx = rdx * rx + r * rxx;
        let rdtt = -0.25 * rdt * chi / d;
        let rxt = 0.25 / d - 0.125 * chi * temp / (d * d);
        let rdxt = rdt * rx + r * rxt;
        let rxxx = 1.0 / chi.pow3() + 0.125 * (temp / d).pow3();
        let rdxxx = rdxx * rx + 2.0 * rdx * rxx + r * rxxx;
        let rxtt = -0.25 / (d * d) * chi + 0.125 * chi.pow2() * temp / d.pow3();
        let rdxtt = rdtt * rx + 2.0 * rdt * rxt + r * rxtt;
        let rxxt = -rxt * temp / d;
        let rdxxt = rdxt * rx + rdx * rxt + rdt * rxx + r * rxxt;

        let z = AutoDiffReal2VarOrder1::zero();
        let mut am = [z; 3];
        let mut amdx = [z; 3];
        let mut amdt = [z; 3];
        let mut amdxx = [z; 3];
        let mut amdtt = [z; 3];
        let mut amdxt = [z; 3];
        let mut amdxxx = z;
        let mut amdxtt = z;
        let mut amdxxt = z;

        for k in 0..3 {
            let kf = k as f64;
            let dm = kf + 0.5 + (kf + 1.0) * chi * temp / 2.0;
            am[k] = chi.powf(kf) * dm / r;
            let fmx1 = 0.5 * (kf + 1.0) * temp / dm;
            let fmx2 = 0.25 * temp / d;
            let fmx = (kf - 0.5) / chi + fmx1 - fmx2;
            amdx[k] = am[k] * fmx;
            let ckm = 0.5 * (kf + 1.0) / dm;
            let fmt1 = ckm * chi;
            let fmt2 = 0.25 * chi / d;
            let fmt = fmt1 - fmt2;
            amdt[k] = am[k] * fmt;
            let fmxx =
                -(kf - 0.5) / chi.pow2() - fmx1 * fmx1 + 2.0 * fmx2 * fmx2;
            amdxx[k] = amdx[k] * fmx + am[k] * fmxx;
            let fmtt = 2.0 * fmt2 * fmt2 - fmt1 * fmt1;
            amdtt[k] = amdt[k] * fmt + am[k] * fmtt;
            amdxt[k] = amdx[k] * fmt
                + am[k]
                    * (ckm * (1.0 - ckm * chi * temp) - 0.25 / d
                        + 0.125 * chi * temp / (d * d));
            if k == 0 {
                let fmxxx = (2.0 * kf - 1.0) / chi.pow3() + 2.0 * fmx1.pow3()
                    - 8.0 * fmx2.pow3();
                amdxxx = amdxx[k] * fmx + 2.0 * amdx[k] * fmxx + am[k] * fmxxx;
                let fmt1dx = ckm - temp * chi * ckm * ckm;
                let fmt2dx = (0.25 - chi * temp * 0.125 / d) / d;
                let fmxt = fmt1dx - fmt2dx;
                let fmttx = 4.0 * fmt2 * fmt2dx - 2.0 * fmt1 * fmt1dx;
                amdxtt = amdxt[k] * fmt
                    + amdt[k] * fmxt
                    + amdx[k] * fmtt
                    + am[k] * fmttx;
                let fmx1dt = ckm - chi * temp * ckm * ckm;
                let fmx2dt = 0.25 / d * (1.0 - 0.5 * chi * temp / d);
                let fmxxt = 4.0 * fmx2 * fmx2dt - 2.0 * fmx1 * fmx1dt;
                amdxxt = amdxt[k] * fmx
                    + amdx[k] * fmxt
                    + amdt[k] * fmxx
                    + am[k] * fmxxt;
            }
        }

        let sq2t = (2.0 * temp).sqrt();
        let sq2t3 = sq2t * sq2t * sq2t;
        let a = 1.0 + chi * temp + sq2t * r;
        let adx = temp + sq2t * rdx;
        let adt = chi + r / sq2t + sq2t * rdt;
        let adxx = sq2t * rdxx;
        let adtt = -r / sq2t3 + 2.0 / sq2t * rdt + sq2t * rdtt;
        let adxt = 1.0 + rdx / sq2t + sq2t * rdxt;
        let adxtt = -rdx / sq2t3 + 2.0 / sq2t * rdxt + sq2t * rdxtt;
        let adxxt = rdxx / sq2t + sq2t * rdxxt;
        let xt1 = chi + 1.0 / temp;
        let aln = a.log();
        let fj0 = 0.5 * xt1 * r - aln / sq2t3;
        let asq3 = a * sq2t3;
        let asq3dx = adx * sq2t3;
        let fj0dx = 0.5 * (r + xt1 * rdx) - adx / asq3;
        let fj0dt = 0.5 * (xt1 * rdt - r / temp.pow2()) - adt / asq3
            + 0.75 / (temp * temp * sq2t) * aln;
        let fj0dxx =
            rdx + 0.5 * xt1 * rdxx + (adx / a).pow2() / sq2t3 - adxx / asq3;
        let fj0dtt = r / temp.pow3() - rdt / temp.pow2()
            + 0.5 * xt1 * rdtt
            + 3.0 / (asq3 * temp) * adt
            + (adt / a).pow2() / sq2t3
            - adtt / asq3
            - 1.875 / (temp * temp * temp * sq2t) * aln;
        let bxt = 1.5 / temp * adx + adx * adt / a - adxt;
        let bxxt = 1.5 / temp * adxx + (adxx * adt + adx * adxt) / a
            - (adx / a).pow2() * adt
            - adxxt;
        let fj0dxt = 0.5 * (rdt - rdx / temp.pow2() + xt1 * rdxt) + bxt / asq3;
        let fj0xxx = rdxx * 1.5
            + 0.5 * xt1 * rdxxx
            + (2.0 * adx * (adxx / a - (adx / a).pow2()) - sq2t * rdxxx
                + adxx / asq3 * asq3dx)
                / asq3;
        let fj0xtt = rdx / temp.pow3() - rdxt / temp.pow2()
            + 0.5 * (rdtt + xt1 * rdxtt)
            + 3.0 / temp * (adxt - adt / asq3 * asq3dx) / asq3
            + (2.0 * adt * (adxt / a - adt * adx / (a * a)) - adxtt
                + adtt * asq3dx / asq3)
                / asq3
            - 1.875 / (temp * temp * temp * sq2t) * adx / a;
        let fj0xxt = 0.5 * (rdxt - rdxx / temp.pow2() + rdxt + xt1 * rdxxt)
            + (bxxt - bxt * asq3dx / asq3) / asq3;

        out.w0 = BlinSeries {
            w: fj0 + PI26 * am[0],
            wdx: fj0dx + PI26 * amdx[0],
            wdt: fj0dt + PI26 * amdt[0],
            wdxx: fj0dxx + PI26 * amdxx[0],
            wdtt: fj0dtt + PI26 * amdtt[0],
            wdxt: fj0dxt + PI26 * amdxt[0],
        };
        out.w0xxx = fj0xxx + PI26 * amdxxx;
        out.w0xtt = fj0xtt + PI26 * amdxtt;
        out.w0xxt = fj0xxt + PI26 * amdxxt;

        let fj1 = (r * r * r / 1.5 - fj0) / temp;
        let fj1dx = (2.0 * r * r * rdx - fj0dx) / temp;
        let fj1dt = (2.0 * r * r * rdt - fj0dt - fj1) / temp;
        let fj1dxx = (4.0 * r * rdx * rdx + 2.0 * r * r * rdxx - fj0dxx) / temp;
        let fj1dtt =
            (4.0 * r * rdt * rdt + 2.0 * r * r * rdtt - fj0dtt - 2.0 * fj1dt)
                / temp;
        let fj1dxt =
            (4.0 * r * rdx * rdt + 2.0 * r * r * rdxt - fj0dxt - fj1dx) / temp;

        out.w1 = BlinSeries {
            w: fj1 + PI26 * am[1],
            wdx: fj1dx + PI26 * amdx[1],
            wdt: fj1dt + PI26 * amdt[1],
            wdxx: fj1dxx + PI26 * amdxx[1],
            wdtt: fj1dtt + PI26 * amdtt[1],
            wdxt: fj1dxt + PI26 * amdxt[1],
        };

        let fj2 = (0.5 * chi * r * r * r - 1.25 * fj1) / temp;
        let fj2dx =
            (0.5 * r * r * r + 1.5 * chi * r * r * rdx - 1.25 * fj1dx) / temp;
        let fj2dt = (1.5 * chi * r * r * rdt - 1.25 * fj1dt - fj2) / temp;
        let fj2dxx = (3.0 * r * rdx * (r + chi * rdx)
            + 1.5 * chi * r * r * rdxx
            - 1.25 * fj1dxx)
            / temp;
        let fj2dtt = (3.0 * chi * r * (rdt * rdt + 0.5 * r * rdtt)
            - 1.25 * fj1dtt
            - 2.0 * fj2dt)
            / temp;
        let fj2dxt = (1.5 * r * rdt * (r + 2.0 * chi * rdx)
            + 1.5 * chi * r * r * rdxt
            - 1.25 * fj1dxt
            - fj2dx)
            / temp;

        out.w2 = BlinSeries {
            w: fj2 + PI26 * am[2],
            wdx: fj2dx + PI26 * amdx[2],
            wdt: fj2dt + PI26 * amdt[2],
            wdxx: fj2dxx + PI26 * amdxx[2],
            wdtt: fj2dtt + PI26 * amdtt[2],
            wdxt: fj2dxt + PI26 * amdxt[2],
        };
    }
    out
}
