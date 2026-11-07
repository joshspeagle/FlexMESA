// SPDX-License-Identifier: LGPL-3.0-or-later
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
        let d = chi * temp * 0.5 + 1.0;
        let r = (chi * d).sqrt();
        let rx = chi.recip() * 0.5 + temp * 0.25 / d;
        let rdx = r * rx;
        let rdt = chi.pow2() * 0.25 / r;
        let rxx = -chi.pow2().recip() * 0.5 - (temp / d).pow2() * 0.125;
        let rdxx = rdx * rx + r * rxx;
        let rdtt = -rdt * chi * 0.25 / d;
        let rxt = d.recip() * 0.25 - chi * temp * 0.125 / (d * d);
        let rdxt = rdt * rx + r * rxt;
        let rxxx = chi.pow3().recip() + (temp / d).pow3() * 0.125;
        let rdxxx = rdxx * rx + rdx * rxx * 2.0 + r * rxxx;
        let rxtt = -chi * 0.25 / (d * d) + chi.pow2() * temp * 0.125 / d.pow3();
        let rdxtt = rdtt * rx + rdt * rxt * 2.0 + r * rxtt;
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
            let dm = chi * temp * (0.5 * (kf + 1.0)) + (kf + 0.5);
            am[k] = chi.powi(k as i32) * dm / r;
            let fmx1 = temp * (0.5 * (kf + 1.0)) / dm;
            let fmx2 = temp * 0.25 / d;
            let fmx = (kf - 0.5) / chi + fmx1 - fmx2;
            amdx[k] = am[k] * fmx;
            let ckm = (0.5 * (kf + 1.0)) / dm;
            let fmt1 = ckm * chi;
            let fmt2 = chi * 0.25 / d;
            let fmt = fmt1 - fmt2;
            amdt[k] = am[k] * fmt;
            let fmxx =
                -(kf - 0.5) / chi.pow2() - fmx1 * fmx1 + fmx2 * fmx2 * 2.0;
            amdxx[k] = amdx[k] * fmx + am[k] * fmxx;
            let fmtt = fmt2 * fmt2 * 2.0 - fmt1 * fmt1;
            amdtt[k] = amdt[k] * fmt + am[k] * fmtt;
            amdxt[k] = amdx[k] * fmt
                + am[k]
                    * (ckm * (-ckm * chi * temp + 1.0) - d.recip() * 0.25
                        + chi * temp * 0.125 / (d * d));
            if k == 0 {
                let fmxxx = (2.0 * kf - 1.0) / chi.pow3() + fmx1.pow3() * 2.0
                    - fmx2.pow3() * 8.0;
                amdxxx = amdxx[k] * fmx + amdx[k] * fmxx * 2.0 + am[k] * fmxxx;
                let fmt1dx = ckm - temp * chi * ckm * ckm;
                let fmt2dx = (-chi * temp * 0.125 / d + 0.25) / d;
                let fmxt = fmt1dx - fmt2dx;
                let fmttx = fmt2 * fmt2dx * 4.0 - fmt1 * fmt1dx * 2.0;
                amdxtt = amdxt[k] * fmt
                    + amdt[k] * fmxt
                    + amdx[k] * fmtt
                    + am[k] * fmttx;
                let fmx1dt = ckm - chi * temp * ckm * ckm;
                let fmx2dt = d.recip() * 0.25 * (-chi * temp * 0.5 / d + 1.0);
                let fmxxt = fmx2 * fmx2dt * 4.0 - fmx1 * fmx1dt * 2.0;
                amdxxt = amdxt[k] * fmx
                    + amdx[k] * fmxt
                    + amdt[k] * fmxx
                    + am[k] * fmxxt;
            }
        }

        let sq2t = (temp * 2.0).sqrt();
        let sq2t3 = sq2t * sq2t * sq2t;
        let a = chi * temp + sq2t * r + 1.0;
        let adx = temp + sq2t * rdx;
        let adt = chi + r / sq2t + sq2t * rdt;
        let adxx = sq2t * rdxx;
        let adtt = -r / sq2t3 + rdt * 2.0 / sq2t + sq2t * rdtt;
        let adxt = rdx / sq2t + sq2t * rdxt + 1.0;
        let adxtt = -rdx / sq2t3 + rdxt * 2.0 / sq2t + sq2t * rdxtt;
        let adxxt = rdxx / sq2t + sq2t * rdxxt;
        let xt1 = chi + temp.recip();
        let aln = a.log();
        let fj0 = xt1 * r * 0.5 - aln / sq2t3;
        let asq3 = a * sq2t3;
        let asq3dx = adx * sq2t3;
        let fj0dx = (r + xt1 * rdx) * 0.5 - adx / asq3;
        let fj0dt = (xt1 * rdt - r / temp.pow2()) * 0.5 - adt / asq3
            + aln * 0.75 / (temp * temp * sq2t);
        let fj0dxx =
            rdx + xt1 * rdxx * 0.5 + (adx / a).pow2() / sq2t3 - adxx / asq3;
        let fj0dtt = r / temp.pow3() - rdt / temp.pow2()
            + xt1 * rdtt * 0.5
            + adt * 3.0 / (asq3 * temp)
            + (adt / a).pow2() / sq2t3
            - adtt / asq3
            - aln * 1.875 / (temp * temp * temp * sq2t);
        let bxt = adx * 1.5 / temp + adx * adt / a - adxt;
        let bxxt = adxx * 1.5 / temp + (adxx * adt + adx * adxt) / a
            - (adx / a).pow2() * adt
            - adxxt;
        let fj0dxt = (rdt - rdx / temp.pow2() + xt1 * rdxt) * 0.5 + bxt / asq3;
        let fj0xxx = rdxx * 1.5
            + xt1 * rdxxx * 0.5
            + (adx * (adxx / a - (adx / a).pow2()) * 2.0 - sq2t * rdxxx
                + adxx / asq3 * asq3dx)
                / asq3;
        let fj0xtt = rdx / temp.pow3() - rdxt / temp.pow2()
            + (rdtt + xt1 * rdxtt) * 0.5
            + (adxt - adt / asq3 * asq3dx) * 3.0 / (temp * asq3)
            + (adt * (adxt / a - adt * adx / (a * a)) * 2.0 - adxtt
                + adtt * asq3dx / asq3)
                / asq3
            - adx * 1.875 / (temp * temp * temp * sq2t * a);
        let fj0xxt = (rdxt * 2.0 - rdxx / temp.pow2() + xt1 * rdxxt) * 0.5
            + (bxxt - bxt * asq3dx / asq3) / asq3;

        out.w0 = BlinSeries {
            w: fj0 + am[0] * PI26,
            wdx: fj0dx + amdx[0] * PI26,
            wdt: fj0dt + amdt[0] * PI26,
            wdxx: fj0dxx + amdxx[0] * PI26,
            wdtt: fj0dtt + amdtt[0] * PI26,
            wdxt: fj0dxt + amdxt[0] * PI26,
        };
        out.w0xxx = fj0xxx + amdxxx * PI26;
        out.w0xtt = fj0xtt + amdxtt * PI26;
        out.w0xxt = fj0xxt + amdxxt * PI26;

        let fj1 = (r * r * r / 1.5 - fj0) / temp;
        let fj1dx = (r * r * rdx * 2.0 - fj0dx) / temp;
        let fj1dt = (r * r * rdt * 2.0 - fj0dt - fj1) / temp;
        let fj1dxx = (r * rdx * rdx * 4.0 + r * r * rdxx * 2.0 - fj0dxx) / temp;
        let fj1dtt =
            (r * rdt * rdt * 4.0 + r * r * rdtt * 2.0 - fj0dtt - fj1dt * 2.0)
                / temp;
        let fj1dxt =
            (r * rdx * rdt * 4.0 + r * r * rdxt * 2.0 - fj0dxt - fj1dx) / temp;

        out.w1 = BlinSeries {
            w: fj1 + am[1] * PI26,
            wdx: fj1dx + amdx[1] * PI26,
            wdt: fj1dt + amdt[1] * PI26,
            wdxx: fj1dxx + amdxx[1] * PI26,
            wdtt: fj1dtt + amdtt[1] * PI26,
            wdxt: fj1dxt + amdxt[1] * PI26,
        };

        let fj2 = (chi * r * r * r * 0.5 - fj1 * 1.25) / temp;
        let fj2dx =
            (r * r * r * 0.5 + chi * r * r * rdx * 1.5 - fj1dx * 1.25) / temp;
        let fj2dt = (chi * r * r * rdt * 1.5 - fj1dt * 1.25 - fj2) / temp;
        let fj2dxx = (r * rdx * (r + chi * rdx) * 3.0
            + chi * r * r * rdxx * 1.5
            - fj1dxx * 1.25)
            / temp;
        let fj2dtt = (chi * r * (rdt * rdt + r * rdtt * 0.5) * 3.0
            - fj1dtt * 1.25
            - fj2dt * 2.0)
            / temp;
        let fj2dxt = (r * rdt * (r + chi * rdx * 2.0) * 1.5
            + chi * r * r * rdxt * 1.5
            - fj1dxt * 1.25
            - fj2dx)
            / temp;

        out.w2 = BlinSeries {
            w: fj2 + am[2] * PI26,
            wdx: fj2dx + amdx[2] * PI26,
            wdt: fj2dt + amdt[2] * PI26,
            wdxx: fj2dxx + amdxx[2] * PI26,
            wdtt: fj2dtt + amdtt[2] * PI26,
            wdxt: fj2dxt + amdxt[2] * PI26,
        };
    }
    out
}
