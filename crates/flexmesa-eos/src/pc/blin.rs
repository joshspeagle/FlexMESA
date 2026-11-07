// SPDX-License-Identifier: LGPL-3.0-or-later
//! Blinnikov et al. (1996) relativistic Fermi-Dirac integral fits (`BLIN9`,
//! `BLIN9a`, `BLIN9b`) from `eos/private/pc_eos.f90` lines 2306-2653.

use flexmesa_autodiff::AutoDiffReal2VarOrder1;

pub use crate::pc::blin_c::blin9c;
use crate::pc::ferinv::fermi10;

/// Single-series output $(W_k, \partial_\chi W_k, \partial_\tau W_k,
/// \partial_{\chi\chi} W_k, \partial_{\tau\tau} W_k, \partial_{\chi\tau} W_k)$.
#[derive(Clone, Copy, Debug, Default)]
pub struct BlinSeries {
    /// Integral value $W_k$.
    pub w: AutoDiffReal2VarOrder1,
    /// First derivative $\partial W_k / \partial \chi$.
    pub wdx: AutoDiffReal2VarOrder1,
    /// First derivative $\partial W_k / \partial \tau$.
    pub wdt: AutoDiffReal2VarOrder1,
    /// Second derivative $\partial^2 W_k / \partial \chi^2$.
    pub wdxx: AutoDiffReal2VarOrder1,
    /// Second derivative $\partial^2 W_k / \partial \tau^2$.
    pub wdtt: AutoDiffReal2VarOrder1,
    /// Mixed derivative $\partial^2 W_k / (\partial \chi \partial \tau)$.
    pub wdxt: AutoDiffReal2VarOrder1,
}

/// Complete output of `BLIN9` (`W0`, `W1`, `W2` and third derivatives of `W0`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Blin9Output {
    /// Series $k = 0$ ($F_{1/2}$).
    pub w0: BlinSeries,
    /// Series $k = 1$ ($F_{3/2}$).
    pub w1: BlinSeries,
    /// Series $k = 2$ ($F_{5/2}$).
    pub w2: BlinSeries,
    /// Third derivative $\partial^3 W_0 / \partial \chi^3$.
    pub w0xxx: AutoDiffReal2VarOrder1,
    /// Third derivative $\partial^3 W_0 / (\partial \chi \partial \tau^2)$.
    pub w0xtt: AutoDiffReal2VarOrder1,
    /// Third derivative $\partial^3 W_0 / (\partial \chi^2 \partial \tau)$.
    pub w0xxt: AutoDiffReal2VarOrder1,
}

#[inline]
fn blend_series(
    fp: AutoDiffReal2VarOrder1,
    fm: AutoDiffReal2VarOrder1,
    a: BlinSeries,
    b: BlinSeries,
) -> BlinSeries {
    BlinSeries {
        w: a.w * fp + b.w * fm,
        wdx: a.wdx * fp + b.wdx * fm,
        wdt: a.wdt * fp + b.wdt * fm,
        wdxx: a.wdxx * fp + b.wdxx * fm,
        wdtt: a.wdtt * fp + b.wdtt * fm,
        wdxt: a.wdxt * fp + b.wdxt * fm,
    }
}

/// Evaluates `BLIN9` (`eos/private/pc_eos.f90` lines 2306-2415).
#[must_use]
pub fn blin9(
    temp: AutoDiffReal2VarOrder1,
    chi: AutoDiffReal2VarOrder1,
) -> Blin9Output {
    const CHI1: f64 = 0.6;
    const CHI2: f64 = 14.0;
    const XMAX: f64 = 30.0;
    const DCHI1: f64 = 0.1;
    const DCHI2: f64 = CHI2 - CHI1 - DCHI1;
    const XSCAL1: f64 = XMAX / DCHI1;
    const XSCAL2: f64 = XMAX / DCHI2;

    let x1 = (chi - CHI1) * XSCAL1;
    let x2 = (chi - CHI2) * XSCAL2;

    if x1.val < -XMAX {
        blin9a(temp, chi)
    } else if x2.val < XMAX {
        let ((fp, fm), a, b) = if x1.val < XMAX {
            (fermi10(x1, XMAX), blin9a(temp, chi), blin9b(temp, chi))
        } else {
            (fermi10(x2, XMAX), blin9b(temp, chi), blin9c(temp, chi))
        };
        Blin9Output {
            w0: blend_series(fp, fm, a.w0, b.w0),
            w1: blend_series(fp, fm, a.w1, b.w1),
            w2: blend_series(fp, fm, a.w2, b.w2),
            w0xxx: a.w0xxx * fp + b.w0xxx * fm,
            w0xtt: a.w0xtt * fp + b.w0xtt * fm,
            w0xxt: a.w0xxt * fp + b.w0xxt * fm,
        }
    } else {
        blin9c(temp, chi)
    }
}

/// Non-degenerate regime $\chi < 0.6$ (`BLIN9a` in `eos/private/pc_eos.f90`
/// lines 2417-2526).
#[must_use]
pub fn blin9a(
    temp: AutoDiffReal2VarOrder1,
    chi: AutoDiffReal2VarOrder1,
) -> Blin9Output {
    const AC: [[f64; 5]; 3] = [
        [
            0.370_450_57,
            0.412_584_37,
            9.777_982e-2,
            5.373_415_3e-3,
            3.874_628_1e-5,
        ],
        [
            0.396_031_09,
            0.694_687_95,
            0.223_227_60,
            1.526_293_4e-2,
            1.308_193_9e-4,
        ],
        [
            0.769_346_19,
            1.789_143_7,
            0.707_549_74,
            5.675_567_2e-2,
            5.557_148_0e-4,
        ],
    ];
    const AU: [[f64; 5]; 3] = [
        [
            0.431_398_81,
            1.759_753_7,
            4.104_465_4,
            7.746_703_8,
            13.457_678,
        ],
        [
            0.817_631_76,
            2.472_333_9,
            5.116_006_1,
            9.044_146_5,
            15.049_882,
        ],
        [
            1.255_846_1,
            3.207_040_6,
            6.123_908_2,
            10.316_126,
            16.597_079,
        ],
    ];

    let mut out = Blin9Output::default();
    let echi = (-chi).exp();

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

        for i in 0..5 {
            let ac = AC[k][i];
            let au = AU[k][i];
            let aa = (-au).exp();
            let sq = (1.0 + 0.5 * au * temp).sqrt();
            let dn = echi + aa;
            let dn2 = dn * dn;
            let dn3 = dn2 * dn;
            let sq3 = sq * sq * sq;
            w = w + sq / dn * ac;
            wdx = wdx + sq / dn2 * ac;
            wdt = wdt + (sq * dn).recip() * (ac * au);
            wdxx = wdxx + sq * (echi - aa) / dn3 * ac;
            wdtt = wdtt - (dn * sq3).recip() * (ac * au * au);
            wdxt = wdxt + (sq * dn2).recip() * (ac * au);
            if k == 0 {
                wdxxx = wdxxx
                    + sq * (echi * echi - echi * (4.0 * aa) + aa * aa)
                        / (dn2 * dn2)
                        * ac;
                wdxtt = wdxtt - (dn2 * sq3).recip() * (ac * au * au);
                wdxxt = wdxxt + (echi - aa) / (sq * dn3) * (ac * au);
            }
        }

        let s = BlinSeries {
            w,
            wdx: wdx * echi,
            wdt: wdt * 0.25,
            wdxx: wdxx * echi,
            wdtt: wdtt * 0.0625,
            wdxt: wdxt * echi * 0.25,
        };
        match k {
            0 => {
                out.w0 = s;
                out.w0xxx = wdxxx * echi;
                out.w0xtt = wdxtt * echi * 0.0625;
                out.w0xxt = wdxxt * echi * 0.25;
            }
            1 => out.w1 = s,
            _ => out.w2 = s,
        }
    }
    out
}

/// Intermediate regime $0.6 \le \chi \le 14$ (`BLIN9b` in `eos/private/pc_eos.f90`
/// lines 2528-2653).
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn blin9b(
    temp: AutoDiffReal2VarOrder1,
    mut chi: AutoDiffReal2VarOrder1,
) -> Blin9Output {
    const AX: [f64; 5] = [
        7.265_351e-2,
        0.269_460_8,
        0.533_122,
        0.786_880_1,
        0.956_931_3,
    ];
    const AXI: [f64; 5] = [
        0.263_560_32,
        1.413_403_1,
        3.596_425_8,
        7.085_810_0,
        12.640_801,
    ];
    const AH: [f64; 5] = [
        3.818_735e-2,
        0.125_673_2,
        0.198_630_8,
        0.197_633_4,
        0.106_542_0,
    ];
    const AV: [f64; 5] = [
        0.295_058_69,
        0.320_648_56,
        7.391_557_0e-2,
        3.608_738_9e-3,
        2.336_989_4e-5,
    ];
    const EPS: f64 = 1.0e-3;

    if chi.val < EPS {
        chi = AutoDiffReal2VarOrder1::constant(EPS);
    }

    let mut out = Blin9Output::default();
    let sqchi = chi.sqrt();

    for k in 0..3 {
        let kf = k as f64;
        let mut w = AutoDiffReal2VarOrder1::zero();
        let mut wdx = AutoDiffReal2VarOrder1::zero();
        let mut wdt = AutoDiffReal2VarOrder1::zero();
        let mut wdxx = AutoDiffReal2VarOrder1::zero();
        let mut wdtt = AutoDiffReal2VarOrder1::zero();
        let mut wdxt = AutoDiffReal2VarOrder1::zero();
        let mut wdxxx = AutoDiffReal2VarOrder1::zero();
        let mut wdxtt = AutoDiffReal2VarOrder1::zero();
        let mut wdxxt = AutoDiffReal2VarOrder1::zero();

        for i in 0..5 {
            let ax = AX[i];
            let axi = AXI[i];
            let ah_axk = AH[i] * ax.powi(k as i32);
            let av = AV[i];

            let ce = ax - 1.0;
            let echi = (chi * ce).exp();
            let de = echi + 1.0;
            let d = chi * temp * (0.5 * ax) + 1.0;
            let h = chi.powi((k + 1) as i32) * sqchi * d.sqrt() / de;
            let hx = (kf + 1.5) / chi + temp * (0.25 * ax) / d - echi * ce / de;
            let hdx = h * hx;
            let hxx = (kf + 1.5) / (chi * chi)
                + (temp / d * ax).pow2() * 0.125
                + echi * (ce / de).pow2();
            let hdxx = hdx * hx - h * hxx;
            let ht = chi * (0.25 * ax) / d;
            let hdt = h * ht;
            let hdtt = -h * ht * ht;
            let htx = chi.recip() - temp * (0.5 * ax) / d;
            let hdxt = hdx * ht + hdt * htx;

            let xichi = chi + axi;
            let dxi = xichi * temp * 0.5 + 1.0;
            let v = xichi.powi(k as i32) * (xichi * dxi).sqrt();
            let vx = (kf + 0.5) / xichi + temp * 0.25 / dxi;
            let vdx = v * vx;
            let vt = xichi * 0.25 / dxi;
            let vdt = v * vt;
            let vxx =
                (kf + 0.5) / (xichi * xichi) + (temp / dxi).pow2() * 0.125;
            let vdxx = vdx * vx - v * vxx;
            let vdtt = -v * vt * vt;
            let vxt = xichi.recip() - temp * 0.5 / dxi;
            let vdxt = vdt * vxt + vdx * vt;

            w = w + h * ah_axk + v * av;
            wdx = wdx + hdx * ah_axk + vdx * av;
            wdt = wdt + hdt * ah_axk + vdt * av;
            wdxx = wdxx + hdxx * ah_axk + vdxx * av;
            wdtt = wdtt + hdtt * ah_axk + vdtt * av;
            wdxt = wdxt + hdxt * ah_axk + vdxt * av;

            if k == 0 {
                let hdxxt = hdxx * ht
                    + hdx * ht * htx
                    + hdxt * htx
                    + hdt
                        * ((temp / d * ax).pow2() * 0.25 - (chi * chi).recip());
                let hdxtt = hdxt * ht - hdx * (chi / d * ax).pow2() * 0.125
                    + hdtt * htx
                    + hdt
                        * (temp * chi * (0.5 * ax) / (d * d) - d.recip())
                        * (0.5 * ax);
                let hxxx = (2.0 * kf + 3.0) / chi.pow3()
                    + (temp / d * ax).pow3() * 0.125
                    - echi * (-echi + 1.0) * (ce / de).pow3();
                let hdxxx = hdxx * hx - hdx * hxx * 2.0 + h * hxxx;

                let vdxxx = vdxx * vx - vdx * vxx * 2.0
                    + v * ((2.0 * kf + 1.0) / xichi.pow3()
                        + (temp / dxi).pow3() * 0.125);
                let vxxt = (-temp * xichi * 0.5 / dxi + 1.0) / dxi;
                let vdxxt = vdxt * vx + vdx * vxxt * 0.25
                    - vdt * vxx
                    - v * temp * 0.25 / dxi * vxxt;
                let vdxtt = vdtt * vxt - vdt * vxxt * 0.5 + vdxt * vt
                    - vdx * (xichi / dxi).pow2() * 0.125;

                wdxxx = wdxxx + hdxxx * ah_axk + vdxxx * av;
                wdxtt = wdxtt + hdxtt * ah_axk + vdxtt * av;
                wdxxt = wdxxt + hdxxt * ah_axk + vdxxt * av;
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
    out
}
