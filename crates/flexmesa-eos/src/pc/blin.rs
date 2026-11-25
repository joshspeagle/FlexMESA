// SPDX-License-Identifier: GPL-3.0-or-later
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
    let chi1 = AutoDiffReal2VarOrder1::constant(0.6);
    let chi2 = AutoDiffReal2VarOrder1::constant(14.0);
    let xmax = AutoDiffReal2VarOrder1::constant(30.0);
    let dchi1 = AutoDiffReal2VarOrder1::constant(0.1);
    let dchi2 = chi2 - chi1 - dchi1;
    let xscal1 = xmax / dchi1;
    let xscal2 = xmax / dchi2;

    let x1 = (chi - chi1) * xscal1;
    let x2 = (chi - chi2) * xscal2;

    if x1.val < -xmax.val {
        blin9a(temp, chi)
    } else if x2.val < xmax.val {
        let ((fp, fm), a, b) = if x1.val < xmax.val {
            (fermi10(x1, xmax.val), blin9a(temp, chi), blin9b(temp, chi))
        } else {
            (fermi10(x2, xmax.val), blin9b(temp, chi), blin9c(temp, chi))
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
            let aa = flexmesa_math::exp(-au);
            let sq = (1.0 + au * temp / 2.0).sqrt();
            let dn = aa + echi;
            w = w + ac * sq / dn;
            wdx = wdx + ac * sq / (dn * dn);
            wdt = wdt + ac * au / (sq * dn);
            wdxx = wdxx + ac * sq * (echi - aa) / dn.pow3();
            wdtt = wdtt - ac * au * au / (dn * sq * sq * sq);
            wdxt = wdxt + ac * au / (sq * dn * dn);
            if k == 0 {
                wdxxx = wdxxx
                    + ac * sq * (echi * echi - 4.0 * echi * aa + aa * aa)
                        / (dn * dn * dn * dn);
                wdxtt = wdxtt - ac * au * au / (dn * dn * sq * sq * sq);
                wdxxt = wdxxt + ac * au * (echi - aa) / (sq * dn * dn * dn);
            }
        }

        let s = BlinSeries {
            w,
            wdx: wdx * echi,
            wdt: 0.25 * wdt,
            wdxx: wdxx * echi,
            wdtt: 0.0625 * wdtt,
            wdxt: 0.25 * wdxt * echi,
        };
        match k {
            0 => {
                out.w0 = s;
                out.w0xxx = wdxxx * echi;
                out.w0xtt = 0.0625 * wdxtt * echi;
                out.w0xxt = 0.25 * wdxxt * echi;
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
            let ah_axk = AH[i] * flexmesa_math::powi(ax, k as i32);
            let av = AV[i];

            let ce = AutoDiffReal2VarOrder1::constant(ax - 1.0);
            let echi = (ce * chi).exp();
            let de = 1.0 + echi;
            let d = 1.0 + ax * chi * temp / 2.0;
            let h = chi.powf(kf + 1.0) * sqchi * d.sqrt() / de;
            let hx = (kf + 1.5) / chi + 0.25 * ax * temp / d - echi * ce / de;
            let hdx = h * hx;
            let hxx = (kf + 1.5) / (chi * chi)
                + 0.125 * (ax * temp / d).pow2()
                + echi * (ce / de).pow2();
            let hdxx = hdx * hx - h * hxx;
            let ht = 0.25 * ax * chi / d;
            let hdt = h * ht;
            let hdtt = -h * ht * ht;
            let htx = 1.0 / chi - 0.5 * ax * temp / d;
            let hdxt = hdx * ht + hdt * htx;

            let xichi = axi + chi;
            let dxi = 1.0 + xichi * temp / 2.0;
            let v = xichi.powf(kf) * (xichi * dxi).sqrt();
            let vx = (kf + 0.5) / xichi + 0.25 * temp / dxi;
            let vdx = v * vx;
            let vt = 0.25 * xichi / dxi;
            let vdt = v * vt;
            let vxx =
                (kf + 0.5) / (xichi * xichi) + 0.125 * (temp / dxi).pow2();
            let vdxx = vdx * vx - v * vxx;
            let vdtt = -v * vt * vt;
            let vxt = 1.0 / xichi - 0.5 * temp / dxi;
            let vdxt = vdt * vxt + vdx * vt;

            if k == 0 {
                let hdxxt = hdxx * ht
                    + hdx * ht * htx
                    + hdxt * htx
                    + hdt * (0.25 * (ax * temp / d).pow2() - 1.0 / (chi * chi));
                let hdxtt = hdxt * ht - hdx * 0.125 * (ax * chi / d).pow2()
                    + hdtt * htx
                    + hdt
                        * 0.5
                        * ax
                        * (temp * 0.5 * ax * chi / (d * d) - 1.0 / d);
                let hxxx = (2.0 * kf + 3.0) / chi.pow3()
                    + 0.125 * (ax * temp / d).pow3()
                    - echi * (1.0 - echi) * (ce / de).pow3();
                let hdxxx = hdxx * hx - 2.0 * hdx * hxx + h * hxxx;

                let vdxxx = vdxx * vx - 2.0 * vdx * vxx
                    + v * ((2.0 * kf + 1.0) / xichi.pow3()
                        + 0.125 * (temp / dxi).pow3());
                let vxxt = (1.0 - 0.5 * temp * xichi / dxi) / dxi;
                let vdxxt = vdxt * vx + vdx * 0.25 * vxxt
                    - vdt * vxx
                    - v * 0.25 * temp / dxi * vxxt;
                let vdxtt = vdtt * vxt - vdt * 0.5 * vxxt + vdxt * vt
                    - vdx * 0.125 * (xichi / dxi).pow2();

                wdxxx = wdxxx + hdxxx * ah_axk + vdxxx * av;
                wdxtt = wdxtt + hdxtt * ah_axk + vdxtt * av;
                wdxxt = wdxxt + hdxxt * ah_axk + vdxxt * av;
            }

            w = w + ah_axk * h + av * v;
            wdx = wdx + ah_axk * hdx + av * vdx;
            wdt = wdt + ah_axk * hdt + av * vdt;
            wdxx = wdxx + ah_axk * hdxx + av * vdxx;
            wdtt = wdtt + ah_axk * hdtt + av * vdtt;
            wdxt = wdxt + ah_axk * hdxt + av * vdxt;
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
