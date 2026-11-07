// SPDX-License-Identifier: LGPL-3.0-or-later
//! Inverse Fermi-Dirac integrals (`FERINV7`), Fermi weight function (`FERMI10`),
//! and Chabrier & Potekhin (1998) relativistic chemical potential fit
//! (`CHEMFIT`, `CHEMFIT7`) from `eos/private/pc_eos.f90`.

use flexmesa_autodiff::AutoDiffReal2VarOrder1;

/// Fermi weight function $f_+(x) = (e^x + 1)^{-1}$ and $f_-(x) = 1 - f_+(x)$
/// (`FERMI10` in `eos/private/pc_eos.f90` lines 1818-1841).
#[must_use]
#[inline]
pub fn fermi10(
    x: AutoDiffReal2VarOrder1,
    xmax: f64,
) -> (AutoDiffReal2VarOrder1, AutoDiffReal2VarOrder1) {
    let xmax_eff = xmax.max(3.0);
    if x.val > xmax_eff {
        (
            AutoDiffReal2VarOrder1::constant(0.0),
            AutoDiffReal2VarOrder1::constant(1.0),
        )
    } else if x.val < -xmax_eff {
        (
            AutoDiffReal2VarOrder1::constant(1.0),
            AutoDiffReal2VarOrder1::constant(0.0),
        )
    } else {
        let fp = 1.0 / (x.exp() + 1.0);
        let fm = 1.0 - fp;
        (fp, fm)
    }
}

/// Inverse non-relativistic Fermi-Dirac integral $X_{k-1/2}(f)$ and its first
/// and second derivatives with respect to $f$ for $n = k = 0, 1, 2, 3$
/// (`FERINV7` in `eos/private/pc_eos.f90` lines 2165-2304, Antia 1993 ApJS
/// 84, 101 rational approximation).
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn ferinv7(
    mut f: AutoDiffReal2VarOrder1,
    n: usize,
) -> (
    AutoDiffReal2VarOrder1,
    AutoDiffReal2VarOrder1,
    AutoDiffReal2VarOrder1,
) {
    const LA: [usize; 4] = [5, 4, 3, 2];
    const LB: [usize; 4] = [6, 3, 4, 3];
    const LD: [usize; 4] = [6, 5, 5, 6];

    const A: [[f64; 6]; 4] = [
        [
            -1.570_044_577_033e4,
            1.001_958_278_442e4,
            -2.805_343_454_951e3,
            4.121_170_498_099e2,
            -3.174_780_572_961e1,
            1.0,
        ],
        [
            1.999_266_880_833e4,
            5.702_479_099_336e3,
            6.610_132_843_877e2,
            3.818_838_129_486e1,
            1.0,
            0.0,
        ],
        [
            1.715_627_994_191e2,
            1.125_926_232_897e2,
            2.056_296_753_055e1,
            1.0,
            0.0,
            0.0,
        ],
        [2.138_969_250_409e2, 3.539_903_493_971e1, 1.0, 0.0, 0.0, 0.0],
    ];

    const B: [[f64; 7]; 4] = [
        [
            -2.782_831_558_471e4,
            2.886_114_034_012e4,
            -1.274_243_093_149e4,
            3.063_252_215_963e3,
            -4.225_615_045_074e2,
            3.168_918_168_284e1,
            -1.008_561_571_363,
        ],
        [
            1.771_804_140_488e4,
            -2.014_785_161_019e3,
            9.130_355_392_717e1,
            -1.670_718_177_489,
            0.0,
            0.0,
            0.0,
        ],
        [
            2.280_653_583_157e2,
            1.193_456_203_021e2,
            1.167_743_113_54e1,
            -3.226_808_804_038e-1,
            3.519_268_762_788e-3,
            0.0,
            0.0,
        ],
        [
            7.108_545_512_71e2,
            9.873_746_988_121e1,
            1.067_755_522_895,
            -1.182_798_726_503e-2,
            0.0,
            0.0,
            0.0,
        ],
    ];

    const C: [[f64; 7]; 4] = [
        [
            2.206_779_160_034e-8,
            -1.437_701_234_283e-6,
            6.103_116_850_636e-5,
            -1.169_411_057_416e-3,
            1.814_141_021_608e-2,
            -9.588_603_457_639e-2,
            1.0,
        ],
        [
            -1.277_060_388_085e-2,
            7.187_946_804_945e-2,
            -4.262_314_235_106e-1,
            4.997_559_426_872e-1,
            -1.285_579_118_012,
            -3.930_805_454_272e-1,
            1.0,
        ],
        [
            -6.321_828_169_799e-3,
            -2.183_147_266_896e-2,
            -1.057_562_799_32e-1,
            -4.657_944_387_545e-1,
            -5.951_932_864_088e-1,
            3.684_471_177_1e-1,
            1.0,
        ],
        [
            -3.312_041_011_227e-2,
            1.315_763_372_315e-1,
            -4.820_942_898_296e-1,
            5.099_038_074_944e-1,
            5.495_613_498_63e-1,
            -1.498_867_562_255,
            1.0,
        ],
    ];

    const D: [[f64; 7]; 4] = [
        [
            8.827_116_613_576e-8,
            -5.750_804_196_059e-6,
            2.429_627_688_357e-4,
            -4.601_959_491_394e-3,
            6.932_122_275_919e-2,
            -3.217_372_489_776e-1,
            3.124_344_749_296,
        ],
        [
            -9.745_794_806_288e-3,
            5.485_432_756_838e-2,
            -3.299_466_243_26e-1,
            4.077_841_975_923e-1,
            -1.145_531_476_975,
            -6.067_091_689_181e-2,
            0.0,
        ],
        [
            -4.381_942_605_018e-3,
            -1.513_236_504_1e-2,
            -7.850_001_283_886e-2,
            -3.407_561_772_612e-1,
            -5.074_812_565_486e-1,
            -1.387_107_009_074e-1,
            0.0,
        ],
        [
            -2.315_515_517_515e-2,
            9.198_776_585_252e-2,
            -3.835_879_295_548e-1,
            5.415_026_856_351e-1,
            -3.847_241_692_193e-1,
            3.739_781_456_585e-2,
            -3.008_504_449_098e-2,
        ],
    ];

    if f.val <= 0.0 {
        f = AutoDiffReal2VarOrder1::constant(1.0e-99);
    }

    if f.val < 4.0 {
        let t = f;
        let mut up = AutoDiffReal2VarOrder1::zero();
        let mut up1 = AutoDiffReal2VarOrder1::zero();
        let mut up2 = AutoDiffReal2VarOrder1::zero();
        for i in (0..=LA[n]).rev() {
            let ai = A[n][i];
            up = up * t + ai;
            if i >= 1 {
                up1 = up1 * t + ai * (i as f64);
            }
            if i >= 2 {
                up2 = up2 * t + ai * ((i * (i - 1)) as f64);
            }
        }
        let mut down = AutoDiffReal2VarOrder1::zero();
        let mut down1 = AutoDiffReal2VarOrder1::zero();
        let mut down2 = AutoDiffReal2VarOrder1::zero();
        for i in (0..=LB[n]).rev() {
            let bi = B[n][i];
            down = down * t + bi;
            if i >= 1 {
                down1 = down1 * t + bi * (i as f64);
            }
            if i >= 2 {
                down2 = down2 * t + bi * ((i * (i - 1)) as f64);
            }
        }
        let x = (t * up / down).log();
        let xdf = 1.0 / t + up1 / up - down1 / down;
        let xdff = -1.0 / (t * t) + up2 / up - (up1 / up).pow2() - down2 / down
            + (down1 / down).pow2();
        (x, xdf, xdff)
    } else {
        let p = -1.0 / (0.5 + n as f64);
        let t = f.powf(p);
        let t1 = p * t / f;
        let t2 = p * (p - 1.0) * t / (f * f);
        let mut up = AutoDiffReal2VarOrder1::zero();
        let mut up1 = AutoDiffReal2VarOrder1::zero();
        let mut up2 = AutoDiffReal2VarOrder1::zero();
        for i in (0..=6).rev() {
            let ci = C[n][i];
            up = up * t + ci;
            if i >= 1 {
                up1 = up1 * t + ci * (i as f64);
            }
            if i >= 2 {
                up2 = up2 * t + ci * ((i * (i - 1)) as f64);
            }
        }
        let mut down = AutoDiffReal2VarOrder1::zero();
        let mut down1 = AutoDiffReal2VarOrder1::zero();
        let mut down2 = AutoDiffReal2VarOrder1::zero();
        for i in (0..=LD[n]).rev() {
            let di = D[n][i];
            down = down * t + di;
            if i >= 1 {
                down1 = down1 * t + di * (i as f64);
            }
            if i >= 2 {
                down2 = down2 * t + di * ((i * (i - 1)) as f64);
            }
        }
        let r = up / down;
        let r1 = (up1 - up * down1 / down) / down;
        let r2 = (up2 - (2.0 * up1 * down1 + up * down2) / down
            + 2.0 * up * (down1 / down).pow2())
            / down;
        let x = r / t;
        let rt = (r1 - r / t) / t;
        let xdf = t1 * rt;
        let xdff = t2 * rt + t1 * t1 * (r2 - 2.0 * rt) / t;
        (x, xdf, xdff)
    }
}

/// Chabrier & Potekhin (1998) analytical fit for $\chi = \mu_e / (k_{\mathrm{B}} T)$
/// from electron density `dens` (in atomic units) and temperature `temp` (in
/// atomic units) (`CHEMFIT` and `CHEMFIT7` in `eos/private/pc_eos.f90` lines
/// 2889-2975 with `KDERIV = 0`).
#[must_use]
pub fn chemfit(
    dens: AutoDiffReal2VarOrder1,
    temp: AutoDiffReal2VarOrder1,
) -> AutoDiffReal2VarOrder1 {
    let denr = dens / 2.573_380_6e6;
    let temr = temp / 1.877_886_5e4;

    let pf0 = (29.608_813_2 * denr).powf(1.0 / 3.0);
    let tf = if pf0.val > 1.0e-4 {
        (1.0 + pf0 * pf0).sqrt() - 1.0
    } else {
        0.5 * pf0 * pf0
    };
    let theta = temr / tf;
    let theta32 = theta * theta.sqrt();
    let q2 = 12.0 + 8.0 / theta32;
    let sqt = temr.sqrt();
    let t1 = (-theta).exp();
    let t2 = t1 * t1;
    let u3 = t2 + 1.612;
    let d3 = 6.192 * theta.powf(0.0944) * t2 + 5.535 * theta.powf(0.698);
    let q3 = 1.365_568_127 - u3 / d3;
    let q1 = if theta.val > 1.0e-5 {
        1.5 * t1 / (1.0 - t1)
    } else {
        1.5 / theta
    };
    let g = (1.0 + q2 * temr * q3 + q1 * sqt) * temr;
    let h = (1.0 + 0.5 * temr / theta) * (1.0 + q2 * temr);
    let ct = 1.0 + g / h;
    let f = (2.0 / 3.0) / theta32;
    let (x, _, _) = ferinv7(f, 1);
    x - 1.5 * ct.log()
}
