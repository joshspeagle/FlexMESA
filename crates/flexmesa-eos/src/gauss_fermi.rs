// SPDX-License-Identifier: LGPL-3.0-or-later
//! Relativistic Fermi-Dirac integrals and analytical derivatives via
//! Aparicio (1998, ApJS 117, 632) four-interval Gauss-Legendre and
//! Gauss-Laguerre quadrature.
//!
//! Ports upstream MESA `eos/private/gauss_fermi.f90` (`dfermi`, `fdfunc1`,
//! `fdfunc2`, `dqleg020`, `dqlag020`) and `eos_fermi_dirac_integral` in
//! `eos/public/eos_lib.f90`.

use flexmesa_autodiff::AutoDiffReal2VarOrder1;

/// Positive abscissae of the 20-point Gauss-Legendre quadrature rule on
/// `(-1, 1)` (`dqleg020` in `gauss_fermi.f90`).
pub const DQLEG020_XG: [f64; 10] = [
    7.652_652_113_349_733_4e-2,
    2.277_858_511_416_450_8e-1,
    3.737_060_887_154_195_6e-1,
    5.108_670_019_508_271_0e-1,
    6.360_536_807_265_150_3e-1,
    7.463_319_064_601_507_9e-1,
    8.391_169_718_222_188_2e-1,
    9.122_344_282_513_259_1e-1,
    9.639_719_272_779_137_9e-1,
    9.931_285_991_850_949_2e-1,
];

/// Weights of the 20-point Gauss-Legendre quadrature rule on `(-1, 1)`
/// (`dqleg020` in `gauss_fermi.f90`).
pub const DQLEG020_WG: [f64; 10] = [
    1.527_533_871_307_258_5e-1,
    1.491_729_864_726_037_5e-1,
    1.420_961_093_183_820_5e-1,
    1.316_886_384_491_766_3e-1,
    1.181_945_319_615_184_2e-1,
    1.019_301_198_172_404_4e-1,
    8.327_674_157_670_474_9e-2,
    6.267_204_833_410_906_4e-2,
    4.060_142_980_038_694_1e-2,
    1.761_400_713_915_211_8e-2,
];

/// Abscissae of the 20-point Gauss-Laguerre quadrature rule on `(0, +inf)`
/// (`dqlag020` in `gauss_fermi.f90`).
pub const DQLAG020_XG: [f64; 20] = [
    7.053_988_969_198_875_3e-2,
    3.721_268_180_016_114_4e-1,
    9.165_821_024_832_735_6e-1,
    1.707_306_531_028_343_9e0,
    2.749_199_255_309_432_1e0,
    4.048_925_313_850_886_9e0,
    5.615_174_970_861_616_5e0,
    7.459_017_453_671_063_3e0,
    9.594_392_869_581_096_8e0,
    1.203_880_254_696_431_6e1,
    1.481_429_344_263_074_0e1,
    1.794_889_552_051_937_6e1,
    2.147_878_824_028_501_1e1,
    2.545_170_279_318_690_6e1,
    2.993_255_463_170_061_2e1,
    3.501_343_424_047_900_0e1,
    4.083_305_705_672_857_1e1,
    4.761_999_404_734_650_2e1,
    5.581_079_575_006_389_9e1,
    6.652_441_652_561_575_4e1,
];

/// Modified weights (`exp(xg) * w_laguerre`) of the 20-point Gauss-Laguerre
/// quadrature rule (`dqlag020` in `gauss_fermi.f90`).
pub const DQLAG020_WG: [f64; 20] = [
    1.810_800_624_189_892_6e-1,
    4.225_567_678_785_639_7e-1,
    6.669_095_467_018_481_5e-1,
    9.153_523_727_830_736_7e-1,
    1.169_539_707_195_546_0e0,
    1.431_354_985_928_206_0e0,
    1.702_981_137_985_022_7e0,
    1.987_015_890_792_747_2e0,
    2.286_635_781_253_430_8e0,
    2.605_834_727_553_833_3e0,
    2.949_783_734_213_950_9e0,
    3.325_395_782_009_319_6e0,
    3.742_255_470_589_810_9e0,
    4.214_236_710_251_880_4e0,
    4.762_518_461_490_209_3e0,
    5.421_726_044_245_574_3e0,
    6.254_012_356_932_421_3e0,
    7.387_314_389_054_434_6e0,
    9.151_328_730_987_479_6e0,
    1.289_338_864_593_999_7e1,
];

/// Evaluates the untransformed Fermi-Dirac integrand `fdfunc1` and its partial
/// derivatives with respect to $\eta$ and $\theta$.
#[inline]
fn fdfunc1(x: f64, dk: f64, eta: f64, theta: f64) -> (f64, f64, f64) {
    let xdk = x.powf(dk);
    let xdkp1 = x * xdk;
    let dxst = (1.0 + 0.5 * x * theta).sqrt();

    if (x - eta) < 100.0 {
        let factor = (x - eta).exp();
        let denom = factor + 1.0;
        let fd = xdk * dxst / denom;
        let fdeta = fd * factor / denom;
        let denom2 = 4.0 * dxst * denom;
        let fdtheta = xdkp1 / denom2;
        (fd, fdeta, fdtheta)
    } else {
        let factor = (eta - x).exp();
        let fd = xdk * dxst * factor;
        let fdeta = fd;
        let denom2 = 4.0 * dxst;
        let fdtheta = (xdkp1 / denom2) * factor;
        (fd, fdeta, fdtheta)
    }
}

/// Evaluates the $z^2 = x$ variable-transformed Fermi-Dirac integrand
/// `fdfunc2` and its partial derivatives with respect to $\eta$ and $\theta$.
#[inline]
fn fdfunc2(x: f64, dk: f64, eta: f64, theta: f64) -> (f64, f64, f64) {
    let xsq = x * x;
    let xdk = x.powf(2.0 * dk + 1.0);
    let xdkp1 = xsq * xdk;
    let dxst = (1.0 + 0.5 * xsq * theta).sqrt();

    if (xsq - eta) < 100.0 {
        let factor = (xsq - eta).exp();
        let denom = factor + 1.0;
        let fd = 2.0 * xdk * dxst / denom;
        let fdeta = fd * factor / denom;
        let denom2 = 4.0 * dxst * denom;
        let fdtheta = 2.0 * xdkp1 / denom2;
        (fd, fdeta, fdtheta)
    } else {
        let factor = (eta - xsq).exp();
        let fd = 2.0 * xdk * dxst * factor;
        let fdeta = fd;
        let denom2 = 4.0 * dxst;
        let fdtheta = (2.0 * xdkp1 / denom2) * factor;
        (fd, fdeta, fdtheta)
    }
}

/// Applies the 20-point Gauss-Legendre rule `dqleg020` on `[a, b]`.
#[inline]
fn dqleg020<F>(
    f: F,
    a: f64,
    b: f64,
    dk: f64,
    eta: f64,
    theta: f64,
) -> (f64, f64, f64)
where
    F: Fn(f64, f64, f64, f64) -> (f64, f64, f64),
{
    let center = f64::midpoint(a, b);
    let hlfrun = 0.5 * (b - a);
    let mut res = 0.0_f64;
    let mut dres = 0.0_f64;
    let mut ddres = 0.0_f64;

    for j in 0..10 {
        let absc1 = center + hlfrun * DQLEG020_XG[j];
        let absc2 = center - hlfrun * DQLEG020_XG[j];
        let (fval1, dfval1, ddfval1) = f(absc1, dk, eta, theta);
        let (fval2, dfval2, ddfval2) = f(absc2, dk, eta, theta);
        let w = DQLEG020_WG[j];
        res += (fval1 + fval2) * w;
        dres += (dfval1 + dfval2) * w;
        ddres += (ddfval1 + ddfval2) * w;
    }

    (res * hlfrun, dres * hlfrun, ddres * hlfrun)
}

/// Applies the 20-point Gauss-Laguerre rule `dqlag020` on `[a, +inf)` with
/// scale `b`.
#[inline]
fn dqlag020<F>(
    f: F,
    a: f64,
    b: f64,
    dk: f64,
    eta: f64,
    theta: f64,
) -> (f64, f64, f64)
where
    F: Fn(f64, f64, f64, f64) -> (f64, f64, f64),
{
    let mut res = 0.0_f64;
    let mut dres = 0.0_f64;
    let mut ddres = 0.0_f64;

    for j in 0..20 {
        let absc = a + b * DQLAG020_XG[j];
        let (fval, dfval, ddfval) = f(absc, dk, eta, theta);
        let w = DQLAG020_WG[j];
        res += fval * w;
        dres += dfval * w;
        ddres += ddfval * w;
    }

    (res * b, dres * b, ddres * b)
}

/// Evaluates the generalized relativistic Fermi-Dirac integral
/// $$F_k(\eta, \theta) = \int_0^\infty
///   \frac{x^k \sqrt{1 + \frac{1}{2} x \theta}}{e^{x - \eta} + 1}\,dx$$
/// and its analytical partial derivatives
/// $(\partial F_k/\partial\eta, \partial F_k/\partial\theta)$ using the
/// four-interval 20-point Gauss-Legendre/Gauss-Laguerre quadrature scheme of
/// Aparicio (1998, ApJS 117, 632).
///
/// Matches `dfermi` in `eos/private/gauss_fermi.f90` and
/// `eos_fermi_dirac_integral` in `eos/public/eos_lib.f90`.
#[must_use]
pub fn dfermi(dk: f64, eta: f64, theta: f64) -> (f64, f64, f64) {
    const D: f64 = 3.3609;
    const SG: f64 = 9.1186e-2;
    const A1: f64 = 6.7774;
    const B1: f64 = 1.1418;
    const C1: f64 = 2.9826;
    const A2: f64 = 3.7601;
    const B2: f64 = 9.3719e-2;
    const C2: f64 = 2.1063e-2;
    const D2: f64 = 3.1084e1;
    const E2: f64 = 1.0056;
    const A3: f64 = 7.5669;
    const B3: f64 = 1.1695;
    const C3: f64 = 7.5416e-1;
    const D3: f64 = 6.6558;
    const E3: f64 = -1.2819e-1;

    let eta1 = SG * (eta - D);
    let xi = if eta1 <= 50.0 {
        (1.0 + eta1.exp()).ln() / SG
    } else {
        eta - D
    };
    let xi2 = xi * xi;

    let x1 = (A1 + B1 * xi + C1 * xi2) / (1.0 + C1 * xi);
    let x2 = (A2 + B2 * xi + C2 * D2 * xi2) / (1.0 + E2 * xi + C2 * xi2);
    let x3 = (A3 + B3 * xi + C3 * D3 * xi2) / (1.0 + E3 * xi + C3 * xi2);

    let s1 = x1 - x2;
    let s2 = x1;
    let s3 = x1 + x3;
    let s12 = s1.sqrt();

    let (res1, dres1, ddres1) = dqleg020(fdfunc2, 0.0, s12, dk, eta, theta);
    let (res2, dres2, ddres2) = dqleg020(fdfunc1, s1, s2, dk, eta, theta);
    let (res3, dres3, ddres3) = dqleg020(fdfunc1, s2, s3, dk, eta, theta);
    let (res4, dres4, ddres4) = dqlag020(fdfunc1, s3, 1.0, dk, eta, theta);

    (
        res1 + res2 + res3 + res4,
        dres1 + dres2 + dres3 + dres4,
        ddres1 + ddres2 + ddres3 + ddres4,
    )
}

/// Evaluates [`dfermi`] on two-variable first-order dual inputs `eta` and
/// `theta`, propagating first-order partial derivatives via the chain rule.
#[must_use]
#[inline]
pub fn dfermi_order1(
    dk: f64,
    eta: AutoDiffReal2VarOrder1,
    theta: AutoDiffReal2VarOrder1,
) -> AutoDiffReal2VarOrder1 {
    let (fd, fdeta, fdtheta) = dfermi(dk, eta.val, theta.val);
    AutoDiffReal2VarOrder1::new(
        fd,
        [
            fdeta * eta.d1[0] + fdtheta * theta.d1[0],
            fdeta * eta.d1[1] + fdtheta * theta.d1[1],
        ],
    )
}
