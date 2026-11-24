// SPDX-License-Identifier: GPL-3.0-or-later
//! Relativistic Fermi-Dirac integrals and analytical derivatives via
//! Aparicio (1998, ApJS 117, 632) four-interval Gauss-Legendre and
//! Gauss-Laguerre quadrature.
//!
//! Ports upstream MESA `eos/private/gauss_fermi.f90` (`dfermi`, `fdfunc1`,
//! `fdfunc2`, `dqleg020`, `dqlag020`) and `eos_fermi_dirac_integral` in
//! `eos/public/eos_lib.f90`.

use flexmesa_autodiff::AutoDiffReal2VarOrder1;

/// Positive abscissae of the 20-point Gauss-Legendre quadrature rule on
/// `(-1, 1)` (`eos/private/gauss_fermi.f90:350-359`).
#[allow(clippy::excessive_precision)]
pub const DQLEG020_XG: [f64; 10] = [
    7.65265211334973337546404093988382110e-2,
    2.27785851141645078080496195368574624e-1,
    3.73706088715419560672548177024927237e-1,
    5.10867001950827098004364050955250998e-1,
    6.36053680726515025452836696226285936e-1,
    7.46331906460150792614305070355641590e-1,
    8.39116971822218823394529061701520685e-1,
    9.12234428251325905867752441203298113e-1,
    9.63971927277913791267666131197277221e-1,
    9.93128599185094924786122388471320278e-1,
];

/// Weights of the 20-point Gauss-Legendre quadrature rule on `(-1, 1)`
/// (`eos/private/gauss_fermi.f90:361-370`).
#[allow(clippy::excessive_precision)]
pub const DQLEG020_WG: [f64; 10] = [
    1.52753387130725850698084331955097593e-1,
    1.49172986472603746787828737001969436e-1,
    1.42096109318382051329298325067164933e-1,
    1.31688638449176626898494499748163134e-1,
    1.18194531961518417312377377711382287e-1,
    1.01930119817240435036750135480349876e-1,
    8.32767415767047487247581432220462061e-2,
    6.26720483341090635695065351870416063e-2,
    4.06014298003869413310399522749321098e-2,
    1.76140071391521183118619623518528163e-2,
];

/// Abscissae of the 20-point Gauss-Laguerre quadrature rule on `(0, +inf)`
/// (`eos/private/gauss_fermi.f90:773-792`).
#[allow(clippy::excessive_precision)]
pub const DQLAG020_XG: [f64; 20] = [
    7.05398896919887533666890045842150958e-2,
    3.72126818001611443794241388761146636e-1,
    9.16582102483273564667716277074183187e-1,
    1.70730653102834388068768966741305070e0,
    2.74919925530943212964503046049481338e0,
    4.04892531385088692237495336913333219e0,
    5.61517497086161651410453988565189234e0,
    7.45901745367106330976886021837181759e0,
    9.59439286958109677247367273428279837e0,
    1.20388025469643163096234092988655158e1,
    1.48142934426307399785126797100479756e1,
    1.79488955205193760173657909926125096e1,
    2.14787882402850109757351703695946692e1,
    2.54517027931869055035186774846415418e1,
    2.99325546317006120067136561351658232e1,
    3.50134342404790000062849359066881395e1,
    4.08330570567285710620295677078075526e1,
    4.76199940473465021399416271528511211e1,
    5.58107957500638988907507734444972356e1,
    6.65244165256157538186403187914606659e1,
];

/// Modified weights (`exp(xg) * w_laguerre`) of the 20-point Gauss-Laguerre
/// quadrature rule (`eos/private/gauss_fermi.f90:794-813`).
#[allow(clippy::excessive_precision)]
pub const DQLAG020_WG: [f64; 20] = [
    1.81080062418989255451675405913110644e-1,
    4.22556767878563974520344172566458197e-1,
    6.66909546701848150373482114992515927e-1,
    9.15352372783073672670604684771868067e-1,
    1.16953970719554597380147822239577476e0,
    1.43135498592820598636844994891514331e0,
    1.70298113798502272402533261633206720e0,
    1.98701589079274721410921839275129020e0,
    2.28663578125343078546222854681495651e0,
    2.60583472755383333269498950954033323e0,
    2.94978373421395086600235416827285951e0,
    3.32539578200931955236951937421751118e0,
    3.74225547058981092111707293265377811e0,
    4.21423671025188041986808063782478746e0,
    4.76251846149020929695292197839096371e0,
    5.42172604424557430380308297989981779e0,
    6.25401235693242129289518490300707542e0,
    7.38731438905443455194030019196464791e0,
    9.15132873098747960794348242552950528e0,
    1.28933886459399966710262871287485278e1,
];

/// Evaluates the untransformed Fermi-Dirac integrand `fdfunc1` and its partial
/// derivatives with respect to $\eta$ and $\theta$.
#[inline]
fn fdfunc1(x: f64, dk: f64, eta: f64, theta: f64) -> (f64, f64, f64) {
    let xdk = flexmesa_math::pow(x, dk);
    let xdkp1 = x * xdk;
    let dxst = (1.0 + 0.5 * x * theta).sqrt();

    if (x - eta) < 100.0 {
        let factor = flexmesa_math::exp(x - eta);
        let denom = factor + 1.0;
        let fd = xdk * dxst / denom;
        let fdeta = fd * factor / denom;
        let denom2 = 4.0 * dxst * denom;
        let fdtheta = xdkp1 / denom2;
        (fd, fdeta, fdtheta)
    } else {
        let factor = flexmesa_math::exp(eta - x);
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
    let xdk = flexmesa_math::pow(x, 2.0 * dk + 1.0);
    let xdkp1 = xsq * xdk;
    let dxst = (1.0 + 0.5 * xsq * theta).sqrt();

    if (xsq - eta) < 100.0 {
        let factor = flexmesa_math::exp(xsq - eta);
        let denom = factor + 1.0;
        let fd = 2.0 * xdk * dxst / denom;
        let fdeta = fd * factor / denom;
        let denom2 = 4.0 * dxst * denom;
        let fdtheta = 2.0 * xdkp1 / denom2;
        (fd, fdeta, fdtheta)
    } else {
        let factor = flexmesa_math::exp(eta - xsq);
        let fd = 2.0 * xdk * dxst * factor;
        let fdeta = fd;
        let denom2 = 4.0 * dxst;
        let fdtheta = (2.0 * xdkp1 / denom2) * factor;
        (fd, fdeta, fdtheta)
    }
}

/// Applies the 20-point Gauss-Legendre rule `dqleg020` on `[a, b]`
/// (`eos/private/gauss_fermi.f90:381-397`).
#[inline]
#[allow(clippy::manual_midpoint)]
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
    let center = 0.5 * (a + b);
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
        flexmesa_math::log(1.0 + flexmesa_math::exp(eta1)) / SG
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
