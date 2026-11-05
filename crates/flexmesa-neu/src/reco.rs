// SPDX-License-Identifier: LGPL-3.0-or-later
//! Recombination (`reco_neu`) neutrino energy loss rates and Antia (1993)
//! Fermi-Dirac rational approximations (`ifermi12`, `zfermim12`).
//!
//! Ports `ifermi12`, `zfermim12` (Antia 1993, ApJS, 84, 101), and `reco_neu`
//! (Itoh et al. 1996, ApJS, 102, 411, §6) from `neu/private/mod_neu.f90`.
//!
//! # Fermi-Dirac Inversion & Recombination Rate
//!
//! - [`ifermi12`] evaluates the inverse Fermi-Dirac integral $F_{1/2}^{-1}(f)$
//!   to relative accuracy $4.19 \times 10^{-9}$ across non-degenerate ($f < 4$)
//!   and degenerate ($f \ge 4$) regimes using Antia (1993, ApJS, 84, 101)
//!   rational minmax approximations:
//!   - For $f < 4$: $\nu(f) = \ln\!\left(f\,P_4(f) / Q_3(f)\right)$.
//!   - For $f \ge 4$: $\nu(f) = f^{2/3}\,P_6(f^{-2/3}) / Q_5(f^{-2/3})$
//!     (evaluated via Horner's rule in decreasing degree order to match
//!     upstream Fortran evaluation bit-for-bit).
//! - [`zfermim12`] evaluates $F_{-1/2}(\nu) = 2\,dF_{1/2}/d\nu$ to relative
//!   accuracy $1.23 \times 10^{-12}$ across $\nu < 2$ and $\nu \ge 2$, enabling
//!   exact analytical differentiation of $\nu(f)$ via the inverse function rule
//!   $d\nu / df = \left[\tfrac{1}{2} F_{-1/2}(\nu)\right]^{-1}$.
//! - [`reco_neu`] evaluates free-bound electron capture neutrino emission over
//!   electron degeneracy parameter $\nu \in [-20, 10]$ (Itoh et al. 1996, §6):
//!   $$Q_{\mathrm{reco}} = \left[(C_V^2 + \tfrac{3}{2} C_A^2) + n_\nu (C_V'^2 +
//!   \tfrac{3}{2} C_A'^2)\right] 2.649 \times 10^{-18}\,Y_e\,\bar{Z}^{13}
//!   \rho\,\frac{J(\zeta_{\mathrm{rec}}, \nu)}{1 + e^{\zeta_{\mathrm{rec}} +
//!   \nu}}$$
//!   where $\zeta_{\mathrm{rec}} = 1.579 \times 10^5\,\bar{Z}^2 / T$ is the
//!   K-shell ionization potential in units of $k_B T$.

use crate::inputs::TFAC6;
use crate::{NeuInputs, NeuRate};
use flexmesa_math::{exp, log, pow};

/// Inverse Fermi-Dirac integral $F_{1/2}^{-1}(f)$ via Antia (1993, ApJS, 84,
/// 101) rational approximation (`ifermi12` in `mod_neu.f90`, max error
/// $4.19 \times 10^{-9}$).
///
/// # Arguments
///
/// * `f` - Dimensionless Fermi-Dirac integral value $F_{1/2}(\nu) > 0$.
///
/// # Returns
///
/// Dimensionless electron chemical potential $\nu = \mu_e / (k_B T)$ (excluding
/// rest mass).
#[must_use]
pub fn ifermi12(f: f64) -> f64 {
    const A1: [f64; 5] = [
        1.999_266_880_833e4,
        5.702_479_099_336e3,
        6.610_132_843_877e2,
        3.818_838_129_486e1,
        1.0,
    ];
    const B1: [f64; 4] = [
        1.771_804_140_488e4,
        -2.014_785_161_019e3,
        9.130_355_392_717e1,
        -1.670_718_177_489,
    ];
    const A2: [f64; 7] = [
        -1.277_060_388_085e-2,
        7.187_946_804_945e-2,
        -4.262_314_235_106e-1,
        4.997_559_426_872e-1,
        -1.285_579_118_012,
        -3.930_805_454_272e-1,
        1.0,
    ];
    const B2: [f64; 6] = [
        -9.745_794_806_288e-3,
        5.485_432_756_838e-2,
        -3.299_466_243_260e-1,
        4.077_841_975_923e-1,
        -1.145_531_476_975,
        -6.067_091_689_181e-2,
    ];
    if f < 4.0 {
        let mut rn = f + A1[3];
        for i in (0..3).rev() {
            rn = rn * f + A1[i];
        }
        let mut den = B1[3];
        for i in (0..3).rev() {
            den = den * f + B1[i];
        }
        log(f * rn / den)
    } else {
        let ff = 1.0 / pow(f, 1.0 / 1.5);
        let mut rn = ff + A2[5];
        for i in (0..5).rev() {
            rn = rn * ff + A2[i];
        }
        let mut den = B2[5];
        for i in (0..5).rev() {
            den = den * ff + B2[i];
        }
        rn / (den * ff)
    }
}

/// Fermi-Dirac integral $F_{-1/2}(x)$ via Antia (1993, ApJS, 84, 101) rational
/// approximation (`zfermim12` in `mod_neu.f90`, max error $1.23 \times
/// 10^{-12}$).
#[must_use]
pub fn zfermim12(x: f64) -> f64 {
    const A1: [f64; 8] = [
        1.714_463_747_044_54e7,
        3.881_483_023_240_68e7,
        3.167_433_853_049_62e7,
        1.145_876_091_921_51e7,
        1.836_963_707_561_53e6,
        1.149_809_981_868_74e5,
        1.982_768_899_247_68e3,
        1.0,
    ];
    const B1: [f64; 8] = [
        9.672_825_874_528_99e6,
        2.873_864_367_317_85e7,
        3.260_701_307_341_58e7,
        1.776_570_278_463_67e7,
        4.816_480_222_678_31e6,
        6.137_095_693_332_07e5,
        3.135_958_543_321_14e4,
        4.350_617_250_807_55e2,
    ];
    const A2: [f64; 12] = [
        -4.466_203_419_249_42e-15,
        -1.586_549_911_462_36e-12,
        -4.444_676_270_422_32e-10,
        -6.847_387_916_217_45e-8,
        -6.649_322_385_281_05e-6,
        -3.699_761_701_939_42e-4,
        -1.122_953_936_870_06e-2,
        -1.609_261_021_244_42e-1,
        -8.524_086_128_774_47e-1,
        -7.455_199_537_639_28e-1,
        2.984_352_074_663_72,
        1.0,
    ];
    const B2: [f64; 12] = [
        -2.233_101_709_623_69e-15,
        -7.941_932_820_714_64e-13,
        -2.225_643_769_562_28e-10,
        -3.432_994_310_798_45e-8,
        -3.339_196_126_789_07e-6,
        -1.864_322_121_870_88e-4,
        -5.697_644_368_805_29e-3,
        -8.349_045_930_671_94e-2,
        -4.787_708_440_094_40e-1,
        -4.997_592_503_741_48e-1,
        1.867_959_649_930_52,
        4.164_859_704_952_88e-1,
    ];
    if x < 2.0 {
        let xx = exp(x);
        let mut rn = xx + A1[6];
        for i in (0..6).rev() {
            rn = rn * xx + A1[i];
        }
        let mut den = B1[7];
        for i in (0..7).rev() {
            den = den * xx + B1[i];
        }
        xx * rn / den
    } else {
        let xx = 1.0 / (x * x);
        let mut rn = xx + A2[10];
        for i in (0..10).rev() {
            rn = rn * xx + A2[i];
        }
        let mut den = B2[11];
        for i in (0..11).rev() {
            den = den * xx + B2[i];
        }
        x.sqrt() * rn / den
    }
}

/// Evaluates volumetric recombination neutrino energy loss rate and partial
/// derivatives (`reco_neu` in `mod_neu.f90`, in $\mathrm{erg\,cm^{-3}\,s^{-1}}$
/// before dividing by $\rho$).
///
/// # Arguments
///
/// * `input` - Precomputed thermodynamic and composition invariants
///   ([`NeuInputs`]).
///
/// # Returns
///
/// Volumetric rate [`NeuRate`] ($Q_{\mathrm{reco}}$ and partial derivatives
/// with respect to $T, \rho, \bar{A}, \bar{Z}$), or [`NeuRate::ZERO`] when
/// $\nu \notin [-20, 10]$.
#[must_use]
pub fn reco_neu(input: &NeuInputs) -> NeuRate {
    let xnum =
        1.10520e8 * input.den * input.ye / (input.temp * input.temp.sqrt());
    let xnumdt = -1.50 * xnum * input.tempi;
    let xnumdd = xnum * input.deni;
    let xnumda = -xnum * input.abari;
    let xnumdz = xnum * input.zbari;

    let nu = ifermi12(xnum);
    if !(-20.0..=10.0).contains(&nu) {
        return NeuRate::ZERO;
    }
    let a0 = 1.0 / (0.5 * zfermim12(nu));
    let nudt = a0 * xnumdt;
    let nudd = a0 * xnumdd;
    let nuda = a0 * xnumda;
    let nudz = a0 * xnumdz;
    let nu2 = nu * nu;
    let nu3 = nu2 * nu;

    let (a1, a2, a3, b, c, d, f1, f2, f3) = if nu < 0.0 {
        (1.51e-2, 2.42e-1, 1.21, 3.71e-2, 9.06e-1, 9.28e-1, 0.0, 0.0, 0.0)
    } else {
        (
            1.23e-2, 2.66e-1, 1.30, 1.17e-1, 8.97e-1, 1.77e-1, -1.20e-2,
            2.29e-2, -1.04e-3,
        )
    };

    let zeta = 1.579e5 * input.zbar * input.zbar * input.tempi;
    let zetadt = -zeta * input.tempi;
    let zetadz = 2.0 * zeta * input.zbari;

    let c00_p = 1.0 / (1.0 + f1 * nu + f2 * nu2 + f3 * nu3);
    let c01_p = f1 + f2 * 2.0 * nu + f3 * 3.0 * nu2;
    let dum = zeta * c00_p;
    let k_nu = -c00_p * c00_p * zeta * c01_p;
    let dumdt = zetadt * c00_p + k_nu * nudt;
    let dumdd = k_nu * nudd;
    let dumda = k_nu * nuda;
    let dumdz = zetadz * c00_p + k_nu * nudz;

    let inv_dum = 1.0 / dum;
    let dd00 = pow(dum, -2.25);
    let dd01 = pow(dum, -4.55);
    let c00 = a1 * inv_dum + a2 * dd00 + a3 * dd01;
    let c01 = -(a1 * inv_dum + 2.25 * a2 * dd00 + 4.55 * a3 * dd01) * inv_dum;

    let z_cnu = exp(c * nu);
    let dd00_g = b * z_cnu * (1.0 + d * dum);
    let gum = 1.0 + dd00_g;
    let gumdt = dd00_g * c * nudt + b * z_cnu * d * dumdt;
    let gumdd = dd00_g * c * nudd + b * z_cnu * d * dumdd;
    let gumda = dd00_g * c * nuda + b * z_cnu * d * dumda;
    let gumdz = dd00_g * c * nudz + b * z_cnu * d * dumdz;

    let z_nu = exp(nu);
    let inv_gum = 1.0 / gum;
    let bigj = c00 * z_nu * inv_gum;
    let bigjdt = (c01 * dumdt + c00 * nudt - c00 * inv_gum * gumdt)
        * z_nu
        * inv_gum;
    let bigjdd = (c01 * dumdd + c00 * nudd - c00 * inv_gum * gumdd)
        * z_nu
        * inv_gum;
    let bigjda = (c01 * dumda + c00 * nuda - c00 * inv_gum * gumda)
        * z_nu
        * inv_gum;
    let bigjdz = (c01 * dumdz + c00 * nudz - c00 * inv_gum * gumdz)
        * z_nu
        * inv_gum;

    let z_zn = exp(zeta + nu);
    let inv_1pz = 1.0 / (1.0 + z_zn);
    let inv_bigj = 1.0 / bigj;
    let sreco = TFAC6
        * 2.649e-18
        * input.ye
        * pow(input.zbar, 13.0)
        * input.den
        * bigj
        * inv_1pz;
    NeuRate {
        val: sreco,
        d_t: sreco * (bigjdt * inv_bigj - z_zn * (zetadt + nudt) * inv_1pz),
        d_rho: sreco * (input.deni + bigjdd * inv_bigj - z_zn * nudd * inv_1pz),
        d_abar: sreco
            * (-input.abari + bigjda * inv_bigj - z_zn * nuda * inv_1pz),
        d_zbar: sreco
            * (14.0 * input.zbari + bigjdz * inv_bigj
                - z_zn * (zetadz + nudz) * inv_1pz),
    }
}
