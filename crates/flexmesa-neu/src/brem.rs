// SPDX-License-Identifier: LGPL-3.0-or-later
//! Bremsstrahlung (`brem_neu`) neutrino energy loss rates and derivatives.
//!
//! Ports `brem_neu_weak_degen`, `brem_neu_liquid_metal`, and `brem_neu` (Itoh
//! et al. 1996, ApJS, 102, 411, §5) from `neu/private/mod_neu.f90`.
//!
//! # Degeneracy Blending Strategy
//!
//! - For $T \ge 0.3\,T_F$, evaluates the weakly degenerate electron gas fit
//!   (`brem_neu_weak_degen`, eqs. 5.1–5.9):
//!   $$Q_{\mathrm{brem,weak}} = 0.5738\,\bar{Z}\,Y_e\,T_8^6\,\rho
//!   \left(C_+^2 F_{\mathrm{brem}} - C_-^2 G_{\mathrm{brem}}\right)$$
//!   where $F_{\mathrm{brem}}(T_8, \eta)$ and
//!   $G_{\mathrm{brem}}(T_8, \rho/\mu_e)$ rational-polynomial approximations
//!   account for electron-ion screening and relativistic kinematics.
//! - For $T \le 0.05\,T_F$, evaluates the strongly degenerate liquid-metal
//!   fit (`brem_neu_liquid_metal`, eqs. 5.16–5.26), combining 5-harmonic
//!   Fourier expansions in $u = (\pi/5)(\log_{10}\rho - 3)$ for $F_b, F_t, G_b,
//!   G_t$ with Coulomb coupling polynomials $v(\Gamma^{-1/3})$ and
//!   $w(\Gamma^{-1/3})$ where $\Gamma = 0.2275\,\bar{Z}^2 T_8^{-1}
//!   (\rho_6 / \bar{A})^{1/3}$.
//! - For $0.05\,T_F < T < 0.3\,T_F$, blends both regimes via the smooth cosine
//!   ramp $\alpha = \tfrac{1}{2}(1 - \cos(\pi\,\mathrm{tfrac}))$ where
//!   $\mathrm{tfrac} = (T / T_F - 0.05) / 0.25$ and $T_F = 5.9302 \times
//!   10^9\,(\sqrt{u} - 1)$ with $u = (\rho_6 Y_e)^{2/3}$, propagating exact
//!   chain-rule partial derivatives with respect to
//!   $(T, \rho, \bar{A}, \bar{Z})$.

use crate::inputs::{TFAC4, TFAC5};
use crate::{NeuInputs, NeuRate};
use flexmesa_const::{ILN10, ONE_THIRD, PI, TWO_THIRDS};
use flexmesa_math::{cos, cospi, pow, sin, sinpi};

/// Angular scale $\pi / 5$ (`fac3` in `mod_neu.f90`).
const FAC3: f64 = PI / 5.0;

/// Weakly degenerate bremsstrahlung fit (`brem_neu_weak_degen` in
/// `mod_neu.f90`, Itoh et al. 1996, eqs. 5.1–5.9).
///
/// Evaluates the rational fits $F_{\mathrm{brem}}(T_8, \eta)$ (eq. 5.2) and
/// $G_{\mathrm{brem}}(T_8, \rho, \mu_e)$ (eq. 5.9) and combines them with the
/// weak-coupling weights `TFAC4` and `TFAC5`.
fn brem_neu_weak_degen(t8: f64, input: &NeuInputs) -> NeuRate {
    let t812 = t8.sqrt();
    let t832 = t8 * t812;
    let t82 = t8 * t8;
    let t83 = t82 * t8;
    let t85 = t82 * t83;
    let t86 = t85 * t8;
    let t8m1 = 1.0 / t8;
    let t8m2 = t8m1 * t8m1;
    let t8m3 = t8m2 * t8m1;
    let t8m5 = t8m3 * t8m2;
    let t8m6 = t8m5 * t8m1;

    // Equation 5.3: eta
    let dum = 7.05e6 * t832 + 5.12e4 * t83;
    let dumdt = (1.5 * 7.05e6 * t812 + 3.0 * 5.12e4 * t82) * 1.0e-8;
    let z = 1.0 / dum;
    let eta = input.rm * z;
    let etadt = -input.rm * z * z * dumdt;
    let etadd = input.rmdd * z;
    let etada = input.rmda * z;
    let etadz = input.rmdz * z;
    let etam1 = 1.0 / eta;
    let etam2 = etam1 * etam1;
    let etam3 = etam2 * etam1;

    // Equation 5.2: fbrem
    let a0 = 23.5 + 6.83e4 * t8m2 + 7.81e8 * t8m5;
    let f0 = (-2.0 * 6.83e4 * t8m3 - 5.0 * 7.81e8 * t8m6) * 1.0e-8;
    let xnum = 1.0 / a0;
    let edum = 1.0 + 1.47 * etam1 + 3.29e-2 * etam2;
    let dz = -1.47 * etam2 - 2.0 * 3.29e-2 * etam3;
    let c00 = 1.26 * (1.0 + etam1);
    let cz = -1.26 * etam2;
    let inv_edum = 1.0 / edum;
    let xden = c00 * inv_edum;
    let xdendt = (cz * etadt - xden * dz * etadt) * inv_edum;
    let xdendd = (cz * etadd - xden * dz * etadd) * inv_edum;
    let xdenda = (cz * etada - xden * dz * etada) * inv_edum;
    let xdendz = (cz * etadz - xden * dz * etadz) * inv_edum;
    let fbrem = xnum + xden;
    let fbremdt = -xnum * xnum * f0 + xdendt;
    let fbremdd = xdendd;
    let fbremda = xdenda;
    let fbremdz = xdendz;

    // Equation 5.9: gbrem
    let ga0 = 230.0 + 6.7e5 * t8m2 + 7.66e9 * t8m5;
    let gf0 = (-2.0 * 6.7e5 * t8m3 - 5.0 * 7.66e9 * t8m6) * 1.0e-8;
    let gz = 1.0 + input.rm * 1.0e-9;
    let gdum = ga0 * gz;
    let gdumdt = gf0 * gz;
    let g_scale = ga0 * 1.0e-9;
    let gxnum = 1.0 / gdum;
    let neg_gx2 = -gxnum * gxnum;
    let gxnumdt = neg_gx2 * gdumdt;
    let gxnumdd = neg_gx2 * g_scale * input.rmdd;
    let gxnumda = neg_gx2 * g_scale * input.rmda;
    let gxnumdz = neg_gx2 * g_scale * input.rmdz;

    let p1 = pow(t8, 3.85);
    let gc00 = 7.75e5 * t832 + 247.0 * p1;
    let gdd00 = (1.5 * 7.75e5 * t812 + 3.85 * 247.0 * p1 / t8) * 1.0e-8;
    let p2 = pow(t8, 1.4);
    let gc01 = 4.07 + 0.0240 * p2;
    let gdd01 = 1.4 * 0.0240 * (p2 / t8) * 1.0e-8;
    let p3 = pow(t8, -0.110);
    let gc02 = 4.59e-5 * p3;
    let gdd02 = -0.11 * 4.59e-5 * (p3 / t8) * 1.0e-8;

    let den_p = pow(input.den, 0.656);
    let gd2 = gc00 * input.rmi + gc01 + gc02 * den_p;
    let gd2dt = gdd00 * input.rmi + gdd01 + gdd02 * den_p;
    let neg_c00_rmi2 = -gc00 * input.rmi * input.rmi;
    let gd2dd =
        neg_c00_rmi2 * input.rmdd + 0.656 * gc02 * pow(input.den, -0.344);
    let gd2da = neg_c00_rmi2 * input.rmda;
    let gd2dz = neg_c00_rmi2 * input.rmdz;
    let gxden = 1.0 / gd2;
    let neg_gxd2 = -gxden * gxden;

    let gbrem = gxnum + gxden;
    let gbremdt = gxnumdt + neg_gxd2 * gd2dt;
    let gbremdd = gxnumdd + neg_gxd2 * gd2dd;
    let gbremda = gxnumda + neg_gxd2 * gd2da;
    let gbremdz = gxnumdz + neg_gxd2 * gd2dz;

    // Itoh et al. (1996) eq. 5.1: combine weak-coupling weights TFAC4 and
    // TFAC5 with the density/composition prefactor 0.5738 * Z * Y_e * T_8^6 *
    // rho and apply the product rule for all four partial derivatives.
    let pref = 0.5738 * input.zbar * input.ye * t86 * input.den;
    let prefdt =
        0.5738 * input.zbar * input.ye * 6.0 * t85 * input.den * 1.0e-8;
    let prefdd = 0.5738 * input.zbar * input.ye * t86;
    let prefda = -pref * input.abari;
    let prefdz = 0.5738 * 2.0 * input.ye * t86 * input.den;
    let comb = TFAC4 * fbrem - TFAC5 * gbrem;
    NeuRate {
        val: pref * comb,
        d_t: prefdt * comb + pref * (TFAC4 * fbremdt - TFAC5 * gbremdt),
        d_rho: prefdd * comb + pref * (TFAC4 * fbremdd - TFAC5 * gbremdd),
        d_abar: prefda * comb + pref * (TFAC4 * fbremda - TFAC5 * gbremda),
        d_zbar: prefdz * comb + pref * (TFAC4 * fbremdz - TFAC5 * gbremdz),
    }
}

/// Evaluates one 5-harmonic liquid-metal Fourier structure function and its
/// density derivative (`brem_neu_liquid_metal` in `mod_neu.f90`, Itoh et al.
/// 1996, eqs. 5.17–5.18).
#[inline]
fn eval_liq_fourier(
    u: f64,
    a0_u: f64,
    cos_h: [f64; 5],
    sin_h: [f64; 5],
    c0: f64,
    cu: f64,
    cc: f64,
    a_cos: [f64; 5],
    b_sin: [f64; 4],
) -> (f64, f64) {
    let val = 0.5 * c0
        + cu * u
        + cc
        + a_cos[0] * cos_h[0]
        + b_sin[0] * sin_h[0]
        + a_cos[1] * cos_h[1]
        + b_sin[1] * sin_h[1]
        + a_cos[2] * cos_h[2]
        + b_sin[2] * sin_h[2]
        + a_cos[3] * cos_h[3]
        + b_sin[3] * sin_h[3]
        + a_cos[4] * cos_h[4];
    let dval = a0_u
        * (cu - a_cos[0] * sin_h[0] + b_sin[0] * cos_h[0]
            - a_cos[1] * sin_h[1] * 2.0
            + b_sin[1] * cos_h[1] * 2.0
            - a_cos[2] * sin_h[2] * 3.0
            + b_sin[2] * cos_h[2] * 3.0
            - a_cos[3] * sin_h[3] * 4.0
            + b_sin[3] * cos_h[3] * 4.0
            - a_cos[4] * sin_h[4] * 5.0);
    (val, dval)
}

/// Strongly degenerate liquid-metal bremsstrahlung fit (`brem_neu_liquid_metal`
/// in `mod_neu.f90`, Itoh et al. 1996, eqs. 5.16–5.26).
///
/// Evaluates the 5-harmonic Chebyshev trigonometric series in
/// $u = (\pi/5)(\log_{10}\rho - 3)$ for the four structure functions
/// $(F_b, F_t, G_b, G_t)$ and their analytical density derivatives
/// $(c_{00}, c_{01}, c_{02}, c_{03})$, then weights them by the ion-coupling
/// polynomials $v(\Gamma)$ and $w(\Gamma)$.
fn brem_neu_liquid_metal(t8: f64, input: &NeuInputs) -> NeuRate {
    // Powers of T_8 = T / 10^8 K
    let t82 = t8 * t8;
    let t83 = t82 * t8;
    let t85 = t82 * t83;
    let t86 = t85 * t8;
    let t8m1 = 1.0 / t8;

    // Itoh et al. (1996) eqs. 5.17–5.18: 5-harmonic Chebyshev recurrence for
    // u = (pi / 5) * (log10(rho) - 3) and its density derivative du/drho.
    let u = FAC3 * (input.logden - 3.0);
    let a0_u = ILN10 * FAC3 * input.deni;
    let cos1 = cos(u);
    let sin1 = sin(u);
    let sin2 = 2.0 * sin1 * cos1;
    let cos2 = 2.0 * cos1 * cos1 - 1.0;
    let sin3 = sin1 * (3.0 - 4.0 * sin1 * sin1);
    let cos3 = cos1 * (4.0 * cos1 * cos1 - 3.0);
    let sin4 = 2.0 * sin2 * cos2;
    let cos4 = 2.0 * cos2 * cos2 - 1.0;
    let sin5 = sin1 * (5.0 - sin1 * sin1 * (20.0 - 16.0 * sin1 * sin1));
    let cos5 = cos1 * (cos1 * cos1 * (16.0 * cos1 * cos1 - 20.0) + 5.0);
    let cos_h = [cos1, cos2, cos3, cos4, cos5];
    let sin_h = [sin1, sin2, sin3, sin4, sin5];

    let (fb, c00) = eval_liq_fourier(
        u,
        a0_u,
        cos_h,
        sin_h,
        0.17946,
        0.00945,
        0.34529,
        [-0.05821, -0.01089, -0.01147, -0.00656, -0.00519],
        [-0.04969, -0.01584, -0.00504, -0.00281],
    );
    let (ft, c01) = eval_liq_fourier(
        u,
        a0_u,
        cos_h,
        sin_h,
        0.06781,
        -0.02342,
        0.24819,
        [-0.00944, -0.01289, -0.00589, -0.00404, -0.00330],
        [-0.02213, -0.01136, -0.00467, -0.00131],
    );
    let (gb, c02) = eval_liq_fourier(
        u,
        a0_u,
        cos_h,
        sin_h,
        0.00766,
        -0.01259,
        0.07917,
        [-0.00710, -0.00028, 0.00232, 0.00044, 0.00158],
        [0.02300, -0.01078, 0.00118, -0.00089],
    );
    let (gt, c03) = eval_liq_fourier(
        u,
        a0_u,
        cos_h,
        sin_h,
        -0.00769,
        -0.00829,
        0.05211,
        [0.00356, -0.00184, 0.00146, 0.00031, 0.00069],
        [0.01052, -0.00354, -0.00014, -0.00018],
    );

    let dum = 2.275e-1
        * input.zbar
        * input.zbar
        * t8m1
        * pow(input.den6 * input.abari, ONE_THIRD);
    let dumdt = -dum * input.tempi;
    let dumdd = ONE_THIRD * dum * input.deni;
    let dumda = -ONE_THIRD * dum * input.abari;
    let dumdz = 2.0 * dum * input.zbari;

    let gm1 = 1.0 / dum;
    let gm2 = gm1 * gm1;
    let gm13 = pow(gm1, ONE_THIRD);
    let gm23 = gm13 * gm13;
    let gm43 = gm13 * gm1;
    let gm53 = gm23 * gm1;

    // Itoh et al. (1996) eqs. 5.21–5.24: Coulomb coupling polynomials v(Gamma)
    // and w(Gamma) in powers of Gamma^{-1/3}, Gamma^{-2/3}, and Gamma^{-1}.
    let v = -0.05483 - 0.01946 * gm13 + 1.86310 * gm23 - 0.78873 * gm1;
    let a0 = ONE_THIRD * 0.01946 * gm43 - TWO_THIRDS * 1.86310 * gm53
        + 0.78873 * gm2;
    let w = -0.06711 + 0.06859 * gm13 + 1.74360 * gm23 - 0.74498 * gm1;
    let a1 = -ONE_THIRD * 0.06859 * gm43 - TWO_THIRDS * 1.74360 * gm53
        + 0.74498 * gm2;

    // Itoh et al. (1996) eqs. 5.19–5.20: interpolate between Born (fb, gb) and
    // strongly coupled liquid-metal (ft, gt) structure factors.
    let fliq = v * fb + (1.0 - v) * ft;
    let fliqdt = a0 * dumdt * (fb - ft);
    let fliqdd = a0 * dumdd * (fb - ft) + v * c00 + (1.0 - v) * c01;
    let fliqda = a0 * dumda * (fb - ft);
    let fliqdz = a0 * dumdz * (fb - ft);

    let gliq = w * gb + (1.0 - w) * gt;
    let gliqdt = a1 * dumdt * (gb - gt);
    let gliqdd = a1 * dumdd * (gb - gt) + w * c02 + (1.0 - w) * c03;
    let gliqda = a1 * dumda * (gb - gt);
    let gliqdz = a1 * dumdz * (gb - gt);

    // Itoh et al. (1996) eq. 5.16: combine strongly degenerate structure
    // factors fliq and gliq with the density/composition prefactor and
    // weak-coupling weights TFAC4 and TFAC5.
    let pref = 0.5738 * input.zbar * input.ye * t86 * input.den;
    let prefdt =
        0.5738 * input.zbar * input.ye * 6.0 * t85 * input.den * 1.0e-8;
    let prefdd = 0.5738 * input.zbar * input.ye * t86;
    let prefda = -pref * input.abari;
    let prefdz = 0.5738 * 2.0 * input.ye * t86 * input.den;
    let comb = TFAC4 * fliq - TFAC5 * gliq;
    NeuRate {
        val: pref * comb,
        d_t: prefdt * comb + pref * (TFAC4 * fliqdt - TFAC5 * gliqdt),
        d_rho: prefdd * comb + pref * (TFAC4 * fliqdd - TFAC5 * gliqdd),
        d_abar: prefda * comb + pref * (TFAC4 * fliqda - TFAC5 * gliqda),
        d_zbar: prefdz * comb + pref * (TFAC4 * fliqdz - TFAC5 * gliqdz),
    }
}

/// Evaluates volumetric bremsstrahlung neutrino energy loss rate and partial
/// derivatives (`brem_neu` in `mod_neu.f90`, in $\mathrm{erg\,cm^{-3}\,s^{-1}}$
/// before dividing by $\rho$).
///
/// # Arguments
///
/// * `input` - Precomputed thermodynamic and composition invariants
///   ([`NeuInputs`]).
///
/// # Returns
///
/// Volumetric rate [`NeuRate`] ($Q_{\mathrm{brem}}$ and partial derivatives
/// with respect to $T, \rho, \bar{A}, \bar{Z}$).
#[must_use]
pub fn brem_neu(input: &NeuInputs) -> NeuRate {
    const TFLO: f64 = 0.05;
    const TFHI: f64 = 0.3;
    const A_TF: f64 = 5.9302e9;

    let t8 = input.temp * 1.0e-8;
    let u = pow(input.den6 * input.ye, TWO_THIRDS);
    let tfermi = A_TF * (u.sqrt() - 1.0);
    if input.temp >= TFHI * tfermi {
        return brem_neu_weak_degen(t8, input);
    }
    if input.temp <= TFLO * tfermi {
        return brem_neu_liquid_metal(t8, input);
    }
    // Smooth cosine blending in the intermediate degeneracy window
    // 0.05 * T_F < T < 0.3 * T_F, propagating partial derivatives of tfrac =
    // (T / T_F - 0.05) / 0.25 with respect to (T, rho, abar, zbar).
    let weak = brem_neu_weak_degen(t8, input);
    let liq = brem_neu_liquid_metal(t8, input);
    let dtf = TFHI - TFLO;
    let tfrac = (input.temp / tfermi - TFLO) / dtf;
    let alfa = 0.5 * (1.0 - cospi(tfrac));
    let beta = 1.0 - alfa;

    let dtfermidu = 0.5 * A_TF * pow(u, -0.5);
    let dtfrac_dtf = -input.temp / (tfermi * tfermi * dtf);
    let dudv = TWO_THIRDS * pow(input.den6 * input.ye, -1.0 / 3.0);
    let dudd = dudv * 1.0e-6 * input.ye;
    let duda = -dudv * input.den6 * input.ye * input.abari;
    let dudz = dudv * input.den6 * input.abari;

    let dtfracdt = 1.0 / (tfermi * dtf);
    let dtfracdd = dtfermidu * dudd * dtfrac_dtf;
    let dtfracda = dtfermidu * duda * dtfrac_dtf;
    let dtfracdz = dtfermidu * dudz * dtfrac_dtf;

    let d_cos = 0.5 * PI * sinpi(tfrac);
    let dalfadt = dtfracdt * d_cos;
    let dalfadd = dtfracdd * d_cos;
    let dalfada = dtfracda * d_cos;
    let dalfadz = dtfracdz * d_cos;
    let diff_val = weak.val - liq.val;
    NeuRate {
        val: alfa * weak.val + beta * liq.val,
        d_t: alfa * weak.d_t + beta * liq.d_t + dalfadt * diff_val,
        d_rho: alfa * weak.d_rho + beta * liq.d_rho + dalfadd * diff_val,
        d_abar: alfa * weak.d_abar + beta * liq.d_abar + dalfada * diff_val,
        d_zbar: alfa * weak.d_zbar + beta * liq.d_zbar + dalfadz * diff_val,
    }
}
