// SPDX-License-Identifier: LGPL-3.0-or-later
//! Pair-production (`pair_neu`) and plasmon-decay (`plas_neu`) neutrino losses.
//!
//! Ports `pair_neu` (Itoh et al. 1996, ApJS, 102, 411, eqs. 2.1–2.8) and
//! `plas_neu` (eqs. 4.1, 4.5–4.11) from `neu/private/mod_neu.f90`.
//!
//! # Physical Regimes
//!
//! - **Pair Production (`pair_neu`)**: Dominant at high temperature ($T \gtrsim
//!   10^9\,\mathrm{K}$) and low-to-moderate density where thermal photons
//!   create $e^+ e^-$ pairs that annihilate into $\nu \bar{\nu}$ pairs:
//!   $$Q_{\mathrm{pair}} = \tfrac{1}{2}\left[(C_V^2 + C_A^2) + n_\nu (C_V'^2 +
//!   C_A'^2)\right]\left[1 + \frac{(C_V^2 - C_A^2) + n_\nu (C_V'^2 -
//!   C_A'^2)}{(C_V^2 + C_A^2) + n_\nu (C_V'^2 + C_A'^2)}\,q_{\mathrm{pair}}
//!   \right] g(\lambda)\,e^{-2/\lambda}\,f_{\mathrm{pair}}(\lambda, \xi)$$
//!   where $\lambda = T / (5.9302 \times 10^9\,\mathrm{K})$ and $\xi = (10^{-9}
//!   \rho Y_e)^{1/3} / \lambda$. Separates the $f_{\mathrm{pair}}$ fit
//!   coefficients at $T_9 = 10$ ($T = 10^{10}\,\mathrm{K}$).
//! - **Plasmon Decay (`plas_neu`)**: Dominant at high electron degeneracy where
//!   transverse (`ft`), longitudinal (`fl`), and mixed (`fxy`) collective
//!   plasma excitations decay into $\nu_e \bar{\nu}_e$ pairs:
//!   $$Q_{\mathrm{plas}} = C_V^2 \left(3.0 \times 10^{21}\,\lambda^9\right)
//!   \gamma^6 e^{-\gamma}\left(f_T + f_L\right) f_{xy}(x, y)$$
//!   where $\gamma^2 = 1.1095 \times 10^{11} (\rho Y_e) / \left[T^2 (1 +
//!   (1.019 \times 10^{-6} \rho Y_e)^{2/3})^{1/2}\right]$ is the squared
//!   dimensionless plasma frequency and $f_T(\gamma)$, $f_L(\gamma)$ are the
//!   transverse and longitudinal plasmon decay form factors.

use crate::inputs::{TFAC0, TFAC3, TFAC4};
use crate::{NeuInputs, NeuRate};
use flexmesa_const::{ILN10, ONE_SIXTH, TWO_THIRDS};
use flexmesa_math::{cos, exp, log10, pow, sin};

/// Evaluates volumetric pair-neutrino energy loss rate and partial derivatives
/// (`pair_neu` in `mod_neu.f90`, in $\mathrm{erg\,cm^{-3}\,s^{-1}}$ before
/// dividing by $\rho$).
///
/// # Arguments
///
/// * `input` - Precomputed thermodynamic and composition invariants
///   ([`NeuInputs`]).
///
/// # Returns
///
/// Volumetric rate [`NeuRate`] ($Q_{\mathrm{pair}}$ and partial derivatives
/// with respect to $T, \rho, \bar{A}, \bar{Z}$).
#[must_use]
pub fn pair_neu(input: &NeuInputs) -> NeuRate {
    // Itoh et al. (1996) equation 2.8: polynomial factor g(lambda)
    let gl = 1.0 - 13.04 * input.xl2
        + 133.5 * input.xl4
        + 1534.0 * input.xl6
        + 918.6 * input.xl8;
    let gldt = input.xldt
        * (-26.08 * input.xl
            + 534.0 * input.xl3
            + 9204.0 * input.xl5
            + 7348.8 * input.xl7);

    // Itoh et al. (1996) equation 2.7: rational-exponential factor f_pair
    let a1 = 6.002e19 + 2.084e20 * input.zeta + 1.872e21 * input.zeta2;
    let a2 = 2.084e20 + 2.0 * 1.872e21 * input.zeta;

    let (b1, b2) = if input.t9 < 10.0 {
        let b = exp(-5.5924 * input.zeta);
        (b, -b * 5.5924)
    } else {
        let b = exp(-4.9924 * input.zeta);
        (b, -b * 4.9924)
    };

    let xnum = a1 * b1;
    let c = a2 * b1 + a1 * b2;
    let xnumdt = c * input.zetadt;
    let xnumdd = c * input.zetadd;
    let xnumda = c * input.zetada;
    let xnumdz = c * input.zetadz;

    let (den_a1, den_a2) = if input.t9 < 10.0 {
        (
            9.383e-1 * input.xlm1 - 4.141e-1 * input.xlm2
                + 5.829e-2 * input.xlm3,
            -9.383e-1 * input.xlm2 + 2.0 * 4.141e-1 * input.xlm3
                - 3.0 * 5.829e-2 * input.xlm4,
        )
    } else {
        (
            1.2383 * input.xlm1 - 8.141e-1 * input.xlm2,
            -1.2383 * input.xlm2 + 2.0 * 8.141e-1 * input.xlm3,
        )
    };

    let b1_z = 3.0 * input.zeta2;
    let xden = input.zeta3 + den_a1;
    let xdendt = b1_z * input.zetadt + den_a2 * input.xldt;
    let xdendd = b1_z * input.zetadd;
    let xdenda = b1_z * input.zetada;
    let xdendz = b1_z * input.zetadz;

    let inv_xden = 1.0 / xden;
    let fpair = xnum * inv_xden;
    let fpairdt = (xnumdt - fpair * xdendt) * inv_xden;
    let fpairdd = (xnumdd - fpair * xdendd) * inv_xden;
    let fpairda = (xnumda - fpair * xdenda) * inv_xden;
    let fpairdz = (xnumdz - fpair * xdendz) * inv_xden;

    // Itoh et al. (1996) equation 2.6: q_pair factor
    let q_a1 = 10.7480 * input.xl2 + 0.3967 * input.xlp5 + 1.005;
    let q_a2 =
        input.xldt * (2.0 * 10.7480 * input.xl + 0.5 * 0.3967 * input.xlmp5);
    let q_xnum = 1.0 / q_a1;
    let q_xnumdt = -q_xnum * q_xnum * q_a2;

    let d_a1 = 7.692e7 * input.xl3 + 9.715e6 * input.xlp5;
    let d_a2 =
        input.xldt * (3.0 * 7.692e7 * input.xl2 + 0.5 * 9.715e6 * input.xlmp5);
    let d_c = 1.0 / d_a1;
    let d_b1 = 1.0 + input.rm * d_c;
    let q_xden = pow(d_b1, -0.3);
    let d_fac = -0.3 * q_xden / d_b1;
    let q_xdendt = -d_fac * input.rm * d_c * d_c * d_a2;
    let q_xdendd = d_fac * input.rmdd * d_c;
    let q_xdenda = d_fac * input.rmda * d_c;
    let q_xdendz = d_fac * input.rmdz * d_c;

    let qpair = q_xnum * q_xden;
    let qpairdt = q_xnumdt * q_xden + q_xnum * q_xdendt;
    let qpairdd = q_xnum * q_xdendd;
    let qpairda = q_xnum * q_xdenda;
    let qpairdz = q_xnum * q_xdendz;

    // Itoh et al. (1996) equation 2.5: combine exponential, gl, fpair, qpair
    let e_a1 = exp(-2.0 * input.xlm1);
    let e_a2 = e_a1 * 2.0 * input.xlm2 * input.xldt;

    let mut spair = e_a1 * fpair;
    let mut spairdt = e_a2 * fpair + e_a1 * fpairdt;
    let mut spairdd = e_a1 * fpairdd;
    let mut spairda = e_a1 * fpairda;
    let mut spairdz = e_a1 * fpairdz;

    let s_prev = spair;
    spair = gl * s_prev;
    spairdt = gl * spairdt + gldt * s_prev;
    spairdd *= gl;
    spairda *= gl;
    spairdz *= gl;

    let w1 = TFAC4 * (1.0 + TFAC3 * qpair);
    let w2 = TFAC4 * TFAC3;
    let s_gl = spair;
    NeuRate {
        val: w1 * s_gl,
        d_t: w1 * spairdt + w2 * qpairdt * s_gl,
        d_rho: w1 * spairdd + w2 * qpairdd * s_gl,
        d_abar: w1 * spairda + w2 * qpairda * s_gl,
        d_zbar: w1 * spairdz + w2 * qpairdz * s_gl,
    }
}

/// Evaluates volumetric plasmon-decay neutrino energy loss rate and partial
/// derivatives (`plas_neu` in `mod_neu.f90`, in $\mathrm{erg\,cm^{-3}\,s^{-1}}$
/// before dividing by $\rho$).
///
/// # Arguments
///
/// * `input` - Precomputed thermodynamic and composition invariants
///   ([`NeuInputs`]).
///
/// # Returns
///
/// Volumetric rate [`NeuRate`] ($Q_{\mathrm{plas}}$ and partial derivatives
/// with respect to $T, \rho, \bar{A}, \bar{Z}$).
#[must_use]
pub fn plas_neu(input: &NeuInputs) -> NeuRate {
    // Itoh et al. (1996) equation 4.6: dimensionless plasmon frequency gamma^2
    let a1 = 1.019e-6 * input.rm;
    let a2 = pow(a1, TWO_THIRDS);
    let a3 = TWO_THIRDS * a2 / a1;
    let b1 = (1.0 + a2).sqrt();
    let b2 = 1.0 / b1;
    let c00 = 1.0 / (input.temp * input.temp * b1);

    let gl2 = 1.1095e11 * input.rm * c00;
    let gl2dt = -2.0 * gl2 * input.tempi;
    let d_rm = input.rm * c00 * b2 * 0.5 * b2 * a3 * 1.019e-6;
    let gl2dd = 1.1095e11 * (input.rmdd * c00 - d_rm * input.rmdd);
    let gl2da = 1.1095e11 * (input.rmda * c00 - d_rm * input.rmda);
    let gl2dz = 1.1095e11 * (input.rmdz * c00 - d_rm * input.rmdz);

    let gl = gl2.sqrt();
    let gl12 = gl.sqrt();
    let gl32 = gl * gl12;
    let gl72 = gl2 * gl32;
    let gl6 = gl2 * gl2 * gl2;
    let gum = 1.0 / gl2;

    // Equation 4.7: transverse plasmon contribution f_T
    let ft = 2.4 + 0.6 * gl12 + 0.51 * gl + 1.25 * gl32;
    let dft_dgl2 =
        (0.25 * 0.6 * gl12 + 0.5 * 0.51 * gl + 0.75 * 1.25 * gl32) * gum;
    let ftdt = dft_dgl2 * gl2dt;
    let ftdd = dft_dgl2 * gl2dd;
    let ftda = dft_dgl2 * gl2da;
    let ftdz = dft_dgl2 * gl2dz;

    // Equation 4.8: longitudinal plasmon contribution f_L
    let fl_num = 8.6 * gl2 + 1.35 * gl72;
    let dfl_num = 8.6 + 1.75 * 1.35 * gl72 * gum;
    let fl_den = 225.0 - 17.0 * gl + gl2;
    let dfl_den = -0.5 * 17.0 * gl * gum + 1.0;
    let inv_fl_den = 1.0 / fl_den;
    let fl = fl_num * inv_fl_den;
    let dfl_dgl2 = (dfl_num - fl * dfl_den) * inv_fl_den;
    let fldt = dfl_dgl2 * gl2dt;
    let fldd = dfl_dgl2 * gl2dd;
    let flda = dfl_dgl2 * gl2da;
    let fldz = dfl_dgl2 * gl2dz;

    // Equations 4.9–4.11: 2D correction factor f_xy(x, y)
    let cc = log10(2.0 * input.rm);
    let xlnt = input.logtemp;
    let xnum = ONE_SIXTH * (17.5 + cc - 3.0 * xlnt);
    let xnumdt = -ILN10 * 0.5 * input.tempi;
    let d_cc = ILN10 * ONE_SIXTH * input.rmi;
    let xnumdd = d_cc * input.rmdd;
    let xnumda = d_cc * input.rmda;
    let xnumdz = d_cc * input.rmdz;

    let xden = ONE_SIXTH * (-24.5 + cc + 3.0 * xlnt);
    let xdendt = ILN10 * 0.5 * input.tempi;
    let xdendd = xnumdd;
    let xdenda = xnumda;
    let xdendz = xnumdz;

    let (fxy, fxydt, fxydd, fxyda, fxydz) = if xnum.abs() > 0.7 || xden < 0.0 {
        (1.0, 0.0, 0.0, 0.0, 0.0)
    } else {
        let p_a1 = 0.39 - 1.25 * xnum - 0.35 * sin(4.5 * xnum);
        let p_a2 = -1.25 - 4.5 * 0.35 * cos(4.5 * xnum);
        let arg = 4.5 * xnum + 0.9;
        let p_b1 = 0.3 * exp(-pow(arg, 2.0));
        let p_b2 = -p_b1 * 2.0 * arg * 4.5;
        let c_val = (xden - 1.6 + 1.25 * xnum).min(0.0);
        let (dumdt, dumdd, dumda, dumdz) = if c_val == 0.0 {
            (0.0, 0.0, 0.0, 0.0)
        } else {
            (
                xdendt + 1.25 * xnumdt,
                xdendd + 1.25 * xnumdd,
                xdenda + 1.25 * xnumda,
                xdendz + 1.25 * xnumdz,
            )
        };
        let d_val = 0.57 - 0.25 * xnum;
        let a3_val = c_val / d_val;
        let g_exp = exp(-a3_val * a3_val);
        let f1 = -g_exp * 2.0 * a3_val / d_val;
        let c01 = f1 * (dumdt + a3_val * 0.25 * xnumdt);
        let c02 = f1 * (dumdd + a3_val * 0.25 * xnumdd);
        let c03 = f1 * (dumda + a3_val * 0.25 * xnumda);
        let c04 = f1 * (dumdz + a3_val * 0.25 * xnumdz);
        let diff_ab = p_a1 - p_b1;
        let ddiff_ab = p_a2 - p_b2;
        (
            1.05 + diff_ab * g_exp,
            ddiff_ab * xnumdt * g_exp + diff_ab * c01,
            ddiff_ab * xnumdd * g_exp + diff_ab * c02,
            ddiff_ab * xnumda * g_exp + diff_ab * c03,
            ddiff_ab * xnumdz * g_exp + diff_ab * c04,
        )
    };

    // Equations 4.1 and 4.5: combine (ft + fl) * fxy * exp(-gl) * gl^6 * pref
    let sum_ftl = ft + fl;
    let mut splas = sum_ftl * fxy;
    let mut splasdt = (ftdt + fldt) * fxy + sum_ftl * fxydt;
    let mut splasdd = (ftdd + fldd) * fxy + sum_ftl * fxydd;
    let mut splasda = (ftda + flda) * fxy + sum_ftl * fxyda;
    let mut splasdz = (ftdz + fldz) * fxy + sum_ftl * fxydz;

    let exp_mgl = exp(-gl);
    let dexp_dgl2 = -0.5 * exp_mgl * gl * gum;
    let s1 = splas;
    splas = exp_mgl * s1;
    splasdt = exp_mgl * splasdt + dexp_dgl2 * gl2dt * s1;
    splasdd = exp_mgl * splasdd + dexp_dgl2 * gl2dd * s1;
    splasda = exp_mgl * splasda + dexp_dgl2 * gl2da * s1;
    splasdz = exp_mgl * splasdz + dexp_dgl2 * gl2dz * s1;

    let dgl6_dgl2 = 3.0 * gl6 * gum;
    let s2 = splas;
    splas = gl6 * s2;
    splasdt = gl6 * splasdt + dgl6_dgl2 * gl2dt * s2;
    splasdd = gl6 * splasdd + dgl6_dgl2 * gl2dd * s2;
    splasda = gl6 * splasda + dgl6_dgl2 * gl2da * s2;
    splasdz = gl6 * splasdz + dgl6_dgl2 * gl2dz * s2;

    let pref = TFAC0 * 3.0e21 * input.xl9;
    let dpref_dt = TFAC0 * 3.0e21 * 9.0 * input.xl8 * input.xldt;
    let s3 = splas;
    NeuRate {
        val: pref * s3,
        d_t: pref * splasdt + dpref_dt * s3,
        d_rho: pref * splasdd,
        d_abar: pref * splasda,
        d_zbar: pref * splasdz,
    }
}
