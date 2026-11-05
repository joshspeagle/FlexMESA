// SPDX-License-Identifier: LGPL-3.0-or-later
//! Photoneutrino (`phot_neu`) energy loss rates and analytical derivatives.
//!
//! Ports `phot_neu` (Itoh et al. 1996, ApJS, 102, 411, eqs. 3.2–3.8 and Table
//! 2) from `neu/private/mod_neu.f90`.
//!
//! # Fourier Expansion Across Temperature Regimes
//!
//! Evaluates the 5-harmonic Fourier series for $a_0(\tau), a_1(\tau),
//! a_2(\tau)$ in each of three temperature intervals:
//! - $10^7 \le T < 10^8\,\mathrm{K}$ ($\tau = \log_{10} T - 7$, $c = 0.5654 +
//!   \tau$)
//! - $10^8 \le T < 10^9\,\mathrm{K}$ ($\tau = \log_{10} T - 8$, $c = 1.5654$)
//! - $T \ge 10^9\,\mathrm{K}$ ($\tau = \log_{10} T - 9$, $c = 1.5654$)
//!
//! Uses Chebyshev multiple-angle recurrences for $\sin(k \omega \tau)$ and
//! $\cos(k \omega \tau)$ ($k = 1, \dots, 5$, $\omega = 5\pi/3$).

use crate::inputs::{TFAC3, TFAC4};
use crate::{NeuInputs, NeuRate};
use flexmesa_const::{ILN10, LN10, PI};
use flexmesa_math::{cos, exp, pow, sin};

/// Angular frequency $\omega_1 = 5\pi / 3$ (`fac1` in `mod_neu.f90`).
const FAC1: f64 = 5.0 * PI / 3.0;

/// Angular frequency $\omega_2 = 10\pi$ (`fac2` in `mod_neu.f90`).
const FAC2: f64 = 10.0 * PI;

/// Table 2 Fourier coefficients `(c_j[0..7], d_j[0..5])` for $j \in \{0, 1,
/// 2\}$.
///
/// Each temperature decade interval ($10^7\text{--}10^8\,\mathrm{K}$,
/// $10^8\text{--}10^9\,\mathrm{K}$, and $\ge 10^9\,\mathrm{K}$) stores 7 cosine
/// coefficients `c_j` and 5 sine coefficients `d_j` for the polynomial
/// numerator weights $a_0(\tau), a_1(\tau), a_2(\tau)$ in Itoh et al. (1996,
/// Table 2, eq. 3.6):
/// $$a_j(\tau) = \tfrac{1}{2} c_{j,0} + \sum_{k=1}^5 \left[c_{j,k}
///   \cos(k \omega_1 \tau) + d_{j,k} \sin(k \omega_1 \tau)\right] +
///   \tfrac{1}{2} c_{j,6} \cos(\omega_2 \tau)$$
struct PhotBandCoeffs {
    /// Cosine harmonic coefficients $c_{0,0 \dots 6}$ for $a_0(\tau)$.
    c0: [f64; 7],
    /// Sine harmonic coefficients $d_{0,1 \dots 5}$ for $a_0(\tau)$.
    d0: [f64; 5],
    /// Cosine harmonic coefficients $c_{1,0 \dots 6}$ for $a_1(\tau)$.
    c1: [f64; 7],
    /// Sine harmonic coefficients $d_{1,1 \dots 5}$ for $a_1(\tau)$.
    d1: [f64; 5],
    /// Cosine harmonic coefficients $c_{2,0 \dots 6}$ for $a_2(\tau)$.
    c2: [f64; 7],
    /// Sine harmonic coefficients $d_{2,1 \dots 5}$ for $a_2(\tau)$.
    d2: [f64; 5],
}

/// Fourier coefficients for $10^7 \le T < 10^8\,\mathrm{K}$ ($\tau = \log_{10}
/// T - 7$).
const BAND_7_8: PhotBandCoeffs = PhotBandCoeffs {
    c0: [1.008e11, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    d0: [0.0; 5],
    c1: [
        8.156e10, 9.728e8, -3.806e9, -4.384e9, -5.774e9, -5.249e9, -5.153e9,
    ],
    d1: [-1.879e10, -9.667e9, -5.602e9, -3.370e9, -1.825e9],
    c2: [
        1.067e11, -9.782e9, -7.193e9, -6.936e9, -6.893e9, -7.041e9, -7.193e9,
    ],
    d2: [-2.919e10, -1.185e10, -7.270e9, -4.222e9, -1.560e9],
};

const BAND_8_9: PhotBandCoeffs = PhotBandCoeffs {
    c0: [
        9.889e10, -4.524e8, -6.088e6, 4.269e7, 5.172e7, 4.910e7, 4.388e7,
    ],
    d0: [-1.135e8, 1.256e8, 5.149e7, 3.436e7, 1.005e7],
    c1: [
        1.813e11, -7.556e9, -3.304e9, -1.031e9, -1.764e9, -1.851e9, -1.928e9,
    ],
    d1: [1.652e9, -3.119e9, -1.839e9, -1.458e9, -8.956e8],
    c2: [
        9.750e10, 3.484e10, 5.199e9, -1.695e9, -2.865e9, -3.395e9, -3.418e9,
    ],
    d2: [-1.548e10, -9.338e9, -5.899e9, -3.035e9, -1.598e9],
};

const BAND_9_UP: PhotBandCoeffs = PhotBandCoeffs {
    c0: [
        9.581e10, 4.107e8, 2.305e8, 2.236e8, 1.580e8, 2.165e8, 1.721e8,
    ],
    d0: [4.724e8, 2.976e8, 2.242e8, 7.937e7, 4.859e7],
    c1: [
        1.459e12, 1.314e11, -1.169e11, -1.765e11, -1.867e11, -1.983e11,
        -1.896e11,
    ],
    d1: [-7.094e11, -3.697e11, -2.189e11, -1.273e11, -5.705e10],
    c2: [
        2.424e11, -3.669e9, -8.691e9, -7.967e9, -7.932e9, -7.987e9, -8.333e9,
    ],
    d2: [-2.254e10, -1.551e10, -7.793e9, -4.489e9, -2.185e9],
};

/// Evaluates one Fourier series $(a_j, da_j/dT)$ from `c[0..7]` and `d[0..5]`.
fn eval_fourier_row(
    c: &[f64; 7],
    d: &[f64; 5],
    cos_k: &[f64; 5],
    sin_k: &[f64; 5],
    last: f64,
    xast: f64,
    taudt: f64,
) -> (f64, f64) {
    let a = 0.5 * c[0]
        + c[1] * cos_k[0]
        + d[0] * sin_k[0]
        + c[2] * cos_k[1]
        + d[1] * sin_k[1]
        + c[3] * cos_k[2]
        + d[2] * sin_k[2]
        + c[4] * cos_k[3]
        + d[3] * sin_k[3]
        + c[5] * cos_k[4]
        + d[4] * sin_k[4]
        + 0.5 * c[6] * last;
    let da_dt = taudt
        * FAC1
        * (-c[1] * sin_k[0] + d[0] * cos_k[0] - c[2] * sin_k[1] * 2.0
            + d[1] * cos_k[1] * 2.0
            - c[3] * sin_k[2] * 3.0
            + d[2] * cos_k[2] * 3.0
            - c[4] * sin_k[3] * 4.0
            + d[3] * cos_k[3] * 4.0
            - c[5] * sin_k[4] * 5.0
            + d[4] * cos_k[4] * 5.0)
        - 0.5 * c[6] * xast * FAC2 * taudt;
    (a, da_dt)
}

/// Evaluates volumetric photoneutrino energy loss rate and partial derivatives
/// (`phot_neu` in `mod_neu.f90`, in $\mathrm{erg\,cm^{-3}\,s^{-1}}$ before
/// dividing by $\rho$).
///
/// Combines the temperature-dependent Fourier expansion $f_{\mathrm{phot}}$
/// (eq. 3.4) with the relativistic degeneracy screening factor
/// $q_{\mathrm{phot}}$ (eq. 3.3):
/// $$Q_{\mathrm{phot}} = \tfrac{1}{2}\left[(C_V^2 + C_A^2) + n_\nu (C_V'^2 +
///   C_A'^2)\right]\left[1 - \frac{(C_V^2 - C_A^2) + n_\nu (C_V'^2 -
///   C_A'^2)}{(C_V^2 + C_A^2) + n_\nu (C_V'^2 + C_A'^2)}\,q_{\mathrm{phot}}
///   \right] (\rho Y_e)\,\lambda^5 f_{\mathrm{phot}}(\lambda, \xi)$$
///
/// # Arguments
///
/// * `input` - Precomputed thermodynamic and composition invariants
///   ([`NeuInputs`]).
///
/// # Returns
///
/// Volumetric rate [`NeuRate`] ($Q_{\mathrm{phot}}$ and partial derivatives
/// with respect to $T, \rho, \bar{A}, \bar{Z}$), or [`NeuRate::ZERO`] when
/// $T \le 10^7\,\mathrm{K}$.
#[must_use]
pub fn phot_neu(input: &NeuInputs) -> NeuRate {
    if input.temp <= 1.0e7 {
        return NeuRate::ZERO;
    }
    let (tau, cc, dccdt, band) = if input.temp < 1.0e8 {
        let t = input.logtemp - 7.0;
        (t, 0.5654 + t, 1.0 / (LN10 * input.temp), &BAND_7_8)
    } else if input.temp < 1.0e9 {
        (input.logtemp - 8.0, 1.5654, 0.0, &BAND_8_9)
    } else {
        (input.logtemp - 9.0, 1.5654, 0.0, &BAND_9_UP)
    };

    let taudt = ILN10 * input.tempi;
    let cos1 = cos(FAC1 * tau);
    let sin1 = sin(FAC1 * tau);
    let sin2 = 2.0 * sin1 * cos1;
    let cos2 = 2.0 * cos1 * cos1 - 1.0;
    let sin3 = sin1 * (3.0 - 4.0 * sin1 * sin1);
    let cos3 = cos1 * (4.0 * cos1 * cos1 - 3.0);
    let sin4 = 2.0 * sin2 * cos2;
    let cos4 = 2.0 * cos2 * cos2 - 1.0;
    let sin5 = sin1 * (5.0 - sin1 * sin1 * (20.0 - 16.0 * sin1 * sin1));
    let cos5 = cos1 * (cos1 * cos1 * (16.0 * cos1 * cos1 - 20.0) + 5.0);
    let last = cos(FAC2 * tau);
    let xast = sin(FAC2 * tau);

    let cos_k = [cos1, cos2, cos3, cos4, cos5];
    let sin_k = [sin1, sin2, sin3, sin4, sin5];
    let (a0, f0) =
        eval_fourier_row(&band.c0, &band.d0, &cos_k, &sin_k, last, xast, taudt);
    let (a1, f1) =
        eval_fourier_row(&band.c1, &band.d1, &cos_k, &sin_k, last, xast, taudt);
    let (a2, f2) =
        eval_fourier_row(&band.c2, &band.d2, &cos_k, &sin_k, last, xast, taudt);

    // Itoh et al. (1996) equation 3.4: f_phot
    let dum = a0 + a1 * input.zeta + a2 * input.zeta2;
    let dumdt = f0
        + f1 * input.zeta
        + a1 * input.zetadt
        + f2 * input.zeta2
        + 2.0 * a2 * input.zeta * input.zetadt;
    let dumdd = a1 * input.zetadd + 2.0 * a2 * input.zeta * input.zetadd;
    let dumda = a1 * input.zetada + 2.0 * a2 * input.zeta * input.zetada;
    let dumdz = a1 * input.zetadz + 2.0 * a2 * input.zeta * input.zetadz;

    let z = exp(-cc * input.zeta);
    let xnum = dum * z;
    let xnumdt = dumdt * z - dum * z * (dccdt * input.zeta + input.zetadt * cc);
    let xnumdd = dumdd * z - dum * z * cc * input.zetadd;
    let xnumda = dumda * z - dum * z * cc * input.zetada;
    let xnumdz = dumdz * z - dum * z * cc * input.zetadz;

    let xden = input.zeta3
        + 6.290e-3 * input.xlm1
        + 7.483e-3 * input.xlm2
        + 3.061e-4 * input.xlm3;
    let dz3 = 3.0 * input.zeta2;
    let xdendt = dz3 * input.zetadt
        - input.xldt
            * (6.290e-3 * input.xlm2
                + 2.0 * 7.483e-3 * input.xlm3
                + 3.0 * 3.061e-4 * input.xlm4);
    let xdendd = dz3 * input.zetadd;
    let xdenda = dz3 * input.zetada;
    let xdendz = dz3 * input.zetadz;

    let inv_xden = 1.0 / xden;
    let fphot = xnum * inv_xden;
    let fphotdt = (xnumdt - fphot * xdendt) * inv_xden;
    let fphotdd = (xnumdd - fphot * xdendd) * inv_xden;
    let fphotda = (xnumda - fphot * xdenda) * inv_xden;
    let fphotdz = (xnumdz - fphot * xdendz) * inv_xden;

    // Itoh et al. (1996) equation 3.3: q_phot
    let q_a0 = 1.0 + 2.045 * input.xl;
    let q_xnum = 0.666 * pow(q_a0, -2.066);
    let q_xnumdt = -2.066 * q_xnum / q_a0 * 2.045 * input.xldt;

    let q_dum = 1.875e8 * input.xl
        + 1.653e8 * input.xl2
        + 8.499e8 * input.xl3
        - 1.604e8 * input.xl4;
    let q_dumdt = input.xldt
        * (1.875e8 + 2.0 * 1.653e8 * input.xl + 3.0 * 8.499e8 * input.xl2
            - 4.0 * 1.604e8 * input.xl3);

    let inv_qdum = 1.0 / q_dum;
    let q_xden = 1.0 + input.rm * inv_qdum;
    let q_xdendt = -input.rm * inv_qdum * inv_qdum * q_dumdt;
    let q_xdendd = input.rmdd * inv_qdum;
    let q_xdenda = input.rmda * inv_qdum;
    let q_xdendz = input.rmdz * inv_qdum;

    let inv_qxden = 1.0 / q_xden;
    let qphot = q_xnum * inv_qxden;
    let qphotdt = (q_xnumdt - qphot * q_xdendt) * inv_qxden;
    let neg_qp = -qphot * inv_qxden;
    let qphotdd = neg_qp * q_xdendd;
    let qphotda = neg_qp * q_xdenda;
    let qphotdz = neg_qp * q_xdendz;

    // Itoh et al. (1996) equation 3.2: combine xl^5, rm, fphot, and qphot
    let s0 = input.xl5 * fphot;
    let s0dt = 5.0 * input.xl4 * input.xldt * fphot + input.xl5 * fphotdt;
    let s0dd = input.xl5 * fphotdd;
    let s0da = input.xl5 * fphotda;
    let s0dz = input.xl5 * fphotdz;

    let s1 = input.rm * s0;
    let s1dt = input.rm * s0dt;
    let s1dd = input.rm * s0dd + input.rmdd * s0;
    let s1da = input.rm * s0da + input.rmda * s0;
    let s1dz = input.rm * s0dz + input.rmdz * s0;

    let w1 = TFAC4 * (1.0 - TFAC3 * qphot);
    let w2 = -TFAC4 * TFAC3;
    let sphot = w1 * s1;
    if sphot <= 0.0 {
        return NeuRate::ZERO;
    }
    NeuRate {
        val: sphot,
        d_t: w1 * s1dt + w2 * qphotdt * s1,
        d_rho: w1 * s1dd + w2 * qphotdd * s1,
        d_abar: w1 * s1da + w2 * qphotda * s1,
        d_zbar: w1 * s1dz + w2 * qphotdz * s1,
    }
}
