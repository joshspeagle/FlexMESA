// SPDX-License-Identifier: LGPL-3.0-or-later
//! Analytical NACRE (Angulo et al. 1999) and Timmes (`fxt`) nuclear reaction
//! rate evaluators ported from `rates/private/ratelib.f90`.

use flexmesa_const::{LN2, ONE_THIRD};

use crate::tfactors::TFactors;
use crate::types::LOW_T9_PP_CUTOFF;

const LOW_T9_CUTOFF: f64 = 1.0e-3;
const FIVE_SIXTHS: f64 = 5.0 / 6.0;

#[inline]
#[allow(clippy::too_many_arguments)]
fn rnacre(
    tf: &TFactors,
    a0: f64,
    a1: f64,
    a2: f64,
    b0: f64,
    b1: f64,
    b2: f64,
    b3: f64,
    b4: f64,
    c0: f64,
    c1: f64,
    d0: f64,
    d1: f64,
    e0: f64,
    e1: f64,
    e2: f64,
) -> f64 {
    let aa = a0 * tf.t9i23 * (-a1 * tf.t9i13 - tf.t92 * (a2 * a2)).exp();
    let bb = (1.0
        + b0 * tf.t9
        + b1 * tf.t92
        + b2 * tf.t93
        + b3 * tf.t94
        + b4 * tf.t95)
        .max(0.0);
    let cc = c0 * tf.t9i32 * (-c1 * tf.t9i).exp();
    let dd = d0 * tf.t9i32 * (-d1 * tf.t9i).exp();
    let ee = e0 * tf.t9.powf(e1) * (-e2 * tf.t9i).exp();
    aa * bb + cc + dd + ee
}

#[inline]
fn rnacre_rev(tf: &TFactors, a0: f64, a1: f64) -> f64 {
    a0 * tf.t932 * (-a1 * tf.t9i).exp()
}

/// Evaluates an analytical `ratelib.f90` subroutine (`op_code` in `3..=28`)
/// returning `(forward_rate, reverse_rate)`.
#[must_use]
pub fn eval_analytic_rate(op_code: u8, tf: &TFactors) -> (f64, f64) {
    match op_code {
        3 => rate_b8ep(tf),
        4 => rate_be7em_fxt(tf),
        5 => rate_be7pg_nacre(tf),
        6 => rate_c12ag_nacre(tf),
        7 => rate_c12c12_fxt_multi(tf),
        8 => rate_c12o16_fxt(tf),
        9 => rate_c12pg_nacre(tf),
        10 => rate_c13pg_nacre(tf),
        11 => rate_dpg_fxt(tf),
        12 => rate_dpg_nacre(tf),
        13 => rate_f19pa_nacre(tf),
        14 => rate_he3he3_nacre(tf),
        15 => rate_he3he4_nacre(tf),
        16 => rate_he3ng_fxt(tf),
        17 => rate_hep_fxt(tf),
        18 => rate_li7pa_nacre(tf),
        19 => rate_n13pg_nacre(tf),
        20 => rate_n14pg_nacre(tf),
        21 => rate_ne20pg_nacre(tf),
        22 => rate_o16ag_nacre(tf),
        23 => rate_o16o16g_fxt(tf),
        24 => rate_o16pg_nacre(tf),
        25 => rate_o18pa_nacre(tf),
        26 => rate_pep_fxt(tf),
        27 => rate_png_fxt(tf),
        28 => rate_pp_nacre(tf),
        _ => (0.0, 0.0),
    }
}

#[inline]
fn rate_b8ep(_tf: &TFactors) -> (f64, f64) {
    (LN2 / 0.77, 0.0)
}

#[inline]
fn rate_be7em_fxt(tf: &TFactors) -> (f64, f64) {
    if tf.t9 < LOW_T9_PP_CUTOFF {
        return (0.0, 0.0);
    }
    let term = if (1.0e-3..=3.0).contains(&tf.t9) {
        let aa = 0.0027 * tf.t9i * (2.515e-3 * tf.t9i).exp();
        let bb = 1.0 - 0.537 * tf.t913 + 3.86 * tf.t923 + aa;
        1.34e-10 * tf.t9i12 * bb
    } else {
        0.0
    };
    (term, 0.0)
}

#[inline]
fn rate_be7pg_nacre(tf: &TFactors) -> (f64, f64) {
    if tf.t9 < LOW_T9_PP_CUTOFF {
        return (0.0, 0.0);
    }
    let term = rnacre(
        tf, 2.61e5, 10.264, 0.0, -5.11e-2, 4.68e-2, -6.60e-3, 3.12e-4, 0.0,
        2.05e3, 7.345, 0.0, 0.0, 0.0, 0.0, 0.0,
    );
    let rev = rnacre_rev(tf, 1.306e10, 1.594);
    (term, rev * term)
}

#[inline]
fn rate_c12ag_nacre(tf: &TFactors) -> (f64, f64) {
    if tf.t9 < LOW_T9_CUTOFF {
        return (0.0, 0.0);
    }
    let aa1 =
        6.66e7 * tf.t9i2 * (-32.123 * tf.t9i13 - tf.t92 / (4.6 * 4.6)).exp();
    let bb1 = (1.0 + 2.54 * tf.t9 + 1.04 * tf.t92 - 0.226 * tf.t93).max(0.0);
    let cc1 = 1.39e3 * tf.t9i32 * (-28.930 * tf.t9i).exp();
    let term_e1 = aa1 * bb1 + cc1;

    let aa2 =
        6.56e7 * tf.t9i2 * (-32.123 * tf.t9i13 - tf.t92 / (1.3 * 1.3)).exp();
    let bb2 = 1.0 + 9.23 * tf.t9 - 13.7 * tf.t92 + 7.4 * tf.t93;
    let term_e2 = aa2 * bb2;

    let term_res = 19.2 * tf.t92 * (-26.9 * tf.t9i).exp();
    let term = term_e1 + term_e2 + term_res;
    let rev = 5.132e10 * tf.t932 * (-83.109 * tf.t9i).exp();
    (term, rev * term)
}

#[inline]
fn rate_c12c12_fxt_multi(tf: &TFactors) -> (f64, f64) {
    if tf.t9 < LOW_T9_CUTOFF {
        return (0.0, 0.0);
    }
    let t9a = tf.t9 / (1.0 + 0.0396 * tf.t9);
    let t9a13 = t9a.powf(ONE_THIRD);
    let t9a56 = t9a.powf(FIVE_SIXTHS);
    let aa =
        4.27e26 * t9a56 * tf.t9i32 * (-84.165 / t9a13 - 2.12e-3 * tf.t93).exp();
    let b24n = if tf.t9 >= 1.5 {
        let bb = 0.055 * (0.976 - 0.789 * tf.t9).exp();
        0.055 - bb
    } else {
        let bb = 1.0 + 0.0789 * tf.t9 + 7.74 * tf.t92;
        let cc = 0.766 * tf.t9i3;
        0.859 * (-bb * cc).exp()
    };
    let (b24p, b24a) = if tf.t9 > 3.0 {
        let bp = ONE_THIRD * (1.0 - b24n);
        (bp, 2.0 * bp)
    } else {
        let bp = 0.5 * (1.0 - b24n);
        (bp, bp)
    };
    let fr1 = aa * b24n;
    let rr1 = if tf.t9 > 0.1 {
        3.93 * (30.16100515 * tf.t9i).exp() * fr1
    } else {
        0.0
    };
    let fr2 = aa * b24p;
    let rr2 = if tf.t9 > 0.1 {
        3.93 * (-25.98325915 * tf.t9i).exp() * fr2
    } else {
        0.0
    };
    let fr3 = aa * b24a;
    let rr3 = if tf.t9 > 0.1 {
        2.42 * (-53.576110995 * tf.t9i).exp() * fr3
    } else {
        0.0
    };
    (fr3 + fr1 + fr2, rr3 + rr1 + rr2)
}

#[inline]
fn rate_c12o16_fxt(tf: &TFactors) -> (f64, f64) {
    if tf.t9 < LOW_T9_CUTOFF || tf.t9 < 0.5 {
        return (0.0, 0.0);
    }
    let t9a = tf.t9 / (1.0 + 0.055 * tf.t9);
    let t9a13 = t9a.powf(ONE_THIRD);
    let t9a23 = t9a13 * t9a13;
    let t9a56 = t9a.powf(FIVE_SIXTHS);
    let aa = (-0.18 * t9a * t9a).exp();
    let bb = 1.06e-3 * (2.562 * t9a23).exp();
    let cc = aa + bb;
    let term = 1.72e31 * t9a56 * tf.t9i32 * (-106.594 / t9a13).exp() / cc;
    (term, 0.0)
}

#[inline]
fn rate_c12pg_nacre(tf: &TFactors) -> (f64, f64) {
    if tf.t9 < LOW_T9_CUTOFF {
        return (0.0, 0.0);
    }
    let term = rnacre(
        tf,
        2.00e7,
        13.692,
        1.0 / 0.46,
        9.89,
        -59.8,
        266.0,
        0.0,
        0.0,
        1.00e5,
        4.913,
        4.24e5,
        21.62,
        0.0,
        0.0,
        0.0,
    );
    let rev = rnacre_rev(tf, 8.847e9, 22.553);
    (term, rev * term)
}

#[inline]
fn rate_c13pg_nacre(tf: &TFactors) -> (f64, f64) {
    if tf.t9 < LOW_T9_CUTOFF {
        return (0.0, 0.0);
    }
    let gs = rnacre(
        tf, 9.57e7, 13.72, 1.0, 3.56, 0.0, 0.0, 0.0, 0.0, 1.50e6, 5.930, 0.0,
        0.0, 6.83e5, -0.864, 12.057,
    );
    let bb = (2.070 * (-37.938 * tf.t9i).exp()).min(1.0);
    let term = gs * (1.0 - bb);
    let rev = rnacre_rev(tf, 1.190e10, 87.619);
    (term, rev * term)
}

#[inline]
fn rate_dpg_fxt(tf: &TFactors) -> (f64, f64) {
    if tf.t9 < LOW_T9_PP_CUTOFF {
        return (0.0, 0.0);
    }
    let aa = 2.24e3 * tf.t9i23 * (-3.720 * tf.t9i13).exp();
    let bb = 1.0 + 0.112 * tf.t913 + 3.38 * tf.t923 + 2.65 * tf.t9;
    let term = aa * bb;
    let rev = 1.63e10 * tf.t932 * (-63.750 * tf.t9i).exp();
    (term, rev * term)
}

#[inline]
fn rate_dpg_nacre(tf: &TFactors) -> (f64, f64) {
    if tf.t9 < LOW_T9_PP_CUTOFF {
        return (0.0, 0.0);
    }
    let term = if tf.t9 <= 0.11 {
        rnacre(
            tf, 1.81e3, 3.721, 0.0, 14.3, -90.5, 395.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        )
    } else {
        rnacre(
            tf, 2.58e3, 3.721, 0.0, 3.96, 0.116, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        )
    };
    let rev = rnacre_rev(tf, 1.63e10, 63.749);
    (term, rev * term)
}

#[inline]
fn rate_f19pa_nacre(tf: &TFactors) -> (f64, f64) {
    if tf.t9 < LOW_T9_CUTOFF {
        return (0.0, 0.0);
    }
    let mut gs = rnacre(
        tf,
        2.62e11,
        18.116,
        1.0 / 0.185,
        6.26e-2,
        0.285,
        4.94e-3,
        11.5,
        7.40e4,
        3.80e6,
        3.752,
        0.0,
        0.0,
        3.27e7,
        -0.193,
        6.587,
    );
    let dd = 7.30e8 * tf.t9.powf(-0.201) * (-16.249 * tf.t9i).exp();
    gs += dd;
    let bb = 0.755 * (-1.755 * tf.t9i - 0.174 * tf.t9).exp();
    let term = gs * (1.0 + bb);
    let rev = rnacre_rev(tf, 6.538e-1, 94.154);
    (term, rev * term)
}

#[inline]
fn rate_he3he3_nacre(tf: &TFactors) -> (f64, f64) {
    if tf.t9 < LOW_T9_PP_CUTOFF {
        return (0.0, 0.0);
    }
    let term = rnacre(
        tf, 5.59e10, 12.277, 0.0, -0.135, 2.54e-2, -1.29e-3, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    );
    let rev = rnacre_rev(tf, 3.392e-10, 149.23);
    (term, rev * term)
}

#[inline]
fn rate_he3he4_nacre(tf: &TFactors) -> (f64, f64) {
    if tf.t9 < LOW_T9_PP_CUTOFF {
        return (0.0, 0.0);
    }
    let term = rnacre(
        tf, 5.46e6, 12.827, 0.0, -0.307, 8.81e-2, -1.06e-2, 4.46e-4, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    );
    let rev = rnacre_rev(tf, 1.113e10, 18.412);
    (term, rev * term)
}

#[inline]
fn rate_he3ng_fxt(tf: &TFactors) -> (f64, f64) {
    if tf.t9 < LOW_T9_CUTOFF {
        return (0.0, 0.0);
    }
    let term = 6.62 * (1.0 + 905.0 * tf.t9);
    let rev = 2.61e10 * tf.t932 * (-238.81 * tf.t9i).exp();
    (term, rev * term)
}

#[inline]
fn rate_hep_fxt(tf: &TFactors) -> (f64, f64) {
    if tf.t9 < LOW_T9_PP_CUTOFF {
        return (0.0, 0.0);
    }
    let term = if tf.t9 <= 3.0 {
        8.78e-13 * tf.t9i23 * (-6.141 * tf.t9i13).exp()
    } else {
        5.9733434e-15
    };
    (term, 0.0)
}

#[inline]
fn rate_li7pa_nacre(tf: &TFactors) -> (f64, f64) {
    if tf.t9 < LOW_T9_PP_CUTOFF {
        return (0.0, 0.0);
    }
    let term = rnacre(
        tf,
        7.20e8,
        8.473,
        1.0 / 6.5,
        1.05,
        -0.653,
        0.185,
        -2.12e-2,
        9.30e-4,
        0.0,
        0.0,
        0.0,
        0.0,
        9.85e6,
        0.576,
        10.415,
    );
    let rev = rnacre_rev(tf, 4.676, 201.30);
    (term, rev * term)
}

#[inline]
fn rate_n13pg_nacre(tf: &TFactors) -> (f64, f64) {
    if tf.t9 < LOW_T9_CUTOFF {
        return (0.0, 0.0);
    }
    let term = rnacre(
        tf,
        4.02e7,
        15.205,
        1.0 / 0.54,
        3.81,
        18.6,
        32.3,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        3.25e5,
        -1.35,
        5.926,
    );
    let rev = rnacre_rev(tf, 3.571e10, 53.705);
    (term, rev * term)
}

#[inline]
fn rate_n14pg_nacre(tf: &TFactors) -> (f64, f64) {
    if tf.t9 < LOW_T9_CUTOFF {
        return (0.0, 0.0);
    }
    let term = rnacre(
        tf,
        4.83e7,
        15.231,
        1.0 / 0.8,
        -2.00,
        3.41,
        -2.43,
        0.0,
        0.0,
        2.36e3,
        3.010,
        0.0,
        0.0,
        6.72e3,
        0.380,
        9.530,
    );
    let rev = rnacre_rev(tf, 2.699e10, 84.677);
    (term, rev * term)
}

#[inline]
fn rate_ne20pg_nacre(tf: &TFactors) -> (f64, f64) {
    if tf.t9 < LOW_T9_CUTOFF {
        return (0.0, 0.0);
    }
    let gs = 2.35e7
        * tf.t9.powf(-1.84)
        * (-19.451 * tf.t9i13).exp()
        * (1.0 + 10.80 * tf.t9)
        + 18.0 * tf.t9i32 * (-4.247 * tf.t9i).exp()
        + 9.83 * tf.t9i32 * (-4.619 * tf.t9i).exp()
        + 6.76e4 * tf.t9.powf(-0.641) * (-11.922 * tf.t9i).exp();
    let bb = (7.929 * (-20.108 * tf.t9i - 0.327 * tf.t9).exp()).min(1.0);
    let term = gs * (1.0 - bb);
    let rev = 4.637e9 * tf.t932 * (-28.214 * tf.t9i).exp();
    (term, rev * term)
}

#[inline]
fn rate_o16ag_nacre(tf: &TFactors) -> (f64, f64) {
    if tf.t9 < LOW_T9_CUTOFF {
        return (0.0, 0.0);
    }
    let term = rnacre(
        tf,
        2.68e10,
        39.760,
        1.0 / 1.6,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        51.1,
        10.32,
        616.1,
        12.200,
        0.41,
        2.966,
        11.900,
    );
    let rev = rnacre_rev(tf, 5.653e10, 54.886);
    (term, rev * term)
}

#[inline]
fn rate_o16o16g_fxt(tf: &TFactors) -> (f64, f64) {
    if tf.t9 < LOW_T9_CUTOFF {
        return (0.0, 0.0);
    }
    let term = 0.10
        * 7.10e36
        * tf.t9i23
        * (-135.93 * tf.t9i13 - 0.629 * tf.t923 - 0.445 * tf.t943
            + 0.0103 * tf.t9 * tf.t9)
            .exp();
    (term, 0.0)
}

#[inline]
fn rate_o16pg_nacre(tf: &TFactors) -> (f64, f64) {
    if tf.t9 < LOW_T9_CUTOFF {
        return (0.0, 0.0);
    }
    let aa = 7.37e7 * tf.t9.powf(-0.82) * (-16.696 * tf.t9i13).exp();
    let bb = 1.0 + 202.0 * (-70.348 * tf.t9i - 0.161 * tf.t9).exp();
    let term = aa * bb;
    let rev = rnacre_rev(tf, 3.037e9, 6.966);
    (term, rev * term)
}

#[inline]
fn rate_o18pa_nacre(tf: &TFactors) -> (f64, f64) {
    if tf.t9 < LOW_T9_CUTOFF {
        return (0.0, 0.0);
    }
    let gs = rnacre(
        tf,
        5.58e11,
        16.732,
        1.0 / 0.51,
        3.2,
        21.8,
        0.0,
        0.0,
        0.0,
        9.91e-14,
        0.232,
        2.58e4,
        1.665,
        3.24e8,
        -0.378,
        6.395,
    );
    let bb = (1.968 * (-25.673 * tf.t9i - 0.083 * tf.t9).exp()).min(1.0);
    let term = gs * (1.0 - bb);
    let rev = rnacre_rev(tf, 1.660e-1, 46.192);
    (term, rev * term)
}

#[inline]
fn rate_pep_fxt(tf: &TFactors) -> (f64, f64) {
    if tf.t9 < LOW_T9_PP_CUTOFF {
        return (0.0, 0.0);
    }
    let term = if tf.t9 <= 3.0 {
        let aa = 1.36e-20 * tf.t9i76 * (-3.380 * tf.t9i13).exp();
        let bb = 1.0 - 0.729 * tf.t913 + 9.82 * tf.t923;
        aa * bb
    } else {
        7.3824387e-21
    };
    (term, 0.0)
}

#[inline]
fn rate_png_fxt(tf: &TFactors) -> (f64, f64) {
    if tf.t9 < LOW_T9_PP_CUTOFF {
        return (0.0, 0.0);
    }
    let aa = 1.0 - 0.8504 * tf.t912 + 0.4895 * tf.t9 - 0.09623 * tf.t932
        + 8.471e-3 * tf.t92
        - 2.80e-4 * tf.t952;
    let term = 4.742e4 * aa;
    let rev = 4.71e9 * tf.t932 * (-25.82 * tf.t9i).exp();
    (term, rev * term)
}

#[inline]
fn rate_pp_nacre(tf: &TFactors) -> (f64, f64) {
    if tf.t9 < LOW_T9_PP_CUTOFF {
        return (0.0, 0.0);
    }
    let term = rnacre(
        tf, 4.08e-15, 3.381, 0.0, 3.82, 1.51, 0.144, -1.14e-2, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
    );
    (term, 0.0)
}
