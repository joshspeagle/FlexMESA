//! Stage coefficient tables for L-stable Rosenbrock-Wanner ODE/DAE integrators.
//!
//! Ports the Butcher/Wanner stage tables (`ros2`, `rose2`, `ros3p`, `ros3pl`,
//! `rodas3`, `rodas4`, `rodasp`) from `num/private/mod_rosenbrock.f` into Safe
//! Rust in `f64` precision.
//!
//! # Mathematical Formulation
//!
//! For an autonomous or non-autonomous system $y'(t) = f(t, y)$, an $s$-stage
//! Rosenbrock-Wanner method with diagonal implicit parameter $\gamma > 0$
//! solves $s$ linear systems sharing a single $LU$ factorization of
//! $E = (\gamma h)^{-1} I - \partial f / \partial y$, and advances the state
//! $y_{n+1} = y_n + \sum_i m_i g_i$ and embedded error estimate
//! $e_{n+1} = \sum_i e_i g_i$.

/// Rosenbrock-Wanner solver variants matching `num_def.f90` and
/// `num/private/mod_rosenbrock.f`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RosenbrockVariant {
    /// 2-stage L-stable 2nd-order Rosenbrock (`ros2`).
    Ros2,
    /// 2-stage L-stable 2nd-order Rosenbrock variant (`rose2`).
    Rose2,
    /// 3-stage 3rd-order Rosenbrock (`ros3p`, Lang & Verwer 2001).
    Ros3p,
    /// 4-stage L-stable 3rd-order Rosenbrock (`ros3pl`).
    Ros3pl,
    /// 4-stage L-stable 3rd-order stiffly accurate Rosenbrock (`rodas3`).
    Rodas3,
    /// 6-stage L-stable 4th-order stiffly accurate Rosenbrock (`rodas4`).
    Rodas4,
    /// 6-stage L-stable 4th-order parabolic Rosenbrock (`rodasp`, Steinebach).
    Rodasp,
}

/// Stage coefficients for a Rosenbrock-Wanner method (`s` stages, $\gamma$).
pub(crate) struct RosCoeffs {
    pub(crate) s: usize,
    pub(crate) gamma: f64,
    pub(crate) order: f64,
    pub(crate) a: [[f64; 6]; 6],
    pub(crate) c: [[f64; 6]; 6],
    pub(crate) alpha: [f64; 6],
    pub(crate) gamma_t: [f64; 6],
    pub(crate) m: [f64; 6],
    pub(crate) e: [f64; 6],
}

/// Returns the stage coefficient table [`RosCoeffs`] for `variant`.
///
/// Maps each [`RosenbrockVariant`] onto its stage count `s`, diagonal parameter
/// $\gamma$, classical order $p$, and lower-triangular matrices $A$ and $C$.
pub(crate) fn coeffs_for(variant: RosenbrockVariant) -> RosCoeffs {
    let mut rc = RosCoeffs {
        s: 4,
        gamma: 0.5,
        order: 3.0,
        a: [[0.0; 6]; 6],
        c: [[0.0; 6]; 6],
        alpha: [0.0; 6],
        gamma_t: [0.0; 6],
        m: [0.0; 6],
        e: [0.0; 6],
    };
    match variant {
        RosenbrockVariant::Ros2 | RosenbrockVariant::Rose2 => {
            // 2-stage L-stable 2nd-order Rosenbrock with gamma = 1 - 1/sqrt(2)
            let g = 1.0 - 0.5 * 2.0_f64.sqrt();
            rc.s = 2;
            rc.gamma = g;
            rc.order = 2.0;
            rc.a[1][0] = 1.0 / g;
            rc.c[1][0] = -2.0 / g;
            rc.alpha = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
            rc.gamma_t = [g, -g, 0.0, 0.0, 0.0, 0.0];
            rc.m = [1.5 / g, 0.5 / g, 0.0, 0.0, 0.0, 0.0];
            rc.e = [0.5 / g, 0.5 / g, 0.0, 0.0, 0.0, 0.0];
        }
        RosenbrockVariant::Ros3p
        | RosenbrockVariant::Ros3pl
        | RosenbrockVariant::Rodas3 => {
            // 4-stage 3rd-order stiffly accurate Rodas3 coefficients
            // (mod_rosenbrock.f lines 1396-1431)
            rc.s = 4;
            rc.gamma = 0.5;
            rc.order = 3.0;
            rc.a[2][0] = 2.0;
            rc.a[3][0] = 2.0;
            rc.a[3][2] = 1.0;
            rc.c[1][0] = 4.0;
            rc.c[2][0] = 1.0;
            rc.c[2][1] = -1.0;
            rc.c[3][0] = 1.0;
            rc.c[3][1] = -1.0;
            rc.c[3][2] = -8.0 / 3.0;
            rc.alpha = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
            rc.gamma_t = [0.5, 1.5, 0.0, 0.0, 0.0, 0.0];
            rc.m = [2.0, 0.0, 1.0, 1.0, 0.0, 0.0];
            rc.e = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        }
        RosenbrockVariant::Rodas4 | RosenbrockVariant::Rodasp => {
            // 6-stage 4th-order L-stable Rodas4 coefficients
            // (Hairer & Wanner, Solving ODEs II; mod_rosenbrock.f lines 1480+)
            rc.s = 6;
            rc.gamma = 0.25;
            rc.order = 4.0;
            rc.alpha = [0.0, 0.386, 0.21, 0.63, 1.0, 1.0];
            rc.gamma_t = [0.25, -0.1043, 0.1035, -0.0362, 0.0, 0.0];
            rc.a[1][0] = 1.544;
            rc.a[2][0] = 0.946_678_528_081_582_6;
            rc.a[2][1] = 0.255_701_169_898_328_4;
            rc.a[3][0] = 3.314_825_187_068_521;
            rc.a[3][1] = 2.896_124_015_972_201;
            rc.a[3][2] = 0.998_641_913_997_781_7;
            rc.a[4][0] = 1.221_224_509_226_641;
            rc.a[4][1] = 6.019_134_481_288_629;
            rc.a[4][2] = 12.537_083_329_320_87;
            rc.a[4][3] = -0.687_886_036_105_895;
            rc.a[5] = [
                rc.a[4][0],
                rc.a[4][1],
                rc.a[4][2],
                rc.a[4][3],
                1.0,
                0.0,
            ];
            rc.c[1][0] = -5.668_8;
            rc.c[2][0] = -2.430_093_356_833_875;
            rc.c[2][1] = -0.206_359_915_709_191_5;
            rc.c[3][0] = -0.107_352_905_815_137_5;
            rc.c[3][1] = -9.594_562_251_023_355;
            rc.c[3][2] = -20.470_286_148_096_16;
            rc.c[4][0] = 7.496_443_313_967_647;
            rc.c[4][1] = -10.246_804_314_643_52;
            rc.c[4][2] = -33.999_903_528_199_05;
            rc.c[4][3] = 11.708_908_932_061_6;
            rc.c[5] = [
                8.083_246_795_921_522,
                -7.981_132_988_064_893,
                -31.521_594_328_743_71,
                16.319_305_431_231_36,
                -6.058_818_238_834_054,
                0.0,
            ];
            rc.m = [
                rc.a[4][0],
                rc.a[4][1],
                rc.a[4][2],
                rc.a[4][3],
                1.0,
                1.0,
            ];
            rc.e = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        }
    }
    rc
}
