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
    pub(crate) newf: [bool; 6],
    pub(crate) rd31: f64,
    pub(crate) rd32: f64,
}

/// Returns the stage coefficient table [`RosCoeffs`] for `variant`.
///
/// Maps each [`RosenbrockVariant`] onto its stage count `s`, diagonal parameter
/// $\gamma$, classical order $p$, and lower-triangular matrices $A$ and $C$.
#[allow(
    clippy::approx_constant,
    clippy::excessive_precision,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]
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
        newf: [true; 6],
        rd31: 0.0,
        rd32: 0.0,
    };
    match variant {
        RosenbrockVariant::Ros2 => {
            // 2-stage L-stable 2nd-order Rosenbrock (mod_rosenbrock.f:1203-1244)
            let g = 1.0 + 1.0 / 2.0_f64.sqrt();
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
        RosenbrockVariant::Rose2 => {
            // 3-stage L-stable 2nd-order Rose2 (mod_rosenbrock.f:1731-1785)
            let sqrt2 = 1.414_213_562_373_1_f64;
            let g = 1.0 / (2.0 + sqrt2);
            let e32 = 6.0 + sqrt2;
            rc.s = 3;
            rc.gamma = g;
            rc.order = 2.0;
            rc.a[1][0] = 0.5 / g;
            rc.a[2][0] = 1.0 / g;
            rc.a[2][1] = 1.0 / g;
            rc.c[1][0] = -1.0 / g;
            rc.c[2][0] = -(e32 + 2.0) / g;
            rc.c[2][1] = -e32 / g;
            rc.rd31 = 2.0;
            rc.rd32 = e32;
            rc.alpha = [0.0, 0.5, 1.0, 0.0, 0.0, 0.0];
            rc.gamma_t = [g, 0.0, g, 0.0, 0.0, 0.0];
            rc.m = [1.0 / g, 1.0 / g, 0.0, 0.0, 0.0, 0.0];
            rc.e = [
                -1.0 / (6.0 * g),
                -1.0 / (3.0 * g),
                1.0 / (6.0 * g),
                0.0,
                0.0,
                0.0,
            ];
        }
        RosenbrockVariant::Ros3p => {
            // 3-stage 3rd-order Ros3p (Lang & Verwer 2001; mod_rosenbrock.f:1254-1305)
            rc.s = 3;
            rc.gamma = 0.788_675_134_594_812_9;
            rc.order = 3.0;
            rc.a[1][0] = 1.267_949_192_431_122_8;
            rc.a[2][0] = 1.267_949_192_431_122_8;
            rc.c[1][0] = -1.607_695_154_586_736_2;
            rc.c[2][0] = -3.464_101_615_137_754_4;
            rc.c[2][1] = -1.732_050_807_568_877_2;
            rc.newf = [true, true, false, true, true, true];
            rc.alpha = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
            rc.gamma_t = [
                0.788_675_134_594_812_9,
                -0.211_324_865_405_187_13,
                -1.077_350_269_189_625_8,
                0.0,
                0.0,
                0.0,
            ];
            rc.m = [
                2.0,
                0.577_350_269_189_625_8,
                0.422_649_730_810_374_2,
                0.0,
                0.0,
                0.0,
            ];
            rc.e = [
                2.0 - 2.113_248_654_051_871_24,
                0.577_350_269_189_625_8 - 1.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ];
        }
        RosenbrockVariant::Ros3pl => {
            // 4-stage L-stable 3rd-order Ros3pl (mod_rosenbrock.f:1316-1384)
            rc.s = 4;
            rc.gamma = 0.435_866_521_508_459;
            rc.order = 3.0;
            rc.a[1][0] = 1.147_140_180_139_521;
            rc.a[2][0] = 2.463_070_773_030_053;
            rc.a[2][1] = 1.147_140_180_139_521;
            rc.a[3][0] = 2.463_070_773_030_053;
            rc.a[3][1] = 1.147_140_180_139_521;
            rc.c[1][0] = -2.631_861_185_781_065;
            rc.c[2][0] = -1.302_364_158_113_095;
            rc.c[2][1] = 2.769_432_022_251_304;
            rc.c[3][0] = -1.552_568_958_732_4;
            rc.c[3][1] = 2.587_743_501_215_153;
            rc.c[3][2] = -1.416_993_298_352_02;
            rc.newf = [true, true, true, false, true, true];
            rc.alpha = [0.0, 0.5, 1.0, 1.0, 0.0, 0.0];
            rc.gamma_t = [
                0.435_866_521_508_459,
                -0.064_133_478_491_541,
                0.111_028_172_512_505,
                0.0,
                0.0,
                0.0,
            ];
            rc.m = [
                2.463_070_773_030_053,
                1.147_140_180_139_521,
                0.0,
                1.0,
                0.0,
                0.0,
            ];
            rc.e = [
                2.463_070_773_030_053 - 2.346_947_683_513_665,
                1.147_140_180_139_521 - 0.456_530_569_451_895,
                -0.056_949_243_945_495,
                1.0 - 0.738_684_936_166_224,
                0.0,
                0.0,
            ];
        }
        RosenbrockVariant::Rodas3 => {
            // 4-stage 3rd-order stiffly accurate Rodas3 (mod_rosenbrock.f:1396-1461)
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
            rc.newf = [true, false, true, true, true, true];
            rc.alpha = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
            rc.gamma_t = [0.5, 1.5, 0.0, 0.0, 0.0, 0.0];
            rc.m = [2.0, 0.0, 1.0, 1.0, 0.0, 0.0];
            rc.e = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        }
        RosenbrockVariant::Rodas4 => {
            // 6-stage 4th-order L-stable Rodas4 (mod_rosenbrock.f:1473-1595)
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
            rc.a[5] =
                [rc.a[4][0], rc.a[4][1], rc.a[4][2], rc.a[4][3], 1.0, 0.0];
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
            rc.m = [rc.a[4][0], rc.a[4][1], rc.a[4][2], rc.a[4][3], 1.0, 1.0];
            rc.e = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        }
        RosenbrockVariant::Rodasp => {
            // 6-stage 4th-order parabolic Rodasp (Steinebach; mod_rosenbrock.f:1600-1725)
            rc.s = 6;
            rc.gamma = 0.25;
            rc.order = 4.0;
            rc.alpha = [0.0, 0.75, 0.21, 0.63, 1.0, 1.0];
            rc.gamma_t = [0.25, -0.5, -0.023504, -0.0362, 0.0, 0.0];
            rc.a[1][0] = 3.0;
            rc.a[2][0] = 1.831_036_793_486_759;
            rc.a[2][1] = 0.495_518_396_743_379_5;
            rc.a[3][0] = 2.304_376_582_692_669;
            rc.a[3][1] = -0.052_492_752_457_430_01;
            rc.a[3][2] = -1.176_798_761_832_782;
            rc.a[4][0] = -7.170_454_962_423_024;
            rc.a[4][1] = -4.741_636_671_481_785;
            rc.a[4][2] = -16.310_026_313_309_71;
            rc.a[4][3] = -1.062_004_044_111_401;
            rc.a[5] =
                [rc.a[4][0], rc.a[4][1], rc.a[4][2], rc.a[4][3], 1.0, 0.0];
            rc.c[1][0] = -12.0;
            rc.c[2][0] = -8.791_795_173_947_035;
            rc.c[2][1] = -2.207_865_586_973_518;
            rc.c[3][0] = 10.817_930_568_571_53;
            rc.c[3][1] = 6.780_270_611_428_266;
            rc.c[3][2] = 19.534_859_446_424_1;
            rc.c[4][0] = 34.190_950_067_496_76;
            rc.c[4][1] = 15.496_711_537_259_63;
            rc.c[4][2] = 54.747_608_759_641_3;
            rc.c[4][3] = 14.160_053_921_485_34;
            rc.c[5] = [
                34.626_058_309_305_32,
                15.300_849_761_144_73,
                56.999_555_786_626_67,
                18.408_070_097_930_95,
                -5.714_285_714_285_717,
                0.0,
            ];
            rc.m = [rc.a[4][0], rc.a[4][1], rc.a[4][2], rc.a[4][3], 1.0, 1.0];
            rc.e = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        }
    }
    rc
}
